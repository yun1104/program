"""Parallel real-environment rollouts against TRL's synchronized vLLM server."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import time

from .common import digest, fingerprint, project_path, read_json, write_json
from .prepare import load_task
from .protocol import token_segments
from .rollout import episode_reward, flatten_episode, run_episode


class RemotePolicy:
    def __init__(self, client, tokenizer, config, model, identity):
        self.client, self.tokenizer, self.config = client, tokenizer, config
        self.model, self.identity, self.turn = model, identity, 0

    def generate(self, messages, segments):
        ids, _ = token_segments(self.tokenizer, messages, segments)
        max_new = min(self.config['max_tokens'], self.config['max_context_tokens'] - len(ids) - 1)
        if max_new <= 0:
            raise ValueError('Context budget exhausted; history is never silently truncated')
        seed = int(fingerprint([self.identity, self.turn])[:8], 16)
        self.turn += 1
        payload = {
            'model': self.model, 'prompt': ids, 'return_token_ids': True,
            'return_tokens_as_token_ids': True, 'temperature': self.config['temperature'],
            'top_p': 1.0, 'top_k': -1, 'max_tokens': max_new, 'seed': seed,
            'logprobs': 0, 'stop': ['</action>'], 'include_stop_str_in_output': True}
        import httpx
        attempts = self.config.get('rpc_max_retries', 4) + 1
        for attempt in range(attempts):
            try:
                response = self.client.post('/v1/completions', json=payload)
                response.raise_for_status()
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                transient = isinstance(error, httpx.TransportError) or error.response.status_code in (429,502,503,504)
                if not transient or attempt + 1 == attempts:
                    raise
                print(f'Sampler RPC retry {attempt+1}/{attempts-1}: {type(error).__name__}; same prompt and seed', flush=True)
                time.sleep(min(.25 * 2**attempt, 2))
        choice = response.json()['choices'][0]
        generated = choice['token_ids']
        logprobs = choice['logprobs']['token_logprobs']
        if len(logprobs) != len(generated) or not all(v is not None and math.isfinite(v) for v in logprobs):
            raise RuntimeError('Sampler returned missing or nonfinite token logprobs')
        return {'text': choice['text'], 'token_ids': generated, 'sampling_logprobs': logprobs,
                'finish_reason': choice['finish_reason'],
                'usage': {'input_tokens': len(ids), 'output_tokens': len(generated)}}


def episode_sample(episode, tokenizer, cost):
    prompt, completion, mask = flatten_episode(episode, tokenizer)
    logprobs = []
    for segment in episode['segments']:
        ids = segment.get('token_ids')
        if ids is None:
            ids = tokenizer.encode(segment['text'], add_special_tokens=False)
        values = segment['sampling_logprobs'] if segment['model'] else [0.0] * len(ids)
        if len(values) != len(ids):
            raise ValueError('Sampler logprobs are not aligned to actual action tokens')
        logprobs.extend(values)
    if len(logprobs) + 1 == len(completion) and mask[-1] == 0:
        logprobs.append(0.0)  # Runtime terminal marker, excluded from the objective.
    if len(logprobs) != len(completion) or len(mask) != len(completion):
        raise ValueError('Rollout token/logprob/environment-mask alignment failed')
    return {'prompt_ids': prompt, 'completion_ids': completion, 'env_mask': mask,
            'logprobs': logprobs, 'episode_score': episode_reward(episode, cost)}


class ServerRollout:
    def __init__(self, rows, project, config, output):
        self.rows = {r['id']: r for r in rows}
        self.project, self.config, self.output = project, config, output
        self.counter = 0
        self.replay_root = project_path(config['replay_rollouts_from']) if config.get('replay_rollouts_from') else None
        self.replay_steps = config.get('replay_steps', 0)
        if self.replay_root:
            prior = read_json(self.replay_root / 'manifest.json')
            old = prior['config']
            for key in ('seed','model','generation','lora'):
                if old[key] != config[key]:
                    raise ValueError(f'Replay recovery changed {key}')
            ignored = {'save_steps','save_total_limit','save_only_model'}
            if {k:v for k,v in old['training'].items() if k not in ignored} != {k:v for k,v in config['training'].items() if k not in ignored}:
                raise ValueError('Replay recovery changed training mathematics or batch ordering')
            if prior['project_config'] != project or prior['train_split_sha256'] != digest(project_path(project['splits'])/'train.jsonl'):
                raise ValueError('Replay source environment or train split changed')

    def has_replay(self, step):
        return self.replay_root is not None and step < self.replay_steps

    def replay(self, prompts, trainer):
        results = []
        for index, key in enumerate(prompts):
            source = self.replay_root / 'rollouts' / f'{self.counter+index:08d}.json'
            episode = read_json(source)
            if episode['row'] != self.rows[key] or episode['step'] != trainer.state.global_step:
                raise ValueError('Saved student rollout does not match current task/group/step')
            sample = episode_sample(episode, trainer.processing_class, self.project.get('call_penalty',.002))
            results.append(sample)
            write_json(self.output/'rollouts'/source.name, {**episode,'replayed_from':str(source)})
        write_json(self.output/'rollout_progress.json', {'step':trainer.state.global_step,
            'completed':len(results),'total':len(results),'source':'saved_student_rollout_replay',
            'note':'Rebuilding updates from logged student behavior, not restoring lost optimizer state.'})
        return results

    def __call__(self, prompts, trainer):
        import httpx
        from accelerate.utils import gather_object, broadcast_object_list
        all_prompts = gather_object(prompts)
        self.counter = trainer.state.global_step * len(all_prompts)
        counts = gather_object([len(prompts)])
        payload = [None]
        started = time.monotonic()
        if trainer.accelerator.is_main_process:
            try:
                if self.has_replay(trainer.state.global_step):
                    payload[0] = {'results': self.replay(all_prompts, trainer)}
                else:
                    payload[0] = {'results': self.sample_online(all_prompts, trainer, started)}
            except Exception as error:
                payload[0] = {'error': f'{type(error).__name__}: {error}'}
        broadcast_object_list(payload, from_process=0)
        if 'error' in payload[0]:
            raise RuntimeError('Online rollout failed: ' + payload[0]['error'])
        rank = trainer.accelerator.process_index
        start = sum(counts[:rank])
        local = payload[0]['results'][start:start+len(prompts)]
        return {key: [r[key] for r in local] for key in local[0]}

    def sample_online(self, all_prompts, trainer, started):
        import httpx
        from .environment import upstream_modules
        from .tool_schema_cache import enable_schema_cache
        upstream_modules(self.project['upstream'])
        enable_schema_cache(self.project['upstream'])
        generation = {**self.config['generation'], 'rpc_max_retries': self.config.get('rpc_max_retries',4)}
        model_name = trainer.vllm_generation.vllm_client.model
        base_url = self.config['training']['vllm_server_base_url']
        results = [None] * len(all_prompts)
        scores = []
        with httpx.Client(base_url=base_url, trust_env=False, timeout=300,
                          limits=httpx.Limits(max_connections=128, max_keepalive_connections=0)) as client:
            def evaluate(index, key):
                if key not in self.rows:
                    raise ValueError('Rollout requested a non-training task')
                row = self.rows[key]
                policy = RemotePolicy(client, trainer.processing_class, generation, model_name,
                    [self.config['seed'], trainer.state.global_step, self.counter, index, key])
                episode = run_episode(load_task(row), policy, self.project)
                sample = episode_sample(episode, trainer.processing_class,
                                        self.project.get('call_penalty', 0.002))
                if len(sample['prompt_ids']) + len(sample['completion_ids']) > generation['max_context_tokens']:
                    raise ValueError('Rollout exceeds context budget')
                return index, row, episode, sample
            with ThreadPoolExecutor(max_workers=min(len(all_prompts), self.config.get('rollout_workers', 128))) as pool:
                futures = [pool.submit(evaluate, i, key) for i, key in enumerate(all_prompts)]
                for future in as_completed(futures):
                    index, row, episode, sample = future.result()
                    results[index] = sample
                    scores.append(sample['episode_score'])
                    write_json(self.output / 'rollouts' / f'{self.counter + index:08d}.json',
                               {'row': row, 'step': trainer.state.global_step, **episode})
                    if len(scores) % 8 == 0 or len(scores) == len(all_prompts):
                        write_json(self.output / 'rollout_progress.json', {
                            'step': trainer.state.global_step, 'completed': len(scores), 'total': len(all_prompts),
                            'elapsed_seconds': time.monotonic()-started,
                            'mean_reward': sum(scores)/len(scores)})
        self.counter += len(results)
        return results
