"""Batched, multi-turn vLLM colocated rollout, isolated from existing GRPO."""
import time

from .common import fingerprint, write_json
from .environment import ToolMazeEnvironment, upstream_modules
from .prepare import load_task
from .protocol import initial_messages, observation_text, parse_action, token_segments
from .rl_vllm_rollout import episode_sample


def episode_coroutine(task, tokenizer, project, generation, row_id):
    env = ToolMazeEnvironment(task, project)
    messages = initial_messages(env.query, env.definitions)
    segments, decisions, steps, errors = [], [], [], []
    invalid = 0
    usage = {'input_tokens': 0, 'output_tokens': 0}
    reason = 'max_rounds'
    completion_offset = 0
    for turn in range(project.get('max_rounds', 16)):
        ids, _ = token_segments(tokenizer, messages, segments)
        budget = min(generation['max_tokens'], generation['max_context_tokens'] - len(ids) - 1)
        if budget <= 0:
            reason = 'generation_error'
            errors.append('Context budget exhausted; no silent truncation')
            break
        # Exact history includes model actions and actual observations. Task key
        # separates perturbation variants without exposing that key to the actor.
        anchor = fingerprint([row_id, ids])
        response = yield {'ids': ids, 'max_tokens': budget, 'turn': turn}
        text, tokens, logprobs = response['text'], response['token_ids'], response['sampling_logprobs']
        usage['input_tokens'] += len(ids)
        usage['output_tokens'] += len(tokens)
        if not tokens or len(tokens) != len(logprobs):
            raise ValueError('Invalid vLLM action token/logprob alignment')
        steps.append({'anchor': anchor, 'start': completion_offset,
                      'end': completion_offset + len(tokens)})
        segments.append({'text': text, 'model': True, 'token_ids': tokens,
                         'sampling_logprobs': logprobs})
        completion_offset += len(tokens)
        try:
            action = parse_action(text, env.allowed)
        except (ValueError, TypeError) as error:
            invalid += 1
            decisions.append({'valid': False})
            name, result = 'protocol', {'error': str(error)}
        else:
            decisions.append({'valid': True, 'type': action['type']})
            result = env.step(action)
            if action['type'] == 'final_answer':
                reason = 'final_answer'
                break
            name = action['tool_name']
        observation = {'text': observation_text(name, result), 'model': False}
        combined, _ = token_segments(tokenizer, messages, segments + [observation])
        if len(combined) + 1 > generation['max_context_tokens']:
            reason = 'context_budget'
            errors.append('Observation exceeds context budget; no silent truncation')
            break
        segments.append(observation)
        completion_offset += len(tokenizer.encode(observation['text'], add_special_tokens=False))
    usage['total_tokens'] = sum(usage.values())
    trace, judgement = env.score(usage)
    strict = bool(judgement['pass'] and env.finished and invalid == 0 and env.validation_errors == 0)
    return {'initial_messages': messages, 'segments': segments, 'trace': trace,
            'judgement': judgement, 'official_pass': bool(judgement['pass']), 'strict_pass': strict,
            'termination': reason, 'invalid_actions': invalid, 'validation_errors': env.validation_errors,
            'tool_calls': env.calls, 'tokens': usage, 'errors': errors, 'action_steps': steps}


class GiGPORollout:
    def __init__(self, rows, project, config, output):
        self.rows = {r['id']: r for r in rows}
        self.project, self.config, self.output = project, config, output
        self.last_samples = None

    def __call__(self, prompts, trainer):
        from vllm import SamplingParams
        import torch
        if trainer.accelerator.num_processes != 1:
            raise ValueError('Initial GiGPO runner supports one training GPU only')
        upstream_modules(self.project['upstream'])
        from .tool_schema_cache import enable_schema_cache
        enable_schema_cache(self.project['upstream'])
        started = time.monotonic()
        engine = trainer.vllm_generation.llm
        engine.wake_up(tags=['kv_cache'])  # weights already synchronized by TRL
        episodes, coroutines, pending = {}, {}, {}
        for i, key in enumerate(prompts):
            if key not in self.rows:
                raise ValueError('GiGPO requested a non-training row')
            coroutine = episode_coroutine(load_task(self.rows[key]), trainer.processing_class,
                                          self.project, self.config['generation'], key)
            coroutines[i] = coroutine
            pending[i] = next(coroutine)
        try:
            while pending:
                indices = list(pending)
                requests, params = [], []
                for i in indices:
                    request = pending[i]
                    requests.append({'prompt_token_ids': request['ids']})
                    seed = int(fingerprint([self.config['seed'], trainer.state.global_step,
                                            i, prompts[i], request['turn']])[:8], 16)
                    params.append(SamplingParams(temperature=self.config['generation']['temperature'],
                        top_p=1.0, top_k=-1, max_tokens=request['max_tokens'], seed=seed,
                        logprobs=0, stop=['</action>'], include_stop_str_in_output=True))
                outputs = engine.generate(requests, sampling_params=params, use_tqdm=False)
                for i, output in zip(indices, outputs):
                    choice = output.outputs[0]
                    response = {'text': choice.text, 'token_ids': list(choice.token_ids),
                        'sampling_logprobs': [float(v[token].logprob)
                            for token, v in zip(choice.token_ids, choice.logprobs)]}
                    try:
                        pending[i] = coroutines[i].send(response)
                    except StopIteration as finished:
                        episodes[i] = finished.value
                        del pending[i]
                write_json(self.output/'rollout_progress.json', {'step': trainer.state.global_step,
                    'completed': len(episodes), 'total': len(prompts),
                    'active': len(pending), 'elapsed_seconds': time.monotonic()-started})
        finally:
            engine.sleep(level=2)
            trainer.vllm_generation._llm_weights_sleeping = True
            torch.cuda.empty_cache()
        samples = []
        for i, key in enumerate(prompts):
            episode = episodes[i]
            sample = episode_sample(episode, trainer.processing_class, self.project.get('call_penalty', .002))
            sample.update(row_id=key, action_steps=episode['action_steps'])
            samples.append(sample)
            write_json(self.output/'rollouts'/f'{trainer.state.global_step:04d}_{i:03d}.json',
                       {'row': self.rows[key], 'step': trainer.state.global_step, **episode})
        self.last_samples = samples
        return {key: [sample[key] for sample in samples] for key in
                ('prompt_ids', 'completion_ids', 'env_mask', 'logprobs', 'episode_score')}
