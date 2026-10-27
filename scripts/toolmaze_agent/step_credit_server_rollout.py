"""StepCredit metadata on the existing real-environment synchronized vLLM rollout.

Only this new run's journals are annotated. No additional environment execution
or model sampling is performed to reconstruct action boundaries and anchors.
"""
from .common import fingerprint, read_json, write_json
from .protocol import token_segments
from .rl_vllm_rollout import ServerRollout


def reconstruct_steps(episode, tokenizer, row_id):
    prefix, _ = token_segments(tokenizer, episode['initial_messages'], [])
    offset, steps = 0, []
    for segment in episode['segments']:
        ids = segment.get('token_ids')
        if ids is None:
            ids = tokenizer.encode(segment['text'], add_special_tokens=False)
        if segment['model']:
            if not ids:
                raise ValueError('Empty model decision')
            steps.append({'anchor': fingerprint([row_id, prefix]),
                          'start': offset, 'end': offset + len(ids)})
        prefix.extend(ids)
        offset += len(ids)
    return steps


class StepCreditServerRollout(ServerRollout):
    def __call__(self, prompts, trainer):
        from accelerate.utils import gather_object
        result = super().__call__(prompts, trainer)
        counts = gather_object([len(prompts)])
        rank = trainer.accelerator.process_index
        start = trainer.state.global_step * sum(counts) + sum(counts[:rank])
        local = []
        for i, key in enumerate(prompts):
            path = self.output/'rollouts'/f'{start+i:08d}.json'
            episode = read_json(path)
            if episode['row']['id'] != key or episode['step'] != trainer.state.global_step:
                raise ValueError('Distributed StepCredit journal ordering mismatch')
            steps = reconstruct_steps(episode, trainer.processing_class, key)
            sample = {name: values[i] for name, values in result.items()}
            sample.update(row_id=key, action_steps=steps)
            local.append(sample)
            write_json(path, {**episode, 'action_steps': steps})
        self.last_samples = local
        return result
