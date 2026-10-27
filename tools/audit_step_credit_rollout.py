#!/usr/bin/env python3
"""Replay identical saved actions through old/new loops; no model or API calls."""
import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import load_config, read_json, write_json
from scripts.toolmaze_agent.step_credit_rollout import episode_coroutine
from scripts.toolmaze_agent.prepare import load_task
from scripts.toolmaze_agent.protocol import token_segments
from scripts.toolmaze_agent.rl_vllm_rollout import episode_sample
from scripts.toolmaze_agent.rollout import run_episode


def audit(path, tokenizer, project):
    source = read_json(path)
    actions = [segment for segment in source['segments'] if segment['model']]
    generation = {'max_tokens': 512, 'max_context_tokens': 8192, 'temperature': .8}
    class Replay:
        def __init__(self):
            self.tokenizer, self.config = tokenizer, generation
            self.index = 0

        def generate(self, messages, segments):
            segment = actions[self.index]
            self.index += 1
            ids, _ = token_segments(tokenizer, messages, segments)
            return {'text': segment['text'], 'token_ids': segment['token_ids'],
                    'sampling_logprobs': segment['sampling_logprobs'],
                    'usage': {'input_tokens': len(ids), 'output_tokens': len(segment['token_ids'])}}
    old = run_episode(load_task(source['row']), Replay(), project)
    coroutine = episode_coroutine(load_task(source['row']), tokenizer, project, generation, source['row']['id'])
    next(coroutine)
    new = None
    for segment in actions:
        try:
            coroutine.send({'text': segment['text'], 'token_ids': segment['token_ids'],
                            'sampling_logprobs': segment['sampling_logprobs']})
        except StopIteration as finished:
            new = finished.value
            break
    if new is None:
        raise AssertionError('New rollout did not terminate at the same decision')
    for key in ('segments', 'official_pass', 'strict_pass', 'termination', 'invalid_actions',
                'validation_errors', 'tool_calls', 'tokens', 'errors', 'judgement'):
        if old[key] != new[key]:
            raise AssertionError(f'Old/new rollout mismatch: {path.name}: {key}')
    a = episode_sample(old, tokenizer, project.get('call_penalty', .002))
    b = episode_sample(new, tokenizer, project.get('call_penalty', .002))
    assert a == b, 'Token/logprob/mask or terminal reward mismatch'
    from scripts.toolmaze_agent.step_credit import assign_advantages
    b.update(row_id=source['row']['id'], action_steps=new['action_steps'])
    assign_advantages([b, copy.deepcopy(b)], 2)
    return {'source': str(path), 'row': source['row']['id'], 'match': True,
            'actions': len(new['action_steps']), 'strict_pass': new['strict_pass']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='runs/grpo_lora_recovery_20261026/rollouts')
    parser.add_argument('--limit', type=int, default=4)
    parser.add_argument('--output', default='reports/step_credit/rollout_replay_audit.json')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT/'runs/sft_full_trl_20261026/final')
    project = load_config('scripts/toolmaze_agent/configs/project.yaml')
    paths = sorted((ROOT/args.source).glob('*.json'))[:args.limit]
    if not paths:
        raise ValueError('No saved trajectories to audit')
    result = {'passed': True, 'cases': [audit(path, tokenizer, project) for path in paths]}
    write_json(ROOT/args.output, result)
    print(result)


if __name__ == '__main__':
    main()
