#!/usr/bin/env python3
"""Replay frozen model actions to check student runtime compatibility, without resampling."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import read_json, write_json
from scripts.toolmaze_agent.prepare import load_task
from scripts.toolmaze_agent.rollout import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = ROOT / args.run
    manifest = read_json(run / 'manifest.json')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ROOT / manifest['config']['model'])
    from scripts.toolmaze_agent.environment import upstream_modules
    upstream_modules(manifest['project_config']['upstream'])
    from scripts.toolmaze_agent.tool_schema_cache import enable_schema_cache
    enable_schema_cache(ROOT / manifest['project_config']['upstream'])
    checked, failures = 0, []

    class Replay:
        def __init__(self, record):
            self.tokenizer = tokenizer
            self.config = manifest['config']['generation']
            self.record = record
            self.actions = iter((i, s) for i, s in enumerate(record['segments']) if s['model'])

        def generate(self, messages, segments):
            if messages != self.record['initial_messages']:
                raise RuntimeError('Initial prompt changed')
            try:
                index, segment = next(self.actions)
            except StopIteration:
                raise ValueError('Recorded generation ended')
            if [s['text'] for s in segments] != [s['text'] for s in self.record['segments'][:index]]:
                raise RuntimeError('Environment observation or student transcript changed')
            return {'text': segment['text'], 'token_ids': segment['token_ids'], 'usage': {}}

    fields = ['official_pass', 'strict_pass', 'termination', 'invalid_actions', 'validation_errors', 'tool_calls']
    for path in sorted((run / 'episodes').glob('*.json')):
        record = read_json(path)
        try:
            if record.get('error'):
                raise RuntimeError('Cannot replay an infrastructure failure')
            actual = run_episode(load_task(record['row']), Replay(record), manifest['project_config'])
            differences = {key: {'saved': record[key], 'replayed': actual[key]}
                           for key in fields if record[key] != actual[key]}
            if [s['text'] for s in actual['segments']] != [s['text'] for s in record['segments']]:
                differences['transcript'] = 'changed'
            if differences:
                failures.append({'id': record['row']['id'], 'differences': differences})
        except Exception as error:
            failures.append({'id': record['row']['id'], 'error': f'{type(error).__name__}: {error}'})
        checked += 1
    report = {'run': args.run, 'checked': checked, 'failures': failures,
              'method': 'Replay every saved action; require identical prompt, observations, action text and outcome. No LLM resampling.'}
    write_json(ROOT / args.output, report)
    print({'checked': checked, 'failures': len(failures)}, flush=True)
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
