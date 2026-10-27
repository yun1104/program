#!/usr/bin/env python3
"""Write a reviewable smoke-test receipt, including actual credit diversity."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/gigpo_smoke_20261026')
    args = parser.parse_args()
    run = ROOT/args.run
    report = ROOT/'reports/gigpo'/run.name
    progress = read_json(run/'progress.json')
    launch = read_json(report/'launch.json')
    batches = []
    for path in sorted((run/'advantages').glob('*.json')):
        data = read_json(path)
        groups, per_sample = defaultdict(list), defaultdict(list)
        episodes = {}
        for row in data['actions']:
            i, t = row['sample'], row['turn']
            if i not in episodes:
                episodes[i] = read_json(run/'rollouts'/f"{data['step']:04d}_{i:03d}.json")
            segments = [s for s in episodes[i]['segments'] if s['model']]
            groups[row['anchor']].append((t, tuple(segments[t]['token_ids']), row['step_advantage']))
            per_sample[i].append(row['step_advantage'])
        diverse = [g for g in groups.values() if len({a[1] for a in g}) > 1]
        batches.append({'step': data['step'], **data['statistics'],
            'diverse_action_anchor_groups': len(diverse),
            'noninitial_diverse_action_anchor_groups': sum(all(a[0] > 0 for a in g) for g in diverse),
            'trajectories_with_varying_step_advantage': sum(max(v)-min(v)>1e-8 for v in per_sample.values()),
            'episodes': [{'row': e['row']['id'], 'strict_pass': e['strict_pass'],
                          'tool_calls': e['tool_calls']} for e in episodes.values()]})
    unchanged = all(digest(ROOT/p) == h for p, h in launch['original_source_sha256'].items())
    result = {'state': progress['state'], 'completed_updates': progress['step'],
        'gpu': launch['gpu'], 'original_sources_unchanged': unchanged,
        'adapter_progress': read_json(run/'adapter_progress.json'), 'batches': batches,
        'unit_tests_log': 'reports/gigpo/unit_tests.log',
        'old_new_rollout_replay_audit': 'reports/gigpo/rollout_replay_audit.json',
        'smoke_rollout_replay_audit': 'reports/gigpo/smoke_rollout_replay_audit.json',
        'limitation': 'Integration smoke test only; no claim of benchmark improvement.'}
    if progress['state'] == 'complete':
        from safetensors.torch import load_file
        import torch
        weights = load_file(run/'final/adapter_model.safetensors')
        result['final_adapter'] = {'tensors': len(weights),
            'all_finite': all(torch.isfinite(v).all().item() for v in weights.values()),
            'lora_only': all('lora_' in name for name in weights),
            'lora_b_l2': sum(v.float().square().sum().item() for name, v in weights.items()
                              if 'lora_B' in name) ** .5}
        result['checkpoint_markers'] = [read_json(p) for p in sorted(run.glob('checkpoint-*/checkpoint_complete.json'))]
        result['training_metrics'] = read_json(run/'training_metrics.json')
        if not unchanged or not result['final_adapter']['all_finite'] or not result['final_adapter']['lora_only']:
            raise AssertionError('Smoke validation failed')
    write_json(report/'smoke_summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
