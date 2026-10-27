"""GiGPO estimators: episode baseline + same-state discounted-return baseline.

Reference: Feng et al., NeurIPS 2025, equations 5, 7, 8.
Conservative anchor: exact policy token prefix, within one task and rollout batch.
No intermediate reward model, ground-truth path, or approximate state clustering.
"""
from collections import defaultdict
import math


def normalized(values, epsilon=1e-4):
    if len(values) < 2:
        return [0.0] * len(values)
    mean = sum(values) / len(values)
    # Match TRL's group sample-standard-deviation normalization.
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return [(v - mean) / (std + epsilon) for v in values]


def assign_advantages(samples, group_size, gamma=0.95, omega=1.0):
    """Return completion-token advantages, per-action diagnostics and coverage.

    The unchanged episode reward is placed at the final MODEL decision (which
    need not be final_answer in a failed episode). Environment tokens and the
    runtime EOS receive exactly zero advantage. Singleton anchors fall back to
    episode-only advantages.
    """
    if not 0 < gamma <= 1 or omega < 0 or group_size < 2:
        raise ValueError('Invalid GiGPO hyperparameters')
    if not samples or len(samples) % group_size:
        raise ValueError('Expected complete contiguous episode groups')
    episode_adv = []
    anchors = defaultdict(list)
    action_rows = []
    for start in range(0, len(samples), group_size):
        group = samples[start:start + group_size]
        if len({s['row_id'] for s in group}) != 1:
            raise ValueError('A GRPO group must contain the same training task')
        episode_adv.extend(normalized([s['episode_score'] for s in group]))
    for i, sample in enumerate(samples):
        if not math.isfinite(sample['episode_score']):
            raise ValueError('Nonfinite reward')
        steps = sample['action_steps']
        if not steps:
            raise ValueError('Empty model trajectory')
        for t, step in enumerate(steps):
            value = gamma ** (len(steps) - 1 - t) * sample['episode_score']
            key = (i // group_size, sample['row_id'], step['anchor'])
            row = dict(sample=i, turn=t, anchor=step['anchor'],
                       start=step['start'], end=step['end'], return_to_go=value,
                       episode_advantage=episode_adv[i], step_advantage=0.0)
            action_rows.append(row)
            anchors[key].append(row)
    for group in anchors.values():
        for row, advantage in zip(group, normalized([r['return_to_go'] for r in group])):
            row['step_advantage'] = advantage
            row['anchor_group_size'] = len(group)
    vectors = [[0.0] * len(s['completion_ids']) for s in samples]
    covered = [[False] * len(v) for v in vectors]
    for row in action_rows:
        i, start, end = row['sample'], row['start'], row['end']
        if not 0 <= start < end <= len(vectors[i]):
            raise ValueError('Invalid model action token span')
        for token in range(start, end):
            if not samples[i]['env_mask'][token] or covered[i][token]:
                raise ValueError('Action spans overlap or include environment tokens')
            covered[i][token] = True
            vectors[i][token] = row['episode_advantage'] + omega * row['step_advantage']
    if any([int(v) for v in c] != s['env_mask'] for c, s in zip(covered, samples)):
        raise ValueError('Action spans do not cover precisely the model tokens')
    noninitial = [r for r in action_rows if r['turn'] > 0]
    stats = {
        'actions': len(action_rows), 'anchor_groups': len(anchors),
        'shared_actions': sum(r['anchor_group_size'] > 1 for r in action_rows),
        'noninitial_actions': len(noninitial),
        'shared_noninitial_actions': sum(r['anchor_group_size'] > 1 for r in noninitial),
        'nonzero_step_advantages': sum(abs(r['step_advantage']) > 1e-8 for r in action_rows),
        'nonzero_noninitial_step_advantages': sum(abs(r['step_advantage']) > 1e-8 for r in noninitial),
        'gamma': gamma, 'omega': omega, 'anchor_mode': 'exact_policy_token_prefix',
    }
    return vectors, action_rows, stats


def policy_token_loss(logps, old_logps, ref_logps, advantages, beta,
                      epsilon_low, epsilon_high, importance_ratio=None):
    """Tokenwise clipped surrogate, sampled KL, optional vLLM correction."""
    import torch
    ratio = (logps - old_logps).exp()
    surrogate = torch.minimum(ratio * advantages,
                              ratio.clamp(1 - epsilon_low, 1 + epsilon_high) * advantages)
    loss = -surrogate
    if importance_ratio is not None:
        loss = loss * importance_ratio
    if beta:
        delta = ref_logps - logps
        loss = loss + beta * (delta.exp() - delta - 1)
    return loss
