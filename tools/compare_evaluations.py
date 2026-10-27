#!/usr/bin/env python3
"""Compare complete paired ToolMaze evaluations without resampling or selecting episodes."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import read_json, write_json


def load(run):
    run = ROOT / run
    manifest = read_json(run / 'manifest.json')
    report = read_json(run / 'summary.json')
    records = [read_json(p) for p in sorted((run / 'episodes').glob('*.json'))]
    paired = {(r['row']['id'], r['trial']): r for r in records}
    expected = {(identity, trial) for identity in manifest['requested_ids']
                for trial in range(manifest['config']['trials'])}
    if set(paired) != expected or len(records) != len(expected):
        raise ValueError(f'Incomplete or duplicate evaluation: {run}')
    return manifest, report, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='runs/eval/qwen35_2b_base_vllm_test_20261026')
    parser.add_argument('--candidate', default='runs/eval/qwen35_2b_sft_vllm_test_20261026')
    parser.add_argument('--output', default='reports/sft/qwen35_2b_sft_vs_base_test')
    args = parser.parse_args()
    bm, bs, base = load(args.baseline)
    cm, cs, candidate = load(args.candidate)
    assert base.keys() == candidate.keys()
    assert bm['split_sha256'] == cm['split_sha256']
    assert bm['project_config'] == cm['project_config']
    for key in ['generation', 'seed', 'trials', 'split']:
        assert bm['config'][key] == cm['config'][key], key
    for key in base:
        assert base[key]['row'] == candidate[key]['row']
    paired = {}
    for metric in ['official_pass', 'strict_pass']:
        paired[metric] = dict(Counter(
            'both_pass' if base[k].get(metric) and candidate[k].get(metric)
            else 'improved' if candidate[k].get(metric)
            else 'regressed' if base[k].get(metric) else 'both_fail' for k in base))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    details = []
    for key in sorted(base):
        b, c = base[key], candidate[key]
        details.append({'id': key[0], 'trial': key[1], 'mode': c['row']['mode'],
            'base_official': b.get('official_pass', False), 'sft_official': c.get('official_pass', False),
            'base_strict': b.get('strict_pass', False), 'sft_strict': c.get('strict_pass', False),
            'sft_termination': c.get('termination'), 'sft_failure_reason': c.get('judgement', {}).get('failure_reason')})
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    comparison = {'baseline': args.baseline, 'candidate': args.candidate, 'n': len(base),
                  'paired': paired, 'baseline_all': bs['breakdown']['all'], 'sft_all': cs['breakdown']['all'],
                  'sft_termination_counts': dict(Counter(r.get('termination', 'error') for r in candidate.values()))}
    write_json(output.with_suffix('.json'), comparison)
    lines = ['# Qwen3.5-2B：SFT 与原模型测试集对比', '',
             f'固定测试集 {len(base)} 题，每题 {cm["config"]["trials"]} 次采样；全量计分，无结果筛选。', '',
             '| 指标 | 原模型 | SFT | 变化 |', '|---|---:|---:|---:|']
    for label, field in [('官方通过率', 'official_pass_rate'), ('额外严格通过率', 'strict_pass_rate')]:
        b, c = bs['breakdown']['all'][field], cs['breakdown']['all'][field]
        lines.append(f'| {label} | {b:.1%}（{round(b*len(base))}/{len(base)}） | {c:.1%}（{round(c*len(base))}/{len(base)}） | {(c-b)*100:+.1f} 个百分点 |')
    lines += ['', '| 模式 | 样本 | 原模型官方通过 | SFT 官方通过 | 原模型严格通过 | SFT 严格通过 |',
              '|---|---:|---:|---:|---:|---:|']
    for mode in range(5):
        key = f'mode/P{mode}'
        b, c = bs['breakdown'][key], cs['breakdown'][key]
        lines.append(f'| P{mode} | {c["n"]} | {b["official_pass_rate"]:.1%} | {c["official_pass_rate"]:.1%} | {b["strict_pass_rate"]:.1%} | {c["strict_pass_rate"]:.1%} |')
    lines += ['', f'逐题变化：`{paired}`。', '',
        f'基础设施错误：原模型 {bs["breakdown"]["all"]["infrastructure_errors"]}，SFT {cs["breakdown"]["all"]["infrastructure_errors"]}；均计入分母。',
        f'生成配置：`{cm["config"]["generation"]}`，thinking 关闭，最多 {cm["project_config"]["max_rounds"]} 轮，无 JSON 强约束解码。',
        '官方通过包括符合任务要求的合理终止；严格通过还要求明确最终回答且全程无协议/schema 错误，均不等于完整业务语义证明。',
        '本结果为一次固定 seed 的采样评测；没有基于测试集选择 checkpoint 或重试。', '',
        '可比性审计：`reports/sft/baseline_runtime_replay.json`、`reports/sft/tokenizer_compatibility.json`。',
        f'原始 SFT 轨迹与官方指标：`{args.candidate}`。']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(comparison, flush=True)


if __name__ == '__main__':
    main()
