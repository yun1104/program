#!/usr/bin/env python3
"""Paired final-checkpoint comparison, preserving the full fixed denominator."""
from collections import Counter
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.compare_evaluations import load
from scripts.toolmaze_agent.common import read_json, write_json


def main():
    runs = {'Base':'runs/eval/qwen35_2b_base_vllm_test_20261026',
            'SFT':'runs/eval/qwen35_2b_sft_vllm_test_20261026',
            'GRPO':'runs/eval/qwen35_2b_grpo_vllm_test_20261027',
            'GiGPO':'runs/eval/qwen35_2b_gigpo_vllm_test_20261027'}
    loaded = {name: load(path) for name,path in runs.items()}
    reference = loaded['SFT']
    for name,(manifest,summary,records) in loaded.items():
        if records.keys() != reference[2].keys() or len(records) != 200:
            raise ValueError(f'Incomplete or unpaired results: {name}')
        for key in ('generation','seed','trials','split'):
            if manifest['config'][key] != reference[0]['config'][key]:
                raise ValueError(f'Evaluation settings differ: {name}/{key}')
        for key in ('split_sha256','project_config'):
            if manifest[key] != reference[0][key]:
                raise ValueError(f'Dataset/environment differs: {name}/{key}')
        if name in ('GRPO','GiGPO'):
            for filename in ('protocol.py','environment.py','rollout.py','vllm_backend.py'):
                key = 'scripts/toolmaze_agent/'+filename
                if manifest['source_sha256'][key] != loaded['GRPO'][0]['source_sha256'][key]:
                    raise ValueError(f'RL evaluation runtimes differ: {name}/{filename}')
                if manifest['source_sha256'][key] != reference[0]['source_sha256'][key]:
                    if filename != 'rollout.py':
                        raise ValueError(f'Evaluation runtime differs from SFT: {name}/{filename}')
                    audit = read_json(ROOT/'reports/rl_eval/grpo_vs_gigpo_20261027/sft_runtime_replay.json')
                    if (audit.get('checked') != 200 or audit.get('failures') != []
                        or audit.get('reference_manifest_identity') != reference[0]['identity']
                        or audit.get('current_rollout_sha256') != manifest['source_sha256'][key]):
                        raise ValueError('Historical rollout change requires a matching complete replay audit')
        for key in records:
            if records[key]['row'] != reference[2][key]['row']:
                raise ValueError('Frozen task row changed')
    metrics = {name: summary['breakdown']['all'] for name,(_,summary,_) in loaded.items()}
    behavior = {name: {
        'protocol_error_episodes': sum(r.get('invalid_actions',0)>0 for r in records.values()),
        'schema_error_episodes': sum(r.get('validation_errors',0)>0 for r in records.values()),
        'terminations': dict(Counter(r.get('termination','infrastructure_error') for r in records.values()))
        } for name,(_,_,records) in loaded.items()}
    paired = {}
    grpo, gigpo = loaded['GRPO'][2], loaded['GiGPO'][2]
    for metric in ('official_pass','strict_pass'):
        paired[metric] = dict(Counter('both_pass' if grpo[k].get(metric) and gigpo[k].get(metric)
            else 'gigpo_only' if gigpo[k].get(metric) else 'grpo_only' if grpo[k].get(metric)
            else 'both_fail' for k in grpo))
    report = ROOT/'reports/rl_eval/grpo_vs_gigpo_20261027'
    report.mkdir(parents=True,exist_ok=True)
    comparison = {'runs':runs,'n':200,'metrics':metrics,'behavior':behavior,'grpo_vs_gigpo_paired':paired,
        'generation':reference[0]['config']['generation'],
        'historical_sft_runtime_audit':'reports/rl_eval/grpo_vs_gigpo_20261027/sft_runtime_replay.json',
        'limitation':'One fixed-seed trial per task. GRPO batch 128/100 updates; GiGPO batch 160/80 updates. Equal 12800-trajectory training budget, different batch size.'}
    write_json(report/'comparison.json',comparison)
    lines = ['# GRPO 与 GiGPO：固定测试集对比','',
        '200 道测试题，每题一次采样；使用各自最终 checkpoint，未按测试结果选择权重或重试。','',
        '| 模型 | 官方通过 | 严格通过 | 基础设施错误 | 平均工具调用 |','|---|---:|---:|---:|---:|']
    for name,values in metrics.items():
        lines.append(f"| {name} | {round(values['official_pass_rate']*200)}/200（{values['official_pass_rate']:.1%}） | {round(values['strict_pass_rate']*200)}/200（{values['strict_pass_rate']:.1%}） | {values['infrastructure_errors']} | {values['mean_tool_calls']:.2f} |")
    lines += ['','| 模式 | GRPO 官方 | GiGPO 官方 | GRPO 严格 | GiGPO 严格 |','|---|---:|---:|---:|---:|']
    for mode in range(5):
        a,b=[loaded[name][1]['breakdown'][f'mode/P{mode}'] for name in ('GRPO','GiGPO')]
        lines.append(f"| P{mode} | {a['official_pass_rate']:.1%} | {b['official_pass_rate']:.1%} | {a['strict_pass_rate']:.1%} | {b['strict_pass_rate']:.1%} |")
    lines += ['',f'逐题配对结果：`{paired}`。','',
        '严格通过要求官方通过、显式最终回答且无协议/schema 错误，仍不是完整业务语义验证。',
        '两组训练均采样 12800 条轨迹，但 GRPO 为 batch 128 / 100 步，GiGPO 为 batch 160 / 80 步；差异不能全部归因于信用分配算法。',
        '本次为固定 seed 的单次采样评测，不据此宣称统计显著优势。','']
    lines += ['历史 SFT 与当前 rollout 文件哈希不同；200 条 SFT 已保存轨迹经当前执行循环回放，要求提示词、工具反馈、动作文本及判分一致，不调用模型重新采样。审计记录：`sft_runtime_replay.json`。','']
    (report/'comparison.md').write_text('\n'.join(lines))
    with (report/'paired_episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['id','trial','mode','grpo_official','gigpo_official','grpo_strict','gigpo_strict'])
        writer.writeheader()
        for key in sorted(grpo):
            a,b=grpo[key],gigpo[key]
            writer.writerow(dict(id=key[0],trial=key[1],mode=a['row']['mode'],
                grpo_official=a.get('official_pass',False),gigpo_official=b.get('official_pass',False),
                grpo_strict=a.get('strict_pass',False),gigpo_strict=b.get('strict_pass',False)))
    print(comparison,flush=True)


if __name__ == '__main__':
    main()
