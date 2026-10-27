#!/usr/bin/env python3
"""Promote completed 25-round measurements, retaining 16-round history."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.toolmaze_agent.common import read_json


def main():
    report=ROOT/'reports/rl_eval/rounds25_20261027'
    data=read_json(report/'comparison.json')
    if data['max_rounds']!=25 or data['n']!=200:
        raise ValueError('Expected complete 25-round comparison')
    if '## 25 轮预算更正后的主结果' in (ROOT/'reports/project/experiment_report.md').read_text():
        print('25-round documentation was already promoted; existing history retained.')
        return
    for run in data['runs'].values():
        progress=read_json(ROOT/run/'progress.json')
        if not progress.get('finished') or progress['completed']!=200:
            raise ValueError('Cannot publish incomplete results')
    labels={'base':'原始 Qwen3.5-2B','sft':'全参语言模型 SFT',
            'grpo':'SFT + LoRA GRPO','step_credit':'SFT + LoRA StepCredit'}
    table=['| 模型 | 官方通过 | 严格通过 | 含协议错误的轨迹 | 达到 25 轮上限 |',
           '|---|---:|---:|---:|---:|']
    for name,v in data['metrics'].items():
        behavior=data['behavior'][name]
        table.append(f"| {labels[name]} | {round(v['official_pass_rate']*200)}/200（{v['official_pass_rate']:.1%}） | {round(v['strict_pass_rate']*200)}/200（{v['strict_pass_rate']:.1%}） | {behavior['protocol_error_episodes']} | {behavior['terminations'].get('max_rounds',0)} |")
    a,b,c=[data['metrics'][n] for n in ('sft','grpo','step_credit')]
    finding=(f"25 轮主结果：GRPO 相对 SFT 的严格通过率变化为 {(b['strict_pass_rate']-a['strict_pass_rate'])*100:+.1f} 个百分点，"
             f"StepCredit 相对 SFT 为 {(c['strict_pass_rate']-a['strict_pass_rate'])*100:+.1f} 个百分点。"
             f"StepCredit 相对 GRPO：官方 {(c['official_pass_rate']-b['official_pass_rate'])*100:+.1f}、"
             f"严格 {(c['strict_pass_rate']-b['strict_pass_rate'])*100:+.1f} 个百分点。"
             '两组训练 batch/更新数和恢复条件不同，且仅单 seed，不把该差异解释为纯算法因果效果。')
    p=ROOT/'README.md';s=p.read_text()
    s='\n'.join(line for line in s.splitlines() if not line.startswith('> **评测口径更正'))+'\n'
    start=s.index('| 模型 | 官方通过 | 严格通过 | 含协议错误的轨迹 |')
    end=s.index('\n\n主要发现',start)
    s=s[:start]+'\n'.join(table)+s[end:]
    start=s.index('主要发现：');end=s.index('\n\n严格通过',start)
    s=s[:start]+finding+s[end:]
    s=s.replace('固定测试集 200 个任务实例，每题一次采样；使用最终权重，',
        '**主结果已按 25 轮预算重新测评**：固定测试集 200 个任务实例，每题一次采样；使用最终权重，',1)
    s=s.replace('[完整评测对照表](reports/rl_eval/grpo_vs_step_credit_20261027/comparison.md) / [逐题 CSV](reports/rl_eval/grpo_vs_step_credit_20261027/paired_episodes.csv)',
        '[25 轮完整对照与 16/25 变化](reports/rl_eval/rounds25_20261027/comparison.md) / [25 轮逐题 CSV](reports/rl_eval/rounds25_20261027/paired_episodes.csv) / [16 轮历史结果](reports/rl_eval/grpo_vs_step_credit_20261027/comparison.md)')
    s=s.replace('[7 组代表性轨迹：成功、双向分歧、共同失败、指标盲区]',
        '[16 轮历史代表性轨迹：成功、双向分歧、共同失败、指标盲区]')
    s=s.replace('最多 16 轮，seed=20261026，无 JSON 强约束解码。两组 RL 各用 7 个单卡副本互斥分片，',
        '最多 25 轮，seed=20261026，无 JSON 强约束解码。本次四组模型均用 8 个单卡副本互斥分片，')
    s=s.replace('python tools/run_rl_evaluation.py --gpus 1,2,3,4,5,6,7\npython tools/compare_rl_evaluations.py',
        'python tools/run_budget_evaluation.py --gpus 0,1,2,3,4,5,6,7\npython tools/compare_budget_evaluations.py')
    s=s.replace('四组完整测试轨迹（共 800 条）','16 轮与 25 轮各四组完整测试轨迹（共 1600 条）')
    s+='\n评测预算更正：官方锁定配置的 `execution.max_rounds=25`，旧 16 轮为项目自设预算。此次全部 800 个模型/任务对重新推理，未复用旧完成结果；训练权重未重训。这里只对齐轮次上限，JSON 单动作协议与 8192/512 token 预算仍为本项目设置，并非完整复现官方原生接口配置。\n'
    p.write_text(s)
    p=ROOT/'reports/project/experiment_report.md';s=p.read_text()
    s='\n'.join(line for line in s.splitlines() if not line.startswith('> **评测口径更正'))+'\n'
    s=s.replace('## 9. 面试介绍口径与材料索引',
                '## 9. 16 轮历史介绍词与材料索引（新介绍使用文首 25 轮数字）')
    title,body=s.split('\n',1)
    addendum=['','## 25 轮预算更正后的主结果','',
        '官方配置采用 execution.max_rounds=25；此前 16 轮为项目自设预算，不能作为官方 25 轮设置报告。本次保留原权重，完整重跑四组各 200 题，没有只延长旧失败样本。','']
    current_pitch=(f"统一 25 轮预算、200 题单次评测下，严格通过率分别为：原模型 {data['metrics']['base']['strict_pass_rate']:.1%}，"
                   f"SFT {a['strict_pass_rate']:.1%}，GRPO {b['strict_pass_rate']:.1%}，StepCredit {c['strict_pass_rate']:.1%}。"
                   '介绍项目时使用这一组新数字；不要把下文 16 轮历史数字与新结果混用。')
    addendum+=table+['',finding,'',current_pitch,'',
        '本次其余生成参数、数据及 judge 保持一致；25 是模型决策次数，包含最终回答和非法动作。8192 上下文仍可能导致提前结束。训练/教师采集仍是原 16 轮预算，未重训；单动作 JSON 协议也不同于官方可并行原生工具接口。',
        '固定 seed 的全新推理不保证与旧运行逐 token 相同，因此 16/25 分数变化不能全部归因于轮次。',
        '[完整新报告与预算变化](../rl_eval/rounds25_20261027/comparison.md) · [800 次重测逐题表](../rl_eval/rounds25_20261027/paired_episodes.csv) · [预算配置审计](../rl_eval/rounds25_20261027/protocol_audit.json)','',
        '---','','## 下文为训练事实与 16 轮历史实验原始记录','',
        '下文历史分数、分项统计、定性案例均保留其原 16 轮口径；当前主结论应优先使用上面的 25 轮结果。','']
    p.write_text(title+'\n'+'\n'.join(addendum)+body)
    (report/'README.md').write_text('# 25 轮评测已完成\n\n四组各 200 题，共 800 次全新评测。\n\n[完整对照与 16/25 变化](comparison.md) · [机器可读统计](comparison.json) · [逐题配对表](paired_episodes.csv)。\n\n旧 16 轮结果、训练权重和历史案例均保留；本次仅更改评测轮数预算。\n')
    print('Updated README and experiment report to completed 25-round measurements.')


if __name__=='__main__':main()
