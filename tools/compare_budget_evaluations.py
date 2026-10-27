#!/usr/bin/env python3
"""Compare four equally budgeted reruns and keep historical 16-round results."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.toolmaze_agent.common import load_config, read_json, write_json
from tools.compare_evaluations import load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',default='scripts/toolmaze_agent/configs/eval_rounds25_suite.yaml')
    args=parser.parse_args();suite=load_config(args.suite)
    loaded={entry['name']:load(load_config(entry['config'])['output']) for entry in suite['models']}
    reference=loaded['base']
    runtime_keys=['scripts/toolmaze_agent/'+f for f in ('protocol.py','environment.py','rollout.py','vllm_backend.py')]
    for name,(manifest,summary,records) in loaded.items():
        if records.keys()!=reference[2].keys() or len(records)!=suite['expected_tasks']:
            raise ValueError('Incomplete or unpaired test set')
        if manifest['project_config']['max_rounds']!=suite['max_rounds']:
            raise ValueError('Mixed round budgets')
        for key in ('generation','seed','trials','split'):
            if manifest['config'][key]!=reference[0]['config'][key]:raise ValueError(f'Mixed {key}')
        for key in ('project_config','split_sha256','toolmaze_commit'):
            if manifest[key]!=reference[0][key]:raise ValueError(f'Mixed {key}')
        for key in runtime_keys:
            if manifest['source_sha256'][key]!=reference[0]['source_sha256'][key]:raise ValueError('Mixed runtime')
        if any(records[k]['row']!=reference[2][k]['row'] for k in records):raise ValueError('Task changed')
    metrics={name:s['breakdown']['all'] for name,(_,s,_) in loaded.items()}
    history={entry['name']:read_json(ROOT/entry['historical_16_run']/'summary.json')['breakdown']['all']
             for entry in suite['models']}
    transitions={}
    for entry in suite['models']:
        name=entry['name'];old=load(entry['historical_16_run'])[2];new=loaded[name][2]
        if old.keys()!=new.keys():raise ValueError('Historical test pairing changed')
        old_capped=[k for k in old if old[k].get('termination')=='max_rounds']
        identical_prefix=[]
        for key in old:
            a=[s.get('token_ids') for s in old[key].get('segments',[]) if s['model']]
            b=[s.get('token_ids') for s in new[key].get('segments',[]) if s['model']]
            if a and a==b[:len(a)]:identical_prefix.append(key)
        transitions[name]={
            'old_max_rounds':len(old_capped),
            'old_capped_now_official_pass':sum(new[k].get('official_pass',False) for k in old_capped),
            'old_capped_now_strict_pass':sum(new[k].get('strict_pass',False) for k in old_capped),
            'strict_gained':sum(not old[k].get('strict_pass') and new[k].get('strict_pass',False) for k in old),
            'strict_lost':sum(old[k].get('strict_pass',False) and not new[k].get('strict_pass') for k in old),
            'complete_old_action_sequence_is_exact_prefix_of_new':len(identical_prefix)}
    behavior={name:{'terminations':dict(Counter(r.get('termination','error') for r in records.values())),
        'protocol_error_episodes':sum(r.get('invalid_actions',0)>0 for r in records.values()),
        'schema_error_episodes':sum(r.get('validation_errors',0)>0 for r in records.values()),
        'max_model_decisions':max(sum(s['model'] for s in r.get('segments',[])) for r in records.values())}
        for name,(_,_,records) in loaded.items()}
    paired={}
    for field in ('official_pass','strict_pass'):
        a,b=loaded['grpo'][2],loaded['step_credit'][2]
        paired[field]=dict(Counter('both_pass' if a[k].get(field) and b[k].get(field)
            else 'step_credit_only' if b[k].get(field) else 'grpo_only' if a[k].get(field)
            else 'both_fail' for k in a))
    report=ROOT/suite['report_output'];report.mkdir(parents=True,exist_ok=True)
    result={'max_rounds':suite['max_rounds'],'n':suite['expected_tasks'],'metrics':metrics,
        'historical_16_metrics':history,'budget_transitions':transitions,'behavior':behavior,'paired_grpo_vs_step_credit':paired,
        'runs':{e['name']:load_config(e['config'])['output'] for e in suite['models']},
        'scope':'Fresh rerun of all 800 model/task pairs. Only evaluation decision limit changes to 25; existing weights retain their original 16-round training/collection budget.',
        'official_alignment':'Aligns the pinned official execution.max_rounds=25. Student JSON single-action protocol and 8192 context/512 action tokens remain project-specific; not a full reproduction of the official inference setup.'}
    write_json(report/'comparison.json',result)
    lines=['# 四组模型：25 轮评测重新对照','',
        '四组最终权重、同一冻结测试集 200 题、每题一次；本次全新推理，不继承旧 16 轮轨迹，不按结果筛选。','',
        '| 模型 | 官方通过 | 严格通过 | 基础设施错误 | 平均工具调用 |','|---|---:|---:|---:|---:|']
    for name,v in metrics.items():
        lines.append(f"| {name} | {round(v['official_pass_rate']*200)}/200（{v['official_pass_rate']:.1%}） | {round(v['strict_pass_rate']*200)}/200（{v['strict_pass_rate']:.1%}） | {v['infrastructure_errors']} | {v['mean_tool_calls']:.3f} |")
    lines+=['','## 16 轮历史结果与 25 轮新结果','',
        '| 模型 | 16 轮官方 | 25 轮官方 | 16 轮严格 | 25 轮严格 |','|---|---:|---:|---:|---:|']
    for name,v in metrics.items():
        h=history[name]
        lines.append(f"| {name} | {h['official_pass_rate']:.1%} | {v['official_pass_rate']:.1%} | {h['strict_pass_rate']:.1%} | {v['strict_pass_rate']:.1%} |")
    lines+=['','| 模型 | 旧轮数上限终止 | 这些题新严格通过 | 全体新获严格通过 | 全体丢失严格通过 | 旧动作序列仍为新序列精确前缀 |',
        '|---|---:|---:|---:|---:|---:|']
    for name,t in transitions.items():
        lines.append(f"| {name} | {t['old_max_rounds']} | {t['old_capped_now_strict_pass']} | {t['strict_gained']} | {t['strict_lost']} | {t['complete_old_action_sequence_is_exact_prefix_of_new']} |")
    lines+=['','## 分扰动模式（官方 / 严格）','',
        '| 模式 | Base | SFT | GRPO | StepCredit |','|---|---:|---:|---:|---:|']
    for mode in range(5):
        values=[loaded[n][1]['breakdown'][f'mode/P{mode}'] for n in loaded]
        lines.append('| P'+str(mode)+' | '+' | '.join(f"{v['official_pass_rate']:.1%} / {v['strict_pass_rate']:.1%}" for v in values)+' |')
    lines+=['','## 计数与限制','',
        '25 指模型决策轮次，包含最终回答和非法动作尝试，不是允许先调用 25 次工具再额外回答。官方原生接口一轮可含多个工具调用，本项目学生协议仍是一轮一个动作。',
        '其他设置保持：temperature=.8、top_p=1、top_k 关闭、上下文 8192、单动作最多 512 tokens、seed=20261026、同一任务数据及 judge。上下文预算仍可能使轨迹在 25 轮前结束。',
        '本次只重评测；SFT/GRPO/StepCredit 权重没有在 25 轮预算下重训，训练与评测预算不同。GRPO/StepCredit 原来的 batch/更新数和回放恢复差异仍存在。',
        '旧 16 轮结果是项目自设预算下的历史结果，不能称为官方 25 轮设置。本次只对齐官方轮次上限，不宣称完全复现官方所有推理设置。',
        '所有测试题重跑，未只延长曾失败的样本；固定 seed 也不保证不同调度下逐 bit 一致，因此预算变化对单题结果不必单调。',
        '严格通过仍不等于完整业务语义正确。每个底层测试任务有五种扰动，实例之间相关；单 seed 结果不用于统计显著性宣称。','',f'GRPO/StepCredit 逐题配对：`{paired}`。','']
    (report/'comparison.md').write_text('\n'.join(lines))
    fields=['id','trial','mode']+[n+'_'+m for n in loaded for m in ('official','strict','termination','decisions')]
    with (report/'paired_episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for key in sorted(reference[2]):
            row={'id':key[0],'trial':key[1],'mode':reference[2][key]['row']['mode']}
            for name,(_,_,records) in loaded.items():
                r=records[key];row.update({name+'_official':r.get('official_pass',False),
                    name+'_strict':r.get('strict_pass',False),name+'_termination':r.get('termination','error'),
                    name+'_decisions':sum(s['model'] for s in r.get('segments',[]))})
            writer.writerow(row)
    print(result,flush=True)


if __name__=='__main__':main()
