#!/usr/bin/env python3
"""Export explicitly selected qualitative cases, without modifying eval scores."""
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import read_json, write_json

RUNS = {
    'SFT': 'runs/eval/qwen35_2b_sft_vllm_test_20261026',
    'GRPO': 'runs/eval/qwen35_2b_grpo_vllm_test_20261027',
    'StepCredit': 'runs/eval/qwen35_2b_step_credit_vllm_test_20261027',
}

CASES = [
    dict(slug='01_retry', title='暂时性错误：SFT 重复操作，GRPO 重试后完成',
         task='C2_task_100:P1', models=['SFT','GRPO'], expected=[False,True],
         lesson='区分 API 暂时失败与模型参数错误；观察 RL 后重复操作与 schema 错误减少的具体表现。',
         analysis='两者第一次 set_light_state 均收到 Gateway Timeout。GRPO 用相同语义参数重试成功，再调用 apply_lock_state、create_document_and_alert 并提交最终回答。SFT 也曾重试成功，但随后重复设置灯和锁，还把 lock_main/state=lock 传给只支持电源开关的 set_power_state，产生 schema 错误，最终达到 16 轮上限。该例不是“SFT 完全不会重试”，而是展示恢复后的任务推进与协议约束差异。',
         boundary='这是事后选取的一组说明性轨迹，不能用单例替代总体通过率。'),
    dict(slug='02_stop', title='永久性错误：停止下游通知，解释部分完成',
         task='C1_task_041:P2', models=['StepCredit'], expected=[True],
         lesson='Agent 的正确行为可以是承认无法完成，而非一律把所有工具调用完。',
         analysis='前置联系信息、库存预留和替代推荐已执行。order_items 持续返回 Feature Discontinued，StepCredit 最终停止，没有继续调用 push_alert 宣称排序已完成，并在回答中说明排序步骤不可用。',
         boundary='模型仍反复调用已明确永久失效的接口，停止策略不够节制；最终回答还手工列出了可能的排序，不能把官方通过理解为最优决策。所有预留与通知均为模拟工具。'),
    dict(slug='03_transient_corruption', title='暂时性隐式污染：无 error 字段，也需要核验和重试',
         task='C1_task_008:P3', models=['GRPO'], expected=[True],
         lesson='JSON 能解析、工具返回成功，并不保证业务字段可信。',
         analysis='build_contact_alert_payload 首次返回空 user_id、priority=none、空 channel，且将请求消息改成了 Your booking is confirmed and completed.；这与输入的 u_alice/high/app 和原消息不一致。GRPO 用相同输入重试，得到 user_id=u_alice、priority=high、channel=app 和原消息后继续任务。',
         boundary='这里只展示 payload 校验与恢复，不验证模拟市场状态是否符合现实，也不代表执行了真实通知或交易。'),
    dict(slug='04_step_credit_win', title='StepCredit 胜出：切换路径并使用正确对象的快照',
         task='C3_task_094:P3', models=['GRPO','StepCredit'], expected=[False,True],
         lesson='识别污染还不够，后续调用必须真正使用恢复后的数据。',
         analysis='GRPO 先取 mesh 快照，收到 thermostat_hallway/cooling 等与 light_bedroom 不符的数据；虽然又获取了干净的 bridge 数据，归档时却仍传入受污染 mesh 快照，返回 light_ref=thermostat_hallway。StepCredit 先调用 bridge，收到明显不合业务含义的内部调试字符串后，改用 mesh 获取 light_bedroom/on/50/white，并以正确快照归档、重启和保存记录。',
         boundary='这是同一任务与 P3 模式，但多路径 first-touch 机制使两条策略命中的故障工具和污染内容不同，不能称为“完全相同 observation 下的反事实动作对照”；也不能从单例证明 StepCredit 的步骤优势导致了获胜。'),
    dict(slug='05_grpo_win', title='GRPO 胜出：永久故障后走有效替代路径',
         task='C3_task_080:P2', models=['GRPO','StepCredit'], expected=[True,False],
         lesson='保留失败的改进方案；比较真实调用数据流，而非只读最终解释。',
         analysis='两者 mesh 快照接口均返回 API Version Deprecated。GRPO 使用 get_light_bridge_payload_native，并通过对应 publish_light_bridge_payload_native 发布有效的 light_living 快照。StepCredit 再次调用失效的 mesh 接口后，把 get_light_status 的另一种结构当作 mesh_snapshot 发布，结果包含 unknown/default；最终回答仍称“retry succeeded”，与观察不符。',
         boundary='schema 校验通过只表示字段形式满足接口约束，不能代替对象来源、字段映射和最终回答一致性的验证。'),
    dict(slug='06_shared_failure', title='双方共同失败：把异常当作小问题，继续传播到下游',
         task='C1_task_041:P4', models=['GRPO','StepCredit'], expected=[False,False],
         lesson='“在最终回答中提到异常”不等于“阻断了错误操作”。',
         analysis='order_items 返回非空的两项 sorted_items，却给出 count=0；列表顺序为价格 399 在前、249 在后，同时 order=asc。两者均未正确停止，继续 push_alert 宣称推荐已排序。GRPO 在最后提到空列表，StepCredit 将 count=0 称为工具 quirk，但告警已经使用了未验证的结果，官方判分均失败。',
         boundary='两个模型均无协议/schema 错误，失败发生在跨字段一致性检查与后续决策层。这适合讨论未来过程验证器，但本项目尚未完成这类新奖励实验。'),
    dict(slug='07_metric_boundary', title='指标边界：严格通过，最终回答仍虚构了备份完成',
         task='C1_task_088:P2', models=['StepCredit'], expected=[True],
         lesson='主动审查通过样本，说明验证器覆盖范围，而不是只找模型失败。',
         analysis='convert_datetime 持续返回 Account Suspended；StepCredit 停止后续工具调用，未执行 snapshot_workspace，因而满足该任务的停止规则并严格通过。但最终回答声称 Workspace backup: Created successfully，还把 1735689600 错解释为 May 25, 2025，并称共尝试 4 次；实际 convert_datetime 调用了 6 次，该时间戳对应 2025-01-01 00:00:00 UTC。',
         boundary='这是已观察到的最终回答一致性缺陷，不是对全部通过样本的语义错误率估计。它证明 strict_pass 的额外检查仍不足以覆盖“没有工具证据却声称成功”。'),
]


def short(value, limit=240):
    text = json.dumps(value,ensure_ascii=False,separators=(',',':'))
    if len(text)>limit:
        text=text[:limit]+'…【截断，见 JSON】'
    return '<code>'+html.escape(text).replace('|','&#124;')+'</code>'


def main():
    output=ROOT/'reports/project'
    output.mkdir(parents=True,exist_ok=True)
    index=['# 16 轮历史代表性轨迹：成功、分歧与指标边界','',
        '这些案例来自 16 轮预算的历史测试结果，不代表新的 25 轮评测结果；事后选取仅用于定性解释，不参与训练、不改变总分、不重新采样，也不代表各类现象的发生频率。',
        '每个案例保留原始 query、完整 JSON 快照与 SHA-256；页面表格只缩略展示参数和 observation，最终回答原文完整保留。',
        'trace 中的 perturbation_status 是训练/评测侧审计标签，**没有作为 observation 输入策略**；文档展示它仅用于解释。所有工具、联系人、账户、通知、交易均为模拟数据与模拟执行。','',
        '| 案例 | 任务 | 模型 | 讲解重点 |','|---|---|---|---|']
    manifest={'selection':'Post-hoc qualitative examples; no resampling, no performance reweighting', 'cases':[]}
    for case in CASES:
        folder=output/'cases'/case['slug'];folder.mkdir(parents=True,exist_ok=True)
        entries=[]
        lines=[f"# {case['title']}",'',f"任务：`{case['task']}`；来源为 16 轮历史评测，每个模型为该题唯一一次固定 seed 测评。",'',
               '## 观察与解释','',case['analysis'],'',f"**可讲的结论：** {case['lesson']}",'',
               f"**边界与反例：** {case['boundary']}",'']
        for model,expected in zip(case['models'],case['expected']):
            name=case['task'].replace(':','_')+'_00.json'
            source=ROOT/RUNS[model]/'episodes'/name
            record=read_json(source)
            if record['row']['id']!=case['task'] or record['strict_pass']!=expected or record.get('error'):
                raise AssertionError(f'Case evidence changed: {source}')
            target=folder/(model.lower()+'.json');shutil.copyfile(source,target)
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            assert hashlib.sha256(target.read_bytes()).hexdigest()==sha
            entry={'model':model,'source':str(source.relative_to(ROOT)),'snapshot':str(target.relative_to(ROOT)),
                   'sha256':sha,'official_pass':record['official_pass'],'strict_pass':record['strict_pass'],
                   'tool_calls':record['tool_calls'],'invalid_actions':record['invalid_actions'],
                   'validation_errors':record['validation_errors'],'termination':record['termination']}
            entries.append(entry)
            lines += [f'## {model}','',
                f"官方通过：`{record['official_pass']}`；严格通过：`{record['strict_pass']}`；工具调用：{record['tool_calls']}；协议错误：{record['invalid_actions']}；schema 错误：{record['validation_errors']}；终止：`{record['termination']}`。",'',
                f"[完整原始快照]({model.lower()}.json) · [运行原文件]({Path('../../../../')/source.relative_to(ROOT)})",'',
                f"SHA-256：`{sha}`",'', '### 用户请求原文','',
                '````text',record['trace']['messages'][0]['content'],'````','',
                '### 实际工具调用与反馈','',
                '| 工具调用序号 | 工具 | 参数（缩略） | observation（缩略） | 审计标记（策略不可见） |',
                '|---:|---|---|---|---|']
            call=None;count=0
            for message in record['trace']['messages']:
                if message['role']=='assistant' and 'tool_call' in message:
                    call=message['tool_call'];count+=1
                elif message['role']=='tool':
                    if call is None:
                        raise AssertionError('Tool result without preceding action')
                    lines.append(f"| {count} | `{message['name']}` | {short(call['arguments'],190)} | {short(message['content'])} | {message.get('metadata',{}).get('perturbation_status','')} |")
            if count!=record['tool_calls']:
                raise AssertionError('Tool-call counter mismatch')
            final=[m.get('content','') for m in record['trace']['messages']
                   if m['role']=='assistant' and m.get('type')=='final_answer']
            # TraceLogger final messages may omit the type discriminator.
            if not final:
                final=[m.get('content','') for m in record['trace']['messages']
                       if m['role']=='assistant' and 'tool_call' not in m]
            lines += ['', '### 最终回答原文','', '````text',
                      '\n'.join(final) if final else '【该轨迹未提交 final_answer】','````','',
                      '### 官方失败原因','', '````text',
                      str(record['judgement'].get('failure_reason')),'````','']
        write_json(folder/'evidence.json',{'task':case['task'],'title':case['title'],'records':entries})
        (folder/'README.md').write_text('\n'.join(lines).rstrip()+'\n')
        index.append(f"| [{case['title']}](cases/{case['slug']}/README.md) | `{case['task']}` | {' / '.join(case['models'])} | {case['lesson']} |")
        manifest['cases'].append({'slug':case['slug'],'task':case['task'],'records':entries})
    index += ['', '建议面试顺序：先讲案例 01 的训练收益，再讲 04/05 的双向分歧，最后用 06/07 解释下一步改进方向与验证器边界。案例 02/03 用于回答“何时重试、何时停止、如何识别隐式错误”。','']
    (output/'representative_trajectories.md').write_text('\n'.join(index))
    write_json(output/'case_manifest.json',manifest)
    print({'cases':len(CASES),'snapshots':sum(len(c['models']) for c in CASES),'output':str(output)})


if __name__=='__main__':
    main()
