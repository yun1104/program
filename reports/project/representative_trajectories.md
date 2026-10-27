# 16 轮历史代表性轨迹：成功、分歧与指标边界

这些案例来自 16 轮预算的历史测试结果，不代表新的 25 轮评测结果；事后选取仅用于定性解释，不参与训练、不改变总分、不重新采样，也不代表各类现象的发生频率。
每个案例保留原始 query、完整 JSON 快照与 SHA-256；页面表格只缩略展示参数和 observation，最终回答原文完整保留。
trace 中的 perturbation_status 是训练/评测侧审计标签，**没有作为 observation 输入策略**；文档展示它仅用于解释。所有工具、联系人、账户、通知、交易均为模拟数据与模拟执行。

| 案例 | 任务 | 模型 | 讲解重点 |
|---|---|---|---|
| [暂时性错误：SFT 重复操作，GRPO 重试后完成](cases/01_retry/README.md) | `C2_task_100:P1` | SFT / GRPO | 区分 API 暂时失败与模型参数错误；观察 RL 后重复操作与 schema 错误减少的具体表现。 |
| [永久性错误：停止下游通知，解释部分完成](cases/02_stop/README.md) | `C1_task_041:P2` | StepCredit | Agent 的正确行为可以是承认无法完成，而非一律把所有工具调用完。 |
| [暂时性隐式污染：无 error 字段，也需要核验和重试](cases/03_transient_corruption/README.md) | `C1_task_008:P3` | GRPO | JSON 能解析、工具返回成功，并不保证业务字段可信。 |
| [StepCredit 胜出：切换路径并使用正确对象的快照](cases/04_step_credit_win/README.md) | `C3_task_094:P3` | GRPO / StepCredit | 识别污染还不够，后续调用必须真正使用恢复后的数据。 |
| [GRPO 胜出：永久故障后走有效替代路径](cases/05_grpo_win/README.md) | `C3_task_080:P2` | GRPO / StepCredit | 保留失败的改进方案；比较真实调用数据流，而非只读最终解释。 |
| [双方共同失败：把异常当作小问题，继续传播到下游](cases/06_shared_failure/README.md) | `C1_task_041:P4` | GRPO / StepCredit | “在最终回答中提到异常”不等于“阻断了错误操作”。 |
| [指标边界：严格通过，最终回答仍虚构了备份完成](cases/07_metric_boundary/README.md) | `C1_task_088:P2` | StepCredit | 主动审查通过样本，说明验证器覆盖范围，而不是只找模型失败。 |

建议面试顺序：先讲案例 01 的训练收益，再讲 04/05 的双向分歧，最后用 06/07 解释下一步改进方向与验证器边界。案例 02/03 用于回答“何时重试、何时停止、如何识别隐式错误”。
