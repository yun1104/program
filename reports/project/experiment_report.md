# ToolMaze 多轮 Agent 后训练实验报告

## 25 轮预算更正后的主结果

官方配置采用 execution.max_rounds=25；此前 16 轮为项目自设预算，不能作为官方 25 轮设置报告。本次保留原权重，完整重跑四组各 200 题，没有只延长旧失败样本。

| 模型 | 官方通过 | 严格通过 | 含协议错误的轨迹 | 达到 25 轮上限 |
|---|---:|---:|---:|---:|
| 原始 Qwen3.5-2B | 26/200（13.0%） | 13/200（6.5%） | 130 | 91 |
| 全参语言模型 SFT | 106/200（53.0%） | 88/200（44.0%） | 53 | 14 |
| SFT + LoRA GRPO | 110/200（55.0%） | 105/200（52.5%） | 21 | 5 |
| SFT + LoRA GiGPO | 114/200（57.0%） | 107/200（53.5%） | 24 | 2 |

25 轮主结果：GRPO 相对 SFT 的严格通过率变化为 +8.5 个百分点，GiGPO 相对 SFT 为 +9.5 个百分点。GiGPO 相对 GRPO：官方 +2.0、严格 +1.0 个百分点。两组训练 batch/更新数和恢复条件不同，且仅单 seed，不把该差异解释为纯算法因果效果。

统一 25 轮预算、200 题单次评测下，严格通过率分别为：原模型 6.5%，SFT 44.0%，GRPO 52.5%，GiGPO 53.5%。介绍项目时使用这一组新数字；不要把下文 16 轮历史数字与新结果混用。

本次其余生成参数、数据及 judge 保持一致；25 是模型决策次数，包含最终回答和非法动作。8192 上下文仍可能导致提前结束。训练/教师采集仍是原 16 轮预算，未重训；单动作 JSON 协议也不同于官方可并行原生工具接口。
固定 seed 的全新推理不保证与旧运行逐 token 相同，因此 16/25 分数变化不能全部归因于轮次。
[完整新报告与预算变化](../rl_eval/rounds25_20261027/comparison.md) · [800 次重测逐题表](../rl_eval/rounds25_20261027/paired_episodes.csv) · [预算配置审计](../rl_eval/rounds25_20261027/protocol_audit.json)

---

## 下文为训练事实与 16 轮历史实验原始记录

下文历史分数、分项统计、定性案例均保留其原 16 轮口径；当前主结论应优先使用上面的 25 轮结果。

更新：2026-10-27。对象：本项目已经完成的教师采集、SFT、GRPO、GiGPO 及最终 checkpoint 测评。本文区分观察结果、实现事实与尚待验证的解释。

## 1. 实验问题与结论

问题：在真实执行模拟工具的多轮任务中，Qwen3.5-2B 能否通过教师蒸馏和 RL 学会正确调用工具、从故障中恢复、避免传播污染数据，并在无法完成时合理停止？在相同终局奖励下，增加步骤级信用分配是否带来一致改善？

已完成链路：原模型 → 教师轨迹 SFT → 从同一 SFT 分别训练 LoRA GRPO / GiGPO → 同一冻结测试协议。GiGPO 属于已有算法思路的适配实现，并非提出新算法。

| 模型 | 官方通过 | 严格通过 | 相对 SFT 的官方变化 | 相对 SFT 的严格变化 |
|---|---:|---:|---:|---:|
| 原始 Qwen3.5-2B | 29/200（14.5%） | 19/200（9.5%） | — | — |
| SFT | 103/200（51.5%） | 82/200（41.0%） | — | — |
| GRPO | 110/200（55.0%） | 105/200（52.5%） | +3.5 个百分点 | +11.5 个百分点 |
| GiGPO | 112/200（56.0%） | 103/200（51.5%） | +4.5 个百分点 | +10.5 个百分点 |

主要结论：SFT 带来最大的提升；两组 RL 进一步改善严格通过。GiGPO 相比 GRPO 官方多通过 2 题，严格少通过 2 题，**本次没有一致优势**。单 seed、训练 batch/更新数不同，不能据此得出算法普遍优劣或统计显著结论。

来源：[机器可读对照](../rl_eval/grpo_vs_gigpo_20261027/comparison.json)、[逐题配对 CSV](../rl_eval/grpo_vs_gigpo_20261027/paired_episodes.csv)。

## 2. 环境、数据与训练边界

### 2.1 固定版本与任务

- ToolMaze 代码：`f8620d841b67a72ac921caee9df87af6ab9befbb`。
- 官方数据：`08b0239a98b0b07839d673f5bc0ef7d836db9296`。
- 共 400 个底层任务、2,000 个任务/扰动实例，覆盖 C1—C4 和 P0—P4。
- 模拟插件实际执行并更新轨迹；工具返回不是模型自编 observation。未连接真实业务 API。
- P0 无故障；P1 暂时性显式错误；P2 永久性显式错误；P3 暂时性隐式污染；P4 永久性隐式污染。
- 判分按任务复杂度区分工具覆盖、调用次数、有效路径及恢复行为。对于多路径任务，永久故障后可能应换路而不是直接停止。

学生能看到 query 和公共工具定义，不能看到标准路径、故障模式、故障注入标记或期望结果。轨迹中的 `perturbation_status` 是审计元数据，不加入策略 observation。

证据：[上游锁定文件](../../tools/upstream.lock.json)、[环境适配](../../scripts/toolmaze_agent/environment.py)。

### 2.2 划分与泄漏控制

在采集前，按 `template_id`、`task_id`、规范化 query 的连通分组切分。P0—P4 不跨集合，任务文件在加载时校验 SHA-256。

| 集合 | 实例 | 底层任务/分组 | 各复杂度实例数 | 各扰动模式实例数 |
|---|---:|---:|---:|---:|
| train | 1600 | 320 | C1—C4 各 400 | P0—P4 各 320 |
| validation | 200 | 40 | C1—C4 各 50 | P0—P4 各 40 |
| test | 200 | 40 | C1—C4 各 50 | P0—P4 各 40 |

200 个测试实例不是 200 个相互独立的底层任务，存在同任务不同扰动之间的相关性。本轮没有使用 validation 搜索超参数，也没有根据 test 选择最优 checkpoint；评测各训练阶段的最终权重。

SFT 与 RL 共享 train 合理：SFT 先学习教师成功执行方式，RL 从学生策略产生行为并依据环境反馈优化。RL 使用所有 1600 道题，包含未获得教师成功样本的 818 道；validation/test 不参与训练。GRPO 的历史日志恢复见第 6 节，不能笼统称为全程全新采样。

### 2.3 教师采集与 SFT 数据

教师服务端模型标识：`deepseek-v4.1-flash`。对全部 train 每题一次采样，最终并发数 64；1600 条中验收 782 条，基础设施错误为 0。

| 模式 | 合格轨迹 |
|---|---:|
| P0 | 226 |
| P1 | 234 |
| P2 | 119 |
| P3 | 125 |
| P4 | 78 |

教师使用 Anthropic 原生工具调用，运行时真实执行模拟工具，再转成学生 JSON 文本动作。保留原生调用记录及 provider 信息，不将教师验收比例直接当作学生同协议 benchmark 成绩。服务端模型名不等同于可验证的权重 commit。

782 条均适配 8192 上下文；序列长度最小 1517、中位数 2749、P95 5239、最大 7164；监督 token 总计 614303。没有截断成功轨迹，也没有把超长的成功前缀伪装成完整成功。

SFT 文件 SHA-256：`b26f624555968e4d6aafd951bf226f3675ab75866b3178f76253b8e14ce369a6`。

证据：[采集状态](../../runs/teacher/opencode_v41_flash_20261026_native_v2/progress.json)、[数据 manifest](../../runs/sft_data/manifest.json)、[数据审计](../teacher/sft_data_audit.json)。

## 3. 协议、判分和奖励

### 3.1 共用学生交互协议

模型生成一个动作 JSON，以 `</action>` 结束。动作是 `tool_call` 或 `final_answer`。运行时先校验工具参数 schema，再执行工具、追加 observation。每条轨迹最多 16 次模型决策；协议错误也会消耗决策轮次。工具调用数不等于模型决策数。

SFT 仅监督模型动作 token；提示和工具返回 label 为 `-100`。RL 保存实际采样 token ID 与采样概率，对模型动作施加损失；环境 token、padding 和运行时补充的终止 EOS 不直接作为策略损失或这里的 KL 损失位置。这不等于切断上下文计算图：observation 仍是后续动作的输入，其上下文计算可影响参数梯度。

### 3.2 官方通过与严格通过

当前锁定版本的 `JudgeSystem` 用程序规则检查实际轨迹，**不调用额外判分模型**。构造函数虽然保留 `llm_client`/`model`，当前判分路径未使用 LLM 调用。

```text
strict_pass = official_pass
              AND explicit final_answer
              AND invalid_actions == 0
              AND validation_errors == 0
```

环境故意注入的工具故障不是 schema 错误。schema 校验仅检查参数形式，不保证业务字段、对象身份、跨工具数据映射正确。strict_pass 也不会完整核对最终自然语言回答。

案例 07 是实际反例：GiGPO 停止了后续工具调用，官方和严格检查均通过，但回答声称完成未调用的备份。不能把“严格通过率”命名为“真实事务端到端语义正确率”。

### 3.3 相同终局奖励

两组 RL 均使用：

$$
R(\tau)=
\begin{cases}
1-\min(0.002N_{tool},0.2),& strict\_pass\\
-0.1\min(N_{invalid}+N_{schema},5),& otherwise
\end{cases}
$$

没有协议/schema 错误的失败为 0。成功才扣调用成本，避免鼓励失败后提前结束来节省调用；没有奖励“任意一次成功工具调用”。GiGPO 实验改变的是信用分配，不是增加了 PRM 或人工步骤奖励。

代码：[交互与奖励](../../scripts/toolmaze_agent/rollout.py)、[官方 judge（锁定版本）](https://github.com/Zhudongsheng75/ToolMaze/blob/f8620d841b67a72ac921caee9df87af6ab9befbb/evaluation/core/judge.py)。

## 4. 训练方法与实际配置

### 4.1 SFT

Qwen3.5-2B 更新全部 1,881,825,088 个语言模型参数，视觉模块冻结；这不是 LoRA，也不是整个多模态模型的所有参数都训练。TRL SFTTrainer + DeepSpeed ZeRO-2，5 张 3090，per-device batch 6，有效 batch 30，2 epochs / 54 steps，lr=1e-5，warmup=3，cosine，BF16。

使用分块 NLL 避免完整 `B × T × vocab` logits。项目内 FLA 加速 Gated Delta Rule，因缺少兼容 causal-conv 内核，卷积使用 PyTorch fallback。完整模型 final 链接至 checkpoint-54；受磁盘限制只保存模型，不保存可精确恢复的优化器状态。

### 4.2 普通 GRPO 与 GiGPO

| 参数 | GRPO | GiGPO |
|---|---|---|
| 起点 | 同一 SFT final | 同一 SFT final |
| 训练题数 | 1600 | 1600 |
| LoRA | r64 / alpha128 / dropout0 | 相同 |
| 可训练参数 | 67276800 | 相同 |
| 训练卡 / vLLM 卡 | 4 / 1 | 5 / 1 |
| per-device batch | 8 | 2 |
| 梯度累积 | 4 | 16 |
| 有效轨迹 batch | 128 | 160 |
| 每题 generations | 8 | 8 |
| 每更新任务数 | 16 | 20 |
| 更新数 | 100 | 80 |
| 最终训练轨迹预算 | 12800 | 12800 |
| lr / schedule / warmup | 1e-5 / cosine / 5 | 相同更新单位设置 |
| KL beta / 参考策略 | .01 / 冻结 SFT | 相同 |
| context / 每动作最大 tokens | 8192 / 512 | 相同 |
| temperature / top_p / top_k | .8 / 1 / 关闭 | 相同 |

两组均为 TRL + DeepSpeed ZeRO-2。通过关闭 adapter 得到 SFT 参考概率，不额外常驻一份参考模型。vLLM 每批采样前同步当前策略；使用保持冻结底座不变的无状态 LoRA 合并，避免反复 BF16 merge/unmerge 积累误差。对采样/训练后端概率差异使用 token 级截断重要性校正，上限 2。

GRPO 梯度路径使用 fused/chunked loss，旧策略和参考策略概率分块计算。当前 Liger GRPO 内核仅接受 `(B,)` 轨迹级优势；GiGPO 改用 checkpointed、64-token 分块的输出头损失，接受 `(B,T)` 优势。两者不应被描述为完全相同的内核实现。

### 4.3 GiGPO 的步骤信用分配

参考 GiGPO 原文的折扣回报、anchor grouping 和组合优势思想，使用采样标准差与 `epsilon=1e-4`，与本项目 TRL 组归一化方式一致：

$$
A_i^E=\frac{R_i-\operatorname{mean}_{j\in group}(R_j)}{\operatorname{std}_{j\in group}(R_j)+10^{-4}}
$$

终局奖励放在最后一个模型决策，包括没有提交最终回答的失败轨迹，因此：

$$
G_{i,t}=\gamma^{T_i-1-t}R_i,\quad
A_{i,t}^S=\frac{G_{i,t}-\operatorname{mean}_{(j,u)\in anchor}(G_{j,u})}{\operatorname{std}_{(j,u)\in anchor}(G_{j,u})+10^{-4}},\quad
A_{i,t}=A_i^E+\omega A_{i,t}^S
$$

`gamma=.95, omega=1`。同一动作的所有实际生成 token 获得相同动作优势。单样本 anchor 和常数回报组的步骤优势为 0，不人为制造信号；`omega=0` 可作为同执行栈的轨迹级消融。

anchor key 是任务身份与**完整策略输入 token 前缀**的 SHA-256，并限制在同一任务 rollout group 内。它没有使用标准路径或未来答案进行状态匹配。它比仅按“工具名/第几步/最近 observation”分组保守，但不能复现所有环境状态回访，尤其在历史分叉后可能错失等价环境状态。

全量版本先按 rank 顺序汇总所有样本，整体构建任务组和 anchor 组，再按样本数切回各 rank。当前每批 rank counts 为 `[32,32,32,32,32]`；专门测试了一个任务组跨 rank 边界的情况。

这是面向现有 TRL token-level clipped objective 的适配，非官方 verl-agent 的全部设置复现；没有新增 critic、PRM、分支采样或环境重置搜索。

### 4.4 步骤信号是否真的存在

对全量 80 份 `advantages/*.json` 求和：

| 计数 | 值 |
|---|---:|
| 模型决策总数 | 125425 |
| 非初始动作数 | 112625 |
| 匹配到共享 anchor 的非初始动作 | 58024（51.52%） |
| 获得非零步骤优势的非初始动作 | 33775（29.99%） |

分母是所有非初始动作，不是仅匹配成功的动作。共享 anchor 不保证组内回报不同，因此匹配率高于非零信号率。这些数字证明步骤级信号被实际构造，不证明它导致了测试提升。

证据：[GiGPO 全量输出](../../runs/gigpo_full_20261027/advantages)、[优势计算](../../scripts/toolmaze_agent/gigpo.py)、[多卡实现](../../scripts/toolmaze_agent/gigpo_distributed.py)、[GiGPO 论文](https://papers.neurips.cc/paper_files/paper/2025/file/420c9f777c0b4f78d515e53cf74d58b2-Paper-Conference.pdf)。

## 5. 评测协议与结果

### 5.1 固定协议

200 个 test 实例，每题 1 次，seed=20261026；BF16 vLLM，temperature .8、top_p 1、top_k 关闭，最大上下文 8192、每次动作 512 tokens、最多 16 轮。不使用强制 JSON 解码，不根据失败重采样，不根据 test 挑选权重。请求/试验对全部进入分母。

两组 RL 各用 7 个单卡 vLLM 副本，任务互斥分片，合并校验全覆盖。每题/每轮 seed 由任务身份和轮次决定，不依赖完成先后；但不能保证不同 batch 调度/硬件下逐 bit 相同。

RL final 为 adapter。评测时在 CPU 用 `(base.float()+delta.float()).to(BF16)` 合并，再卸载 adapter，保持与训练采样器相同的合并公式。PEFT 默认 safe merge 的额外中间舍入在预检时被识别，未进入最终评测。所有合并层检查有限值，保存底座/adapter 来源与合并权重哈希；CPU/GPU 仍可能存在浮点差异。临时完整推理模型在测完后清理，原始模型和 adapter 保留。

历史 SFT 与当前 `rollout.py` 的文件哈希不同。额外回放全部 200 条历史 SFT 动作，要求初始消息、observation、动作文本和通过/终止结果一致，结果 **200/200 一致**，无模型重新采样。GRPO/GiGPO 当前评测协议、环境、rollout 和 vLLM backend 的源码哈希一致。

证据：[SFT 历史回放](../rl_eval/grpo_vs_gigpo_20261027/sft_runtime_replay.json)、[GRPO 权重来源](../../artifacts/qwen35_2b_grpo_model_20261027.json)、[GiGPO 权重来源](../../artifacts/qwen35_2b_gigpo_model_20261027.json)。

### 5.2 P0—P4 分项

每种模式 40 题；下表为官方 / 严格通过率。

| 模式 | SFT | GRPO | GiGPO |
|---|---:|---:|---:|
| P0 | 67.5% / 52.5% | 72.5% / 65.0% | 77.5% / 67.5% |
| P1 | 67.5% / 60.0% | 70.0% / 70.0% | 72.5% / 65.0% |
| P2 | 37.5% / 30.0% | 42.5% / 40.0% | 37.5% / 35.0% |
| P3 | 55.0% / 40.0% | 55.0% / 52.5% | 55.0% / 52.5% |
| P4 | 30.0% / 22.5% | 35.0% / 35.0% | 37.5% / 37.5% |

GiGPO 的 P0、P4 本次略高，P2 低于 GRPO；P1 官方/严格的方向相反。不能概括为“GiGPO 提升了所有故障恢复”。每模式样本少、仅一次采样，不做稳定提升宣称。

### 5.3 逐题配对

| 指标 | 两者都通过 | GiGPO 独有通过 | GRPO 独有通过 | 两者都失败 |
|---|---:|---:|---:|---:|
| 官方 | 100 | 12 | 10 | 78 |
| 严格 | 93 | 10 | 12 | 85 |

不是只有两道题发生变化；官方有 22 道分歧，正负抵消后净差 2 道。严格指标也有 22 道分歧，净差为相反方向。

### 5.4 行为与成本

| 模型 | 含协议错误轨迹 | 含 schema 错误轨迹 | 最终回答结束 | 达到轮数上限 | 上下文预算终止 | 平均工具调用 | 平均累计 tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 121 | 22 | 117 | 81 | 2 | 6.070 | 29003.980 |
| SFT | 51 | 18 | 159 | 39 | 2 | 8.850 | 27669.595 |
| GRPO | 20 | 6 | 186 | 13 | 1 | 8.280 | 25217.045 |
| GiGPO | 27 | 7 | 173 | 26 | 1 | 8.565 | 26272.560 |

协议/schema 计数可能重叠，不是互斥分类。累计 tokens 是所有轮输入和输出之和，可超过单次上下文上限；也不等于模型独立生成的 token 数。

当前 GRPO 的协议错误和轮数上限终止均少于 GiGPO。工具调用和 tokens 是全体题目均值，包含失败和提前结束；不能只凭均值评价成功任务效率。Base 调用更少却能力更差，正说明“便宜失败”不是有效优化。

## 6. 工程验证与恢复事实

1. **大词表内存控制：** SFT 分块 NLL；RL 旧/参考概率分块计算。GiGPO 使用 checkpointed 输出头支持逐动作优势，环境 token 损失屏蔽，并与完整计算比较损失及 hidden-state 梯度。
2. **概率与 token 对齐：** 保存实际采样 token ID，不将生成文本重新编码替代原序列；记录 log-prob，对 vLLM/HF 差异做截断校正。
3. **采样网络容错：** 有限次数重试连接错误和 429/502/503/504，保持同输入和 seed；不重试模型质量失败，不筛成功样本。
4. **GRPO 日志恢复：** 原任务前 10 步没有可用 checkpoint，不能声称精确恢复。最终 recovery run 从 SFT 重建前 10 步，使用先前 1280 条学生轨迹及其概率，之后转在线采样；这是回放重建，与教师轨迹复用不同，也限制了与全新在线 GiGPO 的受控比较。
5. **真实 native 恢复：** 保存并从 checkpoint-2 恢复 optimizer/RNG，恢复 step 和 LoRA_B 范数一致，随后完成第 3 步。两组最终 run 都完成全部步数并保存最终 adapter。
6. **checkpoint 内容：** 冻结底座不重复存储；处理 tied embedding / LM-head 别名，保存 adapter、分片优化器、scheduler 和 RNG。完成标记写在所有 rank 保存后。
7. **GiGPO 从单卡到多卡：** 单卡 3 步、12 条真实轨迹先跑通，再扩展全量；12 条轨迹的前缀 anchor 重建与采样时记录一致；测试了跨 rank 拆分任务组。

最终 `train()` 记录的 runtime：SFT 677.8676 秒，GRPO 24171.3251 秒，GiGPO 22478.1233 秒。GRPO 为最终恢复调用记录，不包含全部历史失败、首次两步与中断等待；两组卡数、batch、内核和回放条件不同，**不能据此宣称 GiGPO 更快**。

证据：[SFT 配置](../../scripts/toolmaze_agent/configs/sft_full_trl.yaml)、[GRPO recovery 配置](../../scripts/toolmaze_agent/configs/rl_lora_recovery.yaml)、[GiGPO 配置](../../scripts/toolmaze_agent/configs/gigpo_full.yaml)、[native 恢复验证](../rl/native_resume_validation.json)、[GRPO 测试日志](../rl_recovery_tests_final.log)、[GiGPO 8 项测试](../gigpo/gigpo_full_20261027/unit_tests.log)、[单卡验证摘要](../gigpo/gigpo_smoke_20261026/smoke_summary.json)、[真实 anchor 重建审计](../gigpo/gigpo_full_20261027/anchor_reconstruction_audit.json)。

## 7. 定性案例与指标边界

[完整案例集](representative_trajectories.md)含 7 组、11 份原始轨迹快照。所有案例从既有 test 结果事后选取，只用于解释，不改变评分，也不用于训练或估计现象频率。

| 案例 | 实际现象 | 应讲清的边界 |
|---|---|---|
| 01，C2_task_100:P1 | GRPO 超时后重试并完成；SFT 恢复后重复调用且有 schema 错误 | 不能说 SFT 完全不会重试 |
| 02，C1_task_041:P2 | GiGPO 对永久失效排序工具最终停止，不继续通知 | 已永久错误仍多次重试，谈不上最优停止 |
| 03，C1_task_008:P3 | GRPO 对空收件人/错误优先级 payload 重试，拿到恢复结果 | 工具无 error 不代表字段可信 |
| 04，C3_task_094:P3 | GiGPO 换路后归档正确灯具；GRPO 归档污染对象 | first-touch 使污染工具/内容不同，不是同 observation 反事实实验 |
| 05，C3_task_080:P2 | GRPO 用有效 bridge 路径；GiGPO 把错误结构作为 mesh 快照 | schema 通过不足以验证数据流，最终“retry succeeded”与观察矛盾 |
| 06，C1_task_041:P4 | 两者看到 count/列表不一致仍继续发模拟通知 | 事后承认异常不能撤销已传播的错误 |
| 07，C1_task_088:P2 | GiGPO 严格通过，却声称完成未执行的备份 | 停止规则通过不等于最终回答有工具证据 |

案例 07 还包含错误日期和错误重试次数。这里只证明存在验证盲区，不能把个案发现推算成全部通过样本的语义错误率。

## 8. 当前可以与不可以归因的结论

**已观察到：** 本数据和协议下，SFT、两组 RL 的严格通过率依次提高；GiGPO 步骤信号确实存在；GiGPO 没有一致优于 GRPO；模型会把异常数据传播给后续工具，也会在规则通过后虚构完成情况。

**不能直接归因：** 不能将 GiGPO 某次获胜断言为步骤优势的因果作用；不能认为官方/严格分数就是完整语义正确；不能将近似相同训练预算视为同 batch 消融；不能凭一次运行给算法排名或声称跨域泛化；不能把模拟工具结果当作真实生产能力。

待验证的解释包括：精确前缀匹配覆盖有限、同回报组缺少局部信号、不同 batch/schedule、终局验证器覆盖不足。P4 教师合格数据较少也是分布事实，但尚无实验确认它导致了 P4 表现。以上均是后续假设，不是已证实原因。

下一轮优先：固定 batch、更新数、seed 及回放条件，比较 `omega=0` 与 `omega=1`；再做多 seed 或按底层任务分组的重复评测。之后才考虑可验证子目标奖励、状态压缩/等价性和最终回答证据校验，避免把更多训练成本花在无法解释的组合改动上。

## 9. 16 轮历史介绍词与材料索引（新介绍使用文首 25 轮数字）

可用的简短叙述：

> 我围绕小模型在多轮工具故障中的恢复能力做了一条完整后训练链路。先按任务组冻结数据划分，采集教师轨迹，对 Qwen3.5-2B 做全参语言模型 SFT，再比较 LoRA GRPO 与适配的 GiGPO。实现重点是实际环境 rollout、模型/环境 token 的损失隔离、显存受限训练和跨卡步骤优势。200 题测试中，严格通过率从原模型 9.5% 提升到 SFT 41%，GRPO 52.5%，GiGPO 51.5%。GiGPO 没有一致胜出，所以我保留了双向失败分析，发现了数据污染传播和最终回答虚构完成的问题，也明确了下一步需要同 batch 消融和更完整的语义验证。

这段话描述项目事实，不应扩写成“原创 GiGPO”“复现所有论文设置”或“证明真实业务成功率”。面试前应能独立解释代码、优势公式、失败轨迹与实验限制。

| 材料 | 位置 |
|---|---|
| 项目总览与运行入口 | [README](../../README.md) |
| 案例目录 | [代表性轨迹](representative_trajectories.md) |
| 案例原始文件、快照与哈希 | [case_manifest.json](case_manifest.json) |
| 评测总体与逐题结果 | [comparison.json](../rl_eval/grpo_vs_gigpo_20261027/comparison.json)、[CSV](../rl_eval/grpo_vs_gigpo_20261027/paired_episodes.csv) |
| SFT 训练 manifest | [manifest.json](../../runs/sft_full_trl_20261026/manifest.json) |
| GRPO 训练 manifest | [manifest.json](../../runs/grpo_lora_recovery_20261026/manifest.json) |
| GiGPO 训练 manifest | [manifest.json](../../runs/gigpo_full_20261027/manifest.json) |
| 教师原始轨迹 | `runs/teacher/opencode_v41_flash_20261026_native_v2/episodes`（本地，不随仓库分发） |
| GRPO 测试原始轨迹 | [episodes](../../runs/eval/qwen35_2b_grpo_vllm_test_20261027/episodes) |
| GiGPO 测试原始轨迹 | [episodes](../../runs/eval/qwen35_2b_gigpo_vllm_test_20261027/episodes) |

本报告是对已有实验记录的整理，没有为得到更好结论额外训练、重采样测试题或改写原始轨迹。

GitHub 分发说明：仓库保留测试轨迹、统计和证据，不包含训练权重、完整教师采集或训练 rollout；历史 manifest 的绝对路径用于审计。发布时增加的数据路径迁移只在原文件不存在时生效，仍核验原始任务哈希，未重算任何历史评测。
