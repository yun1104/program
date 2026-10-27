# 暂时性错误：SFT 重复操作，GRPO 重试后完成

任务：`C2_task_100:P1`；每个模型为该题唯一一次固定 seed 测评。

## 观察与解释

两者第一次 set_light_state 均收到 Gateway Timeout。GRPO 用相同语义参数重试成功，再调用 apply_lock_state、create_document_and_alert 并提交最终回答。SFT 也曾重试成功，但随后重复设置灯和锁，还把 lock_main/state=lock 传给只支持电源开关的 set_power_state，产生 schema 错误，最终达到 16 轮上限。该例不是“SFT 完全不会重试”，而是展示恢复后的任务推进与协议约束差异。

**可讲的结论：** 区分 API 暂时失败与模型参数错误；观察 RL 后重复操作与 schema 错误减少的具体表现。

**边界与反例：** 这是事后选取的一组说明性轨迹，不能用单例替代总体通过率。

## SFT

官方通过：`False`；严格通过：`False`；工具调用：16；协议错误：0；schema 错误：1；终止：`max_rounds`。

[完整原始快照](sft.json) · [运行原文件](../../../../runs/eval/qwen35_2b_sft_vllm_test_20261026/episodes/C2_task_100_P1_00.json)

SHA-256：`b7086d2053a5a2319bbfe1ddd2de620e5157bb51b73eb0e3c76a2b5f2768cd10`

### 用户请求原文

````text
For user_123, please scan the living_room for smart devices, then check the status of the living room light with ID light_living and make sure its target state is on at 30% brightness. After that, check the status of the main lock with ID lock_main and set its target state to lock using PIN 2026. Once everything is done, send me a notification confirming it was executed. You can use the title Untitled in the confirmation.
````

### 实际工具调用与反馈

| 工具调用序号 | 工具 | 参数（缩略） | observation（缩略） | 审计标记（策略不可见） |
|---:|---|---|---|---|
| 1 | `discover_smart_devices` | <code>{&quot;location&quot;:&quot;living_room&quot;}</code> | <code>{&quot;location&quot;:&quot;living_room&quot;,&quot;device_count&quot;:4,&quot;devices&quot;:[{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;name&quot;:&quot;Main Door Lock&quot;,&quot;type&quot;:&quot;smart_lock&quot;},{&quot;device_id&quot;:&quot;light_living&quot;,&quot;name&quot;:&quot;Living Room Main Light&quot;,&quot;type&quot;:&quot;smart_light&quot;},{&quot;device_id&quot;:&quot;ac_living&quot;,&quot;name&quot;:&quot;L…【截断，见 JSON】</code> | clean |
| 2 | `get_light_status` | <code>{&quot;light_id&quot;:&quot;light_living&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;power_state&quot;:&quot;off&quot;,&quot;brightness&quot;:0,&quot;color&quot;:&quot;white&quot;}</code> | clean |
| 3 | `get_iot_device_status` | <code>{&quot;device_id&quot;:&quot;lock_main&quot;}</code> | <code>{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;online_status&quot;:&quot;online&quot;,&quot;power_state&quot;:&quot;on&quot;,&quot;locked&quot;:true,&quot;battery_level&quot;:85}</code> | clean |
| 4 | `set_light_state` | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;,&quot;brightness&quot;:30}</code> | <code>{&quot;error&quot;:&quot;Gateway Timeout&quot;}</code> | perturbed |
| 5 | `get_light_status` | <code>{&quot;light_id&quot;:&quot;light_living&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;power_state&quot;:&quot;off&quot;,&quot;brightness&quot;:0,&quot;color&quot;:&quot;white&quot;}</code> | clean |
| 6 | `set_light_state` | <code>{&quot;brightness&quot;:30,&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;control_smart_light&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;,&quot;brightness&quot;:30,&quot;color&quot;:&quot;white&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 7 | `set_power_state` | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;toggle_device_power&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 8 | `set_light_state` | <code>{&quot;brightness&quot;:30,&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;control_smart_light&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;,&quot;brightness&quot;:30,&quot;color&quot;:&quot;white&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 9 | `set_power_state` | <code>{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;state&quot;:&quot;lock&quot;}</code> | <code>{&quot;error&quot;:&quot;Validation failed&quot;,&quot;details&quot;:[&quot;&#x27;lock_main&#x27; is not one of [&#x27;light_living&#x27;, &#x27;light_bedroom&#x27;, &#x27;ac_living&#x27;, &#x27;ac_bedroom&#x27;, &#x27;tv_living&#x27;, &#x27;tv_bedroom&#x27;]&quot;,&quot;&#x27;lock&#x27; is not one of [&#x27;on&#x27;, &#x27;off&#x27;]&quot;]}</code> | clean |
| 10 | `apply_lock_state` | <code>{&quot;lock_id&quot;:&quot;lock_main&quot;,&quot;state&quot;:&quot;lock&quot;,&quot;pin_code&quot;:&quot;2026&quot;}</code> | <code>{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;new_state&quot;:&quot;locked&quot;,&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 11 | `set_light_state` | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;brightness&quot;:30,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;control_smart_light&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;,&quot;brightness&quot;:30,&quot;color&quot;:&quot;white&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 12 | `apply_lock_state` | <code>{&quot;lock_id&quot;:&quot;lock_main&quot;,&quot;state&quot;:&quot;lock&quot;,&quot;pin_code&quot;:&quot;2026&quot;}</code> | <code>{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;new_state&quot;:&quot;locked&quot;,&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 13 | `set_light_state` | <code>{&quot;brightness&quot;:30,&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;control_smart_light&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;,&quot;brightness&quot;:30,&quot;color&quot;:&quot;white&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 14 | `apply_lock_state` | <code>{&quot;lock_id&quot;:&quot;lock_main&quot;,&quot;state&quot;:&quot;lock&quot;,&quot;pin_code&quot;:&quot;2026&quot;}</code> | <code>{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;new_state&quot;:&quot;locked&quot;,&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 15 | `set_light_state` | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;brightness&quot;:30,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;control_smart_light&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;,&quot;brightness&quot;:30,&quot;color&quot;:&quot;white&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 16 | `set_power_state` | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;toggle_device_power&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |

### 最终回答原文

````text
【该轨迹未提交 final_answer】
````

### 官方失败原因

````text
P1 failed: missing shared tools: ['create_document_and_alert']
````

## GRPO

官方通过：`True`；严格通过：`True`；工具调用：7；协议错误：0；schema 错误：0；终止：`final_answer`。

[完整原始快照](grpo.json) · [运行原文件](../../../../runs/eval/qwen35_2b_grpo_vllm_test_20261027/episodes/C2_task_100_P1_00.json)

SHA-256：`2c896198a3fb2937c682829453745eee728bd5dece14b09240541647c64a2b86`

### 用户请求原文

````text
For user_123, please scan the living_room for smart devices, then check the status of the living room light with ID light_living and make sure its target state is on at 30% brightness. After that, check the status of the main lock with ID lock_main and set its target state to lock using PIN 2026. Once everything is done, send me a notification confirming it was executed. You can use the title Untitled in the confirmation.
````

### 实际工具调用与反馈

| 工具调用序号 | 工具 | 参数（缩略） | observation（缩略） | 审计标记（策略不可见） |
|---:|---|---|---|---|
| 1 | `discover_smart_devices` | <code>{&quot;location&quot;:&quot;living_room&quot;}</code> | <code>{&quot;location&quot;:&quot;living_room&quot;,&quot;device_count&quot;:4,&quot;devices&quot;:[{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;name&quot;:&quot;Main Door Lock&quot;,&quot;type&quot;:&quot;smart_lock&quot;},{&quot;device_id&quot;:&quot;light_living&quot;,&quot;name&quot;:&quot;Living Room Main Light&quot;,&quot;type&quot;:&quot;smart_light&quot;},{&quot;device_id&quot;:&quot;ac_living&quot;,&quot;name&quot;:&quot;L…【截断，见 JSON】</code> | clean |
| 2 | `get_light_status` | <code>{&quot;light_id&quot;:&quot;light_living&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;power_state&quot;:&quot;off&quot;,&quot;brightness&quot;:0,&quot;color&quot;:&quot;white&quot;}</code> | clean |
| 3 | `get_iot_device_status` | <code>{&quot;device_id&quot;:&quot;lock_main&quot;}</code> | <code>{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;online_status&quot;:&quot;online&quot;,&quot;power_state&quot;:&quot;on&quot;,&quot;locked&quot;:true,&quot;battery_level&quot;:85}</code> | clean |
| 4 | `set_light_state` | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;,&quot;brightness&quot;:30}</code> | <code>{&quot;error&quot;:&quot;Gateway Timeout&quot;}</code> | perturbed |
| 5 | `set_light_state` | <code>{&quot;brightness&quot;:30,&quot;device_id&quot;:&quot;light_living&quot;,&quot;state&quot;:&quot;on&quot;}</code> | <code>{&quot;device_id&quot;:&quot;light_living&quot;,&quot;operation&quot;:&quot;control_smart_light&quot;,&quot;changes&quot;:{&quot;power_state&quot;:&quot;on&quot;,&quot;brightness&quot;:30,&quot;color&quot;:&quot;white&quot;},&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 6 | `apply_lock_state` | <code>{&quot;lock_id&quot;:&quot;lock_main&quot;,&quot;state&quot;:&quot;lock&quot;,&quot;pin_code&quot;:&quot;2026&quot;}</code> | <code>{&quot;device_id&quot;:&quot;lock_main&quot;,&quot;new_state&quot;:&quot;locked&quot;,&quot;status&quot;:&quot;executed&quot;}</code> | clean |
| 7 | `create_document_and_alert` | <code>{&quot;title&quot;:&quot;Untitled&quot;,&quot;content&quot;:&quot;Living room smart home automation executed successfully for user_123.\n\n**Summary:**\n- **Living room scan** (4 devices found): lock_main, light_living, ac_li…【截断，见 JSON】</code> | <code>{&quot;message&quot;:&quot;Notification sent successfully&quot;,&quot;notification_id&quot;:&quot;NOTIFEE84C&quot;,&quot;user_id&quot;:&quot;user_123&quot;,&quot;content&quot;:&quot;Living room smart home automation executed successfully for user_123.\n\n**Summary:**\n- **Living room scan** (4 devices found): lock…【截断，见 JSON】</code> | clean |

### 最终回答原文

````text
All four tasks completed successfully:

1. **Living room scan** — 4 devices found: `lock_main` (main door lock), `light_living` (living room main light), `ac_living` (thermostat), `tv_living` (media player).

2. **Light status** — `light_living` was off at 0% brightness. It was re-asked to be on at 30% brightness; the first attempt returned a Gateway Timeout, but the retry succeeded and the light is now on at 30% white.

3. **Main lock** — `lock_main` was already locked, so the target state of "lock" was confirmed executed. (Note: the lock is online and powered on with 85% battery.)

4. **Notification** — Sent to `user_123` (notification ID `NOTIFEE84C`, note ID `NOTE18E3D`, title "Untitled", medium priority, app channel) confirming all steps.
````

### 官方失败原因

````text
None
````
