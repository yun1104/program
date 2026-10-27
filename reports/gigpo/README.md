# Independent GiGPO experiment

The existing GRPO modules, configuration and running job are not modified.

## Launch

```bash
cd /home/ruanwenxuan/program
/home/ruanwenxuan/miniconda3/envs/sftrl/bin/python tools/run_gigpo.py --gpu 4
```

Configuration: `scripts/toolmaze_agent/configs/gigpo_smoke.yaml`.
The default smoke run uses two frozen **train** tasks, four samples per task
group, three optimizer updates, and rank-64 LoRA starting from the completed SFT
checkpoint. This is an integration test, not a benchmark or evidence of gains.

## Algorithm

Reference: Feng et al., **Group-in-Group Policy Optimization for LLM Agent
Training**, NeurIPS 2025, equations 5, 7, 8:
https://papers.neurips.cc/paper_files/paper/2025/file/420c9f777c0b4f78d515e53cf74d58b2-Paper-Conference.pdf

* Episode advantage: task-group standardized terminal reward.
* Action return: `gamma ** remaining_decisions * terminal_reward`.
* Step advantage: standardized action return among matching anchor states.
* Combined advantage: `episode_advantage + omega * step_advantage`.
* Every sampled token of an action receives that action's advantage. Environment
  tokens, padding and runtime-inserted EOS have no policy loss or KL loss.
* No PRM, critic, teacher calls or extra branch rollouts. Terminal reward is
  reused unchanged from the original pipeline, including cost/error penalties.
* Normalization uses sample standard deviation plus `1e-4`, matching our TRL
  baseline. Singleton and constant-return anchor groups contribute zero step
  advantage; a singleton therefore falls back to episode-only credit.

The conservative anchor is the SHA256 of **training row identity plus the exact
full policy input token prefix**, restricted to the current task-group batch.
No ground-truth paths, future outcome labels or hidden fault labels are used in
the input. This deliberately avoids approximate state merging. It can produce
few noninitial shared states. Inspect `advantages/*.json` for coverage and
nonzero noninitial step advantages before running a larger experiment.

## Implementation and limitations

New modules: `gigpo.py`, `gigpo_rollout.py`, `gigpo_trainer.py`,
`train_gigpo.py`. New launcher: `tools/run_gigpo.py`.

One GPU hosts vLLM and the actor. TRL synchronizes stateless merged LoRA weights
at each rollout batch. Batched round-wise vLLM generation executes actual
ToolMaze calls between turns; vLLM sleeps at level 2 before scoring/backprop.
The training stack is TRL 1.13.0 + DeepSpeed ZeRO-2 + PEFT. The installed Liger
GRPO loss only accepts one advantage per trajectory, so we replace its loss
dispatch with a checkpointed, token-chunked output-head loss that supports
per-action advantages. We retain bounded-memory old/reference scoring from the
existing read-only superclass. This version explicitly requires world size 1.

The clipped surrogate operates on token likelihood ratios, as in the existing
TRL baseline. This is an adaptation to that training stack, not a claim of an
identical reproduction of every setting in the official verl-agent code.

For a controlled episode-only ablation, copy the smoke configuration to a new
file, choose a fresh output directory, and set `gigpo.omega: 0`. Keep sampling,
seed, terminal reward and all other settings identical. This removes the step
advantage while retaining the same loss implementation and execution stack.

Artifacts:

* `runs/gigpo_smoke_20261026/rollouts/`: complete actual model/environment traces.
* `runs/gigpo_smoke_20261026/advantages/`: action spans, returns, both advantage
  components, anchor sizes and coverage statistics for each rollout batch.
* `runs/gigpo_smoke_20261026/adapter_progress.json`: evidence of parameter update.
* `runs/gigpo_smoke_20261026/checkpoint-*/`: optimizer/RNG/adapter state.
* `reports/gigpo/gigpo_smoke_20261026/{launch.json,train.log}`: launch receipt,
  GPU assignment, original source hashes and log.
* `reports/gigpo/gigpo_smoke_20261026/smoke_summary.json`: reviewable validation
  receipt, including action diversity within shared states and final adapter
  finite-value checks. Refresh using `python tools/summarize_gigpo.py`.

Native restore (after the selected GPU is free):

```bash
python tools/run_gigpo.py --gpu 4 --resume runs/gigpo_smoke_20261026/checkpoint-2
```

Unit verification:

```bash
python -m unittest scripts.toolmaze_agent.tests.test_gigpo -v
```
