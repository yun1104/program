# Full ToolMaze GiGPO run

Starts from `runs/sft_full_trl_20261026/final`, not from the smoke adapter or
the GRPO adapter. The frozen train split contains 1,600 rows across P0--P4.
Validation and test are not used for training.

* Training GPUs: 0, 3, 4, 5, 6. Dedicated vLLM sampler: GPU 7.
* Language LoRA rank 64, alpha 128; frozen SFT base and vision parameters.
* 8 trajectories/task; 160 trajectories/update = 20 tasks/update.
* Per-device microbatch 2, accumulation 16, world size 5; 80 updates cover
  1,600 tasks once, producing 12,800 trajectories.
* LR 1e-5, warmup 5, cosine, KL beta .01, 8,192-token context,
  maximum 512 generated tokens/action and 16 environment decisions/episode.
* GiGPO gamma .95, omega 1; the same terminal reward as GRPO, no PRM or
  extra rollouts. Anchors are exact policy token prefixes within a task group.
* Global rank-ordered trajectory gathering precedes advantage assignment.
  Global per-action advantages are sliced back to each rank and applied only
  to model tokens. Environment tokens, runtime EOS and padding are excluded.
* Five-GPU DeepSpeed ZeRO-2; custom checkpointed output-head token loss.
* Native adapter/optimizer/RNG checkpoints every update; latest two retained.

The GRPO baseline used 128 trajectories/update and 100 updates. This run uses
all currently idle GPUs, so it has a larger batch and fewer updates, with the
same 12,800-trajectory budget. Interpret an eventual comparison with this
batch-size difference in mind.

## Launch and progress

```bash
python tools/run_gigpo_full.py --gpus 0,3,4,5,6 --sampler-gpu 7
watch -n 5 cat runs/gigpo_full_20261027/live_progress.json
```

Run output: `runs/gigpo_full_20261027/`.
Logs/launch receipt/source snapshots: `reports/gigpo/gigpo_full_20261027/`.
Configuration: `scripts/toolmaze_agent/configs/gigpo_full.yaml`.
Training entry point: `scripts/toolmaze_agent/train_gigpo_full.py`.

`advantages/*.json` contains global action credits, grouping coverage and rank
sample counts. `rollouts/*.json` contains all sampled outcomes, including
failures, with reconstructed action spans/anchors. `adapter_progress.json`
records actual parameter change. The independent live monitor refreshes every
5 seconds; its ETA is approximate and initially includes cold-start costs.

Resume with the same world size and configuration:

```bash
python tools/run_gigpo_full.py --gpus 0,3,4,5,6 --sampler-gpu 7 \
  --resume runs/gigpo_full_20261027/checkpoint-N
```

All original GRPO and single-GPU smoke source/config/output files are retained.
Only new distributed-run files are added. `unit_tests.log` records the eight
passed tests, including a task group split across simulated rank boundaries.
