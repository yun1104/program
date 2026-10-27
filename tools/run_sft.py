#!/usr/bin/env python3
"""Launch TRL SFTTrainer with DeepSpeed on explicitly selected idle GPUs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Explicit physical CUDA indices in PCI bus order")
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/sft_full_trl.yaml")
    parser.add_argument("--log", default="reports/sft/full_trl_20261026.log")
    parser.add_argument("--resume")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if len(set(gpus)) != len(gpus):
        parser.error("GPU indices must be unique")
    environment = dict(os.environ)
    extra = [str(ROOT / ".deps/sft_fast"), str(ROOT / ".deps/causal_conv")]
    environment.update(CUDA_VISIBLE_DEVICES=args.gpus, CUDA_DEVICE_ORDER="PCI_BUS_ID", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="4", MAX_JOBS="4", TRITON_CACHE_DIR=str(ROOT / ".cache/triton_sft"),
                       TORCH_EXTENSIONS_DIR=str(ROOT / ".cache/torch_extensions"))
    environment.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    environment["PYTHONPATH"] = os.pathsep.join(extra + [str(ROOT), environment.get("PYTHONPATH", "")])
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    toolkit = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    if (toolkit / "bin/nvcc").exists():
        environment["CUDA_HOME"] = str(toolkit)
        environment["PATH"] = str(toolkit / "bin") + os.pathsep + environment["PATH"]
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
               f"--nproc_per_node={len(gpus)}", "-m", "scripts.toolmaze_agent.train_sft", "--config", args.config]
    if args.resume:
        command += ["--resume", args.resume]
    log = ROOT / args.log
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Starting DeepSpeed + TRL full SFT on {args.gpus}; log={log}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
