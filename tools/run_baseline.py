#!/usr/bin/env python3
"""Run the frozen Qwen3.5-2B vLLM baseline on explicitly selected GPU(s)."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Explicit CUDA_VISIBLE_DEVICES, e.g. 0")
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/baseline_vllm.yaml")
    parser.add_argument("--log", default="reports/baseline/qwen35_2b_vllm_test.log")
    parser.add_argument("--output")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-order-hash", action="store_true")
    args = parser.parse_args()
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES=args.gpu, VLLM_USE_FLASHINFER_SAMPLER="0",
                       VLLM_WORKER_MULTIPROC_METHOD="spawn", TOKENIZERS_PARALLELISM="false",
                       OMP_NUM_THREADS="4", VLLM_CACHE_ROOT=str(ROOT / ".cache/vllm"))
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    toolkit = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13"
    if (toolkit / "bin/nvcc").is_file():
        environment["CUDA_HOME"] = str(toolkit)
        environment["PATH"] = str(toolkit / "bin") + os.pathsep + environment["PATH"]
    command = [sys.executable, "-u", "-m", "scripts.toolmaze_agent.eval", "--config", args.config,
               "--split", args.split, "--shard-count", str(args.shard_count),
               "--shard-index", str(args.shard_index)]
    if args.output:
        command += ["--output", args.output]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    if args.shard_order_hash:
        command += ["--shard-order-hash"]
    log = ROOT / args.log
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Running baseline on CUDA_VISIBLE_DEVICES={args.gpu}; log: {log}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
