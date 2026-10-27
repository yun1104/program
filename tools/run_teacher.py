#!/usr/bin/env python3
"""Start a bounded, resumable OpenCode teacher collection without exposing credentials."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/teacher_opencode.yaml")
    parser.add_argument("--output", default="runs/teacher/opencode_v41_flash_20261026_native_v2")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log", default="reports/teacher/opencode_v41_flash_collection.log")
    args = parser.parse_args()
    command = [sys.executable, "-u", "-m", "scripts.toolmaze_agent.collect", "--config", args.config,
               "--split", "train", "--output", args.output, "--workers", str(args.workers),
               "--attempts", str(args.attempts)]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    log = ROOT / args.log
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"Teacher collection log: {log}; credentials read internally from configured file", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        process = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
