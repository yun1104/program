#!/usr/bin/env python3
"""Disjoint vLLM replicas, preserving every completed episode from an interrupted run."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import (digest, fingerprint, load_config, project_path,
    provenance, read_json, read_rows, write_json)
from scripts.toolmaze_agent.eval import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,3,5,6")
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/baseline_vllm.yaml")
    parser.add_argument("--shard-dir", default="shards")
    parser.add_argument("--allow-runtime-tuning", action="store_true")
    parser.add_argument("--shard-order-hash", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != len(set(gpus)):
        parser.error("GPU IDs must be distinct")
    config = load_config(args.config)
    project = load_config(config["project_config"])
    run = project_path(config["output"])
    run.mkdir(parents=True, exist_ok=True)
    split_path = project_path(project["splits"]) / f"{config['split']}.jsonl"
    rows = read_rows(split_path)
    prior = read_json(run / "manifest.json") if (run / "manifest.json").exists() else None
    current = provenance(config)
    if prior:
        for key in ("generation", "model", "seed", "trials", "split"):
            if prior["config"][key] != config[key]:
                raise ValueError(f"Cannot reuse episodes with a different {key}")
        if prior["split_sha256"] != digest(split_path) or prior["project_config"] != project:
            raise ValueError("Dataset or environment protocol changed")
        # Partitioning/logging can change, but inputs, runtime, and generation semantics cannot.
        checked = ("protocol.py", "rollout.py", "environment.py")
        if not args.allow_runtime_tuning:
            checked += ("vllm_backend.py",)
        for name in checked:
            key = f"scripts/toolmaze_agent/{name}"
            if prior["source_sha256"][key] != current["source_sha256"][key]:
                raise ValueError(f"Cannot reuse episodes after changing {name}")
        if prior["model_snapshot"] != read_json(project_path(config["model_manifest"])):
            raise ValueError("Model snapshot changed")
        if not (run / "manifest.initial_single_gpu.json").exists():
            write_json(run / "manifest.initial_single_gpu.json", prior)
    expected = {(row["id"], trial) for row in rows for trial in range(config["trials"])}
    inherited = []
    processes, manifests = [], []
    partition_rows = sorted(rows, key=lambda row: fingerprint([config["seed"], "partition-v2", row["id"]])) if args.shard_order_hash else rows
    shard_root = run / args.shard_dir
    for index, gpu in enumerate(gpus):
        shard = shard_root / str(index)
        for row in partition_rows[index::len(gpus)]:
            for trial in range(config["trials"]):
                name = f"{row['task_id']}_{row['mode']}_{trial:02d}.json"
                source, target = run / "episodes" / name, shard / "episodes" / name
                if source.exists() and not target.exists():
                    record = read_json(source)
                    if record["row"] != row or record["trial"] != trial:
                        raise ValueError("Saved episode does not match frozen task")
                    write_json(target, record)
                    inherited.append([row["id"], trial])
        command = [sys.executable, "-u", str(ROOT / "tools/run_baseline.py"), "--gpu", gpu,
                   "--config", args.config, "--output", str(shard), "--split", config["split"],
                   "--shard-count", str(len(gpus)), "--shard-index", str(index),
                   "--log", f"reports/baseline/qwen35_2b_vllm_test_{args.shard_dir}_gpu{gpu}.log"]
        if args.shard_order_hash:
            command += ["--shard-order-hash"]
        processes.append(subprocess.Popen(command, cwd=ROOT))
    write_json(run / "parallel_execution.json", {"gpus": gpus, "shards": len(gpus),
               "inherited_completed_episodes": inherited, "prior_identity": prior.get("identity") if prior else None,
               "shard_dir": args.shard_dir, "runtime_tuning_authorized": args.allow_runtime_tuning,
               "policy": "Reuse every completed result; no success-based selection; seeds depend on task/trial/step only"})
    started = time.monotonic()
    while any(process.poll() is None for process in processes):
        records = [read_json(path) for path in sorted(shard_root.glob("*/episodes/*.json"))]
        keys = [(r["row"]["id"], r["trial"]) for r in records]
        if len(keys) != len(set(keys)) or not set(keys) <= expected:
            raise RuntimeError("Duplicate or unexpected partition results")
        progress = {"completed": len(records), "total": len(expected),
                    "elapsed_seconds": time.monotonic() - started,
                    "official_pass": sum(r.get("official_pass", False) for r in records),
                    "strict_pass": sum(r.get("strict_pass", False) for r in records),
                    "infrastructure_errors": sum(bool(r.get("error")) for r in records),
                    "replicas": len(gpus), "reused_completed": len(inherited)}
        write_json(run / "progress.json", progress)
        print(json.dumps(progress), flush=True)
        time.sleep(10)
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"Replica exit codes: {codes}; partial results are retained")
    records = []
    for index, gpu in enumerate(gpus):
        shard = shard_root / str(index)
        manifest = read_json(shard / "manifest.json")
        manifests.append({"gpu": gpu, "identity": manifest["identity"], "requested_ids": manifest["requested_ids"]})
        for path in sorted((shard / "episodes").glob("*.json")):
            record = read_json(path)
            records.append(record)
            write_json(run / "episodes" / path.name, record)
    keys = [(r["row"]["id"], r["trial"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("Partition union does not equal the full frozen test set")
    manifest = {**current, "project_config": project, "split_sha256": digest(split_path),
                "requested_ids": [r["id"] for r in rows], "domain": None,
                "model_snapshot": read_json(project_path(config["model_manifest"])),
                "runtime_environment": {"CUDA_VISIBLE_DEVICES": gpus}, "partitions": manifests,
                "inherited_completed_episodes": inherited}
    manifest["identity"] = fingerprint(manifest)
    write_json(run / "manifest.json", manifest)
    report = summarize(records, project)
    write_json(run / "summary.json", report)
    write_json(run / "progress.json", {"completed": len(records), "total": len(expected), "finished": True,
               "replicas": len(gpus), "reused_completed": len(inherited),
               "official_pass": sum(r.get("official_pass", False) for r in records),
               "strict_pass": sum(r.get("strict_pass", False) for r in records),
               "infrastructure_errors": sum(bool(r.get("error")) for r in records),
               "elapsed_seconds": time.monotonic() - started})
    print(report["breakdown"]["all"], flush=True)


if __name__ == "__main__":
    main()
