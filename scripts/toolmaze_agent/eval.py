"""Fixed-denominator evaluation with official metrics and separate strict checks."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import os

from .common import digest, fingerprint, load_config, project_path, provenance, read_json, read_rows, write_json
from .environment import upstream_modules
from .policy import APIBackend, HFBackend, load_student
from .prepare import load_task
from .rollout import run_episode


def summarize(records, project):
    _, _, Metrics = upstream_modules(project["upstream"])
    metrics = Metrics()
    cells = defaultdict(list)
    for record in records:
        row = record["row"]
        task = load_task(row)
        if record.get("error"):
            trace = {"messages": [], "tokens": {"total_tokens": 0}}
            judgement = {"pass": False}
        else:
            trace, judgement = record["trace"], record["judgement"]
        metrics.add_result(task, trace, judgement)
        for cell in ("all", f"mode/{row['mode']}", f"complexity/{row['complexity']}",
                     f"cell/{row['complexity']}/{row['mode']}"):
            cells[cell].append(record)
        if "Travel" in row.get("domains", []):
            cells["domain/Travel"].append(record)
    summary = {}
    for key, values in sorted(cells.items()):
        n = len(values)
        summary[key] = {"n": n, "official_pass_rate": sum(v.get("official_pass", False) for v in values) / n,
                        "strict_pass_rate": sum(v.get("strict_pass", False) for v in values) / n,
                        "infrastructure_errors": sum(bool(v.get("error")) for v in values),
                        "mean_tool_calls": sum(v.get("tool_calls", 0) for v in values) / n,
                        "mean_tokens": sum(v.get("tokens", {}).get("total_tokens", 0) for v in values) / n}
    return {"official": metrics.generate_report(), "breakdown": summary,
            "denominator": "All requested task/trial pairs, including infrastructure failures",
            "metric_version": "Pinned ToolMaze commit; PRR is per unique fault hit; RC is exposure-conditioned",
            "strict_pass": "Official pass AND explicit final answer AND no protocol/schema errors; not a complete semantic validator"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/eval.yaml")
    parser.add_argument("--model")
    parser.add_argument("--output")
    parser.add_argument("--split", choices=["validation", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--domain")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-order-hash", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard-count)")
    config = load_config(args.config)
    for key in ("model", "output", "split", "trials"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    project = load_config(config["project_config"])
    split_path = project_path(project["splits"]) / f"{config['split']}.jsonl"
    rows = read_rows(split_path)
    if args.domain:
        rows = [r for r in rows if args.domain.casefold() in {d.casefold() for d in r.get("domains", [])}]
    if args.limit:
        rows = rows[:args.limit]
    if args.shard_order_hash:
        rows.sort(key=lambda row: fingerprint([config["seed"], "partition-v2", row["id"]]))
    rows = rows[args.shard_index::args.shard_count]
    if not rows or config["trials"] < 1:
        raise ValueError("Evaluation needs tasks and positive trials")
    output = project_path(config["output"])
    manifest = {**provenance(config), "project_config": project, "split_sha256": digest(split_path),
                "requested_ids": [r["id"] for r in rows], "domain": args.domain}
    manifest["partition"] = {"index": args.shard_index, "count": args.shard_count,
                             "hash_order": args.shard_order_hash}
    manifest["runtime_environment"] = {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "CUDA_HOME", "VLLM_USE_FLASHINFER_SAMPLER",
        "VLLM_CACHE_ROOT", "OMP_NUM_THREADS")}
    if config.get("model_manifest"):
        manifest["model_snapshot"] = read_json(project_path(config["model_manifest"]))
    manifest["identity"] = fingerprint(manifest)
    if (output / "manifest.json").exists() and read_json(output / "manifest.json")["identity"] != manifest["identity"]:
        raise ValueError("Evaluation protocol changed; choose a new output directory")
    write_json(output / "manifest.json", manifest)
    if config["backend"] == "api":
        backend = APIBackend({**project["teacher"], **config["generation"],
                              "model": config["model"], **config.get("api", {})})
    elif config["backend"] == "hf":
        import torch
        from transformers import set_seed
        if not torch.cuda.is_available():
            raise RuntimeError("HF evaluation requires a visible GPU; CPU smoke tests use tools/smoke.py")
        model_path = str(project_path(config["model"])) if project_path(config["model"]).exists() else config["model"]
        model, tokenizer, _ = load_student(model_path, config["dtype"])
        model.to("cuda")
        backend = HFBackend(model, tokenizer, config["generation"])
    elif config["backend"] == "vllm":
        from .vllm_backend import VLLMBackend
        upstream_modules(project["upstream"])  # Import upstream before starting environment workers.
        if config["vllm"].get("schema_cache"):
            from .tool_schema_cache import enable_schema_cache
            enable_schema_cache(project_path(project["upstream"]))
        backend = VLLMBackend(str(project_path(config["model"]).resolve()), config["generation"],
                              config["vllm"], config["seed"])
    else:
        raise ValueError("backend must be hf, api or vllm")
    records = []
    lock = threading.Lock()
    started = time.monotonic()

    def evaluate_one(row, trial):
        path = output / "episodes" / f"{row['task_id']}_{row['mode']}_{trial:02d}.json"
        if path.exists():
            record = read_json(path)
        else:
            if config["backend"] == "hf":
                seed = int(fingerprint([config["seed"], row["id"], trial])[:8], 16)
                set_seed(seed)
            try:
                policy = backend.session([row["id"], trial]) if config["backend"] == "vllm" else backend
                record = {"row": row, "trial": trial, **run_episode(load_task(row), policy, project)}
            except Exception as error:
                record = {"row": row, "trial": trial, "error": f"{type(error).__name__}: {error}",
                          "official_pass": False, "strict_pass": False}
            write_json(path, record)
        with lock:
            records.append(record)
            write_json(output / "progress.json", {"completed": len(records),
                       "total": len(rows) * config["trials"], "elapsed_seconds": time.monotonic() - started,
                       "official_pass": sum(r.get("official_pass", False) for r in records),
                       "strict_pass": sum(r.get("strict_pass", False) for r in records),
                       "infrastructure_errors": sum(bool(r.get("error")) for r in records)})
            print(f"[{len(records)}/{len(rows) * config['trials']}]", row["id"], trial,
                  record.get("strict_pass"), record.get("error", ""), flush=True)

    if config["backend"] == "vllm":
        with ThreadPoolExecutor(max_workers=config.get("workers", 16)) as executor:
            futures = [executor.submit(evaluate_one, row, trial) for row in rows for trial in range(config["trials"])]
            backend.dispatch_until_done(futures)
            for future in futures:
                future.result()
    else:
        for row in rows:
            for trial in range(config["trials"]):
                evaluate_one(row, trial)
    records.sort(key=lambda record: (record["row"]["id"], record["trial"]))
    report = summarize(records, project)
    write_json(output / "summary.json", report)
    print(report["breakdown"]["all"])


if __name__ == "__main__":
    main()
