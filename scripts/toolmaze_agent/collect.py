"""Resumable, bounded-concurrency API trajectory collection on a frozen fold."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from collections import Counter
import time

from .common import digest, fingerprint, load_config, project_path, provenance, read_json, read_rows, write_json
from .policy import APIBackend
from .prepare import load_task
from .rollout import run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/project.yaml")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--output", default="runs/teacher")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--modes", nargs="+", choices=["P0", "P1", "P2", "P3", "P4"])
    parser.add_argument("--max-infrastructure-errors", type=int, default=5)
    args = parser.parse_args()
    if args.attempts < 1 or args.workers < 1:
        parser.error("attempts and workers must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if args.max_infrastructure_errors < 1:
        parser.error("max-infrastructure-errors must be positive")
    config = load_config(args.config)
    if config.get("schema_cache"):
        from .environment import upstream_modules
        from .tool_schema_cache import enable_schema_cache
        upstream_modules(config["upstream"])
        enable_schema_cache(project_path(config["upstream"]))
    split_path = project_path(config["splits"]) / f"{args.split}.jsonl"
    rows = read_rows(split_path)
    if args.modes:
        rows = [row for row in rows if row["mode"] in args.modes]
    if args.limit:
        rows = rows[:args.limit]
    output = project_path(args.output)
    manifest = {**provenance(config), "split": args.split, "split_sha256": digest(split_path),
                "protocol": "json-action-v1", "teacher": config["teacher"]}
    identity = fingerprint(manifest)
    manifest["identity"] = identity
    if (output / "manifest.json").exists():
        if read_json(output / "manifest.json")["identity"] != identity:
            raise ValueError("Collection config/code/data changed; choose a new output directory")
    else:
        write_json(output / "manifest.json", manifest)
    pending = []
    for row in rows:
        for attempt in range(args.attempts):
            path = output / "episodes" / f"{row['task_id']}_{row['mode']}_{attempt:02d}.json"
            if path.exists():
                saved = read_json(path)
                if saved.get("error"):
                    pending.append((row, attempt, path))
                continue
            pending.append((row, attempt, path))

    def run(item):
        row, attempt, path = item
        backend = None
        try:
            backend = APIBackend({**config["teacher"],
                                  "session_id": "toolmaze-" + fingerprint([str(output), row["id"], attempt])[:32]})
            episode = run_episode(load_task(row), backend, config)
            result = {"row": row, "attempt": attempt, "split": args.split,
                      "resolved_models": sorted(backend.resolved_models), **episode}
            if hasattr(backend, "native_messages"):
                result["teacher_conversation"] = backend.native_messages
                result["teacher_tools"] = backend.native_tools
                result["teacher_action_protocol"] = "Anthropic native tool_use -> student json-action-v1"
        except Exception as error:
            # Failed infrastructure calls are journaled, not mislabeled as valid training examples.
            result = {"row": row, "attempt": attempt, "split": args.split,
                      "error": f"{type(error).__name__}: {error}", "strict_pass": False}
            if backend is not None and hasattr(backend, "native_messages"):
                result["teacher_conversation"] = backend.native_messages
        finally:
            if backend is not None and hasattr(backend, "close"):
                backend.close()
        write_json(path, result)
        return row["id"], result.get("strict_pass", False), result.get("error")

    # Fail before spawning workers when the key/config is missing.
    check = APIBackend(config["teacher"])
    if hasattr(check, "close"):
        check.close()
    started = time.monotonic()
    stopped = False
    existing = [read_json(p) for p in sorted((output / "episodes").glob("*.json"))]
    journal = {(e["row"]["id"], e["attempt"]): e for e in existing}

    def progress(finished=False):
        episodes = list(journal.values())
        summary = {"state": "stopped_on_infrastructure_errors" if stopped else "complete" if finished else "running",
                   "planned": len(rows) * args.attempts, "total": len(episodes),
                   "accepted": sum(e.get("strict_pass", False) for e in episodes),
                   "infrastructure_errors": sum(bool(e.get("error")) for e in episodes),
                   "accepted_by_mode": dict(Counter(e["row"]["mode"] for e in episodes if e.get("strict_pass"))),
                   "input_tokens": sum(e.get("tokens", {}).get("input_tokens", 0) for e in episodes),
                   "output_tokens": sum(e.get("tokens", {}).get("output_tokens", 0) for e in episodes),
                   "workers": args.workers, "attempts": args.attempts,
                   "elapsed_seconds": time.monotonic() - started}
        write_json(output / "progress.json", summary)
        write_json(output / "summary.json", summary)
        return summary

    progress()
    consecutive_errors = 0
    iterator = iter(pending)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        active = {}

        def refill():
            while not stopped and len(active) < args.workers:
                item = next(iterator, None)
                if item is None:
                    break
                active[executor.submit(run, item)] = item

        refill()
        while active:
            done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
            for future in done:
                row, attempt, path = active.pop(future)
                item_id, accepted, error = future.result()
                journal[(row["id"], attempt)] = read_json(path)
                consecutive_errors = consecutive_errors + 1 if error else 0
                if consecutive_errors >= args.max_infrastructure_errors:
                    stopped = True
                print(item_id, "accepted=" + str(accepted), "error=" + str(error), flush=True)
            progress()
            refill()
    summary = progress(finished=True)
    print(summary, flush=True)
    if stopped:
        raise SystemExit("Stopped after consecutive provider/infrastructure failures; unscheduled tasks preserved")


if __name__ == "__main__":
    main()
