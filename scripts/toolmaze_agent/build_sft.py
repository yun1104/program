"""Select validated teacher trajectories without changing the frozen task split."""
from __future__ import annotations

import argparse
from collections import Counter

from .common import digest, project_path, read_json, read_rows, write_json, write_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", default="runs/teacher")
    parser.add_argument("--output", default="runs/sft_data")
    parser.add_argument("--modes", nargs="+", default=["P0", "P1", "P2", "P3", "P4"])
    parser.add_argument("--max-per-task", type=int, default=1)
    args = parser.parse_args()
    collection = project_path(args.collection)
    output = project_path(args.output)
    if output.exists():
        raise FileExistsError("Use a new SFT data directory; frozen artifacts are not overwritten")
    manifest = read_json(collection / "manifest.json")
    if manifest["split"] != "train":
        raise ValueError("SFT builder only accepts a train-fold collection")
    frozen_path = project_path(manifest["config"]["splits"]) / "train.jsonl"
    if digest(frozen_path) != manifest["split_sha256"]:
        raise ValueError("Collection no longer matches the frozen train split")
    frozen_rows = {row["id"]: row for row in read_rows(frozen_path)}
    accepted, rejected = [], Counter()
    groups = Counter()
    for path in sorted((collection / "episodes").glob("*.json")):
        episode = read_json(path)
        row = episode["row"]
        if episode.get("split") != "train" or row["split"] != "train":
            raise ValueError(f"Non-training episode in collection: {path}")
        if frozen_rows.get(row["id"]) != row:
            raise ValueError(f"Episode source is not a member of the frozen training set: {path}")
        if row["mode"] not in args.modes:
            rejected["excluded_mode"] += 1
            continue
        if not episode.get("strict_pass") or episode.get("error"):
            rejected["not_strict_pass"] += 1
            continue
        if groups[row["id"]] >= args.max_per_task:
            rejected["per_task_cap"] += 1
            continue
        groups[row["id"]] += 1
        accepted.append({"id": row["id"], "group": row["group"], "mode": row["mode"],
                         "split": "train", "initial_messages": episode["initial_messages"],
                         "segments": episode["segments"], "source": str(path), "source_sha256": digest(path)})
    if not accepted:
        raise ValueError("No accepted training trajectories")
    write_rows(output / "train.jsonl", accepted)
    write_json(output / "manifest.json", {"collection_identity": manifest["identity"],
               "split_sha256": manifest["split_sha256"], "modes": args.modes,
               "accepted": len(accepted), "rejected": dict(rejected),
               "cells": dict(Counter(row["mode"] for row in accepted)),
               "train_sha256": digest(output / "train.jsonl")})
    print(f"Accepted {len(accepted)} trajectories -> {output}")


if __name__ == "__main__":
    main()
