"""Freeze grouped splits before collection; modes of a task never cross folds."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import re

from .common import digest, fingerprint, load_config, project_path, read_json, write_json, write_rows


def query_key(task):
    query = task.get("user_input", {}).get("query", task.get("task_description", ""))
    return re.sub(r"\W+", " ", query.casefold()).strip()


def grouped_rows(task_dir, seed, ratios, domain=None):
    sources = sorted(task_dir.rglob("*.json"))
    tasks = []
    seen = set()
    for path in sources:
        task = read_json(path)
        if "task_id" not in task or "perturbation_mode" not in task:
            continue
        if domain and domain.casefold() not in {d.casefold() for d in task.get("domains", [])}:
            continue
        key = (task["task_id"], task["perturbation_mode"])
        if key in seen:
            raise ValueError(f"Duplicate task/mode: {key}")
        seen.add(key)
        if not task.get("template_id"):
            raise ValueError(f"Missing template_id: {path}; cannot guarantee grouped split")
        tasks.append((path, task))
    if not tasks:
        raise ValueError(f"No tasks under {task_dir}")
    # Connected components merge duplicate queries as well as template siblings.
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    for _, task in tasks:
        union("template:" + task["template_id"], "query:" + query_key(task))
        union("template:" + task["template_id"], "task:" + task["task_id"])
    groups = defaultdict(list)
    for path, task in tasks:
        groups[find("template:" + task["template_id"])].append((path, task))
    strata = defaultdict(list)
    for group, items in groups.items():
        strata[tuple(sorted({task["complexity"] for _, task in items}))].append(group)
    folds = {}
    for _, keys in sorted(strata.items()):
        keys.sort(key=lambda key: hashlib.sha256(f"{seed}|{key}".encode()).hexdigest())
        if len(keys) < 3:
            raise ValueError("Need at least three independent groups per complexity stratum")
        n_val = max(1, round(len(keys) * ratios[1]))
        n_test = max(1, round(len(keys) * ratios[2]))
        if n_val + n_test >= len(keys):
            raise ValueError("Not enough independent groups for requested ratios")
        for i, key in enumerate(keys):
            folds[key] = "validation" if i < n_val else "test" if i < n_val + n_test else "train"
    rows = []
    for group, items in sorted(groups.items()):
        for path, task in items:
            rows.append({"id": f"{task['task_id']}:{task['perturbation_mode']}",
                         "task_id": task["task_id"], "mode": task["perturbation_mode"],
                         "complexity": task["complexity"], "domains": task.get("domains", []),
                         "template_id": task["template_id"], "group": fingerprint(group)[:20],
                         "split": folds[group], "path": str(path.resolve()), "sha256": digest(path)})
    return rows


def load_task(row):
    path = project_path(row["path"])
    # Preserve frozen manifests and their hashes while allowing a checkout on
    # another machine. Resolve only the known upstream data subtree, then still
    # verify the exact task bytes below.
    if not path.is_file():
        marker = "/data/toolmaze/"
        original = str(row["path"]).replace("\\", "/")
        if marker in original:
            path = project_path("data/toolmaze/" + original.split(marker, 1)[1])
    if digest(path) != row["sha256"]:
        raise ValueError(f"Task changed after split freeze: {path}")
    return read_json(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="scripts/toolmaze_agent/configs/project.yaml")
    parser.add_argument("--domain", help="Optional case-insensitive domain, e.g. Travel")
    args = parser.parse_args()
    config = load_config(args.config)
    directory = project_path(config["splits"])
    if (directory / "manifest.json").exists():
        raise FileExistsError("Frozen split exists; use a new splits directory to create a new protocol")
    rows = grouped_rows(project_path(config["tasks"]), config["seed"], config["split_ratios"], args.domain)
    for split in ("train", "validation", "test"):
        write_rows(directory / f"{split}.jsonl", [r for r in rows if r["split"] == split])
    manifest = {"seed": config["seed"], "ratios": config["split_ratios"], "domain": args.domain,
                "grouping": "Connected components of template_id, task_id, normalized user query",
                "counts": dict(Counter(r["split"] for r in rows)),
                "cells": dict(Counter(f"{r['split']}/{r['complexity']}/{r['mode']}" for r in rows)),
                "files": {s: digest(directory / f"{s}.jsonl") for s in ("train", "validation", "test")}}
    write_json(directory / "manifest.json", manifest)
    print(manifest["counts"])


if __name__ == "__main__":
    main()
