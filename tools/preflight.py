#!/usr/bin/env python3
"""Read-only environment, upstream, split-isolation and optional GPU checks."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.toolmaze_agent.common import digest, read_json, read_rows
from scripts.toolmaze_agent.prepare import load_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    result = {"dependencies": {name: importlib.util.find_spec(name) is not None for name in
              ("torch", "transformers", "trl", "accelerate", "datasets", "openai", "yaml", "jsonschema")}}
    for name, item in read_json(ROOT / "tools/upstream.lock.json").items():
        actual = subprocess.check_output(["git", "-C", str(ROOT / item["path"]), "rev-parse", "HEAD"], text=True).strip()
        if actual != item["revision"]:
            raise RuntimeError(f"{name} revision mismatch")
        result[name] = actual
    directory = ROOT / "data/splits/toolmaze_v1"
    manifest = read_json(directory / "manifest.json")
    assignments = {}
    counts = {}
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        if digest(path) != manifest["files"][split]:
            raise RuntimeError("Split manifest checksum mismatch")
        rows = read_rows(path)
        counts[split] = len(rows)
        for row in rows:
            load_task(row)
            for field in ("group", "template_id", "task_id"):
                key = (field, row[field])
                if assignments.setdefault(key, split) != split:
                    raise RuntimeError(f"Cross-split leakage: {key}")
    result["splits"] = counts
    if args.gpu:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        if not result["cuda_available"]:
            raise RuntimeError("CUDA is not accessible in this process")
        result["visible_gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        value = torch.ones(4, device="cuda")
        if value.sum().item() != 4:
            raise RuntimeError("CUDA tensor sanity check failed")
    print(json.dumps(result, indent=2))
    if not all(result["dependencies"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
