#!/usr/bin/env python3
"""Fetch fixed upstream commits without modifying existing checkouts or environments."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="store_true", help="Also download Qwen3.5-2B weights")
    parser.add_argument("--model-revision", help="Required immutable HF commit when downloading model weights")
    args = parser.parse_args()
    lock = json.loads((ROOT / "tools/upstream.lock.json").read_text())
    for item in lock.values():
        target = ROOT / item["path"]
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", item["url"], str(target)], check=True)
            subprocess.run(["git", "-C", str(target), "checkout", "--detach", item["revision"]], check=True)
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != item["revision"]:
            raise RuntimeError(f"Revision mismatch at {target}; not overwriting existing checkout")
        print(item["path"], actual)
    if args.model:
        if not args.model_revision or len(args.model_revision) != 40:
            parser.error("--model requires --model-revision <40-character HF commit>")
        from huggingface_hub import snapshot_download
        target = ROOT / "models/Qwen3.5-2B"
        snapshot_download("Qwen/Qwen3.5-2B", revision=args.model_revision, local_dir=target)
        (target / "download_manifest.json").write_text(json.dumps({
            "repo_id": "Qwen/Qwen3.5-2B", "revision": args.model_revision}, indent=2) + "\n")


if __name__ == "__main__":
    main()
