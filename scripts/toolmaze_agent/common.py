"""Project paths, configuration, atomic journals, and provenance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_config(path):
    import yaml
    return yaml.safe_load(project_path(path).read_text(encoding="utf-8"))


def git_revision(path):
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def env_secret(name, required=True):
    value = os.environ.get(name)
    if required and not value:
        raise ValueError(f"Set environment variable {name}; never put API keys in config files")
    return value


def credential_secret(config):
    """Resolve one credential without sourcing shell code or logging its value."""
    import re
    import shlex
    value = os.environ.get(config.get("api_key_env", ""))
    if value:
        return value
    if config.get("credentials_file"):
        variable = config.get("credentials_variable", "API_KEY")
        for line in project_path(config["credentials_file"]).read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
            if match and match.group(1) == variable:
                parts = shlex.split(match.group(2), comments=True)
                if len(parts) != 1 or not parts[0]:
                    raise ValueError(f"Credential variable {variable} must contain one nonempty value")
                return parts[0]
        raise ValueError(f"Credential variable {variable} was not found")
    return env_secret(config["api_key_env"])


def provenance(config):
    import importlib.metadata
    versions = {}
    for package in ("torch", "transformers", "trl", "deepspeed", "accelerate", "openai", "datasets", "vllm",
                    "flash-linear-attention", "fla-core", "peft", "liger-kernel"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    source = project_path(config.get("upstream", "tools/ToolMaze"))
    code = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / "scripts/toolmaze_agent").glob("*.py"))}
    return {"config": config, "packages": versions, "source_sha256": code,
            "toolmaze_commit": git_revision(source) if (source / ".git").exists() else None}
