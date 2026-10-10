"""Run manifest. Fields follow RESEARCH_JOURNAL.md Section 5.8 where they apply on a Mac.

Git SHAs are meaningless here because the repo has been untracked since 2026-05-21, so the
manifest records SHA-256 hashes of the harness sources and the prompt file instead.
"""
import datetime
import hashlib
import json
import platform
from pathlib import Path

import torch


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path, extra: dict) -> dict:
    import transformers
    sources = sorted(list((root / "harness").glob("*.py")) + list(root.glob("run_*.py")))
    m = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "torch_threads": torch.get_num_threads(),
        "source_sha256": {str(p.relative_to(root)): sha256_file(p) for p in sources},
    }
    prompts = root / "prompts.json"
    if prompts.exists():
        m["prompts_sha256"] = sha256_file(prompts)
    m.update(extra)
    return m


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
