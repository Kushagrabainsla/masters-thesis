"""Build `code_manifest.json` on the Mac (L-25): the SHA-256 of each file in `analyze_e56.CODE_FILES` as committed at
the design commit, read from git and not from a working copy. The run script uploads it with the package; V2 compares
it with the hashes of the files that run on the Pod (`code_sha.txt`).

Usage: make_manifest.py --commit <sha> --out <file>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
import analyze_e56 as an  # noqa: E402

E55_FILES = ("splice_e55.py", "run_e55.py", "analyze_e55.py")
PATHS = {name: f"experiments/{'gemma4_kv_model' if name in E55_FILES else 'gemma4_kv_tasks'}/{name}"
         for name in an.CODE_FILES}


def manifest(commit: str) -> dict[str, str]:
    """The SHA-256 of the blob of each code file at the commit (`git cat-file blob`: the bytes as committed)."""
    out = {}
    for name, path in PATHS.items():
        blob = subprocess.run(["git", "-C", str(REPO), "cat-file", "blob", f"{commit}:{path}"], capture_output=True,
                              check=True).stdout
        out[name] = hashlib.sha256(blob).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(json.dumps(manifest(args.commit), indent=1, sort_keys=True) + "\n")
    print(f"{args.out}: {len(PATHS)} files at {args.commit}")


if __name__ == "__main__":
    main()
