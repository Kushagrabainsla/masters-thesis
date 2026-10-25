"""Append-only run ledger for all GPU experiments (docs/experiments/PROTOCOL.md, stage S7).

Each run writes one run manifest (JSON, schema below). `add` checks it and appends it as one line of
`runs.jsonl`. A run id is never reused and a line is never changed: a correction is a new line with
`supersedes` set to the old run id.

Manifest fields (required): run_id, experiment, idea, status ("ok" | "failed" | "partial"),
design_commit, config (dict), started_at, finished_at, hardware (dict), software (dict),
metrics ({name: {"values": [floats, one per repeat], "unit": str, "higher_is_better": bool}}),
results_path. Optional: cost_usd, notes, supersedes, label.
`config_hash` is added by `add` (SHA-256 of the canonical config JSON).

Usage:
  ledger.py add <manifest.json> [...]
  ledger.py list [--experiment E-36]
  ledger.py show <run_id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "runs.jsonl"
REQUIRED = ("run_id", "experiment", "idea", "status", "design_commit", "config", "started_at",
            "finished_at", "hardware", "software", "metrics", "results_path")
HARDWARE_KEYS = ("gpu_name", "gpu_count", "driver", "cuda")
STATUS = ("ok", "failed", "partial")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def load(path: Path = LEDGER) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(m: dict, existing_ids: set[str]) -> list[str]:
    errors = [f"missing field: {k}" for k in REQUIRED if k not in m]
    if errors:
        return errors
    if m["run_id"] in existing_ids:
        errors.append(f"run_id already in the ledger: {m['run_id']}")
    if m["status"] not in STATUS:
        errors.append(f"status must be one of {STATUS}")
    errors += [f"hardware.{k} missing" for k in HARDWARE_KEYS if k not in m["hardware"]]
    for name, metric in m["metrics"].items():
        values = metric.get("values")
        if not isinstance(values, list) or not values or not all(isinstance(v, (int, float)) for v in values):
            errors.append(f"metric {name}: values must be a non-empty list of numbers")
        if not isinstance(metric.get("higher_is_better"), bool):
            errors.append(f"metric {name}: higher_is_better must be true or false")
    if m.get("supersedes") and m["supersedes"] not in existing_ids:
        errors.append(f"supersedes an unknown run_id: {m['supersedes']}")
    return errors


def add(paths: list[Path], ledger: Path = LEDGER) -> int:
    existing = {r["run_id"] for r in load(ledger)}
    rows = []
    for p in paths:
        m = json.loads(p.read_text())
        errors = validate(m, existing | {r["run_id"] for r in rows})
        if errors:
            print(f"{p}: rejected", *errors, sep="\n  ", file=sys.stderr)
            return 1
        m["config_hash"] = config_hash(m["config"])
        rows.append(m)
    with open(ledger, "a") as f:
        for m in rows:
            f.write(json.dumps(m, sort_keys=True) + "\n")
    print(f"added {len(rows)} run(s) to {ledger}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("manifests", nargs="+", type=Path)
    ls = sub.add_parser("list")
    ls.add_argument("--experiment")
    sh = sub.add_parser("show")
    sh.add_argument("run_id")
    args = parser.parse_args()
    if args.cmd == "add":
        sys.exit(add(args.manifests))
    rows = load()
    if args.cmd == "list":
        for r in rows:
            if args.experiment and r["experiment"] != args.experiment:
                continue
            hw = r["hardware"]
            print(f"{r['run_id']}\t{r['experiment']}\t{r['status']}\t{hw['gpu_count']}x {hw['gpu_name']}\t"
                  f"{r['config_hash']}\t{r.get('label', '')}")
    else:
        match = [r for r in rows if r["run_id"] == args.run_id]
        if not match:
            sys.exit(f"unknown run_id: {args.run_id}")
        print(json.dumps(match[-1], indent=1))


if __name__ == "__main__":
    main()
