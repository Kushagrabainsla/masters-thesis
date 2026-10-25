"""Registry of the Runpod resources that belong to this thesis (D-59).

The Runpod account is shared with other projects and sessions. A thesis agent acts only on a resource
that has BOTH:
  1. the thesis label: a name that starts with "msthesis-" and the Pod env THESIS_TAG=msthesis, and
  2. a "created" line in this registry, written right after the create call returned its id.
A name or a tag alone is not proof: another session can use the same words. The registry line is
the proof. Everything else on the account is read-only for thesis work.

Name format: msthesis-<experiment>-<purpose>-<yyyymmdd>, for example msthesis-e-36-rehearsal-20261026.

Usage:
  pods.py name <experiment> <purpose>                    print a label for a new resource
  pods.py add <id> <name> <experiment> <kind> [--note t]  record a create (kind: pod, volume, template)
  pods.py end <id> [--note t]                            record a terminate or delete
  pods.py ours                                           print the ids that are thesis-owned and not ended
  pods.py check <id>                                     exit 0 only if the id is thesis-owned and not ended
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "pods.jsonl"
PREFIX = "msthesis-"
TAG = {"THESIS_TAG": "msthesis"}
NAME = re.compile(r"^msthesis-[a-z0-9]+(-[a-z0-9]+)*-\d{8}$")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def label(experiment: str, purpose: str, day: str | None = None) -> str:
    slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")  # noqa: E731
    return f"{PREFIX}{slug(experiment)}-{slug(purpose)}-{day or time.strftime('%Y%m%d', time.gmtime())}"


def load(path: Path = REGISTRY) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def owned(path: Path = REGISTRY) -> dict[str, dict]:
    live: dict[str, dict] = {}
    for e in load(path):
        if e["event"] == "created":
            live[e["id"]] = e
        elif e["event"] == "ended":
            live.pop(e["id"], None)
    return live


def append(entry: dict, path: Path = REGISTRY) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def add(rid: str, name: str, experiment: str, kind: str, note: str = "", path: Path = REGISTRY) -> str | None:
    if not NAME.match(name):
        return f"name must match {NAME.pattern}, for example {label(experiment, 'run')}"
    if any(e["id"] == rid for e in load(path)):
        return f"id already in the registry: {rid}"
    append({"event": "created", "id": rid, "name": name, "experiment": experiment, "kind": kind,
            "at": now(), "note": note}, path)
    return None


def end(rid: str, note: str = "", path: Path = REGISTRY) -> str | None:
    if rid not in owned(path):
        return f"not a live thesis resource: {rid}"
    append({"event": "ended", "id": rid, "at": now(), "note": note}, path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("name")
    n.add_argument("experiment")
    n.add_argument("purpose")
    a = sub.add_parser("add")
    for k in ("id", "name", "experiment", "kind"):
        a.add_argument(k)
    a.add_argument("--note", default="")
    e = sub.add_parser("end")
    e.add_argument("id")
    e.add_argument("--note", default="")
    sub.add_parser("ours")
    c = sub.add_parser("check")
    c.add_argument("id")
    args = parser.parse_args()
    if args.cmd == "name":
        print(label(args.experiment, args.purpose))
    elif args.cmd == "add":
        err = add(args.id, args.name, args.experiment, args.kind, args.note)
        sys.exit(err) if err else print(f"recorded {args.id}")
    elif args.cmd == "end":
        err = end(args.id, args.note)
        sys.exit(err) if err else print(f"ended {args.id}")
    elif args.cmd == "ours":
        for rid, e in owned().items():
            print(f"{rid}\t{e['kind']}\t{e['name']}\t{e['experiment']}\t{e['at']}")
    else:
        sys.exit(0 if args.id in owned() else f"NOT thesis-owned: {args.id}. Do not change it.")


if __name__ == "__main__":
    main()
