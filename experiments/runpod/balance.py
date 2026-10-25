"""Print the live Runpod account balance, spend per hour and spend limit (D-56 rule: read it before
every experiment, never store it).

REST v2 and the MCP tools have no balance route, so this uses the GraphQL `myself` query with
RUNPOD_API_KEY from the repo's gitignored `.env` (or the environment). It never prints the key.

Usage: python3 experiments/runpod/balance.py [--need <dollars>]
With --need, it exits 1 if the balance is below <dollars> + 35 (the floor in runbook Section 2a).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ENV = Path(__file__).resolve().parents[2] / ".env"
FLOOR = 35.0
QUERY = "query { myself { clientBalance currentSpendPerHr spendLimit } }"


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key and ENV.exists():
        for line in ENV.read_text().splitlines():
            if line.startswith("RUNPOD_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit(f"RUNPOD_API_KEY is empty. Put it in {ENV} (gitignored).")
    return key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--need", type=float, default=None, help="cost cap of the planned run in dollars")
    args = parser.parse_args()
    req = urllib.request.Request(
        "https://api.runpod.io/graphql",
        data=json.dumps({"query": QUERY}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.load(resp)
    if body.get("errors"):
        sys.exit(f"Runpod API error: {body['errors'][0].get('message', body['errors'])}")
    me = body["data"]["myself"]
    print(json.dumps(me))
    if args.need is not None and float(me["clientBalance"]) < args.need + FLOOR:
        print(f"Balance is below the run cap plus ${FLOOR:.0f}. Do not book the Pod.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
