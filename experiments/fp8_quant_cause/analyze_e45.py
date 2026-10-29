"""E-45 analysis (README Section 5): pairs P1 to P3, the accuracy gate, V0 and the frozen rule.

Layout under the results root:
  acc/A0, acc/A1                         run_a2.py --no-score accuracy runs (Qwen3-8B-FP8)
  prof/A0, prof/A1                       run_cell.py throughput runs with --profile-kernels
  paired/P<n>/X_i, paired/P<n>/Y_i       run_cell.py throughput runs; ratio = Y / X (second config / first)
  env-*/telemetry.csv                    nvidia-smi samples

Usage: analyze_e45.py --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("nvfp4_dispatch", "safedispatch_testA2", "ledger", "fp8_quant_fusion"):
    sys.path.insert(0, str(HERE.parent / sub))

from analyze_a2 import MARGIN, load_jsonl, paired_interval  # noqa: E402
from analyze_e40 import medians, run_of  # noqa: E402
from analyze_e44 import telemetry_ok  # noqa: E402

DECIDE = ["8", "64", "256"]
REPORT = ["1", "8", "64", "256"]
GO_BAR, KILL_BAR = 1.02, 1.01
MIN_ROUNDS = 4
T_0167 = {3: 3.740, 4: 3.186}  # one-sided, alpha 0.05 / 3
QUANT = "per_token_group_quant"
V0_MAX_FRAC = 0.25


def paired(pdir: Path) -> dict:
    pairs, i = [], 1
    while (pdir / f"X_{i}").exists() and (pdir / f"Y_{i}").exists():
        x, y = run_of(pdir / f"X_{i}"), run_of(pdir / f"Y_{i}")
        pairs.append((all(r.get("status") == "ok" for r in (x, y)), medians(x), medians(y)))
        i += 1
    good = [(mx, my) for ok, mx, my in pairs if ok]
    by = {}
    for bs in REPORT:
        logs = [math.log(my[bs] / mx[bs]) for mx, my in good if bs in mx and bs in my]
        if len(logs) - 1 in T_0167:
            mean, sd = statistics.mean(logs), statistics.stdev(logs)
            half = T_0167[len(logs) - 1] * sd / math.sqrt(len(logs))
            by[bs] = {"ratio": round(math.exp(mean), 4), "lower": round(math.exp(mean - half), 4),
                      "upper": round(math.exp(mean + half), 4), "rounds": len(logs)}
    return {"rounds": len(pairs), "good": len(good), "failed": [k + 1 for k, (ok, _, _) in enumerate(pairs) if not ok],
            "V2": len(good) >= MIN_ROUNDS, "by_batch": by}


def quant_calls(run_dir: Path, bs: str = "64") -> int | None:
    p = run_dir / "kernels.json"
    if not p.exists():
        return None
    v = json.loads(p.read_text()).get("by_batch", {}).get(bs)
    return None if v is None else sum(r["count"] for r in v["kernels"] if QUANT in r["name"])


def accuracy(res: Path) -> dict:
    a, b = res / "acc" / "A0", res / "acc" / "A1"
    ra, rb = load_jsonl(a / "completions.jsonl"), load_jsonl(b / "completions.jsonl")  # {id: row}
    if not ra or not rb or run_of(b).get("status") != "ok":
        return {"A1": None, "status": "missing"}
    diff, lo, hi = paired_interval(ra, rb)
    same = sum(ra[k].get("token_ids") == rb[k].get("token_ids") for k in ra if k in rb)
    return {"diff": diff, "interval95": [lo, hi], "A1": lo > -MARGIN, "same_tokens": same, "n": len(ra), "status": "ok"}


def summarize(res: Path) -> dict:
    q0, q1 = quant_calls(res / "prof" / "A0"), quant_calls(res / "prof" / "A1")
    log1 = (res / "prof" / "A1" / "log.txt").read_text(errors="replace") if (res / "prof" / "A1" / "log.txt").exists() else ""
    v0 = q0 is not None and q1 is not None and q0 > 0 and q1 <= V0_MAX_FRAC * q0 and "-quant_fp8" in log1
    return {"pairs": {p: paired(res / "paired" / p) for p in ("P1", "P2", "P3")}, "accuracy": accuracy(res),
            "quant_calls_bs64": {"A0": q0, "A1": q1}, "V0": v0, "V3": telemetry_ok(res)}


def decide(s: dict) -> str:
    p1, p2 = s["pairs"]["P1"], s["pairs"]["P2"]
    acc = s["accuracy"]
    if acc.get("A1") is False:
        return "KILL (A1 fails the accuracy gate)"
    bad = [k for k in ("V0", "V3") if not s[k]] + ([] if p1["V2"] else ["V2 P1"])
    if bad:
        return f"INVALID ({', '.join(bad)})"
    by1, by2 = p1["by_batch"], p2["by_batch"]
    go = [bs for bs in DECIDE if by1.get(bs, {}).get("lower", 0) >= GO_BAR and by2.get(bs, {}).get("ratio", 0) >= 1.0]
    if go and acc.get("A1"):
        return "GO (" + ", ".join(f"batch {bs}" for bs in go) + ")"
    if all(bs in by1 and by1[bs]["upper"] < KILL_BAR for bs in DECIDE):
        return "KILL (native quant is not faster)"
    why = []
    if not go:
        why.append("no resolved P1 gain with P2 at least 1.00")
    if acc.get("A1") is None:
        why.append("accuracy missing")
    return "INCONCLUSIVE (" + "; ".join(why) + ")"


def p3_label(s: dict) -> str:
    by = s["pairs"]["P3"]["by_batch"]
    return "fusion slowdown holds on M14" if any(by.get(bs, {}).get("lower", 0) >= GO_BAR for bs in DECIDE) else "not shown on M14"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    s = summarize(a.res)
    s["decision"], s["P3"] = decide(s), p3_label(s)
    a.out.write_text(json.dumps(s, indent=1, default=str))
    print(s["decision"], "|", s["P3"])


if __name__ == "__main__":
    main()
