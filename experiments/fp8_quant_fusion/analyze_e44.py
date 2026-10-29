"""E-44 analysis (README Section 5): paired D2/D1 bounds, the microbenchmark condition, validity and the rule.

Layout under the results root:
  kbench/kbench.json                   kbench_e44.py output
  prof/D0, prof/D1, prof/D2            run_cell.py throughput runs with --profile-kernels and DEBUG logs
  paired/D1_i, paired/D2_i             run_cell.py throughput runs, rounds i = 1..6
  env-*/telemetry.csv                  nvidia-smi samples

Usage: analyze_e44.py --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nvfp4_dispatch"))
sys.path.insert(0, str(HERE.parent / "safedispatch_testA2"))
sys.path.insert(0, str(HERE.parent / "ledger"))

from analyze_e40 import HARD_BITS, medians, run_of  # noqa: E402

DECIDE = ["8", "64", "256"]
REPORT = ["1", "8", "64", "256"]
GO_BAR, KILL_BAR, KB_BAR, KB_MIN = 1.02, 1.01, 1.3, 3
MIN_ROUNDS = 5
T_0167 = {3: 3.740, 4: 3.186, 5: 2.912}  # one-sided, alpha 0.05 / 3
FUSED = re.compile(r"rms_norm_per_block_quant|silu_and_mul_per_block_quant")
SELECTED = re.compile(r"Selected (\S+) for Fp8LinearMethod")
ENABLED = re.compile(r"Enabled custom fusions: ([\w, ]+)")
MATCHES = re.compile(r"fusion pass matches: (\{.*\})")


def log_text(d: Path) -> str:
    p = d / "log.txt"
    return p.read_text(errors="replace") if p.exists() else ""


def fused_kernels(d: Path) -> int:
    p = d / "kernels.json"
    if not p.exists():
        return -1  # no profile
    k = json.loads(p.read_text())
    return sum(1 for v in k.get("by_batch", {}).values() for r in v.get("kernels", []) if FUSED.search(r["name"]))


def match_table(text: str) -> dict:
    out = {}
    for m in MATCHES.finditer(text):
        try:
            for k, v in ast.literal_eval(m.group(1)).items():
                out[k] = max(out.get(k, 0), int(v))
        except (ValueError, SyntaxError):
            continue
    return out


def config_info(d: Path) -> dict:
    t = log_text(d)
    sel = SELECTED.findall(t)
    en = ENABLED.findall(t)
    return {"selected": sel[0] if sel else None, "enabled_fusions": en[-1].strip() if en else "",
            "fused_kernels": fused_kernels(d), "match_table": match_table(t), "status": run_of(d).get("status")}


def paired(res: Path) -> dict:
    pairs, i = [], 1
    while (res / "paired" / f"D1_{i}").exists() and (res / "paired" / f"D2_{i}").exists():
        a, b = run_of(res / "paired" / f"D1_{i}"), run_of(res / "paired" / f"D2_{i}")
        pairs.append((all(x.get("status") == "ok" for x in (a, b)), medians(a), medians(b)))
        i += 1
    good = [(m1, m2) for ok, m1, m2 in pairs if ok]
    by = {}
    for bs in REPORT:
        logs = [math.log(m2[bs] / m1[bs]) for m1, m2 in good if bs in m1 and bs in m2]
        if len(logs) >= 4 and (len(logs) - 1) in T_0167:
            mean, sd = statistics.mean(logs), statistics.stdev(logs)
            half = T_0167[len(logs) - 1] * sd / math.sqrt(len(logs))
            by[bs] = {"ratio": round(math.exp(mean), 4), "lower": round(math.exp(mean - half), 4),
                      "upper": round(math.exp(mean + half), 4), "rounds": len(logs)}
    return {"rounds": len(pairs), "good": len(good), "failed": [k + 1 for k, (ok, _, _) in enumerate(pairs) if not ok],
            "V2": len(good) >= MIN_ROUNDS, "by_batch": by}


def kbench(res: Path) -> dict:
    p = res / "kbench" / "kbench.json"
    if not p.exists():
        return {"present": False, "slow_counts": {}, "condition": False}
    k = json.loads(p.read_text())
    counts = {}
    for row in k.get("by_tokens", {}).values():
        for op in ("norm", "act"):
            r = row.get(op, {})
            if r.get("fused_over_unfused", 0) >= KB_BAR:
                counts[op] = counts.get(op, 0) + 1
    return {"present": True, "slow_counts": counts, "condition": any(c >= KB_MIN for c in counts.values())}


def telemetry_ok(res: Path) -> bool:
    rows = busy = hard = 0
    for path in res.glob("env-*/telemetry.csv"):
        for line in path.read_text(errors="replace").splitlines()[1:]:
            parts = [x.strip() for x in line.split(",")]
            try:
                util, reasons = float(parts[4]), int(parts[5], 16)
            except (ValueError, IndexError):
                continue
            rows += 1
            if util > 0:
                busy += 1
                hard += bool(reasons & HARD_BITS)
    return rows >= 100 and busy > 0 and hard / busy <= 0.02


def summarize(res: Path) -> dict:
    cfg = {c: config_info(res / "prof" / c) for c in ("D0", "D1", "D2")}
    d1, d2 = cfg["D1"], cfg["D2"]
    v0 = ("norm_quant" in d1["enabled_fusions"] and "act_quant" in d1["enabled_fusions"]
          and d1["fused_kernels"] > 0 and d2["fused_kernels"] == 0
          and d1["selected"] is not None and d1["selected"] == d2["selected"])
    return {"configs": cfg, "paired": paired(res), "kbench": kbench(res), "V0": v0, "V3": telemetry_ok(res)}


def decide(s: dict) -> str:
    bad = [k for k in ("V0", "V3") if not s[k]] + ([] if s["paired"]["V2"] else ["V2"])
    if bad:
        return f"INVALID ({', '.join(bad)})"
    by = s["paired"]["by_batch"]
    if any(by.get(bs, {}).get("lower", 0) >= GO_BAR for bs in DECIDE) and s["kbench"]["condition"]:
        return "GO"
    if all(bs in by and by[bs]["upper"] < KILL_BAR for bs in DECIDE):
        return "KILL"
    why = []
    if not any(by.get(bs, {}).get("lower", 0) >= GO_BAR for bs in DECIDE):
        why.append("no resolved end-to-end gain")
    if not s["kbench"]["condition"]:
        why.append("microbenchmark condition not met" if s["kbench"]["present"] else "microbenchmark missing")
    return "INCONCLUSIVE (" + "; ".join(why) + ")"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    s = summarize(a.res)
    s["decision"] = decide(s)
    a.out.write_text(json.dumps(s, indent=1, default=str))
    print(s["decision"])


if __name__ == "__main__":
    main()
