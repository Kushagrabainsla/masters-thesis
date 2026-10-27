"""E-39 analysis (README Section 5 and amendment A1): paired rounds of Marlin MoE against R.

Layout: <res>/R_i, C_i for rounds i = 1..N (run_cell.py throughput run.json and log.txt), <res>/env-*/telemetry.csv.
Per round and batch size: log(median C / median R). One-sided t bounds on the paired log ratios, at
alpha 0.05 / 4 for the 4 deciding batch sizes (4, 8, 16, 32). Other sizes are reported only.
Usage: analyze_e39.py --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

SIZES = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
DECIDE = ["4", "8", "16", "32"]
BAR = 1.03
HARD_BITS = 0x8 | 0x20 | 0x40 | 0x80
MIN_ROWS = 100
HARD_MAX = 0.02
# One-sided t quantiles at 1 - 0.0125 by degrees of freedom (A1: Bonferroni over 4 deciding sizes).
T_0125 = {2: 6.205, 3: 4.177, 4: 3.495, 5: 3.163, 6: 2.969, 7: 2.841, 8: 2.752, 9: 2.685, 10: 2.634, 11: 2.593}


def process_medians(run: dict) -> dict[str, float]:
    tp = run.get("throughput") or {}
    n_tok = (run.get("args") or {}).get("throughput_max_tokens", 128)
    return {bs: statistics.median(int(bs) * n_tok / s for s in v["seconds"] if s > 0)
            for bs, v in (tp.get("decode") or {}).items()}


def telemetry(res: Path) -> dict:
    rows = busy = hard = 0
    for path in res.glob("env-*/telemetry.csv"):
        for line in path.read_text(errors="replace").splitlines()[1:]:
            parts = [p.strip() for p in line.split(",")]
            try:
                util, reasons = float(parts[4]), int(parts[5], 16)
            except (ValueError, IndexError):
                continue
            rows += 1
            if util > 0:
                busy += 1
                hard += bool(reasons & HARD_BITS)
    return {"rows": rows, "busy": busy, "hard_fraction": hard / busy if busy else None}


def rounds(res: Path) -> list[dict]:
    out = []
    i = 1
    while (res / f"R_{i}").exists() or (res / f"C_{i}").exists():
        if not ((res / f"R_{i}").exists() and (res / f"C_{i}").exists()):
            break  # the time cap stopped this round after one side: it is not a round
        pair = {}
        for side in ("R", "C"):
            d = res / f"{side}_{i}"
            run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
            log = (d / "log.txt").read_text(errors="replace") if (d / "log.txt").exists() else ""
            pair[side] = {"ok": run.get("status") == "ok", "aot": "Directly load AOT" in log,
                          "medians": process_medians(run) if run.get("status") == "ok" else {}}
        out.append(pair)
        i += 1
    return out


def bounds(logs: list[float]) -> tuple[float, float, float] | None:
    n = len(logs)
    if n < 3 or n - 1 not in T_0125:
        return None
    m, sd = statistics.mean(logs), statistics.stdev(logs)
    half = T_0125[n - 1] * sd / math.sqrt(n)
    return math.exp(m), math.exp(m - half), math.exp(m + half)


def summarize(res: Path) -> dict:
    rs = rounds(res)
    complete = [p for p in rs if all(p[s]["ok"] and p[s]["aot"] for s in ("R", "C"))]
    by_batch, cv = {}, {}
    for bs in SIZES:
        logs = [math.log(p["C"]["medians"][bs] / p["R"]["medians"][bs]) for p in complete
                if bs in p["C"]["medians"] and bs in p["R"]["medians"]]
        r = [p["R"]["medians"][bs] for p in complete if bs in p["R"]["medians"]]
        if len(r) >= 2:
            cv[bs] = round(statistics.stdev(r) / statistics.mean(r), 4)
        b = bounds(logs)
        if b:
            slope = None
            if len(logs) >= 3:  # drift check: slope of the log ratio against the round index (reported)
                xs = list(range(len(logs)))
                xm = statistics.mean(xs)
                slope = sum((x - xm) * (y - statistics.mean(logs)) for x, y in zip(xs, logs)) / sum((x - xm) ** 2 for x in xs)
            by_batch[bs] = {"ratio": round(b[0], 4), "lower": round(b[1], 4), "upper": round(b[2], 4),
                            "rounds": len(logs), "drift_slope": round(slope, 5) if slope is not None else None}
    tel = telemetry(res)
    validity = {
        "complete_rounds": len(complete), "rounds_started": len(rs),
        "V2": len(complete) == len(rs) and len(complete) >= 6,
        "V3": tel["rows"] >= MIN_ROWS and tel["hard_fraction"] is not None and tel["hard_fraction"] <= HARD_MAX,
        "telemetry": tel, "R_cv_reported": cv,
    }
    if not (validity["V2"] and validity["V3"]):
        decision = "INVALID (" + ", ".join(k for k in ("V2", "V3") if not validity[k]) + ")"
    elif any(by_batch.get(bs, {}).get("lower", 0) >= BAR for bs in DECIDE):
        decision = "GO (" + ", ".join(bs for bs in DECIDE if by_batch.get(bs, {}).get("lower", 0) >= BAR) + ")"
    elif all(bs in by_batch and by_batch[bs]["upper"] < BAR for bs in DECIDE):
        decision = "KILL"
    else:
        decision = "INCONCLUSIVE (" + ", ".join(bs for bs in DECIDE if not (bs in by_batch and by_batch[bs]["upper"] < BAR)) + ")"
    return {"validity": validity, "by_batch": by_batch, "decision": decision}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.res)
    args.out.write_text(json.dumps(result, indent=1))
    print(result["decision"])


if __name__ == "__main__":
    main()
