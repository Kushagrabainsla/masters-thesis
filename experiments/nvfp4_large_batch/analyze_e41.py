"""E-41 analysis (README Section 5): A1, paired speed bounds at batch 256 and 512, and the frozen rule.

Layout under the results root:
  acc/R_a, acc/cand_<v>                   run_a2.py --no-score accuracy runs
  paired/<v>/R_i, C_i                     run_cell.py throughput runs, rounds i = 1..8
  env-*/telemetry.csv                     nvidia-smi samples

Usage:
  analyze_e41.py accept --res <dir>                  prints the candidates that pass A1
  analyze_e41.py decide --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nvfp4_dispatch"))
sys.path.insert(0, str(HERE.parent / "safedispatch_testA2"))

from analyze_a2 import MARGIN, load_jsonl, paired_interval  # noqa: E402
from analyze_e40 import HARD_BITS, medians, run_of  # noqa: E402

CANDS = ["flashinfer_cudnn", "flashinfer_cutlass"]
DECIDE = ["256", "512"]
REPORT = ["128", "256", "384", "512"]
BAR = 1.03
MIN_ROUNDS = 6
T_0125 = {2: 6.205, 3: 4.177, 4: 3.495, 5: 3.163, 6: 2.969, 7: 2.841, 8: 2.752, 9: 2.685}  # one-sided, alpha 0.05 / 4
# E-40 lesson: the dense NVFP4 line is "Using <Kernel> for NVFP4 GEMM", not "Selected X for Y".
LINEAR = re.compile(r"Using (\w+) for NVFP4 GEMM")


def linear_kernel(run_dir: Path) -> str | None:
    path = run_dir / "log.txt"
    if not path.exists():
        return None
    m = LINEAR.search(path.read_text(errors="replace"))
    return m.group(1) if m else None


def accept(res: Path) -> dict:
    ref = load_jsonl(res / "acc" / "R_a" / "completions.jsonl")
    out = {}
    for v in CANDS:
        d = res / "acc" / f"cand_{v}"
        rows = load_jsonl(d / "completions.jsonl") if (d / "completions.jsonl").exists() else []
        if not ref or not rows or run_of(d).get("status") != "ok":
            out[v] = {"A1": None, "status": "missing"}
            continue
        diff, lo, hi = paired_interval(ref, rows)
        out[v] = {"diff": diff, "interval95": [lo, hi], "A1": lo > -MARGIN, "status": "ok"}
    return out


def paired(pdir: Path) -> dict:
    pairs, i = [], 1
    while (pdir / f"R_{i}").exists() and (pdir / f"C_{i}").exists():
        r, c = run_of(pdir / f"R_{i}"), run_of(pdir / f"C_{i}")
        pairs.append((all(x.get("status") == "ok" for x in (r, c)), medians(r), medians(c)))
        i += 1
    good = [(m_r, m_c) for ok, m_r, m_c in pairs if ok]
    by_batch = {}
    for bs in REPORT:
        logs = [math.log(m_c[bs] / m_r[bs]) for m_r, m_c in good if bs in m_r and bs in m_c]
        if len(logs) >= 3 and (len(logs) - 1) in T_0125:
            mean, sd = statistics.mean(logs), statistics.stdev(logs)
            half = T_0125[len(logs) - 1] * sd / math.sqrt(len(logs))
            by_batch[bs] = {"ratio": round(math.exp(mean), 4), "lower": round(math.exp(mean - half), 4),
                            "upper": round(math.exp(mean + half), 4), "rounds": len(logs)}
    kr = {linear_kernel(pdir / f"R_{k}") for k in range(1, i)} - {None}
    kc = {linear_kernel(pdir / f"C_{k}") for k in range(1, i)} - {None}
    failed = [k + 1 for k, (ok, _, _) in enumerate(pairs) if not ok]
    return {"rounds": len(pairs), "good": len(good), "failed_rounds": failed,
            "V2": len(good) >= MIN_ROUNDS,  # A1 (R-49): failed rounds are dropped and reported
            "R_kernels": sorted(kr), "C_kernels": sorted(kc), "alias": bool(kc) and kc == kr,
            "by_batch": by_batch}


def telemetry_ok(res: Path) -> bool:
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
    return rows >= 100 and busy > 0 and hard / busy <= 0.02


def summarize(res: Path) -> dict:
    acc = accept(res)
    pr = {v: paired(res / "paired" / v) for v in CANDS if (res / "paired" / v).is_dir()}
    v0 = bool(linear_kernel(res / "acc" / "R_a")) or any(p["R_kernels"] for p in pr.values())
    return {"accept": acc, "paired": pr, "V0": v0, "V3": telemetry_ok(res)}


def decide(s: dict) -> str:
    bad = [k for k in ("V0", "V3") if not s[k]]
    if bad:
        return f"INVALID ({', '.join(bad)})"
    go, open_ = [], []
    for v in CANDS:
        a = s["accept"].get(v, {})
        if a.get("A1") is None:
            open_.append(f"{v} accuracy missing")
            continue
        if not a["A1"]:
            continue
        p = s["paired"].get(v)
        if not p or not p["V2"]:
            open_.append(f"{v} paired speed incomplete")
            continue
        if p["alias"]:
            open_.append(f"{v} selects R's kernel")
            continue
        if any(p["by_batch"].get(bs, {}).get("lower", 0) >= BAR for bs in DECIDE):
            go.append(v)
        elif not all(bs in p["by_batch"] and p["by_batch"][bs]["upper"] < BAR for bs in DECIDE):
            open_.append(f"{v} not resolved")
    if go:
        return "GO (" + ", ".join(go) + ")"
    if open_:
        tie = all(p["by_batch"].get(bs, {}).get("ratio", 9) < BAR for p in s["paired"].values() for bs in DECIDE)
        note = "; every deciding ratio below 1.03: no evidence of a lead" if s["paired"] and tie else ""
        return "INCONCLUSIVE (" + "; ".join(open_) + note + ")"
    return "KILL"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("accept")
    a.add_argument("--res", required=True, type=Path)
    d = sub.add_parser("decide")
    d.add_argument("--res", required=True, type=Path)
    d.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.cmd == "accept":
        acc = accept(args.res)
        (args.res / "accept.json").write_text(json.dumps(acc, indent=1))
        print(" ".join(v for v, x in acc.items() if x.get("A1")))
    else:
        s = summarize(args.res)
        s["decision"] = decide(s)
        args.out.write_text(json.dumps(s, indent=1, default=str))
        print(s["decision"])


if __name__ == "__main__":
    main()
