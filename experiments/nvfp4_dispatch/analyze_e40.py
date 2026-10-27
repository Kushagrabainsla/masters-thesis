"""E-40 analysis (README Sections 4 and 5): the screen, A1, paired speed bounds and the frozen rule.

Cell layout:
  <cell>/screen/R_1, R_2, cand_<v>        run_cell.py throughput runs (one process each)
  <cell>/acc/R_a, cand_<v>                run_a2.py --no-score accuracy runs
  <cell>/paired/<v>/R_i, C_i              run_cell.py throughput runs, rounds i = 1..8

Usage:
  analyze_e40.py screen --cell <dir>                 prints the selected candidates (at most 2)
  analyze_e40.py accept --cell <dir>                 prints the selected candidates that pass A1
  analyze_e40.py decide --cells <D> <M> --out summary.json
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
sys.path.insert(0, str(HERE.parent / "safedispatch_testA"))
sys.path.insert(0, str(HERE.parent / "safedispatch_testA2"))
sys.path.insert(0, str(HERE.parent / "ledger"))

from analyze_a2 import MARGIN, load_jsonl, paired_interval  # noqa: E402

DECIDE = ["1", "8", "32", "128"]  # all in the A1 batch list
SCREEN_BAR = 1.02
BAR = 1.03
MAX_KEEP = 1        # A1 (R-47): budget
V1_MAX = 0.05
NOISY = 0.02        # A1: a screen with R noise above 2% cannot give KILL
MIN_ROUNDS = 5
T_0125 = {2: 6.205, 3: 4.177, 4: 3.495, 5: 3.163, 6: 2.969, 7: 2.841, 8: 2.752, 9: 2.685}
HARD_BITS = 0x8 | 0x20 | 0x40 | 0x80
# vLLM v0.32.0 logs NVFP4 MoE selection as "Using 'X' NvFp4 MoE backend out of ..." (oracle/nvfp4.py),
# which the frozen E-36 regex ("Using X Fp8 MoE backend") does not match. Match both, and linear lines.
SELECTION = re.compile(r"(Selected \S+ for \S+|Using '?[\w-]+'? (?:NvFp4|Fp8) MoE backend)")
FALLBACK = re.compile(r"falling back to normal kernel selection")


def kernel_signature(run_dir: Path, moe_only: bool = False) -> tuple[frozenset, bool]:
    """A1 (R-47): in the MoE cell only --moe-backend changes, so compare only the MoE selection line."""
    path = run_dir / "log.txt"
    lines = path.read_text(errors="replace").splitlines() if path.exists() else []
    sig = frozenset(m.group(1) for l in lines for m in [SELECTION.search(l)] if m)
    if moe_only:
        sig = frozenset(x for x in sig if "MoE backend" in x)
    return sig, any(FALLBACK.search(l) for l in lines)


def run_of(d: Path) -> dict:
    return json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}


def medians(run: dict) -> dict[str, float]:
    tp = run.get("throughput") or {}
    n_tok = (run.get("args") or {}).get("throughput_max_tokens", 128)
    return {bs: statistics.median(int(bs) * n_tok / s for s in v["seconds"] if s > 0)
            for bs, v in (tp.get("decode") or {}).items()}


def screen(cell: Path) -> dict:
    s = cell / "screen"
    r1, r2 = medians(run_of(s / "R_1")), medians(run_of(s / "R_2"))
    moe = cell.name.endswith("M")
    ref_sig, _ = kernel_signature(s / "R_1", moe)
    noise = {bs: abs(r1[bs] - r2[bs]) / max(r1[bs], r2[bs]) for bs in DECIDE if bs in r1 and bs in r2}
    out = {"noise": noise, "V1": bool(noise) and all(v <= V1_MAX for v in noise.values()),
           "V0": bool(ref_sig), "reference_kernels": sorted(ref_sig),
           "noisy": bool(noise) and max(noise.values()) > NOISY, "candidates": {}}
    for d in sorted(p for p in s.glob("cand_*") if p.is_dir()):
        name = d.name[len("cand_"):]
        run = run_of(d)
        if run.get("status") != "ok":
            out["candidates"][name] = {"status": run.get("status", "missing")}
            continue
        sig, fell_back = kernel_signature(d, moe)
        alias = fell_back or (bool(sig) and sig == ref_sig)
        m = medians(run)
        base = {bs: (r1[bs] + r2[bs]) / 2 if bs in r2 else r1[bs] for bs in r1}
        ratios = {bs: m[bs] / base[bs] for bs in m if bs in base}
        best = max((ratios[bs] for bs in DECIDE if bs in ratios), default=0.0)
        out["candidates"][name] = {"status": "ok", "alias": alias, "kernels": sorted(sig),
                                   "ratios": {k: round(v, 4) for k, v in ratios.items()}, "best_deciding": round(best, 4)}
    passing = [(c["best_deciding"], n) for n, c in out["candidates"].items()
               if c.get("status") == "ok" and not c["alias"] and c["best_deciding"] >= SCREEN_BAR]
    out["selected"] = [n for _, n in sorted(passing, reverse=True)[:MAX_KEEP]]
    return out


def accept(cell: Path, selected: list[str]) -> dict:
    ref = load_jsonl(cell / "acc" / "R_a" / "completions.jsonl")
    out = {}
    for v in selected:
        rows = load_jsonl(cell / "acc" / f"cand_{v}" / "completions.jsonl")
        if not ref or not rows or run_of(cell / "acc" / f"cand_{v}").get("status") != "ok":
            out[v] = {"A1": None, "status": "missing"}
            continue
        d, lo, hi = paired_interval(ref, rows)
        out[v] = {"diff": d, "interval95": [lo, hi], "A1": lo > -MARGIN, "status": "ok"}
    return out


def paired(pdir: Path) -> dict:
    pairs, i = [], 1
    while (pdir / f"R_{i}").exists() and (pdir / f"C_{i}").exists():
        r, c = run_of(pdir / f"R_{i}"), run_of(pdir / f"C_{i}")
        ok = all(x.get("status") == "ok" for x in (r, c))
        aot = all("Directly load AOT" in (pdir / f"{k}_{i}" / "log.txt").read_text(errors="replace")
                  for k in ("R", "C") if (pdir / f"{k}_{i}" / "log.txt").exists())
        pairs.append((ok, medians(r), medians(c), aot))
        i += 1
    good = [(m_r, m_c) for ok, m_r, m_c, _ in pairs if ok]
    aot_all = all(a for *_, a in pairs)  # A1 (R-47): reported, not a validity gate
    by_batch = {}
    for bs in DECIDE:
        logs = [math.log(m_c[bs] / m_r[bs]) for m_r, m_c in good if bs in m_r and bs in m_c]
        if len(logs) >= 3 and (len(logs) - 1) in T_0125:
            mean, sd = statistics.mean(logs), statistics.stdev(logs)
            half = T_0125[len(logs) - 1] * sd / math.sqrt(len(logs))
            by_batch[bs] = {"ratio": round(math.exp(mean), 4), "lower": round(math.exp(mean - half), 4),
                            "upper": round(math.exp(mean + half), 4), "rounds": len(logs)}
    return {"rounds": len(pairs), "good": len(good), "aot_all": aot_all,
            "V2": len(good) == len(pairs) and len(good) >= MIN_ROUNDS,
            "by_batch": by_batch}


def telemetry_ok(cell: Path) -> bool:
    rows = busy = hard = 0
    for path in cell.parent.glob("env-*/telemetry.csv"):
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


def decide(cells: dict[str, dict]) -> str:
    go, open_, reasons = [], [], []
    for name, c in cells.items():
        bad = [k for k, ok in (("V0", c["screen"].get("V0", True)), ("V1", c["screen"]["V1"]), ("V3", c["V3"])) if not ok]
        if bad:
            reasons.append(f"{name} invalid ({', '.join(bad)})")
            continue
        if not c["screen"]["selected"] and c["screen"].get("noisy"):
            reasons.append(f"{name} no screen pass, but R noise above 2%")
            continue
        for v in c["screen"]["selected"]:
            a = c["accept"].get(v, {})
            if a.get("A1") is None:
                open_.append(f"{name}/{v} accuracy missing")
                continue
            if not a["A1"]:
                continue
            p = c["paired"].get(v)
            if not p or not p["V2"]:
                open_.append(f"{name}/{v} paired speed incomplete")
                continue
            if any(p["by_batch"].get(bs, {}).get("lower", 0) >= BAR for bs in DECIDE):
                go.append(f"{name}/{v}")
            elif not all(bs in p["by_batch"] and p["by_batch"][bs]["upper"] < BAR for bs in DECIDE):
                open_.append(f"{name}/{v} not resolved")
    if go:
        return "GO (" + ", ".join(go) + ")"
    if reasons or open_:
        return "INCONCLUSIVE (" + "; ".join(reasons + open_) + ")"
    return "KILL"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("screen", "accept"):
        p = sub.add_parser(name)
        p.add_argument("--cell", required=True, type=Path)
    d = sub.add_parser("decide")
    d.add_argument("--cells", nargs="+", required=True, type=Path)
    d.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.cmd == "screen":
        s = screen(args.cell)
        (args.cell / "screen.json").write_text(json.dumps(s, indent=1))
        print(" ".join(s["selected"]))
    elif args.cmd == "accept":
        s = json.loads((args.cell / "screen.json").read_text())
        a = accept(args.cell, s["selected"])
        (args.cell / "accept.json").write_text(json.dumps(a, indent=1))
        print(" ".join(v for v, x in a.items() if x.get("A1")))
    else:
        cells = {}
        for cell in args.cells:
            s = screen(cell)
            a = accept(cell, s["selected"])
            p = {v: paired(cell / "paired" / v) for v in s["selected"] if (cell / "paired" / v).is_dir()}
            cells[cell.name] = {"screen": s, "accept": a, "paired": p, "V3": telemetry_ok(cell)}
        result = {"cells": cells, "decision": decide(cells)}
        args.out.write_text(json.dumps(result, indent=1, default=str))
        print(result["decision"])


if __name__ == "__main__":
    main()
