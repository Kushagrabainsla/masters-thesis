"""E-36 analysis (amendment A1 after R-34): flips, floor, aliases, controls, speed, and the frozen rule.

Cell layout (one directory per cell):
  <cell>/R_a/, <cell>/R_b/   reference backend (auto), two runs, outputs + throughput each
  <cell>/cand_<name>/        each candidate backend, outputs + throughput
Each run directory holds run.json, outputs.jsonl and log.txt (the run's stdout and stderr, written by
session1.sh). Kernel selection is read from log.txt.

Usage: compare.py --cells <cell_dir_D> <cell_dir_M> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

UNSAFE = 0.10  # a candidate that flips at least 10% of prompts is "unsafe"
FLAT_UPPER = 0.02  # Wilson 95% upper bound for "no flips" (needs at least 189 prompts: 3.84 / (n + 3.84))
FLAT_SPEED_MIN = 0.02  # speed spread threshold floor; the real threshold is max(2%, 2 x R noise)
MARGIN = 0.03  # winner margin for CONTINUE and STATIC-WINNER
V1_AGREEMENT = 0.95
PC_MIN = 0.05  # the W8A16 positive control must flip at least 5% of prompts
W8A16 = {"marlin"}  # weight-only FP8: a different arithmetic contract, used as the positive control
KL_POSITIONS = 32
SELECTION = re.compile(r"(Selected \S+ for \S+|Using \S+ Fp8 MoE backend)")
FALLBACK = re.compile(r"falling back to normal kernel selection")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_run(run_dir: Path) -> dict | None:
    path = run_dir / "run.json"
    return json.loads(path.read_text()) if path.exists() else None


def load_outputs(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "outputs.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in rows}


def speeds_of(run: dict | None) -> dict[str, float]:
    decode = ((run or {}).get("throughput") or {}).get("decode", {})
    return {bs: v["median_tokens_per_s"] for bs, v in decode.items()}


def kernel_signature(run_dir: Path) -> tuple[frozenset, bool]:
    """Kernel-selection messages from the run's log.txt (vLLM prefixes each line; search, not match)."""
    path = run_dir / "log.txt"
    lines = path.read_text(errors="replace").splitlines() if path.exists() else []
    selected = frozenset(m.group(1) for l in lines for m in [SELECTION.search(l)] if m)
    fell_back = any(FALLBACK.search(l) for l in lines)
    return selected, fell_back


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def approx_kl(ref_topk: list, cand_topk: list) -> float:
    """KL(ref || cand) over ref's top-k support, renormalized; missing cand mass uses cand's
    smallest listed logprob. Approximate, reported only."""
    total, count = 0.0, 0
    for r_pos, c_pos in list(zip(ref_topk, cand_topk))[:KL_POSITIONS]:
        if not r_pos or not c_pos:
            continue
        r_lp = {int(t): lp for t, lp in r_pos}
        c_lp = {int(t): lp for t, lp in c_pos}
        floor = min(c_lp.values())
        r_norm = math.log(sum(math.exp(v) for v in r_lp.values()))
        c_vals = {t: c_lp.get(t, floor) for t in r_lp}
        c_norm = math.log(sum(math.exp(v) for v in c_vals.values()))
        kl = sum(math.exp(lp - r_norm) * ((lp - r_norm) - (c_vals[t] - c_norm)) for t, lp in r_lp.items())
        total += max(0.0, kl)
        count += 1
    return total / count if count else float("nan")


def compare_pair(ref: dict, other: dict, exclude: set[str]) -> dict:
    ids = [i for i in ref if i in other and i not in exclude]
    flipped, divergences, margins, kls = [], [], [], []
    per_source: dict[str, list[int]] = {}
    for i in ids:
        src = i.split("-", 1)[0]
        d = first_divergence(ref[i]["token_ids"], other[i]["token_ids"])
        per_source.setdefault(src, [0, 0])[1] += 1
        if d is not None:
            flipped.append(i)
            per_source[src][0] += 1
            divergences.append(d)
            topk = ref[i].get("topk") or []
            if d < len(topk) and len(topk[d]) >= 2:
                margins.append(topk[d][0][1] - topk[d][1][1])
        kls.append(approx_kl(ref[i].get("topk") or [], other[i].get("topk") or []))
    n, k = len(ids), len(flipped)
    finite = [x for x in kls if not math.isnan(x)]
    median = lambda xs: sorted(xs)[len(xs) // 2] if xs else None  # noqa: E731
    return {
        "n": n,
        "flipped": k,
        "flip_fraction": k / n if n else float("nan"),
        "wilson95": wilson(k, n),
        "flips_by_source": {s: {"flipped": v[0], "n": v[1]} for s, v in per_source.items()},
        "median_first_divergence": median(divergences),
        "median_margin_at_divergence": median(margins),
        "mean_approx_kl_first32": sum(finite) / len(finite) if finite else None,
    }


def summarize_cell(cell: Path) -> dict:
    run_a, run_b = load_run(cell / "R_a"), load_run(cell / "R_b")
    ref_a, ref_b = load_outputs(cell / "R_a"), load_outputs(cell / "R_b")
    floor_ids = {i for i in ref_a if i in ref_b and ref_a[i]["token_ids"] != ref_b[i]["token_ids"]}
    agreement = 1 - len(floor_ids) / len(ref_a) if ref_a and ref_b else float("nan")
    sp_a, sp_b = speeds_of(run_a), speeds_of(run_b)
    noise = {bs: abs(sp_a[bs] - sp_b[bs]) / max(sp_a[bs], sp_b[bs]) for bs in sp_a if bs in sp_b}
    ref_sig, _ = kernel_signature(cell / "R_a")
    speeds = {"R": {bs: (sp_a[bs] + sp_b.get(bs, sp_a[bs])) / 2 for bs in sp_a}}
    candidates, aliases, controls = {}, {}, {}
    for d in sorted(p for p in cell.glob("cand_*") if p.is_dir()):
        name = d.name[len("cand_"):]
        run = load_run(d)
        if run is None or run.get("status") != "ok":
            candidates[name] = {"status": (run or {}).get("status", "missing"), "error": (run or {}).get("error")}
            continue
        sig, fell_back = kernel_signature(d)
        stats = {"status": "ok", "kernels": sorted(sig), **compare_pair(ref_a, load_outputs(d), floor_ids)}
        if fell_back or (sig and sig == ref_sig):
            aliases[name] = {**stats, "reason": "fell back" if fell_back else "same kernel as R"}
        elif name in W8A16:
            controls[name] = stats
        else:
            candidates[name] = stats
            speeds[name] = speeds_of(run)
    ok = {n: c for n, c in candidates.items() if c.get("status") == "ok"}
    safe = {"R"} | {n for n, c in ok.items() if c["flipped"] == 0}
    unsafe = {n for n, c in ok.items() if c["flip_fraction"] >= UNSAFE}
    batch_sizes = sorted({bs for s in speeds.values() for bs in s}, key=int)

    def ranked(names: set[str], bs: str) -> list[tuple[str, float]]:
        vals = [(n, speeds[n][bs]) for n in names if bs in speeds.get(n, {})]
        return sorted(vals, key=lambda x: -x[1])

    flat_spread, winners, tradeoff = {}, {}, {}
    for bs in batch_sizes:
        threshold = max(FLAT_SPEED_MIN, 2 * noise.get(bs, 0.0))
        rs = ranked(safe, bs)
        if len(rs) >= 2:
            flat_spread[bs] = {"spread": (rs[0][1] - rs[-1][1]) / rs[0][1], "threshold": threshold}
            margin = rs[0][1] / rs[1][1] - 1
            winners[bs] = {"backend": rs[0][0], "margin": margin}
        ru = ranked(unsafe, bs)
        if rs and ru and ru[0][1] / rs[0][1] - 1 >= MARGIN:
            tradeoff[bs] = {"unsafe": ru[0][0], "safe": rs[0][0], "gain": ru[0][1] / rs[0][1] - 1}
    clear = {bs: w["backend"] for bs, w in winners.items() if w["margin"] >= MARGIN}
    flat = (
        bool(ok)
        and all(c["flipped"] == 0 and c["wilson95"][1] <= FLAT_UPPER for c in ok.values())
        and bool(flat_spread)
        and all(v["spread"] < v["threshold"] for v in flat_spread.values())
    )
    cont = bool(unsafe) and (len(set(clear.values())) >= 2 or bool(tradeoff))
    static = (
        len(clear) == len(batch_sizes) > 0 and len(set(clear.values())) == 1 and not tradeoff
    )
    return {
        "reference_agreement": agreement,
        "floor_prompts": len(floor_ids),
        "reference_speed_noise": noise,
        "V1": agreement >= V1_AGREEMENT,
        "V2": len(ok) >= 1,
        # Reported only: with no selection line in R_a, alias detection cannot work (check this first).
        "reference_kernels": sorted(ref_sig),
        "kernel_log_found": bool(ref_sig),
        "candidates": candidates,
        "aliases_negative_control": aliases,
        "w8a16_positive_control": controls,
        "speeds": speeds,
        "safe": sorted(safe),
        "unsafe": sorted(unsafe),
        "fastest_safe_by_batch": winners,
        "unsafe_faster_than_safe": tradeoff,
        "flat_spread_by_batch": flat_spread,
        "flat": flat,
        "continue": cont,
        "static_winner": static,
    }


def session_decision(cells: dict[str, dict]) -> str:
    values = list(cells.values())
    if not all(c["V2"] for c in values):
        return "INCONCLUSIVE (V2: a cell has no working alternative backend)"
    if not all(c["V1"] for c in values):
        return "INCONCLUSIVE (V1: the reference is not reproducible)"
    controls = [c for cell in values for c in cell["w8a16_positive_control"].values()]
    pc_ok = bool(controls) and all(c["flip_fraction"] >= PC_MIN for c in controls)
    if any(c["continue"] for c in values):
        return "CONTINUE"
    if all(c["flat"] for c in values):
        return "STOP-FLAT" if pc_ok else "INCONCLUSIVE (flat, but the W8A16 positive control did not flip)"
    if any(c["static_winner"] for c in values):
        return "STATIC-WINNER"
    return "INCONCLUSIVE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", nargs="+", required=True, help="the FP8 cell directories")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cells = {Path(c).name: summarize_cell(Path(c)) for c in args.cells if Path(c).exists()}
    missing = [c for c in args.cells if not Path(c).exists()]
    result = {"cells": cells, "missing_cells": missing}
    result["decision"] = session_decision(cells) if cells and not missing else "INCONCLUSIVE (cell missing)"
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(result["decision"])


if __name__ == "__main__":
    main()
