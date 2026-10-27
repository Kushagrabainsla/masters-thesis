"""E-38 analysis: validity V1 to V5, ACCEPTABLE (A1, A2), HEADROOM, and the frozen decision (README 6, 7).

Cell layout:
  <cell>/R_a, R_b, cand_<name>, bf16    run_a2.py full runs (completions.jsonl, run.json, tf.json)
  <cell>/speed/<name>/R_1, C_1, R_2, C_2 run_cell.py throughput runs (order R, C, R, C)

Usage:
  analyze_a2.py accept --cell <cell dir> --out accept.json     (A1 and A2 only; the session uses it)
  analyze_a2.py decide --cells <D> <M> --out summary.json       (everything, and the decision)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ledger"))

from compare_runs import median_ratio_interval, verdict  # noqa: E402

MARGIN = 0.02        # A1: non-inferiority margin, 2.0 percentage points
V1_MIN = 0.99
V2_MIN = 0.80
V3_KL, V3_DIS = 1e-4, 0.001
V4_DIS = 0.001
MDE = 0.03
A2_RATIO = 0.5       # A1 amendment (R-43): C's deviation from R_a at most half of the BF16 anchor's
BOOT = 10000


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def paired_interval(ref: dict, cand: dict, n: int = BOOT, seed: int = 0) -> tuple[float, float, float]:
    ids = [i for i in ref if i in cand]
    diffs = [int(cand[i]["correct"]) - int(ref[i]["correct"]) for i in ids]
    if not diffs:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    k = len(diffs)
    draws = sorted(sum(rng.choices(diffs, k=k)) / k for _ in range(n))
    return sum(diffs) / k, draws[int(0.025 * n)], draws[int(0.975 * n)]


def accuracy(rows: dict) -> float | None:
    return sum(r["correct"] for r in rows.values()) / len(rows) if rows else None


def accept(cell: Path) -> dict:
    ref = load_jsonl(cell / "R_a" / "completions.jsonl")
    bf16_tf = load_json(cell / "bf16" / "tf.json") or {}
    bf16_rows = load_jsonl(cell / "bf16" / "completions.jsonl")
    rb_rows = load_jsonl(cell / "R_b" / "completions.jsonl")
    out = {"bf16_kl": bf16_tf.get("kl_mean"), "bf16_top1": bf16_tf.get("top1_disagree_rate"),
           # context (R-43): the BF16 accuracy difference, and R_b as a null check of A1 (should be 0)
           "bf16_accuracy_diff": paired_interval(ref, bf16_rows) if bf16_rows else None,
           "R_b_accuracy_diff": paired_interval(ref, rb_rows) if rb_rows else None,
           "anchor_missing": bf16_tf.get("kl_mean") is None, "errored": [], "candidates": {}}
    for d in sorted(p for p in cell.glob("cand_*") if p.is_dir()):
        name = d.name[len("cand_"):]
        run = load_json(d / "run.json") or {}
        if run.get("status") != "ok":
            out["candidates"][name] = {"status": run.get("status", "missing")}
            out["errored"].append(name)
            continue
        rows = load_jsonl(d / "completions.jsonl")
        tf = load_json(d / "tf.json") or {}
        dp = load_json(d / "decode_prefix.json") or {}
        diff, lo, hi = paired_interval(ref, rows)
        a1 = lo > -MARGIN
        a2 = (not out["anchor_missing"] and tf.get("kl_mean") is not None and tf.get("top1_disagree_rate") is not None
              and tf["kl_mean"] <= A2_RATIO * out["bf16_kl"]
              and tf["top1_disagree_rate"] <= A2_RATIO * (out["bf16_top1"] or 0))
        out["candidates"][name] = {
            "status": "ok", "accuracy": accuracy(rows), "diff": diff, "interval95": [lo, hi],
            "tf_kl_mean": tf.get("kl_mean"), "tf_kl_median": tf.get("kl_median"), "tf_kl_p99": tf.get("kl_p99"),
            "tf_top1_disagree_rate": tf.get("top1_disagree_rate"),
            "decode_prefix_kl_mean": dp.get("kl_mean"), "decode_prefix_top1_disagree_rate": dp.get("top1_disagree_rate"),
            "A1": a1, "A2": a2, "acceptable": a1 and a2,
        }
    out["acceptable"] = sorted(n for n, c in out["candidates"].items() if c.get("acceptable"))
    return out


def decode_values(run: dict) -> dict[str, list[float]]:
    tp = (run or {}).get("throughput") or {}
    n_tok = ((run or {}).get("args") or {}).get("throughput_max_tokens", 128)
    return {bs: [int(bs) * int(n_tok) / s for s in v["seconds"] if s > 0] for bs, v in (tp.get("decode") or {}).items()}


def headroom(speed_dir: Path) -> dict:
    """R, C, R, C: baseline = both R runs, candidate = both C runs, band = max(MDE, R_1 against R_2)."""
    runs = {n: decode_values(load_json(speed_dir / n / "run.json")) for n in ("R_1", "C_1", "R_2", "C_2")}
    sizes = sorted(set.intersection(*(set(v) for v in runs.values())) if all(runs.values()) else set(), key=int)
    level = 1 - 0.05 / max(1, len(sizes))
    by_batch = {}
    for bs in sizes:
        base = runs["R_1"][bs] + runs["R_2"][bs]
        cand = runs["C_1"][bs] + runs["C_2"][bs]
        band = max(MDE, abs(statistics.median(runs["R_2"][bs]) / statistics.median(runs["R_1"][bs]) - 1))
        point, lo, hi = median_ratio_interval(base, cand, level)
        by_batch[bs] = {"ratio": round(point, 4), "interval": [round(lo, 4), round(hi, 4)], "band": round(band, 4),
                        "verdict": verdict(lo, hi, band, True)}
    return {"by_batch": by_batch, "better_at": [bs for bs, v in by_batch.items() if v["verdict"] == "BETTER"],
            "complete": bool(sizes)}


def validity(cell: Path, has_alias: bool) -> dict:
    a, b = load_jsonl(cell / "R_a" / "completions.jsonl"), load_jsonl(cell / "R_b" / "completions.jsonl")
    ids = [i for i in a if i in b]
    v1 = sum(a[i]["token_ids"] == b[i]["token_ids"] for i in ids) / len(ids) if ids else 0.0
    rb = load_json(cell / "R_b" / "tf.json") or {}
    bf = load_json(cell / "bf16" / "tf.json") or {}
    al = load_json(cell / "cand_triton" / "tf.json") if has_alias else None
    out = {
        "V1_agreement": v1, "V1": v1 >= V1_MIN,
        "V2_accuracy": accuracy(a), "V2": (accuracy(a) or 0) >= V2_MIN,
        # 0.0 is a valid rate: test for None, not for falsy (a unit test found this).
        "V3": rb.get("kl_mean") is not None and rb["kl_mean"] < V3_KL
              and rb.get("top1_disagree_rate") is not None and rb["top1_disagree_rate"] < V3_DIS,
        "V3_values": [rb.get("kl_mean"), rb.get("top1_disagree_rate")],
        "V5": bf.get("kl_mean") is not None and rb.get("kl_mean") is not None and bf["kl_mean"] > rb["kl_mean"],
        "V5_values": [bf.get("kl_mean"), rb.get("kl_mean")],
    }
    if has_alias:
        out["V4"] = al is not None and (al.get("top1_disagree_rate") if al.get("top1_disagree_rate") is not None else 1) < V4_DIS
        out["V4_value"] = (al or {}).get("top1_disagree_rate")
    return out


def decide(cells: dict[str, dict], speed_plan: dict[str, list[str]] | None = None) -> str:
    """A1 amendment (R-43). A cell that fails V1 to V5 is INVALID for that cell only. GO needs HEADROOM in
    a valid cell. KILL needs both cells valid, no errored candidate, no missing anchor, and the planned
    speed tests complete with no HEADROOM. speed_plan names the candidates whose speed is measured
    (README Section 9); the others are judged from E-36 data (slower than R), stated in the README."""
    invalid = {n: [k for k in ("V1", "V2", "V3", "V4", "V5") if k in c["validity"] and not c["validity"][k]]
               for n, c in cells.items()}
    invalid = {n: f for n, f in invalid.items() if f}
    valid = {n: c for n, c in cells.items() if n not in invalid}
    head = [(n, k) for n, c in valid.items() for k, h in c["headroom"].items() if h["better_at"]]
    if head:
        return "GO (" + ", ".join(f"{n}/{k}" for n, k in head) + ")"
    if invalid and not valid:
        return "INVALID (" + "; ".join(f"{n}: {', '.join(f)}" for n, f in invalid.items()) + ")"
    problems = [f"{n} invalid ({', '.join(f)})" for n, f in invalid.items()]
    for n, c in valid.items():
        if c["accept"]["errored"]:
            problems.append(f"{n} errored: {', '.join(c['accept']['errored'])}")
        if c["accept"]["anchor_missing"]:
            problems.append(f"{n} BF16 anchor missing")
        for k in (speed_plan or {}).get(n, []):
            if k in c["accept"]["acceptable"] and not c["headroom"].get(k, {}).get("complete"):
                problems.append(f"{n}/{k} speed missing")
    if problems:
        return "INCONCLUSIVE (" + "; ".join(problems) + ")"
    faster_not_ok = [(n, k) for n, c in valid.items() for k, h in c.get("headroom_unacceptable", {}).items() if h["better_at"]]
    if faster_not_ok:
        return "PIVOT (" + ", ".join(f"{n}/{k}" for n, k in faster_not_ok) + " faster but not acceptable)"
    return "KILL"


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("accept")
    a.add_argument("--cell", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    d = sub.add_parser("decide")
    d.add_argument("--cells", nargs="+", required=True, type=Path)
    d.add_argument("--out", required=True, type=Path)
    d.add_argument("--speed-plan", default="cell_M:marlin", help="cell:cand,cand;cell:cand (README Section 9)")
    args = parser.parse_args()
    if args.cmd == "accept":
        result = accept(args.cell)
        args.out.write_text(json.dumps(result, indent=1))
        print(" ".join(result["acceptable"]))
        return
    cells = {}
    for cell in args.cells:
        acc = accept(cell)
        speed = cell / "speed"
        heads = {k: headroom(speed / k) for k in acc["acceptable"] if (speed / k).is_dir()}
        others = {p.name: headroom(p) for p in sorted(speed.glob("*")) if p.is_dir() and p.name not in acc["acceptable"]} if speed.is_dir() else {}
        cells[cell.name] = {"validity": validity(cell, cell.name.endswith("M")), "accept": acc,
                            "headroom": heads, "headroom_unacceptable": others}
    plan = {}
    for part in filter(None, args.speed_plan.split(";")):
        name, _, cands = part.partition(":")
        plan[name] = [c for c in cands.split(",") if c]
    result = {"cells": cells, "speed_plan": plan, "decision": decide(cells, plan)}
    args.out.write_text(json.dumps(result, indent=1, default=str))
    print(result["decision"])


if __name__ == "__main__":
    main()
