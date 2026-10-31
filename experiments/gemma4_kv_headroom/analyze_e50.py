"""E-50 analysis (README Sections 5 to 7): the compact K=V headroom on top of an FP8 KV cache, at a large batch.

Layout under the results root:
  m31f8/   decode_probe.py outputs: Gemma 4 31B FP8 with an FP8 KV cache (the deciding arm)
  m31bf/   the same model with the BF16 KV cache (secondary)
  m26/     Gemma 4 26B-A4B BF16 with the BF16 KV cache: the E-49 cells with the fixed token budget (secondary)
  env-*/   telemetry.csv and microbench_*.json (copy GB/s)

Mechanism of the metric (the E-49 metric at another batch): at a fixed batch B, the decode step time grows with the
context L only through the global-layer attention, so a cell-level line T(L) = c + b L over the cells of one batch
gives b, and the compact cache saves at most S = 0.375 b_low L / T(L) of the step. The pair cells of group X run at
the batch that fills the share --auto-frac of the idle KV pool at context X (32K or 64K). S grows with B, so S at
that batch is a lower bound of S at the pool limit. The 2K and X profiles of a group check the premise (V4a, V4b,
V4c) and give S_att, a device-time estimate from the layer split.

Usage: analyze_e50.py --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gemma4_kv_audit"))

import analyze_e49 as a49  # noqa: E402  (it also puts the reused folders on sys.path)

ARMS = {  # global KV bytes per token and sequence (K and V of the global layers), slope batch, layers, KV cache
    "m31f8": {"name": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "global_bytes": 40960, "batch": 4, "layers": 60, "kv": "fp8"},
    "m31bf": {"name": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "global_bytes": 81920, "batch": 4, "layers": 60, "kv": "auto"},
    "m26": {"name": "google/gemma-4-26B-A4B-it", "global_bytes": 20480, "batch": 8, "layers": 30, "kv": "auto"},
}
a49.MODELS.update({k: v for k, v in ARMS.items() if k not in a49.MODELS})
DECIDING = "m31f8"
GROUPS = [32768, 65536]
GO_S, KILL_S = 0.05, 0.03
SATT_GAP = 0.25
TREATMENT = {  # V11: the engine ran the planned KV cache and attention path (FACT from the vLLM source, README 4)
    "fp8": [re.compile(r"kv_cache_dtype=fp8"), re.compile(r"per-layer FA3/FA4 selection for FP8 KV cache")],
    "auto": [re.compile(r"kv_cache_dtype=auto"), re.compile(r"Using FA4 for all layers")],
}


def spec_of(c: dict) -> str:
    return c.get("spec") or a49.key_of(c)


def split_of(d: Path, t: dict | None, types: list[str]) -> dict:
    """The launch-order layer split of E-49, else a count split: an attention kernel name launched once per
    sliding layer (or once per global layer) in each step belongs to those layers. The count split covers FA3 and
    FA4 kernels side by side: with an FP8 KV cache on SM90, vLLM runs FA3 on the sliding layers and FA4 on the
    global layers."""
    if not t or t.get("lost"):
        return {"ok": False, "why": "no profile"}
    steps = t.get("steps") or 1
    ev = a49.load(d / t["events"], None) if t.get("events") else None
    s = a49.layer_split(ev, steps, types)
    if s["ok"]:
        return s | {"method": "order"}
    n_sl = sum("sliding" in x for x in types)
    n_gl = len(types) - n_sl
    ms = {"sliding": 0.0, "global": 0.0, "other": 0.0}
    for k in t.get("kernels", []):
        if a49.is_attention(k["name"]) and not a49.PREP.search(k["name"]):
            per = k["count"] / steps
            part = "sliding" if per == n_sl else "global" if per == n_gl else "other"
            ms[part] += k["device_us"] / steps / 1000.0
    total = sum(ms.values())
    if not total or not ms["global"] or ms["other"] > 0.05 * total:
        return {"ok": False, "why": f"{s['why']}; count split: {ms['other']:.4f} of {total:.4f} ms not assigned"}
    return {"ok": True, "method": "count", "sliding_ms": round(ms["sliding"], 4), "global_ms": round(ms["global"], 4)}


def group_summary(d: Path, cells: list[dict], tables: dict, types: list[str], gbytes: int, bw: float | None,
                  free_idle: int | None, X: int) -> dict:
    """The pair cells of group X (batch 'aX'): the cell-level fit, S at X, the group checks and S_att."""
    mine = [c for c in cells if c.get("kind") == "pair" and c.get("auto_for") == X]
    used = [c for c in mine if c.get("status") == "ok" and c.get("step_ms")]
    out: dict = {"lost": [spec_of(c) for c in mine if c not in used], "cells": len(used),
                 "finished": {spec_of(c): c["finished"] for c in mine if c.get("finished")}}
    batches = sorted({c["batch"] for c in used})
    ctxs = sorted({c["context"] for c in used})
    if len(batches) != 1 or X not in ctxs or len(ctxs) < 3:
        return out | {"ok": False, "why": f"batches {batches}, contexts {ctxs}: the fit needs one batch, 3 contexts and {X}"}
    B = batches[0]
    pts = [(c["context"], statistics.median(c["step_ms"])) for c in used]
    f = a49.fit(pts)
    med = {L: statistics.median([m for x, m in pts if x == L]) for L in ctxs}
    resid = {L: (med[L] - (f["c"] + f["b"] * L)) / med[L] for L in ctxs}
    lo, hi = ctxs[-2:]   # M2: the slope over the two longest contexts against the fitted slope
    local = (med[hi] - med[lo]) / (hi - lo) / f["b"] if f["b"] > 0 else None
    steps = used[0]["steps_per_block"]
    spread = {spec_of(c): max(c["step_ms"]) / min(c["step_ms"]) - 1 for c in used}
    bw_g = gbytes * B / (f["b"] / 1000.0) / 1e9 if f["b"] > 0 else None
    sure = f["b_low"] > 0 and bw_g is not None
    att: dict = {"V4a": None, "V4b": None, "V4c": None}
    T = med[X]
    lo_t, hi_t = tables.get(f"2048:a{X}"), tables.get(f"{X}:a{X}")
    lo_c = next((c for c in used if spec_of(c) == f"2048:a{X}"), None)
    hi_c = next((c for c in used if spec_of(c) == f"{X}:a{X}"), None)
    s_att = None
    if lo_t and hi_t and lo_c and hi_c and not lo_t.get("lost") and not hi_t.get("lost"):
        dd = a49.dev_ms(hi_t) - a49.dev_ms(lo_t)
        dw = statistics.median(hi_c["step_ms"]) - statistics.median(lo_c["step_ms"])   # unprofiled medians (R-64 R1)
        a_hi = a49.att_ms(hi_t)
        da = a_hi - a49.att_ms(lo_t)
        att.update(device_growth_ms=round(dd, 4), wall_growth_ms=round(dw, 4), attention_growth_ms=round(da, 4),
                   V4a=dw > 0 and dd / dw >= a49.V4_FRAC,
                   V4b=None if a_hi == 0 else dd > 0 and da / dd >= a49.V4_FRAC)
        s0, s1 = split_of(d, lo_t, types), split_of(d, hi_t, types)
        att["split"] = {"2048": s0, str(X): s1}
        if s0["ok"] and s1["ok"] and s0["sliding_ms"] > 0:
            change = s1["sliding_ms"] / s0["sliding_ms"] - 1
            att.update(sliding_change=round(change, 4), V4c=abs(change) <= a49.V4C_TOL)
            s_att = a49.SAVING * (s1["global_ms"] - s0["global_ms"]) / (X - 2048) * X / T
    per_req = next((c["blocks_per_request"] for c in cells if c["context"] == X and c.get("blocks_per_request")), None)
    b_max = free_idle / per_req if free_idle and per_req else None
    checks = {"V2": all(a >= steps - 1 and z <= steps + 1 for c in used for a, z in c["advance_min_max"]),
              "V3": all(abs(r) <= a49.V3_RESID for r in resid.values()),
              "V4a": not sure or att["V4a"] is True,
              "V5": bw is not None and (not sure or bw_g <= a49.M1_MAX * bw),
              "V7": sum(s > a49.V7_SPREAD for s in spread.values()) <= 1}
    S = a49.SAVING * f["b"] * X / T
    return out | {
        "ok": True, "batch": B, "B_max_est": round(b_max, 2) if b_max else None,
        "pool_share": round(B / b_max, 3) if b_max else None, "fit": f, "median_ms": med,
        "residuals": {str(k): round(v, 4) for k, v in resid.items()}, "spread": {k: round(v, 4) for k, v in spread.items()},
        "checks": checks, "valid": all(checks.values()), "why": [k for k, v in checks.items() if not v],
        "local_slope_ratio": round(local, 4) if local is not None else None,
        "M1": sure and bw is not None and bw_g >= a49.M1_FRAC * bw,
        "M2": None if local is None else abs(local - 1) <= a49.M2_LOCAL,
        "bw_global_gbps": round(bw_g, 1) if bw_g else None, "attribution": att, "V4b": att["V4b"], "V4c": att["V4c"],
        "T_ms": round(T, 4), "S": round(S, 4), "S_low": round(a49.SAVING * f["b_low"] * X / T, 4),
        "S_up": round(a49.SAVING * f["b_up"] * X / T, 4),
        "S_att": round(s_att, 4) if s_att is not None else None,
        "S_att_gap": s_att is not None and abs(s_att - S) > SATT_GAP * abs(S)}


def arm_summary(res: Path, key: str, bw: float | None) -> dict:
    d, spec = res / key, ARMS[key]
    run = a49.load(d / "run.json", {})
    if run.get("status") not in ("ok", "ok_pairs", "ok_slope"):
        return {"status": run.get("status", "missing"), "valid": False, "why": ["run not ok"],
                "error": run.get("error", "")[:300], "groups": {}}
    base = a49.model_summary(res, key, bw)   # the slope cells at batch B: V1 to V9 and S at B (E-49 metric)
    if base.get("local_slope_ratio") is None:
        base["M2"] = None   # R-65 C3: no 64K cell means curvature unknown (E-49 counted it as true)
    cells, tables = a49.load(d / "cells.json", []), a49.load(d / "kernels.json", {})
    types = (run.get("model") or {}).get("layer_types") or \
        (["sliding_attention"] * 5 + ["full_attention"]) * (spec["layers"] // 6)
    log = (d / "log.txt").read_text(errors="replace") if (d / "log.txt").exists() else ""
    base["checks"]["V11"] = all(rx.search(log) for rx in TREATMENT[spec["kv"]])
    groups = {}
    for X in GROUPS:
        if any(c.get("auto_for") == X for c in cells):
            groups[str(X)] = g = group_summary(d, cells, tables, types, spec["global_bytes"], bw,
                                               run.get("free_blocks_idle"), X)
            if g.get("ok"):
                base["checks"][f"G{X}"] = g["valid"]
    base["why"] = [k for k, v in base["checks"].items() if not v]
    base["valid"] = not base["why"]
    base["groups"] = groups
    base["status_cells"] = {spec_of(c): c.get("status") for c in cells}
    base["e49_pass"] = {L: c["S_low"] >= GO_S and base["M1"] and base["M2"] is True and base["V4b"] is True
                        and base["V4c"] is True for L, c in base["decide"].items()}
    return base


def decide(s: dict) -> str:
    if s.get("copy_gbps") is None:
        return "INVALID (no copy bandwidth: the M1 and V5 controls cannot run)"
    if not s["V8"]:
        return "INVALID (V8 telemetry)"
    p = s["arms"][DECIDING]
    if p.get("status") not in ("ok", "ok_pairs"):
        return f"INCONCLUSIVE (the FP8 arm did not finish its pair cells: status {p.get('status')}; {p.get('error', '')[:200]})"
    if not p["valid"]:
        return f"INVALID ({DECIDING}: {', '.join(p['why'])})"
    rows = {X: g for X, g in p["groups"].items() if g.get("ok")}
    tag = "; ".join(f"{X}: S {g['S']:.3f} (low {g['S_low']:.3f}, up {g['S_up']:.3f}) at B={g['batch']}" for X, g in rows.items())
    flags = [X for X, g in rows.items() if g["S_att_gap"]]
    note = f" [S_att gap: {', '.join(flags)}]" if flags else ""
    gate = {X: g["M1"] and g["M2"] is True and g["V4b"] is True and g["V4c"] is True for X, g in rows.items()}
    passing = [X for X, g in rows.items() if g["S_low"] >= GO_S and gate[X]]
    if passing:
        return f"GO ({tag}){note}"
    lost = [str(X) for X in GROUPS if str(X) not in rows]
    if lost:
        return f"INCONCLUSIVE (deciding group lost: {', '.join(lost)}; {tag}){note}"
    if all(g["S_up"] < KILL_S for g in rows.values()):
        return f"KILL ({tag}){note}"
    reasons = []
    for X, g in rows.items():
        if g["S_low"] < GO_S:
            continue
        if not g["M1"]:
            reasons.append(f"{X} KERNEL-BOUND (M1)")
        if g["M2"] is not True:
            reasons.append(f"{X} curvature (M2: local slope ratio {g['local_slope_ratio']})")
        if g["V4b"] is None or g["V4c"] is None:
            reasons.append(f"{X} attribution unknown")
        elif not (g["V4b"] and g["V4c"]):
            reasons.append(f"{X} attribution unclear (V4b {g['V4b']}, V4c {g['V4c']})")
    if not any(g["S_low"] >= GO_S for g in rows.values()):
        reasons.append("below the 5% bar")
    return f"PIVOT ({'; '.join(reasons)}; {tag}){note}"


def summarize(res: Path) -> dict:
    bw = a49.copy_gbps(res)
    arms = {k: arm_summary(res, k, bw) for k in ARMS}
    f8, bf = arms["m31f8"].get("median_ms") or {}, arms["m31bf"].get("median_ms") or {}
    saving = {str(L): round(1 - f8[L] / bf[L], 4) for L in f8 if L in bf}   # same Pod, B = 4 (secondary)
    return {"arms": arms, "V8": a49.telemetry_ok(res), "copy_gbps": bw, "fp8_saving_B4": saving}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    s = summarize(a.res)
    s["decision"] = decide(s)
    if a.out:
        a.out.write_text(json.dumps(s, indent=1, default=str))
    print(s["decision"])


if __name__ == "__main__":
    main()
