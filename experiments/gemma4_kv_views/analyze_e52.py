"""E-52 analysis (README Sections 5 to 7, with amendment A1): with fixed kernels (FP16 compute; the compact row read by
the standard kernel through two overlapping views), does the compact K=V layout cut the context-growing global
attention time of Gemma 4 31B decode enough, against its standard twin and the best stock FP8 kernel in vLLM?

Layout under the results root (as E-51):
  kbench/kbench.json   "cells": the sweep; "correctness": the correctness cell; "confirm": the confirmation rows
  kbench/run.json      GPU, versions, status ("ok_groups" once the decision data is complete, "ok" at the end)
  kbench/asm/          PTX, TTGIR and cubin of each Triton kernel at the E-51 configuration
  compile_check.json   the sm_90 compile check (A1, R-71 F2; optional)
  dram_probe.txt       the ncu DRAM bytes attempt (A1, R-71 F5; optional)
  env-*/telemetry.csv  nvidia-smi samples;  env-*/microbench_*.json  copy GB/s

Families: stock (the stock FA4 FP8 dequant kernel, num_splits 32 or automatic), triton_std (`triton_std_fp8h`: the
standard Triton kernel with FP16 compute) and triton_cmp (`triton_cmpv_fp8h`: the same kernel on the views layout).
A1 (R-71 F1): the views kernel runs at the configuration that the standard family selects, so sigma_layout compares one
compiled kernel at one configuration on the two layouts; the views kernel's own selection is reported.
dt(X) = t(X) - t(2,048) at the group batch; sigma_dq = 1 - dt_cmp / dt_stock; sigma_mma = 1 - dt_cmp /
(0.85 dt_stock); sigma_layout = 1 - dt_cmp / dt_std. Kill lines 0.155 (32K) and 0.125 (64K); go line 0.25 (R-68).
The views kernel saves DRAM bytes only, so for a kernel with K = V elasticity e, sigma_layout is about 0.375 x e: the
kill lines need e of 0.413 (32K) and 0.333 (64K).
A1 (R-71 B1): no KILL from a competence test of the views kernel. The rule names three causes when sigma_layout is
below the kill lines: (a) time does not follow bytes (e_h below 0.413 and 0.333), (b) the views do not share their
DRAM reads (e_h 0.6 or more, sigma_layout below 0.25 x e_h), (c) the L2 path or the on-chip work caps (pool share of
the views kernel below 0.3). KILL (byte route only) needs (a) in both groups and a valid K = V oracle (the BF16 control
e_bf16_triton 0.5 or more). Every other case is INCONCLUSIVE with its flags.
A1 (R-71 B2): V4 bounds the Triton decision rows (central spread 2%, dt range 3%) and the stock dt range (5%).
A1 (R-71 F8): the word is computed again with the E-51 stock dt as the baseline; if the two words differ, the result
is INCONCLUSIVE (baseline off).
Reported only: the stock kernel on the views, the K = V elasticities, the pool shares, the cast x stage table, the cast
x layout factorial, BF16, the L2 ratios, the linearity, the PTX counts and identity, the compile check and ncu.

Usage: analyze_e52.py --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("fp8_quant_fusion", "nvfp4_dispatch", "safedispatch_testA", "safedispatch_testA2", "ledger"):
    sys.path.insert(0, str(HERE.parent / sub))

from analyze_e44 import telemetry_ok  # noqa: E402  (hard throttle in at most 2% of busy samples)

GROUPS = {32768: 19, 65536: 11}               # the E-50 pair batches
KILL = {32768: 0.155, 65536: 0.125}           # S = 5% (R-68)
E_LINE = {X: round(v / 0.375, 4) for X, v in KILL.items()}   # 0.4133 and 0.3333: e that reaches the kill line
GO_SIGMA = 0.25                               # S about 8% (R-68)
MMA_GAIN = 0.15                               # FA PR 199: FP8 MMA 8% to 15% faster on the context part (R-68)
TOL_STOCK = 0.02                              # relative L2 error against the reference of the stored values
TOL_TRITON = 0.01                             # each Triton configuration (E-51 A1, R5)
V3_TOL = 0.15                                 # stock FP8 (32 splits) against the E-50 in-engine time at X and dt
SPREAD_MAX = 0.02                             # the central spread of a Triton decision row (median over the repeats)
DT_RANGE_MAX = 0.03                           # the across-repeat range of a Triton primary family's dt (R-70)
DT_RANGE_STOCK = 0.05                         # A1 (R-71 B2): the same for the stock family
BW_MAX = 1.15                                 # a kernel cannot read faster than 1.15 x the copy bandwidth
MEM_BOUND = 0.5                               # the Triton standard kernel must read at least 0.5 x copy on dt
E_SHARE, SHARE_FRAC = 0.6, 0.25               # flag (b): e_h >= 0.6 and sigma_layout < 0.25 x e_h
POOL_CAP = 0.3                                # flag (c): the views pool share below 0.3
E_BF16_MIN = 0.5                              # the K = V oracle is valid on this GPU (R-70 control)
CLOCK_FLAG = 1900                             # a decision row with a mean SM clock below this is flagged (reported)
L2_FLAG = 0.03                                # A1 (R-71 F1): views/std at 11:2K this much below 19:2K is flagged
REPEATS = 5                                   # confirmation repeats (MEASUREMENT.md Section 1.5)
AGREE = 4                                     # the decision must come out the same in at least 4 of the 5 repeats
OK_STATUS = ("ok", "ok_groups")
TARGETS = {(19, 2048): 238.3, (19, 32768): 1358.1, (11, 2048): 143.2, (11, 65536): 1507.2}
TARGETS_STR = {f"{b}:{length}": v for (b, length), v in TARGETS.items()}
E51_STOCK_DT = {32768: 1147.8, 65536: 1344.93}   # A1 (R-71 F8): the E-51 H100 stock s32 dt (us)
FAMILIES = {"stock": ("stock_fp8_s32", "stock_fp8_auto"), "triton_std": ("triton_std_fp8h",),
            "triton_cmp": ("triton_cmpv_fp8h",)}
BF16_FAMILIES = {"stock": ("stock_bf16_s32", "stock_bf16_auto"), "triton_std": ("triton_std_bf16",),
                 "triton_cmp": ("triton_cmpv_bf16",)}
ORACLE = {"stock_fp8_s32_alias": "stock_fp8_s32", "triton_std_fp8h_alias": "triton_std_fp8h",
          "stock_bf16_s32_alias": "stock_bf16_s32", "triton_std_bf16_alias": "triton_std_bf16"}
POOL = {"stock_fp8_s32_pool": "stock_fp8_s32", "triton_std_fp8h_pool": "triton_std_fp8h",
        "triton_cmpv_fp8h_pool": "triton_cmpv_fp8h", "triton_std_bf16_pool": "triton_std_bf16"}
DIAG = ("triton_std_fp8", "triton_std_fp8h", "triton_cmpv_fp8", "triton_cmpv_fp8h", "triton_cmp_fp8",
        "triton_cmp_fp8h")                    # the cast x layout factorial at the E-51 configuration
CAST_STAGE = ("triton_std_fp8h", "triton_std_fp8")   # A1 (R-71 F3): FP16 and BF16 compute at stages 2 and 3
E51 = {"splits": 32, "warps": 4, "tile": 32, "stages": 2}
S32 = {"num_splits": 32}
COMPACT_WORDS = ("cmpv", "cmp_", "_views")    # kernels that read 640 values per token
CONSEQUENCE = {
    "a": "time does not follow bytes: the byte route is closed for the Triton FP16 family; a kernel that stages V once "
         "is not tested",
    "b": "the views do not share their DRAM reads: the next kernel must read each 640-value row once",
    "c": "the L2 path or the on-chip work caps the views kernel: the next step is a profile",
}


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def copy_gbps(res: Path) -> float | None:
    vals = [load(p, {}).get("copy_gb_per_s_median") for p in sorted(res.glob("env-*/microbench_*.json"))]
    vals = [v for v in vals if v]
    return statistics.median(vals) if vals else None


def cfg_key(c: dict) -> str:
    return f'{c["splits"]}/{c["warps"]}/{c["tile"]}/{c["stages"]}'


def key(kernel: str, b: int, length: int, config: dict) -> tuple:
    return kernel, b, length, json.dumps(config, sort_keys=True)


def passed(err, tol: float) -> bool:
    return isinstance(err, (int, float)) and not isinstance(err, bool) and err <= tol


def compact(kernel: str) -> bool:
    return any(w in kernel + "_" for w in COMPACT_WORDS)


def validated(rows: list[dict], corr: dict | None) -> list[dict]:
    """Drop the rows of a configuration that failed the correctness cell (E-51 A1, R5): a Triton configuration whose
    error is missing or above TOL_TRITON, or a stock kernel above TOL_STOCK. A K = V or pool row takes the result of
    its base kernel at the same configuration. Without a correctness cell nothing is dropped (V2 then fails)."""
    if not corr:
        return list(rows)
    keep = []
    for r in rows:
        k = ORACLE.get(r["kernel"]) or POOL.get(r["kernel"]) or r["kernel"]
        entry = (corr.get("fp8" if "fp8" in k else "bf16") or {}).get(k) or {}
        if k.startswith("triton_"):
            ok = passed((entry.get("by_config") or {}).get(cfg_key(r["config"])), TOL_TRITON)
        else:
            ok = passed(entry.get("err"), TOL_STOCK)
        if ok:
            keep.append(r)
    return keep


def deciding_failures(corr: dict | None) -> list[str]:
    """The deciding FP8 kernels that did not pass the correctness cell: the stock kernel with 32 splits (V3 needs it)
    and each primary Triton family without one passing configuration (the fail-fast test)."""
    fp8 = (corr or {}).get("fp8") or {}
    bad = [] if passed((fp8.get("stock_fp8_s32") or {}).get("err"), TOL_STOCK) else ["stock_fp8_s32"]
    for k in FAMILIES["triton_std"] + FAMILIES["triton_cmp"]:
        by = (fp8.get(k) or {}).get("by_config") or {}
        if not any(passed(e, TOL_TRITON) for e in by.values()):
            bad.append(k)
    return bad


def best(rows: list[dict], kernels: tuple[str, ...], b: int, length: int) -> dict | None:
    """The fastest ok configuration of any of `kernels` at (b, length)."""
    ok = [r for r in rows if r["kernel"] in kernels and r["batch"] == b and r["context"] == length
          and r.get("status") == "ok"]
    return min(ok, key=lambda r: r["us"]) if ok else None


def select(rows: list[dict], kernels: tuple[str, ...], X: int) -> tuple[dict, dict] | None:
    """The configuration of a family for group X (E-51 A1, R1): the fastest at X among those that ran at 2K and at X.
    Returns its sweep rows (2K, X)."""
    b = GROUPS[X]
    lo = {key(r["kernel"], b, 0, r["config"]): r for r in rows if r["kernel"] in kernels and r["batch"] == b
          and r["context"] == 2048 and r.get("status") == "ok"}
    pairs = [(lo[key(r["kernel"], b, 0, r["config"])], r) for r in rows if r["kernel"] in kernels and r["batch"] == b
             and r["context"] == X and r.get("status") == "ok" and key(r["kernel"], b, 0, r["config"]) in lo]
    return min(pairs, key=lambda p: p[1]["us"]) if pairs else None


def at_config(rows: list[dict], kernel: str, config: dict, X: int) -> tuple[dict, dict] | None:
    """The sweep rows (2K, X) of one kernel at a fixed configuration (A1, R-71 F1), or None if either is missing."""
    b = GROUPS[X]
    t = {r["context"]: r for r in rows if r["kernel"] == kernel and r["batch"] == b and r["config"] == config
         and r.get("status") == "ok"}
    return (t[2048], t[X]) if 2048 in t and X in t else None


def primary_rows(rows: list[dict], X: int) -> dict:
    """The sweep rows (2K, X) of each primary family: the stock and standard selections, and the views kernel at the
    standard selection (A1, R-71 F1)."""
    std = select(rows, FAMILIES["triton_std"], X)
    return {"stock": select(rows, FAMILIES["stock"], X), "triton_std": std,
            "triton_cmp": at_config(rows, FAMILIES["triton_cmp"][0], std[1]["config"], X) if std else None}


def confirm_plan(rows: list[dict]) -> list[dict]:
    """The rows that the confirmation phase times: per group, at 2K and X, the primary rows, the views kernel's own
    selection, the stock kernel with 32 splits (V3) and on the views, the K = V and pool rows, and (A1, R-71 F3) the
    standard kernel with FP16 and BF16 compute at stages 2 and 3 at the selected splits, warps and tile. `rows` are
    validated sweep rows."""
    plan = []

    def add(kernel, config, b, length):
        p = {"kernel": kernel, "batch": b, "context": length, "config": config}
        if p not in plan:
            plan.append(p)

    for X, b in GROUPS.items():
        prim = primary_rows(rows, X)
        own = select(rows, FAMILIES["triton_cmp"], X)
        for s in list(prim.values()) + [own]:
            for r in s or ():
                add(r["kernel"], r["config"], b, r["context"])
        for length in (2048, X):
            for k in ("stock_fp8_s32", "stock_fp8_s32_views", "stock_fp8_s32_alias", "stock_fp8_s32_pool"):
                add(k, S32, b, length)
            if prim["triton_std"]:
                cfg = prim["triton_std"][1]["config"]
                for k in ("triton_std_fp8h_alias", "triton_std_fp8h_pool", "triton_cmpv_fp8h_pool"):
                    add(k, cfg, b, length)
                for k in CAST_STAGE:
                    for st in (2, 3):
                        add(k, dict(cfg, stages=st), b, length)
    return plan


def conf_index(conf: list[dict] | None) -> dict:
    return {key(e["kernel"], e["batch"], e["context"], e["config"]): e for e in conf or []
            if e.get("status") == "ok" and e.get("repeats")}


def dt_of(ci: dict, kernel: str, config: dict, b: int, X: int) -> tuple[float, list[float]] | None:
    """dt of one configuration from the confirmation: the median of the per-repeat differences, and the list."""
    lo, hi = ci.get(key(kernel, b, 2048, config)), ci.get(key(kernel, b, X, config))
    if not lo or not hi:
        return None
    rep = [round(h - l, 2) for h, l in zip(hi["repeats"], lo["repeats"])]
    return round(statistics.median(rep), 2), rep


def sweep_dt(rows: list[dict], kernel: str, config: dict, b: int, X: int) -> float | None:
    t = {key(r["kernel"], r["batch"], r["context"], r["config"]): r["us"] for r in rows
         if r["kernel"] == kernel and r.get("status") == "ok"}
    lo, hi = t.get(key(kernel, b, 2048, config)), t.get(key(kernel, b, X, config))
    return round(hi - lo, 2) if lo is not None and hi is not None else None


def family_from(sel: tuple[dict, dict] | None, X: int, ci: dict | None) -> dict | None:
    """dt(X) of one family from its sweep rows (2K, X): from the confirmation repeats, or (ci None: BF16, reported
    only) from the one sweep measurement."""
    if not sel:
        return None
    lo, hi = sel
    out = {"kernel": hi["kernel"], "config": hi["config"],
           "sweep": {k: hi.get(k) for k in ("us", "gbps", "regs", "spills", "shared", "clock_mhz")}}
    if ci is None:
        return out | {"dt_us": round(hi["us"] - lo["us"], 2), "dt_rep": [], "t2k_us": lo["us"], "tX_us": hi["us"]}
    d = dt_of(ci, hi["kernel"], hi["config"], GROUPS[X], X)
    if not d:
        return None
    return out | {"dt_us": d[0], "dt_rep": d[1], "t2k_us": ci[key(hi["kernel"], GROUPS[X], 2048, hi["config"])]["us"],
                  "tX_us": ci[key(hi["kernel"], GROUPS[X], X, hi["config"])]["us"]}


def family(rows: list[dict], kernels: tuple[str, ...], X: int, ci: dict | None) -> dict | None:
    return family_from(select(rows, kernels, X), X, ci)


def sigmas(dt_stock: float, dt_std: float, dt_cmp: float, X: int, fp8: bool, bw: float | None) -> dict | None:
    if min(dt_stock, dt_std, dt_cmp) <= 0:
        return None
    b = GROUPS[X]
    grow = lambda cmp: b * (X - 2048) * 4 * (640 if cmp else 1024) * (1 if fp8 else 2)   # bytes on dt
    rate = lambda cmp, dt: grow(cmp) / (dt * 1e-6) / 1e9
    std_bw, cmp_bw = rate(False, dt_std), rate(True, dt_cmp)
    return {"batch": b, "ok": True,
            "sigma_dq": round(1 - dt_cmp / dt_stock, 4),
            "sigma_mma": round(1 - dt_cmp / ((1 - MMA_GAIN) * dt_stock), 4),
            "sigma_layout": round(1 - dt_cmp / dt_std, 4),
            "std_vs_stock": round(dt_std / dt_stock, 4),
            "dt_gbps": {"stock": round(rate(False, dt_stock), 1), "triton_std": round(std_bw, 1),
                        "triton_cmp": round(cmp_bw, 1)},
            "std_memory_bound": bw is not None and std_bw >= MEM_BOUND * bw}


def elasticity(alias, base) -> float | None:
    """e = 2 x (1 - dt_alias / dt): the time elasticity to DRAM bytes (K = V halves them)."""
    a, b = (x[0] if isinstance(x, tuple) else x for x in (alias, base))
    return round(2 * (1 - a / b), 3) if a is not None and b else None


def share(pool, dt) -> float | None:
    p = pool[0] if isinstance(pool, tuple) else pool
    return round(1 - p / dt, 3) if p is not None and dt else None


def bf16_controls(rows: list[dict], X: int) -> dict:
    """The BF16 K = V control (R-70) and the BF16 pool calibration (A1, R-71 F9), from the sweep at the selected
    standard BF16 configuration (reported; e_bf16_triton also gates KILL)."""
    b = GROUPS[X]
    std = select(rows, BF16_FAMILIES["triton_std"], X)
    cfg = std[1]["config"] if std else None
    base = sweep_dt(rows, "triton_std_bf16", cfg, b, X) if cfg else None
    return {"e_bf16_stock": elasticity(sweep_dt(rows, "stock_bf16_s32_alias", S32, b, X),
                                       sweep_dt(rows, "stock_bf16_s32", S32, b, X)),
            "e_bf16_triton": elasticity(sweep_dt(rows, "triton_std_bf16_alias", cfg, b, X), base) if cfg else None,
            "pool_share_bf16": share(sweep_dt(rows, "triton_std_bf16_pool", cfg, b, X), base) if cfg else None}


def group(rows: list[dict], X: int, fp8: bool, bw: float | None, ci: dict | None, bf16c: dict | None = None) -> dict:
    fams = FAMILIES if fp8 else BF16_FAMILIES
    b = GROUPS[X]
    std_sel = select(rows, fams["triton_std"], X)
    fam = {"stock": family(rows, fams["stock"], X, ci), "triton_std": family_from(std_sel, X, ci),
           "triton_cmp": family_from(at_config(rows, fams["triton_cmp"][0], std_sel[1]["config"], X), X, ci)
           if std_sel else None}
    out = {"batch": b, **fam, "triton_cmp_own": family(rows, fams["triton_cmp"], X, ci)}
    m = sigmas(fam["stock"]["dt_us"], fam["triton_std"]["dt_us"], fam["triton_cmp"]["dt_us"], X, fp8, bw) \
        if all(fam.values()) else None
    if not m:
        return out | {"ok": False, "why": "a kernel family is missing, unconfirmed or has no growth"}
    out |= m
    own = out["triton_cmp_own"]
    out["sigma_layout_own"] = round(1 - own["dt_us"] / fam["triton_std"]["dt_us"], 4) if own else None
    if ci is None:                      # BF16: one sweep measurement
        fused = family(rows, ("triton_cmp_bf16",), X, None)
        out["sigma_layout_fused"] = round(1 - fused["dt_us"] / fam["triton_std"]["dt_us"], 4) if fused else None
        return out | bf16_controls(rows, X) | {"repeats": []}
    std_cfg = fam["triton_std"]["config"]
    s32, sv = dt_of(ci, "stock_fp8_s32", S32, b, X), dt_of(ci, "stock_fp8_s32_views", S32, b, X)
    alias = dt_of(ci, "triton_std_fp8h_alias", std_cfg, b, X)
    out["dt_stock_s32"] = s32
    out["dt_stock_s32_views"] = sv
    out["sigma_stock_layout"] = round(1 - sv[0] / s32[0], 4) if s32 and sv and s32[0] > 0 else None
    out["e_stock"] = elasticity(dt_of(ci, "stock_fp8_s32_alias", S32, b, X), s32)
    out["e_h"] = elasticity(alias, fam["triton_std"]["dt_us"])
    out["pool_share"] = {"stock": share(dt_of(ci, "stock_fp8_s32_pool", S32, b, X), s32[0] if s32 else None),
                         "triton_std": share(dt_of(ci, "triton_std_fp8h_pool", std_cfg, b, X),
                                             fam["triton_std"]["dt_us"]),
                         "triton_cmp": share(dt_of(ci, "triton_cmpv_fp8h_pool", std_cfg, b, X),
                                             fam["triton_cmp"]["dt_us"])}
    out["e_bf16_triton"] = (bf16c or {}).get("e_bf16_triton")
    out["cast_stage"] = {f"{k} st{st}": (dt_of(ci, k, dict(std_cfg, stages=st), b, X) or (None,))[0]
                         for k in CAST_STAGE for st in (2, 3)}
    n = min(len(f["dt_rep"]) for f in fam.values())
    reps = []
    for i in range(n):
        r = sigmas(*(fam[f]["dt_rep"][i] for f in ("stock", "triton_std", "triton_cmp")), X, fp8, bw)
        if r:
            r |= {"e_h": elasticity(alias[1][i], fam["triton_std"]["dt_rep"][i]) if alias and len(alias[1]) > i
                  else None, "pool_share": out["pool_share"], "e_bf16_triton": out["e_bf16_triton"]}
        reps.append(r)
    out["repeats"] = reps
    good = [r for r in reps if r]
    out["range"] = {k: [min(r[k] for r in good), max(r[k] for r in good)] for k in ("sigma_dq", "sigma_mma",
                                                                                     "sigma_layout")} if good else {}
    out["dt_range"] = {f: round((max(v["dt_rep"]) - min(v["dt_rep"])) / v["dt_us"], 4) for f, v in fam.items()}
    out["flags"] = flags(X, out)
    return out


def flags(X: int, v: dict) -> list[str]:
    """The causes (A1, R-71 B1) that can explain a sigma_layout below the kill line: (a) time does not follow bytes,
    (b) the views do not share their DRAM reads, (c) the L2 path or the on-chip work caps the views kernel."""
    out, e = [], v.get("e_h")
    if e is not None and e < E_LINE[X]:
        out.append("a")
    if e is not None and e >= E_SHARE and v["sigma_layout"] < SHARE_FRAC * e:
        out.append("b")
    ps = (v.get("pool_share") or {}).get("triton_cmp")
    if ps is not None and ps < POOL_CAP:
        out.append("c")
    return out


def label(g: dict) -> str:
    """The rule of README Section 7 (A1) on one set of group metrics (the medians, or one repeat)."""
    tag = "; ".join(f"{X}: sigma_dq {v['sigma_dq']:.3f}, sigma_mma {v['sigma_mma']:.3f}, sigma_layout "
                    f"{v['sigma_layout']:.3f}, e_h {v.get('e_h')} at B={v['batch']}" for X, v in g.items()
                    if v and v.get("ok"))
    if not all(v and v.get("ok") for v in g.values()):
        return f"INCONCLUSIVE (a deciding group is missing: {[X for X, v in g.items() if not (v and v.get('ok'))]}; {tag})"
    if not all(v["std_memory_bound"] for v in g.values()):
        return f"INCONCLUSIVE (the Triton standard kernel reads below {MEM_BOUND} x copy, so the layout test is weak; {tag})"
    if any(v["sigma_mma"] >= GO_SIGMA and v["sigma_layout"] >= KILL[int(X)] for X, v in g.items()):
        return f"GO ({tag})"
    if all(v["sigma_layout"] < KILL[int(X)] for X, v in g.items()):
        fl = {X: flags(int(X), v) for X, v in g.items()}
        oracle_ok = all((v.get("e_bf16_triton") or 0) >= E_BF16_MIN for v in g.values())
        if all("a" in f for f in fl.values()) and oracle_ok:
            return f"KILL (byte route only: {CONSEQUENCE['a']}; {tag})"
        causes = sorted({c for f in fl.values() for c in f})
        if "a" in causes and not oracle_ok:
            why = "time does not follow bytes, but the K = V oracle is not valid on this GPU (e_bf16_triton below 0.5)"
        elif causes:
            why = "; ".join(CONSEQUENCE[c] for c in causes)
        else:
            why = "no named cause: the views kernel gains less than its bytes predict"
        return f"INCONCLUSIVE (sigma_layout is below the kill lines; flags {fl}: {why}; {tag})"
    if any(v["sigma_layout"] >= GO_SIGMA for v in g.values()):
        reason = "kernel quality: the layout works in Triton, but the compact kernel does not beat the stock bar"
    else:
        reason = "below the bar"
    return f"PIVOT ({reason}; {tag})"


def rebased(g: dict, bw: float | None) -> dict:
    """The group metrics with the E-51 stock dt as the baseline (A1, R-71 F8)."""
    out = {}
    for X, v in g.items():
        m = sigmas(E51_STOCK_DT[int(X)], v["triton_std"]["dt_us"], v["triton_cmp"]["dt_us"], int(X), True, bw)
        out[X] = v | m if m else v
    return out


def decision_rows(rows: list[dict], ci: dict, triton_only: bool = False) -> list[dict]:
    """The confirmation entries that enter the decision: per group, at 2K and X, the primary rows and the stock kernel
    with 32 splits. With triton_only (A1, R-71 B2: the V4 spread bound) only the two Triton families."""
    out = []
    for X, b in GROUPS.items():
        prim = primary_rows(rows, X)
        names = ("triton_std", "triton_cmp") if triton_only else ("stock", "triton_std", "triton_cmp")
        keys = [key(r["kernel"], b, r["context"], r["config"]) for n in names for r in prim[n] or ()]
        if not triton_only:
            keys += [key("stock_fp8_s32", b, length, S32) for length in (2048, X)]
        for k in keys:
            if k in ci and ci[k] not in out:
                out.append(ci[k])
    return out


def cap_share(res: Path) -> dict:
    """The share of busy telemetry samples under the software power cap (0x4) and their mean SM clock (R-70)."""
    busy = cap = 0
    clocks, cap_clocks = [], []
    for path in res.glob("env-*/telemetry.csv"):
        for line in path.read_text(errors="replace").splitlines()[1:]:
            parts = [x.strip() for x in line.split(",")]
            try:
                clock, util, reasons = float(parts[2]), float(parts[4]), int(parts[5], 16)
            except (ValueError, IndexError):
                continue
            if util > 0:
                busy += 1
                clocks.append(clock)
                if reasons & 0x4:
                    cap += 1
                    cap_clocks.append(clock)
    return {"busy": busy, "cap": cap, "cap_share": round(cap / busy, 3) if busy else None,
            "cap_mean_clock": round(statistics.mean(cap_clocks)) if cap_clocks else None,
            "busy_mean_clock": round(statistics.mean(clocks)) if clocks else None,
            "busy_min_clock": min(clocks) if clocks else None}


def checks(res: Path, kb: dict, run: dict, bw: float | None, rows: list[dict], ci: dict, groups: dict) -> dict:
    dec, trit = decision_rows(rows, ci), decision_rows(rows, ci, triton_only=True)
    v3, ratio_2k = {}, {}
    for (b, length), target in TARGETS.items():
        e = ci.get(key("stock_fp8_s32", b, length, S32))
        (ratio_2k if length == 2048 else v3)[f"{b}:{length}"] = round(e["us"] / target, 4) if e else None
    v3b = {}
    for X, b in GROUPS.items():
        d = dt_of(ci, "stock_fp8_s32", S32, b, X)
        v3[f"dt {b}:{X}"] = round(d[0] / (TARGETS[(b, X)] - TARGETS[(b, 2048)]), 4) if d else None
        v3b[f"dt {b}:{X}"] = round(d[0] / E51_STOCK_DT[X], 4) if d else None
    # V5 (E-51 A2): the rows at X and the dt rates; a 2K row can read from L2.
    gbps = [e["batch"] * e["context"] * 4 * (640 if compact(e["kernel"]) else 1024) / (e["us"] * 1e-6) / 1e9
            for e in dec if e["context"] > 2048]
    gbps += [v for g in groups.values() if g.get("ok") for v in g["dt_gbps"].values()]
    spreads = [statistics.median(e["spreads"]) for e in trit]
    stock_spreads = [statistics.median(e["spreads"]) for e in dec if e not in trit]
    trit_ranges = [g["dt_range"][f] for g in groups.values() if g.get("ok") for f in ("triton_std", "triton_cmp")]
    stock_ranges = [g["dt_range"]["stock"] for g in groups.values() if g.get("ok")]
    clocks = [c for e in dec for c in e.get("clocks") or [] if c]
    row_clocks = [[c for c in e.get("clocks") or [] if c] for e in dec]
    gpu = (run.get("gpu") or {}).get("name", "")
    return {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""),
            "V2": not deciding_failures(kb.get("correctness")),
            "V3": all(r is not None and abs(r - 1) <= V3_TOL for r in v3.values()),
            "V4": bool(trit) and all(s <= SPREAD_MAX for s in spreads)
                  and all(r <= DT_RANGE_MAX for r in trit_ranges) and all(r <= DT_RANGE_STOCK for r in stock_ranges),
            "V5": bw is not None and bool(dec) and all(g <= BW_MAX * bw for g in gbps),
            "V6": telemetry_ok(res),
            "_failures": deciding_failures(kb.get("correctness")), "_engine_ratio": v3, "_engine_ratio_2k": ratio_2k,
            "_e51_stock_ratio": v3b, "_decision_rows": len(dec), "_max_spread": max(spreads, default=None),
            "_stock_spreads": stock_spreads, "_max_dt_range": max(trit_ranges, default=None),
            "_stock_dt_range": max(stock_ranges, default=None), "_max_gbps": round(max(gbps), 1) if gbps else None,
            "_cap": cap_share(res), "_clock_min": min(clocks, default=None),
            "_rows_below_clock_flag": sum(1 for rc in row_clocks if rc and statistics.mean(rc) < CLOCK_FLAG)
            if clocks else None}


def decide(s: dict) -> str:
    if s.get("copy_gbps") is None:
        return "INVALID (no copy bandwidth: V5 cannot run)"
    if s.get("fail_fast"):
        return f"INVALID (V2: fail fast, {s['fail_fast']} did not pass the correctness cell)"
    if s["run_status"] not in OK_STATUS:
        return f"INCONCLUSIVE (the benchmark did not finish: {s['run_status']})"
    bad = [k for k, v in s["checks"].items() if not k.startswith("_") and not v]
    if bad:
        return f"INVALID ({', '.join(bad)})"
    g = s["fp8"]
    main = label(g)
    if not all(v.get("ok") for v in g.values()):
        return main
    word, n = main.split(" ")[0], min(len(v["repeats"]) for v in g.values())
    if n < REPEATS:
        return f"INCONCLUSIVE (the decision rows have {n} of {REPEATS} confirmation repeats; {main})"
    words = [label({X: v["repeats"][i] for X, v in g.items()}).split(" ")[0] for i in range(n)]
    agree = words.count(word)
    if agree < AGREE:
        return f"INCONCLUSIVE (the decision does not replicate: {word} on the medians, {words} on the repeats; {main})"
    alt = label(rebased(g, s.get("copy_gbps"))).split(" ")[0]
    if alt != word:
        return f"INCONCLUSIVE (baseline off: {word} with this session's stock kernel, {alt} with the E-51 stock dt; {main})"
    return main.replace(" (", f" (replicated in {agree} of {n} repeats; ", 1)


def factorial(rows: list[dict], bw: float | None) -> dict:
    """The cast x layout factorial at the E-51 configuration, from the sweep (reported): dt, its rate on the bytes
    read, and sigma_layout of each compact layout against the standard kernel of the same compute dtype."""
    out = {}
    for X, b in GROUPS.items():
        g = {}
        for k in DIAG:
            dt = sweep_dt(rows, k, E51, b, X)
            by = b * (X - 2048) * 4 * (640 if compact(k) else 1024)
            g[k] = {"dt_us": dt, "dt_gbps": round(by / (dt * 1e-6) / 1e9, 1) if dt else None,
                    "dt_vs_copy": round(by / (dt * 1e-6) / 1e9 / bw, 3) if dt and bw else None}
        for cmp_k, std_k in (("triton_cmpv_fp8", "triton_std_fp8"), ("triton_cmpv_fp8h", "triton_std_fp8h"),
                             ("triton_cmp_fp8", "triton_std_fp8"), ("triton_cmp_fp8h", "triton_std_fp8h")):
            c, s = g[cmp_k]["dt_us"], g[std_k]["dt_us"]
            g[f"sigma_layout {cmp_k}"] = round(1 - c / s, 4) if c and s else None
        out[str(X)] = g
    return out


def l2_ratios(rows: list[dict], ci: dict) -> dict:
    """Views over standard time where the cache can sit in L2 (reported): at B 4 and 2K at the best shared
    configuration (the L-18 gate of the rehearsal), and (A1, R-71 F1) at 11:2K against 19:2K at the selected
    standard configuration; a views row in L2 at 11:2K (57.7 MB against the 50 MB L2) makes dt larger."""
    t = {key(r["kernel"], r["batch"], r["context"], r["config"]): r["us"] for r in rows if r.get("status") == "ok"}
    out, cands = {}, []
    for c in {json.dumps(r["config"], sort_keys=True) for r in rows if r["kernel"] == "triton_std_fp8h"}:
        cfg = json.loads(c)
        s, v = t.get(key("triton_std_fp8h", 4, 2048, cfg)), t.get(key("triton_cmpv_fp8h", 4, 2048, cfg))
        if s and v:
            cands.append((s + v, cfg, round(v / s, 4)))
    if cands:
        _, cfg, ratio = min(cands, key=lambda x: x[0])
        out["b4_2k"] = {"config": cfg, "views_over_std": ratio}
    r, cfgs = {}, {}
    for X, b in GROUPS.items():                 # each group at its own selected standard configuration
        std = select(rows, FAMILIES["triton_std"], X)
        if std:
            cfg = cfgs[b] = std[1]["config"]
            s, v = ci.get(key("triton_std_fp8h", b, 2048, cfg)), ci.get(key("triton_cmpv_fp8h", b, 2048, cfg))
            r[b] = round(v["us"] / s["us"], 4) if s and v else None
    if r:
        out["2k"] = {"configs": cfgs, "19": r.get(19), "11": r.get(11),
                     "flag": r.get(19) is not None and r.get(11) is not None and r[11] < r[19] - L2_FLAG}
    return out


def linearity(rows: list[dict]) -> dict:
    """t(8K) against the line through t(2K) and t(X) at the group batch, per primary family with its best
    configuration per cell (reported): 0 means linear in L."""
    out = {}
    for X, b in GROUPS.items():
        for name, kernels in FAMILIES.items():
            t2, tX, tL = best(rows, kernels, b, 2048), best(rows, kernels, b, X), best(rows, kernels, b, 8192)
            if t2 and tX and tL:
                line = t2["us"] + (tX["us"] - t2["us"]) * (8192 - 2048) / (X - 2048)
                out[f"{name} {b}:8192"] = round(tL["us"] / line - 1, 4)
    return out


def summarize(res: Path) -> dict:
    kb, run = load(res / "kbench" / "kbench.json", {}), load(res / "kbench" / "run.json", {})
    bw = copy_gbps(res)
    rows = validated(kb.get("cells", []), kb.get("correctness"))
    ci = conf_index(kb.get("confirm"))
    bf16c = {X: bf16_controls(rows, X) for X in GROUPS}
    fp8 = {str(X): group(rows, X, True, bw, ci, bf16c[X]) for X in GROUPS}
    sha = {}
    for r in kb.get("cells", []):
        if r.get("ptx_sha"):
            sha[f"{r['kernel']} {cfg_key(r['config'])}"] = r["ptx_sha"]
    e51key = cfg_key(E51)
    out = {"copy_gbps": bw, "run_status": run.get("status", "missing"), "fail_fast": run.get("fail_fast"),
           "late_error": run.get("late_error"), "gpu": run.get("gpu"), "phases": run.get("phases"),
           "lite_cells": run.get("lite_cells"),
           "versions": {k: run.get(k) for k in ("torch", "triton", "vllm")},
           "checks": checks(res, kb, run, bw, rows, ci, fp8) if kb else {"V1": False},
           "fp8": fp8,
           "bf16": {str(X): group(rows, X, False, bw, None) for X in GROUPS},
           "factorial": factorial(rows, bw), "l2_ratios": l2_ratios(rows, ci), "linearity": linearity(rows),
           "ptx_counts": {f"{r['kernel']} {cfg_key(r['config'])}": r["ptx_counts"] for r in kb.get("cells", [])
                          if r.get("ptx_counts")},
           "ptx_views_identical": (sha.get(f"triton_std_fp8h {e51key}") == sha.get(f"triton_cmpv_fp8h {e51key}"))
           if sha.get(f"triton_std_fp8h {e51key}") else None,
           "compile_check": load(res / "compile_check.json", None), "sass_counts": load(res / "sass_counts.json", None),
           "dram_probe": (res / "dram_probe.txt").read_text()[-4000:] if (res / "dram_probe.txt").exists() else None,
           "dropped_rows": len(kb.get("cells", [])) - len(rows), "correctness": kb.get("correctness")}
    b4 = {}
    for k in ("stock_fp8_s32", "stock_fp8_auto", "stock_fp8_s32_views", "triton_std_fp8h", "triton_cmpv_fp8h"):
        b4[k] = {str(L): (best(rows, (k,), 4, L) or {}).get("us") for L in (2048, 32768)}
    out["batch4_us"] = b4
    return out


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
