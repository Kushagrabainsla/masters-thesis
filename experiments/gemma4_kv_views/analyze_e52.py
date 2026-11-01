"""E-52 analysis (README Sections 5 to 7): with fixed kernels (FP16 compute; the compact row read by the standard
kernel through two overlapping views), does the compact K=V layout cut the context-growing global attention time of
Gemma 4 31B decode enough, against its standard twin and the best stock FP8 kernel in vLLM?

Layout under the results root (as E-51):
  kbench/kbench.json   "cells": the sweep; "correctness": the correctness cell; "confirm": the confirmation rows
  kbench/run.json      GPU, versions, status ("ok_groups" once the decision data is complete, "ok" at the end)
  kbench/asm/          PTX and TTGIR of each Triton kernel at the E-51 configuration
  env-*/telemetry.csv  nvidia-smi samples;  env-*/microbench_*.json  copy GB/s

The metrics and the rule are the E-51 ones (README Section 7 there, A1 and A2) with the E-52 primary families:
  stock       the stock FA4 FP8 dequant kernel, num_splits 32 or automatic (unchanged)
  triton_std  the standard Triton kernel with FP16 compute (`triton_std_fp8h`)
  triton_cmp  the same kernel on the views layout with FP16 compute (`triton_cmpv_fp8h`)
dt(X) = t(X) - t(2,048) at the group batch; sigma_dq = 1 - dt_cmp / dt_stock; sigma_mma = 1 - dt_cmp /
(0.85 dt_stock); sigma_layout = 1 - dt_cmp / dt_std. Kill lines 0.155 (32K) and 0.125 (64K); go line 0.25 (R-68).
R-70 fixes: V3 bounds only the X cells and dt (the 2K ratios are reported); V4 uses the central spread of 30 replays
and also bounds the across-repeat range of each primary family's dt at 3%; V6 reports the share of the software power
cap (0x4) next to the hard-throttle check.
Reported only: the stock kernel on the views (sigma_stock_layout, 32 splits on both sides), the K = V elasticities
(e_stock, e_h; BF16 control e_bf16), the pool shares (1 - dt_pool / dt), the cast and layout factorial at the E-51
configuration (sweep values), BF16, the L2-resident ratio at B 4 and 2K, the linearity and the PTX counts.

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
GO_SIGMA = 0.25                               # S about 8% (R-68)
MMA_GAIN = 0.15                               # FA PR 199: FP8 MMA 8% to 15% faster on the context part (R-68)
TOL_STOCK = 0.02                              # relative L2 error against the reference of the stored values
TOL_TRITON = 0.01                             # each Triton configuration (E-51 A1, R5)
V3_TOL = 0.15                                 # stock FP8 (32 splits) against the E-50 in-engine time at X and dt
SPREAD_MAX = 0.02                             # the central spread of a decision row (median over the repeats)
DT_RANGE_MAX = 0.03                           # the across-repeat range of a primary family's dt (R-70)
BW_MAX = 1.15                                 # a kernel cannot read faster than 1.15 x the copy bandwidth
MEM_BOUND = 0.5                               # the Triton standard kernel must read at least 0.5 x copy on dt
MEM_BOUND_CMP = 0.6                           # a KILL needs the compact kernel at this level on dt (E-51 A1, R2)
CLOCK_FLAG = 1900                             # a decision row with a mean SM clock below this is flagged (reported)
REPEATS = 5                                   # confirmation repeats (MEASUREMENT.md Section 1.5)
AGREE = 4                                     # the decision must come out the same in at least 4 of the 5 repeats
OK_STATUS = ("ok", "ok_groups")
TARGETS = {(19, 2048): 238.3, (19, 32768): 1358.1, (11, 2048): 143.2, (11, 65536): 1507.2}
TARGETS_STR = {f"{b}:{length}": v for (b, length), v in TARGETS.items()}
FAMILIES = {"stock": ("stock_fp8_s32", "stock_fp8_auto"), "triton_std": ("triton_std_fp8h",),
            "triton_cmp": ("triton_cmpv_fp8h",)}
BF16_FAMILIES = {"stock": ("stock_bf16_s32", "stock_bf16_auto"), "triton_std": ("triton_std_bf16",),
                 "triton_cmp": ("triton_cmpv_bf16",)}
ORACLE = {"stock_fp8_s32_alias": "stock_fp8_s32", "triton_std_fp8h_alias": "triton_std_fp8h",
          "stock_bf16_s32_alias": "stock_bf16_s32", "triton_std_bf16_alias": "triton_std_bf16"}
POOL = {"stock_fp8_s32_pool": "stock_fp8_s32", "triton_std_fp8h_pool": "triton_std_fp8h",
        "triton_cmpv_fp8h_pool": "triton_cmpv_fp8h"}
DIAG = ("triton_std_fp8", "triton_std_fp8h", "triton_cmpv_fp8", "triton_cmpv_fp8h", "triton_cmp_fp8",
        "triton_cmp_fp8h")                    # the cast x layout factorial at the E-51 configuration
E51 = {"splits": 32, "warps": 4, "tile": 32, "stages": 2}
S32 = {"num_splits": 32}
COMPACT_WORDS = ("cmpv", "cmp_", "_views")    # kernels that read 640 values per token


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


def confirm_plan(rows: list[dict]) -> list[dict]:
    """The rows that the confirmation phase times: per group, at 2K and X, the selected configuration of each primary
    family, the stock kernel with 32 splits (V3) and on the views, and the K = V and pool rows of the stock kernel and
    of the selected Triton configurations. `rows` are validated sweep rows."""
    plan = []

    def add(kernel, config, b, length):
        p = {"kernel": kernel, "batch": b, "context": length, "config": config}
        if p not in plan:
            plan.append(p)

    for X, b in GROUPS.items():
        sel = {name: select(rows, kernels, X) for name, kernels in FAMILIES.items()}
        for s in sel.values():
            for r in s or ():
                add(r["kernel"], r["config"], b, r["context"])
        for length in (2048, X):
            for k in ("stock_fp8_s32", "stock_fp8_s32_views", "stock_fp8_s32_alias", "stock_fp8_s32_pool"):
                add(k, S32, b, length)
            if sel["triton_std"]:
                for k in ("triton_std_fp8h_alias", "triton_std_fp8h_pool"):
                    add(k, sel["triton_std"][1]["config"], b, length)
            if sel["triton_cmp"]:
                add("triton_cmpv_fp8h_pool", sel["triton_cmp"][1]["config"], b, length)
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


def family(rows: list[dict], kernels: tuple[str, ...], X: int, ci: dict | None) -> dict | None:
    """dt(X) of one family with its selected configuration: from the confirmation repeats, or (ci None: BF16,
    reported only) from the one sweep measurement."""
    sel = select(rows, kernels, X)
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
            "std_memory_bound": bw is not None and std_bw >= MEM_BOUND * bw,
            "cmp_memory_bound": bw is not None and cmp_bw >= MEM_BOUND_CMP * bw}


def elasticity(alias, base) -> float | None:
    """e = 2 x (1 - dt_alias / dt): the time elasticity to DRAM bytes (K = V halves them)."""
    a, b = (x[0] if isinstance(x, tuple) else x for x in (alias, base))
    return round(2 * (1 - a / b), 3) if a is not None and b else None


def group(rows: list[dict], X: int, fp8: bool, bw: float | None, ci: dict | None) -> dict:
    fams = FAMILIES if fp8 else BF16_FAMILIES
    fam = {name: family(rows, kernels, X, ci) for name, kernels in fams.items()}
    b = GROUPS[X]
    out = {"batch": b, **fam}
    m = sigmas(fam["stock"]["dt_us"], fam["triton_std"]["dt_us"], fam["triton_cmp"]["dt_us"], X, fp8, bw) \
        if all(fam.values()) else None
    if not m:
        return out | {"ok": False, "why": "a kernel family is missing, unconfirmed or has no growth"}
    out |= m
    if ci is None:                      # BF16: one sweep measurement; the K = V control from the sweep (R-70)
        std_cfg = fam["triton_std"]["config"]
        out["e_bf16_stock"] = elasticity(sweep_dt(rows, "stock_bf16_s32_alias", S32, b, X),
                                         sweep_dt(rows, "stock_bf16_s32", S32, b, X))
        out["e_bf16_triton"] = elasticity(sweep_dt(rows, "triton_std_bf16_alias", std_cfg, b, X),
                                          sweep_dt(rows, "triton_std_bf16", std_cfg, b, X))
        fused = family(rows, ("triton_cmp_bf16",), X, None)
        out["sigma_layout_fused"] = round(1 - fused["dt_us"] / fam["triton_std"]["dt_us"], 4) if fused else None
        return out | {"repeats": []}
    n = min(len(f["dt_rep"]) for f in fam.values())
    out["repeats"] = [sigmas(*(fam[f]["dt_rep"][i] for f in ("stock", "triton_std", "triton_cmp")), X, fp8, bw)
                      for i in range(n)]
    good = [r for r in out["repeats"] if r]
    out["range"] = {k: [min(r[k] for r in good), max(r[k] for r in good)] for k in ("sigma_dq", "sigma_mma",
                                                                                     "sigma_layout")} if good else {}
    out["dt_range"] = {f: round((max(v["dt_rep"]) - min(v["dt_rep"])) / v["dt_us"], 4) for f, v in fam.items()}
    s32, sv = dt_of(ci, "stock_fp8_s32", S32, b, X), dt_of(ci, "stock_fp8_s32_views", S32, b, X)
    out["dt_stock_s32"] = s32
    out["dt_stock_s32_views"] = sv
    out["sigma_stock_layout"] = round(1 - sv[0] / s32[0], 4) if s32 and sv and s32[0] > 0 else None
    std_cfg, cmp_cfg = fam["triton_std"]["config"], fam["triton_cmp"]["config"]
    out["e_stock"] = elasticity(dt_of(ci, "stock_fp8_s32_alias", S32, b, X), s32)
    out["e_h"] = elasticity(dt_of(ci, "triton_std_fp8h_alias", std_cfg, b, X), fam["triton_std"]["dt_us"])
    pools = {"stock": (dt_of(ci, "stock_fp8_s32_pool", S32, b, X), s32[0] if s32 else None),
             "triton_std": (dt_of(ci, "triton_std_fp8h_pool", std_cfg, b, X), fam["triton_std"]["dt_us"]),
             "triton_cmp": (dt_of(ci, "triton_cmpv_fp8h_pool", cmp_cfg, b, X), fam["triton_cmp"]["dt_us"])}
    out["pool_share"] = {k: round(1 - p[0] / d, 3) if p and d else None for k, (p, d) in pools.items()}
    return out


def label(g: dict) -> str:
    """The rule of README Section 7 (the E-51 rule) on one set of group metrics (the medians, or one repeat)."""
    tag = "; ".join(f"{X}: sigma_dq {v['sigma_dq']:.3f}, sigma_mma {v['sigma_mma']:.3f}, sigma_layout "
                    f"{v['sigma_layout']:.3f} at B={v['batch']}" for X, v in g.items() if v and v.get("ok"))
    if not all(v and v.get("ok") for v in g.values()):
        return f"INCONCLUSIVE (a deciding group is missing: {[X for X, v in g.items() if not (v and v.get('ok'))]}; {tag})"
    if not all(v["std_memory_bound"] for v in g.values()):
        return f"INCONCLUSIVE (the Triton standard kernel reads below {MEM_BOUND} x copy, so the layout test is weak; {tag})"
    if any(v["sigma_mma"] >= GO_SIGMA and v["sigma_layout"] >= KILL[int(X)] for X, v in g.items()):
        return f"GO ({tag})"
    if all(v["sigma_layout"] < KILL[int(X)] for X, v in g.items()):
        if all(v["cmp_memory_bound"] for v in g.values()):
            return f"KILL ({tag})"
        return (f"INCONCLUSIVE (sigma_layout is below the kill lines, but the compact kernel reads below {MEM_BOUND_CMP}"
                f" x copy on dt in a group, so the kernels' on-chip limit can explain it; {tag})")
    if any(v["sigma_layout"] >= GO_SIGMA for v in g.values()):
        reason = "kernel quality: the layout works in Triton, but the compact kernel does not beat the stock bar"
    else:
        reason = "below the bar"
    return f"PIVOT ({reason}; {tag})"


def decision_rows(rows: list[dict], ci: dict) -> list[dict]:
    """The confirmation entries that enter the decision: per group, the selected configuration of each primary family
    at 2K and X, and the stock kernel with 32 splits at 2K and X."""
    out = []
    for X, b in GROUPS.items():
        keys = [key(r["kernel"], b, r["context"], r["config"]) for kernels in FAMILIES.values()
                for r in select(rows, kernels, X) or ()]
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
    dec = decision_rows(rows, ci)
    v3, ratio_2k = {}, {}
    for (b, length), target in TARGETS.items():
        e = ci.get(key("stock_fp8_s32", b, length, S32))
        (ratio_2k if length == 2048 else v3)[f"{b}:{length}"] = round(e["us"] / target, 4) if e else None
    for X, b in GROUPS.items():
        d = dt_of(ci, "stock_fp8_s32", S32, b, X)
        v3[f"dt {b}:{X}"] = round(d[0] / (TARGETS[(b, X)] - TARGETS[(b, 2048)]), 4) if d else None
    # V5 (E-51 A2): the rows at X and the dt rates; a 2K row can read from L2.
    gbps = [e["batch"] * e["context"] * 4 * (640 if compact(e["kernel"]) else 1024) / (e["us"] * 1e-6) / 1e9
            for e in dec if e["context"] > 2048]
    gbps += [v for g in groups.values() if g.get("ok") for v in g["dt_gbps"].values()]
    spreads = [statistics.median(e["spreads"]) for e in dec]
    dt_ranges = [r for g in groups.values() if g.get("ok") for r in g["dt_range"].values()]
    clocks = [c for e in dec for c in e.get("clocks") or [] if c]
    row_clocks = [[c for c in e.get("clocks") or [] if c] for e in dec]
    gpu = (run.get("gpu") or {}).get("name", "")
    return {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""),
            "V2": not deciding_failures(kb.get("correctness")),
            "V3": all(r is not None and abs(r - 1) <= V3_TOL for r in v3.values()),
            "V4": bool(dec) and all(s <= SPREAD_MAX for s in spreads) and all(r <= DT_RANGE_MAX for r in dt_ranges),
            "V5": bw is not None and bool(dec) and all(g <= BW_MAX * bw for g in gbps),
            "V6": telemetry_ok(res),
            "_failures": deciding_failures(kb.get("correctness")), "_engine_ratio": v3, "_engine_ratio_2k": ratio_2k,
            "_decision_rows": len(dec), "_max_spread": max(spreads, default=None),
            "_max_dt_range": max(dt_ranges, default=None), "_max_gbps": round(max(gbps), 1) if gbps else None,
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


def l2_ratio(rows: list[dict]) -> dict:
    """At B 4 and 2K (in L2 on both GPUs): the views time over the standard time at the best shared configuration
    (the L-18 gate of the rehearsal; reported)."""
    t = {key(r["kernel"], r["batch"], r["context"], r["config"]): r["us"] for r in rows if r.get("status") == "ok"}
    cands = []
    for c in {json.dumps(r["config"], sort_keys=True) for r in rows if r["kernel"] == "triton_std_fp8h"}:
        cfg = json.loads(c)
        s, v = t.get(key("triton_std_fp8h", 4, 2048, cfg)), t.get(key("triton_cmpv_fp8h", 4, 2048, cfg))
        if s and v:
            cands.append((s + v, cfg, round(v / s, 4)))
    if not cands:
        return {}
    _, cfg, ratio = min(cands, key=lambda x: x[0])
    return {"config": cfg, "views_over_std": ratio}


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
    fp8 = {str(X): group(rows, X, True, bw, ci) for X in GROUPS}
    out = {"copy_gbps": bw, "run_status": run.get("status", "missing"), "fail_fast": run.get("fail_fast"),
           "late_error": run.get("late_error"), "gpu": run.get("gpu"),
           "versions": {k: run.get(k) for k in ("torch", "triton", "vllm")},
           "checks": checks(res, kb, run, bw, rows, ci, fp8) if kb else {"V1": False},
           "fp8": fp8,
           "bf16": {str(X): group(rows, X, False, bw, None) for X in GROUPS},
           "factorial": factorial(rows, bw), "l2_ratio": l2_ratio(rows), "linearity": linearity(rows),
           "ptx_counts": {f"{r['kernel']} {cfg_key(r['config'])}": r["ptx_counts"] for r in kb.get("cells", [])
                          if r.get("ptx_counts")},
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
