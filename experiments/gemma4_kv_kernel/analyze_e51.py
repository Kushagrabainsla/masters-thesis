"""E-51 analysis (README Sections 5 to 7, with amendment A1): does a compact K=V layout cut the context-growing global
attention time of Gemma 4 31B decode enough, against the best stock FP8 kernel in vLLM?

Layout under the results root:
  kbench/kbench.json   "cells": the sweep (per cell and configuration: us per call, spread, bytes, GB/s, registers);
                       "correctness": the correctness cell; "confirm": the decision rows, timed REPEATS times
  kbench/run.json      GPU, versions, status ("ok_groups" once the decision data is complete, "ok" at the end)
  env-*/telemetry.csv  nvidia-smi samples;  env-*/microbench_*.json  copy GB/s

Mechanism of the metric: at one batch B, the decode attention time of a layer grows with the context L because the
kernel reads the cache of every token. The growth from 2,048 to X, dt(X) = t(X) - t(2,048), is the part that a
smaller cache row can shrink. For each kernel family the sweep selects one configuration per group: the fastest at X
among the configurations that passed the correctness cell and ran at 2K and at X (A1, R1). The confirmation phase
times that configuration at 2K and X REPEATS times, and dt is the median of the per-repeat differences (A1, O3).
  sigma_dq     = 1 - dt_compact / dt_stock          (against the best stock FP8 dequant kernel in this image)
  sigma_mma    = 1 - dt_compact / (0.85 dt_stock)   (against a stock kernel 15% faster on dt, the FA PR 199 size, R-68)
  sigma_layout = 1 - dt_compact / dt_standard       (the two Triton layouts: the layout effect in one code family)
E-50 maps sigma to the step saving: S = sigma / 0.375 x the E-50 ceiling, so S = 5% at sigma 0.155 (32K) and 0.125
(64K), and S is about 8% at sigma 0.25 (R-68).
Diagnostics (reported, not deciding; A1, O1): the oracle rows read K = V, which halves the DRAM bytes and keeps the
on-chip work. e = 2 x (1 - dt_oracle / dt) is the time elasticity to DRAM bytes (1: memory-bound, 0: insensitive),
for the stock kernel with 32 splits (e_stock) and for the selected Triton standard configuration (e_triton).

Usage: analyze_e51.py --res <dir> --out summary.json
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
TOL_TRITON = 0.01                             # each Triton configuration (A1, R5); a dropped page gives about 0.02
V3_TOL = 0.15                                 # stock FP8 (32 splits) against the E-50 in-engine time per call and dt
SPREAD_MAX = 0.02
BW_MAX = 1.15                                 # a kernel cannot read faster than 1.15 x the copy bandwidth
MEM_BOUND = 0.5                               # the Triton standard kernel must read at least 0.5 x copy on dt
MEM_BOUND_CMP = 0.6                           # a KILL needs the compact kernel at this level on dt (A1, R2; E-50 M1)
REPEATS = 5                                   # confirmation repeats (MEASUREMENT.md Section 1.5)
AGREE = 4                                     # the decision must come out the same in at least 4 of the 5 repeats
OK_STATUS = ("ok", "ok_groups")
TARGETS = {(19, 2048): 238.3, (19, 32768): 1358.1, (11, 2048): 143.2, (11, 65536): 1507.2}
FAMILIES = {"stock": ("stock_fp8_s32", "stock_fp8_auto"), "triton_std": ("triton_std_fp8",),
            "triton_cmp": ("triton_cmp_fp8",)}
S32 = {"num_splits": 32}


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def copy_gbps(res: Path) -> float | None:
    vals = [load(p, {}).get("copy_gb_per_s_median") for p in sorted(res.glob("env-*/microbench_*.json"))]
    vals = [v for v in vals if v]
    return statistics.median(vals) if vals else None


def cfg_key(c: dict) -> str:
    return f'{c["splits"]}/{c["warps"]}/{c["tile"]}'


def key(kernel: str, b: int, length: int, config: dict) -> tuple:
    return kernel, b, length, json.dumps(config, sort_keys=True)


def passed(err, tol: float) -> bool:
    return isinstance(err, (int, float)) and not isinstance(err, bool) and err <= tol


def validated(rows: list[dict], corr: dict | None) -> list[dict]:
    """Drop the rows of a configuration that failed the correctness cell (A1, R5): a Triton configuration whose error
    is missing or above TOL_TRITON, or a stock kernel whose error is missing or above TOL_STOCK. Without a
    correctness cell nothing is dropped (V2 then fails)."""
    if not corr:
        return list(rows)
    keep = []
    for r in rows:
        k = r["kernel"]
        entry = (corr.get("fp8" if "fp8" in k else "bf16") or {}).get(k) or {}
        if k.startswith("triton_"):
            ok = passed((entry.get("by_config") or {}).get(cfg_key(r["config"])), TOL_TRITON)
        else:
            ok = passed(entry.get("err"), TOL_STOCK)
        if ok:
            keep.append(r)
    return keep


def deciding_failures(corr: dict | None) -> list[str]:
    """The deciding FP8 kernels that did not pass the correctness cell: the stock kernel with 32 splits (V3 needs it),
    and each Triton family without one passing configuration (A1, R6: the fail-fast test)."""
    fp8 = (corr or {}).get("fp8") or {}
    bad = [] if passed((fp8.get("stock_fp8_s32") or {}).get("err"), TOL_STOCK) else ["stock_fp8_s32"]
    for k in ("triton_std_fp8", "triton_cmp_fp8"):
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
    """The configuration of a family for group X (A1, R1): the fastest at X among those that ran at 2K and at X. One
    configuration at both lengths, as in the engine. Returns its sweep rows (2K, X)."""
    b = GROUPS[X]
    lo = {key(r["kernel"], b, 0, r["config"]): r for r in rows if r["kernel"] in kernels and r["batch"] == b
          and r["context"] == 2048 and r.get("status") == "ok"}
    pairs = [(lo[key(r["kernel"], b, 0, r["config"])], r) for r in rows if r["kernel"] in kernels and r["batch"] == b
             and r["context"] == X and r.get("status") == "ok" and key(r["kernel"], b, 0, r["config"]) in lo]
    return min(pairs, key=lambda p: p[1]["us"]) if pairs else None


def confirm_plan(rows: list[dict]) -> list[dict]:
    """The rows that the confirmation phase times (A1, O3): per group, the selected configuration of each FP8 family at
    2K and X, the stock kernel with 32 splits (V3), and the two oracle rows (K = V). `rows` are validated sweep rows."""
    plan = []

    def add(kernel, config, b, length):
        p = {"kernel": kernel, "batch": b, "context": length, "config": config}
        if p not in plan:
            plan.append(p)

    for X, b in GROUPS.items():
        for kernels in FAMILIES.values():
            for r in select(rows, kernels, X) or ():
                add(r["kernel"], r["config"], b, r["context"])
        std = select(rows, FAMILIES["triton_std"], X)
        for length in (2048, X):
            add("stock_fp8_s32", S32, b, length)
            add("stock_fp8_s32_alias", S32, b, length)
            if std:
                add("triton_std_fp8_alias", std[1]["config"], b, length)
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


def family(rows: list[dict], kernels: tuple[str, ...], X: int, ci: dict | None) -> dict | None:
    """dt(X) of one family with its selected configuration: from the confirmation repeats, or (ci None: the BF16
    groups, reported only) from the one sweep measurement."""
    sel = select(rows, kernels, X)
    if not sel:
        return None
    lo, hi = sel
    out = {"kernel": hi["kernel"], "config": hi["config"],
           "sweep": {k: hi.get(k) for k in ("us", "gbps", "regs", "spills", "shared", "stages")}}
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
    grow = lambda compact: b * (X - 2048) * 4 * (640 if compact else 1024) * (1 if fp8 else 2)   # bytes on dt
    rate = lambda compact, dt: grow(compact) / (dt * 1e-6) / 1e9
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


def group(rows: list[dict], X: int, fp8: bool, bw: float | None, ci: dict | None) -> dict:
    sfx = "fp8" if fp8 else "bf16"
    fam = {"stock": family(rows, (f"stock_{sfx}_s32", f"stock_{sfx}_auto"), X, ci),
           "triton_std": family(rows, (f"triton_std_{sfx}",), X, ci),
           "triton_cmp": family(rows, (f"triton_cmp_{sfx}",), X, ci)}
    out = {"batch": GROUPS[X], **fam}
    m = sigmas(fam["stock"]["dt_us"], fam["triton_std"]["dt_us"], fam["triton_cmp"]["dt_us"], X, fp8, bw) \
        if all(fam.values()) else None
    if not m:
        return out | {"ok": False, "why": "a kernel family is missing, unconfirmed or has no growth"}
    out |= m
    if ci is None:
        return out | {"repeats": []}
    n = min(len(f["dt_rep"]) for f in fam.values())
    out["repeats"] = [sigmas(*(fam[f]["dt_rep"][i] for f in ("stock", "triton_std", "triton_cmp")), X, fp8, bw)
                      for i in range(n)]
    good = [r for r in out["repeats"] if r]
    out["range"] = {k: [min(r[k] for r in good), max(r[k] for r in good)] for k in ("sigma_dq", "sigma_mma",
                                                                                     "sigma_layout")} if good else {}
    b = GROUPS[X]
    s32, s32a = dt_of(ci, "stock_fp8_s32", S32, b, X), dt_of(ci, "stock_fp8_s32_alias", S32, b, X)
    stda = dt_of(ci, "triton_std_fp8_alias", fam["triton_std"]["config"], b, X)
    out["dt_stock_s32"] = s32
    out["e_stock"] = round(2 * (1 - s32a[0] / s32[0]), 3) if s32 and s32a and s32[0] > 0 else None
    out["e_triton"] = round(2 * (1 - stda[0] / fam["triton_std"]["dt_us"]), 3) if stda else None
    return out


def label(g: dict) -> str:
    """The rule of README Section 7 on one set of group metrics (the medians, or one repeat)."""
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
                f" x copy on dt in a group, so a defect of the compact kernel can explain it; {tag})")
    if any(v["sigma_layout"] >= GO_SIGMA for v in g.values()):
        reason = "kernel quality: the layout works in Triton, but the compact kernel does not beat the stock bar"
    else:
        reason = "below the bar"
    return f"PIVOT ({reason}; {tag})"


def decision_rows(rows: list[dict], ci: dict) -> list[dict]:
    """The confirmation entries that enter the decision (A1, R3): per group, the selected configuration of each FP8
    family at 2K and X, and the stock kernel with 32 splits at the four V3 cells."""
    out = []
    for X, b in GROUPS.items():
        keys = [key(r["kernel"], b, r["context"], r["config"]) for kernels in FAMILIES.values()
                for r in select(rows, kernels, X) or ()]
        keys += [key("stock_fp8_s32", b, length, S32) for length in (2048, X)]
        for k in keys:
            if k in ci and ci[k] not in out:
                out.append(ci[k])
    return out


def checks(res: Path, kb: dict, run: dict, bw: float | None, rows: list[dict], ci: dict, groups: dict) -> dict:
    dec = decision_rows(rows, ci)
    v3 = {}
    for (b, length), target in TARGETS.items():
        e = ci.get(key("stock_fp8_s32", b, length, S32))
        v3[f"{b}:{length}"] = round(e["us"] / target, 4) if e else None
    for X, b in GROUPS.items():           # A1: the growth too, since the rule uses dt
        d = dt_of(ci, "stock_fp8_s32", S32, b, X)
        v3[f"dt {b}:{X}"] = round(d[0] / (TARGETS[(b, X)] - TARGETS[(b, 2048)]), 4) if d else None
    # V5 (A2): the rows at X and the dt rates. A 2K row can read from L2 (the S5 rehearsal: the compact FP8 rows at
    # B 11 and 2K, 57.7 MB, read at 1.15 x copy in the 72 MB L2 of the RTX 4090); that only makes dt larger.
    gbps = [e["batch"] * e["context"] * 4 * (640 if "cmp" in e["kernel"] else 1024) / (e["us"] * 1e-6) / 1e9
            for e in dec if e["context"] > 2048]
    gbps += [v for g in groups.values() if g.get("ok") for v in g["dt_gbps"].values()]
    spreads = [statistics.median(e["spreads"]) for e in dec]
    gpu = (run.get("gpu") or {}).get("name", "")
    return {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""),
            "V2": not deciding_failures(kb.get("correctness")),
            "V3": all(r is not None and abs(r - 1) <= V3_TOL for r in v3.values()),
            "V4": bool(dec) and all(s <= SPREAD_MAX for s in spreads),
            "V5": bw is not None and bool(dec) and all(g <= BW_MAX * bw for g in gbps),
            "V6": telemetry_ok(res),
            "_failures": deciding_failures(kb.get("correctness")), "_engine_ratio": v3,
            "_decision_rows": len(dec), "_max_spread": max(spreads, default=None),
            "_max_gbps": round(max(gbps), 1) if gbps else None}


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


def common_layout(rows: list[dict]) -> dict:
    """sigma_layout at the best configuration that both Triton FP8 kernels share (A1, O4; sweep values, reported)."""
    t = {key(r["kernel"], r["batch"], r["context"], r["config"]): r["us"] for r in rows if r.get("status") == "ok"}
    out = {}
    for X, b in GROUPS.items():
        cands = []
        for c in {json.dumps(r["config"], sort_keys=True) for r in rows if r["kernel"] == "triton_std_fp8"}:
            c = json.loads(c)
            ts = [t.get(key(k, b, length, c)) for k in ("triton_std_fp8", "triton_cmp_fp8") for length in (2048, X)]
            if None not in ts and ts[1] > ts[0]:
                cands.append((ts[1] + ts[3], c, ts))
        if cands:
            _, c, ts = min(cands, key=lambda x: x[0])
            out[str(X)] = {"config": c, "sigma_layout": round(1 - (ts[3] - ts[2]) / (ts[1] - ts[0]), 4)}
    return out


def linearity(rows: list[dict]) -> dict:
    """t(L) against the line through t(2K) and t(X) at the group batch, per FP8 family with its best configuration per
    cell (A1, O9; sweep values, reported): 0 means linear in L."""
    out = {}
    for X, b in GROUPS.items():
        for name, kernels in FAMILIES.items():
            t2, tX = best(rows, kernels, b, 2048), best(rows, kernels, b, X)
            for length in (8192, 32768):
                tL = best(rows, kernels, b, length) if length < X else None
                if t2 and tX and tL:
                    line = t2["us"] + (tX["us"] - t2["us"]) * (length - 2048) / (X - 2048)
                    out[f"{name} {b}:{length}"] = round(tL["us"] / line - 1, 4)
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
           "common_config": common_layout(rows), "linearity": linearity(rows),
           "dropped_rows": len(kb.get("cells", [])) - len(rows), "correctness": kb.get("correctness")}
    b4 = {}
    for k in ("stock_fp8_s32", "stock_fp8_auto", "triton_std_fp8", "triton_cmp_fp8", "stock_bf16_auto",
              "stock_bf16_s32"):
        b4[k] = {str(L): (best(rows, (k,), 4, L) or {}).get("us") for L in (2048, 32768, 65536)}
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
