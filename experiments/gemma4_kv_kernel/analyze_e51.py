"""E-51 analysis (README Sections 5 to 7): does a compact K=V layout cut the context-growing global attention time of
Gemma 4 31B decode enough, against the best stock FP8 kernel in vLLM?

Layout under the results root:
  kbench/kbench.json       per cell and kernel configuration: microseconds per call, spread, bytes, GB/s; correctness
  kbench/run.json          GPU, versions, status
  env-*/telemetry.csv      nvidia-smi samples;  env-*/microbench_*.json  copy GB/s

Mechanism of the metric: at one batch B, the decode attention time of a layer grows with the context L because the
kernel reads the cache of every token. The growth from 2,048 to X, dt(X) = t(X) - t(2,048), is the part that a
smaller cache row can shrink. For each kernel family the analysis takes the best configuration per cell.
  sigma_dq     = 1 - dt_compact / dt_stock          (against the best stock FP8 dequant kernel in this image)
  sigma_mma    = 1 - dt_compact / (0.85 dt_stock)   (against a stock kernel 15% faster on dt, the FA PR 199 size, R-68)
  sigma_layout = 1 - dt_compact / dt_standard       (the two Triton layouts: the layout effect in one code family)
E-50 maps sigma to the step saving: S = sigma / 0.375 x the E-50 ceiling, so S = 5% at sigma 0.155 (32K) and 0.125
(64K), and S is about 8% at sigma 0.25 (R-68).

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
TOL_CORRECT = 0.02                            # relative L2 error against the reference of the stored values
V3_TOL = 0.15                                 # stock FP8 (32 splits) against the E-50 in-engine time per call
SPREAD_MAX = 0.02
BW_MAX = 1.15                                 # a kernel cannot read faster than 1.15 x the copy bandwidth
MEM_BOUND = 0.5                               # the Triton standard kernel must read at least 0.5 x copy on dt
TARGETS = {(19, 2048): 238.3, (19, 32768): 1358.1, (11, 2048): 143.2, (11, 65536): 1507.2}
DECIDE_KERNELS = ("stock_fp8_s32", "stock_fp8_auto", "triton_std_fp8", "triton_cmp_fp8")


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def copy_gbps(res: Path) -> float | None:
    vals = [load(p, {}).get("copy_gb_per_s_median") for p in sorted(res.glob("env-*/microbench_*.json"))]
    vals = [v for v in vals if v]
    return statistics.median(vals) if vals else None


def best(rows: list[dict], kernels: tuple[str, ...], b: int, length: int) -> dict | None:
    """The fastest ok configuration of any of `kernels` at (b, length)."""
    ok = [r for r in rows if r["kernel"] in kernels and r["batch"] == b and r["context"] == length
          and r.get("status") == "ok"]
    return min(ok, key=lambda r: r["us"]) if ok else None


def family(rows, kernels, X):
    """dt(X) for one kernel family at the group batch, with the two best rows."""
    b = GROUPS[X]
    lo, hi = best(rows, kernels, b, 2048), best(rows, kernels, b, X)
    if not lo or not hi:
        return None
    return {"dt_us": round(hi["us"] - lo["us"], 2), "t2k": lo, "tX": hi}


def group(rows, X, fp8: bool, bw: float | None) -> dict:
    sfx = "fp8" if fp8 else "bf16"
    stock = family(rows, (f"stock_{sfx}_s32", f"stock_{sfx}_auto"), X)
    std, cmp_ = family(rows, (f"triton_std_{sfx}",), X), family(rows, (f"triton_cmp_{sfx}",), X)
    out = {"batch": GROUPS[X], "stock": stock, "triton_std": std, "triton_cmp": cmp_}
    if not (stock and std and cmp_) or min(stock["dt_us"], std["dt_us"], cmp_["dt_us"]) <= 0:
        return out | {"ok": False, "why": "a kernel family is missing or has no growth"}
    b = GROUPS[X]
    grow = lambda compact: b * (X - 2048) * 4 * (640 if compact else 1024) * (1 if fp8 else 2)   # bytes on dt
    std_bw = grow(False) / (std["dt_us"] * 1e-6) / 1e9
    return out | {"ok": True,
                  "sigma_dq": round(1 - cmp_["dt_us"] / stock["dt_us"], 4),
                  "sigma_mma": round(1 - cmp_["dt_us"] / ((1 - MMA_GAIN) * stock["dt_us"]), 4),
                  "sigma_layout": round(1 - cmp_["dt_us"] / std["dt_us"], 4),
                  "std_vs_stock": round(std["dt_us"] / stock["dt_us"], 4),
                  "dt_gbps": {"stock": round(grow(False) / (stock["dt_us"] * 1e-6) / 1e9, 1), "triton_std": round(std_bw, 1),
                              "triton_cmp": round(grow(True) / (cmp_["dt_us"] * 1e-6) / 1e9, 1)},
                  "std_memory_bound": bw is not None and std_bw >= MEM_BOUND * bw}


def checks(res: Path, kb: dict, run: dict, bw: float | None) -> dict:
    rows = kb.get("cells", [])
    corr = kb.get("correctness") or {}
    used = [r for r in rows if r["kernel"] in DECIDE_KERNELS and r.get("status") == "ok"]
    errs = {}
    for entry in corr.values():
        for name, val in entry.items():
            if isinstance(val, dict) and name in DECIDE_KERNELS:
                errs[name] = val.get("err")
    v3 = {}
    for (b, length), target in TARGETS.items():
        r = next((r for r in rows if r["kernel"] == "stock_fp8_s32" and r["batch"] == b and r["context"] == length
                  and r.get("status") == "ok"), None)
        v3[f"{b}:{length}"] = round(r["us"] / target, 4) if r else None
    gpu = (run.get("gpu") or {}).get("name", "")
    return {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""),
            "V2": len(errs) == len(DECIDE_KERNELS) and all(e is not None and e <= TOL_CORRECT for e in errs.values()),
            "V3": all(r is not None and abs(r - 1) <= V3_TOL for r in v3.values()),
            "V4": bool(used) and all(r["spread"] <= SPREAD_MAX for r in used),
            "V5": bw is not None and all(r["gbps"] <= BW_MAX * bw for r in used),
            "V6": telemetry_ok(res),
            "_errors": errs, "_engine_ratio": v3}


def decide(s: dict) -> str:
    if s.get("copy_gbps") is None:
        return "INVALID (no copy bandwidth: V5 cannot run)"
    if s["run_status"] != "ok":
        return f"INCONCLUSIVE (the benchmark did not finish: {s['run_status']})"
    bad = [k for k, v in s["checks"].items() if not k.startswith("_") and not v]
    if bad:
        return f"INVALID ({', '.join(bad)})"
    g = s["fp8"]
    tag = "; ".join(f"{X}: sigma_dq {v['sigma_dq']:.3f}, sigma_mma {v['sigma_mma']:.3f}, sigma_layout "
                    f"{v['sigma_layout']:.3f} at B={v['batch']}" for X, v in g.items() if v.get("ok"))
    if not all(v.get("ok") for v in g.values()):
        return f"INCONCLUSIVE (a deciding group is missing: {[X for X, v in g.items() if not v.get('ok')]}; {tag})"
    if not all(v["std_memory_bound"] for v in g.values()):
        return f"INCONCLUSIVE (the Triton standard kernel reads below {MEM_BOUND} x copy, so the layout test is weak; {tag})"
    go = [X for X, v in g.items() if v["sigma_mma"] >= GO_SIGMA and v["sigma_layout"] >= KILL[int(X)]]
    if go:
        return f"GO ({tag})"
    if all(v["sigma_layout"] < KILL[int(X)] for X, v in g.items()):
        return f"KILL ({tag})"
    reasons = []
    if any(v["sigma_layout"] >= GO_SIGMA for v in g.values()):
        reasons.append("kernel quality: the layout works in Triton, but the compact kernel does not beat the stock bar")
    else:
        reasons.append("below the bar")
    return f"PIVOT ({'; '.join(reasons)}; {tag})"


def summarize(res: Path) -> dict:
    kb, run = load(res / "kbench" / "kbench.json", {}), load(res / "kbench" / "run.json", {})
    bw = copy_gbps(res)
    rows = kb.get("cells", [])
    out = {"copy_gbps": bw, "run_status": run.get("status", "missing"), "gpu": run.get("gpu"),
           "versions": {k: run.get(k) for k in ("torch", "triton", "vllm")},
           "checks": checks(res, kb, run, bw) if kb else {"V1": False},
           "fp8": {str(X): group(rows, X, True, bw) for X in GROUPS},
           "bf16": {str(X): group(rows, X, False, bw) for X in GROUPS},
           "correctness": kb.get("correctness")}
    b4 = {}
    for k in ("stock_fp8_s32", "stock_fp8_auto", "triton_std_fp8", "triton_cmp_fp8", "stock_bf16_auto", "triton_cmp_bf16"):
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
