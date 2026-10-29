"""E-49 analysis (README Sections 5 to 7): context slope, the K=V saving share S, validity and the frozen rule.

Layout under the results root:
  m31/, m26/                 decode_probe.py outputs (run.json, cells.json, kernels.json, gsm8k.jsonl, log.txt)
  env-*/telemetry.csv        nvidia-smi samples
  env-*/microbench_*.json    gpu_microbench.py (copy GB/s)

Mechanism of the metric: for L >= 1,024 the sliding layers read a fixed window and the weights do not depend
on L, so the growth of the decode step time with L is the global-layer attention. A least-squares line
T(L) = c + b L over the slope cells gives b. The compact K=V cache reads 37.5% fewer global KV bytes, so
the saving share at context L is S = 0.375 b L / T(L). The rule uses the one-sided 95% lower bound of b.

Usage: analyze_e49.py --res <dir> --out summary.json
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
for sub in ("nvfp4_dispatch", "safedispatch_testA", "safedispatch_testA2", "ledger", "fp8_quant_fusion"):
    sys.path.insert(0, str(HERE.parent / sub))

from analyze_e44 import telemetry_ok  # noqa: E402  (hard throttle in at most 2% of busy samples)

SAVING = 0.375                       # 1 - 640 / 1024 (kv_layout_check.py)
MODELS = {                           # global KV bytes per token (BF16, K and V), slope batch, deciding contexts
    "m31": {"name": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "global_bytes": 81920, "batch": 4},
    "m26": {"name": "google/gemma-4-26B-A4B-it", "global_bytes": 20480, "batch": 8},
}
SLOPE_CONTEXTS = [2048, 8192, 32768, 65536]
DECIDE_CONTEXTS = [32768, 65536]
GO_S, KILL_S = 0.05, 0.02
M1_FRAC = 0.6                        # global attention must reach 60% of the copy bandwidth (memory-bound)
V3_RESID = 0.03
V4_FRAC = 0.8
V6_ACC = 0.5
V7_SPREAD = 0.02
ATTN = re.compile(r"flash|fmha|attn|attention|paged|splitkv|split_kv|reduce_segments|merge_attn|decode_kernel", re.I)
CLASSES = [  # exploratory waste table; first match wins
    ("attention", ATTN),
    ("moe", re.compile(r"moe|expert|topk_softmax|align_block|grouped_gemm|marlin_moe", re.I)),
    ("gemm", re.compile(r"gemm|xmma|cutlass|nvjet|matmul|scaled_mm|sm90_|sm100_|cublas|wgmma", re.I)),
    ("norm_act_rope", re.compile(r"norm|gelu|silu|act_and_mul|rotary|rope|triton_(poi|red|per)_", re.I)),
    ("logits_sampling", re.compile(r"softcap|tanh|sample|argmax|top_k|topk|logits|penalt", re.I)),
    ("copy_cache", re.compile(r"memcpy|memset|copy|reshape_and_cache|cache_kernel|concat|cat_", re.I)),
]
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
BACKEND = re.compile(r"Using (\w+) (?:attention )?backend", re.I)
# One-sided 95% t quantiles by degrees of freedom; normal value beyond the table.
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
       11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725}


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def fit(points: list[tuple[float, float]]) -> dict:
    """Ordinary least squares y = c + b x with the one-sided 95% lower bound of b."""
    n = len(points)
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    c = my - b * mx
    sse = sum((y - c - b * x) ** 2 for x, y in points)
    sst = sum((y - my) ** 2 for y in ys) or 1e-12
    df = n - 2
    se = math.sqrt(sse / df / sxx) if df > 0 else float("inf")
    t = T95.get(df, 1.645)
    return {"b": b, "c": c, "b_low": b - t * se, "se_b": se, "r2": 1 - sse / sst, "n": n, "df": df}


def copy_gbps(res: Path) -> float | None:
    vals = [load(p, {}).get("copy_gb_per_s_median") for p in sorted(res.glob("env-*/microbench_*.json"))]
    vals = [v for v in vals if v]
    return statistics.median(vals) if vals else None


def per_step(table: dict) -> tuple[float, float]:
    steps = table.get("steps") or 1
    return table.get("total_device_us", 0) / steps / 1000.0, table.get("wall_ms", 0) / steps


def classes(table: dict) -> dict:
    out: dict[str, float] = {}
    for k in table.get("kernels", []):
        name = next((c for c, rx in CLASSES if rx.search(k["name"])), "other")
        out[name] = out.get(name, 0.0) + k["device_us"]
    total = sum(out.values()) or 1.0
    return {c: round(v / total, 4) for c, v in sorted(out.items(), key=lambda kv: -kv[1])}


def attribution(tables: dict, batch: int) -> dict:
    """V4: between the 2K and 32K profiles, the step-time growth is device time (V4a) in attention kernels (V4b)."""
    lo, hi = tables.get(f"2048:{batch}"), tables.get(f"32768:{batch}")
    if not lo or not hi:
        return {"V4a": None, "V4b": None}
    (d_lo, w_lo), (d_hi, w_hi) = per_step(lo), per_step(hi)

    def att_ms(t: dict) -> float:
        return sum(k["device_us"] for k in t["kernels"] if ATTN.search(k["name"])) / (t.get("steps") or 1) / 1000.0

    dd, dw, da = d_hi - d_lo, w_hi - w_lo, att_ms(hi) - att_ms(lo)
    growth = {}
    for t, sign in ((hi, 1), (lo, -1)):
        for k in t["kernels"]:
            growth[k["name"]] = growth.get(k["name"], 0.0) + sign * k["device_us"] / (t.get("steps") or 1) / 1000.0
    top = sorted(growth.items(), key=lambda kv: -kv[1])[:5]
    return {"device_growth_ms": round(dd, 4), "wall_growth_ms": round(dw, 4), "attention_growth_ms": round(da, 4),
            "V4a": dw > 0 and dd / dw >= V4_FRAC, "V4b": dd > 0 and da / dd >= V4_FRAC,
            "top_growth": [[n[:120], round(v, 4)] for n, v in top]}


def model_summary(res: Path, key: str, bw: float | None) -> dict:
    spec, d = MODELS[key], res / key
    run, cells, tables = load(d / "run.json", {}), load(d / "cells.json", []), load(d / "kernels.json", {})
    if run.get("status") != "ok" or not cells:
        return {"status": run.get("status", "missing"), "valid": False, "why": ["run not ok"]}
    b_slope = spec["batch"]
    slope = [c for c in cells if c["batch"] == b_slope and c["context"] in SLOPE_CONTEXTS]
    pts = [(c["context"], ms) for c in slope for ms in c["step_ms"]]
    f = fit(pts) if len({c["context"] for c in slope}) >= 3 else None
    steps = slope[0]["steps_per_block"] if slope else 0
    v2 = bool(slope) and all(lo >= steps - 1 and hi <= steps + 1 for c in slope for lo, hi in c["advance_min_max"])
    med = {c["context"]: statistics.median(c["step_ms"]) for c in slope}
    resid = {L: (med[L] - (f["c"] + f["b"] * L)) / med[L] for L in med} if f else {}
    v3 = bool(f) and all(abs(r) <= V3_RESID for r in resid.values())   # R^2 is reported only: a flat T(L) has a low R^2
    spread = {c["context"]: max(c["step_ms"]) / min(c["step_ms"]) - 1 for c in slope}
    noisy = [L for L, s in spread.items() if s > V7_SPREAD]
    att = attribution(tables, b_slope)
    gsm = run.get("gsm8k", {})
    gpu = run.get("gpu", {}).get("name", "")
    checks = {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""), "V2": v2, "V3": v3,
              "V4a": att["V4a"] is True, "V6": gsm.get("accuracy", 0) >= V6_ACC, "V7": len(noisy) <= 1}
    bw_glob = None
    if f and f["b"] > 0:
        bw_glob = spec["global_bytes"] * b_slope / (f["b"] / 1000.0) / 1e9   # GB/s; b is ms per context token
    m1 = bw_glob is not None and bw is not None and bw_glob >= M1_FRAC * bw
    decide = {}
    for L in DECIDE_CONTEXTS:
        if f and L in med:
            decide[str(L)] = {"T_ms": round(med[L], 4), "S": round(SAVING * f["b"] * L / med[L], 4),
                              "S_low": round(SAVING * f["b_low"] * L / med[L], 4)}
    log = (d / "log.txt").read_text(errors="replace") if (d / "log.txt").exists() else ""
    kv_tokens = [int(x.replace(",", "")) for x in KV_TOKENS.findall(log)]
    waste = {}
    for name, t in tables.items():
        dev, wall = per_step(t)
        waste[name] = {"classes": classes(t), "device_ms_per_step": round(dev, 3), "wall_ms_per_step": round(wall, 3),
                       "idle_share": round(1 - dev / wall, 4) if wall else None,
                       "top": [[k["name"][:100], k["share"]] for k in t["kernels"][:8]]}
    why = [k for k, v in checks.items() if not v]
    return {"status": "ok", "valid": not why, "why": why, "checks": checks, "fit": f, "median_ms": med,
            "residuals": {str(k): round(v, 4) for k, v in resid.items()}, "spread": {str(k): round(v, 4) for k, v in spread.items()},
            "attribution": att, "V4b": att["V4b"] is True, "M1": m1, "bw_global_gbps": round(bw_glob, 1) if bw_glob else None,
            "copy_gbps": bw, "decide": decide, "gsm8k": gsm, "kv_tokens_logged": kv_tokens[-1] if kv_tokens else None,
            "attention_backends": sorted(set(BACKEND.findall(log))), "waste": waste}


def decide(s: dict) -> str:
    valid = {k: m for k, m in s["models"].items() if m.get("valid")}
    if not valid:
        bad = "; ".join(f"{k}: {', '.join(m.get('why', []))}" for k, m in s["models"].items())
        return f"INVALID ({bad}; or V8 telemetry)" if not s["V8"] else f"INVALID ({bad})"
    if not s["V8"]:
        return "INVALID (V8 telemetry)"
    cells = [(k, L, c) for k, m in valid.items() for L, c in m["decide"].items()]
    big = [(k, L) for k, L, c in cells if c["S_low"] >= GO_S]
    passing = [(k, L) for k, L in big if valid[k]["M1"] and valid[k]["V4b"]]
    each = all(any(p[0] == k for p in passing) for k in valid)
    tag = ", ".join(f"{k}@{L} S={c['S']:.3f} (low {c['S_low']:.3f})" for k, L, c in cells)
    one = " (one model)" if len(valid) == 1 else ""
    if len(passing) >= 2 and each:
        return f"GO{one} ({tag})"
    if cells and all(c["S"] < KILL_S for _, _, c in cells):
        return f"KILL ({tag})"
    reasons = []
    if len(big) > len(passing):
        reasons.append("KERNEL-BOUND or attribution unclear")
    if passing:
        reasons.append("one model or one cell only")
    if not big:
        reasons.append("below the 5% bar")
    return f"PIVOT{one} ({'; '.join(reasons)}; {tag})"


def summarize(res: Path) -> dict:
    bw = copy_gbps(res)
    return {"models": {k: model_summary(res, k, bw) for k in MODELS}, "V8": telemetry_ok(res), "copy_gbps": bw}


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
