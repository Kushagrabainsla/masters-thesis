"""E-49 analysis (README Sections 5 to 7, Amendment A1): context slope, the K=V saving share S, validity and the
frozen rule.

Layout under the results root:
  m31/, m26/                 decode_probe.py outputs (run.json, cells.json, kernels.json, events_*.json,
                             gsm8k.jsonl, log.txt)
  env-*/telemetry.csv        nvidia-smi samples
  env-*/microbench_*.json    gpu_microbench.py (copy GB/s)

Mechanism of the metric: for L >= 1,024 the sliding layers read a fixed window and the weights do not depend
on L, so the growth of the decode step time with L is the global-layer attention. A least-squares line
T(L) = c + b L over the slope-cell medians gives b. The compact K=V cache reads 37.5% fewer global KV bytes, so
the saving share at context L is S = 0.375 b L / T(L), an upper bound for the real saving. The rule uses the
one-sided 95% lower bound of b with the cell as the unit. The profiles check the premise: the growth is
attention device time (V4a, V4b) and the sliding-layer attention stays flat (V4c, from the launch order).

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
MODELS = {                           # global KV bytes per token (BF16, K and V), slope batch, layer count
    "m31": {"name": "RedHatAI/gemma-4-31B-it-FP8-dynamic", "global_bytes": 81920, "batch": 4, "layers": 60},
    "m26": {"name": "google/gemma-4-26B-A4B-it", "global_bytes": 20480, "batch": 8, "layers": 30},
}
SLOPE_CONTEXTS = [2048, 8192, 32768, 65536]
DECIDE_CONTEXTS = [32768, 65536]
PROFILE_CONTEXTS = [2048, 32768, 65536]
GO_S, KILL_S = 0.05, 0.02
M1_FRAC, M1_MAX = 0.6, 1.15          # global attention bandwidth: memory-bound at 60%, impossible above 115% (V5)
V3_RESID, M2_LOCAL = 0.05, 0.25      # V3 residual limit; M2: the 32K to 64K slope within 25% of the fitted slope
V4_FRAC, V4C_TOL = 0.8, 0.05
SPROF_GAP = 0.25
V6_ACC = 0.5
V7_SPREAD = 0.02
ATTN = re.compile(r"flash|fmha|attn|attention|paged|splitkv|split_kv|reduce_segments|merge_attn|decode_kernel", re.I)
KV_WRITE = re.compile(r"reshape_and_cache|concat_and_cache|cache_kernel", re.I)   # O(B) KV writes, not attention
PREP = re.compile(r"prepare|scheduler_metadata|num_splits", re.I)                 # per-step metadata kernels
COPY = re.compile(r"memcpy|memset|direct_copy|copy_kernel|CatArrayBatchedCopy|" + KV_WRITE.pattern, re.I)
CLASSES = [  # exploratory waste table; first match wins (copies first, so the KV write is not attention)
    ("copy_cache", COPY),
    ("attention", ATTN),
    ("moe", re.compile(r"moe|expert|topk_softmax|align_block|grouped_gemm|marlin_moe", re.I)),
    ("gemm", re.compile(r"gemm|xmma|cutlass|nvjet|matmul|scaled_mm|sm90_|sm100_|cublas|wgmma", re.I)),
    ("norm_act_rope", re.compile(r"norm|gelu|silu|act_and_mul|rotary|rope|triton_(poi|red|per)_", re.I)),
    ("logits_sampling", re.compile(r"softcap|tanh|sample|argmax|top_k|topk|logits|penalt", re.I)),
]
TEXT_ONLY = re.compile(r"text-only mode", re.I)
KV_TOKENS = re.compile(r"GPU KV cache size: ([\d,]+) tokens")
MAX_CONC = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x")
ENGINE_LINE = re.compile(r"backend|FA4|FlashAttention|flash_attn|text-only|mm_prefix", re.I)
# One-sided 95% t quantiles by degrees of freedom; normal value beyond the table.
T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860, 9: 1.833, 10: 1.812,
       11: 1.796, 12: 1.782, 13: 1.771, 14: 1.761, 15: 1.753, 16: 1.746, 17: 1.740, 18: 1.734, 19: 1.729, 20: 1.725}


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def fit(points: list[tuple[float, float]]) -> dict:
    """Ordinary least squares y = c + b x with the one-sided 95% bounds of b."""
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
    return {"b": b, "c": c, "b_low": b - t * se, "b_up": b + t * se, "se_b": se, "r2": 1 - sse / sst, "n": n, "df": df}


def is_attention(name: str) -> bool:
    return bool(ATTN.search(name)) and not KV_WRITE.search(name)


def key_of(cell: dict) -> str:
    rep = cell.get("rep", 0)
    return f"{cell['context']}:{cell['batch']}" + (f":{rep}" if rep else "")


def copy_gbps(res: Path) -> float | None:
    vals = [load(p, {}).get("copy_gb_per_s_median") for p in sorted(res.glob("env-*/microbench_*.json"))]
    vals = [v for v in vals if v]
    return statistics.median(vals) if vals else None


def dev_ms(table: dict) -> float:
    return table.get("total_device_us", 0) / (table.get("steps") or 1) / 1000.0


def att_ms(table: dict) -> float:
    return sum(k["device_us"] for k in table["kernels"] if is_attention(k["name"])) / (table.get("steps") or 1) / 1000.0


def classes(table: dict) -> dict:
    out: dict[str, float] = {}
    for k in table.get("kernels", []):
        name = next((c for c, rx in CLASSES if rx.search(k["name"])), "other")
        out[name] = out.get(name, 0.0) + k["device_us"]
    total = sum(out.values()) or 1.0
    return {c: round(v / total, 4) for c, v in sorted(out.items(), key=lambda kv: -kv[1])}


def layer_split(ev: dict | None, steps: int, types: list[str]) -> dict:
    """Attention device ms per step of the sliding and the global layers, from the launch order.

    Each layer launches one group of consecutive attention kernels (a main kernel, maybe a combine kernel).
    Group i of the profile belongs to layer i mod n. The split is used only if the profile has exactly
    steps x n groups and at least 90% of the groups have the first-kernel name of the same layer in step 1.
    """
    if not ev or not types:
        return {"ok": False, "why": "no events or no layer types"}
    names, groups, last = ev["names"], [], None
    for r in ev["events"]:
        name = names[r[1]]
        if not is_attention(name) or PREP.search(name):
            continue
        if groups and r[0] == last + 1:
            groups[-1].append(r)
        else:
            groups.append([r])
        last = r[0]
    n = len(types)
    if len(groups) != steps * n:
        return {"ok": False, "why": f"{len(groups)} attention groups, expected {steps * n}"}
    match = sum(groups[i][0][1] == groups[i % n][0][1] for i in range(len(groups))) / len(groups)
    if match < 0.9:
        return {"ok": False, "why": f"kernel order differs between steps (match {match:.2f})"}
    ms = {"sliding": 0.0, "global": 0.0}
    for i, g in enumerate(groups):
        ms["sliding" if "sliding" in types[i % n] else "global"] += sum(r[3] for r in g) / 1000.0
    return {"ok": True, "groups": len(groups), "match": round(match, 3),
            "sliding_ms": round(ms["sliding"] / steps, 4), "global_ms": round(ms["global"] / steps, 4)}


def attribution(d: Path, tables: dict, batch: int, med: dict, types: list[str]) -> dict:
    """V4a and V4b over the 2K to 32K growth; V4c and the global attention time from the layer split."""
    out: dict = {"V4a": None, "V4b": None, "V4c": None, "split": {}}
    for L in PROFILE_CONTEXTS:
        t = tables.get(f"{L}:{batch}")
        if t:
            ev = load(d / t["events"], None) if t.get("events") else None
            out["split"][str(L)] = layer_split(ev, t.get("steps") or 1, types)
    lo, hi = tables.get(f"2048:{batch}"), tables.get(f"32768:{batch}")
    if not lo or not hi or 2048 not in med or 32768 not in med:
        return out
    dd = dev_ms(hi) - dev_ms(lo)
    dw = med[32768] - med[2048]           # unprofiled block medians: the profile wall has the profiler start
    a_hi = att_ms(hi)
    da = a_hi - att_ms(lo)
    growth: dict[str, float] = {}
    for t, sign in ((hi, 1), (lo, -1)):
        for k in t["kernels"]:
            growth[k["name"]] = growth.get(k["name"], 0.0) + sign * k["device_us"] / (t.get("steps") or 1) / 1000.0
    top = sorted(growth.items(), key=lambda kv: -kv[1])[:5]
    out.update(device_growth_ms=round(dd, 4), wall_growth_ms=round(dw, 4), attention_growth_ms=round(da, 4),
               V4a=dw > 0 and dd / dw >= V4_FRAC,
               V4b=None if a_hi == 0 else dd > 0 and da / dd >= V4_FRAC,   # no attention kernel matched: unknown
               top_growth=[[n[:120], round(v, 4)] for n, v in top])
    s0 = out["split"].get("2048")
    if s0 and s0["ok"] and s0["sliding_ms"] > 0:
        change = {L: s["sliding_ms"] / s0["sliding_ms"] - 1 for L, s in out["split"].items() if L != "2048" and s["ok"]}
        out["sliding_change"] = {L: round(v, 4) for L, v in change.items()}
        if "32768" in change:
            out["V4c"] = all(abs(v) <= V4C_TOL for v in change.values())
    return out


def model_summary(res: Path, key: str, bw: float | None) -> dict:
    spec, d = MODELS[key], res / key
    run, cells, tables = load(d / "run.json", {}), load(d / "cells.json", []), load(d / "kernels.json", {})
    if run.get("status") not in ("ok", "ok_slope") or not cells:
        return {"status": run.get("status", "missing"), "valid": False, "why": ["run not ok"]}
    B = spec["batch"]
    slope = [c for c in cells if c["batch"] == B and c["context"] in SLOPE_CONTEXTS and c.get("status", "ok") == "ok"
             and c["step_ms"]]
    preempted = [key_of(c) for c in cells if c.get("status") == "preempted"]
    cell_med = [(c["context"], statistics.median(c["step_ms"])) for c in slope]
    ctxs = sorted({L for L, _ in cell_med})
    f = fit(cell_med) if len(ctxs) >= 3 and len(cell_med) >= 4 else None     # the cell is the unit
    fb = fit([(c["context"], ms) for c in slope for ms in c["step_ms"]]) if f else None   # block level, reported
    med = {L: statistics.median([m for x, m in cell_med if x == L]) for L in ctxs}
    steps = slope[0]["steps_per_block"] if slope else 0
    v2 = bool(slope) and all(lo >= steps - 1 and hi <= steps + 1 for c in slope for lo, hi in c["advance_min_max"])
    resid = {L: (med[L] - (f["c"] + f["b"] * L)) / med[L] for L in med} if f else {}
    v3 = bool(f) and all(abs(r) <= V3_RESID for r in resid.values())
    local = None
    if f and f["b"] > 0 and 32768 in med and 65536 in med:
        local = (med[65536] - med[32768]) / 32768 / f["b"]
    m2 = local is None or abs(local - 1) <= M2_LOCAL     # gate for a pass: no curvature at long context
    spread = {key_of(c): max(c["step_ms"]) / min(c["step_ms"]) - 1 for c in slope}
    noisy = [k for k, s in spread.items() if s > V7_SPREAD]
    types = (run.get("model") or {}).get("layer_types") or \
        (["sliding_attention"] * 5 + ["full_attention"]) * (spec["layers"] // 6)
    att = attribution(d, tables, B, med, types)
    gsm = run.get("gsm8k", {})
    gpu = run.get("gpu", {}).get("name", "")
    log = (d / "log.txt").read_text(errors="replace") if (d / "log.txt").exists() else ""
    bw_glob = spec["global_bytes"] * B / (f["b"] / 1000.0) / 1e9 if f and f["b"] > 0 else None   # GB/s
    sure = bool(f) and f["b_low"] > 0 and bw_glob is not None    # V4a, V5 and M1 need a clear slope
    checks = {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""), "V2": v2, "V3": v3,
              "V4a": not sure or att["V4a"] is True,
              "V5": bw is not None and (not sure or bw_glob <= M1_MAX * bw),
              "V6": gsm.get("accuracy", 0) >= V6_ACC, "V7": len(noisy) <= 1, "V9": bool(TEXT_ONLY.search(log))}
    m1 = sure and bw is not None and bw_glob >= M1_FRAC * bw
    decide = {}
    for L in DECIDE_CONTEXTS:
        if f and L in med:
            row = {"T_ms": round(med[L], 4), "S": round(SAVING * f["b"] * L / med[L], 4),
                   "S_low": round(SAVING * f["b_low"] * L / med[L], 4), "S_up": round(SAVING * f["b_up"] * L / med[L], 4),
                   "S_block_low": round(SAVING * fb["b_low"] * L / med[L], 4)}
            sp = att["split"].get(str(L))
            if sp and sp["ok"]:
                row["S_prof"] = round(SAVING * sp["global_ms"] / med[L], 4)
                row["S_prof_gap"] = abs(row["S_prof"] - row["S"]) > SPROF_GAP * abs(row["S"])
            decide[str(L)] = row
    by_key = {key_of(c): c for c in cells if c.get("step_ms")}
    waste = {}
    for name, t in tables.items():
        dev, c = dev_ms(t), by_key.get(name)
        wall = statistics.median(c["step_ms"]) if c else None   # the same cell, unprofiled blocks
        waste[name] = {"classes": classes(t), "device_ms_per_step": round(dev, 3),
                       "wall_ms_per_step": round(wall, 3) if wall else None,
                       "idle_share": round(1 - dev / wall, 4) if wall else None,
                       "profile_wall_ms_per_step": round(t.get("wall_ms", 0) / (t.get("steps") or 1), 3),
                       "top": [[k["name"][:100], k["share"]] for k in t["kernels"][:8]]}
    explore = {}
    for c in cells:   # exploratory: S at another batch, predicted from the slope per sequence (byte model)
        if c.get("kind") == "waste" and c["context"] > 1024 and c.get("step_ms") and f:
            T = statistics.median(c["step_ms"])
            explore[key_of(c)] = {"T_ms": round(T, 4), "S_est": round(SAVING * f["b"] / B * c["batch"] * c["context"] / T, 4)}
    kv_tokens = [int(x.replace(",", "")) for x in KV_TOKENS.findall(log)]
    conc = MAX_CONC.findall(log)
    lines = list(dict.fromkeys(ln.strip()[:240] for ln in log.splitlines() if ENGINE_LINE.search(ln)))[:30]
    why = [k for k, v in checks.items() if not v]
    return {"status": run["status"], "valid": not why, "why": why, "checks": checks, "fit": f, "fit_blocks": fb,
            "median_ms": med, "residuals": {str(k): round(v, 4) for k, v in resid.items()},
            "local_slope_ratio": round(local, 4) if local is not None else None,
            "spread": {k: round(v, 4) for k, v in spread.items()}, "preempted": preempted,
            "attribution": att, "V4b": att["V4b"], "V4c": att["V4c"], "M1": m1, "M2": m2,
            "bw_global_gbps": round(bw_glob, 1) if bw_glob else None, "copy_gbps": bw, "decide": decide,
            "exploratory": explore, "gsm8k": gsm, "kv": run.get("kv"), "model": run.get("model"),
            "kv_tokens_logged": kv_tokens[-1] if kv_tokens else None,
            "max_concurrency_logged": float(conc[-1][1]) if conc else None, "engine_lines": lines, "waste": waste}


def decide(s: dict) -> str:
    if s.get("copy_gbps") is None:
        return "INVALID (no copy bandwidth: the M1 and V5 controls cannot run)"
    valid = {k: m for k, m in s["models"].items() if m.get("valid")}
    if not valid:
        bad = "; ".join(f"{k}: {', '.join(m.get('why', []))}" for k, m in s["models"].items())
        return f"INVALID ({bad}; or V8 telemetry)" if not s["V8"] else f"INVALID ({bad})"
    if not s["V8"]:
        return "INVALID (V8 telemetry)"
    cells = [(k, L, c) for k, m in valid.items() for L, c in m["decide"].items()]
    tag = ", ".join(f"{k}@{L} S={c['S']:.3f} (low {c['S_low']:.3f})" for k, L, c in cells)
    one = " (one model)" if len(valid) == 1 else ""
    flags = [f"{k}@{L}" for k, L, c in cells if c.get("S_prof_gap")]
    note = f" [S_prof gap: {', '.join(flags)}]" if flags else ""
    big = [(k, L) for k, L, c in cells if c["S_low"] >= GO_S]
    gate = {k: m["M1"] and m["M2"] and m["V4b"] is True and m["V4c"] is True for k, m in valid.items()}
    passing = [(k, L) for k, L in big if gate[k]]
    each = all(any(p[0] == k for p in passing) for k in valid)
    if len(passing) >= 2 and each:
        return f"GO{one} ({tag}){note}"
    if cells and all(c["S_up"] < KILL_S for _, _, c in cells):
        return f"KILL ({tag}){note}"
    reasons = [f"{k} has no deciding cell" for k, m in valid.items() if not m["decide"]]
    for k in sorted({k for k, _ in big if not gate[k]}):
        m = valid[k]
        if not m["M1"]:
            reasons.append(f"{k} KERNEL-BOUND (M1)")
        if not m["M2"]:
            reasons.append(f"{k} curvature (M2: local slope ratio {m['local_slope_ratio']})")
        if m["V4b"] is None or m["V4c"] is None:
            reasons.append(f"{k} attribution unknown")
        elif not (m["V4b"] and m["V4c"]):
            reasons.append(f"{k} attribution unclear (V4b {m['V4b']}, V4c {m['V4c']})")
    if passing:
        reasons.append("one model or one cell only")
    if not big:
        reasons.append("below the 5% bar")
    return f"PIVOT{one} ({'; '.join(reasons)}; {tag}){note}"


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
