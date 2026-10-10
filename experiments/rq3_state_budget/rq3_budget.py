#!/usr/bin/env python3
"""RQ3: does the best state-class allocation differ across real hybrid models?

The handoff deprioritized a general state planner because, under hand-chosen
NORMALIZED costs, the best simple per-case policy matched it in 86.7% of cases.
This script redoes the question with real per-model constants, and against a
STRONG baseline rather than a naive one:

  * checkpoint placement is optimized by dynamic program for the given
    divergence-depth distribution (this is what Sparse Prefix Caching does),
    not assumed evenly spaced;
  * recurrent state bytes use the MEASURED float32 element size;
  * replay cost comes from each model's own module shapes on the meta device.

Two strategies compete for one byte budget B:
  A. spend all of B on recurrent checkpoints, placed optimally;
  B. pay O(context) for a full-attention post-block activation cache, which
     lets replay skip full-attention blocks, and spend the remainder on
     optimally placed checkpoints.

If A wins everywhere for every model and context, RQ3 dies cleanly.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODELS = [
    "Qwen/Qwen3.5-0.8B-Base",
    "ibm-granite/granite-4.0-h-small",
    "nvidia/Nemotron-H-8B-Base-8K",
]
REC_ELEM_BYTES = 4   # measured: SSM/GDN recurrent state is float32
KV_ELEM_BYTES = 2    # fp16
ACT_ELEM_BYTES = 2   # fp16
GRID = 48            # candidate checkpoint positions for the DP


def model_profile(model_id: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_id)
    tc = getattr(cfg, "text_config", cfg)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)

    lt = getattr(tc, "layer_types", None) or getattr(tc, "layers_block_type", None)
    lt = [str(x) for x in lt]
    # Three classes, not two. Nemotron-H lists 24 linear_attention + 24 mlp + 4
    # full_attention, so "everything that is not attention" would count the MLP
    # blocks as state-holding and double the snapshot size.
    is_fa = [("full" in t or t == "attention") for t in lt]
    is_rec = [("linear" in t or "mamba" in t or "recurrent" in t) for t in lt]
    unknown = [t for t, f, r in zip(lt, is_fa, is_rec) if not f and not r
               and "mlp" not in t]
    if unknown:
        raise RuntimeError(f"unclassified layer types for {model_id}: {set(unknown)}")

    per_layer = defaultdict(int)
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue
        idx = int(parts[parts.index("layers") + 1])
        per_layer[idx] += int(mod.in_features) * int(mod.out_features)

    fa_w = sum(per_layer[i] for i, f in enumerate(is_fa) if f)
    # Everything that is not a full-attention block must be replayed, so MLP
    # block weights belong in rec_w even though they hold no state.
    rec_w = sum(per_layer[i] for i, f in enumerate(is_fa) if not f)
    n_fa = sum(is_fa)
    n_rec = sum(is_rec)          # state-holding layers only
    n_other = len(lt) - n_fa - n_rec

    hidden = int(tc.hidden_size)
    n_q = int(getattr(tc, "num_attention_heads", 0) or 0)
    hd = int(getattr(tc, "head_dim", 0) or (hidden // n_q if n_q else 0))
    n_kv = int(getattr(tc, "num_key_value_heads", n_q) or n_q)
    q_width = n_q * hd

    kv_per_tok = n_fa * 2 * n_kv * hd * KV_ELEM_BYTES
    act_per_tok = n_fa * hidden * ACT_ELEM_BYTES

    # recurrent snapshot bytes
    lk_h = getattr(tc, "linear_num_key_heads", None)
    lk_d = getattr(tc, "linear_key_head_dim", None)
    lv_d = getattr(tc, "linear_value_head_dim", None)
    if lk_h and lk_d and lv_d:
        rec_bytes = n_rec * int(lk_h) * int(lk_d) * int(lv_d) * REC_ELEM_BYTES
        rec_extra_mac = n_rec * 2 * int(lk_h) * int(lk_d) * int(lv_d)
    else:
        # Mamba2 SSM state is [n_heads, head_dim, d_state] per layer. n_heads *
        # head_dim is the MAMBA inner width, which is NOT the MLP
        # intermediate_size; using the latter overstates the snapshot badly.
        # Nemotron-H spells these mamba_num_heads / mamba_head_dim /
        # ssm_state_size, Granite spells them mamba_n_heads / mamba_d_head /
        # mamba_d_state, so both spellings are tried.
        m_h = int(getattr(tc, "mamba_n_heads", 0) or getattr(tc, "mamba_num_heads", 0) or 0)
        m_hd = int(getattr(tc, "mamba_d_head", 0) or getattr(tc, "mamba_head_dim", 0) or 0)
        d_st = int(getattr(tc, "mamba_d_state", 0) or getattr(tc, "ssm_state_size", 0)
                   or getattr(tc, "d_state", 0) or 0)
        if not (m_h and m_hd and d_st):
            raise RuntimeError(f"cannot resolve mamba state dims for {model_id}; "
                               f"refusing to guess")
        rec_bytes = n_rec * m_h * m_hd * d_st * REC_ELEM_BYTES
        rec_extra_mac = n_rec * 2 * m_h * m_hd * d_st

    return {
        "model_id": model_id, "n_fa": n_fa, "n_rec": n_rec, "n_other": n_other,
        "layer_type_counts": {t: lt.count(t) for t in sorted(set(lt))}, "hidden": hidden,
        "fa_weight_mac": fa_w, "rec_weight_mac": rec_w, "q_width": q_width,
        "rec_extra_mac_per_tok": rec_extra_mac,
        "kv_bytes_per_tok": kv_per_tok, "act_bytes_per_tok": act_per_tok,
        "recurrent_snapshot_bytes": rec_bytes,
    }


def replay_mac(p, suffix, context, skip_fa):
    if suffix <= 0:
        return 0.0
    k = context - suffix
    causal_pairs = suffix * k + suffix * (suffix + 1) / 2.0
    fa = suffix * p["fa_weight_mac"] + p["n_fa"] * 2.0 * causal_pairs * p["q_width"]
    rec = suffix * p["rec_weight_mac"] + suffix * p["rec_extra_mac_per_tok"]
    return rec if skip_fa else rec + fa


def depth_pmf(kind, n, grid):
    """Probability that a request diverges at each grid depth."""
    x = (grid + 0.5) / n
    if kind == "uniform":
        w = np.ones_like(x)
    elif kind == "front_loaded":       # most requests diverge early
        w = (1 - x) ** 3 + 1e-9
    elif kind == "back_loaded":        # long shared prefix, late divergence
        w = x ** 3 + 1e-9
    elif kind == "bimodal":            # shared system prompt, then a long tail
        w = np.exp(-((x - 0.1) ** 2) / 0.005) + np.exp(-((x - 0.9) ** 2) / 0.02)
    else:
        raise ValueError(kind)
    return w / w.sum()


def best_expected_cost(p, n, m, pmf, grid, skip_fa):
    """Minimum expected replay cost using at most m optimally placed checkpoints.

    Choose 0 = q0 < q1 < ... < qc from the grid with c <= m, where q0 = 0 means
    "no checkpoint, replay from the start". A divergence at depth d is served by
    the largest qk <= d and replays d - qk tokens. Cost is additive over
    segments, so optimal placement is a DP on the depth grid:

        seg[i][j] = expected cost of depths with index in [i, j), all served by
                    the checkpoint at the left edge of segment i
        g(i, c)   = min cost to cover depths [i, G) given a checkpoint at the
                    left edge of i and c checkpoints still available
        g(i, 0)   = seg[i][G]
        g(i, c)   = min( seg[i][G], min_{j>i} seg[i][j] + g(j, c-1) )

    The answer is g(0, m). This is Sparse Prefix Caching style optimal
    placement, so the comparison is against a strong baseline rather than
    evenly spaced checkpoints.
    """
    G = len(grid)
    seg = [[0.0] * (G + 1) for _ in range(G + 1)]
    for i in range(G + 1):
        base = 0.0 if i == 0 else float(grid[i - 1])
        run = 0.0
        for j in range(i, G):
            run += float(pmf[j]) * replay_mac(p, float(grid[j]) - base, n, skip_fa)
            seg[i][j + 1] = run

    memo: dict[tuple[int, int], float] = {}

    def g(i: int, c: int) -> float:
        if i >= G:
            return 0.0
        if c == 0:
            return seg[i][G]
        key = (i, c)
        if key in memo:
            return memo[key]
        best = seg[i][G]
        for j in range(i + 1, G):
            cand = seg[i][j] + g(j, c - 1)
            if cand < best:
                best = cand
        memo[key] = best
        return best

    return float(g(0, max(0, int(m))))


def main():
    profiles = [model_profile(m) for m in MODELS]
    rows = []
    print(f"{'model':<34} {'snap MiB':>9} {'KV B/tok':>9} {'act B/tok':>10}")
    for p in profiles:
        print(f"{p['model_id']:<34} {p['recurrent_snapshot_bytes']/2**20:>9.2f} "
              f"{p['kv_bytes_per_tok']:>9} {p['act_bytes_per_tok']:>10}")
    print()

    for p in profiles:
        R = p["recurrent_snapshot_bytes"]
        for n in (2048, 8192, 32768):
            grid = np.array([int(round(n * (i + 1) / GRID)) for i in range(GRID)])
            act_bytes = p["act_bytes_per_tok"] * n
            for dist in ("uniform", "front_loaded", "back_loaded", "bimodal"):
                pmf = depth_pmf(dist, n, grid)
                for budget_snaps in (1, 2, 4, 8, 16, 32):
                    B = budget_snaps * R
                    m_a = int(B // R)
                    cost_a = best_expected_cost(p, n, m_a, pmf, grid, skip_fa=False)
                    rem = B - act_bytes
                    if rem < 0:
                        cost_b, m_b = float("inf"), None
                    else:
                        m_b = int(rem // R)
                        cost_b = best_expected_cost(p, n, m_b, pmf, grid, skip_fa=True)
                    winner = "checkpoints" if cost_a <= cost_b else "activations"
                    rows.append({
                        "model_id": p["model_id"], "context_len": n,
                        "distribution": dist, "budget_snapshots": budget_snaps,
                        "budget_bytes": B, "ckpt_only_m": m_a,
                        "ckpt_only_expected_gmac": cost_a / 1e9,
                        "act_plus_ckpt_m": m_b,
                        "act_plus_ckpt_expected_gmac": (None if cost_b == float("inf")
                                                        else cost_b / 1e9),
                        "activation_cache_bytes": act_bytes,
                        "winner": winner,
                        "margin_pct": (None if cost_b == float("inf") or cost_a == 0
                                       else 100.0 * (cost_a - cost_b) / cost_a),
                    })

    with open("results_rq3_budget.json", "w") as f:
        json.dump({
            "models": profiles,
            "grid_points": GRID,
            "recurrent_state_elem_bytes": REC_ELEM_BYTES,
            "baseline": "checkpoint placement optimized by DP per distribution",
            "caveat": ("Analytical MAC model with measured byte constants. The four "
                       "divergence-depth distributions are synthetic sensitivity "
                       "probes, not a production trace."),
            "rows": rows,
        }, f, indent=2)

    # Report only where the winner flips, that is what RQ3 hinges on.
    print("Winner by model / context / distribution, over budget 1..32 snapshots:")
    print(f"{'model':<20} {'ctx':>7} {'distribution':<14} {'winner pattern':<42} flips")
    flips_total = 0
    for p in profiles:
        short = p["model_id"].split("/")[-1][:18]
        for n in (2048, 8192, 32768):
            for dist in ("uniform", "front_loaded", "back_loaded", "bimodal"):
                sel = [r for r in rows if r["model_id"] == p["model_id"]
                       and r["context_len"] == n and r["distribution"] == dist]
                pat = "".join("C" if r["winner"] == "checkpoints" else "A" for r in sel)
                flips = sum(1 for a, b in zip(pat, pat[1:]) if a != b)
                flips_total += flips
                print(f"{short:<20} {n:>7} {dist:<14} {pat:<42} {flips}")
    print(f"\ntotal winner flips across all cells: {flips_total}")
    print("C = optimally placed checkpoints win, A = activation cache wins")
    print("budget order within each row: 1, 2, 4, 8, 16, 32 snapshots")


if __name__ == "__main__":
    main()
