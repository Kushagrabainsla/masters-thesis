"""E-49 free check (no GPU): the compact K=V cache layout of Gemma 4 global layers is exact, and its byte model.

Gemma 4 global (full-attention) layers have no v_proj (`attention_k_eq_v`). With k_raw = W_k x and
n = k_raw / rms(k_raw):
  V = v_norm(k_raw) = n                      (no learnable scale)
  K = rope_p(w * n)                          (k_norm weight w, then proportional RoPE, p = 0.25)
Proportional RoPE with head_dim 512 rotates 64 angle pairs, that is dims 0..63 and 256..319 (neox
rotate_half); the other 384 dims have frequency 0 (identity). vLLM, SGLang and TensorRT-LLM store K and
V (1,024 values per head and token). A compact cache stores [n (512) | K on the 128 rotated dims] = 640
values (37.5% less, the number in the Gemma 4 report). The query absorbs w on the 384 unrotated dims:
  score = (q_rot * w)[N] . n[N] + q_rot[R] . K[R],   output = sum_j p_j n_j.

This script checks the identity in float64 and in BF16 storage on random data with the real shapes,
then prints the byte model for the E-49 cells. Usage:
  uv run --no-project --with numpy python kv_layout_check.py --out results/layout_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEAD = 512
ROPE_ANGLES = int(0.25 * HEAD // 2)          # 64 rotated angle pairs
THETA = 1_000_000.0
ROT = np.r_[0:ROPE_ANGLES, HEAD // 2:HEAD // 2 + ROPE_ANGLES]   # the 128 rotated dims
NOROT = np.setdiff1d(np.arange(HEAD), ROT)

# Per-token KV bytes per layer type (BF16, K and V stored), from the model configs (2026-10-29).
MODELS = {
    "gemma-4-31B-it": {"global_layers": 10, "global_kv_heads": 4, "sliding_layers": 50, "sliding_kv_heads": 16,
                       "sliding_head": 256, "window": 1024, "weight_gb": 31.3, "kind": "dense FP8"},
    # MoE weights read per step: non-expert part plus the expected distinct experts under uniform top-8 routing
    # (an upper bound on bytes, so the saving share below is a lower estimate). 30 x 128 x 3 x 2816 x 704 BF16.
    "gemma-4-26B-A4B-it": {"global_layers": 5, "global_kv_heads": 2, "sliding_layers": 25, "sliding_kv_heads": 8,
                           "sliding_head": 256, "window": 1024, "weight_gb": None, "kind": "MoE BF16",
                           "expert_gb": 45.7, "nonexpert_gb": 5.4, "experts": 128, "top_k": 8},
}
CELLS = {"gemma-4-31B-it": (4, [2048, 8192, 32768, 65536]), "gemma-4-26B-A4B-it": (8, [2048, 8192, 32768, 65536])}


def rope(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Gemma 4 proportional RoPE (neox rotate_half) on the last axis; pos has the shape of x[..., 0]."""
    inv = np.zeros(HEAD // 2)
    inv[:ROPE_ANGLES] = 1.0 / THETA ** (np.arange(0, 2 * ROPE_ANGLES, 2) / HEAD)
    ang = pos[..., None] * inv
    cos, sin = np.concatenate([np.cos(ang)] * 2, -1), np.concatenate([np.sin(ang)] * 2, -1)
    half = np.concatenate([-x[..., HEAD // 2:], x[..., :HEAD // 2]], -1)
    return x * cos + half * sin


def bf16(x: np.ndarray) -> np.ndarray:
    """Round float values to BF16 (round to nearest even) and return them as float64."""
    u = x.astype(np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return u.view(np.float32).astype(np.float64)


def attend(scores: np.ndarray, v: np.ndarray) -> np.ndarray:
    p = np.exp(scores - scores.max(-1, keepdims=True))
    return (p / p.sum(-1, keepdims=True)) @ v


def check(seed: int = 0, keys: int = 3000, qheads: int = 8) -> dict:
    """One KV head with `qheads` query heads (GQA group of the 31B: 32 / 4). Scaling is 1.0 in Gemma 4."""
    rng = np.random.default_rng(seed)
    k_raw = rng.normal(size=(keys, HEAD)) * rng.uniform(0.2, 3.0, size=HEAD)
    n = k_raw / np.sqrt((k_raw**2).mean(-1, keepdims=True) + 1e-6)
    w = rng.uniform(0.5, 1.5, size=HEAD)
    q = rng.normal(size=(qheads, HEAD)) / np.sqrt(HEAD) * 4
    q = q / np.sqrt((q**2).mean(-1, keepdims=True) + 1e-6) * rng.uniform(0.5, 1.5, size=HEAD) / 8
    pos_k = np.arange(keys, dtype=np.float64)
    q_rot = rope(q, np.full(qheads, float(keys)))
    k = rope(w * n, pos_k)
    ref = attend(q_rot @ k.T, n)                                   # standard layout, float64
    qa = q_rot.copy()
    qa[:, NOROT] *= w[NOROT]
    qa[:, ROT] = 0.0
    cache = np.concatenate([n, k[:, ROT]], -1)                     # compact layout: 640 values per key
    comp = attend(qa @ cache[:, :HEAD].T + q_rot[:, ROT] @ cache[:, HEAD:].T, cache[:, :HEAD])
    std16 = attend(q_rot @ bf16(k).T, bf16(n))                     # BF16 cache, standard
    c16 = bf16(cache)
    comp16 = attend(qa @ c16[:, :HEAD].T + q_rot[:, ROT] @ c16[:, HEAD:].T, c16[:, :HEAD])
    scale = np.abs(ref).max()
    return {"keys": keys, "qheads": qheads, "rotated_dims": int(len(ROT)), "values_per_key_standard": 2 * HEAD,
            "values_per_key_compact": int(cache.shape[1]), "saving": 1 - cache.shape[1] / (2 * HEAD),
            "max_rel_err_float64": float(np.abs(comp - ref).max() / scale),
            "max_rel_err_bf16_standard": float(np.abs(std16 - ref).max() / scale),
            "max_rel_err_bf16_compact": float(np.abs(comp16 - ref).max() / scale)}


def byte_model() -> dict:
    """Decode bytes per step: weights (given), sliding KV (window), global KV (context), at the E-49 cells."""
    out = {}
    for name, m in MODELS.items():
        g = m["global_layers"] * m["global_kv_heads"] * HEAD * 2 * 2
        s = m["sliding_layers"] * m["sliding_kv_heads"] * m["sliding_head"] * 2 * 2
        b, ctxs = CELLS[name]
        rows = []
        for ctx in ctxs:
            glob, slid = b * ctx * g, b * min(ctx, m["window"]) * s
            row = {"context": ctx, "batch": b, "global_kv_gb": round(glob / 1e9, 2), "sliding_kv_gb": round(slid / 1e9, 2)}
            weight_gb = m["weight_gb"]
            if weight_gb is None:
                touched = 1 - (1 - m["top_k"] / m["experts"]) ** b
                weight_gb = m["nonexpert_gb"] + touched * m["expert_gb"]
            if weight_gb:
                row["weight_gb"] = round(weight_gb, 1)
                total = glob + slid + weight_gb * 1e9
                row["global_share_of_bytes"] = round(glob / total, 3)
                row["saving_share_of_bytes"] = round(0.375 * glob / total, 3)
            rows.append(row)
        out[name] = {"kind": m["kind"], "global_kv_bytes_per_token": g, "sliding_kv_bytes_per_token": s, "cells": rows}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {"identity": [check(seed) for seed in range(3)], "bytes": byte_model()}
    ok = all(r["max_rel_err_float64"] < 1e-12 and r["max_rel_err_bf16_compact"] < 3 * r["max_rel_err_bf16_standard"] + 1e-6
             for r in res["identity"])
    res["exact"] = ok
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
