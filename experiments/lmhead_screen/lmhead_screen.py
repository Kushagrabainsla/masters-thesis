"""E-48 screen (SCREEN ONLY, no GPU): how many BF16 LM-head rows must be re-read to keep a low-precision head exact.

See README.md for the two designs and the frozen rule.
Usage: uv run --no-project --with torch --with transformers --with ml_dtypes python lmhead_screen.py \
           --model Qwen/Qwen3-0.6B --out results/screen.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import ml_dtypes
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips",
    "A train leaves the station at 3 pm travelling at 60 miles per hour. Another train leaves one hour later at",
    "def merge_sorted(a, b):\n    \"\"\"Merge two sorted lists into one sorted list.\"\"\"\n",
    "SELECT customer_id, COUNT(*) AS orders FROM orders WHERE created_at >= '2026-01-01'",
    "User: Can you explain what a GPU kernel is in simple words?\nAssistant:",
    "User: Write a short poem about the ocean at night.\nAssistant:",
    "The capital of Australia is",
    "Photosynthesis is the process by which plants",
    "In distributed training, a checkpoint stores the model weights and the optimizer state so that",
    "The French Revolution began in 1789 when",
    "To install the package, run the following command in your terminal:",
    "import torch\nimport torch.nn as nn\n\nclass MLP(nn.Module):\n    def __init__(self, d, h):\n",
    "Q: What is the derivative of x^3 + 2x?\nA:",
    "Once upon a time, in a small village near the mountains, there lived",
    "The main difference between TCP and UDP is that",
    "Dear hiring manager, I am writing to apply for the position of",
    "Recipe: chocolate chip cookies. Ingredients: 2 cups flour,",
    "The patient presented with a fever and a cough. The doctor ordered",
    "Translate to French: The weather is nice today.\nFrench:",
    "Breaking news: the central bank announced on Tuesday that interest rates",
    "fn main() {\n    let v: Vec<i32> = (1..10).collect();\n",
    "The theory of general relativity predicts that",
    "Step 1: Preheat the oven to 350 degrees. Step 2:",
    "User: Summarize the plot of Romeo and Juliet in two sentences.\nAssistant:",
]
NEW_TOKENS = 48
K_TAIL = 256
K_CAP = 20000


def quantize(e: np.ndarray, kind: str) -> np.ndarray:
    if kind == "fp8":
        s = np.abs(e).max(axis=1, keepdims=True) / 448.0 + 1e-30
        return (e / s).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * s
    v, d = e.shape
    blk = e.reshape(v, d // 16, 16)
    amax = np.abs(blk).max(axis=2, keepdims=True)
    g = float(np.abs(e).max()) / (6.0 * 448.0)                  # NVFP4 two-level scale: FP32 tensor scale times
    sb = (amax / 6.0 / g).astype(ml_dtypes.float8_e4m3fn).astype(np.float32) * g   # an FP8 E4M3 block scale
    sb = np.where(sb * 6.0 < amax, sb * 1.125, sb)            # keep values inside the E2M1 range
    sb = np.where(sb > 0, sb, 1.0)                            # all-zero blocks
    q = np.clip(blk / sb, -6.0, 6.0).astype(ml_dtypes.float4_e2m1fn).astype(np.float32) * sb
    return q.reshape(v, d)


def lse(x: np.ndarray) -> float:
    m = float(x.max())
    return m + float(np.log(np.exp(x - m).sum()))


def stats_one(z: np.ndarray, zh: np.ndarray, dlt: np.ndarray) -> dict:
    u = zh + dlt
    order = np.argsort(-u)
    cm = np.maximum.accumulate(z[order[:K_CAP + 1]])
    ok = np.nonzero(cm[:-1] >= u[order[1:K_CAP + 1]])[0]
    k_greedy = int(ok[0]) + 1 if len(ok) else K_CAP + 1
    cset = order[:K_TAIL]
    tail = np.ones(len(z), dtype=bool)
    tail[cset] = False
    out = {"k_greedy": k_greedy, "argmax_agree": bool(np.argmax(zh) == np.argmax(z)),
           "max_err": float(np.abs(z - zh).max()), "bound_over_err": float(dlt.max() / max(np.abs(z - zh).max(), 1e-12))}
    for t in (1.0, 0.7):
        num = lse(u[tail] / t)
        den = np.logaddexp(lse(z[cset] / t), num)
        out[f"tail_T{t}"] = float(np.exp(num - den))
    p = np.exp(z - lse(z))
    q = np.exp(zh - lse(zh))
    out["one_minus_tv_T1"] = float(1.0 - 0.5 * np.abs(p - q).sum())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, local_files_only=True, torch_dtype=torch.float32).eval()
    cap = {}
    model.lm_head.register_forward_hook(lambda m, i, o: cap.__setitem__("h", i[0][0, -1].detach().clone()))
    hs, sane = [], 0
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            out = model(ids, use_cache=True)
            for _ in range(NEW_TOKENS):
                h = cap["h"]
                nxt = out.logits[0, -1].argmax()
                hs.append(h.numpy())
                sane += int((model.lm_head.weight @ h).argmax() == nxt)
                out = model(nxt.view(1, 1), past_key_values=out.past_key_values, use_cache=True)
    H = np.stack(hs).astype(np.float32)
    E = model.lm_head.weight.detach().numpy().astype(np.float32)
    v, d = E.shape
    res = {"model": a.model, "states": len(H), "vocab": v, "hidden": d, "hook_matches_argmax": sane / len(H)}
    screens = {k: quantize(E, k) for k in ("fp8", "fp4")}
    rowerr = {k: np.linalg.norm(E - q, axis=1) for k, q in screens.items()}
    per = {k: [] for k in screens}
    for c0 in range(0, len(H), 64):
        hb = H[c0:c0 + 64]
        zb = hb @ E.T
        for k, q in screens.items():
            zhb = hb @ q.T
            for j in range(len(hb)):
                dlt = np.linalg.norm(hb[j]) * rowerr[k] * (1 + 1e-3) + 1e-4
                per[k].append(stats_one(zb[j], zhb[j], dlt))
    bytes_bf16 = v * d * 2
    screen_bytes = {"fp8": v * d + 4 * v, "fp4": v * d // 2 + v * d // 16}
    summ = {}
    for k, rows in per.items():
        kg = np.array([r["k_greedy"] for r in rows])
        tl = np.array([r["tail_T1.0"] for r in rows])
        summ[k] = {
            "k_greedy_median": float(np.median(kg)), "k_greedy_p95": float(np.percentile(kg, 95)),
            "k_greedy_over_cap_frac": float(np.mean(kg > K_CAP)),
            "tail_T1_median": float(np.median(tl)), "tail_T0.7_median": float(np.median([r["tail_T0.7"] for r in rows])),
            "argmax_agree": float(np.mean([r["argmax_agree"] for r in rows])),
            "one_minus_tv_T1_mean": float(np.mean([r["one_minus_tv_T1"] for r in rows])),
            "bound_over_actual_err_median": float(np.median([r["bound_over_err"] for r in rows])),
            "byte_ratio_design1_median": float(np.median((screen_bytes[k] + np.minimum(kg, K_CAP) * d * 2) / bytes_bf16)),
            "byte_ratio_screen_only": screen_bytes[k] / bytes_bf16,
        }
    d1 = any(s["k_greedy_median"] <= 256 and s["k_greedy_p95"] <= 2048 and s["tail_T1_median"] <= 0.02 for s in summ.values())
    d2 = summ["fp4"]["argmax_agree"] >= 0.97 and summ["fp4"]["one_minus_tv_T1_mean"] >= 0.95
    res["summary"] = summ
    res["decision"] = {"design1_certified": "GO" if d1 else "FAIL", "design2_speculative": "GO" if d2 else "FAIL",
                       "overall": "GO" if (d1 or d2) else "KILL"}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("states", "hook_matches_argmax", "summary", "decision")}, indent=1))


if __name__ == "__main__":
    main()
