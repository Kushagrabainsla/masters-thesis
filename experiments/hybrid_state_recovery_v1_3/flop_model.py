#!/usr/bin/env python3
"""Analytical FLOP/MAC model for hybrid-stack suffix replay.

Purpose: establish an architecture-level ceiling on ANY mechanism that reuses
already-valid full-attention (FA) state while rebuilding lagging recurrent state.
Such a mechanism can, at absolute best, remove all FA-block work from the replay.
So its speedup ceiling is exactly the FA-block share of replay FLOPs.

Shapes are read from the real model on the `meta` device (no weights, no GPU),
so the per-layer linear dimensions are the model's actual ones, not guesses.
"""
import json
from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"


def linear_shapes_by_layer():
    cfg = AutoConfig.from_pretrained(MODEL_ID)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    tc = getattr(cfg, "text_config", cfg)
    layer_types = list(tc.layer_types)
    per_layer = defaultdict(int)
    detail = defaultdict(lambda: defaultdict(int))
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue  # lm_head / embeddings handled separately
        idx = int(parts[parts.index("layers") + 1])
        mac = int(mod.in_features) * int(mod.out_features)
        per_layer[idx] += mac
        detail[idx][parts[-1]] = mac
    return cfg, tc, layer_types, dict(per_layer), {k: dict(v) for k, v in detail.items()}


def main():
    cfg, tc, layer_types, per_layer_mac, detail = linear_shapes_by_layer()

    d = int(tc.hidden_size)
    n_q = int(tc.num_attention_heads)
    n_kv = int(tc.num_key_value_heads)
    hd = int(tc.head_dim)
    q_dim = n_q * hd          # total query width, also the AV output width
    kv_dim = n_kv * hd

    lin_k_heads = int(tc.linear_num_key_heads)
    lin_v_heads = int(tc.linear_num_value_heads)
    lin_k_dim = int(tc.linear_key_head_dim)
    lin_v_dim = int(tc.linear_value_head_dim)
    conv_k = int(tc.linear_conv_kernel_dim)

    fa_idx = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    ln_idx = [i for i, t in enumerate(layer_types) if t == "linear_attention"]

    # --- Per-token, per-layer MAC counts that do NOT depend on context length ---
    fa_weight_mac = sum(per_layer_mac[i] for i in fa_idx)
    ln_weight_mac = sum(per_layer_mac[i] for i in ln_idx)

    # Gated-DeltaNet recurrent work per token per layer:
    #   state is [k_heads, k_dim, v_dim]; the delta rule reads and updates it.
    #   Count the two dominant einsums (state read -> output, and state update)
    #   as 2 * heads * k_dim * v_dim MACs per token.
    gdn_state_mac_per_tok = 2 * lin_k_heads * lin_k_dim * lin_v_dim
    # short causal conv over the qkv-ish projected width
    gdn_conv_width = lin_k_heads * lin_k_dim * 2 + lin_v_heads * lin_v_dim
    gdn_conv_mac_per_tok = gdn_conv_width * conv_k

    ln_extra_per_tok = len(ln_idx) * (gdn_state_mac_per_tok + gdn_conv_mac_per_tok)

    rows = []
    for n in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        for frac in [0.5, 0.75, 0.875]:
            k = int(round(n * frac))
            s = n - k
            if s <= 0:
                continue
            # FA attention matmuls for S query positions against N keys:
            #   QK^T : S*N*q_dim MACs ; AV : S*N*q_dim MACs  (per layer)
            # Causal masking within the suffix removes a triangle of size S^2/2.
            causal_pairs = s * k + s * (s + 1) / 2.0
            fa_attn_mac = len(fa_idx) * 2.0 * causal_pairs * q_dim

            fa_total = s * fa_weight_mac + fa_attn_mac
            ln_total = s * ln_weight_mac + s * ln_extra_per_tok
            total = fa_total + ln_total
            rows.append({
                "context_len": n,
                "boundary_frac": frac,
                "boundary": k,
                "suffix_len": s,
                "fa_weight_mac": s * fa_weight_mac,
                "fa_attn_mac": fa_attn_mac,
                "fa_total_mac": fa_total,
                "linear_total_mac": ln_total,
                "replay_total_mac": total,
                # Ceiling for ANY mechanism that reuses FA state perfectly:
                "fa_share_of_replay_pct": 100.0 * fa_total / total,
                # Ceiling for a mechanism that only avoids recomputing K/V
                # projections (frozen KV, still runs attention + O/MLP):
                "kv_proj_share_of_replay_pct": 100.0 * s * sum(
                    detail[i].get("k_proj", 0) + detail[i].get("v_proj", 0) for i in fa_idx
                ) / total,
            })

    out = {
        "model_id": MODEL_ID,
        "dims": {
            "hidden_size": d, "num_attention_heads": n_q, "num_key_value_heads": n_kv,
            "head_dim": hd, "query_width": q_dim, "kv_width": kv_dim,
            "linear_key_heads": lin_k_heads, "linear_key_head_dim": lin_k_dim,
            "linear_value_head_dim": lin_v_dim, "conv_kernel": conv_k,
        },
        "layer_counts": {"full_attention": len(fa_idx), "linear_attention": len(ln_idx)},
        "full_attention_layer_indices": fa_idx,
        "per_token_mac": {
            "fa_weight_mac_all_fa_layers": fa_weight_mac,
            "linear_weight_mac_all_linear_layers": ln_weight_mac,
            "gdn_state_mac_per_token_per_layer": gdn_state_mac_per_tok,
            "gdn_conv_mac_per_token_per_layer": gdn_conv_mac_per_tok,
            "linear_extra_mac_per_token_all_layers": ln_extra_per_tok,
        },
        "fa_weight_share_of_all_weight_mac_pct": 100.0 * fa_weight_mac / (fa_weight_mac + ln_weight_mac),
        "interpretation": (
            "fa_share_of_replay_pct is a hard upper bound on the speedup of any mechanism that "
            "reuses already-valid full-attention state while rebuilding lagging recurrent state, "
            "because such a mechanism cannot remove any linear/recurrent-layer work: the recurrent "
            "state is precisely what is being rebuilt. kv_proj_share_of_replay_pct is the much "
            "tighter bound for the sound frozen-KV variant that still runs attention and the "
            "rest of the FA block."
        ),
        "rows": rows,
    }
    with open("results_flop_model.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"FA layers: {len(fa_idx)} of {len(layer_types)}  indices={fa_idx}")
    print(f"FA weight MAC share of all layer weight MACs: {out['fa_weight_share_of_all_weight_mac_pct']:.2f}%")
    print(f"GDN recurrent state MAC/token/layer: {gdn_state_mac_per_tok:,}")
    print()
    print("Ceiling on any FA-state-reuse mechanism (fa_share_of_replay_pct):")
    print(f"{'ctx':>8} {'frac':>6} {'suffix':>7} {'FA share %':>11} {'KVproj share %':>15}")
    for r in rows:
        print(f"{r['context_len']:>8} {r['boundary_frac']:>6} {r['suffix_len']:>7} "
              f"{r['fa_share_of_replay_pct']:>11.2f} {r['kv_proj_share_of_replay_pct']:>15.3f}")


if __name__ == "__main__":
    main()
