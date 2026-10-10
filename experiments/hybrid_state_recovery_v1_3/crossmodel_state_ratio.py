#!/usr/bin/env python3
"""Cross-model: recurrent snapshot bytes vs full-attention KV bytes.

The upstream objection to dense recurrent checkpointing (vLLM #45702) is that
finer-grained Mamba checkpoints in HBM inflate per-request footprint and cut
batch size. That objection is context-dependent: a recurrent snapshot is O(1)
in context while FA KV is O(context). This script computes, per model, the
context length at which ONE recurrent snapshot costs the same as the FA KV
cache -- i.e. where the objection stops being true.

Configs are read from the HF Hub (config.json only, no weights).
"""
import json

from transformers import AutoConfig

MODELS = [
    "Qwen/Qwen3.5-0.8B-Base",
    "tiiuae/Falcon-H1-0.5B-Instruct",
    "nvidia/Nemotron-H-8B-Base-8K",
    "ibm-granite/granite-4.0-h-small",
    "state-spaces/mamba2-2.7b",
]


def g(cfg, *names, default=None):
    tc = getattr(cfg, "text_config", cfg)
    for n in names:
        for obj in (tc, cfg):
            v = getattr(obj, n, None)
            if v is not None:
                return v
    return default


def analyze(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    tc = getattr(cfg, "text_config", cfg)
    nl = int(g(cfg, "num_hidden_layers", default=0))
    layer_types = g(cfg, "layer_types", "layers_block_type", default=None)

    if layer_types:
        lt = list(layer_types)
        n_fa = sum(1 for t in lt if "full" in str(t) or str(t) == "attention")
        n_rec = sum(1 for t in lt if "linear" in str(t) or "mamba" in str(t) or "recurrent" in str(t))
    else:
        # Mamba-only or unknown layout.
        n_fa, n_rec = 0, nl
        lt = None

    hidden = int(g(cfg, "hidden_size", default=0))
    n_kv = g(cfg, "num_key_value_heads", "num_attention_heads", default=None)
    head_dim = g(cfg, "head_dim", default=None)
    if head_dim is None and hidden and g(cfg, "num_attention_heads"):
        head_dim = hidden // int(g(cfg, "num_attention_heads"))

    # FA KV bytes per token (all FA layers), fp16: 2 tensors * kv_heads * head_dim * 2B
    kv_per_tok = 0
    if n_fa and n_kv and head_dim:
        kv_per_tok = n_fa * 2 * int(n_kv) * int(head_dim) * 2

    # Recurrent snapshot bytes: gated-deltanet style or mamba2 style.
    # dtype note: measured on Qwen3.5-0.8B, the GDN recurrent state is stored in
    # FLOAT32 (4.0 B/elem) even though the model runs in float16, because SSM state
    # accumulates and needs the headroom. We therefore use 4 B/elem for the state
    # and 2 B/elem for the conv window, matching the measurement. Assuming fp16
    # would understate every snapshot below by 2x.
    REC_ELEM_BYTES = 4
    rec_bytes = 0
    style = "unknown"
    lk_h = g(cfg, "linear_num_key_heads")
    lk_d = g(cfg, "linear_key_head_dim")
    lv_d = g(cfg, "linear_value_head_dim")
    if lk_h and lk_d and lv_d:
        style = "gated_deltanet"
        rec_bytes = n_rec * int(lk_h) * int(lk_d) * int(lv_d) * REC_ELEM_BYTES
    else:
        # Mamba2: state is [n_heads, head_dim, d_state] per layer, where
        # n_heads * head_dim is the MAMBA inner width. Do NOT fall back to
        # intermediate_size: that is the MLP width and overstates the snapshot
        # (it inflated Nemotron-H from 96 MiB to 252 MiB in an earlier pass).
        # Nemotron-H uses mamba_num_heads/mamba_head_dim/ssm_state_size;
        # Granite uses mamba_n_heads/mamba_d_head/mamba_d_state.
        m_heads = g(cfg, "mamba_n_heads", "mamba_num_heads")
        m_hd = g(cfg, "mamba_d_head", "mamba_head_dim")
        d_state = g(cfg, "mamba_d_state", "ssm_state_size", "d_state", "state_size")
        if d_state and m_heads and m_hd:
            style = "mamba2_heads"
            rec_bytes = n_rec * int(m_heads) * int(m_hd) * int(d_state) * REC_ELEM_BYTES

    # Post-block activation cache for FA layers: hidden_size fp16 per token per layer.
    act_per_tok = n_fa * hidden * 2 if (n_fa and hidden) else 0
    crossover = (rec_bytes / kv_per_tok) if kv_per_tok else None
    # Context at which the FA activation cache costs as much as ONE recurrent snapshot.
    act_eq_snapshot = (rec_bytes / act_per_tok) if act_per_tok else None
    return {
        "fa_activation_bytes_per_token": act_per_tok,
        "activation_to_kv_byte_ratio": (act_per_tok / kv_per_tok) if kv_per_tok else None,
        "context_where_activation_cache_equals_one_snapshot": act_eq_snapshot,
        "model_id": model_id,
        "num_hidden_layers": nl,
        "n_full_attention_layers": n_fa,
        "n_recurrent_layers": n_rec,
        "hidden_size": hidden,
        "recurrent_style": style,
        "fa_kv_bytes_per_token": kv_per_tok,
        "recurrent_snapshot_bytes": rec_bytes,
        "recurrent_snapshot_MiB": rec_bytes / 2**20 if rec_bytes else None,
        "crossover_context_tokens": crossover,
        "note": "crossover = context length at which FA KV bytes equal ONE recurrent snapshot",
    }


def main():
    rows = []
    for m in MODELS:
        try:
            rows.append(analyze(m))
        except Exception as e:
            rows.append({"model_id": m, "error": repr(e)})
    with open("results_crossmodel_state_ratio.json", "w") as f:
        json.dump(rows, f, indent=2)

    hdr = (f"{'model':<34} {'FA/rec':>7} {'snap MiB':>9} {'KV B/tok':>9} "
           f"{'act B/tok':>10} {'act/KV':>7} {'KV=snap':>9} {'act=snap':>9}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['model_id']:<38} ERROR {r['error'][:60]}")
            continue
        xo = r["crossover_context_tokens"]
        ae = r["context_where_activation_cache_equals_one_snapshot"]
        ratio = r["activation_to_kv_byte_ratio"]
        print(f"{r['model_id']:<34} "
              f"{str(r['n_full_attention_layers'])+'/'+str(r['n_recurrent_layers']):>7} "
              f"{(r['recurrent_snapshot_MiB'] or 0):>9.2f} "
              f"{r['fa_kv_bytes_per_token']:>9} "
              f"{r['fa_activation_bytes_per_token']:>10} "
              f"{(f'{ratio:.2f}x' if ratio else 'n/a'):>7} "
              f"{(f'{xo:,.0f}' if xo else 'n/a'):>9} "
              f"{(f'{ae:,.0f}' if ae else 'n/a'):>9}")


if __name__ == "__main__":
    main()
