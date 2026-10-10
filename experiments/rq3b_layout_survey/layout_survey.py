#!/usr/bin/env python3
"""RQ3b: does any real hybrid model have a layout where divergent state reuse pays?

Established in experiments/rq1_soundness_gate (and validated there against
exhaustive search over subsets): when recurrent state must be rebuilt by replay,
the maximal set of full-attention blocks that can be skipped SOUNDLY, with no
cached activations, is exactly the trailing run of full-attention layers at the
end of the stack.

Reason: if layer i is skipped then layer i+1 has no input, so i+1 must also be
skipped or i must be the last layer. The skipped set is therefore a contiguous
suffix, and it can only contain full-attention layers.

So the achievable sound saving is:

    (trailing_FA_run / n_FA) * (full-attention share of replay work)

This script computes that for every hybrid model whose layout is public, and
compares it against the counterfactual layout that would maximise it.
"""
from __future__ import annotations

import json
from collections import defaultdict

import torch
from transformers import AutoConfig, AutoModelForCausalLM

RAW = json.load(open("results_layouts_raw.json"))

FA_TOKENS = ("full_attention", "attention")
REC_TOKENS = ("linear_attention", "mamba", "recurrent", "conv")
FUSED_TOKENS = ("hybrid",)


def classify(t: str) -> str:
    if t in FA_TOKENS or "full_attention" in t:
        return "fa"
    if t in FUSED_TOKENS:
        return "fused"
    if any(k in t for k in REC_TOKENS):
        return "rec"
    if "mlp" in t:
        return "mlp"
    return "unknown"


def skippable_suffix(kinds: list[str]) -> list[int]:
    """Layers that can be skipped soundly during recurrent-state backfill.

    Backfill only needs correct recurrent state at the boundary. Recurrent layer
    L's state depends on layers below L, so every layer ABOVE the highest
    state-holding layer is never needed at gap positions and can be skipped with
    no cached activations. Anything at or below it must run, because the closure
    condition requires each executing layer's predecessor to produce its input.

    This generalizes the trailing full-attention run: an MLP block sitting above
    the last recurrent block is skippable too.
    """
    state_idx = [i for i, k in enumerate(kinds) if k in ("rec", "fused")]
    if not state_idx:
        return []
    return list(range(max(state_idx) + 1, len(kinds)))


def is_recurrent_first(kinds: list[str]) -> bool:
    """True iff every FA layer comes after every state-holding layer."""
    fa = [i for i, k in enumerate(kinds) if k == "fa"]
    rec = [i for i, k in enumerate(kinds) if k == "rec"]
    return bool(fa) and bool(rec) and min(fa) > max(rec)


def cost_profile(model_id: str, kinds: list[str]):
    """Per-layer Linear MACs from the real module shapes on the meta device."""
    cfg = AutoConfig.from_pretrained(model_id)
    tc = getattr(cfg, "text_config", cfg)
    n_experts = int(getattr(tc, "num_experts", 0) or
                    getattr(tc, "num_local_experts", 0) or
                    getattr(tc, "n_routed_experts", 0) or 0)
    top_k = int(getattr(tc, "num_experts_per_tok", 0) or getattr(tc, "moe_top_k", 0) or 0)
    # For MoE, only top_k of n_experts run per token, so scale expert Linear MACs
    # rather than dropping the model from the cost model.
    moe_scale = (top_k / n_experts) if (n_experts > 1 and top_k > 0) else 1.0
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg)
    per_layer = defaultdict(float)
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        parts = name.split(".")
        if "layers" not in parts:
            continue
        idx = int(parts[parts.index("layers") + 1])
        mac = float(mod.in_features) * float(mod.out_features)
        if "experts" in parts:
            mac *= moe_scale
        per_layer[idx] += mac
    if len(per_layer) < len(kinds):
        return None, f"only resolved {len(per_layer)} of {len(kinds)} layers"

    hidden = int(tc.hidden_size)
    n_q = int(getattr(tc, "num_attention_heads", 0) or 0)
    hd = int(getattr(tc, "head_dim", 0) or (hidden // n_q if n_q else 0))
    fa_w = sum(per_layer[i] for i, k in enumerate(kinds) if k == "fa")
    other_w = sum(per_layer[i] for i, k in enumerate(kinds) if k != "fa")
    return {"fa_weight_mac": fa_w, "other_weight_mac": other_w,
            "q_width": n_q * hd, "n_fa": sum(1 for k in kinds if k == "fa"),
            "moe_scale": moe_scale, "n_experts": n_experts, "top_k": top_k,
            "per_layer_mac": dict(per_layer)}, None


def fa_share_of_replay(p, suffix, context) -> float:
    k = context - suffix
    causal = suffix * k + suffix * (suffix + 1) / 2.0
    fa = suffix * p["fa_weight_mac"] + p["n_fa"] * 2.0 * causal * p["q_width"]
    other = suffix * p["other_weight_mac"]
    return fa / (fa + other)


def suffix_share_of_replay(p, kinds, suffix_layers, suffix, context) -> float:
    """Share of replay work held by the skippable suffix, whatever its layer kinds."""
    if not suffix_layers:
        return 0.0
    k = context - suffix
    causal = suffix * k + suffix * (suffix + 1) / 2.0
    per = p["per_layer_mac"]
    skip_w = sum(per.get(i, 0.0) for i in suffix_layers)
    n_fa_skipped = sum(1 for i in suffix_layers if kinds[i] == "fa")
    skip_cost = suffix * skip_w + n_fa_skipped * 2.0 * causal * p["q_width"]
    total = (suffix * (p["fa_weight_mac"] + p["other_weight_mac"])
             + p["n_fa"] * 2.0 * causal * p["q_width"])
    return skip_cost / total


CONTEXTS = [(2048, 512), (8192, 2048), (32768, 8192), (131072, 32768)]


def main():
    rows = []
    for rec in RAW:
        if "layer_types" not in rec:
            continue
        mid = rec["model_id"]
        kinds = [classify(t) for t in rec["layer_types"]]
        n_fa = sum(1 for k in kinds if k == "fa")
        n_rec = sum(1 for k in kinds if k == "rec")
        n_fused = sum(1 for k in kinds if k == "fused")
        run = skippable_suffix(kinds)

        entry = {
            "model_id": mid,
            "n_layers": len(kinds),
            "n_fa": n_fa, "n_rec": n_rec, "n_fused": n_fused,
            "n_mlp": sum(1 for k in kinds if k == "mlp"),
            "fa_positions": [i for i, k in enumerate(kinds) if k == "fa"],
            "last_layer_kind": kinds[-1],
            "skippable_suffix": run,
            "skippable_suffix_len": len(run),
            "skippable_suffix_kinds": [kinds[i] for i in run],
            "skippable_fa_layers": sum(1 for i in run if kinds[i] == "fa"),
            "skippable_fraction_of_fa": (sum(1 for i in run if kinds[i] == "fa") / n_fa)
                                        if n_fa else 0.0,
            "recurrent_first_layout": is_recurrent_first(kinds),
        }
        if n_fused and not n_fa:
            entry["note"] = ("fused parallel hybrid: attention and recurrent paths live in "
                             "the same block and their outputs are combined, so the "
                             "attention path cannot be skipped independently at all")
            entry["skippable_fraction_of_fa"] = 0.0

        prof, err = (None, "fused layout") if (n_fused and not n_fa) else cost_profile(mid, kinds)
        if prof:
            entry["fa_share_pct"] = {}
            entry["achievable_sound_saving_pct"] = {}
            entry["counterfactual_recurrent_first_saving_pct"] = {}
            for ctx, suf in CONTEXTS:
                share = 100.0 * fa_share_of_replay(prof, suf, ctx)
                entry["fa_share_pct"][str(ctx)] = round(share, 2)
                entry["achievable_sound_saving_pct"][str(ctx)] = round(
                    100.0 * suffix_share_of_replay(prof, kinds, run, suf, ctx), 3)
                entry["counterfactual_recurrent_first_saving_pct"][str(ctx)] = round(share, 2)
        else:
            entry["cost_model_skipped"] = err
        rows.append(entry)

    json.dump({
        "rule": ("maximal sound skip set = trailing run of full-attention layers; "
                 "validated against exhaustive subset search in rq1_soundness_gate"),
        "caveat": ("Achievable saving is an UPPER bound: it assumes skipping a block "
                   "costs nothing and that all FA layers cost the same. Measured "
                   "oracle numbers on Qwen3.5 came in below the analytical share."),
        "rows": rows,
    }, open("results_layout_survey.json", "w"), indent=2)

    print(f"{'model':<38} {'layers':>7} {'FA':>4} {'skip':>6} {'last layer':<17} "
          f"{'rec-first':>10}")
    print("-" * 78)
    for r in rows:
        print(f"{r['model_id'].split('/')[-1][:37]:<38} {r['n_layers']:>7} {r['n_fa']:>4} "
              f"{r['skippable_suffix_len']:>6} {r['last_layer_kind']:<17} "
              f"{str(r['recurrent_first_layout']):>10}")

    print()
    print("Achievable SOUND saving vs the counterfactual best layout, % of replay work")
    print(f"{'model':<30} {'context':>8} {'FA share':>9} {'achievable':>11} {'if rec-first':>13}")
    print("-" * 78)
    for r in rows:
        if "fa_share_pct" not in r:
            continue
        for ctx, _ in CONTEXTS:
            print(f"{r['model_id'].split('/')[-1][:29]:<30} {ctx:>8} "
                  f"{r['fa_share_pct'][str(ctx)]:>8.1f}% "
                  f"{r['achievable_sound_saving_pct'][str(ctx)]:>10.2f}% "
                  f"{r['counterfactual_recurrent_first_saving_pct'][str(ctx)]:>12.1f}%")
    print()
    for r in rows:
        if "cost_model_skipped" in r:
            print(f"  cost model skipped for {r['model_id']}: {r['cost_model_skipped']}")
    n_rf = sum(1 for r in rows if r["recurrent_first_layout"])
    print()
    print(f"models surveyed: {len(rows)}")
    print(f"models with a recurrent-first layout: {n_rf}")
    print(f"models where >50% of FA work is soundly skippable: "
          f"{sum(1 for r in rows if r['skippable_fraction_of_fa'] > 0.5)}")


if __name__ == "__main__":
    main()
