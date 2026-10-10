#!/usr/bin/env python3
"""Fetch layer_types for real hybrid models. Configs only, no weights."""
import json, sys
from collections import Counter
from transformers import AutoConfig

MODELS = [
    "Qwen/Qwen3.5-0.8B-Base",
    "Qwen/Qwen3-Next-80B-A3B-Instruct",
    "nvidia/Nemotron-H-8B-Base-8K",
    "nvidia/Nemotron-H-47B-Base-8K",
    "nvidia/Nemotron-H-56B-Base-8K",
    "ibm-granite/granite-4.0-h-small",
    "ibm-granite/granite-4.0-h-tiny",
    "ibm-granite/granite-4.0-h-micro",
    "ai21labs/AI21-Jamba-Mini-1.7",
    "ai21labs/AI21-Jamba-Large-1.7",
    "Zyphra/Zamba2-2.7B",
    "ibm-ai-platform/Bamba-9B",
    "tiiuae/Falcon-H1-0.5B-Instruct",
    "tiiuae/Falcon-H1-34B-Instruct",
    "microsoft/Phi-4-mini-flash-reasoning",
    "LiquidAI/LFM2-1.2B",
    "state-spaces/mamba2-2.7b",
]

ATTN_KEYS = ("layer_types", "layers_block_type", "hybrid_override_pattern",
             "block_types", "layer_type_list")

def get_layout(model_id):
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
    tc = getattr(cfg, "text_config", cfg)
    for k in ATTN_KEYS:
        for obj in (tc, cfg):
            v = getattr(obj, k, None)
            if v:
                return k, [str(x) for x in v], tc
    return None, None, tc

out = []
for m in MODELS:
    rec = {"model_id": m}
    try:
        key, lt, tc = get_layout(m)
        rec["num_hidden_layers"] = getattr(tc, "num_hidden_layers", None)
        if lt is None:
            rec["error"] = "no layer-type list in config"
        else:
            rec["source_key"] = key
            rec["layer_types"] = lt
            rec["counts"] = dict(Counter(lt))
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    out.append(rec)
    status = rec.get("counts") or rec.get("error")
    print(f"{m:<44} {status}")

json.dump(out, open("results_layouts_raw.json", "w"), indent=2)
print("\nwrote results_layouts_raw.json")
