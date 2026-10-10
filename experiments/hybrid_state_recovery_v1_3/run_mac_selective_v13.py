#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import gc
import hashlib
import inspect
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
import types
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
POC_VERSION = "1.3-sweep-randomized"
OUT = Path("results")


@dataclass
class ModeCfg:
    lengths: list[int]
    boundary_fracs: list[float]
    repetitions: int
    warmups: int


MODES = {
    "quick": ModeCfg([512], [0.75], 2, 1),
    "standard": ModeCfg([512, 1024], [0.5, 0.75, 0.875], 3, 1),
    "deep": ModeCfg([512, 1024, 2048], [0.5, 0.75, 0.875], 5, 2),
    # v1.3: the decisive context-scaling sweep requested by the handoff work queue.
    # 10 reps with randomized method order to suppress thermal/order bias.
    "sweep": ModeCfg([512, 1024, 2048, 4096], [0.5, 0.75, 0.875], 10, 2),
    "sweep_long": ModeCfg([2048, 4096, 8192], [0.5, 0.75, 0.875], 6, 2),
}


def sync() -> None:
    if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def median(xs: list[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def p95(xs: list[float]) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), 95)) if xs else float("nan")


def mps_mem() -> dict[str, int | None]:
    out = {"current_allocated": None, "driver_allocated": None}
    try:
        out["current_allocated"] = int(torch.mps.current_allocated_memory())
    except Exception:
        pass
    try:
        out["driver_allocated"] = int(torch.mps.driver_allocated_memory())
    except Exception:
        pass
    return out


def get_text_config(config):
    return getattr(config, "text_config", config)


def get_layer_types(config) -> list[str]:
    tc = get_text_config(config)
    layer_types = getattr(tc, "layer_types", None)
    if layer_types is None:
        raise RuntimeError("Qwen3.5 text config does not expose layer_types; refusing to guess the hybrid layout")
    return list(layer_types)


def make_ids(vocab_size: int, length: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    upper = max(128, min(vocab_size, 50000))
    return torch.randint(10, upper, (1, length), generator=gen, dtype=torch.long).to(device)


def logits_diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.float().cpu()
    b = b.float().cpu()
    d = (a - b).abs()
    return {
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
    }


def tensor_diff(a: torch.Tensor | None, b: torch.Tensor | None) -> dict[str, Any]:
    if a is None and b is None:
        return {"comparable": True, "max_abs": 0.0, "mean_abs": 0.0, "shape": None}
    if a is None or b is None:
        return {"comparable": False, "reason": "one_missing"}
    if tuple(a.shape) != tuple(b.shape):
        return {"comparable": False, "reason": "shape_mismatch", "shape_a": list(a.shape), "shape_b": list(b.shape)}
    af = a.detach().float().cpu()
    bf = b.detach().float().cpu()
    d = (af - bf).abs()
    return {
        "comparable": True,
        "max_abs": float(d.max().item()),
        "mean_abs": float(d.mean().item()),
        "shape": list(a.shape),
    }


def state_comparison(candidate, reference, layer_types: list[str]) -> dict[str, Any]:
    rows = []
    global_max = 0.0
    comparable = True
    for i, layer_type in enumerate(layer_types):
        if layer_type != "linear_attention":
            continue
        ca = candidate.layers[i]
        rb = reference.layers[i]
        for attr in ("conv_states", "recurrent_states"):
            da = getattr(ca, attr, {})
            db = getattr(rb, attr, {})
            for state_idx in sorted(set(da) | set(db)):
                diff = tensor_diff(da.get(state_idx), db.get(state_idx))
                rows.append({"layer_idx": i, "kind": attr, "state_idx": state_idx, **diff})
                if not diff.get("comparable", False):
                    comparable = False
                elif diff.get("max_abs") is not None:
                    global_max = max(global_max, float(diff["max_abs"]))
    return {"comparable": comparable, "global_max_abs": global_max, "rows": rows}


def attention_fingerprint(cache, layer_types: list[str]) -> dict[str, Any]:
    rows = []
    for i, layer_type in enumerate(layer_types):
        if layer_type != "full_attention":
            continue
        layer = cache.layers[i]
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if t is None:
                rows.append({"layer_idx": i, "kind": name, "missing": True})
                continue
            tf = t.detach().float()
            rows.append({
                "layer_idx": i,
                "kind": name,
                "shape": list(t.shape),
                "sum": float(tf.sum().item()),
                "abs_sum": float(tf.abs().sum().item()),
                "sq_sum": float((tf * tf).sum().item()),
            })
    return {"rows": rows}


def fingerprint_equal(a: dict[str, Any], b: dict[str, Any], atol: float = 1e-3) -> bool:
    ra = a.get("rows", [])
    rb = b.get("rows", [])
    if len(ra) != len(rb):
        return False
    for x, y in zip(ra, rb):
        if x.get("layer_idx") != y.get("layer_idx") or x.get("kind") != y.get("kind") or x.get("shape") != y.get("shape"):
            return False
        for key in ("sum", "abs_sum", "sq_sum"):
            if key in x or key in y:
                if not math.isclose(float(x.get(key, 0.0)), float(y.get(key, 0.0)), rel_tol=1e-6, abs_tol=atol):
                    return False
    return True


def build_cache(model, ids: torch.Tensor, capture_full_layer_outputs: bool = False, layer_types: list[str] | None = None):
    captures: dict[int, torch.Tensor] = {}
    handles = []
    if capture_full_layer_outputs:
        if layer_types is None:
            raise ValueError("layer_types required when capturing full-layer outputs")
        for idx, layer_type in enumerate(layer_types):
            if layer_type != "full_attention":
                continue
            layer = model.model.layers[idx]
            def hook(_module, _inputs, output, idx=idx):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Expected tensor decoder-layer output at layer {idx}, got {type(output)}")
                captures[idx] = output.detach().clone()
            handles.append(layer.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, use_cache=True, logits_to_keep=1)
        sync()
        return out.past_key_values, out.logits[:, -1, :].detach(), captures
    finally:
        for h in handles:
            h.remove()


def state_size_accounting(cache, layer_types: list[str]) -> dict[str, Any]:
    """Measure real per-class state bytes in the live cache.

    The three recovery strategies pay different memory prices:
      * full-attention K/V   : O(context)  and prefix-decomposable
      * post-block activation: O(context)  and prefix-decomposable
      * recurrent/conv state : O(1) in context but NOT prefix-decomposable
    This function measures the first and third directly from the cache object so
    the Pareto comparison uses observed bytes rather than a formula.
    """
    kv_bytes = 0
    recurrent_bytes = 0
    conv_bytes = 0
    for i, layer_type in enumerate(layer_types):
        layer = cache.layers[i]
        if layer_type == "full_attention":
            for name in ("keys", "values"):
                t = getattr(layer, name, None)
                if t is not None:
                    kv_bytes += int(t.numel() * t.element_size())
        else:
            for attr, sink in (("recurrent_states", "rec"), ("conv_states", "conv")):
                d = getattr(layer, attr, {}) or {}
                for t in d.values():
                    if t is None:
                        continue
                    b = int(t.numel() * t.element_size())
                    if sink == "rec":
                        recurrent_bytes += b
                    else:
                        conv_bytes += b
    return {
        "full_attention_kv_bytes": kv_bytes,
        "recurrent_state_bytes": recurrent_bytes,
        "conv_state_bytes": conv_bytes,
        "recurrent_snapshot_total_bytes": recurrent_bytes + conv_bytes,
    }


def clean_mps() -> None:
    gc.collect()
    try:
        torch.mps.empty_cache()
    except Exception:
        pass


class FrozenKVRecoveryCache:
    """Proxy over a Transformers Cache.

    Linear-attention layers come from a boundary cache at K. Full-attention layers
    come from a complete cache at N. During suffix replay K..N, full-attention
    updates are frozen and simply return their already-valid K/V tensors.

    We delegate every other operation to the underlying Cache object, but override
    query/mask geometry so queries are positioned at K while K/V span 0..N.
    """

    def __init__(self, base_cache, layer_types: list[str], recovery_start: int, full_length: int):
        self._base = base_cache
        self.layers = base_cache.layers
        self.layer_types = list(layer_types)
        self.recovery_start = int(recovery_start)
        self.full_length = int(full_length)
        self.frozen_update_calls = 0

    def __getattr__(self, name):
        return getattr(self._base, name)

    def update(self, key_states, value_states, layer_idx: int, *args, **kwargs):
        if self.layer_types[layer_idx] == "full_attention":
            layer = self.layers[layer_idx]
            if getattr(layer, "keys", None) is None or getattr(layer, "values", None) is None:
                raise RuntimeError(f"full-attention layer {layer_idx} has no frozen KV")
            self.frozen_update_calls += 1
            return layer.keys, layer.values
        return self._base.update(key_states, value_states, layer_idx, *args, **kwargs)

    def get_seq_length(self, layer_idx: int = 0):
        # The logical execution frontier is K even though frozen full-attention
        # layers already contain valid K/V through N.  Transformers uses this
        # cache-level length for position/mask preparation in several paths.
        return self.recovery_start

    def get_query_offset(self, layer_idx: int = 0):
        if self.layer_types[layer_idx] == "full_attention":
            return self.recovery_start
        return self._base.get_query_offset(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int):
        if self.layer_types[layer_idx] == "full_attention":
            return self.full_length, 0
        return self._base.get_mask_sizes(query_length, layer_idx)


def _frozen_full_layer_view(layer, full_length: int):
    """Shallow cache-layer view with mask geometry fixed to the frozen K/V.

    `create_causal_mask` asks the selected *cache layer* for mask sizes.  A normal
    DynamicLayer holding N tokens reports N + query_length because it assumes the
    current query will be appended.  In this POC the K/V already contain the
    suffix through N and `FrozenKVRecoveryCache.update()` intentionally does not
    append it.  Therefore the correct key length is exactly N.

    We shallow-copy the layer so the original full reference cache is never
    monkey-patched; the underlying K/V tensors are shared read-only.
    """
    view = copy.copy(layer)

    def frozen_get_mask_sizes(self, query_length: int):
        return int(full_length), 0

    view.get_mask_sizes = types.MethodType(frozen_get_mask_sizes, view)
    return view


def make_recovery_cache(boundary_cache, full_cache, layer_types: list[str], k: int, n: int):
    from transformers.cache_utils import Cache

    merged_layers = []
    for i, layer_type in enumerate(layer_types):
        if layer_type == "full_attention":
            merged_layers.append(_frozen_full_layer_view(full_cache.layers[i], n))
        elif layer_type == "linear_attention":
            merged_layers.append(boundary_cache.layers[i])
        else:
            raise RuntimeError(f"Unsupported layer type in this POC: {layer_type!r} at {i}")
    base = Cache(layers=merged_layers)
    return FrozenKVRecoveryCache(base, layer_types, k, n)


@contextlib.contextmanager
def bypass_kv_projections(model, layer_types: list[str]):
    """Approximate implementation lower bound: avoid K/V matmuls on full-attention layers.

    The Qwen attention forward still calls k_proj/v_proj before cache.update. Because
    FrozenKVRecoveryCache discards those newly-computed K/V and returns frozen K/V,
    these matmuls are dead work. Replace them with zero-allocation-shaped outputs.
    K-norm/RoPE still run, so this is conservative relative to a custom kernel/forward.
    """
    saved = []
    try:
        for i, layer_type in enumerate(layer_types):
            if layer_type != "full_attention":
                continue
            attn = model.model.layers[i].self_attn
            for proj_name in ("k_proj", "v_proj"):
                proj = getattr(attn, proj_name)
                orig = proj.forward
                saved.append((proj, orig))
                def zero_forward(self, x):
                    return torch.zeros((*x.shape[:-1], self.out_features), dtype=x.dtype, device=x.device)
                proj.forward = types.MethodType(zero_forward, proj)
        yield
    finally:
        for proj, orig in reversed(saved):
            proj.forward = orig


@contextlib.contextmanager
def oracle_skip_full_blocks(model, layer_types: list[str], captured_outputs: dict[int, torch.Tensor], k: int, n: int):
    """Oracle lower bound: skip whole full-attention decoder blocks using cached activations.

    This is intentionally *not* a deployable mechanism. It asks how fast recovery could
    be if exact post-block hidden activations were already cached. Its memory cost is
    recorded separately. It distinguishes compute opportunity from state availability.
    """
    saved = []
    try:
        for i, layer_type in enumerate(layer_types):
            if layer_type != "full_attention":
                continue
            layer = model.model.layers[i]
            orig = layer.forward
            saved.append((layer, orig))
            if i not in captured_outputs:
                raise RuntimeError(f"Missing oracle captured output for full-attention layer {i}")
            cached = captured_outputs[i][:, k:n, :]
            def oracle_forward(self, hidden_states, *args, _cached=cached, **kwargs):
                if hidden_states.shape != _cached.shape:
                    raise RuntimeError(f"oracle shape mismatch: got {tuple(hidden_states.shape)}, expected {tuple(_cached.shape)}")
                return _cached
            layer.forward = types.MethodType(oracle_forward, layer)
        yield
    finally:
        for layer, orig in reversed(saved):
            layer.forward = orig


@contextlib.contextmanager
def naive_skip_full_blocks(model, layer_types: list[str]):
    """Incorrect control: replace each full-attention decoder block with identity."""
    saved = []
    try:
        for i, layer_type in enumerate(layer_types):
            if layer_type != "full_attention":
                continue
            layer = model.model.layers[i]
            orig = layer.forward
            saved.append((layer, orig))
            def identity_forward(self, hidden_states, *args, **kwargs):
                return hidden_states
            layer.forward = types.MethodType(identity_forward, layer)
        yield
    finally:
        for layer, orig in reversed(saved):
            layer.forward = orig


def build_recovery_attention_masks(suffix: torch.Tensor, k: int, n: int) -> dict[str, torch.Tensor]:
    """Build explicit Qwen3.5 masks for recovery replay.

    Qwen3.5 normally derives the full-attention KV length from the cache *before*
    entering decoder layers. A frozen cache already contains K/V through N, so
    ordinary dynamic-cache geometry incorrectly predicts N + query_length keys.

    During recovery the suffix queries have absolute positions K..N-1 and the
    frozen full-attention K/V span positions 0..N-1. We therefore pass an
    explicit 4-D eager-attention bias of shape [B, 1, Q, N]. Linear-attention
    layers only need a 2-D padding mask over the suffix itself.
    """
    if suffix.ndim != 2:
        raise ValueError(f"expected suffix input_ids [batch, seq], got {tuple(suffix.shape)}")
    bsz, q_len = suffix.shape
    if q_len != n - k:
        raise ValueError(f"suffix length {q_len} does not match recovery interval [{k}, {n})")

    device = suffix.device
    # Eager attention expects additive float mask: 0 for visible, -inf-like for masked.
    q_abs = torch.arange(k, n, device=device, dtype=torch.long).view(q_len, 1)
    kv_abs = torch.arange(0, n, device=device, dtype=torch.long).view(1, n)
    visible = kv_abs <= q_abs
    full = torch.zeros((q_len, n), device=device, dtype=torch.float32)
    full.masked_fill_(~visible, torch.finfo(torch.float32).min)
    full = full.view(1, 1, q_len, n).expand(bsz, 1, q_len, n)

    # Qwen3.5 GatedDeltaNet uses this as a 2-D padding mask on suffix hidden states.
    linear = torch.ones((bsz, q_len), device=device, dtype=torch.long)
    return {"full_attention": full, "linear_attention": linear}


def timed_recovery(model, suffix: torch.Tensor, cache, k: int, n: int, patch_ctx=None):
    position_ids = torch.arange(k, n, dtype=torch.long, device=suffix.device).unsqueeze(0)
    attention_mask = build_recovery_attention_masks(suffix, k, n)
    ctx = patch_ctx if patch_ctx is not None else contextlib.nullcontext()
    sync()
    mem_before = mps_mem()
    t0 = time.perf_counter()
    with torch.inference_mode(), ctx:
        out = model(
            input_ids=suffix,
            past_key_values=cache,
            use_cache=True,
            position_ids=position_ids,
            attention_mask=attention_mask,
            logits_to_keep=1,
        )
    sync()
    dt = time.perf_counter() - t0
    mem_after = mps_mem()
    return dt, out.past_key_values, out.logits[:, -1, :].detach(), mem_before, mem_after


def load_model(dtype_name: str):
    from transformers import AutoModelForCausalLM

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_name]
    kwargs = dict(dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager")
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    except TypeError:
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval().to("mps")
    sync()
    return model, dtype


def static_compute_analysis(model, layer_types: list[str]) -> dict[str, Any]:
    total_linear_weights = 0
    kv_proj_weights = 0
    q_proj_weights = 0
    full_block_weights = 0
    full_indices = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            total_linear_weights += int(module.weight.numel())
    for i, layer_type in enumerate(layer_types):
        if layer_type != "full_attention":
            continue
        full_indices.append(i)
        layer = model.model.layers[i]
        attn = layer.self_attn
        kv_proj_weights += int(attn.k_proj.weight.numel() + attn.v_proj.weight.numel())
        q_proj_weights += int(attn.q_proj.weight.numel())
        full_block_weights += sum(int(p.numel()) for p in layer.parameters())
    return {
        "total_linear_weight_elements": total_linear_weights,
        "full_attention_kv_projection_weight_elements": kv_proj_weights,
        "full_attention_q_projection_weight_elements": q_proj_weights,
        "full_attention_decoder_block_parameter_elements": full_block_weights,
        "full_attention_layer_indices": full_indices,
        "kv_projection_fraction_of_all_linear_weights": kv_proj_weights / total_linear_weights if total_linear_weights else None,
        "note": "Weight-element fractions are only structural MAC proxies. Attention matmul, norms, activations and recurrent kernels are not included, so the true fraction of total work saved by K/V projection bypass is no larger in general.",
    }


def source_fingerprints(model) -> dict[str, Any]:
    entries = {}
    objects = {
        "text_model_forward": type(model.model).forward,
        "decoder_layer_forward": type(model.model.layers[0]).forward,
    }
    first_full = next(i for i, t in enumerate(get_layer_types(model.config)) if t == "full_attention")
    objects["full_attention_forward"] = type(model.model.layers[first_full].self_attn).forward
    for name, obj in objects.items():
        try:
            src = inspect.getsource(obj)
            entries[name] = {"sha256": hashlib.sha256(src.encode()).hexdigest(), "source_head": src[:1600]}
        except Exception as e:
            entries[name] = {"error": repr(e)}
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=MODES, default="standard")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    failures: list[dict[str, Any]] = []

    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"

    env = {
        "platform": platform.platform(),
        "python": sys.version,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "physical_memory_bytes": psutil.virtual_memory().total,
        "torch": torch.__version__,
        "transformers": transformers_version,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "mode": args.mode,
        "dtype": args.dtype,
        "model_id": MODEL_ID,
        "env_PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    (OUT / "environment.json").write_text(json.dumps(env, indent=2))

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS unavailable. This POC intentionally refuses CPU fallback.")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        print("WARNING: PYTORCH_ENABLE_MPS_FALLBACK=1 is set. Timing may be contaminated by CPU fallback.")

    print(f"POC version: {POC_VERSION}")
    print(f"Loading {MODEL_ID} with eager attention ...")
    model, dtype = load_model(args.dtype)
    layer_types = get_layer_types(model.config)
    counts = Counter(layer_types)
    if set(counts) - {"linear_attention", "full_attention"}:
        raise SystemExit(f"Unexpected layer types for this POC: {dict(counts)}")
    if not counts.get("linear_attention") or not counts.get("full_attention"):
        raise SystemExit(f"Expected hybrid model, got {dict(counts)}")

    tc = get_text_config(model.config)
    model_info = {
        "model_class": type(model).__name__,
        "text_model_class": type(model.model).__name__,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "num_hidden_layers": int(getattr(tc, "num_hidden_layers", len(layer_types))),
        "hidden_size": int(getattr(tc, "hidden_size", -1)),
        "vocab_size": int(getattr(tc, "vocab_size", -1)),
        "layer_type_counts": dict(counts),
        "layer_types": layer_types,
        "attention_implementation": getattr(tc, "_attn_implementation", None),
    }
    (OUT / "model_info.json").write_text(json.dumps(model_info, indent=2))
    (OUT / "static_compute_analysis.json").write_text(json.dumps(static_compute_analysis(model, layer_types), indent=2))
    (OUT / "source_fingerprints.json").write_text(json.dumps(source_fingerprints(model), indent=2))

    cfg = MODES[args.mode]
    vocab = model_info["vocab_size"]
    raw_rows = []
    scenario_rows = []
    state_rows = []
    correctness_rows = []
    oracle_memory_rows = []

    # Warmup.
    warm_ids = make_ids(vocab, 128, 9191, torch.device("mps"))
    for _ in range(cfg.warmups):
        try:
            cache, _, _ = build_cache(model, warm_ids)
            del cache
            sync()
        except Exception as e:
            failures.append({"stage": "warmup", "error": repr(e), "trace": traceback.format_exc()})
            raise
    clean_mps()

    for n in cfg.lengths:
        print(f"\n=== context {n} ===")
        ids = make_ids(vocab, n, 22000 + n, torch.device("mps"))

        # Full reference once per context. Also capture exact outputs of full-attention decoder blocks.
        sync()
        _t_pref = time.perf_counter()
        full_cache, full_logits, full_block_outputs = build_cache(
            model, ids, capture_full_layer_outputs=True, layer_types=layer_types
        )
        full_prefill_s = time.perf_counter() - _t_pref
        full_fp_before = attention_fingerprint(full_cache, layer_types)
        full_activation_bytes = sum(int(t.numel() * t.element_size()) for t in full_block_outputs.values())

        # v1.3: observed byte cost of each state class at this context length. This is
        # what the simple-baseline attack needs: a recurrent snapshot is O(1) in context
        # and removes replay entirely, while activation caching is O(context) and only
        # accelerates replay.
        sizes = state_size_accounting(full_cache, layer_types)
        sizes.update({
            "context_len": n,
            "full_context_activation_bytes": full_activation_bytes,
            "full_prefill_s": full_prefill_s,
            "n_full_attention_layers": counts["full_attention"],
            "n_linear_attention_layers": counts["linear_attention"],
        })
        state_rows.append(sizes)
        print("  state bytes: kv={kv:.2f} MiB  activations={act:.2f} MiB  recurrent_snapshot={rec:.2f} MiB  full_prefill={p:.3f}s".format(
            kv=sizes["full_attention_kv_bytes"] / 2**20,
            act=full_activation_bytes / 2**20,
            rec=sizes["recurrent_snapshot_total_bytes"] / 2**20,
            p=full_prefill_s,
        ))

        for frac in cfg.boundary_fracs:
            k = max(1, min(n - 1, int(round(n * frac))))
            suffix = ids[:, k:n]
            prefix = ids[:, :k]
            suffix_len = n - k
            print(f"boundary {k}/{n} ({k/n:.3f}), suffix {suffix_len}")

            # Oracle activation cache cost for only the replayed suffix and for full prefix coverage.
            suffix_activation_bytes = sum(
                int(t[:, k:n, :].numel() * t.element_size()) for t in full_block_outputs.values()
            )
            oracle_memory_rows.append({
                "context_len": n,
                "boundary": k,
                "suffix_len": suffix_len,
                "full_attention_layers": counts["full_attention"],
                "full_context_activation_bytes": full_activation_bytes,
                "suffix_activation_bytes": suffix_activation_bytes,
            })

            method_times: dict[str, list[float]] = defaultdict(list)
            representative = {}

            # v1.3: execute the four timed methods in a randomized order within each
            # repetition. v1.1/v1.2 always ran normal -> frozen -> bypass -> oracle,
            # so any thermal drift or allocator warmup was systematically charged to
            # the first method (the baseline), biasing every speedup upward.
            order_rng = np.random.default_rng(int(1_000_003 * frac) ^ n)

            def run_normal(prefix=prefix, suffix=suffix, k=k, n=n):
                boundary_cache, _, _ = build_cache(model, prefix)
                dt, cache_after, logits, mb, ma = timed_recovery(model, suffix, boundary_cache, k, n)
                return dt, {"cache": cache_after, "logits": logits}, mb, ma

            def run_frozen(prefix=prefix, suffix=suffix, k=k, n=n):
                boundary_cache, _, _ = build_cache(model, prefix)
                rec = make_recovery_cache(boundary_cache, full_cache, layer_types, k, n)
                dt, cache_after, logits, mb, ma = timed_recovery(model, suffix, rec, k, n)
                return dt, {"cache": cache_after, "logits": logits, "frozen_update_calls": rec.frozen_update_calls}, mb, ma

            def run_bypass(prefix=prefix, suffix=suffix, k=k, n=n):
                boundary_cache, _, _ = build_cache(model, prefix)
                rec = make_recovery_cache(boundary_cache, full_cache, layer_types, k, n)
                with bypass_kv_projections(model, layer_types):
                    dt, cache_after, logits, mb, ma = timed_recovery(model, suffix, rec, k, n)
                return dt, {"cache": cache_after, "logits": logits, "frozen_update_calls": rec.frozen_update_calls}, mb, ma

            def run_oracle(prefix=prefix, suffix=suffix, k=k, n=n):
                boundary_cache, _, _ = build_cache(model, prefix)
                rec = make_recovery_cache(boundary_cache, full_cache, layer_types, k, n)
                with oracle_skip_full_blocks(model, layer_types, full_block_outputs, k, n):
                    dt, cache_after, logits, mb, ma = timed_recovery(model, suffix, rec, k, n)
                return dt, {"cache": cache_after, "logits": logits}, mb, ma

            runners = [
                ("normal_suffix_replay", run_normal),
                ("frozen_full_kv_replay", run_frozen),
                ("frozen_full_kv_bypass_kv_proj", run_bypass),
                ("oracle_cached_full_block_outputs", run_oracle),
            ]

            for rep in range(cfg.repetitions):
                perm = order_rng.permutation(len(runners))
                for slot, idx in enumerate(perm):
                    method, fn = runners[int(idx)]
                    dt, repdata, mb, ma = fn()
                    method_times[method].append(dt)
                    raw_rows.append({
                        "context_len": n, "boundary": k, "suffix_len": suffix_len,
                        "method": method, "rep": rep, "seconds": dt,
                        "exec_slot": slot,
                        "mps_current_before": mb["current_allocated"],
                        "mps_current_after": ma["current_allocated"],
                    })
                    if method not in representative:
                        representative[method] = repdata

            # Deliberately incorrect control once per scenario, not timed as a performance claim.
            try:
                boundary_bad, _, _ = build_cache(model, prefix)
                bad_cache = make_recovery_cache(boundary_bad, full_cache, layer_types, k, n)
                with naive_skip_full_blocks(model, layer_types):
                    bad_dt, bad_cache_after, bad_logits, _, _ = timed_recovery(model, suffix, bad_cache, k, n)
                representative["naive_identity_skip_full_blocks"] = {"cache": bad_cache_after, "logits": bad_logits}
                raw_rows.append({"context_len": n, "boundary": k, "suffix_len": suffix_len, "method": "naive_identity_skip_full_blocks", "rep": 0, "seconds": bad_dt,
                                 "mps_current_before": None, "mps_current_after": None})
            except Exception as e:
                failures.append({"stage": "naive_skip_control", "context_len": n, "boundary": k, "error": repr(e), "trace": traceback.format_exc()})

            base_med = median(method_times["normal_suffix_replay"])
            scenario = {"context_len": n, "boundary": k, "boundary_frac": k / n, "suffix_len": suffix_len, "baseline_median_s": base_med}
            for method, times in method_times.items():
                med = median(times)
                scenario[f"{method}_median_s"] = med
                scenario[f"{method}_p95_s"] = p95(times)
                if method != "normal_suffix_replay":
                    scenario[f"{method}_speedup_pct_vs_baseline"] = 100.0 * (1.0 - med / base_med) if base_med > 0 else None
            scenario_rows.append(scenario)

            # Correctness: compare semantic recovery modes to the normal suffix replay (same chunking), and also to full reference logits.
            base_rep = representative["normal_suffix_replay"]
            for method, repdata in representative.items():
                ld_base = logits_diff(repdata["logits"], base_rep["logits"])
                ld_full = logits_diff(repdata["logits"], full_logits)
                sd_base = state_comparison(repdata["cache"], base_rep["cache"], layer_types)
                sd_full = state_comparison(repdata["cache"], full_cache, layer_types)
                correctness_rows.append({
                    "context_len": n,
                    "boundary": k,
                    "method": method,
                    "logits_vs_normal_suffix": ld_base,
                    "logits_vs_full_reference": ld_full,
                    "linear_state_vs_normal_suffix": {"comparable": sd_base["comparable"], "global_max_abs": sd_base["global_max_abs"]},
                    "linear_state_vs_full_reference": {"comparable": sd_full["comparable"], "global_max_abs": sd_full["global_max_abs"]},
                    "frozen_update_calls": repdata.get("frozen_update_calls"),
                })
                # Save detailed state rows separately for the first context/boundary only to keep bundle small.
                if n == cfg.lengths[0] and frac == cfg.boundary_fracs[0]:
                    (OUT / f"state_diff_{method}.json").write_text(json.dumps({"vs_normal": sd_base, "vs_full": sd_full}, indent=2))

            full_fp_after = attention_fingerprint(full_cache, layer_types)
            if not fingerprint_equal(full_fp_before, full_fp_after):
                failures.append({"stage": "frozen_kv_mutation_check", "context_len": n, "boundary": k, "error": "full attention KV fingerprint changed"})

            print(
                "  medians: normal={:.4f}s frozen={:.4f}s bypass={:.4f}s oracle={:.4f}s".format(
                    scenario["normal_suffix_replay_median_s"],
                    scenario["frozen_full_kv_replay_median_s"],
                    scenario["frozen_full_kv_bypass_kv_proj_median_s"],
                    scenario["oracle_cached_full_block_outputs_median_s"],
                )
            )
            print(
                "  speedup vs normal: frozen={:.1f}% bypass={:.1f}% oracle={:.1f}%".format(
                    scenario["frozen_full_kv_replay_speedup_pct_vs_baseline"],
                    scenario["frozen_full_kv_bypass_kv_proj_speedup_pct_vs_baseline"],
                    scenario["oracle_cached_full_block_outputs_speedup_pct_vs_baseline"],
                )
            )

        del full_cache, full_block_outputs
        clean_mps()

    # Write raw outputs.
    if raw_rows:
        with (OUT / "measurements.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            w.writeheader(); w.writerows(raw_rows)
    (OUT / "scenario_summary.json").write_text(json.dumps(scenario_rows, indent=2))
    (OUT / "correctness.json").write_text(json.dumps(correctness_rows, indent=2))
    (OUT / "oracle_activation_memory.json").write_text(json.dumps(oracle_memory_rows, indent=2))
    (OUT / "state_size_accounting.json").write_text(json.dumps(state_rows, indent=2))
    (OUT / "failures.json").write_text(json.dumps(failures, indent=2))

    # Aggregate screen.
    def speeds(key: str) -> list[float]:
        return [float(r[key]) for r in scenario_rows if r.get(key) is not None and math.isfinite(float(r[key]))]

    frozen_s = speeds("frozen_full_kv_replay_speedup_pct_vs_baseline")
    bypass_s = speeds("frozen_full_kv_bypass_kv_proj_speedup_pct_vs_baseline")
    oracle_s = speeds("oracle_cached_full_block_outputs_speedup_pct_vs_baseline")

    semantic_methods = {"frozen_full_kv_replay", "frozen_full_kv_bypass_kv_proj", "oracle_cached_full_block_outputs"}
    semantic_correct = True
    max_semantic_logit_diff = 0.0
    for row in correctness_rows:
        if row["method"] not in semantic_methods:
            continue
        d = float(row["logits_vs_normal_suffix"]["max_abs"])
        max_semantic_logit_diff = max(max_semantic_logit_diff, d)
        if not math.isfinite(d) or d > 0.15:
            semantic_correct = False
    naive_rows = [r for r in correctness_rows if r["method"] == "naive_identity_skip_full_blocks"]
    naive_incorrect = any(float(r["logits_vs_normal_suffix"]["max_abs"]) > 0.15 for r in naive_rows) if naive_rows else None

    kv_unchanged = not any(f.get("stage") == "frozen_kv_mutation_check" for f in failures)
    med_frozen = median(frozen_s)
    med_bypass = median(bypass_s)
    med_oracle = median(oracle_s)

    # Heuristic research gates, not publication standards.
    if not semantic_correct or not kv_unchanged:
        signal = "invalid_or_incorrect_poc_fix_before_interpreting"
    elif med_bypass >= 10.0:
        signal = "promising_selective_kv_reuse_compute_signal"
    elif med_oracle >= 25.0:
        signal = "kv_only_signal_weak_but_activation_oracle_has_headroom_consider_pivot"
    else:
        signal = "weak_incremental_selective_recovery_signal_deprioritize"

    summary = {
        "interpretation_scope": (
            "Real Qwen3.5-0.8B execution on Apple MPS with eager attention. The POC tests whether preserving already-valid "
            "full-attention KV while rebuilding lagging linear/recurrent state provides incremental compute savings over ordinary "
            "suffix replay. It is not a CUDA/vLLM benchmark."
        ),
        "semantic_recovery_correct": semantic_correct,
        "max_semantic_logit_diff_vs_normal_suffix": max_semantic_logit_diff,
        "frozen_full_attention_kv_unchanged": kv_unchanged,
        "naive_skip_full_blocks_is_incorrect": naive_incorrect,
        "median_speedup_pct": {
            "frozen_full_kv_replay": med_frozen,
            "frozen_full_kv_bypass_kv_proj": med_bypass,
            "oracle_cached_full_block_outputs": med_oracle,
        },
        "screen_signal": signal,
        "heuristic_gates": {
            "promising": "correct + frozen KV unchanged + median K/V-projection-bypass speedup >=10% vs normal suffix replay",
            "activation_pivot": "K/V-only <10% but oracle full-block skip >=25%; requires activation/state design rather than KV-only recovery",
            "weak": "both K/V-only and oracle headroom small",
        },
        "critical_caveat": (
            "The oracle path caches exact post-decoder-block hidden activations and is intentionally not a deployable baseline. "
            "It is an upper-bound probe for how much full-block compute could be removed if extra activation state were retained."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    root = Path.cwd().resolve()
    bundle = root / "mac_state_selective_recovery_results_bundle.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.resolve().rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(root))
        for extra in ("run_mac_selective.py", "requirements.txt", "README.md", "SOURCES_AND_SCOPE.md"):
            ep = root / extra
            if ep.exists():
                z.write(ep, ep.name)
    print(f"\nWrote {bundle}")


if __name__ == "__main__":
    main()
