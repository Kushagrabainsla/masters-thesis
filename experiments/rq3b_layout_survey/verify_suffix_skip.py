#!/usr/bin/env python3
"""Verify the layout prediction: skipping layers above the last recurrent layer.

During recurrent-state backfill the only thing that must come out correct is the
recurrent state at the boundary. Layers above the highest recurrent layer never
feed a recurrent state update, so they can be skipped with no cached activations
and no loss of correctness FOR THAT PURPOSE. Logits for the replayed positions
are not produced, which is fine: backfill does not need them.

Qwen3.5-0.8B ends with a full-attention layer (index 23), and the last recurrent
layer is index 22, so exactly one full-attention block is skippable. The layout
survey predicts a saving of 4.74% at context 2048 and 14.72% at 131072.

Correctness criterion here is recurrent-state equality, not logit equality.
"""
from __future__ import annotations

import contextlib, json, statistics, time, types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
OUT = Path("results")


def sync():
    if hasattr(torch, "mps"):
        torch.mps.synchronize()


def layer_types(cfg):
    tc = getattr(cfg, "text_config", cfg)
    return list(tc.layer_types)


def make_ids(vocab, n, seed):
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    return torch.randint(10, min(vocab, 50000), (1, n), generator=g, dtype=torch.long).to("mps")


@contextlib.contextmanager
def skip_layers_above_last_recurrent(model, lt):
    """Replace the forward of every layer above the last recurrent layer.

    Their outputs are unused during backfill, so returning the input unchanged is
    not an approximation of their value: it is a declaration that the value is
    never read. Recurrent-state equality is checked to prove that.
    """
    last_rec = max(i for i, t in enumerate(lt) if t == "linear_attention")
    targets = list(range(last_rec + 1, len(lt)))
    saved = []
    try:
        for i in targets:
            layer = model.model.layers[i]
            saved.append((layer, layer.forward))
            def passthrough(self, hidden_states, *a, **k):
                return hidden_states
            layer.forward = types.MethodType(passthrough, layer)
        yield targets
    finally:
        for layer, orig in reversed(saved):
            layer.forward = orig


def recurrent_state(cache, lt):
    out = {}
    for i, t in enumerate(lt):
        if t != "linear_attention":
            continue
        cl = cache.layers[i]
        for attr in ("conv_states", "recurrent_states"):
            d = getattr(cl, attr, {}) or {}
            for sidx, tsr in d.items():
                if tsr is not None:
                    out[(i, attr, sidx)] = tsr.detach().float().cpu()
    return out


def state_max_diff(a, b):
    if set(a) != set(b):
        return float("inf")
    return max((a[k] - b[k]).abs().max().item() for k in a) if a else 0.0


def main():
    OUT.mkdir(exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager")
    model.eval().to("mps"); sync()
    lt = layer_types(model.config)
    vocab = getattr(getattr(model.config, "text_config", model.config), "vocab_size")

    last_rec = max(i for i, t in enumerate(lt) if t == "linear_attention")
    print(f"layer types: {len(lt)} layers, last recurrent layer = {last_rec}, "
          f"skippable suffix = {list(range(last_rec+1, len(lt)))} "
          f"({[lt[i] for i in range(last_rec+1, len(lt))]})")

    REPS = 8
    rows, summary = [], []
    for n in (512, 1024, 2048, 4096):
        k = int(n * 0.75)
        ids = make_ids(vocab, n, 5000 + n)
        prefix, suffix = ids[:, :k], ids[:, k:n]

        # warmup
        with torch.inference_mode():
            model(input_ids=make_ids(vocab, 64, 1), use_cache=True, logits_to_keep=1)
        sync()

        times = {"normal": [], "skip_suffix": []}
        ref_state = skip_state = None
        rng = np.random.default_rng(n)
        for rep in range(REPS):
            order = ["normal", "skip_suffix"]
            if rng.random() < 0.5:
                order.reverse()
            for method in order:
                with torch.inference_mode():
                    out = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
                cache = out.past_key_values
                sync()
                pos = torch.arange(k, n, dtype=torch.long, device="mps").unsqueeze(0)
                ctx = (skip_layers_above_last_recurrent(model, lt)
                       if method == "skip_suffix" else contextlib.nullcontext())
                sync(); t0 = time.perf_counter()
                with torch.inference_mode(), ctx:
                    o = model(input_ids=suffix, past_key_values=cache, use_cache=True,
                              position_ids=pos, logits_to_keep=1)
                sync(); dt = time.perf_counter() - t0
                times[method].append(dt)
                rows.append({"context_len": n, "boundary": k, "method": method,
                             "rep": rep, "seconds": dt})
                if rep == 0:
                    st = recurrent_state(o.past_key_values, lt)
                    if method == "normal":
                        ref_state = st
                    else:
                        skip_state = st
                del cache, o

        med_n = statistics.median(times["normal"])
        med_s = statistics.median(times["skip_suffix"])
        saving = 100.0 * (1 - med_s / med_n)
        sdiff = state_max_diff(ref_state, skip_state)
        summary.append({"context_len": n, "boundary": k, "suffix_len": n - k,
                        "normal_median_s": med_n, "skip_median_s": med_s,
                        "measured_saving_pct": saving,
                        "recurrent_state_max_abs_diff": sdiff,
                        "recurrent_state_identical": sdiff == 0.0})
        print(f"  ctx {n:>5} suffix {n-k:>4}: normal={med_n:.4f}s skip={med_s:.4f}s "
              f"saving={saving:+.2f}%  recurrent-state max diff={sdiff:g}")

    pred = json.load(open("results_layout_survey.json"))
    qwen = next(r for r in pred["rows"] if "Qwen3.5" in r["model_id"])
    print("\npredicted vs measured (Qwen3.5-0.8B, boundary 0.75):")
    print(f"{'context':>8} {'predicted':>11} {'measured':>10}")
    for s in summary:
        p = qwen["achievable_sound_saving_pct"].get(str(s["context_len"]))
        print(f"{s['context_len']:>8} {(f'{p:.2f}%' if p else 'n/a'):>11} "
              f"{s['measured_saving_pct']:>9.2f}%")

    json.dump({"model_id": MODEL_ID, "last_recurrent_layer": last_rec,
               "skipped_layers": list(range(last_rec + 1, len(lt))),
               "reps_per_cell": REPS, "randomized_order": True,
               "correctness_criterion": "recurrent-state equality, not logit equality",
               "summary": summary, "raw": rows},
              open(OUT / "suffix_skip_verification.json", "w"), indent=2)
    print(f"\nwrote {OUT / 'suffix_skip_verification.json'}")


if __name__ == "__main__":
    main()
