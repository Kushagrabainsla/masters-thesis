#!/usr/bin/env python3
"""RQ1 gate: can per-group state-reuse soundness be decided at plan level, on a Mac?

Three things are established here, all on CPU with no GPU:

  A. The soundness condition. A reuse plan assigns each (layer, position) to
     REUSE or EXECUTE. Executing layer L at position p needs the hidden state
     from L-1 at p. That is available only if L-1 also executes at p, or if
     L-1's output activation at p was cached. Therefore:

         a plan is sound iff reuse lengths are NON-DECREASING along the layer
         stack, except where a cached activation covers the step down.

     Interleaving is what makes this bite: a deep full-attention layer that
     reuses more than the recurrent layer above it starves that layer.

  B. Real divergent plans from real vLLM, at the pinned commit, on CPU. Mamba
     align mode reproduces vLLM #45238: full attention hits N tokens while the
     recurrent group hits 0.

  C. Cross-validation. The static verdicts are checked against the measured
     equivalence results from the MPS harness in
     experiments/hybrid_state_recovery_v1_3. If the checker disagrees with the
     measurement on any plan, the checker is wrong and this gate fails.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

OUT = Path("results")

# ---------------------------------------------------------------- A. semantics

REUSE = "REUSE"
EXECUTE = "EXECUTE"


@dataclass
class ReusePlan:
    """A per-layer-group reuse plan over a shared prefix of `prefix_len` tokens."""

    name: str
    # layer index -> group id, in execution order. Encodes the interleaving.
    layer_groups: list[int]
    # group id -> number of leading prefix tokens whose state is REUSED
    reuse_len: dict[int, int]
    prefix_len: int
    # layer indices whose post-block output activations are cached for the
    # whole prefix. A cached activation at layer L covers a step down at L+1.
    activation_cached_layers: set[int] = field(default_factory=set)
    # groups whose state is supplied by an external load rather than recomputed.
    externally_loaded_groups: set[int] = field(default_factory=set)
    note: str = ""


def check_soundness(plan: ReusePlan) -> dict:
    """Decide soundness structurally. No model execution involved."""
    violations = []
    n = len(plan.layer_groups)

    # An externally loaded group's state is valid at its own reuse boundary, so
    # for closure purposes it behaves as if it reused that far.
    eff = dict(plan.reuse_len)
    for g in plan.externally_loaded_groups:
        eff[g] = max(eff.get(g, 0), max(plan.reuse_len.values(), default=0))

    for L in range(1, n):
        g_cur, g_prev = plan.layer_groups[L], plan.layer_groups[L - 1]
        r_cur, r_prev = eff.get(g_cur, 0), eff.get(g_prev, 0)
        if r_cur >= r_prev:
            continue  # no step down, inputs exist wherever L executes
        # L executes on [r_cur, prefix_len) but L-1 produced nothing on
        # [r_cur, r_prev). A cached activation at L-1 patches exactly that hole.
        if (L - 1) in plan.activation_cached_layers:
            continue
        violations.append({
            "layer": L,
            "layer_group": g_cur,
            "predecessor_layer": L - 1,
            "predecessor_group": g_prev,
            "executes_from": r_cur,
            "predecessor_produces_from": r_prev,
            "starved_positions": [r_cur, r_prev],
            "why": (
                f"layer {L} (group {g_cur}) executes from token {r_cur}, but its "
                f"predecessor layer {L-1} (group {g_prev}) reuses through {r_prev} "
                f"and so emits no hidden state on [{r_cur}, {r_prev})"
            ),
        })

    return {
        "plan": plan.name,
        "sound": not violations,
        "num_violations": len(violations),
        "violations": violations,
        "effective_reuse_len": eff,
        "note": plan.note,
    }


# ------------------------------------------------- B. real vLLM divergent plans

def probe_real_vllm() -> dict:
    """Drive the pinned vLLM coordinator on CPU and capture real per-group hits."""
    import torch
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_utils import init_none_hash
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec)
    from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request

    init_none_hash(sha256)
    BS = 16
    PROMPT = 128

    out = {"block_size": BS, "prompt_tokens": PROMPT, "cases": []}
    for mode in ("none", "align"):
        specs = [
            FullAttentionSpec(block_size=BS, num_kv_heads=1, head_size=8, dtype=torch.float32),
            MambaSpec(block_size=BS, shapes=((1, 8),), dtypes=(torch.float32,),
                      mamba_cache_mode=mode),
        ]
        cfg = KVCacheConfig(
            num_blocks=256, kv_cache_tensors=[],
            kv_cache_groups=[KVCacheGroupSpec([f"l{i}"], s) for i, s in enumerate(specs)],
        )
        mgr = make_kv_cache_manager(cfg, max_model_len=4096, max_in_flight_tokens=4096,
                                    hash_block_size=BS, enable_caching=True)
        r1 = make_request("r1", list(range(1, PROMPT + 2)), BS, sha256)
        mgr.allocate_slots(r1, len(r1.prompt_token_ids))
        r1.num_computed_tokens = len(r1.prompt_token_ids)
        mgr.cache_blocks(r1, len(r1.prompt_token_ids))

        r2 = make_request("r2", list(range(1, PROMPT + 2)) + [999, 998], BS, sha256)
        _, reconciled, _ = mgr.get_computed_blocks(r2)
        _, per_group = mgr.coordinator.find_longest_cache_hit_per_group(
            r2.block_hashes, r2.num_tokens - 1)
        blocks, num_local, _, diverged = mgr.get_computed_blocks_for_connector(r2)

        out["cases"].append({
            "mamba_cache_mode": mode,
            "coordinator": type(mgr.coordinator).__name__,
            "full_attention_group_id": mgr.coordinator.full_attention_group_id,
            "cache_hit_alignment_tokens": mgr.coordinator._cache_hit_alignment_tokens,
            "per_group_hits": list(per_group),
            "reconciled_hit_default_path": reconciled,
            "connector_path_num_local": num_local,
            "connector_path_hit_diverged": bool(diverged),
            "tokens_discarded_by_reconciliation": max(per_group) - reconciled,
        })
    return out


# -------------------------------------------------------- C. candidate plans

def qwen35_layout() -> list[int]:
    """Qwen3.5-0.8B: 24 layers, full attention at 3,7,11,15,19,23. Group 0 = FA."""
    fa = {3, 7, 11, 15, 19, 23}
    return [0 if i in fa else 1 for i in range(24)]


def candidate_plans(prefix_len: int, fa_hit: int) -> list[ReusePlan]:
    layers = qwen35_layout()
    fa_layers = [i for i, g in enumerate(layers) if g == 0]
    return [
        ReusePlan(
            name="upstream_default_reconciled",
            layer_groups=layers, reuse_len={0: 0, 1: 0}, prefix_len=prefix_len,
            note=("What vLLM does today: min() across groups, so the whole FA hit "
                  "is discarded. Sound, and wasteful."),
        ),
        ReusePlan(
            name="upstream_connector_divergent_load",
            layer_groups=layers, reuse_len={0: fa_hit, 1: 0}, prefix_len=prefix_len,
            externally_loaded_groups={1},
            note=("get_computed_blocks_for_connector: keep the deeper FA hit and let "
                  "the connector supply recurrent state. #52773 load-backfill."),
        ),
        ReusePlan(
            name="issue_53041_proposal_B_recompute_backfill",
            layer_groups=layers, reuse_len={0: fa_hit, 1: 0}, prefix_len=prefix_len,
            note=("'FA layers skip the gap range because their KV is pre-loaded; only "
                  "SWA/Mamba layers run forward.' No external load, no activations."),
        ),
        ReusePlan(
            name="frozen_kv_but_fa_still_executes",
            layer_groups=layers, reuse_len={0: 0, 1: 0}, prefix_len=prefix_len,
            note=("FA blocks execute over the gap but read cached K/V instead of "
                  "recomputing it. Intra-layer reuse, not a skip."),
        ),
        ReusePlan(
            name="activation_cached_fa_skip",
            layer_groups=layers, reuse_len={0: fa_hit, 1: 0}, prefix_len=prefix_len,
            activation_cached_layers=set(fa_layers),
            note="Skip FA blocks using cached post-block hidden states. The oracle path.",
        ),
    ]


MEASURED = {
    # plan name -> (max logit diff vs reference, verdict from the MPS harness)
    "issue_53041_proposal_B_recompute_backfill": (1.0554, "INCORRECT"),
    "frozen_kv_but_fa_still_executes": (0.0117, "CORRECT"),
    "activation_cached_fa_skip": (0.0117, "CORRECT"),
}
NOISE_FLOOR = 0.15


def main() -> int:
    OUT.mkdir(exist_ok=True)
    prefix_len, fa_hit = 128, 128

    plans = candidate_plans(prefix_len, fa_hit)
    verdicts = [check_soundness(p) for p in plans]

    print("=" * 78)
    print("A. STATIC SOUNDNESS VERDICTS  (no model execution)")
    print("=" * 78)
    for v in verdicts:
        mark = "SOUND  " if v["sound"] else "UNSOUND"
        print(f"  [{mark}] {v['plan']}")
        for viol in v["violations"]:
            print(f"            {viol['why']}")

    print()
    print("=" * 78)
    print("B. REAL vLLM PER-GROUP HITS  (pinned commit, CPU only)")
    print("=" * 78)
    try:
        real = probe_real_vllm()
        for c in real["cases"]:
            print(f"  mamba_cache_mode={c['mamba_cache_mode']:<6} "
                  f"per_group_hits={c['per_group_hits']}  "
                  f"default_path={c['reconciled_hit_default_path']:<4} "
                  f"connector_local={c['connector_path_num_local']:<4} "
                  f"diverged={c['connector_path_hit_diverged']}")
            if c["tokens_discarded_by_reconciliation"]:
                print(f"     -> reconciliation discards "
                      f"{c['tokens_discarded_by_reconciliation']} already-valid FA tokens "
                      f"(reproduces vLLM #45238)")
    except Exception as e:
        import traceback
        real = {"error": repr(e), "trace": traceback.format_exc()}
        print(f"  FAILED: {e}")

    print()
    print("=" * 78)
    print("C. CROSS-VALIDATION vs MEASURED EQUIVALENCE (MPS harness)")
    print("=" * 78)
    checks, agree = [], True
    for v in verdicts:
        if v["plan"] not in MEASURED:
            continue
        diff, measured = MEASURED[v["plan"]]
        measured_sound = diff <= NOISE_FLOOR
        ok = (measured_sound == v["sound"])
        agree &= ok
        checks.append({
            "plan": v["plan"], "static_sound": v["sound"],
            "measured_max_logit_diff": diff, "measured_verdict": measured,
            "measured_sound": measured_sound, "agreement": ok,
        })
        print(f"  {'OK ' if ok else 'MISMATCH'}  {v['plan']}")
        print(f"            static={'sound' if v['sound'] else 'unsound'}  "
              f"measured diff={diff} -> {measured}")

    gate_pass = agree and "error" not in real
    print()
    print("=" * 78)
    print(f"GATE: {'PASS' if gate_pass else 'FAIL'}")
    print("=" * 78)
    if gate_pass:
        print("  The static checker agrees with the MPS measurement on every plan it")
        print("  has ground truth for, and real divergent plans can be produced from")
        print("  the pinned vLLM commit on this machine with no GPU.")

    (OUT / "soundness_gate.json").write_text(json.dumps({
        "prefix_len": prefix_len, "fa_hit": fa_hit,
        "layout": "Qwen3.5-0.8B, 24 layers, FA at 3,7,11,15,19,23",
        "static_verdicts": verdicts,
        "real_vllm": real,
        "cross_validation": checks,
        "noise_floor_used": NOISE_FLOOR,
        "gate_pass": gate_pass,
    }, indent=2))
    print(f"\nwrote {OUT / 'soundness_gate.json'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
