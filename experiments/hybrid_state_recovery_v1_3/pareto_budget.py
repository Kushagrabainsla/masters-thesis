#!/usr/bin/env python3
"""Memory-budget Pareto: sparse recurrent checkpoints vs FA-activation caching.

The simple-baseline attack on "cache post-FA-block activations so the backfill can
skip FA blocks" is: spend those same bytes on MORE recurrent checkpoints instead.

Both are replay-cost reducers but they differ structurally:
  * recurrent checkpoint : R bytes each, position-KEYED. M checkpoints cut the
                           expected replay LENGTH to ~N/(2M).
  * FA activation cache  : 6*N*2048 bytes once, position-INDEPENDENT. It cuts the
                           replay COST PER TOKEN by the FA share, but does not
                           shorten the replay at all.

So this is a real allocation question, not a domination. We solve it with the
measured byte constants and the FLOP model's cost curves.
"""
import json

import numpy as np

with open("results_flop_model.json") as f:
    FM = json.load(f)

D = FM["dims"]
N_FA = FM["layer_counts"]["full_attention"]
N_LN = FM["layer_counts"]["linear_attention"]
FA_W = FM["per_token_mac"]["fa_weight_mac_all_fa_layers"]
LN_W = FM["per_token_mac"]["linear_weight_mac_all_linear_layers"]
LN_X = FM["per_token_mac"]["linear_extra_mac_per_token_all_layers"]
Q_W = D["query_width"]

# Measured byte constants (observed in the live cache, v1.3 state accounting).
BYTES_PER_TOK_PER_FA_LAYER_KV = 2 * D["num_key_value_heads"] * D["head_dim"] * 2  # K and V, fp16
BYTES_PER_TOK_PER_LAYER_ACT = D["hidden_size"] * 2                                # fp16 hidden state
ACT_BYTES_PER_TOK = N_FA * BYTES_PER_TOK_PER_LAYER_ACT
# Recurrent snapshot: read the MEASURED bytes out of the live cache rather than
# assuming a dtype. The GDN recurrent state is kept in float32 for numerical
# stability even when the model runs in float16, so an fp16 formula understates it
# by 2x and silently biases this whole comparison in favour of checkpointing.
_acct = json.load(open("results/state_size_accounting.json"))
R = int(_acct[0]["recurrent_snapshot_total_bytes"])
REC_STATE_BYTES = int(_acct[0]["recurrent_state_bytes"])
REC_CONV_BYTES = int(_acct[0]["conv_state_bytes"])
_REC_BYTES_PER_ELEM = REC_STATE_BYTES / (N_LN * D["linear_key_heads"]
                                         * D["linear_key_head_dim"] * D["linear_value_head_dim"])


def replay_mac(suffix, context, skip_fa_blocks: bool):
    """MACs to replay `suffix` tokens ending at absolute position `context`."""
    k = context - suffix
    causal_pairs = suffix * k + suffix * (suffix + 1) / 2.0
    fa = suffix * FA_W + N_FA * 2.0 * causal_pairs * Q_W
    ln = suffix * LN_W + suffix * LN_X
    return ln if skip_fa_blocks else ln + fa


def expected_replay_len(n, m):
    """Expected distance back to the nearest checkpoint <= a uniform divergence depth.

    With m evenly spaced checkpoints over [0, n], a uniformly distributed divergence
    depth sits on average n/(2m) past the nearest one. m=0 means no checkpoint at all,
    so the whole prefix must be replayed.
    """
    if m <= 0:
        return float(n)
    return n / (2.0 * m)


def main():
    print(f"recurrent snapshot R = {R/2**20:.2f} MiB   (MEASURED, constant in context)")
    print(f"  = {REC_STATE_BYTES/2**20:.2f} MiB recurrent state @ {_REC_BYTES_PER_ELEM:.1f} B/elem "
          f"+ {REC_CONV_BYTES/2**20:.2f} MiB conv state")
    print(f"FA activation cache = {ACT_BYTES_PER_TOK} B/token  "
          f"(= {ACT_BYTES_PER_TOK/2**20:.4f} MiB per 1K tokens)")
    print(f"FA KV cache         = {N_FA*BYTES_PER_TOK_PER_FA_LAYER_KV} B/token")
    print()

    rows = []
    for n in [2048, 8192, 32768, 131072]:
        act_bytes = ACT_BYTES_PER_TOK * n
        ckpt_equiv = act_bytes / R
        print(f"=== context {n} ===")
        print(f"  activation cache costs {act_bytes/2**20:8.2f} MiB "
              f"= {ckpt_equiv:6.2f} recurrent checkpoints")
        print(f"  {'budget(MiB)':>12} {'ckpt-only':>26} {'act+ckpt':>26}  winner")
        for budget_ckpts in [1, 2, 4, 8, 16, 32, 64, 128]:
            budget = budget_ckpts * R
            # Strategy A: all budget on checkpoints.
            m_a = int(budget // R)
            s_a = expected_replay_len(n, m_a)
            cost_a = replay_mac(s_a, n, skip_fa_blocks=False)
            # Strategy B: pay for the activation cache, rest on checkpoints.
            rem = budget - act_bytes
            m_b = int(rem // R) if rem >= 0 else -1
            if m_b < 0:
                cost_b = float("inf"); s_b = float("nan"); m_b_disp = "n/a"
            else:
                s_b = expected_replay_len(n, m_b)
                cost_b = replay_mac(s_b, n, skip_fa_blocks=True)
                m_b_disp = str(m_b)
            winner = "checkpoints" if cost_a <= cost_b else "ACTIVATIONS"
            print(f"  {budget/2**20:>12.1f} "
                  f"m={m_a:<4} replay={s_a:>9.1f} {cost_a/1e9:>7.1f} GMAC "
                  f"m={m_b_disp:<4} replay={s_b:>9.1f} {cost_b/1e9:>7.1f} GMAC  {winner}")
            rows.append({
                "context_len": n, "budget_bytes": budget, "budget_in_checkpoints": budget_ckpts,
                "ckpt_only_m": m_a, "ckpt_only_expected_replay": s_a, "ckpt_only_gmac": cost_a / 1e9,
                "act_plus_ckpt_m": m_b, "act_plus_ckpt_expected_replay": s_b,
                "act_plus_ckpt_gmac": None if cost_b == float("inf") else cost_b / 1e9,
                "winner": winner,
            })
        print()

    with open("results_pareto_budget.json", "w") as f:
        json.dump({
            "recurrent_snapshot_bytes": R,
            "activation_bytes_per_token": ACT_BYTES_PER_TOK,
            "fa_kv_bytes_per_token": N_FA * BYTES_PER_TOK_PER_FA_LAYER_KV,
            "model_id": FM["model_id"],
            "divergence_depth_model": "uniform over [0, context]; m evenly spaced checkpoints",
            "caveat": (
                "Analytical MAC model with measured byte constants. It assumes a uniform "
                "divergence-depth distribution and evenly spaced checkpoints, which is the "
                "best case for checkpointing; a heavy-tailed real distribution would change "
                "the crossover. It also charges activation caching nothing for bandwidth."
            ),
            "rows": rows,
        }, f, indent=2)
    print("wrote results_pareto_budget.json")


if __name__ == "__main__":
    main()
