"""E-30 analysis (amendment A1 after R-34): validity, KILL 1, KILL 2 (constrained form), GO, PIVOT.

Input: medians.json from sweep.py:
{
  "limits": {"default": {"prefill": {"0": t, "18": .., "36": .., "54": .., "0_end": t},
                         "gemm":    {...}},
             "680": {...}},
  "caps": {"none": {"T_p": .., "T_d": .., "C_p": .., "C_d": ..}, "150": {...}, "100": {...}, "70": {...}},
  "validity": {"gpu_power_limit_w": .., "prefill_power_median_w": ..}
}
"""

from __future__ import annotations

import argparse
import itertools
import json

KILL1_LOSS = 0.03
GO_LOSS = 0.10
KILL2_GAIN = 0.05
CPU_FLOOR = 0.80  # constrained form: keep at least 80% of the uncapped CPU work
SHARES = (0.3, 0.5, 0.7)
CPU_WEIGHTS = (0.25, 0.5, 1.0)  # reported only
POWER_MIN = 0.60  # prefill must draw at least 60% of the GPU power limit
DRIFT_MAX = 0.02
DOCUMENTED = ("default", "680")


def loss(series: dict, low: str = "0", high: str = "54") -> float:
    return 1 - series[high] / series[low]


def gpu_term(caps, cp, cd, f):
    base = caps["none"]
    return f * caps[cp]["T_p"] / base["T_p"] + (1 - f) * caps[cd]["T_d"] / base["T_d"]


def cpu_term(caps, cp, cd, f):
    base = caps["none"]
    return (f * caps[cp]["C_p"] + (1 - f) * caps[cd]["C_d"]) / (f * base["C_p"] + (1 - f) * base["C_d"])


def weighted_gain(caps: dict, f: float, w: float) -> float:
    names = list(caps)
    u = lambda cp, cd: gpu_term(caps, cp, cd, f) + w * cpu_term(caps, cp, cd, f)  # noqa: E731
    static = max(u(c, c) for c in names)
    oracle = max(u(cp, cd) for cp, cd in itertools.product(names, names))
    return oracle / static - 1


def constrained_gain(caps: dict, f: float) -> float:
    """Max GPU term subject to the CPU term >= CPU_FLOOR; per-phase pair vs one static cap."""
    names = list(caps)
    ok = lambda cp, cd: cpu_term(caps, cp, cd, f) >= CPU_FLOOR - 1e-12  # noqa: E731
    static = max(gpu_term(caps, c, c, f) for c in names if ok(c, c))
    oracle = max(gpu_term(caps, cp, cd, f) for cp, cd in itertools.product(names, names) if ok(cp, cd))
    return oracle / static - 1


def decide(data: dict) -> dict:
    result: dict = {}
    validity = data.get("validity", {})
    power, limit = validity.get("prefill_power_median_w"), validity.get("gpu_power_limit_w")
    if power is not None and limit:
        result["prefill_power_fraction"] = power / limit
        if power / limit < POWER_MIN:
            result["decision"] = "INVALID (prefill is not compute-bound: GPU power under 60% of its limit)"
            return result
    limits = data["limits"]
    drifts = {k: abs(v["prefill"]["0_end"] / v["prefill"]["0"] - 1)
              for k, v in limits.items() if "prefill" in v and "0_end" in v["prefill"]}
    result["drift"] = drifts
    losses = {name: {kind: loss(v[kind]) for kind in ("prefill", "gemm") if kind in v} for name, v in limits.items()}
    result["losses"] = losses
    if any(d > DRIFT_MAX for d in drifts.values()):
        result["decision"] = "INCONCLUSIVE (N=0 drift above 2%)"
        return result
    documented = {k: v for k, v in losses.items() if k in DOCUMENTED and "prefill" in v}
    if not documented:
        result["decision"] = "INCOMPLETE (no prefill data at a documented limit)"
        return result
    kill1 = all(v["prefill"] < KILL1_LOSS and v.get("gemm", 0.0) < KILL1_LOSS for v in documented.values())
    result["kill1"] = kill1
    if kill1:
        result["decision"] = "KILL (KILL 1)"
        return result
    caps = data.get("caps")
    if not caps or "none" not in caps:
        result["decision"] = "INCOMPLETE (KILL 1 failed; CPU caps not testable, KILL 2 unknown)"
        return result
    result["constrained_gain"] = {str(f): constrained_gain(caps, f) for f in SHARES}
    result["weighted_gain"] = {f"f={f},w={w}": weighted_gain(caps, f, w) for f in SHARES for w in CPU_WEIGHTS}
    max_loss = max(v["prefill"] for v in documented.values())
    if all(g < KILL2_GAIN for g in result["constrained_gain"].values()):
        result["decision"] = "KILL (KILL 2: a static CPU cap is enough)"
    elif max_loss >= GO_LOSS:
        result["decision"] = "GO"
    else:
        result["decision"] = "PIVOT (loss 3% to 10%: measurement note)"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("medians")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = decide(json.loads(open(args.medians).read()))
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(result["decision"])


if __name__ == "__main__":
    main()
