"""Noise-aware comparison of a candidate run against a baseline run (docs/experiments/MEASUREMENT.md).

For each metric in both runs: the median of each, the ratio candidate / baseline, and a bootstrap 95%
interval of that ratio (resampling the repeats of each run). The noise band is the larger of the
minimum effect (--mde, default 2%) and the A/A spread, when an A/A run of the baseline is given.

Verdict per metric (direction from higher_is_better):
  BETTER     the whole interval is past +band in the good direction
  WORSE      the whole interval is past -band in the bad direction
  NO_CHANGE  the whole interval is inside +-band
  UNCLEAR    anything else (more repeats needed)
With k metrics, each interval uses a Bonferroni level (1 - 0.05 / k), so that the set has at most a 5%
false-alarm rate.

Before it compares, it checks the fingerprints. A different GPU type, GPU count, driver, CUDA or
software version makes the result a cross-setup comparison: the tool says so, and with --strict it
refuses.

Usage:
  compare_runs.py <baseline_run_id> <candidate_run_id> [--aa <baseline_repeat_run_id>] [--mde 0.02] [--strict]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys

from ledger import load

FINGERPRINT = ("gpu_name", "gpu_count", "driver", "cuda")


def median_ratio_interval(base: list[float], cand: list[float], level: float, n: int = 4000,
                          seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    point = statistics.median(cand) / statistics.median(base)
    draws = sorted(
        statistics.median(rng.choices(cand, k=len(cand))) / statistics.median(rng.choices(base, k=len(base)))
        for _ in range(n)
    )
    lo = draws[int((1 - level) / 2 * n)]
    hi = draws[min(n - 1, int((1 + level) / 2 * n))]
    return point, lo, hi


def aa_spread(base: list[float], again: list[float]) -> float:
    return abs(statistics.median(again) / statistics.median(base) - 1)


def verdict(lo: float, hi: float, band: float, higher_is_better: bool) -> str:
    # Express the interval as a gain in the good direction.
    g_lo, g_hi = (lo - 1, hi - 1) if higher_is_better else (1 - hi, 1 - lo)
    if g_lo > band:
        return "BETTER"
    if g_hi < -band:
        return "WORSE"
    if -band <= g_lo and g_hi <= band:
        return "NO_CHANGE"
    return "UNCLEAR"


def fingerprint_diff(a: dict, b: dict) -> list[str]:
    diffs = [f"hardware.{k}: {a['hardware'].get(k)} != {b['hardware'].get(k)}"
             for k in FINGERPRINT if a["hardware"].get(k) != b["hardware"].get(k)]
    for k in sorted(set(a["software"]) | set(b["software"])):
        if a["software"].get(k) != b["software"].get(k):
            diffs.append(f"software.{k}: {a['software'].get(k)} != {b['software'].get(k)}")
    return diffs


def compare(base: dict, cand: dict, aa: dict | None, mde: float) -> dict:
    names = sorted(set(base["metrics"]) & set(cand["metrics"]))
    level = 1 - 0.05 / max(1, len(names))
    rows = {}
    for name in names:
        b, c = base["metrics"][name], cand["metrics"][name]
        band = mde
        if aa and name in aa["metrics"]:
            band = max(mde, aa_spread(b["values"], aa["metrics"][name]["values"]))
        point, lo, hi = median_ratio_interval(b["values"], c["values"], level)
        rows[name] = {
            "baseline_median": statistics.median(b["values"]),
            "candidate_median": statistics.median(c["values"]),
            "ratio": round(point, 4),
            "interval": [round(lo, 4), round(hi, 4)],
            "band": round(band, 4),
            "repeats": [len(b["values"]), len(c["values"])],
            "verdict": verdict(lo, hi, band, b["higher_is_better"]),
        }
    return {
        "baseline": base["run_id"],
        "candidate": cand["run_id"],
        "same_config": base["config_hash"] == cand["config_hash"],
        "fingerprint_differences": fingerprint_diff(base, cand),
        "interval_level": round(level, 4),
        "only_in_baseline": sorted(set(base["metrics"]) - set(cand["metrics"])),
        "only_in_candidate": sorted(set(cand["metrics"]) - set(base["metrics"])),
        "metrics": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--aa", help="a second run of the baseline config (A/A), to measure the noise")
    parser.add_argument("--mde", type=float, default=0.02, help="minimum effect that matters (fraction)")
    parser.add_argument("--strict", action="store_true", help="refuse when the fingerprints differ")
    args = parser.parse_args()
    runs = {r["run_id"]: r for r in load()}
    for rid in filter(None, (args.baseline, args.candidate, args.aa)):
        if rid not in runs:
            sys.exit(f"unknown run_id: {rid}")
    result = compare(runs[args.baseline], runs[args.candidate], runs.get(args.aa) if args.aa else None, args.mde)
    if result["fingerprint_differences"]:
        print("WARNING: cross-setup comparison:", *result["fingerprint_differences"], sep="\n  ", file=sys.stderr)
        if args.strict:
            sys.exit(2)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
