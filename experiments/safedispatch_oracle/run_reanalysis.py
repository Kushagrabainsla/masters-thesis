"""Development re-analysis of E-23 attempts 1 and 2 (RESEARCH_JOURNAL.md, Amendment A1).

No model runs. Reads the per-position data saved by run_model_pilot.py and reports:
  - two-sided coverage of the original rate-based Jeffreys interval and of amendment A1
  - one-sided 95% upper bound miss rate and tightness against simple random sampling (Wilson)
  - the flip invariant: a greedy flip requires margin <= 2 * max|logit difference| at that
    position. If elementwise |dlogit| <= e then top1 cannot change when margin > 2e. Any
    violation means a bug in the L2 or L3 computation.

The amendment was designed after seeing this data, so results here are development evidence.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from harness import estimator as E, manifest as MF  # noqa: E402


def load(attempt_dir: Path):
    m, f, d = [], [], []
    for jf in sorted((attempt_dir / "per_prompt").glob("[0-9][0-9][0-9].npz")):
        z = np.load(jf)
        m.append(z["margins"]); f.append(z["flip_dec"]); d.append(z["maxdiff"])
    return np.concatenate(m), np.concatenate(f), np.concatenate(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", nargs="+", default=["attempt_01", "attempt_02"])
    ap.add_argument("--reps", type=int, default=500)
    ap.add_argument("--out", default="results/reanalysis_attempts_01_02.json")
    args = ap.parse_args()

    out = {"note": "development evidence only, amendment A1 was designed after seeing this data",
           "attempts": {}}
    for a in args.attempts:
        margins, flips, maxdiff = load(ROOT / "results" / a)
        # invariant: every flip has margin <= 2 * max|dlogit| (small tolerance for float rounding)
        viol = int((flips & (margins > 2.0 * maxdiff + 1e-4)).sum())
        rec = {
            "n_positions": int(len(margins)), "n_flips": int(flips.sum()),
            "invariant_flip_needs_margin_le_2x_maxdiff": {
                "violations": viol, "PASS": viol == 0,
                "max_margin_among_flips": float(margins[flips].max()) if flips.any() else None,
                "min_2x_maxdiff_at_high_margin": float((2 * maxdiff)[margins > 1.0].min()),
            },
            "intervals": {},
        }
        for iv in ("jeffreys_rate", "beta_binomial"):
            c = E.coverage_experiment(margins, flips, budget_frac=0.25, n_rep=args.reps, seed=0, interval=iv)
            rec["intervals"][iv] = {k: c[k] for k in (
                "coverage_stratified", "coverage_srs_wilson", "mean_width_stratified", "mean_width_srs_wilson",
                "one_sided_95_ucb_miss_stratified", "one_sided_95_ucb_miss_srs_wilson",
                "mean_ucb_over_truth_stratified", "mean_ucb_over_truth_srs_wilson",
                "bin_sizes", "bin_flips", "bin_alloc", "truth_flip_rate")}
        out["attempts"][a] = rec

    out["manifest"] = MF.build_manifest(ROOT, {"experiment": "E-23/reanalysis_A1"})
    MF.write_json(ROOT / args.out, out)
    for a, rec in out["attempts"].items():
        inv = rec["invariant_flip_needs_margin_le_2x_maxdiff"]
        print(f"\n{a}: {rec['n_flips']} flips / {rec['n_positions']} positions | invariant PASS={inv['PASS']} "
              f"(violations {inv['violations']}, max margin among flips {inv['max_margin_among_flips']})")
        for iv, r in rec["intervals"].items():
            print(f"  {iv:14s} two-sided coverage {r['coverage_stratified']:.3f} (SRS {r['coverage_srs_wilson']:.3f}) | "
                  f"one-sided UCB miss {r['one_sided_95_ucb_miss_stratified']:.3f} (SRS {r['one_sided_95_ucb_miss_srs_wilson']:.3f}) | "
                  f"UCB/truth {r['mean_ucb_over_truth_stratified']:.2f} (SRS {r['mean_ucb_over_truth_srs_wilson']:.2f})")


if __name__ == "__main__":
    main()
