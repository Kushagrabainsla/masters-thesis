"""C2 of RESEARCH_JOURNAL.md E-23: fault injection into an emulated INT8 GEMM epilogue.

Predeclared pass rule:
  (a) F0 (independent int32 implementation, no fault) has zero differing elements in every seed.
  (b) The bitwise gate reports at least one differing element for each of F1..F4 over all seeds.
  (c) For F1, F2, F3 the max distance is at most 1 bf16 spacing and allclose(rtol=2^-7, atol=0)
      passes them (the blind spot). F4 is reported without an expectation.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from harness import contracts as C, int8_gemm as G, manifest as MF  # noqa: E402

RTOL_ONE_SPACING = 2.0 ** -7  # covers a one bf16 spacing difference at any magnitude


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--k", type=int, default=2048)
    ap.add_argument("--n", type=int, default=2048)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_mode = {mode: {"n": 0, "n_diff": 0, "max_spacing": 0, "allclose_pass": 0, "seeds": args.seeds}
                for mode in ("F0",) + G.FAULTS}
    for seed in range(args.seeds):
        aq, wq, sa, sw = G.make_problem(args.m, args.k, args.n, seed)
        acc = G.acc_exact(aq, wq)
        ref = G.epilogue(acc, sa, sw, "F0")
        for mode in ("F0",) + G.FAULTS:
            if mode == "F0":
                # independent implementation of the same math: int32 accumulation
                cand = G.epilogue(G.acc_int32(aq, wq), sa, sw, "F0")
            else:
                cand = G.epilogue(acc, sa, sw, mode)
            s = C.spacing_stats(ref, cand)
            assert s["n_diff"] == C.bitwise_diff_count(ref, cand)  # the two L1 measures must agree
            d = per_mode[mode]
            d["n"] += s["n"]
            d["n_diff"] += s["n_diff"]
            d["max_spacing"] = max(d["max_spacing"], s["max_spacing"])
            d["allclose_pass"] += int(C.allclose_gate(cand, ref, rtol=RTOL_ONE_SPACING, atol=0.0))
        print(f"seed {seed} done", flush=True)

    for mode, d in per_mode.items():
        d["frac_diff"] = d["n_diff"] / d["n"]

    a_ok = per_mode["F0"]["n_diff"] == 0
    b_ok = all(per_mode[f]["n_diff"] > 0 for f in G.FAULTS)
    c_spec_ok = all(per_mode[f]["max_spacing"] <= 1 for f in ("F1", "F2", "F3"))
    c_blind_ok = all(per_mode[f]["allclose_pass"] == args.seeds for f in ("F1", "F2", "F3"))
    result = {
        "criterion": "C2",
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "seeds": args.seeds,
        "allclose_rtol": RTOL_ONE_SPACING,
        "per_mode": per_mode,
        "checks": {
            "a_F0_zero_diff": a_ok,
            "b_bitwise_detects_F1_to_F4": b_ok,
            "c_spec_F1_F2_F3_within_one_spacing": c_spec_ok,
            "c_blind_spot_allclose_passes_F1_F2_F3": c_blind_ok,
        },
        "PASS": bool(a_ok and b_ok and c_spec_ok and c_blind_ok),
        "notes": {
            "F4_allclose_pass_seeds": per_mode["F4"]["allclose_pass"],
            "F4_max_spacing": per_mode["F4"]["max_spacing"],
            "faults_are_own_constructions": "same fault class as arXiv 2609.00363, not a reproduction of its five faults",
        },
        "manifest": MF.build_manifest(ROOT, {"experiment": "E-23/C2"}),
    }
    MF.write_json(Path(args.out) / "fault_injection.json", result)
    print(f"C2 PASS={result['PASS']}  checks={result['checks']}")
    for mode, d in per_mode.items():
        print(f"  {mode}: n_diff={d['n_diff']}/{d['n']} ({d['frac_diff']:.2e}) max_spacing={d['max_spacing']} "
              f"allclose_pass={d['allclose_pass']}/{args.seeds}")


if __name__ == "__main__":
    main()
