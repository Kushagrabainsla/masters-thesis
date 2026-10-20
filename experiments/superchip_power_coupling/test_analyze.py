"""Unit tests for analyze.py (amendment A1) on synthetic data."""

import unittest

import analyze


def limits(prefill_54, gemm_54=100.0, with_680=None, end=100.0):
    d = {"default": {"prefill": {"0": 100.0, "54": prefill_54, "0_end": end},
                     "gemm": {"0": 100.0, "54": gemm_54}}}
    if with_680 is not None:
        d["680"] = {"prefill": {"0": 100.0, "54": with_680, "0_end": 100.0},
                    "gemm": {"0": 100.0, "54": with_680}}
    return d


VALID = {"gpu_power_limit_w": 900.0, "prefill_power_median_w": 700.0}
FLAT_CAPS = {c: {"T_p": 100.0, "T_d": 100.0, "C_p": 100.0, "C_d": 100.0} for c in ("none", "150", "100", "70")}
# Capping the CPU raises prefill throughput (60 -> 100) and costs CPU work; decode does not care.
PHASE_CAPS = {
    "none": {"T_p": 60.0, "T_d": 100.0, "C_p": 100.0, "C_d": 100.0},
    "150": {"T_p": 75.0, "T_d": 100.0, "C_p": 85.0, "C_d": 85.0},
    "100": {"T_p": 90.0, "T_d": 100.0, "C_p": 70.0, "C_d": 70.0},
    "70": {"T_p": 100.0, "T_d": 100.0, "C_p": 60.0, "C_d": 60.0},
}


class AnalyzeTests(unittest.TestCase):
    def test_invalid_when_prefill_not_compute_bound(self):
        r = analyze.decide({"limits": limits(80.0), "validity": {"gpu_power_limit_w": 900.0,
                                                                 "prefill_power_median_w": 300.0}})
        self.assertTrue(r["decision"].startswith("INVALID"))

    def test_drift(self):
        r = analyze.decide({"limits": limits(80.0, end=96.0), "validity": VALID})
        self.assertTrue(r["decision"].startswith("INCONCLUSIVE"))

    def test_kill1(self):
        r = analyze.decide({"limits": limits(98.5, with_680=98.0), "validity": VALID})
        self.assertEqual(r["decision"], "KILL (KILL 1)")

    def test_kill2_flat_caps(self):
        r = analyze.decide({"limits": limits(85.0, with_680=80.0), "caps": FLAT_CAPS, "validity": VALID})
        self.assertEqual(r["decision"], "KILL (KILL 2: a static CPU cap is enough)")

    def test_go(self):
        r = analyze.decide({"limits": limits(85.0, with_680=70.0), "caps": PHASE_CAPS, "validity": VALID})
        self.assertGreater(max(r["constrained_gain"].values()), 0.05)
        # At f = 0.7 the CPU floor admits no better per-phase pair than the static 150 W cap.
        self.assertAlmostEqual(r["constrained_gain"]["0.7"], 0.0)
        self.assertEqual(r["decision"], "GO")

    def test_pivot_small_loss(self):
        r = analyze.decide({"limits": limits(95.0, with_680=94.0), "caps": PHASE_CAPS, "validity": VALID})
        self.assertTrue(r["decision"].startswith("PIVOT"))

    def test_incomplete_without_caps(self):
        r = analyze.decide({"limits": limits(80.0), "validity": VALID})
        self.assertTrue(r["decision"].startswith("INCOMPLETE"))

    def test_constrained_gain_known_value(self):
        # f = 0.5: best static feasible cap is 150 (GPU term 1.125); best pair is (70, none) (1.3333).
        self.assertAlmostEqual(analyze.constrained_gain(PHASE_CAPS, 0.5), 1.3333333 / 1.125 - 1, places=5)

    def test_static_equals_oracle_when_one_cap_dominates(self):
        caps = {"none": {"T_p": 50.0, "T_d": 50.0, "C_p": 100.0, "C_d": 100.0},
                "70": {"T_p": 100.0, "T_d": 100.0, "C_p": 100.0, "C_d": 100.0}}
        self.assertAlmostEqual(analyze.constrained_gain(caps, 0.5), 0.0)


if __name__ == "__main__":
    unittest.main()
