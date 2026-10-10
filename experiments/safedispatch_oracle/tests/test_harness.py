"""Unit tests for the pilot harness. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import contracts as C, estimator as E, int8_gemm as G  # noqa: E402


def bump(x: torch.Tensor, delta: int) -> torch.Tensor:
    """Move each element by `delta` representable values (bf16/fp16 via int16 view)."""
    return (x.view(torch.int16) + delta).view(x.dtype)


class TestContracts(unittest.TestCase):
    def test_adjacent_bf16_distance_is_one(self):
        x = torch.tensor([1.0, 3.5, 100.0, 0.01], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(C.spacing_distance(x, bump(x, 1)), torch.ones(4, dtype=torch.int64)))

    def test_negative_side_adjacent_and_zero(self):
        x = torch.tensor([-1.0, -3.5], dtype=torch.bfloat16)
        self.assertTrue(torch.equal(C.spacing_distance(x, bump(x, 1)), torch.ones(2, dtype=torch.int64)))
        z = torch.tensor([0.0], dtype=torch.bfloat16)
        nz = torch.tensor([-0.0], dtype=torch.bfloat16)
        self.assertEqual(int(C.spacing_distance(z, nz).item()), 0)          # +0 and -0 are one point
        self.assertEqual(C.bitwise_diff_count(z, nz), 1)                   # but not bitwise equal

    def test_smallest_subnormals_across_zero(self):
        pos = torch.tensor([1], dtype=torch.int16).view(torch.bfloat16)
        neg = torch.tensor([-32767], dtype=torch.int16).view(torch.bfloat16)  # sign bit set, magnitude 1
        self.assertEqual(int(C.spacing_distance(pos, neg).item()), 2)

    def test_bitwise_count(self):
        a = torch.randn(1000).bfloat16()
        b = a.clone()
        b[::10] = bump(b[::10], 1)
        self.assertEqual(C.bitwise_diff_count(a, b), 100)

    def test_allclose_blind_to_one_spacing(self):
        a = torch.randn(20000).bfloat16()
        a = a[a.abs() > 1e-3]
        b = bump(a, 1)
        self.assertTrue(C.allclose_gate(b, a, rtol=2 ** -7))
        self.assertGreater(C.bitwise_diff_count(a, b), 0)
        self.assertFalse(C.allclose_gate(bump(a, 3), a, rtol=2 ** -7))     # three spacings is caught

    def test_margin_and_first_true(self):
        logits = torch.tensor([[3.0, 1.0, 2.0], [0.5, 0.4, 0.0]])
        self.assertTrue(torch.allclose(C.top2_margin(logits), torch.tensor([1.0, 0.1])))
        self.assertEqual(C.first_true(torch.tensor([False, False, True, True])), 2)
        self.assertIsNone(C.first_true(torch.tensor([False, False])))

    def test_kl_zero_for_identical(self):
        x = torch.randn(4, 50)
        self.assertLess(float(C.kl_ref_to_cand(x, x).abs().max()), 1e-6)
        self.assertGreater(float(C.kl_ref_to_cand(x, x + torch.randn(4, 50)).min()), 0.0)


class TestFaults(unittest.TestCase):
    def test_truncation_vs_rounding(self):
        x = torch.tensor([1.0 + 3 * 2.0 ** -9], dtype=torch.float32)       # 0.75 of a bf16 spacing above 1
        self.assertEqual(float(x.to(torch.bfloat16).float()), 1.0 + 2.0 ** -7)   # round to nearest goes up
        self.assertEqual(float(G.truncate_to_bf16(x).float()), 1.0)              # truncation goes down

    def test_f0_int32_matches_float64_path(self):
        aq, wq, sa, sw = G.make_problem(16, 64, 32, seed=1)
        self.assertTrue(torch.equal(G.acc_exact(aq, wq), G.acc_int32(aq, wq)))

    def test_faults_within_one_spacing(self):
        aq, wq, sa, sw = G.make_problem(64, 256, 128, seed=3)
        acc = G.acc_exact(aq, wq)
        ref = G.epilogue(acc, sa, sw, "F0")
        for mode in ("F1", "F2", "F3"):
            s = C.spacing_stats(ref, G.epilogue(acc, sa, sw, mode))
            self.assertLessEqual(s["max_spacing"], 1, mode)
        self.assertGreater(C.spacing_stats(ref, G.epilogue(acc, sa, sw, "F2"))["n_diff"], 0)


class TestEstimator(unittest.TestCase):
    def test_stratify_edges(self):
        idx = E.stratify(np.array([0.0, 0.049, 0.05, 0.2, 5.0, 1e9]))
        self.assertEqual(idx.tolist(), [0, 0, 1, 2, 6, 6])

    def test_allocate_sums_and_caps(self):
        sizes = np.array([10, 50, 200, 700, 0, 40, 5])
        a = E.allocate(sizes, E.WEIGHTS, 250)
        self.assertEqual(int(a.sum()), 250)
        self.assertTrue((a <= sizes).all())
        self.assertEqual(int(E.allocate(sizes, E.WEIGHTS, 10 ** 6).sum()), int(sizes.sum()))

    def test_coverage_on_synthetic_data(self):
        rng = np.random.default_rng(0)
        margins = rng.exponential(0.8, size=6000)
        flips = rng.random(6000) < 0.6 * np.exp(-margins * 15)
        r = E.coverage_experiment(margins, flips, budget_frac=0.25, n_rep=200, seed=1)
        self.assertGreater(r["n_flips"], 30)
        self.assertGreaterEqual(r["coverage_stratified"], 0.88)
        self.assertGreaterEqual(r["coverage_srs_wilson"], 0.88)

    def test_upper_bound_is_safe_when_high_margin_strata_are_exactly_zero(self):
        # Flips are only possible below margin 0.3, as in the pilot. Finding from E-23: the
        # two-sided stratified intervals undercover on the LOWER side here (about 80%, both
        # the rate-based Jeffreys interval and amendment A1), because unsampled zero-flip strata
        # add phantom flips. The one-sided 95% UPPER bound, which a qualification decision uses,
        # does not miss. This test pins the property that holds.
        rng = np.random.default_rng(3)
        margins = rng.exponential(1.0, size=3000)
        flips = (margins < 0.3) & (rng.random(3000) < 0.12)
        for interval in ("jeffreys_rate", "beta_binomial"):
            r = E.coverage_experiment(margins, flips, budget_frac=0.25, n_rep=200, seed=5,
                                      interval=interval)
            self.assertGreater(r["n_flips"], 15)
            self.assertLessEqual(r["one_sided_95_ucb_miss_stratified"], 0.05, interval)

    def test_wilson_sane(self):
        p, lo, hi = E.wilson_interval(5, 100)
        self.assertLess(lo, p)
        self.assertGreater(hi, p)


if __name__ == "__main__":
    unittest.main()
