"""Unit tests for compare.py (amendment A1) on synthetic cells. No vLLM needed."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import compare

REF_KERNEL = ["Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for Fp8BlockScaledMMLinearKernel"]


PREFIX = "(MainProcess pid=37) INFO 10-20 19:29:34 [__init__.py:759] "


def write_run(d: Path, outputs: dict | None, speeds: dict | None, status: str = "ok", kernel_log=None):
    d.mkdir(parents=True)
    run = {"status": status}
    (d / "log.txt").write_text("".join(PREFIX + line + "\n" for line in (kernel_log or [])))
    if speeds is not None:
        run["throughput"] = {"decode": {bs: {"median_tokens_per_s": v} for bs, v in speeds.items()}}
    (d / "run.json").write_text(json.dumps(run))
    if outputs is not None:
        with open(d / "outputs.jsonl", "w") as f:
            for i, toks in outputs.items():
                topk = [[[t, -0.1], [t + 1, -2.5]] for t in toks]
                f.write(json.dumps({"id": i, "token_ids": toks, "topk": topk}) + "\n")


def base(n):
    return {f"{'gsm8k' if i % 2 else 'ultrachat'}-{i}": [i, i + 1, i + 2, i + 3] for i in range(n)}


def flipped(outputs: dict, k: int) -> dict:
    out = dict(outputs)
    for key in list(outputs)[-k:] if k else []:
        out[key] = outputs[key][:2] + [777, 778]
    return out


def make_cell(root: Path, n: int, cands: dict, r_speed: dict, r_speed_b: dict | None = None,
              ref_noise: int = 0, pc_flips: int | None = None) -> Path:
    """cands: name -> (flips, speeds, kernel_log or None)."""
    b = base(n)
    write_run(root / "R_a", b, r_speed, kernel_log=REF_KERNEL)
    noisy = dict(b)
    for key in list(b)[:ref_noise]:
        noisy[key] = [999] + b[key][1:]
    write_run(root / "R_b", noisy, r_speed_b or r_speed, kernel_log=REF_KERNEL)
    for name, (k, sp, log) in cands.items():
        write_run(root / f"cand_{name}", flipped(b, k), sp,
                  kernel_log=log or [f"Selected {name.title()}Kernel for Fp8BlockScaledMMLinearKernel"])
    if pc_flips is not None:
        write_run(root / "cand_marlin", flipped(b, pc_flips), r_speed,
                  kernel_log=["Selected MarlinFP8ScaledMMLinearKernel for Fp8BlockScaledMMLinearKernel"])
    return root


SP = {"1": 100.0, "64": 2000.0}


class CompareTests(unittest.TestCase):
    def test_wilson(self):
        self.assertTrue(0.014 < compare.wilson(0, 256)[1] < 0.015)
        self.assertLessEqual(compare.wilson(0, 189)[1], 0.02)
        self.assertGreater(compare.wilson(0, 188)[1], 0.02)

    def test_first_divergence_and_kl(self):
        self.assertIsNone(compare.first_divergence([1, 2], [1, 2]))
        self.assertEqual(compare.first_divergence([1, 2, 3], [1, 5, 3]), 1)
        topk = [[[1, -0.1], [2, -2.4]]] * 4
        self.assertAlmostEqual(compare.approx_kl(topk, topk), 0.0, places=9)

    def test_stop_flat_with_positive_control(self):
        with tempfile.TemporaryDirectory() as t:
            cands = {"deep_gemm": (0, {"1": 100.5, "64": 2010.0}, None)}
            d = compare.summarize_cell(make_cell(Path(t) / "D", 256, cands, SP, pc_flips=60))
            m = compare.summarize_cell(make_cell(Path(t) / "M", 256, cands, SP))
            self.assertTrue(d["flat"] and m["flat"])
            self.assertEqual(compare.session_decision({"D": d, "M": m}), "STOP-FLAT")

    def test_flat_without_positive_control_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as t:
            cands = {"deep_gemm": (0, {"1": 100.5, "64": 2010.0}, None)}
            d = compare.summarize_cell(make_cell(Path(t) / "D", 256, cands, SP, pc_flips=2))
            self.assertIn("positive control", compare.session_decision({"D": d}))

    def test_alias_is_excluded(self):
        with tempfile.TemporaryDirectory() as t:
            cands = {"flashinfer_cutlass": (5, {"1": 150.0, "64": 3000.0}, REF_KERNEL),
                     "deep_gemm": (0, {"1": 100.0, "64": 2000.0}, None)}
            s = compare.summarize_cell(make_cell(Path(t) / "D", 256, cands, SP))
            self.assertIn("flashinfer_cutlass", s["aliases_negative_control"])
            self.assertNotIn("flashinfer_cutlass", s["speeds"])

    def test_fallback_is_alias(self):
        with tempfile.TemporaryDirectory() as t:
            log = ["--linear-backend=flashinfer_trtllm was requested, but no flashinfer_trtllm kernel exists "
                   "for X layers; falling back to normal kernel selection for this layer."]
            s = compare.summarize_cell(make_cell(Path(t) / "D", 64, {"flashinfer_trtllm": (0, SP, log)}, SP))
            self.assertEqual(s["aliases_negative_control"]["flashinfer_trtllm"]["reason"], "fell back")
            self.assertFalse(s["V2"])

    def test_continue_on_unsafe_tradeoff(self):
        with tempfile.TemporaryDirectory() as t:
            cands = {"triton": (64, {"1": 99.0, "64": 2600.0}, None),
                     "deep_gemm": (0, {"1": 101.0, "64": 1900.0}, None)}
            s = compare.summarize_cell(make_cell(Path(t) / "M", 256, cands, SP))
            self.assertIn("triton", s["unsafe"])
            self.assertIn("64", s["unsafe_faster_than_safe"])
            self.assertTrue(s["continue"])
            self.assertEqual(compare.session_decision({"M": s}), "CONTINUE")

    def test_static_winner(self):
        with tempfile.TemporaryDirectory() as t:
            cands = {"deep_gemm": (0, {"1": 110.0, "64": 2200.0}, None),
                     "triton": (3, {"1": 90.0, "64": 1800.0}, None)}
            s = compare.summarize_cell(make_cell(Path(t) / "D", 256, cands, SP))
            self.assertTrue(s["static_winner"])
            self.assertFalse(s["flat"] or s["continue"])
            self.assertEqual(compare.session_decision({"D": s}), "STATIC-WINNER")

    def test_noise_raises_speed_threshold(self):
        with tempfile.TemporaryDirectory() as t:
            cands = {"deep_gemm": (0, {"1": 103.0, "64": 2060.0}, None)}
            s = compare.summarize_cell(make_cell(Path(t) / "D", 256, cands, SP,
                                                 r_speed_b={"1": 97.0, "64": 1940.0}, pc_flips=40))
            self.assertTrue(s["flat"])  # 3% spread is within 2 x the 6% reference noise

    def test_floor_excluded(self):
        with tempfile.TemporaryDirectory() as t:
            s = compare.summarize_cell(make_cell(Path(t) / "D", 100, {"deep_gemm": (0, SP, None)}, SP, ref_noise=3))
            self.assertEqual(s["floor_prompts"], 3)
            self.assertEqual(s["candidates"]["deep_gemm"]["n"], 97)
            self.assertTrue(math.isclose(s["reference_agreement"], 0.97))

    def test_kernel_signature_from_prefixed_log(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t) / "run"
            write_run(d, None, None, kernel_log=[
                "Selected DeepGemmFp8BlockScaledMMKernel for Fp8BlockScaledMMLinearKernel",
                "Using triton Fp8 MoE backend out of potential backends: ['triton', 'deep_gemm'].",
                "CPU backend doesn't allow to use `torch.set_num_threads` after the thread binding, skip it."])
            sig, fell_back = compare.kernel_signature(d)
            self.assertEqual(sig, frozenset({"Selected DeepGemmFp8BlockScaledMMKernel for Fp8BlockScaledMMLinearKernel",
                                             "Using triton Fp8 MoE backend"}))
            self.assertFalse(fell_back)

    def test_log_files_beside_runs_are_ignored(self):
        with tempfile.TemporaryDirectory() as t:
            root = make_cell(Path(t) / "D", 32, {"deep_gemm": (0, SP, None)}, SP)
            (root / "cand_deep_gemm.log").write_text("not a run")
            s = compare.summarize_cell(root)
            self.assertEqual(sorted(s["candidates"]), ["deep_gemm"])

    def test_missing_candidate_run_does_not_crash(self):
        with tempfile.TemporaryDirectory() as t:
            root = make_cell(Path(t) / "D", 32, {"deep_gemm": (0, SP, None)}, SP)
            (root / "cand_cutlass").mkdir()
            s = compare.summarize_cell(root)
            self.assertEqual(s["candidates"]["cutlass"]["status"], "missing")

    def test_kernel_log_fields(self):
        with tempfile.TemporaryDirectory() as t:
            s = compare.summarize_cell(make_cell(Path(t) / "D", 32, {"deep_gemm": (0, SP, None)}, SP))
            self.assertTrue(s["kernel_log_found"])
            self.assertEqual(s["reference_kernels"], REF_KERNEL)
            self.assertEqual(s["candidates"]["deep_gemm"]["kernels"],
                             ["Selected Deep_GemmKernel for Fp8BlockScaledMMLinearKernel"])
            # The MoE message from vLLM v0.32.0 fp8.py is captured without the backend list.
            moe = "Using TRITON Fp8 MoE backend out of potential backends: ['DEEPGEMM', 'TRITON']."
            write_run(Path(t) / "M" / "R_a", base(4), SP, kernel_log=[moe])
            self.assertEqual(compare.kernel_signature(Path(t) / "M" / "R_a")[0],
                             frozenset({"Using TRITON Fp8 MoE backend"}))
            # An empty reference log is reported, not hidden.
            write_run(Path(t) / "E" / "R_a", base(4), SP)
            write_run(Path(t) / "E" / "R_b", base(4), SP)
            self.assertFalse(compare.summarize_cell(Path(t) / "E")["kernel_log_found"])


if __name__ == "__main__":
    unittest.main()
