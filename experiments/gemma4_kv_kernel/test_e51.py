"""Tests for E-51 with amendment A1 (no GPU, no Triton): the PyTorch helpers of `compact_attn.py` (RoPE against the
E-49 numpy code, the K=V identity of the compact layout, the paged gather and the split and merge rule of the
kernels), the Triton kernel source and the GPU code paths on a CPU emulation (`cpu_dryrun_e51.py`), every decision
path on stub data, the R-69 fixes (R1 to R6, O3), and a stub session from a clean copy of the upload folders.

Run: uv run --no-project --with numpy --with torch --with pytest python -m pytest test_e51.py
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_audit"))

import analyze_e51 as an  # noqa: E402
import compact_attn as ca  # noqa: E402

UPLOAD = ["gemma4_kv_kernel", "runpod", "fp8_quant_fusion", "nvfp4_dispatch", "safedispatch_testA",
          "safedispatch_testA2", "ledger"]   # README Section 11: the folders the Pod needs
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"
KB = [sys.executable, str(HERE / "kbench_e51.py"), "--stub"]


def test_rope_matches_e49() -> None:
    import kv_layout_check as kv

    x = np.random.default_rng(0).standard_normal((5, 3, 512))
    pos = np.array([0.0, 7.0, 1000.0, 32767.0, 65535.0])[:, None].repeat(3, 1)
    got = ca.rope(torch.tensor(x), torch.tensor(pos)).numpy()
    assert np.abs(got - kv.rope(x, pos)).max() < 1e-9


def test_compact_identity() -> None:
    """The compact row [V | K_rot] with the absorbed query gives the standard attention (float64)."""
    gen = torch.Generator().manual_seed(1)
    w = 1 + 0.1 * torch.randn(512, generator=gen, dtype=torch.float64)
    k, v = ca.make_kv(700, 4, w, gen, torch.float64)
    q = ca.rope(torch.randn(32, 512, generator=gen, dtype=torch.float64) / 512 ** 0.5,
                torch.full((32,), 699.0, dtype=torch.float64))
    qc, qr = ca.compact_query(q, w)
    std = ca.gqa_attend(q, k, v)
    cmp_ = ca.gqa_attend_compact(qc, qr, ca.pack_compact(k, v))
    assert float((std - cmp_).norm() / std.norm()) < 1e-12
    assert ca.pack_compact(k, v).shape[-1] == 640 and len(ca.ROT) == 128


def test_paged_reference_and_split_rule() -> None:
    """The paged gather equals the dense reference; the kernels' split rule (with empty splits) and merge are exact."""
    gen = torch.Generator().manual_seed(2)
    w = torch.ones(512, dtype=torch.float64)
    k, v = ca.make_kv(3000, 4, w, gen, torch.float64)
    q = torch.randn(32, 512, generator=gen, dtype=torch.float64) / 512 ** 0.5
    nblk = (3000 + 63) // 64
    bt = torch.randperm(nblk, generator=gen).view(1, nblk)
    kp = torch.zeros(nblk, 64, 4, 512, dtype=torch.float64)
    vp = torch.zeros_like(kp)
    pad = nblk * 64 - 3000
    kp[bt[0]] = torch.cat([k, torch.zeros(pad, 4, 512, dtype=torch.float64)]).view(nblk, 64, 4, 512)
    vp[bt[0]] = torch.cat([v, torch.zeros(pad, 4, 512, dtype=torch.float64)]).view(nblk, 64, 4, 512)
    dense = ca.gqa_attend(q, k, v)
    paged = ca.paged_reference(q[None], kp, vp, bt, torch.tensor([3000]))[0]
    assert float((paged - dense).norm() / dense.norm()) < 1e-6        # float32 reference
    for splits in (1, 4, 8, 32):                                      # 47 blocks: splits 32 leaves empty splits
        out = ca.split_reference(q, k, v, splits, 64)
        assert float((out - dense).norm() / dense.norm()) < 1e-12, splits


def test_kernels_and_gpu_paths_on_the_cpu_emulation() -> None:
    """The Triton source (both layouts, FP8 and BF16, every split and tile, partial pages, empty splits) against the
    references; the engine-layout views; the K = V oracle; bench_cell and confirm (A1)."""
    r = subprocess.run([sys.executable, str(HERE / "cpu_dryrun_e51.py")], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["worst_kernel_err"] < 0.005                            # A1 tolerance 0.01; R-69 saw 0.0021
    for case in out["cases"].values():
        for k, v in case.items():
            if "triton_" in k:
                assert v["all_float"] and v["n"] in (8, 2), (k, v)
            elif "stock_" in k:
                assert v == "error", k                                # no vLLM on the Mac
        assert abs(case["fp8 exact_err"][0] - case["fp8 exact_err"][1]) < 0.005   # both layouts store alike
    assert all(e < 0.005 for e in out["layout"].values()), out["layout"]
    assert out["info_keys"] == ["regs", "shared", "spills"]
    b = out["bench"]
    assert b["triton_ok"] == b["triton"] == 16 and b["stock_error"] == b["stock"] == 4 and b["has_regs_key"]
    assert all(v == 5 for k, v in out["confirm"].items() if k.startswith("triton")), out["confirm"]
    assert all(v == "error" for k, v in out["confirm"].items() if k.startswith("stock")), out["confirm"]


def stub_run(res: Path, *extra: str, copy: float | None = 3045.0, check: bool = True) -> None:
    kb = res / "kbench"
    kb.mkdir(parents=True, exist_ok=True)
    subprocess.run(KB + ["--out", str(kb), *extra], check=check, capture_output=True)
    env(res, copy)


def env(res: Path, copy: float | None = 3045.0) -> None:
    e = res / "env-1"
    e.mkdir(exist_ok=True)
    (e / "telemetry.csv").write_text(TEL + "".join("2026/11/01 10:00:00.000, 600, 1980, 60, 99, 0x0\n" for _ in range(150)))
    if copy is not None:
        (e / "microbench_start.json").write_text(json.dumps({"copy_gb_per_s_median": copy}))


def analyze(res: Path) -> dict:
    s = an.summarize(res)
    s["decision"] = an.decide(s)
    return s


def stub(res: Path, *extra: str, copy: float | None = 3045.0, check: bool = True) -> dict:
    stub_run(res, *extra, copy=copy, check=check)
    return analyze(res)


def edit(res: Path, fn) -> None:
    p = res / "kbench" / "kbench.json"
    d = json.loads(p.read_text())
    fn(d)
    p.write_text(json.dumps(d))


def test_go(tmp_path: Path) -> None:
    s = stub(tmp_path)
    assert s["decision"].startswith("GO (replicated in 5 of 5 repeats;"), s["decision"]
    g = s["fp8"]["32768"]
    assert 0.36 < g["sigma_layout"] < 0.39 and 0.25 < g["sigma_mma"] < 0.29 and g["std_memory_bound"]
    assert g["cmp_memory_bound"] and len(g["repeats"]) == 5 and g["range"]["sigma_mma"][0] > 0.25
    assert 0.75 < g["e_stock"] < 0.85 and 0.9 < g["e_triton"] < 1.1             # stub: stock gain 0.4, Triton memory-bound
    assert all(s["checks"][k] for k in ("V1", "V2", "V3", "V4", "V5", "V6"))
    assert s["fp8"]["32768"]["stock"]["kernel"] == "stock_fp8_auto"              # the stub's auto kernel is faster at X,
    assert s["checks"]["_decision_rows"] == 16                                     # so s32 adds 2 rows per group (V3)
    assert abs(s["checks"]["_engine_ratio"]["dt 19:32768"] - 1) < 0.01
    assert s["bf16"]["65536"]["ok"] and s["batch4_us"]["triton_cmp_fp8"]["65536"]
    assert set(s["common_config"]) == {"32768", "65536"} and abs(s["linearity"]["triton_cmp 11:32768"]) < 0.02


def test_kill_needs_a_memory_bound_compact_kernel(tmp_path: Path) -> None:
    """A1 (R2): a slow compact kernel cannot give KILL; a competent one (>= 0.6 x copy) with no layout gain can."""
    s = stub(tmp_path / "defect", "--stub-cmp-eff", "0.4375")
    assert s["decision"].startswith("INCONCLUSIVE (replicated") and "defect of the compact kernel" in s["decision"]
    assert abs(s["fp8"]["65536"]["sigma_layout"]) < 0.02
    s = stub(tmp_path / "kill", "--stub-std-eff", "0.95", "--stub-cmp-eff", "0.62")
    assert s["decision"].startswith("KILL (replicated in 5 of 5"), s["decision"]
    assert s["fp8"]["32768"]["sigma_layout"] < 0.155 and s["fp8"]["32768"]["cmp_memory_bound"]


def test_pivot_paths(tmp_path: Path) -> None:
    d = stub(tmp_path / "quality", "--stub-std-eff", "0.55", "--stub-cmp-eff", "0.55")["decision"]
    assert d.startswith("PIVOT") and "kernel quality" in d
    d = stub(tmp_path / "bar", "--stub-cmp-eff", "0.547")["decision"]
    assert d.startswith("PIVOT") and "below the bar" in d


def test_invalid_and_inconclusive_paths(tmp_path: Path) -> None:
    d = stub(tmp_path / "engine", "--stub-engine-ratio", "1.3")["decision"]
    assert d.startswith("INVALID") and "V3" in d
    d = stub(tmp_path / "fast", "--stub-cmp-eff", "1.3")["decision"]
    assert d.startswith("INVALID") and "V5" in d
    assert stub(tmp_path / "nocopy", copy=None)["decision"].startswith("INVALID (no copy bandwidth")
    assert stub(tmp_path / "fail", "--stub-fail", check=False)["decision"].startswith("INCONCLUSIVE (the benchmark")
    d = stub(tmp_path / "slow", "--stub-std-eff", "0.3", "--stub-cmp-eff", "0.3")["decision"]
    assert d.startswith("INCONCLUSIVE") and "standard kernel reads below" in d
    d = stub(tmp_path / "groups", "--cells", "19:2048,19:32768,11:2048")["decision"]     # no confirmation phase
    assert d.startswith("INVALID") or d.startswith("INCONCLUSIVE"), d


def test_one_configuration_at_both_lengths() -> None:
    """A1 (R1): dt takes the configuration that is fastest at X, timed at 2K and X, not the best row at each end."""
    def row(k, L, us, cfg):
        return {"kernel": k, "batch": 19, "context": L, "config": cfg, "us": us, "status": "ok"}

    s32, auto, a, b = {"num_splits": 32}, {"num_splits": 0}, {"splits": 32, "warps": 8, "tile": 32}, \
        {"splits": 8, "warps": 8, "tile": 32}
    rows = [row("stock_fp8_s32", 2048, 238.3, s32), row("stock_fp8_s32", 32768, 1358.1, s32),
            row("stock_fp8_auto", 2048, 150.3, auto), row("stock_fp8_auto", 32768, 2100.0, auto),
            row("triton_cmp_fp8", 2048, 150.0, a), row("triton_cmp_fp8", 2048, 110.0, b),
            row("triton_cmp_fp8", 32768, 850.0, a), row("triton_cmp_fp8", 32768, 990.0, b)]
    assert an.family(rows, an.FAMILIES["stock"], 32768, None)["dt_us"] == 1119.8      # the old rule: 1,207.8
    f = an.family(rows, an.FAMILIES["triton_cmp"], 32768, None)
    assert f["dt_us"] == 700.0 and f["config"] == a                                   # the old rule: 740.0


def test_v4_uses_the_decision_rows_only(tmp_path: Path) -> None:
    """A1 (R3): a noisy row that does not decide leaves V4; a noisy decision row fails it."""
    stub_run(tmp_path)

    def noisy_sweep_row(d):
        rows = [r for r in d["cells"] if r["kernel"] == "triton_std_fp8" and r["batch"] == 19 and r["context"] == 2048]
        max(rows, key=lambda r: r["us"])["spread"] = 0.06

    edit(tmp_path, noisy_sweep_row)
    assert analyze(tmp_path)["decision"].startswith("GO (")
    edit(tmp_path, lambda d: next(e for e in d["confirm"] if e["kernel"] == "triton_cmp_fp8").update(spreads=[0.06] * 5))
    assert analyze(tmp_path)["decision"] == "INVALID (V4)"


def test_wrong_configuration_is_dropped(tmp_path: Path) -> None:
    """A1 (R5): the fastest compact configuration fails the correctness cell, so it is never selected."""
    s = stub(tmp_path, "--stub-bad-config", "8/8/32")
    assert s["decision"].startswith("GO (") and s["dropped_rows"] == 10
    assert all(s["fp8"][X]["triton_cmp"]["config"] != {"splits": 8, "warps": 8, "tile": 32} for X in ("32768", "65536"))
    assert s["fp8"]["32768"]["triton_std"]["config"] == {"splits": 8, "warps": 8, "tile": 32}


def test_fail_fast(tmp_path: Path) -> None:
    """A1 (R6): a deciding kernel that fails the correctness cell stops the run; without fail fast, V2 fails."""
    s = stub(tmp_path / "on", "--stub-bad-correct", check=False)
    assert s["decision"] == "INVALID (V2: fail fast, ['triton_cmp_fp8'] did not pass the correctness cell)"
    assert not json.loads((tmp_path / "on" / "kbench" / "kbench.json").read_text())["cells"]
    assert stub(tmp_path / "off", "--stub-bad-correct", "--fail-fast", "off")["decision"] == "INVALID (V2)"


def test_timeout_after_the_groups_keeps_the_decision(tmp_path: Path) -> None:
    """A1 (R4): SIGTERM after the group cells and the confirmation leaves run.json at ok_groups, and it decides."""
    kb = tmp_path / "kbench"
    kb.mkdir()
    p = subprocess.Popen(KB + ["--out", str(kb), "--stub-hang-after-groups", "60"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    run = kb / "run.json"
    for _ in range(300):
        if run.exists() and json.loads(run.read_text()).get("status") == "ok_groups":
            break
        time.sleep(0.1)
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=30) != 0
    r = json.loads(run.read_text())
    assert r["status"] == "ok_groups" and "Timeout" in r["late_error"]
    env(tmp_path)
    s = analyze(tmp_path)
    assert s["decision"].startswith("GO (replicated"), s["decision"]
    assert s["batch4_us"]["triton_cmp_fp8"]["65536"] is None                      # the reported cells did not run


def test_decision_must_replicate(tmp_path: Path) -> None:
    """A1 (O3): GO on the medians but in only 3 of 5 repeats gives INCONCLUSIVE."""
    stub_run(tmp_path)

    def slow_two_repeats(d):
        for e in d["confirm"]:
            if e["kernel"] == "triton_cmp_fp8" and e["context"] > 2048:
                e["repeats"][0] *= 1.15
                e["repeats"][1] *= 1.15

    edit(tmp_path, slow_two_repeats)
    d = analyze(tmp_path)["decision"]
    assert d.startswith("INCONCLUSIVE (the decision does not replicate: GO on the medians, ['PIVOT', 'PIVOT', 'GO'"), d


def clean_copy(tmp_path: Path) -> Path:
    code = tmp_path / "code"
    for sub in UPLOAD:
        shutil.copytree(HERE.parent / sub, code / sub, ignore=shutil.ignore_patterns("results", "dryrun", "__pycache__", "*.gz"))
    return code


def session(code: Path, res: Path, **extra: str) -> subprocess.CompletedProcess:
    e = dict(os.environ, STUB="1", RES=str(res), CODE=str(code), AUTO_STOP="0", **extra)
    return subprocess.run([shutil.which("sh"), str(code / "gemma4_kv_kernel" / "session_e51.sh")], env=e,
                          capture_output=True, text=True, timeout=300)


def test_stub_session(tmp_path: Path) -> None:
    code, res = clean_copy(tmp_path), tmp_path / "e51"
    r = session(code, res)
    assert r.returncode == 0, r.stderr[-2000:]
    log = (res / "session.log").read_text()
    assert "kbench: exit 0" in log and "decision: GO (replicated in 5 of 5" in log
    session(code, res)
    assert "kbench: ok before, skip" in (res / "session.log").read_text()
    session(code, tmp_path / "failed", STUB_ARGS="--stub-fail")
    assert "decision: INCONCLUSIVE" in (tmp_path / "failed" / "session.log").read_text()
