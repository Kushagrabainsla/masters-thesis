"""Tests for E-51 (no GPU, no Triton): the PyTorch helpers of `compact_attn.py` (RoPE against the E-49 numpy code,
the K=V identity of the compact layout, the paged gather and the split and merge rule of the kernels), every
decision path on stub data, and a stub session from a clean copy of the upload folders.

Run: uv run --no-project --with numpy --with torch --with pytest python -m pytest test_e51.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
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


def stub(res: Path, *extra: str, copy: float | None = 3045.0, check: bool = True) -> dict:
    kb = res / "kbench"
    kb.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(HERE / "kbench_e51.py"), "--stub", "--out", str(kb), *extra],
                   check=check, capture_output=True)
    e = res / "env-1"
    e.mkdir(exist_ok=True)
    (e / "telemetry.csv").write_text(TEL + "".join("2026/11/01 10:00:00.000, 600, 1980, 60, 99, 0x0\n" for _ in range(150)))
    if copy is not None:
        (e / "microbench_start.json").write_text(json.dumps({"copy_gb_per_s_median": copy}))
    s = an.summarize(res)
    s["decision"] = an.decide(s)
    return s


def test_go(tmp_path: Path) -> None:
    s = stub(tmp_path)
    assert s["decision"].startswith("GO ("), s["decision"]
    g = s["fp8"]["32768"]
    assert 0.36 < g["sigma_layout"] < 0.39 and 0.25 < g["sigma_mma"] < 0.29 and g["std_memory_bound"]
    assert all(s["checks"][k] for k in ("V1", "V2", "V3", "V4", "V5", "V6"))
    assert s["bf16"]["65536"]["ok"] and s["batch4_us"]["triton_cmp_fp8"]["65536"]


def test_kill_when_the_layout_gives_nothing(tmp_path: Path) -> None:
    """The compact kernel reads 37.5% fewer bytes but at 0.625 of the efficiency: no layout effect."""
    s = stub(tmp_path, "--stub-cmp-eff", "0.4375")
    assert s["decision"].startswith("KILL"), s["decision"]
    assert abs(s["fp8"]["65536"]["sigma_layout"]) < 0.02


def test_pivot_paths(tmp_path: Path) -> None:
    d = stub(tmp_path / "quality", "--stub-std-eff", "0.55", "--stub-cmp-eff", "0.55")["decision"]
    assert d.startswith("PIVOT") and "kernel quality" in d
    d = stub(tmp_path / "bar", "--stub-cmp-eff", "0.547")["decision"]
    assert d.startswith("PIVOT") and "below the bar" in d


def test_invalid_and_inconclusive_paths(tmp_path: Path) -> None:
    assert "V2" in stub(tmp_path / "corr", "--stub-bad-correct")["decision"]
    d = stub(tmp_path / "engine", "--stub-engine-ratio", "1.3")["decision"]
    assert d.startswith("INVALID") and "V3" in d
    d = stub(tmp_path / "fast", "--stub-cmp-eff", "1.3")["decision"]
    assert d.startswith("INVALID") and "V5" in d
    assert stub(tmp_path / "nocopy", copy=None)["decision"].startswith("INVALID (no copy bandwidth")
    assert stub(tmp_path / "fail", "--stub-fail", check=False)["decision"].startswith("INCONCLUSIVE (the benchmark")
    d = stub(tmp_path / "slow", "--stub-std-eff", "0.3", "--stub-cmp-eff", "0.3")["decision"]
    assert d.startswith("INCONCLUSIVE (the Triton standard kernel reads below")


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
    assert "kbench: exit 0" in log and "decision: GO (" in log
    session(code, res)
    assert "kbench: ok before, skip" in (res / "session.log").read_text()
    session(code, tmp_path / "failed", STUB_ARGS="--stub-fail")
    assert "decision: INCONCLUSIVE" in (tmp_path / "failed" / "session.log").read_text()
