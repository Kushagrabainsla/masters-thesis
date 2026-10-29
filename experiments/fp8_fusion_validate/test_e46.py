"""Tests for analyze_e46.py on synthetic data. Run: python3 -m pytest test_e46.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_e46 as an

SIZES = ["1", "8", "64", "256"]
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"


def tp(d: Path, tps: float, f: float = 1.0) -> None:
    d.mkdir(parents=True, exist_ok=True)
    dec = {bs: {"seconds": [int(bs) * 128 / (tps * f)] * 5} for bs in SIZES}
    (d / "run.json").write_text(json.dumps({"status": "ok", "args": {"throughput_max_tokens": 128}, "throughput": {"decode": dec}}))


def acc(d: Path, correct: list[int]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "completions.jsonl").write_text("\n".join(json.dumps({"id": f"q{i}", "correct": bool(c), "token_ids": [i]})
                                                    for i, c in enumerate(correct)) + "\n")
    (d / "run.json").write_text(json.dumps({"status": "ok"}))


def prof(d: Path, fused: bool, sel: str = "DeepGemmFp8BlockScaledMMKernel") -> None:
    tp(d, 1000)
    n = "void vllm::rms_norm_per_block_quant_kernel<x>" if fused else "triton_red_fused_add_rms_norm_0"
    (d / "kernels.json").write_text(json.dumps({"by_batch": {"8": {"total_device_us": 1, "kernels": [{"name": n, "count": 1, "device_us": 1, "share": .1}]}}}))
    (d / "log.txt").write_text(f"INFO Selected {sel} for Fp8LinearMethod\n")


def res(tmp: Path, gains: dict, acc_ok: bool = True, off_fused: bool = False) -> Path:
    base = [1] * 900 + [0] * 100
    acc(tmp / "acc" / "on", base)
    acc(tmp / "acc" / "off", base if acc_ok else [1] * 850 + [0] * 150)
    prof(tmp / "prof" / "on", True)
    prof(tmp / "prof" / "off", off_fused)
    for p, g in gains.items():
        for i in range(1, 6):
            f = 1 + 0.003 * ((i % 3) - 1)
            tp(tmp / "paired" / p / f"X_{i}", 1000)
            tp(tmp / "paired" / p / f"Y_{i}", 1000 * g, f)
    env = tmp / "env-1"
    env.mkdir(exist_ok=True)
    (env / "telemetry.csv").write_text(TEL + "\n".join("2026/10/29 10:00:00.000, 600, 1980, 60, 99, 0x0" for _ in range(150)) + "\n")
    (tmp / "versions.txt").write_text("vllm 0.32.1.dev1\ntorch 2.11\n")
    return tmp


def decision(r: Path) -> str:
    return an.decide(an.summarize(r))


def test_labels(tmp_path: Path) -> None:
    assert decision(res(tmp_path / "a", {"P1": 1.04, "P2": 1.03, "P3": 1.00})).startswith("VALIDATED")
    assert decision(res(tmp_path / "b", {"P1": 0.99, "P2": 1.03, "P3": 1.03})).startswith("NOT REPRODUCED")
    assert decision(res(tmp_path / "c", {"P1": 1.04, "P2": 1.00, "P3": 1.00})).startswith("PARTLY VALIDATED")
    assert "accuracy FAIL" in decision(res(tmp_path / "d", {"P1": 1.04, "P2": 1.03, "P3": 1.03}, acc_ok=False))


def test_validity_and_missing_pair(tmp_path: Path) -> None:
    assert "V0" in decision(res(tmp_path / "a", {"P1": 1.04, "P2": 1.03, "P3": 1.03}, off_fused=True))
    s = an.summarize(res(tmp_path / "b", {"P1": 1.04, "P2": 1.03}))
    assert s["pairs"]["P3"]["label"] == "MISSING" and an.decide(s).startswith("VALIDATED")
