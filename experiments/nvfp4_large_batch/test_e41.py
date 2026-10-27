"""Tests for analyze_e41.py on synthetic data. Run: python3 -m pytest test_e41.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_e41 as an

SIZES = ["128", "256", "384", "512"]
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"
LINE = "(EngineCore pid=1) INFO 10-27 [nvfp4.py:88] Using {} for NVFP4 GEMM\n"


def tp_run(d: Path, tps: float, kernel: str, f: float = 1.0) -> None:
    d.mkdir(parents=True, exist_ok=True)
    decode = {bs: {"seconds": [int(bs) * 128 / (tps * f)] * 5} for bs in SIZES}
    (d / "run.json").write_text(json.dumps({"status": "ok", "args": {"throughput_max_tokens": 128},
                                            "throughput": {"decode": decode}}))
    (d / "log.txt").write_text(LINE.format(kernel))


def acc_run(d: Path, correct: list[int], kernel: str = "KernelR") -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "completions.jsonl").write_text("\n".join(json.dumps({"id": f"q{i}", "correct": bool(c), "token_ids": [i]})
                                                    for i, c in enumerate(correct)) + "\n")
    (d / "run.json").write_text(json.dumps({"status": "ok"}))
    (d / "log.txt").write_text(LINE.format(kernel))


def res(tmp: Path, gain: dict[str, float], acc_ok: bool = True, rounds: int = 8, alias: bool = False,
        r_line: bool = True) -> Path:
    base = [1] * 900 + [0] * 100
    acc_run(tmp / "acc" / "R_a", base, "KernelR" if r_line else "x")
    if not r_line:
        (tmp / "acc" / "R_a" / "log.txt").write_text("no line\n")
    for v in an.CANDS:
        acc_run(tmp / "acc" / f"cand_{v}", base if acc_ok else [1] * 850 + [0] * 150)
        for i in range(1, rounds + 1):
            f = 1 + 0.01 * ((i % 3) - 1)
            p = tmp / "paired" / v
            tp_run(p / f"R_{i}", 1000, "KernelR" if r_line else "x")
            tp_run(p / f"C_{i}", 1000 * gain.get(v, 1.0), "KernelR" if alias else f"K_{v}", f)
            if not r_line:
                (p / f"R_{i}" / "log.txt").write_text("no line\n")
    env = tmp / "env-1"
    env.mkdir(exist_ok=True)
    (env / "telemetry.csv").write_text(TEL + "\n".join("2026/10/27 10:00:00.000, 900, 1900, 60, 99, 0x0000000000000000" for _ in range(150)) + "\n")
    return tmp


def decision(r: Path) -> str:
    return an.decide(an.summarize(r))


def test_dense_line_is_parsed(tmp_path: Path) -> None:
    tp_run(tmp_path / "r", 1, "FlashInferCuteDslNvFp4LinearKernel")
    assert an.linear_kernel(tmp_path / "r") == "FlashInferCuteDslNvFp4LinearKernel"


def test_go_kill_and_inconclusive(tmp_path: Path) -> None:
    assert decision(res(tmp_path / "1", {"flashinfer_cudnn": 1.08})) == "GO (flashinfer_cudnn)"
    assert decision(res(tmp_path / "2", {})) == "KILL"
    assert decision(res(tmp_path / "3", {"flashinfer_cudnn": 1.08}, acc_ok=False)) == "KILL"
    assert decision(res(tmp_path / "4", {"flashinfer_cutlass": 1.03})).startswith("INCONCLUSIVE")


def test_validity(tmp_path: Path) -> None:
    assert decision(res(tmp_path / "1", {"flashinfer_cudnn": 1.08}, rounds=5)).startswith("INCONCLUSIVE")
    assert decision(res(tmp_path / "2", {"flashinfer_cudnn": 1.08}, alias=True)).startswith("INCONCLUSIVE")
    assert decision(res(tmp_path / "3", {"flashinfer_cudnn": 1.08}, r_line=False)) == "INVALID (V0)"
