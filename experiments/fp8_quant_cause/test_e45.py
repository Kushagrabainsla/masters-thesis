"""Tests for analyze_e45.py on synthetic data. Run: python3 -m pytest test_e45.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_e45 as an

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


def prof(d: Path, quant: int, native: bool) -> None:
    tp(d, 1000)
    (d / "kernels.json").write_text(json.dumps({"by_batch": {"64": {"kernels": [
        {"name": "void per_token_group_quant_8bit_register_kernel<x>", "count": quant, "device_us": 1, "share": 0.1}]}}}))
    (d / "log.txt").write_text("compilation_config={'custom_ops': ['" + ("-quant_fp8" if native else "+quant_fp8") + "', 'none']}\n")


def res(tmp: Path, g1: float, g2: float = 1.03, acc_ok: bool = True, q1: int = 0, rounds: int = 5) -> Path:
    base = [1] * 900 + [0] * 100
    acc(tmp / "acc" / "A0", base)
    acc(tmp / "acc" / "A1", base if acc_ok else [1] * 850 + [0] * 150)
    prof(tmp / "prof" / "A0", 4608, False)
    prof(tmp / "prof" / "A1", q1, True)
    for p, g in (("P1", g1), ("P2", g2), ("P3", 1.05)):
        for i in range(1, rounds + 1):
            f = 1 + 0.003 * ((i % 3) - 1)
            tp(tmp / "paired" / p / f"X_{i}", 1000)
            tp(tmp / "paired" / p / f"Y_{i}", 1000 * g, f)
    env = tmp / "env-1"
    env.mkdir(exist_ok=True)
    (env / "telemetry.csv").write_text(TEL + "\n".join("2026/10/29 10:00:00.000, 600, 1980, 60, 99, 0x0" for _ in range(150)) + "\n")
    return tmp


def decision(r: Path) -> str:
    return an.decide(an.summarize(r))


def test_go_kill_inconclusive(tmp_path: Path) -> None:
    assert decision(res(tmp_path / "a", 1.05)).startswith("GO")
    assert decision(res(tmp_path / "b", 0.98)) == "KILL (native quant is not faster)"
    assert decision(res(tmp_path / "c", 1.05, acc_ok=False)) == "KILL (A1 fails the accuracy gate)"
    assert decision(res(tmp_path / "d", 1.05, g2=0.97)).startswith("INCONCLUSIVE")


def test_validity_and_p3(tmp_path: Path) -> None:
    assert "V0" in decision(res(tmp_path / "a", 1.05, q1=4608))
    assert "V2" in decision(res(tmp_path / "b", 1.05, rounds=3))
    s = an.summarize(res(tmp_path / "c", 1.05))
    assert an.p3_label(s) == "fusion slowdown holds on M14"
