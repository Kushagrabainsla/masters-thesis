"""Tests for analyze_e39.py (amendment A1) on synthetic runs. Run: python3 -m pytest test_e39.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_e39 as an

GOOD = "2026/10/27 10:00:{:02d}.000, 600, 1980, 60, 99, 0x0000000000000000"


def make(res: Path, gain: float, noise: float = 0.01, rounds: int = 10, aot: bool = True, telemetry: str | None = None) -> None:
    for i in range(1, rounds + 1):
        f = 1 + noise * ((i % 3) - 1)
        for side, base in (("R", 1000.0), ("C", 1000.0 * gain * f)):
            d = res / f"{side}_{i}"
            d.mkdir(parents=True)
            decode = {bs: {"seconds": [int(bs) * 128 / base * k for k in (1.0, 1.001, 0.999, 1.0, 1.0)]} for bs in an.SIZES}
            (d / "run.json").write_text(json.dumps({"status": "ok", "args": {"throughput_max_tokens": 128},
                                                    "throughput": {"decode": decode}}))
            (d / "log.txt").write_text("INFO Directly load AOT compilation\n" if aot else "")
    env = res / "env-1"
    env.mkdir()
    header = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"
    (env / "telemetry.csv").write_text(telemetry if telemetry is not None else header + "\n".join(GOOD.format(s % 60) for s in range(200)) + "\n")


def test_clear_gain_is_go(tmp_path: Path) -> None:
    make(tmp_path, 1.07)
    assert an.summarize(tmp_path)["decision"].startswith("GO")


def test_no_gain_is_kill(tmp_path: Path) -> None:
    make(tmp_path, 0.98)
    assert an.summarize(tmp_path)["decision"] == "KILL"


def test_small_gain_is_inconclusive(tmp_path: Path) -> None:
    make(tmp_path, 1.03, noise=0.02)
    assert an.summarize(tmp_path)["decision"].startswith("INCONCLUSIVE")


def test_reference_noise_is_reported_not_invalid(tmp_path: Path) -> None:
    make(tmp_path, 1.07, noise=0.005)
    out = an.summarize(tmp_path)
    assert "R_cv_reported" in out["validity"] and not out["decision"].startswith("INVALID")


def test_cold_run_is_invalid(tmp_path: Path) -> None:
    make(tmp_path, 1.07, aot=False)
    assert "V2" in an.summarize(tmp_path)["decision"]


def test_unparsed_telemetry_is_invalid(tmp_path: Path) -> None:
    make(tmp_path, 1.07, telemetry='Field "clocks_event_reasons.active" is not a valid field to query.\n')
    assert "V3" in an.summarize(tmp_path)["decision"]


def test_too_few_rounds_is_invalid(tmp_path: Path) -> None:
    make(tmp_path, 1.07, rounds=4)
    assert "V2" in an.summarize(tmp_path)["decision"]
