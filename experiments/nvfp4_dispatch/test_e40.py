"""Tests for analyze_e40.py on synthetic data. Run: python3 -m pytest test_e40.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_e40 as an

SIZES = ["1", "2", "4", "8", "16", "32", "64", "128", "256"]
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"


def tp_run(d: Path, tps: float, kernel: str = "KernelR", f: float = 1.0, aot: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    decode = {bs: {"seconds": [int(bs) * 128 / (tps * f)] * 5} for bs in SIZES}
    (d / "run.json").write_text(json.dumps({"status": "ok", "args": {"throughput_max_tokens": 128},
                                            "throughput": {"decode": decode}}))
    (d / "log.txt").write_text(f"INFO Selected KernelL for NvFp4LinearMethod\nINFO Using '{kernel}' NvFp4 MoE backend out of potential backends: [].\n"
                               + ("INFO Directly load AOT\n" if aot else ""))


def acc_run(d: Path, correct: list[int]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "completions.jsonl").write_text("\n".join(json.dumps({"id": f"q{i}", "correct": bool(c), "token_ids": [i]})
                                                    for i, c in enumerate(correct)) + "\n")
    (d / "run.json").write_text(json.dumps({"status": "ok"}))


def cell(tmp: Path, gains: dict[str, float], paired_gain: float | None, acc_ok: bool = True) -> Path:
    c = tmp / "cell_M"
    tp_run(c / "screen" / "R_1", 1000)
    tp_run(c / "screen" / "R_2", 1000)
    for v, g in gains.items():
        tp_run(c / "screen" / f"cand_{v}", 1000 * g, kernel="KernelR" if v == "alias" else f"Kernel{v}")
    base = [1] * 900 + [0] * 100
    acc_run(c / "acc" / "R_a", base)
    for v in gains:
        acc_run(c / "acc" / f"cand_{v}", base if acc_ok else [1] * 850 + [0] * 150)
    if paired_gain is not None:
        for v in an.screen(c)["selected"]:
            for i in range(1, 9):
                f = 1 + 0.01 * ((i % 3) - 1)
                tp_run(c / "paired" / v / f"R_{i}", 1000)
                tp_run(c / "paired" / v / f"C_{i}", 1000 * paired_gain, kernel=f"Kernel{v}", f=f)
    env = tmp / "env-1"
    env.mkdir(exist_ok=True)
    (env / "telemetry.csv").write_text(TEL + "\n".join("2026/10/27 10:00:00.000, 600, 1900, 60, 99, 0x0000000000000000" for _ in range(150)) + "\n")
    return c


def summary(c: Path) -> str:
    s = an.screen(c)
    cells = {c.name: {"screen": s, "accept": an.accept(c, s["selected"]),
                      "paired": {v: an.paired(c / "paired" / v) for v in s["selected"] if (c / "paired" / v).is_dir()},
                      "V3": an.telemetry_ok(c)}}
    return an.decide(cells)


def test_screen_drops_aliases_and_keeps_the_best_two(tmp_path: Path) -> None:
    c = cell(tmp_path, {"a": 1.10, "b": 1.05, "c": 1.03, "alias": 1.20, "slow": 0.95}, None)
    assert an.screen(c)["selected"] == ["a"]  # A1: keep the best one


def test_go_needs_accuracy_and_a_resolved_gain(tmp_path: Path) -> None:
    assert summary(cell(tmp_path / "1", {"a": 1.10}, 1.08)).startswith("GO")
    assert summary(cell(tmp_path / "2", {"a": 1.10}, 1.08, acc_ok=False)) == "KILL"
    assert summary(cell(tmp_path / "3", {"a": 1.10}, 0.97)) == "KILL"


def test_nvfp4_moe_log_line_is_parsed(tmp_path: Path) -> None:
    d = tmp_path / "r"
    d.mkdir()
    (d / "log.txt").write_text("(MainProcess pid=1) INFO 10-27 [nvfp4.py:270] Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend out of potential backends: ['B12X'].\n")
    sig, _ = an.kernel_signature(d)
    assert sig == frozenset({"Using 'FLASHINFER_TRTLLM' NvFp4 MoE backend"})


def test_no_screen_pass_is_kill(tmp_path: Path) -> None:
    assert summary(cell(tmp_path, {"a": 1.01, "b": 0.9}, None)) == "KILL"


def test_missing_paired_speed_is_inconclusive(tmp_path: Path) -> None:
    assert summary(cell(tmp_path, {"a": 1.10}, None)).startswith("INCONCLUSIVE")
