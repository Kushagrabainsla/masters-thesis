"""Tests for analyze_e44.py on synthetic data. Run: python3 -m pytest test_e44.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_e44 as an

SIZES = ["1", "8", "64", "256"]
TEL = "timestamp, power.draw [W], clocks.current.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"


def tp_run(d: Path, tps: float, f: float = 1.0, ok: bool = True) -> None:
    d.mkdir(parents=True, exist_ok=True)
    decode = {bs: {"seconds": [int(bs) * 128 / (tps * f)] * 5} for bs in SIZES}
    (d / "run.json").write_text(json.dumps({"status": "ok" if ok else "error", "args": {"throughput_max_tokens": 128},
                                            "throughput": {"decode": decode}}))


def prof(d: Path, selected: str, fusions: str, fused: bool) -> None:
    tp_run(d, 1000)
    names = ["void vllm::rms_norm_per_block_quant_kernel<x>"] if fused else ["triton_red_fused_add_rms_norm_0"]
    (d / "kernels.json").write_text(json.dumps({"tokens": 32, "by_batch": {"8": {"total_device_us": 10,
                                    "kernels": [{"name": n, "count": 1, "device_us": 1, "share": 0.1} for n in names]}}}))
    (d / "log.txt").write_text(f"INFO [compilation.py:317] Enabled custom fusions: {fusions}\n"
                               f"INFO Selected {selected} for Fp8LinearMethod\n"
                               "DEBUG fusion pass matches: {'RMSNormQuantFusionPass': 72, 'ActivationQuantFusionPass': 36}\n")


def res(tmp: Path, gain: float, kb_ratio: float = 1.6, d2_fused: bool = False, rounds: int = 6) -> Path:
    prof(tmp / "prof" / "D0", "FlashInferFp8DeepGEMMDynamicBlockScaledKernel", "norm_quant, act_quant", False)
    prof(tmp / "prof" / "D1", "DeepGemmFp8BlockScaledMMKernel", "norm_quant, act_quant", True)
    prof(tmp / "prof" / "D2", "DeepGemmFp8BlockScaledMMKernel", "", d2_fused)
    for i in range(1, rounds + 1):
        f = 1 + 0.004 * ((i % 3) - 1)
        tp_run(tmp / "paired" / f"D1_{i}", 1000)
        tp_run(tmp / "paired" / f"D2_{i}", 1000 * gain, f)
    kb = tmp / "kbench"
    kb.mkdir(parents=True, exist_ok=True)
    (kb / "kbench.json").write_text(json.dumps({"by_tokens": {str(m): {"norm": {"fused_over_unfused": kb_ratio},
                                                                         "act": {"fused_over_unfused": 1.0}}
                                                              for m in (1, 8, 64, 256, 512)}}))
    env = tmp / "env-1"
    env.mkdir(exist_ok=True)
    (env / "telemetry.csv").write_text(TEL + "\n".join("2026/10/28 10:00:00.000, 600, 1980, 60, 99, 0x0" for _ in range(150)) + "\n")
    return tmp


def decision(r: Path) -> str:
    return an.decide(an.summarize(r))


def test_go_kill_inconclusive(tmp_path: Path) -> None:
    assert decision(res(tmp_path / "a", 1.06)) == "GO"
    assert decision(res(tmp_path / "b", 0.99)) == "KILL"
    assert decision(res(tmp_path / "c", 1.06, kb_ratio=1.1)).startswith("INCONCLUSIVE")


def test_validity(tmp_path: Path) -> None:
    assert "V0" in decision(res(tmp_path / "a", 1.06, d2_fused=True))
    assert "V2" in decision(res(tmp_path / "b", 1.06, rounds=4))


def test_match_table_and_configs(tmp_path: Path) -> None:
    s = an.summarize(res(tmp_path, 1.06))
    assert s["configs"]["D1"]["match_table"] == {"RMSNormQuantFusionPass": 72, "ActivationQuantFusionPass": 36}
    assert s["configs"]["D1"]["fused_kernels"] == 1 and s["configs"]["D2"]["fused_kernels"] == 0
