"""Unit tests for extras.py (amendment A2) on synthetic data. Run: python3 -m pytest test_extras.py"""

from __future__ import annotations

import json
from pathlib import Path

import extras

HEADER = "timestamp, power.draw [W], clocks.sm [MHz], temperature.gpu, utilization.gpu [%], clocks_event_reasons.active\n"


def write_run(d: Path, start: str, end: str, uuid: str = "GPU-1", speeds: dict | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    run = {"status": "ok", "started_at": start, "finished_at": end, "gpu": {"uuid": uuid}}
    if speeds:
        run["throughput"] = {"decode": {bs: {"median_tokens_per_s": v} for bs, v in speeds.items()}}
    (d / "run.json").write_text(json.dumps(run))


def write_kernels(d: Path, shares: dict[str, float]) -> None:
    kernels = [{"name": n, "count": 1, "device_us": s * 100, "share": s} for n, s in shares.items()]
    (d / "kernels.json").write_text(json.dumps({"by_batch": {"64": {"total_device_us": 100, "kernels": kernels}}}))


def write_outputs(d: Path, tokens: dict[str, list[int]]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    rows = [json.dumps({"id": i, "token_ids": t, "topk": []}) for i, t in tokens.items()]
    (d / "outputs.jsonl").write_text("\n".join(rows) + "\n")


def test_telemetry_power_cap_is_not_suspect_but_thermal_is(tmp_path: Path) -> None:
    env = tmp_path / "env-1"
    env.mkdir()
    lines = [
        "2026/10/22 10:00:00.000, 100, 1980, 40, 0, 0x0000000000000001",  # idle: dropped
        "2026/10/22 10:00:01.000, 690, 1980, 60, 99, 0x0000000000000004",  # power cap only
        "2026/10/22 10:00:02.000, 700, 1980, 70, 99, 0x0000000000000004",
        "2026/10/22 10:00:11.000, 650, 1900, 88, 99, 0x0000000000000040",  # hardware thermal
        "2026/10/22 10:00:12.000, 650, 1900, 89, 99, 0x0000000000000000",
    ]
    (env / "telemetry.csv").write_text(HEADER + "\n".join(lines) + "\n")
    write_run(tmp_path / "cell_D" / "R_a", "2026-10-22T10:00:00Z", "2026-10-22T10:00:05Z")
    write_run(tmp_path / "cell_D" / "R_b", "2026-10-22T10:00:10Z", "2026-10-22T10:00:15Z")
    out = extras.summarize(tmp_path)
    a, b = out["runs"]["cell_D/R_a"], out["runs"]["cell_D/R_b"]
    assert a["telemetry"]["samples"] == 2 and a["telemetry"]["power_cap_fraction"] == 1.0
    assert a["suspect_speed"] is False
    assert b["telemetry"]["hard_slowdown_fraction"] == 0.5 and b["suspect_speed"] is True


def test_old_driver_field_name_is_read(tmp_path: Path) -> None:
    env = tmp_path / "env-1"
    env.mkdir()
    header = HEADER.replace("clocks_event_reasons", "clocks_throttle_reasons")
    (env / "telemetry.csv").write_text(header + "2026/10/22 10:00:01.000, 600, 1980, 60, 99, 0x0000000000000008\n")
    write_run(tmp_path / "cell_D" / "R_a", "2026-10-22T10:00:00Z", "2026-10-22T10:00:05Z")
    out = extras.summarize(tmp_path)
    assert out["runs"]["cell_D/R_a"]["telemetry"]["hard_slowdown_fraction"] == 1.0


def test_driver_595_sm_clock_name_is_read(tmp_path: Path) -> None:
    env = tmp_path / "env-1"
    env.mkdir()
    header = HEADER.replace("clocks.sm [MHz]", "clocks.current.sm [MHz]")
    (env / "telemetry.csv").write_text(header + "2026/10/22 10:00:01.000, 600, 1755, 60, 99, 0x0000000000000000\n")
    write_run(tmp_path / "cell_D" / "R_a", "2026-10-22T10:00:00Z", "2026-10-22T10:00:05Z")
    assert extras.summarize(tmp_path)["runs"]["cell_D/R_a"]["telemetry"]["median_sm_mhz"] == 1755


def test_telemetry_error_text_is_skipped(tmp_path: Path) -> None:
    env = tmp_path / "env-1"
    env.mkdir()
    (env / "telemetry.csv").write_text('Field "clocks_event_reasons.active" is not a valid field to query.\n')
    assert extras.summarize(tmp_path)["telemetry_samples"] == 0


def test_kernel_check_uses_gemm_kernels_only(tmp_path: Path) -> None:
    cell = tmp_path / "cell_D"
    for name in ("R_a", "cand_x", "cand_y"):
        write_run(cell / name, "2026-10-22T10:00:00Z", "2026-10-22T10:01:00Z")
    write_kernels(cell / "R_a", {"cutlass_fp8_gemm": 0.6, "flash_attn_fwd": 0.3, "elementwise_tail_7": 0.01})
    write_kernels(cell / "cand_x", {"cutlass_fp8_gemm": 0.58, "flash_attn_fwd": 0.3, "elementwise_tail_9": 0.01})
    write_kernels(cell / "cand_y", {"deep_gemm_fp8_kernel": 0.5, "flash_attn_fwd": 0.3})
    checks = extras.summarize(tmp_path)["kernel_check"]["cell_D"]
    assert checks["cand_x"]["same_gemm_kernels"] is True  # the non-GEMM tail differs, which is ignored
    assert checks["cand_y"]["same_gemm_kernels"] is False
    assert checks["cand_y"]["only_in_candidate"] == ["deep_gemm_fp8_kernel"]


def test_gpu_count(tmp_path: Path) -> None:
    write_run(tmp_path / "cell_D" / "R_a", "2026-10-22T10:00:00Z", "2026-10-22T10:01:00Z")
    write_run(tmp_path / "cell_D" / "cand_y", "2026-10-22T10:02:00Z", "2026-10-22T10:03:00Z", uuid="GPU-2")
    assert extras.summarize(tmp_path)["one_gpu"] is False


def test_b2_ranks_only_reference_and_candidates(tmp_path: Path) -> None:
    summary = {"cells": {"cell_D": {"candidates": {"x": {"status": "ok"}}, "aliases_negative_control": {},
                                    "w8a16_positive_control": {"marlin": {}}}}}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    b2 = tmp_path / "extra" / "b2_cell_D"
    write_run(b2 / "R_a", "2026-10-22T10:00:00Z", "2026-10-22T10:01:00Z", speeds={"1": 100.0, "64": 5000.0})
    write_run(b2 / "cand_x", "2026-10-22T10:02:00Z", "2026-10-22T10:03:00Z", speeds={"1": 120.0, "64": 4000.0})
    write_run(b2 / "cand_marlin", "2026-10-22T10:04:00Z", "2026-10-22T10:05:00Z", speeds={"1": 200.0, "64": 9000.0})
    (b2 / "R_a" / "log.txt").write_text("INFO [kv_cache_utils.py:1] GPU KV cache size: 301,056 tokens\n")
    out = extras.summarize(tmp_path)["phase_b"]["b2_cell_D"]
    assert out["1"]["fastest_eligible"] == "cand_x"
    assert out["64"]["fastest_eligible"] == "R_a"
    assert out["status"]["cand_marlin"] == "w8a16_control"
    assert out["kv_cache_tokens"] == 301056


def test_b0_repeat(tmp_path: Path) -> None:
    write_outputs(tmp_path / "cell_D" / "R_a", {"a": [1, 2], "b": [1, 2]})
    write_outputs(tmp_path / "cell_D" / "cand_x", {"a": [1, 3], "b": [1, 2]})
    write_outputs(tmp_path / "extra" / "b0_cell_D" / "cand_x", {"a": [1, 3], "b": [1, 4]})
    out = extras.summarize(tmp_path)["phase_b"]["b0_cell_D"]
    assert out["repeat_vs_first"]["flipped"] == 1
    assert out["repeat_vs_R_a"]["flipped"] == 2


def test_cold_vs_warm_compares_first_tokens(tmp_path: Path) -> None:
    write_outputs(tmp_path / "cell_D" / "warm_R", {"a": [1, 2], "b": [1, 9]})
    write_outputs(tmp_path / "cell_D" / "R_a", {"a": [1, 2, 3, 4], "b": [1, 2, 3, 4], "c": [5]})
    write_outputs(tmp_path / "cell_D" / "warm_x", {"a": [1, 2]})
    write_outputs(tmp_path / "cell_D" / "cand_x", {"a": [1, 2, 7]})
    (tmp_path / "cell_D" / "R_a" / "log.txt").write_text("INFO Directly load AOT compilation from path\n")
    out = extras.summarize(tmp_path)["cold_vs_warm"]
    assert out["cell_D/warm_R"]["agreement"] == 0.5 and out["cell_D/warm_R"]["measured_aot_hit"] is True
    assert out["cell_D/warm_x"]["agreement"] == 1.0 and out["cell_D/warm_x"]["measured_aot_hit"] is None


def test_warm_runs_are_never_suspect(tmp_path: Path) -> None:
    env = tmp_path / "env-1"
    env.mkdir()
    lines = ["2026/10/22 10:00:01.000, 70, 1500, 50, 99, 0x0000000000000000",
             "2026/10/22 10:00:11.000, 700, 2700, 60, 99, 0x0000000000000000"]
    (env / "telemetry.csv").write_text(HEADER + "\n".join(lines) + "\n")
    write_run(tmp_path / "cell_D" / "warm_R", "2026-10-22T10:00:00Z", "2026-10-22T10:00:05Z")
    write_run(tmp_path / "cell_D" / "R_a", "2026-10-22T10:00:10Z", "2026-10-22T10:00:15Z")
    out = extras.summarize(tmp_path)
    assert out["runs"]["cell_D/warm_R"]["suspect_speed"] is False
    assert out["session_median_sm_mhz"] == 2700


def test_missing_everything(tmp_path: Path) -> None:
    out = extras.summarize(tmp_path)
    assert out["telemetry_samples"] == 0 and out["one_gpu"] is True
