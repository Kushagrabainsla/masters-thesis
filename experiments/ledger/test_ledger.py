"""Tests for ledger.py and compare_runs.py. Run: python3 -m pytest test_ledger.py"""

from __future__ import annotations

import json
from pathlib import Path

import compare_runs
import ledger


def manifest(run_id: str, values: list[float], **over) -> dict:
    m = {
        "run_id": run_id, "experiment": "E-99", "idea": "I-99", "status": "ok", "design_commit": "abc123",
        "config": {"backend": "x"}, "started_at": "2026-10-25T10:00:00Z", "finished_at": "2026-10-25T10:10:00Z",
        "hardware": {"gpu_name": "H100 SXM", "gpu_count": 1, "driver": "580.1", "cuda": "13.0"},
        "software": {"vllm": "0.32.0"},
        "metrics": {"tps": {"values": values, "unit": "tok/s", "higher_is_better": True}},
        "results_path": "x",
    }
    m.update(over)
    return m


def write(tmp: Path, m: dict) -> Path:
    p = tmp / f"{m['run_id']}.json"
    p.write_text(json.dumps(m))
    return p


def test_add_validates_and_is_append_only(tmp_path: Path) -> None:
    led = tmp_path / "runs.jsonl"
    assert ledger.add([write(tmp_path, manifest("r1", [1.0, 2.0]))], led) == 0
    assert ledger.add([write(tmp_path, manifest("r1", [1.0]))], led) == 1  # id reused
    bad = manifest("r2", [])
    assert ledger.add([write(tmp_path, bad)], led) == 1  # no values
    rows = ledger.load(led)
    assert [r["run_id"] for r in rows] == ["r1"] and len(rows[0]["config_hash"]) == 16


def test_supersedes_must_point_to_a_known_run(tmp_path: Path) -> None:
    led = tmp_path / "runs.jsonl"
    assert ledger.add([write(tmp_path, manifest("r2", [1.0], supersedes="nope"))], led) == 1


def rows(base: list[float], cand: list[float], **over) -> tuple[dict, dict]:
    b, c = manifest("b", base), manifest("c", cand, **over)
    for m in (b, c):
        m["config_hash"] = ledger.config_hash(m["config"])
    return b, c


def test_clear_gain_is_better_and_clear_loss_is_worse() -> None:
    b, c = rows([100, 101, 99, 100, 100], [110, 111, 109, 110, 110])
    assert compare_runs.compare(b, c, None, 0.02)["metrics"]["tps"]["verdict"] == "BETTER"
    b, c = rows([100, 101, 99, 100, 100], [90, 91, 89, 90, 90])
    assert compare_runs.compare(b, c, None, 0.02)["metrics"]["tps"]["verdict"] == "WORSE"


def test_lower_is_better_flips_the_direction() -> None:
    b, c = rows([100, 101, 99, 100, 100], [90, 91, 89, 90, 90])
    for m in (b, c):
        m["metrics"]["tps"]["higher_is_better"] = False
    assert compare_runs.compare(b, c, None, 0.02)["metrics"]["tps"]["verdict"] == "BETTER"


def test_small_change_is_no_change_and_noisy_change_is_unclear() -> None:
    b, c = rows([100, 100.2, 99.8, 100, 100.1], [100.5, 100.6, 100.4, 100.5, 100.5])
    assert compare_runs.compare(b, c, None, 0.02)["metrics"]["tps"]["verdict"] == "NO_CHANGE"
    b, c = rows([100, 80, 120], [104, 84, 124])
    assert compare_runs.compare(b, c, None, 0.02)["metrics"]["tps"]["verdict"] == "UNCLEAR"


def test_aa_noise_widens_the_band() -> None:
    b, c = rows([100, 100, 100, 100, 100], [104, 104, 104, 104, 104])
    assert compare_runs.compare(b, c, None, 0.02)["metrics"]["tps"]["verdict"] == "BETTER"
    aa = manifest("aa", [105, 105, 105, 105, 105])  # the same config moved 5% on its own
    out = compare_runs.compare(b, c, aa, 0.02)["metrics"]["tps"]
    assert out["band"] == 0.05 and out["verdict"] == "NO_CHANGE"


def test_fingerprint_differences_are_reported() -> None:
    b, c = rows([100, 100], [100, 100])
    c["hardware"] = {**c["hardware"], "gpu_name": "H200 SXM"}
    c["software"] = {"vllm": "0.33.0"}
    diffs = compare_runs.compare(b, c, None, 0.02)["fingerprint_differences"]
    assert any("gpu_name" in d for d in diffs) and any("vllm" in d for d in diffs)
