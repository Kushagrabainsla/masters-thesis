"""Unit tests for gsm8k.py and analyze_a2.py on synthetic data. Run: python3 -m pytest test_a2.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_a2 as an
import gsm8k
import run_a2


def test_gold_and_extract() -> None:
    assert gsm8k.gold_of("... so 3 + 4 = 7\n#### 1,234") == "1234"
    assert gsm8k.extract("so the answer is \\boxed{18}.") == "18"
    assert gsm8k.extract("first \\boxed{2}, then \\boxed{\\frac{1}{2}} no, \\boxed{$5.50}") == "5.5"
    assert gsm8k.extract("no box here 42") == ""
    assert gsm8k.normalize("7.00") == "7" and gsm8k.normalize("-3") == "-3"


def test_approx_kl_zero_on_equal_and_positive_on_change() -> None:
    p = [(1, -0.1), (2, -2.5), (3, -4.0)]
    assert run_a2.approx_kl(p, p) == 0.0
    assert run_a2.approx_kl(p, [(1, -1.0), (2, -0.5), (3, -4.0)]) > 0.1


def rows(correct: list[int], prefix: str = "q") -> dict:
    return {f"{prefix}{i}": {"id": f"{prefix}{i}", "correct": bool(c), "token_ids": [i, c]} for i, c in enumerate(correct)}


def test_paired_interval_sign() -> None:
    ref = rows([1] * 90 + [0] * 10)
    same = rows([1] * 90 + [0] * 10)
    worse = rows([1] * 80 + [0] * 20)
    d, lo, hi = an.paired_interval(ref, same)
    assert d == 0 and lo == 0 and hi == 0
    d, lo, hi = an.paired_interval(ref, worse)
    assert d == -0.1 and hi < 0


def write_cell(tmp: Path, cand_correct: list[int], cand_kl: float, bf16_kl: float, rb_kl: float = 0.0) -> Path:
    cell = tmp / "cell_D"
    def put(name: str, correct: list[int], kl: float | None, dis: float = 0.0) -> None:
        d = cell / name
        d.mkdir(parents=True)
        (d / "completions.jsonl").write_text("\n".join(json.dumps(r) for r in rows(correct).values()) + "\n")
        (d / "run.json").write_text(json.dumps({"status": "ok"}))
        if kl is not None:
            (d / "tf.json").write_text(json.dumps({"kl_mean": kl, "top1_disagree_rate": dis}))
    base = [1] * 900 + [0] * 100
    put("R_a", base, None)
    put("R_b", base, rb_kl)
    put("bf16", [1] * 880 + [0] * 120, bf16_kl, 0.05)
    put("cand_x", cand_correct, cand_kl, 0.02)
    return cell


def test_accept_rules(tmp_path: Path) -> None:
    cell = write_cell(tmp_path, [1] * 899 + [0] * 101, cand_kl=0.01, bf16_kl=0.05)
    out = an.accept(cell)
    assert out["candidates"]["x"]["acceptable"] is True
    cell2 = write_cell(tmp_path / "b", [1] * 850 + [0] * 150, cand_kl=0.01, bf16_kl=0.05)
    assert an.accept(cell2)["candidates"]["x"]["A1"] is False
    cell3 = write_cell(tmp_path / "c", [1] * 900 + [0] * 100, cand_kl=0.09, bf16_kl=0.05)
    assert an.accept(cell3)["candidates"]["x"]["A2"] is False


def write_speed(d: Path, r: float, c: float) -> None:
    for name, v in (("R_1", r), ("C_1", c), ("R_2", r), ("C_2", c)):
        p = d / name
        p.mkdir(parents=True)
        secs = [128 * 8 / (v * f) for f in (1.0, 1.001, 0.999, 1.0, 1.0005)]
        (p / "run.json").write_text(json.dumps({"args": {"throughput_max_tokens": 128},
                                                "throughput": {"decode": {"8": {"seconds": secs}}}}))


def test_headroom_and_decision(tmp_path: Path) -> None:
    cell = write_cell(tmp_path, [1] * 899 + [0] * 101, cand_kl=0.01, bf16_kl=0.05)
    write_speed(cell / "speed" / "x", r=1000.0, c=1100.0)
    h = an.headroom(cell / "speed" / "x")
    assert h["better_at"] == ["8"]
    v = an.validity(cell, False)
    assert v["V1"] and v["V2"] and v["V3"] and v["V5"]
    cells = {"cell_D": {"validity": v, "accept": an.accept(cell), "headroom": {"x": h}, "headroom_unacceptable": {}}}
    assert an.decide(cells).startswith("GO")
    cells["cell_D"]["headroom"] = {"x": an.headroom(_slow(tmp_path))}
    assert an.decide(cells) == "KILL"


def _slow(tmp: Path) -> Path:
    d = tmp / "slow"
    write_speed(d, r=1000.0, c=990.0)
    return d


def test_invalid_when_reference_is_not_reproducible(tmp_path: Path) -> None:
    cell = write_cell(tmp_path, [1] * 900 + [0] * 100, cand_kl=0.01, bf16_kl=0.05, rb_kl=0.01)
    v = an.validity(cell, False)
    assert v["V3"] is False
    assert an.decide({"cell_D": {"validity": v, "accept": an.accept(cell), "headroom": {}, "headroom_unacceptable": {}}}).startswith("INVALID")


def test_errored_candidate_blocks_kill(tmp_path: Path) -> None:
    cell = write_cell(tmp_path, [1] * 900 + [0] * 100, cand_kl=0.01, bf16_kl=0.05)
    bad = cell / "cand_y"
    bad.mkdir()
    (bad / "run.json").write_text(json.dumps({"status": "error"}))
    acc = an.accept(cell)
    assert acc["errored"] == ["y"]
    cells = {"cell_D": {"validity": an.validity(cell, False), "accept": acc, "headroom": {}, "headroom_unacceptable": {}}}
    assert an.decide(cells).startswith("INCONCLUSIVE")


def test_invalid_cell_does_not_hide_a_valid_go(tmp_path: Path) -> None:
    good = write_cell(tmp_path / "g", [1] * 899 + [0] * 101, cand_kl=0.01, bf16_kl=0.05)
    write_speed(good / "speed" / "x", r=1000.0, c=1100.0)
    bad = write_cell(tmp_path / "b", [1] * 900 + [0] * 100, cand_kl=0.01, bf16_kl=0.05, rb_kl=0.01)
    cells = {
        "cell_M": {"validity": an.validity(good, True), "accept": an.accept(good),
                   "headroom": {"x": an.headroom(good / "speed" / "x")}, "headroom_unacceptable": {}},
        "cell_D": {"validity": an.validity(bad, False), "accept": an.accept(bad), "headroom": {}, "headroom_unacceptable": {}},
    }
    cells["cell_M"]["validity"]["V4"] = True
    assert an.decide(cells).startswith("GO")
    cells["cell_M"]["headroom"] = {}
    assert an.decide(cells).startswith("INCONCLUSIVE")


def test_a2_ratio_needs_half_of_the_anchor(tmp_path: Path) -> None:
    cell = write_cell(tmp_path, [1] * 900 + [0] * 100, cand_kl=0.03, bf16_kl=0.05)
    assert an.accept(cell)["candidates"]["x"]["A2"] is False  # 0.03 > 0.5 x 0.05


def test_decode_prefix_stops_after_first_divergence() -> None:
    ref = {"a": [[(1, -0.1)], [(2, -0.1)], [(3, -0.1)]]}
    out = run_a2.compare_decode_prefix(ref, [[[(1, -0.1)], [(5, -0.1)], [(9, -0.1)]]], ["a"],
                                       {"a": [1, 2, 3]}, {"a": [1, 5, 9]})
    assert out["positions"] == 2 and out["top1_disagree_rate"] == 0.5
