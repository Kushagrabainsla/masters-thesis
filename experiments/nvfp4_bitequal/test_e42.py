"""Tests for analyze_e42.py on synthetic data. Run: python3 -m pytest test_e42.py"""

from __future__ import annotations

import json
from pathlib import Path

import analyze_e42 as an

NAMES = {"R_1": "KR", "R_2": "KR", "R_m": "KR", "R_g": "KR", "marlin": "KM", "flashinfer_cudnn": "KD",
         "flashinfer_cutlass": "KF", "cutlass": "KC", "flashinfer_trtllm": "KT", "flashinfer_cutlass_g": "KF"}


def run(d: Path, delta: float = 0.0, flip: bool = False, ok: bool = True, k: int = 20, name: str | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.json").write_text(json.dumps({"status": "ok" if ok else "error"}))
    (d / "log.txt").write_text(f"INFO Using {name or NAMES[d.name]} for NVFP4 GEMM\n")
    rows = []
    for i in range(256):
        toks = [1, 2, 3, 4] if not (flip and i == 0) else [1, 9, 3, 4]
        topk = [[[t, -0.5 + (delta if j == 2 else 0.0)]] + [[100 + x, -2.0 - x] for x in range(k - 1)] for j, t in enumerate(toks)]
        rows.append(json.dumps({"id": f"q{i}", "token_ids": toks, "topk": topk}))
    (d / "outputs.jsonl").write_text("\n".join(rows) + "\n")


def base(tmp: Path, **cand) -> Path:
    run(tmp / "R_1"); run(tmp / "R_2"); run(tmp / "marlin", delta=1e-3); run(tmp / "R_m", delta=1e-7)
    for v in an.CANDS:
        run(tmp / v, **cand.get(v, {}))
    return tmp


def test_holds(tmp_path: Path) -> None:
    s = an.summarize(base(tmp_path))
    assert s["decision"].startswith("HYPOTHESIS HOLDS") and s["sensitivity"] == "shown"


def test_near_and_different(tmp_path: Path) -> None:
    s = an.summarize(base(tmp_path, flashinfer_cudnn={"delta": 1e-6}, cutlass={"flip": True}))
    assert s["decision"].startswith("HYPOTHESIS FAILS")
    assert s["backends"]["flashinfer_cudnn"]["label"] == "NEAR"
    assert s["backends"]["cutlass"]["label"] == "DIFFERENT"


def test_controls(tmp_path: Path) -> None:
    t = base(tmp_path / "a"); run(t / "R_2", delta=1e-7)
    assert "V1" in an.summarize(t)["decision"]
    t = base(tmp_path / "b"); run(t / "marlin")
    assert "V2" in an.summarize(t)["decision"]
    t = base(tmp_path / "c", flashinfer_trtllm={"ok": False})
    assert an.summarize(t)["backends"]["flashinfer_trtllm"]["label"] == "FAILED"


def test_alias_short_and_failed_hypothesis(tmp_path: Path) -> None:
    assert "alias" in an.summarize(base(tmp_path / "a", cutlass={"name": "KR"}))["decision"]
    assert "short" in an.summarize(base(tmp_path / "b", flashinfer_cutlass={"k": 0}))["decision"]
    assert an.summarize(base(tmp_path / "c", flashinfer_cudnn={"ok": False}))["decision"].startswith("INCONCLUSIVE")


def test_tie_order_does_not_matter(tmp_path: Path) -> None:
    a = {"q": {"token_ids": [1], "topk": [[[1, -0.5], [2, -0.5]] + [[100 + x, -3.0] for x in range(18)]]}}
    b = {"q": {"token_ids": [1], "topk": [[[2, -0.5], [1, -0.5]] + [[100 + x, -3.0] for x in range(18)]]}}
    assert an.compare(a, b)["bit_equal_fraction"] == 1.0
