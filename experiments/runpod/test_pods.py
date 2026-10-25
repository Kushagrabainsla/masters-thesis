"""Tests for pods.py. Run: python3 -m pytest test_pods.py"""

from __future__ import annotations

from pathlib import Path

import pods


def test_label_format() -> None:
    assert pods.label("E-36", "Rehearsal L4", "20261026") == "msthesis-e-36-rehearsal-l4-20261026"
    assert pods.NAME.match(pods.label("E-36", "run"))


def test_only_registered_and_live_ids_are_owned(tmp_path: Path) -> None:
    reg = tmp_path / "pods.jsonl"
    assert pods.add("p1", "msthesis-e-36-run-20261026", "E-36", "pod", path=reg) is None
    assert "p1" in pods.owned(reg)
    assert "other" not in pods.owned(reg)  # a Pod from another session is never owned
    assert pods.end("p1", path=reg) is None
    assert "p1" not in pods.owned(reg)
    assert pods.end("p1", path=reg) is not None  # cannot end twice


def test_unlabelled_name_and_reused_id_are_rejected(tmp_path: Path) -> None:
    reg = tmp_path / "pods.jsonl"
    assert pods.add("p1", "okc-bf16-proj-tuning", "E-36", "pod", path=reg) is not None
    assert pods.add("p2", "msthesis-e-36-run-20261026", "E-36", "pod", path=reg) is None
    assert pods.add("p2", "msthesis-e-36-run-20261026", "E-36", "pod", path=reg) is not None
