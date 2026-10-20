"""Apply the frozen E-31 screening rules."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any


NON_CACHE_FAILURES = {
    "stale_work_applied",
    "stale_event_emitted",
    "logical_history_lost",
}


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def violation_union(rows: list[dict[str, Any]], policy: str) -> set[str]:
    return {
        kind
        for row in rows
        if row["policy"] == policy
        for kind in row["violation_types"]
    }


def decide(result_dir: Path) -> dict[str, Any]:
    source = read_jsonl(result_dir / "source_scenarios.jsonl")
    model_check = json.loads((result_dir / "model_check.json").read_text())
    summary = read_csv(result_dir / "summary.csv")

    source_types = {
        policy: sorted(violation_union(source, policy))
        for policy in (
            "mutable",
            "hash_only",
            "epoch_only",
            "abort_rebuild",
            "synchronous",
            "versioned",
        )
    }
    bounded_types = {
        policy: model_check["policies"][policy]["minimal_violation_types"]
        for policy in model_check["policies"]
    }
    sweep_max = {
        policy: max(
            int(row["max_violations"])
            for row in summary
            if row["policy"] == policy
        )
        for policy in model_check["policies"]
    }

    versioned_failures = set(source_types["versioned"]) | set(bounded_types["versioned"])
    if sweep_max["versioned"]:
        versioned_failures.add("scale_sweep_violation")

    reference_failures = set(source_types["abort_rebuild"]) | set(
        source_types["synchronous"]
    )
    current_failures = set(source_types["mutable"])
    hash_failures = set(source_types["hash_only"])
    hash_non_cache = hash_failures & NON_CACHE_FAILURES
    epoch_failures = set(source_types["epoch_only"])

    rebuild_rows = [
        row
        for row in summary
        if row["policy"] == "abort_rebuild"
        and float(row["update_rate"]) in (0.01, 0.05)
    ]
    rebuild_is_cheap_everywhere = bool(rebuild_rows) and all(
        float(row["median_rebuild_ratio"]) <= 0.03 for row in rebuild_rows
    )
    noisy_rows = [row for row in summary if float(row["timing_cv"]) > 0.10]

    if reference_failures:
        verdict = "INVALID_REFERENCE_MODEL"
        reason = "A conservative reference baseline violated a source-derived invariant."
    elif versioned_failures:
        verdict = "KILL_MECHANISM"
        reason = "The versioned protocol violated a correctness invariant."
    elif not hash_failures:
        verdict = "KILL_SIMPLE_BASELINE"
        reason = "Hash invalidation handled every source-derived scenario."
    elif not epoch_failures:
        verdict = "PIVOT_EPOCH_ONLY"
        reason = "A generation fence handled every source-derived scenario without full version state."
    elif rebuild_is_cheap_everywhere:
        verdict = "PIVOT_TRIVIAL_REBUILD"
        reason = "Full abort and rebuild stayed within the 3% synthetic token-work threshold."
    elif len(current_failures) < 2:
        verdict = "PIVOT_WEAK_PREMISE"
        reason = "The source-derived matrix exposed fewer than two independent failure classes."
    elif not hash_non_cache:
        verdict = "KILL_SIMPLE_BASELINE"
        reason = "Hash invalidation removed every non-cache failure class."
    else:
        verdict = "PASS_TO_INTEGRATION"
        reason = "The versioned protocol passed, while hash invalidation left non-cache failures."

    return {
        "experiment": "E-31",
        "verdict": verdict,
        "reason": reason,
        "source_violation_types": source_types,
        "bounded_violation_types": bounded_types,
        "scale_sweep_max_violations": sweep_max,
        "hash_non_cache_failures": sorted(hash_non_cache),
        "epoch_only_failures": sorted(epoch_failures),
        "rebuild_is_cheap_everywhere": rebuild_is_cheap_everywhere,
        "rebuild_threshold": 0.03,
        "timing_rows": len(summary),
        "timing_rows_with_cv_above_10_percent": len(noisy_rows),
        "timing_is_decisive": False,
        "scope": (
            "This verdict controls only progression to an implementation-level test. "
            "It is not a GO verdict for I-122."
        ),
    }


def render_markdown(decision: dict[str, Any]) -> str:
    lines = [
        "# E-31 frozen decision",
        "",
        f"Verdict: **{decision['verdict']}**",
        "",
        decision["reason"],
        "",
        "## Evidence",
        "",
        f"- Mutable source failures: {decision['source_violation_types']['mutable']}",
        f"- Hash-only source failures: {decision['source_violation_types']['hash_only']}",
        f"- Epoch-only source failures: {decision['source_violation_types']['epoch_only']}",
        f"- Versioned source failures: {decision['source_violation_types']['versioned']}",
        f"- Versioned bounded failures: {decision['bounded_violation_types']['versioned']}",
        f"- Hash-only non-cache failures: {decision['hash_non_cache_failures']}",
        f"- Full rebuild cheap in every low-update cell: {decision['rebuild_is_cheap_everywhere']}",
        "",
        "## Timing validity",
        "",
        (
            f"{decision['timing_rows_with_cv_above_10_percent']} of "
            f"{decision['timing_rows']} timing rows have a coefficient of variation above 10%."
        ),
        "",
        "Docker timing is report-only. Host activity can still affect Docker Desktop.",
        "",
        "## Scope",
        "",
        decision["scope"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = decide(args.result_dir)
    atomic_text(
        args.result_dir / "decision.json",
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
    )
    atomic_text(args.result_dir / "decision.md", render_markdown(decision))


if __name__ == "__main__":
    main()
