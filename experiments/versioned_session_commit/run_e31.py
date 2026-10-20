"""Run the E-31 abstract correctness and cost screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from model import ALL_POLICIES, Policy, SessionRuntime


ROOT = Path(__file__).resolve().parent


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def source_scenarios() -> dict[str, Callable[[SessionRuntime], None]]:
    def stale_hash(runtime: SessionRuntime) -> None:
        runtime.replace_last(11)
        runtime.schedule()
        runtime.complete_oldest()
        runtime.publish_cache()

    def stale_completion(runtime: SessionRuntime) -> None:
        runtime.schedule()
        runtime.append_input(13)
        runtime.complete_oldest()
        runtime.flush_oldest()

    def stale_parser(runtime: SessionRuntime) -> None:
        runtime.schedule()
        runtime.complete_oldest()
        runtime.append_input(17)
        runtime.flush_oldest()

    def split_frontier(runtime: SessionRuntime) -> None:
        runtime.preempt()
        runtime.fold_after_preempt()

    return {
        "stale_hash_after_replace": stale_hash,
        "async_completion_after_update": stale_completion,
        "parser_flush_after_update": stale_parser,
        "logical_materialized_frontier_split": split_frontier,
    }


def run_source_scenarios() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    initial = tuple([10] * 15 + [28])
    for scenario_name, scenario in source_scenarios().items():
        for policy in ALL_POLICIES:
            runtime = SessionRuntime(policy=policy, initial_tokens=initial)
            scenario(runtime)
            stats = runtime.stats()
            row = {
                "scenario": scenario_name,
                **stats,
                "history": runtime.history,
            }
            rows.append(row)
            for violation in runtime.violations:
                violations.append(
                    {
                        "source": "source_scenario",
                        "scenario": scenario_name,
                        "policy": policy.value,
                        **violation,
                    }
                )
    return rows, violations


def bounded_model_check(max_depth: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {"max_depth": max_depth, "policies": {}}
    violations: list[dict[str, Any]] = []

    for policy in ALL_POLICIES:
        start = SessionRuntime(policy=policy, initial_tokens=(1, 2, 3))
        frontier: list[tuple[SessionRuntime, tuple[str, ...]]] = [(start, ())]
        seen = {(0, start.signature())}
        states_by_depth = [1]
        first_by_kind: dict[str, dict[str, Any]] = {}

        for depth in range(max_depth):
            next_frontier: list[tuple[SessionRuntime, tuple[str, ...]]] = []
            for runtime, path in frontier:
                for action in runtime.enabled_actions():
                    candidate = runtime.clone()
                    previous_count = len(candidate.violations)
                    candidate.apply(action)
                    next_path = path + (action,)
                    for violation in candidate.violations[previous_count:]:
                        record = {
                            "source": "bounded_model_check",
                            "policy": policy.value,
                            "path": list(next_path),
                            **violation,
                        }
                        first_by_kind.setdefault(violation["kind"], record)
                    key = (depth + 1, candidate.signature())
                    if key not in seen:
                        seen.add(key)
                        next_frontier.append((candidate, next_path))
            frontier = next_frontier
            states_by_depth.append(len(frontier))

        minimal = [first_by_kind[kind] for kind in sorted(first_by_kind)]
        violations.extend(minimal)
        report["policies"][policy.value] = {
            "states_by_depth": states_by_depth,
            "unique_states": sum(states_by_depth),
            "minimal_violation_types": sorted(first_by_kind),
            "minimal_violations": minimal,
        }

    return report, violations


def build_trace(
    session_count: int,
    turns: int,
    update_rate: float,
    preempt_rate: float,
    outstanding_depth: int,
    seed: int,
) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    trace: list[tuple[int, str]] = []
    for _ in range(turns):
        order = list(range(session_count))
        rng.shuffle(order)
        for session_id in order:
            update = rng.random() < update_rate
            phase = rng.randrange(3) if update else -1
            update_action = "append_input" if rng.random() < 0.5 else "replace_last"
            for _ in range(outstanding_depth):
                trace.append((session_id, "schedule"))
            if phase == 0:
                trace.append((session_id, update_action))
            did_preempt = rng.random() < preempt_rate
            if did_preempt:
                trace.append((session_id, "preempt"))
            for _ in range(outstanding_depth):
                trace.append((session_id, "complete_oldest"))
            if phase == 1:
                trace.append((session_id, update_action))
            for _ in range(outstanding_depth):
                trace.append((session_id, "flush_oldest"))
            trace.append((session_id, "publish_cache"))
            if did_preempt:
                trace.append((session_id, "fold_after_preempt"))
            if phase == 2:
                trace.append((session_id, update_action))
    return trace


def run_trace(
    policy: Policy,
    trace: list[tuple[int, str]],
    session_count: int,
    context_len: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sessions = [
        SessionRuntime(
            policy=policy,
            initial_tokens=tuple([1 + (session_id % 997)] * context_len),
            session_id=session_id,
        )
        for session_id in range(session_count)
    ]
    start_ns = time.perf_counter_ns()
    for session_id, action in trace:
        sessions[session_id].apply(action)
    elapsed_ns = time.perf_counter_ns() - start_ns

    totals: dict[str, Any] = defaultdict(int)
    violation_types: set[str] = set()
    violations: list[dict[str, Any]] = []
    for runtime in sessions:
        stats = runtime.stats()
        for key in (
            "operations",
            "protocol_checks",
            "violations",
            "discarded_work",
            "discarded_events",
            "rebuild_tokens",
            "computed_tokens",
            "drained_work",
            "missing_callbacks",
            "updates",
        ):
            totals[key] += stats[key]
        violation_types.update(stats["violation_types"])
        for violation in runtime.violations:
            if len(violations) >= 50:
                break
            violations.append(
                {
                    "source": "scale_sweep",
                    "policy": policy.value,
                    "session_id": runtime.session_id,
                    **violation,
                }
            )

    totals.update(
        {
            "policy": policy.value,
            "elapsed_ns": elapsed_ns,
            "actions_per_second": totals["operations"] / max(elapsed_ns / 1e9, 1e-12),
            "violation_types": ";".join(sorted(violation_types)),
            "rebuild_ratio": totals["rebuild_tokens"] / max(totals["computed_tokens"], 1),
            "updates_per_session": totals["updates"] / max(session_count, 1),
        }
    )
    return dict(totals), violations


def scale_sweep(repeats: int, turns: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []
    all_violations: list[dict[str, Any]] = []
    for sessions in (1, 10, 100, 1_000):
        for context_len in (16, 256):
            for update_rate in (0.01, 0.05, 0.20):
                for preempt_rate in (0.0, 0.10):
                    for outstanding_depth in (1, 4):
                        for repeat in range(repeats):
                            seed = (
                                sessions * 1_000_003
                                + context_len * 10_007
                                + int(update_rate * 10_000) * 101
                                + int(preempt_rate * 10_000) * 17
                                + outstanding_depth * 13
                            )
                            trace = build_trace(
                                session_count=sessions,
                                turns=turns,
                                update_rate=update_rate,
                                preempt_rate=preempt_rate,
                                outstanding_depth=outstanding_depth,
                                seed=seed,
                            )
                            offset = repeat % len(ALL_POLICIES)
                            policy_order = ALL_POLICIES[offset:] + ALL_POLICIES[:offset]
                            for policy in policy_order:
                                stats, violations = run_trace(
                                    policy=policy,
                                    trace=trace,
                                    session_count=sessions,
                                    context_len=context_len,
                                )
                                row = {
                                    "sessions": sessions,
                                    "context_len": context_len,
                                    "update_rate": update_rate,
                                    "preempt_rate": preempt_rate,
                                    "outstanding_depth": outstanding_depth,
                                    "turns": turns,
                                    "repeat": repeat,
                                    "seed": seed,
                                    **stats,
                                }
                                raw.append(row)
                                for violation in violations:
                                    all_violations.append({**row, **violation})
    return raw, all_violations


def summarize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = (
        "sessions",
        "context_len",
        "update_rate",
        "preempt_rate",
        "outstanding_depth",
        "turns",
        "policy",
    )
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        groups[tuple(row[key] for key in keys)].append(row)

    summary: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items(), key=lambda item: item[0]):
        elapsed = [row["elapsed_ns"] for row in rows]
        rates = [row["actions_per_second"] for row in rows]
        rebuild = [row["rebuild_ratio"] for row in rows]
        median_elapsed = statistics.median(elapsed)
        cv = statistics.pstdev(elapsed) / statistics.mean(elapsed) if len(elapsed) > 1 else 0.0
        kinds = sorted(
            {
                kind
                for row in rows
                for kind in str(row["violation_types"]).split(";")
                if kind
            }
        )
        summary.append(
            {
                **dict(zip(keys, key, strict=True)),
                "repeats": len(rows),
                "median_elapsed_ns": int(median_elapsed),
                "timing_cv": cv,
                "median_actions_per_second": statistics.median(rates),
                "max_violations": max(row["violations"] for row in rows),
                "violation_types": ";".join(kinds),
                "median_protocol_checks": statistics.median(
                    row["protocol_checks"] for row in rows
                ),
                "median_discarded_work": statistics.median(
                    row["discarded_work"] for row in rows
                ),
                "median_discarded_events": statistics.median(
                    row["discarded_events"] for row in rows
                ),
                "median_rebuild_ratio": statistics.median(rebuild),
                "median_updates_per_session": statistics.median(
                    row["updates_per_session"] for row in rows
                ),
            }
        )
    return summary


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_names = (
        "model.py",
        "run_e31.py",
        "decide_e31.py",
        "test_model.py",
        "Dockerfile",
        "run_docker.sh",
        "README.md",
    )
    return {
        "experiment": "E-31",
        "idea": "I-122",
        "status": "screen_only",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "parameters": {
            "model_check_depth": args.model_check_depth,
            "repeats": args.repeats,
            "turns": args.turns,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "docker_image_id": os.environ.get("E31_IMAGE_ID", "unknown"),
            "docker_cpu_limit": os.environ.get("E31_CPU_LIMIT", "unknown"),
            "docker_memory_limit": os.environ.get("E31_MEMORY_LIMIT", "unknown"),
        },
        "source_sha256": {
            name: file_hash(ROOT / name)
            for name in source_names
            if (ROOT / name).exists()
        },
        "validity": [
            "The state machine is abstract. It is not a vLLM execution.",
            "The scale sweep is synthetic. It does not establish production rates.",
            "Wall-clock timing inside Docker is report-only.",
            "Host activity can still affect the Docker Desktop virtual machine.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-check-depth", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--turns", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", build_manifest(args))

    source_rows, source_violations = run_source_scenarios()
    write_jsonl(args.output_dir / "source_scenarios.jsonl", source_rows)

    model_check, model_violations = bounded_model_check(args.model_check_depth)
    write_json(args.output_dir / "model_check.json", model_check)

    raw, scale_violations = scale_sweep(args.repeats, args.turns)
    write_csv(args.output_dir / "throughput.csv", raw)
    write_csv(args.output_dir / "summary.csv", summarize(raw))

    all_violations = source_violations + model_violations + scale_violations
    write_jsonl(args.output_dir / "violations.jsonl", all_violations)


if __name__ == "__main__":
    main()
