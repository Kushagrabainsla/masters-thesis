#!/usr/bin/env python3
"""Audit the current vLLM session-state evidence without executing vLLM."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from datetime import UTC, datetime


REPOSITORY = "vllm-project/vllm"
API_ROOT = f"https://api.github.com/repos/{REPOSITORY}"
ISSUES = (49449, 49377, 36755, 42655, 58776, 60744)
PULLS = (59103, 37160, 43683, 58259, 58778, 60764)
SOURCE_PATHS = (
    "vllm/v1/request.py",
    "vllm/v1/core/sched/scheduler.py",
    "vllm/v1/core/sched/async_scheduler.py",
    "vllm/v1/worker/gpu_model_runner.py",
    "vllm/v1/engine/output_processor.py",
)
TERMS = (
    "session_id",
    "generation",
    "revision",
    "incarnation",
    "epoch",
    "commit",
    "streaming_queue",
    "sched_step_seq",
    "processed_step_seq",
)
TARGET_TEST_SUFFIXES = (
    "tests/v1/core/test_scheduler.py",
    "tests/v1/core/test_async_scheduler.py",
)


def get_bytes(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "masters-thesis-i122-source-audit",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def get_json(path: str) -> dict:
    return json.loads(get_bytes(f"{API_ROOT}{path}"))


def issue_record(number: int) -> dict:
    item = get_json(f"/issues/{number}")
    return {
        "number": number,
        "title": item["title"],
        "state": item["state"],
        "state_reason": item.get("state_reason"),
        "created_at": item["created_at"],
        "closed_at": item.get("closed_at"),
        "updated_at": item["updated_at"],
        "url": item["html_url"],
    }


def pull_record(number: int) -> dict:
    item = get_json(f"/pulls/{number}")
    return {
        "number": number,
        "title": item["title"],
        "state": item["state"],
        "merged": item["merged"],
        "merged_at": item.get("merged_at"),
        "merge_commit_sha": item.get("merge_commit_sha"),
        "updated_at": item["updated_at"],
        "url": item["html_url"],
    }


def source_record(sha: str, path: str) -> dict:
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{sha}/{path}"
    data = get_bytes(url)
    text = data.decode("utf-8")
    lines = text.splitlines()
    hits = {}
    for term in TERMS:
        pattern = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])",
            re.IGNORECASE,
        )
        line_numbers = [index + 1 for index, line in enumerate(lines) if pattern.search(line)]
        if line_numbers:
            hits[term] = line_numbers
    return {
        "path": path,
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "lines": len(lines),
        "term_hits": hits,
    }


def test_stats(sha: str) -> dict:
    archive_url = f"https://codeload.github.com/{REPOSITORY}/tar.gz/{sha}"
    archive = get_bytes(archive_url)
    python_files: list[tuple[str, str]] = []
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            relative = member.name.split("/", 1)[-1]
            if not relative.startswith("tests/v1/") or not relative.endswith(".py"):
                continue
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            python_files.append((relative, extracted.read().decode("utf-8")))

    target_files = [
        (path, text)
        for path, text in python_files
        if path.endswith(TARGET_TEST_SUFFIXES)
        or path.startswith("tests/v1/streaming_input/")
    ]
    combined = "\n".join(text for _, text in target_files)
    test_pattern = re.compile(r"^\s*(?:async\s+)?def\s+test_", re.MULTILINE)
    return {
        "archive_url": archive_url,
        "archive_bytes": len(archive),
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "v1_python_files": len(python_files),
        "target_python_files": len(target_files),
        "target_test_functions": len(test_pattern.findall(combined)),
        "target_parametrize_markers": combined.count("@pytest.mark.parametrize"),
        "target_hypothesis_mentions": len(
            re.findall(r"from hypothesis|import hypothesis|RuleBasedStateMachine", combined)
        ),
        "target_random_mentions": len(
            re.findall(r"from random|import random|random\.", combined)
        ),
        "target_paths": [path for path, _ in target_files],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", help="Use this vLLM commit instead of the current main commit.")
    args = parser.parse_args()

    observed_main = get_json("/git/ref/heads/main")["object"]["sha"]
    sha = args.sha or observed_main
    output = {
        "schema": "i122-source-audit-v1",
        "collected_at": datetime.now(UTC).isoformat(),
        "repository": REPOSITORY,
        "observed_main_sha": observed_main,
        "audited_sha": sha,
        "issues": [issue_record(number) for number in ISSUES],
        "pulls": [pull_record(number) for number in PULLS],
        "source_files": [source_record(sha, path) for path in SOURCE_PATHS],
        "test_stats": test_stats(sha),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
