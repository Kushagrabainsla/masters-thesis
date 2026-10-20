# E-33: Mutable-session CPU corpus pilot

Status: CANCELLED BEFORE RUN.

Idea: I-131. Date: 2026-10-19.

R-25 prescribed this pilot before any E-32 generator implementation.
R-26 blocked the pilot specification.
R-27 removed its thesis value before execution.
Do not run E-33 for thesis evidence.

## Question

Do four merged vLLM fixes provide stable CPU parent-and-fix regression pairs?

## Frozen corpus

Use only PRs 59103, 58259, 58668, and 58778.
Do not replace a case after the first parent test starts.

Use each merge commit as the fix.
Use its only parent as the failing commit.
Copy only the fix's test-file diff onto the parent checkout.
Run the same upstream test node on both checkouts.

PR 44526 is excluded because it closed without merging.
PR 60764 is excluded because it remains open.

Exact commits and test nodes are in `bug_corpus.json`.

## Result classification

A pair passes only when both conditions hold:

1. The parent test fails because its target assertion detects the documented fault.
2. The identical test passes on the fix commit.

Do not count an import error, collection error, timeout, crash, or setup error as a parent failure.
Do not count a test that passes on both commits.

## Frozen decision rules

These thresholds are project heuristics.

- PASS-TO-AMEND: all four pairs pass.
- PIVOT-CORPUS: two or three pairs pass.
- KILL-CURRENT-CORPUS: zero or one pair passes.
- INVALID-ENVIRONMENT: three or more pairs fail before test execution because of one shared environment fault.

If the result is INVALID-ENVIRONMENT, repair only the shared environment.
Keep the corpus, commits, tests, and decision rules unchanged.

PASS-TO-AMEND does not validate I-131.
It permits a revised E-32 design and a second independent review.

## Docker setup

- Base image: `vllm/vllm-openai-cpu@sha256:433cbee5e0b689095484090454b6fcdf0c68fe3eabfbb251a5e39d442917a4a6`.
- Architecture: Linux ARM64 on Docker Desktop.
- CPU limit: four CPUs, pinned to CPUs 0 through 3.
- Memory limit: 4 GiB.
- Network: allowed during image construction only; disabled during each test run.
- Source: exact vLLM commits stored in the image.
- Results: one JSON record and one log per commit.

Treat wall-clock time and memory as feasibility data only.
Do not use timing to decide the research question.

## Pre-run rubber duck

Tempting claim: Four passing pairs show that a stateful generator will find useful bugs.

Actual measurement: The pilot tests whether four upstream regression tests work as CPU differential controls.

Simple explanation: A parent can fail because a copied test depends on newer test helpers.
Classify that outcome as setup failure, not bug reproduction.

Strong negative result: Fewer than four pairs pass.
This result invalidates the current E-32 corpus and its four-case minimum.

Remaining risk: The same author designed the corpus and runs the pilot.
The tests are upstream tests, but the classification still needs a saved failure log.

## Outputs

- `results/<run-id>/manifest.json`;
- `results/<run-id>/pair_results.json`;
- `results/<run-id>/logs/*.log`;
- `results/<run-id>/decision.json`;
- `dossier/bundles/E-33_<run-id>.zip`.

