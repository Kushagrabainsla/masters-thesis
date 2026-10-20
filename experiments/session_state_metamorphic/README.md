# E-32: Stateful metamorphic session test

Status: KILL-DESIGN. NOT RUN.

Idea: I-131. Date: 2026-10-17.

Do not implement or run this harness.
R-26 found unresolved blockers.
R-27 killed I-131 as a thesis direction after the paper audit.

## Research question

Can stateful transition generation with history-equivalence oracles find more
mutable-session faults than current tests and unguided random actions?

## System under test

Use the real vLLM V1 scheduler on CPU.
Use fake `ModelRunnerOutput` values.
Do not load a model or use a GPU in the first screen.

Run the test inside a Linux ARM64 Docker container on the Apple M4 Mac.
Pin every tested vLLM commit and the base image digest.
Bake all source and model configuration into the image before the frozen run.
Disable network access during the frozen run.

## Frozen bug corpus

Use only rows in `bug_corpus.json`.
Do not replace or drop a row after the corpus freeze.
Require four eligible rows before the main screen.

The initial table has four stable merged fixes.

1. Issue 49377 and PR 59103: stale block hashes after token truncation.
2. PR 58259: resumable-request async handoff race.
3. PR 58668: accepted speculative tokens remain counted after stop trimming.
4. Issue 58776 and PR 58778: a stopped preempted session loses queue ownership.

Exclude PR 44526 because it is closed without a merge.
Exclude issue 42655 because its proposed fix is unmerged and its failure is GPU model-runner state.
Exclude issue 60744 because PR 60764 is still open.

## Step 0 corpus and setup pilot

Run this pilot before implementing a generator.

For every corpus row, run the named regression test at the exact parent commit.
Run the same test at the exact merged fix commit.
Require failure at the parent and success at the fix.
Record the test output, exit code, source SHA, image digest, and test duration.
Mark a row `ELIGIBLE` only when both results are clear.
Kill I-131 if fewer than four rows are eligible.
Do not treat an open pull request head as a fixed control.
Measure setup time and test time, but do not use speed as a scientific kill rule.

Freeze the corpus, harness, thresholds, and factor values after Step 0.

## Action alphabet

- add a new resumable request;
- add a streaming continuation;
- schedule one step;
- return a normal token;
- return a stop token;
- preempt a request;
- reset the prefix cache;
- pause and resume the scheduler;
- complete an older in-flight step.

The generator must use only actions that satisfy their public preconditions.

## Oracles

1. History equivalence: an allowed state-reclaim action cannot change logical tokens.
2. Execution equivalence: sync and async executions must commit the same logical history.
3. Queue ownership: each unfinished request has exactly one valid scheduling home.
4. Cache identity: each reusable block hash names the current token block.
5. Progress bounds: computed progress never covers discarded or nonexistent tokens.
6. Liveness: a resumable request with available work becomes schedulable within a bounded number of steps.

Compare the same logical trace with and without the tested transformation.
Keep an independent logical token model for expected history.
Do not read scheduler fields as the expected value.
Define every action precondition and legal outcome in the harness specification.
Use a numeric liveness bound of 8 scheduler steps for the first screen.
Rename oracles 3 to 6 as invariants.

## Baselines

1. The current vLLM scheduler and streaming test suites.
2. Hypothesis `RuleBasedStateMachine` with the same oracles and action budget.
3. Uniform random valid actions with the same oracles and action budget.
4. The domain-aware stateful generator without trace guidance.
5. The full generator with history-equivalence guidance and shrinking.

Run at least 30 fixed seeds per policy.
Use the same configuration hash and wall-clock budget for every policy.

## Independent variables

- tested parent or fix commit;
- generator policy;
- sequence budget from the Step 0 setup pilot;
- maximum trace length: 4, 8, and 12 actions;
- sync or async scheduling;
- prefix caching on or off;
- speculative decoding on or off, where CPU tests support it;
- preemption and reset actions on or off.

## Dependent variables

- known bugs rediscovered;
- unique failure signatures;
- executions per second;
- executions to first failure;
- minimized action count;
- invalid generated sequences;
- false positives on fixed commits;
- peak container memory.

## Frozen screen criteria

These thresholds are project heuristics.

Apply rules in this order:

1. Kill the corpus if fewer than four rows are eligible in Step 0.
2. Kill the tool claim if more than 1% of fixed-commit control sequences fail after independent triage.
3. Hold the result for triage if 1 to 10 of 1,000 controls fail.
4. Require at least one new confirmed failure on current main for a GO claim.
5. Kill the guided-generator claim if Hypothesis or uniform random actions match it.
6. Pivot to a testing-tool result if known bugs are found but no new current-main bug appears.
7. Require at least four eligible known bugs, two misses by both random baselines, and minimized traces of 12 actions or fewer for GO.

Treat speed as an informational feasibility result.
Do not use a fixed sequences-per-second threshold as a scientific kill rule.

## Required controls

- Run each eligible known failing parent and fixed commit.
- Plant one fault for each oracle before the main run.
- Require each planted fault to fail.
- Keep fixed-commit failures separate from known parent failures.
- Triage every fixed-commit failure with a second reviewer or a source citation.
- Save and replay every minimized trace.
- Record the harness commit, source commit, image digest, source hashes, seed, and container limits.
- Require a clean git tree before the frozen run.

## Validity limits

- A CPU scheduler test does not cover GPU runner state.
- A four-bug corpus does not estimate all vLLM bug classes.
- A known-bug rediscovery screen can overfit the generator design.
- Generic property testing applied to vLLM is not enough for a paper.
- A publication needs new bugs, cross-engine transfer, or a new semantic oracle.

## Planned outputs

- `manifest.json`;
- `bug_corpus.json`;
- `runs.jsonl`;
- `failures.jsonl`;
- `minimized_traces/`;
- `summary.csv`;
- `decision.json` and `decision.md`.
