# E-31: Versioned commit for mutable sessions

Status: KILL-DESIGN. NOT RUN.

Do not run this simulator. R-23 found four blocking design faults.
R-24 found that the public failures use narrow fixes on current vLLM.
The source audit also found a stronger testing question.

The rejected code remains here as a research record.
Use `run_source_audit_docker.sh` to repeat the read-only source audit.

Idea: I-122. Date: 2026-10-17.

This experiment is a mechanism screen. It does not execute vLLM or a model.
It tests whether simple cache invalidation covers source-derived state races.

## Research question

Can a generation tag and a commit frontier prevent stale cache, work, parser,
and materialized-state publication after a live session update?

## Policies

1. `mutable` changes live state without a generation fence.
2. `hash_only` invalidates cache identity but does not fence other consumers.
3. `epoch_only` adds a generation fence but keeps one conflated progress frontier.
4. `abort_rebuild` creates a new request incarnation and rebuilds the state.
5. `synchronous` drains queued work and output before an update.
6. `versioned` tags work and events, then rejects stale generations.

## Inputs

The source-derived scenarios abstract these public vLLM reports:

- issue 49449: a non-append update retains a stale block hash;
- issue 36755: asynchronous output reads state after later mutation;
- issue 42655: repeated streaming updates retain stale row identity;
- issue 58776: logical-token and materialized-KV frontiers diverge.

The scale sweep uses synthetic sessions. It varies these values:

- sessions: 1, 10, 100, and 1,000;
- input length: 16 and 256 tokens;
- update probability: 1%, 5%, and 20% per turn;
- preemption probability: 0% and 10% per turn;
- outstanding work depth: 1 and 4;
- repetitions: 5;
- turns per session: 20.

The bounded state search uses one session and a maximum history depth of six.

## Invariants

1. A published cache key identifies its materialized token sequence.
2. Completed work matches the current session revision and frontier.
3. An emitted parser event matches the current session revision.
4. Preemption does not replace logical history with materialized progress.
5. A stale generation cannot publish externally visible state.

## Frozen decision rules

These thresholds are project heuristics.

1. Mark `INVALID_REFERENCE_MODEL` if a conservative reference baseline violates a source invariant.
2. Mark `KILL_MECHANISM` if `versioned` violates an invariant.
3. Mark `KILL_SIMPLE_BASELINE` if `hash_only` handles every source scenario.
4. Mark `PIVOT_EPOCH_ONLY` if one generation tag handles every source scenario.
5. Mark `PIVOT_TRIVIAL_REBUILD` if full rebuild adds at most 3% synthetic token work in every 1% and 5% update cell.
6. Mark `PIVOT_WEAK_PREMISE` if the mutable policy exposes fewer than two failure classes.
7. Mark `PASS_TO_INTEGRATION` only if the versioned policy passes and hash invalidation leaves a non-cache failure.

`PASS_TO_INTEGRATION` is not a GO verdict for I-122. It permits an implementation-level test.

Wall-clock timing is report-only. The decision does not use it.

## Rubber duck before the run

1. A live update changes the meaning of the session state.
2. Cache invalidation repairs only cache identity.
3. Old queued work can still complete after the update.
4. Old parser output can still become externally visible after the update.
5. A generation tag lets each consumer reject work from an older state.
6. A full abort is correct, but it can repeat all prefix computation.
7. The expected result is that hash invalidation fixes only the stale-hash scenario.
8. The expected result is that version fencing fixes all four abstract scenarios.

## What can make this experiment wrong

- The model can encode the proposed answer into its definitions.
- A source issue can have a narrower fix than the abstract scenario.
- Real vLLM can already carry an equivalent version under another name.
- The synthetic update rates are not production workload evidence.
- Python operation counts do not predict serving overhead.
- Docker Desktop uses a Linux virtual machine on macOS.
- Other host applications can still disturb CPU timing inside the virtual machine.

For these reasons, the experiment is a SCREEN. It cannot support a production claim.

## Docker isolation

The runner uses these controls:

- a selected Python base version and a recorded built-image ID;
- no network during tests or measurement;
- a read-only container filesystem;
- a temporary `/tmp` filesystem;
- four selected virtual CPUs;
- a four-gigabyte memory limit;
- five timing repetitions;
- a timing coefficient of variation for every cell.

Docker isolates dependencies and container resources. It does not reserve physical Mac cores.
Close heavy host applications if timing matters. Correctness results do not depend on timing.

## Run

Start Docker Desktop first. Then run:

```sh
./experiments/versioned_session_commit/run_docker.sh
```

The script creates a new timestamped directory under `results/`. It never overwrites an earlier run.

## Outputs

- `manifest.json`: parameters, environment, image ID, and source hashes;
- `source_scenarios.jsonl`: the source-derived matrix;
- `model_check.json`: bounded state-search results;
- `violations.jsonl`: minimal and sampled counterexamples;
- `throughput.csv`: raw repeated measurements;
- `summary.csv`: medians and timing variation;
- `decision.json` and `decision.md`: the frozen verdict.
