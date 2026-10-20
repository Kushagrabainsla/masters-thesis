# I-122 source audit and pivot review

Date: 2026-10-17 in America/Los_Angeles.

Scope: I-122, E-31, and the proposed I-131 pivot.

## Verdict

- Mark E-31 as INVALID DESIGN. Do not run it.
- Mark I-122 as PIVOT. The failures are real, but the global version mechanism lacks evidence.
- Test I-131 as a new MEASURE-FIRST idea.

I-131 is stateful metamorphic testing for mutable inference sessions.

## R-23 different-model review

FACT: Sonnet found four blocking design faults before E-31 ran.

1. The version guard and invariant check used the same predicate.
2. A policy-name branch encoded the expected preemption result.
3. The safety checks allowed valid work loss and permanent stalls.
4. Model constants fixed the rebuild-cost result.

FACT: The reviewer also found that the scenarios did not represent all four cited issues.

FACT: The reviewer recommended a current-source audit before any simulator run.

INTERPRETATION: E-31 cannot falsify its proposed mechanism.

Full review: `dossier/REVIEW_E-31_PRERUN_2026-10-17.md`.

## R-24 current-source audit

The audit ran in Docker with four virtual CPUs and 4 GB of memory.
It used the local `python:3.12-slim` image.
It fetched public GitHub data through the container network.

The audit pinned vLLM main at commit
`da9134bc89364465f68aa679564435936d157a14`.

### Source ledger

| Work | State on 2026-10-17 | What it solves | What it does not solve | Threat |
|---|---|---|---|---|
| vLLM issue 49377 and PR 59103 | Closed; fix merged 2026-10-06 | Drops block hashes for truncated session tokens | It does not define a cross-component transaction | High threat to the stale-hash premise |
| vLLM issue 36755, PR 36812, and PR 36859 | Closed; related fixes merged | Repairs negative metrics and external-token accounting | The fixes are local and the first is a band-aid | Medium threat; E-31 modeled a different race |
| vLLM issue 42655 and PR 43683 | Issue closed; proposed fix closed without merging | Repairs GPU model-runner row mapping after repeated streaming updates | The failure is outside the CPU scheduler pilot | Not an eligible I-131 CPU pair |
| vLLM PR 58259 | Merged 2026-09-28 | Clears stale speculative state and stale in-flight output at one handoff point | It does not add a session generation | High threat to a broad version mechanism |
| vLLM issue 58776 and PR 58778 | Closed; fix merged 2026-10-17 | Separates valid token history from recomputation progress during preemption | It uses a local token boundary | High threat to a new global commit frontier |
| vLLM issue 60744 and PR 60764 | Open; fix PR open | Reclaims KV from parked sessions before a prefix-cache reset | The fix is not merged | Evidence that transition bugs continue |
| A First Look at Bugs in LLM Inference Engines | TOSEM 2026 | Classifies 929 bugs from five engines | It does not build a state-transition tester | Strong adjacent work and direct motivation |

FACT: Issue 49449 remains open and reports vLLM 0.25.1 only.

FACT: A contributor identified issue 49449 as a likely duplicate of issue 49377.

INTERPRETATION: PR 59103 probably fixes the reported cause on current main.
This audit did not rerun the GPU reproducer.

FACT: Current main contains `session_id`, `streaming_queue`, `sched_step_seq`, and
`processed_step_seq` fields in the audited paths.

FACT: The same paths contain no token named `generation`, `revision`, `incarnation`,
`epoch`, or `commit`.

INTERPRETATION: The absence of a version field does not prove that one is needed.

FACT: The selected scheduler and streaming suites contain 185 test functions and
54 parameterization markers.

FACT: These seven files contain no Hypothesis import and no random-action use.

INTERPRETATION: Current tests are broad but example-based in this path.
This source scan does not measure all vLLM testing infrastructure.

### Attacks on I-122

Exact-work attack: No audited field implements the proposed full session generation.
This result alone does not establish novelty or need.

Simple-baseline attack: Narrow fixes solved each closed source failure.
The fixes invalidate block hashes, clear stale state, or select a correct token boundary.

Combination attack: A global generation tag combines known epoch fencing with session state.
The current evidence does not show a constraint that requires this combination.

Relevance attack: Issue 60744 shows that mutable-session transition failures continue.
This supports the problem class, not the proposed mechanism.

Scoop-risk attack: vLLM fixes these issues within days or weeks.
A mechanism tied to one current failure has high scoop risk.

### I-122 decision

PROPOSED CONTRIBUTION, rejected: Add one generation and commit frontier across all session state.

Reason: The evidence does not show that narrow local invariants fail.
Recent fixes use narrow local invariants successfully.

Verdict: PIVOT.

## I-131 candidate

### Secret

Conventional assumption: Example-based scheduler tests cover session transitions when each bug gets a regression test.

Possible non-consensus truth: Mutable sessions create interaction faults that require generated action sequences and semantic equivalence oracles.

### Problem

Streaming updates, preemption, cache reset, speculative output, and queue movement can interact.
Each feature can pass its own tests while a short transition sequence fails.

### Evidence

FACT: PR 58259 fixed one handoff race on 2026-09-28.

FACT: Issue 58776 opened three days before PR 58259 merged and closed on 2026-10-17.

FACT: Issue 60744 remains open and has a CPU-only reproducer.

FACT: Current selected tests use 185 examples but no stateful property generator.

FACT: The TOSEM study found 929 real bugs and states that more than 35% had non-crash symptoms.
It recommends diverse oracles and metamorphic testing across backends.

### Closest prior work

The TOSEM bug study defines bug classes and recommends automated oracles.
It does not implement session-transition generation.

Stateful property testing, model-based testing, and trace-guided generator synthesis are established methods.
They are strong adjacent-domain threats.

Current vLLM tests provide strong real-system baselines.
They do not appear to generate action sequences in the audited session path.

### Exact remaining question

I did not find public work that tests mutable LLM inference sessions with generated transition sequences and history-equivalence oracles.
The contribution cannot be generic property testing applied to vLLM.
It must define useful domain semantics and find failures missed by strong tests.

### Research question

Can stateful transition generation with history-equivalence oracles find more real mutable-session faults than current tests and unguided random actions?

### Hypothesis

HYPOTHESIS: A domain-aware generator will rediscover at least four known bugs with short traces.
It will outperform an equal-budget random-action baseline.

### Negative result

Kill the idea if ordinary random actions find the same failures at the same cost.
Kill the idea if valid behavior causes frequent oracle failures.
Kill the idea if the generator needs issue-specific sequences.

### Cheapest experiment

Run current vLLM scheduler tests in the official ARM CPU container.
Add one history-equivalence oracle and a short action generator.
Use only scheduler objects and fake model-runner output.

Start with the open issue 60744 and six known fixed bugs.
Run each buggy parent commit and each fixed commit.
Measure rediscovery, trace length, executions per second, and false positives.

### Predeclared screen criteria

These thresholds are project heuristics.

GO:

- Rediscover at least four of six known bugs without issue-specific traces.
- Find at least two bugs that an equal-budget random baseline misses.
- Produce no false positive in 1,000 control sequences.
- Minimize each rediscovered failure to at most 12 actions.

PIVOT:

- Find real failures, but only within one fixed feature pair.
- Keep the result as a vLLM testing tool or upstream contribution.

KILL:

- Current tests or unguided random actions match the result.
- More than 1% of valid control sequences fail the oracle.
- The Mac Docker environment cannot execute 100 sequences per second.

### Biggest novelty threat

The strongest threat is the combination attack.
Generic stateful testing plus vLLM is not a systems contribution.

### Product opportunity

User: Inference-engine maintainers and teams that carry private forks.

Product: A CI service that runs semantic transition tests against each engine update.

Wedge: A container that tests one vLLM commit without a GPU.

Moat: A cross-engine bug corpus, oracles, and minimized transition traces.

### Confidence

Confidence that the public gap exists: medium-low.

The test absence is a source scan, not proof of absence.
The TOSEM paper already calls for related testing methods.

## Artifacts

- `experiments/versioned_session_commit/source_audit.py`
- `experiments/versioned_session_commit/run_source_audit_docker.sh`
- `experiments/versioned_session_commit/results/source-audit-20261018T035824Z/audit.json`
- `experiments/versioned_session_commit/results/source-audit-20261018T035824Z/manifest.json`
