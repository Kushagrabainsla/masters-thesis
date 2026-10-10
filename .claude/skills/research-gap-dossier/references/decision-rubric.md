# Research Candidate Decision Rubric

Use this rubric after the two adversarial passes. Scores are aids, not substitutes for evidence.

## Core dimensions

| Dimension | What strong evidence looks like |
|---|---|
| Problem reality | Production traces, measured bottleneck, documented failure mode, or repeatable workload behavior |
| Novelty | Closest public work is identified and the surviving gap is technically specific |
| Non-triviality | Strong simple baselines do not erase the opportunity |
| Research depth | Requires a new mechanism, model, consistency protocol, scheduler policy, or empirical systems finding |
| Evaluation quality | Clear baselines, metrics, workloads, and falsifiable claims |
| Feasibility | Decisive early test is cheap; full evaluation fits available time/resources |
| Scale validity | Claim survives realistic concurrency/topology/multi-tenancy, or scope is explicitly local |
| Publication potential | Contribution is more than implementation/integration and can be articulated as a general result |
| Open-source value | Produces a reusable system, benchmark, trace tool, or measurement harness |
| Career relevance | Builds scarce skills in the user's intended technical direction |
| Scoop risk | Gap is not obviously being closed by an active upstream roadmap in the next 6-12 months |

## Hard gates

A candidate should not be finalized if any of these remain unresolved:

1. The closest three pieces of prior work have not been read closely.
2. The strongest production implementation has not been checked.
3. The exact unanswered sentence cannot be written in one or two lines.
4. The strongest simple baseline has not been identified.
5. No decisive early experiment exists.
6. The result cannot be falsified.
7. The evaluation requires hardware/data that are realistically unavailable.

## Status definitions

- **GO**: gap is defensible and decisive validation is positive enough to justify system work.
- **PIVOT**: domain is promising but the current mechanism/question is weak or already answered.
- **MEASURE-FIRST**: the opportunity itself is not established; run a workload/oracle study before building.
- **KILL**: strong evidence says the gap is solved, trivial, too small, or infeasible.

## Confidence language

Use conservative confidence:
- **High**: multiple current searches and adjacent checks found no direct answer, and the technical distinction is clear.
- **Medium**: gap appears real but fast-moving repos, adjacent work, or unpublished industry systems create uncertainty.
- **Low**: novelty boundary is thin, largely compositional, or dependent on an unvalidated workload assumption.
