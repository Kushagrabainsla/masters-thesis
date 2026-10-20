# E-34 pre-implementation review

Act as a hostile OSDI, SOSP, MLSys, or EuroSys reviewer.

Do not execute code.
Do not edit files.
Use only read, grep, and glob operations.

Read these files in full:

- `dossier/SCOOP_AUDIT_I-123_2026-10-19.md`
- `experiments/crash_consistent_duplex_epoch/README.md`

Read the I-123, I-132, E-34, R-22, and R-28 entries in `RESEARCH_JOURNAL.md`.

Attack these questions.

1. Is I-132 only Apache Flink checkpointing plus a model snapshot?
2. Does vLLM-Omni already expose every required commit frontier?
3. Does Concordia already supply the missing durable recovery epoch?
4. Does the exact remaining gap support a thesis or only an integration patch?
5. Can full per-unit replication solve the supported workloads?
6. Can replay from the session start solve the supported workloads?
7. Does the 200 ms target have any valid role after R-28?
8. Can the Mac Docker test falsify the important systems claim?
9. Are B1 and B2 correct and competitive baselines?
10. Does P1 receive an unfair contract or recovery target?
11. Is the independent oracle actually independent?
12. Can the crash schedule miss an important ambiguous-output cut?
13. Can stable identifiers hide loss, reordering, or payload changes?
14. Do the decision rules allow a synthetic result to become a production claim?
15. What is the cheapest experiment that can kill I-132?

Classify each finding as BLOCKER, MAJOR, or MINOR.

Separate facts from interpretations. Give file and line references.

End with one verdict.

- `SAFE_TO_IMPLEMENT`
- `AMEND_BEFORE_IMPLEMENTATION`
- `KILL_IDEA`

If amendments are required, give the smallest exact amendment set.

Use simple technical English.

