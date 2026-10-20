# E-32 amendment review

Act as a hostile OSDI, SOSP, MLSys, or ISSTA reviewer.

Do not execute code.
Do not edit files.
Use only read, grep, and glob operations.

Read these files in full:

- `experiments/session_state_metamorphic/README.md`
- `experiments/session_state_metamorphic/bug_corpus.json`
- `dossier/SOURCE_AUDIT_I-122_2026-10-17.md`
- `dossier/REVIEW_E-31_PRERUN_2026-10-17.md`
- `dossier/REVIEW_E-32_PRERUN_2026-10-17.md`

Read the I-122, I-131, E-31, E-32, R-23, and R-24 entries in
`RESEARCH_JOURNAL.md`.

Attack these questions:

1. Is I-131 only generic property testing plus vLLM?
2. Does the evidence show a recurring problem after narrow fixes?
3. Are the four frozen corpus items distinct and reproducible on CPU?
4. Does a corpus built from known bugs make the test circular?
5. Are the metamorphic relations valid for every generated action?
6. Can the harness report false positives after legal preemption or reset?
7. Are current vLLM tests and equal-budget random actions strong baselines?
8. Can the full generator pass by encoding the four known traces?
9. Do the GO, PIVOT, and KILL rules decide the question fairly?
10. Is the amended speed treatment fair on this Mac?
11. Is a current vLLM CPU image or build feasible with 8 GB of Docker memory?
12. What is the cheapest experiment that can kill I-131?

Classify each finding as BLOCKER, MAJOR, or MINOR.
Separate facts from interpretations.
Give file and line references.

Review the amendments against the earlier Sonnet findings.
Do not repeat the earlier review without checking the new corpus and decision rules.

End with one verdict:

- `SAFE_TO_IMPLEMENT`
- `AMEND_BEFORE_IMPLEMENTATION`
- `KILL_IDEA`

If amendments are required, give the smallest exact amendment set.
Use simple technical English.
