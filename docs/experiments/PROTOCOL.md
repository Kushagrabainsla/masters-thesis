# Experiment protocol (GPU experiments)

Status: in force from 2026-10-25 (D-58). Every paid GPU experiment follows this protocol. An agent
that runs an experiment reads this file first, then `MEASUREMENT.md` and `RUNPOD.md`.

The protocol has ten stages. Each stage ends at a gate. Do not start a stage before the gate of the
stage before it passes. If a gate fails, stop, write down why, and tell the user.

## Principles

These are the practices that frontier ML labs use for experiments. Each rule below comes from one of
them.

1. **Preregister.** Write the question, the metrics, the decision rule and the budget before any run.
   Freeze them with a git commit. A rule that changes after the data is seen is a new, labelled
   amendment, and its result is exploratory.
2. **Review before you spend.** Every design gets a rubber-duck pass and a hostile review by a
   different model (D-29). Fix the findings before the run.
3. **Run cheap first.** Unit tests, then a Mac dry run, then a stub run of the session script, then a
   short rehearsal on a cheap GPU, then the real run. Each step catches a different failure.
4. **Measure the noise before the effect.** Run the baseline twice (A/A). A difference smaller than
   the A/A spread is not a result.
5. **Change one thing at a time.** Each comparison changes one factor. Everything else is the same
   config, the same Pod and the same software.
6. **Make every run reproducible.** Record the code commit, image tag, model revisions, seeds, config,
   hardware fingerprint and price. A run without its manifest does not count.
7. **Never lose or edit raw data.** Results are append-only. A correction is a new ledger line that
   supersedes the old one, and a retraction goes in the journal Section 8.
8. **Control the money.** Every run has a time cap, a cost cap, a watchdog and the user's approval.
   Read the live balance before each booking.
9. **Separate decision from exploration.** Only the preregistered rule makes a decision. Everything
   else is labelled exploratory.
10. **Write it down the same day.** The result, the post-run review and the journal update finish
    before the next experiment starts.

## Stages and gates

| Stage | Work | Gate (all must be true) | Output |
|---|---|---|---|
| S0 Intake | Link the experiment to one idea (I-nn). Read the journal Sections 0 and 1 and the idea dossier. Write the one question. | The question is falsifiable, and a "no" answer is defined | E-nn id reserved in the journal |
| S1 Preregistration | Fill `templates/PREREGISTRATION.md` in `experiments/<name>/README.md` | Every field is filled; the decision rule names GO, PIVOT and KILL; the budget has a time cap and a cost cap | Frozen design commit |
| S2 Design review | Rubber duck, then a hostile review by a different model (R-nn). Apply the fixes as amendment A1, A2, and so on | Verdict RUN_AS_IS, or AMEND_THEN_RUN with every finding fixed or answered in writing | `dossier/REVIEW_<E-nn>_PRERUN_<date>.md` |
| S3 Local verification | Unit tests; Mac dry run (CPU image) of each script; stub run of the session script (resume, caps, failure of one run); a timing test of every per-item loop at the real data size | All tests pass; the dry run and the stub run end with the expected files; no loop is slower than linear at the real size (E-38) | Test record in the README |
| S4 Approval | Quote the plan, the hardware, the price from a live catalog read, the time cap and the cost cap. Read the live balance (`experiments/runpod/balance.py --need <cap>`) | The user approved this spend in chat (D-nn). The balance check passed | D-nn "Decided by the user" |
| S5 Rehearsal | 20 to 30 minutes on a cheap GPU (L4 or RTX 4090). Check every item marked VERIFY | Every VERIFY item passed, or the design was amended for it | Rehearsal note in the README |
| S6 Run | Book the Pod, save the Pod record, start the session in the background, watch the checkpoints in `RUNPOD.md` | The session ended, or an abort rule stopped it and the reason is logged | Raw results on the volume disk |
| S7 Teardown and record | Copy the results off, check the checksums, terminate the Pod, read billing, add the hardware log row and the ledger rows | Results are on the Mac; billing shows no running charge; the ledger has every ok run | `HARDWARE_LOG.md` row; `experiments/ledger/runs.jsonl` rows |
| S8 Analysis | Apply only the frozen rule. Compare runs with `compare_runs.py`. Label everything else exploratory | The decision is one of the preregistered outcomes, or INVALID with a reason | `results/.../summary.json`, run report |
| S9 Post-run review and write-up | A different model reviews the result. Fill `templates/RUN_REPORT.md`. Update the journal (E-nn result, idea dossier, Section 1, Section 4, D-nn) | The review verdict is recorded; the journal is updated; the work is committed | `dossier/REVIEW_<E-nn>_POSTRUN_<date>.md` |

## Rules for agents

- An AI recommendation is not a decision. Only the user approves spend, a change of plan or a kill.
  Log who decided in the journal Section 10.
- Do not book a Pod, change a frozen rule or skip a gate on your own. If a gate cannot pass, stop and
  report.
- Do not change code between the rehearsal and the run, except a fix that the rehearsal required.
  Commit that fix and record it as an amendment.
- Do not read results before the run ends and then change the plan. If the plan must change during
  a run (for example, an abort), log the reason first.
- Never print, log or commit a credential. Credentials live in the repo's gitignored `.env`.
- The Runpod account is shared (D-59). Label and register every thesis resource, and change only
  resources that `experiments/runpod/pods.py check` accepts (`RUNPOD.md` Section 0).
- If anything in this protocol conflicts with the journal, the journal wins. Tell the user about the
  conflict.

## Abort rules (any run)

Stop the run, keep the data, and tell the user if:
1. The GPU in the hardware record is not the GPU in the plan.
2. The cost so far plus the cost of the remaining plan is above the approved cap.
3. The reference run fails twice.
4. The validity checks of the design fail (for example, the reference does not agree with itself).
5. The balance falls below the run cap plus $35.

## Where things live

| Thing | Place |
|---|---|
| Research state, decisions | `RESEARCH_JOURNAL.md` |
| Experiment design and scripts | `experiments/<name>/` |
| Measurement rules | `docs/experiments/MEASUREMENT.md` |
| Runpod operations and incidents | `docs/experiments/RUNPOD.md` |
| Templates | `docs/experiments/templates/` |
| Run ledger and compare tool | `experiments/ledger/` |
| Hardware log, Runpod tools, thesis Pod registry | `experiments/runpod/` (`pods.jsonl`) |
| Reviews | `dossier/REVIEW_*.md` |
| Credentials | `.env` (gitignored) |
