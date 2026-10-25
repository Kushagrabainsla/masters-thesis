# Experiment standards

Read these before you plan, run or analyze any GPU experiment in this repo. They apply to people and
to AI agents.

| File | Read it for |
|---|---|
| `PROTOCOL.md` | The ten stages (S0 to S9), their gates, the agent rules and the abort rules |
| `MEASUREMENT.md` | Noise control, A/A runs, intervals, the compare verdicts and the run manifest |
| `RUNPOD.md` | Credentials, booking, rehearsal, run checkpoints, teardown and the incident playbook |
| `templates/PREREGISTRATION.md` | The design document of a new experiment |
| `templates/RUN_REPORT.md` | The result write-up after a run |

Tools:
- `experiments/ledger/ledger.py`: the append-only run ledger (`runs.jsonl`).
- `experiments/ledger/compare_runs.py`: the noise-aware comparison of two runs.
- `experiments/runpod/`: hardware record, GPU microbenchmark, live balance check, hardware log.

Short form of the flow: question, preregister, review, test locally, get approval and check the
balance, rehearse on a cheap GPU, run, copy off and record, analyze with the frozen rule, review the
result and update the journal.

`RESEARCH_JOURNAL.md` stays the source of truth. These files say how to run experiments; the journal
says which ones to run and what they found.
