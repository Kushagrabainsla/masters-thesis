# M.S. Thesis Research: Systems for LLM Inference and GPU Efficiency

Kushagra Bainsla, M.S. Computer Science, San Jose State University.

This repository holds the research record for my M.S. thesis. The thesis direction is not chosen yet. The repository records every candidate idea, the experiments that tested them, the adversarial reviews, and the reasons each idea was kept or killed.

## Status (2026-10-11)

- No thesis direction is chosen.
- The strongest open candidate is **I-23 (SafeDispatch)**. The question: when a faster GPU kernel replaces a slower one inside an LLM inference server, does the model output change, and how much speed can a correctness-aware kernel choice recover? The scope is Hopper and Blackwell block-scaled FP8, NVFP4 and MoE kernels.
- I-23 needs an NVIDIA GPU. The evaluation harness is built and validated on a laptop (`experiments/safedispatch_oracle/`). The first GPU run is Test A.
- Many other ideas were tested on a laptop and killed. Each kill has a written reason and kill criteria that were fixed before the run.

For the full current state, read Section 1 of `RESEARCH_JOURNAL.md`.

## Where to start

| If you want to see | Read |
|---|---|
| The current state and next steps | `RESEARCH_JOURNAL.md`, Section 1 |
| The list of all ideas and their status | `RESEARCH_JOURNAL.md`, Section 3 (idea registry) |
| One idea in detail | `RESEARCH_JOURNAL.md`, Section 5. Search by ID, for example `I-23` |
| One experiment or review | `RESEARCH_JOURNAL.md`, Section 6. Search by ID, for example `E-28` or `R-16` |
| Decisions and who made them | `RESEARCH_JOURNAL.md`, Section 10 |
| Retracted claims | `RESEARCH_JOURNAL.md`, Section 8 |

The journal is large. Search it by ID instead of reading it from start to end.

## Method

1. Write the idea as a mechanism with a clear invariant. Do not write benchmarking-only ideas.
2. Check prior art: papers, preprints, and GitHub issues and pull requests of the main serving and training systems.
3. Write the kill criteria before the experiment runs.
4. Run the cheapest experiment that can kill the idea. Most experiments run on a MacBook.
5. Get an adversarial review from a different model before and after the run.
6. Record the result, the review and the decision in the journal. Never delete history. Retractions go in Section 8.

## Repository layout

| Path | Contents |
|---|---|
| `RESEARCH_JOURNAL.md` | The single source of truth: ideas (`I-nn`), experiments (`E-nn`), reviews (`R-nn`), decisions (`D-nn`), prior art and timeline |
| `experiments/` | Code, inputs and results for each experiment. Some folders have their own README |
| `dossier/` | Long-form literature reviews, route searches and adversarial reviews. Each file name has its review ID and date |
| `grants/` | Drafts of GPU compute applications (NAIRR Pilot, NSF ACCESS, NVIDIA Academic Grant) |
| `archive/` | Superseded files. `archive/README.md` maps each file to its replacement |
| `AI_Infra_Thesis_Handoff_Package/`, `HANDOFF_2026-10-08.md`, `*.docx` at the root | Older snapshots of the research state. If they disagree with the journal, the journal is correct |
| `agents.md` | Instructions for AI assistants that work in this repository |

## Experiments

| Folder | Idea | What it tests | Result |
|---|---|---|---|
| `safedispatch_oracle/` | I-23 | Laptop validation of the output-equivalence harness: bitwise gate, greedy-flip detection, flip-rate estimator | Harness passed; GPU test pending |
| `metastable_admission/` | I-80 | Trace-driven simulator with 459 real agent sessions (E-28) for metastability-aware admission control | Killed (with scope limits) |
| `noise_shaped_vcache/` | I-58 | Sigma-delta (error-feedback) quantization of the V cache along the token axis (E-27) | Killed |
| `structurer_ingest_screen/` | I-41 | A small "structurer" LLM in front of the agent model (E-24 to E-26) | Killed |
| `hybrid_state_recovery_v1_3/` | I-13, I-16, I-17 | Recovery cost of hybrid attention and recurrent state, FLOP model, budget trade-offs (E-12 to E-17) | Killed |
| `rq1_soundness_gate/`, `rq2_nccl_source_audit/`, `rq3_state_budget/`, `rq3b_layout_survey/` | I-19, I-18, I-15, I-20 | Early research-question checks: state-reuse soundness, NCCL copy-engine selection, state budgets, cache layouts | See the journal |

Each experiment has a matching `E-nn` entry in the journal with the setup, the kill criteria, the raw results and the review.

## Reproduction

Most experiments run with Python 3 and PyTorch on a MacBook. If a folder has a `requirements.txt`, install it in a virtual environment first:

```bash
python3 -m venv .venv
```

Some log files and scripts contain absolute paths from my laptop. Change them to your local paths before you run the scripts. One large result file uses Git LFS. Install Git LFS before you clone to get it.

Section 9 of the journal lists the commands, the pinned versions (for example the vLLM commit) and the result files for each experiment.

## Notes

- Files in `grants/` are drafts. They are not the submitted versions.
- Some text in the journal and dossiers was drafted with AI assistants. AI recommendations are not decisions. Section 10 of the journal records who made each decision.
