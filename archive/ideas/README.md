# Research ideas

Two live candidates. One brief each. Full evidence, experiments and the record of
what was killed live in `../dossier/RESEARCH_GAP_DOSSIER_2026-09-16.md`.

| # | Idea | Gap in one line | Status | Blocked on |
|---|---|---|---|---|
| [01](01-state-reuse-soundness.md) | Soundness of per-layer-group state reuse | No serving engine states the rule for when reusing cached state per layer type is safe | **GO** | nothing |
| [02](02-collective-engine-arbitration.md) | Compute-aware collective engine choice | NCCL picks SM versus copy engine from a static flag and never measures what the GPU is busy doing | **MEASURE-FIRST** | multi-GPU NVLink node |

## How to choose

Idea 1 is the recommended thesis. It runs entirely on the MacBook, the feasibility
gate has passed, and it has both a correctness result and a design-space result.

Idea 2 is the better paper and the better career fit, and its gap is already proven
from NCCL source. Only its magnitude is unknown, and measuring that needs hardware
that is not available. Keep asking for the node. If it lands in the next few weeks,
switch.

Do not attempt both.

## Closed, with evidence. Do not reopen without new information

| Idea | Why it died |
|---|---|
| Reuse full-attention state to speed up recurrent backfill | Measured flat at zero across an 8x context range; analytical ceiling under 1.2% falling to 0.18% at 128K. Upstream already shipped the sound version, which loads state rather than recomputing it |
| Cache post-block activations to accelerate replay | Optimally placed recurrent checkpoints win at every realistic memory budget, across 3 models, 3 context lengths and 4 divergence distributions |
| Multi-class state budget planner | Two independent methods agree that simple policies capture nearly all the benefit. The divergence distribution does not even change which strategy wins |
| Layer-layout co-design to unlock divergent reuse | 0 of 13 real hybrid models have a favourable layout, and 8 of 13 have nothing skippable. Fixing it means a layout change that needs pretraining to validate. Survives only as a finding inside Idea 1 |
| Typed or heterogeneous state runtime as a broad thesis | Current vLLM already has typed cache specs, multiple cache groups and hybrid coordinators |

## Reproducing anything

Experiment code and results: `../experiments/`. Result bundles:
`../dossier/bundles/`. The reproduction commands are in section 8 of the dossier.

Note that `.venv` in each experiment directory is a symlink to
`~/Downloads/mac_hybrid_selective_recovery_poc/.venv`, the only environment on this
machine with torch and transformers. The default system `python3` has neither.
