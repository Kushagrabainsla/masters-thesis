# Research Status: Read This First

Updated: 18 September 2026

## One-minute summary

The recommended thesis is **soundness of heterogeneous state reuse in hybrid LLM
serving**. The core result is that a reuse plan must be input-closed: every layer
that executes at a position must receive a predecessor output either by executing
the predecessor or by reading a cached activation. This catches unsafe plans that
reuse one layer group farther than another.

The result is already supported by a CPU-only gate against a pinned vLLM checkout,
measured MPS controls, and a 13-model layout survey. The project is ready for
formalization and write-up. A short GPU run would strengthen it but is not required
to begin.

The alternative thesis, **compute-aware NCCL collective-engine arbitration**, has
stronger GPU-systems novelty but is blocked until an NVLink GPU node is available.
Its architectural gap is confirmed from NCCL source; only the performance value is
unknown.

## What has been established

### RQ1: heterogeneous state-reuse soundness — GO

- The static checker identifies the exact layers starved by an unsafe reuse plan.
- It flags the vLLM #53041-style “skip full-attention layers, rebuild recurrent
  state” plan as unsound.
- It agrees with all three measured controls: one unsound plan and two sound plans.
- It reproduces vLLM #45238's divergent per-group cache hits on CPU.
- The 13-model layout survey found no recurrent-first model and no model where more
  than half of full-attention replay work is safely skippable.
- The useful constructive optimization is to skip layers above the last
  state-holding layer during backfill. It preserves recurrent state and gives a
  small, measurable saving on favorable layouts.

Evidence: [soundness gate](../experiments/rq1_soundness_gate/results/soundness_gate.json),
[layout survey](../experiments/rq3b_layout_survey/results_layout_survey.json),
[Idea 1](../ideas/01-state-reuse-soundness.md).

### RQ2: compute-aware collective arbitration — MEASURE-FIRST

NCCL's SM-versus-copy-engine decision is controlled by static policy and static
eligibility checks. The decision point and tuner interface do not receive live SM
occupancy or memory-bandwidth pressure, and NCCL does not collect those signals for
this purpose.

The decisive experiment needs an NVLink-connected multi-GPU node:

1. Run an SM-bound GEMM and a memory-bandwidth-bound streaming kernel.
2. Overlap each with AllGather and AllToAll.
3. Compare `DEFAULT`, `EFFICIENCY`, and `ZERO` CTA policies across message sizes
   and channel counts.
4. Compare the best per-cell policy with the best single static policy.

If the static policy is within noise of the per-cell oracle, kill the idea. If not,
the result supports adaptive engine arbitration.

Evidence: [Idea 2](../ideas/02-collective-engine-arbitration.md),
[NCCL audit](../experiments/rq2_nccl_source_audit/results/nccl_ce_selection_audit.json).

## What is closed

Do not spend more time on these without new evidence:

- Frozen-K/V replay as a performance mechanism: measured benefit is flat at noise.
- Activation-output caching: dominated by recurrent checkpoints at realistic budgets.
- A general multi-class state-budget planner: simple policies win across tested
  models, contexts, and divergence distributions.
- Layer-layout co-design: validating a new layout would require pretraining.
- LLM queueing-boundary idea: a 2026 paper already develops the stronger dynamical
  model.

## What we can do next

### Immediately, on the Mac

1. Turn the input-closure rule into a formal definition and proof.
2. Add more plan-level cases covering mixed block sizes, DCP, speculative decoding,
   offload, and multi-group layouts.
3. Apply the checker to additional shipped or proposed vLLM mechanisms.
4. Package the checker and the safe suffix optimization as a reproducible artifact.
5. Draft the thesis proposal around RQ1, including the negative design-space result.

### If GPU access appears

- Run a small CUDA confirmation for RQ1's selected cases.
- Run the predeclared NCCL arbitration matrix. Do not switch the thesis to RQ2
  before this experiment produces a meaningful oracle gap.

## Decision

Proceed with **RQ1 as the thesis** unless reliable NVLink hardware access appears.
Keep RQ2 warm as the higher-upside systems paper, not as a second simultaneous
thesis track.

For the full record, read
[RESEARCH_GAP_DOSSIER_2026-09-16.md](RESEARCH_GAP_DOSSIER_2026-09-16.md).
