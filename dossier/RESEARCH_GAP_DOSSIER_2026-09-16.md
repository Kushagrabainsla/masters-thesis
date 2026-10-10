# Research Gap Dossier: Iteration 4

Continuation of `AI_Infra_Thesis_Handoff_Package`. Produced 16 September 2026.
Scope: finish the hybrid-state falsification the handoff queued, run the parallel
collective-engine pass, and return a short list of workable research questions.

All numbers below are either (a) measured on this MacBook against the real model,
(b) computed analytically from the real model's module shapes, or (c) quoted from a
primary source. Each claim is labelled. Nothing here reuses the invalidated v1.1
frozen-KV timings.

---

## 0. Headline

The handoff's surviving hybrid-state question is now **killed**, on performance
grounds, with three independent quantitative arguments. The work produced two
reusable assets in the process: a divergent-state replay path that is numerically
exact, and an executable demonstration that the mechanism an active vLLM design
discussion proposes is unsound for interleaved hybrid stacks.

That second asset, not the performance idea, is what survived. The strongest
remaining direction is the **correctness** of heterogeneous per-group state reuse,
not its speed.

Since then, every remaining Mac-testable gate has been run (section 10). Outcome:
RQ1's feasibility gate **passes**, RQ3 is **killed**, and RQ2's architectural gap is
**confirmed from NCCL source** without any GPU. Two candidates remain, and the
choice between them is now a question about hardware access, not about evidence.

---

## 1. What was run this iteration

| # | Experiment | Status | What it settled |
|---|---|---|---|
| 1 | v1.2 explicit-mask fix (was unrun in handoff) | **Valid** | Divergent-state replay is now correct: logit diff `0.0` at 512, `0.0117` noise floor to 4096, vs `1.0554` in v1.1 |
| 2 | v1.3 randomized-order sweep | **Valid** | Removes the execution-order bias that inflated every v1.2 speedup |
| 3 | Analytical FLOP/MAC ceiling model | **Valid** | Hard upper bound on the whole mechanism class, vs context length |
| 4 | Memory-budget Pareto (checkpoints vs activations) | **Valid** | Prices the only variant that could claim the large prize |
| 5 | Cross-model state-ratio analysis | **Valid** | Shows the upstream memory objection is correct at production scale |
| 6 | Novelty pass 2 (papers + upstream issues) | **Done** | Located the exact prior claim and its defect |
| 7 | RQ1 gate: plan-level soundness vs real vLLM on CPU | **PASS** | Static checker matches measurement 3/3; reproduces vLLM #45238 on a laptop |
| 8 | RQ3: optimal checkpoint placement, 3 real models | **Valid** | Kills RQ3; checkpoints win at every realistic budget |
| 9 | RQ2: NCCL 2.31.2 source audit | **Valid** | Engine choice cannot observe compute state; gap is structural |

Artifacts: `experiments/hybrid_state_recovery_v1_3/`, `experiments/rq1_soundness_gate/`,
`experiments/rq2_nccl_source_audit/`, `experiments/rq3_state_budget/`.

### 1.1 The v1.2 fix works, and it invalidates the v1.1 conclusion

The handoff shipped `mac_state_selective_recovery_fix_v1_2.zip` unrun, with the
instruction not to treat it as tested. It is now tested.

FACT. With the explicit 4-D additive causal mask for full-attention layers and a
2-D suffix padding mask for linear-attention layers, the divergent-state recovery
path reproduces normal suffix replay exactly:

```
semantic_recovery_correct            : true
max_semantic_logit_diff_vs_normal    : 0.0          (was 1.0554 in v1.1)
frozen_full_attention_kv_unchanged   : true
naive_skip_full_blocks_is_incorrect  : true
```

(Quick mode, context 512. Across the full sweep to context 4096 the figure is
0.0117, which is the floating-point reordering noise floor, not a structural error.
See section 1.6.)

INTERPRETATION. The v1.1 negative timings were an artifact of a broken mask
geometry, exactly as the handoff suspected. The sign of the result flips: the
mechanism is not slower than baseline, it is marginally faster. The handoff was
right to refuse to kill the idea on v1.1 evidence.

### 1.2 Execution order, not the mechanism, produced most of the v1.2 speedup

FACT. v1.1 and v1.2 always execute methods in the order
normal, frozen, bypass, oracle. Every speedup is measured against the method that
runs first, so warmup and allocator effects are charged to the baseline.
Randomizing method order within each repetition, holding everything else fixed:

| Method | Fixed order | Randomized order |
|---|---|---|
| `frozen_full_kv_replay` | +8.91% | **-1.65%** |
| `frozen_full_kv_bypass_kv_proj` | +9.46% | **+0.74%** |
| `oracle_cached_full_block_outputs` | +18.57% | **+10.82%** |

(Context 512, boundary 384, 2 repetitions. Low repetition count, so treat the
randomized column as a direction, not a precise value. The full 10-repetition
sweep is in section 1.6.)

INTERPRETATION. The apparent K/V-reuse benefit is mostly measurement order. This
is a methodology defect that affected every timing in the handoff's hybrid track,
because all of them used fixed method order.

### 1.3 The architectural ceiling: the prize grows, the sound mechanism does not

A mechanism that reuses already-valid full-attention state while rebuilding lagging
recurrent state cannot remove any linear or recurrent layer work, because the
recurrent state is precisely what is being rebuilt. So its speedup ceiling is
exactly the full-attention share of replay work.

FACT (analytical, from the real model's module shapes; `flop_model.py`). Qwen3.5-0.8B,
6 full-attention layers at indices 3, 7, 11, 15, 19, 23 and 18 gated-DeltaNet layers:

| Context | Ceiling if FA blocks are skipped entirely | Ceiling for the sound frozen-KV variant |
|---|---|---|
| 512 | 23.1% | 1.22% |
| 2,048 | 28.0% | 1.14% |
| 8,192 | 41.9% | 0.92% |
| 32,768 | 67.2% | 0.52% |
| 131,072 | 88.1% | **0.18%** |

(Boundary fraction 0.75. `fa_share_of_replay_pct` and `kv_proj_share_of_replay_pct`
in `results_flop_model.json`.)

INTERPRETATION. These two curves move in opposite directions, and that is the whole
story of this research line. Long context makes full-attention work dominate hybrid
replay, so the prize is real and large. But the only thing the *sound* frozen-KV
mechanism actually avoids is recomputing K/V projections, which is 1.2% of replay at
512 tokens and falls toward zero as context grows, because attention matmul and MLP
work grow while the projection stays linear in suffix length.

To claim the large prize you must skip whole full-attention blocks. That requires
the post-block hidden states, which is a different and more expensive mechanism.

### 1.4 The mechanism that could claim the prize is already publicly proposed, and as written it is unsound

vLLM #53041, "Tiered SWA/Mamba checkpointing (HBM tail + store periodic) +
recompute backfill for divergent hybrid prefix hits", opened 20 August 2026, open,
no implementation. Its Proposal B is the mechanism this research line was pursuing.

FACT (direct quotes from the issue):

> "For the recompute range: run SWA/Mamba layers forward only (FA layers skip, KV already loaded)"

> "In both cases, FA layers skip the gap range because their KV is pre-loaded; only SWA/Mamba layers run forward."

The issue names the difficulty itself under "Technical challenge: per-layer
skip/compute", and offers two implementations: a two-phase prefill for only
SWA/Mamba layers over the gap, or a per-group `num_computed_tokens` skip flag.

FACT (measured). Taken literally, both options are semantically incorrect for an
interleaved stack. In Qwen3.5-0.8B the full-attention layers sit at indices
3, 7, 11, 15, 19, 23, so linear-attention layers 4, 5, 6 consume layer 3's output
at gap positions. If layer 3 does not execute there, its output does not exist.
The harness's `naive_identity_skip_full_blocks` control is exactly this plan, and
it fails the equivalence check as designed.

INTERPRETATION. There is a checkable soundness condition hiding here. At each
position, the set of executed layers must be *input-closed*: for every executed
layer L, either L-1 also executes at that position, or L-1's output activation is
cached. Skipping an interior layer group while running the groups above it violates
closure. Only two plans satisfy it:

1. Reuse the full-attention K/V but still run the full-attention blocks. Sound. This
   is the frozen-KV path, and its ceiling is the 0.18% to 1.22% column above.
2. Skip full-attention blocks using cached post-block activations. Sound. This is
   the oracle path, and it needs O(context) extra memory.

There is no third option. This is the load-bearing finding of the iteration.

### 1.5 Pricing option 2: it loses to simply storing more recurrent checkpoints

The simple-baseline attack on activation caching is to spend the same bytes on more
recurrent checkpoints instead. The two are not substitutes, which is why this needs
computing rather than asserting: a checkpoint shortens the replay, an activation
cache makes each replayed token cheaper.

FACT (measured byte constants plus the FLOP model; `pareto_budget.py`). Qwen3.5-0.8B,
uniform divergence depth, evenly spaced checkpoints, activation cache = 12,288 B/token,
and a **measured** recurrent snapshot of 18.84 MiB:

| Context | Budget where activations first win | Checkpoints that budget buys, alone / alongside activations | Expected replay there | Margin |
|---|---|---|---|---|
| 2,048 | 151 MiB | 8 / 6 | 171 tokens | 5% |
| 8,192 | 302 MiB | 16 / 10 | 410 tokens | 10% |
| 32,768 | 603 MiB | 32 / 11 | 1,489 tokens | 12% |
| 131,072 | 2,412 MiB | 128 / 46 | 1,425 tokens | 70% |

CORRECTION WORTH RECORDING. An earlier pass of this analysis computed the snapshot
size from an fp16 assumption and got 9.63 MiB, which made checkpointing look twice
as cheap as it is. The live cache says otherwise: the gated-DeltaNet recurrent state
is stored in **float32**, 4.0 bytes per element, even though the model runs in
float16, because SSM state accumulates and needs the headroom. The measured snapshot
is 18.84 MiB (18.00 MiB state plus 0.84 MiB conv), which reproduces the handoff
ledger's independently recorded ~19.76 MiB. All numbers above use the measured
value. This mattered: it halved the crossover budget.

INTERPRETATION. Activation caching only wins once the budget already buys 8 to 128
checkpoints per request, that is 151 MiB to 2.4 GiB of state for a single 0.8B-model
request. At realistic budgets of 1 to 4 snapshots (19 to 75 MiB) checkpoint-only wins
at every context tested. The uniform-depth, evenly-spaced assumption is the *best*
case for checkpointing, and a real heavy-tailed depth distribution would favour
checkpointing further, because checkpoints can then be placed where the probability
mass is. That is precisely what Sparse Prefix Caching's dynamic program already does.

So for this model the activation route is not worth pursuing on its own. Section 6
shows why that conclusion does not automatically transfer to production-scale hybrids.

### 1.6 Randomized-order sweep

FACT. Contexts 512 to 4096, boundaries 0.5/0.75/0.875, 10 repetitions each,
method order randomized within every repetition, 12 scenarios, zero failures.
Speedup versus normal suffix replay, median across the three boundaries:

| Context | `frozen_full_kv_replay` | `frozen_full_kv_bypass_kv_proj` | `oracle_cached_full_block_outputs` |
|---|---|---|---|
| 512 | -0.10% | +0.38% | +10.77% |
| 1,024 | +0.65% | +0.68% | +12.24% |
| 2,048 | -0.28% | +1.75% | +15.52% |
| 4,096 | +0.30% | -0.88% | +19.69% |

Full per-boundary range: frozen -0.95% to +1.71%, bypass -1.58% to +2.08%, oracle
+9.28% to +21.12%.

Correctness across the whole sweep:

```
semantic_recovery_correct          : true
max_semantic_logit_diff_vs_normal  : 0.0117
frozen_full_attention_kv_unchanged : true
naive_skip_full_blocks_is_incorrect: true
failures                           : []
```

INTERPRETATION, and it is the decisive result of this iteration:

1. The sound frozen-K/V mechanism is **flat at zero** across a 8x context range. It
   does not improve with scale, it does not degrade, it simply does nothing. This
   matches the analytical ceiling of 1.1% to 1.2% in that range: there was never
   more than about 1% available, and measurement noise is that large.
2. The oracle grows monotonically, 10.8% to 19.7% over the same range, tracking the
   analytical ceiling's growth (23.1% to 34.0%) while sitting below it. It sits below
   because skipping a block still costs a read of the cached activation, and because
   the unoptimized reference recurrent kernels inflate the recurrent share on this
   machine. The handoff's open question, "does oracle headroom grow with context",
   is now answered: **yes, and that is the only part of this idea worth anything.**
3. `max_semantic_logit_diff` of 0.0117 is the floating-point reordering noise floor,
   about 90x smaller than the 1.055 produced by the structurally unsound plan. This
   separation is what makes the RQ1 soundness check practical, and it is direct
   evidence for RQ1's central claim.

---

## 2. Novelty ledger, pass 2

Fresh searches, September 2026. This supersedes the handoff's ledger.

| Source | Date | What it already claims | Effect on our question |
|---|---|---|---|
| vLLM #53041 | 2026-08-20, open | Recompute backfill: keep FA KV, rebuild SWA/Mamba over the gap | **Claims our mechanism.** As written it is unsound for interleaved stacks |
| vLLM #52773 | 2026-08-18, open | Load-backfill of boundary state from external store; measured 80x cost gap vs recompute | Establishes that loading beats recomputing, so recompute backfill is a last resort |
| Sparse Prefix Caching, arXiv 2605.05219 (Shirokikh, Nikolenko) | 2026-04-17 | Sparse exact recurrent checkpoints, resume from deepest, exact suffix recompute, O(NM) DP for checkpoint placement under a depth distribution | Owns the checkpoint-placement lever, which our Pareto shows is the dominant one |
| HYPIC, arXiv 2607.01299 | 2026-07-01 | Position-independent caching for hybrid attention; segment-cumulative transition operator; "seam window" that propagates hidden states through the hybrid stack | Independently arrives at the "you must propagate hidden states" insight, for a different problem (segment concatenation, not gap backfill). 3.25x TTFT |
| ReplaySSM, vLLM #49232 | 2026-07-20, in progress | Cache SSM inputs instead of state; 1.17x to 1.67x decode, up to 2.21x speculative | Orthogonal, kernel level. Does not address FA KV reuse |
| vLLM #55697 | 2026-09-07, open | Application-directed prefix checkpoints for Mamba/hybrid | Active work on the checkpoint-placement lever. Scoop risk for RQ3 |
| vLLM #52959 | 2026-08-19, open | Internal state checkpoints for Mamba align mode | Same lever, HBM tier |
| "Illusion of Equivalence", arXiv 2604.15409 (Chodavarapu, Xu) | 2026-04-20 | KV-cached FP16 inference systematically diverges from cache-free inference, via accumulation order | Supports RQ1's premise. Measurement only, no tool, standard attention only, no multi-group |
| Collective communication survey, arXiv 2608.15118 | 2026-08 | Taxonomy of planning, runtime adaptation, computation coordination | Names RQ2 as an **open problem**, no system does it |

The shortest reviewer rejection of the original question, and it lands:
*"#53041 already proposes keeping the FA hit and rebuilding lagging state, #52773
already measured that loading beats recomputing by 80x, and your own FLOP model says
the sound version of the mechanism is worth under 1.2% and shrinking."*

---

## 3. Decision table

| Candidate | Exact surviving question | What prior work answers | Strongest threat | Next step | Status | Confidence in gap |
|---|---|---|---|---|---|---|
| Hybrid FA-state reuse during recurrent backfill | none | #53041, #52773, Sparse Prefix Caching | Own FLOP ceiling: 0.18% to 1.22% | stop | **KILL** | high |
| Activation caching as a replay accelerator | none standalone | HYPIC seam window | Pareto-dominated by checkpoints at realistic budgets | fold into RQ3 | **KILL** | high |
| General min-cost state planner | reformulated as RQ3 | 324-row normalized screen said simple policies match 86.7% | May collapse to per-model static policy | RQ3 | **PIVOT** | medium |
| **Soundness of per-group state reuse** | RQ1 below | Illusion of Equivalence (FP only); vLLM example tests | Framed as testing, not systems | gate PASSED, section 10.1 | **GO** | high |
| **Compute-state-aware collective engine arbitration** | RQ2 below | NCCL static `NCCL_CTA_POLICY`; tuner plugins | Static policy may capture most gain | gap CONFIRMED in source 10.4, size needs GPUs | **MEASURE-FIRST** | high on gap, unknown on size |
| Real-constant state-class budget allocation | none, closed | Sparse Prefix Caching DP | Optimal placement wins at every realistic budget | closed out, section 10.3 | **KILL** | high |
| Elastic reconfiguration equivalence | not formulated | DynaTrain, ElasWave, ByteCheckpoint | Crowded, and their oracle is loss-curve comparison | backup only | HOLD | low |
| Layer-layout co-design for divergent reuse | none, closed | none found | 0 of 13 real models have a favourable layout; fixing it needs pretraining | folded into RQ1, section 11 | **KILL** | high |

---

## 4. RQ1: soundness of heterogeneous per-group state reuse

**Status: GO, behind one feasibility gate.** Strongest survivor.

### Question

For hybrid LLM serving runtimes that reuse cached state per layer group, can a
checkable soundness condition on reuse plans, combined with a state-equivalence
oracle that separates structural errors from floating-point reordering noise,
certify or refute the reuse plans that current runtimes and RFCs propose, and does
applying it to shipped and proposed mechanisms find unsound plans that existing
example-based tests miss?

Falsified if: the soundness condition is either trivially equivalent to what
runtimes already assert, or too weak to catch the known bug class; or if applying
the oracle to real mechanisms finds nothing that existing tests do not already
catch.

### Why the problem is real

Five independent instances of this bug class, all from primary sources:

- vLLM #57266: with an explicit `--mamba-block-size`, `conv_ssm_forward` writes SSM
  state from an out-of-range strided slice when `num_computed_tokens` is not
  block-aligned, killing EngineCore.
- vLLM #55600: using global block geometry for Mamba/recurrent state can produce
  wrong state or illegal memory access. State-specific geometry is correctness-critical.
- vLLM #47812: hybrid recovery after KV-load failure carried single-group assumptions.
- vLLM #56655: prefix-cache keys omit the prompt-embedding mask, so valid inputs
  reuse K/V computed for a different effective input, producing wrong output.
- vLLM #53041 Proposal B: unsound for interleaved stacks, as demonstrated above.

Plus AI21's public debugging account, "One token to corrupt them all", on a vLLM
Mamba state corruption bug.

The pattern: these optimizations ship or get proposed faster than they can be
verified, the failures are silent or remote from their cause, and the existing
defence is example-based unit tests that demonstrably did not catch them.

### What is already built

The v1.3 harness is most of the oracle. It already:

- achieves exactly `0.0` max logit divergence for a sound divergent-state plan on a
  real hybrid model, which shows the oracle is tight enough to be useful;
- flags the unsound plan at logit divergence above threshold, via a deliberate
  negative control;
- diffs per-layer recurrent and conv state separately from logits;
- fingerprints the full-attention K/V before and after to prove the reused state was
  not mutated;
- refuses to report a performance number when `semantic_recovery_correct` is false.

The distinction between structural unsoundness and floating-point noise is
empirically clean here: sound plans gave `0.0`, the unsound plan gave `1.055`. That
separation is what makes the check robust, and it is the gap in "Illusion of
Equivalence", which measures the noise floor but has no tool and does not touch
multi-group reuse.

### The formal core

A reuse plan assigns each (layer, position) pair to EXECUTE or SKIP. The plan is
sound only if the executed set is input-closed at every position: for each executed
layer L at position p, either L-1 executes at p, or L-1's output activation at p is
cached. Interior skips break closure.

This is one line, it is checkable statically from a layer-type map plus a plan, and
it immediately classifies every variant in section 1.4. Its practical value is that
it converts an open upstream design debate ("is there appetite for the two-phase
prefill approach?") into a decidable question.

### Adversarial objections

1. **This is a testing paper, not a systems paper.** Real risk. Mitigation: the
   contribution is the semantics plus the finding, not the harness. The measurement
   result of section 1.3, that the sound plan's ceiling is under 1.2%, is a systems
   result derived from the semantics.
2. **Differential testing of ML systems already exists.** True for compilers and
   kernels. No public work applies it to serving-runtime state reuse plans, and none
   defines soundness for heterogeneous groups.
3. **vLLM could just write better tests.** They could. The condition is the
   contribution; the bug list shows the tests alone did not suffice.
4. **The bug class may be transient.** Counter-evidence: the interaction space is
   growing, not shrinking. Groups times features (speculative decode, DCP, offload,
   quantized K/V, retention) multiplies.
5. **Career leverage is weaker than a performance thesis.** True. State this openly
   when choosing.

### Next step, and the gate

GATE. Determine what fraction of vLLM's reuse machinery can be exercised without
CUDA. Evidence says the planning and coordinator layer can: the handoff's
`vllm_hybrid_state_gap_results_bundle` already constructed 2, 3 and 4 group
configurations, mixed block sizes, DCP full+Mamba and EAGLE 3-group layouts on this
Mac against a pinned commit. The execution layer will need a GPU. If plan-level
checking covers the #53041 and #55600 classes, the thesis is Mac-feasible with a
small GPU rental for execution-level confirmation.

Run this before committing. It is one day of work against the pinned commit
`e0705731be04aac90838f318fcdae15c56a53080`.

---

## 5. RQ2: compute-state-aware collective execution engine arbitration

**Status: MEASURE-FIRST, blocked on hardware.** Highest novelty of the three.

### Question

Can selecting the collective execution engine, SM-driven versus copy-engine or
zero-SM, from the live resource profile of the concurrently executing compute
kernel improve end-to-end overlapped throughput over both NCCL's default selection
and the best static `NCCL_CTA_POLICY`, while preserving ordering and progress
semantics?

Falsified if a static per-phase or per-layer policy captures most of the gain, or if
the oracle gain over NCCL's default is small.

### Why now, and why it is novel

FACT. NCCL 2.28 added copy-engine collectives and a device API. CE collectives
offload data movement within the NVLink domain from SMs to dedicated copy engines,
reaching zero-SM operation for AllToAll and AllGather.

FACT. The exposed control is `NCCL_CTA_POLICY` set to DEFAULT, EFFICIENCY or ZERO,
a static per-communicator setting, plus `ncclCommInitRankConfig`. NVIDIA's own
announcement does not state how the CE versus SM choice is made, and the
documentation exposes no dedicated CE selection variable. Selection is automatic on
topology and message characteristics, or tuner-driven.

FACT. The August 2026 survey (arXiv 2608.15118) names this open, in its own words:
"A major architectural limitation is their heavy reliance on GPU computing resources
(such as GPU Streaming Multiprocessors and device memory bandwidth) to drive network
progress." Its adaptive-execution section covers MSCCL++, AutoCCL and MCCS; its
coordination section covers CoCoNet, Centauri and NanoFlow. None selects the
execution engine from live compute-kernel resource state.

So the novelty claim is conservative and supportable: I did not find public work
that uses the live resource profile of the concurrent compute kernel as the input to
communication engine selection, and a current comprehensive survey lists the
underlying problem as open.

### Why it is blocked

CE collectives are NVLink-domain. The decisive experiment needs at least two
NVLink-connected GPUs, realistically an 8-GPU H100 or H200 node, to produce a
crossover that means anything. There is no NVIDIA access at present. This trips a
hard gate in the rubric: the evaluation requires hardware that is not realistically
available.

Do not select this as the thesis until access is secured. It is worth keeping warm
because the novelty is the strongest of the three.

### Predeclared decisive experiment

Write this down now so that the moment access appears, the kill test is one week.

1. Two controlled compute kernels: one SM-bound (large GEMM at high occupancy), one
   memory-bandwidth-bound (streaming triad). Verify the intended bottleneck with
   profiler counters, do not assume it.
2. Overlap each with AllGather and AllToAll under `NCCL_CTA_POLICY` DEFAULT,
   EFFICIENCY and ZERO. Sweep message size and channel count.
3. Report end-to-end overlapped throughput, not collective time alone. A CE path
   that is slower in isolation can still win by returning SMs to compute.
4. Oracle: best static policy chosen per (kernel, message size) cell with perfect
   hindsight. This is the number that matters.
5. GO if the per-cell oracle beats the best single global policy by a margin large
   enough to matter after accounting for switching cost. KILL if one static policy is
   within noise of the oracle across the grid. These thresholds are heuristics, not
   literature standards; set the exact numbers before running, not after.

Mac-side work that is possible now: read the NCCL 2.28+ source for the CE selection
path and the tuner plugin's actual inputs, and establish from source whether live
compute state is even available to the decision point. If the tuner cannot observe
concurrent kernel occupancy, that is a concrete systems gap independent of any
benchmark.

---

## 6. RQ3: real-constant state-class budget allocation

**Status: MEASURE-FIRST.** Cheapest to resolve, weakest novelty.

### Question

Given a fixed per-request byte budget for a hybrid model, does the cost-optimal
allocation across the three state classes, full-attention K/V, recurrent snapshots,
and full-attention post-block activations, vary enough across real models and
context lengths that any single static policy loses materially?

Falsified if a per-model static policy lands within a few percent of the per-case
optimum, which would reproduce the handoff's earlier planner result and kill it for
good.

### Why this is a legitimate revival, not a resurrection

The handoff deprioritized the general planner because the best simple per-row policy
matched it in 86.7% of 324 cases, with 0% median gain. That study used
*normalized* costs chosen by hand, and its own caveat says it cannot establish where
real hardware lands.

NEW FACT (this iteration, from HF configs, fp32 recurrent state as measured;
`crossmodel_state_ratio.py`). The real constants vary far more than the normalized
study could express:

| Model | FA / rec layers | One snapshot | K/V B/tok | Activation B/tok | act / K/V | Context where K/V = 1 snapshot | Context where activation cache = 1 snapshot |
|---|---|---|---|---|---|---|---|
| Qwen3.5-0.8B-Base | 6 / 18 | 18.00 MiB | 12,288 | 12,288 | 1.00x | 1,536 | 1,536 |
| granite-4.0-h-small | 4 / 36 | 144.00 MiB | 16,384 | 32,768 | 2.00x | 9,216 | 4,608 |
| Nemotron-H-8B-Base-8K | 4 / 24 | **96.00 MiB** | 16,384 | 32,768 | 2.00x | 6,144 | 3,072 |

(The Qwen3.5 row reproduces the measured 18.00 MiB exactly, which validates the
fp32 assumption used for the other two.)

That is a 5.3x spread in snapshot cost. The activation-to-K/V bytes-per-token ratio
also moves, from 1.0x on Qwen3.5 to 2.0x on both production models, because
grouped-query attention compresses K/V while activations scale with hidden size and
those models have fewer but wider full-attention layers. And the full-attention
share of replay work swings from 23% to 88% across 512 to 131,072 tokens.

The last column is the one that matters. For Nemotron-H-8B at about 3K context the
entire full-attention activation cache costs the same as **one** recurrent snapshot,
so the two strategies are priced comparably rather than one dominating. That looked
like a live question when written. Section 10.3 settles it negatively with optimal
checkpoint placement.

CORRECTION. An earlier pass of this table reported 252 MiB for Nemotron-H. That was
wrong: the script fell back to the MLP `intermediate_size` as the Mamba inner width.
Mamba2 state is `[n_heads, head_dim, d_state]` per layer, and `n_heads * head_dim` is
the Mamba inner width, not the MLP width. Corrected value is 96 MiB.

INTERPRETATION. This still changes one upstream premise. vLLM #45702's objection to
dense recurrent checkpointing, that finer-grained Mamba checkpoints inflate
per-request HBM footprint and cut batch size, is **correct** at production scale: at
96 to 144 MiB per snapshot you cannot store many, and you need 6K to 9K of context
before the full-attention K/V cache equals a single one. So divergence between a
deep full-attention hit and a lagging recurrent state is a real operating condition,
not an artifact of a toy model. What does NOT follow, and what section 10.3 shows is
false, is that this makes a multi-class planner worthwhile.

### Risk

Sparse Prefix Caching already owns checkpoint placement with an exact DP, and vLLM
#55697 and #52959 are actively building in this area. Adding state classes to an
existing DP is incremental. Resolve it cheaply, and be willing to kill it.

### Next step

Recompute section 1.5's Pareto with the Nemotron-H and Granite constants, and with a
real divergence-depth distribution rather than a uniform one. Two days. If the
winning policy is the same for every model and context, kill it.

---

## 7. What not to do

- Do not resurrect the frozen-K/V performance mechanism. The ceiling argument in
  section 1.3 is analytical and does not depend on Apple hardware.
- Do not cite any fixed-method-order timing, including the v1.2 numbers in section
  1.2's left column.
- Do not claim the interleaved-stack soundness finding is novel without first
  reading HYPIC in full. Its seam window reaches the same insight for a different
  problem, and it is the closest thing to prior art on the idea.
- Do not promote RQ2 to the thesis before hardware access exists.
- Do not treat the cross-model table in section 6 as measured serving behaviour. It
  is computed from config files.
- Do not reopen RQ3. Two independent methods, one normalized and one with real
  constants and optimal placement, agree that simple policies win.
- Do not cite section 1.5's evenly spaced Pareto as the result. It is the weak
  baseline; section 10.3 supersedes it.

## 8. Reproduction

```
# hybrid-state measurements and analysis
cd experiments/hybrid_state_recovery_v1_3
.venv/bin/python run_mac_selective_v13.py --mode sweep   # randomized-order sweep
.venv/bin/python flop_model.py                           # architectural ceiling
.venv/bin/python pareto_budget.py                        # evenly spaced Pareto (weak baseline)
.venv/bin/python crossmodel_state_ratio.py               # cross-model byte constants

# RQ1 gate: plan-level soundness against real vLLM, CPU only
cd ../rq1_soundness_gate
VLLM=~/Downloads/vllm_hybrid_state_gap_harness/vendor/vllm
PYTHONPATH=$VLLM .venv/bin/python soundness_gate.py

# RQ3: optimal checkpoint placement vs activation caching, 3 real models
cd ../rq3_state_budget
.venv/bin/python rq3_budget.py

# RQ2: NCCL source audit (needs only a clone, no GPU)
cd ../rq2_nccl_source_audit
git clone --depth 1 https://github.com/NVIDIA/nccl.git /tmp/nccl
python3 audit_nccl_ce_selection.py /tmp/nccl
```

`.venv` in that directory is a **symlink** to
`~/Downloads/mac_hybrid_selective_recovery_poc/.venv`, which is the only env on this
machine with torch and transformers installed. If that directory is cleaned up,
recreate the env from `requirements.txt`. The default system `python3` has no torch.

Environment: Apple M4, 16 GiB unified memory, macOS 26.5.2, torch 2.14.0,
transformers 5.17.0, MPS, float16, eager attention. Qwen3.5-0.8B-Base.
`causal_conv1d` and `flash-linear-attention` are absent, so recurrent kernels use
correct but slow reference implementations. This inflates recurrent cost relative to
attention, which makes every full-attention share reported here a **conservative
underestimate** on this hardware. The analytical model in section 1.3 does not have
this bias.

## 9. Sources

- vLLM #53041, https://github.com/vllm-project/vllm/issues/53041
- vLLM #52773, https://github.com/vllm-project/vllm/issues/52773
- vLLM #57266, https://github.com/vllm-project/vllm/issues/57266
- vLLM #55600, https://github.com/vllm-project/vllm/issues/55600
- vLLM #56655, https://github.com/vllm-project/vllm/issues/56655
- vLLM #55697, https://github.com/vllm-project/vllm/issues/55697
- vLLM #52959, https://github.com/vllm-project/vllm/issues/52959
- vLLM #49232 ReplaySSM, https://github.com/vllm-project/vllm/issues/49232
- Sparse Prefix Caching, https://arxiv.org/abs/2605.05219
- HYPIC, https://arxiv.org/abs/2607.01299
- Illusion of Equivalence, https://arxiv.org/pdf/2604.15409
- Collective communication survey, https://arxiv.org/abs/2608.15118
- NCCL 2.28 CE collectives and device API, https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/
- NCCL environment variables, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Hybrid models in vLLM, https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/
- AI21, One token to corrupt them all, https://www.ai21.com/blog/vllm-debugging-mamba-bug/

---

## 10. Iteration 4b: every Mac-testable gate, completed

Artifacts: `experiments/rq1_soundness_gate/`, `experiments/rq2_nccl_source_audit/`,
`experiments/rq3_state_budget/`.

### 10.1 RQ1 gate: PASS

The gate asked how much of vLLM's reuse machinery can be exercised with no GPU.
Answer: enough. Everything below ran on this MacBook against the pinned commit,
CPU only.

FACT. Real per-group hits from the real coordinator. A 128-token prompt is cached,
then a request sharing that prefix is looked up:

| `mamba_cache_mode` | per-group hits | default path | connector path | diverged |
|---|---|---|---|---|
| `none` | (128, 128) | 128 | 128 | False |
| `align` | **(128, 0)** | **0** | 128 | **True** |

In align mode the default path discards 128 already-valid full-attention tokens
because the recurrent group has no checkpoint at that boundary. That reproduces
vLLM #45238 on a laptop, which is the cheapest imaginable testbed for this problem
area.

FACT. The soundness condition decides every candidate plan statically, with no
model execution, and it agrees with the MPS measurement on all three plans that
have ground truth:

| Plan | Static verdict | Measured max logit diff | Agree |
|---|---|---|---|
| `upstream_default_reconciled` | sound | not measured | n/a |
| `upstream_connector_divergent_load` | sound | not measured | n/a |
| `issue_53041_proposal_B_recompute_backfill` | **unsound** | 1.0554 | yes |
| `frozen_kv_but_fa_still_executes` | sound | 0.0117 | yes |
| `activation_cached_fa_skip` | sound | 0.0117 | yes |

For Proposal B the checker does not merely say "unsound", it names the starved
layers: 4, 8, 12, 16 and 20, each executing from token 0 while its predecessor
full-attention layer reuses through 128 and therefore emits no hidden state on
[0, 128).

INTERPRETATION. RQ1 is feasible on the hardware you already own. The static
condition is strong enough to classify real proposals, precise enough to localize
the defect, and validated against dynamic measurement rather than asserted.

### 10.2 A correction to the novelty ledger: upstream already landed divergent hits

NEW FACT. At the pinned commit, `KVCacheManager.get_computed_blocks_for_connector`
already implements divergent per-group hybrid hits, and the connector base class
exposes `supports_divergent_local_hybrid_hits`. Upstream's own comment states the
rule:

> "Report the full-attention hit as the local prefix ... and flag when that hit ran
> deeper than a lagging group. Such a hit only has a valid Mamba state at its
> boundary if the connector supplies it, so the caller must fall back to
> `get_computed_blocks` to reconcile when no external tokens are found."

INTERPRETATION. This matters two ways. First, #52773's load-backfill is not a
proposal any more, it is shipped, so the performance half of this research line is
even more thoroughly closed than section 2 said. Second, it is direct evidence for
RQ1: upstream keeps itself sound with hand-written conditional guards
("if any hit ran deeper than full attention, fall back and reconcile") spread
across the manager and the scheduler, with the invariant stated only in a
docstring. The closure condition in section 4 is that invariant, written down and
made checkable. That is the contribution.

### 10.3 RQ3: KILL

RQ3 asked whether the best allocation across state classes varies enough across
real models to justify a planner. It does not.

The analysis was rebuilt three times before it produced a trustworthy answer, and
all three fixes changed the result:

1. Recurrent snapshot bytes now use the measured float32 element size.
   Nemotron-H's own config confirms this independently: `mamba_ssm_cache_dtype:
   float32`.
2. Layer classification now has three classes. Nemotron-H lists 24
   `linear_attention` + 24 `mlp` + 4 `full_attention`, so treating "everything
   that is not attention" as state-holding doubled its snapshot to 192 MiB. The
   corrected value is 96 MiB.
3. Checkpoint placement is now optimized by dynamic program per divergence
   distribution, and validated against the closed form n/(2(m+1)) for the uniform
   case. The first DP was wrong: it wrote the same cost into both `dp[jj][c]` and
   `dp[jj][c+1]` and returned a constant regardless of budget. Section 1.5's
   evenly-spaced comparison was a weak baseline; this is the strong one.

FACT. 3 models x 3 context lengths x 4 divergence-depth distributions x budgets
from 1 to 32 snapshots. Optimally placed recurrent checkpoints win at **every**
budget from 1 to 8 snapshots, for every model, at every context, under every
distribution. The activation cache only wins at 16 or 32 snapshots of budget,
which is 288 MiB to 4.6 GiB of state for a single request.

FACT, and this is the interesting negative. The four distributions (uniform,
front-loaded, back-loaded, bimodal) produce almost identical winner patterns. The
crossover is set by the byte budget, not by where requests actually diverge. The
distribution sensitivity that would have justified a planner is absent.

INTERPRETATION. Two independent methods now agree. The handoff's normalized
324-row screen concluded that simple policies capture nearly all the benefit; this
real-constant, strong-baseline version says the same thing more sharply. Close
RQ3. Do not reopen it without a genuinely new cost class.

### 10.4 RQ2: the architectural gap is confirmed without a GPU

The Mac-testable question was whether NCCL's engine-selection point can even
observe concurrent compute-kernel resource pressure. It cannot, and the evidence
is stronger than a literature absence.

FACT (NCCL 2.31.2, commit `fd168324a3`, via `audit_nccl_ce_selection.py`):

1. The entire copy-engine versus SM routing decision is one line, at
   `src/enqueue/enqueue.cc:3305`:
   `if ((info->collConfig.CTAPolicy & NCCL_CTA_POLICY_ZERO) && (ceAvailable || hierCeAvailable))`.
   A user-set static policy, AND a static eligibility predicate. No cost model.
2. `ncclCeAvailable`'s five rejection reasons are all static capability or
   registration properties: collective not implemented for CE, host-backed cuMem
   segments, incomplete NVLink connectivity, symmetric support disabled,
   unsupported window registration type. Not one is a load condition.
3. The tuner plugin is the only extension point for algorithm choice, and it is
   **not consulted for CE routing at all**. Even if it were, `getCollInfo` in the
   latest interface (`tuner_v6.h`) receives only `context, collType, nBytes,
   numPipeOps, collCostTable, numAlgo, numProto, regBuff, nChannels`. No
   occupancy, no bandwidth pressure, no handle to the concurrent kernel.
4. Searching all of `src/` for `occupancy`, `cudaOccupancy`, `smUtil`,
   `GetUtilization`, `activeWarps`, `nvmlDeviceGetUtilizationRates`, `smActive`
   and `achieved_occupancy` returns **zero matches**. The runtime never measures
   the quantity RQ2's policy would key on.

INTERPRETATION. The novelty claim is no longer "I did not find public work that
does this." It is "the input does not exist at the decision point, the only
extension point cannot see it either, and the runtime does not collect it
anywhere." That is a structural gap, and establishing it needed no GPU.

This de-risks RQ2 substantially. Hardware is still required, but only to answer
how much the missing input is worth, not whether it is missing. The
simplest-solution attack is completely untouched: a static policy may well capture
most of the available benefit, and until the microbenchmark in section 5 runs that
remains the single biggest threat to RQ2.

---

---

## 11. Iteration 4c: the layout survey, and the death of Idea 3

Artifacts: `experiments/rq3b_layout_survey/`.

Idea 3 was the one positive-result candidate: if interleaving is what makes
divergent hybrid state reuse unsound, then a layout that avoids interleaving would
make it both sound and valuable. The survey settles it. The design space is empty.

### 11.1 A sharper rule, validated two ways

The closure condition implies something stronger and simpler than section 4 stated.
Recurrent-state backfill only has to produce correct recurrent state at the
boundary. Recurrent layer L's state depends only on layers below L. So **every
layer above the highest state-holding layer can be skipped, with no cached
activations at all**, because nothing ever reads its output during backfill. And
the skipped set must be a contiguous suffix of the stack: if layer i is skipped,
layer i+1 has no input, so i+1 must also be skipped or i must be last.

FACT. Validated by exhaustive search over all subsets of full-attention layers
against the section 4 checker, on six layouts, including interleaved,
recurrent-first, attention-first and paired. The predicted maximal sound set
equals the brute-force maximum in every case.

FACT. Validated dynamically on the real model. Skipping layer 23 of Qwen3.5-0.8B
during replay leaves the recurrent state **bit-identical**, max absolute
difference exactly `0`, at contexts 512, 1024, 2048 and 4096. Note the criterion:
recurrent-state equality, not logit equality. Backfill does not need logits for
replayed positions, and this is the first experiment in the whole line where using
the right correctness criterion changes what counts as sound.

This also generalizes past full attention. An MLP block sitting above the last
recurrent block is skippable too, which is exactly Nemotron-H's situation.

### 11.2 The survey: 13 real hybrid models, zero favourable layouts

FACT (`probe_layouts.py`, `layout_survey.py`; HF configs only, no weights):

| Model | Layers | FA | Skippable suffix | Last layer |
|---|---|---|---|---|
| Qwen3.5-0.8B-Base | 24 | 6 | **1** | full attention |
| Qwen3-Next-80B-A3B | 48 | 12 | **1** | full attention |
| Nemotron-H-8B / 47B / 56B | 52 / 98 / 118 | 4 / 5 / 10 | **1** (an MLP block) | mlp |
| granite-4.0-h small / tiny / micro | 40 | 4 | **0** | recurrent |
| Bamba-9B | 32 | 3 | **0** | recurrent |
| LFM2-1.2B | 16 | 6 | **0** | recurrent |
| Zamba2-2.7B | 54 | 0 | **0** | fused |
| Falcon-H1 0.5B / 34B | 36 / 72 | 0 | **0** | fused |

- Models with a recurrent-first layout: **0 of 13**.
- Models where more than half the full-attention work is soundly skippable: **0 of 13**.
- Models where nothing at all is skippable: **8 of 13**.

Falcon-H1 and Zamba2 are a stronger negative than the table suggests. Their blocks
are fused parallel hybrids: attention and recurrent paths run inside one block and
their outputs are combined, so the attention path cannot be skipped independently
under any layout reasoning at all.

### 11.3 Achievable versus counterfactual

FACT. Share of replay work, analytical, boundary fraction 0.75, MoE expert MACs
scaled by top-k over expert count:

| Model | Context | Achievable now | If layout were recurrent-first |
|---|---|---|---|
| Qwen3.5-0.8B | 2,048 | 4.74% | 28.5% |
| Qwen3.5-0.8B | 131,072 | **14.72%** | **88.3%** |
| Qwen3-Next-80B | 131,072 | 7.48% | 89.8% |
| Nemotron-H-8B | 2,048 | 2.49% | 3.2% |
| Nemotron-H-8B | 131,072 | 1.63% | 36.4% |
| granite-4.0-h-tiny | 131,072 | 0.00% | 67.6% |
| LFM2-1.2B | 131,072 | 0.00% | 82.6% |

Nemotron-H is the interesting inversion: its achievable saving *falls* with context,
from 2.49% to 1.63%, because the one skippable block is an MLP whose cost is linear
in suffix length while attention cost grows superlinearly. The skippable layer is
the wrong kind.

FACT (measured, 8 repetitions per cell, randomized order). Qwen3.5-0.8B,
boundary 0.75: saving of 1.52%, 1.01%, 1.01%, 6.40% at contexts 512, 1024, 2048,
4096. Analytical prediction at 2048 was 4.74%; measured 1.01%.

INTERPRETATION of the shortfall. Measured comes in below analytical in the expected
direction and for the already-documented reason: `causal_conv1d` and
`flash-linear-attention` are absent, so recurrent layers run slow reference
kernels, which inflates the recurrent share and shrinks the measured
full-attention share. The analytical figure is the unbiased one. Do not quote the
measured number as the ceiling, and do not quote the analytical number as measured.

### 11.4 Verdict on Idea 3: KILL as a thesis

The counterfactual is large, 88% at 128K for Qwen3.5. The achievable is 0% for most
real models and about 15% in the single best case. The gap is entirely explained by
where architects put the attention layers, and every surveyed family puts recurrent
or fused blocks last.

Closing that gap means proposing a layout change, and validating that a
recurrent-first hybrid retains model quality requires pretraining. That is far
outside a master's thesis budget, and without it the co-design claim is
unfalsifiable. So Idea 3 does not become a thesis.

What it does become is a finding inside RQ1, and a good one: the soundness
condition does not merely reject an unsound optimization, it *prices the entire
design space* and shows the space is empty for structural reasons that are visible
from config files alone. That is a stronger result than a bare correctness checker,
and it is the kind of design-space pruning that makes a correctness paper a systems
paper.

### 11.5 One concrete upstream contribution that falls out

Skipping layers above the last recurrent layer during backfill is sound, needs no
extra memory, and is worth about 1% to 6% measured on this hardware and up to about
15% analytically at long context for Qwen3.5-family layouts. It is also, as far as
the surveyed code shows, not done: vLLM's backfill path has no notion of a
per-request layer ceiling. That is a small, self-contained, verifiable upstream
patch, and a good first artifact for RQ1 because it demonstrates the invariant
doing useful work rather than only forbidding things.

