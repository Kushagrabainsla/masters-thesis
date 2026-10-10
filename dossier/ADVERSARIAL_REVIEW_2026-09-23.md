# Adversarial Review of the Thesis Candidates

Date: 23 September 2026. Scope: every candidate in `ideas/`, `dossier/`, the two
handoff .docx files, `stateful_elastic_inference_thesis_dossier.docx` and the new
`LLM_Inference_Thesis_Research_Handoff_2026-09-22.docx`.

Method: two adversarial passes, about 70 sources read (papers, GitHub issues and PRs,
engineering blogs, NCCL source). The NCCL audit was re-run on current master. Cited
issues were opened and checked against what the docs claim.

Labels: FACT means a source or a run of code showed it. INTERPRETATION is my reading.
HYPOTHESIS is a prediction that a test could refute. "I did not find" is never a
proof of absence.

---

## 1. Verdicts

| Candidate | Verdict | Main reason | Confidence in gap |
|---|---|---|---|
| SafeDispatch / KernelGuard (Sep 22 doc) | **MEASURE-FIRST, narrowed, needs 1 NVIDIA GPU** | Phenomenon and margin trigger are now published. Only the joint perf-plus-equivalence dispatch decision is open, and upstream is heading there | low to medium |
| Compute-aware collective engine choice (Idea 2) | **MEASURE-FIRST, low priority, needs NVLink GPUs** | Source gap is real. Value is doubtful, and the predeclared experiment cannot test the "live" part | high on gap, low on value |
| Soundness of per-group state reuse (Idea 1) | **KILL as a thesis** | Condition is a data dependence. Motivating bugs are a different class. Closest tools already exist | medium |
| Live migration of hybrid sessions | **KILL** | Llumnix line plus vLLM state transfer plus WorldMove admission control. Own data says single-session copy is cheap | high |
| Heterogeneous elastic training, power-aware switching, elastic reconfiguration | **Not re-reviewed** | Older docs already mark them crowded or untested. I did not search them | n/a |

Nothing is a GO today. The nearest to GO is SafeDispatch, and only after its oracle
test passes.

---

## 2. The field moved. The Sep 22 handoff is behind its own date

The Sep 22 doc says its literature frontier is dated 22 September. It misses these,
all dated before that day. Every row was opened and read.

| Work | Date | What it already shows | Effect |
|---|---|---|---|
| MarginGate, arXiv 2605.30218 | 2026-05-28 | Logit-margin trigger, verify only low-margin steps, 100% deterministic, 2.23x lower latency than always-on LLM-42 on Llama-3.1-8B | The doc's "simple logit-margin baseline" is already a published system |
| Greedy Decoding Is Not Precision-Invariant, arXiv 2609.26621 (TMLR) | 2026-09-22 | BF16 vs FP16 diverges on 59 to 82% of prompts. Margin-triggered FP32 lm_head fix, under 4% overhead at batch 4 or less. Gain fades at larger batch and under FP8 | H2 (structured risk) is established. FP8 is the stated open part |
| Deterministic LLM Inference Across GPU Kernels, arXiv 2609.00363 | 2026-08-25 | Five epilogue faults each move output by at most one bf16 spacing, so a one-spacing tolerance is blind to all of them. Conformance suite | Kills "allclose is insufficient" as a contribution |
| Accelerating Mitigation of Nondeterminism Across GPU Architectures, arXiv 2609.25624 | 2026-09-22 | Fixed-order fused-upcast GEMM, bitwise identical across Ampere, Ada, Hopper, 1.17x to 3.1x faster than the prior deterministic SOTA | Makes the "static conservative kernel" baseline cheap |
| Taming Bitwise Behavior in GPU Kernels with Tensor Core, arXiv 2609.11356 | 2026-09 | Black-box reconstruction of GEMM reduction order, compiler-enforced balanced tree, static bitwise-equivalence checkers. 19 of 27 kernels within 10% of free-order speed on GB300 and H100 | Kernel fingerprinting and static equivalence are taken. The fixed-order baseline is near free |
| Same Request, Different Answer, arXiv 2609.04748 | 2026-09-04 | Enabling prefix cache changes agent trajectories in 36.2% of episodes at 16-bit and 75.0% at 4-bit | Cache-induced divergence is measured. No mitigation yet |
| Silent Hyperparameter, arXiv 2605.19537 | 2026-05-19 | Backend alone shifts benchmark scores up to 16.6 points. Isolates prefix caching, CUDA graphs, kernels, logit defaults | Feature-level attribution is partly covered |
| Deterministic Inference Across TP Sizes (TBIK), arXiv 2511.17826, ICML 2026 | v2 2026-05 | Tree-based invariant kernels give bitwise equality across TP sizes | TP invariance is solved for BF16 |
| CoRun, arXiv 2608.14376 | 2026-08 | Padding gives deterministic inference | Crowded |
| Integer Alibi, arXiv 2608.13756, and its companion | 2026-08 | Already in the doc. Scope limits matter, see 3.1 | Sets the open cells |
| Runtime Observability for Heterogeneous Attention Memory, arXiv 2608.05863 | 2026-08-06 | Formal observability contract with Lean 4 proofs, four cache types including recurrent state, always-on probe under CUDA graphs | Direct threat to any runtime cache-verification idea |
| Ekka, arXiv 2606.04594 (ICML 2026) and GRIEF, arXiv 2605.11202 | 2026-05/06 | Differential debugging against a reference for silent errors. Greybox fuzzing of vLLM and SGLang with log-prob replay, 15 bugs, 2 CVEs | Docs claim "no public work" applies differential testing to serving state reuse. Overstated |
| SGLang Unified Radix Cache | 2026-08-11 | FULL, SWA, MAMBA components on one tree. Reuse boundary advances only when every component validator accepts | The reconciliation rule is shipped as a design |
| vLLM PRs #55868, #55895, #56244 | 2026-09 | Batch-invariant Mamba2 and GDN, with prefix caching on an LCM grid of the FLA chunk size (64) and the mamba block size | Hybrid determinism and alignment are being built now |
| Validating Hybrid-State Cache Recovery, arXiv 2609.15030 | 2026-09-14 | Hybrid state recovery bug found by generation-agreement test | Existing validation of the same bug area |

FACT: the same Sep 22 doc lists DriftBench, LLM-42, LAMP and BGSS correctly, so its
refresh was real but incomplete.

---

## 3. Candidate reviews

### 3.1 SafeDispatch / KernelGuard

Surviving question as written: can real-activation risk features and runtime context
predict unsafe kernel swaps, so a dispatcher beats a static conservative kernel?

**Hypotheses H1 to H4 in the doc, status now**

| Hypothesis | Status |
|---|---|
| H1 non-interchangeability | Published. Integer Alibi: CUTLASS vs Triton INT8 in vLLM gives 0/8, 0/16, 0/64 sequence agreement |
| H2 risk is structured, margin predicts it | Published. Integer Alibi, MarginGate, 2609.26621 |
| H3 fastest safe kernel varies by regime | Open. No paper measures it |
| H4 a cheap rule captures the oracle | Open |

So only H3 and H4 remain, and both hinge on one number: the oracle gain over the best
static safe choice.

**Attacks**

1. Novelty. Measurement, margin gating, determinism and kernel verification are all
   taken. The one unfilled piece is a dispatch decision that trades speed against
   output equivalence. FlashInfer v0.7's autotuner ranks candidates on latency only
   (FACT, its docs say nothing about numerical checks). DA-MoE dispatches on routing
   skew for speed only. vLLM's Q3 roadmap lists "backend selection is deterministic,
   debuggable, and model-aware" under SIG Quantization, plus release-gating accuracy
   tests. INTERPRETATION: upstream will grow toward this piece within months.
2. Simplicity. The doc's strongest baseline is "one static conservative kernel". That
   baseline got cheaper this month (2609.25624: faster than prior deterministic SOTA,
   2609.11356: 19 of 27 kernels within 10% of free order). If the conservative kernel
   costs under a few percent, the oracle gain is small by construction.
3. Oracle. Not measured anywhere. This is the right first experiment and the doc
   already says so.
4. Reviewer rejection sentence: "Integer Alibi shows the divergence, MarginGate and
   LLM-42 show how to gate it, and fixed-order kernels make the safe baseline nearly
   free, so what does dispatch add?"

**What still looks open, from the closest prior art's own scope statement**

Integer Alibi states (FACT, quoted from its limitations): one consumer GPU (sm_89),
one model family at two sizes, one engine version, two INT8 kernels plus one FP8 pair.
"Nothing here licenses claims across GPU generations, model families, or future
kernels." It also has no MoE, no NVFP4 and no multi-GPU. Its FP8 result is
operator-level only, no end-to-end sequence agreement.

Independent evidence that these cells are live: FlashInfer's audit found about 36% of
bugs in low-precision GEMM and MoE and about 31% in MoE, and about 60% of fixes in the
Python dispatch and validation layer. Open B200 reports vLLM #55238 (CUDA graph
replay changes greedy output, Blackwell only, no root cause at fetch time) and #55649
(NVFP4 gated-MoE padding corrupts output, fix PR unmerged) match that pattern.

HYPOTHESIS: on Hopper and Blackwell, for block-scaled FP8, NVFP4 and fused MoE, the
per-regime fastest contract-satisfying backend beats the best single static
contract-satisfying backend by a margin that survives integration.

Scoop risk is high in these cells. The Integer Alibi author posted two papers in 12
days and explicitly invited extension.

**Narrowed question**

> On Hopper and Blackwell, for block-scaled FP8, NVFP4 and fused-MoE backends, does the
> fastest backend that satisfies an explicit output contract differ across serving
> regimes enough to beat the best single fixed-order backend by at least 5% end to end?

**Verdict: MEASURE-FIRST.** Run the oracle test in section 6 and stop early if it
fails. The floor if it fails is still a measurement thesis on cells no one has
filled, plus an open-source harness. Do not build a learned dispatcher first.

### 3.2 Soundness of per-group state reuse (Idea 1)

**Facts checked in the sources**

- FACT: vLLM #53041 is a discussion post by one author. It had zero comments when
  fetched on 23 Sep. It names per-layer selective compute as the "key technical
  challenge" itself. The docs call it "an active upstream design discussion". That
  overstates it.
- FACT: the "five instances of this bug class" are not one class.

| Issue | Actual root cause | Would the input-closure checker catch it? |
|---|---|---|
| #57266 | Out-of-range strided slice when `num_computed_tokens % mamba_block_size != 0` | No |
| #55600 | Unit mismatch: divides by `block_size` (reassigned to the minimum across groups) instead of `mamba_block_size`. One-line fix | No |
| #56655 | Prefix cache key omits the `prompt_is_token_ids` mask | No |
| #47812 | Single-group assumption in KV-load failure recovery (taken from the docs, I did not open it) | Not shown |
| #53041 Proposal B | Skipping FA layers starves downstream recurrent layers | Yes, but it is an unimplemented sketch |

INTERPRETATION: the checker covers one of the five, and that one has no shipped code.
The doc's own falsifier was "too weak to catch the known bug class". By the doc's
evidence, that falsifier has fired.

**Attacks**

1. Triviality. Input-closure says a layer cannot run without its predecessor's
   output. That is the data-dependence graph of a transformer. The non-trivial part
   would be extending it to sliding window, speculative decode, DCP and offload,
   which is unbuilt.
2. Prior art. SGLang's component voting is the shipped reconciliation rule. Ekka and
   GRIEF are differential and fuzz testers for silent serving errors. 2608.05863 has
   a formal contract with Lean 4 for recurrent state. 2609.15030 validates hybrid
   recovery. Upstream is building hybrid batch invariance and prefix caching.
3. The constructive patch. Skipping layers above the last recurrent layer only helps
   the recompute-backfill path. vLLM #52773 measured load-backfill at 80x cheaper and
   the docs say it already shipped, so recompute is a last resort. Worth 1 to 6%
   measured on a path that rarely runs.
4. Career fit. The doc admits it is the weakest of the set.

**What survives:** the 13-model layout survey (0 of 13 layouts favourable, 8 of 13
with nothing skippable) and the checker, as a short workshop or technical note.
Budget one to two weeks, not a thesis.

### 3.3 Compute-aware collective engine choice (Idea 2)

**Audit replicated.** FACT: on NCCL 2.32.3 (master merge of 17 Sep 2026, commit
`12df1a1`, newer than the docs' 2.31.2) the audit script gives the same result. CE
routing is `if ((CTAPolicy & NCCL_CTA_POLICY_ZERO) && (ceAvailable || hierCeAvailable))`
at `src/enqueue/enqueue.cc:3398`. The five rejection reasons are still all static. The
tuner interface is still `tuner_v6`, with no load input. No occupancy or utilization
sensing in `src/`.

**New facts that change the picture**

- FACT: NCCL now has `ncclCollConfig_t` and `*WithConfig` calls (struct version 23200,
  so 2.32). I did not check whether 2.31.2 had it. Per call it sets
  `CTAPolicy`, `minCTAs`, `maxCTAs`, `nvlsCTAs` and `algSelection`. The code comment
  says the per-call value wins. So a framework can already choose SM vs CE per
  collective. The missing piece is a policy, not a mechanism.
- FACT: NVIDIA's NCCL 2.28 blog shows CE multicast AllGather at about 780 vs 620 GB/s
  (1.25x) against the SM path at 4 GB on 8 Blackwell GPUs. Its curve covers 4 MB to
  4 GB with CE ahead throughout, and it names no case where CE loses. Standalone, not
  overlapped.
- FACT: NCCL's Q2 2026 roadmap adds zero-SM AllGather and explores SM-initiated CE
  collectives. Runtime SM-vs-CE selection is not listed.
- FACT: the survey the docs cite (arXiv 2608.15118) does contain the quoted sentence,
  in its "Communication Offloading" future-work paragraph. The same paragraph says
  offloading to copy engines "is already practical in many systems". The docs cite it
  as naming this problem open. It names generic co-design as open.
- FACT: recent work targets the SM budget, not the engine. 2606.09200 cuts total time
  up to 25.5% with occupancy shaping and no vendor library change. 2609.07536 picks
  communication CTA counts analytically. FLUX, Weave and mKernel fuse compute and
  communication.
- FACT: CE covers data-movement collectives (AllGather, AllToAll) inside one NVLink
  domain. TP AllReduce in inference is not a CE collective.

**Attacks**

1. Physical argument. INTERPRETATION, untested: an SM copy and a CE copy read and
   write the same bytes through HBM. So "bandwidth-bound kernel prefers the SM path"
   has no mechanism I can see. What differs is granularity and launch cost: CE suits
   large contiguous moves, SM suits small or scattered ones. That is a message-size
   rule, which NCCL's static eligibility partly encodes already.
2. **Experiment-design flaw.** The predeclared oracle is per (kernel, message size)
   cell, chosen with hindsight. Kernel type and message size are both known at the
   call site with no measurement. So a static per-call table implements that oracle.
   The GO test compares the oracle to the best single global policy. It never tests
   whether a live signal beats a static table, which is the "live resource profile"
   claim. A large oracle gap would show only that a lookup table helps.
3. Threat. Scoop from NVIDIA's own roadmap, and the field moved to fused kernels and
   SM partitioning.

**Verdict: MEASURE-FIRST but low priority, gated on NVLink hardware.** Fix the design
before spending a node:

- Add a within-cell test: fixed kernel class and message size, but varying compute
  pressure (GEMM shapes across arithmetic intensity, attention with different batch
  and context). Only there can a live signal beat a static table.
- Add the baseline "per-call static table from op type and size".
- GO only if a live-signal policy beats that table.

### 3.4 Live migration of many hybrid sessions

- FACT: Llumnix (OSDI 2024), Astrolabe, OServe (reports 1.32x to 1.51x P99 gain over
  Llumnix), Pallas and ReMP cover live migration and topology change for attention
  models. WorldMove covers bit-exact mutable-state migration with incast-aware
  admission control. vLLM already transfers hybrid conv and SSM state in P/D transfer
  with heterogeneous TP.
- FACT: NVIDIA Dynamo Snapshot is single-GPU today and lists multi-GPU as in
  progress. Concordia (arXiv 2606.23521) does GPU-side checkpointing with a standby
  pool for live migration.
- FACT from your own data: one Falcon-H1 session's recurrent state is 27 MB, about
  0.13 to 2 ms at datacenter link speed.
- Rejection sentence: "This is Llumnix with a 100 MB fixed tail, WorldMove's
  admission controller, and vLLM's existing state transfer."
- The doc's own first gate was a fresh novelty scan. It was never run. My partial
  scan found no public system that does live decode-owner migration of a hybrid
  session, but every component is published.

**Verdict: KILL.** Needs 2 to 4+ GPUs and the residual is integration.

---

## 4. Errors and overstatements found in the docs

1. RQ1 conflates three bug classes (indexing, unit mismatch, cache-key omission,
   layer-skip) under one "class" the checker addresses.
2. RQ1 says #53041 is an active design discussion. It has zero comments.
3. RQ1 says no public work applies differential testing to serving state reuse. Ekka
   and GRIEF exist.
4. RQ2 cites the survey as naming CE-vs-SM arbitration open. The survey says CE
   offload is practical and the open item is generic co-design.
5. RQ2's GO test cannot distinguish "live signal" from "static table" (3.3).
6. RQ2's "bandwidth-bound kernel prefers the SM path" has no stated mechanism, and
   the physical argument runs the other way.
7. SafeDispatch's threat list omits the papers in section 2 and treats H1 and H2 as
   open.
8. Live migration's novelty gate was never run.
9. `ideas/README.md` says "two live candidates" and predates the Sep 22 direction.
   `RESEARCH_STATUS.md` (Sep 18) recommends Idea 1 while the Sep 22 doc recommends
   SafeDispatch. Nothing reconciles them.

Credit where due: the NCCL source audit reproduces, the layout survey and checker
agree with their own controls, and the docs consistently marked synthetic and
invalid results.

---

## 5. Gap hypotheses that survive pass 1

All are hypotheses. None has a decisive test run yet.

| Rank | Hypothesis | Why it may be open | Cheapest decisive test | Hardware | Main threat | Confidence |
|---|---|---|---|---|---|---|
| 1 | Contract-satisfying dispatch beats the best fixed-order backend in Hopper/Blackwell block-scaled FP8, NVFP4 and MoE cells (3.1) | Closest paper disclaims these cells. Open B200 bugs. Autotuner ignores numerics | Oracle test, section 6 | 1 Hopper GPU, 1 Blackwell for NVFP4 | Cheap fixed-order kernels, Integer Alibi extension, vLLM model-aware selection | low to medium |
| 2 | Feature-level divergence and speed ledger for cached, multi-turn agent workloads on MoE and hybrid models, multi-GPU | DriftBench varies hardware, framework, precision and model. Silent Hyperparameter is single-GPU standard precision. Cache-divergence paper has no mitigation | Same matrix as 1 with feature flags | 1 to 4 GPUs | DriftBench authors extending | low |
| 3 | NCCL reduction-order and precision contract of the new paths (symmetric kernels, CE, NVLS with FP16 accumulate for FP8) and its end-to-end effect | The NCCL source and docs I grepped state no reduction-order contract. Issue #2360 reports BF16 NVLS results changing with multicast allocation history (no NVIDIA reply visible). A search result says NVLS is deterministic on Hopper and Blackwell with CUDA 12.8+ and NVIDIA's 2.27 blog snippet says FP8 reductions on NVLS accumulate in FP16. I did not trace either to a primary spec | Bitwise-agreement grid across algorithm, protocol, CE, size | 4 to 8 NVLink or NVSwitch GPUs. Bundle with 3.3's test in one node rental | NVIDIA fixing it, vLLM PRs | low |

Fallbacks by outcome of test 1 are in section 6.

---

## 6. Decision procedure, predeclared

Thresholds below are heuristics, not literature standards. Fix them before running.

### Test A: SafeDispatch oracle (1 GPU, about 1 to 2 weeks)

1. Reproduce one documented divergence to validate the harness (Integer Alibi INT8, or
   #55238 on Blackwell if you have it). Not a contribution.
2. Grid: at least two model families (one dense, one MoE), at least three backends per
   operator family, block-scaled FP8 and NVFP4 where hardware allows, BF16 control,
   prefill vs decode, several batch sizes, real captured activations plus synthetic
   controls.
3. Contracts: L1 bitwise equal to a fixed-order reference kernel. L2 zero greedy flips
   on a margin-stratified prompt set. L3 KL bound on logits.
4. Baselines: S1 fastest single always-safe backend. S2 fixed-order deterministic
   kernels. S3 per-shape allow-list from offline qualification. Include MarginGate-style
   gating as a policy comparison, not as a contribution.
5. Oracle gain G = end-to-end tokens/s (or TPOT) of the fastest-safe oracle over the
   best of S1 to S3.

| Result | Decision |
|---|---|
| G at least 5% on 2 or more models, and a runtime rule beats S3 by at least 2% | **GO** for correctness-aware dispatch |
| G under 5%, but at least 10% of default-config substitutions in the open cells violate L2 | **PIVOT** to the measurement thesis: cross-kernel divergence in cells Integer Alibi disclaims, with the harness. Rank 2 in section 5 rides on the same data |
| G under 2% and unsafe fraction under 2% | **KILL** the direction |

### Test B: collective test, only if an NVLink node appears

Bundle with the numerics grid (rank 3) in one rental. Use the doc's original design
plus the within-cell test and the per-call static table from 3.3.

| Result | Decision |
|---|---|
| Live-signal policy beats per-call static table by at least 5% overlapped throughput | GO |
| Per-call table within 2% of the oracle | KILL |

### Calendar (Oct to Dec window, January job start)

| When | Do |
|---|---|
| This week | Secure GPU access. Refresh the threat list with section 2 |
| Weeks 1 to 3 | Test A |
| End of week 3 | Gate: GO, PIVOT or KILL, in writing |
| Week 5 | Claim lock, as in the Sep 22 plan |
| Weeks 6 to 12 | Build only what the gate supports |

---

## 7. Hardware

Every path that is not a KILL needs an NVIDIA GPU. The Sep 22 doc does not state which
GPU you have. The Sep 16 handoff says Mac only.

| Path | Minimum |
|---|---|
| Test A | 1 Hopper GPU. A Blackwell GPU adds NVFP4 |
| Test B and rank 3 | 2 NVLink GPUs to start, 8-GPU NVSwitch node for NVLS and full grid |
| Idea 1 note | MacBook |

If no GPU access exists by mid October, the only viable Mac-only thesis is Idea 1
recast as a small characterization and tool, and the realistic venue is a workshop.

---

## 8. What not to do

- Do not build a learned risk model or dispatcher before the oracle number exists.
- Do not claim novelty for measuring divergence, margin gating, determinism, kernel
  fuzzing, kernel fingerprinting or selective precision. All have 2026 papers.
- Do not reopen live migration, Idea 1 as a thesis, or hybrid batch invariance (vLLM
  PRs are landing).
- Do not spend a node on the original RQ2 design. Fix the within-cell flaw first.
- Do not treat this review as a novelty proof. It is a search log. Re-scan arXiv and
  the repos before the week-5 claim lock.

---

## 9. Sources

Papers and reports

- Integer Alibi, arXiv 2608.13756, 2026-08-13. https://arxiv.org/abs/2608.13756
- Deterministic LLM Inference Across GPU Kernels, arXiv 2609.00363, 2026-08-25. https://arxiv.org/abs/2609.00363
- DriftBench, MLSys 2026. https://mlsys.org/virtual/2026/oral/3799
- MarginGate, arXiv 2605.30218. https://arxiv.org/abs/2605.30218
- Greedy Decoding Is Not Precision-Invariant, arXiv 2609.26621. https://arxiv.org/abs/2609.26621
- Accelerating Mitigation of Nondeterminism Across GPU Architectures, arXiv 2609.25624. https://arxiv.org/abs/2609.25624
- Taming Bitwise Behavior in GPU Kernels with Tensor Core, arXiv 2609.11356. https://arxiv.org/abs/2609.11356
- FPRev, ATC 2025, arXiv 2411.00442. https://arxiv.org/abs/2411.00442
- TBIK, arXiv 2511.17826. https://arxiv.org/abs/2511.17826
- LLM-42, arXiv 2601.17768. https://arxiv.org/abs/2601.17768
- CoRun, arXiv 2608.14376. https://arxiv.org/abs/2608.14376
- Same Request, Different Answer, arXiv 2609.04748. https://arxiv.org/abs/2609.04748
- Silent Hyperparameter, arXiv 2605.19537. https://arxiv.org/abs/2605.19537
- Contract-Grade Verifier, arXiv 2608.12700. https://arxiv.org/abs/2608.12700
- DA-MoE, arXiv 2607.23099. https://arxiv.org/abs/2607.23099
- Runtime Observability for Heterogeneous Attention Memory, arXiv 2608.05863. https://arxiv.org/abs/2608.05863
- Ekka, arXiv 2606.04594. https://arxiv.org/abs/2606.04594
- GRIEF fuzzer, arXiv 2605.11202. https://arxiv.org/abs/2605.11202
- Validating Hybrid-State Cache Recovery, arXiv 2609.15030. https://arxiv.org/abs/2609.15030
- HYPIC, arXiv 2607.01299. https://arxiv.org/abs/2607.01299
- Sparse Prefix Caching, arXiv 2605.05219. https://arxiv.org/abs/2605.05219
- Marconi, arXiv 2411.19379. https://arxiv.org/abs/2411.19379
- Illusion of Equivalence, arXiv 2604.15409. https://arxiv.org/abs/2604.15409
- Collective Communication for Distributed LLM Systems (survey), arXiv 2608.15118. https://arxiv.org/abs/2608.15118
- Resource-aware Computation-Communication Overlap, arXiv 2606.09200. https://arxiv.org/abs/2606.09200
- Analytical Resource Management for MoE Overlap, arXiv 2609.07536. https://arxiv.org/abs/2609.07536
- Llumnix, arXiv 2406.03243. https://arxiv.org/abs/2406.03243
- WorldMove, arXiv 2607.10389. https://arxiv.org/abs/2607.10389
- ReMP, arXiv 2606.18741. https://arxiv.org/abs/2606.18741
- Concordia, arXiv 2606.23521. https://arxiv.org/abs/2606.23521

Upstream issues, PRs and posts

- vLLM Q3 2026 roadmap #48168. https://github.com/vllm-project/vllm/issues/48168
- vLLM #55238, #55649, #53041, #57266, #55600, #56655, #50136, #56370, PR #50505, PR #55868
  (all under https://github.com/vllm-project/vllm/issues/ or /pull/)
- FlashInfer release-quality RFC #3605. https://github.com/flashinfer-ai/flashinfer/issues/3605
- FlashInfer v0.7 autotuner. https://flashinfer.ai/2026/09/22/flashinfer-v07.html
- SGLang Unified Radix Cache. https://www.lmsys.org/blog/2026-08-11-unified-radix-cache/
- NCCL source, v2.32.3, commit 12df1a1. https://github.com/NVIDIA/nccl
- NCCL Q2 2026 roadmap #2090 and issue #2360. https://github.com/NVIDIA/nccl/issues/2090
- NCCL 2.28 CE collectives blog. https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/
- vLLM hybrid SSM disaggregation. https://vllm.ai/blog/2026-04-21-hybrid-ssm-disagg
- NVIDIA Dynamo Snapshot. https://developer.nvidia.com/blog/nvidia-dynamo-snapshot-fast-startup-for-inference-workloads-on-kubernetes/
