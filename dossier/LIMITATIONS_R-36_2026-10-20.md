# Limitations Mining Review R-36

Date: 2026-10-20

Question from the user: go over all papers we reviewed, find the weakness each one leaves for a next
paper, and keep the follow-ups that are not published yet.

Status: literature review only. No experiment ran. No thesis direction is chosen.

## Method

1. Took all 496 arXiv papers in `references/` (the journal's full citation set).
2. Downloaded the arXiv HTML full text of each paper into the session scratchpad (490 full text,
   6 abstract only). Nothing was added to the repo.
3. A script extracted (a) sections whose heading names limitations, future work, open problems,
   discussion or scope, and (b) sentences with future-work cues ("we leave", "future work",
   "out of scope", "does not support", "remains open", and similar).
4. Scope: the 407 papers dated 2025 or 2026. 372 of them state at least one limitation or future
   item. I read the extracted statements of every one of these papers. I skipped 89 older papers,
   because their follow-ups have had time to appear.
5. I grouped the statements into candidate follow-up gaps, removed gaps already in the Section 3
   registry, and attacked each remaining gap with the R-21 checklist (exact work, simple baseline,
   scale, feasibility, scoop).
6. For every gap that looked open, I ran at least one fresh web search and read the closest source.

Limits:
- A stated "future work" is the authors' view. It is not evidence that the gap matters.
- Keyword extraction can miss limitations written without cue words.
- No Sonnet review ran. R-36 is a review, not an experiment.

## 1. What the limitation sections show (patterns)

| Pattern | How often (INTERPRETATION, from reading) | Example papers | Research value |
|---|---|---|---|
| "Single node only; multi-node is future work" | Very common in fault tolerance and overlap papers | FailSafe, AnchorTP, Concordia, ParallelKittens, Weave, Flying Serving, mKernel | Low. Extension work, needs a cluster |
| "Other models, GPUs or dtypes not evaluated" | Most papers | TBIK (no quantized dtypes), Integer Alibi (one kernel pair), DASC (TP <= 8) | Low. Evaluation extension (D-29) |
| "Python prototype; kernel integration is future work" | Common in 2026 preprints | AVMP, Sparse Prefix Caching, LinearKV | Engineering |
| "Upstream engine changes needed" | Common in recovery papers | Dynamo recovery (request state), Fail-Closed Lowering, Belayer (flush-free KV) | Usually engineering, often done by upstream within weeks |
| "Open question about cost attribution or fairness" | A few | DLPM, PrefixShield | Checked below: taken |
| "Open mechanism question" | Rare | Runtime Observability (eviction mediator), Software Aging (growth triggers), KV integrity | Checked below |

FACT (from the extracted text): the most repeated future-work items are scale-out to multi-node,
more models or hardware, and upstream integration. These are not research gaps for this thesis.

## 2. Candidate follow-up gaps and verdicts

Each row is a follow-up that one or more papers name as not done. "Source" is the paper that leaves
the gap. The verdict uses the journal status words.

| ID | Follow-up gap | Source papers that leave it open | Check result | Verdict |
|---|---|---|---|---|
| I-141 | Reuse in-flight requests' KV after a serving-process crash when device memory survives | Belayer ("flush-free KV recovery needs fine-grained consistency tracking"); Dynamo 2609.25451 ("preserving request state ... left as ongoing future work"); LUMEN (one KV checkpoint per request) | Simple-baseline attack: paged KV is append-only, so a committed-length journal written after each step's CUDA event makes reuse safe; torn writes land beyond the committed length. In-place recurrent state is covered by block-boundary state checkpoints plus bounded replay (Sparse Prefix Caching, SuffixReplay). Dynamo argues the dominant loss is serving capacity, not request progress. AnchorTP already keeps KV in a daemon across crashes | KILL (baseline, engineering) |
| I-142 | Fair split of a shared prefix's cost among tenants (Shapley-style) | DLPM 2501.14312 ("which client should pay"); PrefixShield 2608.01657 (multi-principal ownership) | Public repo srujan20/prefix-cost-attribution (2026-08-25) computes exact Shapley over the prefix trie, shows order independence and the impossibility under recomputation. Tree cost games have a classical closed form (airport game). "Who Pays for the KV Cache?" 2609.24991 covers metering | KILL (scoop by public artifact; low systems depth) |
| I-143 | TP-size-invariant deterministic kernels for block-scaled FP8 and other quantized dtypes | TBIK 2511.17826 (ICML 2026): "a natural extension ... quantized data types" | No public FP8 TP-invariant result found in one search. It is an extension of a published method (D-29 ranks this low). It needs 2 or more GPUs. Determinism is crowded (LLM-42, CoRun, ReproFuse, Vosti) | WEAK (GPU only, extension) |
| I-144 | Aging-aware rejuvenation policy for LLM serving engines | Software Aging 2606.11916 ("growth triggers remain open") | Baseline attack: Dynamo shadow engines restore service in 7.3 s (NVIDIA blog) and SGLang reports sub-second warm switches, so a periodic restart at low load captures the benefit. A policy has little room | KILL (baseline) |
| I-145 | Branch-scoped KV so an aborted agent branch leaves no residue | Aborted but Not Forgotten 2608.15939 (structured defenses future) | The fix is to exclude aborted-branch blocks from prefix reuse or salt them by branch id. Copy-on-write branch caches exist (ForkKV, EAGLE-Pangu) | KILL (engineering) |
| I-146 | Integrity protection for high-fan-out shared KV blocks | Bit-Flip Vulnerability 2604.17249 (reachability and defenses open) | Datacenter HBM has ECC; the threat needs non-ECC GPUs. vLLM PR 61578 adds KV checksums for transfers. A fan-out-weighted checksum is a small extension | KILL as thesis (narrow threat model) |
| I-147 | Erasure-coded KV protection for decode-heavy and P/D-disaggregated serving | GhostServe 2605.00831 ("how to provide fault tolerance for both prefill and decode workers remains open") | GhostServe owns the mechanism for prefill. LUMEN and FailSafe cover recovery. Needs a multi-GPU cluster | KILL (extension, infeasible here) |
| I-148 | KV tier redesign when host memory is not larger than HBM (Grace GB200 and GB300) | HiSparse 2608.07009 (host-memory capacity limit) | NVIDIA ICMS standardizes HBM, CPU, NVMe tiers. Needs GB200 or GB300 hardware | KILL (vendor-owned, infeasible) |

Gaps found but not registered, because they are already in the registry or are pure extension:
- Multi-node versions of FailSafe, AnchorTP, Concordia, Weave, Flying Serving: extension, needs cluster.
- Adaptive SM partition selection for MoE comm overlap (Tile-level MoE 2607.19539): Weave 2609.21483 does it.
- Speculative write-tool execution with sandbox rollback (SPORK 2607.03333): I-51 and I-53 area, crowded.
- Compression-friendly collective library (UCCL-Zip 2604.17172): I-69 area, crowded.
- Hardware dirty-page tracking for GPU checkpoints (Concordia): hardware.
- Prefix-cache admission plus placement for hybrids (Sparse Prefix Caching): I-129 area, KILL.
- Periodic KV refresh against FP16 cache divergence (Illusion of Equivalence 2604.15409): I-58 and I-73 area.
- In-flight request preservation across wide-EP membership repair (EEP 2605.10670): needs a 32-GPU cluster.

## 3. Closest call: I-141 in detail

This is the gap that fits the user's checkpoint/restore background best. Three independent papers
name it.

**Why it looked open.** FACT: Dynamo states that its implementation "replays interrupted requests
rather than preserving request state" and leaves preservation as future work. FACT: Belayer says that
recovering the KV cache without refilling "would require fine-grained consistency tracking". FACT:
LUMEN keeps one KV checkpoint per request and falls back to recomputation.

**Why it fails.** INTERPRETATION, by first principles:
1. In a paged engine, step t writes only the slot of token t in the request's last, mutable block.
   Full blocks are immutable. If the process dies mid-step, only slots beyond the committed length
   can be torn.
2. A journal that records the committed token count and block table after the step's CUDA event
   completes is enough. On recovery, reuse KV up to the committed length and re-run one step.
3. In-place recurrent state (Mamba, linear attention) can be torn, but serving engines with hybrid
   prefix caching already keep state copies at block boundaries. Replay of fewer than one block of
   tokens restores it.
4. The prefix-cache index is a hash of token ids. It can be rebuilt from the journal.
5. Dynamo's own analysis says a failure mostly costs ready serving capacity, not request progress.
   The oracle gain of preserving request state is therefore bounded by one re-prefill per in-flight
   request, which prefix-cache hits on the surviving pool already reduce.

**What would reopen it.** A measurement on real hardware showing that (a) a crash leaves torn writes
inside committed blocks (for example through asynchronous KV offload or compaction kernels that move
committed blocks), and (b) the re-prefill cost after crash is a large share of recovery time at a
realistic failure rate.

## 4. Conclusion

FACT: R-36 read the stated limitations of 372 recent papers and checked eight follow-up gaps in
detail. Seven are KILL and one is WEAK (I-143). No gap reached MEASURE-FIRST.

INTERPRETATION: limitation sections mostly list scale-out, more hardware and integration work. When a
paper names a real mechanism gap, upstream projects or other groups often close it within weeks
(Dynamo, vLLM, the Shapley repo). This repeats the R-18 lesson: published text shows bugs and
extensions, not open research gaps.

Recommendation (D-53, open): do not run another literature-based gap search. The live options are the
GPU measurements that R-35 and E-36 already prepare (I-23 Test A; I-118 E-30). I-143 can ride on the
I-23 matrix only if a second GPU is rented.

## 5. Sources (closest works)

| Work | Why it matters | Check |
|---|---|---|
| [Dynamo fast recovery, 2609.25451](https://arxiv.org/abs/2609.25451) | Leaves request-state preservation as future work; argues capacity loss dominates | partial (limitation sections and targeted passages of the HTML) |
| [Belayer, 2608.14635](https://arxiv.org/abs/2608.14635) | Leaves flush-free KV recovery as future work | partial (limitation sections and targeted passages of the HTML) |
| [LUMEN, 2606.17787](https://arxiv.org/abs/2606.17787) | One KV checkpoint per request | partial (limitation sections and targeted passages of the HTML) |
| [AnchorTP, 2511.11617](https://arxiv.org/abs/2511.11617) | Daemon keeps KV and weights across process crash | partial (limitation sections and targeted passages of the HTML) |
| [Sparse Prefix Caching, 2605.05219](https://arxiv.org/abs/2605.05219) | Recurrent-state checkpoints at chosen positions | partial (limitation sections and targeted passages of the HTML) |
| [DLPM, 2501.14312](https://arxiv.org/abs/2501.14312) | Charges the first client for a shared prefix; leaves the split open | partial (limitation sections and targeted passages of the HTML) |
| [PrefixShield, 2608.01657](https://arxiv.org/abs/2608.01657) | Leaves multi-principal ownership open | partial (limitation sections and targeted passages of the HTML) |
| [prefix-cost-attribution repo](https://github.com/srujan20/prefix-cost-attribution) | Exact Shapley over the prefix trie, 2026-08-25 | full-text (README) |
| [Who Pays for the KV Cache?, 2609.24991](https://arxiv.org/abs/2609.24991) | Tenant metering rules | checked (abstract) |
| [TBIK, 2511.17826](https://arxiv.org/abs/2511.17826) | TP-invariant kernels; quantized dtypes left open | partial (limitation sections and targeted passages of the HTML) |
| [Software Aging, 2606.11916](https://arxiv.org/abs/2606.11916) | Aging in GPU serving; growth triggers open | partial (limitation sections and targeted passages of the HTML) |
| [NVIDIA shadow engine blog](https://developer.nvidia.com/blog/restore-llm-inference-capacity-in-seconds-with-shadow-engine-recovery-in-nvidia-dynamo/) | 7.3 s recovery versus 283 s cold restart | search-only |
| [Aborted but Not Forgotten, 2608.15939](https://arxiv.org/abs/2608.15939) | KV residue after rollback | partial (limitation sections and targeted passages of the HTML) |
| [Bit-Flip Vulnerability, 2604.17249](https://arxiv.org/abs/2604.17249) | Shared blocks have no integrity check | partial (limitation sections and targeted passages of the HTML) |
| [GhostServe, 2605.00831](https://arxiv.org/abs/2605.00831) | Erasure-coded KV; P/D fault tolerance open | partial (limitation sections and targeted passages of the HTML) |
| [HiSparse, 2608.07009](https://arxiv.org/abs/2608.07009) | Host tier smaller than HBM on Grace systems | partial (limitation sections and targeted passages of the HTML) |
