# REVIEW R-17 SHORTLIST (hostile review, 2026-10-08)

Reviewer role: senior systems PC member. Goal: kill weak ideas with evidence.

## Labels
- FACT: I opened the source (PDF text, HTML page, or code file) and read the line.
- SNIPPET: I saw the claim only in a search result. I did not open the page. Treat it as unverified.
- INTERPRETATION: my own reasoning or arithmetic.

## Summary table

| ID | Candidate | Verdict | One-line reason |
|----|-----------|---------|-----------------|
| S1 | Flash write filter for KV SSD tier | KILL | The idea is already named, shipped, or published in 4 places. The SSD tier it protects has little value. |
| S2 | Rank-safe block-max pruning for DSA indexer | WEAK | No direct prior work. But the bound formula is wrong for negative weights, and the bound is probably too loose to prune. |
| S3 | Scale-aware ABFT for MXFP8 / NVFP4 | KILL | Block-scaled GEMM on dequantized values is the BF16 case. V-ABFT already covers FP8. |
| S4 | Mixed-criticality scheduling for LLM serving | KILL | The HI-mode budget is vacuous. Priority plus preemption plus max_tokens is the baseline. |
| S5 | Multi-branch speculative frames for world models | KILL | Branches multiply a multi-GB state and GPU time. Co-batching gives 1.16x at B=4. |
| S6 | Hazard-rate (Gittins) KV retention for tool calls | KILL | The 2026 literature already tests the conditional-history estimator and it fails. |

Ranking of survivors (only S2 is not killed): S2 first. Then S1, S4, S6, S3, S5 in order of residual value. See the ranking section.

---

## S1. Flash write filter for the KV SSD tier

### 1. Closest prior work
- FACT. "When Fancy Eviction Fails" (arXiv 2609.28870, Sep 2026). Section 6.3 says an SSD tier from HBM is "generally unnecessary for most workloads" because the working set churns away first. It says a practical SSD tier "must employ strict selective admission". The same paper proposes quick demotion of one-hit prompts and compute-aware eviction. So your mechanism (a) and the filter idea are named there, but not built.
- FACT. "Enabling HBF for Generative Recommendation Serving with Write-Aware KV Cache Policy" (2609.07175). It uses admission-controlled LRU-K. It filters low-reuse users before cache admission. It decouples cache writes from misses. It extends HBF lifetime from about one year to over six years with LRU-10. This is Flashield-style write filtering for KV, with a model.
- FACT. Berkeley HBF characterization (2609.39131 v2, 5 Oct 2026). Buffered cache-aware scheduling with 10% admission headroom lifts the HBF write lifetime estimate from 1.21 to 14.82 years.
- FACT. "HBF Sucks?" (2608.11668). It calls the HBF stream write-heavy and low-reuse. It asks for reuse-aware placement and a write budget.
- SNIPPET. SGLang HiCache ships `write_through_selective`: write to the next tier only after an access count threshold. This is the "admit on second access" baseline, already in production code. Open the SGLang HiCache design page to confirm the threshold semantics.
- FACT. "Where Should the KV Cache Live?" (2609.16215). In its simulator, "predicted-reuse" is byte-identical to recency. Reuse frequency wins for agent and document QA. Deep SSD never meets the latency SLO for chat and agent.
- SNIPPET. py-kvcache (2609.11744): external caching is a setup-specific admission decision. CrossKV (Sogang thesis): bypass cold SSD hits and recompute.

### 2. Rejection sentence
"Flashield-style admission for the KV SSD tier is the standard remedy in at least four 2026 papers and one shipped engine option, and the authors show no new mechanism or evidence that the tier is worth having."

### 3. Simplest-baseline attack
- Baseline A: admit on second eviction or second access (`write_through_selective`). INTERPRETATION: this removes all one-hit blocks. It loses only the first re-read of each block.
- Baseline B: no SSD tier. FACT (2609.28870, 2609.16215): for chat and agent traces the SSD adds little.
- Baseline C: LRU-K with K=2 (2609.07175).
- INTERPRETATION: a session-state rule ("session ended, do not write") is a one-line filter. The server knows session end only if the client signals it. Most APIs do not.

### 4. Oracle and premise attack (first-principles numbers)
- KV bytes per token, DeepSeek-V3 MLA, BF16. FACT: 576 elements x 2 B x 61 layers = 70,272 B. The DeepSeek-V3 hardware paper (2505.09343) states 70 KB per token. The shortlist number is right for V3 in BF16.
- But V3.2 in production uses an FP8 layout. SNIPPET (vLLM PR, CSDN write-up): 656 B MLA plus 132 B indexer key = 788 B per layer. INTERPRETATION: 788 x 61 = 48,068 B, about 48 KB per token. The indexer key size matches the reference code (128 FP8 values plus one FP32 scale, FACT from `model.py`).
- Write volume at 30k uncached tokens per second: 70,272 B gives 182 TB/day (the shortlist says 190). 48,068 B gives 125 TB/day. So the estimate is 30 to 35 percent too high for the FP8 layout.
- Throughput 30k uncached tokens/s per node. SNIPPET (secondary reports of DeepSeek's 2025-02 disclosure): 73.7k input tokens/s per H800 node including a 56.3 percent cache hit rate. INTERPRETATION: 73.7k x 0.437 = about 32k uncached. So 30k is a fair upper-mid figure for a prefill node. It is not a typical node average.
- SSD budget. 8 x 7.68 TB x 1 DWPD = 61.4 TB/day. Correct arithmetic. INTERPRETATION (datasheet not opened): enterprise mixed-use TLC drives are often rated 3 DWPD, which gives about 184 TB/day for 8 drives. SNIPPET: ScaleFlux claims 7 to more than 10 effective DWPD with FDP placement, an unverified vendor claim. So the "190 vs 61" gap is not fixed. The right budget depends on drive class.
- Premise problem. INTERPRETATION: a node with 2 TB of DRAM and 1.44 GB/s of FP8 KV writes holds about 23 minutes of blocks. With BF16 at 2.1 GB/s it holds about 16 minutes. FACT (2609.28870): multi-turn block lifetime has a median of 219 s, and working-set overlap falls to near zero beyond a two-hour gap. So DRAM already serves nearly all session reuse. The SSD only serves reuse gaps of about 20 minutes to 2 hours. That window is thin. The write filter protects a tier with small upside.
- Oracle gain. The oracle saves the writes of blocks never read again. In FACT-backed traces most blocks are one-hit, so the oracle saving is large in bytes. But the useful-hit gain from the tier is small. The thesis metric would be "writes saved at equal hit ratio", and baseline A already gets most of it.

### 5. Verdict
KILL. Narrowed question that remains (too thin for a thesis): "Does prefix-closed admission (child admitted only with its parent) beat per-block second-access admission on depth-weighted recompute savings?" This is a small refinement.

### 6. Premise check (optional, to confirm the kill)
Runs on the M4. Replay the Alibaba Bailian and Mooncake traces with block hashes. Simulate DRAM sizes with LRU. Count the fraction of DRAM-evicted blocks that are read again later, and the FLOPs saved by those reads.
- Kill thresholds (written before running): kill if baseline A gets 90 percent or more of the oracle write saving at equal hit ratio, or if fewer than 30 percent of SSD writes are wasted, or if SSD hits are below 5 percent of all hits.
- Edges to check before accepting "no effect": DRAM sizes from 0.25 to 12 TiB (the range in 2609.28870), trace horizon of at least several days (the 2-hour overlap decay needs long traces), block size 16 and 64 tokens, depth-weighted value (not only block count), agent traces separate from chat traces, and a trace with a persistent hot core.

---

## S2. Rank-safe block-max pruning for the DSA lightning indexer

### 1. Closest prior work
- FACT. DeepSeek-V3.2 paper (2512.02556), Eq. 1: I(t,s) = sum over H^I heads of w(t,j) * ReLU(q(t,j) . k(s)). The weight w(t,j) is a real number derived from the query. There is no sign constraint.
- FACT. Reference code (`DeepSeek-V3.2-Exp/inference/model.py`). `weights_proj` is a plain linear layer with no abs. The key is `wk` then LayerNorm, then RoPE on the first 64 dims, then a Hadamard rotation (`rotate_activation`), then FP8 with one FP32 scale. The config says `index_n_heads=64`, `index_head_dim=128`, `index_topk=2048`.
- FACT. LongCat Sparse Attention (2608.01662): the indexer uses MQA, "a single key is shared across all indexer heads" (confirmed). Table 1: at 1024K tokens the indexer is 90 percent of per-layer latency.
- FACT. Guess-Verify-Refine (NVIDIA, 2604.22312, Apr 2026). It is a bit-exact top-k for DSA decode. It uses the previous step's top-k as a threshold guess. It still scores every key. It gives 1.88x per-operator and up to 7.52 percent TPOT at 100K context. This is the strongest threat, because it already owns the "exact top-k" claim and the "warm start from the previous step" idea.
- SNIPPET. HISA (2603.28458): block mean-pooled coarse filter, then exact token scoring inside survivors. Not rank-safe. Reports above 99 percent IoU with the DSA selection. It was tested on DeepSeek-V3.2 and GLM-5.
- SNIPPET. StreamIndex (2605.02568): chunked partition-merge top-k, exact, no full score tensor. IndexCache and LongCat cross-layer reuse are approximate.
- I searched five phrasings ("rank-safe", "block-max WAND attention", "min max key block bound indexer", "lossless DSA indexer", "indexer pruning upper bound"). I found no paper that does min/max-box rank-safe pruning of the DSA indexer. This is an absence of evidence, not proof.

### 2. Rejection sentence
"The proposed bound is loose in 128 dimensions after a Hadamard rotation, so it prunes almost nothing, and the authors do not show real indexer keys."

### 3. Simplest-baseline attack
- GVR (exact, shipped in TensorRT-LLM) is already the exact baseline. A thesis result must beat GVR, not torch.topk.
- Radix-select is the older baseline.
- HISA-style coarse filter is the "fast but not exact" baseline. INTERPRETATION: if HISA reaches IoU above 99 percent, a reviewer asks why rank-safety matters. The quality loss is already near zero.
- FP8 to FP4 indexer keys halve the key reads with no bound logic. INTERPRETATION, not checked against 2026 work.

### 4. Oracle and premise attack
- "Up to 90 percent of per-layer latency at 1M" is correct but narrow. FACT: Table 1 is for batch size 4, qlen 1, BF16, K=2048, attention module only (indexer plus sparse attention). It is not the whole decode step. The MoE and dense layers are not in it. End-to-end the share is smaller. GVR reports only 7.52 percent TPOT gain at 100K from a 1.88x operator speedup, which shows how small the end-to-end share is at 100K.
- FACT. LongCat says the top-k selection dominates the indexer cost and the scoring is comparatively cheap (Section 4.1.2 text, near the page-level design). So skipping key reads alone does not fix the main cost unless the skip also removes candidates from the top-k stage. A block skip does remove them, but only if blocks are skipped.
- Key bytes: the shortlist says 128 B per token per layer. FACT from code: 128 FP8 bytes plus a 4 B scale = 132 B. Small error.
- The single shared key across heads is correct (FACT). Good for the metadata cost: one min vector and one max vector per block, so the metadata is 2 keys per block. With block size 16 that is 12.5 percent extra reads.
- The bound formula is wrong as written. The shortlist formula is sum_j w_j * ReLU(sum_d max(q_jd kmax_d, q_jd kmin_d)). That is an upper bound only if w_j is at least 0. FACT: w_j is signed. For a head with w_j < 0, the term w_j * ReLU(x) is largest when x is smallest. The correct bound uses the box minimum of q_j . k for those heads: w_j * ReLU(min over the box). A sign-split bound fixes the soundness. It makes the bound looser.
- Looseness. INTERPRETATION plus one synthetic test I ran (script `bm.py` in the scratchpad, numpy, Gaussian keys and queries, d=128, 64 heads, signed weights, L=32768, K=512, sign-split bound): 0 percent of blocks were prunable for block sizes 8, 16 and 64. The median block bound was 26x (B=8), 41x (B=16) and 57x (B=64) the k-th largest score. The cause is that the box bound grows like the L1 norm of q while the real score grows like the L2 norm, a factor near sqrt(d) for d=128. The Hadamard rotation and LayerNorm in the real indexer push keys toward isotropic Gaussian, which is the bad case. Real keys have structure (sinks, local window, clusters), so this synthetic test is a pessimistic lower bound on prunability, not a measurement.
- RoPE makes it worse. The first 64 dims of q and k carry RoPE. Keys in one block sit at different positions, so those coordinates rotate. The min/max range of a rotating coordinate is wide. INTERPRETATION.
- Rank-safe also needs a definition. The reference kernel computes in FP8 with FP32 accumulate. Ties and rounding make the "exact" top-k depend on the kernel. GVR defines exact as equal to the production radix-select output.
- The upper bound for the oracle: even a perfect pruner can skip at most the fraction of blocks whose true max score is below the k-th score. This is not computed. It needs real keys.

### 5. Verdict
WEAK. It is the only candidate with no direct prior work. But the premise (tight enough bounds in 128 dims) is probably false, and the main competitor (GVR) is exact and shipped.
Narrowed research question if it survives: "On real DeepSeek-V3.2 indexer keys, what fraction of key bytes can a sign-correct, rank-safe block bound skip when the threshold is warm-started from the previous decode step, and does it beat GVR end to end at 128K to 1M context?"

### 6. Cheapest decisive premise check
The M4 with 16 GB cannot run DeepSeek-V3.2 (about 690 GB in FP8). I found no small open DSA model (SNIPPET from search: none found for GLM-5 or MiniMax-M3 details; MiniMax-M3 uses a block-index design, not DSA). So the Mac can run only the synthetic test above. That test is not decisive.
Two options for a decisive check:
- Option A (no GPU on hand): look for a published dump of indexer q, w, k tensors. Check the GVR GitHub repository linked from 2604.22312 and the HISA and LongCat repositories. If a dump exists, run the check on the Mac with numpy.
- Option B (rented GPU, a few hours): run DeepSeek-V3.2 on a rented 8-GPU node. Capture the indexer keys, queries and weights at 4 layers (early, middle, late, last) for 20 long prompts. Do the bound analysis offline on the Mac.
Kill thresholds (written before running):
- Kill if the sign-correct block bound skips fewer than 50 percent of blocks at B=16, L=128K, K=2048.
- Kill if net key-byte savings (skipped bytes minus metadata bytes) are below 25 percent at L=128K.
- Kill if the skip rate with a cold start (threshold from the heap only) is below 20 percent of the warm-start rate at L=128K, because decode needs the heap to fill first.
Edges that must be checked before accepting "no effect":
- Context length: 32K, 128K, 512K, 1M (skip rate rises with L because K is fixed).
- Block size: 8, 16, 32, 64. Also tighter bounds: ball bounds or k-means cluster bounds instead of boxes.
- Layers: early, middle, late. Heads: all 64, and a per-layer count of negative-weight heads.
- Prefill and decode separately. Query position inside the context. Prompt type: code, chat, needle-in-haystack, long document.
- K = 512 and 2048. Metric power: use at least 20 prompts and report per-layer distributions, not one mean.
- Compare against GVR, not only radix-select.

---

## S3. Scale-aware ABFT for block-scaled FP8 / NVFP4 GEMM

### 1. Closest prior work
- FACT. V-ABFT (2602.08043). The text covers FP8: for FP8 inputs with FP32 accumulation and FP16 output, the threshold follows the output precision. Its table has an H100 FP8 (E4M3/E5M2) row. So the shortlist note "FP8 is a remark" understates it. Block-scaled MX and NVFP4 are not covered.
- FACT. FP-Sketch (2609.19758, Sep 2026). It verifies an unmodified half-precision GEMM after the fact with sketches. It localizes several errors. It reports 99.4 to 99.9 percent recovery of perplexity damage on Llama-2-7B with a probe cost of 0.78 to 3.06 ms. This is "ABFT without owning the kernel".
- FACT. Syndrome decoding for INT8 GPU GEMM (2609.19743, Sep 2026).
- FACT. Altenbernd et al. (2604.00726) and LLM-PRISM (2604.10390, SNIPPET) study SDC in LLM training with fault injection, including FP8.
- SNIPPET. Security analysis of microscaling formats under fault injection (RISC-V edge). It reports shared-exponent corruption as the catastrophic mode.

### 2. Rejection sentence
"A block-scaled GEMM computes an exact FP32 sum of dequantized products, so the checksum problem is the known BF16 or FP32-accumulate case, and the authors add only a scale-factor bookkeeping step."

### 3. Simplest-baseline attack
Run V-ABFT or FP-Sketch on the dequantized operands. INTERPRETATION: an E2M1 value times an E4M3 scale has 4 mantissa bits. It is exact in BF16 and FP32. So the dequantized GEMM is a BF16-input GEMM with FP32 accumulation. A corrupted scale factor is simply a corrupted element of the dequantized matrix. Any ABFT detects it as a normal error. No new fault surface needs new theory.

### 4. Oracle and premise attack
The maximum gain is detection of faults that a baseline checksum misses. A baseline checksum on dequantized data misses none that a block-scaled one would catch, by the argument above. The only open cost question is overhead of dequantizing for the check, which needs a Blackwell GPU. The FP-Sketch paper shows post hoc verification costs 0.78 to 3.06 ms against 0.35 to 12.47 ms GEMMs, which is a heavy cost in relative terms. A scale-aware method could lower it. That is an engineering result, not a research question.

### 5. Verdict
KILL. No narrowed question that I would defend.

### 6. Premise check
Not run. A CPU emulation on the Mac would only confirm the exactness argument. If you want to confirm the kill, check that the dequantized product is bit-exact in FP32 for random NVFP4 data (one hour of numpy work).

---

## S4. Mixed-criticality scheduling for LLM serving

### 1. Closest prior work
- FACT (abstract opened). Adaptively Robust LLM Inference (2508.14544): Amax schedules on the upper bound of predicted length. Amin starts from the lower bound and adapts. It proves a log-scale competitive ratio.
- FACT (abstract opened). Queueing-aware optimization of reasoning tokens (2601.10274, Jan 2026): per-task thinking-token budget as the control variable, M/G/1 model.
- SNIPPET. Niyama (eager relegation under overload), ProServe, Llumnix (binary priority), Beyond Binary Priorities (2608.16336, 10 priority levels), Medha (length-aware slack), FastServe (MLFQ), Occamy and Cerberus (SEC 2026, edge AI, relax latency of low-criticality apps). Not opened.
- Search of five phrasings found no LLM paper with an explicit LO/HI mode switch on a length overrun. This is an absence in snippets only.

### 2. Rejection sentence
"The HI-mode length budget in an LLM is max_tokens, which is too large to reserve for, so the mixed-criticality guarantee is vacuous and the result reduces to priority scheduling with preemption."

### 3. Simplest-baseline attack
Priority queue plus swap or recompute preemption plus a per-class max_tokens cap. vLLM ships priority scheduling. INTERPRETATION: this gets most of the HI-class protection. The extra gain from a mode switch comes only from the shed action, and shedding already exists as eager relegation in Niyama (SNIPPET).

### 4. Oracle and premise attack
Vestal's model needs C(LO) < C(HI) with a schedulability test in both modes. For LLM decode, C(HI) is the max_tokens cap (thousands of tokens). A HI-mode test that reserves for that cap admits almost nothing. Without a usable C(HI) the formal value is lost, and only the heuristic remains. INTERPRETATION. Also the overrun trigger is observed late: a request that has already produced its LO budget of tokens still needs, on average, many more tokens (heavy tail). The oracle (perfect lengths, SRPT) is the upper bound. Prior work already closes much of that gap with predictions.

### 5. Verdict
KILL as stated. See variant V1 below for the only repair I found.

### 6. Premise check
Not needed for a kill.

---

## S5. Multi-branch speculative frames for interactive world models

### 1. Closest prior work
- FACT. "Stateful Worlds, Stateless Elasticity" (2607.10389). The live state is a multi-GB attention cache, almost all rewritten each step. A session pins one GPU. Co-batching "B=4 buys 1.16x throughput at 3.5x the quantum". It also lists planner fan-out (MPC or tree search forking short branches off a live state) as a tenant class and reports 52 of 52 planner bursts completed.
- SNIPPET (shortlist, not opened by me): Next-Frame Diffusion speculative sampling (2506.01380).
- SNIPPET. Speculative Actions (ICLR 2026, agents): top-K action prediction, up to 55 percent next-action accuracy, up to 20 percent latency cut. Outatime (MobiSys 2015) is the origin and is confirmed as a speculative multi-frame renderer that masks up to 120 ms.

### 2. Rejection sentence
"Speculating over k actions multiplies a multi-GB state copy and k times the GPU work, and the cited co-batching data shows no throughput headroom."

### 3. Simplest-baseline attack
Single-branch speculation on the most likely action (Next-Frame Diffusion), plus input-latency hiding in the client such as local late-warp of the last frame. INTERPRETATION.

### 4. Oracle and premise attack
Best case: perfect action prediction hides the full chunk and network delay. The cost is at least k times the compute of one branch, and the state fork. FACT: B=4 gives 1.16x, so four branches cost near 3.4 times the time of one. If the GPU is already at the frame deadline, there is no spare capacity for branches.

### 5. Verdict
KILL.

### 6. Premise check
Not needed.

---

## S6. Hazard-rate (Gittins) KV retention during tool calls

### 1. Closest prior work
- FACT. Continuum (2511.02230, ICLR 2026), read from the HTML: TTL is set per finished request. It maximizes P(tau, f) x Benefit(r) minus Cost(tau, r). P comes from the empirical CDF of past durations of that tool. Candidates are the recorded durations. Cold start has three tiers with K=100. The shortlist claim "Continuum uses a static TTL threshold" is wrong. The one true gap: the text I read does not condition on elapsed time.
- FACT. ThunderAgent (2602.13692): exponential time decay f(t) = 2^-t. The proof assumes memoryless latencies (Theorem F.1). Its Appendix C says tool latency is heavy-tailed and "prediction intrinsically brittle".
- FACT. "Ask the Tool, Don't Guess" (2609.18849, Sep 2026). It tests an in-flight conditional-history estimator "in the manner of 3Sigma's conditioning on elapsed time". This is the hazard-rate idea. Median remaining-time error was 32.6 percent of call duration at 50 percent through the call, and 82.0 percent at 90 percent through. Continuum's CDF scored 50.0 and 96.8 percent. Tool-reported progress scored 21.8 and 9.0 percent. With hints, p90 TTFT after a tool call fell 20.7 percent (HBM only) against LRU, close to an oracle.
- FACT. The same paper lists InferCept's elapsed rule (a call has as long left as it has run) and per-tool profiles. SNIPPET: SAGA (2605.00528) sets TTL to the p-th percentile of a per-tool latency distribution. Hermes (2506.14851, SNIPPET) already applies Gittins to LLM application scheduling.

### 2. Rejection sentence
"The conditional remaining-time estimator is the 3Sigma and InferCept family, it was implemented and beaten in 2609.18849, and the oracle headroom over LRU is only about 21 percent p90 TTFT."

### 3. Simplest-baseline attack
Continuum's empirical-CDF TTL, or SAGA's percentile TTL, or InferCept's elapsed rule. INTERPRETATION: a Gittins index adds a better policy given a distribution, but 2609.18849 shows the distribution estimate is the bottleneck, not the policy.

### 4. Oracle and premise attack
FACT: an oracle with true remaining time reaches only about the same p90 TTFT gain (20.7 to 20.8 percent) as their progress-hint method, against LRU. So the total headroom for any retention policy on that metric is near 21 percent. A hazard policy sits between Continuum and that oracle. The expected gain is a fraction of 21 percent. Also, a heavy-tailed distribution has a decreasing hazard rate: a call that has run long is likely to run longer. The Gittins rule then says "evict after a short patience time", which is what a TTL already does. INTERPRETATION.

### 5. Verdict
KILL.

### 6. Premise check
If you still want to confirm: build the per-tool timing table from the 459 Exgentic sessions. Fit the hazard shape per tool class. Run Continuum-CDF, InferCept-elapsed, hazard and oracle in the E-28 simulator. Kill if the hazard policy closes less than 30 percent of the gap between Continuum-CDF and the oracle, or if fewer than 5 tool classes have at least 100 samples (the 2609.18849 finding that few tools reach that count).

---

## Ranking of survivors

No candidate is a clean SURVIVES. Ranked by residual value:
1. S2 (WEAK). It has no direct prior work. It needs real DSA keys, so it waits for GPU access or a published dump. Expect the synthetic result (no pruning) to hold unless real keys are very structured.
2. S1 (KILL, thin residual). Cheap to confirm on the Mac in one or two days. A confirmed kill still produces a useful negative note for the journal.
3. S4 (KILL as stated). Repaired only through V1.
4. S6 (KILL).
5. S3 (KILL).
6. S5 (KILL).

## Proposed variants

I checked these against 2025 and 2026 preprints. I found one variant worth proposing. The second slot is filled with a gated plan, not a new idea, because I could not find a second candidate that survived a prior-art search. I list the dead ends I tried at the end.

### V1. Mode-switched imprecise computation for reasoning requests (S4 repaired)
Mechanism transfer:
- Vestal mixed-criticality (RTSS 2007) does the TRIGGER job. A high-criticality (HI) request that passes its LO length budget switches the system to HI mode. A back-switch rule (an idle instant, or hysteresis) returns it to LO mode.
- Imprecise computation (Liu et al., 1991) does the ACTION job. Each request has a mandatory part (the answer) and an optional part (the thinking trace). In HI mode, low-criticality requests have their thinking closed early (budget forcing, `</think>`), instead of being preempted. This keeps their KV cache and avoids recompute.
- A conformal bound on class-specific output length does the BUDGET job. It sets the LO budget C(LO). The C(HI) problem in the S4 attack is removed because HI mode does not reserve for max_tokens. It sheds optional work.
Why this is better than S4: the action is cheap (no KV loss), and the quality cost is measurable from public accuracy-versus-thinking-budget curves.
Prior-art check:
- FACT. 2601.10274 sets static per-type budgets in an M/G/1 model. No dynamic switch.
- SNIPPET. Princeton thesis on SLO-aware early exit, PowerSlider (2608.21719, admission control by observed length and thinking isolation), Cascade (per-request SLO budget), SLOWEAVE (deadline-aware prefill chunking), Niyama (relegates whole requests). None found with an explicit overrun-triggered mode switch that truncates optional thinking. Open PowerSlider and the Princeton thesis before committing. They are the closest threats.
Weakness (be honest): this is simulator-only on the M4, like I-59 and I-60. The overrun frequency of HI requests decides the gain. Kill if P(HI overrun of C(LO)) is below 2 percent in the traces, or if p99 HI latency improves under 15 percent over "priority plus static thinking cap" at equal LO accuracy loss.
Mac premise: simulator with output-length distributions from public reasoning traces and the Exgentic agent traces, plus published accuracy-versus-budget curves. Edges: load from 0.5 to 1.2 utilization, HI share from 5 to 50 percent, tail index of the length distribution, switch hysteresis, LO accuracy floor.

### V2. Conditional plan: S2 with a measured gate, not a new idea
I tried to find a stronger rank-safe design (low-bit surrogate with Cauchy-Schwarz error bound, Lipschitz drift bound across decode steps, as in distributed top-k monitoring). INTERPRETATION: both certificates are loose for the same reason as the box bound. The worst-case error sums over 64 heads linearly, while the score gaps grow like the square root of the head count. So I do not propose them. The plan that remains is S2 with the Section 6 gate, using GVR as the baseline. If the real-key check passes the kill thresholds, S2 becomes a good thesis chapter because it has an exactness guarantee that HISA lacks. If it fails, record the negative result in the journal.

### Dead ends I tried (so you do not repeat them)
- Software rejuvenation with checkpoint-restore for LLM servers. FACT: the 216-hour aging study (2606.11916) measured leak rates of 1.8 KB/h (vLLM), 31 KB/h and 157 KB/h. That is 1 to 110 MB over 30 days. The premise fails.
- AIMD adaptive concurrency for agent fan-out against API rate limits. FACT (snippets): HiveMind (2604.17111) already does AIMD backpressure with header-based limits. NVIDIA NeMo Data Designer ships AIMD.
- Server-side compaction timing for agents. SNIPPET: CliffCompaction (2609.26779) covers it.
- Bloom or cuckoo filter prefix directory for routers. INTERPRETATION: approximate prefix indexes already exist in gateway routers. Not searched in depth.

## Sources opened (FACT)
- arXiv 2609.28870, 2609.39131, 2609.07175, 2609.16215, 2608.11668 (partial grep), 2608.01662, 2604.22312, 2512.02556, 2505.09343, 2609.18849, 2602.13692, 2606.11916, 2607.10389, 2602.08043, 2609.19758, 2508.14544 (abstract), 2601.10274 (abstract), Continuum HTML (arxiv.org/html/2511.02230v2, first 100k characters).
- Hugging Face: deepseek-ai/DeepSeek-V3.2 config.json and DeepSeek-V3.2-Exp inference/model.py.
- Local script: scratchpad `bm.py` (synthetic bound test).

## Sources seen only as search snippets (SNIPPET)
- SGLang HiCache docs (write_through_selective), HISA 2603.28458, StreamIndex 2605.02568, SAGA 2605.00528 (partial text opened for grep only), Hermes 2506.14851, py-kvcache 2609.11744, CrossKV thesis, ScaleFlux announcement, DeepSeek 2025-02 inference statistics (secondary write-ups), vLLM FP8 layout PR, CSDN write-up, Niyama, ProServe, Llumnix, Occamy, PowerSlider 2608.21719, Cascade, SLOWEAVE, CliffCompaction 2609.26779, HiveMind 2604.17111, Speculative Actions (ICLR 2026), Outatime, 2609.19743 (abstract only via search), 2604.10390.
