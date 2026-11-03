# Chapter draft: Does storing the Gemma 4 K=V cache once make decode faster? An exact memory mechanism and a closed byte route (I-182)

Status: DRAFT 1, 2026-11-01; accuracy update after E-53 and R-74 (2026-11-01); E-54 result (PASS) after R-75, and
the R-76 corrections (2026-11-02). Source of truth: `RESEARCH_JOURNAL.md` (I-182, E-49 to E-54, R-64 to R-76, D-73 to D-94, Section 7.30). Every number comes from a repository file, and each list or paragraph names its source; "R-72 C4" is
correction C4 of review R-72. Labels: FACT (measured or read from a source), INTERPRETATION (our reading), HYPOTHESIS
(not tested). A label before a list or a table covers each item that has no label of its own. Thresholds are project
heuristics, not literature standards.

## 1. Summary

FACT (R-64): the global attention layers of Gemma 4 have no V projection. Their key is a weighted, partly rotated copy
of the value. FACT (journal Section 5.27): free source reading found this lead while we planned the D-73 waste audit.

FACT (journal Section 5.27; the E-49 to E-52 READMEs, Section 13; R-72):
- A compact row of 640 values per head and token, in place of 1,024, is exact in real arithmetic. It cuts the global
  KV cache by 37.5%, as the Gemma 4 report states. No engine that we checked stores the cache in this form.
- On one H100, the 31B ceiling of the decode-step saving is 6.65% (32K context) and 11.33% (64K), with a BF16 KV
  cache at batch 4. On top of an FP8 KV cache near the KV pool limit, it is 12.1% and 15.0%.
- About 80% of the context-growing FP8 attention time does not depend on the DRAM bytes. The DRAM time is also 87% to
  90% of it (a co-limit), so 37.5% fewer bytes cut that time by only 4% to 10%.
- That is about 3% of the step near the pool limit, below the 5% bar. These numbers are exploratory, because E-51
  and E-52 are INVALID by their frozen rules.
- The unchanged stock FA4 FP8 kernel of vLLM reads the compact rows through two overlapping strided views. In a
  microbenchmark it has the same error and a 7% to 10% shorter context-growing time.
- INTERPRETATION: the byte route to speed is closed for these H100 kernel classes. The value of I-182 is capacity:
  1.39 to 1.47 times the requests in the same FP8 pool (E-50, arithmetic, not measured).

FACT (journal Section 10): the user decided D-88 on 2026-11-01. It asks for this chapter now and a gated vLLM
prototype with a hard stop at the end of week 6 (Section 7).

FACT (E-53 README Section 13; R-74): gate G3 ran on real 31B activations on 2026-11-01. At KV scale 1.0 the compact
FP8 row has 0.57 to 2.38 times the attention error of the standard FP8 cache per global layer, and layer 41 fails
the preregistered bound (FAIL). INTERPRETATION (R-74): the standard FP8 error itself changes up to 2.4 times with the
grid phase of the KV scale. With tied per-layer scales (ks = w x vs) the compact row has 0.72 times the stock error
(POST-HOC, on the same prompts). FACT (E-54 README Section 13; 2026-11-02): E-54 tested the tied row on 7 fresh
prompts against the stock FP8 cache and the best standard FP8 cache of the same scale search (amendment A1 after
R-75: the K and V scales searched jointly). The frozen decision is PASS: R1 0.670 and R2 1.131 (bound 1.25). The
tied row is better than the stock cache in every layer by the median over prompts (66 of 70 prompt-layer cells), and
the stock FA4 kernel reads it in decode calls (MATCH). R-76 (post-run review, CONFIRM_WITH_CORRECTIONS) recomputed all
70 cells to 2.4e-4 with its own code. It limits the claim: the tied row costs 13% against the best calibrated
standard cache, the scales must be reproduced to 0.1% to 0.25%, and the sample is 7 prompts of technical text.

## 2. The mechanism

### 2.1 K and V in a Gemma 4 global layer

FACT (R-64 Section 1; HF transformers, vLLM, llama.cpp, TensorRT-LLM and the DeepMind JAX code agree):
- A global layer has no `v_proj` (`attention_k_eq_v`). V = RMSNorm(W_k x), with no learned weight.
- K = rope_p(w * V). Here w is the k_norm weight: one vector of 512 values per layer, shared by all KV heads.
- p-RoPE with p = 0.25 rotates 64 of the 256 angle pairs: dims 0 to 63 and 256 to 319. The other pairs have zero
  frequency.
- So K equals w * V on the 384 unrotated dims. The 128 rotated dims differ, because their rotation depends on the
  token position.
- The 31B model has 10 global layers of 60, with 4 KV heads and head_dim 512. The attention scale is 1.0, and the
  attention has no soft-cap.

### 2.2 The compact row

FACT (E-49 README Section 2; `experiments/gemma4_kv_audit/results/layout_check.json`; R-64):
- The row stores V (512 values) and K on the 128 rotated dims: 640 values per head and token.
- The query absorbs w on the unrotated dims, because dot(q, w * V) equals dot(q * w, V).
- The saving is 1 - 640 / 1,024 = 37.5% of the global KV bytes, and the global KV capacity rises 1.6 times.
- `kv_layout_check.py` gives the standard attention with a float64 relative error of 3e-15 to 8e-15.
- The layout is not bit-exact in BF16. On heavy-tailed keys, the median error against float64 was 1.1% (standard)
  and 1.4% (compact). In FP8, the rounding of K on the unrotated dims comes from V (O-14).
- INTERPRETATION (R-64): so a prototype needs its own accuracy gate.

INTERPRETATION: this is the MLA cache shape: one shared row plus a decoupled RoPE part.

### 2.3 The views layout (E-52)

FACT (E-52 README Sections 2, 4 and 10; R-71):
- The row order is [K_rot (128) | V_unrot (384) | V_rot (128)].
- The K view is values 0 to 511. The query takes the same order and carries w on its unrotated part.
- The V view is values 128 to 639. The two views share the 384 values of V_unrot.
- An unchanged attention kernel reads the row through these two strided views. FA4 keeps aligned strided views as
  they are. An independent check of the layout agrees to 1e-15 in float64 (R-71).
- The output order is [unrotated dims | rotated dims]. An engine can fold this order into the o_proj columns, which
  is exact for BF16 weights. Block-scaled FP8 or NVFP4 weights need a block-aligned permutation, a re-quantization or
  a run-time permute.
- Two FP8 scales: if K_rot has the scale s_k and V has s_v, pass s_v as both descales. Then multiply the rotated
  query part by s_k / s_v (a CPU test checks this rule).

## 3. Prior art and the engine gap

FACT (journal Section 7.30; R-64 and R-65 pass 2; R-69):

| Work | What it shows | Relation to I-182 |
|---|---|---|
| Gemma 4 report, arXiv 2607.02770, Section 2 | K=V with p-RoPE cuts the global KV cache by 37.5%; no inference layout | The saving is known |
| Kayyam, Gopal, Lewis, arXiv 2606.04032 | The K=V design | The origin of the idea |
| MLA (DeepSeek, FlashMLA); GTA, arXiv 2505.21487; Grouped Value Attention, arXiv 2609.13285 | A shared row plus a separate RoPE part; the last two are trained from scratch | The same cache-shape family, as architectures |
| vLLM PR #41944 (open) | Removes the duplicated V projection weights (+0% to 1.7%) | Weights only; the cache stays |
| FP8 or NVFP4 KV at head_dim 512 (vLLM PR #60372, #53175; SGLang PR #46965; FlashInfer) | FP8 KV halves all KV bytes; both are lossy | The competitor; it can stack |
| vLLM, SGLang, TensorRT-LLM, llama.cpp, google-deepmind/gemma, litert-torch | Separate full K and V caches | The engine gap |

- FACT: no search found an engine, PR or paper that stores the Gemma 4 K=V cache once. The last search was R-69 on
  2026-10-31. INTERPRETATION: absence in a search is not proof. Private serving code is not visible.
- FACT (R-72): the vLLM loader already maps the K weights into both the K and the V shards of these layers.
  INTERPRETATION: a duplicate upstream change is possible, so gate G2 searches again (Section 7).
- INTERPRETATION: the research novelty is low. We do not claim the saving or the cache shape as new. The contribution
  is the engine mechanism and its measured effect, with the negative speed result.
- FACT: the journal records no prior-art search for the views read. So we do not claim it as new either.

## 4. The experiments

### 4.1 Platform and metrics

FACT (the four READMEs, Section 4; `experiments/runpod/HARDWARE_LOG.md`): each run used a different H100 80GB HBM3
SXM ($3.99/h), after an RTX 4090 rehearsal. All runs used vLLM 0.33.0rc1.dev229+g3966924d5 and Triton 3.8.0. The
models were `RedHatAI/gemma-4-31B-it-FP8-dynamic` (31B) and `google/gemma-4-26B-A4B-it` (26B-A4B, BF16). E-49 and
E-50 timed decode-only engine steps, and E-51 and E-52 timed one 31B global layer.

Metrics (FACT, the preregistrations):
- S(X) = 0.375 x b_low x X / T(X). T is the median step time at context X, and b_low is the one-sided 95% lower
  bound of its slope. S is a ceiling, because it assumes that time follows bytes.
- dt = t(X) - t(2K): the context-growing time of one kernel at the group batch (B 19 at 32K, B 11 at 64K).
- sigma_layout = 1 - dt_compact / dt_standard. sigma_mma = 1 - dt_compact / (0.85 x dt_stock), with a 15% allowance
  for the coming stock FP8 MMA kernel.
- e = 2 x (1 - dt_oracle / dt): the time elasticity to DRAM bytes (Section 6.2). E-52 calls it e_h for its standard
  Triton kernel.
- Kill lines (R-68): sigma 0.155 (32K) and 0.125 (64K) give S = 5%. The go line, sigma 0.25, gives about 8%. The
  e-lines (0.413 and 0.333) are the e values at which 0.375 x e reaches the kill lines.

FACT (journal Section 10): the user approved each spend (D-74, D-75, D-81, D-82, D-84, D-86, D-87).

### 4.2 E-49: the decode context sweep and the first ceiling

FACT (E-49 README Sections 1 and 13; R-65): E-49 asked if the duplicated cache costs 5% or more of the step at 32K
and 64K. The run cost $2.09, plus $0.21 for the rehearsal.

| Quantity | 31B (FP8 weights, BF16 KV), B 4 | 26B-A4B (BF16), B 8 |
|---|---|---|
| S at 32K (one-sided 95% bounds) | 6.65% (6.53% to 6.76%) | 6.09% (lower bound 4.56%) |
| S at 64K | 11.33% (11.14% to 11.53%) | Lost (probe fault) |
| Global attention read rate | 3,139 GB/s (1.03 x copy) | 3,059 GB/s (1.00 x copy) |

- FACT: the frozen decision is PIVOT ("one model or one cell only"). The 31B model passes both cells with every check.
  Attention explains 98.6% of its device-time growth, and its sliding layers stay flat (-2.1% and -3.5%).
- FACT (R-65; journal Section 8.1 row 69): the probe budget (`max_tokens` 236) was too small for chunked prefill, so
  the 26B 64K cells failed. R-65 corrected our first cause, an end token.
- INTERPRETATION (R-65): on an ideal FP8 KV cache, the compact cache would add only about 3.8% (32K) and 7.0% (64K)
  at batch 4. This estimate set the E-50 question.

### 4.3 E-50: the stack on an FP8 KV cache near the pool limit

FACT (E-50 README Sections 2 and 13): E-50 sized the batch to fill about 85% of the KV pool, with the stock FP8 KV
cache. This path runs FA3 on the sliding layers and the FA4 FP8-dequant kernel on the global layers. The run cost
$2.92, plus $0.24.

| Arm | Batch (pool share) at 32K; at 64K | S at 32K (lower bound) | S at 64K (lower bound) |
|---|---|---|---|
| 31B, FP8 KV (deciding) | 19 (0.82); 11 (0.83) | 12.1% (12.0%) | 15.0% (14.9%) |
| 31B, FP8 KV, batch 4 (R-65 estimate: 3.8% and 7.0%) | 4; 4 | 4.84% | 8.62% |
| 31B, BF16 KV near its pool limit | 10 (0.84); 5 (0.74) | 13.7% | 14.5% |
| 26B-A4B, BF16 KV, fixed token budget | 8; 8 | 6.08% (5.72%) | 10.46% (9.84%) |

FACT (E-50 README Section 13; R-68):
- The frozen decision is GO. The FP8 global attention reads at 0.70 and 0.69 x the copy bandwidth.
- Half the bytes (BF16 to FP8) cut the time per context token by only 24.5%, an elasticity of 0.49 (R-68 C1). A
  compact kernel must get 41% (32K) or 33% (64K) of the byte saving to keep S at 5%.
- INTERPRETATION: R-68 estimates the real saving at 4% to 8%.
- Per step, FP8 against BF16 depends on the batch (R-68 C2; row 70). FP8 is 5.8% slower at B 4 and 2K, but 5% faster
  at B 11 against B 10 and 32K.
- The FP8 pool holds 1.96 times the requests of the BF16 pool at 32K. At the measured pair batches, FP8 decodes 516
  against 338 tokens/s at 32K (+52%), and 313 against 184 at 64K (+70%).
- INTERPRETATION (R-68 C3): at equal pool share, the gains are about +53% and +60%.
- INTERPRETATION (arithmetic; R-67 O3, R-68): a compact global cache would fit 1.39 times (32K) and 1.47 times (64K)
  the requests in the same FP8 pool. No run measured this.

### 4.4 E-51: the compact kernel test (INVALID)

FACT (E-51 README Sections 4 and 13; R-70): E-51 timed two Triton split-KV kernels against the stock FA4 FP8 call.
One read the standard layout, and one read the compact row [V | K_rot]. Both cast FP8 to BF16. The run cost $1.42,
plus $0.14.
- The frozen decision is INVALID. V3 failed at one 2K cell (0.837 x the engine time; bound 0.85).
- V4 failed on two stock rows (replay spreads of 4.2% and 2.7%; bound 2%).
- R-70: 14.6% of the busy samples show the software power cap (0x4), which the throttle check did not count.

FACT (exploratory; R-70 Answer 2):

| Quantity | 32K, B 19 | 64K, B 11 |
|---|---|---|
| dt (us): stock FA4 FP8, Triton standard, Triton compact | 1,147.8; 1,015.6; 1,305.4 | 1,344.9; 1,230.5; 1,585.7 |
| sigma_layout (FP8) | -0.285 | -0.289 |
| dt read rate over copy: stock, standard, compact | 0.684, 0.773, 0.376 | 0.699, 0.764, 0.370 |
| K = V elasticity: stock, Triton standard | 0.202, 0.179 | 0.165, 0.196 |
| sigma_layout of the BF16 Triton pair | 0.337 | 0.368 |

FACT (R-70; journal Section 8.1 row 71):
- If V3 and V4 pass, the amended rule gives INCONCLUSIVE in 5 of 5 repeats. The rule before amendment A1 gives KILL,
  so R-69 stopped a false KILL.
- The defect is specific to FP8. At matched configurations, the compact kernel takes 1.04 to 1.55 x the standard
  time in FP8, but 0.63 to 0.80 x in BF16.
- INTERPRETATION (R-70): after the cast, the V tile feeds two dot roles that need different register layouts.
- The K = V oracle removes DRAM bytes only. So E-51 bounds the byte route (at most about 10% of dt), not the layout.
- INTERPRETATION (R-70): "bound on chip" has three candidates. They are the cast, the MMA with M padded from 8 to 64,
  and the one-stage staging chain of FA4.

### 4.5 E-52: FP16 compute and the views layout (INVALID)

FACT (E-52 README Sections 2 to 7; journal Section 7.30):
- FP16 compute: in the Triton 3.8.0 PTX, FP8 to FP16 is one instruction per 2 values. FP8 to BF16 on sm_90 needs
  two more.
- The views layout: the compact kernel runs the same compiled code as its twin, so only the DRAM bytes differ.
- After R-71 (blocker B1), KILL means the byte route only: sigma_layout and e_h below their lines in both groups.
- The run cost about $1.25, plus about $0.31 (estimates; billing pending).

FACT (E-52 README Section 13; R-72): the frozen decision is INVALID (V4). A short Triton row at 19:2K had a 5.2%
replay spread. Its slow repeats followed a stock row at the 700 W cap. This moves dt by 0.09% at most.

FACT (exploratory; R-72 Answer 1):

| Quantity | 32K, B 19 | 64K, B 11 |
|---|---|---|
| dt (us): stock, standard FP16, views, K = V oracle | 1,193.4; 910.0; 871.6; 811.5 | 1,405.7; 1,052.9; 984.1; 926.8 |
| sigma_layout (range of 5 repeats); kill line | 0.0422 (0.0416 to 0.0482); 0.155 | 0.0653 (0.0602 to 0.0685); 0.125 |
| e_h (range of 5 repeats); e-line | 0.216 (0.202 to 0.218); 0.413 | 0.240 (0.237 to 0.272); 0.333 |
| Oracle control e_bf16_triton (valid at 0.5 or more) | 0.856 | 0.817 |
| 1 - pool share: standard, views, stock | 0.827, 0.861, 0.780 | 0.796, 0.850, 0.799 |
| DRAM time at the copy rate over dt (standard) | 0.869 | 0.898 |
| Standard kernel, FP16 against BF16 compute (us) | 910.0 against 1,060.7 | 1,052.9 against 1,205.3 |

FACT (R-72), with V4 ignored:
- The rule gives KILL (byte route only) in 5 of 5 repeats, also with the E-51 stock baseline. The expected
  sigma_layout was 0.25 to 0.37, so H1 is rejected by a large margin (C14).
- At the six best 32K configurations, sigma_layout is 0.042 to 0.101, below the line (C11).
- The pool rows (all bytes from L2) cut dt by only 14% to 22% in three FP8 kernel families. The DRAM time of the
  standard kernel is also 87% to 90% of its dt. INTERPRETATION (C1): two floors overlap, so fewer bytes give little
  time.
- The cause of the floor is open. The data argue against an L2-to-SM limit and exclude the merge kernel, the split
  count and the tensor cores. They do not separate latency, occupancy and issue.
- H4 is false as preregistered (C2). Its rate half holds, but e_h is 0.22 and 0.24 against 0.6. FP16 compute cuts dt
  by 12% to 14% against BF16 compute, so the cast is not most of the time.
- `num_stages` 3 added no K or V buffer and was 2% to 3% slower, so H5 is untested (C3).
- The stock FA4 kernel reads the views with no code change, at the same error (0.00166). Its dt is 9.6% and 6.6%
  shorter, in 10 of 10 repeats. With a BF16 cache, it gains 24.0% and 24.5% on the views (single sweep rows).
- The stock views rows ran 5.7% higher in SM clock. INTERPRETATION (C13): up to about 1.6 points of the gain can be
  clock.
- The standard Triton FP16 kernel has a 19% to 25% shorter dt than the stock kernel, by baseline (C4). Of its 54
  configurations, 21 (32K) and 15 (64K) beat stock. With a BF16 cache, the two kernels tie.
- sigma_mma is 0.141 and 0.176 (go line 0.25). Near the pool limit, the stock views gain is 3.1% and 2.7% of the
  step.

FACT (R-72; journal Section 8.1 row 72): the bullets above carry C1 to C4, C11, C13 and C14. C5 notes that the BF16
pool share (0.48 and 0.46) is below its preregistered range of 0.5 to 1.0. C6 finds that no SASS exists, because
`nvdisasm` is missing in the image. C7 adds the row order to L-19, and C8 to C10 and C12 add scope limits to O-13 to
O-15 and T-11. No correction changes the frozen rule, the code or the decision word.

## 5. What the evidence supports and what it does not

Supported, with scope:
- FACT: the 37.5% saving is exact in real arithmetic, but not bit-exact in BF16 or FP8 (R-64, R-71, O-14).
- FACT: near the FP8 pool limit, the duplicated cache has a ceiling of 12.1% and 15.0% of the step (E-50).
- FACT (exploratory): on the H100 FP8 dequant kernels, the K = V elasticity is 0.17 to 0.24 in three kernel families
  (E-51, E-52).
- FACT (exploratory): the stock FA4 FP8 kernel reads the compact rows through views, at the same error and with no
  loss of speed (E-52).
- FACT (E-54, preregistered PASS; R-76 CONFIRM_WITH_CORRECTIONS): on 7 fresh prompts of technical text and code (4
  plain of 8K, 2 chat of 8K, 1 plain of 32K), the scale-tied compact FP8 row with frozen per-layer scales has 0.67
  times the attention error of the stock FP8 cache and 1.13 times the error of the best jointly calibrated standard
  FP8 cache (0.98 to 1.42 by layer). It beats the stock cache in every global layer by the median over prompts, and in
  66 of 70 prompt-layer cells. The stock FA4 kernel reads it in decode calls with separate K and V descales (gap
  0.0017); the prefill form is not tested. POST-HOC (R-76): the gain needs the scales to 0.1% to 0.25%; a 1% offset
  costs 19% to 24% more error, one phase step (2.2% in scale) removes the gain, and BF16 storage of the scales costs
  about 1%. Part of the gain is the BOS vector, which is the same in every prompt.
- INTERPRETATION: on the H100, the value of I-182 is capacity, not speed.

Not supported, or not tested:
- FACT: no engine run of the compact cache and no throughput number exist. The capacity gain is arithmetic.
- FACT (E-53): on real activations (4 texts of 8K tokens), the compact FP8 row at KV scale 1.0 fails the accuracy
  gate: up to 2.38 times the standard FP8 error (layer 41). The compact row in BF16 has an error of 0 to 0.0105
  against the BF16 cache (exploratory). On fresh prompts (E-54) the row at scale 1.0 fails layer 41 again (2.39):
  in the six 8K prompts (2.2 to 2.6), not in the 32K prompt (1.08; R-76 C5).
- FACT: no model-level quality test exists. E-53 and E-54 measure the attention error of each layer alone (E-55 is
  the planned model-level test).
- FACT: each experiment used one H100 in one session. The image has no `ncu` and no `nvdisasm`, so no DRAM counter
  and no SASS exist.
- FACT: we did not test the FP8 MMA kernel (FA PR 199), the vLLM Triton backend, other GPUs, mixed lengths, prefix
  caching or speculative decoding.
- INTERPRETATION (R-72): the speed result is specific to the H100 class. FACT (E-52 README Section 12): on the
  DRAM-bound RTX 4090, every compact variant gave sigma_layout 0.35 to 0.36.
- FACT (R-70): for Llama-3.1-8B with FA3 on an H100, the vLLM FP8 KV blog gives an FP8 decode slope of 54% of BF16.
  INTERPRETATION: the co-limit belongs to these head-dim-512 dequant kernels, not to all FP8 decode attention.
- HYPOTHESIS (R-72): with a BF16 KV cache, the views would give about 8% of the step. No run measured it.

## 6. Method contributions

### 6.1 Preregistration and two different-model reviews (T-03)

FACT (the four READMEs): each experiment had a frozen rule, a pre-run review by a different model and amendments
before any data. Then came a rehearsal, the user's spend approval, the run and a post-run review with an independent
recompute.

| Experiment | Pre-run review stopped | Post-run review corrected |
|---|---|---|
| E-49 | R-64: a false INVALID from profiler wall time; preempted cells | R-65: the cause of the lost 26B cells (row 69) |
| E-50 | R-67: an untestable pool rule; a short timeout; a weak path check | R-68: the S premise; per-step claims with no batch (row 70) |
| E-51 | R-69: dt over two configurations; a false KILL from a slow compact kernel | R-70: the oracle bounds the byte route only (row 71) |
| E-52 | R-71: a false KILL if the views do not share their reads; V4 on the stock row | R-72: the co-limit, H4, H5 and the SASS cause (row 72) |

FACT (journal Section 8.1 rows 69 to 72): each post-run review corrected at least one recorded claim. INTERPRETATION:
without the reviews, this chapter would give a wrong cause for a lost cell and an "on-chip" bound with no support.

### 6.2 Tools

FACT (RESEARCH_OUTPUTS Section 3; the E-51 and E-52 READMEs, Sections 12 and 13; R-72):

| Tool | What it does | Result in this work |
|---|---|---|
| K = V oracle with a BF16 control (T-10) | One paged tensor as K and V: half the DRAM bytes, the same on-chip work | e 0.885 and 0.991 on the RTX 4090; BF16 control 0.856 and 0.817 on the H100 |
| Pool probe (T-11) | Block tables into an 8 MB page pool, so all bytes come from L2 | DRAM-free floor 80% to 83% of dt |
| sm_90 compile check (T-11) | Each Triton configuration compiled for sm_90 in its own process, with PTX counts | 124 of 124 entries; the same PTX for both layouts in 54 of 54 |
| Views technique (T-11) | A packed row read as the K and V of an unchanged kernel | Stock kernel: 96% to 114% of its bytes-only gain; Triton: 52% to 73% |

INTERPRETATION: the views technique changes the compact cache from a kernel project into a layout and indexing
project.

### 6.3 Lessons (L-10 to L-19)

FACT (RESEARCH_OUTPUTS Section 4):

| ID | Lesson |
|---|---|
| L-10 | Count open engine PRs per code path; read the model report and the engine code |
| L-11 | Set `max_tokens` from the prefill schedule; log each finish reason |
| L-12 | Take wall time from unprofiled blocks (profiled steps were 1.6 to 2.4 times longer) |
| L-13 | Count the image pull (11 min on a cold host) in the cost cap |
| L-14 | Measure how time follows bytes before you call a byte ceiling a speedup |
| L-15 | Select one configuration per family, time it in 5 repeats, check each swept one for correctness |
| L-16 | A short row can sit in L2; bandwidth controls need rows much larger than L2 |
| L-17 | Count the power cap (0x4), log the SM clock, tie a microbenchmark to the engine on dt |
| L-18 | A DRAM-bound rehearsal GPU hides on-chip defects; add an L2-resident gate |
| L-19 | Under the cap, a short row after a heavy row runs slower; bound dt across repeats |

## 7. Next step: the gated engine prototype (D-88)

FACT (journal Section 10, D-88): the user chose (a), this write-up now, and (b), a vLLM prototype as a gated build.
The user did not choose (c), the FP8 MMA kernel test. G3 needs its own spend approval.

| Weeks | Work | Gate or stop rule (R-72) |
|---|---|---|
| 1 | This chapter; the gates G1, G2 and G3 | Stop (b) if G1 needs a scheduler change outside an isolated patch, or if G3 shows a loss above the FP8 KV noise |
| 2 to 5 | Behind a flag: the compact cache write, the views read, the query preparation, the o_proj fold | BF16 and FP8 KV arms |
| 6 | End-to-end capacity and throughput near the KV pool limit | Hard stop at the end of week 6, in any state |
| 7 to 9 | Thesis assembly | Kept free |

G1, the page-size plan (FACT, R-72; the main session checked the vLLM padding branch at commit e070573):
- The compact FP8 page is 163,840 B for 64 tokens. The sliding page is 524,288 B, and the ratio of 3.2 is not an
  integer.
- The vLLM page unifier pads such a page to 524,288 B: 8,192 B per token, twice the 4,096 B of the standard FP8
  global page. INTERPRETATION: with this padding, the compact cache uses more memory, not less.
- INTERPRETATION (R-72): a working plan makes the global page the largest. It holds 1,024 tokens (2,621,440 B, or 5
  sliding pages), with a sliding block of 320 tokens. The scheduler block then becomes 5,120 tokens.
- HYPOTHESIS (R-72): prefix-cache hits then land on 5,120-token units, which hurts prefix caching. G1 checks this
  first.
- FACT (R-65): vLLM `AttentionSpec` has a `head_size_v` field that a prototype can use.

G2 and G3 (FACT, D-88 and R-72):
- G2 repeats the upstream search, and reads the o_proj weight format and the RoPE layout of the checkpoint.
- G3 extracts real K and V of the global layers in one short Pod session (about $2). An offline test then compares the
  compact path with the standard FP8 path, with the FP16 range.
- FACT (E-53, billed $1.01): G3 ran on 2026-11-01 and gave FAIL for the row at KV scale 1.0. D-90 is open: the user
  asked for E-54, the scale-tied row with frozen calibrated scales (R-74 recommendation). R-75 (pre-run review) made
  its calibrated baseline stronger, and the user chose that baseline (D-91, 2026-11-02). E-54 passed (2026-11-02),
  and R-76 confirmed it with corrections. D-94 (open) decides the next step: E-55 (the model-level test) or the
  write-up.

HYPOTHESIS (R-72): (b) needs about $20 to $40 of GPU time. It gives an end-to-end capacity result by week 6 with a
probability of 40% to 55%. FACT (R-72): if a gate fails, the fallback is this chapter plus free items.

## 8. Conclusion

FACT (R-64, R-71; E-52 README Section 13): the compact K=V cache is exact in real arithmetic. The stock FA4 FP8
kernel reads it through two views. INTERPRETATION: on the H100 it is a memory mechanism, not a speed mechanism. The
gated D-88 build tests the open parts: the engine path, the page size and the accuracy on real activations. E-53
showed that an FP8 compact row needs tied, calibrated KV scales (R-74). E-54 tested them on fresh prompts and passed,
and R-76 confirmed the result with corrections. At the attention level, the tied row has 0.67 times the error of the
stock FP8 cache and 1.13 times the error of the best calibrated standard FP8 cache, in 640 instead of 1,024 bytes per
token and KV head. The scales must be reproduced to 0.1% to 0.25%. The model-level effect is not tested (E-55, D-94).

## Appendix A. Artifact map

| Experiment | Package | Results (local) | Reviews | Cost |
|---|---|---|---|---|
| E-49 | `experiments/gemma4_kv_audit/` | `results/e49-h100-20261030/` | R-64, R-65 | $2.09 H100 + $0.21 rehearsal (billed) |
| E-50 | `experiments/gemma4_kv_headroom/` | `results/e50-h100-20261031/` | R-67, R-68 | $2.92 + $0.24 (billed) |
| E-51 | `experiments/gemma4_kv_kernel/` | `results/e51-h100-20261101/` | R-69, R-70 | $1.42 + $0.14 (billed) |
| E-52 | `experiments/gemma4_kv_views/` | `results/e52-h100-20261101/` | R-71, R-72 | About $1.25 + about $0.31 (estimates) |

FACT (the sum of the rows): the four experiments cost about $8.60, and the E-52 billing is pending. Other records:
`dossier/REVIEW_E-49_*.md` to `dossier/REVIEW_E-52_*.md`, `experiments/runpod/HARDWARE_LOG.md`, `RESEARCH_OUTPUTS.md`
and journal Sections 7.30, 8.1 (rows 69 to 72) and 10.
