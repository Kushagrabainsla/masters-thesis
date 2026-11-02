# E-54: accuracy of the scale-tied compact FP8 K=V cache on fresh Gemma 4 31B activations (preregistration)

Status: PREDECLARED 2026-11-01. NOT RUN. S3: 13 tests pass (Section 12). Design review (S2): R-75 pending.
Idea: I-182. Decision: D-90 option (d), recommended by R-74 (`dossier/REVIEW_E-53_POSTRUN_2026-11-01.md`); the user
asked for this plan on 2026-11-01 ("Wait for R-74, and then write the E-54?"). D-90 stays open for the spend and the
build path. Journal entry: Section 6.1 E-54.
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
POST-HOC marks a number that was computed after the E-53 decision. All thresholds are project heuristics unless a
source is given.

## 1. Question
On fresh real Gemma 4 31B activations, does the scale-tied compact FP8 row (640 values per token and KV head; per-layer
scales frozen from the E-53 captures) give an attention-output error no larger than (a) the stock FP8 cache of vLLM
(1,024 values, KV scale 1.0) and (b) a standard FP8 cache with per-layer calibrated scales?
"No" is the FAIL of Section 7: a pooled error ratio above 1.5 against either baseline. A FAIL stops the FP8 compact
design (D-90 then falls back to the BF16 compact build or the write-up).

## 2. Rubber duck
- What E-53 found (FACT, E-53 README Section 13 and R-74): at KV scale 1.0, the compact row [K_rot | V_unrot | V_rot]
  has 0.57 to 2.38 times the attention error of the standard FP8 cache per global layer; layer 41 fails (2.38). The
  compact row rounds K_unrot on the FP8 grid of V. The k_norm weight w is one BF16 value per layer (0.0601 to 0.0654),
  so 16 w is 0.961 to 1.047, and the K grid of the compact row sits up to 0.066 octave from the stock K grid.
- What R-74 added (FACT, POST-HOC): the standard FP8 error is itself a lottery over the grid phase of the scale. In
  layer 41 one 1/16-octave step of the K scale changes it 2.44 times, and the stock scale is the second best of 16
  phases. 4 unrotated dims decide layer 41, and the BOS sink token decides the ratios below 1. For an equivalent storage
  at a random phase, the E-53 rule gives FAIL in 25% of draws, but the pooled RMS ratio over the 10 layers has a null
  median of 1.02 and a 95th percentile of 1.19 (the E5M2 control gives 3.34). With a per-layer tied scale chosen on 3
  prompts, the compact row has 0.72 times the stock error on the 4th prompt (pooled), and 1.06 times the error of a
  calibrated standard cache.
- The mechanism (INTERPRETATION, algebra): store the row as the codes of [K_rot / ks | V_unrot / vs | V_rot / vs] with
  ks = w x vs. The stock FA4 kernel reads codes 0 to 511 as K with k_descale ks and codes 128 to 639 as V with
  v_descale vs. Then ks x code(V_unrot / vs) = w vs x code(V_unrot / vs), which is w x V_unrot = K_unrot up to the FP8
  rounding of V. The query is the plain permuted q (no w in it). So the tied row and a standard cache with the same
  scales (ks, vs) round the same values, except where the BF16 values of K and w x V differ (R-74 C1: 1.6% to 4.7% of
  the unrotated codes flip). FACT (R-74 C5, checked by the author): the pinned FA4 SM90 FP8 kernel takes separate K
  and V descales per (sequence, KV head) and computes in FP16.
- What is new against E-53: calibrated scales (frozen before any E-54 data), two decision baselines (the stock cache
  and a calibrated standard cache), a pooled statistic with a known null, fresh prompts with chat templates and one
  32K prompt (R-74 R2), a tighter V5 (R-74 C8), and a kernel arm on the pinned FA4 kernel (R-74 M5).
- What kills it early: V7 (the tied row against the standard cache with the same scales) outside 0.90 to 1.10 means a
  code error. A FAIL means that the calibration does not transfer to fresh prompts or that the tie costs too much.
- Why this order: the calibration costs nothing (the Mac, the E-53 captures) and is frozen in the design commit; the
  only spend is one capture of fresh prompts on an H100, with the kernel arm on the same Pod.
- Expected (HYPOTHESIS): R1 0.70 to 0.95 and R2 1.05 to 1.25 (R-74 leave-one-prompt-out with a K-only calibrated
  standard: 0.72 and 1.06; the in-sample values of Section 4.6, 0.66 and 1.11, are optimistic); the chat and long
  prompts inside the guards; the kernel arm MATCH. R2 can land above 1.25 (INCONCLUSIVE): the calibrated standard has
  two free scales and the tied row one.
- Not in E-54: the model-level splice test (R-74 M6). It needs a hook that changes the forward pass, a bound with no
  calibration yet and its own rehearsal; it would double the review surface of a run whose job is the storage
  decision. It is E-55, after a PASS (Section 7).

## 3. Hypotheses
- H1 (primary): PASS (Section 7): R1 and R2 at most 1.25 in float32 and in the FP16 emulation, no layer above 2 and no
  test prompt above 1.5.
- H0: FAIL: R1 or R2 above 1.5 (float32).

## 4. Setup (frozen)

### 4.1 Hardware, software and model
- 1 x H100 80GB HBM3 (SXM), Secure Cloud, the E-53 image (vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130,
  transformers 5.20.2), CUDA floor 12.8. Why the H100: the pinned FA4 FP8 kernel of the kernel arm needs SM90, and the
  31B checkpoint needs 80 GB (E-53).
- Model: `RedHatAI/gemma-4-31B-it-FP8-dynamic` at revision `d4ab4f579dd3516f97d8a6a4c98d0653480bad15` (E-50 to E-53).
- FACT (R-74 C5, checked by the author with `gh api`): vLLM 3966924d5 pins flash-attention
  `b273a02621d3c94bc747df67204f591b67f38fae` (`cmake/external_projects/vllm_flash_attn.cmake` line 42).

### 4.2 Engine (`capture_e54.py`, `ENGINE`)
The E-53 engine arguments (eager, BF16 cache "auto", 8,448 batched tokens per step, 4 sequences, utilization 0.85,
no prefix caching, text only) with `max_model_len` 33,024 (32,768 + 256). The 8K prompts run as one prefill step each;
the 32K prompt runs as 4 chunks (vLLM V1 chunks every prefill that is longer than the step).

### 4.3 How the hooks reach the model
As E-53 (E-53 README Section 4.3; `capture_e53.register`, `Store`, `check_prompt`, `save_prompt`): forward pre-hooks
on the Attention modules of the 10 global layers get q and k after RoPE and v after its norm; a forward hook gets the
engine output o; the positions come from the parent module. The Store records each prefill chunk; the run record
counts the chunks of each prompt (`chunks`). The test of Section 12 runs 4 chunks.

### 4.4 Prompts (`capture_e54.py`, `build_prompts`; the plan is `analyze_e54.PLAN`)

| Id | Kind | Files | Tokens |
|---|---|---|---|
| 0 | plain | vLLM Python files | 8,192 |
| 1 | plain | torch Python files | 8,192 |
| 2 | plain | Markdown and reStructuredText files | 8,192 |
| 3 | plain | all of them (mixed) | 8,192 |
| 4 | chat | mixed | 8,192 |
| 5 | chat | mixed | 8,192 |
| 6 | plain | mixed | 32,768 |

- Seed 54; the file order of prompt p is `random.Random(54 * 1000 + p)` over its family, then the rest of the mixed
  family (as E-53). The query rows are the last 256 positions of each prompt.
- A plain prompt is built as in E-53: BOS, then the tokens of each file (at most 3,000), each followed by a blank line,
  cut to the planned length. BOS is its only special id.
- A chat prompt alternates a user turn ("Here is a file. Explain what it does and how it works." and at most 6,000
  characters of a file) and an assistant turn (at most 6,000 characters of the next file). The tokenizer's chat template
  renders the turns, the text is encoded (the template strings become special ids), and the ids are cut to 8,192. BOS
  stays at position 0. The turn markers are the extra sinks that R-74 R2 asks for.
- Every prompt skips a file that E-53 used (`e53_files.json`: the 17 files of the 4 E-53 prompts, by path and by
  SHA-256), a file that an earlier E-54 prompt used, and a file whose own ids hold a special id (E-53 A1, F4).

### 4.5 Analysis (`analyze_e54.py`; on the Pod GPU after the capture; float32 with TF32 off)
Per test prompt p and global layer l, the causal GQA attention of the 256 query rows (scale 1.0, no soft-cap) for the
BF16 reference and for each arm:

| Arm | Storage | Role |
|---|---|---|
| STOCK | standard FP8 cache, ks = vs = 1.0 | decision baseline (a) |
| CALSTD | standard FP8 cache, the frozen calibrated ks and vs of the layer | decision baseline (b) |
| TIED | the tied row: codes of [K_rot / ks \| V_unrot / vs \| V_rot / vs], the frozen vs of the layer, ks = w x vs; plain permuted query; descales | the candidate |
| TIEDSTD | standard FP8 cache with the TIED scales | algebra control (V7) |
| TIED_E5M2 | the TIED row with E5M2 codes | sensitivity control (V6) |
| C0 | the E-53 compact row at scale 1.0 with the BF16 views query | the E-53 replication (reported) |
| BF16C | the compact row in BF16 with the BF16 views query | D-90 option (b) (reported) |

The cast is the vLLM cache write (E-53 README Section 4.5): divide by the scale in float32, saturate at 448, round to
nearest even. Every scale is a float32 value. STOCK, CALSTD and TIED also run in the FP16 emulation of the pinned FA4
kernel (q and P in FP16; the FP8 values are exact in FP16). The analysis also reports the score path alone (V exact)
of TIED and CALSTD, the saturated values of each FP8 arm and the largest |q|.

### 4.6 Calibration (`calibrate_e54.py`; run on the Mac before any E-54 data; frozen in `scales_e54.json`)
- Data: the 4 E-53 captures (the SHA-256 of each file is checked against the E-53 run record).
- Rule (frozen; project heuristics): per layer, the base scales b_V and b_K are the smallest powers of 2 at or above
  2 x max|V| / 448 and 2 x max|K| / 448 (one octave or more of headroom before the saturation). The grid has 32 phases
  per octave, s_j = b x 2^(j / 32). A power of 2 does not move a normal value on the FP8 grid (R-74 C9), so phase 0
  is the phase of the stock scale 1.0: the CALSTD search contains the stock storage (without its K subnormals), and the
  TIED search contains the E-53 row. The objective is the sum over the 4 prompts of the relative error of the attention
  output; the argmin wins, and a tie goes to the lowest phase. TIED: one search over vs (the tied row with ks = w x
  vs). CALSTD: ks from the score-path error (K cast, V exact) and vs from the value-path error (K exact, V cast).
- Design history (before any E-54 data, on the E-53 captures only): the first rule anchored the grid at 2 x max / 448
  and smoothed the curve over the neighbor phases. On layer 41 with 8 phases it chose scales that are worse than the
  stock scale (in sample: CALSTD 0.0444 against STOCK 0.0282 in prompt 0; TIED 1.34 times STOCK), because the stock
  phase was not on the grid and the smoothing preferred a broad minimum. R-74's leave-one-prompt-out data show that the
  sharp minima transfer between prompts (layer 41: 1.11 to 1.13 on each left-out prompt, the oracle 1.11), because one
  sink vector and a few channels that all prompts share make the curve. So the frozen rule anchors the grid at a power
  of 2 and does not smooth.
- Result (FACT, `scales_e54.json`, SHA-256 `202738b9fe8b18cd594aca27b744984ecdcf6c4cfa3044660446e33a9914fdb1`; the Mac
  CPU, 743 s; the 40 input files match the E-53 record). In every layer b_V = 0.125 and b_K = 2^-7. The errors are
  the medians over the 4 calibration prompts (in sample, so optimistic; not a test):

| Layer | w | TIED phase | vs | ks = w vs | CALSTD phases K/V | STOCK | CALSTD | TIED | TIED/STOCK | TIED/CALSTD |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 0.06152 | 3 | 0.13339 | 0.008207 | 2/21 | 0.0440 | 0.0224 | 0.0237 | 0.542 | 1.066 |
| 11 | 0.06030 | 25 | 0.21483 | 0.012955 | 3/5 | 0.0264 | 0.0213 | 0.0201 | 0.757 | 0.967 |
| 17 | 0.06055 | 15 | 0.17299 | 0.010474 | 13/21 | 0.0391 | 0.0296 | 0.0343 | 0.896 | 1.172 |
| 23 | 0.06445 | 21 | 0.19700 | 0.012697 | 26/18 | 0.0736 | 0.0264 | 0.0254 | 0.345 | 0.960 |
| 29 | 0.06250 | 21 | 0.19700 | 0.012312 | 21/21 | 0.0345 | 0.0245 | 0.0245 | 0.709 | 1.000 |
| 35 | 0.06543 | 1 | 0.12774 | 0.008358 | 3/13 | 0.0288 | 0.0245 | 0.0257 | 0.902 | 1.062 |
| 41 | 0.06494 | 15 | 0.17299 | 0.011234 | 15/15 | 0.0259 | 0.0230 | 0.0266 | 0.998 | 1.143 |
| 47 | 0.06348 | 16 | 0.17678 | 0.011221 | 5/16 | 0.0693 | 0.0391 | 0.0464 | 0.686 | 1.287 |
| 53 | 0.06299 | 15 | 0.17299 | 0.010896 | 19/19 | 0.0519 | 0.0310 | 0.0394 | 0.759 | 1.277 |
| 59 | 0.06006 | 24 | 0.21022 | 0.012626 | 30/9 | 0.0276 | 0.0203 | 0.0203 | 0.737 | 1.000 |

- In sample (FACT, the statistics of Section 5 on the calibration prompts): P_STOCK 0.0453, P_CALSTD 0.0268, P_TIED
  0.0298; R1 0.658, R2 1.113; V6 (E5M2) 3.03; V7 (tie) 0.971. Layer 41, the E-53 FAIL layer, goes from 2.38 times the
  stock error (scale 1.0) to 1.00 times. In layer 29 (w = 2^-4) the tied and the calibrated standard scales coincide
  (phases 21/21/21), as the algebra says. INTERPRETATION: the tie costs about 11% of error against the best per-tensor
  standard scales in sample (one free scale against two), and the calibrated standard is a strong baseline (0.59 times
  the stock error). The out-of-sample values decide.

### 4.7 Kernel arm (`kernel_e54.py`; secondary; on the Pod after the analysis)
For prompts 0 (8K) and 6 (32K) and each global layer: three paged FP8 caches from the captured K and V (STOCK, CALSTD
and TIED, with the scales of the analysis), the engine tensor [blocks, KV heads, 64, values] read through
.transpose(1, 2) (as E-52), and the FA4 call of vLLM FlashAttentionImpl.forward with `fa_version=4`, `k_descale` and
`v_descale` of shape [sequences, KV heads]. Each of the 256 query rows is one decode sequence: every sequence shares
the pages, and sequence i sees the keys 0 to qpos[i]. For TIED the K view is codes 0 to 511 and the V view is codes
128 to 639 of the 640-value rows, and the query is the plain permuted q. The gap of a cell is the relative difference
between the kernel output and the FP16 emulation of the same arm.

## 5. Metrics
- err_X(p, l): ||o_X - ref|| / ||ref|| for arm X (float32; err16 for the FP16 emulation).
- P_X: the RMS over the 10 layers of the median over the 7 prompts of err_X(p, l). This is the pooled statistic of
  R-74 M2 (null median 1.02, 95th percentile 1.19 and 99th percentile 1.26 for an equivalent storage at a random phase,
  on the E-53 data; E5M2 3.34).
- Primary: R1 = P_TIED / P_STOCK and R2 = P_TIED / P_CALSTD (lower is better), with R1_fp16 and R2_fp16.
- Guards: g1(l) and g2(l), the median over the prompts of err_TIED / err_STOCK and err_TIED / err_CALSTD per layer;
  R1(p) and R2(p), the per-prompt ratios of the RMS over the layers.
- Minimum effect that matters: a pooled ratio of 1.25 (R-74: the 99th percentile of the null is 1.26). The repeats are
  the 7 prompts and the 10 layers; the computation is deterministic, so no A/A run is needed.
- Reported only: the ratios per prompt class (plain 8K, chat, long), P_CALSTD / P_STOCK (what calibration gives the
  standard cache), C0 against STOCK per layer (does the E-53 FAIL repeat on fresh prompts), the BF16C errors, the
  score-path errors, the saturated values, the largest |q| and the kernel arm gaps.

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU is an H100 80GB.
- V2: the capture finished with the frozen plan (7 prompts, their kinds and lengths, 256 query rows); the hooked layers
  are the 10 global layers; every prompt and layer has metrics with the planned shapes, BF16 and positions; no capture
  prompt had a problem; every capture file has its recorded SHA-256; the engine is the E-53 one (vLLM version, eager,
  BF16 cache); no test prompt takes an E-53 file or a file of another test prompt; a plain prompt has no special id
  after position 0; a chat prompt has BOS at position 0 and 2 or more special ids after it; the analysis used the
  frozen scales file (its SHA-256).
- V3: the identity K_unrot = w x V_unrot holds within 1e-2 in every cell (as E-53), and the module w of every global
  layer is one BF16 value on the unrotated dims and equals the w of the scales file.
- V4: the reference is finite.
- V5 (R-74 C8): the reference matches the engine output: ||ref - o|| / ||o|| <= 0.005 in every cell (E-53: 0.0017).
- V6 (sensitivity): P_TIED_E5M2 / P_TIED >= 1.5 (one mantissa bit less; the in-sample value is in Section 4.6).
- V7 (algebra control): 0.90 <= P_TIED / P_TIEDSTD <= 1.10 (R-74 C1: the per-layer medians are 0.996 to 1.010).

## 7. Decision rule (frozen; project heuristics)
- INVALID: V1 to V7 (the result names them).
- FAIL: R1 > 1.5 or R2 > 1.5 (float32).
- PASS: all of: R1, R2, R1_fp16 and R2_fp16 at most 1.25; every layer has g1 and g2 at most 2.0 (R-74 M2); every test
  prompt has R1(p) and R2(p) at most 1.5 (coverage of the chat and long prompts); no FP16 value that is not finite.
- INCONCLUSIVE: every other result, with each failed condition named.
- FACT (algebra): PASS and FAIL cannot both hold.
- INTERPRETATION: the pooled bound 1.25 passes an equivalent storage at a random phase in about 99% of draws (R-74), and
  the FAIL bound 1.5 is far below the 3.34 of a lost mantissa bit. A per-layer limit of 1.25 is not used: the null
  passes it in only 8% of draws (R-74 M2).
- Consequences (the user confirms each step; written before any data):

| Outcome | When | Action |
|---|---|---|
| PASS | every PASS condition holds | the FP8 compact build continues with the tied scales (D-88 week 2); E-55 (the model-level splice) next |
| FAIL | R1 or R2 above 1.5 | stop the FP8 compact design; D-90 falls back to (b) the BF16 compact build or (a) the write-up |
| INCONCLUSIVE | any other failed PASS condition | the user decides; the report names the conditions; no rerun without a new S4 |
| INVALID (V5 only) | the reference does not model the engine | fix the reference and analyze the copied captures again on the Mac; no new Pod |
| INVALID (other) | V1 to V4, V6, V7, or several checks | fix the cause; a rerun needs a new S4 |

- Kernel arm (secondary; it does not change the outcome above):

| Outcome | When | Meaning |
|---|---|---|
| MATCH | every gap at most 0.005 | the stock FA4 kernel reads the tied row with k_descale not equal to v_descale: no kernel change |
| CLOSE | every gap at most 0.05 | report; the user decides if the gap matters |
| MISMATCH | a gap above 0.05 | the tied row needs a kernel change or another descale path; the user decides |
| NOT RUN or ERROR | no SM90 GPU, no FA4 module, or an exception | report the reason |

## 8. Controls and baselines
- Baselines: STOCK (what vLLM does with this checkpoint) and CALSTD (the best standard per-tensor scales by the same
  calibration data and grid). The standard cache has two free scales; the tied row has one (the price of the tie).
- Algebra control: TIEDSTD, the standard cache with the TIED scales (V7).
- Sensitivity control: TIED_E5M2 (V6).
- Replication: C0, the E-53 row at scale 1.0 (reported). HYPOTHESIS: layer 41 is above 2 again.
- A/A: not needed (no timing; deterministic).
- Negative controls (CPU tests, Section 12): a wrong head mapping, no causal mask, the wrong scale and hidden current
  keys fail V5 (R-74 C8); swapped descales give a kernel MISMATCH; an E-53 file, a chat prompt without markers and a
  plain prompt with a special id fail V2; a changed w fails V3; a broken E5M2 control fails V6; a wrong tied row fails
  V7.

## 9. Budget
- Price: 1 x H100 SXM Secure at $3.99/h (the E-53 live read on 2026-11-01; S4 reads it again). Cost cap: $4.
- Time model (INTERPRETATION, from the E-53 record):

| Step | Estimate | Source |
|---|---|---|
| Create to SSH | 3.5 min (image on the host) to 11 min (pull); delete the Pod at 15 min | E-53 213 s (EUR-NO-2) |
| Setup and the 33.3 GB download | about 5 min (cap 420 s) | E-53 280 s |
| Engine start (eager) | about 1.5 to 2 min | E-53 93.5 s |
| 7 prefills with saves (7.7 GB) | about 1 to 2 min (with the engine: cap 720 s) | E-53: 4 prefills and 3.1 GB in about 15 s |
| Analysis | about 1 to 3 min (cap 360 s) | E-53 14 s for 40 cells and 11 passes; now 70 cells (one at 32K) and 16 passes |
| Kernel arm | about 1 to 3 min (cap 300 s) | the FA4 JIT of E-52, 60 decode calls |
| Copy of RES and CAPDIR (7.7 GB) | about 5 to 7 min inside the AUTO_STOP window (450 s) | E-53: 3.1 GB in 129 s with 8 streams |
| Pod time | about 28 to 36 min, $1.85 to $2.40 | |

- Caps: download 420 s, capture 720 s, analysis 360 s, kernel 300 s; with the setup and the GPU wait they add up to
  1,920 s, the session watchdog `POD_LIMIT`. The session starts about 140 s after the boot watchdog arms, so the session
  watchdog fires at about 2,060 s after boot, before the boot watchdog of 2,400 s (`pod_cmd.py --cap 2400`).
  `AUTO_STOP=1` with a 450 s copy window.
- Worst case: the main session deletes the Pod if it has no SSH port 15 min after the create (inside the wait loop);
  then 15 + 40 = 55 min, $3.66, plus the volume disk. S4 runs `experiments/runpod/balance.py --need 4 --hours 1` live.
- Order of work: prompt 0 with the fail fast, the other prompts, the analysis, the kernel arm. A step that times out
  leaves V2 failing (INVALID); no partial decision is made.
- Copy: in the AUTO_STOP window the main session copies RES, then `k_norm.pt`, then the prompt folders in order with 8
  parallel per-file rsync streams (E-53 A3), and compares the checksums with `capture_sha.txt`. If the window ends
  first, start the Pod again with 0 GPUs (RUNPOD.md Section 6) and finish the copy. The folder `.gitignore` keeps
  `results/*/capture/` out of git.
- S5 rehearsal (needs the user's approval at S4, or a logged waiver): 1 x RTX 4090 Secure ($0.89/h, the E-53 read;
  S4 reads it again), `google/gemma-4-E2B-it` at revision `3e22461f65e89153144f8adb70e3b8c2cc9845a7` (as E-53),
  `CAP_ARGS="--rehearsal --heads 8 --kv-heads 1 --global-layers 3"`, `POD_LIMIT=1200`, boot watchdog 1,500 s, cap
  $0.60 (worst case 40 min, $0.59; plan about 20 min, $0.30). It uses the scales of the run data (phase 0), not the
  frozen file. Expected: `REHEARSAL INVALID (V1, V3, ...)` (not an H100; E2B has no K=V) and the kernel arm NOT RUN
  (SM89).
  VERIFY items: (1) vLLM 0.33.0rc1.dev229+g3966924d5 and the E2B snapshot; (2) the engine starts with
  `max_model_len` 33,024 on 24 GB; (3) the chat template renders, and prompts 4 and 5 have BOS at 0 and 2 or more
  special ids after it; (4) no prompt takes an E-53 file; (5) prompt 6 has 32,768 tokens and its run record shows 4
  chunks in every layer, with 0 problems; (6) 21 capture files and the SHA-256 list; (7) the analysis runs within its
  cap and V5 is at most 0.005; (8) the kernel arm writes NOT RUN with the reason; (9) the copy of CAPDIR matches its
  checksums; (10) AUTO_STOP stops the Pod; (11) every step fits the H100 caps.
- Run steps (S6 and S7): as E-53 (E-53 README Section 9), with the name from `pods.py name E-54 session`, the env
  `THESIS_EXPERIMENT=E-54`, the upload of `gemma4_kv_scaletie`, `gemma4_kv_accuracy` (without `results/` and
  `exploratory/`), `gemma4_kv_kernel` and `runpod`, and `session_e54.sh` with `AUTO_STOP=1`.

## 10. Threats to validity
- Forking paths (R-74 R1): the tie idea and the calibration rule came from the E-53 captures. The test prompts share no
  file with them, and the scales are frozen in the design commit. The rule changed once before freezing (Section 4.6),
  on the calibration data only.
- Coverage (R-74 R2): 2 chat prompts and 1 long prompt are few. The per-prompt guard catches a prompt type that fails
  as a whole; a subtler class effect stays open (reported per class).
- Sinks and channels (R-74 R3): one sink vector and a few channels decide some layers. A new sink (a turn marker) can
  move the best phase. This is the transfer question that E-54 tests.
- BF16 mismatch (R-74 R4): the code flips act as noise of about 6% per cell with tails of +54%. The pooled statistic
  averages it; the per-layer guard (2.0) leaves room for it.
- Scope (R-74 R5): the tie needs one w value per layer. This holds for this checkpoint (V3 checks it).
- Calibration (R-74 R6): the tied row needs per-layer scales from a calibration set. "No calibration needed" is no
  longer a claim. The calibration set is 4 prompts of code and documents.
- Kernel drift (R-74 R7): the FA `main` branch computes the FP8 path with FP8 products and a fixed Q scale. E-54 tests
  the pinned kernel only; a vLLM upgrade needs a new check.
- Model quality (R-74 R8): per-layer attention errors do not give the model quality. E-55 measures it.
- Prefill queries: the 256 query rows are prefill rows, not decode steps. The attention of a query at position t over
  the keys 0 to t is the same computation in both; the kernel arm runs the decode kernel on them.
- Headroom: the scales give one octave or more of headroom over the calibration maxima. A test value above it
  saturates; the saturated counts are reported, and the errors include their effect.
- Engine: the capture runs with a BF16 cache, so each layer sees BF16 attention before it (as E-53).

## 11. Files
- `calibrate_e54.py`: the frozen calibration rule (Section 4.6); writes `scales_e54.json`.
- `scales_e54.json`: the frozen scales, the curves and the in-sample errors (FACT, Section 4.6).
- `e53_files.json`: the 17 files of the 4 E-53 prompts (the exclusion list of V2).
- `capture_e54.py`: the plan, the prompts, the engine and the capture (the hooks and checks of `capture_e53.py`).
- `analyze_e54.py`: the arms, the metrics (`--capture`), the checks V1 to V7 and the frozen rule (summary.json and the
  decision, with the kernel line); without `--capture` it decides again from metrics.json and kernel.json.
- `kernel_e54.py`: the kernel arm (kernel.json); `--standin` uses a CPU stand-in for the Mac tests.
- `session_e54.sh`: the Pod session (versions, hardware record, telemetry, download, capture, analysis, kernel arm,
  decision, checksums, watchdog, AUTO_STOP; `STUB=1` for the Mac).
- `test_e54.py`: the tests (Section 12).
- `.gitignore`: keeps `results/*/capture/` out of git.
- Reused, unchanged: `../gemma4_kv_accuracy/capture_e53.py` and `analyze_e53.py`, `../gemma4_kv_kernel/compact_attn.py`.
- Outputs in `RES` (`/workspace/results/e54`, small): `session.log`, `versions.txt`, `env-*/`, `prompts.json`,
  `capture_run.json`, `capture_log.txt`, `metrics.json`, `analysis_log.txt`, `kernel.json`, `kernel_log.txt`,
  `summary.json`, `decision.txt`, `capture_sha.txt`. In `CAPDIR` (`/workspace/capture_e54`; copied to the Mac as
  `results/e54-h100-<date>/capture/`): the capture files and `k_norm.pt`.

## 12. Verification record (stages S3 and S5)
- S3 (2026-11-01, Mac, no GPU and no vLLM): `uv run --no-project --with numpy --with torch --with pytest python -m
  pytest -q --rootdir <folder> <folder>/test_e54.py -p no:cacheprovider`: 13 passed in 42 s (torch 2.15.0). FACT from
  the tests:
  1. With w = 2^-4 the tied row equals the standard storage with the same scales (1e-5); with the w of layer 41 the
     errors stay within 0.8 to 1.25 of each other.
  2. The kernel form of the tied row (views, descales, the plain permuted query) equals an independent natural-order
     form (1e-5; 1e-3 in FP16).
  3. The calibration finds a planted phase (11 of 32) of the value path; the base is the smallest power of 2 at or
     above 2 x max / 448; a tie goes to the lowest phase; ks = w x vs in float32.
  4. (R-74 C8) A wrong head mapping, no causal mask, the scale 1/sqrt(512) and query positions that hide the current
     key fail V5; the BF16 engine output passes it.
  5. The prompts take no excluded file (by path or by SHA-256), no file twice and no file with a special string; the
     plain prompts have no special id after position 0; the chat prompt has BOS at 0, 2 or more turns and its markers;
     the lengths follow the plan; the same seed gives the same token hashes.
  6. The E-53 Store records a prefill in 4 chunks in position order.
  7. The kernel arm with the CPU stand-in gives MATCH (every gap below 0.005); swapped descales give MISMATCH (STOCK
     still matches, TIED does not); without a GPU the real path writes NOT RUN.
  8. The calibration script on the stub captures writes a scales file with the SHA-256 of its 14 inputs; the analysis
     of a run that is not a stub reads that file, records its SHA-256 and uses its scales.
  9. Every decision path: PASS (stub), FAIL by the pooled ratio, INCONCLUSIVE by the pooled ratio, by one prompt, by
     one layer, by the FP16 ratio and by FP16 values that are not finite; INVALID by V1, V2 (an E-53 file, a chat
     prompt without markers, a plain prompt with a special id), V3 (another w), V5 (with the action "no new Pod"), V6
     and V7.
  10. The session caps: the steps fit in the session watchdog, the session watchdog fires before the boot watchdog,
      the worst case is $3.66 (cap $4) and the rehearsal worst case $0.59 (cap $0.60).
  11. Timing at the real size (E-38): one prompt and layer with every arm (16 attention passes) takes 2.1 s on the Mac
      CPU.
  12. The stub session from a clean copy of the 4 upload folders: capture, analysis, kernel arm (stand-in) and the
      decision `STUB PASS` with the kernel line MATCH; 15 capture files in `capture_sha.txt`; a second run skips the
      capture; `--stub-fail` gives INVALID (V2); `--stub-no-tensors` stops the capture with exit 3 (fail fast).
  Not tested on the Mac: vLLM, the engine start with 33,024 tokens, the chat template of the real tokenizer, the hooks
  in a live engine over 4 chunks, the FA4 kernel and the GPU. The S5 rehearsal covers all but the FA4 kernel.

## Amendments
None yet.
