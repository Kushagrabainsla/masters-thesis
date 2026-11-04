# E-55: model-level accuracy of the compact K=V cache (BF16 and FP8) in the Gemma 4 31B global layers (preregistration)

Status: PREDECLARED 2026-11-02; amendment A1 2026-11-03 (R-77, before any E-55 data); amendment A2 2026-11-03
(after the S5 rehearsal, before any 31B data; D-96). RUN on the 31B 2026-11-04. S3: 27 tests pass in 541 s (Section 12).
S4: D-95 (the user). S5: rehearsal 2 passed the 10 VERIFY items and found two problems, both fixed in A2 (Section 9).
Design review (S2): R-77 AMEND_THEN_RUN (`dossier/REVIEW_E-55_PRERUN_2026-11-02.md`); every finding is fixed or
answered in the A1 list at the end. S6 to S8 (2026-11-04): INVALID (V6); the BF16 baseline is degraded (Section
13). Next: S9 (R-78, the post-run review).
Idea: I-182. Decision: D-94 (the user chose option (1) on 2026-11-02, "Yes, plan E-55": the model-level test with a
16-bit arm and an 8-bit arm). Journal entry: Section 6.1 E-55.
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
All thresholds are project heuristics unless a source is given. "(A1)" marks text that amendment A1 changed.

## 1. Question
At the model level, does the compact K=V cache of the 10 Gemma 4 31B global layers keep the model's next-token
predictions on long documents?
- H16 (primary, 16-bit): with the compact row in BF16, is the change of the per-token negative log-likelihood (NLL)
  inside a margin of 0.001 nats per token (0.1% of perplexity)? (A1: a fixed margin, R-77 F4; 0.1 x the E5M2 effect
  is reported.) "No": the change is outside the margin (PIVOT16 or KILL16, Section 7).
- H8 (secondary, 8-bit): with the frozen E-54 tied scales, is the compact FP8 row no worse than the stock FP8 cache?
  "No": the tied row is measurably worse (PIVOT8 or KILL8).

## 2. Rubber duck
- The claim. The compact row stores [V (512) | K on the 128 rotated dims] per token and KV head, 640 values instead of
  1,024, because K = w V on the 384 unrotated dims. It is exact in real arithmetic (float64 error 3e-15, E-49). In BF16
  it differs only by rounding: E-54 measured 0 to 1% of attention-output error per layer (pooled 0.0054). In FP8, the
  tied row (one scale, ks = w vs) has 0.67 times the stock error and 1.13 times the best calibrated standard error (E-54
  PASS; R-76 CONFIRM_WITH_CORRECTIONS).
- What is new. E-54 and R-76 cover single layers with BF16 attention before each layer. Rounding errors can add up over
  60 layers and 64K positions. No model-level test exists (R-76 C1, "what the PASS does not support"). The user wants
  the 16-bit claim for the paper and chose a test with both arms (D-94).
- What would kill it early. (1) The compact BF16 cache changes NLL by more than 0.01 nats per token: a fault, because
  the layout is exact. (2) The splice does not reach the attention reads (V5, V6, V7). (3) The tied FP8 row is worse
  than the stock FP8 cache at the model level.
- Why a splice and not the engine build (D-88, weeks). Hooks on the 10 global Attention modules of the real engine
  change q, k and v before the cache write. The BF16C splice gives the stock kernel the same q and K values that the
  two views of the compact row give it (E-52, E-54 BF16C), so its result is the result of a correct build, up to the
  summation order. The FP8 arms store the FP8 codes (exact in BF16) and apply the K descale to the scores (the impl
  `scale`) and the V descale to the output, as the FP8 kernel does. On the E-54 captures every arm gives the frozen
  E-54 error to 1e-4 (Section 12). A first form stored bf16(scale x code) and moved the TIED and JOINT errors by 2% to
  6%, because the rounding acts like a small scale change on each code value; it was replaced before any E-55 data.
- Not tested here: the FP8 kernel itself (P in FP16 on the pinned FA4; E-54 kernel arm for decode), the engine write
  path, the 640-value page layout, prefix caching, CUDA graphs, and a full FP8 cache (the sliding layers stay BF16).
- Why NLL. Teacher-forced NLL on long documents measures the whole next-token distribution at every position, needs
  no task data and sees small changes. A greedy-generation set gives a check that a reader can see (identical text).
- Why the frozen E-54 scales. They are the validated ones (E-54 PASS on fresh prompts). New scales (R-76 item 7) would
  need their own attention-level test first; that offline work stays optional and outside E-55.
- (A1) Resolution (R-77 finding 9). The checkpoint quantizes the input of each linear layer to FP8 per token, so a
  change of 0.3% in an attention output flips FP8 codes downstream: every arm is also a new realization of the
  model's quantization noise. BF16R (one more BF16 rounding of the query, no other change) measures that floor.
- Order of work: the warm-up and a one-chunk pre-flight of every arm (A1), the BF16 reference, the BF16C arm (H16),
  the positive control (validity), then STOCK8 and TIED8 (H8), then the reported steps while the time allows.
- Predictions (HYPOTHESIS, written before any E-55 data; checked after the run, not used by the rule):
  |Delta_BF16C| below 1e-4; Delta_STOCK8 0.0002 to 0.005; Delta_TIED8 / Delta_STOCK8 0.4 to 0.8 (E-54 R1 0.67; an NLL
  change that grows with the square of the error gives 0.45); Delta_TIED8 / Delta_JOINT8 1.0 to 1.5; Delta_E5M2 3 to 10
  times Delta_STOCK8; Delta_TIED8_UP above Delta_TIED8. (A1, replacing the BF16_B predictions, R-77 F7 and F9):
  |Delta_BF16R| below 2e-4 and |Delta_BF16C - Delta_BF16R| below 2e-4; the top-1 agreement of BF16R 95% or more (ties
  and near ties of the BF16 logits can flip the top-1 token); the BF16C top-1 agreement at most 0.5 points below that
  of BF16R; the per-token SD of the BF16C change 0.02 to 0.11 nats.

## 3. Hypotheses
- H16 (primary): the compact BF16 cache changes the mean per-token NLL by less than the margin (Section 7).
- H8 (secondary): the tied compact FP8 row has an NLL change no larger than the stock FP8 cache's, within the
  tolerance (Section 7).
- H0: the compact BF16 cache changes the NLL by more than the margin (a fault, or rounding that adds up over layers and
  positions), or the tied FP8 row is worse than the stock FP8 cache.

## 4. Setup (frozen)

### 4.1 Hardware, software and model (as E-54)
- 1 x H100 80GB HBM3 (SXM), Secure Cloud, the E-53 and E-54 image (vLLM 0.33.0rc1.dev229+g3966924d5, torch
  2.14.0+cu130, transformers 5.20.2), CUDA floor 13.0. Why the H100: the 31B checkpoint needs 80 GB, and the pinned
  stack and its numerics are the ones E-53 and E-54 measured.
- Model: `RedHatAI/gemma-4-31B-it-FP8-dynamic` at revision `d4ab4f579dd3516f97d8a6a4c98d0653480bad15` (E-50 to E-54).
  FACT (its `config.json`, read 2026-11-02): 60 layers, global layers 5, 11, ..., 59 (head size 512, 4 KV heads, 32
  query heads, `attention_k_eq_v` true, partial rotary 0.25), sliding window 1,024, vocabulary 262,144, maximum
  context 262,144, final logit soft-cap 30, no `kv_cache_scheme`.

### 4.2 Engine (`run_e55.ENGINE`)
- One engine: eager (`enforce_eager`), BF16 cache ("auto"), `max_model_len` 65,664, 4,096 batched tokens per step
  (chunked prefill), 8 sequences, utilization 0.70, no prefix caching (else an arm could read the KV of the arm before
  it), text only, seed 0, TF32 off. (A1: no second engine; BF16R replaces BF16_B, R-77 F7.)
- (A1, R-77 finding 3 and F6) FACT (vLLM 3966924d5; E-54 log line 24, "Using V2 Model Runner"; `config/vllm.py` 746 to
  783): the engine runs the V2 model runner. It computes the prompt log-probabilities in chunks of 1,024 rows
  (`v1/worker/gpu/sample/prompt_logprob.py`) with a float32 log-softmax of the BF16 logits; the engine side adds one
  empty position 0 and 2 entries per later position (`logprobs.py` 165 to 209, `sampling_params.py` 350 to 383): [the
  prompt token with its rank, the top-1 token]. One request per document (`max_tokens` 1, `detokenize` False).
  INTERPRETATION (R-77): about 1 to 3 GB per call outside the KV pool; utilization 0.70 leaves about 24 GB. The
  engine record holds `use_v2_model_runner` (reported, not a validity condition: both runners compute the same
  float32 log-softmax, and all arms share the runner).
- FACT (flash-attention b273a02, `flash_attn/cute/interface.py` 1421 to 1500 and 1895 to 1905): the FA4 compile key
  holds no softmax scale; the scale is a run-time argument, so a new scale per layer does not compile a new kernel.
  FACT (E-54 capture log): the engine uses the FLASH_ATTN backend (`FlashAttentionImpl`, `self.scale`, flash_attn.py
  1077; every call passes `softmax_scale=self.scale`, 1413 to 1698). R-77 confirmed the route by source (answer 2).

### 4.3 Data (`prepare_e55.py` on the Mac; `data_e55.json`, SHA-256 `8f2e2ecd96402b51...`)
Twelve scored documents in three domains, token ids of the pinned tokenizer (transformers 5.20.2, `tokenizer.json`
SHA-256 `cc8d3a0ce364...`), frozen before any E-55 run. Each document is BOS and then text; no source is used twice.

| Ids | Domain | Source | Tokens |
|---|---|---|---|
| B1 to B4 | books | PG-19 test books 42081, 6412, 40579, 10762 (deepmind/pg19 test list; the public bucket) | 65,536 each |
| P1 to P4 | prose | our own Markdown files in `dossier/` at commit fce8266 (7, 9, 9 and 11 files) | 32,768 each |
| C1 to C4 | code | our own Python files under `experiments/` at fce8266, not this package (11, 11, 14, 14 files) | 32,768 each |

- Seeds: 55 (books), 56 (prose), 57 (code). A book document is the first 65,535 tokens of the first books in the seed
  order with enough tokens. A prose or code document joins the first 6,000 tokens of each file in the seed order, each
  followed by a blank line, cut to 32,768. A source with a special id is skipped.
- Why these domains: the PG-19 books are a standard long-context set (and may be in the training data); the prose and
  code are our own text, written 2026-09 to 2026-11, after the Gemma 4 report (arXiv 2607.02770), so the model has
  not seen them. Results are reported by domain.
- Generation set: 16 chat prompts, 8 from 8 more books and 8 from 8 more dossier files: one user turn "Read the text
  below. Then summarize it in five sentences." with the first 3,000 tokens of the source, rendered by the chat template
  with the generation prompt (3,024 to 3,026 tokens; it ends with the model turn and the empty thought channel).
- `tok_check`: three strings and their Mac ids; the engine tokenizer must give the same ids (V2).
- Scored tokens: the tokens at index 1,025 or more (their context holds more than the 1,024-token sliding window, so
  only the global layers carry the long context): 64,511 per book, 31,743 per prose or code document, 511,988 in all.
- (A1, R-77 O2) Far-repeat subset (reported): the scored tokens whose 12-gram that ends at the token last occurred
  1,025 or more tokens before, so that the copy source is outside the sliding window. FACT (`far_repeat_mask` on the
  frozen data): 3,971 of the 511,988 scored tokens (0.78%).

### 4.4 Splice (`splice_e55.py`)
Forward pre-hooks on the Attention modules of the 10 global layers replace q, k and v before the module writes k and v
to the paged cache. The engine reads K and V from the paged cache in prefill and in decode (R-76, `flash_attn.py`;
R-77 answer 1), so every read sees the arm's values. The 50 sliding layers keep the BF16 cache in every arm. c() is
the vLLM cache cast (saturate, round to nearest even).

| Arm | q | Cache K | Cache V | Score factor (impl scale) | V descale (output) |
|---|---|---|---|---|---|
| BF16 | q | K | V | 1 | 1 |
| BF16C | [q_rot, bf16(q_unrot w)] | [K_rot, V_unrot] | V | 1 | 1 |
| E5M2 | q | c_E5M2(K) | c_E5M2(V) | 1 | 1 |
| STOCK8 | q | c(K) | c(V) | 1 | 1 |
| TIED8 | q | [c(K_rot / ks), c(V_unrot / vs)] | c(V / vs) | ks | vs |
| BF16R (A1) | bf16(q R_C), R_C = 1 + 2^-7 | K | V | f32(1 / R_C) | 1 |
| JOINT8 | q | c(K / ks_j) | c(V / vs_j) | ks_j | vs_j |
| TIED8_UP | q | as TIED8 with vs' = f32(vs 2^(1/32)), ks' = f32(w vs') | | ks' | vs' |

- w, ks, vs, ks_j, vs_j: the frozen `scales_e54.json` (SHA-256 `8efe6e90...`; ks = f32(w vs)). The hook reads w from
  the k_norm weight on the unrotated dims; it must be one value and equal the scales file (V3).
- The FP8 codes are exact in BF16, so the cache holds exactly the codes. The kernel multiplies the scores by the impl
  scale after the dot product, as the FP8 kernel multiplies by the K descale (R-74). The output hook multiplies the
  BF16 output by the V descale (one more BF16 rounding of the output; A1, R-77 answer 3: on the captures this moves
  the TIED8 error from 0.02312 to 0.02327 in layer 5; INTERPRETATION: below 1e-5 nats on D).
- (A1, R-77 F7) BF16R has a true effect of one BF16 rounding of the query (relative 2^-9 at most per value) and no
  other change, as BF16C has a rounding of q_unrot w. Its NLL change is the noise floor of a rounding of this size.

### 4.5 Order of work (`run_e55.py`, `session_e55.sh`)
1. Setup: versions, telemetry (with memory), the SHA-256 of the code that runs (L-25), the checkpoint download.
2. The engine; the tokenizer check; the hooks (V3); one warm-up prompt of 512 tokens (not scored). (A1, R-77 B1) Then
   the hook evidence starts again: the warm-up used the first check of each layer, so before A1 the BF16 arm recorded
   no V5 gap and every run would have ended INVALID.
3. (A1, R-77 F3) Pre-flight: every arm and DET on the first 4,096 tokens of B1 (one chunk, about 30 s). The run stops
   at once (exit 3, INVALID, about $0.7 spent) if a transform check fails, a V5 gap is above 0.01, a V7 check fails, a
   layer misses tokens, a value is not finite, DET is not bit-identical and differs by more than 0.0001 nats per token,
   E5M2 or STOCK8 gives the BF16 log-probabilities bit for bit, or BF16C does while DET is bit-identical. The evidence
   starts again after it, so the arms record their own V5 and V7 evidence.
4. Decision arms, 12 documents each: BF16; DET (P1 and C1 again with the hooks removed); BF16C; E5M2; STOCK8; TIED8.
   Then the record holds `decision_arms_done` (A1).
5. Reported steps, each only if 1.5 times its estimate (the mean arm time times its share of tokens) ends 60 s before
   the soft deadline (A1, R-77 F2): BF16R (A1); JOINT8; GEN (greedy, up to 128 tokens, stopped at the end token (A1,
   R-77 F10; `ignore_eos` before A1), for BF16, BF16C, STOCK8 and TIED8; fixed batch); TIED8_UP. A failed reported
   step is recorded and the run goes on. The record and the hook evidence are written at each step, and SIGTERM (the
   session timeout) saves the record before the exit.
6. The analysis and the frozen decision on the Pod (`analyze_e55.py`); the main session copies RES (about 60 MB) and
   recomputes the decision on the Mac from the copied raw files (S8; A1, R-77 answer 8).

## 5. Metrics
- Per arm X: Delta_X = sum over documents of S_X(d) / sum of n(d), where S_X(d) is the sum over the scored tokens of
  (BF16 log-probability - arm log-probability), the NLL change, and n(d) the number of scored tokens. Unit: nats per
  token; lower is better.
- Interval (A1, R-77 F5): a stratified t interval over documents. The variance of the linearized ratio comes from the
  spread of the residuals S_X(d) - Delta_X n(d) inside each domain (4 documents per domain), with Satterthwaite degrees
  of freedom. FACT (the A1 simulation on the Mac: the plan's token counts, normal document effects, 4,000 runs per
  case): the t interval covers 95.1% to 95.9% with equal or unequal spreads and 93.8% when the books dominate; the
  percentile bootstrap (the rule before A1) covers 85%; a t interval with 9 fixed degrees of freedom covers 89.9% to
  94.2%. The t interval is about 1.5 times wider than the bootstrap. D = Delta_TIED8 - Delta_STOCK8 and E =
  Delta_TIED8 - 1.5 Delta_STOCK8 use the same method on the paired document sums. The percentile bootstrap (10,000
  draws, seed 55) and a bootstrap over 2,048-token blocks are reported.
- Primary of H16: Delta_BF16C and its interval. Margin m16 = 0.001 nats per token: 0.1% of perplexity (A1, R-77 F4:
  before A1, min(0.001, 0.1 x Delta_E5M2), which depends on the data and can fall below the resolution). 0.1 x
  Delta_E5M2 is reported as a second margin.
- Primary of H8: D and its interval; tolerance tol8 = max(0.0002, 0.25 x Delta_STOCK8), also reported in percent of
  Delta_STOCK8 (A1); E and its interval (KILL8, Section 7).
- Secondary (reported): the top-1 agreement of each arm with BF16; Delta by domain and by context bin (1,025 to 8,191,
  8,192 to 32,767, 32,768 to 65,535); Delta over all tokens (index 1 or more; A1 fixes the code, which started at index
  2, R-77 F9); M1 = Delta_TIED8 / Delta_STOCK8 and M2 = Delta_TIED8 / Delta_JOINT8; Delta_TIED8_UP; (A1) BF16R:
  Delta_BF16R and Delta_BF16C - Delta_BF16R with its interval; (A1) the per-token SD of each arm's change (the size of
  a second run, Section 9); (A1) Delta on the far-repeat subset (Section 4.3) with its interval; for GEN, the exact
  matches (Wilson 95%), the median first divergence, the token agreement and the stop reasons.
- Repeats: one per arm. The computation is deterministic (V3 checks it); the unit of replication is the document.

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU is an H100 80GB.
- V2: `data_e55.json` and `scales_e54.json` have the frozen hashes; the documents and lengths are the plan; the engine
  tokenizer gives the `tok_check` ids; the engine is eager with the BF16 cache, no prefix caching and 4,096 batched
  tokens; the model snapshot folder is the pinned revision (A1, R-77 F9: before A1 the code compared the command-line
  value with itself); the SHA-256 of `splice_e55.py`, `run_e55.py`, `analyze_e55.py`, `session_e55.sh`,
  `data_e55.json` and `scales_e54.json` on the Pod equal the design commit (L-25, `code_manifest.json` from
  `make_manifest.py` on the Mac; A1).
- V3: the hooks sit on the 10 global layers (32 query heads, 4 KV heads, head size 512); w per layer is one value and
  equals the scales file; in every arm each layer saw every token of every document; DET equals BF16 bit for bit (the
  identity hooks change nothing, and the engine is deterministic), or, if not bit for bit, DET differs from BF16 by
  at most 0.0001 nats per token on its two documents (a tenth of the margin; the difference is reported).
- V4: every decision arm ran; every array has the planned length and only finite values; run_main status ok, or the
  decision arms finished before a later error or timeout (`decision_arms_done`; A1, R-77 F2).
- V5: the first call of each arm in each layer passes the transform check (`check_transform`: for example the K codes
  on the unrotated dims equal the V codes for the tied arms, and the impl scale is the score factor); on the first
  chunk of the first document of each arm, the engine's attention output equals a float32 reference of the arm (16
  query rows) within 0.01 in every layer (E-54: 0.0017; a kernel that ignored the K descale gives about 1); no hook
  error.
- V6: the E5M2 control changes the NLL: (A2, D-96) its 95% interval excludes 0, with either sign (before A2: the
  lower bound above 0). S5 showed that the sign depends on the model and the text (Section 9); a splice that does
  nothing gives no change, and both forms catch it.
- V7 (A1, R-77 F8): in each layer where the arm's float32 reference and the reference of the unchanged q, k and v
  differ by more than 0.001, the engine output is nearer the arm's reference; BF16C decides in 5 layers or more. Why:
  V5 cannot tell a hooked BF16C from one that the engine ignored (R-77: the two gaps are 0.0034 and 0.0063, both below
  0.01). FACT (E-54): the BF16C error is 0.0025 to 0.0098 in 9 layers; layer 29 has w = 2^-4, so BF16C is exact there.
  INTERPRETATION: the engine's own error (0.0017) and the arm's change come from different roundings, so the gap to
  the unchanged reference is about the root of the sum of their squares; the A1 toy tests confirm the separation.
- (A1, R-77 F2) A V3, V4, V5 or V7 problem of a reported arm (BF16R, JOINT8, TIED8_UP) drops that arm from the
  report; it does not make the run INVALID.

## 7. Decision rule (frozen in A1, before any data; project heuristics)
- INVALID: any of V1 to V7 (the result names them).
- H16, in this order: KILL16 if Delta_BF16C > 0.01; GO16 if its interval lies inside [-0.001, +0.001]; PIVOT16 if the
  lower bound of its interval is above 0.001 (and Delta_BF16C <= 0.01); CHECK16 if the upper bound is below -0.001
  (A1, R-77 F9: rounding cannot improve the NLL by that much, so it is an artifact); otherwise INCONCLUSIVE16, with the
  condition named.
- H8, in this order: GO8 if the upper bound of the interval of D is at most tol8; KILL8 if the lower bound of the
  interval of E = Delta_TIED8 - 1.5 Delta_STOCK8 is above 0 (A1, R-77 F5: TIED8 is worse than 1.5 times STOCK8 with
  95% confidence; before A1 the rule compared the point ratio); PIVOT8 if the lower bound of the interval of D is
  above 0; otherwise INCONCLUSIVE8.
- (A1, R-77 F5) GO8 means "not worse than STOCK8 by more than tol8 nats per token". If Delta_STOCK8 is below 0.0008,
  tol8 is the floor 0.0002, which is more than 25% of Delta_STOCK8; the report gives tol8 in percent of Delta_STOCK8.
- FACT (algebra): within each hypothesis the outcomes cannot overlap, because the rule checks them in order.
- Consequences (the user confirms each step; written before any data):

| Outcome | Action |
|---|---|
| GO16 | the thesis and the paper claim that the compact BF16 cache keeps the model's next-token predictions within 0.001 nats per token (model level, 3 domains, 32K to 64K tokens); the BF16 compact build (D-88) can start, then the engine capacity test |
| PIVOT16 | the compact BF16 cache has a small measured cost: report it; the user decides on the build |
| KILL16 | stop the 16-bit claim and find the cause (a fault is likely, because the layout is exact) before any new run |
| CHECK16 (A1) | a significant improvement cannot come from rounding: treat it as an artifact, find the cause, no claim |
| INCONCLUSIVE16 | report; the user decides on a second run with more documents (Section 9) or another metric (a new S4) |
| GO8 | the compact FP8 row stays a build option after the BF16 build |
| PIVOT8 | the compact FP8 row is measurably worse than the stock FP8 cache: it needs new scales (R-76 offline work) and a new test |
| KILL8 | no compact FP8 claim above the attention level; the build uses the BF16 compact cache only |
| INCONCLUSIVE8 | report; no compact FP8 claim at the model level |
| INVALID | fix the cause; a rerun needs a new S4 |

## 8. Controls and baselines
- Baseline: BF16, the BF16 cache with identity hooks.
- A/A: DET (two documents again without hooks, bit-identical expected, V3) and BF16R (one more BF16 rounding of the
  query: the noise floor of a rounding of the BF16C size, reported). (A1, R-77 F7) BF16R replaces BF16_B (a second
  engine with 2,048 batched tokens): BF16_B changed only the chunk size, so a per-row kernel could give the BF16 values
  bit for bit and measure nothing, and it needed a second engine start.
- Positive controls: E5M2 (must change the NLL, V6) and TIED8_UP (expected above TIED8; reported, the scale tolerance
  of R-76 C4 at the model level).
- Negative control: the identity hooks of BF16 (must change nothing, V3).

## 9. Budget
- Hardware and price: 1 x H100 80GB HBM3 SXM, Secure, on demand; $3.99/h at the last read (E-54, 2026-11-02). S4 reads
  the price and the balance live.
- Caps (A1, R-77 F9): cost cap $4.00; the boot watchdog at 2,640 s after the container starts (`pod_cmd.py --cap
  2640`; 2,820 s before A1); the session watchdog at 2,400 s (2,580 s before A1); AUTO_STOP 300 s after the session
  ends. Worst case: 15 min to SSH (the run script deletes the Pod after 15 min without SSH, as E-54) plus the boot
  watchdog: 59 min, $3.92 at $3.99/h. Before A1 the worst case was 62 min, $4.12.
- Time per step (estimates from E-54: the capture prefill took 1.7 to 2.6 s for 8K tokens and 10.1 s for 32K; the
  model t = 2.24e-4 L + 2.5e-9 L^2 s gives about 25 s for 64K, so about 3 min per arm of 524,288 tokens; the
  prompt log-probabilities add a few seconds per arm, INTERPRETATION): create to SSH 12 min (E-54: 724 s); upload 1
  min; setup 1.5 min; engine 1 min; warm-up and pre-flight 1 min; decision arms with DET 15 to 16 min; BF16R 3 min;
  JOINT8 3 min; GEN 1.5 min; TIED8_UP 3 min; analysis 1 min. Session about 31 min; Pod about 45 min, about $3.0.
- What the time cap cuts first (A1): TIED8_UP, then GEN, then JOINT8, then BF16R (the order of work runs them last to
  first). The decision arms always run; if they cannot finish, the result is INVALID (V4).
- S5 rehearsal (as E-53): 1 x RTX 4090 (or L4), the same image, `google/gemma-4-E2B-it` at revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, `REH=1` (documents cut to 16,384 tokens; no w and shape checks, because
  E2B has no K=V; (A1, R-77 B2) synthetic scales: vs 0.125, ks = f32(w vs), joint ks 0.05 and vs 0.125; the pre-flight
  records its problems and does not stop), about 20 min, cap $0.60. VERIFY: (1) the image and the vLLM version; (2) the
  engine starts with the E-55 arguments; (3) `prompt_logprobs` with `flat_logprobs` passes the parser; (4) the hooks
  fire on every global layer that owns a KV cache, with full token counts; (5) DET equals BF16 in the pre-flight and
  the main phase; (6) E5M2 changes the log-probabilities and the identity does not; (7) the V5 gap, V7 and the impl
  scale work on the rehearsal backend for every arm (A1); (8) the time per document and the memory headroom; (9) GEN
  returns up to 128 tokens per prompt; (10) the session files, the code hashes and the REHEARSAL line, which shows the
  validity problems (A1). The rehearsal backend is not FA4 (SM89), so the H100 pre-flight (Section 4.5) is the check of
  item 7 for FA4. (A1) If no rehearsal GPU is in stock (E-54 A2), the pre-flight limits the loss of a failed FA4 route
  to about $0.7; the user decides on a waiver in S4.
- S5 record (2026-11-03, D-95). Attempt 1 (RTX 4090 Pod `n9c39lsykrmxty`, Secure, no data center shown): no SSH
  port 15 min after the create (the host was still fetching the image at 22:45:58 UTC); deleted with no data, about
  $0.21. The first run script was killed at the end of a Claude session during its SSH wait; the run script now runs
  in its own process session (L-27). Attempt 2 (RTX 4090 Pod `paog5toj98cj38`, EU-RO-1, about 8 min, about $0.12;
  an SSH limit of 600 s and a boot watchdog of 900 s, to stay inside the $0.60 cap): the SSH port 207 s after the
  create; setup 17 s; engine 26.2 s; pre-flight 1.9 s; 12.2 to 12.4 s per arm for 196,608 tokens; GEN 23.4 s;
  session 187 s; no cut. Results: `results/e55-rehearsal2-4090-20261103/`.
  VERIFY: (1) vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, transformers 5.20.2: PASS. (2) The engine: max
  model length 65,664, BF16 auto cache, no prefix caching, 4,096 batched tokens, compilation mode "0" (NONE), CUDA
  graphs NONE, the V2 model runner: PASS, but V2 read the mode "0" as compiled (problem 1). (3) The parser: 288 of
  288 token counts equal the document lengths: PASS. (4) The hooks fire on the 3 E2B global layers that own a KV
  cache (4, 9 and 14; the Triton backend): PASS. (5) DET equals BF16 bit for bit in the pre-flight and the main phase:
  PASS. (6) E5M2 changes the log-probabilities (top-1 agreement 0.962), and the hooks in identity mode do not: PASS.
  (7) V5 gaps 0.0016 to 0.0023 for every arm, the scaled arms included (the K and V descale route works on Triton);
  V7 passes: PASS. (8) Peak memory 19.2 of 24.6 GB: PASS. (9) GEN returns 128 tokens for all 16 prompts in BF16 and
  STOCK8: PASS. (10) The session files, 6 of 6 code hashes and the REHEARSAL line with its problems: PASS.
  Problem 1 (a bug, fixed in A2): `str()` of the vLLM CompilationMode (an IntEnum) gives "0", and V2 needed "NONE" in
  the text, so every H100 run would have been INVALID; the stub wrote "CompilationMode.NONE", so the Mac tests passed
  (L-28). Problem 2 (a rule, changed in A2 by the user's decision D-96): E5M2 lowered the E2B NLL, Delta -0.0104
  [-0.0155, -0.0054] (books -0.0195, prose -0.0169, code +0.0050), so a working splice failed the one-sided V6.
  Expected in E2B and not a fault: BF16C, TIED8 and TIED8_UP at about +2.3 nats (E2B has no K = w V, so the rebuilt K
  is wrong; separations 0.58 to 0.85) and JOINT8 at +1.11 (synthetic scales). BF16R -0.00028 [-0.00096, +0.00040],
  top-1 agreement 0.985.
- (A1, R-77 O5) A second run if a result is INCONCLUSIVE for noise: four times the tokens (48 documents, 16 per domain;
  the first 12 are this run's) halves the interval. About 62 min of decision arms, so about 85 to 95 min of H100, $5.5
  to $6.5. The user decides it in a new S4.
- Balance floor: the protocol rule is the cap plus $35 (abort rule 5). If the live balance is lower, the user decides
  (as D-92 and D-93).

## 10. Threats to validity
1. The splice is not the build. BF16C is the exact compact numerics; the FP8 arms use the BF16 kernel with exact codes
   and the descales in the kernel scale and the output. The FP8 kernel (P in FP16) and the write kernel are not run.
   The E-54 kernel arm matched the FP8 kernel to the BF16 floor in decode. Stays open for the FP8 kernel in prefill.
   (A1) The output hook rounds twice (the BF16 output, then vs); the FP8 kernel rounds once (Section 4.4).
2. Global layers only. The FP8 arms keep the sliding layers in BF16, so Delta_STOCK8 is the effect of FP8 in the 10
   global layers, not of a full FP8 cache. This isolates the I-182 format; a full-cache test is out of scope.
3. Contamination. The PG-19 books may be in the training data; the prose and code are not. Results by domain.
4. Few documents: 12, 4 per domain. (A1, R-77 F5) The percentile bootstrap covers about 85% with 4 documents per
   domain; the decision uses the stratified t interval (about 95% in the A1 simulation), which is about 1.5 times wider.
5. NLL is not task accuracy. GEN shows identical text or not; no task score is measured (no GSM8K, no retrieval). The
   far-repeat subset (0.78% of the scored tokens) is the part that needs the global layers most.
6. One checkpoint (FP8-dynamic weights), one engine version. "BF16" here is the KV cache, not the weights.
7. The scales come from 4 prompts of 8K tokens (E-53 captures). The 64K books are another regime (R-76 C5: the C0 row
   behaves differently at 32K). Results by context bin.
8. Scored tokens start at index 1,025. The first 1,024 tokens (inside the sliding window) are in the all-token Delta.
9. The time cap can cut reported steps; the result names them.
10. (A1, R-77 F6) The prompt log-probabilities: the V2 runner scores 1,024 rows per call (about 1 to 3 GB outside the
    KV pool, INTERPRETATION); utilization 0.70 leaves about 24 GB. The pre-flight scores one 4,096-token chunk first.
11. The impl scale is changed at run time. V5 and V7 check that the engine output follows the arm in every layer.
12. (A1, R-77 finding 9 and F4) Resolution. Each arm is also a new realization of the FP8 activation noise of the
    checkpoint. HYPOTHESIS (R-77 model): a per-token SD of 0.05 to 0.11 nats gives a half-width of 3e-4 to 7e-4 nats
    per token for the t interval (design effect 3). Then GO16 needs |Delta_BF16C| below about 0.0003 to 0.0007, and
    INCONCLUSIVE16 is possible for a noise reason. BF16R and the per-token SD show which case holds.
13. (A1, R-77 F9) KL is not measured (D-94 option 1 named it). The signed NLL change is the metric; a truncated KL needs
    a second pass with fixed token ids (R-77 O3) and fits a later S4.
14. (A1, R-77 answer 8) The analysis checks the hashes of its own file, so it cannot prove its own integrity. The Mac
    recomputes the decision from the copied raw files (S8).

## 11. Files
| File | What it does | Tests |
|---|---|---|
| `prepare_e55.py` | builds `data_e55.json` on the Mac (downloads go to `data/`, which git ignores) | data plan test |
| `data_e55.json` | the 12 documents, 16 generation prompts and the tokenizer check (frozen) | data plan test |
| `splice_e55.py` | the arm transforms, the V5 checks and the hooks with their evidence | capture, toy-module, gap, V7 and fault tests |
| `run_e55.py` | the engine, the hooks, the pre-flight, the arms, GEN, the time cuts; `--stub` with stub Attention modules | parser, engine-order and stub-session tests |
| `analyze_e55.py` | the statistics (t interval), V1 to V7, the pre-flight checks, the frozen rule, the ledger manifest | rule, t, coverage, far-repeat, V2 and manifest tests |
| `session_e55.sh` | the Pod session (watchdogs, code hashes, the main phase, analysis) | stub-session and limit tests |
| `make_manifest.py` (A1) | `code_manifest.json` from the design commit in git (L-25) | manifest test |
| `test_e55.py` | 27 tests (A2) | |

## 12. Verification record (stage S3)
2026-11-02, the Mac (`uv run --no-project --with numpy --with torch --with pytest`): 15 tests pass in 258 s.
- The splice on the E-54 captures (prompt 0, layers 5 and 41): every arm passes its transform check, and BF16C,
  STOCK8, TIED8, JOINT8 and TIED8_UP give the frozen E-54 errors to 1e-4 relative (FACT: to 4 digits or better).
- A toy Attention module: each arm's output equals its emulation bit for bit (positional and keyword calls), the
  token counts and the V5 gaps (below 0.01) are recorded, and detach restores the identity. A toy kernel that ignores
  the impl scale gives a gap above 0.3 for TIED8 (V5 catches it).
- The parser reads the vLLM flat layout and refuses a wrong prompt token or a wrong entry count.
- The decision rule gives each of the 8 outcomes on planted statistics; the bootstrap is stratified and fixed by the
  seed; the data file is the plan (12 documents, BOS once, no source twice, 16 prompts that end with the model turn).
- The stub session (the full documents, 8 arms, synthetic log-probabilities): GO16 GO8 with the default effects;
  KILL16 with a planted BF16C effect of 0.02; KILL8 with TIED8 at 3 times STOCK8; INVALID (V6) with no E5M2 effect;
  INVALID (V3) with a DET rerun 0.001 away from BF16, and GO16 GO8 with a rerun 0.00005 away (not bit-identical); a
  short watchdog cuts JOINT8, GEN, TIED8_UP and the noise phase and keeps the decision; V2 compares the code hashes;
  the ledger accepts the manifest.
- Defect found and fixed before the freeze: the first FP8 form (bf16(scale x code) in the cache) moved the TIED and
  JOINT errors by 2% to 6% on the captures (Section 2); the arms now store the codes.
- Not run on the Mac: vLLM, the FA4 kernel and the 31B model (S5 and V5 cover them).

2026-11-03, amendment A1, the Mac (same command): 26 tests pass in 297 s.
- (B1) The stub engine has 10 small Attention modules with the real hooks and sends every token in the engine order
  (the warm-up without a document, the pre-flight, the arms, DET with the hooks removed, GEN). Every arm records its V5
  gap in every layer. Without the reset after the warm-up, BF16 records none: the R-77 failure, reproduced and fixed.
- (F8) V7: a toy engine that uses the inputs from before the hook passes V5 for BF16C (gap below 0.01) and fails V7;
  the scaled arms fail V5. With the correct engine, V5 and V7 pass for every arm. In the stub session BF16C decides V7
  in the 9 layers whose w is not a power of 2, and layer 29 (w = 2^-4) has a separation of exactly 0.
- (F3) The pre-flight stops a kernel that ignores the impl scale (V5) and an engine that ignores the hook (V7), and a
  DET rerun 0.001 away, before any arm of the 12 documents (exit 3; the session log says PREFLIGHT STOP).
- (F2) A failed reported step (GEN) does not stop TIED8_UP or change the decision. SIGTERM after the decision arms
  saves the record with the hook evidence, and the decision is GO16 GO8.
- (B2) The rehearsal stub runs every arm, GEN and DET with synthetic scales; the REHEARSAL line shows the validity
  problems.
- (F5) The t quantile matches the table (5 values to 1e-5). The coverage test (1,500 simulated runs) passes its
  bounds: the t interval covers 93% to 97.5%, and the bootstrap at least 3 points less (Section 5: 4,000 runs a case).
- (O2) The far-repeat mask finds a copy beyond the window and not one inside it.
- (F9) `make_manifest.py` hashes the committed files: `data_e55.json` and `scales_e54.json` give the frozen hashes.
- The rule gives each of the 10 outcomes, CHECK16 and KILL8 on E included; the worst case fits the cap; the stub
  session ends in about 25 s at the real data size.

2026-11-03, amendment A2, the Mac (same command): 27 tests pass in 541 s.
- The V2 eager check reads the engine record of this vLLM build: "0" and "NONE" pass, a compiled engine ("3", or CUDA
  graphs FULL_AND_PIECEWISE) fails. The stub records the S5 engine values.
- V6 two-sided: a planted E5M2 effect of -0.01 passes V6 (GO16 GO8); no E5M2 effect still gives INVALID.
- The rehearsal 2 record, analysed again on the Mac with A2: the only problem is V1 (the RTX 4090, expected).

## 13. Result (S6 to S8)
Run (S6, 2026-11-03 to 2026-11-04, D-95): 1 x H100 80GB HBM3 SXM, Pod `8ppguohfyhuh9v`, Secure, EU-FR-1, $3.99/h; the
SSH port 354 s after the create; setup 62 s; engine 94.9 s (FlashAttentionImpl); pre-flight 3.5 s, no problems; 71.5 to
72.9 s per arm for 524,288 tokens; GEN 67.7 s; session 828 s; no cut; peak memory 60.3 GB; about 21 min of Pod, about
$1.38 (the first billing read gave $1.00). Results: `results/e55-h100-20261103/` (RES copied, 37 checksums match).
- Decision (frozen rule, A2): **INVALID (V6)**. The E5M2 interval is [-0.0375, +0.1259]; it includes 0. FACT: the Mac
  recompute gives the same line to 1e-16. FACT: the A1 form of V6 (the lower bound above 0) also fails, so A2 does not
  decide the outcome.
- The other validity checks pass. FACT: V5 gaps 0.0017 to 0.0025 for every arm on FlashAttention (the K and V descale
  route works); V7 passes in every deciding layer (BF16C 9 of 9; layer 29 has a separation of 0; the other arms 10 of
  10); 960 of 960 token counts equal the document lengths; DET is bit-identical to BF16; the code hashes match the A2
  commit; no hook errors.
- Statistics (reported; no decision can use them). FACT: Delta (nats/token) and the 95% t interval: BF16C +0.0045
  [-0.0302, +0.0391]; BF16R -0.0039 [-0.0221, +0.0143]; E5M2 +0.0442 [-0.0375, +0.1259]; STOCK8 +0.0026 [-0.0216,
  +0.0269]; TIED8 -0.0336 [-0.0695, +0.0023]; JOINT8 -0.0029 [-0.0216, +0.0159]; TIED8_UP +0.0455 [+0.0210, +0.0699].
  The per-token SD of the change is 1.67 to 1.91 nats for every arm; the top-1 agreement with BF16 is 0.76 to 0.80 for
  every arm, BF16R included. In the pre-flight, only 0.2% to 0.3% of the log-probabilities of any changed arm equal
  BF16.
- Predictions (Section 2). FACT: the top-1 agreement of BF16R is 0.80, not 95% or more; the per-token SD is 1.67 nats,
  not 0.02 to 0.11; |Delta_BF16C| below 1e-4 cannot be tested with an interval of +/-0.035. The resolution (R-77 finding
  9) is about 100 times worse than the plan.
- Exploratory (after the data; not part of the rule). FACT: the BF16 baseline of the 31B is poor on these raw-text
  documents: mean NLL from token 1,025 is 8.6 to 10.5 nats on the books, 5.9 to 7.2 on the prose and 3.9 to 4.7 on the
  code; the top-1 accuracy is 0.11 to 0.20, 0.31 to 0.39 and 0.52 to 0.57. On the same documents (cut to 16,384 tokens)
  the 2B rehearsal model Gemma 4 E2B gives 5.3 to 6.5, 4.1 to 4.8 and 1.8 to 2.4 nats, so the 31B is worse than the 2B
  model on every document. The 31B top-1 predictions collapse onto a few tokens (" same" 1,249 times in P1, "//" 1,268
  times in C1, " own"); E2B predicts the expected tokens at the same positions. The damage does not depend on the row
  in the 4,096-token chunk and starts in the first 1,024 tokens. In the chat-format GEN step the 31B writes coherent and
  correct summaries. INTERPRETATION: the instrument cannot resolve the 0.001-nat margin; the raw-text scoring of this
  checkpoint in this engine is degraded, and every changed arm gives a new chaotic realization of it. The cause is not
  known; R-78 lists the candidates (Section 14).

## Amendments
A2 (2026-11-03, after the S5 rehearsal 2 and before any 31B data; D-96, the user). Both items come from the
rehearsal (PROTOCOL: a fix that the rehearsal required; Section 9, S5 record).
1. The V2 eager check accepts the compilation mode "0", the text that this vLLM build records for CompilationMode.NONE;
   a compiled engine still fails; the stub records the S5 engine values (Sections 6 and 12; L-28).
2. V6 is two-sided: the E5M2 interval must exclude 0, with either sign. The user chose it (D-96, "Go ahead with your
   recommendation") after a plain-language explanation; the decision rules H16 and H8 do not change (Section 6).
Code: `analyze_e55.py`, `run_e55.py` (the stub record only) and `test_e55.py`; V2 compares the Pod code with the A2
commit.

A1 (2026-11-03, before any E-55 data; R-77 AMEND_THEN_RUN, `dossier/REVIEW_E-55_PRERUN_2026-11-02.md`). The author
checked every finding against the code and the vLLM source before the fix.
1. B1: `Splice.reset_evidence` after the warm-up and after the pre-flight; the stub engine carries the real hooks in
   the engine order, and a test replays the order (Sections 4.5 and 12).
2. B2: synthetic scales in the rehearsal; the REHEARSAL line shows the validity problems (Section 9).
3. F1: no noise phase (BF16_B is replaced by BF16R, F7), so a failed second engine cannot invalidate the run.
4. F2: the record and the hook evidence at every step; `decision_arms_done`; a reported arm with a problem is dropped,
   not the run; 1.5 times the estimate for each reported step; a SIGTERM handler (Sections 4.5 and 6).
5. F3: the pre-flight with fail-fast checks (Section 4.5).
6. F4: the fixed margin 0.001; 0.1 x Delta_E5M2 reported (Sections 1, 5 and 7).
7. F5: the stratified t interval with Satterthwaite degrees of freedom; KILL8 on the interval of E; the meaning of
   GO8 and tol8 in percent of Delta_STOCK8 (Sections 5 and 7). The author chose Satterthwaite over 9 fixed degrees of
   freedom (R-77's t interval) because the fixed form covers 89.9% when the books dominate (Section 5).
8. F6: the V2 runner facts and `use_v2_model_runner` in the engine record (Section 4.2); reported, not a V2 condition.
9. F7: BF16R in the same engine, as the A/A (Sections 4.4 and 8).
10. F8: V7, a per-layer check of the engine output against the arm's reference and the unchanged reference, with 5
    deciding BF16C layers; and the pre-flight stop if BF16C gives the BF16 values bit for bit while DET is
    bit-identical.
    Not as R-77 proposed (50% of the tokens must differ): a share of equal tokens depends on the noise and on
    determinism, and a nondeterministic engine would hide an ignored hook in it.
11. F9: the all-token Delta from index 1; the snapshot folder as the revision; new predictions; the watchdogs 2,400 s
    and 2,640 s for a worst case inside the cap; CHECK16; `make_manifest.py`; the KL omission (Sections 2, 5, 6, 7, 9
    and 10).
12. F10: GEN stops at the end token, and the stop reasons are recorded (Sections 4.5 and 5).
13. O2: the far-repeat subset, reported (Sections 4.3 and 5). O5: the size of a second run (Section 9).
Not taken: O1 (a real stock FP8 reference: a third engine start, about $0.35 and 5 min, and a new engine setting; both
FP8 arms of H8 use the same emulation, so their comparison is consistent; the real FP8 kernel stays threat 1); O3 (a
truncated KL: a second pass with fixed ids, a later S4; threat 13); O4 (an FP32 lm_head: it changes the engine from
stock and needs a check of the unquantized head; BF16R measures the noise instead).
