# E-55: model-level accuracy of the compact K=V cache (BF16 and FP8) in the Gemma 4 31B global layers (preregistration)

Status: PREDECLARED 2026-11-02. NOT RUN. S3: 15 tests pass (Section 12). Design review (S2): R-77 pending.
Idea: I-182. Decision: D-94 (the user chose option (1) on 2026-11-02, "Yes, plan E-55": the model-level test with a
16-bit arm and an 8-bit arm). Journal entry: Section 6.1 E-55.
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
All thresholds are project heuristics unless a source is given.

## 1. Question
At the model level, does the compact K=V cache of the 10 Gemma 4 31B global layers keep the model's next-token
predictions on long documents?
- H16 (primary, 16-bit): with the compact row in BF16, is the change of the per-token negative log-likelihood (NLL)
  inside a margin of min(0.001, 0.1 x the E5M2 effect) nats per token? "No": the change is outside the margin
  (PIVOT16 or KILL16, Section 7).
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
  the layout is exact. (2) The splice does not reach the attention reads (V6, the E5M2 control). (3) The tied FP8 row
  is worse than the stock FP8 cache at the model level.
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
- Order of work: the BF16 reference first, the BF16C arm (H16) second, the positive control third (validity), then
  STOCK8 and TIED8 (H8), then the reported steps while the time allows.
- Predictions (HYPOTHESIS, written before any E-55 data; checked after the run, not used by the rule):
  |Delta_BF16C| below 1e-4; Delta_STOCK8 0.0002 to 0.005; Delta_TIED8 / Delta_STOCK8 0.4 to 0.8 (E-54 R1 0.67; an NLL
  change that grows with the square of the error gives 0.45); Delta_TIED8 / Delta_JOINT8 1.0 to 1.5; Delta_E5M2 3 to 10
  times Delta_STOCK8; Delta_TIED8_UP above Delta_TIED8; BF16_B within 1e-4 of BF16 and a top-1 agreement of 99% or
  more; the BF16C top-1 agreement at most 0.5 points below that of BF16_B.

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
- Engine 1 (main phase): eager (`enforce_eager`), BF16 cache ("auto"), `max_model_len` 65,664, 4,096 batched tokens
  per step (chunked prefill), 8 sequences, utilization 0.70, no prefix caching (else an arm could read the KV of the
  arm before it), text only, seed 0, TF32 off. The utilization leaves about 24 GB for the prompt log-probabilities:
  per chunk the engine makes the logits of every prompt token (4,096 x 262,144 in BF16, then float32 log-probabilities).
- Engine 2 (noise phase): the same with 2,048 batched tokens (BF16_B, Section 8).
- FACT (vLLM 3966924d5, `gpu_model_runner.py` 5245 to 5352, `logprobs.py` 165 to 209, `sampling_params.py` 350 to
  383): prompt log-probabilities work with chunked prefill, chunk by chunk; with `prompt_logprobs=1` and
  `flat_logprobs` each position after the first holds [the prompt token with its rank, the top-1 token]. One request
  per document (`max_tokens` 1, `detokenize` False).
- FACT (flash-attention b273a02, `flash_attn/cute/interface.py` 1421 to 1500 and 1895 to 1905): the FA4 compile key
  holds no softmax scale; the scale is a run-time argument, so a new scale per layer does not compile a new kernel.
  FACT (E-54 capture log): the engine uses the FLASH_ATTN backend (`FlashAttentionImpl`, `self.scale`, flash_attn.py
  1077; every call passes `softmax_scale=self.scale`, 1413 to 1698).

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

### 4.4 Splice (`splice_e55.py`)
Forward pre-hooks on the Attention modules of the 10 global layers replace q, k and v before the module writes k and v
to the paged cache. The engine reads K and V from the paged cache in prefill and in decode (R-76, `flash_attn.py`), so
every read sees the arm's values. The 50 sliding layers keep the BF16 cache in every arm. c() is the vLLM cache cast
(saturate, round to nearest even).

| Arm | q | Cache K | Cache V | K descale (impl scale) | V descale (output) |
|---|---|---|---|---|---|
| BF16 | q | K | V | 1 | 1 |
| BF16C | [q_rot, bf16(q_unrot w)] | [K_rot, V_unrot] | V | 1 | 1 |
| E5M2 | q | c_E5M2(K) | c_E5M2(V) | 1 | 1 |
| STOCK8 | q | c(K) | c(V) | 1 | 1 |
| TIED8 | q | [c(K_rot / ks), c(V_unrot / vs)] | c(V / vs) | ks | vs |
| JOINT8 | q | c(K / ks_j) | c(V / vs_j) | ks_j | vs_j |
| TIED8_UP | q | as TIED8 with vs' = f32(vs 2^(1/32)), ks' = f32(w vs') | | ks' | vs' |

- w, ks, vs, ks_j, vs_j: the frozen `scales_e54.json` (SHA-256 `8efe6e90...`; ks = f32(w vs)). The hook reads w from
  the k_norm weight on the unrotated dims; it must be one value and equal the scales file (V3).
- The FP8 codes are exact in BF16, so the cache holds exactly the codes. The kernel multiplies the scores by the impl
  scale after the dot product, as the FP8 kernel multiplies by the K descale (R-74). The output hook multiplies the
  BF16 output by the V descale (one more BF16 rounding of the output).

### 4.5 Order of work (`run_e55.py`, `session_e55.sh`)
1. Setup: versions, telemetry (with memory), the SHA-256 of the code that runs (L-25), the checkpoint download.
2. Engine 1; the tokenizer check; the hooks (V3); one warm-up prompt of 512 tokens (not scored).
3. Decision arms, 12 documents each: BF16; DET (P1 and C1 again with the hooks removed); BF16C; E5M2; STOCK8; TIED8.
4. Reported steps, each only if its estimate (the mean arm time times its share of tokens) ends 60 s before the soft
   deadline: JOINT8; GEN (greedy, 128 tokens, `ignore_eos`, for BF16, BF16C, STOCK8 and TIED8; fixed batch); TIED8_UP.
5. Engine 2 (BF16_B), only if 540 s or more remain before the session watchdog.
6. The analysis and the frozen decision on the Pod (`analyze_e55.py`); the main session copies RES (about 60 MB).

## 5. Metrics
- Per arm X: Delta_X = sum over documents of S_X(d) / sum of n(d), where S_X(d) is the sum over the scored tokens of
  (BF16 log-probability - arm log-probability), the NLL change, and n(d) the number of scored tokens. Unit: nats per
  token; lower is better.
- Interval: a paired bootstrap over documents, stratified by domain (4 per domain), 10,000 draws, seed 55, the 2.5%
  and 97.5% percentiles. D = Delta_TIED8 - Delta_STOCK8 uses the same draws.
- Primary of H16: Delta_BF16C and its interval. Margin m16 = min(0.001, 0.1 x Delta_E5M2): 0.1% of perplexity at most,
  and at most a tenth of the effect of a known lossy storage of the same layers.
- Primary of H8: D and its interval. Tolerance tol8 = max(0.0002, 0.25 x Delta_STOCK8).
- Secondary (reported): the top-1 agreement of each arm with BF16; Delta by domain and by context bin (1,025 to 8,191,
  8,192 to 32,767, 32,768 to 65,535); Delta over all tokens (index 1 or more); M1 = Delta_TIED8 / Delta_STOCK8 and M2 =
  Delta_TIED8 / Delta_JOINT8; Delta_TIED8_UP; BF16_B (the noise floor of the engine); a bootstrap over 2,048-token blocks
  for Delta_BF16C and D; for GEN, the exact matches of the 128 tokens (Wilson 95%), the median first divergence and
  the token agreement.
- Repeats: one per arm. The computation is deterministic (V3 checks it); the unit of replication is the document.

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU is an H100 80GB.
- V2: `data_e55.json` and `scales_e54.json` have the frozen hashes; the documents and lengths are the plan; the engine
  tokenizer gives the `tok_check` ids; both engines are eager with the BF16 cache, no prefix caching and the planned
  batched tokens; the model revision is the pinned one; the SHA-256 of `splice_e55.py`, `run_e55.py`, `analyze_e55.py`,
  `session_e55.sh`, `data_e55.json` and `scales_e54.json` on the Pod equal the design commit (L-25, `code_manifest.json`
  from the Mac).
- V3: the hooks sit on the 10 global layers (32 query heads, 4 KV heads, head size 512); w per layer is one value and
  equals the scales file; in every arm each layer saw every token of every document; DET equals BF16 bit for bit (the
  identity hooks change nothing, and the engine is deterministic), or, if not bit for bit, DET differs from BF16 by
  at most 0.0001 nats per token on its two documents (a tenth of the largest margin; the difference is reported).
- V4: every decision arm ran; every array has the planned length and only finite values; run_main status ok.
- V5: the first call of each arm in each layer passes the transform check (`check_transform`: for example the K codes
  on the unrotated dims equal the V codes for the tied arms, and the impl scale is the K descale); on the first chunk
  of the first document of each arm, the engine's attention output equals a float32 reference of the arm (16 query
  rows) within 0.01 in every layer (E-54: 0.0017; a kernel that ignored the K descale gives about 1); no hook error.
- V6: the E5M2 control changes the NLL: the lower bound of its interval is above 0.

## 7. Decision rule (frozen; project heuristics)
- INVALID: any of V1 to V6 (the result names them).
- H16, in this order: KILL16 if Delta_BF16C > 0.01; GO16 if its interval lies inside [-m16, +m16]; PIVOT16 if the lower
  bound of its interval is above m16 (and Delta_BF16C <= 0.01); otherwise INCONCLUSIVE16, with the condition named.
- H8, in this order: GO8 if the upper bound of the interval of D is at most tol8; KILL8 if the lower bound of that
  interval is above 0 and Delta_TIED8 > 1.5 x Delta_STOCK8; PIVOT8 if the lower bound is above 0; otherwise
  INCONCLUSIVE8.
- FACT (algebra): within each hypothesis the outcomes cannot overlap, because the rule checks them in order.
- Consequences (the user confirms each step; written before any data):

| Outcome | Action |
|---|---|
| GO16 | the thesis and the paper claim that the compact BF16 cache keeps the model's next-token predictions (model level, 3 domains, 32K to 64K tokens); the BF16 compact build (D-88) can start, then the engine capacity test |
| PIVOT16 | the compact BF16 cache has a small measured cost: report it; the user decides on the build |
| KILL16 | stop the 16-bit claim and find the cause (a fault is likely, because the layout is exact) before any new run |
| INCONCLUSIVE16 | report; the user decides on more documents or another metric (a new S4) |
| GO8 | the compact FP8 row stays a build option after the BF16 build |
| PIVOT8 | the compact FP8 row is measurably worse than the stock FP8 cache: it needs new scales (R-76 offline work) and a new test |
| KILL8 | no compact FP8 claim above the attention level; the build uses the BF16 compact cache only |
| INCONCLUSIVE8 | report; no compact FP8 claim at the model level |
| INVALID | fix the cause; a rerun needs a new S4 |

## 8. Controls and baselines
- Baseline: BF16, the BF16 cache with identity hooks.
- A/A: DET (two documents again without hooks, bit-identical expected, V3) and BF16_B (engine 2 with 2,048 batched
  tokens: the noise floor of a routine engine setting, reported).
- Positive controls: E5M2 (must change the NLL, V6) and TIED8_UP (expected above TIED8; reported, the scale tolerance
  of R-76 C4 at the model level).
- Negative control: the identity hooks of BF16 (must change nothing, V3).

## 9. Budget
- Hardware and price: 1 x H100 80GB HBM3 SXM, Secure, on demand; $3.99/h at the last read (E-54, 2026-11-02). S4 reads
  the price and the balance live.
- Caps: cost cap $4.00; the boot watchdog at 2,820 s after the container starts (`pod_cmd.py --cap 2820`); the session
  watchdog at 2,580 s; AUTO_STOP 300 s after the session ends. Worst case: about 12 min of image pull plus 47 min, about
  59 min or $3.92.
- Time per step (estimates from E-54: the capture prefill took 1.7 to 2.6 s for 8K tokens and 10.1 s for 32K; the
  model t = 2.24e-4 L + 2.5e-9 L^2 s gives about 25 s for 64K, so about 3 min per arm of 524,288 tokens):
  create to SSH 12 min (E-54: 724 s); upload 1 min; setup 1.5 min; engine 1 min; decision arms with DET 15 to 16 min;
  JOINT8 3 min; GEN 1.5 min; TIED8_UP 3 min; engine 2 and BF16_B 4 min; analysis 1 min. Session about 31 min; Pod about
  44 min, about $2.95.
- What the time cap cuts first: BF16_B, then TIED8_UP, then GEN, then JOINT8 (the order of work runs them last to
  first). The decision arms always run; if they cannot finish, the result is INVALID (V4).
- S5 rehearsal (as E-53): 1 x RTX 4090 (or L4), the same image, `google/gemma-4-E2B-it` at revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, `REH=1` (documents cut to 16,384 tokens; no w and shape checks, because
  E2B has no K=V), about 20 min, cap $0.60. VERIFY: (1) the image and the vLLM version; (2) the engine starts with the
  E-55 arguments; (3) `prompt_logprobs` with `flat_logprobs` passes the parser; (4) the hooks fire on every global
  layer that owns a KV cache, with full token counts; (5) DET equals BF16; (6) E5M2 changes the log-probabilities and
  the identity does not; (7) the engine-output gap of V5 and the impl scale work on the rehearsal backend; (8) the time
  per document and the memory headroom; (9) GEN returns 128 tokens per prompt; (10) the session files, the code hashes
  and the REHEARSAL decision line. The rehearsal backend is not FA4 (SM89), so V5 on the H100 is the check of item 7
  for FA4.
- Balance floor: the protocol rule is the cap plus $35 (abort rule 5). If the live balance is lower, the user decides
  (as D-92 and D-93).

## 10. Threats to validity
1. The splice is not the build. BF16C is the exact compact numerics; the FP8 arms use the BF16 kernel with exact codes
   and the descales in the kernel scale and the output. The FP8 kernel (P in FP16) and the write kernel are not run.
   The E-54 kernel arm matched the FP8 kernel to the BF16 floor in decode. Stays open for the FP8 kernel in prefill.
2. Global layers only. The FP8 arms keep the sliding layers in BF16, so Delta_STOCK8 is the effect of FP8 in the 10
   global layers, not of a full FP8 cache. This isolates the I-182 format; a full-cache test is out of scope.
3. Contamination. The PG-19 books may be in the training data; the prose and code are not. Results by domain.
4. Few documents: 12, 4 per domain. A bootstrap with 4 documents per stratum gives intervals that are a little too
   narrow (about 13% in width); the block bootstrap is reported as a second view.
5. NLL is not task accuracy. GEN shows identical text or not; no task score is measured (no GSM8K, no retrieval).
6. One checkpoint (FP8-dynamic weights), one engine version. "BF16" here is the KV cache, not the weights.
7. The scales come from 4 prompts of 8K tokens (E-53 captures). The 64K books are another regime (R-76 C5: the C0 row
   behaves differently at 32K). Results by context bin.
8. Scored tokens start at index 1,025. The first 1,024 tokens (inside the sliding window) are in the all-token Delta.
9. The time cap can cut reported steps; the result names them.
10. The prompt log-probabilities need about 6 to 11 GB per 4,096-token chunk outside the KV pool (BF16 logits of 2.1
    GB, float32 log-probabilities of 4.3 GB, and maybe a float32 copy); utilization 0.70 leaves about 24 GB; the first
    document fails fast if not.
11. The impl scale is changed at run time. V5 checks that the engine output follows it in every layer and arm.

## 11. Files
| File | What it does | Tests |
|---|---|---|
| `prepare_e55.py` | builds `data_e55.json` on the Mac (downloads go to `data/`, which git ignores) | data plan test |
| `data_e55.json` | the 12 documents, 16 generation prompts and the tokenizer check (frozen) | data plan test |
| `splice_e55.py` | the arm transforms, the V5 checks and the hooks | capture, toy-module, gap and fault tests |
| `run_e55.py` | the engine, the hooks, the arms, GEN, the time cuts; `--stub` for the Mac | parser and stub-session tests |
| `analyze_e55.py` | the statistics, V1 to V6, the frozen rule, the ledger manifest | rule, bootstrap, V2 and manifest tests |
| `session_e55.sh` | the Pod session (watchdogs, code hashes, phases, analysis) | stub-session and limit tests |
| `test_e55.py` | 15 tests | |

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

## Amendments
None.
