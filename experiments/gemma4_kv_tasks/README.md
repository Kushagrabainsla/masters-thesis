# E-56: output correctness of the compact K=V cache with the 16-bit Gemma 4 31B weights (preregistration)

Status: PREDECLARED 2026-11-04; amendment A1 2026-11-04 (R-79 AMEND_THEN_RUN, before any E-56 data); A2 2026-11-04
(after S5). RUN 2026-11-04 (A100): GO1 GO2 GO3 (Section 13). S3: 15
tests pass (Section 12). Next: S4 (the user's spend approval and a live balance check).
Idea: I-182. Decision: D-97 (the user, 2026-11-04, "Go ahead": a model-level test of output correctness on the 16-bit
weights on one A100 80GB, with a baseline-health gate). Journal entry: Section 6.1 E-56.
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
All thresholds are project heuristics unless a source is given.

## 1. Question
With the 16-bit weights of Gemma 4 31B (`google/gemma-4-31B-it`), does the compact BF16 K=V cache of the 10 global
layers (BF16C; (A1, R-79) the cache holds V on the 384 unrotated dims of K, and the k_norm weight w goes into the
query: q' = bf16(q w) on those dims) give the same task answers as the normal BF16 cache?
"No": the accuracy over the 432 items falls with the compact cache (the 95% interval of the change is below 0), or its
answers change more often than under a rounding-size perturbation by more than 3 points, or the NLL of the reference
answers rises by more than 0.01 nats per token.

## 2. Rubber duck
- The claim. In the global layers K_unrot = w x V_unrot exactly (`attention_k_eq_v`; FACT: the config of both 31B
  checkpoints). The compact row [K_rot | V] stores 640 values instead of 1,024 (37.5% fewer global KV bytes). (A1,
  R-79 point 9) The K view reads V on the unrotated dims, and w goes into the query (the two views of E-52), so in
  BF16 the compact path differs from the normal cache only by one BF16 rounding of q w on those dims. FACT (E-53):
  the attention-output error of this path is 0 to 0.0094 (0 in layer 29, where w = 2^-4).
- What is new since E-55. E-55 was INVALID (R-78): the FP8-dynamic checkpoint scored raw text worse than the 2B E2B
  model, and its per-token noise (SD 1.7 nats) hid any effect. E-56 changes the instrument: the BF16 weights (no FP8
  activation quantization); chat-format tasks, the way the model is used; task accuracy as the primary metric (the
  user asked for output correctness, 2026-11-04); and a baseline-health gate (V8, L-29).
- Memory. The compact cache is emulated: the engine still stores 1,024 values per token and head; the hooks give it
  the values that the compact row would give back. The saving is arithmetic. E-56 tests the output only.
- What kills the run early. The pre-flight (V5, V7 and DET on one chunk) and the health gate after the BF16 arm.
- Order of work. The warm-up; the pre-flight; BF16 and the gate; DET; BF16C; BF16R; then STOCK8 while the time allows.
- Predictions (HYPOTHESIS, written before any E-56 data; checked after the run, not used by the rule): BF16 GSM8K
  accuracy 0.85 to 0.95, needle accuracy 0.95 to 1.0, reference NLL 0.5 to 2.5 nats per token (A1, R-79 B1: human
  GSM8K solutions are not the model's own style; before A1, 0.3 to 0.8) with a top-1 agreement of 0.4 or more; the BF16C accuracy
  change within +/-1 point; the share of answers that differ from BF16 2% to 10% for BF16C and for BF16R alike (greedy
  decoding diverges at near-ties); |Delta_BF16C| of the reference NLL below 5e-4; STOCK8 changes more answers and more
  NLL than BF16C. If the FP8-dynamic checkpoint caused the E-55 degradation, the raw-text NLL of the BF16 31B on the
  health slices B1, P1 and C1 is below the E2B values on the same tokens (E-55 S5: about 5.5, 4.2 and 2.3).

## 3. Hypotheses
- H1 (primary): the task accuracy of BF16C equals that of BF16 within 3 points.
- H2 (secondary): BF16C changes the final answers no more often than BF16R (a rounding-size change of the query) does,
  within 3 points.
- H3 (secondary): the reference-answer NLL of BF16C equals that of BF16 within 0.002 nats per token.
- H0: the compact cache changes the outputs: lower accuracy, more changed answers, or a higher NLL.

## 4. Setup (frozen)
### 4.1 Hardware
1 x A100 80GB (SXM or PCIe), Secure Cloud, on demand, $1.79/h (FACT: live read 2026-11-04, stock LOW in the US and
Europe on CUDA 13.2 hosts), CUDA floor 13.0 (the image). Why: 80 GB holds the 62.6 GB of BF16 weights, at less than
half the H100 price. FA2 has no head size of 512, so on SM80 the global layers run on the Triton backend (FACT: the
E-55 S5 rehearsal on SM89 used TritonAttentionImpl for the 512-wide layers). If no A100 is in stock, an H100 80GB SXM
also passes V1 (the comparison is paired on one GPU); the user approves its price first.

### 4.2 Software
The E-53 to E-55 image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
(vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, transformers 5.20.2). Model `google/gemma-4-31B-it` at revision
`842da3794eaa0b77d5f08bae87a17459d91ff475` (FACT: Apache 2.0, not gated, 62.58 GB in 2 files; the same architecture
as the FP8-dynamic checkpoint of E-50 to E-55: 60 layers, the global layers 5 to 59 every 6, head size 512, 4 KV heads,
sliding window 1,024, final logit softcap 30). Engine: eager, BF16 auto cache, no prefix caching, max model length
10,240, 2,048 batched tokens, 64 sequences, GPU memory 0.92, no image, audio or video inputs, seed 0. Decoding:
greedy (temperature 0), 640 tokens for GSM8K and 128 for the needles ((A2, S5) 400 and 24 before A2; 320 for GSM8K
before A1), stop at `<turn|>`
(id 106; also in the checkpoint's generation config).

### 4.3 Data (`data_e56.json`, SHA-256 `4f8511c89b77481ee5f2b39927de41c99b6da63680ccd8c5f6bcccd6431db672`)
Built by `prepare_e56.py` on the Mac with the tokenizer and chat template of the pinned checkpoint (thinking off, the
template default: the generation prompt ends with an empty thought channel).
- gsm8k: 400 questions of the GSM8K test split (`openai/gsm8k` at `740312add88f781978c0658806c59bc2815b9866`, MIT),
  seed 56. The user turn: "Solve the math problem. Show brief steps, then write the final answer as a number after
  '####'." and the problem. The scoring sequence: the prompt, the reference solution without the calculator notes and
  with its final line "#### <gold>", and `<turn|>`; the scored tokens start at the reference (`answer_start`).
- needle: 32 prompts of 7,666 to 7,668 tokens: 4 PG-19 test books (the E-55 copies, seed 57), 7,600 tokens of each,
  one sentence "The secret passphrase for the archive is <word>-<4 digits>." at 8 depths (5% to 89%, at a line
  break), and the question "What is the secret passphrase for the archive? Answer with the passphrase only."
- health: BOS and the first 2,047 tokens of the E-55 documents B1 (a book), P1 (prose) and C1 (code). B1 is the probe.

### 4.4 Arms (`splice_e55.py`, unchanged; its tests are the E-55 tests)
| Arm | Query | K in the cache | V in the cache | Role |
|---|---|---|---|---|
| BF16 | q | K | V | reference |
| DET | q | K | V | BF16 again with the hooks removed (determinism, V3) |
| BF16C | [q_rot, bf16(q_unrot w)] | [K_rot, V_unrot] | V | decision (the compact row; (A1) w in the query) |
| BF16R | bf16(q (1 + 2^-7)), impl scale 1 / (1 + 2^-7) | K | V | decision (the noise floor of H2) |
| STOCK8 | q | fp8(K) | fp8(V) | reported (the stock 8-bit cache, scale 1) |

### 4.5 Workload and order
Per arm: the probe B1 scored alone (the first call of each arm in each layer: the V5 and V7 evidence needs one prompt
in the call), P1 and C1 scored alone (the raw-text NLL), the 400 scoring sequences (batched, prompt log-probabilities),
the 400 GSM8K answers and the 32 needle answers (greedy, batched; (A1, R-79 O1) the needle calls also return the
prompt log-probabilities). The run: the warm-up (not scored), the evidence
reset, the pre-flight (every arm and DET on the probe; exit 3 on a problem of V5, V7, the counts or DET), the evidence
reset, BF16, the health gate (V8; exit 3 on a problem), DET, BF16C, BF16R, then STOCK8 if 1.5 times the mean arm time
fits before the deadline. Every hooked layer must see at least the expected tokens of each workload: each scored prompt
once; each generated sequence its prompt and all but its last output token (preemption can add recomputed tokens).

## 5. Metrics
- Primary: the task accuracy (the share of the 432 items with a right answer). GSM8K: the first number after the last
  '####', else the last number in the text, equal to the gold as a number. Needle: the gold passphrase in the text.
  BF16C - BF16 with the paired Agresti-Min 95% interval (0.5 added to each cell of the 2 x 2 table). The minimum effect
  that matters: 3 points (heuristic).
- Secondary: the share of items whose final answer differs from the BF16 answer, and BF16C - BF16R with the same
  interval; the share of answers with the same token ids; the reference NLL change (nats per token over the answer
  tokens; Delta = sum of the per-item changes / sum of the answer tokens; a t interval of the linearized ratio over
  items); the accuracy per task; STOCK8 as a reference; the raw-text NLL of the health slices (the R-78 diagnostic).
  (A1) Also reported: the long-context NLL change over the needle prompt tokens from token 1,025 (R-79 O1; the same
  ratio interval over the 32 prompts); the reference top-1 agreement (an alignment check, R-79 B1); the GSM8K answers
  cut at the token limit, the share with '####' and the mean length (R-79 F4); the share of DET answers with other
  token ids (a second noise floor for H2, R-79 F1).
- Repeats: one run; the decoding is greedy, and DET checks that the engine is deterministic.

## 6. Validity checks (a failed check makes the run INVALID; a problem of STOCK8 drops STOCK8, not the run)
- V1: the GPU is an A100 or H100 with 80 GB.
- V2: the data file, the model revision, the engine (eager, BF16 auto cache, no prefix caching, 2,048 batched tokens;
  E-55 A2: the compilation mode "0" is NONE), the tokenizer check, and the code hashes against the design commit
  (`make_manifest.py`, L-25; the E-56 files and the three E-55 files that run).
- V3: the hooks are on the 10 global layers; every count reaches the expected tokens; DET gives the scored
  log-probabilities within 1e-4 nats per token (mean absolute difference). (A1, R-79 F1) The DET answer token ids are
  recorded, not a gate: H1 to H3 stay valid when one answer differs.
- V4: the decision arms BF16, DET, BF16C and BF16R ran.
- V5: the transform checks pass; the engine output equals the arm's float32 reference within 0.01 in every layer (the
  probe); no hook error; finite values.
- V7: in each layer where the arm's reference and the unchanged reference differ by more than 0.001, the engine output
  is nearer the arm's reference; BF16C decides in 5 layers or more (E-55 A1).
- V8 (new, L-29, R-78): the baseline is healthy: BF16 GSM8K accuracy 0.75 or more, needle accuracy 0.75 or more, and a
  reference NLL of 3.0 nats per token or less ((A1, R-79 B1) 1.5 before A1: a healthy instruct model can score human
  solutions near 2; the bound now catches only a broken or misaligned scoring), all items answered.
- (A1, R-79 F2) The pre-flight stops the run for a problem of BF16, BF16C, BF16R or DET only; a STOCK8 problem is
  recorded and STOCK8 is skipped.
- E-55's V6 (a sign-based E5M2 control) is not used (R-78: it tests noise). An interval too wide to decide gives
  INCONCLUSIVE in the rule.

## 7. Decision rule (frozen; project heuristics)
- H1: GO1 if the interval of acc(BF16C) - acc(BF16) is inside [-0.03, +0.03]; KILL1 if its upper bound is below 0;
  else INCONCLUSIVE1.
- H2: GO2 if the upper bound of differ(BF16C) - differ(BF16R) is 0.03 or less; WORSE2 if its lower bound is above
  0.03; else INCONCLUSIVE2.
- H3: GO3 if the interval of Delta_BF16C is inside [-0.002, +0.002] nats per token; KILL3 if its lower bound is above
  0.01; (A1, R-79 F7) SMALL3 if the interval is inside (0.002, 0.01] (a measured small increase); else INCONCLUSIVE3.
- The claim. GO1 with GO2: "with the 16-bit weights, the compact cache keeps the task accuracy within 3 points and
  changes the answers no more often than a rounding-size change of the query". KILL1, WORSE2 or KILL3: no claim; the
  compact BF16 cache changes the outputs. An INCONCLUSIVE result names the wide interval; a larger run needs a new S4.
- (A1, R-79 O3) Reading an INCONCLUSIVE result. H3 is the most decidable hypothesis (its interval is about 2e-4 to
  1e-3 wide). If H1 or H2 is INCONCLUSIVE and H3 is GO3, the claim is "no change in the reference NLL within 0.002
  and no detected accuracy loss; the interval of the accuracy change is [lo, hi]". R-79 power simulation (no true
  effect): GO1 with GO2 in 91%, 63%, 39% and 35% of runs at answer-change rates of 2%, 5%, 10% and 15%.
- INVALID if a validity check fails (the rehearsal records the problems in a REHEARSAL line and decides nothing).

## 8. Controls and baselines
- Baseline: BF16, the engine's own cache.
- A/A: DET (the same arm again without the hooks: must be identical) and BF16R (a rounding-size change of the query:
  how often a change of this size flips an answer).
- Positive reference: STOCK8 (expected to change more answers and more NLL than BF16C; reported, not a validity
  condition).
- Negative control: DET.

## 9. Budget
- Hardware and price: 1 x A100 80GB, Secure, $1.79/h (live read 2026-11-04). Cost cap $2.50.
- Caps (A1, R-79 F3): the boot watchdog at 4,380 s after the container starts (`pod_cmd.py --cap 4380`), the session
  watchdog at 4,200 s, AUTO_STOP 300 s after the session ends, and an SSH limit of 600 s in the run script (900 s,
  4,080 s and 3,900 s before A1). Worst case: 10 min to SSH plus the boot watchdog: 83 min, $2.48. The download has
  up to 3 resumed tries in 1,500 s; the Pod volume is 120 GB (62.6 GB of weights and the caches).
- Time per step (estimates, INTERPRETATION): create to SSH 5 to 12 min; setup and the 62.6 GB download 3 to 10 min;
  engine 3 min; pre-flight 0.5 min; 5 arm runs of about 5.5 min (the GSM8K answers about 3 min, the needles about 1.5,
  the scoring about 0.5); analysis 1 min. Session about 40 min; Pod about 50 min, about $1.50.
- What the time cap cuts first: STOCK8. The decision arms always run; if they cannot finish, the result is INVALID.
- S5 rehearsal: 1 x RTX 4090 ($0.89/h), the same image, `google/gemma-4-E2B-it` at revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`, REH=1 (the gate records and does not stop; E2B has no K = w V), an SSH
  limit of 600 s and a boot watchdog of 1,200 s: worst case 30 min, $0.45. VERIFY: (1) the engine starts with the E-56
  arguments; (2) (A1, R-79 F4) the chat-format answers stop at `<turn|>`: the share of GSM8K answers cut at 400 tokens
  (more than 5% needs an amendment before the run), the share with '####' and the mean length are
  recorded; the reference top-1 agreement is far above chance (an off-by-one alignment gives near chance); the needle
  prompt log-probabilities pass the parser; (3) the batched calls
  reach every hooked layer with the expected tokens; (4) DET reproduces BF16 for the whole workload; (5) V5 and V7 on
  the probe for every arm; (6) V8 is recorded in the REHEARSAL line; (7) the time per arm and the memory headroom;
  (8) the session files and the code hashes.
- S5 record (2026-11-04, D-98): 1 x A100 SXM 80GB, Pod `r94gd7nus3k1d4`, US-MD-1, about 14 min, about $0.43; the SSH
  port 248 s after the create; setup 49 s (E2B); engine 37.0 s (TRITON_ATTN); per arm 108 to 121 s, of which GSM8K
  91 to 102 s; STOCK8 cut by the short watchdog, as planned; results `results/e56-rehearsal-a100-20261104/`.
  VERIFY: (1) the engine arguments: PASS. (2) FAIL by the rule of this item: E2B ignores the closed thought channel and
  writes its reasoning in the answer ("The user wants me to find ..."), so 75% of the GSM8K answers stop at the 400-token
  limit (mean 379 tokens; 25% with '####') and all 32 needle answers stop at 24 tokens before the passphrase; the
  reference top-1 agreement is 0.700 (the alignment is right); the needle prompt log-probabilities pass the parser.
  (3) the batched calls reach every hooked layer with the expected tokens: PASS. (4) DET reproduces BF16 for the
  whole workload (0 answers with other ids): PASS. (5) V5 and V7 on the probe for every arm (Triton): PASS. (6) V8 in
  the REHEARSAL line (E2B: GSM8K 0.302, needle 0.000, reference NLL 3.09): PASS as a record. (7) the time per arm and
  the KV pool line in session.log: PASS. (8) the session files and the 7 code hashes: PASS. INTERPRETATION: the 31B
  answered directly in the E-55 chat-format GEN step, so its answers are shorter, but a short preamble would cut the
  needle answers at 24 tokens and fail V8; A2 raises both limits (a concise answer still stops at `<turn|>`).
- Balance floor: the protocol rule is the cap plus $35. If the live balance is lower, the user decides (as D-95).

## 10. Threats to validity
1. Batch dependence: vLLM kernels are not batch-invariant, so when one sequence diverges, the batch of the others
   changes. DET, BF16R and BF16C share this effect; H2 compares BF16C with BF16R.
2. Greedy near-ties: a tiny change can flip a token and the rest of the answer. BF16R measures how often.
3. The compact cache is emulated; a real compact layout reads the same values (E-52: the stock FA4 kernel reads the
   compact rows through two views with the same error), but a real write kernel is not tested.
4. The A100 runs the global layers on Triton, not FA4 (H100). V5 checks the engine output against the reference on
   the A100 itself.
5. The answer format: an answer without '####' falls back to the last number. The extraction is the same for every
   arm, so it does not bias the paired difference.
6. GSM8K may be in the training data. The comparison is paired, so this does not bias the difference.
7. The GSM8K prompts are short (about 300 tokens), where the global layers matter less; the needles test 7,700
   tokens. Contexts beyond 8K are not tested here (E-54 tested up to 32K at the attention level).
8. One checkpoint and one model family.
9. The tied 8-bit compact row (TIED8) is out of scope: its scales came from the FP8-dynamic checkpoint, so E-54 stays
   an attention-level result with the R-78 caveat.
10. The margins (3 points, 0.002 nats) are heuristics.

## 11. Files
| File | What it does | Tests |
|---|---|---|
| `prepare_e56.py` | builds `data_e56.json` on the Mac (downloads go to `data/`, which git ignores) | data test |
| `data_e56.json` | the 400 GSM8K items, 32 needles, 3 health slices and the tokenizer check (frozen) | data test |
| `run_e56.py` | the engine, the hooks (E-55), the pre-flight, the arms, the gate, the time cut; `--stub` | stub-session tests |
| `analyze_e56.py` | the grading, the paired statistics, V1 to V5, V7, V8, the frozen rule, the ledger manifest | rule, interval, extraction, gate, V2 tests |
| `session_e56.sh` | the Pod session (watchdogs, code hashes, download, main phase, analysis) | stub-session and limit tests |
| `make_manifest.py` | `code_manifest.json` from the design commit in git (L-25) | manifest test |
| `test_e56.py` | the tests | |
| `../gemma4_kv_model/splice_e55.py`, `run_e55.py`, `analyze_e55.py` | the hooks, the parser, the stub modules, the engine checks (imported; hashed in V2) | the E-55 tests |

## 12. Verification record (stage S3)
2026-11-04, the Mac (`uv run --no-project --with numpy --with torch --with pytest --with tokenizers`): 13 tests pass and
1 skips (the manifest test, before the design commit) in 153 s.
- The data file is the plan: the frozen hash; 400 GSM8K items with distinct rows and numeric golds; 32 needles with
  distinct passphrases, each found in its decoded prompt; every scoring sequence starts with its prompt and ends with
  `<turn|>`; every prompt fits the engine length with its answer.
- The extraction and the grading: the number after '####', the last number, the passphrase.
- The Agresti-Min interval covers 93% to 99% at the E-56 size for true differences of 0, -1 and -3 points (2,000 runs
  each); the ratio t interval covers 93% to 97%.
- The rule gives each outcome. The health gate stops a baseline with 50% GSM8K accuracy and one with an E-55-like
  reference NLL of 9 nats.
- The stub session (the real hooks in the engine's call shapes; L-26): GO1 GO2 GO3 with the default effects; every arm
  has its V5 gap in all 10 layers from the probe, and BF16C decides V7 in the 9 layers whose w is not a power of 2; DET
  is bit-identical; KILL1 with a planted BF16C loss of 8%; a STOCK8 failure is recorded and does not change the
  decision; a broken baseline stops after BF16 (exit 3, V8; L-29); an engine that ignores the hook and a DET rerun
  0.001 away stop in the pre-flight; a short watchdog cuts STOCK8 and keeps the decision; SIGTERM after the decision
  arms keeps the decision.
- V2 reads the engine record of this vLLM build ("0", "NONE"; L-28) and compares the code hashes; the ledger accepts
  the manifest.
- Defects found and fixed before the freeze: the stub gave DET different wrong answers from BF16 (it seeded them by the
  arm name); the stub was slow (one torch thread now; the arrays are saved without compression, also on the Pod);
  `analyze_e56.py` put the E-55 folder first on the import path, so a test imported E-55's `make_manifest.py`.
- Not run on the Mac: vLLM, the A100 and the BF16 31B (S5 and the run cover them).

2026-11-04, amendment A1, the Mac (same command): 15 tests pass in 68 s (the manifest test runs now).
- (B1) A healthy model with a reference NLL of 2.0 passes V8; an E-55-like 9 nats still fails it.
- (F2) The pre-flight check of the decision arms ignores a STOCK8 gap of 0.5; the STOCK8 check records it.
- (F1) One DET answer with other token ids is recorded (`same_ids` = answers - 1) and the decision stays GO1 GO2 GO3.
- (F5) The watchdog test cuts STOCK8 for certain (POD_LIMIT 70).
- (F7) "18" and "18.00" are the same answer; a narrow H3 interval in (0.002, 0.01] gives SMALL3.
- (O1) The stub session reports the long-context NLL change of BF16C, BF16R and STOCK8, the reference top-1
  agreement and the GSM8K answer statistics.
- Not testable on the Mac: the engine-start retry (F6) and the download loop (F3); the rehearsal and the run log them.

## 13. Result (S6 to S8)
Run (S6, 2026-11-04, D-98): 1 x A100 SXM 80GB, Pod `a18u5h10b2vxws`, Secure, US-MD-1, $1.79/h; the SSH port 271 s after
the create; setup 220 s (the 62.6 GB download in one try); engine 46.0 s (TRITON_ATTN; KV cache 31,762 tokens);
pre-flight 3.6 s, no problems; 195 to 207 s per arm; all 5 arms, no cut; peak memory 77.5 GB; session 1,293 s; about
27 min of Pod, about $0.81 (the live balance drop). Results: `results/e56-a100-20261104/` (37 checksums match).
- Decision (frozen rule, A1 and A2): **GO1 GO2 GO3**. FACT: the Mac recompute gives the same line.
  - H1: accuracy BF16 0.9769, BF16C 0.9722; the change -0.0046, 95% interval [-0.0124, +0.0032] (inside +/-0.03).
  - H2: answers that differ from BF16: BF16C 0.0046 (2 of 432), BF16R 0.0069 (3), STOCK8 0.0023 (1), DET 0; BF16C -
    BF16R -0.0023 [-0.0087, +0.0041] (the upper bound is below 0.03).
  - H3: the reference NLL change of BF16C -0.00091 [-0.00156, -0.00025] nats per token (inside +/-0.002); BF16R +0.00274
    [+0.00205, +0.00344]; STOCK8 -0.00085 [-0.00174, +0.00004]; the BF16 reference NLL 1.7526, top-1 agreement 0.776.
- Validity: every check passes. FACT: V8: GSM8K 0.975, needle 1.0, reference NLL 1.75; DET bit-identical (432 of 432
  answers; equal log-probabilities); V5 gaps 0.0018 or less for every arm on Triton; V7 passes (BF16C separations 0 in
  layer 29 and 0.0018 to 0.0104 elsewhere); the 7 code hashes match the A2 commit.
- Reported: GSM8K accuracy BF16 0.975, BF16C 0.970, BF16R 0.9725, STOCK8 0.9725; the needles 1.0 in every arm; answers
  with the same token ids: BF16C 0.785, BF16R 0.826, STOCK8 0.741; the GSM8K answers: mean 141 tokens, 0.3% cut at 640,
  99.8% with '####'; the long-context NLL change over the needle prompts from token 1,025: BF16C +0.0066 [+0.0005,
  +0.0127], BF16R +0.0074 [+0.0011, +0.0138], STOCK8 +0.0066 [-0.0022, +0.0154]; the accuracy change of BF16R -0.0023
  [-0.0113, +0.0067].
- Predictions (Section 2): GSM8K 0.975 (predicted 0.85 to 0.95: above); needle 1.0 (in range); reference NLL 1.75 (A1:
  0.5 to 2.5, in range); top-1 0.776 (0.4 or more: yes); the BF16C accuracy change -0.46 points (within +/-1: yes);
  answers that differ 0.46% and 0.69% (predicted 2% to 10%: lower); |Delta_BF16C| 0.0009 (predicted below 5e-4:
  higher, inside the GO3 margin); STOCK8 changes more than BF16C (predicted: not seen; both are at the noise level).
- Exploratory (the R-78 diagnostic): the raw-text NLL of the BF16 31B on the health slices is B1 10.79, P1 7.91 and C1
  5.19 nats per token, as poor as the FP8-dynamic checkpoint in E-55 on the same first tokens. INTERPRETATION: the E-55
  degradation is not caused by the FP8-dynamic checkpoint or by FA4 (E-56 uses BF16 weights and Triton); it comes with
  raw text without a chat template in this engine, while the chat-format scoring is healthy (reference NLL 1.75, top-1
  0.776). The cause is not proven: the instruction-tuned model on raw text, or a raw-text path of vLLM.
- INTERPRETATION: with the 16-bit weights, the compact K=V cache of the 10 global layers keeps the task accuracy (the
  95% interval of the change is [-1.24, +0.32] points) and changes the answers no more often than a rounding-size change
  of the query; its reference-NLL change is smaller than that of the noise arm. The memory saving (37.5% of the
  global-layer KV bytes) is arithmetic, because the cache was emulated. Next: S9 (R-80, the post-run review).

## Amendments
A1 (2026-11-04, after R-79 AMEND_THEN_RUN, before any E-56 data; `dossier/REVIEW_E-56_PRERUN_2026-11-04.md`). The
author checked each finding against the code and the vLLM source notes before the fix. H1 and H2 are unchanged.
1. B1: the V8 reference-NLL bound 1.5 -> 3.0; the reference top-1 agreement is recorded; the prediction is now 0.5 to
   2.5 (Sections 2, 5 and 6).
2. F1: the DET answer token ids are recorded, not a gate; the DET log-probability gate stays (Section 6).
3. F2: a STOCK8 problem in the pre-flight is recorded and skips STOCK8; it does not stop the run (Section 6).
4. F3: the download has up to 3 resumed tries in 1,500 s; the watchdogs move to 4,200 s and 4,380 s and the SSH
   limit to 600 s, so the worst case stays $2.48; the Pod volume is 120 GB (Section 9).
5. F4: 400 GSM8K tokens (320 before); the cut share, the '####' share and the mean length are recorded; the rehearsal
   VERIFY (2) is quantitative (Sections 4.2, 5 and 9).
6. F5: the watchdog test uses POD_LIMIT 70, so the cut does not depend on the speed of the machine (Section 12).
7. F6: the engine lines "GPU KV cache size" and "Maximum concurrency" go to session.log; one engine-start retry
   (utilization 0.88 after a CUDA out-of-memory error, 0.95 after another start error; recorded).
8. F7: the answer comparison of H2 is numeric; SMALL3 names a narrow H3 interval in (0.002, 0.01] (Section 7).
9. O1: the needle calls return the prompt log-probabilities; the long-context NLL change from token 1,025 is reported
   (Section 5). O3: the reading of an INCONCLUSIVE result (Section 7).
10. Point 9: the text of Sections 1, 2 and 4.4 now matches the code: the K view reads V on the unrotated dims and w
    goes into the query.
Not taken: O2 (800 GSM8K items: a new data file; R-79 does not recommend it for this run).

A2 (2026-11-04, after the S5 rehearsal, before any 31B data; a fix that the rehearsal required by VERIFY (2)): the
answer limits 400 -> 640 tokens (GSM8K) and 24 -> 128 tokens (needles). Nothing else changes; no second rehearsal
(the change has no new code path). If the 31B writes long answers, GSM8K takes up to about 6 min per arm, and the
watchdog cuts STOCK8 first.
