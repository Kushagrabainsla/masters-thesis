# R-76: post-run review of E-54 (accuracy of the scale-tied compact FP8 K=V cache on fresh Gemma 4 31B activations)

VERDICT: CONFIRM_WITH_CORRECTIONS (the frozen PASS reproduces from my own code in all 70 cells to 2.4e-4, and every validity and process check holds; of eight corrections, six limit what the PASS may claim, one adds a process record and one is minor)

Date: 2026-11-02.
Reviewer: R-76 (Sonnet 5.5, a different model from the author).
Stage: S9, hostile post-run review of the E-54 H100 run of 2026-11-02 (Pod ux88cwtt24olfp, US-GA-2, frozen decision PASS).
Labels: FACT means that I read a file or a source, ran code or computed a number. INTERPRETATION means my reasoning. HYPOTHESIS means not tested. POST-HOC means that I computed it after the frozen decision, on the E-54 data. It is exploratory and not a test.

Rules kept:
- I wrote this one repo file and no other repo file.
- I did not read `.env`. I made no Runpod, cloud or GPU call. I did not commit, push or post.
- I made read-only `gh api` GET calls on vllm-project/vllm at 3966924d5 (a file list and three source files, saved in the scratchpad only).
- I did not read the author's Mac recompute folder `mac_recompute/`. My recompute uses none of the author's analysis code.
- I ran the author's test file once from the scratchpad with bytecode writing off. It gave 17 passed in 54 s. It left no cache folder in the experiment folders.

Scratch folder: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/20255ed9-038a-44e5-8a60-f7885a8c54ee/scratchpad/r76/`.
It holds my library `r76lib.py` (E4M3 and E5M2 tables, cast, attention, tied row), the scripts that I name below and their logs and outputs.

## Key findings
1. FACT: My own code reproduces the whole E-54 metrics file. The code has its own E4M3 table with explicit uint8 codes. It has its own causal GQA attention in float32. It builds the tied row as 640 explicit codes. It covers all 70 cells (7 prompts, 10 layers) and 14 arms and forms.
2. FACT: The largest relative difference to `metrics.json` is 2.4e-4 over the 420 numbers of the three decision arms (STOCK, CALSTD_JOINT, TIED, float32 and FP16). The median is 3.7e-6. The other arms differ by 1.2e-4 or less. The pooled statistics differ by 4.1e-6 or less.
3. FACT: My own implementation of the Section 5 statistics and the Section 7 rule gives R1 0.6705 and R2 1.1308 (FP16 the same). It gives PASS. The FAIL condition does not hold.
4. FACT: V1 to V7 hold with my own numbers, and no check passes by mistake. The identity error is at most 0.0028 on each of the 384 unrotated dims and at least 1.19 on each of the 128 rotated dims. The V5 gap is 0.00157 to 0.00174, and a wrong head map or a wrong scale gives 1.0 to 1.15.
5. FACT: All 71 capture files match the SHA-256 list of the Pod. The 70 per-file hashes in `capture_run.json` match too. No prompt takes an E-53 file by path or by hash. The chat prompts have 9 special ids after BOS. Prompt 6 ran in 4 chunks in every layer.
6. FACT: The code, the calibration, the kernel arm, the session script, the E-53 list and the scales file are byte-identical at the A2 commit 54814ca, at HEAD 2ba042c and in the working tree. Only the README changed after A2 (the status line and Section 13). The run used the frozen scales (SHA-256 8efe6e90...).
7. FACT: R2 does not test transfer. Under random phases for both arms, R2 has a median of 1.14 (5% to 95%: 0.89 to 1.46) in my null from the frozen curves. The observed R2 1.131 sits at the 49th percentile of that null.
8. POST-HOC: The transfer is real, and R1 shows it. R1 0.670 is below all 100,000 null draws (null 5% to 95%: 0.99 to 1.41). On prompt 0 the frozen phase is the best of 32 phases in 9 of 10 layers.
9. FACT: Part of R1 is one vector. K[0] and V[0] are bit-identical in all 7 prompts and all 10 layers. With position 0 exact in every arm, R1 is 0.80 and R2 is 1.13 (the same as with position 0 rounded). In layers 5, 11 and 23 the tied row is no better than the stock row once position 0 is exact (1.00, 1.36, 1.13).
10. POST-HOC: The gain needs the scales to about 0.25%, and not bit-exact. Rounding the scales to BF16 or FP16 costs 0.9% to 1.2% of the pooled error. A scale offset of 0.5% costs 6% to 12%, and 1% costs 19% to 24%. One phase step (2.2%) gives R2 1.64 to 1.75, which is in the FAIL band.
11. FACT: Context length changes the picture. The C0 row fails layer 41 at 2.2 to 2.6 times stock in the six 8K prompts, and at 1.08 times in the 32K prompt. The 32K prompt has the tied row worse than stock in layers 11 (1.49) and 29 (1.10).
12. FACT: The kernel arm gap is 0.00157 to 0.00170 in all 60 arm-cells. INTERPRETATION: this is the BF16 rounding floor of the kernel output. FACT: the arm tests a decode-style call (query length 1) on a hand-built cache. The engine prefill calls the same function with query length above 1.

## Corrections

C1. The words "accuracy-neutral" and "at the level of the stock FP8 cache" are not exact.
- Where: `RESEARCH_JOURNAL.md` Section 5.27 (near line 4009: "The FP8 compact row is accuracy-neutral at the attention level"); `README.md` Section 13, the last INTERPRETATION bullet.
- Failure: the tied row is not neutral. FACT: it has 0.670 times the pooled stock error (0.80 with position 0 exact), 1.131 times the error of the best calibrated standard cache (1.02 to 1.42 by layer, up to 1.58 in one cell), and a pooled error of 0.029 against the BF16 cache.
- Fix: state these three numbers and delete "accuracy-neutral".

C2. "The tied row beats the stock cache in every layer" is true only for the per-layer medians.
- Where: `README.md` Section 13 (Guards bullet); `RESEARCH_JOURNAL.md` Section 1 (near line 317), the Section 4 log (near line 1873) and Section 5.27 (near line 3964): "beats the stock cache in all 10 layers".
- Failure: FACT: the tied row has a larger error than the stock cache in 4 of 70 cells: prompt 1 layer 41 (1.003), prompt 4 layer 35 (1.107), prompt 6 layer 11 (1.490) and prompt 6 layer 29 (1.097). It beats the stock cache in 66 of 70 cells. It beats the joint baseline in only 15 of 70 cells.
- Fix: write "in every layer by the median over prompts, and in 66 of 70 cells".

C3. The kernel arm sentence has no scope.
- Where: `README.md` Section 13 ("with no kernel change"); `RESEARCH_JOURNAL.md` Section 1 (near line 318) and Section 5.27 (near line 3965): "The stock FA4 kernel reads it".
- Failure: the arm covers 20 cells of prompts 0 and 6. Each cell is 256 decode sequences of query length 1 on shared pages, with the same descales for every sequence and head. The caches are built by the test script, not by the engine write path.
- Fix: add that scope. Add that the engine prefill calls the same FA4 function with query length above 1, and that this form is not tested.

C4. "A build must store and use these float32 scales exactly" is too strong and has no number.
- Where: `README.md` Section 13, INTERPRETATION bullet.
- Failure: POST-HOC data (4 prompts, 40 cells, answer 4 below) give a tolerance. BF16 or FP16 storage of vs and ks costs about 1% of the pooled error. An offset of 0.25% costs about 6%. An offset of 0.5% costs 6% to 12%. An offset of 1% costs 19% to 24%.
- Fix: write "reproduce vs and ks to 0.1% (0.25% at most); BF16 storage is acceptable". Cite this review as the source.

C5. The coverage wording hides the sample sizes.
- Where: `README.md` Section 13 ("7 fresh prompts (plain, chat and 32K)" and "the E-53 FAIL repeats on fresh prompts"); `RESEARCH_JOURNAL.md` Section 5.27 (near line 3964: "on all 7 prompts (2 chat, 1 of 32K)").
- Failure: FACT: the 32K class has one prompt, and the chat class has two. All text is Python code and Markdown or rst files from the image. The last turn marker of chat prompt 4 lies 1,069 tokens before the first query row, and that of prompt 5 lies 409 tokens before it. No prompt is shorter than 8K or longer than 32K. The C0 failure at layer 41 holds in the six 8K prompts (2.2 to 2.6) and not in the 32K prompt (1.08).
- Fix: write "technical text and code, 8K to 32K tokens; one prompt at 32K". Say that the C0 replication holds at 8K.

C6. The rule text says that a PASS needs transfer. The pooled bounds do not test it.
- Where: `README.md` Section 7 ("So a PASS needs transfer"); Section 5.
- Failure: R-75 gave PASS in 30% of no-transfer draws with all guards. My null from the frozen curves gives 62% for the two pooled bounds alone. FACT: the observed R2 (1.131, with transfer) equals the no-transfer median (1.14).
- Fix: say that the evidence of transfer is R1 (0.670 against a null of 0.99 to 1.41), JOINT/STOCK (0.593 against a null median of 1.03) and the phase sweep (answer 4). Do not cite the R2 band as a transfer test.

C7. The session does not record code hashes.
- Where: `session_e54.sh`; `analyze_e54.v2_problems`.
- Failure: V2 checks the SHA-256 of the scales file. It does not check the code that ran on the Pod. I could prove the integrity only by git and by my recompute.
- Fix: add `sha_list "$CODE" > "$RES/code_sha.txt"` to the session script before the capture. Make V2 compare the hashes of the 6 code files with the design commit.

C8 (minor). The session start time differs by 15 s between records.
- Where: `README.md` Section 13 and `HARDWARE_LOG.md` give 21:48:08 UTC. The folder `env-20261102T214753Z` and `session.log` (setup 82 s before 21:49:15) give 21:47:53 UTC.
- Fix: use 21:47:53 UTC.

## Answers to the checks

### 1. Independent recompute
- FACT: Method. `r76lib.py` builds the E4M3 table from the bit layout (127 finite codes, maximum 448). The cast saturates at 448 and rounds to nearest even by code parity. It encodes to uint8 codes and decodes with a 256-entry table.
- FACT: Cast test (`unit_cast.py`). My cast has 0 mismatches against `torch.float8_e4m3fn` on 2,466,036 values. The values include all finite BF16 patterns and all midpoints of the table with their float32 neighbours. My E5M2 cast has 0 mismatches too. A second cast by float32 bit rounding also agrees.
- FACT: Attention. Query head h reads KV head h // 8. The scores are `(q . K) * ks`. The output is `(P . V) * vs`. The descales come after the dot product, as the FA4 kernel does. The scale is 1.0 and the mask is causal.
- FACT: Tied row. The row holds the codes of `[K_rot / ks | V_unrot / vs | V_rot / vs]`, 640 per token and KV head. The K view is codes 0 to 511. The V view is codes 128 to 639. The query is the plain permuted `[q_rot | q_unrot]`.
- FACT: Scales. I read `scales_e54.json` and checked its SHA-256 (8efe6e90..., frozen). I read `w` from `k_norm.pt`. In all 70 cells `w` is one BF16 value on the unrotated dims, equals the scales file, and `ks = float32(w * vs)` holds.
- FACT: Coverage. All 70 cells. Main job: STOCK, CALSTD_JOINT and TIED, float32 and FP16 forms, V5 and V3 (277 s on 5 threads). Extra job: CALSTD, TIEDSTD, TIED_E5M2, TIED_UP, TIED_DN, C0, BF16C and the position-0-exact forms of four arms (about 900 s).

| Arm and form | Largest relative difference to `metrics.json` | Cell |
|---|---|---|
| STOCK float32 and FP16 | 4.1e-5 and 7.9e-5 | prompt 6 layer 5, layer 35 |
| CALSTD_JOINT float32 and FP16 | 1.6e-4 and 2.4e-4 | prompt 6 layer 5 |
| TIED float32 and FP16 | 8.8e-5 and 9.2e-5 | prompt 6 layer 5, layer 35 |
| CALSTD, TIEDSTD, TIED_E5M2 | 1.2e-4, 9.5e-5, 4.1e-5 | prompt 6 |
| TIED_UP, TIED_DN, C0, BF16C | 6.3e-5, 3.6e-5, 1.1e-5, 3.6e-5 | prompt 6 |
| position 0 exact (4 arms) | 4.4e-5 or less | prompt 6 |
| engine gap (V5) | 6.5e-4 | prompt 6 layer 41 |

- INTERPRETATION: The five largest differences are all in the 32K prompt. The cause is probably the float32 sum order over 32K keys (CPU against GPU). It is not a fault. I did not test this.
- FACT: BF16C in layer 29 is 3e-7 in both implementations (w is 2^-4, so the BF16 query is exact). The relative difference there is noise on a near-zero value. I compare it by absolute value.
- FACT: FP16 form. All 70 cells have some q values that FP16 cannot hold exactly. INTERPRETATION: they are tiny BF16 values. My bulk FP16 numbers reuse the float32 scores. I re-ran four cells with q rounded to FP16 and P rounded to FP16. They agree with the author to 2.4e-4 or better.
- FACT: Pooled statistics (my own code on my own errors, and on `metrics.json`; both give the same numbers):

| Statistic | Value | Bound |
|---|---|---|
| P_STOCK, P_CALSTD_JOINT, P_TIED | 0.04316, 0.02559, 0.02894 | none |
| R1 and R1_fp16 | 0.6705 and 0.6705 | at most 1.25 (FAIL above 1.5) |
| R2 and R2_fp16 | 1.1308 and 1.1308 | at most 1.25 (FAIL above 1.5) |
| per-layer g1 (TIED/STOCK) | 0.362 to 0.944 (layer 41) | at most 2 |
| per-layer g2 (TIED/CALSTD_JOINT) | 0.977 to 1.417 (layer 17) | at most 2 |
| per-prompt R1(p) | 0.606 to 0.721 | at most 1.5 |
| per-prompt R2(p) | 1.083 to 1.2506 (prompt 3) | at most 1.5 |

- FACT: The decision PASS follows. E5M2 over TIED is 3.2464 (V6 needs 1.5 or more). TIED over TIEDSTD is 0.9868 (V7 needs 0.90 to 1.10).

### 2. Validity
- FACT: V1. The GPU is "NVIDIA H100 80GB HBM3". V2 has 0 problems. I re-derived its parts below.
- FACT: V3. My identity error is at most 0.00242 (the author: 0.002417). Per dim, all 384 unrotated dims are at most 0.0028 in all 10 layers (prompts 0 and 6). All 128 rotated dims are at least 1.19. Position 0 has `K0 = w V0` on all 512 dims (error 0.0033 in layer 41), as RoPE angle 0 requires. So the rotated set `0 to 63` and `256 to 319` is right.
- FACT: V4. The reference is finite in all 70 cells.
- FACT: V5. My reference gives a gap of 0.00157 to 0.00174 against the engine output. A wrong head map (h mod 4) gives 1.0017. The scale 1/sqrt(512) gives 1.146. So V5 would catch a wrong reference, and the hooks work over 4 chunks (prompt 6 gaps: 0.00162 to 0.00170).
- FACT: V6 and V7 hold (see answer 1). No value saturates (0 in all 70 cells for STOCK, CALSTD_JOINT and TIED, with my own count).
- FACT: SHA-256. I hashed all 71 files in `capture/`. They match `e54/capture_sha.txt` (same 71 names). The 70 prompt-layer hashes in `capture_run.json` match too. The Pod computed those before the analysis.
- FACT: Prompts (`check_prompts.py`, from the token ids in `prompts.json`). All 7 lengths match the plan. BOS is at position 0 in all. The token hash of every prompt matches the record. Plain prompts have 0 special ids after position 0. Chat prompts 4 and 5 have 9 each (id 105 five times, id 106 four times). No marker lies inside the last 256 positions. No file is used twice. No file is on the E-53 list (17 paths and 17 hashes, by both).
- FACT: Engine record. Eager (`compilation_mode` and `cudagraph_mode` NONE). `cache_dtype` auto with a BF16 model, so a BF16 cache. vLLM 0.33.0rc1.dev229+g3966924d5. `max_model_len` 33,024. 8,448 batched tokens. Chunked prefill on. The 10 hooked layers are the 10 global layers, with no sliding window. `chunks` is 1 for prompts 0 to 5 and 4 for prompt 6 in every layer. The image digest is the E-53 one (ledger record).
- FACT: Prompt 6 chunks. `kpos` is 0 to 32767 once each in all 10 layers, so K and V cover the 4 chunks in order. INTERPRETATION: each chunk has 8,448 tokens or fewer, so the last 256 positions fall in the last chunk.
- FACT: Tests. `test_e54.py` gives 17 passed (54 s) on the current tree.
- INTERPRETATION: The checks have the right power. V2, V3 and V5 would fail on the faults that the tests plant. I found no vacuous check.

### 3. Process
- FACT: Git (read-only). Commits: design 05cc73e (2026-11-01 20:54 PST), A1 cd6ec05 (2026-11-02 10:36), A2 54814ca (11:03), attempt-1 record d36b1c2 (12:32), results 2ba042c (14:24). The Pod session ran 21:47 to 21:53 UTC, which is 13:47 to 13:53 PST.
- FACT: Blob check. `analyze_e54.py`, `calibrate_e54.py`, `capture_e54.py`, `kernel_e54.py`, `session_e54.sh`, `test_e54.py`, `scales_e54.json` and `e53_files.json` have the same git blob id at 54814ca and in the working tree. `analyze_e53.py`, `capture_e53.py` and `compact_attn.py` are the same too. `git diff --stat 54814ca HEAD` over the three code folders shows only `README.md` (+37, -1: the status line and Section 13). File times of the code files are all before 11:01 PST on 2026-11-02.
- FACT: Scales lineage. The design-commit file has the SHA-256 that the README names (202738b9...). Its TIED fields, CALSTD fields and `w` are identical to the A1 file. The A1 file adds CALSTD_JOINT. The file has the frozen hash 8efe6e90... today. `metrics.json` embeds exactly the `layers` of that file, and `summary.json` records `_scales_sha256` 8efe6e90....
- FACT: Order of events. The scales were frozen at A1, 3 h 11 min before the session. No code commit lies between A2 and the run. The ledger record names design commit 54814ca.
- FACT: Waivers, all logged as user decisions. D-92: the S5 rehearsal was waived, and the A2 prompt check replaced it. D-92 and D-93 lowered the balance floor to $9 and then $6 (the protocol abort rule 5 says the cap plus $35). Attempt 1 lost $2.71 (L-24). The E-54 total is about $4.10, above the single $4 cap, and the user approved each attempt.
- INTERPRETATION: The waivers did not touch the data. They raised the chance of a lost run, and they did lose one.
- FACT: The Pod-side code hashes are not recorded (C7). The indirect evidence is git plus my recompute, which equals `metrics.json`.

### 4. Interpretation
What the PASS supports:
- FACT: On 7 fresh prompts, the attention-output error of the 10 global layers with the tied FP8 row is 0.670 times the stock FP8 error and 1.131 times the error of the best jointly calibrated standard FP8 cache.
- FACT: The pooled bootstrap over prompts (20,000 draws, POST-HOC) gives R1 0.611 to 0.713 and R2 1.083 to 1.177 (95%). P(R2 above 1.25) is 1.1%. Leave-one-prompt-out gives R2 1.114 to 1.140 and R1 0.656 to 0.672.
- POST-HOC: The calibration transfers. On prompts 0 and 4 I computed the tied error at all 32 phases (640 passes). The frozen phase is the best of 32 in 9 of 10 layers on prompt 0 and in 3 of 10 on prompt 4, and it is in the best 5 in all layers. The error of the frozen phase over the best phase has a median of 1.000 and a maximum of 1.313 (layer 23, prompt 4). The pooled ratio is 1.013 (prompt 0) and 1.087 (prompt 4).
- POST-HOC: Without transfer the numbers would differ. In my null (random phases from the frozen in-sample curves, 100,000 draws), P_TIED/P_STOCK has 5% to 95% of 0.987 to 1.407 and P_JOINT/P_STOCK has 0.872 to 1.229. The observed values are 0.670 and 0.593. Both lie below every draw.
- FACT: The stock FA4 kernel read the tied row with `k_descale` not equal to `v_descale` in 20 cells (and the calibrated standard cache in 20 more). The TIED gaps are 0.00158 to 0.00170, at the BF16 floor. The kernel error against the reference is within 0.8% of the emulation error in every arm-cell.
- FACT: The E-53 layer structure replicates. C0 over STOCK per layer is 0.95, 1.18, 1.21, 0.55, 1.00, 1.06, 2.39, 1.29, 0.89 and 0.82. E-53 gave 0.93, 1.23, 1.19, 0.57, 1.01, 1.19, 2.38, 1.33, 0.98 and 0.82.

What the PASS does not support:
- FACT: Model quality. No model-level test exists. The test uses the K and V of a BF16-cache run, so each layer sees BF16 attention before it. A real FP8 cache adds the errors of all earlier layers. E-55 is the test.
- FACT: The tie costs error. The tied row has 1.13 times the error of the best standard cache. The cost is 1.42 times in layer 17, and 1.58 times in the worst cell (prompt 6, layer 17). Three cells exceed 1.5 (prompt 1 layer 17, prompt 3 layer 47, prompt 6 layer 17). None exceeds 2. The worst-head view gives R1 0.54 and R2 1.05, so it tells the same story.
- FACT: The largest per-prompt R2 is 1.2506 (prompt 3, mixed 8K). It is 0.0006 above the pooled bound 1.25. The frozen rule uses 1.5 for single prompts, so it passes. The bootstrap gives P(pooled R2 above 1.25) of 1.1%.
- FACT: The largest per-layer g2 is 1.417 (layer 17). The in-sample ratio of the README table (Section 4.6) is 1.376 there. The in-sample ratios predict g2 within 0.15 in 9 of 10 layers (layer 47: 1.387 in sample, 1.160 out of sample).
- INTERPRETATION: So the layer 17 cost is a structural price of the single scale, and not a loss of transfer.
- INTERPRETATION: The cost comes from the value path. The score-path error of TIED (V exact) is 0.0232, below JOINT (0.0248). The tied row has one free scale, and the standard cache has two.
- FACT: The BOS vector carries part of R1 (key finding 9). A stock cache with only position 0 exact has a pooled error of 0.0330, or 0.765 times the stock error. The tied row (0.0289) is 0.877 times that. In layers 5, 11 and 23 the advantage is gone when position 0 is exact in both arms. The attention mass on key 0 in the 8K prompt is 0.18, 0.15, 0.19 and 0.15 in layers 5, 11, 17 and 23, and 0.02 to 0.03 in layers 35 to 53. It falls to about half at 32K.
- INTERPRETATION: The BOS part is stable, because the vector is bit-identical in every prompt of this model. It is a property of one vector. It says nothing about the other 8,191 positions.
- FACT: Fragility. The tied row one phase step up or down gives R1 0.97 and 1.04 and R2 1.64 and 1.75. In 8 of 10 layers at least one neighbour phase has 1.3 times the error or more, and in 6 of 10 layers both have. Layers 17 and 47 are flat (1.0 to 1.2 on both sides). The curve over the 32 phases has a range of 2.1 to 2.3 times in layer 17 and 4.5 times in layer 41.
- POST-HOC: Sub-step tolerance (`sens_scale.py`, prompts 0, 2, 4 and 5; baseline R1 0.686 and R2 1.116 on these prompts):

| Scale offset of vs (ks = w vs) | Pooled error against the frozen scales | R2 on these prompts |
|---|---|---|
| minus 1% and plus 1% | +23.9% and +18.6% | 1.38 and 1.32 |
| minus 0.5% and plus 0.5% | +5.8% and +11.5% | 1.18 and 1.24 |
| minus 0.25% and plus 0.25% | +5.2% and +6.4% | 1.17 and 1.19 |
| minus 0.1% and plus 0.1% | +3.9% and minus 0.5% | 1.16 and 1.11 |
| vs and ks rounded to BF16 | +0.9% | 1.127 |
| vs and ks rounded to FP16 | +1.2% | 1.130 |

- INTERPRETATION: The minima are sharp at the 0.5% to 1% scale. A 0.04% shift (FP16 rounding) still costs 1%. So the frozen scales carry a selection effect of about 1% to 4%, and the advantage over the best neighbour is not exactly stable.
- FACT: C0 replication. See C5. The E-53 failure of the untuned row repeats in all six 8K prompts. It does not repeat at 32K (1.08).
- FACT: BF16C. The BF16 compact row has a pooled error of 0.0054, which is 5.3 times below TIED. It needs 1,280 bytes per token and head against 640 for TIED and 1,024 for the stock FP8 cache. It stays the fallback of D-90 option (b).
- FACT: Scope of the kernel arm. See C3. FACT from source (vLLM 3966924d5, `flash_attn.py`): the engine always reads K and V from the paged cache, in prefill too (`k=key_cache, v=value_cache`). It passes `k_descale = layer._k_scale` and `v_descale = layer._v_scale`, expanded to `[sequences, KV heads]`.
- HYPOTHESIS: A model-level test at lengths of 64K and above would change the per-layer picture, because the C0 failure at layer 41 changes between 8K and 32K. No data exists above 32K.
- HYPOTHESIS: A short context (under 2K) puts more attention mass on the BOS key. The tied row would then look better against the stock cache. No data exists below 8K.

### 5. Corrections and next step
Corrections: C1 to C8 above. None changes the frozen decision. The user decides what happens next (journal Section 10).

Inputs for E-55 (the model-level test). All items are my suggestions (HYPOTHESIS), and the author sets the thresholds before the run.
1. Arms: BF16 cache; stock FP8 in all layers; TIED in the 10 global layers (frozen scales) with stock FP8 in the 50 sliding layers; CALSTD_JOINT in the 10 global layers with stock in the sliding layers. Add two controls that must degrade: TIED_E5M2 and TIED one phase step off.
2. Splice point: replace K and V in a pre-hook on each global `Attention` module by their dequantized cache values, in prefill and in decode. FACT: the engine reads the cache in both. A splice that quantizes only at decode would test the wrong thing.
3. Controls: with an identity quantizer, the splice must give the BF16 engine output bit for bit. On the E-54 captures, it must give `err_TIED` to 1e-4.
4. Metrics: per-token NLL (teacher forced) and KL(BF16 to arm) at 2K, 8K, 32K and 64K. Add top-1 agreement, one generation set with a checkable answer, and one long-context retrieval set. Use a paired bootstrap over documents. Write the minimum effect before the run, for example TIED excess NLL at most 1.25 times the stock excess.
5. Data: new files that are on no E-53 or E-54 list. Add prose and a non-code domain. Add chat prompts that end with the generation prompt, so that the query rows sit next to a turn marker. Use at least 3 prompts at 32K and one at 64K.
6. Cost and rehearsal: the E2B model can test the hook mechanics. It has no K=V, so it cannot test the tie. The H100 session is about 30 to 40 min.
7. Free offline work first, on the Mac: a phase grid with 4 times the resolution on the E-53 captures (my data show a gain of 0.5% at plus 0.1%); a calibration on E-53 plus E-54 captures (11 prompts, one at 32K) for a second-generation scale set, tested only on E-55 prompts; the three-scale row of R-75 O4; and a stock cache with only position 0 in BF16 as a cheap competitor.

Inputs for an engine build (FACT from the pinned source unless marked):
1. Scales. Use the per-layer float32 `_k_scale` (= ks) and `_v_scale` (= vs). The backend expands them to the `[sequences, KV heads]` descales already. So the read side needs no new plumbing.
2. Reproduce the 20 numbers. Key them by checkpoint revision d4ab4f57 and layer. Check `w` against the file at load, and refuse another revision. Keep vs and ks within 0.1% (0.25% at most). BF16 storage is acceptable (POST-HOC).
3. Cast. The pinned write path divides by the scale (`__bfloat162float(a) / scale`, then `__NV_SATFINITE` E4M3, `quant_utils.cuh` lines 232 to 241). Keep division in the new kernel. A reciprocal multiply may flip rare codes at ties (HYPOTHESIS: no visible effect).
4. Write kernel. A new kernel must write `[K_rot / ks | V_unrot / vs | V_rot / vs]` in one 640-wide row. `reshape_and_cache_flash` writes K and V with one scale each ([1] or [heads]) into the `(B, H, N, 2*D)` layout (`cache_kernels.cu` lines 320 to 410, `flash_attn.py` line 1448). The K_unrot values are never written.
5. Cache spec and page size. The row is 640 bytes per token and head. The hybrid KV manager must group the global layers apart from the sliding layers (R-72 flagged the page-size risk). FA4 on SM90 with an FP8 cache needs a page of 64 tokens for head size 512 (`flash_attn.py` lines 300 to 325).
6. Query and output order. The kernel needs `[q_rot | q_unrot]`. The output comes in V-view order. Fold both permutations into the QK-norm and RoPE write and into the `o_proj` columns.
7. Prefill. Test the varlen call with query length above 1, causal masks, chunk boundaries, chunked prefill and prefix caching. The kernel arm covered only query length 1.
8. Hardware and graphs. Keep the scope to SM90. The backend comment (`flash_attn.py` lines 1153 to 1156) says that SM100 needs an FP8 query. The capture ran eager, so test CUDA graph capture.
9. Acceptance test. Run the engine with the tied cache on the 7 E-54 prompts. The per-cell error against the BF16 reference must match the emulation to the kernel floor (gap 0.0017) and give R1 and R2 inside the E-54 bands.
10. Gate. Build work that is specific to the FP8 tied row (the write kernel, the cache spec) should wait for E-55 or carry a stop rule. The dtype-neutral parts can go on in parallel (D-90).

## What I could not check
- I ran no GPU code and no vLLM. I did not run the FA4 kernel. My statements on the kernel arm rest on `kernel.json` and the vLLM source.
- I could not read the code that ran on the Pod. I rely on git, file times and my recompute (C7).
- I did not recompute the R-75 Monte Carlo. My null uses the frozen in-sample curves of the E-53 prompts as a proxy for the errors on new prompts. It is a proxy, not a test.
- The author's GPU code divides by a Python float. PyTorch may implement that as a reciprocal multiply on the GPU. My CPU code divides exactly. The agreement of 2.4e-4 bounds the effect. I did not isolate it.
- The sub-step tolerance and the phase sweep use 4 prompts and 2 prompts at 8K. They are POST-HOC and exploratory. They do not cover 32K.
- I made no model-level test, and I did not test the cache write path, decode with a cached FP8 prompt, prefix caching or CUDA graphs.
- I did not re-read the checkpoint config for `kv_cache_scheme`. I rely on E-50 and R-74.
- I did not read `MEASUREMENT.md` or `RUNPOD.md`. I read `PROTOCOL.md` for S9 and the abort rules only.
