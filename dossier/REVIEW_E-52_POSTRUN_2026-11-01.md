# R-72: post-run review of E-52 (fixed compact K=V decode kernels and the views layout, I-182)

VERDICT: CONFIRM_WITH_CORRECTIONS (every number recomputes; INVALID (V4) and the exploratory KILL (byte route only) stand; the "80% on-chip", H4, H5, "24% to 25%" and SASS statements need correction)

Date: 2026-11-01. Reviewer: R-72 (a different model from the author).
Labels: FACT (I read a file, ran code or recomputed a number), INTERPRETATION (my reasoning), HYPOTHESIS (not tested).
I changed no repo file except this one. I did not read `.env`. I made no Runpod, cloud or web call. I posted nothing.
I ran read-only `git log` and `git diff --stat`. I ran no GPU code. I did not run `test_e52.py` (it writes files into the repo).
Scratch scripts (they do not import the author's code): `r72_common.py`, `r72_recompute.py` and `r72_part2.py` to `r72_part12.py` in `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/`.
Data: `experiments/gemma4_kv_views/results/e52-h100-20261101/e52/` and the 4090 rehearsal. Code: the E-52 and E-51 scripts at 3130fcd equal those at dfd99b9 (only the README changed). The 315 local and remote checksums match.
A local vLLM clone (commit 24e80c2, 2026-10-29, in the scratchpad `repos/vllm`) served the page-size facts. The E-52 image is 3966924d5. The two commits can differ.

## Key findings
1. FACT: My script, `summary.json` and README Section 13 agree on every number I checked. INVALID (V4) is correct. V1, V2, V3, V5 and V6 pass.
2. FACT: The word KILL (byte route only) holds on the medians, in 5 of 5 repeats and with the E-51 stock baseline. sigma_layout is 0.042 and 0.065 against the lines 0.155 and 0.125. The tightest margin is e_h at 64K: the highest repeat is 0.272 against the line 0.333.
3. FACT: The V4 failure is a row-order effect under the power cap. The three slow repeats are the three forward-order repeats. The failure moves dt by 0.09% at most and sigma_layout by 0.001. No reasonable rule on this data gives another word.
4. INTERPRETATION: "About 80% on-chip work" is too short. The DRAM-free floor is 80% to 83% of dt. The DRAM time at the copy rate is also 87% to 90% of dt. The kernel has two floors that overlap in part. That explains why 37.5% fewer bytes give 4% to 10%.
5. FACT: The data argue against an L2-to-SM bandwidth limit. They rule out the merge kernel and the split count as the cause of the floor. They do not separate latency, occupancy and instruction issue. The H5 test (`num_stages` 3) added no buffer, so it did not test latency.
6. FACT: The Triton views kernel gets 52% (32K) and 73% (64K) of the gain that its own K = V oracle predicts. The stock FA4 kernel gets 114% and 96%. The stock kernel reads the views exactly and 6.6% to 9.6% faster. Those views rows ran at a higher SM clock.
7. FACT: H4 is false as preregistered. FP16 compute saves 14 to 17 SM-clocks per token and KV head (12% to 14% of dt). The cast does not set "most of the time".
8. FACT: "24% to 25% faster than stock" holds only against this session's stock rows, which ran 7% to 12% below the Triton SM clock. Against the E-51 and E-50 baselines the gain is 19% to 23%.
9. FACT: The SASS counts do not exist. `nvdisasm` is missing in the image. Amendment A2 blamed a parser. The "three conversion instructions" fact is a PTX fact.
10. FACT (page size, local vLLM clone): The compact FP8 page (163,840 B for 64 tokens) does not divide the sliding page (524,288 B). The page unifier pads such a page to 524,288 B. That costs 8,192 B per token, twice the 4,096 B of the standard FP8 global page. This is the first risk of an engine prototype.
11. Recommendation for D-88: take (a) now. Open (b) as a gated build with a hard stop. Do not take (c).

## Answers

### 1. Recompute: INVALID (V4), KILL in 5 of 5 repeats, baseline check
- FACT: My selection equals the author's: standard 16/4/32/2 (32K) and 64/4/32/2 (64K); views at the same configuration; stock `stock_fp8_s32` at 32K and `stock_fp8_auto` at 64K. All 644 sweep rows pass the correctness cell.
- FACT: In the confirmation the auto row at 64K has dt 1,405.7 us and the s32 row has 1,390.7 us. The sweep chose by total time. The difference is 1.1% and does not change a word.

| Quantity | 32K, B 19 | 64K, B 11 |
|---|---|---|
| dt: stock, standard FP16, views, K = V (us) | 1,193.4, 910.0, 871.6, 811.5 | 1,405.7, 1,052.9, 984.1, 926.8 |
| sigma_layout (5 repeats) | 0.0422 (0.0416 to 0.0482) | 0.0653 (0.0602 to 0.0685) |
| sigma_dq, sigma_mma | 0.270, 0.141 | 0.300, 0.176 |
| standard over stock (dt) | 0.7625 | 0.7490 |
| e_h (5 repeats) | 0.216 (0.202 to 0.218) | 0.240 (0.237 to 0.272) |
| e_stock; e_bf16_triton; e_bf16_stock | 0.223; 0.856; 0.588 | 0.184; 0.817; 0.580 |
| pool share: standard, views, stock | 0.173, 0.139, 0.220 | 0.204, 0.150, 0.201 |
| BF16 pool share; BF16 sigma_layout | 0.481; 0.327 | 0.458; 0.296 |
| dt over copy (3,024.95 GB/s): standard, stock | 0.869, 0.662 | 0.898, 0.673 |
| sigma_stock_layout (s32 on both layouts) | 0.0956 | 0.0663 |
| FP16 and BF16 compute, stage 2 (us) | 910.0 and 1,060.7 | 1,052.9 and 1,205.3 |
| FP16 and BF16 compute, stage 3 (us) | 930.2 and 1,085.8 | 1,080.9 and 1,223.8 |
| V3: time ratio; dt ratio | 1.025; 1.066 | 1.003; 1.020 |

- FACT: Word on the medians: KILL. Word in each of the 5 repeats: KILL. Word with the E-51 stock dt (1,147.8 and 1,344.9 us): KILL (sigma_mma 0.107 and 0.139, far from 0.25). Flags a and c in both groups, as in the README.
- FACT: V4 fails on one row only: standard FP16 at 19:2K, median central spread 0.0523 (bound 0.02). All other Triton decision rows are 0.0143 or lower. The dt ranges are 0.14% to 0.63% (Triton) and 0.84% and 1.88% (stock).
- FACT: V6: 0 hard-throttle samples. The software cap (0x4) is in 95 of 502 busy samples (18.9%), mean 1,833 MHz. V5: the largest row rate is 2,589 GB/s and the largest dt rate is 2,717 GB/s (0.898 x copy); the bound is 1.15 x copy.
- FACT: The PTX hashes of the standard and views kernels match in 54 of 54 configurations. The cubins also match in 54 of 54. The selected kernels use 255 registers and no stack.
- FACT: The README numbers I did not list also recompute: the factorial (fused 23.0% and 25.2% slower with FP16; 27.9% and 27.5% with BF16), B 4 (273.7 against 290.3 us), the cost estimate and the Pod times.
- FACT: `label()` and `decide()` in `analyze_e52.py` follow README Section 7. I found no deviation.

### 2. The V4 failure and the power cap
- FACT: The slow repeats are 0, 2 and 4 (0-based). `confirm()` reverses the row order in repeats 1 and 3. In forward order the standard 19:2K row follows `stock_fp8_s32` 19:32K. That row ran at the cap in 5 of 5 repeats (1,715 to 1,745 MHz). In reverse order it follows the standard 19:32K row (1,970 to 1,980 MHz).
- FACT: The three slow repeats read 1,515, 1,605 and 1,715 MHz (cap bit in 3 of 3 samples). The two reverse repeats read 1,980 MHz. The README says "follows heavy rows". That is right and incomplete: the plan order puts a stock row in front of a Triton 2K row.
- FACT: Capped telemetry samples draw 693 W at the median (10th to 90th percentile: 660 to 700 W). Uncapped busy samples average 567 W. The board limit is 700 W. So the cap is a real power cap.
- FACT: A clock drop of up to 23.5% raised the 2K row time by 1.4% to 2.4% only. That row is not clock-bound.

| Treatment of the standard 19:2K row | dt standard (us) | sigma_layout | e_h |
|---|---|---|---|
| as run (median of 5) | 909.98 | 0.0422 | 0.216 |
| clean repeats only (2 and 4) | 910.21 | 0.0424 | 0.217 |
| row set to its minimum, 88.06 us | 910.77 | 0.0430 | 0.218 |

- FACT: The worst case, a 5.8% error on the 2K row, is 5.1 us, or 0.56% of dt. To reach the kill line, the standard dt would need to be 1,032 us (+13%) at 32K and 1,125 us (+6.8%) at 64K.
- FACT: The stock rows ran at the cap in all repeats: mean 1,732 MHz (32K, s32), 1,742 (64K, s32) and 1,823 (64K, auto). The Triton standard rows ran at 1,976 and 1,967 MHz. The cap moves the stock baseline, not sigma_layout. See Answer 6.
- FACT: The alias rows at 64K were slower in the capped repeats (927 against 913 us). So e_h reads 0.24 there and 0.27 in the clean repeats. Both are below 0.333.

| Rule | 32K | 64K | Word |
|---|---|---|---|
| Frozen rule, V4 ignored: sigma_layout | 0.042 | 0.065 | KILL, 5 of 5 |
| V4 on the dt range only | same | same | KILL, 5 of 5 |
| Views at its own selection | 0.087 | 0.065 | KILL |
| Best dt of 54 configurations for each kernel (sweep) | 0.101 | 0.071 | below the lines |
| Total time at X, best of 54 for each kernel (sweep) | 0.067 | 0.069 | below the lines |
| E-51 stock baseline: sigma_mma | 0.107 | 0.139 | KILL |
| sigma_dq at least 0.25, no layout condition | 0.270 | 0.300 | GO, but it measures the kernel |

- INTERPRETATION: No rule that asks the frozen question gives another word. The last row asks a different question: the standard kernel alone is already 0.237 and 0.251 faster than stock. The sweep maximum of sigma_layout (0.162 and 0.170, at 64/4/64/3) is a configuration about 60% slower than the best on total time. It is not a reasonable selection.
- INTERPRETATION: V4 bounds the replay spread of a 90 us row. That bound is far stricter than the effect on dt. L-19 says this. Add the cause: change the row order, or put a cool-down before each 2K row. Add that sigma_layout depends on the configuration (correction C11).

### 3. Is "about 80% on-chip work" valid?
- FACT: One minus the pool share is 0.827 and 0.796 (standard), 0.861 and 0.850 (views), 0.780 and 0.799 (stock). The arithmetic is right.
- FACT: The DRAM floor at the copy rate is 790 us (32K) and 946 us (64K). These are 0.869 and 0.898 of dt. The pool floor is 752 us and 839 us. The kernel runs 11% to 15% above its larger floor.
- FACT: The three points (all bytes, half the bytes, no DRAM bytes) are concave: the first 50% of the bytes gave 10.8% of dt (32K), the next 50% gave 6.5%. This is the shape of a floor.
- INTERPRETATION: The kernel is co-limited. DRAM time and the DRAM-free floor are close, and they overlap in part. A cut of 37.5% in bytes moves the DRAM time from 87% to 54% of the old dt. The floor stays at 80% to 83%. So the gain is small. "80% on-chip" is the floor. It is not "80% of the work".

Instrument checks:
- FACT: The BF16 pool rows move 5.87 TB/s (32K) and 5.67 TB/s (64K), 1.9 times the copy rate. The pool bytes come from L2 in the BF16 rows. The FP8 pool rows move 3.18 and 3.41 TB/s, only 5% and 13% above the copy rate. The rate alone cannot prove L2 residency for FP8. The same table construction and the BF16 result are the proof. This is INTERPRETATION.
- FACT: The BF16 pool share (0.481 and 0.458) is below the preregistered range 0.5 to 1.0 (README Section 8). Section 13 does not say so. The cause is not a broken probe: the BF16 kernel has its own floor of 814 us (52% of its dt).
- FACT: `ncu` is absent (`dram_probe.txt`). No row has a DRAM counter. The claims "K = V halves the DRAM bytes", "the pool has none" and "the views read 640 B" rest on construction plus the BF16 control.
- FACT: The Triton partials are 19.9 MB (16 splits, B 19) and 46.1 MB (64 splits, B 11) (`compact_attn.workspace`). The stock s32 partials are about 39.8 MB at B 19 (arithmetic; I did not read the FA4 layout). The 64K pool share is higher than the 32K share. There is no sign of L2 loss.
- FACT: The K = V oracle is valid on this GPU: e_bf16_triton is 0.856 and 0.817.

| Alternative | What the data say | Status |
|---|---|---|
| DRAM bandwidth alone | dt is 0.87 to 0.90 x copy, but half the bytes give only 11% to 12% | not alone: co-limit |
| L2-to-SM bandwidth | The BF16 pool rows carry 5.7 to 5.9 TB/s through the same path. The FP8 pool rows carry 3.2 to 3.4 TB/s. | not the hardware limit (the FP8 kernel may still use the path badly) |
| FP8 cast work | BF16 against FP16 compute differs by 14 to 17 SM-clocks per token-head (12% to 14% of dt). The BF16 cache has no cast and a floor of 91 to 94 clocks against 78 to 84 for FP8. | at most 14% of dt; not the floor |
| Tensor-core work | About 76 GFLOP issued (16 padded rows) in 752 us is about 10% of the dense FP16 peak (989 TFLOPS, NVIDIA figure) | ruled out (arithmetic) |
| Latency of a single-buffered loop | Both kernel families stage K and V once (R-70 for FA4; R-71 F3 for Triton). The standard kernel is 21% (32K) and 26% (64K) slower than its pool row. `num_stages` 3 added no buffer. | open; fits the data (HYPOTHESIS) |
| Occupancy | The 4-warp FP8 kernels use 255 registers (cubins). A 64K-register file then holds 2 CTAs, 8 warps per SM (arithmetic). The BF16 kernels use 133 to 151 registers: 3 CTAs. The 8-warp tile-32 configurations are 20% to 27% slower than the 4-warp ones. The register count of the 8-warp kernels is not in the data. | open; no row varies registers alone |
| Merge kernel | It has the same split count at 2K and X, so it cancels in dt | ruled out for dt |
| Split count | 16, 32 and 64 splits move dt by 3% to 5% (32K: 912, 895, 887 us; 64K: 1,119, 1,083, 1,061 us) | ruled out as the cause of the floor |
| SM clock | In four rows with a 100 MHz range or more, d ln t / d ln f is -0.07 to 0.28 (n = 5 or 6) | no sign of clock-bound time (weak) |

- INTERPRETATION: The claim is valid in this form: "about 80% of dt does not depend on the DRAM bytes (pool floor 80% to 83%)". It is not valid as "on-chip work" until the open rows are tested. The sentence "it overlaps the DRAM reads, so removing bytes saves little" has the right structure.
- HYPOTHESIS: The single-buffered loop exposes load latency. The DRAM latency exceeds the L2 latency, so the pool row (17% to 20% faster at equal bytes) shows a latency cost, not a bandwidth cost. This predicts a gain from a kernel with a true two-tile prefetch. No E-52 row tests it.

### 4. The stock kernel on the views layout
- FACT (correctness cell: B 2, L 3,000, poisoned tail, synthetic data): FP8 `err` is 0.0016638 (s32 views) against 0.0016622 (s32); 0.0016638 (auto views) against 0.0016623 (auto). `exact_err` is 0.023274 against 0.023278 (s32). BF16 `err` is 0.001846 against 0.001843.
- FACT: No timed cell (B 11 to 19, 32K to 64K) has an output check. This is normal practice and still a limit.
- FACT: The comparison uses `num_splits` 32 on both layouts, the same call and the same graph procedure. Only the strides differ. It is fair on kernel and split setting.
- FACT: The gain is positive in all 10 repeats: 8.2% to 10.7% (32K), 6.2% to 8.1% (64K). Against the selected stock row at 64K (auto) the gain is 7.6%.
- FACT: The sweep singles give 8.0% and 5.2% (s32) and 2.5% and 5.3% (auto). The 32K auto row is a poor stock setting (dt 1,642 against 1,171 us).
- Caveat 1 (FACT): The views rows ran 5.7% higher in SM clock (1,831 against 1,732 MHz at 32K; 1,845 against 1,742 MHz at 64K). They draw less power. With the weak clock sensitivity above, up to about 1.6 points of the gain could be clock. This is INTERPRETATION.
- Caveat 2 (FACT): The central spreads of the stock views rows are 3.5% to 5.5%. The dt ranges are 2.9% and 2.2%. The 64K effect is about three times the range.
- INTERPRETATION: The stock gain matches the bytes-only prediction of its own oracle (0.375 x e_stock = 0.084 and 0.069; measured 0.096 and 0.066). The Triton views kernel gets 0.042 and 0.065 against 0.081 and 0.090. So the Triton views kernel loses about 4 points at 32K. HYPOTHESIS: its two `cp.async` tile loads fetch the 384 shared bytes twice from L2. The TMA kernel does not lose.
- FACT (not in O-14): The stock BF16 s32 kernel gains 24.0% (32K) and 24.5% (64K) on dt with the views (sweep singles). The stock BF16 auto gains 10.1% and 19.9%.

### 5. FP16 against BF16, the factorial and H5
- FACT: FP16 against BF16 compute at the selected configuration: 14.2% and 12.6% (stage 2), 14.3% and 11.7% (stage 3). The factorial singles at 32/4/32/2: 12.4% and 12.9% (standard), 15.6% and 14.5% (views), 15.8% and 14.4% (fused). The claim "13% to 14%" fits stage 2. The honest range is 12% to 14% for the standard kernel.
- FACT: The BF16-compute kernel was timed at the FP16 optimum only. It was not tuned. So the gain is "at the FP16 configuration", not best against best.
- FACT: H4 as preregistered (Section 3) needs a rate of 0.85 x copy or more AND e_h of 0.6 or more. The rate holds (0.87, 0.90). e_h fails (0.22, 0.24). Section 2 states "the BF16 cast sets most of the E-51 Triton FP8 time". That is false (12% to 14%). The label "partly true" is too kind. Use "false; the rate half holds".
- FACT: A1 gave two branches: a cast at 16 per clock and SM costs 64 clocks per token-head; at 64 per clock it costs about 16. The data give 16.9 (32K) and 14.3 (64K) clocks. They match the second branch ("the cast alone cannot explain E-51"). The conversion rate itself is not measured (no SASS).
- FACT: Shared memory is 33,280 / 50,176 / 83,968 B at stage 2 and +4 B at stage 3 for every (warps, tile) cell in all 54 configurations. The PTX differs in 27 of 27 stage pairs. So stage 3 changed the schedule and added no buffer.
- FACT: Stage 3 is 2.2% and 2.7% slower (FP16), 2.4% and 1.5% slower (BF16).
- INTERPRETATION: "H5 is false" is true only for "stage 3 as Triton 3.8.0 compiles it". It does not answer the latency question of R-71 F3.
- FACT: With FP16 compute the views effect doubles in the factorial singles: BF16 compute 0.036 and 0.037, FP16 compute 0.070 and 0.054. The fused kernel stays 23% to 28% slower than its twin. The fused FP16 kernel has a 24 B stack.

### 6. O-14 and O-15: scope
O-14 (views read by the stock kernel):
- Within scope: the numbers (7% to 10%; kernel error 0.00166), the single H100, the microbenchmark label.
- Overstated: "costs no speed" and "needs no new attention kernel" read as engine facts. They are facts about one FA4 FP8 kernel in a microbenchmark.
- Missing: (i) the multimodal Gemma 4 FP8 path uses the Triton backend (R-70 note on vLLM PR 60372). It is not tested with views. (ii) "exact" is algebraic. The FP8 rounding differs (K_unrot comes from V). No accuracy gate ran on real activations. (iii) The o_proj fold is exact for BF16 weights only. (iv) The page-size result (Key finding 10). (v) The stock BF16 gain (24%). (vi) The views rows ran at a higher clock.

O-15 (Triton FP16 kernel against stock FA4):
- Stated: "2 split settings against a 54-configuration sweep". Good.
- FACT: 21 of 54 (32K) and 15 of 54 (64K) Triton configurations beat the best stock row on total time. So the result is not a lucky configuration. It is still tuned against untuned: the stock kernel has no tile or stage knob here, and 8, 16 and 64 splits were not timed.
- FACT: At BF16 the Triton kernel ties the stock kernel (dt ratio 0.992 and 0.999). The advantage exists only in the FP8-dequant regime, where both are not DRAM-bound.
- Missing: the baseline range (19% to 25%, correction C4); the stock rows' lower clock; the specialization of the Triton kernel (head_dim 512, 8 query heads per KV head, no soft-cap, window, sink or ragged batch); FP16 range on real activations (Section 10); "three conversion instructions" is PTX, not SASS.

## Numbered corrections
- C1. README Section 13 "What this means", O-13, journal Section 1 ("about 80% ... on-chip work"): write "about 80% of dt does not depend on DRAM bytes (pool floor 80% to 83%); the DRAM time is also 87% to 90% of dt; the cause of the floor is open (latency, occupancy, issue)". Name the nine rows of Answer 3.
- C2. README Section 13 H4 "partly true": write "false as preregistered (rate holds; e_h 0.22 and 0.24 against 0.6; the cast is 12% to 14% of dt, not most)". Add the 14 to 17 clocks and the A1 branch.
- C3. README Section 13 H5 "false": write "`num_stages` 3 adds no K or V buffer (shared memory +4 B in 54 of 54); it is 2% to 3% slower; the latency question is open".
- C4. README Section 13 and O-15 "24% to 25% faster than stock": write "19% to 25%". Against the E-50 in-engine dt: 18.7% and 22.8%. Against the E-51 stock dt: 20.7% and 21.7%. Against this session: 23.7% and 25.1%. The E-52 stock rows are 3.4% to 4.5% above E-51 and ran at the cap.
- C5. README Section 13: state that the BF16 pool share (0.48, 0.46) is below the preregistered 0.5 to 1.0. Give the BF16 pool rate (5.7 to 5.9 TB/s) as the L2 evidence.
- C6. README Section 12 item 8 and A2: the SASS counts were 0 because `nvdisasm` is missing (`sass/*.sass` hold the line "Could not find executable file 'nvdisasm'"). The parser was not the cause. Say that no SASS counts exist for E-52. HYPOTHESIS: the Triton wheel in the image ships a copy of `nvdisasm`; check with `find`. The cubins are saved, so any Linux box with the CUDA binutils can read them at no GPU cost.
- C7. README Section 13 L-19 and RESEARCH_OUTPUTS L-19: add the cause (the forward-order row follows a stock row at the cap). Fix "E-51 and E-52 are both INVALID for this reason". E-51 failed V3 (a 2K time ratio) and V4 on two stock rows at X (spreads 4.2% and 2.7%). Only E-52 failed V4 on a short Triton row.
- C8. O-14: add the missing items of Answer 6 (Triton backend untested, algebraic exactness, o_proj fold, page size, stock BF16 24%, clock). Replace "costs no speed" with "in a microbenchmark, the stock FA4 FP8 kernel is 7% to 10% faster".
- C9. O-15: add "21 of 54 and 15 of 54 configurations beat stock", the BF16 tie, the clock of the stock rows, the specialization and the PTX label.
- C10. T-11: add the o_proj condition (exact for BF16 weights or a block-aligned permutation) and "the Triton views kernel gets 52% to 73% of the oracle's bytes-only gain".
- C11. README Section 13: report the configuration sensitivity of sigma_layout: 0.042, 0.070, 0.058, 0.059, 0.101 and 0.073 for the six best 32K configurations (single sweep rows except the first), and `sigma_layout_own` (0.087 at 32K) from `summary.json`. All are below the lines.
- C12. O-13 "so about 80% of the time is on-chip work": as C1. "K = V elasticity 0.22 to 0.24" is the Triton FP16 kernel; the stock kernel reads 0.18 to 0.22.
- C13. README Section 13 stock-on-views bullet: add the SM clock (+5.7%), the 3.5% to 5.5% spreads and the 7.6% gain against the 64K auto row.
- C14. The README Section 2 expectation (sigma_layout 0.25 to 0.37, GO) failed by a large margin (0.04 and 0.07). Record that H1 is rejected, not only that the run is INVALID.

No change to the frozen rule, to `analyze_e52.py` or to the decision word is needed.

## Recommendation for D-88
Take (a) now. Open (b) as a gated build with a hard stop at the end of week 6. Do not take (c).

What the evidence gives to each option:
- FACT: The speed route is closed on the H100 for FP8 KV. With R-70 factors (0.3227 at 32K, 0.4 at 64K) the stock-kernel views gain is 3.1% and 2.7% of the step time near the pool limit. The Triton views gain is 1.4% and 2.6%. The project bar is 5%.
- INTERPRETATION: The value of I-182 is capacity. E-50 (exploratory arithmetic) gives 1.39 times (32K) and 1.47 times (64K) more requests in the same FP8 pool. The microbenchmarks show no speed cost for the compact cache. No end-to-end run has measured the capacity or the throughput.
- INTERPRETATION: For BF16 KV the views also give speed. The stock BF16 gain is 24% on dt (single sweep rows). With e_bf16_stock 0.58 and the E-50 ceilings (13.7% and 14.5%) the step gain is about 8% (HYPOTHESIS, not measured).

| Option | Thesis value | Risk | Time and cost | View |
|---|---|---|---|---|
| (a) write-up: mechanism, views trick, measurement method, closed byte route | Medium. It gives a chapter with a mechanism, a reusable method (K = V oracle, pool probe, BF16 control, compile check) and a clean negative result. It has no end-to-end number. | Very low | 1 to 2 weeks, $0 | Do it first. It is the floor of the thesis. |
| (b) vLLM prototype and capacity test | High if it works: the only end-to-end capacity and throughput evidence, and an upstreamable change. | Medium to high | 5 to 6 weeks; GPU about $20 to $40 (HYPOTHESIS) | Do it, gated. |
| (c) FP8 MMA kernel test | Low. HYPOTHESIS: KILL again (R-70; E-52 shows the same floor pattern in three kernels). It adds no capacity. | Medium (overlay of FA PR 199 into the image; a new accuracy bound) | 1 week, $2 to $3 | Skip. Run it only if a newer vLLM image already ships the kernel: then the E-52 harness gives the answer in one session. |

Risks of (b), with what I found:
- FACT (local clone 24e80c2, `kv_cache_planning.py` line 346): `unify_kv_cache_spec_page_size` raises the block size of a smaller page by an integer ratio. If the ratio is not an integer, it pads a non-MLA attention page to the maximum page. The compact FP8 page for 64 tokens is 163,840 B. The sliding page is 524,288 B. The ratio is 3.2. So the compact page would be padded to 524,288 B, which is 8,192 B per token against 4,096 B for the standard FP8 page after unification. No integer block size fixes it, because 640 B x 4 heads has a factor 5.
- INTERPRETATION: A working plan makes the global page the largest: 1,024 tokens = 2,621,440 B = 5 sliding pages, with a sliding block of 320 tokens. Then `scheduler_block_size = math.lcm(...)` becomes 5,120 tokens (`kv_cache_utils.py` line 863). HYPOTHESIS: prefix-cache hits then land on 5,120-token units. This hurts prefix caching. Check it first.
- FACT (same clone): The Gemma 4 loader already maps K weights into both the K and V shards for `attention_k_eq_v` layers (`gemma4.py`). The engine team knows the structure. A duplicate change upstream is possible. R-64 found none on 2026-10-30. Re-scan weekly.
- Open: accuracy on real activations (K_unrot is stored with V's FP8 rounding; the k_norm weight moves into a BF16 query); the o_proj fold for block-scaled FP8 weights (it is exact if the rotated part is one 128-aligned block per head; HYPOTHESIS, check the checkpoint); CUDA graphs; chunked prefill; speculative decode.

Plan for 9 weeks (HYPOTHESIS: my estimate of the odds that (b) reaches an end-to-end capacity result by week 6 is 40% to 55%):
1. Week 1: write (a) (outline, tables, corrections C1 to C14). In parallel run the free gates. G1: the page-size plan and the prefix-cache effect, from source (2 days). G2: the upstream re-scan, the o_proj weight format and the RoPE layout (1 day). G3: extract real Gemma 4 K and V of the global layers from one forward pass (one short Pod session, about $2), then test the compact path against the standard FP8 path off-line (accuracy on real activations, including the FP16 range; 2 days).
2. End of week 1: go or stop for (b). Stop if G1 needs a change to the scheduler that you cannot isolate in a patch, or if G3 shows an accuracy loss above the FP8 KV noise.
3. Weeks 2 to 5: build the cache write, the views read, the query preparation and the o_proj fold behind a flag. Test BF16 KV and FP8 KV arms.
4. Week 6: end-to-end capacity and throughput near the pool limit (arms: standard and compact, BF16 and FP8 KV). Stop here whatever the state.
5. Weeks 7 to 9: thesis assembly. Keep these three weeks free. D-13 is still open and the other chapters (I-23, O-03) need the time.

If a gate fails, the fallback is (a) plus the free items (C6 SASS read from the saved cubins).

## What I could not check
- No SASS exists, so the conversion rate and the loop structure of the Triton and FA4 kernels are unread.
- No DRAM or L2 counter exists for any row. The pool, alias and views claims rest on construction plus the BF16 control.
- No real activations, no H200 or B200 run, no engine step. The rehearsal on the RTX 4090 gives sigma_layout 0.358 for the views (FP16, factorial single). So a DRAM-bound GPU gets close to the full byte gain. INTERPRETATION: the speed result is specific to the H100 class.
- The state of upstream vLLM, SGLang and TensorRT-LLM after 2026-10-29.
- The FP8 MMA kernel of FA PR 199.
