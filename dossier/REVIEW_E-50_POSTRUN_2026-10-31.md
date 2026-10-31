# R-68: post-run review of E-50 (Gemma 4 K=V headroom on top of an FP8 KV cache, I-182)

Date: 2026-10-31. Reviewer: Sonnet (a different model from the author).
VERDICT: CONFIRM_WITH_SCOPE_LIMITS (the GO reproduces from the raw files; no number that I could check is wrong; 7 wording corrections C1 to C7; S is a byte-proportional ceiling, not a forecast)

Scope: `experiments/gemma4_kv_headroom/` (rule frozen at 5eb82c2, repo HEAD 365b5aa), `decode_probe.py`, `analyze_e49.py`, the H100 raw data in `results/e50-h100-20261031/`, the 4090 rehearsal data, E-49 Section 13 and R-65.
Labels: FACT (I read a source or recomputed a number). INTERPRETATION (my reasoning). HYPOTHESIS (not tested).
I changed no repo file except this one. I did not read `.env`. I made no Runpod or cloud call. I used read-only `gh` queries, two web searches and one page fetch.
Scratch scripts are in the session scratchpad folder `r68/` (`refit.py`, `scenarios.py`, `grids.py`, `kdiff.py`, `e49rule.py`).

## Verdict in short
- FACT: The frozen rule gives GO. My own refit (exact t quantiles) matches the author to the fourth digit. `analyze_e50.py` rebuilds `summary.json` as identical JSON.
- FACT: Every validity check passes. No request left early. No preemption. The FA3/FA4 path ran. No Triton kernel ran. Code and rule are unchanged since 5eb82c2.
- FACT: The same data show that the stock FP8 kernel does not follow the bytes. Half the bytes (BF16 to FP8) cut the time per context token by 24.5%, not 50%.
- INTERPRETATION: A compact cache must realize 41% (32K) or 33% (64K) of the byte-proportional saving to keep S above 5%. The measured ratio 0.49 is only a little above that line. The real S is plausibly 4% to 8%.
- INTERPRETATION: The GO license is right (prototype plan only). The first prototype test must be a kernel microbenchmark with a kill line at S 5%.

## Recomputed decision (FACT)
| Group | B | Cells | df | b (ms per 1,000 tokens) | c (ms) | T(X) (ms) | S | S_low | S_up | M2 | Global attention GB/s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32K | 19 | 4 | 2 | 0.36326 | 24.936 | 36.842 | 0.1212 | 0.1203 | 0.1221 | 1.004 | 2,142 (0.70 x copy) |
| 64K | 11 | 5 | 3 | 0.21479 | 21.075 | 35.168 | 0.1501 | 0.1488 | 0.1514 | 1.011 | 2,098 (0.69 x copy) |
- My refit uses `scipy.stats.t.ppf(0.95, df)`: 2.9200 (df 2) and 2.3534 (df 3). The author's table has 2.920 and 2.353. All values equal the author's to the printed digits.
- Both groups pass. S_low is 2.4 and 3.0 times the 5% bar. M1 is 0.70 and 0.69 (bar 0.60). M2 is true.
- V4b is true: attention explains 100.1% and 100.2% of the device growth. V4c is true: sliding attention changes by +0.06% and -0.12%.
- B = 4 slope cells of m31f8 (6 cells, df 4): b 7.84e-5 ms per token, c 17.263 ms, 2,089 GB/s.
- S at B = 4: 4.84% (low 4.64%) at 32K and 8.62% (low 8.27%) at 64K. This matches Section 13.
- m31bf groups: S 13.69% (low 13.47%, df 1) and 14.50% (low 14.35%). m26: S 6.08% (low 5.72%) and 10.46% (low 9.84%), df 4. All match Section 13.
- Post hoc, I applied the frozen E-49 rule to the E-50 BF16 data (m31bf at B = 4, m26 at B = 8). It gives GO with 4 of 4 cells passing. The E-49 record stays PIVOT.

## Checks (FACT unless marked)
| Check | Result | Evidence |
|---|---|---|
| Code and rule frozen | Pass | `git diff 5eb82c2` is empty for `decode_probe.py`, `analyze_e49.py`, `analyze_e50.py`, `session_e50.sh`, `test_e50.py` and `test_e49.py`, at HEAD and in the working tree. README changes since 5eb82c2: status line, S5 record, Section 13. 39 tests pass (19 + 20). |
| Checksums | Pass | `remote_sha.txt` equals `local_sha.txt` (81 lines). `shasum -c` finds 81 of 81 local files OK. |
| V1 | Pass | H100 80GB HBM3, power limit 700 W (SXM), UUID 69045460 in `run.json` and `gpu.csv`. |
| V2, V7 | Pass | All 32 cells advanced exactly 32 tokens in every block. 3 blocks per cell. Largest block spread 0.48% (m31f8). |
| V3 | Pass | m31f8 B = 4 residuals at most 0.61%. Group residuals at most 0.10% (32K) and 0.23% (64K). |
| V4a, V4b, V4c | Pass | B = 4: device growth / wall growth 0.948. Groups: 1.005 and 1.000. Sliding attention +0.06% and -0.12% (B = 4: +0.5% and +0.1%). |
| V5, M1 | Pass | 2,089 / 2,142 / 2,098 GB/s against copy 3,045.0 GB/s (start and end values are equal). The V5 limit is 1.15 x copy. |
| V8 | Pass | 0 hard-throttle samples among 929 busy samples. The soft power cap (bit 0x4) is set in 78% of them. Median 675 W of 700 W and SM clock 1,770 MHz. |
| V9, V11 | Pass | The m31f8 `log.txt` has `kv_cache_dtype=fp8`, "per-layer FA3/FA4 selection for FP8 KV cache on SM90", the FLASH_ATTN backend line, and FlashAttention version 3 and 4. The BF16 arms have the "FA4 for all layers" line. `kernel_unified_attention` is absent from all 17 profiles. |
| Kernel names | Pass | m31f8 per step: FA3 `FlashAttnFwdSm90` 50 (sliding), CuTe `FlashAttentionForwardSm90` 10 (global), FP8 KV write `reshape_and_cache_flash_kernel<bf16, unsigned char, Fp8KVCacheDataType 1>` 60. |
| Layer split | Pass | 480 attention groups = 8 steps x 60 layers in every profile. Match 1.00. The order method works on the FA3/FA4 mix. |
| Pair batch rule | Pass | int(0.85 x 8,017 / 344.5) = 19 and int(0.85 x 8,017 / 601.75) = 11. BF16: int(0.85 x 16,095 / 1,358) = 10 and int(0.85 x 16,095 / 2,390.5) = 5. Graph sizes: 19 runs in 24, 11 in 16, 10 in 16, 5 in 8. |
| Early exit, preemption | Pass | 32 of 32 cells "ok", `finished` empty, preemptions 0 before and after, running = B, waiting 0. Fewest free blocks after timing: 1,421 (FP8) and 2,545 (BF16). Token slack at least 33 against a 5-block worst case. |
| JIT lines | Pass | m31f8: none. m31bf and m26: the CuTe JIT line comes before the first cell. One Triton MoE JIT (m26) occurs in the 8K cell; its blocks agree within 1%. |
| Rehearsal (S5) | Pass | 11 cells ok; B = 35 = int(0.85 x 175,214 / 4,174); decision INVALID (V1, V5, V11) as planned. |
- I checked these Section 13 values against the raw files: pool, B_max, shares, GSM8K, engine start times and copy bandwidth.
- I also checked the FP8-against-BF16 ratios, the tokens per second, S for all arms and the Pod cost ($3.99/h x 43.6 min = $2.90). All match.

## Attacks
### 4a. The byte model for a kernel that is not at the roofline
- FACT: BF16 global attention at B = 4 reads 3,157 GB/s (1.04 x copy; read-only traffic). FP8 reads 2,089 to 2,142 GB/s of FP8 bytes at every batch (0.69 to 0.70 x copy).
- FACT: The FP8 kernel handles 2.1e12 values per second. That is only 1.3 times the BF16 rate.
- FACT: The time per context token falls from 1.038e-4 to 7.84e-5 ms for half the bytes. The time elasticity to bytes is 0.49.
- FACT (FA PR 199 table; H200; head dim 512; num_splits 1; page 64): at B = 64 and 28K the FP16-KV kernel takes 3.76 ms. The FP8 dequant kernel takes 3.31 ms. The FP8 MMA kernel takes 3.07 ms.
- FACT (same table): at B = 1 and 16 the FP8 MMA kernel is within 4% of the FP16 kernel. The dequant kernel is 11% to 12% slower.
- INTERPRETATION: Those FP8 kernels read 2.3 to 2.45 TB/s on an H200 (about half its peak). On the H100 they read 2.1 TB/s (63% of peak). Their time does not follow the bytes. A limit set by on-chip work per KV block fits all of it.
- INTERPRETATION: M1 compares with the copy bandwidth of the same GPU. The same kernel would reach about 52% to 56% of an H200 copy bandwidth near 4.4 TB/s (HYPOTHESIS, not measured). It would fail M1. So M1 tests the GPU as much as the kernel.
- INTERPRETATION: A compact layout cuts DRAM bytes and K-side loads by 37.5%. It keeps the MMA, softmax and V transposition work (512 PV dims; 512 QK dims at best). So the saving is between 0% and 37.5% of the context part.
- INTERPRETATION: The saving exceeds 37.5% only if the new kernel is also better than the stock one. That is a kernel gain, not a layout gain.
- Break-even: S of 5% needs a cut of 15.5% (32K) or 12.5% (64K) of the context-proportional time. That is a realized fraction r of 0.41 and 0.33 of the 37.5%.

| Realized fraction r | S at 32K | S at 64K | Basis |
|---|---|---|---|
| 1.00 | 12.1% | 15.0% | time follows bytes (the author's S) |
| 0.68 | 8.2% | 10.2% | only the roofline part (68% of the time) shrinks |
| 0.49 | 5.9% | 7.4% | BF16 to FP8 elasticity in this run |
| 0.24 | 2.9% | 3.6% | H200 FP16 to FP8, B = 64 (FA PR 199) |
- INTERPRETATION: "Less" is more likely than "more". The real S is plausibly 4% to 8%. It sits near the bar at 32K. The 64K group has more margin.
- Design note (HYPOTHESIS): Gemma 4 rotates dims 0 to 63 and 256 to 319. A kernel can read QK from 6 row chunks plus the 2 K_rot chunks, so QK stays at 512 dims. A naive 640-dim QK adds 25% MMA work.

### 4b. FP8 against BF16 at B = 4, and the larger fixed part c
- FACT: The ratios are right: +5.8% (2K), +5.2% (8K), +1.5% (32K), -2.6% (64K). Medians are 17.319 / 17.913 / 19.928 / 22.356 ms (FP8) against 16.372 / 17.036 / 19.637 / 22.963 ms (BF16). The lines cross at 1.078 / 0.0254 = 42K tokens.
- FACT: Both arms ran in the same Pod. The SM clock medians in the two B = 4 phases are 1,770 and 1,755 MHz.
- FACT: The arms differ in four ways. The sliding layers use FA3 with FP8. The global layers use FA4 with the 32-split cap (grid 16 x 32, against 16 x 8). The block sizes are 64 and 128, against 16 and 32. The KV write kernel differs.
- FACT (kernel diff at 2K, B = 4, device ms per step): sliding attention +0.50 (FA3 main, combine and prepare against FA4). Global attention +0.22 (56 against 34 us per layer; 32 splits of 64 tokens). FP8 KV write +0.15. Other kernels +0.22. The total is +1.10; the fit says c is +1.08.
- FACT: The comparison is fair as "two stock paths in this image at B = 4". It is the BF16 best case (see 4d). Each arm has one engine start and no replicate. E-49 against E-50 BF16 shows a 0.33 to 0.39 ms offset between Pods (2.3% in c).
- FACT: The result does not extend to larger B. I compare FP8 at B = 11 with BF16 at B = 10. At 2K: 21.56 against 19.43 ms (+11%). At 8K: 22.82 against 21.48 (+6%). At 32K: 28.05 against 29.55 (-5.1%).
- FACT: The crossing is near 19K tokens at B = 10 to 11, with one more sequence on the FP8 side.
- FACT: A reviewer on PR 60372 reports that FP8 KV is still 1% to 10% slower than BF16 KV end to end. That agrees with the B = 4 result.

### 4c. Capacity claims
- FACT: The arithmetic is right. FP8: 19 / 36.842 ms = 515.7 tok/s and 11 / 35.168 = 312.8. BF16: 10 / 29.553 = 338.4 and 5 / 27.120 = 184.4. The ratios are 1.524 and 1.697. The pool ratio is 23.27 / 11.85 = 1.96.
- Not fair: the pool shares differ (FP8 0.82 and 0.83; BF16 0.84 and 0.74). The batches differ, so c differs.
- INTERPRETATION: I assume c(B) is linear in B and I take b per sequence from the group fits. At equal pool share the gain is then +53% (32K) and +60% (64K).
- INTERPRETATION: At the integer pool-limit batches (23 against 11; 13 against 6) the gain is +57% and +67%. The +70% at 64K is about 10 points high.
- FACT: The padding (24, 16, 16 and 8 graph slots) hits both arms. The GEMMs are weight-bound, so the effect is small.
- FACT: GSM8K cannot show a loss. FP8 has 29 of 32 correct and BF16 has 28 of 32. 27 of 32 answers are equal. 1 of 32 texts is identical. The scale is 1.0.
- INTERPRETATION (exploratory, R-67 O3): a compact global cache raises the requests at the same pool by 1.39 times (32K) and 1.47 times (64K). Global blocks are 74.3% and 85.1% of the blocks per request. Section 13 omits this line.

### 4d. Slope per sequence at larger B (1.23 to 1.27 times)
- FACT: It is real for the stock BF16 kernel. The ratio is 1.269 (B = 10) and 1.234 (B = 5). For FP8 the ratio is 0.975 (B = 19) and 0.996 (B = 11). It is flat.
- FACT (profile grids): at B = 4 the BF16 global kernel launches 16 x 8 = 128 CTAs (3.05 to 3.12 TB/s). At B = 10 it launches 68 x 2 CTAs over 16 graph slots. At B = 5 it launches 32 x 4 over 8 slots.
- INTERPRETATION: Only 10 and 5 slots hold real sequences. If the padded slots exit at once, about 80 CTAs work in both cases. The measured rates are 2.46 to 2.52 TB/s.
- INTERPRETATION: The cause is the split choice. B = 4 uses 97% of the 132 SMs. B = 5 and B = 10 use about 61%. The byte model is not the cause.
- INTERPRETATION: L2 is not the cause. A 2K, B = 4 cache is 67 MB per layer, the L2 is 50 MB, and each layer reads its own cache once.
- INTERPRETATION: The R-65 model T = 14.48 + 0.336 B + 2.605e-5 B L is a calibration at the BF16 sweet spot. At the two BF16 pair cells it gives 26.4 and 24.7 ms. The measured values are 29.6 and 27.1 ms (-11%, -9%).
- INTERPRETATION: The model gives S 12.1% and 13.0%. The measured S is 13.7% and 14.5%. For FP8 the linear scaling holds (flat slope).
- INTERPRETATION: The R-67 forecast for FP8 (11.8%, 13.7%) is close to the measured 12.1% and 15.0%. The R-65 model with the FP8 term halved implies T near 26 ms. The measured T is 36.8 and 35.2 ms. The match comes from b and T both being about 30% low. This is partly luck.

### 4e. The m26 result and the E-49 gap
- FACT: The data gap is closed. 6 of 6 cells are ok and none left early. `tokens_at_start` is 226 at 64K, as the R-65 replay says.
- FACT: The R-65 fit predicted T 12.6 ms and S 10.5% at 64K. The run gives 12.566 ms and 10.46%.
- FACT: V4b passes by 0.8 points (80.8% against 80%). The MoE kernel adds 0.38 ms from 2K to 32K. S_low at 32K is 5.72%. The 8K cell is +1.4% above the line.
- INTERPRETATION: The MoE growth inflates S by about 0.2 to 0.3 points (S_att is 5.9% and 10.1%).
- Not answered: m26 ran only at B = 8, which is 50% of its 64K pool limit (B_max 16.1). It ran only with the BF16 cache, in one session.
- The E-49 decision stays PIVOT. The post hoc GO is not a change of the E-49 record.

### 4f. A faster FP8 kernel
- FACT (PR 60372, read 2026-10-31): it removes the 32-split cap for FA4 layers and adds the FP8 MMA kernel (FA PR 199, merged 2026-10-29).
- FACT: A reviewer's end-to-end test (TP4, short contexts) reports that the cap removal gives +4.5% and +11.3% decode at concurrency 32 and 128. It reports that the FP8 MMA kernel itself adds 0% to 1%.
- INTERPRETATION: E-50 measured the capped dequant kernel. After the PR the split count changes. The BF16 arm shows that the automatic choice can cost SM coverage. The sign of the change at B = 19 and 11 is unknown (HYPOTHESIS).

| Stock FP8 kernel reaches | S at 32K | S at 64K | Stock step falls by |
|---|---|---|---|
| 2,500 GB/s (BF16 at the pair batches) | 10.9% | 13.5% | 4.6% and 6.4% |
| 2,800 GB/s | 10.0% | 12.5% | 7.6% and 10.0% |
| 3,045 GB/s (copy) | 9.4% | 11.8% | 9.6% and 12.5% |
| 3,157 GB/s (BF16 at B = 4) | 9.2% | 11.5% | 10.4% and 13.4% |
| Context part 8% to 15% faster (FA PR 199 size) | 10.8% to 11.4% | 13.6% to 14.3% | 3% to 6% |
- INTERPRETATION: The ceiling stays above 9% in every case. The GO is robust to a faster stock kernel.
- INTERPRETATION: A roofline-speed stock kernel alone would cut the stock step by 10% to 13%. That is as large as the layout ceiling.

## Corrections (required before the result enters the journal; no number changes)
- C1 (S premise). Section 13 and the journal must say that S assumes time follows bytes. Add the elasticity 0.49, the break-even r* 0.41 and 0.33, and the FA PR 199 kernel data. Write "ceiling above the bar", not "headroom 12% to 15%".
- C2 (FP8 against BF16). Replace "FP8 KV wins on capacity, not step time" by: "At B = 4 FP8 is slower below about 42K. At B = 11 against B = 10 it is 5% faster per step at 32K."
- C3 (capacity at 64K). Say "+70% as measured, with BF16 at share 0.74; about +60% at equal share (R-68 estimate)".
- C4 (PR 60372 sentence). Name the split-cap removal as the main change. The FP8 MMA kernel adds 0% to 1% end to end in the reviewer's test. E-50 measured the capped dequant path.
- C5 (fixed part c). Replace "the FA3/FA4 mix and the dequant kernel" by the split of the +1.08 ms. Sliding FA3 is +0.50, global FA4 is +0.22 (32 splits at 2K), the FP8 KV write is +0.15 and other kernels are +0.22.
- C6 (batch model). The 1.23 to 1.27 inflation is the SM coverage of the BF16 kernel (about 80 of 132 CTAs, against 128 at B = 4). The FP8 slope per sequence is flat. The R-65 model is a B = 4 calibration.
- C7 (m26 and E-49). Add four facts. The E-49 rule applied post hoc gives GO (4 of 4 cells). The V4b margin is 0.8 points. The pool share of m26 is 0.50. The R-65 prediction is confirmed.

## Scope limits
- S is a ceiling. It needs a compact FP8 kernel with time proportional to bytes. No such kernel exists.
- Text-only mode is required. PR 60372 says multimodal Gemma 4 with FP8 KV on Hopper uses Triton. The E-50 log shows "Disabled mm_prefix attention mode" (set by the 0 image, audio and video limits).
- The stock path is the build at 3966924d5 with the 32-split cap. PR 60372 and FA PR 199 change it. The 42K crossing and the FP8 numbers are for this build.
- One H100 SXM, one session, one engine start per arm. No engine replicate. The power cap is active in 78% of busy samples. The cross-Pod shift of c is 2.3%.
- Decode only, uniform contexts, no prefix cache, no speculative decoding, no prefill chunks in the timed steps. The pool share is 85%. Padded graph sizes lower S slightly.
- The 50 sliding layers use 20% of the step (7.4 ms at B = 19). That is outside this question.
- Accuracy: GSM8K n = 32 and scale 1.0. A community llama.cpp test (2026-04-24) reports large KL for q8_0 KV on Gemma 4 (0.108 for 31B, 0.377 for 26B-A4B).
- HYPOTHESIS: "FP8 KV is near lossless" is not settled. The quantizer differs. I did not test it.

## Prior work since 2026-10-30 (FACT, read-only checks on 2026-10-31)
- vLLM PR 60372: open, mergeable, head 3a01ead, updated 2026-10-29. FA PR 199 (SM90 FP8 MMA, head dim 512): merged 2026-10-29.
- vLLM 41944 (V weights only): open. SGLang 46965 (FA4 SM90 FP8 KV): open. vLLM 53175: merged 2026-09-24. vLLM 45434: open, stale.
- I found no work that stores the Gemma 4 K=V cache once. I searched titles in vLLM, SGLang, TensorRT-LLM and FlashInfer, and ran two web searches. This is not proof of absence.

## Recommendations
1. First prototype test: a kernel microbenchmark. No engine. One H100 SXM, about 30 min, about $2 to $3 (the image pull alone took 11 min in E-50). It needs approval and a live balance check.
2. Compare three kernels at B = 19 / 32K and B = 11 / 64K, with 24 and 16 graph slots. The kernels are stock FP8 dequant, stock FP8 MMA (FA PR 199 pin) and compact FP8. Use 32 splits and automatic splits.
3. Metric: sigma = 1 - t_compact / t_stock on the time that grows from 2K to X.
4. Kill line: sigma below 0.155 at 32K and below 0.125 at 64K means S below 5%. Stop the layout work then. The idea becomes a capacity feature (1.39 to 1.47 times the requests), and FP8 already gives 1.96 times.
5. Continue to the accuracy gate only if sigma is at least 0.25 (S about 8%).
6. Preregister that run first. Include r*, the post-PR 60372 baseline and QK at 512 dims. Require that the compact kernel beats the best stock FP8 kernel, not only the dequant one.
7. Accuracy gate (before any engine work): use real Gemma 4 activations at 16K to 64K. Compare compact and standard attention output per layer in FP8 storage. Pass if the extra error is below the FP8-against-BF16 error. Then run GSM8K with n of at least 200 and a needle test at 32K.
8. Page-size plan (HYPOTHESIS): a compact global token uses 4 x 640 = 2,560 B. The 524,288 B page holds 204.8 tokens, so the unify step may fail. A 192-token block wastes 6.25%.
9. After PR 60372 merges, rerun one m31f8 group at 32K and 64K (about $2 to $3) so the baseline is current.
10. Upstream: do not report the dequant path now, because PR 60372 and FA PR 199 replace it. A short data comment on PR 60372 is worth a draft after the post-merge rerun. The author must not post. The user decides.
11. The comment would add three facts. First, H100 numbers at 32K and 64K near the pool limit. Second, the FA3 FP8 sliding layers cost 7.8 us (B = 19) to 9.4 us (B = 11) per sequence and layer, against 7.7 to 8.3 us for BF16 FA4 (B = 4 to 10). FP8 bytes give no gain there, and these layers use 15% to 20% of the step.
12. Third, the BF16 head-dim-512 split choice leaves about 40% of the SMs idle at B = 5 and 10.
