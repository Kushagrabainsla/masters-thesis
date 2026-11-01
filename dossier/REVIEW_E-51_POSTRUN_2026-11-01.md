# R-70: post-run review of E-51 (compact K=V decode attention kernel, I-182)

Date: 2026-11-01. Reviewer: Sonnet (a different model from the author).
VERDICT: CONFIRM_WITH_CORRECTIONS (the numbers and the frozen decision are confirmed; two interpretive statements need correction)

Labels: FACT (I read a file, ran a script or recomputed a number), INTERPRETATION (my reasoning), HYPOTHESIS (not tested).
I changed no repo file except this one. I did not read `.env`. I made no Runpod or cloud call. I posted nothing.
I used read-only `gh` queries (FA4 source at the pin b273a02, FA PR 164 and 199, vLLM PR 60372, the FlashMLA FP8 note), 7 web searches and 2 page fetches.
I did not run `test_e51.py` (it writes files into the repo). I did not compile a kernel (this Mac has no Triton or GPU).
Scratch files: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/r70/` (`recompute.py`, copies of the FA4 source).
Data: `experiments/gemma4_kv_kernel/results/e51-h100-20261101/e51/` and the 4090 rehearsal. The code is the a1e8b13 version. No code changed after it.

## Key findings
- FACT: My script (it does not import the author's code), `analyze_e51.py` and the stored `summary.json` agree on every number and on INVALID (V3, V4). No number in README Section 13 differs.
- FACT: If V3 and V4 are ignored, the A1 rule gives INCONCLUSIVE in 5 of 5 repeats. The rule before A1 gives KILL. So R-69 R2 stopped a false KILL.
- FACT: The compact Triton kernel is a poor twin. Its FP8 time is 1.29 to 1.39 times its BF16 time at half the bytes. It loses to the standard kernel at all 16 matched FP8 configurations. The BF16 pair follows the byte model (sigma_layout 0.34 and 0.37 against 0.375).
- FACT: The stock FA4 FP8 kernel moves K and V through one 32 KB staging buffer with one stage. "Bound on chip" in Section 13 hides this serialized chain.
- INTERPRETATION: The K=V oracle measures DRAM bytes only. The real layout also cuts the FP8 cast work by 37.5%, and a V-centred kernel stages fewer bytes per block. So E-51 bounds the byte part of the gain (about 0.2 x 0.375 of dt). It does not bound the layout gain.
- INTERPRETATION: E-51 does not show that the layout fails. It shows that bytes alone give at most 10% of dt on two FP8 dequant kernels. The FP8 MMA kernel (FA PR 199, in the open vLLM PR 60372) is untested.
- Recommendation: do not take D-85 (a) as a kill. Take (b) with a pool probe, a BF16 oracle control and fixed V3 and V4 (about $2). Do the free half of (c) first.

## Answers

### 1. V1 to V6, the frozen decision, and A2
- FACT: V1 passes (NVIDIA H100 80GB HBM3). V2 passes. The stock s32 error is 0.00166. All 16 configurations of each Triton FP8 family are within 0.01 (worst 0.00189).
- FACT: V3 fails. The ratios to the E-50 targets are 0.8366 (19:2K), 0.9928 (19:32K), 0.8634 (11:2K) and 0.9745 (11:64K). The dt ratios are 1.025 and 0.986. Only 19:2K is outside 0.85 to 1.15.
- FACT: The V3 result at the 2K cells depends on the measurement. The sweep rows give 0.8528 (19:2K) and 0.8481 (11:2K). The confirmation medians give 0.8366 and 0.8634. Pass and fail swap. The 0.85 bound sits inside the 2% noise of the check.
- FACT: V4 fails. The median spreads are 0.0419 (stock 19:32K) and 0.0270 (stock 11:64K). The other 10 decision rows pass (0.003 to 0.019). The Triton standard rows are close to the bound (0.017 to 0.019).
- FACT: V5 passes (largest X row 2,270 GB/s, largest dt rate 2,354 GB/s, bound 3,501). V6 passes with 0 hard-throttle samples in 137 busy samples.
- FACT (not in Section 13): 20 of the 137 busy samples (14.6%) show the software power cap (0x4). Their mean SM clock is 1,918 MHz (-3.1%). The lowest clock is 1,530 MHz. V6 counts only the hard bits (0x8, 0x20, 0x40, 0x80), so it cannot see the cap.
- FACT: INVALID (V3, V4) is correct under A1 and A2. `decide()` returns on a failed check before it applies the rule.
- FACT: Commit a1e8b13 is dated 03:12 UTC. The Pod was created at 03:14 UTC. The uploaded SHA list equals the local list. A2 used rehearsal data only. A test covers it, and a stub test covers the failing V5 case.
- FACT: A2 changed one validity check and the timing procedure. It did not change the rule, the metrics or the cells. It did not change the H100 outcome. The old V5 (all decision rows, with the 2K rows) gives a largest rate of 2,270 GB/s here and passes too.
- FACT: After the warm-up pass, repeat 0 is +0.18% on average against the median of the other four repeats (range -2.0% to +3.6%).
- INTERPRETATION: A2 was legitimate. Two gaps remain. A2 had no second-model review. The rehearsal could not run the stock rows, so the 2% V4 bound came from a GPU without a power cap.

### 2. Exploratory numbers (medians of 5 repeats; dt is the median of per-repeat differences)
| Quantity | 32K, B 19 | 64K, B 11 |
|---|---|---|
| dt: stock s32, Triton std, Triton compact (us) | 1147.80, 1015.58, 1305.36 | 1344.93, 1230.51, 1585.67 |
| sigma_layout, sigma_dq, sigma_mma | -0.2853, -0.1373, -0.3380 | -0.2886, -0.1790, -0.3871 |
| dt rate / copy (3,044.6): stock, std, compact | 0.684, 0.773, 0.376 | 0.699, 0.764, 0.370 |
| oracle dt: stock, std (us) | 1031.74, 924.93 | 1233.76, 1110.06 |
| e_stock, e_triton | 0.202, 0.179 | 0.165, 0.196 |
| 5-repeat range of sigma_layout | -0.297 to -0.280 | -0.294 to -0.286 |
| 5-repeat range of e_stock, e_triton | 0.162 to 0.222, 0.177 to 0.186 | 0.164 to 0.174, 0.187 to 0.201 |
- FACT: No number in Section 13 differs. I also checked the 4090 e_triton (0.8849 and 0.9909), the B 4 times, the BF16 sigma_layout (0.3374 and 0.3675), the 27 of 392 rows above 2% (17.0% at most) and the 40 correctness errors.
- FACT: The standard Triton kernel beats the stock kernel on dt in all 5 repeats (0.879 to 0.917). The Triton family had 16 configurations to choose from, and the stock family had 2.
- FACT: The best Triton configuration (32 splits, 4 warps, tile 32) sits at the edge of the grid in three dimensions. This holds for both kernels and both groups. `num_stages` stayed at 2. The family is not tuned.

### 3. The oracle premise
FACT, stock kernel (vllm-project/flash-attention b273a02; `flash_fwd_sm90.py` lines 152, 331, 883 to 935; `interface.py` lines 178, 851, 1003 to 1016, 1627):
- The FP8 dequant path stages K and V through ONE 32 KB buffer (64 tokens x 512 FP8). The staging pipeline has one stage.
- The K and V buffers have `num_stages=1` above head dim 256. The tile is 64 x 64.
- FA4 packs the 8 query heads of a KV head by default. So M = 64 holds at most 8 real rows (INTERPRETATION: the vLLM wrapper keeps the default).
- The producer stages K[n], then V[n+1]. Each TMA waits until the consumers have cast the previous tile.
- The FP8 load path has no `cache_hint` or `evict` string.
- INTERPRETATION (arithmetic, one CTA per SM): one tile slot takes about 2.08 us (1,148 us for 276 blocks and 2 tiles on one SM). K[p] and V[p] sit two slots apart (K[p], V[p+1], K[p-1], V[p]). The gap is about 4.2 us.

Does K = V halve the DRAM bytes on the H100?
- (a) FA4. INTERPRETATION: the V read most likely hits L2. About 8.7 MB of new lines pass in 4.2 us. That is 17% of the 50 MB. The same SM issues both reads. So the two L2 partitions cannot turn the second read into a DRAM read. They can slow the hit: the far partition takes nearly twice the near latency on a PCIe H100 (Chips and Cheese).
- (b) Triton. INTERPRETATION: the source loads V, then K, in one loop iteration. The two loads of the same bytes are a few instructions apart. L1 or L2 merges them. DRAM bytes halve.
- FACT: E-51 has no DRAM counter. Both statements are untested.
- INTERPRETATION: three small effects push e down. (i) The oracle reads only the V half of each 1,024-byte token row (512-byte runs with 512-byte holes). (ii) The 2K alias row at B 11 (46 MB) is close to the L2 size and shifts e by at most 0.02. (iii) In a one-stage chain an L2 hit shortens the wait, but the issue, the cast and the barriers stay.

What the oracle cannot see:
- FACT (code): K = V keeps the same casts, MMA, staged tiles, shared-memory fill and L2-to-SM bytes. It removes DRAM bytes only.
- FACT (arithmetic): per KV token and SM, the stock kernel takes 128 clocks on dt. The DRAM floor at copy bandwidth is 88 clocks. The wgmma floor at M = 64 is 32 clocks.
- INTERPRETATION: if the second read were free, a serialized chain would give a gain far above 10%. A fully overlapped on-chip limit would give about 0%. The data (10%) lie between. This is why the controls below matter.

Cross-checks:
- FACT: Stock BF16 (page 64, TMA) and Triton std BF16 read 1.02 to 1.05 x copy on dt. So DRAM and the L2-to-SM path carry 3.1 TB/s at these shapes. The FP8 kernels need at most 2.4 TB/s. An L2 bandwidth limit cannot explain e = 0.2.
- FACT: The FP8/BF16 dt ratio is 0.747 and 0.751 (stock) and 0.666 and 0.681 (Triton std), for half the bytes. These are dtype elasticities of 0.50 and 0.64 to 0.67. They differ from the K=V elasticity (0.17 to 0.20).
- FACT: The compact kernel takes 1,305 us in FP8 and 1,010 us in BF16 (32K). At 64K it takes 1,586 us and 1,143 us. FP8 is slower at half the bytes. This is the cleanest sign that the FP8 path, not DRAM, sets the FP8 times.
- FACT: The 4090 and the H100 do not conflict. The same code has the same registers (254 or 255 at 4 warps) and shared memory (50,176 and 57,344 B) on both. The 4090 has 0.92 TB/s of DRAM, so the kernel was DRAM-bound there (e 0.89 and 0.99). No L2 argument is needed.

Alternatives:
- H2 (the oracle fails on the H100): possible, unlikely, for the reasons above. It is the only alternative to the value of e.
- H3 (the strongest alternative to the conclusion): the oracle is valid but is not the layout. A compact kernel casts 640 values per token instead of 1,024. It can stage one V tile per block and a small K_rot tile, instead of K and V. The oracle keeps both full tiles. In a cast-bound or chain-bound kernel the layout gain can exceed 0.375 x e.
- HYPOTHESIS: a crude additive model has a byte gain of 0.075 of dt, a cast share of 6% to 25% of dt cut by 37.5%, and a QK penalty of 0 to 5%. It gives a layout gain of 5% to 17%. The range reaches the kill lines (0.155 and 0.125) only at its top. The staged-tile saving is not in the model.

Cheap measurements that separate them (about 1 min of GPU, same harness):
1. Pool probe. Build the block table from a small pool of physical pages (8 MB), reused by every sequence with a different permutation. After the first call, the bytes come from L2. If the FP8 time falls by under 10%, DRAM never set the time. The fp32 partials (40 MB at B 19) pass through L2 too, so the probe may under-report the DRAM share. Run the BF16 rows as the calibration.
2. BF16 K=V rows for stock BF16 and Triton BF16 std. HYPOTHESIS: e_bf16 is 0.6 to 1.0 if the oracle is valid (the L2-to-SM path may cap it below 1). It is below 0.4 under H2.
3. `ncu` `dram__bytes_read.sum` for K != V and K = V, if the Pod allows counters. Try it. Do not depend on it.
These three separate H1 from H2. Only a compact kernel can test H3: use the Q4 variants first, and later a CuTe kernel that stages V once.

### 4. The compact Triton FP8 kernel
- FACT: At the 16 matched FP8 configurations the compact time is 1.04 to 1.52 x (32K) and 1.05 to 1.55 x (64K) the standard time. The medians are 1.30 and 1.33. At the 4 matched BF16 configurations it is 0.66 to 0.79 x and 0.63 to 0.80 x (the byte ratio is 0.625).
- FACT: Rows per second on dt (1e9): FP8 compact 1.79 and 1.76, BF16 compact 2.31 and 2.44, FP8 std 2.30 and 2.27, BF16 std 1.53 and 1.55. The FP8 compact kernel is the only one that loses to its own BF16 version.
- FACT: Registers and shared memory at 4 warps and tile 32. FP8: std 255 and 50,176 B, compact 255 and 57,344 B. BF16: std 151 and 82,944 B, compact 133 and 62,464 B.
- FACT: At 4 warps and tile 64 the kernels spill 26 (std) and 74 (compact) registers. At 8 warps and tile 32 they use 128 (std) and 172 (compact) registers.
- INTERPRETATION, most likely cause: after the FP8 cast, the V tile sits in registers and feeds two dot roles (`tl.dot(q, tl.trans(v))` and `tl.dot(p, v)`). Each role needs its own operand layout. Each conversion costs a shared-memory round trip and a barrier.
- INTERPRETATION: the standard kernel converts each tile once. In BF16 the tile feeds both roles from the async-copied shared buffer, with no register step. The shared-memory numbers fit this: FP8 compact needs more than std, and BF16 compact needs less.
- INTERPRETATION, second cause: q (about 128 registers), qr (about 32) and the accumulator (64) press on the 255-register ceiling. This limits scheduling. The zeroed 128 columns of `qc` add 25% to the QK MMA work (12.5% of all MMA work).
- HYPOTHESIS, third cause: the V load has two dot users, so the pipeliner may not use `cp.async` for it. Test: count `cp.async`, `bar.sync`, `ld.shared`, `st.shared`, `cvt` and `mma.sync` per loop iteration in the PTX.
- INTERPRETATION: this is a defect of the implementation, not of the layout. BF16 compact works. MLA decode kernels use the same shared-tile design. The extra QK work is avoidable (slice `qc` to the 384 unrotated dims). A CuTe kernel casts V to shared memory once, and wgmma reads that tile in both roles.
- Cheap tests: compile both kernels offline for sm_90 on any x86 Linux host (HYPOTHESIS: `triton.compile` with an explicit `GPUTarget` needs no GPU). Then time three variants: drop `kr`, load V twice (once per role), slice `qc`.

### 5. Do the V3 and V4 failures change what the data support?
- INTERPRETATION: They void the frozen decision. They do not change the exploratory ordering. The 5-repeat ranges are far from every line. sigma_layout is at most -0.280 against +0.155 and +0.125. sigma_mma is at most -0.330 against +0.25. e is at most 0.222 against 0.413 and 0.333.
- FACT: With V3 and V4 forced to pass, all 5 repeats give the word INCONCLUSIVE. The word does not depend on the noise.
- INTERPRETATION: V3 at 2K is a calibration error of the check. dt ties within 2.5%. The 2K time holds fixed costs that the engine has and the microbenchmark lacks (graph padding to 24 and 16 slots, other kernels). The README predicted "a few percent". The shortfall is 14% to 16%.
- INTERPRETATION: The power cap hits the fast kernels (stock and std, 2.1 to 2.4 TB/s) more than the slow compact kernel. So the bias favours the compact kernel. The observed sigma points the other way, so the bias cannot change the sign.
- FACT: The mean clock loss over busy samples is -0.5%. I have no per-replay clock log.
- FACT: Order effects are small. The 2K rows move up to 2.7% between forward and reverse repeats. The X rows move up to 0.8%.

### 6. Scope
Can claim (FACT, with scope):
- On one H100 SXM, with the FA4 FP8 dequant kernel of vLLM 3966924d5 (32-split cap), at B 19 / 32K and B 11 / 64K, head dim 512, 64-token pages and random FP8 data, the K=V probe cut dt by 8% to 10%.
- The author's standard Triton FP8 kernel is 8.5% to 12% faster on dt than the stock kernel in this setup.
- The author's compact Triton FP8 kernel is 29% slower than its twin. The BF16 pair shows sigma_layout 0.34 and 0.37.
- The capacity result of I-182 (1.39 to 1.47 x the requests) is unchanged.

Cannot claim (INTERPRETATION of the evidence):
- A decision. The run is INVALID.
- That FP8 decode attention is on-chip bound on the H100 in general. The vLLM FP8 KV post reports a decode slope of 54% of BF16 for Llama-3.1-8B with FA3. The E-51 finding is for the head-dim-512 dequant kernel.
- That the compact layout cannot be faster. No competent compact kernel exists. The oracle is not the layout.
- Anything about the FP8 MMA kernel (FA PR 199), the post-PR 60372 split choice, an H200 or B200, other batches, mixed lengths, real data or the engine step.
- Any statistic across GPUs or sessions (n = 1 GPU, 1 session).

### 7. Prior work (7 searches, 2 fetches, read-only `gh`)
Found:
- FACT: The FlashMLA Hopper FP8 sparse decode note calls its kernel "dequantization-bound". It needs about 50 clocks per token for the cast against 34 for MMA on an H800. It halves the cast work per CTA with a cluster. It is a different kernel (MLA, 4-step cast). FA4 uses a one-instruction cast, so the numbers do not transfer. The mechanism does.
- FACT: The FA PR 199 body (H200, head dim 512): the FP8 MMA kernel runs at FP16 speed at B 1 and 16 (0.77 against 0.79 ms at 16K). At B 64 and 28K it is 18% faster than FP16 for half the bytes (about 2.4 TB/s, half the H200 peak). R-68 used this table.
- FACT: vLLM PR 60372 (open) reports 1.04 x end to end on one H200 for FP8 MMA plus the split change. Multimodal Gemma 4 with FP8 KV uses Triton.
- FACT: The vLLM FP8 KV post (2026-04-22) reports for an H100 with FA3 a decode slope of 54% of BF16 for Llama-3.1-8B and 80% for gpt-oss-20b. This is counter-evidence for a general claim. It has no head-dim-512 data.
- FACT (from search summaries; I did not open the pages): QServe reports that KV4 dequantization alone reaches the CUDA-core roofline on an A100. arXiv 2502.01070 reports under 25% FP8 decode gain on an H100. SGLang issue 10083 describes the cost of dequantization at read time.
Not found (absence is not proof):
- Any public K=V oracle or equal byte-elasticity measurement for FA3 or FA4 at head dim 512. Any compact K=V kernel for Gemma 4. Any `ncu` data for the FA4 FP8 dequant path. Any FlashInfer note on this.

### 8. Next step for D-85
- Recommendation: (b), changed as below, after the free half of (c). Do not take (a) as a kill now.
- Why not (a) (INTERPRETATION): E-51 is INVALID. The oracle premise has no control. The kernel that E-51 measured (dequant) is the one that PR 60372 would replace. A stop would rest on a probe that is not the layout.
- If the user wants to stop for time, write "speed claim untested", not "no speed benefit". The capacity result stands.
- Why not (c) alone (INTERPRETATION): it fixes a Triton kernel that is not the production candidate. Its offline half is free, so do that half first (Q4).
- Why (b) (INTERPRETATION): it measures the kernel that vLLM would ship. A kernel with no cast leaves the layout close to byte-only. There the K=V oracle is a fair probe.
- HYPOTHESIS: (b) ends in KILL. Its value is a controlled stop on the kernel that matters, for about $2.

Design of (b) (one H100 session, about 30 min, about $2, cap $3; no Pod without the user's approval and the live balance check):
- Rows: the dequant stock row (the tie to E-50), the FP8 MMA rows (PR 199 overlay, s32 and auto) with K=V, the BF16 K=V control (stock BF16 and Triton BF16 std), and the pool probe for each kernel. Drop the FP8 Triton sweeps. They answer nothing new for (b).
- Rehearse the harness changes (pool probe, robust V4) on the 4090 with Triton rows (about $0.15).
- Check that the overlay is live: a different kernel name in a profile, dt_mma / dt_dequant of 0.80 to 1.00 (PR 199 table: 0.87 to 0.93), and a correctness bound set from the PR 199 tests. FP8 P loses accuracy, so the E-51 bound of 0.02 may not fit.
- Stop rule: if the overlay does not run within 10 min of Pod time, run the BF16 control and stop.

Kill line for (b):
- e = 2 x (1 - dt_oracle / dt) on the FP8 MMA kernel, per group, median of 5 repeats. sigma = 0.375 x e. S = 0.3227 x sigma (32K) and 0.4 x sigma (64K).
- The kill lines (sigma 0.155 and 0.125) need e of 0.413 (32K) and 0.333 (64K). The 8% line (sigma 0.25) needs e of 0.667. At the E-51 e of 0.2, S is 2.4% and 3.0%.
- Control: e_bf16 from the BF16 K=V rows. If e_bf16 >= 0.5, use e_corr = e / e_bf16 (at most 1), and flag the run if e_bf16 is below 0.8. Below 0.5 the oracle is invalid on this GPU, and no kill comes from it.
- The correction over-corrects if the L2 path caps the BF16 oracle. That makes a KILL harder, which is the safe direction.
- KILL (byte route): the upper end of the 5-repeat range of e_corr is below 0.413 at 32K and below 0.333 at 64K.
- BUILD PLAN: the lower end is at or above 0.413 (32K) or 0.333 (64K) in at least one group. This allows the accuracy gate and a compact-kernel plan only.
- Else INCONCLUSIVE, which means stop. Write this now.
- Report 1 - T_pool / T as a diagnostic of the DRAM share of the time. It may under-report (see Q3).
- HYPOTHESIS: e_mma falls below both lines. The PR 199 table gives a dtype elasticity of 0.365 for the MMA kernel (H200, B 64). In E-51 the dtype elasticity of the stock kernel (0.50) was 2.5 times its K=V elasticity (0.20).
- Scope: the kill stops the byte route. It does not exclude a gain from fewer staged tiles in a new kernel. Say this in the preregistration.

Fixes to V3 and V4:
- V3: keep the X cells and dt within 15% of E-50 (E-51: 0.993, 0.975, 1.025, 0.986). Report the 2K ratios without a bound. Or add the padded slots (24 and 16) and keep the bound. The FP8 MMA row has no E-50 target, so tie it to the dequant row in the same session.
- V4: the max/min spread over 15 replays reacts to one outlier. Use 30 replays and the spread of the central 20. Bound the across-repeat range of dt at 3% (E-51: 2.7% and 0.6% for the stock rows).
- V4: log the SM clock per replay set with NVML (no root needed). Flag rows with a mean clock below 1,900 MHz.
- V6: count the cap (0x4) as a separate line. Report its share of busy samples (E-51: 14.6%).

## Corrections
1. Section 13 "What this means", journal E-51 and D-85, O-13: state that the K=V oracle measures DRAM bytes only. Replace "a byte-only layout cannot reach S = 5%" with this: bytes explain at most 20% of dt in two FP8 dequant kernels; the compact layout also cuts cast work and staged bytes; E-51 did not measure that. Change D-85 (a) to "stop only if (b) and the BF16 control agree".
2. Same places: "bound mostly on chip" is an INTERPRETATION. Name the candidates (cast, MMA with M padded from 8 to 64, the one-stage staging chain). Add the source fact.
3. Add the V3 noise-edge fact (0.853 and 0.848 on the sweep rows, 0.837 and 0.863 on the confirmation medians).
4. Add the power-cap fact (14.6% of busy samples, mean clock 1,918 MHz, lowest 1,530 MHz). Say that V6 cannot see it. Extend L-17 with this V6 blind spot.
5. Replace "a defect of its own (0.37 x copy against 0.77)" with the matched-configuration and FP8-against-BF16 numbers of Q4.
6. Name three elasticities: dtype (0.50 stock, 0.64 to 0.67 Triton std), K=V (0.17 to 0.20) and real layout (unmeasured). Say that the 4090 was DRAM-bound.
7. Add that the Triton optimum sits at the grid edge and `num_stages` was not swept. The std ceiling is unknown.
8. New lesson: a rehearsal on a GPU with 3.3 x less DRAM bandwidth cannot show on-chip defects. S5 gate 4 passed the compact kernel at 1.01 and 0.99 x copy on the 4090. The H100 shows 0.37. Add a gate on rows per second, or on an L2-resident shape.

## Recommendations
1. D-85: choose (b) as designed in Q8. Do the offline PTX comparison first. Do not choose (a) as a kill.
2. Preregister E-52 before any spend: the rows, the kill line, the controls, the V3 and V4 fixes, and the meaning of INCONCLUSIVE.
3. If (b) says BUILD PLAN, the next object is a compact kernel that stages V once (CuTe, FA4 style). Add the accuracy gate (R-68 recommendation 7).
4. If (b) says KILL, write I-182 as a capacity result plus the corrected O-13. Do not claim "no speed benefit". Claim "bytes alone give at most 10% of dt".
5. If (c) goes ahead, widen the Triton sweep: splits 64, warps 2, tile 16, `num_stages` 3.
6. Do not post upstream now (R-68 recommendation 10 stands). Check whether PR 199 keeps the one-stage staging before any comment.

## Checks I ran
- FACT: reran `analyze_e51.py` (identical `summary.json`). Wrote `recompute.py` (V1 to V6, dt, sigmas, rates, oracle e per repeat, order effects, old V5).
- FACT: ran the author's `label()` and `decide()` with V3 and V4 forced to pass: INCONCLUSIVE, 5 of 5 repeats. The pre-A1 rule gives KILL.
- FACT: recomputed the 4090 numbers, the telemetry (215 samples, 137 busy), the matched-configuration ratios and the E-50 V3 targets (238.3, 1,358.1, 143.2, 1,507.2).
- FACT: read `flash_fwd_sm90.py`, `interface.py` and `copy_utils.py` at b273a02, FA PR 164 and 199, vLLM PR 60372, and the FlashMLA FP8 note.

## Sources
- FlashMLA note: https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20250929-hopper-fp8-sparse-deep-dive.md
- FA4 source: https://github.com/vllm-project/flash-attention/tree/b273a02/flash_attn/cute
- FA PR 199: https://github.com/vllm-project/flash-attention/pull/199 and PR 164: https://github.com/vllm-project/flash-attention/pull/164
- vLLM PR 60372: https://github.com/vllm-project/vllm/pull/60372
- vLLM FP8 KV post: https://vllm.ai/blog/2026-04-22-fp8-kvcache
- H100 L2 (Chips and Cheese): https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth
- QServe: https://arxiv.org/abs/2405.04532. FP8 TCO: https://arxiv.org/abs/2502.01070. SGLang issue: https://github.com/sgl-project/sglang/issues/10083
