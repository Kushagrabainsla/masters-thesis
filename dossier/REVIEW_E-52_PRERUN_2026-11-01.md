VERDICT: AMEND_THEN_RUN

# R-71: pre-run review of E-52 (fixed compact K=V kernels, FP16 compute, two overlapping views)

Date: 2026-11-01. Reviewer: R-71 (a different model from the author). Design commit: 2931aba. Package: `experiments/gemma4_kv_views/`.

Labels: FACT (I read a file, ran code or recomputed a number), INTERPRETATION (my reasoning), HYPOTHESIS (not tested).

I changed no repo file except this one. I did not read `.env`. I made no Runpod or cloud call. I posted nothing. The tests wrote only to temporary folders, and `git status` shows no new file in the repo.

I used read-only `gh` queries (Triton v3.8.0, vLLM 3966924d5, flash-attention b273a02, FlashMLA) and web fetches. I did not compile a kernel. This Mac has no Triton and no GPU.

## Summary
- FACT: the views layout is exact. My independent script agrees to 1e-15 in float64. The 19 tests pass.
- FACT: the Triton 3.8.0 source confirms the conversion sequences that the author quotes.
- Two blockers change the frozen rule. They need amendment A1 before any spend: B1 (the KILL word) and B2 (V4 on the stock row).
- E-52 stays the right next test after D-85. It tests the byte route only (Answer 7).
- D-86 holds only "if R-71 changes neither the plan nor the cost cap". The amendments change the plan. The user must confirm the amended plan.

## Blockers

### B1. The KILL word can fire for the wrong reason
- Where: `README.md` lines 13-14, 42-44, 53, 144-149. `analyze_e52.py` lines 52, 229-230, 289-293.
- FACT (algebra): the views kernel runs the same code as its twin. So its rate on 640 bytes equals 0.625 x rate_std / (1 - sigma_layout).
- FACT: KILL needs that rate at 0.6 x copy or more and sigma_layout below the kill line. So KILL needs a std rate of 0.81 x copy (sigma_layout 0.155), 0.84 (0.125) or 0.96 (sigma_layout 0).
- FACT: I ran the author's `label()` on synthetic dt values (`/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/r71/rule_grid.py`).
- FACT: if the views do not share their DRAM reads, the word is INCONCLUSIVE for a std rate below 0.96 x copy. At 0.96 or above, the word is KILL. If they share, the word is GO.
- Failure scenario: FP16 compute lets the std kernel reach 0.97 x copy. The second read of the overlap goes to DRAM. The word is KILL. The record then says "the layout gives too little". That statement is false, because a kernel that stages V once still reads only 640 bytes.
- INTERPRETATION: a byte-limited kernel cannot give KILL when sharing works. If the K = V elasticity e_h is 0.6 or more, sigma_layout is 0.225 or more. That value is above both kill lines. So KILL can only mean "this two-load implementation did not realize the byte saving".
- FACT: for a std rate below 0.96, the same fault gives INCONCLUSIVE with the text "on-chip limit" (line 292). That text is wrong for unshared reads.
- INTERPRETATION: the 0.6 x copy test was a competence test for a different compact kernel (E-51 A1 R2). The twin has the competence of the std kernel. The competence test belongs on the std kernel and on the sharing evidence.
- Fix (amendment A1, before any data):
  1. Remove the KILL word from the views test. Map the old KILL region to INCONCLUSIVE.
  2. Add three diagnosis flags to `summary.json` and to the label text. (a) "time does not follow bytes": e_h below 0.413 at 32K and below 0.333 at 64K. (b) "views do not share": e_h at 0.6 or more and sigma_layout below 0.25 x e_h. (c) "L2 path or on-chip work caps": the pool share of the views kernel is below 0.3.
  3. Write the consequence of each INCONCLUSIVE case in the README. For case (a), write "the byte route is closed for the Triton FP16 family; a kernel that stages V once is not tested". This is the R-70 scope sentence.
  4. If the user wants a KILL word, define it on case (a) only, as R-70 did for option (b). Add "stops the byte route only".
  5. Update `README.md` lines 13-14, 42-44, 53 and 144-149, the journal E-52 entry and D-85.

### B2. V4 can void the run again on the stock row
- Where: `README.md` lines 130-133. `analyze_e52.py` lines 48-49 and 352-359.
- FACT (E-51 data, recomputed): the stock row at 19:32K had a median spread of 4.19% and an across-repeat dt range of 2.71%. The stock row at 11:64K had 2.70% and 0.63%. The Triton std rows had ranges of 1.02% and 0.52%. The compact rows had 0.26% and 0.16%.
- FACT: E-52 keeps the stock rows in V4 with a 2% spread bound and a 3% range bound. The E-51 stock range sits at 90% of the bound.
- FACT: E-51 stored no per-replay times. Nobody checked the new statistic (the central 20 of 30 replays) on the row that failed in E-51.
- HYPOTHESIS: the stock rows alone give a 25% to 40% chance that V4 voids the run. A void costs about $2 and one more day.
- INTERPRETATION: the stock row is the baseline, not the test object. V3 (15% tie to the engine) and the 4-of-5 replication rule already guard it.
- Failure scenario: all Triton rows are clean, the decision replicates, and V4 returns INVALID because of one stock row at 3.2%.
- Fix (amendment A1, before any data):
  1. Apply the 2% spread bound and the 3% range bound to the two Triton families only. They are the test objects.
  2. For the stock rows, report the spread and the range. Bound the stock dt range at 5%.
  3. Time the stock rows with 7 repeats. This adds about 20 seconds.

## Other fixes

### F1. Compare the two layouts at one configuration
- Where: `README.md` lines 107, 110, 157 and 189. `analyze_e52.py` lines 138-146 and 239-276.
- FACT: `select()` picks the fastest configuration of each family on its own. The README calls the two families "the same compiled kernel" (line 157).
- FACT (E-51 data): both families chose 32/4/32 in both groups. So the effect was zero in E-51. The E-52 grid is 3.4 times larger (HYPOTHESIS: it has more near-ties).
- INTERPRETATION: if the families select different configurations, sigma_layout holds a configuration effect. A 64-split configuration also has empty splits at 2K, so its fixed cost differs. I estimate 1% to 3% of dt.
- FACT (arithmetic): the views row at 11:2K reads 57.7 MB (11 x 2,048 x 4 x 640 bytes). The H100 L2 holds 50 MB. The same row read at 1.15 x copy on the RTX 4090 in E-51 (A2). The std row reads 92 MB.
- INTERPRETATION: part of the views row can hit in L2 over the 20 calls. That lowers t(2K) and raises dt of the views at 64K. The bias lowers sigma_layout by about 1% to 3%. It pushes toward PIVOT and costs nothing to remove.
- Fix: define the primary sigma_layout at the std selection for both layouts. Report the independent selection as a second value. Report the 11:2K views time against its prediction from the 19:2K row (scale by tokens). Flag the result if it is more than 3% lower.

### F2. Run the free offline sm_90 compile before the spend
- Where: `README.md` Section 2 and Section 12; journal D-85.
- FACT: R-70 (Q4) and D-85 name an offline sm_90 compile and a PTX diff as the free half of option (c). The package holds a source read and a CPU emulation. It holds no compile. No PTX of the FP16 or views kernel exists before the spend.
- INTERPRETATION: `triton.compile` with an explicit `GPUTarget("cuda", 90, 32)` needs no GPU (R-70 HYPOTHESIS, not tested here). It runs on any Linux host with the Triton wheel. Docker Desktop is installed on this Mac but was stopped when I checked.
- Failure scenario: a Triton or ptxas abort in one of the 54 configurations ends the process and the data. The risky settings are 2 warps, tile 16, stage 3 and 64 splits. The 4090 rehearsal compiles for sm_89. It cannot show faults that only occur on sm_90.
- Fix: compile the 54 configurations and the 6 diagnostic kernels for sm_90 offline. The cost is $0. Check six items:
  1. `cvt.rn.f16x2.e4m3x2` is present and `cvt.bf16.f16` is absent in the FP16 kernels.
  2. The std call and the views call give the same PTX.
  3. Registers, spills and shared memory for every configuration.
  4. No compile abort.
  5. The SASS instruction that `cvt.bf16.f16` becomes. This settles the H4 rate class (Answer 3).
  6. Whether `cp.async` appears at `num_stages` 3 and not at 2 (F3).
  Record the result in README Section 12.

### F3. Separate the cast effect from the stage effect
- Where: `README.md` lines 77-78 and 162-163. `analyze_e52.py` lines 394-410 (`factorial`).
- FACT (E-51 H100 rows): shared memory equals K tile + V tile + q tile + p tile. The q tile is 16,384 bytes. The p tile is 32 bytes per token. This model matches the six std rows exactly (FP8 50,176 and 83,968; BF16 82,944). A variant matches the four fused rows (FP8 57,344 with two V buffers; BF16 62,464). The E-51 kernels did not double-buffer K or V at `num_stages` 2.
- INTERPRETATION: Triton may need 3 stages to overlap loads whose address comes from the block-table load. If so, `num_stages` 3 is the first real overlap of loads and compute, and it can matter more than the cast (H5: exposed memory latency).
- FACT: the cast x layout factorial runs only at 32/4/32/2. The primary rows can select stage 3. So the factorial cannot separate cast from stage.
- Failure scenario: the selected FP16 configuration is at stage 3 and reads at 0.9 x copy. The author credits FP16 (H4). The cause is the stage. The next step is then wrong.
- Fix: time `triton_std_fp8` (BF16 compute) at the selected FP16 configuration in both groups (4 rows). Time the FP16 kernel at the same splits, warps and tile with stage 2 and stage 3 (4 rows). Report the 2 x 2 table (cast x stage).
- Rehearsal note (HYPOTHESIS, from the model above): at stage 3, a tile of 64 needs about 150 KB with two buffers. With three buffers it needs 215 KB. A tile of 32 needs 83 KB or 116 KB.
- The 4090 limit is 99 KB. So up to 18 configurations per family fail there with an out-of-resources error. The code records them and goes on. No configuration fails on the H100 for shared memory (limit 227 KB). The rehearsal cannot test the likely best H100 rows.

### F4. Add a deadline to the wide sweep
- Where: `kbench_e52.py` lines 50, 105-112 and 624-631. `session_e52.sh` lines 14-15 and 78.
- FACT (stub run of the same flow): the sweep has 628 rows (137 per group cell, 20 per reported cell). The confirmation has 40 entries.
- FACT: the run needs 65 split-kernel compiles and 3 merge compiles if the views call reuses the std binary. It needs up to 127 compiles if not.
- FACT: E-51 `kbench` took 228 s for 392 rows and 44 compiles.
- INTERPRETATION: scaling E-51 gives 10 to 12 minutes. A first-principles count gives 18 to 23 minutes. The author's 10 to 18 minutes lies inside. `KB_TIMEOUT` 1,500 s leaves a margin of 1.1 to 2.5 times.
- HYPOTHESIS: the 2-warp configurations are 2 to 3 times slower and set the upper end. In E-51, 8 warps were 31% slower than 4 warps at tile 32.
- FACT: a timeout before the confirmation leaves status "error" (lines 636-641) and no decision.
- Failure scenario: slow compiles push the sweep past 1,500 s. The Pod bills about $2.5. The result is INCONCLUSIVE.
- Fix: add a deadline. If more than 720 s have passed at the end of a group cell, run only the `LITE` rows at the remaining group cells. Then go to the confirmation.
- Fix: write the elapsed time of each phase into `run.json`. If the user wants to cut cost, drop `warps` 2 (one third of the grid).

### F5. Try a direct DRAM-bytes counter
- Where: `session_e52.sh` (no step exists). R-70 Q3 item 3.
- FACT: E-52 has no DRAM counter. Every row infers byte sharing from time. README line 182 admits this.
- Fix: add `dram_probe.py` (about 30 lines). It launches the std kernel, the alias kernel and the views kernel once each at 19:32K. Run `ncu --metrics dram__bytes_read.sum` on those launches before the sweep. If the Pod returns ERR_NVGPUCTRPERM, write the error to the log and go on. The cost is 1 to 2 minutes. A result of about 640 bytes per token for the views kernel settles the central unknown.

### F6. Fix `ptx_counts` for predicated instructions
- Where: `kbench_e52.py` lines 270-273. `test_e52.py` lines 59-65.
- FACT: `ptx_counts` tests `ln.startswith(op)`. The unit test uses sample lines without a predicate.
- INTERPRETATION: Triton PTX writes global loads, `cp.async` and many `st.shared` as predicated lines such as `@%p3 ld.global.v4.b32 ...`. These lines start with `@`. No PTX of this kernel exists yet, so I could not check it.
- Failure scenario: the `cp.async` count reads 0 although the kernel uses it. R-70 asked for this count to decide the pipelining question. A false 0 hides the answer.
- Fix: strip a leading predicate (`re.sub(r"^@!?%p\d+\s+", "", line)`) before the match. Add a test with predicated lines. Label the counts "whole kernel", because they are not per loop iteration.

### F7. Sample the NVML clock while the GPU runs
- Where: `kbench_e52.py` lines 157-168.
- FACT: the code reads the clock after `z.synchronize()`. The replay has finished at that time.
- INTERPRETATION: NVML then reports an idle GPU. The SM clock and the cap bit may not show the load. The R-70 control (flag rows below 1,900 MHz) can read false.
- Fix: read NVML right after `z.record()` and before `z.synchronize()`. Or sample from a thread during the replays. Keep the 1 Hz nvidia-smi telemetry as the second record.

### F8. Let V3 guard the GO margin
- Where: `README.md` lines 126-129. `analyze_e52.py` line 47 (`V3_TOL`).
- FACT (arithmetic): sigma_mma = 0.25 sits at dt_cmp = 0.6375 x dt_stock. A stock dt that is 15% high (V3 allows it) adds 0.13 to sigma_mma. That shift can turn PIVOT into GO.
- FACT: in E-51 the stock dt tied to the engine within 2.5%.
- Fix: add V3b. Compare the stock s32 dt with the E-51 benchmark values (1,147.8 us and 1,344.9 us). Report sigma_mma a second time with those values as the baseline.
- If the two words differ, the decision is INCONCLUSIVE ("baseline off"). A different H100 unit can differ by a few percent, so a miss is a flag and not INVALID.

### F9. Write the expected range of the controls
- Where: `README.md` line 159 and Section 8.
- HYPOTHESIS: the L2-to-SM path limits the BF16 K = V row. The second read still moves 2 KB per token from L2 to the SM at the same time. E-51 shows 3.1 TB/s through that path. So e_bf16 can read 0.5 to 0.7 although DRAM bytes halve. The README expects "near 1".
- INTERPRETATION: the views kernel needs 1,024 bytes per token from L2 at the DRAM-bound rate of 4.76 G token-heads/s. That is 4.87 TB/s. If the L2 path caps lower, sigma_layout stays below 0.375 although sharing works.
- Fix: write the expected ranges before the run (e_bf16 0.5 to 1.0). Add a BF16 pool row to calibrate the pool probe (R-70 Q3 item 1). The fp32 partials (40 MB at B 19) pass through L2 and make the pool share read low.

### F10. Repair the record and the procedure
- FACT: `README.md` line 179 says the $35 balance floor applies. Journal D-86 says a one-time $12 floor. The journal wins. Update the README.
- FACT: the boot watchdog counts from the container start. No watchdog covers the image pull. A 15-minute pull plus 34 minutes gives 49 minutes, which is about $3.26 at $3.99/h.
- Fix: add an operator rule to delete the Pod if the pull exceeds 15 minutes. Record the amendments as A1 in the README Amendments section. Ask the user to confirm the amended plan (D-86 condition).

## Optional items

### O1. Test the two-scale rule for an engine
- Where: `test_e52.py` (no test).
- Fix: add a CPU test. Store K_rot with scale s_k and V with scale s_v. Pass k_descale = s_v. Multiply the rotated part of the query by s_k / s_v. Check that the output equals the standard output. Check that K_rot fits E4M3 for a large k_norm weight.

### O2. Plan the o_proj fold for quantized weights
- INTERPRETATION: permuting the o_proj input columns inside each head is exact for BF16 weights. Block-scaled FP8 or NVFP4 weights keep one scale per 128 columns. The V-view order mixes dims 64 to 255, 320 to 511, 0 to 63 and 256 to 319, so it crosses those blocks. Plan a runtime permute or a re-quantization.

### O3. Add an early gate for the H4 question
- Fix: after the correctness cell, time four kernels on 19:32K at 32/4/32/2 and 32/4/32/3. They are the std kernel with BF16 compute, the std kernel with FP16 compute, the alias and the views. This takes about 2 minutes.
- If the best FP16 rate on dt is below 0.5 x copy, run only `LITE` rows. This saves about 8 minutes (about $0.5) in the INCONCLUSIVE case. It also gives a clean negative.

### O4. Name the tuning asymmetry in the result
- INTERPRETATION: the Triton families choose from 54 configurations. The stock family chooses from 2 settings. A GO means "tuned Triton beats the 2-setting stock kernel", not "the layout beats the engine kernel". State this in Section 10.

### O5. Poison the unused page slots in the correctness cell
- INTERPRETATION: real engines hold stale values in the tail of the last page. The cell pads with zeros. A mask fault could still pass if the padded value is zero. Fill the tail with NaN (E4M3 has one) and check that the output stays finite.

### O6. Stop at the first sticky CUDA error
- FACT: every per-row `except Exception` records the error and goes on. After an illegal memory access, all later rows fail until the end.
- Fix: if the message holds "illegal memory access" or "device-side assert", write the status and exit. This saves Pod minutes.

## Answers to the seven questions

### 1. Is the views layout exact? Can the stock FA4 kernel read it?
- FACT: my script does not import the author's code. It builds K with an explicit pair rotation (pairs i and i + 256, 64 angles). It compares the views output after `unpermute` with the standard attention. The error is 1.1e-15 (float64, L 700, 8 query heads). Script: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/r71/views_check.py`.
- FACT: ROT has 128 dims (0 to 63, 256 to 319). UNROT has 384 dims. `unpermute` inverts the V-view order. `test_views_identity` checks both.
- FACT: the kernel masks the tail with `tmask` and `other=0.0` (`compact_attn.py` lines 205-216). The correctness cell uses L 3,000, so the last page has 56 tokens. The CPU dry run adds L 64, 130 and 2,049, with empty splits.
- INTERPRETATION: partial pages are exact. The masked load keeps stale NaN values out of the dot. O5 adds a test for it.
- FACT (vLLM wrapper at 3966924d5, `vllm/vllm_flash_attn/flash_attn_interface.py` lines 117-118 and 333): the wrapper checks only `stride(-1) != 1`.
- FACT (FA4 `maybe_contiguous`, `flash_attn/cute/cute_dsl_utils.py` lines 79-124 at b273a02): a strided view stays as it is if three conditions hold. The pointer is 16-byte aligned. `stride(-1)` is 1. Every other stride is a multiple of 16 bytes.
- FACT: both views pass. Their strides are 163,840, 640 and 40,960 elements. The V view starts 128 bytes after the K view (256 bytes for BF16).
- FACT: the compile key holds no stride value, only the zero-stride pattern (`interface.py` lines 1405-1470). K and V use separate TMA atoms (`flash_fwd_sm90.py` lines 390-402). No code assumes equal K and V strides.
- INTERPRETATION: the stock kernel reads the views without a copy. The "_views" rows should run. I did not read the CuTe runtime check of stride divisibility. These rows are reported only, so a failure costs nothing.
- FACT: the dequant path folds q_descale x k_descale into the score and v_descale into the output, per (batch, KV head) (`flash_fwd_sm90.py` lines 1793-1804). One K-view descale covers K_rot and V_unrot together.
- Engine rule for two scales: store K_rot with its own scale s_k and V with s_v. Pass k_descale = s_v and v_descale = s_v. Multiply the rotated part of the query by s_k / s_v in the query-prep op. The score is then exact. If the engine uses one scale for the whole row, no query change is needed. Then K_rot can overflow E4M3 when the k_norm weight is large (O1).

### 2. Do the two views share their DRAM reads?
- FACT: in the loop, the V load comes first and the K load second (`compact_attn.py` lines 206-214). They are a few instructions apart.
- FACT: the shared-memory model in F3 shows one K tile and one V tile in shared memory. INTERPRETATION: both loads issue at the loop head.
- INTERPRETATION: the overlapped lines are requested twice within a short time. The L1 cache, a merge of in-flight misses at L2, or an L2 hit serves the second request. DRAM bytes stay near 640 per token.
- HYPOTHESIS: sharing holds with a probability of 90% or more. Support: the E-51 K = V row on the RTX 4090 gave e 0.89 and 0.99 with the same loads. Nothing in E-51 or E-52 measures it on the H100 (F5).
- The alias row is the sharing control for full overlap. A high e_h shows that a second read of the same line costs no DRAM. The views overlap only 384 of 640 bytes, so F5 is the direct test.
- Rule effect: no false GO. If the reads do not share, sigma_layout stays near 0. A false KILL is possible at a std rate of 0.96 x copy or more. See B1.
- GO region (FACT, arithmetic): let r = dt_std / dt_stock. Then sigma_layout = 0.375 e and sigma_mma = 1 - (1 - 0.375 e) r / 0.85.
- GO needs e of at least 0.78 at r = 0.90 and 0.67 at r = 0.85. It needs 0.54 at r = 0.80 and 0.41 at r = 0.75 or lower.

### 3. Is H4 plausible? Is FP16 compute safe?
- FACT (Triton v3.8.0, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp` lines 183-186, 196-226 and 424-437): E4M3 to FP16 is one `cvt.rn.f16x2.e4m3x2` per 2 values. E4M3 to BF16 with `computeCapability >= 90` is that instruction plus two `cvt.bf16.f16`. Below 90 (the 4090) it is two `cvt.f32.f16` and two `cvt.rn.bf16.f32`. The author's description is correct.
- FACT: the 4090 takes a different BF16 route than the H100. The rehearsal cannot test the BF16 path of the H100.
- FACT: the FlashMLA note (H800, SM90) counts 1/64, 1/64 and 1/16 clock per value for FP8 to half, half to float and float to BF16. NVIDIA's current table lists `cvt.f32.f16` at 64 per clock and SM. It has no row for `cvt.bf16.f16` or for the FP8 conversions. My page fetches were truncated, so the column mapping is uncertain.
- HYPOTHESIS: the rate of `cvt.bf16.f16` on SM90 is unknown. The author assumes 16 per clock and SM (README lines 28-31). If it is 64, the BF16 route costs 16 clocks more than FP16, not 64. Then the cast alone cannot explain the 114 clocks per token. I reproduced the 114 from E-51 at 1.98 GHz.
- INTERPRETATION: H4 is plausible and untested. For: the FP8 compact kernel lost to its own BF16 version. The dtype elasticity is 0.64 to 0.67. The BF16-cache std kernel reads 1.02 to 1.05 x copy. Against: the stock FA4 kernel already casts with one instruction and still has e 0.2. The single-buffered tiles (F3) give a second candidate cause, exposed memory latency.
- FACT: FP16 compute is safe for these inputs. I emulated the kernel math on the CPU with E4M3-stored rows, logit scale 1 to 30 and k_norm spread 0.1 to 3. The FP16 error was 0.0004 to 0.0037. The BF16 error was 0.0014 to 0.045. The largest |q x w| was 26.6, far below 65,504. P is at most 1.
- INTERPRETATION: FP16 is more accurate than BF16 here. E4M3 values convert exactly. The scales apply after the dot in FP32. Real Gemma 4 activations are untested (the accuracy gate after a GO).

### 4. Do the rule changes give a false GO or KILL?
- V3 at X and dt only: safe. It cannot give a false GO or KILL. It no longer guards the GO margin (F8).
- V4: see B2. It is the only change that can void a good run.
- V6 with the cap reported: safe.
- FACT (E-51 data): the sweep minimum differs from the confirmation median by +0.06% to -1.52% for the six selected rows. In all four E-51 sweeps only one configuration was within 2% of the best.
- INTERPRETATION: selection bias with 54 configurations is small, because the confirmation re-measures the selected rows. The open points are independence (F1) and the 2-versus-54 tuning gap (O4).

### 5. Time model, caps and failing configurations
- Counts, time and cost: see F4. The typical cost is $2.2 to $2.4. The worst case is $3.00 at the boot watchdog, if the pull takes 11 minutes or less (F10).
- FACT (E-51 rows, stage 2): shared memory is 50 KB at tile 32 and 84 KB at tile 64. The F3 model gives 33 KB at tile 16. All fit the 99 KB of the 4090.
- The stage 3 failures are in F3. The H100 has no shared-memory failure.
- INTERPRETATION: 2-warp configurations hold the accumulator and the q tile in few threads. They will spill heavily. They should compile and run slowly.
- INTERPRETATION: the rehearsal time is fine. I estimate 7 to 9 minutes against `KB_TIMEOUT` 900 s.

### 6. Code bugs
- FACT: the tests pass (19 passed in 32 s). My command wrote nothing into the repo.
- FACT: I found two real-path faults that the stub hides: F6 (`ptx_counts`) and F7 (NVML sampling).
- FACT: I found no fault in `make_fn`, `make_cell`, the pool table, the correctness references, the selection, the replication path or the `ok_groups` handling. The stub run executes the same analysis path.
- FACT: the CUDA graph capture of Triton and FA4 calls worked in E-51. The new code uses the same path.

### 7. What is missing, and is E-52 the right next test?
- Missing from R-70: the offline sm_90 compile and PTX diff (F2). The `ncu` attempt (F5). The BF16 pool calibration (F9). Per-loop PTX counts (F6). A stop rule for a slow pull (F10).
- Done from R-70: the V3, V4 and V6 fixes, the wide sweep (splits 64, warps 2, tile 16, stage 3) and the pool probe. Also done: the BF16 K = V rows, the NVML clocks, the L2-resident gate and the preregistration.
- INTERPRETATION: E-52 is a bytes-only test by construction. The views kernel casts and stages the same 1,024 values per token as its twin. It cannot measure the 37.5% cast saving or the staging saving that R-70 (H3) named.
- INTERPRETATION: if H4 is false, the answer is INCONCLUSIVE, and nobody has tested a kernel that stages V once. Write this into the INCONCLUSIVE consequence (B1, fix 3).
- INTERPRETATION: E-52 is still the right next test after D-85. Do F2 first. It costs nothing and can remove the two worst failure modes (a compile abort and a wrong cast sequence).

## Checks I ran
- FACT: `uv run --no-project --with numpy --with torch --with pytest python -m pytest -q ... test_e52.py -p no:cacheprovider`: 19 passed in 32 s. Output: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/r71/pytest_e52.txt`.
- FACT: I ran `kbench_e52.py --stub` into a scratch folder. I counted the rows per kernel and cell and the confirmation plan. I also counted the correctness configurations (54, 54, 1, 1, 1, 1 for FP8; 4, 4, 4 for BF16).
- FACT: `rule_grid.py` runs the author's `sigmas()` and `label()` on a grid of synthetic dt values. `views_check.py` runs the independent identity and the FP16 versus BF16 emulation.
- FACT: I recomputed four items from the E-51 H100 `kbench.json`. They are the shared-memory model (10 rows), the across-repeat dt ranges, the sweep versus confirmation differences and the number of configurations near the best.
- FACT: I read Triton v3.8.0 `ElementwiseOpToLLVM.cpp`, vLLM 3966924d5 `flash_attn_interface.py` and `flash_attn.py`, flash-attention b273a02 `interface.py`, `cute_dsl_utils.py` and `flash_fwd_sm90.py`, and the FlashMLA FP8 note.
- Not done: no kernel compile, no GPU run, no `ncu`, no run of `test_e51.py` (R-70 noted that it writes into the repo).
