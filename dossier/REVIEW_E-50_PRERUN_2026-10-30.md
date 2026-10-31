# R-67: pre-run review of E-50 (Gemma 4 K=V headroom on top of an FP8 KV cache, I-182)

Date: 2026-10-30. Reviewer: Sonnet (a different model from the author).
VERDICT: AMEND_THEN_RUN (3 blockers R1 to R3, 3 more required fixes R4 to R6, 11 optional items)

Scope: `experiments/gemma4_kv_headroom/` at commit 26efdfd, the patched `decode_probe.py` and `analyze_e49.py`, the E-49 raw data.
Labels: FACT (I read a source or recomputed a number). INTERPRETATION (my reasoning). HYPOTHESIS (not tested).
I changed no repo file except this one. I did not read `.env`. I made no Runpod call and no cloud change.
Scratch scripts are in the session scratchpad folder `r67/`. vLLM source: GitHub `vllm-project/vllm` at 3966924d5.

## Verdict in short
- FACT: The question is right and the design is sound. The FP8 path exists in the E-49 image, and the source supports the plan.
- FACT: `test_e50.py` passes (14). `test_e49.py` passes (20). The E-49 decision and `summary.json` reproduce exactly (identical JSON).
- INTERPRETATION: Four things can still waste the run or mislabel it. The first is the 1,500 s timeout (R2). The second is the weak V11 (R3).
- INTERPRETATION: The third is the all-or-nothing group validity (R4). The fourth is a rehearsal that cannot test the pool rule (R1).
- INTERPRETATION: The likely outcome is GO. The R-65 model with halved FP8 terms gives S about 11.8% (32K, B 19) and 13.7% (64K, B 11). The main risk is the first FP8 engine start, not the science.

## Required fixes

R1 (BLOCKER for S5). Write the VERIFY list. Fix the rehearsal plan.
- FACT: README Section 12 has no VERIFY items. PROTOCOL S5 needs them. My list is at the end of this file.
- FACT: `session_e50.sh` downloads `$M26` (51.6 GB) even with `ONLY=m31f8`. The E-49 rehearsal set `M31` and `M26` to E2B. Do the same, with the revisions.
- FACT: In the E-49 rehearsal (E2B, 4090, BF16) the idle pool was 88,780 blocks. A request held 2,121 blocks at 32K and 4,174 at 64K.
- INTERPRETATION: An FP8 pool twice as big gives B of 71 (32K) and 36 (64K) at 85%. Both reach or pass `max_num_seqs` 32. The pool rule never runs.
- Fix: Use `PROBE_ARGS="--max-num-seqs 64"`. Run only the 64K group: `PF8=2048:a65536,8192:a65536,32768:a65536,65536:a65536,65536:a65536:1`. B is then pool-derived (about 36).
- FACT: No unit test runs the new `main()` loop (sizes, `auto_batch`, `no_size`, `ok_pairs`). The stub re-implements it. Only the rehearsal runs it. It must end with status "ok".

R2 (BLOCKER for S6). Raise the m31f8 timeout.
- FACT: A timeout kill leaves status "ok_slope". The rule then gives INCONCLUSIVE and the deciding arm is lost.
- FACT: The E-49 rates (prefill 14,577 to 7,137 tokens/s, init 277 s) give m31f8 14.8 min at unchanged speed.
- INTERPRETATION: It takes 19.8 min if FP8 prefill is 1.25 times slower and init takes 460 s. It takes 26 min at 2 times slower. The limit is 25 min.
- FACT: The FP8 FA4 kernels have no JIT warmup. `flash_attn.py` L157 returns no keys for a quantized KV cache. The FP8 prefill speed is unknown.
- Fix: `PROBE_TIMEOUT` 2,100 s for m31f8 only. The worst case adds 10 min ($0.67). The cap $4.50 still holds (worst case about 57 min, $3.8).

R3 (BLOCKER for S6). Make V11 prove the attention path.
- FACT: V11 reads two strings. Scenario I: a log with `Using TRITON_ATTN` after the Gemma4Config line still passes V11 and gives GO.
- FACT: The E-49 log has `Using FLASH_ATTN attention backend out of potential backends` and `Using FlashAttention version 4`.
- Fix: For m31f8 also require `Using FLASH_ATTN attention backend`, `Using FlashAttention version 3` and `Using FlashAttention version 4`.
- Fix: For the BF16 arms require the backend line and version 4. Fail any arm whose profile has `kernel_unified_attention`.
- Fix: Add these lines to the stub log. Add a test that fails V11 on a Triton log.

R4. Make group validity local (amendment to Sections 6 and 7).
- FACT: Scenario B: two cells of the 32K group with a 6% block spread give "INVALID (m31f8: G32768)". The 64K group has S_low 13.7% and would pass.
- FACT: Scenario C: the primary 32K cell is lost and its replicate lives. The group has no profile, V4a fails, and the run is INVALID.
- Fix: A group with a failed G check or a missing profile cell counts as lost. It cannot pass and cannot support KILL.
- Fix: INVALID stays for the slope checks (V1 to V9, V11), V8 and no copy bandwidth. Add two tests.

R5. Correct and extend the README facts (amendment A1).
- FACT (source): FP8 on SM90 sets block size 64 (`flash_attn.py` L300 to L326, `platforms/interface.py` L782). The sliding page is 524,288 B. The global page is 262,144 B.
- FACT (source): `unify_kv_cache_spec_page_size` raises the global block to 128 tokens (`kv_cache_planning.py` L393 to L395). The runner splits it into two 64-token kernel blocks (`worker/utils.py` L515 to L561, L630 to L684).
- INTERPRETATION: I expect no page-size error. The idle pool is about 8,050 blocks. A request holds about 341 blocks at 32K and 597 at 64K. So B is 19 or 20 at 32K and 11 at 64K.
- INTERPRETATION: This chain needs `model_config.get_head_size()` to be 512. `from_layers` takes the maximum over layers (`config/model_arch.py` L129), so it should be 512. A value of 256 would give a 32-token global block and "No common block size" at the start.
- Fix: Add these numbers to Sections 4 and 10. Say that the kernel block split is a first use for this model. A run.json with other values needs a note.
- FACT (source): `aot_schedule` is True at init when the FA version is unset. So the builder takes a split cap of 32 for full CUDA graphs (`flash_attn.py` L708, L770, L865). The BF16 arms force FA4 and keep the cap at 0.
- INTERPRETATION: The FP8 global kernel runs with 32 splits. PR #60372 calls this "the previous CUDA-graph cap of 32". The BF16 kernel picks its own splits.
- Fix: Say in Section 5 that "FP8 saving at B = 4" compares two stock paths. It is not one kernel with a smaller cache.

R6. State the Pod disk and environment in Section 9.
- FACT: Section 9 says the Pod volume disk holds the models. It gives no size. The E-49 Pod had a 120 GB volume at `/workspace` and a 40 GB container disk. The models take 85 GB.
- FACT: E-49 stored the models in `/workspace/cache/hf`. The session script sets no `HF_HOME`, so the Pod environment must set it.
- Fix: Write volume 120 GB, container disk 40 GB, `HF_HOME` and `VLLM_CACHE_ROOT` in Section 9. S4 quotes them.

## Optional items
- O1: Scenario D: a persisted status "ok_pairs" crashes `arm_summary` (`KeyError: 'checks'`). The window is microseconds. Map "ok_pairs" to "ok" before `model_summary`.
- O2: V6 on the FP8 arm. INTERPRETATION: timing does not depend on values, so a low GSM8K score does not break S. A low score is still news about uncalibrated FP8 KV.
- O2 (cont.): FACT: PR #60372 serves a copy of this checkpoint with FP8 KV. Its table shows AIME 86.88 against 87.29 (BF16), GPQA 86.55 against 85.42, MMLU-Pro 86.64 against 86.44. Consider "decision with accuracy caveat" instead of INVALID.
- O3: Capacity (INTERPRETATION). At the 64K pool limit the compact cache adds 1.47 times the requests. The model gives about +46% tokens/s, not 13%. Add this as an exploratory line from the fitted constants. It needs no GPU time.
- O4: Add the batch-scaling check b(B pair) / b(B 4) against B pair / 4. The slope cells at B = 4 and the pair cells give it for free. It tests the byte model of the R-65 extrapolation. The m31bf 32K group has 3 cells (df 1), so its bound is weak.
- O5: Scan `log.txt` for "CuTeDSL JIT compilation during inference" and report the count and times. A JIT spike inside a timed block shows as a V7 spread flag.
- O6: Use `pool_share`. It is computed and never used. KILL needs `pool_share` >= 0.75 in both groups. The KILL argument (3% / 0.85) assumes B near 0.85 of the limit.
- O7: Moving target. FACT: vLLM PR #60372 (open, updated 2026-10-29) swaps the dequant kernel for FP8 MMA on the full-attention layers and removes the 32-split cap. State that E-50 measures the dequant kernel at 3966924d5. Record the FA submodule version.
- O8: Cheaper designs exist. INTERPRETATION: a kernel-only microbenchmark would cost about $0.7, but it cannot test the engine start. Dropping m31bf saves about 9 min ($0.6). I do not recommend a change, because D-76 asked for this scope.
- O9: Move the plan loop of `main()` into a function so a test can drive it with `BudgetEngine`. Write the S7 manifest adapter for three arms (MEASUREMENT.md Section 5).
- O10: Scope text. INTERPRETATION: the result is decode-only with uniform contexts. Prefill chunks inside decode steps and speculative decoding lower S in serving.
- O10 (cont.): INTERPRETATION: FP8 KV looks near lossless on Gemma 4 (see O2). So the exact-cache argument is weak. Only speed and capacity remain.
- O11: V4c has little headroom. E-49 sliding changes were -2.1% and -3.5% against a 5% limit. FA3 is new here. A V4c failure gives PIVOT, not INVALID. Keep the rule frozen and expect this risk.

## Checks I ran
1. Tests. `test_e50.py`: 14 passed in 3.1 s. `test_e49.py`: 20 passed in 2.5 s. FACT.
2. E-49 reproduction. `analyze_e49.py` on the frozen data prints the same PIVOT line. `summary.json` is identical. FACT.
3. Token budget. I swept 13 batch sizes (1 to 32) and 4 contexts (2K to 64K) on the replay engine. Every cell stays "ok". The smallest slack is 31 tokens. FACT (replay model).
4. Scheduler. FACT: chunk alignment to the block size exists only for Mamba "align" mode (`scheduler.py` L432, L860). So the 128-token FP8 block does not shrink the prefill chunk. Decode tokens go first (L795).
5. Guard. `budget_left < need` uses the first-token count of request 0, the largest. It is correct. The 32-token margin covers one step of async lag. INTERPRETATION.
6. Pair batch. `blocks_per_request` = (idle free - free after prefill) / B. It matches E-49: 391 per request at 2K, 1,358 at 32K, 2,391 at 64K. Sliding groups free their blocks and the null block is outside the free count. FACT.
7. Pool safety. FACT: `scheduler_reserve_full_isl` is True and `watermark` is 0 (`config/scheduler.py` L190, L196). A new request needs its full-sequence blocks free. The sliding cap is 81 blocks per group with async scheduling and 49 without (FP8).
8. At 85% the free pool is at least 2.3 times that need (32K: 1,568 against 661; 64K: 2,077 against 917). Decode growth is 1 to 3 blocks per request. INTERPRETATION: no preemption at 85%.
9. Padding. B of 19 runs in the graph of 24, and B of 11 in 16. Weight-bound GEMMs barely change, so S is slightly low. FACT: the E-49 capture sizes were 1, 2, 4, 8, 16, 24, 32, 40, 48, 56 and 64. FULL graphs cover the first seven.
10. Analysis formulas. FACT: 40,960 B = 10 layers x 4 heads x 512 x 2 x 1 B. The t values for df 2 and df 3 are 2.920 and 2.353. S, M1, M2, V4a to V4c and `S_att` match README Sections 5 to 7.
11. Stub scenarios (`r67/scen.py`):
 - A baseline: GO (32K B 22, S 0.120; 64K B 11, S 0.138).
 - B: INVALID (G32768) while the 64K group passes. See R4.
 - C: INVALID (G32768, V4a) after a lost primary cell. See R4.
 - D: crash on status "ok_pairs". See O1.
 - E: order split broken at 32K. The count split carries it. GO.
 - F: odd launch count (49 per step) in the 32K group. V4c unknown there. GO from the 64K group.
 - G: tiny pool (B 2 and 1). PIVOT "below the 5% bar". No false KILL (S_up 3.2%).
 - H: FP8 kernel at 1,500 GB/s. PIVOT KERNEL-BOUND for both groups.
 - I: Triton backend line in the log. V11 passes and the decision is GO. See R3.
12. Shell. `sh -n session_e50.sh` is clean. The arm order, `done_ok`, `capped` and the moved failed folder behave as in E-49. FACT.
13. KILL argument. INTERPRETATION: S(B) / B falls with B when T = c0 + c1 B + beta B X. So S at the pool limit is at most S(B) times B max / B. Rounding gives a factor of at most 1.28 at B = 11 and 1.24 at B = 19.
14. KILL bound. S_up below 3% then means S below 3.9% at the limit, under the 5% bar. The argument holds if B is near 0.85 of the limit (see O6).
15. M2 for the 32K group compares the 8K to 32K slope with the fit. It tests curvature less than the 32K to 64K test of E-49. A GO from this group alone rests on that test and on the 5% V3 limit. INTERPRETATION: the margin is large (S 11.8% against the 5% bar), so I keep the rule.
16. INCONCLUSIVE paths. A lost group and a failed arm give INCONCLUSIVE in the tests. A timeout kill leaves status "ok_slope" and gives INCONCLUSIVE by code reading. A lost primary cell gives INVALID today (R4).

## vLLM facts at 3966924d5 (items a to g)
- (a) FACT: Verified. `models/config.py` L299 to L356 logs "Gemma4 model has heterogeneous head dimensions {...}. Using per-layer FA3/FA4 selection for FP8 KV cache on SM90." The conditions are cache dtype "fp8", SM90 family, version unset, backend unset or FLASH_ATTN. The two string literals join, so the V11 regex matches.
- (b) FACT: Verified. `fa_utils.py` L139 to L167 upgrades FA3 to FA4 for head size above 256 on SM90. `FlashAttentionImpl.__init__` picks the version per layer (L1095 to L1125). `_get_sm90_fa4_fp8_kv_block_size` returns 64 for head 512 (L300 to L326). PR #60372 says FP8 sliding layers keep FA3.
- (c) FACT and INTERPRETATION: see R5. The code allows a manager block that is a multiple of the kernel block ("hybrid_blocks"). The engine should start. The risk is the first use, not a known error.
- (d) FACT: The 31B config has `kv_cache_scheme: null`. `CompressedTensorsKVCacheMethod` (`compressed_tensors.py` L1093 to L1273) creates k, v and q scales as ones and keeps them. Scale 1.0. No warning. No error.
- (e) FACT: `_cudagraph_support` is ALWAYS for FA3 and FA4 (`flash_attn.py` L619 to L622). There is no mix restriction. The E-49 log shows FULL and PIECEWISE capture. INTERPRETATION: it also shows a CuTeDSL compile warning inside FULL capture, with no error.
- (f) FACT: The E-49 engine line has `kv_cache_dtype=auto`, so the FP8 line has `kv_cache_dtype=fp8`. The regex also matches `fp8_e4m3`. The BF16 strings match the E-49 log.
- (g) FACT: Every field exists. `RequestOutput.finished` (`outputs.py` L201). `finish_reason` and `stop_reason` (L93, L94). `Scheduler.requests` (L223), `running` (L240), `num_preemptions` (L1837). `get_num_free_blocks` (`block_pool.py` L912). Finished requests leave `requests` (L3065), so the probe reads preemptions at the first check.

## Time and cost
- Setup: README 6 min. E-49: 348 s. m26: README 8 min. My model 7.7 min (E-49 arm 447 s). m31bf: README 9 min. My model 9.0 min.
- m31f8: README 16 min. My model 14.8 to 21.9 min (see R2). The pair cells need about 390 s of prefill at E-49 speed.
- Total: boot 4.7 + 6 + 15.8 + 7.7 + 9.0 + 3 (AUTO_STOP) = 46 min, about $3.1. The README says 48 min and $3.20. INTERPRETATION (model from E-49 rates).
- Caps: m26 and m31bf start only if the elapsed time is under 2,400 s. Expected starts are 1,300 s and 1,760 s. The cap cuts m31bf first, then m26. This order is right, because D-76 asks for the m26 rerun.
- The m31f8 arm uses `capped 0`. Only a setup of 50 min would skip it.
- HYPOTHESIS: The FP8 engine start fails with a chance of 15 to 25%. The session then runs m26 and m31bf for about $1.1 more and gives secondary data only. Decide at S4 whether to stop the Pod at once.
- Boot watchdog 65 min is $4.32. Cost cap $4.50 holds. The session watchdog (60 min from session start) is the second guard.

## Prior work since R-65 (short)
- FACT: Nothing new stores the Gemma 4 K=V cache once. vLLM #41944 (V weights only) is still open. #61267 and #47507 are LoRA and loading fixes.
- FACT: Competitor path: vLLM #53175 (merged 2026-09-24, in the image), #60372 (open), #45434 (open). SGLang #46965 (open, speed and accuracy TODO).

## VERIFY list for the rehearsal (S5)
1. The image starts. `import vllm` works. Version is 0.33.0rc1.dev229+g3966924d5. `M31` and `M26` are E2B, so no 52 GB download starts.
2. The probe takes `--pair-cells`, `--kv-cache-dtype fp8`, `--waste-cells ""` and `--max-model-len 67584` with no error.
3. The engine line has `kv_cache_dtype=fp8`. `run.json` has `free_blocks_idle` above 0, `kv.cache_dtype` fp8, `kv.groups`, `model.cudagraph_sizes` and no `_error` key.
4. Every slope cell and pair cell is "ok". `finished` is empty. `request_preemptions` are all 0. `budget_left` is at least `need`.
5. With `--max-num-seqs 64`, the 64K B equals int(0.85 x idle / blocks per request) and is below 64. `pool_share` is near 0.85. There is no preemption at that fill.
6. `tokens_at_start` stays below the prefill bound at the largest B. This tests the budget model on the real scheduler above B = 8.
7. Profile files exist. The order split gives 8 x 35 groups on Triton.
8. `analyze_e50.py` ends with no exception. The expected decision is INVALID or INCONCLUSIVE (V1 not H100, V11 second string absent on SM89). This is not a failure.
9. The session script ends with `done after`. A second run prints "ok before, skip". `AUTO_STOP` stops the Pod. A run with `PROBE_TIMEOUT` 60 logs exit 124 and keeps the data.
10. Record init seconds and prefill tokens/s at the largest B for the H100 time model.

Not testable on the 4090 (residual risk, HYPOTHESIS until the paid run):
- The FA3 and FA4 selection and the V11 strings. The FP8 FA4 kernel and its JIT. The FA3 and FA4 kernel names.
- The 64-token block, the 128-token manager block and the kernel split. The real pool size and B on the H100.
- The compressed-tensors scale 1.0 path. The 31B GSM8K score with FP8 KV. The step times, M1 and V8.
- An SM90 smoke run would test more. I did not price it, because I made no Runpod call. D-81 decides.

## What the design gets right
- It reads the source first. The FP8 path, the block size and the log strings are right.
- The token budget and its guard fix the R-65 fault. They hold for B up to 32 on the replay model.
- The batch comes from the measured pool, with blocks per request taken from E-49-style cells in the same arm.
- It has INCONCLUSIVE outcomes for lost cells and M2 "unknown". A lost deciding cell is not a PIVOT.
- Replicate cells at X, cell-level df, and a negative control per group give an A/A inside each group.
- The arm order, time caps and watchdogs match the money limit. Secondary arms never decide.
- Padding and pool share bias S low. A GO is safe on those two points.
