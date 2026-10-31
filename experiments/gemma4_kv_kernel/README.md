# E-51: a compact K=V decode attention kernel for Gemma 4 against the stock vLLM kernel, on 1 x H100 (preregistration)

Status: PREDECLARED 2026-10-31. NOT RUN.
Idea: I-182. Journal entry: Section 6.1 E-51. Follows E-50 and its post-run review R-68 (recommendation 1 to 6).
Request: D-83 (user, 2026-10-31, "Plan the kernel test": a compact FP8 attention kernel timed against the stock vLLM
kernels at batch 19 / 32K and batch 11 / 64K, with the R-68 stop line; plan only, no GPU yet).
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
On 1 x H100, does a decode attention kernel that reads the compact K=V cache of the Gemma 4 31B global layers (640
of 1,024 values per head and token) cut the context-growing attention time enough to keep a step saving of at least
about 8%, against the best stock FP8 kernel of vLLM with an allowance for its coming FP8 MMA kernel? "No" is: the
compact layout cuts the time by less than 15.5% (32K) and 12.5% (64K) even against a standard-layout twin kernel.

## 2. Rubber duck
- E-50 (FACT, R-68): on top of the stock FP8 KV cache near the pool limit, the compact cache could save at most 12.1%
  (32K, B 19) and 15.0% (64K, B 11) of the decode step. This ceiling assumes that time follows bytes. The stock
  kernel's time elasticity to bytes (BF16 to FP8) was 0.49. R-68: S = 5% needs a cut sigma of 0.155 (32K) or 0.125
  (64K) of the context-growing attention time; sigma 0.25 gives S about 8%.
- The target (FACT, E-50 group profiles): the stock FA4 FP8 global attention takes 238 and 1,358 us per call at B 19
  (2K and 32K) and 143 and 1,507 us at B 11 (2K and 64K), main plus combine kernel. On the growth this is 2,135 GB/s,
  0.70 x the copy bandwidth.
- What a compact kernel must do (INTERPRETATION, from the target): at 32K it must read its 640-value rows at 0.52 x
  copy or more to pass the kill line against the stock kernel, and at about 0.69 x copy to reach the go line with
  the FP8 MMA allowance below.
- Why Triton: both layouts in one code family (`compact_attn.py`: split-KV decode, one KV head and its 8 query heads
  per program, online softmax, a log-sum-exp merge). The layout effect is then measured inside the family
  (sigma_layout), and the family's quality against the stock kernel separately. A CuTe (FA4) port costs much more;
  it comes only if the layout works but the Triton kernels are slow (PIVOT "kernel quality").
- Why no engine: the question is the kernel time. E-50 showed that attention is the whole growth of the step (V4b
  about 100%). Check V3 ties the microbenchmark to the engine.
- The stock FP8 MMA kernel (FA PR 199, merged 2026-10-29) is not in this image (it comes with vLLM PR #60372). Its
  table shows 8% to 15% on the context part (R-68). The rule therefore asks the compact kernel to beat a stock time
  15% lower than the measured one (sigma_mma). A rerun after PR #60372 can measure it directly.
- What would kill it early: if the compact Triton kernel is not faster than the standard Triton kernel by the kill
  line, the layout does not help, whatever the kernel quality.
- Expected (INTERPRETATION): a memory-bound Triton pair gives sigma_layout of 0.30 to 0.37. The open question is
  whether the Triton kernels reach about 0.7 x copy on 640-byte FP8 rows.

## 3. Hypotheses
- H1 (primary): in at least one group, sigma_mma >= 0.25 and sigma_layout is above the kill line.
- H0: sigma_layout is below the kill lines (no layout effect), or the compact kernel stays slower than the bar.

## 4. Setup (frozen)
- Hardware: 1 x H100 80GB HBM3 (SXM), Secure Cloud, on-demand, `minCudaVersion` 13.0, US-NE-1 if in stock.
  Rehearsal: RTX 4090 (Triton only; FA4 does not run on SM89).
- Software: the E-49 and E-50 image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
  (vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14, the image's Triton). No model and no engine.
- Shapes (FACT, `gemma4.py` and E-49): one global layer of Gemma 4 31B: 32 query heads, 4 KV heads, head_dim 512,
  scale 1.0, no soft-cap, decode (one query token per sequence), uniform contexts, 64-token pages, a random block
  table (the same for every kernel).
- Cells (B:L): 4:2K, 4:32K, 4:64K; 19:2K, 19:8K, 19:32K; 11:2K, 11:8K, 11:32K, 11:64K. Groups: 32K at B 19 and 64K at
  B 11 (the E-50 pair batches). The 8K cells check linearity (reported).
- Kernels:
  - stock: `vllm.vllm_flash_attn.flash_attn_varlen_func` with the arguments of `FlashAttentionImpl.forward` at
    3966924d5 (`fa_version` 4, window [-1, -1], softcap 0, causal, block table, `seqused_k`, ones as descales for
    FP8). FP8 with `num_splits` 32 (the engine's cap under full CUDA graphs) and 0 (automatic); BF16 with 0 and 32.
    The page is 64 tokens for both (the engine's BF16 page was 32).
  - Triton (`compact_attn.py`): standard (K and V caches) and compact (one [V | K_rot] row tensor), FP8 and BF16; the
    sweep is splits {4, 8, 16, 32} x warps {4, 8} x tile {32, 64} tokens.
- Timing (`kbench_e51.py`): each configuration is captured as 20 calls in one CUDA graph and replayed 15 times; time
  per call = replay time / 20 (median; spread = max / min - 1). The caches are much larger than L2.
- Correctness cell: B 2, L 3,000 (a partial last page), Gemma 4 K=V data (V RMS-normed, w = 1 + 0.1 N(0, 1),
  K = rope_p(w V)), q ~ N(0, 1/512) with RoPE at position 2,999, stored in FP8 and BF16.
- Not timed: the query preparation of the compact kernel (q x w on the 384 unrotated dims, an elementwise op), the
  KV write, and the accuracy beyond the correctness cell.

## 5. Metrics
- Primary (FP8): for each group X, with dt(X) = t(X) - t(2K) at the group batch, the best configuration per cell
  and kernel family, and the best stock split setting per cell:
  - sigma_mma(X) = 1 - dt_compact / (0.85 x dt_stock): against a stock kernel 15% faster on the growth.
  - sigma_layout(X) = 1 - dt_compact / dt_standard: the two Triton layouts.
- Secondary (reported only): sigma_dq = 1 - dt_compact / dt_stock; the same for BF16 (the exact-cache case); the
  B = 4 times; the bandwidth on dt for each family; dt_standard / dt_stock (the Triton quality); the correctness
  errors and the errors against the exact attention (compact against standard storage); the best split, warp and
  tile choices.
- Minimum effect that matters: the kill line (S = 5%) and the go line (S about 8%).

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU is H100 80GB.
- V2: the four deciding kernels (stock FP8 with 32 splits and automatic, Triton standard and compact FP8) are within
  a relative L2 error of 0.02 of the reference of their own stored values.
- V3 (engine tie): the stock FP8 kernel with 32 splits is within 15% of the E-50 in-engine time per call at the four
  group cells (238.3, 1,358.1, 143.2 and 1,507.2 us).
- V4: the replay spread is at most 2% for every deciding measurement.
- V5 (positive control): the copy bandwidth exists, and no deciding measurement reads above 1.15 x copy.
- V6: hard throttle in at most 2% of busy telemetry samples.

## 7. Decision rule (frozen)
Deciding groups: X = 32,768 (B 19) and 65,536 (B 11), FP8. Kill lines (R-68): 0.155 at 32K and 0.125 at 64K.
- GO: in at least one group, sigma_mma >= 0.25 and sigma_layout >= the kill line of that group. A GO allows the next
  steps only: the accuracy gate (R-68 recommendation 7) and an engine prototype plan with the page-size plan.
- KILL: sigma_layout below the kill line in both groups. The layout itself gives too little, so the layout work
  stops; I-182 stays as a capacity note (1.39 to 1.47 times the requests; FP8 KV already gives 1.96 times).
- INCONCLUSIVE: the benchmark did not finish, a group is missing, or the Triton standard kernel reads below 0.5 x copy
  on dt in a group (then the layout test is weak).
- PIVOT: otherwise, with the reason "kernel quality" (sigma_layout >= 0.25 in a group: the layout works, and the next
  step is a compact kernel of FA4 quality) or "below the bar".
- INVALID: V1 to V6 fail, or no copy bandwidth.

## 8. Controls and baselines
- Baseline: the stock FA4 FP8 kernel (best split setting per cell), tied to its in-engine time by V3.
- Layout control: the Triton standard kernel (the same code with separate K and V caches).
- A/A: the replays and the spread check; the 8K cells (linearity, reported).
- Positive control: V5 (no kernel reads faster than the copy bandwidth allows).
- Negative control (secondary): on the same K=V data, the compact and the standard storage give the same attention
  within the error of the dtype.

## 9. Budget
- H100 SXM Secure at the live price (E-50: $3.99/h). Image pull 5 to 11 min (E-50: 11 min on a cold host). The
  benchmark: 10 cells x (4 stock + 64 Triton configurations), about 0.5 s each, plus JIT compiles: about 10 to 15
  min. About 25 to 30 min of Pod time, about $2. Cost cap $3 (boot watchdog 45 min).
- Rehearsal (S5): RTX 4090 ($0.89/h), about 20 min, cap $0.60: the Triton kernels compile and pass V2 on SM89, the
  harness writes every file, the session flow and AUTO_STOP work.
- Caps: `KB_TIMEOUT` 1,800 s; session watchdog 45 min; `AUTO_STOP=1` with a 180 s delay (copy the results then).
- Balance: D-82 (a $11 floor) was for E-50 only, so the $35 floor applies again; S4 checks it live.

## 10. Threats to validity
- Triton quality: a slow Triton family hides the layout effect only if both kernels stop being memory-bound; the
  rule then gives INCONCLUSIVE (the 0.5 x copy check), not KILL.
- No stock FP8 MMA kernel in the image: sigma_mma uses a 15% allowance from the FA PR 199 table (R-68).
- The compact QK product runs over 640 dims (+25% multiply-adds against the 512 needed). This is conservative.
- The BF16 stock kernel uses 64-token pages (the engine: 32); BF16 is secondary.
- The query preparation is not timed; it is an elementwise op on B x 32 x 512 values.
- The microbenchmark runs one layer alone; the engine runs it inside a graph with other kernels. V3 ties them.
- Random cache values: these kernels have no data-dependent paths.
- One GPU, one session, decode only, uniform contexts.

## 11. Files
- `compact_attn.py`: the two Triton kernels (split and merge) and the PyTorch helpers (RoPE, K=V data, compact query
  and row, paged and split references).
- `kbench_e51.py`: the benchmark (stock and Triton, FP8 and BF16), the correctness cell; `--stub` for the Mac tests.
- `analyze_e51.py`: the groups, the sigmas, V1 to V6 and the rule.
- `session_e51.sh`: the Pod session (STUB=1 runs it on a Mac).
- `test_e51.py`: 8 tests (Section 12).
- Upload to the Pod: `gemma4_kv_kernel`, `runpod`, `fp8_quant_fusion`, `nvfp4_dispatch`, `safedispatch_testA`,
  `safedispatch_testA2` and `ledger`, without `results`.

## 12. Verification record (stage S3 and S5)
- S3 (2026-10-31, Mac, no GPU and no Triton): `uv run --no-project --with numpy --with torch --with pytest python -m
  pytest test_e51.py`: 8 passed. The tests: RoPE equals the E-49 numpy code; the compact row with the absorbed query
  gives the standard attention (float64, error below 1e-12); the paged gather equals the dense attention; the split
  rule of the kernels (with empty splits) and the merge are exact; GO, KILL, PIVOT ("kernel quality", "below the
  bar"), INVALID (V2, V3, V5, no copy bandwidth) and INCONCLUSIVE (a failed run, a slow standard kernel) on stub data;
  a stub session from a clean copy, a rerun that skips the finished benchmark, and a failed run. The Triton kernels
  themselves can only run in S5.
- S5 plan: RTX 4090, the image above, boot watchdog CAP 1,800 s, `AUTO_STOP=1`. VERIFY:
  1. The image starts; `import triton` works; the session ends with "done after".
  2. Every Triton configuration compiles, or its error (for example out of shared memory) is recorded and at least
     one configuration per kernel and cell runs.
  3. The four Triton kernels pass V2 on the correctness cell; the compact and standard errors against the exact
     attention are of the same size.
  4. `kbench.json` has every cell; the stock rows have status "error" with a clear reason (FA4 is not on SM89).
  5. The analysis runs and gives INVALID (V1, V2 for the stock kernels, V3), as expected on this GPU.
  6. AUTO_STOP stops the Pod. Record the Triton times for the H100 time model.
  Not testable on the 4090: the stock FA4 call, V3, the H100 times and bandwidths.

## Amendments
None yet.
