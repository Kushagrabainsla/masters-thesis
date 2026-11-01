# E-51: a compact K=V decode attention kernel for Gemma 4 against the stock vLLM kernel, on 1 x H100 (preregistration)

Status: PREDECLARED 2026-10-31. AMENDED A1 2026-10-31 (R-69 pre-run review, before any data). S5 rehearsal PASS
2026-11-01 (RTX 4090), AMENDED A2 (from the rehearsal, before the H100 run). RUN 2026-11-01 on the H100: frozen
decision INVALID (V3, V4). Post-run review R-70 (Sonnet): CONFIRM_WITH_CORRECTIONS; Section 13 has the corrections.
Idea: I-182. Journal entry: Section 6.1 E-51. Follows E-50 and its post-run review R-68 (recommendation 1 to 6).
Request: D-83 (user, 2026-10-31, "Plan the kernel test": a compact FP8 attention kernel timed against the stock vLLM
kernels at batch 19 / 32K and batch 11 / 64K, with the R-68 stop line; plan only, no GPU yet).
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.
Text marked "(A1)" changed in Amendment A1. The version before A1 is commit 0fb1b39. Text marked "(A2)" changed in
Amendment A2; the version before A2 (the rehearsal code) is commit a429b4e.

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
- What would kill it early: if a compact Triton kernel that reads at least 0.6 x copy is not faster than the standard
  Triton kernel by the kill line, the layout does not help. A slower compact kernel gives INCONCLUSIVE, because its
  own defect can explain the result (A1).
- Expected (INTERPRETATION): a memory-bound Triton pair gives sigma_layout of 0.30 to 0.37. The open question is
  whether the Triton kernels reach about 0.7 x copy on 640-byte FP8 rows.

## 3. Hypotheses
- H1 (primary): in at least one group, sigma_mma >= 0.25 and sigma_layout is above the kill line.
- H0: sigma_layout is below the kill lines (no layout effect), or the compact kernel stays slower than the bar.

## 4. Setup (frozen)
- Hardware: 1 x H100 80GB HBM3 (SXM), Secure Cloud, on-demand, `minCudaVersion` 13.0, US-NE-1 if in stock.
  Rehearsal: RTX 4090 (Triton only; the FA4 FP8 dequant path asserts SM90, FACT from the FA4 interface, R-69).
- Software: the E-49 and E-50 image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
  (vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14, the image's Triton). No model and no engine.
- Shapes (FACT, `gemma4.py` and E-49): one global layer of Gemma 4 31B: 32 query heads, 4 KV heads, head_dim 512,
  scale 1.0, no soft-cap, decode (one query token per sequence), uniform contexts, 64-token pages, a random block
  table (the same for every kernel; its rows padded to a multiple of 16 entries, A1).
- Cache layout (A1, R-69 O7): the engine's layout (FACT, `FlashAttentionImpl.forward` at 3966924d5): one
  (blocks, KV heads, 64, 2 x 512) tensor, with K and V as the views `transpose(1, 2).split(512, -1)`. The stock and
  the Triton standard kernel read these views. The compact rows are one (blocks, KV heads, 64, 640) tensor, viewed as
  V and K_rot in the same way.
- Cells (B:L), in this order (A1, R4): the four group cells 19:2K, 19:32K, 11:2K, 11:64K first, then 19:8K, 11:8K,
  11:32K, 4:2K, 4:32K and 4:64K (reported). Groups: 32K at B 19 and 64K at B 11 (the E-50 pair batches). The 8K
  cells (and 32K at B 11) check linearity (reported).
- Kernels:
  - stock: `vllm.vllm_flash_attn.flash_attn_varlen_func` with the arguments of `FlashAttentionImpl.forward` at
    3966924d5 (`fa_version` 4, window [-1, -1], softcap 0, causal, block table, `seqused_k`, ones as descales for
    FP8). FP8 with `num_splits` 32 (the engine's cap under full CUDA graphs) and 0 (automatic); BF16 with 0 and 32.
    The page is 64 tokens for both (the engine's BF16 page was 32).
  - Triton (`compact_attn.py`): standard (K and V views) and compact (one [V | K_rot] row tensor), FP8 and BF16,
    `num_stages` 2. The FP8 sweep at every cell: splits {4, 8, 16, 32} x warps {4, 8} x tile {32, 64} tokens. The
    BF16 sweep at the group cells only: splits {16, 32} x warps {4, 8} x tile 32 (A1, O6).
  - Oracle rows (A1, O1; timing only, reported): the stock FP8 kernel with 32 splits and the selected Triton standard
    configuration, each with K = V (the V view passed as K and V): half the DRAM bytes, the same on-chip work.
- Timing (`kbench_e51.py`): each configuration is captured as 20 calls in one CUDA graph and replayed 15 times; time
  per call = replay time / 20 (median; spread = max / min - 1). The first eager call records the registers, spills
  and shared memory of each Triton configuration (A1, O5).
- Confirmation phase (A1, O3; `MEASUREMENT.md` Section 1.5): after the four group cells, each decision row (Section
  5) and each oracle row is captured in a new graph and timed 5 times, after one untimed pass over all rows (A2; the
  rehearsal's first timed repeat was 5.5% slow). Odd repeats run the rows in the reverse order (A B .. B A). Then
  `run.json` gets the status "ok_groups"; the reported cells follow.
- L2 (A1, O9): the group caches are larger than the 50 MB L2, except the compact FP8 rows at B 11 and 2K (57.7 MB,
  close to it). At B 4 and 2K the standard FP8 cache (33.6 MB) fits in L2 (reported cells only). INTERPRETATION
  (R-69): the effect on dt is below 1%. FACT (S5, A2): on the RTX 4090 (72 MB L2) the compact FP8 rows at B 11 and 2K
  read at 1,061 GB/s, 1.15 x copy. A 2K row that reads from L2 only makes dt larger (conservative).
- Correctness cell: B 2, L 3,000 (a partial last page), Gemma 4 K=V data (V RMS-normed, w = 1 + 0.1 N(0, 1),
  K = rope_p(w V)), q ~ N(0, 1/512) with RoPE at position 2,999, stored in FP8 and BF16. The stock kernels and every
  Triton configuration of the sweep run on it (A1, R5); the output is zeroed before each run.
- Fail fast (A1, R6): if the stock FP8 kernel with 32 splits or a Triton FP8 family fails the correctness cell, the
  run stops (`--fail-fast auto`: on SM90, off on the 4090).
- Not timed: the query preparation of the compact kernel (q x w on the 384 unrotated dims, an elementwise op), the
  KV write, and the accuracy beyond the correctness cell.

## 5. Metrics
- Primary (FP8): for each group X, with dt(X) = t(X) - t(2K) at the group batch:
  - Selection (A1, R1): one configuration per kernel family and group, as the engine uses one configuration at all
    lengths: the fastest at X among the configurations that pass the correctness cell and ran at 2K and at X. The
    stock family takes the faster of `num_splits` 32 and 0 in this way.
  - dt (A1, O3): the median over the 5 confirmation repeats of t(X) - t(2K) of the selected configuration.
  - sigma_mma(X) = 1 - dt_compact / (0.85 x dt_stock): against a stock kernel 15% faster on the growth.
  - sigma_layout(X) = 1 - dt_compact / dt_standard: the two Triton layouts.
- Secondary (reported only): sigma_dq = 1 - dt_compact / dt_stock; the sigmas of each repeat and their range; the
  same for BF16 (the exact-cache case; one sweep measurement); the B = 4 times; the bandwidth on dt for each family;
  dt_standard / dt_stock (the Triton quality); the correctness errors and the errors against the exact attention
  (compact against standard storage); the selected split, warp and tile choices with their registers, spills and
  shared memory (A1, O5); sigma_layout at the best configuration that both Triton kernels share (A1, O4); the
  linearity of the 8K and 32K cells (A1, O9); the oracle elasticities e = 2 x (1 - dt_oracle / dt) for the stock
  kernel (e_stock) and the Triton standard kernel (e_triton) (A1, O1).
- Minimum effect that matters: the kill line (S = 5%) and the go line (S about 8%).

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU is H100 80GB.
- V2 (A1, R5): the stock FP8 kernel with 32 splits is within a relative L2 error of 0.02 of the reference of its
  stored values, and each Triton FP8 family has at least one configuration within 0.01. Before the selection, the
  analysis drops each Triton configuration above 0.01 (or without a result) and each stock kernel above 0.02.
- V3 (engine tie): the stock FP8 kernel with 32 splits is within 15% of the E-50 in-engine time per call at the four
  group cells (238.3, 1,358.1, 143.2 and 1,507.2 us), and (A1) its dt is within 15% of the in-engine dt (1,119.8 us
  at B 19 and 1,364.0 us at B 11).
- V4 (A1, R3): for each decision row, the median over the 5 repeats of the replay spread is at most 2%. The decision
  rows are the selected configuration of each FP8 family at 2K and X, and the stock kernel with 32 splits at the
  four V3 cells.
- V5 (positive control; A2): the copy bandwidth exists, and no decision row at X and no dt rate of an FP8 family
  reads above 1.15 x copy. The 2K rows are left out, because their cache can sit in L2 (Section 4).
- V6: hard throttle in at most 2% of busy telemetry samples.

## 7. Decision rule (frozen; A1 before any data)
Deciding groups: X = 32,768 (B 19) and 65,536 (B 11), FP8. Kill lines (R-68): 0.155 at 32K and 0.125 at 64K.
- GO: in at least one group, sigma_mma >= 0.25 and sigma_layout >= the kill line of that group. A GO allows the next
  steps only: the accuracy gate (R-68 recommendation 7) and an engine prototype plan with the page-size plan.
- KILL (A1, R2): sigma_layout below the kill line in both groups, and the Triton compact kernel reads at least 0.6 x
  copy on dt in both groups (the E-50 M1 bar). The layout itself gives too little, so the layout work stops; I-182
  stays as a capacity note (1.39 to 1.47 times the requests; FP8 KV already gives 1.96 times).
- INCONCLUSIVE: the benchmark did not finish (a status other than "ok" or "ok_groups"; A1, R4), a group or its
  confirmation is missing, the Triton standard kernel reads below 0.5 x copy on dt in a group (then the layout test
  is weak), or (A1, R2) sigma_layout is below the kill lines while the compact kernel reads below 0.6 x copy in a
  group (a defect of the compact kernel can explain the result).
- PIVOT: otherwise, with the reason "kernel quality" (sigma_layout >= 0.25 in a group: the layout works, and the next
  step is a compact kernel of FA4 quality) or "below the bar".
- Replication (A1, O3): the rule above runs on the medians and on each of the 5 repeats alone. The decision stands
  only if at least 4 of the 5 repeats give the same word (GO, KILL, PIVOT or INCONCLUSIVE). Otherwise the result is
  INCONCLUSIVE ("the decision does not replicate"): it is near a line.
- INVALID: V1 to V6 fail, no copy bandwidth, or (A1, R6) the fail-fast stop.

## 8. Controls and baselines
- Baseline: the stock FA4 FP8 kernel (the selected split setting), tied to its in-engine time and dt by V3.
- Layout control: the Triton standard kernel (the same code on the engine's K and V views).
- A/A (A1): the 5 confirmation repeats of each row give its noise and the noise of each dt and sigma.
- Positive control: V5 (no kernel reads faster than the copy bandwidth allows).
- Oracle (A1, O1; reported): the K = V rows give the time elasticity to DRAM bytes of the stock kernel and of the
  Triton standard kernel.
- Negative control (secondary): on the same K=V data, the compact and the standard storage give the same attention
  within the error of the dtype.

## 9. Budget (A1, R7)
- H100 SXM Secure at the live price (E-50: $3.99/h; S4 reads it again). Image pull 5 to 11 min (E-50: 11 min on a
  cold host). Time model (INTERPRETATION, R-69 answer 6 with the A1 changes): about 44 Triton compiles (2 to 4 min),
  4 to 10 FA4 variants (1 to 5 min), the sweep (3 to 6 min) and the confirmation (about 1 min): the benchmark takes
  about 7 to 16 min. Pod time about 22 to 33 min, about $1.50 to $2.20. Cost cap $3.
- Caps: `KB_TIMEOUT` 1,500 s; session watchdog 1,980 s; boot watchdog CAP 2,040 s (34 min after the container
  starts); `AUTO_STOP=1` with a 180 s delay (copy the results then). With an 11 min pull, the worst case is 45 min of
  Pod time, $3.00 at $3.99/h. If the pull takes longer, the main session stops the Pod 45 min after its creation.
- Rehearsal (S5, A1 R8): RTX 4090 ($0.89/h in E-50), the group cells only (`--cells 19:2048,19:32768,11:2048,11:65536`),
  `--stages-bf16 1` (the BF16 standard kernel needs about 128 KB of shared memory at 2 stages; the 4090 has 99 KB),
  `KB_TIMEOUT` 900, boot watchdog CAP 1,800 s, `AUTO_STOP=1`. About 20 min, cap $0.60.
- Balance: D-82 (a $11 floor) was for E-50 only, so the $35 floor applies again; S4 checks it live.

## 10. Threats to validity
- Triton quality: a slow Triton family hides the layout effect. The rule gives INCONCLUSIVE, not KILL, when the
  standard kernel reads below 0.5 x copy, or (A1) when sigma_layout is low and the compact kernel reads below 0.6 x
  copy.
- No stock FP8 MMA kernel in the image: sigma_mma uses a 15% allowance from the FA PR 199 table (R-68). R-69: the
  table shows 7% to 13% less time, and the PR #60372 test 0% to 1% end to end, so the allowance is conservative.
- The oracle rows (A1) do not decide: the stock kernel dequantizes FP8 to BF16 on chip, the coming FP8 MMA kernel
  does not, so its byte elasticity need not hold for that kernel; and the Triton oracle describes the family, not the
  layout. The oracle also assumes that L2 serves the second read of each page (HYPOTHESIS; no control kernel).
- The compact QK product runs over 640 dims (+25% multiply-adds against the 512 needed). This is conservative.
- The BF16 stock kernel uses 64-token pages (the engine: 32), which take the TMA path; the engine's BF16 page 32 took
  the non-TMA path (FACT, R-69). BF16 is secondary.
- No CUDA graph padding (A1, R-69 O7): the engine pads B 19 to 24 and B 11 to 16 graph slots. This adds a fixed cost
  that cancels in dt; V3 at 2K can read a few percent low.
- The query preparation is not timed; it is an elementwise op on B x 32 x 512 values.
- The microbenchmark runs one layer alone; the engine runs it inside a graph with other kernels. V3 ties them.
- Random cache values: these kernels have no data-dependent paths.
- One GPU, one session, decode only, uniform contexts.

## 11. Files
- `compact_attn.py`: the two Triton kernels (split and merge) and the PyTorch helpers (RoPE, K=V data, compact query
  and row, paged and split references).
- `kbench_e51.py`: the benchmark (stock and Triton, FP8 and BF16), the correctness cell, the confirmation phase;
  `--stub` runs the same flow with synthetic numbers for the Mac tests.
- `analyze_e51.py`: the selection, the sigmas, V1 to V6, the rule and the replication test.
- `session_e51.sh`: the Pod session (STUB=1 runs it on a Mac).
- `tl_shim.py` (A1): a CPU emulation of the Triton calls of `compact_attn.py` (from R-69). `cpu_dryrun_e51.py` (A1):
  runs the kernel source and the GPU code paths of `kbench_e51.py` on it.
- `test_e51.py`: 16 tests (Section 12).
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
- S3 after A1 (2026-10-31, Mac): the same command: 15 passed (60 s). New or changed tests:
  - the kernel source and the GPU code paths on the CPU emulation: every split and tile, both layouts, FP8 and BF16,
    B 1 to 3, L 64, 130, 2,049 and 3,000 (partial pages, empty splits): worst error 0.0021 against the reference of
    the stored values; the engine-layout views and the K = V oracle within 0.0023 of their references; `bench_cell`
    and `confirm` run, and the stock rows (no vLLM) come back as "error";
  - R1: one configuration at both lengths (dt 1,119.8 and 700 us where the old rule gave 1,207.8 and 740);
  - R2: a compact kernel at 0.44 x copy gives INCONCLUSIVE; a compact kernel at 0.62 x copy with no layout gain
    gives KILL;
  - R3: a noisy row that does not decide leaves V4; a noisy decision row gives INVALID (V4);
  - R4: SIGTERM after the confirmation leaves "ok_groups", and the analysis decides;
  - R5: a wrong fastest compact configuration is dropped and never selected;
  - R6: fail fast gives INVALID (V2) with no timed cell; without it, V2 fails;
  - O3: GO on the medians but in 3 of 5 repeats gives INCONCLUSIVE.
- S5 plan (A1): RTX 4090, the image above, the settings of Section 9. VERIFY:
  1. The image starts; `import triton, vllm` works; `versions.txt` has the Triton and `nvidia-cutlass-dsl` lines.
  2. Each FP8 Triton kernel has a passing configuration at 2 stages and each BF16 kernel at 1 stage; `kbench.json`
     holds the reason of every failure. The Triton errors on the correctness cell are at most 0.005.
  3. The registers, spills and shared memory of the tile-32 configurations at 4 and 8 warps, both layouts. If only
     the compact kernel spills, sigma_layout is biased: fix that before the H100.
  4. The group cells run at full size. The Triton standard FP8 kernel reads at least 0.5 x the 4090 copy rate on dt;
     if not, stop and fix the kernel (the H100 would give INCONCLUSIVE).
  5. The oracle rows exist; the Triton K = V row is faster than the standard row (e_triton above 0.4 on this
     memory-bound GPU), or the oracle premise is weak.
  6. `run.json` shows "ok_groups" after the confirmation and "ok" at the end.
  7. The stock FP8 rows have status "error" with the FA4 reason; fail fast is off on SM89; the analysis gives the
     expected INVALID (V1, V2, V3).
  8. The largest V4 spread of the decision rows and the share of all rows above 2%. If many rows exceed 2%, change
     the timing before the H100.
  9. AUTO_STOP stops the Pod before the boot watchdog fires.
  10. A rerun skips a finished benchmark; a failed run keeps its partial data. Record the Triton times for the H100
      time model.
  Not testable on the 4090: the stock FA4 call, V3, the H100 times and bandwidths, and fail fast on SM90.
- S5 record (2026-11-01, PASS with A2): Pod yqbmujou3dt42q, RTX 4090 Secure, EU-RO-1, $0.89/h, created 02:25 UTC,
  image pull about 1.5 min, session 02:28 to 02:31 UTC (`kbench` 172 s: the correctness cell, the compiles, the four
  group cells and the confirmation), stopped by AUTO_STOP, results copied (27 files, checksums match), Pod deleted.
  About $0.14 (estimate; billing pending). Results: `results/e51-rehearsal-4090-20261101/`. VERIFY:
  1. PASS: Triton 3.8.0, vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, nvidia-cutlass-dsl 4.8.0,
     quack-kernels 0.6.5 in `versions.txt`.
  2. PASS: all 16 FP8 configurations of each Triton kernel at 2 stages and all 4 BF16 configurations at 1 stage
     compiled and passed; the largest error is 0.0019 (FP8) and 0.0019 (BF16), below 0.005.
  3. PASS: no spills in any configuration; registers 128 (standard FP8, 8 warps) and 175 (compact FP8, 8 warps),
     254 and 255 at 4 warps; shared memory 50 KB and 57 KB at tile 32.
  4. PASS: both FP8 Triton kernels are memory-bound on dt: the standard kernel at 943 and 945 GB/s, the compact kernel
     at 931 and 915 GB/s (copy 920 GB/s). sigma_layout = 0.367 at 32K and 0.355 at 64K (the byte share is 0.375).
  5. PASS: e_triton = 0.885 (32K) and 0.991 (64K); the K = V row reads about half the DRAM bytes.
  6. PASS (by test): the run ended "ok"; with the group cells only, the confirmation is the last step, and the
     "ok_groups" write is in the shared main loop (the R4 test).
  7. PASS: the stock FP8 rows failed with "q.dtype torch.bfloat16 != k.dtype torch.float8_e4m3fn": vLLM's wrapper
     turns on the FP8 dequant path only at capability 9.x (FACT, `flash_attn_interface.py` at 3966924d5). The stock
     BF16 rows failed with "paged KV not supported on SM 8.0". Fail fast was off. The analysis gave INVALID (V1, V2,
     V3, V5); V5 was a false alarm (A2), and under A2 it gives the expected INVALID (V1, V2, V3).
  8. PASS: the largest median replay spread of the decision rows is 0.81%; 2 of 160 sweep rows are above 2% (8.2% at
     most), and they do not decide.
  9. PASS: AUTO_STOP stopped the Pod (EXITED) before the boot watchdog time.
  10. PASS (by test): the rerun skip and the partial-data move are in the stub session test. Time model: the four
      group cells took 172 s on the RTX 4090.
- S3 after A2 (2026-11-01, Mac): 16 passed (a new V5 test: an L2-resident 2K row leaves V5; a fast X row fails it).

## 13. Result (S6 to S8, 2026-11-01 UTC; post-run review R-70: CONFIRM_WITH_CORRECTIONS)
- Run (FACT): Pod kxu827n54y2lb0, H100 80GB HBM3 (sm_90), US-NE-1, Secure, $3.99/h, created 03:14 UTC. The image pull
  took about 11 min (the first 520 s SSH wait timed out; the second try passed). Session 03:29 to 03:33 UTC (`kbench`
  228 s). Results copied (27 files, checksums match); the Pod was deleted in the AUTO_STOP delay (about 03:35 UTC).
  $1.42 billed (the first estimate was about $1.40); E-51 total $1.56 with the rehearsal. Design commit a1e8b13 (A2).
  Copy 3,045 GB/s. Results: `results/e51-h100-20261101/`.
- **Frozen decision: INVALID (V3, V4).** The run does not decide. R-70 confirms it: `decide()` stops at a failed
  check before it applies the rule.
  - V3: the stock FP8 kernel with 32 splits at 19:2K takes 0.837 x its in-engine time (the bound is 0.85); at 11:2K
    0.863; at 19:32K 0.993; at 11:64K 0.975. Its dt ties to the engine within 2.5% (1.025 and 0.986). FACT: the
    fixed cost at 2K is lower here than in the engine (no graph padding, no other kernels; Section 10). FACT (R-70):
    the 2K check sits on its noise edge. The sweep rows give 0.853 (19:2K) and 0.848 (11:2K), so pass and fail swap
    between the sweep and the confirmation. The shortfall is 14% to 16%, not "a few percent" as Section 10 predicted.
  - V4: the stock row at 19:32K has a median replay spread of 4.2% (the bound is 2%), and the stock row at 11:64K
    2.7%. 27 of 392 sweep rows are above 2% (17% at most). FACT (telemetry): the SM clock fell from 1,980 to 1,530 MHz
    at times, and power reached 701 W under the software power cap (reason 0x4). The RTX 4090 spreads were 0.8% at most.
    FACT (R-70): 20 of the 137 busy samples (14.6%) show the cap; their mean SM clock is 1,918 MHz. V6 counts only
    the hard reasons (0x8, 0x20, 0x40, 0x80), so V6 cannot see the cap. INTERPRETATION (R-70): the cap slows the
    fast kernels more than the slow compact kernel, so the bias favours the compact kernel and cannot change the sign.
  - V1, V2, V5 and V6 pass. All 40 Triton configurations and the four stock kernels pass the correctness cell (error
    0.0019 at most). A2 did not change the result: the V5 before A2 also passes (largest rate 2,270 GB/s; R-70).
- Exploratory readout (not a decision; checked by R-70). The numbers are FACT; the meaning is INTERPRETATION.
  - The compact Triton FP8 kernel is slower than its standard twin: sigma_layout -0.285 (32K) and -0.289 (64K),
    sigma_dq -0.137 and -0.179, sigma_mma -0.338 and -0.387. On dt it reads at 0.376 and 0.370 x copy; the standard
    Triton FP8 kernel at 0.773 and 0.764 x copy; the stock FA4 FP8 kernel at 0.684 and 0.699 x copy. Without V3 and
    V4, the A1 rule gives INCONCLUSIVE in 5 of 5 repeats (the compact kernel is below 0.6 x copy). The rule before A1
    would have given KILL (R-69 R2).
  - The compact FP8 defect is specific to FP8 (R-70). At the 16 matched FP8 configurations, the compact time at X is
    1.04 to 1.52 x (32K) and 1.05 to 1.55 x (64K) the standard time (medians 1.30 and 1.33). At the 4 matched BF16
    configurations it is 0.66 to 0.79 x and 0.63 to 0.80 x (the byte ratio is 0.625). The compact FP8 kernel is
    slower than its own BF16 version at half the bytes: dt 1,305 against 1,010 us (32K) and 1,586 against 1,143 us
    (64K). INTERPRETATION (R-70): after the cast, the V tile feeds two dot roles, and each role needs its own
    register layout, so each tile goes through shared memory more times. The 255-register ceiling and the zero
    columns of `qc` (+25% QK multiply-adds) add to it. This is a defect of the implementation, not of the layout.
  - The standard Triton FP8 kernel is faster on dt than the stock FA4 FP8 kernel: dt 0.885 x (32K) and 0.915 x (64K),
    in all 5 repeats. The Triton family had 16 configurations to choose from; the stock family had 2.
  - The oracle (K = V) removes DRAM bytes only. FACT (code; R-70): it keeps the casts, the MMA, the staged tiles, the
    shared-memory fill and the L2-to-SM bytes. It cuts dt by 10.1% and 8.3% for the stock kernel (e_stock 0.202 and
    0.165) and by 8.9% and 9.8% for the Triton standard kernel (e_triton 0.179 and 0.196). The same Triton oracle gave
    0.885 and 0.991 on the RTX 4090, which is DRAM-bound (0.92 TB/s copy).
  - Three elasticities differ (R-70). The dtype elasticity (FP8 against BF16, half the bytes, plus the cast) is 0.50
    for the stock kernel and 0.64 to 0.67 for the Triton standard kernel. The K = V elasticity is 0.17 to 0.20. The
    elasticity of the real compact layout is not measured.
  - FACT (R-70; FA4 source at b273a02): the stock FP8 path stages K and V through one 32 KB shared buffer with a
    one-stage TMA pipeline, casts FP8 to FP16 (the Triton kernels cast to BF16) and uses 64 x 64 tiles. The 8 query
    heads of a KV head fill at most 8 of the 64 MMA rows (INTERPRETATION: the vLLM wrapper keeps this default).
  - INTERPRETATION: bytes explain at most about 20% of dt in the two FP8 dequant kernels on the H100. "Bound on chip"
    is an INTERPRETATION; the candidates are the cast, the MMA with M padded from 8 to 64, and the one-stage staging
    chain. The byte part of the layout gain is then about 0.2 x 0.375 = 7.5% of dt, below the kill lines (0.155 and
    0.125). The real layout also casts 640 values per token instead of 1,024 (37.5% less cast work) and can stage
    fewer bytes per block. E-51 did not measure that part (R-70 H3). HYPOTHESIS (R-70 H2, possible but unlikely): the
    oracle does not halve the DRAM bytes on the H100; E-51 read no DRAM counter. The BF16 rows (1.02 to 1.05 x copy)
    rule out an L2 bandwidth limit.
  - BF16 (one sweep measurement): both Triton BF16 kernels read at about 1.0 x copy on dt, and sigma_layout is 0.337
    (32K) and 0.368 (64K). The layout effect is real when the kernel is bound by bytes. The compact BF16 dt (1,010
    and 1,143 us) is also 34% and 36% below the stock BF16 dt (1,536 and 1,790 us).
  - The selected configuration is 32 splits, 4 warps, tile 32 for each FP8 family; no spills; the 4-warp
    configurations of both layouts use 255 registers. FACT (R-70): this configuration sits at the edge of the grid
    in splits, warps and tile, for both kernels and both groups, and `num_stages` stayed at 2. The best speed of the
    Triton family is not known.
  - B 4 (reported): the stock FP8 kernel takes 287.6 us (32 splits) and 269.5 us (automatic) at 32K; the Triton
    standard kernel 235.8 us; the compact kernel 309.1 us.
- What this means (INTERPRETATION, corrected by R-70): E-51 bounds the byte route only. On one H100, with the two FP8
  dequant kernels at B 11 to 19, halving the DRAM bytes cuts dt by at most about 10%. E-51 does not show that the
  compact layout cannot be faster: no competent compact FP8 kernel exists yet, and the oracle is not the layout. E-51
  says nothing about the FP8 MMA kernel of FA PR 199 (in the open vLLM PR 60372), other GPUs, batches or real data.
  If the speed work stops, the record says "speed claim untested", not "no speed benefit". The capacity result of
  I-182 (1.39 to 1.47 x the requests, E-50) is unchanged.
- Lesson (R-70, RESEARCH_OUTPUTS L-18): the RTX 4090 rehearsal could not show the on-chip defect. It has 3.3 x less
  DRAM bandwidth, so the compact kernel passed S5 gate 4 at 1.01 and 0.99 x copy, and read 0.37 x copy on the H100.
- R-70 (S8, 2026-11-01, Sonnet): CONFIRM_WITH_CORRECTIONS. Every number in this section recomputes from the raw rows
  with an independent script. Write-up: `dossier/REVIEW_E-51_POSTRUN_2026-11-01.md`. Its fixes for a next run: V3 on
  the X cells and dt only (report the 2K ratios); V4 with 30 replays and the spread of the central 20, plus a 3%
  bound on the across-repeat dt range; the SM clock per replay set from NVML; the 0x4 cap share in V6.
- Next: the user decides the next step (D-85).

## Amendments
- A1 (2026-10-31, before any data; from R-69, verdict AMEND_THEN_RUN). Each item has a test (Section 12).
  - R1 (blocker): one configuration per family and group, at both lengths (Section 5).
  - R2 (blocker), changed from the review: a KILL needs the compact kernel at 0.6 x copy on dt in both groups;
    otherwise a low sigma_layout gives INCONCLUSIVE (Section 7). The review also offered a KILL by the Triton oracle
    (sigma_alias below the kill line / 0.375 x 0.5). Not adopted: the Triton oracle describes the family, not the
    layout, and the stock oracle describes the dequant kernel, not the coming FP8 MMA kernel. The oracle rows are
    reported only (Section 10).
  - R3 (blocker): V4 on the decision rows (Section 6).
  - R4 (blocker): the group cells first; "ok_groups" after the confirmation; a SIGTERM is recorded (Section 4).
  - R5 (blocker): every swept configuration in the correctness cell; failed configurations are dropped before the
    selection; the Triton bound is 0.01 (a dropped page gives about 0.02), the stock bound stays 0.02 (Section 6).
  - R6: fail fast after the correctness cell (Section 4).
  - R7: `KB_TIMEOUT` 1,500 s, session watchdog 1,980 s, boot watchdog 2,040 s, cost cap $3 (Section 9).
  - R8: `--stages` and `--stages-bf16`; the rehearsal uses 1 for BF16 and the group cells (Section 9).
  - O1: the oracle rows, reported (Sections 4, 5, 8, 10). O3: the confirmation phase and the replication test
    (required by `MEASUREMENT.md` Section 1.5; the frozen design had one measurement per configuration). O4, O5, O9:
    reported metrics. O6 (part): BF16 at the group cells with 4 configurations. O7 (part): the engine's cache layout.
    O8: the package versions. Also V3 on dt, and the block table rows padded to 16 entries (one Triton
    specialization for all cells).
  - Not adopted: O1's read-only control kernel (the oracle does not decide); O2, the PR 199 FP8 MMA overlay (it
    cannot be rehearsed on the 4090 and adds time to a $3 cap; it is the next step if E-51 gives GO, or PIVOT near
    the go line); O6 part 2, a runtime split count (it changes the kernel that R-69 checked; the compile time is small
    after the BF16 cut); O7 part 2, the graph-slot padding (it cancels in dt; Section 10).
  - Unchanged: the question, the hypotheses, the groups and the cells, the primary metrics, the kill and go lines,
    the 15% FP8 MMA allowance and the stock call.
- A2 (2026-11-01, from the S5 rehearsal, before the H100 run; for the post-run review to check):
  - V5 uses the decision rows at X and the dt rates of the FP8 families, not the 2K rows (Sections 4 and 6). Reason:
    the rehearsal failed V5 only because the compact FP8 rows at B 11 and 2K (57.7 MB) sat in the 72 MB L2 of the
    RTX 4090 (1,061 GB/s, 1.15 x copy). On the H100 (50 MB L2) the same row is close to the L2 size.
  - The confirmation phase starts with one untimed pass over all rows (Section 4). Reason: the first timed repeat of
    the first row was 5.5% slow (196.5 against 186.2 us).
  - Unchanged: the rule, the metrics, the other checks and the cells.
