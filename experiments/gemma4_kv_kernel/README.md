# E-51: a compact K=V decode attention kernel for Gemma 4 against the stock vLLM kernel, on 1 x H100 (preregistration)

Status: PREDECLARED 2026-10-31. AMENDED A1 2026-10-31 (R-69 pre-run review, before any data). NOT RUN.
Idea: I-182. Journal entry: Section 6.1 E-51. Follows E-50 and its post-run review R-68 (recommendation 1 to 6).
Request: D-83 (user, 2026-10-31, "Plan the kernel test": a compact FP8 attention kernel timed against the stock vLLM
kernels at batch 19 / 32K and batch 11 / 64K, with the R-68 stop line; plan only, no GPU yet).
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.
Text marked "(A1)" changed in Amendment A1. The version before A1 is commit 0fb1b39.

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
  5) and each oracle row is captured in a new graph and timed 5 times. Odd repeats run the rows in the reverse order
  (A B .. B A). Then `run.json` gets the status "ok_groups"; the reported cells follow.
- L2 (A1, O9): the group caches are larger than the 50 MB L2, except the compact FP8 rows at B 11 and 2K (57.7 MB,
  close to it). At B 4 and 2K the standard FP8 cache (33.6 MB) fits in L2 (reported cells only). INTERPRETATION
  (R-69): the effect on dt is below 1%.
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
- V5 (positive control): the copy bandwidth exists, and no decision row reads above 1.15 x copy.
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
- `test_e51.py`: 15 tests (Section 12).
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
