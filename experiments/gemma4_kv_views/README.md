# E-52: fixed compact K=V decode attention kernels (FP16 compute, two overlapping views) against their standard twin and the stock vLLM kernel, on 1 x H100 (preregistration)

Status: PREDECLARED 2026-11-01. NOT RUN.
Idea: I-182. Journal entry: Section 6.1 E-52. Follows E-51 (frozen decision INVALID; exploratory readout) and its
post-run review R-70 (CONFIRM_WITH_CORRECTIONS). Request: D-85 (user, 2026-11-01, "1 go ahead": fix the compact
kernel with free work first, then one H100 test with the R-70 controls; the spend waits for D-86).
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
On 1 x H100, with an FP8 KV cache, does the compact K=V layout (640 of 1,024 values per head and token), read by the
standard Triton decode kernel through two overlapping views and with FP16 compute, cut the context-growing attention
time of a Gemma 4 31B global layer by at least the kill line against the same kernel on the standard layout (15.5% at
32K, 12.5% at 64K), and by the go margin against the stock FA4 FP8 kernel? "No" (KILL): sigma_layout is below the kill
lines in both groups while the compact kernel reads at least 0.6 x copy on dt.

## 2. Rubber duck
- E-51 (FACT, README Section 13 there, R-70): the compact Triton FP8 kernel was 29% slower than its twin. Its BF16
  version was 34% to 37% faster than its twin, so the defect is specific to FP8. The Triton standard FP8 kernel read at
  0.77 x copy on dt, and halving its DRAM bytes (K = V) cut dt by only 9% to 10% (e_triton 0.18 to 0.20; 0.9 to 1.0 on
  the RTX 4090). So the E-51 kernels were bound on chip on the H100.
- New FACT (Triton 3.8.0, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp`): FP8 E4M3 to FP16 is
  one `cvt.rn.f16x2.e4m3x2` per 2 values. FP8 E4M3 to BF16 on sm_90 is that instruction plus two scalar
  `cvt.bf16.f16`. The E-51 kernels cast to BF16. FACT (FA4 source at b273a02, R-70): the stock FA4 FP8 path computes in
  FP16 ("the single-instruction fp8->fp16" path).
- FACT (FlashMLA Hopper FP8 note, 2025-09-29): on an H800 (also SM90) an FP8 decode kernel was bound by the
  dequantization, at about 50 clock cycles per 512 values against 34 for the MMA.
- HYPOTHESIS H4 (main session): the BF16 cast sets most of the E-51 Triton FP8 time. Arithmetic (INTERPRETATION): one
  token and KV head has 1,024 values, so the BF16 path has 1,024 scalar conversions. If they run at 16 per clock and
  SM (the rate that the FlashMLA note uses for its float to BF16 step), they take 64 clocks, against a DRAM floor of
  88 clocks and a measured 114 clocks (R-70 arithmetic). The FP16 path needs 512 packed conversions. The low K = V
  elasticity fits H4: halving the bytes does not remove the casts.
- New layout (the views). E-51's fused kernel loads V once and uses it in two dot roles; R-70 (INTERPRETATION) names
  this as the most likely FP8 defect. The views layout stores each row as [K_rot (128) | V_unrot (384) | V_rot (128)].
  Its first 512 values are the K of the standard kernel ([K_rot | V_unrot], with the k_norm weight w moved into the
  query) and its last 512 values are the V ([V_unrot | V_rot]). So the unchanged standard kernel reads the compact
  cache, with one load per dot role, and the 384 shared values come from L2 on the second read. The compact kernel then
  has the same instructions as its twin, and only the DRAM bytes differ (640 against 1,024 per token). The output comes
  in the V-view order; an engine folds that order into the o_proj columns (exact, no run-time cost).
- The stock FA4 kernel reads the same views ("_views" rows, reported): the engine already passes K and V as strided
  views. INTERPRETATION: its gain is about 0.375 x e_stock, so about 7.5% of dt, because the stock kernel is bound on
  chip (E-51).
- What would kill the layout: a twin and a views kernel that read at least 0.6 x copy, with sigma_layout below the
  kill line in both groups (for example, the two views do not share their DRAM reads). If FP16 compute does not make
  the pair bound by bytes, the result is INCONCLUSIVE (the E-51 rule), and the speed question stays open.
- Expected (HYPOTHESIS): H4 holds; the Triton standard FP16 kernel reads at 0.85 x copy or more on dt; sigma_layout
  0.25 to 0.37; GO.
- Not in E-52 (D-85): R-70's option (b), the FP8 MMA kernel of FA PR 199. sigma_mma keeps the 15% allowance for it.

## 3. Hypotheses
- H1 (primary, the E-51 rule): in at least one group, sigma_mma >= 0.25 and sigma_layout >= the kill line.
- H4 (mechanism, reported): with FP16 compute, the Triton standard FP8 kernel reads at 0.85 x copy or more on dt, and
  its K = V elasticity e_h is 0.6 or more.
- H0: sigma_layout is below the kill lines (KILL if the compact kernel reads at 0.6 x copy or more), or the pair stays
  bound on chip (INCONCLUSIVE).

## 4. Setup (frozen)
- Hardware: 1 x H100 80GB HBM3 (SXM), Secure Cloud, on-demand, `minCudaVersion` 13.0. Rehearsal: RTX 4090 (Triton
  only; the FA4 FP8 dequant path asserts SM90).
- Software: the E-49 to E-51 image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
  (vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, Triton 3.8.0, pynvml from the image). No model, no engine.
- Shapes (as E-51): one global layer of Gemma 4 31B: 32 query heads, 4 KV heads, head_dim 512, scale 1.0, no
  soft-cap, decode, uniform contexts, 64-token pages, a random block table (the same for every kernel, its rows padded
  to a multiple of 16 entries).
- Cache layouts:
  - standard (the engine's): one (blocks, KV heads, 64, 2 x 512) tensor; K and V are `transpose(1, 2).split(512)`.
  - views: one (blocks, KV heads, 64, 640) tensor in the row order [K_rot | V_unrot | V_rot]
    (`compact_attn.pack_views`); K view = values 0 to 511, V view = values 128 to 639 (`views_of`); query
    `views_query`; output order [unrotated dims | rotated dims] (`unpermute`).
  - fused (E-51): the same tensor read as [V | K_rot].
- Kernels (`kbench_e52.py`; Triton in `../gemma4_kv_kernel/compact_attn.py`, `cast` "bf16" as E-51 or "fp16"):
  - stock: the E-51 FA4 call; FP8 and BF16; `num_splits` 32 and 0; on the standard cache and on the views ("_views").
  - primary Triton families (FP8 cache, FP16 compute): `triton_std_fp8h` (standard) and `triton_cmpv_fp8h` (views,
    the same compiled kernel). Wide sweep at the group cells and the correctness cell (R-70 recommendation 5): splits
    {16, 32, 64} x warps {2, 4, 8} x tile {16, 32, 64} tokens x `num_stages` {2, 3} (54 configurations). At the
    reported cells: 4 configurations (32 or 64 splits, 4 warps, tile 16 or 32, 2 stages).
  - diagnostic Triton kernels (FP8 cache, at the E-51 configuration 32/4/32/2 only): `triton_std_fp8` (the E-51 twin,
    BF16 compute), `triton_cmpv_fp8` (views, BF16), `triton_cmp_fp8` (the E-51 fused kernel), `triton_cmp_fp8h` (fused,
    FP16). With the primary rows at 32/4/32/2 they make a cast x layout factorial.
  - BF16 cache (secondary, group cells only, the E-51 BF16 grid of 4 configurations): `triton_std_bf16`,
    `triton_cmpv_bf16`, `triton_cmp_bf16`.
  - K = V rows (timing only; the V view as K and V): in the sweep for BF16 (`stock_bf16_s32_alias`,
    `triton_std_bf16_alias`; R-70 control), in the confirmation for FP8 (`stock_fp8_s32_alias`,
    `triton_std_fp8h_alias` at the selected configuration).
  - pool rows (timing only; R-70 probe; confirmation): `stock_fp8_s32_pool`, `triton_std_fp8h_pool`,
    `triton_cmpv_fp8h_pool`. Each block table entry maps into a pool of 8 MB of pages, so after the first call the bytes
    come from L2.
- Timing: each configuration is captured as 20 calls in one CUDA graph and replayed 30 times; time per call = replay
  time / 20 (median). Spread (R-70) = max / min - 1 of the central 20 of the 30 replays. NVML SM clock and clock event
  reasons after every 10th replay (no root needed). The first eager call of each Triton configuration records its
  registers, spills and shared memory; at 19:32K, the PTX and TTGIR of each Triton kernel at 32/4/32/2 are saved with
  instruction counts.
- Order (as E-51 A1): the correctness cell, the four group cells (19:2K, 19:32K, 11:2K, 11:64K), the confirmation
  phase (then `run.json` has "ok_groups"), then the reported cells 19:8K, 11:8K, 4:2K, 4:32K. At 4:2K the standard FP8
  cache (33.6 MB) fits in the L2 of both GPUs.
- Confirmation (as E-51 A1 and A2): after one untimed pass, each decision row and each K = V and pool row is captured
  in a new graph and timed 5 times; odd repeats run in the reverse order.
- Correctness cell (as E-51, seed 52): B 2, L 3,000, Gemma 4 K=V data stored in FP8 and BF16; every configuration of
  every Triton kernel and the stock kernels, each against the float32 reference of its own stored values (the views
  kernels against the views reference), and against the exact attention (the views outputs after `unpermute`). The
  output is zeroed before each run.
- Fail fast (as E-51 A1): if the stock FP8 kernel with 32 splits or a primary Triton family fails the correctness
  cell, the run stops (`--fail-fast auto`: on SM90).
- Not timed: the query preparation (an elementwise op on B x 32 x 512 values), the KV write, the o_proj order.

## 5. Metrics
- Primary (FP8), as E-51 (A1) with the primary families above: for each group X, dt(X) = t(X) - t(2K) at the group
  batch. Selection: one configuration per family and group, the fastest at X among the configurations that pass the
  correctness cell and ran at 2K and at X. dt = the median over the 5 repeats of the per-repeat differences.
  - sigma_mma = 1 - dt_compact / (0.85 x dt_stock).
  - sigma_layout = 1 - dt_compact / dt_standard (the same compiled kernel on the two layouts).
- Secondary (reported only): sigma_dq = 1 - dt_compact / dt_stock; the sigmas of each repeat and their range; the dt
  rate of each family; sigma_stock_layout = 1 - dt(stock s32 views) / dt(stock s32); the K = V elasticities
  e = 2 x (1 - dt_alias / dt) for the stock kernel (e_stock) and the selected standard FP16 configuration (e_h), and
  for BF16 (e_bf16, from the sweep); the pool shares 1 - dt_pool / dt; the cast x layout factorial at 32/4/32/2 (sweep
  values); BF16 (one sweep measurement, with the fused BF16 kernel); the B 4 times; the linearity at 8K; the views over
  standard time at 4:2K; the PTX counts; the NVML clocks of the decision rows; the share of busy telemetry samples
  under the software power cap (0x4).
- Minimum effect that matters: the kill line (S = 5%) and the go line (S about 8%), as E-51 (R-68).

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU is H100 80GB.
- V2 (as E-51 A1): the stock FP8 kernel with 32 splits is within 0.02 of the reference of its stored values, and each
  primary Triton family has at least one configuration within 0.01. Before the selection, the analysis drops each
  Triton configuration above 0.01 (or without a result) and each stock kernel above 0.02; a K = V or pool row follows
  its base kernel.
- V3 (R-70 fix): the stock FP8 kernel with 32 splits is within 15% of the E-50 in-engine time per call at 19:32K
  (1,358.1 us) and 11:64K (1,507.2 us), and its dt is within 15% of the in-engine dt (1,119.8 and 1,364.0 us). The 2K
  ratios are reported without a bound (E-51: 0.837 and 0.863, a fixed cost that the engine has and this benchmark
  lacks).
- V4 (R-70 fix): for each decision row, the median over the 5 repeats of the central spread is at most 2%; and for
  each primary family, the range of its 5 per-repeat dt values is at most 3% of their median. The decision rows are
  the selected configuration of each primary family at 2K and X, and the stock kernel with 32 splits at 2K and X.
- V5 (as E-51 A2): the copy bandwidth exists, and no decision row at X and no dt rate of a primary family reads above
  1.15 x copy.
- V6: hard throttle (0x8, 0x20, 0x40, 0x80) in at most 2% of busy telemetry samples. The software power cap (0x4) is
  reported, not bounded (R-70: E-51 had 14.6% of busy samples under it).

## 7. Decision rule (frozen)
The E-51 rule (Section 7 there, A1), with the E-52 primary families. Deciding groups: X = 32,768 (B 19) and 65,536
(B 11), FP8. Kill lines (R-68): 0.155 at 32K and 0.125 at 64K.
- GO: in at least one group, sigma_mma >= 0.25 and sigma_layout >= the kill line of that group. A GO allows the next
  steps only: the accuracy gate (R-68 recommendation 7) and an engine prototype plan (the views layout, the query and
  o_proj order, the page-size plan).
- KILL: sigma_layout below the kill line in both groups, and the views kernel reads at least 0.6 x copy on dt in both
  groups. The layout gives too little with kernels that read near the bytes, so the speed work stops; I-182 stays a
  capacity result.
- INCONCLUSIVE: the benchmark did not finish (a status other than "ok" or "ok_groups"), a group or its confirmation
  is missing, the Triton standard kernel reads below 0.5 x copy on dt in a group, or sigma_layout is below the kill
  lines while the views kernel reads below 0.6 x copy in a group (the on-chip limit of the kernels can explain it).
- PIVOT: otherwise, with the reason "kernel quality" (sigma_layout >= 0.25 in a group) or "below the bar".
- Replication (as E-51 A1): the rule runs on the medians and on each of the 5 repeats alone. The decision stands only
  if at least 4 of the 5 repeats give the same word. Otherwise INCONCLUSIVE ("the decision does not replicate").
- INVALID: V1 to V6 fail, no copy bandwidth, or the fail-fast stop.

## 8. Controls and baselines
- Baseline: the stock FA4 FP8 kernel (the selected split setting), tied to the engine by V3.
- Layout control: `triton_std_fp8h`, the same compiled kernel as `triton_cmpv_fp8h`; only the addresses differ.
- A/A: the 5 confirmation repeats of each row.
- Positive controls: V5; the BF16 K = V rows (e_bf16 near 1 if K = V halves the DRAM bytes, R-70); the pool rows.
- Negative control: the views and standard kernels give the same attention within the error of the dtype (the
  correctness cell, after `unpermute`).
- Mechanism control (reported): the cast x layout factorial at 32/4/32/2 separates the effect of FP16 compute from
  the effect of the layout, and ties E-52 to the E-51 kernels (`triton_std_fp8`, `triton_cmp_fp8`).

## 9. Budget
- H100 SXM Secure at the live price (E-51: $3.99/h; S4 reads it again). Image pull 5 to 11 min (E-51: about 11 min).
- Time model (INTERPRETATION): about 65 Triton compiles in the correctness cell (3 to 5 min; the views kernel shares
  the compiled standard kernel), 4 to 10 FA4 variants (1 to 5 min), the sweep (about 4 to 6 min) and the confirmation
  (about 2 min): the benchmark takes about 10 to 18 min. Pod time about 25 to 35 min, about $1.70 to $2.40. Cost cap
  $3.
- Caps (as E-51): `KB_TIMEOUT` 1,500 s; session watchdog 1,980 s; boot watchdog CAP 2,040 s after the container
  starts; `AUTO_STOP=1` with a 180 s delay. With an 11 min pull, the worst case is 45 min of Pod time, $3.00 at
  $3.99/h. The group cells and the confirmation come first, so a timeout keeps the decision data.
- Rehearsal (S5): RTX 4090 (E-51: $0.89/h), the group cells and 4:2K, `--fail-fast off`, `KB_TIMEOUT` 900, boot
  watchdog CAP 1,800 s, `AUTO_STOP=1`. About 20 min, cap $0.60. VERIFY items: every Triton layout and compute dtype
  compiles and passes the correctness cell; the PTX files exist and show `cvt.rn.f16x2.e4m3x2` in the FP16 kernels;
  the NVML clock is read; at 4:2K the views time is within 5% of the standard time (the L-18 gate: an L2-resident
  shape, where DRAM does not hide an on-chip difference); the time model of the group cells.
- Balance: the $35 floor applies again (D-84's $14 floor was for E-51 only); S4 checks it live.

## 10. Threats to validity
- The views share their DRAM reads only if the second read finds the lines in L2 or in flight (HYPOTHESIS; no DRAM
  counter). The dt rate on 640 bytes, the pool rows and the stock views rows report it. If they do not share, the
  views kernel reads 1,024 bytes per token, sigma_layout is near 0 and its rate on 640 bytes is near 0.6 x the
  standard rate, so the rule gives INCONCLUSIVE or KILL; the PTX counts and the rates then tell which.
- FP16 range: the query, P and the cache values must fit FP16. FA4 makes the same choice for this path. The
  correctness cell checks it on Gemma 4 data (q ~ N(0, 1/512), RMS-normed V); real activations are not tested here
  (the accuracy gate after a GO).
- The wide sweep has 54 configurations per family: one configuration per family is selected and then confirmed in 5
  repeats, as E-51 (L-15). The selected configuration can sit at the grid edge again; that is reported.
- The stock FA4 kernel may reject the views strides; then the "_views" rows fail and are reported.
- The FP8 MMA stock kernel is not in the image; sigma_mma keeps the 15% allowance (R-68, R-69).
- Power-cap noise (L-17): the central spread, the dt range bound and the NVML clocks.
- No CUDA graph padding (as E-51): the fixed cost cancels in dt; V3 no longer bounds the 2K ratios.
- The query preparation and the o_proj order are not timed.
- One GPU, one session, decode only, uniform contexts, random data.

## 11. Files
- `kbench_e52.py`: the benchmark (cells, sweeps, correctness, confirmation, NVML clocks, PTX dumps, stub mode).
- `analyze_e52.py`: the frozen analysis and rule (summary.json and the decision).
- `session_e52.sh`: the Pod session (records, telemetry, copy bandwidth, the benchmark, the analysis, watchdogs).
- `cpu_dryrun_e52.py`: the GPU code paths on the CPU emulation of Triton.
- `test_e52.py`: the tests (the views identity, the CPU emulation, every decision path, the R-70 fixes, a stub session).
- `../gemma4_kv_kernel/compact_attn.py` (the kernels: `cast`, the views helpers; the E-51 default is unchanged) and
  `tl_shim.py` (FP16 added).
- Upload folders: gemma4_kv_views, gemma4_kv_kernel, runpod, fp8_quant_fusion, nvfp4_dispatch, safedispatch_testA,
  safedispatch_testA2, ledger.

## 12. Verification record (stage S3 and S5)
- S3 (2026-11-01, Mac, no GPU and no Triton): `uv run --no-project --with numpy --with torch --with pytest python -m
  pytest test_e52.py`: 19 passed. `test_e51.py`: 16 passed (the E-51 kernels and rule are unchanged). The CPU dry run:
  every Triton kernel (standard, views and fused layouts; BF16 and FP16 compute; FP8 and BF16 cache) at B 1 to 3,
  partial pages and empty splits is within 0.0021 of the reference of its stored values (FP16 compute 0.0017 at most);
  the views outputs after `unpermute` have the same exact-attention error as the standard ones; the K = V and pool
  steps match their references; the confirmation path gives 5 repeats for every Triton row.

## Amendments
None yet.
