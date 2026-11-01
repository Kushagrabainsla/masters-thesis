# E-52: fixed compact K=V decode attention kernels (FP16 compute, two overlapping views) against their standard twin and the stock vLLM kernel, on 1 x H100 (preregistration)

Status: PREDECLARED 2026-11-01. AMENDED A1 2026-11-01 (R-71 pre-run review, verdict AMEND_THEN_RUN, before any data).
S5 PASS (RTX 4090), AMENDED A2 (diagnostic only). RUN 2026-11-01 on the H100: frozen decision INVALID (V4); the
exploratory label is KILL (byte route only); post-run review R-72: CONFIRM_WITH_CORRECTIONS (Section 13).
Idea: I-182. Journal entry: Section 6.1 E-52. Follows E-51 (frozen decision INVALID; exploratory readout) and its
post-run review R-70 (CONFIRM_WITH_CORRECTIONS). Request: D-85 (user, 2026-11-01, "1 go ahead": fix the compact
kernel with free work first, then one H100 test with the R-70 controls). Spend: D-86 (user, "1"), valid only if R-71
changes neither the plan nor the cap; A1 changes the plan, so the user must confirm it again.
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.
Text marked "(A1)" changed in Amendment A1. The version before A1 is commit 2931aba.

## 1. Question
On 1 x H100, with an FP8 KV cache, does the compact K=V layout (640 of 1,024 values per head and token), read by the
standard Triton decode kernel through two overlapping views and with FP16 compute, cut the context-growing attention
time of a Gemma 4 31B global layer by at least the kill line against the same kernel and configuration on the
standard layout (15.5% at 32K, 12.5% at 64K), and by the go margin against the stock FA4 FP8 kernel? (A1) The views
kernel saves DRAM bytes only. So "no" (KILL, byte route only) is: sigma_layout is below the kill lines in both groups
and the time does not follow the bytes (K = V elasticity e_h below 0.413 at 32K and 0.333 at 64K), with a K = V oracle
that works on this GPU. Every other low result is INCONCLUSIVE with a named cause.

## 2. Rubber duck
- E-51 (FACT, README Section 13 there, R-70): the compact Triton FP8 kernel was 29% slower than its twin. Its BF16
  version was 34% to 37% faster than its twin, so the defect is specific to FP8. The Triton standard FP8 kernel read at
  0.77 x copy on dt, and halving its DRAM bytes (K = V) cut dt by only 9% to 10% (e_triton 0.18 to 0.20; 0.9 to 1.0 on
  the RTX 4090). So the E-51 kernels were bound on chip on the H100.
- New FACT (Triton 3.8.0, `third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/ElementwiseOpToLLVM.cpp`): FP8 E4M3 to FP16 is
  one `cvt.rn.f16x2.e4m3x2` per 2 values. FP8 E4M3 to BF16 on sm_90 is that instruction plus two scalar
  `cvt.bf16.f16` (R-71 confirmed it). The E-51 kernels cast to BF16. FACT (FA4 source at b273a02, R-70): the stock FA4
  FP8 path computes in FP16 ("the single-instruction fp8->fp16" path).
- FACT (FlashMLA Hopper FP8 note, 2025-09-29): on an H800 (also SM90) an FP8 decode kernel was bound by the
  dequantization, at about 50 clock cycles per 512 values against 34 for the MMA.
- HYPOTHESIS H4 (main session): the BF16 cast sets most of the E-51 Triton FP8 time. (A1, R-71) The rate of
  `cvt.bf16.f16` on SM90 has no primary source: NVIDIA's throughput table has no row for it. If it is 16 per clock and
  SM, the BF16 path costs about 64 clocks per token and KV head, near the whole E-51 time (114 clocks; DRAM floor 88);
  if it is 64, it costs about 16 more clocks than the FP16 path, and the cast alone cannot explain E-51. The SASS
  counts of the compile check (Section 4) show which instruction it becomes.
- (A1, R-71 F3) FACT (E-51 rows, recomputed by R-71): the shared memory of every E-51 Triton kernel equals one K tile,
  one V tile, the q tile and the p tile. So at `num_stages` 2 the loads were not double-buffered. HYPOTHESIS H5: the
  E-51 kernels exposed the memory latency, and `num_stages` 3 (in the wide sweep) is a second lever. The cast x stage
  table (Section 4) separates H4 from H5.
- New layout (the views). E-51's fused kernel loads V once and uses it in two dot roles; R-70 (INTERPRETATION) names
  this as the most likely FP8 defect. The views layout stores each row as [K_rot (128) | V_unrot (384) | V_rot (128)].
  Its first 512 values are the K of the standard kernel ([K_rot | V_unrot], with the k_norm weight w moved into the
  query) and its last 512 values are the V ([V_unrot | V_rot]). So the unchanged standard kernel reads the compact
  cache, with one load per dot role, and the 384 shared values come from L1 or L2 on the second read. The compact
  kernel then has the same instructions as its twin, and only the DRAM bytes differ (640 against 1,024 per token). The
  output comes in the V-view order; an engine folds that order into the o_proj columns (exact, no run-time cost).
  FACT (R-71): the layout is exact (an independent check agrees to 1e-15 in float64).
- The stock FA4 kernel reads the same views ("_views" rows, reported). FACT (R-71): the vLLM wrapper checks only
  `stride(-1) == 1`, FA4 keeps aligned strided views as they are, and K and V use separate TMA atoms. INTERPRETATION:
  its gain is about 0.375 x e_stock, so about 7.5% of dt, because the stock kernel is bound on chip (E-51).
- (A1, R-71 B1) What the views test can and cannot show. FACT (algebra): the views kernel runs the twin's code, so its
  rate on 640 bytes is 0.625 x the twin's rate / (1 - sigma_layout). The E-51 KILL test (the compact kernel at 0.6 x
  copy) would therefore fire for a twin at 0.96 x copy whose views do not share their reads: a false KILL of the
  layout. A1 removes it. A low sigma_layout has three possible causes: (a) the time does not follow the bytes, (b) the
  views do not share their DRAM reads, (c) the L2 path or the on-chip work caps the views kernel. Only (a) closes the
  byte route, and only for the Triton FP16 family: a kernel that stages V once (R-70 H3) is not tested by E-52.
- Expected (HYPOTHESIS): H4 or H5 holds; the Triton standard FP16 kernel reads at 0.85 x copy or more on dt;
  sigma_layout 0.25 to 0.37; GO.
- Not in E-52 (D-85): R-70's option (b), the FP8 MMA kernel of FA PR 199. sigma_mma keeps the 15% allowance for it.

## 3. Hypotheses
- H1 (primary, the E-51 rule): in at least one group, sigma_mma >= 0.25 and sigma_layout >= the kill line.
- H4 (mechanism, reported): with FP16 compute, the Triton standard FP8 kernel reads at 0.85 x copy or more on dt, and
  its K = V elasticity e_h is 0.6 or more.
- H5 (A1, mechanism, reported): `num_stages` 3 gives a gain of its own (the cast x stage table).
- H0 (A1): sigma_layout is below the kill lines; KILL (byte route only) if the time does not follow the bytes,
  otherwise INCONCLUSIVE with its cause.

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
  - pool rows (timing only; R-70 probe): each block table entry maps into a pool of 8 MB of pages, so after the first
    call the bytes come from L2. In the confirmation: `stock_fp8_s32_pool`, `triton_std_fp8h_pool`,
    `triton_cmpv_fp8h_pool`; (A1, R-71 F9) in the sweep: `triton_std_bf16_pool` (the calibration).
  - (A1, R-71 F3) cast x stage rows (confirmation, reported): `triton_std_fp8h` and `triton_std_fp8` (BF16 compute) at
    the selected splits, warps and tile with `num_stages` 2 and 3.
- Timing: each configuration is captured as 20 calls in one CUDA graph and replayed 30 times; time per call = replay
  time / 20 (median). Spread (R-70) = max / min - 1 of the central 20 of the 30 replays. NVML SM clock and clock event
  reasons after every 10th replay, (A1, R-71 F7) read while the replay runs. The first eager call of each Triton
  configuration records its registers, spills and shared memory; at 19:32K, the PTX, TTGIR and (A1) cubin of each
  Triton kernel at 32/4/32/2 are saved, with PTX counts that include predicated lines (A1, R-71 F6).
- (A1, R-71 F2) Compile check, before the benchmark, in its own process (`compile_check_e52.py`): every Triton kernel
  and configuration is compiled for sm_90 (`_split_kernel.warmup` with the target set to sm_90; nothing launches). On
  the RTX 4090 it is a cross-compile; on the H100 it fills the Triton cache. An entry that crashes the compiler is
  marked "crashed" by the next run and skipped by the benchmark. Per entry: status, seconds, shared memory, PTX counts
  and SHA, and the cubin. After the benchmark, `sass_counts_e52.py` reads registers, spills and SASS opcodes from the
  cubins with `cuobjdump` (if the image has it).
- (A1, R-71 F5) DRAM probe, after the benchmark (diagnostic): `ncu --metrics dram__bytes_read.sum` on one call each of
  the standard, K = V and views kernels and the stock kernel on both layouts at 19:32K (`dram_probe_e52.py`). If the
  Pod refuses the counters (ERR_NVGPUCTRPERM), the log says so.
- Order (as E-51 A1): the compile check, the correctness cell, the four group cells (19:2K, 19:32K, 11:2K, 11:64K),
  the confirmation phase (then `run.json` has "ok_groups"), the reported cells 19:8K, 11:8K, 4:2K, 4:32K, then the DRAM
  probe and the SASS counts. At 4:2K the standard FP8 cache (33.6 MB) fits in the L2 of both GPUs. (A1, R-71 F4) If
  more than 720 s of the benchmark have passed before a group cell starts, that cell runs the 4 reported-cell
  configurations only, so the confirmation still runs inside `KB_TIMEOUT`; `run.json` lists those cells and the time
  of each phase.
- Confirmation (as E-51 A1 and A2): after one untimed pass, each decision row and each K = V, pool and cast x stage row
  is captured in a new graph and timed 5 times; odd repeats run in the reverse order.
- Correctness cell (as E-51, seed 52): B 2, L 3,000, Gemma 4 K=V data stored in FP8 and BF16; every configuration of
  every Triton kernel and the stock kernels, each against the float32 reference of its own stored values (the views
  kernels against the views reference), and against the exact attention (the views outputs after `unpermute`). The
  output is zeroed before each run. (A1, R-71 O5) The unused slots of the last page hold +-448 (the E4M3 maximum, as
  an engine holds stale values there), so a kernel that does not mask them fails. They are finite, so a kernel that
  masks the scores but multiplies P = 0 by a stale value stays correct.
- Fail fast (as E-51 A1): if the stock FP8 kernel with 32 splits or a primary Triton family fails the correctness
  cell, the run stops (`--fail-fast auto`: on SM90). (A1, R-71 O6) An illegal memory access or a device-side assert
  stops the run at once (every later row would fail).
- Not timed: the query preparation (an elementwise op on B x 32 x 512 values), the KV write, the o_proj order.

## 5. Metrics
- Primary (FP8), as E-51 (A1) with the primary families above: for each group X, dt(X) = t(X) - t(2K) at the group
  batch. Selection: the stock and standard families each take the fastest configuration at X among the configurations
  that pass the correctness cell and ran at 2K and at X. (A1, R-71 F1) The views family runs at the configuration that
  the standard family selected, so sigma_layout compares one compiled kernel at one configuration on the two layouts.
  dt = the median over the 5 repeats of the per-repeat differences.
  - sigma_mma = 1 - dt_compact / (0.85 x dt_stock).
  - sigma_layout = 1 - dt_compact / dt_standard.
  - (A1) e_h = 2 x (1 - dt_alias / dt_standard), the K = V elasticity of the selected standard configuration (in each
    repeat too), and e_bf16_triton, the same for the BF16 standard kernel (sweep): they enter the KILL condition.
- Secondary (reported only): sigma_dq = 1 - dt_compact / dt_stock; the sigmas of each repeat and their range; the dt
  rate of each family; (A1) sigma_layout with the views kernel's own selection; sigma_stock_layout = 1 - dt(stock s32
  views) / dt(stock s32); e_stock; e_bf16_stock; the pool shares 1 - dt_pool / dt (FP8 and BF16); (A1) the cast x stage
  table; the cast x layout factorial at 32/4/32/2 (sweep values); BF16 (one sweep measurement, with the fused BF16
  kernel); the B 4 times; the linearity at 8K; the views over standard time at 4:2K and (A1, R-71 F1) at 11:2K against
  19:2K (flagged if 3% lower: the 11:2K views rows, 57.7 MB, can sit in the 50 MB L2 and make dt larger); the PTX
  counts and (A1) whether the standard and views PTX are the same; the compile check, the SASS counts and the DRAM
  probe; the NVML clocks of the decision rows; the share of busy telemetry samples under the software power cap (0x4).
- Minimum effect that matters: the kill line (S = 5%) and the go line (S about 8%), as E-51 (R-68).

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU is H100 80GB.
- V2 (as E-51 A1): the stock FP8 kernel with 32 splits is within 0.02 of the reference of its stored values, and each
  primary Triton family has at least one configuration within 0.01. Before the selection, the analysis drops each
  Triton configuration above 0.01 (or without a result) and each stock kernel above 0.02; a K = V or pool row follows
  its base kernel.
- V3 (R-70 fix): the stock FP8 kernel with 32 splits is within 15% of the E-50 in-engine time per call at 19:32K
  (1,358.1 us) and 11:64K (1,507.2 us), and its dt is within 15% of the in-engine dt (1,119.8 and 1,364.0 us). The 2K
  ratios are reported without a bound (E-51: 0.837 and 0.863).
- V4 (A1, R-71 B2): for each Triton decision row (the standard selection and the views kernel at it, at 2K and X), the
  median over the 5 repeats of the central spread is at most 2%; the range of the 5 per-repeat dt values of each
  Triton family is at most 3% of their median; and that range of the stock family is at most 5%. The stock spreads
  are reported (E-51: the stock rows had spreads of 4.2% and 2.7% and dt ranges of 2.7% and 0.6%; the Triton rows had
  dt ranges of 1.0% at most).
- V5 (as E-51 A2): the copy bandwidth exists, and no decision row at X and no dt rate of a primary family reads above
  1.15 x copy.
- V6: hard throttle (0x8, 0x20, 0x40, 0x80) in at most 2% of busy telemetry samples. The software power cap (0x4) is
  reported, not bounded (R-70: E-51 had 14.6% of busy samples under it).

## 7. Decision rule (frozen; A1 before any data)
Deciding groups: X = 32,768 (B 19) and 65,536 (B 11), FP8. Kill lines (R-68): 0.155 at 32K and 0.125 at 64K. The
e-lines (A1): e_h 0.413 at 32K and 0.333 at 64K (the e at which a bytes-only gain 0.375 x e reaches the kill line).
- GO: in at least one group, sigma_mma >= 0.25 and sigma_layout >= the kill line of that group. A GO allows the next
  steps only: the accuracy gate (R-68 recommendation 7) and an engine prototype plan (the views layout, the query and
  o_proj order, the scales, the page-size plan).
- KILL, byte route only (A1, R-71 B1): sigma_layout below the kill line in both groups, e_h below the e-line in both
  groups, and the K = V oracle valid on this GPU (e_bf16_triton >= 0.5 in both groups). Consequence: the byte route is
  closed for the Triton FP16 family; a kernel that stages V once is not tested; I-182 keeps its capacity result.
- INCONCLUSIVE: the benchmark did not finish (a status other than "ok" or "ok_groups"), a group or its confirmation
  is missing, or the Triton standard kernel reads below 0.5 x copy on dt in a group (the layout test is weak). (A1) Also
  when sigma_layout is below the kill lines in both groups and KILL does not hold. The result then names its flags and
  their consequences: (a) without a valid oracle: "time does not follow bytes, but the K = V oracle is not valid on
  this GPU"; (b) e_h >= 0.6 and sigma_layout < 0.25 x e_h: "the views do not share their DRAM reads: the next kernel
  must read each 640-value row once"; (c) the views pool share below 0.3: "the L2 path or the on-chip work caps the
  views kernel: the next step is a profile"; no flag: "the views kernel gains less than its bytes predict".
- PIVOT: otherwise, with the reason "kernel quality" (sigma_layout >= 0.25 in a group) or "below the bar".
- Replication (as E-51 A1): the rule runs on the medians and on each of the 5 repeats alone (e_h per repeat). The
  decision stands only if at least 4 of the 5 repeats give the same word. Otherwise INCONCLUSIVE ("the decision does
  not replicate").
- Baseline check (A1, R-71 F8): the rule runs again with the E-51 stock dt (1,147.8 and 1,344.9 us) in place of this
  session's stock dt. If the first word differs, the result is INCONCLUSIVE ("baseline off"). Its ratio is reported.
- INVALID: V1 to V6 fail, no copy bandwidth, or the fail-fast stop.

## 8. Controls and baselines
- Baseline: the stock FA4 FP8 kernel (the selected split setting), tied to the engine by V3 and (A1) to E-51 by the
  baseline check.
- Layout control: `triton_std_fp8h`, the same compiled kernel as `triton_cmpv_fp8h` at the same configuration; only
  the addresses differ.
- A/A: the 5 confirmation repeats of each row.
- Positive controls: V5; the BF16 K = V rows (R-70); the pool rows. (A1, R-71 F9) Expected ranges, written before the
  run: e_bf16 0.5 to 1.0 (the second read still moves through the L2-to-SM path, so it can read below 1 while DRAM
  bytes halve); the BF16 pool share 0.5 to 1.0 (the fp32 partials, 40 MB at B 19, also pass through L2).
- Negative control: the views and standard kernels give the same attention within the error of the dtype (the
  correctness cell, after `unpermute`), and the poisoned tail slots change nothing (A1).
- Mechanism controls (reported): the cast x layout factorial at 32/4/32/2 and (A1) the cast x stage table at the
  selected configuration separate FP16 compute, the stage count and the layout, and tie E-52 to the E-51 kernels.

## 9. Budget
- H100 SXM Secure at the live price ($3.99/h, read 2026-11-01, stock HIGH). Image pull 5 to 11 min (E-51: about 11).
- Time model (A1, R-71 F4): 628 sweep rows, about 40 confirmation entries plus the cast x stage rows, about 65 split
  kernel compiles (the views kernel shares the standard binary). R-71 estimates 10 to 23 min for the benchmark; the
  compile check moves the compiles into its own process first. The deadline (720 s) keeps the confirmation inside
  `KB_TIMEOUT`. Pod time about 25 to 35 min, about $1.70 to $2.40. Cost cap $3.
- Caps (as E-51): `KB_TIMEOUT` 1,500 s; (A1) `CC_TIMEOUT` 300 s per compile-check run (3 runs at most); the ncu probe
  240 s; session watchdog 1,980 s; boot watchdog CAP 2,040 s after the container starts; `AUTO_STOP=1` with a 180 s
  delay. With an 11 min pull, the worst case is 45 min of Pod time, $3.00 at $3.99/h. (A1, R-71 F10) The boot watchdog
  does not cover the image pull, so the main session deletes the Pod if the pull takes more than 15 min.
- Rehearsal (S5): RTX 4090 ($0.89/h, read 2026-11-01, stock HIGH), the group cells and 4:2K, `--fail-fast off`,
  `KB_TIMEOUT` 900, boot watchdog CAP 1,800 s, `AUTO_STOP=1`. About 25 min, cap $0.60. VERIFY items: the sm_90 compile
  check runs (cross-compile) with no crash, and shows `cvt.rn.f16x2.e4m3x2` and no `cvt.bf16.f16` in the FP16 kernels
  and the same PTX for the standard and views calls; every Triton layout and compute dtype compiles for sm_89 and
  passes the correctness cell with the poisoned tail; the NVML clock is read; at 4:2K the views time is within 5% of
  the standard time (the L-18 gate: an L2-resident shape, where DRAM does not hide an on-chip difference); the time
  model of the group cells. FACT (R-71): on the 4090 (99 KB of shared memory) up to 18 configurations per family fail
  at stage 3 with tile 32 or 64; they are recorded and skipped there, and the 4090 cannot test them.
- Balance: (A1) D-86 sets a one-time floor of $12 for E-52 (the live check failed the $35 floor); S4 checks it again.

## 10. Threats to validity
- The views share their DRAM reads only if the second read finds the lines in L1, in L2 or in flight (HYPOTHESIS;
  R-71 puts it at 90% or more). (A1) The flags (b) and (c), the pool rows, the stock views rows and the DRAM probe
  report it; the rule no longer gives KILL for a views kernel that does not share.
- FP16 range: the query, P and the cache values must fit FP16. FA4 makes the same choice for this path. FACT (R-71):
  in an emulation with logit scales 1 to 30 and k_norm spreads 0.1 to 3, the FP16 error was 0.0004 to 0.0037 (BF16 up
  to 0.045). Real activations are not tested here (the accuracy gate after a GO).
- The wide sweep has 54 configurations for the standard family: one configuration is selected and then confirmed in 5
  repeats, as E-51 (L-15). (A1) The views kernel uses that configuration, so no second selection enters sigma_layout.
  (A1, R-71 O4) The Triton family chooses from 54 configurations and the stock family from 2: a GO means "a tuned
  Triton kernel on the views layout beats the 2-setting stock kernel", not "the layout beats the engine kernel".
- The stock FA4 kernel may still reject the views strides at run time; then the "_views" rows fail and are reported.
- The FP8 MMA stock kernel is not in the image; sigma_mma keeps the 15% allowance (R-68, R-69).
- Power-cap noise (L-17): the central spread, the dt range bounds and the NVML clocks.
- No CUDA graph padding (as E-51): the fixed cost cancels in dt; V3 no longer bounds the 2K ratios.
- (A1, R-71 O2) The o_proj fold is exact for BF16 weights. Block-scaled FP8 or NVFP4 o_proj weights keep one scale per
  128 columns, and the V-view order crosses those blocks, so an engine needs a run-time permute or a re-quantization.
- (A1, R-71 O1) Two scales: an engine that stores K_rot with its own scale s_k and V with s_v passes s_v as both
  descales and multiplies the rotated query part by s_k / s_v (tested on the CPU).
- The query preparation and the o_proj order are not timed.
- One GPU, one session, decode only, uniform contexts, random data.

## 11. Files
- `kbench_e52.py`: the benchmark (cells, sweeps, correctness, confirmation, NVML clocks, PTX dumps, deadline,
  compile-check skip, sticky stop, stub mode).
- `analyze_e52.py`: the frozen analysis and rule (summary.json and the decision).
- `session_e52.sh`: the Pod session (records, telemetry, copy bandwidth, the compile check, the benchmark, the DRAM
  probe, the SASS counts, the analysis, watchdogs).
- (A1) `compile_check_e52.py`, `dram_probe_e52.py`, `sass_counts_e52.py`: the diagnostics of R-71 F2 and F5.
- `cpu_dryrun_e52.py`: the GPU code paths on the CPU emulation of Triton.
- `test_e52.py`: the tests.
- `../gemma4_kv_kernel/compact_attn.py` (the kernels: `cast`, the views helpers, (A1) `split_args` for the compile
  check; the E-51 default is unchanged) and `tl_shim.py` (FP16 added).
- Upload folders: gemma4_kv_views, gemma4_kv_kernel, runpod, fp8_quant_fusion, nvfp4_dispatch, safedispatch_testA,
  safedispatch_testA2, ledger.

## 12. Verification record (stage S3 and S5)
- S3 (2026-11-01, Mac, no GPU and no Triton): `uv run --no-project --with numpy --with torch --with pytest python -m
  pytest test_e52.py`: 19 passed. `test_e51.py`: 16 passed (the E-51 kernels and rule are unchanged). The CPU dry run:
  every Triton kernel (standard, views and fused layouts; BF16 and FP16 compute; FP8 and BF16 cache) at B 1 to 3,
  partial pages and empty splits is within 0.0021 of the reference of its stored values (FP16 compute 0.0017 at most);
  the views outputs after `unpermute` have the same exact-attention error as the standard ones; the K = V and pool
  steps match their references; the confirmation path gives 5 repeats for every Triton row.
- S3 after A1 (2026-11-01, Mac): `test_e52.py` 26 passed (new: the two-scale rule, predicated PTX lines, the flags,
  views that do not share give INCONCLUSIVE, the byte-route KILL and its oracle condition, the baseline check, V4 on the
  Triton rows and the stock dt range, the 11:2K flag, the compile-check skip, the sticky stop, the deadline); the CPU
  dry run passes with the poisoned tail slots; `test_e51.py` 16 passed after the `split_args` refactor. Not tested on
  the Mac: `compile_check_e52.py`, `dram_probe_e52.py` and `sass_counts_e52.py` need Triton, a GPU or the CUDA tools;
  the rehearsal runs them (Section 9).
- S5 (2026-11-01, RTX 4090, Pod 90d0y7vy24vxgp, EU-RO-1, $0.89/h; created 09:28:36 UTC, stopped by AUTO_STOP, deleted
  about 10:02 UTC; billing pending): PASS. The image was cached on the host (setup 8 s). FACT:
  1. Compile check: 124 of 124 entries compiled for sm_90 (a cross-compile from sm_89) in 75 s; no crash. The FP16
     kernels hold 128 `cvt.rn.f16x2.e4m3x2` and no `cvt.bf16.f16`; the BF16 kernels hold 128 and 256. The standard
     and views PTX are identical in 54 of 54 configurations. Every kernel has `cp.async`. Shared memory: 50,176 B at
     `num_stages` 2 and 50,180 B at 3 (stage 3 adds no K or V buffer).
  2. Correctness (sm_89, poisoned tail slots): every Triton configuration passes (FP16 compute 0.00166 at most, BF16
     0.00185); the views outputs have the standard exact-attention error (0.0233 FP8, 0.0029 BF16).
  3. NVML: the decision rows read 2,745 MHz or more during the replays. Telemetry: 5% of busy samples under the cap.
  4. L-18 gate: at 4:2K the views time is 1.013 x the standard time (bound 1.05).
  5. Time: correctness 64 s, the four group cells 544 s, the confirmation 179 s; `kbench` 787 s against 900 s; the
     session 877 s. No cell needed the deadline.
  6. The 4090 is DRAM-bound for every Triton FP8 kernel (0.98 to 1.02 x copy on dt), so every compact variant gives
     sigma_layout 0.35 to 0.36 there (the factorial), and the 11:2K flag is set (its 72 MB L2 holds the 57.7 MB views
     rows). As planned, the 4090 says nothing about the H100 speed (L-18).
  7. The decision INVALID (V1, V2, V3) is the expected one: the 4090 has no SM90 stock kernel.
  8. The SASS opcode counts were all 0. (Corrected by R-72, C6: the cause is that `nvdisasm` is missing in the
     image; the saved SASS text holds the line "Could not find executable file 'nvdisasm'". It was not a parser
     fault, as A2 first said. No SASS counts exist for E-52; the cubins are saved for a later read.)

## 13. Result (S6 to S9, 2026-11-01 UTC; post-run review R-72: CONFIRM_WITH_CORRECTIONS; corrections C1 to C14)
- Run (FACT): Pod tj0ad1ljwhqy6i, H100 80GB HBM3 (sm_90), EUR-NO-2, Secure, $3.99/h, created 10:04:08 UTC; SSH after
  211 s. Session 10:09:36 to 10:20:17: compile check 124 of 124 in 83 s; `kbench` 553 s (no cell needed the deadline).
  Results copied (315 files, checksums match); the Pod was deleted in the AUTO_STOP delay (about 10:23 UTC). About
  $1.25 (estimate; billing pending); E-52 about $1.56 with the rehearsal. Design commit 3130fcd (A2). Copy 3,025 GB/s.
  Results: `results/e52-h100-20261101/`.
- **Frozen decision: INVALID (V4).** The run does not decide.
  - V4: one Triton decision row failed. The standard FP16 kernel at 19:2K (16/4/32/2) had central spreads of 5.5%,
    0.2%, 5.8%, 0.15% and 5.2% (median 5.2%; bound 2%). FACT (NVML): those three repeats ran at 1,515, 1,605 and
    1,715 MHz; the two clean ones at 1,980. Every other Triton decision row had a median spread of 1.4% or less. The
    dt ranges were 0.14% to 0.63% for the Triton families (bound 3%) and 1.9% for the stock family (bound 5%).
    Telemetry: 18.9% of busy samples under the software power cap (mean 1,833 MHz, lowest 1,320 MHz). FACT (R-72):
    the three slow repeats are the three forward-order repeats, in which this row follows the stock row at 19:32K
    (at the 700 W cap). The failure moves dt by 0.09% at most and sigma_layout by 0.001.
  - V1, V2, V3, V5 and V6 pass. V3: 1.025 and 1.003 at X; dt 1.066 and 1.020 (2K ratios 0.836 and 0.845, reported).
    Every configuration of every kernel passes the correctness cell with the poisoned tail (FP16 compute 0.00166; the
    stock kernel on the views 0.00166, the same as on the standard layout).
- Exploratory readout (not a decision; checked by R-72, which recomputed every number with its own script). The
  numbers are FACT; the meaning is INTERPRETATION. H1 is rejected by a large margin: the expectation of Section 2
  (sigma_layout 0.25 to 0.37, GO) met 0.04 and 0.07 (R-72 C14).
  - With V4 set aside, the A1 rule gives KILL (byte route only) in 5 of 5 repeats, and the baseline check gives the
    same word. sigma_layout is 0.042 (32K) and 0.065 (64K) against the kill lines 0.155 and 0.125. e_h is 0.216 and
    0.240 (0.20 to 0.27 per repeat) against the e-lines 0.413 and 0.333. The oracle is valid (e_bf16_triton 0.856 and
    0.817). Flags (a) and (c) in both groups. The tightest margin is e_h at 64K (highest repeat 0.272 against
    0.333). Configuration sensitivity (R-72 C11): sigma_layout is 0.042 to 0.101 at the six best 32K configurations,
    and 0.087 with the views kernel's own selection; all are below the lines.
  - The pool rows (every byte from L2) cut dt by only 17% to 20% for the standard FP16 kernel, 14% to 15% for the
    views kernel and 20% to 22% for the stock kernel. The standard FP16 kernel reads at 0.87 to 0.90 x copy on dt.
    INTERPRETATION (corrected by R-72, C1): about 80% of dt does not depend on the DRAM bytes (the pool floor is 80%
    to 83% of dt), and the DRAM time at the copy rate is also 87% to 90% of dt. The kernel is co-limited: the two
    floors overlap, so 37.5% fewer bytes give only 4% to 10%. The cause of the floor is open. FACT (R-72): the data
    argue against an L2-to-SM bandwidth limit (the BF16 pool rows carry 5.7 to 5.9 TB/s) and rule out the merge
    kernel, the split count and the tensor-core work (about 10% of peak); they do not separate load latency (both
    kernel families stage K and V once), occupancy (255 registers, 2 CTAs per SM) and instruction issue.
  - H4 is false as preregistered (corrected by R-72, C2): the rate half holds (0.87 and 0.90 x copy), but e_h is
    0.22 and 0.24 against 0.6. At the same configuration, FP16 compute cuts dt by 14% (32K) and 13% (64K) against
    BF16 compute (910 against 1,061 us; 1,053 against 1,205 us): 14 to 17 SM clocks per token and KV head, 12% to 14%
    of dt, not most of it. This fits the second branch of Section 2 (a conversion rate of 64 per clock and SM). The
    BF16-compute kernel was timed at the FP16 optimum only. H5 is not tested (C3): `num_stages` 3 adds no K or V
    buffer (shared memory +4 B in 54 of 54 configurations) and is 2% to 3% slower, so the latency question stays
    open.
  - The standard Triton FP16 kernel is 19% to 25% faster on dt than the stock FA4 FP8 kernel, by baseline (C4):
    23.7% and 25.1% against this session's stock rows (std_vs_stock 0.763 and 0.749; those rows ran 7% to 12% below
    the Triton SM clock), 20.7% and 21.7% against the E-51 stock dt, 18.7% and 22.8% against the E-50 in-engine dt.
    21 of 54 (32K) and 15 of 54 (64K) Triton configurations beat the best stock row. With a BF16 cache the two tie
    (0.992 and 0.999). The views kernel is 27% and 30% faster than this session's stock rows (sigma_dq 0.270 and
    0.300). sigma_mma is 0.141 and 0.176 (go line 0.25).
  - The stock FA4 kernel reads the views layout with no code change and with the same error. It is 9.6% (32K) and
    6.6% (64K) faster on dt with it (sigma_stock_layout; 32 splits on both layouts; positive in 10 of 10 repeats;
    7.6% against the selected 64K stock row); at B 4 and 32K it takes 273.7 against 290.3 us. Caveats (R-72, C13):
    the views rows ran 5.7% higher in SM clock (1,831 against 1,732 MHz; 1,845 against 1,742), so up to about 1.6
    points can be clock; their central spreads were 3.5% to 5.5%. The stock gain matches its oracle's bytes-only
    prediction (0.375 x e_stock = 0.084 and 0.069); the Triton views kernel gets only 52% and 73% of its own.
    With a BF16 cache the stock kernel gains 24.0% and 24.5% on dt with the views (single sweep rows).
  - BF16 cache: sigma_layout 0.327 and 0.296 (the layout works where the kernel is bound by bytes); e_bf16_stock 0.59
    and 0.58; BF16 pool share 0.48 and 0.46, below the preregistered range 0.5 to 1.0 (C5; the BF16 kernel has its
    own floor of 814 us; its pool rows carry 5.7 to 5.9 TB/s, so their bytes do come from L2).
  - The E-51 fused kernel stays 28% (BF16 compute) and 23% to 25% (FP16) slower than its twin (the factorial).
  - The selected configurations are 16/4/32/2 (32K) and 64/4/32/2 (64K): 255 registers, no spills. The PTX of the
    standard and views calls is identical. ncu is not in the image (no DRAM counter), and nvdisasm is not either (no
    SASS; C6).
- What this means (INTERPRETATION, corrected by R-72): on one H100, about 80% of the context-growing FP8 decode
  attention time of the Gemma 4 global layers does not depend on the DRAM bytes, also after the cast fix, so a layout
  that removes bytes gives only about 4% to 10% of it (about 3% of the step time near the pool limit), with the stock
  kernel or with Triton. The memory saving of I-182 (37.5% of the global KV cache) is algebraically exact, and in this
  microbenchmark the unchanged stock FA4 FP8 kernel reads the compact rows through two views at the same error and is
  7% to 10% faster. Not shown: the engine path (the cache write, the query preparation, the o_proj fold, the page
  size), the Triton backend of vLLM, accuracy on real activations. The speed claim is closed for the byte route on
  these kernel classes. A kernel with less on-chip work per token (FP8 MMA, one that casts V once, or one with a true
  two-tile prefetch) is not tested.
- Lesson (L-19, corrected by R-72, C7): the power cap slowed a short row that follows a heavy stock row in its 3
  forward-order repeats. The row's spread voids V4, but it barely moves dt (the Triton dt ranges were 0.6% at most).
  E-51 was INVALID for other reasons (V3 at a 2K time ratio, and V4 on two stock rows at X).
- R-72 (S9, 2026-11-01, Sonnet): CONFIRM_WITH_CORRECTIONS (C1 to C14, applied above). No change to the frozen rule or
  to the word is needed. Write-up: `dossier/REVIEW_E-52_POSTRUN_2026-11-01.md`. It recommends for D-88: the write-up
  now, the engine prototype as a gated build with a hard stop at week 6, no FP8 MMA test. FACT (R-72, from a local
  vLLM clone; the main session checked the code at commit e070573, `vllm/v1/core/kv_cache_utils.py`): vLLM pads a
  non-MLA attention page that does not divide the largest page; the compact FP8 page (163,840 B for 64 tokens)
  against the sliding page (524,288 B) has the ratio 3.2, so it would be padded: the first risk of a prototype.
- Next: the user decides the next step (D-88).

## Amendments
- A1 (2026-11-01, before any data; from R-71, verdict AMEND_THEN_RUN; `dossier/REVIEW_E-52_PRERUN_2026-11-01.md`):
  - B1 (blocker): the KILL by a competence test of the views kernel is removed (Section 7). KILL is the byte route only
    (R-71 fix 4, the R-70 scope), with the oracle condition added by the main session (e_bf16_triton >= 0.5), and every
    other low result is INCONCLUSIVE with the flags (a), (b), (c) and their consequences (Sections 1, 2, 7).
  - B2 (blocker): V4 bounds the Triton decision rows and the stock dt range at 5%; the stock spreads are reported
    (Section 6). Not adopted: 7 stock repeats (R-71 fix 3), because the replication rule pairs one stock, one standard
    and one views repeat.
  - F1: the views kernel at the standard selection; its own selection reported; the 11:2K flag (Section 5).
  - F2: the sm_90 compile check in its own process, with the SASS counts (Sections 4 and 9). It runs on the Pods: the
    Mac has no Linux host for Triton (Docker Desktop is installed but is not open source and was not started).
  - F3: the cast x stage rows; H5 (Sections 2, 3, 4).
  - F4: the 720 s deadline and the phase times (Sections 4 and 9). Not adopted: dropping warps 2.
  - F5: the ncu DRAM probe (Section 4). F6: predicated PTX lines. F7: NVML read while the replay runs.
  - F8: the baseline check against the E-51 stock dt (Section 7). F9: the expected ranges and the BF16 pool row
    (Section 8). F10: the $12 floor (D-86) and the 15 min pull rule (Section 9).
  - O1: the two-scale test. O2 and O4: Section 10. O5: the poisoned tail slots, with a finite value (448) instead of
    NaN, because a kernel that masks the scores still multiplies P = 0 by the stale value, and 0 x NaN is NaN. O6: the
    sticky-error stop. Not adopted: O3 (an early H4 gate), because the deadline already caps the time.
  - The H4 text no longer states a conversion rate as a fact (Section 2).
  - Unchanged: the question's kill and go lines, the groups and cells, the primary metrics, the 15% FP8 MMA allowance,
    the stock call, the caps and the cost cap.
- A2 (2026-11-01, from the S5 rehearsal, before the H100 session uploaded its code; diagnostic only; its diagnosis was
  wrong, see Section 12 item 8 and R-72 C6): the SASS parser of
  `sass_counts_e52.py` counted 0 opcodes. It now keeps the raw SASS text (`sass/`) and matches an opcode as a capital
  word after the address comment and an optional predicate. Unchanged: the rule, the metrics, the benchmark and every
  other script.
