# E-50: Gemma 4 K=V headroom on top of an FP8 KV cache, near the KV pool limit, on 1 x H100 (preregistration)

Status: PREDECLARED 2026-10-30. NOT RUN.
Idea: I-182. Journal entry: Section 6.1 E-50. Follows E-49 and its post-run review R-65.
Request: D-76 (user, 2026-10-30, "Plan next test": fix the test code, then plan one H100 session of about $3 that
reruns the 26B model and measures the 31B model at larger batches and with an 8-bit KV cache).
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
On 1 x H100 with vLLM nightly, when the server already uses an FP8 KV cache, does the duplicated K=V cache of the
Gemma 4 31B global layers still cost at least 5% of decode step time at 32K or 64K context, at a batch near the KV
pool limit? "No" is an upper bound below 3% at both contexts.

## 2. Rubber duck
- E-49 (FACT, confirmed by R-65): with a BF16 KV cache at batch 4, the compact K=V cache could save at most 6.65%
  (32K) and 11.33% (64K) of the decode step. The global attention (FA4, head_dim 512) reads at the copy bandwidth.
  The 26B-A4B 64K cells were lost to a probe fault: `max_tokens` was too small for the staggered chunked prefill.
- R-65 (INTERPRETATION): S grows with the batch (about 10% to 14% at the KV pool limit). An FP8 KV cache halves all
  KV bytes; on top of it the compact cache adds only about 3.8% (32K) and 7.0% (64K) at batch 4. R-65 gate for a
  prototype: the stacked ceiling must stay at or above 5% at the pool-limit batch. This test measures that number.
- New FACT (vLLM source at the run commit 3966924d5, read 2026-10-30): with `kv_cache_dtype` fp8 on SM90,
  `Gemma4Config` uses per-layer FA3/FA4 selection. The sliding layers (head_dim 256) run FA3 with FP8 KV. The
  global layers (head_dim 512) run the FA4 "SM90 FP8-KV-dequant kernel": BF16 Q, FP8 K and V dequantized in the
  kernel, a 64-token page (`fa_utils.py`, `flash_attn.py`, `models/config.py`). So the competitor path exists in
  the E-49 image. SGLang PR #46965 (open) adds a similar path.
- Mechanism of the stack: a compact FP8 row holds 640 one-byte values per head and token instead of 1,024. It cuts
  37.5% of the FP8 global KV bytes and of the dequant work. The compact QK product has 640 dims (+25% QK work).
- The batch: the FP8 cache doubles the tokens that fit. E-49 recorded 16,095 idle blocks and 1,358 and 2,391 blocks
  per request at 32K and 64K (31B, BF16). So 85% of the FP8 pool is about 20 requests at 32K and 11 at 64K.
- Expected outcome (INTERPRETATION, R-65 batch model with the FP8 global term halved): S about 11% (32K, B 20) and
  13% (64K, B 11), so GO if the FP8 kernel stays memory-bound. If the dequant kernel is slow, PIVOT (KERNEL-BOUND).
- What would kill it early: the FP8 path makes the global attention so cheap that S stays below 3% near the pool
  limit. Then a compact cache is a small engine optimization, not a thesis direction.
- Why this order: the deciding arm first (m31f8). Then the 26B rerun (m26, the E-49 gap that the user asked for).
  Then the BF16 arm (m31bf: the same-Pod FP8 saving and the exact-cache ceiling near the BF16 pool limit).

## 3. Hypotheses
- H1 (primary): with the FP8 KV cache at 85% of the KV pool, S_low >= 5% at 32K or at 64K (31B).
- H0: S_up < 3% at both contexts (FP8 KV takes the gain), or the FP8 global attention is not memory-bound.
- H2 (secondary): with the fixed token budget, the 26B-A4B model at 64K and batch 8 has S near 10.5% (the E-49
  fit prediction in R-65).
- H3 (secondary): with the BF16 cache at 85% of its pool, the 31B ceiling is about 10% to 14% (R-65).

## 4. Setup (frozen)
- Hardware: 1 x H100 80GB HBM3 (SXM), Secure Cloud, on-demand, `minCudaVersion` 13.0, US-NE-1 if in stock (E-49).
- Software: the E-49 image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
  (vLLM 0.33.0rc1.dev229+g3966924d5). The E-49 models and revisions: m31 `RedHatAI/gemma-4-31B-it-FP8-dynamic`
  d4ab4f57 and m26 `google/gemma-4-26B-A4B-it` 4d7ae498.
- Engine: as E-49 (in-process, prefix caching off, text-only mode, prefill chunk 2,048, `max_num_seqs` 32,
  `gpu_memory_utilization` 0.92, default backend, compile and CUDA graphs, greedy, `ignore_eos`, `detokenize=False`).
  Changes: `max_model_len` 67,584 (room for the larger token budget at 64K); `kv_cache_dtype` fp8 in the arm m31f8
  (no calibrated scales, so vLLM uses scale 1.0); no waste cells.
- Expected attention path (FACT from the source): m31f8 logs "Using per-layer FA3/FA4 selection for FP8 KV cache
  on SM90". m31bf and m26 log "Using FA4 for all layers" (E-49). Check V11 reads these lines.
- Arms and cells, in this order:
  - m31f8 (deciding): the E-49 slope cells at B = 4 (2K, 8K, 32K, 64K, then 64K r1 and 32K r1). Group 32K: 2K, 8K,
    32K and 32K r1 at batch a32768. Group 64K: 2K, 8K, 32K, 64K and 64K r1 at batch a65536. Profiles at 2K, 32K
    and 64K of B = 4, and at the 2K and X cells of each group.
  - m26 (secondary): the E-49 slope cells at B = 8; profiles at 2K, 32K and 64K.
  - m31bf (secondary): 2K, 8K, 32K and 64K at B = 4; group 32K (2K, 8K, 32K) and group 64K (2K, 8K, 32K, 64K), no
    replicates; the same profiles as m31f8.
- Pair batch (FACT about the code): B = min(32, floor(0.85 x idle free blocks / blocks per request at X)). The blocks
  per request come from the B = 4 cell at X in the same arm (free blocks before and after its prefill). A batch
  between two CUDA graph sizes runs padded to the next size; `run.json` keeps the sizes.
- Measurement per cell (`decode_probe.py`, R-65 C3): `max_tokens` = (B - 1) x ceil(L / (2,048 - B)) + 4 + 32 x 5 +
  8 + 32 (435 for 64K at B = 8). After the prefill: the scheduler check; a request that left is "finished_early";
  a request without the tokens for 4 warm steps, 5 blocks and the profile is "budget". Then 4 warm steps, 3 to 5
  blocks of 32 timed steps, the scheduler check again, and the finish reason of every request that left.
- Data: as E-49 (windows of GSM8K prompt tokens; seed = L x 1,000 + B + 7,919 x replicate).

## 5. Metrics
- Primary: for each group X of m31f8, S(X) = 0.375 x b_low x X / T(X). T(X) is the median step time at X. b_low is
  the one-sided 95% lower bound of the slope of the cell-level line T = c + b L over the cells of the group (the
  E-49 metric at the pool batch): df 2 for the 32K group, df 3 for the 64K group.
- S is the ceiling of the compact cache on top of FP8 KV (the stacked ceiling), not a speed claim. S grows with B,
  so S at 85% of the pool is a lower bound of S at the pool limit.
- Minimum effect that matters: 5% of decode step time.
- Secondary (reported only): S at B = 4 for m31f8 (R-65 estimate 3.8% and 7.0%); S_att = 0.375 x (global attention
  ms at X - at 2K) / (X - 2,048) x X / T(X) from the layer split, flagged if it differs from S by more than 25%;
  the FP8 saving at B = 4 in the same Pod (1 - T_fp8 / T_bf16 at 2K, 8K, 32K and 64K); the m31bf group S (the
  exact-cache ceiling near the BF16 pool limit); m26 S at 32K and 64K with the E-49 cell test; the pool record
  (idle blocks, blocks per request, B and the B_max estimate); finish reasons; the global attention bandwidth per
  group; GSM8K accuracy per arm.
- Repeats: 3 to 5 blocks of 32 steps per cell; replicate cells at X in the deciding groups.

## 6. Validity checks (a failed check of the m31f8 arm makes the run INVALID)
- V1 to V9 as in E-49, on the B = 4 slope cells: V1 H100 80GB; V2 every block advanced 31 to 33 tokens; V3 every
  context median within 5% of the line; V4a; V5 with the FP8 bytes (40,960 B per token and sequence); V6 GSM8K at
  least 50%; V7 at most one cell with a block spread above 2%; V9 the text-only line.
- V8 (session): hard throttle in at most 2% of busy telemetry samples. No copy bandwidth makes the session INVALID.
- V11 (new): the m31f8 log has "kv_cache_dtype=fp8" and "per-layer FA3/FA4 selection for FP8 KV cache".
- G32768 and G65536 (new): each group with a fit passes V2, V3, V4a (if b_low > 0), V5 and V7 on its own cells.
- A cell counts only with status "ok": all B requests stayed in the engine to the end of the blocks.

## 7. Decision rule (frozen)
Deciding groups: X = 32,768 and 65,536 of m31f8. A group passes if S_low(X) >= 0.05 and the group has M1, M2, V4b
and V4c true:
- M1 (memory-bound): the group's global attention bandwidth (FP8 global bytes x B / b) is at least 60% of the
  copy bandwidth.
- M2 (no curvature): the slope over the two longest contexts of the group is within 25% of the fitted slope. If it
  cannot be computed, M2 is unknown and the group does not pass (R-65 C3).
- V4b (attribution): attention kernels explain at least 80% of the device-time growth from 2K to X (the two
  profiles of the group). Unknown if no kernel matches.
- V4c (negative control): the sliding attention at X is within 5% of 2K. The split comes from the launch order
  (E-49); if that fails, from the kernel counts (a kernel launched once per sliding layer in each step is sliding;
  once per global layer is global; at most 5% of the attention time may stay unassigned). Else V4c is unknown.
Outcomes:
- GO: at least one group passes. A GO allows only a prototype plan with its own preregistration: a kernel
  microbenchmark first, an accuracy gate and the page-size plan (R-64 O5, R-65 recommendation 6).
- KILL: both groups have a fit and S_up < 0.03 in both. I-182 stops as a main direction; its outputs stay.
- INCONCLUSIVE: the m31f8 arm did not finish its pair cells, or a group has no fit (lost cells), and no group
  passes (R-65: a lost deciding cell is not a PIVOT).
- PIVOT: otherwise, with each reason: "KERNEL-BOUND (M1)", "curvature (M2)", "attribution unknown", "attribution
  unclear" or "below the 5% bar".
- INVALID: V8 fails, no copy bandwidth, or a failed check of the m31f8 arm.
- The secondary arms (m26, m31bf) never change the decision.
- Why 3% for KILL (heuristic): at a fixed context, S(B) / B falls as B grows. So a value below 3% at 85% of the pool
  stays below 3.6% at the pool limit, below the 5% bar.

## 8. Controls and baselines
- Baseline: the stock vLLM FP8 KV path (m31f8) is the path that a compact cache would stack on. The BF16 arm (m31bf)
  is the same-Pod reference for the FP8 saving.
- A/A: the blocks in each cell, the replicate cells in the deciding groups and the residual checks.
- Positive control: the copy bandwidth must exist and bound the global attention bandwidth (V5, per group).
- Negative control: the sliding attention stays flat from 2K to X (V4c, per group).
- Cross-Pod check (secondary): the m31bf B = 4 cells against the E-49 m31 cells, after the copy bandwidths agree
  within 3% (MEASUREMENT.md Section 4; E-49: 3,043 to 3,046 GB/s).

## 9. Budget
- H100 SXM Secure at the live price (read again at S4). No network volume; the Pod volume disk holds the models.
- Time (INTERPRETATION, from the E-49 record: engine start 277 s (m31) and 340 s (m26); prefill 7,100 to 15,000
  tokens per second): setup and downloads 6 min; m31f8 16 min; m26 8 min; m31bf 9 min; about 40 min of session.
  With the Pod boot (5 min) and the AUTO_STOP wait (3 min), about 48 min.
- Caps: no new work after 50 min (`NEW_RUN_CAP` 3,000 s); m26 and m31bf start only with 10 min left before that;
  session watchdog 60 min; `PROBE_TIMEOUT` 1,500 s per arm; `AUTO_STOP=1` with a 180 s delay (copy the results in
  that time); boot watchdog CAP 3,900 s (65 min). Cost cap $4.50 (65 min at $3.99/h plus disk). Expected about $3.20.
- KV memory (INTERPRETATION): the pair batches fill 85% of the idle pool; the decode growth is below 2% of the
  blocks per request; the prefill chunk 2,048 keeps the activation peak of E-49.
- Rehearsal (S5): RTX 4090, Gemma 4 E2B (the E-49 S5 model), `ONLY=m31f8` with the FP8 KV cache (Triton on SM89),
  about 15 min, boot watchdog CAP 1,800 s, cap $0.60.
- Order: m31f8 first (it decides), m26 second, m31bf last (the time cap cuts it first). Create-to-start in one tool
  call.

## 10. Threats to validity
- The FP8 scales are not calibrated (scale 1.0). Accuracy can be lower than in a calibrated server; the speed is
  the same. V6 catches only a broken path.
- The FA4 FP8-dequant kernel and the FA3/FA4 mix are first used in the paid run (the 4090 runs Triton). A failed
  engine start gives INCONCLUSIVE, not a false GO, and the session goes on with m26 and m31bf.
- The launch-order split may fail on the FA3/FA4 mix. The count split is the fallback; else V4c is unknown (PIVOT).
- The pair batch comes from the measured pool. A smaller pool gives a smaller B and a lower S (conservative).
- A padded CUDA graph batch adds GEMM rows, so T rises a little and S falls a little (conservative).
- No compact kernel exists, so S is a ceiling. If the FP8 kernel is not memory-bound, the +25% QK work can matter;
  M1 catches that case.
- One GPU, one session, text only; no speculative decoding, no prefix caching, no API server (as E-49).
- The m31bf groups have no replicates (secondary only). The nightly image is a moving target; the digest pins it.

## 11. Files
- `../gemma4_kv_audit/decode_probe.py` (patched for E-50): `token_budget`, the budget guard, finish reasons per
  request, pair cells with `auto_batch`, `--kv-cache-dtype`, the CUDA graph sizes. E-49 behavior is otherwise
  unchanged; `test_e49.py` passes.
- `../gemma4_kv_audit/analyze_e49.py`: one filter line (pair cells are not slope cells). The E-49 decision
  reproduces unchanged from the frozen data.
- `analyze_e50.py`: the groups, S per group, the checks, the count split and the rule. It reuses `analyze_e49.py`
  (the B = 4 slope cells, V1 to V9) and `../fp8_quant_fusion/analyze_e44.py` (telemetry).
- `session_e50.sh`: the Pod session (STUB=1 runs it on a Mac; ONLY selects arms).
- `test_e50.py`: 14 tests (Section 12).
- Upload to the Pod: `gemma4_kv_headroom` and the E-49 list (`gemma4_kv_audit`, `safedispatch_testA2`, `runpod`,
  `fp8_quant_fusion`, `nvfp4_dispatch`, `safedispatch_testA`, `ledger`), without `results`.

## 12. Verification record (stage S3 and S5)
- S3 (2026-10-30, Mac): `uv run --no-project --with numpy --with pytest python -m pytest test_e50.py`: 14 passed.
  The tests: the token budget (435 for 64K at B = 8), the pair cell syntax and `auto_batch`; the R-65 case on a
  fake engine with the vLLM token budget (the old budget leaves 10 tokens and the guard stops the cell; the new
  budget keeps all 8 requests); a request that stops early is "finished_early" with its reason, also during the
  prefill; a cell longer than `max_model_len` is not run; GO, KILL, PIVOT (KERNEL-BOUND, below the bar),
  INCONCLUSIVE (a lost group, a failed FP8 arm) and INVALID (V11, V5, no copy bandwidth) on stub data; M2 unknown
  without a 64K cell; a pair cell with batch 4 stays out of the slope; the count split; a stub session from a
  clean copy; a failed arm, the time cap and a rerun. `test_e49.py`: 20 passed. A stub session ends with GO
  (32K: S 0.120 at B = 22; 64K: S 0.138 at B = 11).

## Amendments
None yet.
