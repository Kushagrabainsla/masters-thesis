# E-49: Gemma 4 decode waste audit and the K=V cache lead on 1 x H100 (preregistration)

Status: PREDECLARED 2026-10-29. AMENDED A1 2026-10-30 (R-64 pre-run review, before any data). S5 rehearsal PASS
2026-10-30. S6 (H100) not started: the live balance check failed after the rehearsal (other spend on the shared account).
Idea: I-182. Journal entry: Section 6.1 E-49. Review: R-64 (`dossier/REVIEW_E-49_PRERUN_2026-10-29.md`).
Request: D-73 (user, 2026-10-29: "Yes, plan it"): one profiling test on a new model type.
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.
Text marked "(A1)" changed in Amendment A1. The version before A1 is commit 56bb221.

## 1. Question
In vLLM nightly on 1 x H100, does the duplicated K and V cache of the Gemma 4 global layers cost at least 5% of
decode step time at 32K and 64K context, so that a compact K=V cache (37.5% fewer global KV bytes) would make
decode at least 5% faster? "No" is a saving share S whose upper bound is below 2% in every deciding cell (A1).

## 2. Rubber duck
- Why Gemma 4 (pass 1 of D-73, 2026-10-29): the hybrid recurrent class that D-73 named is crowded. FACT (GitHub
  title search, open vLLM PRs): GDN 96, Mamba 110, KDA 39, MLA 222, MXFP4 104, MTP 120. Gemma 4 is the most
  downloaded open family of 2026 (FACT, Hugging Face: gemma-4-31B-it 8.4M, gemma-4-26B-A4B-it 6.7M) with 39 open
  vLLM PRs, and its attention has an unusual form.
- The lead came from free source reading, not from a profile. FACT: Gemma 4 global layers have no v_proj
  (`attention_k_eq_v`); V = RMSNorm(W_k x) and K = p-RoPE(w * V), with p = 0.25 (64 of 256 angle pairs rotate,
  dims 0..63 and 256..319). FACT (Gemma 4 report, arXiv 2607.02770, Section 2): K=V with p-RoPE "effectively
  reduc[es] the global KV cache by 37.5%". FACT (source, main branch, 2026-10-29): vLLM `gemma4.py` loads K into
  both the K and V slots and passes separate K and V to a standard `Attention`; SGLang `gemma4_causal.py` and
  TensorRT-LLM `modeling_gemma4.py` (standard `KVCacheLayerSpec`) also store K and V. So every engine checked stores
  1,024 values per head and token where 640 are enough. R-64 confirmed the mechanism in four code bases.
- Mechanism (exact, checked for free): store [V (512) | K on the 128 rotated dims] per head and token; the query
  absorbs w on the 384 unrotated dims; V is a view of the same row. This is the MLA cache shape (a shared row plus
  a decoupled RoPE part). `kv_layout_check.py`: float64 error 8e-15. The layout is exact in real arithmetic, not
  in BF16 bits (R-64): with the HF casting order and heavy-tailed keys, the median error against float64 was 1.1%
  (standard) and 1.4% (compact). A prototype needs its own accuracy gate. E-49 measures time only.
- Byte model (`results/layout_check.json`, INTERPRETATION): the saving is 0.5% to 0.7% of decode bytes at 2K,
  1.9% to 2.7% at 8K, 6.5% to 8.9% at 32K and 11% to 14% at 64K. The global KV capacity rises 1.6 times.
- Prior work (pass 1 and R-64 pass 2): vLLM PR #41944 (open) removes only the duplicated V projection weights;
  PR #60674 fuses the QKV norms and RoPE; GTA (arXiv 2505.21487) has the tied-KV layout idea as an architecture,
  with no kernel. No engine or paper found that stores the Gemma 4 K=V cache once. Not proof of absence.
- The real competitor (A1, R-64 O4): FP8 or NVFP4 KV at head_dim 512 (vLLM PR #60372, #53175; FlashInfer). It cuts
  50% of all KV bytes, is lossy and ships in the nightly. The compact cache is exact and can stack with it. A GO
  must be compared with FP8 KV and with the stacked case in the prototype plan.
- What would kill it early: a fast GPU path that makes the global attention a small share (S below 2%), or a
  global attention kernel so slow that time does not follow bytes (then the lead is the kernel, not the layout).
- Why this order: the free check and the byte model first, then the cheapest GPU test that measures the real
  time share (a context sweep, no new kernel). A prototype kernel comes only after a GO, with its own plan.
- Expected outcome (A1, R-64 D9, INTERPRETATION): GO needs both 64K cells; the 32K cells sit near the 5% bar. The
  likely non-GO outcome is PIVOT. KILL is almost unreachable.

## 3. Hypotheses
- H1 (primary): S >= 5% at 32K and 64K for the dense 31B FP8 model and at 64K for the 26B-A4B BF16 model.
- H0: the global attention is a small part of the decode step (S below 2%), or it is not memory-bound.

## 4. Setup (frozen)
- Hardware: 1 x H100 80GB HBM3 (SXM), Secure Cloud, on-demand, US-NE-1 if in stock, `minCudaVersion` 13.0. Why:
  the most common GPU for one-GPU Gemma 4 serving (the vLLM recipe runs 26B-A4B on 1 x H100).
- Software: `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3` (tag
  `nightly`, commit 3966924d, 2026-10-29). The session records the vLLM version, torch, collect_env and the
  attention backend lines.
- Models (revisions pinned): m31 `RedHatAI/gemma-4-31B-it-FP8-dynamic` d4ab4f57 (33.3 GB, compressed-tensors FP8
  dynamic; BF16 does not leave room for long context on one GPU); m26 `google/gemma-4-26B-A4B-it` 4d7ae498
  (51.6 GB BF16, the vLLM recipe).
- Engine (both): in-process, prefix caching off, text only (`limit_mm_per_prompt` image 0, audio 0 and video 0,
  so vLLM runs in text-only mode (A1, R-64 R3)), prefill chunk `max_num_batched_tokens` 2,048 (A1, R-64 R2),
  `max_model_len` 66,048, `max_num_seqs` 32, `gpu_memory_utilization` 0.92, BF16 KV cache, default attention
  backend, default compile and CUDA graphs, `detokenize=False`, greedy, `ignore_eos`.
- Expected attention backend (A1, R-64 R4, FACT from source): H100 (SM90): FlashAttention 4 for all layers
  (`Gemma4Config` sets `flash_attn_version = 4` when FA4 is supported, SM 9.x to 11.x). RTX 4090 (SM89, the
  rehearsal): TRITON_ATTN. So the FA4 kernel names and its head_dim 512 path are first used in the paid run.
- Data: prompts are windows of real text tokens (the GSM8K test prompts, joined, special tokens removed), a
  different offset per request (seed = context x 1000 + batch + 7,919 x replicate).
- Cells (A1, R-64 R5, R6, O3): slope cells L in {2,048, 8,192, 32,768, 65,536} at B = 4 (m31) and B = 8 (m26), then
  a second instance (replicate 1, new prompts) of 65,536 and 32,768, in that order. Waste cells (exploratory, run
  last): L = 1,024 at B in {1, 8, 32}, and L = 16,384 at B = 8 (m31) or B = 16 (m26). Profiles (torch.profiler, 8
  decode steps, kernel table and every kernel event in start order) at L 2,048, 32,768 and 65,536 of the slope
  batch and at the 1,024 cells.
- Measurement per cell (`decode_probe.py`): add B requests, step until every request is in decode, then check the
  scheduler: B requests running, none waiting, no preemption (A1, R-64 R2). If the check fails, the cell is
  "preempted": no timing, no profile, and the probe goes on. Else 4 warm steps, then blocks of 32 timed engine
  steps (wall clock, CUDA synchronized); 3 blocks, up to 5 if the block spread is above 2%; the scheduler check
  again after the blocks.

## 5. Metrics
- Primary: S(L) = 0.375 x b_low x L / T(L). T(L) is the median step time at L (the median over the cell
  instances at L). b_low is the cell-level one-sided 95% lower bound of the slope b of the least-squares line
  T = c + b L over the cell medians of the slope cells: 6 points, df = 4, t = 2.132 (A1, R-64 R5).
  Mechanism: for L >= 1,024 the sliding layers read a fixed window and the weights do not depend on L, so the
  growth of T with L is the global attention.
- S is an upper bound for the saving (A1, R-64 R7, D7): time may not fall fully with bytes, and the compact QK
  product has 640 dims (+25% QK work). A GO is a ceiling, not a speed claim.
- Minimum effect that matters: 5% of decode step time.
- Secondary (reported only): the point value of S and its upper bound S_up; the block-level bound S_block_low
  (blocks as independent points, the version before A1); S_prof = 0.375 x global attention ms per step (from the
  layer split of the profile) / T(L), flagged if it differs from S by more than 25% (A1, R-64 R6); the achieved
  bandwidth of the global attention (global KV bytes per token x B / b) against the copy bandwidth; prefill
  seconds; the KV record (blocks, groups, page bytes, the logged KV size and maximum concurrency) and the modelled
  compact capacity; the waste table (kernel class shares, top kernels, GPU idle share = 1 - device ms per step /
  the unprofiled median step time of the same cell (A1, R-64 R1)) at every profiled cell; the exploratory S_est at
  the 16K cells (the slope per sequence x B x L / T); GSM8K accuracy.
- Repeats: 3 to 5 blocks of 32 steps per cell; 6 cells per slope fit.

## 6. Validity checks (a failed check makes that model INVALID)
- V1: the GPU name is H100 80GB.
- V2: decode only: in every timed block of the slope cells each request advanced 31 to 33 tokens for 32 steps.
  Preempted cells are left out of the fit and listed (A1).
- V3: linearity: every context median lies within 5% of the fitted line (A1, R-64 O1: was 3%). R^2 is reported.
- V4a: the step-time growth from 2K to 32K is GPU time: device-time growth (profiles) / wall-time growth (the
  unprofiled block medians, A1, R-64 R1) >= 0.8. It applies only if b_low > 0 (A1).
- V5 (A1, R-64 R7): positive control: if b_low > 0, the achieved bandwidth of the global attention is at most 1.15
  x the copy bandwidth. Above it the byte model is wrong.
- V6: GSM8K (first 32 test problems, greedy, 768 tokens, the duplicate BOS removed (A1, R-64 O8)) accuracy >= 50%
  (catches a broken checkpoint path, as in vLLM issue #39407).
- V7: at most one slope cell with a block spread above 2% after 5 blocks.
- V8 (session): hard throttle (HW slowdown, thermal or power brake bits) in at most 2% of busy telemetry samples.
- V9 (A1, R-64 R3): the log has the vLLM line "running in text-only mode".
- Session (A1, R-64 R7): no copy bandwidth measurement makes the session INVALID (M1 and V5 cannot run).
- A run counts if `run.json` has status "ok" or "ok_slope" (the slope cells are done; A1, R-64 R8).

## 7. Decision rule (frozen; A1 before any data)
Deciding cells: L = 32,768 and 65,536 for each valid model (up to 4). A cell passes if S >= 0.05 and the model
passes M1, M2, V4b and V4c:
- M1 (memory-bound): the achieved bandwidth of the global attention is at least 60% of the measured copy
  bandwidth (and V5 holds). If not, fewer bytes may not give less time.
- M2 (A1, R-64 O1): the 32K to 64K slope is within 25% of the fitted slope (no curvature at long context).
- V4b (attribution): attention kernels (name regex in `analyze_e49.py`; KV writes excluded, A1) explain at least
  80% of the device-time growth from 2K to 32K. If no kernel in the 32K profile matches, V4b is unknown (A1).
- V4c (A1, R-64 R6, negative control): the sliding-layer attention ms per step at 32K (and at 64K if profiled) is
  within 5% of 2K. Layers come from the launch order: each layer launches one group of consecutive attention
  kernels, and group i belongs to layer i mod n (n from the model's `layer_types`). The split is used only if
  the profile has exactly 8 x n groups and at least 90% of the groups repeat the first kernel name of the same
  layer in step 1. Otherwise V4c is unknown.
Outcomes:
- GO: at least 2 cells pass and every valid model has a passing cell. If only one model is valid, the label adds
  "(one model)".
- KILL: S_up (from the upper bound of b) below 0.02 in every deciding cell (A1, R-64 O2: was the point value).
- PIVOT: otherwise, with each reason: "KERNEL-BOUND (M1)" (S large but not memory-bound: the head-dim-512
  attention kernel becomes the lead), "curvature (M2)", "attribution unclear" (V4b or V4c false), "attribution
  unknown" (A1: not KERNEL-BOUND), "one model or one cell only", or "below the 5% bar".
- INVALID: no valid model, V8 fails, or no copy bandwidth.
A GO allows the next step only: a prototype plan (compact cache plus kernel) with its own preregistration, a
second pass-2 prior-work search and a review. It is not a speed claim.

## 8. Controls and baselines
- Baseline: the stock vLLM path on the same Pod and image (no change to vLLM is tested here).
- A/A: repeated blocks inside each cell (V7), the replicate cells (the cell-level bound) and the residual check.
- Positive control (A1, R-64 R7): the copy bandwidth must exist and bound the achieved global-attention bandwidth
  (V5).
- Negative control (A1, R-64 R6): the sliding-layer attention must stay flat from 2K to 32K and 64K (V4c).
- Pass 2 (before a GO is reported): repeat the prior-work search in vLLM, SGLang, TensorRT-LLM, llama.cpp,
  FlashInfer and arXiv (K=V cache, Gemma 4 KV, MLA-style layout for GQA models). R-64 did one pass 2 before the run.

## 9. Budget
- H100 SXM Secure at the live price; network volume 120 GB.
- KV memory at the 2,048 chunk (A1, R-64 R2 and Section 5, INTERPRETATION): KV available is 32.6 to 37.6 GiB (m31)
  and 16.5 to 21.0 GiB (m26). The largest need is the peak while the last request of the 64K cell prefills:
  26.4 GiB (m31) and 12.4 GiB (m26). The 32K peak is 16.4 and 7.4 GiB; steady decode at 64K is 23.3 and 11.6
  GiB. At the default chunk (16,384) the m31 peaks are 38.2 and 48.2 GiB: preemption, so the default is not used.
  The exploratory 16K cells need about 16 GiB (m31, B = 8) and 8 GiB (m26, B = 16).
- Time (A1, R-64 Section 5, INTERPRETATION): Pod boot 3 to 5 min; setup and downloads (85 GB) about 10 min; each
  model 13 to 17 min (engine start 5 to 9 min, GSM8K 32, 8 slope and 4 waste cells, 6 profiles); analysis 1 min;
  the AUTO_STOP wait 5 min. About 45 to 55 min in total.
- Caps: no new work after 55 min; the second model starts only if 15 min remain before that (`capped 900`, A1,
  R-64 R8); session watchdog 65 min; `PROBE_TIMEOUT` 1,500 s per model; `AUTO_STOP=1`; boot watchdog CAP 4,200 s
  (70 min). Cost cap $6 for the H100 and $0.60 for the rehearsal. Expected cost about $3 to $4 (H100) and $0.45
  (rehearsal).
- Live catalog read (2026-10-29, `get-gpu-type`, Secure, Pod, CUDA >= 13.0): H100 SXM $3.99/h, stock LOW in
  AP-IN-1, AP-JP-1, EU-FR-1 and US-NE-1. L4 $0.59/h but no stock with CUDA >= 13.0, so the rehearsal uses an
  RTX 4090 ($0.89/h, stock MEDIUM in EU-RO-1; RUNPOD.md allows it). Balance check
  (`balance.py --need 6.6 --hours 1.2`): ok. Read both again before each booking.
- Order: m31 first (the strongest prediction), then m26. A time cap cuts m26 first. Create-to-start in one tool
  call.

## 10. Threats to validity
- The probe runs the in-process engine; the API server overlaps output work in another process. The per-step
  CPU work enters c only, so S is lower than in a server (conservative).
- No new kernel is measured. S assumes that a compact-layout kernel keeps the achieved bandwidth (checked
  indirectly by M1). S is an upper bound (Section 5). The prototype must test the real saving.
- BF16 KV cache only. With an FP8 KV cache the global share halves and S falls by about half.
- One GPU type, two models, text only, no speculative decoding, no prefix caching, fixed batch per model.
- The 26B-A4B weight bytes depend on routing; the byte model assumes uniform routing (an upper bound on bytes).
  Routing also gives a cell effect (different prompts route to different experts); the cell-level bound covers it.
- The nightly image is a moving target; the digest pins it.
- Residual risk (A1, R-64): the FA4 head_dim 512 path, its kernel names and the layer split on FA4 are first used
  in the paid run. The rehearsal GPU cannot run FA4. A wrong name gives "attribution unknown", not a false GO.
- Prototype note (A1, R-64 O5): the compact row is 640 values x 4 heads x 2 B = 5,120 B per token and layer
  (31B); the sliding page is 16,384 B per token and layer. The hybrid allocator needs equal page bytes, so a
  prototype needs odd block sizes (80 and 256 tokens). Plan this before any claim of 1.6 times capacity.

## 11. Files
- `kv_layout_check.py`: the free identity check and the byte model (numpy only). Output `results/layout_check.json`.
- `decode_probe.py`: one model per process; cells, scheduler check, profiles with kernel events, GSM8K sanity, KV
  and layer record; `--stub` for the Mac tests.
- `analyze_e49.py`: the cell-level slope fit, S, M1, M2, V1 to V9, the layer split and the rule; the waste table.
- `session_e49.sh`: the Pod session (STUB=1 runs it on a Mac; ONLY selects the models).
- `test_e49.py`: 20 tests (Section 12).
- Reused: `../safedispatch_testA2/gsm8k.py`, `../fp8_quant_fusion/analyze_e44.py` (telemetry rule),
  `../runpod/gpu_microbench.py`, `../runpod/hardware_record.sh`, `../runpod/pod_cmd.py`.
- Upload to the Pod (A1, R-64 R9): the folders `gemma4_kv_audit`, `safedispatch_testA2`, `runpod`,
  `fp8_quant_fusion`, `nvfp4_dispatch`, `safedispatch_testA` and `ledger` under `experiments/`, without their
  `results` and `dryrun` folders. A test runs the stub session from a clean copy of only these folders.

## 12. Verification record (stage S3 and S5)
- S3 (2026-10-29, Mac): 9 tests passed. A stub session ends with "GO" and skips a finished model on a rerun.
- S3 after A1 (2026-10-30, Mac): `uv run --no-project --with numpy --with pytest python -m pytest test_e49.py`:
  20 passed. New tests: real kernel names and classes (FA3 from the E-46 profiles, Triton, the vLLM KV write);
  a profiler start cost of 1.2 s does not change V4a or the idle share; opaque kernel names give "attribution
  unknown"; sliding growth fails V4c; a cell effect widens the cell-level bound; an impossible bandwidth or no
  copy bandwidth is INVALID; a preempted cell is left out; an "ok_slope" run counts; no text-only line fails V9;
  curvature fails M2; the probe loop on a fake engine (timing, advance, scheduler check, preempted cell, abort);
  a stub session from a clean copy; a failed run, the time cap for the second model and a rerun that keeps the
  failed output.
- S5 (2026-10-30, RTX 4090, EU-RO-1, Pod vbfqjtlowdrfoa, 14 min, about $0.21): PASS, no code change needed.
  vLLM 0.33.0rc1.dev229+g3966924d5 started; `run.json` ok; the log has the text-only line, the KV size
  (1,245,498 tokens) and "FA4 not available, forcing TRITON_ATTN"; `run.json` has the KV groups and 35 layer types;
  all 10 cells ok, the scheduler check ok, every block advanced 32 tokens; the Triton attention kernels
  (`kernel_unified_attention`, `reduce_segments`) match and the KV write (`reshape_and_cache_kernel_flash`) is
  excluded; the layer split is ok at 2K, 32K and 64K (280 groups = 8 x 35, match 1.0); GSM8K 59%; the analysis
  ran (INVALID as expected: V1, and V5 uses the 31B byte constants); AUTO_STOP stopped the Pod; 49 files copied
  with matching checksums. The profile wall per step was 2.0 times the block median at 2K, so R1 was a real
  blocker. Exploratory (E2B on Triton, not a decision): the sliding-layer attention grows from 0.39 to 1.09 ms per
  step between 2K and 32K and stays flat to 64K (V4c false; S_prof 0.078 against S 0.087 at 32K). The global
  attention grows 1.96 ms per 32K tokens, about the copy bandwidth for the E2B byte count (INTERPRETATION).
- S5 plan (as run): RTX 4090 rehearsal, 20 to 30 min, image above, `ONLY=m31`, `M31` and `M26` set to
  `google/gemma-4-E2B-it` revision 3e22461f65e89153144f8adb70e3b8c2cc9845a7 (10.2 GB; no K=V layers; it tests
  the mechanics only), boot watchdog CAP 1,800 s, `AUTO_STOP=1`. VERIFY:
  - the image starts and `import vllm` works; `decode_probe.py` ends with an ok `run.json`;
  - the log has "running in text-only mode" and "GPU KV cache size"; `run.json` has the KV groups and the
    layer types;
  - every slope cell has status ok, a scheduler check with ok true, and every timed block advanced 31 to 33
    tokens;
  - `kernels.json` has attention kernels that the regex matches (Triton names); the events files exist and the
    layer split is ok (8 x n groups) on Triton;
  - GSM8K accuracy above 0; the analysis runs (INVALID is expected: not an H100);
  - the boot watchdog and AUTO_STOP stop the Pod.
  Not testable on the 4090 (residual risk, A1): the FA4 kernel names, the FA4 layer split, the FA4 head_dim 512
  path and the H100 memory margins.

## Amendments
- A1 (2026-10-30, before any data; from R-64, verdict AMEND_THEN_RUN). Each item has a test (Section 12).
  - R1 (blocker): V4a and the idle share use the unprofiled block medians, not the profile wall time (it has the
    profiler start). V4a applies only for a clear slope (b_low > 0).
  - R2 (blocker): prefill chunk 2,048 (the default 16,384 can preempt the 31B cells at 32K and 64K); the KV record;
    a scheduler check after prefill and after the blocks; a failed check marks the cell "preempted".
  - R3: video limit 0 too, so the engine runs in text-only mode; new check V9; the multimodal state in `run.json`.
  - R4: the expected backend per GPU; a table-driven test of the attention regex with real names; the KV write is
    not attention; no matched attention kernel gives "attribution unknown", not KERNEL-BOUND; FA4 items are
    marked "not testable on the 4090".
  - R5: replicate cells at 32K and 64K for both models; the rule bound is cell-level (df = 4); the block-level
    bound is reported.
  - R6: kernel events for the 2K, 32K and 64K profiles (a 64K profile added); the layer split from the launch
    order; new check V4c; S_prof with a 25% flag.
  - R7: M1 has an upper limit (new check V5, INVALID above 1.15 x copy bandwidth); no copy bandwidth is INVALID;
    S is stated as an upper bound.
  - R8: status "ok_slope" after the slope cells; the second model needs 15 min before the cap; a failed output
    folder is moved, not deleted; the session waits for a free GPU between models.
  - R9: the upload folder list and a clean-copy test.
  - Optional items adopted: O1 (V3 residual limit 5% and the new gate M2), O2 (KILL on the upper bound), O3 (one
    exploratory 16K cell per model at a larger batch), O4 (FP8 KV named as the competitor), O5 (the page-size note),
    O6 (the fake-engine test of the probe loop), O8 (the GSM8K tokenizer from the pinned snapshot and no duplicate
    BOS).
  - Optional item not adopted: O7 (a 10 min H100 smoke run, about $0.7). Reason: it needs the same model download
    and engine start as the session, so it costs about as much as the failure it guards against. A failed FA4
    start in the session ends both probes early (about $1 to $1.5 lost).
