# E-49: Gemma 4 decode waste audit and the K=V cache lead on 1 x H100 (preregistration)

Status: PREDECLARED 2026-10-29. NOT RUN. Idea: I-182. Journal entry: Section 6.1 E-49.
Request: D-73 (user, 2026-10-29: "Yes, plan it"): one profiling test on a new model type.
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
In vLLM nightly on 1 x H100, does the duplicated K and V cache of the Gemma 4 global layers cost at least 5% of
decode step time at 32K and 64K context, so that a compact K=V cache (37.5% fewer global KV bytes) would make
decode at least 5% faster? "No" is a saving share S below 2% in every deciding cell.

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
  1,024 values per head and token where 640 are enough.
- Mechanism (exact, checked for free): store [V (512) | K on the 128 rotated dims] per head and token; the query
  absorbs w on the 384 unrotated dims; V is a view of the same row. This is the MLA cache shape (a shared row plus
  a decoupled RoPE part). `kv_layout_check.py`: float64 error 8e-15; with a BF16 cache the error is no larger than
  the standard BF16 layout (0.0028 to 0.0063 against 0.0035 to 0.0070).
- Byte model (`results/layout_check.json`, INTERPRETATION): the saving is 0.5% to 0.7% of decode bytes at 2K,
  1.9% to 2.7% at 8K, 6.5% to 8.9% at 32K and 11% to 14% at 64K. The global KV capacity rises 1.6 times.
- Prior work (pass 1): vLLM PR #41944 (open) removes only the duplicated V projection weights (+0% to 1.7%);
  PR #60674 fuses the QKV norms and RoPE; no PR, issue or code found in vLLM, SGLang, TensorRT-LLM or llama.cpp
  that stores the K=V cache once. Not proof of absence; pass 2 repeats the search (Section 8).
- What would kill it early: a fast GPU path that makes the global attention a small share (S below 2%), or a
  global attention kernel so slow that time does not follow bytes (then the lead is the kernel, not the layout).
- Why this order: the free check and the byte model first, then the cheapest GPU test that measures the real
  time share (a context sweep, no new kernel). A prototype kernel comes only after a GO, with its own plan.

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
- Engine (both): in-process, prefix caching off, text only (`limit_mm_per_prompt` image 0, audio 0),
  `max_model_len` 66,048, `max_num_seqs` 32, `gpu_memory_utilization` 0.92, BF16 KV cache, default attention
  backend, default compile and CUDA graphs, `detokenize=False`, greedy, `ignore_eos`.
- Data: prompts are windows of real text tokens (the GSM8K test prompts, joined, special tokens removed), a
  different offset per request (seed = context x 1000 + batch).
- Cells: slope cells L in {2,048, 8,192, 32,768, 65,536} at B = 4 (m31) and B = 8 (m26). Waste cells L = 1,024 at
  B in {1, 8, 32}. Profiles (torch.profiler, 8 decode steps) at L 2,048 and 32,768 of the slope batch and at the
  waste cells.
- Measurement per cell (`decode_probe.py`): add B requests, step until every request is in decode, 4 warm
  steps, then blocks of 32 timed engine steps (wall clock, CUDA synchronized); 3 blocks, up to 5 if the block
  spread is above 2%.

## 5. Metrics
- Primary: S(L) = 0.375 x b_low x L / T(L), where b_low is the one-sided 95% lower bound of the slope b of the
  least-squares line T(L) = c + b L over all blocks of the slope cells, and T(L) is the median step time at L.
  Mechanism: for L >= 1,024 the sliding layers read a fixed window and the weights do not depend on L, so the
  growth of T with L is the global attention.
- Minimum effect that matters: 5% of decode step time.
- Secondary (reported only): the point value of S; the achieved bandwidth of the global attention
  (global KV bytes per token x B / b) against the copy bandwidth; prefill seconds; the logged KV capacity and the
  modelled compact capacity; the waste table (kernel class shares, top kernels, GPU idle share per step) at
  every profiled cell; GSM8K accuracy.
- Repeats: 3 to 5 blocks of 32 steps per cell; 12 to 20 points per slope fit.

## 6. Validity checks (a failed check makes that model INVALID)
- V1: the GPU name is H100 80GB.
- V2: decode only: in every timed block each request advanced 31 to 33 tokens for 32 steps.
- V3: linearity: every slope-cell median lies within 3% of the fitted line (R^2 reported only).
- V4a: the step-time growth from 2K to 32K is GPU time: device-time growth / wall-time growth >= 0.8.
- V6: GSM8K (first 32 test problems, greedy, 768 tokens) accuracy >= 50% (catches a broken checkpoint path, as in
  vLLM issue #39407).
- V7: at most one slope cell with a block spread above 2% after 5 blocks.
- V8 (session): hard throttle (HW slowdown, thermal or power brake bits) in at most 2% of busy telemetry samples.

## 7. Decision rule (frozen)
Deciding cells: L = 32,768 and 65,536 for each valid model (up to 4). A cell passes if S >= 0.05 and the model
passes M1 and V4b:
- M1 (memory-bound): the achieved bandwidth of the global attention is at least 60% of the measured copy
  bandwidth. If not, fewer bytes may not give less time.
- V4b (attribution): attention kernels (name regex in `analyze_e49.py`) explain at least 80% of the device-time
  growth from 2K to 32K.
Outcomes:
- GO: at least 2 cells pass and every valid model has a passing cell. If only one model is valid, the label adds
  "(one model)".
- KILL: S (point value) below 0.02 in every deciding cell.
- PIVOT: otherwise, with the reason: "KERNEL-BOUND or attribution unclear" (S large but M1 or V4b fails: the
  head-dim-512 attention kernel becomes the lead), "one model or one cell only", or "below the 5% bar".
- INVALID: no valid model, or V8 fails.
A GO allows the next step only: a prototype plan (compact cache plus kernel) with its own preregistration, a
second pass-2 prior-work search and a review. It is not a speed claim.

## 8. Controls and baselines
- Baseline: the stock vLLM path on the same Pod and image (no change to vLLM is tested here).
- A/A: repeated blocks inside each cell (V7) and the residual check (V3).
- Positive control: the slope must be positive and consistent with the byte model (achieved bandwidth below the
  copy bandwidth); a slope above it means the byte model is wrong (reported).
- Negative control: the sliding layers. Their cost must not grow from 2K to 64K (V3 and the profile tables).
- Pass 2 (before a GO is reported): repeat the prior-work search in vLLM, SGLang, TensorRT-LLM, llama.cpp,
  FlashInfer and arXiv (K=V cache, Gemma 4 KV, MLA-style layout for GQA models).

## 9. Budget
- H100 SXM Secure at the live price (E-46: $3.99/h); network volume 120 GB. Setup and downloads (85 GB) about
  12 min; m31 about 12 min (engine start, GSM8K 32, 7 cells, 5 profiles); m26 about 12 min; analysis 1 min.
  About 40 min. Caps: no new work after 55 min, session watchdog 65 min, `PROBE_TIMEOUT` 1,500 s per model,
  `AUTO_STOP=1`, boot watchdog CAP 4,200 s (70 min). Cost cap $6 for the H100 and $0.60 for the rehearsal.
- Live catalog read (2026-10-29, `get-gpu-type`, Secure, Pod, CUDA >= 13.0): H100 SXM $3.99/h, stock LOW in
  AP-IN-1, AP-JP-1, EU-FR-1 and US-NE-1. L4 $0.59/h but no stock with CUDA >= 13.0, so the rehearsal uses an
  RTX 4090 ($0.89/h, stock MEDIUM in EU-RO-1; RUNPOD.md allows it). Balance check
  (`balance.py --need 6.6 --hours 1.2`): ok.
- Order: m31 first (the strongest prediction), then m26. A time cap cuts m26 first. Create-to-start in one tool
  call.

## 10. Threats to validity
- The probe runs the in-process engine; the API server overlaps output work in another process. The per-step
  CPU work enters c only, so S is lower than in a server (conservative).
- No new kernel is measured. S assumes that a compact-layout kernel keeps the achieved bandwidth (checked
  indirectly by M1). The prototype must test it.
- BF16 KV cache only. With an FP8 KV cache the global share halves and S falls by about half.
- One GPU type, two models, text only, no speculative decoding, no prefix caching, fixed batch per model.
- The 26B-A4B weight bytes depend on routing; the byte model assumes uniform routing (an upper bound on bytes).
- The nightly image is a moving target; the digest pins it.

## 11. Files
- `kv_layout_check.py`: the free identity check and the byte model (numpy only). Output `results/layout_check.json`.
- `decode_probe.py`: one model per process; cells, profiles, GSM8K sanity, KV record; `--stub` for the Mac tests.
- `analyze_e49.py`: the slope fit, S, M1, V1 to V8 and the rule; the waste table.
- `session_e49.sh`: the Pod session (STUB=1 runs it on a Mac).
- `test_e49.py`: 9 tests (identity, fit, GO, KILL, PIVOT kernel-bound, PIVOT attribution, INVALID paths, one model
  and noise, stub session with a rerun skip).
- Reused: `../safedispatch_testA2/gsm8k.py`, `../fp8_quant_fusion/analyze_e44.py` (telemetry rule),
  `../runpod/gpu_microbench.py`, `../runpod/hardware_record.sh`, `../runpod/pod_cmd.py`.

## 12. Verification record (stage S3 and S5)
- S3 (2026-10-29, Mac): `uv run --no-project --with numpy --with pytest python -m pytest test_e49.py`: 9 passed.
  A stub session ends with "GO" and skips a finished model on a rerun. `kv_layout_check.py`: exact.
- S5 (to do): RTX 4090 rehearsal, 20 to 30 min, image above, model `google/gemma-4-E2B-it` 3e22461f (no K=V layers; it
  tests the mechanics only). VERIFY: the image starts and `import vllm` works; `decode_probe.py` ends with an ok
  `run.json`; every timed block advanced 31 to 33 tokens; `kernels.json` has attention kernels that the regex
  matches; the log has "GPU KV cache size"; GSM8K accuracy above 0; the analysis runs (INVALID is expected: not
  an H100); the boot watchdog and AUTO_STOP stop the Pod.

## Amendments
(none)
