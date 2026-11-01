# E-53: accuracy of the compact FP8 K=V global cache on real Gemma 4 31B activations (gate G3 of D-88; preregistration)

Status: PREDECLARED 2026-11-01. (A1) AMENDED A1 2026-11-01 after the S2 review R-73 (verdict AMEND_THEN_RUN,
`dossier/REVIEW_E-53_PRERUN_2026-11-01.md`), before any data. (A2) AMENDED A2 2026-11-01 after S5 rehearsal 1 (a bug
in the run record), before any data. (A3) AMENDED A3 2026-11-01 after S5 rehearsal 2 (the copy). S5 PASS (rehearsal
2). RUN 2026-11-01 on the H100: frozen decision FAIL (layer 41: median per-prompt ratio 2.38; Section 13). Post-run
review R-74 pending. S4: D-89 (the user approved the rehearsal and the H100 run
on 2026-11-01, "Run both", with a one-time $10 balance floor).
Idea: I-182. Gate: G3 of D-88 (after G1 PASS and G2 PASS, `dossier/GATE_G1_PAGESIZE_I-182_2026-11-01.md` and
`dossier/GATE_G2_UPSTREAM_I-182_2026-11-01.md`). Journal entry: Section 6.1 E-53 (the main session adds it).
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
All thresholds are project heuristics unless a source is given.
(A1) Text marked "(A1)" changed in Amendment A1. The version before A1 is commit 6c22626.

## 1. Question
On real Gemma 4 31B activations, does the compact FP8 KV storage of the global layers (the E-52 views row
[K_rot | V_unrot | V_rot], quantized to FP8 E4M3 with scale 1.0, read with the views query [q_rot | (q * w)_unrot]
rounded to BF16) give an attention-output error no larger than the standard FP8 KV storage (K and V each quantized to
E4M3 with scale 1.0), and do the views-path values fit FP16 with margin?
(A1, R-73 B4) "No" is the FAIL of Section 7: a global layer where the median over the prompts of the per-prompt ratio
err_compact / err_standard is above 2. A FAIL stops the vLLM prototype (D-88). A views query value above 32,752 (half
the FP16 maximum) is no longer a FAIL: a scaled query can fix it, so it gives INCONCLUSIVE with the action "pre-scale
the query".

## 2. Rubber duck
- What the layout claims (FACT, E-52 and R-71): the views layout is exact in real arithmetic (an independent check
  agrees to 1e-15 in float64). The unchanged stock FA4 FP8 kernel reads the compact rows through two views with the
  same error on random data (E-52 Section 13). Real activations were not tested (E-52 Section 10).
- What is new (FACT, G2): the checkpoint has per-channel FP8 weights and `kv_cache_scheme` null, so vLLM uses the FP8
  KV scale 1.0 for K and for V. In vLLM's `gemma4.py` the k_norm weight w scales the normalized k_proj output before
  RoPE. V has a weightless norm and no RoPE. Dims 0..63 and 256..319 rotate (neox rotate_half).
- Where the two storages differ (INTERPRETATION, algebra): both store the same V values in FP8. Only the scores differ.
  On the 384 unrotated dims the standard storage reads fp8(w * V), and the compact storage reads w * fp8(V), with w
  moved into the BF16 query. The E4M3 relative rounding error has the same bound (2^-4) for all normal values (3
  mantissa bits). So the two errors differ mainly through the range: values below 2^-6 are subnormal (lower relative
  precision), values above 448 saturate, and q * w is rounded to BF16.
- Why it can fail (HYPOTHESIS): (a) if w is large on some dims, V there can sit in the subnormal range while w * V does
  not; (b) a large w makes q * w large, and an FP16-compute kernel (the E-52 FP16 path, the stock FA4 FP8 path) can
  overflow; (c) real K and V can have outliers that random data do not have.
- What kills it early: the fail fast of the capture (Section 4.3), and V3 (the identity K = w * V does not hold on
  real data, for example because the module form of w is wrong or another scale exists).
- Why this order: the engine start is the main cost, so the capture runs once and records everything. The analysis
  then runs on the same GPU from the files. The frozen rule decides.
- Expected (HYPOTHESIS): err_standard 2% to 6% (the CPU tests on Gemma 4 K=V data give about 4.9%); the ratio
  err_compact / err_standard 0.9 to 1.1; the largest |q'| below 1,000; no saturated value in either storage; PASS.
- (A1) What R-73 found (FACT, R-73): the hook path and the FP8 cast match the vLLM source; the worst case was above
  the cost cap and the session watchdog fired after the boot watchdog (B1); no check tied the analysis reference to
  the engine (B2); the saturated-share rule failed a layout that saturates the same values (B3); INCONCLUSIVE and
  INVALID had no action (B4). INTERPRETATION (R-73): on synthetic data the error ratio is 0.90 to 1.07 for log-normal
  w with sigma 0.3 and 1.0, so PASS is the likely result; the gate adds real information through V3, the range of q',
  the saturation and the true error level. A1 answers every item (Amendments).

## 3. Hypotheses
- H1 (primary): PASS (Section 7): in every global layer the compact error stays within 1.25 times (median) and 2
  times (largest) the standard error, (A1) per prompt and also in the FP16 emulation, |q'| has a 10 times FP16 margin,
  no FP16 value is lost, and the compact rows do not saturate more values.
- H0 (A1): FAIL (Section 7): in some global layer the median per-prompt ratio (float32) is above 2.

## 4. Setup (frozen)

### 4.1 Hardware, software and model
- Hardware: 1 x H100 80GB HBM3 (SXM), Secure Cloud, on-demand, `minCudaVersion` 13.0. Volume disk 100 GB at
  `/workspace`, container disk 40 GB. Why this GPU: it is the prototype target and the GPU of E-49 to E-52. (A1, R-73
  F8) No cheap GPU holds the 31B weights, so the S5 rehearsal runs a small Gemma 4 model (Section 9).
- Software (FACT, E-49 to E-52): the image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
  (vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, transformers 5.20.2). The session records the versions.
- Model (FACT, E-50, G2): `RedHatAI/gemma-4-31B-it-FP8-dynamic` at revision `d4ab4f579dd3516f97d8a6a4c98d0653480bad15`
  (33.3 GB). The session downloads it with `snapshot_download(model, revision=...)`, as `session_e50.sh` does.
- Model facts (FACT, G1 and G2; (A1) the pinned `config.json`, read 2026-11-01): 60 layers; 10 global layers (5, 11,
  ..., 59) with 32 query heads, `num_global_key_value_heads` 4 and `global_head_dim` 512; `attention_k_eq_v` true;
  `num_kv_shared_layers` 0; attention scale 1.0; no soft-cap. The capture reads the global layers from the config and
  checks them.

### 4.2 Engine (`capture_e53.py`, `ENGINE`)
- `LLM(model=<id>, revision=<sha>, seed=0, enforce_eager=True, kv_cache_dtype="auto" (BF16), max_model_len=8448,
  max_num_batched_tokens=8448, max_num_seqs=4, gpu_memory_utilization=0.85, enable_prefix_caching=False,
  limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})`, in-process with `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
  The id, the revision and text-only mode are as E-50.
- One `generate` call per prompt, with `max_tokens=1` and temperature 0. `max_num_batched_tokens` is larger than the
  prompt, so each prompt is one prefill step (INTERPRETATION; the hooks also handle chunks by position).
- (A1, R-73 F1) Shape flags: `--heads`, `--kv-heads`, `--head-dim` and `--global-layers` keep the frozen 31B values
  (32, 4, 512, 10) as defaults. Only the S5 rehearsal sets them, with `--rehearsal` (Section 9). The head dim must be
  512 (the views layout).

### 4.3 How the hooks reach the model (FACT, vLLM 3966924d5, read with `gh api` on 2026-11-01)
| Source (file: lines) | What it shows |
|---|---|
| `vllm/envs.py`: 1448 to 1450 | `VLLM_ENABLE_V1_MULTIPROCESSING` is 1 if not set; the session sets 0 |
| `vllm/entrypoints/llm.py`: 341, 604 to 614 | `LLM` builds `LLMEngine.from_engine_args`; `LLM.apply_model(func)` calls `llm_engine.apply_model` |
| `vllm/v1/engine/llm_engine.py`: 186 to 188, 116 to 123, 136 to 138 | multiprocessing only if the variable is set; `EngineCoreClient.make_client(multiprocess_mode=...)`; in-process, `model_executor = engine_core.engine_core.model_executor` |
| `vllm/v1/engine/llm_engine.py`: 503 to 504, 506 to 509 | `apply_model` sends `collective_rpc("apply_model", args=(func,))`; (A1, F8) 506 to 509 read `driver_worker.model_runner.model`; the fallback of the capture is `driver_worker.worker.model_runner.get_model()`; both work (`worker_base.py` 370 to 371) |
| `vllm/v1/engine/core_client.py`: 376 to 401, 498 to 505 | `InprocClient` builds `EngineCore` in this process; its `collective_rpc` calls the engine core |
| `vllm/v1/engine/core.py`: 1152 to 1159 | `EngineCore.collective_rpc` calls `model_executor.collective_rpc` |
| `vllm/v1/executor/uniproc_executor.py`: 54, 90 to 103 | `driver_worker = WorkerWrapperBase`; `run_method(self.driver_worker, method, args, kwargs)` in this process |
| `vllm/v1/serial_utils.py`: 485 to 507 | `run_method` finds a method given by name with `getattr` |
| `vllm/v1/worker/worker_base.py`: 165 to 167, 370 to 371 | `apply_model` returns `fn(self.get_model())`; the wrapper sends unknown attributes to the worker |
| `vllm/v1/worker/gpu_worker.py`: 1164 to 1165; `gpu_model_runner.py`: 3078 to 3086 | `get_model` returns the model (unwrapped from a CUDA graph wrapper) |
| `vllm/config/vllm.py`: 1782 to 1784, 1800 | `enforce_eager` sets `CompilationMode.NONE` and `CUDAGraphMode.NONE`, and turns off the JIT warm-up |
| (A1, F7) `vllm/config/compilation.py`: 37 to 41, 53 to 59 | `CompilationMode` is an `IntEnum` and `CUDAGraphMode` an `Enum`; both have the member `NONE`; the capture writes `.name` |
| `vllm/model_executor/models/gemma4_mm.py`: 1145 to 1150 | the language model has the prefix `language_model` |
| `vllm/model_executor/models/gemma4.py`: 419, 451, 453 to 455, 468 to 470, 513 to 518, 520 to 538 | scale 1.0; `k_norm = RMSNorm` (weighted); `v_norm` with `has_weight=False`; `is_sliding` from `config.layer_types`; `get_rope(..., is_neox_style=True)`; `self.attn = Attention(...)` |
| `vllm/model_executor/models/gemma4.py`: 540 to 575, 745 to 749 | `forward(positions, hidden_states)`: `k_norm` (565), `rotary_emb(positions, q, k)` (567), `v_norm` (570), then `self.attn(q, k, v)` (572); the decoder layer calls `self_attn(positions=..., hidden_states=...)` with keywords |
| (A1, F1) `vllm/model_executor/models/gemma4.py`: 422 to 426, 546 to 553 | a layer at or above `num_hidden_layers - num_kv_shared_layers` is KV-shared and calls `self.attn(q, None, None)`; the capture hooks only the global layers that own a cache |
| `vllm/model_executor/layers/attention/attention.py`: 333 to 339, 490 to 497, 538 to 543 | `layer_name`, `num_heads`, `head_size`, `num_kv_heads`, `sliding_window`; `forward(query, key, value, ...)`; the inputs become [tokens, heads, head_size] |
| (A1, B2) `vllm/model_executor/layers/attention/attention.py`: 510, 532 to 533, 584 | the output has the query dtype and [tokens, heads x head_size_v]; `forward` returns it |
| `vllm/model_executor/layers/attention/attention.py`: 136 to 152, 474 to 484 | the default K and V scales are 1.0; the query is quantized only with an FP8 KV cache (not in this capture) |
| `vllm/model_executor/layers/layernorm.py`: 39 to 44, 145 to 150 | `RMSNorm` computes w * x / rms; `GemmaRMSNorm` (1 + w) is not the class of `gemma4.py` |
| `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py`: 58 to 77 | the frequencies use the head size 512 as the denominator and are 0 for the pairs 64 to 255 |
| (A1, F1) transformers 5.20.2 `models/gemma4/configuration_gemma4.py`: 210 to 223 | a global layer takes `global_head_dim`; it takes `num_global_key_value_heads` only if `attention_k_eq_v` is true |

The hooks (`capture_e53.py`, `register` and `Store`):
- A forward pre-hook with kwargs on each `Attention` module whose parent is a `Gemma4Attention` with `is_sliding`
  False and no KV sharing. These are the global layers that own a KV cache. The hook gets (query, key, value) after
  RoPE.
- (A1, R-73 B2) A forward hook on the same module copies the engine's attention output o for the same rows as q.
- A forward pre-hook with kwargs on that parent gets the `positions` of the same rows.
- Rows kept: K and V at every position below the prompt length; Q and (A1) o at the last 256 positions. The hooks copy
  the rows to the CPU at once. After each prompt the script saves BF16 q, k, v, (A1) o and the positions to
  `CAPDIR/p<i>/L<layer>.pt` and records the SHA-256 of each file. It saves the k_norm weight of each global layer to
  `CAPDIR/k_norm.pt`.
- Fail fast, before the first prompt: (A1) the config must have the planned count of global layers that own a cache
  (10), the hooked layers must be those layers, and each must have the planned heads, KV heads and head size (32, 4,
  512) and no sliding window. Fail fast, after the first prompt: every global layer must have q and (A1) o
  [256, 32, 512] and k and v [8,192, 4, 512] in BF16, the key positions 0 to 8,191 once each, and the last 256 query
  positions. If a check fails, the capture stops at once (status "fail_fast", exit 3).

### 4.4 Prompts (`capture_e53.py`, `build_prompts`)
- 4 prompts of exactly 8,192 token ids: BOS, then the tokens of text files of the image (the tokenizer of the pinned
  snapshot, `add_special_tokens=False`), at most 3,000 tokens per file, with the tokens of a blank line after each file.
- The file order of prompt p comes from `random.Random(53 x 1,000 + p)`. Families: p0 the vllm Python files, p1 the
  torch Python files, p2 the Markdown and reStructuredText files of the whole site-packages folder, p3 all of them. A
  usable file has 2 KB to 200 KB and is not a license, notice, authors or change-log file. If a family runs out, the
  prompt continues with the other files. (A1, R-73 Q6) The builder walks each root once.
- (A1, R-73 F4) A file whose token ids hold a special id (`tokenizer.all_special_ids`; for example the text `<bos>` or
  `<|image|>` in a vllm source file) is skipped. So BOS at position 0 is the only special id of a prompt.
  `prompts.json` records `special_after_0` (it must be 0, V2) and the count of skipped files.
- `prompts.json` keeps the token ids, their SHA-256 (little-endian int32 values), and the path and SHA-256 of each
  file. A rerun with the same plan uses the same prompts.

### 4.5 Analysis (`analyze_e53.py`; on the Pod GPU after the capture; float32 with TF32 off)
- Per prompt and global layer: q [256, 32, 512], k and v [8,192, 4, 512] and (A1) the engine output o [256, 32, 512],
  all BF16 as recorded. (A1, R-73 F2) The analysis checks the SHA-256 of each file against `capture_run.json`.
- Attention: causal (the query at position p reads the keys 0 to p), scale 1.0, query head h reads KV head
  h // (heads / KV heads) (8 for both the 31B and the rehearsal model).
  - (a) Reference: the BF16 K and V (the BF16 KV cache holds exactly these values).
  - (b) Standard FP8: fp8(K) and fp8(V), read with the BF16 query.
  - (c) Compact: row = [K_rot | V_unrot | V_rot] (`compact_attn.pack_views`), fp8(row), K view = values 0 to 511, V
    view = values 128 to 639 (`views_of`), query q' = BF16([q_rot | (q * w)_unrot]) with w in BF16 (`views_query`),
    output in the natural order (`unpermute`).
  - (A1, R-73 B2) The engine gap: ||ref - o|| / ||o|| (V5).
  - (A1, R-73 F3) The FP16 emulation of (b) and (c): q, q' and P in FP16 (the FP8 values are exact in FP16). It enters
    the rule (Section 7).
  - (A1, reported only) The score path and the value path of each storage (FP8 K only, FP8 V only; R-73 F6), the
    compact storage in E5M2 (2 mantissa bits; the sensitivity control, R-73 O2), and the compact path with w from the
    data.
- The FP8 cast (FACT, vLLM 3966924d5): `vllm/v1/attention/backends/flash_attn.py` 1432 to 1466
  (`do_kv_cache_update` calls `reshape_and_cache_flash(..., self.kv_cache_dtype, layer._k_scale, layer._v_scale)`);
  `csrc/cache_kernels.cu` 252 to 262 (`CopyWithScaleOp` calls `fp8::scaled_convert`) and 335 to 423
  (`reshape_and_cache_flash_kernel`, the scales at 375 to 379); `csrc/core/vec_dtypes/dtype_fp8.cuh` 22 to 31 ("fp8"
  is `kFp8E4M3`); `csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh` 231 to 243 (BF16 to FP8:
  `__nv_cvt_float_to_fp8(__bfloat162float(a) / scale, __NV_SATFINITE, fp8_type)`) and 285 to 290 (`__NV_E4M3`). So
  the cast divides by 1.0, rounds to nearest even and saturates at +-448. The analysis clamps to +-448 and then casts
  with `to(torch.float8_e4m3fn)`; a test checks the saturation, a tie and the subnormals.
- w: `w_data` per unrotated dim is the median of K / V over tokens and heads where |V| >= 0.05. The module gives w in
  two possible forms, "weight" and "1 + weight" (a Gemma RMSNorm can use 1 + weight), each in BF16. The form nearest
  to `w_data` is the w of the views query: the data decide. HYPOTHESIS: the data choose "weight" (layernorm.py 39 to
  44). (A1, R-73 F5) Without a module weight, V3 fails (a fit of w to the same data cannot show a wrong module w).

## 5. Metrics
- Primary, per prompt and layer: err_standard and err_compact, the relative L2 error ||O - O_ref|| / ||O_ref|| over
  the 256 query rows, 32 heads and 512 dims. Lower is better.
- (A1, R-73 F3) Per prompt p and layer l: the ratio r = err_compact / err_standard (float32), the ratio r16 of the
  FP16 emulation, and rr = max(r, r16).
- Per layer (the decision): (A1) the median of r over the 4 prompts (FAIL), the median and the largest of rr (PASS);
  the largest |q'|; the count of FP16 output values that are not finite; (A1, R-73 B3) the count of saturated values
  (|x| > 448 before the cast) in the compact rows and in the standard rows, pooled over the prompts.
- Reported only: the ratio of the medians and the ratio of the maxima (the rule before A1); the saturated shares, also
  over the same support (the 128 K_rot values per token and head in both storages) and the K_unrot saturation of the
  standard storage (A1, B3); the subnormal and flush-to-zero shares; the largest |q|; the largest |K view| and
  |V view| before and after the cast (after the cast at most 448, so K' and V' always fit FP16: FACT from the E4M3
  range); the identity error with each form of w and (A1, F5) the largest identity error of one token; the RoPE check
  (Section 8); the largest error of one head; (A1) the largest engine gap of one head; the score-path, value-path,
  E5M2 and data-w errors (Section 4.5).
- Minimum effect that matters: the bounds of Section 7 (no speed metric).
- Repeats: 4 prompts for each of the 10 global layers. The computation is deterministic on fixed tensors.

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU of the capture is H100 80GB.
- V2: every global layer of every prompt is recorded with the expected shapes. In detail: the capture status is "ok";
  the plan is the frozen plan (4 prompts, 8,192 tokens, 256 query rows, (A1) 32 heads, 4 KV heads, head dim 512, 10
  global layers); the hooked layers are the global layers of the config; every prompt has 8,192 tokens; every prompt
  and global layer has metrics with q and (A1) o [256, 32, 512], k and v [8,192, 4, 512] in BF16 and the planned
  positions. The fail fast of the capture is a V2 failure. (A1) Also: no capture prompt has a problem; no prompt has a
  special id after position 0 (R-73 F4); every capture file has its recorded SHA-256 (F2); and in a real run (also the
  rehearsal) vLLM is 0.33.0rc1.dev229+g3966924d5, the compile mode and the CUDA graph mode are NONE, the KV cache is
  BF16, and the config has the planned count of global layers (F5, F7).
- V3: the identity holds: for every prompt and layer, ||K_unrot - w * V_unrot|| / ||K_unrot|| <= 1e-2 (the BF16
  level), with the w that the data chose. (A1, R-73 F5) Also: every global layer has a module k_norm weight, and the
  chosen module form is within 0.02 of `w_data` in every prompt and layer.
- V4: the reference is finite for every prompt and layer.
- V5 (A1, R-73 B2): the analysis reference models the engine: for every prompt and layer, ||ref - o|| / ||o|| <= 0.05.
  INTERPRETATION (R-73): the BF16 output rounding gives about 0.002 to 0.01; a wrong head mapping, scale or mask gives
  values far above 0.05. The rehearsal gives the first real value (Section 9).

## 7. Decision rule (frozen; project heuristics; (A1) changed by A1)
(A1) Per global layer l: med_r(l) is the median over the 4 prompts of r (float32); med_rr(l) and max_rr(l) are the
median and the largest of rr = max(r, r16); q_max(l) is the largest |q'|; nf16(l) is the count of FP16 output values
that are not finite; sat_cmp(l) and sat_std(l) are the counts of saturated values.
- INVALID: V1, V2, V3, V4 or V5 fails (the result names them).
- FAIL (stop the build): (A1) a layer with med_r(l) > 2.
- PASS (the gate opens the build), if every global layer has all five: (A1) med_rr(l) <= 1.25; max_rr(l) <= 2;
  q_max(l) <= 6,550 (a 10 times FP16 margin); nf16(l) = 0; sat_cmp(l) <= sat_std(l).
- INCONCLUSIVE: every other result, with each failed PASS condition named (layer and condition).
- FACT (algebra): PASS and FAIL cannot both hold. (A1, R-73 B3) FACT: K_rot is a part of K and V cannot saturate (RMS
  1 over 512 dims, so |V| <= 22.6), so the compact rows never hold more saturated values than the standard rows; the
  count rule is a sanity check.
- (A1, R-73 F3) INTERPRETATION: if the extra error of the layout is independent of the FP8 error, a ratio of 1.25
  allows an extra error of 0.75 times the FP8 error (the square root of 1.25^2 - 1), and a ratio of 2 allows 1.73
  times. The FAIL uses the float32 ratio only: an FP16-compute problem is a fault of the kernel path, which a scale can
  fix, so it gives INCONCLUSIVE, not FAIL.
- (A1, R-73 B4) Consequences (the user confirms each step; written before any data):

| Outcome | When | Action |
|---|---|---|
| PASS | every PASS condition holds | week 2 of the build starts (D-88) |
| FAIL | a layer with med_r above 2 | stop the build (D-88); I-182 keeps its exact capacity result |
| INCONCLUSIVE (range) | only q_max above 6,550 (above 32,752 is the FP16 range) or nf16 above 0 failed | pre-scale the query (q' times 2^-m with the kernel scale 2^m; the scores stay the same) and test the scale argument in week 2 |
| INCONCLUSIVE (other) | any other failed PASS condition | the user decides; the report lists the layers and the conditions; no rerun without a new S4 |
| INVALID (fail fast) | the capture stopped | fix the cause, then one rerun under a new S4 with a stated cost cap |
| INVALID (V3 only) | the identity or the module w fails | report the recovered w; the user decides |
| INVALID (V5 only) | the reference does not model the engine | fix the reference (head mapping, scale, mask) and analyze the copied captures again on the Mac; no new Pod |
| INVALID (other) | V1, V2, V4, or several checks | fix the cause; a rerun needs a new S4 |

  FACT (R-73; checked by the author): the FlashAttention backend passes the scale to the kernel
  (`softmax_scale=self.scale`, `flash_attn.py` 1413, 1500, 1590, 1622 and 1698). HYPOTHESIS: the FA4 FP8 path accepts
  a scale other than 1.0 (not tested; week 2 tests it).
  The decision text names the action.

## 8. Controls and baselines
- Baseline: the standard FP8 storage (the engine's FP8 KV path with scale 1.0) on the same tensors.
- A/A: not needed (no timing); the spread over the 4 prompts is reported.
- Positive controls: V3 on real data; (A1, R-73 B2) V5 ties the float32 reference to the engine output of the same
  rows; the RoPE check (reported): for each rotated pair the fit of K as the RoPE rotation of (a V_i, b V_j) leaves a
  residual near the BF16 level, and the fit without RoPE does not fit, if and only if the hook point is after RoPE
  (`rope_check`); on the CPU the views path is exact without rounding, and the two errors are similar on well-scaled
  Gemma 4 K=V data.
- (A1, R-73 O2) Sensitivity control (reported): the compact storage in E5M2 gives a ratio near 2 (the stub gives
  about 2), so the harness can see a 2 times storage error.
- Negative controls (CPU tests): broken K=V data fail V3; the wrong form of w fails V3; a capture before RoPE fails the
  RoPE check; (A1) a wrong head mapping fails V5; a missing module weight fails V3.

## 9. Budget
- Price: 1 x H100 SXM Secure at $3.99/h (the E-52 live read on 2026-11-01; S4 reads it again). Cost cap: $3.
- Time model (A1; INTERPRETATION, from the E-50 to E-52 records and the Mac timing):

| Step | Estimate | Source |
|---|---|---|
| Create to SSH | 3.5 min (image on the host) to 11 min (pull); delete the Pod at 15 min | E-52 211 s; E-50 and E-51 about 11 min pulls |
| Upload the code, start the session | under 1 min | E-52 |
| Setup, versions, hardware record | about 1 min | E-52 |
| Download of 33.3 GB | 2 to 4 min (cap 420 s) | E-50: 130 s for this checkpoint |
| Engine start (eager) | 1.5 to 3 min | E-50: 87 s plus 60 s of compile (R-73), which eager mode removes |
| 4 prefills, copies and saves (3.1 GB) | 1 to 2 min (with the engine start: cap 600 s) | the first prefill can compile kernels (no JIT warm-up in eager mode, config/vllm.py 1800) |
| Analysis | 1 to 2 min (cap 240 s) | 3.1 GB read; 2.4 to 4.4 s per prompt and layer on the Mac CPU (test), far less on the H100 |
| (A1, F2) Copy of RES and CAPDIR (3.1 GB) to the Mac | 2 to 5 min, inside the AUTO_STOP window (300 s) | INTERPRETATION (R-73): 3.1 GB at 30 MB/s takes about 100 s |
| Pod time | about 18 to 29 min, $1.20 to $1.95 | |

- Caps (A1, R-73 B1): download 420 s, capture 600 s, analysis 240 s (`DL_TIMEOUT`, `CAP_TIMEOUT`, `AN_TIMEOUT`); the
  steps and the setup add up to about 1,380 s, below the session watchdog of 1,440 s (`POD_LIMIT`). The session starts
  about 140 s after the boot watchdog arms (E-52), so the session watchdog fires at about 1,580 s after boot, before the
  boot watchdog of 1,740 s (`experiments/runpod/pod_cmd.py --cap 1740`). `AUTO_STOP=1` with a 300 s delay (the copy
  window).
- Worst case (A1, R-73 B1): the boot watchdog does not cover the image pull. Rule: the main session deletes the Pod if
  it has no SSH port 15 min after the create call; this deadline sits inside the create-to-SSH wait loop (one tool
  call, RUNPOD.md Section 3), not in a timer. So the worst case is 15 + 29 = 44 min, $2.93, plus the volume disk until
  the delete (about $0.03/h for 100 GB). Balance: S4 runs `experiments/runpod/balance.py --need 3 --hours 0.75` live
  before the booking.
- Order of work (the most decisive data first): the capture of prompt 0 with the fail fast, then prompts 1 to 3, then
  the analysis. The time caps cut the later prompts first; then V2 fails (INVALID), and no partial decision is made.
- (A1, R-73 F2) Keep the raw captures (PROTOCOL.md principle 7). At the end the session writes `capture_sha.txt` (the
  SHA-256 of every file in CAPDIR) and logs the size. In the AUTO_STOP window the main session copies RES and CAPDIR:
  `rsync -rlt -e "ssh -F <config>" <alias>:/workspace/capture_e53/ <repo>/experiments/gemma4_kv_accuracy/results/e53-h100-<date>/capture/`,
  then compares `shasum -a 256` of the copy with `capture_sha.txt`. The folder `.gitignore` keeps `capture/` out of git
  (3.1 GB); the checksum list stays in git. If the window ends before the copy, start the Pod again with 0 GPUs
  (RUNPOD.md Section 6) and finish the copy; if time is short, copy prompt 0 (about 780 MB) and `k_norm.pt` first.
- (A1, R-73 F1) S5 rehearsal (needs the user's approval at S4, as a separate spend, or a waiver logged in the journal
  Section 10):
  - GPU and price: 1 x RTX 4090, Secure Cloud ($0.89/h, the E-52 read on 2026-11-01; S4 reads it again). Cap $0.60.
    Boot watchdog `pod_cmd.py --cap 1500`, `POD_LIMIT=1200`, the same 15 min pull rule: worst case 15 + 25 = 40 min,
    $0.59. Plan about 20 to 25 min, $0.30 to $0.37 (E-50 S5 with the same image and model: 16 min, about $0.24).
  - Model (FACT, the Hugging Face `config.json` at the revision, read 2026-11-01): `google/gemma-4-E2B-it` at revision
    `3e22461f65e89153144f8adb70e3b8c2cc9845a7` (the E-50 S5 model). 35 layers; global layers 4, 9, 14, 19, 24, 29 and 34;
    `num_kv_shared_layers` 20, so only 4, 9 and 14 own a cache; 8 query heads; `num_key_value_heads` 1 and no
    `num_global_key_value_heads`; `global_head_dim` 512; `attention_k_eq_v` false; global RoPE proportional, factor
    0.25, theta 1,000,000 (as the 31B). FACT (transformers 5.20.2, Section 4.3): its global layers have 1 KV head and
    head dim 512. So the GQA ratio (8) and the head dim are the 31B values, and the views layout applies.
  - Session env: `M31=google/gemma-4-E2B-it R31=3e22461f65e89153144f8adb70e3b8c2cc9845a7 POD_LIMIT=1200
    CAP_ARGS="--rehearsal --heads 8 --kv-heads 1 --global-layers 3"`, with `AUTO_STOP=1`.
  - Expected decision: `REHEARSAL INVALID (V1, V3, ...)`: the GPU is not an H100, and E2B has no K=V identity (V3
    fails by design). It cannot test FA4 for head dim 512 (the RTX 4090 runs the Triton backend, E-50 S5), the 31B
    weights or V3.
  - VERIFY items: (1) the image starts and vLLM is 0.33.0rc1.dev229+g3966924d5; (2) `models.txt` has the E2B snapshot
    folder; (3) the engine starts with the E-53 arguments (eager, 8,448 tokens, 4 sequences, utilization 0.85) on
    24 GB; (4) `capture_run.json` has the route "LLM.apply_model", `compilation_mode` "NONE", `cudagraph_mode` "NONE",
    `cache_dtype` "auto", the hooked layers [4, 9, 14] and 3 k_norm weights; (5) prompt 0 has 3 layers and 0 problems,
    and every prompt has `special_after_0` 0; (6) the capture files hold q, k, v and o, with SHA-256 values; (7) the
    analysis runs on CUDA within its cap, and V5 (the engine gap with the Triton BF16 kernel) is at most 0.05: this is
    the first real value of V5; (8) the copy of CAPDIR to the Mac matches `capture_sha.txt`, and its time is
    measured; (9) AUTO_STOP stops the Pod; (10) the times of the prompt builder, the engine start and each step fit the
    H100 caps.
- Run steps for the main session (S6 and S7; A1): create the Pod with the boot watchdog (`pod_cmd.py --cap 1740`), the
  name from `pods.py name E-53 session` and the env `THESIS_TAG=msthesis`, `THESIS_EXPERIMENT=E-53`,
  `HF_HOME=/workspace/cache/hf`, `VLLM_CACHE_ROOT=/workspace/cache/vllm`; (A1, R-73 O5) ask for EUR-NO-2 first
  (HYPOTHESIS: the image can be on the host, as in E-52; not a rule); register the id; wait for SSH inside one tool
  call with the 15 min delete deadline; upload the three folders of Section 11; start `session_e53.sh` with
  `AUTO_STOP=1` in the background; copy RES and CAPDIR in the AUTO_STOP window and compare the checksums; delete the
  Pod. (A1, R-73 O4) In S7, write the ledger manifest from `summary.json` (`ledger_metrics`) and add the
  `HARDWARE_LOG.md` row.

## 10. Threats to validity
- The engine compute dtype: the decision compares the storages with float32 compute and (A1) with the FP16 emulation.
  The stock FA4 FP8 path computes in FP16, and the FA3 path quantizes Q to FP8 (flash_attn.py 1276 to 1296). The FA3
  path is not emulated.
- Prefill queries at 8,192 tokens, not decode at 32K to 64K. HYPOTHESIS: the FP8 error per key does not depend on the
  context length, so the relative output error changes little with more keys. Longer contexts are not tested ((A1) R-73
  O1 is not adopted; Amendments).
- The capture runs with a BF16 KV cache, so every layer sees BF16 attention in the layers before it. The design
  measures the storage error of each layer alone. The growth of the error over the 10 global layers and the effect on
  the model output are not tested (the week-6 end-to-end test of D-88 measures them; (A1) R-73 O3 is not adopted).
- 4 prompts of code and documentation text, not chat. The BOS token is included, so the attention sinks are in the
  data. (A1) Special-token strings in the source text are kept out (Section 4.4).
- Scale 1.0 for K and V (`kv_cache_scheme` null). A calibrated checkpoint needs the two-scale rule of E-52 A1 (O1);
  not tested here.
- (A1) The V5 bound (0.05) is an estimate: nobody checked the rounding inside the FA4 BF16 kernel (R-73). The
  rehearsal gives a Triton value, not an FA4 value.
- (A1) The copy of 3.1 GB can take longer than the 300 s window on a slow link; then a 0-GPU start finishes it
  (Section 9). Its price is not read here.
- Not in E-53: the condition C2 of G1 (the P3 flag log lines and the sliding block-16 timing). It needs another engine
  configuration and a second engine start. The main session decides if it adds it.

## 11. Files
- `capture_e53.py`: the prompts, the engine, the hooks (A1: the output hook), the fail fast, the capture files;
  (A1) the shape flags and `--rehearsal`; `--stub` for the Mac.
- `analyze_e53.py`: the metrics on the GPU (`--capture`), then the frozen checks and rule (summary.json and the
  decision); without `--capture` it decides again from `metrics.json`. (A1) With `--capture` on the copied captures it
  runs on the Mac too (about 2.4 to 4.4 s per prompt and layer).
- `session_e53.sh`: the Pod session (versions, hardware record, telemetry, download, capture, analysis, watchdog,
  AUTO_STOP; `STUB=1`; (A1) `capture_sha.txt`).
- `test_e53.py`: the tests.
- (A1) `.gitignore`: keeps `results/*/capture/` (the copied raw captures) out of git.
- `../gemma4_kv_kernel/compact_attn.py` (unchanged): `rope`, `ROT`, `UNROT`, `pack_views`, `views_of`, `views_query`,
  `unpermute`.
- Upload folders: `gemma4_kv_accuracy`, `gemma4_kv_kernel`, `runpod`.
- Outputs in `RES` (`/workspace/results/e53`, small): `session.log`, `versions.txt`, `env-*/` (telemetry, hardware,
  `models.txt`), `prompts.json`, `capture_run.json`, `capture_log.txt`, `metrics.json`, `analysis_log.txt`,
  `summary.json`, `decision.txt`, (A1) `capture_sha.txt`. In `CAPDIR` (`/workspace/capture_e53`; (A1) copied to the Mac
  as `results/e53-h100-<date>/capture/`): the capture files and `k_norm.pt`.

## 12. Verification record (stages S3 and S5)
- S3 (2026-11-01, Mac, no GPU and no vLLM): `uv run --no-project --with numpy --with torch --with pytest python -m
  pytest -q --rootdir <folder> <folder>/test_e53.py -p no:cacheprovider`: 17 passed in about 100 s (torch 2.15.0).
  FACT from the tests:
  1. The FP8 cast saturates at +-448, sends the tie 2^-10 to 0 and keeps 2^-9 as a subnormal.
  2. The causal GQA reference equals the dense reference of `compact_attn` at the last position and over the prefix.
  3. The views path is exact without rounding (1e-6); with the engine's BF16 rounding of K it agrees to 5e-3.
  4. On Gemma 4 K=V stub data: w is recovered (1e-2), the data choose "weight", "1 + weight" (a module that stores
     w - 1) or the data; the identity error is about 0.0017; the RoPE residual is about 0.0023 after RoPE and about
     1.0 without RoPE; err_standard about 0.049 and the ratio about 1.015 (PASS).
  5. Every decision path: PASS; FAIL by |q'| and by the median ratio; INCONCLUSIVE by the median ratio, the
     largest-error ratio, the saturated share and |q'|; INVALID by V1, V2 (fail fast, a capture error, a plan that is
     not the frozen plan, a missing layer, no files), V3 and V4.
  6. The hooks on a model tree with the vLLM class names and call pattern: only layers 5 and 11 are hooked, the rows
     come in position order over two chunks, the shape checks find a bad chunk.
  7. Timing at the real size (E-38 rule): one prompt and layer (8,192 tokens, 256 queries) takes 4.4 s on the Mac CPU;
     the hook copy of one real-size layer is fast.
  8. The stub session from a clean copy of the upload folders: PASS; a second run skips the capture; `--stub-fail`
     gives INVALID (V2); `--stub-no-tensors` stops the capture with exit 3 and gives INVALID (V2: fail fast).
  Not tested on the Mac: vLLM, the engine start, the hooks inside vLLM, the tokenizer of the snapshot and the GPU.
- (A1) S3 after A1 (2026-11-01, Mac): the same command: 20 passed in about 70 s. New or changed tests (FACT):
  1. B1: the session defaults (420, 600, 240, 1,440, 300 s) keep the step caps inside the session watchdog and the
     session watchdog before the boot watchdog; the worst cases are $2.93 (H100) and $0.59 (rehearsal).
  2. B2: the stub engine output (the BF16 of the float32 attention) gives an engine gap below 0.01 and passes V5; a
     wrong head mapping (KV heads rolled, or query heads rolled with 1 KV head) fails V5 with its action; the output
     hook records the same rows as q on the model tree.
  3. B3: the same saturated K_rot values give equal counts (256 and 256) and pass the count rule, with shares of 1.6
     times on the old support and equal shares on the K_rot support; K_unrot saturation counts only in the standard
     rows; more saturated compact values give INCONCLUSIVE.
  4. B4 and F3: a |q'| above 32,752 gives INCONCLUSIVE "(the FP16 range); action: pre-scale the query", also on the
     real computation with one query value of 50,000; FP16 values that are not finite and a worse FP16 ratio give
     INCONCLUSIVE; the R-73 F3 case (ratio of the maxima 1.5, one prompt at 3 times its own error) is caught by the
     per-prompt ratio; FAIL names the action "stop the build"; every outcome names its action.
  5. F4: a file with the text `<bos>` is skipped and `special_after_0` stays 0; without the filter a second BOS id
     enters the prompt; a prompt with a special id fails V2.
  6. F5: a stub without module weights fails V3; a real-run record without the engine facts fails V2; a rehearsal
     record with compile mode "0" fails V2. F7: an IntEnum mode gives "NONE". F1: the KV-sharing rule gives [4, 9, 14]
     for E2B and the 10 layers for the 31B; the rehearsal shapes (8 heads, 1 KV head, 3 layers) run through the whole
     capture and analysis, and the decision starts with "REHEARSAL".
  7. F2: the session writes `capture_sha.txt` (5 files in the stub); a capture file with another SHA-256 fails V2.
     F6 and O2: the value-path errors of the two storages are equal; the E5M2 ratio is above 1.5.
  8. Timing at the real size with the A1 metrics (11 attention passes): 2.4 s on the Mac CPU.
- S2: R-73 (2026-11-01): AMEND_THEN_RUN; A1 answers it (Amendments).
- S5: (A1) the rehearsal plan is in Section 9.
  - (A2) Rehearsal 1 (2026-11-01, RTX 4090 Pod `kwvva30c4tb2af`, EU-RO-1, about 4 min, D-89): the SSH port 94 s after
    the create; the boot watchdog armed (cap 1,500 s); `RUNPOD_POD_ID` and `RUNPOD_API_KEY` set; vLLM
    0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130, transformers 5.20.2; setup with the E2B download 20 s; 4 prompts
    of 8,192 tokens in 3.5 s, `special_after_0` 0 in each; the engine started in 29.2 s (eager, the E-53 arguments;
    `compilation_mode` and `cudagraph_mode` "NONE", cache "auto", BF16; the log: heterogeneous head dims force the
    TRITON_ATTN backend on sm_89). Then the capture stopped at the run record (`capture_e53.py` line 560):
    transformers 5.20.2 raises `AmbiguousGlobalPerLayerAttributeError`, which is not an AttributeError, for
    `config.head_dim`, so `getattr` with a default does not catch it. The analysis and the decision ran on the empty
    capture: `REHEARSAL INVALID (V1, V2, V3, V4, V5)`, as the frozen rule requires. RES copied (29 files, checksums
    match). The main session deleted the Pod in the AUTO_STOP delay. VERIFY items 1, 2, 3 and the engine part of 4
    hold; items 4 (the route and the hooks) to 10 are open.
  - (A2) FACT (the Mac, transformers 5.20.2, the two pinned configs): the 31B config also raises, for
    `num_key_value_heads` and `head_dim`, so the H100 run would have stopped at the same line. `global_head_dim` and
    `num_global_key_value_heads` read as None in this version; `per_layer_config` gives head dim 512 and (31B) 4 KV
    heads at the global layers.
  - (A2) S3 after A2 (Mac): the same command: 21 passed in about 65 s.
  - Rehearsal 2 (2026-11-01, RTX 4090 Pod `y0p9vr8jhy39n1`, EU-RO-1, about 10 min, after A2): S5 PASS. The SSH port
    169 s after the create. The session took 77 s: setup with the E2B download 20 s, the capture 70 s (prompts 4.9 s,
    engine 31.2 s, 4 prefills with the saves about 8.5 s), the analysis 5 s (metrics 3.1 s on CUDA). The VERIFY items:
    (1) vLLM 0.33.0rc1.dev229+g3966924d5; (2) the E2B snapshot in `models.txt`; (3) the engine with the E-53 arguments
    on 24 GB; (4) the route "LLM.apply_model", modes "NONE", cache "auto", the hooked layers [4, 9, 14] and 3 k_norm
    weights; (5) 0 problems in every prompt, `special_after_0` 0; (6) 12 capture files with q and o [256, 8, 512] and k
    and v [8,192, 1, 512] in BF16, and 13 SHA-256 lines; (7) the largest engine gap (V5, the Triton BF16 kernel)
    0.0017; (8) the copy to the Mac matches `capture_sha.txt` (13 files, 248 MB in 46 s); (9) AUTO_STOP stopped the
    Pod at 300 s; (10) every step is far inside the H100 caps. Decision: `REHEARSAL INVALID (V1, V3)`, as expected (V3:
    E2B has no K=V; the identity error is 1.39). V2, V4 and V5 pass. Ledger: `e53-rehearsal2-4090-20261101`.
  - (A3) FACT (rehearsal 2, in the AUTO_STOP window): one rsync stream to the Mac gives 5.8 MB/s, 4 streams 21.4 MB/s,
    and 8 streams of one file each 16.1 MB/s (20 MB files). At one stream, the 3.1 GB of the H100 run need about 9 min,
    which is longer than the 300 s window.

## 13. Result (stages S6 to S8, 2026-11-01)
- Run (S6, FACT): 1 x H100 80GB HBM3, Pod `8ikp2grsi19ieu`, EUR-NO-2, Secure, $3.99/h, D-89. The SSH port 213 s
  after the create (the image was on the host). The session took 420 s: setup with the 33.3 GB download 280 s (cap
  420 s), the capture 117 s (engine 93.5 s, eager; the 4 prefills with the saves about 15 s), the analysis 14 s on
  CUDA (metrics 12.5 s). The copy of RES (29 files) and of the captures (41 files, 3.1 GB) to the Mac took 136 s with
  8 streams (A3); every checksum matches. The main session deleted the Pod in the AUTO_STOP delay. Cost about $0.98
  (estimate; billing pending). Results: `results/e53-h100-20261101/`; ledger `e53-h100-20261101`.
- Validity (FACT): V1 to V5 pass. V1: H100 80GB HBM3. V2: all 10 global layers (5, 11, ..., 59) in every prompt, with
  the planned shapes; the route "LLM.apply_model"; modes "NONE"; a BF16 cache; vLLM 0.33.0rc1.dev229+g3966924d5; 41
  capture files with their SHA-256. V3: the largest identity error 0.0024 (the BF16 level), the module form "weight"
  in every layer. V4: a finite reference. V5: the largest engine gap 0.0017 (bound 0.05).
- Decision (the frozen rule of Section 7): **FAIL** (layer 41: the median per-prompt ratio 2.38 is above 2). The
  action of the consequence table: stop the build (D-88); I-182 keeps its exact capacity result. The user confirms the
  step (journal D-90).
- Per global layer (FACT; the medians over the 4 prompts; ratio = err_compact / err_standard, float32):

| Layer | 5 | 11 | 17 | 23 | 29 | 35 | 41 | 47 | 53 | 59 |
|---|---|---|---|---|---|---|---|---|---|---|
| err_standard | 0.0440 | 0.0264 | 0.0391 | 0.0736 | 0.0345 | 0.0288 | 0.0259 | 0.0693 | 0.0519 | 0.0276 |
| err_compact | 0.0404 | 0.0333 | 0.0469 | 0.0382 | 0.0347 | 0.0333 | 0.0604 | 0.0980 | 0.0513 | 0.0225 |
| median ratio | 0.930 | 1.230 | 1.191 | 0.566 | 1.007 | 1.189 | 2.384 | 1.334 | 0.983 | 0.818 |
| largest ratio | 1.058 | 1.317 | 1.235 | 0.574 | 1.011 | 1.238 | 2.685 | 1.626 | 1.090 | 0.877 |

- The other PASS conditions hold (FACT): the largest |q'| is 19.2 (bound 6,550); no FP16 value is not finite; 0
  saturated values in both storages; the FP16 ratios equal the float32 ratios to 1e-4. The 4 per-prompt ratios of
  layer 41 are 2.69, 2.11, 2.42 and 2.35.
- Mac recompute (FACT): `analyze_e53.py --device cpu` on the copied captures gives the same decision text; the largest
  difference of the per-layer medians against the Pod run is 3.7e-5 (`exploratory/summary_mac_cpu.json`).
- Exploratory (after the frozen decision, on the same captures; not part of the rule; scripts and outputs in
  `exploratory/`):
  1. FACT: the k_norm weight is one constant in each global layer (0.0600 to 0.0654 on all 512 dims). In layer 29 it
     is 0.0625 = 2^-4 exactly. A power-of-2 scale does not move a value on the FP8 grid, so in layer 29 the two
     storages round the same values: the identity error is 0 and the ratio is 1.007.
  2. FACT: the score path carries the difference. In layer 41 the score-path error is 0.050 (compact) against 0.026
     (standard); the value-path errors are 0.015 and 0.014. The rotated dims give the same score error in both
     storages. The unrotated dims give almost all of the score error (probability-weighted RMS 0.05 to 0.16, against
     0.004 to 0.024 on the rotated dims): in layer 41, 0.138 (compact) against 0.096 (standard); in layer 23, 0.090
     against 0.164 (`e53_score_probe.py`).
  3. FACT: the BF16 mismatch of K and w * V is small: the compact path without FP8 has an error of 0 to 0.0094
     against the BF16 reference (0 in layer 29).
  4. FACT: the attention-sink token at position 0 has the same K in all 4 prompts and takes 13% to 19% of the
     attention in layers 5 to 23 (2% to 8% in the others). With position 0 kept in BF16 in both storages, the
     score-path ratio of layer 5 moves from 0.71 to 1.03, of layer 59 from 0.61 to 0.96; in layer 41 (3%) it stays
     at 1.84 (`e53_sink_probe.py`).
  5. FACT (`e53_kfirst_probe.py`; a layout chosen after the FAIL): the mirror row [K | V_rot] (also 640 values; the K
     view is fp8(K) in another order, so its scores equal the standard storage; V_unrot = fp8(K_unrot) / w) gives
     median ratios 1.00 to 1.30 and largest ratios 1.00 to 1.39. Under the frozen rule this is INCONCLUSIVE (3 layers
     above 1.25), not FAIL.
  INTERPRETATION: the compact row rounds the unrotated part of K on the FP8 grid of V, and the standard storage rounds
  K on its own grid. If w is not a power of 2, the two roundings differ, and a few dominant values (the sink token in
  the early layers; HYPOTHESIS: large channels in layer 41) decide which grid wins in each layer. The FAIL is a
  property of the shared rounding with the KV scale 1.0, not a code error. Both storages also leave K (RMS about
  0.065) partly in the FP8 subnormal range; a calibrated KV scale was not tested.

## Amendments
- A1 (2026-11-01, before any data; from R-73, verdict AMEND_THEN_RUN, `dossier/REVIEW_E-53_PRERUN_2026-11-01.md`):
  - B1 (blocker), adopted: the boot watchdog 1,740 s; the caps 420, 600 and 240 s; the session watchdog 1,440 s, which
    now fires before the boot watchdog; the 15 min pull rule as a deadline inside the create-to-SSH wait loop; the
    worst case 44 min, $2.93 (Section 9). Not needed: R-73's alternative of a $3.30 cap.
  - B2 (blocker), adopted: the output hook, `o` in every capture file, the engine gap and V5 (at most 0.05), a stub test
    and a negative test (Sections 4.3, 4.5, 6, 8).
  - B3 (blocker), adopted: the count rule replaces the share rule; the shares over the same support and the K_unrot
    saturation are reported; the test is updated (Sections 5 and 7).
  - B4 (blocker), adopted: the consequence table; a |q'| above 32,752 is INCONCLUSIVE with the action "pre-scale the
    query"; FAIL is the ratio only (Sections 1, 3 and 7).
  - F1, adopted: the S5 rehearsal on an RTX 4090 with `google/gemma-4-E2B-it` (its config read and recorded), the
    shape flags with the frozen defaults, the rule that hooks only the global layers that own a cache, the VERIFY items
    and the cap $0.60 (Sections 4.2, 4.3 and 9). The user approves or waives it at S4. Not adopted: R-73's HYPOTHESIS
    of a 6-layer cut of the 31B config with dummy weights, because it is not tested and E2B already ran this image in
    E-50 S5.
  - F2, adopted: the captures are copied to the Mac in a 300 s AUTO_STOP window with `capture_sha.txt` and a folder
    `.gitignore`; the analysis checks every SHA-256 (V2); the copy time is in the budget (Sections 9 and 11).
  - F3, adopted: per-prompt ratios for the median and the largest tests; the FP16 emulation enters the PASS tests
    (the larger of the two ratios, and no value that is not finite); the quadrature note. One choice of the author:
    FAIL keeps the float32 ratio, because an FP16-compute fault is fixable by a scale (Section 7).
  - F4, adopted: files with special ids are skipped; `special_after_0` must be 0 (V2).
  - F5, adopted: V3 needs the module weight of every global layer and a match of its form; V2 checks the vLLM version,
    the compile and graph modes, a BF16 cache and the planned count of global layers in a real run; the largest
    identity error of one token is reported, not in the rule: R-73 marks it optional, and a bound for one token has no
    calibration yet (the stub gives 0.002; the rehearsal and the run report it).
  - F6, adopted as reported metrics: the score-path and value-path errors.
  - F7, adopted: the mode names (`.name`).
  - F8, adopted: the README text corrections (Sections 4.1, 4.3 and 9).
  - O1, not adopted: a 16K or 32K prompt changes `max_model_len`, adds 1.4 to 2.7 GB of captures and a chunked
    prefill path, and the B1 caps leave little time for it. The context claim stays a threat (Section 10).
  - O2, adopted as a reported control: the compact storage in E5M2 (Section 8).
  - O3, not adopted: the splice test needs new code and its own rehearsal; the week-6 test covers the growth.
  - O4, adopted in the run steps: the ledger manifest from `summary.json` and the hardware log row (Section 9). No new
    adapter file: the main session writes the manifest, as for E-52.
  - O5, adopted as a note: ask for EUR-NO-2 first (a HYPOTHESIS, not a rule).
  - R-73 notes (Answer 6), adopted: V2 now also fails when a capture prompt after prompt 0 had a problem; the prompt
    builder walks each root once.
  - Unchanged: the question's storages and the FP8 cast, the plan (4 prompts, 8,192 tokens, 256 query rows), the
    model and image, the engine arguments, V1, V4, the PASS bounds 1.25, 2 and 6,550, and the FAIL bound 2.
- A2 (2026-11-01, after S5 rehearsal 1, before any data; Section 12): `cfg_value` reads the config values of the run
  record. For a per-layer attribute it records the values of the global layers from `per_layer_config`, and else the
  error name. These values are information only: the rule reads `global_layers`, which the code reads before that
  line. One new test (`test_config_values_survive_per_layer_attributes`). Next: rehearsal 2 under D-89, inside the
  same $0.60 rehearsal cap. Unchanged: the question, the rule, the checks, the caps and the plan.
- A3 (2026-11-01, after S5 rehearsal 2, before any data; Section 12): the main session copies CAPDIR with 8 parallel
  rsync streams, one file each, in path order (prompt 0 first), and then one rsync of the folder for any missing file.
  INTERPRETATION: the 80 MB files of the H100 run make the share of the handshake small, so 3.1 GB take about 1.5 to
  2.5 min, inside the 300 s window. The fallback of Section 9 (start the Pod again and finish the copy) stays.
  Unchanged: the question, the rule, the checks, the caps and the plan.
