# E-53: accuracy of the compact FP8 K=V global cache on real Gemma 4 31B activations (gate G3 of D-88; preregistration)

Status: PREDECLARED 2026-11-01. NOT RUN. The S2 review (rubber duck and a hostile review by a different model) is not
done yet. No spend is approved: D-88 says that G3 needs its own spend approval (S4).
Idea: I-182. Gate: G3 of D-88 (after G1 PASS and G2 PASS, `dossier/GATE_G1_PAGESIZE_I-182_2026-11-01.md` and
`dossier/GATE_G2_UPSTREAM_I-182_2026-11-01.md`). Journal entry: Section 6.1 E-53 (the main session adds it).
Labels: FACT (a source, a log or a computation was read), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
All thresholds are project heuristics unless a source is given.

## 1. Question
On real Gemma 4 31B activations, does the compact FP8 KV storage of the global layers (the E-52 views row
[K_rot | V_unrot | V_rot], quantized to FP8 E4M3 with scale 1.0, read with the views query [q_rot | (q * w)_unrot]
rounded to BF16) give an attention-output error no larger than the standard FP8 KV storage (K and V each quantized to
E4M3 with scale 1.0), and do the views-path values fit FP16 with margin?
"No" is the FAIL of Section 7: a global layer where the median compact error is more than 2 times the standard error,
or a views query value above 32,752 (half the FP16 maximum). A FAIL stops the vLLM prototype (D-88).

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

## 3. Hypotheses
- H1 (primary): PASS (Section 7): in every global layer the compact error stays within 1.25 times (median) and 2
  times (largest) the standard error, |q'| has a 10 times FP16 margin, and the compact rows do not saturate more.
- H0: FAIL (Section 7): in some global layer the median compact error is more than 2 times the standard error, or
  |q'| is above 32,752.

## 4. Setup (frozen)

### 4.1 Hardware, software and model
- Hardware: 1 x H100 80GB HBM3 (SXM), Secure Cloud, on-demand, `minCudaVersion` 13.0. Volume disk 100 GB at
  `/workspace`, container disk 40 GB. Why this GPU: it is the prototype target and the GPU of E-49 to E-52, and the
  rehearsal GPUs cannot hold the model (Section 9).
- Software (FACT, E-49 to E-52): the image `vllm/vllm-openai@sha256:f255c4fad2f30ff5ee80db14d3a03afe9ae0f10d3470339864ff04745fe44cb3`
  (vLLM 0.33.0rc1.dev229+g3966924d5, torch 2.14.0+cu130). The session records the versions.
- Model (FACT, E-50, G2): `RedHatAI/gemma-4-31B-it-FP8-dynamic` at revision `d4ab4f579dd3516f97d8a6a4c98d0653480bad15`
  (33.3 GB). The session downloads it with `snapshot_download(model, revision=...)`, as `session_e50.sh` does.
- Model facts (FACT, G1 and G2): 60 layers; 10 global layers (5, 11, ..., 59) with 32 query heads, 4 KV heads and
  head_dim 512; attention scale 1.0; no soft-cap. The capture reads the global layers from the config
  (`layer_types == "full_attention"`) and checks them.

### 4.2 Engine (`capture_e53.py`, `ENGINE`)
- `LLM(model=<id>, revision=<sha>, seed=0, enforce_eager=True, kv_cache_dtype="auto" (BF16), max_model_len=8448,
  max_num_batched_tokens=8448, max_num_seqs=4, gpu_memory_utilization=0.85, enable_prefix_caching=False,
  limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})`, in-process with `VLLM_ENABLE_V1_MULTIPROCESSING=0`.
  The id, the revision and text-only mode are as E-50.
- One `generate` call per prompt, with `max_tokens=1` and temperature 0. `max_num_batched_tokens` is larger than the
  prompt, so each prompt is one prefill step (INTERPRETATION; the hooks also handle chunks by position).

### 4.3 How the hooks reach the model (FACT, vLLM 3966924d5, read with `gh api` on 2026-11-01)
| Source (file: lines) | What it shows |
|---|---|
| `vllm/envs.py`: 1448 to 1450 | `VLLM_ENABLE_V1_MULTIPROCESSING` is 1 if not set; the session sets 0 |
| `vllm/entrypoints/llm.py`: 341, 604 to 614 | `LLM` builds `LLMEngine.from_engine_args`; `LLM.apply_model(func)` calls `llm_engine.apply_model` |
| `vllm/v1/engine/llm_engine.py`: 186 to 188, 116 to 123, 136 to 138 | multiprocessing only if the variable is set; `EngineCoreClient.make_client(multiprocess_mode=...)`; in-process, `model_executor = engine_core.engine_core.model_executor` |
| `vllm/v1/engine/llm_engine.py`: 503 to 504, 506 to 509 | `apply_model` sends `collective_rpc("apply_model", args=(func,))`; the attribute path `driver_worker.model_runner.model` (the fallback of the capture) |
| `vllm/v1/engine/core_client.py`: 376 to 401, 498 to 505 | `InprocClient` builds `EngineCore` in this process; its `collective_rpc` calls the engine core |
| `vllm/v1/engine/core.py`: 1152 to 1159 | `EngineCore.collective_rpc` calls `model_executor.collective_rpc` |
| `vllm/v1/executor/uniproc_executor.py`: 54, 90 to 103 | `driver_worker = WorkerWrapperBase`; `run_method(self.driver_worker, method, args, kwargs)` in this process |
| `vllm/v1/serial_utils.py`: 485 to 507 | `run_method` finds a method given by name with `getattr` |
| `vllm/v1/worker/worker_base.py`: 165 to 167, 370 to 371 | `apply_model` returns `fn(self.get_model())`; the wrapper sends unknown attributes to the worker |
| `vllm/v1/worker/gpu_worker.py`: 1164 to 1165; `gpu_model_runner.py`: 3078 to 3086 | `get_model` returns the model (unwrapped from a CUDA graph wrapper) |
| `vllm/config/vllm.py`: 1782 to 1784, 1800 | `enforce_eager` sets `CompilationMode.NONE` and `CUDAGraphMode.NONE`, and turns off the JIT warm-up |
| `vllm/model_executor/models/gemma4_mm.py`: 1145 to 1150 | the language model has the prefix `language_model` |
| `vllm/model_executor/models/gemma4.py`: 419, 451, 453 to 455, 468 to 470, 513 to 518, 520 to 538 | scale 1.0; `k_norm = RMSNorm` (weighted); `v_norm` with `has_weight=False`; `is_sliding` from `config.layer_types`; `get_rope(..., is_neox_style=True)`; `self.attn = Attention(...)` |
| `vllm/model_executor/models/gemma4.py`: 540 to 575, 745 to 749 | `forward(positions, hidden_states)`: `k_norm` (565), `rotary_emb(positions, q, k)` (567), `v_norm` (570), then `self.attn(q, k, v)` (572); the decoder layer calls `self_attn(positions=..., hidden_states=...)` with keywords |
| `vllm/model_executor/layers/attention/attention.py`: 333 to 339, 490 to 497, 538 to 543 | `layer_name`, `num_heads`, `head_size`, `num_kv_heads`, `sliding_window`; `forward(query, key, value, ...)`; the inputs become [tokens, heads, head_size] |
| `vllm/model_executor/layers/attention/attention.py`: 136 to 152, 474 to 484 | the default K and V scales are 1.0; the query is quantized only with an FP8 KV cache (not in this capture) |
| `vllm/model_executor/layers/layernorm.py`: 39 to 44, 145 to 150 | `RMSNorm` computes w * x / rms; `GemmaRMSNorm` (1 + w) is not the class of `gemma4.py` |
| `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py`: 58 to 77 | the frequencies use the head size 512 as the denominator and are 0 for the pairs 64 to 255 |

The hooks (`capture_e53.py`, `register` and `Store`):
- A forward pre-hook with kwargs on each `Attention` module whose parent is a `Gemma4Attention` with `is_sliding`
  False and no KV sharing. These are the global layers only. The hook gets (query, key, value) after RoPE.
- A forward pre-hook with kwargs on that parent gets the `positions` of the same rows.
- Rows kept: K and V at every position below the prompt length; Q at the last 256 positions. The hook copies the rows
  to the CPU at once. After each prompt the script saves BF16 q, k, v and the positions to `CAPDIR/p<i>/L<layer>.pt`
  and records the SHA-256 of each file. It saves the k_norm weight of each global layer to `CAPDIR/k_norm.pt`.
- Fail fast, before the first prompt: the hooked layers must be the global layers of the config, each with 32 heads,
  4 KV heads, head size 512 and no sliding window. Fail fast, after the first prompt: every global layer must have q
  [256, 32, 512] and k and v [8,192, 4, 512] in BF16, the key positions 0 to 8,191 once each, and the last 256 query
  positions. If a check fails, the capture stops at once (status "fail_fast", exit 3).

### 4.4 Prompts (`capture_e53.py`, `build_prompts`)
- 4 prompts of exactly 8,192 token ids: BOS, then the tokens of text files of the image (the tokenizer of the pinned
  snapshot, `add_special_tokens=False`), at most 3,000 tokens per file, with the tokens of a blank line after each file.
- The file order of prompt p comes from `random.Random(53 x 1,000 + p)`. Families: p0 the vllm Python files, p1 the
  torch Python files, p2 the Markdown and reStructuredText files of the whole site-packages folder, p3 all of them. A
  usable file has 2 KB to 200 KB and is not a license, notice, authors or change-log file. If a family runs out, the
  prompt continues with the other files.
- `prompts.json` keeps the token ids, their SHA-256 (little-endian int32 values), and the path and SHA-256 of each
  file. A rerun with the same plan uses the same prompts.

### 4.5 Analysis (`analyze_e53.py`; on the Pod GPU after the capture; float32 with TF32 off)
- Per prompt and global layer: q [256, 32, 512], k and v [8,192, 4, 512], all BF16 as recorded.
- Attention: causal (the query at position p reads the keys 0 to p), scale 1.0, query head h reads KV head h // 8.
  - (a) Reference: the BF16 K and V (the BF16 KV cache holds exactly these values).
  - (b) Standard FP8: fp8(K) and fp8(V), read with the BF16 query.
  - (c) Compact: row = [K_rot | V_unrot | V_rot] (`compact_attn.pack_views`), fp8(row), K view = values 0 to 511, V
    view = values 128 to 639 (`views_of`), query q' = BF16([q_rot | (q * w)_unrot]) with w in BF16 (`views_query`),
    output in the natural order (`unpermute`).
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
  to `w_data` is the w of the views query: the data decide. Without a module weight, the views query uses `w_data`.
  HYPOTHESIS: the data choose "weight" (layernorm.py 39 to 44).

## 5. Metrics
- Primary, per prompt and layer: err_standard and err_compact, the relative L2 error ||O - O_ref|| / ||O_ref|| over
  the 256 query rows, 32 heads and 512 dims. Lower is better. Also their ratio.
- Per layer (the decision): the median and the largest over the 4 prompts of each error and their ratios; the largest
  |q'|; the saturated share (|x| > 448 before the cast) of the stored values, pooled over the prompts (1,024 values per
  token and KV head for the standard rows, 640 for the compact rows).
- Reported only: the subnormal and flush-to-zero shares; the largest |q|; the largest |K view| and |V view| before and
  after the cast (after the cast at most 448, so K' and V' always fit FP16: FACT from the E4M3 range); the identity
  error with each form of w; the RoPE check (Section 8); the largest error of one head. Exploratory: err_compact with
  `w_data`; an FP16-compute emulation (q, q' and P in FP16) with its count of values that are not finite.
- Minimum effect that matters: the bounds of Section 7 (no speed metric).
- Repeats: 4 prompts for each of the 10 global layers. The computation is deterministic on fixed tensors.

## 6. Validity checks (a failed check makes the run INVALID)
- V1: the GPU of the capture is H100 80GB.
- V2: every global layer of every prompt is recorded with the expected shapes. In detail: the capture status is "ok";
  the plan is the frozen plan (4 prompts, 8,192 tokens, 256 query rows); the hooked layers are the global layers of
  the config; every prompt has 8,192 tokens; every prompt and global layer has metrics with q [256, 32, 512], k and v
  [8,192, 4, 512] in BF16 and the planned positions. The fail fast of the capture is a V2 failure.
- V3: the identity holds: for every prompt and layer, ||K_unrot - w * V_unrot|| / ||K_unrot|| <= 1e-2 (the BF16
  level), with the w that the data chose.
- V4: the reference is finite for every prompt and layer.

## 7. Decision rule (frozen; project heuristics)
Per global layer l: med_std(l) and med_cmp(l) are the medians over the 4 prompts of err_standard and err_compact;
max_std(l) and max_cmp(l) are the largest; q_max(l) is the largest |q'|; sat_std(l) and sat_cmp(l) are the pooled
saturated shares.
- INVALID: V1, V2, V3 or V4 fails (the result names them).
- FAIL (stop the build): a layer with med_cmp(l) > 2 x med_std(l), or the largest |q'| over all layers and prompts
  above 65,504 / 2 = 32,752.
- PASS (the gate opens the build), if every global layer has all four: med_cmp(l) <= 1.25 x med_std(l);
  max_cmp(l) <= 2 x max_std(l); q_max(l) <= 6,550 (a 10 times FP16 margin); sat_cmp(l) <= sat_std(l) + 0.001 (0.1
  percentage points).
- INCONCLUSIVE: every other result, with each failed PASS condition named (layer and condition).
- FACT (algebra): PASS and FAIL cannot both hold. INTERPRETATION: the compact rows have 640 values per token and head,
  so the same saturated K_rot values give a share 1.6 times larger than in the standard rows (a test shows it).

## 8. Controls and baselines
- Baseline: the standard FP8 storage (the engine's FP8 KV path with scale 1.0) on the same tensors.
- A/A: not needed (no timing); the spread over the 4 prompts is reported.
- Positive controls: V3 on real data; the RoPE check (reported): for each rotated pair the fit of K as the RoPE
  rotation of (a V_i, b V_j) leaves a residual near the BF16 level, and the fit without RoPE does not fit, if and only if
  the hook point is after RoPE (`rope_check`); on the CPU the views path is exact without rounding, and the two errors
  are similar on well-scaled Gemma 4 K=V data.
- Negative controls (CPU tests): broken K=V data fail V3; the wrong form of w fails V3; a capture before RoPE fails the
  RoPE check.

## 9. Budget
- Price: 1 x H100 SXM Secure at $3.99/h (the E-52 live read on 2026-11-01; S4 reads it again). Cost cap: $3.
- Time model (INTERPRETATION, from the E-50 to E-52 records and the Mac timing):

| Step | Estimate | Source |
|---|---|---|
| Create to SSH | 3.5 min (image on the host) to 11 min (pull) | E-52 211 s; E-50 and E-51 about 11 min pulls |
| Upload the code, start the session | under 1 min | E-52 |
| Setup, versions, hardware record | about 1 min | E-52 |
| Download of 33.3 GB | 2 to 4 min | E-50: 130 s for this checkpoint |
| Engine start (eager) | 1.5 to 3 min | E-50: 122 to 165 s with compile and CUDA graphs; eager mode skips both |
| 4 prefills, copies and saves (3.1 GB) | 1 to 2 min | the first prefill can compile kernels (no JIT warm-up in eager mode, config/vllm.py 1800) |
| Analysis | 1 to 2 min | 3.1 GB read; 4.4 s per prompt and layer on the Mac CPU (test), far less on the H100 |
| AUTO_STOP delay | 3 min | copy the results in this time |
| Pod time | about 15 to 30 min, $1.00 to $2.00 | |

- Caps: download 600 s, capture 900 s, analysis 300 s (`DL_TIMEOUT`, `CAP_TIMEOUT`, `AN_TIMEOUT`); session watchdog
  1,980 s (`POD_LIMIT`); boot watchdog CAP 2,040 s from the container start (`experiments/runpod/pod_cmd.py --cap
  2040`, as E-52); `AUTO_STOP=1` with a 180 s delay. The step caps add up to 1,800 s, below the session watchdog.
- Worst case: with an 11 min pull, 11 + 34 = 45 min, $3.00. The boot watchdog does not cover the image pull, so the
  main session deletes the Pod if the image pull takes more than 15 min. With a pull of exactly 15 min the worst case
  is 49 min, $3.26 (open risk, Section 10).
- Order of work (the most decisive data first): the capture of prompt 0 with the fail fast, then prompts 1 to 3, then
  the analysis. The time caps cut the later prompts first; then V2 fails (INVALID), and no partial decision is made.
- Balance: S4 runs `experiments/runpod/balance.py --need 3 --hours 0.75` live before the booking.
- No GPU rehearsal (stage S5), and why: FACT: the rehearsal GPUs of PROTOCOL.md (L4, RTX 4090) have 24 GB, and the
  checkpoint has 33.3 GB (G2), so neither can load it. The small Gemma 4 models that fit do not have
  `attention_k_eq_v` (G2 Section 3.5), so they cannot test the identity. INTERPRETATION: a 48 GB GPU (for example an
  L40S) could hold the checkpoint, but it runs another attention backend (FA4 head_dim 512 needs SM90) and needs the
  same download and engine start. This design does not use it; the user can choose it at S4.
- What replaces the rehearsal: (1) the stub session (`STUB=1`) from a clean copy of the upload folders; (2) the CPU
  tests, which include the hooks on a model tree with the vLLM class names and call pattern; (3) the fail fast before
  and after the first prompt, so a hook fault costs only the download and the engine start (about 6 to 9 min, about
  $0.40 to $0.60). (4) VERIFY items that the main session reads in the first minutes of the run: "setup done" and the
  `models.txt` line with the snapshot folder; `capture_log.txt` with `{"prompt": 0, "layers": 10, "problems": 0, ...}`;
  "capture: exit 0" in `session.log`; `capture_run.json` with `engine.compilation_mode` NONE and the route
  "LLM.apply_model".
- Run steps for the main session (S6): create the Pod with the boot watchdog (`pod_cmd.py --cap 2040`), the name from
  `pods.py name E-53 session` and the env `THESIS_TAG=msthesis`, `THESIS_EXPERIMENT=E-53`,
  `HF_HOME=/workspace/cache/hf`, `VLLM_CACHE_ROOT=/workspace/cache/vllm`; register the id; upload the three folders of
  Section 11; start `session_e53.sh` with `AUTO_STOP=1` in the background; copy only `RES` (`/workspace/results/e53`);
  check the checksums; delete the Pod.

## 10. Threats to validity
- The engine compute dtype is not modeled in the decision: the analysis compares the storages with float32 compute.
  The stock FA4 FP8 path computes in FP16, and the FA3 path quantizes Q to FP8 (flash_attn.py 1276 to 1296). The FP16
  emulation is reported only.
- Prefill queries at 8,192 tokens, not decode at 32K to 64K. HYPOTHESIS: the FP8 error per key does not depend on the
  context length, so the relative output error changes little with more keys. Longer contexts are not tested.
- The capture runs with a BF16 KV cache, so every layer sees BF16 attention in the layers before it. The design
  measures the storage error of each layer alone. The growth of the error over the 10 global layers and the effect on
  the model output are not tested (the week-6 end-to-end test of D-88 measures them).
- 4 prompts of code and documentation text, not chat. The BOS token is included, so the attention sinks are in the
  data.
- Scale 1.0 for K and V (`kv_cache_scheme` null). A calibrated checkpoint needs the two-scale rule of E-52 A1 (O1);
  not tested here.
- The raw captures (about 3.1 GB) are not copied off the Pod, as the brief asks; they go with the Pod. To keep the run
  repeatable, `prompts.json` keeps the token ids and `capture_run.json` keeps the SHA-256 of every capture file. The
  main session can copy them if it wants to keep them.
- Not in E-53: the condition C2 of G1 (the P3 flag log lines and the sliding block-16 timing). It needs another engine
  configuration and a second engine start. The main session decides if it adds it.
- The worst-case cost with a 15 min pull is $0.26 above the cap (Section 9).

## 11. Files
- `capture_e53.py`: the prompts, the engine, the hooks, the fail fast, the capture files; `--stub` for the Mac.
- `analyze_e53.py`: the metrics on the GPU (`--capture`), then the frozen checks and rule (summary.json and the
  decision); without `--capture` it decides again from `metrics.json`.
- `session_e53.sh`: the Pod session (versions, hardware record, telemetry, download, capture, analysis, watchdog,
  AUTO_STOP; `STUB=1`).
- `test_e53.py`: the tests.
- `../gemma4_kv_kernel/compact_attn.py` (unchanged): `rope`, `ROT`, `UNROT`, `pack_views`, `views_of`, `views_query`,
  `unpermute`.
- Upload folders: `gemma4_kv_accuracy`, `gemma4_kv_kernel`, `runpod`.
- Outputs in `RES` (`/workspace/results/e53`, small): `session.log`, `versions.txt`, `env-*/` (telemetry, hardware,
  `models.txt`), `prompts.json`, `capture_run.json`, `capture_log.txt`, `metrics.json`, `analysis_log.txt`,
  `summary.json`, `decision.txt`. In `CAPDIR` (`/workspace/capture_e53`, not copied): the capture files.

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
- S2: not done.
- S5: no rehearsal (Section 9).

## Amendments
None.
