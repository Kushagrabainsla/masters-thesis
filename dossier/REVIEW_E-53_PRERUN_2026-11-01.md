# R-73: pre-run review of E-53 (accuracy of the compact FP8 K=V global cache on real Gemma 4 31B activations; gate G3 of D-88)

VERDICT: AMEND_THEN_RUN (4 blockers, B1 to B4; the hook path, the FP8 cast and the tests hold)

Date: 2026-11-01. Reviewer: R-73 (Sonnet 5.5, a different model from the author). Stage: S2 hostile review of the design at commit 6c22626.
Labels: FACT (I read a file, ran code or computed a number), INTERPRETATION (my reasoning), HYPOTHESIS (not tested).
Rules kept: I changed no repo file except this one. I did not read `.env`. I made no Runpod or cloud call. I posted nothing.
I made read-only `gh api` GET calls to public vLLM source and read-only Hugging Face config reads (the pinned 31B config, its tokenizer config, the E-50 E2B config). I ran no GPU code and no vLLM code.
FACT: The working tree of `experiments/gemma4_kv_accuracy/` equals commit 6c22626 (`git diff HEAD` is empty). The test run left no `__pycache__` or other file in the repo.
Deviation: four early Bash calls chained `cd` with another command by mistake. All commands used absolute paths, so the results are not affected.
vLLM source: I fetched the files that the README cites at 3966924d5 into `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/r73/v39/`.
Scratch scripts (they import the author's code and write nothing to the repo): `.../scratchpad/r73/explore1.py`, `explore2.py`, `explore3.py`.

## Key findings
1. FACT: Every source cite that I checked in README Sections 4.3 and 4.5 matches the file at 3966924d5 (within 2 lines). The hook path works as the README says (Answer 1).
2. FACT: The FP8 cast emulation matches the cache write: BF16 to float, divide by the scale, `__nv_cvt_float_to_fp8` with `__NV_SATFINITE` and `__NV_E4M3` (Answer 2).
3. FACT: The 17 tests pass (122.6 s on the Mac).
4. FACT: The worst case is above the cost cap, and the session watchdog fires after the boot watchdog, not before it (B1).
5. FACT: No check tests the content of q, and nothing compares the analysis reference with the engine output (B2).
6. FACT: The saturated-share condition fails a layout that saturates the same values as the standard storage. I reproduced it (B3).
7. FACT: README Section 7 gives no action for INCONCLUSIVE or INVALID (B4).
8. INTERPRETATION: PASS is the likely result. The V error is the same in both storages, and the score noise is equal for normal-range values. On synthetic data the error ratio is 0.90 to 1.07 for log-normal w with sigma 0.3 and 1.0 and logit spread 2 to 170. The gate adds real information through V3, the range of q', the saturation and the true error level.
9. FACT: The reason to skip the rehearsal holds for the 31B weights only. The E-50 rehearsal on an RTX 4090 with Gemma 4 E2B cost about $0.24 (F1).
10. FACT: The raw captures leave with the Pod, so nobody can recompute or reuse them (F2).

## Blockers

### B1. The worst case is above the cost cap, and the watchdog order in the script is wrong
- Where: `README.md` lines 196 to 201 and 241; `session_e53.sh` lines 4 to 5 and 18 to 22.
- FACT: README Section 9 gives the worst case as 15 min of pull plus 34 min of boot watchdog. That is 49 min and $3.26. The cost cap is $3.00. The S4 balance check uses 0.75 h.
- FACT: In E-52 the boot watchdog armed at 10:07:18 (`boot_watchdog.log`). The session started at 10:09:36 (`session.log`: setup done at 10:09:47 after 11 s). So the session starts about 2.3 min after the boot watchdog.
- FACT: The session watchdog (POD_LIMIT 1,980 s) then fires about 78 s after the boot watchdog (2,040 s). The comment in `session_e53.sh` line 21 says the opposite.
- FACT: E-50 ran 43 min and cost $2.92 (E-50 README Section 13). Real runs come close to the cap.
- Failure scenario: the pull takes 15 min and a step hangs. The Pod runs 49 min. The cap breaks by $0.26, and abort rule 2 of PROTOCOL.md can act only after the cost exists.
- Fix:
  1. Set the boot watchdog to `pod_cmd.py --cap 1740` (29 min).
  2. Keep the 15 min pull rule. Make it a hard deadline inside the create-to-SSH wait loop (one tool call, RUNPOD.md Section 3). The main session then does not depend on a timer or a notification.
  3. Set `DL_TIMEOUT` 420 s, `CAP_TIMEOUT` 600 s, `AN_TIMEOUT` 240 s and `POD_LIMIT` 1,440 s. The session watchdog then fires at about 1,580 s after boot, before the boot watchdog.
  4. Write the worst case as 15 + 29 = 44 min, $2.93 plus disk (under $0.05). Keep `balance.py --need 3 --hours 0.75`.
- INTERPRETATION: A typical run is 18 to 26 min ($1.2 to $1.7). The new step caps keep 1.7 to 3.2 times margin on the E-50 and E-52 timings.
- FACT: The E-50 download took 130 s. The E-50 engine start took 87 s, and 60 s of that was compile, which eager mode removes.
- Alternative: the user approves a cost cap of $3.30 at S4 and keeps the old caps. That is the user's decision.

### B2. No control ties the analysis reference to the engine
- Where: `capture_e53.py` lines 176 to 202 (`attn_hook`) and 216 to 244 (`register`); `analyze_e53.py` lines 199 to 218 (`layer_metrics`); README Sections 6 and 8.
- FACT: The checks are shapes, dtype, positions, V3 (K against V), V4 (a finite reference) and a reported RoPE fit. No check tests the content of q. No check compares `attend()` with what the engine computed.
- Failure scenario: the recorded q, the head mapping, the scale or the mask differs from the engine. Every check passes, because the analysis is consistent with itself. The ratio gate then describes a model that is not Gemma 4. A FAIL stops a 6 week build. A PASS opens $20 to $40 of GPU spend.
- Fix:
  1. Register a forward hook on the same `Attention` module. Copy `output[qsel]` for the same 256 rows. The module returns `[tokens, 32 x 512]` (vLLM `attention.py` line 584). Keep `qsel` per layer in `Store`.
  2. Save it as `o` beside q, k and v (8.4 MB per layer).
  3. In `layer_metrics` compute `rel(ref, o_engine)`. Add V5: this value is at most 0.05 for every prompt and layer. The author can tighten the bound after the rehearsal (F1).
  4. Add a stub test (engine output = BF16 of `attend`) and a negative test (a swapped head mapping must fail V5).
- INTERPRETATION: The BF16 output rounding gives about 0.002 to 0.01. A wrong GQA mapping, scale or mask gives values far above 0.05. The extra Pod time is zero.

### B3. The saturated-share condition fails a layout that is not worse
- Where: `analyze_e53.py` lines 301 to 302 and 320; README Section 7 (the fourth PASS condition) and lines 167 to 168.
- FACT: K_rot is a subset of K. V cannot saturate: its RMS is 1, so the largest value for 512 dims is about 22.6. So the compact rows never hold more saturated values than the standard rows.
- FACT: The share divides by 640 values for the compact rows and by 1,024 for the standard rows. I ran the author's `layer_metrics` and `layer_summary` with 2 saturated K_rot dims on every token (`explore3.py`). The counts are equal (8,192 and 8,192). The shares are 0.312% and 0.195%. The error ratio is 0.979. `pass["sat"]` is False.
- Failure scenario: Gemma 4 has K_rot values above 448 on 2 or more dims per token and head (HYPOTHESIS: the k_norm weights are not known). The standard storage loses the same values. The run gives INCONCLUSIVE for a layout that is at least as good.
- Fix:
  1. Replace the condition by a count rule: `sat_cmp_count <= sat_std_count`. It holds by construction, so it acts as a sanity check.
  2. Report the shares over the same support (the 128 K_rot values per token and head in both storages).
  3. Report the K_unrot saturation of the standard storage. The compact layout avoids it.
  4. Update `test_saturation_is_counted` and README Section 7 lines 164 to 168.

### B4. INCONCLUSIVE and INVALID have no action, and the FAIL by |q'| is not a real kill
- Where: README Section 1 (lines 15 to 17) and Section 7 (lines 156 to 168); PROTOCOL.md stage S1 ("the decision rule names GO, PIVOT and KILL").
- FACT: README says what PASS and FAIL do. It says nothing about INCONCLUSIVE (a ratio of 1.25 to 2, or a failed condition) or INVALID.
- Failure scenario: a layer has ratio 1.4. The main session must choose after it sees the data. That is a rule change after the data (PROTOCOL.md principle 1).
- FACT: The FlashAttention backend passes the scale to the kernel (`softmax_scale=self.scale`, `flash_attn.py` lines 1077, 1413 and 1500). Gemma 4 uses 1.0.
- INTERPRETATION: A query above 32,752 can be scaled by 2^-m while the kernel scale is set to 2^m. The scores stay the same. So the FP16 range of q' is a fixable condition, not a proof that the layout fails.
- HYPOTHESIS: The FA4 FP8 path accepts a scale other than 1.0. No test exists.
- Fix (the user decides):
  1. PASS: week 2 starts.
  2. FAIL by the ratio (a layer above 2): stop the build, as D-88 says.
  3. |q'| above 32,752: INCONCLUSIVE with the action "pre-scale q' and test the scale argument in week 2".
  4. Other INCONCLUSIVE results: the user decides. The report lists the layers and the failed conditions. No rerun without a new S4.
  5. INVALID by fail-fast: fix the cause, then one rerun under a new S4 with a stated cost cap. INVALID by V3: report the recovered w and let the user decide.

## Fixes

### F1. Do the S5 rehearsal, or get an explicit waiver at S4
- Where: README Section 9 lines 205 to 216; PROTOCOL.md S5; `session_e53.sh` lines 16 and 17.
- FACT: No line of the real path has run. This covers the engine start with these arguments, `apply_model`, the hooks, the real tokenizer, the prompt builder on the image, `torch.load` on CUDA and the analysis on CUDA. The stub and the CPU tests use a fake model tree and a byte tokenizer.
- FACT: The engine arguments differ from E-49 and E-50 in 5 of 7 settings. E-53 uses eager mode, an 8,192-token step, `max_num_seqs` 4, utilization 0.85 and `max_model_len` 8,448.
- FACT: E-49 and E-50 used compile and graphs, 2,048 tokens per step, 32 sequences, utilization 0.92 and a model length of 66,048 (`decode_probe.py` lines 568 to 572).
- FACT: The E-50 rehearsal ran the same image on an RTX 4090 with `google/gemma-4-E2B-it` (revision 3e22461f). It took 16 min and cost about $0.24 (E-50 README Section 12). `session_e53.sh` already reads `M31` and `R31` from the environment.
- FACT: E2B has 8 query heads and 1 KV head. Its global layers are 4, 9, 14, 19, 24, 29 and 34, and head_dim is 512. Only 4, 9 and 14 own a cache (`num_kv_shared_layers` 20). `attention_k_eq_v` is False.
- Failure scenario: the engine start fails with these arguments, or a hook finds no layer. The Pod bills the pull, the download and the engine start (about $0.6 to $1.0). The run then needs a new S4.
- Fix: add two flags with the frozen values as defaults. (1) The shape constants (`capture_e53.py` lines 50 and 252; `analyze_e53.py` lines 41 and 353). (2) `expected` without the KV-shared layers (`capture_e53.py` line 455). Then run the rehearsal.
- INTERPRETATION: The rehearsal covers the engine start, the hooks, the one-step 8K prefill, the saves and the GPU analysis. It also gives the first value of V5. It cannot test FA4, the real weights or V3. The expected decision is INVALID (V1, V3), as in the E-52 rehearsal.
- INTERPRETATION: The cost is about $0.25 to $0.40. The fail-fast without a rehearsal costs about $0.6 and a new approval. The rehearsal changes no frozen rule.
- HYPOTHESIS: A 6-layer cut of the 31B config with `load_format="dummy"` on an RTX 4090 also fits and keeps the real shapes. I did not test it.
- If the user waives S5, record the waiver in the journal Section 10.

### F2. Keep the raw captures
- Where: README Section 10 lines 236 to 238; PROTOCOL.md principle 7.
- FACT: The captures (about 3.1 GB) leave with the Pod. Only `metrics.json` comes back.
- Failure scenario: the result is INCONCLUSIVE and needs a second analysis. A new Pod costs at least $1.5. The post-run reviewer cannot recompute any number, as R-72 did for E-52. The kernel tests of weeks 3 to 5 cannot use real activations.
- Fix: copy `CAPDIR` to the Mac inside the AUTO_STOP window. Raise `AUTO_STOP_DELAY` to 300 s.
- INTERPRETATION: 3.1 GB at 30 MB/s takes 100 s. The longer delay adds about $0.13.
- If time is short, copy prompt 0 (755 MB) and `k_norm.pt`. Compare the SHA-256 values in `capture_run.json` with the copied files. Add the same check to `compute_metrics`.

### F3. Use per-prompt ratios and the FP16 emulation in the rule
- Where: `analyze_e53.py` lines 299, 304 (`ratio_max`), 239 to 242 and 316 to 317.
- FACT: `ratio_max` is the largest compact error divided by the largest standard error. The two can come from different prompts.
- Failure scenario: one prompt has a compact error 3 times its own standard error. Another prompt has a larger standard error. The ratio of the maxima stays below 2, and the layout passes.
- Fix: use r_p = err_cmp / err_std for each prompt. Require median(r_p) at most 1.25 and max(r_p) at most 2.
- FACT: The stock FA4 FP8 path computes in FP16 (E-52 README Section 2). The FP16 emulation is only exploratory now.
- Fix: add `nonfinite_fp16 == 0` as a PASS condition. Report the FP16 ratio next to the float32 ratio. Use the larger of the two ratios for the PASS test.
- INTERPRETATION: A ratio of 1.25 allows an extra layout error of 0.75 times the FP8 error when the errors are independent (sqrt(1.25^2 - 1)). A ratio of 2 allows 1.73 times. The README should say this.

### F4. Special-token strings in the source text
- Where: `capture_e53.py` line 139 and lines 403 to 408.
- FACT: The pinned `tokenizer_config.json` lists special tokens such as `<bos>`, `<eos>`, `<pad>`, `<unk>`, `<|image|>`, `<|turn>` and `<|tool_call>`. GitHub code search (2026-11-01) finds these strings in `vllm/*.py`: `<bos>` in 3 files, `<eos>` in 2, `<pad>` in 1, `<unk>` in 5, `<|image|>` in 11 and `<|turn>` in 2. These are lower bounds.
- Failure scenario: prompt 2 holds a `<bos>` string after 4,000 tokens. The fast tokenizer makes it the id 2. The later rows then attend to a second sink that real text does not have.
- INTERPRETATION: A `<bos>` or `<eos>` in the middle of a prompt acts as a second attention sink. An image id can start a multimodal path. The byte tokenizer of the stub cannot show this.
- Fix: after `encode`, skip any file whose ids contain a special id (`tok.all_special_ids`). Or use the split-special-tokens option. Record `special_ids_in_prompt` in `prompts.json` and require 0 after position 0 in V2.

### F5. Add validity checks that remove silent paths
- Where: `analyze_e53.py` lines 363 to 400 (`summarize`) and lines 137 to 146 (`w_choice`).
- FACT: If a layer has no k_norm weight, `w_choice` uses the data-derived w. Then V3 tests a fit to the same data, and a wrong module weight cannot show.
- Failure scenario: a hook fault leaves one layer without a k_norm weight. V3 passes for that layer by construction. The run reports PASS with one layer unchecked.
- Fix: add checks for the non-stub run. (1) All 10 global layers (5, 11, ..., 59) have a k_norm weight and `w_match` is true. (2) `vllm` equals `0.33.0rc1.dev229+g3966924d5`. (3) `engine.cache_dtype` is "auto" and the compile mode is 0. (4) `len(global_layers)` is 10.
- Optional strengthening of V3: report the largest identity error per token. FACT: the L2 error over 8,192 tokens shows an identity break at one token (the BOS sink) only as 0.011 times the break.

### F6. Report the score-path and the value-path error
- Where: `analyze_e53.py` lines 217 to 225.
- FACT: Both storages hold the same fp8(V). Only the scores differ. In my synthetic runs the value-path error is 0.027 and the score-path error is 0.047 to 0.28 (`explore2.py`).
- Failure scenario: the ratio is 1.02 and nobody can tell whether the score path or the value path carries the error. A later change of the layout then has no baseline for the part that it can change.
- Fix: compute `attend(q, k8, v)` and `attend(q, k, v8)` for the standard storage, and the same two for the compact views. Report the four errors and the score-only ratio. It costs 4 more `attend` calls per layer and no Pod time.

### F7. Record the compile mode as a name
- Where: `capture_e53.py` line 519.
- FACT: `CompilationMode` is an `IntEnum` (`vllm/config/compilation.py` line 37). On Python 3.12 `str()` returns "0", not "NONE". `CUDAGraphMode` is a plain `Enum`, so its string keeps the name.
- Failure scenario: the main session looks for "NONE" in `capture_run.json`, finds "0" and stops a good run, or accepts the wrong text.
- Fix: write `.name`. The VERIFY item in README Section 9 line 215 then matches.

### F8. Correct the README text
- README line 198: the session watchdog does not fire before the boot watchdog (B1).
- README line 74 names `driver_worker.model_runner.model` as the fallback. The code uses `driver_worker.worker.model_runner.get_model()` (line 466). Both work (`worker_base.py` lines 370 to 371). Name the code path.
- README lines 205 to 209: change "no cheap GPU holds the model" to "no cheap GPU holds the 31B weights" (F1).

## Optional items

### O1. One long prompt
- Failure scenario: the claim "the relative error does not depend on the context length" (README Section 10) stays untested, and the target regime is 32K to 64K.
- Add a fifth prompt of 16K or 32K tokens as the last prompt, so the fail-fast on prompt 0 stays the same. It adds 1.4 GB (16K) or 2.7 GB (32K) of captures and tests the context claim of README Section 10. Set `max_model_len` to 33,024. If `max_num_batched_tokens` stays 8,448, the step splits. Then the prompt also tests the chunk path (a risk for that prompt only).

### O2. A sensitivity control
- FACT: The stub tests reach FAIL by editing `metrics.json` (`test_fail_paths`, lines 285 to 290). No test shows that the real computation can give a ratio above 1.25.
- Failure scenario: a PASS with ratios near 1.0 cannot be told from a metric that is blind to storage differences.
- Add one reported row: the compact storage in E5M2. Expected ratio near 2. It shows that the harness resolves a 2 times storage error. Add the same test to the stub.

### O3. A splice test for error growth
- Failure scenario: each layer passes alone, but the error grows over the 10 global layers. E-53 cannot see it.
- INTERPRETATION: In the same engine start, a forward hook could replace the global attention output with the emulated FP8 or compact output. The next-token log-probabilities then show the growth across the 10 layers. It needs new code and its own rehearsal. Leave it to the week 6 test or to a later Pod.

### O4. Ledger and hardware log
- Failure scenario: the run has no ledger row, so `compare_runs.py` cannot use it in a later comparison.
- README Section 9 (run steps) does not mention the ledger manifest or the `HARDWARE_LOG.md` row. MEASUREMENT.md Section 5 and PROTOCOL.md S7 require both. A small adapter is enough.

### O5. Data center
- Failure scenario: a pull of 15 min bills about $1.0 before any work starts, and the Pod is then deleted by the pull rule.
- FACT: E-52 reached SSH in 211 s in EUR-NO-2 with the image on the host. E-50 and E-51 waited about 11 min in US-NE-1.
- HYPOTHESIS: Asking for EUR-NO-2 first can lower the chance of a long pull. A host cache is not guaranteed.

## Answers to the seven questions

### 1. Does the hook path work in vLLM 3966924d5?
- FACT: Yes in source. The call chain has seven links:
  1. `LLM.apply_model` (`llm.py` 604 to 614).
  2. `LLMEngine.apply_model` (`llm_engine.py` 503 to 504).
  3. `InprocClient.collective_rpc` (`core_client.py` 498 to 505).
  4. `EngineCore.collective_rpc` (`core.py` 1152 to 1159).
  5. `UniProcExecutor.collective_rpc` (`uniproc_executor.py` 90 to 103).
  6. `run_method` (`serial_utils.py` 485 to 507).
  7. `WorkerBase.apply_model` (`worker_base.py` 165 to 167).
- FACT: The result is a list with one item, so `[0]` is right.
- FACT: `VLLM_ENABLE_V1_MULTIPROCESSING` is 1 by default and the capture sets 0 (`envs.py` 1448 to 1450; `llm_engine.py` 186 to 188). The client is `InprocClient` when the engine is not multiprocess (`core_client.py` 129 to 137).
- FACT: `Gemma4Attention.forward` calls `k_norm` (line 565), `rotary_emb(positions, q, k)` (567), `v_norm` (570) and `self.attn(q, k, v)` (572). `Attention.forward(query, key, value)` takes them as positional arguments and views them at lines 538 to 543. So a pre-hook on the `Attention` module sees post-RoPE q and k and the normed v, as `[tokens, heads x 512]`.
- FACT: In `k_eq_v` layers the K and V slots hold the same weights (`gemma4.py` 440 to 441, 555 to 558). The decoder layer calls `self_attn(positions=..., hidden_states=...)` with keywords (745 to 749). So the parent pre-hook gets `positions`.
- FACT: `enforce_eager` sets compile mode NONE, graph mode NONE and JIT warm-up off (`config/vllm.py` 1782 to 1800). The layer names end in `layers.N.self_attn.attn` (`gemma4_mm.py` 1144 to 1150).
- FACT: The prompt is 8,192 tokens and `max_num_batched_tokens` is 8,448, so the prefill is one step. The hooks sort rows by position, so a split step also works (tested with two chunks).
- FACT: Prefix caching is off. With it on, cached tokens would skip the layer, and `check_prompt` would report missing key positions. With CUDA graphs the Python hooks would not run, and `check_prompt` would report "no global-layer tensors". Both cases stop the run at prompt 0.
- INTERPRETATION: The hooks copy rows to the CPU at once (`.to("cpu")` is synchronous), so the in-place RoPE and later layers cannot change them. A hook error is recorded and stops the run after prompt 0.
- HYPOTHESIS: The first prefill compiles the FA4 kernels. In E-50 the first graph capture took 4.3 s with the JIT (log line 55). I expect under 30 s.
- Residual risk: the engine arguments are new for this image (F1).

### 2. Is the FP8 cast faithful, and is the views query faithful?
- FACT: `reshape_and_cache_flash_kernel` calls `fp8::scaled_convert` with `k_scale` and `v_scale` (`cache_kernels.cu` 252 to 262, 375 to 379). For BF16 it computes `__nv_cvt_float_to_fp8(float(a) / scale, __NV_SATFINITE, __NV_E4M3)` (`quant_utils.cuh` 231 to 243, 285 to 290). The scale 1.0 divides exactly. The result is round to nearest even with saturation at 448.
- FACT: `fp8()` clamps to +-448 and then casts with `to(torch.float8_e4m3fn)`. The clamp prevents the NaN that the PyTorch cast gives for values of 464 and above. The test checks saturation, the tie 2^-10 and the subnormals.
- FACT: `vllm/ir/ops/layernorm.py` lines 9 to 20 round the normalized value to the weight dtype before the product (`x.to(weight.dtype) * weight`). So K_unrot = BF16(BF16(n) w) and V = BF16(n). The stub uses the same rounding. The identity error 0.0017 is the BF16 product rounding.
- INTERPRETATION: BF16 rounding of q * w is the natural engine form: a BF16 product of two BF16 values. Folding w into the q_norm weight gives a rounding of the same size. The difference is far below the FP8 error.
- FACT: The query is not quantized with an FP8 cache on the SM90 FA4 path (`flash_attn.py` 1160 to 1165). The float32 compute and the FP16 emulation bracket the engine compute (F3).

### 3. Is the decision rule sound?
- Saturated share: not sound (B3).
- Thresholds 1.25 and 2: INTERPRETATION: they are project heuristics. They allow an extra layout error of 0.75 and 1.73 times the FP8 error in quadrature (F3). Both storages use the same data, so the sampling noise of the ratio is small. The expected ratio is near 1.
- "Largest" condition: it compares the maxima of different prompts (F3).
- FP16 limits: the 6,550 bound is a sound PASS margin. The 32,752 FAIL is too strong, because a scaled query fixes it (B4). One value among about 170 million decides it (10 layers, 4 prompts, 256 rows, 32 heads, 512 dims).
- V3: the data only pick between two module forms, so V3 is not circular when the module weight exists. It passes on the BF16 level (0.0017 against 0.01). Without the module weight it is a fit (F5). It is blind to a break at one token (F5).
- Good layout FAIL: mainly by the single |q'| value (B4).
- Bad layout PASS: a wrong q or mask (B2), a per-prompt blow-up (F3), and the FP16 compute path (F3).
- FACT (synthetic): with w from 1+0.1N the ratio is 1.007 to 1.014. With exp(0.3N) and exp(1.0N) it is 0.90 to 1.07. With exp(2.0N) the standard storage saturates and the ratio is 0.40 to 0.48 (`explore1.py`, `explore2.py`).

### 4. Is the measurement enough for the gate question?
- INTERPRETATION: It is enough to catch a broken identity, an FP16 range problem and a large storage difference at 8K. It does not test error growth, long context, the FP16 kernel or chat text. The README states these limits.
- Cheapest additions, no second Pod: (1) the engine-output control (B2); (2) per-prompt ratios and the FP16 ratio in the rule (F3); (3) the score and value split (F6); (4) one 16K or 32K prompt (O1).

### 5. Budget and caps
- B1 gives the numbers. The new bound is 44 min and $2.93.

### 6. Code bugs
- FACT: I ran the tests: 17 passed in 122.6 s. The command in the task ran with no change.
- FACT: I found no defect in the real-path code that I can prove. The code is consistent with the vLLM source.
- Gaps: B2 (no engine control), F4 (special tokens), F5 (silent V3 path), F7 (`str(IntEnum)`).
- FACT: `capture_run.json` can end with status "ok" when prompts 1 to 3 had problems (`capture_e53.py` lines 493 to 504). V2 catches it from `metrics.json`, so this is only a note.
- FACT: The prompt builder walks the whole `dist-packages` folder twice (Markdown and reStructuredText) before the engine starts (`capture_e53.py` lines 84 to 96). The time model does not include it. HYPOTHESIS: under 1 min.

### 7. Is a GPU rehearsal needed?
- The fail-fast bounds the loss at about $0.6, so it is not strictly needed. PROTOCOL.md S5 is a gate, and the user must waive it. A cheap rehearsal exists (F1), so I recommend it.

## Amendment A1 checklist (for the author, before S3 and S4)
1. B1: change the caps and the pull rule in README Section 9, `session_e53.sh` and the S6 recipe.
2. B2: add the engine-output hook, the check V5 and two tests.
3. B3: replace the saturated-share condition by a count rule and update the test.
4. B4: add the consequence table to README Section 7.
5. F1 to F5: apply, or answer each one in writing (PROTOCOL.md S2 gate).
6. Run the 17 tests and the new ones again (S3). Then ask the user for S4 and, if the user wants it, S5.

## What I could not check
- I ran no vLLM and no GPU code. The behavior of the hooks inside a live engine is read from source only.
- I did not read the real k_norm weights or the real q' range. Both decide B3 and B4 in practice.
- I did not check which RMSNorm implementation the H100 selects. The 4090 log shows the native one. Both round to BF16 before the weight product in the source that I read.
- I did not check the rounding inside the FA4 BF16 kernel, so the bound of V5 is an estimate.
- My scratch data are synthetic. They show mechanisms, not the real outcome.
