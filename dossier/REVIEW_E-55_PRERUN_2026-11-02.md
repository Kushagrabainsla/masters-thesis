# R-77: pre-run review of E-55 (model-level accuracy of the compact K=V cache, BF16 and FP8, in the Gemma 4 31B global layers)

VERDICT: AMEND_THEN_RUN. There are two blockers. B1: the BF16 arm never records its V5 gap, so every real run ends INVALID (V5). B2: the S5 rehearsal cannot run the scaled arms. The splice, the K descale route and the log-probability parser are correct by source.

Date: 2026-11-02 to 2026-11-03.
Reviewer: R-77 (Sonnet 5.5, a different model from the author).
Stage: S2 hostile design review of the preregistration in design commit e59dda0 (`experiments/gemma4_kv_model/README.md`).
Labels: FACT means that I read a file or a source, ran code or computed a number. INTERPRETATION means my reasoning. HYPOTHESIS means not tested. Every number from my noise model is a HYPOTHESIS.

Rules kept:
- I wrote this one repo file and no other repo file.
- I did not read `.env`. I made no Runpod, cloud or GPU call. I did not commit, push or post.
- I made read-only `gh api` GET calls. They were the file tree and 22 source files of vllm-project/vllm at 3966924d5, one code search on vllm-project/vllm, and 2 files of vllm-project/flash-attention at b273a02. I saved them in the scratchpad only.
- I ran the author's test file once, with bytecode writing off. It gave 15 passed in 462 s on this Mac. It left no cache folder in the repo.
- One early command chained `cd /tmp` with other commands. It changed no file. I used absolute paths after it.

Scratch folder: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/20255ed9-038a-44e5-8a60-f7885a8c54ee/scratchpad/r77/`.
It holds the vLLM sources (`src/`), my scripts and their logs: `b1_warmup_gap.py`, `b2_reh_scales.py`, `b3_validity_demo.py`, `cap_checks.py`, `quant_amp.py`, `power_sim.py`, `power_sim_h8.py`, `coverage.py`, `data_check.py`.

## Key findings
1. FACT: The splice reaches every read of K and V in this build. `FlashAttentionImpl.forward` takes K and V only from the paged cache (`flash_attn.py` 1257 and 1404 to 1413). `Attention.forward` writes the cache from the `key` and `value` that the pre-hook returns (`attention.py` 553 to 580, `do_kv_cache_update` at `flash_attn.py` 1432). The file has no cascade path. Prefill chunks, the current chunk and decode use one call.
2. FACT: The K descale route works by source. `FlashAttentionImpl.scale` is read on every call (`flash_attn.py` 1413) and nothing caches it. Gemma 4 sets the base scale to 1.0 (`gemma4.py` 419). FA4 takes `softmax_scale` as a run-time Float32 (`flash_fwd_sm90.py` 201) and its compile key has no scale (`cute/interface.py` 1421 to 1500). The caller uses the value that the output hook returns (`gemma4.py` 572 to 574).
3. FACT: The E-54 engine ran the V2 model runner (log line 24: "Using V2 Model Runner"). `config/vllm.py` 746 to 783 selects V2 by default. INTERPRETATION: E-55 uses it too, because no E-55 option is on the list of unsupported features.
   - README Section 4.2 cites the V1 runner (`gpu_model_runner.py` 5245 to 5352). V2 scores prompt tokens in chunks of 1,024 rows (`gpu/sample/prompt_logprob.py`, `CHUNK_SIZE`).
   - The output layout is the same: [prompt token, top-1]. The README memory model is too high. This fails safe (F6).
4. FACT (replay): The BF16 arm never records a V5 gap. The warm-up call uses up the check. The analysis then reports V5 and the run is INVALID (B1).
5. FACT (replay): With `--rehearsal` the scales are None. `set_arm("TIED8")` raises TypeError. The rehearsal cannot test the K descale route (B2).
6. FACT (replay): A failed optional noise phase makes the whole run INVALID (F1). A reported step that overruns its estimate does the same (F2).
7. FACT: The arms reproduce E-54. `attend32` equals the E-53 `attend` exactly (difference 0.0) on prompt 0, layers 5 and 41. I found no numeric fault in `transform`, `check_transform` or `attend32`.
8. FACT: V5 cannot tell a hooked BF16C from an unhooked one. The BF16C error against the exact reference is 0.0034 in layer 5 and 0.0063 in layer 41. The V5 tolerance is 0.01. For BF16C, only V6 (E5M2) shows that the hooks reach the engine (F8).
9. INTERPRETATION: The checkpoint is FP8-dynamic, so every linear layer quantizes its input per token. A tiny upstream change flips FP8 codes.
   - On a Gaussian row of 16,384 values, a 0.1% change grows to 1.0% after the quantizer. A 0.3% change grows to 1.7% (`quant_amp.py`).
   - BF16C changes the attention output by 0.3% to 0.6% in 10 layers. So BF16C probably gives a new realization of the quantization noise of the whole model.
   - HYPOTHESIS: The per-token noise of the log-probability difference is then about 0.05 to 0.11 nats.
10. HYPOTHESIS (model): With that noise and a design effect of 3, the planned interval of Delta_BF16C has a half-width of about 2e-4 to 5e-4. The margin m16 is below that when Delta_E5M2 is below about 0.002 to 0.005. Then GO16 cannot happen. The result is INCONCLUSIVE16 for a reason that has nothing to do with the cache (F4).
11. FACT (simulation): The planned stratified percentile bootstrap (4 documents per domain) covers the true value in 86% of draws, not 95%. A stratified t interval covers 94.7% and is 1.35 times wider. README Section 10 says only "about 13%" (F5).
12. FACT: The data file is the plan. The hash, the lengths, BOS, the 511,988 scored tokens and the 16 prompt endings match. At most 3.7% of the scored tokens of any document repeat a 12-gram from beyond the sliding window. The books B2 to B4 have 0.1% or less. So the test measures typical next-token predictions, not retrieval (answer 7 and O2).
13. INTERPRETATION (from the E-54 log): The engine starts. One 65,664-token request needs about 12.0 GiB of KV cache at 4,096 batched tokens. About 23.9 GiB is available at utilization 0.70.

## Blockers

B1. The BF16 arm never records its V5 gap, so every real run ends INVALID.
- Where: `run_e55.py` 288 to 289 (the warm-up call). `splice_e55.py` 266 to 284 (the `first` flag, the early return for BF16, the `fresh` reference). `analyze_e55.py` 232 to 237 (V5 needs a gap in every layer for every arm in `main["arms"]`, BF16 included).
- Failure: The warm-up call runs with the hooks attached, the arm BF16 and no `new_doc` call. So `fresh` is false and no reference is stored. The call adds `(BF16, layer)` to `checked`. In the BF16 arm, `first` is then false and the hook returns at once. No BF16 gap exists. The analysis adds "BF16: engine-output gaps {}" to V5, and the decision is INVALID.
- Cost: This happens after about 30 min of paid H100 time.
- Why no test catches it: The toy test calls `new_doc` before every call. The stub session writes fake gaps (`run_e55.py` 246). The rehearsal skips INVALID (`analyze_e55.py` 277).
- Evidence (FACT): `b1_warmup_gap.py` replays the real order on ten toy modules (attach, one warm-up call, then every arm with `new_doc`). The gap keys are BF16C, E5M2, JOINT8, STOCK8, TIED8 and TIED8_UP. BF16 is missing. `b3_validity_demo.py` runs the author's `validity` on a record with this shape and gets the V5 problem.
- Fix: After the warm-up call, clear `splice.checked`, `checks`, `gaps`, `ref` and `counts`. Then the first chunk of each arm gives the evidence, as designed. Add a test that replays the engine order (attach, warm-up, then arms) and asserts 7 gaps of 10 layers each. Remove the fake gaps from the stub, or make them follow the same flow.

B2. The S5 rehearsal cannot run TIED8, JOINT8 or TIED8_UP.
- Where: `run_e55.py` 270 (`Splice(..., None if args.rehearsal else scales, ...)`). `splice_e55.py` 61 to 66, 218 to 219 and 243 to 252 (`arm_scales`, `layer_scales`, `set_arm`). `analyze_e55.py` 277 and 340 to 352 (the rehearsal skips INVALID, and `line()` hides the validity problems).
- Failure: In rehearsal mode `scales` is None. `set_arm("TIED8")` reads `sc["tied"]` of None and raises TypeError. The main phase stops after STOCK8. The rehearsal never reaches TIED8, JOINT8, TIED8_UP, GEN or the noise phase.
- Cost: VERIFY items (7), (9) and part of (10) cannot pass. The impl scale route is the largest unknown of E-55, and it gets no test before the H100.
- Second defect: With the scales fixed, the REHEARSAL line would still show no V1 to V5 problem.
- Evidence (FACT): `b2_reh_scales.py` calls `set_arm` for every arm with `scales` None. The first four arms pass. The three scaled arms fail with "TypeError 'NoneType' object is not subscriptable".
- Fix: In rehearsal mode, build synthetic scales for each layer after `register` reads `w`. Use for example vs 0.125, ks = f32(w vs), joint ks 0.01 and joint vs 0.2. Print the V1 to V5 problems in the REHEARSAL line. Add a stub test with `--rehearsal`. Also apply F3. E-54 had no rehearsal GPU in stock (A2), and E-55 may meet the same limit.

## Fixes (should be done)

F1. The optional noise phase can make the whole run INVALID.
- Where: `analyze_e55.py` 167 to 176; `session_e55.sh` 124 to 136.
- Failure: If engine 2 fails to start, `run_noise.json` has status error and no `engine` key. `validity` reads `{}` and reports three V2 problems (not BF16 auto, not eager, no batched tokens). The decision is INVALID, although every decision arm is valid. The session also starts the noise phase when the main phase failed.
- Evidence (FACT): `b3_validity_demo.py`, case 2.
- Fix: Check the noise engine record only if `noise["status"] == "ok"`. Otherwise treat BF16_B as not run. Skip the noise phase when `run_main.json` is not ok.

F2. A reported step can invalidate the decision.
- Where: `run_e55.py` 218 to 225, 322 to 342; `session_e55.sh` 108 to 113; `analyze_e55.py` 153 to 154, 197 to 205, 232 to 237.
- Failure: The soft limit ends a reported step 60 s before the hard kill (`--reserve 60` against `timeout` at DEADLINE minus 120 s). If a reported step runs 60 s over its estimate, `timeout` kills the main phase. The file `run_main.json` then says "running". It has no `counts`, `checks` or `gaps`, because the code writes them only at the end (lines 339 to 342). V4 and V3 fail, although all decision arms are complete. A V3 or V5 problem in JOINT8 or TIED8_UP (reported arms) also makes the decision INVALID.
- Fix: Write `counts`, `checks` and `gaps` with every `save()`. Set `rec["decision_arms_done"] = True` after TIED8. Let V4 accept status "running" or "error" when that flag is true. Judge the reported arms apart: mark the arm invalid, not the run. Set the reserve to 1.5 times the estimate plus 60 s. Install a SIGTERM handler that saves the record.

F3. Add a pre-flight and fail fast inside the main phase.
- Where: `run_e55.py` after line 289; README Sections 4.5 and 6.
- Failure: The first test of FA4 with the changed impl scale is the paid H100 run. A fault in the K descale route, in DET or in code like B1 shows only in the analysis. That is after about 30 min and $2. The main phase also continues after DET fails.
- Fix: After the warm-up, run a pre-flight on the first 4,096 tokens of B1 (one chunk, about 2 s per arm). Run every arm and DET. Stop with exit 3 and a clear reason if any check below fails.
  - The transform check of any arm fails, or its V5 gap is above 0.01.
  - E5M2 or STOCK8 gives the same log-probabilities as BF16.
  - BF16C gives the same log-probabilities as BF16 in more than 50% of the tokens.
  - DET is not bit-identical to BF16 (or it is off by more than 1e-4).
- Clear `checked` and the gaps after the pre-flight (the B1 fix). The cost is about $0.1. A stop here saves about $2.5.

F4. The margin m16 can fall below what the test resolves.
- Where: README Sections 5 and 7; `analyze_e55.py` 319.
- Failure: m16 = min(0.001, 0.1 x Delta_E5M2) shrinks when the effects are small, and that is when noise sets the resolution. The table shows P(GO16) when the true BF16C effect is 0 (HYPOTHESIS). The planned interval has half-width hw. P = 0 when m16 is at or below hw.

  | hw \ m16 | 1e-4 | 2e-4 | 3e-4 | 5e-4 | 1e-3 |
  |---|---|---|---|---|---|
  | 1e-4 | 0.00 | 0.91 | 1.00 | 1.00 | 1.00 |
  | 2e-4 | 0.00 | 0.00 | 0.60 | 0.99 | 1.00 |
  | 3e-4 | 0.00 | 0.00 | 0.00 | 0.74 | 1.00 |
  | 5e-4 | 0.00 | 0.00 | 0.00 | 0.00 | 0.91 |

  The half-width is 1.7 x sigma x sqrt(design effect / 511,988). Here sigma is the per-token SD of the log-probability difference between two arms. The design effect is the factor by which correlation inside a document cuts the number of independent tokens. For sigma 0.05, 0.08 and 0.11 and a design effect of 3, hw is 2.1e-4, 3.3e-4 and 4.5e-4. The Monte Carlo of the author's rule code (`power_sim.log`) agrees. With sigma_c 0.04 and a design effect of 3, P(GO16) is 0.01 when Delta_E5M2 is 0.001. It is 0.51 when Delta_E5M2 is 0.0024.
- Also: m16 depends on the observed Delta_E5M2, so the margin is data-dependent. The README does not state the effect that the test can detect (its resolution).
- Fix: Use the fixed margin 0.001 for GO16. It is the "0.1% of perplexity" claim of the README. Report 0.1 x Delta_E5M2 as a secondary number. Keep V6 as the validity check of E5M2. If the author keeps the relative margin, write the half-width that the rule needs and name INCONCLUSIVE16 as the expected result when hw exceeds m16.

F5. The interval is too narrow, and two rules of H8 are fragile.
- Where: README Sections 5, 7 and 10 (item 4); `analyze_e55.py` 95 to 108 (`draws`, `interval`), 257 to 266 (`decide8`), 320 (`tol8`).
- Failure (interval): With 4 documents per domain, the percentile bootstrap covers 86.1% (`coverage.log`). The stratified t interval covers 94.7% and has 1.35 times the half-width. The false KILL8 or PIVOT8 rate is 5% to 10% when TIED8 equals STOCK8 (`power_sim_h8.log`).
- Failure (KILL8 ratio): KILL8 and PIVOT8 differ only by Delta_TIED8 > 1.5 x Delta_STOCK8. Both numbers are noisy and small. With TIED8 truly 1.32 times worse, KILL8 is 0.14 to 0.26 and PIVOT8 is 0.02 to 0.54 in my scenarios. The two actions differ (new scales and a new test, or no compact FP8 claim).
- Failure (tolerance): tol8 is the absolute floor 2e-4 when Delta_STOCK8 is below 8e-4. That is the likely range. Then GO8 accepts a TIED8 effect that is up to 1.25 to 2 times the STOCK8 effect. In my scenario with Delta_STOCK8 = 2e-4 and TIED8 truly 1.69 times worse, GO8 is 0.20.
- Fix: Use the stratified t interval, or the wider of the document interval and the block interval, as the decision interval. Define KILL8 with the lower bound of D against Delta_STOCK8 (for example lo(D) > 0.5 x Delta_STOCK8). Write in the README that GO8 means "not worse than STOCK8 by more than tol8 nats", and state tol8 in percent of Delta_STOCK8.

F6. The README facts about prompt log-probabilities describe the wrong runner.
- Where: README Section 4.2 (FACT with `gpu_model_runner.py` 5245 to 5352) and Section 10 item 10.
- Failure: The engine runs V2 (finding 3). V2 scores 1,024 rows at a time. INTERPRETATION: the memory per call is about 1 to 3 GB, not 6 to 11 GB. The statement "the first document fails fast if not" has no code behind it. The engine record does not name the runner.
- Fix: Cite `gpu/sample/prompt_logprob.py` and `gpu/sample/logprob.py`. Add `use_v2_model_runner` to `engine_record` and to V2. Delete or implement the fail-fast claim.

F7. BF16_B may not measure noise.
- Where: README Sections 5 and 8; `run_e55.py` 46 and 309 to 310.
- Failure: BF16_B changes only the chunk size (2,048 against 4,096). A per-row kernel gives the same result for any chunk size, so BF16_B may be bit-identical to BF16 (HYPOTHESIS). Then it measures nothing, and the "noise floor" claim and the prediction "within 1e-4" hold by default. It also needs a second engine start (about 5 min).
- Fix: Add the arm BF16R: q' = bf16(q x c) with c = 1 + 2^-7 and impl.scale = 1/c, in the same engine. It adds one BF16 rounding of the same size as the BF16C query rounding, and its true effect is zero. It costs one pass (about 3 min) and no restart. Report Delta_BF16R and Delta_BF16C minus Delta_BF16R. Keep BF16_B as an optional extra.

F8. Add a guard that BF16C differs from BF16.
- Where: `splice_e55.py` 130 to 146; `analyze_e55.py` 197 to 242.
- Failure: V5 cannot see a BF16C that the engine ignored (finding 8). The result would be Delta = 0 exactly and a GO16 by artifact. V6 covers the K and V path only. It does not cover the query path.
- Fix: Add V7. The BF16C log-probabilities must differ from BF16 in at least 50% of the scored tokens. The first-chunk engine output of BF16C must differ from BF16 in the 9 layers whose w is not a power of 2. Layer 29 has w = 2^-4, so BF16C is exact there.

F9. README and code differ in small points.
- `analyze_e55.py` 308 passes 2 for "all tokens", but README Section 5 says "index 1 or more". Token index 1 is dropped. Use 1.
- `run_e55.py` 210 to 211 records the revision from the command line, so V2 compares a constant with itself. Record the snapshot folder name and compare it with the pinned revision.
- README Section 2 predicts a BF16_B top-1 agreement of 99% or more. The logits are BF16 and pass three BF16 operations in the soft cap. So ties and near ties can flip top-1 in a few percent of tokens (HYPOTHESIS). Drop the prediction or use the BF16R value as the reference.
- README Section 9 gives the worst case as 12 min of pull plus 47 min ($3.92). E-54 deletes the Pod after 15 min without SSH. With 15 min the worst case is 60 min ($3.99) under the session watchdog and 62 min ($4.12) under the boot watchdog. Name this in S4.
- `decide16` maps a significant improvement (Delta below minus m16) to INCONCLUSIVE16. BF16C cannot truly improve the NLL. Add an explicit outcome "artifact, check the arm".
- No script builds `code_manifest.json` (L-25). The tests write it by hand. Add `make_manifest.py` and a test, so S6 does not depend on a manual `git show`.
- D-94 option (1) names KL. E-55 has no KL. State the omission in README Section 10.

F10. GEN is hard to read after the end of the answer.
- Where: `run_e55.py` 278 and 328 to 335; `analyze_e55.py` 126 to 144.
- Failure: `ignore_eos` makes the model continue after its end token. A five-sentence summary may end before 128 tokens. After that point the tokens are unstable, so "exact matches of 128 tokens" and "first divergence" mix noise with the effect.
- Fix: Keep the end token position of each output. Compute the divergence only before that position.

## Options (may be done)

O1. Validate the STOCK8 emulation against the real FP8 cache.
- This vLLM build has `kv_cache_dtype="fp8"` with `kv_cache_dtype_skip_layers=["sliding_window"]` (`attention.py` 299 to 316). It gives a real stock FP8 cache in the 10 global layers only. A third engine start and one arm (about 5 min, $0.35) would give Delta_STOCK8_REAL. Compare it with the emulated STOCK8. This tests the real write kernel and the FA4 FP8 kernel (P in FP16) for the baseline of H8.
- The page sizes seem to match: 262,144 bytes per layer page in both groups. This comes from arithmetic on the E-54 KV sizes, which implies 16 KV heads in the sliding layers. INTERPRETATION: the manager accepts the mix.

O2. Pre-register a far-repeat subset and a copy probe.
- FACT: In the data, some scored tokens repeat a 12-gram from beyond the 1,024-token window. The share is 1.1% in B1, 0.1% in B2 and B3, and below 0.05% in B4. It is 0.3% to 0.6% in prose and 1.4% to 3.7% in code. These tokens need the global layers.
- Add the Delta on that subset as a secondary metric before the run. Add one synthetic copy document if the user wants a test of attention precision. Use a random passage that repeats at a distance of 30K to 60K tokens.

O3. Add a truncated KL.
- This build can return the log-probability at chosen token ids per prompt row (`prompt_logprob_token_ids`, `sampling_params.py` 353 to 364 and `gpu/sample/prompt_logprob.py` 56 to 62). Take the BF16 top-k ids per row and score the arms at them. KL is non-negative and easier to read than a signed NLL difference. This needs a second pass with fixed ids and more code. It fits a new S4, not this one.

O4. Compute the lm_head in FP32.
- `hf_overrides={"head_dtype": "float32"}` makes the logits and the soft cap run in FP32 (`logits_processor.py` 92 to 175). This removes the BF16 rounding of the final logits. HYPOTHESIS: it cuts the per-token noise by 20% to 40%. It needs an unquantized lm_head and a check in the rehearsal. It also changes the engine from stock, so name it in the README.

O5. Plan the next step for INCONCLUSIVE.
- If H16 or H8 ends INCONCLUSIVE because of noise, more documents are the only remedy. A fourfold token count halves the interval. Pre-write the size of that second S4 now. Reuse the data of this run for the first 12 documents.

## Answers to the checks

### 1. The splice
- FACT: `Gemma4Attention.forward` (`gemma4.py` 540 to 574) applies k_norm (a standard RMSNorm with weight), RoPE and the weightless v_norm. It calls `self.attn(q, k, v)` with flat [T, heads x 512] tensors and uses the returned value in `o_proj`. The global layers have no KV sharing in the 31B.
- FACT: `Attention.forward` (`attention.py` 490 to 585) calls `unified_kv_cache_update(key, value, ...)` and then `unified_attention_with_output(query, key, value, output, ...)`. The module call runs the pre-hook first, so both ops get the hooked tensors. `kv_cache_update_bypass` is False and the backend has `forward_includes_kv_cache_update = False`.
- FACT: `FlashAttentionImpl.forward` (`flash_attn.py` 1182 to 1430) splits `kv_cache` into `key_cache` and `value_cache` and passes only these to FA4. In `forward`, the `key` and `value` arguments serve only the encoder and DCP paths. The cache write uses them in `do_kv_cache_update`. INTERPRETATION: `reshape_and_cache_flash` with dtype "auto" copies BF16 and applies no scale (the scale arguments serve the FP8 dtypes). I did not read the CUDA kernel.
- FACT: No file in the pinned tree has a cascade attention path. `use_cascade` and `cascade_attention` do not occur in `flash_attn.py`. Prefix caching is off. DCP is off.
- INTERPRETATION: Every read sees the hooked values. This includes the first chunk, the later chunks that read earlier chunks from the cache, and decode.
- FACT: `register()` finds the 10 modules through the parent class `Gemma4Attention` with `is_sliding` False. E-53 and E-54 used the same route on the same build.

### 2. The K descale route
- FACT: `self.scale = float(scale)` is set once (`flash_attn.py` 1077). `softmax_scale=self.scale` is read in the dense path (1413) and in the other paths (1500, 1590, 1622, 1698). I found no copy of the value. The only other reader is `get_per_layer_parameters` (`backends/utils.py` 440), a FlashInfer plan path. This engine uses FLASH_ATTN, so it does not run.
- FACT: The base scale is 1.0 for every layer (`self.scaling = 1.0`, `gemma4.py` 419). `attn_logit_softcapping` reaches the layer as `softcap`. E-54 matched the engine to 0.0017 without a soft cap, so the value is 0 or None.
- FACT: FA4 (`flash_fwd_sm90.py` 194 to 201 and 454) takes `softmax_scale: Float32` and builds `softmax_scale_log2` inside the kernel. The compile key (`interface.py` 1421 to 1500) holds no scale. For a BF16 cache, `vllm_flash_attn/flash_attn_interface.py` sets all descales to None (line 442). So `softmax_scale` alone carries ks.
- INTERPRETATION: The route is correct. V5 on the first chunk would catch a kernel that ignored it. Later chunks use the same call, so the first chunk is enough.
- FACT: The V descale is applied by the output hook. `Attention.forward` returns a view of a preallocated buffer. The hook returns a new tensor, and the caller uses only the returned tensor.

### 3. The numerics
- FACT: The 15 tests pass. The transforms give the E-54 errors to 1e-4 on prompt 0, layers 5 and 41. I added a direct check: `attend32` equals the E-53 `attend` with a difference of 0.0 (`cap_checks.log`).
- FACT: BF16C is the compact numerics. It has one more rounding than a build that folds w into the query norm. The hook rounds q before and after the product with w. INTERPRETATION: BF16C is slightly pessimistic.
- FACT: The output hook rounds to BF16, multiplies by vs and rounds again. The FP8 kernel rounds once. On the captures the two forms differ by 0.0028 relative for TIED8, JOINT8 and TIED8_UP, in layers 5 and 41. STOCK8 and E5M2 have no such step.
- FACT: The TIED8 error moves from 0.02312 to 0.02327 (layer 5) and from 0.02263 to 0.02269 (layer 41). INTERPRETATION: The effect on D is below 1e-5 nats. Name it in the README. The step is as large as half of the BF16C error.
- FACT: The BF16 kernel computes P in BF16 and the FP8 kernel computes P in FP16. All arms share the BF16 kernel, so the comparison with BF16 is consistent. The README says so (Section 10, item 1).
- FACT: `fp8()` divides by a Python float. PyTorch may multiply by the reciprocal on the GPU. A rare code can flip at a tie (R-76 named this). INTERPRETATION: about 1e-6 of the codes flip, which is negligible.

### 4. Prompt log-probabilities
- FACT: V2 computes the rows in chunks of 1,024. It gets `[prompt token, top-1]` through `compute_topk_scores` and a Triton log-softmax in float32 from the BF16 logits. The rank kernel counts logits that are at least the token logit.
- FACT: The engine side adds one `None` for position 0 (`create_prompt_logprobs`) and 2 entries per later position (`append_logprobs_for_next_position`). Both are in `vllm/logprobs.py` 165 to 209 and are called from `engine/logprobs.py`. The last prompt row is dropped, so there are L minus 1 rows. This matches `parse_prompt_logprobs`. The check `rank == 1` for the second entry is true by construction, so it checks nothing.
- FACT: `SamplingParams` has `flat_logprobs` (`sampling_params.py` 373) and `prompt_logprobs` (350). `max_logprobs` defaults to 20, so 1 is allowed. Prefix caching is off.
- FACT: The logits are BF16 in the default engine, and the soft cap runs as three BF16 operations (`logits_processor.py` 113 to 116). This sets a rounding floor of about 0.03 on the target logit near 20 (INTERPRETATION).
- FACT: Alignment is correct. Array index i holds token i + 1, and the scored start index 1,025 maps to array index 1,024.
- INTERPRETATION: The memory is not a risk. V2 needs about 3 GB per call, and 24 GiB lies outside the vLLM budget.

### 5. Determinism and the A/A design
- INTERPRETATION: DET is a good check, but the session does not stop when DET fails (F3). The BF16 arm uses identity hooks that return None, so DET differs from BF16 only if the engine is not deterministic.
- INTERPRETATION: With FP8 activation quantizers, any nondeterminism of one ulp becomes code flips and a full noise realization (finding 9). Then DET exceeds 1e-4 with high probability, and V3 fails. This is correct behavior, but F3 should stop the run early.
- HYPOTHESIS: BF16_B may be bit-identical to BF16 (F7). BF16R is a better A/A.

### 6. The decision rule
- FACT (algebra): The outcomes in each hypothesis are exclusive because the rule checks them in order. I agree with the README.
- FACT: V6 asks only that the E5M2 interval lie above 0. It has power: E5M2 has an attention error of 8% to 12% in 10 layers (E-54 captures), so Delta_E5M2 is probably 1e-3 or more (HYPOTHESIS). In my simulations V6 failed in 0 of 300 draws in every scenario.
- Reachability by an artifact: (a) BF16C ignored by the engine gives GO16 (F8). (b) The narrow interval raises false GO and false KILL8 (F5). (c) The data-dependent margin and the tolerance floor decide GO16 and GO8 with no link to the cache (F4, F5). (d) A crashed noise phase or a reported-step timeout gives INVALID (F1, F2).
- Effect sizes (HYPOTHESIS). Delta_BF16C: about 0 (E-54 BF16C error is 8 times below STOCK, so a second-order law gives 1.6% of Delta_STOCK8). Delta_STOCK8: 2e-4 to 2e-3. Delta_E5M2: 1e-3 to 1e-2. D: minus 0.2 to minus 0.6 times Delta_STOCK8 if the E-54 ratio holds.
- Resolution (HYPOTHESIS). Half-width of Delta_BF16C: 2e-4 to 5e-4 (planned interval) or 3e-4 to 7e-4 (t interval) at a design effect of 3. Half-width of D: about 1.4 times larger.
- HYPOTHESIS (Monte Carlo of the author's rule code, `power_sim.log` and `power_sim_h8.log`, with the stated noise model): Take sigma_s 0.03 and a design effect of 3. Equal arms give GO8 0.65, INCONCLUSIVE8 0.30, PIVOT8 0.02 and KILL8 0.03.
- With a design effect of 8, equal arms give GO8 0.40, INCONCLUSIVE8 0.52 and KILL8 0.07. With TIED8 truly 1.69 times worse and sigma_s 0.03, KILL8 is 0.67 and GO8 is 0.01.
- INTERPRETATION: H16 has a high prior of GO and its main risk is resolution. H8 is a coin flip at the lower end of the effect range.

### 7. The data
- FACT: The hash is `8f2e2ecd...`. The 12 documents have 65,536 or 32,768 tokens. Each has one BOS and no id below 8 after it. The ids stay below 238,000. The 16 prompts have 3,024 to 3,026 tokens and end with the model turn and the empty thought channel.
- FACT: Unique tokens per document: books 5,584 to 9,858 of 65,536; prose 3,563 to 4,263 of 32,768; code 2,487 to 2,796 of 32,768. Code documents C3 and C4 share 18% of their 16-grams (test files). The other pairs share 3% or less. The generation prompts share at most 128 16-grams with a document (G12 and P1, about 4% of the prompt). The runs are independent, so this does no harm.
- FACT: The prose is AI-written review text with tables and ids. The code is research code. Both are narrow and predictable.
- INTERPRETATION: The PG-19 books may be in the training data. Report by domain, as the README says.
- INTERPRETATION: The scored start 1,025 is right. The sliding layers see 1,024 positions, so position 0 leaves their window at token index 1,025.

### 8. Budget, time cuts and watchdogs
- FACT: The estimates fit the E-54 data. E-54 prefill took 1.7 to 2.6 s at 8K and 10.05 s at 32K (`capture_run.json`). The model in the README gives 25 s at 64K and about 3 min per arm.
- FACT: The soft deadline works (F2 names the gap). The session watchdog fires at about 2,720 s after boot, before the boot watchdog at 2,820 s. `AUTO_STOP` must be 1 (the README implies it, the script default is 0).
- INTERPRETATION: The Pod cost is about $2.9 to $3.1 for the plan, and the worst case reaches the cap (F9).
- FACT: The code hash check (L-25) works as designed. The analysis checks the hash list of its own file, so it cannot prove its own integrity. The Mac recompute of the decision from the copied raw files closes this. Name it in S8.
- FACT: The rehearsal plan has the problems in B2. The E2B backend is not FA4, and it has no K=V. The pre-flight (F3) covers FA4 on the H100.

### 9. README against code
- The differences are in F6 and F9. The rest agrees. This covers the arms, the seeds, the 10,000 draws and the margins. It also covers the order of work, the 540 s rule for the noise phase, the hashed files and the 15 tests.

## What I could not check
- I ran no GPU code, no vLLM and no FA4 kernel. The runner, the hook route and the K descale route come from source only.
- I did not measure the per-token noise. The power numbers come from my model. Its inputs (sigma 0.02 to 0.11, design effect 2 to 8, effect equal to sigma squared over 2) are assumptions.
- I did not test the FP8 activation amplification on the real model. `quant_amp.py` uses random rows.
- I did not verify the async scheduler guard that avoids an extra decode step for `max_tokens` 1. I rely on E-54: its `kpos` covers 0 to 32,767 once.
- I did not run the V2 prompt log-probability code. I did not check `torch.topk` tie rules or the bitwise determinism of the FP8 GEMMs.
- I did not read the Gemma 4 `config.json`. I rely on E-50 to E-54 and on the README.
- I did not read the Runpod scripts (`pod_cmd.py`, `pods.py`, `balance.py`) or the ledger code beyond the manifest test.
- I did not read the E2B backend source (TRITON_ATTN) for the impl scale. I read journal Sections 0 and 1 only by search.
- I did not check whether the unquantized lm_head of the checkpoint accepts `head_dtype` (O4).
