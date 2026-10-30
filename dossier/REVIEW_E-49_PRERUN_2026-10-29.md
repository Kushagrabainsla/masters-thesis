# R-64: pre-run review of E-49 (Gemma 4 K=V cache audit, I-182)

Date: 2026-10-29. Reviewer: Sonnet (a different model from the author).
Scope: `experiments/gemma4_kv_audit/` at commit 56bb221.
Labels: FACT (I read a source or ran code). INTERPRETATION (my reasoning).
I changed no repo file except this one. My scratch copies of the sources are outside the repo.
Sources: vLLM at 3966924d (the nightly commit) and at main 4b47db92, HF transformers 420dca92, llama.cpp 36a0e2ff.

## Verdict
AMEND_THEN_RUN.
The mechanism is correct. The question is worth one paid run. The decision rule is mostly well defined.
Nine amendments are required first. R1 and R2 can ruin the whole paid run.
Apply R1 to R9 before the S5 rehearsal, so the rehearsal tests the final code.
- R1 (blocker): V4a and the idle share use profiler wall time. A false INVALID is likely.
- R2 (blocker): at the default prefill chunk, the 31B cells at 32K and 64K can preempt requests.
- R3 (major): "text only" is not text only. Video stays enabled.
- R4 (major): the rehearsal GPU probably uses a different attention path than the H100.
- R5 (major): the "95% bound" does not give 95% coverage.
- R6 (major): the sliding-layer negative control is promised but not coded.
- R7 to R9 (minor): positive control, salvage of partial runs, upload list.

## 1. Mechanism check
- FACT: V = RMSNorm(k_raw) with no weight. K = RoPE(k_norm_weight * RMSNorm(k_raw)).
- FACT: HF sets `value_states = key_states` (the raw k_proj output) when `v_proj` is None. Then `k_norm` (learnable `x * w`, fp32) and RoPE act on K. `v_norm` has `with_scale=False`.
- FACT: vLLM loads k_proj into both the K and V shards of `qkv_proj`. Its `v_norm` is `RMSNorm(has_weight=False)`.
- FACT: llama.cpp (`Vcur = wv ? ... : Kcur`), the DeepMind JAX code and vLLM PR #41944 (`v_norm(k_pre)`) agree.
- FACT: the rotated dims are exactly 0..63 and 256..319. HF: `rope_angles = int(0.25 * 512 // 2) = 64`. The other 192 pairs have zero frequency. `rotate_half` pairs dim i with dim i+256.
- FACT: vLLM zero-pads `inv_freq` the same way. llama.cpp uses freq factors [1.0] x 64 and [1e30] x 192. TRT-LLM uses the same formula. Four code bases agree.
- FACT: only global layers use K=V (`attention_k_eq_v and not is_sliding`). Sliding layers keep their own v_proj.
- FACT: 31B has 10 of 60 global layers with 4 kv heads. 26B-A4B has 5 of 30 with 2 kv heads. Global head_dim is 512. The sliding window is 1,024.
- FACT: nothing breaks the layout. The attention scale is 1.0 in HF, vLLM, TRT-LLM and llama.cpp.
- FACT: `Gemma4TextConfig` has no `attn_logit_softcapping` field. Neither checkpoint sets it. Only `final_logit_softcapping` (30) exists, and it acts on the logits.
- FACT: the k_norm weight has `head_dim` entries and is shared by all kv heads. One vector w per layer is absorbed into the query.
- INTERPRETATION: 37.5% equals 1 - (512 + 128) / 1,024. The Gemma 4 report number is therefore this layout. The report gives no memory layout.
- FACT: I re-ran `kv_layout_check.py`. The JSON is identical (float64 error 3e-15 to 8e-15). The 9 tests pass (11 s).
- INTERPRETATION: the layout is exact in real arithmetic, not in BF16 bits. HF stores K_N = bf16(w * n). The compact row rebuilds w * bf16(n).
- INTERPRETATION: my own check used the HF casting order, heavy-tailed keys and w = exp(N(0,1)). Over 12 trials the median error against float64 was 1.1% (standard) and 1.4% (compact). The maxima were 3.9% and 5.3%.
- INTERPRETATION: so "no larger than the standard layout" (README Section 2) holds on its data only. A prototype needs an accuracy gate. E-49 is not affected.

## 2. Prior work, pass 2
Fresh queries: "k_eq_v", "attention_k_eq_v", "gemma4 kv", "tied KV", "k equals v", "grouped tied attention", "shared K and V cache", "head_dim 512", "FA4 SM90".
Repos: vLLM, SGLang, TRT-LLM, llama.cpp, FlashInfer, gemma.cpp, LiteRT-LM, litert-torch, google-deepmind/gemma. Plus web and arXiv.
GitHub search hit the rate limit on two queries (FlashInfer "k equals v", llama.cpp "grouped tied attention"). I covered those repos with other queries and with code search.
Result: I found no engine or paper that stores the Gemma 4 K=V cache once. This is not proof of absence. Private Google serving code is invisible.

| Hit | Date | Same mechanism? |
|---|---|---|
| Gemma 4 report, arXiv 2607.02770 | 2026-07-02 | Design intent and the 37.5% number. No layout, no engine |
| Kayyam et al., arXiv 2606.04032 (ICML 2026) | 2026-06-01 | Source of K=V. Cache accounting only (50%). No engine layout |
| Zadouri, Strauss, Dao (GTA), arXiv 2505.21487 | 2025-05-27 | Same family: tied KV plus a separate RoPE half, about half the cache. Trained from scratch. Kernel benchmarks cover only MLA and GLA. The repo has modeling files, no kernel |
| TransMLA, arXiv 2502.07864 | 2025-02 | Different: lossy GQA to MLA conversion with training |
| Slim attention, arXiv 2503.05840 | 2025-03 | Different: exact K-only cache for MHA, V recomputed |
| vLLM PR #41944 (open) | 2026-05-07 | No. It drops the duplicate V weights. The cache is unchanged |
| vLLM PR #60674, #53115, #53918 (open) | 2026-08 to 10 | No. Fused QK-norm and RoPE kernels. K and V are both written |
| vLLM PR #38835 (merged) | 2026-04-02 | No. Enables head_dim 512 on SM90 with FA4. This is the E-49 baseline path |
| vLLM PR #60372, #62117 (open), #53175 (merged); FlashInfer PR #3652, #5720, #6445 | 2026-06 to 10 | No. FP8 or NVFP4 KV at head_dim 512. Same bytes, lossy, can stack. The real competitor |
| vLLM PR #60317 (open) | 2026-10-06 | No. Triton attention tuning for head_dim 512 |
| vLLM issue #49878 (open) | 2026-07-26 | No. Hybrid KV sizing for Gemma 4. Used in Section 5 |
| SGLang `gemma4_causal.py`; PR #40964, #28199 | file read 2026-10-29 | No. RadixAttention takes separate k and v. The PRs fix LoRA |
| TRT-LLM `modeling_gemma4.py`; PR #16797; issue #16867 | file read 2026-10-29 | No. The weight mapper copies k_proj to v_proj. Standard `KVCacheLayerSpec`. The PRs fix quant loading |
| llama.cpp `src/models/gemma4.cpp`, `conversion/gemma.py` | file read 2026-10-29 | No. Kcur and Vcur go to `build_attn`. No issue or PR on K=V. ik_llama.cpp #1607 is about sliding-layer size |
| google-deepmind/gemma `_modules.py`; litert-torch `patch.py` | repos pushed 2026-10-28/29 | No. `init_cache` makes separate 'k' and 'v'. The on-device export updates both. gemma.cpp has no Gemma 4 code |

INTERPRETATION: novelty is low. The report states the saving. GTA states the layout idea. The open item is the engine mechanism and its measured effect. The journal already says this.

## 3. Design findings
- D1 FACT (backend). `Gemma4Config` sets `flash_attn_version = 4` for all layers when FA4 is supported. Otherwise it forces TRITON_ATTN.
- D1 FACT. `_is_fa4_supported` accepts SM 9.x, 10.x and 11.x, if the cute package exists. On SM90 the FA backend supports full CUDA graphs.
- D1 FACT. The rehearsal GPU (RTX 4090, SM89) cannot run FA4. It runs Triton. The README says only "default attention backend".
- D1 INTERPRETATION. The H100 kernel names, FA4 JIT and graph path are first used in the paid run.
- D2 FACT (text only). `get_supported_mm_limits` of Gemma 4 always lists "video". A modality with no limit gets 999. vLLM enters text-only mode only when all limits are 0.
- D2 FACT. With image 0 and audio 0 the engine stays multimodal. The encoder is profiled and `is_mm_prefix_lm` stays true.
- D2 FACT. On SM90 with mm_prefix, the first backend in the priority list is the `TRITON_FLASH_ATTN` composite. Its causal route is FA. The vLLM code comment names this case ("image=0 but video allowed").
- D3 FACT (V4a). `decode_probe.py` measures `wall_ms` around the whole `with torch_profile(...)` block. It includes profiler start and stop. `analyze_e49.py` takes the wall growth from these values.
- D3 INTERPRETATION. The first profile of a process pays CUPTI start-up. That profile is the 2K cell. So wall(2K) can exceed wall(32K), and the growth is negative.
- D3 INTERPRETATION. The true 2K to 32K growth is about 3 to 4 ms per step. A jitter of 30 ms in 8 steps flips V4a. The `idle_share` column is wrong for the same reason. The stub hides this, because it has no profiler cost.
- D4 FACT (attribution). README Section 8 promises that sliding layers do not grow with L. The code has no such check.
- D4 FACT. `kernels.json` is aggregated by kernel name. Sliding and global launches cannot be split. V4b counts both.
- D4 INTERPRETATION. FA4 sizes the splits of a local layer from the window (`seqlen_k_loaded`). So the sliding cost is probably flat. This is unproven.
- D5 FACT (statistics). `fit()` is correct. Slope and standard error equal scipy on 200 random sets. The t table is within 0.2% of exact values.
- D5 INTERPRETATION. `fit()` treats 12 to 20 blocks as independent. Blocks of one cell share requests and page placement. The independent units are the 4 cells.
- D5 INTERPRETATION. I simulated a cell-level sd of 0.3 ms with 4 blocks per cell. The nominal 95% lower bound covered the true slope in only 74% of trials.
- D5 INTERPRETATION. MoE routing makes a cell effect likely for 26B-A4B. Different prompts route to different experts, so weight bytes per step differ by cell.
- D6 INTERPRETATION (V3). The 3% limit is 0.5 to 0.7 ms. Split-KV decode efficiency changes with L. Mild curvature can fail V3 and mark a good model INVALID.
- D7 INTERPRETATION (0.375). S assumes time follows bytes. A kernel at 60% of copy bandwidth (the M1 limit) may spend 40% of its time on work that fewer bytes do not remove. Then S is too high by up to 1.67 times.
- D7 INTERPRETATION. The compact QK dot has 640 dims (+25% QK MACs). So S is an upper bound for the saving. The README calls only the bound "conservative". Say so in the result.
- D8 FACT (rule). I traced GO, KILL, PIVOT, INVALID, one model and V8. Every input has an outcome.
- D8 FACT. Gaps: M1 is one-sided, so an impossible bandwidth passes. A missing microbenchmark gives "KERNEL-BOUND" by accident. KILL uses the point value, but GO uses the lower bound.
- D9 INTERPRETATION (power). The byte model saves 8.9% (32K) and 14.3% (64K) for 31B, and 6.5% and 11.1% for 26B-A4B.
- D9 INTERPRETATION. Real step time has more overhead than bytes. So the 32K cells sit near or below the 5% bar, and a GO needs both 64K cells.
- D9 INTERPRETATION. KILL (S below 2% in all cells) is almost unreachable. The likely non-GO outcome is PIVOT. Predeclare this.
- D10 FACT (other L-dependent costs). I checked what else grows with L.
- D10 FACT. The sliding block-table gather runs on the GPU and takes microseconds. The scheduler free and admission walks are O(L) per request, not per step (PR #61724).
- D10 FACT. The KV write is O(B). The async bubble after each `sync()` is constant.
- D10 FACT. Prefill leaves nothing after the first-token loop and 4 warm steps.
- D10 INTERPRETATION. The FA4 `num_splits_heuristic` gives about 8 splits for 16 batch-head tiles, so the combine step is small.
- D10 INTERPRETATION. In the dense model only the global attention grows with L. In the MoE model the expert routing adds a cell effect (D5).
- D11 FACT (async). Async scheduling is on by default (`UniProcExecutor.supports_async_scheduling` is True; two batches in flight). Each `step()` schedules one batch and returns the oldest. A request advances 1 token per call after the first call. V2 (31 to 33) is right.

## 4. Code findings
- C1 FACT: `add_request(request_id: str, prompt, params)`, `step() -> list[RequestOutput]` and `abort_request(list[str])` match the pinned `llm_engine.py`.
- C1 FACT: `{"prompt_token_ids": [...]}` is a valid prompt. `RequestOutput.request_id` is the external id (output_processor lines 441 and 460), so `out.request_id in lens` works.
- C1 FACT: the output kind is cumulative and `stream_interval` is 1. So `len(token_ids)` grows by 1 per call. `SamplingParams(detokenize=False, ignore_eos=True, seed=0)` is valid.
- C1 FACT: `LLM.llm_engine`, `get_tokenizer()` and the `cache_config` fields used by `kv_record` exist.
- C2 FACT: `abort_request` frees the blocks at once, or after the one in-flight step settles (`deferred_frees`). The 3 extra `step()` calls cover this. `step()` returns at once when no request is left.
- C3 INTERPRETATION: the 100,000-step guard in the prefill loop does not stop a cell with a preempted request. The loop ends when the other requests finish (236 tokens). Then V2 fails (see R2).
- C4 FACT: `run.json` has status "started" after GSM8K and "ok" only at the very end. A timeout after the slope cells makes the model INVALID ("run not ok"). The waste cells run after the slope cells.
- C5 FACT: `probe()` runs `rm -rf "$out"` on a rerun. This deletes partial raw data (PROTOCOL principle 7).
- C5 FACT: the new-work cap checks only the start time. A model that starts at 54 min is cut by the 65 min watchdog.
- C6 FACT: the analysis imports `analyze_e44`, `analyze_e40` and `analyze_a2` from sibling folders. README Section 11 does not list the folders to upload. A missing folder gives an ImportError in `decision.txt`. The raw data stays safe.
- C7 FACT: tests cover the analysis paths, not `run_cell`. PROTOCOL S3 asks for a stub session with caps and a failed run. Only the happy path and a rerun skip exist.
- C7 FACT: no test uses real kernel names, an inflated profile wall or a preempted cell.
- C8 FACT (minor): the GSM8K prompts already contain `<bos>`, and vLLM adds a second one. V6 at 50% is lenient enough. The dataset tokenizer is not pinned to the model revision.

## 5. Budget check
Inputs (FACT):
- H100 memory is 81,559 MiB. The budget at 0.92 is 73.3 GiB.
- m31 weights are 33.3 GB = 31.0 GiB. m26 weights are 51.6 GB = 48.1 GiB (HF file sizes).
- Hybrid pool: unified page 256 KiB (31B) and 128 KiB (26B). Sliding block 16 tokens, global block 32 tokens. 5 sliding groups and 1 global group.
- vLLM issue #49878 reports the same page and groups for a 50/10-layer model. One pool block is 2.5 MiB (31B) or 0.625 MiB (26B).
- The default `max_num_batched_tokens` is 16,384 on H100. Two batches are in flight, so 32,768 tokens.
- `remove_skipped_blocks` frees on the processed-token basis. A prefilling request holds (1,023 + in-flight tokens) / 16 + 1 blocks in each sliding group.
- `allocate_slots` returns None (preemption) when the pool is short.

INTERPRETATION: non-KV memory is activations 3 to 6 GiB, graphs 1 to 2.5 GiB and other 0.7 to 1.2 GiB. KV available is 32.6 to 37.6 GiB (m31) and 16.5 to 21.0 GiB (m26).

| Need (GiB) | m31, chunk 16,384 | 8,192 | 4,096 | 2,048 | m26, 16,384 | 2,048 |
|---|---|---|---|---|---|---|
| One max-length request (startup check) | 30.8 | 18.3 | 12.1 | 9.0 | 7.7 | 2.2 |
| Steady decode, 64K cell | 23.3 | 23.3 | 23.3 | 23.3 | 11.6 | 11.6 |
| Peak while the last request prefills, 32K | 38.2 | 25.7 | 19.5 | 16.4 | 12.9 | 7.4 |
| Peak while the last request prefills, 64K | 48.2 | 35.7 | 29.5 | 26.4 | 17.9 | 12.4 |

- INTERPRETATION: m31 at the default chunk passes the startup check by 2 to 7 GiB. The 64K peak (48 GiB) and the 32K peak (38 GiB) exceed the pool. Expect preemption or serialization and a failed V2.
- INTERPRETATION: at chunk 2,048 the m31 64K peak is 26.4 GiB. The margin is at least 6 GiB. m26 fits at every chunk, but the margin is thin at 16,384.
- Time (INTERPRETATION): E-46 set up 24 GB of models in 138 s. So 85 GB takes 8 to 10 min with pip and GSM8K.
- Cold engine start is 5 to 9 min per model. E-46 warm-ups took about 4 min for an 8B model.
- Prefill of 4 x 64K tokens (m31) is about 1 min at 4 to 5K tokens/s. Prefill of 8 x 64K tokens (m26, 4B active) is about 40 s. The other slope cells add about 1 min.
- Blocks and profiles are about 2 min per model.
- The session takes about 40 to 50 min, plus 3 to 5 min of Pod boot. R5 and R6 add about 3 min per model. The README figure of 40 min is slightly low.
- Caps (55 min new work, 65 min watchdog, 70 min boot watchdog, 25 min `PROBE_TIMEOUT`) are consistent with this estimate. AUTO_STOP is the E-46 code, which ran well. R8 fixes the start rule for the second model.
- Cost at $3.99/h for 70 min is $4.66, plus the 5 min AUTO_STOP wait. The $6 cap holds.

## 6. Required amendments (write them as A1 in the README; each needs a test)
R1 (blocker). Fix V4a and the idle share.
- Take the wall growth from the unprofiled block medians: dw = median(step_ms at 32,768) - median(step_ms at 2,048), from `cells.json`.
- Keep the device growth dd from `kernels.json`. V4a = dw > 0 and dd / dw >= 0.8.
- Set `idle_share = 1 - device_ms_per_step / median(step_ms of the same cell)`.
- Add a test where the profile `wall_ms` has a large constant and noise.

R2 (blocker). Remove the preemption risk.
- Set `max_num_batched_tokens=2048` in `LLM(...)`. Add `--max-num-batched-tokens` (default 2048) and record it.
- Do R3 first. With video on, vLLM may refuse a chunk smaller than one video item (INTERPRETATION).
- Record `kv_cache_size_tokens`, `kv_cache_max_concurrency`, `num_gpu_blocks` and `block_size` in `run.json`.
- After each prefill loop, read `sched = engine.engine_core.engine_core.scheduler`. Require `len(sched.running) == B`, `len(sched.waiting) == 0` and `num_preemptions == 0` for the cell requests.
- If a check fails, mark the cell "preempted", abort it, skip the timing and go on.
- Put the 2,048 numbers of Section 5 into the README budget.

R3 (major). Make the engine text only.
- Use `limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0}` (or `language_model_only=True`).
- Require the log line "running in text-only mode". Store `is_mm_prefix_lm` and the backend lines (including the "FA4 for all layers" line) in `run.json`.
- Update README Section 4.

R4 (major). Cover the H100 attention path.
- Write the expected backend per GPU in the README: H100 = FA4 (or the composite if video stays on), RTX 4090 = Triton.
- Add a table-driven test of the `ATTN` regex. Real H100 FA3 names from the E-46 profiles: `cutlass::device_kernel<flash::...FlashAttnFwdSm90...>`, `flash::FlashAttnFwdCombine`, `flash::prepare_varlen_num_blocks_kernel`. Add the FA4 names from the first H100 profile.
- Add `kernel_unified_attention_2d/3d` and `reduce_segments` (Triton). Add `vllm::reshape_and_cache_flash_kernel` as a copy kernel, not attention. Move `copy_cache` before `attention` in `CLASSES`.
- If no kernel in the 32K profile matches `ATTN`, V4b is "unknown". The outcome is then "PIVOT (attribution unknown)", not "KERNEL-BOUND".
- Mark the FA4 items in the S5 VERIFY list "not testable on the 4090". Accept them as residual risk.

R5 (major). Make the 95% bound honest.
- Add a second instance of the 32,768 and 65,536 cells (new seed, a `rep` field, new request ids) for both models.
- Compute the rule bound from cell medians with df = cells - 2 and the one-sided t value (df 4: 2.132). Report the block-level bound too.
- Change the README text to "cell-level 95% lower bound".

R6 (major). Build the sliding-layer control.
- In each profile, save the attention-class kernel events (name, start, duration, grid). Do it for the 2K, 32K and 65,536 cells. Add the 65,536 profile at the slope batch.
- Assign launches to layers by order inside a step. Global layers are every sixth launch group (layers 6, 12, ... in both models).
- New check V4c: sliding attention ms per step at 32K is within 5% of 2K. If not, the slope is not the global attention.
- Report S_prof = 0.375 * global attention ms per step / T(L) next to S. Flag a cell if S and S_prof differ by more than 25%.

R7 (minor). Enforce the positive control.
- M1 passes only if 0.6 * copy_bw <= bw_global <= 1.15 * copy_bw. Above 1.15 is INVALID (the byte model is wrong).
- A missing copy bandwidth is INVALID, not KERNEL-BOUND.
- State in the README that S is an upper bound for the saving.

R8 (minor). Salvage partial runs and fix the caps.
- Write `run.json` status "ok_slope" after the slope cells and their profiles. Accept "ok" or "ok_slope" in `done_ok` and in the analysis. Run the waste cells last.
- Use `capped 900` before the second model.
- Replace `rm -rf "$out"` by a move to `$out.failed.<time>`.
- Wait until `nvidia-smi` shows the GPU free before the second model.

R9 (minor). Fix the upload list.
- List the folders in README Section 11: `gemma4_kv_audit`, `safedispatch_testA2`, `runpod`, `fp8_quant_fusion`, `nvfp4_dispatch`, `safedispatch_testA`, `ledger`.
- Add an S3 test that runs `analyze_e49.py` from a clean copy with only those folders.

## 7. Optional suggestions
- O1: replace the V3 gate by a local-slope gate (slope from 32K to 64K within 25% of the all-cell slope). Widen the residual limit to 5%.
- O2: use the upper bound for KILL (S upper below 2%), as GO uses the lower bound.
- O3: add one exploratory cell with the same cache tokens at a larger batch (m31 at B=16 and L=16K). It shows that S depends on B * L, which matters for real serving.
- O4: name the competitor in the README. FP8 KV cuts 50% of all KV bytes, is lossy and ships in the nightly. A GO should be compared with it and with the stacked case.
- O5: for the prototype plan, note the page-size problem. The compact row is 640 values x 4 heads x 2 B = 5,120 B per token and layer. The sliding page is 16,384 B per token and layer. The hybrid allocator needs equal page bytes, which needs odd block sizes (80 and 256 tokens). Plan this before you claim 1.6 times capacity.
- O6: add a mock-engine test of `run_cell` (fake `add_request`, `step`, `abort_request`). The loop logic then runs on the Mac before the rehearsal.
- O7: a 10 min H100 smoke run of the E2B or 26B model (about $0.7) would test the FA4 path, the kernel names and the guard. It is optional, because the session itself is the next test.
- O8: pin the GSM8K tokenizer to the model revision and strip the duplicate BOS.
