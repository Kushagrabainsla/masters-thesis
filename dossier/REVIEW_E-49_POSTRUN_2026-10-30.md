# R-65: post-run review of E-49 (Gemma 4 K=V cache audit, I-182)

Date: 2026-10-30. Reviewer: Sonnet 5.5 (a different model from the author).
VERDICT: CONFIRM_WITH_SCOPE_LIMITS (decision PIVOT and all m31 numbers hold; the m26 64K cause in the record is wrong)

Scope: `experiments/gemma4_kv_audit/` at commit 4ecd1bf. Raw data: `results/e49-h100-20261030/e49/`.
Labels: FACT (I read a source or recomputed a number). INTERPRETATION (my reasoning). HYPOTHESIS (not tested).
I changed no repo file except this one. I created no cloud resource. I did not read `.env`.
My scratch scripts are in the session scratchpad folder `r65/`, outside the repo.
vLLM source lines below are from GitHub `vllm-project/vllm` at commit 3966924d5. This is the nightly commit of the run.

## Verdict
- FACT: The frozen decision PIVOT is correct. I recomputed it from the raw files, not from `summary.json`.
- FACT: Every m31 number reproduces. All validity checks hold. The layer split and the FA4 kernel attribution are right.
- FACT: The cause of the m26 64K loss in README Section 13 is wrong. The vLLM source contradicts the end-token HYPOTHESIS.
- INTERPRETATION: The cause is the `max_tokens` budget of the probe. A replay of the probe code reproduces the record exactly.
- INTERPRETATION: The planned follow-up (`min_tokens` = `max_tokens`) would fail in the same way.
- Four corrections are required before D-76:
  - C1 (record): replace the end-token HYPOTHESIS with the length-cap finding (section "m26 64K cells").
  - C2 (plan): remove `min_tokens` = `max_tokens` from D-76 (b). Size `max_tokens` from the prefill schedule.
  - C3 (code, before any rerun): add a budget guard and log finish reasons. Make M2 "unknown" when no 64K cell exists.
  - C4 (wording): "costs at least 6.5%" and "could save at most 6.5%" mix a lower bound and a ceiling. Quote the ceiling with its bounds.
- Scope limits: the result is a ceiling, not a speed claim. It holds at batch 4 (31B) and batch 8 (26B). It comes from one H100 in one session.
- The m26 evidence is incomplete, not negative. The frozen rule cannot tell these two cases apart.

## Recomputed decision
- FACT: `analyze_e49.py --res .../e49` prints the same PIVOT line as `decision.txt`. The `summary.json` content equals my rerun.
- FACT: `git diff 8edb552 HEAD` is empty for the four code files. The rule text (README Sections 4 to 8) is unchanged. The 65 Pod and local checksums match.
- FACT: I refit with scipy `linregress` and exact t quantiles (2.1318 for df 4, 2.9200 for df 2). I used `cells.json` only.

| Quantity (cell medians, ms) | m31 (31B FP8, B=4) | m26 (26B-A4B, B=8) |
|---|---|---|
| Cells in the fit | 6 | 4 (both 64K cells left out) |
| c, b (ms per context token) | 15.822, 1.0439e-4 | 9.064, 5.356e-5 |
| se(b), df, t | 8.54e-7, 4, 2.132 | 4.61e-6, 2, 2.920 |
| b_low, b_up | 1.0257e-4, 1.0621e-4 | 4.012e-5, 6.701e-5 |
| R^2 | 0.99973 | 0.98543 |
| S, S_low, S_up at 32K | 6.65%, 6.53%, 6.76% | 6.09%, 4.56%, 7.62% |
| S, S_low, S_up at 64K | 11.33%, 11.14%, 11.53% | none |
| S_block_low (32K, 64K) | 6.57%, 11.21% | 5.79% |
| S_prof (32K, 64K) | 6.92%, 11.44% | 6.20% |
| M1: global attention bandwidth / copy | 3,139 / 3,044.5 = 1.031 | 3,059 / 3,044.5 = 1.005 |
| M2: local slope ratio 32K to 64K | 0.976 | not computable (see C3) |

- FACT, rule trace: m31 passes both deciding cells. S_low is above 5% and M1, M2, V4b and V4c are true.
- FACT: m26 has one deciding cell. Its S_low is 4.56%, below 5%. So no m26 cell passes.
- FACT: GO needs a passing cell for every valid model. m26 is valid, so GO is not reachable.
- FACT: S_up is not below 2% in any cell, so KILL is not reachable. The reason text "one model or one cell only" matches README Section 7.
- INTERPRETATION: The label PIVOT is formally right. In content it means "dense model passes, MoE model undecided".

## Checks
| Check (README text) | m31 | m26 | Match |
|---|---|---|---|
| V1 GPU is H100 80GB | NVIDIA H100 80GB HBM3 | same GPU uuid | yes |
| V2 31 to 33 tokens per block | all 32 (18 blocks) | all 32 (12 blocks in the fit) | yes (A1) |
| V3 residual within 5% | max 0.31% | max 1.47% | yes |
| V4a device / wall growth 2K to 32K, at least 0.8 | 3.195 / 3.312 = 0.965 | 1.947 / 1.744 = 1.116 | yes |
| V4b attention share of device growth, at least 0.8 | 0.986 | 0.8045 | yes, m26 by 0.45 points |
| V4c sliding attention within 5% of 2K | -2.1% (32K), -3.5% (64K) | -2.6% (32K) | yes |
| V5 bandwidth at most 1.15 x copy | 1.031 | 1.005 | yes |
| V6 GSM8K at least 50% | 87.5% | 96.9% | yes |
| V7 at most one cell with spread above 2% | max 0.42% | max 1.78% | yes |
| V8 hard throttle in at most 2% of busy samples | 0 of 387 | same session | yes |
| V9 "running in text-only mode" in the log | present | present | yes |
| Copy bandwidth exists | 3,043.4 (start), 3,045.6 (end) GB/s | same | yes |

- FACT: `run.json` shows `is_mm_prefix_lm` false in both models. Both logs show FlashAttention version 4 for all layers.
- FACT: The m31 KV page holds 262,144 B per 32 tokens. That is 8,192 B per token and layer (4 heads x 512 x 2 B x 2). The cache stores K and V separately.
- FACT, layer split: all 11 profiles give exactly 8 x n groups (480 for m31, 240 for m26). The first-kernel match is 1.000.
- FACT: The kernel names agree with the launch position. The hdim-512 FA4 variant (`...FlashAttentionForwardSm90...tensor000o811121081`) occurs only at global positions (80 of 80 in m31, 40 of 40 in m26). The hdim-256 variant occurs only at sliding positions (400 of 400, 200 of 200).
- FACT: The regex matches the FA4 main kernel (its name has "flash" and "attn") and the combine kernel. No prepare kernel exists. The grouped attention time equals the matched total.
- FACT: The KV write `vllm::reshape_and_cache_flash_kernel` is excluded. It takes 0.134 ms per step in m31 and sits outside the groups.
- FACT: Global attention per layer at 32K takes 356 us in m31 (B=4, 4 KV heads) and 357 us in m26 (B=8, 2 KV heads). Both read 1.074 GB, which is 3.0 TB/s. At 64K, m31 takes 691 us for 2.147 GB, which is 3.11 TB/s.
- FACT: Non-attention device time in m31 is 14.013 (2K), 14.059 (32K) and 14.059 ms (64K). It is flat within 0.33%. This is a negative control for clock or thermal drift.

## m26 64K cells
Record (FACT, `cells.json`):
- Both cells passed the scheduler check after prefill: 8 running, 0 waiting, 19,587 free blocks.
- After 5 blocks: 3 running, 0 waiting, 0 preemptions, 31,747 free blocks. The free blocks rose by 12,160. That is 5.04 times the 2,412.5 blocks of one request.
- Step times fell block by block: 11.87, 11.32, 10.75, 9.73, 8.63 ms. Advance was [6, 32] in block 1 and [0, 32] in blocks 2 to 5.
- The second instance shows the same pattern (11.92, 11.38, 10.75, 9.85, 8.67 ms).

Source evidence against the end-token HYPOTHESIS (FACT):
- `vllm/v1/engine/input_processor.py` L505 to L508: `process_inputs` calls `update_from_generation_config` with the generation config and the tokenizer EOS id.
- `vllm/sampling_params.py` L785 to L786: with `ignore_eos` true, the EOS id is not stored. L803: the extra ids go into `_all_stop_token_ids`. They are 106 and 50 if the tokenizer EOS is 1 (`<eos>`).
- `vllm/sampling_params.py` L804 to L807: the extra ids go into `stop_token_ids` only if `ignore_eos` is false. The probe sets it true, so `stop_token_ids` stays empty.
- `vllm/v1/core/sched/utils.py` L105 to L119 (`check_stop`): a request stops on the EOS id, on `stop_token_ids`, or on the length cap. The length cap is `num_tokens >= max_model_len` or `num_output_tokens >= max_tokens`.
- `vllm/v1/core/sched/utils.py` L126: `min_tokens` is tested after these checks. `vllm/v1/worker/gpu/sample/logit_bias.py` L94 to L106 and L242: it only masks stop-id logits. It cannot prevent a length cap.
- Both `generation_config.json` files list `eos_token_id` [1, 106, 50] (Hugging Face raw files). The probe uses no stop string. Only the length cap can end a request.

Cause (INTERPRETATION, strong support):
- `scheduler.py` L793 and L1049: running requests get tokens first. Waiting requests share the rest of one 2,048 budget.
- A 65,536-token prompt needs 32 or 33 steps. Request i leaves prefill at steps 32, 65, 97, 129, 161, 193, 225 and 257 (model result).
- Request 0 decodes during the prefill of the other seven. It holds 226 tokens when the last request gets its first token.
- `decode_probe.py` sets `max_tokens` = 4 + 5 x 32 + 8 + 64 = 236. Request 0 has 10 tokens left: 4 warm steps and 6 steps in block 1. That is the observed 6.
- The next request ends 32 or 33 steps later, in block 2. One more request ends in each later block. After block 5, three requests remain.
- The model gives average live requests per block of 7.16, 6.19, 5.19, 4.19 and 3.19. The step times fall in the same order.
- The record "5 of 8 ended after about 10 tokens" is imprecise. Only request 0 ended near the 10th timed step. The other four ended one per block.
- Replay (FACT): I ran the repo's own `run_cell` against a fake engine with this budget model. The model has no free parameter. Its rules: decode first, FCFS prefill, 2,048 tokens per step, stop at `max_tokens`.
- Replay result: status "preempted", 5 blocks, advance [[6,32],[0,32],[0,32],[0,32],[0,32]], 3 running, 0 waiting, 0 preemptions. This equals the record.
- Same replay with `max_tokens` 480: status ok, all 8 requests alive in 5 blocks.
- Same replay with `min_tokens` = `max_tokens` = 236: the same failure. The fake engine follows `check_stop`, where `min_tokens` does not stop the length cap.
- A content-driven end token would not give equal block counts for two different prompt sets. The record has equal counts.

Label and exclusion:
- FACT: "preempted" is wrong. Preemptions are 0, waiting is 0, and the freed blocks match five finished requests. The right label is "finished by length cap".
- FACT: The exclusion is right. V2 needs 31 to 33 tokens per request per block. The blocks mix batch 8 down to batch 3. They are not batch-8 timings.
- FACT: The label does not change the decision. The code tests only `status == "ok"`. The label matters for the record and for the follow-up.
- INTERPRETATION: This is a probe design defect. R-64 C3 named the 236-token limit but did not size it against the staggered prefill.
- INTERPRETATION: `FakeEngine` in `test_e49.py` prefills all requests at once and has no `max_tokens`. The S5 rehearsal used batch 4 (98 tokens burned at 64K), so it could not show the loss.
- FACT (replay model): other cells ran close to the same fate. Tokens left after the timed phase: m26 32K 14 (with profile) and 22; m26 16K batch 16: 14; m31 64K: 30 and 38.
- A fourth block would have lost the m26 cells. The m26 32K replicate had spread 1.78%, which is 0.22 points below the extra-block trigger.

Other possible causes and tests:
| Cause | Status | Test |
|---|---|---|
| Extra end token from the generation config (author) | Excluded by the source lines above | Log `finish_reason` ("stop" or "length") and the last token id |
| Preemption | Excluded: counters 0, waiting 0 | Log `num_preemptions` of finished requests too |
| Context cap 66,048 | Excluded: 65,536 + 236 is below it | Same finish-reason log |
| Engine error or abort | No such line in either log | Finish reason "error" or "abort" |
| Budget exhaustion (this review) | Reproduced exactly | Guard before timing; rerun with `max_tokens` about 430 to 480 |

Effect on the decision:
- FACT: GO was not reachable after the loss. m26 kept one deciding cell with S_low 4.56%.
- INTERPRETATION: With a valid 64K cell, the fit predicts T = 12.6 ms and S = 10.5% at 64K. Block 1 shows 11.87 ms at 7.2 live requests. This is consistent. It is not a measurement.

## Attacks
- Profile window (FACT): The profile has 8 steps and serves only attribution. Its wall time is 1.6 to 2.4 times the block median. The rule uses unprofiled medians, so the R1 fix works.
- Profile window (FACT): In m31, device time equals wall time within 0.7%. The replicates, timed after three profiles, agree with the first cells within 0.13% (64K) and 0.03% (32K).
- Async scheduling (INTERPRETATION): The pipeline tail adds a constant fraction to T and b alike. S is a ratio, so it does not change. In m31 no CPU time is exposed (idle +0.08% at 32K).
- Async scheduling (FACT): In m26 the GPU idles 7.1% (2K) and 4.1% (32K). So wall growth (1.744 ms) is below device growth (1.947 ms). The wall value is the user-visible one.
- CUDA graphs (FACT): The logs show FULL graph capture. The kernel runs at the HBM roofline at 32K and 64K. So the graph hides no split-count loss. A compact kernel in a graph is untested.
- FA4 JIT warning (FACT): The CuTeDSL warning appears once per model, at the first GSM8K request (m31 20:11:34, m26 20:20:08). It comes before every cell.
- FA4 JIT warning (FACT): m26 also shows a Triton `fused_moe_kernel` JIT at 20:20:22, between the 2K and 8K results. No timed block shows a stall. The largest spread is 1.78%. The FA4 kernel names are identical at 2K, 32K and 64K.
- MoE routing (FACT): `fused_moe_kernel` grows 0.38 ms from the 2K to the 32K profile. Attention explains 80.45% of the device growth, 0.45 points above the V4b limit.
- MoE routing (FACT): m26 residuals are -1.25%, +1.47% and -0.13%. The standard error of b is 8.6% (df 2), so the m26 bound is wide.
- Throttling (FACT): 0 of 387 busy samples show a hard bit. The software power-cap bit shows in 191 samples (49%). The SM clock ranged from 1,470 to 1,980 MHz (median 1,755 in the m31 window).
- Throttling (INTERPRETATION): Telemetry is 1 Hz and a timed block lasts about 1 s, so it cannot resolve a block. The flat non-attention time shows no clock effect that grows with L.
- Byte model (FACT): Two models with different head counts give the same per-layer time for the same bytes (356 and 357 us for 1.074 GB). The data validate the byte model.
- S as a ceiling (INTERPRETATION): It holds for step time at fixed batch. The kernel reads at 3.14 TB/s, 94% of the 3.35 TB/s vendor peak. Only fewer bytes can help.
- S as a ceiling (INTERPRETATION): Three limits remain. The fixed kernel part (37.5 us per layer at 2K) does not shrink. The compact QK adds 25% MACs. A new kernel must keep the roofline with two data streams.
- S as a ceiling (INTERPRETATION): S does not bound a capacity gain. More capacity can raise throughput beyond S.
- Batch dependence (INTERPRETATION, FACT inputs): The model T = 14.48 + 0.336 B + 2.605e-5 B L ms fits all 10 m31 cells within 0.5%. Leave-one-out predicts the batch-8 16K cell within 0.28%.
- Batch dependence: At 32K the model gives S = 2.0% (B=1), 6.65% (B=4), 10.7% (B=8) and 12.8% (B=11, pool limit 11.9). At 64K it gives 3.9% (B=1), 11.3% (B=4) and about 14% (B=6).
- Batch dependence: The pool limit at 64K is 6.7 by measured blocks. The engine log says 4.39 at 66,048 tokens. At 16K the model gives 6.2% (B=8) and 10.5% (B=19).
- Batch dependence: Batches above 8 are extrapolation. At the capacity limit S is about 10% to 14%, about twice the batch-4 value at 32K.
- FP8 KV (INTERPRETATION): An ideal FP8 KV cache halves all KV bytes, sliding layers too. For m31 at batch 4 it saves 13.2% (32K) and 18.7% (64K). The compact cache saves 6.7% and 11.3%.
- FP8 KV (INTERPRETATION): The compact cache on top of FP8 adds 3.8% (32K) and 7.0% (64K). I assumed the FP8 kernel keeps the roofline and has no dequant cost. FP8 KV is lossy; the compact cache is exact.
- Statistics (FACT): Dropping any one cell moves S(32K) within 6.60% to 6.69% and S(64K) within 11.25% to 11.39%.
- Statistics (FACT): Two-point slopes (32K to 64K 1.019e-4, 2K to 32K 1.078e-4) give S(64K) of 11.1% to 11.7%. There is one Pod, one GPU and one session, so no between-GPU variance is measured.
- Analysis code (FACT): The line `m2 = local is None or ...` makes M2 true when no 64K cell exists. It did not matter here, since m26 had no passing cell. It would let an untested model pass the gate.
- Rule design (INTERPRETATION): A lost deciding cell blocks GO and shows as PIVOT. A new preregistration needs an INCONCLUSIVE outcome for "deciding cell lost".

## Prior work pass 2
Queries (GitHub `gh search`, sorted by update, 2026-10-30):
- vllm-project/vllm: `k_eq_v`, `attention_k_eq_v`, `qkk_proj`, `tied KV cache`, `store K once`, `alias K and V cache gemma`, `gemma4 MLA backend global layers`, `gemma4 halve KV cache`, `KV cache dedup K=V`, `gemma4 head_dim 512 KV cache`, `gemma4 kv cache shared`, `gemma 4 global layer KV cache memory`. Also all `gemma4` and `gemma 4` items created since 2026-10-28.
- sgl-project/sglang, NVIDIA/TensorRT-LLM, ggml-org/llama.cpp, flashinfer-ai/flashinfer: `k_eq_v`, `gemma4 KV cache`, `gemma 4 kv shared K V`, `gemma4`, `head_dim 512`, `Vcur Kcur gemma4`, `gemma 4 V cache`, `gemma4 attention k v`.
- Code search for `attention_k_eq_v` and `k_eq_v` in llama.cpp, vLLM and SGLang. File read of vLLM main 59d58fd26 (2026-10-30): `gemma4.py` and `kv_cache_interface.py`.
- Web, 3 queries: `Gemma 4 K=V attention KV cache store once compact layout inference engine`; `attention_k_eq_v Gemma 4 global layers KV cache memory vLLM SGLang halve`; `arXiv tied key value cache p-RoPE grouped tied attention kernel decode GQA exact 2026`.
- Pages read: HF paper 2609.13285 and github.com/zoecarver/gemma4.

Result (FACT): I found no engine, PR, issue or paper that stores the Gemma 4 K=V cache once. R-64 stays valid.
| Hit | Date | Same mechanism? |
|---|---|---|
| vLLM main `gemma4.py` L440, L555 | 2026-10-30 | No. K weights load into both K and V slots. No commit to this file since 2026-10-28 |
| vLLM PR #41944 (open, comment 10-29) | 2026-05-07 | No. Still only `gemma4.py`. Drops the V projection weights |
| vLLM PR #61267, #62034 (LoRA), #64247 (QK-norm and RoPE fusion), #64774 (RedHatAI spec decoders) | 2026-10-12 to 30 | No |
| SGLang PR #46965 (open): FA4 SM90 FP8 e4m3 KV for head_dim 512 (Gemma-4) | 2026-10-28 | No. This is the FP8 competitor path. R-64 did not list it. No speed data yet |
| FlashInfer PR #8057 (NVFP4 KV launcher), #5720 (H512 FP8-Q/NVFP4-KV), #6445 (merged 10-29) | 2026-09 to 10 | No. Competitor path |
| llama.cpp PR #31406, #31356 | 2026-10-29 and 30 | No. Cross-layer KV sharing in E2B and E4B |
| TensorRT-LLM PR #21339, #18537 | 2026-10-30 | No. Config and weight mapper |
| arXiv 2609.13285 Grouped Value Attention | 2026-09-16 | Same family, not the same mechanism. It caches values, rebuilds keys with a learned map absorbed in the query, and adds a RoPE channel. It trains from scratch. The authors say custom kernels are in test and a release is planned. R-64 missed it |
- FACT: vLLM main `AttentionSpec` has a `head_size_v` field (`kv_cache_interface.py` L560, L590). A prototype may use this hook.
- INTERPRETATION: Novelty stays low (R-64). The open item is the engine mechanism and its measured effect. The FP8 competitor gained one more engine path.

## What the result supports
- FACT: On 1 x H100 SXM with vLLM nightly 0.33.0rc1.dev229, FA4 and a BF16 KV cache, Gemma 4 31B FP8 at batch 4 shows this. The step time grows 0.1044 ms per 1,000 context tokens.
- FACT: Attention explains 98.6% of that growth. It is the global layers, and they stream K and V at the HBM roofline.
- INTERPRETATION: The ceiling for a compact K=V cache is 6.65% at 32K (one-sided 95% bounds 6.53% and 6.76%). At 64K it is 11.33% (11.14% and 11.53%). The real saving at batch 4 is at most this.
- FACT: For 26B-A4B at batch 8 and 32K, the ceiling is 6.09% (bounds 4.56% and 7.62%). Nothing is known at 64K.
- INTERPRETATION: The dense model alone meets the GO criteria with margin. The frozen rule does not allow a one-model GO when both models are valid.

It does not support:
- Any real speedup. No compact kernel exists, and a new kernel must keep the roofline.
- Any statement on accuracy (BF16 error of the compact row, R-64), on the 1.6 times capacity gain, or on the page-size plan.
- Any claim on other batch sizes, GPUs, TP above 1, speculative decoding, prefix caching, an API server, or FP8 KV.
- Any claim that m26 fails. The 64K loss is a probe fault.

## Recommendations
These are recommendations only. D-76 belongs to the user.
1. Free, now: fix the record (C1, C2, C4). Edit README Section 13 and the journal lines for I-182, E-49 and D-76. Keep the exclusion.
2. Free, before any rerun (C3): patch the probe and add tests.
   - Set `max_tokens` = min(`max_model_len` - L, (B-1) x ceil(L / (chunk - B)) + 4 + 32 x max_blocks + profile steps + 32). For L = 65,536 and B = 8 this is 435. The cap is 512.
   - Before timing, stop the cell with status "budget" if any request lacks the tokens for the planned blocks.
   - Log `finish_reason`, token count and `num_preemptions` for every request, finished ones too.
   - Make M2 "unknown" (a PIVOT reason) when no 64K cell exists.
   - Add a test with a fake engine that has the 2,048 token budget and the `max_tokens` stop. Use my replay as a base: scratchpad `r65/replay_probe.py` (the model is in `r65/sim_budget.py`).
3. Cost about $1.5 to $2.0 (18 to 30 min on 1 x H100 at $3.99/h). Rerun m26 only, with the full 6-cell slope, chunk 2,048 and the fixed budget. Write amendment A2 first and keep the rule. Set the AUTO_STOP delay to 60 s. This needs the user's spend approval and a live balance check.
4. Optional, in the same session, about +$0.7 to $1.0 and its own preregistration (E-50). Run m31 at batch 8 with 32K and at batch 5 with 64K. Batch 6 at 64K leaves a thin memory margin at the prefill peak.
5. Also in E-50: run m31 with an FP8 KV cache at batch 4. First read the vLLM source. SGLang PR #46965 says FA4 on SM90 has no FP8 KV path. These cells turn the batch and FP8 estimates above from INTERPRETATION into FACT.
6. Prototype. The engineering effort is not costed. GPU time is about 1 to 2 h ($4 to $8). Start only if the stacked ceiling on FP8 KV stays at or above 5% at the pool-limit batch.
   - Step 1 is a kernel microbenchmark on synthetic KV (batch 4, 32K and 64K) against the FA4 hdim-512 kernel.
   - Gate: at least 85% of the byte-model saving, plus the accuracy gate and the page-size plan (O5).
7. Stop option, $0: stop I-182 as a main direction if step 5 shows a stacked ceiling below 5%. Keep the byte model, the probe, the roofline finding and the length-cap lesson as outputs.
8. If the user wants no more spend, report E-49 as "PIVOT. Dense ceiling 6.7% (32K) and 11.3% (64K) at batch 4. The MoE model is incomplete because of a probe fault."
