# R-79: pre-run review of E-56 (output correctness of the compact K=V cache, 16-bit Gemma 4 31B, one A100 80GB)

VERDICT: AMEND_THEN_RUN. There is one blocker. B1: the V8 reference-NLL limit of 1.5 nats per token is likely to stop a healthy baseline, and the run then ends INVALID after about $1.1 is spent. Five fixes are cheap and should go in the same amendment. The splice, the scoring alignment, the token counts and the A100 path are sound by source and by test.

Date: 2026-11-04. Reviewer: R-79 (Sonnet 5.5, a different model from the author). Stage: S2 design review of the preregistration frozen at commit f1b3502.
Labels: FACT means I read a file or a source, ran code, or computed a number. INTERPRETATION means my reasoning. HYPOTHESIS means not tested.

Rules kept:
- I wrote this one repo file. I changed no other file. I used no GPU, no Runpod tool and no `.env`.
- I made read-only `gh api` calls to vllm-project/vllm at 3966924d5 (`triton_attn.py`, `triton_unified_attention.py`, `model_executor/models/config.py`, `models/gemma4.py`, `fa_utils.py`, `vllm_flash_attn/flash_attn_interface.py`, `v1/worker/gpu/sample/prompt_logprob.py`, `config/vllm.py`).
- I ran the test file once. One command chained `cd` with another command. This broke the shell rule. It changed no file. I used absolute paths after it.
- Scratch folder: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/20255ed9-038a-44e5-8a60-f7885a8c54ee/scratchpad/` (`power.py`, the sources, `pytest_out.txt`).

## Key findings
1. FACT: The test file gives 13 passed and 1 failed in 55 s. The failure is `test_time_cap_cuts_stock8_and_keeps_the_decision`. It depends on the speed of the machine (F5).
2. FACT: On SM80, vLLM forces `TRITON_ATTN` for all 60 layers. `Gemma4Config` (config.py 332 to 366) uses FA4 only if `is_fa_version_supported(4)`. That function needs compute capability 9, 10 or 11 (`flash_attn_interface.py` 62 to 88). So the sliding layers (head size 256) also run on Triton, not only the global layers. E-55 used FA4 for all layers.
3. FACT: `TritonAttentionImpl.forward` passes `softmax_scale=self.scale` on every call (`triton_attn.py` 706). Nothing else caches the scale. BF16R reaches the kernel on the A100. The Triton kernel multiplies the scores in float32 (`triton_unified_attention.py` 385 and 549).
4. FACT: The hybrid KV cache manager stays on (`config/vllm.py` 2246 to 2312). The V2 runner scores prompt logprobs in chunks (`prompt_logprob.py` 287 to 323). So the 262,144-token vocabulary does not cause a memory spike. The utilization 0.92 is probably safe (F6 records it).
5. FACT: The scoring alignment is right. `parse_prompt_logprobs` gives `lp[j]` for token `j + 1`. So `lp[answer_start - 1:]` scores the reference tokens and `<turn|>`. The data file gives 37,877 answer tokens (94.7 per item).
6. FACT (E-55 record): the H100 pre-flight gave engine gaps of 0.0015 to 0.0018 and BF16C separations of 0.0026 to 0.0095 (0 in layer 29). V7 therefore decides in 9 layers. The A100 floor should be similar (the 4090 Triton gaps were 0.0016 to 0.0023).
7. FACT (computation): the first model call of each arm holds only the probe. `warm()` runs first and `reset_evidence` follows the pre-flight. Nothing else calls the model before `run_arm` scores B1. B1 has 2,048 tokens, which equals `max_num_batched_tokens`, so it fills one chunk. E-55 passed the same equality (4,096).
8. FACT (simulation, Section "Power"): at no true effect, GO1 and GO2 together happen in 90%, 63%, 39% and 35% of runs for answer-change rates of 2%, 5%, 10% and 15%.
9. HYPOTHESIS: The 31B instruct model is overconfident on text outside its chat style. E-55 R-78 found NLL 20 to 26 at junk positions, and E2B scored 5.5 nats on raw book text. GSM8K human solutions are such text. So the reference NLL of a healthy BF16 31B is likely 1.0 to 2.5, not 0.3 to 0.8.

## Blockers

### B1. The V8 reference-NLL limit of 1.5 is not calibrated and likely to stop a healthy baseline
- Evidence: `analyze_e56.py` line 37 (`REF_NLL_MAX = 1.5`) and `run_e56.py` 333 to 337. A failed gate raises `HealthError` (exit 3) and the decision is INVALID (V8). README Section 2 predicts 0.3 to 0.8. That is a HYPOTHESIS with no source.
- Mechanism: the reference answers are the human GSM8K solutions, not the model's own style. A model with RL overconfidence assigns NLL 6 to 10 to each style-surprising token. If 10% of the 95 answer tokens per item are like this, the mean is 0.4 + 0.1 x 8 = 1.2. At 15% it is 1.6 (INTERPRETATION, same arithmetic as the E-55 numbers).
- The accuracy gates already catch an E-55-like baseline. A model with 37% junk top-1 tokens cannot reach 75% on GSM8K. So the NLL limit adds little protection and adds a real false-stop risk. My estimate of a false stop is 30% to 45% (HYPOTHESIS).
- Cost of the failure: setup, download, engine start and the BF16 arm, about 35 min, $1.1. The run gives no decision.
- Fix: make the reference NLL a record-only value, or raise the limit to 3.0 (the E-55 degraded value was far above this on raw text). Keep the two accuracy gates. Add the top-1 agreement on the reference tokens (`top1` is already stored) to the record, as a check that the alignment holds. Change the prediction text in Section 2. Use the rehearsal to calibrate: it records the E2B reference NLL on the same 400 references. If E2B gives 2.0 or more, the 1.5 limit is wrong for sure.

## Fixes

### F1. DET answer-id identity makes the run INVALID for a reason that does not hurt the inference
- Evidence: `det_check` (`analyze_e56.py` 194 to 213) needs `same == n` over all 432 answers. One flipped token id gives INVALID (V3).
- Mechanism: DET repeats the BF16 batches exactly, because the offline scheduler is deterministic for the same requests. So bit identity is likely (my estimate: 90%). If a kernel is not run-to-run deterministic, one answer changes, and V3 fails. But H1, H2 and H3 stay valid in that case. The paired design and the BF16R arm already include run-to-run noise.
- Fix: keep the scoring part of DET as a gate (mean absolute log-probability difference at most 1e-4, because H3 needs it). Change the answer-id part to a record. Report the DET answer-change rate as a second noise floor for H2. In the pre-flight, keep the single-probe DET gate as it is.
- Note: DET is a valid determinism check for same-composition batches. It does not measure batch-size noise. Once any output changes length, the batches of BF16C and BF16R differ from the BF16 batches. BF16R measures that noise, so H2 is right to use it.

### F2. The pre-flight stops the run for a STOCK8 problem
- Evidence: `run_e56.py` 309 to 331 loops over `ARMS`, which includes STOCK8, and `preflight_problems` (`analyze_e56.py` 157 to 177) adds its V5 and V7 problems to the list. README Section 6 says a STOCK8 problem drops STOCK8, not the run.
- Fix: in `preflight()`, check only the decision arms for the stop. Record STOCK8 problems in `rec["preflight"]["stock8_problems"]` and skip STOCK8 later. The risk is low (about 3%), but the fix is five lines and removes a contradiction.

### F3. The model download has no retry
- Evidence: `session_e56.sh` 87 to 93. `run_to 900` kills `snapshot_download`. `models.txt` then has no path, `SNAP` is empty and `rc=2` ("no snapshot folder"). The run is lost after the Pod is paid.
- Arithmetic: 62.6 GB in 900 s needs 70 MB/s. E-53 read 33 GB at about 120 MB/s. The margin is 1.7 times. A slower host (my estimate: 10%) loses the run.
- Fix: run the download twice in a loop (the second call resumes the partial files), with `DL_TIMEOUT` 1,500 s in total. Time still fits: 3,900 - 120 - 60 - 1,500 - 250 = 1,970 s for the main phase, and the four decision arms need about 1,300 to 1,900 s (F6). Also check that the container disk or volume holds 62.6 GB plus 20 GB. The run script must set it.

### F4. The gate limits and `max_tokens` have no data
- GSM8K accuracy 0.75 can fail a healthy model if the answers are cut at 320 tokens (the fallback takes the last number in a cut text, which is wrong). Gemma 4 style is long. A truncation rate above 15% is possible (my estimate: 10%).
- The same gate can let a half-broken model pass (0.80 on GSM8K for a model that should give 0.93 or more).
- Fix: make the rehearsal VERIFY (2) quantitative. Record the share of `finish == "length"`, the mean answer length and the share with `####`. Amend `GSM_MAX_TOKENS` to 400 before the run if the rehearsal shows more than 5% cut answers. Raise the GSM8K limit to 0.85 if the rehearsal shows the cut share below 2%. Both are cheap and change nothing else.

### F5. One test depends on the machine speed
- Evidence: `test_time_cap_cuts_stock8_and_keeps_the_decision` failed here. With `POD_LIMIT=80` and `AN_RESERVE=10`, STOCK8 is cut only if the stub arms take more than a few seconds. A fast Mac finishes first.
- Fix: use `POD_LIMIT=70` and `AN_RESERVE=10`. Then `fits` needs `now + 1.5 x est <= START`, which is never true, so the cut is certain. Update the README Section 12 count (14 tests; the manifest test now runs).

### F6. Record the 31B memory and speed facts in the first minutes of the run
- The rehearsal (E2B on an RTX 4090) cannot show the 31B memory, the KV pool or the A100 speed. The engine start for the 31B is the first test of 62.6 GB with utilization 0.92 and 64 sequences.
- My estimates (HYPOTHESIS, no 31B BF16 log exists): KV pool about 12 GiB (0.92 x 79.2 - 58.3 - 2); one 10,240-token sequence needs about 2 GiB with the hybrid manager. So the engine should start. The time per arm: scoring 77k tokens about 35 s, GSM8K answers 60 to 220 s, needles about 110 s: 4.5 to 7 min per arm. Four decision arms: 18 to 28 min.
- Fix: save the engine log lines "GPU KV cache size" and "Maximum concurrency" in `session.log`. Wrap the engine start in a try block. On an out-of-memory error, retry once with `gpu_memory_utilization` 0.88 and write the change in `rec["engine_kwargs"]`. No validity check reads that value. A retry costs about 3 min. Without it, an engine error ends the run.

## Options (decisions for the user; none is needed to run)

### O1. Add a long-context NLL at almost no GPU cost
- The 32 needles give almost no information on the cache effect: accuracy is near 1.0, so they add no discordant pairs. The GSM8K items are short (97 prompt tokens, 192 scored), so the global layers see little context. README threat 7 states this.
- Option: request `prompt_logprobs=1` for the needle generation calls. The prefill is the same. The log-softmax is chunked. Then Delta_NLL over the needle prompt tokens beyond position 1,024 is a long-context NLL at 7.6K tokens. It must be a secondary metric and needs a parser check in the rehearsal.

### O2. Larger sample for H1 and H2
- If the answer-change rate is 10% or more, H1 and H2 are often INCONCLUSIVE even for a harmless cache (Section "Power"). Doubling the GSM8K items to 800 cuts the half-width by 30%. It adds about 3 min per arm, so about 12 min and $0.4. The budget has room if STOCK8 is dropped. This needs a new data file and a new hash. I do not recommend it for this run.

### O3. Pre-declare how to read an INCONCLUSIVE result
- Write in the README that H3 is the most decidable hypothesis (below). If H1 and H2 are INCONCLUSIVE and H3 is GO3, the claim is "no change in NLL within 0.002 and no detected accuracy loss; the interval of the accuracy change is X". Also rename an INCONCLUSIVE3 with a narrow interval above 0.002 (for example 0.005 +/- 0.0003): it is a measured small effect, not an unknown.

## Answers to the questions

### 1. Does the frozen rule answer the question? Is it decidable?
FACT (simulation, 4,000 runs per row, `power.py`): 400 GSM8K items (85% with p = 0.985, 15% with p from Beta(2,2)) and 32 needles (p = 0.99). Each arm redraws the answer of an item with a probability that is higher for uncertain items. The BF16C and BF16R arms have the same rate.

| Answer-change rate | GO1 (no effect) | KILL1 (no effect) | GO2 (no effect) | GO1 and GO2 |
|---|---|---|---|---|
| 2% | 0.99 | 0.02 | 0.91 | 0.91 |
| 5% | 0.83 | 0.02 | 0.74 | 0.63 |
| 10% | 0.49 | 0.02 | 0.74 | 0.39 |
| 15% | 0.44 | 0.02 | 0.73 | 0.35 |

- A true 2-point loss at a 5% rate gives KILL1 in 36% of runs. A 4-point loss gives KILL1 in 84%.
- INTERPRETATION: the rule is correct. The false KILL1 rate is 2%. The weak point is power. The discordant pairs set the half-width: 10 pairs give about 1.5 points, 28 pairs give 2.4 points. At the predicted rate (2% to 10%) the claim "GO1 with GO2" has a chance of 39% to 91%. GO2 is weaker than GO1 because two arms add their discordance.
- H3 is the most decidable. FACT (formula): the interval half-width is about 2 x SD(item mean change) / 20. A per-token noise of 0.02 nats gives an item SD of 0.002 (independent tokens) and a half-width of 2e-4. Correlated tokens give up to 1e-3. So GO3 needs a true Delta below about 1.5e-3. The scoring batches are identical in all arms, so no batch noise enters H3.
- BF16C is a larger perturbation than BF16R. FACT (E-55 separations): median 0.006 against 0.0035. So BF16C changes more answers than BF16R even with no real effect. The 3-point margin of H2 absorbs this at low rates only.

### 2. Batched generation and DET
See F1. The design handles the non-invariance: BF16R and BF16C share it, and H2 compares them. DET is valid for determinism of identical batches. The DET rerun uses `detach` and `reattach` correctly: `detach` calls `set_arm("BF16")` first, so `base_scale` is read again at the right value (1.0 times the base).

### 3. The splice with batched calls
- FACT: the V5 and V7 evidence comes from the probe (point 7 above). The pre-hook counts `k.shape[0]`, which has no padding in eager mode. The hooks are position-independent, so mixed prefill and decode calls cause no fault. The "at least" check is safe whether or not the output ids include the stop token.
- FACT: the BF16C, BF16R and STOCK8 transforms are per-token and per-head. Batch composition does not matter to them.
- INTERPRETATION: the stub drives the real `Splice` in the real call order, but it puts all prefill first, then decode. It does not model a mixed prefill and decode step, preemption or the hybrid cache. None of these changes what the hooks see (shape [T, H, D]). The real risks (engine start, `parse_prompt_logprobs` on batched outputs, the V2 runner) are not testable on the Mac. The rehearsal covers the parser and the batched counts (VERIFY 3 and 4).

### 4. Alignment, template, stop token, extraction
- FACT: alignment (key finding 5). The test checks `answer_start == len(ids)` but cannot check the real `lp` indexing, because the stub builds `lp` by the same rule. Add the reference top-1 agreement to the rehearsal VERIFY: an off-by-one error drops it to chance (below 10%) and a right alignment gives 40% or more for an instruct model.
- FACT: the needle prompts end with `<channel|>` (the test checks it). The scoring sequences join the template and the reference with a special token, so no tokenization merge can occur. `eot_id` is 106 and is in the stop list.
- INTERPRETATION: GSM8K extraction takes the first number after the last `####`. A Markdown heading such as "#### Step 3: total 18" would give 3. The risk is small and equal in all arms. The `differ` flag compares answers as strings: "18" and "18.00" count as different. Compare the numeric values instead (F7, below).

### 5. The V8 limits
See B1 and F4. Needle accuracy 0.75 is safe: a healthy 31B reads 7.7K tokens without trouble. A degraded model at 0.80 on GSM8K would pass.

### 6. The A100 path
Key findings 2 to 4 and F6. Memory and time estimates in README Section 9 are reasonable (my estimate: Pod 50 to 60 min, $1.5 to $1.8). The worst-case cap arithmetic ($2.48) is right. V1 needs both "A100" or "H100" and "80GB" in the device name. This holds for the A100 SXM and PCIe 80GB and the H100 SXM. An H100 PCIe ("NVIDIA H100 PCIe") would fail V1, but README Section 4.1 allows only the H100 80GB SXM.

### 7. Stub fidelity and test coverage
The tests catch logic faults in the rule, the statistics, the gates, the manifest, the watchdog and the stop paths. They would not catch B1, F3, F6 or a bad alignment. The stub gives its synthetic NLL (0.4 per token), so V8 never meets a real value. This is why B1 passed S3.

### 8. Cost, watchdogs and rehearsal
Caps and watchdogs are consistent (`test_session_limits_fit_the_cost_cap` passes). The balance is $7.8; the two caps total $2.95, so about $4.85 remains if both run to the cap. The protocol floor of cap plus $35 is not met, so the user decides (as D-95). The rehearsal does not cover the 31B memory, the A100, the download speed or the V8 values. Add the F4 and B1 records to it, and the engine-log lines of F6 to the real run.

### 9. Other points
- Section 1 and Section 2 say the rebuilt K is bf16(w x V). The code uses K_unrot = V and q' = bf16(q w) (the "two views", `splice_e55.py` 90 to 94). The rounding is on the query side, not on the stored K. The size is the same, and E-53 and E-54 used this form. Write it in Section 4.4.
- E-55 R-78 cause D (a vLLM Gemma 4 K=V defect) and cause C (backend) are still open. E-56 changes the checkpoint and the backend together, so a healthy baseline will not say which one was the cause. This does not matter for E-56. Record it in the journal.
- F7 (small): normalize numeric answers before the `differ` comparison (`float` equality with 1e-6), and name an INCONCLUSIVE3 with a narrow interval above 0.002 as SMALL_EFFECT3 in the report line.

## Chance of a decidable result
HYPOTHESIS (product of my estimates): the Pod, download and engine start 0.88; the pre-flight 0.93; V8 0.54 as frozen (0.77 with B1 amended); DET and the other checks 0.90 (0.95 with F1); H1 decidable given a valid run 0.72 (mix of the rates in the table).
- As frozen: about 0.29.
- With B1, F1, F2 and F3 amended: about 0.45.
- H3 alone is decidable in about 0.85 of the valid runs.

## Summary (10 lines)
1. Verdict: AMEND_THEN_RUN.
2. Blocker B1: the V8 reference-NLL limit (1.5) is uncalibrated and probably too low for human GSM8K solutions scored by an instruct model. Make it record-only or 3.0.
3. F1: make the DET answer-id identity a record. Keep the DET log-probability gate.
4. F2: do not stop the run for a STOCK8 pre-flight problem.
5. F3: add a download retry with resume, 1,500 s in total, and check the disk size.
6. F4: calibrate `max_tokens` (320) and the GSM8K limit (0.75) with the rehearsal. Add the reference top-1 agreement as an alignment check.
7. F5, F6: fix the speed-dependent test. Save the KV pool lines and add an engine-start retry at utilization 0.88.
8. The A100 runs Triton for all 60 layers (FA4 needs SM90 or higher). `impl.scale` reaches the kernel. Alignment, counts, probe isolation and the stub flow are correct.
9. Power: at no true effect, GO1 and GO2 together happen in 91%, 63%, 39% and 35% of runs at answer-change rates of 2%, 5%, 10% and 15%. H3 is the most decidable.
10. Chance of a decidable result: about 0.29 as frozen, about 0.45 after the amendments.
