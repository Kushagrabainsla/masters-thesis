# E-45: Does a fusable native FP8 group quant beat the CUDA quant kernel end to end on H100? (preregistration)

Status: PREDECLARED 2026-10-29. CANCELLED BEFORE RUN (R-58: A1 is a no-op on the default engine). Idea: I-172. Follows E-44 and R-57. User request 2026-10-29: "Find the cause
first". Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
On one H100 SXM with vLLM v0.32.0:
1. (Deciding, default users) For the default engine, is decode faster with the native group quant
   (`compilation_config={"custom_ops": ["-quant_fp8"]}`, so Inductor can fuse the quant into its norm and SiLU
   kernels) than with the default CUDA quant kernel (`+quant_fp8`, forced on for block-FP8 models)? Qwen3-8B-FP8 decides;
   Qwen3-14B-FP8 checks that the sign holds.
2. (Deciding, safety) Does the native quant keep GSM8K accuracy (A1, 1,319 problems) on Qwen3-8B-FP8?
3. (Reported) Does the E-44 fusion slowdown on deep_gemm also hold on Qwen3-14B-FP8?

## 2. Why (rubber duck)
- FACT (E-44 profiles, Qwen3-8B-FP8, 32 decode steps, averages include one prefill step): on the default engine, at
  batch 64, Inductor runs norm and SiLU, then 4,608 separate `per_token_group_quant_8bit` CUDA calls (4 for each layer
  step) take 19.6 ms, 8.1% of device time. At batch 8 the FlashInfer path quantizes inside its own GEMM wrapper.
- FACT (E-44): vLLM's fused CUDA norm+quant and SiLU+quant kernels cost 2x to 3x more per call than the Inductor
  kernels they replace (norm 6.3 against 2.3 us at batch 8; SiLU 19.7 against 6.3 us at batch 64). The fused SiLU
  kernel launches one 128-thread block for each (token, group) pair and reduces in shared memory
  (`csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`).
- FACT (vLLM v0.32.0 `vllm/config/vllm.py`): `+quant_fp8` is forced on for blocked weights, with the comment "TODO
  disable in follow up. On H100 the CUDA kernel is faster than native implementation" and a link to issue #25094.
  #25094 is open; its data are standalone kernel timings (CUDA against compiled torch, mixed results), and an edit asks
  for new numbers. I did not find an end-to-end, in-model measurement.
- HYPOTHESIS: in the model, a native quant lets Inductor fuse it into the norm and SiLU kernels, which removes most
  of the 4 separate quant launches per layer step. Prior: about even odds that this is at least 2% faster at batch 64.
- Risk: the native path makes column-major scales without TMA alignment; DeepGEMM may add a layout copy, or numerics
  may change. V0 and the accuracy check cover this.
- If GO: a vLLM PR or an issue update on #25094 with end-to-end data. If KILL: the comment in vLLM is right in the
  model too, and the separate quant stays.

## 3. Setup (frozen)
1 x H100 80GB HBM3, Secure, on-demand, `vllm/vllm-openai:v0.32.0`, network volume. Environment as in E-44. Models:
`Qwen/Qwen3-8B-FP8` (M8), `Qwen/Qwen3-14B-FP8` (M14). Configs:
- A0: default engine. A1: default engine + `{"custom_ops": ["-quant_fp8"]}`.
- B1: `--linear-backend deep_gemm`. B2: B1 + `{"pass_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}`.
Speed: `run_cell.py --mode throughput`, 128 new tokens, batch 1, 8, 64, 256, 5 repeats, `max_num_seqs` 256, GSM8K
prompts. Profiles: one process for A0 and A1 on M8 (`--profile-kernels --profile-batches 8 64 256`). Accuracy:
`run_a2.py --mode full --no-score` for A0 and A1 on M8 (greedy, 768 tokens).

## 4. Order
Setup; accuracy A0, A1 on M8 (each after a warm-up); profiles A0, A1 on M8; pairs: P1 (M8, A0 against A1), P2 (M14,
A0 against A1), P3 (M14, B1 against B2), each with one warm-up per config and 5 rounds (first config first in odd
rounds, second first in even rounds).

## 5. Rule (frozen)
Per round: log(median second / median first). Deciding sizes 8, 64, 256. One-sided t bounds at alpha 0.05 / 3 (df 4,
t 3.186; df 3, t 3.740 if one round is dropped).
- GO: P1 has a lower bound of at least 1.02 at a deciding size, AND at the same size P2 has a ratio of at least 1.00,
  AND A1 passes the accuracy gate (paired bootstrap lower bound of the GSM8K difference above -2.0 pp), AND V0 holds.
- KILL: every P1 deciding upper bound is below 1.01, or A1 fails the accuracy gate.
- Otherwise INCONCLUSIVE, with the reason.
- P3 is reported with its own label: "fusion slowdown holds on M14" if a deciding lower bound of B2/B1 is at least
  1.02.
- Validity: V0: the A1 profile has at most 25% of the A0 count of `per_token_group_quant` CUDA calls at batch 64, and
  the A1 log shows the `-quant_fp8` setting (not overridden). V2: at least 4 good rounds per pair. V3: telemetry as in
  E-44.

## 6. Budget and safety
H100 SXM Secure at the live price (about $3.99/h). Setup with the 14B download about 10 min; accuracy about 8 min;
profiles about 4 min; P1 about 14 min; P2 and P3 about 20 min each. About 80 min, about $5.30. Session caps: no new
work after 95 min, session watchdog 105 min. Boot watchdog from `experiments/runpod/pod_cmd.py`, CAP 6900 s
(115 min, about $7.65 with a 5 min image pull). Cost cap $8. Launch with `AUTO_STOP=1`. Create-to-start in one call.

## 7. Threats
Two dense Qwen3 models, one GPU, one engine version; end-to-end tokens/s includes prefill; batch 256 runs without CUDA
graphs. A GO does not prove the cause of the E-44 fused-kernel cost; it tests the most useful fix for default users.

## 8. Files
`session_e45.sh`, `analyze_e45.py`, `test_e45.py`. Reused: `../safedispatch_testA/run_cell.py` and
`../safedispatch_testA2/run_a2.py` (both with `--compilation-config`), `../safedispatch_testA2/gsm8k.py` and
`analyze_a2.py`, `../nvfp4_dispatch/analyze_e40.py`, `../runpod/pod_cmd.py`.

## Cancellation (2026-10-29, R-58, before any run)
R-58 (Sonnet pre-run review): DO_NOT_RUN as written. FACT (vLLM v0.32.0
`vllm/model_executor/kernels/linear/scaled_mm/flashinfer.py`, confirmed): the default H100 block-FP8 kernel
`FlashInferFp8DeepGEMMDynamicBlockScaledKernel` sets `apply_input_quant = False` and sends BF16 input into one opaque
custom op, `dynamic_flashinfer_deepgemm_blockscale_gemm`. Inside it, `torch.cond` picks the FlashInfer TRT-LLM path for
M < 32 (its own `scale_1x128` quant) and, for M >= 32, calls `per_token_group_quant_fp8` directly and then DeepGEMM. So
`custom_ops: ["-quant_fp8"]` changes neither branch, and Inductor cannot fuse into the op. A1 would equal A0 (V0 would
fail). Other R-58 facts: the config handling of `-quant_fp8` is correct; the native group quant ignores TMA alignment and
the packed UE8M0 layout. Upside for default users if the M >= 32 quant were fused into the norm and SiLU kernels (from
the E-44 D0 table, 3 of 4 linears per layer): at most about 6% of device time at batch 64 and 256, 0% at batch 8;
realistic 2% to 3%. Reaching it needs a vLLM code change (quantize outside the opaque op for M >= 32), not a flag.
