# U-01 draft: vLLM bug report (NOT FILED)

Status: DRAFT 1, 2026-10-29. Waits for the E-46 post-run review and for the user's approval before posting to
github.com/vllm-project/vllm/issues. Evidence: E-44 (`experiments/fp8_quant_fusion/`), E-46
(`experiments/fp8_fusion_validate/`), R-57, R-58. Register row: `RESEARCH_OUTPUTS.md` U-01, O-03.

---

**Title:** [Performance]: Block-FP8 RMSNorm+quant and SiLU+quant fusions make decode 2-6% slower on H100 when they fire

### Your current environment

- vLLM nightly `0.33.0rc1.dev157+gedad8b03c` (docker `vllm/vllm-openai@sha256:2e735ffeef26457f8a3ff44a62ae182870b10af37adc06ecbb08baca152365c6`), also seen on v0.32.0
- torch 2.14.0+cu130, 1 x H100 80GB HBM3 (SXM), driver 595.91.07
- Full `collect_env` output: (attach `collect_env.txt` from E-46)

### Describe the performance issue

For block-FP8 models (1x128 activation groups), when the `norm_quant` and `act_quant` fusion passes fire, decode on
H100 is slower than with the two passes turned off. The fused CUDA kernels (`rms_norm_per_block_quant`,
`silu_and_mul_per_block_quant`) cost more per call than the Inductor-generated Triton kernels plus
`per_token_group_quant` that they replace.

The fusions fire with `--linear-backend deep_gemm` and `--linear-backend cutlass` (and `triton`). With the default
backend on H100 (`FlashInferFp8DeepGEMMDynamicBlockScaledKernel`), the log says
`Enabled custom fusions: norm_quant, act_quant`, but no fused kernel runs, so the default path is not affected.

**Reproducer** (vLLM's own benchmark, default multiprocess engine):

```bash
# fusions on (default pass config)
vllm bench latency --model Qwen/Qwen3-8B-FP8 --linear-backend deep_gemm --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10
# fusions off
vllm bench latency --model Qwen/Qwen3-8B-FP8 --linear-backend deep_gemm -cc '{"pass_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}' --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10
```

Result (3 alternating rounds each): average latency on 0.908 / 0.938 / 0.942 s, off 0.895 / 0.897 / 0.901 s.
The median with the fusions on is 4.6% higher.

**Paired end-to-end decode throughput** (offline `LLM.generate`, 128 new tokens, 5 alternating rounds per pair,
warm compile cache; ratio = tokens/s with fusions off ÷ on; brackets are one-sided 98.3% bounds):

| Model, backend | batch 1 | batch 8 | batch 64 | batch 256 |
|---|---|---|---|---|
| Qwen3-8B-FP8, deep_gemm (nightly) | 1.026 | 1.022 [1.019, 1.025] | 1.043 [1.034, 1.052] | 1.061 [1.051, 1.071] |
| Qwen3-14B-FP8, deep_gemm (nightly) | 1.039 | 1.027 [1.024, 1.031] | 1.047 [1.027, 1.068] | 1.054 [1.048, 1.059] |
| Qwen3-8B-FP8, cutlass (nightly) | 0.998 | 1.002 | 1.021 [1.018, 1.024] | 1.035 [1.031, 1.040] |
| Qwen3-8B-FP8, deep_gemm (v0.32.0, separate run) | 1.041 | 1.034 | 1.047 | 1.054 |

GEMM time is the same in both configurations; only the norm, SiLU and quant work differs.

**Kernel times inside the model** (torch.profiler, Qwen3-8B-FP8, deep_gemm, v0.32.0, average per call over 32
generation steps; the averages include one prefill step):

| Work | Fusions on (vLLM CUDA kernel) | Fusions off (Inductor Triton kernel) |
|---|---|---|
| RMSNorm + quant, batch 8 | 6.3 us (`rms_norm_per_block_quant`) | 2.3 us (`triton_red_fused_fused_add_rms_norm_per_token_group_fp8_quant`) |
| RMSNorm + quant, batch 64 | 9.2 us | 4.1 us |
| SiLU + quant, batch 8 | 3.8 us (`silu_and_mul_per_block_quant`) | 2.3 us (`triton_poi_fused_mul_per_token_group_fp8_quant_silu_slice`) |
| SiLU + quant, batch 64 | 19.7 us | 6.3 us |

In an isolated CUDA-graph microbenchmark at Qwen3-8B shapes, the fused SiLU kernel was 1.49x slower than the fastest
unfused path at 256 and 512 tokens (and faster at 1 to 8 tokens), and the fused norm kernel was 1.12x to 1.24x slower
at 1 to 512 tokens.

**Accuracy:** GSM8K (1,319 problems, greedy, Qwen3-8B-FP8, deep_gemm): fusions off minus on = +0.38 pp
(95% bootstrap interval -0.61 to +1.36). No accuracy benefit from the fusions.

**Possible cause (not verified):** `silu_and_mul_per_block_quant_kernel` launches one 128-thread block for each
(token, group) pair, with one element per thread and a shared-memory tree reduction; `rms_norm_per_block_quant_kernel`
launches one block per token. The Inductor kernels are vectorized. I did not run ncu.

**Possible fixes:** do not enable `fuse_norm_quant` / `fuse_act_quant` by default on sm90 for the block-FP8 backends
where the fused kernels are slower, or make the fused kernels faster at these shapes.

**Scope and limits:** 1 x H100 only (B200 not tested); dense Qwen3 8B and 14B only (MoE not tested); TP 1;
end-to-end numbers include prefill; batch 256 runs without CUDA graphs (above the largest capture size).

**Related:** #40552 (B200 UE8M0 norm+quant fusion skipped), #62386 (TMA-aligned scales in
`silu_and_mul_per_block_quant`), #45364 / #43224 (manual fusion port; the 1x128 block path is deferred),
#25094 (native group quant against the CUDA kernel). In addition (separate topic), on the default engine the activation
quant for M >= 32 runs inside the opaque `dynamic_flashinfer_deepgemm_blockscale_gemm` op, about 8% of device time at
batch 64 for Qwen3-8B-FP8, so no fusion can reach it.

### Before submitting a new issue...

- [x] I searched for relevant issues (see Related).
