# E-46 reproducer (written by session_e46.sh)
Image: vllm/vllm-openai@sha256:2e735ffeef26457f8a3ff44a62ae182870b10af37adc06ecbb08baca152365c6
vllm 0.33.0rc1.dev157+gedad8b03c
torch 2.14.0+cu130
VLLM_BUILD_COMMIT=unset
Name: vllm
Version: 0.33.0rc1.dev157+gedad8b03c
Summary: A high-throughput and memory-efficient inference and serving engine for LLMs
GPU: NVIDIA H100 80GB HBM3, 595.91.07

Fusions on (default pass config):
  vllm bench latency --model Qwen/Qwen3-8B-FP8 --linear-backend deep_gemm --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10
Fusions off:
  vllm bench latency --model Qwen/Qwen3-8B-FP8 --linear-backend deep_gemm -cc '{"pass_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}' --batch-size 64 --input-len 128 --output-len 128 --num-iters-warmup 3 --num-iters 10
