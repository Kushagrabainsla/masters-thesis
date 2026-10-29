# R-56: Sonnet pre-run review of E-44 (2026-10-28)

Verdict: AMEND_THEN_RUN. All fixes applied (README Amendment A1).
1. FACT: the batch-8 motivation was wrong. The FlashInfer path quantizes inside a TRT-LLM scale_1x128 kernel (7.7 ms),
   so unfused work is 15.9 ms at batch 8 (not 8.3 ms) against 21.0 ms fused; the expected end-to-end gap is about
   1.3%. Batch 64 holds: 33.6 ms against 44.8 ms (about 4.6% of device time). The 0.5% profile filter dropped nothing.
2. FACT: the dict compilation_config turns only the two fusions off; other defaults stay.
3. FACT: deep_gemm selects DeepGemmFp8BlockScaledMMKernel. The "fusion pass matches" line may not print on a cache
   hit. Fixed: VLLM_DISABLE_COMPILE_CACHE=1 for the D0 and D1 profile runs.
4. Kernel benchmark: signatures match; single-call event timing measured CPU launch cost; UE8M0 scales differ from
   the fused format; no Inductor norm baseline; the roofline is meaningless at small sizes. Fixed: CUDA-graph timing
   over rotating buffers, model-path scale format, an Inductor norm variant, comparison with the fastest unfused path,
   a launch floor.
5. FACT: t values correct. AUTO_STOP must be 1; time about 55 min; boot CAP 4800 s; batch 256 runs without graphs.
6. Watchdog: env variables exist in the Pod; JSON quoting in a create-pod cmd is fragile. Fixed: base64 script from
   `experiments/runpod/pod_cmd.py` (2 tests), rehearsed on an L4 with the same image and command.
7. INTERPRETATION: a GO alone is not an upstream PR. A maintainer will ask for the cause (likely one block per token),
   an ncu comparison, more models and an accuracy check. Recorded as follow-ups.
