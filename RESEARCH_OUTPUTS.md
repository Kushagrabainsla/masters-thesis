# Research outputs register

What came out of this thesis research: findings, upstream reports and contributions, reusable methods and tools,
and lessons. One row for each output, with its status and the evidence behind it. The research journal
(`RESEARCH_JOURNAL.md`) stays the source of truth; this file is the index of results that other people can use.

Started 2026-10-29. Update it when an output changes status (for example, a bug report is filed or a fix is merged).

Status words: VALIDATED (passed its preregistered rule and an independent review), CONFIRMED WITH LIMITS (a review
confirmed it with stated scope limits), PENDING VALIDATION, DRAFT, FILED (posted upstream), NEGATIVE (a clear "no").

## 1. Findings

| ID | Finding (simple words) | Status | Scope | Evidence |
|---|---|---|---|---|
| O-01 | vLLM's default kernel backend is already the right choice for block-FP8 on H100 and NVFP4 on B200. Other backends are slower, or at most a few percent faster on a narrow batch band. A correctness-aware dispatcher has little to win here. | NEGATIVE, CONFIRMED WITH LIMITS | Qwen3-8B and Qwen3-30B-A3B; 1 x H100 and 1 x B200; vLLM v0.32.0; batch up to 256 (H100) and 512 (B200) | E-36, E-38, E-39, E-40, E-41; reviews R-42, R-44, R-46, R-48, R-50; decisions D-63, D-64; chapter `thesis/chapters/i23-safedispatch-negative-result.md` |
| O-02 | On B200, five NVFP4 FP4 x FP4 linear kernels (CuTe-DSL, cuDNN, FlashInfer CUTLASS, vLLM CUTLASS, TRT-LLM) give bit-identical logprobs. Batch composition and compile mode change the numbers far more than the kernel choice. | CONFIRMED WITH LIMITS | Qwen3-8B-NVFP4; 256 prompts; eager mode (graph mode for 2 kernels); vLLM v0.32.0 | E-42 (`experiments/nvfp4_bitequal/`); R-51, R-52 |
| O-03 | On H100, vLLM's block-FP8 norm+quant and SiLU+quant fusions make decode 2% to 6% slower when they fire (deep_gemm: every batch size, two models; cutlass: batch 64 and 256), with no accuracy benefit; `vllm bench latency` agrees (4.6%). The fused CUDA kernels cost 2x to 3x more per call than the Inductor kernels they replace. The default engine is not affected (the fusion does not match there). | VALIDATED (E-46) | Qwen3-8B-FP8 and Qwen3-14B-FP8; deep_gemm and cutlass; 1 x H100; vLLM v0.32.0 and nightly 0.33.0rc1.dev157+gedad8b03c; not B200, not MoE, TP 1 | E-44 (`experiments/fp8_quant_fusion/`), E-46 (`experiments/fp8_fusion_validate/`); R-56, R-57, R-59, R-60 |
| O-04 | On the default H100 block-FP8 engine, the activation quant for 32 or more tokens runs inside one opaque custom op (`dynamic_flashinfer_deepgemm_blockscale_gemm`), so the compiler cannot fuse it. It takes 8.1% of device time at batch 64. A code change could recover about 2% to 6%; a flag cannot. | CONFIRMED (source and profile) | Qwen3-8B-FP8; 1 x H100; vLLM v0.32.0 | E-44 D0 profile; `experiments/fp8_quant_cause/README.md` (Cancellation); R-58 |
| O-05 | The cuDNN NVFP4 linear backend fails at start in 7 of 10 runs under CUDA graphs (`CudnnPlanBuildInCaptureError`), and runs in eager mode. | CONFIRMED (observed, cause not proven) | Qwen3-8B-NVFP4; 1 x B200; vLLM v0.32.0 | E-41 (`experiments/nvfp4_large_batch/`), E-42; R-50 |
| O-06 | Three RL frameworks (slime fully async mode, OpenRLHF partial rollout, PipelineRL) keep requests or KV across a weight update with no per-token policy-version record and no check. vLLM and SGLang already emit per-token version spans. | CONFIRMED WITH LIMITS (source audit) | Framework source at pinned commits, 2026-10-28 | E-43 (`experiments/rollout_version_audit/audit.md`); R-55; D-67 |
| O-07 | Exact model symmetries (permuting hidden units, heads or the residual basis, sign flips, power-of-two scales) cannot make the operands a tensor-core lane sees much more alike: 0.5% fewer bit transitions (median), 2.5% at best, against 72% to 82% for a non-exact per-row sort. So "sort the weights to save GPU power" (future work in 2409.18324) does not work with exact transforms. | NEGATIVE, CONFIRMED WITH LIMITS (proxy screen, one small model; reproduced by an independent model in review) | Qwen3-0.6B, BF16 and block-FP8 codes; no GPU power measured | E-47 (`experiments/power_shaping_screen/`); R-62 |
| O-09 | The BF16 LM head is 9% to 10% of decode device time for Qwen3-8B-FP8 at batch 1 to 8 on H100 and already runs near peak bandwidth. An FP8 screen plus a rigorous per-row bound keeps decoding exact with only 2 to 14 BF16 rows re-read (Qwen3-0.6B screen), but this design is already an open vLLM PR (#60566) and gains only about 1% to 3% end to end on H100. | NEGATIVE (prior work), CONFIRMED WITH LIMITS | Qwen3-8B-FP8 profiles (H100); CPU screen on Qwen3-0.6B | E-36 and E-46 profiles; E-48 (`experiments/lmhead_screen/`); R-63 |
| O-08 | In FP8 vLLM serving on 1 x H100 (batch 1 to 256), the GPU ran at its 700 W cap in 43% to 46% of busy seconds (median clock 1875 to 1950 MHz against 1980 MHz). Removing the cap completely would add at most 0.65% to 2.4% average throughput, so power-cap ideas have little room in this regime. | CONFIRMED (our telemetry) | Qwen3-8B-FP8; 1 x H100 SXM; NVML 1-second samples | E-44, E-46 telemetry; R-62 |

## 2. Upstream reports and contributions

| ID | What | Target | Status | Link |
|---|---|---|---|---|
| U-01 | Bug report: block-FP8 norm/SiLU quant fusions are slower than the unfused path on H100 (O-03, with O-04 as context) | vllm-project/vllm | KEPT IN REPO, NOT POSTED (user decision D-68, 2026-10-29); final text in `docs/upstream/U-01_vllm_fp8_fusion_h100.md` | (not filed) |
| U-02 | Possible bug report: cuDNN NVFP4 plan build fails under CUDA graph capture (O-05) | vllm-project/vllm or flashinfer-ai/flashinfer | Not started | (not filed) |
| U-03 | Optional fixes: per-token policy-version records in slime, OpenRLHF, PipelineRL (O-06) | those repositories | Optional career work (D-67) | (not started) |

## 3. Reusable methods and tools

| ID | What | Where |
|---|---|---|
| T-01 | Paired-round speed comparison (alternating order, one-sided t bounds with Bonferroni) that removed false 5% to 7% "leads" from one-process screens | `experiments/nvfp4_dispatch/analyze_e40.py`, `experiments/fp8_quant_cause/analyze_e45.py`; chapter Section 3.3 |
| T-02 | Kernel alias detection from vLLM kernel-selection log lines | `experiments/nvfp4_large_batch/analyze_e41.py`, `experiments/nvfp4_bitequal/analyze_e42.py` |
| T-03 | Experiment protocol with stage gates, preregistration, different-model reviews, run ledger and noise-aware comparison | `docs/experiments/`, `experiments/ledger/` |
| T-04 | Boot watchdog for cloud GPU Pods (the Pod stops itself at a cost cap, counted from boot), with tests and an L4 rehearsal | `experiments/runpod/pod_cmd.py`, `test_pod_cmd.py`; `docs/experiments/RUNPOD.md` Section 3 |
| T-05 | CUDA-graph kernel microbenchmark for small fused kernels (removes launch cost from the timing) | `experiments/fp8_quant_fusion/kbench_e44.py` |
| T-06 | Top-20 logprob bit-equality check with determinism, large-difference and small-change controls | `experiments/nvfp4_bitequal/analyze_e42.py` |
| T-07 | Operand bit-transition screen for weight transforms (no GPU, reads safetensors without torch, exactness check included) | `experiments/power_shaping_screen/toggle_screen.py` |

## 4. Lessons (operations and method)

| ID | Lesson | Evidence |
|---|---|---|
| L-01 | Validate every log parser on a real log of the exact engine version; a new format silently hid an alias | E-40 (cell D V0) |
| L-02 | Cold compile caches change greedy output; warm every configuration before measuring | E-36 rehearsal (amendment A3) |
| L-03 | Time every per-item loop at the real data size before booking a GPU | E-38 first run ($14 lost) |
| L-04 | An idle GPU before the session starts is the largest cost risk; use a boot watchdog and do create-to-start in one step | E-42 incident (about $16 idle) |
| L-05 | A standalone kernel benchmark can miss the in-model cost; profile in the model | E-44, R-57 |
| L-06 | Check where a setting acts before measuring it; a config flag cannot reach inside an opaque op | E-45 cancelled by R-58 |
| L-07 | Before a novelty search on a mechanism, check that its host code path has a future (open RFCs, deprecation banners); read whole design docs, not only the lines a search matches | I-173 / R-61 (vLLM IR deprecated; compiler fusions moving to model code) |
| L-08 | Before a symmetry idea, compare the freedom it has (one choice per dimension) with the freedom the effect needs (one choice per element); a free CPU screen settled I-174 before any GPU spend | I-174 / E-47 / R-62 |
| L-09 | When a profile shows a clear optimization, search the target engine's open PRs by the mechanism's plain name in pass 1; real waste attracts other engineers fast (vLLM PR #60566 had the same idea three weeks earlier) | I-181 / E-48 / R-63 |
| L-10 | Before a profiling test on a new model type, count the engine's open PRs per code path (vLLM: 96 for GDN, 39 for Gemma) and read the model's own design report next to the engine's model code. A report can state a saving that no engine implements (Gemma 4: K=V gives 37.5% less global KV; vLLM, SGLang and TensorRT-LLM store full K and V) | D-73 pass 1 / I-182 / E-49 |
