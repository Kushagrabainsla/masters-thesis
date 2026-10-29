# E-44: Are vLLM's fused block-FP8 norm and SiLU quant kernels slower than the unfused path on H100? (preregistration)

Status: PREDECLARED 2026-10-28. NOT RUN. Idea: I-172 (optimization lead). User request 2026-10-28: "Prepare the probe".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
On one H100 SXM with vLLM v0.32.0 and `Qwen/Qwen3-8B-FP8` (block FP8, 1x128 activation groups):
1. (Deciding) With the same GEMM backend (`--linear-backend deep_gemm`, where vLLM's norm+quant and SiLU+quant
   fusions fire), is decode faster with the two fusions turned off than with them on?
2. (Deciding, diagnostic) At Qwen3-8B shapes, are the fused kernels `rms_norm_per_block_quant` and
   `silu_and_mul_per_block_quant` slower than the two-kernel unfused path, and how far is each from the H100
   memory-bandwidth roofline?
3. (Reported) On the default backend (FlashInfer DeepGEMM), how many fusion patterns match?

## 2. Why (rubber duck)
- FACT (E-36 kernel profiles, 32 decode tokens, kernels above 0.5% share): the default backend
  `FlashInferFp8DeepGEMMDynamicBlockScaledKernel` logs "Enabled custom fusions: norm_quant, act_quant", but its profile
  shows no fused kernel; norm + SiLU + quant take 8.3 ms at batch 8 and 33.8 ms at batch 64. With
  `DeepGemmFp8BlockScaledMMKernel` (fusion fires), the same work takes 21.0 ms and 44.8 ms; GEMM time is equal
  (183.2 ms against 182.1 ms at batch 64). cutlass and triton (fusion fires) show 20 to 22 ms and 41 to 42 ms.
- INTERPRETATION: the fused kernels may be about 2x slower than the unfused path at batch 8. So "make the fusion
  fire on the default path" would make it slower, and the real contribution would be a faster fused kernel or a
  rule that turns the fusion off where it is slower.
- Threat to this reading: profile shares under 0.5% were dropped, the profile ran once per config, and kernel time
  is not end-to-end time. This probe measures end-to-end speed with paired rounds and times the kernels directly.
- If GO: two possible upstream contributions in vLLM: (a) a default that turns the block-quant fusions off on sm90
  where they are slower; (b) an optimized fused kernel that beats the unfused path. If KILL: the profile gap was an
  artifact and I-172 is closed.
- Prior (HYPOTHESIS): about two in three that the fused path is slower end-to-end at batch 8 or more.

## 3. Setup (frozen)
- 1 x H100 80GB HBM3 (sm_90), Secure, on-demand, `vllm/vllm-openai:v0.32.0`, network volume if the data center has
  one (else the Pod volume disk). Environment as in E-36 to E-41 (in-process engine, prefix caching off, CUDA bin on
  PATH, caches under the vLLM cache root, one warm-up per config).
- Configs (all `Qwen/Qwen3-8B-FP8`):
  - D0: default (no backend flag).
  - D1: `--linear-backend deep_gemm` (fusions on by default).
  - D2: `--linear-backend deep_gemm --compilation-config '{"pass_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}'`.
- Speed: E-36 `run_cell.py --mode throughput`, end-to-end tokens/s for 128 new tokens, batch 1, 8, 64, 256, 5 repeats,
  `max_num_seqs` 256, GSM8K prompts.
- Profiles: one process per config with `--profile-kernels --profile-batches 8 64 256` and `VLLM_LOGGING_LEVEL=DEBUG`
  (the debug line "fusion pass matches" gives the match table).
- Microbenchmark `kbench_e44.py` (no model): hidden 4096, intermediate 12288, tokens 1, 8, 64, 256, 512; fused
  against unfused (vLLM CUDA ops, and a torch.compile native path), CUDA events, 200 iterations, median; output
  equality check; achieved bandwidth against 3.35 TB/s.

## 4. Order
1. Setup, microbenchmark.
2. Warm-up D0, D1, D2; profile run D0, D1, D2.
3. 6 paired rounds of D1 and D2 (D1, D2 in odd rounds; D2, D1 in even rounds).

## 5. Rule (frozen)
Per round: log(median D2 / median D1). Deciding batch sizes 8, 64, 256. One-sided t bounds at alpha 0.05 / 3 (df 5,
t 2.912; df 4, t 3.186 if one round is dropped).
- GO: a lower bound of at least 1.02 at a deciding size, AND the microbenchmark shows a fused kernel at least 1.3x
  slower than its unfused path at 3 or more of the 5 token counts.
- KILL: every deciding upper bound is below 1.01 (fusion off is not faster end-to-end).
- Otherwise INCONCLUSIVE, with the reason.
- Validity: V0: the D1 log lists "Enabled custom fusions" with norm_quant and act_quant, and the D1 profile has at
  least one fused kernel (`rms_norm_per_block_quant` or `silu_and_mul_per_block_quant`); the D2 profile has none;
  D1 and D2 logs select the same linear kernel. V2: at least 5 good rounds. V3: at least 100 telemetry rows and at
  most 2% busy samples with a hard slowdown. A failed V gives INVALID.
- The heuristic thresholds 1.02, 1.01 and 1.3x are project choices, not standards.

## 6. Budget and safety
H100 SXM Secure at the live price (about $3.99/h). Setup about 10 min; microbenchmark about 3 min; 6 profile and
warm-up processes and 12 paired processes at about 1.5 to 2 min. About 45 min. Session caps: no new work after
60 min, session watchdog 70 min. Boot watchdog in the Pod start command (RUNPOD.md Section 3): stop at 90 min from
boot. Cost cap $6. The boot watchdog is rehearsed first on an L4 (Section 9).

## 7. Threats
One dense model, one GPU, one engine version; end-to-end tokens/s includes prefill. The MoE cell is not tested. A GO
says the fused kernels are slower in this setup; it does not say why.

## 8. Files
`session_e44.sh`, `kbench_e44.py`, `analyze_e44.py`, `test_e44.py`. Reused: `../safedispatch_testA/run_cell.py`
(new `--compilation-config` flag, default off), `../safedispatch_testA2/gsm8k.py`, `../nvfp4_dispatch/analyze_e40.py`.

## 9. Boot-watchdog rehearsal (stage S5, before E-44)
1 x L4 (cheapest Secure stock), `vllm/vllm-openai:v0.32.0` not needed: `runpod/base` or `ubuntu:22.04` with python3.
Start command: the RUNPOD.md boot watchdog with CAP 180 s, then sshd. Pass: the Pod reaches EXITED or STOPPED by
itself within CAP + 120 s, and `/workspace/boot_watchdog.log` (read after a restart, or the Pod log) shows HTTP 200.
Then delete the Pod. Cost cap $0.50.

## Amendment A1 (2026-10-28, after R-56, before any run; wins over Sections 2 to 9)
R-56 (Sonnet pre-run review): AMEND_THEN_RUN. Changes:
1. Corrected motivation (FACT, E-36, recomputed by R-56): the FlashInfer path quantizes inside a TRT-LLM
   `scale_1x128` kernel (7.7 ms at batch 8), which Section 2 left out. Unfused norm + SiLU + quant: 15.9 ms at batch 8,
   33.6 ms at batch 64. Fused (deep_gemm): 21.0 ms and 44.8 ms. Expected D2 gain: about 2 ms (about 1.3%) at batch 8,
   about 11 ms (about 4.6% of device time) at batch 64. So a GO is likely only at batch 64; batch 8 will probably not
   resolve. D0 is not the comparator for D2 (different GEMM and quant kernels); D0 is reported only.
2. Profile runs D0 and D1 set `VLLM_DISABLE_COMPILE_CACHE=1`, so the "fusion pass matches" line is printed. V0 does
   not depend on it; an empty match table is reported as "not printed".
3. Kernel benchmark rewritten: buffers are allocated once; each variant is captured as 48 calls in one CUDA graph over
   4 rotating buffer sets (so a 512-token input is not served from L2) and replayed; time per call = replay time / 48.
   No clone subtraction. The norm baseline adds a torch.compile add+RMSNorm variant, and the GO condition compares
   the fused kernel with the fastest unfused variant. Scales: the timing uses the model-path format
   (`use_ue8m0 = is_deep_gemm_e8m0_used()`, printed, and TMA-aligned column-major scales); the equality check uses
   float32 scales in both paths. The roofline fraction is replaced by absolute time and a measured launch floor (an
   empty graph node), because 1 to 64 tokens are latency bound.
4. Launch the session with `AUTO_STOP=1`. Expected time about 55 min (cold compiles), about $3.70.
5. Boot watchdog CAP 4800 s (80 min, about $5.32 at $3.99/h). The watchdog is a base64-encoded script written by
   `experiments/runpod/pod_cmd.py`, so no JSON quoting is needed. The L4 rehearsal uses the same image
   (`vllm/vllm-openai:v0.32.0`), the same entrypoint override and the exact generated command, with CAP 180 s.
6. Threat added: batch 256 is above the largest CUDA-graph capture size (128 at max_num_seqs 256), so it runs without
   graphs and is noisier.
7. Contribution path (INTERPRETATION): a GO alone does not make an upstream PR. A maintainer will first ask for the
   cause. HYPOTHESIS (R-56): the fused kernels launch one block per token (64 blocks on 132 SMs at batch 64), so they
   under-use the GPU at small batch. Kernel-only prediction: the fused norm kernel is at least 1.3x slower than the
   fastest unfused path at 8 to 64 tokens and near parity at 512. Follow-ups if GO: an ncu comparison, a second dense
   model and the MoE cell, an accuracy check, then a launch-configuration change to the fused kernels.
