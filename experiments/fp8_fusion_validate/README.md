# E-46: Validation of the block-FP8 fusion slowdown on H100 before a vLLM bug report (preregistration)

Status: PREDECLARED 2026-10-29, AMENDED (A1 after R-59). RUN 2026-10-29: VALIDATED. Idea: I-172. Follows E-44 (result), R-57 (post-run review) and R-58.
User request 2026-10-29: "first validate it properly, and then create a bug report for it".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
Does the E-44 result ("on H100, turning off vLLM's block-FP8 norm+quant and SiLU+quant fusions makes decode faster when
they fire") hold on the current vLLM code, on a second model, on a second backend, and without an accuracy cost?
- P1: Qwen3-8B-FP8, `--linear-backend deep_gemm`, fusions on (X) against off (Y). Replication on nightly.
- P2: Qwen3-14B-FP8, deep_gemm, on against off. Second model.
- P3: Qwen3-8B-FP8, `--linear-backend cutlass`, on against off. Second backend.
- ACC: GSM8K (1,319 problems) for P1 on and off.

## 2. Why (rubber duck)
- FACT (E-44, v0.32.0, Qwen3-8B-FP8, deep_gemm): off ÷ on = 1.041, 1.034, 1.047, 1.054 at batch 1, 8, 64, 256 (all lower
  bounds at least 1.030, 6 rounds). The fused CUDA kernels cost 2x to 3x more per call in the model than the Inductor
  kernels they replace.
- FACT (R-57): one model, one backend, one version, no accuracy run. A maintainer will ask for the latest code, a second
  model, and other backends.
- FACT (GitHub, 2026-10-29): v0.32.0 is the latest release; no commit since 2026-10-18 changes the two fused kernels,
  the two fusion passes, or the default FlashInfer DeepGEMM op. So the effect should still be on main.
- If VALIDATED: write the bug report (draft for the user) and add it to the research outputs register. If NOT
  REPRODUCED on nightly: report the E-44 result as fixed upstream or version-specific; no bug report.
- Prior (HYPOTHESIS): about four in five that P1 reproduces; P2 and P3 less sure (other shapes; cutlass GEMM is slower,
  so the same absolute cost is a smaller share).

## 3. Setup (frozen)
- 1 x H100 80GB HBM3, Secure, on-demand, network volume. Image: vLLM nightly pinned by digest,
  `vllm/vllm-openai@sha256:2e735ffeef26457f8a3ff44a62ae182870b10af37adc06ecbb08baca152365c6` (tag `nightly`,
  updated 2026-10-28). The vLLM version string and commit are recorded at the start.
- Environment as in E-44 (in-process engine, prefix caching off, CUDA bin on PATH, caches under the vLLM cache root,
  one warm-up per config, so every measured run uses a warm compile cache).
- Off = `--compilation-config '{"pass_config": {"fuse_norm_quant": false, "fuse_act_quant": false}}'`.
- Speed: `run_cell.py --mode throughput`, 128 new tokens, batch 1, 8, 64, 256, 5 repeats, `max_num_seqs` 256, GSM8K
  prompts. Profiles: one process for P1 on and off (`--profile-kernels --profile-batches 8 64`).
- Accuracy: `run_a2.py --mode full --no-score` for P1 on and off (greedy, 768 tokens).

## 4. Order
Setup and version record; ACC on, ACC off (warm-up first); profiles P1 on, off; P1, P2, P3, each with one warm-up per
config and 5 rounds (X first in odd rounds, Y first in even rounds).

## 5. Rule (frozen)
Per round log(median Y / median X). Deciding sizes 8, 64, 256. One-sided t bounds at alpha 0.05 / 3 (df 4, t 3.186;
df 3, t 3.740).
- Per pair: REPRODUCED if a deciding lower bound is at least 1.02; ABSENT if every deciding upper bound is below 1.01;
  otherwise UNCLEAR.
- VALIDATED: P1 REPRODUCED, at least one of P2 and P3 REPRODUCED, and the accuracy check passes (the paired bootstrap
  lower bound of the GSM8K difference, off minus on, is above -2.0 pp).
- NOT REPRODUCED: P1 ABSENT.
- Otherwise PARTLY VALIDATED, with each pair's label. A partial result still allows a bug report, with its scope.
- Validity: V0: the "on" profile has fused kernels (`rms_norm_per_block_quant` or `silu_and_mul_per_block_quant`) and
  the "off" profile has none; both select the same linear kernel. V2: at least 4 good rounds in P1. V3: telemetry as
  in E-44. A failed V gives INVALID.

## 6. Budget and safety
H100 SXM Secure at the live price (about $3.99/h). Setup with two model downloads about 10 min; accuracy about 8 min;
profiles about 5 min; P1 about 14 min, P2 about 22 min, P3 about 30 min (the cutlass GEMM is about 2.5x slower), with
cold compiles about 15 min. About 105 min, about $7. Session caps: no new work after 110 min, session watchdog 120 min,
`RUN_TIMEOUT` 600 s, `AUTO_STOP=1`, `AUTO_STOP_DELAY` 300 s. Boot watchdog from `experiments/runpod/pod_cmd.py`, CAP 7800 s
(130 min). Cost cap $9. The pairs run in the order P1, P2, P3, so a time cap cuts P3 first. Create-to-start in one
tool call.

## 7. Threats
Two dense Qwen3 models; one GPU type; end-to-end tokens/s includes prefill; batch 256 runs without CUDA graphs. The
nightly image is a moving target; the digest pins it. The cause inside the fused kernels is not proven here (see E-44
and R-57); the report states the measured effect and the kernel tables.

## 8. Files
`session_e46.sh`, `analyze_e46.py`, `test_e46.py`. Reused: `../safedispatch_testA/run_cell.py`,
`../safedispatch_testA2/run_a2.py`, `gsm8k.py` and `analyze_a2.py`, `../nvfp4_dispatch/analyze_e40.py`,
`../fp8_quant_fusion/analyze_e44.py`, `../runpod/pod_cmd.py`.

## Amendment A1 (2026-10-29, after R-59, before any run; wins over Sections 3 to 6)
R-59 (Sonnet pre-run review): AMEND_THEN_RUN, no blocker. Changes:
1. Image rehearsal on an L4 first (about $0.10, cap $0.50): the pinned nightly digest, the exact generated start
   command, `minCudaVersion` 13.0. Pass: the Pod starts, `python3 -c "import vllm"` works, pip or uv is present,
   `VLLM_BUILD_COMMIT` is set, and the boot watchdog stops the Pod. FACT (R-59): the image is the current `nightly`
   index, CUDA 13.0.2, vLLM commit edad8b03 (2026-10-28), about 9.7 GB compressed; its entrypoint is `vllm serve`, so
   the start command overrides it (as in E-36 to E-44). Network volume 60 GB (models about 25 GB).
2. Setup records `versions.txt` (with `VLLM_BUILD_COMMIT`), `collect_env.txt` and `nvidia-smi.txt`. The `datasets`
   install falls back from `python3 -m pip` to `pip` to `uv pip --system`.
3. Scope (who is affected by default): one profile run of the default engine (`auto`) after P1 shows whether the
   fusion fires there on nightly.
4. Reproducer for maintainers: after P1, 3 alternating on/off rounds of `vllm bench latency` (default multiprocess
   engine, batch 64, 128 in and 128 out, 10 iterations); the session writes `repro.md` with the exact commands, digest
   and commit. B200 is not tested; the report says so.
5. V0 for every pair: the on run lists norm_quant in "Enabled custom fusions", the off run does not, and both select
   the same linear kernel; a pair that fails is INVALID.
6. Waste guards: warm-ups get a 1200 s timeout and one retry; a pair is skipped if a warm-up is not ok; a pair stops
   if round 1 fails in both configs; an early V0 gate after the P1 profiles stops the session (and the Pod) if the
   fusion does not fire with "on" or fires with "off".
7. Rule: a bug report only if P1 is REPRODUCED. A P1 that is UNCLEAR gives NOT CONFIRMED (rerun or no report).
   PARTLY VALIDATED needs P1 REPRODUCED. Bounds are computed for 4 or 5 good rounds (df 3 or 4).
8. Order: P1, then the extras (default profile, bench), then P2, then P3, so a time cap cuts P3 first. Caps: no new
   work after 120 min, session watchdog 130 min, boot CAP 8400 s (140 min). Cost cap $10.50 for the H100, plus $0.50 for
   the L4 rehearsal. Launch with `AUTO_STOP=1`. The analysis runs on the Pod with the sibling folders copied
   (fp8_quant_cause, fp8_quant_fusion, nvfp4_dispatch, safedispatch_testA, safedispatch_testA2, ledger, runpod).
9. Test record (S3, S5, Mac): `test_e46.py` 4 pass; a stub session runs 50 steps with the right decision; a second
   stub with the fusion firing in "off" stops at the V0 gate before any pair.

## Result
Result (2026-10-29, Pod rowtn7ir0qu7t7, 1 x H100 80GB HBM3, US-NE-1, nightly digest = vLLM 0.33.0rc1.dev157+gedad8b03c,
torch 2.14.0+cu130; 05:14 to 06:39 UTC, about $5.70; L4 image rehearsal dumpjkspwsufcv PASS, about $0.10; 145 checksums
match): frozen decision VALIDATED (P1, P2, P3 REPRODUCED; accuracy pass). V0 (profiles and every pair) and V3 pass;
5 of 5 rounds in every pair.
FACT, fusions off ÷ on (one-sided bounds, alpha 0.05/3, df 4):
- P1 Qwen3-8B-FP8, deep_gemm: batch 1 1.026, 8 1.022 [1.019, 1.025], 64 1.043 [1.034, 1.052], 256 1.061 [1.051, 1.071].
- P2 Qwen3-14B-FP8, deep_gemm: 1.039, 8 1.027 [1.024, 1.031], 64 1.047 [1.027, 1.068], 256 1.054 [1.048, 1.059].
- P3 Qwen3-8B-FP8, cutlass: 0.998, 8 1.002 [0.997, 1.007], 64 1.021 [1.018, 1.024], 256 1.035 [1.031, 1.040].
FACT, accuracy (GSM8K, 1,319, deep_gemm): off minus on +0.38 pp [-0.61, +1.36]; 91 of 1,319 token lists identical.
FACT, `vllm bench latency` (multiprocess engine, batch 64, 128 in, 128 out, 3 alternating rounds): median latency on ÷ off
= 1.046 (on 0.908 to 0.942 s, off 0.895 to 0.901 s).
FACT, default engine (nightly, `auto`): FlashInferFp8DeepGEMMDynamicBlockScaledKernel; "Enabled custom fusions:
norm_quant, act_quant", but no fused kernel runs.
INTERPRETATION: the E-44 finding holds on current vLLM main code, on a second model and on a second backend: when the
block-FP8 norm+quant and SiLU+quant fusions fire on H100, decode is 2% to 6% slower at batch 8 to 256 (cutlass: only at
batch 64 and 256), without an accuracy benefit. Default-engine users are not affected. Not tested: B200, MoE models,
TP above 1. The cause inside the fused kernels is not proven (E-44 and R-57 kernel tables).
