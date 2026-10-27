# E-41: Dense NVFP4 large-batch lead on one B200 (preregistration)

Status: PREDECLARED 2026-10-27, AMENDED (A1 after R-49). RUN 2026-10-27: INCONCLUSIVE. Idea: I-23. Follows E-40 (open lead). User request 2026-10-27:
"plan a short paired test of the dense batch-256 lead (cudnn, flashinfer_cutlass) on a B200."
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
For `nvidia/Qwen3-8B-NVFP4` on one B200, is `--linear-backend flashinfer_cudnn` or
`--linear-backend flashinfer_cutlass` at least 3% faster than the default R (FlashInfer CuTe-DSL) at
batch 256 or 512, with GSM8K accuracy within 2 pp of R?

## 2. Why (rubber duck)
- FACT (E-40 screen, one process each): at batch 256, flashinfer_cudnn 1.07, flashinfer_cutlass 1.05
  and cutlass 1.04 against R; at batch 1 to 128 both are 1% to 19% slower. R's own two runs differed by
  4.4% at batch 256, so the lead may be noise.
- If GO: the fastest accuracy-preserving dense NVFP4 backend changes at large batch, which is the I-23
  premise (PIVOT-grade, as in E-39). If KILL: the E-40 lead was noise, and I-23 is closed for NVFP4 at
  these settings.
- FACT (R-48): the E-40 alias paired ratio at batch 256 has sd about 3.4% over 6 rounds, so the 1.04 to
  1.07 screen ratios are 1 to 2 sd of noise. Batch 512 is added because a crossover grows with batch.
- Prior (HYPOTHESIS): about one in four that a candidate reaches a lower bound of 1.03.

## 3. Setup (frozen)
1 x B200 (sm_100), Secure, on-demand, network volume, `vllm/vllm-openai:v0.32.0` with the E-36 to E-40
fixes. Speed: E-36 `run_cell.py --mode throughput`, end-to-end tokens/s for 128 new tokens at batch 128,
256, 384, 512, 5 repeats, `max_num_seqs` 512, GSM8K prompts. Accuracy: E-38 `run_a2.py --no-score` on
GSM8K (1,319), greedy, 768 tokens, for R and both candidates (A1 as in E-38: paired bootstrap, lower
bound of the difference above -2.0 pp).

## 4. Order
Accuracy first (warm-up, R_a, then each candidate with its warm-up). Then for each candidate in the order
flashinfer_cudnn, flashinfer_cutlass: one warm-up for R and C, then 8 paired rounds (R, C in odd rounds
and C, R in even rounds).

## 5. Rule (frozen)
Deciding batch sizes: 256 and 512. Per round: log(median C / median R). One-sided t bounds at alpha
0.05 / 4 (Bonferroni over 2 sizes x 2 candidates; df 7, t 2.841). Batch 128 and 384 are reported only.
- GO: a candidate passes A1 and has a lower bound of at least 1.03 at a deciding size.
- KILL: for both candidates, either A1 fails or every deciding upper bound is below 1.03.
- Otherwise INCONCLUSIVE, with the reason named.
- Validity: V0: the R log names its linear kernel ("Using X for NVFP4 GEMM", E-40 lesson). V2: at least
  6 complete rounds per candidate with every run ok. V3: at least 100 telemetry rows parse and at most 2%
  of busy samples show a hard slowdown. The AOT load is reported only.
- Power (INTERPRETATION): with a 3% per-round sd, the half-width is about 3%, so GO needs a true gain of
  about 6% or more.

## 6. Budget
B200 at the live price (about $7.99/h). Setup about 10 min (image pull, 6 GB model); accuracy 5
processes at about 3 min; paired 2 x 18 processes at about 1.5 to 2 min. About 1 h 30 min; cap: no new
work after 1 h 45 min, watchdog 2 h, cost cap $16.

## 7. Threats
One model, one GPU, one task, end-to-end tokens/s (includes prefill; at batch 256 to 512 prefill is a
larger share). A GO is a speed fact for this setup, not a dispatch system.

## 8. Files
`session_e41.sh`, `analyze_e41.py`, `test_e41.py`. Reused: `../safedispatch_testA/run_cell.py`,
`../safedispatch_testA2/run_a2.py` and `gsm8k.py`, `../nvfp4_dispatch/analyze_e40.py` (accept, paired).

## Amendment A1 (2026-10-27, after R-49, before any run; wins over Sections 4 to 6)
R-49 (Sonnet pre-run review): AMEND_THEN_RUN, no code bug. Changes:
1. 10 paired rounds per candidate (df 9, t 2.685). With a 3.4% sd, GO needs a true gain of about 6%.
2. V2: at least 6 good rounds. A failed round is dropped and reported, not a reason for INVALID.
3. KILL is hard to reach at a true tie. An INCONCLUSIVE with every deciding ratio below 1.03 is reported
   as "no evidence of a lead".
4. Budget (FACT, E-40 timings): about 78 s per paired process; expected about 75 min and about $11 with
   the copy window. Watchdog at 1 h 50 min, so the cost stays under the $16 cap.
5. A resume starts a new time cap. If a resume is necessary, watch the total cost by hand.
6. Test record (S3, S5, 2026-10-27, Mac): `test_e41.py` 3 pass; on the real E-40 logs the dense line is
   parsed (V0 true). Stub session (stub run_cell.py and run_a2.py, shims for timeout, pip, nvidia-smi):
   a failed run is logged with exit 1, a resume redoes only that run (49 skips), the time cap skips
   new work, and the decision is GO for a stub 1.08 gain and not for a stub 1.00.

## Result
Result (2026-10-27, Pod rm31hzrhgrql7x, 1 x B200 sm_100, US-NE-1, 22:34 to 23:56, about $10.70 by the
balance; 140 checksums match): frozen decision INCONCLUSIVE (flashinfer_cudnn paired speed incomplete;
flashinfer_cutlass not resolved; every deciding ratio below 1.03: no evidence of a lead). V0 and V3 pass.
FACT, accuracy: R, flashinfer_cudnn and flashinfer_cutlass give the same tokens on all 1,319 GSM8K items
(92.27%), with three different kernels in the logs (CuTe-DSL, cuDNN, CUTLASS).
FACT, flashinfer_cudnn: 7 of 10 C runs failed at start with `CudnnPlanBuildInCaptureError` (FlashInfer
refuses a cuDNN plan build during CUDA graph capture). 3 good rounds (V2 needs 6). Ratios on 3 rounds:
256 1.017 [0.903, 1.145], 512 1.002 [0.951, 1.056].
FACT, flashinfer_cutlass (10 good rounds; ratio [one-sided bounds]): 128 0.937 [0.927, 0.947],
256 1.015 [0.988, 1.043], 384 1.010 [0.986, 1.035], 512 1.001 [0.985, 1.018].
INTERPRETATION: the E-40 lead (cudnn 1.07, cutlass 1.05 at batch 256) was mostly noise. Cutlass is at
most about 4% faster at batch 256 and is 6% slower at batch 128. cuDNN is not usable with CUDA graphs
in this image without the FlashInfer override. The default stays the best choice across batch sizes.
