# E-42: Are the NVFP4 GEMM kernels bit-equal on B200? (preregistration)

Status: PREDECLARED 2026-10-27, AMENDED (A1 after R-51). RUN 2026-10-28: HYPOTHESIS HOLDS. Idea: I-23 (closed by D-64; this test explains a result, it does not
reopen I-23). Source: E-41 audit and R-50 item 3. User request 2026-10-27: "lets do the recommended steps in
order" (step 3: a small GPU check that compares the three NVFP4 kernels bit for bit).
Labels: FACT, INTERPRETATION, HYPOTHESIS.

## 1. Question
For `nvidia/Qwen3-8B-NVFP4` on one B200, do the NVFP4 linear backends give bit-equal top-20 logprobs at every
generated position, compared with the default R (CuTe-DSL)?

## 2. Why (rubber duck)
- FACT (E-41): R, flashinfer_cudnn and flashinfer_cutlass gave the same tokens on 1,319 of 1,319 GSM8K items.
  On H100 FP8 (E-38), real kernel changes changed most outputs (Marlin 149 of 1,319 the same).
- HYPOTHESIS (R-50): the FP4 x FP4 kernels use the same Blackwell block-scaled MMA with the same K order and
  no split-K, so the GEMM outputs are bit-equal. If true, the E-41 accuracy gate (A1) was vacuous by
  construction, and the thesis must say so. If false, the kernels differ slightly and greedy decoding hid it.
- Why logprobs: equal tokens can hide small logit differences. Logprobs are float32 values that JSON stores
  exactly (Python float repr), so an exact compare is valid.

## 3. Setup (frozen)
1 x B200, Secure, on-demand, network volume, `vllm/vllm-openai:v0.32.0`, E-41 environment. E-36
`run_cell.py --mode outputs`: the first 256 GSM8K prompts, greedy, 64 new tokens, top-20 logprobs,
`max_num_seqs` 256 (one batch), `--enforce-eager` (removes the cudnn CUDA-graph start failure).
Runs, in this order: R_1, flashinfer_cudnn, flashinfer_cutlass, cutlass (vLLM CUTLASS), flashinfer_trtllm,
marlin (negative control: W4A16 dequant, different numerics), R_2 (positive control). Then two graph-mode
runs without `--enforce-eager`: R_g, flashinfer_cutlass_g.

## 4. Metrics (per run against R_1)
Compare each prompt up to its first token difference. Per position: bit-equal if the top-20 lists (token
ids and logprob values) are exactly equal. Report: prompts with equal tokens; bit-equal fraction of
compared positions; max absolute top-1 logprob difference.

## 5. Rule (frozen)
- V1 (determinism): R_2 is bit-equal to R_1 at every position. If not, INVALID.
- V2 (power): marlin is not bit-equal at 1% or more of positions. If not, INVALID (the test cannot see a
  difference).
- Per backend: BIT_EQUAL if every compared position is bit-equal and all 256 token lists are equal.
  NEAR if tokens are equal for all 256 prompts but at least one position is not bit-equal. DIFFERENT
  otherwise. A backend that fails to start is FAILED.
- Graph mode: R_g against R_1 and flashinfer_cutlass_g against R_g are reported with the same labels.
- The hypothesis holds if flashinfer_cudnn and flashinfer_cutlass are BIT_EQUAL.

## 6. Budget
B200 at about $7.99/h. Setup about 7 min; 9 processes at about 1.5 min. About 25 min. No new work after
35 min; watchdog 45 min; cost cap $6.

## 7. Threats
One model, one prompt set, one batch shape (256). Bit-equality at this shape does not prove it at all
shapes. Eager mode is the main mode; graph mode is a check.

## 8. Files
`session_e42.sh`, `analyze_e42.py`, `test_e42.py`. Reused: `../safedispatch_testA/run_cell.py`,
`../safedispatch_testA2/gsm8k.py`, the E-41 session safety code.

## Amendment A1 (2026-10-28, after R-51, before any run; wins over Sections 3 to 7)
R-51 (Sonnet pre-run review): AMEND_THEN_RUN. Changes:
1. V0: each run's log must name its NVFP4 linear kernel ("Using X for NVFP4 GEMM"). Every candidate and marlin
   must name a kernel different from R and from each other; otherwise INVALID (a silent alias would give a
   false BIT_EQUAL).
2. New small-change control R_m: R with `--max-num-seqs 64` (other GEMM M sizes). If R_m differs from R_1,
   the test can see small numeric changes ("sensitivity shown"). If R_m is BIT_EQUAL, the result is
   reported with "sensitivity to M changes not shown". Marlin stays the large-difference control.
3. A position counts only if both lists have 20 entries; a short list makes the run INVALID.
4. If cudnn or flashinfer_cutlass fails to start, the decision is INCONCLUSIVE (not "fails").
5. Lists are compared as {token_id: logprob} maps, so tie order cannot matter; the max difference is
   taken over all shared entries.
6. Run order: R_1, R_2, cudnn, flashinfer_cutlass, marlin, R_g, flashinfer_cutlass_g, R_m, cutlass,
   flashinfer_trtllm. R_g against R_1 is reported only (compile fusions differ from eager).
7. Threats (corrects Section 7): decode M falls from 256 as prompts finish, so several M sizes are covered,
   but M below about 8 is not. The test covers the whole NVFP4 linear path (activation quantization plus
   GEMM), not the GEMM alone.
8. Budget: about 20 to 25 min of script plus about 5 min of boot and the 10 min copy window: about $5.3.
   Watchdog 40 min of script time. Cost cap $7.
9. Test record (S3, S5, Mac): `test_e42.py` 5 pass; a stub session runs all 10 runs and gives the expected
   decision.

## Result
Result (2026-10-28, Pod r37oa2sniy92lq, 1 x B200 sm_100, US-NC-2, Pod volume disk; 59 checksums match):
HYPOTHESIS HOLDS. V0, V1, V2 pass; M sensitivity shown.
FACT (eager, 256 prompts, 16,384 positions of top-20 logprobs against R_1, CuTe-DSL):
- flashinfer_cudnn, flashinfer_cutlass, cutlass (vLLM CUTLASS), flashinfer_trtllm: BIT_EQUAL (256 of 256
  token lists, every position bit-equal, max difference 0.0). Each log names its own kernel.
- R_2 (same settings): BIT_EQUAL. Marlin (W4A16): DIFFERENT (12 of 256 token lists the same, max 13.9).
- R_m (R with max_num_seqs 64): DIFFERENT (132 of 256 the same, 67.7% of positions bit-equal). A change of
  batch composition changes the logits (R-52: composition sensitivity, not GEMM-level; marlin is the
  kernel-level control).
- Graph mode: flashinfer_cutlass_g is BIT_EQUAL to R_g. R_g against R_1 (graph against eager) is
  DIFFERENT (7 of 256 the same): the compiled graph changes the numerics, but not between these backends.
INTERPRETATION: the five FP4 x FP4 NVFP4 linear kernels in vLLM v0.32.0 give bit-equal results for this
model on B200, in eager and graph mode. So the E-41 accuracy gate (A1) was vacuous for these kernels on this model and
shape, and here the backend choice among them is a speed choice only. Batch composition and compile mode change the
numerics far more than the kernel choice.

## Incident (cost)
FACT: the Pod billed from 01:14 UTC, but the session started at 03:34 UTC. The agent waited for a 2-minute
background timer, and the wake-up reached it about 2 h 20 min late. The B200 was idle and billed during
that time. Cost: about $21.60 by the balance (billing $14.08 through 03:00, last hour not yet posted),
against a $7 cap: about $15 over. The session watchdog starts with the session, so it did not cover this
gap. Fix: `docs/experiments/RUNPOD.md` Section 3 now requires a boot watchdog in the start command, and the
create-to-start steps in one tool call.

## Post-run review R-52 (2026-10-28): CONFIRM_WITH_SCOPE_LIMITS
Recomputed independently: all labels match. Scope limits for any citation: one dense model
(Qwen3-8B-NVFP4), 256 GSM8K prompts with 64 greedy tokens, one B200, vLLM v0.32.0, max_num_seqs 256 (M below
about 8 not tested), eager mode for all backends (graph mode only for R and flashinfer_cutlass), whole
linear path (activation quantization plus GEMM). Do not generalize to other GPUs, models or versions.
