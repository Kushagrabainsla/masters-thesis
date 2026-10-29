# R-59: Sonnet pre-run review of E-46 (2026-10-29)

Verdict: AMEND_THEN_RUN, no blocker. All fixes applied (README Amendment A1).
1. FACT: the digest is the current nightly (CUDA 13.0.2, commit edad8b03, entrypoint `vllm serve`); Runpod digest
   support, python3 and pip are unverified. Fix: L4 rehearsal with the digest; minCudaVersion 13.0; 60 GB volume; pip
   fallback.
2. FACT: on main, pass_config fuse flags, `--linear-backend deep_gemm/cutlass`, the "Selected X for Fp8LinearMethod"
   line, VLLM_ENABLE_V1_MULTIPROCESSING and both fused kernel names still exist. Fix: record the build commit and
   collect_env.
3. FACT: the default engine had 0 fusion matches in E-44. Fix: a default-engine profile on nightly; state the scope.
4. INTERPRETATION: maintainers use the multiprocess engine and `vllm bench`. Fix: a `vllm bench latency` on/off
   reproducer and `repro.md`. B200 not testable here.
5. FACT: V0 covered P1 only. Fix: V0 for every pair from the X_1 and Y_1 logs.
6. FACT: a timed-out warm-up could waste about 100 min. Fix: longer warm-up timeout with a retry, skip guards, and an
   early V0 gate.
7. FACT: t values and directions correct. Fix: a report only if P1 reproduces; tests for UNCLEAR P1 and 4 rounds.
8. FACT: budget about 100 min plus about 15 min for the extras. Fix: order P1, extras, P2, P3; caps raised to
   $10.50 with the user's approval.
9. FACT: the analysis needs the sibling folders on the Pod. Fix: copy them (as in E-44).
