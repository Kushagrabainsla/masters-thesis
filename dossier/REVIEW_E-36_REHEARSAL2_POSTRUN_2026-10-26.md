# R-41: Sonnet post-run review of E-36 rehearsal 2e (2026-10-26)

Scope: the frozen A3 prediction (README Section 12) against `results/rehearsal2e-20261026/`, compared
with rehearsal 1 (`results/rehearsal-20261025/`). Method: Claude Sonnet, read-only, its own checks of
the outputs (md5, top-20 equality), logs and `extras_mac.json`.
Verdict: CONFIRM_WITH_SCOPE_LIMITS.

1. FACT: the PASS is sound. R_a, R_b and cand_marlin are separate processes with identical outputs
   (same md5, 256 of 256 including top-20 logprobs); the same comparator gives 0 of 256 for cand_triton.
2. FACT: cand_triton equals R_a on tokens for 37 of 256 (196 of 256 on the first 16 tokens); the log
   warns that it uses the default (untuned) W8A8 block-FP8 config. Exploratory only.
3. FACT: every cold warm-up equalled its measured run on 64 of 64 prompts over 64 tokens.
4. FACT: rehearsal 1 R_a compiled in-process and R_b loaded AOT; in 2e warm_R compiled in-process
   and R_a loaded AOT, and they matched. INTERPRETATION: both cache hypotheses are not supported on the
   4090; the cause stays open (GPU-specific or a one-off).
5. Action taken: the journal and README say "not reproduced on a 4090; cause open".
6. INTERPRETATION: keep A3 as a guard for the H100. Action taken: a "V1 precheck" log line after R_b.
7. FACT: `suspect_speed` on warm-ups came from a lower short-run clock (2520 against 2700 MHz), with no
   power cap or hard slowdown. Action taken: warm-ups are left out of the median and the flag.
8. Note: the `_topk_log_softmax_kernel` JIT warning is in every run (expected). The model files were
   already in the page cache in 2e, so a cold weight load was not tested.
