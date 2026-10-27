# R-48: Sonnet post-run review of E-40 (2026-10-27)

Verdict: CONFIRM_WITH_SCOPE_LIMITS.

1. FACT: INCONCLUSIVE is what the frozen rule gives. Cell D: V0 false (no reference kernel parsed) and V1
   false (R noise 6.98% at batch 128). Cell M: no non-alias screen pass, but R noise 4.27% at batch 128 is
   above the 2% line (A1 item 4), so KILL is barred.
2. FACT: the dense line "Using FlashInferCuteDslNvFp4LinearKernel for NVFP4 GEMM" was not matched. The
   selected flashinfer_cutedsl was an alias (accuracy +0.00 pp [0.00, 0.00]; batch 8 paired 1.016
   [1.005, 1.028]). With the regex fixed, cell D stays INCONCLUSIVE on V1.
3. INTERPRETATION: "the default is fastest at batch 1 to 128" holds for cell M (every real backend 0.42 to
   0.95). For cell D it holds at batch 1 to 32 only; at 128 cudnn is 1.015 inside 7% noise. Scope limits:
   one-process screen, no paired data on a real backend, one model per cell, b12x failed to start,
   end-to-end tokens/s.
4. FACT: the batch-256 lead is inside noise. The alias paired ratio at 256 is 0.943 to 1.038 over 6
   rounds (mean 0.97, sd about 3.4%). R_1 and R_2 differ by 4.5% at 256. A ratio of 1.04 to 1.07 from one
   process is 1 to 2 sd. Only cudnn at 1.07 is a mild outlier, and it is 0.85 to 0.89 at batch 1 to 32.
5. INTERPRETATION: a paired test is worth it only if cheap and preregistered: dense cell, cudnn first,
   fixed V0 regex, alias check, accuracy before any GO. GO needs a true gain of about 6% or more. A gain
   under 7% on one GPU and one 8B model is small for the thesis; skipping the test and keeping it as a
   stated limit is also defensible.
