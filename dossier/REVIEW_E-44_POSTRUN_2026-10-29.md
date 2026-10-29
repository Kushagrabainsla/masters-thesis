# R-57: Sonnet post-run review of E-44 (2026-10-29)

Verdict: CONFIRM_WITH_SCOPE_LIMITS.
1. FACT: paired ratios recomputed from run.json match summary.json; all deciding lower bounds are at least 1.02; the
   kernel-benchmark condition fails (SiLU at least 1.3x at 2 of 5 sizes, norm at most 1.24x), so the rule gives
   INCONCLUSIVE. No rule was applied after the fact.
2. FACT: D1 and D2 differ only in pass_config. D1 has 2272 fused norm and 1152 fused SiLU calls; D2 has none. The
   norm pass did fire (log: "RMSNormQuantFusionPass ... Replaced 1/2/1 patterns"); its absence from the match table is
   a logging artifact. V0 is sound.
3. FACT: D1 minus D2 device time: 3.3 ms per 32 steps at batch 8 (2.1%), 17.5 ms at batch 64 (6.7%). GEMM within 1%.
   No extra copy or layout kernels. In the model, fused SiLU takes 3.8 us at batch 8 and 19.7 us at 64, against
   Inductor SiLU+quant 2.3 us and 6.3 us in D2. D1 has two extra Inductor kernels (4.2 ms at batch 8). No batch-1
   profile. INTERPRETATION: the fused CUDA kernels cost more in the model than the Inductor kernels they replace.
3b. FACT: the kernel benchmark does not predict the in-model cost (fused SiLU 5.8 us in the benchmark against 19.7 us in
   the model at 64 tokens). HYPOTHESIS: input layout or cache state differs. The D2 "unfused" path is Inductor-fused
   norm+quant in Triton.
4. FACT: only deep_gemm was measured end to end. cutlass and triton run the same fused kernels (E-36) with similar
   times, with no unfused comparator. "Default users are not affected" holds only in this setup.
5. INTERPRETATION: a vLLM performance issue with a reproducer is justified, with the three configs, the paired table,
   the kernel tables, the in-model against benchmark gap and the Inductor framing. Caveats: one model, one GPU, one
   version, no cause, no accuracy run, no batch-1 profile, batch 256 without CUDA graphs. A maintainer will ask for
   nsys or ncu, a second model and a warm-cache check. The 3.5% FP8 byte mismatch is probably rounding; unverified.
