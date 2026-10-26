# R-42: Sonnet post-run review of E-36 Session 1 on the H100 (2026-10-26)

Scope: `experiments/safedispatch_testA/results/session1-h100-20261026/`, README Section 13, and the
main-session claims. Method: Claude Sonnet, read-only; own checks of `summary.json`, `extras_mac.json`,
`kernels.json` and the outputs. Verdict: CONFIRM_WITH_SCOPE_LIMITS.

1. FACT: INCONCLUSIVE is what `compare.py` computes (safe set {R}). Flip range 87.5% to 95.3%.
2. FACT: STATIC-WINNER read against all backends fails only at dense batch 32 (deep_gemm 0.974 of R).
3. FACT: R is the fastest eligible backend at every frozen batch size; dense deep_gemm gaps at batch 32
   and 64 are 2 to 3 times the R noise; the Marlin W8A16 control beats R in the MoE cell at batch 4 to 32
   by 1% to 5% (B2).
4. INTERPRETATION: "close to the KILL condition" is not justified. Exact match flags any numerical
   change: eager mode with the same kernels flips 233 and 238 of 256. Margins at divergence are 0.25 and
   the first divergence is near token 24 to 32 (near-tie BF16 quantization, then chaotic divergence).
   Safe statement: no exact-output-preserving alternative exists; tolerance-contract headroom is untested.
5. INTERPRETATION: a tolerance contract is legitimate only as a new, preregistered Test A2 (teacher-forced
   scoring of R's tokens; GSM8K accuracy primary). Do not relabel Session 1.
6. FACT: B0 repeats exact; B1 same pattern; B2 and B3 confirmed (eager 3.3x to 10.8x slower). R's batch
   invariance across max_num_seqs 8 and 64 was not measured.
7. FACT: B1 cold_vs_warm disagreements come from batch composition; the Phase A MoE Marlin warm-up (50%)
   is a nondeterminism flag.
8. FACT: dense deep_gemm and R share `deep_gemm::sm90_fp8_gemm_1d2d_impl`; R uses
   `per_token_group_quant_8bit_register_kernel`, deep_gemm uses fused `silu_and_mul_per_block_quant_kernel`
   and `rms_norm_per_block_quant_kernel`. The flips come from activation quantization and fusion.
9. FACT: validity holds (one UUID, no suspect run, stable microbenchmark, AOT hits); the second telemetry
   logger is an equal small load.

Recommended next step: preregister Test A2 with a tolerance contract, because only that contract can
show whether any speed headroom is usable.
