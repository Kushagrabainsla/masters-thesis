# R-50: Sonnet post-run review of E-41 (2026-10-27)

Verdict: CONFIRM_WITH_SCOPE_LIMITS.
1. FACT: bounds recomputed from run.json match summary.json (cutlass 256: 1.0154 [0.9883, 1.0431]; 512:
   1.0013 [0.9852, 1.0177]). The rule gives INCONCLUSIVE: no lower bound reaches 1.03; the cutlass 256
   upper bound (1.043) blocks KILL; cudnn fails V2 (3 good rounds).
2. FACT: the "no evidence of a lead" label comes from A1 item 3 (written before the run). It must not read
   as "no lead": a cutlass gain of up to about 4% at batch 256 is not excluded. No rule was applied after
   the fact.
3. FACT: token_ids equal on 1,319 of 1,319 items for both candidates; one kernel line per run names a
   different kernel; cudnn took 16.0 s against 13.3 s, init 96 s against 31 s; no fallback logged. The
   lm_head and attention are not NVFP4 linears and are the same in all runs. The A1 gate cannot separate
   these backends. INTERPRETATION: the bit-equal hypothesis (same block-scaled FP4 MMA K order, no
   split-K, shared activation quantization) is plausible. Cheapest test: one GEMM on the same input with
   each backend, torch.equal, with a negative control that should differ.
4. FACT: cudnn C runs failed in rounds 3, 5 to 10; R never failed; no order or warm-up pattern.
   INTERPRETATION: likely an upstream bug (a shape not warmed before capture); a harness cause is not
   excluded; the 3 passing rounds are survivors.
5. INTERPRETATION: closing I-23 for NVFP4 is justified as "default near-optimal here", with the scope
   limits in D-64.
