# R-47: Sonnet pre-run review of E-40 (2026-10-27)

Verdict: AMEND_THEN_RUN; all items handled in README amendment A1.
1. FACT, critical: MoE alias detection broken (the E-36 regex matches only "Fp8 MoE"); NVFP4 linear lines would make every MoE candidate an alias. Fixed: own regex, MoE-only comparison in cell M, V0 gate.
2. FACT, high: budget off by about 2x (accuracy 12 processes, paired 18 per candidate); cell D would be cut. Fixed: 1 candidate per cell, 6 rounds, fewer sizes and repeats, both screens first.
3. FACT, medium: JIT builds can exceed the timeout. Fixed: 900 s timeout, FlashInfer JIT cache on the volume.
4. FACT, medium: R_2 last could be cut, making the cell INVALID. Fixed: R_2 mid-screen.
5. INTERPRETATION, medium: winner's curse only affects the screen; a real gain can screen below 1.02. Fixed: no-pass KILL needs R noise at most 2%.
6. INTERPRETATION, medium: GO needs about 6% to 7%; reported.
7. FACT, medium: the AOT gate could reject custom-op backends. Fixed: reported only.
8. FACT, low: V3 is session-wide. Stated.
9. FACT, low: silent analysis failures. Fixed: exit codes logged.
10. INTERPRETATION, medium: both checkpoints should load and fit; verify the repos (both exist on the Hub: nvidia/Qwen3-8B-NVFP4, nvidia/Qwen3-30B-A3B-NVFP4).
