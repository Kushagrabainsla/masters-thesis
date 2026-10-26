# R-40: Sonnet pre-run review of E-36 amendment A3 (2026-10-25)

Scope: E-36 amendment A3 (warm-up process per config), `session1.sh`, `run_cell.py`, `extras.py`, and
the rehearsal 1 data in `experiments/safedispatch_testA/results/rehearsal-20261025/`.
Method: Claude Sonnet, read-only; `sh -n`, a Python parse, and its own analysis of the rehearsal files.
Verdict: AMEND_THEN_RUN. A3 fits the data but was not yet run on real hardware. No blocking bug.

| # | Label, severity | Finding | Action |
|---|---|---|---|
| 1 | FACT, high | Cold compile alone does not explain the data: a no-op-flag run also compiled cold but matched the warm runs at position 0 on 256 of 256 (tokens 255 of 256). Best fit: first compile in a fresh container (empty Inductor, Triton, autotune disk caches); second suspect: in-process compile against an AOT-loaded artifact. Mechanism not proven | README Section 12 rewritten; no causal claim |
| 2 | FACT, medium | "Four later runs agreed 100%" was wrong: three later same-config runs; the no-op run agreed on 255 of 256 | Corrected |
| 3 | FACT, high | A3 untested on real hardware | Rehearsal 2 with a frozen prediction (README Section 12) |
| 4 | INTERPRETATION, high | Warm-up helps only if the measured run takes the AOT-load path; caches outside `/workspace` are lost on restart | AOT check with WARNING and `aot_hit`; Inductor and Triton caches under `VLLM_CACHE_ROOT` |
| 5 | INTERPRETATION, medium | 16 x 16 warm-up misses the 64-sequence logprob path and mixed prefill and decode; in-memory autotune cannot be warmed | Warm-up is now 64 prompts x 64 tokens with logprobs |
| 6 | INTERPRETATION, high | V1 now compares two AOT-loaded runs; candidate process noise is not measured; `cold_vs_warm` on 16-token prefixes is too weak | Exact top-20 logprob comparison at positions 0 to 15; residual threat recorded; MoE alias `triton` is the zero-flip control |
| 7 | INTERPRETATION, low | Phase A cap 3 h 45 min is inside the Pod limit and cost cap; budget about 4 min per MoE warm-up; Phase B loses about 70 min | Accepted |
| 8 | FACT, low | No shell or Python bug; a failed warm-up would leave a cold measured run | WARNING added; warm folders left out of the B2 ranking |
