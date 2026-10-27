# R-43: Sonnet pre-run review of E-38 (I-23 Test A2) (2026-10-26)

Scope: `experiments/safedispatch_testA2/` (README preregistration, gsm8k.py, run_a2.py, analyze_a2.py,
test_a2.py, session_a2.sh) and the first Mac CPU dry run. Method: Claude Sonnet, read-only.
Verdict: AMEND_THEN_RUN. All findings are handled in README amendment A1.

| # | Label, severity | Finding | Action |
|---|---|---|---|
| 1 | FACT, high | Cell D speed tests ran before cell M, so the cap could cut the only GO path (MoE Marlin); one speed test is 6 processes (15 to 25 min), not 6 min | Order: M accuracy, M Marlin speed, D accuracy; estimate 20 to 25 min per speed test |
| 2 | INTERPRETATION, high | A2 is close to vacuous: same FP8 weights, so a candidate's KL holds only kernel noise, while the anchor holds the whole weight-quantization error | A2 now needs KL and top-1 disagreement at most 0.5 x the anchor's |
| 3 | FACT, high | Teacher forcing tests the prefill path only; the speed difference is in decode | Each config records generation logprobs; `decode_prefix.json` compares them with R_a up to the first divergence (reported) |
| 4 | FACT, high | Speed power: gains of 1% to 5% against a 3% band; 5 repeats are pseudo-replicates (n = 2 against 2 processes); Bonferroni only over batch sizes | KILL is "no gain of 3% or more that the design can resolve"; the 3% bar is justified by product value; both stated |
| 5 | FACT, medium | Code and README differed: INVALID for the whole run, a broken anchor fallback, the speed baseline text, errored candidates allowing KILL | Per-cell INVALID; fallback removed (missing anchor gives INCONCLUSIVE); text aligned; errored candidates give INCONCLUSIVE |
| 6 | FACT, medium | PIVOT was unreachable because extra speed tests were cut first | Speed only for MoE Marlin; the other candidates are judged from E-36 data (slower than R), stated |
| 7 | INTERPRETATION, medium | Goalpost risk: the contract was set after seeing E-36 data; the 2 pp margin may exceed the FP8-against-BF16 gap | Freeze statement; BF16 accuracy difference and an R_b null check reported |
| 8 | FACT, medium | The top-20 approx KL is biased low and the mean is driven by a few positions | Median and p99 reported |
| 9 | FACT, low | Silent truncation of scored positions | The run fails if any position is missing or the counts differ |
| 10 | FACT, low | V3 can fail on prefill nondeterminism; V4 assumes the alias | Pre-declared: a V4 failure means "the alias claim is false" |
| 11 | FACT, medium | The models are about 117 GB; the MoE BF16 anchor leaves about 7 to 9 GB; scoring all prompts at once may use 1.5 to 4 GB host RAM | Volume 200 GB; anchor with `max_num_seqs` 64; scoring in chunks of 256 |
| 12 | FACT, low | No AOT check in speed runs | Added |
| 13 | FACT, medium | Budget underestimated | Re-estimated: about 2 h 15 min to 2 h 45 min; cap 3 h new runs, 3 h 30 min Pod, $15 |
