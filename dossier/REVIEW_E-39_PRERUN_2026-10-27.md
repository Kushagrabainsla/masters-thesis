# R-45: Sonnet pre-run review of E-39 (2026-10-27)

Verdict: AMEND_THEN_RUN. Fixed in README amendment A1.
1. Power: n = 6 against 6 with Bonferroni over 7 sizes needed a 7% to 9% gain; likely INCONCLUSIVE. Fix: 10 paired rounds, one-sided t, 4 deciding sizes.
2. Bootstrap of a median at n = 6 is discrete and anti-conservative. Fix: paired log ratios with t bounds.
3. V1 (R CV at most 3%) would fail about half of normal runs. Fix: reported only.
4. V3 passed with no data if the telemetry field failed. Fix: at least 100 parsed rows; a hard-slowdown fraction, not a single sample.
5. The metric includes prefill. Fix: named end-to-end tokens/s.
6. Dropping A2 is a goalpost move in form. Fix: "retired post hoc" with reason; a GO is PIVOT-grade.
7. Accuracy reuse is thin (one task); the crossover at 256 is not re-measured. Fix: 128 and 256 added (reported).
8. Order is adequate; add a drift check. Done.
9. Budget underestimated. Fix: about 90 min, cap $8, whole-round time cap.
10. Minor test gaps. Fix: tests for unparsed telemetry, too few rounds, cold runs (7 tests pass).
