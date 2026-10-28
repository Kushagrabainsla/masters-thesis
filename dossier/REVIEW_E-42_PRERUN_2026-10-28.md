# R-51: Sonnet pre-run review of E-42 (2026-10-28)

Verdict: AMEND_THEN_RUN. All fixes applied (README Amendment A1).
1. FACT: no kernel check, so a silent alias gives a false BIT_EQUAL. Fixed (V0 kernel names).
2. FACT: marlin is a large-difference control only. Fixed: R_m with max_num_seqs 64.
3. FACT: empty logprob lists compare equal. Fixed: 20 entries required.
4. FACT: a failed hypothesis backend gave "fails". Fixed: INCONCLUSIVE.
5. FACT: tie order in topk(). Fixed: compare as maps. JSON float round-trip is exact.
6. FACT: compare stop rule is correct; max difference now over all entries.
7. INTERPRETATION: the README said "one batch shape" (wrong); graph mode should run earlier. Fixed.
8. FACT: the watchdog plus boot could pass the $6 cap. Fixed: watchdog 40 min, cap $7.
9. INTERPRETATION: the test covers the whole linear path; the result changes only one D-64 caveat, so
   keep the cost near $5.
