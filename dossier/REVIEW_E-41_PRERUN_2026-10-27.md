# R-49: Sonnet pre-run review of E-41 (2026-10-27)

Verdict: AMEND_THEN_RUN (small fixes, no code bug).
1. FACT: flags valid; V0 regex matches the E-40 dense logs; the three kernel names differ; KV cache 143 GiB,
   so max_num_seqs 512 fits; t table correct.
2. FACT: budget from E-40 timings: about 78 s per paired process; total about 65 to 75 min, about $9 to $11.
3. FACT: the 2 h watchdog could pass the $16 cap. Fixed: watchdog 6600 s.
4. FACT: KILL is hard to reach at a true tie (half-width about 3.4%). Fixed: label "no evidence of a lead".
5. INTERPRETATION: power is low; 10 rounds optional. Adopted (df 9).
6. FACT: V2 was stricter than the README. Fixed: at least 6 good rounds, failed rounds reported.
7. FACT: a resume resets the time cap. Noted in A1.
8. FACT: test record (S3) and rehearsal note (S5) missing. Fixed: unit tests and a Mac stub run (A1 item 6).
9. INTERPRETATION: one R_a run is enough (greedy decoding).
