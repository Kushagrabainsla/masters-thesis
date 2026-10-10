# noise_shaped_vcache (I-58)

Noise-shaped (sigma-delta) quantization of the V cache along the token axis. See
`dossier/ROUTES_R-10_2026-10-07.md` and journal Section 5.14.

- `reviewer_ef_sim2.py`: the synthetic check that the R-10 reviewer (a different model, Sonnet)
  wrote. It uses iid Gaussian V and synthetic attention weights. The main session re-ran it on
  2026-10-07 and got the same output. Run it with `python3 reviewer_ef_sim2.py` (standard library
  only, about 15 s).
- E-27 (2026-10-07): `prep_texts.py` builds the texts (a compressed copy is in `data/`),
  `e27_run.py` runs the test, `e27_decide.py` applies the frozen rules, `test_vcache.py` holds the
  unit tests. Results are in `results/`. Verdict: KILL, confirmed by the post-run review R-12. See
  journal Section 6.1, E-27.

Run (about 4.5 hours on an M4 CPU):

```bash
HF_HUB_OFFLINE=1 caffeinate -i ~/Downloads/mac_hybrid_selective_recovery_poc/.venv/bin/python e27_run.py --texts ~/Downloads/e27_data/e27_texts.json --out results --phase all
```
