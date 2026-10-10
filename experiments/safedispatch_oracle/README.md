# SafeDispatch oracle harness (Mac pilot)

Pilot for idea I-23 in `RESEARCH_JOURNAL.md` (experiment E-23). It validates the Test A
harness on the MacBook before any GPU time is spent.

**This is a harness validation, not a research result.** Its numbers say nothing about GPU
kernels. The pass criteria were written into the journal before anything ran (E-23).

## What it checks

| Criterion | What | Script |
|---|---|---|
| C1 | A backend compared with itself across repeated runs is bitwise identical (no false positives) | `run_model_pilot.py` |
| C2 | Fault injection: an emulated INT8 GEMM epilogue with four faults. The bitwise gate finds all of them. A one-spacing `allclose` gate misses three, which shows the blind spot reported in arXiv 2609.00363 | `run_fault_injection.py` |
| C3 | For every prompt, the first teacher-forced decode-mode flip equals the first free-running divergence. This shows the flip data is a sound stand-in for real generation | `run_model_pilot.py` |
| C4 | A margin-stratified estimator of the greedy flip rate has an interval that contains the exhaustively measured rate at least 90% of the time over 500 replicate subsamples | `run_model_pilot.py` |
| C5 | Reported only: KL, max logit difference, prefill versus decode flip agreement | `run_model_pilot.py` |

Contract levels follow the journal, Section 5.8: L1 bitwise and spacing distance, L2 greedy
flips with margins, L3 KL on logits.

## Layout

```
harness/contracts.py     L1, L2, L3 metrics as pure tensor functions
harness/int8_gemm.py     emulated INT8 GEMM and the fault-injected epilogues F1 to F4
harness/estimator.py     margin-stratified flip-rate estimator and coverage simulation
harness/modelrun.py      decode-mode and prefill-mode logits, free running and teacher forced
harness/manifest.py      run manifest (versions, hardware, SHA-256 of sources and prompts)
make_prompts.py          freezes prompts.json from stable repo documents
run_fault_injection.py   C2
run_model_pilot.py       C1, C3, C4, C5
tests/test_harness.py    unit tests
results/attempt_NN/      one directory per attempt, failed attempts are kept
```

## Run

Uses the existing environment (torch 2.14.0, transformers 5.17.0). See journal Section 9.4.

```
cd experiments/safedispatch_oracle
PY=~/Downloads/mac_hybrid_selective_recovery_poc/.venv/bin/python
$PY -m unittest discover -s tests -v
$PY make_prompts.py
$PY run_fault_injection.py --out results/attempt_01
$PY run_model_pilot.py --n-prompts 32 --steps 96 --candidate mps_fp16 --out results/attempt_01
# fallback if the candidate has too few flips (predeclared):
$PY run_model_pilot.py --candidate cpu_bf16 --out results/attempt_02
```

Model: `Qwen/Qwen3-0.6B` (dense, same family as arXiv 2608.13756), downloaded from the Hugging
Face Hub on first use.

## Design notes

- Batch size 1 and exact greedy decoding. Logits are taken on CPU float32, and argmax is
  taken there too, so ties break the same way in free running and in the flip analysis.
- Teacher forcing is done in decode mode (one token at a time with a KV cache) so it uses the
  same kernels as free running. Prefill-mode teacher forcing is also measured, because prefill
  and decode select different kernels.
- The INT8 faults are my own constructions in the class reported by arXiv 2608.13756 and
  2609.00363 (epilogue scale application and output rounding). They are not a reproduction of
  that paper's five faults.
- The estimator uses Jeffreys-prior Beta draws per margin stratum, so a stratum with zero
  observed flips still carries uncertainty.

## Porting to a GPU

Replace the two backends in `run_model_pilot.py` (`CANDIDATES`) with the kernel pairs under
test, and add a kernel-pair abstraction in `harness/`. The contract functions and the
estimator do not depend on the backend.
