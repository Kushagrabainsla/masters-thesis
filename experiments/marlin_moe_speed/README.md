# E-39: Marlin W8A16 MoE small-batch speed on H100 (preregistration)

Status: PREDECLARED 2026-10-27, AMENDED (A1 after R-45). NOT RUN. Idea: I-23. Follows E-36 B2 and E-38 (R-44 open lead).
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
On one H100 SXM, does `--moe-backend marlin` decode Qwen3-30B-A3B-FP8 at least 3% faster than the
default backend R at one or more batch sizes from 1 to 32, with process-level replication? "No" means no
batch size shows a resolvable gain of 3% or more.

## 2. Why (rubber duck)
- FACT: E-36 B2 showed Marlin MoE 1% to 5% faster than R at batch 4 to 32; E-38 showed +3.5% and +7% at
  batch 8 in two processes, UNCLEAR with 2 against 2 processes (R-44), and -10.7% at batch 256.
- FACT: accuracy is already measured. E-38 rerun, same image, same model, H100: Marlin GSM8K 95.07%
  against R 95.07%, difference +0.00 pp [-0.76, +0.76] on 1,319 problems. This test does not measure
  accuracy again.
- Contract (new, written before any E-39 data): task accuracy (E-38 A1) is the quality contract here.
  E-38's A2 (logprob closeness at 0.5 x BF16) is not used: in E-38 every real backend failed it,
  including other FP8 backends of R's own kind, so it does not separate safe from unsafe backends.
  This is a new test; E-38 stays KILL under its own rule.
- If GO: the fastest accuracy-preserving backend changes with batch size (Marlin small, R at 256), which
  is the I-23 premise under an accuracy contract.
- Prior (HYPOTHESIS): about even odds; the effect is near the noise.

## 3. Setup (frozen)
1 x H100 SXM 80 GB, Secure, on-demand, network volume. `vllm/vllm-openai:v0.32.0` with the E-36 and E-38
environment fixes (CUDA bin on PATH, caches under the vLLM cache root, in-process engine, prefix caching
off). Model `Qwen/Qwen3-30B-A3B-FP8`. Prompts: the E-38 GSM8K prompt set (`gsm8k.py`). Speed with
E-36 `run_cell.py --mode throughput`: decode tokens/s at batch 1, 2, 4, 8, 16, 32, 64; 128 new tokens;
5 repeats per process; `max_num_seqs` 64.

## 4. Order
Warm-up R, warm-up C (one process each, A3). Then 6 rounds, each R then C, alternating the order in
odd rounds (R, C) and even rounds (C, R) to cancel drift: 12 measured processes.

## 5. Metric and rule (frozen)
Unit: one process. Per process and batch size: the median of its 5 repeats. Per batch size: ratio =
median over the 6 C processes / median over the 6 R processes. Interval: bootstrap over processes
(resample 6 R and 6 C process medians with replacement, 20,000 draws, seed 0), at level 1 - 0.05 / 7
(Bonferroni over the 7 batch sizes).
- Validity: V1: the R process medians have a coefficient of variation at most 3% at every batch size.
  V2: every measured run loads the AOT compile and ends ok. V3: GPU telemetry shows no hard slowdown.
- GO: at one or more batch sizes from 1 to 32, the lower interval bound is at least 1.03.
- KILL: at every batch size from 1 to 32 the upper bound is below 1.03.
- Otherwise INCONCLUSIVE (name the batch sizes).
- If V1 to V3 fail: INVALID.

## 6. Budget
Image pull up to 15 min, model download about 3 min, 14 processes at about 2.5 min: about 55 min,
cap 1 h 30 min Pod time, cost cap $6.

## 7. Threats
One model, one GPU type, one task for accuracy (from E-38), decode only, synthetic batches of GSM8K
prompts. A GO is a speed fact for this setup, not a dispatch system.

## 8. Files
`session_e39.sh` (the session), `analyze_e39.py` (the rule), `test_e39.py`.

## Amendment A1 (2026-10-27, after R-45, before any run; wins over Sections 4 to 6)
R-45 (Sonnet pre-run review): AMEND_THEN_RUN.
1. Framing: E-38's A2 is retired post hoc (2026-10-27), with this reason: every real backend failed
   it in E-38, so it did not separate backends. Marlin failed A2 in E-38 (KL 0.62 and top-1 0.74 of the
   BF16 gap). A GO here is a speed gain under an A1-only (task accuracy) contract: PIVOT-grade evidence
   for I-23, not a full confirmation. E-38 stays KILL.
2. Design: 10 paired rounds (order R, C in odd rounds and C, R in even rounds). Per round and batch size,
   log(median C / median R). One-sided t bounds at alpha 0.05 / 4 (Bonferroni over the 4 deciding
   sizes: 4, 8, 16, 32). GO: a lower bound at least 1.03. KILL: every upper bound below 1.03. Otherwise
   INCONCLUSIVE. Power (INTERPRETATION): with a 3% per-round sd, the half-width is about 2.5%, so GO
   resolves a true gain of about 5.5% or more. Batch 1, 2, 64, 128 and 256 are reported only (128 and
   256 show whether R is faster at large batch, the crossover).
3. Metric: end-to-end `generate()` tokens/s for 128 new tokens (it includes prefill), not pure decode.
4. Validity: V1 (R noise) is reported only. V2: at least 6 complete rounds and every run ok with an AOT
   load. V3: at least 100 telemetry rows parse and at most 2% of busy samples show a hard slowdown. A
   drift slope of the log ratio against the round is reported.
5. Budget: about 90 min with setup (up to 25 min pull and download) and 20 processes; no new round after
   95 min of session time; watchdog at 105 min; cost cap $8.

## Result (2026-10-27)
Result (2026-10-27, Pod cj79arezidx9ax, 1 x H100 SXM, US-MO-1, about $5.20; 10 complete rounds, V2 and V3
pass, no hard slowdown, 74 checksums match): frozen decision INCONCLUSIVE (4, 8, 16). Marlin / R ratio
(one-sided bounds at alpha 0.0125): batch 4 1.015 [0.983, 1.047]; batch 8 1.025 [1.002, 1.049]; batch 16
1.011 [0.992, 1.031]; batch 32 0.995 [0.976, 1.015]. Reported: batch 1 0.949, 2 0.959, 64 0.959, 128 0.976,
256 0.887 (all below 1). INTERPRETATION: Marlin MoE is about 1% to 2.5% faster at batch 4 to 16 (batch 8 is
above 1 at this level) and slower everywhere else; the gain is below the 3% bar and its upper bound is
under 5%. A batch-dependent dispatch would gain at most a few percent on a narrow band. Post-run review
pending.
