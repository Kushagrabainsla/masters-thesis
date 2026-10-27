# E-40: I-23 NVFP4 backend choice on one B200 (preregistration)

Status: PREDECLARED 2026-10-27, AMENDED (A1 after R-47). RUN 2026-10-27: INCONCLUSIVE. Idea: I-23 (the last open cell after D-63). User request
2026-10-27: "Prepare the I-23 NVFP4 test on one B200 (backend speed plus GSM8K accuracy, paired rounds),
preregistered and reviewed on the Mac."
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics.

## 1. Question
On one B200, is there an NVFP4 linear or MoE backend in vLLM v0.32.0 that keeps GSM8K accuracy within
2 pp of the default backend R and is at least 3% faster than R at one or more deciding batch sizes?
"No" means no backend is both.

## 2. Why (rubber duck)
- FACT (vLLM v0.32.0 source): NVFP4 has many backends. MoE: B12X, FLASHINFER_TRTLLM, FLASHINFER_CUTLASS,
  FLASHINFER_CUTEDSL, VLLM_CUTLASS, MARLIN, HUMMING and others. Linear: cutlass, flashinfer variants,
  b12x, marlin, humming. Hopper FP8 had 3 to 5 working backends.
- FACT (E-36, E-38, E-39 on H100): the default was the fastest FP8 backend; alternatives kept accuracy.
  INTERPRETATION: Blackwell FP4 kernels are newer, so the default choice is less tested, and a faster
  backend is more likely here. This is the premise to test.
- Contract: task accuracy only (E-38 A1); E-38's A2 was retired post hoc (E-39 amendment A1). A GO is
  PIVOT-grade for I-23, as in E-39.
- Prior (HYPOTHESIS): one in three that a backend is 3% or more faster at some batch size.

## 3. Setup (frozen)
- Hardware: 1 x B200 (sm_100), Secure, on-demand, network volume. `vllm/vllm-openai:v0.32.0` with the
  E-36 and E-38 fixes (CUDA bin on PATH, caches under the vLLM cache root, in-process engine, prefix
  caching off, one warm-up per config).
- Models: cell D `nvidia/Qwen3-8B-NVFP4` (`--linear-backend`), cell M `nvidia/Qwen3-30B-A3B-NVFP4`
  (`--moe-backend`). Revisions recorded.
- Candidates. Cell D: cutlass, flashinfer_cutlass, flashinfer_trtllm, flashinfer_cudnn,
  flashinfer_cutedsl, b12x, marlin, humming. Cell M: flashinfer_trtllm, flashinfer_cutlass,
  flashinfer_cutedsl, cutlass, b12x, marlin, humming. A backend that fails to start is recorded and
  dropped. A backend whose log selects R's kernel is an alias and is dropped from the screen.
- Speed: E-36 `run_cell.py --mode throughput`, end-to-end `generate()` tokens/s for 128 new tokens at
  batch 1, 2, 4, 8, 16, 32, 64, 128, 256, 5 repeats per process, `max_num_seqs` 256, GSM8K prompts.
- Accuracy: E-38 `run_a2.py --mode full --no-score`, GSM8K test (1,319), greedy, 768 tokens.

## 4. Phases and order
1. Screen (speed only): R twice (R_1, R_2), then one process per candidate. Ratio per batch size =
   candidate / mean(R_1, R_2). A candidate passes the screen if its ratio is at least 1.02 at one or
   more deciding batch sizes (Section 5) and it is not an alias. Keep at most 2 per cell, ranked by the
   best ratio at a deciding size.
2. Accuracy: R_a once, then each screened candidate once (warm-up first). A1 as in E-38 (paired
   bootstrap, lower bound of the difference above -2.0 pp).
3. Paired speed for each screened candidate that passes A1: 8 rounds (R, C in odd rounds and C, R in
   even rounds) after one warm-up each. The decisive data for the MoE cell comes first.

## 5. Rule (frozen)
Deciding batch sizes: 1, 8, 32, 128 (log-spaced). Per round: log(median C / median R). One-sided t bounds
at alpha 0.05 / 4 (df 7, t 2.841).
- GO: a candidate passes A1 and has a lower bound of at least 1.03 at a deciding batch size.
- KILL: no candidate passes the screen; or every screened candidate fails A1; or every screened
  candidate that passes A1 has upper bounds below 1.03 at all deciding sizes.
- Otherwise INCONCLUSIVE, with the reason named.
- Validity: V1: R_1 and R_2 differ by at most 5% at each deciding size (screen noise). V2: every
  measured paired run is ok and loads the AOT compile; at least 6 complete rounds. V3: at least 100
  telemetry rows parse and at most 2% of busy samples show a hard slowdown. A failed V gives INVALID
  for that cell.
- Power (INTERPRETATION): with a 3% per-round sd, the half-width is about 3%, so GO needs a true gain of
  about 6% or more. The screen at 1.02 accepts some noise so that a real gain is rarely missed.

## 6. Budget
B200 at the live price (about $7.99/h). Setup with image and model downloads up to 25 min; screen
2 cells x (2 + up to 8 candidates) processes at about 2.5 min (up to 50 min); accuracy up to 6
processes at about 4 min (up to 25 min); paired speed up to 4 candidate-cells x 17 processes at about
2.5 min. With the time cap the session stops new work at 2 h 45 min; watchdog at 3 h; cost cap $25.
The paired phase is cut first, in the order MoE before dense.

## 7. Threats
One GPU type, two models, one task, end-to-end tokens/s (includes prefill), a screen of one process per
candidate (a real gain can be missed when the screen noise is high: V1 reports it).

## 8. Files
`session_e40.sh`, `analyze_e40.py` (screen selection and the rule), `test_e40.py`. Reused:
`../safedispatch_testA/run_cell.py`, `../safedispatch_testA2/run_a2.py` and `gsm8k.py`,
`../marlin_moe_speed/analyze_e39.py` (paired bounds), `../runpod/`.

## Amendment A1 (2026-10-27, after R-47, before any run; wins over Sections 3 to 6)
R-47 (Sonnet pre-run review): AMEND_THEN_RUN. Changes:
1. Alias check (FACT, vLLM v0.32.0 `oracle/nvfp4.py`): the NVFP4 MoE line is "Using 'X' NvFp4 MoE
   backend out of potential backends: [...]", which the E-36 regex did not match. `analyze_e40.py` has
   its own regex for both formats, and in cell M it compares only the MoE line (the linear lines are the
   same for every MoE candidate). V0: R_1 must log a selection line, or the cell is INVALID.
2. Budget: keep the best 1 candidate per cell (not 2); 6 paired rounds (df 5, t 3.163); batch sizes
   1, 4, 8, 32, 128, 256 with 3 repeats (deciding 1, 8, 32, 128). Order: both screens first, then cell M
   accuracy and paired speed, then cell D. Estimate about 2 h 30 min with setup; cap 2 h 45 min new
   work, watchdog 3 h, cost cap $25.
3. R_2 runs in the middle of the screen. Run timeout 900 s. FlashInfer JIT builds are kept on the volume.
4. A screen with no pass gives KILL only if the R noise is at most 2%; otherwise INCONCLUSIVE.
5. The AOT load is reported, not a validity gate (custom-op backends may not use the AOT cache).
6. Power (INTERPRETATION): with 6 rounds, GO needs a true gain of about 7% or more; a gain between 0%
   and 7% gives INCONCLUSIVE. The Bonferroni correction covers the 4 sizes, not the 2 candidate-cells.
7. V3 is session-wide (screen and accuracy samples included). Analysis exit codes are logged.

## Result
Result (2026-10-27, Pod 50uqmcyz64wqvu, 1 x B200 sm_100, US-NE-1, about $9; 118 checksums match): frozen
decision INCONCLUSIVE (cell D invalid: V0, V1; cell M: no screen pass, but R noise at batch 128 was 4.3%).
FACT, cell M: R = FLASHINFER_TRTLLM; the alias was caught; every real backend is slower at every size
(cutedsl 0.84 to 0.95, flashinfer_cutlass 0.65 to 0.76, marlin 0.42 to 0.70, humming 0.55 to 0.70,
cutlass 0.51 to 0.61); b12x failed to start.
FACT, cell D: the linear selection line reads "Using FlashInferCuteDslNvFp4LinearKernel for NVFP4 GEMM",
which the E-40 regex did not match (V0). R was CuTe-DSL, so the selected flashinfer_cutedsl was an
undetected alias: accuracy identical (+0.00 pp [0.00, 0.00]) and paired ratios about 1 (batch 8 1.016
[1.005, 1.028]). At the deciding sizes every real backend is slower (0.27 to 0.94). R noise at batch 128
was 7% (V1).
Open lead (reported size, one process each): at batch 256, flashinfer_cudnn 1.07, flashinfer_cutlass 1.05
and cutlass 1.04 against R. A large-batch crossover is possible but not measured.
INTERPRETATION: as on Hopper, the default NVFP4 backend is the fastest at batch 1 to 128 in both cells.
The only possible headroom is at batch 256 in the dense cell. R-48 (post-run review): CONFIRM_WITH_SCOPE_LIMITS; the
batch-256 lead is inside noise (alias paired sd about 3.4% at batch 256).
