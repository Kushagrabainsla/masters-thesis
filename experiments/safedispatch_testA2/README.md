# E-38: I-23 Test A2, a tolerance contract for Hopper block-FP8 backends (preregistration)

Status: PREDECLARED 2026-10-26, AMENDED (A1 after R-43). NOT RUN. Stage S3 of `docs/experiments/PROTOCOL.md`.
Idea: I-23. Decision D-61 (user): test "good enough" instead of "identical".
Labels: FACT, INTERPRETATION, HYPOTHESIS. Thresholds are project heuristics unless a source is given.

## 1. Question
On one H100, is there a Hopper block-FP8 backend (or the W8A16 Marlin path) that keeps task quality
within a tolerance of the default backend R and is also faster than R at some batch size? "No" means:
every backend that is faster than R loses quality, or no backend is faster than R.

## 2. Rubber duck
- What I-23 claims: a dispatch layer that knows speed and output safety can beat any fixed backend.
- What E-36 Session 1 showed (FACT): under exact match, every alternative changes greedy output on
  87.5% to 95.3% of prompts, and R is the fastest FP8 backend at batch 1 to 64 (dense also at 1 to 256).
  R-42: exact match flags any numerical change (eager mode alone flips 91% to 93%), so it cannot measure
  headroom. Only the Marlin W8A16 control was faster than R, in the MoE cell at batch 4 to 32, by 1% to
  5%, which is near the R noise (3.5%).
- What would kill I-23 early (for Hopper block-FP8): no backend that passes the tolerance contract is
  faster than R.
- Honest prior (HYPOTHESIS, written before the run): the dense cell ends KILL (no faster FP8 backend).
  The only GO path is MoE Marlin at batch 4 to 32, if it passes the contract and the speed gain holds
  under an A/B design. The most likely overall outcome is KILL.
- Why now: it is cheap (about 1.5 GPU-hours) and it closes or opens the strongest live idea.

## 3. Hypotheses
- H1: at least one candidate passes the tolerance contract and is BETTER than R (Section 5) at one or
  more batch sizes in a cell.
- H0: no candidate is both acceptable and BETTER.

## 4. Setup (frozen)
- Hardware: 1 x H100 SXM 80 GB (sm_90), Secure Cloud, on-demand, network volume.
- Software: `vllm/vllm-openai:v0.32.0` (digest sha256:befccf0e7e40... as in E-36), the E-36 environment
  fixes (CUDA bin on PATH, caches under the vLLM cache root, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, prefix
  caching off, A3 warm-up per config).
- Models: cell D `Qwen/Qwen3-8B-FP8`, anchor `Qwen/Qwen3-8B` (BF16). Cell M `Qwen/Qwen3-30B-A3B-FP8`,
  anchor `Qwen/Qwen3-30B-A3B` (BF16, about 61 GB; if it does not load, the MoE anchor is reported as
  missing and the MoE KL gate uses the dense anchor's ratio, see Section 6).
- Data: GSM8K `main` test split, all 1,319 problems, at a recorded dataset revision. Prompt: the chat
  template with thinking off, the question, then "Please reason step by step, and put your final answer
  within \boxed{}." File hash in the manifest.
- Decoding: greedy, 768 new tokens, `max_num_seqs` 256, seed 0.
- Configs per cell: R (auto) twice (R_a, R_b), each candidate once, the BF16 anchor once.
  Cell D candidates: `deep_gemm`, `cutlass`, `triton`, `marlin`. Cell M: `deep_gemm`,
  `flashinfer_cutlass`, `marlin`, `triton` (expected alias: negative control).

## 5. Metrics
- Primary: GSM8K accuracy (exact numeric match of the last `\boxed{}` value with the gold answer after
  `####`). Paired difference against R_a on the same 1,319 problems, with a paired bootstrap 95%
  interval (10,000 resamples, seed 0).
- Secondary (teacher-forced): each config scores R_a's completions as prompt continuations
  (`prompt_logprobs` 20, one new token). On the completion positions: the mean approximate KL from R_a
  (top-20, renormalized, as in E-36 `compare.py`), and the top-1 disagreement rate (the fraction of
  positions where the config's argmax is not R_a's argmax). R_a scores itself the same way.
- Speed (only for acceptable candidates): decode tokens/s at batch 1, 4, 8, 16, 32, 64, 128, 256, five
  repeats, in the order R, C, R, C (two processes each) in the same Pod. `compare_runs.py` with the first
  R run as baseline, the second as A/A, `--mde 0.03`, on the median of the two C runs against the
  median of the two R runs.

## 6. Validity checks (any failure: INVALID for that cell)
- V1: R_a and R_b give identical GSM8K completions (exact tokens) on at least 99% of problems.
- V2: R_a GSM8K accuracy is at least 80% (sanity; Qwen3-8B without thinking is expected near 90%).
- V3: teacher-forced, R_b against R_a has KL below 1e-4 and top-1 disagreement below 0.1%.
- V4 (negative control, cell M): the `triton` alias has top-1 disagreement below 0.1%.
- V5 (positive control): the BF16 anchor has a teacher-forced KL above that of R_b (the measure can see
  a real numerical change).

## 7. Decision rule (frozen)
A candidate C is ACCEPTABLE in a cell if both hold:
- A1: the lower bound of the paired 95% interval of (accuracy(C) minus accuracy(R_a)) is above -2.0
  percentage points (non-inferiority margin 2.0 pp);
- A2: the teacher-forced KL(C) is at most the teacher-forced KL of the BF16 anchor (the deviation is
  not larger than the FP8 quantization error itself). If the MoE anchor is missing, use
  KL(C) at most KL_D(BF16) x KL_M(R_b-floor) / KL_D(R_b-floor), reported as a fallback.
A candidate C has HEADROOM if it is ACCEPTABLE and `compare_runs.py` says BETTER at one or more batch
sizes in Section 5.
- GO: at least one candidate has HEADROOM in at least one cell. Then the full Test A is designed with
  this tolerance contract.
- KILL: no candidate has HEADROOM in either cell, and V1 to V5 pass. Then I-23 is closed for Hopper
  block-FP8 at these configurations; NVFP4 on Blackwell stays open.
- PIVOT: a candidate is BETTER but not ACCEPTABLE (speed costs quality), and no candidate has HEADROOM.
  Report the speed and quality trade-off; no dispatch claim.
- Otherwise INCONCLUSIVE, with the failed condition named.

## 8. Controls and baselines
Baseline R_a (auto). A/A: R_b. Negative control: MoE `triton` alias. Positive control: the BF16 anchor
(V5). Marlin is W8A16 (a different arithmetic contract): it may be ACCEPTABLE; that is a valid result.

## 9. Budget
1 x H100 SXM, $3.99/h (live read before booking). Order (the decisive data first):
1. Setup and GSM8K build: 5 min.
2. Cell D: per config one warm-up, then the GSM8K accuracy run, then the teacher-forced score of R_a's
   completions in the same process (7 configs: R_a, R_b, 4 candidates, BF16): about 25 to 35 min.
3. Cell M: the same (7 configs): about 30 to 45 min.
4. Analysis of A1 and A2; then the speed A/B (R, C, R, C) only for ACCEPTABLE candidates: about 6 min per
   candidate and cell.
Time cap: no new run after 2 h 30 min; Pod limit 3 h; cost cap $12. Speed is cut first.

## 10. Threats to validity
- Teacher forcing scores in the prefill path, not the decode path; decode kernels can differ. Reported.
- GSM8K is one task (math reasoning, English); a 2 pp margin on 1,319 problems has limited power for
  small losses. The paired interval is reported in full.
- BF16 anchor as the tolerance bar is a convention (D-61 asked for "good enough"); it is stated, not
  claimed as the only choice.
- Speed noise: E-36 R noise reached 3.5%; the A/B order and the A/A band address it.
- One GPU type, two models.

## 11. Files
- `gsm8k.py`: builds the 1,319 prompts with gold answers; `extract()` parses `\boxed{}`. Tested.
- `run_a2.py`: one config: GSM8K accuracy run, then teacher-forced scoring of a reference file.
- `analyze_a2.py`: V1 to V5, A1, A2, HEADROOM and the frozen decision. Tested.
- `session_a2.sh`: the session (warm-ups, resume, telemetry, AUTO_STOP), reusing
  `../safedispatch_testA/run_cell.py` for warm-ups and speed and `../runpod/` for records.

## 12. Verification record
(empty until stage S3)

## Amendments

### A1 (2026-10-26, after R-43, before any run; this section wins over Sections 5 to 9)
Freeze statement: the contract below was written on 2026-10-26 after E-36 Session 1 (whose flip and
speed data the authors had seen) and before any E-38 data. R-43 (Sonnet pre-run review,
AMEND_THEN_RUN, `dossier/REVIEW_E-38_PRERUN_2026-10-26.md`) found the issues fixed here.
1. A2 is tighter. With the same FP8 weights, a candidate's deviation holds only kernel and activation
   noise, while the BF16 anchor holds the full weight-quantization error, so "at most the anchor" would
   pass almost every candidate. New A2: teacher-forced KL(C) at most 0.5 x KL(BF16) AND top-1
   disagreement(C) at most 0.5 x that of BF16 (heuristic: at most half of the FP8 quantization effect).
   A1 (accuracy non-inferiority, margin 2 pp) is unchanged and does most of the work.
2. Decode path (reported, not in the rule): each config also records top-20 logprobs during its own
   greedy generation, and `decode_prefix.json` compares them with R_a's on the positions up to and
   including the first token divergence (same prefix there). Teacher forcing stays in the prefill path.
3. Reported context: the median and p99 of the per-position KL, the BF16 accuracy difference, and
   A1 for R_b against R_a (a null check: it should be 0).
4. Order and scope (decisive data first): cell M accuracy (all configs), then the speed A/B for
   cell M Marlin, then cell D accuracy. No other speed tests: E-36 Session 1 (Phase A and B2) showed
   that every other FP8 candidate is slower than R at every batch size, so none can have HEADROOM and
   none can give PIVOT; that judgement comes from E-36 data, and the report says so. Speed runs also
   check the AOT load.
5. Speed rule restated as the code computes it: baseline = both R runs (10 repeats), candidate = both C
   runs, band = max(3%, |median(R_2) / median(R_1) - 1|), Bonferroni over the 8 batch sizes only (not
   over candidates and cells). The 5 repeats in one process are pseudo-replicates; the real replication
   is 2 against 2 processes. KILL therefore means "no gain of 3% or more that this design can resolve",
   not "no gain". The 3% bar: a dispatch layer adds complexity, so a smaller gain has little product
   value.
6. Decision handling: a cell that fails V1 to V5 is INVALID for that cell only. GO needs HEADROOM in a
   valid cell. KILL needs both cells valid, no errored candidate, the BF16 anchors present, and the
   planned speed test complete. Otherwise INCONCLUSIVE with the reasons. The missing-anchor fallback of
   Section 7 is removed (a missing anchor gives INCONCLUSIVE). If V4 fails, the report says "the MoE
   `triton` alias claim is false" instead of treating it as a backend finding.
7. Robustness: teacher-forced scoring runs in chunks of 256 prompts and fails the run if any completion
   position is not scored or the position counts differ. The MoE BF16 anchor runs with `max_num_seqs`
   64 (about 61 GB of weights on 80 GB), recorded in its run.json.
8. Budget, re-estimated: the four models are about 117 GB (volume 200 GB); setup with downloads about
   15 to 20 min; cell M accuracy about 50 to 60 min; the Marlin speed A/B about 20 to 25 min; cell D
   accuracy about 40 to 50 min. No new run after 3 h, Pod limit 3 h 30 min, cost cap $15 (at $3.99/h).
