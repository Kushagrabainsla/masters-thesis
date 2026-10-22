# E-36: I-23 Test A, Session 1 (one H100, calibration and first open cells)

Status: PREDECLARED 2026-10-20, AMENDED (A1, Section 9) after the R-34 Sonnet pre-run review
(AMEND_THEN_RUN). Not run. Needs one rented H100 and the user's approval (D-51). Sections 3 to 6 are the
original design; where Section 9 differs, Section 9 wins. The Mac dry run (vLLM CPU backend) passed
mechanically on 2026-10-20 (Section 10). Section 10 also records one change to the mechanism of
Section 9 point 2.
Amendment A2 (Section 11, 2026-10-22) adds Runpod efficiency changes and an exploratory Phase B. Phase A
and its rules do not change.

Labels: FACT, INTERPRETATION, HYPOTHESIS. All thresholds are project heuristics.

## 1. Why a calibration session first

The full Test A (journal Section 5.8) needs 1 to 2 weeks on one GPU. The first self-funded round (D-35)
is about $20, which buys about 4 hours on one H100 (Lambda 1 x H100 SXM, about $4.29 per hour,
SNIPPET from third-party price lists; check lambda.ai before booking). Session 1 therefore has three
jobs:
1. Remove setup risk: pinned image, backend inventory, harness on real Hopper kernels.
2. Measure the two cheapest open cells (block-scaled FP8 dense linear, block-scaled FP8 MoE) at a
   fixed batch configuration.
3. Measure time per cell, to size the full Test A for the grant request.

It also has a predeclared early-stop rule, so the session can end the I-23 line on Hopper FP8 if the
cells are clearly safe and flat.

Relation to the full Test A: the full design is in `dossier/FRONTIER_R-33_2026-10-20.md` Section 7
(R-33, a separate session on 2026-10-20) and journal Section 5.8. Session 1 covers its steps 1 and 2
(reproduce a known divergence, enumerate the backend matrix) and a first, fixed-batch cut of steps 4
and 5 (each candidate against one reference; sequence and speed results). It does not replace the full
Test A (held-out prompts, batch and length buckets, eager and CUDA-graph modes, routing skew, the six
baselines, and the GO/PIVOT/KILL rules there).

## 2. Rubber duck

**What I-23 claims.** Serving engines expose several kernels for the same operator. The fastest one
that keeps the model's output (an explicit contract) can differ by serving regime, so a dispatch layer
that knows both speed and safety beats any single fixed backend.

**What is new since 2026-09-23 (FACT).**
- vLLM v0.32.0 (2026-10-18) makes backend choice a first-class engine option: `--linear-backend`
  (`cutlass`, `deep_gemm`, `triton`, `flashinfer_cutlass`, `marlin`, and more),
  `--moe-backend` (`triton`, `cutlass`, `deep_gemm`, `flashinfer_cutlass`, `flashinfer_trtllm`, and
  more), `linear_backend_per_quant`, and `--attention-backend` (read in `vllm/config/kernel.py` and
  `vllm/engine/arg_utils.py`). So each backend pair is one flag, and a dispatch layer has a clean
  hook.
- vLLM's training-inference consistency stack (blog, 2026-10-14) chooses fixed kernels for one
  reduction order, and FlashInfer v0.7 autotunes by latency (R-21). Both raise the novelty bar: the
  contribution must be the output-aware choice, not the switch.

**What would make I-23 die early.** If every alternative backend gives the same greedy output as the
default, and their speeds are within noise, there is nothing to dispatch. If alternatives often change
the output but one backend is always fastest, a static rule wins.

**Why this order of cells.** Hopper supports block-scaled FP8 (DeepSeek-style 128 x 128 scales) for
dense and MoE layers, with several vLLM backends each. NVFP4 needs Blackwell and waits for a later
session. Integer Alibi (the closest paper) left FP8 end-to-end and MoE open.

## 3. Setup (frozen)

- Hardware: 1 x H100 SXM 80 GB (Hopper, sm_90). Record `nvidia-smi -q`, driver and CUDA versions.
- Software: the official `vllm/vllm-openai:v0.32.0` image. Record its digest.
- Models (Hugging Face, record the revision hash):
  - Cell D (dense): `Qwen/Qwen3-8B-FP8` (block-wise FP8). Control: `Qwen/Qwen3-8B` (BF16).
  - Cell M (MoE): `Qwen/Qwen3-30B-A3B-FP8` (block-wise FP8).
- Prompts: `prompts.py --source hub` builds 256 prompts: 128 from `openai/gsm8k` (test split) and
  128 first user turns from `HuggingFaceH4/ultrachat_200k` (`test_sft`), selected by seed 20261020,
  chat template applied. The file and its SHA-256 go into the manifest.
- Decoding: greedy (temperature 0), 128 new tokens, `logprobs=20`, seed 0, one `generate()` call with
  all 256 prompts, `max_num_seqs=64`. The batch configuration stays the same for every backend.
- Backends: `inventory.py` tries every listed value of `--linear-backend` (Cell D) and `--moe-backend`
  (Cell M) on a 4-prompt smoke run and keeps the ones that start. The reference backend R is the one
  that `auto` selects (from the engine log). Control: BF16 with `--attention-backend FLASH_ATTN` and
  `FLASHINFER`.
- Throughput: per kept backend, output tokens per second at batch sizes 1, 8, 32 and 64 (128 new
  tokens; 64 is the `max_num_seqs` of the flip runs), and prefill time for 64 prompts of about 1,000
  tokens (1 new token). Three repeats; report the median.

## 4. Metrics

- Flip: a prompt is flipped for candidate C if C's greedy tokens differ from R's anywhere in the 128
  tokens. Report the flip fraction with a Wilson 95% interval (E-23: simple random sampling, not the
  failed stratified estimator).
- Floor: R runs twice. Prompts where R differs from itself are excluded from flip counts and reported.
- First-divergence index, and R's top-2 logprob margin at that index.
- Approximate KL on the first 32 positions from the top-20 logprobs (renormalized), reported only.
- Speed ratio of each candidate to R at each batch size.

## 5. Frozen rules for Session 1

Validity:
- V1: R's two runs agree on at least 95% of prompts. If not, the floor is too high for a flip test;
  report it and stop the flip analysis (the speed data still counts).
- V2: at least two backends start in each FP8 cell. If a cell has fewer, report it as "no choice to
  make" for that cell.

Early stop for Hopper FP8 (recommendation to the user, not an automatic kill):
- STOP-FLAT: in both FP8 cells, every kept candidate has 0 flipped prompts (Wilson upper bound at most
  1.5%), and at every batch size the fastest and slowest kept backends differ by under 2%. Then the
  Test A KILL condition (gain under 2%, unsafe under 2%) holds for Hopper FP8 at this configuration.
  Recommend closing I-23 for Hopper FP8; NVFP4 on Blackwell stays the only open cell.
- CONTINUE: in at least one FP8 cell, a candidate flips at least 10% of prompts, and the fastest
  backend differs between at least two batch sizes, or is an unsafe backend at some batch size. Then the
  premise of the full Test A holds; use the timing data to plan it.
- Otherwise: INCONCLUSIVE. Write down which condition failed and plan Session 2 around it.

Stop rules for money: a hard cap of 4 hours or $18, whichever comes first. `session1.sh` stops
starting new configurations after 3 hours 20 minutes and always saves results first.

## 6. Order of work in the session (time budget)

| Step | Work | Budget |
|---|---|---|
| 0 | Pull image, download models and datasets, record manifest | 25 min |
| 1 | Inventory (Cell D, Cell M, BF16 control) | 25 min |
| 2 | Cell D: R twice, then each candidate (flips) | 35 min |
| 3 | Cell M: R twice, then each candidate (flips) | 45 min |
| 4 | Throughput for every kept backend | 50 min |
| 5 | BF16 control (two attention backends) | 15 min |
| 6 | `compare.py`, copy results off the machine, terminate | 15 min |

Total about 3 hours 30 minutes; the margin covers slow downloads.

## 7. Threats to validity

- One batch configuration: flips can depend on batch size (batch invariance). Session 1 fixes it; the
  full Test A varies it.
- 256 prompts, 128 tokens: rare flips (under about 1.5%) are not resolvable.
- Top-20 KL is an approximation.
- The default backend R may itself be a poor reference; the BF16 model is the semantic control, not a
  bitwise reference.
- One model per cell, one GPU type.

## 8. Files

- `prompts.py`: builds the prompt set (`--source hub` on the GPU box, `--source local` for the dry run).
- `inventory.py`: tries each backend value and records start or failure.
- `run_cell.py`: one configuration, greedy outputs with logprobs, plus the throughput sweep.
- `compare.py`: flips, floor, Wilson intervals, KL, speed ratios, and the frozen Session 1 rule.
- `session1.sh`: the whole session with the time cap.
- `test_compare.py`: unit tests for `compare.py` on synthetic data.
- `extras.py` (A2): telemetry, kernel-name and Phase B summaries, reported only. `test_extras.py`: its tests.
- `dryrun/`: the Mac dry run (vLLM CPU backend); mechanics only. `dryrun.sh` runs it;
  `cpu_triton_utils_v0.32.0_pr61992.py` is the CPU-only patch it mounts; `results/` holds the last run.

## 9. Amendment A1 (2026-10-20, after R-34, before any run)

R-34 (Sonnet, `dossier/REVIEW_E-36_E-30_PRERUN_2026-10-20.md`) returned AMEND_THEN_RUN. The main
session confirmed the key code claims in the v0.32.0 source: an unsupported `--linear-backend` falls
back to normal kernel selection with a warning (`model_executor/kernels/linear/__init__.py`
`_resolve_backend_kernels`); the CUDA block-FP8 kernel list has no FlashInfer TRT-LLM kernel and puts
`FlashInferFp8DeepGEMMDynamicBlockScaledKernel` first; `triton_unfused` is not an FP8 MoE backend;
vLLM's reproducibility guide makes offline runs deterministic only with
`VLLM_ENABLE_V1_MULTIPROCESSING=0` or batch invariance.

Changes:
1. Candidates are a fixed list (no inventory phase): Cell D `deep_gemm`, `cutlass`, `triton`, `torch`,
   and `marlin`; Cell M `deep_gemm`, `flashinfer_cutlass`, `hpc`, `marlin`, and `triton`.
2. Every run saves vLLM's kernel-selection log lines in `run.json`. A candidate that logs a fallback,
   or that selects the same kernel as the reference, is an alias: it is excluded from the decision and
   reported as a negative control (it must flip 0 prompts). Cell M `triton` is expected to be an alias
   on Hopper with tensor parallel size 1.
3. Marlin (FP8 weight-only, BF16 activations, a different arithmetic contract) is the positive
   control. It must flip at least 5% of prompts. If it does not, the flip pipeline is suspect and
   STOP-FLAT cannot be applied. An INT8 W8A8 pair (the Integer Alibi divergence) stays an optional manual
   step, because its checkpoint was not verified.
4. Determinism: `VLLM_ENABLE_V1_MULTIPROCESSING=0` in every run. Prefix caching is off in every run.
5. Flip runs use 256 new tokens (GSM8K answers need them); throughput runs use 128. Prompts longer than
   3,072 tokens are dropped by seeded order. Flips are reported per source.
6. Throughput: 5 repeats; the reference runs twice with throughput (R_a, R_b), and the run-to-run
   spread of R sets the noise.
7. Rules (all heuristics, replacing Section 5):
   - V1: R_a and R_b agree on at least 95% of prompts. V2: each cell has at least one working
     candidate that is not an alias.
   - Safe = 0 flipped prompts; unsafe = at least 10%; Marlin is excluded from both.
   - STOP-FLAT: in both cells every candidate is safe with a Wilson 95% upper bound of at most 2%
     (at least 189 prompts), the speed spread among R and the safe candidates is under
     max(2%, 2 x the R noise) at every batch size, and the positive control passed.
   - CONTINUE: in at least one cell an unsafe candidate exists, and either the fastest safe backend
     changes between batch sizes with a margin of at least 3%, or an unsafe candidate is at least 3%
     faster than the fastest safe backend at some batch size.
   - STATIC-WINNER: one safe backend is fastest at every batch size by at least 3% and no unsafe
     candidate beats it by 3%. This is the journal's "performance ordering is static" row: PIVOT to
     static qualification.
   - Otherwise INCONCLUSIVE, with the failed condition named.
8. Time: no inventory, no BF16 control, a 25-minute timeout per run, a 10-second pause between runs,
   and a stop on new runs after 3 hours 10 minutes. Estimate: 14 runs of about 10 minutes plus setup,
   about 2 hours 45 minutes.
9. Robustness: `compare.py` handles missing runs; the analysis always runs; `pip install datasets`
   in step 0; the image entry point is overridden with `--entrypoint sh` (runbook).

Not changed: the models, the prompt sources and seed, `max_num_seqs` 64, and the 4-hour / $18 cap.
`inventory.py` stays in the folder but is no longer used by `session1.sh`.

## 10. Mac dry run and mechanical fixes (2026-10-20, before any GPU run)

Setup (FACT): Docker on the Mac (linux/arm64, 8 CPUs, 7 GB). The image `vllm-cpu:v0.32.0` was built
from the v0.32.0 source with `max_jobs=1`. Model Qwen/Qwen3-0.6B (BF16) from the local cache, no
network. 16 local prompts, 32 flip tokens, 16 throughput tokens, batch sizes 1 and 4, 2 repeats. The
runs were R_a, R_b, and one candidate (`--linear-backend torch`).

Result (FACT, mechanics only): `dryrun/results/dryrun-20261020T194257Z`. All three runs exited 0.
R_a and R_b agreed on 16 of 16 prompts. Top-20 logprobs were saved and read. The candidate flipped 0
of 16 prompts. The decision was INCONCLUSIVE (no positive control on CPU). The speeds are not results.

The dry run found five defects. Each one is fixed:
1. `prompts.py --source local` assumed the repository layout. It now takes `--local-prompts`.
2. `dryrun.sh` wrote a log before its directory existed. It now makes the directory first.
3. vLLM v0.32.0 CPU bug: the CPU fallback `_compute_slot_mapping_kernel_impl` takes its arguments in an
   order different from the Triton kernel. vLLM main fixed it in PR #61992. The dry run mounts a
   patched copy of the file. The patch is CPU-only. The H100 session uses the unmodified official image.
4. `compare.py` read the file `cand_torch.log` as a candidate. Candidates are now directories only.
5. The in-process log handler in `run_cell.py` captured nothing, because vLLM's logging setup replaces
   it. `compare.py` now parses each run's `log.txt` with a search, because vLLM prefixes each line.
   This replaces the mechanism in Section 9 point 2 ("saves ... in `run.json`"). The alias rule does
   not change.

Source check in the v0.32.0 image (FACT): the linear path logs "Selected %s for %s" with `info_once`
(`model_executor/kernels/linear/__init__.py`). The FP8 MoE path logs "Using {backend} Fp8 MoE backend
out of potential backends: [...]" with `info_once` (`fused_moe/oracle/fp8.py`). Both are at INFO
level. An explicit MoE backend that does not support the configuration raises a ValueError. That run
fails; it is not an alias.

Limit and guard: a BF16 model on CPU logs no selection line, so the parser is tested only on synthetic
logs (`test_compare.py`, 14 tests pass). `compare.py` now reports `reference_kernels` and
`kernel_log_found` for each cell, and `kernels` for each candidate. `session1.sh` logs a WARNING if
R_a has no selection line. If `kernel_log_found` is false on the H100, alias detection did not run.
Then check each candidate's `log.txt` by hand before you report the decision. These are reporting
changes only; no threshold or rule changes.

## 11. Amendment A2 (2026-10-22, Runpod efficiency, before any run; amended after R-39)

Reason: D-56 moves the GPU runs to Runpod. The user asked for "maximum details and data with maximum
savings", and accepted a longer run if it gives more data. A Pod bills for each second of the session,
setup and idle time included. The fixed cost of a session (image pull, model download, engine compile)
is paid once. So A2 puts more measurements into the same session, keeps the GPU busy, and never loses
a finished run. Where this section differs from Sections 9 and 10, this section wins. R-39 (Sonnet
pre-run review, `dossier/REVIEW_E-36_A2_PRERUN_2026-10-22.md`) returned AMEND_THEN_RUN; its fixes are
in the text below.

Phase A is not changed. It has the same runs, flags, order, thresholds and 3 h 10 min cap as A1. The
Session 1 decision comes only from the Phase A `summary.json`. `compare.py` is not changed.

### 11.1 Changes to the session

1. Resume. `session1.sh` skips a run whose `run.json` has status ok, and runs a failed or missing run
   again. The results path has no time stamp (`$RESULTS_ROOT/session1`). The elapsed time is kept in
   `elapsed_s`, so the caps count across restarts.
2. One GPU. Each start records the GPU UUID. If it differs from the first start, the script stops
   with exit 2 (override: `ALLOW_GPU_CHANGE=1`). `run.json` also records the GPU name, UUID and SM.
3. Persistent disk. On Runpod, the code, results and caches are on the Pod volume disk (`/workspace`),
   which keeps its data when the Pod stops (runbook Section 2a).
4. One environment folder for each start (`env-<UTC stamp>/`): `nvidia-smi -q`, the GPU UUID, the
   `experiments/runpod/hardware_record.sh` output, versions, model snapshot paths with their revision
   hashes (`models.txt`), GPU telemetry, and a 20-second GPU microbenchmark (BF16 GEMM, device copy) at
   the start and the end (`experiments/runpod/gpu_microbench.py`).
5. GPU telemetry every 1000 ms, in UTC: power, SM clock, temperature, utilization and the clock event
   reasons. If the driver rejects `clocks_event_reasons`, the script uses `clocks_throttle_reasons`.
6. The Pod environment. An SSH shell does not get it, so the script reads only `RUNPOD_POD_ID`,
   `RUNPOD_API_KEY` and `HF_TOKEN` from `/proc/1/environ`. It writes no value to a file.
7. The two models download in the background while the prompts are built; the download ends before
   the first run.
8. `run_cell.py --profile-kernels`: after all measurements of a run, torch.profiler records the CUDA
   kernels of one `generate()` call (32 tokens) at batch 1, 8 and 64 in `kernels.json` (top 150 and
   every kernel with at least 0.5% of device time). It runs on Phase A R_a and every candidate, not on
   R_b. `run.json` is written before the profile and again after it, so a profiler failure cannot lose
   the run. `timeout -k 30` kills a hung run.
9. Bug fix (A1 defect found by the A2 stub test): `session.log` printed `exit $?` after command
   substitutions in the same line, so the logged exit code could be wrong. The code is now read first.
   The analysis never used the logged code, so no A1 rule is affected.
10. Optional `AUTO_STOP=1`: on any exit, the Pod stops itself through the Runpod REST API (three tries,
    HTTP code logged). A watchdog also stops it at `POD_LIMIT` (6 h 40 min). Stop, not terminate, so the
    volume disk keeps the results.

### 11.2 Phase B (exploratory, reported only)

Phase B starts after the Phase A analysis. No new run starts after 6 h. Each part reuses the Phase A
layout. In order of value per GPU-minute:
- B0: the candidate with the most flips in each cell runs again (outputs only). Are its flips
  repeatable?
- B1: the full flip test again at `max_num_seqs` 8 (outputs only), both cells, R twice and every
  candidate. It measures threat 1 of Section 7 (flips that depend on the batch configuration).
- B3: the reference in eager mode (no CUDA graphs), both cells, twice: flips and speed against Phase A
  R_a.
- B2: decode speed at batch sizes 1, 2, 4, 8, 16, 32, 64, 128 and 256 (`max_num_seqs` 256), both cells,
  R twice and every candidate. It shows crossovers between and above the frozen batch sizes. It is
  last, so the cap cuts it first.

### 11.3 Exploratory summaries (`extras.py`, reported only)

`extras.json` holds:
- telemetry per run window: median SM clock and power, peak temperature, the power-cap fraction and
  the hard-slowdown fraction (thermal or power brake);
- `suspect_speed` for a run with a hard-slowdown fraction above 5%, or a median SM clock more than 5%
  below the session median (heuristics);
- a one-GPU check over all runs, and both microbenchmarks;
- a kernel check, a weak second alias signal: the GEMM and MoE kernel names and time shares of each
  candidate against R_a at batch 64. If it disagrees with the log-line rule, report both and read the
  candidate's `log.txt` by hand;
- B0 flips (repeat against first run, and against R_a), B1 flip counts (not the `compare.py` label,
  which is always INCONCLUSIVE without speeds), B2 rankings with the Phase A status of each backend and
  the KV cache size, and B3 eager-mode flips and speed ratios.

### 11.4 Time, cost and tests

Time (INTERPRETATION, from about 10 minutes for each full run; not measured): Phase A 2 h 30 min to
3 h 10 min; B0 10 to 20 min; B1 60 to 90 min; B3 20 to 40 min; B2 60 to 100 min. The last run can start
at 6 h, so the Pod limit is 6 h 40 min. Runpod H100 SXM Secure Cloud was $3.99 per hour on 2026-10-22,
so the full session costs at most about $27, and Phase A only (`PHASE_B=0`) about $13 to $15. The spend
needs the user's approval (D-51, D-57).

Tests (FACT, 2026-10-22, Mac): `test_compare.py` and `test_extras.py`, 22 tests pass. A stub session
(stub `nvidia-smi` that rejects the new field name, stub `timeout`, `pip` and `run_cell.py`, one
candidate forced to fail) wrote 46 runs, the decision, Phase B and `extras.json`. The second start
skipped 43 ok runs and ran the 3 failed runs again. A start with a different GPU UUID stopped with
exit 2. `run_cell.py --profile-kernels` in the vLLM CPU image ended ok and wrote the CPU profiler
failure as `profile_error`; the kernel table itself needs CUDA (VERIFY on the H100).
