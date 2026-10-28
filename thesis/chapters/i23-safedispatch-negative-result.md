# Chapter draft: Is the default kernel backend the wrong choice? A negative result for correctness-aware dispatch (I-23)

Status: DRAFT 1, 2026-10-28. Source of truth: `RESEARCH_JOURNAL.md` (E-36 to E-42, R-39 to R-52, D-63, D-64).
Every number in this chapter comes from a frozen experiment package in `experiments/`. Labels: FACT (measured),
INTERPRETATION (our reading), HYPOTHESIS (not tested). Thresholds are project heuristics, not literature standards.

## 1. Summary

Modern LLM serving engines give several interchangeable implementations ("backends") for the same operation,
for example FP8 or FP4 matrix multiplication and mixture-of-experts (MoE) layers. A correctness-aware dispatch
layer (SafeDispatch, idea I-23) would pick the fastest backend for each regime, but only when the swap keeps
the model output correct. Such a layer is only useful if the engine's default backend is often slower than a
safe alternative.

We tested that premise on two GPU generations and two quantization formats, in seven preregistered
experiments (E-36 to E-42) on rented GPUs. The answer is no for the cells that we measured:

- FACT: on one H100 with block-scaled FP8, the vLLM default backend was the fastest, or the alternatives were
  within about 1% to 2.5% on a narrow batch band (Marlin MoE, batch 4 to 16), below a 3% bar.
- FACT: on one B200 with NVFP4, the default backend was the fastest at batch 1 to 128. At batch 256 to 512 the
  best alternative was at most about 4% faster (not resolved) and 6% slower at batch 128.
- FACT: on the B200, five FP4 x FP4 linear backends give bit-equal logprobs for the tested model. There the
  backend choice is a speed choice only. Batch composition and compile mode change the numbers far more than
  the backend choice.

INTERPRETATION: for these settings, a static default is close to optimal, and a dynamic, correctness-aware
dispatcher has at most a few percent to gain. We closed I-23 for both cells (decisions D-63 and D-64, by the
user). The method (paired rounds, alias detection, preregistered rules, independent reviews) and its failure
modes are reusable contributions.

## 2. Motivation and the hypothesis under test

Serving stacks such as vLLM expose several implementations for GEMM, attention, MoE and quantization. They
implement the same math, but finite-precision behavior, scale application, reduction order and fused
quantization kernels differ. Autoregressive decoding amplifies small numeric changes: one flipped token changes
the whole continuation.

Prior work already shows two facts (journal I-23 dossier, R-05):

- H1 (published): some backend swaps change the output in ways a small allclose test misses.
- H2 (published): unsafe swaps correlate with shape, precision, logit margin and similar features.

The open part, and the premise of a dispatch system, was:

- H3: the fastest safe implementation changes across regimes (batch size, model type, GPU) often enough to
  matter, compared with one static default.

If H3 is false, a dispatcher has no speed to win, whatever its safety logic. So we measured H3 first, before
building anything. This follows the project rule "measure the oracle upper bound before building a scheduler".

## 3. Method

### 3.1 Platform

| Item | Value |
|---|---|
| Engine | `vllm/vllm-openai:v0.32.0` (same image for every run) |
| GPUs | 1 x H100 80GB HBM3 (sm_90) for FP8; 1 x B200 (sm_100) for NVFP4; Runpod Secure Cloud, on-demand |
| Dense models | `Qwen/Qwen3-8B-FP8` (H100), `nvidia/Qwen3-8B-NVFP4` (B200) |
| MoE models | `Qwen/Qwen3-30B-A3B-FP8` (H100), `nvidia/Qwen3-30B-A3B-NVFP4` (B200) |
| Backend flags | `--linear-backend` (dense), `--moe-backend` (MoE) |
| Speed metric | end-to-end `generate()` tokens/s, 128 new tokens, `ignore_eos`, median of repeats |
| Quality metrics | greedy output agreement (E-36), GSM8K accuracy on 1,319 problems (E-38 to E-41), top-20 logprob bit-equality (E-42) |

Every run sets the in-process engine (`VLLM_ENABLE_V1_MULTIPROCESSING=0`), turns prefix caching off, and runs
one warm-up per configuration before the measured runs (amendment A3; see Section 6.2).

### 3.2 Experiment protocol

Each experiment followed the stage gates in `docs/experiments/PROTOCOL.md`:

1. A preregistration README with the question, the frozen rule (GO, KILL, otherwise INCONCLUSIVE) and the
   validity checks, written before any GPU run.
2. Unit tests of the analysis on synthetic data, and a stub or CPU dry run of the session script.
3. A pre-run review by a different model, with an amendment that wins over the original text.
4. The user's approval of the cost cap, and a live balance check.
5. The run, with resume, telemetry, per-run timeouts and an automatic Pod stop.
6. A copy with checksums, teardown, a run ledger entry and a hardware log entry.
7. A post-run review by a different model, then a decision recorded with the name of who decided.

### 3.3 Speed comparison

A single process per backend is noisy. The reference R (the default backend) ran twice and differed by 4% to 7%
at some batch sizes (E-40). So every decision used paired rounds: R and the candidate C alternate (R, C in odd
rounds, C, R in even rounds), each round gives log(median C / median R), and a one-sided t bound with a
Bonferroni correction over the deciding batch sizes decides. GO needs a lower bound of at least 1.03; KILL needs
upper bounds below 1.03. Section 5 shows why this matters: single-process "leads" of 5% to 7% disappeared under
pairing.

### 3.4 Alias detection

A backend flag can fall back to the default kernel without an error. Such an alias looks "identical" and "as
fast". We parse the kernel-selection line of each run's log (for example "Using FlashInferCutlassNvFp4LinearKernel
for NVFP4 GEMM") and drop or invalidate candidates that select R's kernel. Section 6.1 describes a case where the
parser missed a new log format.

### 3.5 Quality contracts

We used three contracts, in this order, because each one showed a limit of the one before:

1. Exact greedy output (E-36). Too strict: any numeric change flips most outputs, including eager mode against
   graph mode.
2. Task accuracy (A1, E-38 to E-41): a paired bootstrap on GSM8K, with the lower bound of the accuracy
   difference above -2.0 percentage points. A distribution contract (A2, teacher-forced KL against a BF16 anchor)
   was also tested in E-38 and retired after it.
3. Bit-equality of top-20 logprobs (E-42), with a determinism control, a large-difference control and a
   small-change control.

## 4. Results

### 4.1 H100, block-scaled FP8 (E-36, E-38, E-39)

| Experiment | Cell | Question | Frozen decision | Key facts |
|---|---|---|---|---|
| E-36 | dense and MoE | Do alternatives change greedy output; does the fastest backend change with batch? | INCONCLUSIVE (only one safe backend) | Every working alternative changed greedy output on 87.5% to 95.3% of 256 prompts; R was the fastest at batch 1 to 64 in both cells (dense R 10,281 tok/s at batch 64) |
| E-38 (rerun) | dense and MoE | Is an alternative within 2 pp on GSM8K, within half of BF16 KL, and at least 3% faster? | KILL | All real candidates pass A1 (within about 1 pp); all fail A2 (KL 0.60 to 0.93 x BF16, bar 0.5); Marlin MoE not faster (batch 8 +5.5% unclear, batch 256 -10.7%) |
| E-39 | MoE | Is Marlin W8A16 MoE at least 3% faster at batch 4 to 32? (10 paired rounds) | INCONCLUSIVE | Marlin / R: batch 4 1.015 [0.983, 1.047], 8 1.025 [1.002, 1.049], 16 1.011 [0.992, 1.031], 32 0.995 [0.976, 1.015]; below 1 at batch 1, 2, 64, 128 and 256 (256: 0.887) |

Values in brackets are one-sided bounds at alpha 0.0125.

INTERPRETATION: on Hopper FP8, the only candidate with any gain is Marlin MoE, about 1% to 2.5% at batch 4 to 16.
A batch-aware dispatcher could win at most a few percent on a narrow band, and loses more if it picks wrong.
Decision D-63 (user): close I-23 for Hopper block-FP8.

### 4.2 B200, NVFP4 (E-40, E-41)

| Experiment | Cell | Question | Frozen decision | Key facts |
|---|---|---|---|---|
| E-40 | dense and MoE | Is any NVFP4 backend at least 3% faster at batch 1 to 128, with A1? | INCONCLUSIVE | MoE: R = FLASHINFER_TRTLLM; every real backend slower at every size (0.42 to 0.95); dense: every real backend slower at batch 1 to 32; screen lead at batch 256 (cudnn 1.07, flashinfer_cutlass 1.05); b12x failed to start |
| E-41 | dense | Is cudnn or flashinfer_cutlass at least 3% faster at batch 256 or 512? (10 paired rounds) | INCONCLUSIVE (no evidence of a lead) | flashinfer_cutlass: batch 128 0.937 [0.927, 0.947], 256 1.015 [0.988, 1.043], 512 1.001 [0.985, 1.018]; cudnn failed at start in 7 of 10 runs |

INTERPRETATION: the default NVFP4 backend is the fastest or tied at every measured size in both cells. The E-40
lead at batch 256 was noise: in E-41, 8 of 10 paired rounds at batch 256 were between 0.97 and 1.02. A cutlass
gain of up to about 4% at batch 256 is not excluded, but it comes with a 6% loss at batch 128. Decision D-64
(user): close I-23 for NVFP4 on B200.

### 4.3 Why the accuracy gate could not separate the NVFP4 backends (E-42)

In E-41, R, cudnn and flashinfer_cutlass gave the same tokens on all 1,319 GSM8K problems (92.27%), although the
logs named three different kernels. On H100 FP8, real backend changes had changed most outputs (E-38: Marlin kept
149 of 1,319 outputs, DeepGEMM 82). E-42 tested the explanation directly.

| Run (against R, CuTe-DSL, eager) | Kernel in log | Same token lists (of 256) | Bit-equal top-20 positions |
|---|---|---|---|
| R_2 (determinism control) | CuTe-DSL | 256 | 16,384 of 16,384 |
| flashinfer_cudnn | cuDNN | 256 | 16,384 of 16,384 |
| flashinfer_cutlass | FlashInfer CUTLASS | 256 | 16,384 of 16,384 |
| cutlass (vLLM) | vLLM CUTLASS | 256 | 16,384 of 16,384 |
| flashinfer_trtllm | TRT-LLM | 256 | 16,384 of 16,384 |
| marlin (W4A16, large-difference control) | Marlin | 12 | 0 |
| R_m (R with max_num_seqs 64, composition control) | CuTe-DSL | 132 | 67.7% |
| R_g (R in graph mode, against eager R) | CuTe-DSL | 7 | 0 |

In graph mode, flashinfer_cutlass was also bit-equal to R in graph mode.

INTERPRETATION: for this model on B200, the five FP4 x FP4 linear paths compute the same bits. A likely reason
(HYPOTHESIS, not tested at the instruction level) is that they share the activation quantization and use the
same block-scaled FP4 MMA with the same K order. So the A1 gate was vacuous for these kernels on this model and
shape. The choice among them changes only speed. In contrast, changing the batch composition or the compile
mode changes the logits strongly. A correctness-aware dispatcher would spend its safety logic on a choice that
has no numeric effect here, while the larger numeric effects come from elsewhere.

### 4.4 Cost

| Experiment | GPU | Cost (about) |
|---|---|---|
| E-36 rehearsals (L4, RTX 4090) | L4, RTX 4090 | $1.40 |
| E-36 Session 1 | H100 | $11.20 |
| E-38 (incomplete run and rerun) | H100 | $14.32 + $7.60 |
| E-39 | H100 | $5.20 |
| E-40 | B200 | $9.00 |
| E-41 | B200 | $10.70 |
| E-42 | B200 | $21.60 (includes about $16 of idle time, Section 6.5) |
| Total | | about $81 |

## 5. Discussion

**What the result says.** For one dense and one MoE model on each of two GPU generations, the vLLM default
backend is already the fastest or within a few percent of the fastest at the batch sizes that we measured. The
dynamic part of the I-23 premise (H3) does not hold here. The safety part of I-23 is still real on Hopper (FP8
alternatives change most greedy outputs), but on these cells it guards a speed gain that does not exist.

**Why this matters.** A dispatch layer has fixed costs: qualification runs, a runtime decision, and the risk of
a wrong pick (for example cutlass is 6% slower at batch 128 on B200, Marlin is 11% slower at batch 256 on H100).
With a best-case gain of 1% to 4% on narrow bands, these costs remove most of the benefit.

**What single-process screens would have concluded.** E-40's one-process screen suggested 5% to 7% gains at
batch 256. Paired rounds showed they were noise. A study without pairing, or with one run per backend, would have
reported a false positive. This is a methodological point for backend-comparison studies in general.

**Why exact output agreement is the wrong contract.** E-36 showed that eager mode against graph mode, with the
same backend, changes 233 and 238 of 256 greedy outputs. E-42 showed the same effect from a change of
`max_num_seqs`. So an exact-match contract rejects every alternative and also rejects routine engine settings.
Task accuracy (A1) is a usable contract; on Hopper all alternatives passed it.

## 6. Lessons from running the experiments

### 6.1 A silent alias from a new log format
The E-40 alias check parsed the FP8-era kernel line. The NVFP4 dense line has a new format ("Using X for NVFP4
GEMM"), so the reference kernel was not found, and the selected candidate (flashinfer_cutedsl) was in fact the
default kernel. Lesson: validate every log parser on a real log of the exact engine version and format, and make
"no kernel line found" a validity failure, not a pass.

### 6.2 Cold caches change greedy output
In the E-36 rehearsal on an L4, the first run of a session (cold torch.compile and Triton caches) agreed with
later identical runs on only 34.8% of prompts. Lesson: warm up each configuration before its measured runs (A3).

### 6.3 Backend start-up failures are part of the result
cuDNN NVFP4 failed at start in 7 of 10 graph-mode runs with `CudnnPlanBuildInCaptureError` (a plan build during
CUDA graph capture), but ran in eager mode. b12x failed to start in both E-40 cells. A dispatcher must treat
"does not start reliably" as a property of the backend.

### 6.4 Analysis code must be timed at the real data size
The first E-38 run lost $14 because a reference loader decompressed the whole array for each position (quadratic
time), which only showed at the full data size. Lesson: time every per-item loop at the real size before booking.

### 6.5 An idle GPU is the largest cost risk
In E-42, the Pod billed for about 2 h 20 min before the session started, because the agent waited for a
background timer whose wake-up came late. The session watchdog starts only with the session. Lesson: a watchdog
in the Pod start command, counted from boot, and no waiting between Pod creation and session start
(`docs/experiments/RUNPOD.md` Section 3).

## 7. Threats to validity

- One model per cell (Qwen3-8B and Qwen3-30B-A3B), one GPU of each type, one engine version (vLLM v0.32.0).
- End-to-end tokens/s with 128 new tokens; prefill is included and its share grows with batch size.
- Batch sizes up to 256 (H100) and 512 (B200); no long-context, no tensor-parallel (TP > 1) runs.
- GSM8K only for task accuracy.
- E-42 covers 256 prompts, 64 new tokens and `max_num_seqs` 256; batch sizes below about 8 were not tested;
  graph mode was checked only for R and flashinfer_cutlass. The bit-equality covers the whole linear path
  (activation quantization plus GEMM), not the GEMM alone.
- Newer engines or kernels can change the default or the ranking. The result is a fact about this setup, not a
  law.

## 8. Conclusion

For block-scaled FP8 on H100 and NVFP4 on B200, the vLLM v0.32.0 default backend is the right static choice for
the tested models and batch sizes. The upside left for a correctness-aware dynamic dispatcher is at most a few
percent on narrow bands, and on B200 the NVFP4 backend choice does not change the numbers at all. We therefore
closed I-23 for both cells. The reusable parts are the paired-round speed method, the alias check, the layered
quality contracts, and the operational lessons in Section 6.

## Appendix A. Artifact map

| Experiment | Package | Results (local) | Reviews |
|---|---|---|---|
| E-36 | `experiments/safedispatch_testA/` | `results/session1-h100-20261026/` | R-39, R-41, R-42 |
| E-38 | `experiments/safedispatch_testA2/` | `results/e38-rerun-20261027/` | R-43, R-44 |
| E-39 | `experiments/marlin_moe_speed/` | `results/e39-h100-20261027/` | R-45, R-46 |
| E-40 | `experiments/nvfp4_dispatch/` | `results/e40-b200-20261027/` | R-47, R-48 |
| E-41 | `experiments/nvfp4_large_batch/` | `results/e41-b200-20261027/` | R-49, R-50 |
| E-42 | `experiments/nvfp4_bitequal/` | `results/e42-b200-20261028/` | R-51, R-52 |

Hardware records: `experiments/runpod/HARDWARE_LOG.md`. Run ledger: `experiments/ledger/runs.jsonl`.
Decisions: journal Section 10 (D-56 to D-65).
