# E-30: I-118 premise test (CPU load and the shared module power budget on GH200)

Status: PREDECLARED (draft 2026-10-16, frozen 2026-10-20), AMENDED (A1, Section 8) after the R-34
Sonnet pre-run review, which said DO_NOT_RUN as written. Not run. Needs one GH200 with root and the
user's approval (D-52). Where Section 8 differs from Sections 2 to 5, Section 8 wins.

Labels: FACT, INTERPRETATION, HYPOTHESIS. All thresholds and weights are project heuristics.

## 1. Rubber duck

On GH200 the GPU gets the module power limit minus what the Grace CPU draws (NVIDIA Grace tuning guide,
FACT). The CPU has priority (2605.01938, FACT). So CPU work on the same module (agent tools, tokenizers,
CPU KV offload) may lower GPU clocks when the module limit binds, mostly during compute-bound prefill;
memory-bound decode draws less (2605.11999). If the effect exists, a phase-aware CPU cap (cap the CPU
only during prefill) could beat one static CPU cap. The static cap is the strong baseline: JUPITER ships
a 100 W default CPU cap for exactly this reason.

What kills it: (1) no measurable prefill loss at realistic limits; (2) a static CPU cap already captures
almost everything a perfect per-phase cap could.

## 2. Preflight (first 10 minutes; stop if it fails)

`preflight.sh` records and checks:
- root (`sudo -n true`), driver, `nvidia-smi -q -d POWER` (module scope present, current and default
  module limit, whether `nvidia-smi -pl <W> -sc 1` is accepted);
- Grace power sensors and caps under `/sys/class/hwmon/*` (`power*_input`, `power*_cap`);
- `stress-ng` present, core count, NUMA layout.

If the module limit cannot be lowered, run the default-limit part only and say so in the result. If
no CPU cap is writable, KILL 2 cannot be tested; report KILL 1 only.

## 3. Measurements (frozen)

- Engine: `vllm/vllm-openai:v0.32.0` (aarch64 image), `Qwen/Qwen3-8B` BF16, pinned to cores 0 to 7
  (`taskset`). Prefill-heavy: 64 prompts of about 4,000 tokens, 1 new token. Decode-heavy: 64 prompts
  of about 200 tokens, 256 new tokens, `ignore_eos`.
- Control: `sweep.py` phase 1, a BF16 matmul loop (8192 x 8192) for 60 s, reporting TFLOPS per second.
- CPU load: `stress-ng --cpu N --metrics-brief` on cores 8 and up, N in {0, 18, 36, 54}.
- Module limits: provider default; 680 W (JUPITER, a documented deployment) if settable.
- CPU caps (for KILL 2, at N = 54): none, 150 W, 100 W, 70 W, applied through hwmon `power1_cap`.
- Logging: `nvidia-smi --query-gpu=... -lms 100` (power, SM clock, clock event reasons), module power
  by `nvidia-smi -q -d POWER` every second, Grace power from hwmon every second.
- Each point: 3 repeats; the median is used.

## 4. Frozen rules

- KILL 1: at the default limit, and at 680 W if settable, prefill throughput with N = 54 is within 3%
  of N = 0 (and the GEMM control within 3%). Then the coupling does not matter at realistic limits.
- KILL 2: let T_p(c), T_d(c) be prefill and decode GPU throughput with CPU cap c, and C_p(c), C_d(c) the
  stress-ng operations per second during each phase. For a prefill time share f in {0.3, 0.5, 0.7},
  U(c_p, c_d) = f * T_p(c_p)/T_p(none) + (1 - f) * T_d(c_d)/T_d(none)
              + (f * C_p(c_p) + (1 - f) * C_d(c_d)) / ((f * C_p(none) + (1 - f) * C_d(none))).
  Static best = max over c_p = c_d. Oracle = max over all (c_p, c_d). KILL 2 if the oracle gain over
  the static best is under 5% for every f. The equal weight of GPU and CPU work is a heuristic.
- GO (to a mechanism): prefill loss of at least 10% at a documented limit, and an oracle gain above 5%
  for some f.
- PIVOT (measurement note): a loss of at least 3% exists only below documented limits, or the loss is
  3% to 10%.
- If KILL 1 holds, KILL 2 is not needed.

## 5. Time and cost

About 2 hours: preflight 10 min, model and image pull 20 min, KILL 1 sweep (2 limits x 4 loads x
2 workloads x 3 repeats, about 1 min each) about 50 min, KILL 2 caps (4 caps x 2 phases x 3 repeats)
about 25 min, analysis and copy-off 15 min. GH200 hourly price: not confirmed (an older listing said
about $1.49 to $3.19 per hour on Lambda, SNIPPET); check before booking. Hard cap: 2.5 hours.

## 6. Threats

- A cloud VM may hide module telemetry or refuse caps (preflight checks).
- The GH200 may run with a module limit far above its draw, so nothing binds (that is a KILL 1 answer).
- stress-ng compute load is not an agent workload; it is an upper bound on CPU power.
- One model, one image, one machine.

## 7. Files

- `preflight.sh`: capability check, writes `preflight.json`.
- `sweep.py`: the GEMM control (phase 1) and the KILL 1 and KILL 2 sweeps on one long-lived vLLM
  engine (phase 2); writes `raw.jsonl` and `medians.json`; `--dry-run` prints the plan.
- Logging: run `nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks_event_reasons.active --format=csv -lms 100`
  in the background during `sweep.py` (command in the GPU runbook).
- `analyze.py`: the frozen rules; `test_analyze.py`: unit tests on synthetic data.

## 8. Amendment A1 (2026-10-20, after R-34, before any run)

R-34 (`dossier/REVIEW_E-36_E-30_PRERUN_2026-10-20.md`) found three blockers and four major issues.
Changes:
1. Prefix caching is off in the engine. The old prefill test repeated the same 64 prompts, so after the
   warm-up it measured cache hits and an idle GPU, and KILL 1 would pass by construction. GPU power is
   now sampled during prefill. Validity rule: the median prefill power at N = 0 must be at least 60% of
   the GPU power limit; otherwise the result is INVALID (the benchmark is not compute-bound).
2. CPU caps: both `hwmon*/power*_cap` and `hwmon*/device/power*_cap` (NVIDIA's documented layout) are
   used. The preflight writes each cap and the module limit to their current values to prove that
   writes work, and checks that 680 W is within the module's [min, max].
3. The CPU-cap sweep runs at the module limit with the largest prefill loss, not at the default.
4. KILL 2 is decided on a constrained form: maximize the GPU term subject to keeping at least 80% of the
   uncapped CPU work (phase-weighted), for f in {0.3, 0.5, 0.7}; KILL 2 if the best per-phase pair beats
   the best feasible static cap by under 5% for every f. The equal-weight utility (Section 4) is still
   reported, for CPU weights 0.25, 0.5 and 1, but does not decide.
5. Loads run in a seeded random order in each repeat, and N = 0 runs again at the end of each limit. If
   N = 0 drifts by more than 2%, the result is INCONCLUSIVE.
6. Each limit and the cap sweep run in their own error handler; `medians.json` is written after each
   stage. The PIVOT clause "only below documented limits" is removed (no lower limit is tested).
7. stress-ng stops with SIGINT (it prints its metrics) and a parse failure is an error, not zero work.
   A 5-second ramp precedes each measurement.
8. The process pins itself to cores 0 to 7 at start (before torch or vLLM start threads) and refuses to
   run with fewer than 62 cores. The GEMM control runs 20 s, 5 repeats.

Rules after A1: INVALID (power) > INCONCLUSIVE (drift) > KILL 1 > KILL 2 (constrained) > GO (prefill
loss of at least 10% at a documented limit and a constrained gain above 5% for some f) > PIVOT (loss 3%
to 10%). All thresholds are heuristics. Unit tests: 9, passing.
