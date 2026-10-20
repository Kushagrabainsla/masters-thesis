# R-34: Sonnet pre-run review of E-36 (I-23) and E-30 (I-118), 2026-10-20

Reviewer: an independent Sonnet agent (D-29), read-only on the repository. It stalled twice (a network
outage and an API usage limit) and was resumed both times. It read both packages, ran both unit-test
suites (8 and 6 tests, all passing at the time), and read the local vLLM v0.32.0 source tree (`V:`).

Main-session checks after the review (FACT, v0.32.0 source): `_resolve_backend_kernels` falls back to
normal selection with a warning when a requested linear backend has no kernel; the CUDA block-FP8
kernel list is FlashInferFp8DeepGEMMDynamic, DeepGemm, Cutlass, B12x, Marlin, Humming, Triton,
BlockWiseTorch (no FlashInfer TRT-LLM); `map_fp8_backend` has no `triton_unfused`; the reproducibility
guide names `VLLM_ENABLE_V1_MULTIPROCESSING=0` or batch invariance for deterministic offline runs.
One reviewer number was corrected: a Wilson 95% upper bound of at most 2% for 0 flips needs n >= 189,
not 188 (3.84 / (n + 3.84)).

The review text follows as the reviewer returned it.

---

# R-34 pre-run review of E-36 and E-30

I did not edit the repository. All scratch output is in `scratchpad/r34/`. Both unit-test suites pass (8 tests and 6 tests). Code claims come from the local vLLM v0.32.0 tree, abbreviated `V:` below.

## E-36 (I-23 Test A, Session 1): AMEND_THEN_RUN

**Checks that pass (FACT)**
- (a) `LLM(**kwargs)` goes to `EngineArgs`. That class has `linear_backend`, `moe_backend` and `attention_backend` (`V:arg_utils.py:553-554,791`; `V:llm.py:170,336`). The attention string is converted to the enum (`arg_utils.py:2693`).
- (b) `FlatLogprobs` yields one `dict[int, Logprob]` per position (`V:logprobs.py:109-155`; `V:outputs.py:83`), so `topk()` works. `logprobs=20` equals the `max_logprobs` default.

**Findings**

1. **BLOCKER, FACT. "Starts" does not mean "is a different kernel."**
   - `_resolve_backend_kernels` falls back to auto with only a WARNING when the requested backend has no kernel for the layer (`V:model_executor/kernels/linear/__init__.py:395-427`).
   - The block-FP8 CUDA list (`:478-488`) has no `flashinfer_trtllm` kernel, so that "candidate" is a copy of auto.
   - `flashinfer_cutlass` maps to `FlashInferFp8DeepGEMMDynamicBlockScaledKernel`, which is first in the auto priority list. It is therefore probably R.
   - For MoE on Hopper with TP, auto moves TRITON to the front (`V:fused_moe/oracle/fp8.py:111-118`), so the `triton` candidate equals R.
   - `triton_unfused` is not in `map_fp8_backend` (`:267-285`), so it fails with certainty.
   - `cutlass` raises "disabled" (`:464-472`; `quantization/fp8.py:522`).
   - `hpc` is documented as "FP8 and Hopper only" (`V:config/kernel.py:328`). It is missing from the MOE list.
   - Aliases pass V2 and enter the flat/spread logic as empty candidates.
   - **Amend:**
     - Static candidate list. Dense: deep_gemm, cutlass, triton, marlin, torch. MoE: deep_gemm, flashinfer_cutlass, marlin, hpc.
     - Save the "Selected %s for %s" and "Using %s Fp8 MoE backend" log lines in `run.json`, and drop any candidate that resolves to R's kernel.

2. **MAJOR, FACT. R = auto is not one fixed kernel, and the script never records what auto picked.**
   - The dynamic kernel uses FlashInfer swapAB for M<32 and DeepGEMM for M>=32 (`V:scaled_mm/flashinfer.py`, class docstring).
   - R's numerics therefore depend on batch composition.
   - `inventory.py` discards stderr on success, and `session1.sh` does not tee logs.
   - **Amend:** tee every run to `$dir/log.txt`. INTERPRETATION: also add a fixed-kernel R (cutlass) as a second reference for Cell D.

3. **BLOCKER, FACT. The floor is likely to fail by construction.**
   - `V:docs/usage/reproducibility.md` says offline scheduling is deterministic only with `VLLM_ENABLE_V1_MULTIPROCESSING=0` or batch invariance.
   - The default is multiprocessing on (`V:envs.py:1463`), and `run_cell.py` sets neither.
   - Requests reach the engine core while it already steps, so batch composition varies per run, and the kernels depend on M.
   - Excluding the prompts where R_a and R_b differ corrects one pair only. Candidate runs carry their own noise.
   - INTERPRETATION: V1 (95% agreement) could fail and turn the 3.5 h session into INCONCLUSIVE.
   - **Amend:** `os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"` in `run_cell.py`.

4. **MAJOR, FACT. STOP-FLAT is unreachable as written.**
   - A Wilson upper bound of 0/n at or below 1.5% needs n>=253. I computed 1.478% at n=256, 1.502% at n=252 and 1.55% at n=244. V1 allows 12 excluded prompts.
   - The spread test covers every kept backend, including Marlin and torch, at four batch sizes.
   - INTERPRETATION: noise from a 3-repeat median of 1 to 5 s runs is about 1 to 3%. R_b is outputs-only, so there is no throughput noise floor.
   - **Amend:**
     - `FLAT_UPPER`=2.0% (needs n>=188).
     - Compute spread only over R plus zero-flip candidates.
     - Spread threshold = max(2%, 2x the R-vs-R throughput spread).
     - At least 5 repeats, and R throughput measured twice.

5. **MAJOR, INTERPRETATION. CONTINUE can fire without evidence for H3, and a decisive outcome falls into INCONCLUSIVE.**
   - CONTINUE needs one candidate flipping 10% or more, and any change in the identity of the fastest backend, with no margin (`compare.py:132`).
   - Marlin FP8 is weight-only with bf16 activations, a different arithmetic contract, so it flips almost surely.
   - H3 concerns the fastest safe backend. A static winner (all safe, one backend 5% or more faster at every batch size) is the journal's "performance ordering is static" row, but here it lands in INCONCLUSIVE.
   - **Amend:**
     - Exclude W8A16 from "unsafe".
     - Require the fastest safe backend to change with a 3% margin.
     - Add an explicit STATIC-WINNER outcome.

6. **MAJOR. Positive control (coordinator question): yes, E-36 needs one.**
   - R-33 step 1 says to stop if the harness cannot see a known divergence.
   - Cheapest control (FACT it exists): Marlin W8A16 against auto in Cell D. It is already in the plan and needs no new download.
   - Pre-declare "expect >=10% flips" (INTERPRETATION). If it shows <5%, treat the flip pipeline as broken and do not apply STOP-FLAT.
   - Free negative control: any alias (for example MoE `triton`) must give 0 flips once item 3 is applied.
   - The Integer Alibi pair (INT8 W8A8, CUTLASS vs Triton) is only a flag change here, because `cutlass` and `triton` map to the INT8 kernels. It needs another checkpoint. `RedHatAI/Qwen3-8B-quantized.w8a8` returned HTTP 401 and is unverified. It adds about 10 min.
   - INTERPRETATION: that divergence was published for sm_89, so a Hopper null is ambiguous. Make it an optional last step, not a gate.

7. **MAJOR. The time budget does not fit in 4 hours.**
   - Counting starts: 16 inventory, then 9 dense, 9 MoE and 3 control.
   - With an assumed cold start of 4 to 6 min (INTERPRETATION: torch.compile, DeepGEMM warmup, graph capture), the total is about 3.9 h of compute plus about 25 min setup. The 3 h 20 min cap would cut MoE candidates.
   - **Cuts:**
     - Delete the inventory phase (saves about 80 min). `run_cell` already records `status:error`.
     - Remove the aliases and certain failures (about 30 min).
     - Drop the BF16 control, or keep only R_a and the FLASHINFER candidate.
     - Wrap each run in `timeout 1500`, because `run_cell` has no timeout.
     - Result: about 2.5 h.

8. **MAJOR, FACT. The prefill throughput measurement hits the prefix cache.**
   - `run_cell.py:96-97` repeats identical prompts, and prefix caching is on for generate models (`arg_utils.py:2991-3022`).
   - `compare.py` ignores prefill anyway.
   - **Amend:** pass `enable_prefix_caching=False` in all runs, or drop prefill.

9. **MAJOR, FACT. `compare.py` crashes on a candidate directory without `run.json`.** I confirmed this in scratch.
   - A skipped control also crashes it, because `session1.sh:59` always passes `--control`.
   - This breaks "analysis always runs".
   - **Amend:** guard missing files and use `|| true`.

10. **MINOR.**
    - The image ENTRYPOINT is `vllm serve` (`V:docker/Dockerfile:1332`), so use `--entrypoint sh`.
    - `datasets` appears only in the rocm and xpu requirements. Run `pip install datasets` first.
    - `dryrun/` is cited in the README but absent.
    - There is no prompt-length filter against `max_model_len=4096`.
    - There is no sleep between processes while GPU memory is released.
    - The image digest is not recorded.

11. **MINOR, INTERPRETATION. Prompt set.**
    - GSM8K plus UltraChat with `enable_thinking=False` is reasonable. The Docker tag v0.32.0 exists for amd64 and arm64.
    - 128 tokens truncates GSM8K answers and biases flips downward. Use 256 tokens for the flip runs only (cheap).
    - Report flips per source.
    - 256 prompts resolve 10% (26/256 gives Wilson 7.0 to 14.5%) but barely resolve 1.5%.

## E-30 (I-118 power coupling): AMEND_THEN_RUN (DO_NOT_RUN as written)

1. **BLOCKER, FACT. The prefill benchmark measures prefix-cache hits.**
   - `sweep.py:170-189` repeats the same 64 prompts of about 256k tokens with default prefix caching.
   - The KV cache for an 8B model at about 144 KB per token is about 36 GB, which fits.
   - After warm-up the GPU is nearly idle, so KILL 1 passes by construction.
   - **Amend:** `enable_prefix_caching=False`. Log GPU power during prefill and require it near the GPU peak, or abort.

2. **BLOCKER, FACT. The hwmon path is wrong.**
   - NVIDIA's guide uses `/sys/class/hwmon/hwmonX/device/power1_cap` in microwatts. It needs `CONFIG_SENSORS_ACPI_POWER=m` and the kernel argument `acpi_power_meter.force_cap_on=y` (guide, power-thermals page).
   - `sweep.py:55-57` and `preflight.sh:16` glob `hwmon*/power*_cap`, which does not traverse `device/`. KILL 2 is silently skipped.
   - INTERPRETATION: a cloud VM cannot change the kernel command line, so caps are probably unavailable.
   - `preflight.sh` says it "changes nothing". README Section 2 claims it checks whether `-pl -sc 1` is accepted, but it does not.
   - **Amend:** glob both patterns. Add idempotent test writes: the current cap value, and `nvidia-smi -pl <current> -sc 1`.

3. **BLOCKER, logic. The cap sweep runs at the default limit (`sweep.py:195`), not at the limit where the coupling binds.**
   - If KILL 1 fails only at 680 W, the cap data has flat T_p. A synthetic check gives oracle gain 0.0 for every f, which reads as KILL 2.
   - **Amend:** run the caps at the limit with the largest prefill loss.

4. **MAJOR, FACT plus INTERPRETATION. The equal-weight utility decides KILL 2 by construction.**
   - The oracle gain over the best static cap comes mostly from the CPU term in decode, where the GPU does not care about the cap.
   - Synthetic check: a 20% prefill loss that a 70 W cap fully recovers at a 40% CPU-ops cost gives oracle gain of only 0.7 to 1.2% for f=0.3 to 0.7. That is KILL 2.
   - GO needs a near-collapse, as `test_go` uses.
   - **Amend:** freeze one more rule. Report the gain for CPU weights {0.25, 0.5, 1} and a constrained form (maximize GPU throughput with CPU ops at least 80%), and decide on the constrained form.

5. **MAJOR. Run order is confounded.**
   - Loads ascend 0, 18, 36, 54 within a limit, with the default limit first. Thermal and boost drift correlates with N.
   - The GEMM control cannot separate drift from coupling.
   - **Amend:** interleave or randomize loads per repeat and re-measure N=0 at the end.

6. **MAJOR. An unsettable limit crashes the sweep and loses the data.**
   - `sh(check=True)` aborts the run (`sweep.py:33,50`). `medians.json` is written only at the end (`:214`).
   - Preflight does not check that 680 W lies between the module Min and Max (the guide's example shows Min 200 W, Max 1000 W).
   - The PIVOT "only below documented limits" is unreachable. The sweep has no lower limit, and the `undocumented` set in `analyze.py:49` is always empty.
   - **Amend:** try/except per limit, write medians per stage, and add a 500 W point or delete the PIVOT clause.

7. **MINOR.**
   - `ops or 0.0` (`sweep.py:210`) turns a stress-ng parse failure into C=0. Fail loudly instead.
   - `stop_stress` sends SIGTERM and relies on stress-ng printing the summary. INTERPRETATION: unverified. Prefer waiting for `--timeout` expiry.
   - The regex fits the `cpu` row of `--metrics-brief`, with group 2 as bogo ops/s (real time).
   - Add a 5 s CPU-power ramp before each measurement.

8. **MINOR. Core numbering.**
   - Engine on cores 0-7 and stress on 8-61 (N=54) is fine on 72 cores.
   - Phase 1 runs unpinned and imports torch before `sched_setaffinity` (`sweep.py:91,165`). That call pins only the calling thread, but the spawned engine inherits it.
   - Pin at program start, and assert `os.cpu_count()>=62`, because a VM may expose fewer cores.

9. **What is right (FACT).** `nvidia-smi -pl W -sc 1` is the module scope (`-sc 0` is GPU). The "Module Power Readings" section has "Default Power Limit", and the regex matches. 680 W is a documented JUPITER point and a sensible second limit.

10. **Time and cost.**
    - Actual measurement is about 30 to 40 min. The 60 s GEMM runs dominate at 24 min.
    - Cut to 20 s for 5 repeats.
    - The 2.5 h cap is generous. Cost is about $3 to $5 at the cited prices, which are unconfirmed.

**Apply the 3 E-30 BLOCKERS, then run.**
