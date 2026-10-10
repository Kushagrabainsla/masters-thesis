---
title: "Certifying Output Equivalence of Fast LLM Inference Kernels on Hopper and Blackwell GPUs"
---

**NAIRR Pilot Start-Up Project Request.** PI: Kushagra Bainsla (M.S. Computer Science, San Jose State University). Faculty advisor: Prof. Saptarshi Sengupta (Department of Computer Science, MICoSys Lab). Resource: TACC Horizon (Grace-Blackwell GB200) if it is in the catalog, otherwise TACC Vista (Grace-Hopper H200). Request: 500 GPU-hours (250 node-hours on Horizon), 300 GB storage, 3 months.

## A. Scientific/Technical Goal

**Objective.** LLM inference servers ship several implementations of the same operation (GEMM, attention, mixture-of-experts) and pick the fastest one for the workload. They are treated as interchangeable because they compute the same function, and recent evidence says they are not. Swapping CUTLASS for Triton INT8 kernels in vLLM produced zero sequence agreement in every test (0 of 8, 0 of 16, 0 of 64), traced to scale application and output rounding after an exact integer accumulator (arXiv 2608.13756). A tolerance check at one bfloat16 spacing cannot see that class of fault (arXiv 2609.00363). DriftBench (MLSys 2026) measured 236,985 prompt-response pairs across 105 configurations and blocked a real upgrade in which 23.85% of safety prompts flipped. Two open vLLM issues (#55238, #55649) report silent output changes on Blackwell GPUs. Decoding amplifies these differences: when the top two logits are close, one rounding step flips a token and changes the rest of the sequence.

**Gap.** The closest study covers one consumer GPU, one model family, INT8 plus one FP8 operator pair, and no mixture-of-experts, NVFP4 or multi-GPU runs. It says its results do not extend "across GPU generations, model families, or future kernels." Existing autotuners choose kernels on latency alone. It is unknown whether the fastest backend that satisfies an explicit output contract varies across serving regimes enough to matter, on the GPUs where block-scaled FP8 and NVFP4 are deployed.

**Research questions and decisions fixed in advance.** RQ1: how often does substituting one backend for another violate an explicit output contract (bitwise, token-flip, logit) for block-scaled FP8, NVFP4 and fused-MoE operators? RQ2: does the fastest contract-satisfying backend vary across batch size, prefill versus decode, and model enough to beat the best single fixed-order backend by at least 5% end to end? If the gain is at least 5% on two models, we build a correctness-aware selector. If it is under 5% but at least 10% of default substitutions violate the contract, we publish the measurement study. If both are under 2%, we report a documented negative result. These thresholds are project heuristics, written before any run. Every outcome is a deliverable.

**Method.** Exact greedy decoding, with decode-mode teacher forcing because prefill and decode select different kernels. Two open-weight model families: a dense Qwen3 model for comparability with prior work and an open mixture-of-experts model. At least three backends per operator family (CUTLASS, Triton, FlashInfer), BF16 as control, and block-scaled FP8 and NVFP4 from public checkpoints, at several batch sizes, with real captured activations and synthetic controls. Baselines are the fastest backend alone, one fixed-order safe backend, a per-shape allow-list, and the oracle. Statistics use simple random sampling with Wilson upper bounds on the flip rate. Performance metrics are time to first token, time per output token, throughput and tail latency.

**Focus areas and dissemination.** The project advances AI methods that improve trust and reproducibility, produces open-source tools, and trains a master's student. Month 1: port the harness, reproduce a documented divergence, decision gate at week 3. Month 2: full grid. Month 3: held-out validation and write-up, then release of the harness, run manifests and result bundles. Reproducible cases are reported upstream to vLLM and FlashInfer. A thesis and technical report follow within six months, and a workshop or conference submission after that.

## B. Estimate of Compute, Storage and Other Resources

**Why this resource.** NVFP4 runs only on Blackwell-class GPUs, and both open Blackwell issues above are on compute capability 10.x parts, which GB200 is. Prior work has no data there. Jobs use one GPU, or two GPUs inside one node for tensor-parallel runs. No multi-node runs are needed. If Horizon is not available, Vista's H200 nodes cover the BF16 and FP8 cells and NVFP4 becomes future work. We are separately requesting NSF ACCESS Explore time on NCSA Delta (Hopper). The two requests target different architectures and different cells.

**How the amount was determined.** We do not have a GPU locally, so this is a planning estimate, not measured GPU time. The work units come from a laptop pilot of the same harness (32 prompts, 96 tokens, five decode runs per configuration), scaled to about 100 configurations. Time assumptions are about 100 tokens per second for an 8B model at batch size one, and 15 minutes per end-to-end benchmark repetition including server start. We will replace the assumptions with measurements in the first two weeks and report actual use.

| Work package | Basis | GPU-hours |
|---|---|---|
| Port harness, reproduce one documented divergence | one-off | 15 |
| Operator-level kernel-pair sweeps (FP8, NVFP4, BF16 GEMM; fused MoE) | 150 pairs, about 8 min each | 20 |
| Model-level contract tests | 100 configurations, about 7 min each | 12 |
| End-to-end serving benchmarks | 100 configurations, 3 repetitions, 15 min each | 75 |
| Held-out prompt sets, reruns, failure investigation, profiling | allowance | 120 |
| Reserve for setup and debugging | allowance | 100 |
| **Total** | | **342** |

We request **500 GPU-hours**, about 45% above the estimate. **Storage: 300 GB**, made up of about 150 GB of model checkpoints across BF16, FP8 and NVFP4 for two model families, about 100 GB of captured activations and logits, and about 50 GB for containers, results and logs. **Other resources:** no API credits, no commercial models, no proprietary data.

## C. Support Needs

**C1. Collaboration and support.** Helpful, not essential: help with container-based installs (vLLM, FlashInfer, CUTLASS, Triton, PyTorch) on the Grace (aarch64) host, guidance on supported CUDA and driver versions, access to GPU performance counters for Nsight Compute (if unavailable we use timing-only analysis), and short-queue access for debugging. We estimate under 10 hours of staff support over the award.

**C2. Compliance and access restrictions.** None. No export-controlled code, no ITAR restrictions, no proprietary or sensitive data, no PHI. Only public open-weight models under their published licenses and public or synthetic text are used. There is no restriction on the region of compute.

## D. Team and Team Preparedness

**Team.** The PI is an M.S. Computer Science student at San Jose State University with experience building GPU checkpoint and restore systems for multi-GPU training and production backend systems. Prof. Saptarshi Sengupta leads the MICoSys Lab, which has held NSF ACCESS and NAIRR allocations, and supervises the project. Citizenship or residency status for federal facility access: [PI to fill in].

**Readiness.** We can begin within two weeks of award. Before requesting compute we built the evaluation harness and validated it on a laptop. Fifteen unit tests pass. A backend compared with itself is bitwise identical across repeated runs. Fault injection into an emulated INT8 GEMM epilogue is detected by the bitwise gate in all four injected faults, while a one-spacing tolerance check misses three. Teacher-forced flips reproduce free-running divergence on 32 of 32 prompts. An invariant (a flip needs a reference margin of at most twice the maximum logit difference) held with zero violations over 6,144 positions. The same pilot falsified a margin-stratified estimator we had designed, so the method uses simple random sampling. These are checks of the method on a 0.6B model, not results about GPU kernels. Decision rules, pinned software versions and the run-manifest format are written. Porting to the target GPU means installing the pinned software in a container and swapping the backend pair.
