# Idea 2: Pick the collective's execution engine from live compute pressure

Status: **MEASURE-FIRST**. Gap confirmed from source. Blocked on NVLink hardware.
Full evidence: `../dossier/RESEARCH_GAP_DOSSIER_2026-09-16.md`, sections 5 and 10.4.

## The problem in plain terms

A GPU can move data for a collective two ways. It can run a kernel on the SMs, or
it can hand the transfer to the copy engines. NCCL 2.28 added copy-engine
collectives, which reach zero-SM operation for AllGather and AllToAll inside an
NVLink domain.

The two paths consume different resources. An SM-driven collective takes SMs away
from the compute kernel running alongside it. A copy-engine collective leaves SMs
alone but adds pressure on memory bandwidth.

So the better choice depends on what the concurrent compute kernel is bottlenecked
on. A compute-bound GEMM wants its SMs back, and should prefer the copy engine. A
bandwidth-bound kernel is already saturating memory, and may prefer the SM path.

## The gap

NCCL cannot make that choice, because the information never reaches the decision
point.

Audited at NCCL **2.31.2**, commit `fd168324a3`:

1. The whole copy-engine versus SM routing decision is one line, at
   `src/enqueue/enqueue.cc:3305`:
   ```c
   if ((info->collConfig.CTAPolicy & NCCL_CTA_POLICY_ZERO) && (ceAvailable || hierCeAvailable))
   ```
   A static user-set policy, AND a static eligibility predicate. No cost model.
2. `ncclCeAvailable`'s five rejection reasons are all static capability or
   registration properties: collective not implemented for CE, host-backed cuMem
   segments, incomplete NVLink connectivity, symmetric support disabled,
   unsupported window registration type. Not one is a load condition.
3. The tuner plugin is the only extension point for algorithm choice, and it is
   **not consulted for CE routing at all**. Even if it were, `getCollInfo` in the
   latest interface receives only `context, collType, nBytes, numPipeOps,
   collCostTable, numAlgo, numProto, regBuff, nChannels`. No occupancy, no
   bandwidth pressure, no handle to the concurrent kernel.
4. Searching all of `src/` for `occupancy`, `cudaOccupancy`, `smUtil`,
   `GetUtilization`, `activeWarps`, `nvmlDeviceGetUtilizationRates`, `smActive`
   and `achieved_occupancy` returns **zero matches**. The runtime never measures
   the quantity the policy would key on.

The user-facing control is `NCCL_CTA_POLICY` set to DEFAULT, EFFICIENCY or ZERO,
resolved once per communicator or per call.

## Independent confirmation that this is open

A comprehensive August 2026 survey (arXiv 2608.15118) names the underlying problem
in its own words:

> "A major architectural limitation is their heavy reliance on GPU computing
> resources (such as GPU Streaming Multiprocessors and device memory bandwidth) to
> drive network progress."

Its adaptive-execution section covers MSCCL++, AutoCCL and MCCS. Its
computation-communication coordination section covers CoCoNet, Centauri and
NanoFlow. None selects the execution engine from live compute-kernel resource
state.

So the novelty claim is structural rather than a literature absence: the input does
not exist at the decision point, the only extension point cannot see it, and the
runtime does not collect it anywhere.

## Research question

> Can selecting the collective execution engine, SM-driven versus copy-engine, from
> the live resource profile of the concurrently executing compute kernel improve
> end-to-end overlapped throughput over both NCCL's default selection and the best
> static `NCCL_CTA_POLICY`, while preserving ordering and progress semantics?

Falsified if a static per-phase or per-layer policy captures most of the gain, or
if the perfect-information gain over NCCL's default is small.

## What is unknown

Everything about magnitude. The gap is proven; its value is not. A static policy
chosen per model or per training phase may well capture most of the benefit, and
that is the single biggest threat to this idea.

## Predeclared decisive experiment

Written down now so that hardware access turns into a result in about a week, and
so the thresholds cannot be chosen after seeing the data.

1. Two controlled compute kernels: one SM-bound (large GEMM at high occupancy), one
   memory-bandwidth-bound (streaming triad). Confirm the intended bottleneck with
   profiler counters rather than assuming it.
2. Overlap each with AllGather and AllToAll under `NCCL_CTA_POLICY` DEFAULT,
   EFFICIENCY and ZERO. Sweep message size and channel count.
3. Report **end-to-end overlapped throughput**, not collective time alone. A copy
   engine path that is slower in isolation can still win by returning SMs to compute.
4. Oracle: the best static policy per (kernel, message size) cell chosen with
   perfect hindsight. This is the number that decides the idea.
5. GO if the per-cell oracle beats the best single global policy by a margin that
   survives switching cost. KILL if one static policy sits within noise of the
   oracle across the grid. These are heuristics, not literature standards, and they
   are fixed before the run.

## Hardware requirement

Copy-engine collectives are NVLink-domain, so this needs at least two
NVLink-connected GPUs and realistically an 8-GPU H100 or H200 node. There is no
such access at present. This is a hard gate: do not select this as the thesis until
the hardware is confirmed rather than promised.

Ask list: SJSU advisor and research cluster, LinkedIn infrastructure contacts,
NVIDIA Academic credits, cloud research credit programs. Frame it as a short block
for a specific microbenchmark, not open-ended access.

## Work possible without hardware, already done

The source audit above. It is reproducible with only a clone:

```
git clone --depth 1 https://github.com/NVIDIA/nccl.git /tmp/nccl
python3 ../experiments/rq2_nccl_source_audit/audit_nccl_ce_selection.py /tmp/nccl
```

## Why keep it warm

Best novelty of the surviving ideas, best fit for GPU-infrastructure roles, and the
hard part of the argument is already established. If the node appears, switch to
this.

## Sources

- NCCL source, https://github.com/NVIDIA/nccl (audited at 2.31.2, `fd168324a3`)
- NCCL 2.28 CE collectives and device API, https://developer.nvidia.com/blog/fusing-communication-and-compute-with-new-device-api-and-copy-engine-collectives-in-nvidia-nccl-2-28/
- NCCL environment variables, https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
- Collective communication survey, https://arxiv.org/abs/2608.15118
