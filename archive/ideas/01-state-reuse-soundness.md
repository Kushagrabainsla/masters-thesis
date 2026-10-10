# Idea 1: When is it safe to reuse cached state per layer group?

Status: **GO**. Feasibility gate passed. Recommended thesis.
Full evidence: `../dossier/RESEARCH_GAP_DOSSIER_2026-09-16.md`, sections 4, 10, 11.

## The problem in plain terms

Hybrid LLMs mix layer types: full attention, sliding window, and recurrent or
linear attention. Serving engines cache each type's state separately, because the
types have different shapes and different lifetimes.

When a new request partly matches a cached one, each type can match to a different
depth. Full attention might match 128 tokens while recurrent state matches 0,
because recurrent state is one position-keyed snapshot and cannot be truncated to
an arbitrary prefix.

The engine then decides, per layer group, which state to reuse and which to
recompute. That decision can be wrong in a way that silently corrupts output.

## The gap

There is no stated rule for when a per-group reuse decision is safe.

vLLM stays safe through scattered one-off guards, with the reasoning living in a
docstring rather than an invariant. Bugs ship anyway:

- **#57266** `conv_ssm_forward` writes SSM state from an out-of-range strided slice
  when `num_computed_tokens` is not block-aligned, killing EngineCore.
- **#55600** using global block geometry for Mamba state produces wrong state or
  illegal memory access.
- **#56655** prefix-cache keys omit the prompt-embedding mask, so a request reuses
  K/V computed for a different effective input.
- **#47812** hybrid recovery after a KV-load failure carried single-group assumptions.
- AI21's public writeup "One token to corrupt them all" on a Mamba corruption bug.

The closest paper, "The Illusion of Equivalence" (arXiv 2604.15409), shows that
KV-cached inference diverges numerically from cache-free inference in FP16. It is
measurement only, standard attention only, no tool, and does not touch multi-group
reuse.

## Research question

> Can the safety condition for per-layer-group state reuse be stated, checked
> mechanically, and used to decide which reuse plans are worth implementing, and
> does applying it to shipped and proposed mechanisms reveal unsoundness that
> existing example-based tests miss?

Falsified if the condition turns out to be equivalent to something engines already
assert, or if applying it finds nothing current tests do not already catch.

## The condition

A reuse plan assigns each (layer, position) to REUSE or EXECUTE. Executing layer L
at position p needs the hidden state from layer L-1 at p. That exists only if L-1
also executes at p, or if L-1's output activation at p was cached. So:

> reuse depths must be non-decreasing along the layer stack, except where a cached
> activation covers a step down.

Interleaving is what makes this bite. A full-attention layer that reuses more than
the recurrent layer above it starves that layer.

For recurrent-state backfill specifically there is a simpler corollary. Backfill
only needs correct recurrent state at the boundary, and recurrent layer L depends
only on layers below it. So the skippable set is exactly the contiguous suffix of
the stack above the highest state-holding layer.

## Evidence already collected

| Check | Result |
|---|---|
| Static checker vs measured equivalence, 3 plans with ground truth | agrees 3 of 3 |
| Unsound plan, measured max logit diff | 1.0554 |
| Sound plans, measured max logit diff | 0.0117 (floating-point noise floor) |
| Suffix rule vs exhaustive subset search, 6 layouts | matches every time |
| Skipping the suffix on the real model, recurrent state | bit-identical, diff exactly 0 |
| vLLM #45238 reproduced on a MacBook, CPU only | per-group hits (128, 0); default path discards 128 valid tokens |

Applied to an active upstream design discussion, vLLM **#53041** Proposal B
("FA layers skip the gap range because their KV is pre-loaded; only SWA/Mamba
layers run forward"), the checker reports unsound and names the starved layers:
4, 8, 12, 16, 20 in Qwen3.5-0.8B.

## Why this is a systems result, not a testing result

The condition also prices the design space, which is the part that makes it useful
to designers rather than only to test engineers.

- Full-attention work is 23% of hybrid replay at 512 tokens and 88% at 128K, so the
  prize grows with context.
- But the only *sound* version of the proposed optimization avoids recomputing K/V
  projections, worth 1.22% at 512 tokens falling to 0.18% at 128K.
- Surveying 13 real hybrid models: zero have a layout where more than half the
  full-attention work is soundly skippable, and 8 of 13 have none at all.

So the condition tells you the entire class of "reuse full-attention state while
rebuilding recurrent state" mechanisms is not worth building, and why. That is
design-space pruning derived from a correctness argument.

## Plan

1. Write the condition and its proof. Extend past two groups to sliding window,
   speculative decode, DCP, and offload.
2. Build the checker into a plan-level tool against a pinned vLLM. Already works on
   CPU.
3. Apply it to shipped and proposed mechanisms. Report what it certifies and what
   it refutes.
4. Confirm a sample of verdicts dynamically on a rented GPU. The Mac already
   confirms the ones tested.
5. Land the upstream patch below as the constructive artifact.

## Constructive artifact that falls out

Skipping layers above the last recurrent layer during backfill is sound, needs no
extra memory, and is worth 1% to 6% measured on Apple MPS and up to about 15%
analytically at long context for Qwen-family layouts. vLLM's backfill path has no
notion of a per-request layer ceiling. Small, self-contained, verifiable, and it
shows the invariant doing useful work rather than only forbidding things.

## Feasibility and risk

Runs on the MacBook you own. vLLM's planning and coordinator layers work on CPU;
only execution-level confirmation needs a GPU, briefly and rentable.

Main risk: it reads as software testing rather than systems, which matters for
GPU-infrastructure hiring. The design-space pricing above is the mitigation, and it
should be foregrounded rather than treated as a side result.

Venues: ASPLOS, EuroSys, ATC. Fallback ISSTA or ICSE.

## Sources

- vLLM issues #53041, #52773, #57266, #55600, #56655, #47812, #45238, #45702, #55697, #52959
- Sparse Prefix Caching, https://arxiv.org/abs/2605.05219
- HYPIC, https://arxiv.org/abs/2607.01299
- The Illusion of Equivalence, https://arxiv.org/pdf/2604.15409
- Hybrid models in vLLM, https://pytorch.org/blog/hybrid-models-as-first-class-citizens-in-vllm/
- AI21, One token to corrupt them all, https://www.ai21.com/blog/vllm-debugging-mamba-bug/
