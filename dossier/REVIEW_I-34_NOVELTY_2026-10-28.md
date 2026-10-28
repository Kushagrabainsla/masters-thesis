# R-53: Novelty and adversarial check of I-34 (2026-10-28)

Verdict (two passes; pass 1 Opus, pass 2 Sonnet with fresh searches): KILL as framed. Recommendation only (D-65).

## Question checked
Do NCCL's newer collective paths (NVLS, copy-engine collectives, symmetric kernels) have a reduction-order
and precision contract, and how does it change TP inference outputs?

## Source ledger
| Source | Date | What it shows | Threat |
|---|---|---|---|
| NVIDIA/nccl #2360 | 2026 | Hopper NVLS BF16 depends on multicast allocation history | Closed as Hopper-specific; R570 driver fix claimed |
| NVIDIA/nccl #2479, PR #2504 | Oct 2026 | Blackwell/B300 BF16/FP16 NVLS not repeatable when count % 4 != 0 (stale lanes); rounding lane-coupled, not round-to-nearest (community) | NVIDIA confirmed Oct 26; fix PR open |
| NVIDIA/Megatron-LM #8138, PR #8180 | Oct 2026 | Unpinned NCCL_ALGO in deterministic mode, 256 B200; same-plan repeatable; about 1.7% faster than Ring | Merged policy |
| pytorch/pytorch #201095 | Oct 2026 | Capability-based NVLS policy in deterministic mode | Open proposal |
| lightseekorg/tokenspeed PR #1978 | Oct 2026 | Fixed-issuer in-switch all-reduce; startup self-check of reduction groups | Shipping mechanism |
| vllm-project/vllm #51290 | Aug 2026 | Fused all-reduce plus RMSNorm broke TP batch invariance; fixed | Fixed |
| chayan-bit/invar #223 | Oct 2026 | Typed collective boundary with fixed order | Open design |
| PyTorch SymmetricMemory docs | 2026 | multimem one-shot all-reduce: reduction order not fixed | Documented |
| TBIK 2511.17826 (ICML 2026) | 2026 | TP-size-invariant fixed-tree matmul and all-reduce | Mechanism exists |
| LLM-42 2601.17768, Vosti 2609.38981, MarginGate 2605.30218, Kernel Contracts 2604.22032, 2609.11356, 2606.00279 | 2026 | Determinism by verification, formal specs, kernel contracts | Neighbours |
| NCCL 2.27 and 2.28 blogs | 2025 | Symmetric kernels FP32 accumulate; CE collectives have no reductions | CE path has no order problem |

## Attacks that kill it
- Solved or in progress upstream (NVIDIA fix, Megatron, PyTorch, TokenSpeed, vLLM).
- What is left is characterization ("which paths agree, how many tokens flip"), which the user's style rule excludes.
- Small oracle: a fixed-order path costs about 1.7% against Ring (Megatron, 256 B200).
- Unstable target: results change with each driver or NCCL release.

## Weak pivot (below the thesis bar)
Keep NVLS speed and exact tokens by re-running the all-reduce on a fixed path only for low-margin tokens
(MarginGate or LLM-42 style). Upside is bounded by about 1.7%. Not recommended.
