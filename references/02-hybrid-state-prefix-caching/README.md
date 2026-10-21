# Hybrid state reuse, prefix caching, recovery

Source: `RESEARCH_JOURNAL.md` Section 7.2. Papers that only a dossier cites are added below the ledger papers.

Check levels are defined in journal Section 0.4. "dossier" or "journal" means the paper is not in the ledger, so it has no check level.

## Papers (9)

| Paper | Authors | Date | What we used it for | Check | Cited in |
|---|---|---|---|---|---|
| [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379) `2411.19379` | Rui Pan, Zhuang Wang, Zhen Jia et al. | 2024-11-28 | First prefix caching for hybrid LLMs, admission and eviction policies; Foundational | search-only | [Journal 7.2](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md) |
| [Sparse Prefix Caching for Hybrid and Recurrent LLM Serving](https://arxiv.org/abs/2605.05219) `2605.05219` | Mikhail Shirokikh, Sergey Nikolenko | 2026-04-17 | Sparse exact recurrent checkpoints, resume from deepest, O(NM) DP for placement. Exact outputs; Owns checkpoint placement | checked | [Journal 7.2](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md), [LIMITATIONS_R-36_2026-10-20](../../dossier/LIMITATIONS_R-36_2026-10-20.md), [RESEARCH_GAP_DOSSIER_2026-09-16](../../dossier/RESEARCH_GAP_DOSSIER_2026-09-16.md) |
| [Asymmetric Virtual Memory Paging for Hybrid Mamba-Transformer Inference](https://arxiv.org/abs/2605.22416) `2605.22416` | An Xuan Nguyen | 2026-05-21 | Position-independent caching for hybrids, asymmetric virtual memory paging; Context | search-only | [Journal 7.2](../../RESEARCH_JOURNAL.md) |
| [HYPIC: Accelerating Hybrid-Attention LLM Serving with Position-Independent Caching](https://arxiv.org/abs/2607.01299) `2607.01299` | Yifei Liu, Juntong Wu, Yang Liu et al. | 2026-07-01 | Position-independent caching for hybrid attention: segment-cumulative transition operator, seam window that propagates hidden states, 3.25x TTFT; Reaches the "propagate hidden states" insight independently | checked | [Journal 7.2](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md), [RESEARCH_GAP_DOSSIER_2026-09-16](../../dossier/RESEARCH_GAP_DOSSIER_2026-09-16.md) |
| [LinearKV: One Cached State Suffices for Position-Independent Caching in Hybrid LLMs](https://arxiv.org/abs/2608.11231) `2608.11231` | Yirui Liu, Ruoling Qi, Longwen Wang et al. | 2026-07-31 | Position-independent caching for hybrids, asymmetric virtual memory paging; Context | search-only | [Journal 7.2](../../RESEARCH_JOURNAL.md) |
| [Runtime Observability for Heterogeneous Attention Memory](https://arxiv.org/abs/2608.05863) `2608.05863` | Fanzhe Wei, Li Liu, Ziyang Wang et al. | 2026-08-06 | Formal contract, Lean 4, four cache types including recurrent, always-on probe under CUDA graphs, 12.4M entry reads; Direct threat to detector ideas | checked (abstract) | [Journal 5.6](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md) |
| [DASC: Decay-Aware State Compression for Hybrid Linear-Attention Serving](https://arxiv.org/abs/2608.30386) `2608.30386` | Yanqi Yu, Pingwei Sun, Jianchao Tan et al. | 2026-08-31 | Compresses KDA recurrent checkpoints 2.63x, TTFT down 42.6%, throughput up 68.4%; Adjacent to I-39 | checked | [Journal 5.11](../../RESEARCH_JOURNAL.md) |
| [Validating Hybrid-State Cache Recovery for GLM-5.3-Flash with vLLM and LMCache](https://arxiv.org/abs/2609.15030) `2609.15030` | Frank Li | 2026-09-14 | Recovery mismatch (scheduler credited one fewer token), generation agreement 34/36 to 36/36. No general condition or tool; Existing validation | checked | [Journal 5.6](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md) |
| [Shared KV Caching for Replicated 27B Inference: Correctness Failures and Performance Boundaries](https://arxiv.org/abs/2609.15021) `2609.15021` | Frank Li | 2026-09-14 | Missing CUDA stream dependency in raw-pointer fallback, found by controlled byte tests; Context | checked | [Journal 7.2](../../RESEARCH_JOURNAL.md) |

## Other sources without an arXiv ID (9)

GitHub issues, documentation, blog posts and papers that the ledger lists without an arXiv ID.

| Work | What it shows | Check |
|---|---|---|
| ReplaySSM (vLLM #49232, Dao AI Lab blog) | Cache SSM inputs, not state. 1.17x to 1.67x decode, up to 2.21x speculative; Orthogonal, kernel level | doc-only |
| SGLang Unified Radix Cache blog | FULL, SWA, MAMBA on one tree. Reusable boundary advances only when all component validators accept; Shipped reconciliation rule | checked |
| vLLM #53041 | Discussion. Proposal A tiered checkpoints. Proposal B recompute backfill. Zero comments at fetch; Prior claim of I-16, subject of I-19 | checked |
| vLLM #52773 | Load-backfill from external store, measured about 80x cheaper than recompute; Closes I-16 performance | doc-only |
| vLLM #57266, #55600, #56655 | Indexing, unit-mismatch and cache-key bugs (details in I-19); Different bug classes | checked |
| vLLM #55697, #52959 | Application-directed prefix checkpoints, internal Mamba checkpoints; Scoop risk for I-15 | doc-only |
| vLLM #45238, #45702, #48455, #53786, #54661, #47812, #49537, #51771, #55196, #54106, #53670, #53504 | Hybrid cache and recovery issues, see I-13; Evidence the area is live | doc-only |
| vLLM hybrid SSM disaggregation blog | NIXL dual descriptor views, 3-descriptor conv transfer, heterogeneous TP. Says nothing on live migration. GDN on roadmap; Kills I-06 novelty | checked |
| AI21 "One token to corrupt them all", PyTorch hybrid-models blog, Kimi K3 hybrid caching | Mamba corruption bug, hybrid support, hybrid cache semantics; Context | doc-only |
