# GPU-plan freshness check (added 2026-10-20, R-35)

Source: `RESEARCH_JOURNAL.md` Section 7.23. Papers that only a dossier cites are added below the ledger papers.

Check levels are defined in journal Section 0.4. "dossier" or "journal" means the paper is not in the ledger, so it has no check level.

## Papers (2)

| Paper | Authors | Date | What we used it for | Check | Cited in |
|---|---|---|---|---|---|
| [LiteTopK: Exploiting the Curse of Dimensionality for a Fused Indexer-TopK Kernel in Long-Context Sparse Attention](https://arxiv.org/abs/2607.11976) `2607.11976` | Ziqi Yin, Jianyang Gao, Peiqi Yin et al. | 2026-07-13 | Fused exact indexer top-k; indexer scores concentrate in a narrow range; Weakens I-83 | checked (abstract) | [Journal 1.1](../../RESEARCH_JOURNAL.md), [GPU_RUNBOOK_2026-10-20](../../dossier/GPU_RUNBOOK_2026-10-20.md) |
| [Sample-Guided Exact Top-K Selection for Long-Context Sparse Attention](https://arxiv.org/abs/2609.08450) `2609.08450` | Siran Liu, Yang Xue, Theo Tang et al. | 2026-09-08 | Sample-guided exact top-k for indexer score rows; I-83 context | checked (abstract) | [Journal 5.16](../../RESEARCH_JOURNAL.md) |

## Other sources without an arXiv ID (5)

GitHub issues, documentation, blog posts and papers that the ledger lists without an arXiv ID.

| Work | What it shows | Check |
|---|---|---|
| vLLM issues 61768, 61905, 61912; PR 61769 | KV-load failure reports orphaned by abort, block reuse or preemption; found by CPU scheduler probes with a mock connector; Supports the I-120 kill | full-text (issues) |
| vLLM RFC 35780 (closed as stale) and PR 42461 (open) | Maintainers propose request-level KV-load error handling: reset to 0 and recompute; Simple baseline for I-120 | full-text (RFC and PR body) |
| vLLM issue 49250, PRs 49252 (closed), 53298, 59793, 45146 | Open fixes for recompute after a rejected load; I-120 context | checked (GitHub API) |
| vLLM v0.32.0 kernel config | `--linear-backend`, `--moe-backend`, `linear_backend_per_quant`, `--attention-backend`, `sparse_indexer_topk_backend`; Hook for I-23; I-83 context | full-text (source) |
| Lambda MIG guide; GPU price lists | `sudo nvidia-smi` on a 1 x GH200; H100 and B200 hourly prices; Session planning | search-only |
