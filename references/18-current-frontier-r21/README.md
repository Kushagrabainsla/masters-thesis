# Current-frontier systems review (added 2026-10-17, R-21)

Source: `RESEARCH_JOURNAL.md` Section 7.16. Papers that only a dossier cites are added below the ledger papers.

Check levels are defined in journal Section 0.4. "dossier" or "journal" means the paper is not in the ledger, so it has no check level.

## Papers (12)

| Paper | Authors | Date | What we used it for | Check | Cited in |
|---|---|---|---|---|---|
| [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665) `2510.09665` | Yuhan Liu, Yihua Cheng, Jiayi Yao et al. | 2025-10-08 | Shared KV storage and transfer integrated with vLLM; I-120 context | checked | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [FlashInfer-Bench: Building the Virtuous Cycle for AI-driven LLM Systems](https://arxiv.org/abs/2601.00227) `2601.00227` | Shanli Xing, Yiyan Zhai, Alexander Jiang et al. | 2026-01-01 | Kernel traces, correctness tests, speed tests, and runtime substitution; High threat to I-23 | full-text (relevant sections) | [Journal 5.18](../../RESEARCH_JOURNAL.md), [ASSUMPTIONS_R-19_2026-10-11](../../dossier/ASSUMPTIONS_R-19_2026-10-11.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [Atomix: Timely, Transactional Tool Use for Reliable Agentic Workflows](https://arxiv.org/abs/2602.14849) `2602.14849` | Bardia Mohammadi, Nearchos Potamitis, Lars Klein et al. | 2026-02-16 | Epoch-tagged transactions and progress frontiers for tool effects; Kills broad external-effect gap | checked | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [MARS: Efficient, Adaptive Co-Scheduling for Heterogeneous Agentic Systems](https://arxiv.org/abs/2604.26963) `2604.26963` | Yifei Wang, Hancheng Ye, Yechen Xu et al. | 2026-04-14 | Joint scheduling of GPU inference, CPU tools, and KV retention; Kills broad CPU-GPU agent scheduling gap | full-text (relevant sections) | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [Crab: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes](https://arxiv.org/abs/2604.28138) `2604.28138` | Tianyuan Wu, Chaokun Chang, Lunxi Cao et al. | 2026-04-30 | Selective OS-state checkpoints for agents; Kills broad agent checkpoint gap | checked | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters](https://arxiv.org/abs/2605.00528) `2605.00528` | Dongxin Guo, Jikun Wu, Siu Ming Yiu | 2026-05-01 | Workflow-atomic scheduling with session affinity and work stealing; Kills workflow-as-unit gap | full-text (relevant sections) | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md), [REVIEW_R-17_SHORTLIST_2026-10-08](../../dossier/REVIEW_R-17_SHORTLIST_2026-10-08.md) |
| [Fail-Closed Lowering of Resident KV Claims onto LLM Serving Runtimes](https://arxiv.org/abs/2606.01387) `2606.01387` | Lukas Stepanek | 2026-05-31 | Claim identity, materialization, and claim-scoped recovery failure; High I-120 threat | full-text | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [KernelBench-Verified: Do LLM-Generated Kernels Actually Beat PyTorch?](https://arxiv.org/abs/2607.16241) `2607.16241` | Yunxiang Zhang, Ping Yu, Jianyu Wang et al. | 2026-06-26 | Stronger tests and realistic baselines reverse a speed claim; Supports I-23 correctness gate | full-text (relevant sections) | [Journal 7.16](../../RESEARCH_JOURNAL.md), [ASSUMPTIONS_R-19_2026-10-11](../../dossier/ASSUMPTIONS_R-19_2026-10-11.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [SpecBox: Speculative Sandbox Scheduling for Efficient LLM Agent Serving](https://arxiv.org/abs/2607.23933) `2607.23933` | Yihui Zhang, Tianyu Wo, Jinghao Wang et al. | 2026-07-27 | Tool-intent prediction and sandbox prewarming during generation; Kills sandbox-prewarming gap | checked | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md), [R-17_ROUND3_SONNET_CANDIDATES_2026-10-08](../../dossier/R-17_ROUND3_SONNET_CANDIDATES_2026-10-08.md) |
| [Resume Means Resume: A Machine-Checked Conformance Contract for Checkpoint, Interrupt, and Resume Semantics in Workflow Persistence Layers](https://arxiv.org/abs/2608.03836) `2608.03836` | Sajjad Khan | 2026-08-04 | Workflow persistence semantics and fault conformance; Kills broad resume-contract gap | full-text (relevant sections) | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |
| [KVShareArena: KV-Cache Reuse Across Contexts and Model Checkpoints](https://arxiv.org/abs/2609.10266) `2609.10266` | Xi Shi, Mengxin Zheng, Qian Lou | 2026-09-09 | Measures reuse across contexts and checkpoints; Very high I-106 threat | checked | [Journal 5.18](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md), [REVIEW_R-19_SHORTLIST_2026-10-11](../../dossier/REVIEW_R-19_SHORTLIST_2026-10-11.md) |
| [TestGRAD: Evolving Test Suites via Failure Pattern Momentum for SWE-Agent Ensemble](https://arxiv.org/abs/2610.10242) `2610.10242` | Pengfei He, Jiayuan Zhou, Shaowei Wang et al. | 2026-10-07 | Iterative tests distinguish four agent-generated patches; Kills I-121 mechanism | full-text (relevant sections) | [Journal 7.16](../../RESEARCH_JOURNAL.md), [FRONTIER_R-21_2026-10-17](../../dossier/FRONTIER_R-21_2026-10-17.md) |

## Other sources without an arXiv ID (8)

GitHub issues, documentation, blog posts and papers that the ledger lists without an arXiv ID.

| Work | What it shows | Check |
|---|---|---|
| Chen and Stratton, Agents of Change | 300 million work events across 718 firms; output rose while review work increased; Supports verification bottleneck only | full-text |
| SWE-bench correctness study, ICSE 2026 | Plausible patches fail full developer tests and differ from oracle behavior; Supports weak-verification premise | full-text |
| Microsoft data-driven test selection | Production test selection saves compute while retaining bug detection; Strong I-121 baseline | checked |
| FlashInfer v0.7 | Serving-mode-aware autotuning with persistent configuration results; High threat to I-23 | checked |
| vLLM training-inference consistency stack | Fixed kernel choices preserve one reduction order; High threat to I-23 | checked |
| Mooncake ToS paper | Metadata recovery and an earlier active-block corruption risk; I-120 evidence | full-text (relevant sections) |
| Mooncake RFC 3692 | Manifest commit and activation plan; runtime paths remain incomplete; Very high I-120 threat | full-text (issue) |
| vLLM issue 49250 | Deterministic wrong output after a rejected shared KV load; I-120 failure evidence | full-text (issue) |
