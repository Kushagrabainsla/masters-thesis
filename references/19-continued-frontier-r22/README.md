# Continued frontier systems review (added 2026-10-17, R-22)

Source: `RESEARCH_JOURNAL.md` Section 7.17. Papers that only a dossier cites are added below the ledger papers.

Check levels are defined in journal Section 0.4. "dossier" or "journal" means the paper is not in the ledger, so it has no check level.

## Papers (4)

| Paper | Authors | Date | What we used it for | Check | Cited in |
|---|---|---|---|---|---|
| [M2K: Making the Model-Kernel Interface Explicit for Reliable CUDA Kernel Verification](https://arxiv.org/abs/2603.24595) `2603.24595` | Mengting He, Shihao Xia, Haomin Jia et al. | 2026-03-06 | Uses model-aware symbolic execution to find CUDA memory bugs; Kills I-127 | checked | [Journal 7.17](../../RESEARCH_JOURNAL.md), [FRONTIER_R-22_2026-10-17](../../dossier/FRONTIER_R-22_2026-10-17.md) |
| [Execution-State Capsules: Graph-Bound Execution-State Checkpoint and Restore for Low-Latency, Small-Batch, On-Device Physical-AI Serving](https://arxiv.org/abs/2606.20537) `2606.20537` | Liang Su | 2026-06-18 | Captures KV, recurrent, convolution, MTP, and metadata state; evaluates one-device restore; I-123 mechanism and threat | checked | [Journal 7.17](../../RESEARCH_JOURNAL.md), [FRONTIER_R-22_2026-10-17](../../dossier/FRONTIER_R-22_2026-10-17.md) |
| [Serving Masked Diffusion LLMs: Characterization and Design Principles from Real Hardware](https://arxiv.org/abs/2608.23807) `2608.23807` | Farhana Amin, Sabiha Afroz, Mona Moghadampanah et al. | 2026-08-24 | Reports weak completion prediction and proposes slot batching; Kills I-125 | checked | [Journal 7.17](../../RESEARCH_JOURNAL.md), [FRONTIER_R-22_2026-10-17](../../dossier/FRONTIER_R-22_2026-10-17.md), [R-18_ORPHAN_FINDINGS_PAPERS_2026-10-08](../../dossier/R-18_ORPHAN_FINDINGS_PAPERS_2026-10-08.md) |
| [Just Let Linear States Forget the Distant Past: Prefix Caching via Suffix Replay for Hybrid LLMs](https://arxiv.org/abs/2609.33477) `2609.33477` | Yirui Liu, Ruoling Qi, Xuaner Wu et al. | 2026-09-27 | Reuses and reconstructs recurrent suffix state; Kills I-129 route | checked | [Journal 7.17](../../RESEARCH_JOURNAL.md), [FRONTIER_R-22_2026-10-17](../../dossier/FRONTIER_R-22_2026-10-17.md) |

## Other sources without an arXiv ID (10)

GitHub issues, documentation, blog posts and papers that the ledger lists without an arXiv ID.

| Work | What it shows | Check |
|---|---|---|
| vLLM streaming requests and Realtime API | Live requests can receive new input and resume generation; I-122 transition | checked |
| vLLM issue 49449 | Session rebuild changes tokens but retains stale block hashes, causing deterministic wrong output; Direct I-122 evidence | full-text (issue) |
| vLLM issue 58776 | Preemption exposes a mismatch between valid token history and materialized KV progress; Direct I-122 evidence | full-text (issue) |
| vLLM issue 36755 | Async output reads request state after a later schedule step mutates it; Direct I-122 evidence | full-text (issue) |
| vLLM issue 42655 | Repeated streaming updates leave stale batch-row identity; Direct I-122 evidence | full-text (issue) |
| vLLM-Omni RFC 5212 | Streaming ASR keeps frontend, convolution, acoustic, recurrent, and output state; token replay is insufficient; Direct I-123 evidence | full-text (issue) |
| vLLM-Omni RFC 6227 | Real-time world-model migration, recovery, and offload remain later work; I-123 roadmap and scoop threat | full-text (issue) |
| vLLM-Omni diffusion continuous batching | `StepScheduler` advances requests independently with separate progress and cancellation; Kills I-125 | checked |
| Mosaic Streaming elastic determinism | Supports deterministic mid-epoch resumption across changed GPU counts; Kills I-128 | checked |
| PyTorch compile caching tutorial | Exports portable compiler artifacts through Mega-Cache; Kills I-130 route | checked |
