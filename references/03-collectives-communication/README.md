# Collectives and compute-communication overlap

Source: `RESEARCH_JOURNAL.md` Section 7.3. Papers that only a dossier cites are added below the ledger papers.

Check levels are defined in journal Section 0.4. "dossier" or "journal" means the paper is not in the ledger, so it has no check level.

## Papers (11)

| Paper | Authors | Date | What we used it for | Check | Cited in |
|---|---|---|---|---|---|
| [FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion](https://arxiv.org/abs/2406.06858) `2406.06858` | Li-Wen Chang, Wenlei Bao, Qi Hou et al. | 2024-06-11 | Overlap and fusion; Crowds I-08 and I-18 | search-only | [Journal 7.3](../../RESEARCH_JOURNAL.md) |
| [TokenWeave: Efficient Compute-Communication Overlap for Distributed LLM Inference](https://arxiv.org/abs/2505.11329) `2505.11329` | Raja Gond, Nipun Kwatra, Ramachandran Ramjee | 2025-05-16 | Overlap and fusion; Crowds I-08 and I-18 | search-only | [Journal 7.3](../../RESEARCH_JOURNAL.md) |
| [Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786) `2507.04786` | Zhiyi Hu, Siyuan Shen, Tommaso Bonato et al. | 2025-07-07 | Protocols and algorithms analysis; Context | search-only | [Journal 7.3](../../RESEARCH_JOURNAL.md) |
| [ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels](https://arxiv.org/abs/2511.13940) `2511.13940` | Stuart H. Sul, Simran Arora, Benjamin F. Spector et al. | 2025-11-17 | Analysis of transfer mechanisms and overheads. Search snippet says CE suits coarse contiguous moves; Context. No CE numbers obtained | checked (abstract) | [Journal 7.3](../../RESEARCH_JOURNAL.md) |
| [Syncopate: Efficient Multi-GPU AI Kernels via Automatic Chunk-Centric Compute-Communication Overlap](https://arxiv.org/abs/2601.20595) `2601.20595` | Xinwei Qiang, Yue Guan, Zhengding Hu et al. | 2026-01-28 | Overlap and fusion; Crowds I-08 and I-18 | search-only | [Journal 7.3](../../RESEARCH_JOURNAL.md) |
| [Resource-aware Computation-Communication Overlap for multi-GPU ML Workloads](https://arxiv.org/abs/2606.09200) `2606.09200` | Minyu Cui, Miquel Pericas | 2026-06-08 | Occupancy shaping, up to 25.5% lower total time, A40, A100, H100, MI250X, no vendor library change; SM budget is the tunable | checked | [Journal 5.5](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md) |
| [Fine-grained Computation-Communication Overlap via Tile-level Signaling and Scheduling for Mixture-of-Experts](https://arxiv.org/abs/2607.19539) `2607.19539` | Minyu Cui, Anna Wingkvist, Morgan Ericsson | 2026-07-21 | Dynamic SM scheduling and fused overlap; Field moved to fusion | search-only | [Journal 7.3](../../RESEARCH_JOURNAL.md) |
| [Collective Communication for Distributed LLM Systems: Planning, Runtime Adaptation, and Computation Coordination](https://arxiv.org/abs/2608.15118) `2608.15118` | Xuebin Song, Menghao Zhang, Yuezheng Liu et al. | 2026-08-15 | Quoted sentence about GPU-resource reliance is in the "Communication Offloading" paragraph, which also says CE offload is "already practical"; Overstated in earlier docs | full-text | [Journal 5.5](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md), [RESEARCH_GAP_DOSSIER_2026-09-16](../../dossier/RESEARCH_GAP_DOSSIER_2026-09-16.md) |
| [Analytical Resource Management for Fine-grained MoE Computation-Communication Overlap](https://arxiv.org/abs/2609.07536) `2609.07536` | Hongyu Liu, Minyu Cui, Miquel Pericas | 2026-09-07 | Picks communication CTA count and SM partition analytically, A100; Same | checked | [Journal 5.5](../../RESEARCH_JOURNAL.md), [ADVERSARIAL_REVIEW_2026-09-23](../../dossier/ADVERSARIAL_REVIEW_2026-09-23.md) |
| [mKernel: Fast Multi-GPU, Multi-Node Fused Kernels](https://arxiv.org/abs/2609.13585) `2609.13585` | Ziming Mao, Yihan Zhang, Shawn Wei Chew et al. | 2026-09-11 | Dynamic SM scheduling and fused overlap; Field moved to fusion | search-only | [Journal 7.3](../../RESEARCH_JOURNAL.md) |
| [Weave: Fine-Grained Dynamic SM Scheduling in an MoE Megakernel for Compute-Communication Overlap](https://arxiv.org/abs/2609.21483) `2609.21483` | Ziyu Huang, Yangjie Zhou, Chenhao Zhu et al. | 2026-09-18 | Dynamic SM scheduling and fused overlap; Field moved to fusion | search-only | [Journal 7.3](../../RESEARCH_JOURNAL.md) |

## Other sources without an arXiv ID (5)

GitHub issues, documentation, blog posts and papers that the ledger lists without an arXiv ID.

| Work | What it shows | Check |
|---|---|---|
| NCCL source v2.32.3 (12df1a1) | CE gate, per-call `ncclCollConfig_t`, tuner v6, no load sensing; Core evidence for I-18 | full-text (relevant files) |
| NCCL 2.28 CE and device API blog | CE multicast AllGather about 780 vs 620 GB/s at 4 GB on 8 Blackwell GPUs. Selection path unstated; CE ahead in NVIDIA data | checked |
| NCCL Q2 2026 roadmap #2090 | Zero-SM AllGather, SM-initiated CE collectives explored; Scoop risk | checked |
| NCCL #2360 | Hopper NVLS BF16 AllReduce results vary with multicast allocation history and across H800 hosts; Evidence for I-34 | checked |
| REACT, PIPEMORPH, MSCCL++, AutoCCL, MCCS, CoCoNet, Centauri, NanoFlow | Adaptation and coordination systems. The last six names are confirmed present in the survey text; Context | doc-only |
