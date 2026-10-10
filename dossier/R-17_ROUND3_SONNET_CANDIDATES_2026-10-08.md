# R-17 ROUND 3 CANDIDATES (Sonnet, 2026-10-08)

Role: systems researcher, hostile to my own ideas.
Scope: areas 1 to 5 of the R-17 brief. Serving-side topics and the already killed list are excluded.

## Labels
- FACT: I opened the source (arXiv abstract page, docs page, repo page) and read the line.
- SNIPPET: I saw the claim only in a search result. I did not open the page. Treat it as unverified.
- INTERPRETATION: my own reasoning. No source shows it.

## Summary
- I screened 28 candidates and 2 survive. Both survivors are weak.
- The area is crowded. In 2026 most direct transfers have a preprint or a shipped feature.
- S1 (checkpoint-aware limited preemption) is the better survivor. It has a clean oracle bound and a Mac premise check.
- S2 (adaptive slow-versus-hang detection) is a spare. It has no public data for a Mac premise check.
- There is no third survivor. I did not force one.
- I did not read full papers. Most kills rest on abstracts or search snippets. Open the PDFs before you cite a kill.

## Raw candidates and verdicts

Format: ID, candidate, mechanism source field, verdict, killing source.

### Area 1: training and post-training

| ID | Candidate | From | Verdict | Killing source |
|----|-----------|------|---------|----------------|
| C01 | Dynamic trainer/rollout GPU split by a feedback controller | control theory | KILL | FACT: DynaResize arXiv 2607.22614 (abstract: switches GPUs between rollout and training, 66.5% over best static). SNIPPET: Libra 2606.03077, BiDiRL 2607.09207 |
| C02 | RCU or epoch-based weight versions, so in-flight rollouts finish on old weights | OS (RCU) | KILL | SNIPPET: Olmo 3 (2512.13961) updates without pause, PipelineRL (2509.19128) in-flight updates, Motif 3 (2608.09119) version tags plus importance correction. INTERPRETATION: mixed-policy rollouts are accepted practice, so a grace period buys little |
| C03 | Read-once and broadcast for checkpoint load | HPC collective I/O | KILL | SNIPPET: Orbax arXiv 2605.23066 (load on one slice, broadcast; reports 2.6x to 6.8x). SNIPPET: Megatron-Core TwoStage and FullyParallel load strategies |
| C04 | Content-defined chunking dedup of GPU checkpoint images | storage (FastCDC, restic) | KILL | SNIPPET: ORNL GPU Merkle-tree incremental checkpointing, Shredder (FAST 2012). INTERPRETATION: Adam changes every parameter each step, so dedup of training state finds little |
| C05 | Prefix-shared forward/backward for GRPO groups | compilers, DB (shared scans) | KILL | SNIPPET: Prefix Grouper 2506.05433, AREAL-DTA 2602.00482, NeMo Automodel prefix_tree |
| C06 | Tiered or background compile to cut restart time | JIT compilers | KILL | SNIPPET: PyTorch Mega-Cache (save/load cache artifacts), BootSeer 2507.12619 (startup analysis) |
| C07 | In-process restart that keeps GPU state | OS (warm reboot) | KILL | SNIPPET: NVRx in-process restart, ByteRobust 2509.16293, PHOENIX 2607.01646 |
| C08 | Rollout trajectory trie compression between rollout and trainer | databases | KILL | INTERPRETATION only, not searched: a token is about 4 bytes, so trajectory ids are small next to logprob tensors and weights |

### Area 2: collective communication

| ID | Candidate | From | Verdict | Killing source |
|----|-----------|------|---------|----------------|
| C09 | Online ring rebuild or reroute around a degraded link or NIC | networking (reroute) | KILL | FACT: R2CCL arXiv 2512.25059 (abstract: lossless failover, under 1% training overhead). SNIPPET: OptCC 2606.01680, Concordia 2606.23521 |
| C10 | Bruck log-step all-to-all for small messages on NVLink | HPC (MPI) | KILL | SNIPPET: Sewell et al. 2024 found no Bruck gain on multi-GPU. FACT: ReTri 2605.26930 targets optical networks only, simulated, so it does not cover NVL72 |
| C11 | Checksum (ABFT) check on all-reduce output | HPC (Huang-Abraham) | KILL | INTERPRETATION: a rigorous floating-point sum bound for 72 bf16 ranks is about n times 2^-9 of the sum of magnitudes. That is too loose to catch anything a gradient-jump check does not catch. SNIPPET: 2604.00726 (gradient-jump detector), 2605.27605 (replica hash), AEGIS OSDI 2026. Same family as the killed V-ABFT |
| C12 | Compensated or stochastic-rounded per-hop reduce in bf16 | numerics (Kahan, SR) | KILL | INTERPRETATION: reduction rounding error is far below gradient noise, and fp32 reduce dtype is a flag in FSDP and Megatron. SNIPPET: 2502.20566 (SR for LLM training) |
| C13 | Error-free transformation in FP8 reduce-scatter | numerics | KILL | SNIPPET: EQuARX 2506.17615 and torchft PR 194 already dequantize, reduce in high precision, requantize |
| C14 | NVLS multicast for RL weight broadcast inside a rack | in-network multicast | KILL | INTERPRETATION only, not searched: a pipelined broadcast already sends each byte once from the source |

### Area 3: numerics

| ID | Candidate | From | Verdict | Killing source |
|----|-----------|------|---------|----------------|
| C15 | Herbie-style accuracy rewriting for fused low-precision epilogues | compilers (Herbie, e-graphs) | KILL | INTERPRETATION: epilogue arithmetic hides behind memory traffic, so no speed gain shows. SNIPPET: FlashAttention-4 (2603.05451) already uses Sollya polynomial exp2 where the arithmetic matters |
| C16 | Static range analysis to remove FP8 amax passes | compilers (abstract interpretation) | KILL | INTERPRETATION only, not searched: microscaling computes block amax in the epilogue, and unit scaling covers the rest |

### Area 4: agent infrastructure

| ID | Candidate | From | Verdict | Killing source |
|----|-----------|------|---------|----------------|
| C17 | Resource governor for many agents on one machine (jobserver tokens, PSI, cgroup) | build systems, OS | KILL | FACT: AgentCgroup arXiv 2602.09345 (eBPF cgroup control per tool call; memory is the bottleneck). SNIPPET: Gentoo steve and guildmaster system jobservers |
| C18 | Shared MCP server process (zygote or multiplexer) | OS (Android zygote) | KILL | SNIPPET: rmcp-mux, mcpd, mcpmu already ship |
| C19 | Record-and-prefetch snapshot restore for agent sandboxes | serverless (REAP, FaaSnap) | KILL | SNIPPET: Sabre OSDI 2024, Wrenn UFFD lazy restore |
| C20 | Cross-sandbox memory dedup (Medes, KSM) | virtualization | KILL | INTERPRETATION: shared read-only rootfs with DAX removes the duplicate page cache. Not searched in depth |
| C21 | Idle sandbox memory offload during LLM calls | OS (swapping) | KILL | SNIPPET: 2608.15127 (4.6x lower average memory in a prototype), SpecBox 2607.23933 |
| C22 | Saga and compensation for tool calls | databases | KILL | SNIPPET: SagaLLM (VLDB 2025), Restate rollback docs |
| C23 | Durable-execution log compaction | event sourcing | KILL | SNIPPET: Golem, Pixir ADR 0018, Stately agent checkpoints |
| C24 | Group commit for human approvals | databases | KILL | SNIPPET: approval-fatigue guides. The problem is human factors, not a system mechanism |
| C25 | CRDT workspace for concurrent coding agents | distributed systems | KILL | SNIPPET: AgentRoom 2608.23740, CodeCRDT 2510.18893 |

### Area 5: cluster operations

| ID | Candidate | From | Verdict | Killing source |
|----|-----------|------|---------|----------------|
| C26 | Lazy or prefetched GPU restore | OS (demand paging) | KILL | SNIPPET: PhoenixOS 2405.12079 (on-demand restore), CRIU-LZ4 (EuroMLSys 2026), MSched |
| C27 | Idle GPU notebook swap-out | OS (swapping) | KILL | SNIPPET: NotebookOS 2503.20591, gpusched |
| C28 | Buddy allocator for NVL72 domains | OS (buddy allocator) | KILL | INTERPRETATION only, not searched: NVSwitch is non-blocking inside the rack, so there is no placement locality to protect |

### Survivors (see below)
- S1 (candidate C29): limited-preemption victim selection for checkpointing training jobs.
- S2 (candidate C30): phi-accrual style slow-versus-hang detection for collectives.

## S1. Checkpoint-aware limited preemption for GPU training queues

### Mechanism and environment
- Mechanism: limited-preemption scheduling from real-time systems. A task is preempted only at defined preemption points. The scheduler picks the task whose next point is closest.
- Environment: a training job has natural preemption points. They are its checkpoint boundaries. The scheduler can read the time of the last checkpoint. The loss of a preemption is the work since that point, times the GPU count.
- Change: score each victim by min(age since last checkpoint, measured save cost on signal), weighted by GPU count. Add a short deferral window. If a victim reaches a checkpoint within W seconds, wait for it.

### Closest prior work
- FACT: Kueue classic preemption orders victims by borrowing queue, then lowest priority, then most recently admitted (kueue.sigs.k8s.io preemption page). It does not read checkpoints or lost work.
- FACT: Aryl, arXiv 2202.07896. The abstract says it reclaims servers by a server preemption cost and minimizes the number of preemptions. The abstract does not say whether it uses checkpoint progress.
- FACT: DeepShare, arXiv 2609.16682. The abstract says cost-aware best-effort preemption. The abstract does not describe the victim cost.
- SNIPPET: Lazer (DAC 2026) predicts when preemption is futile. It does not use checkpoint age.
- SNIPPET: GFS, arXiv 2509.11134 (ASPLOS 2026). Spot instance prediction and eviction reduction. No checkpoint age seen.
- SNIPPET: Alauda guide for Kueue with Kubeflow Trainer. It resumes from checkpoints after eviction. It does not pick victims by checkpoint age.

### Exact gap (one sentence)
No scheduler or paper that I opened or saw in a snippet chooses preemption victims, or defers a preemption, using each training job's time since its last checkpoint together with its measured save cost. This is INTERPRETATION. "I did not find" is not proof of absence.

### Simplest baseline that might erase the gain
- Kueue default order. Most recently admitted is a proxy for small lost work. If a job is younger than its checkpoint interval T, the proxy is exact.
- Checkpoint-on-signal with a grace period (SIGTERM handler, or Slurm signal before kill). Lost work then equals the save cost S, not the age A.
- INTERPRETATION: the policy beats both baselines only when jobs are older than T and S is large (big sharded state) or the node fails with no grace time. Check this first. It decides whether S1 lives.

### Oracle upper bound (INTERPRETATION)
- Assume checkpoints every T with random phase. The age of one random victim is uniform on (0, T), mean T/2.
- With k equally sized candidate victims, the minimum age has mean T/(k+1). For k = 8, the loss falls from T/2 to T/9, about 4.5 times.
- Real gain is smaller. Priority classes, GPU-count fit and topology cut the candidate set k. The oracle with perfect knowledge also needs live checkpoint timestamps.
- Run the bound against the baseline Kueue order. The baseline proxy may already capture most of it.

### Premise check on an Apple M4 Mac with 16 GB and no NVIDIA GPU
- Dataset: AcmeTrace (github.com/InternLM/AcmeTrace), CC-BY-4.0. FACT: the repo is 109 MB, with 880,740 jobs and 470,497 GPU jobs over 6 months on two clusters. Columns include gpu_num, submit_time, start_time, end_time, state, and fail_time and stop_time for Kalos. Helios trace is included for comparison. The full 80 GB utilization data is on HuggingFace and is not needed.
- FACT: the trace has no checkpoint events and no preemption records.
- Metric 1 (real data): the size of the candidate set k at each moment a large job arrives, and the GPU-count distribution of running jobs.
- Metric 2 (assumed data): replay the trace. At each blocked arrival, evict victims by Kueue order, by minimum age, and by oracle. Report lost GPU-hours for T in {10, 30, 60} minutes with random and start-aligned phases.
- Limit: checkpoint phases and priorities are assumptions. The premise check shows only whether k is large enough for the bound to matter. It does not prove a real gain.
- Kill rule to write before the run (heuristic): if min-age beats the Kueue order by less than 15% in lost GPU-hours at k observed in the trace, KILL.

### What needs a GPU later
- Real save cost S for sharded 8-GPU jobs (async checkpoint stall).
- A Kubernetes test with Kueue and real training pods. A kind cluster on the Mac can test the controller logic with fake GPU resources.

### Hostile verdict
- WEAK to MODERATE. It is a heuristic. A reviewer may call it an engineering patch for Kueue.
- It is not a PagedAttention-style transfer with a large visible gain. The limited-preemption idea is old, and the gain may be small.

## S2. Phi-accrual style slow-versus-hang detection for collectives

### Mechanism and environment
- Mechanism: the phi accrual failure detector (Hayashibara et al.). It outputs a suspicion level from the history of heartbeat inter-arrival times. SNIPPET: Akka and Pekko docs describe it.
- Environment: a collective completes at a rate set by message size, topology and phase. A fixed timeout must cover the slowest healthy phase, such as a checkpoint or first-step warmup.
- Change: keep a per-communicator, per-operation-size completion distribution. Use an empirical or log-normal tail, not a normal one. Raise suspicion as the wait passes the learned tail.

### Closest prior work
- FACT: CCL-D, arXiv 2605.04478 (PPoPP 2026). Abstract: rank-level probe plus decision analyzer, deployed on 4,000 GPUs for a year, locates ranks within 6 minutes. The abstract gives no threshold values.
- SNIPPET: a search summary says CCL-D uses a fixed 5-minute hang threshold and that 97% of cases past it do not recover. Verify in the PDF. If true, a fixed threshold is a strong baseline.
- SNIPPET: Unicron 2401.00134, Flare 2502.05413, Guard 2605.17879 (straggler detection and node health).

### Exact gap (one sentence)
I did not find a collective hang detector that learns the completion-time distribution per communicator and operation size and sets its alarm by a suspicion score instead of a fixed timeout. This is INTERPRETATION.

### Simplest baseline that might erase the gain
- Alarm at k times the recent median completion time of the same operation, with a floor. A one-line rule may match the learned detector.
- The fixed 5-minute rule in CCL-D, if confirmed.

### Oracle upper bound (INTERPRETATION)
- The best detector alarms at hang onset plus a small delta. The saving per hang is the fixed timeout minus delta.
- With a 5-minute timeout, the maximum saving is under 5 minutes per hang. Restart cost is 3 to 33 minutes (SNIPPET: BootSeer 2507.12619 and the 504-GPU study 2605.09370). So detection is a minor part of recovery.

### Premise check on the Mac
- NO. I found no public trace of collective completion times. AcmeTrace has job-level fields only.
- A synthetic test is possible, but then it is not real data.

### What needs a GPU later
- A multi-GPU NCCL run with fault injection and a PyTorch process group hook for completion times.

### Hostile verdict
- WEAK. Keep only as a spare. The gain is small, the baseline is strong, and a Mac premise check on real data is not possible.

## What I did not do
- I did not open full PDFs. Most kills use abstracts or snippets.
- I did not search GitHub issues of PyTorch, NCCL, Megatron, TorchTitan, verl or slime directly. The search tool returned few issue pages. A direct issue search may find more shipped features or show more kills.
- C08, C14, C16 and C28 are killed by reasoning only. Treat them as unverified.
- I did not check conference programs (OSDI, SOSP, NSDI, EuroSys, ASPLOS, MLSys, SC, FAST, ATC) one by one. I used web search that returned some of them.

## Suggested next step (a suggestion, not a decision)
- If the student wants a Mac-only check, run the S1 premise (Metric 1 and Metric 2) on AcmeTrace. It needs about two days of work.
- Write the kill rule in the journal before the run, as Section 0.2 rule 5 says.
- Log who decides in Section 10 of the journal.
