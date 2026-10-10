# R-18 Orphan findings from characterization papers (2026-10-08)

## Scope and labels
- An orphan finding is a measured waste or bottleneck with a number. No later paper, merged PR or shipped feature fixes it yet.
- This file reads 30 papers from 2025 and 2026. 26 papers carry a FACT label and 4 carry a SNIPPET label.
- FACT means I opened the paper page or PDF. For arXiv HTML pages, a fetch tool summarized the text, so I did not see every line. For the three OSDI 2026 PDFs (ASI, data pipelines, Weave), I read the raw text.
- SNIPPET means I saw the claim only in a search result.
- Fix status values are FIXED, PARTIAL and ORPHAN. FIXED needs a cited fix. ORPHAN lists what I searched.
- Search tools return US web results and some 2026 preprints. A missing fix may be a search miss. Open the sources before you cite a status.

## Excluded on purpose
- Findings about KV retention across tool gaps, prefix-cache eviction, and cache hit collapse under concurrency. These belong to killed ideas. This removes much of Copilot 2608.00101, TraceLab 2606.30560, KVCache-in-the-wild 2506.02634, and 2605.26297.
- Expert placement and expert-aware batching findings from 2510.05497.
- RL weight sync findings from RollArt 2512.22560.
- Preemption cascade findings from 2410.21680.
- Benchmarking methodology findings: ServeGen provisioning test, masked diffusion token budget below 320, and the ICLR 2026 reasoning study (it gives no numbers).

## The table

| ID | Finding | Number | Source (venue, ID) | Label | Fix status | Fix source |
|----|---------|--------|--------------------|-------|-----------|-----------|
| T01 | Codex tool calls lose time outside command execution (approval, runtime scheduling) | 77.8 h of 418.1 h tool time (18.6%). Mean 1.11 s per call, p99 10.0 s | TraceLab, arXiv 2606.30560 | FACT | ORPHAN | None found. Fused calls (CodeAct, Microsoft Agent Framework blog) cut model turns, not this residual |
| T02 | Failed tool calls start retry loops that burn tokens and time | Deep-loop with failures: 9.1% of turns, 36 LLM calls vs 9 median. Failed run_cmd p95 latency 48x longer. Edit failure rate 95.4% in one benchmark. GPT-5.5 failed runs use 20.2% to 27.5% more tokens | Copilot traces arXiv 2608.00101; Agentic AI Workload Characteristics arXiv 2605.26297 (IISWC 2026 likely) | FACT | PARTIAL | Client-side loop guards: LoopGuard (pkg.go.dev/github.com/loop-eng/loopguard), circuit-breaker patterns. No serving-side fix and no study of the saving |
| T03 | Model switch after error or rate limit destroys the prompt cache | 6.4% of sessions switch. Cache hit rate falls to 8% after a switch (67% drop). Pre-switch failure rate 36% vs 8% baseline | Copilot traces arXiv 2608.00101 | FACT | PARTIAL | Session pinning and deployment affinity: LiteLLM session affinity; OpenHands LLM fallback docs; Claude Code prompt-caching docs. No cost-aware fallback policy found |
| T04 | Context compaction costs time and resets the cache | 7.8% of sessions but 44.2% of tokens. Median 22% of turn time (p90 34%). First call after: median hit drop 66.1%. 3,170 events averaged 156 s (p99 775 s) | Copilot arXiv 2608.00101; arXiv 2608.15127 | FACT | PARTIAL | CliffCompaction arXiv 2609.26779 (cost down 50%, does not address KV); Anthropic server-side compaction with cache breakpoint (secondary article); Beyond Compaction arXiv 2606.11213 |
| T05 | Agent sandboxes sit idle but hold CPU and memory | CPU idle over 80% and memory idle over 50% of time (Aries). 90% of sandboxes use 5% CPU or less (DSec). Container idle median 243 s across turns (Copilot). Teardown cuts instance cost 65% but snapshot restore raises total cost to 4.9x | Aries arXiv 2607.29069; DSec arXiv 2609.22978; arXiv 2608.15127 | FACT | PARTIAL | Firecracker snapshot and pause (Cortex blog); AWS Lambda MicroVMs suspend (AWS docs); AgentCgroup arXiv 2602.09345. No keep-alive policy that beats persistent cost |
| T06 | microVM sandboxes hold duplicate and free memory | Page cache duplicated on host and guest. Free guest pages not returned. Median lifetime 15.5 min, p99 over 3 h. Fix cut peak host memory 40.2% | DSec arXiv 2609.22978 | FACT | PARTIAL | virtio-pmem with DAX (crosvm docs); balloon free-page reporting (Linux, Cloud Hypervisor docs); DSec internal fix. No open default in Firecracker or Kata found |
| T07 | Masked diffusion LLM serving is CPU-dispatch bound | Only 24% of single-request wall time is GPU compute. About 76% is CPU dispatch. One H200, LLaDA-8B | Serving Masked Diffusion LLMs, arXiv 2608.23807 | FACT | PARTIAL | SGLang block diffusion with CUDA graphs (arXiv 2604.11035, 2512.15745); dInfer arXiv 2510.08666; vLLM DiffusionGemma (vLLM blog). Nobody re-measures the 76% share after these |
| T08 | One long-context request slows short requests in the same decode batch | Short request TPOT 8.01 ms alone, 11.08 ms (+38.3%) with one 128K co-request, 14.39 ms (+79.7%) with two | arXiv 2608.15127 | FACT | PARTIAL | L4 arXiv 2512.19179 (length-heterogeneous batches); Medha arXiv 2409.17264 (prefill only); LAPS arXiv 2601.11589 (prefill only). No decode-side KV-byte budget found |
| T09 | Agent search and fetch calls repeat across sessions | 27% of distinct queries cause 67.3% of 373,678 search calls. 24% of URLs cause 64% of fetches. Search is median 35% of session latency | arXiv 2608.15127 | FACT | PARTIAL | Paper's own tool-result cache cut redundant calls 35.2%. Gateway caches: AWS Well-Architected agentic lens; Gravitee MCP caching. No open shared-cache standard in MCP |
| T10 | Multimodal requests spend TTFT before the LLM runs | For half of image requests, 75% of TTFT passes before LLM prefill (download, normalize, encode) | ServeGen, NSDI 2026, arXiv 2505.09999 | FACT | PARTIAL | EPD disaggregation (SGLang LMSYS blog 2026-01-12; vLLM EC connector docs); vllm bench mm-processor. No proof that preprocessing share fell |
| T11 | MCP tool schemas inflate prompts and calls run serially | Completion/prompt ratio 2x to 30x lower than chat. 46K to 877K prompt tokens per task. Prompt grows with tool count. ProMCP (SNIPPET): schema injection 56% to 72% of tokens | MCP characterization arXiv 2511.07426; ProMCP ACL Findings 2026 (SNIPPET) | FACT | PARTIAL | Deferred tool loading: Factory deferred context engine (15.1% mean input cut); Hermes tool-search docs. Serial multi-server calls unfixed |
| T12 | Restart load and save use little storage bandwidth | Restart load 21.5% of 700 GB/s peak read. Save bursts 16.0% of 250 GB/s write. Bottleneck appears near 60 nodes, absent at 2 to 4 nodes | From Detection to Recovery, arXiv 2605.09370 | FACT | PARTIAL | ByteCheckpoint arXiv 2407.20143; DataStates-LLM arXiv 2601.16956; arXiv 2512.24511. None tested on shared NFS at 60 nodes |
| T13 | Automatic retry rarely succeeds and a few nodes cause most exclusions | Auto-retry succeeded 33.3% (12 chains, 73 attempts) vs 12.5% manual. Top 3 of 63 nodes cause over 50% of exclusions | arXiv 2605.09370 | FACT | PARTIAL | NVIDIA Mission Control Autonomous Job Recovery docs; SageMaker HyperPod auto-resume; DESY requeue-with-exclude script; Guard arXiv 2605.17879 (MLSys 2026). No retry-budget study |
| T14 | Lemon nodes attract jobs and cause failures | 1.2% of nodes drew 13% of daily jobs. Removal cut 512+ GPU job failures from 14% to 4% | Revisiting Reliability, arXiv 2410.21680 | FACT | FIXED | Guard arXiv 2605.17879 (MLSys 2026); ByteRobust arXiv 2509.16293 (SOSP 2025) |
| T15 | Online inference leaves SM idle; CPU co-location hurts GPU jobs | Online inference median SM utilization 6%, memory 30%. GenAI serving: 94% memory, 5% SM. Co-located CPU jobs cut P90 SM utilization of training by 18% | ASI, OSDI 2026 (PDF, Alibaba) | FACT | PARTIAL | Orion; iGniter arXiv 2211.01713; Cloud Reamer (CPU co-location under 3.2% interference). Authors list this as an open challenge |
| T16 | Cross-switch placement lowers allreduce bandwidth | Same-ASW placement gives +27% allreduce bandwidth vs cross-ASW | ASI, OSDI 2026 | FACT | PARTIAL | Kueue Topology Aware Scheduling (Nebius, GKE docs); LeaderWorkerSet TAS. No fix for fabric churn that leaves no clean domain |
| T17 | Standby GPUs idle at night | Peak idle standby near 10,000 GPU-hours around midnight. Allocation 68% to 93% | ASI, OSDI 2026 | FACT | FIXED | SpotGPU in the same paper (deployed at Alibaba, not open source) |
| T18 | Multimodal data loading stalls on CPU transformation | Transformation 94.4% of load time (5.05 s of 5.35 s). Slowest host 42.72 s vs 5.35 s average. Cross-DC checkpoint reads for eval wasted 2.6M GPU-hours | Data Pipelines, OSDI 2026 (ByteDance) | FACT | PARTIAL | Same paper offloads transform to storage CPUs (stall down 63.2%; GPU-hours per eval 16,800 to 4,000). ByteDance internal. No open-source equivalent found |
| T19 | vLLM startup is slow and CPU-bound | 20.32 s startup (Llama3.2-3B, H100). One CPU core saturated. Dynamo transform 4.86 s to 5.97 s. Startup varies over 4x across 9 releases | Breaking the Ice, MLSys 2026, arXiv 2606.07362 | FACT | PARTIAL | vLLM sleep mode (vLLM blog); compile cache and AOT load (vLLM docs); Foundry arXiv 2604.06664; roadmap issue vllm#39749. Single-thread startup still open |
| T20 | RL rollout GPUs wait on tool sandbox cold start | Env-wait idle fraction 0.80 at k=4 workers with 8 s cold start, 0.23 at k=64. Speculative prewarming recovers no idle GPU time on real hardware | Environment bubble, COLM 2026 (Amazon Science) | FACT | PARTIAL | RollArt multi-tier image cache (arXiv 2512.22560); DSec on-demand image loading (arXiv 2609.22978). Authors say only a few percent is recoverable with sub-second tools |
| T21 | Rollout dominates RL step time and has a long tail | Rollout 85% of step time (ByteDance). 32B run: 11 days, 385 steps, 128 GPUs | Taming the Long Tail, ASPLOS 2026, arXiv 2511.16665 | FACT | FIXED | TLT (same paper, 1.7x to 2.1x over verl); RollPacker NSDI 2026 arXiv 2509.21009 |
| T22 | Disaggregated RL leaves one pool idle (dependency bubbles) | RollArt: blocking get_batch up to 62% of iteration time as GPU idleness. Env reset up to 78% of rollout under timeouts. Reward GPUs 7.4% utilized | RollArt arXiv 2512.22560; Weave, OSDI 2026 | FACT | FIXED | Weave (1.84x cost efficiency); RollArt serverless reward (88% utilization), bounded staleness |
| T23 | RL data path through one controller is slow | Effective Ray bandwidth about 0.5 GB/s in node, 0.015 GB/s across nodes. Batch 512 caused CPU OOM | RL in the Wild, arXiv 2509.25279 | FACT | FIXED | TransferQueue integrated in verl (April 2026 entry on a mirror page; confirm the merged PR) |
| T24 | Last pipeline stage is the straggler | Last stage causes at least half the slowdown in 39.3% of jobs. Logit layer 9x a transformer layer | Stragglers, OSDI 2025, arXiv 2505.05713 | FACT | FIXED | Vocabulary Parallelism, MLSys 2025, arXiv 2411.05288 (5% to 51% throughput) |
| T25 | Sequence-length imbalance across DP ranks | 21.4% of jobs, average slowdown 1.34x. Prototype +23.9% on a 32K job | arXiv 2505.05713 | FACT | FIXED | WLB-LLM, OSDI 2025, arXiv 2503.17924 (1.23x); ByteScale arXiv 2502.21231 |
| T26 | Unsynchronized Python GC pauses stall whole jobs | Pauses of hundreds of ms. Planned GC every 500 steps gave +12.6% on a 128-rank job | arXiv 2505.05713 | FACT | FIXED | torchtitan PR #148 (gc_schedule, +1.5% on 512 GPUs); SIGMA arXiv 2512.13488 (Megatron manual GC); EROICA arXiv 2506.08528 |
| T27 | Stragglers waste GPU-hours | 42.5% of traced jobs straggle. 10.4% of allocated GPU-hours wasted | arXiv 2505.05713 | FACT | FIXED | Same causes fixed in T24 to T26; Guard arXiv 2605.17879 |
| T28 | Infrastructure failures take large runtime | Infra failures: 0.2% of jobs, 18.7% of runtime. MTTF 7.9 h at 1,024 GPUs, projected 0.23 h at 131,072 GPUs | arXiv 2410.21680 | FACT | FIXED | ByteRobust (ETTR up to 97%, 9,600 GPUs), arXiv 2509.16293 |
| T29 | Fail-slow faults delay jobs | JCT delayed 1.34x on average in a 10,000-GPU cluster | GREYHOUND, ATC 2025 (Falcon, arXiv 2410.12588) | SNIPPET | FIXED | Same paper (over 99% detection accuracy) |
| T30 | Rollout bubbles from long responses | Rollout about 70% of step time; short requests idle while longest finishes | RollPacker, NSDI 2026, arXiv 2509.21009 | SNIPPET | FIXED | Same paper (tail batching) |
| T31 | Serverless model catalogs sit idle | Median active model idle 96% of hours. 83% of models active under 20% of hours (Alibaba, 89 models) | C2CServe, arXiv 2605.19481 | SNIPPET | PARTIAL | C2CServe itself (cold start down 7.1x, GH200 only); ServerlessLLM family |

## Papers read (30)
FACT (26): Copilot 2608.00101; TraceLab 2606.30560; KVCache in the Wild 2506.02634; ServeGen 2505.09999; A Year in LLM Serving 2608.13573; Masked Diffusion Serving 2608.23807; From LLM Inference to Agentic Workloads 2608.15127; Agentic AI Workload Characteristics 2605.26297; MCP Network and Systems Characterization 2511.07426; MoE data movement 2510.05497; ICLR 2026 reasoning serving study (no numbers); Stragglers 2505.05713; Revisiting Reliability 2410.21680; 504 GPUs 2605.09370; ByteRobust 2509.16293; ASI OSDI 2026; Data Pipelines OSDI 2026; Taming the Long Tail 2511.16665; RL in the Wild 2509.25279; RollArt 2512.22560; Weave OSDI 2026; Environment Bubble COLM 2026; DSec 2609.22978; Aries 2607.29069; vLLM cold start 2606.07362; When Scaling Fails 2603.04424 (its scaling table looks illustrative, so I did not use it).
SNIPPET (4): RollPacker 2509.21009; GREYHOUND (ATC 2025); C2CServe 2605.19481; ProMCP (ACL Findings 2026).
Gaps: I found no 2025 or 2026 production power or energy paper with measured waste that I could open. A claim of 14% to 76% execution-idle GPU time has no source I could trace. I left power out.

## Sections for ORPHAN and PARTIAL rows

### T01 Tool-call framework overhead (ORPHAN)
- Searched: "coding agent harness tool execution overhead approval latency", "Codex Claude Code per-call overhead measured", CodeAct and fused tool calls. Found no measurement paper and no fix.
- (a) Size: mean 1.11 s per shell call, 10.8 tool calls per request, so about 12 s of extra latency per request. The GPU does not idle. A session slot and its KV state wait longer. For 18.6% of tool time, expect a small effect on GPU cost.
- (b) Why hard: the cause is a guess (approval and runtime scheduling). The code lives in closed harnesses. The cost is user latency, not fleet cost, so no one bills for it. It may not be worth fixing in the serving stack.
- (c) Classic mechanism: syscall batching (io_uring) and pre-authorized capability sets from OS design. Group approval of a command class works like a capability token.
- (d) Premise check: yes on an Apple M4. Time a fixed shell-command list through an open harness with approval on and off. Public TraceLab data gives the call mix. No GPU needed.

### T02 Failed tool retry loops (PARTIAL)
- Searched: "agent tool retry loop detection circuit breaker", "failed tool calls wasted tokens early termination". Found client libraries only.
- (a) Size: 9.1% of turns run about 4x the LLM calls. Failed benchmark runs use 20% to 27% more tokens. Estimate under 5% of fleet tokens. This is my estimate, not a paper number.
- (b) Why hard: a retry may succeed on the next try, so a stop rule risks cutting good work. The model owner and the serving owner are different parties.
- (c) Classic mechanism: Erlang supervisor restart intensity, TCP-style backoff, and SRE error budgets. A per-session retry budget with a cost cap is the same idea.
- (d) Premise check: yes on an M4 using public TraceLab traces. Count repeated failing calls and measure the token share. Test an offline stop rule against run outcome.

### T03 Model switch cache invalidation (PARTIAL)
- Searched: "model fallback loses prompt cache", "cache-aware provider failover session affinity". Found pinning advice and LiteLLM affinity. No cost-aware policy.
- (a) Size: 6.4% of sessions switch once. Each switch re-prefills the whole context (median call 68K tokens, deep-loop up to 1.1M). Fleet effect is probably under 1% of prefill tokens. The effect is large for the few heavy sessions.
- (b) Why hard: KV state is model-specific. Another model cannot reuse it. Only a cheaper switch decision can help. It may not be worth fixing beyond a policy.
- (c) Classic mechanism: ski-rental. Wait out a short rate limit if the expected wait is below the re-prefill cost. Networks use the same break-even rule for flow rerouting.
- (d) Premise check: partly on an M4. Replay public traces with a cost model. The Copilot trace may not be public. TraceLab data lacks forced switches.

### T04 Compaction cost (PARTIAL)
- Searched: "context compaction invalidates KV prefix cache", "prefix-preserving compaction coding agent". Found CliffCompaction (cost) and API cache breakpoints.
- (a) Size: 3,170 events at 156 s average. The first call after compaction loses 66% of its hit rate. Sessions with compaction produce 44.2% of tokens.
- (b) Why hard: the summary must replace old text, so the prefix must change. Keeping the prefix means keeping a long context, which defeats the goal.
- (c) Classic mechanism: log-structured merge compaction in databases. Compact in the background, in tiers, and keep the hot prefix stable. Copy-on-write snapshots keep old readers valid.
- (d) Premise check: yes on an M4 for the cost model, using TraceLab step sizes. A real hit-rate test needs a GPU: one g6e (L40S) with vLLM and prefix caching covers an 8B model.

### T05 Idle agent sandboxes (PARTIAL)
- Searched: "agent sandbox snapshot pause resume idle cost", "keepalive policy microVM agent". Found vendor snapshots and Lambda suspend.
- (a) Size: CPU idle over 80%. Ideal system could be 3x cheaper than persistent. Snapshot per tool call is 4.9x worse. The gap between 4.9x and 0.33x is the prize.
- (b) Why hard: snapshot cost depends on memory size, and tool calls come in short bursts. A fixed policy loses on one side.
- (c) Classic mechanism: ski-rental keep-alive, and the histogram policy from serverless functions. Demand paging with UFFD gives lazy restore.
- (d) Premise check: yes on an M4. Simulate keep-alive policies on public TraceLab tool gaps with the Aries pricing model. No cloud needed.

### T06 microVM memory duplication (PARTIAL)
- Searched: "microVM memory reclaim free page reporting balloon", "virtio-pmem DAX page cache duplication sandbox". Found the mechanisms, no sandbox-density benchmark.
- (a) Size: 40.2% peak host memory in DSec. Time-integrated memory 21.2% with DAMON and balloon.
- (b) Why hard: virtio-pmem needs raw images and shares host page cache, which raises a side-channel risk in multi-tenant use. Guest pages return slowly.
- (c) Classic mechanism: kernel same-page merging (KSM), copy-on-write base images, and memory ballooning from hypervisors.
- (d) Premise check: a single AWS bare-metal or nested-virt instance is needed for Firecracker. Not an M4. A g6e does not help. Low fit for the thesis.

### T07 Diffusion LLM CPU dispatch (PARTIAL)
- Searched: "diffusion LLM serving CPU launch overhead CUDA graph", "vLLM LLaDA support", "dInfer Fast-dLLM". Found CUDA graph use in SGLang and dInfer. No new profile of host time.
- (a) Size: 76% of single-request time. Batching gives 16x at batch 16. In production with batching, the waste falls, so the realistic size is the idle host share at low load.
- (b) Why hard: each denoising step has a data-dependent mask and variable step count (11 levels). Graph capture needs static shapes.
- (c) Classic mechanism: persistent kernels and command-buffer batching from GPU drivers. Database engines use vectorized execution for the same reason.
- (d) Premise check: yes. One g6e (L40S, 48 GB) fits LLaDA-8B. Profile host versus device time with nsys, then test CUDA graphs. Low cost.

### T08 Long-context decode interference (PARTIAL)
- Searched: "long-context request interference decode batch TPOT", "heterogeneous context decode scheduling". Found prefill-side fixes and L4.
- (a) Size: +38% to +80% TPOT for short requests with one or two 128K co-requests. For a Llama-3-8B-class model, 128K tokens of KV is about 16 GiB. At 3.35 TB/s that is about 5 ms per step. This matches the +3 ms seen. This is my calculation, the paper model may differ.
- (b) Why hard: decode attention must read the KV of every sequence each step. Scheduling cannot remove bytes. Only batching choices can.
- (c) Classic mechanism: I/O schedulers with a byte budget per quantum, weighted fair queuing, and earliest-deadline-first from real-time systems. Admit a decode batch by KV bytes read, not by request count.
- (d) Premise check: yes on one g6e (L40S, 864 GB/s) or one H100. Run vLLM with one 128K co-request and short requests. Measure the TPOT shift against the byte model. Cheap.

### T09 Repeated search and fetch (PARTIAL)
- Searched: "shared tool result cache across agent sessions", "MCP gateway caching 2026". Found vendor gateways. No independent benchmark.
- (a) Size: 67.3% of search calls repeat a distinct query. Own cache removed 35.2% of redundant calls and 19.3% of search latency. Vendor claims 4% to 7% token saving.
- (b) Why hard: results go stale. Side-effecting tools cannot cache. Shared caches risk cross-tenant leaks.
- (c) Classic mechanism: HTTP cache with TTL and validators, request coalescing (singleflight), and CDN edge caching.
- (d) Premise check: yes on an M4 if a public agent trace has query text. The trace in 2608.15127 may not be public. Check before planning.

### T10 Multimodal pre-prefill time (PARTIAL)
- Searched: "multimodal serving encoder disaggregation EPD", "vllm mm-processor preprocessing bottleneck". Found EPD and a profiler.
- (a) Size: 75% of TTFT for half of image requests. Encoder queueing also delays text requests.
- (b) Why hard: the share depends on image size and the CPU count. Decoding is CPU work and hardware decoders are limited.
- (c) Classic mechanism: pipelined I/O with prefetch from storage systems, and offload to a fixed-function decoder (nvJPEG). Network CDNs store resized variants.
- (d) Premise check: yes. `vllm bench mm-processor` on one g6e. A Mac can time the CPU stage with a Hugging Face processor.

### T11 MCP schema and serial calls (PARTIAL)
- Searched: "MCP tool schema token overhead lazy loading", "tool search deferred tools". Found several harnesses with deferred loading.
- (a) Size: about 25K tokens for several servers (a feature request, not measured). With prefix caching these tokens cost about a tenth of the price. The real cost is TTFT on the first turn and context pressure.
- (b) Why hard: the model must guess which tool it needs. Failures are silent. Serial multi-server calls come from the protocol and the model.
- (c) Classic mechanism: lazy linking (PLT) in dynamic loaders and demand paging. The directory of names is resident and the schema loads on first use.
- (d) Premise check: yes on an M4. Count schema tokens of public MCP servers. Measure model accuracy loss only with API access. Low novelty.

### T12 Checkpoint bandwidth underuse (PARTIAL)
- Searched: "checkpoint load restart bandwidth NFS ByteCheckpoint DataStates-LLM", "LLM checkpoint I/O small buffers". Found systems and a benchmark study. None isolates 60-node NFS contention.
- (a) Size: restart uses 21.5% of read peak. Save bursts use 16.0% of write peak. About 78% to 84% of the peak is unused.
- (b) Why hard: small tensors cause many RPCs. NFS queueing grows with node count. Peak numbers may not be reachable for any client.
- (c) Classic mechanism: two-phase collective I/O (ROMIO), write combining, and RPC pipelining (NFS nconnect). Burst buffers absorb save bursts.
- (d) Premise check: partly. One g6e with FSx or NFS can test small-versus-coalesced RPC ratios. The 60-node effect needs a cluster. Low fit on one node.

### T13 Retry success and node concentration (PARTIAL)
- Searched: "automatic job restart success rate bad node exclude", "SageMaker HyperPod auto-resume", "Mission Control Autonomous Job Recovery". Found mechanisms, no success rates.
- (a) Size: 33.3% retry success. The top 3 nodes cause over 50% of exclusions. Sample is small (12 chains).
- (b) Why hard: data is one 63-node cluster. A retry fails when the cause is not a node, such as bad code or data.
- (c) Classic mechanism: Erlang supervision trees with restart intensity, circuit breakers, and git-bisect style node bisection to find the bad member.
- (d) Premise check: no public data for this cluster. The ASI trace (github.com/alibaba/clusterdata, cluster-trace-gpu-v2026) may hold retry chains. Unverified. Fit is low.

### T15 Low online SM use and CPU co-location (PARTIAL)
- Searched: "co-locating online inference with training GPU sharing interference MPS MIG", "CPU-only jobs colocated GPU nodes SM utilization". Found Orion, iGniter, Cloud Reamer. No source for the 18% drop.
- (a) Size: median SM 6% for online inference, over half of jobs. CPU co-location costs 18% of P90 SM for training on the nodes where it runs.
- (b) Why hard: latency isolation on a GPU is weak. A memory-bound LLM job leaves SM free, but it holds nearly all memory (94%).
- (c) Classic mechanism: mixed-criticality scheduling from real-time systems, and priority queues with preemption from networks. For CPU: cpusets and memory-bandwidth partitioning (Intel RDT).
- (d) Premise check: yes for the CPU side. One g6e: run a data loader under stress-ng on the host and measure SM and step time. The 6% claim needs the ASI trace (public).

### T16 Cross-switch placement (PARTIAL)
- Searched: "topology-aware scheduling Kueue Slurm block topology allreduce", "same switch placement allreduce bandwidth". Found Kueue TAS. No Slurm or NCCL numbers.
- (a) Size: +27% allreduce bandwidth for same-switch placement. Gap matters most for large jobs.
- (b) Why hard: churn leaves no clean domain. Defragmentation needs migration, which needs checkpoint cost.
- (c) Classic mechanism: contiguous partition allocation from Blue Gene and buddy allocators from memory management. Compaction by migration.
- (d) Premise check: not on one node. A simulator on an M4 with the public ASI trace is possible. Needs the switch map.

### T18 Data pipeline transformation wall (PARTIAL)
- Searched: the paper's own fix and open alternatives: "offload data transformation storage tier CPU", GPU decode (DALI, nvJPEG).
- (a) Size: 94.4% of load time on CPU transform. 37.4 s straggler gap in one step. Stall reduction 63.2% in ByteDance.
- (b) Why hard: the fix relies on idle storage-node CPUs (20% to 30% used), which most clusters lack. Pre-decoding expands video storage over 40x.
- (c) Classic mechanism: predicate and projection pushdown to the storage tier (computational storage) from databases.
- (d) Premise check: yes. One g6e or H100: compare CPU decode and nvJPEG decode on a video and image mix. Check stall share by step.

### T19 vLLM startup (PARTIAL)
- Searched: "vLLM startup time torch.compile AOT cache CUDA graph", "sleep mode cold start". Found sleep mode, compile cache and Foundry.
- (a) Size: 20.32 s for a 3B model. Serverless scale-out latency is tied to this. Other cold-start parts (image pull, weight fetch) can be larger.
- (b) Why hard: steps run in sequence on one CPU core. Compile cache keys change with the GPU model and torch build.
- (c) Classic mechanism: process checkpoint and restore (CRIU with cuda-checkpoint), fork from a warm zygote, and parallel build graphs with a dependency scheduler.
- (d) Premise check: yes. One g6e or p5 node. Time cuda-checkpoint restore against fresh start. This matches the thesis author's CRIU background. Check vLLM sleep mode first to avoid a duplicate.

### T20 Sandbox cold start in agentic RL (PARTIAL)
- Searched: "RL rollout environment bubble sandbox cold start prewarming", RollArt image cache. Found RollArt and DSec.
- (a) Size: up to 0.80 idle fraction in the worst case (8 s cold start, k=4). Real Docker (0.3 s to 1.2 s start) gives 0.008 to 0.049. Realistic regime is small.
- (b) Why hard: the paper says speculative prewarming gives no gain. With fast starts, the loss is a few percent. It may not be worth fixing.
- (c) Classic mechanism: Little's law sizing of a warm pool, and fork servers (zygotes).
- (d) Premise check: yes on one g6e with Docker and a small model. Or on an M4 with the queue model. Low value if cold start is under 1.2 s.

### T31 Serverless model idle time (PARTIAL, SNIPPET)
- Searched: the paper title and the claim only. I did not open the text.
- (a) Size: median model idle 96% of hours. Open the paper before use.
- (b) Why hard: the fix (C2CServe) needs GH200 NVLink-C2C. Other hardware has 60 GB/s PCIe.
- (c) Classic mechanism: demand paging with a large backing store, and keep-alive histograms.
- (d) Premise check: Alibaba trace use on an M4 is possible if it is public. Unverified.

## How to use this file
- The strongest ORPHAN candidates for a next review are T01, T08, T05 and T19. They have a cheap single-node premise check.
- T07 and T09 follow. They need a short novelty check against the sources above.
- Do not start from T03, T11 or T20. The waste is small in the realistic regime, or others ship fixes.
- Log any chosen item in RESEARCH_JOURNAL.md under Section 0.6 rules. An AI ranking is not a decision.
