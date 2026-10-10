# Review of R-18 orphan findings (2026-10-08)

## Scope and method
- This file attacks five shortlist items from R-18: A (T01), B (T05+T06+T20), C (T19), D (T12+T13), E (T15).
- The reviewer role is a hostile systems PC member.
- I opened each cited source. FACT means I read the paper text or the project page. SNIPPET means I saw it only in a search result.
- Search tools return US web results. A missing fix may be a search miss. Treat "no fix found" as weak evidence.
- Result: no item SURVIVES. B is WEAK as a policy question only. A, C, D and E are KILL.
- An AI verdict is not a decision. Log the decision in RESEARCH_JOURNAL.md Section 10. This review did not edit the journal.

## A. T01 harness overhead outside command execution

### (1) Source check
- Source: TraceLab, arXiv 2606.30560, Section 6.3, plus the TraceLab page "codex_wall_internal_gap". Label: FACT.
- The numbers match: 77.8 h of 418.1 h (18.6%), mean 1.11 s, p99 10.0 s, 253,391 Codex calls.
- The paper defines the "overhead" as end-to-end time minus runner-reported execution time. It is a residual.
- The residual contains permission approval, shell startup, scheduling, output propagation and bookkeeping. The paper does not separate them.
- The TraceLab page says the residual is "an upper bound on client-side waiting". It says the traces keep no approval events.
- Regime: p50 is 0.13 s, p90 is 0.24 s, p99 is 10.0 s. The mean comes from the tail. R-18 section T01 says "mean 1.11 s per call" without this shape.
- The page has a second table. All timed calls give 598.6 h end-to-end, 129.7 h residual (21.7%). The R-18 row uses the subset table. Both are the same dataset.
- R-18 says "about 12 s of extra latency per request". The median-based figure is 10.8 calls x 0.13 s = 1.4 s. The 12 s figure uses the mean, which the tail inflates.

### (2) Prior art searches
- "coding agent tool call latency overhead persistent shell session approval batching harness 2026": no measurement of persistent shell and no approval-batching study. CodeAct fuses calls (Microsoft Agent Framework blog, 2026-04-23).
- "Codex Claude Code shell command startup latency login shell per command persistent session": Anthropic's bash tool is one long-lived process per session. Claude Code release 2.1.51 skips the login shell when a shell snapshot exists (GitHub issue 28043). The persistent-shell mechanism is already shipped.
- "agent harness approval prompt batching permission allowlist": Microsoft Agent Framework has per-tool approval modes. Claude Code and Codex have allowlists and auto-approve modes. These are capability-style pre-approval.
- Result: all three candidate mechanisms exist in shipped products.

### (3) One-sentence rejection
The "waste" is an unexplained residual whose tail is probably a human waiting to approve, and the median is 0.13 s, so no mechanism has measured waste to remove.

### (4) Simplest baseline
Set the harness to auto-approve and reuse one shell. This is a config change in shipped tools.

### (5) Oracle
- Remove the whole residual: 18.6% of tool time. Tool time is only part of session wall time. Model time dominates.
- Remove only the part a harness can fix (p50 class, 0.13 s): about 1.4 s per request.
- The GPU does not idle. The gain is user latency. No fleet cost changes.
- The oracle gain is not meaningful at the p50. The tail is human time or sandbox policy, not a systems mechanism.

### (6) Verdict: KILL
- The cause is unknown, the mechanisms exist, and the thesis would reduce to a measurement paper.

### (7) Survivor check
- None. If you still want to test the cause, one cheap check on an M4 is below. It does not make a thesis.
- Check: run 1,000 `true` and `echo` commands through the Codex CLI with approval off. Compare against a bare `fork+exec`. If the median residual is under 50 ms, stop.

## B. T05 + T06 + T20 sandbox fleets (idle CPU, duplicate memory, cold start)

### (1) Source check
- Aries, arXiv 2607.29069 Section 4.3. Label: FACT.
  - "CPU and memory remain idle for more than 80% and 50% of the time". The 80% is a statement across harnesses and workloads. Idle falls below 20% when conditioned on a tool call.
  - Teardown after each call cuts instance cost 65%. Snapshot capture and restore raise total cost to 4.9x of persistent. The ideal is 3x cheaper than persistent.
  - The regime is an analytical replay. The authors replay recorded tool-call traces under the AWS Lambda MicroVMs price model. It is not a measured fleet. Workloads are OpenClaw and Hermes on SWE-Bench Pro and Terminal-Bench 2. Over 100 production sandboxes add telemetry.
  - The 4.9x number depends on the snapshot price in the billing model. It is not a hardware cost.
- DSec, arXiv 2609.22978. Label: FACT.
  - Virtio-pmem with DAX cut peak host memory 40.2% in a dedicated test cluster. DAMON with balloon free-page reporting cut time-integrated host memory 21.2%.
  - About 90% of sandboxes use 5% or less of requested CPU. Median lifetime is 15.5 min for microVMs.
  - DSec already deploys virtio-pmem with DAX, DAMON with balloon, snapshot pause and resume (4,889 microVM snapshots in one week), and "high-density overcommit". It runs 800 microVMs or 3,200 containers per node.
- T20 (COLM 2026): R-18 itself says the realistic idle fraction is 0.008 to 0.049 with real Docker start times. I did not reopen it. Label: R-18 FACT, not rechecked.

### (2) Prior art searches
- "agent sandbox memory overcommit balloon snapshot restore Firecracker density idle agents 2026": DSec ships the full stack. Wrenn (a Firecracker sandbox project) inflates the balloon before snapshot to shrink memfiles from 20 GB to 1 to 2 GB.
- Prior R-18 searches: Firecracker snapshot and pause (Cortex blog), AWS Lambda MicroVMs suspend, AgentCgroup arXiv 2602.09345.
- Classical: KSM, ballooning, and the serverless keep-alive histogram policy (Serverless in the Wild, ATC 2020). All are decades old.
- Result: the memory mechanism is shipped in a production system. Only the open source default and the keep-alive policy remain.

### (3) One-sentence rejection
DSec already ships balloon, DAX page sharing, snapshot pause and overcommit in production, and the rest is a keep-alive policy that serverless papers solved in 2020.

### (4) Simplest baseline
- Fixed keep-alive timeout, tuned per workload (Aries sweeps this).
- Second baseline: the hybrid histogram policy from serverless.

### (5) Oracle
- Aries gives 3x cheaper than persistent as the ideal. That is the upper bound over the persistent baseline, not over a tuned timeout.
- The gap between the best fixed timeout and the oracle is not reported. It may be small. The thesis prize lives only in this gap.
- The cost is a price-model artifact. Real hosts pay for RAM, not Lambda snapshot fees. On a self-run fleet with balloon and overcommit (DSec), the 4.9x does not apply.
- Verdict on size: meaningful only inside the Lambda price model.

### (6) Verdict
- Memory overcommit, page sharing, snapshot restore (the mechanism in the shortlist): KILL. DSec ships it.
- Per-tool-phase keep-alive policy: WEAK. It is a policy study on an analytical price model, it needs no new mechanism, and a reviewer will call it tuning.

### (7) Premise check for the WEAK policy question
- Narrowed question: does any online keep-alive policy that uses the tool-phase signal (shell command class, file edit vs test run) get within 15% of the offline-optimal cost, and beat the best fixed timeout by more than 20%?
- Platform: Apple M4, no GPU, no cloud. Use the public TraceLab tool-call gaps (check first that per-call timestamps and tool names are public) and the Aries price model.
- Steps: (a) replay the trace under fixed timeouts and compute cost; (b) compute the offline optimum by dynamic programming over gaps; (c) replay a histogram policy; (d) replay a policy with tool-class features.
- Kill thresholds:
  - Best fixed timeout is within 20% of the offline optimum: KILL.
  - Tool-class policy beats the histogram policy by under 10%: KILL.
  - The result flips when the snapshot price changes by 2x: KILL (a price artifact).
- Grid edges to check before accepting "no effect":
  - Snapshot price from 0.1x to 10x of the Lambda price.
  - Snapshot restore latency from 0.1 s to 5 s.
  - Timeout from 1 s to 3,600 s, and the "infinite" persistent case.
  - Both harnesses (OpenClaw, Hermes), both benchmarks, and the DeepResearch workload that Aries excluded.
  - Sandbox memory size from 512 MB to 16 GB, because snapshot cost scales with memory.
  - A bursty trace with short gaps under 5 s, and a trace with human-paced gaps over 60 s.

## C. T19 vLLM startup (20 s, CPU-bound)

### (1) Source check
- Source: Breaking the Ice, arXiv 2606.07362 (MLSys 2026). Label: FACT.
- The numbers match: 20.32 s for Llama3.2-3B on H100. At least one core is at 100% through startup. Startup varies over 4x across nine releases (OPT-6.7B). Six steps: bootstrap, tokenizer, model load, torch compile, KV profiling, CUDA graph capture.
- Regime limits:
  - The 3B model has weight load of only 7 to 10% of startup. A 70B model reverses this: weight load dominates and CPU-bound steps matter less.
  - The paper does not discuss snapshots, CRIU or fork. It names overlap of compile, KV profiling and graph capture as future work.
  - The 4x release variation means any startup number is tied to one vLLM version.
  - L40S and H100 give nearly the same time for most steps.

### (2) Prior art searches
- "vLLM RFC 34303 checkpoint restore cuda-checkpoint snapshot startup": RFC 34303 is open since 2026-02-11, draft, no linked PR (I read the issue). It proposes CRIU plus cuda-checkpoint in tiers. Shipped neighbours: NVIDIA Dynamo Snapshot (preview, about 1 min cold vs about 10 s warm), Parasail (3 to 5x faster for vLLM, with a CRIU patch of about 600 lines for io_uring), Cerebrium, Modal (alpha, GPU snapshots for vLLM and SGLang, claims up to 10x).
- "Modal GPU memory snapshots vLLM cold start cuda-checkpoint": Modal's docs say snapshots do not help when weight loading dominates.
- "CRIUgpu transparent checkpointing GPU accelerated workloads": CRIUgpu is arXiv 2502.16631. NVIDIA provides the CRIU CUDA plugin.
- "LLM inference engine cold start snapshot restore 2026 sub-second": I found no sub-second result. Restores take 5 to 25 s.
- Decisive hit: NVIDIA, "Fast Recovery for LLM Serving via Decoupled Device Memory Lifetime in Dynamo", arXiv 2609.25451 (2026-09-21). I read the abstract and the introduction from the PDF. It uses CRIU and cuda-checkpoint, with upstream CRIU contributions for multi-GPU engines. It covers TP8 and TEP8 on 8xB200, on vLLM and SGLang, four models. It reports recovery under 7 s, 13 to 29 times faster than a warm restart. I did not read the whole paper.
- Result: the exact idea (CRIU plus cuda-checkpoint for vLLM, multi-GPU) exists as a product and as an NVIDIA paper from 17 days ago.

### (3) One-sentence rejection
NVIDIA shipped and published multi-GPU CRIU plus cuda-checkpoint restore for vLLM and SGLang, so the student's mechanism is already the product.

### (4) Simplest baseline
- vLLM with warm compile cache, AOT load and warm page cache.
- Second baseline: vLLM sleep mode (note that RFC 34303 says `cumem` sleep is broken since v0.14.0, issue 32714).

### (5) Oracle
- Dynamo reports under 7 s versus about 100 to 150 s warm restart for 27B to large MoE models (their Figure 1). That is a large gain in the regime where it applies.
- For the 3B model in T19, the oracle is 20 s to about 2 s. Absolute saving is 18 s per replica start.
- The gain matters for scale-to-zero and recovery. It does not matter for a long-running replica.
- The oracle is meaningful, but someone already collected it.

### (6) Verdict: KILL
- The mechanism and the evaluation exist. A student version would be a reimplementation.
- The vLLM RFC is open, but an upstream PR would be engineering work, not a thesis contribution. NVIDIA can land it first.

### (7) Survivor check
- No survivor. A narrower sliver remains that I do not recommend: scale-out onto a new node where no device-resident weights exist. Dynamo's GMS needs surviving GPU state, and snapshots restore device memory at storage speed. Modal says weight-bound restores do not gain. The sliver is "restore time becomes weight-load time at 70B". That is a known limit, not an open problem.
- If the student wants the cheap check anyway: on one g6e.12xlarge (4xL40S) or p5, time (i) cold start, (ii) warm compile cache start, (iii) cuda-checkpoint plus CRIU restore, for 3B, 8B and a 70B FP8 model at TP4. Kill if restore for 70B is within 1.3x of warm-cache start.
- Grid edges: model size 1B to 70B, TP 1/2/4, page cache cold and warm, snapshot on NVMe vs EBS vs tmpfs, vLLM versions (pin them), driver 570 vs 580, restore to a different GPU UUID.

## D. T12 + T13 training restart (storage bandwidth, retry, node exclusion)

### (1) Source check
- Source: From Detection to Recovery, arXiv 2605.09370. Label: FACT.
- Numbers match, with limits:
  - 63-node B200 cluster, 504 GPUs, VAST NFS. 20 restart-loading events, mean read 150.8 GB/s.
  - The 700 GB/s denominator is a cluster configuration value. The paper says it "should not be interpreted as separately measured peak bandwidths". R-18 calls it "peak". The 21.5% is not proven underuse.
  - 150 GB/s is already a high rate. The paper says a 200 to 400 Gbps network upgrade will not help. Mean restart loading is 33 min.
  - Auto-retry 33.3% is 4 of 12 chains (73 attempts), versus 12.5% manual. The sample is tiny.
  - The paper ties limited retry success to gang scheduling over 60 nodes with only 3 spares, not to bad nodes. It says exclusions mix hardware faults, degradation and operator choice.
  - "Top 3 of 63 nodes cause over 50% of exclusions" is over 224 sessions in 73 days, on one cluster.
- R-18 section T13 implies the retry failures come from bad nodes. The paper does not say this. This is an R-18 error to correct.

### (2) Prior art searches
- "in-memory checkpoint peer-to-peer recovery from surviving replicas GPU memory training restart": Gemini (SOSP 2023, 13x faster recovery), REFT (in-memory load 0.8 s vs NFS 9.3 s), and AWS SageMaker HyperPod checkpointless training (announced 2025-12) with peer-to-peer state transfer from healthy accelerators and NeMo support. Claimed recovery: under 2 min versus 15 to 30 min.
- "checkpoint loading broadcast read once data parallel replicas fast restart": Google Orbax single-replica read then broadcast (6.8x on a 2,048-VM CPU cluster, 2.6x on TPU v5e-256). BootSeer (parallel striped reads). ByteCheckpoint (3.51x load).
- "node health scoring exclude flaky GPU nodes job restart history scheduler": Slurm sites build this themselves. Guard (MLSys 2026) and ByteRobust (SOSP 2025) cover lemon node removal. R-18 marks T14 FIXED.
- Result: all three candidate mechanisms exist.

### (3) One-sentence rejection
Peer restore, read-once broadcast and bad-node exclusion are all shipped or published, and the 21.5% and 33.3% numbers come from an unmeasured denominator and 12 retry chains.

### (4) Simplest baseline
Rank 0 per DP group reads the checkpoint and broadcasts it with NCCL. This is a few lines in PyTorch DCP style loaders.

### (5) Oracle
- If restore moves from 33 min to a few seconds by peer copy, the gain is large for this cluster. The gain is already available in HyperPod.
- The failure-history idea has an oracle limited by the 3 spare nodes. With few spares, smarter exclusion cannot help.
- Gain exists but is owned by vendors. It is not open.

### (6) Verdict: KILL

### (7) Survivor check
- None. The only open sub-question is whether read-once broadcast beats parallel shard reads on shared NFS at 60 nodes. That needs a 60-node cluster and does not fit one rented instance.

## E. T15 online inference at 6% SM, CPU co-location cuts SM

### (1) Source check
- Source: ASI, OSDI 2026 (Alibaba). I extracted the PDF text and read the passages. Label: FACT.
- The 6% median SM and 30% median memory are aggregated over all online-inference jobs. The paper splits them: GenAI inference uses 94% of memory and 5% of SM; conventional DNN inference uses 20% of memory and 6% of SM.
- The aggregate is dominated by small DNN jobs (CTR, BERT). Qwen has median SM utilization of 26%, 2.7x CTR and 9.6x BERT. So the LLM figure in the aggregate is not 6%.
- The 18% claim is about GPU training, not inference. The text: median and P90 SM utilization fall 10% and 18% on CPU-job-dominant nodes relative to GPU-job-dominant nodes. It is a cross-node comparison, not a controlled test. It does not show cause.
- R-18 says "CPU co-location cuts P90 SM by 18%". The correct reading is "P90 SM of training jobs is 18% lower on nodes where CPU jobs dominate".
- SM utilization in this metric counts a kernel as busy if it runs on any SM. It does not measure SM occupancy or tensor core use. Memory-bound decode can show high "utilization" and still waste compute. I did not confirm which metric ASI uses.
- The paper itself proposes the harvest idea: a DNN job could run on compute a memory-bound GenAI job leaves idle. It says sharing fails under latency SLOs, MIG is too static and time-sharing has weak isolation. It marks interference as an open challenge.

### (2) Prior art searches
- "harvest idle GPU SMs best-effort jobs online inference SLO spatial sharing": MuxFlow (dynamic SM share by online activity, production at scale), Orion, Herald, ParvaGPU, Hummingbird (arXiv 2601.04071, microsecond SLO-oriented preemption).
- "LLM serving SM partitioning green contexts prefill decode colocation": Nexus, Bullet (1.26x), Drift/MuxWise, semi-PD, DuetServe (ICML 2026, SM partitioning only when decode SLO is at risk).
- "colocate LLM decode with training bubbles harvesting": PipeFill (MLSys 2025, up to 63% utilization), BubbleTea, SpecInF.
- Result: SM-level multiplexing for LLM serving is a crowded 2025 to 2026 area.

### (3) One-sentence rejection
The 6% is mostly small DNN jobs, the 18% is a training-node correlation, and SM spatial sharing with SLO guards has at least six published systems.

### (4) Simplest baseline
MPS with a client SM cap for the best-effort job, plus an SLO monitor that pauses it. Hummingbird and MuxFlow are stronger baselines.

### (5) Oracle
- For LLM decode, harvestable SMs are real because decode is memory-bound. But the best-effort job also needs HBM bandwidth. Decode saturates bandwidth first. A compute-heavy neighbour gains SMs but steals bandwidth.
- The ASI paper says GenAI jobs hold 94% of memory, so there is no room for a second job.
- The realistic oracle is small for bandwidth-bound decode and large only for tiny DNN jobs, which are not the thesis target.

### (6) Verdict: KILL

### (7) Survivor check
- None. If the student wants a cheap check of the real premise: on one g6e (L40S) run vLLM decode at SLO load plus an MPS-capped compute-bound kernel loop. Measure P99 TPOT. Kill if P99 TPOT rises more than 10% at any SM cap that gives the neighbour more than 10% of its solo throughput.
- Grid edges: SM cap 10% to 50%, decode batch 1 to 128, context 1K to 32K, neighbour bandwidth-bound and compute-bound, MPS vs green contexts.

## Is reading characterization papers for orphan findings a productive method?
- Based on this review, the method is a poor main filter and a good question generator. Of five candidates, I found that four had a shipped fix or a near-identical 2026 paper within one hour of searching.
- Three of five source numbers were bent in R-18 when compared to the primary text: the T01 mean hides a p50 of 0.13 s, the T12 "peak" is a configuration value, and the T15 "18%" applies to training nodes by correlation.
- Characterization papers report waste and also usually point to the fix in their own text. They are written close to the fix, and the field moves in weeks, as the NVIDIA paper on 2026-09-21 shows.
- A paper's "open challenge" line is more useful than its headline number. The ASI paper names its own gap, and that gap (multi-resource interference) is the only part that looks open.
- A better method for a PagedAttention-style thesis: start from a mechanism with a clear invariant in another field (for example, copy-on-write, ski-rental, log-structured compaction), then look for a place where a system pays the cost the mechanism removes. Use characterization papers only to check size.
- Rule for next time: open the primary text for every number, search for the fix before you write the idea, and record the regime in the same sentence as the number.

## Corrections to feed back into R-18
- T01: add p50 0.13 s, and state that the residual is not approval. Cut the "about 12 s per request" line.
- T12: state that 700 GB/s is a configuration value, not a measured peak.
- T13: remove the implication that retry failures come from bad nodes. The paper cites gang scheduling and 3 spare nodes.
- T15: state that the 18% is training nodes, cross-node, and correlational. State that the 6% aggregate is mostly CTR and BERT.
- T19: add Dynamo 2609.25451, Dynamo Snapshot docs, Parasail, Modal and vLLM RFC 34303 as fix sources. Change status to FIXED for the multi-GPU case.
- T05 and T06: add that DSec deploys virtio-pmem DAX, balloon, snapshots and overcommit. Change T06 to FIXED in production (closed source).
