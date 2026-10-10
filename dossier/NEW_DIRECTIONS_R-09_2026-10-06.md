# R-09: New research directions from cross-domain transfer

Date: 2026-10-06. Journal IDs: I-46 to I-57, R-09, D-28. Workflow: `.claude/skills/research-gap-dossier`.
Labels: FACT (a source or a run showed it), INTERPRETATION, HYPOTHESIS.

> Pointer (2026-10-07): R-10 killed I-46, I-47, I-48 and I-49 and corrected two claims in this file
> (journal Section 8.1 rows 42 and 43). See `dossier/ROUTES_R-10_2026-10-07.md`. The journal wins.

## 1. Scope and method

- The user asked for 4 to 5 new directions. Use peer-reviewed papers only. Find gaps from first
  principles. Take a solution from a different field and apply it here, so that we can test it.
- Evidence: only peer-reviewed papers (conference, journal, or peer-reviewed workshop). arXiv-only
  preprints do not count as evidence. Many preprints appeared in the searches. Section 7 lists
  them as scoop signals only. Because of this rule, each gap claim is less certain than usual.
- Constraints (journal Section 2): Apple M4 Mac only, no NVIDIA GPU, October to December 2026,
  target roles in GPU and inference performance engineering.
- Each candidate must have a decisive first test that runs on the Mac.
- Process: 12 candidates, attack pass 1 (prior art, simple baseline, scale, data), attack pass 2
  (new targeted searches), then 4 survivors.

## 2. Result in one table

| ID | Direction | Field we borrow from | Status | Confidence in gap |
|---|---|---|---|---|
| I-46 | Sampled simulation of LLM serving traces | CPU simulation sampling (SimPoint, SMARTS) | MEASURE-FIRST | medium |
| I-47 | Closed-loop validity of agent serving evaluation | Queueing theory (open vs closed systems, MVA) | MEASURE-FIRST | low |
| I-48 | Margin-gated quality control for production inference numerics | Factory quality control (CUSUM control charts) | MEASURE-FIRST | low to medium |
| I-49 | Failure-clustering-aware recovery provisioning for training | Earthquake aftershock models (Hawkes, ETAS) | HOLD, data-gated | low |
| I-50 | KV retention across tool and human gaps (ski rental with predictions) | Online algorithms | KILL, crowded | n/a |
| I-51 | Speculative tool execution | CPU branch prediction | KILL, crowded | n/a |
| I-52 | Reproducible summation for batch-invariant kernels | HPC reproducible arithmetic | KILL, crowded, needs GPU | n/a |
| I-53 | Fork of agent sandboxes for tree search | OS and VM fork, CRIU | KILL, crowded | n/a |
| I-54 | Group testing to find faulty GPU nodes | Epidemiology (Dorfman group testing) | KILL, crowded and shipping | n/a |
| I-55 | Hedged requests with prefix-cache affinity | "The Tail at Scale" hedging | KILL, first principles | n/a |
| I-56 | Certified token stability from rounding-error bounds | Numerical analysis | DEPRIORITIZED, folded into I-48 | n/a |
| I-57 | Fused tool calls ("superinstructions") | Interpreter superinstructions | KILL, crowded | n/a |

## 3. Direction I-46: sampled simulation of LLM serving traces

**Idea.** To test a serving change on a long trace, do not replay the full trace. Replay a few
short, representative windows and warm the system state first. This is how computer architects
make slow CPU simulation fast.

**Borrowed from.** SimPoint (ASPLOS 2002) clusters program intervals and simulates one interval
for each cluster. SMARTS (ISCA 2003) takes many small periodic samples. It runs a fast
"functional warming" mode between samples to keep caches correct, and it gives a confidence
interval. SMARTS reported 0.64% average CPI error.

**Problem from first principles.** A serving benchmark on a production-like trace costs hours of
GPU time for each configuration. For agent workloads, the state has a long memory: prefix-cache
contents depend on sessions that started hours earlier. A random short window without warm-up
starts with an empty cache, so it can give a wrong hit rate and wrong TTFT.

**Peer-reviewed prior art.**

| Work | What it solves | What it does not solve |
|---|---|---|
| Vidur (MLSys 2024) | CPU-run simulator of LLM serving with profiled operator costs | Simulates the full trace. No interval sampling |
| Sampling Where It Matters, PKH (EuroMLSys 2026) | Picks which serving configurations to profile, 10% throughput MAPE | Samples configurations, not time windows of one trace |
| UniCache (SIGMETRICS 2026) | Trace-driven prefix-cache simulator and eviction policy | Full-trace simulation |
| KVCache Cache in the Wild (ATC 2025) | Real KV-cache workload traces and workload-aware eviction | No sampling method |

**Exact gap.** I did not find peer-reviewed work that samples time windows of an LLM serving
trace with cache and queue warm-up and gives an error bound. Do not claim simulation of serving,
or configuration sampling, as novel.

**Research question.** Can SMARTS-style systematic sampling with prefix-cache warm-up estimate
mean and p99 TTFT, TPOT, throughput and hit rate of a long agent trace within 3% error at 10x
lower cost, when random windows without warm-up cannot?

**Hypothesis.** Random windows are accurate on chat traces (short memory) and inaccurate on agent
traces (long memory). Warm-up fixes the agent case.

**Decisive Mac test (about 1 to 2 weeks).**
- Use Vidur, or a small discrete-event simulator with a profiled cost model, on the Mac CPU.
- Use public traces: TraceLab (agent sessions with timing, prefix and append tokens) and a chat
  trace (for example the Azure trace that Splitwise, ISCA 2024, released).
- Ground truth: full-trace simulation.
- Samplers: (S1) random windows, no warm-up. (S2) systematic windows with functional warm-up
  (replay skipped requests in a fast cache-only mode). (S3) SimPoint-style clusters of windows
  by workload features, weighted.
- Predeclared criteria (heuristics):
  - KILL if S1 is within 3% on all metrics at 10x reduction on the agent trace.
  - GO if S1 has more than 10% error on p99 TTFT or hit rate on the agent trace, and S2 or S3
    gets under 3% at the same reduction.
  - Otherwise PIVOT.

**Later GPU step.** Validate on real vLLM or SGLang runs that sampled GPU benchmarks match full runs
(a few GPU-hours).

**Hostile reviewer.** "Serving systems have short memory, so any random window with a few seconds of
warm-up is accurate." Answer: that is the KILL test. Prefix caching for agent sessions has a memory
of hours, so the claim can be false. The test decides it.

**Preprint scoop signals (not evidence).** BPW 2608.00423 compresses serving traces for bottleneck
diagnosis, not for accuracy with warm-up. LLM-Emu 2605.00616 and AgentServeSim 2606.09613 are
simulators.

## 4. Direction I-47: closed-loop validity of agent serving evaluation

**Idea.** Most serving benchmarks send requests at random times (open loop). Agents send the next
request only after the last reply and the tool call finish (closed loop). Check if this changes
which scheduler or cache policy looks best.

**Borrowed from.** Schroeder, Wierman and Harchol-Balter, "Open versus closed: a cautionary tale"
(NSDI 2006). They showed that open and closed models give very different conclusions for
scheduling policies. Closed queueing networks are solved by Mean Value Analysis (Reiser and
Lavenberg, JACM 1980). The Engset model handles a finite set of sources with think time.

**Problem from first principles.** In a closed loop, an overloaded server gets less load
automatically. This hides tail latency and changes the benefit of size-based scheduling. Agent
traffic is closed loop with tool latency and human think time.

**Peer-reviewed prior art.**

| Work | What it solves | What it does not solve |
|---|---|---|
| Schroeder et al. (NSDI 2006) | Open vs closed changes policy conclusions, web workloads | LLM serving, agents |
| Parrot (OSDI 2024), InferCept (ICML 2024), KVFlow (NeurIPS 2025) | Serve agent or augmented-LLM programs and evaluate them with real program loops | No study of whether open-loop conclusions transfer |
| Vidur (MLSys 2024) | Simulation with open-loop trace replay | No closed-loop session model |

**Exact gap.** I did not find peer-reviewed work that tests whether LLM serving policy rankings or
capacity estimates change between open-loop and closed-loop agent workloads with real tool and
think times.

**Research question.** With the same request size distributions, do policy rankings or the maximum
number of sessions at an SLO change by more than 20% between open-loop arrivals and closed-loop
replay of TraceLab sessions? Does an MVA or Engset model predict the closed-loop capacity?

**Decisive Mac test (about 1 week, same simulator as I-46).**
- Build two workloads with matched marginals: Poisson arrivals, and closed-loop session replay
  (next request after the recorded tool time and think time).
- Compare FCFS, length-aware SJF, prefix-aware routing, and LRU vs session-aware eviction.
- Predeclared criteria (heuristics):
  - KILL if policy rankings are the same and metric differences are under 5% at all tested loads.
  - GO if one ranking flips, or if a capacity estimate differs by more than 20%.
  - Otherwise PIVOT.

**Hostile reviewer.** "Agent-serving papers already replay closed-loop programs." Answer: partly true.
The open question is whether open-loop chat benchmarks, which most systems papers use, give the
same answer. The scoop risk is high.

**Preprint scoop signals (not evidence).** XPerf 2608.20370 (closed-loop agent replay framework),
AgentServeSim 2606.09613, TraceLab 2606.30560, an MVA-based serving model 2609.20957.

**Note.** I-46 and I-47 use the same simulator and traces. Together they can form one Mac-only
thesis: "Fast and faithful evaluation of LLM serving for agent workloads."

## 5. Direction I-48: margin-gated quality control for production inference numerics

**Idea.** A serving provider can recompute a small sample of tokens on a reference path and compare
them. Benign floating-point noise flips some tokens all the time, so raw mismatch counts are noisy.
Benign noise flips only tokens where the top two logits are close (small margin). A real bug can
flip tokens with a large margin. So, count only large-margin flips. Then run a CUSUM control chart
on that count.

**Borrowed from.** Statistical process control in manufacturing: the CUSUM chart (Page,
Biometrika 1954) and "rational subgrouping", which separates common-cause noise from special-cause
faults. Also sequential testing (Wald's SPRT, 1945).

**Problem from first principles.**
- FACT (E-23, this repo): between CPU fp32 and MPS fp16 or CPU bf16 on Qwen3-0.6B, 0.26% to 1.07% of
  positions flipped. Every flip had margin at most 2 times the maximum logit difference. The largest
  flip margin was 0.178, while the median margin was about 2.
- Context, not a paper: in an Anthropic engineering postmortem (September 2025), an approximate
  top-k miscompilation sometimes dropped the true argmax token, and it stayed undetected for about
  two weeks.
- INTERPRETATION: a raw mismatch alarm must tolerate about 1% benign flips, so a rare bug can hide
  under that floor. A margin gate removes most of the benign floor.

**Peer-reviewed prior art.**

| Work | What it solves | What it does not solve |
|---|---|---|
| Model Equality Testing (ICLR 2025) | Black-box MMD two-sample test: does an API serve the reference model? 11 of 31 endpoints differed | Offline, distribution level, no token-level numerics, no detection delay |
| An Auditing Test to Detect Behavioral Shift (ICLR 2025) | Anytime-valid sequential test of behavior shift, hundreds of examples | Black-box behavior (toxicity, translation), not numeric bugs |
| TOPLOC (ICML 2025) | Verifies that a provider used the claimed model and precision through hidden-state commitments | Verification of a claim, not monitoring under a sample budget |

**Exact gap.** I did not find peer-reviewed work on provider-side, token-level monitoring that
uses the logit margin to separate benign precision noise from real numeric bugs, with a measured
minimum detectable bug rate at a fixed false-alarm rate. Do not claim margin as a flip-risk signal
as novel (the preprint MarginGate uses it for determinism).

**Research question.** At the same shadow budget and false-alarm rate, does a margin-gated CUSUM
monitor detect argmax-changing bugs at a bug rate at least 10x lower than a raw-mismatch monitor
and an MMD output test? Which bug classes are visible to the margin gate?

**Decisive Mac test (about 2 weeks, reuse `experiments/safedispatch_oracle/`).**
- Benign pairs: CPU fp32 reference vs MPS fp16 and CPU bf16 (as in E-23).
- Injected bug classes at rate rho: (B1) approximate top-k that drops the true argmax,
  (B2) logit corruption that boosts a wrong token, (B3) the E-23 epilogue faults F1 to F4,
  (B4) a wrong scale in one layer.
- Monitors at equal budget: M0 raw mismatch count, M1 margin-gated count, M2 MMD output test
  (Model Equality Testing). Each with CUSUM and with a fixed window.
- Metrics: average run length to a false alarm on benign pairs, and detection delay as a function
  of rho.
- Predeclared criteria (heuristics):
  - KILL if, for B1 and B2, fewer than 20% of the bug flips have a margin above the gate threshold
    (bugs look like noise).
  - KILL if M1 detection delay is within 2x of M0 at the same false-alarm rate.
  - GO if M1 lowers the minimum detectable rho by at least 10x vs M0.
  - Otherwise PIVOT.

**Why this one first.** It reuses the E-23 harness. It links to I-23 (the same contracts) and makes
the grant story stronger. It is about inference numerics, which is the target role. A real
production incident motivates it.

**Hostile reviewer.** "Providers already shadow-compare outputs. A margin filter is a trivial
heuristic." Answer: the contribution is the measured minimum detectable bug rate for each bug
class, with and without the gate. If the bugs are not margin-visible, the KILL rule fires.

**Preprint scoop signals (not evidence).** MarginGate 2605.30218 (margin-triggered verification for
determinism), IRIS 2607.20860 and Behavioral Fingerprints 2603.19022 (black-box endpoint
monitoring), 2606.07624 (sequential inference for LLMs), 2607.20129 (CUSUM monitoring for quantized
small models).

## 6. Direction I-49: failure-clustering-aware recovery provisioning (HOLD)

**Idea.** GPU failures may come in clusters, like earthquake aftershocks. If one failure raises the
chance of more failures soon after, keep more hot spares and check health more often for a short
time after a failure, and fewer at other times.

**Borrowed from.** Self-exciting point processes (Hawkes, Biometrika 1971) and the ETAS aftershock
model (Ogata, JASA 1988), which seismologists use for short-term forecasts.

**Peer-reviewed prior art.**

| Work | What it solves | What it does not solve |
|---|---|---|
| Lazy Checkpointing (DSN 2014) | Uses temporal locality of failures to change HPC checkpoint intervals | LLM training, spare pools, in-memory checkpoints |
| Kokolis et al. (HPCA 2025) | Failure rates and MTTF for 4 million jobs and 150 million A100 GPU-hours | No public node-level logs that I found |
| Cui et al., Story of Two GPUs (SC 2025) | H100 vs A100 error characterization on Delta | Public data release not confirmed |
| AcmeTrace (NSDI 2024) | Public job traces with NODE_FAIL end states and timestamps | No node-level or XID failure events |

**Exact gap.** LLM-specific spare provisioning under failure clustering, with near-free
in-memory checkpoints. The HPC version (DSN 2014) exists, so the gap is narrow.

**Cheap test (weak).** Fit a Poisson model and a Hawkes model to cluster-level NODE_FAIL times in
AcmeTrace. KILL if the branching ratio (the share of failures that earlier failures trigger) is
under 0.1. If it is above 0.3, simulate a hazard-adaptive spare pool against a fixed pool.

**Why HOLD.** The final claim needs node-level failure logs. None is public that I could confirm.
Reopen if the advisor's lab or a past employer can give data, with IP clearance (journal 1.4
item 5).

## 7. Candidates that did not survive

| ID | Candidate | Reason (peer-reviewed evidence) | Preprint scoop signals (not evidence) |
|---|---|---|---|
| I-50 | KV retention across tool and human gaps, ski rental with predictions | InferCept (ICML 2024) chooses preserve, swap or discard by call duration. Pensieve (EuroSys 2025) evicts by inactivity and recompute cost. KVFlow (NeurIPS 2025) uses steps-to-execution. KVCache Cache in the Wild (ATC 2025) and UniCache (SIGMETRICS 2026) do workload-aware eviction | CacheTTL/Continuum 2511.02230, Lachesis 2610.08378, Adaptive KV Retention 2608.30830, UNISON 2609.09643, RR-Evict 2609.32278 |
| I-51 | Speculative tool execution | Speculative Actions (ICLR 2026 oral): up to 55% next-action accuracy, up to 20% lower latency. LLMCompiler (ICML 2024) parallel calls, up to 3.7x faster. Dynamic Speculative Agent Planning (ICLR 2026) | PASTE 2603.18897, SMC 2609.03236, AOSpec 2608.00881, SPORK 2607.03333 |
| I-52 | Reproducible summation for batch invariance | Demmel and Nguyen (ARITH 2013) is the source method. The journal killed generic deterministic inference (I-30). Performance claims need a GPU | Vosti 2609.38981, HEAL 2606.21023, 2609.11356 |
| I-53 | Agent sandbox fork for tree search | Base mechanisms are peer-reviewed and mature: SnowFlock (EuroSys 2009), Firecracker (NSDI 2020). Products ship millisecond fork | Planarian 2609.35366, DeltaBox 2605.22781, StateFork 2609.38648, SWE-Replay 2601.22129 |
| I-54 | Group testing for faulty GPU nodes | Greyhound (ATC 2025) finds fail-slow GPUs and links at over 99% accuracy. Minder (NSDI 2025) detects faulty machines in 3.6 s on average. An NVIDIA readiness tool already uses hierarchical group testing (vendor blog) | 2605.17879, 2606.20374 |
| I-55 | Hedged requests with prefix-cache affinity | First principles: LLM latency variance comes mostly from the request itself (prompt and output length), and a hedge sent to a cold replica loses the prefix cache. Load-aware routing (Llumnix, OSDI 2024; Preble, ICLR 2025) is the simple baseline. No public replica-variance data | none checked |
| I-56 | Certified token stability from rounding-error bounds | Rounding-error theory exists (Blanchard, Higham and Higham, IMA J. Numer. Anal. 2021). The journal killed selective high precision (I-31). Sound bounds over a whole network are too loose. The useful part (the margin invariant) moved into I-48 | LAMP 2601.21623, 2503.10251 |
| I-57 | Fused tool calls (superinstructions) | CodeAct (ICML 2024) merges several actions into one code block. TroVE (ICML 2024) induces reusable functions. LLMCompiler (ICML 2024) | none checked |

## 8. Recommendation (not a decision, D-28)

1. Run I-48 first. It takes about 2 weeks on the Mac, it reuses E-23, and it supports I-23.
2. Then run I-46, with I-47 as a sub-study on the same simulator. This is a Mac-only thesis option
   if GPU access does not come.
3. Keep I-49 on HOLD until node-level failure data is available.
4. Before you commit to a direction, do one preprint scoop check for it. The last 6 months of
   preprints are very active in every area that I searched.

## 9. Sources

Check levels follow journal Section 0.4. "search-only" means that I saw the venue in a search
result or a proceedings page, but I did not read the paper.

**Peer-reviewed (evidence).**

| Source | Venue | Check |
|---|---|---|
| Sherwood, Perelman, Hamerly, Calder, "Automatically characterizing large scale program behavior" (SimPoint) | ASPLOS 2002 | prior-knowledge |
| Wunderlich, Wenisch, Falsafi, Hoe, "SMARTS: Accelerating microarchitecture simulation via rigorous statistical sampling" | ISCA 2003 | search-only |
| Agrawal et al., "Vidur: A large-scale simulation framework for LLM inference" | MLSys 2024 | search-only |
| Aydar, Pinto, Venugopal, Chatzopoulos, "Sampling Where It Matters: Predicting LLM Serving Performance" | EuroMLSys 2026 | checked |
| Ouyang, Qiao, Xing, "UniCache" | SIGMETRICS 2026 (POMACS vol. 10 no. 2) | search-only |
| Wang et al., "KVCache Cache in the Wild" | USENIX ATC 2025 | search-only |
| Schroeder, Wierman, Harchol-Balter, "Open versus closed: a cautionary tale" | NSDI 2006 | search-only |
| Reiser and Lavenberg, "Mean-value analysis of closed multichain queuing networks" | JACM 1980 | prior-knowledge |
| Lin et al., "Parrot" | OSDI 2024 | prior-knowledge |
| Abhyankar et al., "InferCept" | ICML 2024 | search-only |
| Yu et al., "Stateful Large Language Model Serving with Pensieve" | EuroSys 2025 | search-only |
| Pan et al., "KVFlow" | NeurIPS 2025 | search-only |
| Gao, Liang, Guestrin, "Model Equality Testing: Which Model Is This API Serving?" | ICLR 2025 | search-only |
| Richter, He, Minervini, Kusner, "An Auditing Test to Detect Behavioral Shift in Language Models" | ICLR 2025 | search-only |
| Ong et al., "TOPLOC" | ICML 2025 | search-only |
| Page, "Continuous inspection schemes" (CUSUM) | Biometrika 1954 | prior-knowledge |
| Wald, "Sequential tests of statistical hypotheses" (SPRT) | Annals of Mathematical Statistics 1945 | prior-knowledge |
| Hawkes, "Spectra of some self-exciting and mutually exciting point processes" | Biometrika 1971 | prior-knowledge |
| Ogata, "Statistical models for earthquake occurrences and residual analysis for point processes" (ETAS) | JASA 1988 | prior-knowledge |
| Tiwari, Gupta, Vazhkudai, "Lazy Checkpointing" | DSN 2014 | search-only |
| Kokolis et al., "Revisiting Reliability in Large-Scale ML Research Clusters" | HPCA 2025 | full-text (R-07) |
| Cui et al., "Story of Two GPUs" | SC 2025 | search-only |
| Hu et al., "Characterization of LLM Development in the Datacenter" (AcmeTrace) | NSDI 2024 | checked (data README) |
| Ye et al., "Speculative Actions" | ICLR 2026 | search-only |
| Kim et al., "An LLM Compiler for Parallel Function Calling" | ICML 2024 | search-only |
| "Dynamic Speculative Agent Planning" | ICLR 2026 | search-only |
| Demmel and Nguyen, "Fast reproducible floating-point summation" | ARITH 2013 | search-only |
| Lagar-Cavilla et al., "SnowFlock" | EuroSys 2009 | prior-knowledge |
| Agache et al., "Firecracker" | NSDI 2020 | prior-knowledge |
| Wu et al., "Greyhound: Hunting fail-slows in hybrid-parallel training at scale" | USENIX ATC 2025 | search-only |
| Deng et al., "Minder" | NSDI 2025 | search-only |
| Sun et al., "Llumnix" | OSDI 2024 | prior-knowledge |
| Srivatsa et al., "Preble" | ICLR 2025 | prior-knowledge |
| Blanchard, Higham, Higham, "Accurately computing the log-sum-exp and softmax functions" | IMA J. Numer. Anal. 2021 | search-only |
| Wang et al., "CodeAct" | ICML 2024 | checked (harness review S15) |
| Wang et al., "TroVE" | ICML 2024 | prior-knowledge |

**Not peer-reviewed, used only as context.**
- Anthropic, "A postmortem of three recent issues" (engineering report, September 2025): three
  infrastructure bugs, including an approximate top-k miscompilation. Search-only.
- E-23 results from this repo (journal Section 6.1).

**Preprints, scoop signals only (not evidence).** TraceLab 2606.30560, XPerf 2608.20370,
AgentServeSim 2606.09613, BPW 2608.00423, LLM-Emu 2605.00616, 2609.20957, MarginGate 2605.30218,
IRIS 2607.20860, Behavioral Fingerprints 2603.19022, 2606.07624, 2607.20129, AIReSim 2603.07041,
Lablup 504-GPU report 2605.09370, Planarian 2609.35366, DeltaBox 2605.22781, StateFork 2609.38648,
SWE-Replay 2601.22129, CacheTTL 2511.02230, Lachesis 2610.08378, 2608.30830, UNISON 2609.09643,
RR-Evict 2609.32278, PASTE 2603.18897, SMC 2609.03236, AOSpec 2608.00881, SPORK 2607.03333,
Vosti 2609.38981, HEAL 2606.21023, 2609.11356, LAMP 2601.21623, 2503.10251.
