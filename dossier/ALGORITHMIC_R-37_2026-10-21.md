# R-37 Algorithmic-optimization sweep of the ML infrastructure stack

Date: 2026-10-21. Requested by the user: "think fundamentally like you are solving competitive algorithmic
problems" and find layers where the algorithm is not optimal. Extra source: the PyTorch Conference North
America 2026 schedule (2026-10-20 to 2026-10-21, San Jose).

Skill: `research-gap-dossier`. Pass 1: main session (code reading plus literature). Pass 2: targeted
searches for each candidate that passed pass 1, then a different-model (Sonnet) hostile review (R-38).

Labels: FACT (a source or a run showed it), INTERPRETATION (my reading), HYPOTHESIS (a prediction a test
can refute).

Read Section 7 first: the R-38 review changed four verdicts. After R-38, all 16 candidates are KILL.

## 1. Summary

- The sweep generated 16 candidates (I-149 to I-164) with four algorithmic lenses (Section 2).
- Before the search, 13 of them looked plausible. After pass 1 and pass 2, 14 are KILL.
- Five of the KILLs come from current source code: the shipping engine already uses the better algorithm,
  or it fixed the slow path in a newer code path.
- Five KILLs come from 2026 preprints or RFCs that already contain the same mechanism.
- Three KILLs come from short theory arguments. These arguments close whole families of ideas
  (proportionate flow shop, large-batch flow-shop convergence, multicast ingress arithmetic).
- One candidate is WEAK and GPU-gated: I-163, critical-path list scheduling for decode megakernels.
- One candidate is an engineering upstream-PR target, not research: I-161, a kept eviction heap for
  SGLang's radix cache. E-37 measured 4.6 ms for each `evict()` call at 50,000 evictable leaves.
- Lesson (INTERPRETATION): in October 2026, the hot paths of vLLM, SGLang and PyTorch DCP already use
  near-optimal algorithms, or a preprint from the last six months proposes one. The algorithmic lens
  finds real slow paths, but the community finds them within weeks.

## 2. Method

Lenses (the competitive-programming view):

1. Full rescans where an incremental update is enough: work per step is O(n), but only O(changes) is
   needed.
2. Greedy or round-robin where an exact algorithm is cheap at the real instance size (flow, LP, DP,
   exchange-argument orderings).
3. Online policies where the future is known: the per-step graph repeats, the batch is known, or the
   training data order is fixed.
4. Theory checks: is the heuristic already optimal, or is the objective insensitive to the decision?

Code read at these commits (FACT, cloned 2026-10-21):

| Repository | Commit | Paths read |
|---|---|---|
| sgl-project/sglang | a3d00eff58b6c22ea7539f882cc10c537634200b | `managers/schedule_policy.py`, `mem_cache/radix_cache.py`, `mem_cache/evict_policy.py`, `eplb/` |
| vllm-project/vllm | 90c4a1a73eecea0428ee4303ad96c9fdeaac3f80 | `v1/core/kv_cache_utils.py`, `v1/engine/core.py`, `utils/hashing.py`, `config/vllm.py`, `v1/sample/ops/penalties.py`, `v1/worker/gpu/sample/penalties.py`, `distributed/kv_transfer/kv_connector/v1/nixl/tp_mapping.py`, `renderers/` |
| pytorch/pytorch | 6dbae335eb883eb25b41e06a0723895f9e20f7bf | `torch/distributed/checkpoint/_dedup_save_plans.py`, `planner_helpers.py` |

## 3. PyTorch Conference 2026 sessions that point at algorithmic decisions

FACT (schedule page, read 2026-10-21). The sessions below name a decision problem. Each one is either
a candidate here or a threat to one.

| Session | Decision problem | Used in |
|---|---|---|
| Parametrized Dynamic Shape CUDA Graphs | Which shapes get a captured graph | I-151 (threat) |
| FlexShard: Pluggable Sharding Placements for Matrix Optimizers and Block-wise Quantization | Shard placement under block and whole-matrix constraints | I-162 (threat) |
| Beyond AdamW: Muon, Dion, and Orthogonalized Optimizer Variants | Distributed Newton-Schulz work | I-162 |
| Elastic Expert Parallelism in vLLM | Expert remapping on scale change | I-156 context |
| State-of-the-Art KV Transfer for Disaggregated LLM Serving in vLLM | Source rank choice for KV pulls | I-157 |
| Prefix Caching for Autoregressive Stages in Multi-Stage Pipelines | Staged pipelines | I-154 context |
| Demo Theater: TPU Megakernels; Helion and CuteDSL sessions | Persistent-kernel task order | I-163 context |
| Smarter Autotuning for Kernels (Bayesian optimization to LLM-guided search); PerfModel | Config search | Not registered: crowded |
| Self-Healing PyTorch Training Through Live GPU Migration; MCCL; Never Hit Zero | Fault tolerance | Not registered: I-135 line, NVIDIA building it |
| Speeding Up torch.compile: A New FakeTensor; Nested Graph Breaks; Precompile for Training | Compile-time work | Not registered: engineering, I-130 line |

## 4. Candidates and verdicts

### I-149 Incremental cache-aware ordering of the waiting queue. KILL.

- FACT (SGLang a3d00eff): `WAITING_QUEUE_PREFIX_MATCH_MAX = 128` with the comment "Arbitrary". If more
  than 128 requests wait, LPM and HRRN switch to FCFS ("Turn off the expensive prefix matching and
  sorting when the #queue is large"). Below the limit, each call matches every waiting request against
  the radix tree again. The cost of one call is O(W x L / page).
- INTERPRETATION: the scheduler drops cache-aware order exactly in overload, when order matters most.
  A competitive-programming fix keeps each request's match length and changes it only when the tree
  changes on its path.
- Why closed: PEEK (2607.02525) keeps an incremental radix tree over the pending queue with O(D) insert
  and remove. "Requests of a Feather" (2605.06046) uses cumulative prefix hashes and a lazy min-heap
  and reports 50 to 90% of latency in tree work. DLPM (2501.14312) reports the matching overhead.

### I-150 Incremental front-end encoding for agent turns. KILL.

- FACT (vLLM 90c4a1a): the renderer has no tokenization cache across turns. The engine core hashes all
  prompt blocks for each request in its input thread (`preprocess_add_request`).
- INTERPRETATION: prefix caching makes GPU prefill O(new tokens), but the CPU path stays O(context) per
  turn, so it is quadratic in turns for one session.
- Why closed: vLLM RFC 47582 (incremental prompt encoding: about 0.5 s for one full encode of a
  200K-token history; median multi-turn TTFT 0.92 s to 0.47 s in one vendor report). TokTier
  (2607.29678): exact stateful CPU plus GPU tokenization for agent serving. Incremental BPE
  (2605.30813). The RFC states that SGLang's SMG front end has an L1 prefix tokenization cache.
- E-37 S2 (FACT, M4, one core): vLLM-style `sha256(pickle)` block hashing costs 7.2 ms for a
  131,072-token prompt and 11.4 ms for 204,800 tokens. This is "minor" under the predeclared rule.

### I-151 Optimal CUDA-graph capture sizes (one-dimensional DP bucketing). KILL.

- FACT (vLLM 90c4a1a `config/vllm.py`): default sizes are 1, 2, 4, then every 8 up to 256, then every 16.
  The worst padding is 7 or 15 slots.
- INTERPRETATION: decode at these batch sizes reads the weights once for the whole batch, so a few
  padded slots cost little. The oracle gain of an optimal bucketing is small.
- Threat: the PyTorch Conference session "Parametrized Dynamic Shape CUDA Graphs" removes the
  need for one graph for each size.

### I-152 Rank balancing for data-parallel attention in DP plus EP decode. KILL.

- Problem: with DP attention, each layer waits for the slowest rank, and attention cost follows the
  rank's KV tokens.
- Why closed: TensorRT-LLM "ADP Balance Strategy" blog (balance ratio, per-iteration balance);
  NanoCP (2605.21100): token-level KV placement and dynamic context parallelism for each request.

### I-153 Exact per-step replica choice for redundant experts. KILL (shipped).

- FACT (SGLang a3d00eff): `ep_dispatch_algorithm` is one of `static`, `dynamic`, `fake`, `lp`, with an
  LP solver in `eplb/lplb_solver.py`.
- Also: LLEP (2601.17111, ICML 2026), EasyBalance (2608.07964).

### I-154 Flow-shop ordering (Johnson's rule) for staged pipelines. KILL by theory and scale.

- Targets: pipeline-parallel microbatch order, encode then prefill in EPD serving, prefill then KV
  transfer in P/D serving.
- FACT (scheduling theory, proportionate flow shop): if each job has the same processing time p_j on
  every stage, the makespan of a permutation schedule on m stages is sum(p_j) + (m - 1) x max(p_j). It
  does not depend on the order. Balanced pipeline stages give this case for forward-only passes.
- INTERPRETATION: 1F1B interleaves backward passes (about 2 x forward), so the case is not exactly
  proportionate. Variable-length imbalance in PP is covered by WLB-LLM (OSDI 2025) through packing.
- INTERPRETATION: for an offline EPD batch with n jobs and random stage times, any order without idle
  time is within O(sqrt(n)) of max(sum p1, sum p2), so the relative gain of Johnson's rule goes to zero
  as n grows. Online serving targets latency, and buffers between the stages remove most idle time.
- INTERPRETATION: MLA KV is about 70 KB for each token for DeepSeek-V3, so KV transfer per token is much
  shorter than prefill per token. Transfer is not a bottleneck stage.
- Search (2026-10-21): no public work applies Johnson's rule to EPD. Absence is not proof (rule 6).

### I-155 Weighted tree pebbling for offline batches with shared prefixes. KILL.

- Why closed: SGLang paper (2312.07104) proves that DFS order of the radix tree is optimal for hit
  rate when the cache holds the longest request. BatchLLM (2412.03594), BlendServe (2411.16102), and
  "Requests of a Feather" (2605.06046, batch size against prefix homogeneity) cover the batch-width part.

### I-156 Minimal-movement resharding on elastic resize. KILL.

- INTERPRETATION (arithmetic): FSDP-style even contiguous shards that go from N to N - 1 ranks move
  about S / 2 of the state S in total, and the largest receive is about S / N. A minimal plan moves S / N.
- Why closed: Tenplex (SOSP 2024) computes minimal reconfiguration plans. The absolute time is small:
  one 17 GB shard at 50 GB/s is about 0.35 s, much less than restart time.

### I-157 Flow-optimal source choice for RL weight sync and MLA KV pulls. KILL.

- FACT (vLLM 90c4a1a `tp_mapping.py`): for MLA, decode rank r reads from remote rank
  r x remote_tp // tp, which spreads reads across the prefill ranks.
- Weight sync: the Perplexity blog reports about 1.3 s for Kimi-K2 with a static schedule; Moonshot
  checkpoint-engine is public. INTERPRETATION: sync is a small share of an RL step, so the oracle gain
  of a flow-optimal plan is small.

### I-158 NVLS multicast for MoE dispatch on NVL72. KILL by arithmetic.

- INTERPRETATION: a multicast group delivers to every member. Map DeepSeek-V3 group-limited routing
  (top 4 of 8 groups, 8 experts) to 9-GPU groups on NVL72. Each token then needs 4 multicasts instead
  of 8 unicasts (egress halves). But it reaches 36 GPUs instead of 8 (ingress rises 4.5 times).

### I-159 Incremental counts for repetition penalties. KILL (fixed).

- FACT (vLLM 90c4a1a): the old path `v1/sample/ops/penalties.py` builds a padded tensor of all output
  tokens on each step (O(B x L) for each step). The new GPU runner keeps a persistent
  `output_bin_counts` tensor (`v1/worker/gpu/sample/penalties.py`).

### I-160 DCP planning complexity. KILL.

- FACT (PyTorch 6dbae335): `create_read_items_for_chunk_list` uses a sweep line with `bisect`.
  `dedup_save_plans` gives each duplicate item to the plan with the smallest size so far, in dict order,
  without sorting by size (list scheduling, not LPT).
- INTERPRETATION: sorting by size (LPT) improves the worst case from about 2 to 4/3 of optimal. But most
  FSDP shards are unique, so the gain is small. Engineering only.

### I-161 Kept eviction heap for radix caches. KILL as research; engineering PR candidate.

- FACT (SGLang a3d00eff `radix_cache.py` `evict()`): each call copies all evictable leaves into a list,
  computes a priority for each with a Python method call, and runs `heapify`. The call is O(E) even
  when it frees a few pages.
- E-37 S1 (FACT, M4, stand-in nodes): 0.88 ms at E = 10,000; 4.6 ms at E = 50,000; 11.2 ms at
  E = 100,000. A heap kept between calls pops 4 leaves in about 2 microseconds.
- Limits: the stand-in does not count the cost of keeping the heap current (one push for each touched
  leaf, O(log E)). E-37 did not measure how often `evict()` runs in production.
- Next step (optional): an upstream PR with a lazy-deletion heap and an SGLang benchmark under memory
  pressure.

### I-162 Work placement for Muon and other matrix optimizers. KILL.

- Why closed: NorMuon (2510.05491, size-sorted round-robin owners), Motif 2 (FLOP-sorted round-robin),
  INTELLECT-3 (overlapped round-robin), Dion3 (communication wall), MatrixFSDP (2607.05895, owner rank
  for each matrix), and the FlexShard session.

### I-163 Critical-path list scheduling for decode megakernels. WEAK, MEASURE-FIRST, GPU for the claim.

- Problem: a decode megakernel runs the task graph of one decode step on all SMs. The graph is the same
  on every step for a given batch size, so the future is known. Current systems place tasks with
  round-robin or wave order.
- FACT: the Mirage Persistent Kernel author writes that MPK uses simple round-robin and lists
  priority-aware policies as future work (Medium post). Cohere's North Mini megakernel deals waves to
  SMs round-robin, adds work stealing, and reports that wave order alone changes performance.
  ForgeMegakernel (2609.12379) reaches 50.5 to 85.9% memory-bandwidth utilization. Event Tensor
  (2604.13327, MLSys 2026) applies static and dynamic scheduling transformations. Weave (2609.21483)
  does dynamic SM scheduling in an MoE megakernel. Fleet (2604.15379) uses scheduler workgroups.
- Search (2026-10-21, three queries): no public work applies HEFT upward rank or critical-path priority
  to SM task order in a decode megakernel. Absence is not proof.
- Research question: for a decode megakernel on one Hopper GPU, does a static schedule from
  critical-path list scheduling (HEFT upward rank on measured task times) reduce step latency compared
  with round-robin wave order plus work stealing, by at least 5% at batch 1 to 64?
- Hostile rejection (shortest): "Decode is memory-bound. Any schedule that keeps HBM busy is optimal,
  and work stealing removes the tail. The MPK team will add priorities in a release."
- Cheapest decisive test: dump the task graph and measured task times of one MPK decode step on one
  H100, then run an offline simulator that replays the measured times with a shared-bandwidth limit.
  Compare round-robin, wave order with stealing, HEFT, and the lower bound
  max(total bytes / bandwidth, critical path). The simulator must not set the task times from the
  schedule (R-23 and R-25 lesson on circular models).
- Kill rule (heuristic): if HEFT is within 3% of wave order plus stealing at all batch sizes, KILL.
- Confidence in the gap: low. Scoop risk: high.

### I-164 Scan-resistant eviction for cyclic agent sessions. KILL.

- INTERPRETATION: if agent sessions come back in about the same order as they left for tool calls, LRU
  evicts the session that returns next (the classic LRU cliff on cyclic access).
- Why closed: ARC-style recency and frequency KV caching (2606.21238), Continuum TTLs, KVFlow, and the
  closed I-45, I-50 and I-87.

## 5. Decision table

| ID | Exact question | Prior work that answers it | Status | Confidence |
|---|---|---|---|---|
| I-149 | Keep LPM order in overload with incremental matching | PEEK, Requests of a Feather, DLPM | KILL | n/a |
| I-150 | O(new tokens) front-end work per agent turn | vLLM RFC 47582, TokTier, Incremental BPE | KILL | n/a |
| I-151 | Optimal capture sizes | vLLM step sizes; parametrized graphs | KILL | n/a |
| I-152 | DP-attention balance | TRT-LLM ADP, NanoCP | KILL | n/a |
| I-153 | Per-step replica LP | SGLang `lp` dispatch, LLEP | KILL | n/a |
| I-154 | Johnson's rule for staged pipelines | Proportionate flow shop theorem; scale | KILL | n/a |
| I-155 | Tree pebbling order | SGLang DFS theorem, BatchLLM, BlendServe | KILL | n/a |
| I-156 | Minimal-movement resharding | Tenplex | KILL | n/a |
| I-157 | Flow-optimal transfer sources | vLLM NIXL mapping; small oracle | KILL | n/a |
| I-158 | NVLS multicast dispatch | Ingress arithmetic | KILL | n/a |
| I-159 | Incremental penalty counts | vLLM GPU runner | KILL | n/a |
| I-160 | DCP planning | Sweep line in PyTorch | KILL | n/a |
| I-161 | Kept eviction heap | None found; engineering | KILL as research, PR candidate | n/a |
| I-162 | Matrix-optimizer placement | NorMuon, MatrixFSDP, FlexShard | KILL | n/a |
| I-163 | Critical-path order in decode megakernels | MPK, Cohere, Event Tensor (partial) | WEAK, MEASURE-FIRST, GPU | low |
| I-164 | Scan-resistant agent KV eviction | 2606.21238, Continuum | KILL | n/a |

## 6. Sources (primary first)

- SGLang, vLLM and PyTorch source at the commits in Section 2 (full-text).
- PEEK, arXiv 2607.02525. Requests of a Feather Must Flock Together, arXiv 2605.06046. DLPM, arXiv
  2501.14312. SGLang, arXiv 2312.07104. BatchLLM, arXiv 2412.03594. BlendServe, arXiv 2411.16102.
- vLLM RFC 47582 (github.com/vllm-project/vllm/issues/47582). TokTier, arXiv 2607.29678. Incremental
  BPE Tokenization, arXiv 2605.30813.
- TensorRT-LLM ADP Balance Strategy blog (nvidia.github.io/TensorRT-LLM, blog10). NanoCP, arXiv 2605.21100.
- LLEP, arXiv 2601.17111. EasyBalance, arXiv 2608.07964.
- Johnson's rule note, arXiv 2512.06119. EPD Disaggregation, arXiv 2501.05460 (ICML 2025).
- Tenplex (SOSP 2024). Perplexity weight-transfer blog (2025). Moonshot checkpoint-engine (GitHub).
- NorMuon, arXiv 2510.05491. MatrixFSDP, arXiv 2607.05895. Motif 2, arXiv 2511.07464. INTELLECT-3,
  arXiv 2512.16144.
- Mirage MPK post (zhihaojia.medium.com). Cohere megakernel blog (cohere.com/blog/megakernels).
  ForgeMegakernel, arXiv 2609.12379. Event Tensor, arXiv 2604.13327. Weave, arXiv 2609.21483. Fleet,
  arXiv 2604.15379.
- Recency/Frequency Adaptive KV Caching, arXiv 2606.21238.
- PyTorch Conference North America 2026 schedule
  (events.linuxfoundation.org/pytorch-conference-north-america/program/schedule/).

## 7. Corrections after R-38 (2026-10-21)

The Sonnet review (R-38, `dossier/REVIEW_R-37_SONNET_2026-10-21.md`) changed four verdicts. The text in
Sections 1 to 5 above is kept as written. These corrections win.

| ID | R-37 verdict | After R-38 | Reason (verified by the main session unless marked) |
|---|---|---|---|
| I-161 | KILL as research, engineering PR candidate ("None found") | KILL | SGLang PRs 38371 and 38454 (open since 2026-09-07 and 09-08) add persistent lazy eviction heaps; PR 41690 amortizes eviction. The default path is the Rust core, which also rebuilds the heap on each eviction start |
| I-163 | WEAK, MEASURE-FIRST, GPU | KILL | Hazy Megakernels `scheduler.py` already has a list scheduler with a critical-path priority function; Cohere's blog reports about 10% from cost-based and brute-force schedulers on a dense model and dropped them for MoE. The proposed replay test was circular |
| I-154 | KILL by theory and scale | KILL, reason corrected | The proportionate flow shop result holds. The O(sqrt n) claim holds only for random order. The gain is 3 to 14% at n of 8 to 128, so the right reason is the objective: online serving optimizes latency, not batch makespan |
| I-158 | KILL by arithmetic | KILL, reason corrected | The arithmetic holds only for static whole-group multicast. DySHARP (2026-05-07, search-only) uses dynamic multimem addressing for each token's target set |

E-37: CONFIRM_WITH_SCOPE_LIMITS. The S1 stand-in models the legacy Python class. The real vLLM hashing
loop is about 9.6 ms at 131,072 tokens (reviewer), still minor on the M4.

Final count: 16 of 16 KILL. No survivor. No experiment is planned from R-37.
