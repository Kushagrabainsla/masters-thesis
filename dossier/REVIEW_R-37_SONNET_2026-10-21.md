# R-38 Sonnet hostile review of R-37 and post-run review of E-37

Date: 2026-10-21. Reviewer: Claude Sonnet (a different model from the main session), read-only, with
fresh web searches and the source clones of R-37. Required by D-29 for each experiment (E-37).

## Verdicts (reviewer)

| Item | Reviewer verdict | Main-session check |
|---|---|---|
| E-37 | CONFIRM_WITH_SCOPE_LIMITS | Accepted. Default SGLang path is the Rust core (see below) |
| I-154 | KILL stands, with two corrections | Accepted |
| I-158 | KILL stands, but for prior art (DySHARP), not arithmetic | Accepted; DySHARP is search-only |
| I-161 | KILL: already in flight upstream | Verified (PRs 38371, 38454 open) |
| I-163 | KILL, downgrade from WEAK | Verified (Hazy code, Cohere blog) |
| New candidates | None survive | Accepted |

## Findings and the main-session verification

1. E-37 fidelity. The S1 stand-in matches the shape of the legacy Python `RadixCache.evict()`. FACT
   (verified 2026-10-21 in SGLang a3d00eff): `mem_cache/registry.py` returns `create_unified_radix_cache`
   by default, and `environ.py:709` sets `SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND = EnvStr("rust")`. PR
   39627 ("Sync Rust TreeCore and make it the default") is merged. The Rust core
   (`rust/sglang-radix-tree/src/components/full.rs`, `evict_device_start`) also clears and refills the
   eviction heap from all evictable leaves on each eviction start. So the O(E) shape is on the default
   path, with a smaller constant (not measured).
2. E-37 S2. The reviewer rebuilt the real vLLM per-block loop (extra keys, list slice): 9.6 ms at 131,072
   tokens and 15.0 ms at 204,800 on an M4, against 7.2 ms and 11.4 ms in the screen. Still "minor" on
   the M4. A slower server core can reach the 20 ms line (HYPOTHESIS).
3. E-37 kept-heap comparison. The 2,250 times ratio is an upper bound: the stand-in has no stale entries
   and no upkeep cost. Independent data: SGLang PR 38371 reports `evict()` p50 of 4.9 ms at 35,000 leaves
   on the Python unified cache and 30 to 33 microseconds with a persistent lazy heap. PR 38454 reports 70
   to 77 ms for each evict at about 41,000 leaves.
4. I-161 prior art. FACT (verified with the GitHub API): PR 38371 (opened 2026-09-07, open) and PR 38454
   (opened 2026-09-08, open) add persistent lazy eviction heaps for the unified radix cache. PR 41690
   (opened 2026-09-29, open) amortizes eviction under sustained pressure. The R-37 text "None found" was
   wrong (Section 8.1 row 64). The reviewer reports that both lazy-heap PRs leave the Rust core unchanged.
5. How often `evict()` runs. The reviewer traced `alloc_paged_token_slots_decode` in
   `mem_cache/allocation.py`: it calls eviction on every decode step, and the allocator evicts only when
   there is a shortfall. So eviction runs on every step under saturation and rarely otherwise.
6. I-154. The reviewer brute-forced 200 random proportionate instances: makespan is order-invariant and
   equals sum(p) + (m - 1) max(p). Two corrections: "any order without idle time is within O(sqrt n)"
   holds only for random order with independent stage times, not for adversarial orders. And the gain
   does not vanish at serving sizes: random order exceeds Johnson's rule by 12 to 14% at n = 8, 7 to 11%
   at n = 32, 3 to 6% at n = 128, and 2% or less at n = 2,048 (two stages, lognormal sigma 1). The
   KILL stands because online serving optimizes latency (SJF and SRPT literature), not batch makespan.
   Finite-buffer pipelines (1F1B in-flight limits) were not checked.
7. I-158. The arithmetic is right for static whole-group multicast. DySHARP ("Accelerating MoE with
   Dynamic In-Switch Computing on Multi-GPUs", Zhang et al., 2026-05-07) uses dynamic multimem
   addressing for each token's target set and reports about 50% less traffic and 1.79 times over COMET in
   training. The reviewer saw only secondary digests; no arXiv ID. Whether it covers inference decode on
   NVL72 is not verified.
8. I-163. FACT (verified 2026-10-21): HazyResearch/Megakernels `megakernels/scheduler.py` has
   `DAG_Node.calc_priority` (longest-cost path) and `assign_dag_to_sms`, a list scheduler with a ready
   heap and an SM free-time heap; the `calc_priority` call is commented out. Round-robin, zig-zag and
   wave modes are beside it. FACT (Cohere blog, 2026-09-08, verified): a greedy cost-based scheduler and a
   brute-force search over hundreds of random schedules each gave about 10% more throughput than
   round-robin on a dense model; Cohere dropped both for MoE because routing makes work dynamic. R-37
   cited this blog but missed that section (Section 8.1 row 65). MPK is arXiv 2512.22219 (v2 2026-06-10);
   the reviewer reports ahead-of-time round-robin plus just-in-time dispatch and future work on global
   policies (the abstract page does not confirm this). Ada-MK (2605.11581) and AutoMegaKernel
   (2606.09682) search schedules offline (reviewer, search-only).
9. I-163 test design. The proposed replay test is circular: task times measured under round-robin
   already include its bandwidth contention, so a shared-bandwidth replay model picks the winner.
10. Missed candidates. Two near misses, neither survives: a distribution-free ski-rental TTL for KV across
    tool calls (closest: Continuum, arXiv 2511.02230, ICLR 2026; a CIDR 2025 ski-rental caching paper for
    database pages), and ordering tool definitions by change rate (engineering; Qwen-code stable tool
    sort, deferred tool loading). Same family as I-149: vLLM `scheduler.py` re-runs
    `get_computed_blocks` each step for a blocked waiting request (not measured).

## Outcome

All 16 R-37 candidates are KILL. The new-gap set is empty again. No experiment is planned from R-37.
