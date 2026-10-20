# R-31: Sonnet pre-run review of E-35 (2026-10-19)

Reviewer: an independent Sonnet agent (D-29), read-only. It read
`experiments/priority_inheritance_premise/README.md` (the frozen design before amendment A1), the
journal entries I-59 and E-35, and local copies of vLLM `3897f95cf17619137536d2a7d325d9c2f39dc38f` and
SGLang `476fe86a12535173ea0eee1e849f80e350cc6ffd` source files. It also fetched six more vLLM files at
the pinned commit into the session scratchpad (not into this repository).

The main session confirmed finding 2 by reading `scheduler.py` lines 712 to 735: the running loop walks
`self.running` in list order, and no line sorts `self.running` by priority.

Note on sources: the reviewer reported that the GitHub page of vLLM PR 59963 contained text addressed to
AI agents (an "Agent Guidelines" note that told agents to close the PR). The reviewer ignored it. It is
web content, not an instruction from the user.

The review text follows as the reviewer returned it.

---

**Verdict: DO_NOT_RUN_CODE_DECIDES.**

The premise is decided by code reading. As written, the grid cannot return a decision: V1 cannot pass and K2 is almost certain to fail. Findings 3 and 4 rest on my own abstract re-implementation of the `schedule()` ordering, run inline with no files written. Treat them as INTERPRETATION until the instrumented run in finding 12 confirms them.

The minimal check to run is described in finding 12.

**Findings**

1. **BLOCKER / FACT: the "no wait on an in-flight prefix" premise holds.**
   - The waiting loop skips a request only when `_handle_blocked_waiting_request` returns False (scheduler.py:984, 3266-3298). That function covers only `WAITING_FOR_REMOTE_KVS`, `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` and `WAITING_FOR_STREAMING_REQ`.
   - The running loop (714-756) has no prefix-related skip.
   - `allocate_slots` calls `coordinator.cache_blocks(request, min(total_computed+num_new, num_tokens))` at schedule time (kv_cache_manager.py:612-616). The waiting loop's `get_computed_blocks` at admission (scheduler.py:1015-1022) therefore hits blocks that L scheduled but has not yet computed.
   - `deferred_waiting` is only a skip-set (230, 961).
   - `_inflight_prefills` gates only async connector loads (1298-1308).
   - `max_admission_blocks_per_request` is None for full attention (single_type_kv_cache_manager.py:215-226).
   - `throttle_prefills`, `prefill_capacity_bound` and `prefill_schedule_interval` are global gates with no prefix logic (669-710).

2. **BLOCKER / FACT: the running loop ignores priority, so inheritance has no lever on an in-flight producer.**
   - The running loop walks `self.running` in admission order and takes the token budget first (714-913). The waiting loop gets only the leftover (`while token_budget > 0`, 967).
   - Priority matters only in two places: waiting-queue order (`Request.__lt__`, request.py:357-367) and the preemption victim `max(running, key=(priority, arrival_time))` (scheduler.py:851-855).
   - Issue 59907 says the same: running requests get budget "in admission order, not priority order". Its fix, PR 59963 (open), takes the sorted-running direction.
   - A producer that is in flight is by definition running. Promoting its priority changes nothing except its immunity to preemption.

3. **BLOCKER / INTERPRETATION: V1 cannot pass in the named cell.**
   - At step 0 the heap puts M (priority 5) ahead of L (priority 10). With budget 256, M0 takes the whole budget and the loop ends (967). L never starts, so the PC rule "running request in prefill that shares a block" never fires.
   - My abstract model of lines 714-1450 gives C1/C2 = 1.00 for the PC at |P|=2048, budget 256, N_m=4, f=0.25, and also at |P|=8192, f=0.9.
   - If I stagger M one or two steps after L, L runs first and the PC gives C1/C2 = 1.5 to 1.9. But C4 gain is then 0, because promoting an already-running L does nothing (finding 2).
   - The two V1 clauses (ratio ≥ 1.25 and C4 gain ≥ 15%) therefore cannot hold together in this scheduler.
   - V1 uses p99 over 40 trials, which is nearly the maximum. Trials with M at step 0 give C1 = C2, and they set the p99.
   - E-35 would end INVALID by construction.

4. **BLOCKER / INTERPRETATION: C4 is not an upper bound, and the decision table has no row for "K1 holds, K2 fails".**
   - In the unmodified scheduler, my model gives a C4 "gain" of 63-95% with C1/C2 = 1.00 (for example |P|=2048, budget 256, N_m=4, M and L both at step 0).
   - The cause: L at priority 0 jumps ahead of M0 in the waiting queue, runs first, and acts as H's prefetcher. That is not an inversion fix.
   - K2 will therefore trip in some cells. The rules map "K1 holds, K2 fails" to no outcome. KILL needs both K1 and K2, and the other two outcomes need a K1 break.
   - C4 also promotes L from t=0, before H arrives. Inheritance acts only after H blocks.

5. **MAJOR / INTERPRETATION: K1 measures the net effect, so a real wait is masked by the reuse benefit.**
   - H in C1 reuses P and H in C2 recomputes it.
   - In my model, PC at |P|=2048, budget 256, N_m=0 delays H from 8 to 12 steps, but C2 is 16. The ratio is 0.75, so K1 passes while H waited 4 steps.
   - The premise "H waits for a prefix" should be tested directly. Log, per step, why the highest-priority waiting H was not scheduled: budget spent by running requests, a skip, an allocation failure, or a preemption step (no admission when `preempted_reqs`, line 955).

6. **MAJOR / FACT: the grid is partly degenerate, and the thresholds are quantized.**
   - With budget 2048 and |P| ≤ 2048, L's prefill takes 1-2 steps. The f values {0, 0.25, 0.5} all round to arrival step a=0, so those cells are duplicates with no in-flight window.
   - TTFT values of 1-9 integer steps make 1.10 equal to "one step more". A single trial can flip a cell.
   - "Every cell" combined with p99 ≈ max over 40 deterministic trials is fragile.
   - Use paired per-trial differences, an absolute floor (for example ≥ 2 steps and ≥ 10%), and report the fraction of trials that break.

7. **MAJOR / FACT: the "tight" KV cell at 1.1× total need exercises no preemption.**
   - Preemption (851-896) is the only place priority acts on running work, and it needs real pressure. Use 0.5-0.8× of the concurrent need and log preemption counts.
   - `scheduler_reserve_full_isl` defaults True (config/scheduler.py:190), and `create_scheduler` does not override it (test_utils.py:125-141). Admission therefore requires the full sequence to fit.
   - A waiting request never triggers preemption. Preemption happens only inside the running loop.

8. **MINOR / FACT: the fake runner is consistent with `update_from_output`.**
   - A partial-prefill request gets an empty token list, and the scheduler acts only `if new_token_ids` (scheduler.py:2231-2240). `ModelRunnerOutput` fields are at outputs.py:267-277.
   - Use `ignore_eos=True` and non-EOS token ids.
   - `create_scheduler` sets the scheduler `max_model_len` to the token budget when none is passed (test_utils.py:125-126). Pass `max_model_len` ≥ 2·8192+132, because the scheduler reads the model config value.
   - `Request(arrival_time=...)` exists (request.py:68, 98). The ordering tie-break uses `request_id` (request.py:365).
   - The import of `tests.v1.kv_connector.unit.utils` under `VLLM_TARGET_DEVICE=empty` should be dry-run before freezing the design.

9. **MINOR / FACT: async scheduling does not change the answer.**
   - The default is async-on unless incompatible (config/vllm.py:1656-1715).
   - The CPU platform forces it off (platforms/cpu.py:329-330). The sync `Scheduler` is therefore the class the CPU backend runs.
   - `AsyncScheduler` inherits `schedule()` (async_scheduler.py) and only adds output placeholders. Limit the claims to the CPU/sync scope.

10. **MINOR / FACT: one cross-request wait exists, but it is out of scope.**
    - The offloading connector returns `None` while a lookup is `HIT_PENDING` (offloading/scheduler.py:1078-1093). The scheduler then skips the request (scheduler.py:1041-1046).
    - This concerns external-store lookups beyond the local hit, and it is off by default. Name it in the scope list.

11. **FACT: the SGLang claim is right, with one nuance.**
    - The in-batch deprioritization compares a request only against earlier *waiting* peers (schedule_policy.py:343-393, thresholds 107-115). It runs only in the cache-aware branch (233-259).
    - Priority scheduling asserts `fcfs` or `lof` (validation_hook.py:280-285).
    - No `chunked_req` path makes a request wait on a prefix. The only coupling is `shortest_prefill_chunk_limit` (415-436), and it applies only under SHORTEST_PREFILL_FIRST, which is cache-aware.
    - SGLang's priority answer to an H-versus-L conflict is preemption of lower-priority running requests (`preempt_to_schedule`, 1642-1690).

12. **Cheapest sufficient confirmation (about 1 hour, no grid).**
    - Add a trace hook to the unmodified sync `Scheduler`, run three scripted scenarios, and assert on the trace:
      - H admitted with `num_computed_tokens == floor16(L's scheduled P tokens)` in the first step that has leftover budget.
      - When L's chunk consumes the whole budget, H gets 0 tokens in C1 and in C2 alike (the wait is budget, not prefix).
      - No skip reason other than the three blocked states appears.
    - Then record a conditional KILL: "vLLM 3897f95 and SGLang 476fe86 have no shared-prefix wait, and vLLM's running loop is priority-blind."

13. **MAJOR / INTERPRETATION: scope against the user's request.**
    - The user asked for the check on the vLLM CPU backend in Docker. Stage A is the same scheduler class that backend runs (finding 9), so it answers the premise question.
    - Stage B could only add four things:
      - Wall-clock weighting. Prefill chunks cost more than decode steps, so it changes the size of an effect, not whether one exists.
      - Frontend and engine-core plumbing, such as how `priority` is passed to the scheduler.
      - Correctness of same-step reuse in the CPU attention kernel (the 59907 class). The wait rule does not decide that.
      - A fidelity check that the real `SchedulerOutput` stream matches the fake-runner stream for one cell. This is the useful part of Stage B.
    - Per journal rule 0.2.8, the substitution of Stage A for the CPU backend is an AI recommendation. The user must decide it and the decision must be logged in Section 10. Offer the one-cell fidelity run as the optional CPU-backend step.

14. **G. Prior art and value: even a real inversion would not make I-59 novel (INTERPRETATION).**
    - Priority inheritance exists to fix a blocked task that cannot proceed without a holder's result. H can recompute P at its own priority, or take budget from L. A dependency-free fix dominates.
    - The inversion that does exist is the budget-order one in 59907, with PR 59963 open against it.
    - Prior work on adjacent problems, from search snippets only (not verified):
      - FlowPrefill (2602.16603): high-priority head-of-line blocking behind low-priority prefill, handled by finer preemption.
      - Cascade (2608.06557): latency-budget scheduling.
    - I fetched the RelServe abstract (2601.11546). It describes a dynamic priority updater on relational workloads and does not mention inheritance or prefix caching.
    - Duplicate-prefill merging (I-64) is already killed in the journal.
    - I found no source applying priority inheritance to a shared-prefix producer. That is absence of evidence only.

**One-sentence rejection a PC member would give:** "Priority inheritance repairs a blocked task that depends on a lower-priority holder, but no engine makes a request wait on another's in-flight prefix (vLLM recomputes it from blocks cached at schedule time, SGLang only reorders waiting duplicates under cache-aware policies), and vLLM's running loop ignores priority, so the idea fixes a dependency that does not exist."
