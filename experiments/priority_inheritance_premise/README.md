# E-35: I-59 premise check (priority inversion through a shared-prefix producer)

Status: DONE (2026-10-20). Frozen decision: KILL (run 2, `results/trace-20261020T060157Z/`). Post-run
review R-32 (`dossier/REVIEW_E-35_POSTRUN_2026-10-20.md`): CONFIRM_WITH_SCOPE_LIMITS. The conclusion is
"no shared-prefix dependency", not "no priority inversion" (Section 12). History: predeclared, then
amended by A1 (Section 10, after the R-31 pre-run review) and A2 (Section 11, after run 1 was INVALID).
The Stage A grid in Sections 3, 5 and 6 is withdrawn and did not run.

Labels: FACT (source read at the pinned commit), INTERPRETATION, HYPOTHESIS.

## 1. Question

In current vLLM, with priority scheduling and prefix caching on, does a high-priority request get
slower because a lower-priority request is computing the same prefix? If yes, how much can priority
inheritance recover?

Decision that this changes: I-59 GO (Stage B, then a mechanism) or I-59 KILL.

## 2. Rubber duck

**What I-59 claims.** Priority inheritance is an operating-system mechanism. A low-priority task that
holds a resource gets the priority of a high-priority task that waits for that resource. I-59 says
that a shared prefix is such a resource: the high-priority request H waits for the low-priority
request L to finish the prefix P, and medium-priority work M starves L. That is the classic
inversion triad (H, M, L). Inheritance lifts L above M while H waits.

**What the claim needs.** A real dependency: H must wait for L. Without a wait there is no inversion,
because H can compute P itself at its own priority.

**What the code shows (FACT, vLLM `3897f95cf17619137536d2a7d325d9c2f39dc38f`).**
- `vllm/v1/core/sched/scheduler.py`, waiting loop (about lines 954 to 1075): a waiting request is
  skipped only when `_handle_blocked_waiting_request` (about line 3266) returns False. The only
  blocked states are `WAITING_FOR_REMOTE_KVS`, `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` and
  `WAITING_FOR_STREAMING_REQ`. No state means "another request is computing my prefix".
- `vllm/v1/core/kv_cache_manager.py`, `allocate_slots`: the manager calls
  `cache_blocks(request, min(total_computed_tokens + num_new_tokens, request.num_tokens))` when it
  schedules the tokens, not when they are computed. A later request therefore gets prefix hits on
  every full block that L has scheduled, and it computes the rest of P itself.
- `scheduler.py` about lines 850 to 857: under KV pressure the PRIORITY policy preempts the running
  request with the largest `(priority, arrival_time)`.
- vLLM issue 59907 (open) reports a different inversion: running requests get the token budget in
  admission order, so an earlier low-priority request can take the budget first. That effect does not
  depend on a shared prefix.

**What the code shows (FACT, SGLang `476fe86a12535173ea0eee1e849f80e350cc6ffd`).**
- `schedule_policy.py` `_compute_prefix_matches` marks a waiting request "temporarily deprioritized"
  when it shares at least 32 tokens with an earlier waiting request. This is a real wait on a shared
  prefix.
- That logic runs only for the cache-aware policies (LPM, DFS weight, HRRN, shortest-prefill-first).
- `arg_groups/validation_hook.py` asserts that priority scheduling needs `fcfs` or `lof`. So the
  shared-prefix wait and priority scheduling cannot run together in SGLang.

**Prediction (HYPOTHESIS).** vLLM has no shared-prefix inversion. A shared prefix can only help H,
because H reuses the blocks that L has already scheduled. Inheritance has nothing to fix. The
expected result is KILL.

**Why run anything.** The prediction comes from reading code, and I can miss a path. Candidate
paths where L could still hurt H through the shared prefix:
1. H waits for L's in-flight prefix. Not found in the code. The test checks it.
2. L takes the token budget first (issue 59907). This also happens with an unshared L. Control C2
   separates it.
3. KV pressure preempts L. Blocks that H already references stay. Other cached blocks of P become
   evictable, so H may recompute more of P. That is at worst the unshared case, not an inversion.
4. H hits blocks that L scheduled but did not write, and then L is un-scheduled in the same step
   (issue 59907). That is a correctness fault. The fake runner cannot see wrong values. Out of scope.
5. H recomputes the rest of P while L computes it later. That wastes L's work, not H's time. It is
   I-64 (killed: singleflight ships in SGLang).

## 3. Stage A design (scheduler level, runs first)

**Engine.** The real vLLM `Scheduler` class (synchronous, not `AsyncScheduler`) at the pinned commit.
Install vLLM from source with `VLLM_TARGET_DEVICE=empty` (Python only, no kernels) in Docker
(`python:3.12-slim`, linux/arm64) with CPU PyTorch. Build the scheduler with vLLM's own test helper
`tests/v1/core/utils.py::create_scheduler` (`scheduling_policy="priority"`,
`enable_prefix_caching=True`, chunked prefill on, block size 16). The model config comes from the
local Qwen3-0.6B snapshot, mounted read-only. No weights are loaded and nothing is downloaded.

**Fake model runner.** After each `schedule()`, the harness returns one sampled token for each
request whose prompt is fully computed after this step, and no token for a request still in prefill.
It calls `update_from_output`. Each request generates 4 tokens, then stops.

**Metric.** H's time to first token in scheduler steps, from the step when H arrives to the step that
returns H's first token. Secondary: total prefill tokens scheduled (duplicate work), the number of
preemptions, and steps until all requests finish.

**Requests in one trial.**
- L: priority 10 (low; in vLLM a lower value runs first). Prompt = prefix P + its own suffix.
  Arrives at step 0.
- M: `N_m` medium-priority requests (priority 5), unshared prompts of length 2 x |P|. Arrive at
  step 0, after L.
- H: priority 0 (high). Prompt = P + a short suffix. Arrives at step `a`, where `a` is a fraction `f`
  of L's prefill steps when L runs alone.

The harness sets each request's `arrival_time` to its arrival step. vLLM orders the priority queue
by `(priority, arrival_time, request_id)` (FACT, `vllm/v1/request.py` `Request.__lt__`), so wall-clock
time must not leak into the order.

**Conditions (paired: same lengths, seeds and arrival steps).**
- C0: no L. H and M only.
- C1: L shares P with H.
- C2: L has a different prefix P' with the same length (control for load and budget order).
- C4: as C1, but L has H's priority 0 (full promotion, an upper bound on what inheritance can give).

**Grid (192 cells).** |P| in {512, 2048, 8192}; `max_num_batched_tokens` in {256, 2048};
f in {0, 0.25, 0.5, 0.9}; `N_m` in {0, 4}; KV blocks in {ample: 10,000; tight: 1.1 x the blocks
that L, M and H need together}; `long_prefill_token_threshold` in {0, budget / 4}.
Per cell: 40 trials. Seeded random suffix lengths (L: 16 to 512 tokens; H: 16 to 128; M: 16 to 128)
and an arrival jitter of 0 to 2 steps. Seed 20261019.

**Positive control (PC).** A harness-only subclass of `Scheduler` injects the missing dependency. It
skips a waiting request in a step when its prompt shares at least one full block with a running
request that is still in prefill. This emulates "H waits for L". PC uses the same grid cells.

## 4. Stage B (full CPU backend, only if Stage A breaks K1)

If Stage A finds a shared-prefix inversion in any cell, build the vLLM CPU backend (linux/arm64) and
measure wall-clock TTFT with Qwen3-0.6B (cached locally) in the worst three cells. Stage B gets its
own predeclared rules before it runs. If Stage A keeps K1, Stage B does not run, because a
scheduler-level null already decides the premise.

## 5. Statistics

Per cell and condition: p50 and p99 of H's TTFT over 40 trials (p99 uses linear interpolation, so it
is close to the maximum). The ratio tests compare paired conditions in the same cell. No significance
test is used, because the scheduler is deterministic for a given seed. Run-to-run noise is zero by
construction, and the trial spread comes only from the seeded lengths and arrival steps.

## 6. Frozen rules (all thresholds are project heuristics)

Validity (checked first):
- V1 (positive control): in the PC cell |P| = 2048, budget = 256, `N_m` = 4, f = 0.25, ample blocks,
  threshold 0, PC must give p99 TTFT(C1) >= 1.25 x p99 TTFT(C2), and a C4 gain >= 15%. If not, the
  harness cannot see the effect, and E-35 is INVALID.
- V2 (termination): every run finishes all requests within 20 x the steps of an ideal serial
  schedule. A cell that fails V2 is reported and excluded. If more than 10% of cells fail V2, E-35 is
  INVALID.

Decision (unmodified vLLM scheduler):
- K1 (no shared-prefix inversion): in every valid cell, p99 TTFT(C1) <= 1.10 x p99 TTFT(C2) and
  p50 TTFT(C1) <= 1.10 x p50 TTFT(C2).
- K2 (no inheritance gain, the I-59 kill rule from R-10): in every valid cell, the p99 gain
  (TTFT(C1) - TTFT(C4)) / TTFT(C1) is under 15%.
- KILL I-59 if K1 and K2 both hold.
- MEASURE-FIRST (go to Stage B) if a cell breaks K1 and its C4 gain is at least 15%.
- PIVOT if a cell breaks K1 but no cell reaches a 15% C4 gain: an inversion exists, but inheritance
  does not fix it. Report the path and stop.

Grid-edge rule (lesson from E-28): before accepting "no effect", report the cells with the largest
C1/C2 ratio and check that the grid edges (|P| = 8192, budget = 256, f = 0.9, tight blocks) ran.

## 7. Threats to validity

- Steps are not time. A step with a large prefill chunk costs more than a decode step. Stage A can
  show the absence of a dependency; it cannot price a wall-clock effect.
- Only the synchronous scheduler. `AsyncScheduler` and pipeline parallelism are not covered.
- One block size (16) and no KV connector, no speculative decoding, no multimodal input.
- The fake runner cannot detect wrong outputs (issue 59907 class).
- SGLang is covered by code reading only.
- The author of the hypothesis designed the positive control. R-31 must attack it.

## 8. Budget and stop rules

Setup: at most 1 hour. If the Python-only vLLM install fails twice, stop and report. Run: at most
3 hours of wall time; if the grid is slower, keep all |P| and f values and cut trials to 20 per cell,
and record the change before running.

## 9. Files

- `README.md`: this file (design and frozen rules).
- `Dockerfile`, `run_docker.sh`: environment (to be written after R-31).
- `harness.py`, `decide.py`, `test_harness.py`: harness, frozen decision script, tests (after R-31).
- `results/`: raw per-trial data and the decision output.

## 10. Amendment A1 (2026-10-19, after R-31, before any measurement)

**Why.** R-31 returned DO_NOT_RUN_CODE_DECIDES. Its blockers:
- (FACT) vLLM never makes a waiting request wait for another request's prefix.
- (FACT) The running loop takes the token budget in admission order and ignores priority (main session
  confirmed: `scheduler.py` lines 712 to 735; nothing sorts `self.running`). Priority acts only on
  the waiting-queue order and on the choice of preemption victim.
- (INTERPRETATION) V1 could not pass in its named cell. C4 is not an inheritance bound: a promoted L
  runs before M and fills the cache before H arrives. The rules had no outcome for "K1 holds, K2
  fails". The main-session smoke test (one cell outside the grid, data discarded) showed the same C4
  effect: H's TTFT fell from 6 to 4 steps with no wait involved.

**What replaces the grid.** A trace check on the unmodified synchronous `Scheduler`, which is the class
that the vLLM CPU backend runs (R-31 finding 9). The harness wraps two scheduler methods to log, for H:
the local prefix-cache hit at admission (`_get_local_prefix_cache_hit`) and every `False` from
`_handle_blocked_waiting_request`. After each `schedule()` it records H's scheduled tokens, whether H is
in `deferred_waiting`, the preemptions in the step, and the total scheduled tokens.

For each step where H waits and gets 0 tokens, the reason is classified in this order: `skip` (H is in
`deferred_waiting`, or the wrapper returned False), `preempt_step` (a preemption happened in the step),
`budget` (all budget was scheduled), `seqs` (the running set is full), `other` (for example an
allocation failure).

Common setup: block size 16, prefix caching on, priority policy, `max_model_len` 32768, H/M/L
priorities 0/5/10, `ignore_eos=True`, 4 output tokens, token id 7 as the fake sample. |P| = 2048,
L suffix 64 tokens, H suffix 32 tokens, unless stated.

**Frozen scenarios and assertions.**
- T1 (admission hit). Budget 256, `long_prefill_token_threshold` 64, so L takes 64 tokens per step
  and leaves budget for H. Ample blocks, no M. H arrives at step a in {1, 10, 25, 40} (L's prefill
  ends after 33 steps). Assert: H is admitted in its arrival step, and its local hit equals
  `16 x floor(min(T_L, |P|) / 16)`, where T_L is L's computed tokens after that step's `schedule()`.
- T2 (budget wait is not a prefix wait). Budget 256, threshold 0, so L takes the whole budget. H
  arrives at step 3. Run C1 (shared P) and C2 (unshared P'). Assert: in both, H gets 0 tokens in
  every step from 3 until L's last prefill step, each with reason `budget`; H's first scheduled step
  is the same in C1 and C2; H's hit is 2048 tokens in C1 and 0 in C2.
- T3 (sweep with pressure). 300 seeded random scenarios (seed 20261019): |P| from 256 to 4096 in
  steps of 16; budget in {64, 256, 1024}; threshold in {0, budget/4}; 0 to 4 M requests (prompts of
  256 to 4096 tokens) arriving at steps 0 to 3; L arriving at step 0 to 2; H arriving in
  [0, L's solo prefill steps]; shared or unshared with equal chance; KV blocks either ample or
  0.6 x the blocks that all requests need together, but never fewer than the largest single request
  needs plus 2 blocks, because `scheduler_reserve_full_isl` (default on) admits a request only when its
  full sequence fits.
  Assert: every run finishes within 20 x the ideal steps, and H never has reason `skip`. Report the
  counts of the other reasons and of preemptions.
- T4 (positive control for the trace). Repeat T1 at a = 10 with the injected-wait scheduler. Assert:
  H has reason `skip` in at least one step while L is still in prefill.

**Frozen decision.**
- T4 fails: INVALID (the trace cannot see a wait).
- T1, T2 and T3 all pass: I-59 KILL, scoped to vLLM `3897f95c` (synchronous scheduler, no KV connector,
  no grammar, no streaming input, no LoRA) and to SGLang `476fe86a` by code reading.
- Any assertion in T1, T2 or T3 fails: I-59 stays WEAK. Record the failing path. No mechanism work
  starts without a new predeclaration and review.

**Scope note.** The user asked for a check on the vLLM CPU backend. The trace check runs the same
scheduler class without the model runner. Whether to add a one-cell CPU-backend fidelity run is the
user's decision (rule 0.2.8, D-48). It is not part of this decision.

Files: `trace_check.py` (the check), `harness.py` (shared request and scheduler helpers; its grid
`main()` is withdrawn and unused).

## 11. Run 1 and amendment A2 (2026-10-19)

**Run 1** (`results/trace-20261020T060013Z/`, kept as is): decision INVALID. T1, T2 and T3 passed. T4
failed: H was never skipped, because L had finished its prefill before H arrived at step 10.

**Cause (FACT).** `scheduler.py` lines 673 to 682 set `long_prefill_token_threshold` to 0 when only one
request is eligible ("lone-request exemption", cited by vLLM issue 60476 as PR 57951). L runs alone
until H arrives, so it prefills 256 tokens per step and ends its 2112-token prefill at step 8. The
trace confirms it: at step 1 (H present) L has 320 computed tokens (256 + 64).

**Amendment A2 (before run 2).** Only the arrival steps change; every assertion stays.
- T1: H arrives at a in {1, 3, 5, 7, 10, 25, 40}. Steps 1 to 7 are inside L's prefill; 10, 25 and 40
  are after it.
- T4: H arrives at a in {1, 5} with the injected-wait scheduler. Precondition: L is still in prefill
  at the end of step a (L computed tokens < L prompt tokens). If the precondition fails, the run is
  INVALID as a scenario defect. If it holds, T4 passes only when H has reason `skip` in at least one
  step while L is in prefill, for both arrival steps.

## 12. Result (run 2), R-32 post-run review, and supplement

**Run 2** (`results/trace-20261020T060157Z/`): decision KILL. All four assertions passed.
- T1: H was admitted in its arrival step at all 7 arrival steps. Its prefix hit equalled what L had
  scheduled: 320, 832, 1344 and 1856 tokens while L was in prefill, and 2048 after it.
- T2: with L taking the whole budget, H waited 5 steps with reason `budget`, the same with a shared
  and an unshared prefix. H started in step 8 in both. The hit was 2048 (shared) and 0 (unshared).
- T3: 300 scenarios (146 with tight KV blocks, 55 preemptions). H never had reason `skip`. Wait
  reasons: `budget` 1729, `other` 310, `preempt_step` 17.
- T4: with the injected wait, H had reason `skip` for 29 steps (a = 1) and 13 steps (a = 5).

**R-32** (Sonnet, post-run): CONFIRM_WITH_SCOPE_LIMITS. It reproduced the run-2 output hash and
accepted A2 as a scenario fix. Its main points:
- (FACT) The `skip` detector is not complete. When `allocate_slots` returns None, the waiting loop
  breaks without a skip signal. A mutation that made H's allocation fail during L's prefill showed up
  as `other`, not `skip`. T4 is partly circular, because the injected wait uses the hook the detector
  wraps.
- (FACT) All 310 `other` steps are H's own allocation failures (full-sequence reservation under tight
  blocks). A shared hit lowers H's block need, so it cannot cause them.
- (FACT) A paired sweep (600 scenarios, shared or unshared prefix, same draws) found no case where
  sharing made H start or get its first token later.
- (INTERPRETATION) Priority inversions exist in vLLM: budget order (issue 59907), capacity waits and
  preemption. None of them depends on a shared prefix, and inheritance cannot fix them because L is
  already running. The conclusion is "no shared-prefix dependency", not "no priority inversion".

**Supplement** (`supplement_check.py`, `results/supplement-20261020T143143Z/`). Added after R-32,
because the reviewer's own scripts were lost when the session scratchpad was cleared. It is not part of
the frozen decision.
- Part P, paired sweep with new seeds: 600 scenarios, all finished. H's first scheduled step: equal 510,
  earlier when shared 90, later when shared 0. First token: earlier when shared 361, equal 239, later 0.
- Part A, allocation labels on the T3 scenarios: `budget` 1729, `preempt_step` 17, and all 310 `other`
  steps are H's own failed allocations (0 unexplained).

**Corrections to this file.**
- Section 10 (A1) says "L's prefill ends after 33 steps". That is wrong: the lone-request exemption
  lets L prefill 256 tokens per step until H arrives (Section 11).
- Section 2 and A1 say that H "computes the rest of P itself". T1 does not test that; it rests on code
  reading (`allocate_slots` and the waiting loop), as R-32 finding 7 says.

**Conclusion (scoped).** In vLLM `3897f95c` (synchronous scheduler, CPU platform, no KV connector,
grammar, streaming input, LoRA or speculative decoding; one model config, block size 16), no wait of a
high-priority request depends on a lower-priority request computing a shared prefix. A shared prefix
never delayed H in 600 paired scenarios. SGLang `476fe86a` reaches the same answer by code reading
only. Priority inheritance through a shared prefix therefore has no dependency to fix. Not covered:
wall-clock effects, async scheduling, KV connectors, and correctness faults of the issue-59907 class.
