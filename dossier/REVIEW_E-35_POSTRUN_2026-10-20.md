# R-32: Sonnet post-run review of E-35 (2026-10-19 to 2026-10-20)

Reviewer: an independent Sonnet agent (D-29). It stopped once on an API usage limit and resumed. It
read the E-35 README (design, amendments A1 and A2), R-31, the harness and trace code, both result
folders, and the pinned vLLM source. It re-ran the trace check in the pinned image and wrote its own
scripts and outputs only in the session scratchpad.

Artifact note: the session scratchpad was cleared before its files were copied into the repository.
The reviewer's scripts (`inspect_other.py`, `paired.py`, `mutation.py`) and outputs are therefore lost.
Before the loss, the main session read `paired.py` in full, confirmed that both runs of a scenario draw
the same random sequence, read `paired.out` (the counts in finding 4), and confirmed that the reviewer's
re-run output had the run-2 hash `57e23ce1...`. The main session then re-implemented findings 3 and 4 as
`experiments/priority_inheritance_premise/supplement_check.py` (see the E-35 README, Section 12).

The review text follows as the reviewer returned it.

---

**Verdict: CONFIRM_WITH_SCOPE_LIMITS.** The KILL of I-59's premise holds: no vLLM wait depends on a shared prefix. I found no BLOCKER. Two MAJOR findings concern how far the evidence goes.

**Reproduction (FACT).** I re-ran `trace_check.py` in the pinned image. The output sha256 equals the manifest `result_sha256` (57e23ce1...). Run 2's manifest hashes match the current files.

**Findings**

1. **MINOR / FACT: A2 is a legitimate scenario fix, not outcome fitting.**
   - `scheduler.py` lines 673 to 682 set the threshold to 0 when `num_running + num_waiting <= 1`. L therefore runs alone at 256 tokens per step. In run 1 only a=1 fell inside L's prefill (L had 320 computed tokens at a=1, 2114 at a=10).
   - Between the two runs, T2 and T3 are identical (same counts: 1729 budget, 17 preempt_step, 310 other, 0 skip). The old T1 points (a=1, 10, 25, 40) give identical results.
   - A2 only added T1 points a=3, 5, 7, and a stricter T4 precondition. These can only add assertions. They cannot turn a T1-T3 failure into a pass.
   - Disclosure is good, and run 1 is kept. README Section 10 still says "L's prefill ends after 33 steps", and its status line does not mention A2.

2. **MAJOR / FACT: the "skip" detector is not complete. The T3 assertion cannot see allocation-style waits.**
   - The waiting loop leaves H without a signal in some paths. On `allocate_slots` returning None, it runs `break` (about line 1349). H is not added to `deferred_waiting`, and `_handle_blocked_waiting_request` is not involved.
   - Mutation test (`r32/mutation.py`): I made H's allocation fail while a same-prefix request was mid-prefill. This is a real prefix wait. The trace gave reasons `other` x29 (a=1) and `other` x13 (a=5), and zero `skip`. The T3 rule "no skip" would pass.
   - So T3's assertion alone does not support "no prefix wait". Two other pieces of evidence do: finding 3 and finding 4.

3. **FACT: all 310 "other" steps are H's own KV-capacity allocation failures.**
   - Script: `r32/inspect_other.py`. I wrapped `allocate_slots`. In every "other" step, H's call in the waiting loop returned None. There were 0 "other" steps without an allocation failure.
   - The cause is the full-sequence-fits gate (`scheduler_reserve_full_isl`) plus the watermark, while running M and L hold blocks. 47 of the 310 steps have a shared prefix with L mid-prefill (for example scenario 148, steps 8 to 12).
   - A shared hit lowers H's block need. It cannot create this wait.
   - The 17 `preempt_step` cases come from the `if not preempted_reqs` gate (about line 955). The victim is the max (priority, arrival_time) request, which has no link to the prefix.
   - Other `break` paths are off or unexercised: `max_num_active_reqs` (the `seqs` count is 0 because MAX_SEQS=16 and there are at most 6 requests), `defer_prefills`, and disabled chunked prefill.
   - The `deferred_waiting` set can hold stale entries, so the detector can only over-report skips. That is the conservative direction.

4. **FACT: a paired test supports the KILL, and the author did not run one.**
   - Script: `r32/paired.py`. It ran 600 scenarios. Each scenario ran twice, with only L's prefix shared or unshared. Blocks were ample or 0.3 to 0.8 of the total need.
   - First scheduled step: equal in 506, shared earlier in 94, shared later in 0.
   - First token: shared earlier in 371, equal in 229, later in 0. All 1200 runs finished.
   - Two cases had one extra `preempt_step` wait when shared, but the same first scheduled step and an earlier first token.
   - Recommendation: add this paired comparison to the check. It does not depend on the reason classifier.

5. **MAJOR / INTERPRETATION: T4 is partly circular.**
   - The injected wait overrides `_handle_blocked_waiting_request`. The detector wraps the same hook.
   - T4 proves the detector sees waits that go through that hook. After the A2 precondition, passing T4 is almost certain. It does not prove completeness (finding 2).
   - The A2 precondition itself is sound.

6. **MAJOR / INTERPRETATION: the conclusion must say "no shared-prefix dependency", not "no priority inversion".**
   - T2 shows H (priority 0) waits 5 steps behind running L (priority 10). The same wait occurs in C1 and C2, so it does not depend on sharing.
   - T3 shows 1729 budget-wait steps, 310 capacity-wait steps, and 55 preemptions. The running loop is priority-blind, and a waiting request never preempts.
   - These inversions are real. Inheritance cannot fix them, because L already runs. This matches issue 59907 and PR 59963.

7. **MINOR / FACT: T1 is weaker than the README says.**
   - T1 asserts that the hit equals `16*floor(min(L computed, 2048)/16)`. This holds at all 7 arrival steps, including 320, 832, 1344 and 1856 inside L's prefill. It shows H reuses blocks L scheduled in the same step, before they are computed.
   - T1 does not assert that H "computes the rest itself". It records no H token counts and no `num_computed_tokens` for H after admission. That claim rests on code reading.
   - T1 uses one configuration: budget 256, threshold 64, ample blocks, no M.

8. **MINOR / FACT: harness checks.**
   - `arrival_time=float(step)` is consistent with `Request.__lt__`.
   - The fake runner samples a token only when `num_computed_tokens >= num_tokens`, which matches `update_from_output`.
   - The termination cap is generous, and all 300 T3 runs finished.
   - The `prefill_tokens` accounting lives in the withdrawn grid code and is unused.
   - `seqs` is classified from the post-schedule `len(running)`. This is approximate but irrelevant because the count is 0.
   - The "L in prefill" flag is ambiguous after preemption, because L's `num_computed_tokens` resets to 0. This does not affect T4.
   - H's arrival window in T3 uses chunk=threshold, but a lone L runs at the full budget. Some arrivals therefore fall after L finishes. Step-level coverage is still large: 976 budget, 43 capacity and 7 preempt waits with a shared prefix and L in prefill.

**Scope limits to freeze**

- It covers the synchronous `Scheduler` at 3897f95c with a fake runner. No kernels or real attention ran.
- It covers no KV connector, grammar, streaming, LoRA, speculative decoding or async scheduling. I did not independently verify the AsyncScheduler claim.
- It uses one model config, block size 16, MAX_SEQS 16, priorities 0/5/10 and random-token prompts. Partial-block and CoW sharing, which is tied to connector paths, is not exercised.
- The 59907-class correctness fault (a hit on blocks that are scheduled but not yet written) is not covered. The CPU-backend fidelity run is still the user's decision (D-48).
- The SGLang claim comes from code reading only.
- The phrase "no inversion" is not supported. Use "no shared-prefix dependency".
