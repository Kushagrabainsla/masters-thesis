# E-31 pre-run review (hostile systems reviewer)

I read the seven files and the E-31 and I-122 entries in `RESEARCH_JOURNAL.md`. I ran no code and edited no file. All traces below are hand derivations from the code. I did not check the four vLLM issues themselves, so I take their descriptions from the README and the journal. I did not read `run_sonnet_review.sh` or `sonnet_review_prompt.md`.

## Verdict

**AMEND_BEFORE_RUN**

The current design cannot produce a kill. Each decision rule is fixed by the model definitions, except one rule that is fixed by arithmetic. The model is repairable, so the design is not beyond repair. The amendments are in the last section.

## BLOCKER findings

**B1. The versioned policy cannot violate an invariant. Rule 2 is vacuous.**
- Fact: The versioned guard and the violation check use the same predicate.
  - Work: guard `model.py:199-203`, check `model.py:229`.
  - Events: guard `model.py:259-264`, checks `model.py:288-298`.
- Fact: Invariant 4 is satisfied by one policy-name branch (`model.py:336`).
- Fact: The `output` field is set at `model.py:190` and `model.py:242` and never compared. The check uses metadata fields and never the output value. Issue 49449 is described as "wrong output", but no check looks at output.
- Interpretation: A guard that is checked against itself cannot fail. `KILL_MECHANISM` can fire only through a coding error.

**B2. The epoch_only versus versioned difference is one `if`, and the failure is injected by policy name.**
- Fact: `EPOCH_ONLY` and `VERSIONED` share the same code at `model.py:158`, `198`, `259` and `312`. They differ only at `model.py:336`, where `fold_after_preempt` omits `VERSIONED` from the lossy list.
- Fact: The lossy path sets `tokens = materialized_tokens or ()` (`model.py:338`). After `preempt`, `materialized_tokens` is `None`, so the history becomes `()` on the first call.
- Fact: The model has no separate "commit frontier" variable. The README (`README.md:17-22`) and the journal describe one.
- Interpretation: The 58776 scenario is a strawman. It is not a mechanism. Rule 4 (`PIVOT_EPOCH_ONLY`) is decided by line 336. Rule 6 (`PIVOT_WEAK_PREMISE`) cannot fire either: by hand trace, `mutable` shows four violation kinds in the source scenarios.

**B3. No safety invariant covers dropped valid work, dropped valid events, or stalls.**
- Fact: `_complete` and `_flush` discard an item whose revision matches but whose frontier differs (`model.py:225-227`, `285-287`). `discarded_work` and `discarded_events` do not separate stale items from valid ones.
- Fact: `test_model.py:60-76` asserts that dropping a valid out-of-order event is the correct result.
- Fact: After a discarded out-of-order completion, `schedule_frontier` stays ahead of `frontier`. Hand trace for `versioned`: `schedule, schedule, complete_newest, complete_oldest, schedule, complete_oldest`.
  - The first complete_newest drops the frontier-1 work.
  - The complete_oldest applies frontier 0 and sets `frontier` to 1.
  - The next work has frontier 2, so every later work is discarded until the next update.
- Fact: `abort_rebuild` clears `pending` and `queue` (`model.py:144-145`) with no accounting.
- Interpretation: A policy that drops every item passes all five invariants. The `PASS` gate has no liveness or completeness check.

**B4. Rule 5 is the only rule that can fire, and its cost metric is set by the model constants.**
- Fact: Each work item costs `len(tokens)+1` (`model.py:223`), which is a full-prefix charge per step. Each abort charges `len(tokens)` (`model.py:149`) for any update, with no prefix reuse.
- Fact: The ratio is `rebuild_tokens / computed_tokens` of the abort policy only (`run_e31.py:251`). It ignores the stale work that `versioned` computes and then discards (`model.py:223` runs before the guard).
- Fact: In the 5% update, depth-1 cell, the hand-derived ratio is about update_rate divided by depth, so about 5%. That is above 3%. Rule 5 needs every 1% and 5% cell to pass (`decide_e31.py:94-96`).
- Fact: The README says a full abort "is correct" (`README.md:76`). With `queue.clear()` as instant, perfect cancellation, `abort_rebuild` is correct by construction.
- Interpretation: The cost unit decides the verdict. A one-token decode step would give a different answer. The journal question "less than 2% CPU overhead at 1,000 sessions" (`RESEARCH_JOURNAL.md:2617-2618`) has no rule. `protocol_checks` is recorded and never used.

## MAJOR findings

**M1. The source scenarios do not match the cited issues.**
- Fact: No scenario or state variable represents batch-row identity (issue 42655). `README.md:30` lists it as a source.
- Fact: `parser_flush_after_update` (`run_e31.py:66-70`) maps to none of the four cited issues.
- Fact: Issue 36755 is "async output reads state after later mutation". `Work.tokens` is an immutable copy taken at schedule time (`model.py:189`). The model tests "old snapshot applied to new state", which is a different race.
- Fact: Every scenario has at most one update. `replace_last` changes only the tail, and the new token is always different (`model.py:173-180`).
- Fact: The model has no cache sharing between sessions, so it cannot show cross-session contamination from a stale hash.

**M2. The bounded search and the sweep cover little of the state space.**
- Fact: `enabled_actions()` uses the default `max_outstanding=2` (`run_e31.py:125`). The sweep uses depth 4.
- Fact: The search has one session, three initial tokens, and depth 6.
- Fact: `unique_states` sums per-depth counts (`run_e31.py:149`), so one state can be counted at several depths. It counts state-depth pairs.
- Fact: `build_trace` uses only `complete_oldest` and `flush_oldest` (`run_e31.py:182`, `186`). The sweep never produces out-of-order completion. Its violation check for `versioned` is nearly vacuous.
- Fact: Sessions in `run_trace` are independent objects (`run_e31.py:201-212`). The session count only scales the loop.
- Fact: The seed omits `repeat` (`run_e31.py:267-273`). The five repeats run the same trace.

**M3. The INVALID rule and the `PASS` rule are narrower than declared.**
- Fact: Rule 1 reads only `source_types` of `abort_rebuild` and `synchronous` (`decide_e31.py:80-83`). Bounded-search and sweep failures of these baselines do not invalidate the run.
- Fact: The README does not define "conservative reference baseline". The code picks two policies.
- Fact: `decide_e31.py:117-119` returns `KILL_SIMPLE_BASELINE` when hash failures are all cache failures. No declared rule says this (`README.md:57-63`).
- Fact: `NON_CACHE_FAILURES` omits `out_of_order_event_emitted` (`decide_e31.py:14-18`). That kind is also missing from the five declared invariants (`README.md:45-51`).
- Fact: The journal kill rule says hash invalidation "handles every interleaving" (`RESEARCH_JOURNAL.md:261`). Rule 3 tests the four source scenarios only.

**M4. The baselines are weak by definition.**
- Fact: `mutable` and `hash_only` get a bare frontier check or no check (`model.py:214`, `276`). All policies reset the frontier to 0 at each update (`model.py:127-129`). This makes old frontier-0 work look current.
- Fact: No policy models a narrow per-issue fix, such as a work snapshot or a monotonic counter without other state.
- Fact: `synchronous` is correct in the model (`test_model.py:89-95`). It has no rule and no stall measure. `drained_work` is unused.
- Interpretation: `PASS` does not show that `versioned` beats a drain. `hash_only` is not "current main", so the journal kill rule "Current main passes the matrix" (`RESEARCH_JOURNAL.md:261`) is untested.

**M5. The freeze cannot be proven.**
- Fact: `git status` at session start shows `?? experiments/versioned_session_commit/`. The files are untracked.
- Fact: The manifest hashes files at run time (`run_e31.py:397-401`). It records no git SHA and no dirty flag.
- Fact: The base image is a mutable tag (`Dockerfile:1`). The manifest records the built image ID, not the base digest.
- Fact: `E31_CPU_SET` is not passed to the container or recorded (`run_docker.sh:31-33`).
- Fact: The unit-test output goes to the terminal only (`run_docker.sh:38`). `mkdir -p` (`run_docker.sh:13`) can merge two runs that start in the same second.

## MINOR findings

- `apply()` can call any non-underscore method, such as `stats` or `clone` (`model.py:367-371`).
- The bounded search has no state cap, progress log, or timeout. The signature includes work and event IDs, so deduplication is weak (`model.py:373-389`).
- Policy order rotates by `repeat % 6` with five repeats (`run_e31.py:282-283`). Timing is report-only, so this affects no verdict.
- `README.md:77-78` states the expected results, and the code matches them (see B1 and B2). Write predictions before the model, not as part of the rubber duck.
- `decision.json` does not link to the manifest or its hashes.

## Short answers to the twelve questions

1. Yes (B1, B2).
2. No (M4, B4).
3. No. There is no liveness check, no output oracle, and the ordering invariant is undeclared (B1, B3, M3).
4. Yes (B3).
5. No (M1).
6. No (M2).
7. No (B4, M2).
8. The order matches `README.md:57-63`. Rule 1 is narrower than declared, and one extra branch is undeclared (M3).
9. Yes (M3).
10. Not yet (M5).
11. Yes. Rules 1, 2, 3, 4, 6, and 7 are fixed by definitions. Rule 5 is fixed by arithmetic (B1 to B4).
12. Yes. See the next section.

## A simpler kill test (question 12)

Run a source audit before E-31, with no model and no GPU:
1. Pin a vLLM commit SHA.
2. For each of the four issues, record whether it is open or fixed, the fix PR, and whether the fix is narrow.
3. Search the streaming-update path for an existing generation, epoch, or incarnation field. The README already names this risk (`README.md:83`), but E-31 never reads vLLM.

If all four have narrow fixes and no generation concept is needed, the "recurring class" premise is weak. After that, test the real scheduler on CPU with a fake model-runner output. I believe vLLM's scheduler unit tests already do this, but I did not verify it. That test would replace the abstract model.

## Smallest exact amendment set

1. **Freeze.** Commit the directory. Record the SHA and file hashes in the journal. Make `run_docker.sh` stop if the tree is dirty. Pin the base image by digest. Record `cpu_set`. Write the unit-test log into the result directory. Use `mkdir` without `-p`.
2. **Independent oracle.** At `flush`, compare `event.output` with `output_token(current tokens, revision, event.frontier)`. Report a new violation `wrong_output_emitted`. Add it to the invariants list and to `NON_CACHE_FAILURES`. Add `out_of_order_event_emitted` to both.
3. **Completeness.**
   - Count `discarded_valid_work` and `discarded_valid_events` separately, for items with matching revision and incarnation.
   - Add a liveness probe at every bounded state: on a clone, run `schedule`, `complete_oldest`, `flush_oldest`. A quiescent state that emits nothing is a violation.
   - Fix `EPOCH_ONLY` and `VERSIONED` so a frontier-only mismatch re-queues the item. Apply the same rule to the baselines.
4. **Remove the name branch at `model.py:336`.** Use one fold code path. Give `versioned` a separate `logical_tokens` and `materialized_tokens`. Give the other policies one shared progress variable. If this is too large, drop scenario 4 and invariant 4 from rules 4 and 7 and report them as information only.
5. **Rule 5.** Compute `extra_work = (abort_total - versioned_total) / versioned_total` on the same trace. Count the stale work that `versioned` discards. Report two rebuild charges: full prefix, and changed suffix only. Apply the 3% threshold to the changed-suffix charge, since it is the strongest baseline. State in the README that this is a model constant and not vLLM evidence.
6. **Rule 1 and rule 7.**
   - Define the reference baselines as `abort_rebuild` and `synchronous` in the README.
   - Evaluate them on the source, bounded, and sweep results.
   - Declare the `KILL_SIMPLE_BASELINE` branch for the "hash leaves only cache failures" case.
   - Align `RESEARCH_JOURNAL.md:261` with rule 3.
7. **Synchronous.** Report `drained_work` per update. Add a README sentence that `PASS` does not rank `versioned` above a drain. Add the 2% overhead question as an open item for the integration test, since this screen cannot measure it.
8. **Scenarios.** Map each scenario to its issue in the README. Remove issue 42655 from the source list, or add a row-identity scenario. Add a two-consecutive-updates scenario with a replace at a non-tail position. Add a shuffled-completion trace to the sweep.
9. **Search reporting.** Rename `unique_states` to `state_depth_pairs`. Write `max_outstanding` into `model_check.json`.
10. **Planted-bug controls.** Before the run, check that the invariant checkers flag three deliberately broken `versioned` variants: guard removed, valid items dropped, and fold branch restored. Each check must flag the planted bug.

I recommend the source audit first. Run E-31 only after these amendments, and treat its result as a spec check on the model.
