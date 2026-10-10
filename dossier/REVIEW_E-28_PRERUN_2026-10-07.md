# R-15 (draft): Adversarial pre-run review of E-28

Date: 2026-10-07. Reviewer: Sonnet 5.5 (different-model review). Status: review only. No repo file was edited. Stage 0, 1 and 2 did not run.

Labels: FACT = I read the code or data, or I ran a test. INTERPRETATION = my reasoning. HYPOTHESIS = untested.
Test scripts are in `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/56d042b4-71c1-432b-9e52-b83a6c684a44/scratchpad/e28_review/` (t1.py, t3.py, t4.py, t5.py, t6.py, t11.py, gen.py, dec/).
Note on vLLM: I did not open the vLLM source in this pass. Claims about vLLM v1 come from memory and are INTERPRETATION. Check them in `vllm/v1/core/sched/scheduler.py` and `kv_cache_manager.py` before you cite them.

## 1. Verdict

Do not run E-28 yet. There are 4 blockers. Two of them (B2, B3) push the simulator toward metastability. One (B4) pushes the K3 rule toward GO-B. One (B1) crashes the runner. There are also 6 major items. Some major items push toward "no collapse" (M2). The net direction of the bias is unknown, so a fix is needed before any result is read.

What passes (FACT):
- `python3 -I test_sim.py` passes all 6 tests.
- Hand-computed step time matches the code (t1.py: 0.42215 s for a 12k prompt with 10 output tokens).
- The KV invariant (`used` = running KV + idle KV, within capacity) holds in 40 random storm runs with tiny caches, cap policy, retries and spikes (t11.py). No session has two outstanding requests. `n_admitted` always equals the number of admitted live sessions.
- A run is deterministic. Three processes with different `PYTHONHASHSEED` gave identical bins (t6.py).
- Thinning of the arrival process is correct in distribution. Pre-spike arrivals are identical in a spike run and a control run with the same seed.

## 2. BLOCKERS

### B1. Calibration and control runs crash
- File: `run_e28.py:66-69`, calls at lines 131 and 146.
- FACT: `base_params` does `replace(Params(), ..., **SPIKE, **kw)`. Stage 0 and the `ctrl` runs pass `spike_len_s` in `kw`. `SPIKE` already holds `spike_len_s`. Python raises `TypeError: got multiple values for keyword argument 'spike_len_s'`. I reproduced it for both call shapes. Only the spike run shape works.
- Effect: the runner crashes. It cannot give a wrong result, but a quick patch could be rushed.
- Fix: `replace(Params(), kv_capacity_tok=..., **{**SPIKE, **kw})`. Add a smoke test that builds every Params shape the runner uses.

### B2. A failed admission flushes the whole idle cache
- File: `sim.py:317-327` (with `evict` at 178-192).
- FACT: for the head of the waiting queue, the code calls `evict(n)`. `evict` frees idle caches one by one. It only returns False after it has freed ALL idle caches. The code then puts back only the head's own cache and stops. The freed caches of other sessions stay lost.
- FACT (t3.py): free 0, two idle caches of 2,000 tokens, a waiting request that needs 8,192. After one step both idle caches were 0 and the request was still waiting.
- INTERPRETATION: vLLM v1 checks the number of free blocks first. It returns "cannot allocate" without evicting. So in vLLM the idle caches survive. In the simulator every step with a blocked queue head wipes all idle caches. Every later turn of an idle session is then a miss. This is exactly the loop E-28 tests (misses make turns slower, slower turns keep sessions live). The simulator builds the loop by itself in any overload.
- Fix: test feasibility first: `free + sum(cached of other sessions) >= need`. Evict only if the test passes. Add a unit test: blocked head, idle caches unchanged.

### B3. Preemption restarts decoding from token 0
- File: `sim.py:358-365` (`r.computed = r.decoded = 0`).
- FACT (t1.py, T2): two requests with 3,000 output tokens. The preempted one had 2,000 tokens decoded. It decoded all 3,000 again, one token per step. It finished at 34 s instead of about 21 s. The extra time (about 13 s) is 2,000 decode steps.
- INTERPRETATION: vLLM v1 keeps the output tokens of a preempted request. After resume it prefills prompt plus outputs in chunks (prefix hits help) and then continues. The cost is about 0.2 s for 5,000 tokens at 30k tok/s, not 13 s. The simulator makes each preemption about 50 times more costly for long outputs. Mean output is 239 tokens and 3.5% of turns exceed 1,000 tokens (FACT). Preemption storms and timeouts are then much too heavy.
- Fix: keep `decoded` in the request. On preemption fold it into the prompt: `prompt_priv += decoded`, `out -= decoded`, then re-prefill. Free the KV. Add a test that a preempted request does not decode more than `out` tokens in total.

### B4. The K3 rules are decided by gateway shedding, not by the collapse
- Files: `sim.py:215-219, 244-249`; `run_e28.py:150-157`; `e28_decide.py:89-101`.
- FACT: in stage 2 a queued session is shed after the client timeout (60 s). A shed session gives 0 turns. `restore` = oracle-cap late goodput / uncapped control late goodput. GO-B fires if `restore < 0.8` at 50% or more of the collapse points.
- INTERPRETATION: G* is defined as the best capped throughput. So an offered load of 0.8 or 0.95 times G* is a utilization of 0.8 or 0.95 of the capped system. With Poisson arrivals and sessions that last about 420 s (30 turns, about 14 s per turn: FACT from the trace gaps and mean output), an M/M/N approximation (N = 8 to 16) says that 19 to 36% of arrivals wait longer than 60 s at load 0.8, and 70 to 80% at load 0.95. These sessions are shed. The capped run cannot reach 0.8 of the control at load 0.95, and barely at 0.8, even if the cap removes the collapse completely. GO-B ("a cap does not fix it") would be true by construction. Collapse points are most likely at the high loads, so the rule is hit hardest there.
- Also: turn goodput hides who was served. A cap that sheds 25% of sessions and serves the rest well looks like 0.75 of control. Session success is not in the decision.
- Fix: (1) Add session-level metrics to `job`: sessions started, `sess_ok`, shed rate, goodput per admitted session. (2) For K3, give the gateway queue a patience of at least 10 session lifetimes, or no shedding. Report queue delay separately. (3) Define "restores" as goodput at least 0.8 times min(control, offered load after shedding), or compare at loads where an ideal cap has slack (at most 0.8). (4) Count a point as "fixed" only if the session success rate is also at least 0.9 times control. Amend the journal before any run.

## 3. MAJOR items

### M1. Scheduler order differs from vLLM v1 in two places
- File: `sim.py:316-336`, `291`, `303`.
- FACT: step 2 admits waiting requests even if step 1 preempted a request. The victim goes to the front of `waiting` and can be admitted again in the same step.
- INTERPRETATION: vLLM v1 skips the waiting queue in a step with a preemption. This is another source of churn that vLLM does not have.
- FACT: the victim is `max(running, key=t_submit)` and running requests are scheduled by `t_submit`. A re-admitted request keeps its age. In vLLM the running list is in admission order and a re-admitted request goes to the end, so it is the first victim again. The simulator spreads preemption more evenly. The two differences push in opposite directions, so the net bias is unknown.
- FACT: ties in `t_submit` pick the first request. This is only an artefact of simultaneous arrivals.
- Fix: keep an explicit admission counter. Preempt the highest counter. Skip admission if a preemption happened in the step. Add a test for each rule.

### M2. The trigger is weak and the load grid may miss the bistable window
- File: `run_e28.py:28-31`, journal E-28 setup.
- FACT: the mean number of private KV tokens for each turn is 26k (M1). So a cache of 436k tokens holds about 17 live sessions, and 216k holds about 8. The mean session life is about 420 s (INTERPRETATION: 30 turns x 14 s).
- INTERPRETATION: at 0.95 G*, a 3x spike for 60 s adds about 0.3 x N* sessions (3 to 4 sessions). The natural sd of the live-session count is sqrt(N*) (about 3 to 4). So the spike is about one sd. A Poisson count with mean f N* exceeds N* with probability 0.01 (f = 0.5), 0.05 (0.65), 0.17 (0.8) and 0.36 (0.95) for N* = 12. In the control the same fluctuations occur all the time. Two outcomes are possible. (a) If one sd is enough to flip, the control itself flips, `stable` is False, and the point drops out. (b) If it is not enough, the spike does nothing. Only a narrow band qualifies. K1 (KILL, no collapse point) can then fire for a reason that is not the physics.
- Also, G* comes from a deterministic closed system (cap = N, always full). The uncapped open system has no such smoothing. The knee in the open system is lower than 0.95 G*.
- FACT: the M4 mix at 440k holds about 41 sessions. The cap sweep stops at 32. The best cap is on the boundary for M4/440k. G* is then not the memory limit. Also (INTERPRETATION) for M4 the GPU is likely the limit, not memory: 32 sessions / 14 s x about 0.6 GPU-s per turn is above 1. The assumption "memory is the limit" fails for some mixes.
- Fix: (1) Size the spike in absolute sessions: add a sweep of spike excess of +25%, +50%, +100% of the cap-fit N*, for example 5x and 10x. (2) Use a finer load grid between 0.5 and 1.0 (steps of 0.05 around the control knee). Find the knee from uncapped control runs (the largest rate with a stable control), not from G*. (3) Extend CAPS to 48 or 64. (4) Report the control flip rate per seed.

### M3. The seeds are not paired after the spike
- File: `sim.py:170-175, 266-269, 372-384`.
- FACT (t4.py): one RNG feeds arrivals, session choice, gaps and jitter. The spike run and the control run share the stream only until the first difference. After the spike the arrival times are unrelated (I printed both lists: 413, 415, 429 vs 443, 462, 530).
- Effect: "paired seeds" (7 of 10) compares two different arrival sequences and two different session mixes. Noise is not reduced. The pairing is nominal.
- Fix: use separate `Random` streams. For example arrivals: `Random(seed*3+0)`; session content: `Random(hash((seed, arrival_index)))`; gaps: `Random((seed, sid))`; jitter: `Random((seed, sid, attempt))`. Use the same seed for spike and control.

### M4. The shared prefix is a free knob and it creates the mix effects that K3 measures
- File: `run_e28.py:42-48`.
- FACT: in the trace, claude_code first calls have p10 = 14,650 tokens (p50 18,034). The other harnesses have p10 = 803 and p50 = 2,647. M1 uses 4,000 for all sessions. M2 (the claude_code subset of M1) uses 14,000. M3 uses 2,000. M4 and M5 use 4,000.
- INTERPRETATION: M1 charges 10k extra private tokens to about 75% of its sessions. M1, M2, M3 and M5 are subsets or variants of the same pool. The "loss of the M1 cap on another mix" is then largely the difference in the chosen prefix, not a difference in the workload. The GO rule needs 3 of 5 mixes. Only M2, M3 and M4 can give it, because M1 is in-sample and M5 is M1 with slow tools (the needed cap is the same: memory fit).
- Fix: model a prefix per harness (two prefix groups, each charged once, resident). Use the same rule in every mix. Or keep one prefix value for all mixes in K3 and give the mix effect a separate row.

### M5. Oracle selection bias against a 10% threshold
- File: `e28_decide.py:87-92, 95`.
- INTERPRETATION: a run has a coefficient of variation of about 0.2 (HYPOTHESIS: roughly 12 live sessions over a 600 s window, which is 1.4 lifetimes). The mean of 10 seeds has about 6%. The oracle takes the maximum of 9 noisy means (about 3 to 4 are near the optimum). Expected upward bias is about 6%. The fixed cap uses the same data (in-sample for M1, and tuned over the four loads). The KILL threshold is 10% loss for 80% of points. A 6% noise bias moves points over the threshold. The rule leans toward "not KILL".
- Fix: select the oracle cap and the fixed cap on seeds 1 to 5. Evaluate both on seeds 6 to 10. Add a bootstrap interval for the loss. Add a "no cap" column to the sweep.

### M6. Output lengths of hosted reasoning models decide the timeout behaviour
- File: `sim.py:92-95` (output taken as is), `run_e28.py:29`.
- FACT: mean output is 239 tokens. 3.5% of turns exceed 1,000 and 1.2% exceed 1,500. 33% of the sessions have at least one turn above 1,000 and 9% above 1,800. The maximum is 50,571.
- INTERPRETATION: with a batch of 10 to 20 sessions a decode step takes about 20 to 40 ms. A turn of 1,500 to 2,000 tokens then reaches the 60 s timeout. The abort frees the decoded tokens. The retry decodes them again. So client A creates its own waste. This is a real effect, but it comes from the output lengths of other models (hidden reasoning tokens, hosted models). An 8B model does not produce them. K2 (are retries needed?) is mixed with this effect. Client C (600 s) is the clean test. B (60 s, 0 retries) is not "retries off": it also ends the session after the first timeout.
- Fix: add a sensitivity run with `out` clipped to 1,024 (or scaled). Report timeouts for each output bucket. Base K2 on the C comparison only, and say so.

## 4. MINOR items

- m1. Arrival time shift. FACT (t5.py): the next arrival is drawn from `self.t` (the end of the step), not from the event time (`sim.py:377-384`). With 0.3 s steps and 1 session/s, 9% of arrivals are lost (1797 to 1844 vs 2000). At E-28 rates (about 0.03 sessions/s) and steps of about 50 ms, the loss is about 0.1%. Fix: use the event time. The fix is one line.
- m2. A late completion counts as success when the deadline falls inside the last step. FACT (t6.py): deadline 1.5 s, finish 2.0 s, `turn_ok` = 1, `timeouts` = 0. The error is at most one step (0.3 to 1 s) against a 60 s timeout. Fix: in `on_done`, compare `self.t` with `r.t_submit + timeout`.
- m3. Warm-up. INTERPRETATION: with 14 s per turn and the trace session lengths, the live population is 86% of steady at 600 s and 93% at 900 s (computed from the length list). The spike lands before steady state and `pre` is low. The `stable` test (late >= 0.9 pre) is then easy to pass. Fix: spike at 2,400 s, pre window 1,800 to 2,400, horizon 6,000 s. Or prime the live set.
- m4. KV capacity. A single H100 (80 GB) at 0.9 utilization, minus 16 GB of weights and 4 to 6 GB of activations and graphs, leaves about 50 GB. That is about 380k tokens at 131,072 bytes each (the bytes per token are correct). 440k needs about 0.97 utilization. Use 380k and 190k, or label 440k as an upper bound.
- m5. Cost constants. 30k tok/s prefill, 1.3e-9 s per (token x context) attention, 6 ms step floor and 4.4e-8 s per context token of decode are plausible for 8B BF16 on H100 (weights read 16 GB / 3.35 TB/s = 4.8 ms; 131 KB per token / 3.35 TB/s = 3.9e-8 s). Summing prefill and decode in one step is a little pessimistic. A 25k miss costs about 1.2 s.
- m6. The "10x miss cost" claim in the README is too high. INTERPRETATION by hand calculation with the code constants: a hit turn costs about 0.5 to 0.6 GPU-seconds at the mean output (239 tokens; decode is the main part) and about 0.3 at the median (100). A miss on a 21k private prompt adds about 1.1 s. The ratio is about 3x to 4x, not 10x. The loop gain, and so the existence of the collapse, depends on this ratio. A sensitivity run is needed: prefill speed x0.5 and x2, `decode_s_per_ctx_tok` x0.5 and x2, `gap_scale` 0.25 and 4.
- m7. Gaps. FACT: claude_code spans have a median duration of 1 ms, so excluding their gaps is correct (the gap would include the hosted model latency). But the p90 duration is 12 s, so about 10% of those spans have real times. The pooled gap list is 75% smolagents (3,309 of 4,441). Truncated at 120 s: mean 6.9 s, median 2.1 s (43 dropped). M2 (claude_code) therefore uses another harness's tool times. State this in the result. Add `gap_scale` 0.25.
- m8. Dropped calls. FACT: 4,046 of 18,239 calls (22%) overlap a previous call (parallel sub-agents), and 630 have no usage. Dropping them removes real load and KV. This favours stability. The other biases (B2, B3) favour collapse. Say so in the write-up.
- m9. Cache hit credit. FACT: in 4.9% of consecutive turns the next prompt is shorter than the previous prompt plus output. In only 0.2% it is shorter by half or more. The simulator gives a full hit on the shorter prompt. Effect is small. The 128k clamp touches only the top 1% of turns (p99 = 106k, max 167k).
- m10. `_median_prompt` (M4 selection) includes zero-token and overlapping calls (`run_e28.py:36-38`). Use the same filter as `load_sessions`.
- m11. G* uses 3 seeds and the maximum of 9 caps. Expect a 3 to 5% upward bias. Use 5 or more seeds, or report G* from a smooth fit.
- m12. Finished and failed sessions keep their cache in the LRU. FACT (t11.py): at the end of a run they hold 0% to 82% of used memory. This matches vLLM (blocks stay until reused), so it is fair. It means "idle cache" is mostly dead. A gateway that frees the cache at session end is a trivial baseline. Add it as a variant.
- m13. Decision script. FACT: it follows the journal rules in order (GO-B, then KILL, then GO). I ran it on synthetic data (gen.py, dec/) and it gave the expected K1, K2 and K3 results. Differences: (a) `paired` checks only the late window, while the journal says "both" windows; (b) when there is no A collapse point but there is a C collapse point, K2 says KILL, which the journal does not declare; (c) `restore` divides by `ctrl_late` with no zero guard; (d) 7 of 10 is hard coded; (e) a mix with no A collapse point is dropped from the GO count, which is consistent with the text. None of these is a blocker. Fix (a) to match the journal, and add the declared branch for (b).
- m14. The restart cost per retry. `max_retries=2` gives 3 attempts. The journal text says "2 retries". This is consistent. No action.

## 5. Answers to the questions in the brief

1. Simulator correctness. KV accounting is right (FACT, invariant tests). Preemption order differs from vLLM (M1) and preemption cost is wrong (B3). Admission has the flush defect (B2). A full-hit admission (n = 0) does not stall: it decodes from the next step, and `evict(1)` is enough. It does not double count tokens. The `hit_tok` statistic counts a re-admission twice, but the decision does not use it. A timed-out waiting request is removed correctly and its cache stays (right). `end_session` and the gateway bookkeeping are consistent (invariant test). Events inside a long step are deferred to the step end (m1, m2). Thinning is correct. Determinism holds.
2. Cost model. See m4, m5, m6. The result depends on the miss/hit ratio, so a sensitivity check is needed.
3. Workload. Replaying hosted-model token counts is acceptable as a shape of agent traffic if you say so. It is not a fair model of an 8B model output (M6). Gaps are handled correctly, with the limits in m7. The shared prefix needs a per-harness model (M4). The 128k cap is fine. Dropping overlapping calls favours stability (m8).
4. Design. The matched control is the right idea but is not paired after the spike (M3) and the spike is small (M2). The windows are long enough for the late test, but the spike lands before steady state (m3). G* is a fair definition for a memory-bound mix. It is not for GPU-bound mixes and it makes loads of 0.8 and 0.95 unstable in an open system (M2). Turn goodput alone is not enough: add session success and shed rate (B4). A cap cannot look better by shedding in turn goodput (it gets less), but it can look worse for the wrong reason (B4).
5. Decision script. See m13.
6. Fixed in advance. Three ways: B4 can give GO-B by construction; M2 can give K1 by a weak trigger; M5 leans against KILL. Fix these first. The cap sweep cannot reach the right cap for M4 at 440k (M2).

## 6. Fix list before the run (in this order)

1. B1: fix `base_params`. Add a smoke test.
2. B2, B3, M1: fix the scheduler. Add unit tests for each rule. Re-run `test_sim.py`.
3. M3: separate RNG streams.
4. B4: session metrics, queue patience, a new "restore" definition. Amend the journal.
5. M2, m3: spike size sweep, a finer load grid based on the control knee, longer warm-up, caps up to 64.
6. M4, M5, M6, m6: per-harness prefix, a split of seeds for the oracle, an output clip variant, a constant-sensitivity grid.
7. Freeze `e28_decide.py` again after these changes (the journal says it is frozen before stage 1).

---

# Second pass (2026-10-08, after sim v2, run_e28 v2, e28_decide v2 and amendment A1)

Scope: I re-read `sim.py`, `run_e28.py`, `e28_decide.py`, `test_sim.py` and amendment A1. I ran `test_sim.py` (13 tests, all pass) and my own synthetic checks (s2.py, s3.py, n1.py, n2.py in the scratchpad folder). I did not run any stage of `run_e28.py`. I did not use the real session pool in any run, except to compute static numbers (largest request size, first-call sizes).

## Verdict

One new BLOCKER (N1, the knee rule). All earlier blockers and major items are fixed. Fix N1 before stage 0b. No other blocker.

## Status of the earlier items

| Item | Status | Evidence |
|---|---|---|
| B1 (double kwarg) | Fixed | `params()` merges into one dict and `replace()` gets no duplicate. I built params for calibrate, control and variant shapes without error. |
| B2 (flush on failed admission) | Fixed | `sim.py:364` checks `free + idle - cached < n` before any eviction. `test_blocked_head_does_not_evict` passes. |
| B3 (decode restart) | Fixed | `preempt()` folds `decoded` into `prompt_priv` and cuts `out`. FACT (s3.py): the old two-request case finished at 27.1 s, not 34 s. The extra time is the real wait for memory (1,000 remaining tokens), not a restart. |
| B4 (shedding decides K3) | Fixed | The cap stage has patience 1e9. Session success enters `restored`. See N5 for a small gap. |
| M1 (scheduler order) | Fixed | Admission-order list, victim `running[-1]`, no admission when `preempted`. |
| M2 (weak trigger) | Fixed in design, but see N1 | Spike sized in N* sessions. Caps to 64. Largest request (119k private tokens) is below the 190k cap, so no request can block the queue forever (FACT, computed). |
| M3 (pairing) | Fixed | FACT (n2.py, synthetic, 20 seeds): pre-spike windows identical; a small spike changes the late window ratio by 0.0002 on average (sd 0.002). The control is now a true paired control. |
| M4 (shared prefix) | Fixed | One prefix per harness in every mix. FACT: first-call p10 is 14,650 (claude_code), 2,575 (smolagents), 614 and 672 (others). The prefixes are at or below p10. Clamped turns: 0.7% (claude_code), 0% (smolagents). |
| M5 (oracle bias) | Fixed | Selection on seeds 1 to 5, evaluation on seeds 6 to 10, bootstrap. |
| M6 / m6 (constants, outputs) | Fixed | Stage 3 variants exist. They are reported, not decided. |
| m1, m2 (event time, late completion) | Fixed | `process_events` runs each event at its own time. `r.dead` skips late completions. |

Checks that passed (FACT): 60 storm runs with tiny caches, spikes, retries, cap policy and finite patience (s2.py). Invariants: `used` = running KV + idle KV; `idle` = sum of cached; waiting requests hold no KV; one outstanding request for each session; `n_admitted` correct; `decoded <= out`; `computed <= prompt_priv`. No assertion failed in 60 runs (1,205 preemptions, 10,190 timeouts). The decision script gives sane results on the synthetic cases (first pass) and `stable` in the final script fails on a truly stable control only 2% of the time (n2.py, 10-seed means).

## New findings

### N1. BLOCKER: the knee rule cannot pass, so K1 is nearly fixed in advance
- File: `run_e28.py:154-176` (`knees`), amendment A1 stage 0b.
- Rule: a load is stable only if the mean late goodput is at least 0.9 x offered AND at least 4 of 5 seeds have late >= 0.9 x pre. The loop stops at the first failure.
- FACT (n1.py, synthetic pool, light load, 0.6 turns/s, no preemptions, 20 seeds): the coefficient of variation of a 600 s window is 0.21. Only 10 of 20 seeds had late >= 0.9 pre. The probability that a random group of 5 seeds has at least 4 passes is 0.15. The mean-of-5 test (late >= 0.9 x offered) passes with probability 0.97.
- Effect: at the first grid point (0.30) the rule fails 85% of the time on a system that is perfectly stable. The knee becomes 0.30 (the default when nothing passes) in most variants. Stage 1 loads are then 0.18 to 0.30 x G*. That is far below any collapse. The result would be "no collapse point", and K1 would KILL I-80 for a statistical reason.
- INTERPRETATION: the per-seed test compares two independent 600 s windows of one run, so its noise is sqrt(2) x 0.21 = 0.3. A 0.9 threshold has no power at that noise.
- Fix: (1) Use only mean-based tests: mean late >= 0.9 x offered and mean late >= 0.9 x mean pre. (2) Add run-level health checks that do not depend on noise: late timeout rate below 2% and late preemption rate below a fixed small number. (3) Stop the grid scan only after two consecutive failures, and set the knee to the last pass before them. (4) Use 10 seeds for the knee, not 5. (5) Add a unit test with a synthetic stable pool that the rule passes at least 90% of the time at f = 0.3. (6) Amend the journal text (A1, stage 0b) before any run.

### N2. MINOR: admission feasibility test is stricter than needed
- File: `sim.py:364`. The test uses `- s.cached`. The true free space after the admission is `free + idle - hit`. They differ only when the cache is longer than the new prompt (about 5% of turns, small difference). The test is conservative and the later `evict` assert holds. No action, or use `- hit`.

### N3. MINOR: the knee test and the capped system disagree on what "stable" means
- The knee uses client C (600 s, 2 retries). Stage 1 uses client A (60 s). A load can be stable under C and unstable under A because of long outputs (R-15 M6). The decision uses A-collapse points, so a "collapse" under A at the knee of C may be a timeout effect and not a cache effect. The sens stage with `out_clip_1024` helps. Report timeouts and session failures per point.

### N4. MINOR: `stable` in the decision script uses 10-seed means
- It fails a stable control 2% of the time (FACT, synthetic). It is acceptable. Note that every dropped point is a lost point for K1 and K3.

### N5. MINOR: gateway wait is not part of `restored`
- With patience 1e9, a cap that queues sessions for a long time still counts as restored if goodput and success are good. This is the honest result for "a cap fixes the collapse", but a reader may see it as too kind. The gateway wait is in the output (`gw_wait`). Print `gw_wait` per oracle point in `e28_decide.py` (not a rule change).

### N6. MINOR: G* still takes the best of 11 caps from 5 seeds
- The winner's-curse bias (3 to 5%) remains and moves every load by the same factor. It does not change the collapse test, because the control uses the same rate. No action.

### N7. MINOR: first token
- A request needs one extra decode step after its prefill (the first token is not produced by the last prefill chunk). The error is one step (about 7 ms) for each turn. No action.

## Remaining items before the run

1. Fix N1 (rule, journal text, unit test). Re-freeze `run_e28.py` and `e28_decide.py`.
2. Optional: N5 print, N2 change.
3. After N1, run the stages in the order of the docstring. I recommend a read-only look at the knee file (only `frac`, `late_turn_ok`, `pre_turn_ok`, timeouts) before stage 1, to confirm that the knee is above 0.5 for most variants. If the knee is still at 0.30 for most variants, stop and report.
