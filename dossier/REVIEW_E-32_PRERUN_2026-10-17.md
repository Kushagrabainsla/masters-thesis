I have read all the inputs and the audit data, so I am now writing the review. I ran no code and edited no file.

# E-32 independent pre-run review

**Verdict: `AMEND_BEFORE_IMPLEMENTATION`.** The idea is not dead. The design cannot yet give a fair GO or KILL. The amendment set is at the end.

Scope notes:
- I could not read the vLLM source or the fix PRs. I did not search the web.
- Statements about vLLM internals are marked UNVERIFIED.
- I checked the `audit.json` data for claims about PR and issue dates.

## BLOCKER findings

**B1. The corpus is not verified, and the GO denominator can change.**
- Fact: `audit.json` lists 6 issues and 6 PRs. PR 44526 and PR 58668 are not in it. `README.md:27` and `README.md:30` name them. Searches of the repo find no other record of them. The journal calls PR 58668 "full-text" at `RESEARCH_JOURNAL.md:4606`, but it stores no parent SHA.
- Fact: PR 60764 is open (`audit.json:121-122`). The SHA `ef0cad2…` is a test-merge SHA of an open PR. It is not a stable fix commit. Item 6 has no fixed control.
- Fact: Issue 42655 closed as `not_planned` (`audit.json:40-41`). PR 43683 closed unmerged (`audit.json:89-91`). The title of issue 42655 is about the "request ID -> previous batch row mapping" (`audit.json:41`). That mapping is GPU model-runner state. `README.md:130` says a CPU scheduler test does not cover it. Item 3 pairs this issue with PR 58259, a scheduler fix. No record shows that PR 58259 fixes issue 42655.
- Fact: `README.md:34-35` lets the author drop and replace cases before the frozen run. GO needs "four of six" (`README.md:98`). If two cases drop out, the rule has no defined meaning.
- Interpretation: No hard evidence shows that at least four of six bugs reproduce on CPU. GO is untestable until each item has a recorded parent SHA, fix SHA, and CPU reproduction.

**B2. The test is circular as designed.**
- Fact: The action alphabet (`README.md:39-47`) contains the trigger of each corpus bug. Examples are the stop token, preempt, prefix-cache reset, older in-flight completion, and session continuation. The oracles (`README.md:53-58`) map almost one to one to the bugs. Cache identity matches 49377. Queue ownership matches 58776. Progress bounds match 44526 and 58668. Liveness matches 60744.
- Fact: "Without issue-specific traces" (`README.md:98`, `README.md:112`) has no operational definition. Alphabet weights, preconditions, and "trace guidance" can all encode the bugs.
- Fact: `README.md:133` already admits the overfit risk. No control is defined for it.
- Interpretation: "Rediscover 4 of 6" measures whether the author put the bug's actions into the alphabet. It does not measure bug-finding power. The R-23 faults (B1, B2 in the E-31 review) were the same kind of error: the design encoded the answer.

**B3. The baselines cannot decide the KILL rule.**
- Fact: `README.md:65` ("current vLLM scheduler and streaming test suites") is run on parent commits. By definition, the parent test suite did not catch these bugs. The baseline loses on every rediscovery case. The KILL clause "current tests match the full generator" (`README.md:111`) cannot fire on the corpus. It can only fire on new bugs, and no step in the design hunts for new bugs.
- Fact: `README.md:66` says "uniform random valid actions". The text does not say that random runs use the same six oracles. Without the oracles, random loses by construction.
- Fact: The "equal execution budget" has no number. The sequence budget is a variable (`README.md:72`) with no value. The seed count is not defined.
- Interpretation: With n=6, "two bugs that random misses" (`README.md:99`) is noise unless random runs use many seeds. Off-the-shelf Hypothesis `RuleBasedStateMachine` with the same oracles is the strong baseline. It is the one that would show whether the domain-aware generator adds anything. It is not in the baseline list.

**B4. The oracles are not valid relations yet. False positives are likely.**
- Fact: `README.md:49` says the generator uses only actions with public preconditions. `README.md:53-58` never lists them. "Preempt a request" (`README.md:43`) is not an engine API. Preemption happens inside the scheduler when KV blocks run out. UNVERIFIED: the harness probably calls a private method, or it must create real KV pressure.
- Fact: Several actions can legally change the outcome:
  - A prefix-cache reset returns "false" while blocks are in use. UNVERIFIED: this matches my memory of vLLM, and I did not check the source.
  - A pause in "abort" mode finishes requests.
  - Async scheduling uses placeholder tokens and a one-step lag after a stop.
  - "Complete an older in-flight step" (`README.md:47`) has no meaning in sync mode.
- Fact: "Logical tokens" (`README.md:53`) equal across a transformation only if the fake `ModelRunnerOutput` is a pure function of (session, position). The README does not define the fake output.
- Fact: "Bounded number of steps" (`README.md:58`) has no bound. A paused scheduler, a streaming session waiting for input, or KV exhaustion can legally block work.
- Fact: Oracle 4 needs the hash of the current token block. If the harness reads `request.block_hashes` for the expected value, it is the E-31 B1 error again. Oracle 5 has the same risk if it reads the scheduler's own token fields.
- Interpretation: Oracles 3 to 6 are state invariants. Only oracles 1 and 2 are metamorphic. The name "metamorphic" overstates the design. A reference model that is independent of the scheduler is missing.

## MAJOR findings

**M1. Decision rules leave gaps and mix feasibility with science.**
- Fact: Zero false positives are needed for GO (`README.md:100`). More than 1% kills (`README.md:111`). One to ten false positives in 1,000 sequences fall in neither band.
- Fact: Nothing says which rule wins when GO and KILL both hold. Outcomes like 3 of 6 are not covered. PIVOT ("only within one fixed feature pair", `README.md:105`) is vague.
- Fact: GO needs no new bug on current main. `README.md:134` says a paper needs new bugs. The screen never measures the one thing that matters for publication.
- Fact: The speed KILL (`README.md:113`) kills on setup cost, not on the research question.
- Fact: `README.md:100` and `README.md:112` do not say whether "1,000 control sequences" is per commit or in total.

**M2. The false-positive definition is circular and the controls are contaminated.**
- Fact: A fixed commit is a main commit between fixes. Issue 49449 and issue 60744 were open at that time (`audit.json:6-7`, `audit.json:56-57`). A real bug at a "fixed" commit counts as a false positive under `README.md:111`.
- Fact: `README.md:124` says "keep fixed-commit failures separate". It gives no triage rule. The author who writes the oracles also judges the false positives.
- Fact: Only "valid" sequences count (`README.md:111`). The generator defines "valid". A filter can hide oracle failures that come from ambiguous preconditions.

**M3. Setup is not shown to be feasible, and one source conflicts with the design.**
- Fact: `SOURCE_AUDIT_I-122_2026-10-17.md:161` says "official ARM CPU container". `README.md:21` says "Pin every tested vLLM commit". An official image is tied to a release tag. It cannot hold arbitrary parent commits.
- Fact: Docker reports 7.65 GiB of memory (`RESEARCH_JOURNAL.md:3829`). A plan to run 12 or more commits needs a per-commit setup.
- Interpretation (UNVERIFIED): A full CPU build from source with C++ extensions for 12 commits on ARM is slow. The scheduler tests probably need only the Python package. I believe `VLLM_TARGET_DEVICE=empty` skips extension builds, but I did not verify it.
- Interpretation (UNVERIFIED): The scheduler test helpers may build a `ModelConfig` from a Hugging Face model id. That needs the network or a baked config. The E-31 audit container had network access, but a frozen run should not.
- Interpretation (UNVERIFIED): The dependency pins change between the earliest and latest commit in the corpus. One image may not fit all of them.

**M4. The 100 sequences per second threshold has no basis.**
- Interpretation (UNVERIFIED): Per-sequence cost depends on setup. Building `VllmConfig` and the scheduler can cost more than 10 ms. Reusing one scheduler leaks state between sequences, and that leak can create or hide failures.
- Fact: E-31 treated Docker timing as report-only because host load affects Docker Desktop (`RESEARCH_JOURNAL.md:3811`). E-32 turns timing into a KILL rule (`README.md:113`). The two rules conflict.
- Fact: The README does not say how speed is measured: warm-up, pinned CPUs, host idle, or median over runs.
- Interpretation: At 100 sequences per second, one hour gives 360,000 sequences. A lower rate may still be enough. The number is not tied to a bug-finding need.

**M5. The evidence of a recurring problem is weak.**
- Fact: `SOURCE_AUDIT_I-122_2026-10-17.md:118` says issue 58776 "reproduced after that fix" (PR 58259). `audit.json:49` shows issue 58776 was created 2026-09-25. PR 58259 merged 2026-09-28 (`audit.json:101`). The issue is older than the fix. The audit data does not show that it reproduced after the fix. A comment in the issue might show it, but no record exists.
- Fact: Each closed bug has a narrow fix in a different component (`SOURCE_AUDIT…:49-50`, `RESEARCH_JOURNAL.md:4257`). The audit says so itself.
- Interpretation: Distinct bugs fixed within days support "bugs exist". They do not support "bugs recur after narrow fixes". The same data fits the reading "maintainers fix reported bugs fast, and a tester adds little". The data supports the existence of the problem class. It does not support the need for the tool.
- Fact: PR 37160 is titled "Simple yet General CPU KV Cache Offloading" (`audit.json:84`). Issue 36755 is about a preemption plus async race (`audit.json:31`). `SOURCE_AUDIT…:46` lists PR 37160 as the fix for issue 36755. Check the link. It may be wrong.

**M6. The test-gap evidence is narrow.**
- Fact: The scan covers seven files and the regex `from hypothesis|import hypothesis|RuleBasedStateMachine` (`source_audit.py:141-143`). It does not cover other test directories, other fuzzers, or CI jobs.
- Fact: The journal marks the closest generator paper, "Trace-Guided Synthesis of Effectful Test Generators", as "search-only" (`RESEARCH_JOURNAL.md:4610`).
- Interpretation: "No stateful property generator" holds for seven files only. The prior-art claim has medium-low confidence (`SOURCE_AUDIT…:206-208`). I agree with that rating.

**M7. The freeze and the factor grid are not controlled.**
- Fact: The E-32 directory is untracked in git (`??` in the session-start status). R-23 M5 reported the same fault for E-31. `README.md:121-126` records the vLLM source commit but not the harness commit or a dirty flag.
- Fact: The factors (`README.md:70-79`) give 4 lengths × 2 × 2 × 2 × 2 = 64 configurations. These multiply by 12 commits and 4 policies. No subset is frozen, and no value fixes the sequence budget.

## MINOR findings

- `README.md:112` and `README.md:98` ask for at most 12 actions. The length factor includes 20 (`README.md:74`). State which length the GO test uses.
- `README.md:46` ("where CPU tests support it") lets the author drop speculative decoding after seeing results. Fix the list before the run.
- The shrinker is not named. Say whether it is Hypothesis or custom code.
- The controls are good: planted faults per oracle, replayed minimized traces, and a manifest (`README.md:119-126`). Keep them.

## Answers to the twelve questions

1. **Generic property testing plus vLLM?** Mostly yes today. The domain content is the six oracles and the fake-output design, and the README does not define either (B4). Without new bugs, the realistic outcome is a testing tool or an upstream contribution. That is the PIVOT outcome already in `README.md:105-107`. Fit with OSDI or SOSP is unlikely. ISSTA or MLSys tool tracks are possible. This is interpretation.
2. **Recurring problem after narrow fixes?** Not shown (M5). The data shows that distinct bugs exist.
3. **Six distinct items, reproducible on CPU?** Not shown (B1). Items 1 and 4 have no record. Item 3 may be GPU runner state. Item 6 has no merged fix. Item 5 looks scheduler-level, as does item 2, but neither has a CPU run.
4. **Circular?** Yes (B2).
5. **Relations valid for every action?** No (B4).
6. **False positives after legal preemption or reset?** Yes, they are likely (B4, M2).
7. **Strong baselines?** No (B3). Current tests lose on the parent by definition. Random has no budget or oracle parity.
8. **Can the full generator pass by encoding the six traces?** Yes (B2). Trace guidance and the alphabet can carry the bugs.
9. **Fair rules?** No (M1).
10. **100 sequences per second meaningful?** No (M4).
11. **Feasible in 8 GB?** Probably, with a source overlay and no extension build. The official-image plan fails for pinned commits. A one-commit pilot is needed first (M3).
12. **Cheapest experiment that can kill I-131?** See below.

## Cheapest kill experiment

Run two steps in order. Each step needs no custom generator.

1. **Step 0, corpus and setup pilot (hours).** For each corpus item, record the parent SHA, fix SHA, and CPU status. Use the fix PR's own regression test as the reproducer. It must fail on the parent and pass on the fix, inside the Docker image. If fewer than four items pass, GO is impossible. Kill or redesign the corpus first. Record the per-sequence setup time here too.
2. **Step 1, off-the-shelf random baseline (about one day).** Write one Hypothesis `RuleBasedStateMachine` with the six oracles and uniform actions. Give it a fixed budget on each parent. Run it on current main as well. If it finds at least four of the known bugs, I-131 reduces to generic property testing. KILL the domain-aware generator claim. If it finds no new bug on current main after the budget, record that too. Build the guided generator only after this step.

## Smallest exact amendment set

1. **Corpus table.** Add `bug_corpus.json` rows before coding. Each row needs the parent SHA, fix SHA, the fix PR's regression test, and the CPU result. Move item 6 to "no fixed control" until PR 60764 merges. Remove or justify item 3. Set the minimum corpus size and fix the GO denominator at it. Forbid replacement after the freeze.
2. **Holdout and circularity control.** Split the corpus. Design the alphabet and oracles on three items, and freeze them. Test on the other three, which the author reads only after the freeze. Define "issue-specific" as: any config, weight, or guidance input that differs per bug or commit. Use one generator config hash for all commits.
3. **Reference model.** The harness keeps its own logical token history, updated only by generated actions. Oracles 4 and 5 recompute from this model. They never read the scheduler's own fields as the expected value.
4. **Oracle spec table.** For each action, list the precondition, the legal outcomes, and the oracle that applies. Include reset returning "false", pause mode, async lag, and the sync-only status of "older in-flight completion". Define the fake `ModelRunnerOutput` as a pure function of (session, position). Give the liveness bound as a number with a stated reason. Rename oracles 3 to 6 as invariants.
5. **Preemption action.** State how the harness preempts: by KV pressure or by a private call. If private, mark it and count it as a validity risk.
6. **Baselines.** Add Hypothesis `RuleBasedStateMachine` with the same oracles. Give every policy the same oracles and the same budget in wall-clock and in sequences. Run at least 30 seeds per policy. Report the median and range of executions to first failure. Add "existing tests plus oracle hooks" as an ablation. Rewrite the "current tests" baseline as the parent-commit suite, and say it is expected to miss.
7. **Decision rules.** Replace rules with a table that covers all outcomes. Set a precedence order. Cover 1 to 10 false positives per 1,000 (neither GO nor KILL, so triage). Say whether 1,000 is per commit. Require at least one new, confirmed bug on current main, or an explicit statement that GO does not claim it. Define PIVOT by a count of bugs and feature pairs.
8. **False-positive triage.** Before the run, write a rule. Any failure at a fixed commit gets a written contract check against vLLM behavior. A confirmed true bug moves to a separate list. The triage uses a second reviewer or a documented source citation, not the oracle author alone.
9. **Speed.** Make speed informational, as E-31 did. Pilot it in Step 0. Report setup and per-action cost separately. Set the sequence budget from the pilot. Keep KILL only if the pilot proves that a useful budget costs more than the thesis can afford.
10. **Environment.** Drop "official ARM CPU container". Pin the base image by digest. Pin each vLLM commit by source overlay. Bake the model config into the image so the run needs no network. Run Step 0 on one parent commit before anything else. Record `cpu_set`, memory limit, and Docker VM size.
11. **Freeze.** Commit the E-32 directory and thresholds before any parent run. Record the git SHA and a clean-tree flag in the manifest. Stop the run if the tree is dirty.
12. **Fix the factor grid.** List the exact configurations to run. State the trace length used for GO. Fix the speculative decoding choice before the run.
13. **Evidence fixes in the source audit.** Correct line 118 of `SOURCE_AUDIT_I-122_2026-10-17.md`. Either cite the comment that shows the reproduction after PR 58259, or remove the claim. Check the PR 37160 link to issue 36755. Log both in Section 8 of the journal if they are wrong.

After these amendments, ask for a short second review of the corpus table and the decision table. Then implement.
