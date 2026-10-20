# E-32 amendment review

**Verdict: `AMEND_BEFORE_IMPLEMENTATION`.** The amendments fix the corpus list and drop the speed kill. They do not fix the circularity, the oracle specification, or the decision rules. The README itself says not to implement before this review passes (`README.md:7`). It does not pass.

## Scope and limits

- I used Read, Grep, and Glob only. One Bash call was denied and I did not retry it.
- I read the five named files, the journal entries for I-122, I-131, E-31, E-32, R-23, R-24, and R-25, and `audit.json`. I also read the E-33 files in `experiments/session_state_corpus_pilot/`, because the journal makes E-33 the real Step 0.
- I could not read vLLM source or fetch any PR. UNVERIFIED marks every claim about vLLM internals. These come from my memory.
- FACT: R-25 was written by "Sonnet" (`RESEARCH_JOURNAL.md:4289`), and I am Sonnet 5.5. This review is a second independent review only if the amendment author was a different model family.

## Check against the R-25 amendment list

| R-25 amendment | Status | Evidence |
|---|---|---|
| 1. Corpus table | Adopted, not verified | `bug_corpus.json:6-64`; every `cpu_status` is `NOT_RUN` |
| 2. Holdout and "issue-specific" definition | **Not adopted** | `README.md:152` only admits the risk |
| 3. Independent reference model | Adopted as one sentence | `README.md:81-82` |
| 4. Oracle spec table, fake output, liveness bound | **Promised, not delivered** | `README.md:83`; the bound of 8 has no reason (`:84`) |
| 5. Preemption method | **Not adopted** | `README.md:69` still says "public preconditions" |
| 6. Baselines | Mostly adopted | Hypothesis, same oracles, 30 seeds (`:90-95`); no ablation; wall-clock budget (`:96`) |
| 7. Decision rules | Partly adopted | Order, hold band, and new-bug rule exist (`:124-132`); no outcome table |
| 8. False-positive triage | Adopted | `README.md:143` |
| 9. Speed informational | Adopted in README only | The journal still has the speed kill (`RESEARCH_JOURNAL.md:3853-3854`) |
| 10. Environment | Adopted in E-33, conflicts with E-32 | See M6 |
| 11. Freeze | Promised | `README.md:146`; git status shows the directories untracked |
| 12. Factor grid | **Not adopted** | `README.md:98-107`; `:106` keeps "where CPU tests support it" |
| 13. Source-audit evidence | Adopted | `SOURCE_AUDIT:116-118`; journal rows 54 and 55 (`:4705-4706`) |

## BLOCKER findings

**B1. The alphabet cannot express two corpus bugs, and the circularity control is missing.**
- FACT: The alphabet has "add a streaming continuation" and "return a normal token" (`README.md:59-67`). It has no truncating or replacing update and no accepted-draft output.
- FACT: B1 is a truncation bug (PR 59103 title, `audit.json:74`). B3 is an accepted-draft bug (`README.md:35`).
- FACT: The alphabet lists "pause and resume" and "reset the prefix cache". No frozen row needs either. Only excluded issue 60744 does (`audit.json:61`).
- FACT: The "issue-specific traces" kill rule is in the journal (`RESEARCH_JOURNAL.md:270`, `:3853`). It is not in the seven README rules (`README.md:126-132`).
- INTERPRETATION: To rediscover B1 and B3, the author must add bug-specific actions after reading the bugs. The alphabet already carries the 60744 trigger. The README lists the overfit risk (`:152`) but defines no control.

**B2. The action and oracle specification does not exist, so false positives cannot be judged.**
- FACT: `README.md:83` says to define preconditions and legal outcomes "in the harness specification". No such file exists.
- FACT: "Same logical trace with and without the transformation" (`README.md:80`) needs a map from step-indexed actions to logical positions. Preemption adds recompute steps. The map is not defined.
- FACT: "Return a stop token" (`README.md:65`) has no stop position rule. A stop chosen by step count changes meaning after a preemption.
- UNVERIFIED: "Preempt a request" is a private scheduler call. The harness can reach states the scheduler never reaches.
- UNVERIFIED: In async mode, computed progress runs ahead of committed tokens through placeholders. Oracle 5 (`README.md:77`) would flag this legal state.
- UNVERIFIED: Engine outputs return in FIFO order. If "complete an older in-flight step" (`README.md:67`) allows reordering, it is an illegal action.
- FACT: Oracle 6 uses a bound of 8 steps (`README.md:84`). A paused scheduler, a full block pool, or a session waiting for input can legally block longer.
- INTERPRETATION: Rules 2 and 3 depend on the false-positive rate. That rate depends on this missing specification.

**B3. Key terms in the decision rules are undefined.**
- FACT: "Match" (rule 5, `:130`), "misses" (rule 7, `:132`), "new" and "confirmed" (rule 4, `:129`) have no definition.
- FACT: "Current main" has no pinned SHA. PR 60764 is open (`audit.json:123`) and can change main.
- FACT: The sequence budget comes "from the Step 0 setup pilot" (`README.md:102`). Step 0 runs before any generator exists (`:44`). Step 0 cannot measure a per-sequence cost.
- INTERPRETATION: With four bugs and 30 seeds, "found" must be a per-seed rate with a threshold. The README gives none.

## MAJOR findings

**M1. Current tests lose on every known bug by construction.**
- FACT: Baseline 1 (`README.md:89`) is the parent-commit suite. These parents failed to catch the bugs.
- INTERPRETATION: The claim "more than current tests" (`README.md:11-12`) can be tested only on pinned current main, or on a mutant set. Say so in the README.

**M2. Budget and rule order favor the wrong evidence.**
- FACT: Policies get equal wall-clock budgets (`README.md:96`). Rows 90-91 say "action budget". Hypothesis has more per-example overhead than custom code.
- FACT: Rule 5 runs before rule 4. A tie on the circular known-bug corpus can kill the guided claim before new-bug evidence counts.
- FACT: Rule 6 gives PIVOT when no new bug appears. `README.md:153` says a tool result is "not enough for a paper".
- INTERPRETATION: Rule 6 hides a thesis-level kill under the word PIVOT.
- FACT: Rule 7 asks for minimized traces of 12 actions or fewer (`:132`). The maximum generated length is 12 (`:103`). The clause is vacuous.
- FACT: No outcome covers a new bug that random also finds, or one known-bug win over random (rule 7 needs two).

**M3. Controls are not fully defined.**
- FACT: "1,000 control sequences" does not say per commit, per cell, or in total. The control commits are not listed.
- FACT: Open issues 49449 and 60744 may still fail on main (`audit.json:5-14`, `:55-64`). 60744 has a CPU reproducer. The README does not say how to count them.
- FACT: The hold band of 1 to 10 failures (`README.md:128`) has no exit rule.

**M4. The factor grid is not frozen and is very large.**
- FACT: 3 lengths x 2 sync/async x 2 cache x 2 spec x 2 reset = 48 cells (`README.md:103-107`).
- FACT: 48 cells x 5 policies x 30 seeds = 7,200 runs per commit. Nine commits (four parents, four fixes, main) give 64,800 runs.
- INTERPRETATION: At 10 s per run, that is 180 hours on one container. At 60 s, it is 1,080 hours.
- INTERPRETATION: Each bug needs one setting (B2 needs async, B3 needs spec, B1 needs cache). If a run uses bug-specific settings, that is an issue-specific config.

**M5. The corpus is a convenience sample, and a small count kills the idea.**
- FACT: All four fixes merged between 2026-09-28 and 2026-10-17 (`audit.json:71`, `:101`, `:111`, plus PR 58668 in the journal).
- FACT: No mining protocol exists (query, date range, inclusion rule, count screened).
- FACT: Fewer than four eligible rows "kills I-131" (`README.md:51`). E-33 says only "KILL-CURRENT-CORPUS" (`RESEARCH_JOURNAL.md:3868`).
- INTERPRETATION: One flaky or API-dependent test kills the idea on the corpus, not on the research question. PIVOT-CORPUS (two or three pairs) has no defined action.

**M6. Step 0 and E-33 are two conflicting specifications.**
- FACT: Step 0 runs "the named regression test at the exact parent commit" (`README.md:46`). The test does not exist at the parent. E-33 copies the test-only diff (`E-33 README:20`).
- FACT: E-32 says to bake source and model config into the image (`README.md:22`). E-33 uses an official image digest (`E-33 README:55`). E-33 says nothing about model config.
- FACT: Two `bug_corpus.json` files exist with different schemas. B1 has a class path in one file and not in the other.
- FACT: `audit.json` has fix SHAs for B1, B2, and B4. It has no SHA for B3's fix (PR 58668) and no parent SHA for any row.
- FACT: E-33 predeclares no expected failure assertion. "Documented fault" (`E-33 README:32`) is a human judgment. It runs each test once.
- UNVERIFIED: Scheduler tests need a Hugging Face model config. With no network, that causes a shared setup failure.
- UNVERIFIED: The image digest may be amd64-only, or its compiled extensions may not match newer parent commits. The image's vLLM version is not recorded.

**M7. The recurrence evidence is not shown.**
- FACT: Issue 58776 opened 2026-09-25 (`audit.json:47`), before PR 58259 merged on 2026-09-28 (`:101`). The audit text is now correct (`SOURCE_AUDIT:116-118`).
- FACT: The four fixes are narrow and in different code (`SOURCE_AUDIT:84-87`).
- FACT: Each corpus bug pairs the session path with one other feature. Examples: truncation with cache, async with resume, spec with stop, preemption with stop.
- INTERPRETATION: The data supports "bugs exist in this path". It does not show "bugs recur after narrow fixes". It also fits "maintainers fix reports quickly".
- INTERPRETATION: Pairwise bugs may be found by running the existing tests over a feature grid. That baseline is missing.

**M8. The journal and the README disagree.**
- FACT: The journal E-32 entry still says six pairs, zero false positives, four lengths, and a 100 sequences/s kill (`RESEARCH_JOURNAL.md:3849-3856`).
- FACT: The README has four rows, three lengths, and no speed kill. AGENTS.md says the journal wins. Section 1 and `:270` also lag.
- FACT: The E-32 and E-33 directories are untracked. The clean-tree rule (`README.md:146`) cannot hold yet.

## MINOR findings

- FACT: `SOURCE_AUDIT:47` still pairs issue 42655 with PR 58259. The corpus says the link is "not established" (`bug_corpus.json:53-58`).
- FACT: The shrinker is not named. The same shrinker must serve all policies, or the 12-action comparison is unfair.
- FACT: `README.md:154` lists "cross-engine transfer" as a publication route. No step measures it.
- FACT: The R-25 heading is dated 2026-10-17 (`RESEARCH_JOURNAL.md:4287`) and the log entry is dated 2026-10-19 (`:1179`).

## Answers to the twelve questions

1. **Generic property testing plus vLLM?** Mostly yes today. The domain content is two undefined items: the oracles and the fake output (B2). The README says so (`:153`). The realistic fit is ISSTA or an MLSys tool track, not OSDI or SOSP. This is interpretation.
2. **Recurring problem after narrow fixes?** Not shown (M7).
3. **Four distinct items, reproducible on CPU?** Distinct by root cause, yes. Evidence for B1, B2, and B4 is in `audit.json`. CPU reproduction is not shown (M6). Every row is `NOT_RUN`.
4. **Circular?** Partly yes (B1). A new bug on pinned main is the only non-circular evidence.
5. **Relations valid for every action?** Not shown, and some are likely invalid (B2).
6. **False positives after legal preemption or reset?** Likely, under async progress, reset no-ops, and the liveness bound (B2).
7. **Strong baselines?** Random: mostly yes, except the budget unit (M2). Current tests: no, except on main (M1).
8. **Can the generator pass by encoding the four traces?** Yes. Any alphabet change after reading the bugs does this (B1).
9. **Fair rules?** No (B3, M2, M3).
10. **Speed treatment fair?** Fair as informational. Equal wall-clock budgets reintroduce speed (M2). UNVERIFIED: Docker Desktop timing on the M4 is noisy because the VM maps to mixed core types.
11. **vLLM CPU image in 8 GB?** INTERPRETATION: The official image plus a Python overlay should fit. The VM has 7.65 GiB (`RESEARCH_JOURNAL.md:354`) and the container cap is 4 GiB. A full C++ source build is not advisable at this memory size, and tests do not need it. Image architecture, vLLM version, and offline model config are unverified (M6).
12. **Cheapest kill experiment?** See below.

## Cheapest kill experiments

Run these in order. Stop at the first kill.

1. **Paper audit (about 1 hour, no Docker).** Write each fix test as a sequence of actions in the alphabet. Run `git blame` on the removed lines of each fix. If every sequence is 6 actions or fewer, exhaustive enumeration beats guidance and the guided claim dies. If no buggy line came from an earlier fix, the recurrence premise is unsupported.
2. **E-33 (hours).** It kills the current corpus only. It cannot kill I-131 alone.
3. **Hypothesis `RuleBasedStateMachine` with the same oracles (about 1 day after the oracle spec).** Run it on the four parents and on pinned main, with an equal sequence budget. If it finds all known bugs and no new bug, kill the idea.

## Smallest exact amendment set

**A1. Merge Step 0 into E-33 and fix E-33 before it runs.**
- Replace `README.md:42-55` with a pointer to E-33.
- Make the E-32 `bug_corpus.json` point to the E-33 file by SHA-256.
- Add to E-33: an expected exception and assertion line per row; three repeats per run; baked Hugging Face config with offline mode; arm64 manifest digest, vLLM version, torch and transformers versions; a check that `git rev-parse <fix>^` equals `parent_sha`; a smoke test on an unrelated test before the first parent run; the commit SHA and clean-tree flag.
- Commit both directories first.
- Define the action for PIVOT-CORPUS.
- Change `README.md:51` to "kill the corpus", not "kill I-131".

**A2. Write the action and oracle table before any code.** Freeze its hash.
- Give every action its parameters, precondition, legal outcomes, and applicable oracle.
- Include a truncating update and k accepted drafts.
- Include a stop at a logical position.
- Define the fake output as a pure function of (session, logical position, stop position).
- Define the map from actions to the logical trace.
- Define how preemption happens (KV pressure, or a private call marked as a validity risk).
- Make completion FIFO.
- Define oracle 5 for async placeholders.
- Give the liveness bound a reason and list exemptions.

**A3. Add the circularity control.**
- Define "issue-specific": any config, weight, guidance, alphabet, or oracle input that differs by bug or commit.
- Use one generator config hash for all commits.
- Restore the kill rule for issue-specific inputs in the README.
- State that the four-bug screen is a recall check and cannot support "finds more bugs".
- Add a prospective holdout: freeze the harness on a date, then test on session-path fix PRs merged later.

**A4. Fix baselines and budgets.**
- Use a sequence budget as the primary budget, with a value written now or a rule that uses only random-baseline pilot data.
- Report wall-clock as secondary, with policies interleaved.
- Use one shrinker for all policies.
- Add bounded exhaustive enumeration to depth 6 as a baseline.
- State that baseline 1 is compared only on pinned main.

**A5. Replace rules 1 to 7 with an outcome table.** These numbers are project heuristics.
- "Found" means the signature appears in at least 15 of 30 seeds within budget. "Missed" means 3 or fewer.
- "Match" means the baseline's found set contains the generator's found set, or median executions to first failure is within 2x on each shared bug.
- Pin a current-main SHA before any generator run.
- "New" means no matching open or closed issue or PR. Count 49449 and 60744 as rediscovered open issues. "Confirmed" means a public-API reproduction plus a second reviewer. Ask the user before any upstream filing.
- Apply rule 5 to known and new bugs together.
- Name the no-new-bug outcome "PIVOT_TOOL_ONLY" and state it is not a thesis claim.
- Cover the new-bug-but-random-also-finds-it case.
- Remove the vacuous 12-action clause, or generate up to 32 actions and require 12 or fewer after shrinking.
- State the false-positive count per fixed commit and per mode. Define the exit from the hold band.

**A6. Freeze the factor grid.**
- List the exact cells, run all cells on all commits, and fix the GO trace length.
- Run a smoke test to decide spec decoding before the freeze.
- Write the compute total (cells x policies x seeds x commits x budget).

**A7. Sync the journal.**
- Rewrite the E-32 entry (`RESEARCH_JOURNAL.md:3844-3860`), the I-131 kill rule (`:270`), and Section 1.
- Log the decision owner in Section 10.
- Fix `SOURCE_AUDIT:47`.

**A8. Add the paper audit as Step -1 and log its result in the journal.**

A1 gates E-33. A2 to A8 gate E-32. Request a short third review of the A2 table and the A5 outcome table before any harness code.

**Verdict: `AMEND_BEFORE_IMPLEMENTATION`.**
