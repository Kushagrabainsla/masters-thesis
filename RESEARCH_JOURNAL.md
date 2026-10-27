# Research Journal: Master's Thesis Source of Truth

Owner: Kushagra Bainsla, MSCS, San Jose State University
Created: 2026-09-23
Last updated: 2026-10-25
Protocol version: 1

This is the single source of truth for the whole research journey: every idea, every
experiment, every review, every kill, and the current plan. If another file in this repo
disagrees with this journal, this journal wins. Older files are historical snapshots.

## Contents

- 0. How to use and maintain this document
- 1. Current state (the only section that is rewritten each session)
- 2. Context and constraints
- 3. Idea registry
- 4. Timeline (append-only)
- 5. Idea dossiers
- 6. Experiments and reviews
- 7. Prior-art ledger
- 8. Errors, retractions, and do-not-cite list
- 9. Artifact map and reproduction
- 10. Decision log
- Appendix A. Source file index
- Appendix B. Compute and funding programs
- Journal changelog

---

## 0. How to use and maintain this document

### 0.1 Reading order

Read Section 1 first. It says where we are and what to do next. Use Section 3 to find any
idea by ID. Go to Section 5 for its full history. Everything else is reference.

This file is long (about 1,900 lines). An AI agent starting a session should read Sections
0 and 1 in full, then search by ID (`I-23`, `E-17`, `R-05`) instead of loading the whole file.
Load it all only for a review pass or when writing a new dossier entry that must cite prior
work.

### 0.2 Rules

1. **Append, do not erase.** History sections (4, 5, 6, 7, 8, 10) only grow. A killed idea
   stays in the registry forever. A wrong claim is marked RETRACTED with a date and reason
   in Section 8, never deleted.
2. **Section 1 is the exception.** Rewrite it at the end of every session so it always
   reflects the truth right now. Move what it used to say into Section 4 as a dated entry.
3. **Dates are absolute** (YYYY-MM-DD). File modification times are labelled as such.
4. **Every claim has a label.** FACT (a source or a run of code showed it), INTERPRETATION
   (my reading of facts), HYPOTHESIS (a prediction a test could refute). Never turn an
   interpretation into a fact.
5. **Predeclare kill criteria before running anything.** Write the GO, PIVOT and KILL
   thresholds into the experiment entry first, then run. Thresholds that are project
   heuristics, not literature standards, are labelled as heuristics.
6. **"I did not find" is not proof of absence.** Novelty claims are stated conservatively
   and name the closest threats.
7. **Preserve failed and invalid experiments.** They explain why the surviving questions are
   narrow. Mark invalid results INVALID and say why. Never cite an invalid result as
   evidence.
8. **Recommendations are not decisions.** A decision is something the user chose. Record
   who decided in Section 10. Until the user decides, an AI recommendation stays a
   recommendation.
9. **Style:** write in ASD-STE100 (Simplified Technical English) from 2026-10-06 (D-27).
   Direct, simple technical language, no em dashes, explain mechanisms rather than restating
   code (from `agents.md`). Older entries stay as they are.

### 0.3 Status vocabulary

| Status | Meaning |
|---|---|
| GO | Gap is defensible and decisive validation is positive enough to justify system work |
| MEASURE-FIRST | Opportunity itself is not established. Run the oracle or workload study before building |
| PIVOT | Domain is promising but the current mechanism or question is weak or answered |
| HOLD | Parked with low confidence. Not killed, not active |
| DEPRIORITIZED | Crowded or weak, but not killed by our own experiment |
| KILL | Strong evidence says solved, trivial, too small, or infeasible. Reopen only with new information logged in Section 4 |
| DORMANT | Proposed but never adjudicated in this repo |
| SCANNED | Searched once for context, not developed. Treated as crowded until reviewed |
| HYPOTHESIS | Candidate gap with no decisive test run yet |

Confidence in the public gap: **high** (multiple current searches and adjacent checks found
no direct answer, distinction is clear), **medium** (gap looks real but fast-moving repos or
unpublished work add uncertainty), **low** (thin, compositional, or rests on an unvalidated
workload assumption).

### 0.4 Source check levels (used in Section 7)

| Level | Meaning |
|---|---|
| full-text | Full text or source code was read |
| checked | Primary page opened on 2026-09-23 (abstract or issue text) and the relevant claim confirmed |
| search-only | Only a search-result snippet was seen |
| doc-only | Cited by an earlier repo document and not independently verified |

### 0.5 IDs

| Prefix | Object | Where |
|---|---|---|
| I-nn | Idea | Sections 3 and 5 |
| E-nn | Experiment or code-level check | Section 6.1 |
| R-nn | Review pass | Section 6.2 |
| D-nn | Decision or recommendation | Section 10 |

IDs are never reused. Old names ("Idea 1", "RQ1", and so on) are aliases listed in Section 3.
Note that "Idea 1" means different things in different old files.

### 0.6 What to update for each kind of event

| Event | Do this |
|---|---|
| New idea | Add a row to Section 3. Add a dossier in Section 5 using template 0.7a. Add a Section 4 entry. Update Section 1 if it becomes active |
| New literature | Add rows to Section 7 with the check level. Add a dated line to the idea's "Research log". Update the idea's verdict only through a review |
| New experiment | Write the entry in Section 6.1 with predeclared criteria before running. Store code in `experiments/<name>/`, results next to it, and a zip in `dossier/bundles/`. Fill in the result after |
| Adversarial review | Follow the `research-gap-dossier` skill. Add an R-nn entry in 6.2. Append a line to each affected idea's "Verdict history". Update the Section 3 status column and Section 1 |
| Kill | Keep the entry. Fill "Why closed" and "What would reopen it" |
| Retraction | Add a row to Section 8. Do not edit the original claim |
| Decision by the user | Add a D-nn entry with "decided by: user" |
| End of session | Update "Last updated" at the top, rewrite Section 1, add Section 4 entries |

### 0.7 Templates

**a. Idea dossier (Section 5)**

```
### I-nn Name
Status: <status>   Confidence in gap: <low|medium|high>   Aliases: <old names>
Origin: <date, where it came from>
Problem: <what concrete system behavior is inefficient, incorrect, or unknown>
Research question: <falsifiable, comparative, exposes what result means "no">
What prior art already answers: <short table or bullets, with Section 7 pointers>
Exact surviving gap: <smallest unanswered sentence, and what not to claim as novel>
Experiments: <E-nn list with one-line results>
Reviews: <R-nn list>
Research log: <dated lines, newest last>
Verdict history: <dated lines, newest last. Never overwrite>
Why closed / what would reopen it: <only for KILL, DEPRIORITIZED, HOLD>
Next step and kill rule: <only for active ideas>
Lessons: <what this taught us about method>
```

**b. Experiment (Section 6.1)**

```
E-nn  Title  (date, idea IDs)
Question:
Predeclared criteria (GO / PIVOT / KILL, heuristics labelled):
Setup: <hardware, software, model, seeds, repetitions, ordering>
Result:
Validity and limits: <VALID | INVALID | SCREEN ONLY, and why>
What it changed:
Artifacts: <paths>
```

**c. Timeline entry (Section 4)**

```
YYYY-MM-DD  [phase]  What happened. Source: <file or URL>. Ideas: I-nn. Experiments: E-nn.
```

**d. Review (Section 6.2)**

```
R-nn  Title  (date)
Scope: <candidates reviewed>
Method: <searches, checks, code re-runs>
Findings: <numbered, each labelled FACT / INTERPRETATION>
Verdicts: <per idea>
Full write-up: <path>
```

### 0.8 Relationship to older files

Older files stay as they are. They are raw sources and snapshots. Do not edit them except
to add a pointer to this journal. Files that become obsolete are moved to `archive/`, never deleted, with a row in
`archive/README.md` giving the original path, the reason, and what replaces them. Only true
junk (`__pycache__`, `.DS_Store`) is deleted. On 2026-09-24 the `ideas/` briefs and
`dossier/RESEARCH_STATUS.md` were archived as superseded. Sections 4, 5 and 10 and older
snapshots still name the original paths, so use `archive/README.md` to find them. Appendix A
maps every file to what it contains.

---

## 1. Current state

**As of 2026-10-26 22:00 UTC, after E-36 Session 1 on the H100 and R-42.** Rewrite this section every session.

**RESUME POINT (read this first).**
- State: E-38 rerun (D-62) done 2026-10-27, Pod 79jekplym1wxsh, about $7.60. Frozen decision: KILL (I-23
  closed for Hopper block-FP8 at these configs). All real candidates pass A1 (accuracy) but fail A2
  (KL 0.6 to 0.8 x BF16, bar 0.5); Marlin MoE speed not BETTER (bs 8 +5.5% UNCLEAR, bs 256 WORSE).
  Results: `experiments/safedispatch_testA2/results/e38-rerun-20261027/` (summary.json, short.txt).
- Runpod: 0 thesis Pods, 0 volumes (deleted 07:31 UTC).
- Done: E-39 ran 2026-10-27 (about $5.20): INCONCLUSIVE; Marlin MoE +1% to 2.5% at batch 4 to 16, slower
  elsewhere. 0 Pods, 0 volumes. R-46 done. D-63 decided (user). E-40 (I-23 NVFP4, one B200) is prepared
  and reviewed (R-47); it waits for the user's spend approval (cap $25).
- Keys: `.env` has `RUNPOD_API_KEY` (read-only, default) and `RUNPOD_API_KEY_WRITE` (ask the user
  before each use). Rotate the write key after the experiments (it was pasted in chat).


### 1.1 Summary

- D-56 (user, 2026-10-22): run the GPU experiments on Runpod instead of local, and keep all details
  about the hardware of each run. The Runpod plugin and MCP server are set up and connected (0 Pods on
  2026-10-22). E-36 has Runpod steps (runbook Section 2a). E-30 needs root and power caps, so it stays on
  a provider with root. Hardware records: `experiments/runpod/`. No Pod was created and nothing was spent.
- E-36 amendment A2 (user: "maximum details and data with maximum savings"): resume, telemetry, kernel
  profiles, a microbenchmark and AUTO_STOP, plus an exploratory Phase B in the same Pod. Phase A and its
  rules do not change. R-39 (Sonnet) returned AMEND_THEN_RUN; all fixes are applied and tested on the
  Mac. Full session at most about $27 at $3.99/h (D-57, recommendation).
- D-58 (user, 2026-10-25): experiment standards in `docs/experiments/` (protocol with ten stage gates,
  measurement rules, Runpod operations and incidents, templates), an append-only run ledger and a
  noise-aware compare tool (`experiments/ledger/`, 7 tests pass), and a local `run-gpu-experiment`
  skill. Runpod access checked 2026-10-25: MCP connected, 0 Pods, SSH key `runpod_ed25519` registered,
  `.env` (gitignored) waits for the user's API key. E-36 now needs a cheap rehearsal Pod (stage S5)
  before the H100 run.
- D-59 (user, 2026-10-25): the Runpod account is shared with other projects and sessions (an H200
  Pod `okc-bf16-proj-tuning` from another session ran on 2026-10-25). Thesis resources are named
  `msthesis-<experiment>-<purpose>-<yyyymmdd>`, tagged `THESIS_TAG=msthesis`, and registered in
  `experiments/runpod/pods.jsonl`; thesis agents change only registered resources. The balance check
  now counts other running spend. `RUNPOD_API_KEY` is in `.env` and the live read works.
- E-36 rehearsal (stage S5, 2026-10-25, 1 x L4, about $0.48, Pod f6qrn1p79awqni, terminated). The user
  approved D-57 (full session) and D-51 at the H100 price. Mechanics passed: SSH start command, Pod
  environment, telemetry, profiler kernel tables, resume layout, microbenchmark, copy with checksums
  (55 files). Defects found and fixed: AUTO_STOP REST stop gets HTTP 403 with the Pod key (GraphQL
  podStop works, now used); `rsync -a` fails on the MooseFS volume (use `-rlt`); driver 595 names the
  SM clock `clocks.current.sm` (`extras.py` fixed). FINDING (FACT, L4, vLLM v0.32.0, Marlin reference):
  the first run of the session (cold torch.compile and Triton JIT) agreed with the later identical runs
  on only 34.8% of prompts; all later identical runs agreed 100%. A new config (cold) differed on 1 of
  256. So cold caches change greedy output. This would fail V1 on the H100. Amendment A3 (warm every
  config before its measured runs) is needed before the H100 run; it is not written yet.
- E-36 rehearsal 2e (2026-10-26, RTX 4090, about $0.40): A3 prediction PASS (README Section 12, R-41).
  cand_triton (a real kernel change) kept the same tokens on only 37 of 256 prompts (exploratory).
- E-36 rehearsal 2 (2026-10-26, about $0.50 over four Pods, all deleted): AUTO_STOP via GraphQL podStop
  works. A stopped Pod with a volume disk could not restart (host GPU taken) and the API refused a
  0-GPU resume, so results were lost. Lesson: use a network volume. One EU-RO-1 L4 host has broken
  Docker storage (image pull fails in a loop). `AUTO_STOP_DELAY` (default 600 s) added to session1.sh.
- No thesis direction is chosen (D-13 is open).
- The strongest candidate that is still alive is I-23 (SafeDispatch). It needs an NVIDIA GPU. There
  is no GPU yet (D-15), and the grant drafts in `grants/` are not submitted (D-19).
- Mac-only work from 2026-10-04 to 2026-10-07:
  - I-41 (a small "structurer" LLM in front of the agent model): killed. The user dropped the line
    (D-26).
  - R-09 (cross-domain directions from peer-reviewed papers, I-46 to I-57): all survivors were killed
    in R-10.
  - D-29 (user, 2026-10-07): use PagedAttention-style mechanism-transfer ideas only, not benchmarking
    papers. Use the rubber-duck method and a different-model (Sonnet) adversarial review for every
    experiment. D-27 (user): write all content in ASD-STE100.
  - R-10: I-58 (noise-shaped V-cache quantization) was the only strong route. E-27 tested it on the
    Mac after two pre-run reviews (R-11). Result: KILL, confirmed by R-12.
  - R-13 (route search 2): direct transfers are crowded. I-80 (metastability-aware admission for
    agent serving) was the only open candidate.
  - D-32 (user, 2026-10-07): review I-80, then a simulator premise check. Combinations of ideas are
    allowed. R-14 (Sonnet) killed I-80 as written and all four combinations except as WEAK.
  - E-28 (trace-driven simulator, 459 real agent sessions, 20,580 runs, reviewed before the run by
    R-15 and after it by R-16): the frozen rule gave K1 KILL. R-16 then found a lasting bimodal bad
    state after large spikes. Short client timeouts help recovery, so the retry story is false. A
    static session cap (Denning, CONCUR) clears the state. I-80 is KILL with scope limits.
- D-33 (user, 2026-10-08): run a third route search (R-17).
- R-17 (2026-10-08): frontier map plus three rounds (a Sonnet attack of six candidates, five
  main-session probes, a Sonnet generation pass over 28 candidates in training, collectives, numerics,
  agent infrastructure and cluster operations). Every Mac-testable candidate has a direct 2026 match.
  I-83 (rank-safe block-max pruning for the DSA lightning indexer) is the only open gap. It is WEAK and
  needs a GPU to dump real indexer keys. Write-up: `dossier/ROUTES_R-17_2026-10-08.md`.
- D-34 (recommendation, open): no fourth route search of this kind; put the I-83 key dump on the GPU
  plan next to I-23 Test A; push the grant applications now.
- D-35 (user, 2026-10-08, recorded second-hand from a parallel session): the user self-funds AWS GPUs
  (first round 1 x H100, about $20). Scope broadened to algorithmic GPU-efficiency ideas.
- R-18 (D-36, user): an orphan-finding scan of GitHub issues (15 repositories) and 30 characterization
  papers. No survivor (I-95 to I-102). Issues show bugs, not gaps; paper headline numbers can mislead.
  Write-up: `dossier/ORPHANS_R-18_2026-10-08.md`.
- D-37 (recommendation, open): stop mining literature for gaps; spend the first H100 hours on I-23
  Test A; start new ideas from a mechanism or algorithm with a clear invariant.
- R-19 (D-38, user's method, 2026-10-11): assumption removal. 40 assumptions, 7 checked, I-103 to I-109.
  Only I-106 (per-layer KV drift across adjacent RL checkpoints, version-aware serving) is open and
  Mac-testable; WEAK. D-39 (recommendation, open): run its premise check or go to I-23.
- R-20 (2026-10-16, the user's three seed topics: threads versus processes by load, KV cache poisoning,
  GPU noisy neighbor): three search rounds and a Sonnet attack, I-110 to I-119. All three topics are
  crowded. One survivor: I-118 (noisy neighbor through the shared CPU-GPU module power budget on GH200
  and GB200), MEASURE-FIRST, low confidence, needs 1 x GH200 for about 2 hours. I-116 (per-tenant I/O QoS
  for KV storage tiers) is WEAK. D-40 (recommendation, open). Write-up: `dossier/GAPS_R-20_2026-10-16.md`.
- R-21 (2026-10-17) surveyed the current systems frontier in two passes. It screened nine possible
  non-consensus truths against papers, production systems, RFCs, repositories, and current issues.
  Broad agent scheduling, agent recovery, sandbox prewarming, verification scheduling, and mid-decode
  checkpointing are already crowded. I-120 survives as atomic runtime activation for remote KV recovery.
  I-121 is KILL. I-23 remains the best first experiment, but its novelty risk is now high.
  Write-up: `dossier/FRONTIER_R-21_2026-10-17.md`.
- R-22 (2026-10-17) continued the same process until three total new gaps survived. It screened
  I-122 to I-130 in two passes. I-122 and I-123 initially survived. Seven candidates were killed.
  Write-up: `dossier/FRONTIER_R-22_2026-10-17.md`.
- R-23 is the required Sonnet pre-run review of E-31. It found four blockers. The abstract model
  encoded its result, lacked liveness, misrepresented source failures, and fixed its cost result.
  E-31 is INVALID DESIGN and did not run.
- R-24 audited current vLLM main at `da9134bc89364465f68aa679564435936d157a14` in Docker.
  Narrow fixes cover the closed source failures. One current reset failure remains open.
  I-122 is PIVOT because the global version mechanism lacks evidence.
- R-25 and R-26 reviewed E-32 before implementation. Both reviews found circular corpus encoding,
  incomplete oracles, and unfair decision rules. E-32 did not run.
- R-27 mapped the four upstream tests to action sequences and ran full-history blame.
  Only one trace fits depth six. No fault line came from an earlier narrow session fix.
  I-131 is KILL as a thesis. E-33 was cancelled before execution.
  Write-up: `dossier/PAPER_AUDIT_I-131_2026-10-19.md`.
- R-28 ran a fresh two-pass scoop audit for I-123. Current vLLM-Omni already owns the duplex
  session ledgers, reconnect journal, playback acknowledgements, and output fences. Its open PR 8485
  adds ordered per-input acknowledgements. Concordia already logs durable GPU-state epochs.
  The old periodic-snapshot oracle is circular for world models and lacks speech-state measurements.
  I-123 is PIVOT. I-132 is the narrower crash-consistent recovery-epoch question.
  Write-up: `dossier/SCOOP_AUDIT_I-123_2026-10-19.md`.
- R-29 is the required independent Sonnet review of E-34. It returned `KILL_IDEA`.
  The proposed policy is standard checkpoint composition. Its baselines are the two endpoints
  of the same checkpoint-interval curve. The deterministic workload removes the only possible
  new constraint. E-34 did not run. I-132 is KILL.
- R-30 (2026-10-19, main session) was a scoop check before the Mac test that the user asked for.
  CacheReforge (2609.30884, 2026-09-25) already stores per-layer version anchors in the KV cache,
  scores drift from adapter weights, and recomputes a bounded layer window. It covers LoRA updates
  only. A positive I-106 result would only extend it to full-weight RL updates, which is an
  evaluation result (D-29). I-106 is KILL. Nothing was downloaded and no experiment was predeclared.
  I-60 is KILL (CacheRoute 2608.19677, llm-d affinity break). I-59 stays WEAK and open. vLLM issue
  59907 reproduced a different priority inversion on the vLLM CPU backend, so a real-engine Mac
  test of I-59 is possible. D-47 (recommendation, open).
- E-35 (2026-10-19 to 2026-10-20, the user's choice under D-47) tested the I-59 premise on vLLM's
  real scheduler in Docker on the Mac. R-31 (Sonnet, pre-run) found that code reading decides the
  premise and replaced the planned grid with a trace check (amendment A1). Run 1 was INVALID: its
  positive-control scenario had no in-flight window. Amendment A2 fixed the arrival steps, and run 2
  gave KILL. R-32 (Sonnet, post-run) confirmed it with scope limits. vLLM has priority inversions
  (budget order, KV capacity, preemption), but none depends on a shared prefix, so inheritance has no
  dependency to fix. A paired sweep (600 scenarios) never found H later with a shared prefix. I-59 is
  KILL. D-48 (an optional CPU-backend fidelity run) and D-49 (next step) are open.
- The Mac-only mechanism ideas are now used up: I-106, I-60 (R-30) and I-59 (E-35) are KILL. I-116
  stays WEAK and needs a separate NVMe test.
- R-33 (2026-10-20, the user gave the R-21 prompt again): a fresh two-pass frontier review of eight
  transitions not in the registry (I-133 to I-140: wide-EP failures, agentic RL recovery, multi-GPU C/R
  with NCCL, GPU snapshots, cross-vendor KV, API rate limits, Mac RDMA meshes, RL reward verification).
  All eight have direct prior work and are KILL. The C/R line, which fits the user's background best, is
  now being built by NVIDIA (nvcf issue 2299, PR 2300). Pass 2 found vLLM PR 53298, which implements the
  I-120 mechanism: I-120 KILL as research. I-23 is the only survivor with a cheap decisive test; a new
  framing threat is 2606.07581. D-50 (recommendation, open). Write-up: `dossier/FRONTIER_R-33_2026-10-20.md`.
- R-36 (2026-10-20, the user's request): mined the stated limitations and future work of all 496 cited
  papers (372 recent papers state some). Eight follow-up gaps checked (I-141 to I-148): seven KILL, one WEAK
  (I-143, FP8 TP-invariant determinism, GPU only). The closest call, crash-time KV reuse (I-141), fails the
  simple-baseline attack. D-53 (recommendation, open). Write-up: `dossier/LIMITATIONS_R-36_2026-10-20.md`.
- R-37 (2026-10-21, D-54 decided by the user): an algorithmic-optimization sweep ("like competitive
  algorithmic problems") with the PyTorch Conference NA 2026 schedule as a source. It read SGLang, vLLM and
  PyTorch DCP hot paths at current main and screened I-149 to I-164. After the R-38 review (Sonnet), all
  16 are KILL: the engine already ships the better algorithm, an open upstream PR or 2026 preprint has it,
  or a theory check shows the decision does not matter for the real objective. R-38 closed the two last
  candidates: I-161 (SGLang eviction heap rebuild; E-37 measured 4.6 ms at 50,000 leaves) is already in
  SGLang PRs 38371 and 38454, and I-163 (critical-path scheduling for decode megakernels) is in Hazy's
  public scheduler, with about 10% measured by Cohere on a dense model. Section 8.1 rows 64 to 66 record
  three R-37 errors. Write-ups: `dossier/ALGORITHMIC_R-37_2026-10-21.md`,
  `dossier/REVIEW_R-37_SONNET_2026-10-21.md`. D-55 (recommendation, open).
- The current new-gap set is empty. I-23 remains the strongest candidate.
- Lesson from R-37 and R-38: for an engineering slow path, search the target repository's open PRs before
  the literature. Confirm which code path is the default before measuring.
- R-35 (2026-10-20, this session; D-49 decided by the user: "Do whatever you can to test locally to test
  or plan, and keep the ideas up-to-date, so that Once we have the GPUs, we will continue on those.")
  checked the GPU-gated ideas again and prepared the GPU work on the Mac. It found more evidence for the
  R-33 I-120 kill (vLLM issues 61768, 61905, 61912 and the maintainers' request-level recovery proposal,
  RFC 35780 and PR 42461) and a further weakness of I-83 (LiteTopK 2607.11976: indexer scores
  concentrate in a narrow range). It wrote two GPU packages with tests: E-36 (I-23 Test A Session 1, one
  H100, linked to the R-33 Test A design) and the frozen E-30 package (I-118, one GH200), plus
  `dossier/GPU_RUNBOOK_2026-10-20.md`. R-34 (Sonnet pre-run review) returned AMEND_THEN_RUN for both
  packages. Amendments A1 are applied, both unit-test suites pass (14 and 9 tests), and the E-36 Mac dry
  run on the vLLM v0.32.0 CPU image passed mechanically. Both packages wait only on the user (D-51, D-52).
- Lessons: in 2026, direct mechanism transfers into LLM infrastructure appear as preprints, issues or
  pull requests within weeks (R-10, R-13, R-17). Measure the premise on real data first. When a frozen
  rule says "no effect", check the edges of the grid (trigger size, horizon, metric power) before
  explaining the absence (row 46).
- `HANDOFF_2026-10-08.md` at the repo root describes the state before R-17. Its Section 4 (D-33) is now
  decided and done; this section wins.

### 1.2 Live ideas

| ID | Idea | Status | Blocker | Next step | Kill rule |
|---|---|---|---|---|---|
| I-23 | SafeDispatch, narrowed to Hopper and Blackwell block-scaled FP8, NVFP4 and MoE cells | MEASURE-FIRST | 1 NVIDIA GPU (Hopper, plus Blackwell for NVFP4) | E-36 Session 1 on one H100 (`dossier/GPU_RUNBOOK_2026-10-20.md`), then the full Test A (R-33 Section 7) | Oracle gain under 2% and unsafe fraction under 2%: KILL. Gain under 5% but unsafe fraction at least 10%: PIVOT to measurement thesis |
| I-18 | Compute-aware collective engine choice | MEASURE-FIRST, low priority | NVLink node (2+ GPUs, 8 for NVLS) | Fix the test design first (within-cell test, per-call static table baseline), then Test B | Per-call static table within 2% of oracle: KILL |
| I-33 | Feature-level divergence and speed ledger | HYPOTHESIS | GPU | Rides on the Test A matrix | None yet |
| I-34 | NCCL new-path numerics contract | HYPOTHESIS | 4 to 8 NVLink GPUs | Bundle with Test B in one node rental | None yet |
| I-19 | State-reuse soundness | KILL as thesis, salvage as a note | none (Mac) | Optional 1 to 2 week short note | n/a |
| I-83 | Rank-safe block-max pruning for the DSA lightning indexer | WEAK, GPU only, lower after R-35 (indexer scores concentrate) | A GPU node to dump real indexer keys (about 3 hours, 8 x H200) | Premise check in `dossier/ROUTES_R-17_2026-10-08.md` Section 6.3; freeze the kill rules in Section 6.1 before any run | Sign-correct bound skips under 50% of blocks at B = 16, L = 128K, K = 2048: KILL |
| I-94 | Phi-accrual detection of slow versus hung collectives (spare) | WEAK | No public collective-timing trace | None planned | None yet |
| I-118 | Power-coupled noisy neighbor on CPU-GPU superchips (GH200, GB200) | MEASURE-FIRST, low | 1 x GH200 with root and module power telemetry (ask the provider first) | E-30 package ready (`experiments/superchip_power_coupling/`, runbook Session 2); ask the provider about root and caps | Prefill throughput change under 3% at both module limits, or best static CPU cap within 5% of an offline oracle: KILL |
| I-116 | Per-tenant I/O QoS for KV storage tiers (mClock transfer) | WEAK | A box with one NVMe-backed KV tier and two clients | None planned | Victim fetch latency rises by under 20%: KILL |

### 1.3 Closed or deprioritized. Do not reopen without new information

I-06, I-07, I-08, I-10, I-13, I-14, I-15, I-16, I-17, I-19 (as thesis), I-20, I-22, I-24,
I-25, I-26, I-30, I-31, I-41 (as thesis mechanism), I-42, I-43 (dropped by the user, D-26),
I-46, I-47, I-48, I-49, I-50 to I-57 (R-09), I-58 (E-27, R-12), I-61 to I-75 (R-10), I-76 to I-79 and
I-81 (R-13), I-80 (R-14, E-28, R-16; an optional E-29 is listed in D-33), I-82, I-84 to I-93 (R-17),
I-110 to I-115, I-117, I-119 (R-20; I-114 has a WEAK dissent from the reviewer), I-121 (R-21),
I-124 to I-130 (R-22), I-122 (PIVOT after R-23 and R-24), I-123 (PIVOT after R-28),
I-131 (R-26 and R-27), I-132 (R-29), I-106 and I-60 (R-30), I-59 (E-35, R-31, R-32), I-120 (R-33),
I-133 to I-140 (R-33), I-141, I-142, I-144 to I-148 (R-36), I-149 to I-164 (R-37 and R-38).
Scanned: I-44,
I-45. Weak backups: I-59, I-60. Reasons are in each dossier.

### 1.4 Waiting on the user

1. **GPU access.** ANSWERED 2026-09-23: none yet, grant planned (D-15). Programs were
   researched on 2026-09-24 (Appendix B). Still open: which to apply to (D-18), and three
   questions for the advisor: can you be added to the lab's existing ACCESS or NAIRR
   allocations, does the lab or GenAI Lab own GPU servers, and will the advisor be the
   faculty applicant for the NVIDIA program. Also whether the lab compute in the 2026-05-20
   log (Vast AI, Lambda Tensorbook) still exists. Blocks I-23, I-120, I-18, I-33, I-34.
2. **The May options.** Are I-01 to I-05 dead? Did the discussions with Prof. Sengupta
   happen, and what came of them? The repo has no record between 2026-05-21 and 2026-09-15.
3. **Choose a direction** once the Test A gate reports, or earlier if you prefer.
4. **Venue and timeline.** Job start in January is assumed. Confirm target venue tier
   (workshop vs ASPLOS/EuroSys/MLSys class).
5. **Employer IP.** If the thesis touches checkpoint/restore work from the LinkedIn
   internship, confirm what may be reused. The NAIRR draft mentions that experience in one
   sentence.
6. **Award details from your labmate.** The PDF is their request, not the award. Ask which
   resource the award named (Horizon?), how many GPU-hours, the award dates, how they selected the
   resource on the form, and how long the decision took. That confirms the Blackwell route.
7. **Review, edit and submit the drafts in `grants/`** (D-19), starting with the advisor email.
8. **D-34 (after R-17).** Accept or change the recommendation: no fourth route search; I-83 key dump
   on the GPU plan; grants now. Alternatives: the E-28 workshop note (D-33 option d) or the I-19 note.
9. **D-41 (after R-21).** Accept or change the recommendation: run I-23 Test A first, then use
   remaining H100 time for the I-120 reproduction.
10. **D-45.** DECIDED 2026-10-19. The user said "go ahead" after the R-27 handoff.
    I-131 stays closed, and work moved to the I-123 audit.
11. **D-46.** The R-29 review met its kill condition. Accept or change the recommendation
    to keep I-132 closed.
12. **D-47.** DECIDED 2026-10-19. The user chose the I-59 premise check. Done as E-35 (KILL).
13. **D-48 (after E-35).** E-35 ran vLLM's scheduler class, not the full CPU backend that the user named.
    Accept that substitution, or ask for a one-cell CPU-backend fidelity run.
14. **D-49.** DECIDED 2026-10-20. The user asked for local tests and plans that keep the ideas current
    until GPUs arrive. Done as R-35 (packages E-36 and E-30, GPU runbook).
15. **D-50 (after R-33).** Accept or change the recommendation: no further broad frontier sweep; use the
    first H100 session for I-23 Test A; treat I-120 as closed.
16. **D-51 (ready: R-34 applied, Mac dry run passed).** Approve the E-36 Session 1 spend (one H100, 4-hour cap, $18 cap).
    Provider is now Runpod (D-56): H100 SXM $3.99/h on 2026-10-22. See D-57 for the A2 plan and its cost.
17a. **D-57 (after R-39).** Approve the E-36 A2 plan: the full session (at most 6 h 40 min, about $27) or Phase A
    only (about $11 to $14). Before booking, read the live Runpod balance (user rule, 2026-10-22: never store it).
17. **D-52 (ready: R-34 applied).** Approve the E-30 session (one GH200, 2.5-hour cap) after the provider confirms
    root and power caps.
18. **D-53 (after R-36).** Accept or change the recommendation: stop literature-based gap searches and
    spend effort on the prepared GPU runs.
19. **D-54.** DECIDED 2026-10-21. The user asked for an algorithmic-optimization sweep. Done as R-37, E-37
    and R-38 (all 16 candidates KILL).
20. **D-55 (after R-38).** Accept or change the recommendation: no further gap sweep of any kind; go to the
    GPU runs (D-51, D-52). Optional career work: the open SGLang lazy-heap PRs or their Rust-core version.

### 1.5 Next actions, in order

1. **D-51 and D-52 by the user.** Both GPU packages are ready: R-34 amendments applied, unit tests
   pass, E-36 Mac dry run passed. On D-51, book one Runpod H100 and follow `dossier/GPU_RUNBOOK_2026-10-20.md`
   Section 2a (D-56, E-36 A2), with the plan the user picks in D-57. Keep the hardware record of the run
   in `experiments/runpod/`. On D-52, first ask the provider about root, the module power section and writable Grace
   caps. D-50 and D-53 are open. Do not reopen I-59, I-60, I-106 or I-120.
1. **D-48 decision by the user.** Accept the scheduler-level substitution in E-35, or run the one-cell
   CPU-backend fidelity check (build the vLLM CPU backend for linux/arm64, compare its scheduler output
   stream with the fake-runner stream for one T2 scenario).
1. **Do not implement or run E-34.** R-29 killed I-132 at design review.
2. **D-41 decision by the user (R-21).** Use the first H100 session for I-23 Test A. Do not reproduce
   the I-120 failure: R-33 closed I-120. Do not implement either full system before its gate passes.
3. **D-40 decision by the user (R-20).** Accept the R-20 verdicts. If I-118 is wanted: ask the GPU
   provider about root, module power telemetry and cap settings, then predeclare E-30 with a rubber duck
   and a Sonnet pre-run review before any paid run.
4. **D-37 decision by the user.** D-39 (the I-106 Mac check) is superseded by R-30.
5. **D-37 decision by the user** (supersedes the open part of D-34). If accepted: plan the first H100
   session for I-23 Test A on the Mac first (cell list, scripts, time per cell, cost cap), with kill
   rules frozen and a Sonnet pre-run review. Confirm the D-35 wording.
6. Grant and compute: pick programs from Appendix B.3 and submit the drafts in `grants/` (D-18, D-19).
   Add the I-83 key dump (about 3 GPU-hours) to the request as a side task.
7. When compute arrives, run I-23 Test A (Section 5.8). Gate in writing at the end of week 3. If D-34 is
   accepted, predeclare the I-83 check (rubber duck, kill rules frozen, Sonnet pre-run review) and run
   the key dump in the same allocation.
8. Optional: write the I-19 short note from existing results.
9. Before committing to any direction, do a preprint scoop check for it.
10. Fill the June to September gap in Section 4 with what the user remembers.

### 1.6 Recent changes

- 2026-10-22: E-36 amendment A2 and R-39 (Sonnet pre-run review, AMEND_THEN_RUN, fixes applied). D-57.
- 2026-10-22: D-56 (user): GPU experiments run on Runpod. Runpod plugin 1.8.0 and MCP server connected.
  New `experiments/runpod/` (hardware record script, hardware log). Runbook Section 2a (E-36 on Runpod).

- 2026-10-21: R-37 algorithmic-optimization sweep (D-54). I-149 to I-164 registered. E-37 screen run. R-38
  (Sonnet) review changed I-161 and I-163 to KILL, so all 16 are KILL. Section 8.1 rows 64 to 66. D-55.

- 2026-10-20: R-34 recorded (Sonnet pre-run review of E-36 and E-30: AMEND_THEN_RUN). A1 applied to
  both packages. The E-36 Mac dry run passed mechanically; it found five defects, all fixed. D-51 and
  D-52 are ready for the user.

- 2026-10-20: R-36 limitations mining. I-141 to I-148: seven KILL, one WEAK (I-143). D-53.

- 2026-10-20: R-35 (this session, D-49): GPU-plan freshness check and local preparation. E-36 package
  (I-23, one H100), E-30 frozen package (I-118, one GH200), GPU runbook. R-34 pending. D-51, D-52.

- 2026-10-20: R-33 fresh frontier pass. I-133 to I-140 KILL. I-120 KILL as research (vLLM PR 53298).
  I-23 MEASURE-FIRST with a new framing threat (2606.07581). D-50.

- 2026-10-20: E-35 done. R-31 (pre-run) replaced the grid with a trace check. Run 1 INVALID, run 2
  KILL, R-32 (post-run) CONFIRM_WITH_SCOPE_LIMITS, plus a paired supplement. I-59 KILL. D-47 decided,
  D-48 and D-49 open. Section 8.1 rows 59 and 60.

- 2026-10-19: R-30 scoop check before the Mac test. CacheReforge (2609.30884) covers the I-106
  mechanism for LoRA updates. I-106 KILL, I-60 KILL, I-59 WEAK and Mac-testable on the vLLM CPU
  backend. Nothing ran and nothing was downloaded. D-47.

- 2026-10-19: R-29 returned `KILL_IDEA` for I-132. The design is standard checkpoint
  composition. Its deterministic Docker workload cannot test the possible GPU replay constraint.
  E-34 is KILL-DESIGN and did not run.

- 2026-10-19: R-28 rejected the old I-123 CPU oracle. I-123 is PIVOT. I-132 narrows the question
  to one durable epoch across model state, input, output, and playback progress. E-34 is design only.

- 2026-10-19: R-26 rejected the amended E-32 design. R-27 then ran the paper audit.
  I-131 is KILL as a thesis. E-32 and E-33 did not run.

- 2026-10-17: R-23 found four blocking faults in E-31. E-31 is INVALID DESIGN and did not run.
  R-24 audited current vLLM in Docker. I-122 is PIVOT. Added I-131 and predeclared E-32.

- 2026-10-17: Claude authentication succeeded in the user's Terminal. The automation process cannot
  read the Keychain entry. Added a read-only Sonnet review prompt and manual wrapper. E-31 did not run.

- 2026-10-17: Docker Desktop now serves Linux containers through the `desktop-linux` context.
  The server has 10 virtual CPUs and 7.65 GiB of memory. Sonnet remains signed out. E-31 did not run.

- 2026-10-17: E-31 predeclared for I-122. The Docker harness, six policies, four source scenarios,
  bounded search, scale sweep, tests, and frozen decision script are ready. No code ran. Docker
  Desktop has an invalid signature. Sonnet is signed out. D-44 records the user's run decision.

- 2026-10-17: R-22 continued R-21 until three total new gaps survived. I-122 and I-123 survive.
  I-124 to I-130 are KILL. The total set is I-120, I-122, and I-123. D-42 is done. D-43 is open.

- 2026-10-17: R-21 current-frontier survey. Nine beliefs screened. I-120 survives as MEASURE-FIRST
  with high scoop risk. I-121 is KILL. I-23 remains the best first experiment. D-41 is open.

- 2026-10-16: R-20 gap search on the user's three seed topics. I-110 to I-119. I-118 MEASURE-FIRST (low),
  I-116 WEAK, others KILL. E-30 drafted, not run. D-40.

- 2026-10-11: R-19 assumption removal (D-38). I-103 to I-109. I-106 WEAK (Mac premise check proposed),
  I-108 WEAK, others KILL. D-39.

- 2026-10-08 (later session): R-18 orphan-finding scan (D-36). I-95 to I-102, all KILL except a WEAK
  keep-alive policy (I-98). D-35 copied from memory notes, D-37 recommendation.

- 2026-10-08 (later session): R-17 route search 3 (D-33 option b). Three rounds, two Sonnet passes,
  I-82 to I-94 registered. I-83 WEAK (GPU only), I-94 WEAK spare, all others KILL. D-34.

- 2026-10-07 to 2026-10-08: D-32 (user: option 1). R-14 killed I-80 as written. E-28 ran after two
  pre-run reviews (R-15) and gave K1 KILL; R-16 confirmed it for the grid, found a lasting bad state
  after large spikes, and corrected the explanation (row 46). I-80 KILL with scope limits. D-33.

- 2026-10-07: route search 2 (R-13, the user's choice in D-31). Five of six candidates were taken by
  2025 and 2026 work. I-80 (metastability-aware admission for agent serving) is open. D-32.

- 2026-10-07: E-27 ran (12:40 to 17:10). Two Sonnet pre-run reviews (R-11) changed the plan
  (amendments A1, A2). Result: I-58 KILL, confirmed with scope limits by a Sonnet post-run review
  (R-12). Two explanations in the rubber duck were corrected (Section 8.1 row 44). D-31.

- 2026-10-07: R-10. The user set the idea style and process (D-29). A Sonnet reviewer attacked the
  routes. I-58 (noise-shaped V cache, per-head gating) is the only strong route. I-46, I-47, I-48
  and I-49 were killed. Two corrections: the E-23 flip invariant is a mathematical identity, and
  the Delta GPU error data is public. Dossier: `dossier/ROUTES_R-10_2026-10-07.md`. D-30.

- 2026-10-06: R-09 found 4 new directions from peer-reviewed papers by cross-domain transfer:
  I-46, I-47, I-48 (MEASURE-FIRST, all Mac-testable) and I-49 (HOLD, no public data). Eight other
  candidates were killed or folded. Dossier: `dossier/NEW_DIRECTIONS_R-09_2026-10-06.md`. D-28.
- 2026-10-06: all content is now written in ASD-STE100 (D-27).

- 2026-10-04: I-41 ingest structurer (the user's idea). Literature pass, E-24 trace study of the
  user's Claude Code transcripts, E-25 local 0.6B screen, R-08 review, dossier
  `dossier/DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md`. I-41 KILL as a thesis mechanism, I-42
  KILL, I-43 DEPRIORITIZED, I-44 and I-45 SCANNED (D-23, awaiting the user). Section 8.1 row 41:
  TraceLab already released public coding-agent traces, which weakens I-28's trace-gap angle.
- 2026-10-05: user dropped the I-41 line (D-26): I-41 KILL accepted, I-43 dropped with no note.
- 2026-10-04: E-26, the I-43 check. TraceLab's public release has no tool text, so the Exgentic
  SWE-bench traces were used (528 MB, user approved, D-24). M = 43.7% of tool-dependent edits copy
  from outside the last 3 tool outputs, so I-43 survives as a short-note candidate (D-25). Claude
  Opus 4.5 alone is 29.1%.

- 2026-09-30: correctness review of the GPU-efficiency review (R-07). Two subagent reviewers, disputed
  items re-read in the primary PDFs, 13 groups of fixes applied to the report and recorded in
  Section 8.1 rows 30 to 40. Main changes: Fishman FP8 divergence is at about 200B tokens (not 2T),
  the DeepSeek-V3 "<0.25%" comparison is from 16B and 230B models, MegaScale overlap is about 35% of
  its gain (not "largely"), the MFU ceiling claim was wrong (up to 59.1%), C3 downgraded to
  WORKING HYPOTHESIS. No idea registered, no status changed. D-22.
- 2026-09-30: evidence-graded review of GPU efficiency for training and inference at the user's
  request, `dossier/LIT_REVIEW_GPU_EFFICIENCY_2026-09-30.md`, a standalone review in the same format as the
  agentic-harness review (companion Claude Doc linked inside). No idea registered, no status changed.
  Findings bear on I-23 (FP8 accumulation error, FP8 instabilities after about 200B tokens,
  batch-dependent outputs) and, only modestly, I-18 (overlap is one source of scale gains, not the
  main one in MegaScale's ablation), Section 7.9. Wording corrected by R-07.
- 2026-09-30: correctness review of the agentic-harness review (R-06). fixes to labels,
  numbers and sources recorded in Section 8.1 rows 25 to 29. No idea registered, no status changed.
- 2026-09-30: literature review on agentic harnesses at the user's request,
  `dossier/LIT_REVIEW_AGENTIC_HARNESSES_2026-09-30.md`. No idea registered, no status changed.
  Nearest thesis link is I-28 (harness traffic as a serving workload), Section 7.8.
- 2026-09-24: Mac pilot E-23 complete. Harness core validated, stratified estimator failed
  its predeclared test, Test A default is now simple random sampling with a Wilson upper bound.
- 2026-09-24: compute programs researched, Appendix B. NAIRR Start-Up is the best-documented fit.
- 2026-09-24: added Section 1.7, the plan while waiting for compute, and D-17 (pending).
- 2026-09-23: user confirmed no GPU yet and plans a compute grant (D-15). Pointer to this
  journal added to `agents.md` (D-16).
- 2026-09-23: R-05 adversarial review with literature pass, NCCL re-audit (E-22), this journal.
- 2026-09-22: new handoff document introduced I-23 and the kernel-correctness direction.
- 2026-09-18: `RESEARCH_STATUS.md` recommended I-19.

### 1.7 Plan while waiting for compute (proposal, 2026-09-24)

Status: proposal. Not yet accepted by the user. Log acceptance in Section 10.

**Compute tiers to request.** Hours are deliberately not stated yet. They should come from
the cell count and per-cell time measured by the Mac pilot, plus a safety margin, not from
a guess.

| Tier | Hardware | Serves | Why this and not less |
|---|---|---|---|
| 1 (required) | 1 Hopper-class GPU (H100 or H200), about 1 to 2 weeks of single-GPU access (R-05 estimate) | I-23 Test A: INT8, block-scaled FP8, BF16, MoE | Integer Alibi already covers a consumer sm_89 GPU. Hopper is what makes the result new (Section 5.8, open cells) |
| 2 (adds NVFP4) | 1 Blackwell GPU, a shorter block | I-23 NVFP4 cells, vLLM #55238 and #55649 reproductions | NVFP4 needs Blackwell. A consumer sm120 part may be cheaper, but FlashInfer #2723 reported broken NVFP4 MoE backends there (search-only), so verify before asking. Likelier route: TACC Horizon GB200 through NAIRR, see Appendix B |
| 3 (optional) | 2 or more NVLink GPUs, 8-GPU NVSwitch node for NVLS, short block | I-18 Test B and I-34 | Copy-engine collectives are NVLink-domain only |

Ask for Tier 1 as the core request. Tiers 2 and 3 are add-ons that do not block Test A.

**Where to apply.** Programs, eligibility, dates and materials are in Appendix B. Headline
findings (2026-09-24): ACCESS Explore lets a graduate student be PI with an advisor letter and
reaches 8-GPU NVLink H200 nodes on NCSA Delta. NAIRR Start-Up accepts graduate students with
an advisor letter, decides within 3 weeks, and offers up to 2,000 GPU-hours on H100-class
systems for 3 months. NVIDIA's "AI Inference, Agents, and Systems Software"
track opens 2026-10-01 but the advisor must apply. The SJSU College of Engineering cluster
has H100 nodes but is restricted to engineering faculty and students, and your program is in
the College of Science. TACC Horizon has Blackwell (GB200) nodes and may be requestable through NAIRR,
as a friend's award suggests, but that is not confirmed on the official pages I could read.
Drafts of all the applications are in `grants/`.

**What a grant application can reuse from this journal.**
- Research question and the narrowed question: Section 5.8.
- Why compute is unavoidable: CUTLASS, Triton, FlashInfer and NVFP4 kernels do not run on
  Apple silicon, and the open cells are Hopper and Blackwell.
- Predeclared GO, PIVOT and KILL thresholds (heuristics): Test A in Section 5.8. A funder
  can see the request is tied to a decision, not open-ended.
- What is delivered in each outcome: GO gives a dispatch system, PIVOT gives a measurement
  study on cells no one has filled plus an open-source harness, KILL gives a documented
  negative result. All three are deliverables.
- Preliminary evidence: the killed-idea record (Sections 5 and 6) shows the falsification
  discipline, and the Mac pilot below shows the harness works.

**Mac-side pilot (no GPU needed). Store in `experiments/safedispatch_oracle/`. STATUS: done
2026-09-24, see E-23. Steps 1 to 4 worked. Pass criterion 5 (the margin-stratified estimate) failed
its test, and the Test A default became simple random sampling.**
1. Run-manifest schema and kernel-pair abstraction, using the fields in Section 5.8.
2. Contract metrics: L1 bitwise comparison, L2 margin-stratified greedy-flip estimate, L3
   KL on logits.
3. Positive controls by fault injection on CPU: emulate an INT8 GEMM whose scale-application
   and output-rounding epilogue differs by one bf16 spacing (the fault class of 2609.00363)
   and confirm the harness flags it while an allclose-style check does not.
4. End-to-end run on a small model (Qwen3-0.6B class, or the Qwen3.5-0.8B already used) between
   two available backends, for example CPU fp32 reference vs MPS fp16.
5. Pilot pass criteria (heuristics, set before running): zero false positives comparing a
   kernel with itself across repeated runs, every injected fault detected, and the
   margin-stratified flip estimate inside the confidence interval of a brute-force
   generation comparison on the same small model.

The pilot is a harness validation, not a research result. Do not cite its numbers as evidence
about GPU kernels. It sizes the grant request and removes setup risk from the scarce GPU time.

**Schedule risk.** The 2026-09-22 handoff assumes October to December as the research window
and says the core claim must survive or fail before the January job. Test A takes 1 to 2
weeks, then claim lock at week 5, then 7 weeks of build. If a grant decision lands after
about mid-November, that plan no longer fits before January.

**Decision needed from the user (D-13 area), by about 2026-10-15:** choose one.
- (a) A short paid bridge rental for Test A only, cost to be checked by the user (I have not
  looked up current prices).
- (b) Make the October deliverable the Mac pilot plus the I-19 note (Section 5.6), and run
  Test A after the grant, accepting that the GPU phase moves past January.
- (c) Ask the lab or advisor about the compute in the 2026-05-20 log as a bridge.
- (d) Use ACCESS Explore as the bridge. If it approves within days as third-party sources
  report, it gives Hopper (H200) time without a paid rental. It does not give Blackwell.

**Fallback.** If neither a bridge nor the grant materializes, the only Mac-only route is the
I-19 note, and the realistic venue is a workshop (see Section 7 of the R-05 write-up for the
reasoning).

---

## 2. Context and constraints

### 2.1 Person and program

- Kushagra Bainsla, MSCS, San Jose State University.
- Lab in the May 2026 log: MICoSys Lab (Sengupta Lab), advisor Prof. Sengupta. The Sep 16
  handoff describes "SJSU research: reliable distributed ML infrastructure, resource-aware
  scheduling, fault tolerance". Whether the advisor relationship is still current is
  unrecorded.
- Verified 2026-09-24 from SJSU pages: the MICoSys Lab (Machine Intelligence and Complex
  Systems) is in the SJSU Department of Computer Science and led by Saptarshi Sengupta, who
  is also listed as GenAI Lab director. The MSCS program is in the College of Science (the
  College of Science's Department of Computer Science). The lab's site lists an NSF ACCESS
  allocation and a NAIRR allocation for two other projects. This matters for compute
  eligibility (Appendix B).
- Coursework: RL course in fall 2026 (from the May log).
- Expects a full-time job in January (2027). The 2026-09-22 handoff treats October to
  December as the window to produce the core result.

### 2.2 Background that shapes the choice of ideas

- **LinkedIn AI Infrastructure internship, summer 2026.** Built and validated a transparent
  CRIU-based checkpoint/restore platform for multi-GPU, cross-node training on
  Kubernetes. Preserves CPU process state and live GPU state. Reported recovery 71.1 s vs
  186.5 s for framework checkpointing (about 62% lower). Estimated about 300,000 GPU-hours
  per year of recomputation across three fabrics. Source: `AI_Infra_Thesis_Master_Handoff_Dossier.docx`
  and `agents.md`.
- **Multi-agent orchestration project "north".** Hot-reloadable agent registries,
  priority-aware inference routing with tiered model pools, append-only ledger for
  per-agent recovery, trust layers that auto-resolve low-stakes actions and hard-gate
  irreversible ones.
- Event-driven distributed systems: queue-decoupled ingestion, jittered backoff.
- Languages and tools: Python, Go, C++, Rust, TypeScript, SQL, Bash. CUDA, cuda-checkpoint,
  NCCL, PyTorch, Kubernetes, Docker, Kafka, gRPC.

### 2.3 Goals

- A thesis or project that is technically deep, publishable at a strong systems or
  ML-systems venue, useful as an open-source artifact, and relevant to frontier AI
  infrastructure roles.
- Target roles named in the 2026-09-22 handoff: Anthropic Performance Engineer (GPU) and
  Performance Engineer (Inference Systems). Gaps to fill with thesis evidence: serious
  CUDA/Triton kernel work, Nsight profiling, roofline reasoning, FP8/FP4 numerics,
  vLLM/SGLang/FlashInfer internals, numerical regression diagnosis.
- Prefers hard infrastructure problems over application ideas and wants aggressive
  adversarial review before committing.

### 2.4 Hard constraints

| Constraint | Detail |
|---|---|
| Time | About three months of full-time research, October to December 2026 |
| Hardware | Apple M4 MacBook Air, 16 GiB unified memory. No NVIDIA GPU of any kind as of 2026-09-23 (confirmed by the user). A compute grant is planned, timing unknown |
| Avoid | Theses whose critical path is specialized physical hardware, human subjects, or custom data collection (2026-09-22 handoff). Note this conflicts with any path needing a GPU node |
| Prefer | Automated GPU sweeps that run unattended and produce reproducible artifacts. A narrow falsifiable claim over a large framework |

### 2.5 Working process

- Cheap falsification on the Mac first: source inspection, real-model correctness tests,
  analytical and oracle studies, CPU and MPS microbenchmarks. Defer CUDA, NCCL and HBM
  claims to NVIDIA hardware.
- Experiments and harnesses run locally and are uploaded as result bundles.
- Two adversarial passes per candidate, kill criteria before building.
- From 2026-10-07 (D-29, decided by the user):
  - Prefer ideas that take a proven algorithm from another field and run it in a new
    environment, with small changes, for a visible gain (the PagedAttention pattern). Do not
    make a benchmarking or measurement paper the contribution. Use measurement only as a check
    (oracle studies, kill tests).
  - Follow the rubber-duck method for every experiment. Before the run, explain each step, the
    expected result and what can make it wrong. After the run, explain the results step by step.
  - Have a different model review each experiment adversarially (code, data and conclusions).
    Fix what it finds, then record the review as an R-nn entry.

### 2.6 Local environment (recorded by E-17)

macOS 26.5.2 arm64, Python 3.12.13, torch 2.14.0, transformers 5.17.0, MPS, float16,
eager attention. `causal_conv1d` and `flash-linear-attention` are absent, so recurrent
kernels run correct but slow reference code. This inflates the recurrent share of runtime
and makes every measured full-attention share a conservative underestimate.

External dependencies live outside the repo, in `~/Downloads`. See Section 9.

---

## 3. Idea registry

Every idea ever proposed, screened, or scanned. Newest verdict in the Status column. Full
history in Section 5.

### 3.1 Phase 0: domain selection (May 2026)

| ID | Idea | Origin | Status | Aliases | Section |
|---|---|---|---|---|---|
| I-01 | ML-driven detection of silent data corruption in LLM training | 2026-05-20 | DORMANT | Option 1 | 5.1 |
| I-02 | GNN prediction of cascading failures in GPU clusters | 2026-05-20 | DORMANT | Option 2 | 5.1 |
| I-03 | Adversarial robustness of AI-infrastructure monitoring | 2026-05-20 | DORMANT | Option 3 | 5.1 |
| I-04 | ML-driven checkpoint policies (LinkedIn aligned) | 2026-05-20 | DORMANT | Option 4 | 5.1 |
| I-05 | RL failure-aware inference scheduling | 2026-05-20 | DORMANT | Option 5 | 5.1 |

### 3.2 Phase 1: systems pivot and first screens (by 2026-09-16)

| ID | Idea | Origin | Status | Aliases | Section |
|---|---|---|---|---|---|
| I-06 | Live migration of many active hybrid LLM sessions | undated, file 2026-09-16 | KILL (2026-09-23) | "Idea 1" in the stateful-elastic dossier | 5.2 |
| I-07 | Activation residency: keep, recompute or offload | V2 screen | DEPRIORITIZED | | 5.3 |
| I-08 | Network-adaptive overlap and granularity | V2 screen | DEPRIORITIZED | | 5.3 |
| I-09 | Opportunistic heterogeneous elastic training | V2 screen | HOLD (backup) | | 5.3 |
| I-10 | GPU-sharing interference scheduling | V2 screen | KILL as standalone | | 5.3 |
| I-11 | Typed hybrid-state allocator | V2 screen | DEPRIORITIZED (led to I-13) | | 5.3 |
| I-12 | Power-aware parallelism switching | V1 screen | HOLD, untested | | 5.3 |

### 3.3 Phase 2: hybrid-state line (2026-09-16)

| ID | Idea | Origin | Status | Aliases | Section |
|---|---|---|---|---|---|
| I-13 | Typed and heterogeneous model-state runtime | vLLM gap harness | KILL as broad thesis | | 5.4 |
| I-14 | REQUIRED / RECONSTRUCTABLE / EPHEMERAL state contract | E-08 | Mechanism only, folded | | 5.4 |
| I-15 | Minimum-cost consistent-state planner | E-09 | KILL (2026-09-17) | RQ3 | 5.4 |
| I-16 | KV-aware selective recovery (reuse full-attention KV while rebuilding recurrent state) | handoff | KILL (2026-09-17) | | 5.4 |
| I-17 | Activation-assisted recovery (cache post-block activations) | E-11 | KILL (2026-09-17) | | 5.4 |

### 3.4 Phase 3: iteration 4 survivors (2026-09-16 to 2026-09-18)

| ID | Idea | Origin | Status | Aliases | Section |
|---|---|---|---|---|---|
| I-18 | Compute-aware collective execution-engine choice (SM vs copy engine) | handoff parallel track | MEASURE-FIRST, low priority | RQ2, "Idea 2" | 5.5 |
| I-19 | Soundness of per-layer-group state reuse | dossier iteration 4 | KILL as thesis (2026-09-23) | RQ1, "Idea 1" in `ideas/` | 5.6 |
| I-20 | Layer-layout co-design to unlock divergent reuse | dossier 4c | KILL (2026-09-18) | "Idea 3", RQ3b | 5.7 |
| I-21 | Elastic reconfiguration equivalence | dossier decision table | HOLD, low confidence | | 5.7 |
| I-22 | LLM queueing-boundary idea | `RESEARCH_STATUS.md` | KILL, thin trail | | 5.7 |

### 3.5 Phase 4: 2026-09-22 handoff

| ID | Idea | Status | Section |
|---|---|---|---|
| I-23 | SafeDispatch / KernelGuard: correctness-aware kernel and backend qualification and selection | MEASURE-FIRST, narrowed (2026-09-23) | 5.8 |
| I-24 | Generic cross-layer LLM profiler | KILL | 5.9 |
| I-25 | Generic inference performance simulator | KILL | 5.9 |
| I-26 | LLM agent that recommends serving configurations | KILL / PIVOT | 5.9 |
| I-27 | LLM-generated GPU kernels | PIVOT | 5.9 |
| I-28 | Agent-aware GPU scheduling | PIVOT | 5.9 |
| I-29 | Generic fault-tolerant inference | PIVOT | 5.9 |
| I-30 | Generic deterministic inference | KILL | 5.9 |
| I-31 | Selective high-precision stabilization as a generic thesis | KILL | 5.9 |
| I-32 | Robotics or weapons-capability evaluation | DEPRIORITIZED, off-career | 5.9 |

### 3.6 Phase 5: 2026-09-23 hypotheses and scans

| ID | Idea | Status | Section |
|---|---|---|---|
| I-33 | Feature-level divergence and speed ledger for cached multi-turn workloads on MoE and hybrid models | HYPOTHESIS, low confidence | 5.10 |
| I-34 | Reduction-order and precision contract of NCCL's newer collective paths | HYPOTHESIS, low confidence | 5.10 |
| I-35 | Multi-GPU process-level checkpoint/restore for inference engines | SCANNED, crowded, NVIDIA in progress | 5.11 |
| I-36 | Runtime cache-integrity verification | SCANNED, crowded | 5.11 |
| I-37 | Fault-tolerant LLM inference | SCANNED, crowded | 5.11 |
| I-38 | RL training-inference mismatch and kernel alignment | SCANNED, crowded | 5.11 |
| I-39 | Hybrid-model determinism and recurrent-state precision | SCANNED, upstream building | 5.11 |
| I-40 | Tensor-parallel determinism and communication-fusion cost | SCANNED, upstream building | 5.11 |

### 3.7 Phase 6: 2026-10-04 ingest structurer and context caching

| ID | Idea | Status | Section |
|---|---|---|---|
| I-41 | Ingest structurer: a small LLM compresses prompts and tool outputs into terse structured form before the main model, so context stays append-only, compaction comes later and the prompt cache lives longer | KILL as thesis mechanism (2026-10-04, R-08), accepted by the user 2026-10-05 (D-26) | 5.12 |
| I-42 | Cache-aligned context editing: run clearing or compaction on requests whose cache is already cold, or batch it | KILL, publicly proposed (claude-code #94177) | 5.12 |
| I-43 | Reference-aware retention: keep tool-output spans the agent will later copy verbatim, compress or clear the rest | DEPRIORITIZED, dropped by the user 2026-10-05 (D-26) after E-26 passed | 5.12 |
| I-44 | Edit-tolerant KV reuse in serving engines, so a middle-span edit does not force suffix re-prefill | SCANNED, crowded | 5.12 |
| I-45 | TTL-resilient KV retention for agent idle gaps (tiered KV storage) | SCANNED, crowded, folded into I-28 | 5.12 |

### 3.8 Phase 7: 2026-10-06 cross-domain directions (R-09)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-46 | Sampled simulation of LLM serving traces with cache warm-up (from SimPoint and SMARTS) | KILL (2026-10-07, R-10): a benchmarking method (D-29) | 5.13 |
| I-47 | Closed-loop validity of agent serving evaluation (from open vs closed queueing models) | KILL (2026-10-07, R-10): a benchmarking method (D-29) | 5.13 |
| I-48 | Margin-gated quality control for production inference numerics (from CUSUM control charts) | KILL (2026-10-07, R-10): DiFR, Log Probability Tracking; heuristic, not a mechanism | 5.13 |
| I-49 | Failure-clustering-aware recovery provisioning for training (from Hawkes and ETAS aftershock models) | KILL as thesis route (2026-10-07, R-10): Delta data is public, MLSys 2026 sparing paper | 5.13 |
| I-50 | KV retention across tool and human gaps with ski rental and predictions | KILL, crowded | 5.13 |
| I-51 | Speculative tool execution (from branch prediction) | KILL, crowded | 5.13 |
| I-52 | Reproducible summation for batch-invariant kernels | KILL, crowded, needs GPU | 5.13 |
| I-53 | Agent sandbox fork for tree search (from VM fork and CRIU) | KILL, crowded | 5.13 |
| I-54 | Group testing to find faulty GPU nodes | KILL, crowded and shipping | 5.13 |
| I-55 | Hedged requests with prefix-cache affinity | KILL, first principles | 5.13 |
| I-56 | Certified token stability from rounding-error bounds | DEPRIORITIZED, folded into I-48 | 5.13 |
| I-57 | Fused tool calls (interpreter superinstructions) | KILL, crowded | 5.13 |

### 3.9 Phase 8: 2026-10-07 mechanism-transfer routes (R-10, different-model review)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-58 | Noise-shaped (sigma-delta) V-cache quantization along the token axis, enabled per head | KILL (2026-10-07, E-27, R-12) | 5.14 |
| I-59 | Priority inheritance for shared-prefix producers in LLM schedulers | KILL (E-35, R-31, R-32: no shared-prefix dependency in vLLM 3897f95c or SGLang 476fe86a) | 5.14 |
| I-60 | Late binding with a locality timeout for continuous-batching routers (Sparrow, delay scheduling) | KILL (R-30: CacheRoute 2608.19677, llm-d affinity break) | 5.14 |
| I-61 | Denning working-set load control against KV thrashing | KILL, crowded | 5.14 |
| I-62 | Bazel-style content-addressed cache of agent tool results | KILL, crowded | 5.14 |
| I-63 | Log-structured KV cache on SSD or high-bandwidth flash | KILL, crowded | 5.14 |
| I-64 | Singleflight merging of duplicate prefill | KILL, shipping | 5.14 |
| I-65 | Flowlet switching for agent session routing | KILL, crowded | 5.14 |
| I-66 | Miss-ratio-curve partitioning of prefix caches | KILL, crowded | 5.14 |
| I-67 | rsync-style sparse BF16 weight sync for RL | KILL, crowded and shipping | 5.14 |
| I-68 | Erasure-coded in-memory checkpoints | KILL, peer-reviewed prior art | 5.14 |
| I-69 | Lossless BF16 compression of collectives | KILL, crowded | 5.14 |
| I-70 | Chandy-Lamport snapshots for async RL pipelines | KILL, crowded | 5.14 |
| I-71 | Regression test selection for agent tests and RL rewards | KILL, crowded | 5.14 |
| I-72 | Smith predictor for autoscaling with model-load dead time | KILL, crowded | 5.14 |
| I-73 | Subtractive dither for the KV cache | KILL, published | 5.14 |
| I-74 | Fencing tokens and idempotency for agent retries | KILL, crowded | 5.14 |
| I-75 | PowerDial dynamic knobs for LLM serving | KILL, crowded | 5.14 |

### 3.10 Phase 9: 2026-10-07 route search 2 (R-13)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-76 | MemGuard-style memory-bandwidth regulation for on-device LLM decode on unified memory | KILL, published | 5.15 |
| I-77 | KV hibernation to SSD for local agent sessions | KILL, crowded | 5.15 |
| I-78 | Thermal-aware decode pacing on fanless laptops | KILL, crowded | 5.15 |
| I-79 | Slack stealing inside decode steps for co-running CPU work | KILL, crowded | 5.15 |
| I-80 | Metastability-aware admission for online agent serving (retries plus prefix-cache eviction) | KILL (R-14; E-28 and R-16, with scope limits) | 5.15 |
| I-81 | Copy-aware speculative decoding for agent edits | KILL, crowded | 5.15 |

### 3.11 Phase 10: 2026-10-08 route search 3 (R-17)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-82 | Flash write filter for the KV cache SSD or HBF tier (Flashield, CacheSack) | KILL, published and shipping | 5.16 |
| I-83 | Rank-safe block-max pruning for the DSA lightning indexer (Block-Max WAND) | WEAK, GPU only, premise check proposed; lower after R-35 (LiteTopK score concentration) | 5.16 |
| I-84 | Scale-aware ABFT for MXFP8 and NVFP4 GEMMs | KILL, reduces to the BF16 case | 5.16 |
| I-85 | Mixed-criticality scheduling under output-length uncertainty (Vestal) | KILL, empty guarantee | 5.16 |
| I-86 | Multi-branch speculative frames for interactive world models (Outatime) | KILL, cost | 5.16 |
| I-87 | Hazard-rate (Gittins) KV retention during agent tool calls | KILL, crowded | 5.16 |
| I-88 | Mode-switched imprecise computation for reasoning requests | KILL, crowded (LASER) | 5.16 |
| I-89 | Expert-load-aware request batching for EP decode | KILL, crowded | 5.16 |
| I-90 | SDC verification in pipeline bubbles | KILL, crowded (SpareTrain) | 5.16 |
| I-91 | KV surgery when reasoning is stripped from earlier turns | KILL, same as I-44 | 5.16 |
| I-92 | Cache pre-warm triggered by user typing | KILL, engineering | 5.16 |
| I-93 | Checkpoint-aware choice of preemption victims in GPU training queues | KILL, Volcano PR | 5.16 |
| I-94 | Phi-accrual detection of slow versus hung collectives | WEAK spare, no public data | 5.16 |

### 3.12 Phase 11: 2026-10-08 orphan-finding scan (R-18)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-95 | Recurrent-state snapshot placement for hybrid-model prefix reuse under agent edits (llama.cpp #22746) | KILL, same as I-15; the measured waste is an implementation defect | 5.17 |
| I-96 | Runtime checks for silent numeric bugs in parallelism composition (Megatron #7452) | KILL as route, context for I-21 and I-23 | 5.17 |
| I-97 | Lower agent-harness overhead outside command execution (TraceLab T01) | KILL, not waste (p50 0.13 s), shipped | 5.17 |
| I-98 | Overcommit, ballooning and snapshot keep-alive for agent sandbox fleets | KILL (mechanism, DSec); keep-alive policy WEAK | 5.17 |
| I-99 | Process-level checkpoint/restore for inference-engine cold start | KILL, NVIDIA Dynamo 2609.25451 and shipped products | 5.17 |
| I-100 | Faster training restart (peer restore, broadcast load, bad-node history) | KILL, shipped | 5.17 |
| I-101 | Harvest idle SMs of online inference for best-effort work | KILL, crowded | 5.17 |
| I-102 | Byte-budgeted decode batches against long-context interference | KILL, STAR, Medha, PLA-Serve | 5.17 |

### 3.13 Phase 12: 2026-10-11 assumption removal (R-19)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-103 | Verify AI-generated kernels (assumption: kernels are written by trusted experts) | KILL as new idea, folds into I-23 as motivation | 5.18 |
| I-104 | Training under grid demand response (assumption: full power at any time) | KILL (Emerald 2507.00909 compares DVFS, pause, reallocation) | 5.18 |
| I-105 | Agent-as-tenant cluster scheduler (assumption: users are humans with few jobs) | KILL (real logs show few duplicates; ALPS, 2603.15916) | 5.18 |
| I-106 | Version-aware serving: per-layer KV drift across adjacent RL checkpoints (assumption: model version fixed for a session) | KILL (R-30: CacheReforge 2609.30884 covers the mechanism for LoRA updates) | 5.18 |
| I-107 | Machine-consumer decode mode (assumption: a human reads the stream) | KILL (Human-Less LLM Serving 2606.20577, jump-forward decoding) | 5.18 |
| I-108 | Isolation vs reuse vs offload under confidential GPUs (assumption: trusted operator) | WEAK, needs a CC GPU (Azure or GCP) | 5.18 |
| I-109 | KV policy from server-side session state (assumption: stateless server) | KILL, same as I-45 and I-50 | 5.18 |

### 3.14 Phase 13: 2026-10-16 seed-topic gap search (R-20)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-110 | Load-adaptive GPU sharing mode (isolated processes at low load, shared context at high load) | KILL (dLoRA, Aegaeon, D-STACK, LithOS, green contexts, 2605.26461) | 5.19 |
| I-111 | Process versus thread for the Python serving front end (free-threaded 3.14t) | KILL, engineering | 5.19 |
| I-112 | Cross-tenant prefix-cache pollution | KILL (PrefixShield 2608.01657) | 5.19 |
| I-113 | Integrity of KV in shared external KV stores (LMCache, Mooncake) | KILL as research (HMAC, checksums) | 5.19 |
| I-114 | Sampled-recompute verifier for position-independent KV reuse (against HijackKV) | KILL by simple baseline; reviewer: WEAK | 5.19 |
| I-115 | Eviction-steering attack on KV compression | KILL (pinning, 2605.18053) | 5.19 |
| I-116 | Per-tenant I/O QoS for KV storage tiers (mClock transfer) | WEAK, premise not measured | 5.19 |
| I-117 | GPU noisy neighbor via MPS, MIG, PCIe | KILL (same as I-10, newer work) | 5.19 |
| I-118 | Power-coupled noisy neighbor on CPU-GPU superchips | MEASURE-FIRST, low | 5.19 |
| I-119 | Contention as a timing-side-channel defense | KILL (2609.06853, KVGov) | 5.19 |

### 3.15 Phase 14: 2026-10-17 current-frontier review (R-21)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-120 | Atomic runtime activation for remote KV recovery | KILL as research (R-33: vLLM PR 53298 implements the mechanism) | 5.20 |
| I-121 | Verification-aware scheduling for coding-agent fleets | KILL (TestGRAD, PatchDiff, CI test selection) | 5.20 |

### 3.16 Phase 15: 2026-10-17 continued frontier review (R-22)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-122 | Versioned commit for mutable real-time sessions | PIVOT to I-131; E-31 INVALID DESIGN | 5.21 |
| I-123 | Commit-indexed recovery for heterogeneous real-time session state | PIVOT after R-28; old CPU oracle rejected | 5.21 |
| I-124 | Authoritative commit frontier for speculative and async inference | KILL (LLM-42, ONNX Runtime GenAI, transactional KV caching) | 5.21 |
| I-125 | Elastic slot batching for masked diffusion LLMs | KILL (vLLM-Omni `StepScheduler`, direct paper) | 5.21 |
| I-126 | Cancellation-debt scheduling for agent branch races | KILL (NIXL leases, SGLang abort protocol, TAPER) | 5.21 |
| I-127 | Online isolation for generated GPU kernels | KILL (M2K, KREX, CUDA process isolation) | 5.21 |
| I-128 | Topology-independent data-loader recovery | KILL (Mosaic Streaming elastic determinism) | 5.21 |
| I-129 | Adaptive recurrent-state checkpoint placement | KILL (Sparse Prefix Caching, replay systems, vLLM RFC 52959) | 5.21 |
| I-130 | Compiler and CUDA-graph warm state in training checkpoints | KILL (PyTorch Mega-Cache, Foundry) | 5.21 |

### 3.17 Phase 16: 2026-10-17 mutable-session test pivot (R-23, R-24)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-131 | Stateful metamorphic testing for mutable inference sessions | KILL as thesis; optional tool-only salvage | 5.22 |

### 3.18 Phase 17: 2026-10-19 crash-recovery cut pivot (R-28, R-29)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-132 | Crash-consistent duplex recovery epoch | KILL at design review | 5.23 |

### 3.19 Phase 18: 2026-10-20 fresh frontier pass (R-33)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-133 | Elastic membership for wide expert-parallel serving | KILL (EEP 2605.10670, SGLang and vLLM elastic EP, Ray Serve) | 5.24 |
| I-134 | Context-aligned environment recovery for agentic RL | KILL (Belayer 2608.14635, RobustRL, DSec) | 5.24 |
| I-135 | Multi-GPU transparent C/R with NCCL P2P, IPC and NVLS on | KILL (NVIDIA nvcf issue 2299 and PR 2300, Singularity, CRIUgpu, GCR) | 5.24 |
| I-136 | Semantics-aware GPU snapshots (skip weights and free KV) | KILL (Dynamo Snapshot, GPU Memory Service, Dynamo issue 16357) | 5.24 |
| I-137 | Cross-vendor KV and numerics portability | KILL (2606.29986) | 5.24 |
| I-138 | Rate-limit-aware scheduling for agent fleets on external APIs | KILL as research (no invariant, provider accounting) | 5.24 |
| I-139 | Local tensor parallelism over Thunderbolt 5 RDMA Mac meshes | KILL (feasibility; EXO, MLX JACCL) | 5.24 |
| I-140 | Reward verification as the agentic RL straggler | KILL as mechanism (GKE Agent Sandbox, DSec, async RL) | 5.24 |

### 3.20 Phase 19: 2026-10-20 limitations mining (R-36)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-141 | Reuse in-flight KV after a serving-process crash when device memory survives | KILL (baseline: committed-length journal; Dynamo capacity argument; AnchorTP) | 5.25 |
| I-142 | Shapley-style split of a shared prefix's cost among tenants | KILL (public repo srujan20/prefix-cost-attribution; DLPM; 2609.24991) | 5.25 |
| I-143 | TP-size-invariant deterministic kernels for block-scaled FP8 | WEAK (extension of TBIK, needs 2+ GPUs) | 5.25 |
| I-144 | Aging-aware rejuvenation for LLM serving engines | KILL (baseline: periodic restart with fast shadow engines) | 5.25 |
| I-145 | Branch-scoped KV so aborted agent branches leave no residue | KILL (engineering; ForkKV, EAGLE-Pangu) | 5.25 |
| I-146 | Integrity protection for high-fan-out shared KV blocks | KILL as thesis (HBM ECC, narrow threat model) | 5.25 |
| I-147 | Erasure-coded KV for decode-heavy and P/D serving | KILL (GhostServe extension, needs cluster) | 5.25 |
| I-148 | KV tier redesign when host memory is not larger than HBM (Grace) | KILL (NVIDIA ICMS, needs GB200 or GB300) | 5.25 |

### 3.21 Phase 20: 2026-10-21 algorithmic-optimization sweep (R-37)

| ID | Idea | Status | Section |
|---|---|---|---|
| I-149 | Incremental cache-aware ordering of the waiting queue (SGLang drops LPM above 128 waiting requests) | KILL (PEEK 2607.02525, Requests of a Feather 2605.06046) | 5.26 |
| I-150 | O(new tokens) front-end work for each agent turn (tokenization, template, block hashing) | KILL (vLLM RFC 47582, TokTier 2607.29678, Incremental BPE 2605.30813) | 5.26 |
| I-151 | Optimal CUDA-graph capture sizes by one-dimensional DP | KILL (small oracle: vLLM step-8 sizes; parametrized dynamic-shape graphs) | 5.26 |
| I-152 | Rank balancing for DP attention in DP plus EP decode | KILL (TensorRT-LLM ADP balance, NanoCP 2605.21100) | 5.26 |
| I-153 | Exact per-step replica choice for redundant experts | KILL (shipped: SGLang `lp` dispatch; LLEP) | 5.26 |
| I-154 | Johnson's-rule ordering for staged pipelines (PP, EPD, P/D) | KILL (proportionate flow shop theorem; online serving optimizes latency, not makespan; reason corrected by R-38) | 5.26 |
| I-155 | Weighted tree pebbling for offline prefix-shared batches | KILL (SGLang DFS theorem, BatchLLM, BlendServe) | 5.26 |
| I-156 | Minimal-movement resharding on elastic resize | KILL (Tenplex; small absolute time) | 5.26 |
| I-157 | Flow-optimal source choice for RL weight sync and MLA KV pulls | KILL (vLLM NIXL spreads MLA reads; small oracle) | 5.26 |
| I-158 | NVLS multicast for MoE dispatch on NVL72 | KILL (static group multicast raises ingress 4.5 times; dynamic addressing is DySHARP, 2026-05; R-38) | 5.26 |
| I-159 | Incremental counts for repetition penalties | KILL (fixed in the vLLM GPU runner) | 5.26 |
| I-160 | DCP planning complexity (read overlap, dedup order) | KILL (sweep line ships; dedup gain small) | 5.26 |
| I-161 | Kept eviction heap for radix caches (SGLang rebuilds a heap of all leaves on each `evict()`) | KILL (R-38: SGLang PRs 38371 and 38454 already add lazy heaps; R-37 had it as an engineering PR candidate) | 5.26 |
| I-162 | Work placement for Muon and other matrix optimizers | KILL (NorMuon, MatrixFSDP 2607.05895, FlexShard) | 5.26 |
| I-163 | Critical-path list scheduling for decode megakernels (the per-step task graph repeats) | KILL (R-38: Hazy Megakernels list scheduler, Cohere about 10% on dense then dropped for MoE; R-37 had WEAK) | 5.26 |
| I-164 | Scan-resistant eviction for agent sessions that return in cyclic order | KILL (2606.21238, Continuum) | 5.26 |

---

## 4. Timeline (append-only)

File times are modification times from the filesystem, not authoring times. The real run
dates of Phase 1 and Phase 2 experiments are not recorded. The 2026-09-16 handoff packages
them as of that date.

### Phase 0: domain selection

```
2026-05-19  [P0]  Domain chosen: machine learning for reliable and efficient AI
                  infrastructure. Reasons: fits lab strengths (time series, adversarial
                  robustness, graph ML), large growing field, matches career path through
                  LinkedIn AI Infrastructure. Source: README.md (git, Entry 1).
2026-05-20  [P0]  Reframed from reliability alone to reliability plus efficiency (Entry 2).
                  Gathered industry numbers (Entry 3): Meta Llama 3, 16,384 H100s over 54
                  days, 419 unexpected failures (one per 3 hours). OPT-175B, 992 A100s over
                  90 days, 112 failures. GPT-4 training 78 to 100+ million USD. IDC AI
                  infrastructure spend 334 billion USD in 2025. Conclusion: field is real
                  and academically open.
2026-05-20  [P0]  Compiled 12 to 13 paper reading list (Entry 4) and broad gaps (Entry 5):
                  SDC detection is primitive, cascading failure prediction underexplored,
                  inference scheduling ignores failures, monitoring itself can be attacked,
                  checkpoint policies are mostly static. Ideas: I-01 to I-05.
2026-05-20  [P0]  Learned the summer LinkedIn project: container checkpoint/restore for
                  distributed training using CRIU, three use cases (preemption, hardware
                  failure recovery at about 5% of nodes, node maintenance), possible paper
                  with the original author of CRIU (Entry 6). Idea: I-04. Added
                  checkpointing literature (Entry 7): Gemini, CheckFreq, JIT
                  checkpointing, LowDiff, convergence-aware checkpointing.
2026-05-20  [P0]  Commits 7262ed3 (14:19) and 800b2b7 (15:12), "Init: Added intial
                  documentation". Only README.md.
2026-05-21  [P0]  Commit 4bd50d5 (14:40) rewrote most of README.md (178 insertions, 175
                  deletions). Content was not diffed here. The committed README's next steps:
                  commit to one project by end of June, check in weekly with Prof.
                  Sengupta, reproduce anchor results on lab compute (Vast AI, Lambda
                  Tensorbook). This is the last git commit in the repo.
```

### Gap

```
2026-05-21 to 2026-09-15  [gap]  No records in the repo. Known from later documents: the
                  LinkedIn internship built and validated CRIU-based checkpoint/restore.
                  The direction moved from ML-policy options (I-01 to I-05) to
                  systems-level GPU and LLM-inference research. When and why is
                  unrecorded. I-01 to I-05 were never formally closed. Ask the user.
```

### Phase 1: live migration and GPU-efficiency screens

```
undated (file 2026-09-16 17:51)  [P1]  stateful_elastic_inference_thesis_dossier.docx.
                  "Status after three validation iterations." Idea I-06 rated "GO,
                  narrowed scope". Experiments E-01 to E-03 on the Mac (Falcon-H1-0.5B,
                  MPS). Text says "Idea 1", implying earlier multi-idea ideation that is
                  not in the repo. Next gate: fresh novelty scan plus a 2+ GPU POC. Neither
                  is ever recorded as done. I-06 is not mentioned in any later document.
2026-09-16  [P1]  Handoff package (files 19:20). Its evidence ledger gives this order:
                  V1 synthetic GPU-efficiency suite (E-04), later withdrawn because it was
                  mostly hard-coded and the user challenged validity. V2 and V2.1 real Mac
                  screens (E-05). V2.2 corrected suite (E-06). Ideas I-07 to I-12.
2026-09-16  [P2]  Hybrid-state line, per the same ledger: vLLM gap harness (E-07) kills the
                  broad typed-state runtime (I-13). Reconstructability POC (E-08, I-14).
                  Min-cost planner (E-09, I-15). Qwen3.5 Mac cached replay (E-10). State-
                  selective recovery v1.1 (E-11) found INVALID. Adversarial self-review
                  (R-01) retracted premature verdicts. Collective-engine idea I-18 seeded
                  as an UNTESTED parallel track.
2026-09-16 19:26  [P2]  agents.md created.
2026-09-16 20:00  [P3]  research-gap-dossier skill installed under .claude/skills. Its
                  interface file lists chatgpt, codex, api and atlas, so it was authored
                  for OpenAI-side tools. Provenance note only.
```

### Phase 3: iteration 4 falsification

```
2026-09-16 evening  [P3]  Iteration 4 runs (file times: v1_3 env symlink 20:16, rq1 22:57,
                  rq3 23:00, NCCL audit result 23:06). E-12 v1.2 fix valid. E-13 order bias
                  found. E-14 FLOP ceiling. E-15 weak Pareto. E-16 cross-model state
                  constants. E-17 randomized sweep. Result: I-16 killed on performance.
2026-09-16  [P3]  Novelty pass 2 (R-02) located vLLM #53041 as the exact prior claim of
                  the mechanism and #52773 as load-backfill that beats recompute by 80x.
2026-09-16 to 09-17  [P3]  Mac-testable gates run (R-03): E-18 RQ1 gate PASS, E-19 RQ3
                  KILL (I-15 killed), E-20 RQ2 source audit confirms structural gap.
2026-09-17  [P3]  Layout survey (E-21) of 13 real hybrid models kills I-20 and produces
                  the suffix-skip finding. Dossier last edited 11:01. ideas/ folder written
                  11:08 to 11:09 with I-19 (recommended) and I-18 (keep warm).
2026-09-18 15:42  [P3]  RQ1 gate and layout survey results rewritten. 15:46
                  RESEARCH_STATUS.md written: proceed with I-19 unless NVLink hardware
                  appears. Also records the LLM queueing-boundary idea (I-22) as closed.
```

### Phase 4: kernel-correctness direction

```
2026-09-22 13:18  [P4]  LLM_Inference_Thesis_Research_Handoff_2026-09-22.docx (file time).
                  New primary direction I-23 SafeDispatch, MEASURE-FIRST. Adds target-role
                  analysis, the January job constraint, a candidate table (I-24 to I-32)
                  with adversarial verdicts, hypotheses H1 to H4, a 12-week plan and pivots.
                  Does not mention I-06, I-18 or I-19, and does not state hardware.
                  This is R-04 (its own adversarial pass).
```

### Phase 5: adversarial review and this journal

```
2026-09-23  [P5]  User asked for an adversarial review with a real literature pass. Ran the
                  research-gap-dossier workflow (R-05): about 70 sources, NCCL audit re-run
                  on 2.32.3 (E-22), cited GitHub issues opened and compared with the docs'
                  claims, survey quote verified in extracted PDF text. Findings: no GO. The
                  Sep 22 handoff missed about a dozen relevant papers and upstream PRs. I-19's
                  motivating bugs are a different class than its checker. I-18's planned
                  test cannot separate a live signal from a static table. I-06 killed.
                  Output: dossier/ADVERSARIAL_REVIEW_2026-09-23.md.
2026-09-23  [P5]  Handoff package checksums verified: all 12 entries OK, files unmodified
                  since 2026-09-16.
2026-09-23  [P5]  This journal created from all repo documents, the git history, the result
                  files, and R-05.
2026-09-23  [P5]  User answers: (1) yes, add a pointer to the journal in agents.md, done.
                  (2) No GPU access yet. The user plans to file a grant for compute.
                  Program, deadline and decision date not yet known. Recorded as D-15 and
                  D-16. Effect: every GPU-dependent idea (I-23, I-18, I-33, I-34) is blocked
                  until compute arrives, and grant lead time threatens the Oct to Dec window.
2026-09-24  [P5]  Added Section 1.7: compute tiers to request, what a grant application can
                  reuse from this journal, a Mac-side pilot for the Test A harness, the
                  schedule risk, and a decision needed by about 2026-10-15 (D-17, pending).
2026-09-24  [P5]  User asked for grant research and the Mac pilot in parallel. Grant survey
                  done, recorded in Appendix B: ACCESS Explore and NAIRR Start-Up accept
                  graduate students with an advisor letter, NVIDIA's inference-systems track
                  opens 2026-10-01 with the advisor as applicant, the SJSU engineering
                  cluster is engineering-only, and the advisor's lab site lists existing
                  ACCESS and NAIRR allocations. The official NAIRR Start-Up page gives the best-documented
                  fit: 3-month award, up to 2,000 GPU-hours, decision within 3 weeks. D-18 pending.
2026-09-24  [P5]  Mac pilot harness built in experiments/safedispatch_oracle/ and
                  E-23 predeclared before any run. Attempt 1 (MPS fp16): C1, C2, C3 pass, C4
                  inconclusive (8 flips). Attempt 2 (CPU bf16, predeclared fallback): C1, C3
                  pass, C4 FAIL, the margin-stratified estimator covered the truth in 5.8% of
                  replicates. Diagnosis: structural zero-flip strata add phantom flips.
                  Amendment A1 tested on development data (0.94 real, 0.80 synthetic), not
                  reliably valid. Flip invariant holds with 0 violations. Test A default set
                  to simple random sampling with a Wilson upper bound. Attempt 3 (held-out)
                  not run, reasons in E-23.
2026-09-24  [P5]  User reported that UT Austin (TACC) gives Blackwell chips with about 1,000 GPU-hours
                  and that a friend obtained it with a NAIRR proposal, and shared the PDF. The PDF
                  could not be read (Read blocked outside the working directory). TACC's docs
                  confirm Horizon has 2,000 GB200 nodes with 2 Blackwell GPUs each and list NAIRR as
                  an allocation route. The NAIRR Start-Up page does not list Horizon, so the route
                  is user-reported and unconfirmed. Drafted the applications in grants/: NAIRR
                  Start-Up proposal (3-page PDF), ACCESS Explore request, advisor email and letter,
                  NVIDIA outline. D-19 pending.
2026-09-24  [P5]  The user copied the labmate's NAIRR proposal into the repo and it was read. It is
                  the request (200 GPU-hours, $20,000 API credits, 200 GB) by the same advisor, not
                  the award, and says nothing about Blackwell. NAIRR's official instructions fixed
                  the required format: sections A to D, at most 3 pages, Times New Roman 11pt or
                  larger, estimates based on prior work or benchmarks. The proposal draft was
                  restructured to match. Appendix B.6.
2026-09-24  [P5]  Cleanup, authorized by the user. Superseded files moved to archive/ with a
                  README (ideas/ briefs, RESEARCH_STATUS.md). The labmate's PDF moved to
                  grants/reference/. A stray root results/ folder turned out to be the E-22 audit
                  output and was filed with the NCCL audit. run_mac_selective.py was kept because
                  the v1.3 bundle script copies it. D-20.
2026-09-30  [P5]  Literature review on agentic harnesses (user request): 55 sources graded
                  checked, search-only or prior-knowledge, 12 claims labelled ESTABLISHED,
                  WORKING HYPOTHESIS or SPECULATIVE. Main findings: harness choice moves cost
                  robustly (up to 40x tokens) and accuracy conditionally (0 to 28 points); no
                  universal multi-agent advantage; benchmark scores are confounded. One widely
                  repeated cross-harness number could not be traced and is marked do-not-cite
                  in the report. Source: dossier/LIT_REVIEW_AGENTIC_HARNESSES_2026-09-30.md.
                  Ideas: I-28 (context only). Experiments: none.
2026-09-30  [P5]  Correctness review of the agentic-harness literature review (user request), R-06.
                  Two independent subagent reviewers, each told the most serious verified errors
                  earn five points. Reviewer A found 15 verified issues, reviewer B 13. The user's
                  disputed items were re-checked against primary pages: METR RCT repos average
                  "22k+" stars (report said 23,000), Kim et al. matched token budgets (report said
                  no compute-matched frontier comparison exists), and one reviewer claim was wrong
                  (TOPAS is a real paper, 2608.25523). Fixes applied to the report, retractions in
                  Section 8.1 rows 25 to 29. No idea registered, no status changed. D-21.
2026-09-30  [P5]  Review of GPU efficiency for training and inference (user request): 54 sources,
                  16 claims labelled consensus (9), working hypothesis (6) or speculative (1).
                  Main findings: tuned dense training MFU at 1K to 16K GPUs has stayed at about 40
                  to 55% since 2022; decode is memory-bound; failures (1,024-GPU MTTF 7.9 h) and
                  stragglers (10.4% of GPU-hours at ByteDance) cost as much as kernel gains; FP8
                  training needs specific recipes and can fail after about 2T tokens; FP4 rests
                  on one vendor run; serving speedups are self-reported against moving baselines.
                  IEA and LBNL reports returned HTTP 403 and are not used. Source:
                  dossier/LIT_REVIEW_GPU_EFFICIENCY_2026-09-30.md, live doc https://claude.ai/code/artifact/4b94def3-488a-453a-a2fe-331981c8781e.
                  Ideas: I-23, I-18 (context only). Experiments: none.
                  [Several numbers in this entry were wrong, see the R-07 entry below and
                  Section 8.1 rows 30 to 40. Original text kept per rule 0.2.1.]
2026-09-30  [P5]  Correctness review of the GPU-efficiency literature review (user request), R-07.
                  Two independent subagent reviewers, each told the most serious verified errors
                  earn five points. Reviewer A listed 5 serious and 6 moderate issues, reviewer B 6
                  serious and 6 moderate. The main session re-read the primary PDFs for the disputed
                  items and confirmed them: Fishman et al. FP8 diverges after about 200B tokens
                  (the report said 2T, which is the size of the run that succeeded), Samsi et al.
                  3 to 4 J per token is from V100 (report said A100), the DeepSeek-V3 "<0.25%"
                  loss gap is from 16B and 230B models, MegaScale reports 59.1% MFU at 3,072 GPUs
                  and overlap is 6.2 of 17.6 ablation points, Thinking Machines timing is from
                  Qwen-3-8B, MT-NLG is 30.2% with attention. Two reviewer items were not acted on
                  (Patterson scope, C13 label). Raw count favored B (6 vs 5), after regrading both
                  had 4 serious findings that held up, so points were left for the user to award.
                  Fixes applied to the report, retractions in Section 8.1 rows 30 to 40. No idea
                  registered, no status changed. D-22.
```

### Phase 6: ingest structurer and context caching

```
2026-10-04  [P6]  User pasted notes proposing a small structurer LLM that rewrites prompts and
                  tool outputs into terse structured facts before the main model, to delay
                  compaction and keep the prompt cache alive. Logged as I-41. Literature pass:
                  ACON, TokenPilot, AgentDiet, Squeez, CAPC, Don't Break the Cache, Keepalive
                  Economics, PatchKV and others (Section 7.10). Ideas: I-41.
2026-10-04  [P6]  E-24 and E-25 predeclared in Section 6.1 before any analysis code ran.
2026-10-04  [P6]  E-24 run on the user's local Claude Code transcripts (10,431 de-duplicated
                  requests, $1,925 at list price). P1 FALSE (compaction 0.39% of break writes,
                  TTL expiry 83.4%), P2 oracle 23.8% at r = 4, P3 net 14.6%, P4 TRUE (71.5% of
                  edits copy tool-output text verbatim). First run double-counted forked sessions;
                  fixed and re-run. Experiments: E-24.
2026-10-04  [P6]  E-25 run: Qwen3-0.6B abstractive structurer kept the later-edited span in 1 of
                  30 Read results, a first-and-last-lines rule in 6 of 30. Experiments: E-25.
2026-10-04  [P6]  Second search pass found TraceLab (2606.30560) and claude-code issue #94177,
                  which report the same cache-miss pattern at larger scale. R-08: I-41 KILL as a
                  thesis mechanism, I-42 KILL, I-43 DEPRIORITIZED, I-44 and I-45 SCANNED.
                  Section 8.1 row 41 qualifies the 2026-09-30 harness review's trace-gap claim.
                  Source: dossier/DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md. D-23.
2026-10-04  [P6]  Section 1 rewritten "as of 2026-10-04". The previous version ("as of
                  2026-09-24") said the same about I-23, I-18, GPU access and grants; the rewrite
                  only adds the I-41 paragraph, the closed-list entries, item 8 in 1.4 and the
                  1.6 line.
2026-10-04  [P6]  User approved downloading TraceLab to run the I-43 check. Not downloaded:
                  the raw README and the sanitized example show the public release drops
                  `tools[].input` and stores no tool output text (only `input_chars`,
                  `result_chars`), so verbatim reuse cannot be measured on it. Release v0.0.2
                  (2026-07-24) is 101 MB jsonl.gz or 160 MB DuckDB. Alternative with full message
                  content: Exgentic `DiscoPosse/agent-llm-traces` (1,781 traces incl. 391 SWE-bench,
                  claude_code harness among 5), 984 MB parquet. Asked the user before downloading.
                  Ideas: I-43.
2026-10-04  [P6]  User chose the SWE-bench part only. Shard index read with HTTP range requests
                  (about 5 MB), shards 11 to 20 downloaded (528 MB) to ~/Downloads/agent-llm-traces.
                  E-26 predeclared, decoders fixed after a smoke test, then run. M = 43.7% pooled,
                  verdict SURVIVES; Claude Opus 4.5 alone 29.1%. I-43 moves to MEASURE-FIRST as a
                  short note only. D-24, D-25. Experiments: E-26. Ideas: I-43.
2026-10-05  [P6]  The journal update for E-26 was lost when the 2026-10-04 session ended mid-write
                  and was re-applied from the saved `results/e26_results.json`. No numbers changed.
2026-10-05  [P6]  User dropped the whole line: I-41 KILL accepted, I-43 dropped (D-26). Downloaded
                  data stays in ~/Downloads/agent-llm-traces until the user removes it.
2026-10-06  [P6]  User decided that all content is written in ASD-STE100 (D-27).
```

### Phase 7: cross-domain directions

```
2026-10-06  [P7]  User asked for 4 to 5 new directions from peer-reviewed papers only, found from
                  first principles, with a solution taken from a different field, and testable.
                  R-09 screened 12 candidates (I-46 to I-57) in two search passes. Survivors:
                  I-46, I-47, I-48 (MEASURE-FIRST) and I-49 (HOLD, no public node-level failure
                  data). Eight candidates were killed or folded. Preprints were not used as
                  evidence. They are listed as scoop signals. Source:
                  dossier/NEW_DIRECTIONS_R-09_2026-10-06.md. D-28. Ideas: I-46 to I-57.
2026-10-07  [P7]  User set the idea style and the experiment process (D-29): PagedAttention-style
                  mechanism transfer, no benchmarking papers, rubber-duck method and a
                  different-model adversarial review for every experiment.
2026-10-07  [P7]  The main session screened about 25 more mechanism transfers. Most were already
                  published or shipping in 2026 (I-61 to I-75). A Sonnet reviewer attacked the
                  shortlist (R-10). The main session re-ran the reviewer's simulation and checked
                  its key citations. Result: I-58 (noise-shaped V cache) is the only strong route,
                  narrowed to per-head gating. I-46, I-47, I-48 and I-49 were killed. I-59 and I-60
                  are weak backups. Two corrections were logged (Section 8.1 rows 42 and 43).
                  Source: dossier/ROUTES_R-10_2026-10-07.md. D-30.
2026-10-07  [P7]  Section 1 rewritten "as of 2026-10-07". The previous version ("as of 2026-10-04")
                  summarized three older recommendations (I-19, I-23, R-05), the I-41 side
                  investigation, and the hardware update. The new version adds D-27, D-29, the
                  I-58 kill, R-13 and D-32, and points to HANDOFF_2026-10-07.md.
2026-10-07  [P7]  User approved E-27 and asked for a review of the test before the run. E-27 was
                  predeclared. Sonnet reviewed the plan and code twice (R-11): 3 blockers and 7 major
                  items, all fixed (amendments A1 and A2), then no blocker. E-27 ran from 12:40 to
                  17:10 on the Mac CPU. The frozen decision script gave KILL (rule a). Sonnet's
                  post-run review (R-12) confirmed the KILL with scope limits and corrected two
                  explanations (Section 8.1 row 44). I-58 KILL. D-31. Experiments: E-27. Ideas: I-58.
2026-10-07  [P7]  Route search 2 (R-13) gave I-80. The user chose D-32 option 1 and allowed
                  combinations of ideas. Sonnet reviewed I-80 and four combinations (R-14): I-80 as
                  written KILL in a toy simulator (retries not needed; a session cap fixes it).
                  E-28 was redesigned as a trace-driven replication. Ideas: I-80.
2026-10-07  [P7]  E-28 predeclared. Sonnet pre-run reviews (R-15, two passes) found 5 blockers in
                  total, all fixed (amendments A1 and A2). Experiments: E-28.
2026-10-08  [P7]  E-28 ran (about 1 h 35 min, 20,580 runs). Frozen decision: K1 KILL (no collapse
                  point). Sonnet post-run review (R-16) confirmed K1 for the grid, but its own check
                  found a lasting bimodal bad state after spikes of 2 x N* or more, and corrected the
                  main session's explanation (Section 8.1 row 46). I-80 KILL with scope limits. D-33.
                  Experiments: E-28. Ideas: I-80.
2026-10-08  [P7]  Section 1 rewritten "as of 2026-10-08". The previous version ("as of 2026-10-07")
                  listed I-80 as an unreviewed HYPOTHESIS and D-32 as open. The new version records
                  D-32, R-14, E-28, R-15, R-16, the I-80 kill and D-33, and points to
                  HANDOFF_2026-10-08.md.
2026-10-08  [P7]  D-33 decided by the user: (b), route search 3 (R-17). Frontier map of the 2025 and
                  2026 systems programs, then three rounds: a Sonnet attack of six candidates, five
                  main-session probes, and a Sonnet generation pass over 28 candidates. Every
                  Mac-testable candidate has a direct 2026 match. I-83 (rank-safe DSA indexer
                  pruning) is WEAK and GPU only; I-94 is a weak spare. D-34 (recommendation).
                  Source: dossier/ROUTES_R-17_2026-10-08.md. Ideas: I-82 to I-94.
2026-10-08  [P7]  Section 1 rewritten again "as of 2026-10-08, after R-17". The previous version
                  listed R-17 as the next action and D-33 as decided but not started. The new
                  version records R-17, I-82 to I-94 and D-34.
2026-10-08  [P7]  The user asked whether reading reports of waste can replace running systems, and
                  approved an orphan-finding scan (D-36, R-18): GitHub issues of 15 repositories and
                  30 characterization papers, a Sonnet harvest and a Sonnet attack. No survivor.
                  D-35 copied from a parallel session's memory notes. D-37 (recommendation).
                  Source: dossier/ORPHANS_R-18_2026-10-08.md. Ideas: I-95 to I-102.
2026-10-08  [P7]  Section 1 rewritten "as of 2026-10-08, after R-18". The previous version
                  recorded R-17 and D-34 as the latest state.
2026-10-11  [P7]  The user proposed assumption removal (D-38, R-19). 40 assumptions, 7 checked by the
                  main session and two Sonnet agents, Sonnet attack of three. I-106 (KV drift across
                  adjacent RL checkpoints) is the only open, Mac-testable question. D-39.
                  Source: dossier/ASSUMPTIONS_R-19_2026-10-11.md. Ideas: I-103 to I-109.
2026-10-11  [P7]  Section 1 rewritten "as of 2026-10-11, after R-19".
2026-10-16  [P7]  The user asked for a gap search on three seed topics (threads versus processes by
                  load, KV cache poisoning, GPU noisy neighbor), repeated against more papers until
                  gaps appear (R-20). Three search rounds and a Sonnet attack. I-118 MEASURE-FIRST
                  (low), I-116 WEAK, others KILL. E-30 drafted. D-40.
                  Source: dossier/GAPS_R-20_2026-10-16.md. Ideas: I-110 to I-119.
2026-10-16  [P7]  Section 1 rewritten "as of 2026-10-16, after R-20". The previous version recorded
                  R-19, D-39 and the I-106 premise check as the latest state.
2026-10-17  [P8]  Two-pass current-frontier review of nine candidate truths. Broad agent scheduling,
                  cross-layer recovery, sandbox prewarming, verification scheduling, and mid-decode
                  checkpointing were killed by current work. I-120 survived with high scoop risk.
                  I-121 was killed. I-23 remained the best first experiment. D-41.
                  Source: dossier/FRONTIER_R-21_2026-10-17.md. Ideas: I-120, I-121.
2026-10-17  [P8]  Section 1 rewritten "as of 2026-10-17, after R-21". The previous version recorded
                  R-20, I-118, and D-40 as the latest state.
2026-10-17  [P8]  The user asked to continue the frontier process until three to five total new gaps
                  survived. I-120 counted as the first gap (D-42). R-22 screened I-122 to I-130 in
                  two passes. I-122 and I-123 survived. I-124 to I-130 were killed. The review stopped
                  at three total gaps: I-120, I-122, and I-123. No experiment ran. D-43.
                  Source: dossier/FRONTIER_R-22_2026-10-17.md.
2026-10-17  [P8]  Section 1 rewritten "as of 2026-10-17, after R-22". The previous version recorded
                  R-21, I-120, and D-41 as the latest state.
2026-10-17  [P8]  The user approved the I-122 Mac test and requested Docker isolation (D-44). E-31
                  was predeclared. The code and container runner were written but not executed.
                  A static audit added an epoch-only baseline and out-of-order event checks. Docker
                  Desktop could not start because its code signature is invalid. The Claude CLI is
                  signed out, so the D-29 Sonnet review is also blocked. No experiment ran.
2026-10-17  [P8]  Section 1 rewritten after the E-31 predeclaration. The previous version recorded
                  R-22 and D-43 as the latest state.
2026-10-17  [P8]  Docker Desktop later served Linux containers through the desktop-linux context.
                  The server reports 10 virtual CPUs and 7.65 GiB of memory. The Claude CLI remains
                  signed out. E-31 remains blocked by its required Sonnet review. No experiment ran.
2026-10-17  [P8]  Claude authentication succeeded in the user's Terminal. The automation process
                  cannot read the macOS Keychain entry. A read-only Sonnet prompt and manual review
                  wrapper are ready. E-31 did not run.
2026-10-17  [P8]  R-23 reviewed E-31 before execution. Sonnet found four blocking design faults.
                  E-31 is INVALID DESIGN and did not run.
                  Source: dossier/REVIEW_E-31_PRERUN_2026-10-17.md. Ideas: I-122. Experiment: E-31.
2026-10-17  [P8]  R-24 audited current vLLM source and issue states in Docker. Narrow fixes cover
                  the closed failures. The global version mechanism lacks evidence. I-122 is PIVOT.
                  I-131 is the replacement MEASURE-FIRST question. E-32 is predeclared, not run.
                  Source: dossier/SOURCE_AUDIT_I-122_2026-10-17.md. Ideas: I-122, I-131.
2026-10-19  [P8]  R-25 reviewed E-32 before implementation. It found four blocking design faults.
                  E-32 requires amendment. E-33 was predeclared as the four-case CPU corpus pilot.
                  Source: dossier/REVIEW_E-32_PRERUN_2026-10-17.md. Ideas: I-131. Experiment: E-33.
2026-10-19  [P8]  R-26 reviewed the amended E-32 design. It found that circularity, oracle contracts,
                  and decision rules remained unresolved. E-32 did not run.
                  Source: dossier/REVIEW_E-32_AMENDMENT_2026-10-19.md. Ideas: I-131.
2026-10-19  [P8]  R-27 mapped four upstream tests to action sequences and ran full-history blame.
                  Only one trace fits depth six. No fault line came from an earlier narrow session fix.
                  I-131 is KILL as a thesis. E-32 and E-33 did not run.
                  Source: dossier/PAPER_AUDIT_I-131_2026-10-19.md. Ideas: I-131.
2026-10-19  [P9]  R-28 ran a fresh two-pass scoop audit for I-123. Current work now covers
                  heterogeneous model snapshots, durable GPU deltas, duplex ledgers, output replay,
                  playback acknowledgements, and ordered input acknowledgements as separate pieces.
                  The old CPU oracle would encode its result. I-123 is PIVOT. I-132 asks whether one
                  durable recovery epoch must join all pieces. E-34 is design only and did not run.
                  Source: dossier/SCOOP_AUDIT_I-123_2026-10-19.md. Ideas: I-123, I-132.
2026-10-19  [P9]  R-29 reviewed E-34 before implementation and returned KILL_IDEA.
                  The policy is standard checkpoint composition. B1 and B2 are its interval
                  endpoints. The deterministic workload omits nondeterministic GPU replay.
                  I-132 is KILL. E-34 is KILL-DESIGN and did not run.
                  Source: dossier/REVIEW_E-34_PRERUN_2026-10-19.md. Idea: I-132.
2026-10-19  [P9]  The user chose Mac tests for now. R-30 ran a scoop check before the I-106
                  premise run. CacheReforge (2609.30884) covers the I-106 mechanism for LoRA
                  updates. I-106 KILL, I-60 KILL, I-59 WEAK and Mac-testable. Nothing ran.
                  Source: this journal, Section 6.2 R-30. Ideas: I-106, I-59, I-60.
2026-10-19  [P9]  The user chose the I-59 premise check (D-47). E-35 predeclared. R-31 (Sonnet)
                  returned DO_NOT_RUN_CODE_DECIDES and proposed a trace check; amendment A1.
                  Run 1 INVALID (no in-flight window, lone-request exemption); amendment A2.
                  Source: experiments/priority_inheritance_premise/README.md. Idea: I-59.
2026-10-20  [P9]  E-35 run 2: KILL. R-32 (Sonnet) CONFIRM_WITH_SCOPE_LIMITS: no shared-prefix
                  dependency, but budget, capacity and preemption inversions exist. Paired
                  supplement: a shared prefix never delayed H in 600 scenarios. I-59 KILL.
                  Source: dossier/REVIEW_E-35_POSTRUN_2026-10-20.md. Idea: I-59.
2026-10-20  [P9]  The user gave the R-21 research prompt again. R-33 ran a fresh two-pass
                  frontier review on eight transitions not in the registry (I-133 to I-140).
                  All eight have direct prior work. Pass 2 found vLLM PR 53298, which
                  implements the I-120 mechanism: I-120 KILL as research. I-23 stays the
                  best survivor; new framing threat 2606.07581. D-50.
                  Source: dossier/FRONTIER_R-33_2026-10-20.md. Ideas: I-120, I-23, I-133 to I-140.
2026-10-20  [P9]  The user asked for the weaknesses each reviewed paper leaves for a next paper.
                  R-36 downloaded the 496 cited papers, extracted limitation and future-work text,
                  read it for the 372 recent papers that state any, and checked eight follow-up
                  gaps (I-141 to I-148). Seven KILL, one WEAK (I-143). D-53.
                  Source: dossier/LIMITATIONS_R-36_2026-10-20.md. Ideas: I-141 to I-148.
2026-10-20  [P9]  The user asked for local tests and plans until GPUs arrive (D-49). R-35 checked
                  the GPU-gated ideas, wrote the E-36 package (I-23, one H100) and the frozen
                  E-30 package (I-118, one GH200), and the GPU runbook. The vLLM CPU image build
                  for the Mac dry run failed three times (network twice, memory once) and is retried.
                  Source: dossier/GPU_RUNBOOK_2026-10-20.md. Ideas: I-23, I-118, I-120, I-83.
2026-10-20  [P9]  R-34 (Sonnet) pre-run review of E-36 and E-30: AMEND_THEN_RUN for both. A backend
                  flag can alias the default kernel; the E-30 prefill test hit the prefix cache.
                  Amendments A1 applied. The vLLM CPU image built on the fourth try (one compile
                  job). The E-36 Mac dry run passed mechanically after five fixes; one is a vLLM
                  v0.32.0 CPU bug that vLLM main fixed in PR 61992.
                  Source: dossier/REVIEW_E-36_E-30_PRERUN_2026-10-20.md. Ideas: I-23, I-118.
2026-10-21  [P9]  The user asked for an algorithmic-optimization sweep ("like competitive algorithmic
                  problems") with the PyTorch Conference NA 2026 schedule as a source (D-54). R-37
                  read SGLang, vLLM and PyTorch DCP source at current main and screened 16
                  candidates (I-149 to I-164). Fourteen KILL, I-161 is an engineering PR candidate,
                  I-163 is WEAK and GPU-gated. E-37 sized two CPU paths on the Mac. R-38 (Sonnet)
                  reviewed R-37 and E-37.
                  Source: dossier/ALGORITHMIC_R-37_2026-10-21.md. Ideas: I-149 to I-164.
                  Experiments: E-37.
2026-10-21  [P9]  R-38 (Sonnet) found SGLang PRs 38371 and 38454 (lazy eviction heaps, open) and
                  Hazy's public megakernel list scheduler plus Cohere's measured 10% on a dense
                  model. The main session verified both. I-161 and I-163 KILL; all 16 R-37
                  candidates are KILL. E-37 CONFIRM_WITH_SCOPE_LIMITS. D-55.
                  Source: dossier/REVIEW_R-37_SONNET_2026-10-21.md. Ideas: I-154, I-158, I-161, I-163.
2026-10-22  [P9]  The user set up Runpod (plugin runpod@runpod 1.8.0, MCP server signed in with
                  OAuth; a list-pods check returned 0 Pods). The user decided to run the GPU
                  experiments on Runpod and to keep all hardware details (D-56). Added
                  experiments/runpod/ and runbook Section 2a. Runpod Pods are unprivileged
                  containers, so E-30 stays on a provider with root. No Pod created, no spend.
                  Source: dossier/GPU_RUNBOOK_2026-10-20.md. Ideas: I-23, I-118. Experiments: E-36, E-30.
2026-10-22  [P9]  The user asked to use Runpod efficiently ("maximum details and data with maximum
                  savings"). E-36 amendment A2: resume, telemetry, kernel profiles, microbenchmark,
                  AUTO_STOP, exploratory Phase B; Phase A unchanged. A read-only Runpod session read
                  prices and billing rules. R-39 (Sonnet) returned AMEND_THEN_RUN; fixes applied, 22
                  tests and a stub session pass. D-57.
                  Source: dossier/REVIEW_E-36_A2_PRERUN_2026-10-22.md. Ideas: I-23. Experiments: E-36.
```

---

## 5. Idea dossiers

Order follows Section 3. Verdict history lines are never overwritten.

### 5.1 Phase 0 options (I-01 to I-05)

Source: `README.md` at git commit 4bd50d5 ("Research Domain Introduction and Literature
Review Log"). Domain: machine learning for reliable and efficient AI infrastructure. The
May log states "Active exploration. Not yet committed to a final project."

| ID | Option | Pitch | Lab and career fit (May log) |
|---|---|---|---|
| I-01 | ML-driven SDC detection in LLM training | Multivariate time-series anomaly detection on loss, gradients, attention statistics and hardware metrics, catching corruption NaN checks miss | Strong fit with lab time-series work. Independent of any industry collaboration |
| I-02 | GNN for cascading GPU-cluster failures | Model the cluster as a graph, predict how failures propagate | Strong fit with lab graph-ML work. Useful for training and serving |
| I-03 | Adversarial robustness of AI-infra monitoring | Attacks on ML-based failure prediction and anomaly detection, defenses using the lab's moving target defense work | Called very novel. Strong lab fit |
| I-04 | ML-driven checkpoint policies | ML layer over static checkpointing: predict failure timing, adapt frequency, choose recovery nodes | Aligned with the LinkedIn internship. Possible paper path via the CRIU author. Risk: dependency on an industry collaboration, narrower than the domain |
| I-05 | RL failure-aware inference scheduling | RL scheduler that handles node failures gracefully | Aligns with fall RL coursework. Less aligned with current lab work |

**Status of all five: DORMANT.** They were never adjudicated in this repo. No later
document mentions them by these names.

**Open questions recorded in May (no answers in the repo):**
1. Which option fits where the lab wants to grow?
2. How much should industry alignment (I-04) weigh against independence (I-01 to I-03)?
3. Is there an existing lab project to join?
4. What is realistic given the summer internship and fall coursework?
5. Which datasets does the lab already have?

**Reading list (19 items, all doc-only):** Gemini (SOSP 2023), CheckFreq (FAST 2021),
Just-In-Time Checkpointing (EuroSys 2024), LowDiff (arXiv 2509.04084), Convergence-Aware
Optimal Checkpointing, Predicting GPU Failures (SYSTOR 2023), LLM-PRISM, Exploring SDC as
a Reliability Challenge in LLM Training, Localizing Irregularities in LLM Training
(NSDI 2025), SENTINEL, FastServe (2305.05920), PARS (2510.03243), RLTune (SoCC 2025),
multi-agent graph RL cluster scheduling (2112.13354), two adversarial anomaly-detection
papers (2022, 2204.09183), When Scaling Fails (2603.04424), Enhancing Reliability in AI
Inference Services (2511.07424), MLPerf Storage v2.0 checkpointing.

**Research log**
- 2026-09-23: incidental search for I-01 found an active 2026 literature on silent data
  corruption: LLM-PRISM, The Anatomy of SDC (2605.04213), SCOUT (2608.11034), ITHICA
  (2605.15638), Syndrome Decoding for SDC in Quantized Integer GPU Arithmetic (2609.19743),
  and SProbe, an inference-side detector. Search-only, not reviewed. Treat I-01 as crowded
  until a review says otherwise.

**Verdict history**
- 2026-05-20: five options recorded, none committed.
- 2026-09-23: still DORMANT. No adjudication. The direction moved to systems research, and
  the reason is not recorded.

**Reopen if:** the user wants a lab-aligned ML-policy thesis, or advisor guidance requires
it. Run a review first. I-04 is the one option that connects to the later checkpoint/restore
theme (I-35).

---

### 5.2 I-06 Live migration of many active hybrid LLM sessions

Status: **KILL (2026-09-23).** Aliases: "Idea 1" in `stateful_elastic_inference_thesis_dossier.docx`.
Origin: that document, undated (file time 2026-09-16 17:51).

**Problem.** Hybrid LLMs mix attention layers with recurrent layers (Mamba/SSM or Gated
DeltaNet). Attention KV grows with context and is mostly append-only. Recurrent state is
fixed-size but rewritten every token. Transformer live migration (Llumnix) works because old
KV never changes: copy it while decoding continues, then pause briefly for the tail. A
state rewritten every token breaks that pattern.

**Research question.** Can a serving system live-migrate many actively decoding hybrid LLM
sessions across GPU replicas with bounded interruption and exact continuation by separating
append-only attention KV from continuously changing recurrent state? Stretch: the same
across TP layouts (TP1 to TP2, TP2 to TP4) without full prompt replay.

**Proposed design.**
1. Start migration at token version v, destination allocates state.
2. Copy stable attention KV in the background while the source keeps decoding.
3. Track the source token and any KV suffix created after the copy began.
4. At a token boundary, establish a migration epoch and stop the source committing more
   tokens for this request.
5. Transfer the remaining KV suffix, the latest recurrent and conv state, and minimal
   sampler and session metadata.
6. If topology changes, reshard before ownership commit.
7. Validate the destination, atomically switch ownership, resume.
8. An admission controller limits simultaneous transfers using fabric bandwidth and
   foreground SLO pressure. Replay is a fallback only.

**Hypothesis.** Most bytes leave the critical path. The hard production problem appears
when many sessions migrate together, when transfer traffic competes with serving traffic, or
when the destination uses a different layout.

**POC plan and gates (from the source doc).**

| POC | Content | Hardware |
|---|---|---|
| A | One session, same topology, GPU A to GPU B, exact cutover | 2 NVIDIA GPUs |
| B | 1, 8, 16, 32, 64+ sessions under foreground decoding, uncontrolled vs admission-controlled | 2 to 4+ GPUs |
| C | Topology change while source keeps decoding (TP1 to TP2 needs 3 GPUs) | 3 to 4+ GPUs |

| Gate | GO | PIVOT or KILL |
|---|---|---|
| Novelty scan | No public system does active hybrid decode-owner migration at our scope | Equivalent system lands first |
| Single-session POC | Correct continuation, controllable final pause | State cannot be extracted or installed safely |
| Scale value | Concurrent migration creates SLO or network pressure that our policy improves | Naive migration is negligible under realistic load |
| Topology value | Live resharding gives an elasticity benefit | Adds complexity without serving benefit |
| Paper story | Mechanism plus nontrivial result plus strong baseline wins | Only plumbing remains |

**Experiments:** E-01 synthetic cutover, E-02 Falcon-H1 state anatomy, E-03 restore
correctness, replay and bandwidth projection. Key own finding (E-03): one session's 27.49 MB
recurrent state costs 0.134 to 2.148 ms to move at 200 to 12.5 GB/s ideal bandwidth. So
single-session copy is cheap and value has to come from many-session pressure or topology
change.

**What prior art already answers (as of 2026-09-23, Section 7).** Llumnix, Astrolabe, OServe,
Pallas and ReMP cover live migration and topology change for attention models. WorldMove
covers bit-exact migration of mutable GPU state with incast-aware admission control. vLLM
already transfers hybrid conv and SSM state in P/D disaggregation, with heterogeneous TP.
NVIDIA Dynamo Snapshot is single-GPU only with multi-GPU in progress. Concordia does
GPU-side checkpointing with a standby pool for live migration.

**Verdict history**
- Undated: GO, narrowed scope. Next gate a fresh novelty scan plus a 2+ GPU POC.
- 2026-09-16: not in the master handoff's candidate map. Effectively orphaned. The novelty
  scan and the POCs are never recorded as done.
- 2026-09-23 (R-05): **KILL.** Rejection sentence: "This is Llumnix with a 100 MB fixed
  tail, WorldMove's admission controller, and vLLM's existing state transfer." Needs 2 to 4+
  GPUs. My partial scan found no public system doing live decode-owner migration of a
  hybrid session, but every component is published.

**Why closed.** Integration residue only, on scarce hardware, with a cheap per-session copy.
**What would reopen it.** A measured result that aggregate hybrid migration violates SLOs
under realistic load, plus multi-GPU access. Or the process-level route (I-35) showing a
distinct contribution.
**Lessons.** Test the scale question first. Per-session cost was never the problem.

---

### 5.3 Phase 1 screens (I-07 to I-12)

All from the 2026-09-16 handoff (V1, V2, V2.1, V2.2 suites, E-04 to E-06). The V1 synthetic
results were withdrawn. Every verdict below rests on V2 and later real Mac screens, which are
screens, not NVIDIA evidence.

**I-07 Activation residency (keep, recompute, offload). DEPRIORITIZED.**
Real PyTorch/MPS forward and backward timing. Keep was fastest at sequence lengths 384, 768
and 1152. No crossover found. Does not reproduce H100 HBM or PCIe behavior. The broad question
is crowded by PyTorch selective activation checkpointing, its memory budget work, and DTR.
Reopen with a narrow question backed by production traces.

**I-08 Network-adaptive overlap and granularity. DEPRIORITIZED.**
V2 was structurally flawed: every strategy sent the same 64 MiB, differing mostly in message
count, with no true overlap, so bulk sync was favored. V2.2 corrected it: 15 shaped-network
conditions, same compute chunks, 64 MiB total. Fine-grained overlap won 6, bulk sync 6,
coarse overlap 3. Median winner advantage about 2.9%, boundaries nearly tied. The crossover
is real but not large. Crowded by Syncopate (OSDI 2026), REACT (2026), PIPEMORPH and NCCL
tuning controls.

**I-09 Opportunistic heterogeneous elastic training. HOLD (backup).**
Alibaba 2026 spot-GPU trace, windows with 4 or more spare GPUs:

| GPU type | Median contiguous spare window | Fraction of windows above 4.36 s |
|---|---|---|
| A10 | about 7.5 s | about 53% |
| A100-80GB | about 305.5 s (5.1 min) | about 96.5% |
| GPU-series-2 | 382 s (6.4 min) | 80% |
| H800 | 7610.5 s (2.1 h) | 98.3% |

The 4.36 s anchor is DynaTrain's reported reconfiguration point for a 235B MoE model. Under
2 s was used for a 70B dense anchor. "Spare capacity" means node inventory minus concurrent
high-priority requested GPUs. It is an upper bound on opportunity, not proof one elastic job
could acquire it. V2's heterogeneous break-even was not trusted: combined CPU plus MPS
throughput exceeded the isolated sum by about 71% in one sample, likely a unified-memory
artifact. Surviving angle: decide whether to accept temporary GPUs of a different capability
class under uncertain lease length and capability-changing redistribution cost. Novelty thin:
GFS, DynaTrain, LiveR and Zorse are adjacent. Trace:
`github.com/alibaba/clusterdata`, `cluster-trace-v2026-spot-gpu`.

**I-10 GPU-sharing interference scheduling. KILL as standalone.**
Separate processes were required because same-process Metal concurrent encoder submission
crashed. Compute-heavy throughput fell 797.9 to 768.3 ops/s (-3.7%). Bandwidth-heavy fell
481.3 to 202.6 (-57.9%). Combined normalized throughput about 1.384. Real asymmetric
interference, but the scheduling space is crowded (Concord, STAO and related multi-tenant
work).

**I-11 Typed hybrid-state allocator. DEPRIORITIZED (led to I-13).**
Modeled, not measured, memory savings from typed pools:

| Scenario | Min | Median | Max |
|---|---|---|---|
| Irregular 2-type (20 full plus 30 sliding) | 16.7% | 36.4% | 53.7% |
| 3-type (plus 6 recurrent) | 9.0% | 39.8% | 59.9% |
| FP8 attention plus recurrent | 1.0% | 50.3% | 62.5% |

vLLM, SGLang, TensorRT-LLM, llama.cpp and AVMP already have separate or packed pools, so an
allocator alone is not novel. The screen motivated "is the missing abstraction broader than
allocation?", which became I-13.

**I-12 Power-aware parallelism switching. HOLD, untested.**
Only an early synthetic screen exists. It cannot support a conclusion. Needs real NVIDIA
power measurements and a novelty review.

---

### 5.4 Phase 2 hybrid-state line (I-13 to I-17)

Hybrid models: attention layers plus recurrent or linear-attention layers (Mamba2, Gated
DeltaNet). Real test model throughout: Qwen3.5-0.8B-Base, 24 layers, 18 Gated DeltaNet and 6
full attention (indices 3, 7, 11, 15, 19, 23), on Apple MPS.

**Evolution of the question (from the handoff):**
Typed state runtime, then reconstructability-aware state, then minimum-cost consistent-state
planner, then planner mostly unnecessary, then multi-state selective recovery, then real
Qwen3.5 tests expose inter-layer dependency, then performance headroom unresolved. The lesson
recorded: "different cache or state types exist" is not the gap. Current runtimes know that.

#### I-13 Typed and heterogeneous model-state runtime. KILL as broad thesis (2026-09-16)

E-07 ran a source-pinned harness against vLLM commit
`e0705731be04aac90838f318fcdae15c56a53080`. Current `HybridKVCacheCoordinator` already
supports multiple heterogeneous groups and fixed-point prefix reconciliation.

| Runtime construction case | Result |
|---|---|
| Full plus Sliding | Accepted |
| Full plus Sliding plus Mamba | Accepted |
| Full plus two Sliding variants plus Mamba | Accepted |
| Sliding plus Mamba without Full | Accepted |
| Full16 plus Mamba64 mixed block sizes | Accepted |
| DCP2 Full plus Mamba | Accepted |
| DCP2 Full plus Sliding | Rejected: only FullAttentionSpec and MambaSpec supported |
| EAGLE plus 3 groups | Accepted (construction only) |

Evidence that heterogeneous-state semantics are still a real runtime problem: vLLM #48455
(draft state can veto reusable target prefix), #53786 (one group's hit granularity
constrains another), #54661 (per-group retention RFC), #47812 (hybrid recovery after
KV-load failure had single-group assumptions), #55600 (global block geometry for Mamba
state), #49537 (hybrid KV transfer boundary), #51771 (EAGLE/MTP with 3+ groups), #55196 and
#54106 (FP8 recurrent memory pinning and group-ratio issues).
INVALID step: the upstream hybrid pytest returned code 4 because `tblib` was missing. Do not
cite it. The runtime and source probes did run.

#### I-14 REQUIRED / RECONSTRUCTABLE / EPHEMERAL contract. Mechanism only, folded

E-08 patched the same vLLM commit with `KVStateReuseMode = REQUIRED | RECONSTRUCTABLE |
EPHEMERAL` and an alternative `find_reconstructable_cache_plan()`. No model-name branches.
Default path unchanged.

| Case | Strict baseline reusable | Semantic reusable | Preserved or rebuilt |
|---|---|---|---|
| A: target 1024, ephemeral 0 | 0 | 1024 | 1024 target tokens avoid re-prefill |
| A: target 65536, ephemeral 0 | 0 | 65536 | 65536 avoid re-prefill |
| B: required 4096, reconstructable 1024 | 1024 | 4096 | 3072 preserved, 3072 rebuilt |
| B: required 65536, reconstructable 8192 | 8192 | 65536 | 57344 preserved and rebuilt |
| B: required 65536, reconstructable 49152 | 49152 | 65536 | 16384 preserved and rebuilt |

5 POC tests and the architecture-neutrality scan passed. **Valid conclusion:** one generic
software contract expresses both opportunities at the planning layer. **Not proven:** that
recurrent state can be rebuilt independently in a real interleaved model, that any Mamba
catch-up kernel runs, or any NVIDIA latency saving. Do not say "real recurrent recovery is
cheap".

#### I-15 Minimum-cost consistent-state planner (later RQ3). KILL (2026-09-17)

Stage 1, E-09 (normalized costs, not GPU timings): 324 rows (6 availability scenarios by 54
cost profiles).

| Metric | Value |
|---|---|
| Distinct optimal planner signatures | 11 |
| Mixed load-plus-rebuild rows | 140 of 324 (43.2%) |
| Best simple policy exactly matches planner | 281 of 324 (86.7%) |
| Median gain vs row-best simple policy | 0% |
| p90 gain | 2.35% |
| Max gain | 15.85% |
| Best global fixed policy | always_load_external |

Screen verdict PIVOT_OR_MEASURE_ONE_REAL_PATH. Multi-state scenarios were the only ones with
interesting gains (about 14 to 16% peak). Deprioritized 2026-09-16.

Stage 2 (revived as RQ3 in dossier iteration 4): real constants and a strong baseline.
E-16 cross-model constants (HF configs, fp32 recurrent state):

| Model | FA / rec layers | One snapshot | K/V B/token | Activation B/token |
|---|---|---|---|---|
| Qwen3.5-0.8B-Base | 6 / 18 | 18.00 MiB | 12,288 | 12,288 |
| granite-4.0-h-small | 4 / 36 | 144.00 MiB | 16,384 | 32,768 |
| Nemotron-H-8B-Base-8K | 4 / 24 | 96.00 MiB | 16,384 | 32,768 |

That is a 5.3x spread in snapshot cost. E-15 (evenly spaced checkpoints) was the weak
baseline. E-19 optimized placement by dynamic program per divergence distribution:
216 rows (3 models by 3 contexts, 2048, 8192, 32768, by 4 distributions by 6 budgets, 1 to
32 snapshots). Optimally placed recurrent checkpoints win at **every** budget from 1 to 8
snapshots, every model, every context, every distribution. The activation cache only wins at
16 or 32 snapshots, 288 MiB to 4.6 GiB per request. The four distributions (uniform,
front-loaded, back-loaded, bimodal) give almost identical winner patterns. The crossover is
set by byte budget, not by where requests diverge.

Verdict: KILL 2026-09-17. Two independent methods agree that simple policies win.
Scoop risk if reopened: Sparse Prefix Caching (exact DP placement), vLLM #55697, #52959.
Reopen only for a genuinely new cost class.

**Corrections made on the way (kept for the record):**
1. Snapshot bytes first assumed fp16, giving 9.63 MiB. The live cache stores gated-DeltaNet
   state in float32: 18.84 MiB measured (18.00 state plus 0.84 conv), matching the
   ledger's independent 19.76 MiB. This halved the crossover budget.
2. Nemotron-H was first reported as 252 MiB, then 192 MiB. The script used the MLP
   `intermediate_size` as the Mamba inner width, then counted every non-attention layer as
   state-holding. Its config lists 24 linear_attention, 24 mlp and 4 full_attention. Corrected
   to 96 MiB.
3. The first dynamic program wrote the same cost into `dp[jj][c]` and `dp[jj][c+1]` and
   returned a constant regardless of budget. Fixed and validated against the closed form
   n / (2 (m + 1)) for the uniform case.
4. Open inconsistency: the result file `results_rq3_budget.json` says `grid_points: 48`, but
   its 216 rows factor as 3 by 3 by 4 by 6 (36 model-context-distribution cells). Not
   resolved. Low impact.

#### I-16 KV-aware selective recovery. KILL (2026-09-17)

**Narrow question (from the handoff).** Given a hybrid model with interleaved full-attention
and recurrent layers, and a divergence where full-attention KV is valid deeper than
recurrent state, can a correct recovery path exploit that to cut meaningful compute at
realistic context lengths, beyond sparse checkpoints, ReplaySSM and existing hybrid serving?

**Verdict history**
- 2026-09-16 handoff: MEASURE-FIRST, not killed, because v1.1's negative timings were
  invalid (E-11). Confidence low to medium on performance, low on novelty.
- 2026-09-16 to 17 (dossier iteration 4): **KILL on performance**, three independent
  arguments:
  1. Measured (E-17): the sound frozen-K/V mechanism is flat at zero across an 8x context
     range (per-context medians between -0.9% and +1.8% across the frozen and bypass
     variants, with the sweep-wide medians at 0.07% and 0.67%).
  2. Analytical ceiling (E-14): the sound frozen-K/V variant is worth 1.22% of replay at 512
     tokens, falling to 0.18% at 131,072.
  3. Prior claim: vLLM #53041 (2026-08-20) proposes exactly this mechanism, and
     `get_computed_blocks_for_connector` at the pinned commit already ships divergent
     hybrid hits, with #52773 measuring load-backfill 80x cheaper than recompute.
- Reviewer rejection sentence: "#53041 already proposes keeping the FA hit and rebuilding
  lagging state, #52773 already measured that loading beats recomputing by 80x, and your own
  FLOP model says the sound version is worth under 1.2% and shrinking."

**Why closed.** Sound version worth about 1% and shrinking, faster variant unsound or
memory-expensive, upstream already shipped the load-based version.
**Assets that survived:** the numerically exact divergent-state replay path (E-12), and the
finding that #53041 Proposal B is unsound for interleaved stacks (became I-19).

#### I-17 Activation-assisted recovery. KILL (2026-09-17)

Idea: cache exact post-decoder-block hidden activations so whole full-attention blocks can
be skipped during replay. E-11 measured 9.53% in one case (context 512, suffix 128, two
reps, MPS), which was too small to generalize. E-17 randomized sweep, oracle speedup medians:
10.77% (512), 12.24% (1024), 15.52% (2048), 19.69% (4096), range 9.28% to 21.12%. It grows
monotonically with context and tracks the analytical ceiling (23.1% to 34.0%) while sitting
below it, because skipping a block still costs a read and the Mac's slow reference recurrent
kernels inflate the recurrent share. The catch is O(context) extra memory: 12,288 B per
token on Qwen3.5-0.8B. E-15 and E-19 show optimally placed checkpoints beat it at every
realistic budget. Closest prior art: HYPIC's seam window, which independently reaches the
same "propagate hidden states" insight for segment concatenation. Verdict: KILL, folded
into I-15.

---

### 5.5 I-18 Compute-aware collective execution-engine choice

Status: **MEASURE-FIRST, low priority (2026-09-23).** Confidence in gap: high on the
structural gap, low on value. Aliases: RQ2, "Idea 2" in `ideas/`.
Origin: 2026-09-16 handoff section 16 (parallel track, UNTESTED). Briefs:
`archive/ideas/02-collective-engine-arbitration.md`, dossier section 5 and 10.4.

**Problem.** A GPU can drive a collective two ways: a kernel on the SMs, or copy engines
(CE). NCCL 2.28 added CE collectives, reaching zero-SM operation for AllGather and AllToAll
inside an NVLink domain. The SM path takes SMs from a concurrent compute kernel. The CE path
leaves SMs alone but adds memory-system pressure.

**Original hypothesis (2026-09-16).** SM-bound compute kernel: prefer CE. Bandwidth-bound
compute kernel: CE traffic may intensify bandwidth contention, so prefer the SM path. A
runtime policy picks the engine from live compute state.

**Research question.** Can selecting the collective execution engine, SM-driven versus
copy-engine, from the live resource profile of the concurrently executing compute kernel
improve end-to-end overlapped throughput over both NCCL's default selection and the best
static `NCCL_CTA_POLICY`, while preserving ordering and progress semantics? Falsified if a
static per-phase or per-layer policy captures most of the gain, or if the perfect-information
gain over NCCL's default is small.

**Source audit (E-20 on NCCL 2.31.2, re-run as E-22 on 2.32.3).** FACT:
1. The whole CE-versus-SM routing decision is one line:
   `if ((CTAPolicy & NCCL_CTA_POLICY_ZERO) && (ceAvailable || hierCeAvailable))`
   (`src/enqueue/enqueue.cc:3305` in 2.31.2, line 3398 in 2.32.3). A static policy AND a
   static eligibility predicate. No cost model.
2. `ncclCeAvailable` has five rejection reasons, all static: collective not implemented for
   CE, host-backed cuMem segments, incomplete NVLink connectivity, symmetric support
   disabled, unsupported window registration type. None is a load condition.
3. The tuner plugin is not consulted for CE routing. Its latest interface (`tuner_v6.h`)
   gives `getCollInfo` only `context, collType, nBytes, numPipeOps, collCostTable, numAlgo,
   numProto, regBuff, nChannels`. No occupancy, bandwidth pressure or handle to the
   concurrent kernel.
4. Searching `src/` for `occupancy`, `cudaOccupancy`, `smUtil`, `GetUtilization`,
   `activeWarps`, `nvmlDeviceGetUtilizationRates`, `smActive`, `achieved_occupancy` gives zero
   matches.

**Original predeclared experiment (2026-09-16).**
1. Two controlled compute kernels: SM-bound (large GEMM at high occupancy) and
   bandwidth-bound (streaming triad). Confirm bottleneck with profiler counters.
2. Overlap each with AllGather and AllToAll under CTA policy DEFAULT, EFFICIENCY, ZERO.
   Sweep message size and channel count.
3. Report end-to-end overlapped throughput, not collective time alone.
4. Oracle: best static policy per (kernel, message size) cell with hindsight.
5. GO if the per-cell oracle beats the best single global policy by a margin surviving
   switching cost. KILL if one static policy is within noise of the oracle. Thresholds are
   heuristics, fixed before the run.

**R-05 findings (2026-09-23).**
- FACT: NCCL now has `ncclCollConfig_t` with `*WithConfig` calls (struct version 23200, so
  2.32; not checked whether 2.31.2 had it). Per call it sets `CTAPolicy`, `minCTAs`,
  `maxCTAs`, `nvlsCTAs`, `algSelection`. The per-call value wins. So SM-vs-CE can already be
  chosen per collective by a framework. The missing piece is a policy, not a mechanism.
- FACT: NVIDIA's NCCL 2.28 blog shows CE multicast AllGather at about 780 vs 620 GB/s (1.25x)
  over the SM path at 4 GB on 8 Blackwell GPUs. CE is ahead across 4 MB to 4 GB, and no case
  where CE loses is named. Standalone, not overlapped.
- FACT: NCCL's Q2 2026 roadmap adds zero-SM AllGather and explores SM-initiated CE
  collectives. Runtime SM-vs-CE selection is not listed.
- FACT: the survey the docs cite (2608.15118) contains the quoted sentence, in its
  "Communication Offloading" future-work paragraph. The same paragraph says offloading to
  copy engines "is already practical in many systems". It names generic co-design as open,
  not CE-vs-SM arbitration.
- FACT: the literature targets SM budget, not engine: 2606.09200 cuts total time up to 25.5%
  with occupancy shaping and no vendor library change, 2609.07536 picks communication CTA
  counts analytically, FLUX, Weave and mKernel fuse compute and communication.
- FACT: CE covers data-movement collectives (AllGather, AllToAll) inside one NVLink domain.
  TP AllReduce in inference is not a CE collective.
- INTERPRETATION, untested: an SM copy and a CE copy read and write the same bytes through
  HBM, so "bandwidth-bound kernel prefers the SM path" has no mechanism I can see. What
  differs is granularity and launch cost (CE suits large contiguous moves). That is a
  message-size rule, which NCCL's static eligibility partly encodes.
- **Design flaw in the original experiment.** Kernel type and message size are known at the
  call site without measurement, so a static per-call table implements the per-cell oracle.
  The GO test compares the oracle to the best single global policy. It never tests whether a
  live signal beats a static table, which is the "live resource profile" claim.

**Revised Test B (2026-09-23, run only if an NVLink node appears).** Bundle with the numerics
grid of I-34 in one rental.
1. Keep the original design.
2. Add a within-cell test: fixed kernel class and message size, varying compute pressure
   (GEMM shapes across arithmetic intensity, attention at different batch and context).
3. Add the baseline "per-call static table from op type and size".
4. GO (heuristic): a live-signal policy beats that table by at least 5% overlapped
   throughput. KILL (heuristic): the per-call table is within 2% of the oracle.

**Hardware.** At least 2 NVLink-connected GPUs, realistically an 8-GPU H100 or H200 node.

**Verdict history**
- 2026-09-16: UNTESTED candidate. Do not promote just because hybrid-state work struggled.
- 2026-09-16 to 17: MEASURE-FIRST. Gap confirmed from source. Size unknown. Blocked on hardware.
- 2026-09-18: keep warm, switch to it if an NVLink node appears (`RESEARCH_STATUS.md`).
- 2026-09-23 (R-05): MEASURE-FIRST but low priority. Source gap replicates on 2.32.3, but
  the mechanism exists per call, CE is ahead in NVIDIA's data, the planned test cannot
  separate live signal from static table, and NVIDIA's roadmap is moving.

**What not to claim.** That the survey names CE-vs-SM arbitration as open, or that a
tuner-plugin can implement it.
**Next step and kill rule.** Fix the design as above before spending a node. Kill if the
per-call static table is within 2% of the oracle.

---

### 5.6 I-19 Soundness of per-layer-group state reuse

Status: **KILL as thesis (2026-09-23), salvage as a short note.** Confidence in gap:
medium. Aliases: RQ1, "Idea 1" in `ideas/`. Origin: dossier iteration 4 section 4. Briefs:
`archive/ideas/01-state-reuse-soundness.md`.

**Problem.** Hybrid models mix full attention, sliding window and recurrent layers. Engines
cache each type separately. On a partial prefix match each type can hit at a different depth
(for example full attention 128 tokens, recurrent 0, because recurrent state is one
position-keyed snapshot). The engine then decides per group what to reuse and what to
recompute. A wrong decision can silently corrupt output.

**Formal core (the "condition").** A reuse plan assigns each (layer, position) to EXECUTE or
SKIP. It is sound only if the executed set is input-closed at every position: for each
executed layer L at position p, either layer L-1 also executes at p, or L-1's output
activation at p is cached. Corollary for recurrent-state backfill: the skippable set is
exactly the contiguous suffix of layers above the highest state-holding layer.

**Research question (as written).** Can the safety condition for per-layer-group state reuse
be stated, checked mechanically, and used to decide which reuse plans are worth
implementing, and does applying it to shipped and proposed mechanisms reveal unsoundness
that existing example-based tests miss? Falsified if the condition is equivalent to what
engines already assert, or applying it finds nothing current tests miss.

**Evidence.**

E-18 (plan-level gate against real vLLM on CPU, pinned commit):
- Real per-group hits from the real coordinator, 128-token prompt cached then a sharing
  request looked up:

| `mamba_cache_mode` | per-group hits | default path | connector path | diverged |
|---|---|---|---|---|
| none | (128, 128) | 128 | 128 | False |
| align | (128, 0) | 0 | 128 | True |

  In align mode the default path discards 128 already-valid full-attention tokens. This
  reproduces vLLM #45238 on a laptop.
- Static checker vs measured logit difference, three plans with ground truth, 3 of 3 agree:

| Plan | Static verdict | Measured max logit diff |
|---|---|---|
| upstream_default_reconciled | sound | not measured |
| upstream_connector_divergent_load | sound | not measured |
| issue_53041_proposal_B_recompute_backfill | unsound | 1.0554 |
| frozen_kv_but_fa_still_executes | sound | 0.0117 |
| activation_cached_fa_skip | sound | 0.0117 |

  For Proposal B the checker names the starved layers 4, 8, 12, 16, 20. The noise floor used
  as a threshold in the harness is 0.15. Gate result: pass.

E-21 (suffix rule): predicted maximal sound skip set equals brute-force exhaustive search
over all subsets of full-attention layers on six layouts. On the real model, skipping layer
23 during replay leaves recurrent state bit-identical (max abs diff exactly 0) at contexts
512, 1024, 2048 and 4096. Correctness criterion: recurrent-state equality, not logit
equality. Measured saving 1.52%, 1.01%, 1.01%, 6.40% (8 reps, randomized). Analytical
prediction at 2048 was 4.74%.

E-21 layout survey: see I-20. Upstream also keeps itself sound with hand-written guards, for
example `get_computed_blocks_for_connector` and the connector flag
`supports_divergent_local_hybrid_hits`.

**Verdict history**
- 2026-09-16: GO, behind one feasibility gate (dossier section 4).
- 2026-09-16 to 17: gate PASS (E-18).
- 2026-09-17: `ideas/01` records "GO, recommended thesis".
- 2026-09-18: `RESEARCH_STATUS.md`: proceed with this as the thesis.
- 2026-09-23 (R-05): **KILL as thesis.**

**R-05 findings (why it was killed).**
- FACT: #53041 is a discussion post by one author, zero comments at fetch time. It names
  per-layer selective compute as the key technical challenge itself. The docs called it an
  active upstream design discussion.
- FACT: the "five instances of this bug class" are different classes:

| Issue | Root cause (from the issue text) | Would the checker catch it? |
|---|---|---|
| #57266 | Out-of-range strided slice when `num_computed_tokens % mamba_block_size != 0` | No |
| #55600 | Unit mismatch, divides by `block_size` (reassigned to the minimum across groups) instead of `mamba_block_size`. One-line fix | No |
| #56655 | Prefix cache key omits the `prompt_is_token_ids` mask | No |
| #47812 | Single-group assumption in KV-load failure recovery (taken from the docs, not opened) | Not shown |
| #53041 Proposal B | Skipping FA layers starves downstream recurrent layers | Yes, but it is an unimplemented sketch |

  The checker covers one of five, and that one has no shipped code. The doc's own falsifier
  ("too weak to catch the known bug class") fired.
- INTERPRETATION: input-closure is the data-dependence graph of a transformer. The
  non-trivial part, extension to sliding window, speculative decode, DCP and offload, is
  unbuilt.
- FACT, prior art: SGLang's Unified Radix Cache (2026-08-11) ships a reuse-boundary rule
  ("advances only when all validators accept"). Ekka (ICML 2026) and GRIEF (2026-05) do
  differential debugging and fuzzing of silent serving errors. Runtime Observability for
  Heterogeneous Attention Memory (2608.05863) has a formal contract with Lean 4 proofs
  covering recurrent state. Validating Hybrid-State Cache Recovery (2609.15030) found a
  recovery bug via generation agreement. vLLM has open PRs for batch-invariant hybrid
  prefix caching (#55868, #55895, #56244).
- FACT, constructive patch: skipping layers above the last recurrent layer only helps the
  recompute-backfill path. #52773 measured load-backfill 80x cheaper and it is shipped, so
  recompute is a last resort. Worth 1 to 6% on a path that rarely runs.
- The original doc's own risk: reads as software testing rather than systems, weakest career
  fit of the set.

**Why closed.** Motivation and checker are mismatched, prior art covers the wider space, and
the constructive artifact is on a rare path.
**Salvage (1 to 2 weeks at most).** A short workshop or technical note: the checker, the
layout survey (I-20), and the suffix-skip verification. Do not build it into a thesis.
**What would reopen it.** Checker-found bugs in shipped code or a wider class (SWA,
speculative decode, DCP, offload) that other tools miss, or upstream implementing recompute
backfill.
**Lessons.** Confirm that a "bug class" is one class before building a tool for it. Read
the cited issues, not just their titles.

---

### 5.7 I-20, I-21, I-22

**I-20 Layer-layout co-design to unlock divergent reuse. KILL (2026-09-18).**
Alias "Idea 3", RQ3b, dossier section 11. If interleaving is what makes divergent reuse
unsound, a layout without interleaving would make it sound and valuable. E-21 surveyed 13
real hybrid models from HF configs (no weights):

| Model | Layers | FA | Skippable suffix | Last layer |
|---|---|---|---|---|
| Qwen3.5-0.8B-Base | 24 | 6 | 1 | full attention |
| Qwen3-Next-80B-A3B | 48 | 12 | 1 | full attention |
| Nemotron-H-8B / 47B / 56B | 52 / 98 / 118 | 4 / 5 / 10 | 1 (an MLP block) | mlp |
| granite-4.0-h small / tiny / micro | 40 | 4 | 0 | recurrent |
| Bamba-9B | 32 | 3 | 0 | recurrent |
| LFM2-1.2B | 16 | 6 | 0 | recurrent |
| Zamba2-2.7B | 54 | 0 | 0 | fused |
| Falcon-H1 0.5B / 34B | 36 / 72 | 0 | 0 | fused |

Recurrent-first layouts: 0 of 13. Models where more than half of full-attention work is
soundly skippable: 0 of 13. Models where nothing is skippable: 8 of 13. Falcon-H1 and Zamba2
are fused parallel hybrids, so the attention path cannot be skipped independently under any
layout. Achievable vs counterfactual share of replay work (boundary 0.75, analytical): for
Qwen3.5-0.8B at 131,072 tokens, achievable 14.72% vs 88.3% if recurrent-first. Nemotron-H-8B
achievable falls with context (2.49% to 1.63%) because the one skippable block is an MLP
whose cost is linear in suffix length. Measured Qwen3.5 savings came in below analytical
because the Mac's recurrent kernels are slow reference code. Quote the analytical figure as
the ceiling, not the measured one.
Verdict: KILL. Closing the gap requires proposing a layout change, and validating that a
recurrent-first hybrid keeps model quality needs pretraining, far outside a master's
budget. It survives only as a finding inside I-19.

**I-21 Elastic reconfiguration equivalence. HOLD, low confidence.**
Not formulated beyond one row in the dossier decision table. Threats: DynaTrain, ElasWave,
ByteCheckpoint. Their oracle is loss-curve comparison. Backup only.

**I-22 LLM queueing-boundary idea. KILL, thin trail.**
Recorded only in `RESEARCH_STATUS.md`: "a 2026 paper already develops the stronger dynamical
model". The paper is not named and no other trail exists in the repo. If it ever matters,
re-derive the evidence.

---

### 5.8 I-23 SafeDispatch / KernelGuard

Status: **MEASURE-FIRST, narrowed (2026-09-23).** Confidence in gap: low to medium.
Origin: `LLM_Inference_Thesis_Research_Handoff_2026-09-22.docx`. Full R-05 write-up:
`dossier/ADVERSARIAL_REVIEW_2026-09-23.md`, section 3.1.

**One sentence.** Build and evaluate a correctness-aware kernel and backend qualification and
selection layer that picks the fastest implementation for an LLM inference workload only
when the substitution is demonstrably safe under an explicit numerical and model-output
contract.

**Why the problem exists.** Serving stacks expose several implementations for GEMM,
attention, MoE, sampling, quantization, CUDA graph paths and communication. The fastest one
depends on shape, routing skew, batch size, precision, hardware and runtime state. They are
treated as interchangeable because they implement the same math, but finite-precision
behavior, scale application, reduction order, padding and layout assumptions and compiler
choices differ. Autoregressive decoding amplifies small changes when logits are close, and
one flipped token changes the whole continuation.

**Research question (2026-09-22).** Can real-activation-aware numerical contracts and
runtime context predict when nominally interchangeable GPU kernels or backends are unsafe to
substitute, enabling faster LLM inference than a static conservative implementation while
preserving an explicit model-output correctness target?

**Hypotheses (2026-09-22) and status after R-05.**

| Hypothesis | Status 2026-09-23 |
|---|---|
| H1 Non-interchangeability: some real operators and regimes give downstream output differences a fixed small-sample allclose test misses | Published (Integer Alibi: CUTLASS vs Triton INT8 in vLLM gave 0/8, 0/16, 0/64 sequence agreement) |
| H2 Structured risk: unsafe swaps correlate with shape, precision or scale statistics, reduction depth, activations, hardware, routing skew, logit margin | Published (Integer Alibi, MarginGate, 2609.26621) |
| H3 The fastest safe implementation varies across regimes often enough to matter versus one conservative kernel | Open. No paper measures it |
| H4 A lightweight rule captures most of the oracle benefit at low overhead | Open |

**Do not claim as novel (2026-09-22).** Dynamic kernel dispatch, GPU autotuning, stronger
kernel fuzzing, cross-hardware drift measurement, deterministic LLM inference, selective
high-precision execution, or the observation that different kernels give different outputs.
R-05 adds: margin gating, kernel fingerprinting and static bitwise-equivalence checking,
TP-invariant kernels.

**Candidate surviving contribution.** A serving-oriented qualification and selection
mechanism mapping real activations plus workload, hardware and backend context to a
performance-optimal choice under an end-to-end model-output constraint, with evidence that
simpler contract checks or static policies leave performance or correctness on the table.

**Correctness hierarchy.**

| Level | Meaning |
|---|---|
| 0 Contract | Shape, dtype, finite and non-finite behavior, aliasing, masks and padding, scale conventions, edge cases |
| 1 Tensor numerical | Absolute, relative or ULP error vs reference, error distribution, backend-pair agreement |
| 2 Layer or hidden state | Activation divergence after the operator and downstream, amplification or attenuation |
| 3 Logit | Max and mean logit error, KL or JS, top-k changes, argmax flips, logit margin |
| 4 Sequence | Exact match under greedy decoding, first-divergence token, divergence rate |
| 5 Task | Accuracy or quality delta on a small evaluation set, only for candidates that pass lower levels but still alter outputs |

**Mechanism options (do not choose before measurement).** Offline qualification matrix
(allow-list per model, hardware, config). Deployment-time gate. Runtime rule-based dispatch.
Runtime learned risk model (only if simple rules fail and data suffices). Fast path plus
sampled verification (overlaps LLM-42 and MarginGate).

**Baselines (2026-09-22).** Fastest-only autotuner. Static conservative reference. Fixed
allclose gate. Contract-grade verifier or fuzzing-inspired gate. Simple logit-margin
threshold. Per-shape allow-list. Oracle fastest-safe choice.

**Strongest objections (2026-09-22) and what R-05 added.**

| Objection | Original answer required | R-05 update |
|---|---|---|
| "Just FlashInfer autotuning plus a correctness test" | Show the model-behavior constraint changes which kernel is chosen and ordinary tests are insufficient or too conservative | FlashInfer v0.7's autotuner ranks on latency only. But vLLM's Q3 roadmap lists model-aware backend selection |
| "DriftBench already predicts drift" | It is configuration-level and leaves mitigation open. Localize to substitutable kernel choices and show a serving action | Stands |
| "Integer Alibi already proves divergence" | Use as prior art, generalize across operators, regimes or dispatch | Its limitations section defines the open cells (see below) |
| "LLM-42 solves consistency" | Different objective (determinism by verify and rollback) | Also MarginGate: margin-triggered verification, 2.23x lower latency than always-on LLM-42 |
| "A simple logit-margin rule is enough" | Benchmark it. If it captures most value, pivot | That baseline is now a published system (MarginGate, 2609.26621) |
| "The phenomenon is too rare" | Kill the runtime-dispatch thesis if the oracle gain is small | The static safe baseline got cheaper: 2609.25624 (1.17x to 3.1x faster than prior deterministic SOTA), 2609.11356 (19 of 27 kernels within 10% of free order) |

**Open cells, from the closest paper's own limitations (FACT).** Integer Alibi states: "One
consumer GPU (sm_89), one model family at two sizes, one engine version, and two INT8 kernel
implementations (plus one FP8 pair and one cross-engine comparison). Nothing here licenses
claims across GPU generations, model families, or future kernels." It has no MoE, no NVFP4,
no multi-GPU, and its FP8 result is operator-level only with no end-to-end sequence
agreement. Independent evidence the cells are live: FlashInfer's audit puts about 36% of bugs
in low-precision GEMM and MoE, about 31% in MoE, and about 60% of fixes in Python dispatch
and validation. Open B200 reports: vLLM #55238 (CUDA graph replay changes greedy output on
sm100 only, no root cause at fetch) and #55649 (NVFP4 gated-MoE end-padding corrupts output,
fix PR unmerged). Scoop risk is high: the Integer Alibi author posted two papers in 12 days.

**R-33 note (2026-10-20).** New framing threat: "Training-Inference Kernel Contracts" (2606.07581,
2026-05-26) defines kernel contracts with an escalation policy that maps violations to routing actions.
It reports no evaluation and does not mention FP8, NVFP4, MoE, Hopper or Blackwell. Add the contract
concept to "do not claim as novel". Add 2609.25624 fixed-order kernels as a Test A baseline where they
run on Hopper. Status unchanged: MEASURE-FIRST.

**R-35 note (2026-10-20).** vLLM v0.32.0 (2026-10-18) has first-class backend switches:
`--linear-backend`, `--moe-backend`, `linear_backend_per_quant` and `--attention-backend` (FACT,
`vllm/config/kernel.py`, `vllm/engine/arg_utils.py`). Each backend pair in Test A is one flag, and a
dispatch layer has a clean hook. The E-36 package (`experiments/safedispatch_testA/`) is Session 1 of
Test A on one H100: calibration, backend inventory, two FP8 cells at a fixed batch configuration, and a
STOP-FLAT / CONTINUE / INCONCLUSIVE rule. No direct new scoop was found.

**R-34 note (2026-10-20).** FACT (v0.32.0 source): when a requested `--linear-backend` has no kernel
for a layer, vLLM logs a warning and uses normal kernel selection. Some backend values are aliases
of the default kernel. A backend flag is therefore no proof of a different kernel. E-36 now detects
aliases from vLLM's kernel-selection log and uses Marlin W8A16 as a positive control.
INTERPRETATION: a dispatch layer for I-23 must record the resolved kernel, not the requested flag.

**Narrowed question (2026-09-23).** On Hopper and Blackwell, for block-scaled FP8, NVFP4 and
fused-MoE backends, does the fastest backend that satisfies an explicit output contract
differ across serving regimes enough to beat the best single fixed-order backend by at least
5% end to end?

**Test A: oracle test (1 GPU, about 1 to 2 weeks). Thresholds are heuristics.**
1. Reproduce one documented divergence to validate the harness (Integer Alibi INT8, or
   #55238 on Blackwell if available). Not a contribution.
2. Grid: at least two model families (one dense, one MoE), at least three backends per
   operator family, block-scaled FP8 and NVFP4 where hardware allows, BF16 control, prefill
   vs decode, several batch sizes, real captured activations plus synthetic controls,
   exact greedy decoding for sequence tests.
3. Contracts: L1 bitwise equal to a fixed-order reference kernel. L2 zero greedy flips on a
   margin-stratified prompt set. L3 KL bound on logits.
4. Baselines: S1 fastest single always-safe backend. S2 fixed-order deterministic kernels.
   S3 per-shape allow-list from offline qualification. Include MarginGate-style gating as a
   policy comparison, not as a contribution.
5. Oracle gain G = end-to-end tokens/s or TPOT of the fastest-safe oracle over the best of
   S1 to S3.

| Result | Decision |
|---|---|
| G at least 5% on 2 or more models, and a runtime rule beats S3 by at least 2% | GO for correctness-aware dispatch |
| G under 5%, but at least 10% of default-config substitutions in the open cells violate L2 | PIVOT to the measurement thesis (I-33 rides on the same data) |
| G under 2% and unsafe fraction under 2% | KILL |

**Original gates from the 2026-09-22 handoff (for the record).**

| Observation | Decision |
|---|---|
| No meaningful cross-kernel output differences under realistic activations | KILL SafeDispatch |
| Differences exist but fixed contract or fuzz tests identify dangerous swaps and performance ordering is static | PIVOT to deployment-time qualification |
| Risk is regime-dependent but a per-shape allow-list captures nearly all oracle benefit | PIVOT to static qualification |
| Risk and performance both vary with runtime context and the fastest safe kernel changes often | GO for runtime dispatch |
| One cheap feature (logit margin) captures almost all risk | Use the simple mechanism as the contribution |
| Safety or dispatch overhead erases the speedup | PIVOT to offline or deployment-time gating |

**12-week plan from the handoff (Oct to Dec).** Week 1 environment and harness (pin
vLLM, PyTorch, CUDA, kernel-selection logging, reproduce one divergence). Week 2 automate
activation capture, kernel-pair execution, metrics, greedy replay, timing. Weeks 3 to 4
problem-existence sweep, random vs real activations, oracle and baselines. Week 5 claim lock,
choose GO, PIVOT or KILL, write a 2-page claim and the strongest reviewer rejection
sentence. Weeks 6 to 7 build the simplest supported mechanism. Week 8 serving integration
(TTFT, TPOT, throughput, tails). Weeks 9 to 10 generalize (second model, operator family,
GPU generation). Week 11 adversarial review and ablations. Week 12 paper draft and open
source. January rule: by the job start, the core claim has already survived or failed.

**Implementation blueprint from the handoff.** vLLM first, SGLang only if a comparison needs
it. FlashInfer plus Triton and CUTLASS or vLLM-native kernels. Nsight Systems and Compute,
CUPTI and NVTX where needed. Python, pandas, JSON or Parquet artifacts. Containerized,
pinned SHAs, fixed seeds, manifests. Suggested layout `safe-dispatch/` with `configs/`,
`harness/`, `contracts/`, `metrics/`, `integrations/`, `policies/`, `experiments/`,
`results/`, `scripts/`, `tests/`, `paper/`. Minimum run-manifest fields: run id, git SHAs,
container image, GPU SKU, driver and CUDA, model and checkpoint hash, dtype and
quantization, operator and kernel id, shape or regime, input source and activation id, seed,
latency samples, tensor metrics, logit and token metrics, task metric, pass or fail at each
correctness level. Candidate risk features, only after measurement: static operator (dtype,
M/N/K, reduction depth, layout, scale convention, GPU architecture), activation (magnitude,
outliers, sparsity), serving context (prefill or decode, batch, sequence length, routing
skew, CUDA-graph mode, parallel config), downstream sensitivity (logit margin, layer
position).

**Fallback pivots (2026-09-22).**

| Pivot | Question | Novelty threat |
|---|---|---|
| A Cross-hardware and backend conformance and release qualification | High-signal workflow with real activations plus contract and model-level checks to approve upgrades | DriftBench, contract-grade verifier, FlashInfer quality RFC |
| B Counterfactual value prediction for kernel and runtime optimizations | Predict TTFT and TPOT impact before implementing | KernelSight-LM, Vidur, OpenAI performance modeling, PROMPTS |
| C Correctness-aware MoE kernel dispatch | Narrow to fused MoE where ordering changes with routing skew | DA-MoE dispatches for performance already |
| D Performance-regression qualification across accelerator generations | Is a change both fast and output-compatible on new hardware | DriftBench, production qualification. Needs two GPU generations |

**Career story and venues (2026-09-22).** "I already knew distributed ML reliability and
multi-GPU infrastructure; my thesis took me down the stack into LLM serving, kernels,
profiling, low precision and numerical correctness." Venue families: MLSys, ASPLOS,
ISCA/HPCA, OSDI/SOSP/NSDI/EuroSys, or a workshop, decided after week 5.

**Questions the handoff left open.** Which operator family gives the best signal (scaled
GEMM, fused MoE, attention, sampling, CUDA-graph path). Is risk static or does it vary enough
to need runtime dispatch. What is the cheapest predictive metric. How much end-to-end speedup
survives the correctness constraint. Does it generalize across GPU generations.

**Verdict history**
- 2026-09-22: MEASURE-FIRST. Novelty confidence medium. Career fit very high. Three-month
  feasibility high if the first claim stays narrow and one-GPU experiments are the critical
  path.
- 2026-09-23 (R-05): MEASURE-FIRST, narrowed to the open cells, GPU-gated. Likely outcome
  is PIVOT to a measurement thesis on cells no one has filled, plus an open-source harness.
  Do not build a learned dispatcher first.
- 2026-10-17 (R-21): MEASURE-FIRST. FlashInfer-Bench, FlashInfer v0.7, and the vLLM
  consistency stack raise novelty risk. The contribution must be the output-aware oracle.

**Next step and kill rule.** Test A above, gate in writing at the end of week 3. Also kill
the dispatch thesis if FlashInfer checks plus a per-shape allow-list capture 95% of oracle gain.

---

### 5.9 I-24 to I-32 (candidates from the 2026-09-22 handoff)

Verdicts are from that handoff's section 4 and were not re-reviewed in R-05 except where noted.

| ID | Candidate | Verdict | Reason and closest threats |
|---|---|---|---|
| I-24 | Generic cross-layer LLM profiler | KILL | TELLER (2608.01975), Nsight, HTA and vLLM tracing already make basic cross-layer profiling and RCA crowded |
| I-25 | Generic inference performance simulator | KILL | Vidur (2405.05465) and KernelSight-LM (2606.28565) cover end-to-end and kernel-level simulation |
| I-26 | LLM agent recommends serving configs | KILL / PIVOT | PROMPTS (Google) and production lab workflows automate this |
| I-27 | LLM-generated GPU kernels | PIVOT | KernelEvolve (Meta), KernelBench-X, Atrex-Bench, FastKernels. Microbenchmark wins may not improve end-to-end serving |
| I-28 | Agent-aware GPU scheduling | PIVOT | Murakkab (OSDI 2026) and others exploit workflow structure. Thin novelty boundary for three months. R-05 search also saw Continuum, SMetric, UNISON on agent KV residency. 2026-10-04 (R-08): the "no public harness traces" angle is weaker than thought: TraceLab (2606.30560, 4,265 Claude Code and Codex sessions, released) and the Exgentic OpenTelemetry dataset exist. TraceLab lists append-length-aware prefill routing, tool-latency-aware cache eviction and cheaper KV storage as open directions. I-45 folded in here |
| I-29 | Generic fault-tolerant inference | PIVOT | Overlaps existing strengths and active serving and recovery work, adds less new career signal. See I-35 and I-37 |
| I-30 | Generic deterministic inference | KILL | LLM-42 directly attacks it. R-05 adds MarginGate, TBIK, CoRun, 2609.25624 |
| I-31 | Selective high-precision stabilization as a generic thesis | KILL | LAMP and BGSS already do risk-based selective precision. R-05 adds 2609.26621's margin-triggered FP32 lm_head |
| I-32 | Robotics or weapons-capability evaluation | DEPRIORITIZED | Not chosen: the user has no strong robotics preference, hardware and logistics risk, and it would move the career signal away from GPU and inference roles |

---

### 5.10 Hypotheses from R-05 (untested)

**I-33 Feature-level divergence and speed ledger for cached multi-turn workloads on MoE and hybrid models.**
Status HYPOTHESIS, confidence low.
- Problem: which serving optimizations are safe to turn on, and what does each buy?
- Why possibly open: DriftBench varies hardware, framework, precision and model, not
  serving features. Silent Hyperparameter (2605.19537) isolates prefix caching and CUDA
  graphs but is single-GPU with standard precision. Same Request, Different Answer
  (2609.04748) measures cache-induced divergence (36.2% of agent trajectories at 16-bit,
  75.0% at 4-bit, 80 episodes) with no mitigation.
- Cheapest test: the Test A matrix with feature flags (prefix cache, CUDA graphs, speculative
  decode, TP size).
- Threat: DriftBench authors extending, the vLLM nightly quality-test effort.
- Hardware: 1 to 4 GPUs.

**I-34 Reduction-order and precision contract of NCCL's newer collective paths.**
Status HYPOTHESIS, confidence low.
- Why possibly open: the NCCL source and docs I grepped state no reduction-order contract.
  NCCL issue #2360 reports BF16 NVLS AllReduce results that change with multicast allocation
  history and differ across hosts (no NVIDIA reply visible). A search result says NVLS is
  deterministic on Hopper and Blackwell with CUDA 12.8+, and NVIDIA's NCCL 2.27 blog snippet
  says FP8 reductions on NVLS accumulate in FP16. Neither is traced to a primary spec.
- Cheapest test: bitwise-agreement grid across algorithm, protocol, CE, symmetric kernels
  and message size on a real node, then the effect on token flips in TP inference.
- Threat: NVIDIA fixing it, vLLM determinism PRs (I-40), TBIK.
- Hardware: 4 to 8 NVLink or NVSwitch GPUs. Bundle with Test B (I-18).

---

### 5.11 Areas scanned on 2026-09-23 (not developed)

Search context only. Levels are search-only unless stated. Treat each as crowded until a
proper review says otherwise.

- **I-35 Multi-GPU process-level checkpoint/restore for inference engines.** NVIDIA Dynamo
  Snapshot is single-GPU only and lists multi-GPU as in progress (checked). vLLM RFC #34303
  is a draft with no maintainer comments and notes NCCL communicators must be rebuilt
  (about 1 to 3 s estimated) and CUDA graphs with NCCL collectives need recapture (checked).
  Concordia (2606.23521) does GPU-side checkpointing with a standby pool for live migration.
  Foundry (2604.06664) materializes CUDA graphs for 99% lower cold start. FlowGPU
  (Euro-Par 2026), PhoenixOS and CRIUgpu cover the mechanism. Closest to the user's LinkedIn
  experience. Scoop risk high. Employer IP question open (Section 1.4).
- **I-36 Runtime cache-integrity verification.** Runtime Observability for Heterogeneous
  Attention Memory (2608.05863, checked): formal contract, Lean 4, four cache types,
  always-on fused probe under CUDA graphs. Also Bit-Flip Vulnerability of Shared KV-Cache
  Blocks (2604.17249) and Aborted but Not Forgotten (2608.15939, KV retention vs rollback in
  agents). Direct threat to any detector idea.
- **I-37 Fault-tolerant LLM inference.** FailSafe (2511.14116), LUMEN (2606.17787), AnchorTP
  (2511.11617), DejaVu (2403.01876), and a Dynamo recovery paper (2609.25451). vLLM Q3
  roadmap targets EP recovery and asynchronous scaling.
- **I-38 RL training-inference mismatch and kernel alignment.** Miles, IsoExec (vLLM blog,
  2026-08-21), RL-Kernel, QaRL (2604.07853), AIS, QUADS, FP8-RL. vLLM's Q3 RL roadmap lists
  trainer and rollout consistency work. RL-Kernel already does "runtime adapters that select
  the operator implementation while keeping the same numerical contract", which is close to
  I-23's idea in the RL setting.
- **I-39 Hybrid-model determinism and recurrent-state precision.** vLLM PRs #55868
  (batch-invariant GDN prefix caching on an LCM grid of FLA chunk size 64 and mamba block
  size), #55895 (pin Mamba2 SSD Triton configs), #56244 (gate inside the scan). Casting SSM
  cache to FP16 reported up to 37 to 40% verbosity increase, with block-scaled INT8 plus
  stochastic rounding largely preserving FP32-cache accuracy. DASC (2608.30386) compresses
  recurrent state checkpoints by 2.63x (checked).
- **I-40 TP determinism and communication-fusion cost.** vLLM #50136 (custom all-reduce
  selection is batch-dependent), PR #50505 (about 2.05x throughput at TP=4 in
  batch-invariant mode, 8006.4 vs 3906.0 tok/s), #56370 (sequence parallelism and fused
  all-reduce plus RMSNorm break batch invariance), follow-ups #56377, #51292, #57042,
  reviewer proposal of absolute rank-order accumulation. Upstream is closing it.

### 5.12 I-41 Ingest structurer, and variants I-42 to I-45

### I-41 Ingest structurer
Status: KILL as a thesis mechanism   Confidence in gap: low   Aliases: "structurer LLM", "compress-on-ingest"
Origin: 2026-10-04, the user's idea (pasted notes): instead of sending raw text, a small LLM that
uses prompt caching rewrites the prompt and each tool output into a terse structured form (facts)
for the main model, so compaction happens much later and the main model keeps its prompt cache longer.
Problem: agent context grows with every tool output; compaction rewrites the prefix and invalidates
the prompt cache; long contexts cost more and degrade quality.
Research question: can small-model compression at ingest, keeping the main context append-only,
cut cost and delay compaction compared with raw history plus threshold compaction, at equal task
success? "No" if compaction is not what breaks the cache, if a perfect compressor saves little, or
if the compressor removes what the agent later needs.
What prior art already answers: ACON (ingest compression by an LLM, distilled to 8B to 14B),
TokenPilot (ingestion-aware compaction with a byte-identical prefix, batched eviction), AgentDiet
(cheaper reflection model, delayed edits to keep the KV cache), SWE-Pruner and Squeez (0.6B and
2B goal-conditioned verbatim line pruning), Claude Code WebFetch (Haiku-class reader), Anthropic
context editing and server compaction, OpenAI compaction, TraceLab and #94177 (cache misses come
from idle gaps). Section 7.10.
Exact surviving gap: none for the mechanism. A measurement of verbatim reuse of tool output by
coding agents was not found in public work (moved to I-43). Do not claim compress-on-ingest,
small-model compressors, cache-aware eviction or the TTL finding as novel.
Experiments: E-24 (P1 FALSE: compaction 0.39% of break writes, TTL expiry 83.4%; P2 oracle 23.8% at
r = 4; P3 compressor cost 0.43%, net 14.6%; P4 TRUE: 71.5% of edits copy tool-output text verbatim).
E-25 (0.6B abstractive structurer kept the later-edited span in 1 of 30; a first-and-last-lines rule
kept 6 of 30).
Reviews: R-08.
Research log:
- 2026-10-04: idea logged from the user's notes. Literature pass, E-24 and E-25 predeclared and run,
  second search pass, dossier `dossier/DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md`.
Verdict history:
- 2026-10-04 (R-08): KILL as a thesis mechanism. Published mechanism, false cache premise for this
  workload and at TraceLab scale, saving inside the published range, rewriting conflicts with
  verbatim reuse.
Why closed: see verdict. As engineering advice it holds in an extractive, recoverable form: keep raw
outputs recoverable, prune lines instead of paraphrasing, compress only large outputs, batch clears.
What would reopen it: evidence that compaction, not idle gaps, dominates cache misses in a
realistic multi-user workload (for example long autonomous runs), or a compressor design that keeps
verbatim spans without a stated goal and beats SWE-Pruner.
Lessons: check what actually breaks the cache in real traces before designing around a
cache-invalidation story. The user's own transcripts were a usable, free trace source.

### I-42 to I-45 (narrowed variants, R-08)
- **I-42 Cache-aligned context editing.** KILL. Proposed publicly in claude-code issue #94177
  (2026-09-14) as "run microcompact before predicted cold requests" and "batch the prune";
  TokenPilot batches evictions, Anthropic offers `clear_at_least`. A harness heuristic with high
  scoop risk.
- **I-43 Reference-aware retention.** DEPRIORITIZED. The measurement in E-24 P4 (71.5% verbatim
  reuse, about half of sources older than the last 3 tool results) was not found in TraceLab or
  elsewhere. Mechanism threats: TokenPilot lifecycle-aware eviction, AgentDiet expired-context
  removal, CWL dependency graph, SWE-Pruner Pro. Next step only if the user wants a short note:
  port the analysis to TraceLab's public traces (needs permission to download). 2026-10-04:
  TraceLab's public release has no tool inputs or output text, so it cannot test this; the
  candidate replacement is the Exgentic dataset (Section 4, Phase 6). Kill rule
  (heuristic): under 20% of tool-sourced edits from outside the last 3 tool results: KILL. At least
  30%: write the note with a retention-policy simulation that counts re-reads as cost.
  Verdict history: 2026-10-04 (R-08) DEPRIORITIZED. 2026-10-04 (E-26) passes the check on 391
  public SWE-bench traces, M = 43.7%, so MEASURE-FIRST as a short note only. Caveat: Claude Opus 4.5
  alone is 29.1% (strict 21.8%). Benchmark runs are short; in the user's own long sessions 38% of
  sources are more than 20 tool outputs back. Still harness-level and outside the thesis focus.
  Next step (only if the user wants the note): a retention-policy simulation (keep-last-K vs an
  oracle that keeps only outputs referenced later) counting retained tokens and forced re-reads,
  on E-24 and E-26 data. Kill rule for the note (heuristic): if keep-last-K with K at most 10
  breaks under 10% of tool-dependent edits in long real sessions while clearing at least half of
  tool tokens, reference-aware retention adds nothing. Gating data problem: multi-user real
  sessions with tool text are not public.
  2026-10-05: dropped by the user (D-26). No note and no simulation. What would reopen it: a public
  multi-user dataset of long real sessions with tool text, or a thesis direction that needs agent
  context retention as a serving-side signal.
- **I-44 Edit-tolerant KV reuse.** SCANNED, crowded: PatchKV (2609.26219), KVEraser (2606.17034),
  CacheBlend, EPIC, MiniPIC, KV Packet, Attuner. Needs a GPU.
- **I-45 TTL-resilient KV retention for agent idle gaps.** SCANNED, crowded: OpenAI 24 h
  retention (search-only), Anthropic 1 h TTL, Keepalive Economics (2607.19214), CachedAttention,
  Mooncake, LMCache. TraceLab lists cheaper KV storage and eviction as an open direction. Folded
  into I-28 as context.

### 5.13 I-46 to I-57 (R-09 cross-domain directions)

Full write-up: `dossier/NEW_DIRECTIONS_R-09_2026-10-06.md`. Only peer-reviewed papers are evidence.
Preprint scoop signals are in that file, Section 7 and Section 9.

### I-46 Sampled simulation of LLM serving traces
Status: MEASURE-FIRST   Confidence in gap: medium
Origin: 2026-10-06, R-09. Borrowed from SimPoint (ASPLOS 2002) and SMARTS (ISCA 2003).
Problem: a serving benchmark on a long production-like trace costs hours of GPU time for each
configuration. Agent traces have a long memory (prefix-cache contents), so a short window without
warm-up can give wrong results.
Research question: can systematic sampling with prefix-cache warm-up estimate mean and p99 TTFT,
TPOT, throughput and hit rate of a long agent trace within 3% error at 10x lower cost, when random
windows without warm-up cannot?
What prior art already answers: Vidur (MLSys 2024) simulates full traces. PKH (EuroMLSys 2026)
samples configurations, not time windows. UniCache (SIGMETRICS 2026) and KVCache Cache in the Wild
(ATC 2025) simulate or characterize full traces.
Exact surviving gap: time-window sampling of serving traces with state warm-up and an error bound.
Do not claim serving simulation or configuration sampling as novel.
Next step and kill rule: Mac test with Vidur or a small simulator, TraceLab and a chat trace,
full-trace simulation as ground truth. KILL if random windows without warm-up are within 3% on all
metrics at 10x reduction on the agent trace. GO if they err by more than 10% on p99 TTFT or hit rate
and a warm-up or clustered sampler gets under 3%. Otherwise PIVOT. Heuristic thresholds.
Verdict history:
- 2026-10-06 (R-09): MEASURE-FIRST.
- 2026-10-07 (R-10): KILL. A benchmarking method, which the user excluded (D-29). On the Mac the
  ground truth is a simulator, so the GPU-hour claim cannot be tested.

### I-47 Closed-loop validity of agent serving evaluation
Status: MEASURE-FIRST   Confidence in gap: low (high scoop risk)
Origin: 2026-10-06, R-09. Borrowed from Schroeder et al., "Open versus closed" (NSDI 2006), Mean
Value Analysis (JACM 1980) and the Engset model.
Problem: most serving benchmarks use open-loop arrivals. Agents are closed loop with tool and think
time. Closed loops hide overload and change scheduling conclusions.
Research question: with matched request sizes, do policy rankings or session capacity at an SLO
change by more than 20% between open-loop arrivals and closed-loop replay of TraceLab sessions?
Does an MVA or Engset model predict closed-loop capacity?
What prior art already answers: Parrot (OSDI 2024), InferCept (ICML 2024) and KVFlow (NeurIPS 2025)
evaluate agent programs in closed loops. None tests whether open-loop conclusions transfer.
Next step and kill rule: same simulator as I-46. KILL if rankings are the same and differences are
under 5% at all tested loads. GO if one ranking flips or a capacity estimate differs by more than
20%. Otherwise PIVOT. Heuristic thresholds. I-46 and I-47 can form one Mac-only thesis.
Verdict history:
- 2026-10-06 (R-09): MEASURE-FIRST.
- 2026-10-07 (R-10): KILL. A benchmarking method, which the user excluded (D-29).

### I-48 Margin-gated quality control for production inference numerics
Status: MEASURE-FIRST   Confidence in gap: low to medium
Origin: 2026-10-06, R-09. Borrowed from CUSUM control charts (Page, Biometrika 1954) and rational
subgrouping in factory quality control. Builds on the E-23 flip invariant.
Problem: benign precision noise flips 0.26% to 1.07% of positions between backends (E-23), and only
at small logit margins. A raw mismatch alarm must tolerate that floor, so a rare numeric bug can
hide under it. An Anthropic engineering postmortem (September 2025, not a paper) reports an
approximate top-k miscompilation that stayed undetected for about two weeks.
Research question: at the same shadow budget and false-alarm rate, does a CUSUM monitor that counts
only large-margin flips detect argmax-changing bugs at a bug rate at least 10x lower than a
raw-mismatch monitor and an MMD output test? Which bug classes are margin-visible?
What prior art already answers: Model Equality Testing (ICLR 2025), the anytime-valid behavioral
shift audit (ICLR 2025) and TOPLOC (ICML 2025) are black-box tests or verification of claims, not
token-level monitoring under a sample budget.
Exact surviving gap: provider-side, margin-gated, token-level numeric monitoring with a measured
minimum detectable bug rate. Do not claim margin as a flip-risk signal as novel.
Next step and kill rule: about 2 weeks on the Mac with the E-23 harness. Inject bug classes B1 to
B4. KILL if fewer than 20% of B1 and B2 flips have a margin above the gate, or if the gated monitor
is within 2x of the raw monitor. GO if the gate lowers the minimum detectable bug rate by at least
10x. Otherwise PIVOT. Heuristic thresholds. Recommended first (D-28).
Verdict history:
- 2026-10-06 (R-09): MEASURE-FIRST.
- 2026-10-07 (R-10): KILL. DiFR (2511.20621) and Log Probability Tracking (ICLR 2026) cover
  reference-based and logprob-based detection. The flip invariant is a mathematical identity
  (Section 8.1 row 42), so the margin gate is a coarse threshold on logit deviation.

### I-49 Failure-clustering-aware recovery provisioning
Status: HOLD (data-gated)   Confidence in gap: low
Origin: 2026-10-06, R-09. Borrowed from Hawkes processes (Biometrika 1971) and the ETAS aftershock
model (JASA 1988).
Problem: if GPU failures cluster in time, a training cluster can keep more hot spares and check
health more often for a short time after each failure.
What prior art already answers: Lazy Checkpointing (DSN 2014) uses temporal locality of failures
for HPC checkpoints. Kokolis et al. (HPCA 2025) and Cui et al. (SC 2025) characterize GPU failures
but do not release node-level logs that I could confirm. AcmeTrace (NSDI 2024) has job-level
NODE_FAIL states only.
Why HOLD: the final claim needs node-level failure logs. A weak first test is possible: fit Poisson
and Hawkes models to AcmeTrace NODE_FAIL times. KILL if the branching ratio is under 0.1.
What would reopen it: node-level failure data from the advisor's lab or a past employer, with IP
clearance (Section 1.4 item 5).
Verdict history:
- 2026-10-06 (R-09): HOLD.
- 2026-10-07 (R-10): KILL as a thesis route. The Delta GPU error data is public (Zenodo, CC BY 4.0),
  so the data gate in R-09 was wrong (Section 8.1 row 43). A model fit is not a mechanism, and
  "Sparing Strategies" (MLSys 2026) gives closed-form spare sizing.

### I-50 to I-57 (killed or folded in R-09)
- **I-50 KV retention across tool and human gaps.** KILL. InferCept (ICML 2024), Pensieve (EuroSys
  2025), KVFlow (NeurIPS 2025), KVCache Cache in the Wild (ATC 2025), UniCache (SIGMETRICS 2026).
  Many 2026 preprints. Overlaps I-45.
- **I-51 Speculative tool execution.** KILL. Speculative Actions (ICLR 2026 oral), LLMCompiler
  (ICML 2024), Dynamic Speculative Agent Planning (ICLR 2026).
- **I-52 Reproducible summation for batch invariance.** KILL. The source method is Demmel and Nguyen
  (ARITH 2013). The journal killed I-30. Performance claims need a GPU.
- **I-53 Agent sandbox fork for tree search.** KILL. SnowFlock (EuroSys 2009) and Firecracker
  (NSDI 2020) are the base. Products ship millisecond fork. Many 2026 preprints.
- **I-54 Group testing for faulty GPU nodes.** KILL. Greyhound (ATC 2025), Minder (NSDI 2025). An
  NVIDIA readiness tool already uses hierarchical group testing (vendor blog).
- **I-55 Hedged requests with prefix-cache affinity.** KILL on first principles: most LLM latency
  variance comes from the request itself, and a hedge to a cold replica loses the prefix cache.
  Load-aware routing (Llumnix, OSDI 2024; Preble, ICLR 2025) is the simple baseline.
- **I-56 Certified token stability from rounding-error bounds.** DEPRIORITIZED. Whole-network sound
  bounds are too loose, and I-31 was killed. The useful part moved into I-48.
- **I-57 Fused tool calls.** KILL. CodeAct (ICML 2024), TroVE (ICML 2024), LLMCompiler (ICML 2024).
Lessons: in 2026, every agent-serving and agent-harness area that I searched has a preprint flood.
Ideas that reuse a mature method from another field and a harness that this repo already has
(I-48) are the cheapest to test.

### 5.14 I-58 to I-75 (R-10 mechanism-transfer routes)

Full write-up: `dossier/ROUTES_R-10_2026-10-07.md`.

### I-58 Noise-shaped V-cache quantization along the token axis
Status: NARROW, MEASURE-FIRST   Confidence in gap: medium (no direct prior art found)
Origin: 2026-10-07, R-10. Borrowed from first-order sigma-delta modulation and error diffusion
(audio and image dithering).
Problem: low-bit V caches round each token independently. The attention output sums values over
many tokens, so independent errors add up.
Mechanism: before rounding token t, add the rounding error of token t-1 (a per-channel carry).
The read path and the attention kernel do not change.
Research question: on real attention, does per-head gated noise shaping lower the attention-output
error and perplexity of a 2-bit V cache by a useful margin over a Hadamard-rotated 2-bit V cache
(the llama.cpp baseline since 2026-04-01)?
What prior art already answers: no work applies error feedback to the V cache along tokens.
Closest: a sigma-delta disclosure for FP8 attention-score tiles (2026-06-30), dither for recurrent
caches (2609.39185), sigma-delta for 1-bit weights (SDQ-LLM), and subtractive dither for KV
(HyperQuant).
Reviews: R-10. The reviewer's synthetic check (re-run): shaped vs plain error 0.05 for diffuse
weights, 0.18 for a smooth tail, 1.20 to 1.55 with 10% to 60% spike mass. One more bit gives 0.43.
Next step and kill rule: predeclare E-27. [The rules in this paragraph are superseded by E-27
amendment A1, Section 6.1.] Step 1 measures the smoothness G = ||Dw|| / ||w|| per
head on Qwen3-0.6B. KILL if under 30% of output energy sits in heads with G < 0.5. Step 2 compares
none, shaping, gated shaping, key-sorted gated shaping and one more bit, all with a 64-block
Hadamard rotation. KILL if gated shaping gives under 1.5x lower median output error than rotation
alone, the perplexity gain is under 0.05, or needle accuracy falls by more than 2 points.
Heuristic thresholds.
Verdict history:
- 2026-10-07 (R-10): NARROW, MEASURE-FIRST.
- 2026-10-07 (E-27, R-12): KILL. On Qwen3-0.6B, only 5% of KV heads gain from shaping, shaping on all
  heads makes the layer-local error 71% worse, and the best possible gate gains under 0.1% in every
  region, length and order that R-12 tested. The attention on the quantized tokens is spiky.
Why closed: the premise (smooth attention weights along the token axis on the quantized region) is
false for this model. The rotated 2-bit V baseline already costs only 0.004 nats.
What would reopen it: a model whose attention over old tokens is measured to be diffuse (G below 0.5 for
a large share of the error energy), for example a long-context retrieval-free workload on a much larger
model. Check G first; it takes minutes.
Lessons: a 10-minute smoothness measurement on real attention would have predicted this. Measure the
premise on real data before building the full test. The two pre-run reviews prevented a wrong step-1
kill and a likely out-of-memory crash.

### I-59 Priority inheritance for shared-prefix producers
Status: WEAK backup   Confidence in gap: low
Origin: 2026-10-07, proposed by the R-10 reviewer. Borrowed from OS priority inheritance.
Problem: a high-priority request can wait behind a low-priority request that prefills a shared
prefix (priority inversion; vLLM issue 59907 reports inversion during prefill).
Next step and kill rule: a discrete-event simulator with synthetic priorities (public traces have no
priority tags). KILL if the high-priority p99 TTFT improves by under 15%.
Verdict history:
- 2026-10-07 (R-10): WEAK backup.
- 2026-10-19 (R-30): WEAK, still open. No source applies priority inheritance to a shared-prefix
  producer. vLLM issue 59907 (open, 2026-10-04) reports a different inversion (running requests get
  the token budget in admission order) and was reproduced with `VLLM_TARGET_DEVICE=cpu`. Fix PR 59963
  is open. RelServe (2601.11546) updates priorities by progress but does not inherit them. A premise
  check on the real vLLM CPU engine in Docker is therefore possible on the Mac. Open design question:
  vLLM recomputes a shared prefix instead of waiting for an in-flight producer, so the inversion needs
  a case where the high-priority request depends on the low-priority prefill.
- 2026-10-20 (E-35, R-31, R-32): KILL. vLLM never makes a waiting request wait for another request's
  prefix, and it publishes prefix blocks when it schedules them. Its running loop takes the token
  budget in admission order and ignores priority. SGLang's shared-prefix wait cannot run together with
  priority scheduling. Run 2 of E-35 confirmed this on the real scheduler class, and a paired sweep
  (600 scenarios) never found the high-priority request later with a shared prefix.
Why closed: inheritance repairs a task that is blocked on a lower-priority holder. No such block exists
through a shared prefix. The inversions that do exist (budget order, KV capacity, preemption) do not
depend on the prefix, and lifting an already-running producer does not change them.
What would reopen it: an engine that makes a request wait for another request's in-flight prefix
(for example singleflight prefill merged with priority scheduling), or a KV-connector path that does
so in practice, measured on a real workload.

### I-60 Late binding for continuous-batching routers
Status: WEAK backup   Confidence in gap: low
Origin: 2026-10-07, proposed by the R-10 reviewer. Borrowed from Sparrow (SOSP 2013) and delay
scheduling (EuroSys 2010).
Next step and kill rule: a simulator. KILL if the p99 TTFT gain over prefix-affine power-of-two
choices is under 15%. Close to the killed routing family (I-55).
Verdict history:
- 2026-10-07 (R-10): WEAK backup.
- 2026-10-19 (R-30): KILL. CacheRoute (2608.19677, Meta) plans prefix-affinity routing and limits
  residual load skew. The llm-d scheduler already breaks affinity when the warm pod's predicted
  latency is much worse. That is the delay-versus-locality trade-off of delay scheduling.

### I-61 to I-75 (screened in R-10, killed)
See Section 3.9 and Section 6 of the R-10 dossier for the prior art behind each kill. In short:
working-set load control (preprints), tool-result caching (TVCache), log-structured flash KV
(Lachesis, Tutti), duplicate-prefill merging (SGLang ships it), flowlet routing (DualMap,
CacheRoute), miss-ratio partitioning (UniCache), sparse RL weight sync (SparseRL-Sync, PULSE, TRL),
erasure-coded checkpoints (GEM at FAST 2027, GhostServe at MLSys 2026), lossless collective
compression (ZipCCL), async RL snapshots (Belayer), test selection for agents (PerfAgent, TDAD),
Smith-predictor autoscaling (2609.20874), subtractive KV dither (HyperQuant), fencing for agent
retries (2609.29095), dynamic knobs (PALS).
Lessons: since PagedAttention, most direct OS, storage and network mechanisms have moved into LLM
serving and RL infrastructure, often within months in 2026. Remaining gaps are narrow numerics
details or need simulator evidence.

### 5.15 I-76 to I-81 (route search 2, R-13)

Origin: 2026-10-07. The user chose option 2 of D-31 (look for new ideas). Method: a list of about 100
classic mechanisms from operating systems, databases, networks, architecture, compilers, numerics,
signal processing and control was mapped to LLM and agent infrastructure. Almost all of them were
already moved. The search then went to new environments (Apple Silicon unified memory) and to the
stated limits of the newest papers. Evidence is search-only unless stated.

- **I-76 Bandwidth regulation for on-device decode.** KILL. "Inference in the Shadows: Taming Memory
  Bandwidth Contention in Mobile LLM Inference with Sereno" (OSDI 2026) and PhaseGate (2610.04537,
  NeurIPS 2026 workshop, measured on M4 Macs) cover it.
- **I-77 KV hibernation to SSD.** KILL. LLMS (2403.11805), mzCache (2609.01338), TierKV (2609.21172).
- **I-78 Thermal pacing.** KILL. ThermE (2610.00267), EnerInfer (2606.23001), PELM (2609.09662).
- **I-79 Slack stealing in decode steps.** KILL. Agent.xpu (2506.24045) fills decode slack on SoCs;
  lm-Meter (2510.06126) proposes overlap in idle windows.
- **I-81 Copy-aware speculative decoding.** KILL. SuffixDecoding, REST and prompt-lookup decoding cover
  drafting from earlier context.
- **I-80 Metastability-aware admission for online agent serving.** HYPOTHESIS.
  - Idea: a load spike fills the server. The server evicts prefix caches. Agents time out and retry,
    and each retry must recompute a long prefix. That extra work can keep the server overloaded after
    the spike ends (a metastable failure, OSDI 2022: retries were the sustaining effect in over half
    of 22 studied incidents). Borrowed fix: retry budgets and load shedding, changed to keep the caches
    of admitted sessions warm and shed new or evicted sessions first.
  - Prior art: CONCUR (2601.22705) applies congestion control to agent thrashing collapse, but only for
    batch inference, with no timeouts or retries. ThunderAgent (2602.13692) and RR-Evict (2609.32278)
    address eviction. FailureAtlas (2607.17525) lists retry storms. No study found that measures the
    loop (overload, eviction, retries, hysteresis) in LLM serving.
  - Premise check before any build (simulator on the Mac): does a short spike cause a collapse that
    lasts after the spike ends, with realistic agent timeouts, retries and prefix-cache sizes? If not,
    KILL.
  - Not yet reviewed by a different model (D-29 requires that before a test).
  - Verdict history: 2026-10-07 (R-14, Sonnet): KILL as written (retries are not needed for the
    collapse in a toy simulator, and a session cap removes it). The narrowed form (an online cap that
    fits the working set) is WEAK. E-28 replicates these findings with real trace sessions.
  - Verdict history: 2026-10-08 (E-28, R-16): KILL CONFIRMED WITH SCOPE LIMITS. The frozen E-28 grid
    found no collapse (K1). The R-16 reviewer check found a lasting, bimodal bad state after large
    spikes (2 x N* or more) and noise-triggered transitions near the knee. Short client timeouts help
    recovery (they shed sessions), so the retry story is false. A static cap clears the state
    (Denning, CONCUR). Open but thin: an online trigger from the bimodal signal, and K3 (one cap for
    mixes with N* from 8 to 48).
  - Why closed: the mechanism (a working-set cap) is published; the retry premise is false in both
    simulators. What would reopen it: evidence that a static cap fails on real mixed traffic (K3), or a
    measured GPU trace that shows the trapped state at realistic burst sizes.
Lesson: in 2026, direct mechanism transfers into LLM infrastructure appear as preprints within weeks.
The remaining gaps are second-order (the stated limits of the newest systems) or need simulator
evidence.

### 5.16 I-82 to I-94 (route search 3, R-17)

Origin: 2026-10-08. The user chose option (b) of D-33. Method: a frontier map of the 2025 and 2026
programs of OSDI, NSDI, SOSP, EuroSys, ASPLOS and MLSys and of the newest preprints, then
first-principles candidates in three rounds. Round 1: six candidates attacked by a Sonnet reviewer.
Round 2: five main-session probes. Round 3: a second Sonnet agent generated and self-screened 28
candidates in training, collectives, numerics, agent infrastructure outside the server and GPU cluster
operations. Full write-up: `dossier/ROUTES_R-17_2026-10-08.md`. Evidence is SNIPPET unless stated.

- Environments checked and found crowded: GUI-agent screenshot KV reuse (GUI-KV and four more), full-
  duplex speech serving (DuplexCadence), DSA KV offload (HiSparse, ECHO at OSDI 2026), diffusion LLM
  serving (BlockServe, BiCache), HBF and SSD KV tiers (2609.39131, 2608.11668, 2609.07175), wide-EP
  fault tolerance (SGLang Elastic EP), multi-agent concurrent edits (STORM, CoAgent), looped LMs,
  test-time-training serving (RW-TTT), world-model serving (2607.10389).
- **I-82 Flash write filter for the KV SSD tier.** KILL. FACT: 2609.07175 (LRU-K admission), 2609.39131
  (buffered admission), 2609.28870 (an SSD tier is mostly unnecessary; median block lifetime 219 s).
  SNIPPET: SGLang HiCache `write_through_selective`. My 190 TB/day estimate was about 30% high for the
  FP8 layout of V3.2 (about 48 KB per token).
- **I-83 Rank-safe block-max pruning for the DSA indexer.** WEAK, GPU only. Idea: per key block, store
  bounds; skip a block when its bound is under the current k-th score, so the top-k is exactly the
  original. FACT: no direct prior work found; GVR (2604.22312, NVIDIA) gives bit-exact top-k 1.88x
  faster with a warm start but still scores every key; the indexer weights are signed (the bound needs
  a sign split). INTERPRETATION: LayerNorm, RoPE and a Hadamard rotation make block bounds loose; a
  synthetic test pruned 0% of blocks. Proposed premise check: `dossier/ROUTES_R-17_2026-10-08.md`
  Section 6.3 (needs a GPU to dump real keys).
  R-35 (2026-10-20): LiteTopK (2607.11976, GLM 5.2 on B200) reports that indexer scores concentrate in a
  narrow range (FACT, abstract); that makes block bounds loose (INTERPRETATION). HPC-Ops Top-K
  (2609.08450) and LiteTopK make exact top-k selection cheaper but still score every key, so they do not
  take I-83's block-skipping idea. vLLM v0.32.0 has a pluggable `sparse_indexer_topk_backend`. No small
  open model with a DSA indexer was found, so no Mac test is possible.
- **I-84 Scale-aware ABFT for block-scaled GEMMs.** KILL. Dequantized values are exact in FP32, so it is
  the BF16 case. V-ABFT (2602.08043), FP-Sketch (2609.19758).
- **I-85 Mixed-criticality scheduling.** KILL. The HI budget is max_tokens; priority plus preemption is
  the baseline.
- **I-86 Speculative world-model frames.** KILL. 2607.10389: co-batching B = 4 gives 1.16x, so each
  branch costs nearly a full frame.
- **I-87 Hazard-rate tool KV retention.** KILL. Continuum (ICLR 2026) uses a per-tool empirical CDF;
  2609.18849 has tools report progress.
- **I-88 Imprecise computation for reasoning.** KILL. LASER (2606.31580), Pascal (2602.11530).
- **I-89 Expert-aware batching.** KILL. ELDR (2607.00466), Sem-MoE (ICLR 2026), 2604.23150.
- **I-90 SDC checks in pipeline bubbles.** KILL. SpareTrain (ICLR 2026), TrainSDC (2608.30769).
- **I-91 KV surgery after reasoning strip.** KILL. Same as I-44 (PatchKV, KVEraser). The waste is
  real: vLLM #39321, SGLang #22373, claude-code #63930.
- **I-92 Typing-triggered pre-warm.** KILL. CacheScout, 2608.14624; a keep-alive ping is the baseline.
- **I-93 Checkpoint-aware preemption victims.** KILL. FACT: Volcano issue #5998 and PR #5999.
- **I-94 Phi-accrual collective hang detection.** WEAK spare. No public collective-timing trace.
- 26 other round 3 candidates were killed by the agent's searches; the list with the killing source
  for each is in `dossier/R-17_ROUND3_SONNET_CANDIDATES_2026-10-08.md`.
Lesson: three route searches (R-10, R-13, R-17) found almost nothing. A direct transfer of a well-known
mechanism into LLM serving, agent infrastructure or training systems appears as a preprint, an issue or
a pull request within weeks in 2026. The one open gap (I-83) is second-order and needs real model
internals that the Mac cannot produce.

### 5.17 I-95 to I-102 (orphan-finding scan, R-18)

Origin: 2026-10-08. The user asked whether reading reports of measured waste can replace running
systems ("instead of running ourselves, can we read papers...") and approved the scan (D-36). Method:
about 600 GitHub issue titles in 15 ML-infrastructure repositories sorted by reactions; 31 quantified
findings from 30 characterization papers with a fix search for each (Sonnet agent); a main-session
screen; a Sonnet attack of the five best rows. Full write-up: `dossier/ORPHANS_R-18_2026-10-08.md`.

- **I-95 Hybrid-model snapshot placement.** KILL. FACT: llama.cpp #22746 users measured a 31.8% miss
  rate and 20.5 s wasted prefill per turn; a fix cut it to 1.9%. Cause: a snapshot position defect, PR
  #26499. The research part is I-15 (killed 2026-09-17).
- **I-96 Parallelism-composition numerics.** KILL as a route. FACT: Megatron #7452 lists 13 silent bugs
  found by `spmd_types` (Meta). Context for I-21 and I-23. High scoop risk.
- **I-97 Harness overhead.** KILL. FACT (reviewer): p50 0.13 s; the 1.11 s mean comes from a 10 s p99
  that is probably human approval. Persistent shells and allowlists ship.
- **I-98 Sandbox fleets.** KILL for the memory mechanism (DSec 2609.22978 ships virtio-pmem with DAX,
  balloon free-page reporting, overcommit). WEAK for a tool-phase keep-alive policy; a reviewer will
  call it tuning. Mac-testable on TraceLab gaps if ever needed.
- **I-99 C/R cold start.** KILL. FACT: NVIDIA, "Fast Recovery for LLM Serving via Decoupled Device
  Memory Lifetime in Dynamo" (2609.25451, 2026-09-21): replica recovery under 7 s, 13 to 29 times
  faster than a warm restart. Its abstract does not say whether it uses CRIU. Modal, Dynamo Snapshot
  and others ship snapshots.
- **I-100 Training restart.** KILL. HyperPod checkpointless training, Gemini, REFT, Orbax, Guard,
  ByteRobust. Two source numbers were misread by the harvest (a configured bandwidth, 12 retry chains).
- **I-101 Idle SM harvest.** KILL. The 6% median SM is mostly small CTR and BERT jobs; Nexus, Bullet,
  DuetServe, MuxFlow and others cover SM sharing.
- **I-102 Byte-budgeted decode batches.** KILL. STAR (2510.13668), Medha (2409.17264), PLA-Serve
  (MLSys 2026), slack-based packing.
Lesson: issue trackers show bugs and regressions, not research gaps. Characterization papers sit close
to their fixes, and their headline numbers can mislead (three of five checked numbers were bent: a mean
hiding a 0.13 s median, a configured value read as a peak, a training-node correlation read as an
inference result). Use them to size a waste, not to find the idea.

### 5.18 I-103 to I-109 (assumption removal, R-19)

Origin: 2026-10-08 to 2026-10-11. The user proposed the method ("Mainframes assume one administrator;
the Internet removes that... Lets do that for AI infra as well") and approved it (D-38). Method: 40
assumptions listed from first principles; 7 kept (breaking now, and removal breaks something concrete);
prior-art checks (main session for P1, two Sonnet agents for P2 to P7); a Sonnet attack of P2, P3, P4.
Write-ups: `dossier/ASSUMPTIONS_R-19_2026-10-11.md`, `dossier/R-19_PRIORART_P2-P4_2026-10-11.md`,
`dossier/R-19_PRIORART_P5-P7_2026-10-11.md`, `dossier/REVIEW_R-19_SHORTLIST_2026-10-11.md`.

- **I-103 AI-generated kernels.** KILL as new idea. Offline verification is crowded: contract-grade
  verifier 2608.12700 (39.5% of 2,638 accepted kernels broken, SNIPPET), Kernel Contracts 2604.22032,
  KernelBench-Verified, ProofWright, Volta. FlashInfer-Bench (2601.00227) injects agent kernels into
  SGLang and vLLM. Deployment-time checking is I-23's question; this strengthens I-23's motivation.
- **I-104 Demand response for training.** KILL. Emerald (2507.00909, FACT from the PDF) compares DVFS,
  pausing and reallocation; PowerSlider covers inference.
- **I-105 Agents as cluster tenants.** KILL. Reviewer analysis of 12 public autoresearch logs: exact
  duplicates 0 to 4%; schedules tied to time progress make early loss a poor predictor; ALPS (ICML
  2026) and 2603.15916.
- **I-106 Version-aware serving.** WEAK. Evidence that the assumption breaks for long sessions is thin:
  Cursor Tab ships checkpoints every 1.5 to 2 hours (FACT), Composer about every 5 hours (reviewer);
  most agent APIs resend history, so an update costs one prefix-cache miss (about 4% of prefill tokens,
  reviewer estimate). vLLM `keep` mode continues on stale KV (SNIPPET). Mechanism pieces exist
  (draining, DeltaZip, PSRL version-tagged KV, PipelineRL stale KV). KVShareArena (2609.10266, FACT)
  tests cross-version KV reuse, but only for distant pairs (base to post-trained, distilled), with no
  per-layer drift and no layer selection from weight deltas. Narrowed question: after an RL update, how
  far does the KV of an existing context drift per layer, and can the weight delta alone predict which
  layers need recomputation? Data: `allenai/OLMo-2-0425-1B-RLVR1`, 13 step revisions (200 to 2600,
  FACT), Apache-2.0, about 2.97 GB each. Reviewer's estimate: about 40% chance to pass the Mac gate.
  Research log, 2026-10-17, R-21: ReKV and the Mooncake RL RFC strengthen the adjacent prior art.
  The exact adjacent-step, per-layer delta predictor was not found. The workload premise remains weak.
  Research log, 2026-10-19, R-30: KVShareArena v2 (FACT, HTML) uses six distant pairs (a sibling
  post-trained from the same base, or the base model). It has no adjacent steps and no per-layer
  boundary. CacheReforge (2609.30884, FACT, abstract and HTML) represents stale KV as layerwise
  mixed-version objects with per-layer adapter anchors, calibrated sensitivity, accumulated adapter
  drift and restart boundaries. It reports 92.4% lower KL than stale reuse while recomputing 5.44% of
  layers, on Qwen2.5 1.5B and 7B with continual LoRA updates. It does not test full-weight updates.
  Verdict history: 2026-10-11 R-19 WEAK. 2026-10-19 R-30 KILL.
  Why closed: the mechanism (per-layer version stamps, a drift score from weight changes, a bounded
  recompute window) is published for LoRA. The residual question (does it transfer to full-weight RL
  updates) is an evaluation of an existing system, which D-29 excludes. A negative drift result would
  mean that no mechanism is needed.
  What would reopen it: evidence that full-weight RL updates break CacheReforge in a way that needs a
  new mechanism, for example drift spread over all layers with no usable restart boundary, together
  with a workload that updates weights during live sessions.
- **I-107 Machine-consumer decode mode.** KILL. 2606.20577 drops streaming SLOs for agent traffic;
  forced spans are SGLang jump-forward decoding.
- **I-108 Confidential-GPU trade-off.** WEAK. PipeLLM (ASPLOS 2025), 2606.23969 and 2608.26575 measure
  CC overheads; the joint isolation, reuse and offload trade-off is open, but a test needs a CC GPU.
- **I-109 Server-side session state.** KILL. Same as I-45 and I-50.
Lesson: assumption removal generated fresher questions than paper mining, but in 2026 each removed
assumption already has early work. The surviving question (I-106) is narrow and measurable on the Mac.

### 5.19 I-110 to I-119 (seed-topic gap search, R-20)

Origin: 2026-10-16. The user named three seed topics and asked for repeated literature checks until gaps
appear: "Threads vs process in some areas where if low traffic, we use processes, if high traffic, we
spin up threads?", "KV cache poisoning?", "Noisy neighbor problem for GPU workloads?". Method: three
search rounds (main session), then a Sonnet attack (R-20). Write-up: `dossier/GAPS_R-20_2026-10-16.md`.

- **I-110 Load-adaptive GPU sharing mode.** KILL. The user's rule is already a common pattern: dLoRA
  (OSDI 2024) merges and unmerges adapters by load; Aegaeon (SOSP 2025) pools low-traffic models; D-STACK
  (2304.13541) and GSLICE swap MPS settings with a standby process; green contexts (2504.14489) and LithOS
  (2504.15465) change SM or TPC partitions at runtime; Flying Serving (2602.22593) and ReMP (2606.18741)
  switch DP and TP online. 2605.26461 gives MPS fault isolation and 355 ms recovery. INTERPRETATION:
  isolation needs come from trust and fault domains and do not fall when load rises.
- **I-111 Process versus thread for the Python front end.** KILL, engineering. vLLM separates the API
  server and EngineCore with ZMQ and has an in-process mode; free-threaded 3.14t support is a dependency
  list (vLLM #28762, SGLang #22889).
- **I-112 Prefix-cache pollution.** KILL. PrefixShield (2608.01657, vLLM, one-touch pollution), CacheCast,
  2608.14561, 2609.18112.
- **I-113 Integrity of shared external KV stores.** KILL as research. HMAC manifest (2606.28958),
  checksum at scheduling time (2604.17249), NDSS 2026 salts; LMCache issues #5507 and #3301 (SNIPPET).
- **I-114 Verifier for position-independent KV reuse.** KILL by a simple baseline; the reviewer votes WEAK.
  HijackKV (2607.19957, FACT from HTML) proposes "verify before reuse" and does not evaluate it.
  Baseline: re-encode each chunk alone before it enters a shared cache, so no attacker prefix enters the
  stored KV; the cost is one chunk-only prefill per unique chunk (1/R per use for reuse factor R), off the
  critical path. Reopen if a real deployment must publish chunk KV captured from user requests with R near 1.
- **I-115 Eviction-steering attack.** KILL. The benign version is published (ACL 2026 "The Pitfalls of KV
  Cache Compression", 2605.18053); pinning instruction tokens defeats the adversarial version.
- **I-116 KV-tier I/O QoS.** WEAK. No per-tenant reservation or limit scheduling for KV fetch bandwidth was
  found; CacheCast, MTDS, ShadowServe (2509.16857) and EVICPRESS (2512.14946) are adjacent. Premise not
  measured.
- **I-117 Classic GPU noisy neighbor.** KILL. Same as I-10, plus 2508.20274, 2604.22430, 2601.07600,
  Hummingbird (2601.04071), Roomie (2607.16784), LithOS.
- **I-118 Power-coupled noisy neighbor on superchips.** MEASURE-FIRST, low confidence.
  Problem: on GH200 and GB200 the GPU gets the module limit minus Grace power (NVIDIA Grace tuning guide,
  FACT), and the CPU has priority (2605.01938 Section II-A, FACT; HPE CUG 2024, SNIPPET). JUPITER runs
  GH200 at about 680 W with a 100 W default CPU cap (FACT). CPU-side work next to the GPU (agent tool
  sandboxes, tokenization, CPU KV offload, CPU attention offload) can take power from GPU prefill.
  Exact gap: no public measurement of this slowdown for LLM serving, and no phase-aware CPU-GPU power split
  for serving on superchips was found. Not new: steering, power capping, phase-aware GPU clocks (2505.21758,
  2609.11133, GreenLLM 2508.16449, POLCA 2308.12908, Harmonia ISCA 2015).
  Research question: under a binding module limit, does CPU-side work lower GPU prefill throughput by a
  meaningful amount, and does a phase-aware CPU cap beat the best static CPU cap?
  Reviewer's rejection: "a vendor power-steering knob that a static CPU cap already handles". Reviewer
  predicts an oracle gain under 5% over a static cap; likely output is a workshop measurement note.
  Next step and kill rule: E-30 (Section 6.1). First ask the provider about root and telemetry.
  R-35 (2026-10-20): E-30 is frozen as a package (`experiments/superchip_power_coupling/`: preflight,
  sweep driver with a dry-run mode, analysis with unit tests). KILL 2 now has a defined objective (equal
  weight of normalized GPU throughput and normalized CPU work; a heuristic). Lambda's MIG guide shows
  `sudo nvidia-smi` on a 1 x GH200 on-demand instance (SNIPPET), so root is likely.
  R-34 (2026-10-20): the pre-run review found three blockers. The prefill test hit the prefix cache, the
  hwmon glob missed the `device/` layout, and the caps ran at the default limit. It also showed that
  the equal-weight KILL 2 objective decides KILL 2 by construction. Amendment A1 fixed all four; KILL 2
  now uses a constrained form (best GPU term with CPU work at least 80% of uncapped). Writable Grace
  caps need the kernel argument `acpi_power_meter.force_cap_on=y`. A cloud VM probably cannot set it,
  so KILL 2 may be untestable there.
  Research log, 2026-10-17, R-21: NVIDIA Dynamic Power Software confirms a runtime steering
  controller. No LLM-specific CPU noisy-neighbor measurement was found. Static CPU capping remains strong.
- **I-119 Contention as a side-channel defense.** KILL. 2609.06853 measures the effect; salts (KVGov
  2608.09225) and selective sharing (SafeKV 2508.08438) already remove the channel.
Verdict history: 2026-10-16 R-20 as listed above.
Lesson: the one survivor came from a hardware change (a shared power budget), not from a paper. The
reviewer reported a GB10 forum anecdote that the main session could not find; it is not cited.

### 5.20 I-120 and I-121 (current-frontier review, R-21)

### I-120 Atomic runtime activation for remote KV recovery

Status: KILL as research (R-33, 2026-10-20). Before: MEASURE-FIRST, high scoop risk. Confidence in gap: medium at the runtime boundary,
low for a full paper. Aliases: transactional KV activation, fail-closed KV recovery.

Origin: 2026-10-17, R-21 current-frontier review.

Problem: A remote store can return correct objects while the runtime exposes a partial or stale
logical cache entry. Shared blocks and concurrent recomputation can then produce wrong output.

Research question: Can provisional private blocks and an epoch-fenced page-table install preserve
recompute-equivalent output under connector faults, with less than 3% overhead?

What prior art already answers:

- Mooncake supplies object consistency, committed metadata updates, and transfer failover.
- Mooncake RFC 3692 proposes invisible uploads, manifests, coverage checks, and activation steps.
- Fail-Closed Lowering defines claim identity and claim-scoped failure outcomes.
- LMCache and vLLM implement remote KV movement and recovery paths.

Exact surviving gap: I did not find an implemented runtime protocol that makes shared-block activation
atomic with request liveness, ownership, and scheduler admission. Do not claim storage transactions as new.

Experiments: none. The cheapest test reproduces vLLM issue 49250 and injects faults around lookup,
transfer, activation, abort, sharing, and recomputation.

Reviews: R-21.

Research log:

- 2026-10-17: Read Mooncake, RFC 3692, Fail-Closed Lowering, LMCache, and current failure issues.
- 2026-10-17: Found deterministic wrong output, partial-retrieve crashes, and long recovery waits.
- 2026-10-20 (R-33): vLLM PR 53298 (open, updated 2026-10-16) says it fixes issue 49250. It orders
  repair of shared KV blocks, blocks prefix-cache hits on invalid blocks, and transfers repair ownership.
  PR 62282 (merged 2026-10-20) fixes a multi-peer write-after-free. PR 61578 adds KV checksums.

- 2026-10-20 (R-35, this session, supports R-33): vLLM issues 61768 (open), 61905 and 61912 (opened
  2026-10-15) and PR 61769 were found by scheduler-level CPU probes with a mock connector (an external
  "inference-correctness-lab"), which is the local test that I-120 planned. Their fix directions
  (allocation identity in failure reports, no prefix hits on failed blocks until repaired) are the I-120
  mechanism. The maintainers' RFC 35780 (closed as stale) and PR 42461 (open) propose request-level
  recovery: reset to 0 and recompute, with async-loaded blocks never hashed. That is the simple baseline.
  Issue 49250 is still open; its fix PR 49252 closed unmerged; PRs 53298, 59793 and 45146 are open.

Verdict history:

- 2026-10-17, R-21: MEASURE-FIRST. The direct failure evidence is strong. RFC 3692 creates very high scoop risk.
- 2026-10-20, R-33: KILL as research. PR 53298 claims the same scope as the surviving gap. When it merges,
  the kill rule "current main passes the matrix" will hold. What would reopen it: PR 53298 closes unmerged,
  and a fault class outside its claims still gives wrong output on main.

Next step and kill rule: Reproduce issue 49250 on current main. Kill the paper route if current main
passes the matrix. Also kill it if correct fresh-block recomputation costs less than 3% at realistic faults.

Lessons: Treat a remote KV hit as a state transition. Object integrity does not prove runtime-state safety.

### I-121 Verification-aware scheduling for coding-agent fleets

Status: KILL. Confidence in gap: low. Aliases: verification-budget scheduler.

Origin: 2026-10-17, R-21 current-frontier review.

Problem: Coding agents increase patch production and review work. Verification resources can become scarce.

Research question: Can a scheduler allocate tests across candidate patches to improve accepted correct patches
under a fixed verification budget?

What prior art already answers: TestGRAD allocates iterative tests across four candidate patches.
PatchDiff generates distinguishing tests. Production CI systems already select and prioritize tests.

Exact surviving gap: None was found. The proposed scheduler is a direct combination of coding-agent ensembles
and established CI test prioritization. The evidence does not show that test compute is the binding resource.

Experiments: none.

Reviews: R-21.

Research log:

- 2026-10-17: Read the Chen and Stratton study, the ICSE SWE-bench study, TestGRAD, and Microsoft test selection.

Verdict history:

- 2026-10-17, R-21: KILL by exact-work, adjacent-domain, and combination attacks.

Why closed / what would reopen it: Reopen only if a public trace shows a new coupling between agent generation
and scarce verification hardware that TestGRAD and CI selection do not model.

Lessons: A measured bottleneck does not imply a new scheduling mechanism.

### 5.21 I-122 to I-130 (continued frontier review, R-22)

### I-122 Versioned commit for mutable real-time sessions

Status: PIVOT to I-131. Confidence in the original public gap: low after R-24.

Origin: 2026-10-17, R-22 continued frontier review.

Problem: A streaming update can change token history while queued work, block hashes, materialized KV,
parser state, and emitted output still refer to an older history.

Research question: Can generation-tagged immutable session versions and a token-boundary publish remove
wrong-state interleavings, with less than 2% CPU overhead at 1,000 sessions?

What prior art already answers: vLLM implements streaming requests, cache invalidation, preemption,
and async scheduling. Transactional KV work separates stable and tentative cache state.

Exact surviving gap at R-22: I did not find a public protocol that commits one generation across logical
token history, cache identity, materialized state, queued work, parser state, and emitted output.
R-24 found no evidence that this global protocol is necessary. Narrow local fixes cover the closed failures.

Experiments: E-31 is INVALID DESIGN and did not run. R-23 found that it encoded the expected result,
lacked liveness, misrepresented source failures, and fixed its cost result through model constants.

Reviews: R-22, R-23, and R-24.

Research log:

- 2026-10-17: Read the vLLM streaming design and issues 49449, 58776, 36755, and 42655.
- 2026-10-17: Searched commit-frontier, epoch, generation, session-version, and transaction terms.
- 2026-10-17: Predeclared E-31. A static audit added an epoch-only baseline and out-of-order checks.
- 2026-10-17: Did not run E-31. Docker Desktop has an invalid signature. Sonnet is signed out.
- 2026-10-17: Docker later served Linux containers through `desktop-linux`. Sonnet remains signed out.
- 2026-10-17: Claude authenticated in the user Terminal. Automation cannot read its Keychain entry.
  Added a manual, read-only Sonnet review wrapper. E-31 remains unrun.
- 2026-10-17: R-23 completed the required Sonnet review. It found four blockers. E-31 did not run.
- 2026-10-17: R-24 pinned vLLM main at `da9134bc89364465f68aa679564435936d157a14`.
  It checked six issues, six fixes, five source paths, and the current scheduler test surface in Docker.
- 2026-10-17: The selected seven scheduler and streaming test files contain 185 test functions,
  54 parameterization markers, and no Hypothesis or random-action use.

Verdict history:

- 2026-10-17, R-22: MEASURE-FIRST. The failures are direct. The cross-component invariant is not public.
- 2026-10-17, R-23: E-31 INVALID DESIGN. The experiment cannot falsify the mechanism.
- 2026-10-17, R-24: PIVOT. The problem class is real, but the global version mechanism lacks evidence.

Why closed / what would reopen it: Reopen the global version mechanism only if a real scheduler test
shows that narrow local invariants cannot prevent a cross-component failure.

Lessons: A request identifier is not a version identifier after live mutation starts.
Absence of a version field is not evidence that a version field is necessary.

### I-123 Commit-indexed recovery for heterogeneous real-time session state

Status: PIVOT to I-132. Confidence in the original public gap: low. Scoop risk: high.

Origin: 2026-10-17, R-22 continued frontier review.

Problem: Real-time speech and world models keep opaque state that tokens cannot reconstruct. A GPU
snapshot alone does not identify a safe external input and output cut.

Research question: Can periodic complete-state snapshots, an input log, and an output commit index recover
a session within 200 ms, with less than 10% overhead and no duplicate committed events?

What prior art already answers: Execution-State Capsules captures heterogeneous model state on one device.
DéjàVu recovers KV state. vLLM-Omni RFCs define streaming state and name migration and recovery as later work.

Exact surviving gap: I did not find a public system that joins heterogeneous model-state snapshots with
an external input log and an acknowledged output frontier for exact real-time recovery.

Experiments: none. R-28 rejected the planned CPU oracle before predeclaration or execution.

Reviews: R-22. The D-29 rubber-duck and different-model reviews are still required before a run.

Research log:

- 2026-10-17: Read vLLM-Omni RFCs 5212 and 6227, Execution-State Capsules, DéjàVu, and WorldMove.
- 2026-10-17: Searched consistent-cut, output-commit, replay, stream-recovery, and checkpoint terms.
- 2026-10-19: R-28 read current vLLM-Omni duplex documentation, RFC 8429, PR 8485, Concordia,
  GhostServe, Flink fault tolerance, and Resume Means Resume.
- 2026-10-19: WorldMove values make the old 200 ms replay oracle invalid for its world-model case.

Verdict history:

- 2026-10-17, R-22: MEASURE-FIRST. The cross-layer cut appears open, but public roadmaps name recovery.
- 2026-10-19, R-28: PIVOT. Current systems expose every component separately. The old oracle is
  circular or under-specified. I-132 keeps only the crash-consistent cross-layer recovery cut.

Why closed: The proposed periodic snapshot and replay oracle does not test a defensible open mechanism.

What would reopen it: A measured speech or world-model trace must expose a recovery trade-off that
full per-unit protection and replay do not solve.

Lessons: A state snapshot is not a session checkpoint without input and output progress.

### I-124 Authoritative commit frontier for speculative and async inference

Status: KILL.

Origin: 2026-10-17, R-22. Problem: Several inference stages can hold tentative tokens and KV state.

Research question: Can one commit frontier make speculation, pipeline parallelism, and async output safe?

What prior art already answers: LLM-42 implements verify and rollback. ONNX Runtime GenAI buffers
speculative output. Transactional KV caching separates committed and speculative KV state.

Exact surviving gap: None as a standalone thesis. I-122 can use this invariant.

Experiments: none. Verdict: KILL by exact-work and different-name attacks.

Why closed: Disable async execution or buffer output until verification commits. These are strong baselines.

### I-125 Elastic slot batching for masked diffusion LLMs

Status: KILL.

Origin: 2026-10-17, R-22. Problem: Masked diffusion requests finish at different denoising steps.

Research question: Can elastic slot replacement improve utilization over fixed synchronous batches?

What prior art already answers: vLLM-Omni's `StepScheduler` advances requests independently. A direct
masked-diffusion serving paper proposes slot batching.

Exact surviving gap: None for the core mechanism. Remaining work is policy tuning inside an existing system.

Experiments: none. Verdict: KILL by exact-work and scoop-risk attacks.

Why closed: Reopen only if a distinct scheduling constraint survives comparison with `StepScheduler`.

### I-126 Cancellation-debt scheduling for agent branch races

Status: KILL.

Origin: 2026-10-17, R-22. Problem: Losing speculative branches can retain remote KV and queued work.

Research question: Can a scheduler account for delayed cancellation cost when it admits agent branches?

What prior art already answers: vLLM NIXL uses renewable leases. The SGLang roadmap includes
generation-tagged abort acknowledgements. TAPER controls branch admission.

Exact surviving gap: No production trace shows that cancellation debt is a dominant remaining cost.

Experiments: none. Verdict: KILL by relevance, exact-work, and adjacent-work attacks.

Why closed: Reopen only if a cancellation-heavy trace shows a policy gap after lease and abort fixes.

### I-127 Online isolation for AI-generated GPU kernels

Status: KILL.

Origin: 2026-10-17, R-22. Problem: Generated kernels can corrupt memory or hang a shared worker.

Research question: Can online isolation qualify generated kernels without process-level cost?

What prior art already answers: M2K finds CUDA memory bugs with model-aware symbolic execution.
KREX runs kernels concurrently. CUDA process contexts provide an ordinary fault boundary.

Exact surviving gap: None that is more than integration around these mechanisms.

Experiments: none. Verdict: KILL by exact-work, adjacent-domain, and combination attacks.

Why closed: Keep this problem as motivation for I-23 kernel qualification.

### I-128 Topology-independent data-loader recovery

Status: KILL.

Origin: 2026-10-17, R-22. Problem: Elastic restart can change the worker topology during an epoch.

Research question: Can data progress resume deterministically across a changed worker count?

What prior art already answers: Mosaic Streaming supports deterministic mid-epoch resumption across
24, 16, and 48 GPUs when the global batch size remains constant and divisible.

Exact surviving gap: None for the proposed mechanism.

Experiments: none. Verdict: KILL by exact-work attack.

Why closed: Reopen only for a new constraint that Mosaic Streaming does not cover.

### I-129 Adaptive recurrent-state checkpoint placement

Status: KILL.

Origin: 2026-10-17, R-22. Problem: Hybrid models need useful recurrent-state anchors for reuse and recovery.

Research question: Can an adaptive policy place recurrent-state checkpoints better than fixed intervals?

What prior art already answers: Sparse Prefix Caching, Tail-Replay, SuffixReplay, HYPIC, DASC, and
vLLM RFC 52959 cover placement, replay, reuse, and internal Mamba checkpoints.

Exact surviving gap: None for the proposed route.

Experiments: none. Verdict: KILL by exact-work and combination attacks. This confirms I-95.

Why closed: Reopen only for a measured constraint that these systems omit.

### I-130 Compiler and CUDA-graph warm state in training checkpoints

Status: KILL.

Origin: 2026-10-17, R-22. Problem: Restart can lose compiler artifacts and captured execution graphs.

Research question: Should training checkpoints include portable compiler and CUDA-graph warm state?

What prior art already answers: PyTorch Mega-Cache exports portable compiler artifacts. Foundry
materializes CUDA graph contexts and reports a startup reduction from 10 minutes to 3.9 seconds.

Exact surviving gap: The training-specific cost premise is not measured. Existing mechanisms cover the route.

Experiments: none. Verdict: KILL by exact-work and baseline attacks.

Why closed: Reopen only after a real training restart breakdown shows a large unhandled cost.

### 5.22 I-131 Stateful metamorphic testing for mutable inference sessions

Status: KILL as thesis. Optional tool-only salvage. Confidence in the public gap: low.

Origin: 2026-10-17, R-23 and R-24 pivot from I-122.

Problem: Streaming updates, preemption, cache reset, speculative output, and queue movement can
interact. Example tests can cover each feature but miss short transition sequences between them.

Research question: Can stateful transition generation with history-equivalence oracles find more
real mutable-session faults than current tests and equal-budget random actions?

What prior art already answers:

- Stateful property testing, model-based testing, and trace-guided generator synthesis are established.
- A TOSEM 2026 study classifies 929 inference-engine bugs and recommends diverse oracles and
  metamorphic tests.
- Current vLLM has broad scheduler and streaming tests with many parameter grids.

Exact surviving gap: I did not find a public tester that generates mutable LLM session transitions
and checks history equivalence across scheduling, preemption, reset, and continuation paths.
Generic property testing applied to vLLM is not a contribution by itself.

Experiments: E-32 did not run. E-33 was cancelled before execution.

Reviews: R-23 recommends a real scheduler test. R-24 defines the pivot. R-25 and R-26 block E-32.
R-27 kills the thesis route after the paper audit.

Research log:

- 2026-10-17: Read the current vLLM issue and fix paths for session truncation, async handoff,
  speculative stop trimming, preempted stop, and parked-session reset.
- 2026-10-17: Audited vLLM main at `da9134bc89364465f68aa679564435936d157a14` in Docker.
- 2026-10-17: Counted 185 tests and 54 parameter grids in seven selected files. Found no
  Hypothesis or random-action use in those files.
- 2026-10-17: Read the TOSEM inference-engine bug study and targeted adjacent testing work.
- 2026-10-19: R-25 found an unstable corpus, circular bug encoding, weak baselines, and incomplete oracles.
- 2026-10-19: Froze four merged parent-and-fix pairs for E-33. Excluded two unmerged fixes.
- 2026-10-19: R-26 found that the amended design still lacked circularity controls and oracle contracts.
- 2026-10-19: R-27 mapped trace lengths and ran full-history blame on the four parent commits.

Verdict history:

- 2026-10-17, R-24: MEASURE-FIRST. Direct failures support the problem. The contribution has a
  high combination risk and must beat current tests plus random actions.
- 2026-10-19, R-25: MEASURE-FIRST. Amend E-32 before implementation. Run E-33 first.
- 2026-10-19, R-26: AMEND BEFORE IMPLEMENTATION. Do not run E-32 or E-33 as written.
- 2026-10-19, R-27: KILL as thesis. The corpus is circular and does not show recurrence after fixes.

Why closed: The known corpus comes from one feature rollout and does not show recurrence after fixes.
The proposed contribution remains generic stateful testing plus vLLM-specific actions.

What would reopen it: A prospective holdout must find new confirmed bugs after the harness freeze.
The tool must beat bounded enumeration, Hypothesis, and equal-budget random actions.

Lessons: A cluster of bugs in one new feature does not prove recurrence after local fixes.

### 5.23 I-132 Crash-consistent duplex recovery epoch

Status: KILL at design review. Confidence in the public gap: low.

Origin: 2026-10-19, R-28 pivot from I-123.

Problem: A valid model-state restore can still duplicate audio, omit accepted input, or restore
conversation history that disagrees with output the user already consumed.

Research question: Can one durable recovery epoch join complete model state, accepted input,
durable output, client receipt, and playback progress after a fail-stop engine crash?

What prior art already answers:

- Execution-State Capsules captures complete heterogeneous model state at a committed boundary.
- Concordia logs committed GPU-state deltas and restores from the last complete record.
- vLLM-Omni owns input, response, playback, history, lease, and model state in one engine session.
- vLLM-Omni journals events for reconnect and its open PR 8485 adds ordered per-input acknowledgements.
- Flink coordinates replayable input, operator state, and transactional or idempotent output.

Former proposed gap: R-28 did not find a public system that persists one recovery epoch across the
heterogeneous model snapshot and the full-duplex protocol frontiers after engine loss.
R-29 found that the proposed mechanism does not establish a new systems constraint.

Do not claim exactly-once network delivery. The client must deduplicate stable event identifiers.

Experiments: E-34 is KILL-DESIGN. It was not implemented or run.

Reviews: R-28 and R-29.

Research log:

- 2026-10-19: Completed the two-pass R-28 scoop and adjacent-domain audit.
- 2026-10-19: Designed a Docker fail-stop gate with correct replay and full-copy baselines.
- 2026-10-19: The rubber-duck review left novelty and combination risk unresolved.
- 2026-10-19: R-29 returned `KILL_IDEA`. It found standard checkpoint composition,
  incomplete baselines, undefined workload targets, and no nondeterministic replay state.

Verdict history:

- 2026-10-19, R-28: DESIGN-REVIEW. Keep only if an independent review finds a new cross-layer
  invariant beyond standard stream checkpointing plus a model snapshot.
- 2026-10-19, R-29: KILL. The review found no new invariant in E-34. The Docker design
  fixes policy costs by construction and cannot answer the important GPU replay question.

Why closed: The proposed epoch is a write-ahead log, a periodic snapshot, an atomic manifest,
and an idempotent sink. The two baselines are the endpoint checkpoint intervals of that policy.
The design therefore met its predeclared novelty kill condition.

What would reopen it: Show that replay from the same real GPU snapshot produces different visible
events. Alternatively, show a real trace where no checkpoint interval, delta snapshot, or hot replica
meets a sourced recovery budget.

Lessons: A model checkpoint is not a session checkpoint. A known distributed-systems pattern is not
new because one state component lives on a GPU.

---

### 5.24 I-133 to I-140 (fresh frontier pass, R-33)

Origin for all: 2026-10-20, R-33. Full write-up: `dossier/FRONTIER_R-33_2026-10-20.md`.
Each idea below is KILL. Experiments: none. Reviews: R-33.

**I-133 Elastic membership for wide expert-parallel serving.** Status: KILL. Confidence in gap: low.
Problem: one failed rank stops a wide-EP instance with fixed membership.
What prior art already answers: EEP (2605.10670) repairs only invalidated state and restores 95% of
throughput in 52 s versus 348 s for a full restart. SGLang Elastic EP, vLLM Elastic EP and Ray Serve LLM
ship the same direction.
Why closed: exact work and shipping systems. What would reopen it: a failure class EEP does not cover
(a slow rank, not a dead rank) with a trace.

**I-134 Context-aligned environment recovery for agentic RL.** Status: KILL. Confidence in gap: low.
What prior art already answers: Belayer (2608.14635) checkpoints container state aligned with LLM
context and reuses shadow workers (up to 42x faster worker recovery). RobustRL and DSec cover role
isolation and rollout lifetime.
Why closed: exact work. What would reopen it: none planned.

**I-135 Multi-GPU transparent C/R with fast transports on.** Status: KILL. Confidence in gap: low.
Problem: cuda-checkpoint cannot restore memory imported from another process (NCCL P2P, CUDA IPC, NVLS).
What prior art already answers: NVIDIA nvcf issue 2299 and PR 2300 (2026-10-06) release and re-create
these mappings at the same virtual addresses and run a multi-process quiesce, checkpoint, remap, resume
sequence. Issue 1183 measured the transports-off path on 8x H100. Singularity (2022) used a device proxy
and a barrier. CRIUgpu is upstream in CRIU 4.0. GCR (FAST 2026) is faster than cuda-checkpoint.
Why closed: the vendor is building the exact mechanism. What would reopen it: a correctness failure that
the remap approach cannot fix (for example in-flight collectives at the cut), shown on real hardware.
Lessons: the user's C/R background is a strong fit, but the platform vendors own this space now.

**I-136 Semantics-aware GPU snapshots.** Status: KILL. Confidence in gap: low.
What prior art already answers: Dynamo Snapshot releases KV before capture (190 GiB to 6 GiB on one
B200 example). GPU Memory Service keeps weights outside the CRIU image. Dynamo issue 16357 tracks
per-engine gaps. Why closed: engineering, shipping. What would reopen it: none planned.

**I-137 Cross-vendor KV and numerics portability.** Status: KILL. Confidence in gap: low.
What prior art already answers: 2606.29986 converts KV formats and precision across vendors and overlaps
the transfer by layer. Why closed: exact work.

**I-138 Rate-limit-aware scheduling for agent fleets.** Status: KILL as research. Confidence in gap: low.
Problem: providers may reserve input plus max_tokens against a tokens-per-minute budget.
Why closed: no mechanism with a clear invariant; the gain depends on undocumented provider accounting;
measurement needs paid API traffic; gateways and output-length prediction already exist.

**I-139 Local tensor parallelism over Thunderbolt 5 RDMA Mac meshes.** Status: KILL. Confidence: low.
Why closed: the user has one Thunderbolt 4 laptop, so the test is not feasible. EXO and MLX JACCL ship
RDMA tensor parallelism.

**I-140 Reward verification as the agentic RL straggler.** Status: KILL as mechanism. Confidence: low.
Evidence: SandMLE reports about 196 s per code execution on MLE-bench. No source measures the share of
RL step time. Why closed: async RL overlaps reward with generation, GKE Agent Sandbox and DSec attack
sandbox start time, and a measurement alone is a benchmarking paper (D-29).

Verdict history (all): 2026-10-20, R-33: KILL.

---

### 5.25 I-141 to I-148 (limitations mining, R-36)

Origin for all: 2026-10-20, R-36. Full write-up: `dossier/LIMITATIONS_R-36_2026-10-20.md`.
Experiments: none. Reviews: R-36. Each idea comes from a gap that a reviewed paper names as future work.

**I-141 Crash-time KV reuse when device memory survives.** Status: KILL. Confidence in gap: low.
Sources of the gap: Dynamo 2609.25451 (request-state preservation is future work), Belayer 2608.14635
(flush-free KV recovery), LUMEN 2606.17787 (one KV checkpoint per request).
Why closed: paged KV is append-only, so a committed-length and block-table journal written after each
step's CUDA event makes reuse safe. Torn writes land beyond the committed length. In-place recurrent
state is covered by block-boundary state checkpoints and bounded replay. Dynamo argues capacity loss,
not request progress, dominates. AnchorTP already keeps KV in a daemon across crashes.
What would reopen it: a real-hardware trace of torn writes inside committed blocks (for example from
asynchronous offload or compaction kernels), plus re-prefill as a large share of recovery time.
Lessons: three papers naming the same future work does not make it research. Apply the baseline attack.

**I-142 Shapley split of shared-prefix cost.** Status: KILL. Confidence in gap: low.
Sources of the gap: DLPM 2501.14312, PrefixShield 2608.01657.
Why closed: the public repo srujan20/prefix-cost-attribution (2026-08-25) computes exact Shapley over the
prefix trie and shows the conflict under recomputation. Tree cost games have a classical closed form.

**I-143 TP-invariant determinism for block-scaled FP8.** Status: WEAK. Confidence in gap: low.
Source of the gap: TBIK 2511.17826 (quantized dtypes are its stated next step).
Why weak: extension of a published method (D-29), needs 2 or more GPUs, determinism is crowded.
Next step and kill rule: none planned. Consider only if a second GPU joins the I-23 rental; KILL if a
public FP8 TP-invariant kernel appears first.

**I-144 Aging-aware rejuvenation.** Status: KILL. Source: Software Aging 2606.11916.
Why closed: Dynamo shadow engines (7.3 s) and SGLang warm switches make periodic restart nearly free.

**I-145 Branch-scoped KV after agent rollback.** Status: KILL. Source: Aborted but Not Forgotten 2608.15939.
Why closed: excluding or salting aborted-branch blocks is engineering; copy-on-write branch caches exist.

**I-146 Integrity of high-fan-out shared KV blocks.** Status: KILL as thesis. Source: 2604.17249.
Why closed: datacenter HBM has ECC, so the threat needs non-ECC GPUs; vLLM PR 61578 adds KV checksums.

**I-147 Erasure-coded KV for decode and P/D.** Status: KILL. Source: GhostServe 2605.00831.
Why closed: extension of GhostServe; LUMEN and FailSafe cover recovery; needs a cluster.

**I-148 KV tiers when host memory is not larger than HBM.** Status: KILL. Source: HiSparse 2608.07009.
Why closed: NVIDIA ICMS owns the tier stack; needs GB200 or GB300 hardware.

Verdict history (all): 2026-10-20, R-36: as listed above.

### 5.26 I-149 to I-164 (algorithmic-optimization sweep, R-37)

Origin for all: 2026-10-21, R-37 (D-54). Full write-up with code facts and sources:
`dossier/ALGORITHMIC_R-37_2026-10-21.md`. Experiments: E-37. Reviews: R-37, R-38.
Lenses: full rescans where an incremental update is enough; greedy where an exact algorithm is cheap;
online policies where the future is known; theory checks of whether the decision matters.

**Closed by current source code (the engine already does it).** I-153 (SGLang `ep_dispatch_algorithm`
`lp` with `lplb_solver.py`), I-157 (vLLM NIXL `tp_mapping.py` spreads MLA source ranks), I-159 (vLLM GPU
runner keeps `output_bin_counts`), I-160 (PyTorch DCP read planning is a sweep line), I-151 (vLLM capture
sizes step 8 then 16, so padding is at most 15 slots; small oracle).
**Closed by 2026 preprints or RFCs.** I-149 (PEEK 2607.02525, Requests of a Feather 2605.06046), I-150
(vLLM RFC 47582, TokTier 2607.29678, Incremental BPE 2605.30813), I-152 (TensorRT-LLM ADP balance,
NanoCP 2605.21100), I-162 (NorMuon, Motif 2, INTELLECT-3, MatrixFSDP 2607.05895, FlexShard session),
I-164 (2606.21238, Continuum), I-155 (SGLang DFS theorem, BatchLLM, BlendServe), I-156 (Tenplex).
**Closed by theory.** I-154: balanced pipeline stages form a proportionate flow shop, where every
permutation has makespan sum(p_j) + (m - 1) x max(p_j); for large offline EPD batches, any order without
idle time approaches max(sum p1, sum p2). I-158: a multicast group delivers to every member, so
group-limited routing on 9-GPU groups raises ingress from 8 to 36 receives for each token.
Lessons: the algorithmic lens finds real slow paths (I-149, I-150 and I-161 were found in code before the
search), but upstream engineers or 2026 preprints reach them within weeks. Check the newest code path
first: vLLM fixed I-159 in a new runner while the old path still has the quadratic cost.

**I-161 Kept eviction heap for radix caches.** Status: KILL as research; engineering PR candidate.
FACT: SGLang a3d00eff `evict()` builds a heap of all evictable leaves on each call. E-37 S1: 4.6 ms at
E = 50,000 and 11.2 ms at E = 100,000 on the M4; a kept heap pops 4 leaves in about 2 microseconds.
Next step (optional, not thesis work): an upstream PR with a lazy-deletion heap, plus an SGLang benchmark
under memory pressure to show how often `evict()` runs.

**I-163 Critical-path list scheduling for decode megakernels.** Status: WEAK, MEASURE-FIRST, GPU for the
claim. Confidence in gap: low. Scoop risk: high.
Problem: a decode megakernel runs the task graph of one step on all SMs. The graph repeats for each batch
size, so an offline schedule is possible. MPK uses round-robin (its author lists priority-aware policies as
future work); Cohere North Mini deals waves round-robin with work stealing and reports that wave order
changes performance.
What prior art answers: Event Tensor (2604.13327, MLSys 2026) static and dynamic scheduling
transformations; ForgeMegakernel (2609.12379) per-SM instruction streams at 50.5 to 85.9% bandwidth
utilization; Weave (2609.21483); Fleet (2604.15379). No public work found applies HEFT or critical-path
priority to SM task order in a decode megakernel (three searches, 2026-10-21).
Research question: on one Hopper GPU, does a static HEFT schedule built from measured task times cut
decode step latency by at least 5% at batch 1 to 64, compared with round-robin wave order plus work
stealing?
Next step and kill rule: dump one MPK decode-step task graph with measured task times on one H100 (can
ride on an E-36 rental), replay it in an offline simulator with a shared-bandwidth limit, and compare
round-robin, waves with stealing, HEFT and the lower bound. The simulator must take task times from the
measurement, not from the schedule. KILL if HEFT is within 3% of waves with stealing at every batch size.

Verdict history (all): 2026-10-21, R-37: as listed above.
- 2026-10-21, R-38 (Sonnet, verified by the main session): I-161 KILL. SGLang PRs 38371 and 38454 (open since
  2026-09-07 and 2026-09-08) add persistent lazy eviction heaps; PR 41690 amortizes eviction. The default
  SGLang path is now the Rust tree core (PR 39627), which also rebuilds the heap on each eviction start.
- 2026-10-21, R-38: I-163 KILL. HazyResearch/Megakernels `scheduler.py` has a list scheduler and a critical-path
  priority function (its call is commented out). The Cohere blog (2026-09-08) reports about 10% from a
  cost-based greedy scheduler and from random-schedule search on a dense model, and dropped both for MoE.
  The planned replay test was circular: measured task times already contain round-robin's contention.
- 2026-10-21, R-38: I-154 stays KILL with a corrected reason. Random order exceeds Johnson's rule by 3 to 14% at
  n of 8 to 128 (reviewer simulation), so scale alone does not kill it. The objective does: online serving
  optimizes latency, where SJF and SRPT apply. The O(sqrt n) bound holds only for random order.
- 2026-10-21, R-38: I-158 stays KILL with a corrected reason. The 4.5 times ingress holds for static whole-group
  multicast only. DySHARP (2026-05-07, search-only, no arXiv ID found) uses dynamic multimem addressing.
- 2026-10-21, R-38: no new candidate survives (near misses: ski-rental KV TTL, close to Continuum 2511.02230;
  tool-definition ordering, engineering).

---

## 6. Experiments and reviews

### 6.1 Experiment registry

Old experiments predate this protocol, so "predeclared criteria" is filled only where the
artifact recorded them. Every experiment before E-22 ran on the Apple M4 MacBook Air unless
stated.

**E-01 Synthetic cutover microbenchmark** (Phase 1, I-06). Question: can pre-copy remove
stable state from the critical path? Setup: single-device MPS, synthetic mutable state, no
real migration. Result: at 64 MB mutable state, stop-and-copy pause about 81.56 ms vs
pre-copy final pause about 21.51 ms. Final pause grew from about 0.60 ms at 1 MB to about
134.09 ms at 256 MB. Validity: SCREEN ONLY. Shows the mechanism behaves as expected and
that real state size is decisive. Artifacts: `idea1_mutable_cutover*.csv` (not in repo).

**E-02 Falcon-H1 hybrid state anatomy** (I-06). Question: how do attention KV and recurrent
state scale on a real hybrid model? Setup: `tiiuae/Falcon-H1-0.5B-Instruct`, Apple MPS, 36
layers. Result:

| Context | Attention KV | Mutable recurrent | Host round trip (recurrent) | Token latency |
|---|---|---|---|---|
| 64 | 1.125 MB | 27.492 MB | 38.58 ms | 2587.59 ms (warm-up) |
| 256 | 4.50 MB | 27.492 MB | 36.46 ms | 63.90 ms |
| 1024 | 18.00 MB | 27.492 MB | 53.40 ms | 149.09 ms |

Validity: structural result valid (KV grows with context, recurrent state stays fixed).
Timings are not representative of H100 or B200. Artifacts: `idea1_real_hybrid.csv`,
`idea1_cache_tensors.json` (not in repo).

**E-03 Restore correctness, replay, bandwidth projection** (I-06). Setup: same model,
contexts 256 and 1024. Result: restored continuation matched for 8 steps at both. Measured
host round trip 171.48 ms and 74.21 ms. One-step replay was 1.16x and 1.21x slower than
copying the latest state. Four-step replay 7.15x and 4.98x slower. Sixteen-step replay 28.32x
and 18.94x slower. Max logit and recurrent-state differences were zero for the replay cases
tested. Ideal transfer of 27.49 MB: 2.148 ms at 12.5 GB/s, 1.074 ms at 25, 0.537 ms at 50,
0.268 ms at 100, 0.134 ms at 200. Sixty-four such sessions total about 1.72 GiB, ideal 137 ms
at 12.5 GB/s, 34 ms at 50, 8.6 ms at 200. Validity: projections are ideal analytical, not
NVLink or RDMA measurements, and exclude protocol, synchronization, verification, incast and
topology-conversion costs. Timings differ from E-02 because of run variance.
Artifacts: `idea1_cutover_correctness.csv`, `idea1_replay_catchup.csv`,
`idea1_bandwidth_projection.csv` (not in repo).

**E-04 V1 synthetic GPU-efficiency suite** (Phase 1, five initial ideas). Result: mostly
hard-coded and synthetic. Validity: **WITHDRAWN.** The user challenged validity and all
GO and KILL claims from it were withdrawn. Never cite as evidence.

**E-05 V2 and V2.1 real Mac screens** (I-07, I-08, I-09, I-10, I-11). Setup: real MPS,
socket and device operations, Apple M4 16 GiB, macOS 26.5.2, PyTorch 2.14.0, CUDA
unavailable. Five screens: activation residency, network crossover, elastic break-even,
sharing interference, hybrid allocator. Results and caveats are in Section 5.3. Validity: use
cautiously. The network test structurally favored bulk sync, and the elastic break-even's
absolute times were artifacts. Artifacts: `AI_Infra_Thesis_Handoff_Package/gpu_efficiency_v2_results_bundle.zip`.

**E-06 V2.2 targeted suite** (I-08, I-09, I-11). Setup: corrected network test (15 shaped
conditions), Alibaba 2026 spot-GPU trace analysis, typed hybrid allocator model. Results in
Section 5.3. Validity: the network crossover is real but small (median winner advantage about
2.9%). Trace windows are an upper bound. Allocator savings are modeled. Artifacts:
`gpu_efficiency_v2_2_results_bundle.zip`.

**E-07 vLLM hybrid-state gap harness** (I-13). Question: is a typed, heterogeneous state
runtime already supported upstream? Setup: source and runtime probes against vLLM commit
`e0705731be04aac90838f318fcdae15c56a53080`. Result: see the construction table in I-13. It
killed the broad thesis. Validity: probes VALID. The upstream pytest step is INVALID
(missing `tblib`, return code 4). Artifacts: `vllm_hybrid_state_gap_results_bundle.zip`.

**E-08 Reconstructability POC** (I-14). Setup: patch adding `KVStateReuseMode` and
`find_reconstructable_cache_plan()` to the same commit. Result: cases in I-14, 5 POC tests and
architecture-neutrality scan pass. Validity: proves planning-layer expressibility only.
Artifacts: `vllm_reconstructability_poc_results_bundle.zip`.

**E-09 Minimum-cost planner sensitivity** (I-15). Setup: 6 availability scenarios by 54
normalized cost profiles, 324 rows, compared common-boundary, never-load, always-load and
per-state min-cost. Result: statistics in I-15. Validity: normalized costs chosen by us, not
GPU timings. Shows mathematical possibility only. Artifacts:
`consistent_state_planner_poc_results_bundle.zip`.

**E-10 Qwen3.5 Mac cached-replay experiment** (I-16 context). Setup: Qwen3.5-0.8B-Base,
MPS, FP16, contexts 256 to 2048, valid boundaries at 50%, 75% and 87.5%. Result: time saved
51.7 to 78.7% at 256, 52.9 to 86.9% at 512, 50.4 to 87.4% at 1024, 50.7 to 86.9% at 2048.
Overall median 74.58%. Full-attention cache grew 8x (about 3.15 MiB to 25.17 MiB) while
recurrent cache stayed about 19.76 MiB (1.10 MiB per layer). Correctness passed. Validity:
structural result valid. The timing mostly proves "less suffix work costs less", a known
caching principle, so it is not evidence for selective recovery. Recurrent kernels were slow
reference code. Artifacts: `mac_hybrid_selective_recovery_results_bundle.zip`.

**E-11 State-selective recovery v1.1** (I-16, I-17). Setup: context 512, boundary 384,
suffix 128, two reps. Result: normal replay 171.72 ms. Frozen full-KV replay 206.66 ms,
frozen plus K/V bypass 218.12 ms (both wrong: max logit diff about 1.055). Oracle with cached
post-block outputs 155.36 ms (9.53% faster, correct in this one case). Validity: **INVALID
for all frozen-KV timings** (`semantic_recovery_correct = false`, mask geometry broken). The
9.53% oracle is one case, not decisive. Artifacts:
`mac_state_selective_recovery_results_bundle.zip`.

**E-12 v1.2 explicit-mask fix** (I-16). Question: does the divergent-state replay become
numerically exact? Setup: explicit 4-D additive causal mask for full-attention layers and a
2-D suffix padding mask for linear-attention layers. Result: `semantic_recovery_correct =
true`, max logit diff 0.0 at context 512 (was 1.0554 in v1.1), 0.0117 floor across the sweep,
frozen full-attention KV unchanged, naive skip control incorrect as intended. Validity:
VALID. Artifacts: `mac_state_selective_recovery_fix_v1_2.zip` (checksum OK).

**E-13 Execution-order bias check** (I-16). Finding: v1.1 and v1.2 always ran methods in the
order normal, frozen, bypass, oracle, charging warm-up to the baseline. Randomizing the order
(context 512, boundary 384, 2 reps): frozen +8.91% became -1.65%, bypass +9.46% became
+0.74%, oracle +18.57% became +10.82%. Validity: low rep count, a direction not a value.
Consequence: every fixed-order timing before v1.3 is suspect.

**E-14 Analytical FLOP and MAC ceiling** (I-16). Setup: real model module shapes,
`flop_model.py`, boundary 0.75. Result, ceiling on replay saving:

| Context | Skip all FA blocks | Sound frozen-K/V variant |
|---|---|---|
| 512 | 23.1% | 1.22% |
| 2,048 | 28.0% | 1.14% |
| 8,192 | 41.9% | 0.92% |
| 32,768 | 67.2% | 0.52% |
| 131,072 | 88.1% | 0.18% |

Validity: VALID, hardware independent. The two curves move in opposite directions: long
context makes attention dominate replay, but the sound variant only saves K/V projections.

**E-15 Memory-budget Pareto, evenly spaced checkpoints** (I-15, I-17). Result: activations
first win at budget 151 MiB (context 2,048), 302 MiB (8,192), 603 MiB (32,768), 2,412 MiB
(131,072). Validity: weak baseline, superseded by E-19. Do not cite as the result.

**E-16 Cross-model state-ratio analysis** (I-15). Setup: HF configs, fp32 recurrent state,
`crossmodel_state_ratio.py`. Result in I-15 (5.3x spread in snapshot cost). Validity: computed
from config files, not measured serving behavior.

**E-17 v1.3 randomized-order sweep** (I-16, I-17).
Predeclared heuristic gates (from `results/summary.json`): promising if correct, frozen KV
unchanged and median K/V-bypass speedup at least 10%. Activation pivot if K/V-only is under
10% but the oracle full-block skip is at least 25%. Weak if both are small.
Setup: contexts 512 to 4096, boundaries 0.5, 0.75, 0.875, 10 reps, order randomized within
every rep, 12 scenarios, zero failures. Result: sweep-wide medians frozen 0.07%, bypass
0.67%, oracle 14.04%. Per-context oracle 10.77%, 12.24%, 15.52%, 19.69%. Max logit diff
0.0117, about 90x below the 1.055 of the unsound plan. Screen signal:
`weak_incremental_selective_recovery_signal_deprioritize`. The activation-pivot gate (oracle
at least 25%) was not met. Validity: VALID. Artifacts:
`experiments/hybrid_state_recovery_v1_3/`, `dossier/bundles/hybrid_state_recovery_v1_3_bundle.zip`.

**E-18 RQ1 gate: plan-level soundness against real vLLM on CPU** (I-19). Results in I-19.
Gate pass. Validity: VALID. Covers the planning and coordinator layer, not execution.
Artifacts: `experiments/rq1_soundness_gate/`, `dossier/bundles/rq1_soundness_gate_bundle.zip`.

**E-19 RQ3 optimal checkpoint placement, 3 real models** (I-15). Results in I-15. Validity:
analytical MAC model with measured byte constants. The four divergence distributions are
synthetic sensitivity probes, not a production trace. Artifacts: `experiments/rq3_state_budget/`.

**E-20 RQ2 NCCL 2.31.2 source audit** (I-18, 2026-09-16 23:06 file time). Results in I-18.
Validity: VALID. Reproducible with a clone. Artifacts: `experiments/rq2_nccl_source_audit/`.

**E-21 Layout survey and suffix-skip verification** (I-19, I-20). Results in I-20 and I-19.
Validity: HF configs only, no weights. Achievable saving is an upper bound. Artifacts:
`experiments/rq3b_layout_survey/`, `dossier/bundles/rq3b_layout_survey_bundle.zip`.

**E-22 NCCL re-audit on 2.32.3** (I-18, 2026-09-23). Question: does the source gap still
hold on current master? Setup: `git clone --depth 50`, master at merge
`12df1a11af` of v2.32.3-1 (2026-09-17), `audit_nccl_ce_selection.py`. Result: same four
findings. CE gate now at `enqueue.cc:3398`. Tuner still `tuner_v6`. No occupancy or
utilization terms in `src/`. New: `ncclCollConfig_t` gives per-call `CTAPolicy`, `minCTAs`,
`maxCTAs`, `nvlsCTAs`, `algSelection`. Validity: VALID. The clone lives in the session scratchpad,
not in the repo. Output saved as `experiments/rq2_nccl_source_audit/results/nccl_ce_selection_audit_2.32.3.json`
(the 2.31.2 output is `nccl_ce_selection_audit.json` in the same folder).

**E-23 SafeDispatch Mac pilot harness** (I-23, 2026-09-24). Status: PREDECLARED on
2026-09-24 before any run. Result section is filled in after the run. Failed attempts are
kept, never overwritten.
Question: does the Test A harness work before any GPU time is spent? This is a harness
validation, not a research result. Do not cite its numbers as evidence about GPU kernels.
Setup: Apple M4 MacBook, torch 2.14.0. Model `Qwen/Qwen3-0.6B` (dense, same family as
Integer Alibi). 32 prompts of about 64 tokens taken from this repo's markdown files by a
fixed seed, 96 greedy tokens each, batch size 1, exact greedy decoding. Reference backend:
CPU fp32. Candidate backend: MPS fp16 (fallback CPU bf16, see C1). Code:
`experiments/safedispatch_oracle/`. Contract levels from Section 5.8: L1 bitwise and
spacing distance, L2 greedy flips with margins, L3 KL on logits.
Predeclared criteria (all heuristics, fixed before running):
- **C1 self-consistency, no false positives.** Run each backend twice on identical inputs in
  decode mode. PASS if logits are bitwise identical across the two runs at every position for
  both backends. If the MPS candidate fails, that is itself a finding (MPS is not run-to-run
  deterministic): switch the candidate to CPU bf16 and re-test C1 for it. If the CPU
  reference fails, stop and report.
- **C2 fault injection.** Emulated INT8 GEMM on CPU, M=256, K=2048, N=2048, 20 seeds,
  per-tensor activation scale, per-channel weight scales, exact integer accumulation done in
  float64, reference epilogue `bf16(acc * (sa * sw))` with the scale product in fp32. Faults:
  F1 scale application order `(acc * sa) * sw`. F2 truncation instead of round-to-nearest-even
  when casting fp32 to bf16. F3 double rounding through fp16. F4 scale stored in bf16 instead
  of fp32. F0 is the no-fault control. PASS if (a) F0 shows zero differing elements in every
  seed, (b) the bitwise gate reports at least one differing element for each of F1 to F4
  aggregated over the 20 seeds (an F1 with zero differences is reported as "undetectable at
  this element count" and counts as a failure of (b)), and (c) for F1, F2 and F3 the maximum
  distance is at most one bf16 spacing and `allclose(rtol=2^-7, atol=0)` passes them, which
  demonstrates the blind spot reported by arXiv 2609.00363. F4 is reported with no expected
  outcome. A violation of (c)'s spacing bound means the fault was mis-specified and is
  reported as such.
- **C3 identity.** For every prompt, the first position where the candidate's decode-mode
  teacher-forced argmax differs from the reference token equals the first position where the
  candidate's own free-running greedy tokens differ from the reference tokens (or both show no
  divergence in 96 tokens). PASS if 100% of prompts agree. Also report how often the
  prefill-mode teacher-forced first flip agrees, as a measurement of prefill versus decode
  kernel mismatch. No pass condition on that second number.
- **C4 estimator coverage.** Population: every (prompt, position) pair, with the candidate
  evaluated exhaustively. Strata: reference top1 minus top2 logit margin with edges
  [0, 0.05, 0.1, 0.25, 0.5, 1, 2, inf]. Budget: 25% of positions, allocated across strata in
  proportion to `N_b * w_b` with `w_b = 1 / (bin_midpoint + 0.1)` and `n_b <= N_b`, using
  the last bin midpoint 3.0. Estimator: `sum_b (N_b / N) * p_b` with `p_b` drawn from
  `Beta(k_b + 0.5, n_b - k_b + 0.5)` (Jeffreys), 95% interval from the 2.5 and 97.5
  percentiles of 2000 draws. PASS if, over 500 replicate subsamples with different seeds,
  the interval contains the exhaustive flip rate in at least 90% of replicates. If the
  exhaustive data has fewer than 15 flips the result is INCONCLUSIVE: rerun with more prompts
  or the CPU bf16 candidate, and log it as a new attempt. Also report the mean interval width
  against simple random sampling with a Wilson interval at the same n. No pass condition on
  that comparison.
- **C5 report only.** L3 KL statistics and the L1 spacing-distance histogram of logits.
Decision rule: pilot PASS if C1, C2 and C3 pass and C4 passes. If any fails, fix the harness
bug (or record the finding) and rerun as a new attempt.
Attempts:
- **Attempt 1 (2026-09-24), complete. C1, C2, C3 PASS. C4 INCONCLUSIVE.** Candidate MPS fp16,
  32 prompts, 96 tokens, 3,072 positions, about 56 s per prompt. Unit tests 14 of 14 pass.
  - **C1 PASS.** Reference (CPU fp32) and candidate (MPS fp16) were bitwise identical across
    repeated runs on all 32 prompts. Zero non-finite logits.
  - **C2 PASS** (`results/attempt_01/fault_injection.json`). F0 control: zero differing
    elements over 10,485,760. F1: 44 differing (4.2e-6), max 1 spacing. F2: 5,235,300 (49.9%),
    max 1 spacing. F3: 656,356 (6.3%), max 1 spacing. F4: 3,905,007 (37.2%), max 2 spacings.
    `allclose(rtol=2^-7)` passed F1, F2, F3 in 20 of 20 seeds and F4 in 3 of 20.
  - **C3 PASS.** 32 of 32 prompts: the first decode-mode teacher-forced flip equals the first
    free-running divergence. 7 of 32 prompts diverged in free running. The prefill-mode
    first flip agreed with the decode-mode first flip on 29 of 32 prompts (90.6%), a measured
    prefill versus decode kernel mismatch on this pair.
  - **C4 INCONCLUSIVE.** 8 flips in 3,072 positions (0.26%), below the predeclared minimum of
    15. All 8 flips were in the lowest margin bin [0, 0.05), where 8 of 70 positions flipped
    (11.4%). The other 3,002 positions across six higher bins had zero flips. Bin sizes were
    70, 60, 190, 294, 383, 542, 1533.
  - **Observation, not a fix.** The reported stratified coverage of 0.0 is not interpretable.
    Every unsampled stratum had zero flips, so the truth (8/3072) sits exactly on the
    boundary, and the Jeffreys equal-tailed draws for those strata are strictly positive, which
    puts the interval's lower limit just above the truth. Simple random sampling with a Wilson
    interval covered 0.97 of replicates. The estimator was left unchanged for attempt 2, per
    the protocol. If attempt 2 shows the same boundary artifact, record it as a finding and
    amend the interval as a logged attempt 3.
  - C5, reported only: decode-mode flip rate 0.26%, prefill-mode 0.23%, mean KL 7.4e-6.
  - Verdict: fall back as predeclared, attempt 2 with the CPU bf16 candidate.
- **Attempt 2 (2026-09-24), complete. C1 and C3 PASS. C4 FAIL.** Candidate CPU bf16, same 32
  prompts and 96 tokens, estimator unchanged, about 57 s per prompt.
  - **C1 PASS.** Both backends bitwise stable, no non-finite logits.
  - **C3 PASS.** 32 of 32 prompts agree. 18 of 32 prompts diverged in free running. The
    prefill-mode first flip agreed with the decode-mode first flip on 23 of 32 (71.9%).
  - **C4 FAIL.** 33 flips in 3,072 positions (1.07%), a conclusive test. Flips by reference
    margin bin: [0, 0.05) 22 of 70, [0.05, 0.1) 8 of 60, [0.1, 0.25) 3 of 190, and 0 of 2,752
    positions in the four higher bins. Over 500 replicates the predeclared stratified interval
    contained the exhaustive flip rate in **5.8%** of replicates (required at least 90%).
    Simple random sampling with a Wilson interval covered 96.6%. Mean interval width 0.0128
    stratified vs 0.0152 simple random.
  - **Diagnosis (INTERPRETATION from the bin table).** The four highest bins hold 2,752 of
    3,072 positions and have exactly zero flips. The Jeffreys pseudo-counts give each of them
    a small positive rate, and those add up to roughly 0.4 percentage points against a true
    rate of 1.07%, so the interval sits above the truth. The design premise, that high-margin
    positions essentially cannot flip, is what breaks a Beta prior on the rate. Attempt 1's
    coverage of 0.0 was the same effect in its extreme form. The flaw is in the estimator's
    design, not in the harness plumbing, and the simple random baseline is fine.
- **Amendment A1 (2026-09-24, post hoc, written before any code change, after seeing
  attempts 1 and 2).** Replace the rate-based Jeffreys interval with a finite-population
  interval. For each stratum the unsampled positions' flips are drawn from a Beta-Binomial
  predictive with Jeffreys parameters (k_b + 0.5, n_b - k_b + 0.5). Total flips equal the
  sampled flips plus the predictive draws, the rate is the total over N, and the 95% interval
  is the 2.5 and 97.5 percentiles of 4000 draws. Everything else is unchanged: strata,
  allocation weights, 25% budget, 500 replicates, the at-least-90% coverage rule, the
  at-least-15-flips rule. Reason: it counts the sampled positions exactly, keeps zero as a
  possible total for unsampled strata, and reflects that the low-margin strata are mostly or
  fully observed. **Decision rules fixed before running it:**
  1. Re-analysis of the attempt 1 and 2 data with the amended interval, no new model runs,
     reported as development evidence only, because the amendment was designed after seeing
     that data.
  2. **Attempt 3 is the held-out test.** 32 new prompts (`prompts_holdout.json`, paragraphs
     not in `prompts.json`), CPU bf16 candidate, same protocol. The amended interval is the
     one under test. PASS if coverage is at least 90% with at least 15 flips.
  3. Efficiency rule: if coverage passes but the mean width is not below the simple random
     Wilson width, record "stratification is not more efficient than simple random sampling at
     this budget" and use simple random sampling as the Test A default.
- **Development re-analysis of A1** (2026-09-24, `run_reanalysis.py`, results in
  `results/reanalysis_attempts_01_02.json`). No model runs. Development evidence only.
  - **Flip invariant PASS on both attempts.** Every flip has margin at most 2 times the
    maximum absolute logit difference: 0 violations in 6,144 positions. The largest margin
    among flips was 0.027 (attempt 1) and 0.178 (attempt 2), while the median margin is about
    2. This is the strongest support for the idea that risk concentrates at small margins, and
    it doubles as a harness bug detector.
  - Two-sided coverage, A1 versus the original interval: attempt 1 1.000 vs 0.000, attempt 2
    0.940 vs 0.058. Simple random Wilson: 0.970 and 0.966.
  - **A1 is not reliably valid.** On a synthetic test with exactly zero flip probability above
    margin 0.3, both the original and A1 undercover (0.825 and 0.795), and every miss is on the
    lower side, the truth falling below the lower limit, because unsampled zero-flip strata
    contribute phantom flips. The unit test that asserted A1 coverage of at least 0.90 on that
    data was falsified and replaced by one that pins the property that holds, the one-sided
    upper bound.
  - **One-sided 95% upper bound, the quantity a qualification decision uses.** Stratified miss
    rate 0.000 on both attempts and on the synthetic data (conservative). Simple random
    Wilson 0.000 to 0.024. But the stratified bound is looser: mean upper bound over truth is
    1.89 vs 1.78 (attempt 2) and 4.65 vs 2.94 (attempt 1). Decision rule 3 fired: the Test A
    default is simple random sampling with a Wilson upper bound.
- **Attempt 3 (held-out test): not run.** Reasons: A1 is not reliably valid (synthetic 0.795),
  decision rule 3 already fired on development data so the default is simple random
  sampling either way, and the holdout builder found only 9 unused paragraphs under the
  current prose filter, so a fair 32-prompt held-out set needs a wider prompt source. Deferred
  to the GPU phase, where a held-out prompt set is a natural part of the grid. The unused
  holdout code and file were removed.
- **Regression check.** After the estimator gained options, re-aggregating attempts 1 and 2 with
  the default interval reproduces the recorded coverage exactly (0.0 and 0.058). All 15 unit
  tests pass. Harness sources changed after attempts 1 and 2 ran (estimator options, a
  `--prompts` flag, `run_reanalysis.py`). Default behavior is unchanged, and `prompts.json`
  is byte identical.
Result: **harness core validated, estimator not validated.** C1, C2 and C3 PASS. C4 FAIL
(attempt 2, 33 flips, coverage 5.8% against a required 90%). By the predeclared decision rule
the pilot is not a full PASS. What the pilot settled:
1. The bitwise and spacing gates, self-consistency, fault detection and the teacher-forced flip
   identity work, so they can be ported to GPU kernel pairs unchanged.
2. Prefill-mode and decode-mode first flips disagreed on 3 of 32 prompts (fp16 candidate) and 9
   of 32 (bf16 candidate), so decode-mode teacher forcing, which the harness uses, is required.
3. Flips concentrate at small margins (the invariant).
4. For Test A use simple random sampling with a Wilson upper bound. Keep the margin-stratified
   estimator as an experimental option until it is validated on held-out data.
These are harness findings on a 0.6B model on a Mac. They say nothing about GPU kernels.
Artifacts: `experiments/safedispatch_oracle/` (code, prompts, results for attempts 1 and 2, the
re-analysis), `dossier/bundles/safedispatch_pilot_bundle.zip`.
Artifacts (planned): `experiments/safedispatch_oracle/`, results next to it, a zip in
`dossier/bundles/`.

**E-24 Trace study: what breaks the prompt cache, and the oracle ceiling for compress-on-ingest**
(I-41, 2026-10-04). Status: PREDECLARED on 2026-10-04 before any analysis code ran. Result is
filled in after the run.
Question: in a real production harness with prompt caching, (a) how much cache-miss volume does
compaction cause compared with session starts, cache expiry and other breaks, (b) what share of
cost and of appended context comes from tool results, (c) how much would a perfect, free
compressor of tool results save, and how much survives once a small-model compressor is paid
for, and (d) how often does the agent later need a tool result verbatim?
Data: the user's own Claude Code transcripts in `~/.claude/projects/**/*.jsonl` (local,
read-only, never copied into the repo, only aggregates reported). The current session
(`dd630f27-...`) is excluded. Main-thread requests (`isSidechain` false) are the unit;
sidechain (subagent) requests are reported separately. Assistant records are de-duplicated by
`requestId` because one API response is written as several records with repeated usage.
Definitions:
- Prompt size of request t: `P_t = input_tokens + cache_creation_input_tokens + cache_read_input_tokens`.
- A request is an incremental hit if `cache_read_t >= 0.9 * P_{t-1}` (previous main-thread
  request in the same session). Otherwise it is a break, classified in this priority order:
  session start, compaction (a `compact_boundary` record between t-1 and t), model switch, TTL
  expiry (gap above 1 h if t-1 wrote 1 h cache, above 5 min if it wrote 5 min cache), other.
- Cost at list prices from the Claude prompt-caching docs page (checked 2026-10-04): base input
  and output per MTok Opus 5.5 4/20, Opus 5 5/25, Sonnet 5.5 2/10, Sonnet 5 2/10, Haiku 4.5
  1/5. Cache write 1.25x (5 min) or 2x (1 h). Cache read 0.1x, except 0.05x for Opus 5.5.
  Subscription plans are ignored; list price is only a common yardstick.
- Tool-result tokens appended at request t are estimated by character share: tool-result
  characters divided by all new characters since t-1 (tool results, user text, attachments, and
  the previous assistant output counted as 4 characters per output token), times
  `input_tokens + cache_creation_t`. This attribution is a heuristic.
- Oracle counterfactual: every tool result is compressed by ratio r in {2, 4, 10} when it is
  ingested, at zero cost, with no information loss and no extra turns, and stays in the
  context until the next compaction or session end. Saved tokens are re-priced at the rate
  each was actually billed at (write or read). Output tokens are unchanged.
- Realistic counterfactual: only tool results of at least 1,024 estimated tokens are compressed
  (ACON's threshold), at r = 4, by a Haiku 4.5 call. Input is the tool result plus 300 tokens of
  instructions (heuristic), uncached. Output is tokens/4 for an abstractive "structured facts"
  rewrite, or 5% of input for an extractive line-selection answer (heuristic).
- Verbatim dependency: for every main-thread `Edit` or `MultiEdit` whose `old_string` has at
  least 20 characters, search all earlier tool results in the same session for that exact
  string, after stripping Read's line-number prefixes. Record found or not, source tool, lag in
  main-thread requests, and whether a compaction lies in between.
Predeclared criteria (all thresholds are project heuristics, not literature standards):
- **P1, the cache premise in the user's idea.** Share of break-request cache-write tokens caused
  by compaction. Below 10%: the premise "compaction is what shortens cache life" is FALSE for
  this workload. At least 25%: SUPPORTED. In between: WEAK.
- **P2, oracle ceiling.** Saving from the perfect free compressor at r = 4, as a share of total
  list-price cost. Below 15%: KILL I-41 as a cost thesis. 15% to 40%: inside the range already
  published by ACON, AgentDiet and TokenPilot, so no new opportunity (PIVOT). Above 40%: the
  opportunity is real here, continue MEASURE-FIRST.
- **P3, compressor overhead.** If the realistic abstractive net saving is at most half of the
  oracle saving at r = 4, record that the compressor's own cost eats most of the gain.
- **P4, verbatim dependency.** If at least 50% of qualifying `old_string`s are found verbatim in
  earlier tool results, record that abstractive (rewriting) ingest compression of file reads is
  unsafe for coding harnesses unless the raw text is recoverable.
- Report only: cost breakdown by billing class, tool-result share of appended tokens by tool,
  compaction count, the lag distribution for P4, and sidechain statistics.
Validity limits declared in advance: one user, one harness (Claude Code), Anthropic models
only, list prices, a character-share token attribution, and transcripts may not log the
compaction call itself or the system prompt. SCREEN ONLY for any claim beyond this workload.
Run notes (2026-10-04): the first run counted forked and resumed sessions twice (825 request IDs
appear in more than one file, about 5% of cache reads). The script was fixed to count each request
ID once while still using duplicates for per-stream state, and re-run. Only the de-duplicated run
is reported. Transcripts are live files, so a re-run on a later date gives slightly different
totals (10,426 vs 10,431 requests between two runs on the same day).
Result (FACT for this workload; 224 files, 216 main-thread sessions, 10,431 requests, $1,925 at
list price):
- Cost shares: cache reads 74.0%, cache writes 17.3%, output 8.7%, uncached input about 0%. 97.2%
  of requests were incremental hits.
- **P1: FALSE.** Break-request cache writes: TTL expiry 83.4%, session start 8.9%, other 7.2%,
  compaction 0.39%. TTL-expiry rewrites cost 9.4% of total spend, compaction rewrites 0.03%, the
  compaction calls (estimated, not logged as requests) 0.07%. 74 gaps above 60 min, 325 above 5 min.
- Compaction: 3 unique events, at 853,974, 967,845 and 968,680 tokens (1M-context sessions),
  down to 9K to 20K, 76 to 244 s each. Tool-result tokens were 28% to 41% of the context at each.
  Median session peak prompt about 26K tokens; 34 of 216 sessions above 150K, 2 above 900K.
- Tool results: 33.9% of appended tokens (Bash 4.07M, Read 1.06M of 5.62M), 31.8% of cost.
  Tokens-per-character deciles 0.25 to 0.36, consistent with the attribution being sane.
- **P2: PIVOT band.** Oracle saving 15.9% (r = 2), 23.8% (r = 4), 28.6% (r = 10). Ceiling 31.8%.
- **P3: not triggered.** Realistic r = 4 (results of at least 1,024 tokens, 1,496 results, 3.52M
  tokens): main-model saving 15.0%, Haiku compressor 0.43% (abstractive) or 0.25% (extractive),
  net 14.6% or 14.8%, which is 0.61 of the oracle.
- **P4: TRUE.** 684 qualifying edits, 71.5% found verbatim in an earlier tool result (Bash 280,
  Read 209, not found 195, 5 across a compaction). Lag in requests: 42% at most 1, 51% at most 3,
  62% at most 10, top decile at least 144. Lag in tool results: 47% within the last 3, 52% within
  the last 5.
- Post hoc, not predeclared (added after the first run): counting the agent's own earlier Write
  content and Edit `new_string` as a competing source, the most recent verbatim source is a tool
  result for 70.9% of edits, the agent's own text for 20.0%, none for 9.1%. Lag for the
  tool-sourced ones: 47% within the last 3 tool results, 52% within the last 5, top 30% at 24 or
  more. P4's conclusion is unchanged.
- Sidechain: 99 requests, $3.35, attribution unreliable (tokens-per-character up to 2.1). Not used.
Validity and limits: VALID as a measurement of this one user's Claude Code use (SCREEN ONLY beyond
it). TraceLab (2606.30560) and claude-code issue #94177 independently report the same cache-miss
pattern at larger scale, so P1 is a replication, not a new finding.
What it changed: the cache-lifetime motivation of I-41 is dropped; the cost motivation is real but
inside the published range; P4 became the only possibly new measurement (I-43).
Artifacts: `experiments/structurer_ingest_screen/e24_trace_study.py`,
`results/e24_results.json` (aggregates only), `dossier/bundles/structurer_ingest_screen_bundle.zip`.

**E-25 Local structurer screen: does a small query-agnostic compressor keep what the agent later
edits?** (I-41, 2026-10-04). Status: PREDECLARED on 2026-10-04 before any run.
Question: when a small local model rewrites a file-read tool result at ingest, without knowing
the future query, does the exact text the agent later edited survive, and what does it cost in
latency on a laptop?
Setup: Apple M4, torch 2.14.0, MPS. Model `Qwen/Qwen3-0.6B` (the only locally cached chat
model besides a 0.5B Falcon-H1), thinking disabled, greedy decoding. Samples: 30 `Read` results
of at most 8,000 characters that E-24 identifies as the verbatim source of a later `Edit`,
drawn with seed 0. Conditions, each targeting about 25% of the input:
- A, abstractive structurer (the user's design): "rewrite as terse structured facts an agent
  will rely on later", max new tokens = input tokens / 4 + 64.
- B, extractive structurer: the model returns line numbers to keep, at most 25% of lines, and
  those lines are kept verbatim.
- C, rule baseline: first and last lines, 25% of lines in total.
- D, random 25% of lines, seed 0.
Metrics: exact survival of the later `old_string` in the compressed text, line recall (share of
the `old_string`'s non-empty lines present verbatim), achieved character compression, wall time
per sample and decode tokens per second.
Predeclared criteria (heuristics): if A's exact survival is below 30%, record that query-agnostic
abstractive structuring at about 4x destroys the spans a coding agent later edits, on this
model. If B does not beat C on exact survival, record that the small model adds nothing over a
positional rule at this budget. Everything else is report only.
Validity: SCREEN ONLY. A 0.6B model is far weaker than a hosted small model such as Haiku 4.5,
so A and B are lower bounds on compressor quality. The structural point being tested (a
query-agnostic rewrite cannot know which span will be needed) does not depend on model size,
but the measured rates do.
Result (2026-10-04, 30 samples, median input 592 tokens, `results/e25_summary.json`):
- A abstractive structurer: exact survival 1 of 30 (3.3%), mean line recall 9.7%, median kept
  16% of characters, median 3.3 s and p90 9.7 s per sample, about 22 tokens per second.
- B extractive structurer: exact 4 of 30 (13.3%), line recall 29.2%, kept 26%, median 2.7 s.
- C first and last lines: exact 6 of 30 (20.0%), line recall 33.3%, kept 24%.
- D random lines: exact 0 of 30, line recall 27.6%, kept 25%.
- **Predeclared readings:** A below 30%, so query-agnostic abstractive structuring at about 4x
  destroyed the later-edited spans on this model. B did not beat C on exact survival, so the small
  model added nothing over a positional rule at this budget.
Validity and limits: SCREEN ONLY. 0.6B model. The 8,000-character cap made the sample short reads,
most of which a 1,024-token ingest threshold would leave alone. A sample-level detail file with
the model outputs is in the session scratchpad only, because it contains the user's code.
What it changed: confirms the structural point of E-24 P4. Without the future query, no tested
condition kept the edited span more than 20% of the time at a 25% budget.
Artifacts: `experiments/structurer_ingest_screen/e25_structurer_screen.py`,
`results/e25_summary.json`.

**E-26 I-43 check on public traces: how far back is the tool output that edits copy from?**
(I-43, 2026-10-04). Status: PREDECLARED on 2026-10-04, after the data was downloaded and its
message format inspected, before any copy or lag statistic was computed.
Question: on multi-model, multi-harness traces, what share of file-editing actions copy text from a
tool output that is older than the last 3 tool outputs (so keep-last-3 clearing, the Anthropic
`clear_tool_uses` default, would already have removed it)?
Data: Exgentic `DiscoPosse/agent-llm-traces`, SWE-bench subset, 391 traces, harnesses claude_code,
openai_solo, smolagents_code, tool_calling, models Claude Opus 4.5, GPT-5.2, Gemini 3 Pro Preview,
DeepSeek-V3.2, Kimi-K2.5. Shards 11 to 20 (528 MB) in `~/Downloads/agent-llm-traces/data/`, user
approved on 2026-10-04. These are benchmark runs, not day-to-day use. TraceLab could not be used
(no tool text in its public release).
Why the unit changed from I-43's rule: in these traces agents edit through bash and Python (the
claude_code harness made 16 `Edit` calls in 391 traces), so `old_string` alone is too thin.
Definitions (fixed before computing anything):
- Trace: per session, the span with the most input messages plus its output messages, in order.
  Report how many traces have a later span with fewer messages than an earlier one (compaction or
  sub-agents), but do not drop them.
- Tool output: each `tool_call_response` result (smolagents: each "Observation" user message),
  decoded from its repr or JSON wrapper to plain text.
- Edit action: an `Edit` or `MultiEdit` call, or a bash command or smolagents code block whose text
  matches a write pattern: `sed -i`, `perl -pi`, `cat >`, `cat <<`, `tee `, `> /`, `>> `,
  `.write(`, `.replace(`, `open(` with a `'w'` or `"w"` mode, `patch `, `git apply`, `str_replace`.
- Copied line: a line of the action text with at least 20 characters after stripping whitespace
  that occurs as a substring of an earlier tool output or of an earlier action's text. Its source is
  the most recent such occurrence. Only lines whose most recent source is a tool output count.
- Lag of a copied line: number of tool outputs after its source and before the action. An action is
  tool-dependent if it has at least one tool-sourced copied line. Its lag is the maximum over those
  lines (all of them must survive for the action to work without a re-read).
Predeclared criteria (I-43's thresholds, heuristics): M = share of tool-dependent edit actions with
lag at least 3, pooled over all 391 traces. **M under 20%: KILL I-43. M at least 30%: I-43
survives as a measurement note (write it up with a retention-policy simulation). In between:
PIVOT (weak, report only).** If fewer than 100 tool-dependent edit actions are found, INCONCLUSIVE.
Report only: M per harness and per model, the line-level share with lag at least 3, the share of
edit actions that are tool-dependent, lag deciles, and the same generalized metric re-run on the
E-24 transcripts as a cross-check against E-24's `Edit`-only figure.
Validity limits declared in advance: SWE-bench runs are short (median 35 turns) and have no human
idle time; the write-pattern detector is a heuristic and will miss some edits and include some
non-edits; substring matching can credit a line to an unrelated earlier output that happens to
contain it.
Amendment before running (2026-10-04, no statistic computed yet): a smoke test that printed only
decoded samples found two decoder bugs (some tool results are already lists, not strings;
smolagents action JSON needed `raw_decode`). Both were fixed. Added a report-only "strict" variant
that counts only in-place edit forms (`Edit`, `MultiEdit`, `sed -i`, `perl -pi`, `str_replace`,
`git apply`, `patch `, `open(..., 'w')`), because the broad patterns also match test scripts. The
decision still uses the predeclared broad detector.
Result (2026-10-04, 23 s run): 391 SWE-bench traces (15 with non-monotonic message counts, kept),
median 33 actions per trace.
- **Pooled: M = 43.7%, verdict SURVIVES** (at least 30%). 3,154 edit actions, 789 tool-dependent
  (25%). Line-level share with lag at least 3: 39.2%. Max-lag deciles 0, 0, 0, 1, 2, 3, 6, 10, 20.
- Strict variant (report only): 1,265 edit actions, 393 tool-dependent, M = 36.1%.
- By harness: claude_code 39.7% (n = 393; strict 28.4%), smolagents_code 48.0% (n = 302),
  tool_calling 42.6% (n = 68), openai_solo 57.7% (n = 26).
- By model: Claude Opus 4.5 29.1% (n = 172; strict 21.8%), Gemini 3 Pro Preview 37.8% (n = 74),
  Kimi-K2.5 45.0% to 49.4%, DeepSeek-V3.2 49.1% to 55.4%, GPT-5.2 n under 12.
- Keep-last-K curve (report only, added after the first run, no new definitions), share of
  tool-dependent edits whose source K would have cleared, for K = 1, 2, 3, 5, 10, 20: pooled 64, 54,
  44, 35, 21, 10%; Claude Opus 4.5 55, 45, 29, 19, 9, 2%; the user's own Claude Code sessions 75,
  65, 60, 55, 47, 38%.
- Cross-check on the user's own transcripts with this generalized metric: 2,842 edit actions, 1,569
  tool-dependent (55%), M = 60.4% (strict 56.2%), max-lag deciles up to 173. E-24's Edit-only
  figure was 53% outside the last 3, so the two methods agree in direction.
Validity and limits: VALID as a screen of benchmark traces. The pooled verdict passes, but the model
closest to the user's workload (Claude Opus 4.5) sits at 29.1%, inside the 20 to 30% PIVOT band,
and at 21.8% on the strict detector, so the result is model-dependent. SWE-bench runs are short;
the user's real sessions show a much heavier tail (38% of sources still more than 20 tool outputs
back), but that is one user. Multi-user real sessions with tool text are not public (TraceLab
strips it). Default keep-last-3 clearing only fires above a token trigger (100K by default), which
most of these short benchmark runs never reach.
What it changed: I-43 passes its predeclared check as a measurement note, not a thesis. The open
question moved to long real sessions: benchmark traces understate how far back edits reach.
Artifacts: `experiments/structurer_ingest_screen/e26_public_trace_check.py`,
`results/e26_results.json`, bundle refreshed. Data in `~/Downloads/agent-llm-traces/` (outside
the repo, with its own `.venv` holding pyarrow).

**E-27 Noise-shaped V-cache quantization screen** (I-58, 2026-10-07). Status: PREDECLARED on
2026-10-07, before any experiment run. Unit tests and one timing run are allowed before the
review. A different model (Sonnet) reviews this plan and the code before the run (D-29). Any change
after that review is logged here as an amendment before the run.
Question: on real attention of a real model, does noise shaping of a 2-bit V cache (per-channel
error feedback along the token axis) lower the attention-output error and the perplexity enough to
beat a Hadamard-rotated 2-bit V cache, the current baseline?
Rubber duck, before the run:
1. The model makes a V vector for each token. A low-bit cache stores a rounded copy.
2. The attention output is a weighted sum of the V vectors. Rounding errors of many tokens add up.
3. Shaping adds the rounding error of token j-1 to token j before rounding. Then errors of neighboring
   tokens cancel when their attention weights are similar.
4. Expected result: a large gain in heads whose weights are smooth along the tokens, and a loss of
   about 1.4x in error on sharp single-token spikes.
5. What can make it wrong: real heads may put most weight on spikes; the test text may be too
   easy or too short; the emulation may not match a real cache; a bug in the carry or the mask.
Setup:
- Model `Qwen/Qwen3-0.6B` (cached locally), fp32, CPU (MPS fp32 only if it matches CPU in a test).
  28 layers, 16 query heads, 8 KV heads, head dimension 128.
- Texts, all on local disk, no download, 4,096-token windows: prose (this repo's Markdown files),
  code (Python source files of an installed package), and agent transcripts (Exgentic SWE-bench
  traces in `~/Downloads/agent-llm-traces/`). 2 windows for each corpus are the calibration split,
  6 for each corpus are the evaluation split. Selection uses a fixed seed.
- Cache emulation (KIVI-style): for a query at position t, tokens j < 4 (sinks) and j >= t - 128
  (recent window) use full-precision V. Tokens 4 <= j < t - 128 use the quantized V. Each token is
  quantized once, in token order. K stays full precision in all conditions, so only the V effect
  is measured.
- Quantizer: rotate V channels with a normalized block-diagonal Hadamard matrix (block 64). Then use
  per-token asymmetric min-max rounding with a group size of 32 channels. Then rotate back.
- Shaping: per KV head and per channel in the rotated domain, u_j = v_j + c_(j-1),
  q_j = Q(u_j) with the scale computed from u_j, c_j = u_j - q_j. The carry starts at 0 at token 4.
- Conditions: C0 full precision; C1u unrotated 2-bit; C1 rotated 2-bit (baseline); C2 rotated 2-bit,
  shaping on all KV heads; C3 rotated 2-bit, shaping only on gated KV heads; C4 as C3 but the carry
  runs in key-sorted order inside each 128-token block (sort by the projection of the block's keys
  on their first principal direction); C5 rotated 3-bit (one more bit).
- Gating (calibration split only): for each layer and KV head, shape if the layer-local
  attention-output error with shaping is lower than without it. C4 has its own gating the same way.
Metrics:
- Step 1, smoothness. From full-precision runs, for each layer and query head, at every 8th query
  position with at least 256 quantized tokens: G^2 = (sum (w_(j+1) - w_j)^2 + w_(b-1)^2) / sum w_j^2
  over the quantized region [4, b), b = t - 128. Error-energy weight e = (sum w_j^2) x (mean
  rounding-error variance of that head's 2-bit rotated V). Head value G_h is the energy-weighted
  root mean of G^2. S_0.5 is the share of total error energy in heads with G_h < 0.5. Also report the
  predicted error-energy ratio with ideal per-head gating, sum E_h min(G_h^2, 1) / sum E_h.
- M1, layer-local attention-output error (no propagation between layers): relative error
  ||o_c - o_fp|| / ||o_fp|| for each layer and query head at every 16th position in the quantized
  regime, evaluation split. Report the median over layers and heads, and the energy-weighted total.
- M2, perplexity with the emulated cache in all layers (errors propagate). Next-token NLL over
  positions 512 to 4,095 of each evaluation window. Report the mean perplexity and the paired
  difference against C1.
- M3, passkey retrieval: 30 prompts of about 3,500 tokens, a 5-digit passkey at a depth between 10%
  and 90%. A prompt counts as correct if the argmax of every answer token, with teacher forcing,
  equals the passkey token (this is the same as greedy decoding of the answer). M3 is used only if C0
  gets at least 80%.
Predeclared criteria (heuristics; the first four come from the R-10 reviewer). [Superseded by
amendment A1 below. Kept for the record.]
- Step 1 KILL: S_0.5 under 30%. Then step 2 does not run.
- Step 2 KILL if any of: (a) the median M1 of C3 is not at least 1.5x lower than that of C1;
  (b) the mean perplexity gain of C3 over C1 is under 0.05; (c) when M3 is used, C3 gets more than
  one prompt fewer than C1 correct.
- GO if no KILL rule fires and the mean perplexity of C3 is at most 1% above that of C5 (one bit saved
  on V).
- Otherwise PIVOT.
- Report only: C1u, C2, C4, the per-corpus breakdown, and per-layer maps of G.
Checks before the run (unit tests): the quantizer error is at most half a step; the carry stays
bounded; the Hadamard matrix is orthonormal; with no quantization the custom attention reproduces the
model's own logits; the masked-correction formula equals an explicit per-query construction; the G
formula matches a Monte Carlo shaped/plain variance ratio with independent carries.
Validity limits declared in advance: one small model, 4K context, CPU emulation (no kernel and no
speed measurement), V only. SCREEN ONLY for other models. If GO, repeat on a second, larger model.
Artifacts (planned): `experiments/noise_shaped_vcache/` (code, unit tests, aggregate results).
**Pre-run review R-11 (Sonnet, 2026-10-07)** found 3 blockers and 7 major issues. The main session
checked them. The verdict was "do not run yet". A first review attempt stopped at a usage limit with
no output.
**Amendment A1 (2026-10-07, after R-11 and before any experiment run).** These replace the matching
parts above:
1. Step 1 (S_0.5) is a diagnostic only, and step 2 always runs. Reason: G^2 assumes independent
   plain rounding errors. Real V does not satisfy this. Test T12 measured actual/independent error
   energy of 8.4 (layer 0) and 3.5 (layer 14) on a repeated test text. The reviewer measured 1.1 to
   1.3 in most layers and 3.4 to 8.1 in layers 0 and 14 on real windows. G is also reported per KV
   head.
2. M1 is the total layer-local error energy per window and condition, with records for each window.
   Rule (a): A = sqrt(sum E(C3) / sum E(C1)) over the evaluation windows. KILL if A > 1/1.5. The 95%
   CI comes from a bootstrap of windows stratified by corpus (B = 10,000).
3. M2 rules use paired per-window NLL differences in nats, with the same bootstrap:
   - INCONCLUSIVE if mean NLL(C1) - NLL(C0) < 0.01 or its CI lower bound <= 0.
   - KILL (b) if mean NLL(C1) - NLL(C3) < 0.01 or its CI lower bound <= 0.
   - GO if no rule fires and mean NLL(C3) - NLL(C5) <= 0.01.
   - Otherwise PIVOT.
   - Order of the rules: (a), INCONCLUSIVE, (b), GO, PIVOT.
4. M3 is report only. The haystack is natural text (unused prose and code windows) with two decoy
   numbers. C1 vs C3 uses an exact McNemar test.
5. Controls, report only: C1d (per-token random sign dither before the rotation) and C4r (shaping in a
   random order inside each 128-token block, with the C3 gates). The gain counts as a smoothness effect
   only if each control captures less than half of the C3 error-energy gain.
6. Gates: shape a KV head only if shaped/plain calibration error energy is below 0.9. Calibration uses
   4 windows for each corpus (12 in total), and evaluation uses 6 for each corpus (18). Report the
   split-half agreement of the gates.
7. Windows: reject a candidate that has 5% or more 16-gram overlap with the windows already selected.
   The code corpus is torch._inductor and torch._dynamo (less boilerplate than torch.nn). The agent text
   keeps dict-typed tool arguments.
8. The baseline name is "KIVI-style rotated 2-bit V (4 full-precision sinks, 128 full-precision recent
   tokens)". It is not the llama.cpp setup (llama.cpp quantizes every token, and its lowest KV type is
   4-bit). Report-only all-quantized variants with no sinks and no recent window: C1z, C3z, C5z
   (perplexity only).
9. Memory: NLL comes from the hidden states and the lm_head in 512-row chunks in float32.
10. Resume: every result file stores a manifest (SHA-256 of vcache.py, e27_run.py and the texts JSON)
    and the gates. A mismatch stops the run.
11. The decision script `e27_decide.py` is frozen before the run.
12. Data: `~/Downloads/e27_data/e27_texts.json`, SHA-256
    0fc58cc1785ed257286644f9cc72a503ecfc053185180da6f9f6d932f95469c1, built by `prep_texts.py`. The model
    is the post-trained Qwen3-0.6B (not Base), used without a chat template. M2 emulates a quantized
    cache at all positions including the prompt (KIVI prefills in full precision).
Limits added by A1: K stays in full precision. The carry makes the write path a serial scan per head.
No speed claim is made. C4 reads a token quantized before its whole 128-token block has left the
window, so a real block-sorted cache would keep that token in full precision a little longer (C4 is
report only).
**Second pre-run review (Sonnet, part of R-11, 2026-10-07):** no blocker. All 12 A1 items were
verified, including the NLL chunk indexing, the passkey prompts, the orders and dither, the bootstrap
indices, the rule order, and a peak memory of about 4 to 5 GB. It listed 11 minor items.
**Amendment A2 (2026-10-07, before any experiment run):**
1. All-quantized variants C1z, C3z and C5z also read the current token quantized (the reviewer found
   that the token itself was still read in full precision). Unit test T10 covers it.
2. The manifest adds the transformers and tokenizers versions, the model snapshot and a SHA-256 of the
   selected window ids. Result files are written atomically. The decision script also checks the
   passkey file and the corpus labels.
3. Measure mode refuses conditions that change the sink or the recent window.
4. Corrections to wording: passkey prompts have about 3,240 tokens, not about 3,500. Because the needle
   goes in by line, its token depth ranges from about 0.16 to 0.77. Haystack windows are token-disjoint
   from the selected windows but can share up to 9.4% of 16-grams with them. M3 is report only.
5. Agent windows repeat more inside a window (16-gram repeat fraction about 0.19 on average) than prose
   (0.01) and code (0.03). Read the per-corpus A before the pooled A. If the CI of A crosses 1/1.5,
   say so in the write-up.
6. Frozen files (SHA-256): `e27_decide.py` c07c0669b78236276a34ec085b98ffb4d35843747ee9bd23e6772961d45d5d65,
   `vcache.py` b64231cdf80e7fac6f6d78c89ea8411a34142c5082c7815bf622fd41e010e8c3,
   `e27_run.py` 31ec10f715fe0a6c3b922b657f76209763206cc9ec2041799e0bbf778994acce. A compressed copy of the
   texts is in `experiments/noise_shaped_vcache/data/e27_texts.json.gz`.
Run: 2026-10-07, 12:40 to 17:10, Apple M4 CPU, fp32. The manifest in every result file matches the
frozen hashes above. The decision came from the frozen `e27_decide.py` (hash checked before use).
Result (FACT for this model and setup; the verdict waits for the post-run review R-12):
- Gates: 12 of 224 (layer, KV head) pairs (5.4%) passed the gate (shaped/plain calibration error
  below 0.9). Split-half agreement of the gates: 0.96.
- Smoothness diagnostic: S_0.5 = 0 for query heads and for KV heads. The predicted error-energy ratio
  with ideal per-head gating is 0.9999.
- M1, layer-local error energy relative to C1 (18 evaluation windows): C2 (shaping on all heads) 1.712,
  C3 (gated) 1.0000, C4 (key-sorted, gated) 0.9997, C4r (random order) 1.0000, C1d (sign dither) 0.912,
  C1u (no rotation) 1.150, C5 (3-bit) 0.167. A = 0.99997 (CI 0.99997 to 0.99998). Per corpus, A is
  0.99997 to 0.99998.
- M2, mean NLL in nats: C0 2.1208, C1 2.1248, C3 2.1243, C5 2.1212, C2 2.1275, C1d 2.1259, C4 2.1244,
  C4r 2.1240, C1u 2.1269. All-quantized variants: C1z 2.2106, C3z 2.2583, C5z 2.1340.
  - Headroom C1 - C0 = 0.0040 (CI 0.0024 to 0.0055), below 0.01.
  - Gain C1 - C3 = 0.0005 (CI -0.0003 to 0.0011).
  - C1 - C2 = -0.0027 (CI -0.0046 to -0.0009): shaping all heads is worse.
  - All-quantized: C1z - C3z = -0.048 (CI -0.058 to -0.037): shaping is worse.
- M3, passkey: 30 of 30 for C0, C1, C3 and C5. McNemar C1 vs C3: no discordant prompts.
- Attribution: C3's energy gain is 0.005%. The dither control gains 8.8%. No smoothness effect.
- Decision script: rule (a) fires first (A > 1/1.5), so KILL. The INCONCLUSIVE rule and rule (b) also
  fire.
Rubber duck, after the run:
1. Only 5% of heads got lower error with shaping, and the two calibration halves agree, so the
   result is stable.
2. Shaping on all heads made the error 71% worse. That is the predicted failure on spiky attention.
3. No head has smooth attention over the quantized region (S_0.5 = 0). The sinks and the recent
   window, which hold the smooth part, stay in full precision in this cache design.
4. The rotated 2-bit baseline costs only 0.004 nats on this model, so there is little to recover.
5. With no full-precision sinks or window, shaping is worse still (0.048 nats), because sinks are the
   strongest spikes.
Artifacts: `experiments/noise_shaped_vcache/results/` (e27_step1.json, e27_ppl.json,
e27_passkey.json, e27_decision.json, e27_log.txt, e27_stdout.txt).
**Post-run review R-12 (Sonnet, 2026-10-07): KILL CONFIRMED WITH SCOPE LIMITS.**
- FACT: an independent reimplementation (own Hadamard, quantizer, shaped scan and error accumulation)
  on 1,536 real tokens and all 28 layers gave the same plain and shaped error energies as the repo
  code (per-cell relative difference at most 9e-7). No bug.
- FACT: the M1 result does not depend on the region or the position sampling. With all heads shaped,
  shaped/plain error energy is 1.73, 1.89, 1.48 and 1.83 for four region definitions (window 32, recent
  tokens quantized, sinks quantized, both). The best possible per-head gate gives at least 0.9995 in
  every region.
- FACT (layer-local level, a subset of layers, Qwen3-0.6B only): shaping does not win at 16K or 32K
  context (all-head ratio 1.59 and 1.76, best gate 0.00%), for K quantization (1.65 to 1.80, best gate
  at most 0.05%), or with an in-sample oracle token order (1.96).
- FACT: the measured C2 ratio of 1.71 agrees with the smoothness diagnostic (energy-weighted mean G^2
  is 1.78).
- Corrections to the rubber duck above (Section 8.1 row 44): point 5 is wrong. C3z reused gates that
  were calibrated for the windowed region. With gates recalibrated for the all-quantized region, NLL
  changes by +0.004 nats on 3 windows (range -0.010 to +0.017), which is neutral. Point 3's claim that
  the smooth part sits in the sinks and the recent window is also wrong. Quantizing the sinks lowers the
  all-head ratio (1.48), and quantizing the recent tokens raises it (1.89). The recent tokens are
  spikier than the older ones.
- INTERPRETATION: M2 had only 0.004 nats of headroom, so rule (b) could not fire in either direction.
  The kill rests on M1. A real perplexity effect of about 0.0005 to 0.0008 nats is possible (C4r gain CI
  0.00007 to 0.0014), but it is far below the 0.01 threshold.
- Minor: the go_bit flag in the decision file is true but meaningless, because C1 is within 0.004 nats
  of C0. A gate agreement of 0.96 is weak evidence of stability, because two 5% gates agree on about 90%
  of cells by chance.
Scope (R-12): Qwen3-0.6B, fp32 CPU emulation, 4K windows of three corpora, 2-bit V with rotated
per-token groups of 32. Length (16K, 32K) and K were checked only at the layer-local level on a subset
of layers. A larger model or another family was not tested.
Verdict: KILL. Validity: VALID for this model and setup.

**E-28 Trace-driven metastability check for I-80** (I-80, also I-61, 2026-10-07). Status: DONE
2026-10-08 (K1 KILL for the frozen grid; scope limits from R-16). PREDECLARED on 2026-10-07, before any
experiment run. Only unit tests and one run-time check (0.1 s for a short
run, metrics not read) ran before this entry.
Question: R-14 found in a toy simulator that a short load spike can leave an agent serving replica in a
lasting low-goodput state, that retries are not needed for it, and that a static session cap removes it.
Do these three findings hold in an independent simulator with real trace sessions and a shared prefix?
If a lasting collapse exists, does one static session cap tuned once fix it across workload mixes, or
is there room for a new mechanism?
Why this design (rubber duck, short form; full text in `experiments/metastable_admission/README.md`):
- A miss on a 25k-token prompt costs about 10 times the GPU time of a hit in the cost model. More live
  sessions than the KV holds cause misses, misses make turns slower, slower turns keep more sessions
  live (Little's law). That loop can keep the system in a bad state after the spike ends.
- A matched control run (same seed, same rate, no spike) separates a metastable state from plain
  overload. If the control also degrades, the point is overload, not metastability.
- Retries are tested by client configs A (60 s timeout, 2 retries), B (60 s, 0 retries) and C (600 s, 2
  retries; with the default SDK timeout a retry almost never fires).
- The static cap is the simple baseline (Denning working-set control, CONCUR). It is tuned once on mix
  M1 and then applied to all mixes, as an operator would do. The per-point best cap is the oracle.
Setup:
- Simulator `experiments/metastable_admission/sim.py`: one replica, vLLM v1 style continuous batching
  with chunked prefill (8,192-token budget, 256 sequences), LRU prefix cache of idle sessions with tail
  eviction, recompute preemption of the newest request, KV capacity in tokens, roofline step-cost model
  for an 8B GQA model on one H100 (constants in `Params`, not fitted to a measured profile). Clients:
  Poisson session arrivals, closed-loop turns, per-request timeout, abort on timeout (prefilled KV stays
  cached), exponential backoff with jitter. Policy "cap": a gateway admits at most N sessions; a queued
  session is shed after the client timeout.
- Sessions: 459 sessions (about 31 calls each) from the Exgentic traces, `data/session_calls.csv`
  (`extract_sessions.py`). Parallel subagent calls and calls with no usage data are dropped. Tool gaps
  come from the harnesses whose spans have real durations, capped at 120 s.
- Mixes: M1 all (shared prefix 4,000), M2 claude_code only (shared 14,000, the p10 first-call size),
  M3 other harnesses (shared 2,000), M4 sessions with median prompt at most 20k (shared 4,000), M5 all
  with tool gaps times 4.
- Stage 0 (calibration): G* = best steady goodput (turns/s) over caps {2, 3, 4, 6, 8, 12, 16, 24, 32}
  at saturating arrivals (1 session/s), 3 seeds, for each mix and KV capacity {220k, 440k}.
- Stage 1: default engine (FCFS, no cap). Offered load {0.5, 0.65, 0.8, 0.95} x G*. Clients A, B, C.
  Spike 3x for 60 s at t = 900 s, horizon 4,800 s. 10 seeds. Each spike run has a matched control.
- Stage 2: the same spike runs under client A with each static cap in the sweep.
- Windows: pre 600 to 900 s, post 1,560 to 2,160 s (10 spike lengths after the spike), late 4,200 to
  4,800 s. Goodput = turns that finish before their timeout, per second.
Predeclared criteria (heuristics adapted from R-14; frozen in `e28_decide.py` before stage 1):
- Collapse point: the control is stable (late at least 0.9 x pre), spike goodput is below 0.7 x
  control in both post and late windows (means over seeds), and at least 7 of 10 paired seeds show it.
- K1 KILL I-80 (no phenomenon): no collapse point under client A or C.
- K2 KILL the retry component: at 80% or more of the A collapse points, B (no retries) also collapses.
- K3, at the A collapse points, with the fixed cap tuned once on M1 (best mean late goodput over the
  four loads for each cache size):
  - GO-B (a cap does not fix it; room for a different mechanism): the oracle cap restores less than
    0.8 x control late goodput at 50% or more of the points.
  - KILL the narrowed I-80 (no room over a fixed cap): fixed-cap loss against the oracle cap is at
    most 10% at 80% or more of the points.
  - GO (room for an adaptive or token-based working-set mechanism): mean fixed-cap loss above 20% in
    at least 3 of the 5 mixes.
  - PIVOT: any other case.
- Order: GO-B is checked first, then KILL, then GO.
Process: a different-model (Sonnet) review of the plan and code before stage 1 (D-29). Amendments go
below this entry before any run. A post-run Sonnet review follows.
**Pre-run review R-15 (Sonnet, 2026-10-07)** found 4 blockers, 6 major items and 14 minor items
(`dossier/REVIEW_E-28_PRERUN_2026-10-07.md`). The main session checked the vLLM claims in the vLLM v1
source (main branch, `vllm/v1/core/sched/scheduler.py` and `kv_cache_manager.py`), and they are correct.
Every blocker and major item is fixed in amendment A1.

**Amendment A1 (2026-10-07, after R-15 and before any experiment run).** These replace the matching
parts of the setup and criteria above.
- Simulator version 2 (R-15 B2, B3, M1, M3, m1, m2):
  - An allocation that does not fit fails without eviction (vLLM `allocate_slots`).
  - Preemption keeps the output tokens. The request re-prefills prompt plus output on resume.
  - Running requests are served in admission order. The victim is the most recently admitted request.
    No waiting request is admitted in a step that preempted a request.
  - Separate random streams for arrival candidates, session content, tool gaps and backoff jitter.
    A spike run and its control share every arrival outside the spike (unit test). The thinning bound
    is the same for both spike sizes and the control.
  - A client event inside a step happens at its own time. A turn that ends after its deadline is a
    timeout.
  - 13 unit tests pass, one for each fix.
- Workload (R-15 M4, m10): one shared prefix for each harness, used in every mix: claude_code 14,000,
  smolagents_code 2,500, other harnesses 800 (from the first-call sizes). The mixes differ only by
  session selection and the tool-gap scale.
- KV capacity (R-15 m4): 380k and 190k tokens (about 50 GB of KV on one H100).
- Timing (R-15 m3): spike at 2,400 s, horizon 6,000 s. Windows: pre 1,800 to 2,400 s, post 3,060 to
  3,660 s, late 5,400 to 6,000 s.
- Load and spike (R-15 M2):
  - Stage 0a: G* and N* = the best mean goodput and its cap over caps {2, 3, 4, 6, 8, 12, 16, 24, 32,
    48, 64} at saturating arrivals, 5 seeds, window 3,600 to 6,000 s.
  - Stage 0b: the knee = the largest load on the grid 0.30 to 1.00 x G* (steps of 0.05) such that the
    uncapped control (client C) is stable at every load up to it: mean late goodput at least 0.9 x
    offered, and at least 4 of 5 seeds with late at least 0.9 x pre.
  - Stage 1 loads: {0.6, 0.7, 0.8, 0.9, 1.0} x knee. Spike: 60 s that adds 0.5 x N* or 1.0 x N*
    expected extra sessions.
- Gateway (R-15 B4): no shedding in stage 2 (patience 1e9 s). The gateway wait is reported.
- Output lengths and constants (R-15 M6, m6): stage 3 (reported, not in the decision) re-runs stages
  0a, 0b and 1 for M1 at 380k with outputs clipped at 1,024, prefill twice and half as fast, decode KV
  reads twice and half as fast, and tool gaps x 0.25. Clients A and C.
- Criteria changes (frozen in `e28_decide.py` version 2):
  - A point is (mix, cache, load, spike size). The paired-seed test needs both post and late windows.
  - K2 uses client C (a 600 s timeout, so retries almost never fire): KILL the retry component if 80%
    or more of the A collapse points also collapse under C. If there is no A point but there is a C
    point, K2 is also KILL. B is reported only.
  - K3: the oracle cap and the fixed cap (tuned on M1 over all loads and spike sizes) are selected on
    seeds 1 to 5 and evaluated on seeds 6 to 10 (R-15 M5). "Restored" needs late goodput at least 0.8 x
    the client-A control and session success at least 0.9 x the control. A bootstrap interval of the
    loss is reported. The rule order stays: GO-B, KILL, GO, PIVOT.
- Not changed: the miss-cost statement in the rubber duck was wrong. R-15 m6 computed about 3 to 4
  times the GPU time of a hit at the mean output, not 10 times (Section 8.1 row 45).
**Second pre-run pass (Sonnet, part of R-15, 2026-10-08):** all A1 fixes verified (60 random storm
runs keep every invariant; spike and control share arrivals and sessions). One new blocker, N1: the
stage 0b knee rule needed 4 of 5 seeds with late at least 0.9 x pre. With a window noise CV of about
0.21, a stable system passes that only 15% of the time, so the knee would fall to 0.30 and K1 could
KILL for a statistical reason. Minor: N2 to N7.

**Amendment A2 (2026-10-08, after the second pass and before any experiment run).**
- Knee rule (N1): mean-based tests only, over 10 seeds: mean late goodput at least 0.85 x offered, mean
  late at least 0.85 x mean pre, late timeouts at most 0.02 and late preemptions at most 0.05 per
  completed turn. The scan stops after two consecutive failures; the knee is the last pass before them.
  The thresholds are 0.85, not 0.9, because a unit test showed that 0.9 passes a stable system with
  CV 0.21 only 86% of the time; 0.85 passes it at least 90% of the time. A collapse (below 0.7)
  still fails the rule.
- N2: the admission check uses the matched cache length. N5: the decision script prints the gateway
  wait for each oracle point (reported, not in the rule).
- Not changed (N3, N4, N6, N7, reported as limits): the knee uses client C while the decision uses
  client A; G* keeps a small best-of-11 bias; each turn has one extra decode step for the first token.
- Procedure (from R-15): after stage 0b, read only the knee values. If the knee is 0.30 for most mixes,
  stop and report before stage 1.
Result (2026-10-08; FACT for this simulator and setup):
- Runs: stage 0a 880, stage 0b 2,400, stage 1 4,500, stage 2 11,000, stage 3 1,800 (about 1 h 35 min
  on 9 cores). The frozen checksums matched before the decision (`results/FROZEN_2026-10-08.sha256`).
- Calibration: goodput rises with the session cap until the KV holds the working set, then falls by
  about half (M1 at 380k: 1.12 turns/s at cap 12, 0.55 at cap 32; hit share 0.94 against 0.39).
  N* is 8 to 48 across mixes and caches. Knees (base mixes): 0.55 to 0.85 x G*.
- Frozen decision (`results/e28_decision.txt`): **K1 KILL (no collapse point)**. K2 and K3 did not
  apply. The lowest spike/control goodput ratio over the 300 spike points is 0.89 (late) and 0.91
  (post). 28 of 100 client-A controls failed the stability filter because of arrival noise (about 9
  arrivals per 600 s); the spike runs share those arrivals, so the ratios are not affected.
- The spike acted: live sessions rose far above N* (for example 22 against 12) and preemptions rose
  (for example from 2 to 24). The extra sessions were served, and goodput went back to the control.
- Stage 3 (reported only): no variant has a collapse point (lowest ratio 0.88, post, prefill x0.5).
Validity and limits: VALID for the frozen grid, with these scope limits (R-16): spike at most 1 x N*
extra sessions, load at most 1.0 x knee, horizon 6,000 s, goodput as the only collapse signal, one
replica, unmeasured cost constants, hosted-model token counts.
Post-run review R-16 (Sonnet, 2026-10-08): K1 KILL is correct for the frozen grid. The main session's
explanation ("the bistable band has no measurable width", "above the knee is plain thrashing") is
wrong (Section 8.1 row 46). Reviewer check (unfrozen, M1 at 380k, 12,000 s, 6 seeds; the main session
re-ran 6 cells and got identical numbers):
- At 1.0 x knee with client C, a spike of 2 x N* traps 2 of 6 seeds and 4 x N* or 8 x N* traps 6 of
  6 for at least 9,000 s. Live sessions end at 22 to 112 against 5 in the control. The state is
  bimodal: preemptions are 0.00 or about 0.36 per second. Late goodput falls only about 20% (0.63
  against 0.79), so the frozen goodput rule could not see it.
- Controls with no spike leave the good state by themselves at higher loads (1 of 6 at 0.85 turns/s,
  5 of 6 at 1.0, which a cap of 12 still carries). This is a noise-triggered metastable transition,
  not plain overload.
- Client A (60 s timeout) never stays trapped, but it fails sessions (24 of 48 extra sessions at a
  4 x N* spike). Short timeouts act as a load shedder. This is the opposite of the I-80 retry story.
- A static cap of 12 or 16 clears the trapped state (late goodput 0.78 to 0.79, no preemptions).
What it changed: I-80 as written (retries sustain the collapse) stays KILL, now for a stronger reason:
retries with short timeouts help recovery. The working-set metastability is real in the trace-driven
model, but a static cap (Denning, CONCUR) removes it. What remains is thin (R-16 Section 9): an online
trigger from the bimodal state signal (CONCUR-adjacent), and K3 (does one cap fit mixes with N* from
8 to 48), which never ran. D-33.
Artifacts: `experiments/metastable_admission/` (code, data, `results/`, README with both rubber
ducks), `dossier/REVIEW_E-28_PRERUN_2026-10-07.md` (R-15), `dossier/REVIEW_R-16_E-28_POSTRUN_2026-10-08.md`
(R-16), bundle `dossier/bundles/metastable_admission_bundle.zip`. The reviewer scripts are
copied to `experiments/metastable_admission/reviewer_r16/`.

**E-30 Superchip power-coupling premise test** (drafted 2026-10-16, I-118). STATUS: FROZEN 2026-10-20 as a package, AMENDED 2026-10-20 (A1 after R-34), NOT RUN.
Waiting for D-52.
Freeze after a rubber duck and a Sonnet pre-run review (D-29).
Question: under a binding module limit, does CPU load on Grace lower GPU prefill throughput, and how
close is the best static CPU cap to an offline oracle?
Predeclared criteria (all heuristics): KILL 1 if prefill throughput changes by under 3% at the provider
default limit and at 680 W. KILL 2 if the best static CPU cap is within 5% of an offline oracle. PIVOT
(measurement note) if the effect appears only below documented deployment limits. GO if prefill loss is
at least 10% at a documented limit and the oracle gap over the best static cap is above 5%.
Setup: 1 x GH200, about 2 hours. Precondition: root, readable module and Grace power, settable module
cap (`nvidia-smi -pl X -sc 1`) and CPU cap (hwmon `power1_cap`). vLLM, an 8B model, prefill-heavy (4K in,
16 out) and decode-heavy requests; vLLM pinned to reserved cores; `stress-ng --cpu` on 0, 18, 36 and 54
other cores; control: a pure CUDA GEMM loop. Log SM clock, GPU, module and Grace power, and
`clocks_event_reasons`.
Frozen 2026-10-20 (R-35): the KILL 2 objective is U = f x T_p(c_p)/T_p(none) + (1 - f) x T_d(c_d)/T_d(none)
+ (f x C_p(c_p) + (1 - f) x C_d(c_d)) / (f x C_p(none) + (1 - f) x C_d(none)), with f in {0.3, 0.5, 0.7}
(prefill time share), T the GPU throughput, C the stress-ng operations per second, and c the CPU cap in
{none, 150, 100, 70} W. KILL 2 if the best per-phase pair beats the best single cap by under 5% for every
f. The equal weight is a heuristic. Package: `experiments/superchip_power_coupling/`.
Amended 2026-10-20 (A1 after R-34, before any run; the package README Section 8 wins). Prefix caching
is off, and prefill must draw at least 60% of the GPU power limit, else INVALID. Both hwmon cap
layouts are read. Preflight writes each knob to its current value and checks that 680 W is within the
module range. The CPU caps run at the limit with the largest prefill loss. KILL 2 is decided on a
constrained form: the best per-phase cap pair against the best single cap, by GPU throughput, with
CPU work at least 80% of uncapped. KILL 2 if that gain is under 5% for every f. The weighted gains
(CPU weight 0.25, 0.5, 1) are reported only. Loads run in random order per repeat, and N = 0 is
measured again at the end; drift above 2% gives INCONCLUSIVE. The "only below documented limits"
PIVOT clause is removed, because the sweep could not reach it. PIVOT now means a prefill loss of 3%
to 10%. Each stage saves when it ends, and a stress-ng parse failure raises. The process pins its
cores at start and needs at least 62. GEMM runs are 20 s x 5. Decision order: INVALID, INCONCLUSIVE,
KILL 1, KILL 2, GO, PIVOT; INCOMPLETE when the data for a step is missing. 9 unit tests pass;
`sweep.py --dry-run` passed on the Mac.
Result: none yet.

**E-31 Versioned session commit mechanism screen** (predeclared 2026-10-17, I-122).
STATUS: INVALID DESIGN, NOT RUN. Do not run this simulator.
Question: does generation fencing prevent stale state across cache identity, queued work, parser
output, and materialized progress when a live session changes?
Setup: a standard-library Python state machine in an isolated Docker container on the Apple M4 Mac.
It tests mutable state, hash invalidation, an epoch-only fence, full abort and rebuild, synchronous
drain, and versioned commit. It includes four source-derived scenarios, a bounded search to depth six, and a synthetic
scale sweep from 1 to 1,000 sessions. Docker uses no network, a read-only filesystem, four virtual
CPUs, and 4 GB of memory. Wall-clock timing is report-only because host load can affect Docker Desktop.
Predeclared criteria, all project heuristics:
1. INVALID if a conservative reference baseline violates a source-derived invariant.
2. KILL-MECHANISM if the versioned policy violates any invariant.
3. KILL-SIMPLE-BASELINE if hash invalidation handles every source-derived scenario.
4. PIVOT-EPOCH-ONLY if one generation tag handles every source-derived scenario.
5. PIVOT-TRIVIAL-REBUILD if full rebuild adds at most 3% synthetic token work in every 1% and 5%
   update cell.
6. PIVOT-WEAK-PREMISE if the mutable policy exposes fewer than two failure classes.
7. PASS-TO-INTEGRATION only if versioning passes and hash invalidation leaves a non-cache failure.
PASS-TO-INTEGRATION is not a GO verdict. It permits an implementation-level vLLM test.
Validity: this is an abstract model and a synthetic workload. It cannot establish vLLM behavior,
production update rates, GPU overhead, or publication novelty.
Artifacts: `experiments/versioned_session_commit/`. Result: none yet.
Blockers found before execution: Docker Desktop 4.43.2 has an invalid code signature and macOS
refuses to launch it. The Claude CLI 2.1.289 is signed out. Two Codex CLI review attempts also failed
while synchronizing managed preferences. Do not treat these failed attempts as an independent review.
Later status: Docker Desktop serves Linux containers through `desktop-linux`. The server reports
10 virtual CPUs and 7.65 GiB of memory. The Sonnet review remains the only execution blocker.
Authentication update: Claude is authenticated in the user's Terminal. The automation process cannot
read the macOS Keychain entry. Run `run_sonnet_review.sh` manually. Do not run E-31 first.
Result: No experiment result exists. R-23 found four blockers before execution. The invariant checks
were circular, one policy branch encoded the expected answer, liveness was absent, and model constants
fixed the cost result. R-24 also found that narrow source fixes weaken the global mechanism premise.
Validity: INVALID. Never cite this simulator as evidence for or against a runtime mechanism.
Artifacts: `experiments/versioned_session_commit/`,
`dossier/REVIEW_E-31_PRERUN_2026-10-17.md`, and
`dossier/SOURCE_AUDIT_I-122_2026-10-17.md`.

**E-32 Stateful metamorphic session test** (predeclared 2026-10-17, I-131).
STATUS: KILL-DESIGN. NOT RUN. R-25 and R-26 found unresolved blockers. R-27 killed the thesis route.
Question: can stateful transition generation with history-equivalence oracles find more mutable-session
faults than current tests and equal-budget random actions?
Predeclared criteria, all project heuristics:
1. GO if the full generator rediscovers at least four of six bugs, finds two bugs missed by random
   actions, has zero false positives in 1,000 fixed-commit sequences, minimizes traces to at most
   12 actions, and runs at least 100 valid sequences per second.
2. PIVOT if it finds real failures only within one fixed feature pair.
3. KILL if current tests or random actions match it, oracle false positives exceed 1%, issue-specific
   traces are required, or local speed is below 100 valid sequences per second.
Setup: Real vLLM V1 scheduler objects on CPU, fake model-runner outputs, six parent and fix pairs,
four trace-length limits, and Linux ARM64 Docker on the Apple M4 Mac. No model or GPU in this screen.
Result: No experiment result exists. The paper audit found circular corpus actions and no recurrence
after earlier narrow fixes.
Validity and limits: INVALID as a thesis test. Never cite this design as bug-finding evidence.
Artifacts: `experiments/session_state_metamorphic/`.

**E-33 Mutable-session CPU corpus pilot** (predeclared 2026-10-19, I-131).
STATUS: CANCELLED BEFORE RUN. R-26 blocked its specification. R-27 removed its thesis value.
Question: do four merged vLLM fixes provide stable CPU parent-and-fix regression pairs?
Predeclared criteria, all project heuristics:
1. PASS-TO-AMEND if all four parent tests fail for the documented fault and pass on the fix.
2. PIVOT-CORPUS if two or three pairs pass.
3. KILL-CURRENT-CORPUS if zero or one pair passes.
4. INVALID-ENVIRONMENT if one shared environment fault prevents three or more tests from executing.
Setup: Run the upstream tests from PRs 59103, 58259, 58668, and 58778. Copy each fix's
test-only diff onto its only parent. Use the pinned official ARM64 CPU image, four CPUs, 4 GiB,
and no network during test execution. Do not replace a case after the first parent run.
Result: No experiment result exists.
Validity and limits: This pilot tests corpus and local setup feasibility only. It does not test a
generator, oracle quality, novelty, or bug-finding value. Timing is informational.
Artifacts: `experiments/session_state_corpus_pilot/`.

**E-34 Crash-consistent duplex recovery epoch** (predeclared 2026-10-19, I-132).
STATUS: KILL-DESIGN. NOT IMPLEMENTED. NOT RUN.
Question: can one durable epoch recover opaque model state and duplex protocol state after a fail-stop
worker crash, with the same visible event sequence as a fault-free reference?
Predeclared criteria, all project heuristics:
1. INVALID if an expected negative control passes.
2. KILL-CORRECTNESS if the periodic-epoch policy fails one legal crash cut.
3. KILL-SIMPLE if full per-unit protection adds no more than 3% steady-state overhead.
4. KILL-REPLAY if full replay recovers within one application unit for the supported workload.
5. PIVOT-GPU if the protocol passes but only Docker timing distinguishes the policies.
6. GO-GPU only if the protocol passes, both simple baselines fail, and sourced workload mapping supports
   the difference. E-34 cannot produce a thesis GO by itself.
Setup: a deterministic opaque-state service, an independent reference model, an external durable client,
and uniform `SIGKILL` cuts. Compare replay-all, full snapshot per unit, and periodic recovery epochs.
Run in Linux ARM64 Docker with four CPUs, 4 GiB, no network, and a named durable volume.
Result: R-29 returned `KILL_IDEA` before implementation. The design met its predeclared
standard-composition kill rule.
Validity and limits: No experiment result exists. Do not cite E-34 as empirical evidence.
Artifacts: `experiments/crash_consistent_duplex_epoch/`.

**E-35 I-59 premise check: priority inversion through a shared-prefix producer** (predeclared
2026-10-19, I-59). STATUS: DONE 2026-10-20. Decision KILL (run 2), confirmed with scope limits by R-32.
The Stage A grid below was withdrawn by amendment A1 after R-31 and did not run.
Question: in current vLLM with priority scheduling and prefix caching, does a high-priority request
get slower because a lower-priority request computes the same prefix, and how much can priority
inheritance recover?
Prediction (HYPOTHESIS, from code reading at vLLM `3897f95c` and SGLang `476fe86a`): no. vLLM never
makes a waiting request wait for another request's prefix, and it publishes prefix blocks when it
schedules them. SGLang has a shared-prefix wait, but only for cache-aware policies, and it rejects
priority scheduling with those policies.
Predeclared criteria (heuristics): V1 positive control (an injected wait must give C1/C2 >= 1.25 and
a C4 gain >= 15% in one named cell, else INVALID); V2 termination. K1: in every cell, p50 and p99
TTFT(C1) <= 1.10 x TTFT(C2). K2: in every cell, the inheritance gain (C1 - C4) / C1 is under 15%.
KILL I-59 if K1 and K2 hold. MEASURE-FIRST (Stage B, full CPU backend) if a cell breaks K1 with a C4
gain of at least 15%. PIVOT if K1 breaks without a 15% gain.
Setup: Stage A drives the real vLLM `Scheduler` (synchronous) with a fake model runner in Docker
(Python-only vLLM, CPU PyTorch 2.14.0, linux/arm64). Metric: H's time to first token in scheduler
steps. Conditions C0 (no L), C1 (shared prefix), C2 (unshared control), C4 (L promoted). 192 cells,
40 seeded trials each, H/M/L priorities 0/5/10. Full design: `experiments/priority_inheritance_premise/README.md`.
Amendment A1 (after R-31, before any measurement): replace the grid with a trace check on the
unmodified synchronous `Scheduler` (the class that the CPU backend runs). T1: H's prefix hit at
admission equals what L has scheduled. T2: when L takes the whole budget, H's wait is budget order,
the same with and without sharing. T3: 300 random scenarios (half with tight KV blocks), H never
skipped. T4: an injected wait must be visible as `skip`. Decision: T4 fails, INVALID; T1 to T3 pass,
KILL; otherwise WEAK.
Amendment A2 (after run 1): run 1 was INVALID because L finished its prefill before H arrived (the
lone-request exemption to `long_prefill_token_threshold`). A2 adds in-flight arrival steps to T1 and
T4, and a T4 precondition. The assertions did not change.
Result (run 2, `results/trace-20261020T060157Z/`): KILL. T1: H's hit equalled L's scheduled tokens at
all 7 arrival steps (320, 832, 1344, 1856, then 2048). T2: 5 budget-wait steps in both conditions,
same start step, hit 2048 against 0. T3: 300 runs finished, 55 preemptions, 0 skips; waits were
`budget` 1729, `other` 310, `preempt_step` 17. T4: 29 and 13 skip steps.
Validity and limits: VALID with scope limits (R-32). The skip detector misses waits that come from a
failed allocation; a supplement labelled all 310 `other` waits as H's own failed allocations, and a
paired sweep (600 scenarios) found no case where a shared prefix made H later. Scope: vLLM `3897f95c`
synchronous scheduler with a fake runner, no KV connector, grammar, streaming, LoRA or speculative
decoding, one model config, block size 16. No wall-clock time. SGLang by code reading only.
What it changed: I-59 KILL. The conclusion is "no shared-prefix dependency", not "no priority
inversion". D-48 and D-49.
Artifacts: `experiments/priority_inheritance_premise/` (README with design, A1, A2 and result;
Dockerfile; harness; trace check; supplement; results with manifests), `dossier/REVIEW_E-35_PRERUN_2026-10-19.md`
(R-31), `dossier/REVIEW_E-35_POSTRUN_2026-10-20.md` (R-32).

**E-36 I-23 Test A, Session 1: calibration and the first two Hopper FP8 cells** (predeclared 2026-10-20,
I-23). STATUS: RUN 2026-10-26 on one H100 SXM (A1, A2, A3). Decision INCONCLUSIVE (only one safe backend).
Rehearsals 1 and 2e on L4 and RTX 4090; post-run review R-42.
Question: on one H100 with vLLM v0.32.0, do alternative linear and MoE backends for block-scaled FP8
change greedy output against the default backend, and does the fastest backend change with batch size?
Predeclared criteria (heuristics): V1 the reference backend agrees with itself on at least 95% of
prompts; V2 at least two backends start per cell. STOP-FLAT (recommend closing I-23 for Hopper FP8) if
in both cells every candidate flips 0 prompts (Wilson upper bound at most 1.5%) and backend speeds differ
by under 2% at every batch size. CONTINUE if a candidate flips at least 10% of prompts and the fastest
backend changes with batch size or is unsafe. Otherwise INCONCLUSIVE.
Setup: `vllm/vllm-openai:v0.32.0`; Cell D `Qwen/Qwen3-8B-FP8` (`--linear-backend`), Cell M
`Qwen/Qwen3-30B-A3B-FP8` (`--moe-backend`), BF16 control with two attention backends; 256 prompts (128
GSM8K, 128 UltraChat, seed 20261020), greedy, 128 tokens, top-20 logprobs, `max_num_seqs` 64; throughput
at batch 1, 8, 32 and 64 and a prefill test. Cost cap: 4 hours or $18. It is Session 1 of the full Test A
in `dossier/FRONTIER_R-33_2026-10-20.md` Section 7 (steps 1 and 2, and a fixed-batch cut of steps 4
and 5). Package: `experiments/safedispatch_testA/` (unit tests pass).
Amended 2026-10-20 (A1 after R-34, before any run; the package README Sections 9 and 10 win). The
candidates are fixed: Cell D `deep_gemm`, `cutlass`, `triton`, `torch`; Cell M `deep_gemm`,
`flashinfer_cutlass`, `hpc`, `triton`. A candidate whose kernel-selection log shows a fallback or the
reference kernel is an alias. It is excluded and must flip 0 prompts (negative control). Marlin
W8A16 is the positive control in both cells. It must flip at least 5% of prompts, else STOP-FLAT
cannot apply. Every run sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` and turns prefix caching off. Flip
runs use 256 new tokens and throughput runs 128; prompts above 3,072 tokens are dropped. R runs twice
(R_a, R_b), and its speed spread sets the noise. V1: R_a and R_b agree on at least 95% of prompts.
V2: each cell has at least one candidate that is not an alias. STOP-FLAT needs a Wilson upper bound
of at most 2% (at least 189 prompts), a speed spread under max(2%, 2 x R noise) at every batch size,
and a passed positive control. CONTINUE and STATIC-WINNER (PIVOT to static qualification) are as in
the README; otherwise INCONCLUSIVE. There is no inventory phase and no BF16 control. Each run has a
25-minute timeout; the session takes about 2 h 45 min.
Mac dry run (2026-10-20, mechanics only): the vLLM v0.32.0 CPU image built from source, Qwen3-0.6B,
16 prompts. All runs exited 0, R_a and R_b agreed on 16 of 16 prompts, and `--linear-backend torch`
flipped 0. The dry run found five defects, all fixed. One is a vLLM v0.32.0 CPU bug (argument order
in `_compute_slot_mapping_kernel_impl`) that vLLM main fixed in PR 61992; only the dry run mounts the
patch. Kernel-selection lines are now parsed from each run's `log.txt`, and `compare.py` reports
`kernel_log_found`. A BF16 model on CPU logs no selection line, so the parser is tested only on
synthetic logs (14 unit tests pass).
Amended 2026-10-22 (A2 for Runpod, D-56; R-39 Sonnet pre-run review AMEND_THEN_RUN, fixes applied;
package README Section 11 wins). Phase A (the runs, flags, order, thresholds, 3 h 10 min cap and
`compare.py`) does not change, and the decision comes only from Phase A. Added: resume (skip ok runs,
elapsed time kept across restarts), a GPU UUID check at each start, code, results and caches on the Pod
volume disk, an environment folder for each start (hardware record, model revisions, telemetry every
1 s in UTC, a GPU microbenchmark at start and end), background model download, a torch.profiler kernel
table at batch 1, 8 and 64 after the measurements of R_a and each candidate, and an optional AUTO_STOP
with a watchdog. A1 defect fixed: the logged exit code could be wrong (the analysis never used it).
Exploratory Phase B (reported only, no new run after 6 h): B0 repeat of the most-flipping candidate, B1
flip test at `max_num_seqs` 8, B3 eager-mode reference, B2 decode sweep at batch 1 to 256. `extras.py`
reports telemetry, `suspect_speed`, a weak GEMM-kernel alias check and the Phase B summaries. Cost on
Runpod H100 SXM Secure ($3.99/h on 2026-10-22): at most about $27 for the full session (Pod limit
6 h 40 min), about $11 to $14 for Phase A only. Tests: 22 unit tests pass; a stub session run checked
resume, the GPU check, the telemetry fallback and AUTO_STOP logging; the profiler path was checked
only on CPU (VERIFY on the H100).
Result (2026-10-26, Pod sm57l4ssfngmvm, 1 x H100 80GB HBM3, EU-NL-1, about $11.20; package README
Section 13): frozen decision INCONCLUSIVE, failed condition "only one safe backend". FACT: V1 256 of 256
in both cells; positive control flipped 243 and 224 of 256; the MoE alias flipped 0. Every working
alternative changed greedy output on 86% to 95% of prompts (dense deep_gemm 232, cutlass 244, triton
240; MoE deep_gemm 235, flashinfer_cutlass 233); torch and hpc are unsupported. The default backend R
was the fastest at batch 1, 8, 32 and 64 in both cells (dense 10,281 tok/s at batch 64). Phase B: the
flips repeat exactly (B0); the same pattern at max_num_seqs 8 (B1); dense R fastest at batch 1 to 256,
MoE differences inside the R noise (B2); eager mode changes 233 and 238 of 256 prompts (B3).
Incident: the FlashInfer TRT-LLM DeepGEMM JIT needs nvcc on PATH; fixed in session1.sh.
Post-run review R-42 (Sonnet): CONFIRM_WITH_SCOPE_LIMITS. Corrections: the flip range is 87.5% to 95.3%;
R misses a literal STATIC-WINNER only at dense batch 32 (2.7% gap); the Marlin W8A16 control is 1% to 5%
faster than R in the MoE cell at batch 4 to 32; dense deep_gemm flips because of different fused
activation-quantization kernels, not the GEMM. INTERPRETATION (after R-42): no exact-output-preserving
alternative exists, and R is the fastest FP8 backend; speed headroom under a tolerance contract is
untested. Exact match detects any numerical change (eager mode alone flips 233 and 238 of 256), so the
KILL condition is trivially true for a safe set of {R}. A first draft said "close to the KILL
condition"; R-42 refuted that wording (Section 8.1).

**E-38 I-23 Test A2: a tolerance contract for Hopper block-FP8 backends** (predeclared 2026-10-26, I-23,
D-61). STATUS: RUN 2026-10-27 (first run INCOMPLETE; rerun D-62 KILL). Package: `experiments/safedispatch_testA2/`.
Question: on one H100, is there a block-FP8 backend (or Marlin W8A16) that keeps GSM8K accuracy within
2 pp of the default R (paired bootstrap non-inferiority, 1,319 problems) and stays within half of the
BF16 anchor's teacher-forced KL and top-1 disagreement, and that is also faster than R by at least 3%
(R, C, R, C in one Pod, `compare_runs.py` verdict BETTER) at some batch size?
Predeclared rule: GO = a candidate with HEADROOM (acceptable and BETTER) in a valid cell; KILL = both
cells valid, no errored candidate, anchors present, the planned Marlin MoE speed test complete, and no
HEADROOM; PIVOT = faster but not acceptable; otherwise INCONCLUSIVE; a cell failing V1 to V5 is INVALID
for that cell. Prior (HYPOTHESIS): KILL is most likely; the only GO path is MoE Marlin (E-36 B2: 1% to 5%
faster at batch 4 to 32). Speed runs only for MoE Marlin; the other candidates are slower in E-36 data.
Setup: vllm/vllm-openai:v0.32.0 with the E-36 fixes; Qwen3-8B-FP8 and Qwen3-30B-A3B-FP8, BF16 anchors
Qwen3-8B and Qwen3-30B-A3B; greedy, 768 tokens, max_num_seqs 256; budget about 2 h 15 min to 2 h 45 min,
cap 3 h new runs, Pod limit 3 h 30 min, $15.
Verification (FACT, Mac): 10 unit tests pass; two vLLM CPU dry runs (Qwen3-0.6B, 4 problems) passed
mechanically (warm-ups, reference distributions, teacher-forced and decode-prefix scoring, accept,
speed A/B, decision). Pre-run review R-43 (Sonnet): AMEND_THEN_RUN, all 13 findings handled in A1.
Result (2026-10-27, Pod tlkcp7qylpj9re, 1 x H100, US-MO-1, $14.32; package README Section 13):
INCOMPLETE. A software defect (the reference load decompressed the whole array per position, about
503 min per array) made every scoring step time out; the cap then skipped the Marlin speed test and
cell D. The frozen `decide` printed INVALID, but V3 to V5 were not measured. Surviving data (FACT, cell
M, 1,319 problems): R_a 95.07%; R_b identical (1,319 of 1,319); deep_gemm, flashinfer_cutlass and the
triton alias +0.00 pp; marlin +0.23 pp [-0.61, +1.06]; BF16 -0.30 pp; every candidate passes A1 although
82 to 134 of 1,319 outputs are identical to R_a. INTERPRETATION: exact match was not a quality
contract. The defect is fixed (11 tests pass, one at the real size). Lessons: watch for every non-zero
exit; time every per-item loop at the real size (PROTOCOL S3, RUNPOD Section 5).
Rerun (D-62, 2026-10-27, Pod 79jekplym1wxsh, about $7.60; README Section 14): frozen decision KILL. All
real candidates pass A1 (within about 1 pp of R) and fail A2 (TF KL 0.60 to 0.93 x BF16, top-1 0.72 to
0.95 x BF16; bar 0.5). Marlin MoE speed not BETTER (batch 8 +5.5% UNCLEAR, batch 256 -10.7% WORSE).
R-44 (Sonnet): CONFIRM_WITH_SCOPE_LIMITS; open lead: a 5% to 7% Marlin MoE gain at batch 4 to 8 is not
excluded by the 2 against 2 design. D-63 (recommendation): close I-23 for Hopper block-FP8.

**E-40 I-23 NVFP4 backend choice on one B200** (predeclared 2026-10-27, I-23, user request). STATUS:
PREDECLARED, AMENDED (A1 after R-47), NOT RUN. Package: `experiments/nvfp4_dispatch/`. Screen every NVFP4
linear (cell D, nvidia/Qwen3-8B-NVFP4) and MoE (cell M, nvidia/Qwen3-30B-A3B-NVFP4) backend once, keep the
best per cell (ratio at least 1.02), check GSM8K A1, then 6 paired rounds. GO: A1 pass and a lower bound
at least 1.03 at batch 1, 8, 32 or 128. Cap $25. Tests: 5 pass.
Result: none yet.

**E-39 Marlin W8A16 MoE small-batch speed on H100** (predeclared 2026-10-27, I-23). STATUS: RUN 2026-10-27,
INCONCLUSIVE. Package: `experiments/marlin_moe_speed/`. Question: is
`--moe-backend marlin` at least 3% faster than R for Qwen3-30B-A3B-FP8 at batch 4, 8, 16 or 32 (10 paired
rounds, one-sided t, Bonferroni over 4)? Accuracy reused from E-38 (Marlin = R on GSM8K); E-38's A2 retired
post hoc; a GO is PIVOT-grade. GO resolves about 5.5% or more. Cap $8. Tests: 7 pass.
Result (2026-10-27, Pod cj79arezidx9ax, 1 x H100 SXM, US-MO-1, about $5.20; 10 complete rounds, V2 and V3
pass, no hard slowdown, 74 checksums match): frozen decision INCONCLUSIVE (4, 8, 16). Marlin / R ratio
(one-sided bounds at alpha 0.0125): batch 4 1.015 [0.983, 1.047]; batch 8 1.025 [1.002, 1.049]; batch 16
1.011 [0.992, 1.031]; batch 32 0.995 [0.976, 1.015]. Reported: batch 1 0.949, 2 0.959, 64 0.959, 128 0.976,
256 0.887 (all below 1). INTERPRETATION: Marlin MoE is about 1% to 2.5% faster at batch 4 to 16 (batch 8 is
above 1 at this level) and slower everywhere else; the gain is below the 3% bar and its upper bound is
under 5%. A batch-dependent dispatch would gain at most a few percent on a narrow band. R-46 (Sonnet):
CONFIRM_WITH_SCOPE_LIMITS; batch 32 is a tie; a 3% gain at batch 8 is not excluded (upper 1.049).

**E-37 Control-plane complexity screen** (predeclared and run 2026-10-21, I-161, I-150). STATUS: RUN.
SCREEN ONLY: an engineering screen, not a research result.
Question: S1, what does one SGLang `evict()` call cost as a function of the number E of evictable leaves,
compared with a heap kept between calls? S2, what does vLLM-style block hashing cost for one prompt?
Predeclared criteria (heuristics): S1 is engineering-relevant if one call at E = 50,000 costs at least
1 ms and the kept heap is at least 10 times cheaper. S2 is engineering-relevant at 20 ms or more for a
131,072-token prompt, minor between 5 and 20 ms, dropped under 5 ms. No result can make a research gap.
Setup: M4 MacBook Air, Python 3.14.6, one core. S1 copies the shape of SGLang a3d00eff `radix_cache.py`
`evict()` (list of all evictable leaves, Python priority call, `heapify`, pop 4) with stand-in nodes. S2
copies vLLM 90c4a1a `hash_block_tokens` with `sha256(pickle)`, block size 16. Best of 5 to 21 repeats.
Result: S1, 0.88 ms at E = 10,000, 4.6 ms at E = 50,000, 11.2 ms at E = 100,000; the kept heap pops 4
leaves in about 2 microseconds. S2, 7.2 ms at 131,072 tokens and 11.4 ms at 204,800 tokens.
Verdict against the criteria: S1 passes (engineering-relevant). S2 is minor.
Validity and limits: SCREEN ONLY. The M4 core is fast, so server times are likely higher. The S1 kept heap
does not count its upkeep (one O(log E) push for each touched leaf). E-37 did not measure how often
`evict()` runs under real memory pressure. Post-run review: R-38 (Sonnet).
What it changed: I-161 becomes an upstream-PR candidate. S2 is folded into the closed I-150.
Artifacts: `experiments/control_plane_complexity_screen/` (README.md, screen.py, result_2026-10-21.json).
Post-run review R-38 (Sonnet): CONFIRM_WITH_SCOPE_LIMITS. S1 models the legacy Python `RadixCache`; the
default is the Rust unified core, which has the same O(E) heap refill with a smaller constant (not
measured). SGLang PR 38371 measured 4.9 ms at 35,000 leaves on the Python unified cache, so E-37 S1 is a
lower bound. The real vLLM hashing loop is about 9.6 ms at 131,072 tokens (reviewer), still minor. The
"upstream-PR candidate" result is withdrawn: PRs 38371 and 38454 already exist (Section 8.1 row 64).

### 6.2 Review registry

**R-01 Adversarial self-review** (2026-09-16, handoff section 12). Retracted or flagged: the
V1 synthetic suite over-interpreted, V2 network structural bias, V2 elastic break-even
artifacts, reconstructability POC proved only a planning abstraction, planner costs were ours,
first cached-replay timing showed a known principle, v1.1 frozen-KV timings invalid, 9.5%
oracle not generalizable, do not promote the collective-engine idea merely because the hybrid
line is uncertain.

**R-02 Novelty pass 2** (2026-09-16 to 17, dossier section 2). Located the exact prior claim of
the hybrid mechanism (#53041, #52773, Sparse Prefix Caching, HYPIC, ReplaySSM, #55697,
#52959, Illusion of Equivalence, the collective survey). Outcome: I-16 rejected by its own
FLOP model plus #52773 and #53041.

**R-03 Mac-testable gates** (2026-09-16 to 18, dossier iteration 4b and 4c). Outcomes: I-19
gate PASS, I-15 KILL, I-18 structural gap confirmed, I-20 KILL, one upstream-patch idea
(suffix skip).

**R-04 2026-09-22 handoff adversarial pass** (handoff sections 3 and 4). Frontier synthesis:
"The interesting boundary is no longer 'can we make kernels fast?' or 'can we test kernels?'
It is whether a production serving system can safely exploit multiple fast implementations
and aggressive low-precision paths without silently changing model behavior, and whether the
selection policy can be cheaper than always using the conservative implementation."
Candidate verdicts in 5.9. Limits: its threat list is incomplete, see R-05.

**R-05 Adversarial review with literature pass** (2026-09-23).
Scope: I-06, I-18, I-19, I-23. Method: two adversarial passes, about 70 sources, NCCL
re-audit, cited GitHub issues opened, survey quote verified in extracted PDF text, handoff
package checksums verified. Findings:
1. FACT: the 2026-09-22 handoff misses MarginGate, 2609.26621, 2609.00363, 2609.25624,
   2609.11356, 2609.04748, 2605.19537, TBIK, CoRun, Runtime Observability, Ekka and GRIEF, the
   SGLang Unified Radix Cache, the vLLM hybrid batch-invariance PRs, and 2609.15030.
2. FACT: I-23's H1 and H2 are already published. H3 and H4 are open. The static safe
   baseline got cheaper.
3. FACT: I-19's motivating bugs are indexing, unit-mismatch and cache-key bugs, not
   layer-skip violations. #53041 is an uncommented discussion.
4. FACT and INTERPRETATION: I-18's source gap replicates, but a per-call NCCL config exists,
   NVIDIA's data shows CE ahead, and the planned test cannot separate a live signal from a
   static table.
5. FACT: I-06's components are all published.
Verdicts: I-23 MEASURE-FIRST narrowed. I-18 MEASURE-FIRST low priority. I-19 KILL as thesis.
I-06 KILL. Nothing is a GO.
Full write-up: `dossier/ADVERSARIAL_REVIEW_2026-09-23.md`. Not done in R-05: I-09, I-12, I-21
and I-01 to I-05 were not re-reviewed. The claim "I did not find" is a search log, not proof.

**R-06 Correctness review of `dossier/LIT_REVIEW_AGENTIC_HARNESSES_2026-09-30.md`** (2026-09-30).
Scope: the literature review only, no idea reviewed. Method: two independent subagent reviewers
(A and B), about 30 primary pages fetched by each, then the disputed items re-checked by the
main session. Findings that were fixed:
1. FACT: the Terminal-Bench pair (70.1 vs 63.8) is from version 2.1. On 2.0 the order reverses
   (Claude Code 58.0, Terminus-2 62.9). It was also listed as "strongest support" for C1 while
   Section 8 said it carried no weight.
2. FACT: the Anthropic 90.2% (internal eval) and 80% (BrowseComp) figures were conflated, and the
   inference "much of the gain is more compute" did not follow.
3. FACT: Kim et al. matched token budgets, so "no compute-matched frontier comparison exists"
   was wrong.
4. FACT: the C8 independence claim ignored the ABC and HAL author overlap. C9 and C10 were
   labelled ESTABLISHED on evidence from overlapping author groups and were downgraded. C11 is
   now OPEN, a new label defined in Section 0 of the report.
5. FACT: the Vats and Golev CI statement was inverted (the abstract says CIs include zero
   except for the largest gap). "Only study reporting CIs" contradicted Section 5. Fan et al.
   planning, Arjmandi n (80 paired), the SWE-bench Pro range, METR stars and sign convention,
   Meta-Harness "top-ranked", the I-23 pointer (should be I-28), and the 36.2% qualifiers were
   corrected.
Disputed and resolved: reviewer B's TOPAS point was a false alarm (TOPAS is 2608.25523).
Reviewer A's "2 to 8 points" for Vats and Golev was wrong, the abstract says 0 to 8.
Re-checked by the main session against abstracts: Fan et al., Meta-Harness, Arjmandi, Vats and
Golev, Kim et al., METR stars. Not re-checked by the main session: the SWE-bench Pro numbers
(one reviewer, Augment blog) and the Terminal-Bench and Anthropic figures (both reviewers agree).

**R-07 Correctness review of `dossier/LIT_REVIEW_GPU_EFFICIENCY_2026-09-30.md`** (2026-09-30).
Scope: the literature review only, no idea reviewed. Method: two independent subagent reviewers
(A and B), each told the most serious verified correctness errors earn five points and that
unverifiable findings would be struck. Each opened the cited PDFs and pages. The main session then
re-read the PDFs or pages for every disputed or single-reviewer item: Samsi, Fishman, DeepSeek-V3,
MegaScale Tables 2 and 3, Korthikanti Tables 3 and 5, PaLM Table 3, DistServe Section 6.2, Llama 3
Table 4, Spec-Bench survey table, Thinking Machines blog, and the Yang, AWQ, Ho and Kurtic abstracts.
Findings that were confirmed and fixed:
1. FACT: Fishman et al. report FP8 divergence after about 200B tokens (220B in the text). 2T is the
   size of the run that succeeded with Smooth-SwiGLU. Found by reviewer A only. The error was in the
   report, journal Section 4 and Section 7.9, and fed the I-23 test-design argument.
2. FACT: the DeepSeek-V3 "<0.25%" loss gap is from 16B (1.33T tokens) and 230B (about 0.9T tokens)
   MoE models, not V3. Only GEMMs are FP8. The "about 2%" error is from a test on random matrices
   with K = 4096. Both reviewers.
3. FACT: Samsi et al. 3 to 4 J per token is from the V100 runs (8, 16, 32 shards), not A100.
   Reviewer B only. Reviewer A listed the figure as correct.
4. FACT: MegaScale ablation (Table 3, 256 GPUs) gives overlap 6.2 of 17.6 points and algorithmic
   changes 8.6. "1.34x comes largely from overlap" was wrong. Both reviewers.
5. FACT: the C4 ceiling claim was wrong. MegaScale Table 2 has 59.1% at 3,072 GPUs, Korthikanti
   Table 5 has 56.3% for 1T without data parallelism. Reviewer A only.
6. FACT: the Thinking Machines 42 s vs 26 s timing is from Qwen-3-8B on one GPU, not the 235B model.
   Reviewer B only.
7. FACT: MT-NLG 30.2% with attention (PaLM Table 3), GPT-3's 21.3% uses non-attention FLOPs. Both.
8. FACT: 15 papers were full-text, not 16. S2 was not opened. Both.
9. FACT: DistServe 12.6x is against vLLM, AWQ's ">3x" is TinyChat speed, the speculative decoding
   "shrinkage" compared different models. C3 downgraded, Ho, Yang, BF16 and C12 wording corrected.
Disputed and resolved: the two reviewers disagreed on Samsi, and reviewer B was right. Reviewer A
flagged the Patterson "100x to 100,000x" scope and the C13 label. Neither was changed: the Patterson
paper was not re-read, and C13 has partial second support from MegaScale.
Score: raw count of findings each reviewer called serious was 6 for B and 5 for A. After the main
session regraded by impact, four findings per reviewer held up as serious (A: Fishman, DeepSeek,
MegaScale, C4 ceiling. B: Samsi, DeepSeek, MegaScale, Thinking Machines). The points were left for
the user to award.
Limits: reviewers are the same model family as the author of the report, so shared blind spots are
possible. The 54 sources were not all re-read; about 15 PDFs and 4 abstracts were. Sources such as
Kumar et al., NVFP4, Mooncake, GEMINI and ByteCheckpoint were not re-read by the main session, so
their numbers rest on the first draft plus whatever the reviewers confirmed.

**R-08 Adversarial review of I-41, the ingest structurer** (2026-10-04).
Scope: the user's idea (I-41) and four narrowed variants (I-42 to I-45). Method: the
`research-gap-dossier` workflow. Pass 1: about 20 web searches and 15 primary pages (ACON,
TokenPilot, AgentDiet, Squeez, CAPC, Don't Break the Cache, Keepalive Economics, TACO, CWL,
2608.16370, PatchKV, Still, AgentKV, Anthropic context-editing and prompt-caching docs). Then E-24
and E-25 with predeclared criteria. Pass 2, fresh searches on the narrowed questions: TraceLab,
claude-code issue #94177, SWE-Pruner, SWE-Pruner Pro, OpenAI 24 h cache retention, the 11-harness
source study.
Findings:
1. FACT: small-model compression of tool outputs at ingest is published (ACON distils its
   compressor into 8B to 14B models; SWE-Pruner uses 0.6B; Squeez 2B; AgentDiet uses a cheaper
   reflection model).
2. FACT: cache-continuity design is published. TokenPilot does ingestion-aware compaction with a
   byte-identical prefix and batches evictions every 3 turns; AgentDiet edits only an older step to
   keep the KV cache; Anthropic's `clear_at_least` exists so a clear is worth its cache write.
3. FACT: the cache premise fails in real use. E-24 P1 (compaction 0.39% of break writes, TTL expiry
   83.4%). TraceLab (4,265 sessions): 95.7% hit rate, misses mostly after human idle gaps,
   compaction in 4.5% of Claude Code sessions. Issue #94177: 43% of one user's cache writes from
   TTL expiry.
4. FACT: E-24 P2 oracle saving 23.8% at r = 4, inside the 15 to 40% band of published results.
5. FACT: E-24 P4, 71.5% of qualifying edits copy tool-output text verbatim; about half of the
   sources are older than the last 3 tool results. E-25: a 0.6B abstractive structurer kept the
   edited span in 1 of 30 cases.
6. INTERPRETATION: the only part I did not find in public work is the verbatim-reuse measurement
   (TraceLab does not report it). It is a measurement, not a mechanism, and it is harness-level,
   outside the GPU and serving focus of the thesis.
7. FACT: the 2026-09-30 harness review's claim that representative public traces of harness
   traffic were missing is contradicted by TraceLab (released 2026-06-29) and the Exgentic
   OpenTelemetry dataset (Section 8.1 row 41). This weakens the trace-gap angle of I-28.
Hostile-reviewer sentence: "This is ACON's observation compression with TokenPilot's cache rules,
motivated by a cache-lifetime problem that TraceLab shows comes from idle gaps, not compaction."
No concrete answer exists, so the mechanism is killed.
Verdicts: I-41 KILL as a thesis mechanism. I-42 KILL. I-43 DEPRIORITIZED (measurement only, next
step needs TraceLab data). I-44 SCANNED, crowded. I-45 SCANNED, crowded, folded into I-28.
Full write-up: `dossier/DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md`.

**R-09 Cross-domain directions from peer-reviewed literature** (2026-10-06).
Scope: 12 new candidates (I-46 to I-57). The user asked for peer-reviewed papers only, first
principles, and a solution taken from another field that we can test.
Method: the `research-gap-dossier` workflow. Pass 1: about 30 web searches across agent serving,
inference numerics, fault tolerance, and evaluation methods. Pass 2: new targeted searches on each
survivor, venue checks, and data-availability checks (AcmeTrace README, Kokolis and Cui data).
Findings:
1. FACT: agent KV retention, speculative tool execution, sandbox fork, and group-testing fault
   localization have peer-reviewed prior art at top venues (ICML 2024, EuroSys 2025, NeurIPS 2025,
   ATC 2025, NSDI 2025, ICLR 2026) and many 2026 preprints. Killed.
2. FACT: the nearest peer-reviewed work to I-46 (PKH, EuroMLSys 2026) samples configurations, not
   time windows of a trace.
3. FACT: the nearest peer-reviewed work to I-48 tests black-box output distributions (ICLR 2025, two
   papers) or verifies provider claims (TOPLOC, ICML 2025).
4. FACT: AcmeTrace (NSDI 2024) has job-level NODE_FAIL states only. I could not confirm public
   node-level GPU failure logs from Kokolis et al. (HPCA 2025) or Cui et al. (SC 2025).
5. INTERPRETATION: because preprints were excluded as evidence, each surviving gap is less certain
   than usual. Do one preprint scoop check before committing to any of them.
Verdicts: I-46, I-47, I-48 MEASURE-FIRST. I-49 HOLD. I-50, I-51, I-52, I-53, I-54, I-55, I-57 KILL.
I-56 DEPRIORITIZED, folded into I-48.
Full write-up: `dossier/NEW_DIRECTIONS_R-09_2026-10-06.md`.

**R-10 Mechanism-transfer routes, adversarial review by a different model** (2026-10-07).
Scope: I-46 to I-49 (re-reviewed under D-29) and new candidates I-58 to I-75.
Method: the main session screened about 25 PagedAttention-style transfers with web searches. A
Sonnet subagent then reviewed the shortlist adversarially (105 tool calls, about 44 minutes).
The main session re-ran the reviewer's simulation (same output) and checked 8 key citations on
their primary pages (DiFR, Log Probability Tracking, the recurrent-cache dither paper, the
sigma-delta disclosure, llama.cpp PR 21038, the Delta dataset, the MLSys 2026 sparing paper, vLLM
issue 59907).
Findings:
1. FACT: I-48's two halves exist: reference replay with a shared seed (DiFR, vLLM integration) and
   cheap logprob tracking (ICLR 2026). The flip invariant is the triangle inequality (row 42).
2. FACT: the Delta node-level GPU error data is public (row 43). Meta's MLSys 2026 paper gives
   closed-form spare sizing.
3. FACT: no prior work applies error feedback to the V cache along tokens. The reviewer's synthetic
   check shows a large gain for smooth attention weights and a loss for spiky weights.
4. FACT: llama.cpp rotates the KV cache with Hadamard matrices since 2026-04-01, so rotated 2-bit V is
   the baseline for I-58.
5. INTERPRETATION: most direct mechanism transfers into LLM infrastructure are taken. I-58 is the
   only strong route, and only in its narrowed, per-head form.
Verdicts: I-58 NARROW, MEASURE-FIRST. I-59, I-60 WEAK backups. I-46, I-47, I-48 KILL. I-49 KILL as a
thesis route. I-61 to I-75 KILL.
Full write-up: `dossier/ROUTES_R-10_2026-10-07.md`.

**R-11 Pre-run reviews of E-27 (Sonnet, two passes)** (2026-10-07).
Scope: the E-27 plan and code before any run (D-29). A first attempt stopped at a usage limit with no
output.
Findings, first pass: 3 blockers (the step-1 kill rule assumed independent rounding errors; the
perplexity rules used absolute perplexity and had no CI; M1 was a mean of ratios with no per-window
records), 7 major items (repetitive passkey filler, no mechanism controls, noisy gates, a mislabelled
baseline, a memory risk of about 14 GB, unsafe resume logic, no frozen decision script), and minor items.
The main session checked them and fixed all of them (amendment A1).
Findings, second pass: no blocker. All A1 items verified with independent checks. 11 minor items; the
ones that affect correctness or reproducibility were fixed (amendment A2).
Effect: the experiment that ran was materially better than the first plan. Without R-11, the step-1
rule could have stopped the test on a wrong assumption, and the perplexity run could have run out of
memory.

**R-12 Post-run review of E-27 (Sonnet)** (2026-10-07).
Scope: E-27 results, code and decision. Method: recomputed the numbers from the result files,
reimplemented the method independently on real data, and tested other regions, lengths, K quantization
and token orders at the layer-local level.
Findings: KILL CONFIRMED WITH SCOPE LIMITS. No bug. Two explanations in the main session's rubber duck
were wrong (Section 8.1 row 44). M2 had no headroom, so the kill rests on M1.
Verdict: I-58 KILL.

**R-13 Route search 2** (2026-10-07). Scope: I-76 to I-81. Method: a map of about 100 classic mechanisms
against LLM and agent infrastructure, then searches for new environments (Apple Silicon) and for the
stated limits of the newest papers (PhaseGate, CONCUR). About 20 web searches, 2 primary pages checked
(PhaseGate and CONCUR abstracts). Findings: five of six candidates are covered by 2025 and 2026 work,
much of it from the last few weeks. I-80 is open but rests on simulator evidence. No different-model
review yet. Verdicts: I-76, I-77, I-78, I-79, I-81 KILL. I-80 HYPOTHESIS.

**R-14 Adversarial review of I-80, with combinations (Sonnet)** (2026-10-07).
Scope: I-80, and the combinations I-80 + I-59, I-60, I-61, I-74 (the user allowed combinations, D-32).
Method: a Sonnet subagent (26 tool calls, about 6.5 minutes). Web searches, 9 primary pages opened, and a
toy discrete-event simulator (synthetic sessions: 8k private prefix, 15 turns, no shared prefix, 4 seeds).
The main session opened the CONCUR, 2606.15555 and Continuum abstracts and confirmed the claims, with one
correction: the 2606.15555 abstract gives "scheduling design principles"; the admission cap detail is
search-only.
Findings:
1. FACT (toy simulator): a lasting collapse after a 30 to 60 s spike exists, but only inside a window of
   offered load (about 0.55 to 1.0 of good-state capacity). Above it the system collapses with no spike.
2. FACT (toy simulator): retries are not needed. With a 600 s timeout no retry fires and the collapse
   stays. Retries = 0 gives the same tail goodput as retries = 2.
3. FACT (toy simulator): a session-concurrency cap sized to the cache removes the collapse (tail 1.2 to
   1.7 turns/s against 0.6 to 0.8). A waiting-queue cap does not help.
4. FACT (abstracts checked): CONCUR (2601.22705) is agent admission control from cache signals, batch
   only. 2606.15555 models eviction-driven instability in memory-constrained serving (no agents, no
   prefix cache, no retries). Continuum (2511.02230) pins KV with a TTL during tool calls, no admission.
5. INTERPRETATION: no paper found applies metastability to LLM serving, but the working-set loop and its
   fix (a session cap) are covered. KVFlow, Pensieve, Autellix, Parrot, SOLA, Andes and QLM were not
   searched in this pass.
Verdicts: I-80 as written (retries plus cache loss) KILL. I-80 narrowed to an online adaptive session cap
WEAK. I-80 + I-61 WEAK. I-80 + I-74 KILL. I-80 + I-59 KILL. I-80 + I-60 WEAK.
Effect: E-28 was redesigned as an independent trace-driven replication of findings 1 to 3, with R-14's
thresholds, and a test of whether one tuned cap is enough.
Full write-up: `dossier/REVIEW_R-14_I-80_2026-10-07.md` (toy simulator code was in the session
scratchpad and is not kept).

**R-15 Pre-run reviews of E-28 (Sonnet, two passes)** (2026-10-07 and 2026-10-08).
Scope: the E-28 plan, simulator, runner and decision script before any run (D-29).
Findings, first pass: 4 blockers (a crash in the runner; a blocked admission flushed every idle cache;
preemption re-decoded all output tokens; gateway shedding could make GO-B true by construction), 6
major items (scheduler order differed from vLLM v1; a weak spike and a coarse load grid; spike and
control runs were not paired after the spike; the shared prefix was a free knob per mix; oracle
selection bias; long hosted-model outputs mixed with timeouts) and 14 minor items. The main session
confirmed the vLLM claims in the vLLM v1 source and fixed all blockers and major items (amendment A1).
Findings, second pass: all A1 fixes verified. One new blocker (N1, a knee rule that a stable system
passes only 15% of the time), fixed in amendment A2.
Effect: without R-15, E-28 would have had a simulator that builds the miss loop by itself (B2, B3)
and a knee rule that would have forced K1 KILL for a statistical reason (N1).
Full write-up: `dossier/REVIEW_E-28_PRERUN_2026-10-07.md`.

**R-16 Post-run review of E-28 (Sonnet)** (2026-10-08).
Scope: E-28 results, code and the main session's rubber duck. Method: recomputed the decision numbers
from the CSV files, checked for bugs that could hide a collapse, and ran an unfrozen reviewer check
(240 runs: M1 at 380k, 12,000 s, spikes up to 8 x N*, clients A and C, 6 seeds; plus a cap check). The
main session re-ran 6 of the cells and got identical numbers.
Findings:
1. FACT: the decision numbers reproduce (lowest ratios 0.893 late and 0.912 post). The pairing is
   exact. No code bug hides a collapse.
2. FACT (reviewer check): a lasting bad state exists at 1.0 x knee after a spike of 2 x N* or more,
   with client C. It is bimodal in the preemption rate. Controls at higher loads leave the good state
   by themselves. E-28 stopped one step short (spike at most 1 x N*) and its goodput rule had no power
   at this depth (about 20% loss).
3. FACT (reviewer check): client A recovers by failing sessions. A static cap of 12 or 16 clears the
   bad state.
4. INTERPRETATION: the main session's "no measurable band" and "plain thrashing above the knee" are
   wrong (Section 8.1 row 46). The R-14 toy sat close to its barrier and had no paired control.
5. INTERPRETATION: the metastability framing is a measurement result. The fix is known (a cap). The
   only open parts are an online trigger from the bimodal signal and K3; both are thin.
Verdict: K1 KILL CONFIRMED for the frozen grid. The claim "no collapse" is NOT CONFIRMED in general.
I-80 as a thesis direction: KILL CONFIRMED WITH SCOPE LIMITS.
Full write-up: `dossier/REVIEW_R-16_E-28_POSTRUN_2026-10-08.md`. Reviewer scripts:
`experiments/metastable_admission/reviewer_r16/`.

**R-17 Route search 3 with two Sonnet passes** (2026-10-08).
Scope: new environments and the stated limits of the newest 2026 papers, for PagedAttention-style
transfers and combinations (D-29, D-32, D-33). Candidates I-82 to I-94 plus 26 unregistered round 3
candidates.
Method: frontier map of the 2025 and 2026 systems programs and preprints. Round 1: Sonnet attack of six
candidates with its own searches (`dossier/REVIEW_R-17_SHORTLIST_2026-10-08.md`). Round 2: five
main-session probes. Round 3: a Sonnet agent generated and self-screened 28 candidates
(`dossier/R-17_ROUND3_SONNET_CANDIDATES_2026-10-08.md`). The main session opened the sources behind
the key kills (GVR 2604.22312, 2609.18849, 2609.28870 full text, Volcano #5998).
Findings:
1. FACT: every Mac-testable candidate has a direct 2026 match (paper, shipped option or pull request).
2. FACT: GVR already gives bit-exact DSA top-k; the DSA indexer weights are signed.
3. INTERPRETATION: the only open gap is I-83, and its premise is doubtful for structural reasons
   (LayerNorm, RoPE, Hadamard make block bounds loose).
4. INTERPRETATION: route search of this kind has a very low yield in 2026 (R-10, R-13, R-17).
Verdicts: I-83 WEAK (GPU only, premise check proposed). I-94 WEAK spare. All others KILL.
Full write-up: `dossier/ROUTES_R-17_2026-10-08.md`.

**R-18 Orphan-finding scan with a Sonnet harvest and a Sonnet attack** (2026-10-08).
Scope: measured waste in GitHub issues (15 repositories) and in 30 characterization papers that has no
fix yet. Candidates I-95 to I-102.
Method: issue search by reactions (main session); paper harvest with a fix search per finding
(`dossier/R-18_ORPHAN_FINDINGS_PAPERS_2026-10-08.md`); main-session screen; Sonnet attack of five rows
(`dossier/REVIEW_R-18_ORPHANS_2026-10-08.md`). The main session opened llama.cpp #22746, Megatron #7452
and 2609.25451.
Findings:
1. FACT: no orphan survives. Every candidate has a shipped fix, a 2026 paper, or is an earlier kill.
2. FACT (reviewer): three of the five checked headline numbers were misread by the harvest.
3. INTERPRETATION: reading reports of waste is a weak filter for new ideas; it is useful only to size
   a waste once an idea exists.
Verdicts: I-98 keep-alive policy WEAK. All others KILL.
Full write-up: `dossier/ORPHANS_R-18_2026-10-08.md`.

**R-19 Assumption removal with two Sonnet prior-art agents and a Sonnet attack** (2026-10-11).
Scope: 40 assumptions of AI infrastructure, 7 kept (I-103 to I-109).
Method: main-session list and triage; P1 checked by the main session; P2 to P7 by two Sonnet agents;
Sonnet attack of P2, P3, P4 (`dossier/REVIEW_R-19_SHORTLIST_2026-10-11.md`). The main session opened
KVShareArena (2609.10266, abstract and HTML) and the OLMo-2 RLVR revision list.
Findings:
1. FACT: each kept assumption is already breaking and already has 2025 or 2026 work.
2. FACT (reviewer): vLLM `keep` mode avoids re-prefill on weight updates; the claim in the brief that
   `clear_cache` forces re-prefill was wrong for current vLLM (SNIPPET).
3. FACT: KVShareArena covers cross-version reuse for distant pairs only.
4. INTERPRETATION: I-106 is the only question that is open, cheap and Mac-testable.
Verdicts: I-106 WEAK (MEASURE-FIRST), I-108 WEAK, others KILL.

**R-20 Seed-topic gap search with a Sonnet attack** (2026-10-16).
Scope: the user's three seed topics; I-110 to I-119.
Method: three main-session search rounds (broad, narrowed, targeted at survivors); primary pages opened
for HijackKV (HTML), 2605.01938 (HTML), the NDSS 2026 paper (PDF text), the NVIDIA Grace tuning guide, and
abstracts of 2609.06853, 2608.09225, 2608.01657, 2604.17249, 2606.28958, 2610.15721, 2610.02233 and
2605.26461. Then a Sonnet attack with fresh searches.
Findings:
1. FACT: topic 1 is a published pattern (dLoRA, Aegaeon, D-STACK, green contexts, LithOS, DP/TP switching).
2. FACT: topic 2 has an NDSS 2026 paper, a USENIX Security 2026 paper and more than ten 2025 to 2026
   preprints that cover collisions, hijacking, bit flips, pollution, side channels and cross-model reuse.
3. FACT: HijackKV proposes a deviation verifier but does not evaluate it.
4. FACT: on Grace superchips the GPU gets the module limit minus Grace power, and the CPU has priority.
5. INTERPRETATION (reviewer): a static CPU cap is a strong baseline for I-118; the likely output is a
   measurement note.
6. The reviewer voted I-114 WEAK, not KILL: re-encode cost for very large chunk stores is not measured.
Verdicts: I-118 MEASURE-FIRST (low), I-116 WEAK, others KILL (I-114 with a WEAK dissent).
Full write-up: `dossier/GAPS_R-20_2026-10-16.md`.

**R-21 Current-frontier systems review** (2026-10-17).
Scope: nine candidate non-consensus truths across agent systems, LLM inference, recovery,
verification, scheduling, KV state, kernel qualification, and coupled CPU-GPU systems.
Method: two independent literature passes and about 50 searches. Pass 1 mapped foundational work,
recent systems, production implementations, benchmarks, and workload evidence. Pass 2 used narrow
mechanism names, adjacent-domain terms, simple baselines, RFCs, repositories, and current issues.
No experiment ran because the two best gates need one NVIDIA GPU.
Findings:
1. FACT: SAGA and MARS already implement workflow-level and CPU-GPU agent scheduling.
2. FACT: Crab, Atomix, and Resume Means Resume cover broad cross-layer agent recovery contracts.
3. FACT: current vLLM and LMCache issues show wrong output, crashes, and long waits during KV recovery.
4. FACT: Mooncake RFC 3692 proposes most of a transactional snapshot path, but not its runtime path.
5. FACT: FlashInfer-Bench and FlashInfer v0.7 cover correctness-gated substitution and production autotuning.
6. INTERPRETATION: I-23 remains the best first experiment, but the contribution must be the output-aware oracle.
7. INTERPRETATION: I-120 has strong failure evidence, but its full-paper novelty has very high scoop risk.
Verdicts: I-120 MEASURE-FIRST. I-121 KILL. I-23 remains MEASURE-FIRST. I-106 and I-118 remain weak.
Full write-up: `dossier/FRONTIER_R-21_2026-10-17.md`.

**R-22 Continued frontier systems review** (2026-10-17).
Scope: I-122 to I-130, with I-120 carried from R-21 toward a three-to-five-gap stopping rule.
Method: two independent literature passes. Pass 1 mapped transitions and candidate beliefs. Pass 2
searched narrow mechanisms, adjacent domains, simple baselines, current repositories, issues, and roadmaps.
No experiment ran.
Findings:
1. FACT: current vLLM issues show wrong output and stale state after live streaming-session mutation.
2. INTERPRETATION: I-122 needs one version boundary across more than token and KV state.
3. FACT: vLLM-Omni states that streaming ASR has opaque state that token replay cannot reconstruct.
4. FACT: Execution-State Capsules captures heterogeneous model state, but evaluates one-device restore.
5. INTERPRETATION: I-123 needs an external input and output commit cut, not only a model-state snapshot.
6. FACT: direct work already covers tentative-token commit, diffusion slot batching, cancellation leases,
   generated-kernel isolation, elastic data progress, recurrent checkpoints, and portable compiler caches.
Verdicts: I-122 and I-123 MEASURE-FIRST. I-124 to I-130 KILL. The review stopped at three total
survivors: I-120, I-122, and I-123.
Full write-up: `dossier/FRONTIER_R-22_2026-10-17.md`.

**R-23 E-31 different-model pre-run review** (2026-10-17).
Scope: E-31 and I-122.
Method: Sonnet read the frozen plan and implementation. It did not execute code or edit files.
Findings:
1. FACT: The version guard and invariant check use the same predicate.
2. FACT: A policy-name branch encodes the expected preemption result.
3. FACT: A policy can pass while it drops valid work or stalls permanently.
4. FACT: Model constants determine the rebuild-cost result.
5. INTERPRETATION: E-31 cannot falsify the proposed mechanism.
Verdicts: E-31 INVALID DESIGN. I-122 requires a current-source audit before any replacement test.
Full write-up: `dossier/REVIEW_E-31_PRERUN_2026-10-17.md`.

**R-24 I-122 current-source audit and pivot review** (2026-10-17).
Scope: I-122, E-31, current vLLM session-state fixes, and I-131.
Method: The Docker audit pinned current main, queried six issues and six pull requests, scanned five
source paths, and counted the selected scheduler test surface. Two targeted literature passes attacked
stateful testing, model-based testing, inference-engine bug work, and adjacent fuzzing.
Findings:
1. FACT: PRs 59103, 58259, 58668, and 58778 use narrow fixes for distinct session-state failures.
2. FACT: Issue 60744 remains open with an open narrow fix PR.
3. FACT: The audited paths have no session generation, revision, incarnation, epoch, or commit field.
4. INTERPRETATION: Field absence does not show that a global version mechanism is necessary.
5. FACT: Seven selected current test files have 185 test functions and 54 parameter grids.
6. FACT: Those files have no Hypothesis import or random-action use.
7. INTERPRETATION: A real stateful metamorphic screen is stronger than another abstract mechanism model.
Verdicts: I-122 PIVOT. I-131 MEASURE-FIRST with medium-low public-gap confidence. E-32 is predeclared.
Full write-up: `dossier/SOURCE_AUDIT_I-122_2026-10-17.md`.

**R-25 E-32 different-model pre-run review** (2026-10-19).
Scope: E-32 and I-131.
Method: Sonnet read the frozen design, source audit, audit data, and journal. It ran no code.
Findings:
1. FACT: Two corpus cases lacked stable merged fix controls.
2. FACT: The action alphabet and oracles mapped directly to the known bugs.
3. FACT: The baselines lacked equal oracle and execution budgets.
4. FACT: Several oracles lacked an independent reference model and complete legal outcomes.
5. INTERPRETATION: E-32 could pass by encoding its target corpus.
Verdicts: E-32 AMEND BEFORE IMPLEMENTATION. I-131 remains MEASURE-FIRST. Run E-33 first.
Full write-up: `dossier/REVIEW_E-32_PRERUN_2026-10-17.md`.

**R-26 E-32 amendment review** (2026-10-19).
Scope: The amended E-32 design, E-33, and I-131.
Method: Sonnet read the amended corpus, decision rules, prior reviews, audit data, and journal.
It ran no code.
Findings:
1. FACT: The action alphabet cannot express two frozen corpus bugs.
2. FACT: The action and oracle contract still does not exist.
3. FACT: The decision rules leave key terms and outcomes undefined.
4. FACT: E-32 and E-33 contain conflicting corpus and environment specifications.
5. INTERPRETATION: The amendment does not remove the circular known-bug recall test.
Verdicts: E-32 AMEND BEFORE IMPLEMENTATION. E-33 blocked as written. I-131 remains unproven.
Full write-up: `dossier/REVIEW_E-32_AMENDMENT_2026-10-19.md`.

**R-27 I-131 paper audit** (2026-10-19).
Scope: Four vLLM regression tests, their production fixes, E-32, E-33, and I-131.
Method: Read every upstream test and patch. Map visible actions. Run full-history blame in Docker.
Findings:
1. FACT: The four traces need 4, 10, 8, and 8 visible actions.
2. FACT: Only one trace fits a depth-six exhaustive baseline.
3. FACT: The E-32 alphabet lacks the truncating-update and accepted-draft actions.
4. FACT: No blamed fault line came from an earlier narrow session-state fix in this corpus.
5. INTERPRETATION: The corpus shows interaction faults during a feature rollout, not recurrence after fixes.
6. INTERPRETATION: The contribution remains generic stateful testing plus vLLM-specific actions.
Verdicts: I-131 KILL as thesis. E-32 KILL-DESIGN. E-33 CANCELLED BEFORE RUN.
Full write-up: `dossier/PAPER_AUDIT_I-131_2026-10-19.md`.

**R-28 I-123 scoop audit and recovery-cut pivot** (2026-10-19).
Scope: I-123, current full-duplex serving, inference checkpointing, world-model migration, and
stream-processing recovery.
Method: Two literature passes. Pass 1 mapped complete state capture, KV recovery, durable GPU deltas,
duplex ledgers, and exact state movement. Pass 2 attacked the narrowed engine-crash recovery cut.
Findings:
1. FACT: Current vLLM-Omni owns input, response, playback, history, lease, and model state in one session.
2. FACT: Its reconnect journal survives a transport drop, not engine loss.
3. FACT: Open PR 8485 adds ordered per-input acknowledgements after caused output.
4. FACT: Concordia appends committed GPU-state deltas to CPU-visible storage and recovers in about 1.5 s.
5. FACT: WorldMove measures a 375.6 ms block and 1.67 GB movable state with 71% to 78% dirty data.
6. INTERPRETATION: The old 200 ms replay oracle is impossible for one world-model replay block.
7. INTERPRETATION: A remaining gap may exist only at the durable cut between model and protocol state.
8. INTERPRETATION: The strongest threat is standard stream checkpointing plus existing inference snapshots.
Verdicts: I-123 PIVOT. I-132 DESIGN-REVIEW. E-34 is predeclared but not implemented or run.
Full write-up: `dossier/SCOOP_AUDIT_I-123_2026-10-19.md`.

**R-29 E-34 independent pre-run review** (2026-10-19).
Scope: I-132, the E-34 design, its baselines, its decision rules, and its Docker workload.
Method: An independent Sonnet review attacked novelty, simple baselines, external validity,
oracle quality, crash coverage, feasibility, and publication value.
Findings:
1. FACT ABOUT THE DESIGN: P1 combines a write-ahead log, periodic snapshots, an atomic manifest,
   and an idempotent output sink.
2. INTERPRETATION: This is standard checkpoint composition for a sequential worker.
3. FACT ABOUT THE DESIGN: B2 is the checkpoint-interval endpoint `K=1`. B1 is the large-`K` endpoint.
4. FACT ABOUT THE DESIGN: The workload is deterministic. Playback-dependent state is dead metadata.
5. INTERPRETATION: The workload removes the possible new constraint of nondeterministic model replay.
6. FACT ABOUT THE DESIGN: The replay target, supported workload, and sourced recovery budget are absent.
7. INTERPRETATION: The frozen Docker sweep measures selected constants, not an external workload fact.
8. HYPOTHESIS: A real GPU nondeterminism replay test could expose a different and narrower problem.
Verdicts: I-132 KILL. E-34 KILL-DESIGN. Do not implement or run it.
Full write-up: `dossier/REVIEW_E-34_PRERUN_2026-10-19.md`.

**R-30 Scoop check before the Mac test** (2026-10-19).
Scope: I-106 (the planned Mac premise check), and the Mac-only backups I-59 and I-60.
Method: main session. Read KVShareArena v2 (HTML) and CacheReforge (abstract and HTML). Searched the
last two weeks for KV reuse across weight updates, shared-prefix priority inversion, and late-binding
routers. Read vLLM issue 59907 and PR 59963 with `gh`. Checked the OLMo-2 RLVR revisions and sizes with
the Hugging Face API. Nothing was downloaded and nothing ran.
Findings:
1. FACT: `allenai/OLMo-2-0425-1B-RLVR1` has 13 step branches (step_200 to step_2600) and `main`, one
   2.97 GB safetensors file each, Apache-2.0. The Mac has 16 GB RAM and 20 GB free disk.
2. FACT: KVShareArena v2 (2026-09-27) tests six distant producer pairs, not adjacent RL steps.
3. FACT: CacheReforge (2609.30884) implements layerwise mixed-version KV with drift-based bounded
   recompute for continual LoRA updates.
4. INTERPRETATION: I-106 reduces to an evaluation of CacheReforge on full-weight RL updates.
5. FACT: CacheRoute (2608.19677) and llm-d cover the delay-versus-locality routing trade-off of I-60.
6. FACT: vLLM issue 59907 shows a different priority inversion and runs on the vLLM CPU backend.
Verdicts: I-106 KILL. I-60 KILL. I-59 WEAK, Mac-testable. Not reviewed by a second model; the kills
rest on primary sources, not on an experiment.

**R-31 E-35 Sonnet pre-run review** (2026-10-19).
Scope: the frozen E-35 design and the code claims behind I-59.
Method: an independent Sonnet agent re-checked the code claims at the pinned vLLM and SGLang commits
and attacked the design. It ran no experiment.
Findings:
1. FACT: vLLM never makes a waiting request wait for another request's in-flight prefix.
2. FACT: the running loop takes the token budget in admission order and ignores priority. The main
   session confirmed this in `scheduler.py` lines 712 to 735.
3. INTERPRETATION: the positive control could not pass in its named cell, and C4 is not an
   inheritance bound; the rules had no outcome for "K1 holds, K2 fails".
4. FACT: the CPU platform turns async scheduling off, so the synchronous `Scheduler` is the class the
   CPU backend runs.
5. FACT: SGLang rejects priority scheduling with the cache-aware policies that hold its shared-prefix
   wait.
Verdicts: DO_NOT_RUN_CODE_DECIDES. Run a minimal trace check instead (amendment A1).
Full write-up: `dossier/REVIEW_E-35_PRERUN_2026-10-19.md`.

**R-32 E-35 Sonnet post-run review** (2026-10-19 to 2026-10-20).
Scope: amendment A2, run 2, and the conclusion.
Method: an independent Sonnet agent re-ran the check in the pinned image (same output hash), and ran
its own mutation, allocation-label and paired scripts in the session scratchpad. Those scripts were
lost when the scratchpad was cleared; the main session re-implemented two of them as a supplement.
Findings:
1. FACT: A2 is a scenario fix; it could only add assertions.
2. FACT: the skip detector misses waits from a failed allocation (shown by a mutation test).
3. FACT: all 310 `other` waits are H's own failed allocations.
4. FACT: in 600 paired scenarios a shared prefix never made H start or get its first token later.
5. INTERPRETATION: priority inversions exist, but none depends on a shared prefix.
Verdicts: CONFIRM_WITH_SCOPE_LIMITS. I-59 KILL, worded as "no shared-prefix dependency".
Full write-up: `dossier/REVIEW_E-35_POSTRUN_2026-10-20.md`.


**R-33 Fresh frontier pass** (2026-10-20).
Scope: eight transitions not in the registry (I-133 to I-140), and a re-check of I-120 and I-23.
Method: main session only. About 20 web searches in two passes. Primary pages opened for the closest
works (arXiv abstracts, NVIDIA blog, nvcf issues 1183 and 2299, vLLM PRs 53298 and 62282 through `gh`).
No Sonnet review: no experiment ran.
Findings:
1. FACT: EEP, SGLang, vLLM and Ray ship elastic wide-EP recovery.
2. FACT: Belayer, RobustRL and DSec cover agentic RL recovery, including sandbox state.
3. FACT: NVIDIA nvcf issue 2299 and PR 2300 implement multi-GPU C/R with imported-memory remap.
4. FACT: Dynamo Snapshot and GPU Memory Service skip KV and separate weights.
5. FACT: vLLM PR 53298 implements the I-120 mechanism; PR 62282 is merged.
6. FACT: 2606.07581 defines kernel contracts without evaluation.
7. INTERPRETATION: broad sweeps on this frontier now return direct matches (8 of 8). The next useful
   information must come from a GPU measurement.
Verdicts: I-133 to I-140 KILL. I-120 KILL as research. I-23 MEASURE-FIRST, unchanged.
Full write-up: `dossier/FRONTIER_R-33_2026-10-20.md`.

**R-34 E-36 and E-30 Sonnet pre-run review** (2026-10-20).
Scope: the E-36 package (I-23 Test A Session 1) and the frozen E-30 package (I-118), before any run.
Method: an independent Sonnet agent, read-only. It ran both unit-test suites and read the vLLM
v0.32.0 source. The main session re-checked the key code claims in that source. No experiment ran.
Findings:
1. FACT: an unsupported `--linear-backend` falls back to normal kernel selection with only a
   warning. Some E-36 candidates were aliases of the reference, and `triton_unfused` cannot run.
2. FACT: offline vLLM runs are deterministic only with `VLLM_ENABLE_V1_MULTIPROCESSING=0` or batch
   invariance. The E-36 floor check could fail by construction.
3. FACT: a Wilson 95% upper bound of at most 1.5% for 0 flips needs at least 253 prompts, so
   STOP-FLAT was unreachable as written. The main session corrected one reviewer number: at most 2%
   needs n >= 189, not 188.
4. FACT: the E-30 prefill test hit the prefix cache, the hwmon glob missed the `device/` layout, and
   the CPU caps ran at the default limit, not at the binding limit.
5. INTERPRETATION: the equal-weight E-30 KILL 2 objective decides KILL 2 by construction, because the
   oracle gain comes mostly from the CPU term in decode.
Verdicts: AMEND_THEN_RUN for both (E-30 DO_NOT_RUN as written). Amendments A1 were applied to both
packages before any run (E-36 README Section 9, E-30 README Section 8).
Full write-up: `dossier/REVIEW_E-36_E-30_PRERUN_2026-10-20.md`.

**R-35 GPU-plan freshness check and local preparation** (2026-10-20, this session, D-49).
Scope: the GPU-gated live ideas (I-23, I-118, I-120, I-83, I-18) and the preparation of their sessions.
Method: main session. GitHub API reads of the vLLM KV-load-failure issues and PRs, RFC 35780 and PR
42461; arXiv abstracts of 2609.08450 and 2607.11976; reading vLLM v0.32.0 and main `3897f95c` source for
backend switches; web searches for small DSA models, GH200 root access, prices and CE-collective work.
Findings:
1. FACT: vLLM issues 61768, 61905 and 61912 come from CPU scheduler-level probes with a mock connector;
   their fix directions are the I-120 mechanism. The maintainers propose request-level recovery (RFC
   35780, PR 42461). This supports the R-33 I-120 kill.
2. FACT: LiteTopK reports indexer score concentration; no small open DSA model exists.
3. FACT: vLLM v0.32.0 exposes `--linear-backend`, `--moe-backend`, `--attention-backend`.
4. FACT (SNIPPET): Lambda 1 x H100 about $4.29 per hour, 1 x B200 about $6.99 per hour; GH200 price not
   found; Lambda's MIG guide shows `sudo nvidia-smi` on a 1 x GH200.
5. No new direct work on I-18 or I-118 was found.
Verdicts: no status change by R-35 alone (I-120 is already KILL by R-33). Outputs: the E-36 and E-30
packages and `dossier/GPU_RUNBOOK_2026-10-20.md`.

**R-36 Limitations mining** (2026-10-20).
Scope: the stated limitations and future work of all 496 cited arXiv papers; eight follow-up gaps.
Method: main session. Downloaded arXiv HTML for 496 papers into the scratchpad (490 full text). A script
extracted limitation, future-work, open-problem and scope sections plus cue sentences. Read the output
for the 372 papers from 2025 and 2026 that state any limitation. Checked eight gaps with targeted reads
and fresh web searches. No Sonnet review: no experiment ran.
Findings:
1. FACT: most stated future work is multi-node scale-out, more models or hardware, or upstream integration.
2. FACT: Dynamo, Belayer and LUMEN each leave crash-time KV reuse open; a committed-length journal solves it.
3. FACT: a public repo computes exact Shapley prefix-cost attribution (2026-08-25).
4. FACT: TBIK leaves quantized TP invariance open; no public FP8 result was found in one search.
5. INTERPRETATION: named mechanism gaps are closed by upstream or other groups within weeks. Limitation
   sections are not a productive gap source for this thesis.
Verdicts: I-141, I-142, I-144 to I-148 KILL. I-143 WEAK.
Full write-up: `dossier/LIMITATIONS_R-36_2026-10-20.md`.

**R-37 Algorithmic-optimization sweep** (2026-10-21, D-54).
Scope: 16 candidates (I-149 to I-164) from four algorithmic lenses, with the PyTorch Conference NA 2026
schedule as an extra source.
Method: main session. Read the hot paths of SGLang a3d00eff, vLLM 90c4a1a and PyTorch 6dbae335 (DCP)
at current main. Pass 1: code facts plus literature searches for each candidate. Pass 2: targeted searches
for the candidates left after pass 1, opened PEEK, NanoCP, vLLM RFC 47582, ForgeMegakernel and Event
Tensor. E-37 sized two CPU paths. R-38 (Sonnet) is the different-model review.
Findings:
1. FACT: SGLang drops LPM and HRRN order above 128 waiting requests ("Arbitrary" limit) and rebuilds an
   eviction heap of all leaves on each `evict()` call.
2. FACT: vLLM has no cross-turn tokenization cache and hashes all prompt blocks for each request in the
   engine input thread. RFC 47582 (open) proposes incremental encoding.
3. FACT: SGLang ships an LP solver for redundant-expert dispatch; vLLM spreads MLA KV source ranks; the
   vLLM GPU runner keeps persistent penalty counts; PyTorch DCP read planning is a sweep line.
4. INTERPRETATION: the visible algorithmic slow paths in the main engines are fixed, being fixed, or covered
   by a 2026 preprint. Three theory checks (proportionate flow shop, large-batch convergence, multicast
   ingress) close whole idea families without experiments.
Verdicts: I-149 to I-160, I-162 and I-164 KILL. I-161 KILL as research, engineering PR candidate. I-163
WEAK, MEASURE-FIRST, GPU. (R-38 changed I-161 and I-163 to KILL.)
Full write-up: `dossier/ALGORITHMIC_R-37_2026-10-21.md`.

**R-38 Sonnet hostile review of R-37 and post-run review of E-37** (2026-10-21).
Scope: E-37 code and conclusions; the theory KILLs I-154 and I-158; the survivor I-163; missed candidates.
Method: Claude Sonnet, read-only, with the R-37 source clones and fresh web searches. The main session
checked its key claims: SGLang default cache path (`registry.py`, `environ.py:709`, Rust `full.rs`), PRs
38371, 38454, 41690 and 39627 with the GitHub API, the Hazy `scheduler.py` source, and the Cohere blog.
Findings:
1. FACT (verified): SGLang PRs 38371 and 38454 already add persistent lazy eviction heaps; R-37's "None found"
   was wrong. The default path is the Rust core, which also refills the heap from all leaves.
2. FACT (verified): Hazy Megakernels has a list scheduler with a critical-path priority function; Cohere
   measured about 10% from cost-based scheduling on a dense model and dropped it for MoE.
3. FACT (reviewer brute force): the proportionate flow shop makespan is order-invariant. INTERPRETATION
   (reviewer simulation): Johnson's rule gains 3 to 14% at n of 8 to 128, so I-154 dies on its objective,
   not on scale.
4. FACT (reviewer, search-only): DySHARP uses dynamic in-switch multicast for MoE; I-158 dies on prior art.
5. INTERPRETATION: the replay test for I-163 was circular. No missed candidate survives.
Verdicts: E-37 CONFIRM_WITH_SCOPE_LIMITS. I-161 KILL. I-163 KILL. I-154 and I-158 KILL with corrected reasons.
Full write-up: `dossier/REVIEW_R-37_SONNET_2026-10-21.md`.

**R-39 Sonnet pre-run review of E-36 amendment A2** (2026-10-22).
Scope: E-36 A2 (Runpod efficiency changes and Phase B), `session1.sh`, `run_cell.py`, `extras.py`,
`experiments/runpod/`, runbook Section 2a. Runpod facts from a read-only Runpod MCP session.
Method: Claude Sonnet, read-only; `sh -n` and the unit tests; no GPU or cloud work.
Findings:
1. FACT: the runbook kept results and code on the container disk, which a Pod stop wipes, so resume
   could not work (blocker). Fixed: volume disk at `/workspace`.
2. INTERPRETATION: an SSH shell does not get the Pod environment, so AUTO_STOP could do nothing
   (blocker). Fixed: three named variables read from `/proc/1/environ`.
3. FACT: a profiler crash could lose a measured run, and AUTO_STOP ran only on the normal path. Fixed.
4. INTERPRETATION: 200 ms telemetry was an untested new condition for Phase A. Fixed: 1 s, UTC, field
   fallback.
5. FACT: the kernel-name alias check was fragile. Fixed: GEMM and MoE kernels only, a weak signal.
6. Minor items (B1 label, B2 ranking, caps across restarts, power-cap bit, KV size): all fixed.
7. FACT: Runpod H100 SXM Secure Cloud $3.99/h, stock LOW in one data center; a $0 balance terminates a
   Pod without a network volume.
Verdicts: E-36 A2 AMEND_THEN_RUN; fixes applied and tested on the Mac. Phase A unchanged.
Full write-up: `dossier/REVIEW_E-36_A2_PRERUN_2026-10-22.md`.

**R-42 Sonnet post-run review of E-36 Session 1 (H100)** (2026-10-26).
Scope: the H100 results, README Section 13, and the main-session claims. Method: Claude Sonnet,
read-only, its own checks of `summary.json`, `extras_mac.json`, `kernels.json` and the outputs.
Findings: (1) FACT: INCONCLUSIVE is what `compare.py` computes; flip range 87.5% to 95.3%. (2) FACT: a
literal STATIC-WINNER reading fails only at dense batch 32. (3) FACT: R is the fastest eligible backend
at every frozen batch size; Marlin W8A16 beats R in the MoE cell at batch 4 to 32 by 1% to 5%. (4)
INTERPRETATION: "close to the KILL condition" is not justified; exact match flags any numerical change
(eager flips 233 and 238 of 256); margins at divergence are 0.25. (5) INTERPRETATION: a tolerance
contract is legitimate only as a new, preregistered Test A2. (6) FACT: B0 to B3 confirmed; R's batch
invariance (8 against 64) unmeasured. (7) FACT: cold_vs_warm in B1 reflects batch composition; the Phase
A MoE Marlin warm-up is a nondeterminism flag. (8) FACT: dense deep_gemm and R share the GEMM kernel but
use different activation-quantization kernels. (9) FACT: validity holds.
Verdicts: E-36 Session 1 CONFIRM_WITH_SCOPE_LIMITS. I-23 stays MEASURE-FIRST; next step D-61.
Full write-up: `dossier/REVIEW_E-36_SESSION1_POSTRUN_2026-10-26.md`.

**R-43 Sonnet pre-run review of E-38 (I-23 Test A2)** (2026-10-26).
Scope: the E-38 package and the first dry run. Verdict: AMEND_THEN_RUN. Main findings: the session ran
cell D speed tests before cell M (the only GO path); A2 was close to vacuous (same FP8 weights); teacher
forcing misses the decode path; speed power is limited (n = 2 against 2 processes, 3% band); code and
README differed in four places; the budget was underestimated. All handled in E-38 amendment A1.
Full write-up: `dossier/REVIEW_E-38_PRERUN_2026-10-26.md`.

**R-44 Sonnet post-run review of the E-38 rerun** (2026-10-27). Verdict: CONFIRM_WITH_SCOPE_LIMITS. KILL is
what the frozen rule gives; A2 fails on top-1 as well as the tail-dominated KL; "not BETTER" means a 3%
gain was not shown with 2 against 2 processes; do not claim beyond H100, Qwen3 FP8, GSM8K.

**R-45 Sonnet pre-run review of E-39** (2026-10-27). AMEND_THEN_RUN (power, V1, V3, framing, budget); all
fixed in E-39 amendment A1. Write-up: `dossier/REVIEW_E-39_PRERUN_2026-10-27.md`.

**R-46 Sonnet post-run review of E-39** (2026-10-27). CONFIRM_WITH_SCOPE_LIMITS. The rule gives INCONCLUSIVE
(4, 8, 16); only batch 8 has a lower bound above 1; batch 32 is a tie; round 1 was a cold low outlier;
closing I-23 for Hopper block-FP8 (D-63, decided by the user) is justified as a decision, not as a proof.

**R-47 Sonnet pre-run review of E-40** (2026-10-27). AMEND_THEN_RUN: broken NVFP4 MoE alias regex (fixed),
budget off by 2x (scope cut), R_2 order, timeouts, AOT gate. Write-up: `dossier/REVIEW_E-40_PRERUN_2026-10-27.md`.

---

## 7. Prior-art ledger

Cumulative. Add rows, never remove. Check levels are defined in section 0.4. Dates are
publication or issue-open dates where known.

`references/` lists every arXiv paper in this ledger and in the dossiers, by topic, with titles and
authors from arXiv. Run `references/_tools/build_references.py` again after you add rows here.

### 7.1 Numerics, determinism, conformance, kernel dispatch

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| Integer Alibi, arXiv 2608.13756 | 2026-08-13 | CUTLASS vs Triton INT8 in vLLM: 0/8, 0/16, 0/64 sequence agreement. Divergence is in scale application and output rounding, not the exact accumulator. One FP8 pair (8.2% to 52.7% of elements differ as K grows). Disclaims other GPUs, MoE, NVFP4, multi-GPU | Defines I-23's open cells | checked |
| Deterministic LLM Inference Across GPU Kernels, 2609.00363 | 2026-08-25 | Tolerance-based conformance is blind to five epilogue faults (each at most one bf16 spacing). Power-of-two scales give bitwise agreement. Conformance suite | Kills "allclose is insufficient" as a contribution | checked |
| DriftBench, MLSys 2026 | 2026 | 236,985 prompt-response pairs, 105 configs (5 models, 4 GPU platforms, 3 frameworks, 3 precisions). Hardware and precision drift predictable (R2 0.909 and 0.763). Framework and model drift idiosyncratic (R2 under 0.48). Blocked an upgrade where 23.85% of safety prompts flipped | Closest high-level threat and support | search-only |
| LLM-42, 2601.17768 | 2026-01-25 | Fast path plus verify-rollback determinism, reuses existing kernels | Mechanism threat | checked |
| MarginGate, 2605.30218 | 2026-05-28 | Margin-triggered verification, 18.56% trigger rate, 2.23x lower latency than always-on LLM-42 (Llama-3.1-8B), 100% deterministic | The doc's logit-margin baseline is a published system | checked |
| Greedy Decoding Is Not Precision-Invariant, 2609.26621 (TMLR) | 2026-09-22 | BF16 vs FP16 diverges on 59 to 82% of prompts. Margin-triggered FP32 lm_head: plus 22 to 36 points agreement on A10G at under 4% overhead (batch 4 or less), weaker at larger batch and under FP8 | H2 published. FP8 is the stated open part | checked |
| Accelerating Mitigation of Nondeterminism Across GPU Architectures, 2609.25624 | 2026-09-22 | Fixed-order fused-upcast GEMM, bitwise identical linear layers across Ampere, Ada, Hopper, 1.17x to 3.1x faster than prior deterministic SOTA | Cheap static conservative baseline | checked |
| Taming Bitwise Behavior in GPU Kernels with Tensor Core, 2609.11356 | 2026-09 | GEMM reduction-order descriptor, black-box reconstruction, compiler-enforced balanced tree, static bitwise-equivalence checkers (PTX and GCN). 19 of 27 kernels within 10% of free order on GB300 and H100 | Fingerprinting and static equivalence taken | checked |
| FPRev, arXiv 2411.00442 (ATC 2025) | 2024-11 | Reveals summation order of black-box functions. Tensor-core trees: 5-way V100, 9-way A100, 17-way H100 | Prior art for fingerprinting | search-only |
| TBIK, 2511.17826 (ICML 2026) | 2025-11-21, v2 2026-05-29 | TP-invariant matmul and reduction primitives, bitwise equal across TP sizes | TP invariance solved for BF16 | checked (abstract) |
| CoRun, 2608.14376 | 2026-08 | Padding for deterministic inference | Crowded | search-only |
| HEAL 2606.21023, Bit-Exact AI Inference Verification 2606.00279 | 2026-06 | Reproducible and verifiable inference | Crowded | search-only |
| Same Request, Different Answer, 2609.04748 | 2026-09-04 | Prefix cache on changes agent trajectory in 36.2% of episodes (16-bit) and 75.0% (4-bit), 80 episodes. Cache off is bit-identical. Cached and recomputed paths still differ from each other | Cache divergence measured, no mitigation | checked |
| Silent Hyperparameter, 2605.19537 | 2026-05-19 | Backend alone shifts benchmark scores up to 16.6 points across five engines. Isolates prefix caching, CUDA graphs, custom kernels, logit defaults | Feature attribution partly covered | checked |
| What We Observe as LLM Behavior Can Be a Side-effect of Inference Backend, 2608.04714 | 2026-08-05 | About 39% of observable variability comes from backend | Context | checked |
| The Illusion of Equivalence, 2604.15409 | 2026-04-16 | KV-cached vs cache-free FP16 diverges, 100% token divergence rate on three models, FP32 eliminates flips. No tool | Supports premise | checked |
| Contract-Grade Verifier, 2608.12700 | 2026-08-13 | 12 adversarial gates, kernel level only. 39.5% of 2,638 generated kernels had severe violations. Standard test accepted 1,487 the verifier rejected | Kernel-level only | checked |
| Correctness Illusion, 2606.20128 | 2026 | Fixed-shape allclose certifies buggy kernels. Seeded fuzzing catches them | Cited by handoff | doc-only |
| FlashInfer release-quality RFC #3605 | 2026 | Seven oracles per op. About 31% of bugs in MoE, about 36% in low-precision GEMM plus MoE, about 60% of fixes in Python dispatch and validation. Cross-backend convention auditor caught a cuDNN bf16 to fp16 miscompute on Hopper | Bugs live in dispatch and low precision | checked |
| FlashInfer v0.7 autotuner | 2026-09-22 | Ranks candidates on latency, `MeasurementPolicy` eager vs CUDA graph, no numerical check described | Gap for I-23 | checked |
| DA-MoE, 2607.23099 | 2026-07-25 | Routing-histogram kernel dispatch for fused MoE, 1.16x to 1.29x geomean, no correctness discussion | Perf-only dispatch exists | checked |
| vLLM #55238 | 2026-09-03 | Open. B200 and B300 only: CUDA graph replay changes greedy output for gemma-4-26B-A4B-it, up to 7.47 nats logprob delta | Live evidence | checked |
| vLLM #55649 | 2026-09-07 | Open. NVFP4 gated-MoE end-padding corrupts Gemma-4-26B at TP2 on B200. Root cause found, fix PR #48624 unmerged | Live evidence | checked |
| vLLM Q3 2026 roadmap #48168 | 2026-07-09 | "backend selection is deterministic, debuggable, and model-aware", release-gating accuracy tests | Upstream heading toward I-23 | checked |
| vLLM-omni RFC #7924 | 2026-09-21 | Backend registration protocol, correctness out of scope | Context | checked |
| LAMP 2601.21623, BGSS (AISTATS 2026) | 2026 | Selective higher-precision recompute by numerical risk | Kills generic selective precision | doc-only |
| Ekka, 2606.04594 (ICML 2026) | 2026-06-03 | Differential debugging against a reference oracle. 4 previously unknown silent errors confirmed in production frameworks. Snippet: root causes framework 30.6%, model 25.5%, kernel backend 24.5%, precision 19.4% | Differential testing of serving exists | checked (abstract) |
| GRIEF, 2605.11202 | 2026-05-11 | Greybox fuzzer with log-prob replay. 15 vulnerabilities in vLLM and SGLang, 10 confirmed, 2 CVEs, including KV-cache isolation failures | Fuzzing of serving state exists | checked |
| A First Look at Bugs in LLM Inference Engines, 2506.09713 | 2025-06 | 432 functionality bugs. About 23% configuration-related | Context | search-only |
| Equivalence Checking of ML GPU Kernels, 2511.12638 | 2025-11 | Formal equivalence checker for kernels | Context | search-only |
| SiliconBench, 2609.19169 | 2026-09 | Speed, memory and fidelity for serving on Apple unified-memory desktops | Mac-native niche, off career target | search-only |

### 7.2 Hybrid state reuse, prefix caching, recovery

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| Marconi, 2411.19379 | 2024-11 | First prefix caching for hybrid LLMs, admission and eviction policies | Foundational | search-only |
| Sparse Prefix Caching, 2605.05219 | 2026-04-17 | Sparse exact recurrent checkpoints, resume from deepest, O(NM) DP for placement. Exact outputs | Owns checkpoint placement | checked |
| HYPIC, 2607.01299 | 2026-07-01 | Position-independent caching for hybrid attention: segment-cumulative transition operator, seam window that propagates hidden states, 3.25x TTFT | Reaches the "propagate hidden states" insight independently | checked |
| ReplaySSM (vLLM #49232, Dao AI Lab blog) | 2026-07-20 | Cache SSM inputs, not state. 1.17x to 1.67x decode, up to 2.21x speculative | Orthogonal, kernel level | doc-only |
| SGLang Unified Radix Cache blog | 2026-08-11 | FULL, SWA, MAMBA on one tree. Reusable boundary advances only when all component validators accept | Shipped reconciliation rule | checked |
| Runtime Observability for Heterogeneous Attention Memory, 2608.05863 | 2026-08-06 | Formal contract, Lean 4, four cache types including recurrent, always-on probe under CUDA graphs, 12.4M entry reads | Direct threat to detector ideas | checked (abstract) |
| Validating Hybrid-State Cache Recovery for GLM-5.3-Flash, 2609.15030 | 2026-09-14 | Recovery mismatch (scheduler credited one fewer token), generation agreement 34/36 to 36/36. No general condition or tool | Existing validation | checked |
| Shared KV Caching for Replicated 27B Inference, 2609.15021 | 2026-09-14 | Missing CUDA stream dependency in raw-pointer fallback, found by controlled byte tests | Context | checked |
| DASC, 2608.30386 | 2026-08 | Compresses KDA recurrent checkpoints 2.63x, TTFT down 42.6%, throughput up 68.4% | Adjacent to I-39 | checked |
| LinearKV 2608.11231, AVMP 2605.22416 | 2026 | Position-independent caching for hybrids, asymmetric virtual memory paging | Context | search-only |
| vLLM #53041 | 2026-08-20 | Discussion. Proposal A tiered checkpoints. Proposal B recompute backfill. Zero comments at fetch | Prior claim of I-16, subject of I-19 | checked |
| vLLM #52773 | 2026-08-18 | Load-backfill from external store, measured about 80x cheaper than recompute | Closes I-16 performance | doc-only |
| vLLM #57266, #55600, #56655 | 2026-09-17, 09-06, 09-13 | Indexing, unit-mismatch and cache-key bugs (details in I-19) | Different bug classes | checked |
| vLLM #55697, #52959 | 2026-09-07, 08-19 | Application-directed prefix checkpoints, internal Mamba checkpoints | Scoop risk for I-15 | doc-only |
| vLLM #45238, #45702, #48455, #53786, #54661, #47812, #49537, #51771, #55196, #54106, #53670, #53504 | 2026 | Hybrid cache and recovery issues, see I-13 | Evidence the area is live | doc-only |
| vLLM hybrid SSM disaggregation blog | 2026-04-21 | NIXL dual descriptor views, 3-descriptor conv transfer, heterogeneous TP. Says nothing on live migration. GDN on roadmap | Kills I-06 novelty | checked |
| AI21 "One token to corrupt them all", PyTorch hybrid-models blog, Kimi K3 hybrid caching | 2025 to 2026 | Mamba corruption bug, hybrid support, hybrid cache semantics | Context | doc-only |

### 7.3 Collectives and compute-communication overlap

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| NCCL source v2.32.3 (12df1a1) | 2026-09-17 | CE gate, per-call `ncclCollConfig_t`, tuner v6, no load sensing | Core evidence for I-18 | full-text (relevant files) |
| NCCL 2.28 CE and device API blog | 2025 | CE multicast AllGather about 780 vs 620 GB/s at 4 GB on 8 Blackwell GPUs. Selection path unstated | CE ahead in NVIDIA data | checked |
| NCCL Q2 2026 roadmap #2090 | 2026 | Zero-SM AllGather, SM-initiated CE collectives explored | Scoop risk | checked |
| NCCL #2360 | date uncertain (page shows 2024-08-19 but cites NCCL 2.26.2, which is later) | Hopper NVLS BF16 AllReduce results vary with multicast allocation history and across H800 hosts | Evidence for I-34 | checked |
| Collective Communication for Distributed LLM Systems (survey), 2608.15118 | 2026-08-15 | Quoted sentence about GPU-resource reliance is in the "Communication Offloading" paragraph, which also says CE offload is "already practical" | Overstated in earlier docs | full-text |
| Resource-aware Computation-Communication Overlap, 2606.09200 | 2026-06-08 | Occupancy shaping, up to 25.5% lower total time, A40, A100, H100, MI250X, no vendor library change | SM budget is the tunable | checked |
| Analytical Resource Management for MoE Overlap, 2609.07536 | 2026-09 | Picks communication CTA count and SM partition analytically, A100 | Same | checked |
| Weave 2609.21483, mKernel 2609.13585, tile-level MoE overlap 2607.19539 | 2026-07 to 09 | Dynamic SM scheduling and fused overlap | Field moved to fusion | search-only |
| ParallelKittens, 2511.13940 | 2025-11 | Analysis of transfer mechanisms and overheads. Search snippet says CE suits coarse contiguous moves | Context. No CE numbers obtained | checked (abstract) |
| FLUX 2406.06858, TokenWeave 2505.11329, Syncopate 2601.20595 | 2024 to 2026 | Overlap and fusion | Crowds I-08 and I-18 | search-only |
| REACT, PIPEMORPH, MSCCL++, AutoCCL, MCCS, CoCoNet, Centauri, NanoFlow | 2026 and earlier | Adaptation and coordination systems. The last six names are confirmed present in the survey text | Context | doc-only |
| Demystifying NCCL, 2507.04786 | 2025-07 | Protocols and algorithms analysis | Context | search-only |

### 7.4 Migration, checkpoint/restore, fault tolerance

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| Llumnix, 2406.03243 (OSDI 2024) | 2024 | Live migration of active requests, pre-copy of append-only KV | Kills I-06 novelty | search-only |
| Astrolabe 2508.03611, OServe 2602.12151 | 2025 to 2026 | Live KV migration for load balancing. OServe reports 1.32x to 1.51x P99 and throughput gain over Llumnix | Successors | search-only |
| Pallas, 2608.16477 | 2026-08 | Proactive KV migration, multi-round | Prior art | search-only |
| ReMP, 2606.18741 | 2026-06 | Runtime TP and PP reconfiguration, 2-D KV migration, most switches in 1 to 7 s | Topology change for transformers | checked |
| WorldMove, 2607.10389 | 2026-07-11 | Bit-identical live relocation of multi-GB GPU cache for world models, 18.8 ms same node, admissibility condition, incast-aware admission, 48 migrations across two clouds | Mutable-state migration exists | checked |
| NVIDIA Dynamo Snapshot blog | 2026 | Single-GPU vLLM and SGLang only. Multi-GPU and NCCL support in progress. gpt-oss-120b restore under 5 s vs about 119 s cold start | Scoop risk for I-35 | checked |
| vLLM RFC #34303 | 2026-02 | CUDA checkpoint/restore in three tiers. NCCL must be re-initialized. Draft with no maintainer comments | Context for I-35 | checked |
| Concordia, 2606.23521 | 2026-06 | GPU-side delta checkpointing, NCCL wrapper, standby pool for live migration | Threat to I-06 and I-35 | search-only |
| Foundry 2604.06664, FlowGPU (Euro-Par 2026), PhoenixOS 2405.12079, CRIUgpu 2502.16631, GPU C/R FAST 2026 | 2024 to 2026 | GPU checkpoint and restore mechanisms | Crowds I-35 | search-only |
| eunomia.dev daily report on GPU checkpoint safety | 2026 | Restorable is not application-consistent. Proposes a "recovery-cut certificate". No author listed | Secondary. Useful framing only | checked |
| FailSafe 2511.14116, LUMEN 2606.17787, AnchorTP 2511.11617, DejaVu 2403.01876, Dynamo recovery 2609.25451 | 2024 to 2026 | Fault-tolerant inference systems | Crowds I-37 | search-only |

### 7.5 RL rollout mismatch

Search-only unless noted: Miles, IsoExec (vLLM blog 2026-08-21), RL-Kernel, QaRL 2604.07853,
AIS 2605.13907, QUADS 2607.15810, FP8-RL, "Reliable RL Scaling Requires Accounting for
Prefill-Decode Kernel Mismatch". vLLM Q3 RL roadmap #48314 (checked): consistency and
alignment section covers logprob and token-ID replay, routing replay, dtype handling.

### 7.6 Elastic training, traces, multi-tenancy (all doc-only)

GFS, DynaTrain, LiveR, Zorse, ElasWave, ByteCheckpoint, Concord, STAO, Syncopate (OSDI 2026),
Alibaba cluster-trace-v2026-spot-gpu.

### 7.7 Frontier signals cited by the 2026-09-22 handoff

| Ref | Title | Check |
|---|---|---|
| R1 | Anthropic, A postmortem of three recent issues (2025) | doc-only |
| R2, R3 | Anthropic AI-resistant technical evaluations, careers page | doc-only |
| R4 | OpenAI, How GPT-5.6 fuses frontier intelligence with frontier efficiency | doc-only |
| R5, R6 | OpenAI careers: Systems Generalist GPT Infrastructure, Software Engineer Inference Performance | doc-only |
| R7 | PROMPTS (Google Research) | doc-only |
| R8 | KernelEvolve (Meta) | doc-only |
| R9, R10 | OpScale, Murakkab (OSDI 2026) | doc-only |
| R11 | NVIDIA blog on FlashInfer | doc-only |
| R12 | TELLER, 2608.01975 | doc-only |
| R13, R14 | KernelSight-LM 2606.28565, Vidur 2405.05465 | doc-only |
| R15 to R17 | KernelBench-X 2605.04956, Atrex-Bench 2607.14541, FastKernels 2605.23215 | doc-only |
| R18 | Correctness Illusion 2606.20128 | doc-only |
| R19 | Contract-Grade Verifier 2608.12700 | checked |
| R20 | Integer Alibi 2608.13756 | checked |
| R21 | DriftBench (MLSys 2026) | search-only |
| R22 | LLM-42 2601.17768 | checked |
| R23, R24 | LAMP 2601.21623, Numerical Fragility in Transformers | doc-only |
| R25 | DA-MoE 2607.23099 | checked |
| R26 | FlashInfer RFC #3605 | checked |
| R27, R28 | vLLM #55238, #55649 | checked |
| R29 | LLM-guided autotuning for Helion kernels (PyTorch blog) | doc-only |

### 7.8 Agentic harnesses (added 2026-09-30)

Full ledger with 55 sources and check levels is Section 9 of
`dossier/LIT_REVIEW_AGENTIC_HARNESSES_2026-09-30.md`. Rows here are the ones most relevant to
this thesis.

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| Autellix, 2502.13965 | 2025-02 | Program-level scheduling of agent LLM calls, 4 to 15x throughput over vLLM (authors' own baseline) | Prior art for I-28-style agent-aware serving | search-only |
| HAL, 2510.11977 (ICLR 2026) | 2025-10-13 | 21,730 rollouts, 9 models, 9 benchmarks, standardized cross-scaffold evaluation | Harness is a controlled variable in agent evaluation | checked |
| Harness or Model?, 2609.11987 | 2026-09-08 | 256 private tasks, no average native-harness advantage, 9.0 and 23.7 point task-level effects, neutral harness 1.2 to 1.6x cost | Moderates "harness beats model" claims | checked |
| Empirical Study of Harness Design for Coding Agents, 2609.20804 | 2026-09-17 | 176 configs, 4 models. Context management matters under tight budgets, planning helps weaker models and mainly cuts cost for stronger ones (corrected 2026-09-30, R-06), bash-only cheaper for strong models | Harness ablation baseline | checked |
| What Does Multi-Harness RL Learn?, 2609.04518 | 2026-09-03 | Eval harness 4.3x effect vs 1.16x for training recipe (Qwen3-8B, 2.14% to 9.27% solve rate) | Eval-harness sensitivity, indirect coupling evidence | checked |
| Towards a Science of Scaling Agent Systems, 2512.08296 | 2025-12 | 260 configs, multi-agent +80.8% to -70.0% by task structure | No universal multi-agent advantage | checked |

### 7.9 GPU efficiency review (added 2026-09-30)

Full source ledger (S1 to S54) with check levels is Section 9 of
`dossier/LIT_REVIEW_GPU_EFFICIENCY_2026-09-30.md`. Rows here are the ones most relevant to live ideas.

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| DeepSeek-V3, 2412.19437 | 2024-12 | FP8 GEMM accumulation on H800 keeps about 14 bits; a preliminary test on random matrices (K = 4096) showed up to about 2% relative error without FP32 promotion every 128 elements; tile and block-wise scaling. Only GEMMs are FP8. The "<0.25% vs BF16" loss gap is from 16B and 230B MoE models, not V3 (corrected by R-07, Section 8.1 row 31) | Direct evidence for I-23's premise that low-precision kernel choice changes numerics | full-text |
| Fishman et al., 2409.12517 (ICLR 2025) | 2024-09 | Standard FP8 diverges after about 200B tokens (SwiGLU outlier amplification); Smooth-SwiGLU trains stably to 2T tokens. The first draft said "appear only after about 2T" (corrected by R-07, row 30) | Validation runs shorter than about 200B tokens can miss precision failures; relevant to I-23 test design | full-text (re-read in R-07) |
| NVIDIA NVFP4 pretraining, 2509.25149 | 2025-09 | 12B model, 10T tokens, NVFP4 matches FP8 loss and accuracy | Single-vendor evidence for the NVFP4 cells in I-23 | checked |
| Thinking Machines, batch-invariant inference blog | 2025-09 | 80 distinct completions from 1,000 temperature-0 runs (Qwen3-235B-A22B-Instruct-2507, vLLM); 1 with batch-invariant kernels. Speed cost is from a separate Qwen-3-8B test: 26 s default, 55 s unoptimized, 42 s improved kernel (corrected by R-07, row 35) | Nearest prior art on output drift vs speed; not peer-reviewed | checked |
| Kokolis et al., 2410.21680 (HPCA 2025) | 2024-10 | 1,024-GPU job MTTF 7.9 h vs 47.7 days for 8 GPUs; ETTR metric | Failure-recovery context for checkpoint/restore background | full-text |
| Lin et al., 2505.05713 (OSDI 2025) | 2025-05 | 42.5% of jobs straggle; 10.4% of GPU-hours wasted; causes mostly imbalance and GC, not hardware | Reliability context | full-text |
| Kurtic et al., 2411.02355 | 2024-11 | 500K+ evals: FP8 W8A8 near-lossless, INT8 1 to 3% loss, INT4 weight-only near 8-bit | Baseline expectation for quantized-kernel accuracy in I-23 | checked |

### 7.10 Context compression and prompt caching for agents (added 2026-10-04, R-08)

Full ledger (S1 to S36) is Section 16 of `dossier/DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md`.
"checked" pages were read through a fetch tool that summarizes with a small model.

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| ACON, 2510.00615 | 2025-10 | LLM compresses observations over 1,024 tokens at ingest and history over 4,096; compressor distilled to Qwen3-8B/14B and Phi-4 at about 95% of teacher accuracy; 26 to 54% lower peak tokens; wall time 73 s to 102 s in one setting | Publishes I-41's mechanism | checked |
| TokenPilot, 2606.17016 (EMNLP 2026 Findings) | 2026-06-15 | Rule-based ingestion-aware compaction with a byte-identical prefix, lifecycle-aware eviction batched every 3 turns, 61% and 87% cost reduction | Publishes I-41's cache-continuity design | checked |
| AgentDiet, 2509.23586 | 2025-09 | Cheaper reflection model removes useless, redundant and expired context, edits only step s-a to keep KV cache; 21 to 36% lower cost | Small-model trajectory reduction with cache awareness | search-only |
| SWE-Pruner, 2601.16746 | 2026-01-23 | 0.6B goal-conditioned line pruner, 23 to 54% fewer tokens with higher success on SWE-Bench Verified | Strongest design for coding agents: verbatim lines, agent states a goal | checked |
| Squeez, 2604.04979 | 2026-04-04 | 2B LoRA pruner of single tool outputs, 92% removed at 0.86 recall | Same | checked |
| TraceLab, 2606.30560 | 2026-06-29 | 4,265 Claude Code and Codex sessions, 357,161 steps, 95.7% prefix hit rate, misses mostly after human idle gaps, compaction in 4.5% of Claude sessions, dataset released | Contradicts I-41's cache premise at scale; weakens I-28's trace-gap angle | checked |
| claude-code issue #94177 | 2026-09-14 | One user, 30 sessions: 43% of cache writes from TTL expiry, 14% from mid-history pruning; proposes cache-aligned compaction | Same finding as E-24 P1; prior proposal of I-42 | checked |
| Don't Break the Cache, 2601.06007 | 2026-01-09 | Prompt caching cuts agent API cost 41 to 80%; system-prompt-only caching most consistent | Caching strategy baseline | checked |
| Keepalive Economics, 2607.19214 | 2026-07-21 | Keepalive pings cut post-pause cost up to 12.5x; about 4 min interval at a 5 min TTL | The direct fix for the cause E-24 found | checked |
| CAPC, 2607.15516 | 2026-07-17 | Two-tier cost model for compression plus caching on Claude Sonnet 4.6 | Cost-model prior art | checked |
| What Does Context Compression Cost an Agent?, 2608.16370 | 2026-08-17 | Compression can raise re-retrieval calls (21.0 to 63.9 for one model) without changing task success | Hidden cost of lossy compression | checked |
| PatchKV, 2609.26219 | 2026-08-18 | Recovers KV after middle-span edits, 2.51 to 3.85x faster resume than full suffix recompute | Infrastructure version (I-44) is being built | checked |
| Anthropic context editing and prompt caching docs | read 2026-10-04 | Clearing tool results invalidates the cached prefix; `clear_at_least`; prices and multipliers (read 0.1x, 0.05x on Opus 5.5; write 1.25x or 2x) | Ground truth for E-24 pricing | checked |

### 7.11 Cross-domain directions (added 2026-10-06, R-09)

Peer-reviewed sources only. The full list of 36 sources and the preprint scoop signals is in
Section 9 of `dossier/NEW_DIRECTIONS_R-09_2026-10-06.md`.

| Work | Venue | What it shows | Effect | Check |
|---|---|---|---|---|
| SMARTS, Wunderlich et al. | ISCA 2003 | Systematic sampling with functional warming, 0.64% average CPI error | Source method for I-46 | search-only |
| Sampling Where It Matters (PKH) | EuroMLSys 2026 | Samples serving configurations, 10% throughput MAPE, 1.6x lower profiling time | Nearest prior art to I-46 | checked |
| Vidur | MLSys 2024 | CPU-run LLM serving simulator with profiled operators | Tool for I-46 and I-47 | search-only |
| Open versus closed, Schroeder et al. | NSDI 2006 | Open and closed workload models give different policy conclusions | Source method for I-47 | search-only |
| Model Equality Testing, Gao et al. | ICLR 2025 | MMD test, 11 of 31 endpoints served a different distribution | Black-box baseline for I-48 | search-only |
| Auditing Test for Behavioral Shift, Richter et al. | ICLR 2025 | Anytime-valid sequential test, hundreds of examples | Sequential baseline for I-48 | search-only |
| TOPLOC, Ong et al. | ICML 2025 | Hidden-state commitments detect model, prompt or precision changes | Verification neighbor of I-48 | search-only |
| Lazy Checkpointing, Tiwari et al. | DSN 2014 | Temporal locality of HPC failures used to tune checkpoints | Closest prior art to I-49 | search-only |
| InferCept, Abhyankar et al. | ICML 2024 | Preserve, swap or discard KV during tool calls | Kills I-50 | search-only |
| KVFlow, Pan et al. | NeurIPS 2025 | Workflow-aware KV eviction and prefetch, up to 2.19x | Kills I-50 | search-only |
| Speculative Actions, Ye et al. | ICLR 2026 | Up to 55% next-action accuracy, up to 20% lower latency | Kills I-51 | search-only |
| Greyhound, Wu et al. | USENIX ATC 2025 | Finds fail-slow GPUs and links at over 99% accuracy | Kills I-54 | search-only |

### 7.12 Mechanism-transfer routes (added 2026-10-07, R-10)

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| DiFR, 2511.20621 | 2025-11-25 | Reference replay with a shared seed detects 4-bit quantization with AUC above 0.999 in 300 tokens; vLLM integration | Kills I-48 | checked |
| Log Probability Tracking of LLM APIs, 2512.03816 | ICLR 2026 | Single-token logprob test detects one fine-tuning step, 1,000x cheaper | Kills I-48 | checked |
| Sparing Strategies to Minimize Reliability Impact On Large Training Jobs | MLSys 2026 | Closed-form spare sizing with a simulation check | Kills I-49 | checked |
| Delta GPU error dataset (Zenodo 15287639) | 2025-08-25 | Public A100 and H100 error events with device IDs, CC BY 4.0 | Corrects R-09 (row 43) | checked |
| llama.cpp PR 21038 | merged 2026-04-01 | Hadamard rotation of Q, K and V before KV quantization | Baseline for I-58 | checked |
| Sigma-delta error correction for tiled attention (TDCommons 10659) | 2026-06-30 | Sigma-delta carry across FP8 attention-score tiles | Closest prior art to I-58 | checked |
| Low-Discrepancy Dither for Quantized Recurrent State Caches, 2609.39185 | 2026-09-30 | Weyl dither for recurrent-state caches | Neighbor of I-58 | checked |
| SparseRL-Sync 2605.07330, PULSE 2602.03839 | 2026 | Sparse lossless BF16 weight sync for RL, about 100x less traffic | Kills I-67 | search-only |
| GEM | FAST 2027 | GPU erasure-coded memory for training recovery | Kills I-68 | search-only |
| TVCache, 2602.10986 | 2026-02 | Stateful tool-value cache for agent post-training | Kills I-62 | search-only |

### 7.13 Route search 3 (added 2026-10-08, R-17)

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| When Fancy Eviction Fails, 2609.28870 | 2026-09 | LRU is robust for prefix caches; block lifetimes are short and session-bound; an SSD tier needs strict admission and is mostly unnecessary | Kills I-82 | full-text |
| Characterizing HBF for LLM Serving, 2609.39131 | 2026-10 | HBF for agentic serving; buffered admission lifts lifetime from 1.21 to 14.82 years | Kills I-82 | checked |
| Write-aware KV policy for HBF, 2609.07175 | 2026-09 | LRU-K admission for generative recommendation KV | Kills I-82 | checked |
| GVR, 2604.22312 | 2026-04 | Bit-exact DSA top-k with a warm start, 1.88x operator, up to 7.52% TPOT at 100K | Baseline for I-83 | checked |
| V-ABFT, 2602.08043 | 2026-02 | Variance-based ABFT thresholds, BF16 and FP16, FP8 remark | Kills I-84 | checked |
| Calibrating the Checksum, 2610.02240 | 2026-09 | One-sided ABFT on BF16 GEMM, false-positive bound | Neighbor of I-84 | checked |
| Stateful Worlds, Stateless Elasticity, 2607.10389 | 2026-07 | Multi-tenant world-model serving; co-batching B = 4 gives 1.16x | Kills I-86 | search-only |
| ThunderAgent, 2602.13692 | 2026-06 (v3) | Program-aware agent serving; memoryless tool-latency assumption | Context for I-87 | checked |
| Ask the Tool, Don't Guess, 2609.18849 | 2026-09 | Tools report progress; beats history-based predictors | Kills I-87 | checked |
| LASER, 2606.31580 | 2026-06 | Load-scaled reasoning budgets with early exit | Kills I-88 | search-only |
| ELDR 2607.00466, Sem-MoE (ICLR 2026) | 2026 | Expert-affinity request routing and batching | Kills I-89 | search-only |
| SpareTrain | ICLR 2026 | DMR for SDC in idle GPU time | Kills I-90 | search-only |
| TraceLab (UW SyFI) | 2026-06 | Public coding-agent traces; 81% of append tokens are repeated prefill | Context for I-91, I-92 | checked |
| Volcano issue #5998, PR #5999 | 2026 | Checkpoint-recency victim ordering | Kills I-93 | checked |

### 7.14 Orphan-finding scan (added 2026-10-08, R-18)

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| llama.cpp issue #22746, PR #26499 | 2026-05 to 2026-09 | Hybrid-model full re-prefill under agent harnesses; 31.8% miss rate, 20.5 s per turn; fix in progress | Kills I-95 (engineering) | checked |
| Megatron-LM issue #7452 | 2026-09-17 | 13 silent numeric bugs in parallelism composition found by `spmd_types` | Context for I-21, I-23 | checked |
| Fast Recovery for LLM Serving in Dynamo, 2609.25451 (NVIDIA) | 2026-09-21 | Replica recovery under 7 s, 13 to 29x faster than warm restart, 4 to 8 GiB per GPU overhead | Kills I-99 | checked |
| From LLM Inference to Agentic Workloads, 2608.15127 | 2026-08 | One 128K co-request raises short-request TPOT by 38% to 80% | Source of I-102 | search-only |
| STAR, 2510.13668; Medha, 2409.17264 | 2025 | Decode rescheduling and long-context decode bounds | Kill I-102 | search-only |
| DSec, 2609.22978 | 2026-09 | Sandbox memory dedup, ballooning, overcommit | Kills I-98 mechanism | search-only (reviewer opened) |

### 7.15 Seed-topic gap search (added 2026-10-16, R-20)

Check levels in this table use the 0.4 meanings with the date 2026-10-16.

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| Cache Me, Catch You (NDSS 2026) | 2026-02 | Six cache attacks in LLM serving (prefix collisions, semantic fuzzy poisoning, block-wise and multimodal collisions); five defenses adopted by vLLM, GPTCache, AIBrix | Topic 2 context, kills I-113 | full-text (partial) |
| HijackKV, 2607.19957 (USENIX Security 2026) | 2026-07 | Hijacked chunk KV in position-independent reuse; recompute defense 1.00x to 3.53x cost; verifier proposed, not evaluated | I-114 | full-text (HTML, partial) |
| Bit-flip KV, 2604.17249 (SECRYPT 2026) | 2026-04 | Shared prefix blocks lack integrity; checksum bounds damage to one batch | Kills I-113 | checked |
| When Latent Agents Lie, 2606.28958 | 2026-06 | KV tampering in multi-agent latent collaboration; HMAC manifest | Kills I-113 | checked |
| To Cache or Not to Cache, 2610.15721 | 2026-10-11 | Safety incompatibility in cross-model KV reuse; VReparo | Topic 2 context | checked |
| Budgeted Cache Repair, 2610.02233 | 2026-09 | Cross-context reuse costs accuracy; selective row recompute | I-114 context | checked |
| PrefixShield, 2608.01657 | 2026-08 | Debt-based admission against cache pollution, in vLLM | Kills I-112 | checked |
| KVGov, 2608.09225 | 2026-08 | Per-principal HMAC salts; defense evaluated in simulation | Kills I-119 | checked |
| 2609.06853 | 2026-09 | Contention lowers KV timing side-channel effect size (d 0.78 to 0.21, GB10) | Kills I-119 | checked |
| Characterization-Guided GPU Fault Resilience in MPS, 2605.26461 | 2026-05 | MPS fault isolation in the open driver; 355 ms recovery | Kills I-110 fault argument | checked |
| dLoRA (OSDI 2024) | 2024 | Merge and unmerge adapters by load | Kills I-110 | search-only |
| Aegaeon (SOSP 2025) | 2025 | Token-level multi-model pooling | Kills I-110 | search-only |
| LithOS, 2504.15465 (SOSP 2025) | 2025 | TPC-level spatial scheduling, transparent | Kills I-110, I-117 | search-only |
| NVIDIA Grace performance tuning guide, power and thermals | current | Module limit minus Grace power goes to the GPU; GB200 GPU target formula | I-118 mechanism | checked |
| Cross-Layer Energy Analysis on GH200, 2605.01938 | 2026-05 | JUPITER GH200 at about 680 W, CPU priority, 100 W default CPU cap | I-118 | full-text (HTML, Section II-A) |
| Patrou et al., 2505.21758 | 2025 | GH200 module capping with steering for an HPC code | Nearest threat to I-118 | search-only |
| TU Dresden, ISPDC 2026 GH200 energy | 2026-07 | Some GH200 budgets not enforced; decode dominates energy | I-118 threat and test risk | search-only |
| 2605.11999 | 2026-05 | Decode does not trigger power caps on H200 | I-118 scale risk | search-only |

### 7.16 Current-frontier systems review (added 2026-10-17, R-21)

Check levels in this table use the 0.4 meanings with the date 2026-10-17.

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| MARS, 2604.26963 | 2026-04 | Joint scheduling of GPU inference, CPU tools, and KV retention | Kills broad CPU-GPU agent scheduling gap | full-text (relevant sections) |
| SAGA, 2605.00528 | 2026-05 | Workflow-atomic scheduling with session affinity and work stealing | Kills workflow-as-unit gap | full-text (relevant sections) |
| SpecBox, 2607.23933 | 2026-07 | Tool-intent prediction and sandbox prewarming during generation | Kills sandbox-prewarming gap | checked |
| Crab, 2604.28138 | 2026-04 | Selective OS-state checkpoints for agents | Kills broad agent checkpoint gap | checked |
| Atomix, 2602.14849 | 2026-02 | Epoch-tagged transactions and progress frontiers for tool effects | Kills broad external-effect gap | checked |
| Resume Means Resume, 2608.03836 | 2026-08 | Workflow persistence semantics and fault conformance | Kills broad resume-contract gap | full-text (relevant sections) |
| Chen and Stratton, Agents of Change | 2026-08 | 300 million work events across 718 firms; output rose while review work increased | Supports verification bottleneck only | full-text |
| SWE-bench correctness study, ICSE 2026 | 2026 | Plausible patches fail full developer tests and differ from oracle behavior | Supports weak-verification premise | full-text |
| TestGRAD, 2610.10242 | 2026-10 | Iterative tests distinguish four agent-generated patches | Kills I-121 mechanism | full-text (relevant sections) |
| Microsoft data-driven test selection | 2020 | Production test selection saves compute while retaining bug detection | Strong I-121 baseline | checked |
| FlashInfer-Bench, 2601.00227 | 2026-01 | Kernel traces, correctness tests, speed tests, and runtime substitution | High threat to I-23 | full-text (relevant sections) |
| FlashInfer v0.7 | 2026-09 | Serving-mode-aware autotuning with persistent configuration results | High threat to I-23 | checked |
| KernelBench-Verified, 2607.16241 | 2026-07 | Stronger tests and realistic baselines reverse a speed claim | Supports I-23 correctness gate | full-text (relevant sections) |
| Integer Alibi, 2608.13756 | 2026-08 | Localizes real INT8 cross-kernel differences | I-23 evidence and threat | checked |
| vLLM training-inference consistency stack | 2026-10 | Fixed kernel choices preserve one reduction order | High threat to I-23 | checked |
| LMCache, 2510.09665 | 2025-10 | Shared KV storage and transfer integrated with vLLM | I-120 context | checked |
| Mooncake ToS paper | 2025 | Metadata recovery and an earlier active-block corruption risk | I-120 evidence | full-text (relevant sections) |
| Mooncake RFC 3692 | 2026-09 | Manifest commit and activation plan; runtime paths remain incomplete | Very high I-120 threat | full-text (issue) |
| Fail-Closed Lowering, 2606.01387 | 2026-05 | Claim identity, materialization, and claim-scoped recovery failure | High I-120 threat | full-text |
| vLLM issue 49250 | 2026-08 | Deterministic wrong output after a rejected shared KV load | I-120 failure evidence | full-text (issue) |
| KVShareArena, 2609.10266 | 2026-09 | Measures reuse across contexts and checkpoints | Very high I-106 threat | checked |

### 7.17 Continued frontier systems review (added 2026-10-17, R-22)

Check levels in this table use the 0.4 meanings with the date 2026-10-17.

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| vLLM streaming requests and Realtime API | 2026-01 | Live requests can receive new input and resume generation | I-122 transition | checked |
| vLLM issue 49449 | 2026 | Session rebuild changes tokens but retains stale block hashes, causing deterministic wrong output | Direct I-122 evidence | full-text (issue) |
| vLLM issue 58776 | 2026 | Preemption exposes a mismatch between valid token history and materialized KV progress | Direct I-122 evidence | full-text (issue) |
| vLLM issue 36755 | 2026 | Async output reads request state after a later schedule step mutates it | Direct I-122 evidence | full-text (issue) |
| vLLM issue 42655 | 2026 | Repeated streaming updates leave stale batch-row identity | Direct I-122 evidence | full-text (issue) |
| vLLM-Omni RFC 5212 | 2026 | Streaming ASR keeps frontend, convolution, acoustic, recurrent, and output state; token replay is insufficient | Direct I-123 evidence | full-text (issue) |
| vLLM-Omni RFC 6227 | 2026 | Real-time world-model migration, recovery, and offload remain later work | I-123 roadmap and scoop threat | full-text (issue) |
| Execution-State Capsules, 2606.20537 | 2026-06 | Captures KV, recurrent, convolution, MTP, and metadata state; evaluates one-device restore | I-123 mechanism and threat | checked |
| DéjàVu, 2403.01876 | 2024 | Adds KV fault tolerance for LLM serving | I-123 baseline | checked |
| Serving Masked Diffusion LLMs, 2608.23807 | 2026-08 | Reports weak completion prediction and proposes slot batching | Kills I-125 | checked |
| vLLM-Omni diffusion continuous batching | current | `StepScheduler` advances requests independently with separate progress and cancellation | Kills I-125 | checked |
| M2K, 2603.24595, SOSP 2026 | 2026-03 | Uses model-aware symbolic execution to find CUDA memory bugs | Kills I-127 | checked |
| Mosaic Streaming elastic determinism | current | Supports deterministic mid-epoch resumption across changed GPU counts | Kills I-128 | checked |
| SuffixReplay, 2609.33477 | 2026-09 | Reuses and reconstructs recurrent suffix state | Kills I-129 route | checked |
| PyTorch compile caching tutorial | current | Exports portable compiler artifacts through Mega-Cache | Kills I-130 route | checked |
| Foundry, 2604.06664 | 2026-04 | Materializes CUDA graph contexts and reduces reported startup time | Kills I-130 route | checked |

### 7.18 Mutable-session source audit and testing pivot (added 2026-10-17, R-24)

Check levels in this table use the 0.4 meanings with the date 2026-10-17.

| Work | Date | What it solves | What it does not solve | Effect | Check |
|---|---|---|---|---|---|
| vLLM PR 59103 | 2026-10-06 | Drops a stale block-hash suffix after streaming-input truncation | Does not define a global session version | Weakens I-122 and supports the I-131 bug corpus | full-text (PR) |
| vLLM PR 58259 | 2026-09-28 | Clears stale async and speculative state at the request handoff | Does not define cross-component commit semantics | Weakens I-122 and supports the I-131 bug corpus | full-text (PR) |
| vLLM PR 58668 | 2026-10-15 | Corrects computed-token accounting after speculative stop trimming | Does not test arbitrary transition sequences | Supports the I-131 bug corpus | full-text (PR) |
| vLLM PR 58778 | 2026-10-17 | Corrects a token boundary and queue ownership during preemption | Does not test other state interactions | Supports the I-131 bug corpus | full-text (PR) |
| vLLM issue 60744 and PR 60764 | 2026-10 | Reproduces a CPU reset fault and proposes parked-session draining | The fix remains open and local | Direct I-131 evidence; weakens the global I-122 mechanism | full-text (issue and PR) |
| A First Look at Bugs in LLM Inference Engines, 2506.09713, TOSEM 2026 | 2026 | Classifies 929 bugs across five engines; more than 35% are non-crash bugs | Does not build a mutable-session transition tester | Supports I-131, but creates a strong novelty threat | full-text |
| Trace-Guided Synthesis of Effectful Test Generators | 2026 | Synthesizes stateful generators from execution traces | Does not target inference-session semantics | Strong generic-generator and combination threat to I-131 | search-only |

### 7.19 I-131 paper audit (added 2026-10-19, R-27)

| Source | What it shows | Effect | Check |
|---|---|---|---|
| vLLM PR 59103 regression test | The trace needs four visible actions and calls a private session-update helper | Circularity threat | full-text (test and patch) |
| vLLM PR 58259 regression test | The trace needs ten visible scheduler and completion actions | Depth-six enumeration does not cover it | full-text (test and patch) |
| vLLM PR 58668 regression test | The trace needs accepted drafts, a stop, and eight visible actions | The E-32 alphabet cannot express it | full-text (test and patch) |
| vLLM PR 58778 regression test | The trace needs preemption, a stop, and eight visible actions | The test calls private preemption | full-text (test and patch) |
| Full-history blame at the four parent commits | No inspected fault line came from an earlier narrow session-state fix | Kills the recurrence premise for this corpus | full-history source check |

### 7.20 Duplex crash-recovery cut (added 2026-10-19, R-28)

Check levels in this table use the 0.4 meanings with the date 2026-10-19.

| Work | Date | What it solves | What it does not solve | Effect | Check |
|---|---|---|---|---|---|
| Execution-State Capsules, 2606.20537 | 2026-06 | Complete heterogeneous model-state snapshots at committed boundaries | No durable external input and output cut | High I-132 threat | full-text |
| WorldMove, 2607.10389 | 2026-07 | Exact live world-model state movement within an interactive block | Requires a live source; no duplex crash recovery | Kills the old I-123 oracle and threatens I-132 | full-text |
| Concordia, 2606.23521 | 2026-06 | Committed GPU delta log and recovery from CPU-visible storage | No client input, output, or playback frontier | Very high I-132 threat | full-text |
| GhostServe, 2605.00831, MLSys 2026 | 2026-03 | Erasure-coded KV checkpointing with under 5% chunk idle time | KV only; no full duplex session state | High checkpoint baseline | full-text |
| vLLM-Omni full-duplex design | current | One engine owner for model state and all session ledgers; reconnect journal | State and journal do not survive engine loss | Direct I-132 system target and threat | full-text (docs) |
| vLLM-Omni RFC 8429 and PR 8485 | 2026-10 | Ordered per-input acknowledgement after caused outputs | No durable engine-crash recovery | Very high I-132 threat | full-text (issue and PR) |
| Apache Flink fault tolerance | current | Replayable input, state checkpoints, and transactional or idempotent sinks | No opaque GPU snapshot under an interactive deadline | Main combination threat | full-text (docs) |
| Resume Means Resume, 2608.03836 | 2026-08 | Machine-checked crash and resume properties for workflow persistence | No model-resident inference state | Adjacent correctness threat | checked |

### 7.21 Scoop check before the Mac test (added 2026-10-19, R-30)

Check levels in this table use the 0.4 meanings with the date 2026-10-19.

| Work | Date | What it solves | What it does not solve | Effect | Check |
|---|---|---|---|---|---|
| CacheReforge, 2609.30884 | 2026-09-25 | Layerwise mixed-version KV, per-layer adapter anchors, drift-based bounded recompute; 92.4% lower KL than stale reuse at 5.44% of layers recomputed | Full-weight updates; only LoRA on Qwen2.5 1.5B and 7B | Kills I-106 | full-text (HTML, partial) |
| KVShareArena v2, 2609.10266 | 2026-09-27 | Cross-checkpoint reuse benchmark with six distant producer pairs | Adjacent RL steps, per-layer boundaries | Context for I-106 | full-text (HTML, partial) |
| CacheRoute, 2608.19677 | 2026-08 | Planned prefix-affinity routing with bounded load skew | Per-request late binding | Kills I-60 | search-only |
| RelServe, 2601.11546 | 2026-01 | Dynamic priority updates by execution progress | Priority inheritance through a shared prefix | Adjacent to I-59 | search-only |
| vLLM issue 59907 and PR 59963 | 2026-10-04 | Priority scheduling faults, including a token-budget inversion, reproduced on the CPU backend | Inversion through a shared-prefix producer | Evidence that I-59 can be tested on the Mac | full-text (issue) |

### 7.22 Fresh frontier pass (added 2026-10-20, R-33)

Check levels in this table use the 0.4 meanings with the date 2026-10-20.

| Work | Date | What it solves | What it does not solve | Effect | Check |
|---|---|---|---|---|---|
| EEP, 2605.10670 | 2026-05-11 | Partial rank failure in wide EP; 95% throughput in 52 s versus 348 s | Slow (not dead) ranks | Kills I-133 | checked (abstract) |
| SGLang Elastic EP (LMSYS blog) | 2026-03-25 | Redundant experts route around failed ranks | | Kills I-133 | search-only |
| vLLM Elastic EP (blog) | 2026-05-14 | Runtime EP resizing | | Kills I-133 | search-only |
| Belayer, 2608.14635 | 2026-07-28 | Rollout and environment failure recovery aligned with context | | Kills I-134 | checked (abstract) |
| RobustRL, 2512.22492 | 2025-12 | Role-based fault isolation in RL post-training | Environment state | Kills I-134 | search-only |
| DSec, 2609.22978 | 2026-09 | Sandbox infrastructure for agentic RL | | Kills I-134, I-140 | search-only |
| NVIDIA nvcf issue 2299, PR 2300 | 2026-10-06 | Multi-GPU C/R with NCCL P2P, IPC, NVLS mappings remapped at same addresses | Merged state, SGLang | Kills I-135 | full-text (issue body) |
| NVIDIA nvcf issue 1183 | 2026-08-25 | Multi-GPU CRIU with transports off, measured on 8x H100 | Fast transports | Kills I-135 | full-text (issue body) |
| Singularity, 2202.07848 | 2022 | Transparent distributed C/R with device proxy and barrier | | Kills I-135 | search-only |
| GCR (FAST 2026) | 2026 | Faster GPU C/R, incremental checkpoints | Multi-GPU NCCL | Context for I-135 | search-only |
| NVIDIA Dynamo Snapshot blog | 2026-05-27 | KV release before capture, GMS weight separation | Multi-GPU (planned) | Kills I-136 | checked |
| Dynamo issue 16357 | 2026-10-13 | Tracks engine gaps in weight and KV skipping | | Kills I-136 | search-only |
| HBM Is Not All You Need, 2606.29986 | 2026-06 | Cross-vendor P/D, KV format and precision conversion | | Kills I-137 | search-only |
| SiliconBench, 2609.19169 | 2026-09 | TB5 RDMA Mac numbers (1.30x to 1.37x on two nodes) | | Context for I-139 | search-only |
| vLLM PR 53298 | open, 2026-10-16 | Fixes issue 49250: shared-block repair order, invalid-block hit block, ownership transfer | Not merged | Kills I-120 | full-text (PR body) |
| vLLM PR 62282 | merged 2026-10-20 | Multi-peer KV write-after-free on the Mooncake path | | Kills I-120 | full-text (PR body) |
| Training-Inference Kernel Contracts, 2606.07581 | 2026-05-26 | Contract vocabulary and routing loop | Evaluation, FP8, NVFP4, MoE | Framing threat to I-23 | checked (abstract) |


### 7.23 GPU-plan freshness check (added 2026-10-20, R-35)

Check levels in this table use the 0.4 meanings with the date 2026-10-20.

| Work | Date | What it shows | Effect | Check |
|---|---|---|---|---|
| vLLM issues 61768, 61905, 61912; PR 61769 | 2026-10-15 | KV-load failure reports orphaned by abort, block reuse or preemption; found by CPU scheduler probes with a mock connector | Supports the I-120 kill | full-text (issues) |
| vLLM RFC 35780 (closed as stale) and PR 42461 (open) | 2026-03 to 2026-07 | Maintainers propose request-level KV-load error handling: reset to 0 and recompute | Simple baseline for I-120 | full-text (RFC and PR body) |
| vLLM issue 49250, PRs 49252 (closed), 53298, 59793, 45146 | 2026-06 to 2026-10 | Open fixes for recompute after a rejected load | I-120 context | checked (GitHub API) |
| LiteTopK, 2607.11976 | 2026-07-13 | Fused exact indexer top-k; indexer scores concentrate in a narrow range | Weakens I-83 | checked (abstract) |
| HPC-Ops Top-K, 2609.08450 | 2026-09-08 | Sample-guided exact top-k for indexer score rows | I-83 context | checked (abstract) |
| vLLM v0.32.0 kernel config | 2026-10-18 | `--linear-backend`, `--moe-backend`, `linear_backend_per_quant`, `--attention-backend`, `sparse_indexer_topk_backend` | Hook for I-23; I-83 context | full-text (source) |
| Lambda MIG guide; GPU price lists | current | `sudo nvidia-smi` on a 1 x GH200; H100 and B200 hourly prices | Session planning | search-only |

### 7.24 Limitations mining (added 2026-10-20, R-36)

Check levels in this table use the 0.4 meanings with the date 2026-10-20. "partial" means the
limitation sections and targeted passages of the full HTML were read.

| Work | Date | Gap it leaves | What closes it | Effect | Check |
|---|---|---|---|---|---|
| Dynamo fast recovery, 2609.25451 | 2026-09 | Request-state preservation | Committed-length journal; its own capacity argument | Source of I-141 | partial |
| LUMEN, 2606.17787 | 2026-06 | Multi-checkpoint KV replication | | Source of I-141 | partial |
| AnchorTP, 2511.11617 | 2025-11 | Multi-node; predictive migration | | Keeps KV across crash; weakens I-141 | partial |
| DLPM (Locality-aware Fair Scheduling), 2501.14312 | 2025-01 | Who pays for a shared prefix | srujan20/prefix-cost-attribution | Source of I-142 | partial |
| srujan20/prefix-cost-attribution (GitHub) | 2026-08-25 | | Exact Shapley over the prefix trie | Kills I-142 | full-text (README) |
| Who Pays for the KV Cache?, 2609.24991 | 2026-09-21 | Cross-group prefix split | | Context for I-142 | checked (abstract) |
| TBIK, 2511.17826 | 2025-11 (ICML 2026) | Quantized dtypes | | Source of I-143 | partial |
| Software Aging in GPU LLM Serving, 2606.11916 | 2026-06 | Growth triggers, rejuvenation | Fast shadow restarts | Source of I-144 | partial |
| NVIDIA shadow engine recovery blog | 2026 | | 7.3 s versus 283 s cold restart | Kills I-144 | search-only |
| Aborted but Not Forgotten, 2608.15939 | 2026-08 | Structured defenses | Branch salting, CoW caches | Source of I-145 | partial |
| Bit-Flip Vulnerability of Shared KV Blocks, 2604.17249 | 2026-04 | Reachability, defenses | HBM ECC, checksums | Source of I-146 | partial |
| GhostServe, 2605.00831 | 2026-05 | P/D and decode fault tolerance | | Source of I-147 | partial |
| HiSparse, 2608.07009 | 2026-08 | Host tier smaller than HBM | NVIDIA ICMS | Source of I-148 | partial |

### 7.25 Algorithmic-optimization sweep (added 2026-10-21, R-37)

Check levels use the 0.4 meanings with the date 2026-10-21.

| Work | Date | What it proves | Effect | Check |
|---|---|---|---|---|
| SGLang source, commit a3d00eff | 2026-10-21 | LPM and HRRN fall back to FCFS above 128 waiting requests; `evict()` rebuilds a heap of all evictable leaves; `ep_dispatch_algorithm` has an `lp` option | Source of I-149 and I-161; kills I-153 | full-text |
| vLLM source, commit 90c4a1a | 2026-10-21 | No cross-turn tokenization cache; full-prompt block hashing in the engine input thread; capture sizes step 8 then 16; persistent penalty counts in the GPU runner; MLA source ranks spread | Kills I-151, I-157, I-159; source of I-150 | full-text |
| PyTorch source, commit 6dbae335 | 2026-10-21 | DCP read planning uses a sweep line; dedup is list scheduling without LPT | Kills I-160 | full-text |
| PEEK, 2607.02525 | 2026-05-10 (v2 2026-09-12) | Incremental radix tree over the pending queue, dual walk against the prefix cache | Kills I-149 | checked (abstract) |
| Requests of a Feather Must Flock Together, 2605.06046 | 2026-05 | Tree work is 50 to 90% of prefix-aware scheduler latency; cumulative prefix hashes and a lazy min-heap | Kills I-149; context for I-155 | search-only |
| DLPM, 2501.14312 | 2025-01 | Repeated prefix matching overhead in global scheduling | Context for I-149 | search-only |
| vLLM RFC 47582 | 2026-07-03, open | Incremental prompt encoding; 782.94 ms full against 10.95 ms incremental at 512K tokens; TTFT 0.92 s to 0.47 s; cites SGLang SMG L1 prefix tokenization cache | Kills I-150 | checked |
| TokTier, 2607.29678 | 2026-07 | Exact stateful CPU plus GPU tokenization for agent serving | Kills I-150 | search-only |
| Incremental BPE Tokenization, 2605.30813 | 2026-05 | Incremental BPE | Kills I-150 | search-only |
| TensorRT-LLM ADP Balance Strategy (tech blog 10) | 2025 | Per-iteration DP-attention balance | Kills I-152 | search-only |
| NanoCP, 2605.21100 | 2026-05-20 | Dynamic context parallelism for each request in DP plus EP decode against KV imbalance | Kills I-152 | checked (abstract) |
| LLEP, 2601.17111 | 2026-01 (ICML 2026) | Spills skewed expert load to less-used devices | Context for I-153 | search-only |
| EasyBalance, 2608.07964 | 2026-08 | Cross-layer MoE load balancing | Context for I-153 | search-only |
| Johnson's rule note, 2512.06119 | 2025-12 | Johnson's rule is O(n log n) for two-machine flow shop makespan | Context for I-154 | search-only |
| EPD Disaggregation, 2501.05460 | 2025 (ICML 2025) | Separate encode, prefill and decode | Context for I-154 | search-only |
| SGLang, 2312.07104 | 2023-12 | DFS order is optimal for hit rate when the cache holds the longest request | Kills I-155 | doc-only (via search summary) |
| BatchLLM, 2412.03594; BlendServe, 2411.16102 | 2024-12; 2024-11 | Offline prefix-aware batching and ordering | Kill I-155 | search-only |
| Tenplex (SOSP 2024) | 2024 | Minimal reconfiguration plans for dynamic parallelism | Kills I-156 | doc-only |
| NorMuon, 2510.05491; Motif 2, 2511.07464; INTELLECT-3, 2512.16144; MatrixFSDP, 2607.05895 | 2025-10 to 2026-07 | Owner-rank and sorted round-robin placement for Newton-Schulz work | Kill I-162 | search-only |
| Mirage MPK post (Zhihao Jia, Medium) | 2025 | MPK uses round-robin task placement; priority-aware policies are future work | Context for I-163 | search-only |
| Cohere megakernel blog | 2026 | Waves dealt round-robin to SMs, work stealing; wave order changes performance | Threat to I-163 | search-only |
| ForgeMegakernel, 2609.12379 | 2026-09-11 | Per-SM instruction streams; 50.5 to 85.9% bandwidth utilization | Threat to I-163 | checked (abstract) |
| Event Tensor, 2604.13327 | 2026-04-14 (MLSys 2026) | Static and dynamic scheduling transformations for megakernels | Threat to I-163 | checked (abstract) |
| Weave, 2609.21483; Fleet, 2604.15379 | 2026-09; 2026-04 | Dynamic SM scheduling in an MoE megakernel; scheduler workgroups on multi-die GPUs | Threats to I-163 | search-only |
| Recency/Frequency Adaptive KV Caching, 2606.21238 | 2026-06 | ARC-style KV cache policy | Kills I-164 | search-only |
| SGLang PRs 38371 and 38454 | 2026-09-07; 2026-09-08 (open) | Persistent lazy eviction heaps for the unified radix cache; 4.9 ms at 35,000 leaves before, 30 to 33 microseconds after (38371) | Kills I-161 (R-38) | checked (titles, dates, state via GitHub API; numbers from R-38) |
| SGLang PR 41690 | 2026-09-29 (open) | Amortized L1 eviction under sustained pressure | Context for I-161 | checked (title via GitHub API) |
| SGLang PR 39627 | merged | Rust TreeCore becomes the default | Scope limit on E-37 | checked (title via GitHub API) |
| HazyResearch/Megakernels `megakernels/scheduler.py` | 2025 | List scheduler (ready heap, SM free-time heap) and a critical-path priority function | Kills I-163 (R-38) | full-text (grep of the file) |
| Cohere megakernel blog | 2026-09-08 | Cost-based greedy and random-schedule search give about 10% over round-robin on a dense model; dropped for MoE | Kills I-163 (R-38); upgrades the row above | checked |
| MPK, 2512.22219 | 2025-12-22 (v2 2026-06-10) | Mega-kernel compiler and runtime with decentralized scheduling across SMs | Context for I-163 | checked (abstract) |
| DySHARP (Zhang et al., "Accelerating MoE with Dynamic In-Switch Computing on Multi-GPUs") | 2026-05-07 | Dynamic multimem addressing for each token's target set | Kills I-158 (R-38) | search-only (secondary digest via R-38; no arXiv ID) |
| Ada-MK, 2605.11581; AutoMegaKernel, 2606.09682 | 2026-05; 2026-06 | Offline schedule search for megakernels | Context for I-163 | search-only (via R-38) |
| Continuum, 2511.02230 (ICLR 2026) | 2025-11 | KV TTL across tool calls from reload cost and tool-latency distribution | Closes the R-38 near miss | search-only |

---

## 8. Errors, retractions, and do-not-cite list

### 8.1 Errors and retractions (append only)

| # | Date | What | Status | Rule going forward |
|---|---|---|---|---|
| 1 | 2026-09-16 | V1 synthetic GPU-efficiency suite over-interpreted | RETRACTED | Synthetic results are screening only |
| 2 | 2026-09-16 | V2 network test structurally favored bulk sync | CORRECTED in V2.2 | Baselines must not differ only by message count |
| 3 | 2026-09-16 | V2 heterogeneous break-even: CPU plus MPS exceeded isolated sum by about 71% | Absolute times untrusted | Unified-memory contention is not H100 behavior |
| 4 | 2026-09-16 | Reconstructability POC read as physical recovery | CORRECTED: planning abstraction only | Say what layer a POC exercises |
| 5 | 2026-09-16 | Planner percentages read as hardware evidence | CORRECTED: normalized sensitivity study | Label normalized costs |
| 6 | 2026-09-16 | First cached-replay timing read as evidence for selective recovery | CORRECTED: known caching principle | Check that a speedup is not just "less work" |
| 7 | 2026-09-16 | v1.1 frozen-KV timings | INVALID: mask geometry broke semantics | Never report a number when `semantic_recovery_correct` is false |
| 8 | 2026-09-16 | 9.5% activation oracle generalized | Not decisive: one case, two reps | Sweep before interpreting |
| 9 | 2026-09-16 | Upstream hybrid pytest cited as evidence | INVALID: missing `tblib`, return code 4 | Check return codes before citing a test |
| 10 | 2026-09-16 | Fixed method order in v1.1 and v1.2 inflated every speedup | CORRECTED in v1.3 | Randomize method order within each rep |
| 11 | 2026-09-16 | Recurrent snapshot assumed fp16 (9.63 MiB) | CORRECTED: fp32, 18.84 MiB measured | Measure byte constants from the live cache |
| 12 | 2026-09-17 | Nemotron-H snapshot 252 MiB, then 192 MiB | CORRECTED: 96 MiB with three-class layer typing | Read all layer types from config |
| 13 | 2026-09-17 | First checkpoint-placement DP returned a constant | CORRECTED, validated against the closed form | Validate a DP against a closed form |
| 14 | 2026-09-17 | Evenly spaced Pareto (E-15) presented as the result | Weak baseline, superseded by E-19 | Use the strongest simple baseline |
| 15 | 2026-09-23 | Docs call the five vLLM bugs one class the checker addresses | OVERSTATED: only #53041 matches, and it is unimplemented | Read each cited issue |
| 16 | 2026-09-23 | Docs call #53041 an active design discussion | OVERSTATED: zero comments, one author | Report comment counts |
| 17 | 2026-09-23 | Docs claim no public work applies differential testing to serving state reuse | OVERSTATED: Ekka, GRIEF exist | Search the tool class, not only the exact application |
| 18 | 2026-09-23 | Docs cite the collective survey as naming CE-vs-SM arbitration open | OVERSTATED: the survey says CE offload is practical and generic co-design is open | Quote the surrounding paragraph |
| 19 | 2026-09-23 | I-18's GO test claimed to test a live signal | FLAWED: a per-call static table implements the oracle | Predeclared tests must isolate the claimed input |
| 20 | 2026-09-23 | I-18 hypothesis that bandwidth-bound compute prefers the SM path | Unsupported: no mechanism, same HBM bytes on both paths (INTERPRETATION) | State the mechanism before the hypothesis |
| 21 | 2026-09-23 | 2026-09-22 handoff threat list called current to 22 Sep | INCOMPLETE: missed about a dozen papers and PRs | Re-scan the last 6 to 8 weeks before every claim lock |
| 22 | 2026-09-23 | I-06 novelty scan listed as the next gate and never run | UNRESOLVED until R-05 partial scan | Record gate completion in Section 4 |
| 23 | 2026-09-23 | `dossier` says all `.venv` symlinks point to one env | CORRECTED: `rq1_soundness_gate/.venv` points to `~/Downloads/vllm_hybrid_state_gap_harness/.venv` | Verify paths before documenting them |
| 24 | 2026-09-23 | RQ3 result file says `grid_points: 48` but has 216 rows over 36 cells | UNRESOLVED, low impact | Reconcile result metadata |
| 25 | 2026-09-30 | Harness review C1: Terminal-Bench 70.1 vs 63.8 given as version 2.0 and as "strongest support" | CORRECTED (R-06): scores are 2.1, order reverses on 2.0 | Record the benchmark version with every leaderboard number |
| 26 | 2026-09-30 | Harness review C4: "no compute-matched frontier comparison exists", Anthropic 80% read as evidence the 90.2% gain is compute | CORRECTED (R-06): Kim et al. matched budgets, the two Anthropic figures are from different evals | Read the method section before saying a control is missing |
| 27 | 2026-09-30 | Harness review C8: "different groups" for ABC and HAL, C9 and C10 labelled ESTABLISHED | CORRECTED (R-06): authors overlap. C9 and C10 downgraded, C11 relabelled OPEN | Check author lists against the independence rule before labelling |
| 28 | 2026-09-30 | Harness review: Vats and Golev CI statement inverted, S11 tagged "checked" from an aggregator | CORRECTED (R-06): abstract says CIs include zero except for the largest gap. Tag now split | Only tag "checked" for primary pages |
| 29 | 2026-09-30 | Harness review: Fan et al. planning, Arjmandi n, SWE-bench Pro range, METR stars, Section 7 I-23 pointer, Meta-Harness wording | CORRECTED (R-06) | Quote abstracts instead of paraphrasing them |
| 30 | 2026-09-30 | GPU review, this journal's Section 4 and 7.9: Fishman et al. FP8 instabilities "appear only after about 2T tokens" | CORRECTED (R-07): standard FP8 diverges after about 200B tokens, 2T is the stable run with Smooth-SwiGLU | Read which run a headline number belongs to before using it as a threshold |
| 31 | 2026-09-30 | GPU review C10: DeepSeek-V3 "<0.25% loss gap" and "end to end" FP8, "about 2% error" as a training measurement | CORRECTED (R-07): gap is from 16B and 230B models, only GEMMs are FP8, the 2% is from random matrices | Check the scale of the model behind each headline result |
| 32 | 2026-09-30 | GPU review C15: Samsi et al. 3 to 4 J per token labelled A100 | CORRECTED (R-07): V100 for the 8, 16, 32 shard runs | Read the hardware table, not the abstract |
| 33 | 2026-09-30 | GPU review Section 7, journal Section 1.6: "MegaScale's 1.34x comes largely from overlap" | CORRECTED (R-07): overlap is 6.2 of 17.6 ablation points, algorithmic changes 8.6 | Read the ablation before attributing a gain |
| 34 | 2026-09-30 | GPU review C4, journal Section 4: MFU "about 40 to 55%, ceiling has not risen since 2022" | CORRECTED (R-07): MegaScale 59.1% at 3,072 GPUs, Korthikanti 56.3% without DP. Now 38 to 59% with caveats | Quote the best and worst reported value, not only the largest-scale one |
| 35 | 2026-09-30 | GPU review debate 8: Thinking Machines "42 s vs 26 s" attached to Qwen3-235B | CORRECTED (R-07): timing is Qwen-3-8B, one GPU, 1,000 sequences of 90 to 110 tokens | Keep each number with its own experiment |
| 36 | 2026-09-30 | GPU review C4: MT-NLG 29.7% mixed with with-attention MFU rows | CORRECTED (R-07): 30.2% with attention. GPT-3's 21.3% is non-attention | State the FLOP convention per row |
| 37 | 2026-09-30 | GPU review method: "16 papers" read in full text, all sources "opened 2026-09-30" | CORRECTED (R-07): 15 full-text, S2 prior-knowledge and not opened | Count check levels from the source table |
| 38 | 2026-09-30 | GPU review C11 and C12: DistServe 12.6x given without baseline, AWQ ">3x" read as an error result | CORRECTED (R-07): 12.6x is vs vLLM (MII gives 1.4x to 2.6x SLO), 3x is TinyChat speed | Attach a baseline to every multiplier |
| 39 | 2026-09-30 | GPU review Section 3 and C8: speculative decoding "shrank on independent re-measurement" | CORRECTED (R-07): different models and methods. Supportable case is EAGLE, reported 2.9 to 3.1x, measured 2.08x | Compare the same method and model before saying a result shrank |
| 40 | 2026-09-30 | GPU review: C3 labelled CONSENSUS, Ho et al. as "LM capability", Yang et al. "across 70+ GPUs", "BF16 in every large run", C12 "8-bit near-lossless", OPT vs Meta "rose" | CORRECTED (R-07): C3 now WORKING HYPOTHESIS, scopes narrowed | Check a label against the rubric in Section 0 of the report |
| 41 | 2026-10-04 | Harness review (2026-09-30) Section 7: "the gap that looks real is the lack of representative, public traces of harness-generated inference traffic" | QUALIFIED (R-08): TraceLab (2606.30560, released 2026-06-29, 4,265 Claude Code and Codex sessions) and the Exgentic OpenTelemetry dataset (1,781 traces) predate the review. The review file is unchanged | Search for released datasets, not only papers, before calling a data gap |
| 42 | 2026-10-07 | E-23 (2026-09-24) called the "flip invariant" (every flip has margin at most 2 times the maximum logit difference) "the strongest support for the idea that risk concentrates at small margins". R-09 used it as a fact for I-48 | CORRECTED (R-10): the invariant is the triangle inequality and holds for any flip by construction. It is a valid harness bug check, not evidence. The empirical part is that the maximum logit difference was small (largest flip margin 0.178 vs a median margin of about 2) | Before citing a pattern as evidence, check whether it follows from the definitions |
| 43 | 2026-10-07 | R-09 (I-49): "I could not confirm public node-level GPU failure logs from Kokolis et al. or Cui et al." | CORRECTED (R-10): the Delta GPU error data of Cui et al. (SC 2025) is public on Zenodo under CC BY 4.0 (2025-08-25) | Search dataset hosts (Zenodo, Hugging Face, GitHub releases), not only papers |
| 44 | 2026-10-07 | E-27 rubber duck after the run, points 3 and 5: "the smooth part sits in the sinks and the recent window", and shaping is worse in the all-quantized variant "because sinks are the strongest spikes" | CORRECTED (R-12): C3z reused gates calibrated for another region. With recalibrated gates the change is neutral (+0.004 nats). Quantizing the sinks lowers the shaped/plain ratio; the recent tokens are the spikiest | Before explaining a variant result, check that it used settings calibrated for that variant |
| 45 | 2026-10-07 | E-28 rubber duck before the run (README and chat): "a cache miss on a 25k-token prompt costs about 10 times the GPU time of a hit" | CORRECTED (R-15 m6): with the code constants a hit turn costs about 0.3 GPU-s at the median output and 0.5 to 0.6 at the mean; a miss adds about 1.1 s. The ratio is about 3 to 4 times | Compute a cost ratio from the model constants, including decode, before stating it |
| 46 | 2026-10-08 | E-28 rubber duck after the run (README point 6): "the bistable band that R-14 saw has no measurable width here", and "above the knee, the uncapped system degrades by itself" read as plain overload | CORRECTED (R-16): the band exists. At 1.0 x knee a spike of 2 x N* or more traps the system for at least 9,000 s, and controls above the knee leave the good state by noise while a cap still carries the load. This is a noise-triggered metastable transition. The frozen K1 KILL is still correct for the frozen grid | When a frozen rule says "no effect", check the edges of the grid (trigger size, horizon, metric power) before explaining why the effect is absent |
| 47 | 2026-10-08 | `dossier/ORPHANS_R-18_2026-10-08.md` lists vLLM #37121 as a "hybrid-model full re-prefill" report | CORRECTED (source check, 2026-10-16): #37121 is "KV cache ~7x memory overestimation for hybrid Mamba/attention models (Qwen3.5)". It reports wrong memory sizing, not re-prefill. The other cited reports (llama.cpp #22746, #21831, vLLM #38988) do match | Open each issue before citing it as evidence for a claim |
| 48 | 2026-09-22 | Section 7.7 (from the 2026-09-22 handoff) lists "OpScale, Murakkab (OSDI 2026)" | CORRECTED (source check, 2026-10-16): Murakkab is in the OSDI 2026 program. OpScale is not in the program; it is the arXiv preprint 2608.13499 (2026-08) | Confirm a venue in the official program before citing it |
| 49 | 2026-09-16 | I-08 (Section 5.3) and Section 7.3 count REACT as crowding the overlap crossover | CORRECTED (source check, 2026-10-16): REACT is arXiv 2609.04417, "Tuning Collective Patterns to Alleviate Congestion in Shared AI Clusters". It adapts collective patterns to network congestion and does not study compute-communication overlap. The I-08 verdict does not change: Syncopate, PipeMorph and NCCL controls still crowd it | Read the abstract before using a name as crowding evidence |
| 50 | 2026-10-17 | Section 7.16 (R-21) names the Chen and Stratton paper "Agents of Change" and says "output rose while review work increased" | CORRECTED (source check, 2026-10-17): the PDF title is "Artificial Intelligence in the Firm: Bottlenecks in Software Production" (Harvard job market paper, version of 2026-08-04). The phrase "Agents of Change" is not in the PDF. The 718-firm and 300-million-event figures match. "Output" here means code volume; finished tasks did not rise by a significant amount (secondary summaries, not checked in the PDF) | Copy a title from the primary document, not from a search result |
| 51 | 2026-10-17 | Section 7.16 (R-21) dates Microsoft "Data-driven test selection at scale" to 2020 | CORRECTED (source check, 2026-10-17): the paper is ESEC/FSE 2021 (Industry track), Mehta et al. | Take the year from the venue listing |
| 52 | 2026-10-17 | R-21 and R-22 issue dates: vLLM issue 49250 as 2026-08 (Section 7.16), vLLM issue 58776 and PR 58778 as 2026-10 (R-22 dossier), Mooncake RFC 3692 as 2026-09 | CORRECTED (source check, 2026-10-17, GitHub API): 49250 opened 2026-07-20; 58776 and PR 58778 opened 2026-09-25; Mooncake 3692 opened 2026-08-26. No claim or verdict depends on these dates | Use the issue's opened date from the API |
| 53 | 2026-10-17 | R-22 treated four mutable-session failures as one direct class and inferred that I-122 needs one global version boundary | RETRACTED by R-23 and R-24: E-31 misrepresented the source failures. Current fixes are narrow and do not show that a global version is necessary | Do not infer a required mechanism from repeated bugs or from a missing field |
| 54 | 2026-10-19 | The I-131 dossier said issue 58776 reproduced after PR 58259 | RETRACTED by R-25: issue 58776 opened three days before PR 58259 merged. The available record does not show a later reproduction | Compare issue creation and fix merge times before claiming recurrence |
| 55 | 2026-10-19 | R-24 linked issue 36755 to PR 37160 | CORRECTED by the GitHub timeline: PR 37160 is CPU KV offloading. Issue 36755 links PRs 36812 and 36859 | Use the issue timeline to verify a fix relationship |
| 56 | 2026-10-19 | R-24 paired issue 42655 with merged PR 58259 | CORRECTED by R-26: issue 42655 describes GPU row mapping. Its proposed PR 43683 did not merge. PR 58259 fixes a different async handoff race | Require an explicit issue link or matching regression test before pairing a fix |
| 57 | 2026-10-19 | The first full-history audit used a shallow clone and assigned old lines to boundary commit `d3e2888c` | INVALID and rerun: full history gives commits 91601ff4, d4d30940, and a8ff2cca | Treat a caret-prefixed blame SHA as a shallow-history boundary |
| 58 | 2026-10-19 | The R-25 review-registry heading used 2026-10-17 | CORRECTED to 2026-10-19. The historical filename still contains 2026-10-17 | Use the review completion date, not the design date |
| 59 | 2026-10-19 | E-35 design: condition C4 (L promoted to H's priority from step 0) was called an upper bound on what inheritance can give | CORRECTED (R-31 and the main-session smoke test): a promoted L runs before M and fills the cache before H arrives, so C4 measures prefetching, not inheritance. The grid was withdrawn (amendment A1) | An oracle for a dynamic mechanism must act only when the mechanism would act |
| 60 | 2026-10-19 | E-35 amendment A1: "L's prefill ends after 33 steps"; E-35 rubber duck: H "computes the rest of P itself" | CORRECTED (run 1 and R-32): the lone-request exemption lets L prefill 256 tokens per step until H arrives, so it ends at step 8. T1 does not test that H computes the rest itself; that claim rests on code reading | Check engine-specific exemptions before fixing scenario timing; do not state as tested what only code reading shows |
| 61 | 2026-10-20 | E-30 KILL 2 objective frozen in R-35 (equal weight of normalized GPU throughput and CPU work) | CORRECTED before any run (R-34): the gain comes mostly from the CPU term in decode, so the objective decides KILL 2 by construction. A1 decides on a constrained form | Test each frozen objective on a synthetic case where the mechanism clearly helps |
| 62 | 2026-10-20 | E-36 as predeclared in R-35: candidates from an inventory of backends that start; STOP-FLAT at a Wilson upper bound of 1.5% with 256 prompts | CORRECTED before any run (R-34): a backend that starts can be an alias of the default kernel, and 0 flips in 256 prompts minus up to 12 floor exclusions cannot reach 1.5%. A1 uses a fixed list, log-based alias detection, and 2% at 189 or more prompts | Check that each pass condition can be reached with the planned sample size |
| 63 | 2026-10-20 | R-34 reviewer: a Wilson upper bound of at most 2% for 0 flips needs n >= 188 | CORRECTED by the main session: n >= 189 (3.84 / (n + 3.84)); a unit test checks it | Recompute reviewer numbers before they enter a rule |
| 64 | 2026-10-21 | R-37 and E-37: I-161 "prior work: none found", "engineering PR candidate"; E-37 S1 presented as the SGLang eviction cost | CORRECTED (R-38, verified): SGLang PRs 38371 and 38454 already add lazy heaps. S1 models the legacy Python class; the default is the Rust core | Search the target repository's open PRs, not only papers, before calling an engineering fix open. Confirm which code path is the default |
| 65 | 2026-10-21 | R-37: I-163 kept as WEAK while citing the Cohere megakernel blog | CORRECTED (R-38, verified): the same blog reports about 10% from cost-based scheduling on a dense model and why it was dropped; Hazy's public scheduler already has a critical-path priority | Read every section of a cited source that bears on the claim, not only the passage that supports it |
| 66 | 2026-10-21 | R-37: I-154 "any order without idle time is within O(sqrt n)"; I-158 "closes the whole family" | CORRECTED (R-38): the bound holds only for random order, and the gain is 3 to 14% at n of 8 to 128; DySHARP covers dynamic multicast | State the assumptions of an asymptotic argument and check small n. An arithmetic kill covers only the variant it computes |
| 67 | 2026-10-26 | E-36 Session 1 first draft: "no dispatch headroom ... close to the Test A KILL condition"; "86% to 95%" flipped; kernel check "different scaling or layout code" | CORRECTED (R-42): under exact match the KILL condition is trivially true for a safe set of {R}, so headroom is untested, not absent; the flip range is 87.5% to 95.3%; dense deep_gemm differs in fused activation-quantization kernels | An exact-match contract that flags every numerical change cannot measure headroom. Check what the rule can show before reading its result as evidence |

### 8.2 Do not cite

- V1 synthetic results (E-04) as evidence of anything.
- v1.1 frozen-KV timings (E-11), and any fixed-method-order timing (v1.1, v1.2, dossier
  section 1.2 left column).
- The normalized planner percentages (E-09) as hardware results.
- The upstream hybrid pytest step in E-07.
- The cross-model table (E-16) as measured serving behavior. It is computed from configs.
- The evenly spaced Pareto (E-15) as the result.
- Measured Qwen savings as a ceiling. Use the analytical ceiling and label it.
- Absolute elastic break-even times from E-05.
- The 9.5% oracle as a general figure.
- "Magnitude" claims for I-18. Only the structural gap is established.
- The earlier docs' claim that the collective survey names I-18 as an open problem.

---

## 9. Artifact map and reproduction

### 9.1 Repository layout

```
masters-thesis/
  RESEARCH_JOURNAL.md                     this file, source of truth
  agents.md                               context file for AI agents (style, background,
                                          pointer to this journal since 2026-09-23)
  LLM_Inference_Thesis_Research_Handoff_2026-09-22.docx   handoff introducing I-23
  stateful_elastic_inference_thesis_dossier.docx          I-06 dossier (Phase 1)
  AI_Infra_Thesis_Handoff_Package/        2026-09-16 handoff. Checksums verified 2026-09-23
  dossier/
    RESEARCH_GAP_DOSSIER_2026-09-16.md    iteration 4 (I-16 to I-20)
    ADVERSARIAL_REVIEW_2026-09-23.md      R-05 full write-up
    DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md   R-08 dossier for I-41 to I-45
    FRONTIER_R-21_2026-10-17.md           current-frontier review and first experiment
    REVIEW_E-31_PRERUN_2026-10-17.md      R-23 invalid-design finding for E-31
    SOURCE_AUDIT_I-122_2026-10-17.md      R-24 source audit and I-131 pivot
    REVIEW_E-32_AMENDMENT_2026-10-19.md   R-26 amendment review
    PAPER_AUDIT_I-131_2026-10-19.md       R-27 paper audit and thesis kill
    bundles/                              result zips for E-17 to E-21
  archive/                                superseded files, see archive/README.md (ideas/ briefs, RESEARCH_STATUS.md)
  experiments/
    hybrid_state_recovery_v1_3/           E-12 to E-17
    rq1_soundness_gate/                   E-18
    rq2_nccl_source_audit/                E-20
    rq3_state_budget/                     E-19
    rq3b_layout_survey/                   E-21
    safedispatch_oracle/                  E-23 Mac pilot harness for I-23 (Test A)
    structurer_ingest_screen/             E-24 trace study and E-25 small-model screen for I-41 (aggregates only)
    versioned_session_commit/             invalid E-31 design and reproducible R-24 source audit
    session_state_metamorphic/            killed E-32 design; never run
    session_state_corpus_pilot/           cancelled E-33 design; never run
  grants/                                 compute application drafts (NAIRR, ACCESS, advisor letter and email, NVIDIA outline)
  .claude/skills/research-gap-dossier/    the review workflow (SKILL.md and references)
  .gitignore                              ignores .DS_Store, __pycache__, *.pyc, .venv
```

The README.md that started the repo (Phase 0 log) is deleted in the working tree but intact
in git: `git show HEAD:README.md`. Its content is summarized in Section 5.1.

### 9.2 Handoff package contents (all checksums OK on 2026-09-23)

| File | Purpose or warning |
|---|---|
| `AI_Infra_Thesis_Master_Handoff_Dossier.docx` | Canonical context as of 2026-09-16 |
| `AI_Infra_Thesis_Experiment_Evidence_Ledger.docx` | Test-by-test record and validity rules |
| `AI_Infra_Thesis_Handoff_README.md` | Plain-text summary and hard warnings |
| `AI_Infra_Thesis_Artifact_Index.csv` | Artifact purpose and warnings |
| `gpu_efficiency_v2_results_bundle.zip` | E-05. Use cautiously |
| `gpu_efficiency_v2_2_results_bundle.zip` | E-06 |
| `vllm_hybrid_state_gap_results_bundle.zip` | E-07 |
| `vllm_reconstructability_poc_results_bundle.zip` | E-08 |
| `consistent_state_planner_poc_results_bundle.zip` | E-09 |
| `mac_hybrid_selective_recovery_results_bundle.zip` | E-10 |
| `mac_state_selective_recovery_results_bundle.zip` | E-11. Frozen-KV semantics INVALID |
| `mac_state_selective_recovery_fix_v1_2.zip` | E-12 code. It was unrun when packaged, since tested |

### 9.3 Reproduction

```
# hybrid-state measurements and analysis (E-12 to E-17)
cd experiments/hybrid_state_recovery_v1_3
.venv/bin/python run_mac_selective_v13.py --mode sweep
.venv/bin/python flop_model.py
.venv/bin/python pareto_budget.py
.venv/bin/python crossmodel_state_ratio.py

# RQ1 gate against real vLLM on CPU (E-18)
cd experiments/rq1_soundness_gate
VLLM=~/Downloads/vllm_hybrid_state_gap_harness/vendor/vllm
PYTHONPATH=$VLLM .venv/bin/python soundness_gate.py

# RQ3 (E-19)
cd experiments/rq3_state_budget && .venv/bin/python rq3_budget.py

# NCCL audit (E-20, E-22), needs only a clone
git clone --depth 50 https://github.com/NVIDIA/nccl.git /tmp/nccl
python3 experiments/rq2_nccl_source_audit/audit_nccl_ce_selection.py /tmp/nccl

# layout survey (E-21)
cd experiments/rq3b_layout_survey
.venv/bin/python probe_layouts.py && .venv/bin/python layout_survey.py
.venv/bin/python verify_suffix_skip.py

# current vLLM source audit for R-24; Docker network access is required
./experiments/versioned_session_commit/run_source_audit_docker.sh

# E-32 and E-33 did not run. See the R-27 paper audit.
```

### 9.4 Environment and external dependencies

- Environment recorded by E-17: macOS 26.5.2 arm64, Python 3.12.13, torch 2.14.0,
  transformers 5.17.0, MPS, float16, eager attention, Qwen3.5-0.8B-Base.
- `.venv` in `hybrid_state_recovery_v1_3`, `rq3_state_budget` and `rq3b_layout_survey` is a
  symlink to `~/Downloads/mac_hybrid_selective_recovery_poc/.venv`. In `rq1_soundness_gate` it
  points to `~/Downloads/vllm_hybrid_state_gap_harness/.venv`. The system `python3` has
  neither torch nor transformers.
- The pinned vLLM checkout is `~/Downloads/vllm_hybrid_state_gap_harness/vendor/vllm` at
  `e0705731be04aac90838f318fcdae15c56a53080`.
- Verified on 2026-09-23: all three external paths exist, the venv imports torch 2.14.0 and
  transformers 5.17.0, and the vendored vLLM is at the pinned commit.
- **Fragility:** these dependencies live in `~/Downloads`, outside the repo. If that folder is
  cleaned, recreate the env from `experiments/hybrid_state_recovery_v1_3/requirements.txt`
  and re-clone vLLM at the pinned commit.
- E-26 data and environment (2026-10-04): `~/Downloads/agent-llm-traces/` holds Exgentic shards 11 to
  20 of `DiscoPosse/agent-llm-traces` (528 MB, CDLA-Permissive-2.0), `shard_index.json`, and a
  separate `.venv` with pyarrow 25.0.1 (Python 3.14). Re-download with the URLs in `probe_shards.py`
  there if it is cleaned.

### 9.5 Version control state

Last commit is 4bd50d5 on 2026-05-21. Everything since (documents, experiments, dossier,
ideas) is untracked, and `README.md` shows as deleted. Nothing in the research history is
under version control. Committing is a decision for the user (D-13 area).

---

## 10. Decision log

"Decided by: user" means the user chose it. "Recommendation" means an AI-assisted session
proposed it and no acceptance is recorded.

| ID | Date | Decision or recommendation | Decided by | Rationale | Status |
|---|---|---|---|---|---|
| D-01 | 2026-05-20 | Commit to one project by end of June after discussing five options with Prof. Sengupta | User (plan in README) | Domain still broad | Not evidenced as done. Superseded by the systems pivot |
| D-02 | 2026-05-20 | Treat the LinkedIn-aligned option (I-04) as one option among several, not a commitment | User | Dependency on an industry collaboration | Superseded |
| D-03 | 2026-09-16 | Withdraw all V1 synthetic GO and KILL verdicts | User challenge, AI agreed | Suite was mostly hard-coded | Done |
| D-04 | 2026-09-16 | Do not choose a thesis until one candidate passes novelty, simple-baseline, oracle and feasibility gates | Recommendation (handoff) | Avoid preserving ideas the evidence killed | Standing rule |
| D-05 | 2026-09-16 | Run hybrid-state falsification and a collective-engine dossier in parallel | Recommendation (handoff) | Neither yet had comparable evidence | Done (iteration 4) |
| D-06 | 2026-09-17 | Kill I-16 on performance grounds | Recommendation (dossier) | Three independent arguments | Standing |
| D-07 | 2026-09-17 | Kill I-15 (RQ3) and I-17 | Recommendation (dossier) | Two methods agree simple policies win | Standing |
| D-08 | 2026-09-18 | Kill I-20 as a thesis, keep as a finding | Recommendation (dossier) | Needs pretraining to validate | Standing |
| D-09 | 2026-09-18 | Proceed with I-19 as the thesis unless NVLink hardware appears. Keep I-18 warm. Do not attempt both | Recommendation (`RESEARCH_STATUS.md`) | Mac-feasible, gate passed | Superseded by D-11 |
| D-10 | 2026-09-22 | Primary direction I-23, MEASURE-FIRST. Robotics not chosen. Avoid hardware-critical theses. 12-week Oct to Dec plan | Recommendation (handoff), constraints stated by user | Career fit, one-GPU experiments | Active, narrowed by D-11 |
| D-11 | 2026-09-23 | I-19 KILL as thesis. I-06 KILL. I-23 narrowed MEASURE-FIRST. I-18 low priority. Nothing is a GO | Recommendation (R-05) | Section 6.2 | **Awaiting user** |
| D-12 | 2026-09-23 | Create this journal as the single source of truth and log every future idea in it | User | Keep the whole research journey in one place | Done |
| D-21 | 2026-09-30 | Have two subagents review the agentic-harness literature review, then verify disputed items and apply fixes | User | Catch correctness errors before citing the review in the thesis | Done |
| D-22 | 2026-09-30 | Have two subagents review the GPU-efficiency literature review with a five-point prize for the most serious findings, then verify disputed items and apply fixes | User | Catch correctness errors before citing the review in the thesis | Done (R-07). Award of the five points left to the user |
| D-23 | 2026-10-04 | I-41 (ingest structurer) KILL as a thesis mechanism, I-42 KILL, I-43 DEPRIORITIZED. Do not pursue unless the user wants the short I-43 measurement on TraceLab data, which needs permission to download it | Recommendation (R-08), accepted by the user 2026-10-05 | Mechanism published, cache premise false in E-24 and TraceLab, rewriting conflicts with verbatim reuse | Accepted (see D-26) |
| D-24 | 2026-10-04 | Download the Exgentic SWE-bench shards (528 MB) and run the I-43 check | User (chose "SWE-bench part only" after TraceLab proved unusable) | TraceLab's public release has no tool text | Done (E-26) |
| D-25 | 2026-10-04 | I-43 passed E-26 (M = 43.7%). Treat it as an optional short note, not a thesis; write it only after the retention-policy simulation, and only if the user wants a side publication | Recommendation | Harness-level, model-dependent (Opus 4.5 at 29%), multi-user real-session data missing | Superseded by D-26 |
| D-26 | 2026-10-05 | Drop the whole line: I-41 KILL accepted, I-43 dropped (no note, no simulation) | User | Mechanisms are already handled by published and shipping work (ACON, TokenPilot, AgentDiet, SWE-Pruner, Anthropic context editing); only the verbatim-reuse measurement was new, and it is harness-level, outside the thesis focus | Done |
| D-27 | 2026-10-06 | Write all content in ASD-STE100 (Simplified Technical English): chat replies, journal entries, dossiers, READMEs, commit messages, code comments. Rule added to `agents.md` | User | Clear, consistent technical writing | Done |
| D-28 | 2026-10-06 | Run I-48 first (about 2 weeks on the Mac, reuses E-23). Then I-46 with I-47 as a sub-study on one simulator. Keep I-49 on HOLD until node-level failure data exists. Do a preprint scoop check before committing to any of them | Recommendation (R-09) | Mac-testable, reuses existing harness, close to the target roles | Superseded by D-30 (R-10 killed I-48, I-46, I-47) |
| D-29 | 2026-10-07 | Prefer PagedAttention-style mechanism-transfer ideas, not benchmarking papers. Use the rubber-duck method for every experiment, and have a different model review each experiment adversarially | User | The user prefers visible mechanism gains and wants independent checks of each result | Active (Section 2.5) |
| D-30 | 2026-10-07 | Predeclare E-27 and run the I-58 test (about 2 days on the Mac), step 1 first (per-head smoothness G). Keep I-59 and I-60 as simulator-only backups | Recommendation (R-10), accepted by the user 2026-10-07 ("yes, start the I-58 test, review the testing before starting") | I-58 is the only strong route after the different-model review | Accepted. E-27 started after the R-11 pre-run reviews |
| D-31 | 2026-10-07 | Accept the I-58 KILL (E-27, R-12). Next step options: a weak backup (I-59 or I-60, simulator only), a new route search, or I-23 when a GPU arrives | Recommendation; the user chose option 2, a new route search ("let's do 2 for now!") | E-27 and R-12 agree; no setting tested on the Mac rescues the idea | Decided by the user: route search (R-13) |
| D-32 | 2026-10-07 | Next: a different-model review of I-80, then its simulator premise check (does a spike cause a lasting collapse?). Backups: I-59, I-60. Or stop searching and execute on one direction, or I-23 with a GPU | Recommendation (R-13); the user chose option 1 on 2026-10-07 ("Option 1 (review I-80, then the simulator premise check)"). The user also allowed combinations of existing ideas ("we can mix multiple ideas for solving current problems or bottlenecks") | Direct transfers are crowded; I-80 is the only open candidate this round | Decided by the user: option 1. Combinations allowed |
| D-33 | 2026-10-08 | After E-28: accept the I-80 KILL. Options: (a) a small predeclared E-29 (spikes 2 to 4 x N*, loads 0.9 to 1.2 x knee, 12,000 s, a preemption-rate state signal, and K3 run regardless of K1), to decide whether one static cap fails across mixes; (b) a third route search that uses the lessons of E-27 and E-28; (c) stop searching and wait for a GPU to run I-23 Test A; (d) write the metastability measurement as a short workshop note | Recommendation (R-16, main session). Main session leaned to (c) with (a) as an optional 1-day check. The user chose (b) on 2026-10-08 ("Lets go ahead with B") | E-28 and R-16 show the phenomenon is real in the model but the fix is known; K3 is the only open question with data | Decided by the user: (b), a third route search in a new session |
| D-34 | 2026-10-08 | After R-17: do not run a fourth route search of the same kind. Put the I-83 key dump (about 3 GPU-hours on one 8-GPU H200 node) on the GPU plan as a side task next to I-23 Test A. Use the time until GPU access for the grant applications (D-18, D-19). Optional alternatives: (d) of D-33 (workshop note on E-28), or the I-19 note | Recommendation (R-17, main session). Not a decision | R-10, R-13 and R-17 found no Mac-testable survivor; the one open gap needs real model internals | **Open, waiting on the user** |
| D-35 | 2026-10-08 | The user will self-fund AWS GPUs to validate ideas; first round 1 x H100, about $20; prepare everything on the Mac first. Scope broadened: ideas need not transfer old mechanisms; algorithmic GPU-efficiency optimizations (data structures and algorithms) are in scope; benchmarking papers stay out | User (recorded in a parallel session's memory notes on 2026-10-08; copied here by the R-18 session, wording second-hand, to be confirmed by the user) | Three route searches found almost nothing | Decided by the user |
| D-36 | 2026-10-08 | Run an orphan-finding scan (R-18): read reports of measured waste (issues, characterization papers) instead of running systems first | Proposed by the main session; the user approved ("yes, go ahead!") | The user asked whether reading can replace running systems | Decided by the user. Done: no survivor |
| D-37 | 2026-10-08 | After R-18: stop mining literature and issues for gaps. Spend the first self-funded H100 hours (D-35) on I-23 Test A, which needs Hopper FP8 and is prepared on the Mac (E-23 harness). Start new ideas from a mechanism or algorithm with a clear invariant, and use papers only to size the waste | Recommendation (R-18, main session and reviewer). Not a decision | R-10, R-13, R-17 and R-18 found no open Mac-testable gap; I-23 is the strongest live idea and fits one H100 | **Open, waiting on the user** |
| D-38 | 2026-10-11 | Run an assumption-removal search (R-19) | Proposed by the user ("Lets try one more time, this time try to think of removing assumptions"); approved 2026-10-11 ("sure, go ahead!") | Paper and issue mining found nothing | Decided by the user. Done |
| D-40 | 2026-10-16 | After R-20: accept the verdicts (I-118 MEASURE-FIRST low, I-116 WEAK, others KILL). If the user wants I-118: ask the GPU provider about root and power telemetry, then freeze E-30 after a rubber duck and a Sonnet pre-run review; about 2 GH200 hours. Otherwise keep D-37 (I-23 Test A first) | Recommendation (R-20). Not a decision | The seed topics are crowded; I-118 is the only open, cheap question and is likely a measurement note | **Open, waiting on the user** |
| D-39 | 2026-10-11 | After R-19: run the I-106 premise check on the Mac (OLMo-2 1B RLVR revisions, about 3 GB each), after a rubber duck, frozen kill rules and a Sonnet pre-run review; or skip it and go to I-23 Test A (D-37) | Recommendation (R-19). Not a decision. The download needs the user's approval | I-106 is narrow; reviewer gives about 40% to pass | **Open, waiting on the user** |
| D-41 | 2026-10-17 | Spend the first H100 session on I-23 Test A. If time remains, reproduce the I-120 vLLM failure. Do not implement either full system until its gate passes | Recommendation (R-21). Not a decision | I-23 has the cheapest decisive oracle test. I-120 has stronger failure evidence but very high RFC scoop risk | **Open, waiting on the user** |
| D-42 | 2026-10-17 | Continue the R-21 process until three to five total new gaps survive. Count I-120 as the first survivor | User | The user wanted repeated narrowing, not one search pass | Done. R-22 stopped at three total survivors |
| D-43 | 2026-10-17 | Run the I-122 CPU state-machine gate first. Keep I-23 as the first H100 test. Run the I-123 CPU oracle before any GPU implementation | Recommendation (R-22). Not a decision | I-122 has direct failures and needs no GPU. I-123 has a cheap oracle. I-120 has high scoop risk | Superseded by R-23 and R-24. E-31 is invalid. I-131 replaces I-122 |
| D-44 | 2026-10-17 | Run the I-122 local experiment in Docker, not directly on the Mac host | User | Isolate dependencies and reduce interference from host use | Decided. R-24 ran in Docker. E-31 did not run because its design is invalid |
| D-45 | 2026-10-19 | Stop I-131 as a thesis direction. Do not run E-32 or E-33 for thesis evidence. Audit I-123 next | Recommendation (R-27); accepted when the user said "go ahead" | Two Sonnet reviews found circularity. Full-history audit found no recurrence after narrow fixes | Decided by the user. R-28 completed the I-123 audit |
| D-46 | 2026-10-19 | Do not run the old I-123 CPU oracle. Review E-34 before implementation. Kill I-132 if it is only standard checkpoint composition | Recommendation (R-28). Not a user decision | R-29 found standard checkpoint composition and returned KILL_IDEA | Review condition met. I-132 is KILL. User acceptance is still open |
| D-47 | 2026-10-19 | After R-30: for Mac work, run the I-59 premise check on the vLLM CPU backend in Docker (rubber duck, frozen kill rules, Sonnet pre-run review first), or prepare I-23 Test A on the Mac. Do not run the I-106 check | Recommendation (R-30). Not a decision. The user chose Mac tests for now | CacheReforge covers I-106; I-60 is crowded; I-59 is the only open Mac-testable mechanism idea | Decided by the user 2026-10-19 ("Start the I-59 premise check on the vLLM CPU backend in Docker"). Done as E-35: I-59 KILL |
| D-48 | 2026-10-20 | E-35 ran vLLM's real scheduler class with a fake model runner, not the full CPU backend that the user named. Accept this substitution, or run a one-cell CPU-backend fidelity check (build the vLLM CPU backend for linux/arm64 and compare its scheduler output stream with the fake-runner stream) | Recommendation (R-31, R-32): accept; the CPU backend runs the same synchronous scheduler class, and the fidelity check cannot change the premise answer | It can only add wall-clock sizes and frontend plumbing, which do not bear on a missing dependency | **Open, waiting on the user** |
| D-49 | 2026-10-20 | After E-35: the Mac-only mechanism ideas are used up (I-106, I-60, I-59 KILL). Plan I-23 Test A on the Mac and rent one H100 (D-35), as D-37 and D-41 recommend; or write the I-19 short note | Recommendation. Not a decision | Every live idea except I-116 now needs an NVIDIA GPU; I-23 is the strongest | Decided by the user 2026-10-20 ("Do whatever you can to test locally to test or plan, and keep the ideas up-to-date, so that Once we have the GPUs, we will continue on those."). Done as R-35 |
| D-50 | 2026-10-20 | After R-33: no further broad frontier sweep. Use the first H100 session for I-23 Test A (with a Sonnet pre-run review and a cost cap). Treat I-120 as closed | Recommendation (R-33). Not a decision | R-33 found direct prior work for 8 of 8 new transitions, and PR 53298 took I-120 | **Open, waiting on the user** |
| D-51 | 2026-10-20 | Approve E-36 Session 1: one H100 (Lambda, about $4.29 per hour), hard caps of 4 hours and $18, after the R-34 amendments and the Mac dry run | Recommendation (R-35). Not a decision | It is the cheapest decisive step for I-23, the strongest live idea | **Open, waiting on the user** |
| D-52 | 2026-10-20 | Approve E-30: one GH200 for 2.5 hours, only after the provider confirms root, module power telemetry and writable CPU caps | Recommendation (R-35). Not a decision | I-118 is low priority; the preflight can fail on a cloud VM | **Open, waiting on the user** |
| D-53 | 2026-10-20 | After R-36: stop literature-based gap searches (route sweeps, orphan scans, frontier passes and limitations mining all returned no MEASURE-FIRST gap). Put the effort into the prepared GPU runs (I-23 Test A, I-118 E-30). Add I-143 only if a second GPU joins a rental | Recommendation (R-36). Not a decision | R-36: 7 of 8 limitation-derived gaps KILL, 1 WEAK | **Open, waiting on the user** |
| D-54 | 2026-10-21 | Run an algorithmic-optimization sweep of the ML infrastructure stack with the research skills, and use the PyTorch Conference NA 2026 schedule as an extra source. Done as R-37, E-37 and R-38 | User ("Think in terms of optimizing anything, any layer, that is not optimized, think fundamentally like you are solving competitive algorithmic problems ... find out places where there is a research gap, or it can be optimized more") | The user chose one more literature-based search, against the D-53 recommendation | Decided. Done |
| D-55 | 2026-10-21 | After R-37 and R-38: repeat D-53. Do not run another gap sweep of any kind (route, orphan, frontier, limitations or algorithmic). Spend the next effort on the prepared GPU runs (D-51 E-36 for I-23; D-52 E-30 for I-118). Optional career work outside the thesis: review or benchmark the open SGLang lazy-heap PRs (38371, 38454) or extend them to the Rust core | Recommendation (R-38). Not a decision | Since R-17, sweeps have produced only WEAK or low-confidence candidates (I-83, I-118, I-143). R-37 found real slow paths, but each one was already fixed, in an open PR, or in a 2026 preprint | **Open, waiting on the user** |
| D-56 | 2026-10-22 | Run the GPU experiments on Runpod instead of local. Keep all details about the hardware of each run (Pod record, `experiments/runpod/hardware_record.sh` output, a row in `experiments/runpod/HARDWARE_LOG.md`, hardware in each E-nn "Setup" line). Changes the provider in D-35 (AWS) and the runbook (Lambda) for E-36. E-30 needs root and power caps, which Runpod Pods do not give, so it stays on a provider with root. Each paid run still needs its own spend approval (D-51, D-52) | User ("for now, lets use runpod for our experiments instead of local. keep all the details about the hardware we are using.") | Runpod is set up and connected in Claude Code | Decided by the user |
| D-57 | 2026-10-22 | After R-39: approve the E-36 A2 plan on one Runpod H100 SXM Secure Cloud Pod ($3.99/h on 2026-10-22): the full session with Phase B (Pod limit 6 h 40 min, at most about $27) or Phase A only (about $11 to $14). Put at least $35 of credit on the account first, because a $0 balance terminates the Pod and its volume disk | Recommendation (R-39 and the main session). Not a decision | The user asked for maximum data per dollar and accepted a longer run; Phase B reuses the paid setup of the same Pod | **Open, waiting on the user** |
| D-58 | 2026-10-25 | Write experiment standards that follow frontier-lab practice, so that an agent can run the GPU experiments without issues: `docs/experiments/` (PROTOCOL, MEASUREMENT, RUNPOD, templates), the run ledger and compare tool in `experiments/ledger/`, the E-36 ledger adapter, a rehearsal stage before each paid run, and a local agent skill | User ("lets create documents so that claude can run these experiments properly without any issues. Proper flows where we follow industry standards of frontier labs during experimentation.") | The user wants reliable runs and evidence of improvement or degradation for each change | Decided by the user. Done |
| D-59 | 2026-10-25 | The Runpod account is shared with other sessions. Label every thesis resource (`msthesis-<experiment>-<purpose>-<yyyymmdd>`, env `THESIS_TAG=msthesis`), register its id in `experiments/runpod/pods.jsonl` at creation, and change only registered resources; everything else is read-only. The balance check counts other running spend | User ("label the thesis related things with a different identifier so that we only handle those things, as other sessions will be working on the same account.") | Another session's H200 Pod used the same account and balance on 2026-10-25 | Decided by the user. Done |
| D-60 | 2026-10-25 | Approve D-57 (the full E-36 A2 session on one H100 SXM, Pod limit 6 h 40 min, cap about $27) and D-51, with the L4 rehearsal first (cap 1 h, about $1) | User (picked "Full session (Recommended)" for the question "Which E-36 plan do you approve for the H100 run after the L4 rehearsal (rehearsal cap: 1 h, about $1)?") | Runpod H100 SXM $3.99/h on 2026-10-25 | Decided by the user. Rehearsal done; H100 run waits on amendment A3 |
| D-61 | 2026-10-26 | After E-36 Session 1 and R-42: preregister Test A2 for I-23 with a tolerance contract (teacher-forced logprob distance on R's tokens, GSM8K accuracy as the primary metric, kill rules frozen first), including Marlin W8A16 and the fused activation-quantization variant; keep Session 1 labelled INCONCLUSIVE. Alternative: close I-23 for Hopper block-FP8 under exact match and move to NVFP4 on Blackwell | User (chose option 1, "Test 'good enough' instead of 'identical'", with "go ahead!") | Exact match makes every alternative unsafe; R is the fastest FP8 backend; only a tolerance contract can show usable headroom | Decided by the user 2026-10-26. Test A2 planned as E-38 |
| D-62 | 2026-10-27 | After the incomplete E-38 run: rerun the missing parts (cell M scoring and the Marlin MoE speed A/B first, then cell D) on the kept network volume with the fixed code, cap about $10; or stop and delete the volume | User ("D-62: rerun the missing E-38 parts (cell M scoring and the Marlin MoE speed A/B first, then cell D) on the kept volume with the fixed code, cap $10.") | A1 passed for every MoE candidate, so the Marlin speed test now decides GO or KILL | Decided by the user 2026-10-27. Rerun in progress |
| D-63 | 2026-10-27 | After the E-38 rerun (KILL) and R-44: close I-23 for Hopper block-FP8 at these configurations; keep NVFP4 on Blackwell as the only open I-23 cell; log the Marlin MoE batch 4 to 8 hint as an open lead | User ("Go ahead" on "close I-23 for Hopper (D-63) and plan the next question") | Every real backend fails A2, and no backend is 3% or more faster | Decided by the user 2026-10-27, after E-39 (INCONCLUSIVE, gain under 3%) |
| D-13 | pending | Choose the thesis direction | User | After the Test A gate, or earlier | **Open** |
| D-14 | pending | Commit the repo to git | User | 4 months of untracked history | **Open** |
| D-15 | 2026-09-23 | Pursue a compute grant. No GPU access yet | User | Every live idea except the killed ones needs an NVIDIA GPU | Active. Program, deadline, decision date unknown |
| D-16 | 2026-09-23 | Add a pointer to this journal in `agents.md` so other AI tools see it | User | One source of truth across tools | Done |
| D-20 | 2026-09-24 | Archive superseded files in `archive/` instead of deleting them, with a README row for each | User | Keep history and make obsolete files obvious | Done |
| D-19 | pending | Review, edit and submit the drafted applications in `grants/`, starting with the advisor email. NAIRR requests Horizon (Blackwell) if selectable, ACCESS Explore requests Delta H200 | User | Blackwell and Hopper cells are covered by different programs | **Open, drafted 2026-09-24** |
| D-17 | pending | Adopt the Section 1.7 plan: Mac-side harness pilot now, compute tiers for the grant, choose bridge rental vs waiting by about 2026-10-15 | User | Grant lead time may not fit the Oct to Dec window | **Open, proposed 2026-09-24** |
| D-18 | pending | Choose which compute programs to apply to and in what order, using Appendix B.3. Suggested: ACCESS Explore and NAIRR Start-Up now, NVIDIA proposal with the advisor as applicant | User | Fastest routes to Hopper time | **Open, proposed 2026-09-24** |

---

## Appendix A. Source file index

| File | Date (file time or text) | Contains |
|---|---|---|
| `README.md` (git HEAD only) | 2026-05-19 to 21 | Phase 0 log: domain, five options, LinkedIn discovery, reading list |
| `stateful_elastic_inference_thesis_dossier.docx` | file 2026-09-16 17:51 | I-06, E-01 to E-03, POC plan, gates |
| `AI_Infra_Thesis_Handoff_Package/AI_Infra_Thesis_Master_Handoff_Dossier.docx` | 2026-09-16 | Constraints, candidate map, hybrid-state evolution, E-07 to E-11, R-01, collective seed |
| `AI_Infra_Thesis_Handoff_Package/AI_Infra_Thesis_Experiment_Evidence_Ledger.docx` | 2026-09-16 | Chronology A, V2 caveats, E-05 to E-11 raw data, novelty threats, work queue |
| `AI_Infra_Thesis_Handoff_Package/AI_Infra_Thesis_Handoff_README.md` | 2026-09-16 | One-paragraph state and hard warnings |
| `AI_Infra_Thesis_Handoff_Package/*.zip`, `.csv`, `SHA256SUMS.txt` | 2026-09-16 | Result bundles and integrity checksums |
| `agents.md` | file 2026-09-16 19:26, edited 2026-09-23 | Focus areas, background, stack, communication style, and a "Read first" pointer to this journal |
| `.claude/skills/research-gap-dossier/` | file 2026-09-16 20:00 | Review workflow, rubric, adversarial checklist, dossier template |
| `dossier/RESEARCH_GAP_DOSSIER_2026-09-16.md` | 2026-09-16, last edit 09-17 11:01 | Iteration 4: E-12 to E-21, R-02, R-03, I-16 to I-20 |
| `archive/ideas/README.md`, `01-state-reuse-soundness.md`, `02-collective-engine-arbitration.md` | file 2026-09-17 11:08, archived 2026-09-24 | Briefs for I-19 and I-18, closed-ideas list (stale). Originally in `ideas/` |
| `archive/dossier/RESEARCH_STATUS.md` | 2026-09-18, archived 2026-09-24 | Status and D-09. Superseded. Originally in `dossier/` |
| `archive/README.md` | 2026-09-24 | Table of archived files: original path, reason, replacement |
| `experiments/**` | 2026-09-16 to 09-18 | Code and results for E-12 to E-21 |
| `LLM_Inference_Thesis_Research_Handoff_2026-09-22.docx` | file 2026-09-22 13:18 | I-23 and I-24 to I-32, R-04, constraints, [R1] to [R29] |
| `dossier/ADVERSARIAL_REVIEW_2026-09-23.md` | 2026-09-23 | R-05 full write-up, source ledger, Test A and B protocols |
| `grants/**` | 2026-09-24 | Drafts: NAIRR Start-Up proposal (md and 3-page pdf), ACCESS Explore request, advisor email, advisor letter (md and docx), NVIDIA outline, build script |
| `experiments/safedispatch_oracle/**` | 2026-09-24 | E-23 harness: contracts, INT8 fault injection, estimator, model pilot, unit tests, frozen `prompts.json`, per-attempt results |
| `dossier/LIT_REVIEW_AGENTIC_HARNESSES_2026-09-30.md` | 2026-09-30 | Evidence-graded literature review of agentic harnesses, 12 claims, 55 sources |
| `dossier/LIT_REVIEW_GPU_EFFICIENCY_2026-09-30.md` | 2026-09-30 | Evidence-graded review of GPU efficiency (training and inference), 16 claims (C1 to C16), 54 sources (S1 to S54). Standalone, same format as the agentic-harness review; chart data as tables. Companion Claude Doc linked in its header. Revised after R-07: 8 consensus, 7 working hypothesis, 1 speculative; the companion Claude Doc was not updated and may still hold the pre-R-07 errors |
| `dossier/DOSSIER_I-41_INGEST_STRUCTURER_2026-10-04.md` | 2026-10-04 | R-08 dossier: I-41 ingest structurer and variants I-42 to I-45, E-24 and E-25 results, 36 sources |
| `experiments/structurer_ingest_screen/**` | 2026-10-04 | E-24 trace-study, E-25 screen and E-26 public-trace scripts, aggregate results only (no transcript text) |
| `dossier/NEW_DIRECTIONS_R-09_2026-10-06.md` | 2026-10-06 | R-09: 12 cross-domain candidates, 4 survivors (I-46 to I-49), peer-reviewed sources only, preprints listed as scoop signals |
| `dossier/ROUTES_R-10_2026-10-07.md` | 2026-10-07 | R-10: mechanism-transfer routes, different-model review, I-58 to I-75 |
| `archive/HANDOFF_2026-10-07.md` | 2026-10-07 | Handoff for a new session (superseded 2026-10-08): rules, current state, D-32 options with steps, environment, key files, a ready prompt |
| `HANDOFF_2026-10-08.md` | 2026-10-08 | Handoff for a new session: rules, current state, D-33 options, environment, key files, a ready prompt |
| `dossier/ROUTES_R-17_2026-10-08.md` | 2026-10-08 | R-17 route search 3: method, environments checked, three rounds, I-82 to I-94, the I-83 premise check proposal |
| `dossier/REVIEW_R-17_SHORTLIST_2026-10-08.md` | 2026-10-08 | Sonnet attack of the six round 1 candidates (part of R-17) |
| `dossier/R-17_ROUND3_SONNET_CANDIDATES_2026-10-08.md` | 2026-10-08 | Sonnet round 3: 28 candidates in training, collectives, numerics, agent infra, cluster ops, with killing sources |
| `dossier/ORPHANS_R-18_2026-10-08.md` | 2026-10-08 | R-18 orphan-finding scan: method, issue findings, paper harvest screen, verdicts, lesson |
| `dossier/R-18_ORPHAN_FINDINGS_PAPERS_2026-10-08.md` | 2026-10-08 | Sonnet harvest: 31 findings from 30 characterization papers with fix status |
| `dossier/REVIEW_R-18_ORPHANS_2026-10-08.md` | 2026-10-08 | Sonnet attack of five orphan rows, with corrections to the harvest |
| `dossier/ASSUMPTIONS_R-19_2026-10-11.md` | 2026-10-11 | R-19: 40 assumptions with triage, P1 check |
| `dossier/R-19_PRIORART_P2-P4_2026-10-11.md` | 2026-10-11 | Sonnet prior-art check of P2 to P4 |
| `dossier/R-19_PRIORART_P5-P7_2026-10-11.md` | 2026-10-11 | Sonnet prior-art check of P5 to P7 |
| `dossier/REVIEW_R-19_SHORTLIST_2026-10-11.md` | 2026-10-11 | Sonnet attack of P2 to P4, I-106 premise check design |
| `dossier/REVIEW_R-14_I-80_2026-10-07.md` | 2026-10-07 | R-14: Sonnet adversarial review of I-80 and four combinations, with a toy simulator |
| `dossier/REVIEW_E-28_PRERUN_2026-10-07.md` | 2026-10-07 to 2026-10-08 | R-15: two Sonnet pre-run reviews of E-28 (blockers, fixes, second pass) |
| `dossier/REVIEW_R-16_E-28_POSTRUN_2026-10-08.md` | 2026-10-08 | R-16: Sonnet post-run review of E-28, with the hysteresis reviewer check |
| `experiments/metastable_admission/` | 2026-10-07 to 2026-10-08 | E-28: trace extraction, simulator, unit tests, runner, frozen decision script, results, README with both rubber ducks, R-16 reviewer scripts |
| `experiments/noise_shaped_vcache/` | 2026-10-07 | I-58: E-27 code (vcache.py, e27_run.py, e27_decide.py, prep_texts.py, test_vcache.py), results, compressed texts (data/), and the R-10 reviewer's synthetic check |
| `dossier/GAPS_R-20_2026-10-16.md` | 2026-10-16 | R-20: seed-topic gap search (threads versus processes, KV cache poisoning, GPU noisy neighbor), verdicts for I-110 to I-119, E-30 draft |
| `dossier/FRONTIER_R-21_2026-10-17.md` | 2026-10-17 | R-21: current-frontier review, killed ideas, four survivors, ranking, and the exact I-23 first experiment |
| `dossier/FRONTIER_R-22_2026-10-17.md` | 2026-10-17 | R-22: continued frontier review, I-122 to I-130, three total survivors, source ledgers, ranking, and the I-122 first experiment |
| `dossier/FRONTIER_R-33_2026-10-20.md` | 2026-10-20 | R-33: fresh frontier pass, I-133 to I-140 (all KILL), I-120 closed by vLLM PR 53298, I-23 re-check, ranking, and the Test A design with R-33 changes |
| `dossier/LIMITATIONS_R-36_2026-10-20.md` | 2026-10-20 | R-36: limitations and future-work mining of all 496 cited papers, eight follow-up gaps (I-141 to I-148), verdicts and sources |
| `dossier/REVIEW_E-31_PRERUN_2026-10-17.md` | 2026-10-17 | R-23: Sonnet pre-run review that invalidated E-31 before execution |
| `dossier/SOURCE_AUDIT_I-122_2026-10-17.md` | 2026-10-17 | R-24: current-source audit, I-122 pivot, and I-131 definition |
| `dossier/REVIEW_E-32_PRERUN_2026-10-17.md` | 2026-10-19 | R-25: first Sonnet review of E-32 |
| `dossier/REVIEW_E-32_AMENDMENT_2026-10-19.md` | 2026-10-19 | R-26: second Sonnet review of amended E-32 |
| `dossier/PAPER_AUDIT_I-131_2026-10-19.md` | 2026-10-19 | R-27: action-sequence and full-history blame audit; kills I-131 as thesis |
| `dossier/SCOOP_AUDIT_I-123_2026-10-19.md` | 2026-10-19 | R-28: two-pass I-123 scoop audit, pivot, and I-132 definition |
| `dossier/REVIEW_E-34_PRERUN_2026-10-19.md` | 2026-10-19 | R-29: independent pre-run review that kills I-132 and E-34 |
| `experiments/versioned_session_commit/` | 2026-10-17 | Invalid E-31 design plus the reproducible Docker source audit for R-24 |
| `experiments/session_state_metamorphic/` | 2026-10-17 to 2026-10-19 | Killed E-32 design; not implemented or run |
| `experiments/session_state_corpus_pilot/` | 2026-10-19 | Cancelled E-33 design; not run |
| `experiments/crash_consistent_duplex_epoch/` | 2026-10-19 | E-34 design and read-only Sonnet review wrapper; not implemented or run |
| `experiments/priority_inheritance_premise/` | 2026-10-19 to 2026-10-20 | E-35: design, amendments A1 and A2, result, Dockerfile (Python-only vLLM at 3897f95c), harness, trace check, supplement, model config, results with manifests |
| `dossier/REVIEW_E-35_PRERUN_2026-10-19.md` | 2026-10-19 | R-31: Sonnet pre-run review of E-35 (DO_NOT_RUN_CODE_DECIDES) |
| `dossier/REVIEW_E-35_POSTRUN_2026-10-20.md` | 2026-10-20 | R-32: Sonnet post-run review of E-35 (CONFIRM_WITH_SCOPE_LIMITS) |
| `dossier/GPU_RUNBOOK_2026-10-20.md` | 2026-10-20 | R-35: GPU session order, costs, commands, and the status of the GPU-gated ideas |
| `dossier/REVIEW_E-36_E-30_PRERUN_2026-10-20.md` | 2026-10-20 | R-34: Sonnet pre-run review of E-36 and E-30 (AMEND_THEN_RUN for both), with main-session source checks |
| `experiments/safedispatch_testA/` | 2026-10-20 | E-36: I-23 Test A Session 1 package (design with amendment A1 and the Mac dry-run record, prompts, run, compare with tests, session script, `dryrun/` with the last dry-run result; `inventory.py` unused after A1) |
| `experiments/superchip_power_coupling/` | 2026-10-16 to 2026-10-20 | E-30: I-118 package (design with amendment A1, preflight, sweep driver, analysis with tests) |
| `dossier/ALGORITHMIC_R-37_2026-10-21.md` | 2026-10-21 | R-37: algorithmic-optimization sweep (I-149 to I-164), PyTorch Conference 2026 session map, code facts at three commits, verdicts and sources |
| `dossier/REVIEW_R-37_SONNET_2026-10-21.md` | 2026-10-21 | R-38: Sonnet hostile review of R-37 and post-run review of E-37 |
| `experiments/control_plane_complexity_screen/` | 2026-10-21 | E-37: predeclared README, screen script, result JSON (SGLang evict heap rebuild, vLLM block hashing) |
| `experiments/runpod/` | 2026-10-22 | D-56: Runpod rules, `hardware_record.sh` (machine record inside a Pod, checked on the Mac), `gpu_microbench.py` (GEMM and copy health check), `HARDWARE_LOG.md` (one row per Pod, append only) |
| `dossier/REVIEW_E-36_A2_PRERUN_2026-10-22.md` | 2026-10-22 | R-39: Sonnet pre-run review of E-36 A2 (AMEND_THEN_RUN), the fix for each finding, Runpod facts, tests |
| `docs/experiments/` | 2026-10-25 | D-58: experiment protocol (stages S0 to S9), measurement standards, Runpod operations and incidents, preregistration and run report templates |
| `experiments/ledger/` | 2026-10-25 | D-58: append-only run ledger (`runs.jsonl`), `ledger.py`, `compare_runs.py`, tests |
| `RESEARCH_JOURNAL.md` | 2026-09-23, updated 2026-10-25 | This file |

---

## Appendix B. Compute and funding programs

Researched 2026-09-24. Amounts, dates and rules change, so verify each on the provider's
own page before applying. Check levels: **official** means I fetched the provider's own
page. **third-party** means an aggregator or a student guide. **search-only** means only a
search snippet. Items I could not confirm are marked "not confirmed".

### B.1 What needs funding and who is applying

| Tier | Hardware | Serves |
|---|---|---|
| 1 (required) | 1 Hopper GPU (H100 or H200), about 1 to 2 weeks | I-23 Test A |
| 2 (adds NVFP4) | 1 Blackwell GPU | I-23 NVFP4 cells |
| 3 (optional) | 2 or more NVLink GPUs, 8-GPU NVSwitch node | I-18 Test B, I-34 |

Applicant profile (official sources): MSCS student at San Jose State University, Department
of Computer Science, which is in the College of Science. Advisor: Prof. Saptarshi Sengupta,
MICoSys Lab, CS Department. The lab's own site lists an NSF ACCESS allocation and a NAIRR
allocation for two other projects. The applicant is a master's student, not PhD, which
excludes some programs below.

### B.2 Shortlist, fastest and most likely first

| Rank | Program | What it offers | Can you apply? | Time to decision | Fits | Check |
|---|---|---|---|---|---|---|
| 1 | The advisor's existing ACCESS and NAIRR allocations | Whatever those projects hold. Multiple users can usually be added to a project, but a different research topic may need a new request or a supplement (not confirmed) | Ask the advisor | Depends on the advisor | Unknown | official (lab site lists them) |
| 2 | **NSF ACCESS, Explore** | 400,000 credits. Review "anytime". Resource NCSA Delta: 4-way A100 40GB with NVLink (100 nodes), 8-way A100 40GB NV12 (6 nodes), **8-way H200 141GB NV18 (8 nodes)**, A40 (100 nodes). Delta GPU allocations are requested in A100 GPU-hours. Delta's user guide charges 1 SU per GPU-hour on A100 and H200 nodes and discounts A40 and MI100 | **Yes.** Graduate students can be PI on Explore and Discover with a signed advisor letter on institutional letterhead | Third-party sources say days, often under 24 hours. The official page says only "anytime" | Tier 1 (H200), Tier 3 (8-GPU NVLink H200 node). No Blackwell | official, turnaround third-party |
| 3 | **NAIRR Pilot, Start-Up** | **3-month** award. Request **one** resource, up to **2,000 GPU-hours** on PSC Bridges-2 GPU, Purdue Anvil AI, SDSC Expanse AI or TACC Vista. Search results and NSF material describe Bridges-2 as 8-GPU H100 SXM5 80GB nodes, Expanse as 4-GPU H100 nodes, Vista as GH200 (search-only, confirm in the catalog) | **Yes.** Graduate students with a faculty supervisor support letter, institutional email. The allocations page mentions a 3-page PDF proposal, the Start-Up page states no page limit | **"Evaluated and a determination made within 3 weeks of submission"** (Start-Up page). The general allocations page says requests by the 15th are decided by the end of the following month, which applies to the longer Research Resources track | Tier 1, and Tier 3 if Bridges-2 (8-GPU H100 nodes). No Blackwell found | official (Start-Up and allocations pages), hardware search-only |
| 3b | **TACC Horizon (Blackwell)**, via NAIRR or NSF LCCF or TxRAS | 2,000 GB200 nodes. Each node has 2 Blackwell GPUs (about 185 GiB HBM3 each per TACC's docs), 1 Grace CPU with 72 cores, 240 GiB LPDDR. 1 SU is 1 node-hour, which is 2 GPU-hours. Queues: `gb` up to 128 nodes for 48 h, `gb-dev` up to 16 nodes for 2 h. TACC's docs name three allocation routes: LCCF, NAIRR Pilot and TxRAS. LCCF large allocations are 125,000 to 500,000 SUs for 6 months, which is not student scale | **User-reported, not confirmed on official pages:** a friend received about 1,000 Blackwell GPU-hours at TACC through a NAIRR proposal (2026-09-24). The NAIRR Start-Up page I read lists Bridges-2, Anvil AI, Expanse AI and Vista, not Horizon, so check the NAIRR resource catalog at submission | Not stated for Horizon. NAIRR Start-Up decisions are within 3 weeks | Tier 1, **Tier 2 (Blackwell, NVFP4)**, and 2 NVLink-connected GPUs per node for a minimal Tier 3 | official (TACC Horizon docs), student route user-reported |
| 4 | **NVIDIA Academic Grant Program, "AI Inference, Agents, and Systems Software"** | 2 DGX Spark units plus up to 30,000 H100 80GB GPU hours, maximum 8 concurrent GPUs, hours expire 6 months after award | **Advisor must apply.** Applicant must be full-time faculty. Graduate students cannot apply directly | Quarterly cycle. **Window opens Thursday 2026-10-01**, no deadline stated on the university page. Notification date not stated. The NVIDIA page itself still says "not currently accepting", which conflicts with the university page | Tier 1, Tier 3 (NVLink not stated), Tier 2 through DGX Spark (see note) | official, dates conflict |
| 5a | Nebius Research Grants and Credits | GPU cloud credits, amount set per proposal. GPU types not stated on the official page | Postgraduate students are listed. Master's students are not named explicitly. One application per person | Reviews through summer 2026, notification "late Q3 to early Q4 2026". The window may be closing | Unknown | official |
| 5b | Lambda Research Grant | Up to $5,000 in Lambda Cloud credits, on-demand H100, B200, A100 instances | Official page does not state eligibility. A third-party source says faculty, PhD students and postdocs, which would exclude a master's student | Rolling. Third-party says days to weeks | Tier 1, Tier 2 (B200). Price per GPU-hour not checked | official plus third-party |
| 5c | Modal for Academics | Official page says up to $10k credits, B200 and H100 without quota. A third-party source says the program is paused with a waitlist and cites $25,000 for researchers | Not stated | Not stated | Tier 1, Tier 2 | official, status conflicts |
| 6a | Chameleon (NSF testbed) | 2 bare-metal nodes with 4 HGX H100 each at CHI@TACC (post dated 2026-04-06), H100 also as VMs at KVM@TACC | Access procedure, PI and student rules, and cost not confirmed | Not stated | Tier 1, partly Tier 3 (NVLink inside the node not stated) | official blog only |
| 6b | CloudLab (NSF testbed) | Clemson `nvidiagh`: 1 NVIDIA H100 (Hopper) per node, 2 nodes. Wisconsin `d8545`: 4 A100 40GB SXM4 per node, 4 nodes | Eligibility not confirmed | Not stated | Tier 1 | official hardware page |
| 7 | SJSU College of Engineering HPC | 2 H100 nodes, 2 A100 nodes, 10 P100 nodes, SLURM, GPU partition time limits stated as 7 days and also 48 hours on the same page. Research access lasts 1 year, requested by the advisor through a Google Form | **Probably not.** "The CoE HPC is only available to be used by CoE faculty and students". Your program and advisor are in the CS Department, College of Science | Not stated | Tier 1 if a College of Engineering faculty member sponsors | official |
| 8 | Google Cloud research credits | PhD students $1,000 a year, faculty and postdocs up to $5,000 once | **No for master's students** (search-only, quotes Google's support text) | 6 to 8 weeks | Too small and slow | search-only |
| 9 | AWS Cloud Credit for Research | Students up to $5,000 | Graduate students are eligible | 90 to 120 days, no expedited review | Too slow | search-only |
| 10 | Azure for Students | $100 credit | Yes | Immediate | Too small | search-only |
| n/a | NVIDIA Graduate Fellowship | Up to about $50,000 to $60,000 (snippets disagree) | PhD only | n/a | Not applicable | search-only |

Notes on the table:
- **DGX Spark and Blackwell.** My general knowledge is that DGX Spark is a small Blackwell
  (GB10) desktop system, and a FlashInfer pull request snippet reports validation on SM121,
  which I read as DGX Spark. Neither the NVIDIA page nor the university page says this.
  Verify before relying on it for the NVFP4 cells. FlashInfer #2723 also reported broken
  NVFP4 MoE backends on SM120 (search-only), so backend maturity is a risk.
- **NVIDIA technology requirement.** Projects "must incorporate NVIDIA models, such as
  Nemotron, Cosmos, or Omniverse wherever possible. All others must make extensive use of
  NVIDIA software and AI distributions, such as NeMo microservices, NVIDIA Dynamo, FLARE, or
  CUDA-X GPU-accelerated libraries." Test A can use Nemotron-family models and CUDA-X
  (cuBLAS, CUTLASS) plus TensorRT-LLM. The topics in scope include "scalable inference
  systems and distributed runtimes" and "inference efficiency via quantization".
- **ACCESS credit to GPU-hour conversion is not confirmed.** The ACCESS pages point to an
  interactive calculator that I could not read, and define a credit as "about one
  core-hour" with GPU-hours costing more. Delta's own user guide (official) charges 1 SU
  per GPU-hour on A100 and H200 nodes. A third-party summary claimed H200 costs 3x an A100
  hour, which Delta's guide does not support. I could not verify either. Read the real
  rate for the Delta GPU resource from `allocations.access-ci.org/exchange_calculator`
  before sizing a request.
- **Blackwell.** TACC Horizon (GB200) is the likeliest route, reported by a friend's award and
  supported by TACC's own docs listing NAIRR as an allocation route, but the NAIRR Start-Up page
  I read does not list Horizon. Confirm in the resource catalog. The GB200 host CPU is Grace, which
  is aarch64, so check that vLLM, FlashInfer and CUTLASS have aarch64 wheels or containers for a
  recent CUDA before counting on it (INTERPRETATION, not checked). Vista (TACC) has no Blackwell:
  its docs list 600 nodes with one H200 each.
- **Right-size the request.** The NVIDIA award allows up to 30,000 H100 hours. Test A needs
  about 1 to 2 weeks of one GPU (R-05 estimate). Asking for far more than the plan justifies
  weakens an application. Fix the hours after the Mac pilot sizes the grid.
- The Sep 2026 NAIRR state: NSF announced on 2026-09-01 that the NAIRR Operations Center will
  be led by the San Diego Supercomputer Center with TACC, and that "all existing
  opportunities remain available on nairrpilot.org".

### B.3 Suggested sequence (proposal, not yet accepted by the user)

| When | Action | Who |
|---|---|---|
| Now | Email the advisor with three questions: (a) can you be added to the lab's existing ACCESS or NAIRR allocations, (b) does the lab or GenAI Lab have GPU servers of its own, (c) will you be the faculty applicant for the NVIDIA program | User |
| Now | Submit an ACCESS Explore request as PI with the advisor letter. Target Delta, H200 first, A100 as fallback. Draft ready: `grants/access_explore_request.md`. Fill the amount from the exchange calculator | User, from the draft |
| Now | Submit a NAIRR Start-Up (advisor support letter, institutional email). In the resource catalog choose TACC Horizon (Blackwell) if selectable, otherwise Vista (H200) or Bridges-2. 500 GPU-hours, 3-month award, decision within 3 weeks. Draft ready: `grants/nairr_startup_proposal.pdf` | User, from the draft |
| From 2026-10-01 | Prepare the NVIDIA proposal (4 pages, template, faculty applicant). Reuse Section 5.8 and Section 1.7 | Advisor as applicant, AI drafts |
| Anytime | Ask about Lambda and Nebius as supplements, using the advisor as applicant if the master's status blocks you | User and advisor |
| Ask SJSU IT | Whether a College of Engineering faculty member could sponsor access to the CoE HPC | Advisor |

If ACCESS Explore works as third-party sources describe, it is a fast route to Hopper (H200),
which shrinks the schedule risk in Section 1.7. It does not give Blackwell. NVFP4 needs
another route.

### B.4 What each application needs

| Program | Materials |
|---|---|
| ACCESS Explore | Project summary, CV or resume as PDF, title, keywords, field of science, signed advisor letter as PDF on institutional letterhead |
| ACCESS Discover | The above plus a one-page project description (1-inch margins, 11 point or larger) |
| NAIRR Start-Up | PDF of at most 3 pages in Times New Roman 11pt or larger with 1-inch margins, in sections A Scientific/Technical Goal, B Estimate of Compute, Storage and Other Resources, C Support Needs (C1, C2), D Team and Team Preparedness. Estimates must be based on prior work or benchmark runs, and the justification must also match the online form entries. Faculty advisor support letter, institutional email, PI CV or biosketch (official proposal instructions, fetched 2026-09-24) |
| NVIDIA Academic Grant | NVIDIA proposal template, at most 4 pages excluding appendices, faculty applicant. University CTTC pre-disclosure applies at some universities (Vanderbilt requires it, SJSU not checked) |
| Nebius | Proposal against their grant priorities. Recipients must agree to co-marketing and public acknowledgement |
| Lambda | Typeform application. Featured research may be published on their site |

### B.5 Open items to verify

- ACCESS credit conversion for Delta H200 and A100 (use the calculator).
- Whether the NVIDIA window has a closing date, and the notification time.
- Confirm the H100 counts and interconnect of Bridges-2, Expanse and Anvil AI in the NAIRR catalog.
- Whether TACC Horizon is selectable in a NAIRR Start-Up, and the allowed size (a friend reports about 1,000 GPU-hours).
- The aarch64 software stack on GB200.
- The award details behind the labmate's proposal (resource, hours, dates). The PDF only shows the request, see B.6.
- Chameleon and CloudLab eligibility and cost for a student on a faculty-led project.
- Whether the lab's existing allocations can host this project.
- Whether the SJSU College of Science or CS Department has its own GPU resources.
- Lambda's official eligibility. Modal's program status. Nebius window still open.

### B.6 A labmate's NAIRR proposal (read 2026-09-24)

The user placed `NAIRR_Proposal.pdf` at the repo root and it now lives in `grants/reference/`. It is another student's work with the same
advisor, so it is gitignored and used for structure only. Names are omitted here.

- **What it is:** a 2-page NAIRR proposal by a labmate and Prof. Sengupta on multi-agent defense for
  time-series sensor attacks. It uses the four NAIRR sections A to D.
- **What it requests (the request, not the award):** 200 GPU-hours on A100 or H100 class instances
  (about $600 at $3 per hour), $20,000 in API credits for an agentic loop, and 200 GB of storage.
- **How it justified the amount:** measured per-batch runtimes on a local laptop GPU (RTX 5060, 8 GB),
  extrapolated to 37.5 GPU-hours for the baseline, then a scaling factor to 200 GPU-hours for sweeps and
  overhead. It also had a concrete readiness section listing completed work.
- **What it does not contain:** anything about Blackwell or 1,000 GPU-hours. The award details (which
  resource, hours, dates) are not in the PDF, so the Horizon route in B.2 is still unconfirmed.
- **INTERPRETATION:** the lab site's NAIRR allocation, "Inferring Latent Attacker Distributions from
  Time-Series Observations", matches this proposal's topic, so joining it would put an unrelated project
  on that allocation. A separate Start-Up request for this thesis is the cleaner route. It also shows
  the advisor has already supported a successful NAIRR request.
- **Applied to our draft:** the A to D structure, concrete numbers, a readiness section with completed
  work, and an explicit statement of how the amount was determined. We have no local GPU, so our estimate
  is labelled a planning estimate and the proposal commits to replacing it with measurements.

---

## Journal changelog (append only)

- 2026-09-23: v1 created. Reconstructed history from the git log, `README.md`, every
  document and result file in the repo, and R-05.
- 2026-09-23: recorded the user's answers (no GPU, grant planned, agents.md pointer). Added
  D-15 and D-16.
- 2026-09-24: added Section 1.7 (plan while waiting for compute) and D-17. Updated the "Last
  updated" date and the Section 1 "As of" line.
- 2026-09-24: added Appendix B (compute and funding programs), E-23 (Mac pilot, with attempts 1
  and 2, amendment A1 and its development re-analysis), D-18, and the harness in
  `experiments/safedispatch_oracle/`.
- 2026-09-24: added TACC Horizon (Blackwell) to Appendix B as user-reported and unconfirmed, and the
  `grants/` folder of application drafts (D-19). The friend's proposal PDF was not read.
- 2026-09-24: read the labmate's NAIRR proposal (Appendix B.6), restructured the NAIRR draft to NAIRR's
  required sections A to D, and added `NAIRR_Proposal.pdf` to `.gitignore`.
- 2026-09-24: cleanup. Created `archive/` (D-20), moved the `ideas/` briefs and `RESEARCH_STATUS.md` there, moved
  the labmate's PDF to `grants/reference/`, filed the E-22 audit output, and updated current-location paths.
- 2026-09-30: added the agentic-harness literature review (Section 4 entry, Section 7.8, Section 1.6
  line, Appendix A row). Section 1 otherwise unchanged: no new research state or decisions.
- 2026-09-30: added the GPU-efficiency review (Section 4 entry, Section 7.9, Section 1.6 line,
  Appendix A row). Section 1 otherwise unchanged: no new research state or decisions.
- 2026-09-30: at the user's request, rewrote `dossier/LIT_REVIEW_GPU_EFFICIENCY_2026-09-30.md` from a raw doc export
  into a standalone review in the agentic-harness format (claims C1 to C16, sources S1 to S54 with check levels,
  excluded-claims section). Updated the Section 1.6, 7.9 and Appendix A pointers.
- 2026-09-30: R-06 correctness review of the agentic-harness review (Section 4 entry, Section 6.2,
  Section 7.8 two row corrections, Section 8.1 rows 25 to 29, D-21, Section 1.6 line). Section 1
  otherwise unchanged: no new research state.
- 2026-09-30: R-07 correctness review of the GPU-efficiency review (Section 4 entry with a pointer
  on the older entry, Section 6.2, Section 7.9 three row corrections, Section 8.1 rows 30 to 40,
  D-22, Section 1.6 lines, Appendix A row). Section 1 otherwise unchanged: no new research state.
- 2026-10-04: I-41 session. Section 3.7 (I-41 to I-45), Section 4 Phase 6, Section 5.12, E-24 and E-25
  (predeclared, then results), R-08, Section 7.10, Section 8.1 row 41, I-28 row note in 5.9, D-23,
  Section 9.1 and Appendix A rows, Section 1 rewritten.
- 2026-10-04: E-26 (predeclared, amended before running, then results), I-43 status and verdict
  history, Section 4 entries, D-24, D-25, Section 1.3, 1.4 and 1.6, Section 9.4 data location.
  Re-applied on 2026-10-05 after the first write was lost.
- 2026-10-05: D-26 (user dropped the I-41 line), I-41 and I-43 registry rows, I-43 log, Section 1.3,
  1.4 (item 8 removed) and 1.6, Section 4 entry.
- 2026-10-06: D-27 (write all content in ASD-STE100), rule 0.2.9 updated, rule added to `agents.md`.
- 2026-10-07: R-09 (written in ASD-STE100). Section 3.8, Section 4 Phase 7, Section 5.13, R-09 in
  Section 6.2, Section 7.11, D-28, Section 1.2, 1.4 and 1.6, Appendix A row.
- 2026-10-07: D-29 (idea style and experiment review process), Section 2.5 updated.
- 2026-10-07: R-10. Section 3.9, I-46 to I-49 status and verdicts, Section 5.14, Section 4 entries,
  R-10 in Section 6.2, Section 7.12, Section 8.1 rows 42 and 43, D-30 (D-28 superseded), Section 1.
- 2026-10-07: E-27 predeclared, amended (A1, A2) after R-11, run, decided (KILL) and reviewed (R-12).
  I-58 KILL. Section 8.1 row 44, D-30 accepted, D-31, Section 1, Sections 3.9, 4, 5.14, 6.1, 6.2.
- 2026-10-07: R-13 route search 2. Section 3.10, Section 5.15, R-13, D-31 decided, D-32, Section 1.
- 2026-10-07: Section 1 rewritten (as of 2026-10-07), handoff file `HANDOFF_2026-10-07.md` added.
- 2026-10-08: D-32 decided (option 1). R-14, E-28 (predeclared, amendments A1 and A2, result), R-15,
  R-16, Section 8.1 rows 45 and 46, I-80 status and verdicts, D-33, Section 4 entries, Appendix A rows,
  Section 1 rewritten.
- 2026-10-08 (later session): R-17 route search 3. Section 3.11, Section 5.16, R-17 in Section 6.2,
  Section 7.13, Section 4 entries, D-34, Appendix A rows, Section 1 rewritten.
- 2026-10-08 (later session): R-18 orphan-finding scan. Section 3.12, Section 5.17, R-18 in Section
  6.2, Section 7.14, D-35 (copied from memory notes), D-36, D-37, Section 4 entries, Appendix A rows,
  Section 1 rewritten.
- 2026-10-11: R-19 assumption removal. Section 3.13, Section 5.18, R-19 in Section 6.2, D-38, D-39,
  Section 4 entries, Appendix A rows, Section 1 rewritten.
- 2026-10-14: added the `references/` folder (447 arXiv papers in 16 topic folders, generated from
  Section 7 and the dossiers) and a pointer to it in Section 7. No idea or status changed.
- 2026-10-16: R-20 seed-topic gap search. Section 3.14, Section 5.19, E-30 draft in Section 6.1, R-20 in
  Section 6.2, Section 7.15, D-40, Section 4 entries, Appendix A row, Section 1 rewritten.
- 2026-10-16: checked every arXiv ID cited in the journal, dossiers and experiment READMEs (476). All 476 resolve to a
  real arXiv paper. 29 new R-20 IDs were read from arXiv abstract pages because the arXiv API returned
  429 and 503. A keyword check flagged 57 title and context pairs; each was read by hand and all match.
  Two are renamed or less obvious: 2512.19179 was "L4" in v1 (now CascadeInfer); 2605.27605 (Laguna
  report) has a cross-replica hash check for SDC. `references/` rebuilt with folder 17 for Section 7.15.
- 2026-10-17: R-21 current-frontier review. Added I-120 and I-121, Section 5.20, R-21, Section 7.16,
  D-41, and `dossier/FRONTIER_R-21_2026-10-17.md`. Rewrote Section 1. No experiment ran.
- 2026-10-17: R-22 continued R-21 until three total new gaps survived. Added I-122 to I-130,
  Section 5.21, R-22, Section 7.17, D-42, D-43, and `dossier/FRONTIER_R-22_2026-10-17.md`.
  I-122 and I-123 survive. I-124 to I-130 are KILL. Rewrote Section 1. No experiment ran.
- 2026-10-17: E-31 predeclared after D-44. Added the Docker state machine, tests, bounded search,
  synthetic scale sweep, and frozen decision script in `experiments/versioned_session_commit/`.
  Docker has an invalid signature. Sonnet is signed out. No experiment ran. Rewrote Section 1.
- 2026-10-17: Docker Desktop later served Linux containers through `desktop-linux`. Updated the
  current E-31 blocker to the required Sonnet review. No experiment ran.
- 2026-10-17: Added the E-31 read-only Sonnet prompt and manual review wrapper. Claude is authenticated
  in the user Terminal, but automation cannot read the Keychain entry. No experiment ran.
- 2026-10-17: R-23 found four blocking design faults in E-31. Marked E-31 INVALID DESIGN and did not run it.
- 2026-10-17: R-24 audited current vLLM at `da9134bc89364465f68aa679564435936d157a14`
  in Docker. Narrow fixes weaken I-122. Changed I-122 to PIVOT and added I-131.
- 2026-10-17: Predeclared E-32 for I-131. Added its frozen design and Sonnet review wrapper.
  E-32 is not implemented or run.
- 2026-10-19: R-25 and R-26 reviewed E-32 before implementation. Both reviews found unresolved
  circularity and oracle faults. E-32 did not run. E-33 was predeclared and then cancelled.
- 2026-10-19: R-27 mapped the four upstream tests and ran full-history blame in Docker.
  I-131 is KILL as a thesis. Added `dossier/PAPER_AUDIT_I-131_2026-10-19.md`.
- 2026-10-19: R-28 ran the fresh I-123 scoop audit. I-123 is PIVOT. Added I-132, the E-34
  design and review wrapper, Section 7.20, D-46, and `dossier/SCOOP_AUDIT_I-123_2026-10-19.md`.
  E-34 is not implemented or run. Rewrote Section 1.
- 2026-10-19: R-29 reviewed E-34 before implementation and returned `KILL_IDEA`.
  I-132 is KILL. E-34 is KILL-DESIGN and did not run. Added
  `dossier/REVIEW_E-34_PRERUN_2026-10-19.md`. Rewrote Section 1.
- 2026-10-16: checked every non-arXiv source: 55 web URLs (53 load; the 2 that return 403 were confirmed
  through Crossref and a search), 66 GitHub issues and PRs (all exist; titles read with `gh`), and the named
  sources in the Section 7 ledger rows without an arXiv ID (all exist). Section 8.1 rows 47 to 49 record one
  wrong issue citation, one wrong venue and one weak crowding claim.
- 2026-10-17: checked the R-21 and R-22 sources (written by a separate Codex session in this folder): 20 arXiv
  IDs (all real, titles match), 18 web links (all load and match), 12 GitHub issues and PRs (all exist,
  titles match), and the ledger rows without links. Section 8.1 rows 50 to 52 record one wrong title, one
  wrong year and three issue-date slips. `references/` rebuilt.
- 2026-10-19: checked the R-23 to R-29 content: every new E, R, D and I entry has its registry row; all
  cited paths exist except the archived `dossier/RESEARCH_STATUS.md`; 1 new arXiv ID (2605.00831), 16 new
  GitHub issues and PRs and 4 vLLM commit SHAs exist and match their claims; the new experiment folders
  hold no secrets or large files. Mapped Sections 7.18 to 7.20 in the references build and rebuilt
  `references/` (489 papers, 22 folders).
- 2026-10-19: R-30 scoop check (main session) before the Mac test. I-106 and I-60 KILL, I-59 WEAK and
  Mac-testable. Section 1, Sections 3 and 4 entries, I-59, I-60 and I-106 dossier lines, R-30, Section
  7.21, D-47. Nothing ran and nothing was downloaded.
- 2026-10-20: E-35 (I-59 premise check) predeclared, amended (A1 after R-31, A2 after run 1), run and
  reviewed (R-32), plus a paired supplement. I-59 KILL. Section 1, Sections 3 and 4 entries, I-59
  dossier, E-35, R-31, R-32, Section 8.1 rows 59 and 60, D-47 decided, D-48, D-49, Appendix A rows.
- 2026-10-20: R-33 fresh frontier pass. I-133 to I-140 registered and KILL. I-120 KILL as research
  (vLLM PR 53298). I-23 note (2606.07581). Section 1, Sections 3 and 4 entries, Section 5.24, I-120 and
  I-23 dossier lines, R-33, Section 7.22, D-50.
- 2026-10-20: R-35 (D-49 decided). Section 1, I-83 registry row, Section 4 entry, I-120, I-83, I-118 and
  I-23 dossier lines, E-30 frozen, E-36 predeclared, R-35, Section 7.23, D-49 decided, D-51, D-52,
  Appendix A rows. R-34 pending.
- 2026-10-20: R-36 limitations mining. I-141 to I-148 registered (seven KILL, one WEAK). Section 1 lines,
  Section 3.20, Section 4 entry, Section 5.25, R-36, Section 7.24, D-53, Appendix A row.
- 2026-10-20: R-34 recorded and applied. Amendments A1 to E-36 and E-30; E-36 Mac dry run on the vLLM
  v0.32.0 CPU image (five defects fixed). Section 1 lines, Section 4 entry, I-23 and I-118 dossier
  lines, E-30 and E-36 entries, R-34, Section 8.1 rows 61 to 63, Appendix A rows. GPU runbook updated.
- 2026-10-20: mapped journal Section 7.24 to `references/26-limitations-r36/` in the references build and
  rebuilt `references/` (497 papers, 26 folders; one new paper, 2609.24991).
- 2026-10-21: R-37 algorithmic-optimization sweep (D-54 decided) and R-38 Sonnet review. I-149 to I-164
  registered (all KILL after R-38). E-37 predeclared, run and reviewed. Section 1 lines, Section 3.21,
  Section 4 entries, Section 5.26, E-37, R-37, R-38, Section 7.25, Section 8.1 rows 64 to 66, D-54, D-55,
  Appendix A rows.
- 2026-10-22: D-56 (user): GPU experiments on Runpod, with a hardware record for each run. Section 1 lines,
  Section 4 entry, D-56, Appendix A row. New `experiments/runpod/`; runbook Section 2a.
- 2026-10-22: E-36 A2 and R-39. E-36 entry, R-39 entry, Section 1 lines, Section 4 entry, D-57, Appendix A rows.
- 2026-10-25: D-58 experiment standards, ledger and compare tool. Section 1 lines, D-58, Appendix A rows.
- 2026-10-25: D-59 shared Runpod account, thesis labels and Pod registry. Section 1 lines, D-59.
- 2026-10-25: E-36 L4 rehearsal, D-60, Section 1 lines, hardware log and ledger rows.
- 2026-10-26: pause, resume point, rehearsal 2 attempts, hardware log rows.
- 2026-10-26: rehearsal 2e PASS, R-41, runbook network volume, resume point rewritten.
- 2026-10-26: E-36 Session 1 result, R-42, D-61, Section 8.1 row 67, resume point rewritten.
- 2026-10-26: E-38 predeclared and amended (A1), R-43, resume point.
- 2026-10-27: E-38 H100 run (INCOMPLETE), partial A1 result, defect fix, lessons, D-62, resume point.
- 2026-10-27: E-38 rerun KILL; resume point. Review pending.
- 2026-10-27: E-38 rerun report, R-44, D-63.
- 2026-10-27: E-39 predeclared and amended, R-45.
- 2026-10-27: E-39 result INCONCLUSIVE; resume point.
- 2026-10-27: D-63 decided (I-23 closed for Hopper block-FP8).
- 2026-10-27: R-46 (E-39 post-run review).
- 2026-10-27: E-40 predeclared and amended, R-47.
