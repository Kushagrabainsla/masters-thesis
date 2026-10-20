# E-34 pre-implementation review (I-132)

**Verdict: `KILL_IDEA`.** Do not implement E-34.

**Scope and limits**
- I used read, grep and glob only.
- I read both required files in full, and the I-123, I-132, E-34, R-22 and R-28 entries in `RESEARCH_JOURNAL.md`.
- I did not open vLLM-Omni, Concordia or any paper. Every statement about them comes from the audit, so I mark those statements as unverified.
- My claims about classic prior art come from my own knowledge, not from a search. Check them before you cite them.

## Blockers

**B-1. P1 is a standard checkpoint composition.** (Q1, Q4)
- FACT: P1 is a write-ahead log, a periodic snapshot, an atomic manifest and an idempotent sink (`README.md:72-88`).
- FACT: The pipeline has one worker and one sequential operator. A consistent cut is therefore one log position, so no marker alignment is needed.
- FACT: `README.md:175` and `README.md:189-190` list "more than Flink plus a model snapshot" as unresolved.
- INTERPRETATION: The README mapping is Flink source, state and sink.
  - Replayable source = the input log.
  - Operator state = the opaque snapshot.
  - Idempotent sink = event IDs plus client deduplication.
- INTERPRETATION: The audit's own rule `KILL-NOVELTY` (`SCOOP_AUDIT:177`, journal `:2974`) already fires.
- Precedent: I-70 (Chandy-Lamport for async RL) was killed as crowded (journal `:730`).

**B-2. B1 and B2 are the end points of P1.** (Q5, Q6, Q9, Q10)
- FACT: K = 1 is in the P1 grid (`README.md:125`). P1 with K = 1 is B2. P1 with K very large is B1.
- INTERPRETATION: The experiment measures one tuning curve: snapshot cost divided by K, against replay cost times K.
- INTERPRETATION: This is the Young/Daly checkpoint-interval trade-off, and a formula gives the optimum. A Docker sweep would only encode the chosen constants. The audit already rejected this pattern for the old oracle (`SCOOP_AUDIT:33-34`).

**B-3. The decision rules cannot be computed.** (Q7, Q14)
- FACT: `README.md:170` says "meets the target". `README.md:171` and `:173` say "supported workload". No section defines either term.
- FACT: `README.md:148` calls wall time secondary. Rules 3 to 6 (`:170-173`) depend on wall time.
- FACT: The audit says Docker timing has no external validity (`SCOOP_AUDIT:131`).
- INTERPRETATION: The steady-state overhead percentage is snapshot time divided by the chosen synthetic unit-compute time. The experimenter picks both terms, so the result is fixed by the setup.
- INTERPRETATION: The two kill rules can fire from non-representative Docker storage speed.
- INTERPRETATION: The only reachable outcome is `PIVOT-GPU`. `GO-GPU` needs "both baselines fail", and Docker cannot show that.

**B-4. The workload does not test the one cross-layer element that could be new.** (Q4, Q12)
- FACT: A transition reads only the prior state and emits events (`README.md:43-48`). No transition reads a playback or acknowledgement frontier.
- FACT: The service is deterministic (`README.md:50`). `README.md:214-216` admits that real GPU behaviour is not covered.
- INTERPRETATION: The playback frontier is dead metadata here. It is a log-trim watermark only.
- INTERPRETATION: Deterministic replay removes the hard case. After a crash, a nondeterministic model cannot regenerate the outputs it already sent. The recovered state must then match the output the user already consumed.
- INTERPRETATION: Only that case (barge-in truncation, history that follows playback, non-reproducible output) could separate I-132 from Flink. E-34 never tests it.

## Major findings

**M-1. vLLM-Omni already exposes the frontiers.** (Q2)
- FACT: The audit says vLLM-Omni owns input, response, playback, history, lease and model state in one session, and keeps a reconnect journal (`SCOOP_AUDIT:77`).
- FACT: PR 8485 is open and adds per-input acknowledgements (`:78`). The audit says the absence of a durable engine-crash path "is not proof of novelty" (`:66`).
- INTERPRETATION: The missing piece is durability, and durability here is a write-through of existing ledgers. That looks like an integration patch.
- FACT: `README.md:85` lists "highest client-received event". The server only learns this through lagging acknowledgements.

**M-2. Concordia is the closest mechanism. Check it first.** (Q3)
- FACT: Concordia logs committed GPU-state deltas and restores in about 1.5 s (`SCOOP_AUDIT:74`).
- INTERPRETATION: The protocol frontiers are a few hundred bytes. Appending them to the Concordia commit record is a metadata extension.
- Unverified: I do not know whether Concordia has an external-metadata or output-commit hook. This is a one-day paper check, and it comes before any code.

**M-3. Classic prior art is missing from the audit.** (Q1, Q3; my own knowledge)
- The output-commit problem and rollback-recovery (Elnozahy et al. survey).
- Remus, which buffers output until the checkpoint commits.
- Optimistic message logging.
- Durable-execution engines (Temporal, Restate), which are B1 as a product.
- Kafka transactions.
- Young/Daly checkpoint intervals.
- I-123 searched "output-commit" terms (journal `:2740`), but the ledger (`SCOOP_AUDIT:68-81`, journal `:4806-4819`) lists none of this work.

**M-4. B1 and B2 are not competitive baselines for large state.** (Q5, Q6, Q9)
- B2 persists a full snapshot to durable storage every unit (`README.md:68`). That is a strawman at 224 MiB or at the 1.67 GB WorldMove state.
- The steelman baselines are missing:
  - delta or dirty-page snapshots (the Concordia approach);
  - asynchronous snapshot to host memory;
  - a hot replica (DejaVu or GhostServe style).
- The state range stops at 224 MiB (`README.md:123,131`). The case that caused the pivot (1.67 GB, 71% to 78% dirty) is outside it.
- The 224 MiB figure is sourced only inside the README. Neither the audit nor the journal ledger carries it.
- B1 needs a replay-speed ratio. Replay of buffered input may run faster than real time. The README has no such parameter, so `KILL-REPLAY` is true or false by construction.
- INTERPRETATION: For sessions of 1,000 units, B1 cannot recover in one unit unless replay runs much faster than real time, so rule 4 gives no information.

**M-5. P1 gets a fairer deal than B1 and B2.** (Q10)
- P1 has a tunable K and the others have no tunable. Choosing the best K per cell after seeing data is a post-hoc choice.
- `ack lag` (`README.md:126`) affects only P1 compaction.
- Fix: freeze K per workload before the run, or report the full curve and give B1 and B2 equal tuning (for example, replay batching).

**M-6. Recovery time includes a container restart.** (Q7)
- Procedure step 5 restarts the worker in a new container (`README.md:156`).
- Container start costs seconds on Docker Desktop. That is far longer than one application unit, for every policy.
- Define recovery as the time from worker start to the first correct output.
- The 200 ms target has no valid role. It was a project heuristic (`SCOOP_AUDIT:29`), and R-28 shows a 375.6 ms block cannot replay in 200 ms.
- A sourced stall budget (speech turn-taking) might apply to speech only. The "one application unit" replacement is also a heuristic.
- `SCOOP_AUDIT:152` mixes steady-state latency added with recovery latency.
- Journal `:2725` still states the 200 ms question.

**M-7. SIGKILL cannot falsify the durability claim.** (Q8)
- FACT: The supervisor uses SIGKILL (`README.md:103`). The kernel keeps page-cache data after a process kill, so fsync order, rename atomicity and torn writes are not exercised.
- INTERPRETATION: The negative control "manifest before state bytes" (`README.md:95`) can fail only if the kill lands inside a partial write. Add a deterministic control that fails by construction, or add a write-order crash checker (ALICE or CrashMonkey style).
- Docker can falsify the logical protocol under process crash and an intact OS.
- Docker cannot falsify the systems claim: GPU snapshot cost, determinism, or any production recovery objective.
- Docker Desktop fsync semantics on the M4 are unclear, so durable-bytes timing is not interpretable.

**M-8. The crash schedule has gaps.** (Q12)
- FACT: Eight markers (`README.md:109-116`). The audit promised a crash "at every durable-operation boundary" (`SCOOP_AUDIT:160` region). The README does not deliver that.
- Missing cuts:
  - Between events of one multi-event unit: after event i of n is durable, sent, or acknowledged. Marker 6 covers only one event.
  - A second crash during recovery, during replay, and during manifest publication or log compaction.
  - The client sends an input and the worker dies before it is durable. The README has no client retry or input deduplication path, and Flink assumes a replayable source.
  - Acknowledgement frontier ahead of or behind the restored epoch.
- Markers 4 and 5 do not exist for B1, which contradicts "same markers for every policy" (`README.md:105-107`). Define abstract markers and map them to no-ops for B1.
- Fix: enumerate every durable-operation boundary exhaustively for traces of 10 units or fewer, and use random cuts for longer traces.

**M-9. Stable identifiers can hide faults.** (Q13)
- FACT: The oracle counts "lost" and "duplicate" events after deduplication (`README.md:136-146`). ID assignment is unspecified.
- Deduplication by ID drops a replayed event with a changed payload. The first copy wins and the fault is invisible.
- Hash IDs hide loss. Reordering is possible if IDs are not dense and monotonic.
- Fix:
  - Use dense per-session sequence IDs.
  - Log every received copy with a payload hash.
  - Fail on a payload mismatch between copies of the same ID.
  - Check order and gaps.
  - Deduplicate before playback, because "visible" means played.
- Add three negative controls: same ID with a changed payload, a dropped event with dense numbering, and a swapped pair.
- `README.md:20` claims at-least-once delivery, but the harness (`:30`) disables the network. It injects no duplication, loss or reorder.

**M-10. The oracle may not be independent.** (Q11)
- FACT: The reference model is "pure" and separate from recovery metadata (`README.md:52-54`).
- The reference may reuse the service transition code, event-ID rule or digest function. Then it shares their bugs.
- It compares only the final digest and the event sequence. Compare per-unit state digests.
- Require a separately written reference with checked-in test vectors.

## Minor findings
- Rule 1 covers a failing negative control only. Add `INVALID` if B1 or B2 fails any cut, because that means the harness is wrong (`README.md:168`).
- Five repeats and a 3% threshold with no confidence interval or equivalence margin (`README.md:159,170`).
- "Write-only mount" does not exist in Docker (`README.md:33`).
- B2 at 224 MiB for 1,000 units writes up to 224 GiB per run. Check the Docker disk and run-time cost.
- The freeze point for "after the protocol is frozen" (`SCOOP_AUDIT:179`) is not defined in the README.
- Passing a finite covering array is not a proof. Say so (`README.md:162-175`).

## Answers by question

| Q | Answer | Severity |
|---|---|---|
| 1 | Yes, as specified. | BLOCKER |
| 2 | The frontiers exist in memory. Only durability is missing. | MAJOR |
| 3 | Probably close. Unverified. Check Concordia's commit-record hook. | MAJOR |
| 4 | Integration patch. A thesis needs a nondeterministic-replay or playback-dependent invariant that E-34 does not test. | BLOCKER |
| 5 | B2 is a strawman and no steelman is tested. For long sessions at 224 MiB or more, a per-unit full copy may meet the target. | MAJOR |
| 6 | The result is set by chosen constants and the replay-speed ratio. | MAJOR |
| 7 | No valid role as a derived threshold. At most a sourced speech stall budget. | MAJOR |
| 8 | It can falsify the logical protocol only. It cannot falsify the systems claim. | MAJOR |
| 9 | Both are correct but not competitive, and they are P1's end points. | BLOCKER/MAJOR |
| 10 | P1 has a tunable K that the baselines lack. | MAJOR |
| 11 | Not shown to be independent. | MAJOR |
| 12 | Yes. See M-8. | MAJOR |
| 13 | Yes. See M-9. | MAJOR |
| 14 | Yes, through the timing-based rules and the undefined target. | BLOCKER |
| 15 | See below. | |

## Cheapest experiment that can kill I-132 (no Docker)
Do these in order. Each takes about a day or less.
1. **Cost model.** Write the Young/Daly model for B1, B2 and P1 with sourced numbers (Capsules 224 MiB, WorldMove 1.67 GB, 375.6 ms, 0.93 GB/s, Concordia 1.5 s, PCIe and NVMe bandwidth). If B2 to host memory fits the budget, `KILL-SIMPLE` is decided on paper.
2. **Model check.** Model-check the protocol (TLA+ or an exhaustive Python state search) over all crash points, including nested crashes. It is stronger than 8 markers. Expect B1, B2 and P1 to pass, which confirms there is no new invariant.
3. **Patch-size check.** Read vLLM-Omni's session object and Concordia's commit API. If a durable write-through of the ledgers plus a commit-record field covers the gap, the finding is an integration patch.
4. **The one test that could keep the idea alive** (needs a GPU for about an hour, so not a kill test): re-run one unit twice from the same snapshot and compare outputs bitwise. If they are identical, replay works as in Flink and the idea dies. If not, an already-sent output cannot be regenerated, and that is a real constraint worth a new design.

## Verdict
`KILL_IDEA`. Do not implement E-34.
- As designed, P1 is a standard checkpoint composition, so the audit's own `KILL-NOVELTY` rule fires.
- Every outcome Docker can produce is either expected (all policies are correct) or set by chosen constants.
- This is an AI recommendation. Journal D-46 stays open until you record the decision.

**Reopen only if** step 4 shows nondeterministic replay, or a measured speech or world-model trace shows that no K, no delta snapshot and no hot replica meets a sourced budget.

**If you overrule the kill, the smallest amendment set is:**
1. Define the target and the supported workload from sourced numbers. Remove the 200 ms target and any threshold that depends on synthetic unit cost.
2. Make Docker kill on correctness only. Move the timing rules to the cost model.
3. Add a playback-dependent transition and a nondeterministic-output mode to the workload.
4. Add steelman baselines: delta snapshot, asynchronous snapshot and a hot replica.
5. Replace the 8 markers with exhaustive durable-operation cuts, nested-crash cuts and per-event cuts. Add client input retry.
6. Use dense IDs, log all received copies with payload hashes, and add the three new negative controls. Inject network duplication, loss and reorder.
7. Use an independently written reference with per-unit digests, and use a write-order checker for the durability claim.
8. Add the prior-art rows from M-3 to the audit before any run.
