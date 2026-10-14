# R-19 hostile review of P4, P3 and P2 (2026-10-11)

Reviewer role: hostile systems PC member (OSDI, NSDI, MLSys). Goal: kill weak ideas with evidence.
Labels: FACT = I opened the primary source. SNIPPET = search result only. INTERPRETATION = my own inference.
Constraints: the student wants a PagedAttention-style mechanism, not a benchmarking paper. Hardware: Apple M4 16 GB, and about $20 for one H100 in round one.

## 0. Verdicts

| ID | Verdict | One-sentence reason |
|----|---------|---------------------|
| P4 | WEAK. Survives only as a narrowed question (Section 1.7). | MVCC for weights is a bag of known parts, and the pinning half has no production need. The cross-checkpoint KV divergence question is open but near two scoops. |
| P3 | KILL | The 17% setup share is mostly agent thinking time, exact duplicates are 0 to 4%, and time-based LR schedules make cross-run early stopping unsound. |
| P2 | KILL | It is a benchmark study. Emerald already compares pause, DVFS and reallocation. Pause and DVFS do not change convergence, so only replica dropping is left, and that is a known batch-size problem. |

Best survivor: P4, narrowed. Its chance to pass the Mac gate is about 40% (INTERPRETATION). It is the only candidate with a mechanism shape (version-tagged blocks plus a layer-wise recompute rule).

Corrections to the brief:
- FACT: vLLM docs say `clear_cache` is deprecated and ignored when `mode="keep"`. With `keep`, in-flight requests continue with old-weight KV and do not re-prefill. The statement "weight reload with clear_cache forces re-prefill" holds only for the wait and flush path.
- FACT: Cursor's own post says Cursor Tab ships a checkpoint every 1.5 to 2 hours. The 5 hour cadence belongs to Composer, which Cursor ships "behind Auto". The Tab post does not mention agents or session length.

---

## 1. P4: model version fixed for a session

### 1.1 Is the assumption breaking for long agent sessions? (attack a)

Evidence for frequent updates:
- FACT: Cursor blog "Improving Composer through real-time RL" says Cursor can ship an improved Composer "behind Auto as often as every five hours". The full loop takes about five hours. Source: https://cursor.com/blog/real-time-rl-for-composer (listed 2026-03-26).
- FACT: The Cursor Tab post gives 1.5 to 2 hours per cycle and says nothing about agents. Source: https://cursor.com/blog/tab-rl.
- FACT: The Composer post does not mention in-flight sessions, KV cache or session length. It says longer background tasks are expected "in the future".
- SNIPPET: Cursor long-running agents exist (one customer quote of about 52 hours). This is an anecdote, not a distribution.
- SNIPPET: I found no cadence data for Cognition SWE-1.5 or SWE-1.6, Claude Code or Codex. Their posts describe training runs, not weight deployment schedules.
- SNIPPET: I found no primary data on how often weights change under a fixed model id. Only blogs and the 2023 Stanford and Berkeley drift study exist.

Result:
- INTERPRETATION: One product (Cursor Composer) has a documented multi-hour cadence. That is one data point, not a trend.
- INTERPRETATION: Most agent APIs resend the whole history on each call. A new version simply reads the old history. The only damage is one prefix-cache miss.
- INTERPRETATION: Providers want the new weights as soon as possible, because the new checkpoint is better by design. Pinning old versions works against the provider's goal.
- FACT: The one place where pinning matters is on-policy RL data. Cursor says it wants data "fully or almost-fully on-policy". Fireworks docs say a turn that must come from one policy version should not cross an async hot-load boundary. Source: https://docs.fireworks.ai/guides/rollout-inference. That is an RL rollout need, not a serving need.

Attack verdict (a): the assumption breaks for the data pipeline of RL products. It does not clearly break for the customer. For short Tab requests it does not break at all, because a request lasts under a second.

### 1.2 Is it a bag of known tricks? (attack b)

| Job in the proposal | Known mechanism | Source and label |
|---|---|---|
| Pin a session to the version it started on | Blue-green rollout with HTTPRoute weights, plus sticky routing by session id | llm-d rollouts docs, NVIDIA Switchyard sticky routing (SNIPPET) |
| New sessions get the new version | Same load-balancer weight shift | Same (SNIPPET) |
| Garbage-collect old versions when sessions end | Connection draining, with a long timeout | SNIPPET |
| Keep old versions cheaply | Delta compression: DeltaZip, EuroSys 2025, arXiv 2312.05215, up to 10x compression and 2x to 12x throughput over baselines | SNIPPET (abstract level) |
| Keep old versions cheaply, RL-style | Sparse delta sync: over 99% sparsity, 14 GB to about 108 MB (SparseRL-Sync, arXiv 2605.07330) | SNIPPET |
| Do not re-prefill in-flight work at an update | vLLM `pause(mode="keep")`; Fireworks hot-load resumes "with its existing active KV/state" | FACT (Fireworks), FACT-by-search (vLLM) |
| Do not mix versions in the prefix cache | PSRL `multi_version_kv`: every KV entry is tagged with its version, lookups at a newer version miss old entries | FACT (PSRL docs) |
| Do not mix versions inside a new rollout group | prime-rl: salt the cache key with the dispatch version, keep in-flight work on its existing cache | FACT (Prime Intellect glossary) |
| Continue with stale KV and correct the training signal | PipelineRL (arXiv 2509.19128), Motif 3 (arXiv 2608.09119, importance weights clipped to [0.2, 5.0]) | SNIPPET |
| Reuse KV between model variants of one architecture | DroidSpeak, NSDI 2026, arXiv 2411.02820: recompute a few layers, up to 4x throughput and about 3.1x faster prefill | FACT (abstract) |
| Benchmark KV reuse across checkpoints | KVShareArena, arXiv 2609.10266, submitted 2026-09-09, revised 2026-09-27 | FACT (abstract) |

KVShareArena facts (abstract opened):
- Five LLMs (4B to 30B), six pairs in which one version reads the cache another version wrote, 11 repair methods, 2,150 questions, 33 settings.
- Most training-free repairs keep their quality when another version wrote the cache.
- A trained repair adapter loses quality in 12 of 18 pair and dataset tests.
- It is a benchmark and not a system. It does not use adjacent RL checkpoints at a fixed small gap (I could not read the body of the PDF, so this last point is SNIPPET level).

Which mechanism does which job:
- Load balancer: pinning and draining.
- DeltaZip or sparse deltas: storage of old versions. They do not remove the need to hold a servable copy.
- `keep` mode: avoids re-prefill.
- Version-tagged KV: correctness.
- DroidSpeak-style layer recompute: cheaper reuse for new sessions.

Does the coupling create a new problem?
- INTERPRETATION: A possible new problem is a joint policy per live session: keep stale, recompute some layers, re-prefill, or stay pinned. The policy depends on a divergence signal and on a cost budget. This is an admission and scheduling problem over three known mechanisms.
- INTERPRETATION: It is not a paper by itself. A PC member will say "this is draining plus DeltaZip plus PipelineRL". The paper needs a new measured fact (how KV diverges across adjacent RL checkpoints) and a new rule that uses it (Section 1.7).

### 1.3 Oracle (attack c)

Inputs. TraceLab and the NVIDIA AIPerf page give 665,453 rounds in 8,058 sessions and 52.56 B cached plus 2.34 B new input tokens (FACT from the earlier R-19 file, TraceLab blog). Derived by me (INTERPRETATION):
- Mean live context per round: 52.56 B / 665,453 = about 79k tokens.
- Mean new input per round: 2.34 B / 665,453 = about 3.5k tokens.
- Rounds per session: 665,453 / 8,058 = about 83.
- A SyFI Trace Atlas snapshot shows 9.2 requests per session on average and a p99 of 137 (SNIPPET). Human idle gaps make wall-clock session length much longer than 83 rounds times 31 s. I did not find the session length distribution.

Cost of a flush (re-prefill all live sessions):
- Per live session per update: one re-prefill of about L = 79k tokens.
- GPU time per session: L / prefill rate. With an assumed 8B dense model at 20k tokens per second on one H100 (assumption to measure), that is about 4 s. Attention cost makes long prefill slower, so treat 4 s as a floor.
- Fleet cost per update: N_live x L tokens. Example: 1,000 live sessions x 79k = 79 M tokens = about 66 GPU-minutes (same assumed rate).
- Share of total prefill work for a developer who works 8 h per day, with updates every 5 h: about 1.6 updates in the active window. Extra prefill = 1.6 x 79k = 127k tokens. Normal prefill = about 900 rounds x 3.5k = 3.2 M tokens (assumes 31 s per round, 8 h). Ratio: about 4%.
- With `keep` mode the extra prefill is 0. The cost then moves to stale-KV quality, which is the question in attack (d).
- Any rolling restart of a fleet (new binary, new kernels) already flushes the prefix cache. The marginal cost of a weight update over a normal deploy is therefore smaller than 4%.
- Provider caches also expire after minutes to one hour of idle time (INTERPRETATION from API docs I did not reopen). Idle sessions were going to re-prefill anyway.

Cost of a resident old version:
- Dense: 16 GB for an 8B model in bf16 (arithmetic). For a trillion-parameter MoE at 8 bit, about 1 TB, which is a whole extra replica group. TP and EP sharding means a second version needs its own GPUs, not a slice of spare HBM.
- Sparse delta: storage is about 0.8% (108 MB of 14 GB, SNIPPET). Execution still needs a materialized copy or a delta kernel on the hot path. I found no measurement of the delta kernel cost for two versions of one model.
- Pin duration: the old pool stays alive until its last session ends. A heavy tail (a 52 hour agent, SNIPPET) keeps a replica group alive for days at low utilisation.

Oracle inequality (INTERPRETATION):
- Pinning wins only if (old-pool GPU-hours until the last pinned session ends) < (GPU-hours of the re-prefill storm it avoids).
- With a 4% prefill penalty, prefill about 30 to 50% of fleet work, the storm costs about 1.2 to 2% of fleet GPU time. One old pool of 5% of the fleet kept for 2 of every 5 hours costs 2%. The two sides are the same order. Pinning has no clear win. The sensitive numbers are session length tail and the prefill share, and the student has neither.

### 1.4 Does stale KV change outputs enough in production? (attack d)

- FACT: PSRL docs say serving later turns from stale-weight KV gives "accuracy degradation" that compounds with staleness. No number is given.
- FACT: Prime Intellect says keeping the cache gives "a hybrid that no single checkpoint reproduces" and extra trainer-inference mismatch on tokens just after an update.
- FACT: vLLM Ascend and API docs warn that a kept request mixes old-weight KV and new-weight decoding.
- SNIPPET: PipelineRL reports a variant with KV recompute and concludes recompute is feasible but lowers throughput. Motif 3 keeps the cache and corrects with clipped importance weights.
- SNIPPET: Fireworks keeps active KV across a swap and warns about single-version turns.
- SNIPPET: An FP16 study (arXiv 2604.15409) says cached and uncached inference give different token sequences in 100% of tested cases, because of rounding. Any divergence metric needs a noise floor of this kind.
- INTERPRETATION: For production, token-level drift between a cached and a fresh run is not visible by itself. The risk is in rare tool-call or JSON format errors, and in silent changes in agent behaviour inside one session. No source measures that.
- INTERPRETATION: RL deltas are tiny and sparse (SNIPPET: over 99% of weights unchanged per step). The KV change may be very small for adjacent checkpoints. This is exactly the open measurement.

Open measurement: nobody has published per-layer K and V divergence and next-token KL for stale KV across adjacent RL-trained checkpoints at a controlled step gap (SNIPPET level search, not a proof).

### 1.5 Closest prior art, rejection, baseline

| Item | Value |
|---|---|
| Closest prior art 1 | DroidSpeak, arXiv 2411.02820, NSDI 2026: cross-variant KV reuse with layer recompute (FACT, abstract) |
| Closest prior art 2 | KVShareArena, arXiv 2609.10266, 2026-09-09: benchmark of reuse across checkpoints (FACT, abstract) |
| Closest prior art 3 | PSRL multi_version_kv and prime-rl version salt (FACT, docs) |
| One-sentence rejection | "This is load-balancer draining plus DeltaZip plus version-tagged KV, and cross-checkpoint KV reuse is DroidSpeak with a smaller gap." |
| Simplest baseline | Rolling update with `keep` mode, route new sessions to new replicas by load-balancer weight, let old replicas drain, accept one prefix miss per session. |
| Oracle | Section 1.3: avoided re-prefill GPU-hours against old-pool GPU-hours. |

### 1.6 Verdict

WEAK. The MVCC framing dies. The part that survives is the divergence question.

### 1.7 Narrowed question (best survivor)

"After an RL or SFT update, how far does the KV cache of an existing context drift per layer, and can the weight delta predict which layers need recomputation, so a serving engine keeps the cache with a bounded quality loss?"

Mechanism shape (PagedAttention-style, if the premise holds):
- Each KV block carries a per-layer version stamp (not one stamp per block).
- On a weight update, the engine computes a per-layer drift score from the weight delta, without running the model.
- A block keeps layers below the threshold and recomputes the layers above it, lazily, when a session touches the block.
- A bound on cumulative drift forces a full re-prefill.

Why it can still be rejected: DroidSpeak already picks layers. The student must show that a weight-delta predictor (no model run) matches measured drift, and that the gain holds for adjacent checkpoints. If the drift is below the noise floor, there is no mechanism to build.

---

## 2. P3: cluster user is an agent

### 2.1 Evidence check (FACT)

Source: https://skypilot.ai/blog/scaling-autoresearch/ (opened).
- About 910 experiments in about 8 hours on 16 GPUs (13 H100, 3 H200). About 700 valid, the rest queued or crashed. Rate about 90 per hour. The single-GPU rate is written as both about 10 and about 12 per hour.
- Cycle time: "~5 min plus ~1 min of setup and agent thinking time". The page does not state 17%. That figure is my arithmetic (1 of 6 minutes).
- "The setup block runs once per cluster."
- The 1 minute includes agent thinking. It is not container or process startup. The page gives no split.
- Plateau: gains below 0.0001 per experiment after about 700 experiments. The page names no duplicates. It says 210 of 910 did not give valid results.
- No results.tsv is linked from the blog.

FACT: Karpathy's `train.py` ties all schedules to `progress = total_training_time / TIME_BUDGET`, with `WARMDOWN_RATIO = 0.5` and `torch.compile` on the model. Compile time is outside the time budget (the loop counts training time).

### 2.2 Attack results

Is it ASHA plus warm pools? Yes.
- Hyperband and ASHA (classic), Themis (arXiv 1907.01484, cluster-level fairness plus a hyper-parameter API) (SNIPPET).
- ALPS (ICML 2026) schedules parallel LLM-driven search with a lineage selector, a bandit, adaptive parallelism and a noise-aware promotion gate (FACT, abstract page). It treats the LLM as a proposer and the scheduler as the stateful part. This is the same idea as "agent as tenant".
- arXiv 2603.15916 (2026-03-16, 10,469 experiments over 27 days) has an orchestrator that deduplicates proposals and schedules a GPU cluster (FACT, abstract).
- Warm pools and process snapshots: MQFQ-Sticky (arXiv 2507.08954), NVIDIA Dynamo snapshot (SNIPPET).
- Forking: Trainy Pluto and ART checkpoint forking (SNIPPET).

Is the 17% share real and worth a mechanism?
- INTERPRETATION: It is real as arithmetic but it is not a scheduler cost. It contains agent LLM latency. A scheduler cannot remove it. A faster agent or speculative proposal queue removes it, and SkyPilot's `sky exec` queue already hides it by keeping the GPU busy.
- INTERPRETATION: Compile and import time are outside the 300 s budget but inside wall time. A warm pool helps with that. The gain is bounded by that share, which no source measures.

What do public logs allow on a Mac? I ran the check (FACT: ran it, data is from the files).
- `gh search code --filename results.tsv` found many public autoresearch logs. I downloaded 12 SakanaAI/fugu demo logs (4 agents x 3 seeds, 123 to 170 rows each), plus a few small ones. I did not use all of them.
- Columns: commit, val_bpb, memory_gb, status, description. There is no timing and no config hash. The 17% setup share cannot be measured from these files.
- Exact duplicate description strings: 0 to 5 per run of 123 to 170 rows (0% to 4%). Exact duplicate val_bpb values: 0 to 10 per run (0% to 6%).
- Fuzzy text match above 0.9 gave 33% to 70% "near duplicates". I checked samples. They are sweeps of one knob ("raise scalar learning rate to 2.25, 2.4, 2.45"). They are not repeats. The fuzzy score is useless as a duplicate measure.
- Share of runs after the best-so-far reaches 90% of the final gain: 2% to 62% (median about 40%). The last 0.001 of gain arrives late.
- Share of runs more than 0.003 val_bpb worse than the running best: 6% to 32% (median about 19%).
- Data quality warning: the three gemini31_pro seeds have identical row counts (123) and keep and discard counts (29 and 94). The opus48_max and fugu_ultra pairs have equal row counts (146, 128, 170) but different file hashes. These may be copies or replays. Treat the per-seed numbers as one sample per agent.

Oracle (INTERPRETATION):
- Perfect memoization saves at most about 0% to 4% of runs.
- Perfect early stop of every run that ends clearly bad, at 20% of its budget, saves about 19% x 0.8 = about 15% of GPU time. The oracle assumes the bad run is known at 20%. Because the LR schedule depends on progress (warmdown starts at 50% of the budget), the loss at 20% of the budget is measured at peak LR and does not rank final loss. The oracle is optimistic.
- Fork-from-parent: most runs change one knob of the best config, so the shared prefix is the first part of a schedule that depends on the total budget. A fork changes the schedule, so the prefix is not reusable (same reason).

### 2.3 Verdict: KILL

- One-sentence rejection: "This is ASHA and warm pools in a scheduler, ALPS and 2603.15916 already treat the agent as the search driver, and the measured waste is below 5% for duplicates and mostly not removable for setup."
- Simplest baseline: SkyPilot `sky exec` queue with a persistent Python worker and `torch.compile` cache on disk.
- Residual note: the right question may be a loss-curve predictor for time-based schedules. That is a modelling paper, not a mechanism.

---

## 3. P2: datacenter cannot draw full power at any time

### 3.1 Evidence (FACT unless noted)

- Emerald Conductor, arXiv 2507.00909 (Jul 2025): I extracted the PDF text. The paper compares knob combinations: DVFS, job pausing, resource reallocation, with Greedy and Fair policies. Its text says "DVFS + Job Pausing, Fair" gave the best result in one metric, "DVFS, Fair" gave the best average job throughput, and "Job Pausing, Greedy" affected the fewest jobs. It says both pausing and reallocation need checkpointing for training jobs, and it treats checkpoint overhead as negligible for its experiments. The text has no convergence analysis (no match for "converge").
- Characterizing Job Power Elasticity, arXiv 2609.11542 (2026-09-10): 131 H200 runs, PFI metric, tokens per second only (FACT, abstract page). No convergence.
- PowerSlider (arXiv 2608.21719) and PALS (arXiv 2605.21427) cover inference (SNIPPET).
- ElasWave (arXiv 2510.00606) and FlexTrain (MLSys 2026) keep the global batch size or accuracy consistent when the replica count changes (SNIPPET).
- A February 2026 curtailment-window training report (561M model) says training quality was preserved (SNIPPET).

### 3.2 Attack

Is this a benchmarking study? Yes. The brief's gap is "nobody compares three knobs on one trace with convergence". Emerald compares the knobs. Convergence is the only missing piece.

What would the mechanism be?
- INTERPRETATION: A power cap changes clock speed. It does not change the reduction order of deterministic kernels, so the loss curve should be the same except for time. I did not find a source that tests this (SNIPPET: search found none). Atomics or autotuning could differ.
- INTERPRETATION: Pause and resume restore the same state. Convergence per step is unchanged. Only wall time and energy change.
- INTERPRETATION: Only replica dropping changes the global batch size and so convergence. That is a known problem with known fixes: keep the global batch with micro-batch reshaping (ElasWave), or rescale the LR with a ramp (Horovod and Ray, AdaBatch).
- So the "convergence on one grid" result mostly says "equal per step" for two of three knobs. The third is the elastic-training literature.

Can anything decisive run on one H100?
- Power caps via `nvidia-smi -pl`: yes (the cap is allowed on a cloud instance only if the provider permits it; I did not verify AWS allows it, SNIPPET-level doubt).
- Replica dropping: no, one GPU has no replicas. Gradient accumulation can emulate a smaller global batch.
- A grid trace: public (CAISO and the 2609.05406 trace statistics).
- A $20 run measures tokens per second per watt at 4 caps. That is a characterization, which is a benchmark, and 2609.11542 already did it with 131 runs.

### 3.3 Closest prior art, rejection, baseline, oracle, verdict

| Item | Value |
|---|---|
| Closest prior art | Emerald Conductor, arXiv 2507.00909 (knob comparison, field test, FACT); Job Power Elasticity, arXiv 2609.11542 (FACT) |
| One-sentence rejection | "Emerald compares pause, DVFS and reallocation already, and the convergence part reduces to batch-size rescaling that ElasWave and FlexTrain solve." |
| Simplest baseline | Cap power with DVFS during the event and keep the global batch. |
| Oracle | Same energy, same deadline, no loss in tokens per step: time-to-target loss of the uncapped run, stretched by the energy ratio. |
| Verdict | KILL |

---

## 4. Best survivor: P4 narrowed. Premise check plan

### 4.1 Mac check, public data only (zero cost)

Data. FACT (HF API, checked 2026-10-11): `allenai/OLMo-2-0425-1B-RLVR1` has 13 step revisions: step_200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600. The model has 1,484,916,736 BF16 parameters (about 3 GB). It fits in 16 GB. The checkpoints are RLVR (RL) steps of one run, which matches the production case (RL deltas) better than pretraining checkpoints.
Not checked: whether the revisions differ in more than a few parameters per step. Measure it first (step 0 below).
Context text: any public agent trajectory set with an open licence (check licence first), plus code files and chat text as controls. Context lengths 512, 2k, 8k. Longer than 8k only if memory allows.

Steps:
0. Delta statistics. For each adjacent pair, compute the fraction of changed bf16 elements and the relative L2 delta per layer. Kill gate: if less than 0.1% of elements change in all pairs, the pair differs by rounding only. Then treat the pair as a noise-floor control.
1. Noise floor. Run the same model and context twice with different chunking (one-shot prefill against chunks of 512). Record per-layer relative K and V error and next-token KL. This is the floor from the FP16 divergence effect (arXiv 2604.15409).
2. Drift. For each adjacent pair (A, B) and for gaps of 2, 4 and 13 checkpoints: compute KV with A, compute KV with B. Record per layer relative L2 of K and V, and the cosine.
3. Quality. Decode with B for 64 tokens in four modes: fresh B KV; stale A KV for all layers; stale A KV except the top-k layers by measured drift (k = 25% and 50%); stale A KV except the top-k layers by weight-delta score (no model run). Metrics: mean next-token KL to the fresh mode, greedy token agreement at horizons 16, 64 and 256, and format validity on a JSON tool-call task.
4. Predictor. Spearman correlation between the per-layer weight-delta score and the measured per-layer KV drift.

Predeclared kill thresholds (all are heuristics, set before the run):
- K1 (too easy, no mechanism): stale A KV for all layers has median KL below 2 times the noise floor at every gap up to 13. Then "keep the cache" is enough. Result: a one-paragraph finding, not a thesis. KILL P4.
- K2 (too hard, no saving): at the smallest gap (200 steps), stale reuse has KL above 10 times the floor, and recomputing 50% of the layers (oracle-chosen by measured drift) still leaves KL above 3 times the floor. Then reuse needs near-full recompute and DroidSpeak gives no saving. KILL P4.
- K3 (no cheap check): Spearman between weight-delta score and measured drift is below 0.7 across layers. Then the divergence check needs a model run, and the cost argument fails. KILL the mechanism. A pure measurement paper remains, which is not allowed.
- K4 (not production-relevant): greedy agreement at 64 tokens stays above 98% for stale-all-layers, and the JSON validity gap is under 1 point. Then output change is invisible and the engine should just keep the cache (same as K1 outcome).
- Survive: KL between 3 and 30 times the floor for stale-all, a 25% layer recompute chosen by weight delta brings KL to at most 3 times the floor, and the Spearman is at least 0.7.

Grid edges to check before accepting "no effect" (lessons from earlier experiments: trigger size, horizon, metric power):
- Trigger size: the 200-step gap may be larger or smaller than a production update. Add weight interpolation A + alpha (B - A), alpha in {0.05, 0.1, 0.25, 0.5, 1.0}, to emulate smaller steps. Also test the largest gap (step_200 against step_2600) to confirm the metric can see a real change. If the largest gap shows no effect, the metric is blind.
- Context length: 512 against 8k. Drift can grow with position. Report KL by token position bucket.
- Horizon: 16, 64, 256 tokens. KL at token 1 can hide error that grows later.
- Metric power: at least 200 contexts per cell. Report bootstrap 95% intervals. Use a positive control: a deliberately corrupted cache (noise added to K at 1% relative norm) must be detected by all metrics.
- Precision: repeat one cell in fp32 to separate rounding noise from real drift.
- Model size: 1B only. Note that results need a 7B check on the H100 (the 1B result can fail at scale).
- Layer choice: report which layers drift (early layers carry most error propagation, INTERPRETATION).

### 4.2 H100 check, only if the Mac check survives (budget)

Price: SNIPPET from third-party trackers: p5.4xlarge (one H100) is $6.88 per hour on demand in us-east-1. Check the AWS page before booking. $20 buys about 2.9 hours. Plan for 2.5 hours and stop at $17.

Plan:
1. (30 min) Repeat the Mac measurement on a 7B or 8B RL checkpoint series with public revisions (OLMo 3 7B RL-Zero repos list step branches such as step_200 to step_1900; check the step set first). Same kill thresholds K1 to K4.
2. (60 min) Measure the real flush cost with vLLM: 64 sessions with 32k context. Compare `keep`, wait plus flush, and a version-salted prefix cache. Record TTFT p50 and p99, re-prefill seconds per session, and memory per resident version. This replaces my 20k tokens per second assumption in Section 1.3.
3. (30 min) Test the mechanism prototype on one cell (layer-wise recompute) only if K1 to K4 pass.

H100 kill: if the measured flush cost for 64 x 32k sessions is under 5 s of added TTFT p99 at 8B, the storm is too small to need a mechanism.

### 4.3 What stays open

- No data on the session-length distribution of agent products. Without it the pinning part cannot be priced.
- No data on how far apart production updates are in weight space.
- I could not read the body of KVShareArena. Read it before writing anything. It may already contain the adjacent-checkpoint result.
- Two preprints (2609.10266 and 2609.11542) are v1 or v2 and unreviewed.

## 5. Sources opened

- FACT: https://cursor.com/blog/real-time-rl-for-composer
- FACT: https://cursor.com/blog/tab-rl
- FACT: https://skypilot.ai/blog/scaling-autoresearch/
- FACT: https://arxiv.org/abs/2609.10266 (KVShareArena)
- FACT: https://arxiv.org/abs/2411.02820 (DroidSpeak)
- FACT: https://arxiv.org/abs/2609.11542
- FACT: https://arxiv.org/abs/2603.15916
- FACT: https://icml.cc/virtual/2026/67431 (ALPS)
- FACT: https://psrl.readthedocs.io/en/latest/design/kv_cache.html
- FACT: https://primeintellect.ai/rl-glossary/kv-cache
- FACT: https://docs.fireworks.ai/guides/rollout-inference
- FACT: https://arxiv.org/pdf/2507.00909 (PDF text extracted locally)
- FACT: karpathy/autoresearch `train.py` (via GitHub API), HF API for OLMo-2-0425-1B-RLVR1 revisions
- SNIPPET: vLLM async RL docs (search result), llm-d rollouts docs, TraceLab pages, AWS price trackers
