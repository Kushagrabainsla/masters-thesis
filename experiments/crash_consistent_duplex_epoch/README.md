# E-34: Crash-consistent duplex recovery epoch

Status: KILL-DESIGN. NOT IMPLEMENTED. NOT RUN.

Idea: I-132. Date: 2026-10-19.

Do not implement or run this experiment. R-29 returned `KILL_IDEA`.

## Question

Can one durable epoch recover opaque model state and duplex protocol state after a fail-stop worker crash?

The recovered client must observe the same event sequence as a fault-free reference.

## Claim boundary

Do not claim exactly-once network delivery.

Use stable event identifiers and client deduplication. The network can deliver an event more than once.

Measure exactly-once visible events after client deduplication.

Treat the Docker run as a correctness screen. Do not infer GPU or production latency from it.

## Environment

Run in one Linux ARM64 Docker image on the Apple M4 Mac.

Disable the container network during the frozen run.

Pin four CPUs and 4 GiB of memory. Pin the image digest.

Use a named Docker volume for durable state. Keep code and results in separate read-only and write-only mounts.

Record Docker, kernel, Python, SQLite, image, and repository versions.

Docker limits reduce variability. They do not isolate the experiment from all Mac host load.

## Workload

Build a deterministic service with opaque byte state.

Each input unit performs these actions.

1. Read the prior opaque state.
2. Apply a deterministic state transition.
3. Emit zero to three ordered output events.
4. Advance one input position.

Include RNG state in the opaque state. Use a fixed seed for each trace.

Use an independent pure reference model. It computes the expected final digest and event sequence.

Do not read implementation metadata as the expected result.

## Correct policies

Use the same client, stable event identifiers, acknowledgement logic, and output journal for all policies.

### B1: Replay all

Persist every input and output event. Keep no intermediate state snapshot.

After a crash, replay all accepted inputs from the initial state.

### B2: Full snapshot per unit

Persist a complete state snapshot and its frontiers after every input unit.

After a crash, restore the latest complete snapshot.

### P1: Periodic epoch

Persist a complete state snapshot every K units.

Persist accepted inputs and output events between snapshots.

Publish one epoch manifest only after all referenced data is durable.

The manifest contains these positions.

- the state-snapshot epoch;
- the highest input included in the state;
- the highest durable server event;
- the highest client-received event;
- the highest playback-acknowledged event.

Ignore an incomplete manifest or data suffix after recovery.

## Negative controls

Use negative controls only to validate the oracle and crash injector.

- Remove the durable output journal.
- Publish the manifest before the state bytes are durable.
- Omit RNG state from the snapshot.
- Reuse a new event identifier after replay.

Each negative control must fail at least one expected cut.

## Crash cuts

An external supervisor terminates the worker with `SIGKILL`.

The supervisor does not inspect the selected policy.

Inject a crash at these uniform markers.

1. Before the input record is durable.
2. After the input record is durable.
3. After the in-memory state transition.
4. During snapshot data creation.
5. After snapshot data is durable and before manifest publication.
6. After the output event is durable and before send.
7. After send and before the client acknowledgement is durable.
8. After the acknowledgement and before log compaction.

Also inject crashes after pseudo-random operation counts. Use fixed seeds.

## Independent variables

- policy: B1, B2, or P1;
- state size: 1, 16, 64, and 224 MiB;
- session length: 1, 10, 100, and 1,000 units;
- snapshot interval K: 1, 2, 4, 8, and 16 units;
- client acknowledgement lag: 0, 1, 4, and 16 events;
- output count: 0 to 3 events per input;
- crash marker;
- fixed seed.

The 224 MiB point comes from the 4K capsule reported by Execution-State Capsules.

Do not run the complete Cartesian product. Freeze a covering array after the review.

## Dependent variables

- final state digest equality;
- visible event sequence equality after client deduplication;
- lost visible events;
- duplicate visible events;
- recovery work in replayed units;
- durable bytes written;
- recovery wall time;
- steady-state wall time;
- peak container memory;
- invalid or incomplete epochs ignored.

Treat wall time as secondary on Docker Desktop.

## Procedure

1. Run the fault-free reference for each trace.
2. Run every policy without a crash.
3. Run each negative control at its expected cut.
4. Run each correct policy at every frozen crash cut.
5. Restart the worker in a new container with the same durable volume.
6. Resume the external client from its durable event position.
7. Compare state and events with the independent reference.
8. Repeat timed cells five times in interleaved policy order.
9. Save raw runs before computing a decision.

## Predeclared criteria

These thresholds are project heuristics.

Apply these rules in order.

1. `INVALID`: Any expected negative control passes.
2. `KILL-CORRECTNESS`: P1 fails one legal crash cut in the frozen correctness matrix.
3. `KILL-SIMPLE`: B2 meets the target with no more than 3% steady-state overhead.
4. `KILL-REPLAY`: B1 recovers within one application unit for the supported workload.
5. `PIVOT-GPU`: P1 passes, but only Docker timing distinguishes the policies.
6. `GO-GPU`: P1 passes, both simple baselines fail, and sourced workload mapping supports the difference.

E-34 cannot produce a thesis GO by itself.

## Required rubber-duck checks

Answer these questions before implementation.

1. Does each policy implement the same client contract?
2. Can B1 or B2 pass every correctness test?
3. Is the reference model independent of recovery metadata?
4. Does the crash supervisor use the same markers for every policy?
5. Can a crash occur during state, manifest, output, and acknowledgement writes?
6. Can stable identifiers hide lost payloads?
7. Does the oracle compare identifiers, payloads, order, and final state?
8. Does any decision depend on Docker storage speed alone?
9. Is I-132 more than Flink plus a model snapshot?
10. Can a simpler atomic manifest solve the whole test?

## Current rubber-duck result

The design passes questions 1 to 8 on paper.

Questions 9 and 10 remain unresolved novelty blockers.

R-29 confirmed both blockers. P1 is standard checkpoint composition.

B1 and B2 are the endpoint checkpoint intervals of P1.

The deterministic workload also removes nondeterministic GPU replay.

See `dossier/REVIEW_E-34_PRERUN_2026-10-19.md`.

## Planned outputs

These outputs were cancelled before implementation.

- `manifest.json`;
- `covering_array.json`;
- `runs.jsonl`;
- `correctness.csv`;
- `performance.csv`;
- `decision.json`;
- `decision.md`;
- `plots/recovery_work.pdf`;
- `plots/durable_bytes.pdf`.

## Validity limits

- Synthetic opaque bytes do not reproduce GPU snapshot behavior.
- Docker Desktop storage does not represent CXL, RDMA, HBM, or production disks.
- A deterministic service does not prove cross-GPU numerical equivalence.
- Stable client deduplication is required.
- A protocol proof does not establish a systems-paper contribution.

