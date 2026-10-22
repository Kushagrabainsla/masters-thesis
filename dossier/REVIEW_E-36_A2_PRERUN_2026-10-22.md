# R-39: Sonnet pre-run review of E-36 amendment A2 (2026-10-22)

Scope: E-36 amendment A2 (Runpod efficiency changes and the exploratory Phase B), `session1.sh`,
`run_cell.py`, `extras.py`, `experiments/runpod/`, runbook Section 2a. `compare.py` is frozen.
Method: Claude Sonnet, read-only. It read all files, ran `sh -n` on `session1.sh` and the unit tests
(18 passed at that time). No GPU or cloud work. Runpod facts came from a separate read-only session
with the Runpod MCP tools on 2026-10-22 (catalog reads, `search-docs`, `lookup-concept`).

Verdict: AMEND_THEN_RUN. Phase A is intact. Five fixes were needed before a run.

## Findings and what the main session did

| # | Severity, label | Finding | Action |
|---|---|---|---|
| 1 | Blocker, FACT | Runbook 2a had no volume disk and used `/code` and `/results` on the container disk, which a Pod stop wipes. Resume could not work. | Fixed: volume disk at `/workspace`; code, results and caches there (runbook 2a) |
| 2 | Blocker, INTERPRETATION | An SSH shell on Runpod does not get the Pod environment, so `RUNPOD_POD_ID`, `RUNPOD_API_KEY`, `HF_TOKEN` can be empty and AUTO_STOP does nothing | Fixed: `session1.sh` reads only these three names from `/proc/1/environ` and writes no value to a file |
| 3 | Major, FACT | `run.json` was written after the profiler, so a profiler crash or timeout lost a measured run | Fixed: `run.json` is written before the profile and again after it; `timeout -k 30` |
| 4 | Major, FACT | AUTO_STOP ran only on the normal path, with no check of the HTTP result | Fixed: stop in the EXIT trap, three tries with the HTTP code logged, and a watchdog at `POD_LIMIT` (6 h 40 min) |
| 5 | Major, INTERPRETATION | 200 ms NVML polling is a new condition for the frozen Phase A; timestamps are local time; old drivers name the field `clocks_throttle_reasons` | Fixed: 1000 ms, six fields, `TZ=UTC`, automatic fallback to the old field name, header-based parsing |
| 6 | Major, FACT | The top-60 kernel-name set check was noise-ranked and fragile | Fixed: keep the top 150 and every kernel with at least 0.5% of time; profile at batch 1, 8 and 64; compare only GEMM and MoE kernel names and time shares at batch 64; documented as a weak second signal |
| 7 | Minor, FACT | B1 has no throughput, so `compare.py` always labels it INCONCLUSIVE | Fixed: `extras.py` reports the B1 flip counts, not the label |
| 8 | Minor, FACT | B2 ranking included aliases, the W8A16 control and failed runs | Fixed: each backend carries its Phase A status; only R and candidates are eligible as fastest |
| 9 | Minor, FACT | Resume reset the time caps and could mix GPUs | Fixed: elapsed time kept in `elapsed_s`; GPU UUID check at each start (exit 2 on a change unless `ALLOW_GPU_CHANGE=1`) |
| 10 | Minor, INTERPRETATION | Bit 0x4 (software power cap) is normal on a loaded H100 SXM, so the 5% rule would flag most runs | Fixed: power-cap and hard-slowdown fractions are separate; "suspect" uses hard bits or an SM clock more than 5% below the session median |
| 11 | Minor, INTERPRETATION | B2 at 256 may queue if the KV cache is too small | Fixed: `extras.py` reports the KV cache size from each B2 R_a log |
| 12 | Minor, FACT | B3 runs R twice (about 8 min) | Kept: the second run is the eager-mode noise reference |
| 13 | None, FACT | No defect in the telemetry stop, the exit-code fix, prefetch timing or the Phase A runs | None |

Efficiency suggestions taken: kernel profiles at batch 1, 8 and 64 (seconds per run); a 20-second
GPU microbenchmark (BF16 GEMM, device copy bandwidth) at the start and the end
(`experiments/runpod/gpu_microbench.py`); B0, a repeat of the most-flipping candidate in each cell
(are its flips repeatable?). Phase B order is now B0, B1, B3, B2, so the speed-only sweep is cut first.
Time estimates are written as ranges.

## Runpod facts (2026-10-22, read-only session)

- FACT: H100 SXM 80GB Secure Cloud $3.99/h, stock LOW in EU-FR-1 only (catalog). Community Cloud
  $2.69/h but stock NONE. H100 PCIe $2.89/h and H100 NVL $3.19/h, both stock NONE.
- FACT: billing is per second for compute, container disk and volume disk. Container disk $0.10/GB/month
  while running; volume disk $0.10/GB/month while running and $0.20/GB/month while stopped.
- FACT: a stop wipes the container disk; the volume disk at `/workspace` keeps its data; a terminate
  deletes the volume disk.
- FACT: if the account balance reaches $0, Pods stop, and Pods without a network volume are terminated.
- FACT: Runpod sets a Pod-scoped `RUNPOD_API_KEY` and `RUNPOD_POD_ID` in each Pod. UNKNOWN: the key's
  scope on `POST /v1/pods/{id}/stop`.
- FACT: spot Pods cannot be made through REST v2, the MCP tools or `runpodctl` (GraphQL only).
- UNKNOWN: billing during the image pull; the `/dev/shm` size rule; GPU performance counters in a Pod.

## Tests after the fixes (FACT, Mac)

- `test_compare.py` and `test_extras.py`: 22 tests pass.
- Stub session (stub `nvidia-smi` that rejects `clocks_event_reasons`, stub `timeout`, `pip` and
  `run_cell.py`, one candidate forced to fail, `AUTO_STOP=1` without Pod variables): telemetry fell back
  to the old field name; 46 runs; Phase A decision, Phase B and `extras.json` written; AUTO_STOP logged
  the missing variables; telemetry stopped. The second start skipped 43 ok runs and ran the 3 failed
  runs again; elapsed time added up across starts. A third start with a different GPU UUID stopped with
  exit 2.
- `run_cell.py --profile-kernels` in the vLLM v0.32.0 CPU image (Qwen3-0.6B): the run ended ok, and the
  profiler failure on CPU was written as `profile_error`. The kernel table itself needs CUDA (VERIFY on
  the H100).
