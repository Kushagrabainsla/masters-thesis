# Runpod GPU runs

Decision D-56 (user, 2026-10-22): run the GPU experiments on Runpod, and keep all details about the
hardware of each run. This folder holds the shared tools for that.

## Files

| File | Purpose |
|---|---|
| `hardware_record.sh` | Runs inside the Pod before the experiment. Writes `<results>/hardware/` (GPU, driver, CUDA, NVLink, clocks, power limits, CPU, memory, disk, OS, cgroup limits, Python stack). Never fails. Reads only an allowlist of `RUNPOD_*` variables, so the API key in the Pod environment is not written. |
| `gpu_microbench.py` | 20-second GPU health check (BF16 GEMM TFLOPS, device copy GB/s). Run at the start and the end of a session, so a weak or throttled host shows. |
| `balance.py` | Prints the live balance, spend per hour and spend limit (GraphQL `myself`; the MCP tools have no balance read). `--need <cap> --hours <h>` exits 1 below the cap plus $35 plus the other sessions' spend for <h> hours. Reads `RUNPOD_API_KEY` from the gitignored `.env` at the repo root and never prints it. |
| `pods.py`, `pods.jsonl` | D-59: label and registry of thesis-owned Runpod resources on the shared account. Change only what `pods.py check <id>` accepts. Append only. |
| `HARDWARE_LOG.md` | One row for each Pod that ran an experiment. Append only. |

## What to keep for each run

1. **The Pod record.** Before the run, read the Pod through the Runpod MCP tools (`get-pod`). Save
   the JSON as `<results>/hardware/pod.json`. It holds the GPU type and count, data center, cloud type
   (Secure or Community), interruptible flag, price per hour, image, disk sizes, vCPU and memory.
2. **The machine record.** Run `hardware_record.sh` inside the Pod at each start of a session. The
   E-36 `session1.sh` does this in each `env-<stamp>/` folder, together with GPU telemetry and
   `gpu_microbench.py`.
3. **The image digest.** Runpod does not show the digest inside the container. Record the tag in
   `HARDWARE_LOG.md`, and record the digest from `docker image inspect` on the Mac if it is pulled
   there.
4. **The times and the cost.** Write the Pod start and stop times (UTC) and the billed cost from the
   Runpod billing read in `HARDWARE_LOG.md`.
5. **The HARDWARE_LOG.md row.** Add it after the Pod is terminated and billing has stopped.

## Use Runpod efficiently

- Put the most data into one session. The image pull, model download and compile are paid once per
  Pod, so extra measurements in the same Pod cost less than a second Pod.
- Never lose a finished run. Keep code, results and caches on the volume disk (`/workspace`), and make
  the session script skip runs that are already done, so a restart continues.
- Never leave the GPU idle. Download models in the background, and stop the Pod at the end
  (`AUTO_STOP=1`, plus a watchdog). Terminate it after the results are copied off: a stopped Pod's
  volume disk still bills ($0.20/GB/month).
- Keep enough credit. Before every experiment, read the live balance with the Runpod tools. Do not
  write the balance in any file. It must be at least the run's cost cap plus $35. At a $0 balance, a
  Pod without a network volume is terminated with its disk.
- Record the GPU for each run, telemetry for the whole session and a microbenchmark at the start and
  the end, so speed results can be checked for throttling or a weak host.

## Rules

- Each paid run still needs the user's approval of its spend (for example D-51 for E-36).
- Use on-demand Secure Cloud Pods (not interruptible), so that a run is not stopped halfway and the
  data center is known.
- Pods are unprivileged containers. Work that needs root on the host, `--privileged`, power caps
  (`nvidia-smi -pl`) or clock locks does not fit Runpod. E-30 (I-118) stays on a provider that gives
  root (D-52).
- Do not run `docker` inside a Pod. The Pod is the container: choose the experiment image as the
  Pod image.
