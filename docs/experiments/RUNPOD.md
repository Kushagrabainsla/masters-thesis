# Runpod operations and incident playbook

Purpose: run a GPU experiment on Runpod with no lost data and no wasted money. This file is general.
An experiment's own steps are in its README and in `dossier/GPU_RUNBOOK_2026-10-20.md`.

## 0. Shared account: what belongs to the thesis (D-59)

The Runpod account, its balance, its SSH keys and its $80/h spend limit are shared with other projects
and sessions. Thesis work acts only on its own resources.

- **Label every thesis resource.** Name: `msthesis-<experiment>-<purpose>-<yyyymmdd>` (make it with
  `python3 experiments/runpod/pods.py name E-36 rehearsal`). Pod env: `THESIS_TAG=msthesis` and
  `THESIS_EXPERIMENT=<E-nn>`. The same prefix applies to templates and volumes.
- **Register it at once.** Right after a create call returns the id:
  `python3 experiments/runpod/pods.py add <id> <name> <E-nn> pod`. After a terminate or delete:
  `python3 experiments/runpod/pods.py end <id>`.
- **Check before every change.** Before a stop, restart, update, terminate or delete:
  `python3 experiments/runpod/pods.py check <id>`. If it fails, do not touch the resource.
- **A label alone is not proof.** Another session can use the same words. Only a registry line made
  by a thesis session proves ownership.
- **Everything else is read-only.** List it, report it to the user if it matters (for example, it
  uses the shared balance), and never change it.
- **Count the other spend.** The balance check uses `--hours`, so another session's running Pods are
  part of the check. Our Pod plus theirs must also stay under the $80/h spend limit.
- **Billing per run.** Use `list-pod-billing` filtered to the thesis Pod id, not the account total.

## 1. Access and credentials

| Need | How | Where it lives |
|---|---|---|
| Control plane (Pods, catalog, billing history) | Runpod MCP tools, OAuth sign-in in `/mcp` | Claude Code settings (no file) |
| Live balance | `python3 experiments/runpod/balance.py` (GraphQL `myself`; the MCP tools have no balance read) | `RUNPOD_API_KEY` in the repo's gitignored `.env` |
| SSH and rsync into a Pod | Key pair `~/.ssh/runpod_ed25519`, public key registered on the account | `~/.ssh/` (never in the repo) |
| The Pod stopping itself | Runpod puts a Pod-scoped `RUNPOD_API_KEY` in each Pod | Pod environment (never written to a file) |

Rules: never print or commit a credential. Never put one in a command line, a URL or a log. If a
key is missing, stop and ask the user to add it to `.env`; do not ask for it in chat.

## 2. Before booking

1. Load the Runpod MCP playbook (`runpod:runpod-mcp` skill).
2. Read live stock and price (`get-gpu-type` with availability, product POD, cloud SECURE, and a CUDA
   floor of 12.8). Quote the price from that read.
3. Compute the cost cap: Pod time limit x price per hour, plus disk. Round up.
4. Read the live balance: `python3 experiments/runpod/balance.py --need <cap> --hours <Pod time limit>`.
   It must pass. It counts the spend of Pods from other sessions that already run.
5. Check that the SSH key is on the account (`get-ssh-keys`).
6. Get the user's approval of the plan and the cap in chat (journal D-nn).

## 3. Choosing the Pod

- Secure Cloud, on-demand, not interruptible: a run is not stopped halfway and the data center is
  known. Use Community Cloud only if the design accepts host variation, and say so.
- The GPU type in the design, never a near substitute (H100 SXM is not H100 PCIe or NVL).
- A volume disk at `/workspace` for code, caches and results. The container disk is wiped on stop.
- Ports `22/tcp`, SSH start on. A non-Runpod image needs a start command that installs and starts
  sshd (see the runbook).
- Pin the CUDA floor (`minCudaVersion` 12.8, or 13.0 for a CUDA 13 image).
- Boot watchdog (E-42 incident): the start command must stop the Pod by itself at the cost cap, counted
  from boot, before any session starts. Put it before sshd in the start command:
  `(sleep CAP; python3 -c "import json,os,urllib.request as u;q={'query':'mutation{podStop(input:{podId:\"%s\"}){id}}'%os.environ['RUNPOD_POD_ID']};r=u.Request('https://api.runpod.io/graphql',json.dumps(q).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+os.environ['RUNPOD_API_KEY']});print(u.urlopen(r,timeout=30).status)" >> /workspace/boot_watchdog.log 2>&1) &`
  (R-52: python3 is certain in the vLLM image; errors go to the log.) It uses the Pod-scoped `RUNPOD_API_KEY`
  that Runpod sets in the Pod, the same key AUTO_STOP uses; no write key from `.env` is needed. Set CAP to the
  cost cap in seconds, above the planned session time. A stopped Pod still bills its disk. Rehearse it once on
  an L4 with CAP 60 before the first GPU run that depends on it.
  The session watchdog (POD_LIMIT) starts only with the session, so it does not cover the time before.
- Do the steps from create to session start in one tool call, with a wait loop inside the call. Do not
  wait for a background timer or a notification while a GPU Pod bills and no session runs.

## 4. Rehearsal (stage S5)

Book the cheapest GPU that runs the same image (L4 at about $0.59/h, or RTX 4090) for 20 to 30
minutes. Check: SSH and rsync work; `hardware_record.sh` and `gpu_microbench.py` run; telemetry starts
with a `timestamp` header; `/proc/1/environ` holds `RUNPOD_POD_ID` and `RUNPOD_API_KEY` (print names
only); one short run of the session script ends with an ok `run.json`; `AUTO_STOP` stops the Pod.
Terminate the rehearsal Pod, then book the real one.

## 5. During the run: checkpoints

| When | Check | If it fails |
|---|---|---|
| Create returned | Register the id (`pods.py add`) before anything else |  |
| Pod RUNNING | Save the Pod record JSON; note the start time (UTC) | Wait; do not create a second Pod while the image pulls |
| After setup | `setup done` in the log; telemetry grows; the GPU in `gpu.csv` is the planned GPU | Abort rule 1 (`PROTOCOL.md`) |
| After the first run | `run.json` ok; no kernel-selection WARNING; the profile table exists | Read `log.txt`; fix only with an amendment |
| Every hour | Elapsed cost against the cap; balance | Abort rules 2 and 5 |
| Every run | The watcher matches any non-zero exit (`exit [1-9]`), not only the success markers | Stop and diagnose before the next run (E-38: a missed exit 124 cost about 3 h) |
| End | `done after` in the log; Pod stopped (AUTO_STOP) | Stop the Pod with the MCP tools |

## 6. Teardown (stage S7)

1. If the Pod is stopped, start it again (0 GPUs is enough to read the disk).
2. Copy the results to the Mac with `rsync -rlt` (the Pod volume disk is a network filesystem that
   refuses `chown`, so `rsync -a` fails). Then write and compare a checksum list on both sides:
   `find <dir> -type f -exec shasum -a 256 {} + | sort -k 2`.
3. Run `pods.py check <id>`, then terminate the Pod (a stopped Pod's volume disk bills
   $0.20/GB/month). Read it again to confirm it is gone, then run `pods.py end <id>`.
4. Read billing with `list-pod-billing` for the thesis Pod id and record the billed cost.
5. Add the row to `experiments/runpod/HARDWARE_LOG.md`.
6. Write the run manifests and add them to the ledger.

## 7. Incident playbook

| Incident | Most likely cause | Action |
|---|---|---|
| No stock for the planned GPU | Capacity | Wait, or try another Secure data center with stock. Do not change the GPU type without the user |
| Pod RUNNING but SSH refused | sshd not started yet, or `ssh.direct` not assigned | Wait for `ssh.direct` in the Pod record; read the Pod logs; check the start command |
| Pod restarted with 0 GPUs | Another renter took the GPU on that host | Copy the data off with a 0-GPU start; create a new Pod; the session script stops if the GPU UUID changed |
| A run hangs | Kernel or profiler hang | The per-run timeout kills it; the script continues; read its `log.txt` |
| The session script died | Crash or SSH loss without nohup | Start it again with the same results root; finished runs are skipped |
| Balance low | Spend higher than planned | Abort rule 5: stop the Pod, copy the data, tell the user |
| AUTO_STOP did not stop the Pod | Key scope or missing variable | Stop the Pod with the MCP tools; log the HTTP code from `session.log` |
| Results incomplete after copy | rsync stopped | Run rsync again, then compare the checksums |
| GPU Pod idle before the session starts | The agent waited for a background timer, and the wake-up came late | Stop the Pod at once; tell the user the cost; the boot watchdog (Section 3) prevents it |

After any incident, write a short note in the run report: what happened, the evidence, the cost, and
what changes next time. No blame, only facts.
