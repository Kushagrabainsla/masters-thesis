# GPU runbook (2026-10-20)

Purpose: when GPU time becomes available, start here. Each session has a frozen design, scripts that
were checked on the Mac, a cost cap, and a decision rule. Prices are SNIPPET-level third-party figures
from 2026-10-20; check the provider page before booking.

Before any paid session: (1) the Sonnet pre-run review of that session is done and its amendments are
applied (D-29); (2) the Mac dry run passed; (3) the user approved the spend.

Provider (D-56, user, 2026-10-22): run the GPU experiments on Runpod. Keep all hardware details of each
run (`experiments/runpod/README.md`, `experiments/runpod/HARDWARE_LOG.md`). Session 1 uses Section 2a
(E-36 amendment A2 after R-39; Runpod H100 SXM $3.99/h on 2026-10-22).
Session 2 (E-30) needs root and power caps, which Runpod Pods do not give, so it stays on a provider
with root. The Lambda prices below are from 2026-10-20; read the Runpod price and stock before booking.

## 1. Session order

| # | Session | Idea | Hardware | Time cap | Cost cap | Status |
|---|---|---|---|---|---|---|
| 1 | E-36 Test A calibration | I-23 | 1 x H100 SXM (Lambda, about $4.29/h) | 4 h | $18 | Ready: A1 applied, Mac dry run passed, 14 tests pass; waits for D-51 |
| 2 | E-30 power-coupling premise | I-118 | 1 x GH200 with root (Lambda; price not confirmed) | 2.5 h | about $8 at $3/h | Ready: A1 applied, `sweep.py --dry-run` passed, 9 tests pass; waits for D-52 and the provider's answers on root and caps |
| 3 | E-36 NVFP4 cells | I-23 | 1 x B200 (Lambda, about $6.99/h) | 3 h | $21 | Only if Session 1 says CONTINUE |
| 4 | Test B and the NCCL contract | I-18, I-34 | 8-GPU NVLink node | 3 h | grant | Grant only; fix the I-18 design first |
| 5 | DSA indexer key dump | I-83 | 8 x H200 (DeepSeek-V3.2 or GLM-5.x) | 3 h | grant | Low priority (see Section 4) |

Dropped from the plan: I-120 (R-33: upstream owns the mechanism, and the maintainers propose a simpler
request-level baseline). I-59, I-60, I-106 are KILL (E-35, R-30).

## 2. Session 1: I-23, E-36 on one H100

Design: `experiments/safedispatch_testA/README.md` (Sections 9 and 10 win over Sections 3 to 6).
Decision: STOP-FLAT, CONTINUE, STATIC-WINNER or INCONCLUSIVE.

Steps on the rented machine (Lambda on-demand images include Docker and the NVIDIA container toolkit):
1. Copy the folder: `rsync -av experiments/safedispatch_testA/ ubuntu@<host>:~/e36/`
2. Pull the image: `docker pull vllm/vllm-openai:v0.32.0` and record `docker image inspect --format '{{.Id}}' vllm/vllm-openai:v0.32.0`.
3. Run:
   `docker run --gpus all --rm --ipc=host -v ~/e36:/code:ro -v ~/results:/results -v ~/.cache/huggingface:/root/.cache/huggingface --entrypoint sh vllm/vllm-openai:v0.32.0 /code/session1.sh`
4. Watch `~/results/session1-*/session.log`. The script stops starting new work after 3 h 10 min.
   After the first run (about 10 min), look for a WARNING line about kernel selection. If one shows,
   alias detection cannot work: let the session finish, then check each candidate's `log.txt` by hand.
5. Check `summary.json`: `kernel_log_found` must be true in each cell.
6. Copy off: `rsync -av ubuntu@<host>:~/results/ experiments/safedispatch_testA/results/`
7. Terminate the instance in the provider console. Check that billing stopped.

## 2a. Session 1 on Runpod (D-56, E-36 amendment A2 after R-39)

The steps in Section 2 assume a VM with Docker. A Runpod Pod is already a container, so the vLLM image
is the Pod image and `session1.sh` runs directly. Phase A (the decision) is the frozen A1 design; A2 adds
resume, telemetry, profiles and an exploratory Phase B (`experiments/safedispatch_testA/README.md`
Section 11). Steps marked VERIFY are not yet checked on a real Pod.

Prices and stock (2026-10-22, Runpod catalog): H100 SXM 80GB Secure Cloud $3.99/h, stock LOW in EU-FR-1
only. Community Cloud and the other H100 types had no stock. Spot Pods cannot be made with the MCP tools.

| Plan | Pod time | Cost at $3.99/h | Data |
|---|---|---|---|
| Full A2 (default) | at most 6 h 40 min | at most about $27 | Phase A decision plus B0, B1, B3, B2 |
| Phase A only (`PHASE_B=0`) | about 2 h 45 min to 3 h 30 min | about $11 to $14 | Phase A decision, telemetry, profiles |

Before you book:
- Read the live account balance: `python3 experiments/runpod/balance.py --need <cost cap> --hours <Pod limit>`.
  The account is shared (D-59), so the check also counts other sessions' running Pods. The MCP
  tools have no balance read, so the script uses `RUNPOD_API_KEY` from the gitignored `.env`. Do not
  use a stored number. The balance must be at least the plan's cost cap plus $35. If it is lower, stop and tell the
  user. FACT (Runpod docs): at a $0 balance, a Pod without a network volume is terminated, and its
  volume disk with the results is deleted.
- Get the user's approval of the spend and the plan (D-51, D-57).

Steps (they implement stages S4 to S7 of `docs/experiments/PROTOCOL.md`):
0. Rehearsal (stage S5): run steps 3 to 8 first on a 1 x L4 Pod (about $0.59/h, 20 to 30 min) with
   `PHASE_B=0 STOP_AFTER=600 DENSE_CANDS=torch MOE_CANDS=triton`. Check every VERIFY item, then
   terminate it. Do not change code between the rehearsal and the real run, except a required fix.
   The FP8 kernels of the real run need sm_90, so the rehearsal checks only the mechanics.
1. Check stock and price with the `runpod:discovery` skill for 1 x H100 SXM 80GB, on-demand, Secure
   Cloud. Record the data center and price. If the price is above $4.20/h, stop and ask the user.
   Do not change to H100 PCIe or NVL: the design is for H100 SXM.
2. Check that the SSH public key is on the Runpod account (`get-ssh-keys`). The key pair is
   `~/.ssh/runpod_ed25519` on the Mac (comment `runpod-masters-thesis`, registered 2026-10-25). The
   private key never leaves the Mac.
3. Create a network volume first (`create-network-volume`, 120 GB, STANDARD, in a data center with
   H100 SXM stock; $0.07/GB/month). Name it with `pods.py name E-36 volume`, then `pods.py add <id>
   <name> E-36 volume`. Rehearsal 2 (2026-10-26) showed why: a volume disk is tied to one host, and
   a stopped Pod could not restart (host GPU taken) and the API refused a 0-GPU resume, so the
   results were lost. A network volume can be mounted on any new Pod in its data center.
   Then create the Pod (`runpod:pod-deploy`), labelled for the thesis (D-59), with `dataCenterIds` set
   to the volume's data center:
   - name from `python3 experiments/runpod/pods.py name E-36 session1`; env `THESIS_TAG=msthesis`
     and `THESIS_EXPERIMENT=E-36`; right after the create returns, `pods.py add <id> <name> E-36 pod`;
   - image `vllm/vllm-openai:v0.32.0`, GPU 1 x H100 SXM 80GB, Secure Cloud, not interruptible;
   - container disk 40 GB; `mounts.network` = the volume at `/workspace` (models about 40 GB, caches,
     results); no volume disk;
   - ports `22/tcp`; SSH start on (this puts the account keys in `PUBLIC_KEY`);
   - environment: `HF_HOME=/workspace/cache/hf`, `VLLM_CACHE_ROOT=/workspace/cache/vllm`, and
     `HF_TOKEN` only if a download needs it. Do not write the token in any file;
   - `entrypoint` `["bash", "-c"]` and `cmd` with one string, because the vLLM image does not
     start sshd (VERIFY):
     `apt-get update -qq && apt-get install -y -qq openssh-server rsync && mkdir -p /root/.ssh /run/sshd && echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys && chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity`
4. Wait for `ssh.direct` in the Pod record. Save the Pod record JSON as
   `experiments/safedispatch_testA/results/session1/pod.json` on the Mac. Note the start time (UTC).
5. Copy the code to the volume disk. Use `rsync -rlt`, not `-a`: `/workspace` is a network filesystem
   (MooseFS, `mfs#<region>.runpod.net`) that refuses `chown` (rehearsal 2026-10-25). Exclude
   `results`, `dryrun`, `__pycache__` and `.pytest_cache`:
   `rsync -rlt -e "ssh -i ~/.ssh/runpod_ed25519 -p <port>" experiments/safedispatch_testA/ root@<ip>:/workspace/code/`
   `rsync -rlt -e "ssh -i ~/.ssh/runpod_ed25519 -p <port>" experiments/runpod/ root@<ip>:/workspace/code/runpod/`
6. First checks on the Pod (about 2 minutes, VERIFY items):
   - `sh /workspace/code/runpod/hardware_record.sh /workspace/check`: the GPU in `gpu.csv` must be
     an H100 80GB HBM3 with sm_90 in `python.txt`. Read `/dev/shm` in `df.txt`.
   - `tr '\0' '\n' < /proc/1/environ | cut -d= -f1 | grep RUNPOD_`: `RUNPOD_POD_ID` and
     `RUNPOD_API_KEY` must be there, or `AUTO_STOP` cannot work. Print names only, never values.
   - `nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv`: if it fails, the script uses
     the old field name; check that the first `telemetry.csv` line starts with `timestamp`.
7. Start the session in the background, so a dropped SSH session does not stop it:
   `ssh -f -n ... 'cd /workspace && CODE=/workspace/code RESULTS_ROOT=/workspace/results AUTO_STOP=1 AUTO_STOP_DELAY=600 HF_HOME=/workspace/cache/hf VLLM_CACHE_ROOT=/workspace/cache/vllm setsid nohup sh /workspace/code/session1.sh > /workspace/nohup.txt 2>&1 < /dev/null &'`
   (`ssh -f -n` and `setsid` keep the local SSH from hanging.) If the Pod logs show the image pull
   failing in a loop ("failed to register layer"), the host is broken: terminate the Pod and create
   it again, if needed with another data center or GPU type (rehearsal 2, 2026-10-26).
   For Phase A only, add `PHASE_B=0`.
8. Watch `/workspace/results/session1/session.log`:
   - after setup: `setup done`, and `env-*/telemetry.csv` grows;
   - after the first run (about 10 min): no WARNING about kernel selection, and `cell_D/R_a/kernels.json`
     has a `by_batch` table (VERIFY: the profiler on the H100). If it has a `profile_error`, the run
     still counts; tell the user;
   - after Phase A: the `decision:` line.
9. If the session dies: run step 7 again. If the Pod stopped and cannot start (host GPU taken):
   terminate it, create a new Pod with the same network volume, and run step 7 again. Finished runs
   are skipped. If the log says `STOP: GPU changed`, ask the user before you set `ALLOW_GPU_CHANGE=1`.
10. At the end, the log prints `AUTO_STOP in 600 s; copy the results now`. Copy off in that window
    (if the Pod already stopped, mount the network volume on a new cheap Pod and copy from there):
    `rsync -rlt -e "ssh -i ~/.ssh/runpod_ed25519 -p <port>" root@<ip>:/workspace/results/ experiments/safedispatch_testA/results/`
    Check that `summary.json`, `extras.json` and every `env-*/` folder arrived.
11. Compare checksums, then `pods.py check <id>`, terminate the Pod, and `pods.py end <id>`. Delete
    the network volume the same way (`delete-network-volume`) after the results are safe on the Mac.
    Read `list-pod-billing` for the Pod id and check that nothing thesis-owned bills.
12. Add the row to `experiments/runpod/HARDWARE_LOG.md`: Pod id, GPU, data center, cloud type, price,
    image tag, driver and CUDA (`env-*/hardware/gpu.csv`, `python.txt`), vCPU, RAM, disks, start and
    stop times, billed cost, the results path, and the microbenchmark numbers.
13. Add the runs to the ledger: `python3 experiments/safedispatch_testA/to_ledger.py --res
    experiments/safedispatch_testA/results/session1 --out experiments/safedispatch_testA/results/session1/manifests
    --design-commit <sha> --pod experiments/safedispatch_testA/results/session1/pod.json`, then
    `python3 experiments/ledger/ledger.py add experiments/safedispatch_testA/results/session1/manifests/*.json`.

## 3. Session 2: I-118, E-30 on one GH200

Design: `experiments/superchip_power_coupling/README.md` (Section 8 wins over Sections 2 to 5).
Decision: INVALID, INCONCLUSIVE, KILL 1, INCOMPLETE, KILL 2, GO, PIVOT.

1. Before booking: ask the provider whether a 1 x GH200 instance gives root, the "Module Power Readings"
   section in `nvidia-smi -q -d POWER`, `nvidia-smi -pl <W> -sc 1`, and writable Grace `power*_cap`.
2. On the machine: `sh preflight.sh`. Its only writes set each knob to its current value. Read
   `preflight/verdict.txt`. If root is missing, stop and terminate. If the module section or the CPU
   caps are absent, the design says which kill test cannot run; decide before you continue.
   Then run `python3 sweep.py --out ~/results/e30-dry --dry-run`. It prints the plan and changes nothing.
3. Install `stress-ng`; pull `vllm/vllm-openai:v0.32.0` (aarch64).
4. Start the logger in the background:
   `nvidia-smi --query-gpu=timestamp,power.draw,clocks.sm,clocks_event_reasons.active --format=csv -lms 100 > ~/results/gpu_log.csv &`
5. Run `python3 sweep.py --out ~/results/e30` (inside the vLLM container with `--privileged`, the host
   `/sys` mounted, and `--pid=host` so `taskset` and the caps work, or directly on the host with the
   vLLM wheel installed).
6. `python3 analyze.py ~/results/e30/medians.json --out ~/results/e30/decision.json`
7. Reset the module limit to its default (`sweep.py` does this at the end; check it), copy off, and
   terminate.

## 4. Status of the GPU-gated ideas (R-33, 2026-10-20)

| Idea | Status | What changed | GPU need |
|---|---|---|---|
| I-23 SafeDispatch | MEASURE-FIRST | vLLM v0.32.0 adds `--linear-backend`, `--moe-backend`, `--attention-backend`, so each backend pair is one flag. No direct scoop found. | Session 1, then 3 |
| I-118 power coupling | MEASURE-FIRST, low | No new work found. Lambda's MIG guide shows `sudo nvidia-smi` on a 1 x GH200 instance (SNIPPET), so root is likely. | Session 2 |
| I-120 KV activation | KILL recommended (R-33, D-50) | Upstream issues 61768, 61905, 61912 and PRs 61769, 53298, 59793 work on the same fault class with CPU scheduler probes; the maintainers' RFC 35780 and PR 42461 propose request-level recovery (reset to 0, recompute), which is the simple baseline. | None |
| I-83 indexer pruning | WEAK, lower | LiteTopK (2607.11976) reports that indexer scores concentrate in a narrow range, which makes block bounds loose; R-17's synthetic test pruned 0%. vLLM v0.32.0 has a pluggable `sparse_indexer_topk_backend`. No small open DSA model exists for a Mac test. | Session 5 only |
| I-18 / I-34 collectives | MEASURE-FIRST low / HYPOTHESIS | No new work found. | Session 4 |

## 5. After each session

Write the result into the experiment's README and the journal (Section 6.1 result, idea dossier,
Section 1), run the post-run Sonnet review, and keep the raw results and manifests in the experiment's
`results/` folder. For a Runpod run, also add the row to `experiments/runpod/HARDWARE_LOG.md` and put
the hardware (GPU, data center, price, driver, CUDA) in the E-nn "Setup" line (D-56).
