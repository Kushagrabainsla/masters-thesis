# GPU runbook (2026-10-20)

Purpose: when GPU time becomes available, start here. Each session has a frozen design, scripts that
were checked on the Mac, a cost cap, and a decision rule. Prices are SNIPPET-level third-party figures
from 2026-10-20; check the provider page before booking.

Before any paid session: (1) the Sonnet pre-run review of that session is done and its amendments are
applied (D-29); (2) the Mac dry run passed; (3) the user approved the spend.

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
`results/` folder.
