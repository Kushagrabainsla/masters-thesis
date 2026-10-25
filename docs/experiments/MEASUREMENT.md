# Measurement standards

Purpose: make every number comparable, so that we can say with evidence whether a change makes a
system better, worse or the same. `PROTOCOL.md` stages S1, S6 and S8 use these rules. All thresholds
here are project heuristics. A design can set stricter ones in its preregistration.

## 1. Before the run: decide what counts

1. Name one **primary metric** and its direction (higher or lower is better). The decision rule uses
   only the primary metric and the preregistered validity checks.
2. Name the **minimum effect that matters** (MDE). Default: 2% for speed, 0 flipped prompts for exact
   output equality. A smaller effect is "no change", even if it is real.
3. Name **secondary metrics**. They are reported, but they do not decide.
4. Name **validity checks** that make a run INVALID (for example, the reference disagrees with itself
   on more than 5% of prompts, or the GPU throttled).
5. Size the **repeats**: at least 5 for speed, so that a 95% interval exists. If the A/A spread is
   larger than half the MDE, add repeats or reduce the noise before you compare anything.

## 2. During the run: control the noise

| Source of noise | Control |
|---|---|
| Cold caches, JIT, autotuning | One warm-up before each timed batch size; record engine start time apart |
| Host drift (heat, neighbors) | Run A and B in the same Pod; alternate them (A, B, A, B) when the design allows |
| Different hosts of one GPU type | Do not compare across Pods without the microbenchmark check (Section 4) |
| Throttling | Telemetry for the whole session; mark runs with a hard slowdown as suspect |
| Batch and scheduling effects | Fix `max_num_seqs`, the batch composition and the order of prompts |
| Software change | Pin the image tag, model revisions and seeds; record them in the manifest |
| Prefix and KV caches | Turn caching off, or state that it is part of the treatment |
| Background processes | Nothing else runs on the GPU; the telemetry poller is slow (1 s) and the same for all runs |

## 3. After the run: compare with intervals, not single numbers

1. **A/A first.** Compare two runs of the baseline. The spread is the noise floor for that metric.
2. **Compare with the tool:**
   `python3 experiments/ledger/compare_runs.py <baseline_id> <candidate_id> --aa <baseline_repeat_id> --mde 0.02 --strict`
3. The tool reports, for each metric: both medians, the ratio, a bootstrap interval of the ratio, the
   noise band (the larger of the MDE and the A/A spread) and one verdict:

| Verdict | Meaning | What to write |
|---|---|---|
| BETTER | The whole interval is past the band, in the good direction | "Improves X by r (interval lo to hi)" |
| WORSE | The whole interval is past the band, in the bad direction | "Degrades X by r (interval lo to hi)" |
| NO_CHANGE | The whole interval is inside the band | "No change larger than the band" |
| UNCLEAR | Anything else | "Not resolved; needs n more repeats" |

4. **Many metrics.** The tool uses a Bonferroni level over the metrics of one comparison, so that the
   set has at most a 5% false-alarm rate. If a design compares many pairs, say how many in the report.
5. **Correctness metrics** (flips, exact match) use a Wilson 95% interval on the count. "0 flips" is
   reported with its upper bound (for example, at most 1.5% at 256 prompts).
6. **Report the absolute numbers**, not only ratios, and the hardware they came from.
7. **Do not cherry-pick.** Report every candidate and every run in the plan, including failures.

## 4. Comparing across sessions and hosts

- The ledger keeps every run, so a baseline from a past session can be compared with a new run.
- The compare tool prints every fingerprint difference (GPU type, count, driver, CUDA, software). With
  `--strict` it refuses. Without `--strict`, label the result "cross-setup".
- Two hosts of the same GPU type can differ. Before a cross-Pod comparison, compare the
  `gpu_microbench.py` results of both Pods. If the GEMM or copy numbers differ by more than 3%, do
  not compare speeds across them. Compare only within each Pod.
- A regression check for a change (for example, a new vLLM version) reruns the baseline config and
  the changed config in the same Pod. A stored baseline alone is not enough.

## 5. Run manifest (one per run)

Each run writes a manifest that `experiments/ledger/ledger.py add` accepts. Required fields:
`run_id`, `experiment`, `idea`, `status`, `design_commit`, `config`, `started_at`, `finished_at`,
`hardware` (`gpu_name`, `gpu_count`, `driver`, `cuda`, plus `gpu_uuid`, `pod_id`, `data_center`,
`price_per_hour` when known), `software`, `metrics` (`values` with one number per repeat, `unit`,
`higher_is_better`) and `results_path`. The ledger adds `config_hash`. E-36 writes its manifests with
`experiments/safedispatch_testA/to_ledger.py`. A new experiment needs its own small adapter.
