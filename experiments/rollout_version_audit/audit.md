# E-43 result: rollout version audit (I-165 premise check)

Date: 2026-10-28. Rules: `README.md` Section 5 (written before any source was read). Commits: `README.md`
Section 7. No code ran. Labels: FACT (source read at the pinned commit), INTERPRETATION, HYPOTHESIS.
Status: frozen decision applied. The D-29 review is pending.

## 1. Audit table

Field meanings are in `README.md` Section 4. F1: in-flight requests. F2: KV after the update. F3: version
record. F4: check. F5: KV provenance (secondary). One row for each shipped mode that touches the update path.

| Framework | Mode | F1 | F2 | F3 | F4 | F5 | Evidence |
|---|---|---|---|---|---|---|---|
| vLLM (engine) | `pause_generation(mode="abort")` (default) | abort | flushed | per-token | none | exact | [V1], [V2], [V5] |
| vLLM (engine) | `mode="keep"`, `clear_cache=True` (default) | keep | recomputed | per-token | none | exact | [V1], [V2], [V3] |
| vLLM (engine) | `mode="keep"`, `clear_cache=False` | keep | kept | per-token | none | hybrid | [V2], [V4], [V5] |
| SGLang (engine) | `pause_generation` `abort` (default) | abort | flushed | per-token | none | exact | [S1], [S3], [S4] |
| SGLang (engine) | `retract` | keep | recomputed | per-token | none | exact | [S1], [S3] |
| SGLang (engine) | `in_place` | keep | kept | per-token | none | hybrid | [S1], [S2], [S3] |
| verl | synchronous PPO (default) | none | flushed | per-step | none | exact | [R1] |
| verl | `one_step_off_policy` | none | flushed | per-step | none | exact | [R2] |
| verl | `fully_async_policy`, `partial_rollout=True` | abort-resubmit | flushed | per-trajectory span | bound | exact | [R1], [R3], [R4] |
| slime | `--flush-cache-interval 1` (default) | abort-resubmit | flushed | per-request | none | exact | [L1], [L2], [L3] |
| slime | `--flush-cache-interval` other than 1 (fully async) | keep | kept | per-request (last version) | none | hybrid | [L1], [L2], [L3], [L4] |
| AReaL | async rollout (default) | abort-resubmit | flushed | per-token | bound | exact | [A1], [A2], [A3] |
| OpenRLHF | `--train.async_enable` without partial rollout | none | flushed | none | none | exact | [O1] |
| OpenRLHF | `--train.partial_rollout_enable` | keep | recomputed | none | none | exact | [O1], [O2], [O3] |
| PipelineRL | in-flight weight update (default) | keep | kept | per-trajectory (start version) | bound | hybrid | [P1], [P2], [P3] |

## 2. Evidence (FACT unless marked)

Link bases: vLLM `vllm-project/vllm@7b87062`, SGLang `sgl-project/sglang@88fea37`, verl
`volcengine/verl@2ab3529`, slime `THUDM/slime@540ea91`, AReaL `inclusionAI/AReaL@fa8bd78`, OpenRLHF
`OpenRLHF/OpenRLHF@dc2a7ad`, PipelineRL `ServiceNow/PipelineRL@58d3934`. Full SHAs are in `README.md`.

**vLLM**
- [V1] Pause modes are `abort`, `wait` and `keep`
  ([`vllm/v1/engine/__init__.py#L33`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/engine/__init__.py#L33)).
  `keep` sets `PAUSED_ALL` ([`vllm/v1/engine/core.py#L2271`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/engine/core.py#L2271)).
- [V2] `_finish_pause` resets caches only if `clear_cache` is true
  ([`core.py#L954-L966`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/engine/core.py#L954-L966)).
- [V3] The reset uses `reset_running_requests=True`, which preempts every running request, so its KV is
  recomputed after resume ([`vllm/v1/core/sched/scheduler.py#L3137-L3155`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/core/sched/scheduler.py#L3137-L3155)).
  A paused `keep` request does not count as unfinished, so the pause completes at once
  ([`scheduler.py#L3090-L3092`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/core/sched/scheduler.py#L3090-L3092)).
- [V4] The vLLM docs state that `clear_cache=False` keeps old-weight KV ("stale KV cache")
  ([`docs/training/async_rl.md#L71`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/docs/training/async_rl.md#L71)).
- [V5] Version record: the engine attaches its current label to each step output
  ([`core.py#L664-L667`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/engine/core.py#L664-L667)).
  The output processor builds `WeightVersionSpan(version, start, end)` over output-token indexes
  ([`vllm/outputs.py#L50`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/outputs.py#L50),
  [`vllm/v1/engine/output_processor.py#L531-L547`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/engine/output_processor.py#L531-L547)).
  The docs say "vLLM doesn't order or check them"
  ([`async_rl.md#L75`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/docs/training/async_rl.md#L75)).
  History (git log): the label came in #49040 (merged 2026-07-28). The per-token spans came in #59582
  (merged 2026-10-11).

**SGLang**
- [S1] Pause modes are `abort`, `retract` and `in_place`
  ([`python/sglang/srt/managers/io_struct.py#L1927-L1946`](https://github.com/sgl-project/sglang/blob/88fea372d47b4cb14261d786e3824beb4fa97fa5/python/sglang/srt/managers/io_struct.py#L1927-L1946)).
  `retract` moves requests to the waiting queue and recomputes KV.
- [S2] `in_place` continues "with the old kv cache"
  ([`io_struct.py#L1935`](https://github.com/sgl-project/sglang/blob/88fea372d47b4cb14261d786e3824beb4fa97fa5/python/sglang/srt/managers/io_struct.py#L1935)).
- [S3] Version record: on a version change the scheduler records an event with the output length for each
  live request, and the output builds per-token spans
  ([`python/sglang/srt/utils/weight_versions.py#L39-L84`](https://github.com/sgl-project/sglang/blob/88fea372d47b4cb14261d786e3824beb4fa97fa5/python/sglang/srt/utils/weight_versions.py#L39-L84)).
  `meta_info["weight_version"]` holds only the last span
  ([`weight_versions.py#L107`](https://github.com/sgl-project/sglang/blob/88fea372d47b4cb14261d786e3824beb4fa97fa5/python/sglang/srt/utils/weight_versions.py#L107)).
  History: #35926 (2026-08-24). No check was found.
- [S4] Weight-update requests default to `flush_cache=True`
  ([`io_struct.py#L1990`, `#L2009`, `#L2039`](https://github.com/sgl-project/sglang/blob/88fea372d47b4cb14261d786e3824beb4fa97fa5/python/sglang/srt/managers/io_struct.py#L1990)).

**verl**
- [R1] Abort path: `pause_generation(mode="abort", clear_cache=reset_prefix_cache)` with
  `reset_prefix_cache=True` by default
  ([`verl/workers/rollout/vllm_rollout/vllm_async_server.py#L1111-L1199`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L1111-L1199)).
  Each response carries the server `global_steps`
  ([`#L733`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L733)).
- [R2] `one_step_off_policy` waits for the batch, then syncs weights, then starts the next batch
  ([`verl/experimental/one_step_off_policy/ray_trainer.py#L385-L408`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/experimental/one_step_off_policy/ray_trainer.py#L385-L408)).
- [R3] Partial rollout resubmits `prompt_ids + final_output.token_ids` after an abort and keeps
  `min_global_steps` and `max_global_steps` for the trajectory
  ([`verl/workers/rollout/llm_server.py#L282-L367`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/workers/rollout/llm_server.py#L282-L367)).
  Without partial rollout, aborted samples are dropped (`#L355-L359`).
- [R4] Bound: `staleness_threshold` limits the share of stale samples
  ([`verl/experimental/fully_async_policy/fully_async_rollouter.py#L421`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/experimental/fully_async_policy/fully_async_rollouter.py#L421)).

**slime**
- [L1] All three weight transports call `pause_generation(mode="abort" if flush_cache else "in_place")`
  ([`slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L115`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L115),
  `update_weight_from_tensor.py#L299`, `update_weight_from_disk_delta.py#L196`).
- [L2] `should_flush_cache` flushes on the first sync and every N syncs; a value of 0 or less never flushes
  ([`slime/utils/weight_sync.py#L4`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/weight_sync.py#L4)).
  `--flush-cache-interval` defaults to 1. Its help text names the other values "PipelineRL" mode
  ([`slime/utils/arguments.py#L356-L365`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/arguments.py#L356-L365)).
- [L3] Version record: a sample appends `meta_info["weight_version"]` once for each generate call
  ([`slime/utils/types.py#L493-L494`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/types.py#L493-L494)).
  With [S3], that value is the last span. slime does not read `meta_info["weight_versions"]`.
- [L4] Staleness is a logged metric only
  ([`slime/utils/staleness.py#L20`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/staleness.py#L20)).

**AReaL**
- [A1] Weight updates send `abort_all_requests: True`, and SGLang flushes by default ([S4])
  ([`areal/engine/sglang_remote.py#L321`, `#L349`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/engine/sglang_remote.py#L321)).
  The colocate path aborts first, then pauses `in_place` with nothing left to keep
  ([`#L458-L478`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/engine/sglang_remote.py#L458-L478)).
- [A2] Each resubmitted segment pins its version before the call and tags every output token with it
  ([`areal/infra/remote_inf_engine.py#L1146-L1196`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/infra/remote_inf_engine.py#L1146-L1196),
  [`areal/api/io_struct.py#L69`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/api/io_struct.py#L69)).
- [A3] Bound: `StalenessManager` limits admission capacity by `max_head_offpolicyness`
  ([`areal/infra/staleness_manager.py#L79-L113`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/infra/staleness_manager.py#L79-L113)).

**OpenRLHF**
- [O1] Without partial rollout, a lock keeps the weight broadcast apart from generation
  ([`openrlhf/trainer/ppo_trainer_async.py#L254-L264`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L254-L264)).
  No version field was found in the repository.
- [O2] With partial rollout, the engine calls `pause_generation(mode="keep")` with the default
  `clear_cache=True` ([`openrlhf/trainer/ray/vllm_engine.py#L131`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ray/vllm_engine.py#L131)).
  The pinned vLLM 0.29.0 (`setup.py#L77`) preempts running requests on that reset
  (`vllm@98dff2a` `vllm/v1/engine/core.py#L827-L847`, `scheduler.py#L2563-L2565`). So F2 is recomputed.
- [O3] The flag help says "In-flight samples may contain tokens from both old and new weights"
  ([`openrlhf/cli/train_ppo_ray.py#L296-L302`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/cli/train_ppo_ray.py#L296-L302)).

**PipelineRL**
- [P1] Weight update: `pause_generation(mode="keep", clear_cache=False)`
  ([`pipelinerl/vllm1.py#L160`](https://github.com/ServiceNow/PipelineRL/blob/58d393458625ad63ed539f2dcd072c85700c557f/pipelinerl/vllm1.py#L160)).
  The pinned vLLM 0.18.1 (`pyproject.toml#L23`) resets caches only if `clear_cache` is true
  (`vllm@a26e8dc` `vllm/v1/engine/core.py#L1499-L1533`, the multiprocess `EngineCoreProc` path). So
  old-weight KV stays.
- [P2] The rollout gets the trainer version at rollout start
  ([`pipelinerl/actor.py#L186`, `#L216`](https://github.com/ServiceNow/PipelineRL/blob/58d393458625ad63ed539f2dcd072c85700c557f/pipelinerl/actor.py#L186)).
  A batch takes the minimum version of its samples
  ([`pipelinerl/finetune/data.py#L210`](https://github.com/ServiceNow/PipelineRL/blob/58d393458625ad63ed539f2dcd072c85700c557f/pipelinerl/finetune/data.py#L210)).
- [P3] Bound: `max_lag` limits submission. In the trainer, a too-old batch is counted, not dropped
  ([`pipelinerl/finetune_loop.py#L679-L684`](https://github.com/ServiceNow/PipelineRL/blob/58d393458625ad63ed539f2dcd072c85700c557f/pipelinerl/finetune_loop.py#L679-L684)).

**vLLM RFC #48311** (read 2026-10-28, open, opened 2026-07-11). It defines the pause, drain, sleep, wake
and resume contract. It does not define a version field. Its linked item #57810 says "a weight update must
invalidate old-weight KV" for offloaded KV.

## 3. Frozen decision

Rules are applied in the order of `README.md` Section 5.

1. **KILL (solved):** NOT MET. No framework has F4 = `check` in any mode. #48311 has no version field.
2. **GO:** MET. Three frameworks keep state in a mode where F3 is not `per-token` and F4 is not `check`:
   - slime, `--flush-cache-interval` other than 1: keep, kept KV, one label for each call, no check.
   - OpenRLHF, `--train.partial_rollout_enable`: keep, recomputed KV, no version record, no check.
   - PipelineRL, default: keep, kept KV, start version for each rollout, bound only.

   vLLM and SGLang keep state but record per-token versions, so they do not count. verl and AReaL do not
   keep state (abort-resubmit with a flush).

**Frozen decision: GO.** By R-54 Section 7, the next step is a 1-GPU fault-injection test that counts
mixed-version tokens. That step needs a predeclared plan and the user's approval before any GPU spend.

## 4. Scope limits on the GO (INTERPRETATION)

These points do not change the frozen decision. They change what a GO is worth.

1. **The version field exists upstream.** vLLM (2026-07-28 label, 2026-10-11 spans) and SGLang (2026-08-24
   spans) already emit per-token version spans. The three GO frameworks do not read them. So the
   "per-token record" part of I-165 is now an integration fix in each framework, not a research gap. R-54
   named this threat ("vLLM maintainers ... may define the version field first"). It happened, through
   #49040 and #59582, not through #48311.
2. **Labels mark sampling weights, not KV provenance.** In the `hybrid` rows (vLLM `keep` with
   `clear_cache=False`, SGLang `in_place`, slime fully async, PipelineRL), a token labelled v2 can come from v2
   weights over KV that v1 computed. No engine and no framework marks these tokens. No code checks a label.
   This part of I-165 is still open.
3. **Rollout logprobs can make labels unnecessary for the loss** (post-hoc field, not predeclared). If the
   trainer uses the logprob that the engine recorded when it sampled each token, importance sampling uses
   the real behavior distribution, also for a hybrid token. PipelineRL uses rollout logprobs as
   `old_logprobs` ([`pipelinerl/finetune/data.py#L150`](https://github.com/ServiceNow/PipelineRL/blob/58d393458625ad63ed539f2dcd072c85700c557f/pipelinerl/finetune/data.py#L150)).
   verl fully async sets `use_rollout_log_probs: True`
   ([`fully_async_ppo_trainer.yaml#L87`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/experimental/fully_async_policy/config/fully_async_ppo_trainer.yaml#L87)).
   slime defaults to actor logprobs (`--use-rollout-logprobs` is off,
   [`arguments.py#L1174`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/arguments.py#L1174)).
   OpenRLHF defaults to actor logprobs with IS correction off (`--algo.advantage.is_correction_level`
   default `off`, [`train_ppo_ray.py#L261-L267`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/cli/train_ppo_ray.py#L261-L267)).
   AReaL was not checked for this field.
   HYPOTHESIS: the in-flight mix adds training error only where the trainer uses actor logprobs (slime and
   OpenRLHF defaults). There it is one case of the general async-staleness error, not a new error class.
4. **Mislabels that a test can show** (HYPOTHESIS, from code reading). slime in its fully async mode labels a
   whole generate call with the last version, so tokens made before the update get the new label ([L3]).
   PipelineRL labels a rollout with its start version, so tokens made after the update get the old label
   ([P2]). A 1-GPU test can confirm both.

## 5. What this means for I-165 (INTERPRETATION, for the D-29 review)

- The premise holds by the frozen rule: three frameworks mix versions in shipped modes without a per-token
  record or a check.
- The research part is smaller than R-54 assumed. Per-token records exist in both engines. Rollout-logprob
  importance sampling covers the loss in two frameworks. What is left: (a) KV provenance for hybrid tokens,
  (b) a check that a label matches the weights and the KV that made a token, (c) the throughput cost of
  exactness (recompute or abort) against keeping stale KV. Item (c) is the R-54 PIVOT contribution.
- Open-source value stays high and is now concrete: three small upstream fixes (slime and PipelineRL read
  the engine spans; OpenRLHF records them).
- Publication potential: LOW. It was LOW to MEDIUM in R-54.
