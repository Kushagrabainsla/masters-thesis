# E-43 result: rollout version audit (I-165 premise check)

Date: 2026-10-28. Rules: `README.md` Section 5 (written before any source was read). Commits: `README.md`
Section 7. No code ran. Labels: FACT (source read at the pinned commit), INTERPRETATION, HYPOTHESIS.
Status: frozen decision applied. D-29 review R-55 (Sonnet): AMEND. The fixes are applied in this version
(Section 6).

## 1. Audit table

Field meanings are in `README.md` Section 4. F1: in-flight requests. F2: KV after the update. F3: version
record. F4: check. F5: KV provenance (secondary). One row for each shipped mode that touches the update path.

| Framework | Mode | F1 | F2 | F3 | F4 | F5 | Evidence |
|---|---|---|---|---|---|---|---|
| vLLM (engine) | `pause_generation(mode="abort")` (default) | abort | flushed | per-token | none | exact | [V1], [V2], [V5] |
| vLLM (engine) | `mode="wait"` | drain | flushed | per-token | none | exact | [V1], [V2], [V5] |
| vLLM (engine) | `mode="keep"`, `clear_cache=True` (default) | keep | recomputed | per-token | none | exact | [V1], [V2], [V3] |
| vLLM (engine) | `mode="keep"`, `clear_cache=False` | keep | kept | per-token | none | hybrid | [V2], [V4], [V5] |
| SGLang (engine) | `abort` (default) | abort | flushed (a) | per-token | none | exact | [S1], [S3], [S4] |
| SGLang (engine) | `retract` | keep | recomputed (a), (b) | per-token | none | exact (b) | [S1], [S3], [S5] |
| SGLang (engine) | `in_place` | keep | kept | per-token | none | hybrid | [S1], [S2], [S3] |
| verl | synchronous PPO (default) | none | flushed | per-step | none | exact | [R1] |
| verl | `one_step_off_policy` | none | flushed | per-step | none | exact | [R2] |
| verl | `fully_async_policy`, `partial_rollout=True` | abort-resubmit | flushed | per-trajectory span | bound | exact | [R1], [R3], [R4] |
| verl | v1 async trainers (separate, colocate) | abort-resubmit | flushed | per-trajectory span | bound | exact | [R3], [R5] |
| slime | stock synchronous rollout (default) | none | flushed | per-call list | none | exact | [L1], [L5] |
| slime | `--partial-rollout` | abort-resubmit | flushed | per-call list | none | exact | [L1], [L3], [L6] |
| slime | `--flush-cache-interval` other than 1 (fully async) | keep | kept | per-call list (last version) | none | hybrid | [L1], [L2], [L3], [L4] |
| AReaL | async rollout (default) | abort-resubmit | flushed | per-token | bound | exact | [A1], [A2], [A3] |
| OpenRLHF | `--train.async_enable` without partial rollout | none | flushed | none | none | exact | [O1] |
| OpenRLHF | `--train.partial_rollout_enable` | keep | recomputed | none | none | exact | [O1], [O2], [O3] |
| PipelineRL | in-flight weight update (default) | keep | kept | per-trajectory (start version) | bound | hybrid | [P1], [P2], [P3] |

(a) Only while the weight-update call keeps the default `flush_cache=True` ([S4]).
(b) Except in disaggregated PREFILL mode, where a live chunked request keeps stale-weight prefix KV ([S5]).

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
  A paused `keep` request does not count as unfinished, so the reset runs when the batch queue is empty
  ([`scheduler.py#L3090-L3092`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/core/sched/scheduler.py#L3090-L3092),
  [`core.py#L1656-L1662`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/v1/engine/core.py#L1656-L1662)).
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
  (merged 2026-10-11). vLLM 0.29.0 (pinned by OpenRLHF) has the label but no spans. vLLM 0.18.1 (pinned by
  PipelineRL) has neither (grep of `vllm/outputs.py` and `vllm/v1/engine/core.py` at both tags).
- [V6] The default `logprobs_mode` is `raw_logprobs`, which is before temperature and top-p
  ([`vllm/config/model.py#L291`](https://github.com/vllm-project/vllm/blob/7b87062308855434fa0747852ccebbe950ee9e4f/vllm/config/model.py#L291)).

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
- [S5] In disaggregated PREFILL mode, the pause keeps a live chunked request, and the code comment says "a
  weight-update pause leaves stale-weight prefix KV (off-policy)"
  ([`python/sglang/srt/managers/scheduler.py#L6010-L6016`](https://github.com/sgl-project/sglang/blob/88fea372d47b4cb14261d786e3824beb4fa97fa5/python/sglang/srt/managers/scheduler.py#L6010-L6016)).

**verl**
- [R1] Abort path: `pause_generation(mode="abort", clear_cache=reset_prefix_cache)` with
  `reset_prefix_cache=True` by default
  ([`verl/workers/rollout/vllm_rollout/vllm_async_server.py#L1111-L1199`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L1111-L1199)).
  The synchronous flush is `reset_prefix_cache(reset_connector=True)`
  ([`#L933-L996`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L933-L996);
  SGLang path `verl/workers/rollout/sglang_rollout/async_sglang_server.py#L484-L548`, reviewer-checked).
  Each response carries the server `global_steps`
  ([`#L733`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L733)).
- [R2] `one_step_off_policy` waits for the batch, then syncs weights, then starts the next batch
  ([`verl/experimental/one_step_off_policy/ray_trainer.py#L385-L408`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/experimental/one_step_off_policy/ray_trainer.py#L385-L408)).
- [R3] Partial rollout resubmits `prompt_ids + final_output.token_ids` after an abort and keeps
  `min_global_steps` and `max_global_steps` for the trajectory
  ([`verl/workers/rollout/llm_server.py#L282-L367`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/workers/rollout/llm_server.py#L282-L367)).
  Without partial rollout, aborted samples are dropped (`#L355-L359`). The v1 trainers always retry (`#L356`).
- [R4] Bound: `staleness_threshold` limits the share of stale samples
  ([`verl/experimental/fully_async_policy/fully_async_rollouter.py#L421`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/experimental/fully_async_policy/fully_async_rollouter.py#L421)).
- [R5] v1 trainers: the replay buffer drops finished groups whose prompt dispatch step is more than
  `max_off_policy_threshold` steps old (default 8, strategy `drop`)
  ([`verl/trainer/ppo/v1/replay_buffer.py#L520-L530`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/ppo/v1/replay_buffer.py#L520-L530),
  [`verl/trainer/config/ppo_trainer.yaml#L298-L303`](https://github.com/volcengine/verl/blob/2ab35292039a51175a4b3d3b3b2d76e951a0d3e4/verl/trainer/config/ppo_trainer.yaml#L298-L303)).
  The rule uses the dispatch step, not the recorded min and max versions, so F4 = bound.

**slime**
- [L1] All three weight transports call `pause_generation(mode="abort" if flush_cache else "in_place")`
  ([`slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L115`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py#L115),
  `update_weight_from_tensor.py#L299`, `update_weight_from_disk_delta.py#L196`).
- [L2] `should_flush_cache` flushes on the first sync and every N syncs; a value of 0 or less never flushes
  ([`slime/utils/weight_sync.py#L4`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/weight_sync.py#L4)).
  `--flush-cache-interval` defaults to 1. Its help text names the other values "PipelineRL" mode
  ([`slime/utils/arguments.py#L356-L365`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/arguments.py#L356-L365)).
  Other values select the fully async rollout (`arguments.py#L2266-L2275`).
- [L3] Version record: a sample appends `meta_info["weight_version"]` once for each generate call
  ([`slime/utils/types.py#L493-L494`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/types.py#L493-L494)).
  With [S3], that value is the last span. slime does not read `meta_info["weight_versions"]`.
- [L4] Staleness is a logged metric only
  ([`slime/utils/staleness.py#L20`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/utils/staleness.py#L20)).
- [L5] The stock loop generates a full rollout, trains, then updates weights, so nothing is in flight at the
  update ([`train.py#L51-L89`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/train.py#L51-L89)).
- [L6] Opt-in `--mask-offpolicy-in-partial-rollout` sets the loss mask to 0 for the old part of a resubmitted
  sample ([`slime/rollout/sglang_rollout.py#L296`](https://github.com/THUDM/slime/blob/540ea919e45b8b929198abce008a103be9cfc779/slime/rollout/sglang_rollout.py#L296)).
  It is a position rule, not a version comparison.

**AReaL**
- [A1] Weight updates send `abort_all_requests: True`, and SGLang flushes by default ([S4])
  ([`areal/engine/sglang_remote.py#L321`, `#L349`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/engine/sglang_remote.py#L321)).
  The colocate path aborts first, then pauses `in_place` with nothing left to keep
  ([`#L458-L478`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/engine/sglang_remote.py#L458-L478)).
  The vLLM path calls `pause_generation(wait_for_inflight_requests=False, clear_cache=True)`, which aborts
  (`areal/engine/vllm_ext/areal_vllm_server.py#L168-L226`, reviewer-checked).
- [A2] Each resubmitted segment pins its version before the call and tags every output token with it
  ([`areal/infra/remote_inf_engine.py#L1146-L1196`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/infra/remote_inf_engine.py#L1146-L1196),
  [`areal/api/io_struct.py#L69`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/api/io_struct.py#L69)).
- [A3] Bound: `StalenessManager` limits admission capacity by `max_head_offpolicyness`
  ([`areal/infra/staleness_manager.py#L79-L113`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/infra/staleness_manager.py#L79-L113)).
- [A4] Rollout logprobs are used by default (`recompute_logprob=False`,
  [`areal/api/cli_args.py#L2144`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/api/cli_args.py#L2144)).
  The loss reads the per-token versions to approximate the proximal logprob
  ([`areal/trainer/ppo/actor.py#L1211-L1250`](https://github.com/inclusionAI/AReaL/blob/fa8bd7895d97659b4d919687210f9123f4273139/areal/trainer/ppo/actor.py#L1211-L1250)).

**OpenRLHF**
- [O1] Without partial rollout, a lock keeps the weight broadcast apart from generation
  ([`openrlhf/trainer/ppo_trainer_async.py#L254-L264`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ppo_trainer_async.py#L254-L264)).
  No version field was found in the repository.
- [O2] With partial rollout, the engine calls `pause_generation(mode="keep")` with the default
  `clear_cache=True` ([`openrlhf/trainer/ray/vllm_engine.py#L131`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/trainer/ray/vllm_engine.py#L131)).
  The pinned vLLM 0.29.0 (`setup.py#L77`) resets caches in `_finish_pause`
  (`vllm@98dff2a` `vllm/v1/engine/core.py#L827-L847`) and preempts running requests on that reset
  (`vllm@98dff2a` `vllm/v1/core/sched/scheduler.py#L2604-L2624`). So F2 is recomputed.
- [O3] The flag help says "In-flight samples may contain tokens from both old and new weights"
  ([`openrlhf/cli/train_ppo_ray.py#L296-L302`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/openrlhf/cli/train_ppo_ray.py#L296-L302)).
  The README pairs partial rollout with `--algo.advantage.is_correction_level token`
  ([`README.md#L739`](https://github.com/OpenRLHF/OpenRLHF/blob/dc2a7ad326f619fc5a47737ea47d1920ce2c0b53/README.md#L739)).
  The CLI default is `off` (`train_ppo_ray.py#L261-L267`).

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
  PipelineRL PR #150 (merged 2026-07-08 into the `fast-llm` branch, not `main`) adds a per-token model
  version. Its author says the per-token path is not runtime-tested.
- [P3] Bound: `max_lag` limits submission. In the trainer, a too-old batch is counted, not dropped
  ([`pipelinerl/finetune_loop.py#L679-L684`](https://github.com/ServiceNow/PipelineRL/blob/58d393458625ad63ed539f2dcd072c85700c557f/pipelinerl/finetune_loop.py#L679-L684)).

**Upstream RFCs and PRs** (read 2026-10-28)
- vLLM #48311 (open, opened 2026-07-11): the pause, drain, sleep, wake and resume contract. It defines no
  version field. Its only version text is an exit criterion: "sleep/wake preserves or invalidates external
  state according to the weight version". It sends weight-update correctness to #48312.
- vLLM #54443 (open, opened 2026-08-30), "Weight version attribution for RL rollouts and Background
  survey". It specifies the per-token `weight_versions` spans. It states: "It does not say which weights
  computed the KV that the token attended to", and "It is not an off-policy correction; that stays on
  rollout logprobs". It reports that "prime-rl, SkyRL and PipelineRL use `keep` on their async update paths
  and retain KV by default", that SkyRL uses a version-keyed `cache_salt`, and that GLM-5 drops a response
  "when current − w0 > τ". The E-43 pre-read did not find this RFC. R-55 found it.
- AReaL PR #1722 (open, opt-in): masks loss targets whose token version is older than
  `max_head_offpolicyness`. If it merges, AReaL gets a per-token version filter. AReaL keeps no state, so it
  would not change Section 3.

## 3. Frozen decision

Rules are applied in the order of `README.md` Section 5.

1. **KILL (solved):** NOT MET. No framework has F4 = `check` in any mode. vLLM #48311 defines no version
   field. INTERPRETATION: the field exists upstream through #54443, #49040 and #59582. The rule names #48311,
   so the literal reading stands. A record is also not a check and not KV provenance, so "solved" is not met
   in spirit either.
2. **GO:** MET. Three frameworks keep state in a mode where F3 is not `per-token` and F4 is not `check`:
   - slime, `--flush-cache-interval` other than 1: keep, kept KV, one label for each call, no check.
   - OpenRLHF, `--train.partial_rollout_enable`: keep, recomputed KV, no version record, no check.
   - PipelineRL, default: keep, kept KV, start version for each rollout, bound only.

   vLLM and SGLang keep state but record per-token versions, so they do not count. verl and AReaL do not
   keep state (abort-resubmit with a flush).

**Frozen decision: GO.**

**Sensitivity (INTERPRETATION).** The GO has no margin. OpenRLHF is the only vote with exact KV. Without
it the count is 2, and rule 3 (PIVOT) would apply. Its clause "no paper measures the cost of exactness" was
not checked. By R-54 Section 7, the GO step is a 1-GPU fault-injection test that counts mixed-version
tokens. That step needs a predeclared plan and the user's approval before any GPU spend.

## 4. Scope limits on the GO (INTERPRETATION)

These points do not change the frozen decision. They change what a GO is worth.

1. **The version record exists upstream.** vLLM (label 2026-07-28, spans 2026-10-11) and SGLang (spans
   2026-08-24) already emit per-token version spans, specified in vLLM #54443. The three GO frameworks do not
   read them. slime can read them now. OpenRLHF and PipelineRL first need a vLLM upgrade, because their
   pinned versions have no spans ([V5]). So the record part of I-165 is integration work, not a research gap.
   R-54 named this threat. It happened before R-54 was written. R-54 did not find #54443.
2. **Labels mark sampling weights, not KV provenance.** #54443 says so directly. In the `hybrid` rows, a
   token labelled v2 can come from v2 weights over KV that v1 computed. No engine and no framework marks
   these tokens, and no code checks a label. This part is still open, but #54443 names it publicly.
3. **Rollout logprobs can make labels unnecessary for the loss** (post-hoc field, not predeclared). If the
   trainer uses the logprob that the engine recorded for each sampled token, importance sampling uses the
   real behavior distribution, also for a hybrid token. Three conditions limit this:
   - The logprob must be in processed mode. vLLM defaults to `raw_logprobs` ([V6]). PipelineRL
     (`conf/base.yaml#L65`) and verl (`rollout/rollout.yaml#L101`) set `processed_logprobs`.
   - Of the 3 GO frameworks, only PipelineRL uses rollout logprobs by default (`finetune/data.py#L150`). slime
     defaults to actor logprobs (`--use-rollout-logprobs` is off, `arguments.py#L1174`), but its usage guide
     shows `--use-rollout-logprobs` with the keep mode (`docs/en/get_started/usage.md#L444-L447`). OpenRLHF
     defaults IS correction to `off`, but its README pairs partial rollout with token-level IS ([O3]). So
     the documented recipes of all three use rollout logprobs. The defaults do not.
   - Labels still matter where the loss reads them. AReaL uses per-token versions in its proximal-logprob
     approximation ([A4]). Staleness filters (GLM-5, AReaL #1722, verl [R5]) also read versions.
   HYPOTHESIS: the in-flight mix adds training error only in off-recipe configurations (actor logprobs or
   raw logprobs). There it is one case of the general async-staleness error, not a new error class.
4. **Mislabels that a test can show** (HYPOTHESIS, from code reading). slime in its fully async mode labels a
   whole generate call with the last version, so tokens made before the update get the new label ([L3]).
   PipelineRL labels a rollout with its start version, so tokens made after the update get the old label
   ([P2]). Multi-turn PipelineRL rollouts get more mislabels, because the label is taken once per rollout.

## 5. What this means for I-165 (INTERPRETATION)

- The premise holds by the frozen rule, with no margin: three frameworks mix versions in shipped modes
  without a per-token record or a check.
- The research part is much smaller than R-54 assumed. The record ships in both engines. #54443 already
  surveys the frameworks and names KV provenance as out of scope. The documented recipes put the loss on
  rollout logprobs. What is left: (a) KV provenance for hybrid tokens, (b) a check that a label matches the
  weights and KV that made a token, (c) the throughput cost of exactness (recompute or abort) against keeping
  stale KV.
- Open-source value: MEDIUM. The upstream fixes are small and partly in progress (PipelineRL #150 on a
  branch, AReaL #1722 open). slime label handling and OpenRLHF version recording are open.
- Publication potential: LOW. It was LOW to MEDIUM in R-54.

## 6. Review R-55 and amendments (2026-10-28)

R-55 (Sonnet, D-29): AMEND. The GO stands. The main session checked the key reviewer claims in the source
before applying them (slime loop, verl v1 replay buffer, vLLM `logprobs_mode`, span history at both pinned
tags, SGLang disaggregated-prefill comment, AReaL defaults, slime and OpenRLHF recipes, #54443, PipelineRL
#150, AReaL #1722). Changes from the first version:
1. slime default row: F1 changed from `abort-resubmit` to `none`. A `--partial-rollout` row was added. F3 is
   now "per-call list".
2. Rows added: verl v1 async trainers, vLLM `wait`.
3. Citations fixed: [O2] preemption is at `scheduler.py#L2604-L2624`; the verl synchronous flush is at
   `vllm_async_server.py#L933-L996`.
4. #54443, #48312 and the #48311 exit-criterion text were added, plus the GO sensitivity.
5. Section 4 point 3 and Section 5 were rewritten: `logprobs_mode`, AReaL, the recipes, and "1 of 3 GO
   frameworks" uses rollout logprobs by default. The vLLM upgrade need was added.
6. SGLang `abort` and `retract` rows got the flush qualifier and the disaggregated-prefill exception.

Reviewer hostile rejection (kept as written): "The GO rests on one non-hybrid vote (OpenRLHF keep plus
recompute). The per-token record now ships in vLLM and SGLang, and vLLM #54443 explicitly leaves KV
provenance and off-policy correction out. The documented recipes of all three GO frameworks already train on
rollout logprobs, so version-exact lifecycle buys correctness only in off-recipe configs."
