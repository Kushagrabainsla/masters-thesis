# E-43: I-165 premise check (rollout version audit)

Status: PREDECLARED 2026-10-28. No framework source was read for this audit before this file was written.
Labels: FACT (source read at the pinned commit), INTERPRETATION, HYPOTHESIS.
Idea: I-165 (version-exact rollout lifecycle). Origin: R-54 Section 7. Decision request: D-66 (user approved
the audit on 2026-10-28).

## 1. Question

In current RL frameworks, what happens to in-flight requests and to their KV cache when the policy weights
change? Does the framework record which policy version made each token, and does it check that record?

Decision that this changes: I-165 GO (1-GPU fault-injection test, then the epoch-scheme design), PIVOT (a
cost-of-exactness measurement only) or KILL.

## 2. Rubber duck

**What I-165 claims.** An RL step trains on tokens and assumes a known behavior policy for each token. Some
frameworks pause the engine, load new weights and resume. Then a request can hold tokens from the old policy
and tokens from the new policy. A token can also come from new weights that read KV computed with old weights.
That token is from neither declared policy.

**What the claim needs.** (a) Shipped code paths that keep requests or KV across a weight update. (b) No
per-token record of the version, or no check of the record. If every framework aborts and recomputes, or
already tags and checks each token, I-165 has nothing to fix.

**Prediction (HYPOTHESIS, from memory, not from this audit).** vLLM and SGLang have pause modes that keep
requests and KV. PipelineRL mixes versions on purpose. AReaL records a version for each output token. verl and
OpenRLHF are synchronous by default. Expected result: borderline between GO and PIVOT.

## 3. Inputs

- Frameworks (7): vLLM, SGLang, verl, slime, AReaL, OpenRLHF, PipelineRL.
- Source: a shallow clone of the default branch of each repository on 2026-10-28. Section 7 records the SHA
  and the commit date. All file and line links use these commits.
- vLLM RFC #48311 (lifecycle contract): read the current text for a version field.
- No GPU. No code runs. This is a source audit only.

## 4. Fields (one table row for each framework and each shipped mode)

A "mode" is a shipped, documented setting or recipe in the repository (not a fork, not an open PR). The
default mode gets its own row. Record a file and line link for each value.

| Field | Values | Meaning |
|---|---|---|
| F1 In-flight requests at a weight update | `none` / `drain` / `abort` / `abort-resubmit` / `keep` | `none`: no request is in flight at the update (synchronous by construction). `drain`: the update waits for requests to finish. `abort`: requests are dropped. `abort-resubmit`: requests are stopped and sent again with the tokens made so far. `keep`: requests stay in the engine and continue after the update |
| F2 KV and prefix cache after the update | `flushed` / `recomputed` / `kept` | `kept`: decoding continues on KV that old weights computed |
| F3 Version record | `none` / `per-step` / `per-trajectory` / `per-request` / `per-token` | The smallest unit that carries a policy version ID in the data that goes to the trainer |
| F4 Check | `none` / `bound` / `check` | `bound`: the code limits staleness (for example a maximum version lag) but does not test each token or trajectory. `check`: code compares a recorded version against the expected version and rejects, asserts or filters on a mismatch, for each token or each trajectory |
| F5 KV provenance (secondary) | `exact` / `hybrid` | `hybrid`: a token can come from new weights over KV that old weights computed. Recorded for the write-up. F5 does not change the frozen decision |

## 5. Frozen decision rules (from R-54 Section 7, made exact here)

Count each framework one time. A framework "keeps state" if at least one of its modes has F1 = `keep` or
F2 = `kept`. Apply the rules in this order. The first rule that matches is the decision.

1. **KILL (solved).** At least 2 frameworks enforce a per-token or per-trajectory single-version invariant
   with F4 = `check` in the mode that keeps state. Or vLLM #48311 now defines a version field.
2. **GO.** At least 3 frameworks keep state in some mode where F3 is not `per-token` and F4 is not `check`.
3. **PIVOT.** At least 2 frameworks keep state and record versions (F3 is not `none`), and no framework or
   cited paper measures the throughput cost of exactness (abort or recompute against keep). Then the
   contribution is the cost measurement only.
4. **KILL (no premise).** No rule above matches. Then frameworks do not keep state across updates often
   enough for the invariant to matter.

These counts are project heuristics from R-54, not literature standards.

## 6. Procedure

1. Clone the 7 repositories. Record the SHA and the commit date in Section 7.
2. For each framework, find the weight-update entry point. Trace it to the engine pause, abort or drain call,
   to the cache flush or keep, and to the place that writes the sample for the trainer.
3. Find every shipped mode that changes this path (async, partial rollout, off-policy, in-flight update).
4. Fill one row for each mode in `audit.md`, with links.
5. Read vLLM #48311 and record if it has a version field.
6. Apply Section 5 and write the decision in `audit.md`.
7. Get a different-model review of the table and the decision (D-29) before the journal records a verdict.

## 7. Pinned commits

Cloned 2026-10-28 (shallow, default branch).

| Framework | Repository | SHA | Commit date |
|---|---|---|---|
| vLLM | vllm-project/vllm | `7b87062308855434fa0747852ccebbe950ee9e4f` | 2026-10-28 |
| SGLang | sgl-project/sglang | `88fea372d47b4cb14261d786e3824beb4fa97fa5` | 2026-10-28 |
| verl | volcengine/verl | `2ab35292039a51175a4b3d3b3b2d76e951a0d3e4` | 2026-10-28 |
| slime | THUDM/slime | `540ea919e45b8b929198abce008a103be9cfc779` | 2026-10-27 |
| AReaL | inclusionAI/AReaL | `fa8bd7895d97659b4d919687210f9123f4273139` | 2026-10-28 |
| OpenRLHF | OpenRLHF/OpenRLHF | `dc2a7ad326f619fc5a47737ea47d1920ce2c0b53` | 2026-09-17 |
| PipelineRL | ServiceNow/PipelineRL | `58d393458625ad63ed539f2dcd072c85700c557f` | 2026-06-19 |

## 8. Limits

- A source audit can miss a path. Each value needs a link, so a reviewer can check it.
- The audit reads code, not behavior. A `keep` mode can be rare in practice. Section 5 does not weight modes
  by use.
- Engines (vLLM, SGLang) supply the pause modes. RL frameworks choose among them. A framework row records the
  mode that the framework calls, not only what the engine offers.
