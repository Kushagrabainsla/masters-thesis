# E-28: metastability premise check for I-80

Status: run on 2026-10-08 (simulator version 2, amendments A1 and A2). Frozen decision: K1 KILL
(no collapse point). The post-run review is R-16. The journal entry E-28 is the record.

## Files

| File | Contains |
|---|---|
| `extract_sessions.py` | Reads the Exgentic agent traces and writes one row for each LLM call |
| `data/session_calls.csv` | 460 sessions, 18,239 calls: input tokens, output tokens, call time, gap |
| `sim.py` | Discrete-event simulator of one serving replica and agent clients |
| `test_sim.py` | Unit tests: KV accounting invariant, cache hit, eviction, timeout and retry, cap policy |
| `run_e28.py` | Runs each grid point with a spike and without a spike (control), and writes window metrics |

## Rubber duck before the run

1. **What the simulator does.** It models one GPU that serves agent sessions. Each session is a
   chain of LLM calls taken from a real trace. Between calls the agent runs a tool. The GPU keeps
   the KV of an idle session in a prefix cache, so the next call computes only the new tokens.
2. **Why a cache miss is expensive.** In the traces, the median call has about 25,000 input tokens
   and about 100 output tokens. A call adds only a small number of new tokens. A hit computes the new
   tokens. A miss computes the full prompt again. With the default cost model, a miss on a 21k
   private prompt adds about 1.1 s of GPU time. A hit turn costs about 0.3 GPU-s at the median
   output and 0.5 to 0.6 GPU-s at the mean output. So a miss costs about 3 to 4 times a hit.
   (Corrected after R-15 m6. The first draft said 10 times; journal Section 8.1 row 45.)
3. **The loop that can make a failure last.** A spike adds sessions. More live sessions need more KV
   than the GPU has. The cache evicts idle sessions, so more turns miss. Misses add GPU work, so turns
   take longer. Sessions then live longer, so more sessions are live at the same time (Little's law).
   More live sessions need more KV. If this loop gains more than it loses, the system stays in the
   bad state after the spike ends. Client timeouts and retries can add work to the loop.
4. **What a "lasting collapse" means.** The arrival rate after the spike is the same as before it. A
   control run with no spike, the same seed and the same rate stays healthy. If the spike run still
   has much lower goodput than the control run long after the spike ends, the state is metastable.
5. **Expected result (HYPOTHESIS).** A collapse is likely when the KV capacity holds only a small
   number of session working sets and the base load is close to that limit. A fixed session cap may
   remove the collapse. If it does, the I-80 mechanism has no room over a simple baseline.
6. **What can make the result wrong.**
   - The cost model is a roofline estimate, not a measurement. Wrong constants move the knee.
   - Tool gaps come from four harnesses only. Claude Code spans have no duration, so their gaps are
     not used.
   - The model has one replica. A real deployment has a router and many replicas.
   - The model has no CPU or SSD KV tier. A second tier (LMCache, HiCache) makes a miss cheaper and
     can remove the loop.
   - Sessions in the trace use hosted models. Their prompt lengths can differ for an 8B model.

## Rubber duck after the run (2026-10-08)

1. **What ran.** Stage 0a (880 runs), stage 0b (2,400), stage 1 (4,500), stage 2 (11,000) and stage 3
   (1,800). Total wall time about 1 hour 35 minutes on 9 cores. The frozen file checksums matched
   before the decision (`results/FROZEN_2026-10-08.sha256`).
2. **Calibration (stage 0a).** Goodput rises with the session cap until the KV holds the working set,
   then falls by about half (for example M1 at 380k: 1.12 turns/s at cap 12, 0.55 at cap 32). This is
   the thrashing curve. The best cap N* is 8 to 48 sessions.
3. **Knee (stage 0b).** The uncapped system stays healthy up to 0.55 to 0.85 of G* (base mixes). Above
   the knee, it loses goodput with no spike at all.
4. **The decision (stage 1).** No point has a spike run below 0.7 of its paired control in both
   windows. The lowest ratio over all 300 spike points is 0.89 (late) and 0.91 (post). The verdict
   is K1 KILL. K2 and K3 do not apply.
5. **The trigger worked.** The spike pushed live sessions far above N* (for example 22 against 12, and
   56 against 48), and preemptions rose sharply (for example from 2 to 24, and from 0 to 204, over
   the 660 s after the spike start). Goodput in the 10 minutes after the spike was
   higher than in the control, because the extra sessions were served. Then the system returned to the
   control level.
6. **Why no lasting state (INTERPRETATION, CORRECTED by R-16).** The first explanation said that the
   bistable band has no measurable width and that the state above the knee is plain overload. That is
   wrong (journal Section 8.1 row 46). The R-16 reviewer check shows a lasting, bimodal bad state at
   1.0 x knee after a spike of 2 x N* or more (client C), and controls above the knee that leave the
   good state by noise while a cap still carries the load. E-28 used spikes of at most 1 x N* and a
   goodput rule with no power at a 20% loss, so it could not see the band.
7. **Noise in the controls.** 28 of 100 client-A controls failed the 0.9 stability filter. The cause is
   arrival noise: about 9 sessions arrive in 600 s, and the late window got 7.2 on average against 8.8
   expected. The spike runs share the same arrivals, so the ratios in point 4 are not affected.
8. **Sensitivity (stage 3).** Outputs clipped at 1,024, prefill x0.5 and x2, decode x0.5 and x2, and
   tool gaps x0.25 all give no collapse (lowest ratio 0.88, post window, prefill x0.5).
9. **What this does not show.** It does not show that a cap is useless. A static cap clears the
   trapped state (R-16 cap check). That fix is Denning working-set control and CONCUR. The model has one replica, no CPU KV tier, and hosted-model token counts.
