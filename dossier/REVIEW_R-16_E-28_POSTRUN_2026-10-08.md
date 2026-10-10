# R-16: Adversarial post-run review of E-28 (I-80 metastability check)

Date: 2026-10-08. Reviewer: Sonnet 5.5 (different-model review). Status: review only. No repo file was edited.
Labels: FACT = I read it or computed it. INTERPRETATION = my reasoning from facts.
New runs are labelled "reviewer check". They are not part of E-28 and are not frozen. Scripts and outputs: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/56d042b4-71c1-432b-9e52-b83a6c684a44/scratchpad/r16/` (`hyst.py`, `hyst.csv`, `an.py`, `cap.py`).

## 1. Verdict

| Claim | Verdict |
|---|---|
| The frozen script returns K1 KILL on the E-28 grid | CONFIRMED (I recomputed it). |
| "No lasting collapse exists; the bistable band has no measurable width" | NOT CONFIRMED. A reviewer check finds a lasting bad state in the same simulator. |
| "Above the knee the system degrades by plain overload" | NOT CONFIRMED. The offered load is below the good-state capacity. The system leaves the good state by random fluctuation. This is metastability. |
| I-80 is dead as a thesis direction | CONFIRMED WITH SCOPE LIMITS. The reason changes. The phenomenon exists, but the fix is a known static cap, and the retry story is wrong. |

Short form: K1 is a correct result for a grid that was too narrow. K1 does not show "no phenomenon".

## 2. Recomputed numbers (FACT)

- Stage 1 file has 4,500 rows: 3,000 spike runs and 1,500 controls. There are 300 spike points (mix, cache, load, spike size, client), 10 seeds each.
- Lowest mean spike/control ratio: 0.893 late (M3, 380k, load 1.0, spike 1.0, client C) and 0.912 post (M4, same cell family). The main session is right.
- The decision file says: collapse points A=0, B=0, C=0. Stable controls: A 72, B 72, C 66 of 100 points each. So 28 of 100 client-A controls fail the stability filter. This matches the report.
- Pairing check: late arrivals differ by 0.0 between every spike run and its control. Pre-window goodput differs by 0.0. So the spike runs share all arrivals outside the spike. The pairing is correct.
- The spike happens. For M1, 380k, load 1.0, spike 1.0, client A, the base rate is 0.0245 sessions/s and the spike multiplier is 9.2. The spike adds about 12 sessions in 60 s.
- Worst single cell by seed (median of 10 paired seeds): 0.77 (M5, 380k, load 1.0, spike 1.0, client C). The 7-of-10 rule fails there only because the mean ratio is above 0.7.
- The lowest ratios sit at the top of the grid (load 1.0 x knee, spike 1.0 x N*, client C). That is the edge nearest the barrier. See Section 4.

I found no code bug that hides a collapse. I checked: spike timing, windows, pairing, goodput count, retry re-send (a retry re-submits the full turn), cache eviction (tail-first LRU, partial hits), and the gap pool (mean 6.9 s, median 2.1 s, so a session lives about 4 minutes in the good state). The model is coherent. The problem is the design of the grid, not a bug.

## 3. The calibration already predicts a bistable band (FACT, from `e28_calibrate.csv`)

M1, 380k cache, saturating arrivals, 5 seeds:

| Session cap | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|
| Goodput (turns/s) | 0.83 | 1.12 | 1.07 | 0.65 | 0.55 | 0.54 | 0.56 |
| Hit share of prompt tokens | 0.96 | 0.94 | 0.80 | 0.47 | 0.39 | 0.39 | 0.38 |

INTERPRETATION: the good state carries about 1.12 turns/s. The bad state carries about 0.55 turns/s. Any offered load between 0.55 and 1.12 can sit in either state. This is the R-14 shape. For 380k, the stage 1 loads were 0.44 to 0.73 turns/s. Only the top two loads (0.66 and 0.73) are inside the band. The knee rule removed the upper part of the band from the grid (see Section 4).

## 4. Reviewer check: hysteresis test (reviewer check, not E-28)

Setup: M1, 380k, same simulator and parameters. Horizon 12,000 s (E-28 used 6,000 s). Offered load 0.73, 0.85, 1.0, 1.1 turns/s. Load 0.73 is exactly 1.0 x knee. Spike of 0, 1, 2, 4, 8 x N* extra sessions in 60 s (N* = 12; E-28 used at most 1 x N*). Clients A (60 s, 2 retries) and C (600 s, 2 retries). 6 seeds. 240 runs, about 6 minutes on 8 cores.

Client C at load 0.73 (mean of 6 seeds; "live" = live sessions at the end, t = 12,000 s):

| Spike | Live at 3,000 s | Live at end | Late goodput | Late preempts per s | Late / control | Seeds below 0.7 |
|---|---|---|---|---|---|---|
| 0 (control) | 8 | 5 | 0.79 | 0.00 | 1.00 | 0 of 6 |
| 1 x N* | 14 | 5 | 0.79 | 0.00 | 1.00 | 0 |
| 2 x N* | 23 | 22 | 0.76 | 0.12 | 0.97 | 0 |
| 4 x N* | 54 | 72 | 0.63 | 0.36 | 0.80 | 2 |
| 8 x N* | 104 | 112 | 0.62 | 0.36 | 0.79 | 2 |

Per seed (FACT): with a 2 x N* spike, 2 of 6 seeds stay in the bad state for 9,000 s (live 36 and 72 at the end, preempt rate about 0.37). The other 4 seeds recover (live 4 to 6). With a 4 x N* spike, 6 of 6 seeds stay bad (live 49 to 91). With an 8 x N* spike, 6 of 6 stay bad. All 6 controls end with 4 to 6 live sessions and zero preemptions.

FACT: this is a hysteresis. The same load gives a good state without a spike and a bad state after a big enough spike. The bad state lasts at least 9,000 s, which is about 35 session lifetimes. The state variable is bimodal: preempts per second are 0.00 or about 0.36, with nothing between.

INTERPRETATION: the barrier is between 14 and about 35 live sessions, near 2 x N*. The E-28 spike (up to 1 x N*) pushed live sessions to about 22 (README point 5) when the base was about 8 to 10. That is just under the barrier at the top load. The E-28 grid stopped one step short of the barrier. This also explains why the lowest ratios sit at the top corner.

Spontaneous transitions (FACT, controls with no spike, client C):

| Load | Controls that left the good state | Live at 6,000 / 9,000 / 12,000 s (mean) |
|---|---|---|
| 0.73 | 0 of 6 | 7 / 9 / 5 |
| 0.85 | 1 of 6 by 12,000 s (seed 6: live 8, 20, 37, 72); seed 5 had a transient at 22 | 10 / 16 / 18 |
| 1.0 | 5 of 6 (live 62, 22, 51, 31 at 6,000 s, then 78 to 120) | 32 / 60 / 84 |
| 1.1 | all by 12,000 s | 40 / 78 / 124 |

INTERPRETATION: the offered load at 1.0 (turns/s) is below the good-state capacity of 1.12, and a cap of 12 carries it (cap check: goodput 1.07, 17 live at the end, preempt 0.001). So this is not plain overload. The good state is locally stable. Random arrival clustering pushes the system over the barrier, and it stays there. The waiting time is long (hundreds to thousands of seconds), and the 6,000 s horizon cuts it off. The journal text "above the knee the uncapped system degrades without a spike (plain thrashing)" and README point 6 are wrong in kind. They describe a noise-triggered metastable transition as overload.

Client A (60 s timeout): no hysteresis at any spike size, up to 8 x N* (live back to 5 at the end). The cost is lost sessions. At load 0.73, a 4 x N* spike fails 24 of 48 extra sessions, and an 8 x N* spike fails 76 of about 96 (reviewer check, `cap.py`). INTERPRETATION: a short timeout with a retry limit acts as a load shedder. It is the opposite of the I-80 retry story. The goodput metric does not count failed sessions, so it hides this cost.

Cap check (reviewer check, 6 seeds, client C, load 0.73, 4 x N* spike): a static cap of 12 gives late goodput 0.78, 0 preempts, 5 live at the end. A cap of 16 gives 0.79 and the same. The uncapped run gives 0.63, 0.36, and 72 live. So a static cap sized from N* clears the trapped state in this slice.

## 5. Why the frozen rule still said KILL (FACT and INTERPRETATION)

1. Spike size. Stage 1 used 0.5 and 1.0 x N*. The barrier is near 2 x N* at the top load. Both sizes recover.
2. Load range. The knee rule keeps loads where the uncapped control is stable on average over 10 seeds. At and above the knee, controls start to fall by themselves, so the grid stops before the band is wide.
3. Horizon. 6,000 s with a spike at 2,400 s. The "late" window is 3,000 s after the spike. A trapped run needs only to stay bad for that long, so this was fine. But spontaneous transitions need more time, and the control windows cannot see them.
4. Effect size. In the hysteresis check at load 0.73, the bad state loses only 20 percent goodput (0.63 against 0.79). The load is only 0.65 x G*, and the bad-state capacity (0.55 to 0.65) is close to the load. The rule needs below 0.7 in at least 7 of 10 seeds. A 4 x N* spike gives a mean ratio of 0.80 and 2 of 6 seeds below 0.7. So even a trapped state would not pass the frozen collapse rule at this load. INTERPRETATION: the threshold 0.7 came from the R-14 toy, where the load was higher in the band and the loss was larger.
5. Metric power. With 5 to 10 live sessions, window goodput is noisy. The same load 0.73 gave pre-window goodput from 0.37 to 1.12 across seeds in the control. A bimodal state signal (preempt rate, or live-session count) is clean. The goodput ratio is weak. The frozen rule did not use the clean signal.
6. Stability filter. 28 of 100 controls failed it because of arrival noise, as the main session says. This shrinks the tested set but does not bias the ratio.

## 6. Why the R-14 toy found a collapse window and E-28 did not (INTERPRETATION)

Model differences that matter:
1. Distance to the barrier. The R-14 toy ran at 75 percent of good capacity with small spikes (2x for 30 s adds about 3 sessions). The band was 1.0 to 1.8 turns/s of a 1.8 capacity. The toy sat close to its barrier. E-28 sat at most at 0.65 x G* with the live base about 8 and the barrier near 22 to 25, then used a spike that reaches about 22. Same shape, different distance.
2. No paired control in the toy. The toy compared the tail with the pre-spike window. Above 1.0 turns/s its own note says "no spike is needed". Part of the toy "spike effect" can be a spontaneous transition. My Section 4 shows these are common near the band top. E-28 pairing removes this error, but the grid then missed the barrier.
3. Workload. The toy used 15 turns, 8k to 20k context, no shared prefix. E-28 uses real traces: about 30 turns, 25k median prompt, shared prefixes of 800 to 14,000 tokens. N* is 12 in E-28 and about 19 in the toy.
4. Bad-state depth. Toy: bad about 0.8 to 1.0 against good 1.8 (loss 45 to 55 percent). E-28: bad 0.55 against good 1.12 (about 50 percent) at saturation, but only 20 percent loss at the tested load, because the load is far below the good capacity.
5. Seeds and horizon. Toy: 4 seeds, short horizon, 30 s windows. E-28: 10 seeds, 600 s windows.

Closer to reality: E-28, for the workload (real token counts, vLLM v1 scheduler order, shared prefix). Neither model has measured GPU constants. E-28 is also more honest about the control. But the toy better located the bistable band, and the E-28 grid did not reach it.

## 7. Other findings

- F1 (design). The spike size was tied to N*, not to the distance between the base live count and the barrier. A spike size defined as "reach 2 to 3 x N* live sessions" would have found the trap.
- F2 (K2, K3 not run). K3 (does a fixed cap tuned once on M1 work for all mixes?) never ran because K1 stopped the chain. The calibration shows N* is 8 to 48 across mixes and caches (M1 380k: 12; M4 380k: 32; M5 190k: 48). A cap of 12 tuned on M1 would cut M4 capacity by a large factor. This is the only open question with data behind it, and E-28 left it unmeasured.
- F3 (knee rule). The knee uses a mean over 10 seeds. At load 0.85 only 1 of 6 controls was trapped by 6,000 s, so the mean looks stable. Bimodal outcomes need a per-seed test.
- F4 (retries). The R-14 and journal claim "retries are not needed" holds. But client A (with retries and a short timeout) is better than C. Retries and timeouts help the system by shedding sessions. K2 (kill the retry component) is right for a stronger reason.
- F5 (wording). The journal and README call the above-knee state "plain thrashing". Replace with "noise-triggered transition from a locally stable state".

## 8. Scope limits of my own checks

- One mix (M1), one cache (380k), 6 seeds, one run of 12,000 s. Not frozen, no criteria written first.
- The 2 x N* spike is a 17-fold surge in the arrival rate for 60 s. A 4 x N* spike is a 33-fold surge. These are large. Real triggers of this size are a replica failover with all sessions reconnecting, or a cache flush. A slow period that holds live sessions above the barrier has the same effect. I did not model them.
- All constants are unmeasured. The location of the barrier depends on the miss cost. A real GPU, a CPU KV tier, or multiple replicas can move or remove it.
- The cap check used 6 seeds in one cell.

## 9. What remains for a mechanism-transfer thesis (INTERPRETATION)

- A cap above the knee is Denning working-set control and CONCUR. The cap check confirms it clears the state. No novelty there.
- The phenomenon is real in the simulator, so the "premise check" part of I-80 passes with scope limits. But the premise plus a known fix is a measurement, not a mechanism.
- Open and cheap to test: one state signal. The bad state is bimodal in preempt rate (0 against 0.36 per turn) and in live-session count. A controller could use this signal to enter a drain mode (shed or defer new sessions) only when the signal trips, without knowing N* in advance. The test is: does it match the per-cell best cap within 10 percent across the mixes where N* varies 8 to 48, while a cap tuned once on M1 loses more than 20 percent? This is K3 as predeclared. It is CONCUR's adaptive idea in an online setting. I rate novelty as thin, and I do not recommend it as a thesis alone.
- The metastability framing for LLM serving (hysteresis, spontaneous transitions, a long waiting time) is a measurement result with a clean signal. R-14 found no prior paper on it. It may support a short paper. It does not support a mechanism-transfer thesis with a new mechanism.

## 10. Recommended journal changes (for the main session; I did not edit)

1. Record K1 KILL as correct for the frozen grid, with the scope limit "spike at most 1 x N*, load at most 1.0 x knee, horizon 6,000 s".
2. Add the reviewer check as R-16 with the table in Section 4. Mark it unfrozen.
3. Correct the interpretation: "bistable band has no measurable width" is retracted (journal Section 8). The band exists. The barrier is near 2 x N* live sessions at 1.0 x knee.
4. If the user wants a follow-up, predeclare a small E-29: spikes of 2, 3, 4 x N*, loads 0.9 to 1.2 x knee, 12,000 s, 10 seeds, a state signal (preempt rate above 0.1 per turn at the end) as the collapse test, and K3 (fixed cap against per-mix oracle) run regardless of K1. Kill criteria must be written before the run.
