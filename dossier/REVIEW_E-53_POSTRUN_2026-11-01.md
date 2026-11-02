# R-74: post-run review of E-53 (accuracy of the compact FP8 K=V global cache; gate G3 of D-88)

VERDICT: CONFIRM_WITH_CORRECTIONS (the frozen computation and the FAIL are correct; the reading of the cause needs 9 corrections, C1 to C9; D-90 stays open)
No correction changes the frozen decision. C1 to C6 and C9 change the reading of the result. C7 and C8 change the record and the next run.

Date: 2026-11-01. Reviewer: R-74 (Sonnet 5.5, a different model from the author). Stage: S9 hostile post-run review of the run `e53-h100-20261101` (design commit d72f53f, result commit d996fd4).
Labels: FACT means that I read a file or source, ran code or computed a number. INTERPRETATION means my reasoning. HYPOTHESIS means not tested. POST-HOC means computed after the frozen decision, by the author or by me.
Rules kept: I changed no repo file except this one and created no folder in the repo. I did not read `.env`. I made no Runpod, cloud or GPU call. I posted nothing and did not commit. I made read-only `gh api` GET calls to public vllm-project repos.
Deviation: a first R-74 attempt stopped at an app restart. I re-read every script of that attempt that I use. I re-ran or re-computed each of its numbers that appears here.
Scratch (second attempt): `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/20255ed9-038a-44e5-8a60-f7885a8c54ee/scratchpad/r74/` with `indep2.py` (with `indep2_lib.py`), `phase_check.py`, `rescue_check.py`, `sink_check.py`, `probe_mask_check.py`, `phase_analysis.py`, `family_cv.py`, `null_mc.py`, `final_checks.py`. First attempt: `.../31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/r74/` (`phase.py`, `flips.py`, `boot.py`, `v5neg.py`, `mech2.py`).

## Key findings
1. FACT: My fresh recompute matches the author in all 40 (prompt, layer) cells to 4e-6 relative. It uses its own E4M3 table, float64 attention and an explicit 640-value row. The decision is the same: FAIL by layer 41 only (per-prompt ratios 2.685, 2.109, 2.419, 2.348; median 2.384).
2. FACT: V1 to V5 pass, and no check passes by mistake in this run. The SHA-256 of all 41 copied files equal `capture_sha.txt`. `analyze_e53.py`, `capture_e53.py`, `session_e53.sh` and `compact_attn.py` did not change after commit 61e4d37 (A2, before the data). The 21 tests pass (76 s).
3. FACT: The FAIL is stable over the query rows and the prompts. The row-bootstrap interval of each layer-41 prompt ratio is within 0.06 of its value. All 4 prompts exceed 2 (the lowest is 2.11). INTERPRETATION: a new capture of the same kind at scale 1.0 gives the same FAIL.
4. FACT, POST-HOC: Four unrotated dims decide layer 41: 70, 113, 219 and 506, in every prompt. With K exact on these 4 dims in both storages, the ratio falls from 2.66 to 1.01 (prompt 0) and from 2.09 to 1.02 (prompt 1). Four random dims change nothing (2.67 and 2.09).
5. FACT, POST-HOC: The ratios below 1 in layers 5, 23 and 59 are one-token effects. Position 0 (the BOS sink) has identical K and V in all 4 prompts (I checked layers 5, 23 and 41; the author checked K in all 10). With token 0 exact, the layer 23 ratio goes from 0.56 and 0.43 to 1.62 and 1.32 (prompts 0 and 1).
6. FACT, POST-HOC: The standard FP8 error is itself a lottery. In layer 41 a K scale shift of 1/16 octave (4%, V scale fixed) moves the standard error by 2.44 times (median over prompts). In that layer the stock scale is the second best of 16 K phases. The log standard deviation of this effect is 0.32 over layers and phases.
7. FACT, POST-HOC: Take an equivalent storage (a standard FP8 cache at another K grid phase, one random phase per layer). The frozen rule gives FAIL in 25% of draws. The PASS median condition (all 10 medians at most 1.25) holds in only 8%. The E-53 pooled RMS ratio over the 10 layers is 1.105, the 79th percentile of that null. The E5M2 control has a pooled ratio of 3.34.
8. FACT, POST-HOC: Choose the tied scale of each layer on 3 prompts and test it on the 4th (leave-one-prompt-out). The tied compact row then has a per-layer median of 0.35 to 1.12 times the stock error, and 0.72 times pooled. This uses the same 4 prompts and the same model.
9. FACT (source): The pinned FA4 SM90 FP8 kernel takes separate K and V descales per (sequence, KV head). It computes in FP16. I found no run (E-51 to E-53) that uses a K descale different from the V descale on two views of one row. E-52 used ones.
10. FACT: The small result files of the run (`metrics.json`, `summary.json`, `decision.txt`, `capture_sha.txt`) are not in git (C7).

## Corrections
C1. The tie "gives the standard FP8 codes up to the BF16 mismatch" (README Section 13, item 6, INTERPRETATION).
- FACT: At tied scales, 1.6% to 4.7% of the unrotated codes differ (prompt 0; 0.0% in layer 29).
- FACT, POST-HOC: Tied compact over tied standard has a per-layer median of 0.996 to 1.010 at vs = 1. The per-cell log standard deviation is 0.064. The range over 640 cells is 0.77 to 1.54. I recomputed 8 extreme cells with explicit descale arithmetic. All 8 match to 4 digits.
- Change: write "equal in the median, not per cell; the flips act as noise of about 6% with tails of +54% and -23%". Fix the E-54 tie bounds on this (D-90).

C2. The "HYPOTHESIS: large channels in layer 41" and the sink text (README Section 13, last INTERPRETATION and item 4).
- Change: write FACT for the 4 channels (key finding 4). They have a rms |q| of 2.1 to 2.8 (layer mean 1.07) and a rms V of 1.9 to 3.2 (layer mean 1.0).
- Add: in layers 5 and 23, token 0 holds most of the standard error (layer 23: 0.0731 to 0.0202; layer 5: 0.0438 to 0.0149 with token 0 exact). So the PASS side of the table is one vector, shared by all prompts.

C3. The mirror row and the screen (README Section 13, items 5 and 6).
- FACT: The mirror row [K | V_unrot-from-K | V_rot] at scale 1.0 needs two V scales: 1/w for V_unrot and 1 for V_rot. One `v_descale` cannot do that. The screen M0 uses fp8(V_rot) at scale 1, so it is not a stock-kernel layout.
- FACT: The screen's MA/SA (0.97 to 1.02) tests the mirror, not the compact row. The mirror scores equal the standard scores by construction. The screen has no tied compact row.
- Change: say so. Cite the phase scan (C_T against S_T, C1) as the direct test of the scale-tied compact row.

C4. "The FAIL is an effect of the scale ratio" (README Section 13, item 6).
- INTERPRETATION: Correct, but one-sided. The stock baseline is a lucky draw in layer 41 (rank 2 of 16 phases). The standard with K scale w is as bad as the compact: layer 41 prompt 0 gives 0.0814 (tied standard), 0.0752 (tied compact) and 0.0282 (stock).
- Change: add the null of key finding 7 under Section 7. Keep the FAIL as logged. Do not retract it.
- Also: the E5M2 control gave 2.0 to 6.0 per layer, so a per-layer ratio of 2.38 cannot tell a lost mantissa bit from a phase draw. The pooled ratio can (1.105 against 3.34).

C5. The engine compute dtype (README Section 10, threat 1; E-52 notes).
- FACT: The pinned `vllm-project/flash-attention` commit b273a02 (2026-10-07; `vllm_flash_attn.cmake` at 3966924d5) computes in FP16 and converts FP8 to FP16 without a scale. It folds q and k descales into the score scale and the v descale into the final normalization. This matches the E-53 FP16 emulation.
- FACT: The `main` branch of that repo changed the SM90 path to FP8 tensor-core products. Q is split in two E4M3 terms at fixed scale 1 ("keep |q| <= 448", "fold larger Q into k_descale"). P is split in two terms (scales 256 and 32). Its error differs from E-53.
- Change: record the commit. Say that E-53 matches the pinned kernel only.

C6. The probes have no causal mask: `exploratory/e53_score_probe.py` (`p = softmax(s_ref)`) and `exploratory/e53_sink_probe.py` (`mass0`; the comment "all keys are visible: qpos >= 7,936" is wrong).
- FACT: With the mask (prompt 0; layers 41, 23 and 5 only): mass0 0.032 to 0.037 (layer 41), 0.139 to 0.151 (layer 23), 0.200 to 0.206 (layer 5); std_un 0.098 to 0.102 (layer 41); cmp_un 0.146 to 0.153. The changes are 0.3% to 14%. No ranking changes in these 3 layers.
- Change: add the mask and rerun, or note the bias. The sink shares in README item 4 ("13% to 19%") are about 1 to 2 points too low in layers 5 to 23.

C7. The run record is not in git. `git status` shows `experiments/gemma4_kv_accuracy/results/` as untracked. Commit 42285f7 holds no file of `results/e53-h100-20261101/`. The README and the folder `.gitignore` say that small result files stay in git. Other experiments do track their small result files (261 tracked files under `results/`).
- Change: add every file of that folder except `capture/` (the author or the user commits; I did not).

C8. The V5 bound. 0.05 is 29 times the observed maximum (0.0017) and as large as the FP8 errors themselves.
- FACT: In my controls (layer 41, prompt 0) a wrong setup gives a gap of 0.22 to 1.17. The cases: interleaved heads 0.99; KV heads rolled 1.17; no mask 0.31; scale 1/sqrt(512) 1.15; query positions shifted by one 0.22.
- Change: use 0.005 for E-54 and keep these controls as tests. V1 accepts any name with "H100" and "80GB". This is harmless here.

C9. "A power-of-2 scale does not move a value on the FP8 grid" (README Section 13, item 1) holds for normal values only.
- FACT: In layer 29 at scale 1.0, 14.35% of the unrotated values still differ between fp8(K) and w x fp8(V). All of them have |K| below 2^-6 (the K subnormal range, 22% of the values). None is above. The ratio 1.007 holds because these small values carry little error.
- Add FACT: w x 16 = 0.961 to 1.047 in all 10 layers (w is one BF16 value on all 512 dims). The K grid sits at most 0.066 octave from the V grid.

## Answers to the checks

### 1. Independent recompute of the frozen decision
- FACT: I computed all 10 layers and 4 prompts. Layers 41, 29 and 5 are in the set. The per-layer medians are 0.9298, 1.2301, 1.1910, 0.5664, 1.0075, 1.1885, 2.3838, 1.3335, 0.9832 and 0.8182. They equal the README table.
- FACT: My E4M3 is a look-up table of the 127 non-negative codes with ties to the even code. It matches `torch.float8_e4m3fn` on every K and V value of the 40 files (0 mismatches).
- FACT: My attention is float64 with an additive causal bias. Query head h reads KV head h // 8. The scale is 1.0 and there is no soft-cap.
- FACT: My compact row is built by index lists. The K view is values 0 to 511. The V view is values 128 to 639. The output is unpermuted by an inverse index.
- FACT: Saturated values: 0. Largest |q'|: 19.25. Identity error: 0.0023 to 0.0024. Engine gap: at most 0.00172. FAIL layers: 41 only.
- FACT: A first-attempt recompute in float32 (a third code path) also matches. The bootstrap over the 256 query rows gives intervals of 0.02 to 0.12 width (layers 5, 11, 23, 41, 47). The 4 KV heads of one layer have ratios from 0.26 to 4.96. So the layer ratio mixes large head effects.

### 2. V1 to V5
- FACT: V1 to V4 pass. The 21 tests include negative controls for V2, V3 and V5 (README Section 12). V3 uses the module weight in the "weight" form. The 10 weights are one BF16 value each. The identity error is 0.0024 (bound 0.01).
- FACT: V5 ties the reference to the engine (FA4 BF16, `Using FlashAttention version 4` in `capture_log.txt`). It catches every wrong mapping that I tried (C8).
- INTERPRETATION: No check passes by mistake in this run. The V5 bound is loose (C8): an FP8 engine cache could pass it in some layers. V2 forbids that case (`cache_dtype` auto).
- FACT: The decision is the frozen rule applied to correct numbers. The rule needs no change to give FAIL.

### 3. Faithfulness to vLLM 3966924d5
- FACT: K is cached after `k_norm` and RoPE, and V after the weightless `v_norm` (`gemma4.py` 540 to 575). The default `_k_scale` and `_v_scale` are 1.0 (`attention.py` 136 to 153). `kv_cache_scheme` is null (E-50 pre-run review record, `dossier/REVIEW_E-50_PRERUN_2026-10-30.md`; I did not re-read the checkpoint). The cast is `scaled_convert` (R-73).
- FACT: The pinned kernel computes in FP16 with raw FP8 values (C5). So the E-53 FP16 emulation models it. Its ratios equal the float32 ratios to 1e-4.
- FACT: A real compact engine writes the same values at scale 1.0. K_rot comes from the K path, and V from `v_norm`.
- FACT: The reference does not favor the standard storage in a material way. The compact path carries the BF16 mismatch (output error 0 to 0.0105 without FP8, maximum in layer 47). It also carries the BF16 rounding of q' (up to 3.7% change of one cell's error). I recomputed the ratio with an ideal reference (K_unrot = w x V exactly, no BF16 product). The medians are 2.33 in layer 41, 0.89 in layer 59 and 0.95 in layer 5, against 2.38, 0.82 and 0.93.

### 4. The exploratory claims and the interpretation
- Claim 1 (one constant per layer): FACT, holds. All 10 weights have 1 unique value (0.06006 to 0.06543); layer 29 is 0.0625.
- Claim 2 (the score path): FACT, holds. The unrotated dims carry the score error. The score-path values move 0.3% to 5% with the causal mask (C6).
- Claim 3 (BF16 mismatch 0 to 0.0094): FACT, holds for the median over prompts. The maximum over prompts is 0.0105.
- Claim 4 (the sink): FACT, holds; it is stronger than written (C2).
- Claims 5 and 6: FACT for the numbers (I checked each range that the README quotes against the JSON summaries). The reading needs C1, C3 and C4.
- Scale-tie algebra: INTERPRETATION, correct. With K = rope(w V) and scalar w, the codes of V_unrot / vs read with k_descale = ks = w vs equal the K codes. The query is the plain permuted q. Only the unrotated 384 dims are shared. K_rot has its own grid.
- Screen code: no bug. `mirror` and `compact` do what the docstring says. FACT: max|K| / (w max|V|) is 0.995 to 1.003 (prompt 0, first-attempt `flips.py`), so ks / vs = w within 0.5%. INTERPRETATION: the largest V sits in an unrotated dim. This is a data property, not a rule. A tied design sets vs from amax V and ks = w vs.
- Is "the standard storage with the same scales" a fair baseline? INTERPRETATION: It is the right control for the layout question (C1). It is the wrong decision baseline, for four reasons.
  1. It passes by algebra (median 1.00).
  2. The stock engine uses scale 1.0. The standard error moves 0.35 to 1.58 times with amax scales (README, SA/S0) and 0.4 to 2.6 times with K phases.
  3. The standard can pick ks and vs freely. The tied row has one free scale per layer. The fair rival is a standard with calibrated scales.
  4. A user compares with the stock cache or with a calibrated cache. POST-HOC (leave-one-prompt-out): the calibrated standard is 0.68 times stock (pooled); the calibrated tied compact is 0.72 times; the compact over the calibrated standard is 1.06 pooled and 0.88 to 1.43 per layer.
- Can the FA4 FP8 path take separate K and V descales for the two views? FACT: Yes at the interface. `flash_attn.py` 1295 to 1296 passes `layer._k_scale` and `layer._v_scale` as two descale tensors [sequences, KV heads]. The pinned kernel folds them separately (C5).
  - HYPOTHESIS: The two-view read with K descale not equal to V descale gives the emulated numbers. E-52 used ones. Its two-scale rule ran on the CPU only.
  - FACT: An engine must also set `_k_scale = w * _v_scale` per layer and write the 640-value row with a new kernel.

### 5. Errors and statistics
- FACT: I found no error in the w form, head mapping, mask, query positions or dim order. V5 and the independent code agree.
- FACT: Probe bugs: C6 only. The layout screen and the k-first probe have no bug that I can show.
- FACT: Statistics. The unit of independent evidence is a layer (10 draws), not a prompt. The 4 prompts share the BOS sink, the same 4 channels and the same model weights, so they agree. The between-prompt spread (2.11 to 2.69) is 10 times the row-bootstrap interval, so the intervals understate the uncertainty. The 4 prompts are code and documents only.
- FACT: The rule's statistic (the maximum over 10 layers of a median ratio) has a wide null (key finding 7). R-73 predicted 0.90 to 1.07 from synthetic data with a random w on each dim.
- INTERPRETATION: Real w is one scalar per layer. So one grid phase acts on all 384 dims, and a few channels or one token decide the error. With a random w per dim the phases average out. This explains why the synthetic estimate was too narrow.
- INTERPRETATION: The FAIL is weak evidence that the layout stores less than the standard. It is firm evidence that the compact row at scale 1.0 is worse than the stock cache in layer 41.

### 6. D-90
See the next section.

## D-90 recommendation (not a decision)
Recommendation: (d) in a changed form (E-54 below), with the dtype-neutral build parts running in parallel. Fall back to (b), then (a), if E-54 does not PASS by the end of week 2.

Reasons.
1. (a) follows the frozen rule. The FAIL stays logged either way. But the rule could PASS an equivalent layout in at most 8% of draws (POST-HOC). (a) also drops a design that the post-hoc data support but do not prove.
2. (b) is the most certain: the BF16 compact error is 0 to 0.0105. But its memory is 1,280 B per token and head. That is 25% above the stock FP8 cache (1,024 B). It helps only BF16 KV users (2,048 to 1,280 B). It still needs the model arm M6, because the BF16 mismatch is systematic.
3. (c) and (d) are one experiment in practice. The only new design that the data support is the scale-tied row (640 B in FP8) with a per-layer scale chosen on calibration prompts.
4. E-54 as proposed passes by algebra, because both sides use the same scales. It needs the changes below.
5. The build parts that do not depend on the dtype (page plan, row write, views read, o_proj fold) serve (b) and (d). Only the scale plumbing waits.
6. HYPOTHESIS: The accuracy arms of E-54 are more likely than not to pass (97% of the 40 post-hoc leave-one-prompt-out cells were at most 1.25 times stock). The real uncertainty sits in the model arm, the chat prompts and the 32K context.

Changes to E-54 (write them before any fresh capture; each needs a review as R-73 did).
- M1. Baselines: the stock FP8 cache at scale 1.0 and a standard with its own calibrated scales are the decision baselines. The same-scale standard is the algebra control (median 0.95 to 1.05 per layer; cells 0.7 to 1.6).
- M2. Statistic: the primary is the pooled RMS ratio over the 10 layers, at most 1.25 against both baselines. POST-HOC null for a random phase: median 1.02, 95th percentile 1.19, 99th 1.26. Observed in leave-one-prompt-out: 0.72 against stock and 1.06 against the calibrated standard. The per-layer limit is 2.0 (the E-53 FAIL value). Do not use 1.25 per layer: the null passes it in only 8% of draws.
- M3. The scale rule: the grid of vs candidates, the calibration prompts, the objective and the tie-break are frozen. Calibration and test prompts are disjoint. Use steps of 1/16 octave or finer (the curve is jagged at that scale).
- M4. Controls: keep the E5M2 row and judge it by the pooled ratio (3.34 in E-53). V5 bound 0.005 (C8).
- M5. A kernel arm: the pinned FA4 reads the standard cache and the tied row (two views, K descale not equal to V descale) on captured tensors. Pin the FA commit (C5).
- M6. A model arm: a hook replaces K and V of the 10 global layers by the emulated storage values in the live engine. Compare next-token log-probabilities (KL, top-1 flips) with BF16 and with stock FP8. This is the R-73 O3 splice test. It needs its own rehearsal.
- M7. Kill rule: if the tied compact is worse than the calibrated standard by more than the pooled bound, stop the FP8 compact design and take (b) or (a).
- Minimal E-54 if time is short: M1 to M4 and M7 on a new capture (about $1). Then M5 and M6 move into the week-6 end-to-end test. That test is late for a design change.

Risks of E-54.
- R1. Forking paths: the layout, the tie and the scale idea came from the E-53 captures. Only a disjoint, frozen test is evidence.
- R2. Coverage. Fresh captures need new prompts, 2 or more chat-template prompts with turn markers, and one 32K prompt. E-53 removed special ids (A1, F4), and special tokens may be extra sinks. The context claim (README Section 10) is untested.
- R3. Sinks and channels: one sink vector or 4 channels decide a layer. A calibrated scale may not transfer to a new sink token. Cross-family transfer is good on E-53 data (POST-HOC: calibrate on code, test on documents: at most 1.12 times stock per layer).
- R4. BF16 mismatch: 1.6% to 4.7% code flips. The tie control has noise of 6% and tails of +54%. A tight tie bound fails on noise alone.
- R5. Scope: the tie needs one scalar w per layer. This holds for this checkpoint only (10 layers).
- R6. Calibration cost: the tied row needs per-layer scales (a calibration step). The claim "no calibration needed" ends.
- R7. Kernel drift: the FA `main` kernel computes in FP8 products with fixed Q scale 1. A vLLM bump changes the accuracy. The plain permuted q (not q x w) keeps Q as in the stock path.
- R8. Model quality: per-layer errors do not give model quality. M6 is the first measure of it. Its bound has no calibration yet.
- R9. Cost and time (INTERPRETATION): capture and analysis cost about $1 (E-53: $0.98). M5 and M6 add about $1 to $2. Review, rehearsal and post-run review take 3 to 5 working days of the 5 weeks.

## Checklist for the author (before the next step)
1. Apply C1 to C9 to the README Section 13 and Section 7, and to the journal (R-74 entry, D-90 text, Section 8 note for the "same codes" reading).
2. Commit the small result files (C7).
3. Write the E-54 preregistration with M1 to M7. Ask for R-75 before any spend.

## What I could not check
- I ran no vLLM and no GPU code. The kernel facts come from source at the pinned commit. I did not check that the Pod image holds that commit.
- I did not re-read `kv_cache_scheme` in the checkpoint. I rely on the E-50 record. I made no Hugging Face call.
- I re-ran the first-attempt phase scan on 8 cells only (all equal to 4 digits). The other 632 cells rely on my reading of its code. I did not re-run the first-attempt first-order model of the score error (`mech1.py`), and I do not use it.
- I did not recompute the amax rows of the screen or the mirror row with my own code. I checked the JSON values, the code and the algebra.
- I did not test chat prompts, 32K context, decode queries or another checkpoint.
- E-53 measures no model-level effect, and neither do I.
- I do not explain why 4 channels decide layer 41. HYPOTHESIS: a coherent rounding gain error on channels with large |q|.
- The leave-one-prompt-out and family-transfer numbers use the same 4 prompts that showed the problem. They are POST-HOC and optimistic.
