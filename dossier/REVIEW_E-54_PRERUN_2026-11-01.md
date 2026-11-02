# R-75: pre-run review of E-54 (accuracy of the scale-tied compact FP8 K=V cache on fresh Gemma 4 31B activations)

VERDICT: AMEND_THEN_RUN (one blocker, B1: CALSTD is not the best standard cache, so R2 flatters the tie by about 8 points; everything else reproduces)

Date: 2026-11-01.
Reviewer: R-75 (Sonnet 5.5, a different model from the author).
Stage: S2 hostile design review of the preregistration in design commit 05cc73e (`experiments/gemma4_kv_scaletie/README.md`).
Labels: FACT means that I read a file or a source, ran code or computed a number. INTERPRETATION means my reasoning. HYPOTHESIS means not tested. POST-HOC means that I computed it on the 4 E-53 captures, which are also the calibration data, so it is not a test.

Rules kept:
- I wrote this one file and no other repo file.
- A first import of the E-54 code wrote `__pycache__` files in two repo folders. I deleted exactly those files (the folder `gemma4_kv_scaletie/__pycache__` and one `.pyc` in `gemma4_kv_accuracy/__pycache__`). Later runs used `python -B`.
- I did not read `.env`. I made no Runpod, cloud or GPU call. I did not commit or post.
- I made read-only GET calls only. They were `gh api` on vllm-project/vllm (3966924d5) and on vllm-project/flash-attention (b273a02). They were also three public Hugging Face files of `RedHatAI/gemma-4-31B-it-FP8-dynamic` at revision d4ab4f57: `tokenizer_config.json`, `chat_template.jinja` and `tokenizer.json` (32 MB). I saved them in the scratchpad only.

Scratch folder: `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/20255ed9-038a-44e5-8a60-f7885a8c54ee/scratchpad/r75/`. It holds my own library `indep_lib.py` (cast, attention, tied row). It also holds the scripts that I name below and their logs.

## Key findings
1. FACT: The tied-row emulation is correct. I rebuilt it with my own code. It has its own E4M3 and E5M2 tables (0 mismatches against torch on 2,000,000 random values and on all ties). It has its own matrix-form attention and an explicit 640-code row. It applies the descales after the dot product, as the kernel does. It reproduces all 200 in-sample cells of `scales_e54.json` (10 layers, 4 E-53 prompts, 5 arms) to 1.4e-5 relative or better (`recompute_cells.py`).
2. FACT: The frozen scales reproduce (`rederive_scales.py`). My code gives the same TIED phase and the same CALSTD K and V phases in all 10 layers. The amax values and the bases are equal. The 32-point curves differ by 2.9e-5 at most.
3. FACT: The real analysis code gives the README numbers on real data (`e53_through_e54_pipeline.py`). I ran `compute_metrics`, `stats` and `verdict` on the 40 E-53 capture files. The result is P_STOCK 0.04532, P_CALSTD 0.02678, P_TIED 0.02980, R1 0.6575, R2 1.1127, E5M2 3.035 and tie 0.9709. The largest engine gap is 0.00172 and the largest identity error is 0.0024. The 13 tests pass (64 s).
4. FACT: The `real` branch of V2 works (`v2_real_shape.py`). No test and no rehearsal runs it. A record with the frozen plan, 7 prompts, 10 layers and the real scales hash passes V2. Ten single faults each fail V2.
5. FACT (source): The pinned stock FA4 SM90 FP8-KV path reads the tied row as the emulation says. The K descale folds into the score scale of the whole K view. The V descale folds into the final normalization. The FP8 tile becomes FP16 with no scale. The call in `kernel_e54.py` uses valid keywords of vLLM's own `flash_attn_interface.py`. The fork file at b273a02 is not the file in the image.
6. FACT: The Gemma 4 chat template and tokenizer are safe for E-54 (`tok_check.py`, `chat_build_check.py`). BOS comes once. No system or thinking block appears with the defaults. `<|turn>` and `<turn|>` are in `all_special_ids`. On a local torch tree the builder gives chat prompts of 8,192 tokens, one BOS and 9 and 11 special ids after position 0. The two chat prompts share their first 19 tokens.
7. INTERPRETATION: The 32K prompt fits the engine settings. The E-53 log shows a maximum concurrency of 4.94 at 8,448 tokens per request. One request of 33,024 tokens needs 1.4 to 2.3 times the KV memory of one 8,448-token request. So the concurrency falls to 2.2 to 3.6. In the worst case (sliding layers at full length) it is 1.27. It stays above 1.
8. POST-HOC: The frozen rule gives PASS in 10 of 10 held-out reconstructions (`transfer_analysis.py`, `mc_pass.py`). I applied the frozen calibration rule to subsets of the 4 E-53 prompts and tested on the others. R1 is 0.61 to 0.72. R2 is 0.92 to 1.17. The largest per-layer ratio is 1.44.
9. POST-HOC: The rule needs transfer to PASS, and FAIL hardly depends on it. With TIED and CALSTD at random phases, I get PASS 30%, INCONCLUSIVE 66% and FAIL 3.7%. With half of the layers at their calibrated phases, I get 54%, 43% and 3.3%. The CALSTD error in this Monte Carlo comes from an additive model (about 4% error per cell).
10. FACT: The gain over STOCK is a phase-search gain on sharp minima. A standard cache with power-of-2 scales and no search has P 0.0460 (additive model), against 0.0453 for STOCK. One phase step (2.2% in scale) changes the tied error by 1.0 to 2.6 times.
11. FACT: Position 0 carries a large part of R1 (`bos_check.py`). Position 0 is the BOS vector, and it is the same in every prompt. With position 0 exact in all arms, the in-sample R1 goes from 0.66 to 0.82 and R2 from 1.11 to 1.15.
12. FACT: CALSTD is not the best standard cache of the same search (B1). The rule picks the K phase with V exact and the V phase with K exact. With both casts, a better pair exists. I searched the 40 best pairs of the additive model with real attends (`joint_top.py`). The in-sample error sum falls by 1.0% to 15.4% in 8 of 10 layers. It does not change in layers 23 and 41.
13. FACT: The joint gain is real out of sample (`joint_lopo.py`). Pooled in sample, P_CALSTD falls from 0.0268 to 0.0251 and R2 rises from 1.113 to 1.189. In leave-one-prompt-out, held-out R2 against the joint baseline is 1.04 to 1.24 (mean 1.15). Against the separable baseline it is 0.97 to 1.16 (mean 1.07).
14. FACT: Saturation cannot happen in this model. V comes from a weightless RMS norm over 512 dims, so |V| is below 22.63. K is w times V, rotated in pairs, so |K| is below 1.48. The data agree: the largest |V| is 22.5 and the largest |K| is 1.43. The README threat "Headroom" is closed.

## Blockers
B1. The decision baseline CALSTD is not the best standard cache, and the README says that it is.
- Where: README Section 8, lines 227 to 228 (the sentence about CALSTD as the best standard per-tensor scales by the same calibration data and grid). Also README Section 4.6 (the CALSTD rule), `calibrate_e54.py` lines 79 to 81 (separate score and value paths), and the CALSTD entries of `scales_e54.json`.
- Failure: the separable search picks a V phase that is not best when K is also cast. My top-40 joint search (finding 12) lowers the in-sample error sum by up to 15.4% in a layer. The joint pairs (K phase, V phase) for layers 5 to 59 are (2,26), (3,16), (13,7), (26,18), (21,19), (3,8), (15,15), (5,7), (19,17) and (22,15). The advantage transfers to held-out prompts: held-out P_joint is 0.92 to 0.94 times P_sep in all 4 folds.
- Why it matters: R2 is the informative statistic, because R1 is partly one vector (finding 11). Against the best standard cache, the tie costs about 19% pooled in sample, not 11%. The PASS bound 1.25 then leaves 6 points in sample. In two of the four held-out folds it leaves less than 1 point (R2 1.243 and 1.236).
- Why it matters (continued): the README range for R2 (1.05 to 1.25) was written for the weaker baseline. A PASS against CALSTD as it is would not show "within 25% of the best standard cache".
- Fix: the user decides between (a) and (b) before S4. The decision goes in the journal Section 10.
- Fix (a): recalibrate CALSTD with a joint search over (ks, vs) with both casts. It runs offline on the Mac. A full grid of 1,024 pairs per layer needs one softmax per K phase (my estimate: about 9 min per layer). Freeze a new `scales_e54.json` and a new `SCALES_SHA256` as amendment A1, because no E-54 data exists. Update README Sections 4.6, 7 and 8. Ask the user to confirm the tolerated tie cost and the R2 bound, because the expected held-out R2 becomes about 1.15.
- Fix (b): keep the separable CALSTD. Change the Section 8 sentence to say "separable calibration". Add a reported-only arm CALSTD_JOINT with frozen joint scales (O3). Say in Sections 1 and 7 that the claim is against the separable baseline.

## Fixes (should be done; each is cheap)
- F1a (README line 44 to 47). The README says that the in-sample R1 and R2 are optimistic. R1 is not: held-out 0.61 to 0.72, in sample 0.66. R2 is pessimistic against the separable baseline: held-out 0.92 to 1.17, in sample 1.11.
- F1b (README lines 177 and 205 to 207). The README justifies the bound 1.25 with the R-74 null. That null randomizes only the K phase against one fixed STOCK cache. Cite the held-out and Monte Carlo numbers of findings 8 and 9 instead.
- F1c (README line 300). Close the "Headroom" threat with finding 14.
- F1d (README Section 7). Say that the FAIL bound 1.5 is looser than R-74 M7 (stop above 1.25), and give the reason.
- F1e (README Section 7, line 214). Add the action for INCONCLUSIVE. Suggestion: if one prompt class or one layer causes it, recalibrate offline on the saved captures. Test the new scales on new prompts under a new S4.
- F2 (`analyze_e54.py`, `layer_metrics` and `stats`). Add reported-only metrics with position 0 kept exact in every arm: R1_x0 and R2_x0. In sample they are 0.82 and 1.15. They show how much of R1 is the shared BOS vector. The cost is 3 attends per cell.
- F3a (`test_e54.py`). Add the test `a54.sha_file(a54.SCALES) == a54.SCALES_SHA256`. The values are equal today, but nothing guards them. V2 would fail the real run if an edit breaks them.
- F3b (`test_e54.py`). Add a test of the `real` branch of `v2_problems`. My `v2_real_shape.py` is a template.
- F4a (`session_e54.sh` line 27). Raise `AUTO_STOP_DELAY` from 450 to 600 s. E-53 measured 24 MB/s (3.1 GB) and 16 MB/s (rehearsal 2). The 7.85 GB of E-54 need 330 to 490 s, plus the RES copy and the checksum compare. The main session deletes the Pod after the copy, so the extra time costs money only when the copy is slow (at most $0.17).
- F4b (README Section 9, the copy step). Copy the prompt folders in order and put prompt 6 last. Then a cut window loses only the largest prompt.
- F5a (`kernel_e54.py` lines 42 and 126). Narrow CLOSE to a gap of at most 0.01, and call a larger gap MISMATCH. The FP8 errors are 0.02 to 0.07, so a gap of 0.05 is as large as the effect.
- F5b (`analyze_e54.py`, `kernel_line`). A timeout leaves `kernel.json` without `outcome`, and `kernel_line` prints "NOT RUN". Print "TIMEOUT" when `finished_at` is missing.
- F6 (`analyze_e54.py` line 511). A NaN R1 or R2 gives FAIL, because `not nan <= 1.5` is true. Make a non-finite R1 or R2 INVALID (V4). The risk is small, because V4 catches most non-finite data first.
- F7 (`capture_e54.py` line 40, `CHAT_ASK`). The two chat prompts start with the same 19 tokens, so their first K and V rows are equal. Use two different ask texts. Or state in README Section 10 that the chat class has one prefix.
- F8 (README Section 9 and `session_e54.sh`). Give the Pod volume disk (E-53 used 100 GB). Add `df -h /workspace` to `session.log`. Record the `chunks` of prompt 6 in `summary.json`.
- F9 (README Sections 4.1 and 4.7). Record where the kernel interface comes from. The `flash_attn_varlen_func` of the image is vLLM's own file `vllm/vllm_flash_attn/flash_attn_interface.py` at 3966924d5. The fork commit b273a02 supplies only `flash_attn/cute`. The CMake file excludes the fork's interface file (lines 70 to 78). Record also that FA4 SM90 FP8-KV needs `page_size == tile_n == 64` and ignores `q_descale`.

## Options (may be done)
- O1. Add reported-only arms TIED at one phase step above and below the frozen scale (vs x 2^(+1/32) and x 2^(-1/32), ks = w vs). They measure the fragility: a build must reproduce the scale to about 1%. The cost is 2 attends per cell.
- O2. Add a reported-only arm P2STD (power-of-2 scales, no search). It shows the share of the gain that comes from the phase search (finding 10).
- O3. Add a reported-only arm CALSTD_JOINT with frozen joint scales (B1, option b).
- O4. After the run, test a three-scale tied row offline on the saved captures (exploratory). K_rot gets its own phase (scale the rotated part of the query). V_rot gets its own phase (fold the scale into `o_proj`). The 384 unrotated dims stay tied. HYPOTHESIS: the gain is modest, because the unrotated dims carry most of the score error (E-53 exploratory item 2).
- O5. Make V2 require 4 chunks for prompt 6 in every layer. The record `chunks` is in `capture_run.json`.

## Answers to the checks

### 1. The tied row (`tied_codes`, `tied_arm`)
- FACT: `tied_codes` (`analyze_e54.py` lines 113 to 117) builds 640 codes per token and KV head. They are cast([K_rot / ks | V_unrot / vs | V_rot / vs]). The scale vector holds ks on the first 128 values and vs on the other 512.
- FACT: `tied_arm` (lines 120 to 129) reads codes 0 to 511 as the K view and codes 128 to 639 as the V view. It multiplies them by ks and by vs. The query is [q_rot | q_unrot] from `views_query(q, ones)`, so it holds no w.
- FACT: `unpermute` puts the output back in the natural order. The first 384 values of the V view are the unrotated dims. The last 128 are the rotated dims.
- FACT (data): the split into 128 rotated and 384 unrotated dims is right. On prompt 0, in all 10 layers, the median K/V ratio is w within 1% on every unrotated dim. In layer 41 it lies between -0.002 and 0.002 on the rotated dims.
- FACT (source): the stock kernel does what the emulation does. It folds `softmax_scale x q_descale x k_descale` into the score scale (`flash_fwd_sm90.py` lines 1439 to 1454 and 1793 to 1804). It uses `v_descale` as the final scale of the softmax normalization (line 1702). It casts the FP8 tile to FP16 without a descale (line 1831).
- FACT: my independent code gives the same errors as `scales_e54.json` in all 200 cells. The largest relative difference is 1.4e-5. The code has its own E4M3 table, its own matrix-form attention, an explicit 640-code row and the descales after the dot product.
- INTERPRETATION: ks = w vs makes the K_unrot values right. ks x code(V_unrot / vs) is about w x V_unrot. That is K_unrot up to the BF16 mismatch. The mismatch flips 1.6% to 4.7% of the codes (R-74 C1). V7 bounds its effect.
- INTERPRETATION: the emulation multiplies the codes by ks before the dot product. The kernel multiplies after it. The two agree to float32 rounding.
- Result: I found no fault in the tied row.

### 2. The calibration
- FACT: my code re-derives all 10 layers. The TIED phases are 3, 25, 15, 21, 21, 1, 15, 16, 15 and 24 for layers 5 to 59. The CALSTD K and V phases equal the file in every layer. The amax values, the bases (2^-7 and 0.125) and the curves agree. The curve difference is 2.9e-5 at most.
- FACT: the rule is implemented as README Section 4.6 says. The base is the smallest power of 2 at or above 2 x amax / 448. There are 32 phases. The argmin takes the lowest phase on a tie. CALSTD uses two separate paths. Every scale is rounded to float32.
- FACT: the headroom is not an issue (finding 14). With the bases 0.125 and 2^-7 the largest code is about 190 at phase 0, far below 448. No value can saturate at any phase.
- FACT: CALSTD is not the best standard cache of the same search (findings 12 and 13, B1). In my top-40 search the best pair differs from the separable pair in 8 of 10 layers.
- INTERPRETATION: the separable search ignores the interaction of the K error and the V error. The additive model has the same blind spot. It is off by 4% per cell at the frozen phases and by up to 40% at random phase pairs.
- FACT: the gain over STOCK comes from the phase search, not from the power-of-2 base. A standard cache with power-of-2 scales and no search has P 0.0460 (additive model) against 0.0453 for STOCK. CALSTD has 0.0268. So CALSTD is a selected minimum of a jagged curve.
- INTERPRETATION: a standard cache with the usual amax scale would sit at a random phase and be much worse than CALSTD. So CALSTD is a strong baseline, but the joint search is stronger.
- INTERPRETATION: the change of the rule before the freeze is not a forking-path problem for the test. The change used the 4 E-53 captures only. The scales are frozen by SHA-256 in the design commit, and V2 compares the hash. No E-54 data existed. The cost is that the in-sample numbers are not evidence, and the README says so.
- INTERPRETATION: the forking path that remains is the idea itself. The tie and the phase search came from the same 4 prompts. The test prompts share no file with them. They share the model, the tokenizer, the text families and the BOS vector (finding 11).

### 3. The statistic and the rule
- FACT: the R-74 null does not transfer. It randomizes the K phase of an equivalent standard cache against one fixed STOCK cache, on 4 prompts. E-54 compares two calibrated arms. My Monte Carlo with random phases for both arms gives R1 median 1.18 (5% to 95%: 0.99 to 1.42). It gives R2 median 1.13 (0.88 to 1.45). So the README sentence about 99% of draws (line 205) is not the right justification.
- POST-HOC: the bounds are sensible for the frozen separable baseline. In 10 held-out reconstructions the rule gives PASS 10 times. I calibrated on 2 or 3 of the 4 E-53 prompts and tested on the rest. R1 is 0.61 to 0.72 and R2 is 0.92 to 1.17. The largest per-layer g1 is 1.21 and the largest g2 is 1.44.
- POST-HOC: the in-sample R2 (1.113) is above the held-out mean (1.06). CALSTD has two selected scales, so it over-fits more than TIED.
- POST-HOC: against the joint baseline (B1) the picture changes. Held-out R2 is 1.04 to 1.24 in the 4 leave-one-prompt-out folds. The in-sample value is 1.19. PASS is still reachable, but the margin to 1.25 is thin.
- POST-HOC: PASS needs transfer, and FAIL does not. With calibrated phases that do not transfer at all, PASS has a probability of 30%, INCONCLUSIVE 66% and FAIL 3.7%. With half of the layers transferring, the figures are 54%, 43% and 3.3%. With 80% transferring, they are 76%, 22% and 2%.
- INTERPRETATION: FAIL is a structural alarm. It needs a pooled ratio above 1.5, and a loss of transfer rarely gives that. A good design with a failed transfer shows as INCONCLUSIVE. So a FAIL is unlikely to be a false FAIL from a bad calibration set. The action of INCONCLUSIVE ("the user decides") needs content (F1e).
- FACT: PASS and FAIL cannot both hold. FAIL needs R1 or R2 above 1.5, and PASS needs both at most 1.25. Every outcome has an action in README Section 7.
- FACT: the rule deviates from R-74 M7. M7 says to stop if the tied row is worse than the calibrated standard by more than the pooled bound (1.25). E-54 stops only above 1.5 and treats 1.25 to 1.5 as INCONCLUSIVE. The README gives no reason (F1d).
- FACT: `verdict` treats a NaN R1 or R2 as FAIL (`analyze_e54.py` line 511). V4 catches most non-finite data first, so the risk is small (F6).

### 4. Validity checks V1 to V7
- V1 (H100 80GB): FACT: it passes for "NVIDIA H100 80GB HBM3", the E-53 name. INTERPRETATION: it fails for a PCIe or NVL name, because those names have no "80GB". That is right for this plan.
- V2: FACT: I ran its `real` branch (finding 4). The plan equality holds after a JSON round trip. The ten faults that fail are a changed plan, a wrong compile mode, an FP8 cache, a wrong vLLM version and a wrong scales hash. The others are an E-53 file, a chat prompt without markers, a plain prompt with a special id, a short prompt and a missing layer.
- V2 (continued): FACT: the E-53 exclusion works only on the E-53 image (README Section 4.1 says so). V2 does not check `chunks` (O5).
- V3: FACT: the check `w_value == scales.w` is an exact float equality. INTERPRETATION: it holds because both sides are BF16 values converted to float. A different checkpoint revision fails it, which is the intent.
- V4: FACT: the reference is a finite float32 attention. I found no issue.
- V5: FACT: the E-53 largest gap is 0.00172. I expect 0.0017 to 0.0022 for chat and 32K prompts. INTERPRETATION: BF16 output rounding gives about 0.0016. BF16 rounding of P gives about 0.001, independent of the length. The bound 0.005 keeps a margin of 2.3 times or more.
- V6: FACT: in sample 3.03 against the bound 1.5. I found no issue.
- V7: FACT: in sample 0.971. INTERPRETATION: the out-of-sample value should be about 1.0. The code flips act as noise of about 6% per cell (R-74 C1) and about 3% pooled. The band 0.90 to 1.10 is safe.
- INTERPRETATION: V6 and V7 cannot find an error that all arms share, for example the cast convention. V5 ties the reference to the engine, not to the cast. The cast was fixed in E-53 and R-73, and I found no reason to doubt it.

### 5. The capture
- FACT: I read the public files of the pinned revision. `chat_template.jinja` starts with `{{- bos_token -}}`. It adds a system block only if `enable_thinking` is true, tools exist or the first message is a system message. It maps "assistant" to "model". It has no alternation check.
- FACT (continued): the template raises only for tool-call arguments that are strings. `add_generation_prompt` is false by default.
- FACT: with transformers 5.20.2 and the pinned `tokenizer.json`, `apply_chat_template(turns, tokenize=False)` gives `<bos><|turn>user\n...<turn|>\n<|turn>model\n...`. `encode(add_special_tokens=False)` reads the first `<bos>` as id 2. So BOS occurs once, and no double BOS can occur. The ids 105 (`<|turn>`) and 106 (`<turn|>`) are in `all_special_ids`.
- FACT: I ran `build_prompts` with the real tokenizer on the files of a local torch install. The Pod holds vllm and torch, so its real file list differs. The plain prompts have 8,192 and 32,768 tokens and no special id after position 0. The chat prompts have 8,192 tokens, 5 and 6 turns, BOS at 0 and 9 and 11 special ids after it. Prompt 5 has 2 markers inside the query rows.
- FACT: the two chat prompts share their first 19 tokens (`<bos><|turn>user\n` and the ask text). Their K and V rows are equal there. The two prompts are less independent than two plain prompts (F7).
- INTERPRETATION: `max_model_len` 33,024 with 8,448 batched tokens works. The E-53 log shows that chunked prefill is on and gives a concurrency of 4.94 at 8,448 tokens.
- INTERPRETATION (continued): I took the page rules from `kv_cache_interface.py`. A full-attention group needs `cdiv(max_model_len, block_size)` pages. A sliding group needs the window plus the in-flight tokens (one or two batches of 8,448), plus one page.
- INTERPRETATION (continued): with these rules, a 33,024-token request needs 1.4 to 2.3 times the memory of an 8,448-token request. The concurrency is 2.2 to 3.6, and 1.27 in the worst case. The 32K prompt runs as 4 chunks of 8,448, 8,448, 8,448 and 7,424 tokens.
- INTERPRETATION: the hooks work over 4 chunks. The pre-hook stores the positions of each chunk. The selection `qsel` is empty in chunks 1 to 3, and `reshape(-1, heads, head)` of an empty tensor is valid. The test `test_hooks_record_a_prefill_in_four_chunks` runs this case. `Store.end` sorts by position, and `check_prompt` fails the run if a position or a shape is wrong.
- FACT: nothing that I read in vLLM 3966924d5 breaks this. The E-53 log shows FA4 for all layers on the H100. The rehearsal uses TRITON_ATTN on sm_89, so the FA4 path with a prefix (chunks 2 to 4) first runs on the H100. It is standard chunked prefill, and I see no risk.

### 6. The kernel arm
- FACT: the call in `fa4_call` (`kernel_e54.py` lines 68 to 79) uses only keywords of vLLM's `flash_attn_varlen_func` at 3966924d5 (`vllm/vllm_flash_attn/flash_attn_interface.py` lines 176 to 218). It matches E-52's `stock_call`. E-52 ran 4 stock FP8 variants on the H100 with status ok in all 32 rows. Its correctness error was 0.00166 against the stored values.
- FACT: the shapes are right. FA4 needs the descales as float32 [sequences, KV heads] with unit stride in the last dim (`cute/interface.py` lines 919 to 930). `torch.full((256, 4))` fits. `q_descale` is None, and FA4 sets it to 1.
- FACT: the page size 64 is right. vLLM forces 64 for head size 512 with an FP8 cache on SM90 (`flash_attn.py` line 324). FA4 asserts `page_size == tile_n` (`cute/interface.py` line 1009), and the tile is 64 for head size 512.
- INTERPRETATION: 256 sequences with `seqused_k` = qpos + 1, `causal=True` and `num_splits=0` give 1,024 work items (256 sequences x 4 KV heads). The split heuristic returns 0, which FA4 treats as no split (`is_split_kv = num_splits > 1`). E-52 used the same path at batch 19.
- INTERPRETATION: a K view with a row stride of 640 bytes and a V view that starts at byte 128 are 16-byte aligned for TMA. E-52 ran such views.
- INTERPRETATION: it will run. The new elements against E-52 are the non-unit descales and the batch of 256. I see no reason that they fail. A failure writes ERROR with its reason and does not change the primary outcome.
- INTERPRETATION: the gap rule is right for MATCH. The kernel writes BF16, so a correct read gives about 0.0017 to 0.0022. The BF16 output gives 0.0016. The P rounding of the emulation adds more at 32K. MATCH at 0.005 keeps a margin of 2.3 times. The CLOSE band up to 0.05 is too wide (F5a).
- FACT: swapped descales give a gap near 1 in the stand-in test, so a MISMATCH is detectable.

### 7. The session and the budget
- FACT: the step caps add up to 420 + 720 + 360 + 300 + 60 + 60 = 1,920 s. This equals `POD_LIMIT`. The checksum list takes 20 to 40 s for 7.7 GB. It and the last decision run have no slack when every step hits its cap. In a normal run the steps need about 10 min.
- FACT: the order of the watchdogs is right. The session watchdog fires at about 2,060 s after boot and the boot watchdog at 2,400 s. `finish()` kills the session watchdog before the copy window, so the window is not cut by it.
- FACT: the boot watchdog stops the Pod (`pod_cmd.py`), and a stopped Pod keeps `/workspace`. RUNPOD.md Section 6 describes the restart with 0 GPUs.
- FACT: the worst case of $3.66 is right: (900 s + 2,400 s) / 3,600 x $3.99. The typical cost is $1.85 to $2.40.
- INTERPRETATION: the copy window of 450 s is tight (F4). E-53 measured 24 MB/s for 3.1 GB and 16 MB/s in rehearsal 2. The 7.85 GB need 330 s at 24 MB/s and 490 s at 16 MB/s, before the RES copy and the checksum compare. The fallback exists.
- FACT: the rehearsal plan is sound. The cap is $0.60 and the worst case is $0.59. Its VERIFY items cover the engine at 33,024 tokens, the chat template, the 4 chunks, the analysis and the copy.
- FACT: the lessons of E-53 are used. They are `cfg_value`, 8 copy streams, the 15 min SSH rule, the fail-fast after prompt 0 and the eager mode names. Two small items are missing: the Pod disk size and a `df` line in `session.log` (F8).

### 8. Scope and the biggest risk
- INTERPRETATION: it is right to leave the splice test (R-74 M6) to E-55. The splice changes the forward pass, has no bound and needs its own rehearsal. R-74 itself allowed a minimal E-54 without M5 and M6. The price is a second Pod session (about $2).
- INTERPRETATION: a PASS is meaningful only for these scales. The gain is a selected minimum, and one phase step changes the tied error by 1.0 to 2.6 times. A build must reproduce the scale to about 1%. The report should say it.
- INTERPRETATION: a PASS says nothing about model quality (R-74 R8), the cache-write path or decode with a cached FP8 prompt. E-55 must test them.
- INTERPRETATION: R1 is partly the transfer of one vector (finding 11). R2 is the informative statistic, and its baseline is not the best one (B1).
- INTERPRETATION: a FAIL is unlikely to be wrong. Random phases give FAIL in 3.7% of draws, and a design that FAILs must have a pooled ratio above 1.5.
- INTERPRETATION: the single biggest risk is an over-read PASS. Four effects push towards it. The baseline is weaker than the best standard cache (B1). R1 is partly one vector (finding 11). The gain rests on sharp minima (finding 10). No model-level test exists.
- INTERPRETATION (continued): fix B1, add F2 and O1, and state these limits in the report.

## What I could not check
- I ran no GPU code and no vLLM. I do not know the outcome of the kernel arm (MATCH, CLOSE or MISMATCH). I do not know the engine start with 33,024 tokens or the hooks over 4 chunks in a live engine. My statements rest on source and on the E-53 log.
- The prompt check used a local torch tree, not the Pod files. The special-id skips and the chat marker counts on the real files stay open until S5.
- My held-out and Monte Carlo numbers use the same 4 prompts as the calibration. They say nothing about chat prompts or 32K. The Monte Carlo uses an additive model for CALSTD.
- My joint search is a top-40 search, not the full grid of 1,024 pairs. So its gain is a lower bound for the in-sample joint optimum.
- I did not test the cache-write path, decode with a cached FP8 prompt or any model-level effect.
- I did not re-read the checkpoint `config.json` for `kv_cache_scheme`. I rely on the E-50 record and R-74.
- I did not read `MEASUREMENT.md`, and I read only Section 6 of `RUNPOD.md`.
- I did not measure the difference between the P rounding of the emulation (normalized P) and of the kernel (unnormalized P). I expect it to be small and equal for all arms.
