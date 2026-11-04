# R-80: post-run review of E-56 (compact 16-bit K=V cache, output correctness)

Reviewer: a hostile second model. Date: 2026-11-04. Scope: the E-56 README (Sections 1 to 9, 13, A1, A2), the code, the frozen data and the A100 run `results/e56-a100-20261104/`.
Labels: FACT (I read it or I computed it), INTERPRETATION (reasoning), HYPOTHESIS (not tested).
I used no GPU, no network and no `.env`. My scripts read only the saved files.

## Verdict: CONFIRM_WITH_CORRECTIONS

FACT: the frozen decision GO1 GO2 GO3 is right. I rebuilt it from the raw files with my own grader. No validity check fails.
INTERPRETATION: the claim "the compact 16-bit cache keeps the final answers" is supported for short GSM8K chats and for a saturated needle test at 7.7K tokens. Three statements in Section 13 are too strong, and two facts in Sections 4.1 and 10 are wrong. The most important finding is new: the raw-text failure of E-55 is a property of the model and the input in this engine, not of the FP8 checkpoint and not of FA4.

## 1. Independent check of the decision

### 1.1 Accuracy and answer changes
I wrote a new extractor (text after the last `####`, strip markdown and `$`, first number). I graded all 5 arms from `outputs_<arm>.json`.

| Arm | Correct, strict `####` rule | Correct, README rule (last number as fallback) |
|---|---|---|
| BF16 | 421 of 432 | 422 |
| DET | 421 | 422 |
| BF16C | 420 | 420 |
| BF16R | 421 | 421 |
| STOCK8 | 421 | 421 |

- FACT: the README grader differs from my strict grader on one item, G299 in BF16, DET, BF16R and STOCK8. That answer loops ("re-evaluate based on a standard calendar") and stops at the 640-token limit with no `####`. Its last number, "$3,200.", equals the gold by chance. BF16C is the only arm whose G299 answer finished (3400, wrong).
- FACT: so the README change BF16C - BF16 is -2 items (-0.0046). With a strict grader it is -1 item (-0.0023), interval [-0.0087, +0.0041]. The README interval is [-0.0124, +0.0032]. Both are inside +/-0.03. GO1 holds either way.
- FACT: the extraction is clean. Of 400 answers per arm, 9 tails across all arms are not a bare number, and all are "25%", "60%" or "50%" with a correct value. The 10 wrong BF16 answers with `####` are real arithmetic or reading errors (for example G089 gives 3.5, gold 3). I found no extraction error.
- FACT: the items that change in final answer are few. BF16C: G218 (192 to 176) and G299. BF16R: G218, G299 and G329 (24 to 23, wrong to right). STOCK8: G299 only. So 3 of the 6 changes come from one looping item that BF16 itself leaves unfinished. Without G299 the counts are 1, 2 and 0.
- FACT: G218 flips in BF16C and BF16R at the same token (position 91, "Assuming the twenty" against "This implies there are") in a problem with a typo in its text. INTERPRETATION: this is a near-tie flip of the sampler path, not a cache effect. BF16C has no flip of its own that BF16R lacks, apart from G299.
- FACT: H2 recomputed: differ BF16C - BF16R = -0.0023, interval [-0.0087, +0.0041]. GO2.

### 1.2 Reference NLL
I recomputed from `arrays/*.npz` (37,877 answer tokens; the count matches `score_ids` minus `answer_start`).

| Arm | Delta (nats per token) | 95% interval | Items worse / better |
|---|---|---|---|
| BF16C | -0.00091 | [-0.00156, -0.00025] | 187 / 213 |
| BF16R | +0.00274 | [+0.00205, +0.00344] | 262 / 138 |
| STOCK8 | -0.00085 | [-0.00174, +0.00004] | 188 / 212 |
| DET | 0 exactly | | |

FACT: these equal `summary.json` to 5 digits. BF16 reference NLL is 1.7526 and top-1 agreement 0.7757. Long-context change from token 1,025: BF16C +0.00659, BF16R +0.00743, STOCK8 +0.00663. All match.

### 1.3 Validity checks
- V1: FACT, A100-SXM4-80GB.
- V2: FACT from `summary.json` and the engine record (kv auto, no prefix caching, mode 0, cudagraph NONE, 2,048 batched tokens, snapshot revision as pinned). I did not recompute the 7 code hashes.
- V3: FACT, DET is bit-identical to BF16 in all arrays (max difference 0.0) and in all 432 answers.
- V5 and V7: FACT from `run_main.json`: BF16 gap 0.0015 to 0.0018 in every layer; BF16C separations 0 in layer 29 and 0.0018 to 0.0104 elsewhere. These use the raw-text probe B1. That is acceptable, because they test kernel numerics and not model health.
- V8: FACT, GSM8K 0.975, needle 1.0, NLL 1.7526 against the A1 bound 3.0.

### 1.4 Did A1 and A2 decide the outcome?
- FACT: under the frozen rules of f1b3502, V8 allows a reference NLL of 1.5 at most. The measured value is 1.7526. The run would have stopped after BF16 (exit 3). DET, BF16C and BF16R would not have run. The outcome under f1b3502 is INVALID.
- INTERPRETATION: A1 B1 (bound 1.5 to 3.0) was written at 20:11, before any 31B data. R-79 gave a reason (human solutions are not the model style). So the amendment is procedurally clean. But it did decide whether a result exists. The README must say this in plain words. The bound sits close to the value: 1.75 is only 0.25 above the old bound.
- FACT: A2 did not decide the outcome. With the old 320-token limit, 8 BF16 answers are cut (about 2%). I cut the saved token ids at 320, decoded them, and graded with the README rule: BF16 382, BF16C 384, BF16R 384, STOCK8 383 of 400. The change BF16C - BF16 would be +2 items. GO1 holds. H3 does not depend on the limit. Needle answers are 7 to 9 tokens, so the old limit of 24 is also enough. Caveat: this is a re-grade of truncated greedy text, not a new run.

## 2. Items where an arm differs from BF16

- FACT: for final answers, see 1.1. For text, the share of items with other token ids is BF16C 93 of 432 (21.5%), BF16R 75 (17.4%), STOCK8 112 (25.9%). All differences are in GSM8K. No needle answer differs in any arm (the 32 texts are identical in 4 arms).
- FACT: BF16C only diverges in 42 items where BF16R does not. BF16R only diverges in 24. Sign test p = 0.036 (my own test, two-sided, not corrected, not preregistered). STOCK8 against BF16C: 52 against 33, p = 0.05.
- FACT: the median first divergence is at token 40 to 46 in all arms. A few start at token 0 (BF16C 1, BF16R 3, STOCK8 4). Example: STOCK8 on G329 starts "To solve the problem" against "1. **Identify".
- INTERPRETATION: text-level divergence is a near-tie effect of any small change. The order for text is BF16R < BF16C < STOCK8. The order for the mean absolute per-token log-probability change is BF16C 0.0169 < BF16R 0.0198 < STOCK8 0.0249. The two orders disagree on BF16C and BF16R, so I do not read a cache defect into the text result. But H2 covers final answers only. The word "outputs" in the commit title and in Section 13 is wider than the evidence.
- FACT: the extraction of numbers with units and with several `####` is safe. Only one answer in all arms (G299) has no `####`. No answer has two.

## 3. What the thesis may claim

Allowed (INTERPRETATION): "With the 16-bit Gemma 4 31B weights on one A100 with greedy decoding, an emulated compact K=V cache for the 10 global layers changes the final answers of 400 GSM8K problems and 32 needle prompts no more than a rounding-size change of the query does. The accuracy change is -0.5 points with an interval of [-1.2, +0.3] points. The reference NLL changes by less than 0.002 nats per token."

Caveats the claim must carry:
1. FACT: the cache is emulated. The memory saving (37.5% of global KV bytes) is arithmetic. No write kernel was tested.
2. FACT: the positive control did not work. STOCK8 (FP8 K and V in the 10 global layers) shows 1 answer change and NLL -0.00085, equal to BF16C. So the test cannot separate an 8-bit cache from BF16C in accuracy or mean NLL at these lengths. The claim stands on the mechanism (BF16C differs from BF16 by one rounding), and E-56 checks that nothing breaks. E-56 does not show that the test would detect a harmful cache. INTERPRETATION: say "no break found", not "sensitive test passed".
3. FACT: GSM8K prompts are 97 tokens on average (maximum 226) and answers about 142 tokens. A whole sequence is about 240 tokens, far below the 1,024 window. The global layers see the same tokens as the sliding layers. Section 10 point 7 says "about 300 tokens" for the prompt. That is wrong. The long-context evidence is only the needles.
4. FACT: the needle test is saturated. All 32 answers are correct in every arm with identical text. It tests one salient fact at 7.7K tokens. It has no power to see a small loss. Nothing above 8K, no multi-hop, no long generation.
5. FACT: A100 with Triton for all layers (see C7), one checkpoint, one vLLM build, 400 GSM8K items. The 3-point margin is a heuristic.
6. FACT: the result is a statement at the level of the final answer. Text diverges in about one answer in five (see Section 2).

Long-context NLL (reported only): FACT, the baseline NLL on needle prompt tokens from 1,025 is 11.04 nats per token with a top-1 share of 0.142. That is the junk regime of Section 4, not a healthy scoring of text. The changes (+0.0066, +0.0074, +0.0066) are equal across the three arms, and the BF16C interval excludes zero by a small margin (+0.0005). INTERPRETATION: it is consistent with noise-size changes. It cannot show that long-context quality is kept, because the model is far from a calibrated predictor of this text. Do not cite it as support.

## 4. The raw-text diagnostic

FACT, new (my computation): I compared the E-55 FP8-dynamic arrays (H100, FA4) and the E-56 BF16 arrays (A100, Triton) on the same first 2,047 tokens.

| Slice | NLL FP8 / BF16 | Per-token correlation | Same top-1 token | Rank 1 FP8 / BF16 |
|---|---|---|---|---|
| B1 | 10.98 / 10.79 | 0.957 | 78.6% | 0.202 / 0.206 |
| P1 | 8.41 / 7.91 | 0.945 | 71.8% | 0.263 / 0.280 |
| C1 | 5.15 / 5.19 | 0.966 | 86.0% | 0.495 / 0.494 |

FACT: the top-1 tokens on B1 in BF16 are " own" (25%), " same" (13%), "//" (13%) and " even" (3%). These are the junk tokens of R-78. In B1, 46% of tokens have rank above 100, with a mean NLL of 17.8.
FACT: the log line says "FA4 not available, forcing TRITON_ATTN backend". So E-56 used Triton for all 60 layers, sliding and global.

Effect on the R-78 cause list (INTERPRETATION):
- B (FP8-dynamic checkpoint): ruled out as the main cause. Two different weight formats give the same token-by-token junk.
- C (FA4): ruled out, for global and sliding layers. The same junk appears on a different backend and GPU.
- A (instruct model on raw text) and D (a defect in the shared vLLM Gemma 4 path, including the K=V path): both stay open. The pipeline, the tokens and the BOS are the same in both runs. A gross D defect is unlikely, because chat scoring in the same engine is healthy (top-1 0.776) and needle retrieval works at 7.7K. A subtle D defect is not excluded.
- G (BOS, text-only mode): not excluded.
- README Section 13 says the failure "comes with raw text without a chat template". The data do not show that. The needle prompt tokens sit inside a chat user turn and still score 11.0 nats. INTERPRETATION: the junk goes with natural text that the model must predict (user content, documents), and not with the missing template. The GSM8K reference text (the assistant turn) scores well, but it is also highly constrained.
- HYPOTHESIS: A is the most likely cause. The test that separates A from D is an engine-independent score of 512 to 1,024 tokens of B1 with Hugging Face transformers on the BF16 weights. It needs a GPU, so it needs a new approval (see C8).

Effect on E-54 (INTERPRETATION): E-54 measured the attention-output error of the compact path against the engine's own tensors. That test does not need a healthy model. E-56 removes the FP8-checkpoint part of the caveat for BF16 results, and the model-level check agrees. The remaining caveat is narrower: the captured Q, K and V came partly from raw text, where the model is in an unusual regime, and the 8-bit compact rows (TIED8) have no model-level test at all. STOCK8 is not TIED8. Keep the caveat for TIED8 and for any E-54 claim about 8-bit rows. Do not extend it to the BF16 compact path.

## 5. Corrections and recommendations

C1. Grading note for G299 (README 5 and 13). BF16 gets credit for a truncated, looping answer by the last-number fallback. Under a strict `####` rule the change is -1 item, not -2. Report both numbers. Also state that 3 of the 6 final-answer changes (BF16C, BF16R, STOCK8) come from G299. Evidence: Section 1.1. The decision does not change.

C2. Disclose that A1 decided validity (README 13 and journal). Under f1b3502, V8 (NLL <= 1.5) fails at 1.7526 and the run stops. Evidence: `git show f1b3502` for the old bound; `summary.json` for the NLL. Write: "A1 B1 was made before any 31B data; without it the run would be INVALID." The old answer limits (A2) do not change the decision (Section 1.4).

C3. Fix the positive-control statement (README 13 "STOCK8 changes more than BF16C: not seen; both are at the noise level"). Say that the positive control did not separate. STOCK8 does change more text (25.9% against 21.5%) and has a larger per-token SD (item SD 0.86 against 0.63) but not the answers or the mean NLL. Restrict the claim as in Section 3, caveat 2.

C4. Replace "keeps the outputs" by "keeps the final answers" in the claim and the title of the run commit when you cite it. Add the text divergence (BF16C 21.5%, BF16R 17.4%, STOCK8 25.9%; C-only 42 against R-only 24, p = 0.036, exploratory). Without it, a reader will think the text is unchanged.

C5. Qualify H3. BF16R, the noise arm, has +0.00274 [+0.00205, +0.00344] and 262 of 400 items worse. It would not pass the GO3 margin of 0.002. BF16C has the opposite sign (-0.00091, interval excludes zero, 187 worse and 213 better). The margin is not calibrated against the noise arm, and BF16R is not a zero-mean A/A. Write that GO3 shows a small, slightly negative shift, and that the sign is unexplained (HYPOTHESIS: a chance result or a rounding detail of the compact query; not tested). The prediction "|Delta_BF16C| below 5e-4" failed and Section 13 already says so.

C6. Correct Section 10 point 7: GSM8K prompts are about 97 tokens (maximum 226), not about 300; sequences are about 240 tokens. State that GSM8K does not test the global layers beyond the sliding window.

C7. Correct Sections 4.1 and 10 point 4: the A100 ran Triton for all layers ("FA4 not available, forcing TRITON_ATTN backend"), not only for the global layers.

C8. Rewrite the raw-text paragraph of Section 13 using Section 4 above: the FP8 checkpoint and FA4 are ruled out as main causes by token-level agreement (correlation 0.95 to 0.97, the same junk tokens); "raw text without template" is not shown, since needle text in a user turn is also junk; A against D is open. Recommendation: before more E-54 or E-55 work on raw text, run a short Hugging Face reference score of B1 (about 1,000 tokens) on the same weights. This needs the user's approval and a balance check.

C9. Housekeeping in the README. The status line still says "Next: S4 (the user's spend approval...)". Section 13 mixes two bases: H1 numbers (BF16 0.9769, BF16C 0.9722) are over 432 items, and "Reported" numbers (0.975, 0.970) are GSM8K only. Label the base on each line. Add that the needle test is saturated (32 of 32 in all arms) as a caveat.

C10. E-54 caveat wording (journal and `RESEARCH_OUTPUTS.md`): keep it for TIED8 and for raw-text captures; drop the FP8-checkpoint reason for the BF16 compact path. Do not claim E-54 is "model-level confirmed" for 8-bit rows.

## 6. What I did not check
The 37 checksums, the 7 code hashes, the pod logs, the memory peak and the cost. The unit tests (S3) and the hook code in `splice_e55.py` beyond its header and arm definitions. Whether `k_norm` weights are the same in the two checkpoints. I found no sign of a problem in these items from the summary files.
