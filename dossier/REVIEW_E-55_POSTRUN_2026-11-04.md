# R-78: post-run review of E-55 (hostile, second model)

Date: 2026-11-04. Reviewer: a different model from the author. Scope: the H100 run `results/e55-h100-20261103/`, the S5
rehearsal `results/e55-rehearsal2-4090-20261103/`, amendments A1 and A2. I changed no other file. I used no GPU, no
network and no `.env`. Scripts are in the session scratchpad (not in the repo).
Labels: FACT (I computed or read it), INTERPRETATION, HYPOTHESIS.

## Verdict: CONFIRM_INVALID_WITH_CORRECTIONS

The INVALID (V6) stands, and it does not depend on A2. The author's hypothesis is mostly right, but it is not complete.
The instrument cannot answer H16 or H8. The cause of the degraded baseline is not known. Three causes remain open (Section 3).

## 1. The INVALID (V6) and A2

- FACT: I recomputed `doc_sums` and `t_interval` from the raw arrays with the repo functions. E5M2: Delta +0.04420, t
  interval (-0.0375, +0.1259). This equals `decision.txt`. The percentile bootstrap (-0.0026, +0.0867) also includes 0.
- FACT: the pre-A2 rule (lower bound above 0) fails too, because the lower bound is -0.0375. So INVALID (V6) holds with the
  one-sided rule, the two-sided rule and the bootstrap. The outcome does not depend on A2.
- FACT: the A2 eager-check fix only moved V2 from a false failure to a pass. V1 to V5 and V7 have no problem entries.
  DET equals BF16 bit for bit on P1 and C1. So the engine is deterministic, and the noise below is not run-to-run noise.
- FACT: the code returns INVALID at V6 before it calls `decide16` and `decide8`. If V6 were waived, BF16C would be
  +0.0045 (-0.0302, +0.0391), and H16 would be INCONCLUSIVE16 by a factor of 30 in width against the 0.001 margin.
  STOCK8 is +0.0026 (-0.0216, +0.0269). TIED8 is -0.0336 (-0.0695, +0.0023). TIED8 "improves" the NLL by 13 times
  the STOCK8 effect. This is noise, not a result.

## 2. Test of the hypothesis

### 2.1 What the author got right
- FACT: BF16 mean NLL (scored tokens, index 1,025 or more): books 9.07, 9.58, 8.57, 10.48; prose 6.03, 6.35, 7.22, 5.93;
  code 4.71, 4.36, 4.12, 3.93. Top-1 accuracy: books 0.11 to 0.20, prose 0.31 to 0.39, code 0.52 to 0.58.
- FACT: E2B (rehearsal, first 16,384 tokens, same pipeline) gives books 5.2 to 6.5, prose 4.1 to 4.8, code 1.8 to 2.4,
  and top-1 accuracy 0.26 to 0.76. On the same first 16,383 tokens, the 31B is worse than E2B in 12 of 12 documents,
  for example B1 10.14 against 5.49, C1 5.09 against 2.30, C3 3.75 against 1.87.
  A healthy 31B model must not be worse than a healthy 2B model of the same family.
- FACT: BF16R (one BF16 rounding of q, relative 2^-9) changes the per-token log-probability with SD 1.665, median
  absolute change 0.373, and 25% of tokens move by more than 1 nat. Top-1 agreement 0.8009. BF16C gives SD 1.670 and
  0.8020. BF16R against BF16C gives SD 1.672, so BF16C and BF16R are independent draws of one noise distribution.
  E2B, same arm: SD 0.110, median 0.0115, 0.07% of tokens above 1 nat, agreement 0.985. The 31B is 15 times larger in
  SD and 32 times larger in median change. A1 predicted SD 0.02 to 0.11 and agreement 95% or more; both failed.
- FACT: even tokens with a BF16 NLL below 1 have a BF16R change SD of 0.98 (B1). Confident tokens are not stable.

### 2.2 What the author missed or must correct
- FACT (alignment is right, so a shift bug is ruled out): for B1, P1 and C1, the top-1 token at index j equals the
  document token j+1 in 19%, 36% and 52% of positions, and it equals tokens j-2 to j+3 (not j+1) in 0.4% to 3%. The
  parser checks that entry 0 is the prompt token (`run_e55.py` 90). `_rank == 1` agrees with `top1 == actual` in
  99.8% to 99.9% of positions (the rest are ties). The prompt log-probability is for the right token.
- FACT (not a chunk bug): NLL by offset inside a 4,096-token chunk, a 1,024-token chunk and a 64-token block is flat
  (B1 8.9 to 9.4, 9.0 to 9.2, 9.0 to 9.2). The V2 runner row chunks do not cause it. The first 3,000 tokens (one prefill
  chunk, inside or near the 1,024 window) are as bad as the rest (B1 10.99), so long-range global attention is not the
  cause. It is present from the first token.
- FACT: the "collapse" is partial and persistent. The junk tokens " same", "//", " even", " own" are top-1 at 37.3% of B1
  positions, 10.2% of P1 and 13.8% of C1, and at 0.09% and 0.03% in E2B. The fraction falls with the position in the
  document (B1 52% in the first 4,096 tokens, 22% in the last chunks). At the other positions the model reads the text:
  decoded top-1 on C1 is `env`, `mkdir`, `(`, `parents`, `=`, ` exist`, `_`, `.csv`, `write`, `_`, `text`, `TEL`, which
  matches the real code. A decoded B1 stretch is ` same that even even little and same same" same same same up
  describe the` against real prose about a Queen and an artist. At junk positions the actual token gets NLL 20 to 26
  (for example " client" after "knows session end only if the": top-1 " same", NLL 25.8). The model puts almost all mass
  on a wrong default token.
- FACT: junk is a position property, not a coin flip. Given junk in BF16, BF16R has junk at the same position in 87% to
  90% of cases; given no junk, in 1.6% to 5.8%. So the defect is systematic. The 1.67 SD comes from large logit
  shifts at about a quarter of positions, not from small shifts everywhere.
- FACT: PG-19 books use CRLF. The `<0x0D>` byte token is 6.9% of B1 tokens and has NLL 16 to 19 in both models. This
  raises the book NLL by about 1.1 nats in both models. It does not explain the 31B gap: the non-CR NLL is 9.7 against 4.7.
- FACT (correction): "worse than E2B" is shown only on the first 16K tokens of each document, because E2B was cut to
  16,384. I compared the same spans above. A full-length comparison does not exist.
- FACT (gap): the arrays store the log-probability of the actual token and the top-1 id and rank, not the top-1
  log-probability. Entropy, calibration and KL cannot be computed. This hides the key fact (how peaked is " same").

## 3. Alternative explanations

| # | Explanation | Evidence for | Evidence against | Status |
|---|---|---|---|---|
| A | Instruction-tuned 31B, raw text, no chat template, strong overconfidence after RL | Junk are generic function words; actual-token NLL 20 to 26; E2B-it (same template-free scoring) is sane; the E5M2 noise lowered E2B NLL in S5 (-0.0104), as noise flattens an overconfident model | E2B is also IT and also raw, and has 0.1% junk; 9 to 10 nats is far beyond usual IT raw-text loss; BF16R SD is 15 times E2B's | HYPOTHESIS, possible amplifier, not sufficient alone |
| B | FP8-dynamic checkpoint (per-token FP8 activations, channel weights) amplifies noise | A1 R-77 finding 9 predicted code flips; chaos is FP8-like; E2B is BF16 | RedHat FP8-dynamic is normally near BF16 quality; no BF16 or HF run exists in the repo to compare | HYPOTHESIS, untested |
| C | FA4 (all 60 layers, head 256 and 512) on H100 has a bug or large error in the sliding path (50 layers) or in prefill with 1,024 window | E2B used Triton on SM89, so backend is a confound; V5 only checks the 10 global layers on 16 rows of the first chunk | V5 passes in the global layers (0.0017 in E-54); GEN text is coherent over 3K prompt plus decode | HYPOTHESIS, untested for sliding layers |
| D | vLLM Gemma 4 K=V global path (`attention_k_eq_v`, v_norm without weight, partial RoPE) computes q, k or v wrongly against the HF reference | E-53 and E-54 compare the cache to the engine's own tensors, never to HF; E2B has no K=V, so the rehearsal could not see it | A wrong q/k/v would hurt the first 1,024 tokens too (it does, NLL 10.2) but GEN is coherent | HYPOTHESIS, untested; the hardest to see |
| E | Bug in V2 runner prompt log-probabilities | The V2 runner is new | Alignment, rank consistency, chunk-offset flatness and E2B on the same runner rule it out | RULED OUT (FACT) |
| F | Parser or data bug (shifted ids, tokenizer mismatch) | None | `tok_check` equal; parser checks the prompt token; the shift test shows only offset 0 matches; the decoded top-1 matches the real text content | RULED OUT (FACT) |
| G | Text-only mode (`limit_mm_per_prompt` 0) or a missing BOS | None | BOS is id 2 at index 0; E2B uses the same mode | Unlikely |
| H | Position effect of long context (global layers) | Junk falls with position | NLL is bad from token 1; flat in 4,096 chunks | RULED OUT as cause |

Read: the code path in `run_e55.py` 79 to 94 and 383 to 389, and the engine log (V2 runner, FLASH_ATTN with FA4, `norm_quant`
and `act_quant` fusions listed but the mode is NONE). The log has no sign of a fallback or a warning about the model.

## 4. Can the instrument ever answer the question?

- FACT: the target is 0.001 nats per token. The per-token SD of an arm change is 1.67. Assuming independent tokens, the
  standard error with 512K tokens is 0.0027, so the margin is below one SE. The document-level t interval is 0.035 wide
  on each side (13 times the iid value), because token errors are correlated inside a document.
- INTERPRETATION: a half-width of 0.0005 is 70 times narrower than 0.035. The half-width falls with the root of the
  document count, so you need about 4,900 times the documents if the clustering holds. A bigger run does not fix it. A paired design does not fix it either, because
  BF16R against BF16C is as large as each against BF16 (independent realizations).
- INTERPRETATION: the instrument works only if the baseline is stable. On a model like E2B (SD 0.11), the variance is about 230 times
  smaller, and the 0.001 margin may become reachable with the planned 12 documents (not proven).
- FACT: no validity check tests baseline health. V1 to V7 check the splice, not the model. V6 as defined (E5M2
  interval excludes 0) is a significance test of a noise effect, so it fails when the noise is large and passes when it is
  small; it cannot tell a dead splice from a noisy baseline. The V5 and V7 gap checks are the real splice evidence.

Cheapest decisive next check (HYPOTHESIS about what it shows; costs need the user's approval and a live balance check):
one H100 Pod session of about 35 min, about $2.5 to $4 at $3.99/h (cap $5), same vLLM pin, on B1, P1 and C1 first 4,096
tokens only (4 prompts of one chunk), and save the top-1 log-probability this time.
1. Reproduce: FP8 checkpoint, FA4, BF16 cache. Expect NLL near 10, 6, 5 (a control).
2. Same checkpoint with `TRITON_ATTN` (tests C; A and B do not change).
3. The doc inside a chat user turn and inside a model turn (tests A: the template).
4. BF16 weights of the same model, `google/gemma-4-31B-it` (62 GB, fits 80 GB at 8K context; tests B). If a base model
   `google/gemma-4-31B` exists, score it as well (tests A directly). I did not check that the name exists.
5. If steps 2 to 4 do not fix it: HF transformers BF16 eager on 2,048 tokens as the oracle (tests D).
Read: if 2 fixes it, C. If 4 (BF16 IT) fixes it, B. If only the base model or the template fixes it, A. If only the
HF oracle is sane, D (a vLLM Gemma 4 K=V defect). No GPU-free check can separate these four.

## 5. A2 and the S5 rehearsal

- FACT: A2 was made after the 4090 rehearsal and before 31B data, with the user's decision D-96. The eager-check fix is
  a plain bug fix, and `analyze_e55.py` 328 now accepts "0" and "NONE". I find no unfairness.
- Finding: the two-sided V6 is logically weaker than it looks. The rehearsal showed that E5M2 had a significant negative
  mean on E2B (-0.0104 [-0.0155, -0.0054]; books -0.0195, prose -0.0169, code +0.0050). INTERPRETATION: the sign flips with
  the domain, so the mean of E5M2 is not a clean "splice works" signal. The rehearsal was a warning that V6 tests noise, and
  the rule should have been replaced by a power check (for example, the BF16R SD and the per-token change of
  E5M2 above a floor, or the V5 and V7 gap evidence alone).
- Finding: the rehearsal did not look at absolute baseline quality (NLL 4.1 mean, 15% of tokens above 10 nats on E2B). A
  mean of 4.1 with 15% of tokens above 10 is itself high and was not flagged. The E2B rehearsal passed all 10 VERIFY items,
  but none of them asked "is the baseline sane?". The CRLF books (NLL 16 for 7% of tokens) add to this in both models.
- FACT: S5 could not test FA4 or the K=V global path (E2B, Triton, no K=V). The H100 pre-flight was the only FA4 check, and
  it checks 10 layers.
- FACT: the A1 prediction table (SD 0.02 to 0.11) came from R-77 and was HYPOTHESIS. The measured SD (1.67) refutes it, and
  the README Section 10 item 12 ("INCONCLUSIVE16 is possible for a noise reason") understated this by a factor of 15.

## 6. Implication for E-53 and E-54

- FACT: E-54 measured the attention-output error (relative to a float32 reference) of cache layouts and scales on the engine's
  own captured q, k and v at the 10 global layers, for 7 fresh prompts (4 plain, 2 chat, 1 32K), plus a kernel arm on the
  pinned FA4 kernel. Frozen result: R1 0.670, R2 1.131, PASS. R-76 already limited it to the attention level.
- INTERPRETATION: the PASS is arithmetic about those tensors. It does not need a good baseline, and a bad final NLL
  does not change it. The BF16 identity K = w V (float64 error 3e-15) holds structurally.
- INTERPRETATION (threat): the PASS carries a hidden premise: the captured q, k and v are what a healthy model produces. If
  the defect is in B, C or D (Section 3), then the activations that gave the scales (E-53 captures) and the sink and
  grid-phase behavior that R-74 found to decide the ratios (BOS sink, layer 41) come from a faulty or noisy forward pass.
  The tied scale (ks = w vs) was calibrated on those tensors. In case A, the tensors are healthy, and E-54 is untouched.
  The 2 chat prompts passing is weak support for A, because they were chat-format.
- Recommendation: keep the E-54 PASS as an attention-level result, label it "on the engine's tensors, baseline health unverified",
  and do not use it for a model-level claim. After Section 4, if B, C or D is the cause, repeat the E-54 capture on the healthy
  setup (about 1 H100 session, no new design) before any build.

## 7. Corrections and recommendations

1. FACT. Record the E-55 result as INVALID (V6) and state that the real cause is a baseline that is not stable, not an
   E5M2 control with low power. Evidence: Sections 1 and 2.1 (SD 1.67, BF16R and BF16C independent, DET identical).
2. FACT. Correct the author's hypothesis text: junk top-1 is partial and persistent (10% to 37% of positions; 87% to 90%
   stay junk across arms; falls with document position), and "worse than E2B" holds on the first 16K tokens only. Evidence: Section 2.2.
3. FACT. Do not run a larger E-55 (the README Section 9 "four times the tokens" plan). The interval needs about 4,900 times the documents
   under the same baseline. Evidence: Section 4.
4. Run the cause check of Section 4 (about $2.5 to $4, one session, after user approval and a live balance check) before
   any new model-level test. Save top-1 log-probability and entropy this time.
5. Add a baseline-health validity check (V8) to any new design: mean NLL of the BF16 baseline and the BF16R per-token SD
   below preset values taken from the check in item 4, before any decision arm.
6. Replace V6 by a power check. A one- or two-sided significance test on a mean that changes sign with the domain tests noise.
7. Run the E-54 recapture only if the cause is B, C or D. Until then, add the premise "engine tensors, baseline health
   unverified" to the E-54 and R-76 text in the journal (Section 8, as a scope note, no deletion).
8. Test the K=V path against the HF reference on a short prompt (item 5 of Section 4). E-53 and E-54 never compared
   the engine's global-layer q, k and v with the HF model.
9. The repo has no 31B quality check outside GEN (16 coherent summaries, 15 of 16 reach the 128-token limit). Add one to
   the Gemma 4 31B protocol, because the E-49 to E-55 work all assumes a healthy model.
10. Record that the rehearsal (E2B) answers splice mechanics only: it cannot see FA4, K=V or FP8-activation effects.

## Summary

1. Verdict: CONFIRM_INVALID_WITH_CORRECTIONS. The INVALID (V6) is right and does not depend on A2: the E5M2 interval is (-0.0375, +0.1259), the one-sided rule and the bootstrap give the same.
2. The author's hypothesis holds: baseline NLL 9 to 10 (books), 6 to 7 (prose), 4 to 5 (code); the 31B is worse than E2B on the same first 16K tokens in 12 of 12 documents.
3. Alignment is correct (only offset 0 matches; rank agrees with top-1; flat across chunk offsets), so a parser, shift or V2 runner chunk bug is ruled out.
4. The defect is persistent and present from token 1: junk top-1 at 37% (B1), 10% (P1), 14% (C1) against 0.1% in E2B; 87% to 90% repeat across arms; BF16R SD 1.67 against 0.11 for E2B.
5. Most likely cause: an unproven mix of the FP8-dynamic checkpoint with the IT model on raw text (A and B), with a backend or K=V path defect (C, D) not excluded. The FP8 checkpoint is my first suspect, because BF16 E2B is stable and this model amplifies a 2^-9 rounding 15 times more.
6. The instrument cannot resolve 0.001 nats: the margin is below one iid SE (0.0027), and the document-level half-width is 0.035; more data does not fix it (about 4,900 times the documents).
7. Threat to E-54: the PASS stands as arithmetic on the engine's own q, k and v, but the claim that those tensors are healthy is unverified (scales and sink behavior came from them). Low threat if the cause is A, real threat if B, C or D.
8. A2 is fair (made before 31B data, user decision D-96), but the two-sided V6 is a noise test, and no check ever asked whether the baseline was healthy.
9. Next check: one H100 session, about $2.5 to $4, on 4,096-token slices: FA4 against Triton, chat template, BF16 weights, base model if it exists, then an HF BF16 oracle.
10. Do not run a larger E-55; save top-1 log-probability next time and add a baseline-health gate.
