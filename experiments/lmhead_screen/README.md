# E-48 screen: can a low-precision LM head stay exact? (SCREEN ONLY)

Status: rule written 2026-10-29 before the run. Idea: I-181. No GPU. Labels: FACT, INTERPRETATION, HYPOTHESIS.

## Why
FACT (our E-36 and E-46 profiles, Qwen3-8B-FP8, 1 x H100): the LM head is a BF16 GEMM that reads 1.24 GB every
decode step. It takes 9.9% (batch 1), 9.3% (batch 8) and 6.1% (batch 64) of device time and already runs at about
3.0 TB/s (90% of peak), so only fewer bytes can make it faster. With FP4 bodies and a BF16 head (common practice),
the head is about a quarter of the bytes read at small batch. A lossy FP8 or FP4 head exists in community checkpoints,
but it changes the output (one study: +15% perplexity for an INT4 head on Qwen3-8B, 2608.02703).

## Question
If a cheap FP8 or FP4 copy of the LM head screens the vocabulary first, how many rows must be read again in BF16 to
keep the output exactly the same as the BF16 head?

## Two exact designs that this screen sizes
1. Certified screen (per step): z = E h is exact; z_hat = E_hat h is the screen. For every row,
   |z_i - z_hat_i| <= d_i = ||h|| * ||e_i - e_hat_i|| (Cauchy-Schwarz; ||e_i - e_hat_i|| is stored once). Read the K
   rows with the largest upper bounds z_hat + d again in BF16. Greedy is certified when the best exact logit in that
   set is at least the (K+1)-th largest upper bound. Temperature sampling stays exact with one rejection step for
   tokens outside the set (accept with probability exp((z - z_hat - d) / T)).
2. Speculative head (per few steps): choose tokens with the screen, check them later with one BF16 read for several
   positions (standard speculative sampling). Its acceptance rate is 1 minus the total variation distance between
   the screen and the exact distributions.

## Method
- Qwen3-0.6B (BF16, tied LM head, vocab 151,936, hidden 1,024) from the local Hugging Face cache, on the Mac CPU in
  float32. 24 short prompts (math, code, chat, facts, prose), greedy decode of 48 new tokens each, so about 1,150
  final hidden states h.
- Screens: FP8 E4M3 with one scale per row; FP4 E2M1 with one FP8 E4M3 scale per 16 values (NVFP4-like).
- Per step: K_greedy (design 1); tail mass outside the top 256 at T = 1.0 and 0.7 (design 1, sampling); argmax
  agreement and 1 - TV at T = 1.0 (design 2).
- Byte model at batch 1: screen bytes plus K BF16 rows, against V x d x 2 bytes.

## Rule (heuristic, frozen before the run)
- Design 1 GO: for FP8 or FP4, the median K_greedy is 256 or less and the 95th percentile is 2,048 or less, and the
  median tail mass outside the top 256 at T = 1.0 is 2% or less.
- Design 2 GO: for the FP4 screen, argmax agreement is 97% or more and the mean 1 - TV at T = 1.0 is 0.95 or more.
- KILL if neither design passes. A GO only justifies a GPU prototype plan with its own preregistration and review.

## Limits
- One small model on CPU; hidden states from greedy decoding of short prompts; bounds ignore the accumulation error of
  the screen GEMM (a small margin is added). A proxy for a GPU kernel, not a speed result.

## Files
`lmhead_screen.py`, `results/screen.json`, `results/screen_stdout.txt`.
Run: `uv run --no-project --with torch --with transformers --with ml_dtypes python lmhead_screen.py --model Qwen/Qwen3-0.6B --out results/screen.json`
(about 3 minutes on an M4 Mac; it needs the model in the local Hugging Face cache).

## Result (2026-10-29): GO for design 1 with the FP8 screen (frozen rule)
FACT (screen, Qwen3-0.6B, 1,152 greedy decode states; the hook state reproduces the model's own argmax in 100%):

| Screen | Bytes vs BF16 | K_greedy median | K_greedy p95 | Tail mass outside top 256, T=1.0 / 0.7 (median) | Argmax agreement | 1 - TV, T=1.0 | Bound / actual max error |
|---|---|---|---|---|---|---|---|
| FP8 E4M3, per-row scale | 0.50 | 2 | 14 | 1.6% / 0.016% | 98.0% | 0.982 | 7.2x |
| NVFP4-like (E2M1, FP8 block-16 scale, FP32 tensor scale) | 0.28 | 143 | 4,135 | 98.9% / 97.4% | 91.0% | 0.929 | 7.4x |

- Design 1 (certified) passes with FP8: re-reading about 2 to 14 BF16 rows keeps greedy decoding exact, and the tail
  mass is small enough for exact temperature sampling with one rejection step. Modelled LM-head bytes at batch 1: 0.50
  of BF16.
- FP4 fails design 1 (the bound is too loose against logit gaps) and design 2 (91% argmax agreement).
- Note: the first run emulated FP4 without the FP32 tensor scale (block scales underflowed); the rerun uses the
  NVFP4 two-level scale. The FP8 numbers did not change.

INTERPRETATION: the Cauchy-Schwarz bound is only about 7 times the real error here, much tighter than the worst case,
so an FP8 screen plus a handful of exact rows can halve the LM-head bytes with exact output. Expected decode gain is the
head share times about one half: about 5% for Qwen3-8B-FP8 at batch 1 to 8 on H100 (head share 9% to 10%), more for
FP4 bodies and small models with large vocabularies. Open questions for review: larger hidden sizes (4,096), the GPU
screen's own accumulation error, top-p and logprob outputs (logprobs need the exact normaliser), and prior work.
