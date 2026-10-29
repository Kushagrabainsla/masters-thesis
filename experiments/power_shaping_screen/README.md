# I-174 screen: can exact model transforms reduce operand bit transitions? (SCREEN ONLY)

Status: rule written 2026-10-29 before the run. Idea: I-174 (R-62). No GPU. Labels: FACT, INTERPRETATION, HYPOTHESIS.

## Question
GPU dynamic power depends on how many operand bits change from one tensor-core k-step to the next. Do exact,
function-preserving transforms of real LLM weights (reordering the K axis, plus sign flips where exact) reduce these
bit transitions enough to justify a GPU power test (E-47)?

## Method
- Weights: Qwen3-0.6B (BF16, from the local Hugging Face cache), layers 0, 13 and 27; q, k, v, o, gate, up, down.
- Codes: BF16 as stored, and FP8 E4M3 with one scale per 128 x 128 block (as block-FP8 checkpoints).
- Proxy: for W [N, K] (PyTorch layout), the mean number of bits that change between the codes at K positions p and
  p + s for the same row n (one MMA lane on consecutive k-steps). Lane stride s = 16 for BF16 (wgmma k16) and 32 for
  FP8 (k32). Stride 1 is also reported (neighbouring bytes in memory).
- K-axis orders:
  - orig: the checkpoint order.
  - random: a random permutation (control).
  - exact: the best order from moves that keep the model function: the MLP hidden units for down (permutation plus
    sign flips through up), the V dimensions inside each KV group for o (permutation plus sign flips), and one shared
    signed permutation of the residual basis for q, k, v, gate and up (optimised on all of them together, so it is
    the realistic case). Greedy nearest-neighbour chains on the sign and high bits, laid out so that chain neighbours
    are s apart.
  - exact_own: the same moves, optimised for one matrix only (an optimistic bound for the shared residual case).
  - sorted: each row sorted on its own (not a valid model): the floor that any ordering can reach.
- FP8 note: K moves across 128-blocks change block scales, so they are not bit-exact for an FP8 checkpoint.

## Rule (heuristic, frozen before the run)
R = transitions(order) / transitions(orig) at the lane stride. Take the median over the 7 matrix types (3 layers each).
- KILL the exact tier for GPU testing if the median R(exact) is 0.95 or more (under 5% fewer transitions). Reason:
  data-dependent power is at most about 40% of GEMM power between synthetic extremes (2409.18324), so under 5% fewer
  transitions gives about 2% power or less.
- PASS to E-47 if R(exact) is 0.90 or less on at least 2 matrix types, in BF16 or FP8.
- Otherwise WEAK: run E-47 only for the sorted bound and the tier-b (rounding) question.
- R(sorted) is the ceiling. If R(sorted) is above 0.80, the proxy says ordering of any kind has little room.

## Limits
- A proxy, not power. Real switching depends on the tile layout, operand paths and the multiplier design. A PASS only
  justifies the GPU test; it is not evidence of a power saving.
- One small model. Activations and token order are not in this screen.

## Files
`toggle_screen.py` (the screen), `results/screen.json` and `results/screen_stdout.txt` (output).
Run: `uv run --no-project --with numpy --with ml_dtypes python toggle_screen.py --model <model.safetensors> --out results/screen.json`
(about 30 s on an M4 Mac).

## Amendment A1 (2026-10-29, added AFTER the first result)
The first result left one exact move untested: a power-of-two scale on each K column (exact in BF16 because only the
exponent changes; the inverse scale goes into the RMSNorm weight or the paired matrix). A1 adds `exact_scaled_own`
(scales, then the per-matrix chain). The frozen rule is unchanged; A1 is reported next to it.

## Result (2026-10-29): KILL for the exact tier (frozen rule)
FACT (screen, Qwen3-0.6B, layers 0, 13, 27). Ratio of lane bit transitions to the checkpoint order (median per type):

| Type | random | exact | exact_own | exact_scaled_own (A1) | sorted (not valid) |
|---|---|---|---|---|---|
| BF16, all 7 types (median) | 1.00 | 0.995 | 0.985 | 0.984 | 0.278 |
| BF16 down_proj (best type) | 1.00 | 0.975 | n/a | 0.976 | 0.154 |
| FP8, all 7 types (median) | 1.00 | 0.998 | 0.988 | 0.992 | 0.175 |

- No type reaches 0.90. The median `exact` is 0.995 (BF16) and 0.998 (FP8). The best exact case cuts transitions by
  about 2.5% (down_proj); the per-matrix optimum (`exact_own`, optimistic) by about 1.5%.
- The toy exactness check (SwiGLU plus RMSNorm with the same moves) differs by 3.4e-13 (float64), so the moves are exact.
- `sorted` (each row sorted on its own; not a valid model) cuts transitions by 72% (BF16) to 82% (FP8). So the
  data-dependent space is large, but exact symmetries cannot reach it.

INTERPRETATION: one permutation (and one sign and scale per column) must serve all N rows of a matrix. For
near-random trained rows of length 1,024 to 3,072, nearest neighbours are only slightly closer than random pairs, so
ordering cannot make consecutive operands much more alike. This matches the first-principles estimate before the run.
Decision: do not run E-47 for the exact tier. Tier b (power-aware rounding, PowerPruning-style) changes the model and
is an accuracy trade-off; it stays WEAK and is not recommended (R-62).
