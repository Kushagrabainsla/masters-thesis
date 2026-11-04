# [RFC] Store the Gemma 4 global-layer KV cache once (`attention_k_eq_v`): 37.5% less global KV memory, read by the stock attention kernel

Status: DRAFT, kept in the repository, NOT POSTED (D-99, 2026-11-04). Posting it needs the user's approval. Before
posting, search the open vLLM PRs and issues again (L-09, L-10), and replace the private repository paths with public
numbers or a public summary.

## 1. Motivation

In the global (full-attention) layers of Gemma 4 31B and 26B-A4B, `attention_k_eq_v=True`. These layers have no
V projection:
- V = RMSNorm(W_k x), with no learned weight.
- K = rope_p(w * V), where w is the k_norm weight and p-RoPE (p = 0.25) rotates 128 of the 512 dims of each head.
- So K equals w * V on the 384 unrotated dims. Only the 128 rotated dims differ, because their rotation depends on the
  position.

The Gemma 4 technical report (arXiv 2607.02770) states that this design cuts the global KV cache by 37.5%. vLLM
stores the full K and the full V in these layers (1,024 values per KV head and token). A search on 2026-11-04 found
no engine that stores them once:
- vLLM PR #41944 (open) removes the duplicate V projection weights, but the cache keeps both tensors.
- vLLM PR #60674 and SGLang PR #25461 fuse compute for these layers.
- HF transformers, llama.cpp and TensorRT-LLM keep separate K and V caches.

Memory per request with a BF16 cache (arithmetic from the layout; the 50 sliding layers keep a fixed 0.8 GiB):

| Context | Today | Compact | Saved |
|---|---|---|---|
| 32K tokens | 3.3 GiB | 2.3 GiB | 0.94 GiB (29%) |
| 64K tokens | 5.8 GiB | 3.9 GiB | 1.9 GiB (32%) |
| 128K tokens | 10.8 GiB | 7.0 GiB | 3.8 GiB (35%) |

Where the KV pool limits the batch, 1.4 to 1.5 times the long requests fit on one GPU. With an FP8 KV cache the bytes
halve, and the percentages stay the same.

## 2. Proposal

Behind an opt-in flag, for Gemma 4 checkpoints with `attention_k_eq_v=True`:

1. Cache layout. One joint row per KV head and token, in the order [K_rot (128) | V_unrot (384) | V_rot (128)]: 640
   values instead of 1,024.
2. Read with the stock kernel. The K view is values 0 to 511 and the V view is values 128 to 639: two overlapping
   strided views of one row. The FA4 path accepts them as they are (the wrapper checks only `stride(-1) == 1`).
3. Query fold. The query takes the same dim order, and its unrotated part carries w. In the released checkpoints, w
   has one value on the unrotated dims of each layer, so the fold can go into the q_norm weight at load time, with no
   run-time cost.
4. Output order. The V view returns the dims as [unrotated | rotated]. A permutation of the o_proj input columns at
   load time restores the order. This is exact for BF16 weights. Block-quantized weights (block FP8, NVFP4) need a
   block-aligned permutation, a re-quantization or a run-time permute.
5. Cache write. A variant of the cache-write op writes the 640-value row from the k and v of the layer.
6. KV cache spec. The global layers use a spec for the joint row (`AttentionSpec` has a `head_size_v` field). The
   page size needs a plan: the compact global page is 3.2 times smaller than the sliding page, which is not an
   integer ratio, and the page unifier pads it, so the padded cache uses more memory, not less. A working plan makes
   the global page the largest (1,024 tokens per global block, or 5 sliding pages with a sliding block of 320 tokens).
   The scheduler block then becomes 5,120 tokens, and prefix-cache hits land on that unit (an open question).
7. FP8. In a shared FP8 row, the unrotated K dims take their rounding from V. At KV scale 1.0 this costs accuracy (up
   to 2.38 times the attention error of the standard FP8 cache in one layer). Tied per-layer scales (ks = w x vs) fix
   it at the attention level (0.67 times the stock FP8 error), but they need calibration to 0.1% to 0.25%. We propose
   BF16 first and FP8 as a follow-up.

## 3. Evidence

From our experiments (one H100 or one A100; vLLM 0.33.0rc1.dev229+g3966924d5):
- Exactness: the layout reproduces the standard attention with a float64 relative error of 3e-15 to 8e-15.
- Kernel: the unchanged FA4 FP8 kernel reads the two views with the same error. In a microbenchmark of the
  context-growing attention time (H100, FP8 KV, batch 11 to 19), it was 9.6% (32K) and 6.6% (64K) faster. Up to about
  1.6 points of this can be the SM clock.
- Speed limit: the context-growing FP8 attention time is about 80% on-chip work on the H100, so 37.5% fewer bytes cut
  it by only 4% to 10%. The value of the change is capacity, not step time.
- Model level: an emulated compact BF16 cache (hooks give the engine the values that the compact row returns) on
  `google/gemma-4-31B-it`, one A100, greedy decoding, 400 GSM8K problems and 32 needle prompts of 7.7K tokens. The
  accuracy is 0.9722 against 0.9769 (the change is -0.46 points, with a 95% interval of [-1.24, +0.32]). 2 of 432 final
  answers change, against 3 for a rounding-size change of the query. The NLL of the reference answers changes by
  -0.0009 nats per token.

## 4. Questions for the maintainers

1. Which KV cache spec API do you prefer for a joint K and V row: `head_size_v`, or a new spec class?
2. Is the page-size plan acceptable, and how does it interact with the prefix-cache block size?
3. Tensor parallelism: the global layers have 4 KV heads, so TP above 4 replicates the heads. Is that acceptable for a
   first version?
4. How should this interact with the FP8 KV paths for head size 512 (PRs #60372 and #53175), speculative decoding and
   the MTP drafter?
5. The 26B-A4B model uses the same design. Should the first version cover both models?

## 5. Limits

- No engine implementation exists yet. The memory numbers are arithmetic from the layout.
- The model-level test used an emulated cache, one checkpoint, greedy decoding and contexts up to 8K tokens.
- The kernel numbers come from a microbenchmark, not from end-to-end serving.
