# G1: page-size plan for the compact K=V global cache of Gemma 4 (I-182, gate G1 of D-88)

VERDICT: PASS. A plan exists that a patch can isolate, and it needs no change to the scheduler or to the KV cache planning code. A few cheap checks remain (Section 5).

Date: 2026-11-01. Author: gate G1 agent (Sonnet 5.5), asked by the main session.
Labels: FACT (I read code or a log, or I computed a number), INTERPRETATION (my reasoning), HYPOTHESIS (not tested).
Rules kept: I changed no repo file except this one. I did not read `.env`. I made no Runpod or cloud call. I posted nothing. I made only read-only `gh api` GET calls to github.com. I ran no GPU code and no vLLM code.

## Sources
- vLLM `3966924d566340e971a6da313e790d05492aa505` (2026-10-29, the image of E-49 to E-52) and `main` at `5d55ef1590c92213e3893dc8e8712f7f2e0bc508` (2026-11-01 08:02 UTC).
- The engine logs and `run.json` of E-49 and E-50 (same commit). They show the real specs, pages and pool sizes.
- Local copies of the fetched files (read only): `/private/tmp/claude-501/-Users-kushagrabainsls-Desktop-mscs-masters-thesis/31ab835c-8541-4e84-9cea-6ae0bd400f3b/scratchpad/g1/v39/` and `.../g1/vmain/`.
- Two scratch models: `.../g1/plans.py` (page unifier, groups, pool size) and `.../g1/sweep.py` (capacity against context length).
- The Gemma 4 31B shapes come from the engine pages, not from the Hugging Face config. FACT (arithmetic on the logged pages): 60 layers, 50 sliding (16 KV heads x 256) and 10 global (4 KV heads x 512), no KV sharing (6 groups x 10 layers). The sliding window of 1,024 is in the E-49 README and the model below reproduces the engine concurrency with it.

## Key findings
1. FACT: The page unifier sits in `vllm/v1/core/kv_cache_planning.py` (`unify_kv_cache_spec_page_size`, line 346), not in `kv_cache_utils.py`. The scheduler block size sits in `kv_cache_utils.py` (`resolve_kv_cache_block_sizes`, `math.lcm` at line 863).
2. FACT: I diffed 30 files that matter for KV planning, backends, the worker and Gemma 4. 22 are byte-identical at 3966924d5 and at main. The other 8 differ only in hunks that do not touch KV planning (prefix hash helper, HiSparse reservations, MiniMax and speculative config, a quantized-query flag, a CPU runner flag, data-parallel and speculative-decode code). `Gemma4Config`, `Attention.get_kv_cache_spec` and the unifier did not change. So every statement below holds for both commits.
3. FACT: R-72 is right. The compact FP8 page (163,840 B for 64 tokens) does not divide the sliding page (524,288 B). The ratio is 3.2. The unifier pads the compact page to 524,288 B. That gives 8,192 B per token, twice the 4,096 B of the stock FP8 global page. The pool then holds 0.57 times the stock number of requests at 32K, not 1.39 times.
4. INTERPRETATION: R-72 names one working plan (global block 1,024, sliding block 320, scheduler block 5,120). It works, but it is the weakest of the working plans. It loses capacity below 14.6K tokens and it makes prefix-cache hits land on 5,120 tokens.
5. FACT: The engine already has the pieces for a cheaper plan. A sliding layer can pad its page up to a shared page (`cache_config.skip_page_size_padded`, used by `Attention.get_kv_cache_spec`). Upstream uses this pattern for NVFP4 plus skip layers (`platforms/interface.py`, `_align_heterogeneous_kv_block_size`). A backend can own its packing (`AttentionBackend.customize_spec`, precedent in `TritonAttentionBackend`). With equal pages the unifier returns at once (line 370).
6. The smallest isolated patch is plan P3 for FP8 KV on H100 (pad the sliding page to the compact page) and plan B3 for BF16 KV and for Triton FP8 (pad the compact page to the sliding page, block 48). Section 3 gives the numbers and the files.

## 1. What vLLM builds today for Gemma 4 31B

FACT sources: E-50 `m31f8/run.json` and `log.txt` (FP8), E-49 `m31/run.json` and `log.txt` and E-50 `m31bf` (BF16). The code path is `Attention.get_kv_cache_spec` (attention.py:611), `customize_attention_spec` (worker/utils.py:476), `get_kv_cache_groups` (kv_cache_planning.py:1109), `unify_kv_cache_spec_page_size` (kv_cache_planning.py:346), `_get_kv_cache_groups_uniform_page_size` (kv_cache_planning.py:416) and `get_kv_cache_config_from_groups` (kv_cache_planning.py:642).

| Item | FP8 KV (E-50 m31f8) | BF16 KV (E-49 m31, E-50 m31bf) |
|---|---|---|
| Layout and kernels | LBNHC; sliding FA3 (multiple of 16), global FA4 SM90 FP8 dequant (kernel block exactly 64) | LBNHC; FA4 for all layers (multiple of 16) |
| Sliding spec | `SlidingWindowSpec`, block 64, 8,192 B/token, page 524,288 B | block 16, 16,384 B/token, page 262,144 B |
| Global spec before unify | `FullAttentionSpec`, block 64, 4,096 B/token, page 262,144 B | block 16, 8,192 B/token, page 131,072 B |
| Global after unify | block 128 (ratio 2), page 524,288 B | block 32 (ratio 2), page 262,144 B |
| Groups | 5 sliding x 10 layers, 1 global x 10 layers | same |
| Bytes per pool block | 5,242,880 (10 x 524,288) | 2,621,440 |
| Pool blocks (log) | 8,018 at 39.15 GiB | 16,096 at 39.3 GiB |
| Group sizes and scheduler block | (64,64,64,64,64,128), lcm 128 | (16,16,16,16,16,32), lcm 32 |
| Concurrency (log) | 8.59x at 67,584 tokens | 4.39x at 66,048 tokens |

How the unifier works (FACT, kv_cache_planning.py:370 to 408):
- If all pages are equal, it returns the specs unchanged.
- Else it takes the largest page. For each smaller page it checks `max_page % page == 0`. If true, it multiplies the block size by the ratio.
- If false and the layer is a non-MLA attention layer, it sets `page_size_padded = max_page`. The block size does not change.
- Mamba layers are always padded. MLA layers raise `NotImplementedError`.

How the block size of each side is chosen (FACT):
- The global spec takes `cache_config.block_size` (64 for FA4 FP8, from `update_block_size_for_backend`, platforms/interface.py:755; the log line is "Setting kv cache block size to 64").
- The sliding spec takes the largest kernel block whose page fits `page_budget` (attention.py:101 to 130 and 642 to 663). The budget is `skip_page_size_padded`, or `sliding bytes per token x cache_config.block_size` if it is unset.
- A padded page cannot be split into kernel blocks (`compute_layout_strides`, kv_cache_interface.py:396 to 399, an assertion). A dense page can (`create_kv_cache_views`, kv_cache_interface.py:442 to 452).
- The scheduler block size is the lcm of the group block sizes (kv_cache_utils.py:863). Without a Mamba group, prefix-cache hits land on this value (`partial_hash_hits_enabled`, kv_cache_utils.py:762 to 801; coordinator `_cache_hit_alignment_tokens`, kv_cache_coordinator.py:709).

Check of my model against the engine (FACT, computed):

| Quantity | Engine log | My model |
|---|---|---|
| FP8 pool blocks; concurrency; group sizes; lcm | 8,018; 8.59x; (64 x5, 128); 128 | 8,017; 8.59x; same; same |
| BF16 pool blocks; concurrency; group sizes; lcm | 16,096; 4.39x; (16 x5, 32); 32 | 16,097; 4.39x; same; same |
| Blocks per request at 32K and 64K (FP8, R-67) | 341 and 597 | 341 and 597 |

The one-block difference comes from the rounded memory in the log (39.15 and 39.3 GiB). I use the same pool memory for every plan, because the compact cache changes no weights.

## 2. What happens with a compact global spec and the current unifier

Compact bytes per token and layer for 31B: FP8 2,560 B (4 heads x 640 x 1 B); BF16 5,120 B. Both are 37.5% below the stock numbers.

How to express the spec (FACT):
- `head_size_v` works for the page math: `state_content_size_bytes = (head_size + head_size_v) x dtype size` (kv_cache_interface.py:585 to 590). But `Attention.forward` also uses `head_size_v` for the value view and the output shape (attention.py:532 and 543). So it would break the 512-wide V.
- The clean way is `customize_spec` with `state_content_bytes = 640 x dtype size`. `TritonAttentionBackend.customize_spec` (triton_attn.py:307 to 319) is a precedent. The model then passes prepared `k'` and `v'` (512 values each) to the unchanged `Attention` module.

Result with the current unifier (FACT for code path, computed for numbers):

| Case | What the code does | Bytes per global token | Requests in the pool at 32K and 64K |
|---|---|---|---|
| FP8, block 64 (default) | 524,288 % 163,840 = 32,768, not zero, so pad to 524,288. Scheduler block 64 | 8,192 (stock 4,096) | 13.4 and 7.2 (stock 23.5 and 13.4), 0.57x and 0.54x |
| FP8, `--block-size 128` | Sliding follows to 128. Compact page 327,680 B is padded. Kernel block 64 differs from 128 | n/a | engine fails at start (padded pages cannot be split) |
| BF16, block 16 | page 81,920 B, pad to 262,144 | 16,384 (stock 8,192) | 6.8 and 3.6 (stock 11.9 and 6.8), 0.57x and 0.54x |
| BF16, `--block-size 32` | page 163,840 B, sliding follows to 32 (524,288 B), pad | 16,384 | same as above |

INTERPRETATION: Without a change, the compact cache makes the engine worse than stock, not better. No integer block size b makes the compact page (2,560 b) divide the sliding page (2^19 B), because 2,560 has the factor 5. The reverse needs b to be a multiple of 1,024.

## 3. Plans that avoid the padding waste

Setup for the numbers: Gemma 4 31B, one H100, the E-50 pool (39.15 GiB for FP8, 39.3 GiB for BF16). Requests that fit = pool blocks / blocks per request. Blocks per request at context L = ceil(L / global block) + (sliding groups) x (ceil(1,024 / sliding block) + 1). The engine's own concurrency log (8.59x) and R-67 (341 and 597 blocks) match this formula. The ideal compact gain with no padding and the stock sliding allocation is 1.39x at 32K and 1.47x at 64K. This agrees with R-72 and E-50.

### 3.1 FP8 KV on H100 (global kernel block exactly 64, sliding FA3)

| Plan | Sliding block / page (B) | Global block / page (B) | Bytes per pool block | Scheduler block | Requests at 32K / 64K (stock 23.5 / 13.4) | Gain at 32K / 64K | Contexts with gain below 1 |
|---|---|---|---|---|---|---|---|
| P1 naive (current unifier) | 64 / 524,288 | 64 / padded 524,288 | 5,242,880 | 64 | 13.4 / 7.2 | 0.57 / 0.54 | all |
| P2 global 1,024, sliding x5 to 320 (R-72) | 320 / 2,621,440 | 1,024 / 2,621,440 | 26,214,400 | 5,120 | 28.1 / 18.0 | 1.20 / 1.34 | below 14.6K (worst 0.70x at 1K) |
| P3 pad the sliding page | 16 / padded 163,840 (natural 131,072) | 64 / 163,840 dense | 1,638,400 | 64 | 30.7 / 19.0 | 1.30 / 1.42 | below 5.6K (worst 0.87x at 1K) |
| P4 dense 80 and 256 | 80 / 655,360 | 64 scaled x4 to 256 / 655,360 | 6,553,600 | 1,280 | 32.4 / 19.7 | 1.38 / 1.47 | none |
| P5 packed groups (`VLLM_KV_CACHE_LAYOUT=BLNHC`) | 64, 17 groups of 3 layers (1,572,864 B used) | 64 / 163,840 | 1,638,400 | 64 | 32.0 / 19.5 | 1.36 / 1.45 | below 1.8K (worst 0.98x) |

What each plan needs (FACT from code unless marked):
- P3. Set `cache_config.skip_page_size_padded = 163,840`. The sliding spec then gets block 16 (`_largest_kernel_block_within` returns 16) and `page_size_padded = 163,840`. All pages are equal, so `unify_kv_cache_spec_page_size` returns at line 370. Groups stay 5 x 10 sliding plus 1 x 10 global. The sliding kernel block equals the manager block (16), so the split assertion does not fire. Sliding waste is 20% of the sliding page. P3 keeps 78% (32K) and 88% (64K) of the ideal gain. Hit alignment improves from 128 to 64 tokens.
- P4. The same hook with 655,360. The sliding block is 80. The unifier scales the compact global block 64 to 256 by ratio 4 (a dense page, so the 4 kernel blocks of 64 are legal). No padding waste. P4 keeps 97% and 99% of the ideal gain. Hit alignment becomes 1,280 tokens. INTERPRETATION: FA3 takes any page size on the local (sliding) path, because FA3 turns the paged TMA path off for local attention (`get_pagedkv_tma`, vllm-project/flash-attention b273a02, `hopper/flash_api.cpp` lines 464 to 475). This is not tested.
- P2. `customize_spec` sets the global block to 1,024. The unifier scales the sliding block by 5 to 320. No scheduler code changes, but the coarse blocks cost capacity and prefix caching.
- P5. Needs no spec padding. `get_kv_cache_groups` calls `_get_packed_kv_cache_groups` first (kv_cache_planning.py:1155) when the layout is block-outermost. It makes 17 sliding groups of at most 3 layers and 1 global group (anchor bytes 1,638,400). FACT: a block-outermost layout is chosen only by the env var or a backend that declares it (`resolve_kv_cache_layout`, backends/utils.py:302 to 399); FlashAttention declares none. INTERPRETATION: 18 groups make 17 separate `SpecGroup`s in the hybrid coordinator (specs differ by layer names, kv_cache_coordinator.py:718 to 775), and the FA kernels were never run on this layout in our tests. Treat P5 as a fallback.

### 3.2 BF16 KV, and FP8 KV with the Triton backend (kernel block any multiple of 16)

| Plan | Sliding block / page | Global block / page | Bytes per pool block | Scheduler block | Requests at 32K / 64K | Gain at 32K / 64K |
|---|---|---|---|---|---|---|
| B0 BF16 stock | 16 / 262,144 | 32 / 262,144 | 2,621,440 | 32 | 11.9 / 6.8 | 1.00 |
| B3 BF16 pad the compact page | 16 / 262,144 | 48 / padded 262,144 (natural 245,760) | 2,621,440 | 48 | 16.0 / 9.5 | 1.34 / 1.40 |
| T0 FP8 Triton stock | 16 / 131,072 | 32 / 131,072 | 1,310,720 | 32 | 23.8 / 13.5 | 1.00 |
| T2 FP8 Triton pad the compact page | 16 / 131,072 | 48 / padded 131,072 (natural 122,880) | 1,310,720 | 48 | 31.8 / 19.0 | 1.34 / 1.40 |

- The B0 numbers match the E-49 log. The T0 and T2 numbers come from my model only, because no engine log exists for Triton with FP8 KV.
- FACT: `customize_spec` can return `replace(spec, block_size=48, state_content_bytes=..., page_size_padded=...)`. The padded global page equals the sliding page, so the unifier returns at line 370. The global kernel block equals the manager block (48), so no split is needed. Waste is 6.7% of the compact page. There is no loss at any context length above 1K.
- FACT: for FP8 on SM90 the FA4 child needs exactly 64 (flash_attn.py:299 to 326 and 356 to 367). The composite backend intersects the sizes of its children (composite.py:118). So with FA in the composite the FP8 global block is 64, and T2 is not legal there. Use P3 or P4.
- HYPOTHESIS: FA4 BF16 head_dim 512 accepts a 48-token page. E-49 ran 32. This needs one start-up test.

### 3.3 Options that fail the gate

- Two global layers per page. FACT: in the default LBNHC layout every layer owns its own region, and one block id already spans all 10 layers of a group. The count of global layers is fixed at 10. Only the block-outermost layout lets groups with different layer counts share a block. That is P5.
- A separate KV cache group. FACT: the compact global layers are already their own group. The groups share one `BlockPool` (kv_cache_coordinator.py:120) and one backing allocation (worker/utils.py:579 to 581). Equal bytes per block is a hard rule.
- A separate pool with its own block size. INTERPRETATION: this changes `BlockPool`, the coordinator and admission. It is a scheduler change. It fails the gate.
- Sliding groups of 25 layers with block 16 and global block 128 (scheduler block 128, no waste). The equation holds (25 x 131,072 = 10 x 327,680). But `_get_kv_cache_groups_uniform_page_size` forces equal layer counts per group (line 521 to 532). It needs new grouping code and a non-layer-outer layout for mixed pages (`validate_kv_cache_layout`, line 608). It fails the gate.

### 3.4 Smallest isolated patch

P3 for FP8 on H100, and B3 for BF16. Patch scope:
- New code, needed in every plan: the views read and the cache write in `FlashAttentionImpl.forward` (flash_attn.py:1257) and `do_kv_cache_update` (flash_attn.py:1448), the same in `TritonAttentionImpl` (triton_attn.py:655 to 657 and 828), `customize_spec` for the compact spec, and the Gemma 4 layer code (`Gemma4Attention.forward`, gemma4.py:554 to 572; the o_proj column order; the query scale by the k_norm weight). Precedent for a subclassed backend: `flash_attn_diffkv.py`.
- P3 only: about 5 lines in `Gemma4Config.verify_and_update_config` (models/config.py:299 to 359) that set `cache_config.skip_page_size_padded`. Nothing resets this field later, because `update_block_size_for_backend` writes it only when skip layers are set (platforms/interface.py:938).
- B3 only: `customize_spec` of the compact global spec sets the block size 48 and the padded page. No config hook.
- No edit in `kv_cache_planning.py`, `kv_cache_utils.py`, `kv_cache_coordinator.py`, `block_pool.py`, `single_type_kv_cache_manager.py` or the scheduler.

Recommendation: build P3 first, with the pad constant as a parameter. Test P4 (constant 655,360) as the capacity-maximizing variant in the week-6 run, because it has no loss at any context length. Choose by the sliding-attention check in Section 5 and by whether prefix caching matters. P4 coarsens hits to 1,280 tokens. The E-50 style end-to-end test runs with prefix caching off.

## 4. Other engine risks the code shows

1. FACT: On Hopper with image inputs, the platform prefers the `TRITON_FLASH_ATTN` composite (platforms/cuda.py:170 to 174). It routes image-mask requests to Triton and causal requests to FA (composite.py:87 to 115). The Triton child writes the cache for both (`do_kv_cache_update` goes to `general_impl`, composite.py:268 to 270) and FA reads it. Both children must read the views, and the Triton child must also write them. HYPOTHESIS: `unified_attention` can read the views as is, because it takes separate K and V base pointers and strides (triton_unified_attention.py:229 to 236). No E-52 row tested it. The E-52 Triton kernels were our own.
2. FACT: The cache write needs no new kernel if the inputs are prepared. `reshape_and_cache_flash` takes K and V as separate views. It uses the K strides for both and checks `key_cache.stride(0) == value_cache.stride(0)` (cache_kernels.cu:979 to 983). `triton_reshape_and_cache_flash` does the same (triton_reshape_and_cache_flash.py:411 to 416). Pass `k' = [K_rot | V_unrot]` and `v' = [V_unrot | V_rot]`. Both writes put equal bytes into the 384 shared values (INTERPRETATION: the double write is benign).
   - This needs `k_scale == v_scale`. Else the shared region gets two scales, and FA4 dequantizes the whole K view with `k_descale`.
   - FACT: the default scales are 1.0 (E-50 notes: `kv_cache_scheme` is null). A calibrated checkpoint would break the trick.
   - Fallback: a small write kernel that writes the 640 values once.
3. FACT: E-52 tested decode with one query token and a "poisoned tail". It did not test prefill chunks (q_len above 1, 2,048-token chunks). The engine writes the chunk to the cache and then attends over the cache, so the views are used in prefill too. HYPOTHESIS: they work. It needs a test.
4. FACT: E-52 allocated its caches as (blocks, KV heads, 64, C), head-major (kbench_e52.py:250 to 251). The engine default is LBNHC, token-major (N before H). FA4 reads K and V as strided views with separate TMA descriptors (R-71 in E-52). The TMA needs 16-byte aligned base and strides. The V view starts at +128 B. The token stride is H x 640 B (2,560 for H = 4; 1,280 for H = 2; 640 for H = 1). The block stride is 163,840 B. All are multiples of 16. INTERPRETATION: legal. Retest on LBNHC.
5. FACT: FA4 needs no scheduler metadata (`aot_schedule` is on only for FA3, flash_attn.py:708). CUDA graph support is `ALWAYS` for FA 3 and 4 (flash_attn.py:619). The compact path adds elementwise ops inside the graph (query scale and reorder, `k'` and `v'` gather). The FA4 split cap of 32 for full graphs stays (R-67). `unified_kv_cache_update` is a splitting op.
   - FACT: the fused passes `fuse_rope_kvcache` and `fuse_qk_norm_rope_kvcache` write standard K and V rows. They are off in the E-50 log. Keep them off for compact layers.
6. FACT: `needs_kv_cache_zeroing` is true only for Mamba or mixed precision (kv_cache_interface.py:1652). Keep the compact spec's dtype and quant mode equal to the sliding spec, or block zeroing turns on.
7. INTERPRETATION: P3 changes the sliding layers to 16-token pages in FP8. FA3 local attention does not use the paged TMA path at any page size, and the BF16 arms of E-49 and E-50 already ran 16-token sliding pages (with FA4). Still, FA3 FP8 at 16 tokens is untested here. P3 also costs capacity below 5.6K tokens, because of the 20% sliding padding. Do not claim a gain for short contexts.
8. FACT: The ratio 3.2 holds for tensor parallel 1, 2 and 4. At 8 the global heads are replicated (1 head) and the ratio is 1.6 (still not an integer). Only the constants change. On SM100 the FA4 head_dim 256 kernel needs 128-token sliding pages (`FA4_HD256_PAGE_SIZE`, fa_utils.py:14 and 224 to 233), so the numbers change. The prototype targets one H100.
9. Not a G1 item: the upstream duplicate scan, the o_proj weight format and the accuracy on real activations are G2 and G3.

## 5. Verdict and conditions

PASS. Reasons:
- FACT: P3 (FP8, H100) and B3 (BF16) unify the pages through code that exists. The unifier returns early on equal pages. The group builder, the pool and the coordinator run as for stock Gemma 4.
- FACT: The patch is a backend hook, a few config lines and the attention and model changes the prototype needs anyway.
- FACT: The numbers beat the naive plan and the R-72 plan: 1.30x and 1.42x at 32K and 64K (P3), 1.34x and 1.40x (B3), against 0.57x for the naive plan.
- INTERPRETATION: The gain stays below the ideal (1.39x and 1.47x) by the padding. P4 closes most of the gap.

Conditions to carry into week 2 (cheap):
1. C1, CPU only, no spend: run `get_kv_cache_configs` with mock specs (the pattern of `tests/v1/core/test_kv_cache_utils.py`) for P3 and P4. Expect group sizes (16 x5, 64), lcm 64, about 25.6K pool blocks for P3, and (80 x5, 256), lcm 1,280, about 6.4K blocks for P4. If the output differs, G1 reopens.
2. C2, inside the G3 Pod session (about $0 extra): start the engine with the P3 flag and read the three log lines (group sizes, lcm, pool blocks). Time the sliding-attention kernels at block 16 against block 64 in FP8 for one decode step. Fail line: more than 5% slower. On a fail, use P4 or P5.
3. C3: if B3 is wanted, start FA4 BF16 head_dim 512 with 48-token pages once.

Stop rule from R-72 ("stop if G1 needs a scheduler change that you cannot isolate"): not triggered.

## What I could not check
- I ran no vLLM and no GPU code. The model in `plans.py` mirrors the unifier, the group rule and the pool formula. It matches six engine numbers (Section 1).
- I did not fetch the Hugging Face config. The shapes come from the engine pages.
- I did not run FA3 FP8 at 16 or 80 tokens, FA4 BF16 at 48 tokens, the Triton `unified_attention` on the views, or the BLNHC layout with FA.
- I did not scan upstream for a duplicate change (G2). The recent main commit titles (222 commits since 3966924d5) show nothing about Gemma 4 KV.
- The GitHub compare API listed only 300 files, so I diffed the named files one by one instead.
