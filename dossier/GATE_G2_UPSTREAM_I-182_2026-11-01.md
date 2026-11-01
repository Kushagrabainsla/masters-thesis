# Gate G2 for I-182: upstream re-scan, o_proj weight format and RoPE layout

Date: 2026-11-01 (the scan ran from about 03:20 to 04:15 PST). Gate: G2 of D-88. Idea: I-182.
Author: a Sonnet 5.5 sub-agent for the main session. Mode: read-only (GitHub reads, web search, Hugging Face reads).
This file is the only repo file that the agent wrote. The agent did not commit, push or post. It did not read `.env` or call Runpod.
Labels: FACT means the agent read a source. INTERPRETATION and HYPOTHESIS are the agent's own reasoning.
"I did not find X" means that a search returned no match. It does not prove that X does not exist.

## 1. Verdict: PASS

The three pass conditions hold.

1. No overlapping upstream work. I did not find a PR, issue or RFC, with activity from 2026-10-25 to 2026-11-01, that stores the Gemma 4 global K=V cache once or compactly. I did not find one that uses `head_size_v` for Gemma 4 or builds an MLA-like cache for it. Section 3 lists each related hit and why it does not overlap.
2. An exact o_proj path exists. The 31B checkpoint of E-49 and E-50 stores o_proj as FP8 with one scale for each output row. A per-head column permutation is exact for that format and for BF16. For 128-column block formats, a run-time permute is exact and costs about 0.1% of a decode step (INTERPRETATION). See Section 4.
3. The RoPE layout matches E-52. Dims 0..63 and 256..319 rotate, in neox rotate_half style. The k_norm weight scales the normalized projection before the rotation. See Section 5.

Four notes for the main session. None of them blocks the gate.

- N1: The E-52 README sentence on block-scaled o_proj (lines 249 to 250) is wrong for NVFP4 and incomplete for FP8 per-channel. Replacement text is in Section 4.8.
- N2: A third-party doc claims that vLLM already stores K=V once. The E-49 run record and the code show that it does not (Section 3.5).
- N3: Open vLLM PR #60674 and PR #41944 edit the same `gemma4.py` code that the prototype must change. Expect rebase work.
- N4: The upstream moves fast. 4,285 vLLM issues and PRs changed in 7 days. Repeat this scan before the week-6 stop.

## 2. Scope and method

Window: 2026-10-25 to 2026-11-01. An older item counts only if it had new activity in the window.

| Pass | Tool | Coverage | Result |
|---|---|---|---|
| P1 keyword search | `gh search prs` and `gh search issues`, `--updated ">=2026-10-25"` | 10 repos, 12 queries, 2 kinds (240 searches, 0 errors) | 17 unique items |
| P2 full lists | `gh api` on `repos/<repo>/issues?since=2026-10-25` | 14 repos, 12,424 items; titles filtered by regex; about 40 bodies read | Section 3 |
| P3 global search | `gh search` without `--repo` | 8 queries, 2 kinds | known items and third-party projects |
| P4 code search | `gh search code` | `attention_k_eq_v` on all of GitHub (first 100 files, 46 repos); `attention_k_eq_v` and `k_eq_v` in 7 named repos | Section 3.4 |
| P5 web search | WebSearch | 8 queries | no new work |
| P6 Hugging Face | HF API, discussions | 3 Gemma 4 repos, all 221 discussions | none on this topic |

I also read the engine model code at HEAD. HEAD commits at scan time (FACT):

| Repo | HEAD | Commit time (UTC) |
|---|---|---|
| vllm-project/vllm | 5d55ef159 | 2026-11-01 08:02 |
| sgl-project/sglang | c757c4bb7 | 2026-11-01 09:29 |
| NVIDIA/TensorRT-LLM | 5d86481f6 | 2026-11-01 06:20 |
| ggml-org/llama.cpp | dde879bc4 | 2026-11-01 06:46 |
| flashinfer-ai/flashinfer | e302d382b | 2026-11-01 11:42 |
| LMCache/LMCache | 0f6513d3a | 2026-11-01 09:28 |
| huggingface/transformers | ca8c871ed | 2026-10-31 23:09 |

The full search strings are in Appendix A.

## 3. Part 1: upstream re-scan

### 3.1 Summary

I did not find overlapping work. The closest items remove the duplicate V projection or fuse the norm and RoPE kernels. Each of them still writes both K and V to the cache.

### 3.2 Hits

Dates are created / last updated. "None" in the last column means no overlap with I-182.

Group A: Gemma 4 K=V items. Each one works on weights, projections or code structure.

| Item | Title | Dates | State | Overlap |
|---|---|---|---|---|
| vLLM #41944 | Gemma4 KV-projection optimization: remove redundant V projection on k_eq_v layers | 2026-05-07 / 2026-10-29 | open | None. FACT: the diff keeps `self.attn(q, k, v)` and sets `v_head_size=0` in `QKVParallelLinear`. It removes V weights only. |
| vLLM #61267 | [Bugfix][LoRA] respect `attention_k_eq_v=True` QKK LoRA layers | 2026-10-12 / 2026-10-30 | open | None. LoRA weights. See Section 4.7 for the o_proj LoRA case. |
| vLLM #60674 | [Perf][Gemma4] fused Triton kernels for QKV norm/RoPE, pre/post-FFN norms, PLE, MoE routing | 2026-10-08 / 2026-10-26 | open | None for the cache. Merge risk. FACT: the body names the global `k_eq_v` path 26 times. |
| vLLM #43227 | [Bugfix][Gemma4MoE] Fix AutoRound quantized Gemma4 MoE loading | 2026-05-20 / 2026-10-28 | open | None |
| TensorRT-LLM #19166 | Optimize Gemma4 shared K/V projection and FA2 preparation for RTX 6000 PRO | 2026-09-14 / 2026-10-28 | open | None. FACT: the body says "Compute shared K/V once for unquantized decode and prefill projections". The 4 changed files hold no KV-cache manager. |
| TensorRT-LLM #21339, #21255, #18537 | HF weight mapper, Gemma4 unified test, layer-aware config accessors | created 2026-10-30, 2026-10-29, 2026-09-01; all updated by 2026-10-30 | open, merged, closed | None |
| transformers #50189 | Split KV-sharing and rope attention variants into per-layer classes (gemma4: +k_eq_v) | 2026-10-31 / 2026-10-31 | open | None. A module refactor. |
| vllm-ascend fork #2 and #3 (Alex-stack-hub, not upstream) | Gemma4 A5 pipeline; RFC | 2026-10-24 / 2026-10-26 | open | None. FACT: it projects Q+K only and fuses norm, RoPE and the K/V cache write in one operator. K and V still go to the cache. |

Group B: lossy competitors (FP8 or NVFP4 KV cache at head_dim 512). They can stack with I-182.

| Item | Title | Dates | State | Overlap |
|---|---|---|---|---|
| vLLM #60372 | [Kernel] Integrate FA4 SM90 FP8 MMA for Gemma-4 full attention | 2026-10-07 / 2026-10-29 | open | None. See Section 3.3. |
| vLLM #62117 | [Attention] Support NVFP4 KV cache at head_dim 512 (Gemma 4) on SM100 | 2026-10-17 / 2026-10-30 | open | None |
| vLLM #63642 | [Perf][Kernel] Reduce shared-memory staging for per-tensor FP8 KV dequantization | 2026-10-26 / 2026-10-29 | open | None. It speeds the Triton FP8 path on Gemma 4 (head size 512). |
| SGLang #46965 | FA4 SM90 FP8 e4m3 KV cache for head_dim 512 (Gemma-4) | 2026-10-28 / 2026-10-28 | open | None |
| FlashInfer #8057 | validate-once launcher for the NVFP4 KV materializer | 2026-10-30 / 2026-10-31 | open | None. FlashInfer HEAD also has `tests/attention/test_gemma4_fa2_accuracy.py` (FP8 KV, head_dim 512). |
| flash-attention #199 | Add SM90 FP8 MMA forward for head dimension 512 | 2026-09-24 / 2026-10-29 | merged 2026-10-29 | None |

Group C: allocator and layout work. It matters for G1 and the build. It is not the same mechanism.

| Item | Title | Dates | State | Relevance |
|---|---|---|---|---|
| vLLM #64307 (issue), #64344 (PR) | packed KV-cache block stride for unequal per-group page sizes | 2026-10-28 / 2026-10-28 | open | G1. FACT: it packs groups with unequal page sizes into one slot. HYPOTHESIS: it could remove the padding (ratio 3.2) that R-72 named. Its numbers are synthetic or tiny-model. |
| vLLM #56156 | Pack heterogeneous KV-cache pages for hybrid models | 2026-09-09 / 2026-10-30 | open | G1 |
| vLLM #60996 | Size packed groups by pooled bytes per request, not layer padding | 2026-10-10 / 2026-10-26 | open | G1 |
| vLLM #64401 | [RFC] Large-page KV cache allocator: one KV cache group per spec | 2026-10-29 / 2026-10-29 | open | G1 |
| vLLM #61452 | Move KV cache allocation policy out of KVCacheSpecs (RFC #42449 part 2) | 2026-10-13 / 2026-10-31 | open | G1; it changes `KVCacheSpec` |
| vLLM #56996 | Source Triton metadata builder geometry from kv_cache_spec | 2026-09-15 / 2026-10-30 | open | Names Gemma 4 groups; no K=V |
| vLLM #64141 | [Bugfix][Attention] Avoid overwriting shared KV cache during scale view creation | 2026-10-27 / 2026-11-01 | open | Scale views of packed pages across hybrid groups; no K=V |
| vLLM #64572 | Triton and other eager backends re-derive K/V cache views on every call | 2026-10-29 / 2026-10-30 | open | The views read path; cost of view set-up |
| vLLM #64889, #64679, #64064, #61095 | `head_size_v` hits: rope-KV fusion, EAGLE geometry, MiMo-V2.6 on ROCm, DiffKV prefill | 2026-10-11 to 2026-10-31 | 3 open, 1 merged | None for Gemma 4 |

Group D: cross-layer KV sharing (Gemma 4 E2B and E4B) and sliding-window work. It is a different mechanism: vLLM #63553 and #64095; transformers #47290; llama.cpp #31406 (merged 2026-10-31); TensorRT-LLM #20971 and #18912. None of these stores K=V once inside one layer.

### 3.3 The two PRs that you named

| Field | vLLM #41944 | vLLM #60372 |
|---|---|---|
| State | open, `needs-rebase`, merge state CONFLICTING | open, merge state MERGEABLE |
| Created / last updated | 2026-05-07 / 2026-10-29T01:34Z | 2026-10-07 / 2026-10-29T17:49Z |
| Last code commit | 2026-06-11 (73e5a224d) | 2026-10-29 (3a01ead8e), 16 commits |
| New activity since 2026-10-25 | One mergify conflict notice on 2026-10-29 | See the list below |
| Changes the KV cache layout | No | No |
| Mentions `k_eq_v` or `K=V` | In its title | 0 matches in body, comments and reviews |
| Overlap | None | None |

New activity on #60372 (FACT):
- 2026-10-26: reviewer simon-veitner-redhat approved, then wrote "Not approving yet". An H200 check gave GSM8K 0.9697 (0.9719 with the merge-base FA).
- 2026-10-27: the same reviewer asked for a new end-to-end measurement.
- 2026-10-28: jhaotingc posted 1.04x output throughput on one H200 (796.5 to 831.1 tok/s). The author updated the description.
- 2026-10-29: a commit pinned the merged FlashAttention PR #199 (merged 2026-10-29 16:58Z).

The PR adds an FP8 guard in `flash_attn_diffkv.py` (+14 lines). INTERPRETATION: a prototype that uses the DiffKV backend cannot use an FP8 KV cache.

### 3.4 What the engines do today (FACT, HEAD sources)

| Engine | Where | What it does for `attention_k_eq_v` |
|---|---|---|
| vLLM | `gemma4.py` 89 to 134, 440 to 450, 556 to 573 | Loads K weights into the K and V shards of `qkv_proj`. Passes distinct k and v to `Attention`. |
| vLLM DSpark drafter | `gemma4_dspark.py` 53, 119 | Uses the flag, then calls `self.attn(q, k, v)`. |
| SGLang | `gemma4_causal.py` 1338 to 1347, 427 to 441 | Loads `k_proj` into both shards. `RadixAttention` has no `v_head_dim`. |
| TensorRT-LLM | `modeling_gemma4.py` 481 to 486, 588 to 592 | The weight mapper copies `k_proj` to `v_proj`. V gets `v_norm` before `k_norm`. |
| llama.cpp | `src/models/gemma4.cpp` 84, 259 to 281 | `Vcur = wv ? mul_mat : Kcur`. V gets a weightless norm. K gets `k_norm` and RoPE. Both go to `build_attn`. |
| FlashInfer, LMCache | code search | I did not find `attention_k_eq_v` or `k_eq_v` in either repo. |

Code search for `attention_k_eq_v` in vLLM lists only model files and tests. It lists no file under `vllm/v1`. INTERPRETATION: the allocator has no K=V special case.

### 3.5 Third-party code and one wrong claim

Code search found 46 other repos that mention `attention_k_eq_v` in the first 100 files. I read the matches for the ones that touch caches.

- EasyDeL (`esurge/core/interface.py`) uses the flag only to choose the KV head count in its cache spec. I found no single-tensor cache option there.
- tpu-inference (`gemma4.py` line 431) sets `v = k` and passes both to its attention call.
- quant.cpp copies K to V with `memcpy`. mlx-lm and cactus read the flag. I did not read their cache code.
- spiritbuun/buun-llama-cpp (a fork; MIT; last push 2026-10-28) has a comment on a "Coupled K==V cache" in `ggml/src/ggml-cuda/fattn.cu`. The comment names the DeepSeek-V4 latent and Gemma 4. FACT: its Gemma 4 graph (`src/models/gemma4.cpp` 257 to 269) and its drafter graph (`src/models/dflash.cpp` 991 to 1004) still build distinct K and V. INTERPRETATION: the coupled path serves latent caches. It has no compact row with a separate rotated part.
- CoreML-LLM docs (April 2026) record a "K=V alias" idea. They reject it for Gemma 4 E2B because that model has `attention_k_eq_v=false`.

The doc `noonghunna/club-3090/docs/KV_MATH.md` (lines 589 and 609 to 616; last commit 2026-09-17) says: "vLLM's allocator EXPLOITS this". The E-49 run record contradicts it. FACT: `experiments/gemma4_kv_audit/results/e49-h100-20261030/e49/m31/run.json`, `kv.groups[5]`, shows a `FullAttentionSpec` with block size 32 and page bytes 262,144 for each of the 10 layers. That equals 32 tokens x 4 heads x (512 + 512) values x 2 bytes. The page holds K and V. The doc also uses 16 KV heads for the global layers. The config says 4.

### 3.6 What I did not find

- I did not find an RFC or issue that asks for a compact or single-copy K=V cache for Gemma 4.
- I did not find a PR that sets `head_size_v` for Gemma 4, or that gives Gemma 4 an MLA-style cache.
- I did not find a new paper after 2026-10-25 on this layout. The nearest known works are GTA (2505.21487) and GVA (2609.13285) in journal Section 7.30. Web search returned no new work.
- I did not find Hugging Face discussions on this topic. I read all discussions of three repos: 144 for 31B-it, 70 for 26B-A4B-it and 7 for the 31B base model.

### 3.7 Limits of the scan

- P2 filters titles only. I read bodies for about 40 items. A PR with a generic title and no Gemma word could pass the filter.
- P1 and P3 search words in titles, bodies and comments, but the search index can lag by hours.
- Private repos and unpublished branches are invisible to this scan.
- Items from before 2026-10-25 with no later activity are out of the window. R-64 and R-69 covered them.

## 4. Part 2: the o_proj weight format

### 4.1 Checkpoints (FACT)

| Name in E-49 and E-50 | Hugging Face id | Revision | Size |
|---|---|---|---|
| m31 (deciding model) | `RedHatAI/gemma-4-31B-it-FP8-dynamic` | `d4ab4f579dd3516f97d8a6a4c98d0653480bad15` (pinned d4ab4f57) | 33.3 GB |
| m26 | `google/gemma-4-26B-A4B-it` | `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (pinned 4d7ae498) | 51.6 GB |

The HF API `sha` field matches both pins. The `config.json` files at `main` and at the pinned revisions are byte-identical.

### 4.2 Quantization metadata and tensor headers (FACT)

I read `config.json` and the safetensors headers with HTTP range requests. No weights were downloaded.

| Item | m31 (FP8-dynamic) | m26 (BF16) |
|---|---|---|
| `quantization_config` | compressed-tensors 0.14.1.dev28, format `float-quantized`, group `FP8_DYNAMIC` | none |
| Weights | 8-bit float, `strategy: channel`, symmetric, static, `block_structure: null`, `group_size: null` | BF16 |
| Input activations | 8-bit float, `strategy: token`, dynamic | not quantized |
| `ignore` | vision modules, `lm_head`, `embed_tokens` | not applicable |
| Global layers | 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 | 5, 11, 17, 23, 29 |
| `o_proj.weight` on each global layer | `F8_E4M3` [5376, 16384] | `BF16` [2816, 8192] |
| `o_proj.weight_scale` | `BF16` [5376, 1]: one scale for each output row | none |
| `v_proj` on global layers | absent (`attention_k_eq_v`) | absent |
| `k_norm.weight` on global layers | `BF16` [512] | `BF16` [512] |

The input size of o_proj is 32 heads x 512 = 16,384 for m31 and 16 x 512 = 8,192 for m26.

Note: `model.safetensors.index.json` of m31 lists 1,188 tensors and no `weight_scale` tensor. The shard headers list 1,598 tensors, including the scales. Use the headers.

Answer: o_proj is BF16 in m26. It is FP8 E4M3 with per-output-channel scales in m31. It does not use block scales, so there is no block shape.

### 4.3 The fold

Let `perm` be the V-view order inside one head: `UNROT + ROT`, that is dims 64..255, 320..511, 0..63, 256..319 (`compact_attn.py` lines 39 to 41). Let the stored weight be `W[:, h*512 + c]`. The folded weight is `W'[:, h*512 + j] = W[:, h*512 + perm[j]]` for each head `h`. With the attention output in the V-view order, `W' o' = W o`. This holds in real arithmetic.

### 4.4 Exactness by format (the rule and the result)

A block scale stays valid if each new block of `g` columns draws from one old block. The four segments of `perm` start at old dims 0, 64, 256 and 320 and at new positions 0, 192, 384 and 448. All of these are multiples of 64. So the fold keeps blocks whole when `g` divides 64. It breaks them when `g` is 128.

| Block size g | New blocks per head | New blocks that draw from 2 old blocks | Fold exact? |
|---|---|---|---|
| 16 (NVFP4) | 32 | 0 | yes |
| 32 | 16 | 0 | yes |
| 64 | 8 | 0 | yes |
| 128 | 4 | 3 | no |

FACT: I computed this table with a 5-line script (Appendix B). It also checks that `perm` is a bijection.

Results for the Gemma 4 checkpoints that I checked (FACT from the configs; INTERPRETATION where I read an ignore or exclude list):

| Checkpoint | o_proj format | Fold exact? |
|---|---|---|
| `RedHatAI/gemma-4-31B-it-FP8-dynamic` (E-49, E-50) | FP8, 1 scale per output row; activations FP8 per token | Yes. A row scale depends only on the set of values in the row, and a permutation keeps that set. The FP8 bytes move unchanged. The per-token activation scale also depends only on the set of values in the token vector. |
| `google/gemma-4-26B-A4B-it` (E-49) | BF16 | Yes |
| `google/gemma-4-31B-it` | BF16 | Yes |
| `nvidia/Gemma-4-31B-IT-NVFP4` | the `self_attn` of all 60 layers is excluded from NVFP4 | Yes. o_proj is not quantized (INTERPRETATION from the exclude list). |
| `RedHatAI/gemma-4-31B-it-NVFP4` | NVFP4, group 16, FP8 block scales | Yes, if scales move with their blocks |
| `google/gemma-4-31B-it-qat-w4a16-ct` | INT4, group 32, weight only | Yes, if groups move |
| `QuantTrio/gemma-4-31B-it-AWQ` | INT4, group 64, zero point | Yes, if groups, scales and zero points move. An AWQ repack is needed. |
| `RedHatAI/gemma-4-31B-it-FP8-block` | FP8, weight blocks 128 x 128; activation groups of 128 | No |

The GEMM sums in a new order. The result can differ in the last FP32 bits. This is below the BF16 output rounding (INTERPRETATION).

### 4.5 Toy check (FACT: I ran it; INTERPRETATION: toy data)

I ran a pure-Python float64 test with random Gaussian weights and outlier columns. It used 2 heads, 128 output rows and 4 tokens. The FP8 rounding was E4M3.

- Unquantized: the output difference after the fold is 0.0.
- FP8 per-channel plus per-token (the m31 format): the row scales are equal after the permutation. The permuted quantized weights equal the permuted original quantized weights, bit for bit. The activation scale is unchanged. The output difference is 0.0.
- FP8 block 64 and groups of 64: 0 of 131,072 elements change. The output difference is 0.0.
- FP8 block 128 and groups of 128, fold by requantization: 49.2% of the elements change their quantized value. The output moves by up to 3.5% (relative L2). The format's own error against unquantized is 4.0% on the same data. This is toy data. Real weights may differ.

### 4.6 The cheapest exact alternative for 128-column blocks

Permute the attention output back at run time, before o_proj. The o_proj weights stay as they are. The result equals the stock result.

Cost per token (INTERPRETATION; arithmetic, not measured):
- One global layer, one token: 32 heads x 512 values x 2 bytes = 32,768 bytes read and 32,768 bytes written.
- Ten global layers (31B): 655,360 bytes per token per step. At 3.0 TB/s (E-49 copy bandwidth) this is 0.22 microseconds per token.
- Batch 19 (E-50, 32K): 12.5 MB per step, which takes 4 microseconds. Add 10 small graph kernels at 2 to 5 microseconds each. The total is about 25 to 55 microseconds per step.
- The E-50 fit gives a step of 36.9 ms (32K, batch 19) and 35.2 ms (64K, batch 11). The permute costs about 0.07% to 0.16% of the step.
- HYPOTHESIS: a fused gather in the per-token FP8 quantization kernel of o_proj removes the extra pass. This needs a custom kernel.

Answer: the run-time permute adds about 0.22 microseconds of memory time per token and step. It adds 25 to 55 microseconds of launch time per step. The sum is below 0.2% of a decode step.

Requantization is not exact (Section 4.5). A layout with natural V order, `[V (512) | K_rot (128)]`, needs no permute. However, the E-51 fused kernel was 23% to 29% slower than its twin (E-52 README). So the views layout plus the fold stays the plan.

### 4.7 Build notes

- Apply the fold only to the global layers: 10 layers for 31B and 5 layers for 26B-A4B. Sliding layers keep the standard cache.
- Apply it at load time to the stored FP8 or BF16 tensor. Treat FP8 as `uint8` if the index op is not available for `float8_e4m3fn`.
- Tensor parallelism shards o_proj by whole heads, and the same permutation applies to every head. So the fold commutes with TP (INTERPRETATION).
- The fold cost at load time is a one-time move of 88 MB for each global layer on 31B (FP8).
- A LoRA adapter on o_proj of a global layer needs the same column permutation on its A matrix. vLLM PR #61267 shows that LoRA on K=V layers already needs special code.

### 4.8 Text to change (the main session decides)

E-52 README lines 249 to 250 (A1, R-71 O2) say that block-scaled FP8 or NVFP4 o_proj weights keep one scale per 128 columns. FACT: the NVFP4 configs say group 16. The m31 checkpoint has no column blocks. Proposed text:

"The o_proj fold is exact for BF16 and for FP8 per-channel weights (the E-49 and E-50 checkpoint). It is also exact for any block or group size that divides 64 (NVFP4 16, int4 group 32, group 64). It is not exact for 128-column blocks (FP8 block 128 x 128, group 128). For those formats, permute the attention output at run time."

## 5. Part 3: the RoPE layout and K/V loading in vLLM

All line numbers are from commit `3966924d5`. FACT: `gemma4.py`, `gemma4_rope.py`, `rotary_embedding/__init__.py`, `rotary_embedding/common.py` and `csrc/pos_encoding_kernels.cu` are byte-identical at main HEAD `5d55ef159`.

### 5.1 Which dims rotate, and how

| Fact | Source (line) |
|---|---|
| The config for global layers has `partial_rotary_factor` 0.25, `rope_theta` 1,000,000 and `rope_type` "proportional". | 31B and 26B `config.json`; `gemma4.py` 481 |
| `get_rope` sets `rotary_dim = int(512 x 0.25) = 128` and picks the proportional class. | `rotary_embedding/__init__.py` 72, 156 to 166 |
| `rope_angles = 128 // 2 = 64`. `nope_angles = 256 - 64 = 192`. | `gemma4_rope.py` 40, 42 |
| The class sets the base `rotary_dim` to 512, so the kernel applies the rotation to all 256 pairs. The nope pairs get the identity. | `gemma4_rope.py` 48 to 55 |
| `inv_freq[i] = 1 / (theta ^ (2i / 512))` for i in 0..63. The exponent uses the head size 512, not 128. `inv_freq[i] = 0` for i in 64..255. | `gemma4_rope.py` 64 to 77 |
| Neox style is on: `is_neox_style=True`. | `gemma4.py` 513 to 517 |
| The kernel pairs dim `i` with dim `i + 256`. It computes `x' = x cos - y sin` and `y' = y cos + x sin`. | `csrc/pos_encoding_kernels.cu` 16 to 17, 31 to 32, 49 |
| The cache has `rot_dim` = 512 columns, so `embed_dim` = 256. | `csrc/pos_encoding_kernels.cu` 154; `rotary_embedding/base.py` 102 |

Answer: the pairs (i, i + 256) for i = 0..63 rotate. These are dims 0..63 and 256..319, that is 128 of 512 dims. The other 384 dims pass through unchanged. The style is neox rotate_half (split halves), not interleaved. The attention scale is 1.0 (`gemma4.py` 419). The 31B config has no attention logit soft-cap key.

Cross-check (FACT): the Hugging Face code uses the same formula. `_compute_proportional_rope_parameters` uses `rope_angles = int(0.25 x 512 // 2) = 64`, a head-size denominator, and zero padding. `rotate_half` splits the halves (`modeling_gemma4.py` 772 to 777).

### 5.2 The role of the k_norm weight

- `self.k_norm = RMSNorm(head_dim, eps)` has a learnable weight (`gemma4.py` 451). The weight has 512 entries per layer and is shared by all KV heads of the layer.
- `self.v_norm = RMSNorm(head_dim, eps, has_weight=False)` has no weight (`gemma4.py` 453 to 455).
- In `forward`: `k = self.k_norm(k)` (565) runs before `rotary_emb(positions, q, k)` (567). `v = self.v_norm(v)` (570) has no RoPE.
- The HF `Gemma4RMSNorm` computes `norm(x) * weight`, not `norm(x) * (1 + weight)` (`modeling_gemma4.py` 205 to 221).
- So V = n and K = rope(w * n), with n the weightless RMS norm of the projection and w the k_norm weight. This is the E-52 identity.
- On the 384 unrotated dims, K[d] = w[d] * V[d]. So w can move into the query: q'[d] = q[d] * w[d]. On the 128 rotated dims, the rotation mixes dim i with dim i + 256 and depends on the position. So K_rot must be stored (128 values).
- INTERPRETATION: HF rounds `norm(x) * w` to BF16 before the rotation. The compact layout rounds V first and multiplies by w later. The two are equal in real arithmetic, not in BF16 bits. G3 must measure this.

### 5.3 How vLLM loads K and V for `attention_k_eq_v` layers today (FACT)

- The checkpoint has `k_proj` and no `v_proj` on the global layers (Section 4.2).
- The weights mapper sends `k_proj` to the shards `["k", "v"]` of `qkv_proj` (`gemma4.py` 100 to 110, 121). The same K weights fill both slots. The module comment says "V == K automatically" (440 to 441, 556).
- `QKVParallelLinear` allocates Q, K and V slots for all non-shared layers (442 to 450). So the duplicate V projection exists in the weights and in the GEMM. PR #41944 would remove it.
- `Attention(...)` gets no `head_size_v` (520 to 538). The default is `head_size_v = head_size` (`attention.py` 337).
- The cache page counts K and V: `(head_size + head_size_v) * dtype_size` (`kv_cache_interface.py` 590).
- Measured (E-49 run record, `kv.groups[5]`): `FullAttentionSpec`, block size 32, 262,144 bytes per layer. This equals 32 tokens x 4 heads x (512 + 512) values x 2 bytes. The page holds K and V at 512 values each.
- Hooks for the prototype: `AttentionSpec` has `head_size_v`, `page_size_padded`, `num_head_slots` and `state_content_bytes` (`kv_cache_interface.py` 560 to 568). A backend can change the spec in `customize_spec` (`attention.py` 644).

### 5.4 Match with E-52

`compact_attn.py` lines 36 to 42 set `ROT = [0..63] + [256..319]` and `UNROT` to the other 384 dims in ascending order. Its `rope()` helper (line 51) is neox rotate_half with zero frequency for pairs 64 and up, and theta 1,000,000. `pack_views` (line 83) stores `[K_rot | V_unrot | V_rot]`. `views_query` (line 89) puts q on the rotated dims and `q * w` on the others. All of this agrees with Section 5.1 and Section 5.2. The E-52 layout matches the engine.

## 6. Reasoning for the verdict

- Condition 1 holds on the evidence of Section 3. The confidence is medium to high for the window. The scan has the limits in Section 3.7.
- Condition 2 holds for every checkpoint that E-49 and E-50 used. It holds for BF16 and for all block sizes that divide 64. The one failing family (128-column blocks) has a cheap exact path.
- Condition 3 holds. I found no difference between the vLLM code and the E-52 helpers.
- Risks that remain:
  - R1: vLLM PR #60674 and #41944 change `gemma4.py`. The prototype must rebase on them.
  - R2: The allocator PRs of Group C may change `KVCacheSpec` before week 6. They may also remove the page padding of G1 (HYPOTHESIS).
  - R3: The club-3090 claim can mislead a reader. Cite the E-49 run record when you discuss capacity.
  - R4: BF16 bit differences of the compact layout are untested on real activations (G3).

## Appendix A: search strings

P1 (per repo, `--updated ">=2026-10-25" --limit 50`; prs and issues): `attention_k_eq_v`, `k_eq_v`, `gemma4 kv cache`, `gemma 4 kv cache`, `gemma4 global attention`, `gemma4 head_size_v`, `head_size_v`, `gemma MLA kv cache`, `gemma4 shared kv`, `gemma4 compact kv`, `gemma4 v_proj`, `gemma4 head_dim 512`.
Repos: vllm-project/vllm, sgl-project/sglang, NVIDIA/TensorRT-LLM, ggml-org/llama.cpp, flashinfer-ai/flashinfer, LMCache/LMCache, huggingface/transformers, google-deepmind/gemma, google-ai-edge/LiteRT-LM, ml-explore/mlx-lm.

P2 (`since=2026-10-25T00:00:00Z`, `state=all`): vllm-project/vllm (4,285 items), sgl-project/sglang (3,201), NVIDIA/TensorRT-LLM (916), ggml-org/llama.cpp (937), flashinfer-ai/flashinfer (1,498), LMCache/LMCache (474), vllm-project/flash-attention (8), huggingface/transformers (478), google-deepmind/gemma (3), google-ai-edge/litert-torch (67), google-ai-edge/LiteRT-LM (227), ml-explore/mlx-lm (38), ollama/ollama (287), google/gemma.cpp (5). ml-explore/mlx-vlm returned 404.
Title regexes: `gemma|k_eq_v|k=v|head_size_v|diffkv|mla|shared.?kv|tied.?kv|compact.?kv|unified.?kv|kv.?sharing|cross.?layer`; and a second pattern that needs a KV word (`kv`, `k/v`, `key value`) with `reduc|sav|footprint|memory|share|alias|dedup|compact|compress|halv|layout|page|hybrid|group|tied|same|single|once|capacity|head_size|geometry`.

P3 (no `--repo`, `--updated ">=2026-10-25" --limit 40`): `attention_k_eq_v`, `k_eq_v`, `gemma4 k_eq_v cache`, `gemma 4 tied K V cache`, `gemma4 single KV tensor`, `gemma4 MLA cache`, `gemma 4 global layers KV cache half`, `gemma4 head_size_v`.

P4: `gh search code "attention_k_eq_v"` (100 files, 46 repos). `gh search code "attention_k_eq_v"` and `"k_eq_v"` with `--repo` for vLLM, SGLang, TensorRT-LLM, llama.cpp, FlashInfer, LMCache and vllm-project/flash-attention. `gh search code` with `--repo spiritbuun/buun-llama-cpp` for `coupled K==V`, `k_eq_v`, `kv_coupled`, `v_is_k`, `is_coupled`.

P5 (web): `Gemma 4 attention_k_eq_v KV cache store once single cache K=V global layers inference engine`; `Gemma 4 global attention layers shared K and V cache memory saving vLLM SGLang llama.cpp pull request November 2026`; `Gemma 4 MLA-like compact KV cache k_eq_v p-RoPE "37.5%" KV cache implementation`; `"attention_k_eq_v" github pull request kv cache`; `"k_eq_v" Gemma 4 "KV cache" halve global layers one buffer RFC issue vllm sglang`; `Gemma 4 31B KV cache memory global layers keys equal values deduplicate cache engine optimization`; `Gemma 4 global attention K=V p-RoPE store V plus rotated key part only compact cache 640 values MLA-style exact inference`; `arXiv 2026 tied key value cache RoPE decoupled "k_norm" absorbed into query Gemma 4 engine implementation KV cache halved`.

P6: `https://huggingface.co/api/models/<repo>/discussions?p=0..2` for google/gemma-4-31B-it, google/gemma-4-26B-A4B-it, google/gemma-4-31B (7 discussions).

## Appendix B: the block check (FACT: output shown)

```python
ROT = list(range(0, 64)) + list(range(256, 320))
UNROT = [d for d in range(512) if d not in set(ROT)]
PERM = UNROT + ROT
for g in (16, 32, 64, 128):
    print(g, max(len({PERM[p] // g for p in range(n, n + g)}) for n in range(0, 512, g)))
# 16 1 / 32 1 / 64 1 / 128 2   (for g = 128 the per-block counts are [2, 2, 1, 2])
```

The toy FP8 test script and the downloaded sources are in the agent scratchpad. They are not in the repo.
