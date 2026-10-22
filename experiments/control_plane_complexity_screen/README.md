# E-37 Control-plane complexity screen (R-37)

Status: predeclared 2026-10-21 before the run. SCREEN ONLY. This is an engineering screen, not a
research result. It sizes two CPU-side costs that R-37 found in current source code.

## Sources read (2026-10-21)

- SGLang `a3d00eff58b6c22ea7539f882cc10c537634200b`, `python/sglang/srt/mem_cache/radix_cache.py`
  `evict()`: each call builds a list of all evictable leaves, computes a priority for each leaf with
  a Python method call, and runs `heapq.heapify`. The cost of one call is O(E), where E is the
  number of evictable leaves, even when the call frees only a few pages.
- vLLM `90c4a1a73eecea0428ee4303ad96c9fdeaac3f80`, `vllm/v1/core/kv_cache_utils.py`
  `hash_block_tokens()` and `vllm/utils/hashing.py` `sha256()`: each full block of the prompt gets
  `sha256(pickle.dumps((parent_hash, tuple(block_tokens), extra_keys)))`. The default algorithm is
  `sha256`. The engine core runs this in its input thread (`preprocess_add_request`), in the same
  Python process as the scheduler loop. Each agent turn hashes the full prompt again.

## Questions

- S1. What is the cost of one SGLang `evict()` call as a function of E, compared with a heap that
  is kept up to date between calls (lazy deletion)?
- S2. What is the cost of vLLM-style block hashing for one prompt as a function of prompt length?

## Predeclared criteria (heuristics, not literature standards)

- S1 is engineering-relevant if one call at E = 50,000 costs at least 1 ms on the M4 and the kept-up
  heap is at least 10 times cheaper. If not, drop S1.
- S2 is engineering-relevant if hashing a 131,072-token prompt costs at least 20 ms on one M4 core.
  If it costs under 5 ms, drop S2. Between 5 ms and 20 ms, record it as minor.
- Neither result can make a research gap. A positive result only lists an upstream PR candidate.

## What can make it wrong

- The M4 core is faster than many server cores, so the times are a lower bound for servers.
- S1 uses a stand-in node class with the same `last_access_time` field and the same priority call.
  It does not run SGLang itself. Real nodes carry more fields, so real calls are slower.
- S1 does not measure how often `evict()` runs. That depends on memory pressure and is not known.
- S2 copies the vLLM function exactly but does not import vLLM. `xxhash` is not installed, so only
  `sha256` (the default) and a cbor-free variant are timed.

## Run

    python3 -I experiments/control_plane_complexity_screen/screen.py

## Post-run explanation (rubber duck, written after the run on 2026-10-21)

Note: the pre-run text above lists the criteria and the risks. It does not state an expected number. This
section is written after the run.

1. S1 rebuild cost. The rebuild does three O(E) passes: copy the set to a list, call `get_priority` for
   each leaf, and `heapify`. The times grow almost linearly: 0.07 ms at E = 1,000, 0.88 ms at 10,000,
   4.6 ms at 50,000, 11.2 ms at 100,000. That is about 0.1 microseconds for each leaf, which fits Python
   tuple creation plus one method call.
2. S1 kept heap. Popping 4 leaves from a heap that already exists is O(4 log E), about 2 microseconds at
   every E. The ratio grows with E (50 to 5,060 times) because one side is linear and the other is
   logarithmic. The ratio is not the real saving: a kept heap also pays one O(log E) push each time a leaf
   is touched or created. That upkeep is spread over normal cache traffic and is not on the evict path.
3. S1 against the rule. 4.6 ms is above the 1 ms bar, and the ratio is above 10, so S1 passes. The
   unknown is the call rate. If `evict()` runs once in many steps, the cost does not matter.
4. S2. 8,192 blocks for a 131,072-token prompt take 7.2 ms, about 0.9 microseconds for each block (pickle
   of a 3-tuple plus SHA-256 of about 100 bytes). This is in the "minor" band (5 to 20 ms). It is
   smaller than the 0.5 s tokenization cost that vLLM RFC 47582 reports for a 200K-token history, so the
   tokenizer, not the hash, is the main front-end cost.
5. What the numbers do not show: real SGLang nodes, real call rates, server CPUs, and GIL contention with
   the scheduler loop.
