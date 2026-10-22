"""E-37 screen: CPU cost of two control-plane paths found in R-37. See README.md."""

import hashlib
import heapq
import json
import pickle
import platform
import random
import statistics
import sys
import time


def best_ms(fn, repeats=7):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    return min(times), statistics.median(times)


# S1: SGLang-style evict() against a heap kept up to date between calls.
class Node:
    __slots__ = ("last_access_time", "alive")

    def __init__(self, t):
        self.last_access_time = t
        self.alive = True

    def __lt__(self, other):
        return self.last_access_time < other.last_access_time


class LRUStrategy:
    def get_priority(self, node):
        return node.last_access_time


def sglang_style_evict(evictable_leaves, strategy, k):
    # Same shape as radix_cache.py evict(): rebuild the heap from all leaves on each call.
    leaves = list(evictable_leaves)
    heap = [(strategy.get_priority(n), n) for n in leaves]
    heapq.heapify(heap)
    out = []
    while len(out) < k and heap:
        out.append(heapq.heappop(heap)[1])
    return out


def kept_heap_evict(heap, k):
    # Heap kept between calls; a leaf touched since its push is skipped (lazy deletion).
    out = []
    while len(out) < k and heap:
        prio, n = heapq.heappop(heap)
        if n.alive and n.last_access_time == prio:
            out.append(n)
    return out


def screen_s1():
    rows = []
    strategy = LRUStrategy()
    for e in (1_000, 10_000, 50_000, 100_000):
        rng = random.Random(0)
        nodes = [Node(rng.random()) for _ in range(e)]
        leaves = set(nodes)
        k = 4  # a small eviction: a few pages for one decode step
        rebuild = best_ms(lambda: sglang_style_evict(leaves, strategy, k))
        heap = [(n.last_access_time, n) for n in nodes]
        heapq.heapify(heap)

        def kept():
            # Pop k, then push them back so every repeat sees the same heap size.
            got = kept_heap_evict(heap, k)
            for n in got:
                heapq.heappush(heap, (n.last_access_time, n))

        kept_t = best_ms(kept, repeats=21)
        rows.append({
            "E": e,
            "rebuild_ms_min": round(rebuild[0], 4),
            "rebuild_ms_median": round(rebuild[1], 4),
            "kept_heap_ms_min": round(kept_t[0], 5),
            "ratio": round(rebuild[0] / max(kept_t[0], 1e-6), 1),
        })
    return rows


# S2: vLLM-style chained block hashing of one prompt.
NONE_HASH = hashlib.sha256(b"none").digest()


def vllm_sha256(obj):
    return hashlib.sha256(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).digest()


def hash_prompt(tokens, block_size=16, hash_fn=vllm_sha256):
    parent = NONE_HASH
    out = []
    for i in range(0, len(tokens) - len(tokens) % block_size, block_size):
        parent = hash_fn((parent, tuple(tokens[i:i + block_size]), None))
        out.append(parent)
    return out


def screen_s2():
    rows = []
    rng = random.Random(1)
    for n in (8_192, 32_768, 131_072, 204_800):
        tokens = [rng.randrange(151_000) for _ in range(n)]
        t = best_ms(lambda: hash_prompt(tokens), repeats=5)
        rows.append({"prompt_tokens": n, "blocks": n // 16,
                     "sha256_pickle_ms_min": round(t[0], 2),
                     "sha256_pickle_ms_median": round(t[1], 2)})
    return rows


if __name__ == "__main__":
    result = {
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "platform": platform.platform(),
        "S1_evict": screen_s1(),
        "S2_block_hash": screen_s2(),
    }
    print(json.dumps(result, indent=2))
