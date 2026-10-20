"""E-35 supplement, added after the R-32 post-run review. Not part of the frozen decision.

Part P (R-32 finding 4): paired sweep. Each scenario runs twice, and only L's prefix differs
(shared with H or not). Both runs draw the same random sequence. Counts how often H starts or gets
its first token later when the prefix is shared.

Part A (R-32 finding 3): label every T3 wait with reason "other". A wrapper on the scheduler's
`kv_cache_manager.allocate_slots` records each call for H that returns None (no blocks).

Runs inside the e35 image. Prints one JSON document.
"""

from __future__ import annotations

import collections
import json
import math
import random
import sys

sys.path.insert(0, "/code")

import harness as hz  # noqa: E402
import trace_check as tc  # noqa: E402

BASE_TRACING = tc.tracing


def instrumented(base):
    cls = BASE_TRACING(base)

    class Instrumented(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.e35_h_first_token = None
            self.e35_h_alloc_none: set[int] = set()
            original = self.kv_cache_manager.allocate_slots

            def allocate_slots(request, *a, **k):
                result = original(request, *a, **k)
                if request.request_id == "H" and result is None:
                    self.e35_h_alloc_none.add(self.e35_step)
                return result

            self.kv_cache_manager.allocate_slots = allocate_slots

        def update_from_output(self, out, runner_out):
            result = super().update_from_output(out, runner_out)
            if self.e35_h_first_token is None and "H" in runner_out.req_ids:
                if runner_out.sampled_token_ids[runner_out.req_ids.index("H")]:
                    self.e35_h_first_token = self.e35_step
            return result

    return Instrumented


tc.tracing = instrumented


def build(i, shared, fraction):
    rng = random.Random(f"E35S:{i}")
    prefix_len = 16 * rng.randint(16, 256)
    budget = rng.choice((64, 256, 1024))
    threshold = rng.choice((0, budget // 4))
    l_prompt, h_prompt = tc.two_party(
        rng, prefix_len, rng.randint(16, 512), rng.randint(16, 128), shared
    )
    l_arr = rng.randint(0, 2)
    chunk = budget if threshold == 0 else threshold
    h_arr = l_arr + rng.randint(0, math.ceil(len(l_prompt) / chunk))
    arrivals = {l_arr: [hz.make_request("L", l_prompt, hz.PRIORITY["L"], l_arr)]}
    for j in range(rng.randint(0, 4)):
        m_arr = rng.randint(0, 3)
        arrivals.setdefault(m_arr, []).append(
            hz.make_request(f"M{j}", hz.tokens(rng, rng.randint(256, 4096)), hz.PRIORITY["M"], m_arr)
        )
    arrivals.setdefault(h_arr, []).append(hz.make_request("H", h_prompt, hz.PRIORITY["H"], h_arr))
    reqs = [r for rs in arrivals.values() for r in rs]
    need = [tc.blocks_for(len(r.prompt_token_ids) + hz.MAX_TOKENS) for r in reqs]
    blocks = 10_000 if fraction is None else max(math.ceil(fraction * sum(need)), max(need) + 2) + 1
    return arrivals, budget, threshold, blocks


def part_p(n=600):
    counts = collections.Counter()
    later = []
    for i in range(n):
        fraction = random.Random(f"E35Sf:{i}").choice((None, None, 0.3, 0.45, 0.6, 0.8))
        res = {}
        for shared in (True, False):
            arrivals, budget, threshold, blocks = build(i, shared, fraction)
            trace, sched, finished, steps, h_first = tc.simulate(arrivals, budget, threshold, blocks)
            res[shared] = (h_first, sched.e35_h_first_token, finished)
        (fs, ts, fins), (fu, tu, finu) = res[True], res[False]
        counts["n"] += 1
        counts["both_finished"] += int(fins and finu)
        counts["first_" + ("equal" if fs == fu else "shared_earlier" if fs < fu else "shared_later")] += 1
        counts["token_" + ("equal" if ts == tu else "shared_earlier" if ts < tu else "shared_later")] += 1
        if fs > fu or ts > tu:
            later.append({"i": i, "fraction": fraction, "shared": res[True], "unshared": res[False]})
    return {"counts": dict(counts), "shared_later_cases": later}


def part_a(n=300):
    """Same scenarios as trace_check.t3 (same seed and draws), with allocation labels."""
    rng = random.Random(f"{tc.SEED}:T3")
    reasons = collections.Counter()
    for _ in range(n):
        prefix_len = 16 * rng.randint(16, 256)
        budget = rng.choice((64, 256, 1024))
        threshold = rng.choice((0, budget // 4))
        shared = rng.random() < 0.5
        l_prompt, h_prompt = tc.two_party(
            rng, prefix_len, rng.randint(16, 512), rng.randint(16, 128), shared
        )
        l_arr = rng.randint(0, 2)
        chunk = budget if threshold == 0 else threshold
        h_arr = l_arr + rng.randint(0, math.ceil(len(l_prompt) / chunk))
        arrivals = {l_arr: [hz.make_request("L", l_prompt, hz.PRIORITY["L"], l_arr)]}
        for j in range(rng.randint(0, 4)):
            m_arr = rng.randint(0, 3)
            m_prompt = hz.tokens(rng, rng.randint(256, 4096))
            arrivals.setdefault(m_arr, []).append(
                hz.make_request(f"M{j}", m_prompt, hz.PRIORITY["M"], m_arr)
            )
        arrivals.setdefault(h_arr, []).append(hz.make_request("H", h_prompt, hz.PRIORITY["H"], h_arr))
        reqs = [r for rs in arrivals.values() for r in rs]
        need = [tc.blocks_for(len(r.prompt_token_ids) + hz.MAX_TOKENS) for r in reqs]
        if rng.random() < 0.5:
            blocks = max(math.ceil(0.6 * sum(need)), max(need) + 2) + 1
        else:
            blocks = 10_000
        trace, sched, finished, steps, h_first = tc.simulate(arrivals, budget, threshold, blocks)
        for record in trace:
            reason = record.get("h_wait_reason")
            if reason is None:
                continue
            if reason == "other":
                reason = "other_h_alloc_none" if record["step"] in sched.e35_h_alloc_none else "other_unexplained"
            reasons[reason] += 1
    return {"reason_counts": dict(reasons)}


def main() -> None:
    print(json.dumps({"P": part_p(), "A": part_a()}, indent=1))


if __name__ == "__main__":
    main()
