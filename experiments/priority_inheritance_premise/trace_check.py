"""E-35 amendment A1: trace check for a shared-prefix wait in vLLM's scheduler.

Frozen design: README.md Section 10. Runs inside the e35 image. Prints one JSON document.
"""

from __future__ import annotations

import json
import math
import random
import sys

sys.path.insert(0, "/code")

import harness as hz  # noqa: E402  (request and scheduler helpers)
from tests.v1.core.utils import create_scheduler  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402

SEED = 20261019
MODEL = "/model"
MAX_SEQS = 16


def tracing(base):
    class Tracing(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.e35_step = 0
            self.e35_hits: dict[str, list[tuple[int, int]]] = {}
            self.e35_blocked: set[tuple[str, int]] = set()

        def _get_local_prefix_cache_hit(self, request):
            result = super()._get_local_prefix_cache_hit(request)
            self.e35_hits.setdefault(request.request_id, []).append((self.e35_step, int(result[1])))
            return result

        def _handle_blocked_waiting_request(self, request):
            ok = super()._handle_blocked_waiting_request(request)
            if not ok:
                self.e35_blocked.add((request.request_id, self.e35_step))
            return ok

    return Tracing


def blocks_for(prompt_tokens: int) -> int:
    return math.ceil(prompt_tokens / hz.BLOCK_SIZE)


def simulate(arrivals, budget, threshold, num_blocks, injected=False):
    """Run one scenario. arrivals: {step: [Request]}. Returns (trace, scheduler, finished, steps)."""
    cls = tracing(hz.InjectedWaitScheduler if injected else hz.CountingScheduler)
    sched = create_scheduler(
        model=MODEL,
        max_num_seqs=MAX_SEQS,
        max_num_batched_tokens=budget,
        enable_prefix_caching=True,
        long_prefill_token_threshold=threshold,
        num_blocks=num_blocks,
        block_size=hz.BLOCK_SIZE,
        max_model_len=32_768,
        skip_tokenizer_init=True,
        scheduling_policy="priority",
        scheduler_cls=cls,
    )
    all_reqs = [r for rs in arrivals.values() for r in rs]
    total = sum(len(r.prompt_token_ids) + hz.MAX_TOKENS for r in all_reqs)
    cap = 20 * (math.ceil(total / budget) + hz.MAX_TOKENS * len(all_reqs) + max(arrivals)) + 1
    l_prompt = next((len(r.prompt_token_ids) for r in all_reqs if r.request_id == "L"), None)
    h_arrival = next(s for s, rs in arrivals.items() for r in rs if r.request_id == "H")
    trace = []
    h_first_sched = None
    h_prefill_done = False
    step = 0
    while step <= cap:
        for request in arrivals.get(step, []):
            sched.add_request(request)
        sched.e35_step = step
        before = sched.e35_preemptions
        out = sched.schedule()
        scheduled = dict(out.num_scheduled_tokens)
        h = sched.requests.get("H")
        l_req = sched.requests.get("L")
        l_computed = l_req.num_computed_tokens if l_req is not None else l_prompt
        h_tokens = scheduled.get("H", 0)
        if h_tokens and h_first_sched is None:
            h_first_sched = step
        record = {
            "step": step,
            "h_tokens": h_tokens,
            "total": sum(scheduled.values()),
            "preempted": sched.e35_preemptions - before,
            "running": len(sched.running),
            "l_computed": l_computed,
            "h_in_deferred": h is not None and h in sched.deferred_waiting,
            "h_blocked": ("H", step) in sched.e35_blocked,
        }
        if step >= h_arrival and not h_prefill_done and h_tokens == 0:
            if record["h_in_deferred"] or record["h_blocked"]:
                reason = "skip"
            elif record["preempted"]:
                reason = "preempt_step"
            elif record["total"] >= budget:
                reason = "budget"
            elif record["running"] >= MAX_SEQS:
                reason = "seqs"
            else:
                reason = "other"
            record["h_wait_reason"] = reason
        trace.append(record)
        req_ids = list(scheduled)
        sampled = []
        for rid in req_ids:
            req = sched.requests[rid]
            sampled.append([7] if req.num_computed_tokens >= req.num_tokens else [])
        sched.update_from_output(
            out,
            ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
        if "H" in req_ids and sampled[req_ids.index("H")]:
            h_prefill_done = True
        step += 1
        if step > max(arrivals) and not sched.has_unfinished_requests():
            break
    finished = not sched.has_unfinished_requests()
    return trace, sched, finished, step, h_first_sched


def wait_reasons(trace):
    return [r["h_wait_reason"] for r in trace if "h_wait_reason" in r]


def two_party(rng, prefix_len=2048, l_suffix=64, h_suffix=32, shared=True):
    prefix = hz.tokens(rng, prefix_len)
    other = hz.tokens(rng, prefix_len)
    l_prompt = (prefix if shared else other) + hz.tokens(rng, l_suffix)
    h_prompt = prefix + hz.tokens(rng, h_suffix)
    return l_prompt, h_prompt


def t1(injected=False, arrivals_list=(1, 3, 5, 7, 10, 25, 40)):
    results = []
    for a in arrivals_list:
        rng = random.Random(f"{SEED}:T1:{a}")
        l_prompt, h_prompt = two_party(rng)
        arrivals = {0: [hz.make_request("L", l_prompt, hz.PRIORITY["L"], 0)]}
        arrivals.setdefault(a, []).append(hz.make_request("H", h_prompt, hz.PRIORITY["H"], a))
        trace, sched, finished, steps, h_first = simulate(arrivals, 256, 64, 10_000, injected)
        rec = trace[a]
        expected = 16 * (min(rec["l_computed"], 2048) // 16)
        hits = sched.e35_hits.get("H", [])
        results.append(
            {
                "a": a,
                "finished": finished,
                "h_first_scheduled_step": h_first,
                "h_hit_calls": hits,
                "expected_hit": expected,
                "l_computed_at_a": rec["l_computed"],
                "reasons": wait_reasons(trace),
                "l_in_prefill_at_a": rec["l_computed"] < len(l_prompt),
                "skip_while_l_in_prefill": any(
                    r.get("h_wait_reason") == "skip" and r["l_computed"] < len(l_prompt)
                    for r in trace
                ),
                "pass": finished
                and h_first == a
                and bool(hits)
                and hits[0] == (a, expected),
            }
        )
    return results


def t2():
    out = {}
    for name, shared in (("C1", True), ("C2", False)):
        rng = random.Random(f"{SEED}:T2")
        l_prompt, h_prompt = two_party(rng, shared=shared)
        arrivals = {
            0: [hz.make_request("L", l_prompt, hz.PRIORITY["L"], 0)],
            3: [hz.make_request("H", h_prompt, hz.PRIORITY["H"], 3)],
        }
        trace, sched, finished, steps, h_first = simulate(arrivals, 256, 0, 10_000)
        out[name] = {
            "finished": finished,
            "h_first_scheduled_step": h_first,
            "reasons_before_first": [
                r["h_wait_reason"] for r in trace if "h_wait_reason" in r and r["step"] < h_first
            ],
            "h_hit_calls": sched.e35_hits.get("H", []),
        }
    c1, c2 = out["C1"], out["C2"]
    out["pass"] = (
        c1["finished"]
        and c2["finished"]
        and c1["h_first_scheduled_step"] == c2["h_first_scheduled_step"]
        and len(c1["reasons_before_first"]) > 0
        and all(x == "budget" for x in c1["reasons_before_first"] + c2["reasons_before_first"])
        and c1["h_hit_calls"][:1] and c1["h_hit_calls"][0][1] == 2048
        and c2["h_hit_calls"][:1] and c2["h_hit_calls"][0][1] == 0
    )
    out["pass"] = bool(out["pass"])
    return out


def t3(n=300):
    rng = random.Random(f"{SEED}:T3")
    counts = {"skip": 0, "preempt_step": 0, "budget": 0, "seqs": 0, "other": 0}
    failures = []
    preemptions = 0
    tight_runs = 0
    for i in range(n):
        prefix_len = 16 * rng.randint(16, 256)
        budget = rng.choice((64, 256, 1024))
        threshold = rng.choice((0, budget // 4))
        shared = rng.random() < 0.5
        l_prompt, h_prompt = two_party(
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
        need = [blocks_for(len(r.prompt_token_ids) + hz.MAX_TOKENS) for r in reqs]
        if rng.random() < 0.5:
            num_blocks = max(math.ceil(0.6 * sum(need)), max(need) + 2) + 1
            tight_runs += 1
        else:
            num_blocks = 10_000
        trace, sched, finished, steps, h_first = simulate(arrivals, budget, threshold, num_blocks)
        reasons = wait_reasons(trace)
        for x in reasons:
            counts[x] += 1
        preemptions += sched.e35_preemptions
        if not finished or "skip" in reasons:
            failures.append(
                {"i": i, "finished": finished, "skip": "skip" in reasons, "budget": budget,
                 "threshold": threshold, "prefix_len": prefix_len, "num_blocks": num_blocks}
            )
    return {
        "n": n,
        "tight_runs": tight_runs,
        "reason_counts": counts,
        "preemptions": preemptions,
        "failures": failures,
        "pass": not failures,
    }


def main() -> None:
    result = {
        "T1": t1(),
        "T2": t2(),
        "T3": t3(),
        "T4": t1(injected=True, arrivals_list=(1, 5)),
    }
    t1_pass = all(r["pass"] for r in result["T1"])
    t4_pre = all(r["l_in_prefill_at_a"] for r in result["T4"])
    t4_pass = t4_pre and all(r["skip_while_l_in_prefill"] for r in result["T4"])
    if not t4_pass:
        decision = "INVALID"
    elif t1_pass and result["T2"]["pass"] and result["T3"]["pass"]:
        decision = "KILL"
    else:
        decision = "WEAK"
    result["assertions"] = {
        "T1": t1_pass,
        "T2": result["T2"]["pass"],
        "T3": result["T3"]["pass"],
        "T4": t4_pass,
        "T4_precondition": t4_pre,
    }
    result["decision"] = decision
    print(json.dumps(result, indent=1, default=str))


if __name__ == "__main__":
    main()
