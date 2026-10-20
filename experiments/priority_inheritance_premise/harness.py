"""E-35 Stage A harness: drive the real vLLM Scheduler with a fake model runner.

Runs inside the e35 image (Python-only vLLM at the pinned commit). One trial = one scheduler,
one workload, one condition. The metric is H's time to first token in scheduler steps.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, "/opt/vllm")  # vLLM's own test helpers (tests/v1/core/utils.py)

from tests.v1.core.utils import create_scheduler  # noqa: E402
from vllm.sampling_params import SamplingParams  # noqa: E402
from vllm.utils.hashing import sha256  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.core.sched.scheduler import Scheduler  # noqa: E402
from vllm.v1.outputs import ModelRunnerOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402

BLOCK_SIZE = 16
MAX_TOKENS = 4
EOS_TOKEN_ID = 50256
PRIORITY = {"H": 0, "M": 5, "L": 10}
CONDITIONS = ("C0", "C1", "C2", "C4")

init_none_hash(sha256)
BLOCK_HASHER = get_request_block_hasher(BLOCK_SIZE, sha256)


class CountingScheduler(Scheduler):
    """The unmodified vLLM scheduler, plus a counter of preemptions."""

    e35_preemptions = 0

    def _preempt_request(self, *args, **kwargs):
        self.e35_preemptions += 1
        return super()._preempt_request(*args, **kwargs)


class InjectedWaitScheduler(CountingScheduler):
    """Positive control: a waiting request with no computed tokens waits while an unfinished
    request that admitted earlier shares its first full block and has not finished its prefill.
    The owner may be running, preempted or waiting. This emulates singleflight prefill."""

    def _handle_blocked_waiting_request(self, request: Request) -> bool:
        if not super()._handle_blocked_waiting_request(request):
            return False
        if request.num_computed_tokens or not request.block_hashes:
            return True
        first = request.block_hashes[0]
        for other in self.requests.values():
            if other is request or not other.block_hashes:
                continue
            if other.block_hashes[0] != first:
                continue
            started = other.num_computed_tokens > 0 or other.num_preemptions > 0
            if started and other.num_computed_tokens < other.num_prompt_tokens:
                return False
        return True


@dataclass(frozen=True)
class Cell:
    prefix_len: int
    budget: int
    f: float
    n_m: int
    blocks: str  # "ample" or "tight"
    threshold: str  # "0" or "quarter"


@dataclass
class TrialResult:
    cell: dict
    condition: str
    control: str  # "vllm" or "pc"
    trial: int
    h_arrival: int
    h_first_token_step: int | None
    h_ttft_steps: int | None
    total_steps: int
    prefill_tokens: int
    preemptions: int
    finished: bool


def grid() -> list[Cell]:
    return [
        Cell(p, b, f, m, k, t)
        for p, b, f, m, k, t in itertools.product(
            (512, 2048, 8192), (256, 2048), (0.0, 0.25, 0.5, 0.9), (0, 4),
            ("ample", "tight"), ("0", "quarter"),
        )
    ]


def tokens(rng: random.Random, n: int) -> list[int]:
    return [rng.randrange(1000, 30000) for _ in range(n)]


def make_request(rid: str, prompt: list[int], priority: int, arrival: int) -> Request:
    params = SamplingParams(ignore_eos=True, max_tokens=MAX_TOKENS)
    params.update_from_generation_config({}, EOS_TOKEN_ID)
    return Request(
        request_id=rid,
        prompt_token_ids=prompt,
        sampling_params=params,
        pooling_params=None,
        arrival_time=float(arrival),
        priority=priority,
        block_hasher=BLOCK_HASHER,
    )


def workload(cell: Cell, trial: int, seed: int):
    """Return the prompts and H's arrival step for one trial. Same for every condition."""
    rng = random.Random(f"{seed}:{asdict(cell)}:{trial}")
    prefix = tokens(rng, cell.prefix_len)
    other_prefix = tokens(rng, cell.prefix_len)
    l_suffix = tokens(rng, rng.randint(16, 512))
    h_suffix = tokens(rng, rng.randint(16, 128))
    m_prompts = [tokens(rng, 2 * cell.prefix_len + rng.randint(16, 128)) for _ in range(cell.n_m)]
    chunk = cell.budget if cell.threshold == "0" else max(BLOCK_SIZE, cell.budget // 4)
    l_solo_steps = math.ceil((cell.prefix_len + len(l_suffix)) / chunk)
    h_arrival = round(cell.f * l_solo_steps) + rng.randint(0, 2)
    return prefix, other_prefix, l_suffix, h_suffix, m_prompts, h_arrival


def build_requests(cell: Cell, condition: str, trial: int, seed: int):
    prefix, other_prefix, l_suffix, h_suffix, m_prompts, h_arrival = workload(cell, trial, seed)
    arrivals: dict[int, list[Request]] = {}
    if condition != "C0":
        l_prompt = (prefix if condition in ("C1", "C4") else other_prefix) + l_suffix
        l_priority = PRIORITY["H"] if condition == "C4" else PRIORITY["L"]
        arrivals.setdefault(0, []).append(make_request("L", l_prompt, l_priority, 0))
    for i, prompt in enumerate(m_prompts):
        arrivals.setdefault(0, []).append(make_request(f"M{i}", prompt, PRIORITY["M"], 0))
    arrivals.setdefault(h_arrival, []).append(
        make_request("H", prefix + h_suffix, PRIORITY["H"], h_arrival)
    )
    total_tokens = sum(len(r.prompt_token_ids) + MAX_TOKENS for rs in arrivals.values() for r in rs)
    return arrivals, h_arrival, total_tokens


def num_blocks_for(cell: Cell, total_tokens: int) -> int:
    if cell.blocks == "ample":
        return 10_000
    # Tight: 1.1 x the blocks that all requests need together, plus the null block.
    return math.ceil(1.1 * math.ceil(total_tokens / BLOCK_SIZE)) + 1


def run_trial(cell: Cell, condition: str, control: str, trial: int, seed: int, model: str):
    arrivals, h_arrival, total_tokens = build_requests(cell, condition, trial, seed)
    threshold = 0 if cell.threshold == "0" else max(BLOCK_SIZE, cell.budget // 4)
    scheduler = create_scheduler(
        model=model,
        max_num_seqs=16,
        max_num_batched_tokens=cell.budget,
        enable_prefix_caching=True,
        long_prefill_token_threshold=threshold,
        num_blocks=num_blocks_for(cell, total_tokens),
        block_size=BLOCK_SIZE,
        max_model_len=32_768,
        skip_tokenizer_init=True,
        scheduling_policy="priority",
        scheduler_cls=InjectedWaitScheduler if control == "pc" else CountingScheduler,
    )
    ideal_steps = math.ceil(total_tokens / cell.budget) + MAX_TOKENS * (len(arrivals) + 6)
    cap = 20 * ideal_steps + h_arrival
    first_token: dict[str, int] = {}
    prefill_tokens = 0
    step = 0
    while step <= cap:
        for request in arrivals.get(step, []):
            scheduler.add_request(request)
        out = scheduler.schedule()
        req_ids = list(out.num_scheduled_tokens.keys())
        sampled = []
        for rid in req_ids:
            req = scheduler.requests[rid]
            # schedule() already advanced num_computed_tokens by the scheduled tokens.
            scheduled = out.num_scheduled_tokens[rid]
            before = req.num_computed_tokens - scheduled
            prefill_tokens += max(0, min(scheduled, req.num_prompt_tokens - before))
            sampled.append([7] if req.num_computed_tokens >= req.num_tokens else [])
        runner_out = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=sampled,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        )
        scheduler.update_from_output(out, runner_out)
        for rid, toks in zip(req_ids, sampled):
            if toks and rid not in first_token:
                first_token[rid] = step
        step += 1
        if step > max(arrivals) and not scheduler.has_unfinished_requests():
            break
    finished = not scheduler.has_unfinished_requests()
    h_first = first_token.get("H")
    return TrialResult(
        cell=asdict(cell),
        condition=condition,
        control=control,
        trial=trial,
        h_arrival=h_arrival,
        h_first_token_step=h_first,
        h_ttft_steps=None if h_first is None else h_first - h_arrival + 1,
        total_steps=step,
        prefill_tokens=prefill_tokens,
        preemptions=scheduler.e35_preemptions,
        finished=finished,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--control", choices=("vllm", "pc"), required=True)
    parser.add_argument("--cells", default="all", help="'all' or a JSON list of cell dicts")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cells = grid() if args.cells == "all" else [Cell(**c) for c in json.loads(args.cells)]
    started = time.time()
    with open(args.out, "w", encoding="utf-8") as sink:
        for cell in cells:
            for trial in range(args.trials):
                for condition in CONDITIONS:
                    result = run_trial(cell, condition, args.control, trial, args.seed, args.model)
                    sink.write(json.dumps(asdict(result)) + "\n")
    print(json.dumps({"cells": len(cells), "seconds": round(time.time() - started, 1)}))


if __name__ == "__main__":
    main()
