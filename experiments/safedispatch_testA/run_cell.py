"""Run one E-36 configuration with vLLM.

Modes:
  smoke       start the engine and generate 8 tokens for 4 prompts (used by inventory.py)
  outputs     greedy outputs with top-k logprobs for the whole prompt set -> outputs.jsonl
  throughput  output tokens/s at several batch sizes and prefill tokens/s -> throughput.json
  both        outputs, then throughput, on one engine
  warmup      A3: the first process of a config, before its measured runs. It does the same kinds of
              work at a small size (greedy outputs with logprobs, every decode batch size, prefill), so
              that the compile, JIT and autotune caches on disk exist. Its outputs are kept to measure
              the cold-cache effect (extras.py), and are never used for a decision.

The batch configuration (max_num_seqs, one generate() call) is the same for every backend.

Amendment A1 (R-34): the engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0, the deterministic
offline mode in vLLM's reproducibility guide) and prefix caching is off. Kernel selection is read later
by compare.py from the run's log.txt (an in-process log handler was removed: the dry run showed that
vLLM's logging setup during engine start drops it).
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path


PROFILE_MIN_SHARE = 0.005


def build_llm(args):
    from vllm import LLM

    kwargs = dict(
        model=args.model,
        seed=0,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
    )
    for name in ("linear_backend", "moe_backend", "attention_backend"):
        value = getattr(args, name)
        if value and value != "auto":
            kwargs[name] = value
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    started = time.time()
    llm = LLM(**kwargs)
    return llm, round(time.time() - started, 2), kwargs


def topk(position) -> list[list[float]]:
    """Return [[token_id, logprob], ...] sorted by logprob, for one position."""
    items = position.items() if hasattr(position, "items") else position
    pairs = [[int(tid), float(getattr(lp, "logprob", lp))] for tid, lp in items]
    return sorted(pairs, key=lambda p: -p[1])


def run_outputs(llm, prompts, args, out: Path) -> dict:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=args.logprobs, seed=0)
    started = time.time()
    results = llm.generate([p["prompt"] for p in prompts], params, use_tqdm=False)
    seconds = time.time() - started
    with open(out / "outputs.jsonl", "w", encoding="utf-8") as sink:
        for meta, res in zip(prompts, results):
            completion = res.outputs[0]
            sink.write(
                json.dumps(
                    {
                        "id": meta["id"],
                        "token_ids": list(completion.token_ids),
                        "topk": [topk(pos) for pos in (completion.logprobs or [])],
                    }
                )
                + "\n"
            )
    tokens = sum(len(r.outputs[0].token_ids) for r in results)
    return {"outputs_seconds": round(seconds, 3), "outputs_tokens": tokens}


def timed_generate(llm, prompts, max_tokens) -> float:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True, seed=0)
    started = time.time()
    llm.generate(prompts, params, use_tqdm=False)
    return time.time() - started


def run_throughput(llm, prompts, prefill, args) -> dict:
    texts = [p["prompt"] for p in prompts]
    decode = {}
    n_tok = args.throughput_max_tokens
    for bs in args.batch_sizes:
        batch = (texts * (bs // len(texts) + 1))[:bs]
        timed_generate(llm, batch, 8)  # warm-up for this batch size
        runs = [timed_generate(llm, batch, n_tok) for _ in range(args.repeats)]
        decode[str(bs)] = {
            "seconds": runs,
            "median_tokens_per_s": round(bs * n_tok / statistics.median(runs), 2),
        }
    timed_generate(llm, prefill[:2], 1)
    runs = [timed_generate(llm, prefill, 1) for _ in range(args.repeats)]
    return {"decode": decode, "prefill_seconds": runs, "prefill_prompts": len(prefill)}


def profile_kernels(llm, prompts, args) -> dict:
    """Amendment A2: names and device time of the CUDA kernels of one generate() call at each profile
    batch size, from torch.profiler.

    The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0), so the profiler sees its kernels.
    It runs after all measurements of the run, so it cannot change them. The table keeps every kernel
    with at least PROFILE_MIN_SHARE of the device time, and at least the top 150. Reported only.
    """
    import torch
    from torch.profiler import ProfilerActivity, profile

    texts = [p["prompt"] for p in prompts]
    tables = {}
    for bs in args.profile_batches:
        batch = (texts * (bs // len(texts) + 1))[:bs]
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            timed_generate(llm, batch, args.profile_tokens)
            torch.cuda.synchronize()
        rows = []
        for evt in prof.key_averages():
            us = getattr(evt, "self_device_time_total", None)
            if us is None:
                us = getattr(evt, "self_cuda_time_total", 0)
            if us > 0:
                rows.append({"name": evt.key[:300], "count": evt.count, "device_us": round(us, 1)})
        rows.sort(key=lambda r: -r["device_us"])
        total = sum(r["device_us"] for r in rows) or 1.0
        for r in rows:
            r["share"] = round(r["device_us"] / total, 5)
        kept = [r for i, r in enumerate(rows) if i < 150 or r["share"] >= PROFILE_MIN_SHARE]
        tables[str(bs)] = {"total_device_us": round(total, 1), "kernels": kept}
    return {"tokens": args.profile_tokens, "by_batch": tables}


def gpu_identity() -> dict:
    """Amendment A2: the GPU of this run, so that runs from different Pods or restarts can be told apart."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"cuda": False}
        p = torch.cuda.get_device_properties(0)
        return {
            "cuda": True,
            "name": p.name,
            "uuid": str(getattr(p, "uuid", "")),
            "sm": f"{p.major}{p.minor}",
            "sms": p.multi_processor_count,
            "memory_mib": p.total_memory // 2**20,
            "torch_cuda": torch.version.cuda,
        }
    except Exception as exc:  # reported only
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}


def run_warmup(llm, prompts, prefill, args, out: Path) -> dict:
    from vllm import SamplingParams

    n = args.warmup_prompts
    params = SamplingParams(temperature=0.0, max_tokens=args.warmup_tokens, logprobs=args.logprobs, seed=0)
    started = time.time()
    results = llm.generate([p["prompt"] for p in prompts[:n]], params, use_tqdm=False)
    with open(out / "outputs.jsonl", "w", encoding="utf-8") as sink:
        for meta, res in zip(prompts[:n], results):
            c = res.outputs[0]
            sink.write(json.dumps({"id": meta["id"], "token_ids": list(c.token_ids),
                                   "topk": [topk(pos) for pos in (c.logprobs or [])]}) + "\n")
    texts = [p["prompt"] for p in prompts]
    for bs in args.batch_sizes:
        timed_generate(llm, (texts * (bs // len(texts) + 1))[:bs], 8)
    timed_generate(llm, prefill[:2], 1)
    return {"warmup_seconds": round(time.time() - started, 3), "warmup_prompts": n}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "outputs", "throughput", "both", "warmup"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear-backend", default="auto")
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--attention-backend", default=None)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=256, help="new tokens in the flip runs")
    parser.add_argument("--throughput-max-tokens", type=int, default=128)
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="use only the first N prompts (dry run)")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--profile-kernels", action="store_true", help="A2: kernel table after the measurements")
    parser.add_argument("--warmup-prompts", type=int, default=64, help="A3 (R-40): prompts in a warm-up run")
    parser.add_argument("--warmup-tokens", type=int, default=64, help="A3 (R-40): new tokens in a warm-up run")
    parser.add_argument("--profile-batches", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--profile-tokens", type=int, default=32)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = json.loads((Path(args.prompts_dir) / "prompts.json").read_text())
    prefill = json.loads((Path(args.prompts_dir) / "prefill.json").read_text())
    if args.limit:
        prompts, prefill = prompts[: args.limit], prefill[: max(2, args.limit // 4)]

    import vllm

    record = {
        "args": vars(args),
        "vllm_version": vllm.__version__,
        "python": sys.version.split()[0],
        "machine": platform.machine(),
        "env": {k: v for k, v in os.environ.items() if k.startswith("VLLM_")},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_identity(),
    }
    try:
        llm, init_seconds, kwargs = build_llm(args)
        record.update(init_seconds=init_seconds, engine_kwargs=kwargs, status="started")
        if args.mode == "smoke":
            from vllm import SamplingParams

            llm.generate([p["prompt"] for p in prompts[:4]], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)
        if args.mode == "warmup":
            record.update(run_warmup(llm, prompts, prefill, args, out))
        if args.mode in ("outputs", "both"):
            record.update(run_outputs(llm, prompts, args, out))
        if args.mode in ("throughput", "both"):
            record["throughput"] = run_throughput(llm, prompts, prefill, args)
        record["status"] = "ok"
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # A2 (R-39 item 3): write the record before profiling, so a profiler crash, hang or timeout
        # cannot lose the measured run. run.json is written again after the profile.
        (out / "run.json").write_text(json.dumps(record, indent=1))
        if args.profile_kernels and args.mode != "warmup":  # a profiler failure never fails the run
            try:
                (out / "kernels.json").write_text(json.dumps(profile_kernels(llm, prompts, args), indent=1))
            except Exception as exc:
                record["profile_error"] = f"{type(exc).__name__}: {exc}"[:1000]
    except Exception as exc:  # recorded, then re-raised so the caller sees a non-zero exit
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        (out / "run.json").write_text(json.dumps(record, indent=1))
        raise
    (out / "run.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("status", "init_seconds") if k in record}))


if __name__ == "__main__":
    main()
