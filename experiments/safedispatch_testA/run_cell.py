"""Run one E-36 configuration with vLLM.

Modes:
  smoke       start the engine and generate 8 tokens for 4 prompts (used by inventory.py)
  outputs     greedy outputs with top-k logprobs for the whole prompt set -> outputs.jsonl
  throughput  output tokens/s at several batch sizes and prefill tokens/s -> throughput.json
  both        outputs, then throughput, on one engine

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "outputs", "throughput", "both"), required=True)
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
    }
    try:
        llm, init_seconds, kwargs = build_llm(args)
        record.update(init_seconds=init_seconds, engine_kwargs=kwargs, status="started")
        if args.mode == "smoke":
            from vllm import SamplingParams

            llm.generate([p["prompt"] for p in prompts[:4]], SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)
        if args.mode in ("outputs", "both"):
            record.update(run_outputs(llm, prompts, args, out))
        if args.mode in ("throughput", "both"):
            record["throughput"] = run_throughput(llm, prompts, prefill, args)
        record["status"] = "ok"
    except Exception as exc:  # recorded, then re-raised so the caller sees a non-zero exit
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        (out / "run.json").write_text(json.dumps(record, indent=1))
        raise
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "run.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record[k] for k in ("status", "init_seconds") if k in record}))


if __name__ == "__main__":
    main()
