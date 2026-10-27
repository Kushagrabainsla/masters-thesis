"""Run one E-38 config: the GSM8K accuracy run, then teacher-forced scoring of the reference completions.

Modes:
  warmup  the first process of a config (E-36 A3): a small accuracy pass and a small scoring pass, so
          the compile, JIT and autotune caches exist before the measured run. Never used for a decision.
  full    accuracy on every problem (completions.jsonl), then teacher-forced scoring:
          - with --reference-dir: score that run's completions and compare with its stored distribution
            (tf.json: KL and top-1 disagreement per problem);
          - without it, this run is the reference: it scores its own completions and stores the
            distribution (reference_tf.npz) for the other configs.

The engine runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) with prefix caching off, as in E-36.
Usage: run_a2.py --mode full --model M --prompts-dir P --out O [--reference-dir R] [--linear-backend X]
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "safedispatch_testA"))

from gsm8k import extract  # noqa: E402

TOPK = 20


def build_llm(args):
    from vllm import LLM

    kwargs = dict(model=args.model, seed=0, max_num_seqs=args.max_num_seqs, max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False)
    for name in ("linear_backend", "moe_backend"):
        value = getattr(args, name)
        if value and value != "auto":
            kwargs[name] = value
    if args.enforce_eager:
        kwargs["enforce_eager"] = True
    started = time.time()
    return LLM(**kwargs), round(time.time() - started, 2), kwargs


def topk_of(position) -> list[tuple[int, float]]:
    items = position.items() if hasattr(position, "items") else position
    pairs = [(int(t), float(getattr(lp, "logprob", lp))) for t, lp in items]
    return sorted(pairs, key=lambda p: -p[1])[:TOPK]


def approx_kl(ref: list[tuple[int, float]], cand: list[tuple[int, float]]) -> float:
    """KL(ref || cand) on ref's top-k support, renormalized; a token missing in cand gets cand's smallest
    listed logprob (the E-36 compare.py rule)."""
    r = dict(ref)
    c = dict(cand)
    floor = min(c.values())
    r_norm = math.log(sum(math.exp(v) for v in r.values()))
    c_vals = {t: c.get(t, floor) for t in r}
    c_norm = math.log(sum(math.exp(v) for v in c_vals.values()))
    return max(0.0, sum(math.exp(lp - r_norm) * ((lp - r_norm) - (c_vals[t] - c_norm)) for t, lp in r.items()))


def accuracy_run(llm, items, args, out: Path):
    """A1 (R-43): greedy with top-20 logprobs, so the decode path can be compared with R_a (decode_prefix)."""
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                            logprobs=None if getattr(args, "no_score", False) else TOPK, seed=0)
    started = time.time()
    results = llm.generate([x["prompt"] for x in items], params, use_tqdm=False)
    seconds = time.time() - started
    decode = [[topk_of(pos) for pos in (res.outputs[0].logprobs or [])] for res in results]
    correct = 0
    with open(out / "completions.jsonl", "w") as sink:
        for x, res in zip(items, results):
            c = res.outputs[0]
            pred = extract(c.text)
            ok = bool(pred) and pred == x["gold"]
            correct += ok
            sink.write(json.dumps({"id": x["id"], "prompt_token_ids": list(res.prompt_token_ids),
                                   "token_ids": list(c.token_ids), "pred": pred, "gold": x["gold"],
                                   "correct": ok, "finish": c.finish_reason}) + "\n")
    return {"accuracy": correct / len(items), "correct": correct, "n": len(items),
            "accuracy_seconds": round(seconds, 2)}, decode


def score(llm, rows: list[dict], chunk: int = 256):
    """Teacher-forced: the top-k distribution at each completion position of each row. Scored in chunks
    of 256 prompts to bound host memory (R-43). Every completion position must be scored."""
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOPK, seed=0)
    out = []
    for i in range(0, len(rows), chunk):
        part = rows[i:i + chunk]
        prompts = [TokensPrompt(prompt_token_ids=r["prompt_token_ids"] + r["token_ids"]) for r in part]
        for r, res in zip(part, llm.generate(prompts, params, use_tqdm=False)):
            start, n = len(r["prompt_token_ids"]), len(r["token_ids"])
            pl = res.prompt_logprobs or []
            if len(pl) < start + n or any(pl[p] is None for p in range(start, start + n)):
                raise RuntimeError(f"teacher-forced scoring missed positions for {r['id']}")
            out.append([topk_of(pl[p]) for p in range(start, start + n)])
    return out


def save_reference(dists, ids, path: Path) -> None:
    import numpy as np

    offsets, toks, lps = [0], [], []
    for d in dists:
        for pos in d:
            pos = (pos + [(-1, -1e9)] * TOPK)[:TOPK]
            toks.append([t for t, _ in pos])
            lps.append([v for _, v in pos])
        offsets.append(len(toks))
    np.savez_compressed(path, ids=np.array(ids), offsets=np.array(offsets, dtype=np.int64),
                        toks=np.array(toks, dtype=np.int32).reshape(-1, TOPK),
                        lps=np.array(lps, dtype=np.float32).reshape(-1, TOPK))


def load_reference(path: Path) -> dict[str, list[list[tuple[int, float]]]]:
    import numpy as np

    # E-38 H100 run 2026-10-27: z["toks"] reads and decompresses the whole array on every access, so the
    # old per-position z["toks"][p] took about 503 min per array at 383,821 positions. Load each once.
    z = np.load(path)
    ids, offsets, toks, lps = z["ids"], z["offsets"], z["toks"].tolist(), z["lps"].tolist()
    ref = {}
    for i, rid in enumerate(ids):
        a, b = int(offsets[i]), int(offsets[i + 1])
        ref[str(rid)] = [[(t, v) for t, v in zip(toks[p], lps[p]) if t >= 0] for p in range(a, b)]
    return ref


def summary_of(kls: list[float], disagree: int, positions: int, per: dict) -> dict:
    srt = sorted(kls)
    q = lambda f: srt[min(len(srt) - 1, int(f * len(srt)))] if srt else None  # noqa: E731
    return {"positions": positions, "kl_mean": sum(kls) / positions if positions else None,
            "kl_median": q(0.5), "kl_p99": q(0.99),
            "top1_disagree_rate": disagree / positions if positions else None, "per_problem": per}


def compare_to_reference(ref, dists, ids) -> dict:
    """Teacher-forced: both sides scored the same tokens, so the position counts must be equal (R-43)."""
    per, allkl, disagree = {}, [], 0
    for rid, d in zip(ids, dists):
        r = ref.get(rid, [])
        if len(r) != len(d):
            raise RuntimeError(f"position count differs for {rid}: {len(r)} against {len(d)}")
        kls = [approx_kl(r[p], d[p]) for p in range(len(d))]
        dis = sum(r[p][0][0] != d[p][0][0] for p in range(len(d)))
        per[rid] = {"positions": len(d), "kl_mean": sum(kls) / len(d) if d else None, "top1_disagree": dis}
        allkl += kls
        disagree += dis
    return summary_of(allkl, disagree, len(allkl), per)


def compare_decode_prefix(ref, dists, ids, ref_tokens, tokens) -> dict:
    """A1 (R-43): decode path. Compare the generation logprobs with R_a's on the positions before the
    first token divergence (same prefix, so the distributions are comparable). Reported only."""
    per, allkl, disagree = {}, [], 0
    for rid, d in zip(ids, dists):
        r, a, b = ref.get(rid, []), ref_tokens.get(rid, []), tokens.get(rid, [])
        k = 0
        while k < min(len(a), len(b), len(r), len(d)) and a[k] == b[k]:
            k += 1
        k = min(k + 1, len(r), len(d))  # include the first divergent position: same prefix there too
        kls = [approx_kl(r[p], d[p]) for p in range(k)]
        dis = sum(r[p][0][0] != d[p][0][0] for p in range(k))
        per[rid] = {"positions": k, "top1_disagree": dis}
        allkl += kls
        disagree += dis
    return summary_of(allkl, disagree, len(allkl), per)


def gpu_identity() -> dict:
    try:
        import torch

        p = torch.cuda.get_device_properties(0)
        return {"name": p.name, "uuid": str(getattr(p, "uuid", "")), "sm": f"{p.major}{p.minor}"}
    except Exception as exc:  # reported only
        return {"error": f"{type(exc).__name__}: {exc}"[:200]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("warmup", "full"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--linear-backend", default="auto")
    parser.add_argument("--moe-backend", default="auto")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=0, help="first N problems only (dry run)")
    parser.add_argument("--no-score", action="store_true", help="E-40: accuracy only (no logprobs, no scoring)")
    parser.add_argument("--warmup-problems", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=64)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    items = json.loads((args.prompts_dir / "gsm8k.json").read_text())
    if args.limit:
        items = items[: args.limit]
    import vllm

    record = {"args": {k: str(v) for k, v in vars(args).items()}, "vllm_version": vllm.__version__,
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": gpu_identity()}
    try:
        llm, init_seconds, kwargs = build_llm(args)
        record.update(init_seconds=init_seconds, engine_kwargs=kwargs)
        if args.mode == "warmup":
            args.max_tokens = args.warmup_tokens
            acc, _ = accuracy_run(llm, items[: args.warmup_problems], args, args.out)
            record.update(acc)
            if not args.no_score:
                rows = [json.loads(l) for l in (args.out / "completions.jsonl").read_text().splitlines()]
                score(llm, rows[:16])
        else:
            acc, decode = accuracy_run(llm, items, args, args.out)
            record.update(acc)
            if args.no_score:
                record["status"] = "ok"
                record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                (args.out / "run.json").write_text(json.dumps(record, indent=1))
                print(json.dumps({"status": "ok", "accuracy": record.get("accuracy")}))
                return
            own = [json.loads(l) for l in (args.out / "completions.jsonl").read_text().splitlines()]
            own_ids = [r["id"] for r in own]
            started = time.time()
            if args.reference_dir:
                rows = [json.loads(l) for l in (args.reference_dir / "completions.jsonl").read_text().splitlines()]
                ids = [r["id"] for r in rows]
                tf = compare_to_reference(load_reference(args.reference_dir / "reference_tf.npz"), score(llm, rows), ids)
                (args.out / "tf.json").write_text(json.dumps(tf))
                dp = compare_decode_prefix(load_reference(args.reference_dir / "reference_decode.npz"), decode, own_ids,
                                           {r["id"]: r["token_ids"] for r in rows}, {r["id"]: r["token_ids"] for r in own})
                (args.out / "decode_prefix.json").write_text(json.dumps(dp))
                record.update(tf_kl_mean=tf["kl_mean"], tf_top1_disagree_rate=tf["top1_disagree_rate"],
                              decode_prefix_kl_mean=dp["kl_mean"], decode_prefix_top1_disagree_rate=dp["top1_disagree_rate"])
            else:
                save_reference(score(llm, own), own_ids, args.out / "reference_tf.npz")
                save_reference(decode, own_ids, args.out / "reference_decode.npz")
            record["score_seconds"] = round(time.time() - started, 2)
        record["status"] = "ok"
    except Exception as exc:
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        (args.out / "run.json").write_text(json.dumps(record, indent=1))
        raise
    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.out / "run.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: record.get(k) for k in ("status", "accuracy", "tf_kl_mean", "tf_top1_disagree_rate")}))


if __name__ == "__main__":
    main()
