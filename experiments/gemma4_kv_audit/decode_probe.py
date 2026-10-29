"""E-49 decode probe: decode-only step time of one Gemma 4 model over a context sweep, plus kernel tables.

One process per model, one in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0, prefix caching off,
text only). For each cell (context L, batch B) the probe adds B requests of exactly L prompt tokens, steps
the engine until every request is in decode, runs 4 warm steps, then times blocks of `--steps` engine
steps (wall clock, CUDA synchronized at both ends). Selected cells also get a torch.profiler kernel table
over `--profile-steps` decode steps. The requests are then aborted, so the next cell starts empty.

Outputs in --out: run.json (status, engine and KV capacity record), cells.json (per cell: step times,
token advance check, prefill time), kernels.json (tables per profiled cell), gsm8k.jsonl (sanity check).
--stub writes synthetic files with the same layout (no vLLM, no GPU), for the Mac tests.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safedispatch_testA2"))
PROFILE_MIN_SHARE = 0.002
WARM_STEPS = 4


def cell_list(text: str) -> list[tuple[int, int]]:
    """'2048:4,8192:4' -> [(2048, 4), (8192, 4)]."""
    return [tuple(int(x) for x in c.split(":")) for c in text.split(",") if c]


def token_stream(tokenizer, gsm8k: list[dict], need: int) -> list[int]:
    """Real text tokens (GSM8K prompts, joined) repeated until `need` tokens; special tokens removed."""
    ids: list[int] = []
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    for row in gsm8k:
        ids.extend(t for t in tokenizer.encode(row["prompt"], add_special_tokens=False) if t not in special)
        if len(ids) >= 300_000:
            break
    while len(ids) < need:
        ids = ids + ids
    return ids


def prompts_for(stream: list[int], ctx: int, batch: int, cell_seed: int) -> list[list[int]]:
    """B different windows of exactly ctx tokens (different offsets, so no two prompts are equal)."""
    rng = random.Random(cell_seed)
    span = len(stream) - ctx - 1
    return [stream[(rng.randrange(span) + 7919 * i) % span:][:ctx] for i in range(batch)]


def sync() -> None:
    import torch

    torch.cuda.synchronize()


def kernel_table(prof) -> dict:
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
    kept = [r for i, r in enumerate(rows) if i < 200 or r["share"] >= PROFILE_MIN_SHARE]
    return {"total_device_us": round(total, 1), "kernels": kept}


def run_cell(engine, stream, ctx: int, batch: int, args, profile: bool) -> tuple[dict, dict | None]:
    from vllm import SamplingParams

    new_tokens = WARM_STEPS + args.steps * args.max_blocks + args.profile_steps + 64
    params = SamplingParams(temperature=0.0, max_tokens=new_tokens, ignore_eos=True, detokenize=False, seed=0)
    ids = [f"c{ctx}b{batch}r{i}" for i in range(batch)]
    lens = dict.fromkeys(ids, 0)

    def step() -> None:
        for out in engine.step():
            if out.request_id in lens:
                lens[out.request_id] = len(out.outputs[0].token_ids)

    t0 = time.perf_counter()
    for rid, toks in zip(ids, prompts_for(stream, ctx, batch, ctx * 1000 + batch)):
        engine.add_request(rid, {"prompt_token_ids": toks}, params)
    guard = 0
    while min(lens.values()) < 1:
        step()
        guard += 1
        if guard > 100_000:
            raise RuntimeError("prefill did not finish")
    sync()
    prefill_s = time.perf_counter() - t0
    for _ in range(WARM_STEPS):
        step()
    blocks, advance = [], []
    while len(blocks) < args.max_blocks:
        before = dict(lens)
        sync()
        t = time.perf_counter()
        for _ in range(args.steps):
            step()
        sync()
        blocks.append(round((time.perf_counter() - t) / args.steps * 1000, 4))
        adv = [lens[r] - before[r] for r in ids]
        advance.append([min(adv), max(adv)])
        if len(blocks) >= args.min_blocks and max(blocks) / min(blocks) - 1 <= args.block_spread:
            break
    table = None
    if profile:
        from torch.profiler import ProfilerActivity, profile as torch_profile

        sync()
        t = time.perf_counter()
        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_steps):
                step()
            sync()
        wall_ms = (time.perf_counter() - t) * 1000
        table = kernel_table(prof) | {"steps": args.profile_steps, "wall_ms": round(wall_ms, 2)}
    engine.abort_request(ids)
    for _ in range(3):  # let the engine release the aborted requests
        engine.step()
    cell = {"context": ctx, "batch": batch, "prefill_s": round(prefill_s, 3), "step_ms": blocks,
            "advance_min_max": advance, "steps_per_block": args.steps}
    return cell, table


def gsm8k_check(llm, gsm8k: list[dict], n: int, out: Path) -> dict:
    from gsm8k import extract
    from vllm import SamplingParams

    rows = gsm8k[:n]
    res = llm.generate([r["prompt"] for r in rows], SamplingParams(temperature=0.0, max_tokens=768, seed=0), use_tqdm=False)
    correct = 0
    with open(out / "gsm8k.jsonl", "w") as sink:
        for r, o in zip(rows, res):
            text = o.outputs[0].text
            ok = extract(text) == r["gold"]
            correct += ok
            sink.write(json.dumps({"id": r["id"], "gold": r["gold"], "pred": extract(text), "correct": ok, "tail": text[-300:]}) + "\n")
    return {"n": len(rows), "correct": correct, "accuracy": round(correct / max(1, len(rows)), 4)}


def kv_record(llm) -> dict:
    rec = {}
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        rec = {"num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size, "cache_dtype": str(cc.cache_dtype)}
    except Exception as exc:  # reported only
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


def stub(args, out: Path) -> None:
    """Synthetic outputs: step time T = c + b * L per cell, with noise; attention kernels grow with L.

    b is the time to read the global K and V of one context token for the whole batch at --stub-bw-gbps.
    """
    rng = random.Random(args.stub_seed)
    global_bytes = 20480 if "26B" in args.model else 81920
    cells, tables = [], {}
    for ctx, batch in cell_list(args.cells) + cell_list(args.waste_cells):
        b = global_bytes * batch / (args.stub_bw_gbps * 1e9) * 1000.0      # ms per context token
        c = args.stub_c_ms * (batch / 4) ** 0.3
        base = c + b * ctx
        steps = [round(base * (1 + rng.uniform(-args.stub_noise, args.stub_noise)), 4) for _ in range(3)]
        lag = 1 if args.stub_bad_advance else 0
        cells.append({"context": ctx, "batch": batch, "prefill_s": round(ctx * batch / 2e4, 3), "step_ms": steps,
                      "advance_min_max": [[args.steps - lag * 5, args.steps] for _ in steps], "steps_per_block": args.steps})
        if f"{ctx}:{batch}" in args.profile_cells.split(","):
            att = b * ctx * 1000 * args.profile_steps * (0.3 if args.stub_bad_attribution else 1.0)
            other = c * 1000 * args.profile_steps * 0.9
            ks = [{"name": "void flash::flash_fwd_splitkv_kernel<512>", "count": 10 * args.profile_steps, "device_us": round(att, 1)},
                  {"name": "sm90_xmma_gemm_e4m3", "count": 300 * args.profile_steps, "device_us": round(other * 0.8, 1)},
                  {"name": "triton_poi_fused_add_rms_norm_0", "count": 400 * args.profile_steps, "device_us": round(other * 0.2, 1)}]
            if args.stub_bad_attribution:
                ks.append({"name": "void vllm::block_table_gather", "count": 10, "device_us": round(b * ctx * 1000 * args.profile_steps * 0.7, 1)})
            total = sum(k["device_us"] for k in ks)
            for k in ks:
                k["share"] = round(k["device_us"] / total, 5)
            tables[f"{ctx}:{batch}"] = {"total_device_us": round(total, 1), "kernels": ks, "steps": args.profile_steps,
                                        "wall_ms": round(base * args.profile_steps, 2)}
    (out / "cells.json").write_text(json.dumps(cells, indent=1))
    (out / "kernels.json").write_text(json.dumps(tables, indent=1))
    acc = args.stub_accuracy
    with open(out / "gsm8k.jsonl", "w") as sink:
        for i in range(32):
            sink.write(json.dumps({"id": f"g{i}", "correct": i < round(acc * 32)}) + "\n")
    (out / "log.txt").write_text("INFO Using FLASH_ATTN backend\nINFO GPU KV cache size: 400,000 tokens\n")
    (out / "run.json").write_text(json.dumps({"status": "ok", "stub": True, "args": {k: str(v) for k, v in vars(args).items()},
                                              "gsm8k": {"n": 32, "accuracy": acc}, "kv": {"num_gpu_blocks": 25000, "block_size": 16},
                                              "gpu": {"name": "STUB H100 80GB HBM3"}}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gsm8k", type=Path, help="gsm8k.json from ../safedispatch_testA2/gsm8k.py")
    ap.add_argument("--cells", default="2048:4,8192:4,32768:4,65536:4", help="slope cells L:B (same B)")
    ap.add_argument("--waste-cells", default="1024:1,1024:8,1024:32")
    ap.add_argument("--profile-cells", default="2048:4,32768:4,1024:1,1024:8,1024:32")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--min-blocks", type=int, default=3)
    ap.add_argument("--max-blocks", type=int, default=5)
    ap.add_argument("--block-spread", type=float, default=0.02)
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--gsm8k-n", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=66048)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-seed", type=int, default=0)
    ap.add_argument("--stub-c-ms", type=float, default=15.0, help="context-free step time at B=4")
    ap.add_argument("--stub-bw-gbps", type=float, default=2500.0, help="global attention bandwidth")
    ap.add_argument("--stub-noise", type=float, default=0.004)
    ap.add_argument("--stub-accuracy", type=float, default=0.9)
    ap.add_argument("--stub-bad-advance", action="store_true")
    ap.add_argument("--stub-bad-attribution", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.stub:
        stub(args, out)
        print("stub ok")
        return

    import torch
    import vllm
    from vllm import LLM

    gsm8k = json.loads(args.gsm8k.read_text())
    record = {"args": {k: str(v) for k, v in vars(args).items()}, "vllm_version": vllm.__version__,
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": "started",
              "gpu": {"name": torch.cuda.get_device_name(0), "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))}}
    try:
        t = time.time()
        llm = LLM(model=args.model, revision=args.revision, seed=0, max_num_seqs=args.max_num_seqs,
                  max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
                  enable_prefix_caching=False, limit_mm_per_prompt={"image": 0, "audio": 0})
        record.update(init_seconds=round(time.time() - t, 1), kv=kv_record(llm))
        record["gsm8k"] = gsm8k_check(llm, gsm8k, args.gsm8k_n, out)
        (out / "run.json").write_text(json.dumps(record, indent=1))
        engine = llm.llm_engine
        cells_all = cell_list(args.cells) + cell_list(args.waste_cells)
        stream = token_stream(llm.get_tokenizer(), gsm8k, max(c for c, _ in cells_all) + 8)
        profiled = set(args.profile_cells.split(","))
        cells, tables = [], {}
        for ctx, batch in cells_all:
            cell, table = run_cell(engine, stream, ctx, batch, args, f"{ctx}:{batch}" in profiled)
            cells.append(cell)
            if table:
                tables[f"{ctx}:{batch}"] = table
            (out / "cells.json").write_text(json.dumps(cells, indent=1))
            (out / "kernels.json").write_text(json.dumps(tables, indent=1))
            print(json.dumps({k: cell[k] for k in ("context", "batch", "step_ms")}), flush=True)
        record.update(status="ok", finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    except Exception as exc:  # recorded, then re-raised so the caller sees a non-zero exit
        record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000])
        (out / "run.json").write_text(json.dumps(record, indent=1))
        raise
    (out / "run.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"status": record["status"], "init_seconds": record.get("init_seconds")}))


if __name__ == "__main__":
    main()
