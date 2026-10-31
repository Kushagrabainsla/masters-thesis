"""E-49 decode probe: decode-only step time of one Gemma 4 model over a context sweep, plus kernel tables.

One process per model, one in-process engine (VLLM_ENABLE_V1_MULTIPROCESSING=0, prefix caching off, text
only, prefill chunk 2,048 tokens). For each cell (context L, batch B, replicate r) the probe adds B requests of
exactly L prompt tokens, steps the engine until every request is in decode, checks the scheduler (B running,
none waiting, no preemption), runs 4 warm steps, then times blocks of `--steps` engine steps (wall clock, CUDA
synchronized at both ends). Selected cells also get a torch.profiler kernel table and every kernel event in
start order over `--profile-steps` decode steps. The requests are then aborted, so the next cell starts empty.

Outputs in --out: run.json (status, engine, KV and layer record), cells.json (per cell: step times, token
advance check, scheduler check, prefill time), kernels.json (tables per profiled cell), events_<cell>.json
(kernel events per profiled cell; the raw trace is kept gzipped), gsm8k.jsonl (sanity check).
run.json status: "started", "ok_slope" (the slope cells are done), "ok_pairs" (the pair cells are done too), "ok"
(all cells are done) or "error".
--stub writes synthetic files with the same layout (no vLLM, no GPU), for the Mac tests.

E-50 additions (R-65 C3): each cell sets `max_tokens` from the prefill schedule (`token_budget`) and checks before
the timing that every request has the tokens for the planned steps (status "budget"); every request that leaves
early is recorded with its finish reason (status "finished_early"). Pair cells (`--pair-cells`, batch "aL") take
their batch from the KV pool: the share `--auto-frac` of the requests of context L that fit the idle pool, with the
blocks per request measured at an earlier cell of context L. `--kv-cache-dtype` selects the KV cache type.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import argparse
import gzip
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "safedispatch_testA2"))
PROFILE_MIN_SHARE = 0.002
WARM_STEPS = 4
BUDGET_MARGIN = 32


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cell_list(text: str) -> list[tuple[int, int | str, int]]:
    """'2048:4,65536:4:1,2048:a32768' -> [(2048, 4, 0), (65536, 4, 1), (2048, 'a32768', 0)]; the third field is the
    replicate. A batch 'aL' is set at run time from the KV pool (`auto_batch`)."""
    out: list[tuple[int, int | str, int]] = []
    for c in text.split(","):
        if c:
            f = c.split(":")
            out.append((int(f[0]), f[1] if f[1].startswith("a") else int(f[1]), int(f[2]) if len(f) > 2 else 0))
    return out


def cell_key(ctx: int, batch: int, rep: int = 0) -> str:
    return f"{ctx}:{batch}" if rep == 0 else f"{ctx}:{batch}:{rep}"


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


def kernel_events(prof, path: Path) -> dict:
    """Every kernel of one profile in start order: [seq, name id, start us, duration us, grid, stream].

    The analysis assigns the attention launches to layers by this order (R-64 R6). The raw Chrome trace is kept
    gzipped next to the events file.
    """
    raw = path.with_name(path.stem + "_trace.json")
    prof.export_chrome_trace(str(raw))
    trace = json.loads(raw.read_text())
    with open(raw, "rb") as src, gzip.open(f"{raw}.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    raw.unlink()
    kernels = sorted((e for e in trace.get("traceEvents", []) if e.get("cat") == "kernel" and e.get("ph") == "X"),
                     key=lambda e: e["ts"])
    names: dict[str, int] = {}
    rows = []
    for i, e in enumerate(kernels):
        a = e.get("args") or {}
        rows.append([i, names.setdefault(e["name"], len(names)), round(e["ts"], 3), round(e.get("dur", 0), 3),
                     a.get("grid"), a.get("stream")])
    return {"names": list(names), "events": rows}


def scheduler_check(engine, batch: int, ids: list[str] | None = None) -> dict:
    """The cell's requests are all running, none is waiting and none was preempted (R-64 R2). With ids, the
    preemption count of each request is kept too, so a request that leaves later still has one (R-65 C3)."""
    try:
        sched = engine.engine_core.engine_core.scheduler
        running = list(sched.running)
        rec = {"running": len(running), "waiting": len(sched.waiting),
               "preemptions": sum(int(getattr(r, "num_preemptions", 0) or 0) for r in running)}
    except Exception as exc:  # the advance check (V2) still applies
        return {"known": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    try:
        rec["free_blocks"] = sched.kv_cache_manager.block_pool.get_num_free_blocks()
    except Exception:
        pass
    if ids:
        try:
            reqs = sched.requests
            rec["request_preemptions"] = {i: int(reqs[i].num_preemptions) for i in ids if i in reqs}
        except Exception:
            pass
    rec["known"] = True
    rec["ok"] = rec["running"] == batch and rec["waiting"] == 0 and rec["preemptions"] == 0
    return rec


def token_budget(ctx: int, batch: int, args) -> int:
    """`max_tokens` of one cell (R-65 C3). Decode tokens go first in the token budget of each step, so request 0
    decodes while the other B - 1 requests prefill, for at most ceil(L / (chunk - B)) steps each. Then come the warm
    steps, the most blocks, the profile steps and a margin."""
    chunk = getattr(args, "max_num_batched_tokens", 2048)
    prefill = (batch - 1) * math.ceil(ctx / max(1, chunk - batch))
    return prefill + WARM_STEPS + args.steps * args.max_blocks + args.profile_steps + BUDGET_MARGIN


def auto_batch(spec: str, sizes: dict[int, float], free_idle: int | None, args) -> int | None:
    """Batch 'aL': the share --auto-frac of the requests of context L that fit the idle KV pool. The blocks per
    request come from the first earlier cell of context L (free blocks before and after its prefill)."""
    per_req = sizes.get(int(spec[1:]))
    if not per_req or not free_idle:
        return None
    return max(1, min(args.max_num_seqs, int(args.auto_frac * free_idle / per_req)))


def run_cell(engine, stream, ctx: int, batch: int, rep: int, args, profile: bool, out: Path,
             spec: str | None = None) -> tuple[dict, dict | None]:
    from vllm import SamplingParams

    budget = token_budget(ctx, batch, args)
    cell = {"context": ctx, "batch": batch, "rep": rep, "max_tokens": budget, "steps_per_block": args.steps,
            "step_ms": [], "advance_min_max": []}
    max_len = getattr(args, "max_model_len", None)
    if max_len and ctx + budget > max_len:   # the requests cannot hold the planned steps: not run
        cell.update(status="budget", prefill_s=0.0, sched={})
        return cell, None
    params = SamplingParams(temperature=0.0, max_tokens=budget, ignore_eos=True, detokenize=False, seed=0)
    ids = [f"c{ctx}b{batch}r{rep}i{i}" for i in range(batch)]
    lens = dict.fromkeys(ids, 0)
    finished: dict[str, dict] = {}

    def step() -> None:
        for out_ in engine.step():
            if out_.request_id in lens:
                o = out_.outputs[0]
                lens[out_.request_id] = len(o.token_ids)
                if getattr(out_, "finished", False):   # why the request left: length cap, stop or abort (R-65 C3)
                    finished[out_.request_id] = {"reason": getattr(o, "finish_reason", None),
                                                 "tokens": len(o.token_ids), "stop": getattr(o, "stop_reason", None)}

    t0 = time.perf_counter()
    for rid, toks in zip(ids, prompts_for(stream, ctx, batch, ctx * 1000 + batch + 7919 * rep)):
        engine.add_request(rid, {"prompt_token_ids": toks}, params)
    guard = 0
    while min(lens.values()) < 1:
        step()
        guard += 1
        if guard > 100_000:
            raise RuntimeError("prefill did not finish")
    sync()
    cell.update(prefill_s=round(time.perf_counter() - t0, 3), sched=scheduler_check(engine, batch, ids),
                tokens_at_start=max(lens.values()))
    cell["budget_left"] = budget - cell["tokens_at_start"]
    need = WARM_STEPS + args.steps * args.max_blocks + (args.profile_steps if profile else 0)
    table = None
    if finished:                           # a request left during the prefill of the others: no timing
        cell["status"] = "finished_early"
    elif cell["sched"].get("ok") is False:   # preempted or queued: no timing, no profile
        cell["status"] = "preempted"
    elif cell["budget_left"] < need:       # a request would reach its length cap inside the timed steps (R-65)
        cell["status"] = "budget"
    else:
        for _ in range(WARM_STEPS):
            step()
        blocks, advance = cell["step_ms"], cell["advance_min_max"]
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
            if len(blocks) >= args.min_blocks and max(blocks) <= min(blocks) * (1 + args.block_spread):
                break
        cell["sched_after"] = scheduler_check(engine, batch, ids)
        # A request that left is not a preemption (R-65: the E-49 m26 64K cells hit the length cap).
        cell["status"] = "finished_early" if finished else \
            "preempted" if cell["sched_after"].get("ok") is False else "ok"
    if profile and cell["status"] == "ok":
        from torch.profiler import ProfilerActivity, profile as torch_profile

        sync()
        t = time.perf_counter()
        with torch_profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_steps):
                step()
            sync()
        wall_ms = (time.perf_counter() - t) * 1000   # has the profiler start and stop: reported only (R-64 R1)
        name = "events_" + (spec or cell_key(ctx, batch, rep)).replace(":", "_") + ".json"
        ev = kernel_events(prof, out / name)
        (out / name).write_text(json.dumps(ev))
        table = kernel_table(prof) | {"steps": args.profile_steps, "wall_ms": round(wall_ms, 2), "events": name,
                                      "events_n": len(ev["events"])}
        if finished:   # a request left inside the profile: the table is not of batch B, so the analysis skips it
            table["lost"] = True
    cell["finished"] = finished
    engine.abort_request(ids)
    for _ in range(3):  # let the engine release the aborted requests
        engine.step()
    return cell, table


def gsm8k_check(llm, gsm8k: list[dict], n: int, out: Path) -> dict:
    from gsm8k import extract
    from vllm import SamplingParams

    rows = gsm8k[:n]
    bos = getattr(llm.get_tokenizer(), "bos_token", None) or ""
    # The chat template already has BOS and vLLM adds it again, so remove the first one (R-64 O8).
    prompts = [r["prompt"][len(bos):] if bos and r["prompt"].startswith(bos) else r["prompt"] for r in rows]
    res = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=768, seed=0), use_tqdm=False)
    correct = 0
    with open(out / "gsm8k.jsonl", "w") as sink:
        for r, o in zip(rows, res):
            text = o.outputs[0].text
            ok = extract(text) == r["gold"]
            correct += ok
            sink.write(json.dumps({"id": r["id"], "gold": r["gold"], "pred": extract(text), "correct": ok, "tail": text[-300:]}) + "\n")
    return {"n": len(rows), "correct": correct, "accuracy": round(correct / max(1, len(rows)), 4), "bos_removed": bool(bos)}


def kv_record(llm) -> dict:
    """KV capacity record (R-64 R2). The analysis also reads the logged KV size and maximum concurrency."""
    rec = {}
    try:
        cc = llm.llm_engine.vllm_config.cache_config
        rec.update(num_gpu_blocks=cc.num_gpu_blocks, block_size=cc.block_size, cache_dtype=str(cc.cache_dtype))
    except Exception as exc:  # reported only
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        kcc = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_config
        rec["num_blocks"] = kcc.num_blocks
        rec["groups"] = [{"layers": len(g.layer_names), "spec": type(g.kv_cache_spec).__name__,
                          "block_size": g.kv_cache_spec.block_size, "page_bytes": g.kv_cache_spec.page_size_bytes}
                         for g in kcc.kv_cache_groups]
    except Exception as exc:  # reported only
        rec["groups_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return rec


def model_record(llm) -> dict:
    """Layer types (the analysis splits the profiles by layer), the multimodal state (R-64 R3, R6) and the CUDA
    graph batch sizes (a batch between two sizes runs padded to the next size)."""
    try:
        vc = llm.llm_engine.vllm_config
        mc = vc.model_config
    except Exception as exc:  # reported only
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    rec = {}
    for name, get in (("layer_types", lambda: list(mc.hf_text_config.layer_types)),
                      ("is_multimodal_model", lambda: bool(mc.is_multimodal_model)),
                      ("is_mm_prefix_lm", lambda: bool(mc.is_mm_prefix_lm)),
                      ("mm_limits", lambda: str(mc.multimodal_config.limit_per_prompt)),
                      ("cudagraph_sizes", lambda: list(vc.compilation_config.cudagraph_capture_sizes))):
        try:
            rec[name] = get()
        except Exception as exc:  # reported only
            rec[name + "_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def stub_events(types: list[str], ctx: int, batch: int, b: float, c: float, args) -> dict:
    """One synthetic profile in start order. Per layer: a GEMM, the KV write and one attention group (a main
    kernel, plus a combine kernel for global layers). Global groups grow with L; sliding groups do not, unless
    --stub-sliding-growth is set."""
    n_glob = types.count("full_attention")
    glob = b * ctx / n_glob * (0.3 if args.stub_bad_attribution else 1.0)     # ms per global layer
    slide = 0.012 * batch / 4 * (1 + args.stub_sliding_growth * ctx / 65536)   # ms per sliding layer
    other = max(0.1, 0.9 * c - 0.012 * batch / 4 * (len(types) - n_glob)) / len(types)
    att = ["kernel_7", "kernel_8", "kernel_9"] if args.stub_opaque_names else \
        ["void flash::FlashAttnFwdSm90<hdim256>", "void flash::FlashAttnFwdSm90<hdim512>", "void flash::FlashAttnFwdCombine"]
    names = ["sm90_xmma_gemm_e4m3", "void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16>", *att,
             "argmax_kernel", "void vllm::block_table_gather"]
    rows: list[list] = []
    clock = [0.0]

    def add(nid: int, ms: float, grid: list[int]) -> None:
        rows.append([len(rows), nid, round(clock[0], 3), round(ms * 1000, 3), grid, 7])
        clock[0] += ms * 1000 + 1.0

    for _ in range(args.profile_steps):
        for kind in types:
            add(0, other, [132, 1, 1])
            add(1, 0.002, [batch, 1, 1])
            if kind == "full_attention":
                add(3, glob * 0.9, [1, 4, batch * 8])
                add(4, glob * 0.1, [1, 4, batch])
            else:
                add(2, slide, [1, 16, batch])
        if args.stub_bad_attribution:
            add(6, 0.7 * b * ctx, [batch, 1, 1])
        add(5, 0.01, [1, 1, 1])
    return {"names": names, "events": rows}


def table_from_events(ev: dict) -> dict:
    agg: dict[int, list] = {}
    for r in ev["events"]:
        a = agg.setdefault(r[1], [0, 0.0])
        a[0] += 1
        a[1] += r[3]
    total = sum(v[1] for v in agg.values()) or 1.0
    rows = [{"name": ev["names"][k], "count": v[0], "device_us": round(v[1], 1), "share": round(v[1] / total, 5)}
            for k, v in agg.items()]
    rows.sort(key=lambda r: -r["device_us"])
    return {"total_device_us": round(total, 1), "kernels": rows}


def stub(args, out: Path) -> None:
    """Synthetic outputs: step time T = c + b * L per cell, with block noise and an optional cell effect.

    b is the time to read the global K and V of one context token for the whole batch at --stub-bw-gbps (half the
    bytes with an FP8 KV cache). The idle KV pool has --stub-pool-blocks blocks (twice as many with FP8), and a
    request of context L holds 0.0365 L + 30 blocks (about the E-49 31B record), so pair cells get a batch as on the
    GPU.
    """
    rng = random.Random(args.stub_seed)
    fp8 = args.kv_cache_dtype.startswith("fp8")
    global_bytes, n_layers = (20480, 30) if "26B" in args.model else (81920, 60)
    global_bytes //= 2 if fp8 else 1
    types = (["sliding_attention"] * 5 + ["full_attention"]) * (n_layers // 6)
    todo = [(c, "slope") for c in cell_list(args.cells)]
    if not args.stub_stop_after_slope:
        todo += [(c, "pair") for c in cell_list(args.pair_cells)] + [(c, "waste") for c in cell_list(args.waste_cells)]
    profiled = {cell_key(*c) for c in cell_list(args.profile_cells)}
    preempt = {cell_key(*c) for c in cell_list(args.stub_preempt)}
    free_idle = args.stub_pool_blocks * (2 if fp8 else 1)
    sizes: dict[int, float] = {}
    cells, tables, first = [], {}, True
    for (ctx, batch, rep), kind in todo:
        key = cell_key(ctx, batch, rep)
        spec = None
        if isinstance(batch, str):
            spec, batch = key, auto_batch(batch, sizes, free_idle, args)
        cell = {"context": ctx, "batch": batch, "rep": rep, "steps_per_block": args.steps, "kind": kind}
        if spec:
            cell.update(spec=spec, auto_for=int(spec.split(":")[1][1:]))
        if batch is None:
            cells.append(cell | {"status": "no_size", "step_ms": [], "advance_min_max": []})
            continue
        if ctx not in sizes:   # the first cell of a context measures the blocks per request, as on the GPU
            sizes[ctx] = 0.0365 * ctx + 30
            cell["blocks_per_request"] = round(sizes[ctx], 1)
        b = global_bytes * batch / (args.stub_bw_gbps * 1e9) * 1000.0      # ms per context token
        c = args.stub_c_ms * (batch / 4) ** 0.3
        base = (c + b * ctx + args.stub_curve * c * (ctx / 65536) ** 2) * (1 + rng.gauss(0, args.stub_cell_sd))
        cell.update(prefill_s=round(ctx * batch / 2e4, 3), max_tokens=token_budget(ctx, batch, args), finished={})
        if key in preempt:
            cells.append(cell | {"status": "preempted", "sched": {"known": True, "ok": False, "running": batch - 1,
                                                                   "waiting": 1, "preemptions": 1},
                                 "step_ms": [], "advance_min_max": []})
            continue
        steps = [round(base * (1 + rng.uniform(-args.stub_noise, args.stub_noise)), 4) for _ in range(3)]
        lag = 1 if args.stub_bad_advance else 0
        cells.append(cell | {"status": "ok", "sched": {"known": True, "ok": True, "running": batch, "waiting": 0,
                                                       "preemptions": 0},
                             "step_ms": steps, "advance_min_max": [[args.steps - lag * 5, args.steps] for _ in steps]})
        if key in profiled:
            ev = stub_events(types, ctx, batch, b, c, args)
            name = "events_" + key.replace(":", "_") + ".json"
            (out / name).write_text(json.dumps(ev))
            # The profile wall has the profiler start and stop; the first profile also pays the CUPTI start.
            wall = base * args.profile_steps + args.stub_profile_wall_ms * (3 if first else 1) + rng.uniform(0, 20)
            tables[key] = table_from_events(ev) | {"steps": args.profile_steps, "wall_ms": round(wall, 2), "events": name,
                                                   "events_n": len(ev["events"])}
            first = False
    (out / "cells.json").write_text(json.dumps(cells, indent=1))
    (out / "kernels.json").write_text(json.dumps(tables, indent=1))
    acc = args.stub_accuracy
    with open(out / "gsm8k.jsonl", "w") as sink:
        for i in range(32):
            sink.write(json.dumps({"id": f"g{i}", "correct": i < round(acc * 32)}) + "\n")
    dims = "{'sliding_attention': 256, 'full_attention': 512}"
    log = ["INFO Using FLASH_ATTN backend", "INFO GPU KV cache size: 400,000 tokens",
           "INFO Maximum concurrency for 66,048 tokens per request: 6.06x",
           f"INFO Initializing a V1 LLM engine with config: model='{args.model}', kv_cache_dtype={args.kv_cache_dtype}, seed=0",
           f"INFO Gemma4 model has heterogeneous head dimensions {dims}. " +
           ("Using per-layer FA3/FA4 selection for FP8 KV cache on SM90." if fp8 else
            "Using FA4 for all layers to avoid mixed FA3/FA4 penalty.")]
    if not args.stub_multimodal:
        log.append("INFO All limits of multimodal modalities supported by the model are set to 0, running in text-only mode.")
    (out / "log.txt").write_text("\n".join(log) + "\n")
    status = "error" if args.stub_fail else "ok_slope" if args.stub_stop_after_slope else "ok"
    (out / "run.json").write_text(json.dumps({"status": status, "stub": True, "args": {k: str(v) for k, v in vars(args).items()},
                                              "gsm8k": {"n": 32, "accuracy": acc}, "free_blocks_idle": free_idle,
                                              "kv": {"num_gpu_blocks": free_idle + 1, "block_size": 16,
                                                     "cache_dtype": args.kv_cache_dtype},
                                              "model": {"layer_types": types, "is_multimodal_model": True},
                                              "gpu": {"name": "STUB H100 80GB HBM3"}}, indent=1))
    if args.stub_fail:
        raise SystemExit("stub failure (--stub-fail)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gsm8k", type=Path, help="gsm8k.json from ../safedispatch_testA2/gsm8k.py")
    ap.add_argument("--cells", default="2048:4,8192:4,32768:4,65536:4,65536:4:1,32768:4:1",
                    help="slope cells L:B[:replicate], one batch")
    ap.add_argument("--pair-cells", default="", help="E-50: cells L:aL2[:replicate] with the batch set from the KV "
                    "pool for context L2, run after the slope cells")
    ap.add_argument("--waste-cells", default="1024:1,1024:8,1024:32,16384:8", help="exploratory cells, run last")
    ap.add_argument("--profile-cells", default="2048:4,32768:4,65536:4,1024:1,1024:8,1024:32")
    ap.add_argument("--auto-frac", type=float, default=0.85, help="share of the idle KV pool that an 'aL' batch fills")
    ap.add_argument("--kv-cache-dtype", default="auto", help="auto (the model dtype) or fp8")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--min-blocks", type=int, default=3)
    ap.add_argument("--max-blocks", type=int, default=5)
    ap.add_argument("--block-spread", type=float, default=0.02)
    ap.add_argument("--profile-steps", type=int, default=8)
    ap.add_argument("--gsm8k-n", type=int, default=32)
    ap.add_argument("--max-model-len", type=int, default=66048)
    ap.add_argument("--max-num-seqs", type=int, default=32)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048, help="prefill chunk (R-64 R2)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-seed", type=int, default=0)
    ap.add_argument("--stub-c-ms", type=float, default=15.0, help="context-free step time at B=4")
    ap.add_argument("--stub-bw-gbps", type=float, default=2500.0, help="global attention bandwidth")
    ap.add_argument("--stub-noise", type=float, default=0.004, help="block noise (uniform, relative)")
    ap.add_argument("--stub-cell-sd", type=float, default=0.0, help="cell effect (normal, relative)")
    ap.add_argument("--stub-curve", type=float, default=0.0, help="quadratic term at 64K, relative to c")
    ap.add_argument("--stub-sliding-growth", type=float, default=0.0, help="sliding attention growth at 64K")
    ap.add_argument("--stub-profile-wall-ms", type=float, default=0.0, help="profiler start and stop cost")
    ap.add_argument("--stub-preempt", default="", help="cells L:B[:r] that the scheduler preempts")
    ap.add_argument("--stub-pool-blocks", type=int, default=16095, help="idle KV pool blocks with the BF16 cache")
    ap.add_argument("--stub-accuracy", type=float, default=0.9)
    ap.add_argument("--stub-bad-advance", action="store_true")
    ap.add_argument("--stub-bad-attribution", action="store_true")
    ap.add_argument("--stub-opaque-names", action="store_true", help="attention kernels the regex does not match")
    ap.add_argument("--stub-multimodal", action="store_true", help="no text-only log line")
    ap.add_argument("--stub-stop-after-slope", action="store_true")
    ap.add_argument("--stub-fail", action="store_true")
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
              "started_at": now(), "status": "started",
              "gpu": {"name": torch.cuda.get_device_name(0), "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))}}
    try:
        t = time.time()
        llm = LLM(model=args.model, revision=args.revision, seed=0, max_num_seqs=args.max_num_seqs,
                  max_model_len=args.max_model_len, max_num_batched_tokens=args.max_num_batched_tokens,
                  gpu_memory_utilization=args.gpu_memory_utilization, enable_prefix_caching=False,
                  kv_cache_dtype=args.kv_cache_dtype,
                  limit_mm_per_prompt={"image": 0, "audio": 0, "video": 0})   # all 0: text-only mode (R-64 R3)
        record.update(init_seconds=round(time.time() - t, 1), kv=kv_record(llm), model=model_record(llm))
        record["gsm8k"] = gsm8k_check(llm, gsm8k, args.gsm8k_n, out)
        (out / "run.json").write_text(json.dumps(record, indent=1))
        engine = llm.llm_engine
        slope, pairs, waste = cell_list(args.cells), cell_list(args.pair_cells), cell_list(args.waste_cells)
        plan = [(c, "slope") for c in slope] + [(c, "pair") for c in pairs] + [(c, "waste") for c in waste]
        stream = token_stream(llm.get_tokenizer(), gsm8k, max(c[0] for c, _ in plan) + 8)
        profiled = {cell_key(*c) for c in cell_list(args.profile_cells)}
        free_idle = scheduler_check(engine, 0).get("free_blocks")
        record["free_blocks_idle"] = free_idle
        sizes: dict[int, float] = {}   # blocks per request by context, from the first cell of each context
        cells, tables = [], {}
        for i, ((ctx, batch, rep), kind) in enumerate(plan):
            key = cell_key(ctx, batch, rep)
            spec = None
            if isinstance(batch, str):
                spec, batch = key, auto_batch(batch, sizes, free_idle, args)
            if batch is None:   # no earlier cell of that context: the pair cell cannot be sized
                cell, table = {"context": ctx, "batch": None, "rep": rep, "status": "no_size", "step_ms": [],
                               "advance_min_max": []}, None
            else:
                cell, table = run_cell(engine, stream, ctx, batch, rep, args, key in profiled, out, spec)
            cell["kind"] = kind
            if spec:
                cell.update(spec=spec, auto_for=int(spec.split(":")[1][1:]))
            free = (cell.get("sched") or {}).get("free_blocks")
            if free_idle and free is not None and cell["sched"].get("ok") and ctx not in sizes:
                sizes[ctx] = (free_idle - free) / batch
                cell["blocks_per_request"] = round(sizes[ctx], 1)
            cells.append(cell)
            if table:
                tables[key] = table
            (out / "cells.json").write_text(json.dumps(cells, indent=1))
            (out / "kernels.json").write_text(json.dumps(tables, indent=1))
            print(json.dumps({k: cell[k] for k in ("context", "batch", "rep", "status", "step_ms")}), flush=True)
            if i == len(slope) - 1:   # the E-49 rule needs only the slope cells (R-64 R8)
                record.update(status="ok_slope", slope_done_at=now())
                (out / "run.json").write_text(json.dumps(record, indent=1))
            if pairs and i == len(slope) + len(pairs) - 1:   # the E-50 rule needs the pair cells too
                record.update(status="ok_pairs", pairs_done_at=now())
                (out / "run.json").write_text(json.dumps(record, indent=1))
        record.update(status="ok", finished_at=now())
    except Exception as exc:  # recorded, then re-raised so the caller sees a non-zero exit
        kept = record["status"] if record["status"] in ("ok_slope", "ok_pairs") else "error"
        record.update(status=kept, error=f"{type(exc).__name__}: {exc}"[:2000])
        (out / "run.json").write_text(json.dumps(record, indent=1))
        raise
    (out / "run.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({"status": record["status"], "init_seconds": record.get("init_seconds")}))


if __name__ == "__main__":
    main()
