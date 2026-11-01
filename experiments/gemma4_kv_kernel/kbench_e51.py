"""E-51 kernel microbenchmark (README Sections 4 to 6, with amendment A1): the decode attention of the Gemma 4 31B
global layers (32 query heads, 4 KV heads, head_dim 512, scale 1.0, no soft-cap) on one GPU, with the stock vLLM FA4
kernel and with the two Triton layouts of `compact_attn.py` (standard K and V, compact K=V), for the FP8 and the BF16
cache.

Order (A1): the correctness cell, the four group cells, the confirmation phase, then the reported cells. The
correctness cell (B = 2, L = 3,000, built as Gemma 4 K=V data) checks the stock kernels and every Triton configuration
of the sweep against the float32 reference of its own stored values. If a deciding FP8 kernel fails there, the run
stops (fail fast; on by default where the stock FP8 kernel must run, SM90). The confirmation phase times each decision
row (analyze_e51.confirm_plan) REPEATS times, each time in a new CUDA graph, with the order reversed every other
repeat (A B .. B A). After it, run.json has the status "ok_groups", so a later timeout keeps the decision; at the end
it has "ok".

Per cell (batch B, context L): random cache values in 64-token pages with a random block table (the same for every
kernel), in the engine's layout (FlashAttentionImpl.forward at 3966924d5): one (blocks, KV heads, 64, 2 x 512) tensor,
viewed as K and V by transpose(1, 2).split(512); the compact rows are one (blocks, KV heads, 64, 640) tensor, viewed
as V and K_rot in the same way. Each configuration is captured as CALLS calls in one CUDA graph and replayed; time per
call = replay time / CALLS (median and spread over REPLAYS). A configuration that raises is recorded, not fatal. BF16
is secondary: its Triton sweep runs at the group cells only, with 4 configurations (A1, O6).
--stub runs the same flow with synthetic numbers (no GPU, no vLLM, no Triton) for the Mac tests.

Usage: kbench_e51.py --out <dir> [--cells B:L,...] [--stages N] [--stages-bf16 N] [--fail-fast auto|on|off] [--stub]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_e51 as an  # noqa: E402  (the selection rule, the tolerances and the repeat count)

HQ, HK, D, BS = 32, 4, 512, 64
CALLS, REPLAYS = 20, 15
GROUP_CELLS = ((19, 2048), (19, 32768), (11, 2048), (11, 65536))     # the decision cells, first (A1, R4)
CELLS = "19:2048,19:32768,11:2048,11:65536,19:8192,11:8192,11:32768,4:2048,4:32768,4:65536"
SPLITS, WARPS, TILES = (4, 8, 16, 32), (4, 8), (32, 64)   # the FP8 Triton sweep; a tile is the tokens per loop step
BF16_CONFIGS = tuple((s, w, 32) for s in (16, 32) for w in (4, 8))   # the BF16 Triton sweep, group cells only
STOCK = {"stock_fp8_s32": ("fp8", 32), "stock_fp8_auto": ("fp8", 0), "stock_bf16_auto": ("bf16", 0),
         "stock_bf16_s32": ("bf16", 32)}
TRITON = {"triton_std_fp8": ("fp8", False), "triton_cmp_fp8": ("fp8", True), "triton_std_bf16": ("bf16", False),
          "triton_cmp_bf16": ("bf16", True)}
ORACLE = {"stock_fp8_s32_alias": "stock_fp8_s32", "triton_std_fp8_alias": "triton_std_fp8"}   # K = V, timing only
ROW = {False: 2 * D, True: D + 128}           # cache values per token and KV head (standard, compact)
# E-50 in-engine FA4 FP8 global attention per call (main plus combine kernel, us), from the group profiles.
TARGETS = {"19:2048": 238.3, "19:32768": 1358.1, "11:2048": 143.2, "11:65536": 1507.2}


class Timeout(BaseException):
    """SIGTERM from the session's `timeout`: recorded in run.json. Not an Exception, so no per-configuration handler
    catches it."""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cell_list(text: str) -> list[tuple[int, int]]:
    return [(int(c.split(":")[0]), int(c.split(":")[1])) for c in text.split(",") if c]


def nbytes(b: int, length: int, compact: bool, fp8: bool) -> int:
    return b * length * HK * ROW[compact] * (1 if fp8 else 2)


def triton_configs(name: str, cell: tuple[int, int] | None = None) -> list[tuple[int, int, int]]:
    """The Triton sweep (splits, warps, tile) of a cache dtype at a cell (None: the correctness cell)."""
    if name == "fp8":
        return list(itertools.product(SPLITS, WARPS, TILES))
    return list(BF16_CONFIGS) if cell is None or cell in GROUP_CELLS else []


def stages_for(args, name: str) -> int:
    return args.stages_bf16 if name == "bf16" and args.stages_bf16 else args.stages


def graph_time(step) -> tuple[float, float]:
    """step() runs one call. Returns the median microseconds per call and the replay spread (max / min - 1)."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(CALLS):
            step()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPLAYS):
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        z.record()
        z.synchronize()
        times.append(a.elapsed_time(z) * 1000.0 / CALLS)
    del g
    return statistics.median(times), max(times) / min(times) - 1


def fill_(t, gen, scale: float = 0.5) -> None:
    """Random values in chunks (FP8 has no randn), so no large float32 temporary is made."""
    import torch

    step = max(1, t.shape[0] // 16)
    for i in range(0, t.shape[0], step):
        part = t[i:i + step]
        part.copy_((torch.randn(part.shape, generator=gen, device=t.device, dtype=torch.bfloat16) * scale).to(t.dtype))


def stock_call(q, kc, vc, bt, seqs, cu, max_k, out, splits: int, fp8: bool):
    """The FA4 call of vLLM FlashAttentionImpl.forward (3966924d5) for a full-attention decode layer."""
    import torch
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    desc = torch.ones(q.shape[0], HK, dtype=torch.float32, device=q.device) if fp8 else None
    return lambda: flash_attn_varlen_func(
        q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seqs, max_seqlen_k=max_k,
        softmax_scale=1.0, causal=True, alibi_slopes=None, window_size=[-1, -1], block_table=bt, softcap=0.0,
        scheduler_metadata=None, fa_version=4, q_descale=None, k_descale=desc, v_descale=desc, dynamic_causal=None,
        num_splits=splits, s_aux=None, mask_mod=None, aux_tensors=None)


def rel_err(a, b) -> float:
    return float((a.float() - b.float()).norm() / b.float().norm())


def views(kv, cc):
    """K and V of the engine's (blocks, Hk, BS, 2D) tensor, and V and K_rot of the compact (blocks, Hk, BS, 640)
    tensor, each as [blocks, BS, Hk, X] views (cc may be None)."""
    k, v = kv.transpose(1, 2).split(D, dim=-1)
    cv, cr = cc.transpose(1, 2).split(D, dim=-1) if cc is not None else (None, None)
    return k, v, cv, cr


def make_cell(b: int, length: int, dev, dt, gen, compact: bool = True) -> dict:
    """The tensors of one cell for one cache dtype. The block table rows are padded to a multiple of 16 entries, so
    the correctness cell and the timed cells share one Triton specialization (no second compile)."""
    import torch

    nblk = math.ceil(length / BS)
    bt = torch.zeros(b, 16 * math.ceil(nblk / 16), dtype=torch.int32, device=dev)
    bt[:, :nblk] = torch.randperm(b * nblk, generator=gen, device=dev).view(b, nblk).to(torch.int32)
    kv = torch.empty(b * nblk, HK, BS, 2 * D, dtype=dt, device=dev)
    cc = torch.empty(b * nblk, HK, BS, D + 128, dtype=dt, device=dev) if compact else None
    for t in (kv, cc):
        if t is not None:
            fill_(t, gen)
    q = torch.randn(b, HQ, D, generator=gen, device=dev, dtype=torch.bfloat16) / math.sqrt(D)
    k, v, cv, cr = views(kv, cc)
    return {"q": q, "qc": q, "qr": q[..., :128].contiguous(), "k": k, "v": v, "cv": cv, "cr": cr, "kv": kv, "cc": cc,
            "bt": bt, "seqs": torch.full((b,), length, dtype=torch.int32, device=dev),
            "cu": torch.arange(b + 1, dtype=torch.int32, device=dev), "max_k": length,
            "out": torch.empty(b, HQ, D, dtype=torch.bfloat16, device=dev), "ws": {}}


def make_fn(kname: str, cfg: dict, c: dict, stages: int):
    """The step of one kernel configuration on the cell tensors c. A Triton step takes an optional `info` dict."""
    import compact_attn as ca

    base = ORACLE.get(kname, kname)
    alias = base != kname                       # the oracle reads V as K and as V
    if base in STOCK:
        return stock_call(c["q"], c["v"] if alias else c["k"], c["v"], c["bt"], c["seqs"], c["cu"], c["max_k"],
                          c["out"], cfg["num_splits"], STOCK[base][0] == "fp8")
    if cfg["splits"] not in c["ws"]:
        c["ws"][cfg["splits"]] = ca.workspace(c["q"], cfg["splits"])
    ws = c["ws"][cfg["splits"]]
    kw = dict(splits=cfg["splits"], num_warps=cfg["warps"], tile=cfg["tile"], num_stages=stages)
    if TRITON[base][1]:
        return lambda info=None: ca.triton_attn(c["qc"], c["cv"], c["bt"], c["seqs"], c["out"], ws, qr=c["qr"],
                                                cache_r=c["cr"], info=info, **kw)
    return lambda info=None: ca.triton_attn(c["q"], c["v"], c["bt"], c["seqs"], c["out"], ws,
                                            cache_k=c["v"] if alias else c["k"], info=info, **kw)


def timed(row: dict, factory, errors: list, where: str) -> None:
    """Time one configuration into `row`: the median us and the spread, or the error."""
    try:
        fn = factory()
        if row["kernel"].startswith("triton_"):
            info = {}
            fn(info)                            # one eager call: the compile, the registers and the shared memory
            row.update(info)
        us, spread = graph_time(fn)
        row.update(us=round(us, 2), spread=round(spread, 4), status="ok")
    except Exception as exc:  # recorded, not fatal (for example out of shared memory)
        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
        errors.append(f"{where}: {traceback.format_exc()[-800:]}")


def correctness(dev, fp8_dtype, errors: list, args, b: int = 2, length: int = 3000) -> dict:
    """A partial last page: Gemma 4 K=V data, every kernel configuration against the reference of its stored values,
    and both layouts against the exact attention."""
    import torch
    import compact_attn as ca

    gen = torch.Generator().manual_seed(51)
    w = 1 + 0.1 * torch.randn(D, generator=gen, dtype=torch.float64)
    nblk = math.ceil(length / BS)
    bt = torch.zeros(b, 16 * math.ceil(nblk / 16), dtype=torch.int32)
    bt[:, :nblk] = torch.randperm(b * nblk, generator=gen).view(b, nblk).to(torch.int32)
    q = torch.randn(b, HQ, D, generator=gen, dtype=torch.float64) / math.sqrt(D)
    q_rope = ca.rope(q, torch.full((b, HQ), float(length - 1), dtype=torch.float64))
    qc, qr = ca.compact_query(q_rope, w)
    kv = [ca.make_kv(length, HK, w, gen, torch.float64) for _ in range(b)]
    exact = torch.stack([ca.gqa_attend(q_rope[s], kv[s][0], kv[s][1]) for s in range(b)])
    kv32 = torch.zeros(b * nblk, HK, BS, 2 * D)       # float32 pages, engine layout; FP8 cannot be index-assigned on the CPU
    c32 = torch.zeros(b * nblk, HK, BS, D + 128)
    pad = nblk * BS - length
    for s in range(b):
        kp = torch.cat([kv[s][0], torch.zeros(pad, HK, D, dtype=torch.float64)]).view(nblk, BS, HK, D)
        vp = torch.cat([kv[s][1], torch.zeros(pad, HK, D, dtype=torch.float64)]).view(nblk, BS, HK, D)
        blocks = bt[s, :nblk].long()
        kv32[blocks] = torch.cat([kp, vp], -1).transpose(1, 2).float()
        c32[blocks] = ca.pack_compact(kp, vp).transpose(1, 2).float()
    seqs = torch.full((b,), length, dtype=torch.int32)
    res = {}
    for name, dt in (("fp8", fp8_dtype), ("bf16", torch.bfloat16)):
        kv_t, c_t = kv32.to(dt), c32.to(dt)
        k_h, v_h, _, _ = views(kv_t, c_t)
        # References from the stored (dequantized) values, as float32 copies (no FP8 gather on the CPU).
        ref_std = ca.paged_reference(q_rope.to(torch.bfloat16), k_h.float(), v_h.float(), bt, seqs)
        ref_cmp = ca.paged_reference(qc.to(torch.bfloat16), None, None, bt, seqs, qr=qr.to(torch.bfloat16),
                                     cache_c=c_t.transpose(1, 2).float())
        kv_d, c_d = kv_t.to(dev), c_t.to(dev)
        k, v, cv, cr = views(kv_d, c_d)
        qd, qcd, qrd = (t.to(torch.bfloat16).to(dev) for t in (q_rope, qc, qr))
        c = {"q": qd, "qc": qcd, "qr": qrd, "k": k, "v": v, "cv": cv, "cr": cr, "bt": bt.to(dev),
             "seqs": seqs.to(dev), "cu": torch.arange(b + 1, dtype=torch.int32, device=dev), "max_k": length,
             "out": torch.empty(b, HQ, D, dtype=torch.bfloat16, device=dev), "ws": {}}
        entry = {"exact_err_ref_standard": rel_err(ref_std, exact), "exact_err_ref_compact": rel_err(ref_cmp, exact)}
        stages = stages_for(args, name)
        for kname, (kdt, splits) in STOCK.items():
            if kdt != name:
                continue
            try:
                c["out"].zero_()
                make_fn(kname, {"num_splits": splits}, c, stages)()
                torch.cuda.synchronize()
                entry[kname] = {"err": rel_err(c["out"].cpu(), ref_std), "exact_err": rel_err(c["out"].cpu(), exact)}
            except Exception as exc:  # recorded, not fatal
                entry[kname] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                errors.append(f"correctness {kname}: {traceback.format_exc()[-800:]}")
        for kname, (kdt, compact) in TRITON.items():
            if kdt != name:
                continue
            by, exact_errs = {}, []
            for s_, w_, t_ in triton_configs(name):
                cfg = {"splits": s_, "warps": w_, "tile": t_}
                try:
                    c["out"].zero_()                  # a configuration that writes nothing must not pass
                    make_fn(kname, cfg, c, stages)()
                    torch.cuda.synchronize()
                    by[an.cfg_key(cfg)] = rel_err(c["out"].cpu(), ref_cmp if compact else ref_std)
                    exact_errs.append(rel_err(c["out"].cpu(), exact))
                except Exception as exc:  # recorded, not fatal (for example out of shared memory)
                    by[an.cfg_key(cfg)] = f"{type(exc).__name__}: {exc}"[:200]
                    errors.append(f"correctness {kname} {cfg}: {traceback.format_exc()[-800:]}")
            ok = [e for e in by.values() if isinstance(e, float)]
            entry[kname] = {"by_config": by, "err_max": max(ok, default=None),
                            "exact_err_max": max(exact_errs, default=None)}
        res[name] = entry
    return res


def bench_cell(b: int, length: int, dev, fp8_dtype, errors: list, args) -> list[dict]:
    import torch

    gen = torch.Generator(device=dev).manual_seed(b * 100_003 + length)
    rows = []
    for name, dt in (("fp8", fp8_dtype), ("bf16", torch.bfloat16)):
        cfgs = triton_configs(name, (b, length))
        c = make_cell(b, length, dev, dt, gen, compact=bool(cfgs))
        stages = stages_for(args, name)
        for kname, (kdt, splits) in STOCK.items():
            if kdt != name:
                continue
            row = {"kernel": kname, "batch": b, "context": length, "config": {"num_splits": splits},
                   "bytes": nbytes(b, length, False, name == "fp8")}
            timed(row, lambda: make_fn(kname, row["config"], c, stages), errors, f"{kname} {b}:{length}")
            rows.append(row)
        for kname, (kdt, compact) in TRITON.items():
            if kdt != name:
                continue
            for s_, w_, t_ in cfgs:
                row = {"kernel": kname, "batch": b, "context": length,
                       "config": {"splits": s_, "warps": w_, "tile": t_}, "stages": stages,
                       "bytes": nbytes(b, length, compact, name == "fp8")}
                timed(row, lambda: make_fn(kname, row["config"], c, stages), errors,
                      f"{kname} {row['config']} {b}:{length}")
                rows.append(row)
        del c
        torch.cuda.empty_cache()
    for r in rows:
        if r.get("status") == "ok":
            r["gbps"] = round(r["bytes"] / (r["us"] * 1e-6) / 1e9, 1)
        print(json.dumps({k: r.get(k) for k in ("kernel", "batch", "context", "config", "us", "status")}), flush=True)
    return rows


def confirm(plan: list[dict], dev, fp8_dtype, errors: list, args) -> list[dict]:
    """Time each FP8 row of `plan` REPEATS times, each time in a new CUDA graph. Odd repeats run the rows in the
    reverse order (A B .. B A), so a slow drift hits every row alike."""
    import torch

    cells, entries = {}, []
    for p in plan:
        cell = (p["batch"], p["context"])
        if cell not in cells:       # the seed of bench_cell: the same values and block table as the sweep
            gen = torch.Generator(device=dev).manual_seed(cell[0] * 100_003 + cell[1])
            cells[cell] = make_cell(*cell, dev, fp8_dtype, gen)
        e = dict(p, repeats=[], spreads=[], status="ok")
        try:
            e["fn"] = make_fn(p["kernel"], p["config"], cells[cell], args.stages)
        except Exception as exc:  # recorded, not fatal
            e.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
            errors.append(f"confirm {p}: {traceback.format_exc()[-800:]}")
        entries.append(e)
    for r in range(an.REPEATS):
        for e in entries if r % 2 == 0 else entries[::-1]:
            if e["status"] != "ok":
                continue
            try:
                us, spread = graph_time(e["fn"])
                e["repeats"].append(round(us, 2))
                e["spreads"].append(round(spread, 4))
            except Exception as exc:  # recorded, not fatal
                e.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
                errors.append(f"confirm {e['kernel']} {e['batch']}:{e['context']}: {traceback.format_exc()[-800:]}")
    for e in entries:
        e.pop("fn", None)
        if e["status"] == "ok":
            e["us"] = round(statistics.median(e["repeats"]), 2)
        print(json.dumps({k: e.get(k) for k in ("kernel", "batch", "context", "config", "us", "status")}), flush=True)
    del cells
    torch.cuda.empty_cache()
    return entries


def stub_time(args, kname: str, b: int, length: int, cfg: dict) -> float:
    """Synthetic us per call: fixed + max(DRAM time, on-chip time), times a configuration penalty. The stock FP8 kernel
    with 32 splits takes the E-50 values at the four target cells (times --stub-engine-ratio); an oracle (K = V)
    reads half the bytes."""
    base = ORACLE.get(kname, kname)
    alias = base != kname
    if base in STOCK:
        fp8 = STOCK[base][0] == "fp8"
        eff = args.stub_stock_eff if fp8 else args.stub_stock_bf16_eff
        t = args.stub_fixed_us + nbytes(b, length, False, fp8) / (eff * args.stub_bw * 1e9) * 1e6
        if base == "stock_fp8_s32" and f"{b}:{length}" in TARGETS:
            t = TARGETS[f"{b}:{length}"] * args.stub_engine_ratio
        return args.stub_fixed_us + (t - args.stub_fixed_us) * (1 - args.stub_stock_alias_gain) if alias else t
    kdt, compact = TRITON[base]
    eff = args.stub_cmp_eff if compact else args.stub_std_eff
    hbm = nbytes(b, length, compact, kdt == "fp8") / (2 if alias else 1) / (eff * args.stub_bw * 1e9) * 1e6
    chip = args.stub_onchip_ns * 1e-3 * b * length * HK * (1.125 if compact else 1.0)
    penalty = (1.0 + 0.02 * abs(math.log2(cfg["splits"] / 8)) + (0.01 if cfg["warps"] == 4 else 0.0) +
               (0.01 if cfg["tile"] == 64 else 0.0))
    return (args.stub_fixed_us + max(hbm, chip)) * penalty


def stub_rows(args, b: int, length: int, rng) -> list[dict]:
    rows = []
    for kname, (kdt, splits) in STOCK.items():
        rows.append({"kernel": kname, "batch": b, "context": length, "config": {"num_splits": splits},
                     "bytes": nbytes(b, length, False, kdt == "fp8")})
    for kname, (kdt, compact) in TRITON.items():
        for s_, w_, t_ in triton_configs(kdt, (b, length)):
            rows.append({"kernel": kname, "batch": b, "context": length,
                         "config": {"splits": s_, "warps": w_, "tile": t_}, "stages": stages_for(args, kdt),
                         "bytes": nbytes(b, length, compact, kdt == "fp8"), "regs": 128, "spills": 0, "shared": 65536})
    for r in rows:
        r.update(status="ok", spread=0.004,
                 us=round(stub_time(args, r["kernel"], b, length, r["config"]) * (1 + rng.uniform(-0.003, 0.003)), 2))
        r["gbps"] = round(r["bytes"] / (r["us"] * 1e-6) / 1e9, 1)
    return rows


def stub_correctness(args) -> dict:
    bad = set(filter(None, args.stub_bad_config.split(",")))
    res = {}
    for name in ("fp8", "bf16"):
        entry = {"exact_err_ref_standard": 0.03, "exact_err_ref_compact": 0.035}
        for kname, (kdt, _) in STOCK.items():
            if kdt == name:
                entry[kname] = {"err": 0.004, "exact_err": 0.03}
        for kname, (kdt, _) in TRITON.items():
            if kdt != name:
                continue
            by = {}
            for cfg in triton_configs(name):
                k = "/".join(map(str, cfg))
                wrong = kname == "triton_cmp_fp8" and (args.stub_bad_correct or k in bad)
                by[k] = 0.5 if wrong else 0.004
            entry[kname] = {"by_config": by, "err_max": max(by.values()), "exact_err_max": 0.03}
        res[name] = entry
    return res


def stub_confirm(args, plan: list[dict], rng) -> list[dict]:
    out = []
    for p in plan:
        t = stub_time(args, p["kernel"], p["batch"], p["context"], p["config"])
        reps = [round(t * (1 + rng.gauss(0, args.stub_repeat_noise)), 2) for _ in range(an.REPEATS)]
        out.append(dict(p, repeats=reps, spreads=[0.004] * an.REPEATS, status="ok", us=round(statistics.median(reps), 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--stages", type=int, default=2, help="Triton num_stages")
    ap.add_argument("--stages-bf16", type=int, default=0, help="num_stages of the BF16 kernels (0: --stages)")
    ap.add_argument("--fail-fast", choices=("auto", "on", "off"), default="auto",
                    help="stop after the correctness cell if a deciding FP8 kernel fails it (auto: on SM90)")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-seed", type=int, default=0)
    ap.add_argument("--stub-bw", type=float, default=3045.0, help="copy GB/s")
    ap.add_argument("--stub-stock-eff", type=float, default=0.70, help="stock FP8 kernel bandwidth / copy")
    ap.add_argument("--stub-stock-bf16-eff", type=float, default=0.85)
    ap.add_argument("--stub-std-eff", type=float, default=0.70, help="Triton standard kernel bandwidth / copy")
    ap.add_argument("--stub-cmp-eff", type=float, default=0.70, help="Triton compact kernel bandwidth / copy")
    ap.add_argument("--stub-fixed-us", type=float, default=30.0)
    ap.add_argument("--stub-engine-ratio", type=float, default=1.0, help="stock FP8 s32 against the E-50 values")
    ap.add_argument("--stub-onchip-ns", type=float, default=0.0, help="on-chip ns per token and KV head (0: off)")
    ap.add_argument("--stub-stock-alias-gain", type=float, default=0.4, help="stock oracle: share of the growth removed")
    ap.add_argument("--stub-repeat-noise", type=float, default=0.002, help="relative noise of a confirmation repeat")
    ap.add_argument("--stub-bad-config", default="", help="compact FP8 configurations (s/w/t,...) that fail correctness")
    ap.add_argument("--stub-bad-correct", action="store_true", help="every compact FP8 configuration fails correctness")
    ap.add_argument("--stub-hang-after-groups", type=float, default=0.0, help="seconds to wait after ok_groups")
    ap.add_argument("--stub-fail", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    record = {"args": {k: str(v) for k, v in vars(args).items()}, "started_at": now(), "status": "started"}
    if args.stub:
        rng, dev, fp8, sm90 = random.Random(args.stub_seed), None, None, True
        record.update(stub=True, gpu={"name": "STUB NVIDIA H100 80GB HBM3"})
    else:
        import torch

        cap = torch.cuda.get_device_capability(0)
        record.update(gpu={"name": torch.cuda.get_device_name(0), "capability": f"{cap[0]}.{cap[1]}",
                           "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))},
                      torch=torch.__version__)
        for mod in ("triton", "vllm"):
            try:
                record[mod] = __import__(mod).__version__
            except Exception as exc:  # reported only
                record[mod] = f"unavailable: {type(exc).__name__}"
        rng, dev, fp8, sm90 = None, torch.device("cuda"), torch.float8_e4m3fn, cap[0] == 9
    fail_fast = args.fail_fast == "on" or (args.fail_fast == "auto" and sm90)
    record["fail_fast_mode"] = "on" if fail_fast else "off"
    result = {"cells": [], "correctness": None, "confirm": None}

    def save(status: str | None = None) -> None:
        (out / "kbench.json").write_text(json.dumps(result, indent=1))
        if status:
            (out / "run.json").write_text(json.dumps(record | {"status": status, "errors": errors[-50:]}, indent=1))

    def on_term(signum, frame):
        raise Timeout("SIGTERM (the session timeout)")

    signal.signal(signal.SIGTERM, on_term)
    try:
        if not args.skip_correctness:
            result["correctness"] = stub_correctness(args) if args.stub else correctness(dev, fp8, errors, args)
            save()
            bad = an.deciding_failures(result["correctness"])
            if bad and fail_fast:
                record["fail_fast"] = bad
                raise RuntimeError(f"fail fast: {bad} did not pass the correctness cell")
        if args.stub and args.stub_fail:
            raise RuntimeError("stub failure (--stub-fail)")
        done = set()
        for b, length in cell_list(args.cells):
            rows = stub_rows(args, b, length, rng) if args.stub else bench_cell(b, length, dev, fp8, errors, args)
            result["cells"].extend(rows)
            done.add((b, length))
            save()
            if result["confirm"] is None and set(GROUP_CELLS) <= done:
                plan = an.confirm_plan(an.validated(result["cells"], result["correctness"]))
                result["confirm"] = stub_confirm(args, plan, rng) if args.stub else confirm(plan, dev, fp8, errors, args)
                save("ok_groups")                       # the decision data is complete (A1, R4)
                if args.stub_hang_after_groups:
                    time.sleep(args.stub_hang_after_groups)
        record.update(status="ok", finished_at=now())
    except BaseException as exc:  # recorded, then re-raised so the caller sees a non-zero exit
        late = result["confirm"] is not None             # after the confirmation, the decision data stays usable
        record.update(status="ok_groups" if late else "error")
        record["late_error" if late else "error"] = f"{type(exc).__name__}: {exc}"[:2000]
        save(record["status"])
        raise
    save(record["status"])
    print(json.dumps({"status": record["status"], "errors": len(errors)}))


if __name__ == "__main__":
    main()
