"""E-52 kernel microbenchmark (README Sections 4 to 6): the decode attention of the Gemma 4 31B global layers (32 query
heads, 4 KV heads, head_dim 512, scale 1.0, no soft-cap) on one GPU, with the stock vLLM FA4 kernel and the Triton
kernels of `../gemma4_kv_kernel/compact_attn.py`, for the FP8 and the BF16 cache.

What is new against E-51 (D-85, R-70):
- FP16 compute ("h" kernels): the cache values, the query and P become FP16 before each dot (E-51: BF16).
- The views layout ("cmpv" kernels): the compact row [K_rot | V_unrot | V_rot] is read by the unchanged standard kernel
  through two overlapping 512-value views (`compact_attn.views_of`), with the query of `views_query`. The stock FA4
  kernel reads the same views ("_views" rows).
- The wide Triton sweep (R-70 recommendation 5): splits {16, 32, 64} x warps {2, 4, 8} x tile {16, 32, 64} x
  num_stages {2, 3} for the two primary families at the group cells and the correctness cell.
- Timing (R-70): 30 replays; the spread is that of the central 20; the SM clock and the clock event reasons from NVML
  every 10 replays.
- Controls (R-70): BF16 K = V rows (sweep), FP8 K = V rows and pool rows (confirmation). A pool row reads every page
  from an 8 MB pool of pages, so after the first call its bytes come from L2.
- The PTX and TTGIR of each Triton kernel at the E-51 configuration (32/4/32/2), with instruction counts.

Order (as E-51 A1): the correctness cell, the four group cells, the confirmation phase ("ok_groups" after it), then
the reported cells. A configuration that raises is recorded, not fatal. --stub runs the same flow with synthetic
numbers (no GPU, no vLLM, no Triton) for the Mac tests.

Usage: kbench_e52.py --out <dir> [--cells B:L,...] [--fail-fast auto|on|off] [--no-asm] [--stub ...]
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
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import analyze_e52 as an  # noqa: E402  (the selection rule, the tolerances, the families and the repeat count)

HQ, HK, D, BS = 32, 4, 512, 64
CALLS, REPLAYS, CENTRAL = 20, 30, 20          # R-70: the spread of the central 20 of 30 replays
CLOCK_EVERY = 10                              # NVML clock sample after every 10th replay
GROUP_CELLS = ((19, 2048), (19, 32768), (11, 2048), (11, 65536))     # the decision cells, first
CELLS = "19:2048,19:32768,11:2048,11:65536,19:8192,11:8192,4:2048,4:32768"
SPLITS, WARPS, TILES, STAGES = (16, 32, 64), (2, 4, 8), (16, 32, 64), (2, 3)   # the wide sweep
LITE = ((32, 4, 16, 2), (32, 4, 32, 2), (64, 4, 16, 2), (64, 4, 32, 2))   # the primary families at the reported cells
E51_CFG = (32, 4, 32, 2)                      # the E-51 selection: the diagnostic kernels run only this
BF16_CONFIGS = tuple((s, w, 32, 2) for s in (16, 32) for w in (4, 8))   # as E-51, group cells only
POOL_BYTES = 8 << 20                          # pool rows: every page from an 8 MB pool (R-70)
STOCK = {  # name: (cache dtype, num_splits, layout)
    "stock_fp8_s32": ("fp8", 32, "std"), "stock_fp8_auto": ("fp8", 0, "std"),
    "stock_fp8_s32_views": ("fp8", 32, "views"), "stock_fp8_auto_views": ("fp8", 0, "views"),
    "stock_bf16_s32": ("bf16", 32, "std"), "stock_bf16_auto": ("bf16", 0, "std"),
    "stock_bf16_s32_views": ("bf16", 32, "views"), "stock_bf16_auto_views": ("bf16", 0, "views")}
TRITON = {  # name: (cache dtype, layout, compute dtype, sweep)
    "triton_std_fp8h": ("fp8", "std", "fp16", "wide"), "triton_cmpv_fp8h": ("fp8", "views", "fp16", "wide"),
    "triton_std_fp8": ("fp8", "std", "bf16", "e51"), "triton_cmpv_fp8": ("fp8", "views", "bf16", "e51"),
    "triton_cmp_fp8": ("fp8", "fused", "bf16", "e51"), "triton_cmp_fp8h": ("fp8", "fused", "fp16", "e51"),
    "triton_std_bf16": ("bf16", "std", "bf16", "bf16"), "triton_cmpv_bf16": ("bf16", "views", "bf16", "bf16"),
    "triton_cmp_bf16": ("bf16", "fused", "bf16", "bf16")}
ORACLE = an.ORACLE                            # K = V (the V view as K and V), timing only
POOL = an.POOL                                # pages from the pool, timing only
SWEEP_ALIAS = ("stock_bf16_s32_alias", "triton_std_bf16_alias")   # the BF16 K = V control, in the sweep
ROW = {"std": 2 * D, "views": D + 128, "fused": D + 128}          # cache values per token and KV head
PTX_OPS = ("cvt.rn.f16x2.e4m3x2", "cvt.bf16.f16", "cvt.f32.f16", "cvt.rn.bf16.f32", "mma.sync", "wgmma.mma_async",
           "ldmatrix", "ld.shared", "st.shared", "bar.sync", "cp.async", "ld.global")
_NV: dict = {}


class Timeout(BaseException):
    """SIGTERM from the session's `timeout`: recorded in run.json. Not an Exception, so no per-configuration handler
    catches it."""


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cell_list(text: str) -> list[tuple[int, int]]:
    return [(int(c.split(":")[0]), int(c.split(":")[1])) for c in text.split(",") if c]


def base_of(kname: str) -> str:
    return ORACLE.get(kname) or POOL.get(kname) or kname


def layout_of(kname: str) -> str:
    base = base_of(kname)
    return STOCK[base][2] if base in STOCK else TRITON[base][1]


def nbytes(b: int, length: int, layout: str, fp8: bool) -> int:
    return b * length * HK * ROW[layout] * (1 if fp8 else 2)


def cfg_dict(c: tuple) -> dict:
    return {"splits": c[0], "warps": c[1], "tile": c[2], "stages": c[3]}


def triton_configs(kind: str, cell: tuple[int, int] | None = None) -> list[tuple[int, int, int, int]]:
    """The Triton configurations (splits, warps, tile, stages) of a sweep kind at a cell (None: the correctness cell)."""
    if kind == "wide":
        return list(itertools.product(SPLITS, WARPS, TILES, STAGES)) if cell is None or cell in GROUP_CELLS \
            else list(LITE)
    if kind == "e51":
        return [E51_CFG]
    return list(BF16_CONFIGS) if cell is None or cell in GROUP_CELLS else []


def sm_clock() -> tuple[int, int] | None:
    """(SM clock in MHz, clock event reasons bit mask) from NVML, or None (no root needed)."""
    try:
        import pynvml

        if "h" not in _NV:
            pynvml.nvmlInit()
            _NV["h"] = pynvml.nvmlDeviceGetHandleByIndex(0)
        reasons = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
            pynvml.nvmlDeviceGetCurrentClocksThrottleReasons
        return int(pynvml.nvmlDeviceGetClockInfo(_NV["h"], pynvml.NVML_CLOCK_SM)), int(reasons(_NV["h"]))
    except Exception:  # reported as missing
        return None


def central_spread(times: list[float]) -> float:
    """max / min - 1 of the central CENTRAL values (R-70: one outlier replay must not decide V4)."""
    srt = sorted(times)
    lo = max(0, (len(srt) - CENTRAL) // 2)
    mid = srt[lo:lo + CENTRAL] or srt
    return mid[-1] / mid[0] - 1


def graph_time(step) -> tuple[float, float, dict]:
    """step() runs one call. Returns the median microseconds per call, the central spread and the clock record."""
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
    times, clocks, caps = [], [], 0
    for i in range(REPLAYS):
        a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        z.record()
        z.synchronize()
        times.append(a.elapsed_time(z) * 1000.0 / CALLS)
        if i % CLOCK_EVERY == CLOCK_EVERY - 1:
            clk = sm_clock()
            if clk:
                clocks.append(clk[0])
                caps += bool(clk[1] & 0x4)
    del g
    clock = {"clock_mhz": round(statistics.mean(clocks)) if clocks else None,
             "clock_min": min(clocks) if clocks else None, "cap_samples": caps, "clock_samples": len(clocks)}
    return statistics.median(times), central_spread(times), clock


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
    """[blocks, BS, Hk, X] views: K and V of the engine's (blocks, Hk, BS, 2D) tensor; V and K_rot of the fused row
    order [V | K_rot]; the K and V views of the views row order [K_rot | V_unrot | V_rot] (cc may be None)."""
    import compact_attn as ca

    k, v = kv.transpose(1, 2).split(D, dim=-1)
    if cc is None:
        return {"k": k, "v": v}
    t = cc.transpose(1, 2)
    kview, vview = ca.views_of(t)
    return {"k": k, "v": v, "cv": t[..., :D], "cr": t[..., D:], "kview": kview, "vview": vview}


def pool_table(bt, block_bytes: int):
    """The block table with every entry mapped into a pool of POOL_BYTES (R-70 pool probe)."""
    return bt % max(1, POOL_BYTES // block_bytes)


def make_cell(b: int, length: int, dev, dt, gen, compact: bool = True) -> dict:
    """The tensors of one cell for one cache dtype. The block table rows are padded to a multiple of 16 entries, so
    the correctness cell and the timed cells share one Triton specialization. The compact tensor serves both row
    orders (the values are random, so only the reads differ)."""
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
    el = 1 if dt != torch.bfloat16 else 2
    return {"q": q, "qv": q, "qc": q, "qr": q[..., :128].contiguous(), **views(kv, cc), "kv": kv, "cc": cc, "bt": bt,
            "bt_pool_std": pool_table(bt, BS * HK * ROW["std"] * el),
            "bt_pool_cmp": pool_table(bt, BS * HK * ROW["views"] * el),
            "seqs": torch.full((b,), length, dtype=torch.int32, device=dev),
            "cu": torch.arange(b + 1, dtype=torch.int32, device=dev), "max_k": length,
            "out": torch.empty(b, HQ, D, dtype=torch.bfloat16, device=dev), "ws": {}}


def make_fn(kname: str, cfg: dict, c: dict):
    """The step of one kernel configuration on the cell tensors c. A Triton step takes an optional `info` dict."""
    import compact_attn as ca

    base = base_of(kname)
    alias, pool = kname in ORACLE, kname in POOL
    if base in STOCK:
        kdt, _, layout = STOCK[base]
        q, kk, vv = (c["qv"], c["kview"], c["vview"]) if layout == "views" else (c["q"], c["k"], c["v"])
        bt = (c["bt_pool_cmp"] if layout == "views" else c["bt_pool_std"]) if pool else c["bt"]
        return stock_call(q, vv if alias else kk, vv, bt, c["seqs"], c["cu"], c["max_k"], c["out"],
                          cfg["num_splits"], kdt == "fp8")
    _, layout, cast, _ = TRITON[base]
    if cfg["splits"] not in c["ws"]:
        c["ws"][cfg["splits"]] = ca.workspace(c["q"], cfg["splits"])
    ws = c["ws"][cfg["splits"]]
    kw = dict(splits=cfg["splits"], num_warps=cfg["warps"], tile=cfg["tile"], num_stages=cfg["stages"], cast=cast)
    if layout == "fused":
        return lambda info=None: ca.triton_attn(c["qc"], c["cv"], c["bt"], c["seqs"], c["out"], ws, qr=c["qr"],
                                                cache_r=c["cr"], info=info, **kw)
    q, kk, vv = (c["qv"], c["kview"], c["vview"]) if layout == "views" else (c["q"], c["k"], c["v"])
    bt = (c["bt_pool_cmp"] if layout == "views" else c["bt_pool_std"]) if pool else c["bt"]
    return lambda info=None: ca.triton_attn(q, vv, bt, c["seqs"], c["out"], ws, cache_k=vv if alias else kk,
                                            info=info, **kw)


def ptx_counts(ptx: str) -> dict:
    """Static counts of the instructions that H4 and the fused defect name (README Section 5)."""
    lines = [ln.strip() for ln in ptx.splitlines()]
    return {op: sum(1 for ln in lines if ln.startswith(op)) for op in PTX_OPS}


def timed(row: dict, factory, errors: list, where: str, asm_dir: Path | None = None) -> None:
    """Time one configuration into `row`: the median us, the central spread and the clock record, or the error."""
    try:
        fn = factory()
        if row["kernel"].startswith("triton_"):
            info = {"asm": asm_dir is not None}
            fn(info)                            # one eager call: the compile, the registers and the shared memory
            ptx, ttgir = info.pop("ptx", None), info.pop("ttgir", None)
            row.update(info)
            if asm_dir is not None and ptx:
                asm_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{row['kernel']}_{an.cfg_key(row['config']).replace('/', '-')}"
                (asm_dir / f"{stem}.ptx").write_text(ptx)
                if ttgir:
                    (asm_dir / f"{stem}.ttgir").write_text(ttgir)
                row["ptx_counts"] = ptx_counts(ptx)
        us, spread, clock = graph_time(fn)
        row.update(us=round(us, 2), spread=round(spread, 4), status="ok", **clock)
    except Exception as exc:  # recorded, not fatal (for example out of shared memory)
        row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
        errors.append(f"{where}: {traceback.format_exc()[-800:]}")


def correctness(dev, fp8_dtype, errors: list, args, b: int = 2, length: int = 3000) -> dict:
    """A partial last page: Gemma 4 K=V data, every kernel configuration against the reference of its stored values,
    and each layout against the exact attention (the views outputs after `unpermute`)."""
    import torch
    import compact_attn as ca

    gen = torch.Generator().manual_seed(52)
    w = 1 + 0.1 * torch.randn(D, generator=gen, dtype=torch.float64)
    nblk = math.ceil(length / BS)
    bt = torch.zeros(b, 16 * math.ceil(nblk / 16), dtype=torch.int32)
    bt[:, :nblk] = torch.randperm(b * nblk, generator=gen).view(b, nblk).to(torch.int32)
    q = torch.randn(b, HQ, D, generator=gen, dtype=torch.float64) / math.sqrt(D)
    q_rope = ca.rope(q, torch.full((b, HQ), float(length - 1), dtype=torch.float64))
    qc, qr = ca.compact_query(q_rope, w)
    qv = ca.views_query(q_rope, w)
    kv = [ca.make_kv(length, HK, w, gen, torch.float64) for _ in range(b)]
    exact = torch.stack([ca.gqa_attend(q_rope[s], kv[s][0], kv[s][1]) for s in range(b)])
    kv32 = torch.zeros(b * nblk, HK, BS, 2 * D)       # float32 pages; FP8 cannot be index-assigned on the CPU
    c32, v32 = torch.zeros(b * nblk, HK, BS, D + 128), torch.zeros(b * nblk, HK, BS, D + 128)
    pad = nblk * BS - length
    for s in range(b):
        kp = torch.cat([kv[s][0], torch.zeros(pad, HK, D, dtype=torch.float64)]).view(nblk, BS, HK, D)
        vp = torch.cat([kv[s][1], torch.zeros(pad, HK, D, dtype=torch.float64)]).view(nblk, BS, HK, D)
        blocks = bt[s, :nblk].long()
        kv32[blocks] = torch.cat([kp, vp], -1).transpose(1, 2).float()
        c32[blocks] = ca.pack_compact(kp, vp).transpose(1, 2).float()
        v32[blocks] = ca.pack_views(kp, vp).transpose(1, 2).float()
    seqs = torch.full((b,), length, dtype=torch.int32)
    res = {}
    for name, dt in (("fp8", fp8_dtype), ("bf16", torch.bfloat16)):
        kv_t, c_t, v_t = kv32.to(dt), c32.to(dt), v32.to(dt)
        h = views(kv_t, v_t)
        # References from the stored (dequantized) values, as float32 copies (no FP8 gather on the CPU).
        ref = {"std": ca.paged_reference(q_rope.to(torch.bfloat16), h["k"].float(), h["v"].float(), bt, seqs),
               "fused": ca.paged_reference(qc.to(torch.bfloat16), None, None, bt, seqs, qr=qr.to(torch.bfloat16),
                                           cache_c=c_t.transpose(1, 2).float()),
               "views": ca.paged_reference(qv.to(torch.bfloat16), h["kview"].float(), h["vview"].float(), bt, seqs)}
        exact_of = {"std": lambda o: o, "fused": lambda o: o, "views": ca.unpermute}
        dv, dc = views(kv_t.to(dev), v_t.to(dev)), views(kv_t.to(dev), c_t.to(dev))
        c = {"q": q_rope.to(torch.bfloat16).to(dev), "qv": qv.to(torch.bfloat16).to(dev),
             "qc": qc.to(torch.bfloat16).to(dev), "qr": qr.to(torch.bfloat16).to(dev),
             "k": dv["k"], "v": dv["v"], "kview": dv["kview"], "vview": dv["vview"], "cv": dc["cv"], "cr": dc["cr"],
             "bt": bt.to(dev), "seqs": seqs.to(dev), "cu": torch.arange(b + 1, dtype=torch.int32, device=dev),
             "max_k": length, "out": torch.empty(b, HQ, D, dtype=torch.bfloat16, device=dev), "ws": {}}
        entry = {f"exact_err_ref_{k}": rel_err(exact_of[k](r), exact) for k, r in ref.items()}
        for kname, (kdt, splits, layout) in STOCK.items():
            if kdt != name:
                continue
            try:
                c["out"].zero_()
                make_fn(kname, {"num_splits": splits}, c)()
                torch.cuda.synchronize()
                o = c["out"].cpu()
                entry[kname] = {"err": rel_err(o, ref[layout]), "exact_err": rel_err(exact_of[layout](o), exact)}
            except Exception as exc:  # recorded, not fatal
                entry[kname] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                errors.append(f"correctness {kname}: {traceback.format_exc()[-800:]}")
        for kname, (kdt, layout, _, kind) in TRITON.items():
            if kdt != name:
                continue
            by, exact_errs = {}, []
            for cfg_t in triton_configs(kind):
                cfg = cfg_dict(cfg_t)
                try:
                    c["out"].zero_()                  # a configuration that writes nothing must not pass
                    make_fn(kname, cfg, c)()
                    torch.cuda.synchronize()
                    o = c["out"].cpu()
                    by[an.cfg_key(cfg)] = rel_err(o, ref[layout])
                    exact_errs.append(rel_err(exact_of[layout](o), exact))
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
    rows, group = [], (b, length) in GROUP_CELLS
    asm_dir = args.out / "asm" if args.asm and (b, length) == (19, 32768) else None
    for name, dt in (("fp8", fp8_dtype), ("bf16", torch.bfloat16)):
        c = make_cell(b, length, dev, dt, gen)
        stock = [k for k, v in STOCK.items() if v[0] == name]
        stock += [k for k in SWEEP_ALIAS if k.startswith("stock") and name in k] if group else []
        for kname in stock:
            row = {"kernel": kname, "batch": b, "context": length,
                   "config": {"num_splits": STOCK[base_of(kname)][1]},
                   "bytes": nbytes(b, length, layout_of(kname), name == "fp8")}
            timed(row, lambda: make_fn(kname, row["config"], c), errors, f"{kname} {b}:{length}")
            rows.append(row)
        trit = [(k, v[3]) for k, v in TRITON.items() if v[0] == name]
        trit += [(k, TRITON[base_of(k)][3]) for k in SWEEP_ALIAS if k.startswith("triton") and name in k] if group \
            else []
        for kname, kind in trit:
            for cfg_t in triton_configs(kind, (b, length)):
                row = {"kernel": kname, "batch": b, "context": length, "config": cfg_dict(cfg_t),
                       "bytes": nbytes(b, length, layout_of(kname), name == "fp8")}
                dump = asm_dir if cfg_t == E51_CFG and kname in TRITON else None
                timed(row, lambda: make_fn(kname, row["config"], c), errors,
                      f"{kname} {row['config']} {b}:{length}", dump)
                rows.append(row)
        del c
        torch.cuda.empty_cache()
    for r in rows:
        if r.get("status") == "ok":
            r["gbps"] = round(r["bytes"] / (r["us"] * 1e-6) / 1e9, 1)
        print(json.dumps({k: r.get(k) for k in ("kernel", "batch", "context", "config", "us", "status")}), flush=True)
    return rows


def confirm(plan: list[dict], dev, fp8_dtype, errors: list, args) -> list[dict]:
    """Time each FP8 row of `plan` REPEATS times, each time in a new CUDA graph, after one untimed pass. Odd repeats
    run the rows in the reverse order (A B .. B A), so a slow drift hits every row alike."""
    import torch

    cells, entries = {}, []
    for p in plan:
        cell = (p["batch"], p["context"])
        if cell not in cells:       # the seed of bench_cell: the same values and block table as the sweep
            gen = torch.Generator(device=dev).manual_seed(cell[0] * 100_003 + cell[1])
            cells[cell] = make_cell(*cell, dev, fp8_dtype, gen)
        e = dict(p, repeats=[], spreads=[], clocks=[], caps=[], status="ok")
        try:
            e["fn"] = make_fn(p["kernel"], p["config"], cells[cell])
        except Exception as exc:  # recorded, not fatal
            e.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
            errors.append(f"confirm {p}: {traceback.format_exc()[-800:]}")
        entries.append(e)
    for e in entries:               # E-51 A2: one untimed pass (a cold first repeat)
        if e["status"] == "ok":
            try:
                graph_time(e["fn"])
            except Exception as exc:  # recorded, not fatal
                e.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
                errors.append(f"confirm warm-up {e['kernel']}: {traceback.format_exc()[-800:]}")
    for r in range(an.REPEATS):
        for e in entries if r % 2 == 0 else entries[::-1]:
            if e["status"] != "ok":
                continue
            try:
                us, spread, clock = graph_time(e["fn"])
                e["repeats"].append(round(us, 2))
                e["spreads"].append(round(spread, 4))
                e["clocks"].append(clock.get("clock_mhz"))
                e["caps"].append(clock.get("cap_samples"))
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
    """Synthetic us per call: fixed + max(DRAM time, on-chip time), times a configuration penalty. DRAM bytes follow
    the layout (views: 640 values, if the two views share their reads), half for K = V, none for a pool row. The
    on-chip time per token and KV head depends on the compute dtype (--stub-onchip-bf16-ns, --stub-onchip-fp16-ns);
    the fused layout adds --stub-fused-ns. The stock FP8 kernel with 32 splits takes the E-50 values at the four
    target cells (times --stub-engine-ratio); the stock views rows remove --stub-stock-views-gain of the growth."""
    base = base_of(kname)
    alias, pool = kname in ORACLE, kname in POOL
    fp8 = "fp8" in base
    if base in STOCK:
        layout = STOCK[base][2]
        eff = args.stub_stock_eff if fp8 else args.stub_stock_bf16_eff
        t = args.stub_fixed_us + nbytes(b, length, "std", fp8) / (eff * args.stub_bw * 1e9) * 1e6
        if base.startswith("stock_fp8_s32") and f"{b}:{length}" in an.TARGETS_STR:
            t = an.TARGETS_STR[f"{b}:{length}"] * args.stub_engine_ratio
        gain = ((args.stub_stock_alias_gain if fp8 else 0.5) if alias else 0.0) + \
            (args.stub_stock_views_gain if layout == "views" else 0.0)
        gain += args.stub_pool_gain if pool else 0.0
        return args.stub_fixed_us + (t - args.stub_fixed_us) * (1 - gain)
    _, layout, cast, _ = TRITON[base]
    eff = args.stub_cmp_eff if layout != "std" else args.stub_std_eff
    dram = 0.0 if pool else nbytes(b, length, layout, fp8) / (2 if alias else 1)
    hbm = dram / (eff * args.stub_bw * 1e9) * 1e6
    ns = (args.stub_onchip_fp16_ns if cast == "fp16" else args.stub_onchip_bf16_ns) if fp8 else 0.0
    ns += args.stub_fused_ns if layout == "fused" else 0.0
    chip = ns * 1e-3 * b * length * HK
    penalty = (1.0 + 0.02 * abs(math.log2(cfg["splits"] / 32)) + 0.01 * abs(math.log2(cfg["warps"] / 4)) +
               0.01 * abs(math.log2(cfg["tile"] / 32)) + (0.005 if cfg["stages"] == 3 else 0.0))
    return (args.stub_fixed_us + max(hbm, chip)) * penalty


def stub_rows(args, b: int, length: int, rng) -> list[dict]:
    group = (b, length) in GROUP_CELLS
    rows = []
    for name in ("fp8", "bf16"):
        stock = [k for k, v in STOCK.items() if v[0] == name]
        stock += [k for k in SWEEP_ALIAS if k.startswith("stock") and name in k] if group else []
        for kname in stock:
            rows.append({"kernel": kname, "batch": b, "context": length,
                         "config": {"num_splits": STOCK[base_of(kname)][1]},
                         "bytes": nbytes(b, length, layout_of(kname), name == "fp8")})
        trit = [(k, v[3]) for k, v in TRITON.items() if v[0] == name]
        trit += [(k, TRITON[base_of(k)][3]) for k in SWEEP_ALIAS if k.startswith("triton") and name in k] if group \
            else []
        for kname, kind in trit:
            for cfg_t in triton_configs(kind, (b, length)):
                rows.append({"kernel": kname, "batch": b, "context": length, "config": cfg_dict(cfg_t),
                             "bytes": nbytes(b, length, layout_of(kname), name == "fp8"), "regs": 128, "spills": 0,
                             "shared": 65536})
    for r in rows:
        r.update(status="ok", spread=0.004, clock_mhz=1980, clock_min=1980, cap_samples=0, clock_samples=3,
                 us=round(stub_time(args, r["kernel"], b, length, r["config"]) * (1 + rng.uniform(-0.003, 0.003)), 2))
        r["gbps"] = round(r["bytes"] / (r["us"] * 1e-6) / 1e9, 1)
    return rows


def stub_correctness(args) -> dict:
    bad = set(filter(None, args.stub_bad_config.split(",")))
    res = {}
    for name in ("fp8", "bf16"):
        entry = {"exact_err_ref_std": 0.03, "exact_err_ref_fused": 0.035, "exact_err_ref_views": 0.035}
        for kname, (kdt, _, _) in STOCK.items():
            if kdt == name:
                entry[kname] = {"err": 0.004, "exact_err": 0.03}
        for kname, (kdt, _, _, kind) in TRITON.items():
            if kdt != name:
                continue
            by = {}
            for cfg_t in triton_configs(kind):
                k = an.cfg_key(cfg_dict(cfg_t))
                wrong = kname == "triton_cmpv_fp8h" and (args.stub_bad_correct or k in bad)
                by[k] = 0.5 if wrong else 0.004
            entry[kname] = {"by_config": by, "err_max": max(by.values()), "exact_err_max": 0.03}
        res[name] = entry
    return res


def stub_confirm(args, plan: list[dict], rng) -> list[dict]:
    out = []
    for p in plan:
        t = stub_time(args, p["kernel"], p["batch"], p["context"], p["config"])
        reps = [round(t * (1 + rng.gauss(0, args.stub_repeat_noise)), 2) for _ in range(an.REPEATS)]
        out.append(dict(p, repeats=reps, spreads=[args.stub_spread] * an.REPEATS, clocks=[1980] * an.REPEATS,
                        caps=[0] * an.REPEATS, status="ok", us=round(statistics.median(reps), 2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--fail-fast", choices=("auto", "on", "off"), default="auto",
                    help="stop after the correctness cell if a deciding FP8 kernel fails it (auto: on SM90)")
    ap.add_argument("--no-asm", dest="asm", action="store_false", help="do not save the PTX and TTGIR")
    ap.add_argument("--skip-correctness", action="store_true")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-seed", type=int, default=0)
    ap.add_argument("--stub-bw", type=float, default=3045.0, help="copy GB/s")
    ap.add_argument("--stub-stock-eff", type=float, default=0.70, help="stock FP8 kernel bandwidth / copy")
    ap.add_argument("--stub-stock-bf16-eff", type=float, default=1.0)
    ap.add_argument("--stub-std-eff", type=float, default=0.95, help="Triton standard kernel bandwidth / copy")
    ap.add_argument("--stub-cmp-eff", type=float, default=0.95, help="Triton compact kernels bandwidth / copy")
    ap.add_argument("--stub-fixed-us", type=float, default=30.0)
    ap.add_argument("--stub-engine-ratio", type=float, default=1.0, help="stock FP8 s32 against the E-50 values")
    ap.add_argument("--stub-onchip-bf16-ns", type=float, default=0.435, help="on-chip ns per token and KV head, BF16")
    ap.add_argument("--stub-onchip-fp16-ns", type=float, default=0.10, help="the same with FP16 compute")
    ap.add_argument("--stub-fused-ns", type=float, default=0.20, help="extra on-chip ns of the fused layout")
    ap.add_argument("--stub-stock-alias-gain", type=float, default=0.10, help="stock oracle: share of the growth removed")
    ap.add_argument("--stub-stock-views-gain", type=float, default=0.075, help="stock views: share removed")
    ap.add_argument("--stub-pool-gain", type=float, default=0.10, help="stock pool rows: share removed")
    ap.add_argument("--stub-repeat-noise", type=float, default=0.002, help="relative noise of a confirmation repeat")
    ap.add_argument("--stub-spread", type=float, default=0.004, help="central replay spread of a confirmation row")
    ap.add_argument("--stub-bad-config", default="", help="views FP16 configurations (s/w/t/st,...) that fail")
    ap.add_argument("--stub-bad-correct", action="store_true", help="every views FP16 configuration fails")
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
                      torch=torch.__version__, nvml_clock=sm_clock())
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
                save("ok_groups")                       # the decision data is complete
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
