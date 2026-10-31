"""E-51 kernel microbenchmark (README Sections 4 to 6): the decode attention of the Gemma 4 31B global layers (32 query
heads, 4 KV heads, head_dim 512, scale 1.0, no soft-cap) on one GPU, with the stock vLLM FA4 kernel and with the
two Triton layouts of `compact_attn.py` (standard K and V, compact K=V), for the FP8 and the BF16 cache.

Per cell (batch B, context L): random cache values in 64-token pages with a random block table (the same for every
kernel). Each kernel configuration is captured as CALLS calls in one CUDA graph and replayed; time per call = replay
time / CALLS (median and spread over REPLAYS). The caches are far larger than L2, so one buffer set is enough. The
stock kernel uses the arguments of vLLM's FlashAttentionImpl.forward at 3966924d5; the Triton kernels are timed at
every split and warp count of the sweep, and the analysis takes the best per cell. A correctness cell (B = 2,
L = 3,000, built as Gemma 4 K=V data) checks every kernel against the float32 reference of its own stored values,
and both layouts against the exact attention. A configuration that raises is recorded, not fatal.
--stub writes synthetic results (no GPU, no vLLM, no Triton) for the Mac tests.

Usage: kbench_e51.py --out <dir> [--cells B:L,...] [--stub ...]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HQ, HK, D, BS = 32, 4, 512, 64
CALLS, REPLAYS = 20, 15
CELLS = "4:2048,4:32768,4:65536,19:2048,19:8192,19:32768,11:2048,11:8192,11:32768,11:65536"
SPLITS, WARPS, TILES = (4, 8, 16, 32), (4, 8), (32, 64)   # the Triton sweep; a tile is the tokens per loop step
STOCK = {"stock_fp8_s32": ("fp8", 32), "stock_fp8_auto": ("fp8", 0), "stock_bf16_auto": ("bf16", 0),
         "stock_bf16_s32": ("bf16", 32)}
TRITON = {"triton_std_fp8": ("fp8", False), "triton_cmp_fp8": ("fp8", True), "triton_std_bf16": ("bf16", False),
          "triton_cmp_bf16": ("bf16", True)}
ROW = {False: 2 * D, True: D + 128}           # cache values per token and KV head (standard, compact)
# E-50 in-engine FA4 FP8 global attention per call (main plus combine kernel, us), from the group profiles.
TARGETS = {"19:2048": 238.3, "19:32768": 1358.1, "11:2048": 143.2, "11:65536": 1507.2}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def cell_list(text: str) -> list[tuple[int, int]]:
    return [(int(c.split(":")[0]), int(c.split(":")[1])) for c in text.split(",") if c]


def nbytes(b: int, length: int, compact: bool, fp8: bool) -> int:
    return b * length * HK * ROW[compact] * (1 if fp8 else 2)


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


def correctness(dev, fp8_dtype, errors: list) -> dict:
    """B = 2, L = 3,000 (a partial last page): Gemma 4 K=V data, every kernel against its references."""
    import torch
    import compact_attn as ca

    gen = torch.Generator().manual_seed(51)
    b, length = 2, 3000
    w = 1 + 0.1 * torch.randn(D, generator=gen, dtype=torch.float64)
    nblk = math.ceil(length / BS)
    perm = torch.randperm(b * nblk, generator=gen)
    bt = perm.view(b, nblk).to(torch.int32)
    q = torch.randn(b, HQ, D, generator=gen, dtype=torch.float64) / math.sqrt(D)
    q_rope = ca.rope(q, torch.full((b, HQ), float(length - 1), dtype=torch.float64))
    qc, qr = ca.compact_query(q_rope, w)
    kv = [ca.make_kv(length, HK, w, gen, torch.float64) for _ in range(b)]
    exact = torch.stack([ca.gqa_attend(q_rope[s], kv[s][0], kv[s][1]) for s in range(b)])
    k32 = torch.zeros(b * nblk, BS, HK, D)            # float32 pages; FP8 cannot be index-assigned on the CPU
    v32 = torch.zeros_like(k32)
    c32 = torch.zeros(b * nblk, BS, HK, D + 128)
    pad = nblk * BS - length
    for s in range(b):
        kp = torch.cat([kv[s][0], torch.zeros(pad, HK, D, dtype=torch.float64)]).view(nblk, BS, HK, D)
        vp = torch.cat([kv[s][1], torch.zeros(pad, HK, D, dtype=torch.float64)]).view(nblk, BS, HK, D)
        blocks = bt[s].long()
        k32[blocks], v32[blocks], c32[blocks] = kp.float(), vp.float(), ca.pack_compact(kp, vp).float()
    seqs = torch.full((b,), length, dtype=torch.int32)
    res = {}
    for name, dt in (("fp8", fp8_dtype), ("bf16", torch.bfloat16)):
        kc, vc, cc = k32.to(dt), v32.to(dt), c32.to(dt)
        # References from the stored (dequantized) values, as float32 copies (no FP8 gather on the CPU).
        ref_std = ca.paged_reference(q_rope.to(torch.bfloat16), kc.float(), vc.float(), bt, seqs)
        ref_cmp = ca.paged_reference(qc.to(torch.bfloat16), None, None, bt, seqs, qr=qr.to(torch.bfloat16),
                                     cache_c=cc.float())
        g = {k: t.to(dev) for k, t in (("kc", kc), ("vc", vc), ("cc", cc), ("bt", bt), ("seqs", seqs))}
        qd, qcd, qrd = (t.to(torch.bfloat16).to(dev) for t in (q_rope, qc, qr))
        cu = torch.arange(b + 1, dtype=torch.int32, device=dev)
        out = torch.empty(b, HQ, D, dtype=torch.bfloat16, device=dev)
        entry = {"exact_err_ref_standard": rel_err(ref_std, exact), "exact_err_ref_compact": rel_err(ref_cmp, exact)}
        for kname, (kdt, splits) in STOCK.items():
            if kdt != name:
                continue
            try:
                stock_call(qd, g["kc"], g["vc"], g["bt"], g["seqs"], cu, length, out, splits, name == "fp8")()
                torch.cuda.synchronize()
                entry[kname] = {"err": rel_err(out.cpu(), ref_std), "exact_err": rel_err(out.cpu(), exact)}
            except Exception as exc:  # recorded, not fatal
                entry[kname] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                errors.append(f"correctness {kname}: {traceback.format_exc()[-800:]}")
        for kname, (kdt, compact) in TRITON.items():
            if kdt != name:
                continue
            try:
                ws = ca.workspace(qd, 8)
                if compact:
                    ca.triton_attn(qcd, g["cc"][..., :D], g["bt"], g["seqs"], out, ws, splits=8, qr=qrd,
                                   cache_r=g["cc"][..., D:])
                    ref = ref_cmp
                else:
                    ca.triton_attn(qd, g["vc"], g["bt"], g["seqs"], out, ws, splits=8, cache_k=g["kc"])
                    ref = ref_std
                torch.cuda.synchronize()
                entry[kname] = {"err": rel_err(out.cpu(), ref), "exact_err": rel_err(out.cpu(), exact)}
            except Exception as exc:  # recorded, not fatal
                entry[kname] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
                errors.append(f"correctness {kname}: {traceback.format_exc()[-800:]}")
        res[name] = entry
    return res


def bench_cell(b: int, length: int, dev, fp8_dtype, errors: list) -> list[dict]:
    import torch
    import compact_attn as ca

    gen = torch.Generator(device=dev).manual_seed(b * 100_003 + length)
    nblk = math.ceil(length / BS)
    perm = torch.randperm(b * nblk, generator=gen, device=dev)
    bt = perm.view(b, nblk).to(torch.int32).contiguous()
    seqs = torch.full((b,), length, dtype=torch.int32, device=dev)
    cu = torch.arange(b + 1, dtype=torch.int32, device=dev)
    out = torch.empty(b, HQ, D, dtype=torch.bfloat16, device=dev)
    q = torch.randn(b, HQ, D, generator=gen, device=dev, dtype=torch.bfloat16) / math.sqrt(D)
    qr = q[..., :128].contiguous()
    rows = []
    for name, dt in (("fp8", fp8_dtype), ("bf16", torch.bfloat16)):
        kc = torch.empty(b * nblk, BS, HK, D, dtype=dt, device=dev)
        vc = torch.empty_like(kc)
        cc = torch.empty(b * nblk, BS, HK, D + 128, dtype=dt, device=dev)
        for t in (kc, vc, cc):
            fill_(t, gen)
        for kname, (kdt, splits) in STOCK.items():
            if kdt != name:
                continue
            row = {"kernel": kname, "batch": b, "context": length, "config": {"num_splits": splits},
                   "bytes": nbytes(b, length, False, name == "fp8")}
            try:
                us, spread = graph_time(stock_call(q, kc, vc, bt, seqs, cu, length, out, splits, name == "fp8"))
                row.update(us=round(us, 2), spread=round(spread, 4), status="ok")
            except Exception as exc:  # recorded, not fatal
                row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
                errors.append(f"{kname} {b}:{length}: {traceback.format_exc()[-800:]}")
            rows.append(row)
        for kname, (kdt, compact) in TRITON.items():
            if kdt != name:
                continue
            for splits in SPLITS:
                ws = ca.workspace(q, splits)
                for warps in WARPS:
                    for tile in TILES:
                        row = {"kernel": kname, "batch": b, "context": length,
                               "config": {"splits": splits, "warps": warps, "tile": tile},
                               "bytes": nbytes(b, length, compact, name == "fp8")}
                        if compact:
                            fn = lambda: ca.triton_attn(q, cc[..., :D], bt, seqs, out, ws, splits=splits, num_warps=warps,
                                                        tile=tile, qr=qr, cache_r=cc[..., D:])
                        else:
                            fn = lambda: ca.triton_attn(q, vc, bt, seqs, out, ws, splits=splits, num_warps=warps,
                                                        tile=tile, cache_k=kc)
                        try:
                            us, spread = graph_time(fn)
                            row.update(us=round(us, 2), spread=round(spread, 4), status="ok")
                        except Exception as exc:  # recorded, not fatal (for example out of shared memory)
                            row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:300])
                            errors.append(f"{kname} {row['config']} {b}:{length}: {traceback.format_exc()[-800:]}")
                        rows.append(row)
        del kc, vc, cc
        torch.cuda.empty_cache()
    for r in rows:
        if r.get("status") == "ok":
            r["gbps"] = round(r["bytes"] / (r["us"] * 1e-6) / 1e9, 1)
        print(json.dumps({k: r.get(k) for k in ("kernel", "batch", "context", "config", "us", "status")}), flush=True)
    return rows


def stub(args, out: Path) -> None:
    """Synthetic results: t = fixed + bytes / (efficiency x copy bandwidth), with noise; the stock FP8 kernel with 32
    splits takes the E-50 values at the four target cells (times --stub-engine-ratio)."""
    rng = random.Random(args.stub_seed)
    eff = {"stock_fp8": args.stub_stock_eff, "stock_bf16": args.stub_stock_bf16_eff,
           "triton_std": args.stub_std_eff, "triton_cmp": args.stub_cmp_eff}
    rows = []
    for b, length in cell_list(args.cells):
        for kname, (kdt, splits) in STOCK.items():
            key = f"{b}:{length}"
            t = args.stub_fixed_us + nbytes(b, length, False, kdt == "fp8") / (eff[f"stock_{kdt}"] * args.stub_bw * 1e9) * 1e6
            if kname == "stock_fp8_s32" and key in TARGETS:
                t = TARGETS[key] * args.stub_engine_ratio
            rows.append({"kernel": kname, "batch": b, "context": length, "config": {"num_splits": splits},
                         "bytes": nbytes(b, length, False, kdt == "fp8"), "status": "ok",
                         "us": round(t * (1 + rng.uniform(-0.003, 0.003)), 2), "spread": 0.004})
        for kname, (kdt, compact) in TRITON.items():
            fam = "triton_cmp" if compact else "triton_std"
            for splits in SPLITS:
                for warps in WARPS:
                    for tile in TILES:
                        penalty = (1.0 + 0.02 * abs(math.log2(splits / 8)) + (0.01 if warps == 4 else 0.0) +
                                   (0.01 if tile == 64 else 0.0))
                        t = (args.stub_fixed_us + nbytes(b, length, compact, kdt == "fp8") /
                             (eff[fam] * args.stub_bw * 1e9) * 1e6) * penalty
                        rows.append({"kernel": kname, "batch": b, "context": length,
                                     "config": {"splits": splits, "warps": warps, "tile": tile},
                                     "bytes": nbytes(b, length, compact, kdt == "fp8"), "status": "ok",
                                     "us": round(t * (1 + rng.uniform(-0.003, 0.003)), 2), "spread": 0.004})
    for r in rows:
        r["gbps"] = round(r["bytes"] / (r["us"] * 1e-6) / 1e9, 1)
    err = 0.5 if args.stub_bad_correct else 0.004
    corr = {n: {"exact_err_ref_standard": 0.03, "exact_err_ref_compact": 0.035,
                **{k: {"err": err if k == "triton_cmp_fp8" else 0.004, "exact_err": 0.03} for k in list(STOCK) + list(TRITON)
                   if k.endswith(n) or f"_{n}_" in k}} for n in ("fp8", "bf16")}
    (out / "kbench.json").write_text(json.dumps({"cells": rows, "correctness": corr}, indent=1))
    (out / "run.json").write_text(json.dumps({"status": "error" if args.stub_fail else "ok", "stub": True,
                                              "gpu": {"name": "STUB NVIDIA H100 80GB HBM3"},
                                              "args": {k: str(v) for k, v in vars(args).items()}}, indent=1))
    if args.stub_fail:
        raise SystemExit("stub failure (--stub-fail)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cells", default=CELLS)
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
    ap.add_argument("--stub-bad-correct", action="store_true")
    ap.add_argument("--stub-fail", action="store_true")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.stub:
        stub(args, out)
        print("stub ok")
        return

    import torch

    record = {"args": {k: str(v) for k, v in vars(args).items()}, "started_at": now(), "status": "started",
              "gpu": {"name": torch.cuda.get_device_name(0),
                      "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))},
              "torch": torch.__version__}
    for mod in ("triton", "vllm"):
        try:
            record[mod] = __import__(mod).__version__
        except Exception as exc:  # reported only
            record[mod] = f"unavailable: {type(exc).__name__}"
    errors: list[str] = []
    try:
        dev = torch.device("cuda")
        fp8 = torch.float8_e4m3fn
        result = {"cells": [], "correctness": None}
        if not args.skip_correctness:
            result["correctness"] = correctness(dev, fp8, errors)
            (out / "kbench.json").write_text(json.dumps(result, indent=1))
        for b, length in cell_list(args.cells):
            result["cells"].extend(bench_cell(b, length, dev, fp8, errors))
            (out / "kbench.json").write_text(json.dumps(result, indent=1))
        record.update(status="ok", finished_at=now())
    except Exception as exc:  # recorded, then re-raised so the caller sees a non-zero exit
        record.update(status="error", error=f"{type(exc).__name__}: {exc}"[:2000])
        (out / "run.json").write_text(json.dumps(record | {"errors": errors[-50:]}, indent=1))
        raise
    (out / "run.json").write_text(json.dumps(record | {"errors": errors[-50:]}, indent=1))
    print(json.dumps({"status": record["status"], "errors": len(errors)}))


if __name__ == "__main__":
    main()
