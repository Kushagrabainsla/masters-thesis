"""E-44 kernel microbenchmark (README Section 3 and Amendment A1): fused against unfused block-FP8 norm+quant and
SiLU+quant at Qwen3-8B shapes (hidden 4096, intermediate 12288, group 128).

Timing (A1 item 3): buffers are allocated once; each variant is captured as CALLS calls in one CUDA graph over NSETS
rotating buffer sets, then replayed; time per call = replay time / CALLS. So Python and launch costs are not measured,
and large inputs are not served from L2. A launch floor (a graph of trivial kernels) is reported. Scales in the timed
runs use the model-path format (UE8M0 when DeepGEMM E8M0 is on, TMA-aligned column-major); the equality check uses
float32 scales in both paths. A variant that raises is recorded, not fatal.

Usage: kbench_e44.py --out <dir> [--tokens 1 8 64 256 512]   (writes <dir>/kbench.json and <dir>/run.json)
"""

from __future__ import annotations

import argparse
import json
import statistics
import traceback
from pathlib import Path

HIDDEN, INTER, GROUP, EPS = 4096, 12288, 128, 1e-6
CALLS, NSETS, REPLAYS = 48, 4, 20


def graph_time(step) -> float:
    """step(k) runs one call on buffer set k. Returns the median microseconds per call."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            for k in range(NSETS):
                step(k)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(CALLS):
            step(i % NSETS)
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPLAYS):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000.0 / CALLS)
    return statistics.median(times)


def setup():
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    try:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
        e8 = bool(is_deep_gemm_e8m0_used())
    except Exception:  # noqa: BLE001
        e8 = False

    @torch.compile(fullgraph=True)
    def add_rms_native(x, r, w):
        s = x + r
        v = s.float().pow(2).mean(-1, keepdim=True)
        return (s.float() * torch.rsqrt(v + EPS)).to(x.dtype) * w, s

    @torch.compile(fullgraph=True)
    def silu_mul_native(t):
        a, b = t.chunk(2, dim=-1)
        return torch.nn.functional.silu(a) * b

    return torch, ops, fp8_utils.per_token_group_quant_fp8, e8, add_rms_native, silu_mul_native


def bench_tokens(m: int, env) -> dict:
    torch, ops, quant, e8, add_rms_native, silu_mul_native = env
    dev, fp8 = "cuda", torch.float8_e4m3fn
    g = torch.Generator(device=dev).manual_seed(m)
    xs = [torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(NSETS)]
    rs = [torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(NSETS)]
    w = torch.rand(HIDDEN, device=dev, dtype=torch.bfloat16, generator=g) + 0.5
    gus = [torch.randn(m, 2 * INTER, device=dev, dtype=torch.bfloat16, generator=g) for _ in range(NSETS)]
    outs = [torch.empty(m, INTER, device=dev, dtype=torch.bfloat16) for _ in range(NSETS)]
    q = dict(column_major_scales=True, tma_aligned_scales=True, use_ue8m0=e8)

    norm = {
        "unfused_cuda": lambda k: (ops.fused_add_rms_norm(xs[k], rs[k], w, EPS), quant(xs[k], GROUP, **q)),
        "unfused_inductor": lambda k: quant(add_rms_native(xs[k], rs[k], w)[0], GROUP, **q),
        "fused": lambda k: ops.rms_norm_per_block_quant(xs[k], w, EPS, fp8, [1, GROUP], residual=rs[k],
                                                        is_scale_transposed=True, tma_alignment=4),
    }
    act = {
        "unfused_cuda": lambda k: (torch.ops._C.silu_and_mul(outs[k], gus[k]), quant(outs[k], GROUP, **q)),
        "unfused_inductor": lambda k: quant(silu_mul_native(gus[k]), GROUP, **q),
        "fused": lambda k: ops.silu_and_mul_per_block_quant(gus[k], GROUP, fp8, is_scale_transposed=True),
    }
    row = {}
    for op, table in (("norm", norm), ("act", act)):
        r = {}
        for label, fn in table.items():
            try:
                r[label] = {"us": round(graph_time(fn), 3)}
            except Exception as e:  # noqa: BLE001
                r[label] = {"error": f"{type(e).__name__}: {e}"[:300]}
        unf = [r[k]["us"] for k in ("unfused_cuda", "unfused_inductor") if "us" in r.get(k, {})]
        if unf and "us" in r.get("fused", {}):
            r["fused_over_unfused"] = round(r["fused"]["us"] / min(unf), 4)  # against the fastest unfused path
        row[op] = r
    row["equality"] = equality(m, env)
    return row


def equality(m: int, env) -> dict:
    """Fresh tensors, float32 scales in both paths, fused against the vLLM CUDA unfused path."""
    torch, ops, quant, _, _, _ = env
    out = {}
    try:
        g = torch.Generator(device="cuda").manual_seed(1000 + m)
        x = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
        r = torch.randn(m, HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g)
        w = torch.rand(HIDDEN, device="cuda", dtype=torch.bfloat16, generator=g) + 0.5
        xa, ra = x.clone(), r.clone()
        ops.fused_add_rms_norm(xa, ra, w, EPS)
        qa, sa = quant(xa, GROUP, column_major_scales=True, use_ue8m0=False)
        qb, sb = ops.rms_norm_per_block_quant(x.clone(), w, EPS, torch.float8_e4m3fn, [1, GROUP], residual=r.clone(),
                                              is_scale_transposed=True)
        out["norm"] = compare(qa, sa, qb, sb)
        gu = torch.randn(m, 2 * INTER, device="cuda", dtype=torch.bfloat16, generator=g)
        o = torch.empty(m, INTER, device="cuda", dtype=torch.bfloat16)
        torch.ops._C.silu_and_mul(o, gu)
        qa, sa = quant(o, GROUP, column_major_scales=True, use_ue8m0=False)
        qb, sb = ops.silu_and_mul_per_block_quant(gu, GROUP, torch.float8_e4m3fn, is_scale_transposed=True)
        out["act"] = compare(qa, sa, qb, sb)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    return out


def compare(qa, sa, qb, sb) -> dict:
    import torch

    same = (qa.view(torch.uint8) == qb.view(torch.uint8)).float().mean().item()
    rel = ((sa.float() - sb.float()).abs() / sa.float().abs().clamp_min(1e-12)).max().item()
    return {"fp8_bytes_equal": round(same, 6), "scale_max_rel_diff": rel}


def launch_floor() -> float:
    import torch

    t = torch.zeros(1, device="cuda")
    return round(graph_time(lambda k: t.add_(1)), 3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 256, 512])
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    env = setup()
    torch = env[0]
    result = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "shapes": [HIDDEN, INTER, GROUP],
              "calls_per_graph": CALLS, "buffer_sets": NSETS, "use_ue8m0": env[3], "by_tokens": {}}
    try:
        result["launch_floor_us"] = launch_floor()
    except Exception as e:  # noqa: BLE001
        result["launch_floor_us"] = f"error: {e}"[:200]
    for m in a.tokens:
        try:
            row = bench_tokens(m, env)
        except Exception:  # noqa: BLE001
            row = {"error": traceback.format_exc()[-600:]}
        result["by_tokens"][str(m)] = row
        print(m, json.dumps(row), flush=True)
    (out / "kbench.json").write_text(json.dumps(result, indent=1))
    (out / "run.json").write_text(json.dumps({"status": "ok"}))  # the session's resume check


if __name__ == "__main__":
    main()
