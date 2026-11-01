"""CPU dry run of the E-52 GPU code paths. It installs `tl_shim` (the CPU emulation of the Triton calls of
`compact_attn.py`), replaces the CUDA graph timer by one call, makes the sweeps small, and runs:
  1. `kbench_e52.correctness` for a partial last page, empty splits and B = 1 to 3: every Triton kernel (standard,
     views and fused layouts; BF16 and FP16 compute) against the reference of its stored values, and each layout
     against the exact attention (the views outputs after `unpermute`);
  2. the engine-layout cell of `make_cell`: the standard, views, fused, K = V and pool steps against references;
  3. `bench_cell` and `confirm`, the timed paths.
vLLM is not installed here, so every stock row must come back with status "error". The result is one JSON line. It
checks the logic and the data layout, not the Triton compile, shared memory, registers or speed.

Run: uv run --no-project --with numpy --with torch python cpu_dryrun_e52.py
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))
sys.dont_write_bytecode = True

import tl_shim  # noqa: E402

tl_shim.install()
import torch  # noqa: E402

torch.cuda.synchronize = lambda *a, **k: None
import compact_attn as ca  # noqa: E402
import kbench_e52 as kb  # noqa: E402

assert ca.triton is not None, "the shim must load before compact_attn"
kb.graph_time = lambda step: (step(), (1.0, 0.0, {"clock_mhz": None, "clock_min": None, "cap_samples": 0,
                                                  "clock_samples": 0}))[1]
kb.SPLITS, kb.WARPS, kb.TILES, kb.STAGES = (8,), (4,), (16, 32), (2,)   # the warps and stages do not change the result
kb.LITE = ((8, 4, 32, 2),)
kb.BF16_CONFIGS = ((16, 4, 32, 2),)
CPU = torch.device("cpu")


def main() -> None:
    out = {"cases": {}, "errors": 0}
    worst = 0.0
    for b, length in ((2, 3000), (1, 64), (2, 130), (3, 2049)):
        errors: list[str] = []
        res = kb.correctness(CPU, torch.float8_e4m3fn, errors, Namespace(), b=b, length=length)
        case = {}
        for name, entry in res.items():
            for k, v in entry.items():
                if k.startswith("triton_"):
                    vals = list((v.get("by_config") or {}).values())
                    case[f"{name} {k}"] = {"n": len(vals), "all_float": all(isinstance(e, float) for e in vals),
                                           "max": max((e for e in vals if isinstance(e, float)), default=None),
                                           "exact_max": v.get("exact_err_max")}
                    worst = max([worst] + [e for e in vals if isinstance(e, float)])
                elif k.startswith("stock_"):
                    case[f"{name} {k}"] = "error" if "error" in v else "ran"
                elif k.startswith("exact_err_ref_"):
                    case[f"{name} {k}"] = round(v, 4)
        out["cases"][f"{b}:{length}"] = case
    out["worst_kernel_err"] = worst

    gen = torch.Generator().manual_seed(7)
    c = kb.make_cell(2, 130, CPU, torch.float8_e4m3fn, gen)
    cfg = {"splits": 8, "warps": 4, "tile": 32, "stages": 2}
    f = lambda t: t.float()
    pool_std, pool_cmp = c["bt_pool_std"], c["bt_pool_cmp"]
    refs = {"triton_std_fp8h": ca.paged_reference(c["q"], f(c["k"]), f(c["v"]), c["bt"], c["seqs"]),
            "triton_std_fp8": ca.paged_reference(c["q"], f(c["k"]), f(c["v"]), c["bt"], c["seqs"]),
            "triton_std_fp8h_alias": ca.paged_reference(c["q"], f(c["v"]), f(c["v"]), c["bt"], c["seqs"]),
            "triton_std_fp8h_pool": ca.paged_reference(c["q"], f(c["k"]), f(c["v"]), pool_std, c["seqs"]),
            "triton_cmpv_fp8h": ca.paged_reference(c["qv"], f(c["kview"]), f(c["vview"]), c["bt"], c["seqs"]),
            "triton_cmpv_fp8h_pool": ca.paged_reference(c["qv"], f(c["kview"]), f(c["vview"]), pool_cmp, c["seqs"]),
            "triton_cmp_fp8h": ca.paged_reference(c["qc"], None, None, c["bt"], c["seqs"], qr=c["qr"],
                                                  cache_c=c["cc"].transpose(1, 2).float())}
    out["layout"] = {}
    for kname, ref in refs.items():
        info = {"asm": True}
        kb.make_fn(kname, cfg, c)(info)
        out["layout"][kname] = round(kb.rel_err(c["out"], ref), 5)
    out["info_keys"] = sorted(info)
    out["pool_blocks"] = [int(pool_std.max()) + 1, int(pool_cmp.max()) + 1]

    errors = []
    rows = kb.bench_cell(2, 128, CPU, torch.float8_e4m3fn, errors, Namespace(asm=False, out=HERE))
    out["bench"] = {"triton_ok": sum(r["status"] == "ok" for r in rows if r["kernel"].startswith("triton_")),
                    "triton": sum(r["kernel"].startswith("triton_") for r in rows),
                    "stock_error": sum(r["status"] == "error" for r in rows if r["kernel"].startswith("stock_")),
                    "stock": sum(r["kernel"].startswith("stock_") for r in rows),
                    "has_regs_key": all("regs" in r for r in rows if r["kernel"].startswith("triton_"))}
    plan = [{"kernel": k, "batch": 2, "context": L, "config": cfg if k.startswith("triton") else {"num_splits": 32}}
            for L in (128, 192) for k in ("triton_std_fp8h", "triton_cmpv_fp8h", "triton_std_fp8h_alias",
                                          "triton_std_fp8h_pool", "triton_cmpv_fp8h_pool", "stock_fp8_s32",
                                          "stock_fp8_s32_views", "stock_fp8_s32_alias", "stock_fp8_s32_pool")]
    conf = kb.confirm(plan, CPU, torch.float8_e4m3fn, errors, Namespace())
    out["confirm"] = {e["kernel"] + f" {e['context']}": (len(e["repeats"]) if e["status"] == "ok" else "error")
                      for e in conf}
    out["errors"] = len(errors)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
