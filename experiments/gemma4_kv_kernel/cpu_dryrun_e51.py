"""CPU dry run of the E-51 GPU code paths (A1). It installs `tl_shim` (a CPU emulation of the Triton calls of
`compact_attn.py`), replaces the CUDA graph timer by one call, and runs on small cells:
  1. `kbench_e51.correctness` (every split and tile of the sweep, both layouts, FP8 and BF16) for a partial last
     page, empty splits and B = 1 to 3: each Triton configuration against the reference of its stored values;
  2. the engine-layout cell of `make_cell`: the standard, compact and oracle (K = V) steps against references;
  3. `bench_cell` and `confirm`, the timed paths.
vLLM is not installed here, so every stock row must come back with status "error". The result is one JSON line. It
checks the logic and the data layout, not the Triton compile, shared memory, registers or speed.

Run: uv run --no-project --with numpy --with torch python cpu_dryrun_e51.py
"""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import tl_shim  # noqa: E402

tl_shim.install()
import torch  # noqa: E402

torch.cuda.synchronize = lambda *a, **k: None
import compact_attn as ca  # noqa: E402
import kbench_e51 as kb  # noqa: E402

assert ca.triton is not None, "the shim must load before compact_attn"
kb.graph_time = lambda step: (step(), (1.0, 0.0))[1]
kb.WARPS = (8,)                                   # the warp count does not change the emulated result
kb.BF16_CONFIGS = ((16, 8, 32), (4, 8, 32))
ARGS = Namespace(stages=2, stages_bf16=0)
CPU = torch.device("cpu")


def floats(entry: dict) -> list:
    return [e for e in (entry.get("by_config") or {}).values()]


def main() -> None:
    out = {"cases": {}, "errors": 0}
    worst = 0.0
    for b, length in ((2, 3000), (1, 64), (2, 130), (3, 2049)):
        errors: list[str] = []
        res = kb.correctness(CPU, torch.float8_e4m3fn, errors, ARGS, b=b, length=length)
        case = {}
        for name, entry in res.items():
            for k, v in entry.items():
                if k.startswith("triton_"):
                    vals = floats(v)
                    case[f"{name} {k}"] = {"n": len(vals), "all_float": all(isinstance(e, float) for e in vals),
                                           "max": max((e for e in vals if isinstance(e, float)), default=None)}
                    worst = max([worst] + [e for e in vals if isinstance(e, float)])
                elif k.startswith("stock_"):
                    case[f"{name} {k}"] = "error" if "error" in v else "ran"
            case[f"{name} exact_err"] = [round(entry["exact_err_ref_standard"], 4),
                                         round(entry["exact_err_ref_compact"], 4)]
        out["cases"][f"{b}:{length}"] = case
    out["worst_kernel_err"] = worst

    gen = torch.Generator().manual_seed(7)
    c = kb.make_cell(2, 130, CPU, torch.float8_e4m3fn, gen)
    cfg = {"splits": 8, "warps": 8, "tile": 32}
    refs = {"triton_std_fp8": ca.paged_reference(c["q"], c["k"].float(), c["v"].float(), c["bt"], c["seqs"]),
            "triton_std_fp8_alias": ca.paged_reference(c["q"], c["v"].float(), c["v"].float(), c["bt"], c["seqs"]),
            "triton_cmp_fp8": ca.paged_reference(c["qc"], None, None, c["bt"], c["seqs"], qr=c["qr"],
                                                 cache_c=c["cc"].transpose(1, 2).float())}
    out["layout"] = {}
    for kname, ref in refs.items():
        info = {}
        kb.make_fn(kname, cfg, c, 2)(info)
        out["layout"][kname] = round(kb.rel_err(c["out"], ref), 5)
    out["info_keys"] = sorted(info)

    errors = []
    rows = kb.bench_cell(2, 128, CPU, torch.float8_e4m3fn, errors, ARGS)
    out["bench"] = {"triton_ok": sum(r["status"] == "ok" for r in rows if r["kernel"].startswith("triton_")),
                    "triton": sum(r["kernel"].startswith("triton_") for r in rows),
                    "stock_error": sum(r["status"] == "error" for r in rows if r["kernel"].startswith("stock_")),
                    "stock": sum(r["kernel"].startswith("stock_") for r in rows),
                    "has_regs_key": all("regs" in r for r in rows if r["kernel"].startswith("triton_"))}
    plan = [{"kernel": k, "batch": 2, "context": L, "config": cfg if k.startswith("triton") else {"num_splits": 32}}
            for L in (128, 192) for k in ("triton_std_fp8", "triton_cmp_fp8", "triton_std_fp8_alias", "stock_fp8_s32",
                                          "stock_fp8_s32_alias")]
    conf = kb.confirm(plan, CPU, torch.float8_e4m3fn, errors, ARGS)
    out["confirm"] = {e["kernel"] + f" {e['context']}": (len(e["repeats"]) if e["status"] == "ok" else "error")
                      for e in conf}
    out["errors"] = len(errors)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
