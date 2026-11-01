"""E-52 compile check (A1, R-71 F2): compile every Triton kernel and configuration of the benchmark for sm_90 in its
own process, before the benchmark. Nothing launches: `_split_kernel.warmup` compiles only, with the target of the
active driver set to sm_90. On the RTX 4090 rehearsal this is a cross-compile; on the H100 it fills the Triton cache,
so a compiler crash cannot stop the benchmark.

Each entry is written as "compiling" before its compile and saved at once. If the process dies, a second run marks
that entry "crashed" and goes on with the rest; `kbench_e52.py --compile-check` then skips the crashed entries. Per
entry: the status, the seconds, the shared memory (Triton metadata), the PTX counts and SHA; the cubin goes to
<out>/cubin/ for `sass_counts_e52.py`.

Usage: compile_check_e52.py --out <results dir> [--target 90]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target", type=int, default=90)
    args = ap.parse_args()
    import torch
    import triton
    from triton.backends.compiler import GPUTarget

    import analyze_e52 as an
    import compact_attn as ca
    import kbench_e52 as kb

    path = args.out / "compile_check.json"
    res = json.loads(path.read_text()) if path.exists() else {"runs": 0, "kernels": {}}
    for e in res["kernels"].values():
        if e.get("status") == "compiling":         # the process died during this compile
            e["status"] = "crashed"
    res["runs"] += 1
    drv = triton.runtime.driver.active
    res["native"] = str(drv.get_current_target())
    target = GPUTarget("cuda", args.target, 32)
    drv.get_current_target = lambda: target         # warmup compiles for this target; nothing launches
    res["target"] = str(target)
    (args.out / "cubin").mkdir(parents=True, exist_ok=True)

    def save() -> None:
        path.write_text(json.dumps(res, indent=1))

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1)
    cells = {name: kb.make_cell(2, 2048, dev, dt, gen) for name, dt in (("fp8", torch.float8_e4m3fn),
                                                                         ("bf16", torch.bfloat16))}
    for kname, (kdt, layout, cast, kind) in kb.TRITON.items():
        c = cells[kdt]
        for cfg_t in kb.triton_configs(kind):
            cfg = kb.cfg_dict(cfg_t)
            name = f"{kname} {an.cfg_key(cfg)}"
            if name in res["kernels"]:
                continue
            res["kernels"][name] = {"status": "compiling"}
            save()
            t1 = time.time()
            try:
                ws = ca.workspace(c["q"], cfg["splits"])
                kw = dict(splits=cfg["splits"], num_warps=cfg["warps"], num_stages=cfg["stages"], tile=cfg["tile"],
                          cast=cast)
                if layout == "fused":
                    grid, a, k2 = ca.split_args(c["qc"], c["cv"], c["bt"], c["seqs"], ws, qr=c["qr"],
                                                cache_r=c["cr"], **kw)
                else:
                    q, kk, vv = (c["qv"], c["kview"], c["vview"]) if layout == "views" else (c["q"], c["k"], c["v"])
                    grid, a, k2 = ca.split_args(q, vv, c["bt"], c["seqs"], ws, cache_k=kk, **kw)
                kern = ca._split_kernel.warmup(*a, grid=grid, **k2)
                asm = getattr(kern, "asm", None) or {}
                md = getattr(kern, "metadata", None)
                ptx, cubin = asm.get("ptx") or "", asm.get("cubin")
                entry = {"status": "ok", "seconds": round(time.time() - t1, 2),
                         "shared": md.get("shared") if isinstance(md, dict) else getattr(md, "shared", None),
                         "ptx_sha": hashlib.sha256(ptx.encode()).hexdigest()[:16] if ptx else None,
                         "ptx_counts": kb.ptx_counts(ptx) if ptx else None}
                if isinstance(cubin, (bytes, bytearray)):
                    (args.out / "cubin" / f"{kname}_{an.cfg_key(cfg).replace('/', '-')}.cubin").write_bytes(cubin)
            except Exception as exc:  # recorded, not fatal (for example out of resources at compile time)
                entry = {"status": "error", "seconds": round(time.time() - t1, 2),
                         "error": f"{type(exc).__name__}: {exc}"[:300], "trace": traceback.format_exc()[-600:]}
            res["kernels"][name] = entry
            save()
    res["done"] = True
    save()
    print(json.dumps({k: sum(1 for e in res["kernels"].values() if e.get("status") == k)
                      for k in ("ok", "error", "crashed")}))


if __name__ == "__main__":
    main()
