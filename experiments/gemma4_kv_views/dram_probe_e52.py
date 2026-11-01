"""E-52 DRAM bytes probe (A1, R-71 F5; diagnostic only): one call each of the standard, K = V and views kernels (FP16
compute, 32/4/32/2) and of the stock kernel on both layouts at 19:32K, for
`ncu --metrics dram__bytes_read.sum`. If the views share their reads, the views kernel reads about 640 bytes per
token and KV head, the standard kernel about 1,024 and the K = V row about 512. Without counter access, ncu prints
ERR_NVGPUCTRPERM; the session records that and goes on.

Usage: ncu --metrics dram__bytes_read.sum python3 dram_probe_e52.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))


def main() -> None:
    import torch

    import kbench_e52 as kb

    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(19 * 100_003 + 32768)
    c = kb.make_cell(19, 32768, dev, torch.float8_e4m3fn, gen)
    tokens = 19 * 32768 * kb.HK
    print(f"token-heads {tokens}; expected DRAM bytes: std {tokens * 1024}, alias {tokens * 512}, "
          f"views {tokens * 640} (plus the partials)", flush=True)
    cfg = {"splits": 32, "warps": 4, "tile": 32, "stages": 2}
    for kname in ("triton_std_fp8h", "triton_std_fp8h_alias", "triton_cmpv_fp8h", "stock_fp8_s32",
                  "stock_fp8_s32_views"):
        try:
            fn = kb.make_fn(kname, cfg if kname.startswith("triton") else {"num_splits": 32}, c)
            fn()
            torch.cuda.synchronize()
            print(f"ran {kname}", flush=True)
        except Exception:  # reported only
            print(f"failed {kname}: {traceback.format_exc()[-400:]}", flush=True)


if __name__ == "__main__":
    main()
