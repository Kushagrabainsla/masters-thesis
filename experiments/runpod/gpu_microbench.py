"""Short GPU health check for a rented Pod (D-56): BF16 GEMM throughput and device memory bandwidth.

Run it at the start and at the end of a session. A weak or throttled host shows as a low or falling
number. About 20 seconds. Reported only. Usage: python3 gpu_microbench.py --out <file.json>
"""

from __future__ import annotations

import argparse
import json
import statistics
import time


def timed(fn, repeats: int) -> list[float]:
    import torch

    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) / 1000.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=8192, help="GEMM size n x n x n")
    parser.add_argument("--mib", type=int, default=4096, help="copy size in MiB")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    result: dict = {"recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import torch

        result["gpu"] = torch.cuda.get_device_name(0)
        a = torch.randn(args.n, args.n, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(args.n, args.n, device="cuda", dtype=torch.bfloat16)
        gemm = timed(lambda: a @ b, args.repeats)
        result["bf16_gemm_tflops_median"] = round(2 * args.n**3 / statistics.median(gemm) / 1e12, 1)
        del a, b
        src = torch.empty(args.mib * 2**20, device="cuda", dtype=torch.uint8)
        dst = torch.empty_like(src)
        copy = timed(lambda: dst.copy_(src), args.repeats)
        # A device copy reads and writes each byte once.
        result["copy_gb_per_s_median"] = round(2 * src.numel() / statistics.median(copy) / 1e9, 1)
        result["status"] = "ok"
    except Exception as exc:  # reported only
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
