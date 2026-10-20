"""Try each backend value for one model in a fresh process; record which ones start.

Each attempt runs `run_cell.py --mode smoke` as a subprocess (so a crash or a stuck GPU context does
not affect the next attempt) with a time limit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

LINEAR = ["auto", "cutlass", "deep_gemm", "triton", "flashinfer_cutlass", "flashinfer_trtllm", "marlin", "torch"]
MOE = ["auto", "triton", "cutlass", "deep_gemm", "flashinfer_cutlass", "flashinfer_trtllm", "marlin", "triton_unfused"]
ATTENTION = ["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("linear", "moe", "attention"), required=True)
    parser.add_argument("--prompts-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--values", nargs="*", default=None, help="override the value list")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra run_cell.py args")
    args = parser.parse_args()

    values = args.values or {"linear": LINEAR, "moe": MOE, "attention": ATTENTION}[args.kind]
    flag = {"linear": "--linear-backend", "moe": "--moe-backend", "attention": "--attention-backend"}[args.kind]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in values:
        run_dir = out / f"smoke_{args.kind}_{value}"
        cmd = [sys.executable, str(HERE / "run_cell.py"), "--mode", "smoke", "--model", args.model,
               "--prompts-dir", args.prompts_dir, "--out", str(run_dir), flag, value, *args.extra]
        started = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
            code, tail = proc.returncode, (proc.stderr or "")[-1500:]
        except subprocess.TimeoutExpired as exc:
            code, tail = "timeout", str(exc)[-500:]
        rows.append({"value": value, "exit": code, "seconds": round(time.time() - started, 1),
                     "starts": code == 0, "stderr_tail": tail if code != 0 else ""})
        print(json.dumps({k: rows[-1][k] for k in ("value", "exit", "seconds")}), flush=True)
    (out / f"inventory_{args.kind}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
