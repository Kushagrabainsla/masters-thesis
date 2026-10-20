"""E-30 driver (amendment A1 after R-34): the KILL 1 and KILL 2 sweeps on one GH200.

Phase 1 (torch only): GEMM control at each CPU load and module limit.
Phase 2 (one vLLM engine, prefix caching off): prefill throughput at each CPU load and module limit,
with GPU power sampled during prefill; then, at the limit with the largest prefill loss, prefill and
decode throughput at each CPU cap under full CPU load.
CPU load is stress-ng on cores outside the engine's cores, in a seeded random order per repeat; N = 0
is measured again at the end of each limit to detect drift. Results are written after every stage.
Writes raw.jsonl and medians.json (the input of analyze.py). --dry-run prints the plan only.
"""

from __future__ import annotations

import os
import sys

ENGINE_CORES = set(range(8))
STRESS_FIRST_CORE = 8
MIN_CORES = 62

if "--dry-run" not in sys.argv:
    if (os.cpu_count() or 0) < MIN_CORES:
        raise SystemExit(f"need at least {MIN_CORES} cores, found {os.cpu_count()}")
    os.sched_setaffinity(0, ENGINE_CORES)  # before torch or vLLM start any thread

import argparse
import glob
import json
import random
import re
import signal
import statistics
import subprocess
import threading
import time
from pathlib import Path

LOADS = (0, 18, 36, 54)
CPU_CAPS_W = ("none", "150", "100", "70")
SEED = 20261020
RAMP_SECONDS = 5


def sh(cmd: list[str], dry: bool) -> str:
    if dry:
        print("DRY:", " ".join(cmd))
        return ""
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def default_module_limit(dry: bool) -> str:
    """Read 'Default Power Limit' from the 'Module Power Readings' section of nvidia-smi."""
    if dry:
        return "1000"
    section = sh(["nvidia-smi", "-q", "-d", "POWER"], dry).split("Module Power Readings", 1)[-1]
    m = re.search(r"Default Power Limit\s*:\s*([\d.]+)\s*W", section)
    if not m:
        raise SystemExit("no module default power limit found; run preflight.sh first")
    return str(int(float(m.group(1))))


def set_module_limit(watts: str, dry: bool) -> None:
    value = default_module_limit(dry) if watts == "default" else watts
    sh(["sudo", "nvidia-smi", "-pl", value, "-sc", "1"], dry)


def cpu_cap_files() -> list[str]:
    """NVIDIA's guide uses /sys/class/hwmon/hwmonX/device/power1_cap (microwatts)."""
    found = glob.glob("/sys/class/hwmon/hwmon*/power*_cap") + glob.glob("/sys/class/hwmon/hwmon*/device/power*_cap")
    return sorted(set(found))


def set_cpu_cap(cap: str, dry: bool) -> None:
    for f in cpu_cap_files() or ["/sys/class/hwmon/hwmonX/device/power1_cap"]:
        if cap == "none":
            max_file = Path(f.replace("_cap", "_cap_max"))
            value = max_file.read_text().strip() if not dry and max_file.exists() else "max"
        else:
            value = str(int(cap) * 1_000_000)  # microwatts
        sh(["sudo", "sh", "-c", f"echo {value} > {f}"], dry)


def start_stress(n: int, dry: bool):
    if n == 0:
        return None
    cores = f"{STRESS_FIRST_CORE}-{STRESS_FIRST_CORE + n - 1}"
    cmd = ["taskset", "-c", cores, "stress-ng", "--cpu", str(n), "--timeout", "900s", "--metrics-brief"]
    if dry:
        print("DRY:", " ".join(cmd))
        return None
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(RAMP_SECONDS)  # let CPU power ramp before the measurement
    return proc


def stop_stress(proc) -> float | None:
    """Stop stress-ng with SIGINT (it prints its metrics) and return bogo ops/s (real time)."""
    if proc is None:
        return None
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=60)
    m = re.search(r"cpu\s+(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+([\d.]+)", out or "")
    if not m:
        raise RuntimeError("could not parse stress-ng metrics:\n" + (out or "")[-800:])
    return float(m.group(2))


class PowerSampler:
    """Sample GPU power with nvidia-smi every 200 ms while active."""

    def __init__(self):
        self.samples: list[float] = []
        self._stop = threading.Event()

    def __enter__(self):
        def loop():
            while not self._stop.is_set():
                out = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True).stdout.strip().splitlines()
                if out:
                    try:
                        self.samples.append(float(out[0]))
                    except ValueError:
                        pass
                time.sleep(0.2)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    def median(self) -> float | None:
        return statistics.median(self.samples) if self.samples else None


def gpu_power_limit() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=power.limit", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0])


def gemm_tflops(seconds: float) -> float:
    import torch

    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    n, started = 0, time.time()
    while time.time() - started < seconds:
        a @ b
        n += 1
        if n % 50 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return 2 * 8192**3 * n / (time.time() - started) / 1e12


def prompts(tokenizer, n: int, target_tokens: int, seed: int) -> list[str]:
    """Distinct random-word prompts (prefix caching is off, and each prompt differs anyway)."""
    rng = random.Random(seed)
    words = ["system", "power", "budget", "kernel", "token", "memory", "graph", "queue", "cache", "agent"]
    out = []
    for _ in range(n):
        text = " ".join(rng.choice(words) for _ in range(target_tokens))
        out.append(tokenizer.decode(tokenizer(text)["input_ids"][:target_tokens]))
    return out


def orders(repeats: int, tag: str) -> list[list[int]]:
    rng = random.Random(f"{SEED}:{tag}")
    return [rng.sample(LOADS, len(LOADS)) for _ in range(repeats)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--limits", nargs="+", default=["default", "680"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--gemm-seconds", type=float, default=20.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dry = args.dry_run
    raw = open(out / "raw.jsonl", "a")
    medians: dict = {"limits": {}, "caps": {}, "validity": {}, "errors": []}

    def record(**kw):
        kw["t"] = time.time()
        raw.write(json.dumps(kw) + "\n")
        raw.flush()

    def save():
        (out / "medians.json").write_text(json.dumps(medians, indent=1))

    def sweep_loads(limit: str, kind: str, measure) -> dict:
        vals = {n: [] for n in LOADS}
        for order in orders(args.repeats, f"{kind}:{limit}"):
            for n in order:
                proc = start_stress(n, dry)
                v, extra = measure()
                ops = stop_stress(proc)
                record(phase=kind, limit=limit, load=n, value=v, stress_ops=ops, **extra)
                vals[n].append(v)
        proc = None
        end_v, _ = measure()  # N = 0 again at the end: drift check
        record(phase=kind, limit=limit, load=0, value=end_v, end_check=True)
        series = {str(n): statistics.median(v) for n, v in vals.items() if v}
        series["0_end"] = end_v
        return series

    # Phase 1: GEMM control.
    for limit in args.limits:
        try:
            set_module_limit(limit, dry)
            if dry:
                print(f"DRY: GEMM {args.gemm_seconds:.0f} s x {args.repeats} repeats per load, loads in random order")
                continue
            series = sweep_loads(limit, "gemm", lambda: (gemm_tflops(args.gemm_seconds), {}))
            medians["limits"].setdefault(limit, {})["gemm"] = series
        except Exception as exc:  # keep going with the other limits
            medians["errors"].append(f"gemm {limit}: {type(exc).__name__}: {exc}"[:500])
        save()
    set_module_limit("default", dry)

    if dry:
        print("DRY: phase 2: one vLLM engine pinned to cores 0-7, prefix caching off")
        for limit in args.limits:
            print(f"DRY: limit={limit}: prefill (64 x 4000 tokens, 1 new) at loads {list(LOADS)} in random order,"
                  f" GPU power sampled, N=0 again at the end")
        print("DRY: CPU caps", list(CPU_CAPS_W), "at the limit with the largest prefill loss, load 54,"
              " prefill and decode; cap files:", cpu_cap_files() or "none found on this machine")
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, max_num_seqs=64, max_model_len=8192, gpu_memory_utilization=0.85, seed=0,
              enable_prefix_caching=False)
    long_p, short_p = prompts(tok, 64, 4000, 1), prompts(tok, 64, 200, 2)
    pre = SamplingParams(temperature=0.0, max_tokens=1, seed=0)
    dec = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=True, seed=0)
    limit_w = gpu_power_limit()
    medians["validity"]["gpu_power_limit_w"] = limit_w

    def measure(kind: str):
        batch, params, tokens = (long_p, pre, 64 * 4000) if kind == "prefill" else (short_p, dec, 64 * 256)
        with PowerSampler() as power:
            started = time.time()
            llm.generate(batch, params, use_tqdm=False)
            seconds = time.time() - started
        return tokens / seconds, {"gpu_power_median_w": power.median()}

    measure("prefill")  # warm-up
    for limit in args.limits:
        try:
            set_module_limit(limit, dry)
            medians["limits"].setdefault(limit, {})["prefill"] = sweep_loads(limit, "prefill", lambda: measure("prefill"))
        except Exception as exc:
            medians["errors"].append(f"prefill {limit}: {type(exc).__name__}: {exc}"[:500])
        save()
    set_module_limit("default", dry)

    powers = [json.loads(l).get("gpu_power_median_w") for l in open(out / "raw.jsonl")
              if '"phase": "prefill"' in l and '"load": 0' in l]
    powers = [p for p in powers if p]
    medians["validity"]["prefill_power_median_w"] = statistics.median(powers) if powers else None

    losses = {k: 1 - v["prefill"]["54"] / v["prefill"]["0"] for k, v in medians["limits"].items() if "prefill" in v}
    cap_limit = max(losses, key=losses.get) if losses else "default"
    medians["validity"]["cap_sweep_limit"] = cap_limit
    if cpu_cap_files():
        try:
            set_module_limit(cap_limit, dry)
            for cap in CPU_CAPS_W:
                set_cpu_cap(cap, dry)
                point = {}
                for kind, tkey, ckey in (("prefill", "T_p", "C_p"), ("decode", "T_d", "C_d")):
                    tv, cv = [], []
                    for _ in range(args.repeats):
                        proc = start_stress(54, dry)
                        v, extra = measure(kind)
                        ops = stop_stress(proc)
                        record(phase=f"cap_{kind}", cap=cap, limit=cap_limit, load=54, value=v, stress_ops=ops, **extra)
                        tv.append(v)
                        cv.append(ops)
                    point[tkey], point[ckey] = statistics.median(tv), statistics.median(cv)
                medians["caps"][cap] = point
                save()
        except Exception as exc:
            medians["errors"].append(f"caps: {type(exc).__name__}: {exc}"[:500])
        finally:
            set_cpu_cap("none", dry)
            set_module_limit("default", dry)
    save()
    print("wrote", out / "medians.json", "| cap files:", cpu_cap_files(), "| errors:", len(medians["errors"]))


if __name__ == "__main__":
    main()
