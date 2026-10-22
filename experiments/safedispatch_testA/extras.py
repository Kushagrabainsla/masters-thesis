"""E-36 amendment A2: exploratory summaries. Reported only; the Session 1 decision comes from compare.py.

1. GPU telemetry for each run window: median SM clock and power, peak temperature, the fraction of busy
   samples at the software power cap, and the fraction with a hard slowdown (thermal or power brake).
   A run is "suspect" if the hard fraction is above 5%, or if its median SM clock is more than 5% below
   the session median of all runs (R-39 item 10).
2. A kernel check (a weak second alias signal, R-39 item 6): the time shares of the GEMM and MoE
   kernels of each candidate against R_a at profile batch 64. Same set of GEMM and MoE kernel names
   means "same_gemm_kernels". The log-line rule of compare.py stays the alias rule.
3. Phase B: B0 (repeat of the most-flipping candidate), B1 flip counts (not the decision label,
   R-39 item 7), B2 decode speeds with the Phase A status of each backend and the KV cache size, B3
   eager-mode flips and speed against Phase A R_a.

Usage: extras.py --res <session dir> --out extras.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path

from compare import compare_pair, load_outputs, load_run, speeds_of

POWER_CAP_BIT = 0x4
HARD_BITS = 0x8 | 0x20 | 0x40 | 0x80  # hardware slowdown, software thermal, hardware thermal, power brake
HARD_LIMIT = 0.05
CLOCK_LIMIT = 0.05
GEMM = re.compile(r"gemm|cutlass|marlin|moe|deep_gemm|fp8|sm90|wgmma", re.IGNORECASE)
KV = re.compile(r"KV cache size: ([\d,]+) tokens")


def column(header: list[str], prefix: str) -> int | None:
    for i, name in enumerate(header):
        if name.strip().startswith(prefix):
            return i
    return None


def read_telemetry(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, newline="", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header or not header[0].strip().startswith("timestamp"):
                continue
            idx = {
                k: column(header, p)
                for k, p in (("power", "power.draw"), ("sm", "clocks.sm"), ("temp", "temperature.gpu"),
                             ("util", "utilization.gpu"), ("reasons", "clocks_"))
            }
            for rec in reader:
                if len(rec) != len(header):
                    continue
                try:
                    ts = datetime.strptime(rec[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
                    row = {"t": ts.timestamp()}
                    for k, i in idx.items():
                        if i is None:
                            continue
                        row[k] = int(rec[i].strip(), 16) if k == "reasons" else float(rec[i])
                    rows.append(row)
                except ValueError:
                    continue
    return sorted(rows, key=lambda r: r["t"])


def window(run: dict) -> tuple[float, float] | None:
    try:
        parse = lambda s: datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()  # noqa: E731
        return parse(run["started_at"]), parse(run["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None


def telemetry_for(run: dict | None, rows: list[dict]) -> dict | None:
    w = window(run or {})
    if not w:
        return None
    busy = [r for r in rows if w[0] <= r["t"] <= w[1] and r.get("util", 0) > 0]
    if not busy:
        return {"samples": 0}
    reasons = [r.get("reasons", 0) for r in busy]
    return {
        "samples": len(busy),
        "median_sm_mhz": statistics.median(r.get("sm", 0) for r in busy),
        "median_power_w": statistics.median(r.get("power", 0) for r in busy),
        "max_temp_c": max(r.get("temp", 0) for r in busy),
        "power_cap_fraction": sum(1 for x in reasons if x & POWER_CAP_BIT) / len(busy),
        "hard_slowdown_fraction": sum(1 for x in reasons if x & HARD_BITS) / len(busy),
    }


def gemm_shares(run_dir: Path, batch: str = "64") -> dict[str, float] | None:
    path = run_dir / "kernels.json"
    if not path.exists():
        return None
    table = json.loads(path.read_text()).get("by_batch", {}).get(batch)
    if not table:
        return None
    shares: dict[str, float] = {}
    for k in table["kernels"]:
        if GEMM.search(k["name"]):
            shares[k["name"]] = shares.get(k["name"], 0.0) + k.get("share", 0.0)
    return shares


def kv_tokens(run_dir: Path) -> int | None:
    path = run_dir / "log.txt"
    if not path.exists():
        return None
    for line in path.read_text(errors="replace").splitlines():
        m = KV.search(line)
        if m:
            return int(m.group(1).replace(",", ""))
    return None


def run_dirs(res: Path) -> list[Path]:
    return sorted(p.parent for p in res.rglob("run.json"))


def phase_a_status(summary: dict, cell: str) -> dict[str, str]:
    c = summary.get("cells", {}).get(cell, {})
    status = {"R_a": "reference", "R_b": "reference"}
    for name, v in c.get("candidates", {}).items():
        status[f"cand_{name}"] = "candidate" if v.get("status") == "ok" else "failed"
    for name in c.get("aliases_negative_control", {}):
        status[f"cand_{name}"] = "alias"
    for name in c.get("w8a16_positive_control", {}):
        status[f"cand_{name}"] = "w8a16_control"
    return status


def fastest_by_batch(cell: Path, status: dict[str, str]) -> dict:
    """Rank R and the Phase A candidates only; aliases, the control and failed runs are listed apart."""
    speeds = {d.name: speeds_of(load_run(d)) for d in sorted(cell.iterdir()) if d.is_dir()}
    out = {"kv_cache_tokens": kv_tokens(cell / "R_a"), "status": {n: status.get(n, "unknown") for n in speeds}}
    for bs in sorted({b for s in speeds.values() for b in s}, key=int):
        ranked = sorted(((n, s[bs]) for n, s in speeds.items() if bs in s), key=lambda x: -x[1])
        eligible = [r for r in ranked if status.get(r[0]) in ("reference", "candidate")]
        out[bs] = {"ranking": ranked, "fastest_eligible": eligible[0][0] if eligible else None}
    return out


def summarize(res: Path) -> dict:
    rows = read_telemetry(sorted(res.glob("env-*/telemetry.csv")))
    result: dict = {"telemetry_samples": len(rows), "runs": {}, "kernel_check": {}, "phase_b": {}, "microbench": {}}
    for d in run_dirs(res):
        run = load_run(d)
        result["runs"][str(d.relative_to(res))] = {
            "status": (run or {}).get("status"),
            "gpu_uuid": ((run or {}).get("gpu") or {}).get("uuid"),
            "telemetry": telemetry_for(run, rows),
        }
    clocks = [r["telemetry"]["median_sm_mhz"] for r in result["runs"].values()
              if r["telemetry"] and r["telemetry"].get("samples")]
    session_clock = statistics.median(clocks) if clocks else None
    for r in result["runs"].values():
        t = r["telemetry"]
        r["suspect_speed"] = bool(
            t and t.get("samples")
            and (t["hard_slowdown_fraction"] > HARD_LIMIT
                 or (session_clock and t["median_sm_mhz"] < (1 - CLOCK_LIMIT) * session_clock))
        )
    result["session_median_sm_mhz"] = session_clock
    uuids = {r["gpu_uuid"] for r in result["runs"].values() if r["gpu_uuid"]}
    result["gpu_uuids"] = sorted(uuids)
    result["one_gpu"] = len(uuids) <= 1
    for p in sorted(res.glob("env-*/microbench_*.json")):
        result["microbench"][f"{p.parent.name}/{p.stem}"] = json.loads(p.read_text())

    summary = json.loads((res / "summary.json").read_text()) if (res / "summary.json").exists() else {}
    x = res / "extra"
    for cell in ("cell_D", "cell_M"):
        ref = gemm_shares(res / cell / "R_a")
        checks = {}
        for d in sorted((res / cell).glob("cand_*")):
            cand = gemm_shares(d)
            if ref is None or cand is None:
                checks[d.name] = None
            else:
                checks[d.name] = {
                    "same_gemm_kernels": set(cand) == set(ref),
                    "gemm_share_candidate": round(sum(cand.values()), 4),
                    "gemm_share_R": round(sum(ref.values()), 4),
                    "only_in_candidate": sorted(set(cand) - set(ref))[:10],
                }
        result["kernel_check"][cell] = checks

        b0 = x / f"b0_{cell}"
        for d in sorted(b0.glob("cand_*")) if b0.is_dir() else []:
            first, again = load_outputs(res / cell / d.name), load_outputs(d)
            ref_a = load_outputs(res / cell / "R_a")
            result["phase_b"][f"b0_{cell}"] = {
                "candidate": d.name,
                "repeat_vs_first": compare_pair(first, again, set()) if first and again else None,
                "repeat_vs_R_a": compare_pair(ref_a, again, set()) if ref_a and again else None,
            }
        if (x / "summary_b1.json").exists():
            b1 = json.loads((x / "summary_b1.json").read_text()).get("cells", {}).get(f"b1_{cell}", {})
            result["phase_b"][f"b1_{cell}"] = {
                "reference_agreement": b1.get("reference_agreement"),
                "flips": {n: {k: v.get(k) for k in ("flipped", "n", "wilson95")}
                          for n, v in {**b1.get("candidates", {}), **b1.get("aliases_negative_control", {}),
                                       **b1.get("w8a16_positive_control", {})}.items()},
            }
        if (x / f"b2_{cell}").is_dir():
            result["phase_b"][f"b2_{cell}"] = fastest_by_batch(x / f"b2_{cell}", phase_a_status(summary, cell))
        b3 = x / f"b3_{cell}"
        if (b3 / "R_a").is_dir():
            ref_a, eager = load_outputs(res / cell / "R_a"), load_outputs(b3 / "R_a")
            graph_speed, eager_speed = speeds_of(load_run(res / cell / "R_a")), speeds_of(load_run(b3 / "R_a"))
            result["phase_b"][f"b3_{cell}"] = {
                "eager_vs_graph_flips": compare_pair(ref_a, eager, set()) if ref_a and eager else None,
                "eager_over_graph_speed": {
                    bs: eager_speed[bs] / graph_speed[bs] for bs in eager_speed if bs in graph_speed
                },
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = summarize(Path(args.res))
    Path(args.out).write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({"one_gpu": result["one_gpu"], "telemetry_samples": result["telemetry_samples"]}))


if __name__ == "__main__":
    main()
