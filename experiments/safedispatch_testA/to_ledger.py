"""Write one run manifest for each ok E-36 run (docs/experiments/PROTOCOL.md, stage S7).

Metrics per run: decode tokens/s at each batch size (one value per repeat), prefill seconds (one value
per repeat) and engine start seconds. Flip counts stay in summary.json: they compare two runs, so
they are not a per-run metric.

Usage: to_ledger.py --res <session dir> --out <manifest dir> --design-commit <git sha>
         [--pod <pod.json>] [--label <text>]
Then: python3 experiments/ledger/ledger.py add <manifest dir>/*.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def first_gpu_row(res: Path) -> dict:
    for path in sorted(res.glob("env-*/hardware/gpu.csv")):
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if len(rows) >= 2:
            return {h.strip(): v.strip() for h, v in zip(rows[0], rows[1])}
    return {}


def cuda_version(res: Path) -> str:
    for path in sorted(res.glob("env-*/versions.txt")):
        parts = path.read_text().split()
        if len(parts) >= 3:
            return parts[2]
    return "unknown"


def metrics_of(run: dict) -> dict:
    out = {}
    tp = run.get("throughput") or {}
    n_tok = run["args"].get("throughput_max_tokens", 128)
    for bs, v in (tp.get("decode") or {}).items():
        out[f"decode_tps_bs{bs}"] = {
            "values": [int(bs) * n_tok / s for s in v["seconds"] if s > 0],
            "unit": "tokens/s", "higher_is_better": True,
        }
    if tp.get("prefill_seconds"):
        out["prefill_s"] = {"values": tp["prefill_seconds"], "unit": "s", "higher_is_better": False}
    if run.get("init_seconds") is not None:
        out["engine_start_s"] = {"values": [run["init_seconds"]], "unit": "s", "higher_is_better": False}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--res", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--design-commit", required=True)
    parser.add_argument("--pod", type=Path, help="the Runpod Pod record saved at booking")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    pod = json.loads(args.pod.read_text()) if args.pod and args.pod.exists() else {}
    gpu = first_gpu_row(args.res)
    session = args.res.name
    written = 0
    for rj in sorted(args.res.rglob("run.json")):
        run = json.loads(rj.read_text())
        if run.get("status") != "ok":
            continue
        rel = rj.parent.relative_to(args.res)
        if not isinstance(run.get("args"), dict):
            print(f"skip {rel}: run.json has no argument record")
            continue
        a = run["args"]
        manifest = {
            "run_id": f"E-36/{pod.get('id', 'nopod')}/{session}/{rel}".replace(" ", "_"),
            "experiment": "E-36", "idea": "I-23", "status": "ok",
            "design_commit": args.design_commit,
            "config": {k: a.get(k) for k in ("model", "linear_backend", "moe_backend", "attention_backend",
                                              "max_num_seqs", "max_model_len", "enforce_eager", "mode",
                                              "batch_sizes", "repeats", "throughput_max_tokens", "max_tokens")},
            "started_at": run.get("started_at"), "finished_at": run.get("finished_at"),
            "hardware": {
                "gpu_name": (run.get("gpu") or {}).get("name", gpu.get("name", "unknown")),
                "gpu_uuid": (run.get("gpu") or {}).get("uuid"),
                "gpu_count": 1,
                "driver": gpu.get("driver_version", "unknown"),
                "cuda": cuda_version(args.res),
                "pod_id": pod.get("id"), "data_center": pod.get("dataCenterId"),
                "price_per_hour": pod.get("cost") or pod.get("costPerHr"),
            },
            "software": {"vllm": run.get("vllm_version"), "image": "vllm/vllm-openai:v0.32.0"},
            "metrics": metrics_of(run),
            "results_path": str(rj.parent),
            "label": args.label or str(rel),
        }
        (args.out / f"{str(rel).replace('/', '__')}.json").write_text(json.dumps(manifest, indent=1))
        written += 1
    print(f"wrote {written} manifest(s) to {args.out}")


if __name__ == "__main__":
    main()
