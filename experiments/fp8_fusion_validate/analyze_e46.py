"""E-46 analysis (README Section 5): per-pair labels, the accuracy check, validity and the overall label.

Layout under the results root:
  versions.txt                           vLLM and torch versions on the Pod
  acc/on, acc/off                        run_a2.py --no-score accuracy runs (Qwen3-8B-FP8, deep_gemm)
  prof/on, prof/off                      run_cell.py throughput runs with --profile-kernels
  paired/P<n>/X_i (on), Y_i (off)        run_cell.py throughput runs; ratio = off / on
  env-*/telemetry.csv                    nvidia-smi samples

Usage: analyze_e46.py --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for sub in ("nvfp4_dispatch", "safedispatch_testA2", "ledger", "fp8_quant_fusion", "fp8_quant_cause"):
    sys.path.insert(0, str(HERE.parent / sub))

from analyze_a2 import MARGIN, load_jsonl, paired_interval  # noqa: E402
from analyze_e40 import run_of  # noqa: E402
from analyze_e44 import config_info, telemetry_ok  # noqa: E402
from analyze_e45 import paired  # noqa: E402  (bounds for 4 or 5 good rounds: df 3 or 4)

DECIDE = ["8", "64", "256"]
REPRO, ABSENT = 1.02, 1.01
MIN_ROUNDS_P1 = 4


def pair_label(p: dict) -> str:
    by = p["by_batch"]
    if any(by.get(bs, {}).get("lower", 0) >= REPRO for bs in DECIDE):
        return "REPRODUCED"
    if all(bs in by and by[bs]["upper"] < ABSENT for bs in DECIDE):
        return "ABSENT"
    return "UNCLEAR"


def accuracy(res: Path) -> dict:
    on, off = load_jsonl(res / "acc" / "on" / "completions.jsonl"), load_jsonl(res / "acc" / "off" / "completions.jsonl")
    if not on or not off or run_of(res / "acc" / "off").get("status") != "ok":
        return {"pass": None, "status": "missing"}
    diff, lo, hi = paired_interval(on, off)  # off minus on
    same = sum(on[k].get("token_ids") == off[k].get("token_ids") for k in on if k in off)
    return {"diff": diff, "interval95": [lo, hi], "pass": lo > -MARGIN, "same_tokens": same, "n": len(on), "status": "ok"}


def v0_profiles(res: Path) -> bool:
    cfg = {s: config_info(res / "prof" / s) for s in ("on", "off")}
    return (cfg["on"]["fused_kernels"] > 0 and cfg["off"]["fused_kernels"] == 0
            and cfg["on"]["selected"] is not None and cfg["on"]["selected"] == cfg["off"]["selected"])


def v0_pair(pdir: Path) -> bool | None:
    """A1 (R-59): the on run lists norm_quant in "Enabled custom fusions", the off run does not, same linear kernel."""
    x, y = config_info(pdir / "X_1"), config_info(pdir / "Y_1")
    if x["status"] is None and y["status"] is None:
        return None
    return ("norm_quant" in x["enabled_fusions"] and "norm_quant" not in y["enabled_fusions"]
            and x["selected"] is not None and x["selected"] == y["selected"])


def summarize(res: Path) -> dict:
    cfg = {s: config_info(res / "prof" / s) for s in ("on", "off", "default")}
    v0 = v0_profiles(res)
    pairs = {p: paired(res / "paired" / p) for p in ("P1", "P2", "P3")}
    for name, p in pairs.items():
        p["V0"] = v0_pair(res / "paired" / name)
        p["label"] = ("INVALID" if p["V0"] is False else pair_label(p)) if p["good"] >= 3 else "MISSING"
    ver = (res / "versions.txt").read_text().strip() if (res / "versions.txt").exists() else ""
    return {"versions": ver, "configs": cfg, "pairs": pairs, "accuracy": accuracy(res), "V0": v0, "V3": telemetry_ok(res)}


def decide(s: dict) -> str:
    p1 = s["pairs"]["P1"]
    bad = [k for k in ("V0", "V3") if not s[k]] + ([] if p1["good"] >= MIN_ROUNDS_P1 else ["V2 P1"])
    if bad:
        return f"INVALID ({', '.join(bad)})"
    labels = {k: v["label"] for k, v in s["pairs"].items()}
    tag = ", ".join(f"{k} {v}" for k, v in labels.items())
    acc = s["accuracy"].get("pass")
    acc_txt = {True: "accuracy pass", False: "accuracy FAIL", None: "accuracy missing"}[acc]
    if labels["P1"] == "ABSENT":
        return f"NOT REPRODUCED ({tag})"
    if labels["P1"] != "REPRODUCED":  # A1 (R-59): no bug report unless P1 reproduces
        return f"NOT CONFIRMED ({tag}; {acc_txt}; rerun or no report)"
    if "REPRODUCED" in (labels["P2"], labels["P3"]) and acc:
        return f"VALIDATED ({tag}; accuracy pass)"
    return f"PARTLY VALIDATED ({tag}; {acc_txt})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True, type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--gate", action="store_true", help="A1: check V0 on the profiles only; exit 1 if it fails")
    a = p.parse_args()
    if a.gate:
        ok = v0_profiles(a.res)
        print("V0 ok" if ok else "V0 FAILED")
        sys.exit(0 if ok else 1)
    s = summarize(a.res)
    s["decision"] = decide(s)
    a.out.write_text(json.dumps(s, indent=1, default=str))
    print(s["decision"])


if __name__ == "__main__":
    main()
