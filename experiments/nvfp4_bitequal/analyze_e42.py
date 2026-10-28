"""E-42 analysis (README Sections 4 and 5, Amendment A1): bit-equality of top-20 logprobs against R_1.

Usage: analyze_e42.py --res <dir> --out summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CANDS = ["flashinfer_cudnn", "flashinfer_cutlass", "cutlass", "flashinfer_trtllm"]
HYP = ["flashinfer_cudnn", "flashinfer_cutlass"]
TOPK = 20
V2_MIN = 0.01
LINEAR = re.compile(r"Using (\w+) for NVFP4 GEMM")


def kernel(run: Path) -> str | None:
    p = run / "log.txt"
    m = LINEAR.search(p.read_text(errors="replace")) if p.exists() else None
    return m.group(1) if m else None


def load(run: Path) -> dict[str, dict] | None:
    rj, oj = run / "run.json", run / "outputs.jsonl"
    if not rj.exists() or json.loads(rj.read_text()).get("status") != "ok" or not oj.exists():
        return None
    return {r["id"]: r for r in (json.loads(l) for l in oj.read_text().splitlines() if l.strip())}


def as_dict(pos: list) -> dict[int, float]:
    return {int(t): lp for t, lp in pos}  # A1 (R-51): order-free, so ties cannot change the compare


def compare(ref: dict, other: dict) -> dict:
    same_tokens = positions = equal = short = 0
    max_diff = 0.0
    for pid, a in ref.items():
        b = other.get(pid)
        if b is None:
            continue
        same_tokens += a["token_ids"] == b["token_ids"]
        for ta, tb, ka, kb in zip(a["token_ids"], b["token_ids"], a["topk"], b["topk"]):
            if len(ka) < TOPK or len(kb) < TOPK:  # A1: an empty list must not count as equal
                short += 1
                break
            positions += 1
            da, db = as_dict(ka), as_dict(kb)
            equal += da == db
            max_diff = max([max_diff] + [abs(da[t] - db[t]) for t in da.keys() & db.keys()])
            if ta != tb:
                break
    return {"prompts": len(ref), "same_tokens": same_tokens, "positions": positions, "short": short,
            "bit_equal_fraction": equal / positions if positions else 0.0, "max_abs_diff": max_diff}


def label(c: dict | None) -> str:
    if c is None:
        return "FAILED"
    if c["same_tokens"] != c["prompts"]:
        return "DIFFERENT"
    return "BIT_EQUAL" if c["bit_equal_fraction"] == 1.0 else "NEAR"


def pair(res: Path, ref: dict, name: str) -> dict:
    o = load(res / name)
    c = compare(ref, o) if o else None
    return {"label": label(c), "kernel": kernel(res / name), **(c or {})}


def summarize(res: Path) -> dict:
    r1 = load(res / "R_1")
    if r1 is None:
        return {"decision": "INVALID (R_1 missing)"}
    k1 = kernel(res / "R_1")
    out = {"R_kernel": k1, "controls": {n: pair(res, r1, n) for n in ("R_2", "marlin", "R_m")},
           "backends": {v: pair(res, r1, v) for v in CANDS}, "graph": {}}
    rg = load(res / "R_g")
    if rg:
        out["graph"]["R_g_vs_R_1"] = pair(res, r1, "R_g")  # reported only: compile fusions differ from eager
        out["graph"]["flashinfer_cutlass_g_vs_R_g"] = pair(res, rg, "flashinfer_cutlass_g")
    c = out["controls"]
    bad = []
    if not k1:
        bad.append("V0 no R kernel line")
    names = [b["kernel"] for b in list(out["backends"].values()) + [c["marlin"]] if b["label"] != "FAILED"]
    if any(n is None or n == k1 for n in names) or len(set(names)) != len(names):  # A1: alias check
        bad.append("V0 alias or missing kernel line")
    if c["R_2"]["label"] != "BIT_EQUAL":
        bad.append("V1")
    if c["marlin"]["label"] == "FAILED" or (1 - c["marlin"]["bit_equal_fraction"]) < V2_MIN:
        bad.append("V2")
    if any(b.get("short") for b in [c["R_2"]] + list(out["backends"].values())):
        bad.append("V0 short logprob lists")
    out["sensitivity"] = ("shown" if c["R_m"]["label"] in ("NEAR", "DIFFERENT")
                          else "not shown (R_m equals R_1)" if c["R_m"]["label"] == "BIT_EQUAL" else "missing")
    if bad:
        out["decision"] = "INVALID (" + "; ".join(bad) + ")"
        return out
    labels = ", ".join(f"{v} {b['label']}" for v, b in out["backends"].items())
    if any(out["backends"][v]["label"] == "FAILED" for v in HYP):
        out["decision"] = f"INCONCLUSIVE ({labels})"
    elif all(out["backends"][v]["label"] == "BIT_EQUAL" for v in HYP):
        out["decision"] = f"HYPOTHESIS HOLDS ({labels}; M sensitivity {out['sensitivity']})"
    else:
        out["decision"] = f"HYPOTHESIS FAILS ({labels})"
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--res", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    s = summarize(a.res)
    a.out.write_text(json.dumps(s, indent=1))
    print(s["decision"])


if __name__ == "__main__":
    main()
