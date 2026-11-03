"""The author's checks of the R-76 post-run review (POST-HOC; the frozen E-54 decision does not change).

1. From `metrics.json`: the cells where TIED is worse than STOCK (R-76 C2), the cells where TIED beats CALSTD_JOINT,
   the cells above 1.5 against CALSTD_JOINT, C0/STOCK in layer 41 per prompt (C5), the position-0-exact ratios per
   layer (key finding 9), and the cells where the frozen phase beats both neighbour phases (TIED_UP and TIED_DN).
2. With the E-54 analysis code (`analyze_e54.tied_arm`, not the reviewer's library): the TIED error on prompt 0 at
   scale offsets of vs (ks = w vs) and with both scales rounded to BF16 (C4).

Usage: r76_checks.py [<reviewer sens.jsonl>]   (with the file: print the largest relative difference to it)
Output: r76_checks.json next to this file. The copied captures must be in results/e54-h100-20261102/capture.
"""
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
E = HERE.parent
sys.path.insert(0, str(E))
import analyze_e54 as a  # noqa: E402

RES = E / "results/e54-h100-20261102"
LAYERS = [5, 11, 17, 23, 29, 35, 41, 47, 53, 59]
DELTAS = [-0.01, -0.0025, 0.0, 0.0025, 0.01]


def cell_checks() -> dict:
    p = json.loads((RES / "e54/metrics.json").read_text())["prompts"]
    cells = [(k, str(n)) for k in p for n in LAYERS]
    err = {c: p[c[0]][c[1]]["err"] for c in cells}
    x0 = {c: p[c[0]][c[1]]["err_x0"] for c in cells}
    med = statistics.median
    return {
        "tied_worse_than_stock": [[k, n, err[k, n]["TIED"] / err[k, n]["STOCK"]] for k, n in cells
                                  if err[k, n]["TIED"] >= err[k, n]["STOCK"]],
        "tied_beats_joint": sum(err[c]["TIED"] < err[c]["CALSTD_JOINT"] for c in cells),
        "tied_over_joint_above_1.5": [[k, n, err[k, n]["TIED"] / err[k, n]["CALSTD_JOINT"]] for k, n in cells
                                      if err[k, n]["TIED"] > 1.5 * err[k, n]["CALSTD_JOINT"]],
        "c0_over_stock_layer41": {k: err[k, "41"]["C0"] / err[k, "41"]["STOCK"] for k in p},
        "x0_tied_over_stock": {n: med([x0[k, str(n)]["TIED"] for k in p]) / med([x0[k, str(n)]["STOCK"] for k in p])
                               for n in LAYERS},
        "frozen_phase_beats_both_neighbours": sum(err[c]["TIED"] < min(err[c]["TIED_UP"], err[c]["TIED_DN"])
                                                  for c in cells),
        "cells": len(cells),
    }


def offsets() -> dict:
    sc = json.loads((E / "scales_e54.json").read_text())["layers"]
    bf16 = lambda x: float(torch.tensor(x, dtype=torch.float32).to(torch.bfloat16).float())  # noqa: E731
    out = {}
    for n in LAYERS:
        cap = torch.load(RES / f"capture/p0/L{n:02d}.pt", map_location="cpu", weights_only=True)
        q, k, v = (cap[x].float() for x in ("q", "k", "v"))
        qpos = cap["qpos"].long()
        w, vs0, ks0 = sc[str(n)]["w"], sc[str(n)]["tied"]["vs"], sc[str(n)]["tied"]["ks"]
        ref = a.an.attend(q, k, v, qpos)
        row = {}
        for d in DELTAS:
            vs = a.f32(vs0 * (1 + d))
            row[str(d)] = a.an.rel(a.tied_arm(q, k, v, qpos, a.f32(w * vs), vs), ref)
        row["both_bf16"] = a.an.rel(a.tied_arm(q, k, v, qpos, bf16(ks0), bf16(vs0)), ref)
        out[str(n)] = row
        print(n, {x: round(y, 5) for x, y in row.items()}, flush=True)
    rms = lambda xs: (sum(x * x for x in xs) / len(xs)) ** 0.5  # noqa: E731
    base = rms([out[str(n)]["0.0"] for n in LAYERS])
    pooled = {x: rms([out[str(n)][x] for n in LAYERS]) / base - 1 for x in out[str(LAYERS[0])]}
    return {"prompt": 0, "layers": out, "pooled_change": pooled}


def main() -> None:
    torch.set_num_threads(6)
    res = {"cells": cell_checks(), "offsets": offsets()}
    if len(sys.argv) > 1:
        rev = {d["layer"]: d for d in map(json.loads, open(sys.argv[1])) if d["p"] == 0}
        res["largest_rel_diff_to_reviewer"] = max(
            abs(y / (rev[int(n)]["var"]["both_bf16"] if x == "both_bf16" else rev[int(n)]["delta"][x]) - 1)
            for n, row in res["offsets"]["layers"].items() for x, y in row.items())
    (HERE / "r76_checks.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res["cells"].items() if k != "x0_tied_over_stock"}),
          res["offsets"]["pooled_change"], res.get("largest_rel_diff_to_reviewer"), sep="\n")


if __name__ == "__main__":
    main()
