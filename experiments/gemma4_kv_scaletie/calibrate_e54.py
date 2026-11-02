"""E-54 calibration (README Section 4.6; run once on the Mac before any E-54 data, then frozen): per global layer, the
KV scales of the scale-tied compact row (TIED) and of the calibrated standard FP8 cache (CALSTD), from the 4 E-53
captures. The E-54 test prompts take no file of these 4 prompts (V2), so the test is a holdout test.

Rule (frozen; project heuristics):
- Base scales: b_V and b_K are the smallest powers of 2 at or above 2 x max|V| / 448 and 2 x max|K| / 448, the
  maxima over the 4 calibration prompts. The largest calibration value then maps to at most 224: one octave or more
  of headroom before the E4M3 saturation at 448.
- Grid: 32 phases per octave, s_j = b x 2^(j / 32) for j = 0 to 31. A power of 2 does not move a normal value on the
  FP8 grid (R-74 C9), so the phase j is the only free part of a scale, and phase 0 is the phase of the stock scale
  1.0. So the CALSTD search contains the stock storage (without its K subnormals), and the TIED search contains the
  E-53 row at scale 1.0.
- Objective: E(j) = the sum over the 4 prompts of the relative L2 error of the attention output against the BF16
  reference (float32, causal GQA, as E-53). j* = the argmin of E; on a tie, the lowest j. No smoothing: the error
  curve is jagged (R-74: one 1/16-octave step can change it 2.4 times), but its minima transfer between prompts (R-74
  leave-one-prompt-out: layer 41 gives 1.11 to 1.13 on each left-out prompt, the oracle 1.11), because one sink
  vector and a few channels that all prompts share make the curve. A smoothed rule chose worse scales in the author's
  8-phase trial on layer 41 (README Section 4.6).
- TIED: vs = b_V x 2^(j*/32), where E uses the tied row with ks = w x vs; then ks = w x vs. One free scale.
- CALSTD: ks = b_K x 2^(jK/32) with jK from the score-path error (K cast, V exact), and vs = b_V x 2^(jV/32) with jV
  from the value-path error (K exact, V cast). Two free scales, each on its own path.
- CALSTD_JOINT (amendment A1, R-75 B1): the pair (jK, jV) of the argmin of E over all 32 x 32 pairs with both casts.
  A fast form ranks the pairs (one softmax per K phase); the exact path of the analysis decides among the best 8; on
  a tie, the lowest (jK, jV).
- Every scale is rounded to float32 (the engine holds the KV scales as float32 tensors).
Writes the scales, the chosen phases, the curves, the in-sample errors of every arm (information only, not a test),
w, and the SHA-256 of every input file. The tie needs one w value per layer: the script stops if a layer has more.

Usage: calibrate_e54.py --capture <E-53 capture dir> --run <E-53 capture_run.json> --out scales_e54.json
       [--device cpu] [--phases 32] [--layers 5,11]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_e54 as a54  # noqa: E402
from analyze_e54 import an, ca, f32, q8, std_arm, tied_arm  # noqa: E402

PHASES = 32
HEADROOM = 2.0                          # the largest calibration value maps to 448 / HEADROOM


def pick(e: list[float]) -> int:
    """The argmin of the curve; the lowest j on a tie."""
    return min(range(len(e)), key=lambda j: (e[j], j))


def base_scale(amax: float) -> float:
    """The smallest power of 2 at or above HEADROOM x amax / 448."""
    return 2.0 ** math.ceil(math.log2(HEADROOM * amax / an.FP8_MAX))


def phase_scale(base: float, j: int, phases: int = PHASES) -> float:
    return f32(base * 2.0 ** (j / phases))


def joint_curve(caps: list[tuple], refs: list[torch.Tensor], b_k: float, b_v: float,
                phases: int = PHASES) -> list[list[float]]:
    """E[jK][jV]: the summed relative error of the standard cache with both casts, for every pair of phases (A1, R-75
    B1). The same quantity as std_arm and an.rel: one softmax per K phase, and one product with the stacked V casts."""
    e = [[0.0] * phases for _ in range(phases)]
    for (q, k, v, qpos), ref in zip(caps, refs):
        nq, hq, length, hk = q.shape[0], q.shape[1], k.shape[0], k.shape[1]
        g, dv = hq // hk, v.shape[-1]
        future = torch.arange(length, device=q.device)[None, :] > qpos[:, None]
        vst = [torch.stack([q8(v[:, h], phase_scale(b_v, j, phases)) for j in range(phases)], dim=1)
               .reshape(length, phases * dv) for h in range(hk)]
        norm = float(torch.linalg.vector_norm(ref.double()))
        for jk in range(phases):
            kq = q8(k, phase_scale(b_k, jk, phases))
            err2 = torch.zeros(phases, dtype=torch.float64)
            for h in range(hk):
                s = torch.einsum("qgd,ld->qgl", q[:, h * g:(h + 1) * g], kq[:, h])
                p = torch.softmax(s.masked_fill(future[:, None, :], float("-inf")), dim=-1)
                o = (p.reshape(nq * g, length) @ vst[h]).view(nq, g, phases, dv)
                err2 += ((o - ref[:, h * g:(h + 1) * g, None, :]).double() ** 2).sum(dim=(0, 1, 3)).cpu()
            for jv in range(phases):
                e[jk][jv] += math.sqrt(float(err2[jv])) / norm
    return e


def layer_calibration(caps: list[tuple], w: float, phases: int = PHASES) -> dict:
    """The TIED and CALSTD scales of one layer from its calibration captures [(q, k, v, qpos), ...]."""
    amax_k = max(float(c[1].abs().max()) for c in caps)
    amax_v = max(float(c[2].abs().max()) for c in caps)
    b_k, b_v = base_scale(amax_k), base_scale(amax_v)
    refs = [an.attend(q, k, v, qpos) for q, k, v, qpos in caps]

    def curve(fn) -> list[float]:
        return [sum(an.rel(fn(c, j), r) for c, r in zip(caps, refs)) for j in range(phases)]

    e_t = curve(lambda c, j: tied_arm(c[0], c[1], c[2], c[3], f32(w * phase_scale(b_v, j, phases)),
                                      phase_scale(b_v, j, phases)))
    e_k = curve(lambda c, j: an.attend(c[0], q8(c[1], phase_scale(b_k, j, phases)), c[2], c[3]))
    e_v = curve(lambda c, j: an.attend(c[0], c[1], q8(c[2], phase_scale(b_v, j, phases)), c[3]))
    e_j = joint_curve(caps, refs, b_k, b_v, phases)
    jt, jk, jv = pick(e_t), pick(e_k), pick(e_v)
    recheck = [{"phase_k": a, "phase_v": b, "fast": x,
                "exact": sum(an.rel(std_arm(*c, phase_scale(b_k, a, phases), phase_scale(b_v, b, phases)), r)
                             for c, r in zip(caps, refs))}
               for x, a, b in sorted((e_j[a][b], a, b) for a in range(phases) for b in range(phases))[:8]]
    best = min(recheck, key=lambda r: (r["exact"], r["phase_k"], r["phase_v"]))
    vs_t = phase_scale(b_v, jt, phases)
    sc = {"w": w, "amax_k": amax_k, "amax_v": amax_v, "base_k": b_k, "base_v": b_v,
          "tied": {"phase": jt, "vs": vs_t, "ks": f32(w * vs_t)},
          "calstd": {"phase_k": jk, "phase_v": jv, "ks": phase_scale(b_k, jk, phases),
                     "vs": phase_scale(b_v, jv, phases)},
          "calstd_joint": {"phase_k": best["phase_k"], "phase_v": best["phase_v"],
                           "ks": phase_scale(b_k, best["phase_k"], phases),
                           "vs": phase_scale(b_v, best["phase_v"], phases), "recheck": recheck},
          "curves": {"tied": e_t, "calstd_score": e_k, "calstd_value": e_v, "calstd_joint": e_j}}
    ks_s, vs_s, ks_t = sc["calstd"]["ks"], sc["calstd"]["vs"], sc["tied"]["ks"]
    ks_j, vs_j = sc["calstd_joint"]["ks"], sc["calstd_joint"]["vs"]
    arms = {"STOCK": lambda c: std_arm(*c, 1.0, 1.0), "CALSTD": lambda c: std_arm(*c, ks_s, vs_s),
            "CALSTD_JOINT": lambda c: std_arm(*c, ks_j, vs_j),
            "TIED": lambda c: tied_arm(*c, ks_t, vs_t), "TIEDSTD": lambda c: std_arm(*c, ks_t, vs_t),
            "TIED_E5M2": lambda c: tied_arm(*c, ks_t, vs_t, cast=an.e5m2)}
    sc["in_sample_err"] = {a: [an.rel(fn(c), r) for c, r in zip(caps, refs)] for a, fn in arms.items()}
    return sc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, type=Path, help="the E-53 capture folder (p0..p3, k_norm.pt)")
    ap.add_argument("--run", required=True, type=Path, help="the E-53 capture_run.json (the SHA-256 of each file)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--phases", type=int, default=PHASES)
    ap.add_argument("--layers", default="", help="a subset of the layers (tests); default all")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    run = json.loads(args.run.read_text())
    want = {(str(e["id"]), str(layer)): f["sha256"] for e in run["prompts"] for layer, f in e["files"].items()}
    pids = sorted({p for p, _ in want}, key=int)
    kn_path = args.capture / "k_norm.pt"
    knorm = torch.load(kn_path, map_location="cpu", weights_only=True)
    layers = sorted(int(x) for x in knorm)
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    out = {"created_at": a54.now(), "rule": {"phases": args.phases, "headroom": HEADROOM, "base": "power of 2",
                                             "objective": "sum over the calibration prompts of the relative L2 error",
                                             "tie_break": "lowest phase",
                                             "calstd": "separable: K phase with V exact, V phase with K exact",
                                             "calstd_joint": "A1: argmin over all phase pairs with both casts; the "
                                                             "exact path decides among the best 8 of the fast form"},
           "inputs": {"capture_run": str(args.run), "capture_run_sha256": a54.sha_file(args.run),
                      "k_norm_sha256": a54.sha_file(kn_path), "prompts": pids, "files": {}},
           "torch": torch.__version__, "device": args.device, "layers": {}}
    t0 = time.time()
    for layer in layers:
        wm = an.bf16(knorm[layer].float())
        wu = wm[torch.tensor(ca.UNROT)]
        if int(torch.unique(wu).numel()) != 1:
            sys.exit(f"layer {layer}: the k_norm weight has {int(torch.unique(wu).numel())} values; the tie needs one")
        caps = []
        for p in pids:
            f = args.capture / f"p{p}" / f"L{layer:02d}.pt"
            b = f.read_bytes()
            got = hashlib.sha256(b).hexdigest()
            if got != want[(p, str(layer))]:
                sys.exit(f"{f}: SHA-256 {got} is not the E-53 record {want[(p, str(layer))]}")
            out["inputs"]["files"][f"p{p}/L{layer:02d}.pt"] = got
            c = torch.load(io.BytesIO(b), map_location=args.device, weights_only=True)
            caps.append((c["q"].float(), c["k"].float(), c["v"].float(), c["qpos"].long()))
        form, _, _ = an.w_choice(an.w_from_data(caps[0][1], caps[0][2]), knorm[layer])
        if form != "weight":
            sys.exit(f"layer {layer}: the data choose the module form {form!r}, not 'weight'")
        out["layers"][str(layer)] = layer_calibration(caps, float(wu[0]), args.phases)
        s = out["layers"][str(layer)]
        print(f"layer {layer}: tied phase {s['tied']['phase']} (vs {s['tied']['vs']:.6g}), calstd phases "
              f"{s['calstd']['phase_k']}/{s['calstd']['phase_v']}, joint {s['calstd_joint']['phase_k']}/"
              f"{s['calstd_joint']['phase_v']}, after {time.time() - t0:.0f} s", flush=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")
    # in-sample summary (information only): the same pooled statistics as the decision, on the calibration prompts
    rows = {layer: [{"err": {a: s["in_sample_err"][a][i] for a in s["in_sample_err"]}} for i in range(len(pids))]
            for layer, s in out["layers"].items()}
    P = {a: a54.pooled(rows, "err", a) for a in ("STOCK", "CALSTD", "CALSTD_JOINT", "TIED", "TIEDSTD", "TIED_E5M2")}
    out["in_sample"] = {"P": P, "R1": a54.ratio(P["TIED"], P["STOCK"]), "R2": a54.ratio(P["TIED"], P["CALSTD"]),
                        "R2_joint": a54.ratio(P["TIED"], P["CALSTD_JOINT"]),
                        "e5m2": a54.ratio(P["TIED_E5M2"], P["TIED"]), "tie": a54.ratio(P["TIED"], P["TIEDSTD"]),
                        "note": "in-sample (the calibration prompts): not a test; R2 is pessimistic (R-75 F1a)"}
    out["seconds"] = round(time.time() - t0, 1)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["in_sample"]), flush=True)


if __name__ == "__main__":
    main()
