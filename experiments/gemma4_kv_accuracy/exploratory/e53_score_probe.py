"""EXPLORATORY (after the frozen E-53 decision): where does the extra score error of the compact storage come from?
Per prompt and global layer: (1) the compact path without FP8 (both views in BF16) against the BF16 reference, which
isolates the BF16 mismatch of K and w * V; (2) the score errors before the softmax, weighted by the reference
probabilities, split into the rotated and the unrotated dims; (3) the largest raw |q| on each part.
Reads the copied captures only. Usage: e53_score_probe.py <repo> <capture dir> <out json>"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

repo, cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(repo / "experiments/gemma4_kv_accuracy"), str(repo / "experiments/gemma4_kv_kernel")]
import analyze_e53 as an  # noqa: E402
import compact_attn as ca  # noqa: E402

ROT, UN = torch.tensor(an.ROT), torch.tensor(an.UN)
kn = torch.load(cap_dir / "k_norm.pt", map_location="cpu", weights_only=True)


def wrms(ds, p):
    """sqrt(sum p * ds^2 / sum p) over every query, head and key: the probability-weighted RMS of a score error."""
    return float(((p * ds.pow(2)).sum() / p.sum()).sqrt())


rows = []
for pdir in sorted(cap_dir.glob("p*")):
    for f in sorted(pdir.glob("L*.pt")):
        cap = torch.load(f, map_location="cpu", weights_only=True)
        layer = int(cap["layer"])
        q, k, v = (cap[n].float() for n in ("q", "k", "v"))
        qpos = cap["qpos"].long()
        w = kn[layer].float()
        ref = an.attend(q, k, v, qpos)
        row = ca.pack_views(k, v)
        kview_b, vview_b = ca.views_of(row)
        kview8, _ = ca.views_of(an.fp8(row))
        qv = an.bf16(ca.views_query(q, w))
        e_cmp_bf16 = an.rel(ca.unpermute(an.attend(qv, kview_b, vview_b, qpos)), ref)
        k8 = an.fp8(k)
        g = q.shape[1] // k.shape[1]
        acc = {n: [] for n in ("std_rot", "std_un", "cmp_rot", "cmp_un", "cmpb_un", "s_rms")}
        for h in range(k.shape[1]):
            qh, qvh = q[:, h * g:(h + 1) * g], qv[:, h * g:(h + 1) * g]
            s_ref = torch.einsum("qgd,ld->qgl", qh, k[:, h])
            p = torch.softmax(s_ref, -1)
            part = lambda a, b, idx: torch.einsum("qgd,ld->qgl", a[..., idx], b[:, h][..., idx])
            nrot = len(an.ROT)
            acc["std_rot"].append(wrms(part(qh, k8 - k, ROT), p))
            acc["std_un"].append(wrms(part(qh, k8 - k, UN), p))
            # the compact K view is [K_rot | V_unrot]: its first nrot values pair with q_rot, the rest with (q*w)_unrot
            s_cmp_rot = torch.einsum("qgd,ld->qgl", qvh[..., :nrot], kview8[:, h, :nrot])
            s_cmp_un = torch.einsum("qgd,ld->qgl", qvh[..., nrot:], kview8[:, h, nrot:])
            s_cmpb_un = torch.einsum("qgd,ld->qgl", qvh[..., nrot:], kview_b[:, h, nrot:])
            acc["cmp_rot"].append(wrms(s_cmp_rot - part(qh, k, ROT), p))
            acc["cmp_un"].append(wrms(s_cmp_un - part(qh, k, UN), p))
            acc["cmpb_un"].append(wrms(s_cmpb_un - part(qh, k, UN), p))
            acc["s_rms"].append(wrms(s_ref, p))
        r = {"prompt": pdir.name, "layer": layer, "e_cmp_bf16": e_cmp_bf16,
             "q_rot_max": float(q[..., ROT].abs().max()), "q_un_max": float(q[..., UN].abs().max()),
             "q_un_rms": float(q[..., UN].pow(2).mean().sqrt()), "q_rot_rms": float(q[..., ROT].pow(2).mean().sqrt())}
        r |= {n: float(torch.tensor(x).pow(2).mean().sqrt()) for n, x in acc.items()}
        rows.append(r)
        print(json.dumps(r), flush=True)
summary = {}
for layer in sorted({r["layer"] for r in rows}):
    rs = [r for r in rows if r["layer"] == layer]
    summary[layer] = {k: st.median(r[k] for r in rs) for k in rs[0] if k not in ("prompt", "layer")}
out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
keys = ["e_cmp_bf16", "std_rot", "cmp_rot", "std_un", "cmp_un", "cmpb_un", "s_rms", "q_rot_max", "q_un_max", "q_un_rms"]
print("layer " + " ".join(f"{k:>10}" for k in keys))
for layer, s in summary.items():
    print(f"{layer:5d} " + " ".join(f"{s[k]:10.4f}" for k in keys))
