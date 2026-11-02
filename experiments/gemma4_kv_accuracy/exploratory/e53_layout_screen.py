"""EXPLORATORY screen (after the frozen E-53 FAIL, on the same captures; the layouts were chosen after the result):
which 640-value row keeps the FP8 accuracy of the standard 1,024-value cache, against a baseline with the same KV
scale? Per prompt and global layer, the attention-output error against the BF16 reference of:
  S0  standard, scale 1.0 (the E-53 baseline)    C0  [K_rot | V_unrot | V_rot], scale 1.0 (the E-53 compact row)
  M0  mirror [K | V_rot], scale 1.0               S4  standard, K stored as 16 K (a power-of-2 KV scale), V as V
  C4  C0 with K_rot stored as 16 K_rot            M4  mirror with K stored as 16 K; V_unrot = fp8(16 K_unrot) / (16 w)
  SA  standard with per-tensor amax scales        MA  mirror with the SA scales
The mirror row stores fp8 of K (512 values) and of V_rot (128): its scores equal the standard storage with the same K
scale, and only V_unrot is rounded on the K grid. Usage: e53_layout_screen.py <repo> <capture dir> <out json>"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

repo, cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(repo / "experiments/gemma4_kv_accuracy"), str(repo / "experiments/gemma4_kv_kernel")]
import analyze_e53 as an  # noqa: E402
import compact_attn as ca  # noqa: E402

UN = torch.tensor(an.UN)
NROT = len(an.ROT)
kn = torch.load(cap_dir / "k_norm.pt", map_location="cpu", weights_only=True)


def q8(x, s):
    """The FP8 cache cast with a per-tensor scale s (stored x / s, read back times s)."""
    return an.fp8(x / s) * s


rows = []
for pdir in sorted(cap_dir.glob("p*")):
    for f in sorted(pdir.glob("L*.pt")):
        cap = torch.load(f, map_location="cpu", weights_only=True)
        layer = int(cap["layer"])
        q, k, v = (cap[n].float() for n in ("q", "k", "v"))
        qpos = cap["qpos"].long()
        w = kn[layer].float()
        ref = an.attend(q, k, v, qpos)
        err = {}

        def mirror(k8, v8):
            v_hat = v8.clone()
            v_hat[..., UN] = k8[..., UN] / w[UN]
            return an.rel(an.attend(q, k8, v_hat, qpos), ref)

        def compact(rot_scale):
            row = ca.pack_views(k, v)
            s = torch.ones(row.shape[-1])
            s[:NROT] = 1.0 / rot_scale
            kview, vview = ca.views_of(q8(row, s))
            qv = an.bf16(ca.views_query(q, w))
            return an.rel(ca.unpermute(an.attend(qv, kview, vview, qpos)), ref)

        k0, v0 = q8(k, 1.0), q8(v, 1.0)
        k4 = q8(k, 1.0 / 16)
        ks, vs = float(k.abs().amax()) / 448, float(v.abs().amax()) / 448
        ka, va = q8(k, ks), q8(v, vs)
        err["S0"] = an.rel(an.attend(q, k0, v0, qpos), ref)
        err["C0"] = compact(1.0)
        err["M0"] = mirror(k0, v0)
        err["S4"] = an.rel(an.attend(q, k4, v0, qpos), ref)
        err["C4"] = compact(16.0)
        err["M4"] = mirror(k4, v0)
        err["SA"] = an.rel(an.attend(q, ka, va, qpos), ref)
        err["MA"] = mirror(ka, va)
        rows.append({"prompt": pdir.name, "layer": layer, "k_scale_amax": ks, "v_scale_amax": vs} | err)
        print(json.dumps(rows[-1]), flush=True)

PAIRS = [("C0", "S0"), ("M0", "S0"), ("S4", "S0"), ("C4", "S4"), ("M4", "S4"), ("M4", "S0"), ("SA", "S0"),
         ("MA", "SA")]
summary = {}
for layer in sorted({r["layer"] for r in rows}):
    rs = [r for r in rows if r["layer"] == layer]
    summary[layer] = {f"{a}/{b}": {"median": st.median(r[a] / r[b] for r in rs), "max": max(r[a] / r[b] for r in rs)}
                      for a, b in PAIRS}
out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
print("median per-prompt ratio (largest in brackets)")
print("layer " + " ".join(f"{a + '/' + b:>14}" for a, b in PAIRS))
for layer, s in summary.items():
    print(f"{layer:5d} " + " ".join(f"{s[a + '/' + b]['median']:6.3f} ({s[a + '/' + b]['max']:5.2f})" for a, b in PAIRS))
