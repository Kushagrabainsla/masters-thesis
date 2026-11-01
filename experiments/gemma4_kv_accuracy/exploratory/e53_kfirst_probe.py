"""EXPLORATORY (after the frozen E-53 decision, on the same captures; the layout was chosen after the FAIL): the
mirror layout [K (512) | V_rot (128)], also 640 values. Its K view is fp8(K) in another dim order, so its scores equal
the standard storage; its V view is [fp8(K_unrot) / w | fp8(V_rot)]. Per prompt and global layer: err_std, err_cmp
(as analyze_e53.py) and err_kfirst against the same BF16 reference, also with the FP16 emulation.
Usage: e53_kfirst_probe.py <repo> <capture dir> <out json>"""
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
rows = []
for pdir in sorted(cap_dir.glob("p*")):
    for f in sorted(pdir.glob("L*.pt")):
        cap = torch.load(f, map_location="cpu", weights_only=True)
        layer = int(cap["layer"])
        q, k, v = (cap[n].float() for n in ("q", "k", "v"))
        qpos = cap["qpos"].long()
        w = kn[layer].float()
        ref = an.attend(q, k, v, qpos)
        k8, v8 = an.fp8(k), an.fp8(v)
        e_std = an.rel(an.attend(q, k8, v8, qpos), ref)
        row8 = an.fp8(ca.pack_views(k, v))
        kview, vview = ca.views_of(row8)
        qv = an.bf16(ca.views_query(q, w))
        e_cmp = an.rel(ca.unpermute(an.attend(qv, kview, vview, qpos)), ref)
        v_hat = v8.clone()                                   # V_rot from fp8(V_rot); V_unrot from fp8(K_unrot) / w
        v_hat[..., UN] = k8[..., UN] / w[UN]
        o_kf = an.attend(q, k8, v_hat, qpos)                 # scores from fp8(K): the same as the standard storage
        e_kf = an.rel(o_kf, ref)
        e_kf16 = an.rel(an.attend(an.fp16(q), k8, v_hat, qpos, torch.float16), ref)
        e_std16 = an.rel(an.attend(an.fp16(q), k8, v8, qpos, torch.float16), ref)
        rows.append({"prompt": pdir.name, "layer": layer, "e_std": e_std, "e_cmp": e_cmp, "e_kfirst": e_kf,
                     "e_std_fp16": e_std16, "e_kfirst_fp16": e_kf16})
        print(json.dumps(rows[-1]), flush=True)
summary = {}
for layer in sorted({r["layer"] for r in rows}):
    rs = [r for r in rows if r["layer"] == layer]
    summary[layer] = {"med_ratio_cmp": st.median(r["e_cmp"] / r["e_std"] for r in rs),
                      "med_ratio_kfirst": st.median(r["e_kfirst"] / r["e_std"] for r in rs),
                      "max_ratio_kfirst": max(max(r["e_kfirst"] / r["e_std"], r["e_kfirst_fp16"] / r["e_std_fp16"])
                                              for r in rs),
                      "med_std": st.median(r["e_std"] for r in rs), "med_kfirst": st.median(r["e_kfirst"] for r in rs)}
out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
print("layer  ratio_cmp  ratio_kfirst  max_ratio_kfirst  med_std  med_kfirst")
for layer, s in summary.items():
    print(f"{layer:5d} {s['med_ratio_cmp']:10.3f} {s['med_ratio_kfirst']:13.3f} {s['max_ratio_kfirst']:17.3f} "
          f"{s['med_std']:8.4f} {s['med_kfirst']:11.4f}")
