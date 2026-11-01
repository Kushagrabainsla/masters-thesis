"""EXPLORATORY (after the frozen E-53 decision): does the attention-sink token at position 0 drive the score-path
error ratio? For every prompt and global layer: the score-path errors of the two storages as in analyze_e53.py, the
same errors with position 0 kept in BF16 (not cast to FP8) in both storages, and the attention mass on position 0.
Reads the copied captures only. Usage: e53_sink_probe.py <repo> <capture dir> <out json>"""
import json
import statistics as st
import sys
from pathlib import Path

import torch

repo, cap_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path[:0] = [str(repo / "experiments/gemma4_kv_accuracy"), str(repo / "experiments/gemma4_kv_kernel")]
import analyze_e53 as an  # noqa: E402
import compact_attn as ca  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
kn = torch.load(cap_dir / "k_norm.pt", map_location="cpu", weights_only=True)
rows, k0 = [], {}
for pdir in sorted(cap_dir.glob("p*")):
    for f in sorted(pdir.glob("L*.pt")):
        cap = torch.load(f, map_location="cpu", weights_only=True)
        layer = int(cap["layer"])
        q, k, v = (cap[n].float() for n in ("q", "k", "v"))
        qpos = cap["qpos"].long()
        w = kn[layer].float()                                              # the module weight (form "weight")
        ref = an.attend(q, k, v, qpos)
        k8 = an.fp8(k)
        row = ca.pack_views(k, v)
        row8 = an.fp8(row)
        kview, _ = ca.views_of(row8)
        _, vview_b = ca.views_of(row)
        qv = an.bf16(ca.views_query(q, w))
        e_std = an.rel(an.attend(q, k8, v, qpos), ref)
        e_cmp = an.rel(ca.unpermute(an.attend(qv, kview, vview_b, qpos)), ref)
        k8x, row8x = k8.clone(), row8.clone()
        k8x[0], row8x[0] = k[0], row[0]                                     # position 0 kept in BF16
        kview_x, _ = ca.views_of(row8x)
        e_std_x = an.rel(an.attend(q, k8x, v, qpos), ref)
        e_cmp_x = an.rel(ca.unpermute(an.attend(qv, kview_x, vview_b, qpos)), ref)
        g = q.shape[1] // k.shape[1]
        mass0 = torch.stack([torch.softmax(torch.einsum("qgd,ld->qgl", q[:, h * g:(h + 1) * g], k[:, h]), -1)[..., 0]
                             for h in range(k.shape[1])]).mean().item()     # all keys are visible: qpos >= 7,936
        k0.setdefault(layer, []).append(k[0].clone())
        rows.append({"prompt": pdir.name, "layer": layer, "e_std": e_std, "e_cmp": e_cmp, "e_std_x0": e_std_x,
                     "e_cmp_x0": e_cmp_x, "mass0": mass0})
        print(json.dumps(rows[-1]), flush=True)
summary = {}
for layer in sorted({r["layer"] for r in rows}):
    rs = [r for r in rows if r["layer"] == layer]
    summary[layer] = {"ratio": st.median(r["e_cmp"] / r["e_std"] for r in rs),
                      "ratio_x0": st.median(r["e_cmp_x0"] / r["e_std_x0"] for r in rs),
                      "e_std": st.median(r["e_std"] for r in rs), "e_std_x0": st.median(r["e_std_x0"] for r in rs),
                      "e_cmp": st.median(r["e_cmp"] for r in rs), "e_cmp_x0": st.median(r["e_cmp_x0"] for r in rs),
                      "mass0": st.median(r["mass0"] for r in rs),
                      "k0_same_in_all_prompts": all(torch.equal(k0[layer][0], x) for x in k0[layer][1:])}
out.write_text(json.dumps({"rows": rows, "summary": summary}, indent=1))
print("layer  ratio  ratio_x0  e_std  e_std_x0  e_cmp  e_cmp_x0  mass0  k0_same")
for layer, s in summary.items():
    print(f"{layer:5d} {s['ratio']:6.3f} {s['ratio_x0']:8.3f} {s['e_std']:6.4f} {s['e_std_x0']:8.4f} {s['e_cmp']:6.4f} "
          f"{s['e_cmp_x0']:8.4f} {s['mass0']:6.3f} {s['k0_same_in_all_prompts']}")
