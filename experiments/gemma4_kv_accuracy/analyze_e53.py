"""E-53 analysis (README Sections 5 to 7; gate G3 of D-88): on real Gemma 4 31B activations, does the compact FP8 KV
storage of the global layers (the E-52 views row [K_rot | V_unrot | V_rot] in FP8 E4M3 with scale 1.0, read with the
views query [q_rot | (q * w)_unrot] rounded to BF16) give an attention-output error no larger than the standard FP8 KV
storage (K and V each in E4M3 with scale 1.0), and do the views-path values fit FP16 with margin?

The script has two steps:
1. Metrics (on the Pod GPU, with `--capture <dir>`). For each prompt and global layer, it reads the BF16 q, k and v that
   capture_e53.py recorded. It computes in float32 the causal attention output of the recorded queries over all earlier
   tokens (8 query heads per KV head, scale 1.0, no soft-cap) for (a) the reference (BF16 K and V), (b) the standard FP8
   storage and (c) the compact views storage (output after `unpermute`). It writes the small file metrics.json.
2. Decision (on the Pod or on the Mac, from metrics.json): the validity checks V1 to V4 and the frozen rule.

The FP8 cast is the cast of the vLLM cache write (FACT, vLLM 3966924d5, README Section 4): BF16 to float, divide by the
scale (1.0), then `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)`. That is round to nearest even with
saturation at +-448. This script clamps to +-448 and then casts with `to(torch.float8_e4m3fn)`.

w is the k_norm weight on the 384 unrotated dims (K = w * V there). The script recovers w from the data (the median of
K_unrot / V_unrot per dim) and reads it from the module. A Gemma RMSNorm can scale by (1 + weight), so the data decide
which module form the views query uses. All thresholds are project heuristics (README Section 7).

Usage: analyze_e53.py --res <dir> [--capture <dir>] [--device auto|cpu|cuda] [--out summary.json]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import compact_attn as ca  # noqa: E402  (rope, ROT, UNROT, pack_views, views_of, views_query, unpermute)

HQ, HK, D = 32, 4, 512                # the Gemma 4 31B global layers (config; G1, G2)
GQA = HQ // HK                        # 8 query heads per KV head
PLAN = {"prompts": 4, "tokens": 8192, "q_rows": 256}   # the frozen plan (README Section 4)
FP8_MAX = 448.0                       # E4M3 saturation (__NV_SATFINITE)
FP8_TINY = 2.0 ** -6                  # the smallest normal E4M3 value
FP16_MAX = 65504.0
ID_TOL = 1e-2                         # V3: K_unrot against w * V_unrot (the BF16 level)
MED_PASS = 1.25                       # PASS: median err_compact <= 1.25 x median err_standard, in every layer
MAX_PASS = 2.0                        # PASS: largest err_compact <= 2 x largest err_standard, in every layer
MED_FAIL = 2.0                        # FAIL: a layer with median err_compact > 2 x median err_standard
Q_PASS = 6550.0                       # PASS: largest |q'| at most 6,550 (a 10 x FP16 margin)
Q_FAIL = FP16_MAX / 2                 # FAIL: largest |q'| above 32,752
SAT_PP = 0.001                        # PASS: compact saturated share <= standard share + 0.1 percentage points
V_FLOOR = 0.05                        # the |V| floor for the ratio K / V (V has RMS 1)
W_MATCH = 0.02                        # reported: a module form of w agrees with the data
ROPE_OK = 0.03                        # reported: the RoPE fit residual of a capture after RoPE
UN = ca.UNROT
ROT = ca.ROT


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def fp8(x: torch.Tensor) -> torch.Tensor:
    """The vLLM FP8 cache cast with scale 1.0: saturate at +-448, round to nearest even (E4M3)."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def fp16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).float()


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """||a - b|| / ||b|| in float64."""
    d = float(torch.linalg.vector_norm((a - b).double()))
    n = float(torch.linalg.vector_norm(b.double()))
    if n > 0:
        return d / n
    return 0.0 if d == 0 else math.inf


def head_err(o: torch.Tensor, ref: torch.Tensor) -> float:
    """The largest relative L2 error over the 32 query heads (reported)."""
    d = (o - ref).double().pow(2).sum((0, 2)).sqrt()
    n = ref.double().pow(2).sum((0, 2)).sqrt().clamp_min(1e-30)
    return float((d / n).max())


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: torch.Tensor, p_dtype=None) -> torch.Tensor:
    """Causal GQA attention in float32 of the queries at positions qpos over the keys 0 .. qpos (scale 1.0, no
    soft-cap, as Gemma 4): q [Nq, 32, Dk], k [L, 4, Dk], v [L, 4, Dv] -> [Nq, 32, Dv]. Query head h reads KV head
    h // 8. With p_dtype, P is rounded to that dtype before the PV product (reported emulation)."""
    nq, length = q.shape[0], k.shape[0]
    future = torch.arange(length, device=q.device)[None, :] > qpos[:, None]          # [Nq, L]
    out = torch.empty(nq, HQ, v.shape[-1], dtype=torch.float32, device=q.device)
    for h in range(HK):
        s = torch.einsum("qgd,ld->qgl", q[:, h * GQA:(h + 1) * GQA], k[:, h])
        p = torch.softmax(s.masked_fill(future[:, None, :], float("-inf")), dim=-1)
        if p_dtype is not None:
            p = p.to(p_dtype).float()
        out[:, h * GQA:(h + 1) * GQA] = torch.einsum("qgl,ld->qgd", p, v[:, h])
    return out


def w_from_data(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """w on the 384 unrotated dims: per dim, the median over tokens and heads of K / V where |V| >= V_FLOOR."""
    un = torch.tensor(UN, device=k.device)
    kk, vv = k[..., un].reshape(-1, len(UN)), v[..., un].reshape(-1, len(UN))
    r = torch.where(vv.abs() >= V_FLOOR, kk / vv, torch.full_like(vv, float("nan")))
    return torch.nanmedian(r, dim=0).values


def full_w(w_un: torch.Tensor) -> torch.Tensor:
    """A 512-value w from its 384 unrotated values (1.0 on the rotated dims, which the views query does not use)."""
    w = torch.ones(D, device=w_un.device)
    w[torch.tensor(UN, device=w_un.device)] = w_un
    return w


def w_forms(w_mod: torch.Tensor | None, device) -> dict:
    """The two module forms of w, rounded to BF16 as an engine stores them."""
    if w_mod is None:
        return {}
    w_mod = w_mod.to(device).float()
    return {"weight": bf16(w_mod), "one_plus_weight": bf16(1.0 + w_mod)}


def w_choice(w_data: torch.Tensor, w_mod: torch.Tensor | None) -> tuple[str, torch.Tensor, dict]:
    """The w of the views query: the module form nearest to the data (the data decide between weight and 1 + weight).
    Without a module weight: the data, rounded to BF16."""
    un = torch.tensor(UN, device=w_data.device)
    forms = w_forms(w_mod, w_data.device)
    if not forms:
        return "data", bf16(full_w(w_data)), {}
    dist = {f: rel(c[un], w_data) for f, c in forms.items()}
    form = min(dist, key=lambda f: dist[f] if math.isfinite(dist[f]) else math.inf)
    return form, forms[form], dist


def identity_err(k: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> float:
    """V3: the relative error of K on the unrotated dims against w * V there."""
    un = torch.tensor(UN, device=k.device)
    return rel(w[un] * v[..., un], k[..., un])


def rope_check(k: torch.Tensor, v: torch.Tensor, kpos: torch.Tensor, w_mod: torch.Tensor | None) -> dict:
    """Reported (the capture point): fit K on each rotated pair (i, i + 256) as the neox RoPE rotation of (a V_i, b V_j)
    with the Gemma 4 global frequencies, and as no rotation. For a capture after RoPE the first fit leaves a residual
    near the BF16 level and the second fit fails. The fitted (a, b) are w on the rotated dims."""
    dev = k.device
    i = torch.arange(ca.ROPE_ANGLES, device=dev)
    j = i + D // 2
    inv = 1.0 / ca.THETA ** (torch.arange(0, 2 * ca.ROPE_ANGLES, 2, dtype=torch.float64, device=dev) / D)
    ang = kpos.to(dev, torch.float64)[:, None, None] * inv                       # [L, 1, 64]
    c, s = torch.cos(ang), torch.sin(ang)
    ki, kj, vi, vj = (x.double() for x in (k[..., i], k[..., j], v[..., i], v[..., j]))   # [L, 4, 64]
    den_i, den_j = (vi * vi).sum((0, 1)), (vj * vj).sum((0, 1))
    tot = (ki * ki + kj * kj).sum()
    a = (vi * (c * ki + s * kj)).sum((0, 1)) / den_i
    b = (vj * (c * kj - s * ki)).sum((0, 1)) / den_j
    res = ((ki - a * vi * c + b * vj * s) ** 2 + (kj - b * vj * c - a * vi * s) ** 2).sum()
    a0, b0 = (vi * ki).sum((0, 1)) / den_i, (vj * kj).sum((0, 1)) / den_j
    res0 = ((ki - a0 * vi) ** 2 + (kj - b0 * vj) ** 2).sum()
    out = {"resid": float((res / tot).sqrt()), "resid_no_rope": float((res0 / tot).sqrt())}
    out["after_rope"] = out["resid"] <= ROPE_OK < out["resid_no_rope"]
    w_rot = torch.cat([a, b]).float()                                           # the ROT order: 0..63, 256..319
    rot = torch.tensor(ROT, device=dev)
    for f, c_w in w_forms(w_mod, dev).items():
        out[f"w_rot_vs_{f}"] = rel(c_w[rot], w_rot)
    return out


def stats(x: torch.Tensor, x8: torch.Tensor) -> dict:
    """Counts of one stored tensor: values, saturated (|x| > 448), subnormal after the cast (0 < |x8| < 2^-6), flushed
    to zero, not finite, and the largest |x| before and after the cast."""
    a = x.abs()
    fin = torch.isfinite(x)
    return {"n": x.numel(), "sat": int((a > FP8_MAX).sum()),
            "sub": int(((x8 != 0) & (x8.abs() < FP8_TINY)).sum()), "flush": int(((x != 0) & (x8 == 0)).sum()),
            "nonfinite": int((~fin).sum()), "max_pre": float(a[fin].max()) if bool(fin.any()) else math.nan,
            "max_post": float(torch.nan_to_num(x8.abs(), nan=0.0).max())}


def merge(*parts: dict) -> dict:
    out = {k: sum(p[k] for p in parts) for k in ("n", "sat", "sub", "flush", "nonfinite")}
    out |= {k: max(p[k] for p in parts) for k in ("max_pre", "max_post")}
    return out


def layer_metrics(cap: dict, w_mod: torch.Tensor | None, device) -> dict:
    """Every metric of one prompt and global layer (README Section 5) from its recorded BF16 tensors."""
    q, k, v = (cap[n].to(device).float() for n in ("q", "k", "v"))
    qpos = cap["qpos"].to(device).long()
    kpos = cap["kpos"].to(device).long() if "kpos" in cap else torch.arange(k.shape[0], device=device)
    out = {"shapes": {n: list(cap[n].shape) for n in ("q", "k", "v")},
           "dtypes": {n: str(cap[n].dtype).replace("torch.", "") for n in ("q", "k", "v")},
           "kpos_ok": bool(torch.equal(kpos, torch.arange(k.shape[0], device=device))),
           "qpos_first": int(qpos.min()), "qpos_last": int(qpos.max())}
    w_data = w_from_data(k, v)
    form, w, dist = w_choice(w_data, w_mod)
    forms = w_forms(w_mod, device)
    out |= {"w_form": form, "w_dist": dist, "w_match": bool(dist) and min(dist.values()) <= W_MATCH,
            "w_min": float(w[torch.tensor(UN, device=device)].min()),
            "w_max": float(w[torch.tensor(UN, device=device)].max()),
            "identity_err": identity_err(k, v, w), "identity_err_data": identity_err(k, v, bf16(full_w(w_data))),
            "identity_err_forms": {f: identity_err(k, v, c) for f, c in forms.items()},
            "w_data": [round(float(x), 6) for x in w_data], "rope": rope_check(k, v, kpos, w_mod)}
    ref = attend(q, k, v, qpos)                                                 # (a) BF16 K and V
    out["ref_finite"] = bool(torch.isfinite(ref).all())
    k8, v8 = fp8(k), fp8(v)
    o_std = attend(q, k8, v8, qpos)                                             # (b) standard FP8
    row = ca.pack_views(k, v)                                                   # [K_rot | V_unrot | V_rot]
    row8 = fp8(row)
    kview, vview = ca.views_of(row8)
    qv = bf16(ca.views_query(q, w))                                             # [q_rot | (q * w)_unrot] in BF16
    o_cmp = ca.unpermute(attend(qv, kview, vview, qpos))                        # (c) compact views
    out |= {"err_std": rel(o_std, ref), "err_cmp": rel(o_cmp, ref),
            "err_std_head_max": head_err(o_std, ref), "err_cmp_head_max": head_err(o_cmp, ref),
            "q_max": float(q.abs().max()), "qv_max": float(qv.abs().max()),
            "qv_over_fp16": int((qv.abs() > FP16_MAX).sum()),
            "kview_max_pre": float(torch.cat([row[..., :len(ROT)], row[..., len(ROT):len(ROT) + len(UN)]], -1).abs().max()),
            "vview_max_pre": float(v.abs().max()), "kview_max_post": float(kview.abs().max()),
            "vview_max_post": float(vview.abs().max()),
            "store_std": merge(stats(k, k8), stats(v, v8)), "store_std_k": stats(k, k8), "store_std_v": stats(v, v8),
            "store_cmp": stats(row, row8)}
    # Exploratory (reported only): the views query with w from the data, and an FP16-compute emulation (q and P in
    # FP16; the FP8 values are exact in FP16), as the stock FA4 FP8 path computes in FP16 (E-52).
    qw = bf16(ca.views_query(q, bf16(full_w(w_data))))
    out["err_cmp_wdata"] = rel(ca.unpermute(attend(qw, kview, vview, qpos)), ref)
    o_std16 = attend(fp16(q), k8, v8, qpos, torch.float16)
    o_cmp16 = ca.unpermute(attend(fp16(qv), kview, vview, qpos, torch.float16))
    out |= {"err_std_fp16": rel(o_std16, ref), "err_cmp_fp16": rel(o_cmp16, ref),
            "nonfinite_fp16": int((~torch.isfinite(o_std16)).sum() + (~torch.isfinite(o_cmp16)).sum())}
    return out


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def compute_metrics(cap_dir: Path, res: Path, device: str) -> dict:
    """Step 1: metrics.json from the capture files <cap_dir>/p<i>/L<layer>.pt and <cap_dir>/k_norm.pt. The file is
    written again after each prompt, so a timeout keeps the finished prompts."""
    torch.backends.cuda.matmul.allow_tf32 = False                               # float32 means float32
    torch.backends.cudnn.allow_tf32 = False
    out = {"started_at": now(), "device": device, "torch": torch.__version__, "status": "started", "prompts": {},
           "errors": []}
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
    path = res / "metrics.json"
    kn_path = cap_dir / "k_norm.pt"
    knorm = torch.load(kn_path, map_location="cpu", weights_only=True) if kn_path.exists() else {}
    out["k_norm_layers"] = sorted(int(x) for x in knorm)
    t0 = time.time()
    for pdir in sorted(cap_dir.glob("p*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1):
        if not pdir.is_dir() or not pdir.name[1:].isdigit():
            continue
        pid = pdir.name[1:]
        for f in sorted(pdir.glob("L*.pt")):
            layer = int(f.stem[1:])
            try:
                cap = torch.load(f, map_location=device, weights_only=True)
                out["prompts"].setdefault(pid, {})[str(layer)] = layer_metrics(cap, knorm.get(layer), device)
            except Exception as exc:  # recorded; V2 then fails for this layer
                out["errors"].append(f"p{pid} L{layer}: {type(exc).__name__}: {exc}"[:300])
        out["seconds"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(out, indent=1))
        print(f"metrics: prompt {pid} done after {out['seconds']} s", flush=True)
    out.update(status="ok", finished_at=now(), seconds=round(time.time() - t0, 1))
    path.write_text(json.dumps(out, indent=1))
    return out


def ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    if b > 0:
        return a / b
    return 1.0 if a == 0 else math.inf


def layer_summary(rows: list[dict]) -> dict:
    """The aggregates of one global layer over the prompts, and the PASS conditions of README Section 7."""
    es, ec = [r["err_std"] for r in rows], [r["err_cmp"] for r in rows]
    finite = all(math.isfinite(x) for x in es + ec)
    med_s = statistics.median(es) if finite else math.nan
    med_c = statistics.median(ec) if finite else math.nan
    max_s, max_c = (max(es), max(ec)) if finite else (math.nan, math.nan)

    def share(key: str, field: str) -> float:
        return sum(r[key][field] for r in rows) / max(1, sum(r[key]["n"] for r in rows))

    out = {"prompts": len(rows), "err_std": es, "err_cmp": ec, "finite": finite, "med_std": med_s, "med_cmp": med_c,
           "ratio_med": ratio(med_c, med_s), "max_std": max_s, "max_cmp": max_c, "ratio_max": ratio(max_c, max_s),
           "q_max": max(r["qv_max"] for r in rows), "sat_std": share("store_std", "sat"),
           "sat_cmp": share("store_cmp", "sat"), "sub_std": share("store_std", "sub"),
           "sub_cmp": share("store_cmp", "sub"), "flush_std": share("store_std", "flush"),
           "flush_cmp": share("store_cmp", "flush"), "identity_max": max(r["identity_err"] for r in rows),
           "w_forms": sorted({r["w_form"] for r in rows}), "w_match": all(r["w_match"] for r in rows),
           "rope_after": all(r["rope"]["after_rope"] for r in rows),
           "rope_resid_max": max(r["rope"]["resid"] for r in rows),
           "rope_resid_no_rope_min": min(r["rope"]["resid_no_rope"] for r in rows),
           "exploratory": {k: statistics.median([r[k] for r in rows]) for k in
                           ("err_cmp_wdata", "err_std_fp16", "err_cmp_fp16", "err_std_head_max", "err_cmp_head_max")}
           | {"nonfinite_fp16": sum(r["nonfinite_fp16"] for r in rows),
              "kview_max_pre": max(r["kview_max_pre"] for r in rows),
              "vview_max_pre": max(r["vview_max_pre"] for r in rows), "q_max_raw": max(r["q_max"] for r in rows)}}
    out["pass"] = {"median": finite and out["ratio_med"] <= MED_PASS, "largest": finite and out["ratio_max"] <= MAX_PASS,
                   "q": out["q_max"] <= Q_PASS, "sat": out["sat_cmp"] <= out["sat_std"] + SAT_PP}
    return out


def v2_problems(run: dict, met: dict, prompts: dict) -> list[str]:
    """V2: the capture finished, its plan is the frozen plan (a stub may use a small plan), the hooked layers are the
    global layers of the config, and every global layer of every prompt has metrics with the expected shapes."""
    bad = []
    if not run:
        return ["no capture record (capture_run.json)"]
    if run.get("status") != "ok":
        bad.append(f"the capture status is {run.get('status')!r}: {run.get('error') or run.get('fail_fast') or ''}"[:300])
    plan = run.get("plan") or {}
    if not run.get("stub") and any(plan.get(k) != v for k, v in PLAN.items()):
        bad.append(f"the plan {plan} is not the frozen plan {PLAN}")
    expected = (run.get("config") or {}).get("global_layers") or []
    hooked = sorted(h["layer"] for h in (run.get("hooks") or {}).get("hooked", []))
    if not expected:
        bad.append("the config names no global layer")
    elif hooked != sorted(expected):
        bad.append(f"hooked layers {hooked} are not the config global layers {sorted(expected)}")
    lengths = {str(p["id"]): p["n_tokens"] for p in prompts.get("prompts", [])}
    q_rows = plan.get("q_rows")
    for pid in range(plan.get("prompts") or 0):
        length = lengths.get(str(pid))
        if length != plan.get("tokens"):
            bad.append(f"prompt {pid} has {length} tokens, the plan {plan.get('tokens')}")
        got = (met.get("prompts") or {}).get(str(pid), {})
        for layer in expected:
            m = got.get(str(layer))
            if m is None:
                bad.append(f"prompt {pid} layer {layer}: no metrics")
                continue
            want = {"q": [q_rows, HQ, D], "k": [length, HK, D], "v": [length, HK, D]}
            if m["shapes"] != want:
                bad.append(f"prompt {pid} layer {layer}: shapes {m['shapes']}, expected {want}")
            if set(m["dtypes"].values()) != {"bfloat16"}:
                bad.append(f"prompt {pid} layer {layer}: dtypes {m['dtypes']}")
            if not m["kpos_ok"] or (length and (m["qpos_first"], m["qpos_last"]) != (length - q_rows, length - 1)):
                bad.append(f"prompt {pid} layer {layer}: positions do not match the plan")
    return bad


def summarize(res: Path) -> dict:
    run = load(res / "capture_run.json", {})
    prompts = load(res / "prompts.json", {})
    met = load(res / "metrics.json", {})
    expected = (run.get("config") or {}).get("global_layers") or []
    pids = [str(i) for i in range((run.get("plan") or {}).get("prompts") or 0)]   # only the prompts of the plan
    got = met.get("prompts") or {}
    rows = {str(layer): [got[p][str(layer)] for p in pids if str(layer) in got.get(p, {})] for layer in expected}
    layers = {layer: layer_summary(r) for layer, r in rows.items() if r}
    flat = [m for r in rows.values() for m in r]
    gpu = (run.get("gpu") or {}).get("name", "")
    v2 = v2_problems(run, met, prompts)
    id_bad = [(p, layer, m["identity_err"]) for p in pids for layer, m in got.get(p, {}).items()
              if not m["identity_err"] <= ID_TOL]
    ref_bad = [(p, layer) for p in pids for layer, m in got.get(p, {}).items() if not m["ref_finite"]]
    checks = {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""),
              "V2": not v2,
              "V3": bool(flat) and not id_bad,
              "V4": bool(flat) and not ref_bad,
              "_gpu": gpu, "_v2_problems": v2[:20], "_v2_count": len(v2),
              "_identity_max": max((m["identity_err"] for m in flat), default=None),
              "_identity_failures": id_bad[:20], "_ref_nonfinite": ref_bad[:20],
              "_metrics_status": met.get("status"), "_metrics_errors": (met.get("errors") or [])[:20]}
    ledger = {"err_std": [m["err_std"] for m in flat], "err_cmp": [m["err_cmp"] for m in flat],
              "ratio": [ratio(m["err_cmp"], m["err_std"]) for m in flat]}
    return {"capture_status": run.get("status", "missing"), "fail_fast": run.get("fail_fast"), "stub": run.get("stub"),
            "plan": run.get("plan"), "gpu": run.get("gpu"), "versions": {k: run.get(k) for k in ("vllm", "torch")},
            "engine": run.get("engine"), "config": run.get("config"), "hooks": {k: v for k, v in
                                                                                 (run.get("hooks") or {}).items()
                                                                                 if k != "hooked"},
            "prompts": [{k: p.get(k) for k in ("id", "family", "n_tokens", "token_sha256")}
                        for p in prompts.get("prompts", [])],
            "metrics_device": met.get("device"), "metrics_gpu": met.get("gpu"), "metrics_seconds": met.get("seconds"),
            "checks": checks, "layers": layers,
            "q_max": max((v["q_max"] for v in layers.values()), default=None),
            "ledger_metrics": {k: {"values": [x for x in v if math.isfinite(x)], "unit": "relative L2 error"
                                   if k != "ratio" else "err_cmp / err_std", "higher_is_better": False}
                               for k, v in ledger.items()}}


def fmt(x: float) -> str:
    return f"{x:,.1f}" if x is not None and math.isfinite(x) else str(x)


def decide(s: dict) -> str:
    """The frozen rule (README Sections 6 and 7)."""
    c = s["checks"]
    if s.get("capture_status") == "fail_fast":
        return f"INVALID (V2: fail fast after the first prompt: {'; '.join(s.get('fail_fast') or [])[:400]})"
    bad = [k for k in ("V1", "V2", "V3", "V4") if not c.get(k)]
    if bad:
        why = {"V1": f"GPU {c.get('_gpu')!r}", "V2": "; ".join(c.get("_v2_problems") or [])[:300],
               "V3": f"largest identity error {c.get('_identity_max')}", "V4": f"reference not finite in "
                                                                               f"{c.get('_ref_nonfinite')}"}
        return f"INVALID ({', '.join(bad)}: " + "; ".join(f"{k} {why[k]}" for k in bad) + ")"
    L = s["layers"]
    fail = [f"layer {layer}: median err_compact {v['med_cmp']:.4f} is {v['ratio_med']:.2f} x err_standard "
            f"{v['med_std']:.4f} (above {MED_FAIL})" for layer, v in L.items() if v["ratio_med"] > MED_FAIL]
    if s["q_max"] is not None and s["q_max"] > Q_FAIL:
        fail.append(f"largest |q'| {fmt(s['q_max'])} above {fmt(Q_FAIL)}")
    rm = [v["ratio_med"] for v in L.values()]
    rx = [v["ratio_max"] for v in L.values()]
    tag = (f"{len(L)} global layers; median err_standard {min(v['med_std'] for v in L.values()):.4f} to "
           f"{max(v['med_std'] for v in L.values()):.4f}; median ratio {min(rm):.3f} to {max(rm):.3f} (bound "
           f"{MED_PASS}); largest-error ratio {min(rx):.3f} to {max(rx):.3f} (bound {MAX_PASS}); largest |q'| "
           f"{fmt(s['q_max'])} (bound {fmt(Q_PASS)}); saturated share compact up to "
           f"{100 * max(v['sat_cmp'] for v in L.values()):.4f}% against standard up to "
           f"{100 * max(v['sat_std'] for v in L.values()):.4f}% (bound +{100 * SAT_PP:.1f} pp)")
    if fail:
        return "FAIL (" + "; ".join(fail) + f"; {tag})"
    miss = []
    for layer, v in L.items():
        p = v["pass"]
        if not p["median"]:
            miss.append(f"layer {layer}: median ratio {v['ratio_med']:.3f} above {MED_PASS}")
        if not p["largest"]:
            miss.append(f"layer {layer}: largest-error ratio {v['ratio_max']:.3f} above {MAX_PASS}")
        if not p["q"]:
            miss.append(f"layer {layer}: largest |q'| {fmt(v['q_max'])} above {fmt(Q_PASS)}")
        if not p["sat"]:
            miss.append(f"layer {layer}: saturated share compact {100 * v['sat_cmp']:.4f}% above standard "
                        f"{100 * v['sat_std']:.4f}% + {100 * SAT_PP:.1f} pp")
    if not miss:
        return f"PASS ({tag})"
    return "INCONCLUSIVE (" + "; ".join(miss) + f"; {tag})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, type=Path, help="results folder (capture_run.json, prompts.json)")
    ap.add_argument("--capture", type=Path, help="capture folder: compute metrics.json from it first")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.capture:
        compute_metrics(a.capture, a.res, pick_device(a.device))
    s = summarize(a.res)
    s["decision"] = decide(s)
    if a.out:
        a.out.write_text(json.dumps(s, indent=1, default=str))
    print(s["decision"])


if __name__ == "__main__":
    main()
