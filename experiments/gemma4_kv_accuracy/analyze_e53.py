"""E-53 analysis (README Sections 5 to 7; gate G3 of D-88; amendment A1 after R-73): on real Gemma 4 31B activations,
does the compact FP8 KV storage of the global layers (the E-52 views row [K_rot | V_unrot | V_rot] in FP8 E4M3 with
scale 1.0, read with the views query [q_rot | (q * w)_unrot] rounded to BF16) give an attention-output error no larger
than the standard FP8 KV storage (K and V each in E4M3 with scale 1.0), and do the views-path values fit FP16 with
margin?

The script has two steps:
1. Metrics (on the Pod GPU, or later on the Mac from the copied captures, with `--capture <dir>`). For each prompt and
   global layer, it reads the BF16 q, k, v and (A1) the engine output o that capture_e53.py recorded, and checks the
   SHA-256 of each file against capture_run.json (A1, R-73 F2). It computes in float32 the causal attention output of
   the recorded queries over all earlier tokens (GQA, scale 1.0, no soft-cap) for (a) the reference (BF16 K and V),
   (b) the standard FP8 storage and (c) the compact views storage (output after `unpermute`). It writes metrics.json.
2. Decision (on the Pod or on the Mac, from metrics.json): the validity checks V1 to V5 and the frozen rule.

The FP8 cast is the cast of the vLLM cache write (FACT, vLLM 3966924d5, README Section 4.5): BF16 to float, divide by
the scale (1.0), then `__nv_cvt_float_to_fp8(..., __NV_SATFINITE, __NV_E4M3)`. That is round to nearest even with
saturation at +-448. This script clamps to +-448 and then casts with `to(torch.float8_e4m3fn)`.

w is the k_norm weight on the 384 unrotated dims (K = w * V there). The script recovers w from the data (the median of
K_unrot / V_unrot per dim) and reads it from the module. A Gemma RMSNorm can scale by (1 + weight), so the data decide
which module form the views query uses. (A1, R-73 F5) V3 needs the module weight of every global layer.

A1 rule (README Section 7): per prompt the ratio r = err_cmp / err_std (float32) and r16 (the FP16 emulation); PASS uses
max(r, r16): its median <= 1.25 and its largest <= 2 in every layer, |q'| <= 6,550, no FP16 value that is not finite,
and no more saturated compact values than standard ones (counts, R-73 B3). FAIL: a layer with a median float32 ratio
above 2. A |q'| above the FP16 range is INCONCLUSIVE with the action "pre-scale the query" (R-73 B4). All thresholds
are project heuristics.

Usage: analyze_e53.py --res <dir> [--capture <dir>] [--device auto|cpu|cuda] [--out summary.json]
"""

from __future__ import annotations

import argparse
import hashlib
import io
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

D = 512                               # head_dim of the global layers (the views layout needs 512)
PLAN = {"prompts": 4, "tokens": 8192, "q_rows": 256, "heads": 32, "kv_heads": 4, "head_dim": 512,
        "global_layers": 10}          # the frozen plan (README Section 4; A1 adds the shapes)
VLLM_VERSION = "0.33.0rc1.dev229+g3966924d5"   # (A1, R-73 F5) the image of E-49 to E-52
EAGER = "NONE"                        # (A1, R-73 F7) the compile and CUDA graph mode names in eager mode
BF16_CACHE = ("auto", "bfloat16", "bf16")
FP8_MAX = 448.0                       # E4M3 saturation (__NV_SATFINITE)
E5M2_MAX = 57344.0                    # E5M2 saturation (the O2 control, reported)
FP8_TINY = 2.0 ** -6                  # the smallest normal E4M3 value
FP16_MAX = 65504.0
ID_TOL = 1e-2                         # V3: K_unrot against w * V_unrot (the BF16 level)
W_MATCH = 0.02                        # V3 (A1, F5): the chosen module form of w agrees with the data
ENGINE_TOL = 0.05                     # V5 (A1, B2): the reference against the engine output
MED_PASS = 1.25                       # PASS: median per-prompt ratio <= 1.25, in every layer
MAX_PASS = 2.0                        # PASS: largest per-prompt ratio <= 2, in every layer
MED_FAIL = 2.0                        # FAIL: a layer with a median float32 per-prompt ratio above 2
Q_PASS = 6550.0                       # PASS: largest |q'| at most 6,550 (a 10 x FP16 margin)
Q_RANGE = FP16_MAX / 2                # (A1, B4) above 32,752: INCONCLUSIVE, action "pre-scale the query"
V_FLOOR = 0.05                        # the |V| floor for the ratio K / V (V has RMS 1)
ROPE_OK = 0.03                        # reported: the RoPE fit residual of a capture after RoPE
UN = ca.UNROT
ROT = ca.ROT
ACTION = {  # (A1, R-73 B4) the consequence of each outcome (README Section 7)
    "PASS": "week 2 of the build starts (D-88)",
    "FAIL": "stop the build (D-88); I-182 keeps its exact capacity result",
    "range": "pre-scale the query (q' times 2^-m with the kernel scale 2^m) and test the scale argument in week 2; "
             "the user confirms",
    "other": "the user decides; the report lists the layers and the conditions; no rerun without a new S4",
    "fail_fast": "fix the cause, then one rerun under a new S4 with a stated cost cap",
    "V3": "report the recovered w; the user decides",
    "V5": "the reference does not model the engine: fix the reference (head mapping, scale, mask) and analyze the "
          "copied captures again on the Mac; no new Pod",
    "invalid": "fix the cause; a rerun needs a new S4",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def fp8(x: torch.Tensor) -> torch.Tensor:
    """The vLLM FP8 cache cast with scale 1.0: saturate at +-448, round to nearest even (E4M3)."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def e5m2(x: torch.Tensor) -> torch.Tensor:
    """(A1, R-73 O2) The same cast to E5M2 (2 mantissa bits): the sensitivity control, reported only."""
    return x.clamp(-E5M2_MAX, E5M2_MAX).to(torch.float8_e5m2).float()


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
    """The largest relative L2 error over the query heads (reported)."""
    d = (o - ref).double().pow(2).sum((0, 2)).sqrt()
    n = ref.double().pow(2).sum((0, 2)).sqrt().clamp_min(1e-30)
    return float((d / n).max())


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: torch.Tensor, p_dtype=None) -> torch.Tensor:
    """Causal GQA attention in float32 of the queries at positions qpos over the keys 0 .. qpos (scale 1.0, no
    soft-cap, as Gemma 4): q [Nq, Hq, Dk], k [L, Hk, Dk], v [L, Hk, Dv] -> [Nq, Hq, Dv]. Query head h reads KV head
    h // (Hq / Hk). With p_dtype, P is rounded to that dtype before the PV product (the FP16 emulation)."""
    nq, hq, length, hk = q.shape[0], q.shape[1], k.shape[0], k.shape[1]
    g = hq // hk
    future = torch.arange(length, device=q.device)[None, :] > qpos[:, None]          # [Nq, L]
    out = torch.empty(nq, hq, v.shape[-1], dtype=torch.float32, device=q.device)
    for h in range(hk):
        s = torch.einsum("qgd,ld->qgl", q[:, h * g:(h + 1) * g], k[:, h])
        p = torch.softmax(s.masked_fill(future[:, None, :], float("-inf")), dim=-1)
        if p_dtype is not None:
            p = p.to(p_dtype).float()
        out[:, h * g:(h + 1) * g] = torch.einsum("qgl,ld->qgd", p, v[:, h])
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
    Without a module weight: the data, rounded to BF16 (then V3 fails, A1)."""
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


def identity_token_max(k: torch.Tensor, v: torch.Tensor, w: torch.Tensor) -> float:
    """(A1, R-73 F5) Reported: the largest identity error of one token and head (an L2 error over all tokens hides a
    break at one token, for example the BOS sink)."""
    un = torch.tensor(UN, device=k.device)
    kk = k[..., un].double()
    d = (w[un].double() * v[..., un].double() - kk).pow(2).sum(-1).sqrt()
    n = kk.pow(2).sum(-1).sqrt().clamp_min(1e-30)
    return float((d / n).max())


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
    ki, kj, vi, vj = (x.double() for x in (k[..., i], k[..., j], v[..., i], v[..., j]))   # [L, Hk, 64]
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
    names = [n for n in ("q", "k", "v", "o") if n in cap]
    out = {"shapes": {n: list(cap[n].shape) for n in names},
           "dtypes": {n: str(cap[n].dtype).replace("torch.", "") for n in names},
           "kpos_ok": bool(torch.equal(kpos, torch.arange(k.shape[0], device=device))),
           "qpos_first": int(qpos.min()), "qpos_last": int(qpos.max()), "w_module": w_mod is not None}
    un_idx, rot_idx = torch.tensor(UN, device=device), torch.tensor(ROT, device=device)
    w_data = w_from_data(k, v)
    form, w, dist = w_choice(w_data, w_mod)
    forms = w_forms(w_mod, device)
    out |= {"w_form": form, "w_dist": dist, "w_match": bool(dist) and min(dist.values()) <= W_MATCH,
            "w_min": float(w[un_idx].min()), "w_max": float(w[un_idx].max()),
            "identity_err": identity_err(k, v, w), "identity_err_data": identity_err(k, v, bf16(full_w(w_data))),
            "identity_err_forms": {f: identity_err(k, v, c) for f, c in forms.items()},
            "identity_token_max": identity_token_max(k, v, w),
            "w_data": [round(float(x), 6) for x in w_data], "rope": rope_check(k, v, kpos, w_mod)}
    ref = attend(q, k, v, qpos)                                                 # (a) BF16 K and V
    out["ref_finite"] = bool(torch.isfinite(ref).all())
    if "o" in cap:                                                              # (A1, R-73 B2) V5
        o_eng = cap["o"].to(device).float()
        out["engine_gap"] = rel(ref, o_eng) if o_eng.shape == ref.shape else math.inf
        out["engine_gap_head_max"] = head_err(ref, o_eng) if o_eng.shape == ref.shape else math.inf
    else:
        out["engine_gap"] = out["engine_gap_head_max"] = None
    k8, v8 = fp8(k), fp8(v)
    o_std = attend(q, k8, v8, qpos)                                             # (b) standard FP8
    row = ca.pack_views(k, v)                                                   # [K_rot | V_unrot | V_rot]
    row8 = fp8(row)
    kview, vview = ca.views_of(row8)
    kview_b, vview_b = ca.views_of(row)                                         # the same views without FP8
    qv = bf16(ca.views_query(q, w))                                             # [q_rot | (q * w)_unrot] in BF16
    o_cmp = ca.unpermute(attend(qv, kview, vview, qpos))                        # (c) compact views
    nrot = len(ROT)
    out |= {"err_std": rel(o_std, ref), "err_cmp": rel(o_cmp, ref),
            "err_std_head_max": head_err(o_std, ref), "err_cmp_head_max": head_err(o_cmp, ref),
            "q_max": float(q.abs().max()), "qv_max": float(qv.abs().max()),
            "qv_over_fp16": int((qv.abs() > FP16_MAX).sum()),
            "kview_max_pre": float(row[..., :D].abs().max()), "vview_max_pre": float(v.abs().max()),
            "kview_max_post": float(kview.abs().max()), "vview_max_post": float(vview.abs().max()),
            "store_std": merge(stats(k, k8), stats(v, v8)), "store_std_k": stats(k, k8), "store_std_v": stats(v, v8),
            "store_std_krot": stats(k[..., rot_idx], k8[..., rot_idx]),           # (A1, B3) the same support
            "store_std_kunrot": stats(k[..., un_idx], k8[..., un_idx]),         # (A1, B3) the compact rows avoid it
            "store_cmp": stats(row, row8), "store_cmp_krot": stats(row[..., :nrot], row8[..., :nrot])}
    # (A1, R-73 F3) the FP16 emulation enters the rule: q, q' and P in FP16 (the FP8 values are exact in FP16), as the
    # stock FA4 FP8 path computes in FP16 (E-52).
    o_std16 = attend(fp16(q), k8, v8, qpos, torch.float16)
    o_cmp16 = ca.unpermute(attend(fp16(qv), kview, vview, qpos, torch.float16))
    out |= {"err_std_fp16": rel(o_std16, ref), "err_cmp_fp16": rel(o_cmp16, ref),
            "nonfinite_fp16": int((~torch.isfinite(o_std16)).sum() + (~torch.isfinite(o_cmp16)).sum())}
    # Reported only: w from the data; (A1, R-73 F6) the score path and the value path of each storage; (A1, R-73 O2)
    # the compact storage in E5M2.
    qw = bf16(ca.views_query(q, bf16(full_w(w_data))))
    out["err_cmp_wdata"] = rel(ca.unpermute(attend(qw, kview, vview, qpos)), ref)
    out |= {"err_std_score": rel(attend(q, k8, v, qpos), ref), "err_std_value": rel(attend(q, k, v8, qpos), ref),
            "err_cmp_score": rel(ca.unpermute(attend(qv, kview, vview_b, qpos)), ref),
            "err_cmp_value": rel(ca.unpermute(attend(qv, kview_b, vview, qpos)), ref)}
    kv5, vv5 = ca.views_of(e5m2(row))
    out["err_cmp_e5m2"] = rel(ca.unpermute(attend(qv, kv5, vv5, qpos)), ref)
    return out


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def compute_metrics(cap_dir: Path, res: Path, device: str) -> dict:
    """Step 1: metrics.json from the capture files <cap_dir>/p<i>/L<layer>.pt and <cap_dir>/k_norm.pt. (A1, R-73 F2)
    The SHA-256 of each file is compared with capture_run.json. The file is written again after each prompt, so a
    timeout keeps the finished prompts."""
    torch.backends.cuda.matmul.allow_tf32 = False                               # float32 means float32
    torch.backends.cudnn.allow_tf32 = False
    run = load(res / "capture_run.json", {})
    want = {(str(e["id"]), str(layer)): f.get("sha256") for e in run.get("prompts", [])
            for layer, f in (e.get("files") or {}).items()}
    out = {"started_at": now(), "device": device, "torch": torch.__version__, "status": "started", "prompts": {},
           "errors": [], "sha_checked": 0, "sha_mismatch": []}
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
                b = f.read_bytes()
                got = hashlib.sha256(b).hexdigest()
                if (pid, str(layer)) in want:
                    out["sha_checked"] += 1
                    if want[(pid, str(layer))] != got:
                        out["sha_mismatch"].append(f"p{pid} L{layer}")
                cap = torch.load(io.BytesIO(b), map_location=device, weights_only=True)
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


def worst(a: float, b: float) -> float:
    """The larger ratio; NaN if either one is NaN (then the PASS test fails)."""
    return math.nan if math.isnan(a) or math.isnan(b) else max(a, b)


def layer_summary(rows: list[dict]) -> dict:
    """The aggregates of one global layer over the prompts, and the PASS conditions of README Section 7 (A1)."""
    es, ec = [r["err_std"] for r in rows], [r["err_cmp"] for r in rows]
    r32 = [ratio(r["err_cmp"], r["err_std"]) for r in rows]                      # (A1, F3) per-prompt ratios
    r16 = [ratio(r["err_cmp_fp16"], r["err_std_fp16"]) for r in rows]
    rr = [worst(a, b) for a, b in zip(r32, r16)]
    finite = all(math.isfinite(x) for x in r32 + rr)
    med = lambda xs: statistics.median(xs) if all(math.isfinite(x) for x in xs) else math.nan   # noqa: E731
    count = lambda key, field="sat": sum(r[key][field] for r in rows)                          # noqa: E731
    share = lambda key, field: count(key, field) / max(1, count(key, "n"))                     # noqa: E731
    out = {"prompts": len(rows), "err_std": es, "err_cmp": ec, "ratio": r32, "ratio_fp16": r16, "ratio_used": rr,
           "finite": finite, "med_ratio": med(r32), "med_ratio_used": med(rr),
           "max_ratio_used": max(rr) if finite else math.nan,
           "med_std": med(es), "med_cmp": med(ec), "ratio_of_medians": ratio(med(ec), med(es)),
           "ratio_of_maxima": ratio(max(ec), max(es)) if finite else math.nan,
           "q_max": max(r["qv_max"] for r in rows), "nonfinite_fp16": sum(r["nonfinite_fp16"] for r in rows),
           "sat_std_count": count("store_std"), "sat_cmp_count": count("store_cmp"),
           "sat_std_kunrot_count": count("store_std_kunrot"),
           "sat_share_krot_std": share("store_std_krot", "sat"), "sat_share_krot_cmp": share("store_cmp_krot", "sat"),
           "sat_std": share("store_std", "sat"), "sat_cmp": share("store_cmp", "sat"),
           "sub_std": share("store_std", "sub"), "sub_cmp": share("store_cmp", "sub"),
           "flush_std": share("store_std", "flush"), "flush_cmp": share("store_cmp", "flush"),
           "identity_max": max(r["identity_err"] for r in rows),
           "identity_token_max": max(r["identity_token_max"] for r in rows),
           "w_forms": sorted({r["w_form"] for r in rows}), "w_match": all(r["w_match"] for r in rows),
           "w_module": all(r["w_module"] for r in rows),
           "engine_gap_max": max((r["engine_gap"] for r in rows if r["engine_gap"] is not None), default=None),
           "rope_after": all(r["rope"]["after_rope"] for r in rows),
           "rope_resid_max": max(r["rope"]["resid"] for r in rows),
           "rope_resid_no_rope_min": min(r["rope"]["resid_no_rope"] for r in rows),
           "exploratory": {k: statistics.median([r[k] for r in rows]) for k in
                           ("err_cmp_wdata", "err_std_fp16", "err_cmp_fp16", "err_std_head_max", "err_cmp_head_max",
                            "err_std_score", "err_std_value", "err_cmp_score", "err_cmp_value", "err_cmp_e5m2")}
           | {"ratio_e5m2": statistics.median([ratio(r["err_cmp_e5m2"], r["err_std"]) for r in rows]),
              "ratio_score": statistics.median([ratio(r["err_cmp_score"], r["err_std_score"]) for r in rows]),
              "kview_max_pre": max(r["kview_max_pre"] for r in rows),
              "vview_max_pre": max(r["vview_max_pre"] for r in rows), "q_max_raw": max(r["q_max"] for r in rows)}}
    out["pass"] = {"median": finite and out["med_ratio_used"] <= MED_PASS,
                   "largest": finite and out["max_ratio_used"] <= MAX_PASS,
                   "q": out["q_max"] <= Q_PASS, "fp16": out["nonfinite_fp16"] == 0,
                   "sat": out["sat_cmp_count"] <= out["sat_std_count"]}
    return out


def v2_problems(run: dict, met: dict, prompts: dict) -> list[str]:
    """V2: the capture finished, its plan is the frozen plan (a stub or the S5 rehearsal may use its own plan), the
    hooked layers are the planned global layers of the config, and every global layer of every prompt has metrics with
    the expected shapes. (A1) Also: no prompt of the capture has a problem, no special id after position 0 (R-73 F4),
    every capture file has its recorded SHA-256 (F2), and in a real run the engine is the planned one (F5, F7)."""
    bad = []
    if not run:
        return ["no capture record (capture_run.json)"]
    if run.get("status") != "ok":
        bad.append(f"the capture status is {run.get('status')!r}: {run.get('error') or run.get('fail_fast') or ''}"[:300])
    plan = run.get("plan") or {}
    if not (run.get("stub") or run.get("rehearsal")) and any(plan.get(k) != v for k, v in PLAN.items()):
        bad.append(f"the plan {plan} is not the frozen plan {PLAN}")
    expected = (run.get("config") or {}).get("global_layers") or []
    hooked = sorted(h["layer"] for h in (run.get("hooks") or {}).get("hooked", []))
    if not expected:
        bad.append("the config names no global layer")
    elif hooked != sorted(expected):
        bad.append(f"hooked layers {hooked} are not the config global layers {sorted(expected)}")
    if expected and len(expected) != plan.get("global_layers"):
        bad.append(f"the config has {len(expected)} global layers, the plan {plan.get('global_layers')}")
    if not run.get("stub"):                                                     # (A1, R-73 F5, F7)
        eng = run.get("engine") or {}
        if run.get("vllm") != VLLM_VERSION:
            bad.append(f"vLLM {run.get('vllm')!r}, expected {VLLM_VERSION}")
        for key in ("compilation_mode", "cudagraph_mode"):
            if eng.get(key) != EAGER:
                bad.append(f"engine {key} {eng.get(key)!r}, expected {EAGER}")
        if str(eng.get("cache_dtype")) not in BF16_CACHE:
            bad.append(f"engine cache dtype {eng.get('cache_dtype')!r}, expected a BF16 cache")
    for e in run.get("prompts", []):
        if e.get("problems"):
            bad.append(f"capture prompt {e['id']}: {'; '.join(e['problems'])[:200]}")
    if met.get("sha_mismatch"):
        bad.append(f"capture files with another SHA-256: {met['sha_mismatch'][:10]}")
    built = {str(p["id"]): p for p in prompts.get("prompts", [])}
    q_rows = plan.get("q_rows")
    hq, hk, hd = plan.get("heads"), plan.get("kv_heads"), plan.get("head_dim")
    for pid in range(plan.get("prompts") or 0):
        p = built.get(str(pid)) or {}
        length = p.get("n_tokens")
        if length != plan.get("tokens"):
            bad.append(f"prompt {pid} has {length} tokens, the plan {plan.get('tokens')}")
        if p.get("special_after_0", 0):
            bad.append(f"prompt {pid} has {p['special_after_0']} special ids after position 0")
        got = (met.get("prompts") or {}).get(str(pid), {})
        for layer in expected:
            m = got.get(str(layer))
            if m is None:
                bad.append(f"prompt {pid} layer {layer}: no metrics")
                continue
            want = {"q": [q_rows, hq, hd], "k": [length, hk, hd], "v": [length, hk, hd], "o": [q_rows, hq, hd]}
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
    every = [(p, str(layer)) for p in pids for layer in expected]
    id_bad = [(p, layer, got[p][layer]["identity_err"]) for p, layer in every if layer in got.get(p, {})
              and not got[p][layer]["identity_err"] <= ID_TOL]
    w_bad = [(p, layer) for p, layer in every if layer in got.get(p, {})
             and not (got[p][layer]["w_module"] and got[p][layer]["w_match"])]
    ref_bad = [(p, layer) for p, layer in every if layer in got.get(p, {}) and not got[p][layer]["ref_finite"]]
    gaps = [got[p][layer].get("engine_gap") for p, layer in every if layer in got.get(p, {})]
    eng_bad = [(p, layer, got[p][layer].get("engine_gap")) for p, layer in every if layer in got.get(p, {})
               and not (got[p][layer].get("engine_gap") is not None and got[p][layer]["engine_gap"] <= ENGINE_TOL)]
    checks = {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""),
              "V2": not v2,
              "V3": bool(flat) and not id_bad and not w_bad,
              "V4": bool(flat) and not ref_bad,
              "V5": bool(flat) and not eng_bad,
              "_gpu": gpu, "_v2_problems": v2[:20], "_v2_count": len(v2),
              "_identity_max": max((m["identity_err"] for m in flat), default=None),
              "_identity_failures": id_bad[:20], "_w_failures": w_bad[:20], "_ref_nonfinite": ref_bad[:20],
              "_engine_gap_max": max((g for g in gaps if g is not None), default=None),
              "_engine_failures": eng_bad[:20],
              "_metrics_status": met.get("status"), "_metrics_errors": (met.get("errors") or [])[:20],
              "_sha_checked": met.get("sha_checked")}
    ledger = {"err_std": [m["err_std"] for m in flat], "err_cmp": [m["err_cmp"] for m in flat],
              "ratio": [ratio(m["err_cmp"], m["err_std"]) for m in flat]}
    return {"capture_status": run.get("status", "missing"), "fail_fast": run.get("fail_fast"), "stub": run.get("stub"),
            "rehearsal": run.get("rehearsal"), "plan": run.get("plan"), "gpu": run.get("gpu"),
            "versions": {k: run.get(k) for k in ("vllm", "torch")}, "engine": run.get("engine"),
            "config": run.get("config"),
            "hooks": {k: v for k, v in (run.get("hooks") or {}).items() if k != "hooked"},
            "prompts": [{k: p.get(k) for k in ("id", "family", "n_tokens", "token_sha256", "special_after_0",
                                               "skipped_special")} for p in prompts.get("prompts", [])],
            "metrics_device": met.get("device"), "metrics_gpu": met.get("gpu"), "metrics_seconds": met.get("seconds"),
            "checks": checks, "layers": layers,
            "q_max": max((v["q_max"] for v in layers.values()), default=None),
            "ledger_metrics": {k: {"values": [x for x in v if math.isfinite(x)], "unit": "relative L2 error"
                                   if k != "ratio" else "err_cmp / err_std", "higher_is_better": False}
                               for k, v in ledger.items()}}


def fmt(x: float) -> str:
    return f"{x:,.1f}" if x is not None and math.isfinite(x) else str(x)


def decide(s: dict) -> str:
    """The frozen rule (README Sections 6 and 7, A1), with the action of each outcome (A1, R-73 B4)."""
    word = verdict(s)
    return f"REHEARSAL {word}" if s.get("rehearsal") else word


def verdict(s: dict) -> str:
    c = s["checks"]
    if s.get("capture_status") == "fail_fast":
        return (f"INVALID (V2: fail fast after the first prompt: {'; '.join(s.get('fail_fast') or [])[:400]}; "
                f"action: {ACTION['fail_fast']})")
    bad = [k for k in ("V1", "V2", "V3", "V4", "V5") if not c.get(k)]
    if bad:
        why = {"V1": f"GPU {c.get('_gpu')!r}", "V2": "; ".join(c.get("_v2_problems") or [])[:300],
               "V3": f"largest identity error {c.get('_identity_max')}, layers without a module w or a match "
                     f"{c.get('_w_failures')}",
               "V4": f"reference not finite in {c.get('_ref_nonfinite')}",
               "V5": f"engine gap above {ENGINE_TOL} in {c.get('_engine_failures')}"}
        act = ACTION["V3"] if bad == ["V3"] else ACTION["V5"] if bad == ["V5"] else ACTION["invalid"]
        return f"INVALID ({', '.join(bad)}: " + "; ".join(f"{k} {why[k]}" for k in bad) + f"; action: {act})"
    L = s["layers"]
    mr = [v["med_ratio_used"] for v in L.values()]
    xr = [v["max_ratio_used"] for v in L.values()]
    tag = (f"{len(L)} global layers; median err_standard {min(v['med_std'] for v in L.values()):.4f} to "
           f"{max(v['med_std'] for v in L.values()):.4f}; median per-prompt ratio {min(mr):.3f} to {max(mr):.3f} "
           f"(bound {MED_PASS}); largest per-prompt ratio {min(xr):.3f} to {max(xr):.3f} (bound {MAX_PASS}); "
           f"largest |q'| {fmt(s['q_max'])} (bound {fmt(Q_PASS)}); FP16 values not finite "
           f"{sum(v['nonfinite_fp16'] for v in L.values())}; saturated values compact "
           f"{sum(v['sat_cmp_count'] for v in L.values())} against standard "
           f"{sum(v['sat_std_count'] for v in L.values())}; largest engine gap {c.get('_engine_gap_max') or 0:.4f}")
    fail = [f"layer {layer}: median per-prompt ratio {v['med_ratio']:.2f} above {MED_FAIL}" for layer, v in L.items()
            if v["med_ratio"] > MED_FAIL]
    if fail:
        return "FAIL (" + "; ".join(fail) + f"; action: {ACTION['FAIL']}; {tag})"
    rng, other = [], []
    for layer, v in L.items():
        p = v["pass"]
        if not p["q"]:
            over = v["q_max"] > Q_RANGE
            rng.append(f"layer {layer}: largest |q'| {fmt(v['q_max'])} above {fmt(Q_RANGE if over else Q_PASS)} "
                       f"({'the FP16 range' if over else 'the 10 times FP16 margin'})")
        if not p["fp16"]:
            rng.append(f"layer {layer}: {v['nonfinite_fp16']} FP16 values not finite")
        if not p["median"]:
            other.append(f"layer {layer}: median per-prompt ratio {v['med_ratio_used']:.3f} above {MED_PASS}")
        if not p["largest"]:
            other.append(f"layer {layer}: largest per-prompt ratio {v['max_ratio_used']:.3f} above {MAX_PASS}")
        if not p["sat"]:
            other.append(f"layer {layer}: {v['sat_cmp_count']} saturated compact values against "
                         f"{v['sat_std_count']} standard")
    if not rng and not other:
        return f"PASS ({tag}; action: {ACTION['PASS']})"
    act = ACTION["range"] if not other else ACTION["other"]
    return "INCONCLUSIVE (" + "; ".join(rng + other) + f"; action: {act}; {tag})"


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
