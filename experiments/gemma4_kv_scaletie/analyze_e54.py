"""E-54 analysis (README Sections 5 to 7; I-182; D-90 option d after R-74): on fresh real Gemma 4 31B activations,
does the scale-tied compact FP8 row keep the attention accuracy of the standard FP8 cache?

The tied row is the E-52 views row [K_rot | V_unrot | V_rot] stored as the FP8 codes of [K_rot / ks | V_unrot / vs |
V_rot / vs], with ks = w x vs in each global layer (w is the k_norm weight: one BF16 value per layer, R-74 C9). The
stock FA4 kernel reads codes 0 to 511 as K with k_descale ks and the plain permuted query, and codes 128 to 639 as V with
v_descale vs. Since K_unrot = w x V_unrot, ks x code(V_unrot / vs) = w x V_unrot = K_unrot up to the FP8 and BF16
rounding (README Section 2). `calibrate_e54.py` froze the scales in `scales_e54.json` from the E-53 captures before any
E-54 data existed.

Arms, per test prompt and global layer (float32 causal GQA attention, scale 1.0, no soft-cap, as E-53):
  STOCK      the standard FP8 cache with ks = vs = 1.0 (vLLM's choice for this checkpoint: `kv_cache_scheme` null)
  CALSTD     the standard FP8 cache with the frozen calibrated ks and vs of the layer
  TIED       the scale-tied compact row with the frozen vs of the layer and ks = w x vs
  TIEDSTD    the standard FP8 cache with the TIED scales (the algebra control, V7)
  TIED_E5M2  the TIED row with E5M2 codes (the sensitivity control, V6)
  C0         the E-53 compact row at scale 1.0 with the BF16 views query (the E-53 replication; reported)
  BF16C      the compact row in BF16 with the BF16 views query (D-90 option b; reported)
STOCK, CALSTD and TIED also run in the FP16 emulation of the pinned FA4 kernel (q and P in FP16; README Section 10).

Two steps, as E-53: (1) metrics.json from the capture files (on the Pod GPU, or on the Mac from the copied captures);
(2) the validity checks V1 to V7 and the frozen rule (README Sections 6 and 7) from metrics.json, with the kernel arm
(`kernel_e54.py`, kernel.json) as a secondary line.

Usage: analyze_e54.py --res <dir> [--capture <dir>] [--scales <json>] [--device auto|cpu|cuda] [--out summary.json]
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
sys.path.insert(0, str(HERE.parent / "gemma4_kv_accuracy"))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import analyze_e53 as an  # noqa: E402  (fp8, e5m2, fp16, bf16, attend, rel, head_err, w_choice, identity_err)
import compact_attn as ca  # noqa: E402  (ROT, UNROT, pack_views, views_of, views_query, unpermute)

D = ca.HEAD
NROT = len(ca.ROT)
SCALES = HERE / "scales_e54.json"
SCALES_SHA256 = "202738b9fe8b18cd594aca27b744984ecdcf6c4cfa3044660446e33a9914fdb1"   # README Section 4.6; V2 compares
PROMPTS = (  # the frozen test prompts (README Section 4.4): id, kind, family, tokens
    (0, "plain", "vllm_py", 8192), (1, "plain", "torch_py", 8192), (2, "plain", "docs", 8192),
    (3, "plain", "mixed", 8192), (4, "chat", "mixed", 8192), (5, "chat", "mixed", 8192),
    (6, "plain", "mixed", 32768))
PLAN = {"prompts": [{"id": i, "kind": k, "family": f, "tokens": t} for i, k, f, t in PROMPTS], "q_rows": 256,
        "seed": 54, "heads": 32, "kv_heads": 4, "head_dim": 512, "global_layers": 10}
CLASSES = {"plain8k": (0, 1, 2, 3), "chat": (4, 5), "long": (6,)}   # reported per class
ENGINE_TOL = 0.005                     # V5 (R-74 C8): the reference against the engine output
E5M2_MIN = 1.5                         # V6: P(TIED_E5M2) / P(TIED) at least 1.5 (one mantissa bit less)
TIE_LO, TIE_HI = 0.90, 1.10            # V7: P(TIED) / P(TIEDSTD) inside this band (the same scales)
POOL_PASS = 1.25                       # PASS: R1 and R2 at most 1.25 (float32 and FP16) (R-74 M2)
POOL_FAIL = 1.5                        # FAIL: R1 or R2 above 1.5 (float32)
LAYER_PASS = 2.0                       # PASS: every per-layer median ratio at most 2 (R-74 M2)
PROMPT_PASS = 1.5                      # PASS: every test prompt's pooled ratio at most 1.5 (coverage, R-74 R2)
ARMS = ("STOCK", "CALSTD", "TIED", "TIEDSTD", "TIED_E5M2", "C0", "BF16C")
ARMS16 = ("STOCK", "CALSTD", "TIED")
ACTION = {  # the consequence of each outcome (README Section 7); the user confirms each step
    "PASS": "the FP8 compact build continues with the tied scales (D-88 week 2); E-55 (the model-level splice) next",
    "FAIL": "stop the FP8 compact design; D-90 falls back to (b) the BF16 compact build or (a) the write-up",
    "other": "the user decides; the report names the conditions; no rerun without a new S4",
    "fail_fast": "fix the cause, then one rerun under a new S4 with a stated cost cap",
    "V5": "the reference does not model the engine: fix the reference and analyze the copied captures again on the "
          "Mac; no new Pod",
    "invalid": "fix the cause; a rerun needs a new S4",
}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(p: Path, default):
    return json.loads(p.read_text()) if p.exists() else default


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- the storages


def f32(x: float) -> float:
    """x rounded to float32: the engine holds the KV scales as float32 tensors."""
    return float(torch.tensor(x, dtype=torch.float32))


def q8(x: torch.Tensor, s: float, cast=an.fp8) -> torch.Tensor:
    """The cache cast with a per-tensor scale s: cast(x / s), read back times s (vLLM divides by the scale before the
    saturating cast; the kernel multiplies by the descale)."""
    return cast(x / s) * s


def std_arm(q, k, v, qpos, ks: float, vs: float, cast=an.fp8, fp16: bool = False) -> torch.Tensor:
    """The standard cache: K and V each cast with its own scale. fp16: q and P in FP16 (the pinned FA4 kernel)."""
    if fp16:
        return an.attend(an.fp16(q), q8(k, ks, cast), q8(v, vs, cast), qpos, torch.float16)
    return an.attend(q, q8(k, ks, cast), q8(v, vs, cast), qpos)


def tied_codes(k, v, ks: float, vs: float, cast=an.fp8) -> torch.Tensor:
    """The 640 codes of the tied row: cast([K_rot / ks | V_unrot / vs | V_rot / vs])."""
    s = torch.full((NROT + D,), vs, device=k.device)
    s[:NROT] = ks
    return cast(ca.pack_views(k, v) / s)


def tied_arm(q, k, v, qpos, ks: float, vs: float, cast=an.fp8, fp16: bool = False, v_exact: bool = False):
    """The tied row as the stock kernel reads it: the K view (codes 0 to 511) with k_descale ks and the plain permuted
    query [q_rot | q_unrot], the V view (codes 128 to 639) with v_descale vs; the output in the natural dim order.
    v_exact: the V view without the cast (the score path alone, reported)."""
    kc, vc = ca.views_of(tied_codes(k, v, ks, vs, cast))
    vv = ca.views_of(ca.pack_views(k, v))[1] if v_exact else vc * vs
    qp = ca.views_query(q, torch.ones(D, device=q.device))
    if fp16:
        return ca.unpermute(an.attend(an.fp16(qp), kc * ks, vv, qpos, torch.float16))
    return ca.unpermute(an.attend(qp, kc * ks, vv, qpos))


def e53_compact_arm(q, k, v, qpos, w: torch.Tensor, cast=an.fp8) -> torch.Tensor:
    """The E-53 compact row at scale 1.0, read with the BF16 views query [q_rot | (q w)_unrot]; cast None: the BF16
    row (D-90 option b)."""
    row = ca.pack_views(k, v)
    kv, vv = ca.views_of(row if cast is None else cast(row))
    return ca.unpermute(an.attend(an.bf16(ca.views_query(q, w)), kv, vv, qpos))


def sat_count(x: torch.Tensor, s: float) -> int:
    return int((x.abs() / s > an.FP8_MAX).sum())


def layer_metrics(cap: dict, w_mod: torch.Tensor | None, sc: dict, device) -> dict:
    """Every metric of one test prompt and global layer (README Section 5). sc: the scales of the layer."""
    q, k, v = (cap[n].to(device).float() for n in ("q", "k", "v"))
    qpos = cap["qpos"].to(device).long()
    kpos = cap["kpos"].to(device).long() if "kpos" in cap else torch.arange(k.shape[0], device=device)
    names = [n for n in ("q", "k", "v", "o") if n in cap]
    out = {"shapes": {n: list(cap[n].shape) for n in names},
           "dtypes": {n: str(cap[n].dtype).replace("torch.", "") for n in names},
           "kpos_ok": bool(torch.equal(kpos, torch.arange(k.shape[0], device=device))),
           "qpos_first": int(qpos.min()), "qpos_last": int(qpos.max()), "w_module": w_mod is not None}
    form, w, dist = an.w_choice(an.w_from_data(k, v), w_mod)
    wu = w[torch.tensor(ca.UNROT, device=device)]                               # the tie needs one value here (V3)
    out |= {"w_form": form, "w_match": bool(dist) and min(dist.values()) <= an.W_MATCH,
            "w_scalar": int(torch.unique(wu).numel()) == 1, "w_value": float(wu[0]),
            "identity_err": an.identity_err(k, v, w), "identity_token_max": an.identity_token_max(k, v, w)}
    ref = an.attend(q, k, v, qpos)
    out["ref_finite"] = bool(torch.isfinite(ref).all())
    o_eng = cap["o"].to(device).float() if "o" in cap else None
    out["engine_gap"] = None if o_eng is None else (an.rel(ref, o_eng) if o_eng.shape == ref.shape else math.inf)
    ks_s, vs_s = sc["calstd"]["ks"], sc["calstd"]["vs"]
    ks_t, vs_t = sc["tied"]["ks"], sc["tied"]["vs"]
    o = {"STOCK": std_arm(q, k, v, qpos, 1.0, 1.0), "CALSTD": std_arm(q, k, v, qpos, ks_s, vs_s),
         "TIED": tied_arm(q, k, v, qpos, ks_t, vs_t), "TIEDSTD": std_arm(q, k, v, qpos, ks_t, vs_t),
         "TIED_E5M2": tied_arm(q, k, v, qpos, ks_t, vs_t, cast=an.e5m2), "C0": e53_compact_arm(q, k, v, qpos, w),
         "BF16C": e53_compact_arm(q, k, v, qpos, w, cast=None)}
    o16 = {"STOCK": std_arm(q, k, v, qpos, 1.0, 1.0, fp16=True),
           "CALSTD": std_arm(q, k, v, qpos, ks_s, vs_s, fp16=True),
           "TIED": tied_arm(q, k, v, qpos, ks_t, vs_t, fp16=True)}
    out["err"] = {a: an.rel(x, ref) for a, x in o.items()}
    out["err16"] = {a: an.rel(x, ref) for a, x in o16.items()}
    out["nonfinite_fp16"] = sum(int((~torch.isfinite(x)).sum()) for x in o16.values())
    out["head_err_max"] = {a: an.head_err(o[a], ref) for a in ARMS16}
    # reported: the score path alone (V exact) of TIED and CALSTD; the saturated values of each FP8 arm
    out["err_score"] = {"CALSTD": an.rel(an.attend(q, q8(k, ks_s), v, qpos), ref),
                        "TIED": an.rel(tied_arm(q, k, v, qpos, ks_t, vs_t, v_exact=True), ref)}
    row = ca.pack_views(k, v)
    out["sat"] = {"STOCK": sat_count(k, 1.0) + sat_count(v, 1.0), "CALSTD": sat_count(k, ks_s) + sat_count(v, vs_s),
                  "TIED": sat_count(row[..., :NROT], ks_t) + sat_count(row[..., NROT:], vs_t)}
    out["q_max"] = float(q.abs().max())
    return out


# ---------------------------------------------------------------- step 1: metrics


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def data_scales(cap_dir: Path, layers: list[int], knorm: dict) -> dict:
    """The scales of a stub or a rehearsal (not the frozen file): the base scales of the calibration rule (the smallest
    power of 2 at or above 2 x max / 448 over the prompts of the run), phase 0, and ks = w x vs for the tied row (README
    Section 4.6)."""
    out = {}
    for layer in layers:
        mk = mv = 0.0
        for f in sorted(cap_dir.glob(f"p*/L{layer:02d}.pt")):
            c = torch.load(f, map_location="cpu", weights_only=True)
            mk, mv = max(mk, float(c["k"].float().abs().max())), max(mv, float(c["v"].float().abs().max()))
        wm = knorm.get(layer)
        w = float(an.bf16(wm.float())[ca.UNROT[0]]) if wm is not None else 1.0
        bk, bv = (2.0 ** math.ceil(math.log2(max(2 * m / an.FP8_MAX, 2.0 ** -40))) for m in (mk, mv))
        out[str(layer)] = {"w": w, "calstd": {"ks": bk, "vs": bv}, "tied": {"ks": f32(w * bv), "vs": bv}}
    return out


def compute_metrics(cap_dir: Path, res: Path, device: str, scales_path: Path | None) -> dict:
    """Step 1: metrics.json from <cap_dir>/p<i>/L<layer>.pt and <cap_dir>/k_norm.pt, with the SHA-256 of each file
    against capture_run.json (as E-53). The file is written again after each prompt, so a timeout keeps the finished
    prompts."""
    torch.backends.cuda.matmul.allow_tf32 = False                               # float32 means float32
    torch.backends.cudnn.allow_tf32 = False
    run = load(res / "capture_run.json", {})
    want = {(str(e["id"]), str(layer)): f.get("sha256") for e in run.get("prompts", [])
            for layer, f in (e.get("files") or {}).items()}
    out = {"started_at": now(), "device": device, "torch": torch.__version__, "status": "started", "prompts": {},
           "errors": [], "sha_checked": 0, "sha_mismatch": []}
    if device == "cuda":
        out["gpu"] = torch.cuda.get_device_name(0)
    kn_path = cap_dir / "k_norm.pt"
    knorm = torch.load(kn_path, map_location="cpu", weights_only=True) if kn_path.exists() else {}
    out["k_norm_layers"] = sorted(int(x) for x in knorm)
    if run.get("stub") or run.get("rehearsal"):
        layers = sorted({int(f.stem[1:]) for f in cap_dir.glob("p*/L*.pt")})
        scales, out["scales_source"] = data_scales(cap_dir, layers, knorm), "data (stub or rehearsal)"
    else:
        sp = scales_path or SCALES
        scales, out["scales_source"] = json.loads(sp.read_text())["layers"], str(sp.name)
        out["scales_sha256"] = sha_file(sp)
    out["scales"] = scales
    path = res / "metrics.json"
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
                out["prompts"].setdefault(pid, {})[str(layer)] = layer_metrics(cap, knorm.get(layer),
                                                                               scales[str(layer)], device)
            except Exception as exc:  # recorded; V2 then fails for this layer
                out["errors"].append(f"p{pid} L{layer}: {type(exc).__name__}: {exc}"[:300])
        out["seconds"] = round(time.time() - t0, 1)
        path.write_text(json.dumps(out, indent=1))
        print(f"metrics: prompt {pid} done after {out['seconds']} s", flush=True)
    out.update(status="ok", finished_at=now(), seconds=round(time.time() - t0, 1))
    path.write_text(json.dumps(out, indent=1))
    return out


# ---------------------------------------------------------------- step 2: checks and rule


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs and all(math.isfinite(x) for x in xs) else math.nan


def rms(xs: list[float]) -> float:
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs and all(math.isfinite(x) for x in xs) else math.nan


def ratio(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.nan
    if b > 0:
        return a / b
    return 1.0 if a == 0 else math.inf


def pooled(rows: dict[str, list[dict]], key: str, arm: str) -> float:
    """P_X: the RMS over the layers of the median over the prompts of the error of arm X (README Section 5)."""
    return rms([med([r[key][arm] for r in rs]) for rs in rows.values()])


def prompt_pooled(rows: dict[str, list[dict]], pids: list[str], arm: str, base: str) -> dict[str, float]:
    """Per test prompt: sqrt(sum over layers of err_X^2) / sqrt(sum over layers of err_base^2)."""
    out = {}
    for i, p in enumerate(pids):
        a = [rs[i]["err"][arm] for rs in rows.values()]
        b = [rs[i]["err"][base] for rs in rows.values()]
        out[p] = ratio(rms(a), rms(b))
    return out


def stats(rows: dict[str, list[dict]], pids: list[str]) -> dict:
    """The decision statistics and the reported numbers (README Sections 5 and 7)."""
    P = {a: pooled(rows, "err", a) for a in ARMS}
    P16 = {a: pooled(rows, "err16", a) for a in ARMS16}
    g = lambda arm, base: {layer: med([ratio(r["err"][arm], r["err"][base]) for r in rs])   # noqa: E731
                           for layer, rs in rows.items()}
    out = {"P": P, "P16": P16,
           "R1": ratio(P["TIED"], P["STOCK"]), "R2": ratio(P["TIED"], P["CALSTD"]),
           "R1_fp16": ratio(P16["TIED"], P16["STOCK"]), "R2_fp16": ratio(P16["TIED"], P16["CALSTD"]),
           "g1": g("TIED", "STOCK"), "g2": g("TIED", "CALSTD"),
           "R1_prompt": prompt_pooled(rows, pids, "TIED", "STOCK"),
           "R2_prompt": prompt_pooled(rows, pids, "TIED", "CALSTD"),
           "e5m2": ratio(P["TIED_E5M2"], P["TIED"]), "tie": ratio(P["TIED"], P["TIEDSTD"]),
           # reported only
           "calstd_vs_stock": ratio(P["CALSTD"], P["STOCK"]), "c0_vs_stock": ratio(P["C0"], P["STOCK"]),
           "g_c0": g("C0", "STOCK"), "bf16c_median": {layer: med([r["err"]["BF16C"] for r in rs])
                                                     for layer, rs in rows.items()},
           "per_layer_median": {a: {layer: med([r["err"][a] for r in rs]) for layer, rs in rows.items()}
                                for a in ARMS},
           "sat": {a: sum(r["sat"][a] for rs in rows.values() for r in rs) for a in ARMS16},
           "q_max": max(r["q_max"] for rs in rows.values() for r in rs),
           "nonfinite_fp16": sum(r["nonfinite_fp16"] for rs in rows.values() for r in rs)}
    cls = {}
    for name, ids in CLASSES.items():
        idx = [i for i, p in enumerate(pids) if int(p) in ids]
        if idx:
            sub = {layer: [rs[i] for i in idx] for layer, rs in rows.items()}
            cls[name] = {"R1": ratio(pooled(sub, "err", "TIED"), pooled(sub, "err", "STOCK")),
                         "R2": ratio(pooled(sub, "err", "TIED"), pooled(sub, "err", "CALSTD"))}
    out["classes"] = cls
    return out


def e53_overlap(prompts: dict, e53: dict) -> list[str]:
    paths = {f["path"] for f in e53.get("files", [])}
    shas = {f["sha256"] for f in e53.get("files", [])}
    return [f"prompt {p['id']}: {f['path']}" for p in prompts.get("prompts", []) for f in p.get("files", [])
            if f["path"] in paths or f["sha256"] in shas]


def v2_problems(run: dict, met: dict, prompts: dict, e53: dict) -> list[str]:
    """V2: the capture finished with the frozen plan (a stub or the S5 rehearsal may use its own plan), the hooked
    layers are the planned global layers, every prompt and layer has metrics with the planned shapes, no capture
    prompt had a problem, the SHA-256 of every capture file matches, the engine is the planned one, the test prompts
    take no E-53 file and no file twice, a plain prompt has no special id after position 0, a chat prompt starts with
    BOS and has 2 or more special ids after it, and the analysis used the frozen scales file."""
    bad = []
    if not run:
        return ["no capture record (capture_run.json)"]
    if run.get("status") != "ok":
        bad.append(f"the capture status is {run.get('status')!r}: {run.get('error') or run.get('fail_fast') or ''}"[:300])
    plan = run.get("plan") or {}
    real = not (run.get("stub") or run.get("rehearsal"))
    if real and plan != PLAN:
        bad.append(f"the plan {plan} is not the frozen plan")
    expected = (run.get("config") or {}).get("global_layers") or []
    hooked = sorted(h["layer"] for h in (run.get("hooks") or {}).get("hooked", []))
    if not expected:
        bad.append("the config names no global layer")
    elif hooked != sorted(expected):
        bad.append(f"hooked layers {hooked} are not the config global layers {sorted(expected)}")
    if expected and len(expected) != plan.get("global_layers"):
        bad.append(f"the config has {len(expected)} global layers, the plan {plan.get('global_layers')}")
    if not run.get("stub"):
        eng = run.get("engine") or {}
        if run.get("vllm") != an.VLLM_VERSION:
            bad.append(f"vLLM {run.get('vllm')!r}, expected {an.VLLM_VERSION}")
        for key in ("compilation_mode", "cudagraph_mode"):
            if eng.get(key) != an.EAGER:
                bad.append(f"engine {key} {eng.get(key)!r}, expected {an.EAGER}")
        if str(eng.get("cache_dtype")) not in an.BF16_CACHE:
            bad.append(f"engine cache dtype {eng.get('cache_dtype')!r}, expected a BF16 cache")
    for e in run.get("prompts", []):
        if e.get("problems"):
            bad.append(f"capture prompt {e['id']}: {'; '.join(e['problems'])[:200]}")
    if met.get("sha_mismatch"):
        bad.append(f"capture files with another SHA-256: {met['sha_mismatch'][:10]}")
    if met.get("errors"):
        bad.append(f"metrics errors: {met['errors'][:5]}")
    bad += [f"an E-53 file in a test prompt: {x}" for x in e53_overlap(prompts, e53)[:10]]
    names = [f["path"] for p in prompts.get("prompts", []) for f in p.get("files", [])]
    if len(names) != len(set(names)):
        bad.append("a file is used in more than one test prompt")
    if real and SCALES_SHA256 is not None and met.get("scales_sha256") != SCALES_SHA256:
        bad.append(f"the scales file SHA-256 {met.get('scales_sha256')} is not the frozen {SCALES_SHA256}")
    built = {str(p["id"]): p for p in prompts.get("prompts", [])}
    q_rows = plan.get("q_rows")
    hq, hk, hd = plan.get("heads"), plan.get("kv_heads"), plan.get("head_dim")
    for spec in plan.get("prompts") or []:
        pid = str(spec["id"])
        p = built.get(pid) or {}
        length = p.get("n_tokens")
        if length != spec["tokens"]:
            bad.append(f"prompt {pid} has {length} tokens, the plan {spec['tokens']}")
        if spec["kind"] == "chat":
            if not p.get("bos_at_0") or (p.get("special_after_0") or 0) < 2:
                bad.append(f"chat prompt {pid}: BOS at 0 {p.get('bos_at_0')}, special ids after 0 "
                           f"{p.get('special_after_0')} (2 or more expected)")
        elif p.get("special_after_0", 0):
            bad.append(f"prompt {pid} has {p['special_after_0']} special ids after position 0")
        got = (met.get("prompts") or {}).get(pid, {})
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


def kernel_line(k: dict) -> tuple[str, str]:
    """The secondary outcome of the kernel arm (README Sections 7 and 8)."""
    if not k:
        return "NOT RUN", "no kernel.json"
    gaps = k.get("max_gap") or {}
    if k.get("outcome") in ("NOT RUN", "ERROR"):
        return k["outcome"], str(k.get("reason", ""))[:200]
    return k.get("outcome", "NOT RUN"), "largest gap " + ", ".join(f"{a} {g:.4f}" for a, g in gaps.items())


def summarize(res: Path) -> dict:
    run = load(res / "capture_run.json", {})
    prompts = load(res / "prompts.json", {})
    met = load(res / "metrics.json", {})
    e53 = load(HERE / "e53_files.json", {})
    expected = (run.get("config") or {}).get("global_layers") or []
    pids = [str(s["id"]) for s in (run.get("plan") or {}).get("prompts", [])]
    got = met.get("prompts") or {}
    complete = [p for p in pids if all(str(layer) in got.get(p, {}) for layer in expected)]
    rows = {str(layer): [got[p][str(layer)] for p in complete] for layer in expected}
    flat = [m for rs in rows.values() for m in rs]
    gpu = (run.get("gpu") or {}).get("name", "")
    v2 = v2_problems(run, met, prompts, e53)
    scales = met.get("scales") or {}
    every = [(p, str(layer)) for p in pids for layer in expected if str(layer) in got.get(p, {})]
    id_bad = [(p, layer, got[p][layer]["identity_err"]) for p, layer in every
              if not got[p][layer]["identity_err"] <= an.ID_TOL]
    w_bad = [(p, layer) for p, layer in every
             if not (got[p][layer]["w_module"] and got[p][layer]["w_match"] and got[p][layer]["w_scalar"]
                     and got[p][layer]["w_value"] == (scales.get(layer) or {}).get("w"))]
    ref_bad = [(p, layer) for p, layer in every if not got[p][layer]["ref_finite"]]
    eng_bad = [(p, layer, got[p][layer]["engine_gap"]) for p, layer in every
               if not (got[p][layer]["engine_gap"] is not None and got[p][layer]["engine_gap"] <= ENGINE_TOL)]
    st = stats(rows, complete) if flat and len(complete) == len(pids) else {}
    checks = {"V1": "H100" in gpu and "80GB" in gpu.replace(" ", ""),
              "V2": not v2,
              "V3": bool(flat) and not id_bad and not w_bad,
              "V4": bool(flat) and not ref_bad,
              "V5": bool(flat) and not eng_bad,
              "V6": bool(st) and st["e5m2"] >= E5M2_MIN,
              "V7": bool(st) and TIE_LO <= st["tie"] <= TIE_HI,
              "_gpu": gpu, "_v2_problems": v2[:20], "_v2_count": len(v2),
              "_identity_max": max((m["identity_err"] for m in flat), default=None),
              "_identity_failures": id_bad[:20], "_w_failures": w_bad[:20], "_ref_nonfinite": ref_bad[:20],
              "_engine_gap_max": max((m["engine_gap"] for m in flat if m["engine_gap"] is not None), default=None),
              "_engine_failures": eng_bad[:20], "_e5m2": st.get("e5m2"), "_tie": st.get("tie"),
              "_metrics_status": met.get("status"), "_sha_checked": met.get("sha_checked"),
              "_scales_source": met.get("scales_source"), "_scales_sha256": met.get("scales_sha256")}
    ledger = {"R1": [st["R1"]] if st else [], "R2": [st["R2"]] if st else [],
              "err_TIED": [m["err"]["TIED"] for m in flat], "err_STOCK": [m["err"]["STOCK"] for m in flat],
              "err_CALSTD": [m["err"]["CALSTD"] for m in flat]}
    return {"capture_status": run.get("status", "missing"), "fail_fast": run.get("fail_fast"), "stub": run.get("stub"),
            "rehearsal": run.get("rehearsal"), "plan": run.get("plan"), "gpu": run.get("gpu"),
            "versions": {k: run.get(k) for k in ("vllm", "torch")}, "engine": run.get("engine"),
            "config": run.get("config"),
            "prompts": [{k: p.get(k) for k in ("id", "kind", "family", "n_tokens", "token_sha256", "special_after_0",
                                               "bos_at_0", "skipped_special", "excluded")}
                        for p in prompts.get("prompts", [])],
            "metrics_device": met.get("device"), "metrics_gpu": met.get("gpu"), "metrics_seconds": met.get("seconds"),
            "checks": checks, "stats": st, "kernel": load(res / "kernel.json", {}),
            "ledger_metrics": {k: {"values": [x for x in v if math.isfinite(x)],
                                   "unit": "relative L2 error" if k.startswith("err") else "pooled error ratio",
                                   "higher_is_better": False} for k, v in ledger.items()}}


def decide(s: dict) -> str:
    """The frozen rule (README Sections 6 and 7), with the action of each outcome and the kernel arm line."""
    word = verdict(s)
    k_out, k_why = kernel_line(s.get("kernel") or {})
    text = f"{word}; kernel arm (secondary): {k_out} ({k_why})"
    if s.get("stub"):
        return "STUB " + text
    return f"REHEARSAL {text}" if s.get("rehearsal") else text


def verdict(s: dict) -> str:
    c = s["checks"]
    if s.get("capture_status") == "fail_fast":
        return (f"INVALID (V2: fail fast after the first prompt: {'; '.join(s.get('fail_fast') or [])[:400]}; "
                f"action: {ACTION['fail_fast']})")
    bad = [k for k in ("V1", "V2", "V3", "V4", "V5", "V6", "V7") if not c.get(k)]
    if bad:
        why = {"V1": f"GPU {c.get('_gpu')!r}", "V2": "; ".join(c.get("_v2_problems") or [])[:300],
               "V3": f"largest identity error {c.get('_identity_max')}, layers without a matching scalar module w "
                     f"{c.get('_w_failures')}",
               "V4": f"reference not finite in {c.get('_ref_nonfinite')}",
               "V5": f"engine gap above {ENGINE_TOL} in {c.get('_engine_failures')}",
               "V6": f"E5M2 sensitivity {c.get('_e5m2')} below {E5M2_MIN}",
               "V7": f"tie control {c.get('_tie')} outside {TIE_LO} to {TIE_HI}"}
        act = ACTION["V5"] if bad == ["V5"] else ACTION["invalid"]
        return f"INVALID ({', '.join(bad)}: " + "; ".join(f"{k} {why[k]}" for k in bad) + f"; action: {act})"
    st = s["stats"]
    tag = (f"R1 {st['R1']:.3f} and R2 {st['R2']:.3f} (FP16 {st['R1_fp16']:.3f} and {st['R2_fp16']:.3f}); "
           f"P STOCK {st['P']['STOCK']:.4f}, CALSTD {st['P']['CALSTD']:.4f}, TIED {st['P']['TIED']:.4f}; per-layer "
           f"TIED/STOCK {min(st['g1'].values()):.3f} to {max(st['g1'].values()):.3f}; E5M2 {st['e5m2']:.2f}; tie "
           f"{st['tie']:.3f}; C0/STOCK {st['c0_vs_stock']:.3f}; largest |q| {st['q_max']:.1f}; largest engine gap "
           f"{c.get('_engine_gap_max') or 0:.4f}")
    fail = [f"{n} {st[n]:.3f} above {POOL_FAIL}" for n in ("R1", "R2") if not st[n] <= POOL_FAIL]
    if fail:
        return "FAIL (" + "; ".join(fail) + f"; action: {ACTION['FAIL']}; {tag})"
    other = [f"{n} {st[n]:.3f} above {POOL_PASS}" for n in ("R1", "R2", "R1_fp16", "R2_fp16")
             if not st[n] <= POOL_PASS]
    for name, key in (("TIED/STOCK", "g1"), ("TIED/CALSTD", "g2")):
        other += [f"layer {layer}: median {name} {x:.3f} above {LAYER_PASS}" for layer, x in st[key].items()
                  if not x <= LAYER_PASS]
    for name, key in (("R1", "R1_prompt"), ("R2", "R2_prompt")):
        other += [f"prompt {p}: {name} {x:.3f} above {PROMPT_PASS}" for p, x in st[key].items()
                  if not x <= PROMPT_PASS]
    if st["nonfinite_fp16"]:
        other.append(f"{st['nonfinite_fp16']} FP16 values not finite")
    if not other:
        return f"PASS ({tag}; action: {ACTION['PASS']})"
    return "INCONCLUSIVE (" + "; ".join(other) + f"; action: {ACTION['other']}; {tag})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, type=Path, help="results folder (capture_run.json, prompts.json)")
    ap.add_argument("--capture", type=Path, help="capture folder: compute metrics.json from it first")
    ap.add_argument("--scales", type=Path, help=f"the frozen scales file (default {SCALES.name})")
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.capture:
        compute_metrics(a.capture, a.res, pick_device(a.device), a.scales)
    s = summarize(a.res)
    s["decision"] = decide(s)
    if a.out:
        a.out.write_text(json.dumps(s, indent=1, default=str))
    print(s["decision"])


if __name__ == "__main__":
    main()
