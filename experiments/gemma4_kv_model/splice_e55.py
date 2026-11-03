"""E-55 splice (README Sections 4.4 and 4.5, amendment A1): the arm transforms of the global-layer attention, and the
vLLM hooks that apply them.

How the splice reaches the cache (FACT, vLLM 3966924d5; E-53 README Section 4.3; R-77 answer 1): `Gemma4Attention.
forward` (models/gemma4.py 540 to 575) applies k_norm, RoPE and the weightless v_norm, then calls `self.attn(q, k, v)`.
A forward pre-hook on that `Attention` module replaces q, k and v before the module writes k and v to the paged cache.
The engine reads K and V from the paged cache in prefill and in decode (R-76, `flash_attn.py`), so every read sees the
values of the arm. The cache stays BF16 ("auto").

The FP8 arms store the FP8 codes themselves (each E4M3 or E5M2 value is exact in BF16) and put the two descales where
the FP8 kernel applies them (R-74; E-54 README Section 4.7): the K descale multiplies the scores (the Attention impl
`scale`, FlashAttentionImpl `softmax_scale`, flash_attn.py 1077 and 1413 to 1698; Gemma 4 uses 1.0), and the V descale
multiplies the output (a forward hook). So the scores are ks (q . c) and the output is vs (P . c), as in the FP8
kernel, with no BF16 rounding of scale x code. (A first form stored bf16(scale x code); on the E-54 captures it moved
the TIED and JOINT errors by 2% to 6%, because the rounding acts like a small scale change on each code value.)

Arms (the global layers only; the 50 sliding layers keep the BF16 cache in every arm):
  BF16      identity (the reference; the hooks only count)
  BF16C     the compact row in BF16 as the stock kernel reads its two views: K_unrot = V_unrot, q_unrot = bf16(q w)
            (E-54 BF16C, E-52 views): the exact compact numerics, in the natural dim order
  E5M2      K and V cast to FP8 E5M2 at scale 1.0 (the positive control, V6)
  STOCK8    K and V cast to FP8 E4M3 at scale 1.0 (the stock vLLM FP8 cache)
  TIED8     the tied row with the frozen E-54 scales: K codes [c(K_rot / ks) | c(V_unrot / vs)] with the K descale ks,
            V codes c(V / vs) with the V descale vs, ks = f32(w vs): the two views of E-54
  BF16R     q' = bf16(q R_C) with the impl scale 1 / R_C: one more BF16 rounding of the query and no other change. Its
            effect is the effect of a rounding of the size of the BF16C rounding: the A/A of BF16C (A1, R-77 F7)
  JOINT8    the standard cache with the frozen E-54 CALSTD_JOINT scales: c(K / ks) and c(V / vs) with ks and vs
  TIED8_UP  TIED8 with vs one phase step up (vs 2^(1/32), ks = f32(w vs)): the scale control (reported)
c() is the vLLM cache cast: saturate at the format maximum, round to nearest even.
"""

from __future__ import annotations

import torch

FP8_MAX, E5M2_MAX = 448.0, 57344.0
STEP = 2.0 ** (1 / 32)                       # one phase step of the E-54 calibration grid
R_C = 1.0 + 2.0 ** -7                        # BF16R: the query factor (the impl scale takes 1 / R_C)
ARMS = ("BF16", "BF16C", "E5M2", "STOCK8", "TIED8", "BF16R", "JOINT8", "TIED8_UP")
SCALED = ("TIED8", "JOINT8", "TIED8_UP")     # the arms with FP8 descales (a V descale on the output)


def rot_dims(head: int, rotary: float = 0.25) -> tuple[list[int], list[int]]:
    """The rotated and unrotated dims of Gemma 4 proportional RoPE (neox halves): for head 512 the rotated dims are
    0..63 and 256..319 (E-49; `compact_attn.ROT`)."""
    half = int(head * rotary) // 2
    rot = list(range(half)) + list(range(head // 2, head // 2 + half))
    rs = set(rot)
    return rot, [d for d in range(head) if d not in rs]


def f32(x: float) -> float:
    return float(torch.tensor(x, dtype=torch.float32))


def fp8(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def e5m2(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(-E5M2_MAX, E5M2_MAX).to(torch.float8_e5m2).float()


def arm_scales(arm: str, sc: dict | None) -> dict:
    """The score factor ks (the factor of the impl scale: the K descale, or 1 / R_C for BF16R) and the output factor
    vs (the V descale) of one layer for an arm, from the frozen `scales_e54.json` entry sc (w, tied, calstd_joint)."""
    if arm in ("TIED8", "TIED8_UP"):
        vs = sc["tied"]["vs"] if arm == "TIED8" else f32(sc["tied"]["vs"] * STEP)
        ks = sc["tied"]["ks"] if arm == "TIED8" else f32(sc["w"] * vs)
        return {"ks": ks, "vs": vs}
    if arm == "JOINT8":
        return {"ks": sc["calstd_joint"]["ks"], "vs": sc["calstd_joint"]["vs"]}
    if arm == "BF16R":
        return {"ks": f32(1.0 / R_C), "vs": 1.0}
    return {"ks": 1.0, "vs": 1.0}


def transform(arm: str, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, w: float, scales: dict,
              rot: list[int], unrot: list[int]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The arm's q, k and v as the cache and the kernel get them: q [T, Hq, D], k and v [T, Hk, D] (BF16 in the
    engine), same dtype out. For SCALED arms k and v hold codes; the caller applies ks to the scores and vs to the
    output. The arithmetic runs in float32 and is rounded once."""
    dt = k.dtype
    if arm == "BF16":
        return q, k, v
    if arm == "BF16R":
        return (q.float() * R_C).to(q.dtype), k, v
    ri = torch.tensor(rot, device=k.device)
    ui = torch.tensor(unrot, device=k.device)
    if arm == "BF16C":
        q2, k2 = q.clone(), k.clone()
        q2[..., ui] = (q[..., ui].float() * w).to(dt)
        k2[..., ui] = v[..., ui]
        return q2, k2, v
    kf, vf = k.float(), v.float()
    if arm == "E5M2":
        return q, e5m2(kf).to(dt), e5m2(vf).to(dt)
    if arm == "STOCK8":
        return q, fp8(kf).to(dt), fp8(vf).to(dt)
    ks, vs = scales["ks"], scales["vs"]
    if arm == "JOINT8":
        return q, fp8(kf / ks).to(dt), fp8(vf / vs).to(dt)
    if arm in ("TIED8", "TIED8_UP"):
        cv = fp8(vf / vs)
        ck = torch.empty_like(kf)
        ck[..., ri] = fp8(kf[..., ri] / ks)
        ck[..., ui] = cv[..., ui]                        # the K view reads the V codes (the tie)
        return q, ck.to(dt), cv.to(dt)
    raise ValueError(f"unknown arm {arm}")


def attend32(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, qpos: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Causal GQA attention in float32 of the queries at positions qpos over the keys 0 .. qpos: q [Nq, Hq, D], k and
    v [L, Hk, D] -> [Nq, Hq, D]. Query head h reads KV head h // (Hq / Hk) (as E-53 `attend`)."""
    nq, hq, length, hk = q.shape[0], q.shape[1], k.shape[0], k.shape[1]
    g = hq // hk
    kk = k.float().repeat_interleave(g, dim=1)                              # [L, Hq, D]
    vv = v.float().repeat_interleave(g, dim=1)
    s = torch.einsum("qhd,khd->hqk", q.float(), kk) * scale
    future = torch.arange(length, device=q.device)[None, :] > qpos[:, None]
    s = s.masked_fill(future[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), vv)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    n = float(torch.linalg.vector_norm(b.double()))
    return float(torch.linalg.vector_norm((a - b).double())) / n if n > 0 else 0.0


def emulate(q, k2, v2, qpos, scales: dict, arm: str, attend) -> torch.Tensor:
    """The attention output of an arm in float32 (the Mac tests): ks on the scores and vs on the output, as the hooks
    set them in the engine (both are 1 for BF16, BF16C, E5M2 and STOCK8). attend(q, k, v, qpos) is a float32 causal
    GQA attention."""
    return attend(q.float(), k2.float() * scales["ks"], v2.float(), qpos) * scales["vs"]


def check_transform(arm: str, q, k, v, q2, k2, v2, w: float, scales: dict, rot: list[int],
                    unrot: list[int]) -> list[str]:
    """V5: the transformed tensors have the arm's defining property (run once per arm and layer on the GPU)."""
    bad = []
    ri = torch.tensor(rot, device=k.device)
    ui = torch.tensor(unrot, device=k.device)
    same = lambda a, b: bool(torch.equal(a, b))  # noqa: E731
    if arm == "BF16":
        if not (q2 is q and k2 is k and v2 is v):
            bad.append("BF16: the identity changed a tensor")
        return bad
    if arm == "BF16R":
        if not same(q2, (q.float() * R_C).to(q.dtype)) or not (k2 is k and v2 is v):
            bad.append("BF16R: the query is not bf16(q R_C), or K or V changed")
        return bad
    if arm == "BF16C":
        if not same(k2[..., ui], v[..., ui]) or not same(k2[..., ri], k[..., ri]) or not same(v2, v):
            bad.append("BF16C: K_unrot is not V_unrot, or K_rot or V changed")
        if not same(q2[..., ui], (q[..., ui].float() * w).to(q.dtype)) or not same(q2[..., ri], q[..., ri]):
            bad.append("BF16C: the query is not [q_rot | bf16(q_unrot w)]")
        return bad
    if not same(q2, q):
        bad.append(f"{arm}: the query changed")
    if arm in ("STOCK8", "E5M2"):
        cast = fp8 if arm == "STOCK8" else e5m2
        if not same(k2.float(), cast(k.float())) or not same(v2.float(), cast(v.float())):
            bad.append(f"{arm}: the stored values are not the {arm} cast of the inputs")
        return bad
    ks, vs = scales["ks"], scales["vs"]
    if not same(k2.float(), fp8(k2.float())) or not same(v2.float(), fp8(v2.float())):
        bad.append(f"{arm}: a stored value is not an E4M3 code")
    if not same(v2.float(), fp8(v.float() / vs)):
        bad.append(f"{arm}: the V codes are not c(V / vs)")
    if arm == "JOINT8":
        if not same(k2.float(), fp8(k.float() / ks)):
            bad.append("JOINT8: the K codes are not c(K / ks)")
    else:
        if not same(k2.float()[..., ri], fp8(k.float()[..., ri] / ks)):
            bad.append(f"{arm}: the K codes on the rotated dims are not c(K_rot / ks)")
        if not same(k2[..., ui], v2[..., ui]):
            bad.append(f"{arm}: the K codes on the unrotated dims are not the V codes (the tie)")
    return bad


def qkv_of(args: tuple, kwargs: dict):
    """q, k, v of an `Attention.forward` call (positional, as gemma4.py 572, or by keyword) and a function that puts
    new tensors back in the same places."""
    if len(args) >= 3:
        return args[0], args[1], args[2], lambda q, k, v: ((q, k, v) + tuple(args[3:]), kwargs)
    names = ("query", "key", "value")
    vals = [args[i] if i < len(args) else kwargs.get(n) for i, n in enumerate(names)]

    def put(q, k, v):
        new, rest = dict(kwargs), list(args)
        for i, (n, x) in enumerate(zip(names, (q, k, v))):
            if i < len(rest):
                rest[i] = x
            else:
                new[n] = x
        return tuple(rest), new
    return vals[0], vals[1], vals[2], put


GAP_ROWS = 16                                # the query rows of the engine-output check (V5, V7)


class Splice:
    """The hooks of the global layers. `set_arm` selects the transform and sets the score factor on each Attention
    impl; `counts` records the tokens that each layer saw per document; `checks` holds the transform results (V5: arm
    -> problems). For the first chunk of the first document of each arm (16 query rows): `gaps` holds the relative gap
    between the engine's attention output and a float32 reference of the arm (V5: a kernel that ignored the K descale
    gives a gap near 1), `gaps_id` the gap to the reference of the unchanged q, k and v, and `seps` the gap between the
    two references (V7: the engine output must be nearer the arm's reference, A1, R-77 F8). `reset_evidence` starts
    the evidence again; the run calls it after the warm-up and after the pre-flight (R-77 B1)."""

    def __init__(self, w: dict[int, float], scales: dict[int, dict] | None, heads: dict[int, tuple[int, int, int]]):
        self.w, self.scales, self.heads = w, scales, heads
        self.arm, self.doc = "BF16", None
        self.modules: dict[int, object] = {}
        self.handles: dict[int, tuple] = {}
        self.base_scale: dict[int, float] = {}
        self.fresh: set[int] = set()
        self.reset_evidence()

    def reset_evidence(self) -> None:
        self.counts: dict[tuple[str, str, int], int] = {}
        self.checks: dict[str, list[str]] = {}
        self.checked: set[tuple[str, int]] = set()
        self.gaps: dict[str, dict[int, float]] = {}
        self.gaps_id: dict[str, dict[int, float]] = {}
        self.seps: dict[str, dict[int, float]] = {}
        self.ref: dict[int, tuple] = {}
        self.errors: dict[str, list[str]] = {}

    def evidence(self) -> dict:
        """The evidence with the keys of the run record (JSON: string keys)."""
        js = lambda d: {a: {str(k): x for k, x in g.items()} for a, g in d.items()}  # noqa: E731
        return {"checks": self.checks, "counts": {f"{a}|{d}|{lay}": n for (a, d, lay), n in self.counts.items()},
                "gaps": js(self.gaps), "gaps_id": js(self.gaps_id), "seps": js(self.seps),
                "hook_errors": self.errors}

    def new_doc(self, doc: str) -> None:
        """Call before each document: the next call of each layer is the document's first chunk (positions 0 ...)."""
        self.doc = doc
        self.fresh = set(self.modules)

    def layer_scales(self, layer: int, arm: str | None = None) -> dict:
        return arm_scales(arm or self.arm, self.scales[layer] if self.scales else None)

    def attach(self, layer: int, module) -> None:
        """Register the pre-hook and the output hook on one global-layer Attention module."""
        self.modules[layer] = module
        impl = getattr(module, "impl", None)
        self.base_scale[layer] = float(getattr(impl, "scale", 1.0))
        self.handles[layer] = (module.register_forward_pre_hook(self.pre_hook(layer), with_kwargs=True),
                               module.register_forward_hook(self.out_hook(layer)))

    def detach(self) -> list[tuple[int, object]]:
        """Remove every hook and restore the impl scales (the determinism rerun of V3); returns what reattach needs."""
        self.set_arm("BF16")
        saved = list(self.modules.items())
        for pre, post in self.handles.values():
            pre.remove()
            post.remove()
        self.handles.clear()
        return saved

    def reattach(self, saved: list[tuple[int, object]]) -> None:
        for layer, module in saved:
            self.attach(layer, module)

    def set_arm(self, arm: str) -> None:
        """Select the arm and set the score factor of each layer (the impl scale; the base value for ks = 1)."""
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm = arm
        for layer, module in self.modules.items():
            impl = getattr(module, "impl", None)
            if impl is not None:
                impl.scale = self.base_scale[layer] * self.layer_scales(layer)["ks"]

    def pre_hook(self, layer: int):
        hq, hk, head = self.heads[layer]
        rot, unrot = rot_dims(head)

        def pre(module, args, kwargs):
            try:
                q, k, v, put = qkv_of(args, kwargs)
                if k is None or v is None:
                    return None
                t = k.shape[0]
                key = (self.arm, str(self.doc), layer)
                self.counts[key] = self.counts.get(key, 0) + t
                fresh = layer in self.fresh
                self.fresh.discard(layer)
                first = (self.arm, layer) not in self.checked
                if self.arm == "BF16" and not first:
                    return None
                sc = self.layer_scales(layer)
                q3, k3, v3 = q.view(t, hq, head), k.view(t, hk, head), v.view(t, hk, head)
                q2, k2, v2 = transform(self.arm, q3, k3, v3, self.w[layer], sc, rot, unrot)
                if first:
                    self.checked.add((self.arm, layer))
                    bad = check_transform(self.arm, q3, k3, v3, q2, k2, v2, self.w[layer], sc, rot, unrot)
                    impl = getattr(module, "impl", None)
                    want = self.base_scale[layer] * sc["ks"]
                    if impl is not None and float(impl.scale) != want:
                        bad.append(f"{self.arm}: the impl scale is {impl.scale}, not {want}")
                    self.checks.setdefault(self.arm, []).extend(f"layer {layer}: {b}" for b in bad)
                    if fresh and t >= GAP_ROWS:   # the references of the post hook: positions 0 .. t-1 of this chunk
                        self.ref[layer] = (self.arm, q2[-GAP_ROWS:].clone(), k2, v2, want, sc["vs"],
                                           q3[-GAP_ROWS:].clone(), k3, v3, self.base_scale[layer])
                if self.arm == "BF16":
                    return None
                return put(q2.reshape(q.shape), k2.reshape(k.shape), v2.reshape(v.shape))
            except Exception as exc:  # recorded; V5 fails
                self.errors.setdefault(self.arm, []).append(f"layer {layer}: {type(exc).__name__}: {exc}"[:300])
                raise
        return pre

    def out_hook(self, layer: int):
        hq, _, head = self.heads[layer]

        def post(module, args, output):
            o = output[0] if isinstance(output, (tuple, list)) else output
            new = None
            if self.arm in SCALED:
                new = (o.float() * self.layer_scales(layer)["vs"]).to(o.dtype)
            ref = self.ref.pop(layer, None)
            if ref is not None:
                arm, qr, k2, v2, ks, vs, q0, k0, v0, base = ref
                try:
                    t = k2.shape[0]
                    qpos = torch.arange(t - GAP_ROWS, t, device=k2.device)
                    got = (o if new is None else new).reshape(t, hq, head)[-GAP_ROWS:].float()
                    want = attend32(qr, k2, v2, qpos, ks) * vs
                    self.gaps.setdefault(arm, {})[layer] = rel(got, want)
                    if arm == "BF16":                # the two references are the same tensors
                        self.gaps_id.setdefault(arm, {})[layer] = self.gaps[arm][layer]
                        self.seps.setdefault(arm, {})[layer] = 0.0
                    else:
                        want0 = attend32(q0, k0, v0, qpos, base)
                        self.gaps_id.setdefault(arm, {})[layer] = rel(got, want0)
                        self.seps.setdefault(arm, {})[layer] = rel(want, want0)
                except Exception as exc:  # recorded; V5 fails
                    self.errors.setdefault(arm, []).append(f"layer {layer} gap: {type(exc).__name__}: {exc}"[:300])
            if new is None:
                return None
            return (new,) + tuple(output[1:]) if isinstance(output, (tuple, list)) else new
        return post
