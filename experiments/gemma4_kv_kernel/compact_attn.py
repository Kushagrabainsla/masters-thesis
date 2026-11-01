"""E-51 kernels: split-KV decode attention for the Gemma 4 global layers, with the standard K and V cache and with
the compact K=V cache, in one Triton code family, plus a PyTorch reference.

Gemma 4 global layers (FACT, E-49 and `kv_layout_check.py`): V = n (the RMS-normed k_proj output, no weight) and
K = rope_p(w * n) (k_norm weight w, proportional RoPE with p = 0.25: dims 0..63 and 256..319 rotate). So K equals
w * V on the other 384 dims. The compact cache stores one row per token and KV head: [V (512) | K on the 128
rotated dims] = 640 values, against 1,024 for K and V. The query takes w on the 384 unrotated dims:
    score = qc . V + qr . K_rot,   qc = q * w on the unrotated dims (0 on the rotated dims),   qr = q on the
    rotated dims.
The two kernels differ only in the layout. Each program handles one sequence, one KV head (its GQA query heads as
the rows of one tile) and one split of the KV blocks, with an online softmax; a second kernel merges the splits by
their log-sum-exp. The compact QK product runs over the 512 V dims (the rotated dims of qc are zero) plus the 128
K_rot dims: 25% more QK multiply-adds than needed, which is cheap because decode attention is memory-bound at these
shapes (INTERPRETATION; R-68 suggests 512-dim QK for a later kernel).

Without a GPU (the Mac tests) only the PyTorch helpers run: `rope`, `make_kv`, `compact_query`, `pack_compact`,
`paged_reference` and `split_reference` (the same split and merge rule as the kernels, in PyTorch).

E-52 adds two things (the E-51 behaviour stays the default):
- `cast`: the dtype that the cache values, the query and P take before each dot. E-51 used BF16. FACT (Triton 3.8.0,
  `ElementwiseOpToLLVM.cpp`): FP8 E4M3 to FP16 is one `cvt.rn.f16x2.e4m3x2` per 2 values; to BF16 on sm_90 it adds two
  scalar `cvt.bf16.f16`. The stock FA4 FP8 path computes in FP16 for this reason.
- The views layout: the compact row in the order [K_rot (128) | V_unrot (384) | V_rot (128)]. Its first 512 values
  are the K of the unchanged standard kernel ([K_rot | V_unrot], with w moved into the query) and its last 512 values
  are the V ([V_unrot | V_rot]). The two views overlap in 384 values, so a kernel that reads both reads 640 values
  per token from DRAM. The output comes in the V-view order ([unrotated dims | rotated dims]); `unpermute` restores
  the natural order (an engine can fold the order into the o_proj columns instead).
"""

from __future__ import annotations

import math

import torch

HEAD = 512
ROPE_ANGLES = 64                     # 0.25 x 512 / 2 angle pairs rotate
THETA = 1_000_000.0                  # Gemma 4 global rope theta (as `kv_layout_check.py`)
ROT = list(range(0, ROPE_ANGLES)) + list(range(HEAD // 2, HEAD // 2 + ROPE_ANGLES))   # the 128 rotated dims
RD = len(ROT)
UNROT = [d for d in range(HEAD) if d not in set(ROT)]                               # the 384 unrotated dims
KV_OFFSET = RD                       # the V view starts after K_rot in the views row (E-52)

try:  # Triton is in the Pod image; the Mac tests run without it
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on the Mac
    triton = None


def rope(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Gemma 4 proportional RoPE (neox rotate_half) on the last axis; pos has the shape of x[..., 0]."""
    inv = torch.zeros(HEAD // 2, dtype=torch.float64, device=x.device)
    inv[:ROPE_ANGLES] = 1.0 / THETA ** (torch.arange(0, 2 * ROPE_ANGLES, 2, dtype=torch.float64, device=x.device) / HEAD)
    ang = pos.to(torch.float64)[..., None] * inv
    cos, sin = torch.cat([torch.cos(ang)] * 2, -1).to(x.dtype), torch.cat([torch.sin(ang)] * 2, -1).to(x.dtype)
    half = torch.cat([-x[..., HEAD // 2:], x[..., :HEAD // 2]], -1)
    return x * cos + half * sin


def make_kv(tokens: int, heads: int, w: torch.Tensor, gen: torch.Generator, dtype=torch.float32):
    """K and V of `tokens` positions (0 .. tokens - 1) built as in Gemma 4: V = rms-normed, K = rope(w * V)."""
    v = torch.randn(tokens, heads, HEAD, generator=gen, dtype=torch.float64)
    v = v / v.pow(2).mean(-1, keepdim=True).sqrt()
    pos = torch.arange(tokens, dtype=torch.float64)[:, None].expand(tokens, heads)
    k = rope(w * v, pos)
    return k.to(dtype), v.to(dtype)


def compact_query(q_rope: torch.Tensor, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """qc: q * w on the unrotated dims, 0 on the rotated dims; qr: q on the rotated dims (pairs with K_rot)."""
    rot = torch.tensor(ROT, device=q_rope.device)
    qc = q_rope * w
    qc[..., rot] = 0
    return qc, q_rope[..., rot].contiguous()


def pack_compact(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The compact row [V (512) | K on the rotated dims (128)] on the last axis."""
    return torch.cat([v, k[..., torch.tensor(ROT, device=k.device)]], -1)


def pack_views(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The E-52 compact row [K_rot (128) | V_unrot (384) | V_rot (128)] on the last axis (640 values)."""
    rot, un = torch.tensor(ROT, device=k.device), torch.tensor(UNROT, device=k.device)
    return torch.cat([k[..., rot], v[..., un], v[..., rot]], -1)


def views_query(q_rope: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The query of the K view [K_rot | V_unrot]: q on the rotated dims, then q * w on the unrotated dims."""
    rot, un = torch.tensor(ROT, device=q_rope.device), torch.tensor(UNROT, device=q_rope.device)
    return torch.cat([q_rope[..., rot], (q_rope * w)[..., un]], -1)


def unpermute(o: torch.Tensor) -> torch.Tensor:
    """An output in the V-view order ([unrotated dims | rotated dims]) back in the natural dim order."""
    out = torch.empty_like(o)
    out[..., torch.tensor(UNROT, device=o.device)] = o[..., :len(UNROT)]
    out[..., torch.tensor(ROT, device=o.device)] = o[..., len(UNROT):]
    return out


def gqa_attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Dense reference for one sequence: q [Hq, D], k and v [L, Hk, D] -> [Hq, D] (GQA: Hq / Hk query heads per KV
    head)."""
    hq, hk = q.shape[0], k.shape[1]
    kk = k.repeat_interleave(hq // hk, dim=1)                 # [L, Hq, D]
    s = torch.einsum("hd,lhd->hl", q, kk) * scale
    return torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v.repeat_interleave(hq // hk, dim=1))


def gqa_attend_compact(qc: torch.Tensor, qr: torch.Tensor, row: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """The same attention from the compact rows: row [L, Hk, 640] = [V | K_rot]."""
    hq, hk = qc.shape[0], row.shape[1]
    v, kr = row[..., :HEAD].repeat_interleave(hq // hk, dim=1), row[..., HEAD:].repeat_interleave(hq // hk, dim=1)
    s = (torch.einsum("hd,lhd->hl", qc, v) + torch.einsum("hd,lhd->hl", qr, kr)) * scale
    return torch.einsum("hl,lhd->hd", torch.softmax(s, -1), v)


def gather(cache: torch.Tensor, block_table: torch.Tensor, b: int, length: int) -> torch.Tensor:
    """The first `length` tokens of sequence b from a paged cache [blocks, BS, H, X] -> [length, H, X]."""
    bs = cache.shape[1]
    blocks = block_table[b, : math.ceil(length / bs)].long()
    return cache[blocks].reshape(-1, *cache.shape[2:])[:length]


def paged_reference(q, cache_k, cache_v, block_table, seq_lens, scale=1.0, qr=None, cache_c=None, k_scale=1.0,
                    v_scale=1.0, r_scale=1.0) -> torch.Tensor:
    """Float32 reference from the stored (dequantized) cache values. Standard: cache_k and cache_v. Compact: q is qc,
    qr is given and cache_c holds the rows."""
    out = []
    for b in range(q.shape[0]):
        length = int(seq_lens[b])
        if cache_c is not None:
            row = gather(cache_c, block_table, b, length).float()
            row = torch.cat([row[..., :HEAD] * v_scale, row[..., HEAD:] * r_scale], -1)
            out.append(gqa_attend_compact(q[b].float(), qr[b].float(), row, scale))
        else:
            k = gather(cache_k, block_table, b, length).float() * k_scale
            v = gather(cache_v, block_table, b, length).float() * v_scale
            out.append(gqa_attend(q[b].float(), k, v, scale))
    return torch.stack(out)


def split_reference(q, k, v, splits: int, block: int, scale: float = 1.0) -> torch.Tensor:
    """The kernels' split rule and log-sum-exp merge, in PyTorch, for one sequence (k and v [L, Hk, D]): split s
    takes blocks [s * per, min((s + 1) * per, n)) with per = ceil(n / splits); an empty split has lse = -inf."""
    hq, hk = q.shape[0], k.shape[1]
    kk, vv = k.repeat_interleave(hq // hk, dim=1), v.repeat_interleave(hq // hk, dim=1)
    n = math.ceil(k.shape[0] / block)
    per = math.ceil(n / splits)
    outs, lses = [], []
    for s in range(splits):
        lo, hi = s * per * block, min((s + 1) * per, n) * block
        if lo >= min(hi, k.shape[0]):
            outs.append(torch.zeros_like(q))
            lses.append(torch.full((hq,), float("-inf"), dtype=q.dtype))
            continue
        sc = torch.einsum("hd,lhd->hl", q, kk[lo:hi]) * scale
        m = sc.max(-1).values
        p = torch.exp(sc - m[:, None])
        outs.append(torch.einsum("hl,lhd->hd", p, vv[lo:hi]) / p.sum(-1)[:, None])
        lses.append(m + torch.log(p.sum(-1)))
    lse = torch.stack(lses)                                  # [splits, Hq]
    wts = torch.exp(lse - lse.max(0).values)
    return (torch.stack(outs) * wts[..., None]).sum(0) / wts.sum(0)[:, None]


if triton is not None:

    @triton.jit
    def _split_kernel(Q, QR, K, V, R, BT, SEQ, OP, LP,
                      s_qb, s_qh, s_qrb, s_qrh,
                      s_kn, s_kt, s_kh, s_vn, s_vt, s_vh, s_rn, s_rt, s_rh,
                      s_btb, s_ob, s_oh, s_os, s_lb, s_lh,
                      k_scale, v_scale, r_scale, sm_scale,
                      GQA: tl.constexpr, MP: tl.constexpr, D: tl.constexpr, RDIM: tl.constexpr,
                      BS: tl.constexpr, TB: tl.constexpr, SPLITS: tl.constexpr, COMPACT: tl.constexpr,
                      CAST: tl.constexpr):
        # A split owns whole pages of BS tokens; it reads each page as BS / TB tiles of TB tokens.
        b = tl.program_id(0)
        hk = tl.program_id(1)
        sp = tl.program_id(2)
        seq = tl.load(SEQ + b)
        nblk = tl.cdiv(seq, BS)
        per = tl.cdiv(nblk, SPLITS)
        j0 = sp * per
        j1 = tl.minimum(j0 + per, nblk)
        SUB: tl.constexpr = BS // TB
        rows = tl.arange(0, MP)
        rmask = rows < GQA
        qh = hk * GQA + rows
        dims = tl.arange(0, D)
        q = tl.load(Q + b * s_qb + qh[:, None] * s_qh + dims[None, :], mask=rmask[:, None], other=0.0).to(CAST)
        rdims = tl.arange(0, RDIM)
        # Standard layout: QR is Q and qr is not used.
        qr = tl.load(QR + b * s_qrb + qh[:, None] * s_qrh + rdims[None, :], mask=rmask[:, None], other=0.0).to(CAST)
        m_i = tl.full([MP], float("-inf"), tl.float32)
        l_i = tl.zeros([MP], tl.float32)
        acc = tl.zeros([MP, D], tl.float32)
        for t in range(j0 * SUB, j1 * SUB):
            j = t // SUB
            toks = (t % SUB) * TB + tl.arange(0, TB)
            pb = tl.load(BT + b * s_btb + j).to(tl.int64)
            tmask = (j * BS + toks) < seq
            v = tl.load(V + pb * s_vn + toks[:, None] * s_vt + hk * s_vh + dims[None, :], mask=tmask[:, None],
                        other=0.0).to(CAST)
            if COMPACT:
                kr = tl.load(R + pb * s_rn + toks[:, None] * s_rt + hk * s_rh + rdims[None, :], mask=tmask[:, None],
                             other=0.0).to(CAST)
                s = tl.dot(q, tl.trans(v)) * (v_scale * sm_scale) + tl.dot(qr, tl.trans(kr)) * (r_scale * sm_scale)
            else:
                k = tl.load(K + pb * s_kn + toks[:, None] * s_kt + hk * s_kh + dims[None, :], mask=tmask[:, None],
                            other=0.0).to(CAST)
                s = tl.dot(q, tl.trans(k)) * (k_scale * sm_scale)
            s = tl.where(tmask[None, :], s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(CAST), v) * v_scale
            m_i = m_new
        ok = l_i > 0
        safe = tl.where(ok, l_i, 1.0)
        out = tl.where(ok[:, None], acc / safe[:, None], 0.0)
        lse = tl.where(ok, m_i + tl.log(safe), float("-inf"))
        tl.store(OP + b * s_ob + qh[:, None] * s_oh + sp * s_os + dims[None, :], out, mask=rmask[:, None])
        tl.store(LP + b * s_lb + qh * s_lh + sp, lse, mask=rmask)

    @triton.jit
    def _merge_kernel(OP, LP, O, s_ob, s_oh, s_os, s_lb, s_lh, s_Ob, s_Oh, D: tl.constexpr, SPLITS: tl.constexpr):
        b = tl.program_id(0)
        h = tl.program_id(1)
        sp = tl.arange(0, SPLITS)
        lse = tl.load(LP + b * s_lb + h * s_lh + sp)
        wts = tl.exp(lse - tl.max(lse, 0))
        wts = tl.where(lse == float("-inf"), 0.0, wts)
        dims = tl.arange(0, D)
        o = tl.load(OP + b * s_ob + h * s_oh + sp[:, None] * s_os + dims[None, :])
        out = tl.sum(o * wts[:, None], 0) / tl.sum(wts, 0)
        tl.store(O + b * s_Ob + h * s_Oh + dims, out.to(O.dtype.element_ty))


def views_of(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The K view (values 0 to 511) and the V view (values 128 to 639) of views rows [..., 640]."""
    return rows[..., :HEAD], rows[..., KV_OFFSET:KV_OFFSET + HEAD]


def workspace(q: torch.Tensor, splits: int) -> tuple[torch.Tensor, torch.Tensor]:
    b, hq, d = q.shape
    return (torch.empty(b, hq, splits, d, dtype=torch.float32, device=q.device),
            torch.empty(b, hq, splits, dtype=torch.float32, device=q.device))


def triton_attn(q, cache_v, block_table, seq_lens, out, ws, *, splits: int, num_warps: int = 8, num_stages: int = 2,
                tile: int = 32, cache_k=None, qr=None, cache_r=None, k_scale=1.0, v_scale=1.0, r_scale=1.0,
                sm_scale=1.0, info: dict | None = None, cast: str = "bf16"):
    """One decode attention call. Standard: cache_k and cache_v [blocks, BS, Hk, 512]. Compact: q is qc, qr is the
    rotated query part, cache_v and cache_r are the two column views of one [blocks, BS, Hk, 640] row tensor. The E-52
    views layout is the standard call on the two views of `views_of` with the query of `views_query`.
    `tile` tokens are read per loop step (it divides the page size). `cast` ("bf16" or "fp16") is the dtype of the
    dot inputs (E-52). If `info` is a dict, it gets the registers, spills and shared memory bytes of the compiled split
    kernel (A1, O5; None where Triton does not give them); with info["asm"] set, also its PTX, TTGIR and cubin."""
    grid, args, kw = split_args(q, cache_v, block_table, seq_lens, ws, splits=splits, num_warps=num_warps,
                                num_stages=num_stages, tile=tile, cache_k=cache_k, qr=qr, cache_r=cache_r,
                                k_scale=k_scale, v_scale=v_scale, r_scale=r_scale, sm_scale=sm_scale, cast=cast)
    b, hq, d = q.shape
    op, lp = ws
    kern = _split_kernel[grid](*args, **kw)
    if info is not None:
        md = getattr(kern, "metadata", None)
        want_asm = info.pop("asm", False)
        info.update(regs=getattr(kern, "n_regs", None), spills=getattr(kern, "n_spills", None),
                    shared=md.get("shared") if isinstance(md, dict) else getattr(md, "shared", None))
        if want_asm:
            asm = getattr(kern, "asm", None) or {}
            info.update(ptx=asm.get("ptx"), ttgir=asm.get("ttgir"), cubin=asm.get("cubin"))
    _merge_kernel[(b, hq)](op, lp, out, op.stride(0), op.stride(1), op.stride(2), lp.stride(0), lp.stride(1),
                           out.stride(0), out.stride(1), D=d, SPLITS=splits, num_warps=4)
    return out


def split_args(q, cache_v, block_table, seq_lens, ws, *, splits: int, num_warps: int, num_stages: int, tile: int,
               cache_k=None, qr=None, cache_r=None, k_scale=1.0, v_scale=1.0, r_scale=1.0, sm_scale=1.0,
               cast: str = "bf16"):
    """The grid, positional arguments and keyword arguments of one `_split_kernel` launch (shared by `triton_attn`
    and the E-52 compile check, which calls `_split_kernel.warmup` with them)."""
    compact = cache_r is not None
    b, hq, d = q.shape
    hk, bs = cache_v.shape[2], cache_v.shape[1]
    gqa = hq // hk
    op, lp = ws
    k_t = cache_v if compact else cache_k
    r_t = cache_r if compact else cache_k
    qr_t = qr if compact else q
    args = (q, qr_t, k_t, cache_v, r_t, block_table, seq_lens, op, lp,
            q.stride(0), q.stride(1), qr_t.stride(0), qr_t.stride(1),
            k_t.stride(0), k_t.stride(1), k_t.stride(2), cache_v.stride(0), cache_v.stride(1), cache_v.stride(2),
            r_t.stride(0), r_t.stride(1), r_t.stride(2),
            block_table.stride(0), op.stride(0), op.stride(1), op.stride(2), lp.stride(0), lp.stride(1),
            k_scale, v_scale, r_scale, sm_scale)
    kw = dict(GQA=gqa, MP=max(16, triton.next_power_of_2(gqa)), D=d, RDIM=RD, BS=bs, TB=tile, SPLITS=splits,
              COMPACT=compact, CAST=tl.float16 if cast == "fp16" else tl.bfloat16, num_warps=num_warps,
              num_stages=num_stages)
    return (b, hk, splits), args, kw
