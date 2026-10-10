"""E-27 library: V-cache quantization with and without noise shaping, and a cache-emulating
attention function for Qwen3 in transformers.

Cache model (KIVI-style): for a query at position t, tokens j < sink and j >= t - recent keep
full-precision V; tokens sink <= j < t - recent use the quantized V. K is never quantized.
Defaults: sink = SINK, recent = RECENT. A condition can override them ("sink", "recent").
Plan, amendment A1 and criteria: RESEARCH_JOURNAL.md, Section 6.1, E-27.
"""

import math

import torch

SINK = 4
RECENT = 128
GROUP = 32
HBLOCK = 64
SORT_BLOCK = 128
MIN_REGION = 256    # a query position is measured only if it reads at least this many quantized tokens


def hadamard(n, dtype=torch.float32):
    """Normalized Sylvester Hadamard matrix (orthonormal), n a power of two."""
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return (h / math.sqrt(n)).to(dtype)


def rotate(v, inverse=False):
    """Block-diagonal Hadamard rotation of the last dimension (blocks of HBLOCK channels)."""
    h = hadamard(HBLOCK, v.dtype).to(v.device)
    shape = v.shape
    x = v.reshape(*shape[:-1], shape[-1] // HBLOCK, HBLOCK)
    x = x @ (h.T if inverse else h)
    return x.reshape(shape)


def quantize_groups(u, bits):
    """Asymmetric min-max round-to-nearest over groups of GROUP channels in the last dimension.
    Returns (dequantized values, step size per group broadcast to u's shape)."""
    shape = u.shape
    x = u.reshape(*shape[:-1], shape[-1] // GROUP, GROUP)
    lo = x.amin(-1, keepdim=True)
    hi = x.amax(-1, keepdim=True)
    levels = (1 << bits) - 1
    step = (hi - lo) / levels
    safe = torch.where(step > 0, step, torch.ones_like(step))
    q = torch.round((x - lo) / safe) * safe + lo
    q = torch.where(step > 0, q, x)
    return q.reshape(shape), step.expand_as(x).reshape(shape)


def _signs(T, H, D, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (T, H, D), generator=g) * 2 - 1).float()


def _block_order(keys, start, end, H, order, seed):
    """Token order of one block for each head: 'time', 'keys' (projection of the keys on their
    first principal direction) or 'random'. Returns [H, n] token indices."""
    n = end - start
    if order == "keys" and n > 2:
        kb = keys[start:end].transpose(0, 1).to(torch.float64)        # [H, n, D]
        kc = kb - kb.mean(1, keepdim=True)
        _, _, vh = torch.linalg.svd(kc, full_matrices=False)
        proj = (kc @ vh[:, 0, :].unsqueeze(-1)).squeeze(-1)           # [H, n]
        return torch.argsort(proj, dim=1) + start
    if order == "random":
        g = torch.Generator().manual_seed(seed * 1000003 + start)
        return torch.stack([torch.randperm(n, generator=g) for _ in range(H)]) + start
    return torch.arange(start, end).unsqueeze(0).expand(H, n)


def quantize_sequence(v, bits, rotated=True, shaping=False, order="time", keys=None,
                      dither=False, sink=SINK, seed=0):
    """Quantize V tokens sink..T-1. v: [T, H, D] (one sequence).

    shaping=False: each token is rounded on its own.
    shaping=True: per head and channel, u_j = v_j + c_prev, q_j = Q(u_j), c = u_j - q_j, with the
    scale computed from u_j. The carry runs in token order inside blocks of SORT_BLOCK tokens and
    passes from block to block. order: 'time', 'keys' (needs keys [T, H, D]) or 'random'.
    dither=True: multiply each token by seeded random signs per channel before the rotation and
    undo them after dequantization (decorrelates the errors of repeated tokens).
    Returns vhat [T, H, D]; rows j < sink are copies of v (they are never read quantized).
    """
    T, H, D = v.shape
    if T <= sink:
        return v.clone()
    s = _signs(T, H, D, seed) if dither else None
    x = v * s if dither else v
    x = rotate(x) if rotated else x.clone()
    out = x.clone()
    if not shaping:
        q, _ = quantize_groups(x[sink:], bits)
        out[sink:] = q
    else:
        carry = torch.zeros(H, D, dtype=x.dtype)
        heads = torch.arange(H)
        start = sink
        while start < T:
            end = min(start + SORT_BLOCK, T)
            idx_all = _block_order(keys, start, end, H, order, seed)
            for p in range(end - start):
                idx = idx_all[:, p]
                u = x[idx, heads] + carry
                q, _ = quantize_groups(u, bits)
                carry = u - q
                out[idx, heads] = q
            start = end
    out = rotate(out, inverse=True) if rotated else out
    return out * s if dither else out


def step_variance(v, bits, rotated=True):
    """Mean rounding-error variance (step^2 / 12) of plain per-token quantization over tokens
    SINK..T-1, for each head. v: [T, H, D]. Returns [H]."""
    x = rotate(v) if rotated else v
    _, step = quantize_groups(x[SINK:], bits)
    return (step ** 2 / 12).mean(dim=(0, 2))


def smoothness_terms(w, b, sink=SINK):
    """For weights w over positions [sink, b) of one query: numerator and denominator of G^2.
    numerator = sum (w_(j+1) - w_j)^2 over the region + w_(b-1)^2 (the carry that leaves the
    region is not compensated), denominator = sum w_j^2. Diagnostic only (amendment A1): it assumes
    independent plain rounding errors, which real V does not satisfy."""
    seg = w[..., sink:b]
    num = ((seg[..., 1:] - seg[..., :-1]) ** 2).sum(-1) + seg[..., -1] ** 2
    den = (seg ** 2).sum(-1)
    return num, den


class State:
    """Global switches read by the attention function. One instance, set before each forward."""

    def __init__(self):
        self.mode = "fp"          # "fp", "quant" (propagated), or "measure" (fp output, record)
        self.cond = None          # condition dict for mode "quant"
        self.measure_conds = {}   # name -> condition dict, for mode "measure"
        self.gates = {}           # name -> bool tensor [layers, kv_heads]; shape where True
        self.record = None        # dict filled in mode "measure"
        self.chunk = 256


STATE = State()


def _vhat(v_seq, k_seq, cond, layer):
    """Quantized V for one layer and one sequence under a condition. v_seq, k_seq: [T, H, D]."""
    kw = dict(rotated=cond["rotated"], dither=cond.get("dither", False),
              sink=cond.get("sink", SINK), seed=1000 + layer)
    plain = quantize_sequence(v_seq, cond["bits"], shaping=False, **kw)
    if not cond.get("shaping"):
        return plain
    order = cond.get("order", "time")
    shaped = quantize_sequence(v_seq, cond["bits"], shaping=True, order=order,
                               keys=k_seq if order == "keys" else None, **kw)
    gate = cond.get("gate")
    if gate is None:
        return shaped
    g = STATE.gates[gate][layer].to(v_seq.device).view(1, -1, 1)
    return torch.where(g, shaped, plain)


def attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Causal attention with an emulated quantized V cache. Batch size must be 1, no padding.
    query [1, Hq, T, D], key/value [1, Hkv, T, D]. Returns ([1, T, Hq, D], None)."""
    assert query.shape[0] == 1
    _, hq, T, D = query.shape
    hkv = key.shape[1]
    rep = hq // hkv
    layer = module.layer_idx
    q = query[0]
    k = key[0].repeat_interleave(rep, dim=0)       # [Hq, T, D], same order as repeat_kv
    v = value[0].repeat_interleave(rep, dim=0)
    v_seq = value[0].transpose(0, 1)                # [T, Hkv, D]
    k_seq = key[0].transpose(0, 1)

    delta, sink, recent, self_q = None, SINK, RECENT, 0
    if STATE.mode == "quant":
        vh = _vhat(v_seq, k_seq, STATE.cond, layer)
        delta = (vh - v_seq).transpose(0, 1).repeat_interleave(rep, dim=0)  # [Hq, T, D]
        sink = STATE.cond.get("sink", SINK)
        recent = STATE.cond.get("recent", RECENT)
        self_q = 1 if STATE.cond.get("self_quantized") else 0   # all-quantized variants read their own token quantized
    deltas = {}
    if STATE.mode == "measure":
        for name, cond in STATE.measure_conds.items():
            assert "sink" not in cond and "recent" not in cond, "measure mode uses the default region"
            vh = _vhat(v_seq, k_seq, cond, layer)
            deltas[name] = (vh - v_seq).transpose(0, 1).repeat_interleave(rep, dim=0)
        rec = STATE.record
        var2 = step_variance(v_seq, 2, rotated=True).repeat_interleave(rep)  # [Hq]

    out = torch.empty(hq, T, D, dtype=q.dtype)
    pos = torch.arange(T)
    for s in range(0, T, STATE.chunk):
        e = min(s + STATE.chunk, T)
        scores = (q[:, s:e] @ k.transpose(1, 2)) * scaling          # [Hq, c, T]
        causal = pos.unsqueeze(0) > pos[s:e].unsqueeze(1)            # [c, T]
        scores = scores.masked_fill(causal, float("-inf"))
        p = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        o = p @ v
        if delta is not None:
            tq = pos[s:e].unsqueeze(1)
            j = pos.unsqueeze(0)
            m = ((j >= sink) & (j < tq - recent + self_q)).to(p.dtype)  # [c, T]
            o = o + (p * m) @ delta
        if STATE.mode == "measure":
            for t in range(s, e):
                b = t - RECENT
                if b - SINK < MIN_REGION:
                    continue
                if t % 8 == 0:
                    num, den = smoothness_terms(p[:, t - s], b)       # [Hq]
                    en = den * var2
                    rec["g_num"][layer] += (num * var2).double()
                    rec["g_den"][layer] += en.double()
                if t % 16 == 0:
                    ofp = o[:, t - s]                                 # [Hq, D]
                    nfp = (ofp ** 2).sum(-1)
                    rec["n_pos"][layer] += 1
                    rec["fp_energy"][layer] += nfp.double()
                    for name, dlt in deltas.items():
                        err = p[:, t - s, SINK:b].unsqueeze(1) @ dlt[:, SINK:b]  # [Hq, 1, D]
                        rec["err_energy"][name][layer] += (err.squeeze(1) ** 2).sum(-1).double()
        out[:, s:e] = o
    return out.transpose(0, 1).unsqueeze(0), None


def new_record(layers, hq, names):
    z = lambda: torch.zeros(layers, hq, dtype=torch.float64)
    return {
        "g_num": z(), "g_den": z(), "fp_energy": z(),
        "n_pos": torch.zeros(layers, dtype=torch.float64),
        "err_energy": {n: z() for n in names},
    }
