"""I-174 screen (SCREEN ONLY, no GPU): operand bit transitions per MMA lane under exact K-axis transforms.

For W [N, K] (PyTorch layout), the proxy is the mean number of changed bits between the codes at K positions p and
p + s of the same row (one lane on consecutive k-steps). See README.md for the frozen rule.

Usage: uv run --with numpy --with ml_dtypes python toggle_screen.py --model <model.safetensors> --out results/screen.json
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import ml_dtypes
import numpy as np

POP = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1).astype(np.int64)
LAYERS = (0, 13, 27)
RESID = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj")
STRIDE = {"bf16": 16, "fp8": 32}  # wgmma k per instruction
NBITS = 9  # sign + 8 exponent bits of BF16 drive the chain similarity
HEAD_DIM, Q_PER_KV = 128, 2


def load(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        out = {}
        for name in names:
            m = hdr[name]
            assert m["dtype"] == "BF16", (name, m["dtype"])
            a, b = m["data_offsets"]
            f.seek(8 + n + a)
            out[name] = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(m["shape"]).copy()
    return out


def to_f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def fp8_codes(w: np.ndarray, blk: int = 128) -> np.ndarray:
    """FP8 E4M3 codes, one scale (amax / 448) per blk x blk block, as block-FP8 checkpoints."""
    q = np.empty(w.shape, dtype=np.uint8)
    for i in range(0, w.shape[0], blk):
        for j in range(0, w.shape[1], blk):
            t = w[i:i + blk, j:j + blk]
            s = max(float(np.abs(t).max()) / 448.0, 1e-30)
            q[i:i + blk, j:j + blk] = (t / s).astype(ml_dtypes.float8_e4m3fn).view(np.uint8)
    return q


def transitions(codes: np.ndarray, s: int) -> float:
    x = codes[:, :-s] ^ codes[:, s:]
    if x.dtype == np.uint16:
        t = POP[(x & 0xFF).astype(np.uint8)].sum() + POP[(x >> 8).astype(np.uint8)].sum()
    else:
        t = POP[x].sum()
    return float(t) / x.size


def grams(mats: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Sign-plane and other-plane agreement (sum of +1/-1 products) between K columns, over all rows of mats."""
    k = mats[0].shape[1]
    sg, rest = np.zeros((k, k)), np.zeros((k, k))
    for m in mats:
        for r in range(0, m.shape[0], 2048):
            blk = m[r:r + 2048]
            for b in range(NBITS):
                pl = (1.0 - 2.0 * ((blk >> (15 - b)) & 1)).astype(np.float32)
                g = pl.T @ pl
                if b == 0:
                    sg += g
                else:
                    rest += g
    return sg, rest


def layout(chain: np.ndarray, s: int) -> np.ndarray:
    """order[p] = column; chain neighbours land s apart (same lane, consecutive k-steps)."""
    k = len(chain)
    steps = k // s
    i = np.arange(k)
    order = np.empty(k, dtype=np.int64)
    order[(i % steps) * s + i // steps] = chain
    return order


def chain_moves(sg: np.ndarray, rest: np.ndarray, flips: bool) -> tuple[np.ndarray, np.ndarray]:
    """Greedy nearest-neighbour chain on bit agreement; sign flips (if exact) align each column with its predecessor."""
    sim = (np.abs(sg) if flips else sg) + rest
    k = sim.shape[0]
    seen = np.zeros(k, dtype=bool)
    chain = [int(np.argmax(sim.sum(1)))]
    seen[chain[0]] = True
    for _ in range(k - 1):
        row = np.where(seen, -np.inf, sim[chain[-1]])
        nxt = int(np.argmax(row))
        chain.append(nxt)
        seen[nxt] = True
    sign = np.ones(k)
    if flips:
        for a, b in zip(chain, chain[1:]):
            sign[b] = sign[a] * (1.0 if sg[a, b] >= 0 else -1.0)
    return np.array(chain), sign


def apply(codes: np.ndarray, order: np.ndarray, sign: np.ndarray) -> np.ndarray:
    out = codes[:, order].copy()
    out[:, sign[order] < 0] ^= np.uint16(0x8000)
    return out


def pow2_equalize(codes: np.ndarray, groups: list[np.ndarray] | None = None) -> list[np.ndarray] | np.ndarray:
    """A1: scale each K column by a power of two so that its RMS is near the common RMS. Exact in BF16 (only the
    exponent changes); the inverse scale goes into the RMSNorm weight (residual inputs) or the paired matrix
    (up rows for down, V rows for o). For residual inputs, all matrices behind one norm share one scale vector."""
    mats = groups if groups is not None else [codes]
    f = [to_f32(m) for m in mats]
    col_rms = np.sqrt(np.mean(np.vstack(f) ** 2, axis=0)) + 1e-30
    target = np.exp(np.mean(np.log(col_rms)))
    scale = np.exp2(np.round(np.log2(target / col_rms))).astype(np.float32)
    out = [((x * scale).view(np.uint32) >> 16).astype(np.uint16) for x in f]
    return out if groups is not None else out[0]


def sorted_rows(codes: np.ndarray, s: int) -> np.ndarray:
    f = to_f32(codes)
    idx = np.argsort(f, axis=1, kind="stable")
    k = codes.shape[1]
    srt = np.take_along_axis(codes, idx, axis=1)
    return srt[:, layout(np.arange(k), s)]


def measure(codes: np.ndarray, fmt: str) -> dict:
    if fmt == "bf16":
        return {"lane": transitions(codes, STRIDE["bf16"]), "s1": transitions(codes, 1)}
    q = fp8_codes(to_f32(codes))
    return {"lane": transitions(q, STRIDE["fp8"]), "s1": transitions(q, 1)}


def group_moves(w: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """o_proj: permute and flip the V dims inside each KV group (both q heads of the group get the same move)."""
    k = w.shape[1]
    order, sign = np.arange(k), np.ones(k)
    group = HEAD_DIM * Q_PER_KV
    for g0 in range(0, k, group):
        blocks = [w[:, g0 + h * HEAD_DIM:g0 + (h + 1) * HEAD_DIM] for h in range(Q_PER_KV)]
        sg, rest = grams([np.vstack(blocks)])
        chain, sgn = chain_moves(sg, rest, flips=True)
        local = layout(chain, s)
        for h in range(Q_PER_KV):
            base = g0 + h * HEAD_DIM
            order[base:base + HEAD_DIM] = base + local
            sign[base:base + HEAD_DIM] = sgn
    return order, sign


def exactness_check(rng: np.random.Generator) -> float:
    """Toy SwiGLU plus RMSNorm in float64: the moves used here leave the output unchanged."""
    d, h = 64, 192
    x = rng.standard_normal(d)
    g, wg, wu, wd = rng.standard_normal(d), rng.standard_normal((h, d)), rng.standard_normal((h, d)), rng.standard_normal((d, h))
    def f(x, g, wg, wu, wd):
        n = g * x / np.sqrt(np.mean(x * x))
        a = wg @ n
        return wd @ (a / (1 + np.exp(-a)) * (wu @ n))
    y = f(x, g, wg, wu, wd)
    pi, sig = rng.permutation(h), rng.choice([-1.0, 1.0], h)          # hidden units: permute, flip up and down
    rp, rs = rng.permutation(d), rng.choice([-1.0, 1.0], d)          # residual basis: signed permutation
    y2 = f(rs * x[rp], g[rp], wg[pi][:, rp] * rs, (sig[:, None] * wu[pi])[:, rp] * rs, ((wd[:, pi] * sig)[rp]) * rs[:, None])
    return float(np.max(np.abs(rs * y[rp] - y2)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    err = exactness_check(rng)
    names = [f"model.layers.{l}.{m}.weight" for l in LAYERS for m in RESID + ("self_attn.o_proj", "mlp.down_proj")]
    w = load(a.model, names)
    res: dict = {"exactness_max_abs_err": err, "rows": []}

    shared = {}
    for fmt, s in STRIDE.items():
        sg, rest = grams([w[f"model.layers.{l}.{m}.weight"] for l in LAYERS for m in RESID])
        chain, sign = chain_moves(sg, rest, flips=True)
        shared[fmt] = (layout(chain, s), sign)

    for name in names:
        typ = name.split(".")[-2]
        c = w[name]
        k = c.shape[1]
        row = {"name": name, "type": typ, "shape": list(c.shape)}
        for fmt, s in STRIDE.items():
            v = {"orig": measure(c, fmt), "random": measure(c[:, rng.permutation(k)], fmt)}
            if typ == "down_proj":
                sg, rest = grams([c])
                ch, sgn = chain_moves(sg, rest, flips=True)
                v["exact"] = measure(apply(c, layout(ch, s), sgn), fmt)
            elif typ == "o_proj":
                order, sgn = group_moves(c, s)
                v["exact"] = measure(apply(c, order, sgn), fmt)
            else:
                order, sgn = shared[fmt]
                v["exact"] = measure(apply(c, order, sgn), fmt)
                sg, rest = grams([c])
                ch, sgn2 = chain_moves(sg, rest, flips=True)
                v["exact_own"] = measure(apply(c, layout(ch, s), sgn2), fmt)
            # A1 (added after the first result): power-of-two column scaling, then the same chain moves on the result.
            if typ in ("down_proj", "o_proj"):
                cs = pow2_equalize(c)
            else:
                lyr = name.split(".")[2]
                behind = RESID[:3] if typ in ("q_proj", "k_proj", "v_proj") else RESID[3:]
                grp = [w[f"model.layers.{lyr}.{m}.weight"] for m in behind]
                cs = pow2_equalize(c, grp)[[f"model.layers.{lyr}.{m}.weight" for m in behind].index(name)]
            if typ == "o_proj":
                order, sgn = group_moves(cs, s)
            else:
                sg, rest = grams([cs])
                ch, sgn = chain_moves(sg, rest, flips=True)
                order = layout(ch, s)
            v["exact_scaled_own"] = measure(apply(cs, order, sgn), fmt)
            v["sorted"] = measure(sorted_rows(c, s), fmt)
            row[fmt] = v
        res["rows"].append(row)
        print(name, {f: {k2: round(v2["lane"], 3) for k2, v2 in row[f].items()} for f in STRIDE})

    summary = {}
    for fmt in STRIDE:
        for typ in sorted({r["type"] for r in res["rows"]}):
            rows = [r for r in res["rows"] if r["type"] == typ]
            for var in ("random", "exact", "exact_own", "exact_scaled_own", "sorted"):
                rs = [r[fmt][var]["lane"] / r[fmt]["orig"]["lane"] for r in rows if var in r[fmt]]
                if rs:
                    summary.setdefault(fmt, {}).setdefault(typ, {})[var] = round(float(np.median(rs)), 4)
    res["summary_ratio_vs_orig"] = summary
    decision = {}
    for fmt, by in summary.items():
        ex = [v["exact"] for v in by.values()]
        a1 = [v["exact_scaled_own"] for v in by.values()]
        decision[fmt] = {"median_exact": round(float(np.median(ex)), 4),
                         "types_at_or_below_0.90": sum(e <= 0.90 for e in ex),
                         "A1_median_exact_scaled_own": round(float(np.median(a1)), 4),
                         "A1_types_at_or_below_0.90": sum(e <= 0.90 for e in a1),
                         "median_sorted": round(float(np.median([v["sorted"] for v in by.values()])), 4)}
    med = min(d["median_exact"] for d in decision.values())
    hits = max(d["types_at_or_below_0.90"] for d in decision.values())
    res["decision"] = ("PASS to E-47" if hits >= 2 else "KILL exact tier" if med >= 0.95 else "WEAK", decision)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(summary, indent=1))
    print("decision:", res["decision"], "exactness err:", err)


if __name__ == "__main__":
    main()
