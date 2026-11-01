"""E-53 capture (README Sections 4 and 11; gate G3 of D-88; amendment A1 after R-73): the inputs (query, key, value)
and the output of the attention modules of the Gemma 4 31B global layers, after RoPE, on 4 real prompts of 8,192
tokens, in vLLM 3966924d5 in-process.

How the hooks reach the model (FACT, vLLM 3966924d5; README Section 4.3 gives every line):
- `LLM.apply_model(fn)` runs fn(model) in the worker (vllm/entrypoints/llm.py 604 to 614). With
  VLLM_ENABLE_V1_MULTIPROCESSING=0 the engine core and the worker run in this process (v1/engine/llm_engine.py 186 to
  188 and 116 to 138, v1/engine/core_client.py 376 to 401, v1/executor/uniproc_executor.py 90 to 103), so a hook can
  write into an object of this script.
- `Gemma4Attention.forward` (models/gemma4.py 540 to 575) applies k_norm (565), RoPE to q and k (567) and the
  weightless v_norm (570), then calls `self.attn(q, k, v)` (572). A forward pre-hook on `self.attn` (an `Attention`
  module) gets q and k after RoPE and v after its norm, as [tokens, heads x 512] tensors. (A1, R-73 B2) A forward hook
  on the same module gets the engine's attention output [tokens, heads x 512] (attention.py 584).
- The decoder layer calls `self_attn(positions=..., hidden_states=...)` with keywords (gemma4.py 745 to 749), so a
  pre-hook with kwargs on `self_attn` gets the positions of the same rows.
- `enforce_eager=True` sets the compilation mode NONE and the CUDA graph mode NONE (config/vllm.py 1782 to 1784), so
  the modules run as Python modules and the hooks fire.

Order: build the prompts (a fixed seed, text files of the image), start the engine, put the hooks on the global layers
that own a KV cache, then one `generate` call per prompt with max_tokens 1 (prefill only). Fail fast: if the hooked
modules or the first prompt do not match the plan (no global-layer tensors, other shapes, dtype or positions), the
script stops at once with the status "fail_fast". The captures (about 780 MB per prompt) go to --capture; only small
files go to --out. (A1, R-73 F1) The shape flags (--heads, --kv-heads, --global-layers) keep the frozen 31B values
as defaults; a rehearsal on a small Gemma 4 model sets them and --rehearsal. --stub writes synthetic captures built
like Gemma 4 (no GPU, no vLLM, no tokenizer) for the Mac tests.

Usage: capture_e53.py --snapshot <dir> --model-id <id> --revision <sha> --out <res> --capture <dir> [shape flags]
       capture_e53.py --stub --out <res> --capture <dir> [--stub-...]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "gemma4_kv_kernel"))

import compact_attn as ca  # noqa: E402  (rope and the dim lists, for the stub)

HQ, HK, D = 32, 4, 512                     # the global layers of Gemma 4 31B (config; G1, G2): the flag defaults
N_GLOBAL = 10                              # the global layers of Gemma 4 31B (5, 11, ..., 59): the flag default
TOKENS, Q_ROWS, N_PROMPTS = 8192, 256, 4   # the frozen plan (README Section 4)
SEED = 53
FAMILIES = ("vllm_py", "torch_py", "docs", "mixed")   # prompt p takes its files from FAMILIES[p % 4]
FILE_MIN, FILE_MAX = 2_000, 200_000        # bytes of a usable text file
FILE_TOKENS = 3_000                        # at most this many tokens of one file, so a prompt has 3 files or more
SKIP_WORDS = ("LICENSE", "COPYING", "NOTICE", "AUTHORS", "CHANGELOG")
TOY_SPECIAL = frozenset({0, 1, 2})         # the special ids of the stub tokenizer (2 is its BOS)
ENGINE = {"seed": 0, "enforce_eager": True, "kv_cache_dtype": "auto", "max_model_len": 8448,
          "max_num_batched_tokens": 8448, "max_num_seqs": 4, "gpu_memory_utilization": 0.85,
          "enable_prefix_caching": False, "limit_mm_per_prompt": {"image": 0, "audio": 0, "video": 0}}
# max_num_batched_tokens >= the prompt, so each prompt is one prefill step; text-only mode as E-49 and E-50.


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def ids_sha(ids: list[int]) -> str:
    """SHA-256 of the token ids as little-endian int32 values."""
    return sha(b"".join(int(i).to_bytes(4, "little", signed=True) for i in ids))


# ---------------------------------------------------------------- prompts


def package_dir(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    return Path(list(spec.submodule_search_locations)[0])


def text_files(roots: list[Path], exts: tuple[str, ...]) -> list[Path]:
    """The usable text files under the roots, sorted: 2 KB to 200 KB, not a license or change log. (A1, R-73 Q6) One
    walk of each root for all the suffixes."""
    out = set()
    for r in roots:
        for f in r.rglob("*"):
            if f.suffix not in exts:
                continue
            try:
                ok = f.is_file() and FILE_MIN <= f.stat().st_size <= FILE_MAX
            except OSError:
                continue
            if ok and not any(w in f.name.upper() for w in SKIP_WORDS):
                out.add(f)
    return sorted(out)


def sources(roots: dict[str, list[Path]]) -> dict[str, list[Path]]:
    """The file list of each family: Python files of vllm and of torch, Markdown and reStructuredText files of the
    whole site-packages folder, and all of them (mixed)."""
    s = {"vllm_py": text_files(roots["vllm_py"], (".py",)), "torch_py": text_files(roots["torch_py"], (".py",)),
         "docs": text_files(roots["docs"], (".md", ".rst"))}
    s["mixed"] = sorted(set(s["vllm_py"]) | set(s["torch_py"]) | set(s["docs"]))
    return s


def image_roots() -> tuple[dict[str, list[Path]], Path]:
    vdir, tdir = package_dir("vllm"), package_dir("torch")
    site = vdir.parent
    return {"vllm_py": [vdir], "torch_py": [tdir], "docs": [site]}, site


def build_prompts(src: dict[str, list[Path]], encode, bos: int | None, base: Path, seed: int = SEED,
                  n: int = N_PROMPTS, tokens: int = TOKENS, special: frozenset = frozenset()) -> list[dict]:
    """n prompts of exactly `tokens` token ids: BOS, then the tokens of text files in a seeded random order (at most
    FILE_TOKENS per file), each followed by the tokens of a blank line. Prompt p takes its files from family
    FAMILIES[p % 4]; if the family runs out, it continues with the other files of the mixed family. (A1, R-73 F4) A
    file whose ids hold a special id (for example the text "<bos>" in a source file) is skipped, so BOS is the only
    special id, at position 0."""
    sep = list(encode("\n\n"))
    out = []
    for p in range(n):
        fam = FAMILIES[p % len(FAMILIES)]
        rng = random.Random(seed * 1000 + p)
        order = list(src[fam])
        rng.shuffle(order)
        seen = set(order)
        rest = [f for f in src["mixed"] if f not in seen]
        rng.shuffle(rest)
        ids = [bos] if bos is not None else []
        used, skipped = [], []
        for f in order + rest:
            if len(ids) >= tokens:
                break
            try:
                raw = f.read_bytes()
                text = raw.decode("utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            t = list(encode(text))
            try:
                name = str(f.relative_to(base))
            except ValueError:
                name = str(f)
            if special and any(i in special for i in t):
                skipped.append(name)
                continue
            t = t[:FILE_TOKENS]
            if not t:
                continue
            take = (t + sep)[: tokens - len(ids)]
            ids += take
            used.append({"path": name, "sha256": sha(raw), "tokens": len(take)})
        out.append({"id": p, "family": fam, "files": used, "n_tokens": len(ids), "token_sha256": ids_sha(ids),
                    "special_after_0": sum(1 for i in ids[1:] if i in special), "skipped_special": len(skipped),
                    "token_ids": [int(i) for i in ids]})
    return out


# ---------------------------------------------------------------- hooks


class Store:
    """What the hooks record for one prompt: per global layer the rows of q and of the engine output o at the last
    q_rows positions, and the rows of k and v at every position below the prompt length, with their positions. The
    rows go to the CPU at once."""

    def __init__(self, q_rows: int):
        self.q_rows, self.length = q_rows, None
        self.pos, self.rows, self.qsel, self.parts, self.errors = {}, {}, {}, {}, []

    def begin(self, length: int) -> None:
        self.length, self.pos, self.rows, self.qsel, self.parts, self.errors = length, {}, {}, {}, {}, []

    def pos_hook(self, layer: int):
        def hook(module, args, kwargs):
            p = kwargs.get("positions", args[0] if args else None)
            if isinstance(p, torch.Tensor):
                self.pos[layer] = (p[0] if p.dim() == 2 else p).detach()
            return None
        return hook

    def attn_hook(self, layer: int, heads: int, kv_heads: int, head: int):
        def hook(module, args, kwargs):
            if self.length is None:              # outside a capture (for example a warm-up run)
                return None
            try:
                q, k, v = (args[i] if i < len(args) else kwargs.get(n) for i, n in enumerate(("query", "key", "value")))
                n = q.shape[0]
                pos, src = self.pos.pop(layer, None), "positions"
                if pos is None or pos.shape[0] != n:
                    start = self.rows.get(layer, 0)
                    pos, src = torch.arange(start, start + n, device=q.device), "row order"
                self.rows[layer] = self.rows.get(layer, 0) + n
                pos = pos.to(q.device, torch.int64)
                keep = pos < self.length
                qsel = keep & (pos >= self.length - self.q_rows)
                self.qsel[layer] = qsel                                  # (A1, B2) the output hook uses the same rows
                part = self.parts.setdefault(layer, {"q": [], "qpos": [], "k": [], "v": [], "kpos": [], "o": [],
                                                     "src": [], "dtype": str(q.dtype)})
                part["q"].append(q[qsel].reshape(-1, heads, head).to("cpu"))
                part["qpos"].append(pos[qsel].to("cpu"))
                part["k"].append(k[keep].reshape(-1, kv_heads, head).to("cpu"))
                part["v"].append(v[keep].reshape(-1, kv_heads, head).to("cpu"))
                part["kpos"].append(pos[keep].to("cpu"))
                part["src"].append(src)
            except Exception as exc:  # recorded; the check after the prompt fails
                self.errors.append(f"layer {layer}: {type(exc).__name__}: {exc}"[:300])
            return None
        return hook

    def out_hook(self, layer: int, heads: int, head_v: int):
        """(A1, R-73 B2) The engine's attention output of the same rows as q (Attention.forward returns
        [tokens, heads x head_size_v], attention.py 584)."""
        def hook(module, args, output):
            if self.length is None:
                return None
            try:
                qsel = self.qsel.pop(layer)
                o = output[0] if isinstance(output, (tuple, list)) else output
                self.parts[layer]["o"].append(o[qsel].reshape(-1, heads, head_v).to("cpu"))
            except Exception as exc:  # recorded; the check after the prompt fails
                self.errors.append(f"layer {layer} output: {type(exc).__name__}: {exc}"[:300])
            return None
        return hook

    def end(self) -> dict:
        out = {}
        for layer, p in self.parts.items():
            qpos, kpos = torch.cat(p["qpos"]), torch.cat(p["kpos"])
            qo, ko = torch.argsort(qpos), torch.argsort(kpos)
            d = {"q": torch.cat(p["q"])[qo].contiguous(), "qpos": qpos[qo], "k": torch.cat(p["k"])[ko].contiguous(),
                 "v": torch.cat(p["v"])[ko].contiguous(), "kpos": kpos[ko], "src": sorted(set(p["src"])),
                 "dtype": p["dtype"]}
            if p["o"] and sum(x.shape[0] for x in p["o"]) == qpos.shape[0]:
                d["o"] = torch.cat(p["o"])[qo].contiguous()
            out[layer] = d
        self.length = None
        return out


def global_layers(layer_types: list[str], num_hidden_layers: int, num_kv_shared_layers: int = 0) -> list[int]:
    """(A1, R-73 F1) The global (full-attention) layers that own a KV cache: a KV-shared layer (index at or above
    num_hidden_layers - num_kv_shared_layers, gemma4.py 422 to 426) calls its attention with no key and value
    (gemma4.py 546 to 553)."""
    first_shared = num_hidden_layers - (num_kv_shared_layers or 0)
    return [i for i, x in enumerate(layer_types) if x == "full_attention" and i < first_shared]


def register(model, store: Store) -> dict:
    """Pre-hooks on the Attention modules of the global layers (the parent is a Gemma4Attention with is_sliding False
    and no KV sharing) and on their parents (the positions), and (A1) a forward hook on each of those Attention modules
    (the output). Returns the hooked modules and the k_norm weights."""
    mods = dict(model.named_modules())
    hooked, knorm = [], {}
    for name, m in mods.items():
        hit = re.search(r"layers\.(\d+)\.self_attn\.attn$", name)
        if not hit or type(m).__name__ != "Attention" or not hasattr(m, "layer_name"):
            continue
        parent = mods.get(name.rsplit(".", 1)[0])
        if parent is None or type(parent).__name__ != "Gemma4Attention":
            continue
        if getattr(parent, "is_sliding", True) or getattr(parent, "is_kv_shared_layer", False):
            continue
        layer = int(hit.group(1))
        head_v = getattr(m, "head_size_v", None) or m.head_size
        m.register_forward_pre_hook(store.attn_hook(layer, m.num_heads, m.num_kv_heads, m.head_size),
                                    with_kwargs=True)
        m.register_forward_hook(store.out_hook(layer, m.num_heads, head_v))
        parent.register_forward_pre_hook(store.pos_hook(layer), with_kwargs=True)
        kn = getattr(parent, "k_norm", None)
        if kn is not None and isinstance(getattr(kn, "weight", None), torch.Tensor):
            knorm[layer] = kn.weight.detach().float().cpu().clone()
        hooked.append({"layer": layer, "name": name, "num_heads": m.num_heads, "num_kv_heads": m.num_kv_heads,
                       "head_size": m.head_size, "head_size_v": head_v,
                       "sliding_window": getattr(m, "sliding_window", None),
                       "kv_cache_dtype": str(getattr(m, "kv_cache_dtype", None)),
                       "k_norm": None if kn is None else {"class": type(kn).__name__,
                                                          "has_weight": getattr(kn, "has_weight", None),
                                                          "eps": getattr(kn, "variance_epsilon", None)}})
    hooked.sort(key=lambda h: h["layer"])
    return {"hooked": hooked, "k_norm": knorm}


def hook_problems(hooked: list[dict], expected: list[int], shapes: tuple[int, int, int] = (HQ, HK, D),
                  n_global: int | None = None) -> list[str]:
    """The fail-fast check before the first prompt: the hooked layers are the planned global layers, with the planned
    heads, KV heads and head size, and no sliding window."""
    bad = []
    if n_global is not None and len(expected) != n_global:
        bad.append(f"the config has {len(expected)} global layers with a KV cache, the plan {n_global}")
    if sorted(h["layer"] for h in hooked) != sorted(expected):
        bad.append(f"hooked layers {[h['layer'] for h in hooked]} are not the config global layers {expected}")
    for h in hooked:
        if (h["num_heads"], h["num_kv_heads"], h["head_size"]) != tuple(shapes):
            bad.append(f"layer {h['layer']}: heads {h['num_heads']}, KV heads {h['num_kv_heads']}, head size "
                       f"{h['head_size']}, expected {shapes[0]}, {shapes[1]}, {shapes[2]}")
        if h["sliding_window"] is not None:
            bad.append(f"layer {h['layer']}: sliding window {h['sliding_window']}")
    return bad


def check_prompt(data: dict, expected: list[int], length: int, q_rows: int, errors: list[str],
                 shapes: tuple[int, int, int] = (HQ, HK, D)) -> list[str]:
    """The fail-fast check of one prompt (and part of V2): every expected global layer has q and (A1) o
    [q_rows, heads, head_dim] and k and v [length, kv_heads, head_dim] in BF16, with the key positions 0 .. length - 1
    once each and the last q_rows positions."""
    hq, hk, d_ = shapes
    bad = list(errors)
    if not data:
        return bad + ["no global-layer tensors were recorded"]
    for layer in expected:
        d = data.get(layer)
        if d is None:
            bad.append(f"layer {layer}: no tensors")
            continue
        for n in ("q", "o"):
            if n not in d:
                bad.append(f"layer {layer}: no {n} rows")
            elif list(d[n].shape) != [q_rows, hq, d_]:
                bad.append(f"layer {layer}: {n} shape {list(d[n].shape)}, expected {[q_rows, hq, d_]}")
        for n in ("k", "v"):
            if list(d[n].shape) != [length, hk, d_]:
                bad.append(f"layer {layer}: {n} shape {list(d[n].shape)}, expected {[length, hk, d_]}")
        if d["dtype"] != "torch.bfloat16":
            bad.append(f"layer {layer}: dtype {d['dtype']}, expected torch.bfloat16")
        if not torch.equal(d["kpos"], torch.arange(length)):
            bad.append(f"layer {layer}: the key positions are not 0 .. {length - 1} once each")
        if not torch.equal(d["qpos"], torch.arange(length - q_rows, length)):
            bad.append(f"layer {layer}: the query positions are not the last {q_rows}")
    extra = sorted(set(data) - set(expected))
    if extra:
        bad.append(f"tensors of layers that are not global: {extra}")
    return bad


def save_prompt(cap_dir: Path, pid: int, data: dict) -> dict:
    """<cap_dir>/p<pid>/L<layer>.pt per global layer (BF16 q, k, v, (A1) o and the positions), with the SHA-256 of each
    file."""
    d = cap_dir / f"p{pid}"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for layer, t in sorted(data.items()):
        buf = io.BytesIO()
        torch.save({k: t[k] for k in ("q", "qpos", "k", "v", "kpos", "o") if k in t} | {"layer": layer,
                                                                                         "prompt": pid}, buf)
        b = buf.getvalue()
        (d / f"L{layer:02d}.pt").write_bytes(b)
        out[str(layer)] = {"sha256": sha(b), "bytes": len(b), "positions_from": t["src"]}
    return out


# ---------------------------------------------------------------- stub data


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).to(torch.float64)


def stub_layer(length: int, q_rows: int, w: torch.Tensor, gen: torch.Generator, args) -> dict:
    """Synthetic tensors built as in Gemma 4 and rounded as the engine rounds them: V = the weightless RMS norm n (BF16),
    K = RoPE(BF16(V * w)) in BF16, q = RoPE(a random query) in BF16, and (A1) o = the BF16 of the float32 attention
    (the engine output; --stub-swap-heads gives it a wrong head mapping)."""
    hq, hk = args.heads, args.kv_heads
    n = torch.randn(length, hk, D, generator=gen, dtype=torch.float64)
    v = bf(n / n.pow(2).mean(-1, keepdim=True).sqrt())
    kpos = torch.arange(length)
    k = bf(ca.rope(bf(v * bf(w)), kpos.double()[:, None].expand(length, hk)))
    qpos = torch.arange(length - q_rows, length)
    qr = torch.randn(q_rows, hq, D, generator=gen, dtype=torch.float64) * args.stub_q_scale / math.sqrt(D)
    if args.stub_q_max:
        qr[0, 0, ca.UNROT[0]] = args.stub_q_max                                  # one large query value
    q = bf(ca.rope(qr, qpos.double()[:, None].expand(q_rows, hq)))
    if args.stub_break_identity:                                                 # K is no longer w * V
        un = torch.tensor(ca.UNROT)
        k[..., un] = bf(k[..., un] * (1 + 0.2 * torch.randn(length, hk, len(ca.UNROT), generator=gen,
                                                            dtype=torch.float64)))
    if args.stub_big_k:                                                          # some K values above 448
        k[: max(1, length // 10), :, args.stub_big_k_dim] = args.stub_big_k
    if args.stub_nan:
        k[0, 0, 0] = float("nan")
    import analyze_e53 as an                                                    # the float32 attention

    qe, ke, ve = q.float(), k.float(), v.float()
    if args.stub_swap_heads:                                                     # a wrong head mapping
        if hk > 1:
            ke, ve = ke.roll(1, dims=1), ve.roll(1, dims=1)
        else:
            qe = qe.roll(1, dims=1)
    o = an.attend(qe, ke, ve, qpos).to(torch.bfloat16)
    return {"q": q.to(torch.bfloat16), "qpos": qpos, "k": k.to(torch.bfloat16), "v": v.to(torch.bfloat16),
            "kpos": kpos, "o": o, "src": ["positions"], "dtype": "torch.bfloat16"}


def toy_encode(text: str) -> list[int]:
    """The stub tokenizer: one id per byte (ids 3 to 258); the text "<bos>" becomes the special id 2, as a fast
    tokenizer does with a special-token string."""
    out: list[int] = []
    for i, part in enumerate(text.split("<bos>")):
        if i:
            out.append(2)
        out += [b + 3 for b in part.encode("utf-8")]
    return out


def mode_name(x) -> str:
    """(A1, R-73 F7) The name of an enum member (CompilationMode is an IntEnum, so str() gives "0"), else str()."""
    return getattr(x, "name", None) or str(x)


# ---------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="results folder (small files)")
    ap.add_argument("--capture", required=True, type=Path, help="capture folder (about 3.1 GB on the Pod)")
    ap.add_argument("--snapshot", type=Path, help="the pinned model snapshot folder (tokenizer)")
    ap.add_argument("--model-id", default="RedHatAI/gemma-4-31B-it-FP8-dynamic")
    ap.add_argument("--revision", default="d4ab4f579dd3516f97d8a6a4c98d0653480bad15")
    ap.add_argument("--prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--tokens", type=int, default=TOKENS)
    ap.add_argument("--q-rows", type=int, default=Q_ROWS)
    ap.add_argument("--seed", type=int, default=SEED)
    # (A1, R-73 F1) the shape flags: the defaults are the frozen 31B values; a rehearsal sets them and --rehearsal
    ap.add_argument("--heads", type=int, default=HQ)
    ap.add_argument("--kv-heads", type=int, default=HK)
    ap.add_argument("--head-dim", type=int, default=D, help="must be 512 (the views layout)")
    ap.add_argument("--global-layers", type=int, default=N_GLOBAL, help="global layers that own a KV cache")
    ap.add_argument("--rehearsal", action="store_true", help="the S5 rehearsal (a small model; not the frozen plan)")
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stub-gpu", default="STUB NVIDIA H100 80GB HBM3")
    ap.add_argument("--stub-layers", type=int, default=2, help="global layers in the stub (5, 11, ...)")
    ap.add_argument("--stub-tokens", type=int, default=640)
    ap.add_argument("--stub-q-rows", type=int, default=32)
    ap.add_argument("--stub-prompts", type=int, default=2)
    ap.add_argument("--stub-seed", type=int, default=0)
    ap.add_argument("--stub-text-root", type=Path, default=HERE.parent, help="text files for the stub prompts")
    ap.add_argument("--stub-w-form", choices=("weight", "one_plus_weight", "none"), default="weight",
                    help="how the module stores w in the stub")
    ap.add_argument("--stub-q-scale", type=float, default=2.0, help="the score spread of the stub queries")
    ap.add_argument("--stub-q-max", type=float, default=0.0, help="one query value of this size")
    ap.add_argument("--stub-big-k", type=float, default=0.0, help="K values of this size on one dim")
    ap.add_argument("--stub-big-k-dim", type=int, default=0, help="the dim of --stub-big-k (0 rotates, 64 does not)")
    ap.add_argument("--stub-break-identity", action="store_true")
    ap.add_argument("--stub-swap-heads", action="store_true", help="the stub engine output with a wrong head mapping")
    ap.add_argument("--stub-nan", action="store_true")
    ap.add_argument("--stub-no-tensors", action="store_true", help="the first prompt records nothing (fail fast)")
    ap.add_argument("--stub-fail", action="store_true", help="an error after the engine start")
    args = ap.parse_args()
    out, cap_dir = args.out, args.capture
    out.mkdir(parents=True, exist_ok=True)
    cap_dir.mkdir(parents=True, exist_ok=True)
    if args.stub:
        args.prompts, args.tokens, args.q_rows = args.stub_prompts, args.stub_tokens, args.stub_q_rows
        args.global_layers = args.stub_layers
    shapes = (args.heads, args.kv_heads, args.head_dim)
    record = {"args": {k: str(v) for k, v in vars(args).items()}, "started_at": now(), "status": "started",
              "stub": bool(args.stub), "rehearsal": bool(args.rehearsal), "model_id": args.model_id,
              "revision": args.revision,
              "plan": {"prompts": args.prompts, "tokens": args.tokens, "q_rows": args.q_rows, "seed": args.seed,
                       "heads": args.heads, "kv_heads": args.kv_heads, "head_dim": args.head_dim,
                       "global_layers": args.global_layers},
              "engine_kwargs": ENGINE, "prompts": [], "fail_fast": None}

    def save(status: str | None = None) -> None:
        if status:
            record["status"] = status
        (out / "capture_run.json").write_text(json.dumps(record, indent=1, default=str))

    def stop(problems: list[str]) -> None:
        record.update(fail_fast=problems, finished_at=now())
        save("fail_fast")
        print(json.dumps({"status": "fail_fast", "problems": problems[:10]}), flush=True)
        sys.exit(3)

    try:
        t0 = time.time()
        if args.head_dim != ca.HEAD:
            raise ValueError(f"--head-dim {args.head_dim}: the views layout needs {ca.HEAD}")
        if args.stub:
            record.update(gpu={"name": args.stub_gpu}, torch=torch.__version__, vllm="STUB")
            roots = {"vllm_py": [args.stub_text_root], "torch_py": [args.stub_text_root],
                     "docs": [args.stub_text_root]}
            base, encode, bos, special = args.stub_text_root, toy_encode, 2, TOY_SPECIAL
        else:
            record.update(gpu={"name": torch.cuda.get_device_name(0),
                               "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                               "uuid": str(getattr(torch.cuda.get_device_properties(0), "uuid", ""))},
                          torch=torch.__version__)
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(args.snapshot))
            roots, base = image_roots()
            encode, bos = (lambda s: tok.encode(s, add_special_tokens=False)), tok.bos_token_id
            special = frozenset(int(i) for i in tok.all_special_ids)
            record["tokenizer"] = {"class": type(tok).__name__, "bos_token_id": bos, "special_ids": sorted(special)}
        ppath = out / "prompts.json"
        prompts = None
        if ppath.exists():                                       # a rerun uses the same prompts
            old = json.loads(ppath.read_text())
            if old.get("plan") == record["plan"]:
                prompts = old["prompts"]
        if prompts is None:
            src = sources(roots)
            record["source_files"] = {k: len(v) for k, v in src.items()}
            prompts = build_prompts(src, encode, bos, base, seed=args.seed, n=args.prompts, tokens=args.tokens,
                                    special=special)
            ppath.write_text(json.dumps({"plan": record["plan"], "prompts": prompts}))
        record["prompts_built"] = [{k: p.get(k) for k in ("id", "family", "n_tokens", "token_sha256",
                                                          "special_after_0", "skipped_special")}
                                   | {"files": len(p["files"])} for p in prompts]
        record["phases"] = {"prompts": round(time.time() - t0, 1)}
        short = [p["id"] for p in prompts if p["n_tokens"] != args.tokens]
        if short:
            raise RuntimeError(f"prompts {short} have fewer than {args.tokens} tokens")
        save()

        if args.stub:
            expected = [5 + 6 * i for i in range(args.stub_layers)]
            record["config"] = {"global_layers": expected, "layer_types_count": len(expected),
                                "num_attention_heads": args.heads, "note": "stub"}
            gen = torch.Generator().manual_seed(args.stub_seed)
            ws = {layer: (1 + 0.1 * torch.randn(D, generator=gen, dtype=torch.float64)).to(torch.bfloat16).double()
                  for layer in expected}
            knorm = {} if args.stub_w_form == "none" else \
                {layer: (w if args.stub_w_form == "weight" else w - 1).float() for layer, w in ws.items()}
            record["hooks"] = {"route": "stub",
                               "hooked": [{"layer": layer, "name": f"stub.layers.{layer}.self_attn.attn",
                                           "num_heads": args.heads, "num_kv_heads": args.kv_heads, "head_size": D,
                                           "head_size_v": D, "sliding_window": None} for layer in expected]}
            if args.stub_fail:
                raise RuntimeError("stub failure (--stub-fail)")
        else:
            os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # the session sets it too (envs.py 1448)
            from vllm import LLM, SamplingParams
            import vllm

            record["vllm"] = vllm.__version__
            t = time.time()
            llm = LLM(model=args.model_id, revision=args.revision, **ENGINE)   # as E-50: the id and the pinned revision
            record["phases"]["engine"] = round(time.time() - t, 1)
            vc = llm.llm_engine.vllm_config
            record["engine"] = engine_record(vc)
            cfg = vc.model_config.hf_text_config
            layer_types = list(cfg.layer_types)
            expected = global_layers(layer_types, int(cfg.num_hidden_layers),
                                     int(getattr(cfg, "num_kv_shared_layers", 0) or 0))
            record["config"] = {"global_layers": expected, "layer_types_count": len(layer_types)} | {
                k: str(getattr(cfg, k, None)) for k in ("num_attention_heads", "num_key_value_heads",
                                                        "num_global_key_value_heads", "head_dim", "global_head_dim",
                                                        "attention_k_eq_v", "num_kv_shared_layers", "rms_norm_eps",
                                                        "rope_parameters")}
            store = Store(args.q_rows)
            try:
                reg = llm.apply_model(lambda model: register(model, store))[0]
                route = "LLM.apply_model"
            except Exception as exc:  # the attribute path of v1/engine/llm_engine.py 506 to 509
                record["apply_model_error"] = f"{type(exc).__name__}: {exc}"[:300]
                model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.get_model()
                reg, route = register(model, store), "model_executor.driver_worker.worker.model_runner"
            record["hooks"] = {"route": route, "hooked": reg["hooked"]}
            knorm = reg["k_norm"]
            params = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
        torch.save(knorm, cap_dir / "k_norm.pt")
        record["k_norm_layers"] = sorted(knorm)
        bad = hook_problems(record["hooks"]["hooked"], expected, shapes, args.global_layers)
        if bad:
            stop(bad)
        save()

        for p in prompts:
            length = p["n_tokens"]
            t = time.time()
            if args.stub:
                errors = []
                data = {} if (args.stub_no_tensors and p["id"] == 0) else \
                    {layer: stub_layer(length, args.q_rows, ws[layer], gen, args) for layer in expected}
                first = None
            else:
                store.begin(length)
                res = llm.generate([{"prompt_token_ids": p["token_ids"]}], params, use_tqdm=False)
                data, errors = store.end(), list(store.errors)
                first = list(res[0].outputs[0].token_ids[:1])
            problems = check_prompt(data, expected, length, args.q_rows, errors, shapes)
            entry = {"id": p["id"], "seconds": round(time.time() - t, 2), "first_token": first, "problems": problems}
            if problems and p["id"] == 0:
                record["prompts"].append(entry)
                stop(problems)
            entry["files"] = save_prompt(cap_dir, p["id"], data)
            entry["seconds_with_save"] = round(time.time() - t, 2)
            record["prompts"].append(entry)
            save()
            print(json.dumps({"prompt": p["id"], "layers": len(data), "problems": len(problems),
                              "seconds": entry["seconds_with_save"]}), flush=True)
        record["phases"]["total"] = round(time.time() - t0, 1)
        record.update(finished_at=now())
        save("ok")
    except SystemExit:
        raise
    except BaseException as exc:  # recorded, then re-raised so the session sees a non-zero exit
        record.update(error=f"{type(exc).__name__}: {exc}"[:2000], traceback=traceback.format_exc()[-4000:],
                      finished_at=now())
        save("error")
        raise
    print(json.dumps({"status": record["status"], "prompts": len(record["prompts"])}), flush=True)


def engine_record(vc) -> dict:
    """The engine facts that the README names: compile and CUDA graph modes (A1, F7: as names), the KV cache dtype,
    the dtype. V2 checks some of them (A1, F5)."""
    rec = {}
    for name, get in (("compilation_mode", lambda: mode_name(vc.compilation_config.mode)),
                      ("cudagraph_mode", lambda: mode_name(vc.compilation_config.cudagraph_mode)),
                      ("cache_dtype", lambda: str(vc.cache_config.cache_dtype)),
                      ("block_size", lambda: vc.cache_config.block_size),
                      ("model_dtype", lambda: str(vc.model_config.dtype)),
                      ("max_model_len", lambda: vc.model_config.max_model_len),
                      ("max_num_batched_tokens", lambda: vc.scheduler_config.max_num_batched_tokens),
                      ("quantization", lambda: str(vc.model_config.quantization)),
                      ("architectures", lambda: list(vc.model_config.architectures)),
                      ("multiprocessing_env", lambda: os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"))):
        try:
            rec[name] = get()
        except Exception as exc:  # reported only
            rec[name + "_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


if __name__ == "__main__":
    main()
