"""E-54 kernel arm (README Section 8; secondary, R-74 M5): does the stock FA4 FP8 kernel of the pinned vLLM image read
the scale-tied compact row through its two views with k_descale not equal to v_descale, as the emulation says?

For the test prompts 0 (8K) and 6 (32K) and every global layer, the script builds three paged FP8 caches from the
captured K and V with the scales that the analysis used (metrics.json): STOCK (the standard cache, descales 1),
CALSTD (the standard cache, the calibrated descales) and TIED (the 640-value rows: the K view is codes 0 to 511 with
k_descale ks and the plain permuted query, the V view is codes 128 to 639 with v_descale vs). Each of the 256 captured
query rows is one decode sequence: all sequences share the same pages, and sequence i sees the keys 0 .. qpos[i]. The
cache is the engine tensor [blocks, KV heads, 64, values] read through .transpose(1, 2), and the call is the FA4 call of
vLLM FlashAttentionImpl.forward, as E-52 (`kbench_e52.stock_call`, `views`).

The gap of a cell is ||o_kernel - o_emu|| / ||o_emu||, where o_emu is the FP16 emulation of the same arm
(`analyze_e54`). The kernel writes BF16, so a correct read gives about 0.002 (the E-53 engine gap was 0.0017); a lost
or swapped descale gives a gap near 1 (the tests). Outcome: MATCH (every gap at most 0.005), CLOSE (every gap at most
0.05), MISMATCH (a gap above 0.05), NOT RUN (no SM90 GPU or no FA4 module; the reason is recorded), ERROR (an
exception; the reason is recorded). The outcome does not change the primary decision (README Section 7).
--standin replaces FA4 with a float32 CPU paged attention (the Mac tests: the same caches, views, block table and
descales).

Usage: kernel_e54.py --res <dir> --capture <dir> [--standin] [--prompts 0,6] [--swap-descales]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze_e54 as a54  # noqa: E402
from analyze_e54 import an, ca  # noqa: E402

BS = 64                                 # the block size of E-52 (kbench_e52.BS)
MATCH_TOL, CLOSE_TOL = 0.005, 0.05
FP8 = torch.float8_e4m3fn
ARMS = ("STOCK", "CALSTD", "TIED")


def paged(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """codes [L, Hk, X] (FP8 code values as float) -> the engine tensor [blocks, Hk, BS, X] in FP8 and the block count.
    The kernel reads it through .transpose(1, 2) as [blocks, BS, Hk, X] (E-52)."""
    length, hk, x = codes.shape
    nblk = math.ceil(length / BS)
    buf = torch.zeros(nblk * BS, hk, x, dtype=FP8, device=codes.device)
    buf[:length] = codes.to(FP8)
    return buf.view(nblk, BS, hk, x).transpose(1, 2).contiguous(), nblk


def arm_cache(k, v, arm: str, ks: float, vs: float):
    """(K view, V view, block count, permuted query) of one arm; the views are [blocks, BS, Hk, 512]."""
    if arm == "TIED":
        eng, nblk = paged(a54.tied_codes(k, v, ks, vs))
        kview, vview = ca.views_of(eng.transpose(1, 2))
        return kview, vview, nblk, True
    eng, nblk = paged(torch.cat([an.fp8(k / ks), an.fp8(v / vs)], -1))
    kk, vv = eng.transpose(1, 2).split(ca.HEAD, dim=-1)
    return kk, vv, nblk, False


def fa4_call(q, kc, vc, bt, seqs, max_k: int, kd, vd) -> torch.Tensor:
    """The FA4 call of vLLM FlashAttentionImpl.forward (3966924d5) for a decode step, as E-52."""
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    out = torch.empty_like(q)
    cu = torch.arange(q.shape[0] + 1, dtype=torch.int32, device=q.device)
    flash_attn_varlen_func(q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=1, seqused_k=seqs,
                           max_seqlen_k=max_k, softmax_scale=1.0, causal=True, alibi_slopes=None, window_size=[-1, -1],
                           block_table=bt, softcap=0.0, scheduler_metadata=None, fa_version=4, q_descale=None,
                           k_descale=kd, v_descale=vd, dynamic_causal=None, num_splits=0, s_aux=None, mask_mod=None,
                           aux_tensors=None)
    return out


def standin_call(q, kc, vc, bt, seqs, max_k: int, kd, vd) -> torch.Tensor:
    """The CPU stand-in: per sequence, the keys 0 .. seqs[i] - 1 from the pages times the descale of each KV head, q and
    P in FP16 (as the emulation), the output rounded to BF16 (the kernel writes BF16)."""
    out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    kf, vf = kc.float(), vc.float()                                  # (CPU indexing of FP8 tensors is limited)
    for i in range(q.shape[0]):
        n = int(seqs[i])
        kk = ca.gather(kf, bt, i, n) * kd[i][None, :, None]
        vv = ca.gather(vf, bt, i, n) * vd[i][None, :, None]
        o = an.attend(an.fp16(q[i:i + 1].float()), kk, vv, torch.tensor([n - 1], device=q.device), torch.float16)
        out[i] = o[0].to(torch.bfloat16)
    return out


def cell(cap: dict, sc: dict, device, call, swap: bool = False) -> dict:
    """The three arms of one prompt and layer: the gap to the emulation and the errors against the reference."""
    q, k, v = (cap[n].to(device).float() for n in ("q", "k", "v"))
    qpos = cap["qpos"].to(device).long()
    nq, hk = q.shape[0], k.shape[1]
    ref = an.attend(q, k, v, qpos)
    scales = {"STOCK": (1.0, 1.0), "CALSTD": (sc["calstd"]["ks"], sc["calstd"]["vs"]),
              "TIED": (sc["tied"]["ks"], sc["tied"]["vs"])}
    out = {}
    for arm in ARMS:
        ks, vs = scales[arm]
        kc, vc, nblk, perm = arm_cache(k, v, arm, ks, vs)
        qq = (ca.views_query(q, torch.ones(ca.HEAD, device=device)) if perm else q).to(torch.bfloat16).contiguous()
        bt = torch.arange(nblk, dtype=torch.int32, device=device).repeat(nq, 1)
        seqs = (qpos + 1).to(torch.int32)
        dk, dv = (vs, ks) if swap else (ks, vs)                      # --swap-descales: the negative control
        kd = torch.full((nq, hk), dk, dtype=torch.float32, device=device)
        vd = torch.full((nq, hk), dv, dtype=torch.float32, device=device)
        o = call(qq, kc, vc, bt, seqs, k.shape[0], kd, vd).float()
        o = ca.unpermute(o) if perm else o
        emu = (a54.tied_arm(q, k, v, qpos, ks, vs, fp16=True) if arm == "TIED"
               else a54.std_arm(q, k, v, qpos, ks, vs, fp16=True))
        out[arm] = {"gap": an.rel(o, emu), "err_kernel": an.rel(o, ref), "err_emu16": an.rel(emu, ref),
                    "ks": ks, "vs": vs, "finite": bool(torch.isfinite(o).all())}
    return out


def outcome(cells: dict) -> tuple[str, dict]:
    gaps = {a: max((c[a]["gap"] if c[a]["finite"] else math.inf) for c in cells.values()) for a in ARMS}
    worst = max(gaps.values())
    word = "MATCH" if worst <= MATCH_TOL else "CLOSE" if worst <= CLOSE_TOL else "MISMATCH"
    return word, gaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True, type=Path, help="results folder (metrics.json; kernel.json goes here)")
    ap.add_argument("--capture", required=True, type=Path)
    ap.add_argument("--prompts", default="0,6")
    ap.add_argument("--standin", action="store_true", help="a CPU stand-in for FA4 (the Mac tests)")
    ap.add_argument("--swap-descales", action="store_true", help="the negative control (tests)")
    args = ap.parse_args()
    rec = {"started_at": a54.now(), "standin": args.standin, "prompts": args.prompts, "cells": {},
           "tolerances": {"match": MATCH_TOL, "close": CLOSE_TOL}, "swap_descales": args.swap_descales}
    path = args.res / "kernel.json"

    def save(**kw) -> None:
        rec.update(kw)
        path.write_text(json.dumps(rec, indent=1, default=str))

    try:
        if args.standin:
            device, call = "cpu", standin_call
        else:
            if not torch.cuda.is_available():
                return save(outcome="NOT RUN", reason="no CUDA device", finished_at=a54.now())
            cap_ = torch.cuda.get_device_capability(0)
            rec["gpu"] = {"name": torch.cuda.get_device_name(0), "capability": ".".join(map(str, cap_))}
            if cap_ < (9, 0):
                return save(outcome="NOT RUN", reason=f"capability {cap_}: the FA4 FP8 path needs SM90",
                            finished_at=a54.now())
            try:
                import vllm
                import vllm.vllm_flash_attn as vfa
                rec["vllm"] = vllm.__version__
                rec["fa_module"] = getattr(vfa, "__file__", None)
            except Exception as exc:
                return save(outcome="NOT RUN", reason=f"no FA4 module: {type(exc).__name__}: {exc}"[:300],
                            finished_at=a54.now())
            device, call = "cuda", fa4_call
        scales = json.loads((args.res / "metrics.json").read_text())["scales"]
        t0 = time.time()
        for pid in [int(x) for x in args.prompts.split(",")]:
            for f in sorted((args.capture / f"p{pid}").glob("L*.pt")):
                layer = f.stem[1:].lstrip("0") or "0"
                cap = torch.load(f, map_location=device, weights_only=True)
                rec["cells"][f"p{pid} L{layer}"] = cell(cap, scales[layer], device, call, args.swap_descales)
                save(seconds=round(time.time() - t0, 1))
        word, gaps = outcome(rec["cells"]) if rec["cells"] else ("ERROR", {})
        save(outcome=word, max_gap=gaps, reason="" if rec["cells"] else "no capture files", finished_at=a54.now())
    except Exception as exc:  # recorded; the primary decision does not depend on it
        save(outcome="ERROR", reason=f"{type(exc).__name__}: {exc}"[:500], traceback=traceback.format_exc()[-3000:],
             finished_at=a54.now())
    print(json.dumps({k: rec.get(k) for k in ("outcome", "max_gap", "reason")}, default=str), flush=True)


if __name__ == "__main__":
    main()
