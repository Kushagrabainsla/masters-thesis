"""Emulated INT8 GEMM with fault-injected epilogues, for C2 in the journal (E-23).

Why this exists: arXiv 2608.13756 and 2609.00363 report that two INT8 GEMM kernels with
identical interfaces diverge only in the epilogue (scale application and output rounding),
because the int32 accumulator is exact. A tolerance-based gate at one bf16 spacing cannot
see such faults. This module builds controlled faults of that class on the CPU so the
harness can be shown to detect them before any GPU time is spent.

Every fault is specified so it differs from the reference by at most one bf16 spacing,
except F4, which is reported without an expectation.

The faults here are my own constructions in the same fault class. They are not a
reproduction of the five faults in 2609.00363.
"""
import torch

BF16_MASK_LOW16 = -65536  # 0xFFFF0000 as int32


def make_problem(m: int, k: int, n: int, seed: int):
    """Quantized activations (per tensor scale) and weights (per channel scales)."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(m, k, generator=g)
    w = torch.randn(k, n, generator=g) * 0.05
    sa = (a.abs().max() / 127.0).float()
    sw = (w.abs().amax(dim=0) / 127.0).float()
    aq = torch.clamp(torch.round(a / sa), -127, 127).to(torch.int8)
    wq = torch.clamp(torch.round(w / sw), -127, 127).to(torch.int8)
    return aq, wq, sa, sw


def acc_exact(aq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """Exact integer accumulation. |acc| <= 127*127*K < 2^53, so float64 is exact."""
    return aq.double() @ wq.double()


def acc_int32(aq: torch.Tensor, wq: torch.Tensor) -> torch.Tensor:
    """Independent implementation of the same exact accumulation, for the F0 control."""
    return (aq.to(torch.int32) @ wq.to(torch.int32)).double()


def truncate_to_bf16(x32: torch.Tensor) -> torch.Tensor:
    """Round toward zero when casting fp32 to bf16 (drop the low 16 bits)."""
    bits = x32.contiguous().view(torch.int32) & BF16_MASK_LOW16
    return bits.view(torch.float32).to(torch.bfloat16)


def epilogue(acc: torch.Tensor, sa: torch.Tensor, sw: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert the exact accumulator to bf16 output.

    F0 reference: bf16_rne(fp32(acc) * (sa * sw)), scale product in fp32.
    F1 scale application order: (fp32(acc) * sa) * sw.
    F2 truncation instead of round to nearest even at the bf16 cast.
    F3 double rounding through fp16.
    F4 scales stored in bf16 instead of fp32.
    """
    acc32 = acc.to(torch.float32)
    if mode in ("F0", "F2", "F3"):
        out32 = acc32 * (sa * sw)
    elif mode == "F1":
        out32 = (acc32 * sa) * sw
    elif mode == "F4":
        sa_q = sa.to(torch.bfloat16).to(torch.float32)
        sw_q = sw.to(torch.bfloat16).to(torch.float32)
        out32 = acc32 * (sa_q * sw_q)
    else:
        raise ValueError(mode)
    if mode == "F2":
        return truncate_to_bf16(out32)
    if mode == "F3":
        return out32.to(torch.float16).to(torch.bfloat16)
    return out32.to(torch.bfloat16)


FAULTS = ("F1", "F2", "F3", "F4")
