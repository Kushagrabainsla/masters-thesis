"""Contract levels for SafeDispatch (RESEARCH_JOURNAL.md, Section 5.8).

L1 tensor level: bitwise equality and distance in units of the dtype's own spacing.
L2 sequence level: greedy argmax flips and reference logit margins.
L3 logit level: KL divergence and max absolute logit difference.

Everything here is a pure function on tensors so it can be unit tested without a model.
"""
import torch

# Integer view of each float dtype. Same width, so a bit pattern maps one to one.
_INT_VIEW = {
    torch.float64: torch.int64,
    torch.float32: torch.int32,
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
}


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(_INT_VIEW[t.dtype])


def bitwise_diff_count(a: torch.Tensor, b: torch.Tensor) -> int:
    """Number of elements whose bit patterns differ. NaN payloads are compared by bits."""
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    return int((_bits(a) != _bits(b)).sum().item())


def ordered_int(t: torch.Tensor) -> torch.Tensor:
    """Map floats to int64 whose order matches float order.

    Adjacent representable floats map to adjacent integers, so the difference of two
    mapped values is the distance in spacings (ULPs of that dtype). +0 and -0 both map
    to 0. A float with the sign bit set has two's complement value `min + m` where m is
    its magnitude bits, so `min - x` recovers `-m`.
    """
    it = _INT_VIEW[t.dtype]
    x = _bits(t).to(torch.int64)
    minv = torch.iinfo(it).min
    return torch.where(x < 0, minv - x, x)


def spacing_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.dtype == b.dtype
    return (ordered_int(a) - ordered_int(b)).abs()


def spacing_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    d = spacing_distance(a, b)
    n = d.numel()
    n_diff = int((d > 0).sum().item())
    return {
        "n": n,
        "n_diff": n_diff,
        "frac_diff": n_diff / n if n else 0.0,
        "max_spacing": int(d.max().item()) if n else 0,
    }


def allclose_gate(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float = 0.0) -> bool:
    """A conventional fixed-tolerance gate. The blind-spot baseline for L1."""
    return bool(torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol))


# ---- L2 ---------------------------------------------------------------------------

def top2_margin(logits: torch.Tensor) -> torch.Tensor:
    """Top1 minus top2 logit, per row. logits: [T, V] float32."""
    top2 = logits.topk(2, dim=-1).values
    return top2[:, 0] - top2[:, 1]


def first_true(mask: torch.Tensor):
    """Index of the first True, or None."""
    idx = torch.nonzero(mask, as_tuple=False)
    return int(idx[0].item()) if idx.numel() else None


# ---- L3 ---------------------------------------------------------------------------

def kl_ref_to_cand(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> torch.Tensor:
    """Per-row KL(p_ref || p_cand) in float32. Rows are positions."""
    lp_r = torch.log_softmax(ref_logits.float(), dim=-1)
    lp_c = torch.log_softmax(cand_logits.float(), dim=-1)
    return (lp_r.exp() * (lp_r - lp_c)).sum(dim=-1)


def max_abs_logit_diff(ref_logits: torch.Tensor, cand_logits: torch.Tensor) -> torch.Tensor:
    return (ref_logits.float() - cand_logits.float()).abs().amax(dim=-1)
