"""A small CPU emulation of the parts of `triton` and `triton.language` that `compact_attn.py` uses (from the R-69
review, 2026-10-31; A1 removes its pointer cache, so a new tensor at a reused address is never read stale).

`install()` puts the emulation in `sys.modules`; then the SAME Python source of the @triton.jit kernels runs program
by program on torch CPU tensors. It checks the algorithm (strides, masks, the split rule, the online softmax, the
merge, the scales). It does NOT check the Triton compile, shared memory, registers or speed.
"""

from __future__ import annotations

import itertools
import sys
import types

import torch


class Ptr:
    """A pointer: a flat view of a tensor's whole storage and an element offset (an int or an int tensor)."""

    def __init__(self, flat: torch.Tensor, off, dtype):
        self.flat, self.off, self.dtype = flat, off, dtype

    def __add__(self, other):
        o = other.to(torch.int64) if isinstance(other, torch.Tensor) else other
        return Ptr(self.flat, self.off + o, self.dtype)

    __radd__ = __add__


class DType:
    def __init__(self, t):
        self.element_ty = t


def wrap(t: torch.Tensor, writable: bool) -> Ptr:
    """A pointer to element 0 of the view t. Read-only FP8 storage is read as a float32 copy (CPU FP8 indexing is
    limited); outputs are written in place."""
    st = t.untyped_storage()
    base = torch.empty(0, dtype=t.dtype).set_(st, 0, (st.nbytes() // t.element_size(),), (1,))
    if not writable and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        base = base.float()
    return Ptr(base, t.storage_offset(), DType(t.dtype))


class _Constexpr:
    pass


class _TL(types.ModuleType):
    constexpr = _Constexpr
    float32 = torch.float32
    bfloat16 = torch.bfloat16
    int64 = torch.int64
    int32 = torch.int32

    def __init__(self):
        super().__init__("triton.language")
        self._pid = (0, 0, 0)

    def program_id(self, axis):
        return torch.tensor(self._pid[axis], dtype=torch.int32)

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b

    @staticmethod
    def minimum(a, b):
        return torch.minimum(torch.as_tensor(a), torch.as_tensor(b))

    @staticmethod
    def maximum(a, b):
        return torch.maximum(torch.as_tensor(a), torch.as_tensor(b))

    @staticmethod
    def arange(a, b):
        return torch.arange(a, b, dtype=torch.int32)

    @staticmethod
    def full(shape, v, dtype):
        return torch.full(tuple(shape), v, dtype=dtype)

    @staticmethod
    def zeros(shape, dtype):
        return torch.zeros(tuple(shape), dtype=dtype)

    @staticmethod
    def trans(x):
        return x.T

    @staticmethod
    def dot(a, b):
        assert a.shape[0] >= 16 and a.shape[1] >= 16 and b.shape[1] >= 16, (a.shape, b.shape)
        assert a.dtype == b.dtype, (a.dtype, b.dtype)
        return a.float() @ b.float()

    @staticmethod
    def where(c, a, b):
        return torch.where(c, a, b)

    @staticmethod
    def max(x, axis):
        return x.max(dim=axis).values

    @staticmethod
    def sum(x, axis):
        return x.sum(axis)

    @staticmethod
    def exp(x):
        return torch.exp(x)

    @staticmethod
    def log(x):
        return torch.log(x)

    @staticmethod
    def load(p: Ptr, mask=None, other=None):
        off = p.off if isinstance(p.off, torch.Tensor) else torch.tensor(p.off, dtype=torch.int64)
        if mask is None:
            return p.flat[off]
        off, mask = torch.broadcast_tensors(off, torch.as_tensor(mask))
        val = p.flat[torch.where(mask, off, torch.zeros_like(off)).to(torch.int64)]
        fill = torch.zeros((), dtype=val.dtype) if other is None else torch.tensor(other, dtype=val.dtype)
        return torch.where(mask, val, fill)

    @staticmethod
    def store(p: Ptr, val, mask=None):
        off = p.off if isinstance(p.off, torch.Tensor) else torch.tensor(p.off, dtype=torch.int64)
        val = torch.as_tensor(val)
        if mask is None:
            off, val = torch.broadcast_tensors(off, val)
            p.flat[off.reshape(-1).to(torch.int64)] = val.reshape(-1).to(p.flat.dtype)
            return
        off, val, mask = torch.broadcast_tensors(off, val, torch.as_tensor(mask))
        sel = mask.reshape(-1)
        p.flat[off.reshape(-1)[sel].to(torch.int64)] = val.reshape(-1)[sel].to(p.flat.dtype)


tl = _TL()


class _Jit:
    def __init__(self, fn):
        self.fn = fn
        self.params = list(fn.__code__.co_varnames[: fn.__code__.co_argcount])

    def __getitem__(self, grid):
        def launch(*args, num_warps=None, num_stages=None, **kw):
            call = dict(zip(self.params, args)) | kw
            for name, a in list(call.items()):
                if isinstance(a, torch.Tensor):           # the outputs (OP, LP, O) are written; the rest is read
                    call[name] = wrap(a, writable=name in ("OP", "LP", "O"))
            gx, gy, gz = (tuple(grid) + (1, 1))[:3]
            for pid in itertools.product(range(gx), range(gy), range(gz)):
                tl._pid = pid
                self.fn(**call)

        return launch


def install() -> None:
    tr = types.ModuleType("triton")
    tr.jit = lambda fn: _Jit(fn)
    tr.next_power_of_2 = lambda n: 1 << (n - 1).bit_length()
    tr.language = tl
    sys.modules["triton"] = tr
    sys.modules["triton.language"] = tl
