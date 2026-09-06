"""

Complements ``functional.py`` (codegen) and ``_shape_funcs.py``. Every
public name here is expressed over existing differentiable primitives
(mm/cat/narrow/gather/index_select/pow/sort/searchsorted/bincount/...),
so autograd flows through composition wherever the underlying ops are
``test/test_composite_funcs.py``.

Deliberate narrows (documented per function):
- conjugation-bit queries report physical layout (tp has no conj bit);
- ``cartesian_prod`` accepts 1-D inputs.
"""

import itertools
import math

import numpy as np

import tensorplay
from tensorplay._C import DType
from tensorplay.autograd.function import Function
from tensorplay.graph import capture_call as _capture_call

__all__ = [
    "absolute", "arccos", "arccosh", "arcsin", "arcsinh", "arctan",
    "arctanh", "arctan2",
    "acos_", "asin_", "atan_", "acosh_", "asinh_", "atanh_",
    "concat", "concatenate", "ger", "adjoint",
    "detach", "diagflat", "numel", "scalar_tensor",
    "chain_matmul", "matrix_power", "kron", "vander",
    "tril_indices", "triu_indices", "cartesian_prod", "combinations",
    "trapezoid", "trapz", "cumulative_trapezoid",
    "quantile", "nanquantile", "histc", "histogram",
    "isin", "unique_consecutive", "repeat_interleave", "kaiser_window",
    "lstm_cell", "rnn_relu_cell", "rnn_tanh_cell",
    "put", "resolve_conj", "resolve_neg", "is_conj", "is_neg",
    "can_cast", "promote_types", "result_type", "is_nonzero",
    "is_same_size", "get_device",
    "alias_copy", "t_copy", "permute_copy", "transpose_copy",
    "squeeze_copy", "unsqueeze_copy", "select_copy", "slice_copy",
    "narrow_copy", "diagonal_copy", "unbind_copy", "split_copy",
    "view_copy", "unfold_copy", "expand_copy", "reshape_as",
    "unsafe_chunk", "unsafe_split",
    "rms_norm", "cosine_similarity",
    "max_pool1d", "avg_pool1d", "adaptive_avg_pool1d",
    "adaptive_max_pool1d",
    "dropout", "dropout_", "alpha_dropout", "feature_dropout",
    "feature_dropout_", "feature_alpha_dropout", "rrelu", "rrelu_",
    "bilinear", "ctc_loss", "embedding_bag", "conv_tbc",
    "max_pool3d", "max_pool1d_with_indices", "native_channel_shuffle",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm_dim(dim, ndim):
    d = int(dim)
    if d < 0:
        d += ndim
    if d < 0 or d >= ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of "
            f"[{-ndim}, {ndim - 1}], but got {dim})"
        )
    return d


def _as_tensor(x):
    if isinstance(x, tensorplay.Tensor):
        return x
    return tensorplay.as_tensor(x)


def _band(a, b):
    return tensorplay.logical_and(a, b)


def _slice_getitem(t, dim, start, end, step=1):
    sl = [slice(None)] * t.dim()
    sl[dim] = slice(int(start), int(end), int(step))
    return t[tuple(sl)]


def _make_arc_inplace(base, deriv):
    class _ArcInplaceFn(Function):
        @staticmethod
        def forward(ctx, self):
            ctx.pre = self.clone()
            res = getattr(self, base)()
            with tensorplay.no_grad():
                self.copy_(res)
            return self

        @staticmethod
        def backward(ctx, grad_output):
            return (grad_output * deriv(ctx.pre),)

    def op(self):
        return _ArcInplaceFn.apply(self)

    op.__name__ = base + "_"
    return op


def _d_acos(x):
    return tensorplay.full_like(x, 1.0).sub_(x.mul(x)).rsqrt().neg()


def _d_asin(x):
    return tensorplay.full_like(x, 1.0).sub_(x.mul(x)).rsqrt()


def _d_atan(x):
    return x.mul(x).add(1.0).reciprocal()


def _d_acosh(x):
    return x.mul(x).sub(1.0).rsqrt()


def _d_asinh(x):
    return x.mul(x).add(1.0).rsqrt()


def _d_atanh(x):
    return x.neg().mul(x).add(1.0).reciprocal()


acos_ = _make_arc_inplace("acos", _d_acos)
asin_ = _make_arc_inplace("asin", _d_asin)
atan_ = _make_arc_inplace("atan", _d_atan)
acosh_ = _make_arc_inplace("acosh", _d_acosh)
asinh_ = _make_arc_inplace("asinh", _d_asinh)
atanh_ = _make_arc_inplace("atanh", _d_atanh)


# ---------------------------------------------------------------------------
# aliases / semantic wrappers
# ---------------------------------------------------------------------------

def absolute(input):
    """Alias of :func:`abs`."""
    return input.abs()


def arccos(input):
    return input.acos()


def arcsin(input):
    return input.asin()


def arctan(input):
    return input.atan()


def arccosh(input):
    return input.acosh()


def arcsinh(input):
    return input.asinh()


def arctanh(input):
    return input.atanh()


def arctan2(input, other):
    """atan2(y, x) = 2*atan(y / (hypot + x)); native atan2 CPU kernel is
    currently unregistered in this tree, so compose. Edge narrows vs
"""
    y = input if isinstance(input, tensorplay.Tensor) else _as_tensor(input)
    x = other if isinstance(other, tensorplay.Tensor) else _as_tensor(other)
    xd = x.to(DType.float64)
    yd = y.to(DType.float64)
    r = xd.mul(xd).add(yd.mul(yd)).sqrt()
    den = r.add(xd)
    den_safe = tensorplay.where(
        den.eq(0), tensorplay.full_like(den, 1e-300), den)
    base = yd.div(den_safe).atan().mul(2.0)
    neg_half_pi = -math.pi / 2.0
    left = tensorplay.where(
        yd.lt(0),
        tensorplay.full_like(base, neg_half_pi),
        tensorplay.full_like(base, math.pi / 2.0),
    )
    out = tensorplay.where(_band(xd.lt(0), yd.eq(0)), left, base)
    od = _float_out_dtype(input.dtype)
    return out.to(od)


def concat(tensors, dim=0, *, out=None):
    result = tensorplay.cat(tensors=list(tensors), dim=dim)
    if out is not None:
        out.copy_(result)
        return out
    return result


def concatenate(tensors, dim=0, *, out=None):
    return concat(tensors, dim=dim, out=out)


def ger(input, vec2):
    return input.outer(vec2)


def adjoint(input):
    out = input.transpose(-2, -1)
    if input.dtype in (DType.complex64, DType.complex128,
                       DType.bcomplex32, DType.complex32):
        out = tensorplay.conj(out)
    return out


def detach(input):
    return input.detach()


def diagflat(input, offset=0):
    return input.reshape([-1]).diag(offset)


def numel(obj):
    return obj.numel()


def scalar_tensor(s, *, dtype=None, device=None):
    t = tensorplay.tensor([s], dtype=dtype, device=device)
    return t.reshape([])


# ---------------------------------------------------------------------------
# linear algebra / structured
# ---------------------------------------------------------------------------

def chain_matmul(*matrices):
    if len(matrices) == 0:
        raise RuntimeError("chain_matmul(): expected at least one matrix")
    out = matrices[0]
    for m in matrices[1:]:
        out = out.mm(m)
    return out


def matrix_power(input, n):
    n = int(n)
    if input.dim() != 2:
        raise RuntimeError("matrix_power(): expected a 2-D square matrix")
    if input.size(0) != input.size(1):
        raise RuntimeError("matrix_power(): expected a square matrix")
    result = tensorplay.eye(input.size(0), dtype=input.dtype,
                            device=input.device)
    base = tensorplay.linalg.inv(input) if n < 0 else input
    e = -n if n < 0 else n
    while e > 0:
        if e & 1:
            result = result.mm(base)
        e >>= 1
        if e:
            base = base.mm(base)
    return result


def kron(input, other):
    a = input if input.dim() >= 1 else input.reshape([1])
    b = other if other.dim() >= 1 else other.reshape([1])
    if a.dim() != b.dim():
        raise RuntimeError(
            f"kron(): number of dimensions must match, got {a.dim()} "
            f"and {b.dim()}"
        )
    ndim = a.dim()
    ashape = list(a.size())
    bshape = list(b.size())
    pa = []
    pb = []
    merged = []
    for s_a, s_b in zip(ashape, bshape):
        pa.extend([s_a, 1])
        pb.extend([1, s_b])
        merged.append(s_a * s_b)
    out = a.reshape(pa).mul(b.reshape(pb))
    return out.reshape(merged)


def vander(x, N=None, increasing=False):
    if x.dim() != 1:
        raise RuntimeError("vander(): expected a 1-D input tensor")
    n = x.numel() if N is None else int(N)
    if n < 0:
        raise ValueError("vander(): N must be non-negative")
    exps = tensorplay.arange(n, device=x.device)
    if not increasing:
        exps = tensorplay.flip(exps, dims=[0])
    return x.unsqueeze(1).pow(exps.unsqueeze(0))


def tril_indices(row, col, offset=0, *, dtype=DType.int64, device=None,
                 pin_memory=False):
    row, col, offset = int(row), int(col), int(offset)
    if row < 0 or col < 0:
        raise ValueError("tril_indices(): row and col must be non-negative")
    rr = tensorplay.arange(row, device=device).unsqueeze(1).repeat([1, col])
    cc = tensorplay.arange(col, device=device).unsqueeze(0).repeat([row, 1])
    mask = cc.le(rr + offset)
    r_flat = rr.masked_select(mask)
    c_flat = cc.masked_select(mask)
    out = tensorplay.stack([r_flat, c_flat], dim=0)
    if dtype != DType.int64:
        out = out.to(dtype)
    if pin_memory:
        out = out.pin_memory()
    return out


def triu_indices(row, col, offset=0, *, dtype=DType.int64, device=None,
                 pin_memory=False):
    row, col, offset = int(row), int(col), int(offset)
    if row < 0 or col < 0:
        raise ValueError("triu_indices(): row and col must be non-negative")
    rr = tensorplay.arange(row, device=device).unsqueeze(1).repeat([1, col])
    cc = tensorplay.arange(col, device=device).unsqueeze(0).repeat([row, 1])
    mask = cc.ge(rr + offset)
    r_flat = rr.masked_select(mask)
    c_flat = cc.masked_select(mask)
    out = tensorplay.stack([r_flat, c_flat], dim=0)
    if dtype != DType.int64:
        out = out.to(dtype)
    if pin_memory:
        out = out.pin_memory()
    return out


def cartesian_prod(*tensors):
    if len(tensors) == 0:
        raise RuntimeError("cartesian_prod(): expected at least one tensor")
    ts = [t if t.dim() >= 1 else t.reshape([1]) for t in tensors]
    if any(t.dim() != 1 for t in ts):
        raise NotImplementedError(
            "cartesian_prod(): only 1-D inputs are supported"
        )
    grids = tensorplay.meshgrid(list(ts), indexing="ij")
    cols = [g.reshape([-1, 1]).to(ts[0].dtype) for g in grids]
    return tensorplay.cat(cols, dim=1)



def combinations(input, r=2, with_replacement=False):
    n = input.size(0)
    r = int(r)
    if r < 0:
        raise ValueError("combinations(): r must be non-negative")
    rr = r
    gen = (itertools.combinations_with_replacement(range(n), r)
           if with_replacement else itertools.combinations(range(n), r))
    idx_list = list(gen)
    tail = list(input.size())[1:]
    if rr == 0:
        return tensorplay.zeros([0] + tail, dtype=input.dtype,
                                device=input.device)
    if not idx_list:
        return tensorplay.zeros([0] + tail, dtype=input.dtype,
                                device=input.device)
    idx = tensorplay.tensor(idx_list, dtype=DType.int64, device=input.device)
    picked = input.index_select(0, idx.reshape([-1]))
    return picked.reshape([idx.size(0), rr] + tail)


# ---------------------------------------------------------------------------
# statistics / numerics
# ---------------------------------------------------------------------------

_INT_DTYPES = (DType.uint8, DType.int8, DType.int16, DType.int32,
               DType.int64)


def _float_out_dtype(dt):
    return DType.float64 if dt == DType.float64 else DType.float32


def trapezoid(y, x=None, *, dx=None, dim=-1):
    d = _norm_dim(dim, y.dim())
    n = y.size(d)
    if n == 0:
        if x is not None:
            xs = _as_tensor(x)
            if xs.dim() == 1 and xs.numel() != 0:
                raise RuntimeError(
                    "trapezoid(): there must be one x value for every "
                    "sample in y along dim"
                )
        return y.narrow(d, 0, 0).sum(d)
    if x is None:
        step = dx if dx is not None else 1.0
        avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)).mul(step / 2.0)
    else:
        xs = _as_tensor(x)
        if xs.dim() == 1:
            if xs.numel() != y.size(d):
                raise RuntimeError(
                    "trapezoid(): there must be one x value for every "
                    "sample in y along dim"
                )
            view = [1] * y.dim()
            view[d] = xs.numel()
            xs = xs.reshape(view)
        n = y.size(d)
        dxs = xs.narrow(d, 1, n - 1).sub(xs.narrow(d, 0, n - 1))
        avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)) \
            .mul(0.5).mul(dxs)
    return avg.sum(d)


def trapz(y, x=None, *, dx=None, dim=-1):
    return trapezoid(y, x, dx=dx, dim=dim)


def cumulative_trapezoid(y, x=None, *, dx=None, dim=-1):
    d = _norm_dim(dim, y.dim())
    n = y.size(d)
    if n == 0:
        if x is not None:
            xs = _as_tensor(x)
            if xs.dim() == 1 and xs.numel() != 0:
                raise RuntimeError(
                    "cumulative_trapezoid(): there must be one x value for "
                    "every sample in y along dim"
                )
        return y.narrow(d, 0, 0)
    if x is None:
        step = dx if dx is not None else 1.0
        avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)).mul(step / 2.0)
    else:
        xs = _as_tensor(x)
        if xs.dim() == 1:
            if xs.numel() != y.size(d):
                raise RuntimeError(
                    "cumulative_trapezoid(): there must be one x value for "
                    "every sample in y along dim"
                )
            view = [1] * y.dim()
            view[d] = xs.numel()
            xs = xs.reshape(view)
        n = y.size(d)
        dxs = xs.narrow(d, 1, n - 1).sub(xs.narrow(d, 0, n - 1))
        avg = y.narrow(d, 0, n - 1).add(y.narrow(d, 1, n - 1)) \
            .mul(0.5).mul(dxs)
    return avg.cumsum(d)


def quantile(input, q, dim=None, keepdim=False, *, interpolation="linear"):
    """Compute quantiles through the dispatcher composite implementation.

    All five interpolation modes, float32/float64, NaN semantics, and output
    shapes are handled here.  A Python-number q is
"""
    if isinstance(q, tensorplay.Tensor):
        qs = q
    elif isinstance(q, (int, float)):
        qs = tensorplay.tensor(q, dtype=input.dtype, device=input.device)
    else:
        raise TypeError(
            "quantile() received an invalid combination of arguments - got "
            f"(Tensor, {type(q).__name__}), but expected (Tensor, Tensor)")
    return tensorplay._C.quantile(input, qs, dim=dim, keepdim=keepdim,
                                  interpolation=interpolation)


def nanquantile(input, q, dim=None, keepdim=False, *, interpolation="linear"):
    if isinstance(q, tensorplay.Tensor):
        qs = q
    elif isinstance(q, (int, float)):
        qs = tensorplay.tensor(q, dtype=input.dtype, device=input.device)
    else:
        raise TypeError(
            "nanquantile() received an invalid combination of arguments - got "
            f"(Tensor, {type(q).__name__}), but expected (Tensor, Tensor)")
    return tensorplay._C.nanquantile(input, qs, dim=dim, keepdim=keepdim,
                                     interpolation=interpolation)


def histc(input, bins=100, min=0, max=0):
    bins = int(bins)
    if bins <= 0:
        raise ValueError("histc(): bins must be a positive integer")
    v = input.reshape([-1]).to(DType.float64)
    lo, hi = float(min), float(max)
    if lo == 0 and hi == 0:
        if v.numel() == 0:
            return tensorplay.zeros(
                [bins], dtype=input.dtype, device=input.device)
        lo = v.min().item()
        hi = v.max().item()
    if lo > hi:
        raise RuntimeError("histc(): upper bound must be larger than "
                           "lower bound")
    if lo == hi or v.numel() == 0:
        return tensorplay.zeros([bins], dtype=input.dtype, device=input.device)
    width = (hi - lo) / bins
    in_range = _band(v.ge(lo), v.le(hi))
    idx = ((v - lo) / width).floor().to(DType.int64) \
        .clamp(0, bins - 1).masked_select(in_range)
    counts = tensorplay.bincount(idx, minlength=bins).narrow(0, 0, bins)
    return counts.to(input.dtype)


def histogram(input, bins=100, range=None, *, weight=None, density=False):
    """
    composite): hist/bin_edges in the input dtype, linspace edges, aminmax
    outer-edge inference with the numpy empty-range expansion, weight in the
    input dtype and density normalization."""
    if isinstance(bins, tensorplay.Tensor):
        return tensorplay._C.histogram(input, bins, weight=weight,
                                       density=density)
    return tensorplay._C.histogram(input, bins, range=range, weight=weight,
                                   density=density)


def _weighted_bincount(idx, weight, nbins, density, widths):
    if weight is not None:
        w = weight.reshape([-1]).to(DType.float64)
        acc = tensorplay.zeros(
            [nbins], dtype=DType.float64, device=idx.device)
        acc = acc.index_put([idx], w, accumulate=True)
    else:
        acc = tensorplay.bincount(idx, minlength=nbins).narrow(0, 0, nbins) \
            .to(DType.float64)
    if density:
        total = acc.sum().item()
        if total > 0:
            acc = acc.div(widths.mul(total))
    return acc


def isin(elements, test_elements, *, assume_unique=False, invert=False):
    el = elements.reshape([-1])
    te = test_elements.reshape([-1])
    if te.numel() == 0:
        out = tensorplay.zeros(
            el.numel(), dtype=DType.bool, device=elements.device)
    else:
        ts, _ = tensorplay.sort(te)
        pos = tensorplay.searchsorted(ts, el, right=True)
        cand = pos.sub(1).clamp(0, ts.numel() - 1)
        found = ts.index_select(0, cand).eq(el)
        out = _band(found, pos.ge(1))
    out = out.reshape(elements.size())
    if invert:
        return out.logical_not()
    return out


def unique_consecutive(input, return_inverse=False, return_counts=False,
                       dim=None):
    if dim is not None:
        dim = _norm_dim(dim, input.dim())
        moved = tensorplay.movedim(input, dim, 0)
        n = moved.size(0)
        if n == 0:
            output = tensorplay.empty(
                list(input.shape), dtype=input.dtype, device=input.device)
            empty_index = tensorplay.empty(
                [0], dtype=DType.int64, device=input.device)
            if return_inverse and return_counts:
                return output, empty_index, empty_index
            if return_inverse:
                return output, empty_index
            if return_counts:
                return output, empty_index
            return output

        flat = moved.reshape([n, -1])
        if n == 1:
            change = tensorplay.ones(
                [1], dtype=DType.bool, device=input.device)
        else:
            adjacent = flat.narrow(0, 1, n - 1).ne(
                flat.narrow(0, 0, n - 1))
            change = tensorplay.cat(
                [tensorplay.ones([1], dtype=DType.bool, device=input.device),
                 tensorplay.any(adjacent, dim=1)],
                dim=0,
            )
        kept = tensorplay.nonzero(change).reshape([-1])
        num_runs = kept.numel()
        output = flat.index_select(0, kept).reshape(
            [num_runs] + list(moved.shape[1:]))
        output = tensorplay.movedim(output, 0, dim)
        inverse = change.to(DType.int64).cumsum(0).sub(1)
        counts = tensorplay.zeros(
            [num_runs], dtype=DType.int64, device=input.device)
        counts = counts.scatter_add(
            0, inverse,
            tensorplay.ones([n], dtype=DType.int64, device=input.device))
        if return_inverse and return_counts:
            return output, inverse, counts
        if return_inverse:
            return output, inverse
        if return_counts:
            return output, counts
        return output
    v = input.reshape([-1])
    n = v.numel()
    if n == 0:
        outs = [tensorplay.zeros([0], dtype=input.dtype, device=input.device)]
        if return_inverse:
            outs.append(tensorplay.zeros([0], dtype=DType.int64,
                                         device=input.device))
        if return_counts:
            outs.append(tensorplay.zeros([0], dtype=DType.int64,
                                         device=input.device))
        return outs[0] if len(outs) == 1 else tuple(outs)
    ne = v.narrow(0, 1, n - 1).ne(v.narrow(0, 0, n - 1))
    change = tensorplay.cat(
        [tensorplay.ones([1], dtype=DType.bool, device=input.device), ne])
    gid = change.to(DType.int64).cumsum(0).sub(1)
    values = v.masked_select(change)
    ngid = int(gid.narrow(0, n - 1, 1).item()) + 1
    outs = [values]
    if return_inverse:
        outs.append(gid)
    if return_counts:
        counts = tensorplay.zeros(
            [ngid], dtype=DType.int64, device=input.device) \
            .scatter_add(
                0, gid,
                tensorplay.ones([n], dtype=DType.int64, device=input.device))
        outs.append(counts)
    return outs[0] if len(outs) == 1 else tuple(outs)


def repeat_interleave(input, repeats, dim=None, *, output_size=None):
    src = input
    if dim is None:
        src = input.reshape([1, -1])
        axis = 1
    else:
        axis = _norm_dim(dim, input.dim())
    if isinstance(repeats, tensorplay.Tensor) or not isinstance(
            repeats, (int, float, bool)):
        rep = _as_tensor(repeats).to(DType.int64).to(device=input.device)
        rep = rep.reshape([-1])
        if rep.numel() == 1 and src.size(axis) != 1:
            rep = tensorplay.full([src.size(axis)], int(rep[0].item()),
                                  dtype=DType.int64, device=input.device)
    else:
        rep = tensorplay.full([src.size(axis)], int(repeats),
                              dtype=DType.int64, device=input.device)
    if rep.numel() != src.size(axis):
        raise RuntimeError(
            "repeat_interleave(): repeats must have the same length as the "
            "selected dimension"
        )
    if bool((rep < 0).any()):
        raise ValueError("repeat_interleave(): repeats must be non-negative")
    ends = rep.cumsum(0)
    total = 0 if rep.numel() == 0 else int(ends[-1].item())
    if output_size is not None and int(output_size) != total:
        raise RuntimeError(
            "repeat_interleave(): output_size does not match the repeated "
            f"length ({int(output_size)} != {total})"
        )
    if total == 0:
        output_shape = list(src.shape)
        output_shape[axis] = 0
        picked = tensorplay.empty(
            output_shape, dtype=input.dtype, device=input.device)
        return picked.reshape([-1]) if dim is None else picked
    pos = tensorplay.arange(total, dtype=DType.int64, device=input.device)
    seg = tensorplay.searchsorted(ends.contiguous(), pos, right=True) \
        .clamp(0, src.size(axis) - 1)
    moved = src.transpose(0, axis) if axis != 0 else src
    picked = moved.index_select(0, seg)
    if axis != 0:
        picked = picked.transpose(0, axis)
    if dim is None:
        return picked.reshape([-1])
    return picked


def kaiser_window(window_length, periodic=True, beta=12.0, *,
                  dtype=None, layout=None, device=None, pin_memory=False,
                  requires_grad=False):
    n = int(window_length)
    if n < 0:
        raise ValueError("kaiser_window(): window_length must be non-negative")
    if layout not in (None, tensorplay.strided):
        raise ValueError("kaiser_window(): only strided layout is supported")
    if dtype is None:
        dtype = tensorplay.get_default_dtype()
    if n <= 1:
        w = tensorplay.ones([n], dtype=dtype, device=device)
        if pin_memory:
            w = w.pin_memory()
        return w.requires_grad_(True) if requires_grad else w
    denom = n if periodic else n - 1
    if denom > 0:
        t = tensorplay.arange(n, dtype=DType.float64, device=device)
        alpha = denom / 2.0
        x = (t - alpha) / alpha
        arg = (1.0 - x.pow(2)).clamp(min=0.0).sqrt().mul(float(beta))

        def _i0_series(z):
            # I0(z) = sum_k ((z^2)/4)^k / (k!)^2, converged in float64
            out = tensorplay.ones_like(z)
            term = tensorplay.ones_like(z)
            for k in range(1, 40):
                term = term.mul(z.mul(z).div(4.0)).div(k * k)
                out = out.add(term)
            return out

        w = _i0_series(arg).div(_i0_series(
            tensorplay.full([], float(beta), dtype=DType.float64,
                            device=device)))
    else:
        w = tensorplay.ones([n], dtype=DType.float64, device=device)
    w = w.to(dtype)
    if pin_memory:
        w = w.pin_memory()
    return w.requires_grad_(True) if requires_grad else w


# ---------------------------------------------------------------------------
# rnn cells
# ---------------------------------------------------------------------------

def lstm_cell(input, hx, cx, w_ih, w_hh, b_ih=None, b_hh=None):
    gates = input.mm(w_ih.t()).add(hx.mm(w_hh.t()))
    if b_ih is not None:
        gates = gates.add(b_ih)
    if b_hh is not None:
        gates = gates.add(b_hh)
    w = gates.size(1) // 4
    gi = gates.narrow(1, 0, w)
    gf = gates.narrow(1, w, w)
    go = gates.narrow(1, 2 * w, w)
    gg = gates.narrow(1, 3 * w, w)
    i = gi.sigmoid()
    f = gf.sigmoid()
    g = gg.tanh()
    o = go.sigmoid()
    cy = f.mul(cx).add(i.mul(g))
    hy = o.mul(cy.tanh())
    return hy, cy


def rnn_relu_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    h = input.mm(w_ih.t()).add(hx.mm(w_hh.t()))
    if b_ih is not None:
        h = h.add(b_ih)
    if b_hh is not None:
        h = h.add(b_hh)
    return h.relu()


def rnn_tanh_cell(input, hx, w_ih, w_hh, b_ih=None, b_hh=None):
    h = input.mm(w_ih.t()).add(hx.mm(w_hh.t()))
    if b_ih is not None:
        h = h.add(b_ih)
    if b_hh is not None:
        h = h.add(b_hh)
    return h.tanh()


# ---------------------------------------------------------------------------
# misc tensor utilities
# ---------------------------------------------------------------------------

def put(input, index, source):
    flat = input.clone().reshape([-1])
    idx = index.to(DType.int64).reshape([-1])
    src = source.to(input.dtype).reshape([-1])
    if src.numel() == 0 and idx.numel() > 0:
        raise RuntimeError("put(): source must not be empty when indices "
                           "are given")
    if src.numel() < idx.numel():
        reps = (idx.numel() + src.numel() - 1) // src.numel()
        src = src.repeat([reps])
    if src.numel() != idx.numel():
        src = src.narrow(0, 0, idx.numel())
    flat = flat.index_put([idx], src)
    return flat.reshape(list(input.size()))


def resolve_conj(input):
    return input


def resolve_neg(input):
    return input


def is_conj(input):
    return False


def is_neg(input):
    return False


_PROMOTE_TABLE = {
    "bool": {'bool': 'bool', 'uint8': 'uint8', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'float16': 'float16', 'bfloat16': 'bfloat16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "uint8": {'bool': 'uint8', 'uint8': 'uint8', 'int8': 'int16', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'float16': 'float16', 'bfloat16': 'bfloat16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "int8": {'bool': 'int8', 'uint8': 'int16', 'int8': 'int8', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'float16': 'float16', 'bfloat16': 'bfloat16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "int16": {'bool': 'int16', 'uint8': 'int16', 'int8': 'int16', 'int16': 'int16', 'int32': 'int32', 'int64': 'int64', 'float16': 'float16', 'bfloat16': 'bfloat16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "int32": {'bool': 'int32', 'uint8': 'int32', 'int8': 'int32', 'int16': 'int32', 'int32': 'int32', 'int64': 'int64', 'float16': 'float16', 'bfloat16': 'bfloat16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "int64": {'bool': 'int64', 'uint8': 'int64', 'int8': 'int64', 'int16': 'int64', 'int32': 'int64', 'int64': 'int64', 'float16': 'float16', 'bfloat16': 'bfloat16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "float16": {'bool': 'float16', 'uint8': 'float16', 'int8': 'float16', 'int16': 'float16', 'int32': 'float16', 'int64': 'float16', 'float16': 'float16', 'bfloat16': 'float32', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "bfloat16": {'bool': 'bfloat16', 'uint8': 'bfloat16', 'int8': 'bfloat16', 'int16': 'bfloat16', 'int32': 'bfloat16', 'int64': 'bfloat16', 'float16': 'float32', 'bfloat16': 'bfloat16', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "float32": {'bool': 'float32', 'uint8': 'float32', 'int8': 'float32', 'int16': 'float32', 'int32': 'float32', 'int64': 'float32', 'float16': 'float32', 'bfloat16': 'float32', 'float32': 'float32', 'float64': 'float64', 'complex64': 'complex64', 'complex128': 'complex128'},
    "float64": {'bool': 'float64', 'uint8': 'float64', 'int8': 'float64', 'int16': 'float64', 'int32': 'float64', 'int64': 'float64', 'float16': 'float64', 'bfloat16': 'float64', 'float32': 'float64', 'float64': 'float64', 'complex64': 'complex128', 'complex128': 'complex128'},
    "complex64": {'bool': 'complex64', 'uint8': 'complex64', 'int8': 'complex64', 'int16': 'complex64', 'int32': 'complex64', 'int64': 'complex64', 'float16': 'complex64', 'bfloat16': 'complex64', 'float32': 'complex64', 'float64': 'complex128', 'complex64': 'complex64', 'complex128': 'complex128'},
    "complex128": {'bool': 'complex128', 'uint8': 'complex128', 'int8': 'complex128', 'int16': 'complex128', 'int32': 'complex128', 'int64': 'complex128', 'float16': 'complex128', 'bfloat16': 'complex128', 'float32': 'complex128', 'float64': 'complex128', 'complex64': 'complex128', 'complex128': 'complex128'},
}


_RANK = {
    DType.bool: 0,
    DType.uint8: 1, DType.int8: 2, DType.int16: 3, DType.int32: 4,
    DType.int64: 5,
    DType.float16: 6, DType.bfloat16: 6, DType.float32: 7, DType.float64: 8,
    DType.complex64: 9, DType.complex128: 10,
}
_CATEGORY = {
    DType.bool: 0,
    DType.uint8: 1, DType.int8: 1, DType.int16: 1, DType.int32: 1,
    DType.int64: 1,
    DType.float16: 2, DType.bfloat16: 2, DType.float32: 2, DType.float64: 2,
    DType.complex64: 3, DType.complex128: 3,
}


def _dt(dt):
    return dt if isinstance(dt, DType) else DType(dt)


def promote_types(type1, type2):
    a, b = _dt(type1), _dt(type2)
    an, bn = a.name, b.name
    if an in _PROMOTE_TABLE and bn in _PROMOTE_TABLE:
        return getattr(DType, _PROMOTE_TABLE[an][bn])
    if _CATEGORY.get(a) != _CATEGORY.get(b):
        hi = a if _CATEGORY.get(a, 0) > _CATEGORY.get(b, 0) else b
        cat = _CATEGORY[hi]
        for name in ("float32", "complex64"):
            if getattr(DType, name) is not None and \
                    _CATEGORY.get(getattr(DType, name)) == cat:
                pass
        return hi
    return a if _RANK[a] >= _RANK[b] else b


def can_cast(from_, to):
    a, b = _dt(from_), _dt(to)
    an, bn = a.name, b.name
    if an in _PROMOTE_TABLE and bn in _PROMOTE_TABLE:
        return _PROMOTE_TABLE[an][bn] == bn
    ca, cb = _CATEGORY[a], _CATEGORY[b]
    if ca != cb:
        return ca < cb
    return _RANK[a] <= _RANK[b]


def result_type(*args):
    best = None
    saw_float_scalar = False
    for item in args:
        if isinstance(item, tensorplay.Tensor):
            dt = item.dtype
        elif isinstance(item, bool):
            dt = None
        elif isinstance(item, int):
            dt = None
        elif isinstance(item, float):
            dt = None
            saw_float_scalar = True
        else:
            dt = _dt(item)
        if dt is not None:
            best = dt if best is None else promote_types(best, dt)
    if best is None:
        if saw_float_scalar:
            return DType.float32
        if all(isinstance(a, bool) for a in args):
            return DType.bool
        return DType.int64
    if saw_float_scalar and _CATEGORY.get(best, 2) == 1:
        return DType.float32
    if saw_float_scalar and best == DType.bool:
        return DType.float32
    return best


def is_nonzero(input):
    n = input.numel()
    if n == 0:
        raise RuntimeError(
            "Boolean value of Tensor with no values is ambiguous"
        )
    if n != 1:
        raise RuntimeError(
            "Boolean value of Tensor with more than one value is ambiguous"
        )
    return input.item() != 0


def is_same_size(input, other):
    return list(input.size()) == list(other.size())


def get_device(input):
    idx = getattr(input.device, "index", None)
    return -1 if idx is None else int(idx)


# ---------------------------------------------------------------------------
# *_copy family
# ---------------------------------------------------------------------------

def alias_copy(input):
    return input.clone()


def t_copy(input):
    return input.t().clone()


def permute_copy(input, dims):
    return input.permute(list(dims)).clone()


def transpose_copy(input, dim0, dim1):
    return input.transpose(dim0, dim1).clone()


def squeeze_copy(input, dim=None):
    if dim is None:
        return input.squeeze()
    return input.squeeze(dim)


def unsqueeze_copy(input, dim):
    return input.unsqueeze(dim)


def select_copy(input, dim, index):
    return input.select(dim, index).clone()


def slice_copy(input, dim=0, start=None, end=None, step=1):
    d = _norm_dim(dim, input.dim())
    n = input.size(d)
    start = 0 if start is None else int(start)
    end = n if end is None else int(end)
    return _slice_getitem(input, d, start, end, step).contiguous()


def narrow_copy(input, dim, start, length):
    return input.narrow(dim, start, length).clone()


def diagonal_copy(input, diagonal=0, dim1=-2, dim2=-1):
    return input.diagonal(diagonal, dim1, dim2).clone()


def unbind_copy(input, dim=0):
    return [t.clone() for t in tensorplay.unbind(input, dim)]


def split_copy(input, split_size_or_sections, dim=0):
    parts = tensorplay.split(input, split_size_or_sections, dim)
    return [p.clone() for p in parts]


def view_copy(input, size):
    return input.view(list(size)).clone()


def reshape_as(input, other):
    # input.reshape(other.shape).
    return input.reshape(list(other.shape))


def unfold_copy(input, dimension, size, step):
    return input.unfold(dimension, size, step).clone()


def expand_copy(input, size, *, implicit=False):
    return input.expand(list(size)).clone()


def unsafe_chunk(input, chunks, dim=0):
    return tensorplay.chunk(input, chunks, dim)


def unsafe_split(input, split_size, dim=0):
    return tensorplay.split(input, split_size, dim)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------

def rms_norm(input, normalized_shape, weight=None, eps=None):
    from tensorplay.nn import functional as F
    return F.rms_norm(input, list(normalized_shape), weight, eps)


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    from tensorplay.nn import functional as F
    return F.cosine_similarity(x1, x2, dim, eps)


def max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False):
    from tensorplay.nn import functional as F
    return F.max_pool1d(input, kernel_size, stride=stride, padding=padding,
                        dilation=dilation, ceil_mode=ceil_mode)


def avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False,
               count_include_pad=True):
    from tensorplay.nn import functional as F
    return F.avg_pool1d(input, kernel_size, stride=stride, padding=padding,
                        ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad)


def adaptive_avg_pool1d(input, output_size):
    from tensorplay.nn import functional as F
    return F.adaptive_avg_pool1d(input, output_size)


def adaptive_max_pool1d(input, output_size):
    from tensorplay.nn import functional as F
    return F.adaptive_max_pool1d(input, output_size)


# ---------------------------------------------------------------------------
# exposes at top level. Lazy import: nn.functional is not available during
# the tensorplay bootstrap (this module loads before the nn package).
# ---------------------------------------------------------------------------


def dropout(input, p=0.5, training=True, inplace=False):
    from tensorplay.nn import functional as _F
    return _F.dropout(input, p=p, training=training, inplace=inplace)


def dropout_(input, p=0.5, training=True):
    from tensorplay.nn import functional as _F
    return _F.dropout_(input, p=p, training=training)


def alpha_dropout(input, p=0.5, training=True, inplace=False):
    from tensorplay.nn import functional as _F
    return _F.alpha_dropout(input, p=p, training=training, inplace=inplace)


def feature_dropout(input, p=0.5, training=False, inplace=False):
    from tensorplay.nn import functional as _F
    return _F.feature_dropout(input, p=p, training=training,
                              inplace=inplace)


def feature_dropout_(input, p=0.5, training=True):
    from tensorplay.nn import functional as _F
    return _F.feature_dropout_(input, p=p, training=training)


def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    from tensorplay.nn import functional as _F
    return _F.feature_alpha_dropout(input, p=p, training=training,
                                    inplace=inplace)


def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False,
          inplace=False):
    from tensorplay.nn import functional as _F
    return _F.rrelu(input, lower=lower, upper=upper, training=training,
                    inplace=inplace)


def rrelu_(input, lower=1.0 / 8, upper=1.0 / 3, training=False):
    from tensorplay.nn import functional as _F
    return _F.rrelu_(input, lower=lower, upper=upper, training=training)


def bilinear(input1, input2, weight, bias=None):
    from tensorplay.nn import functional as _F
    return _F.bilinear(input1, input2, weight, bias)


def ctc_loss(log_probs, targets, input_lengths, target_lengths,
             blank=0, reduction="mean", zero_infinity=False):
    from tensorplay.nn import functional as _F
    return _F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                       blank=blank, reduction=reduction,
                       zero_infinity=zero_infinity)


def embedding_bag(input, weight, offsets=None, max_norm=None,
                  norm_type=2, scale_grad_by_freq=False, mode="mean",
                  sparse=False, per_sample_weights=None,
                  include_last_offset=False, padding_idx=None):
    from tensorplay.nn import functional as _F
    return _F.embedding_bag(input, weight, offsets=offsets,
                            max_norm=max_norm, norm_type=norm_type,
                            scale_grad_by_freq=scale_grad_by_freq,
                            mode=mode, sparse=sparse,
                            per_sample_weights=per_sample_weights,
                            include_last_offset=include_last_offset,
                            padding_idx=padding_idx)


def conv_tbc(input, weight, bias=None, pad=0):
    from tensorplay.nn import functional as _F
    return _F.conv_tbc(input, weight, bias, pad=pad)


def max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1,
               ceil_mode=False, return_indices=False):
    from tensorplay.nn import functional as _F
    return _F.max_pool3d(input, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, ceil_mode=ceil_mode,
                         return_indices=return_indices)


def max_pool1d_with_indices(input, kernel_size, stride=None, padding=0,
                            dilation=1, ceil_mode=False):
    from tensorplay.nn import functional as _F
    return _F.max_pool1d_with_indices(input, kernel_size, stride=stride,
                                      padding=padding, dilation=dilation,
                                      ceil_mode=ceil_mode)


def native_channel_shuffle(input, groups):
    from tensorplay.nn import functional as _F
    return _F.channel_shuffle(input, groups)
