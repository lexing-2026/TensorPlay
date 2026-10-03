"""CUDA matrix products whose left operand is a live transpose view.

A weight gradient is ``grad.t() @ input``: the left operand is a transpose
view of a row-major buffer.  The classic BLAS call reads it through its
transpose flag; the plan-based call used for a fused bias (or when that
library is pinned) describes a row-major left operand only and must be handed
a dense copy.  Every case compares against the same product in float64.
"""

import numpy as np
import pytest

import tensorplay as tp

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")

TOLERANCE = {tp.float64: 1e-12, tp.float32: 1e-4, tp.float16: 2e-2, tp.bfloat16: 5e-2}


@pytest.fixture(params=["default", "cublas", "cublaslt"])
def blas_backend(request):
    before = tp.backends.cuda.preferred_blas_library()
    tp.backends.cuda.preferred_blas_library(request.param)
    yield request.param
    tp.backends.cuda.preferred_blas_library(before)


def _operands(dtype, m=96, k=80, n=72):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((k, m))
    b = rng.standard_normal((k, n))
    bias = rng.standard_normal(n)
    as_tp = lambda v: tp.tensor(v, device="cuda").to(dtype)  # noqa: E731
    # The values the kernel reads, so the reference sees the same rounding.
    a64, b64, bias64 = (as_tp(v).double().cpu().numpy() for v in (a, b, bias))
    return as_tp(a), as_tp(b), as_tp(bias), a64, b64, bias64


def _close(got, want, dtype):
    got = got.double().cpu().numpy()
    err = np.abs(got - want).max() / (np.abs(want).max() + 1e-12)
    assert err < TOLERANCE[dtype], err


@pytest.mark.parametrize("dtype", [tp.float64, tp.float32, tp.float16, tp.bfloat16])
def test_transposed_left_operand(dtype, blas_backend):
    a, b, bias, a64, b64, bias64 = _operands(dtype)
    left = a.t()
    assert not left.is_contiguous()
    _close(tp.mm(left, b), a64.T @ b64, dtype)
    _close(left @ b, a64.T @ b64, dtype)
    _close(tp.addmm(bias, left, b), a64.T @ b64 + bias64, dtype)
    # Both operands transposed views.
    _close(tp.mm(left, b.t().contiguous().t()), a64.T @ b64, dtype)


def test_weight_gradient_matches_float64():
    tp.manual_seed(0)
    x = tp.randn(64, 48, device="cuda", requires_grad=True)
    w = tp.randn(40, 48, device="cuda", requires_grad=True)
    g = tp.randn(64, 40, device="cuda")
    tp.nn.functional.linear(x, w).backward(g)
    want = g.double().t() @ x.detach().double()
    _close(w.grad, want.cpu().numpy(), tp.float32)
