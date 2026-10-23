"""The row kernels, and the lowerings that hand rows over to them.

Each kernel is checked against a float32 reference on the shapes the benchmark
uses, and each lowering is checked to reach its kernel only for the calls the
kernel can answer, leaving the rest of the framework's own path.
"""

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
import tensorplay.compiler.backends.stax.op_lowerings as lowerings
import tensorplay.compiler.backends.stax.codegen.cutedsl.kernels.softmax as softmax_module
from tensorplay.compiler.backends.stax.codegen.cutedsl.kernels import (
    ce_bwd,
    ce_fwd,
    rmsnorm_bwd,
    rmsnorm_fwd,
    softmax_bwd,
    softmax_fwd,
)

cuda_only = pytest.mark.skipif(not tp.cuda.is_available(), reason="needs a CUDA device")

DTYPES = [tp.bfloat16, tp.float32]
SHAPES = [(8, 4096), (3, 8192)]


def _softmax_row(x):
    return tp.softmax(x, -1)


def _softmax_row_sum(x):
    return tp.softmax(x, -1).sum()


def _softmax_transposed(x):
    return tp.softmax(x.transpose(0, 1), -1)


def _rms_row(x, w):
    return F.rms_norm(x, (w.shape[0],), w, 1e-6)


@cuda_only
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_softmax_rows_match_a_float32_reference(dtype, shape):
    x = tp.randn(*shape, device="cuda", dtype=dtype)
    y = tp.softmax(x.float(), -1)
    dy = tp.randn(*shape, device="cuda", dtype=dtype)

    out = softmax_fwd(x)
    assert out.dtype == dtype
    assert tp.allclose(out.float(), y, atol=3e-2, rtol=0)

    dx = softmax_bwd(out, dy)
    want = y * (dy.float() - (y * dy.float()).sum(-1, keepdim=True))
    assert dx.dtype == dtype
    assert tp.allclose(dx.float(), want, atol=3e-2, rtol=2e-2)


@cuda_only
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_rows_match_a_float32_reference(dtype, shape):
    x = tp.randn(*shape, device="cuda", dtype=dtype)
    w = tp.randn(shape[1], device="cuda", dtype=dtype)
    dy = tp.randn(*shape, device="cuda", dtype=dtype)

    rstd = tp.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
    xh = x.float() * rstd
    out, got_rstd = rmsnorm_fwd(x, w, 1e-6)
    assert out.dtype == dtype
    assert got_rstd.dtype == tp.float32
    assert tp.allclose(got_rstd, rstd.reshape(shape[0]), atol=1e-5, rtol=1e-4)
    assert tp.allclose(out.float(), xh * w.float(), atol=3e-2, rtol=2e-2)

    dx, dw = rmsnorm_bwd(x, w, dy, got_rstd)
    wdy = dy.float() * w.float()
    want_dx = (wdy - xh * (xh * wdy).mean(-1, keepdim=True)) * rstd
    assert dx.dtype == dtype
    assert dw.dtype == dtype
    assert tp.allclose(dx.float(), want_dx, atol=3e-2, rtol=2e-2)
    assert tp.allclose(dw.float(), (dy.float() * xh).sum(0), atol=5e-2, rtol=2e-2)


@cuda_only
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_cross_entropy_rows_match_a_float32_reference(dtype, shape):
    rows, classes = shape
    x = tp.randn(*shape, device="cuda", dtype=dtype)
    target = tp.randint(0, classes, (rows,), device="cuda").to(tp.int64)
    dloss = tp.randn(rows, device="cuda", dtype=tp.float32)

    logp = tp.log_softmax(x.float(), -1)
    picked = tp.gather(logp, 1, target.reshape(rows, 1)).reshape(rows)
    loss, lse = ce_fwd(x, target)
    assert loss.dtype == tp.float32
    assert lse.dtype == tp.float32
    assert tp.allclose(loss, -picked, atol=3e-2, rtol=1e-2)
    assert tp.allclose(lse, tp.logsumexp(x.float(), -1), atol=3e-2, rtol=1e-2)

    dx = ce_bwd(x, target, lse, dloss)
    classes_axis = tp.arange(classes, device="cuda").reshape(1, classes)
    onehot = (classes_axis == target.reshape(rows, 1)).float()
    want = (tp.softmax(x.float(), -1) - onehot) * dloss.reshape(rows, 1)
    assert dx.dtype == dtype
    assert tp.allclose(dx.float(), want, atol=3e-2, rtol=2e-2)


@cuda_only
def test_a_compiled_kernel_is_reloaded_from_its_object(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_STAX_CACHE_DIR", str(tmp_path))
    # Earlier tests share the table, so start from an empty one and take the
    # single entry this call adds.
    softmax_module._fwd_cache.clear()
    x = tp.randn(4, 2048, device="cuda", dtype=tp.float32)
    first = softmax_fwd(x)
    (key, built), = softmax_module._fwd_cache.items()
    assert list(tmp_path.rglob("*.o"))
    # A later process starts with an empty table and finds the object instead
    # of building the kernel again.
    softmax_module._fwd_cache.clear()
    again = softmax_fwd(x)
    assert softmax_module._fwd_cache[key] is not built
    assert tp.allclose(again.float(), first.float(), atol=0, rtol=0)


@cuda_only
def test_softmax_rows_are_answered_by_the_row_kernel(monkeypatch):
    reached = []
    real = lowerings._row_kernel

    def counting(name, *args):
        reached.append(name)
        return real(name, *args)

    monkeypatch.setattr(lowerings, "_row_kernel", counting)
    x = tp.randn(8, 4096, device="cuda", dtype=tp.bfloat16)
    out = tp.compile(_softmax_row, strict_native=True)(x)
    assert reached == ["softmax_forward"]
    assert tp.allclose(out.float(), tp.softmax(x.float(), -1), atol=3e-2, rtol=0)


@cuda_only
def test_softmax_gradient_of_a_reduction_is_answered_alone(monkeypatch):
    reached = []
    real = lowerings._row_kernel

    def counting(name, *args):
        reached.append(name)
        return real(name, *args)

    monkeypatch.setattr(lowerings, "_row_kernel", counting)
    x = tp.randn(8, 4096, device="cuda", dtype=tp.bfloat16, requires_grad=True)
    out = tp.compile(_softmax_row_sum, strict_native=True)(x)
    # The backward region is built on the first gradient call, so the
    # backward lowering only runs here.
    tp.autograd.grad(out, (x,))
    assert "softmax_forward" in reached
    assert "softmax_backward_uniform" in reached


@cuda_only
def test_a_non_contiguous_row_keeps_the_loop_lowering(monkeypatch):
    reached = []
    real = lowerings._row_kernel

    def counting(name, *args):
        reached.append(name)
        return real(name, *args)

    monkeypatch.setattr(lowerings, "_row_kernel", counting)
    x = tp.randn(4096, 8, device="cuda", dtype=tp.bfloat16)
    out = tp.compile(_softmax_transposed, strict_native=True)(x)
    assert reached == []
    assert tp.allclose(
        out.float(), tp.softmax(x.float().transpose(0, 1), -1), atol=3e-2, rtol=0
    )


@cuda_only
def test_rms_norm_rows_are_answered_by_the_row_kernel(monkeypatch):
    reached = []
    real = lowerings._row_kernel

    def counting(name, *args):
        reached.append(name)
        return real(name, *args)

    monkeypatch.setattr(lowerings, "_row_kernel", counting)
    # The fused normalisation is the branch a training step takes, so both
    # operands ask for gradients.
    x = tp.randn(8, 4096, device="cuda", dtype=tp.bfloat16, requires_grad=True)
    w = tp.randn(4096, device="cuda", dtype=tp.bfloat16, requires_grad=True)
    out = tp.compile(_rms_row, strict_native=True)(x, w)
    assert reached == ["rms_norm"]
    assert tp.allclose(out.float(), F.rms_norm(x, (4096,), w, 1e-6).float(),
                       atol=3e-2, rtol=0)
