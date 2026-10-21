"""rms_norm's fused entry reads views, and its backward takes a term mask.

The fused entry reports the row statistic its backward consumes, so a
training call and the composed reference must answer the same numbers
however the operands are laid out: a strided input view and a strided
weight view are read element by element, and both gradients follow.
The backward answers each requested term on its own.
"""

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [tp.float32, tp.float64])
def test_fused_norm_reads_strided_operands(device, dtype):
    tp.manual_seed(0)
    x = tp.randn(2, 8, 4, 16, device=device, dtype=dtype).transpose(1, 2)
    xa = x.detach().requires_grad_(True)
    # A strided weight view of the normalized width.
    wsrc = tp.randn(32, device=device, dtype=dtype)
    wa = wsrc[::2].detach().requires_grad_(True)
    assert not xa.is_contiguous()
    assert not wa.is_contiguous()

    out = F.rms_norm(xa, [16], wa, 1e-6)
    grad = tp.randn(xa.shape, device=device, dtype=dtype)
    grad_x, grad_w = tp.autograd.grad(out, [xa, wa], grad)

    xb = xa.detach().contiguous().requires_grad_(True)
    wb = wa.detach().contiguous().requires_grad_(True)
    ref = F._rms_norm_composite(xb, [16], wb, 1e-6)
    ref_x, ref_w = tp.autograd.grad(ref, [xb, wb], grad)

    assert tp.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert tp.allclose(grad_x, ref_x, atol=1e-4, rtol=1e-4)
    assert tp.allclose(grad_w, ref_w, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_norm_backward_answers_the_requested_terms(device):
    tp.manual_seed(0)
    x = tp.randn(4, 8, device=device)
    w = tp.randn(8, device=device)
    _, rstd = tp._fused_rms_norm(x, [8], w, 1e-6)
    grad = tp.randn(4, 8, device=device)

    both_x, both_w = tp._C._fused_rms_norm_backward(
        grad, x, [8], rstd, w, [True, True])
    input_only, no_weight = tp._C._fused_rms_norm_backward(
        grad, x, [8], rstd, w, [True, False])
    no_input, weight_only = tp._C._fused_rms_norm_backward(
        grad, x, [8], rstd, w, [False, True])

    assert no_weight.numel() == 0
    assert no_input.numel() == 0
    assert tp.allclose(input_only, both_x)
    assert tp.allclose(weight_only, both_w)


@pytest.mark.parametrize("device", DEVICES)
def test_fused_norm_without_affine_terms(device):
    tp.manual_seed(0)
    x = tp.randn(3, 5, 16, device=device).transpose(0, 1)
    xa = x.detach().requires_grad_(True)
    out = F.rms_norm(xa, [16], None, 1e-6)
    grad = tp.randn(xa.shape, device=device)
    grad_x, = tp.autograd.grad(out, xa, grad)

    xb = xa.detach().contiguous().requires_grad_(True)
    ref = F._rms_norm_composite(xb, [16], None, 1e-6)
    ref_x, = tp.autograd.grad(ref, xb, grad)

    assert tp.allclose(out, ref, atol=1e-5, rtol=1e-5)
    assert tp.allclose(grad_x, ref_x, atol=1e-4, rtol=1e-4)
