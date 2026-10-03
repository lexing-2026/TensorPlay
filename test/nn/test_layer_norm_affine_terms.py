"""layer_norm's weight and bias: their shape is the normalized shape.

A weight or bias of another shape would be read one element per normalized
element past its end, so it is refused.  One of the right shape laid out as a
view (expanded along an axis, or strided) is read element by element like any
other.
"""

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_affine_terms_of_another_shape_are_refused(device):
    x = tp.randn(3, 7, 32, device=device)
    with pytest.raises(RuntimeError, match="weight shape mismatch"):
        F.layer_norm(x, (7, 32), tp.ones(32, device=device))
    with pytest.raises(RuntimeError, match="bias shape mismatch"):
        F.layer_norm(x, (7, 32), None, tp.zeros(224, device=device))


@pytest.mark.parametrize("device", DEVICES)
def test_affine_terms_laid_out_as_views(device):
    tp.manual_seed(0)
    x = tp.randn(3, 7, 32, device=device, requires_grad=True)
    w = tp.randn(32, device=device, requires_grad=True)
    b = tp.randn(64, device=device, requires_grad=True)
    out = F.layer_norm(x, (7, 32), w[None].expand(7, 32), b[::2][None].expand(7, 32))

    x2 = x.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    b2 = b.detach().clone().requires_grad_(True)
    ref = F.layer_norm(
        x2, (7, 32), w2[None].expand(7, 32).contiguous(), b2[::2][None].expand(7, 32).contiguous())

    assert tp.allclose(out, ref, atol=1e-5)
    grad = tp.randn(3, 7, 32, device=device)
    out.backward(grad)
    ref.backward(grad)
    for a, c in ((x, x2), (w, w2), (b, b2)):
        assert tp.allclose(a.grad, c.grad, atol=1e-4, rtol=1e-4)
