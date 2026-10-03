"""Single-precision convolutions with TF32 turned off run in full precision.

The fast heuristic's candidates for a float32 convolution can all be
tensor-core engines, which round through TF32; with TF32 off those are not
allowed, and the plain engines come from the library's fallback list.  Each
case compares the forward and both gradients with float64.
"""

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F

pytestmark = pytest.mark.skipif(not tp.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def no_tf32():
    allowed = tp.backends.cudnn.allow_tf32
    tp.backends.cudnn.allow_tf32 = False
    yield
    tp.backends.cudnn.allow_tf32 = allowed


@pytest.mark.parametrize("shape, weight", [
    ((8, 64, 32, 32), (64, 64, 3, 3)),
    ((8, 128, 16, 16), (128, 128, 3, 3)),
    ((8, 64, 16, 16), (128, 64, 1, 1)),
])
def test_float32_without_tf32_matches_float64(shape, weight, no_tf32):
    tp.manual_seed(0)
    x = tp.randn(*shape, device="cuda", requires_grad=True)
    w = tp.randn(*weight, device="cuda", requires_grad=True)
    pad = weight[-1] // 2
    y = F.conv2d(x, w, padding=pad)
    g = tp.randn(*y.shape, device="cuda")
    got = (y,) + tuple(tp.autograd.grad(y, (x, w), g))

    x64 = x.detach().double().requires_grad_(True)
    w64 = w.detach().double().requires_grad_(True)
    y64 = F.conv2d(x64, w64, padding=pad)
    want = (y64,) + tuple(tp.autograd.grad(y64, (x64, w64), g.double()))
    for a, b in zip(got, want):
        # TF32 would be off by about 1e-3 here.
        assert ((a.double() - b).abs().max() / b.abs().max()).item() < 1e-4


def test_relu_backward_in_float64():
    x = tp.randn(64, device="cuda", dtype=tp.float64, requires_grad=True)
    g = tp.randn(64, device="cuda", dtype=tp.float64)
    (gx,) = tp.autograd.grad(F.relu(x), (x,), g)
    assert tp.equal(gx, tp.where(x > 0, g, tp.zeros_like(g)))
