"""grid_sample differentiates twice in its input and its sampling grid.

The backward kernels return (grad_input, grad_grid); a create_graph pass
differentiates both again with respect to the incoming gradient, the input
and the grid, for every interpolation and padding mode.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])
PADDINGS = ["zeros", "border", "reflection"]


def leaf(*shape, device, seed, scale=1.0, shift=0.0):
    tp.manual_seed(2024 + seed)
    t = (tp.rand(*shape, dtype=tp.float64) * 2 - 1) * scale + shift
    return t.to(device).requires_grad_(True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("padding_mode", PADDINGS)
@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
def test_grid_sample_2d_differentiates_twice(device, mode, padding_mode, align_corners):
    x = leaf(2, 3, 5, 6, device=device, seed=1)
    # Points spill past the border so every padding rule is exercised.
    grid = leaf(2, 4, 3, 2, device=device, seed=2, scale=1.15)

    def fn(inp, g):
        return F.grid_sample(inp, g, mode=mode, padding_mode=padding_mode,
                             align_corners=align_corners)

    assert gradgradcheck(fn, (x, grid), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("padding_mode", PADDINGS)
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
def test_grid_sample_3d_differentiates_twice(device, mode, padding_mode, align_corners):
    x = leaf(1, 2, 3, 4, 5, device=device, seed=3)
    grid = leaf(1, 2, 3, 2, 3, device=device, seed=4, scale=1.15)

    def fn(inp, g):
        return F.grid_sample(inp, g, mode=mode, padding_mode=padding_mode,
                             align_corners=align_corners)

    assert gradgradcheck(fn, (x, grid), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
def test_grid_sample_hessian_in_the_grid_is_symmetric(device):
    # d2 L / dgrid2 for L = <w, grid_sample(x, grid)>: the mixed partials of
    # one point agree whichever coordinate is differentiated first.
    x = leaf(1, 2, 6, 6, device=device, seed=5).detach()
    w = leaf(1, 2, 2, 2, device=device, seed=6).detach()
    grid = leaf(1, 2, 2, 2, device=device, seed=7, scale=0.8)
    out = F.grid_sample(x, grid, mode="bicubic", align_corners=False)
    (g,) = tp.autograd.grad((out * w).sum(), grid, create_graph=True)
    gx, = tp.autograd.grad(g[0, 0, 0, 0], grid, retain_graph=True)
    gy, = tp.autograd.grad(g[0, 0, 0, 1], grid)
    assert abs(gx[0, 0, 0, 1].item() - gy[0, 0, 0, 0].item()) < 1e-10
