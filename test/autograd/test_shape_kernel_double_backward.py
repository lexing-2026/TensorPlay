"""Pooling, upsampling, padding, embedding and view kernels differentiate twice.

The backward of these ops runs a dedicated kernel (max_pool2d_with_indices_
backward, upsample_bilinear2d_backward, reflection_pad2d_backward,
embedding_dense_backward, ...).  Under create_graph that kernel is itself
recorded, so a second pass (a gradient penalty, a Hessian-vector product, a
meta-learning step) reaches the incoming gradient through it.  Each case is
checked against finite differences of the first derivative; the inputs hold
distinct positive values so the max pools have no ties.
"""
import random

import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def distinct(shape, device, seed=0):
    """A float64 tensor of pairwise-distinct positive values in a fixed order."""
    n = 1
    for s in shape:
        n *= s
    order = list(range(n))
    random.Random(seed).shuffle(order)
    values = [0.2 + 0.37 * k / n * 3 + 0.01 * (k % 7) for k in order]
    return tp.tensor(values, dtype=tp.float64, device=device).reshape(*shape)


def check(fn, shape, device, nonlinear=True, seed=0):
    x = distinct(shape, device, seed).requires_grad_(True)
    assert gradgradcheck(fn, (x,))
    if nonlinear:
        # The forward input also feeds the kernel's own slots; a nonlinear
        # producer keeps a wrong slot from cancelling out.
        assert gradgradcheck(lambda t: fn(t * t * t), (x,))


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

MAX_POOLS = {
    "max_pool1d": ((1, 2, 9), lambda x: F.max_pool1d(x, 3, 2, 1)),
    "max_pool1d_dilated": ((1, 2, 9), lambda x: F.max_pool1d(x, 2, 1, 0, 2)),
    "max_pool2d": ((1, 2, 5, 5), lambda x: F.max_pool2d(x, 2, 1)),
    "max_pool2d_strided": ((2, 1, 5, 6), lambda x: F.max_pool2d(x, 3, 2, 1)),
    "max_pool2d_ceil": ((1, 1, 5, 5), lambda x: F.max_pool2d(x, 2, 2, 0, 1, True)),
    "max_pool2d_unbatched": ((2, 5, 5), lambda x: F.max_pool2d(x, 2, 1)),
    "max_pool3d": ((1, 1, 4, 4, 4), lambda x: F.max_pool3d(x, 2, 1)),
    "max_pool3d_strided": ((1, 2, 5, 4, 4), lambda x: F.max_pool3d(x, 2, 2, 1)),
    "max_pool3d_unbatched": ((1, 4, 4, 4), lambda x: F.max_pool3d(x, 2, 1)),
    "adaptive_max_pool1d": ((1, 2, 7), lambda x: F.adaptive_max_pool1d(x, 3)),
    "adaptive_max_pool2d": ((1, 2, 5, 6), lambda x: F.adaptive_max_pool2d(x, (3, 2))),
    "adaptive_max_pool3d": ((1, 1, 5, 4, 5), lambda x: F.adaptive_max_pool3d(x, (3, 2, 3))),
    "adaptive_max_pool3d_unbatched": ((2, 5, 4, 5), lambda x: F.adaptive_max_pool3d(x, (2, 3, 2))),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(MAX_POOLS))
def test_max_pools_differentiate_twice(device, name):
    shape, fn = MAX_POOLS[name]
    check(fn, shape, device)


INDEXED_POOLS = {
    "max_pool1d": ((1, 2, 9), lambda x: F.max_pool1d(x, 3, 2, 1, return_indices=True)),
    "max_pool2d": ((1, 2, 5, 5), lambda x: F.max_pool2d(x, 2, 1, return_indices=True)),
    "max_pool3d": ((1, 1, 4, 4, 4), lambda x: F.max_pool3d(x, 2, 1, return_indices=True)),
    "adaptive_max_pool1d": ((1, 2, 7), lambda x: F.adaptive_max_pool1d_with_indices(x, 3)),
    "adaptive_max_pool2d": ((1, 2, 5, 6),
                            lambda x: F.adaptive_max_pool2d_with_indices(x, (3, 2))),
    "adaptive_max_pool3d": ((1, 1, 5, 4, 5),
                            lambda x: F.adaptive_max_pool3d(x, (3, 2, 3), return_indices=True)),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(INDEXED_POOLS))
def test_max_pools_returning_indices_differentiate_twice(device, name):
    shape, fn = INDEXED_POOLS[name]
    check(lambda x: fn(x)[0], shape, device)


AVG_POOLS = {
    "avg_pool1d": ((1, 2, 9), lambda x: F.avg_pool1d(x, 3, 2, 1)),
    "avg_pool1d_exclude_pad": ((1, 2, 9), lambda x: F.avg_pool1d(x, 3, 2, 1, count_include_pad=False)),
    "avg_pool2d": ((1, 2, 5, 5), lambda x: F.avg_pool2d(x, 2, 1)),
    "avg_pool2d_padded": ((2, 1, 5, 6), lambda x: F.avg_pool2d(x, 3, 2, 1)),
    "avg_pool2d_ceil_exclude_pad": (
        (1, 1, 5, 5), lambda x: F.avg_pool2d(x, 2, 2, 1, True, False)),
    "avg_pool2d_divisor": ((1, 1, 5, 5), lambda x: F.avg_pool2d(x, 2, 1, 0, False, True, 3)),
    "avg_pool3d": ((1, 1, 4, 4, 4), lambda x: F.avg_pool3d(x, 2, 1)),
    "avg_pool3d_padded": ((1, 2, 5, 4, 4), lambda x: F.avg_pool3d(x, 2, 2, 1)),
    "adaptive_avg_pool1d": ((1, 2, 7), lambda x: F.adaptive_avg_pool1d(x, 3)),
    "adaptive_avg_pool2d": ((1, 2, 5, 6), lambda x: F.adaptive_avg_pool2d(x, (3, 4))),
    "adaptive_avg_pool2d_unbatched": ((2, 5, 6), lambda x: F.adaptive_avg_pool2d(x, (2, 3))),
    "adaptive_avg_pool3d": ((1, 1, 5, 4, 5), lambda x: F.adaptive_avg_pool3d(x, (3, 2, 3))),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(AVG_POOLS))
def test_average_pools_differentiate_twice(device, name):
    shape, fn = AVG_POOLS[name]
    check(fn, shape, device)


FRACTIONAL = {
    "fractional_max_pool2d": (
        (1, 2, 7, 7),
        lambda x, s: F.fractional_max_pool2d(x, 2, output_size=(4, 3), _random_samples=s),
        (1, 2, 2)),
    "fractional_max_pool3d": (
        (1, 1, 5, 5, 5),
        lambda x, s: F.fractional_max_pool3d(x, 2, output_size=(3, 3, 2), _random_samples=s),
        (1, 1, 3)),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(FRACTIONAL))
def test_fractional_max_pools_differentiate_twice(device, name):
    shape, fn, sample_shape = FRACTIONAL[name]
    samples = tp.tensor([0.31, 0.74, 0.52, 0.18, 0.63, 0.9][: sample_shape[0] * sample_shape[1] * sample_shape[2]],
                        dtype=tp.float64, device=device).reshape(*sample_shape)
    check(lambda x: fn(x, samples), shape, device)


UNPOOLS = {
    "max_unpool1d": ((1, 2, 8), (1, 2, 4), lambda y, i: F.max_unpool1d(y, i, 2),
                     lambda x: F.max_pool1d(x, 2, return_indices=True)),
    "max_unpool2d": ((1, 2, 4, 4), (1, 2, 2, 2), lambda y, i: F.max_unpool2d(y, i, 2),
                     lambda x: F.max_pool2d(x, 2, return_indices=True)),
    "max_unpool3d": ((1, 1, 4, 4, 4), (1, 1, 2, 2, 2), lambda y, i: F.max_unpool3d(y, i, 2),
                     lambda x: F.max_pool3d(x, 2, return_indices=True)),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(UNPOOLS))
def test_max_unpools_differentiate_twice(device, name):
    full_shape, pooled_shape, unpool, pool = UNPOOLS[name]
    _, indices = pool(distinct(full_shape, device))
    check(lambda y: unpool(y, indices), pooled_shape, device)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

EMBEDDING_CASES = {
    "plain": dict(),
    "padding_idx": dict(padding_idx=2),
    "scale_by_freq": dict(scale_grad_by_freq=True),
    "padding_idx_and_scale": dict(padding_idx=2, scale_grad_by_freq=True),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(EMBEDDING_CASES))
def test_embedding_differentiates_twice(device, name):
    options = EMBEDDING_CASES[name]
    indices = tp.tensor([[1, 2, 2], [4, 1, 0]], dtype=tp.int64, device=device)
    check(lambda w: F.embedding(indices, w, **options), (6, 3), device)


def test_embedding_with_negative_indices_differentiates_twice():
    indices = tp.tensor([-1, 2, -1, 0])
    check(lambda w: F.embedding(indices, w, scale_grad_by_freq=True), (5, 2), "cpu")


BAG_INDICES = [0, 2, 1, 2, 4, 3, 0]
BAG_OFFSETS = [0, 3, 4]


def bag_inputs(device):
    return (tp.tensor(BAG_INDICES, dtype=tp.int64, device=device),
            tp.tensor(BAG_OFFSETS, dtype=tp.int64, device=device))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["sum", "mean", "max"])
@pytest.mark.parametrize("scale", [False, True])
def test_embedding_bag_differentiates_twice(device, mode, scale):
    if mode == "max" and scale:
        pytest.skip("scale_grad_by_freq does not apply to max mode")
    indices, offsets = bag_inputs(device)
    check(lambda w: F.embedding_bag(indices, w, offsets, mode=mode, scale_grad_by_freq=scale),
          (5, 3), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["sum", "mean"])
def test_embedding_bag_with_padding_idx_differentiates_twice(device, mode):
    indices, offsets = bag_inputs(device)
    check(lambda w: F.embedding_bag(indices, w, offsets, mode=mode, padding_idx=2), (5, 3), device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("scale", [False, True])
def test_embedding_bag_per_sample_weights_differentiate_twice(device, scale):
    indices, offsets = bag_inputs(device)
    w = distinct((5, 3), device).requires_grad_(True)
    psw = tp.tensor([0.5, 1.5, 0.7, 1.1, 0.9, 1.3, 0.6], dtype=tp.float64, device=device,
                    requires_grad=True)

    def bags(weight, weights):
        return F.embedding_bag(indices, weight, offsets, mode="sum", per_sample_weights=weights,
                               scale_grad_by_freq=scale)

    assert gradgradcheck(bags, (w, psw))


@pytest.mark.parametrize("device", DEVICES)
def test_embedding_bag_with_last_offset_differentiates_twice(device):
    indices = tp.tensor(BAG_INDICES, dtype=tp.int64, device=device)
    offsets = tp.tensor(BAG_OFFSETS + [len(BAG_INDICES)], dtype=tp.int64, device=device)
    check(lambda w: F.embedding_bag(indices, w, offsets, mode="mean", include_last_offset=True),
          (5, 3), device)


# ---------------------------------------------------------------------------
# Upsampling
# ---------------------------------------------------------------------------

UPSAMPLES = {
    "nearest1d": ((1, 2, 5), lambda x: F.interpolate(x, size=8, mode="nearest")),
    "nearest1d_scale": ((1, 2, 5), lambda x: F.interpolate(x, scale_factor=1.5, mode="nearest")),
    "nearest_exact1d": ((1, 2, 5), lambda x: F.interpolate(x, size=8, mode="nearest-exact")),
    "nearest2d": ((1, 2, 3, 4), lambda x: F.interpolate(x, size=(5, 7), mode="nearest")),
    "nearest_exact2d": ((1, 2, 3, 4), lambda x: F.interpolate(x, size=(5, 7), mode="nearest-exact")),
    "nearest3d": ((1, 1, 2, 3, 3), lambda x: F.interpolate(x, size=(3, 4, 5), mode="nearest")),
    "nearest_exact3d": (
        (1, 1, 2, 3, 3), lambda x: F.interpolate(x, size=(3, 4, 5), mode="nearest-exact")),
    "linear1d": ((1, 2, 5), lambda x: F.interpolate(x, size=8, mode="linear", align_corners=False)),
    "linear1d_align": ((1, 2, 5), lambda x: F.interpolate(x, size=8, mode="linear", align_corners=True)),
    "linear1d_shrink": ((1, 2, 8), lambda x: F.interpolate(x, size=5, mode="linear", align_corners=False)),
    "bilinear": ((1, 2, 3, 4), lambda x: F.interpolate(x, size=(5, 6), mode="bilinear", align_corners=False)),
    "bilinear_align": ((1, 2, 3, 4), lambda x: F.interpolate(x, size=(5, 6), mode="bilinear", align_corners=True)),
    "bilinear_scale": ((1, 1, 3, 4), lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")),
    "bicubic": ((1, 2, 3, 4), lambda x: F.interpolate(x, size=(5, 6), mode="bicubic", align_corners=False)),
    "bicubic_align": ((1, 2, 3, 4), lambda x: F.interpolate(x, size=(5, 6), mode="bicubic", align_corners=True)),
    "trilinear": ((1, 1, 2, 3, 3), lambda x: F.interpolate(x, size=(3, 4, 5), mode="trilinear", align_corners=False)),
    "trilinear_align": ((1, 1, 2, 3, 3), lambda x: F.interpolate(x, size=(3, 4, 5), mode="trilinear", align_corners=True)),
    "bilinear_antialias": (
        (1, 1, 6, 7), lambda x: F.interpolate(x, size=(3, 4), mode="bilinear", antialias=True)),
    "bicubic_antialias": (
        (1, 1, 6, 7), lambda x: F.interpolate(x, size=(3, 4), mode="bicubic", antialias=True)),
    "area": ((1, 2, 5, 6), lambda x: F.interpolate(x, size=(3, 4), mode="area")),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(UPSAMPLES))
def test_interpolation_differentiates_twice(device, name):
    shape, fn = UPSAMPLES[name]
    check(fn, shape, device)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

PADS = {
    "constant1d": ((1, 2, 5), lambda x: F.pad(x, (2, 1), value=0.5)),
    "constant2d": ((1, 2, 3, 4), lambda x: F.pad(x, (1, 2, 2, 1))),
    "constant3d": ((1, 1, 2, 3, 3), lambda x: F.pad(x, (1, 0, 1, 1, 0, 2))),
    "constant_negative": ((1, 2, 5, 6), lambda x: F.pad(x, (-1, 2, 1, -2))),
    "reflect1d": ((1, 2, 5), lambda x: F.pad(x, (2, 3), mode="reflect")),
    "reflect2d": ((1, 2, 4, 5), lambda x: F.pad(x, (2, 1, 1, 2), mode="reflect")),
    "reflect3d": ((1, 1, 3, 4, 4), lambda x: F.pad(x, (1, 2, 1, 1, 2, 1), mode="reflect")),
    "replicate1d": ((1, 2, 5), lambda x: F.pad(x, (2, 3), mode="replicate")),
    "replicate2d": ((1, 2, 3, 4), lambda x: F.pad(x, (2, 1, 1, 3), mode="replicate")),
    "replicate3d": ((1, 1, 2, 3, 3), lambda x: F.pad(x, (1, 2, 1, 1, 2, 1), mode="replicate")),
    "circular1d": ((1, 2, 5), lambda x: F.pad(x, (2, 3), mode="circular")),
    "circular2d": ((1, 2, 3, 4), lambda x: F.pad(x, (2, 1, 1, 2), mode="circular")),
    "circular3d": ((1, 1, 2, 3, 3), lambda x: F.pad(x, (1, 2, 1, 1, 1, 1), mode="circular")),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(PADS))
def test_padding_differentiates_twice(device, name):
    shape, fn = PADS[name]
    check(fn, shape, device)


# ---------------------------------------------------------------------------
# Views and patch extraction
# ---------------------------------------------------------------------------

VIEWS = {
    "unfold_functional": ((1, 2, 5, 5), lambda x: F.unfold(x, 2, padding=1, stride=2)),
    "unfold_dilated": ((1, 1, 5, 5), lambda x: F.unfold(x, 2, dilation=2)),
    "unfold_unbatched": ((2, 4, 4), lambda x: F.unfold(x, 2)),
    "fold": ((1, 4, 9), lambda x: F.fold(x, (4, 4), 2, stride=1)),
    "fold_padded": ((1, 4, 4), lambda x: F.fold(x, (3, 3), 2, padding=1, stride=2)),
    "tensor_unfold": ((2, 7), lambda x: x.unfold(1, 3, 2)),
    "tensor_unfold_overlap": ((2, 6), lambda x: x.unfold(1, 3, 1)),
    "diagonal": ((4, 5), lambda x: x.diagonal()),
    "diagonal_offset": ((2, 4, 5), lambda x: x.diagonal(1, 1, 2)),
    "permute": ((2, 3, 4), lambda x: x.permute(2, 0, 1)),
    "squeeze": ((2, 1, 3, 1), lambda x: x.squeeze()),
}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(VIEWS))
def test_views_and_patches_differentiate_twice(device, name):
    shape, fn = VIEWS[name]
    check(fn, shape, device)


# ---------------------------------------------------------------------------
# A gradient penalty through a pool keeps its term
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_gradient_penalty_through_max_pool_reaches_the_scale(device):
    # y = max_pool2d(w * x) = 4 w (the max sits at x = 4), f = y^2 = 16 w^2.
    # df/dx is nonzero at the max only: 2 y w = 8 w^2, so the penalty
    # ||df/dx||^2 = 64 w^4 and d(penalty)/dw = 256 w^3 = 32 at w = 0.5.  The
    # pooling backward kernel used to return a gradient with no graph, which
    # made this derivative silently 0.
    x = tp.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=tp.float64, device=device,
                  requires_grad=True)
    w = tp.tensor(0.5, dtype=tp.float64, device=device, requires_grad=True)
    y = F.max_pool2d(w * x, 2)
    f = (y * y).sum()
    (gx,) = tp.autograd.grad(f, x, create_graph=True)
    expected_gx = tp.tensor([[[[0.0, 0.0], [0.0, 2.0]]]], dtype=tp.float64, device=device)
    assert tp.allclose(gx, expected_gx)
    (gw,) = tp.autograd.grad((gx * gx).sum(), w)
    assert abs(gw.item() - 32.0) < 1e-9
