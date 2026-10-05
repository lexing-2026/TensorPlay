"""Convolution and normalization layers differentiate twice.

The backward of a convolution runs a pair of transposed convolutions and the
backward of a normalization layer runs a fused kernel that returns the input,
weight and bias gradients together.  Under create_graph those kernels are
recorded, so a gradient penalty, a Hessian-vector product or a meta-learning
step reaches the incoming gradient, the input and the parameters through
them.  Each case is checked against finite differences of the first
derivative; the last test builds a small convolutional critic and checks the
gradient of a gradient-norm penalty with respect to its parameters.
"""
import pytest

import tensorplay as tp
import tensorplay.nn.functional as F
from tensorplay.autograd import gradgradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def rand(*shape, device="cpu", seed=0):
    tp.manual_seed(1234 + seed)
    return tp.randn(*shape, dtype=tp.float64).to(device).requires_grad_(True)


def check(fn, tensors):
    assert gradgradcheck(fn, tuple(tensors), atol=1e-5, rtol=1e-3)


# ---------------------------------------------------------------------------
# Convolution
# ---------------------------------------------------------------------------

# (input shape, weight shape, bias, keyword arguments)
CONVS = {
    1: [
        ((2, 4, 7), (6, 4, 3), True, {}),
        ((2, 4, 7), (6, 4, 3), False, {"stride": 2, "padding": 1}),
        ((1, 4, 9), (6, 2, 3), True, {"dilation": 2, "groups": 2}),
    ],
    2: [
        ((2, 2, 5, 5), (3, 2, 3, 3), True, {}),
        ((1, 2, 6, 5), (3, 2, 3, 2), False, {"stride": (2, 1), "padding": (1, 0)}),
        ((1, 4, 5, 5), (4, 2, 2, 2), True, {"groups": 2, "dilation": 2}),
        ((1, 2, 5, 4), (2, 2, 3, 3), True, {"padding": 2}),
    ],
    3: [
        ((1, 2, 4, 4, 4), (2, 2, 2, 2, 2), True, {}),
        ((1, 2, 5, 4, 4), (4, 1, 2, 2, 2), False,
         {"stride": 2, "padding": 1, "groups": 2}),
    ],
}

CONV_FN = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_convolutions_differentiate_twice(device, rank):
    for case, (xs, ws, with_bias, kwargs) in enumerate(CONVS[rank]):
        x = rand(*xs, device=device, seed=case)
        w = rand(*ws, device=device, seed=case + 10)
        tensors = [x, w]
        if with_bias:
            tensors.append(rand(ws[0], device=device, seed=case + 20))

        def fn(x, w, b=None, rank=rank, kwargs=kwargs):
            return CONV_FN[rank](x, w, b, **kwargs)

        check(fn, tensors)


# (input shape, weight shape, bias, keyword arguments); a transposed
# convolution's weight is (in, out / groups, kernel...).
TRANSPOSED = {
    1: [
        ((2, 4, 5), (4, 3, 3), True, {}),
        ((2, 4, 5), (4, 3, 3), False, {"stride": 2, "padding": 1, "output_padding": 1}),
        ((1, 4, 5), (4, 2, 3), True, {"groups": 2, "dilation": 2}),
    ],
    2: [
        ((2, 2, 4, 4), (2, 3, 3, 3), True, {}),
        ((1, 2, 4, 3), (2, 2, 3, 2), False,
         {"stride": (2, 1), "padding": (1, 0), "output_padding": (1, 0)}),
        ((1, 4, 3, 3), (4, 2, 2, 2), True, {"groups": 2, "stride": 2}),
    ],
    3: [
        ((1, 2, 3, 3, 3), (2, 2, 2, 2, 2), True, {}),
        ((1, 2, 3, 3, 3), (2, 1, 2, 2, 2), False,
         {"stride": 2, "padding": 1, "output_padding": 1, "groups": 2}),
    ],
}

TRANSPOSED_FN = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("rank", [1, 2, 3])
def test_transposed_convolutions_differentiate_twice(device, rank):
    for case, (xs, ws, with_bias, kwargs) in enumerate(TRANSPOSED[rank]):
        x = rand(*xs, device=device, seed=case)
        w = rand(*ws, device=device, seed=case + 10)
        tensors = [x, w]
        if with_bias:
            tensors.append(rand(ws[1] * kwargs.get("groups", 1), device=device, seed=case + 20))

        def fn(x, w, b=None, rank=rank, kwargs=kwargs):
            return TRANSPOSED_FN[rank](x, w, b, **kwargs)

        check(fn, tensors)


@pytest.mark.parametrize("device", DEVICES)
def test_convolution_weight_gradient_matches_explicit_formula(device):
    """d/dw of <ggi, dL/dx> for a linear loss is the weight gradient with ggi as input."""
    x = rand(1, 2, 5, 5, device=device)
    w = rand(3, 2, 3, 3, device=device, seed=1)
    out = F.conv2d(x, w, padding=1)
    probe = rand(*out.shape, device=device, seed=2).detach()
    (gx,) = tp.autograd.grad(out, x, probe, create_graph=True)
    ggi = rand(*x.shape, device=device, seed=3).detach()
    (gw,) = tp.autograd.grad(gx, w, ggi)
    # <ggi, conv_T(probe, w)> = <conv(ggi, w), probe>; its weight gradient is
    # the gradient of that bilinear form.
    ggi_leaf = ggi.clone().requires_grad_(True)
    (expected,) = tp.autograd.grad((F.conv2d(ggi_leaf, w, padding=1) * probe).sum(), w)
    assert tp.allclose(gw, expected, atol=1e-9)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("shape,normalized", [((3, 5), (5,)), ((2, 3, 4), (4,)), ((2, 3, 4), (3, 4))])
def test_layer_norm_differentiates_twice(device, affine, shape, normalized):
    x = rand(*shape, device=device)
    tensors = [x]
    if affine:
        tensors += [rand(*normalized, device=device, seed=1), rand(*normalized, device=device, seed=2)]

    def fn(x, w=None, b=None):
        return F.layer_norm(x, normalized, w, b, 1e-5)

    check(fn, tensors)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("shape", [(4, 3), (4, 3, 5), (3, 2, 3, 3), (2, 2, 2, 2, 2)])
def test_batch_norm_training_differentiates_twice(device, affine, shape):
    x = rand(*shape, device=device)
    tensors = [x]
    if affine:
        tensors += [rand(shape[1], device=device, seed=1), rand(shape[1], device=device, seed=2)]

    def fn(x, w=None, b=None):
        return F.batch_norm(x, None, None, w, b, True, 0.1, 1e-5)

    check(fn, tensors)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("affine", [True, False])
def test_batch_norm_eval_differentiates_twice(device, affine):
    x = rand(3, 3, 4, 4, device=device)
    running_mean = rand(3, device=device, seed=3).detach()
    running_var = (rand(3, device=device, seed=4).detach() ** 2 + 0.5)
    tensors = [x]
    if affine:
        tensors += [rand(3, device=device, seed=1), rand(3, device=device, seed=2)]

    def fn(x, w=None, b=None):
        return F.batch_norm(x, running_mean, running_var, w, b, False, 0.1, 1e-5)

    check(fn, tensors)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("groups,shape", [(1, (2, 4, 3)), (2, (2, 4, 3, 3)), (4, (2, 4, 2, 2)),
                                          (2, (2, 6, 2, 2, 2))])
def test_group_norm_differentiates_twice(device, affine, groups, shape):
    channels = shape[1]
    if channels % groups:
        groups = 1
    x = rand(*shape, device=device)
    tensors = [x]
    if affine:
        tensors += [rand(channels, device=device, seed=1), rand(channels, device=device, seed=2)]

    def fn(x, w=None, b=None):
        return F.group_norm(x, groups, w, b, 1e-5)

    check(fn, tensors)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("shape", [(2, 3, 5), (2, 3, 3, 3)])
def test_instance_norm_differentiates_twice(device, affine, shape):
    x = rand(*shape, device=device)
    tensors = [x]
    if affine:
        tensors += [rand(shape[1], device=device, seed=1), rand(shape[1], device=device, seed=2)]

    def fn(x, w=None, b=None):
        return F.instance_norm(x, None, None, w, b, True, 0.1, 1e-5)

    check(fn, tensors)


@pytest.mark.parametrize("device", DEVICES)
def test_instance_norm_running_statistics_differentiates_twice(device):
    x = rand(2, 3, 4, 4, device=device)
    running_mean = rand(3, device=device, seed=3).detach()
    running_var = (rand(3, device=device, seed=4).detach() ** 2 + 0.5)
    tensors = [x, rand(3, device=device, seed=1), rand(3, device=device, seed=2)]

    def fn(x, w, b):
        return F.instance_norm(x, running_mean, running_var, w, b, False, 0.1, 1e-5)

    check(fn, tensors)


@pytest.mark.parametrize("device", DEVICES)
def test_norm_backward_third_derivative(device):
    """The second-derivative nodes are built from recorded operations."""
    x = rand(3, 5, device=device)
    w = rand(5, device=device, seed=1)

    def first(x, w):
        y = (F.layer_norm(x, (5,), w, None, 1e-5) ** 2).sum()
        (gx,) = tp.autograd.grad(y, x, create_graph=True)
        return gx

    assert gradgradcheck(first, (x, w), atol=1e-4, rtol=1e-3)


# ---------------------------------------------------------------------------
# Gradient penalty through a convolutional critic
# ---------------------------------------------------------------------------

def critic(x, w1, b1, w2, b2, gamma, beta):
    h = F.conv2d(x, w1, b1, padding=1)
    h = F.batch_norm(h, None, None, gamma, beta, True, 0.1, 1e-5)
    h = F.leaky_relu(h, 0.2)
    h = F.conv2d(h, w2, b2, stride=2)
    return (h * h).sum() + h.sum()


def penalty(x, params):
    x = x.detach().requires_grad_(True)
    out = critic(x, *params)
    (gx,) = tp.autograd.grad(out, x, create_graph=True)
    return (gx * gx).sum()


@pytest.mark.parametrize("device", DEVICES)
def test_gradient_penalty_through_conv_batch_norm_matches_finite_differences(device):
    x = rand(4, 2, 5, 5, device=device).detach()
    params = [
        rand(3, 2, 3, 3, device=device, seed=1) * 0.5,
        rand(3, device=device, seed=2),
        rand(2, 3, 2, 2, device=device, seed=3) * 0.5,
        rand(2, device=device, seed=4),
        rand(3, device=device, seed=5) + 1.0,
        rand(3, device=device, seed=6),
    ]
    params = [p.detach().requires_grad_(True) for p in params]
    value = penalty(x, params)
    grads = tp.autograd.grad(value, params)

    eps = 1e-6
    for p, g in zip(params, grads):
        flat = p.detach().reshape(-1).clone()
        numeric = tp.zeros_like(flat)
        for i in range(flat.numel()):
            plus, minus = flat.clone(), flat.clone()
            plus[i] += eps
            minus[i] -= eps
            args_plus = [q.detach() if q is not p else plus.reshape(p.shape) for q in params]
            args_minus = [q.detach() if q is not p else minus.reshape(p.shape) for q in params]
            numeric[i] = (penalty(x, args_plus) - penalty(x, args_minus)) / (2 * eps)
        assert tp.allclose(g.reshape(-1), numeric, atol=1e-5, rtol=1e-4), (p.shape, g, numeric)


# ---------------------------------------------------------------------------
# First-order kernels the second derivatives rest on
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
def test_conv_transpose1d_bias_gradient_is_the_channel_sum(device):
    x = rand(2, 4, 5, device=device)
    w = rand(4, 3, 3, device=device, seed=1)
    b = rand(3, device=device, seed=2)
    out = F.conv_transpose1d(x, w, b)
    probe = rand(*out.shape, device=device, seed=3).detach()
    (gb,) = tp.autograd.grad(out, b, probe)
    assert tp.allclose(gb, probe.sum(dim=(0, 2)), atol=1e-12)


def test_float64_batch_norm_keeps_double_precision_in_channels_last():
    x = rand(4, 3, 5, 5).detach()
    w = (rand(3, seed=1) + 1.0).detach()
    b = rand(3, seed=2).detach()
    mean = x.mean(dim=(0, 2, 3), keepdim=True)
    var = ((x - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
    expected = (x - mean) * (var + 1e-5) ** -0.5 * w.reshape(1, 3, 1, 1) + b.reshape(1, 3, 1, 1)
    for t in (x, x.contiguous(memory_format=tp.channels_last)):
        out = F.batch_norm(t, None, None, w, b, True, 0.1, 1e-5)
        assert tp.allclose(out, expected, atol=1e-13)


def test_float64_group_norm_matches_the_definition():
    x = rand(2, 4, 3, 3).detach()
    w = rand(4, seed=1).detach()
    b = rand(4, seed=2).detach()
    grouped = x.reshape(2, 2, -1)
    mean = grouped.mean(dim=2, keepdim=True)
    var = ((grouped - mean) ** 2).mean(dim=2, keepdim=True)
    norm = ((grouped - mean) * (var + 1e-5) ** -0.5).reshape(2, 4, 3, 3)
    expected = norm * w.reshape(1, 4, 1, 1) + b.reshape(1, 4, 1, 1)
    assert tp.allclose(F.group_norm(x, 2, w, b, 1e-5), expected, atol=1e-13)
