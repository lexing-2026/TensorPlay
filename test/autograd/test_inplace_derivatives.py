"""In-place spellings differentiate like their functional twins.

An in-place op records the same backward as the out-of-place operator: a
formula that reads the input sees it as it was before the update, and one
that reads the result sees the updated tensor.  Each case compares the
in-place gradient with the out-of-place one under non-uniform weights.
"""
import pytest

import tensorplay as tp

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def _const(device, *shape, value=None):
    if value is not None:
        return tp.full(shape, value, dtype=tp.float64, device=device)
    return tp.linspace(0.2, 1.4, 9, dtype=tp.float64, device=device).reshape(*shape)


UNARY = {
    "exp_": (lambda a: a.exp_(), lambda a: a.exp()),
    "sigmoid_": (lambda a: a.sigmoid_(), lambda a: a.sigmoid()),
    "tanh_": (lambda a: a.tanh_(), lambda a: a.tanh()),
    "sin_": (lambda a: a.sin_(), lambda a: a.sin()),
    "abs_": (lambda a: a.abs_(), lambda a: a.abs()),
    "square_": (lambda a: a.square_(), lambda a: a.square()),
    "clamp_": (lambda a: a.clamp_(-0.5, 0.5), lambda a: a.clamp(-0.5, 0.5)),
    "pow_": (lambda a: a.pow_(3), lambda a: a.pow(3)),
    "reciprocal_": (lambda a: a.reciprocal_(), lambda a: a.reciprocal()),
    "cumsum_": (lambda a: a.cumsum_(0), lambda a: a.cumsum(0)),
    "tril_": (lambda a: a.tril_(), lambda a: a.tril()),
    "t_": (lambda a: a.t_(), lambda a: a.t()),
    "unsqueeze_": (lambda a: a.unsqueeze_(0), lambda a: a.unsqueeze(0)),
    "leaky_relu_": (lambda a: tp.nn.functional.leaky_relu_(a, 0.1),
                    lambda a: tp.nn.functional.leaky_relu(a, 0.1)),
}


def _grad(fn, x, weights):
    out = fn(x * 1)
    (g,) = tp.autograd.grad((out * weights.reshape(out.shape)).sum(), x)
    return g


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(UNARY))
def test_an_in_place_op_differentiates_like_its_twin(device, name):
    inplace, functional = UNARY[name]
    x = tp.tensor([[-1.2, 0.3, 0.9], [0.5, -0.4, 1.1], [0.7, 0.2, -0.8]],
                  dtype=tp.float64, device=device, requires_grad=True)
    weights = _const(device, 3, 3)
    assert tp.allclose(_grad(inplace, x, weights), _grad(functional, x, weights))


@pytest.mark.parametrize("device", DEVICES)
def test_binary_in_place_ops_differentiate_both_operands(device):
    for inplace, functional in (
            (lambda a, b: a.mul_(b), lambda a, b: a * b),
            (lambda a, b: a.div_(b), lambda a, b: a / b),
            (lambda a, b: a.lerp_(b, 0.3), lambda a, b: a.lerp(b, 0.3)),
            (lambda a, b: a.addcmul_(b, b, value=0.5), lambda a, b: a.addcmul(b, b, value=0.5)),
            (lambda a, b: a.xlogy_(b), lambda a, b: a.xlogy(b))):
        grads = []
        for fn in (inplace, functional):
            x = tp.tensor([0.5, 1.5, 2.5], dtype=tp.float64, device=device,
                          requires_grad=True)
            w = tp.tensor([2.0, 3.0, 5.0], dtype=tp.float64, device=device,
                          requires_grad=True)
            out = fn(x * 1, w)
            out.mul(tp.tensor([1.0, 10.0, 100.0], dtype=tp.float64,
                              device=device)).sum().backward()
            grads.append((x.grad, w.grad))
        assert tp.allclose(grads[0][0], grads[1][0])
        assert tp.allclose(grads[0][1], grads[1][1])


@pytest.mark.parametrize("device", DEVICES)
def test_a_tensor_fill_value_collects_the_filled_positions(device):
    x = tp.tensor([1.0, 2.0, 3.0, 4.0], dtype=tp.float64, device=device, requires_grad=True)
    value = tp.tensor(9.0, dtype=tp.float64, device=device, requires_grad=True)
    mask = tp.tensor([True, False, True, False], device=device)
    weights = tp.tensor([1.0, 10.0, 100.0, 1000.0], dtype=tp.float64, device=device)
    for fill in (lambda a: a.masked_fill(mask, value), lambda a: a.masked_fill_(mask, value)):
        x.grad = None
        value.grad = None
        (fill(x * 1) * weights).sum().backward()
        assert x.grad.tolist() == [0.0, 10.0, 0.0, 1000.0]
        assert value.grad.item() == 101.0


@pytest.mark.parametrize("device", DEVICES)
def test_an_in_place_activation_differentiates_twice(device):
    x = tp.tensor([0.3, -0.7], dtype=tp.float64, device=device, requires_grad=True)
    (g,) = tp.autograd.grad((x * 1).sigmoid_().sum(), x, create_graph=True)
    s = tp.sigmoid(x.detach())
    assert tp.allclose(g, s * (1 - s))
    (gg,) = tp.autograd.grad(g.sum(), x)
    assert tp.allclose(gg, s * (1 - s) * (1 - 2 * s))


@pytest.mark.parametrize("device", DEVICES)
def test_a_shape_update_moves_no_element(device):
    x = tp.arange(6, dtype=tp.float64, device=device).view(2, 3)
    x.t_()
    assert x.shape == (3, 2) and x.stride() == (1, 3)
    assert x.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
    x.unsqueeze_(1)
    assert x.shape == (3, 1, 2)
    x.squeeze_()
    assert x.shape == (3, 2)
    x.swapaxes_(0, 1)
    assert x.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


@pytest.mark.parametrize("device", DEVICES)
def test_a_view_that_reshapes_itself_reads_its_base_through_the_new_shape(device):
    def leaf():
        return tp.arange(6, dtype=tp.float64, device=device, requires_grad=True)

    a = leaf()
    v = (a * 1).view(2, 3)
    v.t_()
    (v * tp.arange(1, 7, dtype=tp.float64, device=device).view(3, 2)).sum().backward()
    assert a.grad.tolist() == [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]

    a = leaf()
    v = (a * 1)[1:5]
    v.unsqueeze_(1)
    (v * tp.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=tp.float64,
                   device=device)).sum().backward()
    assert v.shape == (4, 1)
    assert a.grad.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 0.0]

    a = leaf()
    v = (a * 1).view(1, 6, 1)[:, 2:5]
    v.squeeze_()
    (v * tp.tensor([1.0, 2.0, 3.0], dtype=tp.float64, device=device)).sum().backward()
    assert a.grad.tolist() == [0.0, 0.0, 1.0, 2.0, 3.0, 0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_reshaped_view_follows_later_updates_of_its_base_and_itself(device):
    a = tp.arange(6, dtype=tp.float64, device=device, requires_grad=True)
    b = a * 1
    v = b.view(2, 3)
    v.t_()
    b.mul_(2)
    (v * tp.arange(1, 7, dtype=tp.float64, device=device).view(3, 2)).sum().backward()
    assert a.grad.tolist() == [2.0, 6.0, 10.0, 4.0, 8.0, 12.0]

    a = tp.arange(6, dtype=tp.float64, device=device, requires_grad=True)
    v = (a * 1).view(2, 3)
    v.t_()
    v.mul_(tp.tensor([[1.0], [10.0], [100.0]], dtype=tp.float64, device=device))
    v.sum().backward()
    assert a.grad.tolist() == [1.0, 10.0, 100.0, 1.0, 10.0, 100.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_leaf_that_requires_grad_refuses_an_in_place_update(device):
    x = tp.tensor([0.5, 1.0], dtype=tp.float64, device=device, requires_grad=True)
    with pytest.raises(RuntimeError, match="leaf Variable"):
        x.exp_()
    with tp.no_grad():
        x.exp_()
