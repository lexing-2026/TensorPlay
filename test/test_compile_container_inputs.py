"""Tensors passed inside lists, tuples and dicts keep their gradients.

A compiled function sees a tensor nested in a container argument as an input
of its own, so the result requires grad when that tensor does and backward
returns the gradient to it.
"""
import pytest

import tensorplay as tp


def _leaf(value):
    return tp.tensor([value], requires_grad=True)


def test_list_argument_gradients():
    @tp.compile
    def f(x):
        return x[0] * x[0] + 2.0 * x[1] * x[1]

    a, b = _leaf(3.0), _leaf(-2.0)
    out = f([a, b])
    assert out.requires_grad
    assert out.item() == 17.0
    out.backward()
    assert a.grad.item() == 6.0
    assert b.grad.item() == -8.0
    # A second call reuses the compiled region and accumulates.
    f([a, b]).backward()
    assert a.grad.item() == 12.0
    assert b.grad.item() == -16.0


def test_nested_dict_and_tuple_argument_gradients():
    @tp.compile
    def f(d, k):
        return d["p"] * d["q"][0] + k

    a, b = _leaf(3.0), _leaf(-2.0)
    out = f({"p": a, "q": (b,)}, 1.5)
    assert out.requires_grad
    assert out.item() == -4.5
    out.backward()
    assert a.grad.item() == -2.0
    assert b.grad.item() == 3.0


def test_container_of_plain_tensors_stays_non_differentiable():
    @tp.compile
    def f(x):
        return x[0] * x[1]

    out = f([tp.tensor([2.0]), tp.tensor([5.0])])
    assert not out.requires_grad
    assert out.item() == 10.0


def test_mixed_container_only_differentiates_the_leaves_that_ask():
    @tp.compile
    def f(x):
        return x[0] * x[1]

    a = _leaf(2.0)
    c = tp.tensor([5.0])
    out = f([a, c])
    out.backward()
    assert a.grad.item() == 5.0
    assert c.grad is None


def test_power_operator_with_python_numbers():
    x = tp.tensor([0.5, 2.0], requires_grad=True)
    y = x ** 2
    assert y.tolist() == pytest.approx([0.25, 4.0])
    assert (x ** 0.5).tolist() == pytest.approx([0.70710678, 1.41421356])
    assert (2 ** x).tolist() == pytest.approx([1.41421356, 4.0])
    assert (2.0 ** x).tolist() == pytest.approx([1.41421356, 4.0])
    y.sum().backward()
    assert x.grad.tolist() == pytest.approx([1.0, 4.0])
