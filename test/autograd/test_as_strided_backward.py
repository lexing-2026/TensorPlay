"""Gradients through as_strided views.

A view and its input are windows onto one storage, so a gradient is
assembled where they meet: every element a view reads more than once sums
the contributions, and an input whose own layout is strided or offset reads
its gradient back from the right elements.  The expected values are worked
out by hand from the element each view position reads.
"""
import pytest

import tensorplay as tp
from tensorplay.autograd.gradcheck import gradcheck

DEVICES = ["cpu"] + (["cuda"] if tp.cuda.is_available() else [])


def _leaf(values, device):
    return tp.tensor(values, dtype=tp.float64, device=device, requires_grad=True)


@pytest.mark.parametrize("device", DEVICES)
def test_a_repeated_element_collects_every_contribution(device):
    x = _leaf([1.0, 2.0, 3.0, 4.0], device)
    x.as_strided((3,), (0,)).sum().backward()
    assert x.grad.tolist() == [3.0, 0.0, 0.0, 0.0]

    # Sliding windows: y[i, j] reads x[i + j].
    x = _leaf([1.0, 2.0, 3.0, 4.0], device)
    weights = tp.tensor([[1.0, 10.0], [100.0, 1000.0], [1e4, 1e5]],
                        dtype=tp.float64, device=device)
    (x.as_strided((3, 2), (1, 1)) * weights).sum().backward()
    assert x.grad.tolist() == [1.0, 110.0, 11000.0, 100000.0]


@pytest.mark.parametrize("device", DEVICES)
def test_a_strided_or_offset_input_reads_its_own_elements(device):
    # A transposed input: offset 1 of its storage is x[0, 1], not x[1, 0].
    x = _leaf([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], device)
    x.t().as_strided((2,), (1,), 1).sum().backward()
    assert x.grad.tolist() == [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # An offset input with no explicit offset starts where the input does.
    x = _leaf([1.0, 2.0, 3.0, 4.0, 5.0], device)
    (x[2:].as_strided((2,), (2,)) * tp.tensor([10.0, 100.0], dtype=tp.float64,
                                               device=device)).sum().backward()
    assert x.grad.tolist() == [0.0, 0.0, 10.0, 0.0, 100.0]


@pytest.mark.parametrize("device", DEVICES)
def test_every_spelling_records_the_view(device):
    for view in (lambda t: t.as_strided((2,), (2,)),
                 lambda t: tp.as_strided(t, (2,), (2,)),
                 lambda t: tp._C.as_strided(t, (2,), (2,))):
        x = _leaf([1.0, 2.0, 3.0, 4.0], device)
        y = view(x)
        assert y.grad_fn is not None
        y.sum().backward()
        assert x.grad.tolist() == [1.0, 0.0, 1.0, 0.0]


@pytest.mark.parametrize("device", DEVICES)
def test_an_in_place_update_of_a_view_reaches_the_base(device):
    x = _leaf([1.0, 2.0, 3.0, 4.0], device)
    a = x.clone()
    a[1:3].mul_(2.0)
    a.sum().backward()
    assert x.grad.tolist() == [1.0, 2.0, 2.0, 1.0]

    # The base itself is a transposed view of the root.
    x = _leaf([[1.0, 2.0], [3.0, 4.0]], device)
    a = x.clone().t()
    a[0].mul_(3.0)  # a[0] holds x[0, 0] and x[1, 0]
    (a * tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float64,
                   device=device)).sum().backward()
    assert x.grad.tolist() == [[3.0, 3.0], [6.0, 4.0]]


@pytest.mark.parametrize("device", DEVICES)
def test_the_backward_records_for_a_second_derivative(device):
    x = _leaf([1.0, 2.0, 3.0], device)
    y = x.as_strided((2, 2), (1, 1))  # [[x0, x1], [x1, x2]]
    (g,) = tp.autograd.grad((y * y).sum(), x, create_graph=True)
    assert g.tolist() == [2.0, 8.0, 6.0]
    (gg,) = tp.autograd.grad(g.sum(), x)
    assert gg.tolist() == [2.0, 4.0, 2.0]


def test_overlapping_views_pass_the_numerical_gradient_check():
    x = tp.tensor([0.3, -1.2, 0.8, 2.5, -0.4], dtype=tp.float64, requires_grad=True)
    assert gradcheck(lambda t: t.as_strided((3, 3), (1, 1)) ** 2, x)
    assert gradcheck(lambda t: t.as_strided((2, 2), (0, 2), 1) ** 2, x)
    assert gradcheck(lambda t: t[1:].as_strided((2, 2), (2, 1)) ** 3, x)
