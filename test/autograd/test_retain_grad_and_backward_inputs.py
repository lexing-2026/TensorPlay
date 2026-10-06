"""retain_grad keeps a non-leaf's gradient; backward(inputs=...) fills only the named tensors.

Also the batched, materialized and dict forms of autograd.grad.
"""

import warnings

import pytest

import tensorplay as tp


def leaf(*values):
    return tp.tensor(list(values), dtype=tp.float64, requires_grad=True)


# ---------------------------------------------------------------------------
# retain_grad
# ---------------------------------------------------------------------------

def test_a_retained_intermediate_keeps_its_gradient():
    x = leaf(1.0, 2.0, 3.0)
    y = x * 2
    y.retain_grad()
    assert y.retains_grad
    (y * y).sum().backward()
    assert y.grad.tolist() == [4.0, 8.0, 12.0]
    assert x.grad.tolist() == [8.0, 16.0, 24.0]


def test_a_retained_gradient_accumulates_over_passes():
    x = leaf(1.0, -1.0)
    y = x.exp()
    y.retain_grad()
    y.sum().backward(retain_graph=True)
    y.sum().backward()
    assert y.grad.tolist() == [2.0, 2.0]


def test_a_leaf_keeps_its_gradient_without_retaining_it():
    x = leaf(1.0)
    x.retain_grad()
    assert not x.retains_grad
    (x * 3).sum().backward()
    assert x.grad.tolist() == [3.0]


def test_retain_grad_refuses_a_tensor_without_gradient():
    with pytest.raises(RuntimeError, match="requires_grad=False"):
        tp.ones(2).retain_grad()


def test_the_retained_gradient_is_what_the_tensor_hooks_hand_on():
    x = leaf(1.0, 2.0)
    y = x * 1
    y.retain_grad()
    y.register_hook(lambda g: g * 10)
    y.sum().backward()
    assert y.grad.tolist() == [10.0, 10.0]
    assert x.grad.tolist() == [10.0, 10.0]


def test_a_retained_tensor_rewritten_in_place_keeps_the_new_values_gradient():
    x = leaf(1.0, 2.0)
    y = x * 1
    y.retain_grad()
    y.mul_(3)
    (y * y).sum().backward()
    # d/dy of y^2 at the rewritten y = 3x.
    assert y.grad.tolist() == [6.0, 12.0]
    assert x.grad.tolist() == [18.0, 36.0]


def test_each_retained_output_of_one_node_keeps_its_own_gradient():
    x = tp.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tp.float64, requires_grad=True)
    a, b = x.unbind(0)
    a.retain_grad()
    b.retain_grad()
    (a * 2 + b * 5).sum().backward()
    assert a.grad.tolist() == [2.0, 2.0]
    assert b.grad.tolist() == [5.0, 5.0]


def test_a_retained_gradient_is_a_copy_of_the_incoming_one():
    x = leaf(1.0, 2.0)
    y = x * 1
    y.retain_grad()
    z = y * 1
    z.sum().backward()
    y.grad.add_(1)
    assert x.grad.tolist() == [1.0, 1.0]


# ---------------------------------------------------------------------------
# backward(inputs=...)
# ---------------------------------------------------------------------------

def test_only_the_named_leaves_accumulate():
    a, b = leaf(1.0, 2.0), leaf(3.0, 4.0)
    (a * b).sum().backward(inputs=[a])
    assert a.grad.tolist() == [3.0, 4.0]
    assert b.grad is None


def test_tensor_backward_takes_one_tensor_as_inputs():
    a, b = leaf(1.0), leaf(2.0)
    (a * b).sum().backward(inputs=b)
    assert a.grad is None
    assert b.grad.tolist() == [1.0]


def test_a_named_non_leaf_keeps_its_gradient_and_the_leaves_below_it_none():
    a = leaf(1.0, 2.0)
    h = a * 3
    (h * h).sum().backward(inputs=[h])
    assert h.grad.tolist() == [6.0, 12.0]
    assert a.grad is None


def test_autograd_backward_takes_a_dict_of_inputs():
    a, b = leaf(2.0), leaf(5.0)
    tp.autograd.backward((a * b).sum(), inputs={"b": b})
    assert a.grad is None
    assert b.grad.tolist() == [2.0]


def test_empty_inputs_are_refused():
    a = leaf(1.0)
    with pytest.raises(RuntimeError, match="cannot be empty"):
        (a * 2).sum().backward(inputs=[])
    with pytest.raises(RuntimeError, match="cannot be empty"):
        tp.autograd.backward((a * 2).sum(), inputs=[])


def test_grad_variables_is_the_deprecated_spelling_of_grad_tensors():
    a = leaf(1.0, 2.0)
    with pytest.warns(FutureWarning, match="grad_variables"):
        tp.autograd.backward(a * 2, grad_variables=tp.ones(2, dtype=tp.float64))
    assert a.grad.tolist() == [2.0, 2.0]


# ---------------------------------------------------------------------------
# autograd.grad
# ---------------------------------------------------------------------------

def test_batched_grad_outputs_give_one_vector_jacobian_product_each():
    x = leaf(1.0, 2.0, 3.0)
    y = x * x
    v = tp.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=tp.float64)
    (g,) = tp.autograd.grad(y, x, v, is_grads_batched=True)
    assert g.tolist() == [[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 4.0, 6.0]]


def test_a_batched_grad_of_an_unused_input_is_none():
    x, unused = leaf(1.0, 2.0), leaf(5.0)
    y = x * 3
    v = tp.ones(4, 2, dtype=tp.float64)
    gx, gu = tp.autograd.grad(y, (x, unused), v, is_grads_batched=True, allow_unused=True)
    assert tuple(gx.shape) == (4, 2)
    assert gu is None


def test_batched_grad_outputs_must_carry_the_batch_dimension():
    x = leaf(1.0, 2.0)
    with pytest.raises(RuntimeError, match="is_grads_batched"):
        tp.autograd.grad(x * 2, x, tp.ones(2, 3, dtype=tp.float64), is_grads_batched=True)


def test_materialized_grads_turn_unused_inputs_into_zeros():
    x, unused = leaf(1.0, 2.0), leaf(5.0, 6.0, 7.0)
    gx, gu = tp.autograd.grad((x * 2).sum(), (x, unused), materialize_grads=True)
    assert gx.tolist() == [2.0, 2.0]
    assert gu.tolist() == [0.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="allow_unused"):
        tp.autograd.grad((x * 2).sum(), (x, unused), materialize_grads=True, allow_unused=False)


def test_grad_with_a_dict_of_inputs_returns_a_dict():
    a, b = leaf(2.0), leaf(3.0)
    grads = tp.autograd.grad((a * b).sum(), {"a": a, "b": b})
    assert set(grads) == {"a", "b"}
    assert grads["a"].tolist() == [3.0]
    assert grads["b"].tolist() == [2.0]


def test_only_inputs_false_warns_and_is_ignored():
    a = leaf(1.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (g,) = tp.autograd.grad((a * 4).sum(), a, only_inputs=False)
    assert any("only_inputs" in str(w.message) for w in caught)
    assert g.tolist() == [4.0]
