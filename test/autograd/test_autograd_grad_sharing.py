"""Leaf gradients stay private when one gradient tensor fans out.

An addition hands the same incoming gradient to both operands.  When one
operand is a leaf and the other feeds a node that is still collecting
gradients, the leaf's stored ``.grad`` and the node's buffer start out as the
same tensor; later accumulation must not write through one into the other.
"""
import numpy as np

import tensorplay as tp


def _leaf(values):
    return tp.tensor(np.asarray(values, dtype=np.float32), requires_grad=True)


def test_leaf_grad_not_overwritten_by_shared_buffer_accumulation():
    a = _leaf([1.0, 2.0, 3.0])
    b = _leaf([4.0, 5.0, 6.0])
    x = _leaf([7.0, 8.0, 9.0])
    # x collects one gradient per branch; a and b each see a single branch.
    out = ((a + x) * 2.0 + (b + x) * 3.0).sum()
    out.backward()
    np.testing.assert_array_equal(a.grad.numpy(), [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(b.grad.numpy(), [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(x.grad.numpy(), [5.0, 5.0, 5.0])


def test_backward_matches_grad_for_shared_operand():
    a = _leaf([[0.5, -1.0], [2.0, 0.25]])
    b = _leaf([[1.5, 0.5], [-2.0, 1.0]])
    c = _leaf([[0.1, 0.2], [0.3, 0.4]])
    x = _leaf([[1.0, 2.0], [3.0, 4.0]])

    def loss():
        return ((a + x) * (b + x) + (c + x) * (c + x)).sum()

    expected = tp.autograd.grad(loss(), [a, b, c, x])
    loss().backward()
    for leaf, want in zip((a, b, c, x), expected):
        np.testing.assert_allclose(leaf.grad.numpy(), want.numpy(), rtol=0, atol=0)


def test_two_leaves_of_one_add_get_independent_grads():
    a = _leaf([1.0, 2.0])
    b = _leaf([3.0, 4.0])
    (a + b).sum().backward()
    a.grad.mul_(10.0)
    np.testing.assert_array_equal(a.grad.numpy(), [10.0, 10.0])
    np.testing.assert_array_equal(b.grad.numpy(), [1.0, 1.0])


def test_grad_tensor_identity_kept_across_backward_calls():
    a = _leaf([1.0, 2.0])
    (a * 3.0).sum().backward()
    held = a.grad
    (a * 4.0).sum().backward()
    # First-order accumulation updates the stored tensor in place, so the
    # handle taken after the first call observes the running sum.
    np.testing.assert_array_equal(held.numpy(), [7.0, 7.0])
    np.testing.assert_array_equal(a.grad.numpy(), [7.0, 7.0])


def test_sdpa_value_grad_backward_matches_grad():
    rng = np.random.RandomState(0)
    shape = (2, 3, 8, 16)
    wq = _leaf(rng.randn(*shape) * 0.02)
    wk = _leaf(rng.randn(*shape) * 0.02)
    wv = _leaf(rng.randn(*shape) * 0.02)
    x = _leaf(rng.randn(*shape))

    def loss():
        out = tp.nn.functional.scaled_dot_product_attention(
            wq + x, wk + x, wv + x, is_causal=True)
        return out.sum()

    expected = tp.autograd.grad(loss(), [wq, wk, wv, x])
    loss().backward()
    for leaf, want in zip((wq, wk, wv, x), expected):
        np.testing.assert_allclose(leaf.grad.numpy(), want.numpy(), rtol=0, atol=1e-6)
