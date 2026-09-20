import numpy as np

import tensorplay as tp


def test_broadcast_grad():
    # (2, 3) + (3,) broadcast through expand must reduce the gradient
    # back to the operand's own shape.
    rng = np.random.RandomState(0)
    x_np = rng.randn(2, 3).astype(np.float32)
    b_np = rng.randn(3).astype(np.float32)

    x = tp.tensor(x_np, requires_grad=True)
    b = tp.tensor(b_np, requires_grad=True)

    b_expanded = b.expand(x.shape)
    y2 = x + b_expanded
    loss2 = y2.sum()
    loss2.backward()

    assert b.grad is not None, "expanded operand received no gradient"
    assert b.grad.shape == b.shape, f"grad shape {b.grad.shape} != operand shape {b.shape}"
    # d(sum(x + b.expand)) / db reduces one contribution per broadcast row.
    np.testing.assert_allclose(
        b.grad.numpy(), np.full(3, x_np.shape[0], dtype=np.float32),
        rtol=1e-5, atol=1e-6)

if __name__ == "__main__":
    test_broadcast_grad()
