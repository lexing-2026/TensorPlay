import numpy as np

import tensorplay as tp

def test_comparison():
    t1 = tp.tensor((1.0, 2.0, 3.0))
    t2 = tp.tensor([1.0, 2.5, 2.0])

    picked = t2[t1 < 2.0]
    np.testing.assert_allclose(picked.numpy(), [1.0], rtol=1e-6, atol=1e-7)

def test_matmul():
    # 2D
    a = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = tp.tensor([[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose((a @ b.t()).numpy(), a.numpy(), rtol=1e-6, atol=1e-7)

    # Batched (3D): (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
    B = 2
    M, K, N = 2, 3, 2

    a_batch = tp.ones(B, M, K)
    b_batch = tp.ones(B, K, N)

    c_batch = a_batch @ b_batch
    assert list(c_batch.shape) == [B, M, N]
    # every output element sums K ones
    np.testing.assert_allclose(c_batch[0].numpy(), np.full((M, N), K, dtype=np.float64),
                               rtol=1e-6, atol=1e-7)

def test_rpow():
    x = tp.tensor([1.0, 2.0, 3.0])

    # Python binds -0.2 ** x as -(0.2 ** x); the tensor sees the
    # right operand of ** only.
    res = -0.2 ** x
    np.testing.assert_allclose(res.numpy(), [-0.2, -0.04, -0.008], rtol=1e-6, atol=1e-7)

if __name__ == "__main__":
    test_comparison()
    test_matmul()
    test_rpow()
