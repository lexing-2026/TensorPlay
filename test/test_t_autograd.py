import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np

import tensorplay as tp

def test_t_autograd():
    x = tp.randn([4, 4], requires_grad=True)
    y = x.t()

    assert y.grad_fn is not None, "t() is not tracked by autograd"

    z = y.sum()
    z.backward()

    assert x.grad is not None, "x.grad missing after backward"
    # d(x.t().sum()) / dx is all ones regardless of the view.
    np.testing.assert_allclose(x.grad.numpy(), np.ones((4, 4)), rtol=1e-6, atol=1e-7)

if __name__ == "__main__":
    test_t_autograd()
