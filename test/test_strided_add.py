import torch
import tensorplay as tp
import pytest
from torch.utils.dlpack import to_dlpack

pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="requires CUDA build of tensorplay"
)

def test_strided_add():
    # Accumulate-grad scenario: the incoming gradient is a transposed
    # view, so add_ must honor the view's strides rather than walking
    # the source buffer linearly.
    # 1. d(weight.t()) - contiguous (128, 64)
    t_dW_t = torch.randn(128, 64, device="cuda")
    tp_dW_t = tp.from_dlpack(to_dlpack(t_dW_t))

    # 2. d(weight) - transposed view
    t_dW = t_dW_t.t()
    tp_dW = tp_dW_t.t()

    # 3. weight.grad - contiguous (64, 128)
    t_grad = torch.randn(64, 128, device="cuda")
    tp_grad = tp.from_dlpack(to_dlpack(t_grad))

    # 4. Accumulate: grad += dW
    t_grad.add_(t_dW)
    tp_grad.add_(tp_dW)

    tp_grad_torch = torch.from_dlpack(tp.to_dlpack(tp_grad))
    assert torch.allclose(tp_grad_torch, t_grad, atol=1e-3), (
        f"max diff {(tp_grad_torch - t_grad).abs().max().item()}")

if __name__ == "__main__":
    test_strided_add()
