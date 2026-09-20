import torch
import tensorplay as tp
import pytest
from torch.utils.dlpack import to_dlpack

pytestmark = pytest.mark.skipif(
    not tp.cuda.is_available(), reason="requires CUDA build of tensorplay"
)

def test_matmul_transpose_ops():
    # Mimic the autograd weight-gradient route: input.transpose(-2, -1)
    # feeding matmul as a non-contiguous view.
    t_in = torch.randn(32, 128, device="cuda")
    t_go = torch.randn(32, 64, device="cuda")

    # Target: dW' = X^T G
    t_dW_prime = t_in.t().mm(t_go)

    tp_in = tp.from_dlpack(to_dlpack(t_in))
    tp_go = tp.from_dlpack(to_dlpack(t_go))
    tp_dW_prime = tp_in.transpose(-2, -1).matmul(tp_go)

    tp_dW_prime_torch = torch.from_dlpack(tp.to_dlpack(tp_dW_prime))
    assert torch.allclose(tp_dW_prime_torch, t_dW_prime, atol=1e-3), (
        f"max diff {(tp_dW_prime_torch - t_dW_prime).abs().max().item()}")

if __name__ == "__main__":
    test_matmul_transpose_ops()
