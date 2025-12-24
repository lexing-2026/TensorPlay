import torch
import tensorplay as tp
import numpy as np

def test_matmul_transpose_ops():
    print("Testing Matmul with transpose(-2, -1)...")
    
    t_in = torch.randn(32, 128, device="cuda")
    t_go = torch.randn(32, 64, device="cuda")
    
    # Target: dW' = X^T G
    t_dW_prime = t_in.t().mm(t_go)
    
    # TP
    from torch.utils.dlpack import to_dlpack
    tp_in = tp.from_dlpack(to_dlpack(t_in))
    tp_go = tp.from_dlpack(to_dlpack(t_go))
    
    # Mimic autograd: input.transpose(-2, -1).matmul(grad)
    tp_dW_prime = tp_in.transpose(-2, -1).matmul(tp_go)
    
    tp_dW_prime_torch = torch.from_dlpack(tp.to_dlpack(tp_dW_prime))
    
    if not torch.allclose(tp_dW_prime_torch, t_dW_prime, atol=1e-3):
        print("FAIL: Matmul mismatch")
        print("Max diff:", (tp_dW_prime_torch - t_dW_prime).abs().max().item())
    else:
        print("PASS: Matmul match")

if __name__ == "__main__":
    try:
        test_matmul_transpose_ops()
    except Exception as e:
        print(e)
