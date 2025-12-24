import torch
import tensorplay as tp
import numpy as np
from torch.utils.dlpack import to_dlpack

def test_strided_add():
    print("Testing Strided Add (Accumulate Grad scenario)...")
    
    # Target: weight.grad (64, 128) contiguous
    # Source: d(weight) (64, 128) transposed view of (128, 64)
    
    # 1. Create d(weight.t()) - contiguous (128, 64)
    t_dW_t = torch.randn(128, 64, device="cuda")
    tp_dW_t = tp.from_dlpack(to_dlpack(t_dW_t))
    
    # 2. Create d(weight) - transposed view
    t_dW = t_dW_t.t()
    tp_dW = tp_dW_t.t()
    
    # 3. Create weight.grad - contiguous (64, 128)
    t_grad = torch.randn(64, 128, device="cuda")
    tp_grad = tp.from_dlpack(to_dlpack(t_grad))
    
    # 4. Accumulate: grad += dW
    t_grad.add_(t_dW)
    tp_grad.add_(tp_dW)
    
    # 5. Check result
    tp_grad_torch = torch.from_dlpack(tp.to_dlpack(tp_grad))
    
    print("TP grad strides (existing):", tp_grad.stride())
    
    # Test initialization from strided view
    tp_grad_new = tp.Tensor(tp_dW) # Copy constructor sharing impl? No, tp.Tensor(other) in python creates alias?
    # Actually tp_dW is a Tensor object.
    # In C++, Tensor(Tensor) shares impl.
    # In Python, we just assign.
    tp_grad_init = tp_dW
    print("TP grad strides (init from view):", tp_grad_init.stride())
    
    if not torch.allclose(tp_grad_torch, t_grad, atol=1e-3):
        print("FAIL: Strided Add mismatch")
        print("Max diff:", (tp_grad_torch - t_grad).abs().max().item())
        
        # Debug
        print("Torch[0,1]:", t_grad[0,1].item())
        print("TP[0,1]:", tp_grad_torch[0,1].item())
        
        # Check if TP just added linearly ignoring strides
        # If linear add: grad.flat[i] += dW.flat[i]
        # dW.flat (view) iterates over dW_t row-major (because strides are (1, 64))
        # dW_t row 0: [0,0], [0,1], ...
        # grad row 0: [0,0], [0,1], ...
        
        # dW element at (0, 1) is dW_t(1, 0).
        # In linear iteration of dW_t: index 1 is dW_t(0, 1).
        # So dW.flat[1] gives dW_t(0, 1).
        # But we want dW(0, 1) which is dW_t(1, 0).
        # dW_t(1, 0) is at index 64 in dW_t.
        
        # So if linear add, we add dW_t(0, 1) to grad(0, 1).
        # But we should add dW_t(1, 0) to grad(0, 1).
        
        val_linear = (t_grad - t_dW).add_(t_dW_t.reshape(64, 128)) # Simulate linear add (wrong)
        if torch.allclose(tp_grad_torch, val_linear, atol=1e-3):
            print("CONFIRMED: TP is doing linear add ignoring strides!")
            
    else:
        print("PASS: Strided Add match")

if __name__ == "__main__":
    try:
        test_strided_add()
    except Exception as e:
        print(e)
