import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorplay as tp

def test_t_autograd():
    print("Testing t() Autograd...")
    
    x = tp.randn([4, 4], requires_grad=True)
    y = x.t()
    
    if y.grad_fn is None:
        print("FAIL: y.grad_fn is None. t() is not tracked!")
    else:
        print(f"PASS: y.grad_fn is {y.grad_fn.name}")
        
    z = y.sum()
    z.backward()
    
    if x.grad is None:
        print("FAIL: x.grad is None")
    else:
        print("PASS: x.grad is computed")
        print("x.grad strides:", x.grad.stride())

if __name__ == "__main__":
    try:
        test_t_autograd()
    except Exception as e:
        print(e)
