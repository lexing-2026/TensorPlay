import torch
import tensorplay as tp
from torch.utils.dlpack import to_dlpack

def test_transpose_behavior():
    print("Testing Transpose Behavior...")
    
    # Create contiguous tensor (128, 64)
    t = tp.randn([128, 64])
    print("Original shape:", t.shape)
    print("Original strides:", t.stride())
    
    # Transpose
    t_t = t.t()
    print("Transposed shape:", t_t.shape)
    print("Transposed strides:", t_t.stride())
    
    if t_t.shape != [64, 128]:
        print("FAIL: Shape mismatch")
    if t_t.stride() != [1, 64]: # Expected strides for (128, 64) -> (64, 128)
        print("FAIL: Stride mismatch. Expected [1, 64]")
    else:
        print("PASS: Strides are correct")

if __name__ == "__main__":
    try:
        test_transpose_behavior()
    except Exception as e:
        print(e)
