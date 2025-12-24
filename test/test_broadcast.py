import tensorplay as tp
import numpy as np

def test_broadcast_grad():
    print("Testing Broadcast Gradient...")
    
    # Shape: (2, 3) + (3,)
    x_np = np.random.randn(2, 3).astype(np.float32)
    b_np = np.random.randn(3).astype(np.float32)
    
    x = tp.tensor(x_np, requires_grad=True)
    b = tp.tensor(b_np, requires_grad=True)
    
    y = x + b
    # loss = y.sum()
    # loss.backward()
    
    # Try with expand
    print("Testing with expand...")
    x.grad = None
    b.grad = None
    
    b_expanded = b.expand(x.shape)
    y2 = x + b_expanded
    loss2 = y2.sum()
    loss2.backward()
    
    print(f"b.grad shape (with expand): {b.grad.shape}")
    if b.grad.shape != b.shape:
        print(f"FAIL: b.grad shape {b.grad.shape} != b.shape {b.shape}")
    else:
        print("PASS: b.grad shape matches")

if __name__ == "__main__":
    test_broadcast_grad()
