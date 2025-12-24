import tensorplay as tp
import numpy as np
import torch

def test_numpy_conversion():
    t = tp.ones((2, 2))
    n = t.numpy()
    x = tp.ones_like(t)
    assert isinstance(n, np.ndarray)
    assert np.all(n == 1)

def test_torch_conversion():
    t = tp.ones((2, 2))
    n = torch.tensor(t)
    x = tp.ones_like(t)
    assert isinstance(n, torch.Tensor)
    assert torch.all(n == 1)

def test_requires_grad_error():
    t = tp.ones((2, 2), requires_grad=True)
    try:
        t.numpy()
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "Can't call numpy() on Tensor that requires grad" in str(e)
    
    # Should work after detach
    n = t.detach().numpy()
    assert isinstance(n, np.ndarray)
    assert np.all(n == 1)

if __name__ == "__main__":
    try:
        test_numpy_conversion()
        print("test_numpy_conversion passed")
        test_requires_grad_error()
        print("test_requires_grad_error passed")
        test_torch_conversion()
        print("test_torch_conversion passed")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
