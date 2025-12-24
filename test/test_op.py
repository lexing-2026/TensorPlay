
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.dlpack import to_dlpack as torch_to_dlpack
from torch.utils.dlpack import from_dlpack as torch_from_dlpack
import tensorplay as tp
import tensorplay.nn as tp_nn
import tensorplay.nn.functional as tp_F

# Set device
DEVICE = "cpu"
torch_device = torch.device(DEVICE)
tp_device = tp.device(DEVICE)

def to_tp(torch_tensor):
    if not torch_tensor.is_contiguous():
        torch_tensor = torch_tensor.contiguous()
    # DLPack transfer
    capsule = torch_to_dlpack(torch_tensor)
    tp_tensor = tp.from_dlpack(capsule)
    return tp_tensor

def to_torch(tp_tensor):
    capsule = tp.to_dlpack(tp_tensor)
    return torch_from_dlpack(capsule)

def compare(name, tp_out, torch_out, tol=1e-4):
    t_out = to_torch(tp_out)
    # Check shape
    if t_out.shape != torch_out.shape:
        print(f"FAIL: {name} shape mismatch. TP: {t_out.shape}, Torch: {torch_out.shape}")
        return False
        
    diff = (t_out - torch_out).abs().max().item()
    if diff > tol:
        print(f"FAIL: {name} max diff: {diff}")
        # Print some stats
        print(f"    TP mean: {t_out.float().mean().item()}, Torch mean: {torch_out.float().mean().item()}")
        print(f"    TP std: {t_out.float().std().item()}, Torch std: {torch_out.float().std().item()}")
        return False
    print(f"PASS: {name} max diff: {diff}")
    return True

def test_relu():
    print("\n--- Testing ReLU ---")
    shape = (32, 64)
    x = torch.randn(shape, device=torch_device, requires_grad=True)
    
    # Forward
    tp_x = to_tp(x)
    tp_x.requires_grad = True
    
    torch_out = F.relu(x)
    tp_out = tp_F.relu(tp_x)
    
    if not compare("ReLU Forward", tp_out, torch_out): return

    # Backward
    grad = torch.randn_like(torch_out)
    tp_grad = to_tp(grad)
    
    torch_out.backward(grad)
    tp_out.backward(tp_grad)
    
    compare("ReLU Backward (grad_input)", tp_x.grad, x.grad)

def test_linear():
    print("\n--- Testing Linear ---")
    B, In, Out = 32, 128, 64
    x = torch.randn(B, In, device=torch_device, requires_grad=True)
    w = torch.randn(Out, In, device=torch_device, requires_grad=True)
    b = torch.randn(Out, device=torch_device, requires_grad=True)
    
    tp_x = to_tp(x); tp_x.requires_grad = True
    tp_w = to_tp(w); tp_w.requires_grad = True
    tp_b = to_tp(b); tp_b.requires_grad = True
    
    torch_out = F.linear(x, w, b)
    tp_out = tp_F.linear(tp_x, tp_w, tp_b)
    
    if not compare("Linear Forward", tp_out, torch_out): return
    
    grad = torch.randn_like(torch_out)
    tp_grad = to_tp(grad)
    
    torch_out.backward(grad)
    tp_out.backward(tp_grad)
    
    compare("Linear Backward (grad_input)", tp_x.grad, x.grad)
    compare("Linear Backward (grad_weight)", tp_w.grad, w.grad)
    compare("Linear Backward (grad_bias)", tp_b.grad, b.grad)

def test_conv2d():
    print("\n--- Testing Conv2d ---")
    # N, C, H, W
    x = torch.randn(4, 3, 32, 32, device=torch_device, requires_grad=True)
    # Out, In, kH, kW
    w = torch.randn(6, 3, 5, 5, device=torch_device, requires_grad=True)
    # Out
    b = torch.randn(6, device=torch_device, requires_grad=True)
    
    tp_x = to_tp(x); tp_x.requires_grad = True
    tp_w = to_tp(w); tp_w.requires_grad = True
    tp_b = to_tp(b); tp_b.requires_grad = True
    
    torch_out = F.conv2d(x, w, b, stride=1, padding=0)
    tp_out = tp_F.conv2d(tp_x, tp_w, tp_b, stride=1, padding=0)
    
    if not compare("Conv2d Forward", tp_out, torch_out): return
    
    grad = torch.randn_like(torch_out)
    tp_grad = to_tp(grad)
    
    torch_out.backward(grad)
    tp_out.backward(tp_grad)
    
    compare("Conv2d Backward (grad_input)", tp_x.grad, x.grad)
    compare("Conv2d Backward (grad_weight)", tp_w.grad, w.grad)
    compare("Conv2d Backward (grad_bias)", tp_b.grad, b.grad)

def test_max_pool2d():
    print("\n--- Testing MaxPool2d ---")
    x = torch.randn(4, 6, 28, 28, device=torch_device, requires_grad=True)
    
    tp_x = to_tp(x); tp_x.requires_grad = True
    
    torch_out = F.max_pool2d(x, kernel_size=2, stride=2)
    tp_out = tp_F.max_pool2d(tp_x, kernel_size=2, stride=2)
    
    if not compare("MaxPool2d Forward", tp_out, torch_out): return
    
    grad = torch.randn_like(torch_out)
    tp_grad = to_tp(grad)
    
    torch_out.backward(grad)
    tp_out.backward(tp_grad)
    
    compare("MaxPool2d Backward (grad_input)", tp_x.grad, x.grad)

def test_log_softmax():
    print("\n--- Testing LogSoftmax ---")
    x = torch.randn(32, 10, device=torch_device, requires_grad=True)
    
    tp_x = to_tp(x); tp_x.requires_grad = True
    
    torch_out = F.log_softmax(x, dim=1)
    tp_out = tp_F.log_softmax(tp_x, dim=1)
    
    if not compare("LogSoftmax Forward", tp_out, torch_out): return
    
    grad = torch.randn_like(torch_out)
    tp_grad = to_tp(grad)
    
    torch_out.backward(grad)
    tp_out.backward(tp_grad)
    
    compare("LogSoftmax Backward (grad_input)", tp_x.grad, x.grad)

def test_nll_loss():
    print("\n--- Testing NLLLoss ---")
    input = torch.randn(32, 10, device=torch_device, requires_grad=True)
    target = torch.randint(0, 10, (32,), device=torch_device).long()
    
    tp_input = to_tp(input); tp_input.requires_grad = True
    tp_target = to_tp(target) # int64
    
    # PyTorch NLLLoss
    torch_out = F.nll_loss(input, target, reduction='mean')
    tp_out = tp_F.nll_loss(tp_input, tp_target, reduction='mean')
    
    if not compare("NLLLoss Forward", tp_out, torch_out): return
    
    # Backward
    torch_out.backward()
    tp_out.backward()
    
    compare("NLLLoss Backward (grad_input)", tp_input.grad, input.grad)

def test_cross_entropy():
    print("\n--- Testing CrossEntropyLoss ---")
    # Logits
    input = torch.randn(32, 10, device=torch_device, requires_grad=True)
    target = torch.randint(0, 10, (32,), device=torch_device).long()
    
    tp_input = to_tp(input); tp_input.requires_grad = True
    tp_target = to_tp(target)
    
    torch_out = F.cross_entropy(input, target)
    tp_out = tp_F.cross_entropy(tp_input, tp_target)
    
    if not compare("CrossEntropy Forward", tp_out, torch_out): return
    
    torch_out.backward()
    tp_out.backward()
    
    compare("CrossEntropy Backward (grad_input)", tp_input.grad, input.grad)

def test_matmul_transpose():
    print("\n--- Testing Matmul Transpose ---")
    # A: (128, 32). B: (32, 64). Result: (128, 64).
    # But A comes from X.t() where X is (32, 128).
    
    X = torch.randn(32, 128, device=torch_device)
    G = torch.randn(32, 64, device=torch_device)
    
    tp_X = to_tp(X)
    tp_G = to_tp(G)
    
    # Target: X.t() @ G
    torch_out = X.t().matmul(G)
    
    # TP:
    # tp_X.t() is not contiguous.
    # matmul should handle it.
    tp_out = tp_X.t().matmul(tp_G)
    
    compare("Matmul Transpose (X.t() @ G)", tp_out, torch_out)
    
    # Test transpose of result (mimic Linear grad_weight)
    tp_out_t = tp_out.t()
    torch_out_t = torch_out.t()
    compare("Matmul Transpose Result Transposed", tp_out_t, torch_out_t)

# def test_contiguous_transpose():
#     print("\n--- Testing Contiguous Transpose ---")
#     # A: (4, 8)
#     # A.t(): (8, 4)
#     # contiguous(A.t()) should be (8, 4) physically transposed.
#     
#     A = torch.arange(32, device=torch_device, dtype=torch.float32).reshape(4, 8)
#     tp_A = to_tp(A)
#     
#     tp_At = tp_A.t()
#     tp_At_contig = tp_At.contiguous()
#     
#     torch_At = A.t()
#     torch_At_contig = torch_At.contiguous()
#     
#     if not compare("Contiguous Transpose", tp_At_contig, torch_At_contig):
#         print("Debugging Contiguous Transpose:")
#         print("Torch:\n", torch_At_contig)
#         print("TP:\n", to_torch(tp_At_contig))
#         return

if __name__ == "__main__":
    try:
        test_relu()
        # test_contiguous_transpose()
        test_matmul_transpose() # Added debug test
        test_linear()
        test_conv2d()
        test_max_pool2d()
        test_log_softmax()
        test_nll_loss()
        test_cross_entropy()
        print("\nAll tests finished.")
    except Exception as e:
        print(f"\nTest failed with exception: {e}")
        import traceback
        traceback.print_exc()
