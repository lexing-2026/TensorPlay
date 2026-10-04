import torch
import tensorplay as tp
import tensorplay.nn as tp_nn
import numpy as np

def to_tp(t_torch, requires_grad=False):
    t_np = t_torch.detach().numpy()
    t_tp = tp.tensor(t_np, dtype=tp.float32, device=tp.device("cpu"), requires_grad=requires_grad)
    return t_tp

def check(name, tp_tensor, torch_tensor, atol=1e-4, rtol=1e-3):
    assert tp_tensor is not None and torch_tensor is not None, f"{name}: missing tensor"

    tp_np = tp_tensor.detach().numpy()
    torch_np = torch_tensor.detach().numpy()

    assert tp_np.shape == torch_np.shape, (
        f"{name}: shape mismatch {tp_np.shape} vs {torch_np.shape}")

    diff = np.abs(tp_np - torch_np)
    assert np.allclose(tp_np, torch_np, atol=atol, rtol=rtol), (
        f"{name}: max diff {np.max(diff)}")

def test_conv2d():
    N, C, H, W = 2, 1, 32, 32
    OutC, K = 6, 5

    np.random.seed(42)
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tp = to_tp(x_torch, requires_grad=True)

    conv_torch = torch.nn.Conv2d(C, OutC, K)
    conv_tp = tp_nn.Conv2d(C, OutC, K)

    conv_tp.weight.data = to_tp(conv_torch.weight).data
    conv_tp.bias.data = to_tp(conv_torch.bias).data

    y_torch = conv_torch(x_torch)
    y_tp = conv_tp(x_tp)

    check("Conv2d Forward", y_tp, y_torch)

    grad_output_np = np.random.randn(*y_torch.shape).astype(np.float32)
    grad_output_torch = torch.tensor(grad_output_np)
    grad_output_tp = to_tp(grad_output_torch)

    y_torch.backward(grad_output_torch)
    y_tp.backward(grad_output_tp)

    check("Conv2d Grad Input", x_tp.grad, x_torch.grad)
    check("Conv2d Grad Weight", conv_tp.weight.grad, conv_torch.weight.grad)
    check("Conv2d Grad Bias", conv_tp.bias.grad, conv_torch.bias.grad)

def test_linear():
    N, InF, OutF = 32, 120, 84

    np.random.seed(42)
    x_np = np.random.randn(N, InF).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tp = to_tp(x_torch, requires_grad=True)

    lin_torch = torch.nn.Linear(InF, OutF)
    lin_tp = tp_nn.Linear(InF, OutF)

    lin_tp.weight.data = to_tp(lin_torch.weight).data
    lin_tp.bias.data = to_tp(lin_torch.bias).data

    y_torch = lin_torch(x_torch)
    y_tp = lin_tp(x_tp)

    check("Linear Forward", y_tp, y_torch)

    grad_output_np = np.random.randn(*y_torch.shape).astype(np.float32)
    grad_output_torch = torch.tensor(grad_output_np)
    grad_output_tp = to_tp(grad_output_torch)

    y_torch.backward(grad_output_torch)
    y_tp.backward(grad_output_tp)

    check("Linear Grad Input", x_tp.grad, x_torch.grad)
    check("Linear Grad Weight", lin_tp.weight.grad, lin_torch.weight.grad)
    check("Linear Grad Bias", lin_tp.bias.grad, lin_torch.bias.grad)

def test_relu():
    N, C, H, W = 2, 6, 28, 28
    np.random.seed(42)
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tp = to_tp(x_torch, requires_grad=True)

    y_torch = torch.relu(x_torch)
    y_tp = x_tp.relu()

    check("ReLU Forward", y_tp, y_torch)

    grad_output_np = np.random.randn(*y_torch.shape).astype(np.float32)
    grad_output_torch = torch.tensor(grad_output_np)
    grad_output_tp = to_tp(grad_output_torch)

    y_torch.backward(grad_output_torch)
    y_tp.backward(grad_output_tp)

    check("ReLU Grad Input", x_tp.grad, x_torch.grad)

def test_max_pool2d():
    N, C, H, W = 2, 6, 28, 28
    K, S = 2, 2

    np.random.seed(42)
    x_np = np.random.randn(N, C, H, W).astype(np.float32)
    x_torch = torch.tensor(x_np, requires_grad=True)
    x_tp = to_tp(x_torch, requires_grad=True)

    pool_torch = torch.nn.MaxPool2d(K, S)
    pool_tp = tp_nn.MaxPool2d(K, S)

    y_torch = pool_torch(x_torch)
    y_tp = pool_tp(x_tp)

    check("MaxPool2d Forward", y_tp, y_torch)

    grad_output_np = np.random.randn(*y_torch.shape).astype(np.float32)
    grad_output_torch = torch.tensor(grad_output_np)
    grad_output_tp = to_tp(grad_output_torch)

    y_torch.backward(grad_output_torch)
    y_tp.backward(grad_output_tp)

    check("MaxPool2d Grad Input", x_tp.grad, x_torch.grad)

def test_cross_entropy():
    N, C = 32, 10

    np.random.seed(42)
    x_np = np.random.randn(N, C).astype(np.float32)
    target_np = np.random.randint(0, C, size=(N,)).astype(np.int64)

    x_torch = torch.tensor(x_np, requires_grad=True)
    target_torch = torch.tensor(target_np)

    x_tp = to_tp(x_torch, requires_grad=True)
    target_tp = tp.tensor(target_np, dtype=tp.int64, device=tp.device("cpu"))

    loss_torch = torch.nn.CrossEntropyLoss()(x_torch, target_torch)
    loss_tp = tp_nn.CrossEntropyLoss()(x_tp, target_tp)

    check("CrossEntropyLoss Forward", loss_tp, loss_torch)

    loss_torch.backward()
    loss_tp.backward()

    check("CrossEntropyLoss Grad Input", x_tp.grad, x_torch.grad)

if __name__ == "__main__":
    test_conv2d()
    test_relu()
    test_max_pool2d()
    test_linear()
    test_cross_entropy()


def test_activations_in_double_keep_double_precision():
    # A float64 activation and its gradient are computed in double: they agree
    # with the closed forms to the last few bits, contiguous or strided.
    import math
    import tensorplay.nn.functional as F

    xs = np.linspace(-4.5, 4.5, 37)
    scale, alpha = 1.0507009873554805, 1.6732632423543772
    cases = {
        "elu": (lambda x: F.elu(x, 1.3),
                lambda v: v if v > 0 else 1.3 * math.expm1(v),
                lambda v: 1.0 if v > 0 else 1.3 * math.exp(v)),
        "selu": (lambda x: F.selu(x),
                 lambda v: scale * v if v > 0 else scale * alpha * math.expm1(v),
                 lambda v: scale if v > 0 else scale * alpha * math.exp(v)),
        "celu": (lambda x: F.celu(x, 0.7),
                 lambda v: v if v > 0 else 0.7 * math.expm1(v / 0.7),
                 lambda v: 1.0 if v > 0 else math.exp(v / 0.7)),
        "softplus": (lambda x: F.softplus(x, 2.0, 5.0),
                     lambda v: v if 2.0 * v > 5.0 else math.log1p(math.exp(2.0 * v)) / 2.0,
                     lambda v: 1.0 if 2.0 * v > 5.0 else 1.0 / (1.0 + math.exp(-2.0 * v))),
        "leaky_relu": (lambda x: F.leaky_relu(x, 0.2),
                       lambda v: v if v >= 0 else 0.2 * v,
                       lambda v: 1.0 if v > 0 else 0.2),
        "silu": (lambda x: F.silu(x),
                 lambda v: v / (1.0 + math.exp(-v)),
                 lambda v: (1.0 / (1.0 + math.exp(-v))) * (1.0 + v * (1.0 - 1.0 / (1.0 + math.exp(-v))))),
    }
    for strided in (False, True):
        base = tp.tensor(np.repeat(xs, 2) if strided else xs, dtype=tp.float64)
        for name, (fn, value, slope) in cases.items():
            x = (base[::2] if strided else base).detach().requires_grad_(True)
            y = fn(x)
            (g,) = tp.autograd.grad(y.sum(), [x])
            for v, got_y, got_g in zip(xs, y.tolist(), g.tolist()):
                assert abs(got_y - value(v)) <= 1e-13 * max(1.0, abs(value(v))), (name, v)
                assert abs(got_g - slope(v)) <= 1e-13 * max(1.0, abs(slope(v))), (name, v)
