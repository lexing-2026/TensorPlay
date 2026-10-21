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

def _narrow_loss_rows(host, codes, weight, reduction):
    # The row the target names, negated and scaled by that class's weight; rows
    # naming the skipped code answer zero and carry no weight.
    counted = [i for i in range(host.shape[0]) if codes[i] != -100]
    picked = [-float(host[i, codes[i]]) * (float(weight[codes[i]])
                                            if weight is not None else 1.0)
              for i in counted]
    weight_sum = sum(float(weight[codes[i]]) if weight is not None else 1.0
                     for i in counted)
    if reduction == 0:
        per_row = np.zeros(host.shape[0], np.float64)
        for position, i in enumerate(counted):
            per_row[i] = picked[position]
        # An unreduced batch reports no normalizer at all.
        return per_row, 0.0, weight_sum
    total = sum(picked)
    if reduction == 1:
        total /= weight_sum
    return np.float64(total), weight_sum, weight_sum


def test_likelihood_loss_in_narrow_planes():
    # A narrow likelihood plane is answered in its own type: the walk reads one
    # class per row and totals in double, so the only rounding is the answer's.
    rng = np.random.default_rng(11)
    for tp_dtype, np_dtype, torch_dtype, tol in (
            (tp.float16, np.float16, torch.float16, 6e-3),
            (tp.bfloat16, None, torch.bfloat16, 6e-2)):
        for rows, classes in ((5, 7), (64, 33), (3, 1000)):
            logits = rng.normal(0.0, 2.5, (rows, classes))
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            host = np.log(probs / probs.sum(axis=1, keepdims=True))
            cast = host.astype(np_dtype) if np_dtype else host.astype(np.float32)
            codes = rng.integers(0, classes, rows)
            codes[0] = -100
            for reduction in (0, 1, 2):
                for weighted in (False, True):
                    weight = (np.full(classes, 0.5, np.float32)
                              if weighted else None)
                    label = ["none", "mean", "sum"][reduction]
                    x = tp.tensor(cast, dtype=tp_dtype, device=tp.device("cpu"))
                    t = tp.tensor(codes, device=tp.device("cpu"))
                    w = (tp.tensor(weight, dtype=tp_dtype,
                                   device=tp.device("cpu"))
                         if weighted else None)
                    got, reported = tp._C.nll_loss(x, t, w, reduction, -100)
                    wanted, normalizer, weight_sum = _narrow_loss_rows(
                        host, codes, weight, reduction)
                    answer = got.detach().to(tp.float32).numpy().astype(np.float64)
                    assert got.dtype == tp_dtype, (tp_dtype, got.dtype)
                    assert reported.dtype == tp_dtype, (tp_dtype, reported.dtype)
                    np.testing.assert_allclose(answer.reshape(-1),
                                               np.asarray(wanted).reshape(-1),
                                               atol=tol, rtol=tol)
                    assert abs(float(reported) - normalizer) < 1e-4

                    reference = torch.nn.functional.nll_loss(
                        torch.tensor(cast, dtype=torch_dtype),
                        torch.tensor(codes, dtype=torch.long),
                        torch.tensor(weight, dtype=torch_dtype) if weighted else None,
                        reduction=label, ignore_index=-100)
                    np.testing.assert_allclose(
                        answer.reshape(-1),
                        reference.detach().float().numpy().reshape(-1),
                        atol=tol, rtol=tol)

                    # Every row contributes one entry of its own class, scaled by
                    # the same normalizer the reduction divided the loss by.  The
                    # seed a sum leaves behind is one value spread across the row
                    # axis, so the walk reads it through a layout of its own.
                    x = x.detach().requires_grad_(True)
                    seed = 1.0 if reduction == 1 else 1.0 / weight_sum
                    traced, _ = tp._C.nll_loss(x, t, w, reduction, -100)
                    scale = traced.sum() * seed if reduction == 0 else traced * seed
                    (grad,) = tp.autograd.grad(scale, [x])
                    expected = np.zeros((rows, classes), np.float64)
                    for i in range(rows):
                        if codes[i] != -100:
                            expected[i, codes[i]] = -(
                                float(weight[codes[i]]) if weighted else 1.0)
                            expected[i, codes[i]] /= weight_sum
                    assert grad.dtype == tp_dtype, (tp_dtype, grad.dtype)
                    np.testing.assert_allclose(
                        grad.detach().to(tp.float32).numpy().astype(np.float64),
                        expected, atol=tol, rtol=tol)

    # A seed that reaches the walk as a slice of the rows keeps its own layout:
    # only the rows the slice names receive an entry.
    for tp_dtype, np_dtype, tol in ((tp.float32, np.float32, 1e-5),
                                    (tp.float16, np.float16, 6e-3)):
        rows, classes = 6, 5
        host = np.log(np.linspace(0.2, 0.8, rows * classes)
                      .reshape(rows, classes).astype(np_dtype))
        codes = np.arange(rows) % classes
        x = tp.tensor(host, dtype=tp_dtype,
                      device=tp.device("cpu")).requires_grad_(True)
        traced, _ = tp._C.nll_loss(x, tp.tensor(codes, device=tp.device("cpu")),
                                   None, 0, -100)
        (grad,) = tp.autograd.grad(traced[::2].sum(), [x])
        wanted = np.zeros((rows, classes), np.float64)
        for i in range(0, rows, 2):
            wanted[i, codes[i]] = -1.0
        np.testing.assert_allclose(
            grad.detach().to(tp.float32).numpy().astype(np.float64),
            wanted, atol=tol, rtol=tol)

    integral = tp.tensor(np.zeros((4, 5), np.int32), dtype=tp.int32,
                         device=tp.device("cpu"))
    try:
        tp._C.nll_loss(integral, tp.tensor(np.zeros(4, np.int64),
                                           device=tp.device("cpu")),
                       None, 1, -100)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a likelihood loss read an integral plane")


if __name__ == "__main__":
    test_conv2d()
    test_relu()
    test_max_pool2d()
    test_linear()
    test_cross_entropy()
    test_likelihood_loss_in_narrow_planes()


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
