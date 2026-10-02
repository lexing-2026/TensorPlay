import math

import pytest
import tensorplay as tp
import tensorplay.optim as optim
from tensorplay.optim import lr_scheduler

def test_optimizers():
    print("Testing Optimizers...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    
    # Test SGD
    print("Testing SGD...")
    opt = optim.SGD(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    # grad = 2, param = 1 - 0.1 * 2 = 0.8
    assert abs(params[0].item() - 0.8) < 1e-5
    print("SGD passed")

    # Test Adam
    print("Testing Adam...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.Adam(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("Adam passed")

    # Test AdamW
    print("Testing AdamW...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.AdamW(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("AdamW passed")

    # Test RMSprop
    print("Testing RMSprop...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.RMSprop(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("RMSprop passed")

    # Test Adagrad
    print("Testing Adagrad...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.Adagrad(params, lr=0.1)
    opt.zero_grad()
    loss = params[0] * 2
    loss.backward()
    opt.step()
    print("Adagrad passed")

def test_schedulers():
    print("\nTesting Schedulers...")
    params = [tp.Tensor([1.0], requires_grad=True)]
    opt = optim.SGD(params, lr=0.1)
    
    # StepLR
    print("Testing StepLR...")
    scheduler = lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    print(f"Initial LR: {scheduler.get_last_lr()}")
    scheduler.step()
    print(f"Step 1 LR: {scheduler.get_last_lr()}")
    assert abs(scheduler.get_last_lr()[0] - 0.01) < 1e-5
    print("StepLR passed")


def _tensor_values(tensor):
    return [tensor[index].item() for index in range(int(tensor.shape[0]))]


def _assert_values_close(actual, expected, tolerance=5e-5):
    assert len(actual) == len(expected)
    for observed, target in zip(actual, expected):
        assert abs(observed - target) <= tolerance


@pytest.mark.parametrize("device", ["cpu", "cuda"] if tp.cuda.is_available() else ["cpu"])
def test_multi_tensor_optimizer_fast_paths(device):
    # Supplying gradients directly keeps this test focused on optimizer
    # batching; CUDA autograd for scalar reductions is tested separately.
    initial = [[1.0, -2.0], [0.5, 3.0]]
    gradients = [[0.2, -0.4], [0.7, 0.1]]

    params = [
        tp.Tensor(values, device=device, requires_grad=True)
        for values in initial
    ]
    optimizer = optim.SGD(
        params, lr=0.1, momentum=0.9, dampening=0.2, weight_decay=0.01
    )
    expected = [row[:] for row in initial]
    momentum_buffers = [[None] * len(row) for row in initial]
    for step in range(2):
        for param, gradient in zip(params, gradients):
            param.grad = tp.Tensor(gradient, device=device)
        for parameter_index, (values, gradient) in enumerate(zip(expected, gradients)):
            for value_index, grad in enumerate(gradient):
                update = grad + 0.01 * values[value_index]
                if step == 0:
                    momentum_buffers[parameter_index][value_index] = update
                else:
                    momentum_buffers[parameter_index][value_index] = (
                        0.9 * momentum_buffers[parameter_index][value_index]
                        + 0.8 * update
                    )
                values[value_index] -= 0.1 * momentum_buffers[parameter_index][value_index]
        optimizer.step()

    if device == "cuda":
        tp.cuda.synchronize()
    for param, target in zip(params, expected):
        _assert_values_close(_tensor_values(param), target)

    params = [
        tp.Tensor(values, device=device, requires_grad=True)
        for values in initial
    ]
    optimizer = optim.Adam(
        params, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )
    expected = [row[:] for row in initial]
    first_moments = [[0.0] * len(row) for row in initial]
    second_moments = [[0.0] * len(row) for row in initial]
    for step in range(1, 3):
        for param, gradient in zip(params, gradients):
            param.grad = tp.Tensor(gradient, device=device)
        correction1 = 1.0 - 0.9 ** step
        correction2 = 1.0 - 0.999 ** step
        for parameter_index, (values, gradient) in enumerate(zip(expected, gradients)):
            for value_index, grad in enumerate(gradient):
                grad = grad + 0.01 * values[value_index]
                first_moments[parameter_index][value_index] = (
                    0.9 * first_moments[parameter_index][value_index] + 0.1 * grad
                )
                second_moments[parameter_index][value_index] = (
                    0.999 * second_moments[parameter_index][value_index]
                    + 0.001 * grad * grad
                )
                denominator = (
                    math.sqrt(second_moments[parameter_index][value_index])
                    / math.sqrt(correction2)
                    + 1e-8
                )
                values[value_index] -= (
                    0.1 / correction1
                    * first_moments[parameter_index][value_index]
                    / denominator
                )
        optimizer.step()

    if device == "cuda":
        tp.cuda.synchronize()
    for param, target in zip(params, expected):
        _assert_values_close(_tensor_values(param), target)

if __name__ == "__main__":
    test_optimizers()
    test_schedulers()
