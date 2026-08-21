"""Shared implementation details for TensorPlay optimizers.

The public optimizer classes intentionally stay close to ``torch.optim``.
These helpers keep validation, state allocation, and the small pieces of
elementwise math identical across the implementations without adding a
dependency on PyTorch itself.
"""

import math

import tensorplay as tp


def scalar_value(value, name="value"):
    """Return a Python float from a scalar or a scalar TensorPlay tensor."""

    if isinstance(value, tp.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be a scalar")
        return float(value.item())
    return float(value)


def validate_nonnegative(value, name):
    value = scalar_value(value, name)
    if value < 0.0:
        raise ValueError(f"Invalid {name} value: {value}")
    return value


def validate_unit_interval(value, name, *, inclusive_one=False):
    value = scalar_value(value, name)
    upper_ok = value <= 1.0 if inclusive_one else value < 1.0
    if value < 0.0 or not upper_ok:
        bound = "between 0 and 1" if inclusive_one else "in [0, 1)"
        raise ValueError(f"Invalid {name} value: {value}; expected {bound}")
    return value


def zeros_like(param):
    # Torch uses memory_format=torch.preserve_format for optimizer state.
    # TensorPlay's generated zeros_like currently materializes a contiguous
    # result, so reproduce preserve_format explicitly for strided parameters.
    if param.is_contiguous():
        return tp.zeros(param.shape, dtype=param.dtype, device=param.device)
    strides = tuple(param.stride())
    if any(stride < 0 for stride in strides):
        return tp.zeros(param.shape, dtype=param.dtype, device=param.device)
    storage_numel = 1
    for size, stride in zip(param.shape, strides):
        if size:
            storage_numel += (size - 1) * stride
    storage = tp.empty((storage_numel,), dtype=param.dtype, device=param.device)
    return storage.as_strided(param.shape, strides).zero_()


def full_like(param, value):
    if param.is_contiguous():
        return tp.full(param.shape, value, dtype=param.dtype, device=param.device)
    strides = tuple(param.stride())
    if any(stride < 0 for stride in strides):
        return tp.full(param.shape, value, dtype=param.dtype, device=param.device)
    storage_numel = 1
    for size, stride in zip(param.shape, strides):
        if size:
            storage_numel += (size - 1) * stride
    storage = tp.empty((storage_numel,), dtype=param.dtype, device=param.device)
    return storage.as_strided(param.shape, strides).fill_(value)


def state_step(state, *, param=None, device=None, capturable=False):
    """Increment Torch-compatible scalar-tensor optimizer state.

    Torch stores optimizer ``step`` as a float32 scalar tensor.  It is hosted
    on CPU for the ordinary path and on the parameter device for capturable
    paths.  Keeping the value tensorized is important for both state_dict
    compatibility and CUDA graph capture.
    """

    step = ensure_state_step(
        state, param=param, device=device, capturable=capturable
    )

    step.add_(1.0)
    return step


def ensure_state_step(state, *, param=None, device=None, capturable=False):
    """Return a Torch-style scalar step without incrementing it."""

    if device is None:
        if capturable and param is not None:
            device = param.device
        else:
            device = tp.device("cpu")

    step = state.get("step")
    if not isinstance(step, tp.Tensor):
        step = tp.tensor(float(step or 0), dtype=tp.float32, device=device)
        state["step"] = step
    elif step.device != device:
        step = step.to(device=device)
        state["step"] = step
    return step


def step_value(step, *, capturable=False):
    """Return a host scalar only on the non-capturable path."""

    if capturable:
        return step
    return scalar_value(step, "step")


def scalar_tensor(value, *, device=None):
    return tp.tensor(float(value), dtype=tp.float32, device=device)


def capturable_supported(param):
    """Torch's eager capturable path is accelerator-only."""

    if not param.is_cuda:
        raise RuntimeError(
            "If capturable=True, params and state_steps must be on a supported device: ['cuda']"
        )


def foreach_enabled(group, params):
    """Resolve Torch's ``foreach=None`` default for an eager parameter group."""

    setting = group.get("foreach", None)
    if setting is True and group.get("differentiable", False):
        raise RuntimeError("_foreach ops don't support autograd")
    if setting is not None:
        return bool(setting)
    if group.get("differentiable", False):
        return False
    # Torch selects foreach by default for accelerator groups and keeps the
    # ordinary CPU path scalar.  Mixed-device groups are split by the caller
    # only when foreach was requested explicitly.
    return bool(params) and all(param.is_cuda for param in params)


def gradient(param, maximize=False):
    grad = param.grad
    if grad is None:
        return None
    return -grad if maximize else grad


def add_weight_decay(param, grad, weight_decay):
    if weight_decay:
        if isinstance(weight_decay, tp.Tensor):
            return grad + param * weight_decay
        return grad.add(param, alpha=weight_decay)
    return grad


def decoupled_weight_decay(param, lr, weight_decay):
    if weight_decay:
        param.mul_(1.0 - lr * weight_decay)


def elementwise_max(lhs, rhs):
    """Torch maximum semantics, including NaN propagation."""

    return tp.maximum(lhs, rhs)


def elementwise_min(lhs, rhs):
    return tp.minimum(lhs, rhs)


def scalar_pow(base, exponent):
    """Power with Tensor exponent support for capturable/differentiable paths."""

    if isinstance(base, tp.Tensor):
        return tp.exp(tp.log(base) * exponent)
    if isinstance(exponent, tp.Tensor):
        return tp.exp(exponent * math.log(float(base)))
    return base ** exponent


def dot(lhs, rhs):
    return float((lhs * rhs).sum().item())


def max_abs(tensor):
    return float(tensor.abs().max().item())


def flatten(tensor):
    return tensor.reshape((-1,))


def require_dense_gradient(param, optimizer_name):
    if param.grad is not None and param.grad.is_sparse:
        raise RuntimeError(
            f"{optimizer_name} does not support sparse gradients"
        )


def check_finite_scalar(value, name):
    value = scalar_value(value, name)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return value
