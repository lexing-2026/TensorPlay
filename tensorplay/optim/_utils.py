"""Shared implementation details for TensorPlay optimizers.

These helpers centralize validation, state allocation, and small pieces of
elementwise math without adding a
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
    # TensorPlay's generated zeros_like currently materializes a contiguous
    # result, so reproduce preserve_format explicitly for strided parameters.
    if param.is_contiguous():
        return tp.zeros(list(param.shape), dtype=param.dtype, device=param.device)
    strides = tuple(param.stride())
    if any(stride < 0 for stride in strides):
        return tp.zeros(list(param.shape), dtype=param.dtype, device=param.device)
    storage_numel = 1
    for size, stride in zip(param.shape, strides):
        if size:
            storage_numel += (size - 1) * stride
    # Initialize the backing storage before creating the view.  TensorPlay's
    # zero_ kernel intentionally follows the contiguous fast path for a raw
    # strided view, which can leave the portion after the first contiguous
    # chunk untouched.  Filling the 1-D storage itself is both safe for gaps
    storage = tp.zeros((storage_numel,), dtype=param.dtype, device=param.device)
    return storage.as_strided(param.shape, strides)


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
    storage = tp.full((storage_numel,), value, dtype=param.dtype, device=param.device)
    return storage.as_strided(param.shape, strides)


def state_step(state, *, param=None, device=None, capturable=False):
    """

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

    if device is None:
        if param is not None and param.device.type != "cpu":
            # Keep step counters colocated with non-CPU params: the native
            # fused/foreach kernels read them on-device, and a host-side
            # bump would force a dispatcher round trip per parameter.
            device = param.device
        elif capturable and param is not None:
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

    if not param.is_cuda:
        raise RuntimeError(
            "If capturable=True, params and state_steps must be on a supported device: ['cuda']"
        )


def foreach_enabled(group, params):

    setting = group.get("foreach", None)
    if setting is True and group.get("differentiable", False):
        raise RuntimeError("_foreach ops don't support autograd")
    if group.get("capturable", False) and params:
        for param in params:
            capturable_supported(param)
    if setting is not None:
        return bool(setting)
    if group.get("differentiable", False):
        return False
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
