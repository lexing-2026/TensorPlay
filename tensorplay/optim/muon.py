import math

import tensorplay as tp

from ._utils import scalar_value, validate_nonnegative, zeros_like
from .optimizer import Optimizer


EPS = 1e-7
DEFAULT_A = 3.4445
DEFAULT_B = -4.7750
DEFAULT_C = 2.0315
DEFAULT_NS_STEPS = 5


def _zeropower_via_newtonschulz(grad, coefficients, ns_steps, eps):
    if ns_steps >= 100:
        raise ValueError("Number of steps must be less than 100")
    if grad.ndim != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = coefficients
    original_dtype = grad.dtype

    # Keep the work tensor in bfloat16 for the whole iteration, exactly as
    # torch.optim.Muon does.  TensorPlay's matmul kernels have native
    # bfloat16 accumulation, so converting each intermediate back to
    # float32 would introduce a different rounding point.
    work = grad.to(tp.bfloat16)
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.t()
    work.div_(work.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram = work @ work.t()
        gram_update = tp.addmm(gram, gram, gram, beta=b, alpha=c)
        work = tp.addmm(work, gram_update, work, beta=a)
    if transposed:
        work = work.t()
    return work


def _adjust_lr(lr, adjust_lr_fn, shape):
    a, b = shape[0], shape[1]
    if adjust_lr_fn is None or adjust_lr_fn == "original":
        ratio = math.sqrt(max(1.0, float(a) / float(b)))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(a, b))
    elif adjust_lr_fn == "spectral_unclamped":
        ratio = math.sqrt(float(a) / float(b))
    else:
        ratio = 1.0
    return lr * ratio


class Muon(Optimizer):
    """Muon optimizer with Newton-Schulz orthogonalized momentum."""

    def __init__(self, params, lr=1e-3, weight_decay=0.1, momentum=0.95,
                 nesterov=True,
                 ns_coefficients=(DEFAULT_A, DEFAULT_B, DEFAULT_C),
                 eps=EPS, ns_steps=DEFAULT_NS_STEPS, adjust_lr_fn=None):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if scalar_value(lr, "learning rate") < 0.0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if momentum < 0.0:
            raise ValueError(f"momentum should be >= 0 but is: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"weight decay should be >= 0 but is: {weight_decay}")
        if adjust_lr_fn not in (
            None, "original", "match_rms_adamw", "spectral_unclamped"
        ):
            raise ValueError(
                f"Adjust learning rate function {adjust_lr_fn} is not supported"
            )
        if len(ns_coefficients) != 3:
            raise ValueError("Coefficients must be a tuple of exactly 3 values")
        if ns_steps < 0 or ns_steps >= 100:
            raise ValueError("ns_steps must be in [0, 100)")
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_coefficients=tuple(float(x) for x in ns_coefficients),
            eps=eps,
            ns_steps=ns_steps,
            adjust_lr_fn=adjust_lr_fn,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(
                        "Muon only supports 2D parameters whereas found "
                        f"{p.shape}"
                    )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()
        with tp.no_grad():
            for group in self.param_groups:
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                momentum = group["momentum"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.is_complex():
                        raise RuntimeError("Muon does not support complex parameters")
                    if p.grad is None:
                        continue
                    if p.grad.ndim != 2:
                        raise ValueError("Param gradient must be a 2D matrix")
                    if p.grad.is_sparse:
                        raise RuntimeError("Muon does not support sparse gradients")
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = zeros_like(p.grad)
                    buf = state["momentum_buffer"]
                    grad = p.grad
                    buf.lerp_(grad, 1.0 - momentum)
                    update = grad.lerp(buf, momentum) if group["nesterov"] else buf
                    update = _zeropower_via_newtonschulz(
                        update,
                        group["ns_coefficients"],
                        group["ns_steps"],
                        eps,
                    )
                    adjusted_lr = _adjust_lr(
                        lr, group["adjust_lr_fn"], p.shape
                    )
                    p.mul_(1.0 - lr * weight_decay)
                    p.add_(update, alpha=-adjusted_lr)
        return loss
