import math

import tensorplay as tp

from ._utils import (
    decoupled_weight_decay,
    ensure_state_step,
    gradient,
    scalar_value,
    state_step,
    validate_nonnegative,
    zeros_like,
)
from .optimizer import Optimizer
from ._foreach import adafactor as _foreach_adafactor


def _dtype_epsilon(param):
    if param.dtype == tp.float16:
        return 9.765625e-4
    if param.dtype == tp.bfloat16:
        return 7.8125e-3
    if param.dtype == tp.float64:
        return 2.220446049250313e-16
    return 1.1920928955078125e-7


class Adafactor(Optimizer):
    """Memory-efficient Adafactor with Torch-compatible defaults."""

    def __init__(self, params, lr=1e-2, beta2_decay=-0.8,
                 eps=(None, 1e-3), d=1.0, weight_decay=0.0, *,
                 foreach=None, maximize=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        if lr_value < 0:
            raise ValueError(f"Learning rate should be >= 0 but is: {lr}")
        if beta2_decay > 0.0:
            raise ValueError(
                f"beta2_decay should be <= 0 but is: {beta2_decay}"
            )
        eps1, eps2 = eps
        if eps1 is not None:
            eps1 = validate_nonnegative(eps1, "epsilon1")
        eps2 = validate_nonnegative(eps2, "epsilon2")
        if d < 1.0:
            raise ValueError(f"Clipping threshold d should be >= 1 but is: {d}")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        defaults = dict(
            lr=lr,
            beta2_decay=beta2_decay,
            eps=(eps1, eps2),
            d=d,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if p_state and not isinstance(p_state.get("step"), tp.Tensor):
                    p_state["step"] = tp.tensor(
                        float(p_state["step"]), dtype=tp.float32,
                        device=tp.device("cpu"),
                    )

    def _initialize_state(self, p, eps):
        state = self.state[p]
        if state:
            return state
        state["step"] = tp.tensor(0.0, dtype=tp.float32, device=tp.device("cpu"))
        if p.ndim > 1:
            row_shape = list(p.shape)
            row_shape[-1] = 1
            col_shape = list(p.shape)
            col_shape[-2] = 1
            state["row_var"] = tp.zeros(
                row_shape, dtype=p.dtype, device=p.device
            )
            state["col_var"] = tp.zeros(
                col_shape, dtype=p.dtype, device=p.device
            )
        else:
            state["variance"] = zeros_like(p)
        return state

    @tp.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            lr_value = scalar_value(lr, "lr")
            beta2_decay = scalar_value(group["beta2_decay"], "beta2_decay")
            eps1, eps2 = group["eps"]
            d = scalar_value(group["d"], "d")
            weight_decay = scalar_value(group["weight_decay"], "weight_decay")
            maximize = group.get("maximize", False)

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradients")
                if p.is_complex():
                    raise RuntimeError("Adafactor does not support complex parameters")
                self._initialize_state(p, (eps1, eps2))

            if active and group.get("foreach") is True:
                steps = [ensure_state_step(self.state[p], param=p)
                         for p in active]
                if _foreach_adafactor(
                        active, [p.grad for p in active],
                        [self.state[p].get("row_var") for p in active],
                        [self.state[p].get("col_var") for p in active],
                        [self.state[p].get("variance") for p in active],
                        steps, lr=lr, beta2_decay=beta2_decay,
                        weight_decay=weight_decay, eps1=eps1, eps2=eps2,
                        d=d, maximize=maximize):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._initialize_state(p, (eps1, eps2))
                grad = gradient(p, maximize)
                step_t = state_step(state)
                step = scalar_value(step_t, "step")
                if eps1 is None:
                    eps1_value = _dtype_epsilon(p)
                else:
                    eps1_value = eps1

                one_minus_beta2_t = step ** beta2_decay
                rho_t = min(lr_value, 1.0 / math.sqrt(step))
                alpha = max(
                    eps2,
                    float(p.norm(2).item()) / math.sqrt(p.numel()),
                ) * rho_t
                # Torch updates the parameter object itself here.  Going
                # through ``.data`` would discard the differentiable view of
                # the optimizer step and makes higher-order gradients observe
                # a silent pointer mutation.
                decoupled_weight_decay(p, lr_value, weight_decay)

                if p.ndim > 1:
                    row_var = state["row_var"]
                    col_var = state["col_var"]
                    row_mean = (
                        grad.norm(dim=-1, keepdim=True)
                        .square()
                        .div_(p.shape[-1])
                    )
                    col_mean = (
                        grad.norm(dim=-2, keepdim=True)
                        .square()
                        .div_(p.shape[-2])
                    )
                    row_var.lerp_(row_mean, one_minus_beta2_t)
                    col_var.lerp_(col_mean, one_minus_beta2_t)
                    var_estimate = row_var @ col_var
                    row_mean_factor = row_var.mean(
                        dim=-2, keepdim=True
                    ).clamp(min=eps1_value)
                    var_estimate = var_estimate.div(row_mean_factor)
                else:
                    variance = state["variance"]
                    variance.lerp_(grad * grad, one_minus_beta2_t)
                    var_estimate = variance.clone()

                update = var_estimate.clamp(
                    min=eps1_value * eps1_value
                ).rsqrt()
                update.mul_(grad)
                clip = max(
                    1.0,
                    float(update.norm(2).item())
                    / (math.sqrt(update.numel()) * d),
                )
                p.add_(update, alpha=-alpha / clip)
        return loss
