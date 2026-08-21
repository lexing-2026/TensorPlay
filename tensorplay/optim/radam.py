import math
import tensorplay as tp

from ._utils import (
    add_weight_decay,
    capturable_supported,
    decoupled_weight_decay,
    ensure_state_step,
    foreach_enabled,
    gradient,
    scalar_value,
    scalar_pow,
    state_step,
    validate_nonnegative,
    validate_unit_interval,
    zeros_like,
)
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import radam as _foreach_radam


class RAdam(Optimizer):
    """Rectified Adam optimizer."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, decoupled_weight_decay=False, *,
                 foreach=None, maximize=False, capturable=False,
                 differentiable=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        if lr_value < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1 = validate_unit_interval(betas[0], "beta parameter at index 0")
        beta2 = validate_unit_interval(betas[1], "beta parameter at index 1")
        eps = validate_nonnegative(eps, "epsilon")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        defaults = dict(
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if p_state and not isinstance(p_state.get("step"), tp.Tensor):
                    p_state["step"] = tp.tensor(
                        float(p_state["step"]), dtype=tp.float32,
                        device=p.device if group["capturable"] else tp.device("cpu"),
                    )

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            lr_value = scalar_value(lr, "lr")
            beta1, beta2 = group["betas"]
            eps = scalar_value(group["eps"], "eps")
            weight_decay = scalar_value(group["weight_decay"], "weight_decay")
            decoupled = group.get("decoupled_weight_decay", False)
            maximize = group.get("maximize", False)
            capturable = group.get("capturable", False)
            differentiable = group.get("differentiable", False)
            rho_inf = 2.0 / (1.0 - beta2) - 1.0

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32,
                        device=p.device if capturable else tp.device("cpu"),
                    )
                    state["exp_avg"] = zeros_like(p)
                    state["exp_avg_sq"] = zeros_like(p)

            if active and foreach_enabled(group, active):
                steps = [
                    ensure_state_step(self.state[p], param=p,
                                      capturable=capturable)
                    for p in active
                ]
                fused_lr = (lr.to(device=active[0].device)
                            if isinstance(lr, tp.Tensor) and
                            lr.device != active[0].device else lr)
                if _foreach_radam(
                        active, [p.grad for p in active],
                        [self.state[p]["exp_avg"] for p in active],
                        [self.state[p]["exp_avg_sq"] for p in active], steps,
                        lr=fused_lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay,
                        decoupled_weight_decay=decoupled,
                        maximize=maximize, capturable=capturable,
                        differentiable=differentiable):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                state = self.state[p]

                grad = gradient(p, maximize)
                if capturable:
                    capturable_supported(p)
                if decoupled:
                    decoupled_weight_decay(p, lr_value, weight_decay)
                else:
                    grad = add_weight_decay(p, grad, weight_decay)
                step_t = state_step(state, param=p, capturable=capturable)
                step = step_t if capturable else float(step_t.item())
                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                exp_avg = tp.view_as_real(state["exp_avg"]) if is_complex else state["exp_avg"]
                exp_avg_sq = tp.view_as_real(state["exp_avg_sq"]) if is_complex else state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                bias_correction1 = 1.0 - scalar_pow(beta1, step)
                bias_correction2 = 1.0 - scalar_pow(beta2, step)
                rho_t = rho_inf - (
                    2.0 * step * scalar_pow(beta2, step) / bias_correction2
                )

                def compute_rect():
                    return (
                        (rho_t - 4.0)
                        * (rho_t - 2.0)
                        * rho_inf
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t)
                    ) ** 0.5

                def compute_adaptive_lr():
                    exp_avg_sq_sqrt = exp_avg_sq.sqrt()
                    if differentiable:
                        exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
                    else:
                        exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)
                    return bias_correction2 ** 0.5 / exp_avg_sq_sqrt

                if capturable:
                    update = tp.where(
                        rho_t > 5.0,
                        compute_rect() * compute_adaptive_lr(),
                        1.0,
                    )
                    param.add_(exp_avg / bias_correction1 * lr * update, alpha=-1.0)
                else:
                    if rho_t > 5.0:
                        param.add_(
                            exp_avg / bias_correction1
                            * compute_adaptive_lr()
                            * lr_value
                            * compute_rect(),
                            alpha=-1.0,
                        )
                    else:
                        param.add_(exp_avg, alpha=-lr_value / bias_correction1)
        return loss
