import tensorplay as tp

from ._utils import (
    add_weight_decay,
    capturable_supported,
    ensure_state_step,
    foreach_enabled,
    gradient,
    scalar_value,
    state_step,
    scalar_pow,
    validate_nonnegative,
    validate_unit_interval,
    zeros_like,
)
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import adamax as _foreach_adamax


class Adamax(Optimizer):
    """Adamax optimizer matching ``torch.optim.Adamax``."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, foreach=None, *, maximize=False,
                 differentiable=False, capturable=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        if lr_value < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        eps = validate_nonnegative(eps, "epsilon")
        beta1 = validate_unit_interval(betas[0], "beta parameter at index 0")
        beta2 = validate_unit_interval(betas[1], "beta parameter at index 1")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        defaults = dict(
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
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
            beta1, beta2 = group["betas"]
            eps = scalar_value(group["eps"], "eps")
            weight_decay = scalar_value(group["weight_decay"], "weight_decay")
            maximize = group.get("maximize", False)
            capturable = group.get("capturable", False)
            differentiable = group.get("differentiable", False)

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("Adamax does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32,
                        device=p.device if capturable else tp.device("cpu"),
                    )
                    state["exp_avg"] = zeros_like(p)
                    state["exp_inf"] = zeros_like(p)

            if active and foreach_enabled(group, active):
                steps = [
                    ensure_state_step(self.state[p], param=p,
                                      capturable=capturable)
                    for p in active
                ]
                if _foreach_adamax(
                        active, [p.grad for p in active],
                        [self.state[p]["exp_avg"] for p in active],
                        [self.state[p]["exp_inf"] for p in active], steps,
                        lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, maximize=maximize,
                        capturable=capturable, differentiable=differentiable):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adamax does not support sparse gradients")
                state = self.state[p]

                grad = gradient(p, maximize)
                grad = add_weight_decay(p, grad, weight_decay)
                if capturable:
                    capturable_supported(p)
                step_t = state_step(state, param=p, capturable=capturable)
                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                exp_avg = tp.view_as_real(state["exp_avg"]) if is_complex else state["exp_avg"]
                exp_inf = tp.view_as_real(state["exp_inf"]) if is_complex else state["exp_inf"]
                exp_avg.lerp_(grad, 1.0 - beta1)
                decayed_inf = exp_inf.mul(beta2)
                current_inf = grad.abs().add(eps)
                exp_inf.copy_(tp.maximum(decayed_inf, current_inf))
                if capturable:
                    neg_bias_correction = scalar_pow(beta1, step_t) - 1.0
                    neg_bias_correction.div_(lr)
                    denom = exp_inf * neg_bias_correction
                    param.addcdiv_(exp_avg, denom)
                else:
                    step = float(step_t.item())
                    clr = scalar_value(lr, "lr") / (1.0 - scalar_pow(beta1, step))
                    param.addcdiv_(exp_avg, exp_inf, value=-clr)
        return loss
