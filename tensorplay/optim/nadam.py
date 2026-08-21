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
from ._foreach import nadam as _foreach_nadam


class NAdam(Optimizer):
    """Nesterov-accelerated Adam optimizer."""

    def __init__(self, params, lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, momentum_decay=0.004,
                 decoupled_weight_decay=False, *, foreach=None,
                 maximize=False, capturable=False, differentiable=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        if lr_value < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        beta1 = validate_unit_interval(betas[0], "beta parameter at index 0")
        beta2 = validate_unit_interval(betas[1], "beta parameter at index 1")
        eps = validate_nonnegative(eps, "epsilon")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        momentum_decay = validate_nonnegative(momentum_decay, "momentum_decay")
        defaults = dict(
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
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
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if not p_state:
                    continue
                state_device = p.device if group["capturable"] else tp.device("cpu")
                for name in ("step", "mu_product"):
                    if not isinstance(p_state.get(name), tp.Tensor):
                        p_state[name] = tp.tensor(
                            float(p_state[name]), dtype=tp.float32,
                            device=state_device,
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
            momentum_decay = scalar_value(group["momentum_decay"], "momentum_decay")
            decoupled = group.get("decoupled_weight_decay", False)
            maximize = group.get("maximize", False)
            capturable = group.get("capturable", False)
            differentiable = group.get("differentiable", False)

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state_device = p.device if capturable else tp.device("cpu")
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32, device=state_device
                    )
                    state["mu_product"] = tp.tensor(
                        1.0, dtype=tp.float32, device=state_device
                    )
                    state["exp_avg"] = zeros_like(p)
                    state["exp_avg_sq"] = zeros_like(p)

            if active and foreach_enabled(group, active):
                steps = [
                    ensure_state_step(self.state[p], param=p,
                                      capturable=capturable)
                    for p in active
                ]
                mu_products = [self.state[p]["mu_product"] for p in active]
                fused_lr = (lr.to(device=active[0].device)
                            if isinstance(lr, tp.Tensor) and
                            lr.device != active[0].device else lr)
                if _foreach_nadam(
                        active, [p.grad for p in active],
                        [self.state[p]["exp_avg"] for p in active],
                        [self.state[p]["exp_avg_sq"] for p in active],
                        mu_products, steps, beta1=beta1, beta2=beta2,
                        lr=fused_lr, weight_decay=weight_decay,
                        momentum_decay=momentum_decay, eps=eps,
                        decoupled_weight_decay=decoupled, maximize=maximize,
                        capturable=capturable, differentiable=differentiable):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
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
                bias_correction2 = 1.0 - scalar_pow(beta2, step)
                mu = beta1 * (1.0 - 0.5 * scalar_pow(0.96, step * momentum_decay))
                mu_next = beta1 * (
                    1.0 - 0.5 * scalar_pow(0.96, (step + 1) * momentum_decay)
                )
                mu_product = state["mu_product"]
                mu_product.mul_(mu)

                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                exp_avg = tp.view_as_real(state["exp_avg"]) if is_complex else state["exp_avg"]
                exp_avg_sq = tp.view_as_real(state["exp_avg_sq"]) if is_complex else state["exp_avg_sq"]
                exp_avg.lerp_(grad, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.div(bias_correction2).sqrt()
                if differentiable or capturable:
                    denom = denom.add(eps)
                    mu_product_next = mu_product * mu_next
                    grad_update = grad * (
                        -lr * (1.0 - mu) / (1.0 - mu_product)
                    )
                    exp_avg_update = exp_avg * (
                        -lr * mu_next / (1.0 - mu_product_next)
                    )
                    param.addcdiv_(grad_update, denom)
                    param.addcdiv_(exp_avg_update, denom)
                else:
                    denom.add_(eps)
                    mu_product_next = float(mu_product.item()) * mu_next
                    param.addcdiv_(
                        grad,
                        denom,
                        value=-lr_value
                        * (1.0 - mu)
                        / (1.0 - float(mu_product.item())),
                    )
                    param.addcdiv_(
                        exp_avg,
                        denom,
                        value=-lr_value
                        * mu_next
                        / (1.0 - mu_product_next),
                    )
        return loss
