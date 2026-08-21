import tensorplay as tp

from ._utils import (
    add_weight_decay,
    capturable_supported,
    gradient,
    scalar_value,
    foreach_enabled,
    ensure_state_step,
    state_step,
    validate_nonnegative,
    validate_unit_interval,
    zeros_like,
)
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import rmsprop as _foreach_rmsprop


class RMSprop(Optimizer):
    """RMSprop optimizer matching Torch's centered and momentum variants."""

    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8,
                 weight_decay=0, momentum=0, centered=False,
                 capturable=False, foreach=None, maximize=False,
                 differentiable=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        lr = lr if isinstance(lr, tp.Tensor) else lr_value
        if lr_value < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        alpha = validate_nonnegative(alpha, "alpha")
        eps = validate_nonnegative(eps, "epsilon")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        momentum = validate_nonnegative(momentum, "momentum")
        defaults = dict(
            lr=lr,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            centered=centered,
            capturable=capturable,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
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
            alpha = scalar_value(group["alpha"], "alpha")
            eps = scalar_value(group["eps"], "eps")
            weight_decay = scalar_value(group["weight_decay"], "weight_decay")
            momentum = scalar_value(group["momentum"], "momentum")
            centered = group["centered"]
            maximize = group.get("maximize", False)
            capturable = group.get("capturable", False)
            differentiable = group.get("differentiable", False)

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32,
                        device=p.device if capturable else tp.device("cpu"),
                    )
                    state["square_avg"] = zeros_like(p)
                    if momentum > 0:
                        state["momentum_buffer"] = zeros_like(p)
                    if centered:
                        state["grad_avg"] = zeros_like(p)

            if active and foreach_enabled(group, active):
                steps = [
                    ensure_state_step(self.state[p], param=p,
                                      capturable=capturable)
                    for p in active
                ]
                square_avgs = [self.state[p]["square_avg"] for p in active]
                grad_avgs = ([self.state[p]["grad_avg"] for p in active]
                             if centered else [])
                momentum_buffers = ([self.state[p]["momentum_buffer"] for p in active]
                                    if momentum > 0 else [])
                if _foreach_rmsprop(
                        active, [p.grad for p in active], square_avgs,
                        grad_avgs, momentum_buffers, steps,
                        lr=lr, alpha=alpha, eps=eps,
                        weight_decay=weight_decay, momentum=momentum,
                        centered=centered, maximize=maximize,
                        capturable=capturable, differentiable=differentiable):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                state = self.state[p]

                grad = gradient(p, maximize)
                grad = add_weight_decay(p, grad, weight_decay)
                if capturable:
                    capturable_supported(p)
                state_step(state, param=p, capturable=capturable)
                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                square_avg = tp.view_as_real(state["square_avg"]) if is_complex else state["square_avg"]
                momentum_buffer = (
                    tp.view_as_real(state["momentum_buffer"])
                    if momentum > 0 and is_complex else
                    (state["momentum_buffer"] if momentum > 0 else None)
                )
                grad_avg = (
                    tp.view_as_real(state["grad_avg"])
                    if centered and is_complex else
                    (state["grad_avg"] if centered else None)
                )
                square_avg.mul_(alpha).addcmul_(grad, grad, value=1.0 - alpha)

                if centered:
                    grad_avg.lerp_(grad, 1.0 - alpha)
                    avg = square_avg.addcmul(
                        grad_avg, grad_avg, value=-1.0
                    ).sqrt_()
                else:
                    avg = square_avg.sqrt()
                if differentiable:
                    avg = avg.add(eps)
                else:
                    avg.add_(eps)

                if momentum > 0:
                    buf = momentum_buffer
                    buf.mul_(momentum).addcdiv_(grad, avg)
                    if isinstance(lr, tp.Tensor):
                        param.add_(buf * (-lr))
                    else:
                        param.add_(buf, alpha=-lr)
                else:
                    if isinstance(lr, tp.Tensor):
                        param.add_(grad / avg * (-lr))
                    else:
                        param.addcdiv_(grad, avg, value=-lr)
        return loss
