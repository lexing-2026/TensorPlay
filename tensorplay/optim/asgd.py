import tensorplay as tp

from ._utils import (
    add_weight_decay,
    capturable_supported,
    ensure_state_step,
    foreach_enabled,
    gradient,
    scalar_value,
    state_step,
    validate_nonnegative,
    zeros_like,
)
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import asgd as _foreach_asgd


class ASGD(Optimizer):
    """Averaged stochastic gradient descent."""

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75,
                 t0=1e6, weight_decay=0, foreach=None, maximize=False,
                 differentiable=False, capturable=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        if lr_value < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        lambd = validate_nonnegative(lambd, "lambd")
        alpha = validate_nonnegative(alpha, "alpha")
        t0 = validate_nonnegative(t0, "t0")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
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
                if not p_state:
                    continue
                for name, default in (("step", 0.0), ("eta", group["lr"]),
                                      ("mu", 1.0)):
                    if not isinstance(p_state.get(name), tp.Tensor):
                        value = default if name != "step" else p_state[name]
                        p_state[name] = tp.tensor(
                            scalar_value(value, name), dtype=tp.float32,
                            device=p.device,
                        )

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            lr_value = scalar_value(lr, "lr")
            lambd = scalar_value(group["lambd"], "lambd")
            alpha = scalar_value(group["alpha"], "alpha")
            t0 = scalar_value(group["t0"], "t0")
            weight_decay = scalar_value(group["weight_decay"], "weight_decay")
            maximize = group.get("maximize", False)
            capturable = group.get("capturable", False)
            differentiable = group.get("differentiable", False)

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("ASGD does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32, device=p.device
                    )
                    state["eta"] = tp.tensor(
                        lr_value, dtype=tp.float32, device=p.device
                    )
                    state["mu"] = tp.tensor(
                        1.0, dtype=tp.float32, device=p.device
                    )
                    state["ax"] = zeros_like(p)

            if active and foreach_enabled(group, active):
                steps = [ensure_state_step(self.state[p], param=p, device=p.device)
                         for p in active]
                if _foreach_asgd(
                        active, [p.grad for p in active],
                        [self.state[p]["ax"] for p in active],
                        [self.state[p]["mu"] for p in active],
                        [self.state[p]["eta"] for p in active], steps,
                        lr=lr, lambd=lambd, t0=t0, alpha=alpha,
                        weight_decay=weight_decay, maximize=maximize,
                        capturable=capturable, differentiable=differentiable):
                    continue
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("ASGD does not support sparse gradients")
                state = self.state[p]

                grad = gradient(p, maximize)
                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                ax = tp.view_as_real(state["ax"]) if is_complex else state["ax"]
                grad = add_weight_decay(param, grad, weight_decay)
                if capturable:
                    capturable_supported(p)
                step_t = state_step(state, param=p, device=p.device)
                eta = state["eta"]
                mu = state["mu"]

                if capturable:
                    param.mul_(1.0 - lambd * eta)
                    param.add_(grad * eta, alpha=-1.0)
                else:
                    eta_value = float(eta.item())
                    param.mul_(1.0 - lambd * eta_value)
                    param.add_(grad, alpha=-eta_value)

                if capturable or float(mu.item()) != 1.0:
                    ax.add_(param.sub(ax).mul(mu))
                else:
                    ax.copy_(param)

                if capturable:
                    eta.copy_(lr / ((1.0 + lambd * lr * step_t) ** alpha))
                    mu.copy_(1.0 / tp.maximum(step_t - t0, tp.ones_like(step_t)))
                else:
                    step = float(step_t.item())
                    eta.copy_(tp.tensor(
                        lr_value / ((1.0 + lambd * lr_value * step) ** alpha),
                        dtype=tp.float32,
                        device=p.device,
                    ))
                    mu.copy_(tp.tensor(
                        1.0 / max(1.0, step - t0),
                        dtype=tp.float32,
                        device=p.device,
                    ))
        return loss
