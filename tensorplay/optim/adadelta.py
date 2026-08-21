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
    validate_unit_interval,
    zeros_like,
)
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import adadelta as _foreach_adadelta


class Adadelta(Optimizer):
    """Adadelta optimizer matching ``torch.optim.Adadelta``."""

    def __init__(self, params, lr=1.0, rho=0.9, eps=1e-6,
                 weight_decay=0, foreach=None, *, capturable=False,
                 maximize=False, differentiable=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        if lr_value < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        rho = validate_unit_interval(rho, "rho", inclusive_one=True)
        eps = validate_nonnegative(eps, "epsilon")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            capturable=capturable,
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
            rho = scalar_value(group["rho"], "rho")
            eps = scalar_value(group["eps"], "eps")
            weight_decay = scalar_value(group["weight_decay"], "weight_decay")
            maximize = group.get("maximize", False)
            capturable = group.get("capturable", False)
            differentiable = group.get("differentiable", False)

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("Adadelta does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32,
                        device=p.device if capturable else tp.device("cpu"),
                    )
                    state["square_avg"] = zeros_like(p)
                    state["acc_delta"] = zeros_like(p)

            if active and foreach_enabled(group, active):
                steps = [
                    ensure_state_step(self.state[p], param=p,
                                      capturable=capturable)
                    for p in active
                ]
                if _foreach_adadelta(
                        active, [p.grad for p in active],
                        [self.state[p]["square_avg"] for p in active],
                        [self.state[p]["acc_delta"] for p in active], steps,
                        lr=lr, rho=rho, eps=eps,
                        weight_decay=weight_decay, maximize=maximize,
                        capturable=capturable, differentiable=differentiable):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adadelta does not support sparse gradients")
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
                acc_delta = tp.view_as_real(state["acc_delta"]) if is_complex else state["acc_delta"]
                square_avg.mul_(rho).addcmul_(grad, grad, value=1.0 - rho)
                std = square_avg.add(eps).sqrt_()
                delta = acc_delta.add(eps).sqrt_()
                if differentiable:
                    delta = delta.clone()
                delta.div_(std).mul_(grad)
                acc_delta.mul_(rho).addcmul_(
                    delta, delta, value=1.0 - rho
                )
                if isinstance(lr, tp.Tensor):
                    param.add_(delta * (-lr))
                else:
                    param.add_(delta, alpha=-lr)
        return loss
