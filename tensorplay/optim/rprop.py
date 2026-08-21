import tensorplay as tp

from ._utils import (
    capturable_supported,
    ensure_state_step,
    foreach_enabled,
    full_like,
    gradient,
    scalar_value,
    state_step,
    validate_nonnegative,
    zeros_like,
)
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._foreach import rprop as _foreach_rprop


class Rprop(Optimizer):
    """Resilient backpropagation optimizer."""

    def __init__(self, params, lr=1e-2, etas=(0.5, 1.2),
                 step_sizes=(1e-6, 50), *, capturable=False,
                 foreach=None, maximize=False, differentiable=False):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        lr_value = scalar_value(lr, "learning rate")
        if lr_value < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        etaminus, etaplus = etas
        if not 0.0 < etaminus < 1.0 < etaplus:
            raise ValueError(f"Invalid eta values: {etaminus}, {etaplus}")
        step_min, step_max = step_sizes
        step_min = validate_nonnegative(step_min, "minimum step size")
        step_max = validate_nonnegative(step_max, "maximum step size")
        if step_min > step_max:
            raise ValueError("minimum step size cannot exceed maximum step size")
        defaults = dict(
            lr=lr,
            etas=(etaminus, etaplus),
            step_sizes=(step_min, step_max),
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
            etaminus, etaplus = group["etas"]
            step_min, step_max = group["step_sizes"]
            maximize = group.get("maximize", False)
            capturable = group.get("capturable", False)
            differentiable = group.get("differentiable", False)

            active = [p for p in group["params"] if p.grad is not None]
            for p in active:
                if p.grad.is_sparse:
                    raise RuntimeError("Rprop does not support sparse gradients")
                state = self.state[p]
                if not state:
                    lr_value = scalar_value(group["lr"], "lr")
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32,
                        device=p.device if capturable else tp.device("cpu"),
                    )
                    state["prev"] = zeros_like(p)
                    state["step_size"] = full_like(
                        p,
                        complex(lr_value, lr_value) if p.is_complex() else lr_value,
                    )

            if active and foreach_enabled(group, active):
                steps = [
                    ensure_state_step(self.state[p], param=p,
                                      capturable=capturable)
                    for p in active
                ]
                if _foreach_rprop(
                        active, [p.grad for p in active],
                        [self.state[p]["prev"] for p in active],
                        [self.state[p]["step_size"] for p in active], steps,
                        step_size_min=step_min, step_size_max=step_max,
                        etaminus=etaminus, etaplus=etaplus,
                        maximize=maximize, capturable=capturable,
                        differentiable=differentiable):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Rprop does not support sparse gradients")
                state = self.state[p]

                grad = gradient(p, maximize)
                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                if capturable:
                    capturable_supported(p)
                state_step(state, param=p, capturable=capturable)
                prev = tp.view_as_real(state["prev"]) if is_complex else state["prev"]
                step_size = tp.view_as_real(state["step_size"]) if is_complex else state["step_size"]
                sign = grad.mul(prev.clone() if differentiable else prev).sign()
                positive = sign.gt(0)
                negative = sign.lt(0)
                zero = sign.eq(0)
                sign = (positive.to(dtype=step_size.dtype) * etaplus +
                        negative.to(dtype=step_size.dtype) * etaminus + zero)
                step_size.copy_((step_size * sign).clamp(
                    min=step_min, max=step_max
                ))

                # A sign reversal cancels this update. The zeroed gradient is
                # also what Torch stores as the previous gradient.
                grad = grad.clone()
                grad = tp.where(sign.eq(etaminus), 0.0, grad)
                param.addcmul_(grad.sign(), step_size, value=-1.0)
                prev.copy_(grad)
        return loss
