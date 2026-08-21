import tensorplay as tp

from ._utils import (
    add_weight_decay,
    gradient,
    ensure_state_step,
    foreach_enabled,
    full_like,
    scalar_value,
    scalar_pow,
    state_step,
    validate_nonnegative,
    zeros_like,
)
from .optimizer import Optimizer, _use_grad_for_differentiable
from ._fused import adagrad as _fused_adagrad
from ._foreach import adagrad as _foreach_adagrad


class Adagrad(Optimizer):
    """Adagrad optimizer with the Torch-compatible public options."""

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0,
                 initial_accumulator_value=0, eps=1e-10, foreach=None, *,
                 maximize=False, differentiable=False, fused=None):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if scalar_value(lr, "learning rate") < 0.0:
            raise ValueError(f"Invalid learning rate value: {lr}")
        lr_decay = validate_nonnegative(lr_decay, "lr_decay")
        weight_decay = validate_nonnegative(weight_decay, "weight_decay")
        initial_accumulator_value = validate_nonnegative(
            initial_accumulator_value, "initial_accumulator_value"
        )
        eps = validate_nonnegative(eps, "epsilon")
        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            eps=eps,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused and differentiable:
            raise RuntimeError("`fused` does not support `differentiable`")
        if fused and foreach:
            raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

        # Torch initializes Adagrad's state eagerly in the constructor.
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = tp.tensor(
                    0.0,
                    dtype=tp.float32,
                    device=p.device if group.get("fused") else tp.device("cpu"),
                )
                state["sum"] = full_like(
                    p,
                    (complex(group["initial_accumulator_value"],
                             group["initial_accumulator_value"])
                     if p.is_complex() else group["initial_accumulator_value"]),
                )

    def __setstate__(self, state):
        super().__setstate__(state)
        fused = None
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if p_state and not isinstance(p_state.get("step"), tp.Tensor):
                    p_state["step"] = tp.tensor(
                        float(p_state["step"]), dtype=tp.float32,
                        device=p.device if fused else tp.device("cpu"),
                    )

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = scalar_value(group["lr"], "lr")
            lr_decay = scalar_value(group["lr_decay"], "lr_decay")
            weight_decay = scalar_value(group["weight_decay"], "weight_decay")
            eps = scalar_value(group["eps"], "eps")
            maximize = group.get("maximize", False)
            differentiable = group.get("differentiable", False)
            fused = group.get("fused", False)

            active = [p for p in group["params"] if p.grad is not None]
            if active and fused:
                if any(p.grad.is_sparse or p.is_complex() or
                       not p.is_floating_point() for p in active):
                    raise RuntimeError(
                        "`fused` does not support sparse gradients or complex/non-floating parameters"
                    )
                grads = [p.grad if p.grad.is_contiguous() else p.grad.clone()
                         for p in active]
                state_sums, state_steps = [], []
                for p in active:
                    state = self.state[p]
                    # The fused functional wrapper owns the step increment;
                    # keep the state unchanged when the native entry point is
                    # unavailable so the scalar fallback increments exactly
                    # once.
                    step_t = ensure_state_step(
                        state, param=p, capturable=True
                    )
                    state_sums.append(state["sum"])
                    state_steps.append(step_t)
                fused_lr = group["lr"]
                if isinstance(fused_lr, tp.Tensor) and fused_lr.device != active[0].device:
                    fused_lr = fused_lr.to(device=active[0].device)
                if _fused_adagrad(
                        active, grads, state_sums, state_steps,
                        lr=fused_lr, lr_decay=lr_decay,
                        weight_decay=weight_decay, eps=eps,
                        maximize=maximize,
                        grad_scale=getattr(self, "grad_scale", None),
                        found_inf=getattr(self, "found_inf", None)):
                    continue

            if active and not fused and foreach_enabled(group, active):
                grads = [p.grad if p.grad.is_contiguous() else p.grad.clone()
                         for p in active]
                state_sums = [self.state[p]["sum"] for p in active]
                state_steps = [self.state[p]["step"] for p in active]
                if _foreach_adagrad(
                        active, grads, state_sums, state_steps,
                        lr=group["lr"], lr_decay=lr_decay,
                        weight_decay=weight_decay, eps=eps,
                        maximize=maximize, differentiable=differentiable):
                    continue

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    if weight_decay:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )
                    raise RuntimeError(
                        "TensorPlay does not yet provide sparse Adagrad tensors"
                    )
                state = self.state[p]
                if not state:
                    state["step"] = tp.tensor(
                        0.0, dtype=tp.float32, device=tp.device("cpu")
                    )
                    state["sum"] = full_like(
                        p,
                        (complex(group["initial_accumulator_value"],
                                 group["initial_accumulator_value"])
                         if p.is_complex() else group["initial_accumulator_value"]),
                    )

                grad = gradient(p, maximize)
                grad = add_weight_decay(p, grad, weight_decay)
                step_t = state_step(
                    state, param=p,
                    capturable=bool(group.get("fused", False)),
                )
                step = scalar_value(step_t, "step")
                clr = lr / (1.0 + (step - 1) * lr_decay)
                is_complex = p.is_complex()
                param = tp.view_as_real(p) if is_complex else p
                grad = tp.view_as_real(grad) if is_complex else grad
                state_sum = tp.view_as_real(state["sum"]) if is_complex else state["sum"]
                state_sum.addcmul_(grad, grad, value=1)
                if differentiable:
                    std = state_sum.sqrt() + eps
                else:
                    std = state_sum.sqrt().add_(eps)
                param.addcdiv_(grad, std, value=-clr)
        return loss
