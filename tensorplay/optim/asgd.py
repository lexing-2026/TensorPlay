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
from .optimizer import (
    Optimizer,
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
)


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

    def _init_group(self, group, params_with_grad, grads, mus, axs, etas, state_steps):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= p.is_complex()
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("ASGD does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = tp.zeros((), dtype=_get_scalar_dtype())
                state["eta"] = (
                    tp.tensor(
                        _to_scalar(group["lr"]),
                        device=p.device,
                        dtype=_get_scalar_dtype(),
                    )
                    .clone()
                    .detach()
                )
                state["mu"] = tp.ones((), device=p.device, dtype=_get_scalar_dtype())
                state["ax"] = zeros_like(p)

            mus.append(state["mu"])
            axs.append(state["ax"])
            etas.append(state["eta"])
            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        self._accelerator_graph_capture_health_check()

        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            mus = []
            axs = []
            etas = []
            state_steps = []

            has_complex = self._init_group(
                group, params_with_grad, grads, mus, axs, etas, state_steps
            )
            asgd(
                params_with_grad,
                grads,
                axs,
                mus,
                etas,
                state_steps,
                lambd=group["lambd"],
                lr=group["lr"],
                t0=group["t0"],
                alpha=group["alpha"],
                weight_decay=group["weight_decay"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )
        return loss
def _single_tensor_asgd(
    params,
    grads,
    axs,
    mus,
    etas,
    state_steps,
    *,
    lambd,
    lr,
    t0,
    alpha,
    weight_decay,
    maximize,
    differentiable,
    capturable,
    has_complex,
):
    lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        mu = mus[i]
        ax = axs[i]
        eta = etas[i]
        step_t = state_steps[i]

        if not tp.compiler.is_compiling() and capturable:
            supported = _get_capturable_supported_devices()
            if not (
                param.device.type
                == mu.device.type
                == eta.device.type
                == step_t.device.type
                and param.device.type in supported
            ):
                raise AssertionError(
                    "If capturable=True, params, mus, etas, and state_steps "
                    f"must be on supported devices: {supported}."
                )

        if param.is_complex():
            grad = tp.view_as_real(grad)
            param = tp.view_as_real(param)
            ax = tp.view_as_real(ax)

        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        if capturable:
            param.mul_(1 - lambd * eta)
            param.addcmul_(grad, eta, value=-1)
        else:
            eta_value = _get_value(eta)
            param.mul_(1 - lambd * eta_value)
            param.add_(grad, alpha=-eta_value)

        if capturable or mu.item() != 1:
            ax.add_(param.sub(ax).mul_(mu))
        else:
            ax.copy_(param)

        if capturable:
            eta.copy_(lr / ((1 + lambd * lr * step_t) ** alpha))
            mu.copy_(1 / tp.maximum(step_t - t0, tp.ones_like(step_t)))
        else:
            step = _get_value(step_t)
            new_eta = tp.as_tensor(
                lr / ((1 + lambd * lr * step) ** alpha),
                device=eta.device,
            )
            eta.copy_(new_eta)
            new_mu = tp.as_tensor(1 / max(1, step - t0), device=mu.device)
            mu.copy_(new_mu)


def _multi_tensor_asgd(
    params,
    grads,
    axs,
    mus,
    etas,
    state_steps,
    *,
    lambd,
    lr,
    t0,
    alpha,
    weight_decay,
    maximize,
    differentiable,
    capturable,
    has_complex,
):
    if not params:
        return
    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    if not tp.compiler.is_compiling() and capturable:
        supported = _get_capturable_supported_devices(supports_xla=False)
        if not all(
            p.device.type == mu.device.type == eta.device.type == step.device.type
            and p.device.type in supported
            for p, mu, eta, step in zip(params, mus, etas, state_steps)
        ):
            raise AssertionError(
                "If capturable=True, params, mus, etas, and state_steps "
                f"must be on supported devices: {supported}."
            )

    lr = _to_scalar(lr)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, axs, mus, etas, state_steps]
    )
    for (
        grouped_params,
        grouped_grads,
        grouped_axs,
        grouped_mus,
        grouped_etas,
        grouped_state_steps,
    ), _ in grouped_tensors.values():
        if not grouped_params:
            continue
        if has_complex:
            _view_as_real(grouped_params, grouped_grads, grouped_axs)

        if maximize:
            grouped_grads = tp._foreach_neg(grouped_grads)

        if (
            not tp.compiler.is_compiling()
            and grouped_state_steps[0].device.type == "cpu"
        ):
            tp._foreach_add_(
                grouped_state_steps,
                tp.tensor(
                    1.0,
                    dtype=grouped_state_steps[0].dtype,
                    device=tp.device("cpu"),
                ),
                alpha=1.0,
            )
        else:
            tp._foreach_add_(grouped_state_steps, 1)

        if weight_decay != 0:
            if maximize:
                tp._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
                intermediate = grouped_grads
            else:
                intermediate = tp._foreach_add(
                    grouped_grads, grouped_params, alpha=weight_decay
                )
            tp._foreach_add_(intermediate, grouped_params, alpha=lambd)
        else:
            intermediate = tp._foreach_add(
                grouped_grads, grouped_params, alpha=lambd
            )

        tp._foreach_addcmul_(grouped_params, intermediate, grouped_etas, value=-1)
        intermediate = tp._foreach_sub(grouped_params, grouped_axs)
        tp._foreach_addcmul_(grouped_axs, intermediate, grouped_mus)

        if capturable:
            new_mus = tp._foreach_sub(grouped_state_steps, t0)
            tp._foreach_maximum_(new_mus, 1.0)
            tp._foreach_reciprocal_(new_mus)
            tp._foreach_copy_(grouped_mus, new_mus)

            new_etas = tp._foreach_mul(grouped_state_steps, lambd)
            tp._foreach_mul_(new_etas, lr)
            tp._foreach_add_(new_etas, 1)
            tp._foreach_pow_(new_etas, alpha)
            tp._foreach_reciprocal_(new_etas)
            tp._foreach_mul_(new_etas, lr)
            tp._foreach_copy_(grouped_etas, new_etas)
        else:
            device = grouped_etas[0].device
            new_etas = [
                tp.as_tensor(
                    lr / ((1 + lambd * lr * _get_value(step)) ** alpha),
                    device=device,
                )
                for step in grouped_state_steps
            ]
            new_mus = [
                tp.as_tensor(1 / max(1, _get_value(step) - t0), device=device)
                for step in grouped_state_steps
            ]
            tp._foreach_copy_(grouped_etas, new_etas)
            tp._foreach_copy_(grouped_mus, new_mus)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_asgd)
def asgd(
    params,
    grads,
    axs,
    mus,
    etas,
    state_steps,
    foreach=None,
    maximize=False,
    differentiable=False,
    capturable=False,
    has_complex=False,
    *,
    lambd,
    lr,
    t0,
    alpha,
    weight_decay,
):
    if not tp.compiler.is_compiling() and not all(
        isinstance(value, tp.Tensor) for value in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of "
            "singleton tensors"
        )
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    func = _multi_tensor_asgd if foreach else _single_tensor_asgd
    func(
        params,
        grads,
        axs,
        mus,
        etas,
        state_steps,
        lambd=lambd,
        lr=lr,
        t0=t0,
        alpha=alpha,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
