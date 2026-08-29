import tensorplay as tp

from ._utils import (
    scalar_value,
    validate_nonnegative,
    validate_unit_interval,
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


__all__ = ["Adamax", "adamax"]


class Adamax(Optimizer):

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

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= p.is_complex()
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adamax does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = (
                    tp.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else tp.tensor(0.0, dtype=_get_scalar_dtype())
                )
                state["exp_avg"] = zeros_like(p)
                state["exp_inf"] = zeros_like(p)

            exp_avgs.append(state["exp_avg"])
            exp_infs.append(state["exp_inf"])
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
            exp_avgs = []
            exp_infs = []
            state_steps = []
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            foreach = group["foreach"]
            maximize = group["maximize"]
            differentiable = group["differentiable"]
            capturable = group["capturable"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
            )
            adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=eps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                capturable=capturable,
                has_complex=has_complex,
            )
        return loss


def _single_tensor_adamax(
    params, grads, exp_avgs, exp_infs, state_steps, *, eps, beta1, beta2,
    lr, weight_decay, maximize, differentiable, capturable, has_complex,
):
    lr = _to_scalar(lr)
    for index, param in enumerate(params):
        grad = grads[index] if not maximize else -grads[index]
        exp_avg = exp_avgs[index]
        exp_inf = exp_infs[index]
        step_t = state_steps[index]
        if capturable and not tp.compiler.is_compiling():
            supported = _get_capturable_supported_devices()
            if not (
                param.device.type == step_t.device.type
                and param.device.type in supported
            ):
                raise AssertionError(
                    "If capturable=True, params and state_steps must be on "
                    f"supported devices: {supported}."
                )
        step_t.add_(1)
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        if param.is_complex():
            param = tp.view_as_real(param)
            grad = tp.view_as_real(grad)
            exp_avg = tp.view_as_real(exp_avg)
            exp_inf = tp.view_as_real(exp_inf)
        if differentiable:
            # Keep the source edge for higher-order gradients.
            exp_avg.mul_(beta1).add_(grad * (1 - beta1))
        else:
            exp_avg.lerp_(grad, 1 - beta1)
        if differentiable:
            candidate = tp.maximum(exp_inf * beta2, grad.abs() + eps)
            exp_inf.copy_(candidate)
        else:
            exp_inf.copy_(tp.maximum(
                exp_inf.mul_(beta2), grad.abs().add_(eps)
            ))
        if capturable:
            neg_bias_correction = beta1 ** step_t - 1
            neg_bias_correction.div_(lr)
            param.addcdiv_(
                exp_avg, exp_inf * neg_bias_correction
            )
        else:
            bias_correction = 1 - beta1 ** _get_value(step_t)
            param.addcdiv_(
                exp_avg, exp_inf, value=-lr / bias_correction
            )


def _multi_tensor_adamax(
    params, grads, exp_avgs, exp_infs, state_steps, *, eps, beta1, beta2,
    lr, weight_decay, maximize, differentiable, capturable, has_complex,
):
    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")
    if not params:
        return
    if capturable and not tp.compiler.is_compiling():
        supported = _get_capturable_supported_devices(supports_xla=False)
        if not all(
            p.device.type == step.device.type and p.device.type in supported
            for p, step in zip(params, state_steps)
        ):
            raise AssertionError(
                "If capturable=True, params and state_steps must be on "
                f"supported devices: {supported}."
            )
    lr = _to_scalar(lr)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_infs, state_steps]
    )
    for (
        grouped_params, grouped_grads, grouped_exp_avgs,
        grouped_exp_infs, grouped_state_steps,
    ), _ in grouped_tensors.values():
        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads,
                grouped_exp_avgs, grouped_exp_infs,
            )
        if maximize:
            grouped_grads = tp._foreach_neg(grouped_grads)
        if (not tp.compiler.is_compiling() and
                grouped_state_steps[0].device.type == "cpu"):
            tp._foreach_add_(
                grouped_state_steps,
                tp.tensor(1.0, dtype=grouped_state_steps[0].dtype,
                          device=tp.device("cpu")),
                alpha=1.0,
            )
        else:
            tp._foreach_add_(grouped_state_steps, 1)
        if weight_decay != 0:
            if maximize:
                tp._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = tp._foreach_add(
                    grouped_grads, grouped_params, alpha=weight_decay
                )
        tp._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)
        tp._foreach_mul_(grouped_exp_infs, beta2)
        if not maximize and weight_decay == 0:
            grouped_grads = tp._foreach_abs(grouped_grads)
        else:
            tp._foreach_abs_(grouped_grads)
        tp._foreach_add_(grouped_grads, eps)
        tp._foreach_maximum_(grouped_exp_infs, grouped_grads)
        if capturable:
            bias_corrections = tp._foreach_pow(beta1, grouped_state_steps)
            tp._foreach_sub_(bias_corrections, 1)
            tp._foreach_div_(bias_corrections, lr)
            denom = tp._foreach_mul(grouped_exp_infs, bias_corrections)
            tp._foreach_addcdiv_(grouped_params, grouped_exp_avgs, denom)
        else:
            # One host transfer for all step counters (CUDA sync otherwise).
            if grouped_state_steps and grouped_state_steps[0].is_cuda:
                steps_host = tp.stack(grouped_state_steps).tolist()
            else:
                steps_host = [_get_value(step) for step in grouped_state_steps]
            bias_corrections = [
                1 - beta1 ** float(step)
                for step in steps_host
            ]
            step_size = [
                -_get_value(lr) / correction
                for correction in bias_corrections
            ]
            tp._foreach_addcdiv_(
                grouped_params, grouped_exp_avgs, grouped_exp_infs, step_size
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamax)
def adamax(
    params, grads, exp_avgs, exp_infs, state_steps, foreach=None,
    maximize=False, differentiable=False, capturable=False, has_complex=False,
    *, eps, beta1, beta2, lr, weight_decay,
):
    if not tp.compiler.is_compiling() and not all(
        isinstance(value, tp.Tensor) for value in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of "
            "singleton tensors"
        )

    # Fused CPU/CUDA Adamax validates all pairs and states in the native
    # dispatcher.  Avoid repeating the full layout scan in Python for the
    # steady-state optimizer loop; invalid layouts take the cold fallback.
    native_candidate = (
        not differentiable
        and not capturable
        and not has_complex
        and bool(params)
        and params[0].device.type in ("cpu", "cuda")
        and params[0].is_floating_point()
        and params[0].dtype in (
            tp.float16, tp.bfloat16, tp.float32, tp.float64
        )
    )
    if native_candidate:
        try:
            tp._fused_adamax_(
                params,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                lr=scalar_value(lr, "lr"),
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                weight_decay=weight_decay,
                maximize=maximize,
            )
            return
        except NotImplementedError:
            pass

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    if foreach:
        _multi_tensor_adamax(
            params, grads, exp_avgs, exp_infs, state_steps, eps=eps,
            beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay,
            maximize=maximize, differentiable=differentiable,
            capturable=capturable, has_complex=has_complex,
        )
    else:
        _single_tensor_adamax(
            params, grads, exp_avgs, exp_infs, state_steps, eps=eps,
            beta1=beta1, beta2=beta2, lr=lr, weight_decay=weight_decay,
            maximize=maximize, differentiable=differentiable,
            capturable=capturable, has_complex=has_complex,
        )
