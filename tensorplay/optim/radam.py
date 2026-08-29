import tensorplay as tp

from ._utils import scalar_value, zeros_like
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

__all__ = ["RAdam", "radam"]


class RAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        decoupled_weight_decay=False,
        *,
        foreach=None,
        maximize=False,
        capturable=False,
        differentiable=False,
    ):
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "decoupled_weight_decay": decoupled_weight_decay,
            "differentiable": differentiable,
        }
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
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not tp.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        tp.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(),
                            device=p.device,
                        )
                        if group["capturable"]
                        else tp.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= tp.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        tp.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else tp.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["exp_avg"] = zeros_like(p)
                    state["exp_avg_sq"] = zeros_like(p)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
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
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
            )

            radam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                has_complex=has_complex,
            )
        return loss


def _single_tensor_radam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    state_steps,
    *,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    decoupled_weight_decay,
    differentiable,
    maximize,
    capturable,
    has_complex,
):
    lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if not tp.compiler.is_compiling() and capturable:
            supported = _get_capturable_supported_devices()
            if not (
                param.device.type == step_t.device.type
                and param.device.type in supported
            ):
                raise AssertionError(
                    "If capturable=True, params and state_steps must be on "
                    f"supported devices: {supported}."
                )

        if tp.is_complex(param):
            param = tp.view_as_real(param)
            grad = tp.view_as_real(grad)
            exp_avg = tp.view_as_real(exp_avg)
            exp_avg_sq = tp.view_as_real(exp_avg_sq)

        step_t += 1
        step = step_t if capturable else _get_value(step_t)

        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        if differentiable:
            # mul/add retains the source edge for higher-order grads.
            exp_avg.mul_(beta1).add_(grad * (1 - beta1))
        else:
            exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        bias_corrected_exp_avg = exp_avg / bias_correction1
        rho_inf = 2 / (1 - beta2) - 1
        rho_t = rho_inf - 2 * step * (beta2 ** step) / bias_correction2

        def _compute_rect():
            return (
                (rho_t - 4)
                * (rho_t - 2)
                * rho_inf
                / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
            ) ** 0.5

        def _compute_adaptive_lr():
            exp_avg_sq_sqrt = exp_avg_sq.sqrt()
            if differentiable:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add(eps)
            else:
                exp_avg_sq_sqrt = exp_avg_sq_sqrt.add_(eps)
            return bias_correction2 ** 0.5 / exp_avg_sq_sqrt

        if capturable:
            update = tp.where(
                rho_t > 5.0, _compute_rect() * _compute_adaptive_lr(), 1.0
            )
            param.add_(bias_corrected_exp_avg * lr * update, alpha=-1.0)
        else:
            if rho_t > 5.0:
                param.add_(
                    bias_corrected_exp_avg
                    * lr
                    * _compute_adaptive_lr()
                    * _compute_rect(),
                    alpha=-1.0,
                )
            else:
                param.add_(bias_corrected_exp_avg * lr, alpha=-1.0)


def _multi_tensor_radam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    state_steps,
    *,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
    decoupled_weight_decay,
    differentiable,
    maximize,
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
            p.device.type == step.device.type and p.device.type in supported
            for p, step in zip(params, state_steps, strict=True)
        ):
            raise AssertionError(
                "If capturable=True, params and state_steps must be on "
                f"supported devices: {supported}."
            )

    lr = _to_scalar(lr)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]
    )
    for (
        grouped_params,
        grouped_grads,
        grouped_exp_avgs,
        grouped_exp_avg_sqs,
        grouped_state_steps,
    ), _ in grouped_tensors.values():
        if not grouped_params:
            continue

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

        if has_complex:
            _view_as_real(
                grouped_params,
                grouped_grads,
                grouped_exp_avgs,
                grouped_exp_avg_sqs,
            )
        if maximize:
            grouped_grads = tp._foreach_neg(grouped_grads)

        rho_inf = 2 / (1 - beta2) - 1
        if capturable:
            bias_correction1 = tp._foreach_pow(beta2, grouped_state_steps)
            tp._foreach_neg_(bias_correction1)
            tp._foreach_add_(bias_correction1, 1)
            bias_correction2 = tp._foreach_pow(beta2, grouped_state_steps)
            tp._foreach_mul_(bias_correction2, grouped_state_steps)
            tp._foreach_mul_(bias_correction2, 2)
            tp._foreach_div_(bias_correction2, bias_correction1)
            tp._foreach_neg_(bias_correction2)
            tp._foreach_add_(bias_correction2, rho_inf)
            rho_t_list = bias_correction2
        else:
            # One host transfer for all step counters (avoids per-tensor
            # CUDA synchronizations).
            if grouped_state_steps and grouped_state_steps[0].is_cuda:
                steps_host = tp.stack(grouped_state_steps).tolist()
            else:
                steps_host = [_get_value(step) for step in grouped_state_steps]
            rho_t_list = [
                rho_inf
                - 2 * float(step) * (beta2 ** float(step))
                / (1 - beta2 ** float(step))
                for step in steps_host
            ]

        if weight_decay != 0:
            if decoupled_weight_decay:
                tp._foreach_mul_(grouped_params, 1 - lr * weight_decay)
            elif maximize:
                tp._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = tp._foreach_add(
                    grouped_grads, grouped_params, alpha=weight_decay
                )

        tp._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)
        tp._foreach_mul_(grouped_exp_avg_sqs, beta2)
        tp._foreach_addcmul_(
            grouped_exp_avg_sqs, grouped_grads, grouped_grads, 1 - beta2
        )
        del grouped_grads

        if capturable:
            num = tp._foreach_sub(rho_t_list, 4)
            sub2 = tp._foreach_sub(rho_t_list, 2)
            tp._foreach_mul_(num, sub2)
            del sub2
            tp._foreach_mul_(num, rho_inf)
            rho_inf = (rho_inf - 4) * (rho_inf - 2)
            denom = tp._foreach_mul(rho_t_list, rho_inf)
            tp._foreach_div_(num, denom)
            del denom
            tp._foreach_sqrt_(num)
            rect = [
                tp.where(rho_t > 5.0, value, 0.0)
                for value, rho_t in zip(num, rho_t_list, strict=True)
            ]
            del num
            del rho_t_list
            unrect_step_size = [
                tp.where(value > 0, 0.0, 1.0) for value in rect
            ]
            tp._foreach_mul_(unrect_step_size, lr)

            bias_correction1 = tp._foreach_pow(beta1, grouped_state_steps)
            tp._foreach_neg_(bias_correction1)
            tp._foreach_add_(bias_correction1, 1)
            tp._foreach_div_(unrect_step_size, bias_correction1)
            tp._foreach_neg_(unrect_step_size)

            bias_correction2 = tp._foreach_pow(beta2, grouped_state_steps)
            tp._foreach_neg_(bias_correction2)
            tp._foreach_add_(bias_correction2, 1)
            tp._foreach_sqrt_(bias_correction2)
            tp._foreach_mul_(bias_correction2, lr)
            tp._foreach_mul_(bias_correction2, rect)
            del rect
            tp._foreach_neg_(bias_correction2)
            tp._foreach_div_(bias_correction2, bias_correction1)
            del bias_correction1
        else:
            rect = [
                ((rho_t - 4) * (rho_t - 2) * rho_inf
                 / ((rho_inf - 4) * (rho_inf - 2) * rho_t)) ** 0.5
                if rho_t > 5 else 0
                for rho_t in rho_t_list
            ]
            unrectified = [0 if value > 0 else 1.0 for value in rect]
            bias_correction1 = [
                1 - beta1 ** _get_value(step)
                for step in grouped_state_steps
            ]
            unrect_step_size = [
                (lr * value / correction) * -1
                for value, correction in zip(
                    unrectified, bias_correction1, strict=True
                )
            ]
            bias_correction2 = [
                ((1 - beta2 ** _get_value(step)) ** 0.5)
                * (lr * value / correction)
                * -1
                for step, value, correction in zip(
                    grouped_state_steps, rect, bias_correction1, strict=True
                )
            ]

        buffer = tp._foreach_sqrt(grouped_exp_avg_sqs)
        tp._foreach_add_(buffer, eps)
        tp._foreach_div_(buffer, bias_correction2)
        tp._foreach_reciprocal_(buffer)
        tp._foreach_add_(buffer, unrect_step_size)
        tp._foreach_addcmul_(grouped_params, grouped_exp_avgs, buffer)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_radam)
def radam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    state_steps,
    decoupled_weight_decay=False,
    foreach=None,
    differentiable=False,
    capturable=False,
    has_complex=False,
    maximize=False,
    *,
    beta1,
    beta2,
    lr,
    weight_decay,
    eps,
):
    if not all(isinstance(value, tp.Tensor) for value in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of "
            "singleton tensors"
        )

    native_device = params[0].device.type if params else None
    native = (
        not differentiable
        and not capturable
        and not has_complex
        and native_device in ("cpu", "cuda")
        and bool(params)
        # CPU and CUDA both round reduced floating point intermediates at the
        and params[0].dtype in (
            tp.float16, tp.bfloat16, tp.float32, tp.float64
        )
        and all(
            p.device.type == native_device
            and p.is_contiguous()
            and p.is_floating_point()
            and p.dtype == params[0].dtype
            for p in params
        )
        and all(
            g.device.type == native_device
            and g.is_contiguous()
            and g.dtype == params[0].dtype
            for g in grads
        )
        and all(
            step.device.type == "cpu"
            and step.is_contiguous()
            and step.numel() == 1
            and step.dtype in (tp.float32, tp.float64)
            for step in state_steps
        )
    )
    if native:
        tp._fused_radam_(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            lr=scalar_value(lr, "lr"),
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
        )
        return

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    func = _multi_tensor_radam if foreach else _single_tensor_radam
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
