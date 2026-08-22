import tensorplay as tp

from ._utils import zeros_like
from .optimizer import (
    Optimizer,
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
)

__all__ = ["NAdam", "nadam"]


class NAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        momentum_decay=4e-3,
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
        if not 0.0 <= momentum_decay:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum_decay": momentum_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
            "maximize": maximize,
            "foreach": foreach,
            "capturable": capturable,
            "differentiable": differentiable,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0:
                    if not tp.is_tensor(p_state["step"]):
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
                    if not tp.is_tensor(p_state["mu_product"]):
                        mu_prod_val = p_state["mu_product"]
                        p_state["mu_product"] = (
                            tp.tensor(
                                mu_prod_val,
                                dtype=_get_scalar_dtype(),
                                device=p.device,
                            )
                            if group["capturable"]
                            else tp.tensor(mu_prod_val, dtype=_get_scalar_dtype())
                        )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= tp.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = (
                        tp.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else tp.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["mu_product"] = (
                        tp.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else tp.tensor(1.0, dtype=_get_scalar_dtype())
                    )
                    state["exp_avg"] = zeros_like(p)
                    state["exp_avg_sq"] = zeros_like(p)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                mu_products.append(state["mu_product"])
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
            mu_products = []
            state_steps = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
            )

            nadam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum_decay=group["momentum_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )
        return loss


def _single_tensor_nadam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    mu_products,
    state_steps,
    *,
    beta1,
    beta2,
    lr,
    weight_decay,
    momentum_decay,
    eps,
    decoupled_weight_decay,
    maximize,
    capturable,
    differentiable,
    has_complex,
):
    lr = _to_scalar(lr)

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        mu_product = mu_products[i]
        step_t = state_steps[i]

        if tp.is_complex(param):
            param = tp.view_as_real(param)
            grad = tp.view_as_real(grad)
            exp_avg = tp.view_as_real(exp_avg)
            exp_avg_sq = tp.view_as_real(exp_avg_sq)

        if not tp.compiler.is_compiling() and capturable:
            supported = _get_capturable_supported_devices()
            if not (
                param.device.type == mu_product.device.type == step_t.device.type
                and param.device.type in supported
            ):
                raise AssertionError(
                    "If capturable=True, params, mu_products and state_steps "
                    f"must be on supported devices: {supported}."
                )

        step_t += 1
        step = step_t if capturable else _get_value(step_t)
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)

        mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
        mu_next = beta1 * (
            1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay))
        )

        mu_product.mul_(mu)
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.div(bias_correction2).sqrt()

        if differentiable or capturable:
            denom = denom.add(eps)
            mu_product_next = mu_product * mu_next
            grad = grad * (-lr * (1.0 - mu) / (1.0 - mu_product))
            exp_avg = exp_avg * (-lr * mu_next / (1.0 - mu_product_next))
            param.addcdiv_(grad, denom)
            param.addcdiv_(exp_avg, denom)
        else:
            mu_product_next = _get_value(mu_product) * mu_next
            denom.add_(eps)
            param.addcdiv_(
                grad,
                denom,
                value=(-lr * (1.0 - mu) / (1.0 - _get_value(mu_product))),
            )
            param.addcdiv_(
                exp_avg,
                denom,
                value=(-lr * mu_next / (1.0 - mu_product_next)),
            )


def _multi_tensor_nadam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    mu_products,
    state_steps,
    *,
    beta1,
    beta2,
    lr,
    weight_decay,
    momentum_decay,
    eps,
    decoupled_weight_decay,
    maximize,
    capturable,
    differentiable,
    has_complex,
):
    if not params:
        return
    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    if not tp.compiler.is_compiling() and capturable:
        supported = _get_capturable_supported_devices(supports_xla=False)
        if not all(
            p.device.type == mu_product.device.type == step.device.type
            and p.device.type in supported
            for p, mu_product, step in zip(
                params, mu_products, state_steps, strict=True
            )
        ):
            raise AssertionError(
                "If capturable=True, params, mu_products, and state_steps "
                f"must be on supported devices: {supported}."
            )

    lr = _to_scalar(lr)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps]
    )
    for (
        grouped_params,
        grouped_grads,
        grouped_exp_avgs,
        grouped_exp_avg_sqs,
        grouped_mu_products,
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
        exp_avg_sq_sqrt = tp._foreach_sqrt(grouped_exp_avg_sqs)

        if capturable:
            exponent = tp._foreach_mul(grouped_state_steps, momentum_decay)
            mus = tp._foreach_pow(0.96, exponent)
            tp._foreach_mul_(mus, -0.5)
            tp._foreach_add_(mus, 1.0)
            tp._foreach_mul_(mus, beta1)

            tp._foreach_add_(exponent, momentum_decay)
            mu_nexts = tp._foreach_pow(0.96, exponent)
            tp._foreach_mul_(mu_nexts, -0.5)
            tp._foreach_add_(mu_nexts, 1.0)
            tp._foreach_mul_(mu_nexts, beta1)
            del exponent

            bias_correction_sqrt = tp._foreach_pow(beta2, grouped_state_steps)
            tp._foreach_sub_(bias_correction_sqrt, 1.0)
            tp._foreach_neg_(bias_correction_sqrt)
            tp._foreach_sqrt_(bias_correction_sqrt)
        else:
            bias_correction_sqrt = [
                (1 - beta2 ** _get_value(step)) ** 0.5
                for step in grouped_state_steps
            ]
            mus = [
                beta1 * (1.0 - 0.5 * (0.96 ** (_get_value(step) * momentum_decay)))
                for step in grouped_state_steps
            ]
            mu_nexts = [
                beta1
                * (1.0 - 0.5 * (0.96 ** ((_get_value(step) + 1) * momentum_decay)))
                for step in grouped_state_steps
            ]

        tp._foreach_mul_(grouped_mu_products, mus)
        tp._foreach_div_(exp_avg_sq_sqrt, bias_correction_sqrt)
        tp._foreach_add_(exp_avg_sq_sqrt, eps)
        del bias_correction_sqrt

        if capturable:
            tp._foreach_sub_(mus, 1.0)
            tp._foreach_mul_(mus, lr)
            denom = tp._foreach_sub(grouped_mu_products, 1.0)
            tp._foreach_neg_(denom)
            tp._foreach_div_(mus, denom)
            step_size_grads = mus
            del denom

            denom = tp._foreach_mul(grouped_mu_products, mu_nexts)
            tp._foreach_mul_(mu_nexts, lr)
            tp._foreach_sub_(denom, 1.0)
            tp._foreach_div_(mu_nexts, denom)
            step_size_expavg = mu_nexts
            del denom

            numerator = tp._foreach_mul(step_size_grads, grouped_grads)
            tp._foreach_addcmul_(
                numerator, step_size_expavg, grouped_exp_avgs
            )
            tp._foreach_addcdiv_(grouped_params, numerator, exp_avg_sq_sqrt)
        else:
            step_size_grads = _stack_if_compiling(
                [
                    (_get_value(lr) * (1.0 - mu)
                     / (1.0 - _get_value(mu_product))) * -1
                    for mu_product, mu in zip(
                        grouped_mu_products, mus, strict=True
                    )
                ]
            )
            step_size_expavg = _stack_if_compiling(
                [
                    (_get_value(lr) * mu_next
                     / (1.0 - _get_value(mu_product) * mu_next)) * -1
                    for mu_product, mu_next in zip(
                        grouped_mu_products, mu_nexts, strict=True
                    )
                ]
            )
            tp._foreach_addcdiv_(
                grouped_params,
                grouped_grads,
                exp_avg_sq_sqrt,
                step_size_grads,
            )
            tp._foreach_addcdiv_(
                grouped_params,
                grouped_exp_avgs,
                exp_avg_sq_sqrt,
                step_size_expavg,
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_nadam)
def nadam(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    mu_products,
    state_steps,
    decoupled_weight_decay=False,
    foreach=None,
    capturable=False,
    differentiable=False,
    has_complex=False,
    maximize=False,
    *,
    beta1,
    beta2,
    lr,
    weight_decay,
    momentum_decay,
    eps,
):
    if not all(
        isinstance(value, tp.Tensor) for value in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of "
            "singleton tensors"
        )
    if not all(
        isinstance(value, tp.Tensor) for value in mu_products
    ):
        raise RuntimeError(
            "API has changed, `mu_products` argument must contain a list of "
            "singleton tensors"
        )
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    func = _multi_tensor_nadam if foreach else _single_tensor_nadam
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        momentum_decay=momentum_decay,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
        eps=eps,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
    )
