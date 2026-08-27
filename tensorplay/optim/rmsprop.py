import tensorplay as tp

from ._utils import (
    scalar_value,
    validate_nonnegative,
    zeros_like,
)
from .optimizer import (
    Optimizer,
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
)


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

    def _init_group(
        self, group, params_with_grad, grads, square_avgs,
        momentum_buffer_list, grad_avgs, state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= p.is_complex()
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("RMSprop does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            if not state:
                state["step"] = tp.zeros(
                    (), dtype=_get_scalar_dtype(),
                    device=p.device if group["capturable"] else tp.device("cpu"),
                )
                state["square_avg"] = zeros_like(p)
                if group["momentum"] > 0:
                    state["momentum_buffer"] = zeros_like(p)
                if group["centered"]:
                    state["grad_avg"] = zeros_like(p)

            square_avgs.append(state["square_avg"])
            state_steps.append(state["step"])
            if group["momentum"] > 0:
                momentum_buffer_list.append(state["momentum_buffer"])
            if group["centered"]:
                grad_avgs.append(state["grad_avg"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            grad_avgs = []
            momentum_buffer_list = []
            state_steps = []
            has_complex = self._init_group(
                group, params_with_grad, grads, square_avgs,
                momentum_buffer_list, grad_avgs, state_steps,
            )
            rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=group["lr"],
                alpha=group["alpha"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                centered=group["centered"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )
        return loss


def _single_tensor_rmsprop(
    params, grads, square_avgs, grad_avgs, momentum_buffer_list, state_steps,
    *, lr, alpha, eps, weight_decay, momentum, centered, maximize,
    differentiable, capturable, has_complex,
):
    if not tp.compiler.is_compiling():
        lr = _to_scalar(lr)

    for i, param in enumerate(params):
        step = state_steps[i]
        if not tp.compiler.is_compiling() and capturable:
            supported = _get_capturable_supported_devices()
            if not (
                param.device.type == step.device.type
                and param.device.type in supported
            ):
                raise AssertionError(
                    "If capturable=True, params and state_steps must be on "
                    f"supported devices: {supported}."
                )

        grad = grads[i] if not maximize else -grads[i]
        step.add_(1)
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        is_complex_param = param.is_complex()
        if is_complex_param:
            param = tp.view_as_real(param)
            grad = tp.view_as_real(grad)
            square_avg = tp.view_as_real(square_avgs[i])
        else:
            square_avg = square_avgs[i]

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)
        if centered:
            grad_avg = grad_avgs[i]
            if is_complex_param:
                grad_avg = tp.view_as_real(grad_avg)
            if differentiable:
                # Preserve the source edge needed by differentiable mode.
                grad_avg.mul_(alpha).add_(grad * (1 - alpha))
            else:
                grad_avg.lerp_(grad, 1 - alpha)
            centered_var = square_avg.addcmul(
                grad_avg, grad_avg, value=-1
            )
            avg = centered_var.sqrt() if differentiable else centered_var.sqrt_()
        else:
            avg = square_avg.sqrt() if differentiable else square_avg.sqrt_()
        avg = avg.add(eps) if differentiable else avg.add_(eps)

        if momentum > 0:
            buf = momentum_buffer_list[i]
            if is_complex_param:
                buf = tp.view_as_real(buf)
            buf.mul_(momentum).addcdiv_(grad, avg)
            if isinstance(lr, tp.Tensor):
                param.add_(buf * (-lr))
            else:
                param.add_(buf, alpha=-lr)
        elif isinstance(lr, tp.Tensor):
            param.add_(grad / avg * (-lr))
        else:
            param.addcdiv_(grad, avg, value=-lr)


def _multi_tensor_rmsprop(
    params, grads, square_avgs, grad_avgs, momentum_buffer_list, state_steps,
    *, lr, alpha, eps, weight_decay, momentum, centered, maximize,
    differentiable, capturable, has_complex,
):
    if len(params) == 0:
        return
    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")

    if not tp.compiler.is_compiling() and capturable:
        supported = _get_capturable_supported_devices()
        if not all(
            p.device.type == step.device.type
            and p.device.type in supported
            for p, step in zip(params, state_steps)
        ):
            raise AssertionError(
                "If capturable=True, params and state_steps must be on "
                f"supported devices: {supported}."
            )

    if not tp.compiler.is_compiling():
        lr = _to_scalar(lr)
    grouped = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, square_avgs, grad_avgs, momentum_buffer_list, state_steps]
    )
    for (
        grouped_params, grouped_grads, grouped_square_avgs, grouped_grad_avgs,
        grouped_momentum_buffer_list, grouped_state_steps,
    ), _ in grouped.values():
        if has_complex:
            states_and_grads = [grouped_grads, grouped_square_avgs]
            if momentum > 0:
                states_and_grads.append(grouped_momentum_buffer_list)
            if centered:
                states_and_grads.append(grouped_grad_avgs)
            _view_as_real(grouped_params, *states_and_grads)

        if maximize:
            grouped_grads = tp._foreach_neg(grouped_grads)

        if (
            not tp.compiler.is_compiling()
            and grouped_state_steps[0].device.type == "cpu"
        ):
            tp._foreach_add_(
                grouped_state_steps,
                tp.tensor(
                    1.0, dtype=grouped_state_steps[0].dtype,
                    device=tp.device("cpu"),
                ),
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

        tp._foreach_mul_(grouped_square_avgs, alpha)
        tp._foreach_addcmul_(
            grouped_square_avgs, grouped_grads, grouped_grads, value=1 - alpha
        )
        if centered:
            tp._foreach_lerp_(grouped_grad_avgs, grouped_grads, 1 - alpha)
            avg = tp._foreach_addcmul(
                grouped_square_avgs, grouped_grad_avgs, grouped_grad_avgs,
                value=-1,
            )
            tp._foreach_sqrt_(avg)
            tp._foreach_add_(avg, eps)
        else:
            avg = tp._foreach_sqrt(grouped_square_avgs)
            tp._foreach_add_(avg, eps)

        if momentum > 0:
            tp._foreach_mul_(grouped_momentum_buffer_list, momentum)
            tp._foreach_addcdiv_(grouped_momentum_buffer_list, grouped_grads, avg)
            if capturable and isinstance(lr, tp.Tensor):
                momentum_lr = tp._foreach_mul(grouped_momentum_buffer_list, -lr)
                tp._foreach_add_(grouped_params, momentum_lr)
            else:
                lr_value = scalar_value(lr, "lr")
                tp._foreach_add_(
                    grouped_params, grouped_momentum_buffer_list, alpha=-lr_value
                )
        elif capturable and isinstance(lr, tp.Tensor):
            tp._foreach_div_(avg, -lr)
            tp._foreach_addcdiv_(grouped_params, grouped_grads, avg)
        else:
            lr_value = scalar_value(lr, "lr")
            tp._foreach_addcdiv_(
                grouped_params, grouped_grads, avg, value=-lr_value
            )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rmsprop)
def rmsprop(
    params, grads, square_avgs, grad_avgs, momentum_buffer_list, state_steps,
    foreach=None, maximize=False, differentiable=False, capturable=False,
    has_complex=False, *, lr, alpha, eps, weight_decay, momentum, centered,
):
    if not tp.compiler.is_compiling() and not all(
        isinstance(value, tp.Tensor) for value in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of "
            "singleton tensors"
        )

    # The native CPU/CUDA kernels validate every tensor pair and state list
    # in C++.  Probe the cheap first tensor metadata here and let an
    # unsupported layout fall through to Torch's foreach implementation.  A
    # Python all(...) over every parameter was measurably more expensive than
    # the optimizer kernel for the many-small-tensor case.
    native_candidate = (
        not differentiable
        and not capturable
        and not has_complex
        and bool(params)
        and params[0].device.type in ("cpu", "cuda")
        and params[0].is_floating_point()
    )
    if native_candidate:
        try:
            tp._fused_rmsprop_(
                params,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=scalar_value(lr, "lr"),
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
                maximize=maximize,
            )
            return
        except NotImplementedError:
            # The C++ validator uses NotImplementedError for layouts and
            # dtypes that the fused path cannot consume.  Those are valid
            # foreach inputs, so retain Torch's fallback semantics.
            pass

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    if foreach:
        _multi_tensor_rmsprop(
            params, grads, square_avgs, grad_avgs, momentum_buffer_list,
            state_steps, lr=lr, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered,
            maximize=maximize, differentiable=differentiable,
            capturable=capturable, has_complex=has_complex,
        )
    else:
        _single_tensor_rmsprop(
            params, grads, square_avgs, grad_avgs, momentum_buffer_list,
            state_steps, lr=lr, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered,
            maximize=maximize, differentiable=differentiable,
            capturable=capturable, has_complex=has_complex,
        )
