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
from .optimizer import (
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _to_scalar,
    _view_as_real,
)


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

    def _init_group(
        self, group, params_with_grad, grads, square_avgs, acc_deltas, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= p.is_complex()
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("Adadelta does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = (
                    tp.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else tp.zeros((), dtype=_get_scalar_dtype())
                )
                state["square_avg"] = zeros_like(p)
                state["acc_delta"] = zeros_like(p)

            square_avgs.append(state["square_avg"])
            acc_deltas.append(state["acc_delta"])
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
            square_avgs = []
            acc_deltas = []
            state_steps = []
            lr = group["lr"]
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            foreach = group["foreach"]
            maximize = group["maximize"]
            differentiable = group["differentiable"]
            capturable = group["capturable"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
            )
            adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
                capturable=capturable,
                foreach=foreach,
                differentiable=differentiable,
                has_complex=has_complex,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                maximize=maximize,
            )
        return loss


def _single_tensor_adadelta(
    params, grads, square_avgs, acc_deltas, state_steps, *, lr, rho, eps,
    weight_decay, maximize, differentiable, capturable, has_complex,
):
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
    for param, grad, square_avg, acc_delta, step in zip(
        params, grads, square_avgs, acc_deltas, state_steps
    ):
        step.add_(1)
        grad = grad if not maximize else -grad
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)
        is_complex = param.is_complex()
        if is_complex:
            grad = tp.view_as_real(grad)
            square_avg = tp.view_as_real(square_avg)
            acc_delta = tp.view_as_real(acc_delta)
        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        if differentiable:
            # Do not mutate the temporary saved by add() in-place before its
            # backward pass; Torch's differentiable path treats these as
            # out-of-place temporaries.
            std = square_avg.add(eps).sqrt()
            delta = acc_delta.add(eps).sqrt()
        else:
            std = square_avg.add(eps).sqrt_()
            delta = acc_delta.add(eps).sqrt_()
        if differentiable:
            delta = delta.div(std).mul(grad)
        else:
            delta.div_(std).mul_(grad)
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)
        if is_complex:
            delta = tp.view_as_complex(delta)
        if isinstance(lr, tp.Tensor):
            param.add_(delta * (-lr))
        else:
            param.add_(delta, alpha=-lr)


def _multi_tensor_adadelta(
    params, grads, square_avgs, acc_deltas, state_steps, *, lr, rho, eps,
    weight_decay, maximize, differentiable, capturable, has_complex,
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
    grouped = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, square_avgs, acc_deltas, state_steps]
    )
    for (
        device_params, device_grads, device_square_avgs,
        device_acc_deltas, device_state_steps,
    ), _ in grouped.values():
        if has_complex:
            _view_as_real(
                device_params, device_grads, device_square_avgs, device_acc_deltas
            )
        if (not tp.compiler.is_compiling() and
                device_state_steps[0].device.type == "cpu"):
            tp._foreach_add_(
                device_state_steps,
                tp.tensor(1.0, dtype=device_state_steps[0].dtype,
                          device=tp.device("cpu")),
                alpha=1.0,
            )
        else:
            tp._foreach_add_(device_state_steps, 1)
        if maximize:
            device_grads = tp._foreach_neg(device_grads)
        if weight_decay != 0:
            if maximize:
                tp._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = tp._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )
        tp._foreach_mul_(device_square_avgs, rho)
        tp._foreach_addcmul_(
            device_square_avgs, device_grads, device_grads, value=1 - rho
        )
        std = tp._foreach_add(device_square_avgs, eps)
        tp._foreach_sqrt_(std)
        deltas = tp._foreach_add(device_acc_deltas, eps)
        tp._foreach_sqrt_(deltas)
        tp._foreach_div_(deltas, std)
        tp._foreach_mul_(deltas, device_grads)
        tp._foreach_mul_(device_acc_deltas, rho)
        tp._foreach_addcmul_(device_acc_deltas, deltas, deltas, value=1 - rho)
        if capturable and isinstance(lr, tp.Tensor):
            tp._foreach_mul_(deltas, -lr)
            tp._foreach_add_(device_params, deltas)
        else:
            tp._foreach_add_(device_params, deltas, alpha=-lr)


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adadelta)
def adadelta(
    params, grads, square_avgs, acc_deltas, state_steps, capturable=False,
    foreach=None, differentiable=False, has_complex=False, *, lr, rho, eps,
    weight_decay, maximize,
):
    if not tp.compiler.is_compiling() and not all(
        isinstance(value, tp.Tensor) for value in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of "
            "singleton tensors"
        )

    native_cpu = (
        not differentiable
        and not capturable
        and not has_complex
        and bool(params)
        and all(
            p.device.type == "cpu"
            and p.is_contiguous()
            and p.is_floating_point()
            and p.dtype == params[0].dtype
            for p in params
        )
        and all(
            g.device.type == "cpu"
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
    if native_cpu:
        tp._fused_adadelta_(
            params,
            grads,
            square_avgs,
            acc_deltas,
            state_steps,
            lr=scalar_value(lr, "lr"),
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        return

    native_cuda = (
        not differentiable
        and not capturable
        and not has_complex
        and bool(params)
        and all(
            p.device.type == "cuda"
            and p.is_contiguous()
            and p.is_floating_point()
            and p.dtype == params[0].dtype
            for p in params
        )
        and all(
            g.device.type == "cuda"
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
    if native_cuda:
        tp._fused_adadelta_(
            params,
            grads,
            square_avgs,
            acc_deltas,
            state_steps,
            lr=scalar_value(lr, "lr"),
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
        )
        return

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    if foreach:
        _multi_tensor_adadelta(
            params, grads, square_avgs, acc_deltas, state_steps, lr=lr,
            rho=rho, eps=eps, weight_decay=weight_decay, maximize=maximize,
            differentiable=differentiable, capturable=capturable,
            has_complex=has_complex,
        )
    else:
        _single_tensor_adadelta(
            params, grads, square_avgs, acc_deltas, state_steps, lr=lr,
            rho=rho, eps=eps, weight_decay=weight_decay, maximize=maximize,
            differentiable=differentiable, capturable=capturable,
            has_complex=has_complex,
        )
