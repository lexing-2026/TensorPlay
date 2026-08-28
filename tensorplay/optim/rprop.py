import tensorplay as tp

from ._utils import (
    full_like,
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
    _use_grad_for_differentiable,
    _view_as_real,
)


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

    def _init_group(
        self, group, params, grads, prevs, step_sizes, state_steps
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= p.is_complex()
            params.append(p)
            grad = p.grad
            if grad.is_sparse:
                raise RuntimeError("Rprop does not support sparse gradients")
            grads.append(grad)

            state = self.state[p]
            if not state:
                state["step"] = tp.zeros(
                    (), dtype=_get_scalar_dtype(),
                    device=p.device if group["capturable"] else tp.device("cpu"),
                )
                state["prev"] = zeros_like(p)
                lr = scalar_value(group["lr"], "lr")
                state["step_size"] = full_like(
                    grad,
                    complex(lr, lr) if p.is_complex() else lr,
                )

            prevs.append(state["prev"])
            step_sizes.append(state["step_size"])
            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = []
            grads = []
            prevs = []
            step_sizes = []
            state_steps = []
            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]
            has_complex = self._init_group(
                group, params, grads, prevs, step_sizes, state_steps
            )
            rprop(
                params,
                grads,
                prevs,
                step_sizes,
                state_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )
        return loss


def _single_tensor_rprop(
    params, grads, prevs, step_sizes, state_steps, *, step_size_min,
    step_size_max, etaminus, etaplus, maximize, capturable, differentiable,
    has_complex,
):
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        prev = prevs[i]
        step_size = step_sizes[i]
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

        step.add_(1)
        if param.is_complex():
            grad = tp.view_as_real(grad)
            prev = tp.view_as_real(prev)
            param = tp.view_as_real(param)
            step_size = tp.view_as_real(step_size)

        sign = grad.mul(prev.clone() if differentiable else prev).sign()
        sign.copy_(tp.where(sign.gt(0), etaplus, sign))
        sign.copy_(tp.where(sign.lt(0), etaminus, sign))
        sign.copy_(tp.where(sign.eq(0), 1, sign))
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)
        grad = grad.clone()
        grad.copy_(tp.where(sign.eq(etaminus), 0, grad))
        param.addcmul_(grad.sign(), step_size, value=-1)
        prev.copy_(grad)


def _multi_tensor_rprop(
    params, grads, prevs, step_sizes, state_steps, *, step_size_min,
    step_size_max, etaminus, etaplus, maximize, capturable, differentiable,
    has_complex,
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

    grouped = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, prevs, step_sizes, state_steps]
    )
    for (
        grouped_params, grouped_grads, grouped_prevs,
        grouped_step_sizes, grouped_state_steps,
    ), _ in grouped.values():
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

        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_prevs,
                grouped_step_sizes,
            )

        signs = tp._foreach_mul(grouped_grads, grouped_prevs)
        if maximize:
            tp._foreach_neg_(signs)
        tp._foreach_copy_(grouped_prevs, grouped_grads)
        if maximize:
            tp._foreach_neg_(grouped_prevs)
        grouped_grads = grouped_prevs

        tp._foreach_sign_(signs)
        for sign in signs:
            sign.copy_(tp.where(sign.gt(0), etaplus, sign))
            sign.copy_(tp.where(sign.lt(0), etaminus, sign))
            sign.copy_(tp.where(sign.eq(0), 1, sign))

        tp._foreach_mul_(grouped_step_sizes, signs)
        for step_size in grouped_step_sizes:
            step_size.clamp_(step_size_min, step_size_max)

        for i, grad in enumerate(grouped_grads):
            grad.copy_(tp.where(signs[i].eq(etaminus), 0, grad))
        grad_signs = [grad.sign() for grad in grouped_grads]
        tp._foreach_addcmul_(
            grouped_params, grad_signs, grouped_step_sizes, value=-1
        )


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rprop)
def rprop(
    params, grads, prevs, step_sizes, state_steps, foreach=None,
    capturable=False, maximize=False, differentiable=False, has_complex=False,
    *, step_size_min, step_size_max, etaminus, etaplus,
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
            and p.dtype in (
                tp.float16, tp.bfloat16, tp.float32, tp.float64
            )
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
        tp._fused_rprop_(
            params,
            grads,
            prevs,
            step_sizes,
            state_steps,
            step_size_min=step_size_min,
            step_size_max=step_size_max,
            etaminus=etaminus,
            etaplus=etaplus,
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
        tp._fused_rprop_(
            params,
            grads,
            prevs,
            step_sizes,
            state_steps,
            step_size_min=step_size_min,
            step_size_max=step_size_max,
            etaminus=etaminus,
            etaplus=etaplus,
            maximize=maximize,
        )
        return

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    if foreach:
        _multi_tensor_rprop(
            params, grads, prevs, step_sizes, state_steps,
            step_size_min=step_size_min, step_size_max=step_size_max,
            etaminus=etaminus, etaplus=etaplus, capturable=capturable,
            maximize=maximize, differentiable=differentiable,
            has_complex=has_complex,
        )
    else:
        _single_tensor_rprop(
            params, grads, prevs, step_sizes, state_steps,
            step_size_min=step_size_min, step_size_max=step_size_max,
            etaminus=etaminus, etaplus=etaplus, capturable=capturable,
            maximize=maximize, differentiable=differentiable,
            has_complex=has_complex,
        )
