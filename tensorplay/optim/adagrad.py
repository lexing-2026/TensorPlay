from typing import cast

import tensorplay as tp
from tensorplay import Tensor

from ._utils import (
    full_like,
    scalar_value,
    validate_nonnegative,
)
from .optimizer import (
    DeviceDict,
    Optimizer,
    ParamsT,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _get_scalar_dtype,
    _get_value,
    _to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
)


__all__ = ["Adagrad", "adagrad"]


class Adagrad(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-2,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
        foreach: bool | None = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
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

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
            self._need_device_dtype_check_for_fused = True
            self._step_supports_amp_scaling = True

        # Torch initializes Adagrad's state eagerly in the constructor.
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["step"] = tp.tensor(
                    0.0,
                    dtype=_get_scalar_dtype(is_fused=group["fused"]),
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
        # define "fused" for the state migration below
        fused = None
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if len(p_state) != 0 and not isinstance(p_state["step"], tp.Tensor):
                    p_state["step"] = tp.tensor(
                        float(p_state["step"]),
                        dtype=_get_scalar_dtype(is_fused=fused),
                        device=p.device if group["fused"] else tp.device("cpu"),
                    )

        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and isinstance(
            state_values[0]["step"], tp.Tensor
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = tp.tensor(
                    float(s["step"]), dtype=_get_scalar_dtype(is_fused=fused)
                )

    def share_memory(self) -> None:
        """Calls tensor.share_memory_() on the state sum tensors."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["sum"].share_memory_()

    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        has_sparse_grad, has_complex = False, False
        for p in group["params"]:
            if p.grad is not None:
                if group["fused"] and getattr(
                    self, "_need_device_dtype_check_for_fused", True
                ):
                    _device_dtype_check_for_fused(p)
                    self._need_device_dtype_check_for_fused = False
                has_sparse_grad |= p.grad.is_sparse
                has_complex |= p.is_complex()
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    if group["fused"]:
                        _device_dtype_check_for_fused(p)

                    state["step"] = (
                        tp.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["fused"]
                        else tp.tensor(0.0, dtype=_get_scalar_dtype())
                    )

                    initial_accumulator_value = self.defaults[
                        "initial_accumulator_value"
                    ]
                    init_value = (
                        complex(initial_accumulator_value, initial_accumulator_value)
                        if p.is_complex()
                        else initial_accumulator_value
                    )
                    state["sum"] = full_like(p, init_value)
                state_sums.append(state["sum"])
                state_steps.append(state["step"])
        return has_sparse_grad, has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            state_sums = []
            state_steps = []
            has_sparse_grad, has_complex = self._init_group(
                group, params_with_grad, grads, state_sums, state_steps
            )
            adagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )
        return loss


def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    return tp.sparse_coo_tensor(grad_indices, values, size)


def _single_tensor_adagrad(
    params, grads, state_sums, state_steps, grad_scale, found_inf, *, lr,
    weight_decay, lr_decay, eps, has_sparse_grad, maximize, differentiable,
    has_complex,
):
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")
    lr = _to_scalar(lr)
    for param, grad, state_sum, step_t in zip(
        params, grads, state_sums, state_steps, strict=True
    ):
        step_t += 1
        step = _get_value(step_t)
        grad = grad if not maximize else -grad
        if weight_decay != 0:
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients"
                )
            grad = grad.add(param, alpha=weight_decay)
        clr = lr / (1 + (step - 1) * lr_decay)
        if grad.is_sparse:
            grad = grad.coalesce()
            indices = grad._indices()
            values = grad._values()
            state_sum.add_(_make_sparse(grad, indices, values.pow(2)))
            std = state_sum.sparse_mask(grad)
            std_values = std._values().sqrt_().add_(eps)
            param.add_(
                _make_sparse(grad, indices, values / std_values), alpha=-clr
            )
        else:
            is_complex = param.is_complex()
            if is_complex:
                grad = tp.view_as_real(grad)
                state_sum = tp.view_as_real(state_sum)
                param = tp.view_as_real(param)
            state_sum.addcmul_(grad, grad, value=1)
            if differentiable:
                std = state_sum.sqrt() + eps
            else:
                std = state_sum.sqrt().add_(eps)
            param.addcdiv_(grad, std, value=-clr)
            if is_complex:
                param = tp.view_as_complex(param)
                state_sum = tp.view_as_complex(state_sum)


def _multi_tensor_adagrad(
    params, grads, state_sums, state_steps, grad_scale, found_inf, *, lr,
    weight_decay, lr_decay, eps, has_sparse_grad, maximize, differentiable,
    has_complex,
):
    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")
    if not params:
        return
    lr = _to_scalar(lr)
    grouped_tensorlists = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]
    )
    for (
        device_params_,
        device_grads_,
        device_state_sums_,
        device_state_steps_,
    ), _ in grouped_tensorlists.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_state_sums = cast(list[Tensor], device_state_sums_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        if device_has_sparse_grad:
            _single_tensor_adagrad(
                device_params,
                device_grads,
                device_state_sums,
                device_state_steps,
                lr=lr,
                weight_decay=weight_decay,
                lr_decay=lr_decay,
                eps=eps,
                has_sparse_grad=True,
                maximize=maximize,
                differentiable=differentiable,
                has_complex=has_complex,
                grad_scale=grad_scale,
                found_inf=found_inf,
            )
            continue
        if has_complex:
            _view_as_real(device_params, device_grads, device_state_sums)
        if maximize:
            device_grads = tp._foreach_neg(device_grads)
        if not tp.compiler.is_compiling() and device_state_steps[0].device.type == "cpu":
            tp._foreach_add_(
                device_state_steps,
                tp.tensor(1.0, device=tp.device("cpu")),
                alpha=1.0,
            )
        else:
            tp._foreach_add_(device_state_steps, 1)
        if weight_decay != 0:
            if maximize:
                tp._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = tp._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )
        minus_clr = [
            -lr / (1 + (_get_value(step) - 1) * lr_decay)
            for step in device_state_steps
        ]
        tp._foreach_addcmul_(
            device_state_sums, device_grads, device_grads, value=1
        )
        std = tp._foreach_sqrt(device_state_sums)
        tp._foreach_add_(std, eps)
        if weight_decay != 0 or maximize:
            tp._foreach_mul_(device_grads, minus_clr)
            numerator = device_grads
        else:
            numerator = tp._foreach_mul(device_grads, minus_clr)
        tp._foreach_addcdiv_(device_params, numerator, std)


def _fused_adagrad(
    params, grads, state_sums, state_steps, grad_scale, found_inf, *, lr,
    weight_decay, lr_decay, eps, has_sparse_grad, maximize, differentiable,
    has_complex,
):
    if not params:
        return
    if has_sparse_grad or has_complex:
        raise RuntimeError("`fused` does not support sparse grad or complex param")
    if differentiable:
        raise RuntimeError(
            "adagrad with fused=True does not support differentiable=True"
        )

    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )
    lr_dict: DeviceDict | None = (
        {lr.device: lr}
        if isinstance(lr, Tensor) and lr.device.type != "cpu"
        else None
    )

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]
    )
    for (device, _), (
        (
            device_params_,
            device_grads_,
            device_state_sums_,
            device_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_state_sums = cast(list[Tensor], device_state_sums_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device, non_blocking=True)
            )
        if found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(
                device, found_inf.to(device, non_blocking=True)
            )
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)
            lr = lr_dict[device]

        tp._foreach_add_(device_state_steps, 1)
        tp._fused_adagrad_(
            device_params,
            device_grads,
            device_state_sums,
            device_state_steps,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        if device_found_inf is not None:
            tp._foreach_sub_(
                device_state_steps,
                [device_found_inf] * len(device_state_steps),
            )


def adagrad(
    params, grads, state_sums, state_steps, fused=None, grad_scale=None,
    found_inf=None, has_sparse_grad=False, foreach=None, differentiable=False,
    has_complex=False, *, lr, weight_decay, lr_decay, eps, maximize,
):
    if not all(isinstance(value, tp.Tensor) for value in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of "
            "singleton tensors"
        )
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    fused = bool(fused)
    foreach = bool(foreach)
    if fused:
        _fused_adagrad(
            params, grads, state_sums, state_steps, grad_scale, found_inf,
            lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, eps=eps,
            has_sparse_grad=has_sparse_grad, maximize=maximize,
            differentiable=differentiable, has_complex=has_complex,
        )
    elif foreach:
        _multi_tensor_adagrad(
            params, grads, state_sums, state_steps, grad_scale, found_inf,
            lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, eps=eps,
            has_sparse_grad=has_sparse_grad, maximize=maximize,
            differentiable=differentiable, has_complex=has_complex,
        )
    else:
        _single_tensor_adagrad(
            params, grads, state_sums, state_steps, grad_scale, found_inf,
            lr=lr, weight_decay=weight_decay, lr_decay=lr_decay, eps=eps,
            has_sparse_grad=has_sparse_grad, maximize=maximize,
            differentiable=differentiable, has_complex=has_complex,
        )
