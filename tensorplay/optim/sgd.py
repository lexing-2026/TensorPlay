from typing import cast

import tensorplay as tp
from tensorplay import Tensor

from ._utils import scalar_value
from .optimizer import (
    DeviceDict,
    Optimizer,
    _default_to_fused_or_foreach,
    _device_dtype_check_for_fused,
    _to_scalar,
    _use_grad_for_differentiable,
    ParamsT,
)


__all__ = ["SGD", "sgd"]


class SGD(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float | Tensor = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float | Tensor = 0,
        nesterov: bool = False,
        *,
        maximize: bool = False,
        foreach: bool | None = None,
        differentiable: bool = False,
        fused: bool | None = None,
    ) -> None:
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if scalar_value(lr, "lr") < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if scalar_value(weight_decay, "weight_decay") < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        maximize=maximize, foreach=foreach,
                        differentiable=differentiable, fused=fused)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False
        for p in group["params"]:
            if p.grad is None:
                continue
            if group["fused"] and getattr(
                self, "_need_device_dtype_check_for_fused", True
            ):
                _device_dtype_check_for_fused(p)
                self._need_device_dtype_check_for_fused = False
            params.append(p)
            grads.append(p.grad)
            if p.grad.is_sparse:
                has_sparse_grad = True
            if group["momentum"] != 0:
                momentum_buffer_list.append(self.state[p].get("momentum_buffer"))
        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params, grads, momentum_buffer_list = [], [], []
            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )
            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )
            if group["momentum"] != 0:
                for p, momentum_buffer in zip(
                    params, momentum_buffer_list, strict=True
                ):
                    self.state[p]["momentum_buffer"] = momentum_buffer

        return loss


def _single_tensor_sgd(
    params,
    grads,
    momentum_buffer_list,
    grad_scale,
    found_inf,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
    has_sparse_grad,
):
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")
    lr = _to_scalar(lr)
    for index, param in enumerate(params):
        grad = grads[index] if not maximize else -grads[index]
        if weight_decay != 0:
            if isinstance(weight_decay, tp.Tensor):
                if weight_decay.requires_grad:
                    grad = grad.addcmul_(param.clone(), weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)
            else:
                grad = grad.add(param, alpha=weight_decay)
        if momentum != 0:
            buf = momentum_buffer_list[index]
            if buf is None:
                buf = grad.detach().clone()
                momentum_buffer_list[index] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)
            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf
        if isinstance(lr, tp.Tensor):
            if lr.requires_grad:
                param.addcmul_(grad, lr, value=-1)
            else:
                param.add_(grad, alpha=-lr)
        else:
            param.add_(grad, alpha=-lr)


def _multi_tensor_sgd(
    params,
    grads,
    momentum_buffer_list,
    grad_scale,
    found_inf,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
    has_sparse_grad,
):
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")
    if not params:
        return
    lr = _to_scalar(lr)
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True
    )
    for (device_params, device_grads, device_buffers), indices in grouped_tensors.values():
        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads if grad is not None
        )
        if maximize:
            device_grads = tp._foreach_neg(device_grads)
        if weight_decay != 0:
            if maximize:
                tp._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = tp._foreach_add(
                    device_grads, device_params, alpha=weight_decay
                )
        if momentum != 0:
            bufs = []
            all_states = all(buf is not None for buf in device_buffers)
            if all_states:
                bufs = list(device_buffers)
                tp._foreach_mul_(bufs, momentum)
                tp._foreach_add_(bufs, device_grads, alpha=1 - dampening)
            else:
                for index, buf in enumerate(device_buffers):
                    if buf is None:
                        buf = device_buffers[index] = momentum_buffer_list[
                            indices[index]
                        ] = device_grads[index].detach().clone()
                    else:
                        buf.mul_(momentum).add_(device_grads[index], alpha=1 - dampening)
                    bufs.append(buf)
            if nesterov:
                tp._foreach_add_(device_grads, bufs, alpha=momentum)
            else:
                device_grads = bufs
        if not device_has_sparse_grad:
            if isinstance(lr, tp.Tensor) and tp.compiler.is_compiling():
                tp._foreach_add_(device_params, tp._foreach_mul(device_grads, -lr))
            else:
                tp._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            for index, param in enumerate(device_params):
                param.add_(device_grads[index], alpha=-lr)


def _fused_sgd(
    params,
    grads,
    momentum_buffer_list,
    grad_scale,
    found_inf,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
    has_sparse_grad,
):
    if not params:
        return
    if has_sparse_grad:
        raise RuntimeError("`_fused_sgd` does not support sparse gradients")
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    no_momentum_buffer = momentum == 0
    is_first_step = (
        all(t is None for t in momentum_buffer_list) and not no_momentum_buffer
    )
    if is_first_step:
        for i, g in enumerate(grads):
            momentum_buffer_list[i] = tp.empty_like(g)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=False
    )
    for (device, _), (
        (device_params_, device_grads_, device_momentum_buffer_list),
        _,
    ) in grouped_tensors.items():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_grad_scale, device_found_inf = None, None
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device)
            )
        if found_inf_dict is not None and found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(
                device, found_inf.to(device)
            )
        tp._fused_sgd_(
            device_params,
            device_grads,
            []
            if no_momentum_buffer
            else cast(list[Tensor], device_momentum_buffer_list),
            weight_decay=weight_decay,
            momentum=momentum,
            lr=lr,
            dampening=dampening,
            nesterov=nesterov,
            maximize=maximize,
            is_first_step=is_first_step,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )


def sgd(
    params,
    d_p_list,
    momentum_buffer_list,
    has_sparse_grad=False,
    foreach=None,
    fused=None,
    grad_scale=None,
    found_inf=None,
    *,
    weight_decay,
    momentum,
    lr,
    dampening,
    nesterov,
    maximize,
):
    if foreach is None and fused is None:
        fused, foreach = _default_to_fused_or_foreach(
            params, differentiable=False, use_fused=False
        )
    foreach = bool(foreach)
    fused = bool(fused)
    if foreach:
        _multi_tensor_sgd(
            params, d_p_list, momentum_buffer_list, grad_scale, found_inf,
            weight_decay=weight_decay, momentum=momentum, lr=lr,
            dampening=dampening, nesterov=nesterov, maximize=maximize,
            has_sparse_grad=has_sparse_grad,
        )
    elif fused:
        _fused_sgd(
            params, d_p_list, momentum_buffer_list, grad_scale, found_inf,
            weight_decay=weight_decay, momentum=momentum, lr=lr,
            dampening=dampening, nesterov=nesterov, maximize=maximize,
            has_sparse_grad=has_sparse_grad,
        )
    else:
        _single_tensor_sgd(
            params, d_p_list, momentum_buffer_list, grad_scale, found_inf,
            weight_decay=weight_decay, momentum=momentum, lr=lr,
            dampening=dampening, nesterov=nesterov, maximize=maximize,
            has_sparse_grad=has_sparse_grad,
        )
