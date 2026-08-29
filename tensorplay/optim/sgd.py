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
        group_params = group["params"]
        momentum_value = group["momentum"]
        has_momentum = momentum_value != 0
        fused = group["fused"]
        need_fused_check = fused and getattr(
            self, "_need_device_dtype_check_for_fused", True
        )
        native_cuda_group = (
            group["foreach"] is not False
            and not isinstance(group["lr"], tp.Tensor)
            and not isinstance(group["weight_decay"], tp.Tensor)
        )
        saw_momentum_buffer = False
        saw_missing_momentum_buffer = False
        state = self.state
        for p in group_params:
            grad = p.grad
            if grad is None:
                continue
            if need_fused_check:
                _device_dtype_check_for_fused(p)
                self._need_device_dtype_check_for_fused = False
                need_fused_check = False
            params.append(p)
            grads.append(grad)
            if grad.is_sparse:
                has_sparse_grad = True
                native_cuda_group = False
            # momentum multiply and add separately.  For half and bfloat16
            # that intermediate write is observable, whereas the fused MTA
            # kernel intentionally keeps the update in opmath precision.  Keep
            # the fused shortcut for f32/f64, and let low-precision groups use
            # the native foreach sequence below.  Explicit fused=True still
            # reaches _fused_sgd through its ordinary fused branch.
            if len(params) == 1 and (
                p.device.type != "cuda" or p.dtype not in (tp.float32, tp.float64)
            ):
                native_cuda_group = False
            if has_momentum:
                buffer = state[p].get("momentum_buffer")
                momentum_buffer_list.append(buffer)
                saw_momentum_buffer |= buffer is not None
                saw_missing_momentum_buffer |= buffer is None
        momentum_buffer_state = (
            2 if saw_momentum_buffer and saw_missing_momentum_buffer
            else 1 if saw_momentum_buffer else 0
        )
        return has_sparse_grad, native_cuda_group, momentum_buffer_state

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with tp.enable_grad():
                loss = closure()

        state = self.state
        for group in self.param_groups:
            params, grads, momentum_buffer_list = [], [], []
            has_sparse_grad, native_cuda_group, momentum_buffer_state = self._init_group(
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
                native_cuda_group=native_cuda_group,
                momentum_buffer_state=momentum_buffer_state,
            )
            # Existing buffers remain the same objects after an in-place
            # kernel update.  Avoid rewriting 100+ state dictionaries on the
            # steady-state step; only first/mixed initialization needs the
            # Python-side state writeback.
            if group["momentum"] != 0 and momentum_buffer_state != 1:
                for p, momentum_buffer in zip(
                    params, momentum_buffer_list, strict=True
                ):
                    state[p]["momentum_buffer"] = momentum_buffer

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
    lr_s = _to_scalar(lr)
    # Common-case fast path: run each parameter through the native batch
    # kernel independently (same per-parameter contract as the composed
    # body below, one vectorized call instead of ~6 dispatched ops).
    if (not isinstance(lr_s, tp.Tensor)
            and not isinstance(weight_decay, tp.Tensor)
            and not maximize
            and params
            and not has_sparse_grad
            and all(
                p.device.type in ("cpu", "cuda")
                and p.dtype in (tp.float32, tp.float64)
                and p.is_contiguous()
                and g.device == p.device
                and g.dtype == p.dtype
                and g.is_contiguous()
                and not g.is_sparse
                for p, g in zip(params, grads, strict=True)
            )):
        for index, param in enumerate(params):
            plist = [param]
            glist = [grads[index]]
            first_step = False
            bufs = []
            if momentum != 0:
                buf = momentum_buffer_list[index]
                if buf is None:
                    buf = glist[0].detach().clone()
                    momentum_buffer_list[index] = buf
                    first_step = True
                bufs = [buf]
            tp._fused_sgd_(
                plist, glist, bufs,
                weight_decay=weight_decay, momentum=momentum, lr=lr_s,
                dampening=dampening, nesterov=nesterov, maximize=False,
                is_first_step=first_step)
        return

    lr = _to_scalar(lr)
    for index, param in enumerate(params):
        grad = grads[index] if not maximize else -grads[index]
        if weight_decay != 0:
            if isinstance(weight_decay, tp.Tensor):
                if weight_decay.requires_grad:
                    grad = grad.addcmul_(param.clone(), weight_decay)
                else:
                    # TensorPlay's alpha overload accepts Python scalars only;
                    # keep a scalar Tensor weight decay elementwise instead.
                    grad = grad + param * weight_decay
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
            # eager mode.  TensorPlay's overload does not, so use the
            # equivalent elementwise product for both grad-tracked and
            # ordinary scalar Tensor learning rates.
            param.addcmul_(grad, lr, value=-1)
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
    native_cuda_group=None,
    momentum_buffer_state=None,
):
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")
    if not params:
        return
    lr = _to_scalar(lr)

    # The eager CUDA default is already a single homogeneous parameter group
    # in the common case.  Preserve the original list and enter the native
    # MTA kernel directly; building the device/dtype grouping dictionary and
    # copying three lists on every step costs more than the kernel for the
    # many-small-tensor optimizer workload.  Mixed-device/dtype and partially
    # initialized momentum groups still use the grouped fallback below.
    if native_cuda_group is None:
        native_cuda_group = (
            not has_sparse_grad
            and params
            and params[0].device.type == "cuda"
            and params[0].dtype in (tp.float32, tp.float64)
            and not isinstance(lr, tp.Tensor)
            and not isinstance(weight_decay, tp.Tensor)
            and all(
                p.is_contiguous()
                and g.device == p.device
                and g.dtype == p.dtype
                and g.is_contiguous()
                and not g.is_sparse
                for p, g in zip(params, grads, strict=True)
            )
        )
    else:
        native_cuda_group = (
            native_cuda_group
            and params[0].device.type == "cuda"
            and params[0].is_floating_point()
        )
    if native_cuda_group:
        if momentum != 0:
            # A partially initialized state list cannot be handed to the
            # fused helper; leave that uncommon transition on the grouped
            # foreach path so state initialization remains identical to
            if momentum_buffer_state == 2:
                native_cuda_group = False
            elif momentum_buffer_state is None:
                have_buf = [buf is not None for buf in momentum_buffer_list]
                if any(have_buf) and not all(have_buf):
                    native_cuda_group = False
    if native_cuda_group:
        # Reuse the fused helper's C++ device/dtype grouping.  Besides
        # avoiding a second Python list copy, it allocates first-step
        # momentum buffers directly and follows the same path as the
        # explicit fused optimizer option.  The homogeneous checks above
        # and the C++ validation keep unsupported groups on the fallback.
        try:
            _fused_sgd(
                params,
                grads,
                momentum_buffer_list,
                grad_scale,
                found_inf,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                has_sparse_grad=has_sparse_grad,
                native_cuda_group=native_cuda_group,
                momentum_buffer_state=momentum_buffer_state,
            )
            return
        except NotImplementedError:
            # Fused CUDA validation is the authoritative homogeneous-layout
            # check.  Unsupported views/dtypes can still use the foreach
            # composition below; the exception path is cold for normal model
            # parameter groups and avoids rechecking every tensor in Python.
            pass

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True
    )
    for (device_params, device_grads, device_buffers), indices in grouped_tensors.values():
        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads if grad is not None
        )

        # Native batch kernel: one dispatcher call advances the whole group
        # (horizontally fused work list + vectorized inner loops); math is
        # identical to the composed path below.
        use_native = (
            not device_has_sparse_grad
            and bool(device_params)
            and device_params[0].dtype in (tp.float32, tp.float64)
            and device_params[0].device.type in ("cpu", "cuda")
            and not isinstance(lr, tp.Tensor)
            and not isinstance(weight_decay, tp.Tensor)
            and all(
                p.is_contiguous()
                and g.device == p.device
                and g.dtype == p.dtype
                and g.shape == p.shape
                and g.is_contiguous()
                and not g.is_sparse
                for p, g in zip(device_params, device_grads, strict=True)
            )
        )
        # CUDA: prefer the fused kernel (single launch, on-device steps).
        first_step = False
        native_bufs: list = []
        if use_native and device_params[0].device.type == "cuda":
            if momentum != 0:
                have_buf = [b is not None for b in device_buffers]
                if all(have_buf):
                    native_bufs = list(device_buffers)
                elif not any(have_buf):
                    for bi, gr in enumerate(device_grads):
                        bf = gr.detach().clone()
                        device_buffers[bi] = bf
                        momentum_buffer_list[indices[bi]] = bf
                        native_bufs.append(bf)
                    first_step = True
                else:
                    use_native = False
            if use_native:
                tp._fused_sgd_(
                    device_params,
                    device_grads,
                    native_bufs,
                    weight_decay=weight_decay,
                    momentum=momentum,
                    lr=lr,
                    dampening=dampening,
                    nesterov=nesterov,
                    maximize=maximize,
                    is_first_step=first_step,
                )
                continue
        first_step = False
        native_bufs: list = []
        if use_native and momentum != 0:
            have_buf = [buf is not None for buf in device_buffers]
            if all(have_buf):
                native_bufs = list(device_buffers)
            elif not any(have_buf):
                for buf_idx, grad in enumerate(device_grads):
                    buf = grad.detach().clone()
                    device_buffers[buf_idx] = buf
                    momentum_buffer_list[indices[buf_idx]] = buf
                    native_bufs.append(buf)
                first_step = True
            else:
                use_native = False
        if use_native:
            tp._fused_sgd_(
                device_params,
                device_grads,
                native_bufs,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                is_first_step=first_step,
            )
            continue

        if maximize:
            device_grads = tp._foreach_neg(device_grads)
        if weight_decay != 0:
            if isinstance(weight_decay, tp.Tensor):
                device_grads = tp._foreach_add(
                    device_grads, tp._foreach_mul(device_params, weight_decay)
                )
            elif maximize:
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
            if isinstance(lr, tp.Tensor):
                tp._foreach_add_(device_params, tp._foreach_mul(device_grads, -lr))
            else:
                tp._foreach_add_(device_params, device_grads, alpha=-lr)
        else:
            for index, param in enumerate(device_params):
                if isinstance(lr, tp.Tensor):
                    param.add_(device_grads[index] * (-lr))
                else:
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
    native_cuda_group=None,
    momentum_buffer_state=None,
):
    if not params:
        return
    if has_sparse_grad:
        raise RuntimeError("`_fused_sgd` does not support sparse gradients")

    no_momentum_buffer = momentum == 0
    is_first_step = (
        (momentum_buffer_state == 0
         if momentum_buffer_state is not None
         else all(t is None for t in momentum_buffer_list))
        and not no_momentum_buffer
    )

    # In the steady state the optimizer already knows that this is one
    # homogeneous CUDA group and that every momentum buffer exists.  Call
    # the native entry point directly instead of rebuilding the device/dtype
    # grouping dictionary on every step.  C++ remains the authoritative
    # validator; an unsupported mixed/layout group falls through unchanged.
    if (
        native_cuda_group
        and grad_scale is None
        and found_inf is None
        and params[0].device.type == "cuda"
        and params[0].is_floating_point()
        and (no_momentum_buffer or momentum_buffer_state == 1)
    ):
        try:
            tp._fused_sgd_(
                params,
                grads,
                [] if no_momentum_buffer else momentum_buffer_list,
                weight_decay=weight_decay,
                momentum=momentum,
                lr=lr,
                dampening=dampening,
                nesterov=nesterov,
                maximize=maximize,
                is_first_step=False,
                grad_scale=None,
                found_inf=None,
            )
            return
        except NotImplementedError:
            pass

    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
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
    native_cuda_group=None,
    momentum_buffer_state=None,
):
    if foreach is None and fused is None:
        # The group initializer already established the common CUDA native
        # on every step; the native helper still validates the complete list.
        if (
            native_cuda_group
            and params
            and params[0].device.type == "cuda"
            and params[0].is_floating_point()
        ):
            fused, foreach = False, True
        else:
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
            native_cuda_group=native_cuda_group,
            momentum_buffer_state=momentum_buffer_state,
        )
    elif fused:
        _fused_sgd(
            params, d_p_list, momentum_buffer_list, grad_scale, found_inf,
            weight_decay=weight_decay, momentum=momentum, lr=lr,
            dampening=dampening, nesterov=nesterov, maximize=maximize,
            has_sparse_grad=has_sparse_grad,
            native_cuda_group=native_cuda_group,
            momentum_buffer_state=momentum_buffer_state,
        )
    else:
        _single_tensor_sgd(
            params, d_p_list, momentum_buffer_list, grad_scale, found_inf,
            weight_decay=weight_decay, momentum=momentum, lr=lr,
            dampening=dampening, nesterov=nesterov, maximize=maximize,
            has_sparse_grad=has_sparse_grad,
        )
