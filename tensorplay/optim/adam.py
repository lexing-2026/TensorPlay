import math
from typing import cast
import tensorplay as tp
from .optimizer import (
    DeviceDict,
    Optimizer,
    _default_to_fused_or_foreach,
    _disable_dynamo_if_unsupported,
    _get_capturable_supported_devices,
    _get_value,
    _stack_if_compiling,
    _to_scalar,
    _use_grad_for_differentiable,
)
from ._utils import (
    capturable_supported,
    ensure_state_step,
    scalar_value,
    zeros_like,
)


__all__ = ["Adam", "adam"]


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, *, foreach=None,
                 maximize=False, capturable=False, differentiable=False,
                 fused=None, decoupled_weight_decay=False):
        if isinstance(lr, tp.Tensor) and foreach and not capturable:
            raise ValueError(
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )
        if isinstance(lr, tp.Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= scalar_value(lr, "lr"):
            raise ValueError("Invalid learning rate: {}".format(lr))
        if scalar_value(eps, "eps") < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if isinstance(betas[0], tp.Tensor) and betas[0].numel() != 1:
            raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], tp.Tensor) and betas[1].numel() != 1:
            raise ValueError("Tensor betas[1] must be 1-element")
        if not ((isinstance(betas[0], tp.Tensor) and isinstance(betas[1], tp.Tensor)) or
                (not isinstance(betas[0], tp.Tensor) and
                 not isinstance(betas[1], tp.Tensor))):
            raise ValueError("betas must be either both floats or both Tensors")
        if isinstance(betas[0], tp.Tensor) and foreach and not capturable:
            raise ValueError(
                "betas[0] as a Tensor is not supported for capturable=False and foreach=True"
            )
        if isinstance(betas[1], tp.Tensor) and foreach and not capturable:
            raise ValueError(
                "betas[1] as a Tensor is not supported for capturable=False and foreach=True"
            )
        if not 0.0 <= scalar_value(betas[0], "beta parameter at index 0") < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= scalar_value(betas[1], "beta parameter at index 1") < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= scalar_value(weight_decay, "weight_decay"):
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        foreach=foreach, maximize=maximize,
                        capturable=capturable, differentiable=differentiable,
                        fused=fused,
                        decoupled_weight_decay=decoupled_weight_decay)
        super(Adam, self).__init__(params, defaults)
        # Eager-only dispatch cache; it is intentionally not part of the
        # serialized optimizer state.
        self._tp_adam_layout_cache = {}
        if fused and differentiable:
            raise RuntimeError("`fused` does not support `differentiable`")
        if fused and foreach:
            raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "_tp_adam_layout_cache"):
            self._tp_adam_layout_cache = {}
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, {})
                if p_state and not isinstance(p_state.get("step"), tp.Tensor):
                    p_state["step"] = tp.tensor(
                        float(p_state["step"]), dtype=tp.float32,
                        device=p.device if (group["capturable"] or group["fused"])
                        else tp.device("cpu"),
                    )

    def _init_group(
            self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs,
            max_exp_avg_sqs, state_steps):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue
            has_complex |= p.is_complex()
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError(
                    "Adam does not support sparse gradients, please consider SparseAdam instead"
                )
            grads.append(p.grad)

            state = self.state[p]
            if len(state) == 0:
                state["step"] = tp.tensor(
                    0.0,
                    dtype=tp.float32,
                    # non-fused optimizer keeps the scalar step on CPU even
                    # when the parameter lives on CUDA.  This avoids a host
                    # synchronization for every ordinary foreach step and is
                    # also part of the native optimizer state contract.
                    device=p.device
                    if (group["capturable"] or group["fused"])
                    else tp.device("cpu"),
                )
                state["exp_avg"] = zeros_like(p)
                state["exp_avg_sq"] = zeros_like(p)
                if group["amsgrad"]:
                    state["max_exp_avg_sq"] = zeros_like(p)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])
            if group["differentiable"] and state["step"].requires_grad:
                raise RuntimeError(
                    "`requires_grad` is not supported for `step` in differentiable mode"
                )
            if (
                group["foreach"]
                and isinstance(group["lr"], tp.Tensor)
                and not group["capturable"]
            ):
                raise RuntimeError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                decoupled_weight_decay=group["decoupled_weight_decay"],
                layout_cache=self._tp_adam_layout_cache.setdefault(id(group), {}),
            )

        return loss


def _adam_layout_cache_key(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
        amsgrad, step_device):
    """Build a cheap fingerprint for the fused-layout probe.

    Materializing ``shape`` tuples for every tensor is costly for large
    optimizer groups.  For the positive (contiguous) case, ``stride`` and
    ``numel`` capture shape changes.  Parameter/state identities and data
    pointers catch storage replacement; gradients intentionally use layout
    metadata only so a newly allocated gradient from ``set_to_none=True``
    does not invalidate an otherwise identical optimizer group.  The native
    kernel still performs the authoritative full validation on every call.
    """
    key = [amsgrad, step_device, len(params)]
    for p, g, m, v in zip(params, grads, exp_avgs, exp_avg_sqs, strict=True):
        key.append((
            id(p), id(m), id(v),
            p.data_ptr(), m.data_ptr(), v.data_ptr(),
            p.numel(), g.numel(), m.numel(), v.numel(),
            g.dtype, g.device,
            p.is_contiguous(), g.is_contiguous(),
            m.is_contiguous(), v.is_contiguous(),
            p.stride(), g.stride(), m.stride(), v.stride(),
        ))
    if amsgrad:
        for p, max_v in zip(params, max_exp_avg_sqs, strict=True):
            key.append((
                id(p), id(max_v), max_v.data_ptr(), max_v.numel(),
                max_v.is_contiguous(), max_v.stride(),
            ))
    for step in state_steps:
        key.append((
            id(step), step.data_ptr(), step.numel(),
            step.is_contiguous(), step.stride(),
        ))
    return tuple(key)


def _adam_fused_layout_ready(
    params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
    amsgrad, step_device, layout_cache=None,
):
    """Check the layout accepted by the native fused Adam kernels.

    The Python reference implementation accepts strided tensors.  Keeping
    this check at the dispatch boundary is important: an explicit fused
    attempt must not turn an otherwise valid foreach/single call into a
    backend layout error, and a failed probe must not advance ``step`` twice.
    """
    if not params or len(params) != len(grads):
        return False
    first_device = params[0].device
    first_dtype = params[0].dtype
    if first_dtype not in (tp.float32, tp.float64):
        # The exact low-precision path is only valid for the ordinary
        # non-capturable host-step foreach sequence.  Device-step/capturable
        # callers retain the existing fused or composed dispatch.
        if step_device != "cpu" or first_dtype not in (tp.float16, tp.bfloat16):
            return False
    if first_device.type not in ("cpu", "cuda"):
        return False
    if len(exp_avgs) != len(params) or len(exp_avg_sqs) != len(params):
        return False
    if len(state_steps) != len(params):
        return False
    if amsgrad and len(max_exp_avg_sqs) != len(params):
        return False
    if not amsgrad and max_exp_avg_sqs:
        return False

    cache_key = None
    if layout_cache is not None:
        cache_key = _adam_layout_cache_key(
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
            state_steps, amsgrad, step_device,
        )
        if layout_cache.get("key") == cache_key:
            return layout_cache["ready"]

    ready = True
    for p, g, m, v in zip(params, grads, exp_avgs, exp_avg_sqs, strict=True):
        if (
            p.device != first_device or p.dtype != first_dtype
            or not p.is_contiguous()
            or g.is_sparse or g.device != first_device
            or g.dtype != first_dtype or g.shape != p.shape
            or not g.is_contiguous()
            or m.device != first_device or m.dtype != first_dtype
            or m.shape != p.shape or not m.is_contiguous()
            or v.device != first_device or v.dtype != first_dtype
            or v.shape != p.shape or not v.is_contiguous()
        ):
            ready = False
            break
    if ready and amsgrad:
        for p, max_v in zip(params, max_exp_avg_sqs, strict=True):
            if (
                max_v.device != first_device or max_v.dtype != first_dtype
                or max_v.shape != p.shape or not max_v.is_contiguous()
            ):
                ready = False
                break
    if ready and step_device == "cpu":
        ready = all(
            step.device.type == "cpu"
            and step.numel() == 1
            and step.is_contiguous()
            for step in state_steps
        )
    elif ready:
        ready = all(
            step.device == first_device
            and step.numel() == 1
            and step.is_contiguous()
            for step in state_steps
        )
    if layout_cache is not None:
        layout_cache["key"] = cache_key
        layout_cache["ready"] = ready
    return ready


def _single_tensor_adam(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
        grad_scale, found_inf, *, amsgrad, has_complex, beta1, beta2, lr,
        weight_decay, eps, maximize, capturable, differentiable,
        decoupled_weight_decay, layout_cache=None):
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")

    # Common-case fast path: run each parameter through the native batch
    # kernel independently (same per-parameter contract as the composed
    # body below, one vectorized call instead of ~15 dispatched ops).
    if not capturable and not differentiable:
        lr_s = _to_scalar(lr)
        b1_s = _to_scalar(beta1)
        b2_s = _to_scalar(beta2)
        fused_ready = (
            isinstance(lr_s, (int, float))
            and isinstance(b1_s, (int, float))
            and isinstance(b2_s, (int, float))
            and not isinstance(weight_decay, tp.Tensor)
            and not isinstance(eps, tp.Tensor)
            and _adam_fused_layout_ready(
                params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                state_steps, amsgrad, "cpu",
                layout_cache if not has_complex else None,
            )
        )
        if fused_ready:
            for i, param in enumerate(params):
                plist = [param]
                glist = [grads[i]]
                mlist = [exp_avgs[i]]
                vlist = [exp_avg_sqs[i]]
                xlist = [max_exp_avg_sqs[i]] if amsgrad else []
                step_t = state_steps[i]
                host_cuda_step = (
                    param.device.type == "cuda" and step_t.device.type == "cpu"
                )
                if not host_cuda_step:
                    step_t.add_(1)
                fused_fn = (
                    tp._fused_adamw_
                    if decoupled_weight_decay
                    else tp._fused_adam_
                )
                fused_fn(
                    plist, glist, mlist, vlist, xlist, [step_t],
                    lr=lr_s, beta1=b1_s, beta2=b2_s,
                    weight_decay=scalar_value(weight_decay, "weight_decay"),
                    eps=scalar_value(eps, "eps"), amsgrad=amsgrad,
                    maximize=maximize)
            return

    lr = _to_scalar(lr)
    beta1 = _to_scalar(beta1)
    beta2 = _to_scalar(beta2)
    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        lr_param = (
            lr.to(device=param.device) if isinstance(lr, tp.Tensor) else lr
        )
        weight_decay_param = (
            weight_decay.to(device=param.device)
            if isinstance(weight_decay, tp.Tensor)
            else weight_decay
        )

        if capturable:
            supported = _get_capturable_supported_devices()
            if param.device.type != step_t.device.type or param.device.type not in supported:
                raise AssertionError(
                    "If capturable=True, params and state_steps must be on supported devices: "
                    f"{supported}."
                )
        step_t.add_(1)

        if weight_decay != 0:
            if decoupled_weight_decay:
                param.mul_(1 - lr_param * weight_decay_param)
            elif isinstance(weight_decay, tp.Tensor):
                grad = grad + param * weight_decay_param
            else:
                grad = grad.add(param, alpha=weight_decay)

        is_complex = param.is_complex()
        if is_complex:
            grad = tp.view_as_real(grad)
            exp_avg = tp.view_as_real(exp_avg)
            exp_avg_sq = tp.view_as_real(exp_avg_sq)
            param = tp.view_as_real(param)
            if amsgrad:
                max_exp_avg_sqs[i] = tp.view_as_real(max_exp_avg_sqs[i])

        beta1_param = (
            beta1.to(device=exp_avg.device, dtype=exp_avg.dtype)
            if isinstance(beta1, tp.Tensor) else beta1
        )
        beta2_param = (
            beta2.to(device=exp_avg_sq.device, dtype=exp_avg_sq.dtype)
            if isinstance(beta2, tp.Tensor) else beta2
        )
        if isinstance(beta1_param, tp.Tensor):
            exp_avg.lerp_(grad, 1 - beta1_param)
        else:
            exp_avg.mul_(beta1_param).add_(grad * (1 - beta1_param))
        if isinstance(beta2_param, tp.Tensor):
            exp_avg_sq.lerp_(grad * grad, 1 - beta2_param)
        else:
            exp_avg_sq.mul_(beta2_param).add_(grad * grad * (1 - beta2_param))

        tensor_math = (
            isinstance(lr_param, tp.Tensor)
            or isinstance(beta1_param, tp.Tensor)
            or isinstance(beta2_param, tp.Tensor)
        )
        if capturable or differentiable or tensor_math:
            step = step_t if (capturable or differentiable) else _get_value(step_t)
            bias_correction1 = 1 - beta1_param ** step
            bias_correction2 = 1 - beta2_param ** step
            step_size = lr_param / bias_correction1
            step_size_neg = -step_size
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]
                max_exp_avg_sq.copy_(tp.maximum(max_exp_avg_sq, exp_avg_sq))
                numerator = max_exp_avg_sq.sqrt()
            else:
                numerator = exp_avg_sq.sqrt()
            bias_correction2_sqrt = (
                bias_correction2.sqrt()
                if isinstance(bias_correction2, tp.Tensor)
                else bias_correction2 ** 0.5
            )
            denom = numerator / (bias_correction2_sqrt * step_size_neg)
            denom.add_(eps / step_size_neg)
            update = exp_avg.clone() if differentiable else exp_avg
            param.addcdiv_(update, denom)
        else:
            step = _get_value(step_t)
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]
                max_exp_avg_sq.copy_(tp.maximum(max_exp_avg_sq, exp_avg_sq))
                denom = (max_exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


def _multi_tensor_adam(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
        grad_scale, found_inf, *, amsgrad, has_complex, beta1, beta2, lr,
        weight_decay, eps, maximize, capturable, differentiable,
        decoupled_weight_decay, layout_cache=None):
    if not params:
        return
    if grad_scale is not None or found_inf is not None:
        raise AssertionError("Expected grad_scale and found_inf to be None")
    if differentiable:
        raise AssertionError("_foreach ops don't support autograd")
    if isinstance(lr, tp.Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )
    if isinstance(beta1, tp.Tensor) and not capturable:
        raise ValueError(
            "beta1 as a Tensor is not supported for capturable=False and foreach=True"
        )
    if isinstance(beta2, tp.Tensor) and not capturable:
        raise ValueError(
            "beta2 as a Tensor is not supported for capturable=False and foreach=True"
        )
    if not tp.compiler.is_compiling() and capturable:
        supported = _get_capturable_supported_devices()
        if not all(
            p.device.type == step.device.type
            and p.device.type in supported
            for p, step in zip(params, state_steps, strict=True)
        ):
            raise AssertionError(
                "If capturable=True, params and state_steps must be on "
                f"supported devices: {supported}."
            )

    device_params = list(params)
    device_grads = list(grads)
    device_exp_avgs = list(exp_avgs)
    device_exp_avg_sqs = list(exp_avg_sqs)
    device_state_steps = list(state_steps)
    device_max_exp_avg_sqs = list(max_exp_avg_sqs)
    if has_complex:
        device_params = [tp.view_as_real(p) if p.is_complex() else p for p in device_params]
        device_grads = [tp.view_as_real(g) if g.is_complex() else g for g in device_grads]
        device_exp_avgs = [tp.view_as_real(v) if v.is_complex() else v for v in device_exp_avgs]
        device_exp_avg_sqs = [tp.view_as_real(v) if v.is_complex() else v for v in device_exp_avg_sqs]
        if amsgrad:
            device_max_exp_avg_sqs = [
                tp.view_as_real(v) if v.is_complex() else v
                for v in device_max_exp_avg_sqs
            ]

    native_host_steps = (
        device_params
        and device_params[0].device.type == "cuda"
        and device_state_steps
        and device_state_steps[0].device.type == "cpu"
        and not capturable
        and not has_complex
        and not isinstance(lr, tp.Tensor)
        and not isinstance(beta1, tp.Tensor)
        and not isinstance(beta2, tp.Tensor)
        and not isinstance(weight_decay, tp.Tensor)
        and not isinstance(eps, tp.Tensor)
        and _adam_fused_layout_ready(
            device_params, device_grads, device_exp_avgs, device_exp_avg_sqs,
            device_max_exp_avg_sqs, device_state_steps, amsgrad, "cpu",
            layout_cache if not has_complex else None,
        )
    )
    if not native_host_steps:
        tp._foreach_add_(device_state_steps, 1)

    # fused host-step path; use it before materialising the composed foreach
    # intermediates.  The host kernel is minimizing internally, so materialize
    # also preserves the native Adam/AdamW distinction: coupled decay is added
    # to that gradient, while decoupled decay is applied to the parameter.
    if native_host_steps:
        fn_mta = (tp._fused_adamw_ if decoupled_weight_decay
                  else tp._fused_adam_)
        fn_mta(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad,
            maximize=maximize,
            exact=True,
        )
        return

    # Native batch kernel: one dispatcher call advances the whole group
    # (horizontally fused work list + vectorized inner loops), instead of
    # ~15 foreach pointwise launches.  Math is identical to the composed
    # path below; unsupported configurations fall through to it.
    if (
        not capturable
        and not differentiable
        and device_params
        and device_params[0].dtype in (tp.float32, tp.float64)
        and device_params[0].device.type in ("cpu", "cuda")
        # fused kernels require state_steps colocated with params
        and device_state_steps
        and device_state_steps[0].device.type == device_params[0].device.type
        and _adam_fused_layout_ready(
            device_params, device_grads, device_exp_avgs, device_exp_avg_sqs,
            device_max_exp_avg_sqs, device_state_steps, amsgrad, "device",
            layout_cache if not has_complex else None,
        )
    ):
        # Native fused kernels read state_steps on-device (no per-parameter
        # host synchronization) and fuse the whole update into one launch.
        simple_wd = not isinstance(weight_decay, tp.Tensor)
        if simple_wd:
            # state_steps were already incremented above.
            fn_mta = (tp._fused_adamw_ if decoupled_weight_decay
                      else tp._fused_adam_)
            fn_mta(
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
                list(device_max_exp_avg_sqs) if amsgrad else [],
                device_state_steps,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                weight_decay=weight_decay,
                eps=eps,
                amsgrad=amsgrad,
                maximize=maximize,
                exact=True,
            )
            return

    if maximize:
        device_grads = tp._foreach_neg(device_grads)

    if weight_decay != 0:
        if decoupled_weight_decay:
            tp._foreach_mul_(device_params, 1 - lr * weight_decay)
        elif isinstance(weight_decay, tp.Tensor):
            device_grads = tp._foreach_add(
                device_grads, tp._foreach_mul(device_params, weight_decay)
            )
        elif maximize:
            tp._foreach_add_(device_grads, device_params, alpha=weight_decay)
        else:
            device_grads = tp._foreach_add(
                device_grads, device_params, alpha=weight_decay
            )

    if isinstance(beta1, tp.Tensor):
        beta1 = beta1.to(device=device_params[0].device, dtype=device_params[0].dtype)
        tp._foreach_mul_(device_exp_avgs, beta1)
        tp._foreach_add_(
            device_exp_avgs,
            tp._foreach_mul(device_grads, 1 - beta1),
        )
    else:
        tp._foreach_lerp_(device_exp_avgs, device_grads, 1 - beta1)

    tp._foreach_mul_(device_exp_avg_sqs, beta2)
    if isinstance(beta2, tp.Tensor):
        scaled_grads = tp._foreach_mul(device_grads, 1 - beta2)
        tp._foreach_addcmul_(device_exp_avg_sqs, scaled_grads, device_grads, value=1.0)
    else:
        tp._foreach_addcmul_(
            device_exp_avg_sqs, device_grads, device_grads, value=1 - beta2
        )

    if capturable:
        bias_correction1 = tp._foreach_pow(beta1, device_state_steps)
        bias_correction2 = tp._foreach_pow(beta2, device_state_steps)
        tp._foreach_sub_(bias_correction1, 1)
        tp._foreach_sub_(bias_correction2, 1)
        tp._foreach_neg_(bias_correction2)
        tp._foreach_div_(bias_correction1, lr)
        tp._foreach_reciprocal_(bias_correction1)
        tp._foreach_sqrt_(bias_correction2)
        if amsgrad:
            tp._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
            denom = tp._foreach_sqrt(device_max_exp_avg_sqs)
        else:
            denom = tp._foreach_sqrt(device_exp_avg_sqs)
        tp._foreach_div_(denom, bias_correction2)
        tp._foreach_add_(denom, eps)
        tp._foreach_div_(denom, bias_correction1)
        tp._foreach_addcdiv_(device_params, device_exp_avgs, denom)
    else:
        bias_correction1 = [1 - beta1 ** _get_value(step) for step in device_state_steps]
        bias_correction2 = [1 - beta2 ** _get_value(step) for step in device_state_steps]
        step_size = _stack_if_compiling([-lr / bc for bc in bias_correction1])
        if amsgrad:
            tp._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)
            denom = tp._foreach_sqrt(device_max_exp_avg_sqs)
        else:
            denom = tp._foreach_sqrt(device_exp_avg_sqs)
        tp._foreach_div_(denom, [bc ** 0.5 for bc in bias_correction2])
        tp._foreach_add_(denom, eps)
        tp._foreach_addcdiv_(device_params, device_exp_avgs, denom, step_size)


def _fused_adam(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
        grad_scale, found_inf, *, amsgrad, has_complex, beta1, beta2, lr,
        weight_decay, eps, maximize, capturable, differentiable,
        decoupled_weight_decay, layout_cache=None):
    if not params:
        return
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    beta1 = _to_scalar(beta1)
    beta2 = _to_scalar(beta2)

    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    # Keep a CUDA lr tensor on the same device as each grouped parameter list;
    # a CPU lr tensor remains the scalar-like input accepted by the backend.
    lr_dict: DeviceDict | None = (
        {lr.device: lr}
        if isinstance(lr, tp.Tensor) and lr.device.type != "cpu"
        else None
    )
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )
    for (device, _), (
        (
            device_params_,
            device_grads_,
            device_exp_avgs_,
            device_exp_avg_sqs_,
            device_max_exp_avg_sqs,
            device_state_steps_,
        ),
        _,
    ) in grouped_tensors.items():
        device_params = cast(list[tp.Tensor], device_params_)
        device_grads = cast(list[tp.Tensor], device_grads_)
        device_exp_avgs = cast(list[tp.Tensor], device_exp_avgs_)
        device_exp_avg_sqs = cast(list[tp.Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[tp.Tensor], device_state_steps_)

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

        host_cuda_steps = (
            device.type == "cuda"
            and device_state_steps
            and device_state_steps[0].device.type == "cpu"
        )
        if not host_cuda_steps:
            tp._foreach_add_(device_state_steps, 1)
        fn = tp._fused_adamw_ if decoupled_weight_decay else tp._fused_adam_
        fn(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
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


@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adam)
def adam(
        params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
        foreach=None, capturable=False, differentiable=False, fused=None,
        grad_scale=None, found_inf=None, has_complex=False,
        decoupled_weight_decay=False, *, amsgrad, beta1, beta2, lr,
        weight_decay, eps, maximize, layout_cache=None):
    """Functional API that performs the Adam algorithm computation."""
    # TensorPlay has a native CPU fused body with the same state/update
    # contract; select it only for the ordinary homogeneous eager layout so
    # the public default avoids one Python/native dispatch per parameter.
    default_dispatch = fused is None and foreach is None
    if default_dispatch:
        cpu_native_ready = (
            not capturable
            and not differentiable
            and grad_scale is None
            and found_inf is None
            and not has_complex
            and bool(params)
            and params[0].device.type == "cpu"
            and not isinstance(lr, tp.Tensor)
            and not isinstance(beta1, tp.Tensor)
            and not isinstance(beta2, tp.Tensor)
            and not isinstance(weight_decay, tp.Tensor)
            and not isinstance(eps, tp.Tensor)
            and _adam_fused_layout_ready(
                params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs,
                state_steps, amsgrad, "cpu", layout_cache,
            )
        )
        if cpu_native_ready:
            fused, foreach = True, False
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        if foreach and isinstance(lr, tp.Tensor) and not capturable:
            foreach = False
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False
    if not tp.compiler.is_compiling() and not all(
        isinstance(step, tp.Tensor) for step in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if fused:
        func = _fused_adam
    elif foreach:
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        grad_scale,
        found_inf,
        amsgrad=amsgrad,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        decoupled_weight_decay=decoupled_weight_decay,
        layout_cache=layout_cache,
    )
