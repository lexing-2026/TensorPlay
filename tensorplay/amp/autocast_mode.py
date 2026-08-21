# mypy: allow-untyped-defs
import collections
import functools
import threading
import warnings
import weakref
from typing import Any

import tensorplay
import tensorplay._C as _C

try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

__all__ = [
    "autocast_decorator",
    "autocast",
    "is_autocast_available",
    "custom_fwd",
    "custom_bwd",
]

# ---------------------------------------------------------------------------
# Autocast state
#
# torch implements the functions below as C++ bindings (torch._C) backed by
# per-device autocast state on the dispatcher.  TensorPlay has no autocast
# dispatch key yet, so the same contract is provided with thread-local state.
# The public signatures match the torch bindings, including the
# ``device_type='cuda'`` defaults.
# ---------------------------------------------------------------------------

_SUPPORTED_DEVICES = ("cuda", "cpu")

_FLOAT_DTYPES = (
    tensorplay.float16,
    tensorplay.bfloat16,
    tensorplay.float32,
    tensorplay.float64,
)

_DEFAULT_DTYPE = {"cuda": tensorplay.float16, "cpu": tensorplay.bfloat16}

_tls = threading.local()


class _AutocastState:
    __slots__ = ("enabled", "dtype")

    def __init__(self, device_type):
        self.enabled = False
        self.dtype = _DEFAULT_DTYPE.get(device_type, tensorplay.float16)


def _get_autocast_states():
    states = getattr(_tls, "states", None)
    if states is None:
        states = _tls.states = {}
    return states


def _get_autocast_state(device_type):
    states = _get_autocast_states()
    state = states.get(device_type)
    if state is None:
        state = states[device_type] = _AutocastState(device_type)
    return state


def _get_nesting():
    return getattr(_tls, "nesting", 0)


def _set_nesting(nesting):
    _tls.nesting = nesting


def _get_cache_enabled():
    return getattr(_tls, "cache_enabled", True)


def _set_cache_enabled(flag):
    _tls.cache_enabled = flag


_cast_cache = {}


def is_autocast_available(device_type: str) -> bool:
    r"""
    Return a bool indicating if autocast is available on :attr:`device_type`.

    Args:
        device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu'.
            The type is the same as the `type` attribute of a :class:`tensorplay.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
    return device_type in _SUPPORTED_DEVICES


def is_autocast_enabled(device_type: str = "cuda") -> bool:
    r"""Return True if autocast mode is enabled on :attr:`device_type`."""
    return _get_autocast_state(device_type).enabled


def get_autocast_dtype(device_type: str = "cuda"):
    r"""Return the dtype to be used for autocasting on :attr:`device_type`."""
    return _get_autocast_state(device_type).dtype


def set_autocast_enabled(device_type: str, enabled: bool) -> None:
    r"""Enable or disable autocast mode on :attr:`device_type`."""
    _get_autocast_state(device_type).enabled = enabled


def set_autocast_dtype(device_type: str, dtype) -> None:
    r"""Set the dtype to be used for autocasting on :attr:`device_type`."""
    _get_autocast_state(device_type).dtype = dtype


def get_autocast_gpu_dtype():
    r"""
    Return the dtype to be used for CUDA autocasting.

    .. warning::
        Kept for backward compatibility. Prefer :func:`get_autocast_dtype`.
    """
    return get_autocast_dtype("cuda")


def get_autocast_cpu_dtype():
    r"""
    Return the dtype to be used for CPU autocasting.

    .. warning::
        Kept for backward compatibility. Prefer :func:`get_autocast_dtype`.
    """
    return get_autocast_dtype("cpu")


def autocast_increment_nesting() -> int:
    r"""Increments the autocast nesting level and returns the new value."""
    nesting = _get_nesting() + 1
    _set_nesting(nesting)
    return nesting


def autocast_decrement_nesting() -> int:
    r"""Decrements the autocast nesting level and returns the new value."""
    nesting = _get_nesting() - 1
    _set_nesting(nesting)
    return nesting


def clear_autocast_cache() -> None:
    r"""Clear the autocast weight cache."""
    _cast_cache.clear()


def set_autocast_cache_enabled(flag: bool) -> None:
    r"""Set whether the autocast weight cache is enabled."""
    _set_cache_enabled(flag)


def is_autocast_cache_enabled() -> bool:
    r"""Return True if the autocast weight cache is enabled."""
    return _get_cache_enabled()


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    decorate_autocast.__script_unsupported = (  # type: ignore[attr-defined]
        "@autocast() decorator is not supported in script mode"
    )
    return decorate_autocast


def _is_cuda_available() -> bool:
    try:
        import tensorplay.cuda as cuda

        return cuda.is_available()
    except Exception:
        return False


def _is_cuda_bf16_supported() -> bool:
    try:
        import tensorplay.cuda as cuda

        if not cuda.is_available():
            return False
        major, _minor = cuda.get_device_capability()
        return major >= 8
    except Exception:
        return False


class autocast:
    r"""
    Instances of :class:`autocast` serve as context managers or decorators that
    allow regions of your script to run in mixed precision.

    In these regions, ops run in an op-specific dtype chosen by autocast
    to improve performance while maintaining accuracy.

    When entering an autocast-enabled region, Tensors may be any type.
    You should not call ``half()`` or ``bfloat16()`` on your model(s) or inputs when using autocasting.

    :class:`autocast` should wrap only the forward pass(es) of your network, including the loss
    computation(s).  Backward passes under autocast are not recommended.
    Backward ops run in the same type that autocast used for corresponding forward ops.

    Example for CUDA Devices::

        # Creates model and optimizer in default precision
        model = Net().cuda()
        optimizer = optim.SGD(model.parameters(), ...)

        for input, target in data:
            optimizer.zero_grad()

            # Enables autocasting for the forward pass (model + loss)
            with tensorplay.autocast(device_type="cuda"):
                output = model(input)
                loss = loss_fn(output, target)

            # Exits the context manager before backward()
            loss.backward()
            optimizer.step()

    :class:`autocast` can also be used as a decorator, e.g., on the ``forward`` method of your model::

        class AutocastModel(nn.Module):
            ...

            @tensorplay.autocast(device_type="cuda")
            def forward(self, input): ...

    Floating-point Tensors produced in an autocast-enabled region may be ``float16``.
    After returning to an autocast-disabled region, using them with floating-point
    Tensors of different dtypes may cause type mismatch errors.  If so, cast the Tensor(s)
    produced in the autocast region back to ``float32`` (or other dtype if desired).

    ``autocast(enabled=False)`` subregions can be nested in autocast-enabled regions.
    Locally disabling autocast can be useful, for example, if you want to force a subregion
    to run in a particular ``dtype``.

    The autocast state is thread-local.  If you want it enabled in a new thread, the context manager or decorator
    must be invoked in that thread.

    Args:
        device_type(str, required):  Device type to use. Possible values are: 'cuda' and 'cpu'.
                                     The type is the same as the `type` attribute of a :class:`tensorplay.device`.
                                     Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        enabled(bool, optional):  Whether autocasting should be enabled in the region.
            Default: ``True``
        dtype(tensorplay.dtype, optional):  Data type for ops run in autocast. It uses the default value
            (``tensorplay.float16`` for CUDA and ``tensorplay.bfloat16`` for CPU), given by
            :func:`~tensorplay.get_autocast_dtype`, if :attr:`dtype` is ``None``.
            Default: ``None``
        cache_enabled(bool, optional):  Whether the weight cache inside autocast should be enabled.
            Default: ``True``
    """

    def __init__(
        self,
        device_type: str,
        dtype: Any | None = None,
        enabled: bool = True,
        cache_enabled: bool | None = None,
    ):
        if not isinstance(device_type, str):
            raise ValueError(
                f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
            )
        self.fast_dtype = (
            get_autocast_dtype(device_type) if dtype is None else dtype
        )
        self.device = device_type
        if not is_autocast_available(self.device):
            raise RuntimeError(
                f"User specified an unsupported autocast device_type '{self.device}'"
            )

        device_supported_dtypes = [tensorplay.bfloat16, tensorplay.float16]

        self._cache_enabled = (
            is_autocast_cache_enabled()
            if cache_enabled is None
            else cache_enabled
        )

        device_name = self.device.upper()
        if enabled:
            # Special case for CUDA AMP and bfloat16 support
            if self.device == "cuda":
                if not _is_cuda_available():
                    warnings.warn(
                        "CUDA is not available. Disabling autocast.",
                        stacklevel=2,
                    )
                    enabled = False
                elif (
                    self.fast_dtype == tensorplay.bfloat16
                    and not _is_cuda_bf16_supported()
                ):
                    raise RuntimeError(
                        "Current CUDA Device does not support bfloat16. Please switch dtype to float16."
                    )
            elif self.fast_dtype not in device_supported_dtypes:
                error_message = (
                    f"In {device_name} autocast, but the target dtype is not supported. Disabling autocast.\n"
                    f"{device_name} Autocast only supports dtypes of "
                    + ", ".join(map(str, device_supported_dtypes))
                    + " currently."
                )
                warnings.warn(error_message, stacklevel=2)
                enabled = False
        self._enabled = enabled

    def __enter__(self):
        self.prev_cache_enabled = is_autocast_cache_enabled()
        self.prev = is_autocast_enabled(self.device)
        self.prev_fastdtype = get_autocast_dtype(self.device)
        set_autocast_enabled(self.device, self._enabled)
        set_autocast_dtype(self.device, self.fast_dtype)
        autocast_increment_nesting()
        set_autocast_cache_enabled(self._cache_enabled)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        # Drop the cache when we exit to a nesting level that's outside any instance of autocast.
        if autocast_decrement_nesting() == 0:
            clear_autocast_cache()
        set_autocast_enabled(self.device, self.prev)
        set_autocast_dtype(self.device, self.prev_fastdtype)
        set_autocast_cache_enabled(self.prev_cache_enabled)
        return False

    def __call__(self, func):
        if not callable(func):
            raise TypeError(
                f"autocast()(func) requires a callable, but got {type(func).__name__}. "
                f"Did you mean to use autocast as a context manager? For example:\n"
                f"    with tensorplay.autocast(device_type=...):\n"
                f"        output = model(input)"
            )
        return autocast_decorator(self, func)


# Subclass to distinguish autocast variables created by _enter_autocast (and not managed by a with statement)
class _UnmanagedAutocast(autocast):
    pass


# These functions aren't meant for public usage.
# They are what we trace into a graph during pre_dispatch tracing
# when we encounter an autocast context manager.
def _enter_autocast(*vals):
    mode = _UnmanagedAutocast(*vals)
    mode.__enter__()
    return mode


def _exit_autocast(mode):
    mode.__exit__(None, None, None)


# Casts Tensors and containers of Tensors.  Special-cases passthroughs for strings and np.ndarrays, which
# may be falsely detected as "Iterables."
def _cast(value, device_type: str, dtype):
    if isinstance(value, tensorplay.Tensor):
        is_eligible = (
            value.is_floating_point()
            and value.device.type == device_type
            and (value.dtype is not tensorplay.float64)
        )
        return value.to(dtype) if is_eligible else value
    elif isinstance(value, (str, bytes)):
        return value
    elif HAS_NUMPY and isinstance(
        value,
        np.ndarray,  # pyrefly: ignore [missing-attribute]
    ):
        return value
    elif isinstance(value, collections.abc.Mapping):
        return {
            _cast(k, device_type, dtype): _cast(v, device_type, dtype)
            for k, v in value.items()
        }
    elif isinstance(value, collections.abc.Iterable):
        iterable = (_cast(v, device_type, dtype) for v in value)
        if isinstance(value, (list, tuple)):
            return type(value)(iterable)
        else:
            return iterable
    else:
        return value


def custom_fwd(
    fwd=None,
    *,
    device_type: str,
    cast_inputs=None,
):
    """
    Create a helper decorator for ``forward`` methods of custom autograd functions.

    Autograd functions are subclasses of :class:`tensorplay.autograd.Function`.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu'.
            The type is the same as the `type` attribute of a :class:`tensorplay.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        cast_inputs (:class:`tensorplay.dtype` or None, optional, default=None):  If not ``None``,
            when ``forward`` runs in an autocast-enabled region, casts incoming
            floating-point Tensors to the target dtype (non-floating-point Tensors are not affected),
            then executes ``forward`` with autocast disabled.
            If ``None``, ``forward``'s internal ops execute with the current autocast state.

    .. note::
        If the decorated ``forward`` is called outside an autocast-enabled region,
        :func:`custom_fwd<custom_fwd>` is a no-op and ``cast_inputs`` has no effect.
    """
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )
    if fwd is None:
        return functools.partial(
            custom_fwd, device_type=device_type, cast_inputs=cast_inputs
        )

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        args[0]._dtype = get_autocast_dtype(device_type)
        if cast_inputs is None:
            args[0]._fwd_used_autocast = is_autocast_enabled(device_type)
            return fwd(*args, **kwargs)
        else:
            autocast_context = is_autocast_enabled(device_type)
            args[0]._fwd_used_autocast = False
            if autocast_context:
                with autocast(device_type=device_type, enabled=False):
                    return fwd(
                        *_cast(args, device_type, cast_inputs),
                        **_cast(kwargs, device_type, cast_inputs),
                    )
            else:
                return fwd(*args, **kwargs)

    return decorate_fwd


# Autograd ensures incoming gradients are the same type as forward outputs.  Allowing a separate
# cast_inputs argument on custom_bwd is unnecessary and could cause errors if it doesn't match
# cast_inputs supplied to custom_fwd.
def custom_bwd(bwd=None, *, device_type: str):
    """Create a helper decorator for backward methods of custom autograd functions.

    Autograd functions are subclasses of :class:`tensorplay.autograd.Function`.
    Ensures that ``backward`` executes with the same autocast state as ``forward``.

    Args:
        device_type(str):  Device type to use. 'cuda', 'cpu'.
            The type is the same as the `type` attribute of a :class:`tensorplay.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
    if not isinstance(device_type, str):
        raise ValueError(
            f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
        )
    if bwd is None:
        return functools.partial(custom_bwd, device_type=device_type)

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with autocast(
            device_type=device_type,
            enabled=args[0]._fwd_used_autocast,
            dtype=args[0]._dtype,
        ):
            return bwd(*args, **kwargs)

    return decorate_bwd


# ---------------------------------------------------------------------------
# Op-level autocast policies
#
# torch routes ops through AutocastCUDA/AutocastCPU dispatch keys implemented
# in C++.  TensorPlay applies the equivalent per-op policies by wrapping the
# bound operators at import time (_install_autocast).  The wrappers resolve
# the active autocast dtype from the thread-local state at call time.
# ---------------------------------------------------------------------------


class _CastFunction(tensorplay.autograd.Function):
    @staticmethod
    def forward(ctx, input, dtype):
        ctx.src_dtype = input.dtype
        return input.to(dtype)

    @staticmethod
    def backward(ctx, grad_output, *extra):
        return grad_output.to(ctx.src_dtype), None


def _cast(src, dst_type):
    if src.dtype == dst_type:
        return src
    if src.requires_grad:
        return _CastFunction.apply(src, dst_type)
    return src.to(dst_type)


def _cached_cast(src, dst_type):
    if not is_autocast_cache_enabled() or (src.requires_grad and not src.is_leaf):
        return _cast(src, dst_type)
    key = id(src)
    entry = _cast_cache.get(key)
    if entry is not None:
        ref, casted = entry
        if ref is None or ref() is src:
            return casted
    casted = _cast(src, dst_type)
    try:
        ref = weakref.ref(src)
    except TypeError:
        ref = None
    _cast_cache[key] = (ref, casted)
    return casted


def _is_eligible(arg, device_type, dst_type):
    if not isinstance(arg, tensorplay.Tensor):
        return False
    if arg.dtype not in _FLOAT_DTYPES:
        return False
    if arg.dtype == tensorplay.float64:
        return False
    if arg.dtype == dst_type:
        return False
    if arg.device.type != device_type:
        return False
    return True


def _cast_arg(arg, dst_type, device_type):
    if _is_eligible(arg, device_type, dst_type):
        return _cast(arg, dst_type)
    return arg


def _lower_precision_fp(device_type, func, *args, **kwargs):
    if is_autocast_enabled(device_type):
        fast_dtype = get_autocast_dtype(device_type)
        args = tuple(
            _cached_cast(arg, fast_dtype)
            if _is_eligible(arg, device_type, fast_dtype)
            else arg
            for arg in args
        )
        kwargs = {
            k: _cached_cast(v, fast_dtype)
            if _is_eligible(v, device_type, fast_dtype)
            else v
            for k, v in kwargs.items()
        }
    return func(*args, **kwargs)


def _fp32(device_type, func, *args, **kwargs):
    if is_autocast_enabled(device_type):
        args = tuple(_cast_arg(arg, tensorplay.float32, device_type) for arg in args)
        kwargs = {
            k: _cast_arg(v, tensorplay.float32, device_type)
            for k, v in kwargs.items()
        }
    return func(*args, **kwargs)


def _fp32_set_opt_dtype(device_type, func, *args, **kwargs):
    if is_autocast_enabled(device_type):
        args = tuple(_cast_arg(arg, tensorplay.float32, device_type) for arg in args)
        kwargs = {
            k: _cast_arg(v, tensorplay.float32, device_type)
            for k, v in kwargs.items()
        }
    if "dtype" in kwargs and (
        kwargs["dtype"] is None or kwargs["dtype"] == tensorplay.undefined
    ):
        kwargs["dtype"] = tensorplay.float32
    return func(*args, **kwargs)


def _promote(device_type, func, *args, **kwargs):
    if is_autocast_enabled(device_type):
        tensors = [
            a
            for a in list(args) + list(kwargs.values())
            if isinstance(a, tensorplay.Tensor)
            and a.dtype in _FLOAT_DTYPES
            and a.device.type == device_type
        ]
        if tensors:
            if any(
                t.dtype in (tensorplay.float32, tensorplay.float64) for t in tensors
            ):
                target = tensorplay.float32
            else:
                target = tensors[0].dtype
            args = tuple(_cast_arg(arg, target, device_type) for arg in args)
            kwargs = {
                k: _cast_arg(v, target, device_type) for k, v in kwargs.items()
            }
    return func(*args, **kwargs)


_LOWER_PRECISION_FP_MODULE_OPS = [
    "mm",
    "matmul",
    "addmm",
    "bmm",
    "baddbmm",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose2d",
    "conv_transpose3d",
]

_FP32_MODULE_OPS = [
    "exp",
    "log",
    "pow",
    "rsqrt",
    "acos",
    "asin",
    "cosh",
    "sinh",
    "tan",
    "layer_norm",
    "group_norm",
    "nll_loss",
    "mse_loss",
]

_FP32_SET_OPT_DTYPE_MODULE_OPS = ["softmax", "log_softmax"]

_PROMOTE_MODULE_OPS = ["atan2"]

_LOWER_PRECISION_FP_METHOD_OPS = ["mm", "matmul", "bmm"]

_FP32_METHOD_OPS = [
    "exp",
    "log",
    "pow",
    "rsqrt",
    "acos",
    "asin",
    "cosh",
    "sinh",
    "tan",
]

_FP32_SET_OPT_DTYPE_METHOD_OPS = ["softmax", "log_softmax", "sum", "prod"]

_PROMOTE_METHOD_OPS = ["atan2"]


def _infer_device_type(args, kwargs, default="cpu"):
    for a in args:
        if isinstance(a, tensorplay.Tensor):
            return a.device.type
    for v in kwargs.values():
        if isinstance(v, tensorplay.Tensor):
            return v.device.type
    return default


def _wrap_module_func(func, policy):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        device_type = _infer_device_type(args, kwargs)
        return policy(device_type, func, *args, **kwargs)

    wrapped._tp_autocast_wrapped = True
    return wrapped


def _wrap_method(method, policy):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        device_type = self.device.type
        return policy(device_type, method, self, *args, **kwargs)

    wrapped._tp_autocast_wrapped = True
    return wrapped


def _patch(original, name, wrapped):
    setattr(original, name, wrapped)


def _install_autocast():
    import tensorplay.nn.functional as _nnf

    if hasattr(_nnf, "linear") and not getattr(
        getattr(_nnf, "linear"), "_tp_autocast_wrapped", False
    ):
        _patch(_nnf, "linear", _wrap_module_func(_nnf.linear, _lower_precision_fp))

    for name in _LOWER_PRECISION_FP_MODULE_OPS:
        fn = getattr(_C, name, None)
        if fn is not None and not getattr(fn, "_tp_autocast_wrapped", False):
            _patch(_C, name, _wrap_module_func(fn, _lower_precision_fp))
    for name in _FP32_MODULE_OPS:
        fn = getattr(_C, name, None)
        if fn is not None and not getattr(fn, "_tp_autocast_wrapped", False):
            _patch(_C, name, _wrap_module_func(fn, _fp32))
    for name in _FP32_SET_OPT_DTYPE_MODULE_OPS:
        fn = getattr(_C, name, None)
        if fn is not None and not getattr(fn, "_tp_autocast_wrapped", False):
            _patch(_C, name, _wrap_module_func(fn, _fp32_set_opt_dtype))
    for name in _PROMOTE_MODULE_OPS:
        fn = getattr(_C, name, None)
        if fn is not None and not getattr(fn, "_tp_autocast_wrapped", False):
            _patch(_C, name, _wrap_module_func(fn, _promote))
    for name in _LOWER_PRECISION_FP_METHOD_OPS:
        m = getattr(_C.TensorBase, name, None)
        if m is not None and not getattr(m, "_tp_autocast_wrapped", False):
            _patch(_C.TensorBase, name, _wrap_method(m, _lower_precision_fp))
    for name in _FP32_METHOD_OPS:
        m = getattr(_C.TensorBase, name, None)
        if m is not None and not getattr(m, "_tp_autocast_wrapped", False):
            _patch(_C.TensorBase, name, _wrap_method(m, _fp32))
    for name in _FP32_SET_OPT_DTYPE_METHOD_OPS:
        m = getattr(_C.TensorBase, name, None)
        if m is not None and not getattr(m, "_tp_autocast_wrapped", False):
            _patch(_C.TensorBase, name, _wrap_method(m, _fp32_set_opt_dtype))
    for name in _PROMOTE_METHOD_OPS:
        m = getattr(_C.TensorBase, name, None)
        if m is not None and not getattr(m, "_tp_autocast_wrapped", False):
            _patch(_C.TensorBase, name, _wrap_method(m, _promote))
