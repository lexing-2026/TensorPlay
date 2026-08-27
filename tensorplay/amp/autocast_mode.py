# mypy: allow-untyped-defs
import collections
import functools
import warnings
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

# The autocast state lives in the dispatcher (tensorplay._C), mirroring
# torch, where is_autocast_enabled/get_autocast_dtype/... are C++ bindings
# into at::autocast.  Re-export them here under the same names.
is_autocast_enabled = _C.is_autocast_enabled
get_autocast_dtype = _C.get_autocast_dtype
set_autocast_enabled = _C.set_autocast_enabled
set_autocast_dtype = _C.set_autocast_dtype
autocast_increment_nesting = _C.autocast_increment_nesting
autocast_decrement_nesting = _C.autocast_decrement_nesting
clear_autocast_cache = _C.clear_autocast_cache
is_autocast_cache_enabled = _C.is_autocast_cache_enabled
set_autocast_cache_enabled = _C.set_autocast_cache_enabled

# Fused enter/exit (single binding per context transition); absent in older
# compiled extensions, where we fall back to the granular calls.
_fused_enter = getattr(_C, "_autocast_enter", None)
_fused_exit = getattr(_C, "_autocast_exit", None)


def get_autocast_gpu_dtype():
    r"""
    Return the dtype to be used for CUDA autocasting.

    .. warning::
        Kept for backward compatibility. Prefer :func:`get_autocast_dtype`.
    """
    return _C.get_autocast_gpu_dtype()


def get_autocast_cpu_dtype():
    r"""
    Return the dtype to be used for CPU autocasting.

    .. warning::
        Kept for backward compatibility. Prefer :func:`get_autocast_dtype`.
    """
    return _C.get_autocast_cpu_dtype()


def is_autocast_available(device_type: str) -> bool:
    r"""
    Return a bool indicating if autocast is available on :attr:`device_type`.

    Args:
        device_type(str):  Device type to use. Possible values are: 'cuda', 'cpu'.
            The type is the same as the `type` attribute of a :class:`tensorplay.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
    """
    return _C._is_autocast_available(device_type)


def autocast_decorator(autocast_instance, func):
    @functools.wraps(func)
    def decorate_autocast(*args, **kwargs):
        with autocast_instance:
            return func(*args, **kwargs)

    decorate_autocast.__script_unsupported = (  # type: ignore[attr-defined]
        "@autocast() decorator is not supported in script mode"
    )
    return decorate_autocast


def _amp_definitely_not_available() -> bool:
    try:
        import tensorplay.cuda as cuda

        return not cuda.is_available()
    except Exception:
        return True


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
        # Upstream checks availability BEFORE resolving the default dtype so
        # unsupported devices raise RuntimeError (not the binding's ValueError).
        self.device = device_type
        if not is_autocast_available(self.device):
            raise RuntimeError(
                f"User specified an unsupported autocast device_type '{self.device}'"
            )
        self.fast_dtype = (
            get_autocast_dtype(device_type) if dtype is None else dtype
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
                if _amp_definitely_not_available():
                    warnings.warn(
                        "CUDA is not available or tensorplay_xla is imported. Disabling autocast.",
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
        if _fused_enter is not None:
            self._prev = _fused_enter(
                self.device, self.fast_dtype, self._enabled, self._cache_enabled
            )
            return self
        self.prev_cache_enabled = is_autocast_cache_enabled()
        self.prev = is_autocast_enabled(self.device)
        self.prev_fastdtype = get_autocast_dtype(self.device)
        set_autocast_enabled(self.device, self._enabled)
        set_autocast_dtype(self.device, self.fast_dtype)
        autocast_increment_nesting()
        set_autocast_cache_enabled(self._cache_enabled)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any):  # type: ignore[override]
        if _fused_exit is not None:
            _fused_exit(self.device, self._prev)
            return False
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
