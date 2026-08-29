"""Device-type test infrastructure.

Test methods declare a ``device`` keyword argument (and optionally ``dtype``);
:func:`instantiate_device_type_tests` then generates one concrete test method
per device (and per dtype), so both the standard unittest runner and pytest
pick them up:

    class TestFoo(TestCase):
        def test_add(self, device):
            x = tp.rand(4, device=device)

        @dtypes(tp.float32, tp.int64)
        def test_fill(self, device, dtype):
            ...

    instantiate_device_type_tests(TestFoo, globals())
"""

import inspect
import os
import unittest
from functools import wraps

import tensorplay as tp

__all__ = [
    "deviceCountAtLeast",
    "onlyCUDA",
    "onlyCPU",
    "onlyNativeDeviceTypes",
    "onlyOn",
    "skipIf",
    "skipCPUIf",
    "skipCUDAIf",
    "skipCPUIfNoLapack",
    "skipGPUIf",
    "skipMeta",
    "expectedFailure",
    "expectedFailureCPU",
    "expectedFailureCUDA",
    "precisionOverride",
    "toleranceOverride",
    "largeTensorTest",
    "dtypes",
    "dtypesIfCPU",
    "dtypesIfCUDA",
    "dtype_name",
    "get_all_dtypes",
    "get_all_device_types",
    "instantiate_device_type_tests",
]


def dtype_name(dtype) -> str:
    """Returns the pretty name of the dtype (e.g. ``int64``)."""
    return str(dtype).split(".")[-1]


def get_all_device_types() -> list[str]:
    """Returns all device types the suite can run on."""
    return list(_available_device_types)


def get_all_dtypes(*args, **kwargs):
    """See :mod:`tensorplay.testing._internal.common_dtype` for the canonical implementation."""
    from tensorplay.testing._internal.common_dtype import get_all_dtypes as fn

    return fn(*args, **kwargs)

# Device types the suite can cover. "cpu" is always present; "cuda" is added
# when a device is visible at import time.
_available_device_types = ["cpu"]
if tp.cuda.is_available():
    _available_device_types.append("cuda")


def deviceCountAtLeast(count, devices):
    """Skips the test unless at least ``count`` of the given device types exist."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            available = sum(
                1 if d == "cpu" else tp.cuda.device_count() for d in devices
            )
            if available < count:
                raise unittest.SkipTest(
                    f"requires {count} of {devices}, found {available}"
                )
            return fn(self, *args, **kwargs)

        wrapper._device_count_at_least = (count, devices)
        return wrapper

    return decorator


def _device_filter_decorator(device_types):
    def decorator(fn):
        fn._device_types = set(device_types)
        return fn

    return decorator


def onlyCPU(fn=None):
    """Restricts the test to CPU devices."""
    if fn is None:
        return lambda f: _device_filter_decorator(("cpu",))(f)
    return _device_filter_decorator(("cpu",))(fn)


def onlyCUDA(fn=None):
    """Restricts the test to CUDA devices."""
    if fn is None:
        return lambda f: _device_filter_decorator(("cuda",))(f)
    return _device_filter_decorator(("cuda",))(fn)


def onlyNativeDeviceTypes(fn):
    """Restricts the test to non-CPU devices (no-op when none exist)."""
    return _device_filter_decorator(("cuda",))(fn)


def onlyOn(*device_types):
    """Restricts the test to the given device types."""
    return _device_filter_decorator(device_types)


def _skip_decorator(device_type, condition, reason):
    def decorator(fn):
        checks = getattr(fn, "_skip_checks", [])
        checks = list(checks) + [(device_type, condition, reason)]
        fn._skip_checks = checks
        return fn

    return decorator


def skipIf(condition, reason):
    """Skips the test on every device when ``condition`` is true."""
    return _skip_decorator(None, condition, reason)


def skipCPUIf(condition, reason):
    """Skips the test on CPU when ``condition`` is true."""
    return _skip_decorator("cpu", condition, reason)


def skipCUDAIf(condition, reason):
    """Skips the test on CUDA when ``condition`` is true."""
    return _skip_decorator("cuda", condition, reason)


def skipGPUIf(condition, reason):
    """Skips the test on every GPU device when ``condition`` is true."""
    return _skip_decorator("gpu", condition, reason)


def skipMeta(fn):
    """Skips the (non-executing) meta-device variants of the test."""
    return fn


def skipCPUIfNoLapack(fn):
    """Skips the test on CPU when Lapack support is unavailable."""
    def condition():
        try:
            tp.linalg.cholesky(tp.eye(2))
            return False
        except Exception:
            return True

    return _skip_decorator("cpu", condition, "compiled without Lapack")(fn)


def _expected_failure_decorator(device_type, reason=None):
    def decorator(fn):
        marks = getattr(fn, "_expected_failures", {})
        marks = dict(marks)
        marks[device_type] = reason
        fn._expected_failures = marks
        return fn

    return decorator


def expectedFailure(fn=None):
    """Marks the test as an expected failure on every device."""
    if fn is None:
        return lambda f: _expected_failure_decorator(None)(f)
    return _expected_failure_decorator(None)(fn)


def expectedFailureCPU(fn):
    """Marks the test as an expected failure on CPU."""
    return _expected_failure_decorator("cpu")(fn)


def expectedFailureCUDA(fn):
    """Marks the test as an expected failure on CUDA."""
    return _expected_failure_decorator("cuda")(fn)


def precisionOverride(prec):
    """Raises the minimum ``atol`` used by all comparisons of the test class."""
    def decorator(cls_or_fn):
        cls_or_fn._precision = prec
        return cls_or_fn

    return decorator


def toleranceOverride(rtol, atol):
    """Raises the minimum ``rtol`` and ``atol`` used by all comparisons."""
    def decorator(cls_or_fn):
        cls_or_fn._rel_tol = rtol
        cls_or_fn._precision = atol
        return cls_or_fn

    return decorator


def largeTensorTest(size, device=None):
    """Skips the test when the requested tensor size exceeds the budget.

    ``size`` may be an integer element count or a callable evaluated with the
    current device. The budget is controlled by ``TP_LARGE_TENSOR_SIZE``
    (elements; default 2**26).
    """
    def check(device_type):
        threshold = int(os.environ.get("TP_LARGE_TENSOR_SIZE", 2**26))
        try:
            needed = size(device_type) if callable(size) else size
        except Exception:
            return False
        return needed > threshold

    def decorator(fn):
        checks = getattr(fn, "_skip_checks", [])
        checks = list(checks) + [(None, check, "tensor too large")]
        fn._skip_checks = checks
        return fn

    return decorator


def _dtypes_decorator(device_type, dtypes):
    def decorator(fn):
        per_device = getattr(fn, "_dtypes", {})
        existing = per_device.get(device_type, ())
        per_device[device_type] = tuple(existing) + tuple(dtypes)
        fn._dtypes = per_device
        return fn

    return decorator


def dtypes(*dtype_args):
    """Parametrizes the test over the given dtypes for every device."""
    return _dtypes_decorator(None, dtype_args)


def dtypesIfCPU(*dtype_args):
    """Parametrizes the test over the given dtypes for CPU only."""
    return _dtypes_decorator("cpu", dtype_args)


def dtypesIfCUDA(*dtype_args):
    """Parametrizes the test over the given dtypes for CUDA only."""
    return _dtypes_decorator("cuda", dtype_args)


def _resolve_dtypes(fn, device_type):
    per_device = getattr(fn, "_dtypes", None)
    if per_device is None:
        return (None,)
    entries = []
    entries.extend(per_device.get(None, ()))
    entries.extend(per_device.get(device_type, ()))
    return tuple(entries) if entries else (None,)


def _format_dtype_suffix(dtype) -> str:
    name = str(dtype).rsplit(".", 1)[-1]
    return name


def _make_device_test_wrapper(cls, fn, device_type, dtype):
    """Creates a concrete test method bound to one device (and one dtype)."""
    device_str = device_type
    suffix = device_type
    kwargs = {"device": device_str}
    if dtype is not None:
        suffix += f"_{_format_dtype_suffix(dtype)}"
        kwargs["dtype"] = dtype

    @wraps(fn)
    def wrapper(self):
        for check_device, condition, reason in getattr(fn, "_skip_checks", []):
            if (check_device is None or check_device == device_type
                    or (check_device == "gpu" and device_type != "cpu")) and condition:
                raise unittest.SkipTest(reason)
        return fn(self, **kwargs)

    expected = getattr(fn, "_expected_failures", {})
    if None in expected or device_type in expected:
        wrapper = unittest.expectedFailure(wrapper)

    wrapper.__name__ = f"{fn.__name__}_{suffix}"
    wrapper.__doc__ = f"{fn.__name__} on {device_str}" + (
        f" with dtype {dtype}" if dtype is not None else ""
    )
    wrapper._original_test = fn
    return wrapper


def instantiate_device_type_tests(cls, globals_dict, except_for=(), only_for=None):
    """Generates per-device (and per-dtype) test methods for ``cls``.

    Scans the class for methods named ``test_*`` that accept a ``device``
    keyword argument, replaces them with concrete methods named
    ``test_<name>_<device>[_<dtype>]``, and registers the class under its
    original name in ``globals_dict``.

    Args:
        cls: The test class to instantiate.
        globals_dict: The caller's module globals; used to register the class
            so runners can discover the generated methods.
        except_for: Device types to exclude from instantiation.
        only_for: If given, only these device types are instantiated.
    """
    # Environment overrides, matching the standard runner conventions.
    env_only_for = os.environ.get("TP_TESTING_DEVICE_ONLY_FOR", "")
    env_except_for = os.environ.get("TP_TESTING_DEVICE_EXCEPT_FOR", "")
    if env_only_for:
        only_for = set(env_only_for.split(","))
    if env_except_for:
        except_for = tuple(set(list(except_for or []) + env_except_for.split(",")))

    device_types = [
        d
        for d in _available_device_types
        if d not in except_for and (only_for is None or d in only_for)
    ]

    for name in list(vars(cls)):
        if not name.startswith("test_"):
            continue
        fn = vars(cls)[name]
        if not inspect.isfunction(fn):
            continue
        params = inspect.signature(fn).parameters
        if "device" not in params:
            continue

        delattr(cls, name)
        allowed = getattr(fn, "_device_types", None)
        for device_type in device_types:
            if allowed is not None and device_type not in allowed:
                continue
            for dtype in _resolve_dtypes(fn, device_type):
                wrapper = _make_device_test_wrapper(cls, fn, device_type, dtype)
                setattr(cls, wrapper.__name__, wrapper)

    globals_dict[cls.__name__] = cls
    return cls
