# mypy: allow-untyped-defs
r"""CUDA namespace and runtime helpers.

This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize
GPUs for computation.

It is lazily initialized, so you can always import it, and use
:func:`is_available()` to determine if your system supports CUDA.


- legacy per-dtype ``Storage``/``Tensor`` classes are not provided
  (tensorplay has no typed storages);
- ``cudart()`` returns ``None`` (no ctypes runtime binding is exposed);
- RNG state introspection and graph capture are provided by native bindings.
"""

import os
import platform
import threading
import traceback
import warnings
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Optional

import tensorplay
import tensorplay._C
from tensorplay._C import Device
from tensorplay._C import _cuda as _lcuda

from . import gds as gds
from ._utils import _dummy_type, _LazySeedTracker, classproperty, _get_device_index
from .gds import GdsFile, is_gds_available  # noqa: F401
from .graphs import (
    CUDAGraph,
    export_dot,
    graph,
    graph_pool_handle,
    is_current_stream_capturing,
    make_graphed_callables,
)
from .green_contexts import GreenContext
from .streams import Event, ExternalStream, Stream


try:
    _cudart = getattr(tensorplay._C, "_cudart", None)
except ImportError:
    _cudart = None


class version:

    hip: str | None = None
    cuda: str | None = None


try:
    _ver_int = _lcuda.get_version()
    version.cuda = f"{_ver_int // 1000}.{(_ver_int % 1000) // 10}"
except Exception:
    version.cuda = None


_initialized = False
_tls = threading.local()
_initialization_lock = threading.Lock()
_queued_calls: list[
    tuple[Callable[[], None], list[str]]
] = []  # don't invoke these until initialization occurs
_is_in_bad_fork = lambda: False  # noqa: E731

_HAS_PYNVML = False
_PYNVML_ERR = None
try:
    import pynvml  # type: ignore[import]

    _HAS_PYNVML = True
except ModuleNotFoundError:
    pass
except ImportError as err:
    _PYNVML_ERR = err  # sometimes a lib is installed but the import fails for some other reason, so we log the error for later

_lazy_seed_tracker = _LazySeedTracker()

# Define dummy _CudaDeviceProperties type if TensorPlay was compiled without CUDA
if hasattr(_lcuda, "_CudaDeviceProperties"):
    _CudaDeviceProperties = _lcuda._CudaDeviceProperties
else:
    _CudaDeviceProperties = _dummy_type("_CudaDeviceProperties")


def _exchange_device(device: int) -> int:
    if device < 0:
        return -1
    prev = current_device()
    set_device(device)
    return prev


def _maybe_exchange_device(device: int) -> int:
    if device < 0:
        return -1
    prev = current_device()
    set_device(device)
    return prev


has_half: bool = True
has_magma: bool = False

default_generators: tuple = ()


def _is_compiled() -> bool:
    r"""Return true if compiled with CUDA support."""
    return hasattr(_lcuda, "_CudaStream")


def _nvml_based_avail() -> bool:
    return os.getenv("TENSORPLAY_NVML_BASED_CUDA_CHECK") == "1"


def is_available() -> bool:
    r"""
    Return a bool indicating if CUDA is currently available.

    .. note:: This function will NOT poison fork if the environment variable
        ``TENSORPLAY_NVML_BASED_CUDA_CHECK=1`` is set.
    """
    if not _is_compiled():
        return False
    if _nvml_based_avail():
        # The user has set an env variable to request this availability check that attempts to avoid fork poisoning by
        # using NVML at the cost of a weaker CUDA availability assessment. Note that if NVML discovery/initialization
        # fails, this assessment falls back to the default CUDA Runtime API assessment (`cudaGetDeviceCount`)
        return device_count() > 0
    else:
        # The default availability inspection never throws and returns 0 if the driver is missing or can't
        # be initialized. This uses the CUDA Runtime API `cudaGetDeviceCount` which in turn initializes the CUDA Driver
        # API via `cuInit`
        return _lcuda.is_available()


def is_bf16_supported(including_emulation: bool = True):
    r"""Return a bool indicating if the current CUDA device supports dtype bfloat16."""
    # If CUDA is not available, then it does not support bf16 either
    if not is_available():
        return False

    device_idx = current_device()

    if get_device_properties(device_idx).major >= 8:
        return True

    if not including_emulation:
        return False

    # Finally try to create a bfloat16 device tensor.
    return _check_bf16_tensor_supported(device_idx)


@lru_cache(maxsize=16)
def _check_bf16_tensor_supported(device_idx: int):
    try:
        tensorplay.tensor([1.0], dtype=tensorplay.bfloat16).to(
            Device(tensorplay.DeviceType.CUDA, device_idx)
        )
        return True
    except Exception:
        return False


def is_tf32_supported() -> bool:
    r"""Return a bool indicating if the current CUDA device supports dtype tf32."""
    # tf32 is supported on CUDA platforms that natively (i.e. no emulation)
    # support bfloat16.
    return is_bf16_supported(including_emulation=False)


def _sleep(cycles):
    _lcuda._sleep(cycles)


def _extract_arch_version(arch_string: str) -> int:
    """Extracts the architecture string from a CUDA version"""
    base = arch_string.split("_", maxsplit=2)[1]
    base = base.removesuffix("a").removesuffix("f")
    return int(base)


class _CompatInterval:
    """
    Defines a range of compute capabilities starting at a given
    version and going up to the end of that major version. This
    also allows excluding specific versions from the range.
    """

    def __init__(self, start, exclude: set[int] | None = None):
        self.major, self.minor = start // 10, start % 10
        self.exclude = set() if exclude is None else exclude

    def __contains__(self, x):
        if x in self.exclude:
            return False
        x_major, x_minor = x // 10, x % 10
        return x_major == self.major and x_minor >= self.minor

    def __str__(self):
        result = f">={self.major}.{self.minor},<{self.major + 1}.0"
        if len(self.exclude) > 0:
            exceptions = ", ".join(f"{x // 10}.{x % 10}" for x in self.exclude)
            result += f" except {{{exceptions}}}"
        return result


class _CompatSet:
    """
    A set of compute capabilities. It exists primarily to support custom
    printing logic and is otherwise equivalent to a plain python set().
    """

    def __init__(self, values: set[int]):
        self.values = values

    def __contains__(self, x):
        return x in self.values

    def __str__(self):
        return "{" + ", ".join(f"{v // 10}.{v % 10}" for v in self.values) + "}"


# (code SM)->(device SM required to execute the code)
DEVICE_REQUIREMENT: dict[int, _CompatSet | _CompatInterval] = {
    50: _CompatInterval(start=50, exclude={53}),
    52: _CompatInterval(start=52, exclude={53}),
    53: _CompatSet({53}),
    60: _CompatInterval(start=60, exclude={62}),
    61: _CompatInterval(start=61, exclude={62}),
    62: _CompatSet({62}),
    70: _CompatInterval(start=70, exclude={72}),
    72: _CompatSet({72}),
    75: _CompatInterval(start=75),
    80: _CompatInterval(start=80, exclude={87}),
    86: _CompatInterval(start=86, exclude={87}),
    87: _CompatSet({87}),
    89: _CompatInterval(start=89),
    90: _CompatInterval(start=90),
    100: _CompatInterval(start=100, exclude={101}),
    101: _CompatSet({101, 110}),  # 101 was renamed to 110
    103: _CompatInterval(start=103),
    110: _CompatSet({101, 110}),  # 101 was renamed to 110
    120: _CompatInterval(start=120),
    121: _CompatInterval(start=121),
}


def _code_compatible_with_device(device_cc: int, code_cc: int):
    compatible_devices = DEVICE_REQUIREMENT.get(code_cc)
    if compatible_devices is None:
        warnings.warn(
            f"TensorPlay was compiled with an unknown compute capability {code_cc // 10}.{code_cc % 10}.",
            stacklevel=2,
        )
        return device_cc in _CompatInterval(start=code_cc)
    return device_cc in compatible_devices


def _check_cubins():
    incompatible_device_warn = """
{} with CUDA capability sm_{} is not compatible with the current TensorPlay installation.
The current TensorPlay install supports CUDA capabilities {}.
If you want to use the {} GPU with TensorPlay, please check the instructions on how to rebuild with support for your GPU.
"""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return
    supported_sm = [_extract_arch_version(arch) for arch in arch_list if "sm_" in arch]
    for idx in range(device_count()):
        cap_major, cap_minor = get_device_capability(idx)
        # NVIDIA GPU compute architectures are backward compatible within major version
        supported = any(sm // 10 == cap_major for sm in supported_sm)
        if not supported:
            device_name = get_device_name(idx)
            capability = cap_major * 10 + cap_minor
            warnings.warn(
                incompatible_device_warn.format(
                    device_name, capability, " ".join(arch_list), device_name
                ),
                stacklevel=2,
            )


def is_initialized():
    r"""Return whether TensorPlay's CUDA state has been initialized."""
    return _initialized and not _is_in_bad_fork()


def _lazy_call(callable, **kwargs):
    with _initialization_lock:
        if is_initialized():
            callable()
        else:
            global _lazy_seed_tracker
            if kwargs.get("seed_all", False):
                _lazy_seed_tracker.queue_seed_all(callable, traceback.format_stack())
            elif kwargs.get("seed", False):
                _lazy_seed_tracker.queue_seed(callable, traceback.format_stack())
            else:
                # Don't store the actual traceback to avoid memory cycle
                _queued_calls.append((callable, traceback.format_stack()))


class DeferredCudaCallError(Exception):
    pass


try:
    AcceleratorError = tensorplay._C.AcceleratorError
except AttributeError:

    class AcceleratorError(RuntimeError):  # type: ignore[no-redef]
        pass

try:
    OutOfMemoryError = tensorplay._C.OutOfMemoryError
except AttributeError:

    class OutOfMemoryError(RuntimeError):  # type: ignore[no-redef]
        pass


def init():
    r"""Initialize TensorPlay's CUDA state.

    You may need to call this explicitly if you are interacting with
    TensorPlay via its C API, as Python bindings for CUDA functionality
    will not be available until this initialization takes place.
    Ordinary users should not need this, as all of TensorPlay's CUDA methods
    automatically initialize CUDA state on-demand.

    Does nothing if the CUDA state is already initialized.
    """
    _lazy_init()


def _lazy_init():
    global _initialized, _queued_calls
    if is_initialized() or hasattr(_tls, "is_initializing"):
        return
    with _initialization_lock:
        # We be double-checked locking, boys!  This is OK because
        # the above test was GIL protected anyway.  The inner test
        # is for when a thread blocked on some other thread which was
        # doing the initialization; when they get the lock, they will
        # find there is nothing left to do.
        if is_initialized():
            return
        # It is important to prevent other threads from entering _lazy_init
        # immediately, while we are still guaranteed to have the GIL, because some
        # of the C calls we make below will release the GIL
        if not _is_compiled():
            raise AssertionError("TensorPlay not compiled with CUDA enabled")
        if _is_in_bad_fork():
            raise RuntimeError(
                "Cannot re-initialize CUDA in forked subprocess. To use CUDA with "
                "multiprocessing, you must use the 'spawn' start method"
            )
        # This function throws if there's a driver initialization error, no GPUs
        # are found or any other error occurs. The native runtime initializes
        _lcuda.current_device()
        _tls.is_initializing = True

        _queued_calls.extend(calls for calls in _lazy_seed_tracker.get_calls() if calls)

        try:
            for queued_call, orig_traceback in _queued_calls:
                try:
                    queued_call()
                except Exception as e:
                    msg = (
                        f"CUDA call failed lazily at initialization with error: {str(e)}\n\n"
                        f"CUDA call was originally invoked at:\n\n{''.join(orig_traceback)}"
                    )
                    raise DeferredCudaCallError(msg) from e
        finally:
            delattr(_tls, "is_initializing")
        _initialized = True


def cudart():
    r"""Retrieves the CUDA runtime API module.

    This function initializes the CUDA runtime environment if it is not already
    initialized and returns the CUDA runtime API module (_cudart).

    Returns:
        module or None: The CUDA runtime API module, or ``None`` when no
        ctypes runtime binding is exposed by this build.
    """
    _lazy_init()
    return _cudart


class cudaStatus:
    SUCCESS: int = 0
    ERROR_NOT_READY: int = 34


class CudaError(RuntimeError):
    def __init__(self, code: int) -> None:
        msg = "CUDA error"
        if _cudart is not None:
            msg = _cudart.cudaGetErrorString(_cudart.cudaError(code))
        super().__init__(f"{msg} ({code})")


def check_error(res: int) -> None:
    r"""Raise an error if the result of a CUDA runtime API call is not success."""
    if res != cudaStatus.SUCCESS:
        raise CudaError(res)


def _require_native(name: str):
    raise RuntimeError(
        f"tensorplay was built without {name} support "
        f"(missing native binding `_cuda.{name}`)"
    )


class _DeviceGuard:
    def __init__(self, index: int):
        self.idx = index
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = _maybe_exchange_device(self.prev_idx)
        return False


class device:
    r"""Context-manager that changes the selected device.

    Args:
        device (tensorplay.Device or int): device index to select. It's a no-op if
            this argument is a negative integer or ``None``.
    """

    def __init__(self, device: Any):
        self.idx = _get_device_index(device, optional=True)
        self.prev_idx = -1

    def __enter__(self):
        self.prev_idx = _exchange_device(self.idx)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        self.idx = _maybe_exchange_device(self.prev_idx)
        return False


class device_of(device):
    r"""Context-manager that changes the current device to that of given object.

    You can use both tensors and storages as arguments. If a given object is
    not allocated on a GPU, this is a no-op.

    Args:
        obj (Tensor or Storage): object allocated on the selected device.
    """

    def __init__(self, obj):
        idx = obj.get_device() if obj.is_cuda else -1
        super().__init__(idx)


def set_device(device: Any) -> None:
    r"""Set the current device.

    Usage of this function is discouraged in favor of :any:`device`. In most
    cases it's better to use ``CUDA_VISIBLE_DEVICES`` environmental variable.

    Args:
        device (tensorplay.Device or int): selected device. This function is a no-op
            if this argument is negative.
    """
    device = _get_device_index(device)
    if device >= 0:
        _lcuda.set_device(device)


def get_device_name(device: Any = None) -> str:
    r"""Get the name of a device.

    Args:
        device (tensorplay.Device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~tensorplay.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


def get_device_capability(device: Any = None) -> tuple[int, int]:
    r"""Get the cuda capability of a device.

    Args:
        device (tensorplay.Device or int or str, optional): device for which to return the
            device capability. This function is a no-op if this argument is
            a negative integer. It uses the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        tuple(int, int): the major and minor cuda capability of the device
    """
    prop = get_device_properties(device)
    return prop.major, prop.minor


def get_device_properties(device: Any = None) -> "_CudaDeviceProperties":
    r"""Get the properties of a device.

    Args:
        device (tensorplay.Device or int or str, optional): device for which to return the
            properties of the device.  It uses the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is ``None``
            (default).

    Returns:
        _CudaDeviceProperties: the properties of the device
    """
    _lazy_init()
    device = _get_device_index(device, optional=True)
    if device < 0 or device >= device_count():
        raise AssertionError("Invalid device id")
    return _lcuda.get_device_properties(device)


def can_device_access_peer(device: Any, peer_device: Any) -> bool:
    r"""Check if peer access between two devices is possible."""
    _require_native("canDeviceAccessPeer")


class StreamContext:
    r"""Context-manager that selects a given stream.

    All CUDA kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        Stream (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note:: Streams are per-device.
    """

    cur_stream: Optional["Stream"]

    def __init__(self, stream: Optional["Stream"]):
        self.stream = stream
        self.idx = _get_device_index(None, True)

    def __enter__(self):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # Return if stream is None or CUDA device not available
        if cur_stream is None or self.idx == -1:
            return
        self.src_prev_stream = current_stream(None)

        # If the stream is not on the current device, then
        # set the current stream on the device
        if self.src_prev_stream.device != cur_stream.device:
            with device(cur_stream.device):
                self.dst_prev_stream = current_stream(cur_stream.device)
        set_stream(cur_stream)

    def __exit__(self, type: Any, value: Any, traceback: Any):
        # Local cur_stream variable for type refinement
        cur_stream = self.stream
        # If stream is None or no CUDA device available, return
        if cur_stream is None or self.idx == -1:
            return

        # Reset the stream on the original device
        # and destination device
        if self.src_prev_stream.device != cur_stream.device:  # type: ignore[union-attr]
            set_stream(self.dst_prev_stream)  # type: ignore[arg-type]
        set_stream(self.src_prev_stream)  # type: ignore[arg-type]


def stream(stream_: Optional["Stream"]) -> StreamContext:
    r"""Wrap around the Context-manager StreamContext that selects a given stream.

    Arguments:
        stream_ (Stream): selected stream. This manager is a no-op if it's
            ``None``.
    .. note::
        Streams are per-device.
    """
    return StreamContext(stream_)


def set_stream(stream: Stream):
    r"""Set the current stream. This is a wrapper API to set the stream.
        Usage of this function is discouraged in favor of the ``stream``
        context manager.

    Args:
        stream (Stream): selected stream. This function is a no-op
            if this argument is ``None``.
    """
    if stream is None:
        return
    _lcuda.set_stream(stream._stream)


def _parse_visible_devices() -> list[int] | list[str]:
    r"""Parse CUDA_VISIBLE_DEVICES environment variable."""
    var = os.getenv("CUDA_VISIBLE_DEVICES")

    if var is None:
        return list(range(64))

    def _strtoul(s: str) -> int:
        """Return -1 or positive integer sequence string starts with."""
        if not s:
            return -1
        for idx, c in enumerate(s):
            if not (c.isdigit() or (idx == 0 and c in "+-")):
                break
            if idx + 1 == len(s):
                idx += 1
        return int(s[:idx]) if idx > 0 else -1

    def parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
        rcs: list[str] = []
        for elem in lst.split(","):
            # Repeated id results in empty set
            if elem in rcs:
                return []
            # Anything other but prefix is ignored
            if not elem.startswith(prefix):
                break
            rcs.append(elem)
        return rcs

    if var.startswith("GPU-"):
        return parse_list_with_prefix(var, "GPU-")
    if var.startswith("MIG-"):
        return parse_list_with_prefix(var, "MIG-")
    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: list[int] = []
    for elem in var.split(","):
        x = _strtoul(elem.strip())
        # Repeated ordinal results in empty set
        if x in rc:
            return []
        # Negative value aborts the sequence
        if x < 0:
            break
        rc.append(x)
    return rc


def _raw_device_count_nvml() -> int:
    r"""Return number of devices as reported by NVML or negative value if NVML discovery/initialization failed."""
    from ctypes import byref, c_int, CDLL

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return -1
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return -1
    del nvml_h
    return dev_count.value


def _raw_device_uuid_nvml() -> list[str] | None:
    r"""Return list of device UUID as reported by NVML or None if NVM discovery/initialization failed."""
    from ctypes import byref, c_int, c_void_p, CDLL, create_string_buffer

    nvml_h = CDLL("libnvidia-ml.so.1")
    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=2)
        return None
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Can't get nvml device count", stacklevel=2)
        return None
    uuids: list[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn("Can't get device handle", stacklevel=2)
            return None
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn("Can't get device UUID", stacklevel=2)
            return None
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    del nvml_h
    return uuids


def _transform_uuid_to_ordinals(candidates: list[str], uuids: list[str]) -> list[int]:
    r"""Given the set of partial uuids and list of known uuids builds a set of ordinals excluding ambiguous partials IDs."""

    def uuid_to_ordinal(candidate: str, uuids: list[str]) -> int:
        best_match = -1
        for idx, uuid in enumerate(uuids):
            if not uuid.startswith(candidate):
                continue
            # Ambiguous candidate
            if best_match != -1:
                return -1
            best_match = idx
        return best_match

    rc: list[int] = []
    for candidate in candidates:
        idx = uuid_to_ordinal(candidate, uuids)
        # First invalid ordinal stops parsing
        if idx < 0:
            break
        # Duplicates result in empty set
        if idx in rc:
            return []
        rc.append(idx)
    return rc


def _device_count_nvml() -> int:
    r"""Return number of devices as reported by NVML taking CUDA_VISIBLE_DEVICES into account.

    Negative value is returned if NVML discovery or initialization has failed.
    """
    visible_devices = _parse_visible_devices()
    if not visible_devices:
        return 0
    try:
        if type(visible_devices[0]) is str:
            # Skip MIG parsing
            if visible_devices[0].startswith("MIG-"):
                return -1
            uuids = _raw_device_uuid_nvml()
            if uuids is None:
                return -1
            visible_devices = _transform_uuid_to_ordinals(visible_devices, uuids)
        else:
            raw_cnt = _raw_device_count_nvml()
            if raw_cnt <= 0:
                return raw_cnt
            # Trim the list up to a maximum available device
            for idx, val in enumerate(visible_devices):
                if val >= raw_cnt:
                    return idx
    except OSError:
        return -1
    except AttributeError:
        return -1
    return len(visible_devices)


_cached_device_count: int | None = None


def device_count() -> int:
    r"""
    Return the number of GPUs available.

    .. note:: This API will NOT poison fork if NVML discovery succeeds.
    """
    global _cached_device_count
    if not _is_compiled():
        return 0
    if _cached_device_count is not None:
        return _cached_device_count
    if _initialized or hasattr(_tls, "is_initializing"):
        r = _lcuda.device_count()
    else:
        nvml_count = _device_count_nvml()
        r = _lcuda.device_count() if nvml_count < 0 else nvml_count
    # NB: Do not cache the device count prior to CUDA initialization, because
    # the number of devices can change due to changes to CUDA_VISIBLE_DEVICES
    # setting prior to CUDA initialization.
    if _initialized:
        _cached_device_count = r
    return r


def get_arch_list() -> list[str]:
    r"""Return list CUDA architectures this library was compiled for."""
    if not _is_compiled():
        return []
    # No arch-flags binding exists in this build; report nothing rather than
    # guessing, which also disables the cubin compatibility checks.
    return []


def get_gencode_flags() -> str:
    r"""Return NVCC gencode flags this library was compiled with."""
    arch_list = get_arch_list()
    if len(arch_list) == 0:
        return ""
    arch_list_ = [arch.split("_") for arch in arch_list]
    return " ".join(
        [
            f"-gencode compute=compute_{arch},code={kind}_{arch}"
            for (kind, arch) in arch_list_
        ]
    )


def current_device() -> int:
    r"""Return the index of a currently selected device."""
    _lazy_init()
    return _lcuda.current_device()


def synchronize(device: Any = None) -> None:
    r"""Wait for all kernels in all streams on a CUDA device to complete.

    Args:
        device (tensorplay.Device or int, optional): device for which to synchronize.
            It uses the current device, given by :func:`~tensorplay.cuda.current_device`,
            if :attr:`device` is ``None`` (default).
    """
    _lazy_init()
    with _DeviceGuard(_get_device_index(device, optional=True)):
        return _lcuda.synchronize(-1)


def ipc_collect():
    r"""Force collects GPU memory after it has been released by CUDA IPC."""
    _require_native("ipc_collect")


def current_stream(device: Any = None) -> Stream:
    r"""Return the currently selected :class:`Stream` for a given device.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            the currently selected :class:`Stream` for the current device, given
            by :func:`~tensorplay.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    core = _lcuda.current_stream(_get_device_index(device, optional=True))
    return Stream(_stream=core)


def default_stream(device: Any = None) -> Stream:
    r"""Return the default :class:`Stream` for a given device.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            the default :class:`Stream` for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is ``None``
            (default).
    """
    _lazy_init()
    core = _lcuda.default_stream(_get_device_index(device, optional=True))
    return Stream(_stream=core)


def get_stream_from_external(data_ptr: int, device: Any = None) -> Stream:
    r"""Return a :class:`Stream` from an externally allocated CUDA stream."""
    _require_native("getStreamFromExternal")


def current_blas_handle():
    r"""Return cublasHandle_t pointer to current cuBLAS handle"""
    _require_native("current_blas_handle")


def current_solver_handle():
    r"""Return cusolverDnHandle_t pointer to current cuSOLVER handle"""
    _require_native("current_solver_handle")


def set_sync_debug_mode(debug_mode: int | str) -> None:
    r"""Set the debug mode for cuda synchronizing operations.

    Not enforced by this TensorPlay build; the signature is retained for API
    compatibility.

    Args:
        debug_mode(str or int): if "default" or 0, don't error or warn on synchronizing operations,
            if "warn" or 1, warn on synchronizing operations, if "error" or 2, error out synchronizing operations.
    """
    if isinstance(debug_mode, str):
        mapping = {"default": 0, "warn": 1, "error": 2}
        if debug_mode not in mapping:
            raise RuntimeError(
                "invalid value of debug_mode, expected one of `default`, `warn`, `error`"
            )


def get_sync_debug_mode() -> int:
    r"""Return current value of debug mode for cuda synchronizing operations.

    Always returns ``0`` in this build (mode is not enforced).
    """
    return 0


# pyrefly: ignore [deprecated]
from .memory import *  # noqa: F403
from .random import *  # noqa: F403


################################################################################
# NVML-backed metrics
################################################################################


def _get_pynvml_handler(device: Any = None):
    if not _HAS_PYNVML:
        raise ModuleNotFoundError(
            "nvidia-ml-py does not seem to be installed or it can't be imported."
        ) from _PYNVML_ERR
    from pynvml import NVMLError_DriverNotLoaded

    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded as e:
        raise RuntimeError("cuda driver can't be loaded, is cuda enabled?") from e

    device_idx = _get_nvml_device_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    return handle


def _get_nvml_device_index(device: Any) -> int:
    r"""Return the NVML index of the device, taking CUDA_VISIBLE_DEVICES into account."""
    idx = _get_device_index(device, optional=True)
    visible_devices = _parse_visible_devices()
    if len(visible_devices) > 0 and type(visible_devices[0]) is str:
        uuids = _raw_device_uuid_nvml()
        if uuids is None:
            raise RuntimeError("Can't get device UUIDs")
        visible_devices = _transform_uuid_to_ordinals(visible_devices, uuids)
    if idx < 0 or idx >= len(visible_devices):
        raise RuntimeError(
            f"device {idx} is not visible (CUDA_VISIBLE_DEVICES={visible_devices})"
        )
    return visible_devices[idx]


def device_memory_used(device: Any = None) -> int:
    r"""Return used global (device) memory in bytes as given by `nvidia-smi`.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
    """
    import pynvml

    handle = _get_pynvml_handler(device)
    return pynvml.nvmlDeviceGetMemoryInfo(handle).used


def memory_usage(device: Any = None) -> int:
    r"""Return the percent of time over the past sample period during which global (device)
    memory was being read or written as given by `nvidia-smi`.

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    import pynvml

    handle = _get_pynvml_handler(device)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).memory


def utilization(device: Any = None) -> int:
    r"""Return the percent of time over the past sample period during which one or
    more kernels was executing on the GPU as given by `nvidia-smi`.

    Warning: Each sample period may be between 1 second and 1/6 second,
    depending on the product being queried.
    """
    import pynvml

    handle = _get_pynvml_handler(device)
    return pynvml.nvmlDeviceGetUtilizationRates(handle).gpu


def temperature(device: Any = None) -> int:
    r"""Return the average temperature of the GPU sensor in Degrees C (Centigrades).

    The average temperature is computed based on past sample period as given by `nvidia-smi`.
    """
    import pynvml

    handle = _get_pynvml_handler(device)
    # 0 refers to the temperature sensor for the GPU die.
    return pynvml.nvmlDeviceGetTemperature(handle, 0)


def power_draw(device: Any = None) -> int:
    r"""Return the average power draw of the GPU sensor in mW (MilliWatts)
        over the past sample period as given by `nvidia-smi` for Fermi or newer fully supported devices.
    """
    import pynvml

    handle = _get_pynvml_handler(device)
    return pynvml.nvmlDeviceGetPowerUsage(handle)


def clock_rate(device: Any = None) -> int:
    r"""Return the clock speed of the GPU SM in MHz (megahertz) over the past sample period as given by `nvidia-smi`."""
    import pynvml

    handle = _get_pynvml_handler(device)
    return pynvml.nvmlDeviceGetClockInfo(handle, 1)


from . import jiterator, nccl, nvtx, profiler, sparse, tunable  # noqa: E402
from .graph_annotations import (  # noqa: E402
    clear_kernel_annotations,
    get_kernel_annotations,
    mark_kernels,
)


__all__ = [
    "AcceleratorError",
    "CUDAGraph",
    "CUDAPluggableAllocator",
    "CudaError",
    "DeferredCudaCallError",
    "Event",
    "ExternalStream",
    "GdsFile",
    "GreenContext",
    "MemPool",
    "OutOfMemoryError",
    "Stream",
    "StreamContext",
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_disabled",
    "caching_allocator_enable",
    "can_device_access_peer",
    "change_current_allocator",
    "check_error",
    "clock_rate",
    "cudaStatus",
    "cudart",
    "current_blas_handle",
    "current_device",
    "current_solver_handle",
    "current_stream",
    "default_generators",
    "default_stream",
    "device",
    "device_count",
    "device_memory_used",
    "device_of",
    "empty_cache",
    "export_dot",
    "gds",
    "get_allocator_backend",
    "get_arch_list",
    "get_device_capability",
    "get_device_name",
    "get_device_properties",
    "get_gencode_flags",
    "get_per_process_memory_fraction",
    "get_rng_state",
    "get_rng_state_all",
    "get_stream_from_external",
    "get_sync_debug_mode",
    "graph",
    "graph_annotations",
    "graph_pool_handle",
    "graphs",
    "has_half",
    "has_magma",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "init",
    "initial_seed",
    "ipc_collect",
    "is_available",
    "is_bf16_supported",
    "is_current_stream_capturing",
    "is_gds_available",
    "is_initialized",
    "is_tf32_supported",
    "jiterator",
    "list_gpu_processes",
    "make_graphed_callables",
    "manual_seed",
    "manual_seed_all",
    "max_memory_allocated",
    "max_memory_reserved",
    "mem_get_info",
    "memory",
    "memory_allocated",
    "memory_reserved",
    "memory_snapshot",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "memory_summary",
    "memory_usage",
    "nccl",
    "nvtx",
    "power_draw",
    "profiler",
    "random",
    "reset_accumulated_host_memory_stats",
    "reset_accumulated_memory_stats",
    "reset_peak_host_memory_stats",
    "reset_peak_memory_stats",
    "seed",
    "seed_all",
    "set_device",
    "set_per_process_memory_fraction",
    "set_rng_state",
    "set_rng_state_all",
    "set_stream",
    "set_sync_debug_mode",
    "sparse",
    "stream",
    "streams",
    "synchronize",
    "temperature",
    "tunable",
    "utilization",
    "use_mem_pool",
]

# Submodules referenced by __all__
from . import graph_annotations as graph_annotations, graphs as graphs, jiterator, memory as memory, nccl, nvtx, profiler, random as random, sparse, streams as streams, tunable  # noqa: E402
