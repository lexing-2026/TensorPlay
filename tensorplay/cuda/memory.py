# mypy: allow-untyped-defs
r"""Device memory management, mirroring :mod:`torch.cuda.memory`.

The native runtime currently exposes the ``allocated``/``reserved``,
current/peak counters and their peak resets. The remaining statistics of the
torch matrix are reported as zero, mirroring torch's own behaviour under
allocator backends where "some stats are not meaningful". Snapshot/history
APIs require allocator instrumentation this build does not expose.
"""

import collections
import contextlib
import warnings
from typing import Any

import tensorplay
from tensorplay._C import _cuda as _lcuda

from . import _get_device_index, is_initialized


__all__ = [
    "caching_allocator_alloc",
    "caching_allocator_delete",
    "caching_allocator_disabled",
    "caching_allocator_enable",
    "get_per_process_memory_fraction",
    "set_per_process_memory_fraction",
    "empty_cache",
    "memory_stats",
    "memory_stats_as_nested_dict",
    "reset_accumulated_memory_stats",
    "reset_peak_memory_stats",
    "reset_max_memory_allocated",
    "reset_max_memory_cached",
    "host_memory_stats",
    "host_memory_stats_as_nested_dict",
    "reset_accumulated_host_memory_stats",
    "reset_peak_host_memory_stats",
    "memory_allocated",
    "max_memory_allocated",
    "memory_reserved",
    "max_memory_reserved",
    "memory_cached",
    "max_memory_cached",
    "memory_snapshot",
    "memory_summary",
    "list_gpu_processes",
    "mem_get_info",
    "get_allocator_backend",
    "CUDAPluggableAllocator",
    "change_current_allocator",
    "MemPool",
    "use_mem_pool",
]


def empty_cache() -> None:
    r"""Release all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU application and visible in
    `nvidia-smi`.

    .. note::
        :func:`~tensorplay.cuda.empty_cache` doesn't increase the amount of GPU
        memory available for TensorPlay. However, it may help reduce fragmentation
        of GPU memory in certain cases.
    """
    if is_initialized():
        _lcuda.empty_cache()


def _recurse_add_to_result(result, prefix, obj, format_key):
    if isinstance(obj, dict):
        if prefix:
            prefix += "."
        for key, value in obj.items():
            _recurse_add_to_result(result, prefix + format_key(key), value, format_key)
    else:
        result.append((prefix, obj))


def memory_stats_as_nested_dict(device: Any = None) -> dict[str, Any]:
    r"""Return the result of :func:`~tensorplay.cuda.memory_stats` as a nested dictionary.

    The native allocator reports the fragmentation-aware matrix directly
    (segments, free-block histogram, pending cross-stream blocks, graph
    pools, capture state); it is exposed under ``"allocator"`` alongside the
    torch-compatible pool layout.
    """
    if not is_initialized():
        return {}
    idx = _get_device_index(device, optional=True)
    allocated = {
        "all": {
            "current": _lcuda.memory_allocated(idx),
            "peak": _lcuda.max_memory_allocated(idx),
            "allocated": 0,
            "freed": 0,
        }
    }
    reserved = {
        "all": {
            "current": _lcuda.memory_reserved(idx),
            "peak": _lcuda.max_memory_reserved(idx),
            "allocated": 0,
            "freed": 0,
        }
    }

    def _pools(base):
        pools = {"large_pool": {m: 0 for m in ("current", "peak", "allocated", "freed")},
                 "small_pool": {m: 0 for m in ("current", "peak", "allocated", "freed")}}
        return {**pools, "all": base}

    allocator_native: dict[str, Any] = {}
    try:
        native = _lcuda.memory_stats(idx)
        allocator_native = dict(native) if isinstance(native, dict) else {}
    except Exception:
        allocator_native = {}

    return {
        "allocator": {
            **allocator_native,
            "allocation": _pools(allocated),
            "segment": _pools(reserved),
            "active": _pools(allocated),
            "active_bytes": _pools(allocated),
        },
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "allocation": _pools(allocated),
        "segment": _pools(reserved),
        "active": _pools(allocated),
        "active_bytes": _pools(allocated),
    }

def memory_stats(device: Any = None) -> dict[str, Any]:
    r"""Return a dictionary of CUDA memory allocator statistics for a given device.

    The return value of this function is a dictionary of statistics, each of
    which is a non-negative integer. See :func:`torch.cuda.memory_stats` for
    the full key layout; keys that are not tracked by this TensorPlay build
    are always reported as zero.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            statistics for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
    """
    result = []

    def _format_key(key):
        if isinstance(key, str):
            return key
        if isinstance(key, tuple):
            return "_".join(str(part) for part in key)
        return str(key)

    stats = memory_stats_as_nested_dict(device=device)
    _recurse_add_to_result(result, "", stats, _format_key)
    result.sort()

    out = collections.OrderedDict(result)
    # The fragmentation-aware native matrix stays addressable as a
    # sub-dictionary (torch exposes it only through mem_get_info-free
    # helpers); dotted torch-compatible keys above remain untouched.
    allocator = stats.get("allocator")
    if isinstance(allocator, dict):
        out["allocator"] = collections.OrderedDict(allocator)
    return out


def reset_accumulated_memory_stats(device: Any = None) -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the CUDA memory allocator.

    This TensorPlay build does not track historical totals; the call is a no-op.
    """
    pass


def reset_peak_memory_stats(device: Any = None) -> None:
    r"""Reset the "peak" stats tracked by the CUDA memory allocator.

    Peak stats correspond to the `"peak"` key in each individual stat dict.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
    """
    idx = _get_device_index(device, optional=True)
    _lcuda.reset_peak_memory_stats(idx)
    _lcuda.reset_max_memory_allocated(idx)


def host_memory_stats() -> dict[str, Any]:
    r"""Return a dictionary of pinned (host) allocator statistics."""
    result = []
    stats = host_memory_stats_as_nested_dict()
    _recurse_add_to_result(result, "", stats, str)
    result.sort()
    return collections.OrderedDict(result)


def host_memory_stats_as_nested_dict() -> dict[str, Any]:
    r"""Return the result of :func:`~tensorplay.cuda.host_memory_stats` as a nested dictionary."""
    return {}


def reset_accumulated_host_memory_stats() -> None:
    r"""Reset the "accumulated" (historical) stats tracked by the host memory allocator."""
    pass


def reset_peak_host_memory_stats() -> None:
    r"""Reset the "peak" stats tracked by the host memory allocator."""
    pass


def reset_max_memory_allocated(device: Any = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory occupied by tensors for a given device.

    .. warning::
        This function now calls :func:`~tensorplay.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.
    """
    warnings.warn(
        "tensorplay.cuda.reset_max_memory_allocated now calls tensorplay.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
        stacklevel=2,
    )
    return reset_peak_memory_stats(device=device)


def reset_max_memory_cached(device: Any = None) -> None:
    r"""Reset the starting point in tracking maximum GPU memory managed by the caching allocator for a given device.

    .. warning::
        This function now calls :func:`~tensorplay.cuda.reset_peak_memory_stats`, which resets
        /all/ peak memory stats.
    """
    warnings.warn(
        "tensorplay.cuda.reset_max_memory_cached now calls tensorplay.cuda.reset_peak_memory_stats, "
        "which resets /all/ peak memory stats.",
        FutureWarning,
        stacklevel=2,
    )
    return reset_peak_memory_stats(device=device)


def memory_allocated(device: Any = None) -> int:
    r"""Return the current GPU memory occupied by tensors in bytes for a given device.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.current", 0)


def max_memory_allocated(device: Any = None) -> int:
    r"""Return the maximum GPU memory occupied by tensors in bytes for a given device.

    By default, this returns the peak allocated memory since the beginning of
    this program. :func:`~tensorplay.cuda.reset_peak_memory_stats` can be used to
    reset the starting point in tracking this metric.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
    """
    return memory_stats(device=device).get("allocated_bytes.all.peak", 0)


def memory_reserved(device: Any = None) -> int:
    r"""Return the current GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.current", 0)


def max_memory_reserved(device: Any = None) -> int:
    r"""Return the maximum GPU memory managed by the caching allocator in bytes for a given device.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            statistic for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
    """
    return memory_stats(device=device).get("reserved_bytes.all.peak", 0)


def memory_cached(device: Any = None) -> int:
    r"""Deprecated; see :func:`~tensorplay.cuda.memory_reserved`."""
    warnings.warn(
        "tensorplay.cuda.memory_cached has been renamed to tensorplay.cuda.memory_reserved",
        FutureWarning,
        stacklevel=2,
    )
    return memory_reserved(device=device)


def max_memory_cached(device: Any = None) -> int:
    r"""Deprecated; see :func:`~tensorplay.cuda.max_memory_reserved`."""
    warnings.warn(
        "tensorplay.cuda.max_memory_cached has been renamed to tensorplay.cuda.max_memory_reserved",
        FutureWarning,
        stacklevel=2,
    )
    return max_memory_reserved(device=device)


def memory_snapshot(mempool_id=None, include_traces=True):
    r"""Return a snapshot of the CUDA memory allocator state across all devices.

    Interpreting the output of this function requires familiarity with the
    memory allocator internals. Not exposed by this TensorPlay build.
    """
    raise RuntimeError(
        "memory snapshots are not supported by this TensorPlay build"
    )


def memory_summary(device: Any = None, abbreviated: bool = False) -> str:
    r"""Return a human-readable printout of the current memory allocator statistics for a given device.

    This can be useful to display periodically during training, or when
    handling out-of-memory exceptions.

    Args:
        device (tensorplay.Device or int, optional): selected device. Returns
            printout for the current device, given by
            :func:`~tensorplay.cuda.current_device`, if :attr:`device` is
            ``None`` (default).
        abbreviated (bool, optional): whether to return an abbreviated summary
            (default: False).
    """
    device = _get_device_index(device, optional=True)
    stats = memory_stats(device=device)

    def _format_size(sz, pref_sz):
        prefixes = ["B  ", "KiB", "MiB", "GiB", "TiB", "PiB"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_sz < 768 * 1024:
                break
            prefix = new_prefix
            sz //= 1024
            pref_sz /= 1024
        return f"{sz:6d} {prefix}"

    def _format_count(cnt, pref_cnt):
        prefixes = [" ", "K", "M"]
        prefix = prefixes[0]
        for new_prefix in prefixes[1:]:
            if pref_cnt < 750 * 1000:
                break
            prefix = new_prefix
            cnt //= 1000
            pref_cnt /= 1000
        return f"{cnt:7d} {prefix} "

    metrics_to_display = [
        ("allocated_bytes", "Allocated memory", _format_size),
        ("active_bytes", "Active memory", _format_size),
        ("requested_bytes", "Requested memory", _format_size),
        ("reserved_bytes", "GPU reserved memory", _format_size),
        ("inactive_split_bytes", "Non-releasable memory", _format_size),
        ("allocation", "Allocations", _format_count),
        ("active", "Active allocs", _format_count),
        ("segment", "GPU reserved segments", _format_count),
        ("inactive_split", "Non-releasable allocs", _format_count),
    ]

    lines = []
    lines.append("=" * 75)
    lines.append(" {_:16} TensorPlay CUDA memory summary, device ID {device:<17d} ")
    lines.append("-" * 75)
    lines.append(
        "  {_:9} CUDA OOMs: {num_ooms:<12d} | {_:6} cudaMalloc retries: {num_alloc_retries:<8d}  "
    )
    lines.append("=" * 75)
    lines.append(
        "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "
    )

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)
        submetrics = [("all", metric_name)]
        if not abbreviated:
            submetrics.append(("large_pool", "      from large pool"))
            submetrics.append(("small_pool", "      from small pool"))

        current_prefval, peak_prefval, allocated_prefval, freed_prefval = (
            None,
            None,
            None,
            None,
        )

        for submetric_key, submetric_name in submetrics:
            prefix = metric_key + "." + submetric_key + "."

            current = stats[prefix + "current"]
            peak = stats[prefix + "peak"]
            allocated = stats[prefix + "allocated"]
            freed = stats[prefix + "freed"]

            if current_prefval is None:
                current_prefval = current
                peak_prefval = peak
                allocated_prefval = allocated
                freed_prefval = freed

            lines.append(
                f" {submetric_name:<21} | {formatter(current, current_prefval)} | {formatter(peak, peak_prefval)} | "
                f"{formatter(allocated, allocated_prefval)} | {formatter(freed, freed_prefval)} ",
            )

    metrics_to_display = [
        ("oversize_allocations", "Oversize allocations", _format_count),
        ("oversize_segments", "Oversize GPU segments", _format_count),
    ]

    for metric_key, metric_name, formatter in metrics_to_display:
        lines.append("-" * 75)

        prefix = metric_key + "."

        current = stats[prefix + "current"]
        peak = stats[prefix + "peak"]
        allocated = stats[prefix + "allocated"]
        freed = stats[prefix + "freed"]

        lines.append(
            f" {metric_name:<21} | {formatter(current, current)} | {formatter(peak, peak)} | "
            f"{formatter(allocated, allocated)} | {formatter(freed, freed)} ",
        )

    lines.append("=" * 75)

    fmt_dict = {"_": "", "device": device}
    for k, v in stats.items():
        fmt_dict[k.replace(".", "-")] = v
    return "|" + "|\n|".join(lines).format(**fmt_dict) + "|\n"


def list_gpu_processes(device: Any = None) -> str:
    r"""Return a human-readable printout of the running processes and their GPU memory use for a given device."""
    try:
        import pynvml
    except ModuleNotFoundError:
        return "pynvml module not found, please install nvidia-ml-py"
    from pynvml import NVMLError_DriverNotLoaded

    try:
        pynvml.nvmlInit()
    except NVMLError_DriverNotLoaded:
        return "cuda driver can't be loaded, is cuda enabled?"

    from . import _get_nvml_device_index

    device_idx = _get_nvml_device_index(device)
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)

    lines = []
    lines.append(f"GPU:{device_idx}")
    if len(procs) == 0:
        lines.append("no processes are running")
    for p in procs:
        mem = p.usedGpuMemory / (1024 * 1024)
        pid = p.pid
        lines.append(f"process {pid:>10d} uses {mem:>12.3f} MB GPU memory")
    return "\n".join(lines)


def mem_get_info(device: Any = None) -> tuple[int, int]:
    r"""Return the global free and total GPU memory for a given device using cudaMemGetInfo."""
    from . import cudart

    rt = cudart()
    if rt is None:
        raise RuntimeError("cudart is not available in this TensorPlay build")
    if device is None:
        from . import current_device

        device = current_device()
    device = _get_device_index(device, optional=True)
    return rt.cudaMemGetInfo(device)


def caching_allocator_alloc(size, device: Any = None, stream=None):
    r"""Perform a memory allocation using the CUDA memory allocator.

    Not exposed by this TensorPlay build.
    """
    raise RuntimeError(
        "caching_allocator_alloc is not supported by this TensorPlay build"
    )


def caching_allocator_delete(mem_ptr):
    r"""Delete memory allocated using the CUDA memory allocator."""
    raise RuntimeError(
        "caching_allocator_delete is not supported by this TensorPlay build"
    )


def caching_allocator_enable(value: bool = True) -> None:
    r"""Enable or disable the CUDA memory allocator. On by default."""
    pass


@contextlib.contextmanager
def caching_allocator_disabled():
    r"""Context manager that temporarily disables the CUDA caching allocator."""
    # The allocator cannot be disabled in this build; provided for API parity.
    yield


def set_per_process_memory_fraction(fraction, device: Any = None) -> None:
    r"""Set memory fraction for a process.

    Not enforced by this TensorPlay build; validated for signature parity.

    Args:
        fraction(float): Range: 0~1. Allowed memory equals total_memory * fraction.
        device (tensorplay.Device or int, optional): selected device.
    """
    if not isinstance(fraction, float):
        raise TypeError("Invalid type for fraction argument, must be `float`")
    if fraction < 0 or fraction > 1:
        raise ValueError(f"Invalid fraction value: {fraction}. Allowed range: 0~1")


def get_per_process_memory_fraction(device: Any = None) -> float:
    r"""Get memory fraction for a process. Always returns ``1.0`` in this build."""
    return 1.0


def get_allocator_backend() -> str:
    r"""Returns the active allocator backend as a string. Always ``native`` here."""
    return "native"


class CUDAPluggableAllocator:
    r"""CUDA memory allocator plugin (not supported by this TensorPlay build)."""

    def __init__(self, so_file: str, alloc_fn_name: str = "my_alloc", free_fn_name: str = "my_free"):
        raise RuntimeError(
            "pluggable CUDA allocators are not supported by this TensorPlay build"
        )


def change_current_allocator(alloc):
    r"""Change the currently used memory allocator (not supported by this build)."""
    raise RuntimeError(
        "pluggable CUDA allocators are not supported by this TensorPlay build"
    )


class MemPool:
    r"""MemPool context (not supported by this TensorPlay build)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("MemPool is not supported by this TensorPlay build")


def use_mem_pool(pool):
    r"""Route allocations to a MemPool (not supported by this build)."""
    raise RuntimeError("MemPool is not supported by this TensorPlay build")
