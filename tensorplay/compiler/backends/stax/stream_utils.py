"""Stream utilities for Inductor codegen."""

from __future__ import annotations

import functools

import tensorplay as tp
from typing import TYPE_CHECKING

from .stream_constants import (
    DEFAULT_STREAM,
    DEFAULT_STREAM_IDX,
    STREAM_NAME_TEMPLATE,
)




__all__ = [
    "AOTI_SUPPORTED_STREAM_OP_NAMES",
    "AOTI_UNSUPPORTED_STREAM_OP_REASONS",
    "DEFAULT_STREAM",
    "DEFAULT_STREAM_IDX",
    "STREAM_NAME_TEMPLATE",
    "get_raw_stream_name",
    "get_stream_name",
]


AOTI_SUPPORTED_STREAM_OP_NAMES: dict[str, str] = {
    "tp.ops.streams.record_event.default": "record_event",
    "tp.ops.streams.wait_event.default": "wait_event",
    # A targeted single-event host wait (cudaEventSynchronize); required by e.g.
    # pinned non-blocking copies. Unlike synchronize_stream/device, it does not
    # block the whole stream/device, so it is safe to emit inside an AOTI Run().
    "tp.ops.streams.synchronize_event.default": "synchronize_event",
}


AOTI_UNSUPPORTED_STREAM_OP_REASONS: dict[str, str] = {
    "tp.ops.streams.synchronize_stream.default": (
        "Host-blocking sync ops are not supported inside an AOTI Run(). "
        "Use record_event + wait_event for device-side ordering instead."
    ),
    "tp.ops.streams.synchronize_device.default": (
        "Host-blocking device synchronization is not supported inside an AOTI Run()."
    ),
    "tp.ops.streams.wait_stream.default": (
        "wait_stream is not supported in AOTI cpp_wrapper. Use explicit "
        "record_event on the waited-on stream + wait_event on the waiting stream."
    ),
}


@functools.lru_cache
def get_stream_name(stream_idx: int) -> str:
    """Generate CUDA Stream name from stream index number.

    Args:
        stream_idx: Non-negative index number. 0 refers to the default stream, others refer to side
            streams.
    """
    if stream_idx == 0:
        return DEFAULT_STREAM
    else:
        return STREAM_NAME_TEMPLATE.format(stream_idx=stream_idx)


@functools.lru_cache
def _raw_stream_name_for_device(device_idx: int) -> str:
    return f"raw_stream{device_idx}"


def get_raw_stream_name(device_idx: int) -> str:
    """Generate variable name for a raw stream handle on the given device."""
    # Under compile-on-one-rank the wrapper must be byte-identical across ranks, so the
    # stream variable name cannot carry a rank-specific device index.
    from .coor import _coor_enabled

    if _coor_enabled():
        return "raw_stream"
    return _raw_stream_name_for_device(device_idx)


# [device-as-parameter] Name of the function-local variable holding the runtime current
# device index under compile-on-one-rank, so generated code is byte-identical across ranks.
COOR_DEVICE_IDX_VAR = "_coor_device_idx"


def coor_device_str(device: tp.device | None) -> str:
    """Device string to emit into generated code (benchmark and autotune harnesses).

    Under compile-on-one-rank drop the index (``cuda:0`` -> ``cuda``) so the emitted text
    is rank-identical and its tensors land on the running rank's current device.
    """
    from .coor import _coor_enabled

    if _coor_enabled() and device is not None:
        return device.type
    return str(device)


def coor_benchmark_device_idx(device_idx: int) -> tuple[str | None, int | str]:
    """(preamble line, index token) for a Triton kernel's benchmark harness.

    The harness is emitted into the kernel's own source, which is what the kernel hash
    covers, so under compile-on-one-rank it must resolve the device at run time rather
    than bake a rank-specific index -- otherwise kernels are not rank-identical whenever
    ``config.benchmark_kernel`` is on. The preamble line (when not None) must be emitted
    inside the harness function before the returned token is used.
    """
    from .coor import _coor_enabled

    if not _coor_enabled():
        return None, device_idx

    from tp._inductor.virtualized import V

    expr = V.graph.device_ops.current_device_idx_expr()
    return f"{COOR_DEVICE_IDX_VAR} = {expr}", COOR_DEVICE_IDX_VAR
