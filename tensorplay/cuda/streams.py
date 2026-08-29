# mypy: allow-untyped-defs
# pylint: disable=useless-parent-delegation
r"""CUDA stream and event wrappers.

The underlying objects are ``tensorplay._C._cuda._CudaStream`` /
``tensorplay._C._cuda._CudaEvent`` when native CUDA support is available.
"""

from __future__ import annotations

from typing import Any, Optional

import tensorplay
import tensorplay._C
from tensorplay._C import Device
from tensorplay._C import _cuda as _lcuda

from ._utils import _dummy_type, _get_device_index


if not hasattr(_lcuda, "_CudaStream"):
    # Define dummy base classes (build without CUDA support)
    _CudaStreamBase = _dummy_type("_CudaStream")
    _CudaEventBase = _dummy_type("_CudaEvent")
    _has_core = False
else:
    _CudaStreamBase = _lcuda._CudaStream
    _CudaEventBase = _lcuda._CudaEvent
    _has_core = True


class Stream:
    r"""Wrapper around a CUDA stream.

    A CUDA stream is a linear sequence of execution that belongs to a specific
    device, independent from other streams. It supports with statement as a
    context manager to ensure the operators within the with block are running
    on the corresponding stream.  See the CUDA semantics documentation for details.

    Args:
        device(tensorplay.Device or int, optional): a device on which to allocate
            the stream. If :attr:`device` is ``None`` (default) or a negative
            integer, this will use the current device.
        priority(int, optional): priority of the stream, which can be positive, 0, or negative.
            A lower number indicates a higher priority. By default, the priority is set to 0.

    """

    def __new__(cls, device=None, priority=0, **kwargs):
        if not _has_core:
            raise RuntimeError("tensorplay.cuda.Stream requires CUDA support")
        core_stream = kwargs.pop("_stream", None)
        if core_stream is None:
            from . import device as device_ctx

            idx = _get_device_index(device, optional=True)
            with device_ctx(idx):
                core_stream = _lcuda.get_stream_from_pool(int(priority), idx)
        self = super().__new__(cls)
        self.__dict__["_stream"] = core_stream
        return self

    @property
    def device(self) -> Device:
        return self._stream.device

    @property
    def cuda_stream(self) -> int:
        return self._stream.cuda_stream

    @property
    def stream_id(self) -> int:
        return self._stream.cuda_stream

    @property
    def device_index(self) -> int:
        return int(self.device.index)

    @property
    def device_type(self) -> str:
        return "cuda"

    def wait_event(self, event: "Event | tensorplay.Event") -> None:
        r"""Make all future work submitted to the stream wait for an event."""
        event.wait(self)

    def wait_stream(self, stream: "Stream") -> None:
        r"""Synchronize with another stream."""
        self.wait_event(stream.record_event())

    def record_event(self, event: Optional["Event"] = None):
        r"""Record an event.

        Args:
            event (Event, optional): event to record. If not given, a new one
                will be allocated.

        Returns:
            Recorded event.
        """
        if event is None:
            event = Event()
        event.record(self)
        return event

    def query(self) -> bool:
        r"""Check if all the work submitted has been completed."""
        return self._stream.query()

    def synchronize(self) -> None:
        r"""Wait for all the kernels in this stream to complete."""
        self._stream.synchronize()

    def __enter__(self):
        from . import StreamContext

        self._ctx = StreamContext(self)
        self._ctx.__enter__()
        return self

    def __exit__(self, type: Any, value: Any, traceback: Any):
        from . import StreamContext

        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            ctx.__exit__(type, value, traceback)
        return False

    def __eq__(self, o) -> bool:
        if isinstance(o, Stream):
            return bool(self._stream == o._stream)
        return False

    def __hash__(self):
        return hash((self.cuda_stream, self.device.index))

    def __repr__(self):
        return f"<tensorplay.cuda.Stream device={self.device} cuda_stream={self.cuda_stream:#x}>"


class ExternalStream(Stream):
    r"""Wrapper around an externally allocated CUDA stream.

    This class is used to wrap streams allocated in other libraries in order
    to facilitate data exchange and multi-library interactions.

    .. note:: This class doesn't manage the stream life-cycle, it is the user
       responsibility to keep the referenced stream alive while this class is
       being used.

    Args:
        stream_ptr(int): Integer representation of the `cudaStream_t` value
            allocated externally.
        device(tensorplay.Device or int, optional): the device where the stream
            was originally allocated. If device is specified incorrectly,
            subsequent launches using this stream may fail.
    """

    def __new__(cls, stream_ptr, device=None, **kwargs):
        if not _has_core:
            raise RuntimeError("tensorplay.cuda.Stream requires CUDA support")
        raise RuntimeError(
            "External streams are not exposed by this TensorPlay build"
        )


class Event:
    r"""Wrapper around a CUDA event.

    CUDA events are synchronization markers that can be used to monitor the
    device's progress, to accurately measure timing, and to synchronize CUDA
    streams.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time
            (default: ``False``)
        blocking (bool, optional): if ``True``, :meth:`wait` will be blocking (default: ``False``)
        interprocess (bool): if ``True``, the event can be shared between processes
            (default: ``False``)
    """

    def __new__(
        cls, enable_timing=False, blocking=False, interprocess=False, external=False
    ):
        if not _has_core:
            raise RuntimeError("tensorplay.cuda.Event requires CUDA support")
        self = super().__new__(cls)
        self.__dict__["_event"] = _lcuda._CudaEvent(
            enable_timing, blocking, interprocess
        )
        return self

    @classmethod
    def from_ipc_handle(cls, device, handle):
        r"""Reconstruct an event from an IPC handle on the given device."""
        raise RuntimeError("IPC events are not supported by this TensorPlay build")

    @property
    def device(self):
        return self._event.device

    @property
    def cuda_event(self) -> int:
        return self._event.cuda_event

    def record(self, stream: Optional[Stream] = None):
        r"""Record the event in a given stream.

        Args:
            stream (Stream, optional): Uses ``tensorplay.cuda.current_stream()`` if no stream is specified.
        """
        if stream is None:
            from . import current_stream

            stream = current_stream()
        self._event.record(stream._stream)

    def wait(self, stream: Optional[Stream] = None) -> None:
        r"""Make all future work submitted to the given stream wait for this event.

        Args:
            stream (Stream, optional): Uses ``tensorplay.cuda.current_stream()`` if no stream is specified.
        """
        if stream is None:
            from . import current_stream

            stream = current_stream()
        self._event.wait(stream._stream)

    def query(self):
        r"""Check if all work currently captured by event has completed."""
        return self._event.query()

    def elapsed_time(self, end_event: "Event"):
        r"""Return the time elapsed.

        Time reported in milliseconds after the event was recorded and
        before the end_event was recorded.

        Args:
            end_event (Event): the end event.
        """
        return self._event.elapsed_time(end_event._event)

    def synchronize(self) -> None:
        r"""Wait for the event to complete."""
        self._event.synchronize()

    def ipc_handle(self):
        r"""Return an IPC handle of this event."""
        raise RuntimeError("IPC events are not supported by this TensorPlay build")

    def __repr__(self) -> str:
        if self.cuda_event:
            return f"<tensorplay.cuda.Event {self.cuda_event:#x}>"
        else:
            return "<tensorplay.cuda.Event uninitialized>"
