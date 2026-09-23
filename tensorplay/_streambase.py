# mypy: allow-untyped-defs
"""Shared stream/event contract for every device backend.

``StreamBase``/``EventBase`` spell the ordering primitives once so
generic code can treat host and accelerator streams uniformly; each
backend subclasses them with its own implementation.
"""

__all__ = ["StreamBase", "EventBase"]


class StreamBase:
    """Ordered execution context owned by a device."""

    def query(self) -> bool:
        """Whether all work submitted to this stream has completed."""
        raise NotImplementedError

    def synchronize(self) -> None:
        """Block until all work submitted to this stream completes."""
        raise NotImplementedError

    def wait_event(self, event) -> None:
        """Make this stream wait for an event recorded elsewhere."""
        raise NotImplementedError

    def wait_stream(self, stream) -> None:
        """Make this stream wait for all work on another stream."""
        raise NotImplementedError

    def record_event(self, event=None):
        """Record an event on this stream and return it."""
        raise NotImplementedError


class EventBase:
    """Completion marker recorded on a stream."""

    def query(self) -> bool:
        """Whether the event has been reached by its stream."""
        raise NotImplementedError

    def record(self, stream=None) -> None:
        """Record the event on the given (or current) stream."""
        raise NotImplementedError

    def synchronize(self) -> None:
        """Block until the event has been reached."""
        raise NotImplementedError

    def wait(self, stream=None) -> None:
        """Make the given (or current) stream wait for this event."""
        raise NotImplementedError
