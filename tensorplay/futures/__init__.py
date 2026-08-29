"""Asynchronous future values used by TensorPlay.

Provides :class:`Future`, the asynchronous execution primitive used by
TensorPlay's implementation is pure Python: completion is backed by an optional
``_completer`` callable (e.g. a CUDA event synchronization) that runs on the
first ``wait()``/``value()`` call, after which ``then`` callbacks fire in
registration order on derived futures.
"""

import threading
from typing import Any, Callable, List, Optional

__all__ = ["Future"]


class Future:
    """

    Example::

        >>> fut = tp.futures.Future()
        >>> fut.set_result(1)
        >>> fut.value()
        1
        >>> chained = other.then(lambda f: f.value()[0])
        >>> chained.value()
        tensor(...)
    """

    def __init__(self, *, _completer: Optional[Callable[[], Any]] = None) -> None:
        self._cond = threading.Condition()
        self._done = False
        self._result: Any = None
        self._callbacks: List[Callable[["Future"], Any]] = []
        # Lazily-invoked producer that must make this future complete
        # (e.g. synchronize the underlying Work) before value() returns.
        self._completer = _completer

    def done(self) -> bool:
        """Returns whether the future is complete (non-blocking)."""
        return self._done

    def is_done(self) -> bool:
        """Alias of :meth:`done`."""
        return self.done()

    def _resolve(self) -> None:
        if self._done:
            return
        if self._completer is not None:
            completer, self._completer = self._completer, None
            completer()

    def set_result(self, result: Any) -> None:
        with self._cond:
            if self._done:
                raise RuntimeError("Future is already done")
            self._result = result
            self._done = True
            self._cond.notify_all()
        for cb in list(self._callbacks):
            cb(self)

    def wait(self) -> Any:
        """Blocks until complete and returns the result value."""
        with self._cond:
            if not self._done:
                self._resolve()
            while not self._done:
                self._cond.wait()
            return self._result

    def value(self) -> Any:
        """Gets the result value, blocking until it completes.

        ``ValueError``-on-error behavior loosely; tp stores exceptions as
        results and re-raises them here).
        """
        result = self.wait()
        if isinstance(result, BaseException):
            raise result
        return result

    def then(self, callback: Callable[["Future"], Any]) -> "Future":
        """Adds a callback mapped over this future; returns the new future.

        The callback receives *this* future and its return value becomes the
        the derived future drives *this* future (and therefore the underlying
        through ``.then()`` chains.
        """

        def _run_cb(_fut: "Future") -> None:
            try:
                derived.set_result(callback(self))
            except BaseException as e:  # propagate into the derived future
                derived.set_result(e)

        derived = Future()
        # Drive the base future to completion when the derived future is
        # first waited on. Without this the lazily-invoked base completer
        # (e.g. the CUDA-event sync behind a collective Work) would never
        # run and the derived future would block forever.
        derived._completer = self.wait
        self.add_done_callback(_run_cb)
        return derived

    def add_done_callback(self,
                          callback: Callable[["Future"], Any]) -> None:
        """Appends a callback run when this future completes."""
        with self._cond:
            if not self._done:
                self._callbacks.append(callback)
                should_run = False
            else:
                should_run = True
        if should_run:
            callback(self)
