"""tensorplay.multiprocessing is a wrapper around the native :mod:`multiprocessing` module.

It registers custom reducers, that use shared memory to provide shared
views on the same data in different processes. Once the tensor is moved
to shared_memory (see :func:`~tensorplay.Tensor.share_memory_`), it will be possible
to send it to other processes without making any copies.

The API is 100% compatible with the original module - it's enough to change
``import multiprocessing`` to ``import tensorplay.multiprocessing`` to have all the
tensors sent through the queues or shared via other mechanisms, moved to shared
memory.

Because of the similarity of APIs we do not document most of this package
contents, and we recommend referring to the docs of the original module.
"""

import ctypes
import multiprocessing
import sys

import tensorplay

from .reductions import init_reductions


__all__ = ["set_sharing_strategy", "get_sharing_strategy", "get_all_sharing_strategies"]


from multiprocessing import *  # noqa: F403


__all__ += multiprocessing.__all__


# Linux-specific prctl(2) wrapper: ask the kernel to deliver a signal to this
# process when its parent dies, so that non-daemonic children spawned with the
# prctl(2) marker cannot outlive a crashed parent.
_PR_SET_PDEATHSIG = 1


def _prctl_pr_set_pdeathsig(signal_value) -> None:
    """Set the parent-death signal of the calling process (best effort).

    Args:
        signal_value: Signal number to deliver when the parent terminates.
            Uses the Linux prctl(2) ``PR_SET_PDEATHSIG`` operation; a no-op on
            platforms without it.
    """
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, int(signal_value), 0, 0, 0)
    except OSError:
        pass


"""Add helper functions to name the calling thread (best effort)."""


def _set_thread_name(name: str) -> None:
    """Set the name of the calling thread.

    Args:
        name (str): Name to assign; the kernel keeps at most 16 bytes
            (including the terminating null) of it.
    """
    if sys.platform != "linux":
        return
    try:
        with open("/proc/thread-self/comm", "w") as comm:
            comm.write(name[:15])
    except OSError:
        pass


def _get_thread_name() -> str | None:
    """Get the name of the calling thread.

    Returns:
        str | None: Name of the calling thread, or ``None`` if unavailable.
    """
    if sys.platform != "linux":
        return None
    try:
        with open("/proc/thread-self/comm", "r") as comm:
            return comm.read().strip()
    except OSError:
        return None


if sys.platform == "darwin" or sys.platform == "win32":
    _sharing_strategy = "file_system"
    _all_sharing_strategies = {"file_system"}
else:
    _sharing_strategy = "file_descriptor"
    _all_sharing_strategies = {"file_descriptor", "file_system"}


def set_sharing_strategy(new_strategy):
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    if new_strategy not in _all_sharing_strategies:
        raise AssertionError(
            f"invalid sharing strategy {new_strategy!r}, "
            f"expected one of {_all_sharing_strategies}"
        )
    _sharing_strategy = new_strategy


def get_sharing_strategy():
    """Return the current strategy for sharing CPU tensors."""
    return _sharing_strategy


def get_all_sharing_strategies():
    """Return a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies


"""Add helper function to spawn N processes and wait for completion of any of
them."""
from .spawn import (  # noqa: E402
    ENV_VAR_PARALLEL_START,
    ProcessContext,
    ProcessExitedException,
    ProcessRaisedException,
    spawn,
    start_processes,
)


init_reductions()

# Leak ResourceTracker at exit for Python-3.12 on MacOS
from multiprocessing.resource_tracker import ResourceTracker as _RT  # noqa: E402


if (
    sys.platform == "darwin"
    and sys.version_info >= (3, 12, 2)
    and hasattr(_RT, "__del__")
):
    import atexit

    def _leak_RT_at_exit():
        def _noop(x):
            pass

        _RT.__del__ = _noop  # type: ignore[attr-defined]

    atexit.register(_leak_RT_at_exit)
