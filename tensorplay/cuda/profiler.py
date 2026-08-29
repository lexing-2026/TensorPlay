# mypy: allow-untyped-defs

import contextlib

from . import check_error, cudart


__all__ = ["start", "stop", "profile"]


def start():
    r"""Starts cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to start the profiler.
    """
    rt = cudart()
    if rt is None:
        raise RuntimeError("cudart is not available in this TensorPlay build")
    check_error(rt.cudaProfilerStart())


def stop():
    r"""Stops cuda profiler data collection.

    .. warning::
        Raises CudaError in case of it is unable to stop the profiler.
    """
    rt = cudart()
    if rt is None:
        raise RuntimeError("cudart is not available in this TensorPlay build")
    check_error(rt.cudaProfilerStop())


@contextlib.contextmanager
def profile():
    r"""
    Enable profiling.

    Context Manager to enabling profile collection by the active profiling tool from CUDA backend.
    Example:
        >>> import tensorplay as tp
        >>> model = tp.nn.Linear(20, 30).cuda()
        >>> inputs = tp.randn(128, 20).cuda()
        >>> with tp.cuda.profiler.profile():
        ...     model(inputs)
    """
    try:
        start()
        yield
    finally:
        stop()
