# mypy: allow-untyped-defs
r"""CUDA graph capture, mirroring :mod:`torch.cuda.graphs`.

Graph capture requires runtime support this TensorPlay build does not expose;
every public name is present and raises a descriptive :class:`RuntimeError`
on use, matching torch's behaviour on builds without CUDA graph support.
"""

import contextlib


__all__ = [
    "CUDAGraph",
    "graph",
    "graph_pool_handle",
    "is_current_stream_capturing",
    "make_graphed_callables",
    "make_graphed_autograd_function",
    "export_dot",
    "export_graph_data",
]


class CUDAGraph:
    r"""Wrapper around a CUDA graph.

    Not supported by this TensorPlay build; instantiation raises.
    """

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "CUDA graphs are not supported by this TensorPlay build"
        )


def is_current_stream_capturing():
    r"""Return True if CUDA graph capture is underway on the current stream.

    Always returns ``False`` when CUDA is not initialized or graph capture is
    unavailable.
    """
    from . import is_initialized

    if not is_initialized():
        return False
    return False


def graph_pool_handle():
    r"""Return an opaque token representing the id of a graph memory pool."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )


def export_dot(file_path: str) -> str:
    r"""Export the last captured CUDA graph to a DOT file."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )


def export_graph_data(graph_id: int) -> dict:
    r"""Serialize a captured CUDA graph into a dictionary of node data."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )


@contextlib.contextmanager
def graph(cuda_graph, pool=None, stream=None, capture_error_mode="global"):
    r"""Context-manager that captures CUDA work into a ``tensorplay.cuda.CUDAGraph``.

    Not supported by this TensorPlay build.

    Args:
        cuda_graph (CUDAGraph): the graph object to capture into.
        pool (optional): an opaque token returned by :func:`graph_pool_handle`.
        stream (Stream, optional): the stream on which to capture.
        capture_error_mode (str, optional): specifies the cudaStreamCaptureMode
            for the capture. Can be "global", "thread_local" or "relaxed".
    """
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )
    yield  # pragma: no cover


def make_graphed_callables(
    callables, sample_args, num_warmup_iters=3, allow_unused_input=False
):
    r"""Callables that run per-iteration with CUDA graph capture (not supported)."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )


def make_graphed_autograd_function(
    fwd_body,
    bwd_body,
    num_warmup_iters,
    fwd_input_mask,
    bwd_input_mask,
    mask_output,
    static_arg_strs=(),
):
    r"""Wrap forward/backward bodies for graph capture (not supported)."""
    raise RuntimeError(
        "CUDA graphs are not supported by this TensorPlay build"
    )
