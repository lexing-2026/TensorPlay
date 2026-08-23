# mypy: allow-untyped-defs
r"""CUDA graph capture, mirroring :mod:`torch.cuda.graphs`.

Backed by the native graph bindings on ``tensorplay._C`` (capture on a
dedicated side stream, graph-private allocator pool, replay against static
buffers).  On builds without CUDA the same names exist and raise a
descriptive :class:`RuntimeError` on use, matching torch's behaviour.

Typical use follows torch.cuda.graph::

    g = tensorplay.cuda.CUDAGraph()
    static_input = x.clone()
    # warm up on the capture stream so lazy state lands outside the capture
    with tensorplay.cuda.graph(g):
        static_output = model(static_input)
    # ...later, refresh inputs and re-run without host-side launch overhead:
    static_input.copy_(new_batch)
    g.replay()

Static input/output tensors must stay alive for as long as the graph; their
addresses are baked into the executable.  Random-number ops inside a capture
replay identical values (RNG offset prologue/epilogue is not implemented).
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


def _native():
    try:
        from .. import _C
    except Exception as exc:  # pragma: no cover - import failure diagnostics
        raise RuntimeError(
            "CUDA graphs require tensorplay._C; import failed: "
            f"{exc!r}. Was TensorPlay built with CUDA support?"
        ) from exc
    required = (
        "cuda_graph_begin_capture",
        "cuda_graph_end_capture",
        "cuda_graph_instantiate",
        "cuda_graph_launch",
    )
    missing = [name for name in required if not hasattr(_C, name)]
    if missing:
        raise RuntimeError(
            "CUDA graphs are not supported by this TensorPlay build "
            f"(missing native symbols: {', '.join(missing)})"
        )
    return _C


class CUDAGraph:
    r"""Wrapper around a CUDA graph, mirroring ``torch.cuda.CUDAGraph``."""

    def __init__(self):
        self._native = _native()
        self._handle = None

    def capture_begin(self):
        if self._handle is not None:
            raise RuntimeError(
                "this CUDAGraph already holds a capture; create a new one"
            )
        # The native side owns the single live-capture slot; handles attach
        # to it at capture_end.
        self._native.cuda_graph_begin_capture()

    def capture_end(self):
        if self._handle is not None:
            raise RuntimeError("capture_end called twice on one CUDAGraph")
        self._handle = self._native.cuda_graph_end_capture()

    def instantiate(self):
        """Compile the captured template (happens automatically at replay)."""

        if self._handle is None:
            raise RuntimeError("capture_end must run before instantiate")
        self._native.cuda_graph_instantiate(self._handle)

    def replay(self):
        if self._handle is None:
            raise RuntimeError(
                "cannot replay before completing a capture"
            )
        self._native.cuda_graph_instantiate(self._handle)
        self._native.cuda_graph_launch(self._handle)

    def reset(self):
        """Destroy the executable and free its private memory pool.

        All tensors allocated during the capture must be released first.
        """

        if self._handle is None:
            return
        handle, self._handle = self._handle, None
        self._native.cuda_graph_destroy(handle)

    def __del__(self):
        try:
            self.reset()
        except Exception:
            pass

    def enable_debug_mode(self):
        pass  # debug output not implemented; kept for API parity

    def debug_dump(self, path):  # noqa: ARG002 - path accepted for parity
        raise NotImplementedError(
            "CUDA graph debug dumps are not supported by this build"
        )


@contextlib.contextmanager
def graph(cuda_graph, pool=None, stream=None, capture_error_mode="global"):
    r"""Context-manager that captures CUDA work into a ``tensorplay.cuda.CUDAGraph``.

    Args:
        cuda_graph (CUDAGraph): the graph object to capture into.
        pool: unsupported (each capture gets its own private memory pool).
        stream (Stream, optional): unsupported; capture always runs on the
            dedicated side stream exposed by the runtime.
        capture_error_mode (str, optional): ``"global"`` (default) makes any
            unsafe CUDA call anywhere in the process fail the capture.
            ``"thread_local"`` and ``"relaxed"`` are rejected because the
            native binding always captures in global mode.
    """

    if pool is not None:
        raise NotImplementedError(
            "user-supplied graph memory pools are not supported; every "
            "CUDAGraph owns its pool"
        )
    if stream is not None:
        raise NotImplementedError(
            "custom capture streams are not supported; the runtime dedicates "
            "one side stream per device"
        )
    if capture_error_mode != "global":
        raise ValueError(
            "capture_error_mode must be 'global'; got "
            f"{capture_error_mode!r}"
        )
    cuda_graph.capture_begin()
    try:
        yield cuda_graph
    finally:
        cuda_graph.capture_end()


def graph_pool_handle():
    r"""Return an opaque token representing the id of a graph memory pool."""

    raise NotImplementedError(
        "shared graph memory pools are not supported by this build; each "
        "CUDAGraph owns a private pool"
    )


def is_current_stream_capturing():
    r"""Return True if CUDA graph capture is underway on the current thread."""

    from . import is_initialized

    if not is_initialized():
        return False
    try:
        from .. import _C

        probe = getattr(_C, "cuda_is_capturing", None)
    except ImportError:
        return False
    return bool(probe()) if probe is not None else False


def export_dot(file_path: str) -> str:
    r"""Export the last captured CUDA graph to a DOT file."""
    raise NotImplementedError(
        "CUDA graph DOT export is not supported by this build"
    )


def export_graph_data(graph_id: int) -> dict:
    r"""Serialize a captured CUDA graph into a dictionary of node data."""
    raise NotImplementedError(
        "CUDA graph serialization is not supported by this build"
    )


def make_graphed_callables(
    callables, sample_args, num_warmup_iters=3, allow_unused_input=False
):
    r"""Callables that run per-iteration with CUDA graph capture (not supported)."""
    raise NotImplementedError(
        "make_graphed_callables requires autograd-graph integration that "
        "this build does not provide; use tensorplay.compiler.cudagraphs."
        "CudaGraphManager for inference-style capture/replay"
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
    raise NotImplementedError(
        "make_graphed_autograd_function requires autograd-graph integration "
        "that this build does not provide"
    )
