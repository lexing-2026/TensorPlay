# mypy: allow-untyped-defs
"""Device-agnostic capture/replay graphs for the current accelerator.

``Graph`` records a sequence of operations on the active accelerator
backend and replays it with reduced launch overhead.
"""

from typing import Any

from tensorplay.cuda.graphs import CUDAGraph

__all__ = ["Graph"]


class Graph(CUDAGraph):
    """Capture/replay graph on the current accelerator device.

    Args:
        keep_graph: accepted for generic code; the executable is
            compiled when capture ends in this backend.
        pool: ``None`` captures into a fresh private pool, otherwise a
            pool id from :func:`graph_pool_handle`, another graph, or
            another graph's pool id shares that pool.
        capture_error_mode: ``"global"`` fails the capture on unsafe
            calls anywhere in the process, ``"thread_local"`` only
            watches this thread, ``"relaxed"`` skips the guards.
    """

    def __init__(
        self,
        keep_graph: bool = False,
        *,
        pool: Any = None,
        capture_error_mode: str = "global",
    ) -> None:
        super().__init__()
        self.keep_graph = keep_graph
        self.graph_pool = pool
        self.capture_error_mode = capture_error_mode

    def capture_begin(
        self,
        pool: Any = "__unset__",
        capture_error_mode: Any = "__unset__",
        stream: Any = None,
    ) -> None:
        """Begin capture on the current stream with the stored settings."""
        super().capture_begin(
            self.graph_pool if pool == "__unset__" else pool,
            self.capture_error_mode
            if capture_error_mode == "__unset__"
            else capture_error_mode,
            stream,
        )

    def pool(self):
        """Opaque id of this graph's memory pool, shareable with others."""
        return self.pool_id

    def __enter__(self):
        from . import empty_cache, empty_host_cache, synchronize

        synchronize()
        empty_cache()
        empty_host_cache()
        self.capture_begin()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.capture_end()
