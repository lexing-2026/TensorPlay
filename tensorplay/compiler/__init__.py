"""TensorPlay compiler frontend and backend registry."""

from .api import compile, is_compiling, reset
from .graph import Graph, GraphCaptureError, GraphModule, Node, Proxy, Tracer
from .registry import (
    get_default_backend,
    list_backends,
    lookup_backend,
    register_backend,
    register_debug_backend,
    register_experimental_backend,
    set_default_backend,
)

__all__ = [
    "Graph",
    "GraphCaptureError",
    "GraphModule",
    "Node",
    "Proxy",
    "Tracer",
    "compile",
    "get_default_backend",
    "is_compiling",
    "list_backends",
    "lookup_backend",
    "register_backend",
    "register_debug_backend",
    "register_experimental_backend",
    "reset",
    "set_default_backend",
]

