"""TensorPlay compiler frontend and backend registry."""

from .api import compile, is_compiling, reset
from .graph import Graph, GraphCaptureError, GraphModule, Node, Proxy, Tracer, gate
from .guards import Guard, GuardChain, format_recompile_reasons  # noqa: F401
from .passes import (
    ConstFold,
    DeadCodeElimination,
    PassBase,
    PassManager,
    PassResult,
    ShapeProp,
)
from .registry import (
    get_default_backend,
    list_backends,
    lookup_backend,
    register_backend,
    register_debug_backend,
    register_experimental_backend,
    set_default_backend,
    unregister_backend,
)

__all__ = [
    "ConstFold",
    "DeadCodeElimination",
    "Graph",
    "GraphCaptureError",
    "GraphModule",
    "Guard",
    "GuardChain",
    "Node",
    "PassBase",
    "PassManager",
    "PassResult",
    "Proxy",
    "ShapeProp",
    "Tracer",
    "compile",
    "format_recompile_reasons",
    "gate",
    "get_default_backend",
    "is_compiling",
    "list_backends",
    "lookup_backend",
    "register_backend",
    "register_debug_backend",
    "register_experimental_backend",
    "reset",
    "set_default_backend",
    "unregister_backend",
]


from .aot import AOTError, build_aot  # noqa: F401

from .decompositions import DecomposePass  # noqa: F401
from .codecache import CodeCache, default_cache  # noqa: F401
from .cudagraphs import CudaGraphError, CudaGraphManager  # noqa: F401
from .fx_passes import NormalizeOperators, PointwiseFusionHint  # noqa: F401
