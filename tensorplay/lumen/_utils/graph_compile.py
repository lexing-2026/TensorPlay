#!/usr/bin/env python3

"""Graph-level helpers for the lumen package.

These helpers build on the graph capture stack:

- :func:`compile_forward` traces a forward function once and compiles the
  resulting graph, returning a callable that is reused for every subsequent
  forward pass. If tracing or compilation is not possible, the original forward
  function is returned unchanged.
- :func:`rewrite_activation` replaces every ReLU node in a traced graph with a
  custom autograd function, which is how GuidedBackprop and Deconvolution are
  implemented without registering backward hooks on the user's modules.
"""

from typing import Any, Callable, Dict, Optional, Set, Tuple, Union

import tensorplay as tensorplay
from tensorplay.graph import GraphModule, symbolic_trace

# Targets that represent a ReLU activation in a traced graph.
_RELU_TARGETS: Set[Any] = {tensorplay.relu, tensorplay.nn.functional.relu}

_CACHED_COMPILED: Dict[int, Callable] = {}


def _is_tensorplay_module(forward_fn: Callable) -> bool:
    return isinstance(forward_fn, tensorplay.nn.Module)


def compile_forward(forward_fn: Callable) -> Callable:
    """Compile a forward function for repeated use.

    The forward function is traced into a :class:`GraphModule` and then
    compiled with :func:`tensorplay.compiler.compile`. The compiled callable is
    cached per module instance so repeated attributions reuse it.

    If tracing or compilation raises, the original ``forward_fn`` is returned
    so callers keep working without the graph backend. A warning is emitted in
    that case.
    """
    if not _is_tensorplay_module(forward_fn):
        return forward_fn

    module_id = id(forward_fn)
    cached = _CACHED_COMPILED.get(module_id)
    if cached is not None:
        return cached

    try:
        traced = symbolic_trace(forward_fn)
        compiled = tensorplay.compiler.compile(traced)
        _CACHED_COMPILED[module_id] = compiled
        return compiled
    except Exception as e:  # pragma: no cover - backend dependent
        import warnings

        warnings.warn(
            "Could not compile the forward function, falling back to eager "
            "execution. Reason: {}".format(e),
            stacklevel=2,
        )
        return forward_fn


def rewrite_activation(
    graph_module: GraphModule,
    autograd_fn: type,
) -> GraphModule:
    """Replace ReLU call nodes with ``autograd_fn.apply``.

    Every node in ``graph_module`` whose target is a ReLU function is replaced
    by a call to the given :class:`~tensorplay.autograd.Function` subclass. The
    modified graph is recompiled and returned (the input module is not mutated).
    """
    graph = graph_module.graph
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in _RELU_TARGETS:
            with graph.inserting_before(node):
                new_node = graph.call_function(
                    autograd_fn.apply, node.args, node.kwargs
                )
            node.replace_all_uses_with(new_node)
            graph.erase_node(node)
    graph_module.recompile()
    return graph_module


def trace_with_rewritten_relu(
    forward_fn: Callable,
    autograd_fn: type,
) -> Optional[GraphModule]:
    """Trace ``forward_fn`` and rewrite its ReLU nodes.

    Returns the rewritten :class:`GraphModule`, or ``None`` if tracing fails
    (e.g. the model contains data-dependent control flow).
    """
    try:
        traced = symbolic_trace(forward_fn)
        return rewrite_activation(traced, autograd_fn)
    except Exception:
        return None


__all__ = [
    "compile_forward",
    "rewrite_activation",
    "trace_with_rewritten_relu",
]