#!/usr/bin/env python3

"""Graph-mode utilities that leverage the graph capture stack.

Unlike black-box attribution, which only sees the forward function's inputs and
outputs, the helpers here capture the model's forward pass into a
:class:`~tensorplay.graph.GraphModule` once. The traced module is then reused
for every forward and backward evaluation, which enables:

- trace-once execution: repeated attribution calls (e.g. per-sample
  explanations) skip re-tracing;
- graph introspection: node names and intermediate values are available to
  layer/neuron attribution without requiring a module object reference;
- a foundation for graph rewriting (e.g. gradient-modifying methods) and
  compiled execution on top of the traced graph.
"""

from typing import Callable, List, Optional, Tuple, Union

import tensorplay as tensorplay
from tensorplay import Tensor
from tensorplay.graph import GraphModule, get_graph_node_names, symbolic_trace

from tensorplay.lumen._utils.typing import TargetType


def trace(forward_func: Callable) -> Callable:
    """Trace ``forward_func`` into a :class:`~tensorplay.graph.GraphModule`.

    Returns the input unchanged when tracing is not supported (e.g. models with
    data-dependent control flow or custom Python operations that cannot be
    symbolically traced). The traced module keeps the same call signature as
    the original forward function.
    """
    if isinstance(forward_func, GraphModule):
        return forward_func
    try:
        return symbolic_trace(forward_func)
    except Exception:
        return forward_func


def node_names(model: Callable) -> List[str]:
    """Return the node names of the traced forward graph.

    Works on any model that can be symbolically traced. The returned list
    contains the module-call node names (``forward``, ``fc1``, ...) which can
    be used to select intermediate layers for layer attribution.
    """
    return get_graph_node_names(model)[0]


def graph_gradient_func(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
) -> Tuple[Tensor, ...]:
    """A drop-in ``gradient_func`` that traces the forward function once and
    computes gradients on the traced :class:`GraphModule`.

    The forward function is traced into a :class:`GraphModule` on the first
    call; later calls reuse the traced module, skipping re-tracing and making
    the computation introspectable by the graph stack.
    """
    from tensorplay.lumen._utils.gradient import compute_gradients

    traced_forward = trace(forward_fn)
    return compute_gradients(
        traced_forward,
        inputs,
        target_ind=target_ind,
        additional_forward_args=additional_forward_args,
    )