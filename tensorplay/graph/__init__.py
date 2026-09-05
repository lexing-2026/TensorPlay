"""Graph namespace and feature-extraction helpers.

The modules in this package own graph capture, graph values, tracing, and
execution.  This package also exposes the frontend features that operate on
captured models:

- :func:`get_graph_node_names` / :func:`create_feature_extractor`, the
- :meth:`Graph.to_dot` / :meth:`Graph.draw` visualization helpers;
- :class:`NodePathTracer`, which records the qualified module path behind
  every ``call_module`` node;
- :func:`wrap`, an identity decorator for marking wrapped callables.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from ._utils import (
    GraphCaptureError,
    capture_call,
    capturing,
    compiler_context,
    gate_outcome,
    _iter_proxies,
)
from .graph import CodeGen, Graph
from .graph_module import GraphModule
from .node import Node, has_side_effect, map_arg
from .proxy import Proxy, gate
from .symbolic_trace import symbolic_trace, wrap
from .tracer import NodePathTracer, Tracer
from .annotate import annotate
from .immutable_collections import immutable_dict, immutable_list
from .interpreter import Interpreter, Transformer
from .subgraph_rewriter import (
    Match,
    ReplacedPatterns,
    replace_pattern,
    replace_pattern_with_filters,
)
from .tensor_type import Dyn, TensorType, is_consistent, is_more_precise
__all__ = [
    "Dyn",
    "CodeGen",
    "Graph",
    "GraphCaptureError",
    "GraphModule",
    "Interpreter",
    "Match",
    "Node",
    "NodePathTracer",
    "Proxy",
    "ReplacedPatterns",
    "TensorType",
    "Tracer",
    "Transformer",
    "annotate",
    "capture_call",
    "capturing",
    "compiler_context",
    "create_feature_extractor",
    "get_graph_node_names",
    "gate",
    "gate_outcome",
    "has_side_effect",
    "immutable_dict",
    "immutable_list",
    "is_consistent",
    "is_more_precise",
    "map_arg",
    "replace_pattern",
    "replace_pattern_with_filters",
    "symbolic_trace",
    "wrap",
]


ReturnNodes = Union[Iterable[str], Mapping[str, Optional[str]]]


def _as_return_nodes(return_nodes: ReturnNodes) -> Dict[str, str]:
    if isinstance(return_nodes, Mapping):
        mapping = dict(return_nodes)
    else:
        mapping = {name: name for name in return_nodes}
    resolved: Dict[str, str] = {}
    for original_name, new_name in mapping.items():
        resolved[original_name] = new_name if new_name is not None else original_name
    return resolved


def _trace_with_paths(model: Any, tracer_kwargs: Dict[str, Any]) -> Tuple[NodePathTracer, GraphModule]:
    tracer = NodePathTracer(**tracer_kwargs)
    return tracer, tracer.trace(model)


def _selectable_names(tracer: NodePathTracer, graph_module: GraphModule) -> List[str]:
    names = [
        node.name for node in graph_module.graph.nodes if node.op != "output"
    ]
    merged = dict.fromkeys(names)
    for qualname in sorted(tracer.node_to_qualname.values()):
        merged.setdefault(qualname, None)
    return sorted(merged)


def _get_node_names_once(model: Any, tracer_kwargs: Dict[str, Any]) -> List[str]:
    tracer, graph_module = _trace_with_paths(model, tracer_kwargs)
    return _selectable_names(tracer, graph_module)


def get_graph_node_names(
    model: Any,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
    suppress_warning: bool = False,
) -> Tuple[List[str], List[str]]:
    """List selectable node names for :func:`create_feature_extractor`.

    The model is traced twice - once in train mode, once in eval mode - and
    each trace contributes one sorted list.  A name may be either a semantic
    node name (``relu_1``) or the qualified path of a leaf module
    (``layer1.0.conv1``), so callers can pick whichever reads better.
    """

    tracer_kwargs = dict(tracer_kwargs or {})
    was_training = model.training
    try:
        model.eval()
        eval_names = _get_node_names_once(model, tracer_kwargs)
        model.train()
        train_names = _get_node_names_once(model, tracer_kwargs)
    finally:
        model.train(was_training)
    if train_names != eval_names and not suppress_warning:
        warnings.warn(
            "The graphs traced in train and eval mode differ; pass "
            "mode-specific return_nodes to create_feature_extractor() for "
            "exact cuts.",
            stacklevel=2,
        )
    return train_names, eval_names


def _resolve_return_nodes(
    tracer: NodePathTracer,
    graph_module: GraphModule,
    requested: Mapping[str, str],
) -> List[Node]:
    nodes_by_name: Dict[str, Node] = {}
    nodes_by_qualname: Dict[str, Node] = {}
    for node in graph_module.graph.nodes:
        nodes_by_name[node.name] = node
        qualname = tracer.node_to_qualname.get(node)
        if qualname is not None:
            nodes_by_qualname.setdefault(qualname, node)

    resolved: List[Node] = []
    seen_outputs: Dict[str, str] = {}
    for original_name, new_name in requested.items():
        if new_name in seen_outputs:
            raise ValueError(
                f"Two return nodes would be emitted under '{new_name}' "
                f"('{seen_outputs[new_name]}' and '{original_name}')"
            )
        seen_outputs[new_name] = original_name
        node = nodes_by_name.get(original_name)
        if node is None:
            node = nodes_by_qualname.get(original_name)
        if node is None:
            available = "\n".join(
                sorted(set(nodes_by_name) | set(nodes_by_qualname))
            )
            raise ValueError(
                f"Requested return node '{original_name}' is not present in "
                f"the graph. Available nodes:\n{available}"
            )
        if node.op in ("placeholder", "output"):
            raise ValueError(
                f"'{original_name}' refers to a {node.op} node and cannot be "
                "used as a feature output"
            )
        resolved.append(node)
    return resolved


def _pruned_graph_copy(
    tracer: NodePathTracer,
    graph_module: GraphModule,
    requested: Mapping[str, str],
    *,
    suppress_warning: bool,
) -> Graph:
    graph = graph_module.graph
    selected = _resolve_return_nodes(tracer, graph_module, requested)
    output_value = selected[0] if len(selected) == 1 else tuple(selected)
    graph.output(output_value)
    removed = graph.eliminate_dead_code()
    if removed and not suppress_warning:
        warnings.warn(
            f"{removed} operation(s) that are not used to compute the "
            "requested outputs have been dropped from the extracted graph.",
            stacklevel=3,
        )
    graph.lint()
    return graph


def _split_target(target: str) -> Tuple[List[str], str]:
    parts = target.split(".")
    return parts[:-1], parts[-1]


def _get_by_path(root: Any, target: str) -> Any:
    value = root
    for part in target.split("."):
        value = getattr(value, part)
    return value


def _register_at_path(holder: Any, target: str, value: Any) -> None:
    from tensorplay import nn as _nn
    from tensorplay.nn.parameter import Parameter

    parent_parts, leaf_name = _split_target(target)
    container = holder
    for part in parent_parts:
        existing = container._modules.get(part)
        if existing is None:
            existing = _nn.Module()
            container.add_module(part, existing)
        container = existing
    if isinstance(value, Parameter):
        container.register_parameter(leaf_name, value)
    elif isinstance(value, _nn.Module):
        container.add_module(leaf_name, value)
    else:
        container.register_buffer(leaf_name, value)


def create_feature_extractor(
    model: Any,
    return_nodes: Optional[ReturnNodes] = None,
    *,
    train_return_nodes: Optional[ReturnNodes] = None,
    eval_return_nodes: Optional[ReturnNodes] = None,
    suppress_warning: bool = False,
    tracer_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Build a module returning intermediate graph nodes of ``model``.

    This is the counterpart of
    Use :func:`get_graph_node_names` to discover selectable names, then pass
    either an iterable of names or a ``{name: new_output_name}`` mapping::

        extractor = create_feature_extractor(
            model, return_nodes={"layer1.0.relu": "feat1", "avgpool": "pooled"}
        )
        features = extractor(images)

    The returned module keeps two pruned graphs (one for train mode, one for
    eval mode) and switches between them through ``.train()`` / ``.eval()``.
    Leaf submodules feeding the selected nodes are deep-copied under their
    original qualified paths, so ``state_dict()`` remains compatible with the
    source model for loading pretrained weights.

    Args:
        model: model to extract features from.
        return_nodes: names (or mapping) selecting the graph nodes to keep.
        train_return_nodes / eval_return_nodes: mode-specific selection;
            exactly one of ``return_nodes`` or this pair must be provided.
        suppress_warning: silence dead-code and train/eval mismatch warnings.
        tracer_kwargs: keyword arguments forwarded to :class:`NodePathTracer`
            (e.g. ``concrete_args``).
    """

    from tensorplay import nn as _nn

    if return_nodes is not None:
        if train_return_nodes is not None or eval_return_nodes is not None:
            raise ValueError(
                "Pass either return_nodes or both train_return_nodes and "
                "eval_return_nodes, not both forms"
            )
        train_return_nodes = eval_return_nodes = return_nodes
    elif train_return_nodes is None or eval_return_nodes is None:
        raise ValueError(
            "Pass either return_nodes or both train_return_nodes and "
            "eval_return_nodes"
        )

    requested_train = _as_return_nodes(train_return_nodes)
    requested_eval = _as_return_nodes(eval_return_nodes)
    tracer_kwargs = dict(tracer_kwargs or {})

    was_training = model.training
    try:
        model.eval()
        eval_tracer, eval_gm = _trace_with_paths(model, tracer_kwargs)
        model.train()
        train_tracer, train_gm = _trace_with_paths(model, tracer_kwargs)
    finally:
        model.train(was_training)

    train_graph = _pruned_graph_copy(
        train_tracer, train_gm, requested_train, suppress_warning=suppress_warning
    )
    eval_graph = _pruned_graph_copy(
        eval_tracer, eval_gm, requested_eval, suppress_warning=suppress_warning
    )
    signature = train_gm.signature

    class DualFeatureExtractor(_nn.Module):
        """Module executing the pruned graphs, switching on training mode."""

        def __init__(self) -> None:
            super().__init__()
            self._extractor_graphs = {"train": train_graph, "eval": eval_graph}
            self._extractor_signature = signature
            self._extractor_executors: Dict[str, GraphModule] = {}
            self._copy_required_state(model)

        def _copy_required_state(self, root: Any) -> None:
            targets = set()
            for graph in self._extractor_graphs.values():
                for node in graph.nodes:
                    if node.op in ("call_module", "get_attr"):
                        targets.add(node.target)
            for target in sorted(targets):
                _register_at_path(self, target, copy.deepcopy(_get_by_path(root, target)))

        def _executor(self) -> GraphModule:
            mode = "train" if self.training else "eval"
            executor = self._extractor_executors.get(mode)
            if executor is None:
                executor = GraphModule(
                    self,
                    self._extractor_graphs[mode],
                    self._extractor_signature,
                )
                self._extractor_executors[mode] = executor
            return executor

        @property
        def graph(self) -> Graph:
            """Active (mode-dependent) pruned graph."""

            return self._executor().graph

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self._executor()(*args, **kwargs)

    extractor = DualFeatureExtractor()
    extractor.train(mode=was_training)
    return extractor
