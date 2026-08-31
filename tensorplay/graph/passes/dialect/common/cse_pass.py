"""Common subexpression elimination for graph dialects."""

from __future__ import annotations

from typing import Any

from ...._pytree import tree_flatten
from ....graph import Graph
from ....graph_module import GraphModule
from ....node import Node
from ...infra.pass_base import PassBase, PassResult

__all__ = ["CSEPass", "get_CSE_banned_ops"]


def _lookup_public_ops(names: tuple[str, ...]) -> set[Any]:
    """Resolve operation objects lazily so importing graph passes stays cheap."""

    import tensorplay

    result: set[Any] = set()
    for name in names:
        value = getattr(tensorplay, name, None)
        if value is not None:
            result.add(value)
    try:
        from tensorplay.nn import functional
    except ImportError:
        functional = None
    if functional is not None:
        for name in names:
            value = getattr(functional, name, None)
            if value is not None:
                result.add(value)
    return result


def get_CSE_banned_ops() -> set[Any]:
    """Return random and stateful operations that must not be merged."""

    return _lookup_public_ops(
        (
            "dropout",
            "_fused_dropout",
            "_standard_gamma",
            "bernoulli",
            "multinomial",
            "native_dropout",
            "normal",
            "poisson",
            "binomial",
            "rrelu",
            "rand_like",
            "rand",
            "randint",
            "randn",
            "randperm",
            "add_",
            "sub_",
            "mul_",
            "div_",
            "pow_",
            "lerp_",
            "relu_",
            "sigmoid_",
            "tanh_",
        )
    )


def _target_without_overload(target: Any) -> Any:
    return getattr(target, "overloadpacket", target)


def _is_mutating_target(node: Node) -> bool:
    if node.op == "call_method":
        target = node.target
        return isinstance(target, str) and target.endswith("_")
    if node.op == "call_function":
        name = getattr(node.target, "__name__", None)
        return isinstance(name, str) and name.endswith("_")
    return False


class CSEPass(PassBase):
    """Merge equivalent pure calls while preserving graph connectivity."""

    def __init__(self, banned_ops: set[Any] | None = None) -> None:
        self.banned_ops = set() if banned_ops is None else banned_ops

    def call(self, graph_module: GraphModule) -> PassResult:
        modified = False
        new_graph = Graph()
        env: dict[Node, Node] = {}
        hash_env: dict[tuple[Any, int], Node] = {}
        token_map: dict[tuple[Any, int], tuple[Any, ...]] = {}

        for node in graph_module.graph.nodes:
            target = _target_without_overload(node.target)
            if (
                node.op in {"placeholder", "output", "get_attr", "call_module"}
                or target in self.banned_ops
                or _is_mutating_target(node)
            ):
                new_node = new_graph.node_copy(node, lambda value: env[value])
                env[node] = new_node
                continue

            if node.op not in {"call_function", "call_method"}:
                new_node = new_graph.node_copy(node, lambda value: env[value])
                env[node] = new_node
                continue

            def substitute(value: Any) -> tuple[tuple[Any, ...], Any]:
                values, spec = tree_flatten(value)
                for index, item in enumerate(values):
                    if isinstance(item, Node) and item in env:
                        values[index] = env[item]
                return tuple(values), spec

            args, args_spec = substitute(node.args)
            kwargs, kwargs_spec = substitute(node.kwargs)
            token = (node.target, args, args_spec, kwargs, kwargs_spec)
            try:
                hash_arg = hash((args, kwargs))
                hash_val = (node.target, hash_arg)
                previous = hash_env.get(hash_val)
                if previous is not None and token_map[hash_val] == token:
                    env[node] = previous
                    modified = True
                    continue
            except TypeError:
                hash_val = None

            new_node = new_graph.node_copy(node, lambda value: env[value])
            env[node] = new_node
            if hash_val is not None and hash_val not in hash_env:
                hash_env[hash_val] = new_node
                token_map[hash_val] = token

        if not modified:
            return PassResult(graph_module, False)
        cse_graph_module = GraphModule(
            graph_module.root, new_graph, graph_module.signature
        )
        cse_graph_module.meta.update(graph_module.meta)
        cse_graph_module._graph_attrs.update(graph_module._graph_attrs)
        return PassResult(cse_graph_module, True)
