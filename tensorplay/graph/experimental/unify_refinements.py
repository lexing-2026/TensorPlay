from __future__ import annotations

from typing import Any

from ..graph import Graph
from ..graph_module import GraphModule
from ..tensor_type import TensorType
from .graph_gradual_typechecker import Refine
from .refinement_types import Equality
from .unification import Var, unify

__all__ = [
    "check_for_type_equality",
    "convert_eq",
    "infer_symbolic_types",
    "infer_symbolic_types_single_pass",
    "substitute_all_types",
    "substitute_solution_one_type",
    "unify_eq",
]


def convert_eq(equalities: list[Equality]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    return tuple(item.lhs for item in equalities), tuple(item.rhs for item in equalities)


def unify_eq(equalities: list[Equality]) -> dict[Any, Any] | bool:
    left, right = convert_eq(equalities)
    return unify(left, right)


def substitute_solution_one_type(mapping: dict[object, object], value: object) -> Any:
    if isinstance(value, Var):
        return mapping.get(value, value)
    if isinstance(value, TensorType):
        return TensorType(
            tuple(substitute_solution_one_type(mapping, item) for item in value.dims)
        )
    if isinstance(value, list):
        return [substitute_solution_one_type(mapping, item) for item in value]
    if isinstance(value, tuple):
        return tuple(substitute_solution_one_type(mapping, item) for item in value)
    if isinstance(value, dict):
        return {
            key: substitute_solution_one_type(mapping, item)
            for key, item in value.items()
        }
    return value


def substitute_all_types(graph: Graph, mapping: dict[object, object] | bool) -> None:
    if mapping is False:
        return
    substitutions = dict(mapping)
    changed = True
    while changed:
        changed = False
        for key, value in tuple(substitutions.items()):
            replacement = substitute_solution_one_type(substitutions, value)
            if replacement != value:
                substitutions[key] = replacement
                changed = True
    for node in graph.nodes:
        if node.type is not None:
            node.type = substitute_solution_one_type(substitutions, node.type)


def infer_symbolic_types_single_pass(traced: GraphModule) -> None:
    refinement = Refine(traced)
    refinement.refine()
    substitute_all_types(traced.graph, unify_eq(refinement.constraints))


def infer_symbolic_types(traced: GraphModule) -> None:
    infer_symbolic_types_single_pass(traced)
    infer_symbolic_types_single_pass(traced)
    Refine(traced).symbolic_relations()


def check_for_type_equality(first: Graph, second: Graph) -> bool:
    return all(left.type == right.type for left, right in zip(first.nodes, second.nodes))
