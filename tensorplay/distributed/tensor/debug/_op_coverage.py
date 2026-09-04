from __future__ import annotations

import csv
from collections import Counter
from operator import itemgetter
from typing import Any

from .._api import DTensor

__all__ = ["fwd_bwd_compiler", "get_inductor_decomp_graphs", "print_op_coverage_summary"]


graphs: list[Any] = []


def fwd_bwd_compiler(graph_module: Any, _example_inputs: Any = None) -> Any:
    graphs.append(graph_module)
    return graph_module


def _graph_nodes(graph_module: Any) -> list[Any]:
    graph = getattr(graph_module, "graph", graph_module)
    nodes = getattr(graph, "nodes", ())
    return list(nodes)


def _capture_graph(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    from tensorplay.graph import symbolic_trace

    del args, kwargs
    return symbolic_trace(model)


def get_inductor_decomp_graphs(
    model: Any, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None
) -> list[Any]:
    graphs.clear()
    try:
        graph_module = _capture_graph(model, args, kwargs or {})
    except (ImportError, RuntimeError, TypeError, ValueError):
        graph_module = getattr(model, "graph", None)
    if graph_module is not None:
        fwd_bwd_compiler(graph_module, args)
    return list(graphs)


def _target_name(target: Any) -> str:
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name
    qualname = getattr(target, "__qualname__", None)
    if isinstance(qualname, str):
        return qualname
    return str(target)


def _supported(target: Any) -> bool:
    propagator = DTensor._op_dispatcher.sharding_propagator
    if target in propagator.op_to_rules:
        return True
    try:
        _kind, rule = propagator._global_rule(target)
    except (AttributeError, TypeError, ValueError):
        rule = None
    if rule is not None:
        return True
    name = _target_name(target)
    return any(_target_name(operation) == name for operation in propagator.op_to_rules)


def print_op_coverage_summary(
    model: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    output_csv: bool = False,
) -> list[tuple[str, str, int, bool]]:
    op_counts: Counter[Any] = Counter()
    for graph_module in get_inductor_decomp_graphs(model, args, kwargs):
        for node in _graph_nodes(graph_module):
            if getattr(node, "op", None) != "call_function":
                continue
            target = getattr(node, "target", None)
            op_counts[target] += 1

    rows = [
        (
            _target_name(target),
            str(getattr(target, "_schema", "")),
            count,
            _supported(target),
        )
        for target, count in op_counts.items()
    ]
    rows.sort(key=itemgetter(2), reverse=True)
    headers = ("Operator", "Schema", "Total Count", "Supported")
    try:
        from tabulate import tabulate

        print(tabulate(rows, headers=headers))
    except ImportError:
        print(" | ".join(headers))
        for row in rows:
            print(" | ".join(str(value) for value in row))
    if output_csv:
        with open("op_summary.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers)
            writer.writerows(rows)
    return rows
