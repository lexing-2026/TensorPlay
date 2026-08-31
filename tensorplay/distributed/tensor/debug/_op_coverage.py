"""Operator coverage summaries for a callable model."""

from __future__ import annotations

from collections import Counter
from typing import Any

__all__ = ["get_inductor_decomp_graphs", "print_op_coverage_summary"]


def get_inductor_decomp_graphs(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> list[Any]:
    result = model(*args, **(kwargs or {}))
    graph = getattr(result, "graph", None)
    return [graph] if graph is not None else []


def print_op_coverage_summary(model: Any, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None, *, output_csv: bool = False) -> list[tuple[str, int]]:
    graphs = get_inductor_decomp_graphs(model, args, kwargs)
    counts: Counter[str] = Counter()
    for graph in graphs:
        for node in getattr(graph, "nodes", ()):
            counts[str(getattr(node, "target", node))] += 1
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if output_csv:
        import csv

        with open("op_summary.csv", "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("operator", "count"))
            writer.writerows(rows)
    return rows
