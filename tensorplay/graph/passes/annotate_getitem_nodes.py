"""Record useful producer and index metadata on getitem nodes."""

from __future__ import annotations

import operator
import typing
from typing import Any

from ..graph import Graph

__all__ = ["annotate_getitem_nodes"]


def annotate_getitem_nodes(graph: Graph) -> None:
    """Annotate each indexed result with its source node and index value."""

    for node in graph.nodes:
        if node.op != "call_function" or node.target is not operator.getitem:
            continue
        if len(node.args) < 2:
            continue
        producer, index = node.args[:2]
        if not isinstance(producer, type(node)) or producer.type is None:
            continue
        producer_type = producer.type
        args = typing.get_args(producer_type)
        origin = typing.get_origin(producer_type)
        if origin is tuple or getattr(producer_type, "_name", None) == "Tuple":
            if len(args) == 2 and args[1] is Ellipsis:
                node.type = args[0]
            elif isinstance(index, int) and index < len(args):
                node.type = args[index]
        elif origin is list or getattr(producer_type, "_name", None) == "List":
            if len(args) == 1:
                node.type = args[0]
        elif hasattr(producer_type, "__annotations__") and isinstance(index, str):
            node.type = producer_type.__annotations__.get(index)
        if node.type is not None:
            node.meta["getitem_index"] = index
