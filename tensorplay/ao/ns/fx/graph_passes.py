"""Graph passes for cleaning up quantize/dequantize pairs.

These passes operate on ``tensorplay.graph.GraphModule`` instances produced by
the graph-mode quantization workflow.  Removing adjacent quantize/dequantize
pairs is useful before shadowing a float model with a quantized one, so that
the numeric comparison does not count the round trip as a layer.
"""

from __future__ import annotations

import tensorplay as tp
from tensorplay.graph import GraphModule
from tensorplay.graph.node import Node

__all__ = [
    "remove_quant_dequant_pairs",
    "remove_quantize_dequantize_pairs",
    "add_quantize_dequantize_pairs",
]


def _is_dequantize(node: Node) -> bool:
    if node.op == "call_method":
        return node.target in ("dequantize", "dequantize_")
    if node.op == "call_module":
        return node.target.endswith("dequantize") or "DeQuantize" in node.target
    return False


def _is_quantize(node: Node) -> bool:
    if node.op == "call_method":
        target = str(node.target)
        return "quantize" in target and "dequantize" not in target
    if node.op == "call_module":
        return "Quantize" in node.target and "DeQuantize" not in node.target
    return False


def remove_quant_dequant_pairs(gm: GraphModule) -> GraphModule:
    """Remove adjacent ``quantize -> dequantize`` pairs from the graph.

    When a quantize node's output feeds a dequantize node and the dequantize
    node has no other users, the pair is redundant and is removed by rewiring
    the dequantize node's users to the quantize node's input.
    """
    graph = gm.graph
    nodes = list(graph.nodes)
    to_erase: list[Node] = []
    for node in nodes:
        if node.op == "output" or node in to_erase:
            continue
        if not _is_dequantize(node):
            continue
        if len(node.args) == 0:
            continue
        input_node = node.args[0]
        if not isinstance(input_node, Node):
            continue
        if input_node in to_erase or not _is_quantize(input_node):
            continue
        if node not in input_node.users:
            continue
        source = input_node.args[0] if len(input_node.args) > 0 else None
        if source is None:
            continue
        node.replace_all_uses_with(source)
        # erase the consumer before its producer so the producer's user
        # count drops to zero first
        to_erase.append(node)
        to_erase.append(input_node)
    for n in to_erase:
        graph.erase_node(n)
    graph.lint()
    return gm


def remove_quantize_dequantize_pairs(gm: GraphModule) -> GraphModule:
    """Alias of :func:`remove_quant_dequant_pairs`."""
    return remove_quant_dequant_pairs(gm)


def add_quantize_dequantize_pairs(
    gm: GraphModule,
    node: Node,
    scale: float = 1.0,
    zero_point: int = 0,
    dtype: str = "qint8",
) -> GraphModule:
    """Insert a ``quantize -> dequantize`` round trip after ``node``.

    The inserted nodes are attached as ``get_attr``-backed call_methods using
    the provided scale/zero-point.  All users of ``node`` are rewired to the
    dequantize output, which is useful for per-layer quantization debugging.
    """
    graph = gm.graph
    if node.op == "output":
        raise ValueError("cannot insert quantize/dequantize after the output node")

    scale_attr = f"{node.name}_ns_scale"
    zp_attr = f"{node.name}_ns_zero_point"
    if not hasattr(gm, scale_attr):
        setattr(gm, scale_attr, tp.tensor(scale))
    if not hasattr(gm, zp_attr):
        setattr(gm, zp_attr, tp.tensor(zero_point, dtype=tp.int64))

    # insert before the output node so the created nodes stay ordered
    output_node = next(n for n in graph.nodes if n.op == "output")
    with graph.inserting_before(output_node):
        scale_node = graph.get_attr(scale_attr)
        zp_node = graph.get_attr(zp_attr)
        quant_node = graph.call_method(
            "quantize_per_tensor", (node, scale_node, zp_node, dtype)
        )
        dequant_node = graph.call_method("dequantize", (quant_node,))
    # rewire every user of ``node`` except the freshly inserted quantize node
    for user in list(node.users):
        if user is quant_node:
            continue
        user.replace_input_with(node, dequant_node)
    graph.lint()
    return gm