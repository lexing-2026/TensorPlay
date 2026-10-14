"""Pattern utilities for the FX numeric suite.

Fusion patterns are derived from the native backend config, whose patterns are
written in input-to-output order (``(nn.Conv2d, nn.ReLU)``).  The numeric suite
matches graphs in **reverse** order (output node first), matching how fusion
patterns are consumed by the graph-mode quantization workflow.
"""

from __future__ import annotations

from typing import Any

from tensorplay import nn
from tensorplay.nn import functional as F
from tensorplay.graph import GraphModule
from tensorplay.graph.node import Node
from tensorplay.ao.quantization import ObserverBase, FakeQuantizeBase
from tensorplay.ao.quantization.backend_config import get_native_backend_config
from tensorplay.ao.quantization.utils import getattr_from_fqn

from .ns_types import NSNodeTargetType

__all__ = [
    "get_type_a_related_to_b",
    "get_type_b_related_to_a",
    "get_reversed_fusions",
    "end_node_matches_reversed_fusion",
]


def get_type_a_related_to_b(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
) -> set[tuple[NSNodeTargetType, NSNodeTargetType]]:
    """Return all bidirectional (op, related op) pairs from the given sets."""
    type_a_related_to_b: set[tuple[NSNodeTargetType, NSNodeTargetType]] = set()
    for s in base_name_to_sets_of_related_ops.values():
        s_list = list(s)
        for idx_0 in range(len(s_list)):
            for idx_1 in range(idx_0, len(s_list)):
                type_a_related_to_b.add((s_list[idx_0], s_list[idx_1]))
                type_a_related_to_b.add((s_list[idx_1], s_list[idx_0]))
    return type_a_related_to_b


def get_type_b_related_to_a(
    base_name_to_sets_of_related_ops: dict[str, set[NSNodeTargetType]],
) -> set[tuple[NSNodeTargetType, NSNodeTargetType]]:
    """Alias of :func:`get_type_a_related_to_b` (symmetric relation)."""
    return get_type_a_related_to_b(base_name_to_sets_of_related_ops)


NSFusionElType = (
    type
    | str  # call_method name, example: "dequantize"
    | tuple[str, Any]  # call_method name and first argument, example: ("to", float16)
)
NSFusionType = (
    tuple[NSFusionElType, NSFusionElType]
    | tuple[NSFusionElType, NSFusionElType, NSFusionElType, NSFusionElType]
)


def get_reversed_fusions() -> list[tuple[NSFusionType, int]]:
    """Return potential fusions in reverse order, with the base-op index.

    The returned list contains ``(fusion_pattern, base_op_idx)`` pairs.  The
    fusion pattern lists ops output-first; ``base_op_idx`` is the index of the
    op within the pattern that should be used to match related ops (specified
    in non-reverse order).
    """
    results: list[tuple[NSFusionType, int]] = []
    default_base_op_idx = 0

    backend_config = get_native_backend_config()
    seen: set[tuple] = set()
    for config in backend_config.configs:
        quant_pattern = config.pattern
        if not isinstance(quant_pattern, tuple):
            continue
        # only patterns of multiple ops are fusions
        if len(quant_pattern) < 2:
            continue
        # reverse to output-first order and deduplicate
        reversed_pattern = tuple(reversed(quant_pattern))
        if reversed_pattern in seen:
            continue
        seen.add(reversed_pattern)
        results.append((reversed_pattern, default_base_op_idx))

        # also register observer/fake-quant-headed variants
        for cls in (ObserverBase, FakeQuantizeBase):
            new_pattern = (cls, *reversed_pattern)
            results.append((new_pattern, default_base_op_idx))

    # fp16 emulation, BN fusion and quantized-module fusion patterns that are
    # not in the quantization backend config are defined here.
    fp16_em_base_op_idx = 1
    patterns_to_add = [
        # linear-relu fp16 emulation: fp16_to_fp32 -> linear -> relu -> fp32_to_fp16
        ((("to", "float16"), F.relu, F.linear, "dequantize"), fp16_em_base_op_idx),
        ((nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm1d, nn.Conv1d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm2d, nn.Conv2d), default_base_op_idx),
        ((nn.ReLU, nn.BatchNorm3d, nn.Conv3d), default_base_op_idx),
    ]
    # quantized modules keep their module structure in this framework (rather
    # than being lowered to call_function ops), so fusion patterns must also
    # recognize them to align subgraph boundaries between float and quantized
    # graphs.
    from tensorplay.ao.nn import quantized as nnq

    quantized_fusions = [
        ((nn.ReLU, nnq.QuantizedLinear), default_base_op_idx),
        ((nn.ReLU, nnq.Conv1d), default_base_op_idx),
        ((nn.ReLU, nnq.Conv2d), default_base_op_idx),
        ((nn.ReLU, nnq.Conv3d), default_base_op_idx),
        ((nn.ReLU, nnq.BatchNorm2d), default_base_op_idx),
        ((nn.ReLU, nnq.BatchNorm3d), default_base_op_idx),
    ]
    for p in quantized_fusions:
        patterns_to_add.append(p)
        patterns_to_add.append(((ObserverBase, *p[0]), p[1]))
        patterns_to_add.append(((FakeQuantizeBase, *p[0]), p[1]))
    for p in patterns_to_add:
        results.append(p)
        results.append(((ObserverBase, *p[0]), p[1]))
        results.append(((FakeQuantizeBase, *p[0]), p[1]))

    return results


def end_node_matches_reversed_fusion(
    end_node: Node,
    reversed_fusion: NSFusionType,
    gm: GraphModule,
    seen_nodes: set[Node],
) -> bool:
    """Return whether ``end_node`` is the end of a match of ``reversed_fusion``."""
    cur_node = end_node
    for fusion_idx in range(len(reversed_fusion)):
        # each node can only belong to one matched pattern
        if cur_node in seen_nodes:
            return False

        cur_fusion_el = reversed_fusion[fusion_idx]

        if cur_node.op == "call_function":
            fusion_el_is_fun = (not isinstance(cur_fusion_el, str)) and (
                not isinstance(cur_fusion_el, type)
            )
            if fusion_el_is_fun:
                if cur_node.target != cur_fusion_el:
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == "call_module":
            fusion_el_is_mod = isinstance(cur_fusion_el, type)
            if fusion_el_is_mod:
                if not isinstance(cur_node.target, str):
                    return False
                target_mod = getattr_from_fqn(gm, cur_node.target)
                if not isinstance(target_mod, cur_fusion_el):
                    return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False

        elif cur_node.op == "call_method":
            fusion_el_is_meth_with_second_arg = (
                isinstance(cur_fusion_el, tuple) and len(cur_fusion_el) == 2
            )
            fusion_el_is_meth_without_args = isinstance(cur_fusion_el, str)
            if fusion_el_is_meth_without_args or fusion_el_is_meth_with_second_arg:
                if fusion_el_is_meth_without_args:
                    if cur_node.target != cur_fusion_el:
                        return False
                else:
                    if cur_node.target != cur_fusion_el[0]:
                        return False
                    elif len(cur_node.args) < 2:
                        return False
                    elif cur_node.args[1] != cur_fusion_el[1]:
                        return False
                if len(cur_node.args) > 0 and isinstance(cur_node.args[0], Node):
                    cur_node = cur_node.args[0]
                else:
                    return False
            else:
                return False
        else:
            return False

    return True