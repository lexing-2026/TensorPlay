"""Type aliases and result-value types for the FX numeric suite."""

import enum
from collections.abc import Callable
from typing import Any, NamedTuple

from tensorplay.graph.node import Node

__all__ = [
    "NSSingleResultValuesType",
    "NSSubgraph",
    "NSSingleResultType",
    "NSResultsType",
    "NSNodeTargetType",
]


class NSSingleResultValuesType(str, enum.Enum):
    WEIGHT = "weight"
    NODE_OUTPUT = "node_output"
    NODE_INPUT = "node_input"


class NSSubgraph(NamedTuple):
    start_node: Node
    end_node: Node
    base_op_node: Node


# {
#   'type': 'weight',          # one of NSSingleResultValuesType
#   'values': [tensor, ...],
#   'prev_node_name': 'linear1',
#   'prev_node_target_type': Linear,   # module/function/method
#   'ref_node_name': 'linear1',
#   'index_within_arg': 0,
#   'index_of_arg': 0,
#   'comparisons': [tensor, ...],
#   'comparison_fn_name': 'sqnr',
#   'qconfig_str': '...',
# }
NSSingleResultType = dict[str, Any]

# {
#   'layer_name_1': {
#     'node_output': {
#       'model_name_a': [NSSingleResultType, ...],
#       'model_name_b': [NSSingleResultType, ...],
#     },
#   },
# }
NSResultsType = dict[str, dict[str, dict[str, list[NSSingleResultType]]]]

# Target type of a node: a function/module class for call_function/call_module,
# or a method name string for call_method.
NSNodeTargetType = Callable | str