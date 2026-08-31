"""Built-in placement rule registry."""

from ._common_rules import OutputSharding, einop_rule, pointwise_rule
from ._math_ops import Reduction, get_placement_from_reduction_op, map_placements_after_reduction
from ._tensor_ops import cat_single_dim_strategy, stack_strategy
from ._view_ops import dim_flatten, dim_movedim, dim_transpose, propagate_shape_and_sharding
from .utils import (
    normalize_dim,
    normalize_dims,
    register_op_strategy,
    register_prop_rule,
    replicate_op_strategy,
)

__all__ = [
    "OutputSharding",
    "Reduction",
    "einop_rule",
    "pointwise_rule",
    "get_placement_from_reduction_op",
    "map_placements_after_reduction",
    "cat_single_dim_strategy",
    "stack_strategy",
    "dim_flatten",
    "dim_movedim",
    "dim_transpose",
    "propagate_shape_and_sharding",
    "normalize_dim",
    "normalize_dims",
    "register_op_strategy",
    "register_prop_rule",
    "replicate_op_strategy",
]
