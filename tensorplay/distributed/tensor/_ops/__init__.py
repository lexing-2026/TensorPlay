"""Built-in placement rule registry."""

from ._common_rules import OutputSharding, einop_rule, pointwise_rule
from ._math_ops import (
    NormReduction,
    Reduction,
    get_placement_from_reduction_op,
    map_placements_after_reduction,
)
from ._tensor_ops import cat_single_dim_strategy, stack_strategy
from .autogen import auto_register_op_variants
from ._view_ops import (
    dim_flatten,
    dim_movedim,
    dim_transpose,
    propagate_shape_and_sharding,
    register_view_ops,
)
from .single_dim_strategy import register_single_dim_strategy
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
    "NormReduction",
    "einop_rule",
    "pointwise_rule",
    "get_placement_from_reduction_op",
    "map_placements_after_reduction",
    "cat_single_dim_strategy",
    "stack_strategy",
    "auto_register_op_variants",
    "dim_flatten",
    "dim_movedim",
    "dim_transpose",
    "propagate_shape_and_sharding",
    "register_view_ops",
    "normalize_dim",
    "normalize_dims",
    "register_op_strategy",
    "register_prop_rule",
    "replicate_op_strategy",
    "register_single_dim_strategy",
]
