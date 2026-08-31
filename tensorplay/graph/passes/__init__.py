"""Native graph transformations and pass orchestration."""

from .base import PassBase, PassResult
from .constant_folding import ConstFold
from .dead_code_elimination import DeadCodeElimination
from .decompose import DecomposePass
from .fusion_hint import POINTWISE_FUSED_OP_NAMES, PointwiseFusionHint
from .fake_tensor_prop import FakeTensorProp
from .annotate_getitem_nodes import annotate_getitem_nodes
from .canonicalize import canonicalize_graph, rename_nodes_to_canonical
from .graph_drawer import GraphDrawer
from .graph_manipulation import (
    get_size_of_all_nodes,
    get_size_of_node,
    get_tensor_meta,
    replace_target_nodes_with,
    size_bytes,
)
from .graph_transform_observer import GraphTransformObserver
from .operator_support import (
    OpSupports,
    OperatorSupport,
    OperatorSupportBase,
    any_chain,
    chain,
    create_op_support,
)
from .param_fetch import (
    default_matching,
    extract_attrs_for_lowering,
    lift_lowering_attrs_to_nodes,
)
from .normalize import NormalizeOperators
from .pass_manager import (
    PassManager,
    inplace_wrapper,
    log_hook,
    loop_pass,
    these_before_those_pass_constraint,
    this_before_that_pass_constraint,
)
from .runtime_assert import insert_deferred_runtime_asserts
from .shape_prop import ShapeProp, TensorMetadata
from ._tensorify_python_scalars import tensorify_python_scalars
from .split_utils import (
    Component,
    getattr_recursive,
    move_non_tensor_nodes_on_boundary,
    setattr_recursive,
    split_by_tags,
)
from .splitter_base import (
    GraphNetAccNodesFinder,
    GraphNetSplitterInternalError,
    NodeEvent,
    NodeEventTracker,
    SplitResult,
    Subgraph,
    generate_inputs_for_submodules,
)
from .split_module import Partition, split_module, split_module_simple
from .reinplace import reinplace
from .regional_inductor import regional_inductor
from .regional_inductor_invoke_subgraph import regional_inductor_invoke_subgraph
from .backends import CudaGraphsSupport, partition_cudagraphs
from .tools_common import (
    CALLABLE_NODE_OPS,
    GraphAccFusionsFinder,
    get_acc_ops_name,
    get_node_target,
    is_node_output_tensor,
    legalize_graph,
    stable_topological_sort,
)

__all__ = [
    "ConstFold",
    "CALLABLE_NODE_OPS",
    "Component",
    "DeadCodeElimination",
    "DecomposePass",
    "GraphDrawer",
    "GraphAccFusionsFinder",
    "GraphNetAccNodesFinder",
    "GraphNetSplitterInternalError",
    "FakeTensorProp",
    "GraphTransformObserver",
    "NormalizeOperators",
    "OpSupports",
    "OperatorSupport",
    "OperatorSupportBase",
    "POINTWISE_FUSED_OP_NAMES",
    "PassBase",
    "PassManager",
    "PassResult",
    "Partition",
    "PointwiseFusionHint",
    "ShapeProp",
    "SplitResult",
    "Subgraph",
    "TensorMetadata",
    "annotate_getitem_nodes",
    "any_chain",
    "canonicalize_graph",
    "chain",
    "create_op_support",
    "default_matching",
    "extract_attrs_for_lowering",
    "get_size_of_all_nodes",
    "get_size_of_node",
    "get_tensor_meta",
    "get_acc_ops_name",
    "get_node_target",
    "generate_inputs_for_submodules",
    "inplace_wrapper",
    "insert_deferred_runtime_asserts",
    "lift_lowering_attrs_to_nodes",
    "log_hook",
    "loop_pass",
    "rename_nodes_to_canonical",
    "replace_target_nodes_with",
    "size_bytes",
    "is_node_output_tensor",
    "legalize_graph",
    "stable_topological_sort",
    "split_by_tags",
    "move_non_tensor_nodes_on_boundary",
    "getattr_recursive",
    "setattr_recursive",
    "NodeEvent",
    "NodeEventTracker",
    "tensorify_python_scalars",
    "these_before_those_pass_constraint",
    "this_before_that_pass_constraint",
    "CudaGraphsSupport",
    "partition_cudagraphs",
    "reinplace",
    "regional_inductor",
    "regional_inductor_invoke_subgraph",
    "split_module",
    "split_module_simple",
]
