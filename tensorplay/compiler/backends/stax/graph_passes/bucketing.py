import collections
import contextlib
import logging
import operator
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Literal, TYPE_CHECKING, TypeAlias

import sympy

import tensorplay as tp
import tensorplay.distributed as dist
from tensorplay.utils import _pytree as pytree
from ....._higher_order_ops._hop_base import (
    detect_fake_mode,
    disable_proxy_modes_tracing,
)
from ..comm_analysis import (
    get_collective_type_from_kernel_name,
    NCCL_COLL,
)
from .utils import BitsetAncestors, _stable_topological_sort_region
from ..compile_log import timed_block
from ..compile_log import trace_structured
from ....._higher_order_ops.utils import make_fx
from .....graph import Graph, GraphModule, Node
from .....graph.traceback import NodeSource, NodeSourceAction
from .....graph.experimental.sympy_functions import OrderedSet


if TYPE_CHECKING:
    from .....distributed.distributed_core import GroupName


logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

overlap_log = tp.getArtifactLogger(__name__, "overlap")


def _resolve_group_name(group_name: Any) -> "GroupName":
    """Resolve group_name to a GroupName string.

    In compile-on-one-rank graphs, collective ops receive their
    group_name argument as an FX Node reference (pointing to a
    mesh_get_process_group call) rather than a string literal. For
    bucketing key purposes we resolve via the ProcessGroup stored in
    node.meta["val"].
    """
    if isinstance(group_name, str):
        return group_name  # pyrefly: ignore [bad-return]
    pg = group_name.meta["val"]
    return pg.group_name


BucketMode: TypeAlias = Literal[
    "default", "custom_ops", "custom_ops_multidtype", "coalesced"
]


def _default_bucket_mode() -> BucketMode:
    from .. import config

    return config.aten_distributed_optimizations.bucket_mode or "default"


# Helper functions moved to top for better organization
def _ag_group_key(node: Node) -> tuple[str, tp.dtype]:  # type: ignore[name-defined]
    _, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    return (_resolve_group_name(group_name), dtype)


def _ag_group_key_multidtype(node: Node) -> tuple[str]:
    _, group_size, group_name = node.args
    return (_resolve_group_name(group_name),)


def _rs_group_key(node: Node) -> tuple[str, str, tp.dtype]:  # type: ignore[name-defined]
    _, reduce_op, group_size, group_name = node.args
    dtype = node.meta["val"].dtype
    if not isinstance(reduce_op, str):
        raise AssertionError(f"expected reduce_op to be str, got {type(reduce_op)}")
    return (_resolve_group_name(group_name), reduce_op, dtype)


def _ar_group_key(node: Node) -> tuple[str, str, tp.dtype]:
    _, reduce_op, group_name = node.args
    dtype = node.meta["val"].dtype
    if not isinstance(reduce_op, str):
        raise AssertionError(f"expected reduce_op to be str, got {type(reduce_op)}")
    return (_resolve_group_name(group_name), reduce_op, dtype)


def _compute_foreach_groups(
    ag_ins: list[tp.Tensor],
    out_dtypes: list[tp.dtype],  # type: ignore[name-defined]
) -> list[int] | None:
    """
    Compute groups of indices that have the same src/dst dtype and shape.

    Groups tensors by (src_dtype, dst_dtype, shape) to avoid falling back to the foreach slow path.

    Returns a flat list with -1 as group delimiter, or None if only one group exists.
    For example, groups [[0, 2], [1]] would be encoded as [0, 2, -1, 1].
    """
    groups: defaultdict[tuple[tp.dtype, tp.dtype, tuple[int, ...]], list[int]] = (
        defaultdict(list)
    )
    for i, (ag_in, out_dtype) in enumerate(zip(ag_ins, out_dtypes)):
        shape = tuple(
            _hint_int_or_raise(s, context="all-gather foreach grouping")
            for s in ag_in.shape
        )
        key = (ag_in.dtype, out_dtype, shape)
        groups[key].append(i)

    if len(groups) <= 1:
        return None

    # Encode as flat list with -1 as delimiter
    result: list[int] = []
    for i, group_indices in enumerate(groups.values()):
        result.extend(group_indices)
        if i < len(groups) - 1:
            result.append(-1)

    return result


# Bucketing has two separate shape contracts:
#
# 1. Semantic tensor shapes stay symbolic.  The traced bucket merge graph must
#    preserve runtime tensor expressions such as numel(), split sizes,
#    tp.empty extents, narrow offsets/lengths, and reshape shapes.  Those
#    values describe actual tensor semantics and must not be specialized from
#    optimization hints.
#
# 2. Optimization-policy choices may use hints.  Bucket byte accounting and
#    foreach grouping need Python integers to decide how to group collectives.
#    For those policy-only decisions, concrete ints, backed SymInts, hinted
#    unbacked SymInts, and derived expressions from hinted symbols are valid.
#    Unhinted symbolic values fail fast instead of forcing guards or guessing.
#
# In short: hints may decide which bucket/group we choose, but they must never
# replace symbolic sizes in the graph we trace for the bucketed collective.
def _hint_int_or_raise(value: object, *, context: str) -> int:
    if type(value) is int:
        return value

    if not isinstance(value, tp.SymInt):
        raise AssertionError(f"Expected int or SymInt for {context}, got {type(value)}")

    node = value.node
    if node._hint is not None:
        return int(node._hint)
    shape_env = node.shape_env
    if shape_env is None:
        raise AssertionError(f"ShapeEnv is required to hint {context}: {value}")

    expr = sympy.sympify(node.expr).xreplace(shape_env.replacements)
    expr = expr.xreplace(shape_env.backed_var_to_val)
    expr = expr.xreplace(shape_env.var_to_hint_override)
    if isinstance(expr, sympy.Expr):
        expr = expr.expand(identity=True)
    if getattr(expr, "free_symbols", None):
        raise RuntimeError(
            f"Could not extract optimization hint for {context}: {value}. "
            "Collective bucketing requires hinted symbolic sizes for policy "
            "decisions."
        )
    return int(expr)


def _numel_hint_or_raise(tensor: tp.Tensor, *, context: str) -> int:
    return _hint_int_or_raise(tensor.numel(), context=context)


def _size_bytes_hint_or_raise(
    tensor: tp.Tensor,
    *,
    dtype: tp.dtype | None = None,
    context: str,
) -> int:
    element_size = tensor.element_size() if dtype is None else dtype.itemsize
    return _numel_hint_or_raise(tensor, context=context) * element_size


def _get_collective_node_from_wait(node: Node) -> Node | None:
    """Given a wait node, return the collective it waits on.

    Handles both standard (wait -> collective) and coalesced
    (wait -> getitem -> coalesced_collective) patterns.
    Returns None if the node is not a wait on a recognized NCCL collective.
    """
    if not is_wait_tensor(node):
        return None
    arg = node.args[0]
    if not isinstance(arg, Node):
        raise AssertionError(f"expected arg to be a Node, got {type(arg)}")
    if arg.op != "call_function":
        return None
    if arg.target is operator.getitem:
        if not isinstance(arg.args[0], Node):
            raise AssertionError(
                f"expected arg.args[0] to be a Node, got {type(arg.args[0])}"
            )
        arg = arg.args[0]
        if arg.op != "call_function":
            return None
    if not isinstance(arg.target, Callable):
        return None
    # pyrefly: ignore [missing-attribute]
    coll: NCCL_COLL = get_collective_type_from_kernel_name(arg.target.name())
    if coll == NCCL_COLL.UNSUPPORTED:
        return None
    return arg


def _schedulable_wait_node(node: Node) -> bool:
    """Check if this wait node is schedulable (waits on a recognized NCCL collective)."""
    return _get_collective_node_from_wait(node) is not None


def _populate_node_meta(
    bucket_nodes: list[Node], new_nodes: list[Node]
):
    if bucket_nodes:
        for n in new_nodes:
            # For the following keys, we only store the information of the first node so
            # gm.print_readable shows some information
            # Full information is stored in "bucketing_{key}_sources"
            for key, default in [
                ("nn_module_stack", ""),
                ("fwd_nn_module_stack", ""),
                ("stack_trace", ""),
                ("custom", {}),
            ]:
                n.meta[key] = bucket_nodes[0].meta.get(key, default)

                # Collect sources from all bucket nodes for this metadata key, for debugging purposes only
                bucketing_sources_key = f"bucketing_{key}_sources"
                # Use set to remove duplicates
                if key == "stack_trace":
                    sources = OrderedSet(
                        [
                            node.meta.get(key, default)
                            for node in bucket_nodes
                            if node.meta.get(key, default)
                        ]
                    )
                else:
                    # type might not be hashable
                    sources = [
                        node.meta.get(key, default)
                        for node in bucket_nodes
                        if node.meta.get(key, default)
                    ]
                n.meta[bucketing_sources_key] = sources

            # used by tp provenance tracking
            n.meta["from_node"] = [
                NodeSource(
                    original_node,
                    "bucketing_pass",
                    [NodeSourceAction.CREATE, NodeSourceAction.REPLACE],
                )
                for original_node in bucket_nodes
            ]


def _meta_arg(arg: object) -> object:
    if isinstance(arg, Node):
        return arg.meta.get("val", arg)
    return arg


def _same_tensor_metadata(lhs: tp.Tensor, rhs: tp.Tensor) -> bool:
    def same_dim(lhs_dim: object, rhs_dim: object) -> bool:
        from ..sizevars import statically_known_true

        try:
            return statically_known_true(lhs_dim == rhs_dim)
        except Exception:
            return lhs_dim == rhs_dim

    def same_dims(lhs_dims: tuple[object, ...], rhs_dims: tuple[object, ...]) -> bool:
        return len(lhs_dims) == len(rhs_dims) and all(
            same_dim(lhs_dim, rhs_dim) for lhs_dim, rhs_dim in zip(lhs_dims, rhs_dims)
        )

    return (
        lhs.dtype == rhs.dtype
        and lhs.device == rhs.device
        and lhs.layout == rhs.layout
        and same_dims(tuple(lhs.shape), tuple(rhs.shape))
        and same_dims(tuple(lhs.stride()), tuple(rhs.stride()))
        and same_dim(lhs.storage_offset(), rhs.storage_offset())
    )


def _same_metadata(lhs: object, rhs: object) -> bool:
    lhs_leaves, lhs_spec = pytree.tree_flatten(lhs)
    rhs_leaves, rhs_spec = pytree.tree_flatten(rhs)
    if lhs_spec != rhs_spec or len(lhs_leaves) != len(rhs_leaves):
        return False
    for lhs_leaf, rhs_leaf in zip(lhs_leaves, rhs_leaves):
        if isinstance(lhs_leaf, tp.Tensor) or isinstance(rhs_leaf, tp.Tensor):
            if not isinstance(lhs_leaf, tp.Tensor) or not isinstance(
                rhs_leaf, tp.Tensor
            ):
                return False
            if not _same_tensor_metadata(lhs_leaf, rhs_leaf):
                return False
    return True


def _recompute_changed_user_metadata(start_users: list[Node]) -> None:
    """
    Repair fake metadata after bucketing replaces a value with a layout-different
    equivalent. The replacement is semantically valid, but downstream stride
    metadata is no longer valid until consumers are re-run from metadata.
    """
    worklist = collections.deque(start_users)
    queued: OrderedSet[Node] = OrderedSet(start_users)
    while worklist:
        node = worklist.popleft()
        queued.discard(node)
        if node.op != "call_function" or not callable(node.target):
            continue
        if "val" not in node.meta:
            continue

        args = pytree.tree_map(_meta_arg, node.args)
        kwargs = pytree.tree_map(_meta_arg, node.kwargs)
        if any(
            isinstance(leaf, Node)
            for leaf in pytree.tree_leaves((args, kwargs))
        ):
            continue

        fake_mode = detect_fake_mode((node.meta.get("val"), args, kwargs))
        try:
            with fake_mode if fake_mode is not None else contextlib.nullcontext():
                new_val = node.target(*args, **kwargs)
        except Exception:
            logger.debug(
                "Skipping metadata repair for bucketing user %s",
                node.name,
                exc_info=True,
            )
            continue

        if _same_metadata(node.meta["val"], new_val):
            continue

        node.meta["val"] = new_val
        for user in node.users:
            if user not in queued:
                queued.add(user)
                worklist.append(user)


def bucket_key(node: Node, mode: BucketMode | None = None) -> object | None:
    if is_all_gather_into_tensor(node):
        group_key_fn = (
            _ag_group_key_multidtype if mode and "multidtype" in mode else _ag_group_key
        )
        return group_key_fn(node)
    elif is_reduce_scatter_tensor(node):
        return _rs_group_key(node)
    elif is_all_reduce_tensor(node):
        return _ar_group_key(node)
    else:
        return None


def pick_bucket_dtype(dtypes: list[tp.dtype]) -> tp.dtype:  # type: ignore[name-defined]
    if len(dtypes) == 0:
        raise AssertionError("expected at least one dtype, got empty list")
    return min(dtypes, key=operator.attrgetter("itemsize"))


def bucket_cap_mb_by_bucket_idx_default(bucket_id: int) -> float:
    """
    Determine the size of a bucket based on its ID.

    Args:
    bucket_id (int): The ID of the bucket.

    Returns:
    float: The size of the bucket.
    """
    return 2000.0


def bucket_all_gather(
    gm: GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode | None = None,
) -> None:
    mode = mode or _default_bucket_mode()
    if bucket_cap_mb_by_bucket_idx is None:
        from .bucketing import (
            bucket_cap_mb_by_bucket_idx_default,  # pyrefly: ignore [missing-module-attribute]
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    ag_buckets = bucket_all_gather_by_mb(gm, bucket_cap_mb_by_bucket_idx, None, mode)
    if len(ag_buckets) == 0:
        return
    merge_all_gather(gm, ag_buckets, mode)


def bucket_reduce_scatter(
    gm: GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: BucketMode | None = None,
) -> None:
    mode = mode or _default_bucket_mode()
    if bucket_cap_mb_by_bucket_idx is None:
        from .bucketing import (
            bucket_cap_mb_by_bucket_idx_default,  # pyrefly: ignore [missing-module-attribute]
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    rs_buckets = bucket_reduce_scatter_by_mb(
        gm, bucket_cap_mb_by_bucket_idx, None, mode
    )
    if len(rs_buckets) == 0:
        return
    merge_reduce_scatter(gm, rs_buckets, mode)


def is_all_gather_into_tensor(node: Node) -> bool:  # type: ignore[arg-type]
    return node.op == "call_function" and (
        node.target == tp.ops.tp.all_gather_into_tensor.default
        or node.target == tp.ops.tp.all_gather_into_tensor_out.default
    )


def is_reduce_scatter_tensor(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is tp.ops.tp.reduce_scatter_tensor.default
    )


def is_wait_tensor(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is tp.ops.tp.wait_tensor.default
    )


def is_all_reduce_tensor(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is tp.ops.tp.all_reduce.default
    )


def is_all_to_all_tensor(node: Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is tp.ops.tp.all_to_all_single.default
    )


def get_collective_type(node: Node) -> str:
    """Get the collective type name for a node."""
    if is_all_gather_into_tensor(node):
        return "all_gather"
    elif is_reduce_scatter_tensor(node):
        return "reduce_scatter"
    elif is_all_reduce_tensor(node):
        return "all_reduce"
    return ""


def get_full_bucket_key(
    node: Node, bucket_mode: BucketMode | None
) -> tuple[str, Any]:
    """Get the full bucket key including collective type and bucket key."""
    return (get_collective_type(node), bucket_key(node, mode=bucket_mode))


def is_wait_tensor_from_all_gather_into_tensor(node: Node) -> bool:
    return is_wait_tensor(node) and is_all_gather_into_tensor(node.args[0])  # type: ignore[arg-type]


def is_fsdp_all_gather(
    node: Node,
    all_node_ancestors: BitsetAncestors | None = None,
) -> bool:
    """Check if an all_gather derives from exactly one placeholder (parameter).

    When all_node_ancestors is provided, uses it for O(|ancestors|) lookup.
    Otherwise delegates to the BFS implementation in fsdp.py.
    """
    if not is_all_gather_into_tensor(node):
        return False
    if all_node_ancestors is not None:
        phs = (
            a for a in all_node_ancestors.iter_ancestors(node) if a.op == "placeholder"
        )
        return next(phs, None) is not None and next(phs, None) is None
    from .fsdp import is_fsdp_all_gather as _is_fsdp_all_gather

    return _is_fsdp_all_gather(node)


def is_fsdp_reduce_scatter(node: Node) -> bool:
    """
    Check if a reduce_scatter node is FSDP-related by verifying its output flows
    directly to graph outputs through only unary ops (e.g., to_copy, wait).
    """
    if not is_reduce_scatter_tensor(node):
        return False

    visited: OrderedSet[Node] = OrderedSet()
    stack = [node]

    while stack:
        curr = stack.pop()
        if curr in visited:
            continue
        visited.add(curr)

        for user in curr.users:
            if user.op == "output":
                continue
            # Non-unary op means computation with external data
            if len(user.all_input_nodes) != 1:
                return False
            stack.append(user)

    return True


def collect_node_descendants(
    graph: Graph,
) -> dict[Node, OrderedSet[Node]]:
    """
    Collects the descendants of each node in the graph.
    Args:
        graph (Graph): The graph to collect descendants from.
    Returns:
        dict[Node, OrderedSet[Node]]: A dictionary mapping each node to its descendants.
    """
    node_descendants: dict[Node, OrderedSet[Node]] = (
        collections.defaultdict(OrderedSet)
    )
    outdegree = collections.defaultdict(int)
    queue = []

    for node in graph.nodes:
        n_outdegree = len(node.users)
        if n_outdegree == 0:
            queue.append(node)
        else:
            outdegree[node] = len(node.users)

    while queue:
        node = queue.pop()
        for input_node in node.all_input_nodes:
            node_descendants[input_node] |= node_descendants[node]
            node_descendants[input_node].add(node)
            outdegree[input_node] -= 1

            if outdegree[input_node] == 0:
                queue.append(input_node)

    return node_descendants


def greedy_bucket_collective_by_mb(
    gm: GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_node: Callable[[Node], bool],
    node_group_key: Callable[[Node], Any],
    filter_wait_node: Callable[[Node], bool] | None = None,
) -> list[list[Node]]:
    """
    Bucketing adjacent collectives with equal node_group_key.
    We can not bucket non adjacent collectives,
    as this will effectively change the order of collectives.
    Reordering can lead to different order on different ranks.
    """
    g = gm.graph
    found_candidates = False
    for node in g.nodes:
        if filter_node(node):
            found_candidates = True
            break
    if not found_candidates:
        return []

    # Build forward adjacency list for incremental descendant tracking
    children: dict[Node, list[Node]] = collections.defaultdict(list)
    for node in g.nodes:
        for inp in node._input_nodes:
            children[inp].append(node)

    nodes_groups: list[list[Node]] = []
    cur_group: list[Node] = []
    cur_group_key = None

    for node in g.nodes:
        if is_wait_tensor(node) and filter_node(node.args[0]):
            if (filter_wait_node is None) or filter_wait_node(node):
                coll_node = node.args[0]
                group_key = node_group_key(coll_node)
                if group_key == cur_group_key:
                    cur_group.append(coll_node)
                else:
                    if len(cur_group) > 1:
                        nodes_groups.append(cur_group)
                    cur_group = [coll_node]
                    cur_group_key = group_key

    if len(cur_group) > 1:
        nodes_groups.append(cur_group)

    def _add_descendants(node: Node, desc: OrderedSet[Node]) -> None:
        """Forward BFS from node, adding all reachable nodes to desc."""
        stack = [node]
        while stack:
            n = stack.pop()
            for child in children[n]:
                if child not in desc:
                    desc.add(child)
                    stack.append(child)

    buckets: list[list[Node]] = []
    for nodes in nodes_groups:
        cur_bucket: list[Node] = []
        cur_bucket_descendents: OrderedSet[Node] = OrderedSet()
        cur_bucket_size_bytes: int = 0
        cur_bucket_id: int = 0
        bucket_size_bytes = int(
            bucket_cap_mb_by_bucket_idx(cur_bucket_id) * 1024 * 1024
        )
        for node in nodes:
            if node in cur_bucket_descendents:
                # if there is a path from node to the current bucket, we cannot horizontally fuse (bucket)
                continue
            if "val" not in node.meta:
                raise AssertionError(f"expected 'val' in node.meta for {node}")
            n_val = node.meta["val"]
            out_size_bytes = _size_bytes_hint_or_raise(
                n_val, context="collective bucket output size"
            )
            n_input_val = node.all_input_nodes[0].meta["val"]
            in_size_bytes = _size_bytes_hint_or_raise(
                n_input_val, context="collective bucket input size"
            )
            size_bytes = max(out_size_bytes, in_size_bytes)
            if cur_bucket_size_bytes + size_bytes > bucket_size_bytes and cur_bucket:
                # Current bucket is full, create new bucket
                if len(cur_bucket) > 1:
                    buckets.append(cur_bucket)
                cur_bucket = []
                cur_bucket_size_bytes = 0
                cur_bucket_id += 1
                cur_bucket_descendents = OrderedSet()
            cur_bucket_size_bytes += size_bytes
            cur_bucket.append(node)
            _add_descendants(node, cur_bucket_descendents)
        if len(cur_bucket) > 1:
            buckets.append(cur_bucket)
    return buckets


def bucket_all_gather_by_mb(
    gm: GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[Node], bool] | None = None,
    mode: BucketMode | None = None,
) -> list[list[Node]]:
    """
    Identifies all all_gather nodes and groups them into buckets,
    based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (GraphModule): GraphModule where to bucket all_gathers.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets at the start,
            as first all_gather is usually exposed.  Interface of bucket_cap_mb_by_bucket_idx
            is `bucket_cap_mb_by_bucket_idx_default` function that is default value for `bucket_cap_mb_by_bucket_idx`.
        filter_wait_node (Callable[[Node], bool] | None): If specified,
            only all_gather nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[Node]]: List of buckets, where each bucket is a list of all_gather nodes.
    """
    mode = mode or _default_bucket_mode()

    group_key_fn = (
        _ag_group_key_multidtype if mode and "multidtype" in mode else _ag_group_key
    )

    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_all_gather_into_tensor,
        group_key_fn,
        filter_wait_node,
    )


def bucket_reduce_scatter_by_mb(
    gm: GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[Node], bool] | None = None,
    mode: BucketMode | None = None,
) -> list[list[Node]]:
    """
    Identifies all reduce_scatter nodes and groups them into buckets,
        based on size limit `bucket_cap_mb_by_bucket_idx`.

    Args:
        gm (GraphModule): GraphModule where to bucket reduce_scatters.
        bucket_cap_mb_by_bucket_idx (Callable[[int], float]): Callable to specify cap of the bucket
            in megabytes by bucket idx.  The idea of `bucket_cap_mb_by_bucket_idx` is to allow
            to specify different sizes of the buckets.
        filter_wait_node (Callable[[Node], bool] | None): If specified,
            only reduce_scatter nodes with wait_node that satisfy `filter_wait_node` will be bucketed.

    Returns:
        list[list[Node]]: List of buckets, where each bucket is a list of reduce_scatter nodes.
    """
    mode = mode or _default_bucket_mode()

    if mode is not None and "multidtype" in mode:
        raise AssertionError("reduce scatter bucketing does not support multidtype")

    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_reduce_scatter_tensor,
        _rs_group_key,
        filter_wait_node,
    )


def bucket_all_reduce_by_mb(
    gm: GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float],
    filter_wait_node: Callable[[Node], bool] | None = None,
) -> list[list[Node]]:
    return greedy_bucket_collective_by_mb(
        gm,
        bucket_cap_mb_by_bucket_idx,
        is_all_reduce_tensor,
        _ar_group_key,
        filter_wait_node,
    )


def bucket_all_reduce(
    gm: GraphModule,
    bucket_cap_mb_by_bucket_idx: Callable[[int], float] | None = None,
    mode: str | None = None,
) -> None:
    if bucket_cap_mb_by_bucket_idx is None:
        from .bucketing import (
            bucket_cap_mb_by_bucket_idx_default,  # pyrefly: ignore [missing-module-attribute]
        )

        bucket_cap_mb_by_bucket_idx = bucket_cap_mb_by_bucket_idx_default
    ar_buckets = bucket_all_reduce_by_mb(gm, bucket_cap_mb_by_bucket_idx)
    if len(ar_buckets) == 0:
        return
    for bucket in ar_buckets:
        merge_all_reduce_bucket(gm.graph, bucket, mode)


@tp.library.custom_op("bucketing::_pre_bucket_reduce_scatter", mutates_args={})
def _pre_bucket_reduce_scatter(
    rs_ins: list[tp.Tensor],
    group_size: int,
) -> tp.Tensor:
    rs_ins_flattened = [x.reshape(group_size, -1) for x in rs_ins]
    new_rs_in = tp.cat(rs_ins_flattened, dim=1).flatten()
    return new_rs_in


def _pre_bucket_reduce_scatter_fake(
    rs_ins: list[tp.Tensor],
    group_size: int,
) -> tp.Tensor:
    out_numel = sum(rs_in.numel() for rs_in in rs_ins)
    return tp.empty((out_numel,), device=rs_ins[0].device, dtype=rs_ins[0].dtype)


_pre_bucket_reduce_scatter.register_fake(_pre_bucket_reduce_scatter_fake)


def reduce_scatter_merge_fn_to_trace_custom_ops(
    rs_ins: list[tp.Tensor],
    group_name: Any,
    group_size: int,
    reduce_op: str,
    reduce_dtype: tp.dtype,  # type: ignore[name-defined]
    device: tp.device,  # type: ignore[name-defined]
) -> list[tp.Tensor]:  # type: ignore[no-untyped-def]
    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]
    new_out_numels = [x.numel() // group_size for x in rs_ins]

    new_rs_in = tp.ops.tp._pre_bucket_reduce_scatter(rs_ins, group_size)

    # TODO - either use tp.cat or make sure tp foreach codegen
    # fires more reliably
    new_rs_out = tp.ops.tp.wait_tensor(
        tp.ops.tp.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_out_flat = new_rs_out.split(new_out_numels, 0)
    new_outs = [x.reshape(s) for x, s in zip(new_out_flat, new_out_sizes)]
    return new_outs


def reduce_scatter_merge_fn_to_trace(
    rs_ins: list[tp.Tensor],
    group_name: Any,
    group_size: int,
    reduce_op: str,
    reduce_dtype: tp.dtype,  # type: ignore[name-defined]
    device: tp.device,  # type: ignore[name-defined]
) -> list[tp.Tensor]:  # type: ignore[no-untyped-def]
    rs_ins_flattened = [x.reshape(group_size, -1) for x in rs_ins]

    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]
    new_out_numels = [x.numel() // group_size for x in rs_ins]

    new_rs_in = tp.cat(rs_ins_flattened, dim=1).flatten()

    new_rs_out = tp.ops.tp.wait_tensor(
        tp.ops.tp.reduce_scatter_tensor.default(
            new_rs_in, reduce_op, group_size, group_name
        )
    )
    new_out_flat = new_rs_out.split(new_out_numels, 0)
    new_outs = [x.reshape(s) for x, s in zip(new_out_flat, new_out_sizes)]
    return new_outs


def reduce_scatter_merge_fn_coalesced(
    rs_ins: list[tp.Tensor],
    group_name: Any,
    group_size: int,
    reduce_op: str,
    reduce_dtype: tp.dtype,
    device: tp.device,
) -> list[tp.Tensor]:
    """Bucketed RS via NCCL's coalesced API (ncclGroupStart/End).

    Avoids cat-ing inputs into one buffer; instead passes the tensor list
    directly to reduce_scatter_tensor_coalesced for zero-copy batching.
    """
    rs_ins_flat = [x.reshape(-1) for x in rs_ins]
    new_out_sizes = [(x.shape[0] // group_size,) + x.shape[1:] for x in rs_ins]

    rs_outs = tp.ops.tp.reduce_scatter_tensor_coalesced(
        rs_ins_flat, reduce_op, group_size, group_name
    )
    rs_outs = [tp.ops.tp.wait_tensor(o) for o in rs_outs]
    return [o.reshape(s) for o, s in zip(rs_outs, new_out_sizes)]


def all_reduce_merge_fn_to_trace(
    ar_ins: list[tp.Tensor],
    group_name: Any,
    reduce_op: str,
    reduce_dtype: tp.dtype,  # type: ignore[name-defined]
    device: tp.device,  # type: ignore[name-defined]
) -> list[tp.Tensor]:  # type: ignore[no-untyped-def]
    ar_ins_flattened = [x.reshape(-1) for x in ar_ins]
    new_ar_in = tp.cat(ar_ins_flattened)
    new_ar_out = tp.ops.tp.wait_tensor(
        tp.ops.tp.all_reduce.default(new_ar_in, reduce_op, group_name)
    )
    split_sizes = [x.numel() for x in ar_ins]
    new_outs_flat = new_ar_out.split(split_sizes)
    new_outs = [x.reshape(ar_in.shape) for x, ar_in in zip(new_outs_flat, ar_ins)]
    return new_outs


# Every dtype, for serialising one through a custom opm ops
# TODO: custom ops support list[dtype] input
_ALL_DTYPES = tuple(
    [
        getattr(tp, attr)
        for attr in dir(tp)
        if isinstance(getattr(tp, attr), tp.dtype)
    ]
)


@tp.library.custom_op("bucketing::_pre_bucket_all_gather", mutates_args={})
def _pre_bucket_all_gather(
    ag_ins: list[tp.Tensor],
    group_size: int,
    dtype: tp.dtype,  # type: ignore[name-defined]
    out_dtype_ints: list[
        int
    ],  # dtype enum values, that inputs are converted to before all_gather
    rank: int,
    foreach_group_indices: list[int] | None = None,
) -> tp.Tensor:
    """
    Pre-bucket all gather operation.

    Args:
        ag_ins: Input tensors to gather
        group_size: Size of the process group
        dtype: Target dtype for the bucket
        out_dtype_ints: Dtype enum values for each input
        rank: Current rank
        foreach_group_indices: Optional flat list of grouped indices with -1 as delimiter.
            E.g., [0, 2, -1, 1] means groups [[0, 2], [1]].
    """
    # Convert int indices back to tp.dtype
    out_dtypes = [_ALL_DTYPES[d] for d in out_dtype_ints]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes, strict=True)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = tp.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    foreach_copy_dsts = tp.split(new_ag_in, ins_split_sizes)
    # View each destination slice as its output dtype, then copy
    # The copy operation handles dtype conversion from input dtype to output dtype
    foreach_copy_dsts_typed = [
        dst.view(out_dtype)
        for dst, out_dtype in zip(foreach_copy_dsts, out_dtypes, strict=True)
    ]
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]

    # Parse pre-computed groups from flat list with -1 delimiters
    if foreach_group_indices is not None:
        groups_list: list[list[int]] = []
        current_group: list[int] = []
        for idx in foreach_group_indices:
            if idx == -1:
                if current_group:
                    groups_list.append(current_group)
                    current_group = []
            else:
                current_group.append(idx)
        # Add last group if not empty
        if current_group:
            groups_list.append(current_group)

        # Call foreach_copy_ per group
        for group_indices in groups_list:
            group_dsts = [foreach_copy_dsts_typed[idx] for idx in group_indices]
            group_srcs = [ag_ins_flattened[idx] for idx in group_indices]
            tp._foreach_copy_(group_dsts, group_srcs)
    else:
        # No grouping provided - single foreach_copy_ call
        tp._foreach_copy_(foreach_copy_dsts_typed, ag_ins_flattened)
    return new_ag_out


def _pre_bucket_all_gather_fake(
    ag_ins: list[tp.Tensor],
    group_size: int,
    dtype: tp.dtype,  # type: ignore[name-defined]
    out_dtype_ints: list[int],
    rank: int,
    foreach_group_indices: list[int] | None = None,
) -> tp.Tensor:
    out_dtypes = [_ALL_DTYPES[d] for d in out_dtype_ints]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes, strict=True)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = tp.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    return new_ag_out


_pre_bucket_all_gather.register_fake(_pre_bucket_all_gather_fake)


def all_gather_merge_fn_to_trace_custom_ops(
    _ag_ins: list[tp.Tensor],
    group_name: Any,
    group_size: int,
    dtype: tp.dtype,  # type: ignore[name-defined]
    out_dtypes: list[tp.dtype],  # type: ignore[name-defined]
    rank: int,
) -> list[tp.Tensor]:
    # Don't create convert_element_type ops - _pre_bucket_all_gather handles conversion
    # by viewing destination slices as output dtypes and letting copy do the conversion
    ag_ins = _ag_ins
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes_bytes = [
        ag_in.numel() * out_dtype.itemsize
        for ag_in, out_dtype in zip(ag_ins, out_dtypes)
    ]
    bucket_dtype_size_bytes = dtype.itemsize
    ins_split_sizes = [
        _bytes // bucket_dtype_size_bytes for _bytes in ins_split_sizes_bytes
    ]
    ag_input_numel = sum(ins_split_sizes)

    # Convert out_dtypes to indices for custom_op
    # TODO: custom ops support list[dtype] input
    out_dtype_ints = [_ALL_DTYPES.index(dt) for dt in out_dtypes]

    # Pre-compute foreach groups for better foreach_copy_ performance
    foreach_group_indices = _compute_foreach_groups(ag_ins, out_dtypes)

    new_ag_out = tp.ops.tp._pre_bucket_all_gather(
        ag_ins,
        group_size,
        dtype,
        out_dtype_ints,
        rank,
        foreach_group_indices,
    )
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    wait_tensor = tp.ops.tp.wait_tensor(
        tp.ops.tp.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs_bucket_dtype = tp.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.view(out_dtype).reshape((shape[0] * group_size,) + shape[1:])
        for o, shape, out_dtype in zip(outs_bucket_dtype, ins_sizes, out_dtypes)
    ]
    return outs_reshaped


def all_gather_merge_fn_to_trace(
    ag_ins: list[tp.Tensor],
    group_name: Any,
    group_size: int,
    dtype: tp.dtype,  # type: ignore[name-defined]
    out_dtypes: list[tp.dtype],  # type: ignore[name-defined]
    rank: int,
) -> list[tp.Tensor]:
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = tp.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    new_ag_in = new_ag_out.narrow(0, ag_input_numel * rank, ag_input_numel)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    # The compiler fuses copy_(cat(...)) into 1 Triton kernel with no allocation for cat.
    # _foreach_copy_(..., ag_ins_flattened) emits separate kernel per item,
    # resulting in large number of small triton kernels to launch.
    new_ag_in.copy_(tp.cat(ag_ins_flattened))
    wait_tensor = tp.ops.tp.wait_tensor(
        tp.ops.tp.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs = tp.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.reshape((shape[0] * group_size,) + shape[1:])
        for o, shape in zip(outs, ins_sizes)
    ]
    return outs_reshaped


def all_gather_merge_fn_to_trace_functional(
    ag_ins: list[tp.Tensor],
    group_name: Any,
    group_size: int,
    dtype: tp.dtype,  # type: ignore[name-defined]
    out_dtypes: list[tp.dtype],  # type: ignore[name-defined]
    rank: int,
    use_fsdp_ag_copy_in: bool = False,
) -> list[tp.Tensor]:
    # Implementation that is functional in graph,
    # but uses custom op tp.ops.tp.all_gather_copy_in.
    ins_sizes = [ag_in.shape for ag_in in ag_ins]
    ins_split_sizes = [ag_in.numel() for ag_in in ag_ins]
    ag_input_numel = sum(ins_split_sizes)
    device = ag_ins[0].device
    new_ag_out = tp.empty(ag_input_numel * group_size, dtype=dtype, device=device)
    ag_ins_flattened = [ag_in.reshape(-1) for ag_in in ag_ins]
    if use_fsdp_ag_copy_in:
        new_ag_in, new_ag_out = tp.ops.tp.all_gather_copy_in(
            ag_ins_flattened, new_ag_out, ins_split_sizes, ag_input_numel, rank
        )
    else:
        new_ag_in = tp.cat(ag_ins_flattened, dim=0)
    wait_tensor = tp.ops.tp.wait_tensor(
        tp.ops.tp.all_gather_into_tensor_out.default(
            new_ag_in, group_size, group_name, out=new_ag_out
        )
    )
    new_ag_out_reshaped = wait_tensor.reshape(group_size, -1)
    outs = tp.split_with_sizes(
        new_ag_out_reshaped,
        ins_split_sizes,
        dim=1,
    )
    outs_reshaped = [
        o.reshape((shape[0] * group_size,) + shape[1:])
        for o, shape in zip(outs, ins_sizes)
    ]
    return outs_reshaped


def _trace(fn, inps) -> GraphModule:  # type: ignore[no-untyped-def]
    with timed_block("fx.bucketing._trace"):
        fake_mode = detect_fake_mode(inps)
        if fake_mode is None:
            raise AssertionError("expected a fake mode to be detected, got None")
        shape_env = fake_mode.shape_env
        pending_unbacked = None
        ignorable_unbacked = None
        if shape_env is not None:
            pending_unbacked = list(shape_env.pending_fresh_unbacked_symbols)
            ignorable_unbacked = list(shape_env.ignorable_fresh_unbacked_symbols)
            shape_env.pending_fresh_unbacked_symbols.clear()
            shape_env.ignorable_fresh_unbacked_symbols.clear()
        try:
            with fake_mode, disable_proxy_modes_tracing():
                out = make_fx(fn)(*inps)
        finally:
            if shape_env is not None:
                if pending_unbacked is None:
                    raise AssertionError("expected pending_unbacked to be set")
                if ignorable_unbacked is None:
                    raise AssertionError("expected ignorable_unbacked to be set")
                shape_env.pending_fresh_unbacked_symbols[:] = pending_unbacked
                shape_env.ignorable_fresh_unbacked_symbols[:] = ignorable_unbacked
        for node in out.graph.find_nodes(
            op="call_function", target=tp.ops.tp.detach.default
        ):
            node.replace_all_uses_with(node.args[0])
            out.graph.erase_node(node)
        return out


def _insert_fn_trace_before_node(  # type: ignore[no-untyped-def]
    g: Graph,
    fn_to_trace,
    inps,
    insert_before_node: Node,
    g_fn_inps: list[Node],
    g_fn_outs: list[Node],
) -> tuple[dict[Node, Node], list[Node]]:  # type: ignore[no-untyped-def]
    """
    Helper function that traces :attr:`fn_to_trace` with inputs
    :attr:`inps`.
    The result function graph will be inserted before :attr:`insert_before_node`,
    using :attr:`g_fn_inps` nodes of original graph as inputs of function graph,
    function graph outputs will replace :attr:`g_fn_outs` in original graph.

    Returns:
        (replacements, new_nodes): Dictionary mapping old to new nodes, and list of all newly inserted nodes
    """
    with timed_block(
        "fx.bucketing._insert_fn_trace_before_node"):
        fn_gm = _trace(
            fn_to_trace,
            inps,
        )
        fn_g = fn_gm.graph
        fn_g_ins = fn_g.find_nodes(op="placeholder")
        env = {fn_g_ins[idx]: g_fn_inps[idx] for idx in range(len(g_fn_inps))}
        g_fn_new_outs: list[Node] = []
        new_nodes: list[Node] = []  # Track all newly inserted nodes

        with g.inserting_before(insert_before_node):
            for _n in fn_g.nodes:
                if _n.op == "placeholder":
                    continue
                _new_n = g.node_copy(_n, lambda x: env[x])
                env[_n] = _new_n
                if _n.op == "output":
                    g_fn_new_outs = _new_n.args[0]  # type: ignore[assignment]
                    g.erase_node(_new_n)
                else:
                    new_nodes.append(_new_n)  # Track non-output nodes

        replacements = {  # noqa: C416
            orig_out: new_out for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs)
        }
        replaced_users = []
        for orig_out, new_out in zip(g_fn_outs, g_fn_new_outs):
            replaced_users.extend(orig_out.users)
            orig_out.replace_all_uses_with(new_out)
        _recompute_changed_user_metadata(replaced_users)

        return replacements, new_nodes


def has_mergeable_all_gather_convert_dtype(n: Node) -> bool:
    node_in = n.args[0]
    return (
        is_all_gather_into_tensor(n)
        and isinstance(node_in, Node)
        and node_in.op == "call_function"
        and (
            node_in.target is tp.ops.tp.convert_element_type.default
            or node_in.target is tp.ops.tp._to_copy.default
        )
        and len(node_in.users) == 1
    )


def _sort_bucket_region(
    g: Graph,
    new_nodes: list[Node],
    new_nodes_inputs: list[Node],
) -> None:
    """Topologically sort the smallest region spanning *new_nodes* and their inputs.

    After bucketing inserts *new_nodes*, some *new_nodes_inputs* may sit after
    the new collective in the linked list.  This sorts just the affected
    region so every input precedes its consumer.

    Complexity: O(D) where D = distance between the farthest input and
    new_nodes in the linked list.  No full-graph enumeration.
    """
    if not new_nodes or not new_nodes_inputs:
        return

    new_set: OrderedSet[Node] = OrderedSet(new_nodes)
    external: OrderedSet[Node] = OrderedSet(
        [inp for inp in new_nodes_inputs if inp not in new_set]
    )
    if not external:
        return

    # Walk backward/forward from the new_nodes span to find all external
    # inputs.  ``remaining`` counts how many we still need to locate;
    # each walk stops as soon as its share is found.
    first = new_nodes[0]
    last = new_nodes[-1]
    remaining = len(external)

    cursor: Node | None = first.prev
    while cursor is not None and cursor.op != "placeholder" and remaining > 0:
        if cursor in external:
            first = cursor
            remaining -= 1
        cursor = cursor.prev

    cursor = last.next
    while cursor is not None and cursor.op != "output" and remaining > 0:
        if cursor in external:
            last = cursor
            remaining -= 1
        cursor = cursor.next

    if first is last:
        return

    region: OrderedSet[Node] = OrderedSet()
    cursor = first
    while cursor is not None:
        region.add(cursor)
        if cursor is last:
            break
        cursor = cursor.next

    _stable_topological_sort_region(g, region)


def process_collective_bucket(
    g: Graph,
    bucket_nodes: list[Node],
    fn_to_trace: Callable[..., list[tp.Tensor]],
    trace_args_fn: Callable[[list[Node]], tuple[Any, ...]],
    insert_before: Node | None = None,
    wait_insertion_point: Node | None = None,
    extra_graph_inps: list[Node] | None = None,
) -> tuple[list[Node], dict[Node, Node]]:
    """
    Process a single bucket of collective operation nodes with flexible insertion control.

    Args:
        g: The graph to modify
        bucket_nodes: Nodes in the current bucket to process
        fn_to_trace: Function to trace and insert
        trace_args_fn: Function to create trace arguments from inputs
        insert_before: Where to insert the traced function (default: after last bucket node)
        wait_insertion_point: If provided, move all nodes from wait() onwards to before this node
        extra_graph_inps: Additional non-tensor graph nodes to wire as traced
            inputs (appended after tensor inputs). Used for compile-on-one-rank
            graphs where group_name is a Node reference that make_fx proxies
            as an opaque input.

    Returns:
        new_nodes: List of all newly inserted nodes
        replacements: Dictionary mapping old wait nodes to new output nodes
    """
    # Collect inputs and waits from current bucket
    bucket_ins: list[Node] = []
    bucket_waits: list[Node] = []
    ag_node_to_pre_nodes: dict[Node, list[Node]] = defaultdict(list)

    for n in bucket_nodes:
        if len(n.users) != 1:
            raise AssertionError(f"Expected single user for {n}, got {n.users}")
        wait_n = next(iter(n.users))

        # Handle convert_element_type operations (for all_gather)
        node_in = n.args[0]
        if has_mergeable_all_gather_convert_dtype(n):
            # pyrefly: ignore [bad-argument-type]
            ag_node_to_pre_nodes[n].append(node_in)
            # pyrefly: ignore [missing-attribute]
            node_in = node_in.args[0]

        if not isinstance(node_in, Node):  # Ensure node_in is a Node
            raise AssertionError(f"expected node_in to be a Node, got {type(node_in)}")
        bucket_ins.append(node_in)
        bucket_waits.append(wait_n)

    # Create trace arguments
    trace_args = trace_args_fn(bucket_ins)

    # Determine insertion point
    if insert_before is None:
        insert_before = bucket_nodes[-1].next

    g_fn_inps = bucket_ins + (extra_graph_inps or [])

    # Insert traced function and get replacements + new nodes
    replacements, new_nodes = _insert_fn_trace_before_node(
        g,
        fn_to_trace,
        trace_args,
        insert_before,
        g_fn_inps,
        bucket_waits,
    )

    # If requested, move wait nodes and everything after to specified location
    if wait_insertion_point is not None:
        # Find the first wait node in new_nodes
        wait_start_idx = None
        for i, node in enumerate(new_nodes):
            if is_wait_tensor(node):
                wait_start_idx = i
                break

        # Move all nodes from wait onwards (including the wait)
        if wait_start_idx is not None:
            nodes_to_move = new_nodes[wait_start_idx:]
            for node in nodes_to_move:
                wait_insertion_point.prepend(node)

    # Preserve metadata from original collective nodes to new bucketed nodes
    if bucket_nodes:
        overlap_log.debug(
            "Bucketing nodes: %s, New nodes: %s",
            ",".join([n.name for n in bucket_nodes]),
            ",".join([n.name for n in new_nodes]),
        )
    _populate_node_meta(bucket_nodes, new_nodes)

    # Erase old nodes
    for node, wait_n in zip(bucket_nodes, bucket_waits):
        g.erase_node(wait_n)
        g.erase_node(node)
        # Erase any convert_element_type nodes we tracked
        for pre_node in reversed(ag_node_to_pre_nodes[node]):
            g.erase_node(pre_node)

    _sort_bucket_region(g, new_nodes, g_fn_inps)

    return new_nodes, replacements


def merge_reduce_scatter_bucket(
    g: Graph,
    rs_nodes: list[Node],
    mode: BucketMode | None = None,
    insert_before: Node | None = None,
    wait_insertion_point: Node | None = None,
) -> tuple[list[Node], dict[Node, Node]]:
    mode = mode or _default_bucket_mode()
    # Validate bucket consistency
    rs0 = rs_nodes[0]
    rs0_val = rs0.meta["val"]
    _, reduce_op, group_size, group_name = rs0.args
    group_name_str = _resolve_group_name(group_name)
    reduce_dtype = rs0_val.dtype
    device = rs0_val.device

    for n in rs_nodes:
        rs_val = n.meta["val"]
        if not (
            n.args[1] == reduce_op
            and n.args[2] == group_size
            and _resolve_group_name(n.args[3]) == group_name_str
            and rs_val.device == device
            and rs_val.dtype == reduce_dtype
        ):
            raise AssertionError(
                f"reduce_scatter node {n} does not match bucket parameters"
            )

    # Choose merge function based on mode
    rs_merge_fn = reduce_scatter_merge_fn_to_trace
    if mode == "coalesced":
        rs_merge_fn = reduce_scatter_merge_fn_coalesced
    elif mode and "custom_ops" in mode:
        rs_merge_fn = reduce_scatter_merge_fn_to_trace_custom_ops

    group_name_val = (
        group_name.meta["val"] if isinstance(group_name, Node) else group_name
    )

    def create_trace_args(bucket_ins: list[Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_name_val,
            group_size,
            reduce_op,
            reduce_dtype,
            device,
        )

    return process_collective_bucket(
        g,
        rs_nodes,
        rs_merge_fn,
        create_trace_args,
        insert_before=insert_before,
        wait_insertion_point=wait_insertion_point,
        extra_graph_inps=(
            [group_name] if isinstance(group_name, Node) else None
        ),
    )


def merge_all_reduce_bucket(
    g: Graph,
    ar_nodes: list[Node],
    mode: str | None = None,
    insert_before: Node | None = None,
    wait_insertion_point: Node | None = None,
) -> tuple[list[Node], dict[Node, Node]]:
    ar0 = ar_nodes[0]
    ar0_val = ar0.meta["val"]
    _, reduce_op, group_name = ar0.args
    group_name_str = _resolve_group_name(group_name)
    reduce_dtype = ar0_val.dtype
    device = ar0_val.device

    for n in ar_nodes:
        ar_val = n.meta["val"]
        if not (
            n.args[1] == reduce_op
            and _resolve_group_name(n.args[2]) == group_name_str
            and ar_val.device == device
            and ar_val.dtype == reduce_dtype
        ):
            raise AssertionError(
                f"all_reduce node {n} does not match bucket parameters"
            )

    ar_merge_fn = all_reduce_merge_fn_to_trace

    group_name_val = (
        group_name.meta["val"] if isinstance(group_name, Node) else group_name
    )

    def create_trace_args(bucket_ins: list[Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_name_val,
            reduce_op,
            reduce_dtype,
            device,
        )

    return process_collective_bucket(
        g,
        ar_nodes,
        ar_merge_fn,
        create_trace_args,
        insert_before=insert_before,
        wait_insertion_point=wait_insertion_point,
        extra_graph_inps=(
            [group_name] if isinstance(group_name, Node) else None
        ),
    )


def merge_all_gather_bucket(
    g: Graph,
    ag_nodes: list[Node],
    mode: BucketMode | None = None,
    insert_before: Node | None = None,
    wait_insertion_point: Node | None = None,
) -> tuple[list[Node], dict[Node, Node]]:
    mode = mode or _default_bucket_mode()
    from .....distributed.distributed_core import _resolve_process_group

    ag0 = ag_nodes[0]
    _, group_size, group_name = ag0.args
    group_name_str = _resolve_group_name(group_name)
    _ag_dtypes: list[tp.dtype] = []  # type: ignore[name-defined]

    for n in ag_nodes:
        if not (
            n.args[1] == group_size and _resolve_group_name(n.args[2]) == group_name_str
        ):
            raise AssertionError(
                f"all_gather node {n} does not match bucket parameters"
            )
        _ag_dtypes.append(n.meta["val"].dtype)

    bucket_dtype = pick_bucket_dtype(_ag_dtypes)

    # Choose merge function based on mode
    ag_merge_fn = all_gather_merge_fn_to_trace
    if mode == "coalesced":
        logger.info("coalesced bucket_mode not supported for all_gather, using default")
    elif mode and "custom_ops" in mode:
        ag_merge_fn = all_gather_merge_fn_to_trace_custom_ops  # type: ignore[assignment]

    # pyrefly: ignore [bad-argument-type]
    rank: int = dist.get_rank(_resolve_process_group(group_name_str))

    group_name_val = (
        group_name.meta["val"] if isinstance(group_name, Node) else group_name
    )

    def create_trace_args(bucket_ins: list[Node]) -> tuple[Any, ...]:
        return (
            pytree.tree_map(lambda node: node.meta["val"], bucket_ins),
            group_name_val,
            group_size,
            bucket_dtype,
            _ag_dtypes,
            rank,
        )

    return process_collective_bucket(
        g,
        ag_nodes,
        ag_merge_fn,
        create_trace_args,
        insert_before=insert_before,
        wait_insertion_point=wait_insertion_point,
        extra_graph_inps=(
            [group_name] if isinstance(group_name, Node) else None
        ),
    )


def merge_reduce_scatter(
    gm: GraphModule,
    rs_buckets: list[list[Node]],
    mode: BucketMode | None = None,
) -> None:
    """
    Merges specified buckets of reduce_scatter to joint reduce_scatter.
    """
    mode = mode or _default_bucket_mode()
    with timed_block("fx.bucketing.merge_reduce_scatter"):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_bucketing_passes_reduce_scatter_buckets",
                "encoding": "string",
            },
            payload_fn=lambda: str(rs_buckets),
        )

        g = gm.graph

        for rs_nodes in rs_buckets:
            merge_reduce_scatter_bucket(g, rs_nodes, mode)


def merge_all_gather(
    gm: GraphModule,
    ag_buckets: list[list[Node]],
    mode: BucketMode | None = None,
) -> None:
    """
    Merges specified buckets of all_gather to joint all_gather.
    """
    mode = mode or _default_bucket_mode()
    with timed_block("fx.bucketing.merge_all_gather"):
        trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_bucketing_passes_all_gather_buckets",
                "encoding": "string",
            },
            payload_fn=lambda: str(ag_buckets),
        )

        g = gm.graph

        for ag_nodes in ag_buckets:
            merge_all_gather_bucket(g, ag_nodes, mode)
