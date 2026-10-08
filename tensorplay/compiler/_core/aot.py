"""Ahead-of-time graph partitioning over the canonical graph (L4, v2).

local vector-Jacobian rules append tagged backward nodes into the SAME
joint graph (``meta["is_backward"]``), then a partitioner extracts the
forward/backward pair by tag membership -- ``partition_default`` today,
a min-cut strategy behind the same signature later.
"""

from __future__ import annotations

import heapq
import inspect
import operator
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from tensorplay.graph import Graph, GraphModule, Node


class AOTError(RuntimeError):
    """Raised when a forward region cannot be differentiated."""


_LEAF_OPS = ("placeholder", "get_attr")


# ---------------------------------------------------------------------------
# Node combinators (backward construction never needs Proxy)
# ---------------------------------------------------------------------------


def _sum_to_size_op():
    """The operation that reduces a gradient to the shape its value had.

    Resolved when it is first needed rather than at import, because the
    operation table is reached through the framework and this module is part of
    what reaches it.
    """

    import tensorplay as tp

    return tp.ops.tp.sum_to_size.default


def _reshape_op():
    """The operation that gives a gradient the shape its value had."""

    import tensorplay as tp

    return tp.ops.tp.reshape.default


def _index_add_op():
    """The operation that accumulates a gradient back where it came from."""

    import tensorplay as tp

    return tp.ops.tp.index_add.default


def _scatter_add_op():
    """The operation that accumulates a gradient back where it was read from."""

    import tensorplay as tp

    return tp.ops.tp.scatter_add.default


def _emit(
    graph: Graph,
    op: str,
    target: Any,
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
) -> Node:
    return graph.create_node(op, target, args, kwargs or {})


def _chain_add(graph: Graph, contributions: List[Node]) -> Node:
    result = contributions[0]
    for extra in contributions[1:]:
        result = _emit(graph, "call_function", operator.add, (result, extra))
    return result


def _reduce_to_shape(
    graph: Graph,
    grad: Node,
    current_shape: Optional[Tuple[int, ...]],
    target_shape: Optional[Tuple[int, ...]],
) -> Node:
    if current_shape is None or target_shape is None:
        return grad
    extra = len(current_shape) - len(target_shape)
    if extra > 0:
        # Reducing a gradient down to the shape the value came from is one
        # operation, not a loop of reductions: a call to a method of a value
        # would be a call to a method of a value this compiler has not made,
        # and the shape it reduces to is the only thing being asked for.
        grad = _emit(
            graph,
            "call_function",
            _sum_to_size_op(),
            (grad, tuple(target_shape)),
            {},
        )
        return grad
    if tuple(current_shape) != tuple(target_shape):
        grad = _emit(
            graph,
            "call_function",
            _reshape_op(),
            (grad, tuple(target_shape)),
            {},
        )
    return grad


def _ones_like(graph: Graph, value: Node) -> Node:
    zero = _emit(graph, "call_function", operator.mul, (value, 0))
    return _emit(graph, "call_function", operator.add, (zero, 1))


def _zeros_like(value: Any) -> Any:
    # Resolve lazily so importing the AOT core does not initialize the public
    # functional module while its graph helpers are still being loaded.
    from tensorplay.functional import zeros_like

    return zeros_like(value)


# ---------------------------------------------------------------------------
# Joint-graph rule emission
# ---------------------------------------------------------------------------


class _JointBuilder:
    def __init__(self, fwd_gm: GraphModule) -> None:
        self.fwd = fwd_gm
        self.graph = fwd_gm.graph
        self.tangent = self.graph.placeholder("tangent")
        self.tangent.meta["is_backward"] = True

    def bwd(self, op: str, target: Any, args: Tuple[Any, ...], kwargs: Optional[Dict[str, Any]] = None) -> Node:
        node = _emit(self.graph, op, target, args, kwargs)
        node.meta["is_backward"] = True
        return node

    def shape_of(self, value: Any) -> Optional[Tuple[int, ...]]:
        if isinstance(value, Node):
            value = value.meta.get("val")
        shape = getattr(value, "shape", None)
        if callable(shape):
            shape = shape()
        try:
            return tuple(int(dim) for dim in shape)
        except (TypeError, ValueError):
            return None

    def reduce_for(self, grad: Node, producer: Node, leaf: Any) -> Node:
        return _reduce_to_shape(
            self.graph, grad, self.shape_of(producer), self.shape_of(leaf)
        )


def _rule_mul(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
    a_node, b_node = node.args[0], node.args[1]
    return {
        a_node: b.reduce_for(b.bwd("call_function", operator.mul, (go, b_node)), node, a_node),
        b_node: b.reduce_for(b.bwd("call_function", operator.mul, (go, a_node)), node, b_node),
    }


def _rule_add(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
    a_node, b_node = node.args[0], node.args[1]
    return {
        a_node: b.reduce_for(go, node, a_node),
        b_node: b.reduce_for(go, node, b_node),
    }


def _rule_sub(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
    a_node, b_node = node.args[0], node.args[1]
    neg = b.bwd("call_function", operator.neg, (go,))
    return {
        a_node: b.reduce_for(go, node, a_node),
        b_node: b.reduce_for(neg, node, b_node),
    }


def _rule_truediv(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
    a_node, b_node = node.args[0], node.args[1]
    neg_go = b.bwd("call_function", operator.neg, (go,))
    num = b.bwd("call_function", operator.mul, (neg_go, a_node))
    denom = b.bwd("call_function", operator.mul, (b_node, b_node))
    db = b.bwd("call_function", operator.truediv, (num, denom))
    da = b.bwd("call_function", operator.truediv, (go, b_node))
    return {
        a_node: b.reduce_for(da, node, a_node),
        b_node: b.reduce_for(db, node, b_node),
    }


def _rule_neg(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
    return {node.args[0]: b.bwd("call_function", operator.neg, (go,))}


def _rule_index_select(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
    self_node, dim, index = node.args
    zeros = b.bwd("call_function", _zeros_like, (self_node,))
    grad = b.bwd(
        "call_function", _index_add_op(), (zeros, dim, index, go)
    )
    return {self_node: b.reduce_for(grad, node, self_node)}


def _rule_gather(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
    self_node, dim, index = node.args[:3]
    zeros = b.bwd("call_function", _zeros_like, (self_node,))
    grad = b.bwd(
        "call_function", _scatter_add_op(), (zeros, dim, index, go)
    )
    return {self_node: b.reduce_for(grad, node, self_node)}


def _method_rule(formula: Callable[[_JointBuilder, Node, Node], Node]):
    def rule(b: _JointBuilder, node: Node, go: Node) -> Dict[Any, Node]:
        inner = node.args[0]
        return {inner: formula(b, go, inner)}

    return rule


_RULES: Dict[Tuple[str, Any], Callable] = {
    ("call_function", operator.add): _rule_add,
    ("call_function", operator.sub): _rule_sub,
    ("call_function", operator.mul): _rule_mul,
    ("call_function", operator.truediv): _rule_truediv,
    ("call_function", operator.neg): _rule_neg,
    ("call_function", "index_select"): _rule_index_select,
    ("call_function", "gather"): _rule_gather,
    ("call_method", "index_select"): _rule_index_select,
    ("call_method", "gather"): _rule_gather,
    # Method spellings of the same arithmetic (x.mul(y) traces as
    # call_method) share the operator rules: identical args layout.
    ("call_method", "add"): _rule_add,
    ("call_method", "sub"): _rule_sub,
    ("call_method", "mul"): _rule_mul,
    ("call_method", "truediv"): _rule_truediv,
    ("call_method", "div"): _rule_truediv,
    ("call_method", "neg"): _rule_neg,
    ("call_method", "relu"): _method_rule(
        lambda b, go, s: b.bwd("call_function", operator.mul, (go, b.bwd("call_function", operator.gt, (s, 0))))
    ),
    ("call_method", "sum"): _method_rule(
        # formula reads only the input's SIZES, never its values, so emit a
        # metadata-only expand (no edge to the forward node) and the
        # partitioner correctly saves nothing for sum.
        lambda b, go, s: b.bwd(
            "call_method", "expand", (go, b.shape_of(s))
        )
    ),
    ("call_method", "sin"): _method_rule(
        lambda b, go, s: b.bwd("call_function", operator.mul, (b.bwd("call_method", "cos", (s,)), go))
    ),
    ("call_method", "cos"): _method_rule(
        lambda b, go, s: b.bwd(
            "call_function",
            operator.mul,
            (b.bwd("call_function", operator.neg, (b.bwd("call_method", "sin", (s,)),)), go),
        )
    ),
    ("call_method", "exp"): _method_rule(
        lambda b, go, s: b.bwd("call_function", operator.mul, (b.bwd("call_method", "exp", (s,)), go))
    ),
    ("call_method", "log"): _method_rule(
        lambda b, go, s: b.bwd("call_function", operator.truediv, (go, s))
    ),
}


# ---------------------------------------------------------------------------
# Partitioner (default: save every forward node consumed by backward)
# ---------------------------------------------------------------------------


def _copy_nodes(
    nodes: Sequence[Node],
    outputs: Sequence[Any],
    external_as_inputs: bool,
) -> Tuple[Graph, Dict[Node, Node], List[Node]]:
    """Copy ``nodes`` into a fresh graph, remapping internal references.

    References outside the subset become placeholders when
    ``external_as_inputs`` is true.
    """

    graph = Graph()
    mapping: Dict[Node, Node] = {}

    def remap(value: Any) -> Any:
        if isinstance(value, Node):
            if value not in mapping and value.meta.get("is_subgraph"):
                # A graph an operation is handed is read where it is used: it
                # is part of the program, not a value passed between halves.
                mapping[value] = graph.get_attr(value.target)
                mapping[value].meta.update(value.meta)
            if value not in mapping:
                if not external_as_inputs:
                    raise AOTError(f"unmapped node {value.name} during extraction")
                mapping[value] = graph.placeholder(value.name)
                mapping[value].meta.update(value.meta)
            return mapping[value]
        if isinstance(value, tuple):
            return tuple(remap(v) for v in value)
        if isinstance(value, list):
            return [remap(v) for v in value]
        if isinstance(value, dict):
            return {k: remap(v) for k, v in value.items()}
        if isinstance(value, slice):
            return value
        return value

    for node in nodes:
        new_args = tuple(remap(a) for a in node.args)
        new_kwargs = {k: remap(v) for k, v in node.kwargs.items()}
        clone = graph.create_node(node.op, node.target, new_args, new_kwargs, name=node.name)
        clone.meta.update(node.meta)
        mapping[node] = clone
    graph.output(tuple(remap(o) for o in outputs) if len(outputs) > 1 else remap(outputs[0]))
    return graph, mapping, [mapping[n] for n in nodes]


def _forward_writes(fwd_nodes: List[Node]) -> List[Node]:
    """The forward operations held for what they write rather than return."""

    return [
        n for n in fwd_nodes
        if n.op == "call_function" and n.is_impure(impure_random=False)
    ]


def _backward_reach(bwd_nodes: List[Node], bwd_out_args: List[Any], stop: set) -> set:
    """Every node the backward computes or reads, not looking past ``stop``."""

    reached: set = set()
    stack = [*bwd_nodes, *(a for a in bwd_out_args if isinstance(a, Node))]
    while stack:
        n = stack.pop()
        if n in reached:
            continue
        reached.add(n)
        if n.op in _LEAF_OPS or n in stop:
            continue
        stack.extend(n.all_input_nodes)
    return reached


def _snapshot_inputs_read_after_write(
    joint: Graph, writes: List[Node], reached: set
) -> List[Node]:
    """Give the backward a copy of each input it reads that the forward writes.

    The forward's results are written back into an input -- a buffer updated
    in place -- before the backward runs, so a backward reading that input
    would read the value after the write.  It reads a copy taken before the
    write instead, as does every other reader; the copies are returned for
    the caller to keep for the backward.
    """

    import tensorplay

    written: Dict[Node, List[Node]] = {}
    for n in writes:
        dest = n.args[0] if n.args else None
        if _op_name(n) == "copy_" and isinstance(dest, Node) and dest.op == "placeholder":
            written.setdefault(dest, []).append(n)
    snapshots = []
    for primal, its_writes in written.items():
        if primal not in reached:
            continue
        with joint.inserting_after(primal):
            snap = joint.call_function(tensorplay.ops.tp.clone.default, (primal,))
        snap.meta.update({k: v for k, v in primal.meta.items() if k != "is_backward"})
        # Its value is a tensor of its own: one sharing the input's memory
        # would say the copy is the input, and the copy would be dropped.
        value = primal.meta.get("val")
        if isinstance(value, tensorplay.Tensor):
            snap.meta["val"] = value.clone()
        elif _is_tensor_val(value):
            import copy

            snapshot_meta = copy.copy(value)
            snapshot_meta._storage_id = object()
            snap.meta["val"] = snapshot_meta
        primal.replace_all_uses_with(
            snap, delete_user_cb=lambda u, s=snap, w=its_writes: u is not s and u not in w
        )
        snapshots.append(snap)
    return snapshots


def partition_default(
    joint_gm: GraphModule, *, num_fwd_outputs: int = 1, policy: str = "save_needed"
):
    """

    Joint output args are ``(fwd..., bwd...)``; ``num_fwd_outputs`` marks the
    boundary. ``save_needed`` saves every forward value the backward reads
    (the default partition policy); ``recompute_all`` saves nothing and clones
    producer chains into the backward graph. Returns
    ``(fw_gm, bw_gm, input_kinds, input_keys, saved_names, leaf_targets)``
    where backward inputs carry a role tag for name-based binding.
    """

    joint = joint_gm.graph
    fwd_nodes: List[Node] = []
    bwd_nodes: List[Node] = []
    for node in joint.nodes:
        if node.op == "output":
            continue
        if node.meta.get("is_backward"):
            bwd_nodes.append(node)
        else:
            fwd_nodes.append(node)

    out_args = [a for a in joint.output_node.args]
    if out_args and isinstance(out_args[0], tuple):
        out_args = list(out_args[0])
    user_outputs = out_args[:num_fwd_outputs]
    bwd_out_args = out_args[num_fwd_outputs:]

    # An input the forward writes into and the backward reads is read through
    # a copy taken before the write, which the backward then reads as a kept
    # value.
    snapshots = _snapshot_inputs_read_after_write(
        joint,
        _forward_writes(fwd_nodes),
        _backward_reach(
            bwd_nodes,
            bwd_out_args,
            set() if policy == "recompute_all" else set(fwd_nodes),
        ),
    )
    if snapshots:
        fwd_nodes = [
            n for n in joint.nodes if n.op != "output" and not n.meta.get("is_backward")
        ]

    candidate_saved = [
        n for n in fwd_nodes
        if (policy != "recompute_all" or n in snapshots)
        and n.op not in _LEAF_OPS
        and (n in snapshots or any(u.meta.get("is_backward") for u in n.users))
    ]

    fw_graph, _, _ = _copy_nodes(fwd_nodes, [*user_outputs, *candidate_saved], False)

    if policy == "recompute_all":
        # No saved activations: backward references to forward values are
        # satisfied by cloning the producer chains into the backward graph
        # (the same recompute closure the min-cut extractor uses); only
        # leaves and the tangent stay external.
        bw_graph = Graph()
        bw_map: Dict[Node, Node] = {}
        input_kinds = []
        input_keys = []

        def ensure(node: Node) -> Node:
            if node in bw_map:
                return bw_map[node]
            if node.meta.get("is_subgraph"):
                clone = bw_graph.get_attr(node.target)
                clone.meta.update(node.meta)
                bw_map[node] = clone
                return clone
            if node in snapshots:
                clone = bw_graph.placeholder(node.name)
                clone.meta.update(node.meta)
                bw_map[node] = clone
                input_kinds.append("saved")
                input_keys.append(node.name)
                return clone
            if node.op in _LEAF_OPS:
                clone = bw_graph.placeholder(node.name)
                clone.meta.update(node.meta)
                bw_map[node] = clone
                if node.op == "placeholder" and node.meta.get("is_backward"):
                    input_kinds.append("tangent")
                    input_keys.append(clone.name)
                else:
                    input_kinds.append("leaf")
                    input_keys.append(
                        node.target if isinstance(node.target, str) else node.name
                    )
                return clone
            new_args = tuple(ensure(a) if isinstance(a, Node) else a for a in node.args)
            new_kwargs = {
                k: ensure(v) if isinstance(v, Node) else v
                for k, v in node.kwargs.items()
            }
            clone = bw_graph.create_node(
                node.op, node.target, new_args, new_kwargs, name=node.name
            )
            clone.meta.update(node.meta)
            bw_map[node] = clone
            return clone

        for node in bwd_nodes:
            ensure(node)
    else:
        bw_graph, bw_map, _ = _copy_nodes(bwd_nodes, bwd_out_args, True)

        # Role-tag each auto-generated backward placeholder.
        rev = {v: k for k, v in bw_map.items()}
        input_kinds = []
        input_keys = []
        for p in bw_graph.placeholders:
            old = rev[p]
            if old.meta.get("is_backward"):
                input_kinds.append("tangent")
                input_keys.append(p.name)
            elif old.op in _LEAF_OPS:
                input_kinds.append("leaf")
                input_keys.append(
                    old.target if isinstance(old.target, str) else old.name
                )
            else:
                input_kinds.append("saved")
                input_keys.append(old.name)

    # An input that does not take a gradient has none, and the joint graph
    # says so by having nothing there.  There is no node to carry, so it is
    # not carried: the backward graph produces what it does produce, and the
    # input that wanted no gradient wanted none.
    mapped_bwd_outs = [bw_map[a] for a in bwd_out_args if a is not None]
    bw_graph.output(
        mapped_bwd_outs[0] if len(mapped_bwd_outs) == 1 else tuple(mapped_bwd_outs)
    )

    def _gm(graph: Graph) -> GraphModule:
        sig = inspect.Signature(
            [
                inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for p in graph.placeholders
            ]
        )
        # Rooted on the joint module: attributes the partitioned graphs
        # still read (lifted constants) resolve there.
        return GraphModule(joint_gm, graph, sig)

    return (
        _gm(fw_graph),
        _gm(bw_graph),
        input_kinds,
        input_keys,
        [n.name for n in candidate_saved],
    )


_RECOMPUTABLE_OPS = {
    ("call_function", operator.add),
    ("call_function", operator.sub),
    ("call_function", operator.mul),
    ("call_function", operator.truediv),
    ("call_function", operator.neg),
    ("call_method", "add"),
    ("call_method", "sub"),
    ("call_method", "mul"),
    ("call_method", "truediv"),
    ("call_method", "div"),
    ("call_method", "neg"),
    ("call_method", "relu"),
    ("call_method", "sum"),
    ("call_method", "sin"),
    ("call_method", "cos"),
    ("call_method", "exp"),
}

_INF = float("inf")

#: Operations cheap enough to be computed a second time in the backward
#: rather than kept from the forward: one element at a time, a view, a
#: conversion, a small reduction.  An operation tagged pointwise is one of
#: these whether it is listed or not.
_RECOMPUTABLE_NAMES = frozenset({
    "add", "sub", "div", "truediv", "atan2", "mul", "max", "min", "pow",
    "remainder", "fmod", "__and__", "__or__", "__xor__", "__lshift__",
    "__rshift__", "eq", "ne", "ge", "gt", "le", "lt", "abs", "bitwise_not",
    "ceil", "floor", "frac", "neg", "relu", "round", "silu", "trunc", "log",
    "log10", "log1p", "log2", "lgamma", "exp", "expm1", "erf", "erfc", "cos",
    "acos", "cosh", "sin", "asin", "sinh", "tan", "atan", "tanh", "atanh",
    "sqrt", "rsqrt", "reciprocal", "sigmoid", "softplus", "threshold",
    "threshold_backward", "clamp", "where", "lerp", "addcmul", "gelu",
    "gelu_backward", "sum", "mean", "_grad_sum_to_size", "sum_to_size",
    "amax", "to", "type_as", "getitem", "squeeze", "unsqueeze", "rsub",
    "_to_copy", "clone", "full_like", "var", "std", "select", "_unsafe_view",
    "view", "expand", "slice", "reshape", "broadcast_tensors",
    "scalar_tensor", "ones", "new_zeros", "lift_fresh_copy", "arange", "triu",
    "var_mean", "isinf", "any", "full", "as_strided", "zeros", "empty",
    "empty_like", "argmax", "maximum", "index", "gather", "alias", "t",
    "permute", "split", "chunk", "zeros_like",
})

#: Operations that only re-read memory another value owns.
_VIEW_NAMES = frozenset({
    "squeeze", "unsqueeze", "alias", "view", "slice", "t", "expand",
    "as_strided", "permute", "select", "split", "chunk",
})

#: Operations that draw from a generator: computing one again gives another
#: draw, so they are fusible but never computed twice.
_RANDOM_NAMES = frozenset({"native_dropout", "rand_like", "randn_like"})

#: Farther from the backward than any node of a real graph.
_FAR = int(1e9)


def _op_name(node: Node) -> Optional[str]:
    """The operation a call node names, without its overload."""

    if node.op == "call_method":
        return node.target if isinstance(node.target, str) else None
    if node.op != "call_function":
        return None
    target = node.target
    if target is operator.getitem:
        return "getitem"
    name = getattr(target, "_opname", None) or getattr(target, "__name__", None)
    return name if isinstance(name, str) else None


def _is_recomputable(node: Node) -> bool:
    if (node.op, node.target) in _RECOMPUTABLE_OPS:
        return True
    name = _op_name(node)
    if name is None:
        return False
    if name in _RECOMPUTABLE_NAMES:
        return True
    return "pointwise" in (getattr(node.target, "tags", None) or ())


def _is_view(node: Node) -> bool:
    return _op_name(node) in _VIEW_NAMES


def _is_fusible_op(node: Node) -> bool:
    return _is_recomputable(node) or _op_name(node) in _RANDOM_NAMES


def _can_fuse(a: Node, b: Node) -> bool:
    """Whether ``b`` can be computed in the loop that computes ``a``."""

    # A join reads its operands in place, whatever produced them; it is not
    # itself read in place by what follows.
    if _op_name(b) == "cat":
        return True
    return _is_fusible_op(a) and _is_fusible_op(b)


def _nbytes(val: Any) -> int:
    """The memory a recorded value occupies; nothing for what is not a tensor."""

    if isinstance(val, (list, tuple)):
        return sum(_nbytes(v) for v in val)
    numel = getattr(val, "numel", None)
    dtype = getattr(val, "dtype", None)
    if not callable(numel) or dtype is None:
        return 0
    return int(numel()) * int(getattr(dtype, "itemsize", 4) or 4)


def _is_tensor_val(val: Any) -> bool:
    return hasattr(val, "shape") and hasattr(val, "dtype") and callable(
        getattr(val, "numel", None)
    )


def _choose_saved_values(
    joint: Graph,
    user_outputs: Sequence[Any],
    bwd_out_args: Sequence[Any],
    *,
    heuristics: bool = True,
) -> Optional[set]:
    """The forward values worth keeping for the backward, by a minimum cut.

    Every forward value the backward reads is either kept or computed again
    from values that are kept.  Keeping costs memory and a write in the
    forward; computing again costs arithmetic the backward can do inside the
    loop that reads the result.  The choice is a minimum cut of the forward's
    data flow: each value is an edge weighted by its size, a value that cannot
    be computed again is tied to the source, a value the backward reads is
    tied to the sink, and the cheapest set of edges that separates the two is
    what is kept.

    A value cannot be computed again when its operation is not a cheap one,
    when the backward hands it to something that needs it in memory anyway,
    or when it is a reduction -- small to keep and a whole pass to redo.  A
    value read both nearby and far away, or sitting at the end of a very long
    run of fusible operations, is kept as well, since computing it again
    would drag the whole run into the backward.  Values nearer the backward
    are preferred, and a value that is in memory already is cheaper to keep
    than one that would have to be written for the purpose.

    ``None`` when the graph has no finite cut: something the backward needs
    can neither be kept nor computed again, and the caller keeps everything.
    """

    nodes = [n for n in joint.nodes if n.op != "output"]
    is_bw = lambda n: bool(n.meta.get("is_backward"))
    is_leaf = lambda n: n.op in _LEAF_OPS

    # What the program's own results are computed from.
    required_fw: set = set()
    stack = [o for o in user_outputs if isinstance(o, Node)]
    while stack:
        n = stack.pop()
        if n in required_fw or is_bw(n):
            continue
        required_fw.add(n)
        stack.extend(n.all_input_nodes)
    fw_order: Dict[Node, int] = {}
    for n in nodes:
        if n in required_fw:
            fw_order[n] = len(fw_order)

    # How many forward steps separate a value from its first backward reader.
    dist: Dict[Node, int] = {}
    for n in reversed(list(joint.nodes)):
        if n.op == "output":
            dist[n] = _FAR
        elif n not in required_fw:
            dist[n] = 0
        else:
            dist[n] = min([dist.get(u, _FAR) + 1 for u in n.users] or [_FAR])

    read_by_backward = {a for a in bwd_out_args if isinstance(a, Node) and not is_bw(a)}

    def materialized_in_backward(node: Node) -> bool:
        if _is_view(node):
            return False
        pending = [node]
        seen = {node}
        while pending:
            cur = pending.pop()
            for user in cur.users:
                if user not in required_fw and not _can_fuse(cur, user):
                    return True
                if _is_view(user) and user not in seen:
                    seen.add(user)
                    pending.append(user)
        return False

    def ban_reason(node: Node) -> Optional[str]:
        if node.op not in ("call_function", "call_method"):
            return None
        if _op_name(node) == "getitem":
            return None
        if not _is_recomputable(node):
            return "not a cheap operation"
        if materialized_in_backward(node):
            return "in memory in the backward anyway"
        inputs = sum(_nbytes(a.meta.get("val")) for a in node.args if isinstance(a, Node))
        if _nbytes(node.meta.get("val")) * 4 < inputs:
            return "a reduction"
        return None

    def materialized(node: Node) -> bool:
        if node.op == "placeholder":
            return True
        return not all(_can_fuse(node, user) for user in node.users)

    def weight(node: Node) -> float:
        val = node.meta.get("val")
        if not _is_tensor_val(val):
            # Several values, or none: there is no one buffer to keep.
            return _INF
        size = int(_nbytes(val) * (1.1 ** max(min(dist[node], 100), 1)))
        return size if materialized(node) else size * 2

    source, sink = "__S__", "__T__"
    capacity: Dict[str, Dict[str, float]] = {}
    banned: set = set()

    def edge(u: str, v: str, c: float) -> None:
        capacity.setdefault(u, {})[v] = c

    def ban(node: Node) -> None:
        if _is_view(node) or is_leaf(node) or is_bw(node):
            return
        banned.add(node)
        edge(source, f"{node.name}_in", _INF)

    flow_nodes = [n for n in nodes if not is_bw(n) and not is_leaf(n)]
    for node in flow_nodes:
        if node in required_fw and ban_reason(node):
            ban(node)
        edge(f"{node.name}_in", f"{node.name}_out", weight(node))
        if node in read_by_backward:
            edge(f"{node.name}_out", sink, _INF)
        for user in node.users:
            if user.op == "output":
                continue
            if is_bw(user):
                edge(f"{node.name}_out", sink, _INF)
            else:
                edge(f"{node.name}_out", f"{user.name}_in", _INF)

    if heuristics:
        # A value read both by a nearby operation and by one beyond the first
        # operation that cannot be fused: the far reader is kept, since the
        # two readers would not end up in one loop anyway.
        def first_unfusible(start_nodes: List[Node], max_range: int) -> int:
            heap: List[Tuple[int, int, Node, bool]] = []
            pushed = set()
            for n in start_nodes:
                heapq.heappush(heap, (fw_order[n], id(n), n, True))
            while heap:
                _, _, node, fusible = heapq.heappop(heap)
                if not fusible:
                    return fw_order[node]
                for user in node.users:
                    if user not in required_fw or fw_order[user] > max_range:
                        continue
                    entry = (fw_order[user], id(user), user, _can_fuse(node, user))
                    key = (id(user), entry[3])
                    if key not in pushed:
                        pushed.add(key)
                        heapq.heappush(heap, entry)
            return max_range

        for used in sorted(required_fw, key=fw_order.__getitem__):
            fw_users = [u for u in used.users if u in required_fw]
            if not fw_users:
                continue
            first = first_unfusible(fw_users, max(fw_order[u] for u in fw_users))
            for user in tuple(used.users):
                if (
                    user in required_fw
                    and fw_order[user] > first
                    and _can_fuse(used, user)
                    and user not in banned
                ):
                    ban(user)

        # The end of a very long run of fusible operations is kept, so that a
        # chain running the length of the program is not computed twice.
        visited: set = set()
        for start in nodes:
            if start not in required_fw:
                continue
            start_order = fw_order[start]
            heap2: List[Tuple[int, int, Node]] = [(start_order, id(start), start)]
            while heap2:
                _, _, cur = heapq.heappop(heap2)
                if cur in visited:
                    continue
                visited.add(cur)
                if fw_order[cur] > start_order + 100 and not heap2:
                    ban(cur)
                    break
                for user in cur.users:
                    if user in required_fw and _can_fuse(cur, user) and user not in banned:
                        heapq.heappush(heap2, (fw_order[user], id(user), user))

    # No finite cut when the source reaches the sink through edges that cannot
    # be cut at all.
    reach = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, c in capacity.get(u, {}).items():
            if c == _INF and v not in reach:
                reach.add(v)
                queue.append(v)
    if sink in reach:
        return None

    _, reachable = _mincut_maxflow(capacity, source, sink)
    return {
        n for n in flow_nodes
        if f"{n.name}_in" in reachable and f"{n.name}_out" not in reachable
    }


def _mincut_maxflow(
    capacity: Dict[str, Dict[str, float]], source: str, sink: str
) -> Tuple[float, set]:
    """Edmonds-Karp max flow; returns (flow, residual-reachable set)."""

    total_flow = 0.0
    while True:
        parent: Dict[str, Optional[str]] = {source: None}
        queue = deque([source])
        while queue and sink not in parent:
            u = queue.popleft()
            for v, c in capacity.get(u, {}).items():
                if c > 0 and v not in parent:
                    parent[v] = u
                    queue.append(v)
        if sink not in parent:
            break
        path = []
        v = sink
        while parent[v] is not None:
            u = parent[v]
            path.append((u, v))
            v = u
        aug = min(capacity[u][v] for u, v in path)
        for u, v in path:
            capacity[u][v] -= aug
            capacity.setdefault(v, {}).setdefault(u, 0.0)
            capacity[v][u] += aug
        total_flow += aug
    reachable = {source}
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v, c in capacity.get(u, {}).items():
            if c > 0 and v not in reachable:
                reachable.add(v)
                queue.append(v)
    return total_flow, reachable


def partition_min_cut(
    joint_gm: GraphModule,
    *,
    num_fwd_outputs: int = 1,
    memory_budget: Optional[int] = None,
    ban_fusible_chains: bool = True,
):
    """Split a tagged joint graph, keeping the cheapest cut of forward values.

    A minimum cut over the forward's data flow decides which values the
    backward reads from memory and which it computes again inside its own
    loops (see ``_choose_saved_values``); the backward graph then carries a
    copy of every operation between the kept values and its own.
    ``ban_fusible_chains`` turns on the rules that keep a value read far from
    where it was made or sitting at the end of a very long fusible run.
    ``memory_budget`` is accepted for callers that pass one; the cut is
    always the one that is cheapest to run.
    """

    joint = joint_gm.graph
    fwd_nodes: List[Node] = []
    bwd_nodes: List[Node] = []
    for node in joint.nodes:
        if node.op == "output":
            continue
        if node.meta.get("is_backward"):
            bwd_nodes.append(node)
        else:
            fwd_nodes.append(node)

    out_args = [a for a in joint.output_node.args]
    if out_args and isinstance(out_args[0], tuple):
        out_args = list(out_args[0])
    user_outputs = out_args[:num_fwd_outputs]
    bwd_out_args = out_args[num_fwd_outputs:]

    saved_set = _choose_saved_values(
        joint, user_outputs, bwd_out_args, heuristics=ban_fusible_chains
    )
    if saved_set is None:
        return partition_default(joint_gm, num_fwd_outputs=num_fwd_outputs)

    # A forward operation that writes into something -- an input updated in
    # place, say a buffer counting the batches it has seen -- is held for that
    # write, which nothing reads but the program after the call.  An input it
    # writes and the backward reads, directly or through a value it computes
    # again, is read through a copy taken before the write and kept.
    writes = _forward_writes(fwd_nodes)
    snapshots = _snapshot_inputs_read_after_write(
        joint, writes, _backward_reach(bwd_nodes, bwd_out_args, saved_set)
    )
    if snapshots:
        saved_set = set(saved_set) | set(snapshots)
        fwd_nodes = [
            n for n in joint.nodes if n.op != "output" and not n.meta.get("is_backward")
        ]

    # The forward holds what its results and the kept values are computed
    # from; a value only the backward reads is computed there.
    needed: set = set()
    stack = [o for o in [*user_outputs, *saved_set, *writes] if isinstance(o, Node)]
    while stack:
        n = stack.pop()
        if n in needed:
            continue
        needed.add(n)
        stack.extend(n.all_input_nodes)
    fwd_nodes = [n for n in fwd_nodes if n.op == "placeholder" or n in needed]

    fw_graph, _, _ = _copy_nodes(fwd_nodes, [*user_outputs, *sorted(saved_set, key=lambda x: x.name)], False)

    # Backward graph with recompute closure: references outside the save set
    # are cloned recursively (memoised) instead of auto-placeholdered.
    bw_graph = Graph()
    bw_map: Dict[Node, Node] = {}
    input_kinds: List[str] = []
    input_keys: List[str] = []

    def ensure(node: Node) -> Node:
        if node in bw_map:
            return bw_map[node]
        # Only the explicit input list (saved values + tangent) becomes
        # placeholders; every
        # other reachable node -- including backward-internal ones -- is
        # cloned recursively into the extracted graph.  A placeholder has no
        # producer to clone from, so it is an input of this half whatever the
        # saved set holds: without it a tangent read here is recreated as a
        # node named like an input, which the half does not declare and the
        # lowering cannot resolve.
        if node.meta.get("is_subgraph"):
            clone = bw_graph.get_attr(node.target)
            clone.meta.update(node.meta)
            bw_map[node] = clone
            return clone
        external = (
            node.op in _LEAF_OPS
            or node in saved_set
            or node.op == "placeholder"
        )
        if external:
            clone = bw_graph.placeholder(node.name)
            clone.meta.update(node.meta)
            bw_map[node] = clone
            if node.op == "placeholder" and node.meta.get("is_backward"):
                input_kinds.append("tangent")
                input_keys.append(clone.name)
            elif node.op in _LEAF_OPS:
                input_kinds.append("leaf")
                input_keys.append(
                    node.target if isinstance(node.target, str) else node.name
                )
            else:
                input_kinds.append("saved")
                input_keys.append(node.name)
            return clone
        # Nested arguments carry nodes too -- a multi-output call hands back a
        # tuple, an indexing takes one -- and a node reached inside a
        # container is cloned and mapped like any other, or it crosses over as
        # a reference to a node this half does not hold.
        def remap(value):
            if isinstance(value, Node):
                return ensure(value)
            if isinstance(value, (list, tuple)):
                return type(value)(remap(v) for v in value)
            if isinstance(value, dict):
                return {k: remap(v) for k, v in value.items()}
            return value

        new_args = tuple(remap(a) for a in node.args)
        new_kwargs = {k: remap(v) for k, v in node.kwargs.items()}
        clone = bw_graph.create_node(node.op, node.target, new_args, new_kwargs, name=node.name)
        clone.meta.update(node.meta)
        bw_map[node] = clone
        return clone

    # Everything the backward computes -- its own operations and the forward
    # ones it computes again -- is copied in the order the joint graph holds
    # it, so each copy finds its operands already made and a long run of
    # recomputed operations is not walked by recursion.
    wanted: set = set()
    stack = [*bwd_nodes, *(a for a in bwd_out_args if isinstance(a, Node))]
    while stack:
        n = stack.pop()
        if n in wanted:
            continue
        wanted.add(n)
        if n.op in _LEAF_OPS or n in saved_set:
            continue
        stack.extend(n.all_input_nodes)
    for node in joint.nodes:
        if node in wanted:
            ensure(node)
    # An input that does not take a gradient has none, and the joint graph
    # says so by having nothing there.  There is no node to carry, so it is
    # not carried: the backward graph produces what it does produce, and the
    # input that wanted no gradient wanted none.
    mapped_bwd_outs = [bw_map[a] for a in bwd_out_args if a is not None]
    bw_graph.output(
        mapped_bwd_outs[0] if len(mapped_bwd_outs) == 1 else tuple(mapped_bwd_outs)
    )

    def _gm(graph: Graph) -> GraphModule:
        sig = inspect.Signature(
            [
                inspect.Parameter(p.name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for p in graph.placeholders
            ]
        )
        # Rooted on the joint module: attributes the partitioned graphs
        # still read (lifted constants) resolve there.
        return GraphModule(joint_gm, graph, sig)

    return (
        _gm(fw_graph),
        _gm(bw_graph),
        input_kinds,
        input_keys,
        [n.name for n in sorted(saved_set, key=lambda x: x.name)],
    )


class AotResult:
    """Forward/backward pair plus execution helpers."""

    def __init__(
        self,
        forward_gm: GraphModule,
        backward_gm: GraphModule,
        placeholder_names: Sequence[str],
        leaf_targets: Sequence[str],
        saved_names: Sequence[str],
        input_kinds: Sequence[str] = (),
        input_keys: Sequence[str] = (),
        num_user_outputs: int = 1,
    ) -> None:
        self.forward_gm = forward_gm
        self.backward_gm = backward_gm
        self.placeholder_names = list(placeholder_names)
        self.leaf_targets = list(leaf_targets)
        self.saved_names = list(saved_names)
        self.input_kinds = list(input_kinds)
        self.input_keys = list(input_keys)
        self.num_user_outputs = num_user_outputs

    def forward(self, *args: Any) -> Tuple[Any, Tuple[Any, ...]]:
        outputs = self.forward_gm.forward(*args)
        if not isinstance(outputs, tuple):
            # Zero-saved extraction: the forward graph's single output is
            # returned unwrapped by the interpreter.
            outputs = (outputs,)
        user_out: Any = (
            tuple(outputs[: self.num_user_outputs])
            if self.num_user_outputs > 1
            else outputs[0]
        )
        return user_out, tuple(outputs[self.num_user_outputs :])

    def value_and_grad(
        self, *args: Any, grad_output: Any = None
    ) -> Tuple[Any, Dict[str, Any]]:
        user_out, saved = self.forward(*args)
        if grad_output is None:
            if isinstance(user_out, tuple):
                # Total-derivative tangent for multi-output regions, matching
                # eager `sum(t.sum() for t in outs).backward()`.
                grad_output = sum(o.sum() for o in user_out) * 0 + 1
            else:
                grad_output = user_out * 0 + 1
        # Backward placeholder order interleaves saved values and leaves
        # (extraction auto-placeholders external refs in first-use order),
        # so bind by the role tags recorded at partition time.
        saved_by_name = dict(zip(self.saved_names, saved))
        leaf_args = dict(zip(self.placeholder_names, args))
        kwargs: Dict[str, Any] = {}
        for p, kind, key in zip(
            self.backward_gm.graph.placeholders, self.input_kinds, self.input_keys
        ):
            if kind == "tangent":
                kwargs[p.name] = grad_output
            elif kind == "saved":
                kwargs[p.name] = saved_by_name[key]
            else:
                kwargs[p.name] = leaf_args[key]
        grads = self.backward_gm.forward(**kwargs)
        if len(self.leaf_targets) == 1:
            grads = (grads,)
        return user_out, dict(zip(self.leaf_targets, grads))


def build_aot(
    graph_module: GraphModule,
    *,
    sample_inputs: Dict[str, Any],
    required_grads: Optional[Sequence[str]] = None,
    policy: str = "save_needed",
    partitioner: str = "default",
    memory_budget: Optional[int] = None,
) -> AotResult:
    """Differentiate a captured region into an AOT forward/backward pair.

    ``policy`` selects the save strategy: ``"save_needed"`` stashes exactly
    the forward values the backward reads; ``"recompute_all`` saves nothing
    and rematerializes them inside the backward graph. ``partitioner``
    selects the splitting strategy: ``"default"`` (structural save-need) or
    ``"min_cut"`` (memory-optimal cut, P3-L4b).
    """

    bindings = {
        p.name: sample_inputs[p.name]
        for p in graph_module.graph.placeholders
        if p.name in sample_inputs
    }
    missing = [p.name for p in graph_module.graph.placeholders if p.name not in bindings]
    if missing:
        raise AOTError(f"AOT requires sample inputs for: {sorted(missing)}")
    graph_module._interpret(_record_meta=True, **bindings)

    builder = _JointBuilder(graph_module)
    out_arg = graph_module.graph.output_node.args[0]
    # regions (tuple returns) each receive the unit tangent, which reproduces
    # the total derivative d(sum(out_i))/dx that eager `sum(...).backward()`
    # computes.
    if isinstance(out_arg, (tuple, list)):
        out_elements = list(out_arg)
    else:
        out_elements = [out_arg]
    if not all(isinstance(elem, Node) for elem in out_elements):
        raise AOTError("constant outputs cannot be differentiated")

    adjoint: Dict[Node, List[Node]] = {
        elem: [builder.tangent] for elem in out_elements
    }
    grad_outputs: List[Tuple[str, Node]] = []
    for node in reversed(list(graph_module.graph.nodes)):
        if node.op == "output":
            continue
        contributions = adjoint.pop(node, None)
        if not contributions:
            continue
        go = contributions[0]
        for extra in contributions[1:]:
            go = builder.bwd("call_function", operator.add, (go, extra))
        if node.op in _LEAF_OPS:
            target = node.target if isinstance(node.target, str) else node.name
            grad_outputs.append((target, go))
            continue
        rule = _RULES.get((node.op, node.target))
        if rule is None and node.op == "call_function":
            rule = _RULES.get((node.op, getattr(node.target, "__name__", None)))
        if rule is None:
            raise AOTError(
                f"no derivative registered for {node.op}[{getattr(node.target, '__name__', node.target)}]"
            )
        for input_value, contribution in rule(builder, node, go).items():
            if isinstance(input_value, Node):
                adjoint.setdefault(input_value, []).append(contribution)

    if not grad_outputs:
        raise AOTError("no leaf gradients were computed")
    # Backward outputs follow primal input order -- the reverse sweep
    # discovers leaf gradients bottom-up, so restore the
    # forward placeholder order before emitting the joint output.
    placeholder_order = {
        p.name: idx for idx, p in enumerate(graph_module.graph.placeholders)
    }
    grad_outputs.sort(
        key=lambda item: placeholder_order.get(item[0], len(placeholder_order))
   )
    # Tag every node appended after the original output as backward. Rules
    # emit through several helpers (some via untagged _emit), so a positional
    # sweep is the only reliable way to close the tag set.
    seen_output = False
    for n in graph_module.graph.nodes:
        if n.op == "output":
            seen_output = True
            continue
        if seen_output:
            n.meta["is_backward"] = True
    graph_module.graph.output((out_arg, *[g for _, g in grad_outputs]))

    if partitioner == "min_cut":
        (
            forward_gm,
            backward_gm,
            input_kinds,
            input_keys,
            saved_names_list,
        ) = partition_min_cut(graph_module, memory_budget=memory_budget)
    else:
        (
            forward_gm,
            backward_gm,
            input_kinds,
            input_keys,
            saved_names_list,
        ) = partition_default(graph_module, policy=policy)

    leaf_targets = [name for name, _ in grad_outputs]
    if required_grads is not None:
        missing = set(required_grads) - set(leaf_targets)
        if missing:
            raise AOTError(f"requested gradients unavailable for: {sorted(missing)}")

    return AotResult(
        forward_gm=forward_gm,
        backward_gm=backward_gm,
        placeholder_names=[p.name for p in forward_gm.graph.placeholders],
        leaf_targets=leaf_targets,
        saved_names=saved_names_list,
        input_kinds=input_kinds,
        input_keys=input_keys,
        num_user_outputs=len(out_elements),
    )
