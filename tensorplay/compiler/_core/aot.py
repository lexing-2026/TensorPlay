"""Ahead-of-time graph partitioning over the canonical graph (L4, v2).

local vector-Jacobian rules append tagged backward nodes into the SAME
joint graph (``meta["is_backward"]``), then a partitioner extracts the
forward/backward pair by tag membership -- ``partition_default`` today,
a min-cut strategy behind the same signature later.
"""

from __future__ import annotations

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
    for _ in range(max(0, extra)):
        grad = _emit(graph, "call_method", "sum", (grad,), {"dim": 0})
        current_shape = current_shape[1:]
    if tuple(current_shape) != tuple(target_shape):
        grad = _emit(
            graph, "call_method", "reshape", (grad,), {"shape": tuple(target_shape)}
        )
    return grad


def _ones_like(graph: Graph, value: Node) -> Node:
    zero = _emit(graph, "call_function", operator.mul, (value, 0))
    return _emit(graph, "call_function", operator.add, (zero, 1))


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
            if value not in mapping:
                if not external_as_inputs:
                    raise AOTError(f"unmapped node {value.name} during extraction")
                mapping[value] = graph.placeholder(value.name)
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
        clone.meta.update({k: v for k, v in node.meta.items() if k != "val"})
        mapping[node] = clone
    graph.output(tuple(remap(o) for o in outputs) if len(outputs) > 1 else remap(outputs[0]))
    return graph, mapping, [mapping[n] for n in nodes]


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

    candidate_saved = [
        n for n in fwd_nodes
        if policy != "recompute_all"
        and n.op not in _LEAF_OPS
        and any(u.meta.get("is_backward") for u in n.users)
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
            if node.op in _LEAF_OPS:
                clone = bw_graph.placeholder(node.name)
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
            clone.meta.update(
                {k: v for k, v in node.meta.items() if k != "val"}
            )
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

    mapped_bwd_outs = [bw_map[a] for a in bwd_out_args]
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
        return GraphModule(None, graph, sig)

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
    """Memory-optimal split of a tagged joint graph (P3-L4b design).

    Min-cut over the forward DAG decides which backward-referenced values are
    saved vs recomputed inside the backward graph; must-save nodes carry an
    infinite-capacity edge so they are never cut.
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

    def _weight(node: Node) -> float:
        if memory_budget is None:
            return 1.0
        val = node.meta.get("val")
        numel = getattr(val, "numel", None)
        n = numel() if callable(numel) else 1
        return float(max(1, int(n) * 4))

    has_bw_user = lambda n: any(u.meta.get("is_backward") for u in n.users)

    # ban_fusible_chains: interior nodes of a recomputable chain are forced
    # into the saved set so chains are cut at boundaries only.
    chain_interior = set()
    if ban_fusible_chains:
        for n in fwd_nodes:
            if n.op in _LEAF_OPS or (n.op, n.target) not in _RECOMPUTABLE_OPS:
                continue
            producers = [
                a for a in n.args if isinstance(a, Node) and a.op not in _LEAF_OPS
            ]
            interior = (
                all((p.op, p.target) in _RECOMPUTABLE_OPS for p in producers)
                and producers
                and any((u.op, u.target) in _RECOMPUTABLE_OPS for u in n.users)
            )
            if interior:
                chain_interior.add(n)

    candidates = [n for n in fwd_nodes if n.op not in _LEAF_OPS and has_bw_user(n)]

    # Must-save: get_attr (params), fusible-chain interiors, and any node
    # consumed by BOTH the forward-output subtree and backward (dual-use).
    user_set = {o for o in user_outputs if isinstance(o, Node)}
    fw_needed = set()
    stack = list(user_set)
    while stack:
        n = stack.pop()
        if n in fw_needed or n.op in _LEAF_OPS:
            continue
        fw_needed.add(n)
        stack.extend(a for a in n.args if isinstance(a, Node))
    must_save = {
        n for n in candidates
        if n.op == "get_attr"
        or n in chain_interior
        or n in fw_needed
    }

    source, sink = "__S__", "__T__"
    capacity: Dict[str, Dict[str, float]] = {}

    def _edge(u: str, v: str, c: float) -> None:
        capacity.setdefault(u, {})[v] = c

    for out_arg in user_outputs:
        if isinstance(out_arg, Node) and out_arg.op not in _LEAF_OPS:
            _edge(source, f"n_{out_arg.name}", _INF)
    for n in fwd_nodes:
        if n.op in _LEAF_OPS:
            continue
        key = f"n_{n.name}"
        for a in n.args:
            if isinstance(a, Node) and a.op not in _LEAF_OPS:
                _edge(f"n_{a.name}", key, _INF)
        if n in candidates:
            _edge(key, sink, _INF if n in must_save else _weight(n))

    _, reachable = _mincut_maxflow(capacity, source, sink)
    # Sink-side candidates (unreachable in the residual graph) keep their
    # intact save edges -> saved; reachable ones were cut -> recomputed.
    saved_set = {
        n for n in candidates
        if f"n_{n.name}" not in reachable or n in must_save
    }
    # An empty save set is legal when every backward formula reads only
    # metadata or inputs (e.g. sum's expand-only derivative): the backward
    # then recomputes/clones everything it needs.

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
        # cloned recursively into the extracted graph.
        external = (
            node.op in _LEAF_OPS
            or node in saved_set
        )
        if external:
            clone = bw_graph.placeholder(node.name)
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
        new_args = tuple(ensure(a) if isinstance(a, Node) else a for a in node.args)
        new_kwargs = {
            k: ensure(v) if isinstance(v, Node) else v
            for k, v in node.kwargs.items()
        }
        clone = bw_graph.create_node(node.op, node.target, new_args, new_kwargs, name=node.name)
        bw_map[node] = clone
        return clone

    for node in bwd_nodes:
        ensure(node)
    mapped_bwd_outs = [bw_map[a] for a in bwd_out_args]
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
        return GraphModule(None, graph, sig)

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
