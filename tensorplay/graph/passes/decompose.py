"""Operator decomposition pass (L5-M4).

operators into the primitive set *before* AOT, so the derivative registry
only has to cover primitives. Every expansion below uses ops that already
have local vector-Jacobian rules, which keeps decomposed graphs
differentiable by construction.

This table deliberately lands every rewrite on the Stax/Triton fusible
primitive set (``POINTWISE_FUSED_OP_NAMES``): add/sub/mul/div/pow/neg/exp/
log/sqrt/tanh/relu/sigmoid/square plus scalar constants, and — since the
comparison/``where`` primitives joined the fused surface — the select-based
family (elu/selu/gelu/leaky_relu/relu6/hardtanh/hardsigmoid/hardshrink/
softshrink/threshold/clamp).  A small primitive
set covers a wide operator surface,
so each entry below multiplies *natively compiled* coverage without any
new kernel.  Rules that meet an argument shape they cannot express return
the original node; the backend then handles the operator or falls back.

Rules are keyed by semantic op name and fire on both ``call_method`` and
``call_function`` sites, matching how the tracer records tensor methods
versus ``tensorplay.nn.functional`` wrappers.
"""

from __future__ import annotations

import math
import operator
from typing import Any, Callable, Dict

from tensorplay.graph import Graph, GraphModule, Node
from .base import PassBase, PassResult
from .dead_code_elimination import DeadCodeElimination


_DECOMP_METHODS: Dict[str, Callable[[Graph, Node], Node]] = {}


def _method(name: str):
    def register(fn: Callable[[Graph, Node], Node]) -> None:
        _DECOMP_METHODS[name] = fn

    return register


# --- small builders -------------------------------------------------------


def _unop(graph: Graph, kind: str, x: Any) -> Node:
    """Apply a tensor method (exp/log/sqrt/tanh/...) to ``x``."""

    return graph.create_node("call_method", kind, (x,))


def _binop(graph: Graph, op: Any, lhs: Any, rhs: Any) -> Node:
    return graph.create_node("call_function", op, (lhs, rhs))


# --- existing core rules ----------------------------------------------------


@_method("sigmoid")
def _sigmoid(graph: Graph, node: Node) -> Node:
    """sigmoid(x) -> 1 / (1 + exp(-x))"""
    x = node.args[0]
    neg_x = graph.create_node("call_function", operator.neg, (x,))
    exp_x = _unop(graph, "exp", neg_x)
    one = _binop(graph, operator.add, exp_x, 1)
    return _binop(graph, operator.truediv, 1, one)


@_method("silu")
def _silu(graph: Graph, node: Node) -> Node:
    """silu(x) -> x * sigmoid(x)"""
    x = node.args[0]
    sig = _DECOMP_METHODS["sigmoid"](graph, node)
    return _binop(graph, operator.mul, x, sig)


_DECOMP_METHODS["swish"] = _silu


@_method("reciprocal")
def _reciprocal(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.truediv, 1, x)


@_method("square")
def _square(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.mul, x, x)


# --- activations ------------------------------------------------------------


@_method("softplus")
def _softplus(graph: Graph, node: Node) -> Node:
    """softplus(x) -> log(1 + exp(x))

    log(1 + .) below, so both spellings converge on the same primitive chain.
    """
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    summed = _binop(graph, operator.add, exp_x, 1)
    return _unop(graph, "log", summed)


@_method("mish")
def _mish(graph: Graph, node: Node) -> Node:
    """mish(x) -> x * tanh(softplus(x))"""
    x = node.args[0]
    softplus = _DECOMP_METHODS["softplus"](graph, node)
    tanh_sp = _unop(graph, "tanh", softplus)
    return _binop(graph, operator.mul, x, tanh_sp)


@_method("tanhshrink")
def _tanhshrink(graph: Graph, node: Node) -> Node:
    """tanhshrink(x) -> x - tanh(x)"""
    x = node.args[0]
    tanh_x = _unop(graph, "tanh", x)
    return _binop(graph, operator.sub, x, tanh_x)


# --- logarithm / exponential family -----------------------------------------


@_method("log1p")
def _log1p(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    summed = _binop(graph, operator.add, x, 1)
    return _unop(graph, "log", summed)


@_method("expm1")
def _expm1(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    return _binop(graph, operator.sub, exp_x, 1)


@_method("exp2")
def _exp2(graph: Graph, node: Node) -> Node:
    """exp2(x) -> 2 ** x"""
    x = node.args[0]
    return _binop(graph, operator.pow, 2, x)


@_method("log10")
def _log10(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    log_x = _unop(graph, "log", x)
    return _binop(
        graph, operator.mul, log_x, 1.0 / math.log(10.0)
    )


@_method("logit")
def _logit(graph: Graph, node: Node) -> Node:
    """logit(x) -> log(x) - log(1 - x)

    avoids a division whose numerator/denominator can both underflow.
    """
    x = node.args[0]
    log_x = _unop(graph, "log", x)
    one_minus = _binop(graph, operator.sub, 1, x)
    log_one_minus = _unop(graph, "log", one_minus)
    return _binop(graph, operator.sub, log_x, log_one_minus)


# --- hyperbolic / trig family ------------------------------------------------


@_method("sinh")
def _sinh(graph: Graph, node: Node) -> Node:
    """sinh(x) -> (e - 1/e) / 2"""
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    inv_exp = _binop(graph, operator.truediv, 1, exp_x)
    diff = _binop(graph, operator.sub, exp_x, inv_exp)
    return _binop(graph, operator.mul, diff, 0.5)


@_method("cosh")
def _cosh(graph: Graph, node: Node) -> Node:
    """cosh(x) -> (e + 1/e) / 2"""
    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    inv_exp = _binop(graph, operator.truediv, 1, exp_x)
    total = _binop(graph, operator.add, exp_x, inv_exp)
    return _binop(graph, operator.mul, total, 0.5)


@_method("asinh")
def _asinh(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    sq = _binop(graph, operator.mul, x, x)
    inner = _binop(graph, operator.add, sq, 1)
    root = _unop(graph, "sqrt", inner)
    total = _binop(graph, operator.add, x, root)
    return _unop(graph, "log", total)


@_method("acosh")
def _acosh(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    sq = _binop(graph, operator.mul, x, x)
    inner = _binop(graph, operator.sub, sq, 1)
    root = _unop(graph, "sqrt", inner)
    total = _binop(graph, operator.add, x, root)
    return _unop(graph, "log", total)


@_method("atanh")
def _atanh(graph: Graph, node: Node) -> Node:
    """atanh(x) -> log((1 + x) / (1 - x)) / 2"""
    x = node.args[0]
    num = _binop(graph, operator.add, 1, x)
    den = _binop(graph, operator.sub, 1, x)
    ratio = _binop(graph, operator.truediv, num, den)
    logged = _unop(graph, "log", ratio)
    return _binop(graph, operator.mul, logged, 0.5)


@_method("sec")
def _sec(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    cos_x = _unop(graph, "cos", x)
    return _binop(graph, operator.truediv, 1, cos_x)


@_method("csc")
def _csc(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    sin_x = _unop(graph, "sin", x)
    return _binop(graph, operator.truediv, 1, sin_x)


@_method("cot")
def _cot(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    cos_x = _unop(graph, "cos", x)
    sin_x = _unop(graph, "sin", x)
    return _binop(graph, operator.truediv, cos_x, sin_x)


# --- angle / scaling conversions ---------------------------------------------


@_method("rad2deg")
def _rad2deg(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.mul, x, 180.0 / math.pi)


@_method("deg2rad")
def _deg2rad(graph: Graph, node: Node) -> Node:
    x = node.args[0]
    return _binop(graph, operator.mul, x, math.pi / 180.0)


# --- binary composites --------------------------------------------------------


@_method("squared_difference")
def _squared_difference(graph: Graph, node: Node) -> Node:
    lhs, rhs = node.args[0], node.args[1]
    diff = _binop(graph, operator.sub, lhs, rhs)
    return _binop(graph, operator.mul, diff, diff)


@_method("lerp")
def _lerp(graph: Graph, node: Node) -> Node:
    """lerp(start, end, weight) -> start + weight * (end - start)

    Scalar-weight and tensor-weight spellings both work: ``weight`` flows
    through as an operand and broadcasts like any primitive input.
    """

    args = tuple(node.args) + (
        node.kwargs.get("weight") if node.kwargs else None,
    )
    start, end = args[0], args[1]
    weight = args[2] if len(args) > 2 else None
    if weight is None:
        raise ValueError("lerp decomposition requires a weight argument")
    diff = _binop(graph, operator.sub, end, start)
    scaled = _binop(graph, operator.mul, weight, diff)
    return _binop(graph, operator.add, start, scaled)


@_method("addcmul")
def _addcmul(graph: Graph, node: Node) -> Node:
    """addcmul(input, t1, t2, value=1) -> input + value * t1 * t2"""

    value = node.kwargs.get("value", 1) if node.kwargs else 1
    inp, t1, t2 = node.args[0], node.args[1], node.args[2]
    product = _binop(graph, operator.mul, t1, t2)
    scaled = (
        product
        if value == 1
        else _binop(graph, operator.mul, product, value)
    )
    return _binop(graph, operator.add, inp, scaled)


@_method("addcdiv")
def _addcdiv(graph: Graph, node: Node) -> Node:
    """addcdiv(input, t1, t2, value=1) -> input + value * t1 / t2"""

    value = node.kwargs.get("value", 1) if node.kwargs else 1
    inp, t1, t2 = node.args[0], node.args[1], node.args[2]
    quotient = _binop(graph, operator.truediv, t1, t2)
    scaled = (
        quotient
        if value == 1
        else _binop(graph, operator.mul, quotient, value)
    )
    return _binop(graph, operator.add, inp, scaled)


# --- select-based family (where + comparison primitives) --------------------
#
# These rules are the compare-based rewrites the primitive set used to
# defer: they expand onto ``where``/comparison/order primitives, which the
# Triton code generator fuses like any other pointwise op.


def _scalar_arg(node: Node, position: int, keyword: str, default: Any = None):
    """Read an optional scalar argument from positional args or kwargs."""

    if len(node.args) > position:
        return node.args[position]
    if node.kwargs:
        return node.kwargs.get(keyword, default)
    return default


def _where(graph: Graph, cond: Any, then_value: Any, else_value: Any) -> Node:
    import tensorplay

    return graph.create_node(
        "call_function", tensorplay.where, (cond, then_value, else_value)
    )


def _clamp_bound(graph: Graph, kind: str, value: Any, bound: Any) -> Node:
    """clamp_min/clamp_max against a scalar or tensor bound.

    Scalar bounds go through the tensor method (the functional minimum/
    maximum kernels require tensor operands), tensor bounds through the
    functional form; both spellings land on the same fused order op.
    """

    if isinstance(bound, Node):
        import tensorplay

        return graph.create_node(
            "call_function", getattr(tensorplay, kind), (value, bound)
        )
    return graph.create_node("call_method", kind, (value, bound))


@_method("elu")
def _elu(graph: Graph, node: Node) -> Node:
    """elu(x, α) -> where(x > 0, x, α (eˣ - 1)); scale/input_scale stay raw."""

    x = node.args[0]
    alpha = _scalar_arg(node, 1, "alpha", 1.0)
    scale = _scalar_arg(node, 2, "scale", 1.0)
    input_scale = _scalar_arg(node, 3, "input_scale", 1.0)
    if scale != 1 or input_scale != 1:
        return node
    exp_x = _unop(graph, "exp", x)
    shifted = _binop(graph, operator.sub, exp_x, 1)
    branch = _binop(graph, operator.mul, alpha, shifted)
    gate = _binop(graph, operator.gt, x, 0)
    return _where(graph, gate, x, branch)


@_method("selu")
def _selu(graph: Graph, node: Node) -> Node:
    """selu(x) -> s where(x > 0, x, α (eˣ - 1)) with the fixed SELU constants."""

    x = node.args[0]
    exp_x = _unop(graph, "exp", x)
    shifted = _binop(graph, operator.sub, exp_x, 1)
    branch = _binop(
        graph, operator.mul, 1.6732632423543772, shifted
    )
    gate = _binop(graph, operator.gt, x, 0)
    selected = _where(graph, gate, x, branch)
    return _binop(graph, operator.mul, 1.0507009873554805, selected)


@_method("gelu")
def _gelu(graph: Graph, node: Node) -> Node:
    """gelu(x) -> x Φ(x); Φ(x) = 0.5 (1 + erf(x / √2)), or the tanh form."""

    x = node.args[0]
    approximate = (node.kwargs or {}).get("approximate", "none")
    if approximate == "tanh":
        # 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))
        square = _binop(graph, operator.mul, x, x)
        cube = _binop(graph, operator.mul, x, square)
        inner = _binop(
            graph, operator.add, x, _binop(graph, operator.mul, 0.044715, cube)
        )
        scaled = _binop(graph, operator.mul, math.sqrt(2.0 / math.pi), inner)
        tanh = _unop(graph, "tanh", scaled)
        one = _binop(graph, operator.add, 1, tanh)
        half = _binop(graph, operator.mul, 0.5, one)
        return _binop(graph, operator.mul, x, half)
    if approximate not in (None, "none"):
        return node
    root = _binop(graph, operator.mul, x, 0.7071067811865476)
    erf = _unop(graph, "erf", root)
    one = _binop(graph, operator.add, 1, erf)
    half = _binop(graph, operator.mul, 0.5, one)
    return _binop(graph, operator.mul, x, half)


@_method("leaky_relu")
def _leaky_relu(graph: Graph, node: Node) -> Node:
    """leaky_relu(x, s) -> where(x > 0, x, s x)"""

    x = node.args[0]
    slope = _scalar_arg(node, 1, "negative_slope", 0.01)
    gate = _binop(graph, operator.gt, x, 0)
    branch = _binop(graph, operator.mul, x, slope)
    return _where(graph, gate, x, branch)


@_method("relu6")
def _relu6(graph: Graph, node: Node) -> Node:
    """relu6(x) -> min(relu(x), 6)"""

    x = node.args[0]
    gate = _binop(graph, operator.gt, x, 0)
    selected = _where(graph, gate, x, 0.0)
    return _clamp_bound(graph, "clamp_max", selected, 6.0)


@_method("hardtanh")
def _hardtanh(graph: Graph, node: Node) -> Node:
    """hardtanh(x, lo, hi) -> min(max(x, lo), hi)"""

    x = node.args[0]
    lo = _scalar_arg(node, 1, "min_val", -1.0)
    hi = _scalar_arg(node, 2, "max_val", 1.0)
    lower = _clamp_bound(graph, "clamp_min", x, lo)
    return _clamp_bound(graph, "clamp_max", lower, hi)


@_method("hardsigmoid")
def _hardsigmoid(graph: Graph, node: Node) -> Node:
    """hardsigmoid(x) -> clamp((x + 3) / 6, 0, 1)"""

    x = node.args[0]
    shifted = _binop(graph, operator.add, x, 3)
    scaled = _binop(graph, operator.truediv, shifted, 6)
    lower = _clamp_bound(graph, "clamp_min", scaled, 0)
    return _clamp_bound(graph, "clamp_max", lower, 1)


@_method("hardshrink")
def _hardshrink(graph: Graph, node: Node) -> Node:
    """hardshrink(x, λ) -> where(|x| > λ, x, 0)"""

    x = node.args[0]
    lambd = _scalar_arg(node, 1, "lambd", 0.5)
    magnitude = _unop(graph, "abs", x)
    gate = _binop(graph, operator.gt, magnitude, lambd)
    return _where(graph, gate, x, 0.0)


@_method("softshrink")
def _softshrink(graph: Graph, node: Node) -> Node:
    """softshrink(x, λ) = x ∓ λ outside [-λ, λ], 0 inside; nested selects."""

    x = node.args[0]
    lambd = _scalar_arg(node, 1, "lambd", 0.5)
    upper_gate = _binop(graph, operator.gt, x, lambd)
    upper_branch = _binop(graph, operator.sub, x, lambd)
    lower_gate = _binop(graph, operator.lt, x, -lambd)
    lower_branch = _binop(graph, operator.add, x, lambd)
    inner = _where(graph, lower_gate, lower_branch, 0.0)
    return _where(graph, upper_gate, upper_branch, inner)


@_method("threshold")
def _threshold(graph: Graph, node: Node) -> Node:
    """threshold(x, t, v) -> where(x > t, x, v)"""

    x = node.args[0]
    level = _scalar_arg(node, 1, "threshold")
    value = _scalar_arg(node, 2, "value")
    if level is None or value is None:
        return node
    gate = _binop(graph, operator.gt, x, level)
    return _where(graph, gate, x, value)


@_method("clamp")
def _clamp(graph: Graph, node: Node) -> Node:
    """clamp(x, min, max) -> min(max(x, min), max); a None bound drops a side."""

    x = node.args[0]
    lo = _scalar_arg(node, 1, "min")
    hi = _scalar_arg(node, 2, "max")
    if lo is None and hi is None:
        return node
    result = x
    if lo is not None:
        result = _clamp_bound(graph, "clamp_min", result, lo)
    if hi is not None:
        result = _clamp_bound(graph, "clamp_max", result, hi)
    return result


# --- row-normalization rewrites --------------------------------------------
#
# These land on the same primitive set as the table above, but they are kept
# out of the default pipeline on purpose: each composite here already has a
# dedicated fused kernel, so expanding it unconditionally would trade one
# pass over the input for several.  A backend that folds the expansion back
# into a single kernel -- and the surrounding region with it -- opts in
# through :class:`DecomposeRowNormalizations`.

_ROW_NORM_METHODS: Dict[str, Callable[[Graph, Node], Node]] = {}


def _row_norm(name: str):
    def register(fn: Callable[[Graph, Node], Node]) -> None:
        _ROW_NORM_METHODS[name] = fn

    return register


def _softmax_arguments(node: Node) -> Any:
    """Read ``(dim,)`` from a softmax-family call, or ``None`` to decline.

    The trailing positional arguments are ``dim`` then ``dtype``; a request
    to compute in another dtype needs a conversion these rules do not emit.
    """

    rest = list(node.args[1:])
    kwargs = dict(node.kwargs or {})
    if len(rest) > 2:
        return None
    dim: Any = rest[0] if rest else None
    dtype: Any = rest[1] if len(rest) >= 2 else None
    if "dim" in kwargs:
        if dim is not None:
            return None
        dim = kwargs.pop("dim")
    if "dtype" in kwargs:
        if len(rest) >= 2:
            return None
        dtype = kwargs.pop("dtype")
    if kwargs:
        return None
    if dtype is not None and str(dtype).rsplit(".", 1)[-1].lower() != "undefined":
        return None
    if dim is None or isinstance(dim, bool) or not isinstance(dim, int):
        return None
    return (int(dim),)


def _shifted_row(graph: Graph, x: Any, dim: int) -> tuple[Node, Node]:
    """``x`` minus its row maximum, together with the exponential of it.

    Subtracting the maximum is what keeps the exponential finite for large
    inputs; it cancels exactly in both quotients below.
    """

    largest = graph.create_node("call_method", "amax", (x, [dim], True))
    shifted = _binop(graph, operator.sub, x, largest)
    return shifted, _unop(graph, "exp", shifted)


@_row_norm("softmax")
def _softmax(graph: Graph, node: Node) -> Node:
    """softmax(x, dim) -> e / sum(e, dim), e = exp(x - amax(x, dim))"""

    parsed = _softmax_arguments(node)
    if parsed is None:
        return node
    (dim,) = parsed
    _, exponentials = _shifted_row(graph, node.args[0], dim)
    total = graph.create_node("call_method", "sum", (exponentials, dim, True))
    return _binop(graph, operator.truediv, exponentials, total)


@_row_norm("log_softmax")
def _log_softmax(graph: Graph, node: Node) -> Node:
    """log_softmax(x, dim) -> (x - m) - log(sum(exp(x - m), dim)), m the row max"""

    parsed = _softmax_arguments(node)
    if parsed is None:
        return node
    (dim,) = parsed
    shifted, exponentials = _shifted_row(graph, node.args[0], dim)
    total = graph.create_node("call_method", "sum", (exponentials, dim, True))
    return _binop(graph, operator.sub, shifted, _unop(graph, "log", total))


def _rewrite(
    graph_module: GraphModule, table: Dict[str, Callable[[Graph, Node], Node]]
) -> PassResult:
    graph = graph_module.graph
    changed = False
    for node in list(graph.nodes):
        if node.op == "call_method":
            name = node.target if isinstance(node.target, str) else None
        elif node.op == "call_function":
            name = getattr(node.target, "__name__", None)
        else:
            continue
        if name is None:
            continue
        rule = table.get(name)
        if rule is None:
            continue
        # Replacement sub-chains must precede the replaced node's users:
        # create them directly before the original site.  inserting_before
        with graph.inserting_before(node):
            replacement = rule(graph, node)
        if replacement is node:
            # Rule declined this spelling (unsupported argument shape);
            # leave the operator for the backend to handle or fall back.
            continue
        node.replace_all_uses_with(replacement)
        changed = True
    if not changed:
        return PassResult(graph_module, False)
    DeadCodeElimination()(graph_module)
    return PassResult(graph_module, True)


class DecomposePass(PassBase):
    """Rewrite registered composite methods into derivative-covered primitives."""

    def __call__(self, graph_module: GraphModule) -> PassResult:
        return _rewrite(graph_module, _DECOMP_METHODS)


class DecomposeRowNormalizations(PassBase):
    """Expand the softmax family into reductions and elementwise primitives.

    Opt-in: run it only where the expansion is going to be fused back into
    one kernel, since the composites it replaces are single fused kernels
    themselves.
    """

    def __call__(self, graph_module: GraphModule) -> PassResult:
        return _rewrite(graph_module, _ROW_NORM_METHODS)


def row_normalization_names() -> frozenset:
    """Operator names :class:`DecomposeRowNormalizations` knows how to expand."""

    return frozenset(_ROW_NORM_METHODS)
