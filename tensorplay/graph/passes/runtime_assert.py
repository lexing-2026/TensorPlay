"""Insert explicit runtime checks collected during graph analysis."""

from __future__ import annotations

import dis
import math
import sys
from collections.abc import Iterable, Mapping
from typing import Any

from ..graph import GraphCaptureError
from ..graph_module import GraphModule
from ..node import Node

__all__ = ["insert_deferred_runtime_asserts"]


def _get_example_value(node: Node) -> Any:
    value = node.meta.get("val")
    return value if value is not None else node.meta.get("example_value")


def _get_sym_val(node: Node) -> Any:
    import sympy

    from ..experimental.sym_node import SymNode

    value = _get_example_value(node)
    if isinstance(value, SymNode):
        return value.expr
    if isinstance(value, sympy.Basic):
        return value
    return None


def _assertion_condition(node: Node) -> Any:
    if node.args:
        return node.args[0]
    return node.kwargs.get("cond", node.kwargs.get("self"))


def _expression_from_value(value: Any) -> Any:
    import sympy

    from ..experimental.sym_node import SymNode

    if isinstance(value, SymNode):
        return value.expr
    if isinstance(value, sympy.Basic):
        return value
    return None


def _static_string_from_callable(fn: Any) -> str | None:
    code = getattr(fn, "__code__", None)
    if (
        code is None
        or code.co_argcount != 0
        or code.co_posonlyargcount != 0
        or code.co_kwonlyargcount != 0
    ):
        return None

    ignored_opnames = {
        "CACHE",
        "COPY_FREE_VARS",
        "EXTENDED_ARG",
        "NOP",
        "RESUME",
    }
    instructions = [
        instruction
        for instruction in dis.get_instructions(fn)
        if instruction.opname not in ignored_opnames
    ]
    if (
        len(instructions) == 1
        and instructions[0].opname == "RETURN_CONST"
        and isinstance(instructions[0].argval, str)
    ):
        return instructions[0].argval
    if (
        len(instructions) == 2
        and instructions[0].opname == "LOAD_CONST"
        and isinstance(instructions[0].argval, str)
        and instructions[1].opname == "RETURN_VALUE"
    ):
        return instructions[0].argval
    return None


def _assertion_message(
    graph_module: GraphModule,
    node: Node,
    condition: Any,
    expression: Any,
    assert_target: Any,
) -> str:
    if len(node.args) > 1:
        message = node.args[1]
    elif node.target is assert_target:
        message = node.kwargs.get("message")
    else:
        message = node.kwargs.get("assert_msg")
    if isinstance(message, str):
        return message
    if isinstance(message, Node) and message.op == "get_attr":
        value: Any = graph_module
        if not isinstance(message.target, str):
            raise AssertionError(
                f"expected a string message target, got {type(message.target).__name__}"
            )
        for part in message.target.split("."):
            value = getattr(value, part)
        if callable(value):
            static_message = _static_string_from_callable(value)
            if static_message is not None:
                return static_message
        if isinstance(value, str):
            return value
    return f"Runtime assertion failed for expression {expression} on node '{condition}'"


def _normalise_expression(shape_env: Any, expression: Any) -> Any:
    replace = getattr(shape_env, "replace", None)
    return replace(expression) if callable(replace) else expression


def _iter_deferred(assertions: Any) -> Iterable[tuple[Any, Any, str | None, Any]]:
    if isinstance(assertions, Mapping):
        for symbol, values in assertions.items():
            if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
                values = (values,)
            for value in values:
                expression = getattr(value, "expr", value)
                message = getattr(value, "msg", None)
                stack = getattr(value, "stack", None)
                yield symbol, expression, message, stack
        return
    if assertions is None:
        return
    if isinstance(assertions, (str, bytes)) or not isinstance(assertions, Iterable):
        assertions = (assertions,)
    for value in assertions:
        expression = getattr(value, "expr", value)
        message = getattr(value, "msg", None)
        stack = getattr(value, "stack", None)
        yield None, expression, message, stack


def _is_finite_bound(value: Any, *, lower: bool) -> bool:
    import sympy

    if value is None:
        return False
    if lower and value in (-sympy.oo, -math.inf):
        return False
    if not lower and value in (sympy.oo, math.inf):
        return False
    if not lower and isinstance(value, int) and value == sys.maxsize - 1:
        return False
    return True


def _node_expression(node: Node) -> Any:
    condition = _assertion_condition(node)
    if isinstance(condition, Node):
        return _get_sym_val(condition)
    return _expression_from_value(condition)


def insert_deferred_runtime_asserts(
    graph_module: GraphModule,
    shape_env: Any,
    name: str,
    export: bool = False,
) -> None:
    """Materialize deferred symbolic conditions as executable graph nodes."""

    import sympy

    from ...functional import (
        _assert_scalar,
        sym_constrain_range,
        sym_constrain_range_for_size,
    )

    del export
    if shape_env is None:
        raise GraphCaptureError("a shape environment is required")
    graph = graph_module.graph
    deferred = getattr(shape_env, "deferred_runtime_asserts", None)
    if deferred is None:
        deferred = getattr(shape_env, "runtime_asserts", ())
    deferred_items = list(_iter_deferred(deferred))

    original_nodes = list(graph.nodes)
    user_assert_exprs: set[Any] = set()
    seen_assert_exprs: set[Any] = set()
    for node in original_nodes:
        if node.op != "call_function" or node.target is not _assert_scalar:
            continue
        condition = _assertion_condition(node)
        if condition is True or condition is sympy.true:
            if not node.users:
                graph.erase_node(node)
            continue
        expression = _node_expression(node)
        if expression is None:
            continue
        expression = _normalise_expression(shape_env, expression)
        if expression in seen_assert_exprs:
            if not node.users:
                graph.erase_node(node)
            continue
        seen_assert_exprs.add(expression)
        user_assert_exprs.add(expression)
        node.meta["runtime_assert_expr"] = expression

    for node in list(graph.nodes):
        if node.op != "call_function":
            continue
        if node.target not in (sym_constrain_range, sym_constrain_range_for_size):
            continue
        if node.users:
            raise GraphCaptureError(
                f"cannot remove range constraint node {node.name} with users"
            )
        graph.erase_node(node)

    expression_to_node: dict[Any, Node] = {}
    for node in list(graph.nodes):
        if node.op in {"placeholder", "output"}:
            continue
        expression = _get_sym_val(node)
        if expression is None:
            continue
        expression = _normalise_expression(shape_env, expression)
        if not getattr(expression, "free_symbols", ()):
            continue
        if node.meta.get("unbacked_bindings"):
            expression_to_node.setdefault(expression, node)
            continue
        previous = expression_to_node.get(expression)
        if previous is None:
            expression_to_node[expression] = node
            continue
        node.replace_all_uses_with(previous)
        if not node.users:
            graph.erase_node(node)

    symbol_nodes: dict[Any, Node] = {}
    for node in graph.nodes:
        expression = _get_sym_val(node)
        if isinstance(expression, sympy.Symbol):
            symbol_nodes.setdefault(expression, node)

    pending: list[tuple[Any, str, Any]] = []
    pending_exprs: set[Any] = set()

    def add_pending(expression: Any, message: str | None, stack: Any = None) -> None:
        expression = _normalise_expression(shape_env, expression)
        if expression in user_assert_exprs or expression in pending_exprs:
            return
        pending_exprs.add(expression)
        if message is None:
            message = f"Runtime assertion failed for expression {expression} in {name}"
        pending.append((expression, message, stack))

    for symbol, expression, message, stack in deferred_items:
        del symbol
        add_pending(expression, message, stack)

    ranges = getattr(shape_env, "var_to_range", {})
    range_symbols = {
        symbol
        for symbol, _, _, _ in deferred_items
        if isinstance(symbol, sympy.Symbol)
    }
    range_symbols.update(
        symbol
        for symbol in getattr(shape_env, "unbacked_inputs", set())
        if symbol in symbol_nodes
    )
    for symbol in sorted(range_symbols, key=str):
        value_range = ranges.get(symbol)
        if value_range is None:
            continue
        lower = getattr(value_range, "lower", None)
        upper = getattr(value_range, "upper", None)
        if _is_finite_bound(lower, lower=True):
            add_pending(
                sympy.Ge(symbol, lower),
                f"Runtime assertion failed for expression {symbol} >= {lower}",
            )
        if _is_finite_bound(upper, lower=False):
            add_pending(
                sympy.Le(symbol, upper),
                f"Runtime assertion failed for expression {symbol} <= {upper}",
            )

    if pending:
        output = graph.output_node
        expressions = [expression for expression, _, _ in pending]
        with graph.inserting_before(output):
            values = graph.materialize_symints(expressions)
            for (expression, message, stack), value in zip(pending, values):
                if isinstance(value, bool) and value:
                    continue
                assertion = graph.call_function(_assert_scalar, (value, message))
                assertion.meta["runtime_assert_expr"] = expression
                if stack is not None:
                    assertion.meta["stack_trace"] = stack
                source = symbol_nodes.get(
                    next(iter(getattr(expression, "free_symbols", ())), None)
                )
                if source is not None:
                    for key in ("nn_module_stack", "custom"):
                        if key in source.meta:
                            assertion.meta[key] = source.meta[key]

    graph.lint()
    recompile = getattr(graph_module, "recompile", None)
    if callable(recompile):
        recompile()
