"""Canonical graph normalization rules.

Canonical forms make downstream pattern matching (fusion hints, decomposition
tables, codegen templates) reliable:

- commutative binaries carry their scalar constant on the right
  (``2 * x`` -> ``x * 2``), so identity/constant patterns only need one side;
- algebraic identities are folded away: ``x + 0``, ``x - 0``, ``x * 1``,
  ``x / 1``, ``x ** 1``, ``neg(neg(x))`` -> ``x``.

``x * 0`` is deliberately NOT folded: for floats it propagates NaN/Inf and
no-op by construction.
"""

from __future__ import annotations

import operator
from typing import Any

from .base import PassBase, PassResult

__all__ = ["NormalizeOperators"]

_COMMUTATIVE = frozenset({operator.add, operator.mul})

_IDENTITY_RIGHT = {
    operator.add: 0,
    operator.sub: 0,
    operator.mul: 1,
    operator.truediv: 1,
    operator.pow: 1,
}


def _is_scalar_literal(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_node_like(value: Any) -> bool:
    return hasattr(value, "meta") and hasattr(value, "erase_node")


def _replace_node_everywhere(graph, node, replacement) -> None:
    """Point every consumer AND the graph outputs at ``replacement``, then erase."""
    for out in graph.outputs:
        new_args = tuple(replacement if a is node else a for a in out.args)
        if new_args != out.args:
            out.args = new_args
    node.replace_all_uses_with(replacement)
    node.erase_node()


class NormalizeOperators(PassBase):
    """Rewrite the graph into the canonical form described above."""

    def __call__(self, graph_module) -> PassResult:
        modified = False
        graph = graph_module.graph
        for node in list(graph.nodes):
            if node.op != "call_function":
                continue
            target = node.target

            if target in _COMMUTATIVE and len(node.args) == 2:
                lhs, rhs = node.args
                if _is_scalar_literal(lhs) and not _is_scalar_literal(rhs):
                    if _is_node_like(rhs):
                        node.args = (rhs, lhs)
                        modified = True
                        continue

            if target is operator.neg and len(node.args) == 1:
                inner = node.args[0]
                if (
                    isinstance(inner, type(node))
                    and inner.op == "call_function"
                    and inner.target is operator.neg
                    and len(inner.args) == 1
                ):
                    # neg(neg(x)) == x, even when the outer neg feeds the
                    # graph output directly (outputs get rewritten too).
                    _replace_node_everywhere(graph, node, inner.args[0])
                    if not inner.users:
                        inner.erase_node()
                    modified = True
                    continue

            if target in _IDENTITY_RIGHT and len(node.args) == 2:
                rhs = node.args[1]
                if _is_scalar_literal(rhs) and rhs == _IDENTITY_RIGHT[target]:
                    replacement = node.args[0]
                    if _is_node_like(replacement):
                        _replace_node_everywhere(graph, node, replacement)
                        modified = True
                        continue
        return PassResult(graph_module, modified)
