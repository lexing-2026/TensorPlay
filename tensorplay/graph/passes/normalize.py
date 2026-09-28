"""Canonical graph normalization rules.

Canonical forms make downstream pattern matching (fusion hints, decomposition
tables, codegen templates) reliable:

- commutative binaries carry their scalar constant on the right
  (``2 * x`` -> ``x * 2``), so identity/constant patterns only need one side;
- algebraic identities are folded away: ``x + 0``, ``x - 0``, ``x * 1``,
  ``x / 1``, ``x ** 1``, ``neg(neg(x))`` -> ``x``;
- augmented assignments whose mutation cannot be observed (the mutated
  operand has no other use in the graph) rewrite to the out-of-place op.

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

# ``a op= b`` mutates ``a``; outside of the expression itself the mutation is
# only observable through other live references to that same tensor.  When the
# captured graph's sole remaining use of the operand is the in-place node, no
# reference can observe it and the node rewrites to the out-of-place op.
_INPLACE_BINARY = {
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.imul: operator.mul,
    operator.itruediv: operator.truediv,
}


def _lookup_by_identity(table: dict[Any, Any], target: Any) -> Any:
    """The entry whose key *is* ``target``, or ``None`` if there is none.

    Written as a walk rather than a lookup because the key must be found by
    being the same object, and asking a dict for that would first demand the
    key be hashable. A target naming an operation a type provides carries the
    binding that class gave it, which need not be hashable, and a rewrite keyed
    on a particular operation should not be put off by targets of another kind
    appearing alongside it in the same graph.
    """

    for key, value in table.items():
        if key is target:
            return value
    return None


def _is_one_of(target: Any, candidates: frozenset) -> bool:
    return any(target is candidate for candidate in candidates)


def _inplace_binary_of(target: Any) -> Any:
    return _lookup_by_identity(_INPLACE_BINARY, target)


def _identity_right_of(target: Any) -> Any:
    return _lookup_by_identity(_IDENTITY_RIGHT, target)


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
            # What is being asked here is whether the target is one of a short
            # list of particular operations, so it is compared against them by
            # identity rather than looked up. Looking a target up in a table
            # requires it to be hashable, and a target that names an operation
            # a type provides is not required to be: it can carry the binding
            # its class gave it, and that makes it unfit to be a table key. A
            # rewrite that is about a particular operation should not depend on
            # the others being of a different kind.
            if _is_one_of(target, _COMMUTATIVE) and len(node.args) == 2:
                lhs, rhs = node.args
                if _is_scalar_literal(lhs) and not _is_scalar_literal(rhs):
                    if _is_node_like(rhs):
                        node.args = (rhs, lhs)
                        modified = True
                        continue

            inplace = _inplace_binary_of(target)
            if inplace is not None and len(node.args) == 2:
                lhs = node.args[0]
                if (
                    _is_node_like(lhs)
                    and len(lhs.users) == 1
                    and next(iter(lhs.users)) is node
                ):
                    node.target = inplace
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

            identity_right = _identity_right_of(target)
            if identity_right is not None and len(node.args) == 2:
                rhs = node.args[1]
                if _is_scalar_literal(rhs) and rhs == identity_right:
                    replacement = node.args[0]
                    if _is_node_like(replacement):
                        _replace_node_everywhere(graph, node, replacement)
                        modified = True
                        continue
        return PassResult(graph_module, modified)
