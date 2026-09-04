"""Constraint accumulation with suggested fixes for dynamic shapes.

Export-time dimension contracts arrive as a stream of range bounds, naming
ties, and linear relations.  This module collects that stream, checks it for
consistency (overlapping ranges, matching relations, sizes implied by the
example inputs), and renders repair suggestions in the same textual form that
:func:`refine_dynamic_shapes_from_suggested_fixes` parses, so a failed export
can be turned into a working specification mechanically.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from .dynamic_shapes import Constraint, _linear_name

__all__ = ["DimConstraints"]


@dataclasses.dataclass
class _Bound:
    """Mutable interval [lo, hi]; ``None`` marks an unbounded side."""

    lo: int | None = None
    hi: int | None = None

    def intersect(self, lo: int | None, hi: int | None) -> bool:
        """Tighten in place; returns False when the interval becomes empty."""

        if lo is not None and (self.lo is None or lo > self.lo):
            self.lo = lo
        if hi is not None and (self.hi is None or hi < self.hi):
            self.hi = hi
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            return False
        return True


@dataclasses.dataclass
class _Relation:
    root: str
    scale: int
    offset: int

    def key(self) -> tuple[str, int, int]:
        return (self.root, self.scale, self.offset)

    def render(self) -> str:
        return _linear_name(self.root, self.scale, self.offset)


@dataclasses.dataclass
class _Violation:
    message: str
    name: str | None
    suggested: str | None


class DimConstraints:
    """Accumulates dimension constraints and reports inconsistent ones.

    Add :class:`Constraint` records (optionally paired with the concrete size
    observed on the example input), then call :meth:`solve`.  Failed solves
    render machine-readable repair lines via :meth:`pretty_print` and
    :meth:`suggested_fixes`.
    """

    def __init__(self) -> None:
        self._ranges: dict[str, _Bound] = {}
        self._relations: dict[str, _Relation] = {}
        self._sites: dict[str, list[tuple[Any, int]]] = {}
        self._observed: dict[str, int] = {}
        self._roots: dict[str, set[str]] = {}
        self._violations: list[_Violation] = []
        self._solved: bool | None = None

    # -- collection ---------------------------------------------------------

    def add(self, constraint: Constraint, observed_size: int | None = None) -> None:
        """Record one constraint, with the example size when available."""

        if not isinstance(constraint, Constraint):
            raise TypeError(f"expected a Constraint, got {type(constraint).__name__}")
        self._solved = None
        name = constraint.name
        if name is not None:
            self._sites.setdefault(name, []).append((constraint.source, constraint.dim))
        if observed_size is not None and name is not None:
            previous = self._observed.setdefault(name, int(observed_size))
            if previous != int(observed_size):
                # a shared name declares equality; disagreeing example sizes
                # cannot be repaired automatically, so suggest specializing
                # to the first observed size
                self._violations.append(
                    _Violation(
                        f"dimension {name!r} has conflicting example sizes "
                        f"{previous} and {int(observed_size)}",
                        name,
                        f"    {name} = {previous}",
                    )
                )
        if constraint.root is not None:
            relation = _Relation(constraint.root, constraint.scale, constraint.offset)
            known = self._relations.get(name)
            if known is not None and known.key() != relation.key():
                self._violations.append(
                    _Violation(
                        f"dimension {name!r} is defined both as {known.render()} "
                        f"and {relation.render()}",
                        name,
                        f"    {name} = {relation.render()}",
                    )
                )
            else:
                self._relations[name] = relation
                self._roots.setdefault(constraint.root, set()).add(name)
            if constraint.min is not None or constraint.max is not None:
                self._add_range(name, constraint.min, constraint.max)
            return
        if name is not None:
            self._add_range(name, constraint.min, constraint.max)
        elif constraint.min is not None and constraint.max is not None and constraint.min > constraint.max:
            self._violations.append(
                _Violation(
                    f"static range [{constraint.min}, {constraint.max}] is empty",
                    None,
                    None,
                )
            )

    def _add_range(self, name: str, lo: int | None, hi: int | None) -> None:
        bound = self._ranges.setdefault(name, _Bound())
        if not bound.intersect(lo, hi):
            self._violations.append(
                _Violation(
                    f"dimension {name!r} has an empty range after intersecting "
                    f"[{lo}, {hi}] with the recorded bounds",
                    name,
                    self._range_fix(name),
                )
            )

    # -- solving --------------------------------------------------------------

    def solve(self) -> bool:
        """Check all recorded constraints; True when they are consistent."""

        if self._solved is not None:
            return self._solved
        for name, relation in self._relations.items():
            if name in self._roots:
                self._violations.append(
                    _Violation(
                        f"dimension {name!r} is both a relation root and derived",
                        name,
                        None,
                    )
                )
        for name, bound in self._ranges.items():
            relation = self._relations.get(name)
            if relation is None:
                continue
            root_bound = self._ranges.get(relation.root)
            if root_bound is None:
                continue
            lo = hi = None
            if root_bound.lo is not None:
                lo = relation.scale * root_bound.lo + relation.offset
            if root_bound.hi is not None:
                hi = relation.scale * root_bound.hi + relation.offset
            if not bound.intersect(lo, hi):
                self._violations.append(
                    _Violation(
                        f"dimension {name!r} = {relation.render()} admits no size "
                        f"satisfying both bounds",
                        name,
                        f"    {name} = {relation.render()}",
                    )
                )
        self._solved = not self._violations
        return self._solved

    @property
    def violations(self) -> list[str]:
        """Human-readable problem descriptions accumulated so far."""

        return [item.message for item in self._violations]

    def suggested_fixes(self) -> list[str]:
        """Repair lines in the ``name = expression`` suggested-fix format."""

        fixes: list[str] = []
        for item in self._violations:
            if item.suggested is not None:
                fixes.append(item.suggested)
        return fixes

    def pretty_print(self, source_names: Mapping[int, str] | None = None) -> str:
        """Render the problem report, ending with a suggested-fixes block."""

        if self.solve():
            return "all dimension constraints are consistent"
        lines = list(self.violations)
        fixes = self.suggested_fixes()
        if fixes:
            lines.append("Suggested fixes:")
            lines.extend(fixes)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty_print()

    # -- helpers ---------------------------------------------------------------

    def _range_fix(self, name: str) -> str:
        bound = self._ranges.get(name)
        if bound is None:
            return f"    {name} = Dim({name!r})"
        parts = [f"Dim({name!r}"]
        if bound.lo is not None:
            parts.append(f"min={bound.lo}")
        if bound.hi is not None:
            parts.append(f"max={bound.hi}")
        return "    {} = {})".format(name, ", ".join(parts))


def attach_observed_sizes(
    constraints: Sequence[Constraint],
    combined_args: Any,
    dynamic_shapes: Any,
) -> list[tuple[Constraint, int | None]]:
    """Pair each constraint with the example size at its own site.

    Sizes are keyed by the ``(input object, dim index)`` site so that one
    symbolic name shared across inputs yields one observation per site —
    disagreeing observations are exactly the inconsistency the solver looks
    for.
    """

    from .dynamic_shapes import Dim as _NamedDim

    size_at_site: dict[tuple[int, int], int] = {}

    def walk(value: Any, spec: Any) -> None:
        shape = getattr(value, "shape", None)
        if shape is None or spec is None:
            return
        try:
            sizes = tuple(int(item) for item in shape)
        except TypeError:
            return
        entries = (
            list(spec.items())
            if isinstance(spec, dict)
            else list(enumerate(spec))
            if isinstance(spec, (tuple, list))
            else []
        )
        for index, dim in entries:
            if index < len(sizes) and isinstance(dim, _NamedDim):
                size_at_site[(id(value), index)] = sizes[index]

    def walk_tree(node: Any, spec_node: Any) -> None:
        if isinstance(node, dict) and isinstance(spec_node, dict):
            for key in node:
                walk_tree(node[key], spec_node.get(key))
        elif isinstance(node, (tuple, list)) and isinstance(spec_node, (tuple, list)):
            for item, item_spec in zip(node, spec_node):
                walk_tree(item, item_spec)
        else:
            walk(node, spec_node)

    walk_tree(combined_args, dynamic_shapes)
    return [
        (constraint, size_at_site.get((id(constraint.source), constraint.dim)))
        for constraint in constraints
    ]
