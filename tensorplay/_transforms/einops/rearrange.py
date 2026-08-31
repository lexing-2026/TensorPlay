"""Rearranging a tensor by naming its axes.

``rearrange(x, "b h w c -> b c h w")`` says what the axes *are*, so the reader
never has to decode a permutation of integers.  Parentheses group axes:
``"(b h) w -> b h w"`` splits an axis whose factors are given as keyword
arguments, and ``"b h w -> b (h w)"`` merges two.  ``...`` stands for any
number of leading or interior axes that pass straight through.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, Union

__all__ = ["rearrange"]

_ELLIPSIS = "…"


class _AnonymousAxis:
    """A literal axis length written in the pattern, such as the ``1`` in
    ``"b h w -> b h w 1"``.  Distinct occurrences stay distinct."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = int(value)
        if self.value < 1:
            raise ValueError(
                f"Anonymous axis should be a positive integer, not {self.value}"
            )

    def __repr__(self) -> str:
        return f"{self.value}-axis"


def _is_valid_identifier(name: str) -> bool:
    if name == _ELLIPSIS:
        return True
    if not str.isidentifier(name):
        return False
    if name[0] == "_" or name[-1] == "_":
        return False
    return True


class _ParsedExpression:
    """One side of a pattern, as a list of axis groups."""

    def __init__(self, expression: str, *, allow_underscore: bool = False) -> None:
        self.has_ellipsis = False
        self.has_ellipsis_parenthesized = False
        self.identifiers: set[Any] = set()
        self.composition: list[Union[list[Any], str]] = []

        if "." in expression:
            if "..." not in expression:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...)"
                )
            if str.count(expression, "...") != 1 or str.count(expression, ".") != 3:
                raise ValueError(
                    "Expression may contain dots only inside ellipsis (...); only "
                    "one ellipsis for tensor is allowed"
                )
            expression = expression.replace("...", _ELLIPSIS)
            self.has_ellipsis = True

        bracket_group: list[Any] | None = None

        def add_axis_name(name: str) -> None:
            if name in self.identifiers:
                if not (allow_underscore and name == "_"):
                    raise ValueError(f"Indexing expression contains duplicate axis {name}")
            if name == _ELLIPSIS:
                self.identifiers.add(_ELLIPSIS)
                if bracket_group is None:
                    self.composition.append(_ELLIPSIS)
                else:
                    bracket_group.append(_ELLIPSIS)
                    self.has_ellipsis_parenthesized = True
                return
            is_number = str.isdecimal(name)
            if is_number and int(name) == 1:
                # A literal 1 adds nothing to a group and is a bare axis alone.
                if bracket_group is None:
                    self.composition.append([])
                return
            axis_name: Any = _AnonymousAxis(name) if is_number else name
            if not (is_number or _is_valid_identifier(name)):
                raise ValueError(f"Invalid axis identifier: {name}")
            self.identifiers.add(axis_name)
            if bracket_group is None:
                self.composition.append([axis_name])
            else:
                bracket_group.append(axis_name)

        current_identifier = None
        for char in expression:
            if char in "() ":
                if current_identifier is not None:
                    add_axis_name(current_identifier)
                current_identifier = None
                if char == "(":
                    if bracket_group is not None:
                        raise ValueError("Axis composition is one-level (brackets are not allowed inside brackets)")
                    bracket_group = []
                elif char == ")":
                    if bracket_group is None:
                        raise ValueError("Brackets are not balanced")
                    self.composition.append(bracket_group)
                    bracket_group = None
            elif str.isalnum(char) or char in ["_", _ELLIPSIS]:
                if current_identifier is None:
                    current_identifier = char
                else:
                    current_identifier += char
            else:
                raise ValueError(f"Unknown character '{char}'")

        if bracket_group is not None:
            raise ValueError(f"Imbalanced parentheses in expression: {expression}")
        if current_identifier is not None:
            add_axis_name(current_identifier)


def _report(pattern: str, message: str) -> ValueError:
    return ValueError(f"{message}\n Expression: '{pattern}'")


@functools.lru_cache(256)
def _parse_pattern(pattern: str) -> tuple[_ParsedExpression, _ParsedExpression]:
    if "->" not in pattern:
        raise ValueError(f"Pattern must contain '->'\n Expression: '{pattern}'")
    left_str, right_str = pattern.split("->")
    left = _ParsedExpression(left_str)
    right = _ParsedExpression(right_str)
    if not left.has_ellipsis and right.has_ellipsis:
        raise _report(pattern, f"Ellipsis found in right side, but not left side of a pattern")
    if left.has_ellipsis and left.has_ellipsis_parenthesized:
        raise _report(pattern, f"Ellipsis is parenthesis in the left side is not allowed")
    return left, right


def rearrange(
    tensor: Any,
    pattern: str,
    **axes_lengths: int,
) -> Any:
    """Reshapes and permutes ``tensor`` according to ``pattern``.

    Args:
        tensor: the tensor to rearrange, or a sequence of tensors, which is
            stacked along a new leading axis first.
        pattern (str): ``"<input axes> -> <output axes>"``.  Names bind by
            position on the left and select by name on the right.  Parentheses
            group axes; ``...`` matches any number of axes.
        **axes_lengths: sizes for axes the pattern splits an input axis into,
            which cannot be inferred from the shape alone.

    Example:

        >>> x = tensorplay.randn(2, 3, 4)
        >>> rearrange(x, "b h w -> b w h").shape
        tensorplay.Size(2, 4, 3)
        >>> rearrange(x, "b h w -> b (h w)").shape
        tensorplay.Size(2, 12)
        >>> rearrange(x, "(b1 b2) h w -> b1 b2 h w", b1=1).shape
        tensorplay.Size(1, 2, 3, 4)
    """
    if not isinstance(tensor, Sequence) or hasattr(tensor, "shape"):
        working = tensor
    else:
        import tensorplay

        working = tensorplay.stack(list(tensor))

    left, right = _parse_pattern(pattern)

    # -- bind every left-hand name to a concrete length ---------------------
    if left.has_ellipsis:
        n_named = sum(1 for group in left.composition if group is not _ELLIPSIS)
        n_ellipsis = working.dim() - n_named
        if n_ellipsis < 0:
            raise _report(
                pattern,
                f"Wrong shape: expected at least {n_named} dims. Received "
                f"{working.dim()}-dim tensor.",
            )
    else:
        n_ellipsis = 0
        if len(left.composition) != working.dim():
            raise _report(
                pattern,
                f"Wrong shape: expected {len(left.composition)} dims. Received "
                f"{working.dim()}-dim tensor.",
            )

    shape = list(working.shape)
    ellipsis_names: list[str] = []
    known: dict[Any, int] = {}
    for name, length in axes_lengths.items():
        known[name] = int(length)

    # Names generated for the ellipsis axes cannot clash with pattern names.
    ellipsis_names = [f"_ellipsis_{i}" for i in range(n_ellipsis)]

    decomposed_input: list[list[Any]] = []
    axis_pos = 0
    for group in left.composition:
        if group is _ELLIPSIS:
            for name in ellipsis_names:
                known[name] = shape[axis_pos]
                decomposed_input.append([name])
                axis_pos += 1
            continue
        size = shape[axis_pos]
        axis_pos += 1
        unknown = [
            axis
            for axis in group
            if not isinstance(axis, _AnonymousAxis) and axis not in known
        ]
        product = 1
        for axis in group:
            if isinstance(axis, _AnonymousAxis):
                product *= axis.value
            elif axis in known:
                product *= known[axis]
        if len(unknown) > 1:
            raise _report(
                pattern,
                f"Could not infer sizes for {unknown}: pass them as keyword arguments.",
            )
        if len(unknown) == 1:
            if product == 0 or size % product != 0:
                raise _report(
                    pattern,
                    f"Shape mismatch: axis of length {size} is not divisible by "
                    f"the known factors {product}.",
                )
            known[unknown[0]] = size // product
            product = size
        if product != size:
            raise _report(
                pattern,
                f"Shape mismatch: axis of length {size} does not match the "
                f"specified length {product}.",
            )
        decomposed_input.append(list(group))

    # -- split the input into one dimension per named axis ------------------
    flat_input_axes = [axis for group in decomposed_input for axis in group]
    split_shape = [
        axis.value if isinstance(axis, _AnonymousAxis) else known[axis]
        for axis in flat_input_axes
    ]
    if split_shape != list(working.shape):
        working = working.reshape(split_shape)

    # -- permute into the output order --------------------------------------
    right_groups: list[list[Any]] = []
    for group in right.composition:
        if group is _ELLIPSIS:
            right_groups.extend([name] for name in ellipsis_names)
        else:
            right_groups.append(list(group))
    flat_output_axes = [axis for group in right_groups for axis in group]

    input_index = {}
    for position, axis in enumerate(flat_input_axes):
        input_index[id(axis) if isinstance(axis, _AnonymousAxis) else axis] = position

    permutation = []
    for axis in flat_output_axes:
        key = id(axis) if isinstance(axis, _AnonymousAxis) else axis
        if key not in input_index:
            raise _report(
                pattern, f"Identifier '{axis}' on the right side is not present on the left."
            )
        permutation.append(input_index[key])
    if len(set(permutation)) != len(permutation):
        raise _report(pattern, "Identifiers on the right side are not unique.")
    if len(permutation) != len(flat_input_axes):
        missing = [
            axis
            for axis in flat_input_axes
            if (id(axis) if isinstance(axis, _AnonymousAxis) else axis)
            not in {
                (id(a) if isinstance(a, _AnonymousAxis) else a) for a in flat_output_axes
            }
        ]
        raise _report(
            pattern,
            f"Identifiers {missing} are present on the left side but not the "
            "right; rearrange does not reduce axes.",
        )
    if permutation != list(range(len(permutation))):
        working = working.permute(tuple(permutation))

    # -- merge the output groups -------------------------------------------
    final_shape = []
    for group in right_groups:
        size = 1
        for axis in group:
            size *= axis.value if isinstance(axis, _AnonymousAxis) else known[axis]
        final_shape.append(size)
    if final_shape != list(working.shape):
        working = working.reshape(final_shape)
    return working
