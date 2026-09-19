"""Functional form of an arbitrary mutating operator.

``auto_functionalized(_mutable_op, **kwargs)`` runs ``_mutable_op`` on
private copies of the arguments it mutates and returns
``(*outputs, *new_values)``: the operator's own outputs followed by the
final value of every mutated argument, in schema order.  A graph that calls
it instead of the mutating operator is free of mutation; the new values take
the place of the mutated arguments downstream.

Every argument travels as a keyword named after the schema, so the node
records the full call.  Outputs that alias a mutated argument are not
repeated among ``outputs``: the corresponding new value stands for them.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import Any

import tensorplay as tp

from ._hop_base import HigherOrderOperator

__all__ = [
    "auto_functionalized",
    "clone_preserve_strides",
    "get_mutable_args",
    "normalize_arguments",
    "returns_without_aliases",
]


def get_mutable_args(op: Any) -> tuple[list[str], list[Any]]:
    """Names and schema types of the arguments ``op`` writes to."""

    mutated = op._schema.mutated_arguments()
    return [a.name for a in mutated], [a.type for a in mutated]


def returns_without_aliases(op: Any) -> list[int]:
    """Indices of the returns that do not alias a mutated argument."""

    written = set()
    for argument in op._schema.mutated_arguments():
        written |= set(argument.alias_info.before_set)
    kept = []
    for index, ret in enumerate(op._schema.returns):
        alias = ret.alias_info
        if alias is not None and alias.is_write and alias.before_set & written:
            continue
        kept.append(index)
    return kept


def normalize_arguments(op: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """All of ``(args, kwargs)`` as keywords named by the schema."""

    normalized: dict[str, Any] = {}
    for index, argument in enumerate(op._schema.arguments):
        if argument.name in kwargs:
            normalized[argument.name] = kwargs[argument.name]
        elif index < len(args) and not argument.kwarg_only:
            normalized[argument.name] = args[index]
        else:
            normalized[argument.name] = argument.default_value
    return normalized


def _required_storage_length(shape: Any, strides: Any, storage_offset: int) -> int:
    if reduce(operator.mul, shape, 1) == 0:
        return 0
    max_offset = sum((size - 1) * stride for size, stride in zip(shape, strides))
    return 1 + storage_offset + max_offset


def clone_preserve_strides(x: Any) -> Any:
    """A copy of ``x`` with its exact sizes, strides and storage offset."""

    needed = _required_storage_length(tuple(x.shape), tuple(x.stride()), x.storage_offset())
    buffer = tp.as_strided(x, (needed,), (1,), 0).clone()
    return tp.as_strided(buffer, tuple(x.shape), tuple(x.stride()), x.storage_offset())


def _clone_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [clone_preserve_strides(item) for item in value]
    return clone_preserve_strides(value)


auto_functionalized = HigherOrderOperator("auto_functionalized")


@auto_functionalized.py_impl("CompositeExplicitAutograd")
def auto_functionalized_dense(_mutable_op: Any, **kwargs: Any) -> tuple[Any, ...]:
    new_kwargs = dict(kwargs)
    new_values = []
    names, _ = get_mutable_args(_mutable_op)
    for name in names:
        new_kwargs[name] = _clone_value(kwargs[name])
        new_values.append(new_kwargs[name])
    schema_args = _mutable_op._schema.arguments
    positional = [new_kwargs[a.name] for a in schema_args if not a.kwarg_only]
    keywords = {a.name: new_kwargs[a.name] for a in schema_args if a.kwarg_only}
    out = _mutable_op(*positional, **keywords)
    outs = out if isinstance(out, tuple) else (out,)
    if not _mutable_op._schema.returns:
        outs = ()
    kept = tuple(outs[i] for i in returns_without_aliases(_mutable_op) if i < len(outs))
    return (*kept, *new_values)
