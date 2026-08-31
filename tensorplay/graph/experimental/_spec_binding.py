from __future__ import annotations

import inspect
from itertools import zip_longest
from typing import Any, TYPE_CHECKING

import tensorplay as tp

from ...utils import _pytree
from .dynamic_spec import (
    DictSpec,
    IntermediateSpec,
    IntVar,
    LeafSpec,
    ObjectSpec,
    SeqSpec,
    ShapesSpec,
    TensorSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["_bind_spec_to_args", "_walk_spec"]


def _walk_spec(
    user_spec: IntermediateSpec | None,
    arg_value: Any,
    where: str,
) -> list[LeafSpec]:
    if user_spec is None:
        return [None] * len(_pytree.tree_leaves(arg_value))
    if isinstance(user_spec, SeqSpec):
        if not isinstance(arg_value, (list, tuple)):
            raise ValueError(f"{where}: sequence specification requires a sequence")
        if len(user_spec) > len(arg_value):
            raise ValueError(f"{where}: specification contains unused positions")
        result: list[LeafSpec] = []
        for index, (value, spec) in enumerate(
            zip_longest(arg_value, list(user_spec), fillvalue=None)
        ):
            result.extend(_walk_spec(spec, value, f"{where}[{index}]"))
        return result
    if isinstance(user_spec, DictSpec):
        if not isinstance(arg_value, dict):
            raise ValueError(f"{where}: mapping specification requires a dictionary")
        missing = set(user_spec) - set(arg_value)
        if missing:
            raise ValueError(f"{where}: unknown mapping entries {sorted(missing, key=repr)!r}")
        result = []
        for key, value in arg_value.items():
            result.extend(
                _walk_spec(
                    user_spec._entries.get(key), value, f"{where}[{key!r}]"
                )
            )
        return result
    if isinstance(user_spec, ObjectSpec):
        result = []
        for name, spec in user_spec.items():
            if not hasattr(arg_value, name):
                raise ValueError(f"{where}: object has no attribute {name!r}")
            result.extend(_walk_spec(spec, getattr(arg_value, name), f"{where}.{name}"))
        if result:
            return result
        return [None]
    if isinstance(user_spec, TensorSpec):
        if not isinstance(arg_value, tp.Tensor):
            raise ValueError(f"{where}: tensor specification requires a tensor")
        shape = getattr(arg_value, "shape", ())
        if callable(shape):
            shape = shape()
        if len(user_spec) != len(tuple(shape)):
            raise ValueError(f"{where}: tensor rank does not match the specification")
        return [user_spec]
    if isinstance(user_spec, (IntVar, int)):
        if not isinstance(arg_value, int) or isinstance(arg_value, bool):
            raise ValueError(f"{where}: integer specification requires an integer")
        return [user_spec]
    raise TypeError(f"{where}: unsupported specification {type(user_spec).__name__}")


def _bind_spec_to_args(
    function: Callable[..., Any],
    args: Any,
    kwargs: dict[str, Any] | None,
    shapes_spec: ShapesSpec,
) -> tuple[list[LeafSpec], list[Any], Any]:
    kwargs = dict(kwargs or {})
    flat_args, input_spec = _pytree.tree_flatten((args, kwargs))
    params = shapes_spec._params
    if params is None:
        return [None] * input_spec.num_leaves, flat_args, input_spec

    callable_object = getattr(function, "forward", function)
    parameters = list(inspect.signature(callable_object).parameters.values())
    varargs_index = next(
        (
            index
            for index, parameter in enumerate(parameters)
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL
        ),
        len(parameters),
    )
    result: list[LeafSpec] = []
    matched_named: set[str] = set()
    matched_varkw: set[str] = set()
    named = params._named_args
    varargs = list(params._varargs or [])
    for index, value in enumerate(args[:varargs_index]):
        name = parameters[index].name
        spec = named.get(name)
        if name in named:
            matched_named.add(name)
        result.extend(_walk_spec(spec, value, f"spec[{name!r}]"))
    for index, value in enumerate(args[varargs_index:]):
        spec = varargs[index] if index < len(varargs) else None
        result.extend(_walk_spec(spec, value, f"spec['*args'][{index}]"))
    for name, value in kwargs.items():
        if name in named:
            spec = named[name]
            matched_named.add(name)
        elif params._varkw is not None and name in params._varkw:
            spec = params._varkw[name]
            matched_varkw.add(name)
        else:
            spec = None
        result.extend(_walk_spec(spec, value, f"spec[{name!r}]"))

    unmatched = set(named) - matched_named
    if params._varkw is not None:
        unmatched |= set(params._varkw) - matched_varkw
    if unmatched:
        raise ValueError(f"spec entries do not match supplied arguments: {sorted(unmatched)!r}")
    if len(result) != input_spec.num_leaves:
        raise AssertionError(
            f"spec traversal produced {len(result)} leaves, expected {input_spec.num_leaves}"
        )
    return result, flat_args, input_spec
