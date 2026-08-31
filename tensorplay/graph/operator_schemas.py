"""Signature inspection and argument normalization for graph call nodes."""

from __future__ import annotations

import inspect
import numbers
import types
import typing
import warnings
from typing import Any, Literal, NamedTuple, overload

from ._compatibility import compatibility


@compatibility(is_backward_compatible=False)
class ArgsKwargsPair(NamedTuple):
    """Container for normalized positional and keyword arguments."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]


_manual_overrides: dict[Any, list[inspect.Signature]] = {}


def _signature_list(target: Any) -> list[inspect.Signature] | None:
    override = _manual_overrides.get(target)
    if override is not None:
        return override
    try:
        return [inspect.signature(inspect.unwrap(target))]
    except (TypeError, ValueError):
        return None


@compatibility(is_backward_compatible=False)
def get_signature_for_operation(
    op: Any,
    return_schemas: bool = False,
) -> list[inspect.Signature] | tuple[list[inspect.Signature] | None, None] | None:
    """Return inspectable call signatures for an operation."""

    signatures = _signature_list(op)
    if return_schemas:
        return signatures, None
    return signatures


@compatibility(is_backward_compatible=False)
def check_for_mutable_operation(
    target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    """Reject common in-place and explicit-output call forms."""

    del args
    name = getattr(target, "__name__", "")
    if isinstance(name, str) and name.endswith("_"):
        raise RuntimeError(f"operation {name!r} mutates an input")
    if "out" in kwargs and kwargs["out"] is not None:
        raise RuntimeError("operations with an explicit output are not functional")


@compatibility(is_backward_compatible=False)
def create_type_hint(value: object) -> object:
    """Construct a useful homogeneous container annotation when possible."""

    if not isinstance(value, (list, tuple)):
        return value
    origin = list if isinstance(value, list) else tuple
    if not value:
        return origin[Any] if origin is list else tuple[Any, ...]
    types_seen = [item if isinstance(item, type) else type(item) for item in value]
    common = types_seen[0]
    for candidate in types_seen[1:]:
        if issubclass(candidate, common):
            continue
        if issubclass(common, candidate):
            common = candidate
            continue
        warnings.warn(f"could not infer a common type for {value!r}", stacklevel=2)
        common = Any
        break
    return origin[common] if origin is list else tuple[common, ...]


def _type_origin(value: Any) -> Any:
    return typing.get_origin(value) or getattr(value, "__origin__", None)


@compatibility(is_backward_compatible=False)
def type_matches(signature_type: Any, argument_type: Any) -> bool:
    """Return whether an argument annotation accepts a concrete type."""

    if signature_type in (Any, object) or signature_type is argument_type:
        return True
    if argument_type is Any:
        return False
    origin = _type_origin(signature_type)
    if origin in (typing.Union, types.UnionType):
        return any(type_matches(item, argument_type) for item in typing.get_args(signature_type))
    if origin is list:
        element = typing.get_args(signature_type)[0] if typing.get_args(signature_type) else Any
        arg_origin = _type_origin(argument_type)
        if arg_origin is list:
            args = typing.get_args(argument_type)
            return not args or type_matches(element, args[0])
        return argument_type is list and element is Any
    if origin is tuple:
        expected = typing.get_args(signature_type)
        actual = typing.get_args(argument_type)
        if not expected:
            return argument_type in (tuple, typing.Tuple)
        if len(expected) == 2 and expected[1] is Ellipsis:
            return all(type_matches(expected[0], item) for item in actual)
        return len(expected) == len(actual) and all(
            type_matches(left, right) for left, right in zip(expected, actual)
        )
    if signature_type is numbers.Number and argument_type in (int, float, complex):
        return True
    if inspect.isclass(signature_type) and inspect.isclass(argument_type):
        return issubclass(argument_type, signature_type)
    return False


def _args_kwargs_to_normalized_args_kwargs(
    signature: inspect.Signature,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    normalize_to_only_use_kwargs: bool,
) -> ArgsKwargsPair | None:
    parameters = list(signature.parameters.values())
    if any(
        parameter.kind
        in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for parameter in parameters
    ):
        return None
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        return None
    bound.apply_defaults()
    positional: list[Any] = []
    normalized_kwargs: dict[str, Any] = {}
    for index, parameter in enumerate(parameters):
        value = bound.arguments[parameter.name]
        if (
            not normalize_to_only_use_kwargs
            and index < len(args)
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ):
            positional.append(value)
        elif parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            normalized_kwargs[parameter.name] = value
    return ArgsKwargsPair(tuple(positional), normalized_kwargs)


@compatibility(is_backward_compatible=False)
def normalize_function(
    target: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    arg_types: tuple[Any, ...] | None = None,
    kwarg_types: dict[str, Any] | None = None,
    normalize_to_only_use_kwargs: bool = False,
) -> ArgsKwargsPair | None:
    """Bind a function call and materialize omitted defaults."""

    del arg_types, kwarg_types
    if kwargs is None:
        kwargs = {}
    signature = _signature_list(target)
    if not signature:
        return None
    normalized = [
        item
        for item in (
            _args_kwargs_to_normalized_args_kwargs(
                candidate, args, kwargs, normalize_to_only_use_kwargs
            )
            for candidate in signature
        )
        if item is not None
    ]
    if len(normalized) == 1:
        return normalized[0]
    if len(normalized) > 1:
        raise RuntimeError(f"call signature for {target!r} is ambiguous")
    return None


@compatibility(is_backward_compatible=False)
def _normalize_function_or_error(
    target: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    arg_types: tuple[Any, ...] | None = None,
    kwarg_types: dict[str, Any] | None = None,
    normalize_to_only_use_kwargs: bool = False,
) -> ArgsKwargsPair:
    result = normalize_function(
        target,
        args,
        kwargs,
        arg_types,
        kwarg_types,
        normalize_to_only_use_kwargs,
    )
    if result is None:
        raise RuntimeError(f"failed to normalize call to {target!r}")
    return result


@compatibility(is_backward_compatible=False)
def normalize_module(
    root: Any,
    target: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    normalize_to_only_use_kwargs: bool = False,
) -> ArgsKwargsPair | None:
    """Normalize a call to a named child module."""

    module = root
    try:
        for atom in target.split("."):
            module = getattr(module, atom)
    except AttributeError as exc:
        raise RuntimeError(f"module target {target!r} does not exist") from exc
    forward = getattr(module, "forward", module)
    return normalize_function(
        forward, args, kwargs, normalize_to_only_use_kwargs=normalize_to_only_use_kwargs
    )


__all__ = [
    "ArgsKwargsPair",
    "check_for_mutable_operation",
    "create_type_hint",
    "get_signature_for_operation",
    "normalize_function",
    "normalize_module",
    "type_matches",
]
