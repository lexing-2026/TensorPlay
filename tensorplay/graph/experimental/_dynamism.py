from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ...utils import _pytree

__all__ = [
    "clone_and_convert_to_meta",
    "module_to_nested_dict",
    "normalize_source_name",
    "track_dynamism_across_examples",
]


def transform_fn(value: Any) -> Any:
    if hasattr(value, "clone") and hasattr(value, "to") and hasattr(value, "shape"):
        clone = value.clone()
        try:
            return clone.to(device="meta")
        except (TypeError, RuntimeError) as exc:
            raise RuntimeError("meta conversion is unavailable for this tensor") from exc
    return value


def normalize_source_name(name: str) -> str:
    return re.sub(r"\.([a-zA-Z_][a-zA-Z0-9_]*)", r"['\1']", name)


def _iter_public_values(value: Any) -> Iterable[tuple[str, Any]]:
    for name in dir(value):
        if name.startswith("_"):
            continue
        try:
            item = getattr(value, name)
        except (AttributeError, NotImplementedError):
            continue
        if callable(item):
            continue
        yield name, item


def module_to_nested_dict(module: Any) -> dict[str, Any]:
    parameters: dict[str, Any] = {}
    modules: dict[str, Any] = {}
    result: dict[str, Any] = {"_parameters": parameters, "_modules": modules}

    tensor_type = None
    try:
        import tensorplay as tp

        tensor_type = tp.Tensor
    except (ImportError, AttributeError):
        pass

    for name, value in _iter_public_values(module):
        if isinstance(value, (int, float)) and type(value) is not bool:
            result[name] = value
        elif tensor_type is not None and isinstance(value, tensor_type):
            result[name] = value

    for method_name in ("named_parameters", "named_buffers"):
        method = getattr(module, method_name, None)
        if callable(method):
            for name, value in method(recurse=False):
                parameters[name] = value

    children = getattr(module, "named_children", None)
    if callable(children):
        for name, child in children():
            modules[name] = module_to_nested_dict(child)
    return result


def _flatten_with_paths(value: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], Any]]:
    if isinstance(value, dict):
        result: list[tuple[tuple[Any, ...], Any]] = []
        for key, item in value.items():
            result.extend(_flatten_with_paths(item, path + (key,)))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            result.extend(_flatten_with_paths(item, path + (index,)))
        return result
    return [(path, value)]


def _shape_of(value: Any) -> tuple[Any, ...] | None:
    if isinstance(value, (int, float)) and type(value) is not bool:
        return (value,)
    shape = getattr(value, "shape", None)
    if callable(shape):
        shape = shape()
    if shape is None:
        return None
    try:
        return tuple(shape)
    except TypeError:
        return None


def track_dynamism_across_examples(example_inputs: list[Any]) -> dict[Any, Any]:
    tracking: dict[tuple[Any, ...], list[set[Any]]] = {}
    for example in example_inputs:
        if isinstance(example, dict):
            candidate = example.get("self")
            if candidate is not None and hasattr(candidate, "named_children"):
                example = dict(example)
                example["self"] = module_to_nested_dict(candidate)
        for path, value in _flatten_with_paths(example):
            shape = _shape_of(value)
            if shape is None:
                continue
            dimensions = tracking.setdefault(path, [set() for _ in shape])
            while len(dimensions) < len(shape):
                dimensions.append(set())
            for index, dimension in enumerate(shape):
                dimensions[index].add(dimension)

    result: dict[Any, Any] = {}
    for path, dimensions in tracking.items():
        if not path:
            continue
        key = path[0]
        rendered = "L" + "".join(str(part) for part in path)
        result.setdefault(key, {})[rendered] = tuple(
            len(values) > 1 for values in dimensions
        )
    return result


def clone_and_convert_to_meta(example_input: Any) -> Any:
    return _pytree.tree_map(transform_fn, example_input)
