"""Dynamic dimension declarations and input-shape utilities."""

from __future__ import annotations

import dataclasses
import inspect
import re
import sys
from collections.abc import Callable, Mapping
from enum import Enum, auto
from collections.abc import Iterator
from typing import Any

__all__ = [
    "AdditionalInputs",
    "Constraint",
    "ConstraintsExceededError",
    "Dim",
    "DerivedDim",
    "ShapesCollection",
    "dims",
    "refine_dynamic_shapes_from_suggested_fixes",
]


class ConstraintsExceededError(RuntimeError):
    """A runtime input violated the declared dynamic-shape contract.

    Raised by the assertions inserted into captured graphs and by export-time
    validation when an example input falls outside a declared range.  It is a
    ``RuntimeError`` so callers written against plain runtime failures keep
    working.
    """


class _DimHintType(Enum):
    AUTO = auto()
    STATIC = auto()
    DYNAMIC = auto()


@dataclasses.dataclass(frozen=True)
class _DimHint:
    type: _DimHintType
    min: int | None = None
    max: int | None = None
    _factory: bool = True

    @staticmethod
    def AUTO() -> "_DimHint":
        return _DimHint(_DimHintType.AUTO)

    @staticmethod
    def STATIC() -> "_DimHint":
        return _DimHint(_DimHintType.STATIC)

    @staticmethod
    def DYNAMIC() -> "_DimHint":
        return _DimHint(_DimHintType.DYNAMIC)

    def __call__(self, min: int | None = None, max: int | None = None) -> "_DimHint":
        if not self._factory:
            raise TypeError(f"{type(self).__name__!s} object is not callable")
        if min is not None and (type(min) is not int or min < 0):
            raise ValueError(f"min must be a non-negative integer, got {min!r}")
        if max is not None and (type(max) is not int or max < 0):
            raise ValueError(f"max must be a non-negative integer, got {max!r}")
        if min is not None and max is not None and min > max:
            raise ValueError(f"min must be no greater than max, got {min} and {max}")
        return _DimHint(self.type, min, max, False)

    def __repr__(self) -> str:
        values = [self.type.name]
        if self.min is not None:
            values.append(f"min={self.min}")
        if self.max is not None:
            values.append(f"max={self.max}")
        return f"DimHint({', '.join(values)})"


class Dim:
    """A named symbolic dimension with an optional finite range."""

    AUTO = _DimHint.AUTO()
    STATIC = _DimHint.STATIC()
    DYNAMIC = _DimHint.DYNAMIC()

    def __init__(
        self,
        name: str,
        *,
        min: int | None = None,
        max: int | None = None,
    ) -> None:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"dimension name must be an identifier, got {name!r}")
        lower = 0 if min is None else min
        upper = max
        if type(lower) is not int or lower < 0:
            raise ValueError(f"min must be a non-negative integer, got {min!r}")
        if upper is not None and (type(upper) is not int or upper < lower):
            raise ValueError(f"max must be an integer no less than min, got {max!r}")
        self.__name__ = name
        self.min = lower
        self.max = upper

    @property
    def name(self) -> str:
        return self.__name__

    def _derive(self, scale: int, offset: int) -> "_DerivedDim":
        return _DerivedDim(_linear_name(self.__name__, scale, offset), self, scale, offset)

    def __add__(self, other: Any) -> "_DerivedDim":
        if type(other) is not int:
            raise NotImplementedError("dimension addition requires an integer")
        return self._derive(1, other)

    def __radd__(self, other: Any) -> "_DerivedDim":
        return self + other

    def __sub__(self, other: Any) -> "_DerivedDim":
        if type(other) is not int:
            raise NotImplementedError("dimension subtraction requires an integer")
        return self._derive(1, -other)

    def __rsub__(self, other: Any) -> "_DerivedDim":
        raise NotImplementedError("a dimension cannot be negated")

    def __mul__(self, other: Any) -> "_DerivedDim":
        if type(other) is not int or other <= 0:
            raise NotImplementedError("dimension multiplication requires a positive integer")
        return self._derive(other, 0)

    def __rmul__(self, other: Any) -> "_DerivedDim":
        return self * other

    def __repr__(self) -> str:
        bounds = []
        if self.min != 0:
            bounds.append(f"min={self.min}")
        if self.max is not None:
            bounds.append(f"max={self.max}")
        suffix = f", {', '.join(bounds)}" if bounds else ""
        return f"Dim({self.__name__!r}{suffix})"


def _linear_name(root: str, scale: int, offset: int) -> str:
    """Render ``scale * root + offset`` in canonical expression form."""

    if scale == 1 and offset == 0:
        return root
    base = root if scale == 1 else f"{scale}*{root}"
    if offset == 0:
        return base
    sign = "+" if offset > 0 else "-"
    return f"{base} {sign} {abs(offset)}"


class _StaticDim(Dim):
    """Dimension pinned to one concrete size by an integer specification."""

    def __init__(self, value: int) -> None:
        if type(value) is not int or isinstance(value, bool) or value < 0:
            raise ValueError(f"static dimension requires a non-negative int, got {value!r}")
        self.__name__ = str(value)
        self.value = value

    @property
    def name(self) -> str:
        return self.__name__

    @property
    def min(self) -> int:
        return self.value

    @property
    def max(self) -> int:
        return self.value

    def __repr__(self) -> str:
        return self.__name__


class _DerivedDim(Dim):
    """A positive linear expression ``scale * root + offset`` over one base dimension."""

    def __init__(self, name: str, root: Dim, scale: int, offset: int) -> None:
        if isinstance(root, _DerivedDim):
            scale, offset, root = scale * root.scale, scale * root.offset + offset, root.root
        if scale <= 0:
            raise NotImplementedError("derived dimensions require a positive scale")
        self.__name__ = name
        self.root = root
        self.scale = scale
        self.offset = offset

    @property
    def name(self) -> str:
        return self.__name__

    def _evaluate(self, value: int) -> int:
        return self.scale * value + self.offset

    @property
    def min(self) -> int:
        value = self._evaluate(self.root.min)
        if value < 0:
            raise ValueError(
                f"derived dimension {self.__name__!r} has a negative lower bound; "
                f"specify a larger min for the root {self.root.__name__!r}"
            )
        return value

    @property
    def max(self) -> int | None:
        if self.root.max is None:
            return None
        value = self._evaluate(self.root.max)
        if value > sys.maxsize - 1:
            raise ValueError(f"derived dimension {self.__name__!r} exceeds the integer range")
        return value

    def _derive(self, scale: int, offset: int) -> "_DerivedDim":
        return _DerivedDim(
            _linear_name(self.root.__name__, self.scale * scale, self._compose_offset(offset)),
            self.root,
            self.scale * scale,
            scale * self.offset + offset,
        )

    def _compose_offset(self, offset: int) -> int:
        return self.offset + offset

    def __repr__(self) -> str:
        return self.__name__


DerivedDim = _DerivedDim
"""Public alias for a linear expression ``scale * root + offset`` over one
base dimension."""


def dims(*names: str, min: int | None = None, max: int | None = None) -> tuple[Dim, ...]:
    """Construct several named dimensions with shared bounds."""

    return tuple(Dim(name, min=min, max=max) for name in names)


@dataclasses.dataclass(frozen=True)
class Constraint:
    """A range restriction attached to one input dimension.

    ``name`` ties the constraint to a named :class:`Dim` shared across
    inputs (equalities are implied).  ``root``/``scale``/``offset`` describe a
    derived dimension whose size equals ``scale * root_size + offset``.
    Constraints without a ``name`` come from dim hints or static entries.
    """

    source: Any
    dim: int
    name: str | None = None
    min: int | None = None
    max: int | None = None
    warn_only: bool = False
    root: str | None = None
    scale: int = 1
    offset: int = 0

    def _replace_range(self, *, min: int | None = None, max: int | None = None) -> "Constraint":
        lower = self.min if min is None else min
        upper = self.max if max is None else max
        if upper is not None and lower is not None and lower > upper:
            raise ValueError("constraint range is empty")
        return dataclasses.replace(self, min=lower, max=upper)

    def __ge__(self, value: int) -> "Constraint":
        return self._replace_range(min=value)

    def __gt__(self, value: int) -> "Constraint":
        return self._replace_range(min=value + 1)

    def __le__(self, value: int) -> "Constraint":
        return self._replace_range(max=value)

    def __lt__(self, value: int) -> "Constraint":
        return self._replace_range(max=value - 1)

    def __bool__(self) -> bool:
        raise TypeError("a dimension constraint cannot be used as a boolean")

    @property
    def serializable_spec(self) -> dict[str, Any]:
        spec = {"source": self.source, "dim": self.dim, "min": self.min, "max": self.max}
        if self.name is not None:
            spec["name"] = self.name
        if self.root is not None:
            spec["root"] = self.root
            spec["scale"] = self.scale
            spec["offset"] = self.offset
        return spec


class _IntWrapper:
    def __init__(self, value: int) -> None:
        self.value = value


def _is_shape_value(value: Any) -> bool:
    return hasattr(value, "shape") or isinstance(value, _IntWrapper)


def _map_tree(value: Any, fn: Callable[[Any], Any]) -> Any:
    if isinstance(value, dict):
        return {key: _map_tree(item, fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_tree(item, fn) for item in value]
    if isinstance(value, tuple):
        return type(value)(_map_tree(item, fn) for item in value)
    return fn(value)


def _combine_args(model: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None) -> Any:
    callable_obj = getattr(model, "forward", model)
    signature = inspect.signature(callable_obj)
    bound = signature.bind_partial(*args, **dict(kwargs or {}))
    bound.apply_defaults()
    return dict(bound.arguments)


class ShapesCollection:
    """Associate shape specifications with tensor objects by identity."""

    def __init__(self) -> None:
        self._shapes: dict[int, Any] = {}

    def __setitem__(self, value: Any, shape: Any) -> None:
        if not _is_shape_value(value):
            raise TypeError(f"cannot assign a shape to {type(value).__name__}")
        key = id(value)
        previous = self._shapes.get(key, inspect.Parameter.empty)
        if previous is not inspect.Parameter.empty and previous != shape:
            raise ValueError(f"shape already assigned as {previous!r}")
        self._shapes[key] = shape

    def __getitem__(self, value: Any) -> Any:
        return self._shapes.setdefault(id(value), {})

    def __len__(self) -> int:
        return len(self._shapes)

    def dynamic_shapes(
        self, model: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None
    ) -> Any:
        seen: set[int] = set()

        def find(value: Any) -> Any:
            key = id(value)
            if key in self._shapes:
                seen.add(key)
                return self._shapes[key]
            return None

        result = _map_tree(_combine_args(model, args, kwargs), find)
        missing = set(self._shapes) - seen
        if missing:
            raise ValueError("some assigned shape values were not found in the inputs")
        return result


def _shape_snapshot(value: Any) -> Any:
    if hasattr(value, "shape"):
        try:
            return tuple(int(item) for item in value.shape)
        except Exception:
            return tuple(value.shape)
    if isinstance(value, dict):
        return {key: _shape_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_shape_snapshot(item) for item in value)
    return value


def _mark_dynamism(value: Any, *others: Any) -> Any:
    if others and any(type(value) is not type(other) for other in others):
        raise ValueError("additional inputs have incompatible value types")
    if isinstance(value, int) and not isinstance(value, bool):
        return None if all(value == other for other in others) else Dim.DYNAMIC
    if any(value != other for other in others):
        raise ValueError("additional inputs have incompatible static values")
    return None


class AdditionalInputs:
    """Infer dynamic shape markers from representative input sets."""

    def __init__(self) -> None:
        self._examples: list[tuple[tuple[Any, ...], dict[str, Any] | None]] = []

    def add(self, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None) -> None:
        if type(args) is not tuple:
            raise TypeError("representative args must be a tuple")
        if kwargs is not None and type(kwargs) is not dict:
            raise TypeError("representative kwargs must be a dict or None")
        self._examples.append((args, None if kwargs is None else dict(kwargs)))

    def dynamic_shapes(
        self, model: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any] | None = None
    ) -> Any:
        snapshots = [
            _shape_snapshot(_combine_args(model, current_args, current_kwargs))
            for current_args, current_kwargs in [(args, kwargs), *self._examples]
        ]
        if len(snapshots) == 1:
            return _map_tree(snapshots[0], lambda value: None)

        def merge(values: list[Any]) -> Any:
            first = values[0]
            if isinstance(first, dict):
                if not all(isinstance(item, dict) and item.keys() == first.keys() for item in values):
                    raise ValueError("additional inputs have incompatible mappings")
                return {key: merge([item[key] for item in values]) for key in first}
            if isinstance(first, (list, tuple)):
                if not all(type(item) is type(first) and len(item) == len(first) for item in values):
                    raise ValueError("additional inputs have incompatible sequences")
                result = [merge([item[index] for item in values]) for index in range(len(first))]
                return type(first)(result)
            return _mark_dynamism(first, *values[1:])

        return merge(snapshots)

    def verify(self, program: Any) -> None:
        module = program.module()
        for args, kwargs in self._examples:
            module(*args, **(kwargs or {}))


def _replace_dim(value: Any, replacements: Mapping[str, Any]) -> Any:
    if isinstance(value, Dim):
        return replacements.get(value.name, value)
    if isinstance(value, dict):
        return {key: _replace_dim(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_dim(item, replacements) for item in value]
    if isinstance(value, tuple):
        return tuple(_replace_dim(item, replacements) for item in value)
    return value


def _collect_named_dims(value: Any, found: dict[str, Dim] | None = None) -> dict[str, Dim]:
    found = {} if found is None else found
    if isinstance(value, _DerivedDim):
        found.setdefault(value.__name__, value)
        return _collect_named_dims(value.root, found)
    if isinstance(value, Dim):
        found.setdefault(value.__name__, value)
    elif isinstance(value, dict):
        for item in value.values():
            _collect_named_dims(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_named_dims(item, found)
    return found


def refine_dynamic_shapes_from_suggested_fixes(message: str, dynamic_shapes: Any) -> Any:
    """Apply suggested fixes (range refinements, specializations, relations).

    Supported fix forms::

        name = Dim('name', min=..., max=...)  # refine a range
        name = 4                              # specialize to a constant
        dy = dx + 1                           # tie a dim to another with a relation
        dy = 2*dx                             # positive multiple of another dim

    ``dx`` must name a dimension already present in ``dynamic_shapes`` or be
    defined by an earlier fix line.
    """

    if not isinstance(message, str):
        raise TypeError("message must be a string")
    section = message.split("Suggested fixes:", 1)
    if len(section) != 2:
        raise ValueError("suggested fixes were not found")

    known_dims = _collect_named_dims(dynamic_shapes)
    fixes: list[tuple[str, str]] = []
    for line in section[1].splitlines():
        line = line.split("#", 1)[0].strip()
        match = re.match(r"([A-Za-z_]\w*)\s*=\s*(.+)", line)
        if match:
            fixes.append((match.group(1), match.group(2).strip()))
    if not fixes:
        raise ValueError("no supported shape fixes were found")

    resolved: dict[str, Any] = {}

    def resolve(expression: str) -> Any:
        if expression in resolved:
            return resolved[expression]
        dim_match = re.match(
            r"Dim\(['\"]([A-Za-z_]\w*)['\"](?:,\s*min=(\d+))?(?:,\s*max=(\d+))?\)",
            expression,
        )
        if dim_match:
            value = Dim(
                dim_match.group(1),
                min=int(dim_match.group(2)) if dim_match.group(2) else None,
                max=int(dim_match.group(3)) if dim_match.group(3) else None,
            )
            resolved[expression] = value
            return value
        if re.fullmatch(r"\d+", expression):
            value = int(expression)
            resolved[expression] = value
            return value
        linear = re.match(
            r"(?:(\d+)\s*\*\s*)?([A-Za-z_]\w*)(?:\s*([+-])\s*(\d+))?",
            expression,
        )
        if linear and linear.group(0).strip() == expression.strip():
            scale = int(linear.group(1)) if linear.group(1) else 1
            offset = int(linear.group(4) or 0)
            if linear.group(3) == "-":
                offset = -offset
            root_name = linear.group(2)
            if root_name in resolved and isinstance(resolved[root_name], Dim):
                root = resolved[root_name]
            elif root_name in known_dims:
                root = known_dims[root_name]
            else:
                raise ValueError(
                    f"fix references unknown dimension {root_name!r}; it must appear "
                    f"in dynamic_shapes or in an earlier suggested fix"
                )
            if scale == 1 and offset == 0:
                value = root
            elif offset:
                value = root * scale + offset
            else:
                value = root * scale
            resolved[expression] = value
            return value
        raise ValueError(f"unsupported suggested fix expression: {expression!r}")

    replacements: dict[str, Any] = {}
    for name, expression in fixes:
        value = resolve(expression)
        resolved[name] = value
        if value is not None:
            replacements[name] = value
    return _replace_dim(dynamic_shapes, replacements)


def _iter_pairs(values: Any, specs: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[tuple[Any, ...], Any, Any]]:
    """Walk an input tree and a specification tree in lockstep."""

    if isinstance(values, dict) and isinstance(specs, dict):
        for key in values:
            yield from _iter_pairs(values[key], specs.get(key), path + (key,))
    elif isinstance(values, (tuple, list)) and isinstance(specs, (tuple, list)):
        if len(specs) != len(values):
            raise ValueError(
                f"dynamic shape specification at inputs[{_render_path(path)}] has "
                f"{len(specs)} entries but the input has {len(values)}"
            )
        for index, (item, spec) in enumerate(zip(values, specs)):
            yield from _iter_pairs(item, spec, path + (index,))
    else:
        yield path, values, specs


def _render_path(path: tuple[Any, ...]) -> str:
    rendered = ""
    for item in path:
        rendered = f"{rendered}[{item!r}]" if isinstance(item, str) else f"{rendered}[{item}]"
    return rendered or "[]"


def _tensor_size(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None or isinstance(value, _IntWrapper):
        return None
    try:
        return tuple(int(item) for item in shape)
    except TypeError:
        return None


def _suggest_fix(name: str, dim: Dim | None, size: int) -> str:
    if dim is None:
        return f"    {name} = {size}  # specialize to a constant"
    return f"    {name} = {dim!r}"


def _check_dynamic_shapes(
    combined_args: Any,
    dynamic_shapes: Any,
) -> None:
    """Validate a dynamic_shapes specification against the example inputs."""

    if not dynamic_shapes:
        return
    for path, value, spec in _iter_pairs(combined_args, dynamic_shapes):
        size = _tensor_size(value)
        if size is None:
            if spec is not None and not isinstance(spec, _DimHint):
                raise ValueError(
                    f"cannot associate shape {spec!r} at dynamic_shapes{_render_path(path)} "
                    f"with non-tensor input of type {type(value).__name__}"
                )
            continue
        if spec is None:
            continue
        entries = (
            list(spec.items()) if isinstance(spec, dict) else list(enumerate(spec))
        )
        for index, dim in entries:
            if dim is None:
                continue
            if not isinstance(index, int) or index < 0:
                raise TypeError(f"dimension index must be a non-negative int, got {index!r}")
            if index >= len(size):
                raise ValueError(
                    f"dimension index {index} at dynamic_shapes{_render_path(path)} is out "
                    f"of range for a {len(size)}-dimensional input"
                )
            if type(dim) is int and not isinstance(dim, bool):
                if size[index] != dim:
                    raise ValueError(
                        f"input size {size[index]} of dimension {index} at "
                        f"dynamic_shapes{_render_path(path)} does not match the expected "
                        f"static size {dim}"
                    )
                continue
            if isinstance(dim, _DimHint):
                if dim.min is not None and size[index] < dim.min:
                    raise ValueError(
                        f"input size {size[index]} of dimension {index} at "
                        f"dynamic_shapes{_render_path(path)} violates min={dim.min}"
                    )
                if dim.max is not None and size[index] > dim.max:
                    raise ValueError(
                        f"input size {size[index]} of dimension {index} at "
                        f"dynamic_shapes{_render_path(path)} violates max={dim.max}"
                    )
            elif isinstance(dim, _StaticDim):
                if size[index] != dim.value:
                    raise ValueError(
                        f"input size {size[index]} of dimension {index} at "
                        f"dynamic_shapes{_render_path(path)} does not match the expected "
                        f"static size {dim.value}"
                    )
            elif isinstance(dim, Dim):
                if size[index] < dim.min or (dim.max is not None and size[index] > dim.max):
                    lower = f"min={dim.min}" if dim.min else ""
                    upper = f"max={dim.max}" if dim.max is not None else ""
                    bound = ", ".join(item for item in (lower, upper) if item)
                    # widen to include the observed size: the repaired range
                    # covers the example input, keeping the dim dynamic
                    suggested = Dim(
                        dim.__name__,
                        min=min(dim.min, size[index]),
                        max=None if dim.max is None else max(dim.max, size[index]),
                    )
                    raise ValueError(
                        f"input size {size[index]} of dimension {index} at "
                        f"dynamic_shapes{_render_path(path)} violates {dim.__name__}"
                        f"({bound}). Suggested fixes:\n"
                        f"{_suggest_fix(dim.__name__, suggested, size[index])}"
                    )


def _process_dynamic_shapes(
    combined_args: Any,
    dynamic_shapes: Any,
) -> list[Constraint]:
    """Turn a validated dynamic_shapes specification into dimension constraints."""

    if not dynamic_shapes:
        return []
    _check_dynamic_shapes(combined_args, dynamic_shapes)
    constraints: list[Constraint] = []
    for path, value, spec in _iter_pairs(combined_args, dynamic_shapes):
        size = _tensor_size(value)
        if size is None or spec is None:
            continue
        entries = (
            list(spec.items()) if isinstance(spec, dict) else list(enumerate(spec))
        )
        for index, dim in entries:
            if dim is None:
                continue
            if isinstance(dim, _StaticDim):
                constraints.append(
                    Constraint(value, index, name=None, min=dim.value, max=dim.value)
                )
            elif isinstance(dim, _DerivedDim):
                constraints.append(
                    Constraint(
                        value,
                        index,
                        name=dim.__name__,
                        min=dim.min,
                        max=dim.max,
                        root=dim.root.__name__,
                        scale=dim.scale,
                        offset=dim.offset,
                    )
                )
            elif isinstance(dim, Dim):
                constraints.append(
                    Constraint(value, index, name=dim.__name__, min=dim.min, max=dim.max)
                )
            elif isinstance(dim, _DimHint):
                relaxed = dim.type is not _DimHintType.STATIC
                constraints.append(
                    Constraint(
                        value,
                        index,
                        name=None,
                        min=dim.min,
                        max=dim.max,
                        warn_only=relaxed,
                    )
                )
    return constraints


def _constraint_program(
    constraints: list[Constraint],
) -> tuple[list[Constraint], dict[str, dict[str, int | None]]]:
    """Group constraints into runtime assertions and per-name range bounds."""

    asserts: list[Constraint] = []
    ranges: dict[str, dict[str, int | None]] = {}
    anchors: dict[str, tuple[Any, int]] = {}
    for constraint in constraints:
        if constraint.root is not None:
            asserts.append(constraint)
            ranges.setdefault(
                constraint.name, {"min": constraint.min, "max": constraint.max}
            )
            continue
        if constraint.name is not None:
            if constraint.name in ranges:
                known = ranges[constraint.name]
                if known["min"] != constraint.min or known["max"] != constraint.max:
                    raise ValueError(
                        f"found conflicting definitions for symbolic dimension "
                        f"{constraint.name!r}: {known} and "
                        f"{{'min': {constraint.min}, 'max': {constraint.max}}}"
                    )
                if not constraint.warn_only:
                    asserts.append(constraint)  # equality with the anchor
            else:
                ranges[constraint.name] = {"min": constraint.min, "max": constraint.max}
                if not constraint.warn_only:
                    asserts.append(constraint)
        elif not constraint.warn_only:
            asserts.append(constraint)
    return asserts, ranges
