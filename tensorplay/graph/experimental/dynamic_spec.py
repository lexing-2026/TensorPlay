from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Iterator, Sequence
from typing import Any, TypeAlias

from .sym_node import SymBool, SymInt, SymNode

__all__ = [
    "STATIC",
    "DictSpec",
    "IntVar",
    "LeafIntSpec",
    "LeafSpec",
    "IntermediateSpec",
    "ObjectSpec",
    "ParamsSpec",
    "ParamsSpecValue",
    "SeqSpec",
    "ShapeVar",
    "ShapesSpec",
    "TensorSpec",
    "dynamic_spec",
]

STATIC = None
_INDENT = "  "
_DYNAMIC_SPEC_ATTR = "_dynamic_spec"
_SPEC_SHAPE_ENV: object | None = None
_SPEC_SHAPE_ENV_LOCK = threading.Lock()


def _get_spec_shape_env() -> object:
    global _SPEC_SHAPE_ENV
    if _SPEC_SHAPE_ENV is None:
        with _SPEC_SHAPE_ENV_LOCK:
            if _SPEC_SHAPE_ENV is None:
                _SPEC_SHAPE_ENV = object()
    return _SPEC_SHAPE_ENV


class IntVar(SymInt):
    """A symbolic integer variable used by a dynamic specification."""

    _uid_counter = itertools.count()

    def __init__(
        self,
        name: str | None = None,
        *,
        min: int | None = None,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        self.name = name or "anon"
        self._uid = next(type(self)._uid_counter)
        self.min = min
        self.max = max
        self.optimization_hint = optimization_hint
        hint = optimization_hint
        super().__init__(
            f"{self.name}#{self._uid}",
            _get_spec_shape_env(),
            int,
            hint,
            lambda values: values.get(f"{self.name}#{self._uid}", hint),
        )

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        parts = [f"{self.name}#{self._uid}"]
        if self.min is not None:
            parts.append(f"min={self.min}")
        if self.max is not None:
            parts.append(f"max={self.max}")
        if self.optimization_hint is not None:
            parts.append(f"optimization_hint={self.optimization_hint}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": type(self).__name__,
            "name": self.name,
            "min": self.min,
            "max": self.max,
            "optimization_hint": self.optimization_hint,
        }


class ShapeVar(IntVar):
    """A non-negative symbolic dimension variable."""

    def __init__(
        self,
        name: str | None = None,
        *,
        min: int = 0,
        max: int | None = None,
        optimization_hint: int | None = None,
    ) -> None:
        if min < 0:
            raise ValueError("ShapeVar requires a non-negative lower bound")
        super().__init__(name, min=min, max=max, optimization_hint=optimization_hint)


def _validate_spec_sym(value: Any, *, where: str) -> None:
    if isinstance(value, SymNode) and value.shape_env is not _get_spec_shape_env():
        raise TypeError(f"{where}: symbolic values must originate from a spec variable")


def _validate_spec_tree(value: Any, *, where: str) -> None:
    _validate_spec_sym(value, where=where)
    if isinstance(value, TensorSpec):
        for index, item in enumerate(value):
            _validate_spec_tree(item, where=f"{where}.dims[{index}]")
    elif isinstance(value, ObjectSpec):
        for name, item in value.items():
            _validate_spec_tree(item, where=f"{where}.{name}")
    elif isinstance(value, DictSpec):
        for key, item in value.items():
            _validate_spec_tree(item, where=f"{where}[{key!r}]")
    elif isinstance(value, SeqSpec):
        for index, item in enumerate(value):
            _validate_spec_tree(item, where=f"{where}[{index}]")
    elif isinstance(value, ParamsSpec):
        for name, item in value.named_args.items():
            _validate_spec_tree(item, where=f"{where}[{name!r}]")
        for index, item in enumerate(value.varargs or []):
            _validate_spec_tree(item, where=f"{where}['*args'][{index}]")
        for name, item in (value.varkw or {}).items():
            _validate_spec_tree(item, where=f"{where}['**kwargs'][{name!r}]")


def _jsonable(value: Any) -> Any:
    return value.to_jsonable() if hasattr(value, "to_jsonable") else value


class TensorSpec:
    """Per-dimension specification for one tensor value."""

    def __init__(self, dims: Sequence[Any]) -> None:
        self._specs = list(dims)
        for index, value in enumerate(self._specs):
            if value is not None and not isinstance(value, (int, SymNode)):
                raise TypeError(
                    f"TensorSpec dim {index} must be an integer specification"
                )
            _validate_spec_sym(value, where=f"TensorSpec dim {index}")

    def __getitem__(self, index: int) -> Any:
        return self._specs[index]

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._specs)

    def __repr__(self) -> str:
        lines = ["Tensor:"]
        lines.extend(f"{_INDENT}{index}: {value!r}" for index, value in enumerate(self._specs))
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "TensorSpec", "dims": [_jsonable(value) for value in self._specs]}


class ObjectSpec:
    """Specification for selected attributes of a Python object."""

    def __init__(self, fields: dict[str, Any] | None = None) -> None:
        self._fields = dict(fields or {})
        for name, value in self._fields.items():
            _validate_spec_tree(value, where=f"ObjectSpec.{name}")

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def items(self):
        return self._fields.items()

    def __repr__(self) -> str:
        lines = ["object_spec:"]
        for name, value in self._fields.items():
            rendered = repr(value)
            if "\n" in rendered:
                lines.append(f"{_INDENT}.{name}:")
                lines.extend(_INDENT * 2 + line for line in rendered.splitlines())
            else:
                lines.append(f"{_INDENT}.{name}: {rendered}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "ObjectSpec", "fields": {k: _jsonable(v) for k, v in self._fields.items()}}


class DictSpec:
    """Specification for selected entries of a mapping value."""

    def __init__(self, entries: dict[str | int, Any] | None = None) -> None:
        self._entries = dict(entries or {})
        if any(not isinstance(key, (str, int)) for key in self._entries):
            raise TypeError("DictSpec keys must be strings or integers")
        for key, value in self._entries.items():
            _validate_spec_tree(value, where=f"DictSpec[{key!r}]")

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self):
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def items(self):
        return self._entries.items()

    def __repr__(self) -> str:
        lines = ["dict_spec:"]
        for key, value in self._entries.items():
            rendered = repr(value)
            lines.append(f"{_INDENT}[{key!r}]: {rendered}")
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "DictSpec", "entries": {str(k): _jsonable(v) for k, v in self._entries.items()}}


class SeqSpec:
    """Per-position specification for list and tuple values."""

    def __init__(self, entries: Sequence[Any]) -> None:
        self._entries = list(entries)
        for index, value in enumerate(self._entries):
            _validate_spec_tree(value, where=f"SeqSpec[{index}]")

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def __getitem__(self, index: int) -> Any:
        return self._entries[index]

    def __repr__(self) -> str:
        lines = ["seq_spec:"]
        lines.extend(f"{_INDENT}[{index}]: {value!r}" for index, value in enumerate(self._entries))
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "SeqSpec", "entries": [_jsonable(value) for value in self._entries]}


LeafIntSpec: TypeAlias = IntVar | SymInt | int | None
LeafSpec: TypeAlias = LeafIntSpec | TensorSpec
IntermediateSpec: TypeAlias = LeafSpec | ObjectSpec | DictSpec | SeqSpec
ParamsSpecValue: TypeAlias = IntermediateSpec | list[IntermediateSpec] | dict[str, IntermediateSpec]


class ParamsSpec:
    """Specification for named, variadic, and keyword arguments."""

    _VARARGS_KEY = "*args"
    _VARKW_KEY = "**kwargs"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self._named_args: dict[str, Any] = {}
        self._varargs: list[Any] | None = None
        self._varkw: dict[str, Any] | None = None
        for key, value in (params or {}).items():
            if key == self._VARARGS_KEY:
                if not isinstance(value, list):
                    raise ValueError("ParamsSpec '*args' must be a list")
                self._varargs = list(value)
            elif key == self._VARKW_KEY:
                if not isinstance(value, dict):
                    raise ValueError("ParamsSpec '**kwargs' must be a dictionary")
                self._varkw = dict(value)
            elif key.startswith("*"):
                raise ValueError(f"unknown variadic specification key {key!r}")
            else:
                self._named_args[key] = value
            _validate_spec_tree(value, where=f"ParamsSpec[{key!r}]")

    @property
    def named_args(self) -> dict[str, Any]:
        return dict(self._named_args)

    @property
    def varargs(self) -> list[Any] | None:
        return None if self._varargs is None else list(self._varargs)

    @property
    def varkw(self) -> dict[str, Any] | None:
        return None if self._varkw is None else dict(self._varkw)

    def __repr__(self) -> str:
        entries = [f"{key}: {value!r}" for key, value in self._named_args.items()]
        if self._varargs is not None:
            entries.append(f"{self._VARARGS_KEY}: {self._varargs!r}")
        if self._varkw is not None:
            entries.append(f"{self._VARKW_KEY}: {self._varkw!r}")
        return "\n".join(entries)

    def to_jsonable(self) -> dict[str, Any]:
        params = {key: _jsonable(value) for key, value in self._named_args.items()}
        if self._varargs is not None:
            params[self._VARARGS_KEY] = [_jsonable(value) for value in self._varargs]
        if self._varkw is not None:
            params[self._VARKW_KEY] = {key: _jsonable(value) for key, value in self._varkw.items()}
        return {"type": "ParamsSpec", "params": params}


class ShapesSpec:
    """Top-level dynamic specification with optional scalar assumptions."""

    def __init__(
        self,
        params: ParamsSpec | dict[str, Any] | None = None,
        *,
        globals: Any = None,
        assumptions: Sequence[SymBool] | None = None,
    ) -> None:
        if globals is not None:
            raise NotImplementedError("global shape specifications are not supported")
        self._params = ParamsSpec(params) if isinstance(params, dict) else params
        if self._params is not None and not isinstance(self._params, ParamsSpec):
            raise TypeError("params must be a ParamsSpec, dictionary, or None")
        self._assumptions = list(assumptions or [])
        for index, assumption in enumerate(self._assumptions):
            if not isinstance(assumption, SymBool):
                raise TypeError(f"assumption {index} must be a symbolic boolean")
            _validate_spec_sym(assumption, where=f"assumptions[{index}]")

    @property
    def params(self) -> ParamsSpec | None:
        return self._params

    @property
    def assumptions(self) -> list[SymBool]:
        return list(self._assumptions)

    def __repr__(self) -> str:
        lines = ["shapes_spec:"]
        if self._params is not None:
            lines.append(f"{_INDENT}params:")
            lines.extend(_INDENT * 2 + line for line in repr(self._params).splitlines())
        if self._assumptions:
            lines.append(f"{_INDENT}assumptions:")
            lines.extend(_INDENT * 2 + repr(value) for value in self._assumptions)
        return "\n".join(lines)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "ShapesSpec",
            "params": None if self._params is None else self._params.to_jsonable(),
            "assumptions": [repr(value) for value in self._assumptions],
        }


def _coerce_to_shapes_spec(value: Any) -> ShapesSpec | None:
    if value is None:
        return None
    if isinstance(value, ShapesSpec):
        return value
    if isinstance(value, (dict, ParamsSpec)):
        return ShapesSpec(value)
    raise TypeError("dynamic specification must be a dictionary or a specification object")


def dynamic_spec(spec: Any) -> Callable[[Any], Any]:
    """Attach a validated dynamic specification to a callable."""

    resolved = _coerce_to_shapes_spec(spec)
    if resolved is None:
        raise TypeError("dynamic_spec requires a specification")

    def decorate(fn: Any) -> Any:
        if hasattr(fn, _DYNAMIC_SPEC_ATTR):
            raise ValueError("a dynamic specification is already attached")
        setattr(fn, _DYNAMIC_SPEC_ATTR, resolved)
        return fn

    return decorate


def _resolve_dynamic_shapes(fn_or_module: Any, dynamic_shapes_kwarg: Any) -> Any:
    fn = getattr(fn_or_module, "forward", fn_or_module)
    attached = getattr(fn, _DYNAMIC_SPEC_ATTR, None)
    if attached is not None and dynamic_shapes_kwarg is not None:
        raise ValueError("provide either an attached specification or a call-site specification")
    return attached if attached is not None else dynamic_shapes_kwarg
