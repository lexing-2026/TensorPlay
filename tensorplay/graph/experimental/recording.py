from __future__ import annotations

import functools
import inspect
import itertools
import logging
from dataclasses import dataclass
from typing import Any, Callable, ParamSpec, TypeVar

from ..node import Node

_P = ParamSpec("_P")
_R = TypeVar("_R")

log = logging.getLogger(__name__)

__all__ = [
    "FakeTensorMeta",
    "NotEqualError",
    "ShapeEnvEvent",
    "record_shapeenv_event",
    "replay_shape_env_events",
    "shape_env_check_state_equal",
]


def value_to_str(value: Any) -> str:
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return "{" + ", ".join(f"{key}: {item}" for key, item in items) + "}"
    if isinstance(value, set):
        return "{" + ", ".join(str(item) for item in sorted(value, key=str)) + "}"
    return str(value)


def assert_equal(old: Any | None, new: Any) -> Any:
    if old is not None and old is not new:
        raise AssertionError("event arguments refer to different state objects")
    return new


def compare_vars(
    first: dict[str, Any],
    second: dict[str, Any],
    map_value: Callable[[str, object], object] = lambda _name, value: value,
) -> list[tuple[str, str, str]]:
    if set(first) != set(second):
        raise NotEqualError(
            "field set mismatch",
            [("fields", repr(sorted(first)), repr(sorted(second)))],
        )
    result: list[tuple[str, str, str]] = []
    for name in sorted(first):
        left = map_value(name, first[name])
        right = map_value(name, second[name])
        if left != right:
            result.append((f"{name}: values differ", value_to_str(left), value_to_str(right)))
    return result


def maybe_convert_node(value: Any, shape_env: Any) -> Any:
    if not isinstance(value, Node):
        return value
    mapping = getattr(shape_env, "name_to_node", None)
    if not isinstance(mapping, dict) or value.name not in mapping:
        raise AssertionError(f"node {value.name!r} is not present in the replay graph")
    return mapping[value.name]


def replacearg(
    args: list[Any], kwargs: dict[str, Any], index: int, key: str, fn: Callable[[Any], Any]
) -> None:
    if index < len(args):
        args[index] = fn(args[index])
    if key in kwargs:
        kwargs[key] = fn(kwargs[key])


def retlog(value: _R) -> _R:
    return value


def is_create_fx_call_function(event: ShapeEnvEvent) -> bool:
    return event.name in {"_create_graph_call", "create_graph_call", "_create_fx_call_function"}


def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    return fn


def _tree_map(value: Any, fn: Callable[[Any], Any]) -> Any:
    mapped = fn(value)
    if mapped is not value:
        return mapped
    if isinstance(value, tuple):
        return tuple(_tree_map(item, fn) for item in value)
    if isinstance(value, list):
        return [_tree_map(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map(item, fn) for key, item in value.items()}
    if isinstance(value, set):
        return {_tree_map(item, fn) for item in value}
    return value


def _find_shape_env(value: Any) -> Any | None:
    found: Any | None = None

    def visit(item: Any) -> Any:
        nonlocal found
        candidate = item
        if hasattr(candidate, "shape_env") and getattr(candidate, "shape_env") is not None:
            candidate = getattr(candidate, "shape_env")
        elif hasattr(candidate, "events") and hasattr(candidate, "should_record_events"):
            candidate = item
        else:
            return item
        if found is not None and found is not candidate:
            raise AssertionError("event arguments reference different shape environments")
        found = candidate
        return item

    _tree_map(value, visit)
    return found


def _replace_for_replay(value: Any, old_env: Any, new_env: Any) -> Any:
    if value is old_env:
        return new_env
    if isinstance(value, Node):
        mapping = getattr(new_env, "name_to_node", None)
        if isinstance(mapping, dict) and value.name in mapping:
            return mapping[value.name]
        return value
    if hasattr(value, "shape_env") and getattr(value, "shape_env") is old_env:
        with_shape_env = getattr(value, "with_shape_env", None)
        if callable(with_shape_env):
            return with_shape_env(new_env)
        try:
            clone = value.clone()
            clone.shape_env = new_env
            return clone
        except (AttributeError, TypeError):
            return value
    return value


@dataclass
class ShapeEnvEvent:
    """One recorded state-changing call and the inputs used by that call."""

    f: Callable[..., Any]
    args: list[object] | None = None
    kwargs: dict[str, Any] | None = None
    tracked_fakes: list[Any] | None = None
    name: str | None = None

    def run(self, shape_env: Any | None = None) -> Any:
        args = list(self.args or [])
        kwargs = dict(self.kwargs or {})
        recorded_env = _find_shape_env((args, kwargs))
        if shape_env is not None and recorded_env is not None:
            args = _tree_map(args, lambda value: _replace_for_replay(value, recorded_env, shape_env))
            kwargs = _tree_map(kwargs, lambda value: _replace_for_replay(value, recorded_env, shape_env))
        if self.f is type(shape_env) and shape_env is None:
            return self.f(*args, **kwargs)
        if shape_env is None:
            return self.f(*args, **kwargs)
        if inspect.ismethod(self.f) and self.f.__self__ is not None:
            return self.f.__func__(shape_env, *args[1:], **kwargs)
        if args and args[0] is not shape_env and _find_shape_env(args[0]) is not None:
            args = [shape_env, *args[1:]]
        return self.f(*args, **kwargs)

    def __str__(self) -> str:
        name = self.name or getattr(self.f, "__name__", type(self.f).__name__)
        return f"event: {name} ({self.args}, {self.kwargs})"

    def is_create_graph_call(self) -> bool:
        return self.name in {"_create_graph_call", "create_graph_call"}

    def is_evaluate_expr(self) -> bool:
        return self.name == "evaluate_expr"

    def is_defer_runtime_assert(self) -> bool:
        return self.name in {"guard_or_defer_runtime_assert", "defer_runtime_assert"}


_NEST = 0


def record_shapeenv_event(
    *, save_tracked_fakes: bool = False, name: str | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a state-changing method with replayable event recording."""

    def decorate(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        if not callable(fn):
            raise AssertionError(f"expected callable, got {type(fn).__name__}")
        parameters = inspect.getfullargspec(fn).args
        if not parameters or parameters[0] != "self":
            raise AssertionError("record_shapeenv_event expects an instance method")
        event_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            global _NEST
            owner = args[0] if args else None
            enabled = bool(getattr(owner, "should_record_events", False))
            recording = bool(getattr(owner, "is_recording", False))
            events = getattr(owner, "events", None)
            if not enabled or recording or not isinstance(events, list):
                return fn(*args, **kwargs)

            _NEST += 1
            event = ShapeEnvEvent(
                fn,
                list(args),
                dict(kwargs),
                getattr(owner, "_snapshot_tracked_fakes", lambda: None)()
                if save_tracked_fakes
                else None,
                event_name,
            )
            events.append(event)
            try:
                recorder = getattr(owner, "_recording", None)
                if callable(recorder):
                    with recorder():
                        return event.run(owner)
                return event.run(owner)
            except Exception:
                if events and events[-1] is event:
                    events.pop()
                log.exception("failed while recording event %s", event_name)
                raise
            finally:
                _NEST -= 1

        return wrapper

    return decorate


def replay_shape_env_events(events: list[ShapeEnvEvent]) -> Any:
    """Construct a fresh state object by replaying a recorded event list."""

    if not events:
        raise AssertionError("at least one event is required")
    first = events[0]
    shape_env = first.run()
    for event in events[1:]:
        event.run(shape_env)
    return shape_env


@dataclass
class FakeTensorMeta:
    tensor_size: tuple[Any, ...]
    tensor_stride: tuple[Any, ...]
    tensor_storage_offset: Any
    is_nested: bool

    def size(self) -> tuple[Any, ...]:
        return self.tensor_size

    def stride(self) -> tuple[Any, ...]:
        return self.tensor_stride

    def storage_offset(self) -> Any:
        return self.tensor_storage_offset

    def dim(self) -> int:
        return len(self.tensor_size)

    @staticmethod
    def from_fake(fake: Any) -> "FakeTensorMeta":
        def read(name: str, default: Any = None) -> Any:
            value = getattr(fake, name, default)
            return value() if callable(value) else value

        return FakeTensorMeta(
            tuple(read("size", ())),
            tuple(read("stride", ())),
            read("storage_offset", 0),
            bool(read("is_nested", False)),
        )


class NotEqualError(Exception):
    """Raised when two state objects differ after normalization."""

    def __init__(self, msg: str, mismatched: list[tuple[str, str, str]]) -> None:
        details = "\n".join(
            f"==> {field}\n  > left: {left}\n  > right: {right}"
            for field, left, right in mismatched
        )
        super().__init__(f"state objects are not equal: {msg}\n\n{details}")


def shape_env_check_state_equal(
    env1: Any,
    env2: Any,
    non_state_variable_names: tuple[str, ...] = (),
    map_value: Callable[[str, object], object] = lambda _name, value: value,
) -> None:
    """Compare state fields after dropping runtime-only fields."""

    first = vars(env1).copy()
    second = vars(env2).copy()
    for name in non_state_variable_names:
        first.pop(name, None)
        second.pop(name, None)
    if set(first) != set(second):
        raise NotEqualError(
            "field set mismatch",
            [("fields", repr(sorted(first)), repr(sorted(second)))],
        )
    mismatched: list[tuple[str, str, str]] = []
    for name in sorted(first):
        left = map_value(name, first[name])
        right = map_value(name, second[name])
        if left != right:
            mismatched.append((f"{name}: values differ", str(left), str(right)))
    if mismatched:
        raise NotEqualError("field values differ", mismatched)
