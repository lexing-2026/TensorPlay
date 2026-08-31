"""Pass sequencing and fixpoint execution."""

from __future__ import annotations

import logging
from functools import wraps
from inspect import unwrap
from typing import Any, Callable, Sequence

from .base import PassBase, PassResult, _as_graph_module

__all__ = [
    "PassManager",
    "inplace_wrapper",
    "log_hook",
    "loop_pass",
    "these_before_those_pass_constraint",
    "this_before_that_pass_constraint",
]

logger = logging.getLogger(__name__)


def inplace_wrapper(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an in-place pass so it returns the object it received."""

    @wraps(fn)
    def wrapped(obj: Any, *args: Any, **kwargs: Any) -> Any:
        fn(obj, *args, **kwargs)
        return obj

    return wrapped


def log_hook(fn: Callable[..., Any], level: int = logging.INFO) -> Callable[..., Any]:
    """Log a pass result while preserving the wrapped callable's contract."""

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        logger.log(level, "ran graph pass %s; result=%r", fn, result)
        return result

    return wrapped


def loop_pass(
    base_pass: Callable[[Any], Any],
    n_iter: int | None = None,
    predicate: Callable[[Any], bool] | None = None,
) -> Callable[[Any], Any]:
    """Repeat a pass a fixed number of times or while a predicate is true."""

    if (n_iter is None) == (predicate is None):
        raise AssertionError("exactly one of n_iter and predicate is required")

    @wraps(base_pass)
    def wrapped(value: Any) -> Any:
        if n_iter is not None:
            if n_iter < 0:
                raise ValueError("n_iter must be non-negative")
            for _ in range(n_iter):
                value = base_pass(value)
            return value
        while predicate(value):  # type: ignore[misc]
            value = base_pass(value)
        return value

    return wrapped


def this_before_that_pass_constraint(this: Callable[..., Any], that: Callable[..., Any]):
    return lambda first, second: first != that or second != this


def these_before_those_pass_constraint(
    these: Callable[..., Any], those: Callable[..., Any]
):
    return lambda first, second: unwrap(first) != those or unwrap(second) != these


class PassManager:
    """Run transformations in order until a complete round is unchanged."""

    def __init__(
        self,
        passes: Sequence[Callable[[Any], Any]] = (),
        constraints: Sequence[Callable[[Any, Any], bool]] = (),
        *,
        run_passes_once: bool = False,
        max_iterations: int = 16,
    ) -> None:
        self.passes = list(passes)
        self.constraints = list(constraints)
        self._validated = False
        self.run_passes_once = run_passes_once
        self.max_iterations = max_iterations

    def add(self, fn: Callable[[Any], Any]) -> "PassManager":
        self.passes.append(fn)
        self._validated = False
        return self

    add_pass = add

    @classmethod
    def build_from_passlist(cls, passes: Sequence[Callable[[Any], Any]]) -> "PassManager":
        return cls(passes)

    def add_constraint(self, constraint: Callable[[Any, Any], bool]) -> None:
        self.constraints.append(constraint)
        self._validated = False

    def remove_pass(self, names: Sequence[str]) -> None:
        wanted = set(names)
        self.passes = [item for item in self.passes if item.__name__ not in wanted]
        self._validated = False

    def replace_pass(self, target: Callable[..., Any], replacement: Callable[..., Any]) -> None:
        self.passes = [replacement if item.__name__ == target.__name__ else item for item in self.passes]
        self._validated = False

    def validate(self) -> None:
        for index, first in enumerate(self.passes):
            for second in self.passes[index + 1 :]:
                if not all(constraint(first, second) for constraint in self.constraints):
                    raise RuntimeError("graph pass schedule constraint was violated")
        self._validated = True

    def __call__(self, target: Any) -> PassResult:
        if not self._validated:
            self.validate()
        graph_module = _as_graph_module(target)
        any_modified = False
        rounds = 1 if self.run_passes_once else self.max_iterations
        for _ in range(rounds):
            round_modified = False
            for fn in self.passes:
                result = fn(graph_module)
                # Both the legacy pass protocol and the infrastructure pass
                # protocol return named tuples with these two fields.  Use
                # the structural contract so passes from either layer keep
                # their replacement GraphModule and modification flag.
                has_result_shape = (
                    result is not None
                    and hasattr(result, "graph_module")
                    and hasattr(result, "modified")
                )
                modified = (
                    bool(result.modified)
                    if has_result_shape
                    else result is not None and result is not graph_module
                )
                if has_result_shape:
                    graph_module = result.graph_module
                elif result is not None and hasattr(result, "graph"):
                    graph_module = result
                round_modified = round_modified or modified
            graph_module.graph.lint()
            if not round_modified:
                break
            any_modified = True
        return PassResult(graph_module, any_modified)
