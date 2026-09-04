"""Diagnostics for graph capture failures and inferred constraints."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

from ._trace import export

__all__ = [
    "DraftExportReport",
    "ExpressionCreatedNode",
    "FailureReport",
    "FailureType",
    "LogRecord",
    "draft_export",
    "get_loc",
    "prettify_frame_locals",
    "prettify_stack",
]


class FailureType(IntEnum):
    MISSING_KERNEL = 1
    DATA_DEPENDENT_ERROR = 2
    GUARD_ADDED = 3
    MISMATCHED_KERNEL = 4

    def __str__(self) -> str:
        return self.name


def prettify_stack(stack: list[dict[str, Any]], str_to_filename: dict[int, str]) -> str:
    lines = []
    for frame in stack:
        filename = str_to_filename.get(frame.get("filename"), frame.get("filename", "<unknown>"))
        lines.append(f"File {filename}, line {frame.get('line', '?')}, in {frame.get('name', '<unknown>')}")
        if frame.get("loc"):
            lines.append(f"    {frame['loc']}")
    return "\n".join(lines)


def prettify_frame_locals(loc: str, locals: dict[str, Any], symbols: dict[str, Any]) -> str:
    lines = [loc]
    lines.extend(f"{name}: {value}" for name, value in locals.items())
    lines.extend(f"{name}: {value}" for name, value in symbols.items() if value is not None)
    return "\n".join(lines)


def get_loc(filename: str, lineno: int) -> str | None:
    try:
        return Path(filename).read_text().splitlines()[lineno - 1].strip()
    except (OSError, IndexError):
        return None


class FailureReport:
    def __init__(self, failure_type: FailureType, data: dict[str, Any], xfail: bool = False) -> None:
        self.failure_type = failure_type
        self.data = dict(data)
        self.xfail = xfail

    def __repr__(self) -> str:
        return f"FailureReport({self.failure_type!s}, xfail={self.xfail}, data={self.data!r})"

    def print(self, str_to_filename: dict[int, str] | None = None) -> str:
        text = self.data.get("message", self.data.get("expr", self.failure_type.name))
        stack = self.data.get("stack")
        if stack:
            text = f"{text}\n{prettify_stack(stack, str_to_filename or {})}"
        return f"{self.failure_type.name}: {text}"


class DraftExportReport:
    def __init__(
        self,
        failures: list[FailureReport],
        str_to_filename: dict[int, str] | None = None,
        expressions_created: dict[int, dict[str, Any]] | None = None,
        op_profiles: dict[str, set[Any]] | None = None,
        exported_program: Any = None,
    ) -> None:
        self.failures = list(failures)
        self.str_to_filename = dict(str_to_filename or {})
        self.expressions_created = dict(expressions_created or {})
        self.op_profiles = dict(op_profiles or {})
        self.exported_program = exported_program

    @property
    def success(self) -> bool:
        return self.successful()

    def successful(self) -> bool:
        return not any(not failure.xfail for failure in self.failures)

    def raise_on_failure(self) -> None:
        if not self.successful():
            raise RuntimeError("\n".join(failure.print(self.str_to_filename) for failure in self.failures))

    def __repr__(self) -> str:
        return f"DraftExportReport({self.failures!r})"

    def __str__(self) -> str:
        if self.successful():
            return "graph capture completed without recorded failures"
        return "\n".join(failure.print(self.str_to_filename) for failure in self.failures)

    def apply_suggested_fixes(self) -> Any:
        """Re-export with dynamic shapes repaired from recorded suggestions.

        Requires the report to carry the original capture inputs; refinement
        reuses the textual fixes rendered by the dimension-constraint solver.
        """

        from .dynamic_shapes import refine_dynamic_shapes_from_suggested_fixes

        capture = getattr(self, "_capture_inputs", None)
        if capture is None:
            raise RuntimeError(
                "suggested fixes cannot be applied without the original capture inputs"
            )
        model, args, kwargs, dynamic_shapes = capture
        refined = dynamic_shapes
        for failure in self.failures:
            message = failure.data.get("message", "")
            if "Suggested fixes:" in message:
                refined = refine_dynamic_shapes_from_suggested_fixes(message, refined)
        self._capture_inputs = (model, args, kwargs, refined)
        retry = draft_export(model, *args, dynamic_shapes=refined, **dict(kwargs))
        if retry.success:
            self.failures = []
            self.exported_program = retry.exported_program
        return self.exported_program


@dataclass
class ExpressionCreatedNode:
    result_id: int
    argument_ids: list[int]
    record: dict[str, object]
    visited: bool = False


class LogRecord:
    def __init__(self) -> None:
        self.log_count: dict[int, int] = {}
        self.logs: list[tuple[str, dict[str, Any]]] = []

    def add(self, name: str, data: dict[str, Any]) -> None:
        self.logs.append((name, dict(data)))
        key = hash((name, repr(sorted(data.items(), key=lambda item: item[0]))))
        self.log_count[key] = self.log_count.get(key, 0) + 1


def draft_export(model: Any, *args: Any, dynamic_shapes: Any = None, **kwargs: Any) -> "DraftExportReport":
    """Capture a program, reporting failures instead of raising them.

    On success the returned report carries the captured program, checked two
    ways: structural validation of the graph, and a numerical comparison of
    the captured program against eager execution on the example inputs.  When
    capture fails, the report records the failure and ``exported_program``
    stays ``None``; call ``raise_on_failure()`` to surface the errors.
    """

    try:
        program = export(model, *args, dynamic_shapes=dynamic_shapes, **kwargs)
    except Exception as exc:
        report = DraftExportReport(
            [FailureReport(FailureType.DATA_DEPENDENT_ERROR, {"message": str(exc)})]
        )
        report._capture_inputs = (model, args, dict(kwargs), dynamic_shapes)
        if "Suggested fixes:" in str(exc):
            report.failures[0].xfail = False
        return report
    report = DraftExportReport([], exported_program=program)
    report._capture_inputs = (model, args, dict(kwargs), dynamic_shapes)
    try:
        program.validate()
    except Exception as exc:
        report.failures.append(
            FailureReport(FailureType.MISMATCHED_KERNEL, {"message": str(exc)})
        )
    _compare_against_eager(report, program, model, args, kwargs)
    return report


def _flatten_numbers(value: Any) -> list[Any]:
    if isinstance(value, (tuple, list)):
        leaves: list[Any] = []
        for item in value:
            leaves.extend(_flatten_numbers(item))
        return leaves
    if isinstance(value, dict):
        leaves = []
        for item in value.values():
            leaves.extend(_flatten_numbers(item))
        return leaves
    return [value]


def _values_match(reference: Any, candidate: Any, tolerance: float) -> bool:
    left = _flatten_numbers(reference)
    right = _flatten_numbers(candidate)
    if len(left) != len(right):
        return False
    for expected, actual in zip(left, right):
        if hasattr(expected, "shape") or hasattr(actual, "shape"):
            try:
                if tuple(expected.shape) != tuple(actual.shape):
                    return False
                delta = (expected - actual).abs().max().item()
                scale = expected.abs().max().item()
                if delta > tolerance * max(scale, 1.0):
                    return False
            except Exception:
                if expected is not actual:
                    return False
        elif expected != actual:
            return False
    return True


def _compare_against_eager(
    report: "DraftExportReport",
    program: Any,
    model: Any,
    args: tuple[Any, ...],
    kwargs: Any,
) -> None:
    """Record a kernel-mismatch failure when graph and eager disagree."""

    try:
        eager = model(*args, **dict(kwargs))
        captured = program(*args, **dict(kwargs))
    except Exception as exc:
        report.failures.append(
            FailureReport(
                FailureType.DATA_DEPENDENT_ERROR,
                {"message": f"execution check failed: {exc}"},
            )
        )
        return
    if not _values_match(eager, captured, tolerance=1e-4):
        report.failures.append(
            FailureReport(
                FailureType.MISMATCHED_KERNEL,
                {
                    "message": (
                        "captured graph disagrees with eager execution on the "
                        "example inputs"
                    )
                },
            )
        )
