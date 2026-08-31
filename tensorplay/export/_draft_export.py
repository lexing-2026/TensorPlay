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

    def apply_suggested_fixes(self) -> None:
        raise NotImplementedError("automatic application of suggestions is unavailable")


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

    On success the returned report carries the captured program.  When
    capture fails, the report records the failure and ``exported_program``
    stays ``None``; call ``raise_on_failure()`` to surface the errors.
    """

    try:
        program = export(model, *args, dynamic_shapes=dynamic_shapes, **kwargs)
    except Exception as exc:
        return DraftExportReport(
            [FailureReport(FailureType.DATA_DEPENDENT_ERROR, {"message": str(exc)})]
        )
    report = DraftExportReport([], exported_program=program)
    try:
        program.validate()
    except Exception as exc:
        report.failures.append(
            FailureReport(FailureType.MISMATCHED_KERNEL, {"message": str(exc)})
        )
    return report
