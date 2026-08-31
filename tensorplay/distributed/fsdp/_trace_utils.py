"""Tracing records for sharded module execution."""

from dataclasses import dataclass, field
from typing import Any

__all__ = ["TracingConfig", "_ParamUsageInfo", "_ExecutionInfo", "_ExecOrderTracer"]


@dataclass
class TracingConfig:
    limit_all_gathers: bool = True
    record_module_names: bool = False


@dataclass
class _ParamUsageInfo:
    fqn: str
    module_name: str
    used: bool = False


@dataclass
class _ExecutionInfo:
    module_name: str
    parameters: list[str] = field(default_factory=list)


class _ExecOrderTracer:
    def __init__(self, config: TracingConfig | None = None) -> None:
        self.config = config or TracingConfig()
        self.records: list[_ExecutionInfo] = []

    def record(self, module_name: str, parameters: list[str] | None = None) -> None:
        self.records.append(_ExecutionInfo(module_name, list(parameters or ())))

    def reset(self) -> None:
        self.records.clear()
