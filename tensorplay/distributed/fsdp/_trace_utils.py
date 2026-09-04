"""Tracing records for sharded module execution."""

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from ...graph import Proxy, Tracer


@dataclass
class TracingConfig:
    tracer: Any = field(default_factory=Tracer)
    concrete_args: dict[str, Any] | None = None


class _ParamUsageInfo(NamedTuple):
    module: Any
    named_params: list[tuple[str, Any]]


class _ExecutionInfo:
    def __init__(self, root_module: Any) -> None:
        self.curr_module = root_module
        self.module_forward_order: list[Any] = [root_module]
        self.module_to_param_usage_infos: dict[Any, list[_ParamUsageInfo]] = {
            root_module: []
        }
        self.param_forward_order: list[Any] = []
        self.visited_params: set[Any] = set()


class _ExecOrderTracer:
    def __init__(self, config: TracingConfig | None = None) -> None:
        self.config = config or TracingConfig()
        self.exec_info: _ExecutionInfo | None = None
        self.records: list[_ExecutionInfo] = []

    @contextmanager
    def patch_tracer(self, tracer: Any, root_module: Any):
        self.exec_info = _ExecutionInfo(root_module)
        self.records.append(self.exec_info)
        original_call_module = tracer.call_module
        original_create_proxy = tracer.create_proxy
        tracer.call_module = functools.partial(
            self._patched_call_module, original_call_module, self.exec_info
        )
        fqn_to_param = dict(root_module.named_parameters())
        tracer.create_proxy = functools.partial(
            self._patched_create_proxy,
            original_create_proxy,
            self.exec_info,
            fqn_to_param,
        )
        try:
            yield self.exec_info
        finally:
            tracer.call_module = original_call_module
            tracer.create_proxy = original_create_proxy

    def _patched_call_module(
        self, call_module: Any, exec_info: _ExecutionInfo, module: Any,
        forward: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        exec_info.module_forward_order.append(module)
        named_params = list(module.named_parameters())
        current = exec_info.curr_module
        if named_params:
            exec_info.module_to_param_usage_infos.setdefault(current, []).append(
                _ParamUsageInfo(module, named_params)
            )
        previous = current
        exec_info.curr_module = module
        exec_info.module_to_param_usage_infos.setdefault(module, [])
        output = call_module(module, forward, args, kwargs)
        exec_info.curr_module = previous
        return output

    def _patched_create_proxy(
        self, create_proxy: Any, exec_info: _ExecutionInfo,
        fqn_to_param: dict[str, Any], kind: str, target: Any,
        args: tuple[Any, ...], kwargs: dict[str, Any], *extra: Any
    ) -> Any:
        try:
            proxy = create_proxy(kind, target, args, kwargs, *extra)
        except TypeError:
            proxy = create_proxy(kind, target, args, kwargs)
        current = exec_info.curr_module
        if kind in ("call_function", "call_method"):
            named_params: list[tuple[str, Any]] = []
            for arg in args or ():
                if isinstance(arg, Proxy) and getattr(arg.node, "target", None) in fqn_to_param:
                    name = arg.node.target
                    param = fqn_to_param[name]
                    named_params.append((name, param))
                    if param not in exec_info.visited_params:
                        exec_info.visited_params.add(param)
                        exec_info.param_forward_order.append(param)
            if named_params:
                exec_info.module_to_param_usage_infos.setdefault(current, []).append(
                    _ParamUsageInfo(current, named_params)
                )
        elif kind == "call_module":
            named_params = list(current.named_parameters())
            if named_params:
                exec_info.module_to_param_usage_infos.setdefault(current, []).append(
                    _ParamUsageInfo(current, named_params)
                )
            for _, param in named_params:
                if param not in exec_info.visited_params:
                    exec_info.visited_params.add(param)
                    exec_info.param_forward_order.append(param)
        return proxy

    def reset(self) -> None:
        self.exec_info = None
        self.records.clear()


__all__ = ["TracingConfig", "_ParamUsageInfo", "_ExecutionInfo", "_ExecOrderTracer"]
