from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import tensorplay as tp

from ..interpreter import Transformer
from ..node import Node
from ..operator_schemas import create_type_hint, normalize_function, normalize_module
from ..proxy import Proxy
from .schema_type_annotation import AnnotateTypesWithSchema

__all__ = ["NormalizeArgs", "NormalizeOperators"]


class NormalizeArgs(Transformer):
    """Materialize callable defaults and normalize positional arguments."""

    def __init__(self, module: Any, normalize_to_only_use_kwargs: bool = True) -> None:
        super().__init__(module)
        self.normalize_to_only_use_kwargs = normalize_to_only_use_kwargs
        self.node_map: dict[Proxy, Node] = {}

    def _argument_type(self, value: Any) -> Any:
        if isinstance(value, Proxy):
            return value.node.type or value.node.meta.get("type", Any)
        return type(value)

    def run_node(self, node: Node) -> Any:
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        if node.op == "call_function":
            arg_types = tuple(create_type_hint(self._argument_type(value)) for value in args)
            kwarg_types = {key: self._argument_type(value) for key, value in kwargs.items()}
            result = self.call_function(node.target, args, kwargs, arg_types, kwarg_types)
        else:
            result = super().run_node(node)
        if node.op != "output" and isinstance(result, Proxy):
            self.node_map[result] = node
            result.node.meta.update(node.meta)
            result.node.type = node.type
        return result

    def call_function(
        self,
        target: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        arg_types: tuple[Any, ...] | None = None,
        kwarg_types: dict[str, Any] | None = None,
    ) -> Proxy:
        if not callable(target):
            raise TypeError(f"call target must be callable, got {type(target).__name__}")
        normalized = normalize_function(
            target,
            args,
            kwargs,
            arg_types,
            kwarg_types,
            self.normalize_to_only_use_kwargs,
        )
        if normalized is None:
            return super().call_function(target, args, kwargs)
        return self.tracer.create_proxy("call_function", target, normalized.args, normalized.kwargs)

    def call_module(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        if not isinstance(target, str):
            raise TypeError(f"module target must be a string, got {type(target).__name__}")
        normalized = normalize_module(
            self.module.root,
            target,
            args,
            kwargs,
            self.normalize_to_only_use_kwargs,
        )
        if normalized is None:
            return super().call_module(target, args, kwargs)
        return super().call_module(target, normalized.args, normalized.kwargs)


class NormalizeOperators(AnnotateTypesWithSchema):
    """Canonicalize arithmetic spellings into one operation family."""

    binary_magic_method_remap: dict[Callable[..., Any], Callable[..., Any]] = {
        tp.add: operator.add,
        tp.mul: operator.mul,
        tp.sub: operator.sub,
        tp.div: operator.truediv,
        operator.add: tp.add,
        operator.mul: tp.mul,
        operator.sub: tp.sub,
        operator.truediv: tp.div,
        operator.floordiv: getattr(tp, "floor_divide", operator.floordiv),
        operator.mod: getattr(tp, "remainder", operator.mod),
        operator.eq: tp.eq,
        operator.ne: tp.ne,
        operator.lt: tp.lt,
        operator.le: tp.le,
        operator.gt: tp.gt,
        operator.ge: tp.ge,
    }

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        if callable(target) and target in self.binary_magic_method_remap and len(args) == 2 and not kwargs:
            target = self.binary_magic_method_remap[target]
        return super().call_function(target, args, kwargs)

