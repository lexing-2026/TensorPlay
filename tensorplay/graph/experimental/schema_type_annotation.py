from __future__ import annotations

import inspect
from typing import Any

from ..graph_module import GraphModule
from ..interpreter import Transformer
from ..proxy import Proxy

__all__ = ["AnnotateTypesWithSchema"]


class AnnotateTypesWithSchema(Transformer):
    """Attach Python return annotations to values produced by a graph."""

    def __init__(
        self,
        module: GraphModule,
        annotate_functionals: bool = True,
        annotate_modules: bool = True,
        annotate_get_attrs: bool = True,
    ) -> None:
        super().__init__(module)
        self.annotate_functionals = annotate_functionals
        self.annotate_modules = annotate_modules
        self.annotate_get_attrs = annotate_get_attrs

    @staticmethod
    def _return_annotation(target: Any) -> Any:
        if not callable(target):
            return None
        try:
            annotation = inspect.signature(target).return_annotation
        except (TypeError, ValueError):
            return None
        return None if annotation is inspect.Signature.empty else annotation

    def _set_type(self, proxy: Proxy, inferred: Any) -> Proxy:
        if inferred is not None and proxy.node.type is None:
            proxy.node.type = inferred
        return proxy

    def call_function(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        result = super().call_function(target, args, kwargs)
        if self.annotate_functionals:
            return self._set_type(result, self._return_annotation(target))
        return result

    def call_module(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        result = super().call_module(target, args, kwargs)
        if self.annotate_modules:
            module = self.fetch_attr(target)
            return self._set_type(result, self._return_annotation(getattr(module, "forward", module)))
        return result

    def get_attr(self, target: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Proxy:
        result = super().get_attr(target, args, kwargs)
        if self.annotate_get_attrs and isinstance(target, str):
            value = self.fetch_attr(target)
            inferred = getattr(value, "__annotations__", {}).get("return")
            if inferred is None and isinstance(value, type):
                inferred = value
            self._set_type(result, inferred)
        return result

