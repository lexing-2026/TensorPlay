from __future__ import annotations

from types import TracebackType
from typing import Any

import tensorplay


def _native_module() -> Any:
    native = getattr(getattr(tensorplay, "_C", None), "_distributed_autograd", None)
    if native is None:
        raise RuntimeError(
            "the native distributed autograd runtime is not available"
        )
    return native


def is_available() -> bool:
    return getattr(getattr(tensorplay, "_C", None), "_distributed_autograd", None) is not None


def is_initialized() -> bool:
    native = getattr(getattr(tensorplay, "_C", None), "_distributed_autograd", None)
    return bool(native is not None and native._is_initialized())


def backward(
    context_id: int,
    roots: Any,
    retain_graph: bool = False,
) -> None:
    _native_module().backward(int(context_id), roots, bool(retain_graph))


def get_gradients(context_id: int) -> dict[Any, Any]:
    return dict(_native_module().get_gradients(int(context_id)))


class context:
    def __enter__(self) -> int:
        native = _native_module()
        self.autograd_context = native._new_context()
        self._context_id = int(self.autograd_context._context_id())
        return self._context_id

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        _native_module()._release_context(self._context_id)
        self.autograd_context = None


DistAutogradContext = getattr(
    getattr(tensorplay, "_C", None), "_distributed_autograd", None
)
if DistAutogradContext is not None:
    DistAutogradContext = DistAutogradContext.DistAutogradContext


__all__ = [
    "DistAutogradContext",
    "backward",
    "context",
    "get_gradients",
    "is_available",
    "is_initialized",
]
