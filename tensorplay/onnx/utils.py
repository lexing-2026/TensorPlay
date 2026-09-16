"""Registration utilities for custom operator lowerings.

Bridges user-registered symbolic handlers into the exporter's lowering
registry, the counterpart of the built-in handlers in
``tensorplay.onnx._composite_ops``.
"""

from __future__ import annotations

from typing import Callable

from ._composite_ops import (
    register as _register_builtin,
    register_method as _register_method,
)
from .errors import OnnxExporterError

__all__ = [
    "register_custom_op_symbolic",
    "unregister_custom_op_symbolic",
    "register_custom_method_symbolic",
]


def register_custom_op_symbolic(op_name: str,
                                symbolic_fn: Callable,
                                opset_version: int,
                                params: str = "") -> None:
    """Register a symbolic handler for a custom operator.

    Args:
        op_name: the captured operator name (the ``call_function`` target's
            plain name, or the ``call_method`` name).
        symbolic_fn: handler called with an ``OpContext`` and the declared
            parameters; it emits ONNX nodes through the context's builder.
        opset_version: minimum opset the handler requires (recorded for
            diagnostics; the export still targets one global opset).
        params: space-separated parameter names passed positionally to
            ``symbolic_fn``.
    """
    if opset_version < 1:
        raise ValueError("opset_version must be a positive integer")

    def handler(context):
        return symbolic_fn(context, *[
            getattr(context, name) for name in params.split() if name
        ])

    _register_builtin(op_name, params or "", module=None, methods=False)(handler)


def register_custom_method_symbolic(op_name: str, symbolic_fn: Callable,
                                    opset_version: int, params: str = "") -> None:
    """Register a symbolic handler used only for ``call_method`` nodes."""
    if opset_version < 1:
        raise ValueError("opset_version must be a positive integer")
    _register_method(op_name, params or "")(symbolic_fn)


def unregister_custom_op_symbolic(op_name: str, domain: str = "",
                                  opset_version: int = 1) -> None:
    """Remove a previously registered custom handler.

    Built-in handlers cannot be removed this way; requesting that raises.
    """
    from ._composite_ops import _ANY_MODULE_HANDLERS, _METHOD_HANDLERS

    if not domain:
        removed = _ANY_MODULE_HANDLERS.pop(op_name, None)
        removed = _METHOD_HANDLERS.pop(op_name, None) or removed
        if removed is None:
            raise OnnxExporterError(
                f"unregister_custom_op_symbolic(): no custom handler named "
                f"{op_name!r}")
        return
    raise OnnxExporterError(
        "unregister_custom_op_symbolic(): a non-empty domain does not match "
        "any registered handler namespace")
