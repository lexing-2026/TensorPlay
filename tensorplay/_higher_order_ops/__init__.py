"""Higher-order operators.

A higher-order operator takes callables (graphs) among its inputs.  The
operators in this package expose a dispatch-key registry whose default
registration is the composite eager implementation; under graph capture the
call is recorded as one opaque node.

``cond`` and ``while_loop`` are re-exported at the package root; ``map`` and
``scan`` stay under this namespace.

The operators load lazily: modules in this package reach back into the
graph and nn layers, so an eager import here would close a cycle during
package initialization.
"""

import sys
import types
from typing import Any

__all__ = [
    "auto_functionalized",
    "cond",
    "cond_op",
    "map",
    "map_impl",
    "scan",
    "scan_op",
    "while_loop",
    "while_loop_op",
    "while_loop_stack_output_op",
    "omni_attention",
    "omni_attention_backward",
]

_LAZY_ATTRS = {
    "auto_functionalized": "tensorplay._higher_order_ops.auto_functionalize",
    "cond": "tensorplay._higher_order_ops.cond",
    "cond_op": "tensorplay._higher_order_ops.cond",
    "map": "tensorplay._higher_order_ops.map",
    "map_impl": "tensorplay._higher_order_ops.map",
    "scan": "tensorplay._higher_order_ops.scan",
    "scan_op": "tensorplay._higher_order_ops.scan",
    "while_loop": "tensorplay._higher_order_ops.while_loop",
    "while_loop_op": "tensorplay._higher_order_ops.while_loop",
    "while_loop_stack_output_op": "tensorplay._higher_order_ops.while_loop",
    "omni_attention": "tensorplay._higher_order_ops.omni_attention",
    "omni_attention_backward": "tensorplay._higher_order_ops.omni_attention",
}


class _OperatorPackage(types.ModuleType):
    """This package, keeping its operator names bound to the operators.

    Several operators share a name with the module that defines them, and
    importing that module binds the name on the package to the module -- after
    which the operator imported by name is a module, and calling it fails.  The
    module stays reachable as an import; only the name keeps meaning the
    operator.
    """

    def __setattr__(self, name: str, value: Any) -> None:
        if (
            isinstance(value, types.ModuleType)
            and _LAZY_ATTRS.get(name) == value.__name__
        ):
            return
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _OperatorPackage


def __getattr__(name: str) -> Any:
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
