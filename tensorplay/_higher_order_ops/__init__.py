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
}


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
