"""Helpers for identifying observed graph modules."""

from tensorplay.ao.quantization.observer import ObserverBase
from tensorplay.ao.quantization.fake_quantize import FakeQuantizeBase

__all__ = [
    "is_observed_module",
    "is_observed_standalone_module",
    "is_observed_standalone_module_node",
]


def is_observed_module(module):
    """Return whether a module carries observation machinery."""
    return hasattr(module, "activation_post_process") or any(
        isinstance(child, (ObserverBase, FakeQuantizeBase))
        for child in module.children()
    )


def is_observed_standalone_module(module):
    return is_observed_module(module)


def is_observed_standalone_module_node(node):
    return node.op == "call_module" and is_observed_module(node.target)