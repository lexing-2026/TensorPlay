"""Tracer that keeps quantizable modules as graph leaf nodes."""

from collections.abc import Callable

from tensorplay.graph import Tracer
from tensorplay.ao.nn.intrinsic import _FusedModule
from tensorplay.ao.quantization import FakeQuantizeBase, ObserverBase
from tensorplay.ao.quantization.stubs import DeQuantStub, QuantStub

__all__ = ["QuantizationTracer"]


class QuantizationTracer(Tracer):
    """A tracer that records quantizable ``tensorplay.nn`` modules as leaves."""

    def __init__(
        self, skipped_module_names: list[str] | None = None,
        skipped_module_classes: list[Callable] | None = None,
    ):
        super().__init__()
        self.skipped_module_names = skipped_module_names or []
        self.skipped_module_classes = skipped_module_classes or []

    def is_leaf_module(self, m, module_qualified_name: str) -> bool:
        tp = __import__("tensorplay")
        module_under_nn = m.__module__.startswith(
            ("tensorplay.nn", "tensorplay.ao.nn")
        ) and not isinstance(m, tp.nn.Sequential)
        return (
            module_under_nn
            or isinstance(m, (QuantStub, DeQuantStub, ObserverBase, FakeQuantizeBase))
            or module_qualified_name in self.skipped_module_names
            or type(m) in self.skipped_module_classes
            or isinstance(m, _FusedModule)
        )