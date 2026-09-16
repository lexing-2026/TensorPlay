"""Quantization configurations.

A :class:`QConfig` pairs observer factories for activations and weights.
Observers are registered as classes (or ``with_args``-specialized callables),
never instances; the preparation step instantiates fresh observers per layer.
"""

from __future__ import annotations

from collections import namedtuple

import tensorplay
from tensorplay import nn

from .observer import (
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
    default_dynamic_quant_observer,
    default_observer,
    default_weight_observer,
)

__all__ = [
    "QConfig",
    "QConfigMapping",
    "default_qconfig",
    "default_dynamic_qconfig",
    "default_per_channel_qconfig",
    "default_weight_only_qconfig",
]


class QConfig(namedtuple("QConfig", ["activation", "weight"])):
    """Describes how to quantize a layer: observer factories for activations
    and weights.

    Both fields must be observer **classes** or callables returning observer
    instances.  Use ``MyObserver.with_args(x=1)`` to override constructor
    arguments.
    """

    __slots__ = ()

    def __new__(cls, activation, weight):
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "QConfig received observer instance, please pass observer "
                "class instead. Use MyObserver.with_args(x=1) to override "
                "arguments to constructor if needed")
        return super().__new__(cls, activation, weight)


class QConfigMapping:
    """Pattern-based assignment of qconfigs to modules.

    Resolution order for a module: exact fully-qualified name, then module
    type, then the global qconfig.
    """

    def __init__(self):
        self.global_qconfig = None
        self.object_type_qconfigs: dict = {}
        self.module_name_qconfigs: dict = {}

    def set_global(self, qconfig):
        self.global_qconfig = qconfig
        return self

    def set_object_type(self, object_type, qconfig):
        self.object_type_qconfigs[object_type] = qconfig
        return self

    def set_module_name(self, module_name, qconfig):
        self.module_name_qconfigs[module_name] = qconfig
        return self

    def to_dict(self) -> dict:
        return {
            "": self.global_qconfig,
            "object_type": dict(self.object_type_qconfigs),
            "module_name": dict(self.module_name_qconfigs),
        }

    @classmethod
    def from_dict(cls, qconfig_dict) -> "QConfigMapping":
        mapping = cls()
        if qconfig_dict.get(""):
            mapping.set_global(qconfig_dict[""])
        for object_type, qconfig in qconfig_dict.get("object_type", {}).items():
            mapping.set_object_type(object_type, qconfig)
        for module_name, qconfig in qconfig_dict.get("module_name", {}).items():
            mapping.set_module_name(module_name, qconfig)
        return mapping


default_qconfig = QConfig(activation=default_observer, weight=default_weight_observer)

default_per_channel_qconfig = QConfig(
    activation=MovingAverageMinMaxObserver.with_args(dtype=tensorplay.qint8),
    weight=PerChannelMinMaxObserver.with_args(ch_axis=0),
)

default_dynamic_qconfig = QConfig(
    activation=default_dynamic_quant_observer,
    weight=default_weight_observer,
)

default_weight_only_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=tensorplay.qint8, is_dynamic=False),
    weight=PerChannelMinMaxObserver.with_args(ch_axis=0),
)
