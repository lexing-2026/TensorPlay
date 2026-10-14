"""Quantization configurations.

A :class:`QConfig` pairs observer factories for activations and weights.
Observers are registered as classes (or ``with_args``-specialized callables),
never instances; the preparation step instantiates fresh observers per layer.
"""

from __future__ import annotations

from collections import namedtuple, OrderedDict
import re

import tensorplay
from tensorplay import nn

from .observer import (
    MovingAverageMinMaxObserver,
    PerChannelMinMaxObserver,
    PlaceholderObserver,
    default_dynamic_quant_observer,
    default_observer,
    default_weight_observer,
    default_per_channel_weight_observer,
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


_QCONFIG_STYLE_ORDER = [
    "global_qconfig",
    "object_type_qconfigs",
    "module_name_regex_qconfigs",
    "module_name_qconfigs",
    "module_name_object_type_order_qconfigs",
]


class QConfigMapping:
    """Pattern-based assignment of qconfigs to modules.

    Resolution order for a module (increasing match priority): global qconfig,
    object type, module-name regex, exact fully-qualified name, then the
    ``(module name, object type, occurrence index)`` triple.
    """

    def __init__(self):
        self.global_qconfig = None
        self.object_type_qconfigs: OrderedDict = OrderedDict()
        self.module_name_regex_qconfigs: OrderedDict = OrderedDict()
        self.module_name_qconfigs: OrderedDict = OrderedDict()
        self.module_name_object_type_order_qconfigs: OrderedDict = OrderedDict()

    def set_global(self, qconfig):
        self.global_qconfig = qconfig
        return self

    def set_object_type(self, object_type, qconfig):
        self.object_type_qconfigs[object_type] = qconfig
        return self

    def set_module_name_regex(self, module_name_regex, qconfig):
        self.module_name_regex_qconfigs[module_name_regex] = qconfig
        return self

    def set_module_name(self, module_name, qconfig):
        self.module_name_qconfigs[module_name] = qconfig
        return self

    def set_module_name_object_type_order(self, module_name, object_type, index, qconfig):
        self.module_name_object_type_order_qconfigs[(module_name, object_type, index)] = qconfig
        return self

    def get_global_qconfig(self):
        return self.global_qconfig

    def get_object_type_qconfig(self, object_type):
        return self.object_type_qconfigs.get(object_type)

    def get_module_name_regex_qconfig(self, module_name):
        for regex, qconfig in self.module_name_regex_qconfigs.items():
            if re.fullmatch(regex, module_name):
                return qconfig
        return None

    def get_module_name_qconfig(self, module_name):
        return self.module_name_qconfigs.get(module_name)

    def get_module_name_object_type_order_qconfig(self, module_name, object_type, index):
        return self.module_name_object_type_order_qconfigs.get((module_name, object_type, index))

    def _get_qconfig_for_module(self, module, prefix):
        """Resolve the qconfig for a module following match priority order."""
        qconfig = self.module_name_object_type_order_qconfigs.get((prefix, type(module), -1))
        if qconfig is not None:
            return qconfig
        qconfig = self.module_name_qconfigs.get(prefix)
        if qconfig is not None:
            return qconfig
        qconfig = self.get_module_name_regex_qconfig(prefix)
        if qconfig is not None:
            return qconfig
        qconfig = self.object_type_qconfigs.get(type(module))
        if qconfig is not None:
            return qconfig
        return self.global_qconfig

    def to_dict(self) -> dict:
        return {
            "": self.global_qconfig,
            "object_type": list(self.object_type_qconfigs.items()),
            "module_name_regex": list(self.module_name_regex_qconfigs.items()),
            "module_name": list(self.module_name_qconfigs.items()),
            "module_name_object_type_order": [
                (*k, v) for k, v in self.module_name_object_type_order_qconfigs.items()
            ],
        }

    @classmethod
    def from_dict(cls, qconfig_dict) -> "QConfigMapping":
        mapping = cls()
        if qconfig_dict.get(""):
            mapping.set_global(qconfig_dict[""])
        for object_type, qconfig in qconfig_dict.get("object_type", []):
            mapping.set_object_type(object_type, qconfig)
        for regex, qconfig in qconfig_dict.get("module_name_regex", []):
            mapping.set_module_name_regex(regex, qconfig)
        for module_name, qconfig in qconfig_dict.get("module_name", []):
            mapping.set_module_name(module_name, qconfig)
        for module_name, object_type, index, qconfig in qconfig_dict.get(
            "module_name_object_type_order", []
        ):
            mapping.set_module_name_object_type_order(module_name, object_type, index, qconfig)
        return mapping

    def __repr__(self):
        output = self.__class__.__name__ + " ("
        for style_name in _QCONFIG_STYLE_ORDER:
            output += f"\n {style_name}"
            qconfigs = getattr(self, style_name)
            if isinstance(qconfigs, OrderedDict) and len(qconfigs) > 0:
                for key, qconfig in qconfigs.items():
                    output += f"\n  {key}: {qconfig}"
            else:
                output += f"\n  {qconfigs}"
        return output + "\n)"


default_qconfig = QConfig(activation=default_observer, weight=default_weight_observer)

default_per_channel_qconfig = QConfig(
    activation=default_observer,
    weight=default_per_channel_weight_observer,
)

default_dynamic_qconfig = QConfig(
    activation=default_dynamic_quant_observer,
    weight=default_weight_observer,
)

from .fake_quantize import default_fake_quant, default_weight_fake_quant, default_per_channel_weight_fake_quant

default_weight_only_qconfig = QConfig(activation=nn.Identity, weight=default_weight_fake_quant)

default_qat_qconfig = QConfig(activation=default_fake_quant, weight=default_weight_fake_quant)
default_per_channel_qat_qconfig = QConfig(activation=default_fake_quant, weight=default_per_channel_weight_fake_quant)
__all__ += ['default_qat_qconfig', 'default_per_channel_qat_qconfig']

from .observer import HistogramObserver, default_per_channel_weight_observer
from .fake_quantize import (
    FakeQuantize, FusedMovingAvgObsFakeQuantize, default_dynamic_fake_quant,
    default_fused_act_fake_quant, default_fused_wt_fake_quant,
    default_fused_per_channel_wt_fake_quant,
)

default_qat_qconfig_v2 = QConfig(default_fused_act_fake_quant, default_fused_wt_fake_quant)
default_dynamic_qat_qconfig = QConfig(default_dynamic_fake_quant, default_weight_fake_quant)


def get_default_qconfig(backend='x86', version=0):
    if backend not in ('x86', 'fbgemm', 'qnnpack', 'onednn') or version != 0:
        raise ValueError('Unsupported calibration configuration backend or version')
    return QConfig(HistogramObserver.with_args(reduce_range=backend in ('x86', 'fbgemm')),
                   default_weight_observer if backend == 'qnnpack' else default_per_channel_weight_observer)


def get_default_qat_qconfig(backend='x86', version=1):
    if backend not in ('x86', 'fbgemm', 'qnnpack', 'onednn') or version not in (0, 1):
        raise ValueError('Unsupported training configuration backend or version')
    cls = FusedMovingAvgObsFakeQuantize if version else FakeQuantize
    weights = ((default_fused_wt_fake_quant if backend == 'qnnpack' else default_fused_per_channel_wt_fake_quant)
               if version else (default_weight_fake_quant if backend == 'qnnpack' else default_per_channel_weight_fake_quant))
    return QConfig(cls.with_args(quant_min=0, quant_max=255, reduce_range=backend in ('x86', 'fbgemm')), weights)


__all__ += ['default_qat_qconfig_v2', 'default_dynamic_qat_qconfig', 'get_default_qconfig', 'get_default_qat_qconfig']

from .observer import default_placeholder_observer, default_float_qparams_observer, default_float_qparams_observer_4bit
from .fake_quantize import default_embedding_fake_quant
float_qparams_weight_only_qconfig = QConfig(default_placeholder_observer, default_float_qparams_observer)
float_qparams_weight_only_qconfig_4bit = QConfig(default_placeholder_observer, default_float_qparams_observer_4bit)
default_embedding_qat_qconfig = QConfig(default_placeholder_observer, default_embedding_fake_quant)
per_channel_dynamic_qconfig = QConfig(default_dynamic_quant_observer, default_per_channel_weight_observer)
__all__ += ['float_qparams_weight_only_qconfig', 'default_embedding_qat_qconfig', 'per_channel_dynamic_qconfig']
__all__ += ['float_qparams_weight_only_qconfig_4bit']

float16_dynamic_qconfig = QConfig(
    PlaceholderObserver.with_args(dtype=tensorplay.float16, is_dynamic=True),
    PlaceholderObserver.with_args(dtype=tensorplay.float16))
__all__ += ['float16_dynamic_qconfig']

import copy
import warnings
from typing_extensions import TypeAliasType

from .observer import (
    MinMaxObserver,
    NoopObserver,
    ReuseInputObserver,
    default_debug_observer,
    default_reuse_input_observer,
    per_channel_weight_observer_range_neg_127_to_127,
    weight_observer_range_neg_127_to_127,
)
from .fake_quantize import (
    default_embedding_fake_quant_4bit,
    default_fused_per_channel_wt_fake_quant,
    default_fused_wt_fake_quant,
    fused_per_channel_wt_fake_quant_range_neg_127_to_127,
    fused_wt_fake_quant_range_neg_127_to_127,
    default_per_channel_weight_fake_quant,
    default_weight_fake_quant,
    FakeQuantizeBase,
)

QConfigAny = TypeAliasType("QConfigAny", QConfig | None)


class QConfigDynamic(namedtuple("QConfigDynamic", ["activation", "weight"])):
    """Describes how to dynamically quantize a layer.

    Both fields must be observer **classes** or callables returning observer
    instances, never instances themselves.
    """

    __slots__ = ()

    def __new__(cls, activation=nn.Identity, weight=nn.Identity):
        if isinstance(weight, nn.Module):
            raise ValueError(
                "QConfigDynamic received observer instance, please pass observer "
                "class instead. Use MyObserver.with_args(x=1) to override "
                "arguments to constructor if needed")
        return super().__new__(cls, activation, weight)


default_debug_qconfig = QConfig(
    weight=default_weight_observer, activation=default_debug_observer
)
float16_static_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=tensorplay.float16),
    weight=PlaceholderObserver.with_args(dtype=tensorplay.float16),
)
default_activation_only_qconfig = QConfig(
    activation=default_fake_quant, weight=nn.Identity
)
default_reuse_input_qconfig = QConfig(
    activation=default_reuse_input_observer, weight=NoopObserver
)
default_quint8_weight_qconfig = QConfig(
    activation=HistogramObserver, weight=MinMaxObserver
)
default_embedding_qat_qconfig_4bit = QConfig(
    activation=NoopObserver.with_args(dtype=tensorplay.float32),
    weight=default_embedding_fake_quant_4bit,
)
default_symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=tensorplay.qint8, reduce_range=False, eps=2**-12
    ),
    weight=weight_observer_range_neg_127_to_127,
)
default_per_channel_symmetric_qnnpack_qconfig = QConfig(
    activation=HistogramObserver.with_args(
        dtype=tensorplay.qint8, reduce_range=False, eps=2**-12
    ),
    weight=per_channel_weight_observer_range_neg_127_to_127,
)
default_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=tensorplay.qint8,
        reduce_range=False,
        eps=2**-12,
    ),
    weight=fused_wt_fake_quant_range_neg_127_to_127,
)
default_per_channel_symmetric_qnnpack_qat_qconfig = QConfig(
    activation=FusedMovingAvgObsFakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=tensorplay.qint8,
        reduce_range=False,
        eps=2**-12,
    ),
    weight=fused_per_channel_wt_fake_quant_range_neg_127_to_127,
)
_default_fp32_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=tensorplay.float32),
    weight=PlaceholderObserver.with_args(dtype=tensorplay.float32),
)
_default_quint8_placeholder_qconfig = QConfig(
    activation=PlaceholderObserver.with_args(dtype=tensorplay.quint8),
    weight=None,
)


def _assert_valid_qconfig(qconfig, mod):
    """Verify that ``qconfig`` is usable with module ``mod``."""
    if qconfig is None:
        return
    is_conv_transpose_mod = isinstance(
        mod,
        (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d),
    )
    if is_conv_transpose_mod:
        if qconfig.weight is None:
            return
        example_observer = qconfig.weight()
        is_per_channel = isinstance(
            example_observer,
            (
                PerChannelMinMaxObserver,
                MovingAveragePerChannelMinMaxObserver,
            ),
        )
        if is_per_channel:
            raise AssertionError(
                "Per channel weight observer is not supported yet for ConvTranspose{n}d."
            )


def _obs_or_fq_ctr_equals(obs_or_fq1, obs_or_fq2):
    if isinstance(obs_or_fq1, type) and isinstance(obs_or_fq2, type):
        return obs_or_fq1 == obs_or_fq2
    if hasattr(obs_or_fq1, "p") and hasattr(obs_or_fq2, "p"):
        return _partial_wrapper_equals(obs_or_fq1, obs_or_fq2)
    return obs_or_fq1 == obs_or_fq2


def _partial_wrapper_equals(obs_or_fq1, obs_or_fq2):
    """Compare two ``with_args`` partial wrappers structurally."""
    obs_or_fq1_keywords = copy.copy(obs_or_fq1.p.keywords)
    obs_or_fq2_keywords = copy.copy(obs_or_fq2.p.keywords)
    keywords_equal = True
    if "observer" in obs_or_fq1_keywords and "observer" in obs_or_fq2_keywords:
        keywords_equal = keywords_equal and _obs_or_fq_ctr_equals(
            obs_or_fq1_keywords["observer"], obs_or_fq2_keywords["observer"]
        )
        obs_or_fq1_keywords.pop("observer")
        obs_or_fq2_keywords.pop("observer")
    keywords_equal = keywords_equal and obs_or_fq1_keywords == obs_or_fq2_keywords
    return (
        obs_or_fq1.p.func == obs_or_fq2.p.func
        and obs_or_fq1.p.args == obs_or_fq2.p.args
        and keywords_equal
    )


def qconfig_equals(q1, q2):
    """Return whether two qconfigs are structurally equal."""
    if q1 is None or q2 is None:
        return q1 == q2
    try:
        activation_same = _obs_or_fq_ctr_equals(q1.activation, q2.activation)
        weight_same = _obs_or_fq_ctr_equals(q1.weight, q2.weight)
        return activation_same and weight_same
    except AttributeError:
        return q1 == q2


def _activation_is_memoryless(qconfig):
    """Return whether the activation observer forgets history between calls."""

    def _is_memoryless(observer):
        return (
            hasattr(observer, "averaging_constant") and observer.averaging_constant == 1
        )

    act = qconfig.activation()
    if isinstance(act, FakeQuantizeBase) and hasattr(act, "activation_post_process"):
        return _is_memoryless(act.activation_post_process)
    else:
        return _is_memoryless(act)


def _is_reuse_input_qconfig(qconfig):
    return (
        qconfig is not None
        and isinstance(qconfig.activation(), ReuseInputObserver)
        and isinstance(qconfig.weight(), NoopObserver)
    )


def _add_module_to_qconfig_obs_ctr(qconfig, module=None):
    """Specialize the qconfig constructors to create observers on the module device."""
    if module is None or qconfig is None or qconfig._fields != ("activation", "weight"):
        return qconfig

    def get_factory_kwargs_based_on_module_device():
        if not isinstance(module, nn.Module):
            raise AssertionError("module must be an instance of nn.Module")
        devices = {p.device for p in module.parameters()} | {
            p.device for p in module.buffers()
        }
        device = next(iter(devices)) if len(devices) > 0 else None
        return None if device is None else {"device": device}

    def configure_constructor_to_put_obs_on_module_device(original_constructor):
        try:
            check = original_constructor.with_args(factory_kwargs=None)
            check()
            return original_constructor.with_callable_args(
                factory_kwargs=get_factory_kwargs_based_on_module_device
            )
        except AttributeError:
            return original_constructor
        except TypeError:
            return original_constructor

    activation = configure_constructor_to_put_obs_on_module_device(qconfig.activation)
    weight = configure_constructor_to_put_obs_on_module_device(qconfig.weight)
    return QConfig(activation, weight)


def get_default_qconfig_dict(backend="x86", version=0):
    """Return the default post-training qconfig mapping as a plain dictionary."""
    return QConfigMapping().set_global(get_default_qconfig(backend, version)).to_dict()


def get_default_qat_qconfig_dict(backend="x86", version=1):
    """Return the default QAT qconfig mapping as a plain dictionary."""
    return QConfigMapping().set_global(get_default_qat_qconfig(backend, version)).to_dict()


__all__ += [
    "QConfigDynamic",
    "default_debug_qconfig",
    "float16_static_qconfig",
    "default_activation_only_qconfig",
    "default_reuse_input_qconfig",
    "default_quint8_weight_qconfig",
    "default_embedding_qat_qconfig_4bit",
    "default_symmetric_qnnpack_qconfig",
    "default_per_channel_symmetric_qnnpack_qconfig",
    "default_symmetric_qnnpack_qat_qconfig",
    "default_per_channel_symmetric_qnnpack_qat_qconfig",
    "get_default_qconfig_dict",
    "get_default_qat_qconfig_dict",
    "QConfigAny",
    "qconfig_equals",
]
