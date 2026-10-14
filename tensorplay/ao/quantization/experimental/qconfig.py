"""Experimental qconfigs for APoT quantization."""

import tensorplay as tp

from ..observer import MinMaxObserver
from ..fake_quantize import FakeQuantize
from ..qconfig import QConfig

from .fake_quantize import APoTFakeQuantize

default_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=tp.per_tensor_symmetric, dtype=tp.quint8
)
default_weight_symmetric_fake_quant = FakeQuantize.with_args(
    observer=MinMaxObserver, qscheme=tp.per_tensor_symmetric, dtype=tp.qint8
)

uniform_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant,
    weight=default_weight_symmetric_fake_quant,
)
apot_weight_qconfig_8bit = QConfig(
    activation=default_symmetric_fake_quant,
    weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=tp.qint8),
)
apot_qconfig_8bit = QConfig(
    activation=APoTFakeQuantize.with_args(b=8, k=2, dtype=tp.quint8),
    weight=APoTFakeQuantize.with_args(b=8, k=2, dtype=tp.qint8),
)
uniform_qconfig_4bit = QConfig(
    activation=default_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
    weight=default_weight_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
)
apot_weight_qconfig_4bit = QConfig(
    activation=default_symmetric_fake_quant.with_args(quant_min=0, quant_max=15),
    weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=tp.qint8),
)
apot_qconfig_4bit = QConfig(
    activation=APoTFakeQuantize.with_args(b=4, k=2, dtype=tp.quint8),
    weight=APoTFakeQuantize.with_args(b=4, k=2, dtype=tp.qint8),
)