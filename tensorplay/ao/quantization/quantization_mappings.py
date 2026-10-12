"""Default float-to-quantized module swap tables for the eager workflow.

``convert`` walks a calibrated model and replaces each float module whose
class appears in :data:`DEFAULT_STATIC_QUANT_MODULE_MAPPINGS` with its
quantized counterpart, initialized through the target class's ``from_float``.
"""

import copy

import tensorplay.nn as nn
from tensorplay.ao.nn import intrinsic as nni
from tensorplay.ao.nn import qat as nnqat
from tensorplay.ao.nn import quantized as nnq
from tensorplay.ao.nn.intrinsic import quantized as nniq
from tensorplay.ao.nn.intrinsic import qat as nniqat
from tensorplay.ao.nn.quantized import dynamic as nnqd
from tensorplay.ao.quantization.stubs import DeQuantStub, QuantStub

__all__ = [
    "DEFAULT_QAT_MODULE_MAPPINGS",
    "DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS",
    "DEFAULT_STATIC_QUANT_MODULE_MAPPINGS",
    "get_default_dynamic_quant_module_mappings",
    "get_default_qat_module_mappings",
    "get_default_static_quant_module_mappings",
    "get_embedding_qat_module_mappings",
]

# Default map for swapping float modules to quantized ones
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS: dict = {
    QuantStub: nnq.Quantize,
    DeQuantStub: nnq.DeQuantize,
    nn.BatchNorm2d: nnq.BatchNorm2d,
    nn.BatchNorm3d: nnq.BatchNorm3d,
    nn.Conv1d: nnq.Conv1d,
    nn.Conv2d: nnq.Conv2d,
    nn.Conv3d: nnq.Conv3d,
    nn.ELU: nnq.ELU,
    nn.Hardswish: nnq.Hardswish,
    nn.Hardsigmoid: nnq.Hardsigmoid,
    nn.LeakyReLU: nnq.LeakyReLU,
    nn.Linear: nnq.Linear,
    nn.LayerNorm: nnq.LayerNorm,
    nn.GroupNorm: nnq.GroupNorm,
    nn.InstanceNorm1d: nnq.InstanceNorm1d,
    nn.InstanceNorm2d: nnq.InstanceNorm2d,
    nn.InstanceNorm3d: nnq.InstanceNorm3d,
    nn.ReLU: nnq.ReLU,
    nn.ReLU6: nnq.ReLU6,
    nn.Sigmoid: nnq.Sigmoid,
    nn.Tanh: nnq.Tanh,
    # wrapper modules
    nnq.FloatFunctional: nnq.QFunctional,
    # fused modules
    nni.BNReLU2d: nniq.BNReLU2d,
    nni.BNReLU3d: nniq.BNReLU3d,
    nni.ConvReLU1d: nniq.ConvReLU1d,
    nni.ConvReLU2d: nniq.ConvReLU2d,
    nni.ConvReLU3d: nniq.ConvReLU3d,
    nni.LinearReLU: nniq.LinearReLU,
    # qat modules
    nnqat.Linear: nnq.Linear,
    nnqat.Conv2d: nnq.Conv2d,
    nnqat.Conv3d: nnq.Conv3d,
}

# Default map for swapping float modules to qat modules
DEFAULT_QAT_MODULE_MAPPINGS: dict = {
    nn.Conv1d: nnqat.Conv1d,
    nn.Conv2d: nnqat.Conv2d,
    nn.Conv3d: nnqat.Conv3d,
    nn.Linear: nnqat.Linear,
    nni.ConvReLU1d: nniqat.ConvReLU1d,
    nni.ConvReLU2d: nniqat.ConvReLU2d,
    nni.ConvReLU3d: nniqat.ConvReLU3d,
    nni.LinearReLU: nniqat.LinearReLU,
    nni.LinearBn1d: nniqat.LinearBn1d,
}
for _rank in (1, 2, 3):
    for _relu in ('', 'ReLU'):
        _name = f'ConvBn{_relu}{_rank}d'
        DEFAULT_QAT_MODULE_MAPPINGS[getattr(nni, _name)] = getattr(nniqat, _name)
        DEFAULT_STATIC_QUANT_MODULE_MAPPINGS[getattr(nniqat, _name)] = getattr(nniq if _relu else nnq, f'ConvReLU{_rank}d' if _relu else f'Conv{_rank}d')
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS[getattr(nnqat, f'Conv{_rank}d')] = getattr(nnq, f'Conv{_rank}d')
    DEFAULT_STATIC_QUANT_MODULE_MAPPINGS[getattr(nniqat, f'ConvReLU{_rank}d')] = getattr(nniq, f'ConvReLU{_rank}d')
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS[nniqat.LinearReLU] = nniq.LinearReLU
DEFAULT_STATIC_QUANT_MODULE_MAPPINGS[nniqat.LinearBn1d] = nnq.Linear

# Default map for swapping dynamic modules
DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS: dict = {
    nn.Linear: nnqd.Linear,
    nnqat.dynamic.Linear: nnqd.Linear,
    nni.LinearReLU: nniq.dynamic.LinearReLU,
}


def get_default_static_quant_module_mappings() -> dict:
    """Get module mapping for post-training static quantization."""
    return copy.deepcopy(DEFAULT_STATIC_QUANT_MODULE_MAPPINGS)


def get_default_qat_module_mappings() -> dict:
    """Get module mapping for quantization-aware training."""
    return copy.deepcopy(DEFAULT_QAT_MODULE_MAPPINGS)


def get_embedding_qat_module_mappings() -> dict:
    mapping = get_default_qat_module_mappings()
    mapping[nn.Embedding] = nnqat.Embedding
    mapping[nn.EmbeddingBag] = nnqat.EmbeddingBag
    return mapping


def get_default_dynamic_quant_module_mappings() -> dict:
    """Get module mapping for post-training dynamic quantization."""
    return copy.deepcopy(DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS)
