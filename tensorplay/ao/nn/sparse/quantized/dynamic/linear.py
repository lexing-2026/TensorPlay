"""Block-sparse dynamically quantized linear module."""

import tensorplay as tp
from tensorplay import nn
from tensorplay.ao.nn.quantized.dynamic.modules.linear import Linear as _DynamicLinear
from tensorplay.ao.nn.quantized.packed import quantize_module_weight

__all__ = ["Linear"]


class Linear(_DynamicLinear):
    """A dynamically quantized sparse linear module.

    The sparse block shape is recorded but compute currently runs through the
    dense dynamic kernels.
    """

    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, row_block_size, col_block_size,
                 bias=True, dtype=tp.qint8):
        super().__init__(in_features, out_features, bias, dtype)
        self.row_block_size = int(row_block_size)
        self.col_block_size = int(col_block_size)

    def _get_name(self):
        return "SparseQuantizedDynamicLinear"

    def set_weight_bias(self, w, b, row_block_size=None, col_block_size=None):
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                f"row_block_size and col_block_size must not be None, got {row_block_size=}, {col_block_size=}"
            )
        self.row_block_size = int(row_block_size)
        self.col_block_size = int(col_block_size)
        super().set_weight_bias(w, b)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                "nnq." + cls.__name__ + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        from tensorplay.ao.quantization.qconfig import default_dynamic_qconfig

        weight_observer = mod.qconfig.weight() if mod.qconfig is not None and mod.qconfig.weight is not None else default_dynamic_qconfig.weight()
        weight = mod.weight
        if getattr(mod.qconfig, "mask", False):
            weight = mod.qconfig.mask * mod.weight
        weight_observer(weight)
        dtype = weight_observer.dtype
        if dtype != tp.qint8:
            raise AssertionError(f"Weight observer must have dtype qint8, got {dtype}")
        _w_sc, w_zp = weight_observer.calculate_qparams()
        if isinstance(w_zp, tp.Tensor):
            if bool(tp.any(w_zp.bool())):
                raise AssertionError("All weight zero points must map to 0")
        elif w_zp != 0:
            raise AssertionError(f"Weight zero point must map to 0, got {w_zp}")
        qweight = quantize_module_weight(mod, use_precomputed_fake_quant)
        from ..utils import LinearBlockSparsePattern

        row_block_size, col_block_size = LinearBlockSparsePattern.block_size()
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            row_block_size,
            col_block_size,
            dtype=dtype,
        )
        qlinear.set_weight_bias(qweight, mod.bias, row_block_size, col_block_size)
        return qlinear