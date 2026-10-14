"""Block-sparse quantized linear modules.

The native sparse packing kernels are not yet available, so these modules
preserve the block-sparse API and metadata while computing through the
dense packed integer kernels.
"""

import tensorplay as tp
from tensorplay import nn
from tensorplay.ao.nn.quantized.packed import PackedWeight, quantize_module_weight
from tensorplay.ao.nn.quantized.linear import QuantizedLinear

__all__ = ["LinearPackedParams", "Linear"]


class LinearPackedParams(PackedWeight):
    """Packed weight storage carrying a block-sparse shape."""

    def __init__(self, row_block_size=1, col_block_size=4, dtype=tp.qint8):
        if dtype != tp.qint8:
            raise NotImplementedError("Linear prepacking only supports QINT8")
        self.row_block_size = int(row_block_size)
        self.col_block_size = int(col_block_size)
        self.dtype = dtype
        wq = tp._C.quantize_per_tensor(tp.zeros(1, 1), 1.0, 0, dtype)
        super().__init__(wq, None)

    def _get_name(self):
        return "SparseQuantizedLinearPackedParams"

    def set_weight_bias(self, weight, bias, row_block_size=None, col_block_size=None):
        if row_block_size is not None:
            self.row_block_size = int(row_block_size)
        if col_block_size is not None:
            self.col_block_size = int(col_block_size)
        super().set_weight_bias(weight, bias)


class Linear(QuantizedLinear):
    """A quantized sparse linear module."""

    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features, out_features, row_block_size, col_block_size,
                 bias=True, dtype=tp.qint8):
        if dtype != tp.qint8:
            raise NotImplementedError("Only QINT8 is supported for Sparse Quantized Linear")
        bias_tensor = tp.zeros(out_features, dtype=tp.float32) if bias else None
        super().__init__(in_features, out_features, bias=bias_tensor)
        self.row_block_size = int(row_block_size)
        self.col_block_size = int(col_block_size)

    def _get_name(self):
        return "SparseQuantizedLinear"

    def set_weight_bias(self, w, b, row_block_size=None, col_block_size=None):
        if row_block_size is None or col_block_size is None:
            raise AssertionError(
                "row_block_size and col_block_size must not be None, "
                f"got row_block_size={row_block_size}, col_block_size={col_block_size}"
            )
        self.row_block_size = int(row_block_size)
        self.col_block_size = int(col_block_size)
        super().set_weight_bias(w, b)

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        if type(mod) is not cls._FLOAT_MODULE:
            raise AssertionError(
                cls._get_name()
                + ".from_float only works for "
                + cls._FLOAT_MODULE.__name__
            )
        if not hasattr(mod, "sparse_params"):
            raise AssertionError(
                "Expecting the Linear to have `sparse_params`. Make sure you have provided arguments "
                'in the `sparsifier.squash_mask(params_to_save=("sparse_block_shape",))` method.'
            )
        sparse_block_shape = mod.sparse_params.get("sparse_block_shape", None)
        if not isinstance(sparse_block_shape, (tuple, list)) or len(sparse_block_shape) != 2:
            raise AssertionError(
                f"sparse_block_shape must be a 2-tuple, got {sparse_block_shape}"
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        activation_post_process = mod.activation_post_process
        weight_post_process = mod.qconfig.weight()
        weight_post_process(mod.weight)
        dtype = weight_post_process.dtype
        act_scale, act_zp = activation_post_process.calculate_qparams()
        if dtype != tp.qint8:
            raise AssertionError(f"Weight observer must have dtype qint8, got {dtype}")
        w_sc, w_zp = weight_post_process.calculate_qparams()
        if isinstance(w_zp, tp.Tensor):
            if bool(tp.any(w_zp.bool())):
                raise AssertionError("All weight zero points must map to 0")
        elif w_zp != 0:
            raise AssertionError(f"Weight zero point must map to 0, got {w_zp}")
        qweight = quantize_module_weight(mod, use_precomputed_fake_quant)
        row_block_size = sparse_block_shape[0]
        col_block_size = sparse_block_shape[1]
        qlinear = cls(
            mod.in_features,
            mod.out_features,
            row_block_size,
            col_block_size,
            dtype=dtype,
        )
        qlinear.set_weight_bias(qweight, mod.bias, row_block_size, col_block_size)
        qlinear.out_scale = float(act_scale)
        qlinear.out_zero_point = int(act_zp)
        return qlinear