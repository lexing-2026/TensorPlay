"""Base class and weight quantization helpers for reference quantized modules."""

import typing

import tensorplay as tp
from tensorplay import nn

__all__ = [
    "ReferenceQuantizedModule",
]


def _quantize_weight(
    weight: tp.Tensor,
    weight_qscheme,
    weight_dtype,
    weight_scale: tp.Tensor,
    weight_zero_point: tp.Tensor,
    weight_axis_int: int,
) -> tp.Tensor:
    """Quantize a float weight using the stored affine parameters."""
    if weight_dtype == tp.float16:
        return weight.to(weight_dtype)

    if weight_qscheme == tp.per_tensor_affine:
        if weight_dtype in [tp.quint8, tp.qint8, tp.qint32]:
            return tp.quantize_per_tensor(
                weight, weight_scale, weight_zero_point, weight_dtype
            )
    elif weight_qscheme in [
        tp.per_channel_affine,
        tp.per_channel_affine_float_qparams,
    ]:
        if weight_dtype in [tp.quint8, tp.qint8, tp.quint4x2, tp.qint32]:
            if weight_qscheme == tp.per_channel_affine:
                zp = weight_zero_point.to(tp.int64)
            else:
                zp = weight_zero_point
            return tp.quantize_per_channel(
                weight, weight_scale, zp, weight_axis_int, weight_dtype
            )
    raise ValueError(f"Unsupported dtype and qscheme: {weight_dtype}, {weight_qscheme}")


def _quantize_and_dequantize_weight(
    weight: tp.Tensor,
    weight_qscheme,
    weight_dtype,
    weight_scale: tp.Tensor,
    weight_zero_point: tp.Tensor,
    weight_axis_int: int,
) -> tp.Tensor:
    """Quantize and dequantize the weight to simulate quantized numerics."""
    if weight_qscheme in [
        tp.per_tensor_affine,
        tp.per_channel_affine,
        tp.per_channel_affine_float_qparams,
    ]:
        weight_quant = _quantize_weight(
            weight,
            weight_qscheme,
            weight_dtype,
            weight_scale,
            weight_zero_point,
            weight_axis_int,
        )
        return weight_quant.dequantize()
    return weight


def _save_weight_qparams(
    destination,
    prefix,
    weight_qscheme,
    weight_dtype,
    weight_scale,
    weight_zero_point,
    weight_axis,
):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == tp.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis


def _get_weight_qparam_keys(state_dict: dict[str, typing.Any], prefix: str):
    keys = ["weight_qscheme", "weight_dtype"]
    weight_qscheme = state_dict[prefix + "weight_qscheme"]
    if weight_qscheme is not None:
        keys.append("weight_scale")
        keys.append("weight_zero_point")
        if weight_qscheme == tp.per_channel_affine:
            keys.append("weight_axis")
    return keys


class ReferenceQuantizedModule(nn.Module):
    """A quantized module that keeps float weights and fake-quantizes them in
    forward, used as the lowering target of graph-mode quantization."""

    _IS_REFERENCE = True

    def _init_weight_qparams(self, weight_qparams, device):
        if weight_qparams is None:
            weight_qparams = {
                "qscheme": tp.per_tensor_affine,
                "dtype": tp.quint8,
                "scale": 1.0,
                "zero_point": 0,
            }

        self.weight_qscheme = weight_qparams["qscheme"]
        self.weight_dtype = weight_qparams["dtype"]
        if self.weight_qscheme not in [
            None,
            tp.per_tensor_affine,
            tp.per_channel_affine,
            tp.per_channel_affine_float_qparams,
        ]:
            raise AssertionError(
                f"qscheme: {self.weight_qscheme} is not supported in reference quantized {self._get_name()}"
            )
        if self.weight_dtype in [
            tp.quint8,
            tp.qint8,
            tp.quint4x2,
            tp.qint32,
        ]:
            zero_point_dtype = (
                weight_qparams["zero_point"].dtype
                if isinstance(weight_qparams["zero_point"], tp.Tensor)
                else tp.int32
            )
            w_scale = weight_qparams["scale"]
            w_scale_tensor = (
                w_scale.detach().clone()
                if isinstance(w_scale, tp.Tensor)
                else tp.tensor(w_scale, dtype=tp.float32, device=device)
            )
            self.register_buffer("weight_scale", w_scale_tensor)
            w_zp = weight_qparams["zero_point"]
            w_zp_tensor = (
                w_zp.detach().clone()
                if isinstance(w_zp, tp.Tensor)
                else tp.tensor(w_zp, dtype=zero_point_dtype, device=device)
            )
            self.register_buffer("weight_zero_point", w_zp_tensor)
            if self.weight_qscheme in [
                tp.per_channel_affine,
                tp.per_channel_affine_float_qparams,
            ]:
                w_axis = weight_qparams["axis"]
                w_axis_tensor = (
                    w_axis.detach().clone()
                    if isinstance(w_axis, tp.Tensor)
                    else tp.tensor(w_axis, dtype=tp.int32, device=device)
                )
                self.register_buffer("weight_axis", w_axis_tensor)
            else:
                self.register_buffer(
                    "weight_axis", tp.tensor(0, dtype=tp.int32, device=device)
                )
        else:
            self.register_buffer(
                "weight_scale", tp.tensor(1.0, dtype=tp.float32, device=device)
            )
            self.register_buffer(
                "weight_zero_point", tp.tensor(0, dtype=tp.int32, device=device)
            )
            self.register_buffer(
                "weight_axis", tp.tensor(0, dtype=tp.int32, device=device)
            )

        self.is_decomposed: bool = weight_qparams.get("is_decomposed", False)
        self.weight_axis_int: int = int(self.weight_axis.item())  # type: ignore[operator, assignment]
        self.weight_quant_min: int | None = weight_qparams.get("quant_min")
        self.weight_quant_max: int | None = weight_qparams.get("quant_max")

    def get_weight(self):
        """Return the fake-quantized (quantize-dequantize) weight tensor."""
        if self.is_decomposed:
            raise NotImplementedError("Decomposed reference weights are not supported")
        return _quantize_and_dequantize_weight(
            self.weight,
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_axis_int,
        )

    def get_quantized_weight(self):
        """Return the quantized weight tensor with its affine parameters."""
        if self.is_decomposed:
            raise NotImplementedError("Decomposed reference weights are not supported")
        return _quantize_weight(
            self.weight,
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_axis_int,
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        _save_weight_qparams(
            destination,
            prefix,
            self.weight_qscheme,
            self.weight_dtype,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_axis,
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for key in _get_weight_qparam_keys(state_dict, prefix):
            setattr(self, key, state_dict[prefix + key])
            state_dict.pop(prefix + key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            False,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )