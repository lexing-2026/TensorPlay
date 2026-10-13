"""INT8 weight-only quantized training built on native quantized ops.

The weight is quantized to int8 row-wise codes on the fly and ``linear``
routes through a straight-through estimator: the forward casts the int8
codes to the input dtype and rescales by the per-row scale, and the backward
passes a float gradient to the full-precision weight. The parameter stores the
full-precision weight (so any optimizer, including fused kernels, works), and
the int8 codes/scales are derived from the current weight on every forward.
"""

from typing import Optional, Tuple, Union

import tensorplay as tp
from tensorplay import Tensor, nn
from tensorplay.autograd import Function
from tensorplay.overrides import redispatch_function

from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler
from tensorplay.ao.quantization.bitnet import _op_base_name

__all__ = [
    "Int8WeightOnlyQuantizedTrainingConfig",
    "int8_weight_only_quantized_training",
    "Int8QuantizedTrainingLinearWeight",
]


def _quantize_int8_rowwise(
    tensor: tp.Tensor, stochastic_rounding: bool = False, eps: float = 1e-12
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Absmax symmetric quantization to int8 with per-row scales.

    Returns ``(int_data, scale)`` with ``int_data`` of shape ``(N, K)`` and
    ``scale`` of shape ``(N,)``. With ``stochastic_rounding`` each code rounds
    up with probability equal to the fractional part, so small weight updates
    are not systematically rounded away during training.
    """
    scale = tensor.abs().amax(dim=-1) / 127.0
    inv_scale = 1.0 / scale.float().clamp(min=eps)
    tensor = tensor.float() * inv_scale.view(-1, 1)
    if stochastic_rounding:
        tensor = (tensor + tp.rand_like(tensor)).floor()
    else:
        tensor = tensor.round()
    tensor = tensor.clamp(-128, 127).to(tp.int8)
    return tensor, scale


class Int8QuantizedTrainingLinearWeight(nn.Parameter):
    """Trainable INT8 weight-only parameter.

    The parameter's float storage is the full-precision weight, kept in sync by
    the optimizer through ordinary (possibly fused) kernels. ``linear``
    quantizes the current weight to int8 row-wise codes with a straight-through
    estimator, so the training numerics match INT8 weight-only quantization
    while remaining compatible with any optimizer.
    """

    def __init__(self, data: tp.Tensor):
        super().__init__(data, requires_grad=True)
        self._data = data

    @classmethod
    def from_float(cls, tensor: tp.Tensor) -> "Int8QuantizedTrainingLinearWeight":
        """Wrap a float weight into an INT8 weight-only training parameter."""
        return cls(tensor.detach())

    @property
    def int_data(self) -> tp.Tensor:
        """The int8 codes for the current weight."""
        int_data, _ = _quantize_int8_rowwise(self._data.detach())
        return int_data

    @property
    def scale(self) -> tp.Tensor:
        """The per-row scales for the current weight."""
        _, scale = _quantize_int8_rowwise(self._data.detach())
        return scale

    def dequantize(self) -> tp.Tensor:
        """Return the float tensor represented by the current int8 codes."""
        int_data, scale = _quantize_int8_rowwise(self._data.detach())
        return int_data.to(scale.dtype) * scale.view(-1, 1)

    @property
    def data(self) -> tp.Tensor:
        return self._data

    @data.setter
    def data(self, value: tp.Tensor) -> None:
        Tensor.data.__set__(self, value)
        if isinstance(value, Int8QuantizedTrainingLinearWeight):
            self._data = value._data
        else:
            self._data = value

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        result = Int8QuantizedTrainingLinearWeight(self._data.clone())
        result.requires_grad = self.requires_grad
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        return (Int8QuantizedTrainingLinearWeight, (self._data,))

    def __tensor_flatten__(self):
        return ["_data"], []

    @classmethod
    def __tensor_unflatten__(
        cls, inner_tensors, tensor_meta, outer_size=None, outer_stride=None
    ):
        del outer_size, outer_stride
        return cls(inner_tensors["_data"])

    def fsdp_pre_all_gather(
        self,
        mesh,
        outer_size=None,
        outer_stride=None,
        module=None,
        mp_policy=None,
    ):
        int_data, scale = _quantize_int8_rowwise(self._data.detach())
        if mp_policy is not None and getattr(mp_policy, "param_dtype", None) is not None:
            scale = scale.to(mp_policy.param_dtype)
        return (int_data, scale), None

    def fsdp_post_all_gather(
        self,
        all_gather_outputs,
        metadata,
        param_dtype,
        *,
        out: Optional["Int8QuantizedTrainingLinearWeight"] = None,
    ):
        int_data, scale = all_gather_outputs
        dequant = int_data.to(scale.dtype) * scale.view(-1, 1)
        if out is not None:
            if not isinstance(out, Int8QuantizedTrainingLinearWeight):
                raise TypeError(
                    "out must be an Int8QuantizedTrainingLinearWeight, got "
                    f"{type(out).__name__}"
                )
            Tensor.data.__set__(out, dequant)
            out._data = dequant
            return
        rebuilt = Int8QuantizedTrainingLinearWeight(dequant)
        rebuilt._fsdp_int_data = int_data
        rebuilt._fsdp_scale = scale
        return rebuilt, all_gather_outputs

    @classmethod
    def __tensorplay_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        op_name = _op_base_name(func)
        if op_name == "linear":
            return _int8_weight_only_linear(args, kwargs)
        if op_name == "_to_copy":
            return _int8_to_copy(args, kwargs)
        if op_name == "clone":
            return _int8_clone(func, args, kwargs)
        if op_name == "slice":
            return _int8_slice(func, args, kwargs)
        if op_name == "split":
            return _int8_split(func, args, kwargs)
        if op_name == "cat":
            return _int8_cat(func, args, kwargs)
        if op_name in ("view", "as_strided"):
            self = args[0]
            return Int8QuantizedTrainingLinearWeight(self._data)
        if op_name in ("zeros_like", "new_zeros"):
            return _int8_zeros_like(args, kwargs)
        if op_name in ("sub", "mul"):
            return _int8_sub_mul(func, args, kwargs)
        return redispatch_function(func, types, args, kwargs)


def _int8_weight_only_linear(args, kwargs):
    input = args[0] if args else kwargs.get("input")
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    return _Int8WeightOnlyLinear.apply(input, weight, bias)


def _int8_to_copy(args, kwargs):
    self = args[0]
    device = kwargs.get("device", None)
    dtype = kwargs.get("dtype", None)
    if device is None and dtype is None:
        return Int8QuantizedTrainingLinearWeight(self._data)
    return Int8QuantizedTrainingLinearWeight(
        self._data.to(device=device, dtype=dtype)
    )


def _int8_clone(func, args, kwargs):
    self = args[0]
    return Int8QuantizedTrainingLinearWeight(
        func(self._data, *args[1:], **kwargs)
    )


def _int8_slice(func, args, kwargs):
    self = args[0]
    return Int8QuantizedTrainingLinearWeight(
        func(self._data, *args[1:], **kwargs)
    )


def _int8_split(func, args, kwargs):
    self = args[0]
    return [
        Int8QuantizedTrainingLinearWeight(data)
        for data in func(self._data, *args[1:], **kwargs)
    ]


def _int8_cat(func, args, kwargs):
    tensors = args[0] if args else kwargs.get("tensors")
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    if dim != 0:
        raise NotImplementedError(
            "Int8QuantizedTrainingLinearWeight only supports cat at dim=0"
        )
    return Int8QuantizedTrainingLinearWeight(func([t._data for t in tensors], dim))


def _int8_zeros_like(args, kwargs):
    self = args[0]
    dtype = kwargs.get("dtype", self.dtype)
    device = kwargs.get("device", self.device)
    return tp.zeros(tuple(self.shape), dtype=dtype, device=device)


def _int8_sub_mul(func, args, kwargs):
    new_args = tuple(
        x.dequantize() if isinstance(x, Int8QuantizedTrainingLinearWeight) else x
        for x in args
    )
    return func(*new_args, **kwargs)


class _Int8WeightOnlyLinear(Function):
    """Straight-through estimator for INT8 weight-only training."""

    @staticmethod
    def forward(ctx, input, weight, bias):
        if isinstance(weight, Int8QuantizedTrainingLinearWeight):
            data = weight._data.detach()
        else:
            data = weight
        int_data, scale = _quantize_int8_rowwise(data)
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None
        out = (input @ int_data.T.to(input.dtype)) * scale
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        if isinstance(weight, Int8QuantizedTrainingLinearWeight):
            data = weight._data.detach()
        else:
            data = weight
        int_data, scale = _quantize_int8_rowwise(data)
        grad_input = (grad_output * scale) @ int_data.to(grad_output.dtype)
        grad_weight = grad_output.view(-1, weight.shape[0]).T @ input.view(
            -1, weight.shape[1]
        )
        grad_bias = grad_output.view(-1, weight.shape[0]).sum(0) if ctx.bias else None
        return grad_input, grad_weight, grad_bias


class Int8WeightOnlyQuantizedTrainingConfig:
    """Configuration marker for :func:`quantize_` INT8 weight-only training."""


# Backwards-compatible alias matching the upstream config name.
int8_weight_only_quantized_training = Int8WeightOnlyQuantizedTrainingConfig


@register_quantize_module_handler(Int8WeightOnlyQuantizedTrainingConfig)
def _int8_weight_only_training_transform(module, config):
    """Replace an ``nn.Linear`` weight with an INT8 weight-only training param."""
    del config  # Marker config without fields.
    module.weight = Int8QuantizedTrainingLinearWeight.from_float(module.weight)
    return module