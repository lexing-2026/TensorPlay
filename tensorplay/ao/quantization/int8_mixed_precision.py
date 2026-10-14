"""INT8 mixed-precision training built on native quantized ops.

During forward and backward, activations and weights are dynamically
quantized to int8 with row/column scales and the matmul runs through the
native signed integer linear path (int8 tensor cores on accelerators). The
result is rescaled to the original precision, giving INT8 tensor-core
throughput while keeping full-precision weights for the optimizer. The
straight-through estimator treats the quantizers as the identity so the
full-precision weights stay trainable.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import tensorplay as tp
from tensorplay import Tensor, nn, _C
from tensorplay.ao.nn.quantized.packed import PackedWeight
from tensorplay.autograd import Function
from tensorplay.overrides import redispatch_function

from tensorplay.ao.quantization.quantize_api import register_quantize_module_handler
from tensorplay.ao.quantization.bitnet import (
    _REWRAP_BITNET_OPS,
    _op_base_name,
    quantize_int8_rowwise,
)

__all__ = [
    "Int8MixedPrecisionTrainingConfig",
    "int8_mixed_precision_training",
    "Int8MixedPrecisionTrainingLinearWeight",
    "Int8MixedPrecisionTrainingLinear",
    "scaled_int8_mm",
]


@dataclass
class Int8MixedPrecisionTrainingConfig:
    """Configuration for INT8 mixed-precision training.

    Attributes:
        output: quantize activations and weights for the forward matmul.
        grad_input: quantize the backward matmul that produces ``grad_input``.
        grad_weight: quantize the backward matmul that produces ``grad_weight``.
        module_swap: replace ``nn.Linear`` modules with
            :class:`Int8MixedPrecisionTrainingLinear` instead of wrapping the
            weight in a tensor subclass.
    """

    output: bool = True
    grad_input: bool = True
    grad_weight: bool = True
    module_swap: bool = False


# Backwards-compatible alias matching the workflow config name.
int8_mixed_precision_training = Int8MixedPrecisionTrainingConfig


def _int8_mm(A_i8: tp.Tensor, B_t_i8: tp.Tensor, row_scale: tp.Tensor, col_scale: tp.Tensor) -> tp.Tensor:
    """Native int8 matmul: ``(A_i8 @ B_t_i8.T) * row_scale * col_scale``.

    ``A_i8`` is ``(M, K)`` row-major int8 and ``B_t_i8`` is ``(N, K)`` int8 (the
    transposed weight in output-major layout), so the product is ``(M, N)``.
    """
    if A_i8.device.type == "cuda":
        # Native int8 tensor-core path.
        out = tp.ops.tp._int_mm(A_i8, B_t_i8.t())
    else:
        # Native SIMD quantized linear path for CPU.
        x_q = _C._make_per_tensor_quantized_tensor(A_i8, 1.0, 0)
        w_q = _C._make_per_tensor_quantized_tensor(B_t_i8.contiguous(), 1.0, 0)
        packed_weight = PackedWeight(w_q, None)
        out = packed_weight.run(x_q, 1.0, 0, output_float=True)
    return out * col_scale.view(1, -1) * row_scale.view(-1, 1)


def scaled_int8_mm(A: tp.Tensor, B: tp.Tensor, row_scale: tp.Tensor, col_scale: tp.Tensor) -> tp.Tensor:
    """Native scaled int8 matmul: ``(A @ B) * row_scale * col_scale``.

    ``A`` is ``(M, K)`` int8, ``B`` is ``(K, N)`` int8, ``row_scale`` is ``(M,)``
    and ``col_scale`` is ``(N,)``. The integer product is computed through the
    native signed integer linear path (int8 tensor cores on CUDA, the SIMD
    quantized linear on CPU) and rescaled to float.
    """
    return _int8_mm(A, B.t(), row_scale, col_scale)


def _dynamic_int8_mm(A: tp.Tensor, B: tp.Tensor) -> tp.Tensor:
    """Dynamically quantize ``A`` and ``B`` and run an INT8 matmul.

    ``A`` may have any leading batch dims; ``B`` is exactly 2-D. The result is
    ``A @ B`` rescaled to the original precision.
    """
    A_i8, A_scale = quantize_int8_rowwise(A.reshape(-1, A.shape[-1]))
    B_t_i8, B_scale = quantize_int8_rowwise(B.T)
    out = _int8_mm(
        A_i8.contiguous(),
        B_t_i8.contiguous(),
        A_scale.contiguous(),
        B_scale.contiguous(),
    )
    return out.view(*A.shape[:-1], out.shape[-1])


class Int8MixedPrecisionTrainingLinearWeight(nn.Parameter):
    """Trainable INT8 mixed-precision weight parameter.

    The parameter's storage is the full-precision float weight. Any ``linear``
    call routes through :class:`_Int8MixedPrecisionTrainingLinearFunction`,
    which dynamically quantizes activations and weights to int8 for the
    forward (and optionally the backward) matmuls. Shape-preserving operators
    keep the wrapper; all other operators fall through to the native kernel on
    the shared storage, so in-place optimizer steps stay in sync with ``_data``.
    """

    def __init__(self, data: tp.Tensor, config: Optional[Int8MixedPrecisionTrainingConfig] = None):
        super().__init__(data, requires_grad=True)
        self.config = (
            config if config is not None else Int8MixedPrecisionTrainingConfig()
        )

    @property
    def data(self) -> tp.Tensor:
        return self._data

    @data.setter
    def data(self, value: tp.Tensor) -> None:
        # Keep the C++ storage and the Python-side float view in sync.  Device
        # and dtype transfers arrive as a re-wrapped weight, so unwrap it.
        Tensor.data.__set__(self, value)
        self._data = (
            value._data
            if isinstance(value, Int8MixedPrecisionTrainingLinearWeight)
            else value
        )

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        result = Int8MixedPrecisionTrainingLinearWeight(
            self._data.clone(), self.config
        )
        result.requires_grad = self.requires_grad
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        return (Int8MixedPrecisionTrainingLinearWeight, (self._data, self.config))

    def __tensor_flatten__(self):
        return ["_data"], [self.config]

    @classmethod
    def __tensor_unflatten__(
        cls, inner_tensors, tensor_meta, outer_size=None, outer_stride=None
    ):
        del outer_size, outer_stride
        return cls(inner_tensors["_data"], *tensor_meta)

    def fsdp_pre_all_gather(
        self,
        mesh,
        outer_size=None,
        outer_stride=None,
        module=None,
        mp_policy=None,
    ):
        data = self._data
        if mp_policy is not None and getattr(mp_policy, "param_dtype", None) is not None:
            data = data.to(mp_policy.param_dtype)
        return (data,), (self.config,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs,
        metadata,
        param_dtype,
        *,
        out: Optional["Int8MixedPrecisionTrainingLinearWeight"] = None,
    ):
        (data,) = all_gather_outputs
        (config,) = metadata
        if out is not None:
            if not isinstance(out, Int8MixedPrecisionTrainingLinearWeight):
                raise TypeError(
                    "out must be an Int8MixedPrecisionTrainingLinearWeight, got "
                    f"{type(out).__name__}"
                )
            return
        return Int8MixedPrecisionTrainingLinearWeight(data, config), all_gather_outputs

    @classmethod
    def __tensorplay_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        op_name = _op_base_name(func)
        if op_name == "linear":
            return _int8mp_linear_from_subclass(args, kwargs)
        if op_name in _REWRAP_BITNET_OPS:
            return _rewrap_int8mp_weight(func, args, kwargs)
        return redispatch_function(func, types, args, kwargs)


def _int8mp_linear_from_subclass(args, kwargs):
    input = args[0] if args else kwargs.get("input")
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    return _Int8MixedPrecisionTrainingLinearFunction.apply(
        input, weight, bias, weight.config
    )


def _rewrap_int8mp_weight(func, args, kwargs):
    self = args[0]
    inner = (self._data,) + args[1:]
    out = func(*inner, **kwargs)
    if isinstance(out, Tensor):
        return Int8MixedPrecisionTrainingLinearWeight(out.detach(), self.config)
    if isinstance(out, (tuple, list)):
        return type(out)(
            Int8MixedPrecisionTrainingLinearWeight(item.detach(), self.config)
            if isinstance(item, Tensor)
            else item
            for item in out
        )
    return out


class _Int8MixedPrecisionTrainingLinearFunction(Function):
    """Autograd function for INT8 mixed-precision training."""

    @staticmethod
    def forward(ctx, input, weight, bias, config):
        if isinstance(weight, Int8MixedPrecisionTrainingLinearWeight):
            config = weight.config
            weight = weight._data
        ctx.config = config
        ctx.save_for_backward(input, weight)
        ctx.bias = bias is not None
        weight = weight.to(input.dtype)
        if config.output:
            out = _dynamic_int8_mm(input, weight.T)
        else:
            out = input @ weight.T
        out = out + bias if bias is not None else out
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        weight = weight.to(input.dtype)
        config = ctx.config
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            if config.grad_input:
                grad_input = _dynamic_int8_mm(grad_output, weight)
            else:
                grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_output = grad_output.view(-1, weight.shape[0])
            input = input.view(-1, weight.shape[1])
            if config.grad_weight:
                grad_weight = _dynamic_int8_mm(input.T, grad_output).T
            else:
                grad_weight = grad_output.T @ input
        if ctx.needs_input_grad[2] and ctx.bias:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None


class Int8MixedPrecisionTrainingLinear(nn.Linear):
    """Linear module whose forward/backward use INT8 mixed-precision matmuls."""

    def __init__(self, *args, config: Optional[Int8MixedPrecisionTrainingConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = (
            config if config is not None else Int8MixedPrecisionTrainingConfig()
        )

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return _Int8MixedPrecisionTrainingLinearFunction.apply(
            input, self.weight, self.bias, self.config
        )


@register_quantize_module_handler(Int8MixedPrecisionTrainingConfig)
def _int8_mixed_precision_training_transform(module, config):
    """Convert an ``nn.Linear`` module to INT8 mixed-precision training."""
    if config.module_swap:
        module.__class__ = Int8MixedPrecisionTrainingLinear
        module.config = config
        return module
    module.weight = Int8MixedPrecisionTrainingLinearWeight(
        module.weight.detach(), config
    )
    return module