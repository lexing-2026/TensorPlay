"""BitNet b1.58 style ternary quantization built on native quantized ops.

The weight is quantized to ternary codes in {-1, 0, 1} scaled by the tensor
absmean. Codes are stored packed at two bits per element (four per byte) using
the same bit layout as the reference BitNet packer, so the packed buffers are
interchangeable. Activations are quantized row-wise to signed 8-bit codes. The
matmul runs through the native signed integer linear path (int8 tensor cores on
accelerators) and is rescaled by the row and tensor scales.

Training uses a straight-through estimator: the forward executes the quantized
recipe exactly, while the backward passes gradients to the float weight and
activation as if the round/clamp steps were the identity (scaled so the
gradient magnitudes match the integer matmul, following the reference BitNet
training recipe). This makes ``BitNetLinear`` trainable with any optimizer.
"""

import math
from typing import Optional, Tuple

import tensorplay as tp
from tensorplay import Tensor, nn, _C
from tensorplay.ao.nn.quantized.packed import PackedWeight
from tensorplay.autograd import Function
from tensorplay.overrides import redispatch_function

__all__ = [
    "get_bitnet_scale",
    "quantize_bitnet_weight",
    "pack_ternary",
    "unpack_ternary",
    "quantize_int8_rowwise",
    "bitnet_linear",
    "bitnet_linear_packed",
    "BitNetLinear",
    "BitNetTrainingLinearWeight",
    "BitNetPacked2bitLinearWeight",
    "BitNetTrainingConfig",
    "bitnet_training",
    "quantize_",
    "precompute_bitnet_scale_for_fsdp",
]


def get_bitnet_scale(weight: tp.Tensor, eps: float = 1e-5) -> tp.Tensor:
    """Return the per-tensor absmean scale used by BitNet b1.58.

    The scale is ``mean(|W|)`` clamped away from zero so the reciprocal is
    finite for all-zero weights.
    """
    return weight.abs().mean().clamp(min=eps)


def quantize_bitnet_weight(
    weight: tp.Tensor,
    scale: Optional[tp.Tensor] = None,
    eps: float = 1e-5,
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Quantize a float weight to ternary codes and an absmean scale.

    Returns ``(codes, scale)`` where ``codes`` has values in {-1, 0, 1} stored
    as int8 and ``scale`` is a rank-0 float32 tensor.
    """
    scale = get_bitnet_scale(weight, eps) if scale is None else scale.clamp(min=eps)
    codes = (weight.float() / scale).round().clamp(-1, 1).to(tp.int8)
    return codes, scale


def pack_ternary(codes: tp.Tensor) -> tp.Tensor:
    """Pack int8 ternary codes {-1, 0, 1} into uint8 storage, four per byte.

    The last dimension is padded with zeros to a multiple of four. Each 2-bit
    field holds ``code & 0b11`` (``-1`` -> ``0b11``, ``0`` -> ``0``,
    ``1`` -> ``0b01``), with the first element of each group occupying the two
    most significant bits, matching the reference BitNet packer.
    """
    x = codes.to(tp.int8)
    columns = x.shape[-1]
    padded = columns
    if columns % 4:
        padded = columns + (4 - columns % 4)
        pad = padded - columns
        x = tp.cat([x, tp.zeros((*x.shape[:-1], pad), dtype=tp.int8, device=x.device)], dim=-1)
    x = x.to(tp.int32)
    high = (x[..., 0::4] & 0b11) << 6
    upper_mid = (x[..., 1::4] & 0b11) << 4
    lower_mid = (x[..., 2::4] & 0b11) << 2
    low = x[..., 3::4] & 0b11
    packed = (high | upper_mid | lower_mid | low).to(tp.uint8)
    return packed.reshape(*x.shape[:-1], (padded + 3) // 4)


def unpack_ternary(packed: tp.Tensor, columns: int) -> tp.Tensor:
    """Unpack uint8 2-bit fields back into int8 ternary codes {-1, 0, 1}.

    ``columns`` is the original number of logical elements along the packed
    dimension; trailing padded columns are dropped.
    """
    x = packed.to(tp.int8)
    fields = tp.stack([x >> 6, (x << 2) >> 6, (x << 4) >> 6, (x << 6) >> 6], dim=-1)
    codes = fields.reshape(*packed.shape[:-1], -1)
    if codes.shape[-1] > columns:
        codes = codes[..., :columns]
    return codes


def quantize_int8_rowwise(
    input: tp.Tensor, eps: float = 1e-5
) -> Tuple[tp.Tensor, tp.Tensor]:
    """Quantize the trailing dimension of ``input`` to int8 with row scales.

    For each row ``x`` the scale is ``max(|x|) / 127`` and the codes are
    ``round(x / scale)`` clamped to [-127, 127]. Returns ``(codes, scales)``
    with ``codes`` int8 and ``scales`` the per-row float32 scale.
    """
    absmax = input.abs().amax(dim=-1, keepdim=True).clamp(min=eps)
    row_scale = absmax / 127.0
    codes = (input.float() / row_scale).round().clamp(-127, 127).to(tp.int8)
    return codes, row_scale


class _BitNetLinearFunction(Function):
    """Straight-through estimator for the BitNet b1.58 linear.

    The forward follows :func:`bitnet_linear` exactly: row-wise int8
    activation codes, ternary weight codes, a native signed integer matmul,
    and rescaling by the row and tensor scales. The backward treats the
    quantizers as the identity:

        d loss / d input  = (grad_output @ codes) * tensor_scale
        d loss / d weight = grad_output.T @ (codes_x * row_scale)
        d loss / d bias   = grad_output.sum(0)

    The row and tensor scales cancel with the quantizer derivatives so the
    gradients have the same magnitude as the integer matmul, matching the
    reference BitNet training recipe.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, scale, eps):
        batch_dims = input.shape[:-1]
        x = input.reshape(-1, weight.shape[1])
        x_i8, row_scale = quantize_int8_rowwise(x, eps)
        data = weight._data if isinstance(weight, BitNetTrainingLinearWeight) else weight
        codes, tensor_scale = quantize_bitnet_weight(data, scale, eps)
        ctx.save_for_backward(x_i8, row_scale, codes, tensor_scale)
        if x.device.type == 'cuda':
            # Native int8 tensor-core path: no re-packing needed per step.
            out = tp.ops.tp._int_mm(x_i8, codes.t().contiguous())
        else:
            # Native SIMD quantized linear path for CPU.
            x_q = _C._make_per_tensor_quantized_tensor(x_i8, 1.0, 0)
            w_q = _C._make_per_tensor_quantized_tensor(codes.contiguous(), 1.0, 0)
            packed_weight = PackedWeight(w_q, None)
            out = packed_weight.run(x_q, 1.0, 0, output_float=True)
        out = out * row_scale * tensor_scale
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_dims, weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x_i8, row_scale, codes, tensor_scale = ctx.saved_tensors
        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.reshape(-1, codes.shape[0])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = (grad_output @ codes.to(grad_output.dtype)) * tensor_scale
            grad_input = grad_input.reshape(*batch_dims, codes.shape[1])
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.T @ (x_i8 * row_scale)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None, None


def bitnet_linear(
    input: tp.Tensor,
    weight: tp.Tensor,
    bias: Optional[tp.Tensor] = None,
    scale: Optional[tp.Tensor] = None,
    eps: float = 1e-5,
) -> tp.Tensor:
    """BitNet b1.58 linear: ternary weight times int8 row-wise activations.

    ``weight`` has shape ``(out_features, in_features)``. The forward computes

        y = round_clip(x / row_absmax * 127) @ round_clip(W / mean(|W|))
            * row_scale * weight_scale + bias

    using the native signed integer linear path for the matmul, and the
    straight-through estimator provides gradients for training. The returned
    tensor has the float dtype of ``input``.
    """
    return _BitNetLinearFunction.apply(input, weight, bias, scale, eps)


def bitnet_linear_packed(
    input: tp.Tensor,
    packed: tp.Tensor,
    scale: tp.Tensor,
    in_features: int,
    bias: Optional[tp.Tensor] = None,
    eps: float = 1e-5,
) -> tp.Tensor:
    """BitNet b1.58 linear from pre-packed ternary weights.

    ``packed`` is the output of :func:`pack_ternary` (uint8, four codes per
    byte), ``scale`` is the absmean tensor scale and ``in_features`` the number
    of input channels. This is the inference path: the weight quantization and
    packing happen once ahead of time.
    """
    batch_dims = input.shape[:-1]
    x = input.reshape(-1, in_features)
    x_i8, row_scale = quantize_int8_rowwise(x, eps)
    codes = unpack_ternary(packed, in_features)
    x_q = _C._make_per_tensor_quantized_tensor(x_i8, 1.0, 0)
    w_q = _C._make_per_tensor_quantized_tensor(codes.contiguous(), 1.0, 0)
    packed_weight = PackedWeight(w_q, None)
    out = packed_weight.run(x_q, 1.0, 0, output_float=True)
    out = out * row_scale * scale
    if bias is not None:
        out = out + bias
    return out.reshape(*batch_dims, packed.shape[0])


class BitNetLinear(nn.Module):
    """Linear layer with BitNet b1.58 ternary weights and int8 activations.

    The weight parameter stays in float; every forward quantizes it to ternary
    codes with the absmean scale and quantizes the activation row-wise to int8.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(tp.empty((out_features, in_features)))
        if bias:
            self.bias = nn.Parameter(tp.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: tp.Tensor) -> tp.Tensor:
        return bitnet_linear(input, self.weight, self.bias)


# ---------------------------------------------------------------------------
# BitNet training tensor subclass and model-level quantization API
# ---------------------------------------------------------------------------

_REWRAP_BITNET_OPS = frozenset({
    "t",
    "detach",
    "empty_like",
    "new_zeros",
    "slice",
    "view",
    "as_strided",
    "_to_copy",
    "_pin_memory",
    "split",
    "clone",
})


def _op_base_name(func) -> str:
    """Return the overload-free operator name for a dispatch hook."""
    name = getattr(func, "__name__", func)
    if isinstance(name, str):
        return name.split(".")[0]
    return str(func)


def _all_reduce_avg(tensor: tp.Tensor) -> None:
    """Average ``tensor`` across the default process group when initialized."""
    try:
        import tensorplay.distributed as dist
    except (ImportError, AttributeError):
        return
    available = getattr(dist, "is_available", lambda: False)()
    initialized = getattr(dist, "is_initialized", lambda: False)()
    if available and initialized:
        dist.all_reduce(tensor, op=dist.ReduceOp.AVG)


class BitNetTrainingLinearWeight(nn.Parameter):
    """Trainable BitNet b1.58 weight parameter.

    The parameter's storage is the float weight itself.  Any ``linear`` call
    that uses the tensor as ``weight`` is routed to the straight-through
    estimator (:class:`_BitNetLinearFunction`).  Shape-preserving operators
    keep the wrapper; every other operator falls through to the native kernel
    on the shared storage, so in-place optimizer steps stay visible through
    both the parameter and its ``_data`` view.

    The FSDP2 hooks quantize and pack the weight to two bits per element
    before all-gather and rebuild a :class:`BitNetPacked2bitLinearWeight` after
    it, so the communication volume is one quarter of the float weight.
    """

    def __init__(self, data: tp.Tensor, precomputed_scale: Optional[tp.Tensor] = None):
        super().__init__(data, requires_grad=True)
        self._precomputed_scale = precomputed_scale

    @property
    def data(self) -> tp.Tensor:
        return self._data

    @data.setter
    def data(self, value: tp.Tensor) -> None:
        # Keep the C++ storage and the Python-side float view in sync.  The
        # base descriptor replaces the parameter's impl (or copies metadata into
        # it), while ``_data`` tracks the same storage for quantization and
        # FSDP packing.  Device/dtype transfers arrive as a re-wrapped
        # BitNetTrainingLinearWeight (``Module._apply`` assigns ``param.data``),
        # so unwrap it before mirroring.
        Tensor.data.__set__(self, value)
        self._data = value._data if isinstance(value, BitNetTrainingLinearWeight) else value

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        result = BitNetTrainingLinearWeight(self._data.clone(), self._precomputed_scale)
        result.requires_grad = self.requires_grad
        memo[id(self)] = result
        return result

    def __reduce_ex__(self, proto):
        return (BitNetTrainingLinearWeight, (self._data, self._precomputed_scale))

    def __tensor_flatten__(self):
        if self._precomputed_scale is not None:
            return ["_data", "_precomputed_scale"], []
        return ["_data"], []

    @classmethod
    def __tensor_unflatten__(
        cls, inner_tensors, tensor_meta, outer_size=None, outer_stride=None
    ):
        del outer_size, outer_stride
        return cls(
            inner_tensors["_data"],
            inner_tensors.get("_precomputed_scale"),
        )

    def fsdp_pre_all_gather(self, mesh):
        scale = self._precomputed_scale
        if scale is None:
            scale = get_bitnet_scale(self._data)
            _all_reduce_avg(scale)
        codes, _ = quantize_bitnet_weight(self._data, scale)
        packed = pack_ternary(codes)
        return (packed,), (scale,)

    def fsdp_post_all_gather(
        self,
        all_gather_outputs,
        metadata,
        param_dtype,
        *,
        out: Optional["BitNetPacked2bitLinearWeight"] = None,
    ):
        (packed,) = all_gather_outputs
        (scale,) = metadata
        scale = scale.to(param_dtype)
        if out is not None:
            if not isinstance(out, BitNetPacked2bitLinearWeight):
                raise TypeError(
                    "out must be a BitNetPacked2bitLinearWeight, got "
                    f"{type(out).__name__}"
                )
            out.scale = scale
            return
        return BitNetPacked2bitLinearWeight(packed, scale), all_gather_outputs

    @classmethod
    def __tensorplay_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        op_name = _op_base_name(func)
        if op_name == "linear":
            return _bitnet_linear_from_subclass(args, kwargs)
        if op_name in _REWRAP_BITNET_OPS:
            return _rewrap_bitnet_weight(func, args, kwargs)
        return redispatch_function(func, types, args, kwargs)


def _bitnet_linear_from_subclass(args, kwargs):
    input = args[0] if args else kwargs.get("input")
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    return _BitNetLinearFunction.apply(input, weight, bias, None, 1e-5)


def _rewrap_bitnet_weight(func, args, kwargs):
    self = args[0]
    inner = (self._data,) + args[1:]
    out = func(*inner, **kwargs)
    if isinstance(out, Tensor):
        return BitNetTrainingLinearWeight(out.detach(), self._precomputed_scale)
    if isinstance(out, (tuple, list)):
        return type(out)(
            BitNetTrainingLinearWeight(item.detach(), self._precomputed_scale)
            if isinstance(item, Tensor)
            else item
            for item in out
        )
    return out


class BitNetPacked2bitLinearWeight(Tensor):
    """Packed BitNet b1.58 weight: four ternary codes per byte.

    The logical shape is ``(out_features, in_features)`` while ``int_data``
    holds the packed uint8 codes (``in_features / 4`` columns).  ``linear``
    dispatches to the packed straight-through estimator
    (:class:`_BitNetPacked2bitLinear`).
    """

    def __init__(self, int_data: tp.Tensor, scale: tp.Tensor):
        super().__init__()
        self.data = int_data
        self.int_data = int_data
        self.scale = scale

    @property
    def shape(self):
        return (self.int_data.shape[0], self.int_data.shape[1] * 4)

    @property
    def dtype(self):
        return self.scale.dtype

    @property
    def device(self):
        return self.int_data.device

    @property
    def ndim(self):
        return 2

    @property
    def is_cuda(self):
        return self.int_data.is_cuda

    def dequantize(self, out_dtype=None):
        """Unpack the ternary codes and rescale to a float tensor."""
        codes = unpack_ternary(self.int_data, self.int_data.shape[-1] * 4)
        out = codes.to(self.scale.dtype) * self.scale
        if out_dtype is not None:
            out = out.to(out_dtype)
        return out

    def __tensor_flatten__(self):
        return ["int_data", "scale"], []

    @classmethod
    def __tensor_unflatten__(
        cls, inner_tensors, tensor_meta, outer_size=None, outer_stride=None
    ):
        del outer_size, outer_stride
        return cls(inner_tensors["int_data"], inner_tensors["scale"])

    @classmethod
    def __tensorplay_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        op_name = _op_base_name(func)
        if op_name == "linear":
            return _bitnet_packed_linear(args, kwargs)
        if op_name == "as_strided":
            # FSDP2 views the all-gather buffer through as_strided; the
            # logical shape is fixed, so keep the packed payload untouched.
            self = args[0]
            return BitNetPacked2bitLinearWeight(self.int_data, self.scale)
        if op_name in ("detach", "clone"):
            return _rewrap_packed_weight(func, args, kwargs)
        return redispatch_function(func, types, args, kwargs)


def _bitnet_packed_linear(args, kwargs):
    input = args[0] if args else kwargs.get("input")
    weight = args[1] if len(args) > 1 else kwargs.get("weight")
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    return _BitNetPacked2bitLinear.apply(input, weight, bias, 1e-5)


def _rewrap_packed_weight(func, args, kwargs):
    self = args[0]
    inner = (self.int_data,) + args[1:]
    out = func(*inner, **kwargs)
    if isinstance(out, Tensor):
        return BitNetPacked2bitLinearWeight(out, self.scale)
    return out


class _BitNetPacked2bitLinear(Function):
    """Straight-through estimator for packed BitNet b1.58 weights.

    The forward unpacks the ternary codes, runs the native signed integer
    matmul and rescales.  The backward passes the activation gradient through
    the row-wise int8 quantizer; the packed codes are not differentiable.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, eps):
        in_features = weight.int_data.shape[-1] * 4
        batch_dims = input.shape[:-1]
        x = input.reshape(-1, in_features)
        x_i8, row_scale = quantize_int8_rowwise(x, eps)
        codes = unpack_ternary(weight.int_data, in_features)
        tensor_scale = weight.scale
        ctx.save_for_backward(x_i8, row_scale, codes, tensor_scale)
        if x.device.type == 'cuda':
            out = tp.ops.tp._int_mm(x_i8, codes.t().contiguous())
        else:
            x_q = _C._make_per_tensor_quantized_tensor(x_i8, 1.0, 0)
            w_q = _C._make_per_tensor_quantized_tensor(codes.contiguous(), 1.0, 0)
            packed_weight = PackedWeight(w_q, None)
            out = packed_weight.run(x_q, 1.0, 0, output_float=True)
        out = out * row_scale * tensor_scale
        if bias is not None:
            out = out + bias
        return out.reshape(*batch_dims, weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x_i8, row_scale, codes, tensor_scale = ctx.saved_tensors
        batch_dims = grad_output.shape[:-1]
        grad_output = grad_output.reshape(-1, codes.shape[0])
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = (grad_output @ codes.to(grad_output.dtype)) * tensor_scale
            grad_input = grad_input.reshape(*batch_dims, codes.shape[1])
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None


class BitNetTrainingConfig:
    """Configuration marker for :func:`quantize_` BitNet b1.58 conversion."""


# Backwards-compatible alias matching the upstream ``bitnet_training`` name.
bitnet_training = BitNetTrainingConfig


def _is_linear(module, name) -> bool:
    return isinstance(module, nn.Linear)


def quantize_(
    model: nn.Module,
    config: BitNetTrainingConfig,
    filter_fn=None,
    device=None,
) -> nn.Module:
    """Convert Linear weights in ``model`` to BitNet b1.58 in-place.

    Each matching ``nn.Linear`` keeps its module structure and bias, while
    ``weight`` is replaced by a :class:`BitNetTrainingLinearWeight` parameter
    whose ``linear`` forward runs the straight-through-estimator BitNet matmul.

    Args:
        model: the module tree to convert.
        config: a :class:`BitNetTrainingConfig` (``bitnet_training()``).
        filter_fn: optional ``(module, fqn) -> bool`` predicate; defaults to
            matching ``nn.Linear`` modules.
        device: optional device to move the model to before conversion.
    """
    if not isinstance(config, BitNetTrainingConfig):
        raise TypeError(
            "quantize_ expects a BitNetTrainingConfig, e.g. "
            "quantize_(model, bitnet_training())"
        )
    filter_fn = _is_linear if filter_fn is None else filter_fn
    if device is not None:
        model = model.to(device)
    for name, module in list(model.named_modules()):
        if filter_fn(module, name) and isinstance(module, nn.Linear):
            module.weight = BitNetTrainingLinearWeight(module.weight.detach())
    return model


def precompute_bitnet_scale_for_fsdp(module: nn.Module) -> None:
    """Compute the global absmean scale for BitNet parameters (FSDP).

    Run after the optimizer step.  A single all-reduce computes the average
    absmean across all BitNet weight shards and caches it on each parameter so
    :meth:`BitNetTrainingLinearWeight.fsdp_pre_all_gather` does not need to
    synchronize again.
    """
    bitnet_params = [
        p
        for p in module.parameters()
        if isinstance(p, BitNetTrainingLinearWeight)
    ]
    if not bitnet_params:
        return
    scales = [get_bitnet_scale(p._data) for p in bitnet_params]
    stacked = tp.stack(scales)
    _all_reduce_avg(stacked)
    for param, scale in zip(bitnet_params, stacked.unbind()):
        param._precomputed_scale = scale.detach()