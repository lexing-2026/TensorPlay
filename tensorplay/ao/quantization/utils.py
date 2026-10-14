"""Shared utilities for the eager and graph quantization workflows."""

import functools
import warnings
from collections import OrderedDict
from collections.abc import Callable
from inspect import getfullargspec, signature
from typing import Any
from typing_extensions import TypeAliasType

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn import functional as F
from tensorplay.graph import Node
from tensorplay.nn.utils.parametrize import is_parametrized

from .quant_type import QuantType

__all__ = [
    "NodePattern",
    "Pattern",
    "MatchAllNode",
    "get_combined_dict",
    "is_per_tensor",
    "is_per_channel",
    "getattr_from_fqn",
    "get_qparam_dict",
    "activation_dtype",
    "weight_dtype",
    "activation_is_statically_quantized",
    "activation_is_dynamically_quantized",
    "activation_is_int8_quantized",
    "activation_is_int32_quantized",
    "weight_is_quantized",
    "weight_is_statically_quantized",
    "op_is_int8_dynamically_quantized",
    "get_qconfig_dtypes",
    "get_quant_type",
    "check_min_max_valid",
    "calculate_qmin_qmax",
    "has_no_children_ignoring_parametrizations",
    "to_underlying_dtype",
    "determine_qparams",
    "validate_qmin_qmax",
]

NodePattern = TypeAliasType(
    "NodePattern", tuple[Node, Node] | tuple[Node, tuple[Node, Node]] | Any
)

# Graph quantizer instance; defined separately to avoid circular imports.
QuantizerCls = Any

Pattern = TypeAliasType(
    "Pattern",
    Callable
    | tuple[Callable, Callable]
    | tuple[Callable, tuple[Callable, Callable]]
    | Any,
)


class MatchAllNode:
    """Node pattern that matches any node, used in fusion patterns."""


module_type_list = {
    nn.ReLU,
    nn.ReLU6,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.Identity,
    nn.Hardsigmoid,
    nn.Sigmoid,
    nn.Tanh,
}
func_list = {
    F.adaptive_avg_pool1d,
    F.adaptive_avg_pool2d,
    F.adaptive_avg_pool3d,
    F.elu,
    F.hardswish,
    F.instance_norm,
    F.layer_norm,
    F.leaky_relu,
    F.silu,
    F.mish,
    F.dropout,
    F.max_pool1d,
    F.max_pool2d,
    F.max_pool3d,
    F.relu,
    F.hardtanh,
    F.hardtanh_,
    F.hardsigmoid,
    F.sigmoid,
    tp.transpose,
    tp.repeat_interleave,
    tp.sigmoid,
    tp.squeeze,
    tp.stack,
    tp.sum,
    tp.tanh,
    tp.unsqueeze,
    tp.cat,
}
method_list = {
    tp.mean,
    "relu",
    "relu_",
    "contiguous",
    "detach",
    "detach_",
    "hardsigmoid",
    "hardsigmoid_",
    "permute",
    "repeat",
    "repeat_interleave",
    "reshape",
    "resize_",
    "shape",
    "sigmoid",
    "sigmoid_",
    "size",
    "squeeze",
    "squeeze_",
    "tanh",
    "tanh_",
    "transpose",
    "unsqueeze",
    "unsqueeze_",
    "view",
}

def check_node(node, modules):
    is_call_function = node.op == "call_function" and node.target in func_list
    is_call_method = node.op == "call_method" and node.target in method_list
    is_call_module = (
        node.op == "call_module" and type(modules[str(node.target)]) in module_type_list
    )
    return is_call_function, is_call_method, is_call_module


def get_combined_dict(default_dict, additional_dict):
    """Merge two dictionaries, with ``additional_dict`` overriding defaults."""
    d = default_dict.copy()
    d.update(additional_dict)
    return d


def is_per_tensor(qscheme):
    return qscheme == tp.per_tensor_affine or qscheme == tp.per_tensor_symmetric


def is_per_channel(qscheme):
    return qscheme in [
        tp.per_channel_affine,
        tp.per_channel_affine_float_qparams,
        tp.per_channel_symmetric,
    ]


def getattr_from_fqn(obj: Any, fqn: str) -> Any:
    """Resolve a dotted path such as ``"foo.bar.baz"`` against ``obj``."""
    return functools.reduce(getattr, fqn.split("."), obj)


def to_underlying_dtype(qdtype):
    DTYPE_MAPPING = {
        tp.quint8: tp.uint8,
        tp.qint8: tp.int8,
        tp.qint32: tp.int32,
        tp.quint4x2: tp.uint8,
        tp.quint2x4: tp.uint8,
        tp.uint8: tp.uint8,
        tp.int8: tp.int8,
        tp.uint16: tp.uint16,
        tp.int16: tp.int16,
        tp.int32: tp.int32,
        tp.float8_e5m2: tp.float8_e5m2,
        tp.float8_e4m3fn: tp.float8_e4m3fn,
    }
    if qdtype not in DTYPE_MAPPING:
        raise AssertionError("Unsupported dtype: " + str(qdtype))
    return DTYPE_MAPPING[qdtype]


def get_qparam_dict(observer_or_fake_quant):
    from .observer import PlaceholderObserver

    qscheme = getattr(observer_or_fake_quant, "qscheme", None)
    dtype = observer_or_fake_quant.dtype
    qparams = {"qscheme": qscheme, "dtype": dtype}

    if not qscheme or isinstance(observer_or_fake_quant, PlaceholderObserver):
        return {"qscheme": None, "dtype": dtype}

    if is_per_tensor(qscheme):
        qscheme = tp.per_tensor_affine
    elif is_per_channel(qscheme):
        if qscheme == tp.per_channel_symmetric:
            qscheme = tp.per_channel_affine
        qparams["axis"] = observer_or_fake_quant.ch_axis
    else:
        raise RuntimeError(f"Unrecognized qscheme: {qscheme}")
    qparams["qscheme"] = qscheme

    scale, zero_point = observer_or_fake_quant.calculate_qparams()
    qparams["scale"] = scale
    qparams["zero_point"] = zero_point

    if hasattr(observer_or_fake_quant, "quant_min"):
        qparams["quant_min"] = observer_or_fake_quant.quant_min
    if hasattr(observer_or_fake_quant, "quant_max"):
        qparams["quant_max"] = observer_or_fake_quant.quant_max

    return qparams


def get_swapped_custom_module_class(
    custom_module, custom_module_class_mapping, qconfig
):
    """Return the observed/quantized custom module class for ``custom_module``."""
    quant_type = get_quant_type(qconfig)
    class_mapping = custom_module_class_mapping.get(quant_type, {})
    if type(custom_module) not in class_mapping:
        raise AssertionError(
            "did not find corresponding observed "
            f"module class for {type(custom_module)} in mapping: {class_mapping}"
        )
    return class_mapping[type(custom_module)]


def activation_dtype(qconfig):
    if qconfig is None:
        raise AssertionError("qconfig must be provided to determine activation dtype")
    activation = qconfig.activation()
    return activation.dtype


def weight_dtype(qconfig):
    if qconfig is None:
        raise AssertionError("qconfig must be provided to determine weight dtype")
    weight = qconfig.weight()
    return weight.dtype


def activation_is_statically_quantized(qconfig):
    """Return whether the activation is statically quantized."""
    return activation_dtype(qconfig) in [
        tp.quint8,
        tp.qint8,
        tp.qint32,
        tp.float16,
        tp.uint8,
        tp.int8,
        tp.int16,
        tp.int32,
        tp.float8_e5m2,
        tp.float8_e4m3fn,
    ] and (not activation_is_dynamically_quantized(qconfig))


def activation_is_dynamically_quantized(qconfig):
    _activation_dtype, _, activation_is_dynamic = get_qconfig_dtypes(qconfig)
    return activation_is_dynamic


def activation_is_int8_quantized(qconfig):
    return activation_dtype(qconfig) in [
        tp.quint8,
        tp.qint8,
        tp.uint8,
        tp.int8,
    ]


def activation_is_int32_quantized(qconfig):
    return activation_dtype(qconfig) in [tp.qint32, tp.int32]


def weight_is_quantized(qconfig):
    return weight_dtype(qconfig) in [
        tp.quint8,
        tp.qint8,
        tp.float16,
        tp.quint4x2,
        tp.uint8,
        tp.int8,
        tp.int16,
        tp.int32,
        tp.float8_e5m2,
        tp.float8_e4m3fn,
    ]


def weight_is_statically_quantized(qconfig):
    return weight_dtype(qconfig) in [tp.quint8, tp.qint8, tp.uint8, tp.int8]


def op_is_int8_dynamically_quantized(qconfig) -> bool:
    activation_dtype, weight_dtype, activation_is_dynamic = get_qconfig_dtypes(qconfig)
    return (
        activation_dtype in [tp.quint8, tp.uint8]
        and weight_dtype in [tp.qint8, tp.int8]
        and activation_is_dynamic
    )


def get_qconfig_dtypes(qconfig):
    """Return ``(activation_dtype, weight_dtype, activation_is_dynamic)``."""
    if qconfig is None:
        raise AssertionError("qconfig must be provided to extract dtypes")
    activation = qconfig.activation()
    weight = qconfig.weight()
    act_is_dynamic = getattr(activation, "is_dynamic", False)
    return (activation.dtype, weight.dtype, act_is_dynamic)


def get_quant_type(qconfig):
    """Map a qconfig to its :class:`QuantType` family."""
    if qconfig is None:
        raise AssertionError("qconfig must be provided to determine quant type")
    activation = qconfig.activation()
    weight = qconfig.weight()
    static_dtypes = [
        tp.quint8,
        tp.qint8,
        tp.quint4x2,
        tp.qint32,
        tp.uint8,
        tp.int8,
        tp.int16,
        tp.int32,
        tp.float8_e5m2,
        tp.float8_e4m3fn,
    ]
    if weight.dtype in static_dtypes:
        if hasattr(activation, "is_dynamic") and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype in static_dtypes:
            return QuantType.STATIC
        else:
            return QuantType.WEIGHT_ONLY

    if weight.dtype == tp.float16:
        if hasattr(activation, "is_dynamic") and activation.is_dynamic:
            return QuantType.DYNAMIC
        elif activation.dtype == tp.float16:
            return QuantType.STATIC

    raise Exception(  # noqa: TRY002
        f"Unrecognized dtype combination in get_quant_type: activation({activation.dtype}),"
        f"weight({weight.dtype})"
    )


def check_min_max_valid(min_val: tp.Tensor, max_val: tp.Tensor) -> bool:
    """Check that min/max are populated and min is less than max."""
    if min_val.numel() == 0 or max_val.numel() == 0:
        warnings.warn(
            "must run observer before calling calculate_qparams. "
            + "Returning default values.",
            stacklevel=2,
        )
        return False

    if min_val.dim() == 0 or max_val.dim() == 0:
        if min_val == float("inf") and max_val == float("-inf"):
            warnings.warn(
                "must run observer before calling calculate_qparams. "
                + "Returning default values.",
                stacklevel=2,
            )
            return False

        if min_val > max_val:
            raise AssertionError(f"min {min_val} should be less than max {max_val}")
    else:
        if tp.any(min_val > max_val):
            raise AssertionError(f"min {min_val} should be less than max {max_val}")

    return True


def calculate_qmin_qmax(
    quant_min: int,
    quant_max: int,
    has_customized_qrange: bool,
    dtype: tp.dtype,
    reduce_range: bool,
) -> tuple[int, int]:
    """Compute the effective qmin/qmax for an observer dtype and range mode."""
    if has_customized_qrange:
        if dtype in [tp.qint32, tp.int32]:
            initial_quant_min, initial_quant_max = 0, 2**32 - 1
        else:
            initial_quant_min, initial_quant_max = 0, 255
        custom_quant_min, custom_quant_max = quant_min, quant_max
        if custom_quant_min is not None and custom_quant_max is not None:
            initial_quant_min, initial_quant_max = (
                custom_quant_min,
                custom_quant_max,
            )

        qrange_len = initial_quant_max - initial_quant_min + 1
        if dtype in [tp.qint8, tp.int8]:
            if not (0 < qrange_len <= 256):
                raise AssertionError(
                    "quantization range should be positive and not exceed the maximum bit range (=256)."
                )
        elif dtype in [tp.qint32, tp.int32]:
            if not (0 < qrange_len <= 2**32):
                raise AssertionError(
                    "quantization range should be positive and not exceed the maximum bit range (=4294967296)."
                )
        if reduce_range:
            quant_min, quant_max = quant_min // 2, quant_max // 2
    else:
        if dtype in [tp.qint8, tp.int8]:
            if reduce_range:
                quant_min, quant_max = -64, 63
            else:
                quant_min, quant_max = -128, 127
        elif dtype in [tp.quint8, tp.uint8]:
            if reduce_range:
                quant_min, quant_max = 0, 127
            else:
                quant_min, quant_max = 0, 255
        elif dtype in [tp.qint32, tp.int32]:
            quant_min, quant_max = -1 * (2**31), (2**31) - 1
        elif dtype == tp.uint16:
            quant_min, quant_max = 0, 2**16 - 1
        elif dtype == tp.int16:
            quant_min, quant_max = -(2**15), 2**15 - 1
        else:
            quant_min, quant_max = 0, 15
    return quant_min, quant_max


def _parent_name(target):
    """Split ``'foo.bar'`` into ``('foo', 'bar')``."""
    r = target.rsplit(".", 1)
    if len(r) == 1:
        return "", r[0]
    else:
        return r[0], r[1]


def has_no_children_ignoring_parametrizations(module):
    """Check that a module has no children, ignoring parametrization bookkeeping."""
    if len(module._modules) == 0:
        return True
    elif is_parametrized(module):
        return len(module._modules) == 1 and "parametrizations" in module._modules
    else:
        return False


def _get_path_of_module(
    root: nn.Module, submodule: nn.Module
) -> str | None:
    """Return the fully qualified name of ``submodule`` inside ``root``."""
    for n, p in root.named_modules():
        if submodule is p:
            return n
    return None


def _get_signature_locals(f: Callable, loc: dict[str, Any]) -> dict[str, Any]:
    """Return the local keyword arguments that appear in ``f``'s signature."""
    return {k: v for k, v in loc.items() if k in signature(f).parameters}


def _get_default_kwargs(f: Callable) -> "OrderedDict[str, Any]":
    """Return all default keyword arguments from a function signature."""
    kwargs = {}
    for name, param in signature(f).parameters.items():
        if param.default is not param.empty:
            kwargs[name] = param.default
        elif param.kind is param.VAR_POSITIONAL:
            kwargs[name] = ()
        elif param.kind is param.VAR_KEYWORD:
            kwargs[name] = {}
    return OrderedDict(kwargs)


def _normalize_kwargs(func: Callable, loc: dict[str, Any]) -> "OrderedDict[str, Any]":
    """Fill in default arguments from a function signature for local kwargs."""
    default_kwargs = _get_default_kwargs(func)
    local_kwargs = _get_signature_locals(func, loc)
    normalized_kwargs = default_kwargs.copy()
    for attr, val in local_kwargs.items():
        if attr in normalized_kwargs:
            normalized_kwargs[attr] = val
    return normalized_kwargs


def validate_qmin_qmax(quant_min: int, quant_max: int) -> None:
    """Validate a user-specified quantization range."""
    if not (quant_min <= 0 <= quant_max):
        raise AssertionError("Used-specified quantization range must include 0.")
    if quant_min >= quant_max:
        raise AssertionError(
            "qmin must be strictly less than qmax for user-specified quantization range."
        )


def determine_qparams(
    min_val: tp.Tensor,
    max_val: tp.Tensor,
    quant_min: int,
    quant_max: int,
    dtype: tp.dtype,
    eps: tp.Tensor,
    has_customized_qrange: bool,
    qscheme: tp.qscheme = tp.per_tensor_affine,
) -> tuple[tp.Tensor, tp.Tensor]:
    """Compute scale and zero point from min/max, per tensor or per channel."""
    if not check_min_max_valid(min_val, max_val):
        return tp.tensor([1.0], device=min_val.device.type), tp.tensor(
            [0], device=min_val.device.type
        )

    min_val_neg = tp.min(min_val, tp.zeros_like(min_val))
    max_val_pos = tp.max(max_val, tp.zeros_like(max_val))

    device = min_val_neg.device
    scale = tp.ones(min_val_neg.size(), dtype=tp.float64, device=device)
    zero_point = tp.zeros(min_val_neg.size(), dtype=tp.int64, device=device)
    eps = eps.to(device)

    if qscheme == tp.per_tensor_symmetric or qscheme == tp.per_channel_symmetric:
        max_val_pos = tp.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        scale = tp.max(scale, eps)
        if dtype in [tp.uint8, tp.quint8]:
            if has_customized_qrange:
                zero_point = zero_point.new_full(
                    zero_point.size(), (quant_min + quant_max) // 2
                )
            else:
                zero_point = zero_point.new_full(zero_point.size(), 128)
    elif qscheme == tp.per_channel_affine_float_qparams:
        scale = (max_val - min_val) / float(quant_max - quant_min)
        scale = tp.where(scale > eps, scale, tp.ones_like(scale))
        zero_point = -1 * min_val / scale
    else:
        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = tp.max(scale, eps)
        zero_point = quant_min - tp.round(min_val_neg / scale).to(tp.int64)
        zero_point = tp.clamp(zero_point, quant_min, quant_max)

    if len(scale.shape) == 0:
        scale = tp.tensor([float(scale)], dtype=scale.dtype, device=device)
    if len(zero_point.shape) == 0:
        zero_point = tp.tensor(
            [int(zero_point)], dtype=zero_point.dtype, device=device
        )
        if qscheme == tp.per_channel_affine_float_qparams:
            zero_point = tp.tensor(
                [float(zero_point)], dtype=zero_point.dtype, device=device
            )

    return scale.to(tp.float64), zero_point.to(tp.int64)


def _get_num_pos_args(f: Callable) -> int:
    """Return the number of positional arguments of ``f``."""
    return len(getfullargspec(f).args)


def get_fqn_to_example_inputs(
    model: nn.Module, example_inputs: tuple[Any, ...]
) -> dict[str, tuple[Any, ...]]:
    """Map fully qualified submodule names to the example inputs they saw.

    Works by temporarily intercepting module ``__call__``, flattening keyword
    arguments into positional ones using the forward signature.
    """
    root = model
    fqn_to_example_inputs = {}

    def _patched_module_call(self, *args, **kwargs):
        submodule_example_inputs = list(args).copy()
        normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
        num_args = _get_num_pos_args(self.forward) - 1
        num_to_pop = num_args - len(submodule_example_inputs)
        while num_to_pop and normalized_kwargs:
            normalized_kwargs.popitem(last=False)
            num_to_pop -= 1
        submodule_example_inputs.extend(normalized_kwargs.values())
        submodule_example_inputs_tuple = tuple(submodule_example_inputs)
        fqn = _get_path_of_module(root, self)
        if fqn is not None:
            fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
        return orig_module_call(self, *args, **kwargs)

    orig_module_call = nn.Module.__call__
    nn.Module.__call__ = _patched_module_call  # type: ignore[method-assign]
    try:
        model(*example_inputs)
    finally:
        nn.Module.__call__ = orig_module_call  # type: ignore[method-assign]
    return fqn_to_example_inputs


def _assert_and_get_unique_device(module: nn.Module) -> Any:
    """Return the single device used by a module, or raise on multiple devices."""
    devices = {p.device for p in module.parameters()} | {
        p.device for p in module.buffers()
    }
    if {tp.device("cpu"), tp.device("meta")} == devices:
        warnings.warn(
            "Both 'meta' and 'cpu' are present in the list of devices. Module can have one device. We Select 'cpu'.",
            stacklevel=2,
        )
        devices = {tp.device("cpu")}
    if len(devices) > 1:
        raise AssertionError(
            "prepare only works with cpu or single-device CUDA modules, "
            f"but got devices {devices}"
        )
    device = next(iter(devices)) if len(devices) > 0 else None
    return device