"""ONNX lowering for TensorPlay graph nodes.

Every captured ``call_function`` / ``call_method`` node is translated by a
handler registered here.  Handlers receive an :class:`OpContext`, read their
arguments by *name* (so positional and keyword capture behave identically) and
emit ONNX nodes through the :class:`GraphBuilder`.

Ops that ONNX expresses directly become a single node; the rest are lowered to
an equivalent ONNX subgraph (``linear`` -> ``Gemm``/``MatMul+Add``, ``silu`` ->
``Mul(x, Sigmoid(x))``, ``group_norm`` -> explicit moments, ...).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

import numpy as np
from onnx import helper, numpy_helper

from ._type_mapping import _dtype_to_onnx, _np_dtype_to_onnx, _to_numpy

__all__ = [
    "GraphBuilder",
    "OpContext",
    "Value",
    "lookup_function_handler",
    "lookup_method_handler",
]

_INT64_MAX = np.iinfo(np.int64).max
_INT64_MIN = np.iinfo(np.int64).min


class UnsupportedOperatorError(NotImplementedError):
    """Raised when a captured operation has no ONNX lowering."""


# ---------------------------------------------------------------------------
# Values and graph building
# ---------------------------------------------------------------------------


class Value:
    """A tensor flowing through the ONNX graph under construction."""

    __slots__ = ("name", "shape", "dtype")

    def __init__(
        self,
        name: str,
        shape: tuple | None = None,
        dtype: np.dtype | None = None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = np.dtype(dtype) if dtype is not None else None

    @property
    def rank(self) -> int | None:
        return None if self.shape is None else len(self.shape)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Value({self.name!r}, shape={self.shape}, dtype={self.dtype})"


class GraphBuilder:
    """Accumulates ONNX nodes, initializers and unique value names."""

    def __init__(self, opset: int, name: str = "tensorplay_model") -> None:
        self.opset = int(opset)
        self.name = name
        self.nodes: list[Any] = []
        self.initializers: list[Any] = []
        self.value_info: list[Any] = []
        self._used_names: set[str] = set()
        self._counters: dict[str, int] = {}
        self._constant_cache: dict[tuple, str] = {}

    # -- naming -------------------------------------------------------------

    def reserve(self, name: str) -> str:
        self._used_names.add(name)
        return name

    def unique(self, prefix: str) -> str:
        prefix = _sanitize(prefix)
        index = self._counters.get(prefix, 0)
        while True:
            candidate = prefix if index == 0 else f"{prefix}_{index}"
            index += 1
            if candidate not in self._used_names:
                self._counters[prefix] = index
                self._used_names.add(candidate)
                return candidate

    # -- constants ----------------------------------------------------------

    def initializer(self, array: np.ndarray, name_hint: str = "const") -> str:
        name = self.unique(name_hint)
        self.initializers.append(numpy_helper.from_array(array, name))
        return name

    def constant(
        self,
        value: Any,
        dtype: Any = None,
        name_hint: str = "const",
    ) -> str:
        """Materialize a python/tensor constant as a cached initializer."""

        array = _to_numpy(value)
        if dtype is not None:
            array = array.astype(np.dtype(dtype), copy=False)
        key = (array.dtype.str, array.shape, array.tobytes())
        cached = self._constant_cache.get(key)
        if cached is not None:
            return cached
        name = self.initializer(array, name_hint)
        self._constant_cache[key] = name
        return name

    def int64_1d(self, values: Sequence[int], name_hint: str = "axes") -> str:
        return self.constant(
            np.asarray(list(values), dtype=np.int64), name_hint=name_hint
        )

    # -- nodes --------------------------------------------------------------

    def op(
        self,
        op_type: str,
        inputs: Sequence[str],
        *,
        num_outputs: int = 1,
        name_hint: str | None = None,
        outputs: Sequence[str] | None = None,
        **attrs: Any,
    ) -> Any:
        """Emit one ONNX node and return its output name (or list of names)."""

        hint = name_hint or op_type.lower()
        if outputs is None:
            outputs = [
                self.unique(hint if num_outputs == 1 else f"{hint}_{index}")
                for index in range(num_outputs)
            ]
        else:
            outputs = [self.reserve(name) for name in outputs]
        attrs = {key: value for key, value in attrs.items() if value is not None}
        self.nodes.append(
            helper.make_node(op_type, list(inputs), list(outputs), **attrs)
        )
        return outputs[0] if len(outputs) == 1 else list(outputs)

    def require_opset(self, minimum: int, feature: str) -> None:
        if self.opset < minimum:
            raise UnsupportedOperatorError(
                f"{feature} requires ONNX opset >= {minimum}, got {self.opset}"
            )


def _sanitize(name: str) -> str:
    return "".join(char if char.isalnum() or char == "_" else "_" for char in str(name))


# ---------------------------------------------------------------------------
# Handler context
# ---------------------------------------------------------------------------


class OpContext:
    """Argument access plus emission helpers handed to every handler."""

    __slots__ = ("b", "node_name", "params", "args", "kwargs", "out_shape", "out_dtype")

    def __init__(
        self,
        builder: GraphBuilder,
        node_name: str,
        params: Sequence[str],
        args: Sequence[Any],
        kwargs: dict[str, Any],
        out_shape: tuple | None = None,
        out_dtype: np.dtype | None = None,
    ) -> None:
        self.b = builder
        self.node_name = node_name
        self.params = list(params)
        self.args = list(args)
        self.kwargs = dict(kwargs)
        self.out_shape = out_shape
        self.out_dtype = out_dtype

    # -- argument access ----------------------------------------------------

    def get(self, param: str, default: Any = None) -> Any:
        if param in self.kwargs:
            return self.kwargs[param]
        try:
            index = self.params.index(param)
        except ValueError as exc:  # pragma: no cover - handler bug
            raise KeyError(
                f"{param!r} is not declared in the parameter list {self.params}"
            ) from exc
        if index < len(self.args):
            return self.args[index]
        return default

    def has(self, param: str) -> bool:
        return self.get(param, _MISSING) is not _MISSING

    @property
    def x(self) -> Any:
        """First declared argument (the input tensor for nearly every op)."""

        return self.get(self.params[0])

    # -- value helpers ------------------------------------------------------

    def name(self, value: Any, name_hint: str = "const") -> str:
        """ONNX value name for ``value``, materializing constants on demand."""

        if isinstance(value, Value):
            return value.name
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return self.b.constant(value, name_hint=name_hint)

    def cast_like(self, value: Any, reference: Any, name_hint: str = "const") -> str:
        """Name for ``value``, materialized with ``reference``'s dtype."""

        if isinstance(value, (Value, str)):
            return self.name(value)
        dtype = self.dtype(reference)
        return self.b.constant(value, dtype=dtype, name_hint=name_hint)

    def shape(self, value: Any) -> tuple | None:
        if isinstance(value, Value):
            return value.shape
        if value is None:
            return None
        try:
            return tuple(_to_numpy(value).shape)
        except Exception:  # noqa: BLE001 - non-tensor argument
            return None

    def dtype(self, value: Any) -> np.dtype | None:
        if isinstance(value, Value):
            return value.dtype
        if value is None:
            return None
        try:
            return _to_numpy(value).dtype
        except Exception:  # noqa: BLE001 - non-tensor argument
            return None

    def rank(self, value: Any, what: str = "input") -> int:
        shape = self.shape(value)
        if shape is None:
            raise UnsupportedOperatorError(
                f"{self.node_name}: the rank of {what} is unknown; export with "
                "example inputs so shapes can be propagated"
            )
        return len(shape)

    def dim_size(self, value: Any, axis: int, what: str = "input") -> int:
        shape = self.shape(value)
        if shape is None:
            raise UnsupportedOperatorError(
                f"{self.node_name}: the shape of {what} is unknown; export with "
                "example inputs so shapes can be propagated"
            )
        return int(shape[axis])

    # -- emission -----------------------------------------------------------

    def op(self, op_type: str, inputs: Sequence[Any], **kwargs: Any) -> Any:
        kwargs.setdefault("name_hint", f"{self.node_name}_{op_type.lower()}")
        resolved = [item if isinstance(item, str) else self.name(item) for item in inputs]
        return self.b.op(op_type, resolved, **kwargs)

    def unary(self, op_type: str, **attrs: Any) -> str:
        return self.op(op_type, [self.x], **attrs)

    def binary(self, op_type: str) -> str:
        other = self.get(self.params[1])
        return self.op(op_type, [self.x, self.cast_like(other, self.x)])


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

Handler = Callable[[OpContext], Any]

_FUNCTION_HANDLERS: dict[tuple[str, str], tuple[Handler, list[str]]] = {}
_ANY_MODULE_HANDLERS: dict[str, tuple[Handler, list[str]]] = {}
_METHOD_HANDLERS: dict[str, tuple[Handler, list[str]]] = {}

_TP_MODULES = ("tensorplay.functional", "tensorplay.nn.functional", "tensorplay")


def register(
    name: str,
    params: str,
    *,
    module: str | None = None,
    methods: bool = True,
) -> Callable[[Handler], Handler]:
    """Register a handler for a captured function (and same-named method)."""

    param_list = params.split()

    def decorate(handler: Handler) -> Handler:
        entry = (handler, param_list)
        if module is None:
            _ANY_MODULE_HANDLERS[name] = entry
        else:
            _FUNCTION_HANDLERS[(module, name)] = entry
        if methods:
            _METHOD_HANDLERS.setdefault(name, entry)
        return handler

    return decorate


def register_method(name: str, params: str) -> Callable[[Handler], Handler]:
    """Register a handler used only for ``call_method`` nodes."""

    param_list = params.split()

    def decorate(handler: Handler) -> Handler:
        _METHOD_HANDLERS[name] = (handler, param_list)
        return handler

    return decorate


def alias(name: str, target: str, *, params: str | None = None) -> None:
    """Register ``name`` using the handler already registered for ``target``."""

    entry = _ANY_MODULE_HANDLERS.get(target) or _METHOD_HANDLERS.get(target)
    if entry is None:  # pragma: no cover - registration order bug
        raise KeyError(f"no handler registered for {target!r}")
    handler, param_list = entry
    if params is not None:
        param_list = params.split()
    _ANY_MODULE_HANDLERS[name] = (handler, param_list)
    _METHOD_HANDLERS.setdefault(name, (handler, param_list))


def lookup_function_handler(
    module: str, name: str
) -> tuple[Handler, list[str]] | None:
    entry = _FUNCTION_HANDLERS.get((module, name))
    if entry is not None:
        return entry
    return _ANY_MODULE_HANDLERS.get(name)


def lookup_method_handler(name: str) -> tuple[Handler, list[str]] | None:
    return _METHOD_HANDLERS.get(name)


# ---------------------------------------------------------------------------
# Shared lowering helpers
# ---------------------------------------------------------------------------


def _normalize_axis(axis: int, rank: int) -> int:
    axis = int(axis)
    return axis + rank if axis < 0 else axis


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _pair_attr(value: Any, count: int) -> list[int]:
    values = _as_int_list(value)
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise UnsupportedOperatorError(
            f"expected {count} values, got {values}"
        )
    return values


def _reduce(
    ctx: OpContext,
    onnx_op: str,
    data: Any,
    dims: Any,
    keepdim: bool,
    *,
    axes_input_since: int,
) -> str:
    """Emit a Reduce* node, honoring the opset that moved ``axes`` to an input."""

    keepdims = 1 if keepdim else 0
    if dims is None or (isinstance(dims, (list, tuple)) and not dims):
        axes: list[int] | None = None
    else:
        axes = _as_int_list(dims)
    if axes is None:
        return ctx.op(onnx_op, [data], keepdims=keepdims)
    if ctx.b.opset >= axes_input_since:
        return ctx.op(
            onnx_op, [data, ctx.b.int64_1d(axes, f"{ctx.node_name}_axes")], keepdims=keepdims
        )
    return ctx.op(onnx_op, [data], axes=axes, keepdims=keepdims)


def _reduce_sum(ctx: OpContext, data: Any, dims: Any, keepdim: bool) -> str:
    return _reduce(ctx, "ReduceSum", data, dims, keepdim, axes_input_since=13)


def _squeeze(ctx: OpContext, data: Any, axes: Sequence[int]) -> str:
    axes = list(axes)
    if not axes:
        return ctx.op("Identity", [data])
    if ctx.b.opset >= 13:
        return ctx.op("Squeeze", [data, ctx.b.int64_1d(axes, f"{ctx.node_name}_sq_axes")])
    return ctx.op("Squeeze", [data], axes=axes)


def _unsqueeze(ctx: OpContext, data: Any, axes: Sequence[int]) -> str:
    axes = list(axes)
    if not axes:
        return ctx.op("Identity", [data])
    if ctx.b.opset >= 13:
        return ctx.op("Unsqueeze", [data, ctx.b.int64_1d(axes, f"{ctx.node_name}_us_axes")])
    return ctx.op("Unsqueeze", [data], axes=axes)


def _reshape(ctx: OpContext, data: Any, shape: Sequence[int]) -> str:
    return ctx.op(
        "Reshape",
        [data, ctx.b.int64_1d(shape, f"{ctx.node_name}_shape")],
    )


def _cast(ctx: OpContext, data: Any, np_dtype: Any) -> str:
    return ctx.op("Cast", [data], to=int(_np_dtype_to_onnx(np_dtype)))


def _scalar(ctx: OpContext, value: Any, dtype: Any, hint: str = "scalar") -> str:
    return ctx.b.constant(
        np.asarray(value, dtype=np.dtype(dtype)), name_hint=f"{ctx.node_name}_{hint}"
    )


def _slice(
    ctx: OpContext,
    data: Any,
    starts: Sequence[int],
    ends: Sequence[int],
    axes: Sequence[int],
    steps: Sequence[int] | None = None,
) -> str:
    inputs = [
        data,
        ctx.b.int64_1d(starts, f"{ctx.node_name}_starts"),
        ctx.b.int64_1d(ends, f"{ctx.node_name}_ends"),
        ctx.b.int64_1d(axes, f"{ctx.node_name}_slice_axes"),
    ]
    if steps is not None:
        inputs.append(ctx.b.int64_1d(steps, f"{ctx.node_name}_steps"))
    return ctx.op("Slice", inputs)


def _float_dtype(ctx: OpContext, value: Any) -> np.dtype:
    dtype = ctx.dtype(value)
    if dtype is None or dtype.kind != "f":
        return np.dtype(np.float32)
    return dtype


# ---------------------------------------------------------------------------
# Pointwise unary ops
# ---------------------------------------------------------------------------

_SIMPLE_UNARY = {
    "abs": "Abs",
    "neg": "Neg",
    "exp": "Exp",
    "log": "Log",
    "sqrt": "Sqrt",
    "ceil": "Ceil",
    "floor": "Floor",
    "round": "Round",
    "sign": "Sign",
    "sin": "Sin",
    "cos": "Cos",
    "tan": "Tan",
    "asin": "Asin",
    "acos": "Acos",
    "atan": "Atan",
    "sinh": "Sinh",
    "cosh": "Cosh",
    "asinh": "Asinh",
    "acosh": "Acosh",
    "atanh": "Atanh",
    "erf": "Erf",
    "relu": "Relu",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "reciprocal": "Reciprocal",
    "logical_not": "Not",
    "invert": "Not",
    "isnan": "IsNaN",
    "isinf": "IsInf",
    "softsign": "Softsign",
    "hardswish": "HardSwish",
    "det": "Det",
    "nonzero": "NonZero",
}

for _name, _onnx_op in _SIMPLE_UNARY.items():
    register(_name, "input")(
        lambda ctx, _op=_onnx_op: ctx.unary(_op)
    )

_SIMPLE_BINARY = {
    "mul": "Mul",
    "multiply": "Mul",
    "div": "Div",
    "divide": "Div",
    "true_divide": "Div",
    "truediv": "Div",
    "pow": "Pow",
    "matmul": "MatMul",
    "mm": "MatMul",
    "bmm": "MatMul",
    "maximum": "Max",
    "minimum": "Min",
    "eq": "Equal",
    "lt": "Less",
    "le": "LessOrEqual",
    "gt": "Greater",
    "ge": "GreaterOrEqual",
    "logical_and": "And",
    "logical_or": "Or",
    "logical_xor": "Xor",
    "and": "And",
    "or": "Or",
    "xor": "Xor",
    "bitwise_and": "BitwiseAnd",
    "bitwise_or": "BitwiseOr",
    "bitwise_xor": "BitwiseXor",
}

for _name, _onnx_op in _SIMPLE_BINARY.items():
    register(_name, "input other")(
        lambda ctx, _op=_onnx_op: ctx.binary(_op)
    )


def _scaled_other(ctx: OpContext) -> str:
    """Second operand, pre-multiplied by ``alpha`` when one was given."""

    other = ctx.cast_like(ctx.get("other"), ctx.x)
    alpha = ctx.get("alpha", 1)
    if alpha is None or float(alpha) == 1.0:
        return other
    return ctx.op("Mul", [other, ctx.cast_like(float(alpha), ctx.x)])


@register("add", "input other alpha")
def _handle_add(ctx: OpContext) -> str:
    return ctx.op("Add", [ctx.x, _scaled_other(ctx)])


@register("sub", "input other alpha")
def _handle_sub(ctx: OpContext) -> str:
    return ctx.op("Sub", [ctx.x, _scaled_other(ctx)])


alias("subtract", "sub")


@register("remainder", "input other")
def _handle_remainder(ctx: OpContext) -> str:
    """``x - floor(x / y) * y``: the result takes the divisor's sign."""

    other = ctx.cast_like(ctx.get("other"), ctx.x)
    dtype = ctx.dtype(ctx.x)
    if dtype is not None and dtype.kind in "iu":
        # Integer Mod already rounds the quotient towards negative infinity.
        return ctx.op("Mod", [ctx.x, other], fmod=0)
    quotient = ctx.op("Floor", [ctx.op("Div", [ctx.x, other])])
    return ctx.op("Sub", [ctx.x, ctx.op("Mul", [quotient, other])])


@register("fmod", "input other")
def _handle_fmod(ctx: OpContext) -> str:
    """``x - trunc(x / y) * y``, which is what ``Mod(fmod=1)`` computes."""

    return ctx.op("Mod", [ctx.x, ctx.cast_like(ctx.get("other"), ctx.x)], fmod=1)


@register("ne", "input other")
def _handle_ne(ctx: OpContext) -> str:
    return ctx.op("Not", [ctx.binary("Equal")])


@register("floordiv", "input other")
def _handle_floordiv(ctx: OpContext) -> str:
    other = ctx.cast_like(ctx.get("other"), ctx.x)
    dtype = ctx.dtype(ctx.x)
    if dtype is not None and dtype.kind in "iu":
        as_float = _cast(ctx, ctx.x, np.float32)
        divided = ctx.op("Div", [as_float, _cast(ctx, other, np.float32)])
        return _cast(ctx, ctx.op("Floor", [divided]), dtype)
    return ctx.op("Floor", [ctx.op("Div", [ctx.x, other])])


alias("floor_divide", "floordiv")


@register("square", "input")
def _handle_square(ctx: OpContext) -> str:
    return ctx.op("Mul", [ctx.x, ctx.x])


@register("rsqrt", "input")
def _handle_rsqrt(ctx: OpContext) -> str:
    return ctx.op("Reciprocal", [ctx.op("Sqrt", [ctx.x])])


@register("log2", "input")
def _handle_log2(ctx: OpContext) -> str:
    scale = _scalar(ctx, 1.0 / math.log(2.0), _float_dtype(ctx, ctx.x), "log2")
    return ctx.op("Mul", [ctx.op("Log", [ctx.x]), scale])


@register("log10", "input")
def _handle_log10(ctx: OpContext) -> str:
    scale = _scalar(ctx, 1.0 / math.log(10.0), _float_dtype(ctx, ctx.x), "log10")
    return ctx.op("Mul", [ctx.op("Log", [ctx.x]), scale])


@register("log1p", "input")
def _handle_log1p(ctx: OpContext) -> str:
    one = _scalar(ctx, 1.0, _float_dtype(ctx, ctx.x), "one")
    return ctx.op("Log", [ctx.op("Add", [ctx.x, one])])


@register("expm1", "input")
def _handle_expm1(ctx: OpContext) -> str:
    one = _scalar(ctx, 1.0, _float_dtype(ctx, ctx.x), "one")
    return ctx.op("Sub", [ctx.op("Exp", [ctx.x]), one])


@register("clamp", "input min max")
def _handle_clamp(ctx: OpContext) -> str:
    minimum = ctx.get("min")
    maximum = ctx.get("max")
    inputs: list[Any] = [ctx.x]
    inputs.append("" if minimum is None else ctx.cast_like(minimum, ctx.x))
    if maximum is not None:
        inputs.append(ctx.cast_like(maximum, ctx.x))
    return ctx.op("Clip", inputs)


@register("clamp_min", "input min")
def _handle_clamp_min(ctx: OpContext) -> str:
    return ctx.op("Clip", [ctx.x, ctx.cast_like(ctx.get("min"), ctx.x)])


@register("clamp_max", "input max")
def _handle_clamp_max(ctx: OpContext) -> str:
    return ctx.op("Clip", [ctx.x, "", ctx.cast_like(ctx.get("max"), ctx.x)])


@register("where", "condition input other")
def _handle_where(ctx: OpContext) -> str:
    condition = ctx.get("condition")
    body = ctx.get("input")
    other = ctx.get("other")
    return ctx.op(
        "Where",
        [condition, ctx.cast_like(body, other), ctx.cast_like(other, body)],
    )


@register("masked_fill", "input mask value")
def _handle_masked_fill(ctx: OpContext) -> str:
    value = ctx.get("value")
    return ctx.op(
        "Where", [ctx.get("mask"), ctx.cast_like(value, ctx.x), ctx.x]
    )


@register("lerp", "input end weight")
def _handle_lerp(ctx: OpContext) -> str:
    start, end = ctx.x, ctx.get("end")
    weight = ctx.cast_like(ctx.get("weight"), start)
    delta = ctx.op("Sub", [end, start])
    return ctx.op("Add", [start, ctx.op("Mul", [delta, weight])])


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


@register("gelu", "input approximate")
def _handle_gelu(ctx: OpContext) -> str:
    approximate = ctx.get("approximate", "none") or "none"
    if ctx.b.opset >= 20:
        return ctx.op("Gelu", [ctx.x], approximate=str(approximate))
    dtype = _float_dtype(ctx, ctx.x)
    half = _scalar(ctx, 0.5, dtype, "half")
    one = _scalar(ctx, 1.0, dtype, "one")
    if approximate == "tanh":
        alpha = _scalar(ctx, math.sqrt(2.0 / math.pi), dtype, "gelu_alpha")
        beta = _scalar(ctx, 0.044715, dtype, "gelu_beta")
        three = _scalar(ctx, 3.0, dtype, "three")
        cubed = ctx.op("Pow", [ctx.x, three])
        inner = ctx.op("Add", [ctx.x, ctx.op("Mul", [beta, cubed])])
        tanh = ctx.op("Tanh", [ctx.op("Mul", [alpha, inner])])
        return ctx.op(
            "Mul", [ctx.op("Mul", [half, ctx.x]), ctx.op("Add", [one, tanh])]
        )
    inv_sqrt2 = _scalar(ctx, 1.0 / math.sqrt(2.0), dtype, "inv_sqrt2")
    erf = ctx.op("Erf", [ctx.op("Mul", [ctx.x, inv_sqrt2])])
    return ctx.op("Mul", [ctx.op("Mul", [half, ctx.x]), ctx.op("Add", [one, erf])])


@register("silu", "input inplace")
def _handle_silu(ctx: OpContext) -> str:
    return ctx.op("Mul", [ctx.x, ctx.op("Sigmoid", [ctx.x])])


alias("swish", "silu")


@register("mish", "input")
def _handle_mish(ctx: OpContext) -> str:
    if ctx.b.opset >= 18:
        return ctx.op("Mish", [ctx.x])
    softplus = ctx.op("Softplus", [ctx.x])
    return ctx.op("Mul", [ctx.x, ctx.op("Tanh", [softplus])])


@register("hardsigmoid", "input")
def _handle_hardsigmoid(ctx: OpContext) -> str:
    return ctx.op("HardSigmoid", [ctx.x], alpha=1.0 / 6.0, beta=0.5)


@register("leaky_relu", "input negative_slope inplace")
def _handle_leaky_relu(ctx: OpContext) -> str:
    return ctx.op("LeakyRelu", [ctx.x], alpha=float(ctx.get("negative_slope", 0.01)))


@register("elu", "input alpha scale input_scale")
def _handle_elu(ctx: OpContext) -> str:
    scale = ctx.get("scale", 1)
    input_scale = ctx.get("input_scale", 1)
    if float(scale) != 1.0 or float(input_scale) != 1.0:
        raise UnsupportedOperatorError(
            "elu with scale/input_scale != 1 has no ONNX equivalent"
        )
    return ctx.op("Elu", [ctx.x], alpha=float(ctx.get("alpha", 1.0)))


@register("selu", "input")
def _handle_selu(ctx: OpContext) -> str:
    return ctx.op("Selu", [ctx.x])


@register("celu", "input alpha")
def _handle_celu(ctx: OpContext) -> str:
    return ctx.op("Celu", [ctx.x], alpha=float(ctx.get("alpha", 1.0)))


@register("hardtanh", "input min_val max_val inplace")
def _handle_hardtanh(ctx: OpContext) -> str:
    low = ctx.cast_like(float(ctx.get("min_val", -1.0)), ctx.x)
    high = ctx.cast_like(float(ctx.get("max_val", 1.0)), ctx.x)
    return ctx.op("Clip", [ctx.x, low, high])


@register("relu6", "input inplace")
def _handle_relu6(ctx: OpContext) -> str:
    return ctx.op(
        "Clip", [ctx.x, ctx.cast_like(0.0, ctx.x), ctx.cast_like(6.0, ctx.x)]
    )


@register("threshold", "input threshold value inplace")
def _handle_threshold(ctx: OpContext) -> str:
    limit = ctx.cast_like(float(ctx.get("threshold")), ctx.x)
    value = ctx.cast_like(float(ctx.get("value")), ctx.x)
    return ctx.op("Where", [ctx.op("Greater", [ctx.x, limit]), ctx.x, value])


@register("softplus", "input beta threshold")
def _handle_softplus(ctx: OpContext) -> str:
    beta = float(ctx.get("beta", 1.0))
    if beta == 1.0:
        return ctx.op("Softplus", [ctx.x])
    dtype = _float_dtype(ctx, ctx.x)
    beta_const = _scalar(ctx, beta, dtype, "beta")
    scaled = ctx.op("Softplus", [ctx.op("Mul", [ctx.x, beta_const])])
    return ctx.op("Div", [scaled, beta_const])


@register("prelu", "input weight")
def _handle_prelu(ctx: OpContext) -> str:
    weight = ctx.get("weight")
    rank = ctx.rank(ctx.x)
    weight_shape = ctx.shape(weight)
    slope: Any = weight
    if rank > 2 and weight_shape is not None and int(np.prod(weight_shape)) > 1:
        slope = _reshape(
            ctx, ctx.name(weight), [int(np.prod(weight_shape))] + [1] * (rank - 2)
        )
    return ctx.op("PRelu", [ctx.x, slope])


@register("softmax", "input dim dtype")
def _handle_softmax(ctx: OpContext) -> str:
    return ctx.op("Softmax", [ctx.x], axis=int(ctx.get("dim", -1) or -1))


@register("log_softmax", "input dim dtype")
def _handle_log_softmax(ctx: OpContext) -> str:
    return ctx.op("LogSoftmax", [ctx.x], axis=int(ctx.get("dim", -1) or -1))


@register("glu", "input dim")
def _handle_glu(ctx: OpContext) -> str:
    dim = _normalize_axis(ctx.get("dim", -1), ctx.rank(ctx.x))
    size = ctx.dim_size(ctx.x, dim)
    if size % 2:
        raise UnsupportedOperatorError("glu requires an even split dimension")
    half = size // 2
    first, second = ctx.op(
        "Split",
        [ctx.x, ctx.b.int64_1d([half, half], f"{ctx.node_name}_split")],
        axis=dim,
        num_outputs=2,
    )
    return ctx.op("Mul", [first, ctx.op("Sigmoid", [second])])


@register("dropout", "input p training inplace")
def _handle_dropout(ctx: OpContext) -> str:
    if not bool(ctx.get("training", True)) or float(ctx.get("p", 0.5)) == 0.0:
        return ctx.op("Identity", [ctx.x])
    ratio = _scalar(ctx, float(ctx.get("p", 0.5)), np.float32, "ratio")
    training = _scalar(ctx, True, np.bool_, "training")
    outputs = ctx.op("Dropout", [ctx.x, ratio, training], num_outputs=2)
    return outputs[0]


@register("identity", "input")
def _handle_identity(ctx: OpContext) -> str:
    return ctx.op("Identity", [ctx.x])


for _passthrough in ("detach", "contiguous", "clone", "alias"):
    register(_passthrough, "input")(lambda ctx: ctx.op("Identity", [ctx.x]))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def _variadic_ints(ctx: OpContext, param: str) -> list[int]:
    """Read ``x.view(2, 3)`` and ``x.view([2, 3])`` alike."""

    value = ctx.get(param)
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    index = ctx.params.index(param)
    tail = ctx.args[index:]
    if tail:
        return [int(item) for item in tail]
    if value is None:
        return []
    return [int(value)]


@register("sum", "input dim keepdim dtype")
def _handle_sum(ctx: OpContext) -> str:
    return _reduce_sum(ctx, ctx.x, ctx.get("dim"), bool(ctx.get("keepdim", False)))


@register("mean", "input dim keepdim dtype")
def _handle_mean(ctx: OpContext) -> str:
    return _reduce(
        ctx, "ReduceMean", ctx.x, ctx.get("dim"), bool(ctx.get("keepdim", False)),
        axes_input_since=18,
    )


@register("prod", "input dim keepdim dtype")
def _handle_prod(ctx: OpContext) -> str:
    return _reduce(
        ctx, "ReduceProd", ctx.x, ctx.get("dim"), bool(ctx.get("keepdim", False)),
        axes_input_since=18,
    )


@register("amax", "input dim keepdim")
def _handle_amax(ctx: OpContext) -> str:
    return _reduce(
        ctx, "ReduceMax", ctx.x, ctx.get("dim"), bool(ctx.get("keepdim", False)),
        axes_input_since=18,
    )


@register("amin", "input dim keepdim")
def _handle_amin(ctx: OpContext) -> str:
    return _reduce(
        ctx, "ReduceMin", ctx.x, ctx.get("dim"), bool(ctx.get("keepdim", False)),
        axes_input_since=18,
    )


def _minmax(ctx: OpContext, reduce_op: str, arg_op: str, elementwise: str) -> Any:
    """``max``/``min``: elementwise, whole-tensor reduction, or (values, indices).

    ``tensorplay.functional.max(input, dim, keepdim)`` reduces; the second
    argument is only a tensor for the ``Tensor.max(other)`` overload.
    """

    dim = ctx.get("dim")
    if isinstance(dim, Value):
        return ctx.op(elementwise, [ctx.x, dim])
    keepdim = bool(ctx.get("keepdim", False))
    if dim is None:
        return _reduce(ctx, reduce_op, ctx.x, None, keepdim, axes_input_since=18)
    axis = _normalize_axis(dim, ctx.rank(ctx.x))
    values = _reduce(ctx, reduce_op, ctx.x, [axis], keepdim, axes_input_since=18)
    indices = ctx.op(arg_op, [ctx.x], axis=axis, keepdims=1 if keepdim else 0)
    return [values, indices]


@register("max", "input dim keepdim")
def _handle_max(ctx: OpContext) -> Any:
    return _minmax(ctx, "ReduceMax", "ArgMax", "Max")


@register("min", "input dim keepdim")
def _handle_min(ctx: OpContext) -> Any:
    return _minmax(ctx, "ReduceMin", "ArgMin", "Min")


def _argreduce(ctx: OpContext, onnx_op: str) -> str:
    dim = ctx.get("dim")
    keepdim = bool(ctx.get("keepdim", False))
    data: Any = ctx.x
    if dim is None:
        data = _reshape(ctx, ctx.x, [-1])
        dim, keepdim = 0, False
    return ctx.op(onnx_op, [data], axis=int(dim), keepdims=1 if keepdim else 0)


@register("argmax", "input dim keepdim")
def _handle_argmax(ctx: OpContext) -> str:
    return _argreduce(ctx, "ArgMax")


@register("argmin", "input dim keepdim")
def _handle_argmin(ctx: OpContext) -> str:
    return _argreduce(ctx, "ArgMin")


def _bool_reduce(ctx: OpContext, onnx_op: str) -> str:
    as_int = _cast(ctx, ctx.x, np.int32)
    reduced = _reduce(
        ctx, onnx_op, as_int, ctx.get("dim"), bool(ctx.get("keepdim", False)),
        axes_input_since=18,
    )
    return _cast(ctx, reduced, np.bool_)


@register("all", "input dim keepdim")
def _handle_all(ctx: OpContext) -> str:
    return _bool_reduce(ctx, "ReduceMin")


@register("any", "input dim keepdim")
def _handle_any(ctx: OpContext) -> str:
    return _bool_reduce(ctx, "ReduceMax")


@register("logsumexp", "input dim keepdim")
def _handle_logsumexp(ctx: OpContext) -> str:
    return _reduce(
        ctx, "ReduceLogSumExp", ctx.x, ctx.get("dim"),
        bool(ctx.get("keepdim", False)), axes_input_since=18,
    )


@register("cumsum", "input dim dtype")
def _handle_cumsum(ctx: OpContext) -> str:
    axis = _scalar(ctx, int(ctx.get("dim", 0)), np.int64, "axis")
    return ctx.op("CumSum", [ctx.x, axis])


def _reduced_count(ctx: OpContext, dims: Any) -> int:
    shape = ctx.shape(ctx.x)
    if shape is None:
        raise UnsupportedOperatorError(
            f"{ctx.node_name}: variance needs a known input shape"
        )
    if dims is None or (isinstance(dims, (list, tuple)) and not dims):
        axes = range(len(shape))
    else:
        axes = [_normalize_axis(axis, len(shape)) for axis in _as_int_list(dims)]
    count = 1
    for axis in axes:
        count *= int(shape[axis])
    return count


def _variance(ctx: OpContext) -> tuple[str, np.dtype]:
    dims = ctx.get("dim")
    keepdim = bool(ctx.get("keepdim", False))
    correction = float(ctx.get("correction", 1) or 0)
    dtype = _float_dtype(ctx, ctx.x)
    mean = _reduce(ctx, "ReduceMean", ctx.x, dims, True, axes_input_since=18)
    centered = ctx.op("Sub", [ctx.x, mean])
    squares = ctx.op("Mul", [centered, centered])
    total = _reduce_sum(ctx, squares, dims, keepdim)
    denominator = max(_reduced_count(ctx, dims) - correction, 1.0)
    return ctx.op("Div", [total, _scalar(ctx, denominator, dtype, "count")]), dtype


@register("var", "input correction dim keepdim")
def _handle_var(ctx: OpContext) -> str:
    variance, _ = _variance(ctx)
    return variance


@register("std", "input correction dim keepdim")
def _handle_std(ctx: OpContext) -> str:
    variance, _ = _variance(ctx)
    return ctx.op("Sqrt", [variance])


def _p_norm(ctx: OpContext, data: Any, p: float, dims: Any, keepdim: bool) -> str:
    if p == 2.0:
        return _reduce(ctx, "ReduceL2", data, dims, keepdim, axes_input_since=18)
    if p == 1.0:
        return _reduce(ctx, "ReduceL1", data, dims, keepdim, axes_input_since=18)
    absolute = ctx.op("Abs", [data])
    if math.isinf(p):
        onnx_op = "ReduceMax" if p > 0 else "ReduceMin"
        return _reduce(ctx, onnx_op, absolute, dims, keepdim, axes_input_since=18)
    dtype = _float_dtype(ctx, data)
    exponent = _scalar(ctx, p, dtype, "p")
    powered = ctx.op("Pow", [absolute, exponent])
    total = _reduce_sum(ctx, powered, dims, keepdim)
    inverse = _scalar(ctx, 1.0 / p, dtype, "inv_p")
    return ctx.op("Pow", [total, inverse])


@register("norm", "input p dim keepdim", methods=False)
def _handle_norm(ctx: OpContext) -> str:
    return _p_norm(
        ctx, ctx.x, float(ctx.get("p", 2.0) or 2.0), ctx.get("dim"),
        bool(ctx.get("keepdim", False)),
    )


@register_method("norm", "input dim p keepdim")
def _handle_norm_method(ctx: OpContext) -> str:
    """``Tensor.norm`` has a ``(p)`` and a ``(dim, p, keepdim)`` overload."""

    first = ctx.get("dim")
    if first is None or isinstance(first, (list, tuple)):
        dims = None if first is None else _as_int_list(first)
        p = float(ctx.get("p", 2.0) or 2.0)
        keepdim = bool(ctx.get("keepdim", False))
    else:
        dims, p, keepdim = None, float(first), False
    return _p_norm(ctx, ctx.x, p, dims, keepdim)


@register("topk", "input k dim largest sorted impl")
def _handle_topk(ctx: OpContext) -> list[str]:
    k = ctx.b.constant(
        np.asarray([int(ctx.get("k"))], dtype=np.int64), name_hint=f"{ctx.node_name}_k"
    )
    return ctx.op(
        "TopK",
        [ctx.x, k],
        num_outputs=2,
        axis=int(ctx.get("dim", -1)),
        largest=1 if ctx.get("largest", True) else 0,
        sorted=1 if ctx.get("sorted", True) else 0,
    )


@register("sort", "input dim descending")
def _handle_sort(ctx: OpContext) -> list[str]:
    rank = ctx.rank(ctx.x)
    axis = _normalize_axis(ctx.get("dim", -1), rank)
    size = ctx.dim_size(ctx.x, axis)
    k = ctx.b.constant(
        np.asarray([size], dtype=np.int64), name_hint=f"{ctx.node_name}_k"
    )
    return ctx.op(
        "TopK",
        [ctx.x, k],
        num_outputs=2,
        axis=axis,
        largest=1 if ctx.get("descending", False) else 0,
        sorted=1,
    )


# ---------------------------------------------------------------------------
# Shape / layout
# ---------------------------------------------------------------------------


@register("reshape", "input shape")
def _handle_reshape(ctx: OpContext) -> str:
    return _reshape(ctx, ctx.x, _variadic_ints(ctx, "shape"))


alias("view", "reshape", params="input shape")
@register("flatten", "input start_dim end_dim")
def _handle_flatten(ctx: OpContext) -> str:
    """ONNX ``Flatten`` always yields a 2-D tensor, so only the ``start_dim=1``
    case maps onto it directly; every other range becomes a ``Reshape``.
    """

    rank = ctx.rank(ctx.x)
    start = _normalize_axis(ctx.get("start_dim", 0) or 0, rank)
    end_dim = ctx.get("end_dim", -1)
    end = _normalize_axis(-1 if end_dim is None else end_dim, rank)
    if end == rank - 1:
        if start == 0:
            return _reshape(ctx, ctx.x, [-1])
        if start == 1:
            return ctx.op("Flatten", [ctx.x], axis=1)
    shape = ctx.shape(ctx.x)
    merged = int(np.prod(shape[start : end + 1])) if end >= start else 1
    new_shape = list(shape[:start]) + [merged] + list(shape[end + 1 :])
    return _reshape(ctx, ctx.x, new_shape)


@register("unflatten", "input dim sizes")
def _handle_unflatten(ctx: OpContext) -> str:
    rank = ctx.rank(ctx.x)
    dim = _normalize_axis(ctx.get("dim"), rank)
    shape = list(ctx.shape(ctx.x))
    sizes = _as_int_list(ctx.get("sizes"))
    if -1 in sizes:
        known = int(np.prod([size for size in sizes if size != -1])) or 1
        sizes = [shape[dim] // known if size == -1 else size for size in sizes]
    return _reshape(ctx, ctx.x, shape[:dim] + sizes + shape[dim + 1 :])


@register("transpose", "input dim0 dim1")
def _handle_transpose(ctx: OpContext) -> str:
    rank = ctx.rank(ctx.x)
    dim0 = _normalize_axis(ctx.get("dim0", 0), rank)
    dim1 = _normalize_axis(ctx.get("dim1", 1), rank)
    perm = list(range(rank))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return ctx.op("Transpose", [ctx.x], perm=perm)


alias("swapaxes", "transpose")
alias("swapdims", "transpose")


@register("t", "input")
def _handle_t(ctx: OpContext) -> str:
    rank = ctx.rank(ctx.x)
    if rank < 2:
        return ctx.op("Identity", [ctx.x])
    return ctx.op("Transpose", [ctx.x], perm=[1, 0])


@register("permute", "input dims")
def _handle_permute(ctx: OpContext) -> str:
    rank = ctx.rank(ctx.x)
    perm = [_normalize_axis(axis, rank) for axis in _variadic_ints(ctx, "dims")]
    return ctx.op("Transpose", [ctx.x], perm=perm)


@register("squeeze", "input dim")
def _handle_squeeze(ctx: OpContext) -> str:
    dim = ctx.get("dim")
    if dim is None:
        shape = ctx.shape(ctx.x)
        if shape is None:
            return ctx.op("Squeeze", [ctx.x])
        axes = [index for index, size in enumerate(shape) if size == 1]
        return _squeeze(ctx, ctx.x, axes)
    rank = ctx.rank(ctx.x)
    return _squeeze(ctx, ctx.x, [_normalize_axis(axis, rank) for axis in _as_int_list(dim)])


@register("unsqueeze", "input dim")
def _handle_unsqueeze(ctx: OpContext) -> str:
    rank = ctx.rank(ctx.x) + 1
    axes = [_normalize_axis(axis, rank) for axis in _as_int_list(ctx.get("dim"))]
    return _unsqueeze(ctx, ctx.x, axes)


@register("expand", "input size implicit")
def _handle_expand(ctx: OpContext) -> str:
    sizes = _variadic_ints(ctx, "size")
    shape = ctx.shape(ctx.x)
    if shape is not None:
        offset = len(sizes) - len(shape)
        sizes = [
            int(shape[index - offset]) if size == -1 else size
            for index, size in enumerate(sizes)
        ]
    return ctx.op("Expand", [ctx.x, ctx.b.int64_1d(sizes, f"{ctx.node_name}_shape")])


alias("broadcast_to", "expand", params="input size")


@register("expand_as", "input other")
def _handle_expand_as(ctx: OpContext) -> str:
    other = ctx.get("other")
    return ctx.op("Expand", [ctx.x, ctx.op("Shape", [other])])


@register("repeat", "input repeats")
def _handle_repeat(ctx: OpContext) -> str:
    repeats = _variadic_ints(ctx, "repeats")
    rank = ctx.rank(ctx.x)
    data: Any = ctx.x
    if len(repeats) > rank:
        shape = list(ctx.shape(ctx.x))
        data = _reshape(ctx, ctx.x, [1] * (len(repeats) - rank) + shape)
    return ctx.op(
        "Tile", [data, ctx.b.int64_1d(repeats, f"{ctx.node_name}_repeats")]
    )


alias("tile", "repeat", params="input dims")


@register("cat", "tensors dim")
def _handle_cat(ctx: OpContext) -> str:
    tensors = ctx.get("tensors")
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    return ctx.op("Concat", list(tensors), axis=int(ctx.get("dim", 0) or 0))


alias("concat", "cat")
alias("concatenate", "cat")


@register("stack", "tensors dim")
def _handle_stack(ctx: OpContext) -> str:
    tensors = ctx.get("tensors")
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    axis = int(ctx.get("dim", 0) or 0)
    rank = ctx.rank(tensors[0]) + 1
    axis = _normalize_axis(axis, rank)
    expanded = [_unsqueeze(ctx, tensor, [axis]) for tensor in tensors]
    return ctx.op("Concat", expanded, axis=axis)


def _emit_split(ctx: OpContext, data: Any, sizes: Sequence[int], axis: int) -> list[str]:
    return ctx.op(
        "Split",
        [data, ctx.b.int64_1d(sizes, f"{ctx.node_name}_split")],
        axis=axis,
        num_outputs=len(sizes),
    )


@register("split", "input split_size dim")
def _handle_split(ctx: OpContext) -> Any:
    rank = ctx.rank(ctx.x)
    axis = _normalize_axis(ctx.get("dim", 0) or 0, rank)
    total = ctx.dim_size(ctx.x, axis)
    split_size = ctx.get("split_size")
    if isinstance(split_size, (list, tuple)):
        sizes = [int(size) for size in split_size]
    else:
        step = int(split_size)
        sizes = [step] * (total // step)
        if total % step:
            sizes.append(total % step)
    if len(sizes) == 1:
        # Still a one-element sequence, so downstream indexing keeps working.
        return [ctx.op("Identity", [ctx.x])]
    return _emit_split(ctx, ctx.x, sizes, axis)


@register("chunk", "input chunks dim")
def _handle_chunk(ctx: OpContext) -> Any:
    rank = ctx.rank(ctx.x)
    axis = _normalize_axis(ctx.get("dim", 0) or 0, rank)
    total = ctx.dim_size(ctx.x, axis)
    chunks = int(ctx.get("chunks"))
    step = -(-total // chunks)
    sizes = []
    remaining = total
    while remaining > 0:
        sizes.append(min(step, remaining))
        remaining -= sizes[-1]
    if len(sizes) == 1:
        return [ctx.op("Identity", [ctx.x])]
    return _emit_split(ctx, ctx.x, sizes, axis)


@register("narrow", "input dim start length")
def _handle_narrow(ctx: OpContext) -> str:
    rank = ctx.rank(ctx.x)
    axis = _normalize_axis(ctx.get("dim"), rank)
    start = int(ctx.get("start"))
    length = int(ctx.get("length"))
    return _slice(ctx, ctx.x, [start], [start + length], [axis])


@register("flip", "input dims")
def _handle_flip(ctx: OpContext) -> str:
    rank = ctx.rank(ctx.x)
    axes = [_normalize_axis(axis, rank) for axis in _variadic_ints(ctx, "dims")]
    return _slice(
        ctx, ctx.x, [-1] * len(axes), [_INT64_MIN] * len(axes), axes, [-1] * len(axes)
    )


@register("gather", "input dim index")
def _handle_gather(ctx: OpContext) -> str:
    index = _cast(ctx, ctx.get("index"), np.int64)
    return ctx.op("GatherElements", [ctx.x, index], axis=int(ctx.get("dim")))


@register("index_select", "input dim index")
def _handle_index_select(ctx: OpContext) -> str:
    index = _cast(ctx, ctx.get("index"), np.int64)
    return ctx.op("Gather", [ctx.x, index], axis=int(ctx.get("dim")))


@register("take", "input index")
def _handle_take(ctx: OpContext) -> str:
    flat = _reshape(ctx, ctx.x, [-1])
    index = _cast(ctx, ctx.get("index"), np.int64)
    return ctx.op("Gather", [flat, index], axis=0)


@register("tril", "input diagonal")
def _handle_tril(ctx: OpContext) -> str:
    k = _scalar(ctx, int(ctx.get("diagonal", 0) or 0), np.int64, "k")
    return ctx.op("Trilu", [ctx.x, k], upper=0)


@register("triu", "input diagonal")
def _handle_triu(ctx: OpContext) -> str:
    k = _scalar(ctx, int(ctx.get("diagonal", 0) or 0), np.int64, "k")
    return ctx.op("Trilu", [ctx.x, k], upper=1)


@register("one_hot", "input num_classes")
def _handle_one_hot(ctx: OpContext) -> str:
    num_classes = int(ctx.get("num_classes", -1))
    if num_classes < 0:
        raise UnsupportedOperatorError(
            "one_hot needs an explicit num_classes to export (num_classes=-1 "
            "depends on the runtime values of the input)"
        )
    depth = _scalar(ctx, num_classes, np.int64, "depth")
    values = ctx.b.constant(
        np.asarray([0, 1], dtype=np.int64), name_hint=f"{ctx.node_name}_values"
    )
    return ctx.op("OneHot", [_cast(ctx, ctx.x, np.int64), depth, values], axis=-1)


@register("pixel_shuffle", "input upscale_factor")
def _handle_pixel_shuffle(ctx: OpContext) -> str:
    return ctx.op(
        "DepthToSpace", [ctx.x], blocksize=int(ctx.get("upscale_factor")), mode="CRD"
    )


@register("pixel_unshuffle", "input downscale_factor")
def _handle_pixel_unshuffle(ctx: OpContext) -> str:
    # ONNX SpaceToDepth interleaves the channel axis in the opposite order
    # (DCR), so the required layout is spelled out explicitly.
    factor = int(ctx.get("downscale_factor"))
    batch, channels, height, width = (int(size) for size in ctx.shape(ctx.x))
    split = _reshape(
        ctx,
        ctx.x,
        [batch, channels, height // factor, factor, width // factor, factor],
    )
    ordered = ctx.op("Transpose", [split], perm=[0, 1, 3, 5, 2, 4])
    return _reshape(
        ctx,
        ordered,
        [batch, channels * factor * factor, height // factor, width // factor],
    )


_CAST_METHODS = {
    "float": np.float32,
    "double": np.float64,
    "half": np.float16,
    "long": np.int64,
    "int": np.int32,
    "short": np.int16,
    "char": np.int8,
    "byte": np.uint8,
    "bool": np.bool_,
}

for _method, _np_type in _CAST_METHODS.items():
    register_method(_method, "input")(
        lambda ctx, _type=_np_type: _cast(ctx, ctx.x, _type)
    )


@register("to", "input dtype")
def _handle_to(ctx: OpContext) -> str:
    dtype = ctx.get("dtype")
    if dtype is None or isinstance(dtype, str):
        # ``.to(device)`` / ``.to("cpu")`` is a no-op for the exported graph.
        return ctx.op("Identity", [ctx.x])
    try:
        onnx_type = _dtype_to_onnx(dtype)
    except TypeError:
        return ctx.op("Identity", [ctx.x])
    return ctx.op("Cast", [ctx.x], to=int(onnx_type))


alias("type", "to")


@register("type_as", "input other")
def _handle_type_as(ctx: OpContext) -> str:
    dtype = ctx.dtype(ctx.get("other"))
    if dtype is None:
        return ctx.op("Identity", [ctx.x])
    return _cast(ctx, ctx.x, dtype)


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------


@register("linear", "input weight bias")
def _handle_linear(ctx: OpContext) -> str:
    weight, bias = ctx.get("weight"), ctx.get("bias")
    shape = ctx.shape(ctx.x)
    if shape is not None and len(shape) == 2:
        inputs = [ctx.x, weight] + ([bias] if bias is not None else [])
        return ctx.op("Gemm", inputs, transB=1)
    transposed = ctx.op("Transpose", [weight], perm=[1, 0])
    product = ctx.op("MatMul", [ctx.x, transposed])
    if bias is None:
        return product
    return ctx.op("Add", [product, bias])


@register("addmm", "input mat1 mat2 beta alpha")
def _handle_addmm(ctx: OpContext) -> str:
    return ctx.op(
        "Gemm",
        [ctx.get("mat1"), ctx.get("mat2"), ctx.x],
        alpha=float(ctx.get("alpha", 1)),
        beta=float(ctx.get("beta", 1)),
    )


@register("baddbmm", "input batch1 batch2 beta alpha")
def _handle_baddbmm(ctx: OpContext) -> str:
    product = ctx.op("MatMul", [ctx.get("batch1"), ctx.get("batch2")])
    alpha = float(ctx.get("alpha", 1))
    beta = float(ctx.get("beta", 1))
    dtype = _float_dtype(ctx, ctx.x)
    if alpha != 1.0:
        product = ctx.op("Mul", [product, _scalar(ctx, alpha, dtype, "alpha")])
    base: Any = ctx.x
    if beta != 1.0:
        base = ctx.op("Mul", [ctx.x, _scalar(ctx, beta, dtype, "beta")])
    return ctx.op("Add", [base, product])


# ---------------------------------------------------------------------------
# Convolution and pooling
# ---------------------------------------------------------------------------


def _spatial_rank(ctx: OpContext, weight: Any) -> int:
    shape = ctx.shape(weight)
    if shape is None:
        raise UnsupportedOperatorError(
            f"{ctx.node_name}: convolution needs a known weight shape"
        )
    return len(shape) - 2


def _conv_attrs(ctx: OpContext, spatial: int) -> dict[str, Any]:
    padding = ctx.get("padding", 0)
    attrs: dict[str, Any] = {
        "strides": _pair_attr(ctx.get("stride", 1), spatial),
        "dilations": _pair_attr(ctx.get("dilation", 1), spatial),
        "group": int(ctx.get("groups", 1)),
    }
    if isinstance(padding, str):
        attrs["auto_pad"] = "SAME_UPPER" if padding == "same" else "VALID"
    else:
        pads = _pair_attr(padding, spatial)
        attrs["pads"] = pads + pads
    return attrs


@register("conv1d", "input weight bias stride padding dilation groups")
@register("conv2d", "input weight bias stride padding dilation groups")
@register("conv3d", "input weight bias stride padding dilation groups")
def _handle_conv(ctx: OpContext) -> str:
    weight = ctx.get("weight")
    bias = ctx.get("bias")
    spatial = _spatial_rank(ctx, weight)
    inputs = [ctx.x, weight] + ([bias] if bias is not None else [])
    attrs = _conv_attrs(ctx, spatial)
    attrs["kernel_shape"] = list(ctx.shape(weight)[2:])
    return ctx.op("Conv", inputs, **attrs)


@register("conv_transpose1d", "input weight bias stride padding output_padding groups dilation")
@register("conv_transpose2d", "input weight bias stride padding output_padding groups dilation")
@register("conv_transpose3d", "input weight bias stride padding output_padding groups dilation")
def _handle_conv_transpose(ctx: OpContext) -> str:
    weight = ctx.get("weight")
    bias = ctx.get("bias")
    spatial = _spatial_rank(ctx, weight)
    inputs = [ctx.x, weight] + ([bias] if bias is not None else [])
    attrs = _conv_attrs(ctx, spatial)
    attrs["kernel_shape"] = list(ctx.shape(weight)[2:])
    output_padding = _pair_attr(ctx.get("output_padding", 0), spatial)
    if any(output_padding):
        attrs["output_padding"] = output_padding
    return ctx.op("ConvTranspose", inputs, **attrs)


def _pool_attrs(ctx: OpContext, spatial: int) -> dict[str, Any]:
    pads = _pair_attr(ctx.get("padding", 0), spatial)
    return {
        "kernel_shape": _pair_attr(ctx.get("kernel_size"), spatial),
        "strides": _pair_attr(
            ctx.get("stride") if ctx.get("stride") is not None else ctx.get("kernel_size"),
            spatial,
        ),
        "pads": pads + pads,
        "ceil_mode": 1 if ctx.get("ceil_mode", False) else 0,
    }


@register("max_pool1d", "input kernel_size stride padding dilation ceil_mode return_indices")
@register("max_pool2d", "input kernel_size stride padding dilation ceil_mode return_indices")
@register("max_pool3d", "input kernel_size stride padding dilation ceil_mode return_indices")
def _handle_max_pool(ctx: OpContext) -> Any:
    spatial = ctx.rank(ctx.x) - 2
    attrs = _pool_attrs(ctx, spatial)
    attrs["dilations"] = _pair_attr(ctx.get("dilation", 1), spatial)
    if not ctx.get("return_indices", False):
        return ctx.op("MaxPool", [ctx.x], **attrs)
    values, indices = ctx.op("MaxPool", [ctx.x], num_outputs=2, **attrs)
    # ONNX numbers the indices flat over (C, *spatial); this op numbers them
    # within each (batch, channel) plane, so the plane origin is subtracted.
    flat_attrs = {
        "kernel_shape": [1] * spatial,
        "strides": [1] * spatial,
        "pads": [0] * (2 * spatial),
    }
    _, plane_indices = ctx.op("MaxPool", [ctx.x], num_outputs=2, **flat_attrs)
    axes = list(range(2, 2 + spatial))
    origin = _slice(ctx, plane_indices, [0] * spatial, [1] * spatial, axes)
    return [values, ctx.op("Sub", [indices, origin])]


@register("avg_pool1d", "input kernel_size stride padding ceil_mode count_include_pad divisor_override")
@register("avg_pool2d", "input kernel_size stride padding ceil_mode count_include_pad divisor_override")
@register("avg_pool3d", "input kernel_size stride padding ceil_mode count_include_pad divisor_override")
def _handle_avg_pool(ctx: OpContext) -> str:
    if ctx.get("divisor_override") is not None:
        raise UnsupportedOperatorError(
            "avg_pool with divisor_override has no ONNX equivalent"
        )
    spatial = ctx.rank(ctx.x) - 2
    attrs = _pool_attrs(ctx, spatial)
    attrs["count_include_pad"] = 1 if ctx.get("count_include_pad", True) else 0
    return ctx.op("AveragePool", [ctx.x], **attrs)


def _adaptive_pool(ctx: OpContext, global_op: str, pool_op: str) -> str:
    spatial = ctx.rank(ctx.x) - 2
    output_size = _pair_attr(ctx.get("output_size"), spatial)
    if all(size == 1 for size in output_size):
        return ctx.op(global_op, [ctx.x])
    shape = ctx.shape(ctx.x)
    spatial_shape = [int(size) for size in shape[2:]]
    if any(
        size % out for size, out in zip(spatial_shape, output_size)
    ):  # pragma: no cover - only evenly tiled windows are representable
        raise UnsupportedOperatorError(
            "adaptive pooling only exports when the input divides the output size "
            f"evenly (input {spatial_shape}, output {output_size})"
        )
    strides = [size // out for size, out in zip(spatial_shape, output_size)]
    return ctx.op(
        pool_op,
        [ctx.x],
        kernel_shape=strides,
        strides=strides,
        pads=[0] * (2 * spatial),
    )


@register("adaptive_avg_pool1d", "input output_size")
@register("adaptive_avg_pool2d", "input output_size")
@register("adaptive_avg_pool3d", "input output_size")
def _handle_adaptive_avg_pool(ctx: OpContext) -> str:
    return _adaptive_pool(ctx, "GlobalAveragePool", "AveragePool")


@register("adaptive_max_pool1d", "input output_size")
@register("adaptive_max_pool2d", "input output_size")
@register("adaptive_max_pool3d", "input output_size")
def _handle_adaptive_max_pool(ctx: OpContext) -> str:
    return _adaptive_pool(ctx, "GlobalMaxPool", "MaxPool")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def _affine_or_default(ctx: OpContext, value: Any, size: int, fill: float, hint: str) -> Any:
    if value is not None:
        return value
    dtype = _float_dtype(ctx, ctx.x)
    return ctx.b.constant(
        np.full((size,), fill, dtype=dtype), name_hint=f"{ctx.node_name}_{hint}"
    )


@register("batch_norm", "input running_mean running_var weight bias training momentum eps")
def _handle_batch_norm(ctx: OpContext) -> str:
    channels = ctx.dim_size(ctx.x, 1)
    scale = _affine_or_default(ctx, ctx.get("weight"), channels, 1.0, "scale")
    offset = _affine_or_default(ctx, ctx.get("bias"), channels, 0.0, "offset")
    mean = _affine_or_default(ctx, ctx.get("running_mean"), channels, 0.0, "mean")
    variance = _affine_or_default(ctx, ctx.get("running_var"), channels, 1.0, "var")
    epsilon = float(ctx.get("eps", 1e-5))
    if ctx.get("training", False):
        ctx.b.require_opset(14, "batch_norm in training mode")
        outputs = ctx.op(
            "BatchNormalization",
            [ctx.x, scale, offset, mean, variance],
            num_outputs=3,
            epsilon=epsilon,
            momentum=1.0 - float(ctx.get("momentum", 0.1)),
            training_mode=1,
        )
        return outputs[0]
    return ctx.op(
        "BatchNormalization",
        [ctx.x, scale, offset, mean, variance],
        epsilon=epsilon,
        momentum=1.0 - float(ctx.get("momentum", 0.1)),
    )


@register("layer_norm", "input normalized_shape weight bias eps")
def _handle_layer_norm(ctx: OpContext) -> str:
    normalized_shape = _as_int_list(ctx.get("normalized_shape"))
    axis = -len(normalized_shape)
    size = int(np.prod(normalized_shape))
    epsilon = float(ctx.get("eps", 1e-5))
    weight = ctx.get("weight")
    bias = ctx.get("bias")
    if ctx.b.opset >= 17:
        scale = weight if weight is not None else ctx.b.constant(
            np.ones(normalized_shape, dtype=_float_dtype(ctx, ctx.x)),
            name_hint=f"{ctx.node_name}_scale",
        )
        inputs = [ctx.x, scale] + ([bias] if bias is not None else [])
        outputs = ctx.op(
            "LayerNormalization", inputs, num_outputs=1, axis=axis, epsilon=epsilon
        )
        return outputs
    dtype = _float_dtype(ctx, ctx.x)
    axes = list(range(axis, 0))
    mean = _reduce(ctx, "ReduceMean", ctx.x, axes, True, axes_input_since=18)
    centered = ctx.op("Sub", [ctx.x, mean])
    variance = _reduce(
        ctx, "ReduceMean", ctx.op("Mul", [centered, centered]), axes, True,
        axes_input_since=18,
    )
    denominator = ctx.op(
        "Sqrt", [ctx.op("Add", [variance, _scalar(ctx, epsilon, dtype, "eps")])]
    )
    result: Any = ctx.op("Div", [centered, denominator])
    if weight is not None:
        result = ctx.op("Mul", [result, weight])
    if bias is not None:
        result = ctx.op("Add", [result, bias])
    return result


@register("group_norm", "input num_groups weight bias eps")
def _handle_group_norm(ctx: OpContext) -> str:
    groups = int(ctx.get("num_groups"))
    rank = ctx.rank(ctx.x)
    channels = ctx.dim_size(ctx.x, 1)
    dtype = _float_dtype(ctx, ctx.x)
    epsilon = float(ctx.get("eps", 1e-5))
    original_shape = ctx.op("Shape", [ctx.x])
    grouped = _reshape(ctx, ctx.x, [0, groups, -1])
    mean = _reduce(ctx, "ReduceMean", grouped, [2], True, axes_input_since=18)
    centered = ctx.op("Sub", [grouped, mean])
    variance = _reduce(
        ctx, "ReduceMean", ctx.op("Mul", [centered, centered]), [2], True,
        axes_input_since=18,
    )
    denominator = ctx.op(
        "Sqrt", [ctx.op("Add", [variance, _scalar(ctx, epsilon, dtype, "eps")])]
    )
    normalized = ctx.op("Div", [centered, denominator])
    result: Any = ctx.op("Reshape", [normalized, original_shape])
    affine_shape = [channels] + [1] * (rank - 2)
    weight, bias = ctx.get("weight"), ctx.get("bias")
    if weight is not None:
        result = ctx.op("Mul", [result, _reshape(ctx, weight, affine_shape)])
    if bias is not None:
        result = ctx.op("Add", [result, _reshape(ctx, bias, affine_shape)])
    return result


@register(
    "instance_norm",
    "input running_mean running_var weight bias use_input_stats momentum eps",
)
def _handle_instance_norm(ctx: OpContext) -> str:
    if not ctx.get("use_input_stats", True):
        raise UnsupportedOperatorError(
            "instance_norm with running statistics has no ONNX equivalent"
        )
    channels = ctx.dim_size(ctx.x, 1)
    scale = _affine_or_default(ctx, ctx.get("weight"), channels, 1.0, "scale")
    offset = _affine_or_default(ctx, ctx.get("bias"), channels, 0.0, "offset")
    return ctx.op(
        "InstanceNormalization",
        [ctx.x, scale, offset],
        epsilon=float(ctx.get("eps", 1e-5)),
    )


@register("local_response_norm", "input size alpha beta k")
def _handle_local_response_norm(ctx: OpContext) -> str:
    size = int(ctx.get("size"))
    if size % 2 == 0:
        raise UnsupportedOperatorError(
            f"ONNX LRN requires an odd window size, got {size}"
        )
    # ONNX LRN already divides the window sum by ``size``, so alpha passes
    # through unscaled.
    return ctx.op(
        "LRN",
        [ctx.x],
        size=size,
        alpha=float(ctx.get("alpha", 1e-4)),
        beta=float(ctx.get("beta", 0.75)),
        bias=float(ctx.get("k", 1.0)),
    )


@register("embedding", "input weight padding_idx max_norm norm_type scale_grad_by_freq sparse")
def _handle_embedding(ctx: OpContext) -> str:
    if ctx.get("max_norm") is not None:
        raise UnsupportedOperatorError("embedding with max_norm has no ONNX equivalent")
    indices = _cast(ctx, ctx.x, np.int64)
    return ctx.op("Gather", [ctx.get("weight"), indices], axis=0)


@register("normalize", "input p dim eps")
def _handle_normalize(ctx: OpContext) -> str:
    dim = int(ctx.get("dim", 1))
    p = float(ctx.get("p", 2.0))
    eps = float(ctx.get("eps", 1e-12))
    denominator = _p_norm(ctx, ctx.x, p, [dim], True)
    clipped = ctx.op(
        "Clip", [denominator, _scalar(ctx, eps, _float_dtype(ctx, ctx.x), "eps")]
    )
    return ctx.op("Div", [ctx.x, clipped])


# ---------------------------------------------------------------------------
# Resizing and padding
# ---------------------------------------------------------------------------

_RESIZE_MODES = {
    "nearest": "nearest",
    "nearest-exact": "nearest",
    "linear": "linear",
    "bilinear": "linear",
    "trilinear": "linear",
    "bicubic": "cubic",
    "area": "linear",
}


@register(
    "interpolate",
    "input size scale_factor mode align_corners recompute_scale_factor antialias",
)
def _handle_interpolate(ctx: OpContext) -> str:
    mode = str(ctx.get("mode", "nearest") or "nearest")
    if mode not in _RESIZE_MODES:
        raise UnsupportedOperatorError(f"interpolate mode {mode!r} is not supported")
    if mode == "area":
        raise UnsupportedOperatorError("interpolate(mode='area') has no ONNX equivalent")
    align_corners = bool(ctx.get("align_corners") or False)
    rank = ctx.rank(ctx.x)
    spatial = rank - 2
    if align_corners:
        coordinate_mode = "align_corners"
    elif mode.startswith("nearest"):
        coordinate_mode = "asymmetric"
    else:
        coordinate_mode = "pytorch_half_pixel"
    attrs: dict[str, Any] = {
        "mode": _RESIZE_MODES[mode],
        "coordinate_transformation_mode": coordinate_mode,
    }
    if _RESIZE_MODES[mode] == "nearest":
        attrs["nearest_mode"] = "floor"
    size = ctx.get("size")
    scale_factor = ctx.get("scale_factor")
    empty = ctx.b.constant(np.zeros((0,), dtype=np.float32), name_hint="resize_roi")
    if size is not None:
        sizes = _pair_attr(size, spatial)
        shape = ctx.shape(ctx.x)
        target = [int(shape[0]), int(shape[1])] + sizes
        sizes_name = ctx.b.int64_1d(target, f"{ctx.node_name}_sizes")
        empty_scales = ctx.b.constant(
            np.zeros((0,), dtype=np.float32), name_hint="resize_scales"
        )
        return ctx.op("Resize", [ctx.x, empty, empty_scales, sizes_name], **attrs)
    if scale_factor is None:
        raise UnsupportedOperatorError("interpolate needs size or scale_factor")
    raw = list(scale_factor) if isinstance(scale_factor, (list, tuple)) else [scale_factor]
    if len(raw) == 1:
        raw = raw * spatial
    if len(raw) != spatial:
        raise UnsupportedOperatorError(
            f"interpolate expects {spatial} scale factors, got {raw}"
        )
    factors = [float(item) for item in raw]
    scales = ctx.b.constant(
        np.asarray([1.0, 1.0] + factors, dtype=np.float32),
        name_hint=f"{ctx.node_name}_scales",
    )
    return ctx.op("Resize", [ctx.x, empty, scales], **attrs)


_PAD_MODES = {
    "constant": "constant",
    "reflect": "reflect",
    "replicate": "edge",
    "circular": "wrap",
}


@register("pad", "input pad mode value")
def _handle_pad(ctx: OpContext) -> str:
    mode = str(ctx.get("mode", "constant") or "constant")
    if mode not in _PAD_MODES:
        raise UnsupportedOperatorError(f"pad mode {mode!r} is not supported")
    if mode == "circular":
        ctx.b.require_opset(19, "pad(mode='circular')")
    amounts = _as_int_list(ctx.get("pad"))
    if len(amounts) % 2:
        raise UnsupportedOperatorError("pad expects pairs of (begin, end) values")
    rank = ctx.rank(ctx.x)
    begins = [0] * rank
    ends = [0] * rank
    # The padding list starts at the LAST dimension, in (begin, end) pairs.
    for index in range(len(amounts) // 2):
        axis = rank - 1 - index
        begins[axis] = amounts[2 * index]
        ends[axis] = amounts[2 * index + 1]
    pads = ctx.b.int64_1d(begins + ends, f"{ctx.node_name}_pads")
    inputs: list[Any] = [ctx.x, pads]
    value = ctx.get("value", 0)
    if mode == "constant" and value is not None:
        inputs.append(ctx.cast_like(float(value), ctx.x))
    return ctx.op("Pad", inputs, mode=_PAD_MODES[mode])


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def _apply_reduction(ctx: OpContext, value: str, reduction: str) -> str:
    if reduction == "none":
        return value
    if reduction == "sum":
        return _reduce_sum(ctx, value, None, False)
    return _reduce(ctx, "ReduceMean", value, None, False, axes_input_since=18)


@register("mse_loss", "input target reduction")
def _handle_mse_loss(ctx: OpContext) -> str:
    difference = ctx.op("Sub", [ctx.x, ctx.get("target")])
    squares = ctx.op("Mul", [difference, difference])
    return _apply_reduction(ctx, squares, str(ctx.get("reduction", "mean") or "mean"))


@register("l1_loss", "input target size_average reduce reduction weight")
def _handle_l1_loss(ctx: OpContext) -> str:
    difference = ctx.op("Abs", [ctx.op("Sub", [ctx.x, ctx.get("target")])])
    return _apply_reduction(ctx, difference, str(ctx.get("reduction", "mean") or "mean"))


@register(
    "cross_entropy",
    "input target weight size_average ignore_index reduce reduction label_smoothing",
)
def _handle_cross_entropy(ctx: OpContext) -> str:
    if float(ctx.get("label_smoothing", 0.0) or 0.0) != 0.0:
        raise UnsupportedOperatorError(
            "cross_entropy with label_smoothing has no ONNX equivalent"
        )
    target = ctx.get("target")
    dtype = ctx.dtype(target)
    if dtype is not None and dtype.kind == "f":
        raise UnsupportedOperatorError(
            "cross_entropy with probability targets has no ONNX equivalent"
        )
    inputs: list[Any] = [ctx.x, _cast(ctx, target, np.int64)]
    weight = ctx.get("weight")
    if weight is not None:
        inputs.append(weight)
    return ctx.op(
        "SoftmaxCrossEntropyLoss",
        inputs,
        reduction=str(ctx.get("reduction", "mean") or "mean"),
        ignore_index=int(ctx.get("ignore_index", -100)),
    )


@register("nll_loss", "input target weight size_average ignore_index reduce reduction")
def _handle_nll_loss(ctx: OpContext) -> str:
    inputs: list[Any] = [ctx.x, _cast(ctx, ctx.get("target"), np.int64)]
    weight = ctx.get("weight")
    if weight is not None:
        inputs.append(weight)
    return ctx.op(
        "NegativeLogLikelihoodLoss",
        inputs,
        reduction=str(ctx.get("reduction", "mean") or "mean"),
        ignore_index=int(ctx.get("ignore_index", -100)),
    )


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@register("getitem", "input index", methods=False)
def _handle_getitem(ctx: OpContext) -> Any:
    data = ctx.x
    key = ctx.get("index")
    if isinstance(data, (list, tuple)):
        return data[int(key)]
    keys = key if isinstance(key, tuple) else (key,)
    rank = ctx.rank(data)
    explicit = sum(1 for item in keys if item is not None and item is not Ellipsis)
    expanded: list[Any] = []
    for item in keys:
        if item is Ellipsis:
            expanded.extend([slice(None)] * (rank - explicit))
        else:
            expanded.append(item)
    starts: list[int] = []
    ends: list[int] = []
    axes: list[int] = []
    steps: list[int] = []
    squeeze_axes: list[int] = []
    unsqueeze_axes: list[int] = []
    gathers: list[tuple[int, Any]] = []
    axis = 0
    output_axis = 0
    for item in expanded:
        if item is None:
            unsqueeze_axes.append(output_axis)
            output_axis += 1
            continue
        if isinstance(item, slice):
            if item != slice(None):
                start = 0 if item.start is None else int(item.start)
                stop = _INT64_MAX if item.stop is None else int(item.stop)
                step = 1 if item.step is None else int(item.step)
                starts.append(start)
                ends.append(stop)
                axes.append(axis)
                steps.append(step)
            axis += 1
            output_axis += 1
            continue
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            index = int(item)
            starts.append(index)
            ends.append(_INT64_MAX if index == -1 else index + 1)
            axes.append(axis)
            steps.append(1)
            squeeze_axes.append(axis)
            axis += 1
            continue
        gathers.append((axis, item))
        axis += 1
        output_axis += 1
    result: Any = data
    if axes:
        result = _slice(ctx, result, starts, ends, axes, steps)
    for gather_axis, index in gathers:
        result = ctx.op(
            "Gather", [result, _cast(ctx, index, np.int64)], axis=gather_axis
        )
    if squeeze_axes:
        result = _squeeze(ctx, result, sorted(squeeze_axes))
    if unsqueeze_axes:
        result = _unsqueeze(ctx, result, sorted(unsqueeze_axes))
    return result
