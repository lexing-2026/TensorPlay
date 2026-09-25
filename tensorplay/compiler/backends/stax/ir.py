"""Value types, node attributes, and the AOT graph builder.

Everything the other layers share lives here: graph helpers, the
symbolic Tensor values the reverse pass is built from, and the buffer
that turns buffered elementwise instructions into native nodes.
"""
from __future__ import annotations

import numbers
import operator
import typing
from typing import Any

from ....graph import GraphModule, Node

if typing.TYPE_CHECKING:
    from .aot_autograd import _AotNativeGraphBuilder


def _consume_template(template: Any, outputs: list[Any], index: int) -> tuple[Any, int]:
    kind = template[0]
    if kind in {"leaf", "tensor"}:
        if index >= len(outputs):
            raise RuntimeError("native graph produced too few outputs")
        return outputs[index], index + 1
    if kind == "nested":
        return _consume_template(template[1], outputs, index)
    if kind == "tuple":
        result = []
        for item in template[1]:
            value, index = _consume_template(item, outputs, index)
            result.append(value)
        return tuple(result), index
    if kind == "list":
        result = []
        for item in template[1]:
            value, index = _consume_template(item, outputs, index)
            result.append(value)
        return result, index
    if kind == "dict":
        result = {}
        for key, item in template[1]:
            value, index = _consume_template(item, outputs, index)
            result[key] = value
        return result, index
    raise RuntimeError(f"unknown native output template {kind!r}")


def _nodes(value: Any):
    if isinstance(value, Node):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _nodes(item)
    elif isinstance(value, slice):
        yield from _nodes(value.start)
        yield from _nodes(value.stop)
        yield from _nodes(value.step)

def _target_name(target: Any) -> str:
    if target is operator.add:
        return "add"
    if target is operator.sub:
        return "sub"
    if target is operator.mul:
        return "mul"
    if target is operator.truediv:
        return "div"
    if target is operator.pow:
        return "pow"
    if target is operator.matmul:
        return "matmul"
    if target is operator.neg:
        return "neg"
    if target is operator.pos:
        return "pos"
    return getattr(target, "__name__", str(target))

def _is_scalar(value: Any) -> bool:
    return isinstance(value, (bool, int, float))

def _set_scalar_attr(native_node: Any, value: Any, position: int) -> None:
    if isinstance(value, bool) or isinstance(value, int):
        native_node.set_int_attr("scalar_value", int(value))
    elif isinstance(value, numbers.Real):
        native_node.set_float_attr("scalar_value", float(value))
    else:
        raise TypeError(f"unsupported Stax scalar constant: {type(value)!r}")
    native_node.set_int_attr("scalar_position", position)

def _set_named_scalar_attr(native_node: Any, key: str, value: Any) -> bool:
    if isinstance(value, bool) or isinstance(value, int):
        native_node.set_int_attr(key, int(value))
    elif isinstance(value, numbers.Real):
        native_node.set_float_attr(key, float(value))
    else:
        return False
    return True

def _int_list(value: Any) -> list[int] | None:
    """Return a constant integer list accepted by a native Stax node."""

    if not isinstance(value, (tuple, list)):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return [int(item) for item in value]

def _spatial_int_list(
    value: Any,
    *,
    default: list[int] | None = None,
    length: int = 2,
) -> list[int] | None:
    """Normalize a scalar or fixed-rank spatial argument."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return [int(value)] * int(length)
    if value is None:
        return None if default is None else list(default)
    result = _int_list(value)
    if result == [] and default is not None:
        return list(default)
    if result is None or len(result) != int(length):
        return None
    return result

def _set_int_list_attr(native_node: Any, key: str, value: Any) -> bool:
    values = _int_list(value)
    if values is None:
        return False
    native_node.set_ints_attr(key, values)
    return True

def _normalize_pointwise_grad_output(grad_output: Any, reference: Any) -> Any:
    """Match the output shape expected by a fused elementwise backward.

    TensorPlay's current reduction backward may hand a scalar tangent to a
    custom Function for ``output.sum().backward()``. The compiled backward
    contract supplies the expanded tangent, so normalize that boundary here
    before entering either the p10 or Triton backward kernel.
    """

    if (
        grad_output.numel() == 1 and reference.numel() != 1
    ) or not grad_output.is_contiguous():
        import tensorplay

        return tensorplay.ones_like(reference, requires_grad=False) * grad_output
    return grad_output

def _attach_fast_call(lowering: Any, exec_fn: Any = None) -> None:
    """Install the C steady-state trampoline for a compiled lowering.

    ``exec_fn`` selects the steady-state execution entry; the default is the
    native ``Graph.execute`` bound method.  Lowerings with a direct kernel
    entry pass it here so the trampoline skips the graph walk.
    """
    import tensorplay

    installer = tensorplay._C._stax.install_call_trampoline
    tail = [
        lowering.graph_module._get_attr(target)
        for target in lowering.attribute_targets
    ]
    tail.extend(lowering.constant_values)
    lowering._fast_call = installer(
        lowering,
        exec_fn if exec_fn is not None else lowering.graph.execute,
        tail,
        tensorplay.Tensor,
        len(lowering.placeholders),
        int(getattr(lowering, "_output_count", 1)),
        getattr(lowering, "_gradient_plan", None) is not None,
        int(getattr(lowering, "_native_direct", 0) or 0),
    )

def _metadata_fingerprint(value: Any) -> Any:
    """Metadata snapshot for tensors outside the version-counter contract.

    Only reached when ``_version`` is unavailable and the tensor is not an
    inference tensor; every component is normalized so the snapshot stays
    comparable across calls.
    """

    shape = getattr(value, "shape", ())
    if callable(shape):
        shape = shape()
    try:
        shape = tuple(int(item) for item in shape)
    except (TypeError, ValueError):
        shape = repr(shape)
    stride = getattr(value, "stride", ())
    if callable(stride):
        stride = stride()
    try:
        stride = tuple(int(item) for item in stride)
    except (TypeError, ValueError):
        stride = repr(stride)
    dtype = getattr(value, "dtype", None)
    if callable(dtype):
        dtype = dtype()
    device = getattr(value, "device", None)
    if callable(device):
        device = device()
    return ("metadata", shape, stride, dtype, device)

class _NativeLowering:
    def __init__(
        self,
        graph_module: GraphModule,
        graph: Any,
        attribute_targets: list[str],
        constant_values: list[Any] | None = None,
        output_count: int = 1,
        native_values: dict[Node, Any] | None = None,
        output_spec: Any = None,
        public_output_count: int | None = None,
        mutations: list[tuple[int, int]] | None = None,
        autocast_outputs: list[tuple[Node, Any]] | None = None,
        saved_result_outputs: list[tuple[Node, int]] | None = None,
    ) -> None:
        self.graph_module = graph_module
        self.graph = graph
        self.placeholders = graph_module.graph.placeholders
        self.attribute_targets = attribute_targets
        self.constant_values = list(constant_values or [])
        self._output_count = output_count
        self._public_output_count = (
            output_count if public_output_count is None else public_output_count
        )
        self._output_spec = output_spec
        # (input position, absolute output index) pairs: the native graph
        # computes buffer updates functionally and the wrapper copies each
        # result back into the source input after every execution.
        self._mutations = list(mutations or [])
        self.native_values = dict(native_values or {})
        self.autocast_outputs = list(autocast_outputs or [])
        # Captured operators whose forward reported values beside its result
        # (an attention normalizer, a normalization's row statistics), with
        # how many each reports, in graph-output order after the converted
        # values.
        self.saved_result_outputs = list(saved_result_outputs or [])
        self._tensorplay_codegen = "stax-native"
        # (id, _version) memo of the last resolved input vector; attributes
        # and constants appended by _bind_inputs are process-stable.
        self._bind_fp: Any = None
        self._last_bound_inputs: list[Any] | None = None
        # Mutating graphs copy results back into module state in Python after
        # execution; the C trampoline bypasses that epilogue, so they keep the
        # interpreted wrapper as their only execution entry.
        if not self._mutations:
            _attach_fast_call(self)

    @staticmethod
    def _input_route_fingerprint(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        import tensorplay

        def fp(value: Any) -> Any:
            if isinstance(value, tensorplay.Tensor):
                try:
                    version = value._version
                except RuntimeError:
                    # Inference tensors carry no version counter and are
                    # immutable, so the identity alone keys the entry: no
                    # metadata snapshot, no divert on the next call.
                    if getattr(value, "is_inference", lambda: False)():
                        return ("t", id(value), None)
                    version = _metadata_fingerprint(value)
                return (
                    "t",
                    id(value),
                    version,
                )
            return ("o", id(value))

        items = [fp(item) for item in args]
        items.extend((k, fp(v)) for k, v in sorted(kwargs.items()))
        return tuple(items)

    def _bind_inputs_fresh(self, *args: Any, **kwargs: Any) -> list[Any]:
        bound = self.graph_module.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        inputs = [
            bound.arguments[node.target if isinstance(node.target, str) else node.name]
            for node in self.placeholders
        ]
        inputs.extend(
            self.graph_module._get_attr(target) for target in self.attribute_targets
        )
        inputs.extend(self.constant_values)
        return inputs

    def _bind_inputs(self, *args: Any, **kwargs: Any) -> list[Any]:
        # Attribute targets and constants are process-stable; user inputs are
        # covered by the (id, _version) fingerprint, so an unchanged call
        # reuses the previously resolved input list without signature binding.
        fp = self._input_route_fingerprint(args, kwargs)
        if fp == self._bind_fp:
            return list(self._last_bound_inputs)
        inputs = self._bind_inputs_fresh(*args, **kwargs)
        self._bind_fp = fp
        self._last_bound_inputs = inputs
        return inputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._bind_inputs(*args, **kwargs)
        outputs = self.graph.execute(inputs)
        for position, output_index in self._mutations:
            inputs[position].copy_(outputs[output_index])
        public_outputs = outputs[: self._public_output_count]
        if self._output_spec is not None:
            value, consumed = _consume_template(self._output_spec, public_outputs, 0)
            if consumed != len(public_outputs):
                raise RuntimeError("native graph produced an unexpected output count")
            return value
        if len(public_outputs) == 1:
            return public_outputs[0]
        return tuple(public_outputs)

def _traced_value(graph_module: GraphModule, node: Node) -> Any:
    """The tensor a node carried when the region was captured, or ``None``."""

    if node.op == "get_attr":
        try:
            return graph_module._get_attr(node.target)
        except (AttributeError, KeyError, RuntimeError):
            return None
    return node.meta.get("val")

def _tensor_layout(value: Any) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    try:
        shape = tuple(int(item) for item in value.shape)
        stride = tuple(int(item) for item in value.stride())
    except (AttributeError, TypeError, ValueError):
        return None
    return shape, stride

class _AotShape(tuple):
    """Tensor metadata that supports both ``shape`` and ``shape()`` schemas."""

    def __new__(cls, value: Any):
        return super().__new__(cls, (int(item) for item in value))

    def __call__(self) -> tuple[int, ...]:
        return tuple(self)

def _shape_argument(shape: tuple[Any, ...]) -> Any:
    """Normalize a shape spelled as extents or as one sequence.

    Reductions in the registered decompositions write the extent form
    (``x.reshape(N, C, HxW)``), while builder callers pass a single tuple.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        return tuple(shape[0])
    return tuple(int(item) if isinstance(item, int) else item for item in shape)


class _AotNativeSymbol:
    """A symbolic Tensor value used while materializing an AOT backward graph."""

    __slots__ = ("builder", "value", "shape", "dtype")

    def __init__(self, builder: "_AotNativeGraphBuilder", value: Any, shape: Any, dtype: Any = None):
        self.builder = builder
        self.value = value
        self.shape = _AotShape(shape)
        self.dtype = dtype

    def _binary(self, op_name: str, other: Any) -> "_AotNativeSymbol":
        return self.builder.binary(op_name, self, other)

    def __add__(self, other: Any) -> "_AotNativeSymbol":
        return self._binary("add", other)

    def __radd__(self, other: Any) -> "_AotNativeSymbol":
        return self.builder.binary("add", other, self)

    def __sub__(self, other: Any) -> "_AotNativeSymbol":
        return self._binary("sub", other)

    def __rsub__(self, other: Any) -> "_AotNativeSymbol":
        return self.builder.binary("sub", other, self)

    def __mul__(self, other: Any) -> "_AotNativeSymbol":
        return self._binary("mul", other)

    def __rmul__(self, other: Any) -> "_AotNativeSymbol":
        return self.builder.binary("mul", other, self)

    def __truediv__(self, other: Any) -> "_AotNativeSymbol":
        return self._binary("div", other)

    def __rtruediv__(self, other: Any) -> "_AotNativeSymbol":
        return self.builder.binary("div", other, self)

    def __neg__(self) -> "_AotNativeSymbol":
        return self.builder.unary("neg", self)

    def __pos__(self) -> "_AotNativeSymbol":
        return self.builder.unary("pos", self)

    def t(self) -> "_AotNativeSymbol":
        return self.builder.unary("t", self, shape=self.shape[::-1])

    def mm(self, other: Any) -> "_AotNativeSymbol":
        return self.builder.binary("mm", self, other)

    def matmul(self, other: Any) -> "_AotNativeSymbol":
        return self.builder.binary("matmul", self, other)

    def reshape(self, *shape: Any) -> "_AotNativeSymbol":
        return self.builder.reshape(self, _shape_argument(shape))

    def view(self, *shape: Any) -> "_AotNativeSymbol":
        return self.builder.reshape(self, _shape_argument(shape))

    def expand(self, shape: Any) -> "_AotNativeSymbol":
        return self.builder.expand(self, shape)

    def unsqueeze(self, dim: int) -> "_AotNativeSymbol":
        return self.builder.unsqueeze(self, dim)

    def squeeze(self, dim: Any = None) -> "_AotNativeSymbol":
        return self.builder.squeeze(self, dim)

    def sum(self, dim: Any = None, keepdim: bool = False) -> "_AotNativeSymbol":
        return self.builder.sum(self, dim, keepdim)

    def to(self, dtype: Any) -> "_AotNativeSymbol":
        return self.builder.cast(self, dtype)

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        result = 1
        for item in self.shape:
            result *= item
        return result

class _AotNativeTuple:
    __slots__ = ("values", "node", "mask")

    def __init__(self, values: tuple[_AotNativeSymbol, ...]):
        self.values = values
        self.node = None
        self.mask = None

# Opcodes the elementwise program buffer can absorb, sharing the numeric
# contract of the stax fused-pointwise evaluators (p10 CPU and CUDA).
_FUSED_PROGRAM_OPCODES = {
    "add": 1,
    "sub": 2,
    "mul": 3,
    "div": 4,
    "neg": 6,
    "pos": 7,
    "abs": 8,
    "sin": 9,
    "cos": 10,
    "exp": 11,
    "log": 12,
    "sigmoid": 13,
    "sqrt": 14,
    "square": 15,
    "tanh": 16,
    "relu": 17,
}

# Ops whose capture-time dispatch narrows tensor inputs to a reduced element
# type before the matrix product.  The reverse pass re-narrows operands of
# these products to the same element type instead of letting every kernel
# widen to the accumulation type.
_AUTOCAST_GEMM_OPS = {
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "linear",
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "addbmm",
    "baddbmm",
    "scaled_dot_product_attention",
}

# Output storage tags carried in the graph attribute: these are the scalar
# type codes the evaluators decode, so an emitted result may narrow or widen
# its float-domain value independently of the program inputs, and a type
# edge rides the producer kernel instead of a separate pass.
_FUSED_OUT_DTYPE_CODES = {"float32": 8, "float64": 9, "float16": 10, "bfloat16": 11}

_DTYPE_WIDTHS = {"float16": 2, "bfloat16": 2, "float32": 4, "float64": 8}

def _dtype_width(dtype: Any) -> int:
    """Byte width of a floating element type; unknown types rank as float32."""
    return _DTYPE_WIDTHS.get(str(dtype).rsplit(".", 1)[-1].lower(), 4)

# Pointwise compute nodes whose derivative expressions are pure arithmetic:
# a gradient edge towards one of them that only widens its element type can
# stay narrow, because the consuming formulas widen operands to a common
# arithmetic width themselves and no separate conversion pass is needed.
_DEFERRABLE_PROMOTION_OPS = frozenset({
    "add", "sub", "mul", "div", "truediv", "neg", "pos", "exp", "log",
    "sqrt", "rsqrt", "exp2", "erf", "square", "abs", "sigmoid", "tanh",
    "relu", "silu", "pow", "clamp", "minimum", "maximum", "dropout",
    "where", "lt", "le", "gt", "ge", "eq", "ne",
})

def _adjoint_promotion_deferrable(target: Node, from_dtype: Any,
                                  to_dtype: Any) -> bool:
    if target.op != "call_function":
        return False
    if _target_name(target.target) not in _DEFERRABLE_PROMOTION_OPS:
        return False
    return _dtype_width(from_dtype) < _dtype_width(to_dtype)

class _AotFusedSpec:
    """A frozen elementwise program kept until every spilled temp is read."""

    __slots__ = ("ops", "temps", "out_dtypes")

    def __init__(self, ops, temps, out_dtypes):
        self.ops = ops
        self.temps = temps
        self.out_dtypes = out_dtypes


#: Names this layer owns.  The driver re-exports them, so the
#: import surface of the package does not change: value types and node attributes shared by every layer.
__all__ = [
    "_AUTOCAST_GEMM_OPS",
    "_AotFusedSpec",
    "_AotNativeSymbol",
    "_AotNativeTuple",
    "_AotShape",
    "_DEFERRABLE_PROMOTION_OPS",
    "_DTYPE_WIDTHS",
    "_FUSED_OUT_DTYPE_CODES",
    "_FUSED_PROGRAM_OPCODES",
    "_NativeLowering",
    "_adjoint_promotion_deferrable",
    "_attach_fast_call",
    "_consume_template",
    "_dtype_width",
    "_int_list",
    "_is_scalar",
    "_metadata_fingerprint",
    "_nodes",
    "_normalize_pointwise_grad_output",
    "_set_int_list_attr",
    "_set_named_scalar_attr",
    "_set_scalar_attr",
    "_spatial_int_list",
    "_target_name",
    "_tensor_layout",
    "_traced_value",
]
