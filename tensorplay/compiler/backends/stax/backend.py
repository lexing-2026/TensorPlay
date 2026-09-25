"""The TensorPlay native graph compiler backend.

This is a compiler backend, not a second public compiler frontend:
``tensorplay.compile`` owns
capture, guards, specialization, and graph-break policy; this module owns
lowering and executable generation for the canonical graph.

small lazy adapter.  Native Stax code is loaded only when the backend is
actually selected, keeping import-time overhead out of the frontend.
"""

from __future__ import annotations

import operator
import numbers
import re
from typing import Any

from ....graph.passes import POINTWISE_FUSED_OP_NAMES
from ....graph import GraphModule, Node
from ....library import CustomOpDef as _CustomOpDef


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


def _native_value_leaves(value: Any, values: dict[Node, Any]) -> list[Any]:
    if isinstance(value, Node):
        native = values.get(value)
        if isinstance(native, tuple):
            return list(native)
        return [] if native is None else [native]
    if isinstance(value, tuple | list):
        result: list[Any] = []
        for item in value:
            result.extend(_native_value_leaves(item, values))
        return result
    if isinstance(value, dict):
        result = []
        for item in value.values():
            result.extend(_native_value_leaves(item, values))
        return result
    if isinstance(value, slice):
        result = []
        for item in (value.start, value.stop, value.step):
            result.extend(_native_value_leaves(item, values))
        return result
    return []


def _native_output_spec(value: Any, values: dict[Node, Any]) -> Any:
    if isinstance(value, Node):
        native = values.get(value)
        if isinstance(native, tuple):
            custom = value.meta.get("custom")
            template = custom.get("nested_output_template") if isinstance(custom, dict) else None
            if template is not None:
                return ("nested", template)
            # Native view operators expose a flat tuple of Tensor values.
            # Their graph output is already the public tuple, so no custom
            # pytree template is needed to rebuild it.
            return ("tuple", tuple(("leaf",) for _ in native))
        if native is None:
            raise RuntimeError(f"native graph has no value for output {value.name!r}")
        return ("leaf",)
    if isinstance(value, tuple):
        return ("tuple", tuple(_native_output_spec(item, values) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_native_output_spec(item, values) for item in value))
    if isinstance(value, dict):
        return (
            "dict",
            tuple((key, _native_output_spec(item, values)) for key, item in value.items()),
        )
    raise RuntimeError("native graph outputs must be tensor values")


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


_NATIVE_OPS = {
    "add",
    "sub",
    "mul",
    "div",
    "pow",
    "matmul",
    "t",
    "linear",
    "neg",
    "pos",
    "abs",
    "sin",
    "cos",
    "exp",
    "log",
    "sigmoid",
    "sqrt",
    "rsqrt",
    "square",
    "tanh",
    "sign",
    "relu",
    "silu",
    "conj",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cosh",
    "erf",
    "erfc",
    "exp2",
    "expm1",
    "floor",
    "log1p",
    "log2",
    "reciprocal",
    "round",
    "sinh",
    "tan",
    "trunc",
    "erfinv",
    "erfcx",
    "lgamma",
    "i0",
    "tanhshrink",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "where",
    "clamp",
    "clamp_min",
    "clamp_max",
    "gelu",
    "softmax",
    "log_softmax",
    "layer_norm",
    "transpose",
    "select",
    "slice",
    "stack",
    "repeat",
    "index_select",
    "gather",
    "embedding",
    "constant_pad_nd",
    "pad",
    "mm",
    # Tensor kernels used by the ResNet inference graph.  These are kept in
    # the native graph instead of falling back to the generated Python
    # executor; the latter still calls every functional wrapper through the
    # interpreter and is not a compiled path in any meaningful sense.
    "conv2d",
    "conv2d_relu",
    "add_relu",
    "batch_norm",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "max_pool1d_with_indices",
    "max_pool2d_with_indices",
    "max_pool3d_with_indices",
    "avg_pool1d",
    "avg_pool3d",
    "adaptive_avg_pool2d",
    "adaptive_avg_pool1d",
    "adaptive_avg_pool3d",
    "adaptive_max_pool1d",
    "adaptive_max_pool2d",
    "adaptive_max_pool3d",
    "flatten",
    "view",
    "reshape",
    "permute",
    "permute_backward",
    "contiguous",
    "unsqueeze",
    "squeeze",
    "zeros_like",
    "float",
    "group_norm",
    "avg_pool2d",
    "interpolate",
    "scaled_dot_product_attention",
    "dropout",
    "sum",
    "mean",
    "cat",
    "expand",
    "chunk",
    "split",
    "split_with_sizes",
    "unbind",
}

# The fused-op name set is shared by the graph pass and this lowering.
_CPU_FUSED_OPS = POINTWISE_FUSED_OP_NAMES

# Opcodes of the CPU fused interpreter (p10 StaxPointwiseKernels switch).
# This is a strict subset of _CPU_FUSED_OPS: graphs whose programs contain
# Triton-only opcodes fall back per lowering instead of reaching the CPU
# program runner.
_CPU_FUSED_OPCODES = {
    "add": 1,
    "sub": 2,
    "mul": 3,
    "div": 4,
    "pow": 5,
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
    "relu_grad": 18,
    "abs_grad": 19,
}

# Triton-only opcode extension.  Emitted programs keep the shared triple
# format; the extra opcodes never reach the CPU interpreter because the CPU
# program builder runs against _CPU_FUSED_OPCODES.
#
# ``where`` is ternary, so it is encoded as a two-instruction pair:
# ``where`` carries (cond, a) and the immediately following ``where_rest``
# carries (cond, b).  The code generator pairs them by adjacency; ``where``
# itself emits no source.  ``cast`` stores the target float-dtype id in its
# rhs operand slot (it has a single value operand).
_TRITON_EXTRA_OPCODES = {
    "lt": 20,
    "le": 21,
    "gt": 22,
    "ge": 23,
    "eq": 24,
    "ne": 25,
    "where": 26,
    "where_rest": 27,
    "minimum": 28,
    "maximum": 29,
    "clamp_min": 30,
    "clamp_max": 31,
    "rsqrt": 32,
    "exp2": 33,
    "erf": 34,
    "cast": 35,
}

_TRITON_OPCODES = dict(_CPU_FUSED_OPCODES, **_TRITON_EXTRA_OPCODES)

# Cast targets accepted by the fused program, keyed by ``str(dtype)``.
# Non-float casts (bool/int) stay uncompiled for now: the program's value
# space is float and the store path types outputs off the sample dtype.
_CAST_DTYPE_IDS = {
    "tensorplay.float16": 1,
    "tensorplay.bfloat16": 2,
    "tensorplay.float32": 3,
    "tensorplay.float64": 4,
}

_STAX_CUDA_OP_COUNTER = 0
_STAX_CUDA_OP_CACHE: dict[Any, str] = {}

# Backward-compatibility alias: the autograd gate keeps covering exactly the
# ops whose elementwise VJP rules exist (the CPU interpreter's surface minus
# pow).  Triton-only opcodes stay out, so training graphs using them fall
# back instead of producing a wrong gradient.
_CPU_FUSED_AUTOGRAD_OPS = frozenset(_CPU_FUSED_OPCODES) - {"pow"}


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


class _CpuFusedPointwiseLowering(_NativeLowering):
    """Executable wrapper for Stax's vectorized CPU expression kernel."""

    def __init__(
        self,
        graph_module: GraphModule,
        graph: Any,
        attribute_targets: list[str],
        expected_shape: tuple[int, ...],
        expected_dtype: Any,
        expected_device: Any,
        gradient_plan: tuple[list[int], list[float], tuple[int, ...]] | None = None,
        strict_native: bool = False,
        native_runner: Any = None,
        native_direct: int = 0,
        expected_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
        | None = None,
    ) -> None:
        super().__init__(graph_module, graph, attribute_targets)
        self._expected_shape = expected_shape
        self._expected_dtype = expected_dtype
        self._expected_device = expected_device
        # Per-input (shape, strides) pinned at lowering time; set only for
        # broadcast/strided specializations whose generated addressing is
        # valid for exactly these layouts.
        self._expected_layouts = expected_layouts
        self._gradient_plan = gradient_plan
        self._strict_native = strict_native
        self._tensorplay_codegen = "stax-fused-cpu"
        self._autograd_function: Any | None = None
        # Runtime-generated C kernel for the native route: straight-line
        # compiler-scheduled code replacing the program interpreter when the
        # system compiler is available.  None keeps the graph-execute path.
        self._native_runner = native_runner
        # Address of the kernel's pointer-level entry (0 = absent); the C
        # steady-state trampoline reads it for the direct launch path.
        self._native_direct = int(native_direct) if native_direct else 0
        # Route memo (id, _version, requires_grad) per input: eligibility and
        # autograd routing are pure functions of these, so steady-state calls
        # skip the per-input shape/dtype/device/contiguity probes entirely.
        # In-place mutation bumps _version; fresh tensors have fresh ids.
        self._route_fp: tuple[Any, ...] | None = None
        self._route: str | None = None
        _attach_fast_call(self, exec_fn=self._native_runner)
        if gradient_plan is not None:
            from ....autograd import Function

            lowering = self

            class _FusedPointwiseAutograd(Function):
                @staticmethod
                def forward(ctx: Any, *forward_inputs: Any) -> Any:
                    ctx.save_for_backward(*forward_inputs)
                    return lowering._execute_inputs(list(forward_inputs))

                @staticmethod
                def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
                    grad_output = grad_outputs[0] if grad_outputs else None
                    if grad_output is None:
                        return (None,) * len(ctx.saved_tensors)
                    gradients = lowering._execute_backward(
                        ctx.saved_tensors,
                        grad_output,
                    )
                    return gradients

            self._autograd_function = _FusedPointwiseAutograd

    @staticmethod
    def _eligible_inputs(
        inputs: list[Any],
        expected_shape: tuple[int, ...],
        expected_dtype: Any,
        expected_device: Any,
        expected_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
        | None = None,
    ) -> bool:
        try:
            import tensorplay

            tensor_type = tensorplay.Tensor
        except (AttributeError, ImportError):
            return False
        if expected_layouts is not None:
            if len(inputs) != len(expected_layouts):
                return False
            # Broadcast/strided specializations pin each input's exact
            # (shape, strides): the generated addressing was proven for
            # that layout, so any deviation must re-lower.
            # The expected device is captured from the region's own sample
            # inputs, so comparing against it is what pins the device.
            return bool(inputs) and all(
                isinstance(value, tensor_type)
                and value.dtype == expected_dtype
                and value.device == expected_device
                and tuple(int(item) for item in value.shape) == layout[0]
                and tuple(int(item) for item in value.stride()) == layout[1]
                for value, layout in zip(inputs, expected_layouts)
            )
        return bool(inputs) and all(
            isinstance(value, tensor_type)
            and value.dtype == expected_dtype
            and value.device == expected_device
            and tuple(int(item) for item in value.shape) == expected_shape
            and value.is_contiguous()
            for value in inputs
        )

    def _execute_inputs(self, inputs: list[Any]) -> Any:
        if self._native_runner is not None:
            return self._native_runner(inputs)
        outputs = self.graph.execute(inputs)
        if len(outputs) != 1:
            return tuple(outputs)
        return outputs[0]

    def _execute_backward(
        self,
        inputs: tuple[Any, ...],
        grad_output: Any,
    ) -> tuple[Any, ...]:
        gradients = []
        if self._gradient_plan is None:
            raise RuntimeError("Stax fused pointwise backward plan is missing")
        import tensorplay

        grad_output = _normalize_pointwise_grad_output(grad_output, inputs[0])
        program, constants, output_refs = self._gradient_plan
        gradients = tensorplay._C._stax.execute_fused_pointwise_multi(
            [*inputs, grad_output],
            program,
            constants,
            output_refs,
        )
        return tuple(gradients)

    @staticmethod
    def _input_route_fingerprint(value: Any) -> Any:
        import tensorplay

        if isinstance(value, tensorplay.Tensor):
            try:
                version = value._version
            except RuntimeError:
                # Inference tensors are immutable: the identity alone keys
                # the entry, no metadata snapshot needed.
                if getattr(value, "is_inference", lambda: False)():
                    return ("t", id(value), None)
                version = _metadata_fingerprint(value)
            return (
                "t",
                id(value),
                version,
                bool(getattr(value, "requires_grad", False)),
            )
        return ("o", id(value))

    def _resolve_route(self, inputs: list[Any]) -> str:
        if not self._eligible_inputs(
            inputs,
            self._expected_shape,
            self._expected_dtype,
            self._expected_device,
            getattr(self, "_expected_layouts", None),
        ):
            return "fallback"
        if self._gradient_plan is not None and any(
            value.requires_grad for value in inputs
        ):
            return "autograd"
        return "native"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not kwargs and len(args) == len(self.placeholders):
            inputs = list(args)
        else:
            bound = self.graph_module.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            inputs = [
                bound.arguments[node.target if isinstance(node.target, str) else node.name]
                for node in self.placeholders
            ]
        fp = tuple(self._input_route_fingerprint(value) for value in inputs)
        if fp != self._route_fp:
            self._route = self._resolve_route(inputs)
            self._route_fp = fp
        route = self._route
        if route == "fallback":
            raise RuntimeError(
                "Stax fused CPU lowering received inputs outside its "
                "compiled specialization"
            )
        if route == "autograd":
            if self._autograd_function is None:
                raise RuntimeError("Stax fused pointwise autograd function is missing")
            return self._autograd_function.apply(*inputs)
        return self._execute_inputs(inputs)


class _CudaFusedPointwiseLowering:
    def __init__(
        self,
        graph_module: GraphModule,
        runner: Any,
        expected_dtype: Any,
        expected_device: Any,
        expected_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        strict_native: bool = False,
    ) -> None:
        self.graph_module = graph_module
        self.placeholders = graph_module.graph.placeholders
        self._runner = runner
        self._expected_dtype = expected_dtype
        self._expected_device = expected_device
        self._expected_layouts = expected_layouts
        self._strict_native = strict_native
        self._route_fp: tuple[Any, ...] | None = None
        self._route: str | None = None
        self._fallback = None if strict_native else graph_module.recompile()
        self._tensorplay_codegen = "stax-cuda"

    def _resolve_route(self, inputs: list[Any]) -> str:
        import tensorplay

        if len(inputs) != len(self._expected_layouts):
            return "fallback"
        for value, (shape, stride) in zip(inputs, self._expected_layouts):
            if (
                not isinstance(value, tensorplay.Tensor)
                or not value.device.is_cuda()
                or value.dtype != self._expected_dtype
                or value.device != self._expected_device
                or tuple(int(item) for item in value.shape) != shape
                or tuple(int(item) for item in value.stride()) != stride
                or value.requires_grad
            ):
                return "fallback"
        return "native"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not kwargs and len(args) == len(self.placeholders):
            inputs = list(args)
        else:
            bound = self.graph_module.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            inputs = [
                bound.arguments[node.target if isinstance(node.target, str) else node.name]
                for node in self.placeholders
            ]
        fingerprint = tuple(
            _CpuFusedPointwiseLowering._input_route_fingerprint(value)
            for value in inputs
        )
        if fingerprint != self._route_fp:
            self._route = self._resolve_route(inputs)
            self._route_fp = fingerprint
        if self._route == "fallback":
            if self._strict_native:
                raise RuntimeError(
                    "Stax native CUDA lowering received inputs outside its "
                    "compiled specialization"
                )
            assert self._fallback is not None
            return self._fallback(*args, **kwargs)
        return self._runner(inputs)


# Generated Triton kernels expand every program instruction into straight-line
# source, removing the interpreter's per-element dispatch/fetch round trip,
# which is what limits deep fused chains to arithmetic latency.  Below ~1k
# elements the balance flips: a single C launch beats the Python-side
# dispatch, and the GPU time is negligible either way.  Everything larger
# goes to the generated kernel even when tiny -- the jiterator wrapper pays a
# fixed per-call broadcast materialization that dwarfs both launch styles.
_TRITON_POINTWISE_MIN_NUMEL = 1 << 10


def _cuda_triton_pointwise_runner(
    program: list[int],
    constants: list[float],
    output_refs: tuple[int, ...],
    example_inputs: list[Any],
    output_dtypes: tuple[str, ...] | None = None,
) -> Any | None:
    """Lower one flat pointwise program to a generated Triton kernel.

    Returns a runner honouring the interpreter's contract (input list in,
    output tensor(s) out), or None whenever Triton is unavailable, the sample
    tensors disagree on shape, or generation/compilation fails -- every miss
    stays on the interpreter route unchanged.  ``output_dtypes`` opts the
    plan into per-input dtype freedom: loads promote through each input's
    own dtype and stores narrow to the named output dtype, instead of the
    single shared dtype the historic contract assumes.
    """
    try:
        from .codegen.triton import runtime_available
    except ImportError:
        return None
    if not runtime_available():
        return None
    shapes = tuple(
        tuple(int(item) for item in value.shape) for value in example_inputs
    )
    # The program's output spans the aligned broadcast of the operands, so
    # the reference shape is their elementwise maximum, not necessarily the
    # first operand's shape (a per-channel bias may lead the feed order).
    rank = max(len(shape) for shape in shapes)
    reference = tuple(
        max(
            (1,) * (rank - len(shape)) + shape
            for shape in shapes
        )[dim]
        for dim in range(rank)
    )
    broadcast = any(shape != reference for shape in shapes)
    numel = 1
    for dim in reference:
        numel *= dim
    if numel < _TRITON_POINTWISE_MIN_NUMEL:
        return None
    try:
        from .codegen.triton import _autotune_launch

        runner = _autotune_launch(
            "pw-plan",
            program,
            constants,
            tuple(output_refs),
            example_inputs,
            input_shapes=shapes if broadcast else None,
            reference_shape=reference,
            bucket_numel=numel,
            output_dtypes=output_dtypes,
        )
        runner(example_inputs)
    except Exception:  # noqa: BLE001 - any miss keeps the interpreter route
        return None
    return runner


def _lower_cuda_fused_pointwise(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    strict_native: bool = False,
    dynamic: bool = False,
) -> _CudaFusedPointwiseLowering | None:
    if dynamic:
        return None
    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if not example_inputs or len(example_inputs) > 32:
        return None
    if any(not isinstance(value, tensor_type) for value in example_inputs):
        return None
    first = example_inputs[0]
    if (
        not first.device.is_cuda()
        or first.dtype
        not in (
            tensorplay.float16,
            tensorplay.bfloat16,
            tensorplay.float32,
            tensorplay.float64,
        )
        or any(value.requires_grad for value in example_inputs)
    ):
        return None
    if any(
        value.device != first.device or value.dtype != first.dtype
        for value in example_inputs[1:]
    ):
        return None
    pointwise = _build_pointwise_program(graph_module)
    if pointwise is None:
        return None
    external_nodes, program, constants, _instructions, output_ref = pointwise
    if len(external_nodes) != len(example_inputs):
        return None
    runner = _cuda_triton_pointwise_runner(
        program, constants, (output_ref,), example_inputs
    )
    if runner is None:
        try:
            from .codegen.cuda import compile_program

            runner = compile_program(
                program,
                constants,
                (output_ref,),
                len(external_nodes),
                example_inputs,
            )
            runner(example_inputs)
        except (AssertionError, RuntimeError, TypeError, ValueError):
            return None
    layouts = tuple(
        (
            tuple(int(item) for item in value.shape),
            tuple(int(item) for item in value.stride()),
        )
        for value in example_inputs
    )
    return _CudaFusedPointwiseLowering(
        graph_module,
        runner,
        first.dtype,
        first.device,
        layouts,
        strict_native,
    )


_ARITHMETIC_OPS = frozenset({"add", "sub", "mul", "div", "pow"})
_COMPARISON_OPS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})
_ORDER_OPS = frozenset({"minimum", "maximum", "clamp_min", "clamp_max"})
_UNARY_OPS = frozenset(
    {
        "neg",
        "pos",
        "abs",
        "sin",
        "cos",
        "exp",
        "log",
        "sigmoid",
        "sqrt",
        "square",
        "tanh",
        "relu",
        "rsqrt",
        "exp2",
        "erf",
    }
)
_CAST_METHOD_DTYPES = {
    "float": "tensorplay.float32",
    "half": "tensorplay.float16",
    "double": "tensorplay.float64",
}


def _build_pointwise_program(
    graph_module: GraphModule,
    *,
    skip_node: Node | None = None,
    output_override: Node | None = None,
    allow_empty: bool = False,
    opcodes: dict[str, int] | None = None,
    nodes: list[Node] | None = None,
    extra_refs: dict[Node, int] | None = None,
    input_slots: int | None = None,
    constants: list[float] | None = None,
    extra_outputs: list[Node] | None = None,
) -> tuple[list[Node], list[int], list[float], list[tuple[str, int, int, int]], int] | None:
    """Encode one canonical pointwise graph as Stax's postfix program.

    ``skip_node`` excludes one node from the program (used by the Triton
    reduction-epilogue path, which folds a full-reduction ``sum`` tail into
    the kernel instead of lowering it as an op), with ``output_override``
    naming the program's result node.

    ``nodes`` narrows the walk to one ordered slice of the region, with
    ``extra_refs`` naming values the slice may read but does not compute and
    ``input_slots`` widening the reference space those extra names live in.
    A caller passing ``constants`` accumulates into it, so several slices of
    one region share a single constant pool and a single reference space.

    ``opcodes`` selects the target opcode table: the CPU fused interpreter
    supports the base table only, while the Triton code generator accepts
    the extended surface (comparisons, ``where``, order relations, casts).
    Values are typed numerically — comparisons yield booleans, everything
    else yields floats, and the program output must be a float value (the
    store path types outputs off the sample dtype).

    ``extra_outputs`` names additional result nodes the kernel stores
    alongside the main one (horizontal fusion).  When given, the return
    grows a sixth element: a tuple of refs for those nodes, in call order —
    each becomes one more ``out_ptr`` at the shared reference shape.
    """

    table = _TRITON_OPCODES if opcodes is None else opcodes

    external_nodes = list(graph_module.graph.placeholders)
    base = len(external_nodes) if input_slots is None else int(input_slots)
    refs: dict[Node, int] = {
        node: index for index, node in enumerate(external_nodes)
    }
    refs.update(extra_refs or {})
    ref_types: dict[int, str] = {index: "num" for index in range(base)}
    program: list[int] = []
    if constants is None:
        constants = []
    instructions: list[tuple[str, int, int, int]] = []
    temp_count = 0

    def constant_ref(value: Any) -> int:
        if not _is_scalar(value):
            raise TypeError("Stax CPU pointwise constants must be scalar")
        constants.append(float(value))
        return -len(constants)

    def value_ref(value: Any) -> int:
        if isinstance(value, Node):
            if value not in refs:
                raise ValueError("pointwise graph references an unavailable value")
            return refs[value]
        return constant_ref(value)

    def value_type(ref: int) -> str:
        return ref_types.get(ref, "num")

    def emit(op_name: str, lhs: int, rhs: int = -1) -> int | None:
        nonlocal temp_count
        code = table.get(op_name)
        if code is None:
            return None
        program.extend((code, lhs, rhs))
        result = base + temp_count
        temp_count += 1
        instructions.append((op_name, lhs, rhs, result))
        return result

    for node in graph_module.graph.nodes if nodes is None else nodes:
        if node is skip_node:
            continue
        if node.op in {"placeholder", "output"}:
            continue
        if node.op not in {"call_function", "call_method"}:
            return None
        op_name = _target_name(node.target)
        if op_name not in _CPU_FUSED_OPS:
            return None
        kwargs = node.kwargs or {}
        result_type = "num"

        if op_name in {"add", "sub"} and (
            len(node.args) == 3
            or ("alpha" in (node.kwargs or {}) and len(node.args) == 2)
        ):
            if len(node.args) == 3 and "alpha" not in kwargs:
                lhs, rhs, alpha = node.args
            else:
                if len(node.args) != 2:
                    return None
                lhs, rhs = node.args
                alpha = kwargs.get("alpha", 1)
            if not _is_scalar(alpha):
                return None
            lhs_ref = value_ref(lhs)
            rhs_ref = value_ref(rhs)
            if alpha != 1:
                scaled = emit("mul", rhs_ref, constant_ref(alpha))
                if scaled is None:
                    return None
                ref_types[scaled] = "num"
                rhs_ref = scaled
            node_ref = emit(op_name, lhs_ref, rhs_ref)
            if node_ref is None:
                return None
            refs[node] = node_ref
            ref_types[node_ref] = "num"
            continue
        if op_name in _ARITHMETIC_OPS or op_name in _COMPARISON_OPS or (
            op_name in _ORDER_OPS
        ):
            if kwargs or len(node.args) != 2:
                return None
            node_ref = emit(
                op_name, value_ref(node.args[0]), value_ref(node.args[1])
            )
            if node_ref is None:
                return None
            if op_name in _COMPARISON_OPS:
                result_type = "bool"
            refs[node] = node_ref
            ref_types[node_ref] = result_type
            continue
        if op_name in _UNARY_OPS:
            if kwargs or len(node.args) != 1:
                return None
            node_ref = emit(op_name, value_ref(node.args[0]))
            if node_ref is None:
                return None
            refs[node] = node_ref
            ref_types[node_ref] = "num"
            continue
        if op_name == "where":
            if kwargs or len(node.args) != 3:
                return None
            cond_ref = value_ref(node.args[0])
            a_ref = value_ref(node.args[1])
            b_ref = value_ref(node.args[2])
            # v1 contract: a boolean condition selects between float values.
            # Numeric conditions and boolean branches stay uncompiled.
            if value_type(cond_ref) != "bool":
                return None
            if value_type(a_ref) != "num" or value_type(b_ref) != "num":
                return None
            then_ref = emit("where", cond_ref, a_ref)
            if then_ref is None:
                return None
            ref_types[then_ref] = "num"
            node_ref = emit("where_rest", cond_ref, b_ref)
            if node_ref is None:
                return None
            refs[node] = node_ref
            ref_types[node_ref] = "num"
            continue
        if op_name in _CAST_METHOD_DTYPES or op_name == "to":
            if op_name == "to":
                if set(kwargs) - {"dtype"} or len(node.args) > 2:
                    return None
                dtype_value = (
                    node.args[1] if len(node.args) > 1 else kwargs.get("dtype")
                )
                if dtype_value is None:
                    return None
                dtype_key = str(dtype_value)
            else:
                if kwargs or len(node.args) != 1:
                    return None
                dtype_key = _CAST_METHOD_DTYPES[op_name]
            dtype_id = _CAST_DTYPE_IDS.get(dtype_key)
            if dtype_id is None:
                return None
            node_ref = emit("cast", value_ref(node.args[0]), dtype_id)
            if node_ref is None:
                return None
            refs[node] = node_ref
            ref_types[node_ref] = "num"
            continue
        return None

    output_values = (
        [output_override]
        if output_override is not None
        else [
            value
            for output in graph_module.graph.outputs
            for value in _nodes(output.args)
        ]
    )
    if output_override is not None and extra_outputs is not None:
        output_values.extend(extra_outputs)
    expected_outputs = 1 if extra_outputs is None else 1 + len(extra_outputs)
    if (not program and not allow_empty) or len(output_values) != expected_outputs or (
        output_values[0] not in refs
    ):
        return None
    if value_type(refs[output_values[0]]) != "num":
        # A boolean program output would need a typed store path.
        return None
    if extra_outputs is None:
        return external_nodes, program, constants, instructions, refs[output_values[0]]
    extra_output_refs = []
    for node in extra_outputs:
        if node not in refs or value_type(refs[node]) != "num":
            return None
        extra_output_refs.append(refs[node])
    return (
        external_nodes,
        program,
        constants,
        instructions,
        refs[output_values[0]],
        tuple(extra_output_refs),
    )


def _broadcast_shape(shapes: tuple[tuple[int, ...], ...]) -> tuple[int, ...] | None:
    """Broadcast several shapes to one result shape (``None`` on mismatch)."""

    rank = max((len(s) for s in shapes), default=0)
    result: list[int] = []
    for dim in range(rank):
        extent = 1
        for shape in shapes:
            idx = dim - (rank - len(shape))
            d = shape[idx] if idx >= 0 else 1
            if d != 1:
                if extent != 1 and extent != d:
                    return None
                extent = d
        result.append(extent)
    return tuple(result)


def _lower_cpu_fused_pointwise(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    strict_native: bool = False,
    dynamic: bool = False,
) -> _CpuFusedPointwiseLowering | None:
    """Build one CPU expression program for a pointwise graph.

    The specialization requires matching contiguous float32 CPU tensors.  For
    grad-enabled pointwise graphs, Stax also emits a vectorized reverse-mode
    program and attaches it through TensorPlay's Function contract.  General
    broadcasting, views, and unsupported derivatives stay on the native p10
    path.

    Two program surfaces are attempted in order: the base opcode table,
    which every execution route (compiled kernel, program interpreter,
    fused backward) can run, and the extended surface (comparisons,
    ``where``, order relations, casts), which only the runtime-generated C
    kernel can execute — extended programs therefore require a successful
    native build and a grad-free graph.
    """

    if dynamic:
        return None
    try:
        import tensorplay

        native_module = getattr(tensorplay._C, "_stax", None)
        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if native_module is None or not hasattr(native_module.Graph, "execute"):
        return None
    if not example_inputs or any(not isinstance(value, tensor_type) for value in example_inputs):
        return None
    first = example_inputs[0]
    if (
        not first.device.is_cpu()
        or first.dtype != tensorplay.float32
        or not first.is_contiguous()
    ):
        return None
    if any(
        value.device != first.device or value.dtype != first.dtype
        for value in example_inputs[1:]
    ):
        return None
    # Broadcast/strided acceptance: inputs may differ in shape or layout as
    # long as the emitter can prove every address it generates contiguous
    # within a vector.  Everything else keeps the generic fallback.
    input_shapes = tuple(
        tuple(int(item) for item in value.shape) for value in example_inputs
    )
    input_strides = tuple(
        tuple(int(item) for item in value.stride()) for value in example_inputs
    )
    output_shape = input_shapes[0]
    if _broadcast_shape(input_shapes) != output_shape:
        return None
    try:
        from .codegen.cpp import analyze_input_modes, layouts_addressable

        input_modes = analyze_input_modes(
            input_shapes, input_strides, output_shape, lane_count=16
        )
    except (TypeError, ValueError):
        return None
    if input_modes is None:
        # Row-structured addressing widens the accepted surface: column
        # broadcasts, per-row scalars, and strided rows compile as a row
        # loop instead of losing the fused route.
        try:
            row_accepted = layouts_addressable(
                input_shapes, input_strides, output_shape, lane_count=16
            )
        except (TypeError, ValueError):
            return None
        if not row_accepted:
            return None
    # Legacy surface (every input flat) keeps the program-interpreter
    # fallback; anything else requires the compiled kernel, whose generated
    # addressing is only valid for these exact layouts, and a grad-free
    # graph (the fused backward program assumes flat inputs).
    layouts_only = input_modes is None or any(
        mode != "flat" for mode, _ in input_modes
    )
    if layouts_only and any(
        value.requires_grad for value in example_inputs
    ):
        return None

    extended = False
    try:
        pointwise = _build_pointwise_program(
            graph_module, opcodes=_CPU_FUSED_OPCODES
        )
        if pointwise is None:
            pointwise = _build_pointwise_program(
                graph_module, opcodes=_TRITON_OPCODES
            )
            extended = pointwise is not None
            if extended and any(
                value.requires_grad for value in example_inputs
            ):
                return None
    except (TypeError, ValueError, RuntimeError):
        return None
    if pointwise is None:
        return None
    external_nodes, program, constants, instructions, output_ref = pointwise
    if len(external_nodes) != len(example_inputs):
        return None

    native_runner: Any = None
    native_direct = 0
    try:
        from .codegen.cpp import build_cpu_native_kernel

        built = build_cpu_native_kernel(
            instructions,
            constants,
            len(external_nodes),
            output_ref,
            shape=first.shape,
            device=first.device,
            input_shapes=input_shapes,
            input_strides=input_strides,
        )
        if isinstance(built, tuple):
            native_runner, native_direct = built
        else:
            native_runner = built
    except Exception:
        native_runner = None
        native_direct = 0
    if extended and native_runner is None:
        return None
    # Broadcast/strided layouts have no interpreter-compatible program: the
    # compiled kernel is the only route that can address them.
    if layouts_only and native_runner is None:
        return None

    try:
        graph = native_module.Graph()
        native_values: dict[Node, Any] = {
            node: graph.add_input() for node in external_nodes
        }
        output_values = [
            value for output in graph_module.graph.outputs for value in _nodes(output.args)
        ]
        if len(output_values) != 1:
            return None
        fused = graph.create_node("fused_pointwise", output_values[0].name)
        for node in external_nodes:
            fused.add_input(native_values[node])
        fused.set_int_attr("input_count", len(external_nodes))
        fused.set_ints_attr("program", program)
        fused.set_floats_attr("constants", constants)
        graph.register_output(fused.add_output())
    except (TypeError, ValueError, RuntimeError):
        return None

    gradient_plan: tuple[list[int], list[float], tuple[int, ...]] | None = None
    if not extended and not layouts_only and any(
        value.requires_grad for value in example_inputs
    ):
        if any(op_name not in _CPU_FUSED_AUTOGRAD_OPS for op_name, *_ in instructions):
            return None
        try:
            gradient_plan = _build_fused_gradient_graphs(
                len(external_nodes),
                instructions,
                program,
                constants,
                len(program) // 3,
                output_ref,
            )
        except (TypeError, ValueError, RuntimeError):
            return None
        if gradient_plan is None:
            return None

    return _CpuFusedPointwiseLowering(
        graph_module,
        graph,
        [],
        tuple(int(item) for item in first.shape),
        first.dtype,
        first.device,
        gradient_plan,
        strict_native,
        native_runner,
        native_direct,
        expected_layouts=(
            tuple(zip(input_shapes, input_strides)) if layouts_only else None
        ),
    )


# Reduction spellings the fused CPU reduction path recognizes.  ``max``/``min``
# are accepted only in their whole-tensor form: with a dimension they return a
# value/index pair, which is a different lowering contract.
_REDUCTION_METHODS = frozenset(
    {"sum", "mean", "prod", "max", "min", "amax", "amin", "var", "std"}
)
_DIMLESS_ONLY_REDUCTIONS = frozenset({"max", "min"})
_DIM_ONLY_REDUCTIONS = frozenset({"amax", "amin"})
# The variance family carries a degrees-of-freedom correction and spells its
# trailing arguments differently from the plain value reductions.
_VARIANCE_REDUCTIONS = frozenset({"var", "std"})


def _reduction_dtype_ok(value: Any) -> bool:
    """Whether a reduction's ``dtype`` argument keeps the float32 contract."""

    if value is None:
        return True
    import tensorplay

    if value is tensorplay.float32:
        return True
    # The captured call carries the sentinel that means "keep the input
    # dtype"; the route already pinned float32 inputs.
    return str(value).rsplit(".", 1)[-1].lower() == "undefined"


def _parse_variance_reduction(
    name: str, node: Node, rank: int
) -> Any:
    """Read one variance-family node into a :class:`ReduceSpec`.

    The free function spells its trailing arguments
    ``(correction, dim, keepdim)`` while the tensor method spells them
    ``(dim, correction, keepdim)``; both accept the same keywords.  The
    default correction is one everywhere, matching the eager surface.
    """

    from .codegen.cpp_reduction import ReduceSpec

    rest = list(node.args[1:])
    if len(rest) > 3:
        return None
    kwargs = dict(node.kwargs or {})
    if set(kwargs) - {"dim", "correction", "keepdim"}:
        return None
    if node.op == "call_function":
        positional = ("correction", "dim", "keepdim")
    else:
        positional = ("dim", "correction", "keepdim")
    values: dict[str, Any] = {
        "correction": 1,
        "dim": None,
        "keepdim": False,
    }
    for index, key in enumerate(positional):
        if index < len(rest):
            values[key] = rest[index]
    for key, value in kwargs.items():
        if key in positional[: len(rest)]:
            # A keyword duplicating a positional slot is ambiguous.
            return None
        values[key] = value

    correction = values["correction"]
    keepdim = values["keepdim"]
    if isinstance(correction, bool) or not isinstance(correction, int):
        return None
    if not isinstance(keepdim, bool):
        return None
    dim = values["dim"]
    if dim is None:
        return ReduceSpec(name, tuple(range(rank)), False, correction)
    if isinstance(dim, bool):
        return None
    if isinstance(dim, int):
        dims: tuple[int, ...] = (int(dim),)
    elif isinstance(dim, (tuple, list)) and dim and all(
        isinstance(item, int) and not isinstance(item, bool) for item in dim
    ):
        dims = tuple(int(item) for item in dim)
    else:
        return None
    return ReduceSpec(name, dims, keepdim, correction)


def _parse_reduction(node: Node, rank: int) -> Any:
    """Read one reduction node into a :class:`ReduceSpec`, or ``None``.

    Positional and keyword spellings both resolve here: the trailing
    positional arguments of a reduction are ``dim``, ``keepdim``, ``dtype``
    in that order.
    """

    from .codegen.cpp_reduction import ReduceSpec

    if node.op not in {"call_function", "call_method"}:
        return None
    name = _target_name(node.target)
    if name not in _REDUCTION_METHODS:
        return None
    args = list(node.args)
    if not args or not isinstance(args[0], Node):
        return None
    if name in _VARIANCE_REDUCTIONS:
        return _parse_variance_reduction(name, node, rank)
    rest = args[1:]
    kwargs = dict(node.kwargs or {})
    if len(rest) > 3:
        return None
    dim: Any = None
    keepdim: Any = False
    if len(rest) >= 1:
        dim = rest[0]
    if len(rest) >= 2:
        keepdim = rest[1]
    if len(rest) >= 3 and not _reduction_dtype_ok(rest[2]):
        return None
    if "dim" in kwargs:
        if dim is not None:
            return None
        dim = kwargs.pop("dim")
    if "keepdim" in kwargs:
        if len(rest) >= 2:
            return None
        keepdim = kwargs.pop("keepdim")
    if "dtype" in kwargs and not _reduction_dtype_ok(kwargs.pop("dtype")):
        return None
    if kwargs:
        return None
    if not isinstance(keepdim, bool):
        return None

    if dim is None:
        if name in _DIM_ONLY_REDUCTIONS:
            return None
        return ReduceSpec(name, tuple(range(rank)), False)
    if name in _DIMLESS_ONLY_REDUCTIONS:
        return None
    if isinstance(dim, bool):
        return None
    if isinstance(dim, int):
        dims: tuple[int, ...] = (int(dim),)
    elif isinstance(dim, (tuple, list)) and dim and all(
        isinstance(item, int) and not isinstance(item, bool) for item in dim
    ):
        dims = tuple(int(item) for item in dim)
    else:
        return None
    return ReduceSpec(name, dims, keepdim)


class _CpuFusedReductionLowering:
    """Executable wrapper for Stax's fused CPU reduction kernel.

    The kernel owns the whole region: it evaluates the pointwise expression
    and the reduction in one pass, allocates its own output, and returns the
    wrapped tensor, so the steady-state call never builds an intermediate.
    """

    def __init__(
        self,
        graph_module: GraphModule,
        expected_dtype: Any,
        expected_device: Any,
        expected_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        native_runner: Any,
        native_direct: int,
        out_shape: tuple[int, ...],
        strict_native: bool = False,
    ) -> None:
        self.graph_module = graph_module
        # No interpreter-executable graph backs this region: the compiled
        # kernel is the only route, and the route check below guarantees the
        # inputs it was specialized for.
        self.graph = None
        self.placeholders = graph_module.graph.placeholders
        self.attribute_targets: list[str] = []
        self.constant_values: list[Any] = []
        self.native_values: dict[Node, Any] = {}
        self._output_count = 1
        self._public_output_count = 1
        self._output_spec = None
        self._tensorplay_codegen = "stax-fused-cpu-reduce"
        self._expected_dtype = expected_dtype
        self._expected_device = expected_device
        self._expected_layouts = expected_layouts
        self._out_shape = out_shape
        self._strict_native = strict_native
        self._native_runner = native_runner
        self._native_direct = int(native_direct) if native_direct else 0
        self._route_fp: tuple[Any, ...] | None = None
        self._route: str | None = None
        _attach_fast_call(self, exec_fn=native_runner)

    def _resolve_route(self, inputs: list[Any]) -> str:
        if not _CpuFusedPointwiseLowering._eligible_inputs(
            inputs,
            (),
            self._expected_dtype,
            self._expected_device,
            self._expected_layouts,
        ):
            return "fallback"
        if any(getattr(value, "requires_grad", False) for value in inputs):
            # Reverse mode over a fused reduction is not part of this
            # lowering's contract; a grad-carrying call re-lowers.
            return "fallback"
        return "native"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not kwargs and len(args) == len(self.placeholders):
            inputs = list(args)
        else:
            bound = self.graph_module.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            inputs = [
                bound.arguments[
                    node.target if isinstance(node.target, str) else node.name
                ]
                for node in self.placeholders
            ]
        fp = tuple(
            _CpuFusedPointwiseLowering._input_route_fingerprint(value)
            for value in inputs
        )
        if fp != self._route_fp:
            self._route = self._resolve_route(inputs)
            self._route_fp = fp
        if self._route == "fallback":
            raise RuntimeError(
                "Stax fused CPU reduction received inputs outside its "
                "compiled specialization"
            )
        return self._native_runner(inputs)


def _lower_cpu_fused_reduction(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    strict_native: bool = False,
    dynamic: bool = False,
) -> _CpuFusedReductionLowering | None:
    """Build one fused CPU kernel for a pointwise region ending in a reduction.

    The region's single output must be a reduction whose operand is used
    nowhere else, so folding it into the reduction loop cannot change what any
    other node observes.  Everything upstream of the reduction is encoded as
    the same expression program the pointwise path uses.
    """

    if dynamic:
        return None
    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if not example_inputs or any(
        not isinstance(value, tensor_type) for value in example_inputs
    ):
        return None
    first = example_inputs[0]
    if (
        not first.device.is_cpu()
        or first.dtype != tensorplay.float32
        or not first.is_contiguous()
    ):
        return None
    if any(
        value.device != first.device or value.dtype != first.dtype
        for value in example_inputs[1:]
    ):
        return None
    if any(value.requires_grad for value in example_inputs):
        return None

    input_shapes = tuple(
        tuple(int(item) for item in value.shape) for value in example_inputs
    )
    input_strides = tuple(
        tuple(int(item) for item in value.stride()) for value in example_inputs
    )
    in_shape = _broadcast_shape(input_shapes)
    if in_shape is None or not in_shape:
        return None

    output_values = [
        value
        for output in graph_module.graph.outputs
        for value in _nodes(output.args)
    ]
    if len(output_values) != 1 or not isinstance(output_values[0], Node):
        return None
    reduce_node = output_values[0]
    spec = _parse_reduction(reduce_node, len(in_shape))
    if spec is None:
        return None
    source = reduce_node.args[0]
    if not isinstance(source, Node) or len(source.users) != 1:
        return None

    try:
        pointwise = _build_pointwise_program(
            graph_module,
            skip_node=reduce_node,
            output_override=source,
            allow_empty=True,
            opcodes=_TRITON_OPCODES,
        )
    except (TypeError, ValueError, RuntimeError):
        return None
    if pointwise is None:
        return None
    external_nodes, _program, constants, instructions, output_ref = pointwise
    if len(external_nodes) != len(example_inputs):
        return None

    try:
        from .codegen.cpp_reduction import build_cpu_reduction_kernel

        built = build_cpu_reduction_kernel(
            instructions,
            constants,
            len(external_nodes),
            output_ref,
            spec,
            in_shape=in_shape,
            device=first.device,
            input_shapes=input_shapes,
            input_strides=input_strides,
        )
    except Exception:
        built = None
    if built is None:
        return None
    runner, direct, out_shape = built

    return _CpuFusedReductionLowering(
        graph_module,
        first.dtype,
        first.device,
        tuple(zip(input_shapes, input_strides)),
        runner,
        direct,
        out_shape,
        strict_native,
    )


def _elem_dependencies(
    target: Node, kinds: dict[Node, str], stop: set[Node] | None = None
) -> list[Node]:
    """Order the elementwise nodes one value depends on, producers first.

    Placeholders, row values, and anything in ``stop`` are read through
    references rather than recomputed, so they end the walk.  A node reached
    from two stages and not staged appears in both of their orders: that
    stage re-evaluates it while the row is still in cache, which is cheaper
    than the memory traffic of materializing it.
    """

    if target.op == "placeholder" or (stop is not None and target in stop):
        return []
    order: list[Node] = []
    seen: set[Node] = set()
    stack: list[tuple[Node, bool]] = [(target, False)]
    while stack:
        node, expanded = stack.pop()
        if expanded:
            order.append(node)
            continue
        if node in seen:
            continue
        seen.add(node)
        stack.append((node, True))
        operands = list(_nodes(node.args)) + list(_nodes(node.kwargs or {}))
        for operand in operands:
            if stop is not None and operand in stop:
                continue
            if kinds.get(operand) == "elem" and operand.op != "placeholder":
                if operand not in seen:
                    stack.append((operand, False))
    return order


def _plan_row_fusion(
    graph_module: GraphModule,
    in_shape: tuple[int, ...],
    input_shapes: tuple[tuple[int, ...], ...],
    *,
    stage: bool = True,
) -> Any:
    """Split a region into row stages, or return ``None``.

    A node is *elementwise* when it carries one value per input element and
    *row-valued* when it carries one value per row: reductions over the
    trailing axis turn the former into the latter, and every later stage may
    read row values as broadcasts.  The region qualifies when at least one
    such reduction exists and every node lands in one of the two classes.
    """

    from .codegen.cpp_rowfusion import ROW_OPS, RowFusion, RowStep, _ROW_UNARY

    rank = len(in_shape)
    if rank < 2 or any(int(extent) <= 0 for extent in in_shape):
        return None
    extent = int(in_shape[-1])
    rows = 1
    for size in in_shape[:-1]:
        rows *= int(size)

    placeholders = list(graph_module.graph.placeholders)
    if not placeholders or len(placeholders) != len(input_shapes):
        return None
    # Every input has to span the reduced axis: a value that is constant along
    # it would be row-valued, and the classification below assumes it is not.
    for shape in input_shapes:
        if len(shape) > rank:
            return None
        aligned = (1,) * (rank - len(shape)) + tuple(int(size) for size in shape)
        if aligned[-1] != extent:
            return None

    output_values = [
        value
        for output in graph_module.graph.outputs
        for value in _nodes(output.args)
    ]
    if len(output_values) != 1 or not isinstance(output_values[0], Node):
        return None
    target = output_values[0]
    if target.op == "placeholder":
        return None

    kinds: dict[Node, str] = {node: "elem" for node in placeholders}
    row_shapes: dict[Node, tuple[int, ...]] = {}
    slots: dict[Node, int] = {}
    staged: list[tuple[str, Node, Any]] = []
    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output"}:
            continue
        if node.op not in {"call_function", "call_method"}:
            return None
        spec = _parse_reduction(node, rank)
        if spec is not None:
            if spec.op in _VARIANCE_REDUCTIONS:
                # Row fusion folds plain values; the variance family carries
                # moment accumulators its combine cannot express.
                return None
            normalized = spec.normalized(rank)
            if normalized is None or normalized.dims != (rank - 1,):
                return None
            source = node.args[0]
            if not isinstance(source, Node) or kinds.get(source) != "elem":
                return None
            kinds[node] = "row"
            slots[node] = len(slots)
            row_shapes[node] = (
                tuple(in_shape[:-1]) + (1,)
                if normalized.keepdim
                else tuple(in_shape[:-1])
            )
            staged.append(("reduce", node, normalized))
            continue
        operands = list(_nodes(node.args)) + list(_nodes(node.kwargs or {}))
        if any(operand not in kinds for operand in operands):
            return None
        if not operands or any(kinds[operand] == "elem" for operand in operands):
            kinds[node] = "elem"
            continue
        name = _target_name(node.target)
        if name not in ROW_OPS or (node.kwargs or {}):
            return None
        arity = 1 if name in _ROW_UNARY else 2
        if len(node.args) != arity:
            return None
        if any(
            not isinstance(arg, Node) and not _is_scalar(arg)
            for arg in node.args
        ):
            return None
        shapes = {
            row_shapes[arg] for arg in node.args if isinstance(arg, Node)
        }
        if len(shapes) != 1:
            return None
        kinds[node] = "row"
        slots[node] = len(slots)
        row_shapes[node] = next(iter(shapes))
        staged.append(("rowop", node, name))

    if not slots or not any(entry[0] == "reduce" for entry in staged):
        return None
    # A row value an elementwise stage reads has to broadcast along the
    # reduced axis, which is what the kept trailing dimension expresses.
    keep = tuple(in_shape[:-1]) + (1,)
    for node in slots:
        if any(kinds.get(user) == "elem" for user in node.users):
            if row_shapes[node] != keep:
                return None

    input_count = len(placeholders)
    extra_refs = {node: input_count + slot for node, slot in slots.items()}

    # A value a reduction pass computes and a later pass needs again is worth
    # keeping: the pass already holds it in a register, so staging it costs
    # one store and saves the later pass the whole expression behind it.
    reduce_sources = [
        node.args[0] for kind, node, _payload in staged if kind == "reduce"
    ]
    later: list[set[Node]] = []
    seen_later: set[Node] = set()
    for source in reversed(reduce_sources[1:]):
        seen_later |= set(_elem_dependencies(source, kinds))
        later.append(set(seen_later))
    later.reverse()
    if kinds[target] == "elem":
        output_deps = set(_elem_dependencies(target, kinds))
    else:
        output_deps = set()
    stages: dict[Node, int] = {}
    for index, source in enumerate(reduce_sources) if stage else ():
        if source.op == "placeholder" or source in stages:
            continue
        reused = output_deps | (later[index] if index < len(later) else set())
        if source in reused:
            stages[source] = len(stages)

    total_inputs = input_count + len(slots) + len(stages)
    stage_refs = {
        node: input_count + len(slots) + slot for node, slot in stages.items()
    }
    constants: list[float] = []

    def elem_program(node: Node, available: dict[Node, int]) -> Any:
        try:
            return _build_pointwise_program(
                graph_module,
                output_override=node,
                allow_empty=True,
                opcodes=_TRITON_OPCODES,
                nodes=_elem_dependencies(node, kinds, stop=set(available)),
                extra_refs={**extra_refs, **available},
                input_slots=total_inputs,
                constants=constants,
            )
        except (TypeError, ValueError, RuntimeError):
            return None

    def row_operand(value: Any) -> int | None:
        if isinstance(value, Node):
            return extra_refs.get(value)
        if not _is_scalar(value):
            return None
        constants.append(float(value))
        return -len(constants)

    steps: list[Any] = []
    available: dict[Node, int] = {}
    for entry_kind, node, payload in staged:
        if entry_kind == "reduce":
            source = node.args[0]
            built = elem_program(source, available)
            if built is None:
                return None
            instructions, output_ref = built[3], built[4]
            stage = stage_refs.get(source)
            steps.append(
                RowStep(
                    kind="reduce",
                    slot=slots[node],
                    op=payload.op,
                    instructions=tuple(instructions),
                    output_ref=output_ref,
                    stage=-1 if stage is None else stage - input_count - len(slots),
                )
            )
            if stage is not None:
                available[source] = stage
            continue
        lhs = row_operand(node.args[0])
        rhs = -1 if payload in _ROW_UNARY else row_operand(node.args[1])
        if lhs is None or rhs is None:
            return None
        steps.append(
            RowStep(kind="rowop", slot=slots[node], op=payload, lhs=lhs, rhs=rhs)
        )

    if kinds[target] == "elem":
        built = elem_program(target, available)
        if built is None:
            return None
        out_instructions, out_ref = built[3], built[4]
        if not out_instructions:
            return None
        output_kind = "elem"
        out_shape = tuple(int(size) for size in in_shape)
    else:
        output_kind = "row"
        out_instructions = []
        out_ref = extra_refs[target]
        out_shape = row_shapes[target]

    return RowFusion(
        input_count=input_count,
        row_slots=len(slots),
        stage_slots=len(stages),
        constants=tuple(constants),
        steps=tuple(steps),
        output_kind=output_kind,
        out_instructions=tuple(out_instructions),
        out_ref=out_ref,
        reduce_extent=extent,
        rows=rows,
        in_shape=tuple(int(size) for size in in_shape),
        out_shape=tuple(int(size) for size in out_shape),
    )


class _CpuRowFusionLowering(_CpuFusedReductionLowering):
    """Executable wrapper for a row-staged CPU kernel.

    The route contract is the reduction kernel's: one compiled specialization
    that owns the whole region, allocates its output, and declines anything
    outside the layouts it was built for.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tensorplay_codegen = "stax-fused-cpu-rowfuse"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().__call__(*args, **kwargs)
        except RuntimeError as error:
            if "fused CPU reduction" in str(error):
                raise RuntimeError(
                    "Stax row-staged CPU kernel received inputs outside its "
                    "compiled specialization"
                ) from None
            raise


def _expand_row_normalizations(graph_module: GraphModule) -> GraphModule | None:
    """Rewrite the softmax family into primitives on a copy of the region.

    The composites this expands are single fused kernels of their own, so the
    expansion is only worth having when it is fused back into one kernel.
    Working on a copy is what makes that conditional: a region the row-staged
    planner then declines keeps the operators -- and the kernels -- it had.
    """

    from ....graph.passes import DecomposeRowNormalizations, row_normalization_names

    known = row_normalization_names()
    present = False
    for node in graph_module.graph.nodes:
        if node.op == "call_method":
            name = node.target if isinstance(node.target, str) else None
        elif node.op == "call_function":
            name = getattr(node.target, "__name__", None)
        else:
            continue
        if name in known:
            present = True
            break
    if not present:
        return None
    try:
        clone = _copy_region_graph(graph_module.graph)
        expanded = GraphModule(
            graph_module.root, clone, graph_module.signature
        )
        result = DecomposeRowNormalizations()(expanded)
    except Exception:
        return None
    return result.graph_module if result.modified else None


def _copy_region_graph(graph: Any) -> Any:
    """Duplicate a graph's nodes without duplicating what they carry.

    Node metadata holds the traced tensor values, so a deep copy of the
    region would clone every intermediate; node-level copies keep those
    references shared, which is all a rewrite needs.
    """

    from ....graph import Graph, map_arg

    clone = Graph()
    mapping: dict[Node, Node] = {}
    for node in graph.nodes:
        if node.op == "output":
            continue
        mapping[node] = clone.node_copy(node, lambda value: mapping[value])
    for node in graph.nodes:
        if node.op == "output":
            clone.output(
                map_arg(node.args[0], lambda value: mapping[value]), node.type
            )
    return clone


def _row_fusion_plan(
    graph_module: GraphModule,
    example_inputs: list[Any],
    on_device: str,
) -> Any:
    """Guard a region and plan it as row stages; ``None`` when it is not one.

    The plan itself carries no device: the same stages compile to a CPU loop
    nest or to one CUDA program per row, so both lowerings share this front
    end and differ only in the generator they hand the plan to.
    """

    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if not example_inputs or any(
        not isinstance(value, tensor_type) for value in example_inputs
    ):
        return None
    first = example_inputs[0]
    on = first.device.is_cuda() if on_device == "cuda" else first.device.is_cpu()
    if not on or first.dtype != tensorplay.float32 or not first.is_contiguous():
        return None
    if any(
        value.device != first.device
        or value.dtype != first.dtype
        or not value.is_contiguous()
        for value in example_inputs[1:]
    ):
        return None
    if any(value.requires_grad for value in example_inputs):
        return None

    input_shapes = tuple(
        tuple(int(item) for item in value.shape) for value in example_inputs
    )
    input_strides = tuple(
        tuple(int(item) for item in value.stride()) for value in example_inputs
    )
    in_shape = _broadcast_shape(input_shapes)
    if in_shape is None or not in_shape:
        return None

    # Keeping a value alive across a reduction is a win where the buffer
    # sits in cache and a loss where it sits in a register file: one device
    # stages, the other recomputes.
    stage = on_device != "cuda"
    module = graph_module
    fusion = _plan_row_fusion(module, in_shape, input_shapes, stage=stage)
    if fusion is None:
        expanded = _expand_row_normalizations(graph_module)
        if expanded is None:
            return None
        fusion = _plan_row_fusion(
            expanded, in_shape, input_shapes, stage=stage
        )
        if fusion is None:
            return None
        module = expanded
    return module, fusion, input_shapes, input_strides, first


class _CudaRowFusionLowering(_CpuFusedReductionLowering):
    """Executable wrapper for a row-staged CUDA kernel.

    One program per output row, the row resident for the whole region: the
    inputs are read once no matter how many reductions the region contains.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tensorplay_codegen = "stax-fused-cuda-rowfuse"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().__call__(*args, **kwargs)
        except RuntimeError as error:
            if "fused CPU reduction" in str(error):
                raise RuntimeError(
                    "Stax row-staged CUDA kernel received inputs outside its "
                    "compiled specialization"
                ) from None
            raise


def _lower_cuda_row_fusion(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    strict_native: bool = False,
    dynamic: bool = False,
) -> Any:
    """Build one CUDA kernel for a region with reductions in the middle.

    Splitting such a region at every reduction gives one kernel per stage,
    each streaming the input again and writing an intermediate the next one
    reads back.  Keeping the row resident across the stages removes all of
    that traffic and leaves one launch.
    """

    if dynamic:
        return None
    planned = _row_fusion_plan(graph_module, example_inputs, "cuda")
    if planned is None:
        return None
    module, fusion, input_shapes, input_strides, first = planned

    try:
        from .codegen.triton_rowfusion import build_cuda_row_fusion_kernel

        launch = build_cuda_row_fusion_kernel(
            fusion,
            input_shapes=input_shapes,
            input_strides=input_strides,
        )
    except Exception:
        launch = None
    if launch is None:
        return None

    return _CudaRowFusionLowering(
        module,
        first.dtype,
        first.device,
        tuple(zip(input_shapes, input_strides)),
        launch,
        0,
        fusion.out_shape,
        strict_native,
    )


def _lower_cpu_row_fusion(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    strict_native: bool = False,
    dynamic: bool = False,
) -> Any:
    """Build one CPU kernel for a region with reductions in the middle.

    The reduction results feed elementwise work over the same axis they were
    reduced along, so the region cannot be expressed as a pointwise program
    or as a pointwise program ending in a reduction.  Staging it per row keeps
    every intermediate in cache instead of writing it out and reading it back.
    """

    if dynamic:
        return None
    planned = _row_fusion_plan(graph_module, example_inputs, "cpu")
    if planned is None:
        return None
    module, fusion, input_shapes, input_strides, first = planned

    try:
        from .codegen.cpp_rowfusion import build_cpu_row_fusion_kernel

        built = build_cpu_row_fusion_kernel(
            fusion,
            device=first.device,
            input_shapes=input_shapes,
            input_strides=input_strides,
        )
    except Exception:
        built = None
    if built is None:
        return None
    runner, direct = built

    return _CpuRowFusionLowering(
        module,
        first.dtype,
        first.device,
        tuple(zip(input_shapes, input_strides)),
        runner,
        direct,
        fusion.out_shape,
        strict_native,
    )


# ---------------------------------------------------------------------------
# Mixed regions: generated kernels between operators that run as they are


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


def _segment_externals(nodes: tuple[Node, ...]) -> list[Node]:
    """Values a segment reads from outside itself, in first-use order."""

    inside = set(nodes)
    externals: list[Node] = []
    seen: set[Node] = set()
    for node in nodes:
        operands = list(_nodes(node.args)) + list(_nodes(node.kwargs or {}))
        for operand in operands:
            if operand in inside or operand in seen:
                continue
            seen.add(operand)
            externals.append(operand)
    return externals


class _SegmentKernelPlan:
    """Planned translation unit for one fusible segment, before building.

    Holds everything the build call needs — program instructions, layouts,
    reduction spec — so the planning walk stays on the calling thread while
    the builds themselves can run concurrently.
    """

    __slots__ = (
        "externals",
        "input_shapes",
        "input_strides",
        "instructions",
        "constants",
        "output_ref",
        "source_layout",
        "reduction",
    )

    def __init__(
        self,
        externals,
        input_shapes,
        input_strides,
        instructions,
        constants,
        output_ref,
        source_layout,
        reduction,
    ) -> None:
        self.externals = externals
        self.input_shapes = input_shapes
        self.input_strides = input_strides
        self.instructions = instructions
        self.constants = constants
        self.output_ref = output_ref
        self.source_layout = source_layout
        self.reduction = reduction


def _plan_segment_kernel(
    graph_module: GraphModule,
    segment: Any,
) -> _SegmentKernelPlan | None:
    """Plan one fusible segment's kernel; ``None`` when it is not expressible."""

    externals = _segment_externals(segment.nodes)
    if not externals or len(externals) > 16:
        return None
    layouts = []
    for node in externals:
        layout = _tensor_layout(_traced_value(graph_module, node))
        if layout is None:
            return None
        layouts.append(layout)
    input_shapes = tuple(shape for shape, _stride in layouts)
    input_strides = tuple(stride for _shape, stride in layouts)
    refs = {node: index for index, node in enumerate(externals)}

    reduce_node = segment.tail if segment.kind == "pw+red" else None
    body = [node for node in segment.nodes if node is not reduce_node]
    source = segment.producer if reduce_node is not None else segment.nodes[-1]
    if source is None:
        return None
    try:
        program = _build_pointwise_program(
            graph_module,
            output_override=source,
            allow_empty=reduce_node is not None,
            opcodes=_TRITON_OPCODES,
            nodes=body,
            extra_refs=refs,
            input_slots=len(externals),
        )
    except (TypeError, ValueError, RuntimeError):
        return None
    if program is None:
        return None
    _external, _encoded, constants, instructions, output_ref = program

    layout = _tensor_layout(_traced_value(graph_module, source))
    if layout is None:
        return None
    return _SegmentKernelPlan(
        externals=externals,
        input_shapes=input_shapes,
        input_strides=input_strides,
        instructions=instructions,
        constants=constants,
        output_ref=output_ref,
        source_layout=layout,
        reduction=segment.reduction if reduce_node is not None else None,
    )


def _compile_segment_kernel(
    plan: _SegmentKernelPlan,
    device: Any,
) -> tuple[list[Node], Any] | None:
    """Build one planned segment; return its inputs and callable runner."""

    if plan.reduction is None:
        try:
            from .codegen.cpp import build_cpu_native_kernel

            built = build_cpu_native_kernel(
                plan.instructions,
                plan.constants,
                len(plan.externals),
                plan.output_ref,
                shape=plan.source_layout[0],
                device=device,
                input_shapes=plan.input_shapes,
                input_strides=plan.input_strides,
            )
        except Exception:
            return None
        if built is None:
            return None
        runner = built[0] if isinstance(built, tuple) else built
        return plan.externals, runner

    try:
        from .codegen.cpp_reduction import build_cpu_reduction_kernel

        built = build_cpu_reduction_kernel(
            plan.instructions,
            plan.constants,
            len(plan.externals),
            plan.output_ref,
            plan.reduction,
            in_shape=plan.source_layout[0],
            device=device,
            input_shapes=plan.input_shapes,
            input_strides=plan.input_strides,
        )
    except Exception:
        return None
    if built is None:
        return None
    return plan.externals, built[0]


def _kernel_step(runner: Any, sources: tuple[int, ...]):
    """Close over one generated kernel and the slots holding its inputs."""

    def run(values: list[Any]) -> Any:
        return runner([values[slot] for slot in sources])

    return run


class _CpuSegmentedLowering:
    """Runs a mixed region: generated kernels between untouched operators.

    A region that mixes fusible work with operators the generators do not
    cover used to lose the compiled route entirely.  Here the fusible runs
    become one kernel each and the rest of the region runs exactly as it
    was captured, so a pointwise chain between two matrix products costs a
    single pass over its data instead of one pass per operator.
    """

    def __init__(
        self,
        graph_module: GraphModule,
        steps: list[tuple],
        slot_count: int,
        constants: dict[int, Any],
        output_slot: int,
        expected_dtype: Any,
        expected_device: Any,
        expected_layouts: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
        strict_native: bool = False,
    ) -> None:
        self.graph_module = graph_module
        self.graph = None
        self.placeholders = graph_module.graph.placeholders
        self.attribute_targets: list[str] = []
        self.constant_values: list[Any] = []
        self.native_values: dict[Node, Any] = {}
        self._output_count = 1
        self._public_output_count = 1
        self._output_spec = None
        self._tensorplay_codegen = "stax-fused-cpu-segments"
        self._steps = steps
        self._template: list[Any] = [None] * slot_count
        for slot, value in constants.items():
            self._template[slot] = value
        self._output_slot = output_slot
        self._expected_dtype = expected_dtype
        self._expected_device = expected_device
        self._expected_layouts = expected_layouts
        self._strict_native = strict_native
        self._route_fp: tuple[Any, ...] | None = None
        self._route: str | None = None
        _attach_fast_call(self, exec_fn=self._execute)

    def _execute(self, inputs: list[Any]) -> Any:
        # The captured constants never move, so the value table starts as a
        # copy of a template that already holds them.
        values = self._template.copy()
        values[: len(inputs)] = inputs
        for step, target, release in self._steps:
            values[target] = step(values)
            # Dropping an intermediate as soon as its last reader has run
            # hands the buffer straight back to the allocator, so the next
            # operator writes into memory that is still warm.
            for slot in release:
                values[slot] = None
        return values[self._output_slot]

    def _resolve_route(self, inputs: list[Any]) -> str:
        if not _CpuFusedPointwiseLowering._eligible_inputs(
            inputs,
            (),
            self._expected_dtype,
            self._expected_device,
            self._expected_layouts,
        ):
            return "fallback"
        if any(getattr(value, "requires_grad", False) for value in inputs):
            return "fallback"
        return "native"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not kwargs and len(args) == len(self.placeholders):
            inputs = list(args)
        else:
            bound = self.graph_module.signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            inputs = [
                bound.arguments[
                    node.target if isinstance(node.target, str) else node.name
                ]
                for node in self.placeholders
            ]
        fp = tuple(
            _CpuFusedPointwiseLowering._input_route_fingerprint(value)
            for value in inputs
        )
        if fp != self._route_fp:
            self._route = self._resolve_route(inputs)
            self._route_fp = fp
        if self._route == "fallback":
            raise RuntimeError(
                "Stax segmented CPU region received inputs outside its "
                "compiled specialization"
            )
        return self._execute(inputs)


def _lower_cpu_segmented(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    strict_native: bool = False,
    dynamic: bool = False,
) -> _CpuSegmentedLowering | None:
    """Compile the fusible runs of a region the whole-region paths declined.

    The scheduler partitions the region without store-time epilogues; every
    pointwise run and every run ending in a reduction becomes one generated
    kernel, and each remaining operator stays a single call between them.
    """

    if dynamic:
        return None
    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if not example_inputs or any(
        not isinstance(value, tensor_type) for value in example_inputs
    ):
        return None
    first = example_inputs[0]
    if not first.device.is_cpu() or first.dtype != tensorplay.float32:
        return None
    if any(
        value.device != first.device or value.dtype != first.dtype
        for value in example_inputs[1:]
    ):
        return None
    if any(value.requires_grad for value in example_inputs):
        return None
    if len(example_inputs) != len(graph_module.graph.placeholders):
        return None

    output_values = [
        value
        for output in graph_module.graph.outputs
        for value in _nodes(output.args)
    ]
    if len(output_values) != 1 or not isinstance(output_values[0], Node):
        return None
    final = output_values[0]

    def is_pointwise(node: Node) -> bool:
        if node.op not in {"call_function", "call_method"}:
            return False
        return _target_name(node.target) in _CPU_FUSED_OPS

    def classify_reduction(node: Node) -> Any:
        if node.op not in {"call_function", "call_method"}:
            return None
        if not node.args or not isinstance(node.args[0], Node):
            return None
        layout = _tensor_layout(_traced_value(graph_module, node.args[0]))
        if layout is None:
            return None
        return _parse_reduction(node, len(layout[0]))

    from .scheduler import segment_graph

    # Scheduled without epilogues, like the training path: a single-user
    # pointwise tail over an eager operator stays its own kernel here
    # (composing it into the operator's store is a whole-region-path
    # capability), so a mixed region never loses its kernels to a schedule
    # this path cannot wire.
    segments = segment_graph(
        graph_module,
        is_pointwise=is_pointwise,
        classify_reduction=classify_reduction,
        allow_epilogue=False,
    )
    if segments is None:
        return None
    if any(len(segment.exports) > 1 for segment in segments):
        # Horizontal fusion (extra kernel stores) is a Triton-path
        # capability; the CPU mixed scheduler keeps one value per kernel.
        return None
    if any(segment.epilogue for segment in segments):
        return None
    compiled_kinds = {"pw", "pw+red"}
    if not any(segment.kind in compiled_kinds for segment in segments):
        return None
    if not any(segment.kind == "extern" for segment in segments):
        # A region that is fusible end to end belongs to the whole-region
        # paths; reaching here means they declined it for another reason.
        return None

    slots: dict[Node, int] = {
        node: index for index, node in enumerate(graph_module.graph.placeholders)
    }
    constants: dict[int, Any] = {}
    steps: list[tuple] = []
    next_slot = len(slots)

    def slot_for(node: Node) -> int | None:
        if node in slots:
            return slots[node]
        if node.op != "get_attr":
            return None
        value = _traced_value(graph_module, node)
        if not isinstance(value, tensor_type):
            return None
        nonlocal next_slot
        slots[node] = next_slot
        constants[next_slot] = value
        next_slot += 1
        return slots[node]

    def extern_step(node: Node):
        """Close over one operator's call, resolving its operands by slot.

        The plan is built once: an argument is either a slot to read or a
        value to pass through, so the steady-state call walks a flat list
        instead of rebuilding the captured argument structure.
        """

        target = node.target
        op = node.op
        kwargs_template = dict(node.kwargs or {})
        table = slots
        simple = all(
            not isinstance(item, (list, tuple, dict, slice))
            for item in (*node.args, *kwargs_template.values())
        )
        if not simple:
            from ....graph import map_arg

            def run_general(values: list[Any]) -> Any:
                resolve = lambda item: values[table[item]]  # noqa: E731
                args = map_arg(node.args, resolve)
                kwargs = map_arg(kwargs_template, resolve)
                if op == "call_function":
                    return target(*args, **kwargs)
                return getattr(args[0], target)(*args[1:], **kwargs)

            return run_general

        plan = tuple(
            (table[item], None) if isinstance(item, Node) else (-1, item)
            for item in node.args
        )
        keys = tuple(kwargs_template)
        key_plan = tuple(
            (table[item], None) if isinstance(item, Node) else (-1, item)
            for item in kwargs_template.values()
        )

        if not keys and op == "call_function":

            def run_positional(values: list[Any]) -> Any:
                return target(
                    *[
                        values[slot] if value is None else value
                        for slot, value in plan
                    ]
                )

            return run_positional

        def run(values: list[Any]) -> Any:
            args = [values[slot] if value is None else value for slot, value in plan]
            kwargs = {
                key: (values[slot] if value is None else value)
                for key, (slot, value) in zip(keys, key_plan)
            }
            if op == "call_function":
                return target(*args, **kwargs)
            return getattr(args[0], target)(*args[1:], **kwargs)

        return run

    def add_extern(node: Node) -> bool:
        nonlocal next_slot
        if node.op == "get_attr":
            return slot_for(node) is not None
        if node.op not in {"call_function", "call_method"}:
            return False
        operands = list(_nodes(node.args)) + list(_nodes(node.kwargs or {}))
        for operand in operands:
            if slot_for(operand) is None:
                return False
        sources = tuple(slots[operand] for operand in operands)
        slots[node] = next_slot
        next_slot += 1
        steps.append((extern_step(node), slots[node], sources))
        return True

    # Planning and building are separate walks: the plans fix every
    # segment's expressibility on the calling thread, the builds overlap on
    # the build pool, and the wiring below replays the same per-segment
    # decisions — kernel step versus individual operators — with the same
    # slot order a one-kernel-at-a-time walk would have produced.
    plans = [
        _plan_segment_kernel(graph_module, segment)
        if segment.kind != "extern"
        else None
        for segment in segments
    ]
    from .parallel_compile import run_builds

    builds = run_builds(
        [
            lambda plan=plan: _compile_segment_kernel(plan, first.device)
            for plan in plans
            if plan is not None
        ]
    )
    build_results = iter(builds)

    compiled_count = 0
    for segment, plan in zip(segments, plans):
        if plan is not None:
            built = next(build_results)
            if built is not None:
                externals, runner = built
                sources = []
                for node in externals:
                    source = slot_for(node)
                    if source is None:
                        return None
                    sources.append(source)
                export = segment.export_node
                if export is None:
                    return None
                slots[export] = next_slot
                next_slot += 1
                steps.append(
                    (
                        _kernel_step(runner, tuple(sources)),
                        slots[export],
                        tuple(sources),
                    )
                )
                compiled_count += 1
                continue
            # One run the generators cannot express does not cost the region
            # its other kernels: those operators run individually instead.
        for node in segment.nodes:
            if not add_extern(node):
                return None
    if compiled_count == 0:
        return None

    if final not in slots:
        return None

    # Liveness: a value is dropped right after the step that reads it last,
    # so long regions do not hold every intermediate alive to the end.
    last_use: dict[int, int] = {}
    for index, (_step, _target, sources) in enumerate(steps):
        for slot in sources:
            last_use[slot] = index
    output_slot = slots[final]
    plan = [
        (
            step,
            target,
            tuple(
                slot
                for slot, index in last_use.items()
                if index == position and slot != output_slot
            ),
        )
        for position, (step, target, _sources) in enumerate(steps)
    ]

    input_shapes = tuple(
        tuple(int(item) for item in value.shape) for value in example_inputs
    )
    input_strides = tuple(
        tuple(int(item) for item in value.stride()) for value in example_inputs
    )
    return _CpuSegmentedLowering(
        graph_module,
        plan,
        next_slot,
        constants,
        output_slot,
        first.dtype,
        first.device,
        tuple(zip(input_shapes, input_strides)),
        strict_native,
    )


def _build_fused_gradient_graphs(
    input_count: int,
    instructions: list[tuple[str, int, int, int]],
    forward_program: list[int],
    forward_constants: list[float],
    forward_temp_count: int,
    output_ref: int,
) -> tuple[list[int], list[float], tuple[int, ...]] | None:
    """Create one shared fused reverse-mode program for all inputs.

    The forward intermediates are emitted once.  Each input derivative then
    extends that same program and records one final temporary, allowing the
    native kernel to evaluate all gradients in one vector loop.
    """

    def remap_forward_ref(ref: int) -> int:
        if ref >= input_count:
            return ref + 1  # reserve the final external input for grad_output
        return ref

    remapped_forward_program: list[int] = []
    for offset in range(0, len(forward_program), 3):
        remapped_forward_program.extend(
            (
                forward_program[offset],
                remap_forward_ref(forward_program[offset + 1]),
                remap_forward_ref(forward_program[offset + 2]),
            )
        )

    program = list(remapped_forward_program)
    constants = list(forward_constants)
    temp_count = forward_temp_count
    zero_ref = -(len(constants) + 1)
    constants.append(0.0)
    one_ref = -(len(constants) + 1)
    constants.append(1.0)
    two_ref = -(len(constants) + 1)
    constants.append(2.0)
    grad_output_ref = input_count
    output_refs: list[int] = []

    def emit(op_name: str, lhs: int, rhs: int = -1) -> int:
        nonlocal temp_count
        if op_name not in _CPU_FUSED_OPCODES:
            raise ValueError(f"unsupported fused derivative op: {op_name}")
        program.extend((_CPU_FUSED_OPCODES[op_name], lhs, rhs))
        result = input_count + 1 + temp_count
        temp_count += 1
        return result

    def is_zero(ref: int) -> bool:
        return ref == zero_ref

    def is_one(ref: int) -> bool:
        return ref == one_ref

    def add_ref(lhs: int, rhs: int) -> int:
        if is_zero(lhs):
            return rhs
        if is_zero(rhs):
            return lhs
        return emit("add", lhs, rhs)

    def sub_ref(lhs: int, rhs: int) -> int:
        if is_zero(rhs):
            return lhs
        return emit("sub", lhs, rhs)

    def mul_ref(lhs: int, rhs: int) -> int:
        if is_zero(lhs) or is_zero(rhs):
            return zero_ref
        if is_one(lhs):
            return rhs
        if is_one(rhs):
            return lhs
        return emit("mul", lhs, rhs)

    def neg_ref(ref: int) -> int:
        if is_zero(ref):
            return ref
        return emit("neg", ref)

    def div_ref(lhs: int, rhs: int) -> int:
        if is_zero(lhs):
            return zero_ref
        if is_one(rhs):
            return lhs
        return emit("div", lhs, rhs)

    adjoints: dict[int, int] = {
        remap_forward_ref(output_ref): grad_output_ref,
    }

    def add_adjoint(ref: int, contribution: int) -> None:
        if ref < 0 or is_zero(contribution):
            return
        adjoints[ref] = add_ref(adjoints.get(ref, zero_ref), contribution)

    for op_name, lhs, rhs, result in reversed(instructions):
        lhs = remap_forward_ref(lhs)
        rhs = remap_forward_ref(rhs)
        result = remap_forward_ref(result)
        grad = adjoints.get(result, zero_ref)
        if is_zero(grad):
            continue

        if op_name == "add":
            add_adjoint(lhs, grad)
            add_adjoint(rhs, grad)
        elif op_name == "sub":
            add_adjoint(lhs, grad)
            add_adjoint(rhs, neg_ref(grad))
        elif op_name == "mul":
            add_adjoint(lhs, mul_ref(grad, rhs))
            add_adjoint(rhs, mul_ref(grad, lhs))
        elif op_name == "div":
            add_adjoint(lhs, div_ref(grad, rhs))
            denominator = mul_ref(rhs, rhs)
            add_adjoint(rhs, neg_ref(div_ref(mul_ref(grad, lhs), denominator)))
        elif op_name == "neg":
            add_adjoint(lhs, neg_ref(grad))
        elif op_name == "pos":
            add_adjoint(lhs, grad)
        elif op_name == "abs":
            add_adjoint(lhs, mul_ref(grad, emit("abs_grad", lhs)))
        elif op_name == "sin":
            add_adjoint(lhs, mul_ref(grad, emit("cos", lhs)))
        elif op_name == "cos":
            add_adjoint(lhs, mul_ref(grad, neg_ref(emit("sin", lhs))))
        elif op_name == "exp":
            add_adjoint(lhs, mul_ref(grad, result))
        elif op_name == "log":
            add_adjoint(lhs, div_ref(grad, lhs))
        elif op_name == "sigmoid":
            local = mul_ref(result, sub_ref(one_ref, result))
            add_adjoint(lhs, mul_ref(grad, local))
        elif op_name == "sqrt":
            add_adjoint(lhs, div_ref(grad, mul_ref(two_ref, result)))
        elif op_name == "square":
            add_adjoint(lhs, mul_ref(grad, mul_ref(two_ref, lhs)))
        elif op_name == "tanh":
            local = sub_ref(one_ref, mul_ref(result, result))
            add_adjoint(lhs, mul_ref(grad, local))
        elif op_name == "relu":
            add_adjoint(lhs, mul_ref(grad, emit("relu_grad", lhs)))
        else:
            return None

    # Make every output a temporary.  This also handles a disconnected input
    # (constant zero) and an input that receives grad_output directly.
    for input_ref in range(input_count):
        output_refs.append(emit("pos", adjoints.get(input_ref, zero_ref)))

    # Remove forward values that are not needed by any local derivative.  For
    # example, d(sin(x))/dx uses cos(x), not the forward sin(x) result; this
    # follows the derivative graph rather than blindly replaying all of
    # the forward graph.
    total_input_count = input_count + 1
    instruction_count = len(program) // 3
    live = [False] * instruction_count
    pending = list(output_refs)
    while pending:
        ref = pending.pop()
        if ref < total_input_count:
            continue
        instruction = ref - total_input_count
        if instruction < 0 or instruction >= instruction_count or live[instruction]:
            continue
        live[instruction] = True
        offset = instruction * 3
        pending.extend((program[offset + 1], program[offset + 2]))

    compact_refs: dict[int, int] = {}
    next_instruction = 0
    for instruction, is_live in enumerate(live):
        if is_live:
            compact_refs[total_input_count + instruction] = (
                total_input_count + next_instruction
            )
            next_instruction += 1

    compact_program: list[int] = []
    for instruction, is_live in enumerate(live):
        if not is_live:
            continue
        offset = instruction * 3
        compact_program.extend(
            (
                program[offset],
                compact_refs.get(program[offset + 1], program[offset + 1]),
                compact_refs.get(program[offset + 2], program[offset + 2]),
            )
        )
    output_refs = [compact_refs[ref] for ref in output_refs]

    program = compact_program
    if len(program) // 3 > 64:
        return None
    return program, constants, tuple(output_refs)


def _fold_eval_conv_batch_norm(
    graph_module: GraphModule,
    example_inputs: list[Any],
) -> dict[Node, tuple[Node, Any, Any]]:
    """Precompute inference BatchNorm parameters for Conv2d users.

    ResNet inference contains the stable pattern ``conv2d -> batch_norm``.
    Folding the running-statistics transform into the convolution removes one
    full feature-map kernel and its intermediate write.  The optimization is
    intentionally restricted to eval-mode BatchNorm with a single Conv2d
    user, so training graphs and branched tensors retain the ordinary native
    operators.

    The returned tensors are compile-time constants owned by the native
    lowering.  TensorPlay's public compile contract currently has no
    parameter-version guard, therefore this pass is only enabled for the
    inference lowering path; callers that mutate parameters must recompile.
    """

    # Do not fold a graph that is being differentiated with respect to its
    # runtime inputs.  The folded parameters are inference constants, while
    # eval-mode autograd still needs the original parameter edges.
    if any(getattr(value, "requires_grad", False) for value in example_inputs):
        return {}

    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return {}

    # Folding changes parameter dataflow, so it is valid only for the
    # no-grad inference specialization that the benchmark requests.  A
    # grad-enabled compile must retain the ordinary Conv/BN autograd edges.
    if tensorplay.is_grad_enabled():
        return {}

    def tensor_attr(value: Any) -> Any | None:
        if not isinstance(value, Node) or value.op != "get_attr":
            return None
        attribute = graph_module._get_attr(value.target)
        return attribute if isinstance(attribute, tensor_type) else None

    folded: dict[Node, tuple[Node, Any, Any]] = {}
    for batch_norm in graph_module.graph.nodes:
        if (
            batch_norm.op != "call_function"
            or _target_name(batch_norm.target) != "batch_norm"
            or len(batch_norm.args) != 8
            or batch_norm.kwargs
            or batch_norm.args[5] is not False
        ):
            continue

        conv = batch_norm.args[0]
        if (
            not isinstance(conv, Node)
            or conv.op != "call_function"
            or _target_name(conv.target) != "conv2d"
            or len(conv.args) != 7
            or conv.kwargs
            or conv.users != {batch_norm}
        ):
            continue

        running_mean = tensor_attr(batch_norm.args[1])
        running_var = tensor_attr(batch_norm.args[2])
        bn_weight = tensor_attr(batch_norm.args[3])
        bn_bias = tensor_attr(batch_norm.args[4])
        conv_weight = tensor_attr(conv.args[1])
        conv_bias = tensor_attr(conv.args[2])
        eps = batch_norm.args[7]
        if (
            running_mean is None
            or running_var is None
            or conv_weight is None
            or not isinstance(eps, numbers.Real)
            or conv.args[2] is not None and conv_bias is None
            or batch_norm.args[3] is not None and bn_weight is None
            or batch_norm.args[4] is not None and bn_bias is None
        ):
            continue

        try:
            with tensorplay.no_grad():
                running_mean = running_mean.detach()
                running_var = running_var.detach()
                conv_weight = conv_weight.detach()
                weight_coeff = tensorplay.rsqrt(running_var + float(eps))
                scale = (
                    bn_weight.detach()
                    if bn_weight is not None
                    else tensorplay.ones_like(running_var)
                ) * weight_coeff
                folded_weight = conv_weight * scale.reshape((-1, 1, 1, 1))
                base_bias = (
                    conv_bias.detach()
                    if conv_bias is not None
                    else tensorplay.zeros_like(running_mean)
                )
                folded_bias = (
                    (base_bias - running_mean) * scale
                    + (
                        bn_bias.detach()
                        if bn_bias is not None
                        else tensorplay.zeros_like(running_mean)
                    )
                )
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Keep the ordinary native Conv+BN path if a backend dtype or
            # device cannot materialize the folded constants.
            continue

        folded[conv] = (batch_norm, folded_weight, folded_bias)
    return folded


_NATIVE_OP_SUPPORT: dict[str, bool] = {}


def _native_runs_linear() -> bool:
    """Whether the loaded runtime can execute a fused ``linear`` node.

    The lowering and the runtime library are built separately, and a tree can
    hold one newer than the other, so a node this module knows how to emit is
    not necessarily one the runtime knows how to run.  Probed once per
    process with a one-node graph; a runtime that cannot run it keeps the
    transpose-product-add form, which every runtime can.
    """

    known = _NATIVE_OP_SUPPORT.get("linear")
    if known is not None:
        return known
    supported = False
    try:
        import tensorplay

        native = tensorplay._C._stax
        graph = native.Graph()
        node = graph.create_node("linear", "probe")
        node.add_input(graph.add_input())
        node.add_input(graph.add_input())
        graph.register_output(node.add_output())
        graph.execute([tensorplay.zeros((1, 1)), tensorplay.zeros((1, 1))])
        supported = True
    except Exception:  # noqa: BLE001 - any failure means "emit the long form"
        supported = False
    _NATIVE_OP_SUPPORT["linear"] = supported
    return supported


def _register_stax_cuda_pointwise_op(
    program: list[int],
    constants: list[float],
    output_refs: tuple[int, ...],
    input_count: int,
    example_values: list[Any],
    output_dtypes: tuple[str, ...] | None = None,
) -> str | None:
    global _STAX_CUDA_OP_COUNTER
    key = (
        tuple(program),
        tuple(constants),
        output_refs,
        input_count,
        output_dtypes,
        tuple(
            (
                tuple(int(item) for item in value.shape),
                str(value.dtype),
                str(value.device),
            )
            for value in example_values
        ),
    )
    cached = _STAX_CUDA_OP_CACHE.get(key)
    if cached is not None:
        return cached
    # Plans with sample tensors compile through the Triton program generator:
    # the kernel reads broadcast operands in place and skips the launch-time
    # materialization the generic jiterator wrapper pays per call.  Sample
    # outputs name one output dtype per output port, which also unlocks plans
    # whose inputs do not all share that dtype.  Example-less registrations
    # keep the jiterator route -- with no sample shapes there is nothing to
    # compile a reference geometry from.
    fallback = None
    if example_values and output_dtypes and len(output_dtypes) == len(
        output_refs
    ):
        try:
            from .codegen.triton import _supports_runtime_inputs

            rank = max(
                len(tuple(int(item) for item in value.shape))
                for value in example_values
            )
            reference_shape = tuple(
                max(
                    (1,) * (rank - len(tuple(int(item) for item in value.shape)))
                    + tuple(int(item) for item in value.shape)
                    for value in example_values
                )[dim]
                for dim in range(rank)
            )
            runner = _cuda_triton_pointwise_runner(
                program, constants, output_refs, example_values, output_dtypes
            )
            if runner is not None:
                mixed_inputs = any(
                    value.dtype != example_values[0].dtype
                    for value in example_values[1:]
                )
                try:
                    from .codegen.cuda import compile_program as _jit_compile

                    fallback = _jit_compile(
                        program,
                        constants,
                        output_refs,
                        input_count,
                        example_values,
                    )
                    if example_values:
                        fallback(example_values)
                except (AssertionError, RuntimeError, TypeError, ValueError):
                    return None
                try:
                    from ....library import _define_or_get

                    name = f"tp_stax::pointwise_{_STAX_CUDA_OP_COUNTER}"
                    _STAX_CUDA_OP_COUNTER += 1
                    op = _define_or_get(name, None)

                    def kernel(
                        *inputs: Any,
                        _runner: Any = runner,
                        _fallback: Any = fallback,
                        _reference: tuple[int, ...] = reference_shape,
                        _mixed: bool = mixed_inputs,
                    ) -> Any:
                        # The generated kernel addresses every input as
                        # contiguous.  A strided operand (an expanded
                        # per-channel view, for example) is materialized here
                        # -- a tiny copy next to the wrapper the fallback
                        # runs, which re-materializes the whole broadcast.
                        values = [
                            value
                            if value.is_contiguous()
                            else value.contiguous()
                            for value in inputs
                        ]
                        # allow_grad: this op runs inside the native graph
                        # whose reverse pass is a separately compiled
                        # program, so parameter inputs carrying
                        # requires_grad need no autograd node here --
                        # rejecting them would drop every plan that reads a
                        # parameter onto the materializing fallback.
                        if not _supports_runtime_inputs(
                            values,
                            allow_grad=True,
                            reference_shape=_reference,
                            allow_mixed_dtypes=_mixed,
                        ):
                            return _fallback(values)
                        result = _runner(values)
                        if isinstance(result, list):
                            return tuple(result)
                        return result

                    op.register_kernel("cuda")(kernel)
                    _STAX_CUDA_OP_CACHE[key] = name
                    return name
                except (AssertionError, RuntimeError, TypeError, ValueError):
                    pass
        except Exception:  # noqa: BLE001 - registration misses fall through
            pass
    try:
        from .codegen.cuda import compile_program
        from ....library import _define_or_get

        runner = compile_program(
            program,
            constants,
            output_refs,
            input_count,
            example_values,
        )
        if example_values:
            runner(example_values)
        name = f"tp_stax::pointwise_{_STAX_CUDA_OP_COUNTER}"
        _STAX_CUDA_OP_COUNTER += 1
        op = _define_or_get(name, None)

        def kernel(*inputs: Any, _runner: Any = runner) -> Any:
            return _runner(list(inputs))

        op.register_kernel("cuda")(kernel)
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return None
    _STAX_CUDA_OP_CACHE[key] = name
    return name


def _native_fused_pointwise_plans(
    graph_module: GraphModule,
    required_nodes: set[Node],
) -> dict[Node, tuple[Any, ...]]:
    try:
        import tensorplay

        from .scheduler import segment_graph
    except (AttributeError, ImportError):
        return {}
    if not tensorplay.is_grad_enabled():
        return {}

    def is_pointwise(node: Node) -> bool:
        if node.op not in {"call_function", "call_method"}:
            return False
        return _target_name(node.target) in _CPU_FUSED_OPS

    segments = segment_graph(
        graph_module,
        is_pointwise=is_pointwise,
        classify_reduction=lambda node: None,
        allow_epilogue=False,
    )
    if segments is None:
        return {}

    def make_plan(nodes: tuple[Node, ...]):
        if len(nodes) < 2:
            return None
        inside = set(nodes)
        exports = [
            node
            for node in nodes
            if node in required_nodes
            or any(user not in inside for user in node.users)
        ]
        if not exports:
            exports = [nodes[-1]]
        externals = _segment_externals(nodes)
        if not 1 <= len(externals) <= 32:
            return None
        external_values = [_traced_value(graph_module, node) for node in externals]
        if any(
            not isinstance(value, tensorplay.Tensor)
            or not value.device.is_cuda()
            for value in external_values
        ):
            return None
        input_shapes = [
            tuple(int(item) for item in value.shape) for value in external_values
        ]
        output_shape = _broadcast_shape(input_shapes)
        if output_shape is None or not output_shape:
            return None
        export_values = [_traced_value(graph_module, node) for node in exports]
        if any(
            not isinstance(value, tensorplay.Tensor)
            or tuple(int(item) for item in value.shape) != output_shape
            for value in export_values
        ):
            return None
        output_dtypes = {value.dtype for value in export_values}
        if len(output_dtypes) != 1:
            return None
        output_dtype = next(iter(output_dtypes))
        try:
            built = _build_pointwise_program(
                graph_module,
                output_override=exports[0],
                extra_outputs=exports[1:],
                opcodes=_TRITON_OPCODES,
                nodes=nodes,
                extra_refs={node: index for index, node in enumerate(externals)},
                input_slots=len(externals),
            )
        except (TypeError, ValueError, RuntimeError):
            return None
        if built is None:
            return None
        _external, program, constants, _instructions, output_ref, *extra_refs = built
        output_refs = (output_ref, *tuple(extra_refs[0])) if extra_refs else (output_ref,)
        instruction_count = len(program) // 3
        if not 1 <= instruction_count <= 128 or len(constants) > 4096:
            return None
        if len(output_refs) != len(exports) or len(output_refs) > 32:
            return None
        final_ref = len(externals) + instruction_count - 1
        uniform_output_dtype = all(
            value.dtype == output_dtype for value in external_values
        )
        # Sample tensors describe the plan for the generated kernel: the
        # generator reads only their shape/dtype/device, never the values.
        # Contiguous values go in as-is; non-contiguous ones stand in as
        # uninitialized buffers of the same layout -- allocating them without
        # a fill keeps plan registration off the memory peak of a large
        # compile, and the warm-up launches overwrite whatever the buffers
        # hold.
        examples = [
            value
            if value.is_contiguous()
            else tensorplay.empty(
                tuple(int(item) for item in value.shape),
                dtype=value.dtype,
                device=value.device,
            )
            for value in external_values
        ]
        output_dtype_names = tuple(
            repr(output_dtype) for _ in range(len(output_refs))
        )
        op_name = _register_stax_cuda_pointwise_op(
            program,
            constants,
            output_refs,
            len(externals),
            examples,
            output_dtype_names,
        )
        if op_name is None and (
            not uniform_output_dtype
            or (len(output_refs) == 1 and output_ref != final_ref)
        ):
            # These plans have no execution route without a registered op:
            # the native stride kernel carries one storage dtype, and a
            # single output that is not the program's tail node needs the op
            # to surface it.
            return None
        return (
            frozenset(nodes),
            tuple(externals),
            program,
            constants,
            output_refs,
            tuple(exports),
            op_name,
        )

    plans: dict[Node, tuple[Any, ...]] = {}
    for segment in segments:
        if segment.kind != "pw":
            continue
        cursor = 0
        while cursor < len(segment.nodes):
            best = None
            for end in range(cursor + 2, len(segment.nodes) + 1):
                plan = make_plan(segment.nodes[cursor:end])
                if plan is not None:
                    best = (end, plan)
            if best is None:
                cursor += 1
                continue
            end, plan = best
            for node in segment.nodes[cursor:end]:
                plans[node] = plan
            cursor = end
    return plans


_FUSED_FWD_BINARY_OPS = frozenset({"add", "sub", "mul", "div"})
_FUSED_FWD_UNARY_OPS = frozenset(
    {
        "neg",
        "pos",
        "abs",
        "sin",
        "cos",
        "exp",
        "log",
        "sigmoid",
        "sqrt",
        "square",
        "tanh",
        "relu",
    }
)
_FUSED_FWD_MAX_INSTRUCTIONS = 64
_FUSED_FWD_CUDA_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_FUSED_FWD_CPU_DTYPES = frozenset({"float32"})


class _ForwardFusedProgram:
    """A frozen pointwise program kept alive until every spilled temp is read."""

    __slots__ = ("ops", "temps", "positions")

    def __init__(self, ops, temps):
        self.ops = ops
        self.temps = temps
        self.positions = {node: index for index, node in enumerate(temps)}


class _ForwardPointwiseFuser:
    """Buffers same-shape pointwise ops and emits each buffer as one node.

    Lowering walks the graph in order.  An absorbable pointwise op joins the
    program buffer instead of creating its own native node.  The buffer is
    emitted lazily, when a non-pointwise consumer needs one of the buffered
    results: only the results some outside node actually reads become node
    outputs, and a single-result emit keeps just the dependency closure of
    that result, with the result last in the program (the one-output
    evaluator returns the final temp).  Dead intermediates therefore never
    reach memory -- one chain costs one kernel launch and one write per live
    result instead of one round trip per op.

    Programs are immutable once flushed.  A spilled (not yet emitted) temp
    remembers its program and is re-emitted on demand when a later consumer
    reads it; re-emission rebuilds the same arithmetic over a narrower
    closure, so each consumer pays only for what it uses.
    """

    def __init__(self, graph, values, sample_of, blocked):
        self._graph = graph
        self._values = values
        self._sample_of = sample_of
        self._blocked = blocked
        self._ops: list[tuple[int, tuple[str, Any], tuple[str, Any]]] = []
        self._temps: dict[Any, int] = {}
        self._order: list[Any] = []
        self._shape: tuple[int, ...] | None = None
        self._dtype: str | None = None
        self._is_cuda: bool | None = None
        self._pending: dict[Any, tuple[_ForwardFusedProgram, int]] = {}

    @staticmethod
    def _allowed_dtypes(is_cuda: bool) -> frozenset:
        return _FUSED_FWD_CUDA_DTYPES if is_cuda else _FUSED_FWD_CPU_DTYPES

    def _reset(self) -> None:
        self._ops = []
        self._temps = {}
        self._order = []
        self._shape = None
        self._dtype = None
        self._is_cuda = None

    def _describe(self, sample: Any) -> tuple[tuple[int, ...], str, bool] | None:
        try:
            shape = tuple(int(item) for item in sample.shape)
            name = str(sample.dtype).rsplit(".", 1)[-1]
            is_cuda = bool(sample.device.is_cuda())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None
        return shape, name, is_cuda

    def _bind_layout(self, shape: tuple[int, ...], name: str, is_cuda: bool) -> bool:
        if name not in self._allowed_dtypes(is_cuda):
            return False
        if self._shape is None:
            self._shape = shape
            self._dtype = name
            self._is_cuda = is_cuda
            return True
        if name != self._dtype or is_cuda != self._is_cuda:
            return False
        merged = _broadcast_shape((self._shape, shape))
        if merged is None:
            return False
        self._shape = merged
        return True

    def _operand(self, value: Any) -> tuple[str, Any] | None:
        if isinstance(value, Node):
            if value in self._temps:
                return ("sym", value)
            if value in self._pending:
                if not self._reemit(value):
                    return None
                return ("sym", value)
            native = self._values.get(value)
            if native is None or isinstance(native, (list, tuple)):
                return None
            sample = self._sample_of(value)
            if sample is None:
                return None
            described = self._describe(sample)
            if described is None:
                return None
            if not self._bind_layout(*described):
                return None
            return ("sym", value)
        if _is_scalar(value):
            return ("const", float(value))
        return None

    def absorb(self, node: Node) -> bool:
        """Join a pointwise op into the buffer; False leaves the graph untouched."""
        saved_layout = (self._shape, self._dtype, self._is_cuda)

        def reject() -> bool:
            self._shape, self._dtype, self._is_cuda = saved_layout
            return False

        if node in self._blocked:
            return reject()
        op_name = _target_name(node.target)
        if op_name in _FUSED_FWD_BINARY_OPS:
            if len(node.args) != 2 or node.kwargs:
                return reject()
            lhs = self._operand(node.args[0])
            rhs = self._operand(node.args[1])
            if lhs is None or rhs is None or (lhs[0] == "const" and rhs[0] == "const"):
                return reject()
            operands = (lhs, rhs)
        elif op_name in _FUSED_FWD_UNARY_OPS:
            if len(node.args) != 1:
                return reject()
            if node.kwargs and node.kwargs != {"inplace": False}:
                return reject()
            lhs = self._operand(node.args[0])
            if lhs is None or lhs[0] == "const":
                return reject()
            operands = (lhs, ("const", 0.0))
        else:
            return reject()
        if len(self._ops) >= _FUSED_FWD_MAX_INSTRUCTIONS:
            return reject()
        if self._append(node, _FUSED_PROGRAM_OPCODES[op_name], operands):
            return True
        return reject()

    def _append(self, node: Node, opcode: int, operands) -> bool:
        sample = self._sample_of(node)
        if sample is None:
            return False
        described = self._describe(sample)
        if described is None or not self._bind_layout(*described):
            return False
        index = len(self._ops)
        self._ops.append((opcode, operands[0], operands[1]))
        self._temps[node] = index
        self._order.append(node)
        return True

    def _reemit(self, node: Node) -> bool:
        spec, index = self._pending.pop(node)
        return self._emit(spec, (index,))

    def ensure_for(self, consumer: Node) -> bool:
        """Materialize spilled temps this consumer reads."""
        if not self._pending:
            return True
        groups: dict[int, tuple[_ForwardFusedProgram, list[int]]] = {}
        for value in _nodes((consumer.args, consumer.kwargs)):
            entry = self._pending.get(value)
            if entry is None:
                continue
            spec, index = entry
            key = id(spec)
            group = groups.get(key)
            if group is None:
                group = (spec, [])
                groups[key] = group
            group[1].append(index)
        for spec, indices in groups.values():
            for index in indices:
                del self._pending[spec.temps[index]]
            if not self._emit(spec, tuple(indices)):
                return False
        return True

    def flush_for(self, consumer: Node) -> bool:
        """Emit the buffer before a non-absorbable consumer is lowered."""
        if not self._ops:
            return True
        wanted = [
            self._temps[value]
            for value in _nodes((consumer.args, consumer.kwargs))
            if value in self._temps
        ]
        spec = _ForwardFusedProgram(tuple(self._ops), tuple(self._order))
        self._reset()
        wanted_set = set(wanted)
        for index, temp in enumerate(spec.temps):
            if index not in wanted_set:
                self._pending[temp] = (spec, index)
        if not wanted:
            return True
        return self._emit(spec, tuple(wanted))

    def _emit(self, spec: _ForwardFusedProgram, indices: tuple[int, ...]) -> bool:
        temps = spec.temps
        wanted = {temps[index] for index in indices}
        closure: set[int] = set()
        stack = list(indices)
        while stack:
            index = stack.pop()
            if index in closure:
                continue
            closure.add(index)
            for kind, value in spec.ops[index][1:]:
                if kind == "sym":
                    position = spec.positions.get(value)
                    if position is not None:
                        stack.append(position)
        sequence = sorted(closure)
        if len(indices) == 1:
            # The one-output evaluator returns the final temp of the program.
            sequence.remove(indices[0])
            sequence.append(indices[0])
        mapping = {index: position for position, index in enumerate(sequence)}

        inputs: list[Any] = []
        input_nodes: list[Node] = []
        input_samples: list[Any] = []
        input_pos: dict[int, int] = {}
        constants: list[float] = []
        const_pos: dict[float, int] = {}
        for index in sequence:
            for kind, value in spec.ops[index][1:]:
                if kind == "const" or not isinstance(value, Node):
                    if value not in const_pos:
                        const_pos[value] = len(constants)
                        constants.append(value)
                    continue
                if value in spec.positions:
                    continue
                key = id(value)
                if key not in input_pos:
                    native = self._values.get(value)
                    if native is None or isinstance(native, (list, tuple)):
                        return False
                    input_pos[key] = len(inputs)
                    inputs.append(native)
                    input_nodes.append(value)
                    input_samples.append(self._sample_of(value))
        output_sample = self._sample_of(temps[indices[0]]) if indices else None
        output_shape = None
        output_dtype = None
        if output_sample is not None:
            try:
                output_shape = tuple(int(item) for item in output_sample.shape)
                output_dtype = output_sample.dtype
            except (AttributeError, TypeError, ValueError):
                return False
        if output_shape is None:
            return False
        if len(indices) > 1:
            output_shapes = []
            for index in indices:
                sample = self._sample_of(temps[index])
                try:
                    output_shapes.append(tuple(int(item) for item in sample.shape))
                except (AttributeError, TypeError, ValueError):
                    return False
            if len(set(output_shapes)) > 1:
                return all(self._emit(spec, (index,)) for index in indices)
        anchor = next(
            (
                position
                for position, sample in enumerate(input_samples)
                if sample is not None
                and tuple(int(item) for item in sample.shape) == output_shape
                and sample.dtype == output_dtype
            ),
            None,
        )
        if anchor is None:
            return False
        if anchor:
            anchor_input = inputs[anchor]
            anchor_node = input_nodes[anchor]
            anchor_sample = input_samples[anchor]
            inputs = [anchor_input, *inputs[:anchor], *inputs[anchor + 1 :]]
            input_nodes = [anchor_node, *input_nodes[:anchor], *input_nodes[anchor + 1 :]]
            input_samples = [anchor_sample, *input_samples[:anchor], *input_samples[anchor + 1 :]]
            input_pos = {id(value): position for position, value in enumerate(input_nodes)}
        input_count = len(inputs)
        program: list[int] = []
        for index in sequence:
            opcode, lhs, rhs = spec.ops[index]
            program.append(opcode)
            for kind, value in (lhs, rhs):
                if kind == "const" or not isinstance(value, Node):
                    program.append(-const_pos[value] - 1)
                elif value in spec.positions:
                    program.append(input_count + mapping[spec.positions[value]])
                else:
                    program.append(input_pos[id(value)])
        output_refs = [input_count + mapping[index] for index in indices]
        op_name = None
        if 1 <= input_count <= 32 and len(indices) <= 32:
            examples = (
                input_samples
                if all(
                    value is not None
                    and hasattr(value, "device")
                    and bool(value.device.is_cuda())
                    and value.is_contiguous()
                    and tuple(int(item) for item in value.shape) == output_shape
                    and value.dtype == output_dtype
                    for value in input_samples
                )
                else []
            )
            if not examples:
                op_name = None
            else:
                op_name = _register_stax_cuda_pointwise_op(
                    program,
                    constants,
                    tuple(output_refs),
                    input_count,
                    examples,
                    (repr(output_dtype),) * len(output_refs),
                )
        node = self._graph.create_node(
            "custom_op" if op_name is not None else "fused_pointwise",
            f"fused_pw_{len(self._graph.nodes)}",
        )
        for value in inputs:
            node.add_input(value)
        if op_name is None:
            node.set_int_attr("input_count", input_count)
            node.set_ints_attr("program", program)
            node.set_floats_attr("constants", constants)
            if len(indices) > 1:
                node.set_ints_attr("output_refs", output_refs)
        else:
            node.set_str_attr("op_name", op_name)
        for index in indices:
            self._values[temps[index]] = node.add_output()
        return True

    def finish(self, required: set[Node]) -> None:
        """Materialize every buffered or spilled result the outputs need."""
        if self._ops:
            wanted = [
                self._temps[temp] for temp in self._order if temp in required
            ]
            spec = _ForwardFusedProgram(tuple(self._ops), tuple(self._order))
            self._reset()
            if wanted:
                self._emit(spec, tuple(wanted))
        for temp, (spec, index) in list(self._pending.items()):
            if temp in required:
                del self._pending[temp]
                self._emit(spec, (index,))


def _lower_native(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_fusion: bool = True,
    extra_output_nodes: list[Node] | None = None,
    save_autocast_inputs: bool = False,
) -> _NativeLowering | None:
    """Lower the canonical graph into the native Stax IR when possible."""

    try:
        import tensorplay

        native_module = getattr(tensorplay._C, "_stax", None)
    except (AttributeError, ImportError):
        native_module = None
    if native_module is None:
        return None
    if not hasattr(native_module.Graph, "execute"):
        return None

    try:
        import tensorplay

        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if len(example_inputs) != len(graph_module.graph.placeholders):
        return None
    if any(not isinstance(value, tensor_type) for value in example_inputs):
        # Stax's native ABI is Tensor-only.  The generated GraphModule remains
        # the correct compiled path for scalar/keyword placeholders.
        return None

    graph = native_module.Graph()
    values: dict[Node, Any] = {}
    literal_values: dict[int, Any] = {}
    attribute_targets: list[str] = []
    constant_values: list[Any] = []
    required_nodes = {
        value
        for output in graph_module.graph.outputs
        for value in _nodes(output.args)
    }
    required_nodes.update(extra_output_nodes or [])
    fused_pointwise_plans = (
        _native_fused_pointwise_plans(graph_module, required_nodes)
        if use_fusion and extra_output_nodes is not None
        else {}
    )
    emitted_fused_pointwise_nodes: set[Node] = set()
    folded_convs = _fold_eval_conv_batch_norm(graph_module, example_inputs)
    # convolution inputs, weights, and outputs.  Its generated wrapper uses
    # ``empty_strided`` tensors with the channels-last strides and the current
    # rather than assuming NCHW.  Materialize the same weight layout once for
    # folded constants; runtime activations are converted by a native graph
    # node immediately before the first convolution that consumes them.
    use_channels_last = bool(
        example_inputs
        and example_inputs[0].device.is_cuda()
        and not tensorplay.is_grad_enabled()
    )
    if use_channels_last and folded_convs:
        try:
            folded_with_layout: dict[Node, tuple[Node, Any, Any]] = {}
            for conv, (batch_norm, weight, bias) in folded_convs.items():
                # NCHW logical shape, NHWC physical storage, then reinterpret
                # as NCHW. This layout is represented by generated weight
                # strides, e.g. [K*C*R*S, 1, S*C, C].
                physical_weight = weight.permute((0, 2, 3, 1)).clone()
                channels_last_weight = physical_weight.permute((0, 3, 1, 2))
                folded_with_layout[conv] = (
                    batch_norm,
                    channels_last_weight,
                    bias,
                )
            folded_convs = folded_with_layout
        except (AttributeError, RuntimeError, TypeError, ValueError):
            # Keep the already validated folding path if this backend cannot
            # materialize the specialized layout for a particular dtype.
            use_channels_last = False
    folded_native_inputs: dict[Node, tuple[Any, Any]] = {}
    folded_batch_norm_to_conv = {
        batch_norm: conv for conv, (batch_norm, _, _) in folded_convs.items()
    }

    # belong in the native ABI.  Inference Conv+BN folding leaves the original
    # parameter nodes in the captured graph, but the native graph consumes only the folded
    # weight/bias constants.  Passing those dead tensors through Python and
    # C++ on every invocation is pure call-boundary overhead.
    live_attribute_nodes: set[Node] = set()
    visited_nodes: set[Node] = set()

    def visit_native_dependency(value: Any) -> None:
        if isinstance(value, Node):
            if value in visited_nodes:
                return
            visited_nodes.add(value)
            if value.op == "get_attr":
                live_attribute_nodes.add(value)
                return
            if value in folded_convs or value in folded_batch_norm_to_conv:
                visit_native_dependency(value.args[0])
                return
            for argument in value.args:
                visit_native_dependency(argument)
            for argument in value.kwargs.values():
                visit_native_dependency(argument)
            return
        if isinstance(value, (tuple, list)):
            for item in value:
                visit_native_dependency(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit_native_dependency(item)
        elif isinstance(value, slice):
            visit_native_dependency(value.start)
            visit_native_dependency(value.stop)
            visit_native_dependency(value.step)

    for output in graph_module.graph.outputs:
        visit_native_dependency(output.args)
    for extra_node in extra_output_nodes or []:
        visit_native_dependency(extra_node)
    # Buffer-update nodes are epilogue state: even when no consumer reads
    # their value, the update must run, so their module-state operands stay
    # live inputs of the native graph.
    for node in graph_module.graph.nodes:
        if node.op == "call_method" and node.target in {"add_", "sub_", "mul_"}:
            visit_native_dependency(node)

    fused_relu_convs: dict[Node, Node] = {}
    fused_add_relus: dict[Node, Node] = {}
    fused_relu_nodes: set[Node] = set()
    layout_values: dict[Node, bool] = {}
    channels_last_values: dict[Node, Any] = {}
    autocast_values: dict[tuple[Node, Any], Any] = {}
    mutations: list[tuple[int, Any]] = []
    # Keep bias in the convolution during training so the backend can place
    # it in the same execution plan as the convolution.  The inference path
    # retains the specialized post-convolution handling.
    peel_conv_bias = bool(
        example_inputs
        and example_inputs[0].device.is_cuda()
        and not tensorplay.is_grad_enabled()
    )
    autocast_dtype = (
        tensorplay.get_autocast_dtype("cuda")
        if save_autocast_inputs and tensorplay.is_autocast_enabled("cuda")
        else None
    )
    # This is the same producer/sole-consumer legality check used by
    # training graph because add_relu has a generated autograd formula; the
    # Conv->ReLU and Conv+BN folding paths remain inference-only.
    if use_fusion:
        for relu in graph_module.graph.nodes:
            if (
                relu.op != "call_function"
                or _target_name(relu.target) != "relu"
                or len(relu.args) != 1
                or relu.kwargs not in ({}, {"inplace": False}, {"inplace": True})
            ):
                continue
            source = relu.args[0]
            if (
                isinstance(source, Node)
                and source.op == "call_function"
                and _target_name(source.target) == "add"
                and source.users == {relu}
            ):
                if len(source.args) == 2:
                    lhs, rhs = source.args
                    alpha = 1
                elif len(source.args) == 3:
                    lhs, rhs, alpha = source.args
                else:
                    continue
                if (
                    alpha == 1
                    and isinstance(lhs, Node)
                    and isinstance(rhs, Node)
                ):
                    fused_add_relus[source] = relu
                continue
            if tensorplay.is_grad_enabled():
                continue
            conv = folded_batch_norm_to_conv.get(source)
            if conv is None:
                conv = source
                if not (
                    isinstance(conv, Node)
                    and conv.op == "call_function"
                    and _target_name(conv.target) == "conv2d"
                ):
                    continue
            if source.users != {relu} or conv.users != ({source} if source is not conv else {relu}):
                continue
            fused_relu_convs[conv] = relu
    input_positions: dict[Node, int] = {}
    next_input_index = 0
    for node in graph_module.graph.placeholders:
        values[node] = graph.add_input()
        input_positions[node] = next_input_index
        next_input_index += 1
        layout_values[node] = False

    def channels_last_value(node: Any) -> Any | None:
        """Return the native value in the preferred 4-D layout."""

        if not isinstance(node, Node) or node not in values:
            return None
        if not use_channels_last:
            return values[node]
        if layout_values.get(node, False):
            return values[node]
        cached = channels_last_values.get(node)
        if cached is not None:
            return cached
        reorder = graph.create_node("channels_last", f"{node.name}_channels_last")
        reorder.add_input(values[node])
        converted = reorder.add_output()
        channels_last_values[node] = converted
        return converted

    def saved_autocast_value(node: Node, value: Any) -> Any:
        if autocast_dtype is None:
            return value
        sample = _traced_value(graph_module, node)
        if getattr(sample, "dtype", None) != tensorplay.float32:
            return value
        key = (node, autocast_dtype)
        cached = autocast_values.get(key)
        if cached is not None:
            return cached
        cast = graph.create_node("cast", f"{node.name}_autocast")
        cast.add_input(value)
        cast.set_str_attr("dtype", str(autocast_dtype).rsplit(".", 1)[-1])
        converted = cast.add_output()
        autocast_values[key] = converted
        return converted

    # Register all live module attributes before synthetic folded weights so
    # Graph::execute's input order is independent of where get_attr nodes are
    # placed in the Python graph.
    for node in graph_module.graph.nodes:
        if node.op != "get_attr" or node not in live_attribute_nodes:
            continue
        attribute = graph_module._get_attr(node.target)
        if not isinstance(attribute, tensor_type):
            return None
        values[node] = graph.add_input()
        input_positions[node] = next_input_index
        next_input_index += 1
        attribute_targets.append(node.target)

    for node in graph_module.graph.nodes:
        folded = folded_convs.get(node)
        if folded is None:
            continue
        _, folded_weight, folded_bias = folded
        native_weight = graph.add_input()
        native_bias = graph.add_input()
        next_input_index += 2
        folded_native_inputs[node] = (native_weight, native_bias)
        constant_values.extend((folded_weight, folded_bias))

    def fuser_sample(node: Node) -> Any:
        sample = _traced_value(graph_module, node)
        if sample is not None:
            return sample
        if node.op == "placeholder":
            try:
                return example_inputs[graph_module.graph.placeholders.index(node)]
            except (ValueError, IndexError):
                return None
        if node.op == "get_attr":
            try:
                return graph_module._get_attr(node.target)
            except (AttributeError, KeyError, RuntimeError):
                return None
        return None

    fuser = _ForwardPointwiseFuser(
        graph,
        values,
        fuser_sample,
        set(fused_pointwise_plans)
        | fused_relu_nodes
        | set(fused_add_relus)
        | set(fused_add_relus.values()),
    )

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if node.op not in {"call_function", "call_method"}:
            return None
        if use_fusion:
            try:
                if not fuser.ensure_for(node):
                    return None
                if node not in emitted_fused_pointwise_nodes and fuser.absorb(node):
                    continue
                if not fuser.flush_for(node):
                    return None
            except (KeyError, RuntimeError, TypeError, ValueError):
                return None
            if node in emitted_fused_pointwise_nodes:
                continue
        pointwise_plan = fused_pointwise_plans.get(node) if use_fusion else None
        if pointwise_plan is not None:
            (
                plan_nodes,
                externals,
                program,
                constants,
                output_refs,
                exports,
                op_name,
            ) = pointwise_plan
            if any(values.get(external) is None for external in externals):
                return None
            native_node = graph.create_node(
                "custom_op" if op_name is not None else "fused_pointwise",
                f"{node.name}_fused",
            )
            if op_name is not None:
                native_node.set_str_attr("op_name", op_name)
            else:
                native_node.set_int_attr("input_count", len(externals))
                native_node.set_ints_attr("program", program)
                native_node.set_floats_attr("constants", constants)
            for external in externals:
                native_node.add_input(values[external])
            native_outputs = {
                exported: native_node.add_output() for exported in exports
            }
            if op_name is None and len(exports) > 1:
                native_node.set_ints_attr("output_refs", output_refs)
            for exported, native_output in native_outputs.items():
                values[exported] = native_output
                layout_values[exported] = False
            emitted_fused_pointwise_nodes.update(plan_nodes)
            continue
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) == 2
            and isinstance(node.args[0], Node)
            and isinstance(node.args[1], tuple)
            and len(node.args[1]) == 2
            and isinstance(node.args[1][0], slice)
            and node.args[1][0] == slice(None)
            and node.args[1][1] is None
            and node.args[0] in values
        ):
            native_node = graph.create_node("unsqueeze", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("dim", 1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue
        if (
            node.op == "call_function"
            and node.target is operator.getitem
            and len(node.args) == 2
            and isinstance(node.args[0], Node)
            and isinstance(node.args[1], int)
            and isinstance(values.get(node.args[0]), tuple)
        ):
            source_values = values[node.args[0]]
            index = node.args[1]
            if index < 0:
                index += len(source_values)
            if index < 0 or index >= len(source_values):
                return None
            values[node] = source_values[index]
            layout_values[node] = False
            continue
        op_name = _target_name(node.target)
        # User-defined operators enter the native dispatcher bridge.  Their
        # output arity is carried by the graph metadata because the native
        # value table reserves one value per result.
        if isinstance(node.target, _CustomOpDef):
            if node.kwargs or not node.args:
                return None
            native_node = graph.create_node("custom_op", node.name)
            native_node.set_str_attr("op_name", node.target.name)
            for argument in node.args:
                resolved = values.get(argument) if isinstance(argument, Node) else None
                if resolved is None or isinstance(resolved, tuple):
                    return None
                native_node.add_input(resolved)
            custom = node.meta.get("custom")
            output_count = 1
            if isinstance(custom, dict):
                output_count = int(custom.get("nested_output_count", 1))
            if output_count < 1:
                return None
            output_values = tuple(native_node.add_output() for _ in range(output_count))
            values[node] = output_values[0] if output_count == 1 else output_values
            continue
        # ``ReLU(inplace=True)`` is an aliasing detail of the captured graph.
        # The native kernel accepts that schema explicitly; other keyword
        # combinations are rejected by this lowering.
        if node.kwargs:
            if op_name == "relu":
                if node.kwargs not in ({"inplace": False}, {"inplace": True}):
                    return None
            elif op_name == "silu":
                # The native graph has no mutation output for SiLU. Preserve
                # the in-place contract by using the general executor there.
                if node.kwargs not in ({"inplace": False},):
                    return None
            elif op_name not in {
                "sum",
                "mean",
                "add",
                "sub",
                "silu",
                "gelu",
                "softmax",
                "log_softmax",
                "layer_norm",
                "where",
                "clamp",
                "clamp_min",
                "clamp_max",
                "unsqueeze",
                "squeeze",
                "zeros_like",
                "permute_backward",
                "expand",
                "cat",
                "chunk",
                "split",
                "split_with_sizes",
                "unbind",
                "slice",
                "stack",
                "repeat",
                "embedding",
                "scaled_dot_product_attention",
            }:
                return None
        if op_name in {"add_", "sub_", "mul_"}:
            # In-place updates of module state (e.g. a batch counter) have no
            # aliasing target inside the functional native graph.  Lower the
            # arithmetic functionally; the executable wrapper copies the
            # result back into the source input after every execution.
            if len(node.args) != 2:
                return None
            source_node, delta = node.args
            if not isinstance(source_node, Node) or source_node not in input_positions:
                return None
            if not _is_scalar(delta) and not (
                isinstance(delta, Node) and delta in values
            ):
                return None
            native_node = graph.create_node(op_name[:-1], node.name)
            native_node.add_input(values[source_node])
            if _is_scalar(delta):
                _set_scalar_attr(native_node, delta, 1)
            else:
                native_node.add_input(values[delta])
            mutation_value = native_node.add_output()
            values[node] = mutation_value
            layout_values[node] = False
            mutations.append((input_positions[source_node], mutation_value))
            continue
        if op_name not in _NATIVE_OPS:
            return None

        if op_name == "relu" and node in fused_relu_nodes:
            source = node.args[0]
            if source not in values:
                return None
            # The producer was lowered with a fused Conv+ReLU primitive; the
            # ReLU value is an alias of that already-activated output.
            values[node] = values[source]
            layout_values[node] = layout_values.get(source, False)
            continue

        if op_name == "relu" and node in fused_add_relus.values():
            source = node.args[0]
            if not isinstance(source, Node) or source not in values:
                return None
            # The residual add is lowered as add_relu below, so the ReLU
            # node observes the already-activated output without another
            # native launch.
            values[node] = values[source]
            layout_values[node] = layout_values.get(source, False)
            continue

        def node_value(value: Any) -> Any | None:
            if isinstance(value, Node):
                return values.get(value)
            if isinstance(value, tensor_type):
                key = id(value)
                resolved = literal_values.get(key)
                if resolved is None:
                    resolved = graph.add_input()
                    literal_values[key] = resolved
                    constant_values.append(value)
                return resolved
            return None

        def sample_value(value: Any) -> Any | None:
            if not isinstance(value, Node):
                return None
            sample = _traced_value(graph_module, value)
            if sample is not None:
                return sample
            if value.op == "placeholder":
                try:
                    return example_inputs[graph_module.graph.placeholders.index(value)]
                except (ValueError, IndexError):
                    return None
            if value.op == "get_attr":
                try:
                    return graph_module._get_attr(value.target)
                except (AttributeError, KeyError, RuntimeError):
                    return None
            return None

        def add_tensor_input(native_node: Any, value: Any) -> bool:
            resolved = node_value(value)
            if resolved is None or isinstance(resolved, tuple):
                return False
            native_node.add_input(resolved)
            return True

        if op_name in {"eq", "ne", "lt", "le", "gt", "ge"}:
            if len(node.args) != 2 or node.kwargs:
                return None
            lhs, rhs = node.args
            lhs_is_tensor = isinstance(lhs, (Node, tensor_type))
            rhs_is_tensor = isinstance(rhs, (Node, tensor_type))
            if lhs_is_tensor == rhs_is_tensor:
                if not lhs_is_tensor:
                    return None
            elif not lhs_is_tensor and not _is_scalar(lhs):
                return None
            elif not rhs_is_tensor and not _is_scalar(rhs):
                return None
            native_node = graph.create_node(op_name, node.name)
            if lhs_is_tensor and not add_tensor_input(native_node, lhs):
                return None
            if rhs_is_tensor and not add_tensor_input(native_node, rhs):
                return None
            if not lhs_is_tensor:
                _set_scalar_attr(native_node, lhs, 0)
            elif not rhs_is_tensor:
                _set_scalar_attr(native_node, rhs, 1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "where":
            if len(node.args) != 3 or node.kwargs:
                return None
            condition, self_value, other_value = node.args
            if not isinstance(condition, Node) or condition not in values:
                return None
            self_is_tensor = isinstance(self_value, (Node, tensor_type))
            other_is_tensor = isinstance(other_value, (Node, tensor_type))
            if not self_is_tensor and not _is_scalar(self_value):
                return None
            if not other_is_tensor and not _is_scalar(other_value):
                return None
            native_node = graph.create_node("where", node.name)
            native_node.add_input(values[condition])
            if self_is_tensor:
                if not add_tensor_input(native_node, self_value):
                    return None
            elif not _set_named_scalar_attr(native_node, "self_scalar", self_value):
                return None
            if other_is_tensor:
                if not add_tensor_input(native_node, other_value):
                    return None
            elif not _set_named_scalar_attr(native_node, "other_scalar", other_value):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"clamp", "clamp_min", "clamp_max"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs and set(node.kwargs) - {"min", "max"}:
                return None
            source = node.args[0]
            if source not in values:
                return None
            args = list(node.args[1:])
            kwargs = dict(node.kwargs or {})
            if op_name == "clamp":
                if len(args) > 2 or (args and "min" in kwargs) or (
                    len(args) > 1 and "max" in kwargs
                ):
                    return None
                lower = args[0] if args else kwargs.get("min")
                upper = args[1] if len(args) > 1 else kwargs.get("max")
                if lower is None and upper is None:
                    return None
            else:
                if len(args) != 1 or kwargs:
                    return None
                lower = args[0] if op_name == "clamp_min" else None
                upper = args[0] if op_name == "clamp_max" else None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            for key, bound in (("min", lower), ("max", upper)):
                tensor_bound = isinstance(bound, (Node, tensor_type))
                native_node.set_int_attr(
                    "min_tensor" if key == "min" else "max_tensor",
                    int(tensor_bound),
                )
                if bound is None:
                    continue
                if tensor_bound:
                    if not add_tensor_input(native_node, bound):
                        return None
                elif not _set_named_scalar_attr(
                    native_node, "min_scalar" if key == "min" else "max_scalar", bound
                ):
                    return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "gelu":
            if len(node.args) != 1 or not isinstance(node.args[0], Node):
                return None
            approximate = node.kwargs.get("approximate", "none")
            if not isinstance(approximate, str) or approximate not in {"none", "tanh"}:
                return None
            native_node = graph.create_node("gelu", node.name)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            native_node.set_str_attr("approximate", approximate)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"softmax", "log_softmax"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            source = node.args[0]
            if node.op == "call_method":
                if len(node.args) not in {2, 3} or kwargs:
                    return None
                dim = node.args[1]
                dtype = node.args[2] if len(node.args) == 3 else tensorplay.undefined
            else:
                if len(node.args) not in {2, 3}:
                    return None
                dim = node.args[1]
                dtype = node.args[2] if len(node.args) == 3 else tensorplay.undefined
            if source not in values or isinstance(dim, bool) or not isinstance(dim, int):
                return None
            if dtype is None:
                dtype = tensorplay.undefined
            dtype_name = getattr(dtype, "name", None)
            if dtype_name != "undefined":
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim", int(dim))
            native_node.set_int_attr("dtype", 22)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "layer_norm":
            if len(node.args) != 5 or node.kwargs:
                return None
            source, normalized_shape, weight, bias, eps = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            shape = _int_list(normalized_shape)
            if shape is None or not isinstance(eps, numbers.Real):
                return None
            native_node = graph.create_node("layer_norm", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("normalized_shape", shape)
            for attr_name, optional_node in (("has_weight", weight), ("has_bias", bias)):
                if optional_node is None:
                    native_node.set_int_attr(attr_name, 0)
                elif add_tensor_input(native_node, optional_node):
                    native_node.set_int_attr(attr_name, 1)
                else:
                    return None
            native_node.set_float_attr("eps", float(eps))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "transpose":
            if len(node.args) != 3 or node.kwargs:
                return None
            source, dim0, dim1 = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            if any(isinstance(dim, bool) or not isinstance(dim, int) for dim in (dim0, dim1)):
                return None
            native_node = graph.create_node("transpose", node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim0", int(dim0))
            native_node.set_int_attr("dim1", int(dim1))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "select":
            if len(node.args) != 3 or node.kwargs:
                return None
            source, dim, index = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            if any(isinstance(item, bool) or not isinstance(item, int) for item in (dim, index)):
                return None
            native_node = graph.create_node("select", node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim", int(dim))
            native_node.set_int_attr("index", int(index))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "slice":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs:
                return None
            args = list(node.args[1:])
            if len(args) > 4:
                return None
            args.extend([None] * (4 - len(args)))
            dim, start, end, step = args
            if not isinstance(dim, int) or isinstance(dim, bool):
                return None
            if step is None:
                step = 1
            if not isinstance(step, int) or isinstance(step, bool) or step <= 0:
                return None
            native_node = graph.create_node("slice", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("dim", int(dim))
            native_node.set_int_attr("step", int(step))
            for key, bound in (("start", start), ("end", end)):
                if bound is None:
                    native_node.set_int_attr("has_" + key, 0)
                elif isinstance(bound, int) and not isinstance(bound, bool):
                    native_node.set_int_attr("has_" + key, 1)
                    native_node.set_int_attr(key, int(bound))
                else:
                    return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "stack":
            if not node.args or not isinstance(node.args[0], (tuple, list)):
                return None
            tensors = node.args[0]
            dim = node.args[1] if len(node.args) > 1 else 0
            if node.kwargs or not tensors or not isinstance(dim, int):
                return None
            native_node = graph.create_node("stack", node.name)
            for item in tensors:
                if not add_tensor_input(native_node, item):
                    return None
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "repeat":
            if len(node.args) != 2 or node.kwargs or not isinstance(node.args[0], Node):
                return None
            repeats = _int_list(node.args[1])
            if repeats is None or node.args[0] not in values:
                return None
            native_node = graph.create_node("repeat", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_ints_attr("repeats", repeats)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "index_select" or op_name == "gather":
            kwargs = dict(node.kwargs or {})
            if node.op == "call_method":
                if set(kwargs) - {"dim", "index"} or not node.args:
                    return None
                source = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "index" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                index = positional[1] if len(positional) > 1 else kwargs.get("index")
                if dim is None or index is None:
                    return None
            else:
                if len(node.args) > 3 or set(kwargs) - {"dim", "index"}:
                    return None
                source = node.args[0]
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "index" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                index = node.args[2] if len(node.args) > 2 else kwargs.get("index")
                if dim is None or index is None:
                    return None
            if not isinstance(source, Node) or source not in values:
                return None
            if not isinstance(dim, int) or isinstance(dim, bool):
                return None
            if node_value(index) is None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            if not add_tensor_input(native_node, index):
                return None
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "embedding":
            if len(node.args) != 7 or node.kwargs:
                return None
            index_node, weight_node, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse = node.args
            if not isinstance(index_node, Node) or not isinstance(weight_node, Node):
                return None
            if index_node not in values or weight_node not in values:
                return None
            if max_norm is not None or norm_type != 2.0:
                return None
            if not isinstance(padding_idx, (int, type(None))) or isinstance(padding_idx, bool):
                return None
            if not isinstance(scale_grad_by_freq, bool) or not isinstance(sparse, bool):
                return None
            native_node = graph.create_node("embedding", node.name)
            native_node.add_input(values[weight_node])
            native_node.add_input(values[index_node])
            native_node.set_int_attr("padding_idx", -1 if padding_idx is None else int(padding_idx))
            native_node.set_int_attr("scale_grad_by_freq", int(scale_grad_by_freq))
            native_node.set_int_attr("sparse", int(sparse))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"constant_pad_nd", "pad"}:
            if (len(node.args) != 3 and len(node.args) != 4) or node.kwargs:
                return None
            if len(node.args) == 4:
                source, pad, mode, value = node.args
                if mode != "constant":
                    return None
            else:
                source, pad, value = node.args
            if not isinstance(source, Node) or source not in values:
                return None
            padding = _int_list(pad)
            if padding is None or not _is_scalar(value):
                return None
            native_node = graph.create_node("constant_pad_nd", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("pad", padding)
            if not _set_named_scalar_attr(native_node, "value", value):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"view", "reshape"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if kwargs:
                return None
            shape_args = list(node.args[1:])
            if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)):
                shape_args = list(shape_args[0])
            shape = _int_list(shape_args)
            if shape is None or node.args[0] not in values:
                return None
            native_node = graph.create_node("reshape", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_ints_attr("shape", shape)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "permute":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs or node.args[0] not in values:
                return None
            dims_args = list(node.args[1:])
            if len(dims_args) == 1 and isinstance(dims_args[0], (tuple, list)):
                dims_args = list(dims_args[0])
            dims = _int_list(dims_args)
            if dims is None:
                return None
            native_node = graph.create_node("permute", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_ints_attr("dims", dims)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "contiguous":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.args[0] not in values:
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"memory_format"} or len(node.args) > 2:
                return None
            memory_format = node.args[1] if len(node.args) > 1 else kwargs.get(
                "memory_format", 0
            )
            if not isinstance(memory_format, int) or isinstance(memory_format, bool):
                return None
            native_node = graph.create_node("contiguous", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("memory_format", int(memory_format))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "unsqueeze":
            if len(node.args) != 2 or not isinstance(node.args[0], Node):
                return None
            if node.kwargs or node.args[0] not in values:
                return None
            dim = node.args[1]
            if isinstance(dim, bool) or not isinstance(dim, int):
                return None
            native_node = graph.create_node("unsqueeze", node.name)
            native_node.add_input(values[node.args[0]])
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "squeeze":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim"} or len(node.args) > 2:
                return None
            if len(node.args) > 1 and "dim" in kwargs:
                return None
            dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
            source = node.args[0]
            if source not in values:
                return None
            native_node = graph.create_node("squeeze", node.name)
            native_node.add_input(values[source])
            if dim is not None:
                if isinstance(dim, bool):
                    return None
                if isinstance(dim, int):
                    native_node.set_int_attr("dim", int(dim))
                elif not _set_int_list_attr(native_node, "dims", dim):
                    return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "zeros_like":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            if node.kwargs or len(node.args) not in {1, 4}:
                # The native node preserves the template dtype and device.
                # Explicit metadata changes require a typed factory node and
                # are kept on the general lowering path.
                return None
            if len(node.args) == 4 and (
                node.args[1] is not tensorplay.undefined
                or node.args[2] is not None
                or node.args[3] is not False
            ):
                return None
            source = node.args[0]
            if source not in values:
                return None
            sample = sample_value(source)
            if sample is None:
                sample = sample_value(node)
            try:
                shape = [int(item) for item in sample.shape]
            except (AttributeError, TypeError, ValueError):
                return None
            native_node = graph.create_node("zeros_like", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("shape", shape)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "permute_backward":
            if len(node.args) != 3 or node.kwargs:
                return None
            grad_node, source_node, dims = node.args
            if (
                not isinstance(grad_node, Node)
                or not isinstance(source_node, Node)
                or grad_node not in values
                or source_node not in values
                or not _set_int_list_attr(
                    native_node := graph.create_node("permute_backward", node.name),
                    "dims",
                    dims,
                )
            ):
                return None
            native_node.add_input(values[grad_node])
            native_node.add_input(values[source_node])
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "float":
            if len(node.args) != 1 or node.kwargs or not isinstance(node.args[0], Node):
                return None
            if node.args[0] not in values:
                return None
            native_node = graph.create_node("float", node.name)
            native_node.add_input(values[node.args[0]])
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "group_norm":
            if len(node.args) != 5 or node.kwargs:
                return None
            input_node, num_groups, weight_node, bias_node, eps = node.args
            if (
                not isinstance(input_node, Node)
                or input_node not in values
                or isinstance(num_groups, bool)
                or not isinstance(num_groups, int)
                or not isinstance(eps, numbers.Real)
            ):
                return None
            native_node = graph.create_node("group_norm", node.name)
            native_node.add_input(values[input_node])
            native_node.set_int_attr("num_groups", int(num_groups))
            native_node.set_float_attr("eps", float(eps))
            for attr_name, optional_node in (
                ("has_weight", weight_node),
                ("has_bias", bias_node),
            ):
                if optional_node is None:
                    native_node.set_int_attr(attr_name, 0)
                    continue
                if not add_tensor_input(native_node, optional_node):
                    return None
                native_node.set_int_attr(attr_name, 1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"avg_pool1d", "avg_pool2d", "avg_pool3d"}:
            if len(node.args) not in {6, 7} or node.kwargs:
                return None
            input_node, kernel_size, stride, padding, ceil_mode, count_include_pad = node.args[:6]
            divisor = node.args[6] if len(node.args) == 7 else None
            if not isinstance(input_node, Node) or input_node not in values:
                return None
            if not isinstance(ceil_mode, bool) or not isinstance(count_include_pad, bool):
                return None
            spatial_rank = int(op_name[-2])
            kernel = _spatial_int_list(kernel_size, length=spatial_rank)
            stride_values = _spatial_int_list(
                stride, default=kernel, length=spatial_rank
            )
            padding_values = _spatial_int_list(
                padding, default=[0] * spatial_rank, length=spatial_rank
            )
            if kernel is None or stride_values is None or padding_values is None:
                return None
            if divisor is not None and (
                isinstance(divisor, bool) or not isinstance(divisor, int)
            ):
                return None
            if op_name == "avg_pool1d" and divisor is not None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[input_node])
            native_node.set_ints_attr("kernel_size", kernel)
            native_node.set_ints_attr("stride", stride_values)
            native_node.set_ints_attr("padding", padding_values)
            native_node.set_int_attr("ceil_mode", int(ceil_mode))
            native_node.set_int_attr("count_include_pad", int(count_include_pad))
            if divisor is not None:
                native_node.set_int_attr("divisor_override", int(divisor))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "interpolate":
            if len(node.args) != 7 or node.kwargs:
                return None
            input_node, size, scale_factor, mode, align_corners, recompute, antialias = node.args
            if (
                not isinstance(input_node, Node)
                or input_node not in values
                or scale_factor is not None
                or mode != "nearest"
                or align_corners is not None
                or recompute is not None
                or antialias is not False
            ):
                return None
            sample = sample_value(input_node)
            try:
                spatial_rank = int(len(sample.shape)) - 2
            except (AttributeError, TypeError):
                return None
            if spatial_rank not in (1, 2, 3):
                return None
            output_size = _spatial_int_list(size, length=spatial_rank)
            if output_size is None:
                return None
            native_node = graph.create_node("interpolate", node.name)
            native_node.add_input(values[input_node])
            native_node.set_ints_attr("output_size", output_size)
            native_node.set_int_attr("spatial_rank", spatial_rank)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "scaled_dot_product_attention":
            if len(node.args) < 3 or len(node.args) > 8:
                return None
            query, key, value_node = node.args[:3]
            if any(not isinstance(item, Node) or item not in values for item in (query, key, value_node)):
                return None
            names = ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")
            options = dict(node.kwargs or {})
            for index, item in enumerate(node.args[3:]):
                if names[index] in options:
                    return None
                options[names[index]] = item
            if set(options) - set(names):
                return None
            if options.get("attn_mask") is not None:
                return None
            dropout_p = options.get("dropout_p", 0.0)
            if not isinstance(dropout_p, numbers.Real) or float(dropout_p) != 0.0:
                return None
            if options.get("scale") is not None or options.get("enable_gqa", False):
                return None
            is_causal = options.get("is_causal", False)
            if not isinstance(is_causal, bool):
                return None
            attention_inputs = []
            for input_node in (query, key, value_node):
                input_value = values[input_node]
                if autocast_dtype is not None and getattr(node.meta.get("val"), "dtype", None) == autocast_dtype:
                    input_value = saved_autocast_value(input_node, input_value)
                attention_inputs.append(input_value)
            native_node = graph.create_node("scaled_dot_product_attention", node.name)
            for input_value in attention_inputs:
                native_node.add_input(input_value)
            native_node.set_int_attr("is_causal", int(is_causal))
            native_node.set_int_attr("impl", 0)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "dropout":
            if len(node.args) != 4 or node.kwargs or not isinstance(node.args[0], Node):
                return None
            input_node, probability, training, inplace = node.args
            if input_node not in values or not isinstance(probability, numbers.Real):
                return None
            if not isinstance(training, bool) or not isinstance(inplace, bool):
                return None
            if float(probability) == 0.0 or not training:
                values[node] = values[input_node]
                layout_values[node] = layout_values.get(input_node, False)
                continue
            if inplace:
                return None
            native_node = graph.create_node("dropout", node.name)
            native_node.add_input(values[input_node])
            native_node.set_float_attr("p", float(probability))
            native_node.set_int_attr("training", int(training))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        alpha_form = op_name in {"add", "sub"} and (
            len(node.args) == 3
            or (len(node.args) == 2 and "alpha" in (node.kwargs or {}))
        )
        if op_name in {"add", "sub", "mul", "div", "pow", "matmul"} and not alpha_form:
            if len(node.args) != 2 or node.kwargs:
                return None
            lhs, rhs = node.args
            lhs_is_tensor = isinstance(lhs, (Node, tensor_type))
            rhs_is_tensor = isinstance(rhs, (Node, tensor_type))
            if not lhs_is_tensor and not _is_scalar(lhs):
                return None
            if not rhs_is_tensor and not _is_scalar(rhs):
                return None
            if not lhs_is_tensor and not rhs_is_tensor:
                return None
            native_node = graph.create_node(op_name, node.name)
            if lhs_is_tensor:
                if not add_tensor_input(native_node, lhs):
                    return None
            if rhs_is_tensor:
                if not add_tensor_input(native_node, rhs):
                    return None
            if not lhs_is_tensor:
                _set_scalar_attr(native_node, lhs, 0)
            elif not rhs_is_tensor:
                _set_scalar_attr(native_node, rhs, 1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"conv2d", "conv2d_relu"}:
            if len(node.args) != 7:
                return None
            input_node, weight_node, bias_node, stride, padding, dilation, groups = node.args
            fused_relu = fused_relu_convs.get(node)
            use_conv_relu = op_name == "conv2d_relu" or (
                fused_relu is not None and not peel_conv_bias
            )
            # A consumer must be created after the layout-conversion node
            # feeding it: native execution walks nodes in creation order.
            conv_input = channels_last_value(input_node)
            if conv_input is None:
                return None
            if autocast_dtype is not None:
                conv_input = saved_autocast_value(input_node, conv_input)
            folded_inputs = folded_native_inputs.get(node)
            bias_input = None
            bias_tensor = None
            if folded_inputs is not None:
                weight_input = folded_inputs[0]
                bias_input = folded_inputs[1]
                folded_spec = folded_convs.get(node)
                bias_tensor = folded_spec[2] if folded_spec is not None else None
            else:
                weight_input = channels_last_value(weight_node)
                if weight_input is None:
                    return None
                if autocast_dtype is not None:
                    weight_input = saved_autocast_value(weight_node, weight_input)
                if bias_node is not None:
                    bias_input = node_value(bias_node)
                    if bias_input is None:
                        return None
                    if peel_conv_bias and not use_conv_relu:
                        try:
                            bias_tensor = graph_module._get_attr(bias_node.target)
                        except (AttributeError, TypeError):
                            return None
            native_node = graph.create_node(
                "conv2d_relu" if use_conv_relu else "conv2d",
                node.name,
            )
            native_node.add_input(conv_input)
            native_node.add_input(weight_input)
            if bias_node is None and folded_inputs is None:
                native_node.set_int_attr("has_bias", 0)
            if bias_input is None:
                native_node.set_int_attr("has_bias", 0)
            elif not peel_conv_bias or use_conv_relu:
                native_node.add_input(bias_input)
                native_node.set_int_attr("has_bias", 1)
            else:
                # the cuDNN call because cuDNN is slower with it.  Keep the
                # bias as a broadcast pointwise input after the convolution.
                native_node.set_int_attr("has_bias", 0)
            for key, value, default in (
                ("stride", stride, [1, 1]),
                ("padding", padding, [0, 0]),
                ("dilation", dilation, [1, 1]),
            ):
                normalized = _spatial_int_list(value, default=default)
                if normalized is None or not _set_int_list_attr(
                    native_node, key, normalized
                ):
                    return None
            if isinstance(groups, bool) or not isinstance(groups, int):
                return None
            native_node.set_int_attr("groups", int(groups))
            conv_value = native_node.add_output()
            values[node] = conv_value
            layout_values[node] = use_channels_last

            if peel_conv_bias and bias_input is not None and not use_conv_relu:
                if bias_tensor is None or not hasattr(bias_tensor, "shape"):
                    return None
                bias_shape = tuple(int(item) for item in bias_tensor.shape)
                if len(bias_shape) != 1:
                    return None
                bias_view = graph.create_node("reshape", f"{node.name}_bias_view")
                bias_view.add_input(bias_input)
                bias_view.set_ints_attr("shape", [1, bias_shape[0], 1, 1])
                bias_value = bias_view.add_output()
                add_node = graph.create_node(
                    "add_relu" if fused_relu is not None else "add",
                    f"{node.name}_bias_add",
                )
                add_node.add_input(conv_value)
                add_node.add_input(bias_value)
                values[node] = add_node.add_output()
                layout_values[node] = use_channels_last and fused_relu is not None
                if fused_relu is not None:
                    fused_relu_nodes.add(fused_relu)
            elif use_conv_relu and fused_relu is not None:
                fused_relu_nodes.add(fused_relu)
            continue

        if op_name == "batch_norm":
            if len(node.args) != 8:
                return None
            folded_batch_norm = next(
                (
                    batch_norm
                    for batch_norm, _, _ in folded_convs.values()
                    if batch_norm is node
                ),
                None,
            )
            if folded_batch_norm is not None:
                values[node] = values[node.args[0]]
                layout_values[node] = layout_values.get(node.args[0], False)
                continue
            input_node, running_mean, running_var, weight, bias, training, momentum, eps = node.args
            native_node = graph.create_node("batch_norm", node.name)
            if not add_tensor_input(native_node, input_node):
                return None
            optional_inputs = (
                ("has_running_mean", running_mean),
                ("has_running_var", running_var),
                ("has_weight", weight),
                ("has_bias", bias),
            )
            for attr_name, optional_node in optional_inputs:
                if optional_node is None:
                    native_node.set_int_attr(attr_name, 0)
                    continue
                if not add_tensor_input(native_node, optional_node):
                    return None
                native_node.set_int_attr(attr_name, 1)
            if not isinstance(training, bool) or not isinstance(momentum, numbers.Real) or not isinstance(eps, numbers.Real):
                return None
            native_node.set_int_attr("training", int(training))
            native_node.set_float_attr("momentum", float(momentum))
            native_node.set_float_attr("eps", float(eps))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"max_pool1d", "max_pool2d", "max_pool3d"}:
            if len(node.args) not in {6, 7}:
                return None
            input_node, kernel_size, stride, padding, dilation, ceil_mode = node.args[:6]
            return_indices = node.args[6] if len(node.args) == 7 else False
            if not isinstance(return_indices, bool) or not isinstance(ceil_mode, bool):
                return None
            spatial_rank = int(op_name[-2])
            kernel = _spatial_int_list(kernel_size, length=spatial_rank)
            stride_values = _spatial_int_list(stride, default=kernel, length=spatial_rank)
            padding_values = _spatial_int_list(
                padding, default=[0] * spatial_rank, length=spatial_rank
            )
            dilation_values = _spatial_int_list(
                dilation, default=[1] * spatial_rank, length=spatial_rank
            )
            if any(value is None for value in (kernel, stride_values, padding_values, dilation_values)):
                return None
            native_op = f"{op_name}_with_indices" if return_indices else op_name
            native_node = graph.create_node(native_op, node.name)
            if not add_tensor_input(native_node, input_node):
                return None
            native_node.set_ints_attr("kernel_size", kernel)
            native_node.set_ints_attr("stride", stride_values)
            native_node.set_ints_attr("padding", padding_values)
            native_node.set_ints_attr("dilation", dilation_values)
            native_node.set_int_attr("ceil_mode", int(ceil_mode))
            if return_indices:
                values[node] = (native_node.add_output(), native_node.add_output())
            else:
                values[node] = native_node.add_output()
            # The cuDNN tensor descriptor and output follow the input layout;
            # a later convolution can therefore consume the max-pool result
            # without an NCHW round-trip.
            layout_values[node] = use_channels_last and layout_values.get(
                input_node, False
            )
            continue

        if op_name in {"adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_avg_pool3d"}:
            if len(node.args) != 2:
                return None
            spatial_rank = int(op_name[-2])
            output_size = _spatial_int_list(node.args[1], length=spatial_rank)
            if output_size is None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.set_ints_attr("output_size", output_size)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"adaptive_max_pool1d", "adaptive_max_pool2d", "adaptive_max_pool3d"}:
            if len(node.args) != 2:
                return None
            spatial_rank = int(op_name[-2])
            output_size = _spatial_int_list(node.args[1], length=spatial_rank)
            if output_size is None:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.set_ints_attr("output_size", output_size)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "flatten":
            # The canonical signature is (input, start_dim=0, end_dim=-1);
            # capture stamps the end_dim default, so both the two- and
            # three-argument forms arrive here.  The native node flattens
            # (1, -1) only, which is what every accepted form must request.
            if (
                node.kwargs
                or len(node.args) not in (2, 3)
                or node.args[1] != 1
                or (len(node.args) == 3 and node.args[2] != -1)
            ):
                return None
            native_node = graph.create_node("flatten", node.name)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            native_node.set_int_attr("start_dim", 1)
            native_node.set_int_attr("end_dim", -1)
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"sum", "mean"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim", "keepdim"}:
                return None
            if node.op == "call_method":
                source = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "keepdim" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                keepdim = positional[1] if len(positional) > 1 else kwargs.get(
                    "keepdim", False
                )
            else:
                source = node.args[0]
                if len(node.args) > 3:
                    return None
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "keepdim" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                keepdim = (
                    node.args[2]
                    if len(node.args) > 2
                    else kwargs.get("keepdim", False)
                )
            if source not in values or not isinstance(keepdim, bool):
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            if dim is not None:
                dims = [int(dim)] if isinstance(dim, int) else _int_list(dim)
                if dims is None:
                    return None
                native_node.set_ints_attr("dim", dims)
                native_node.set_int_attr("keepdim", int(keepdim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "unsqueeze":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim"} or len(node.args) > 2:
                return None
            if len(node.args) > 1 and "dim" in kwargs:
                return None
            dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
            source = node.args[0]
            if source not in values or isinstance(dim, bool) or not isinstance(dim, int):
                return None
            native_node = graph.create_node("unsqueeze", node.name)
            native_node.add_input(values[source])
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "expand":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"size", "implicit"}:
                return None
            source = node.args[0]
            if len(node.args) > 2 or (len(node.args) > 1 and "size" in kwargs):
                return None
            shape = node.args[1] if len(node.args) > 1 else kwargs.get("size")
            if source not in values or _int_list(shape) is None:
                return None
            native_node = graph.create_node("expand", node.name)
            native_node.add_input(values[source])
            native_node.set_ints_attr("shape", _int_list(shape))
            if "implicit" in kwargs:
                if not isinstance(kwargs["implicit"], bool):
                    return None
                native_node.set_int_attr("implicit", int(kwargs["implicit"]))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "cat":
            if not node.args:
                return None
            kwargs = dict(node.kwargs or {})
            if set(kwargs) - {"dim", "out"}:
                return None
            tensors = node.args[0]
            if not isinstance(tensors, (tuple, list)) or not tensors:
                return None
            if any(not isinstance(item, Node) or item not in values for item in tensors):
                return None
            if len(node.args) > 2 or (len(node.args) > 1 and "dim" in kwargs):
                return None
            dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim", 0)
            if not isinstance(dim, int) or kwargs.get("out") is not None:
                return None
            native_node = graph.create_node("cat", node.name)
            for item in tensors:
                native_node.add_input(values[item])
            native_node.set_int_attr("dim", int(dim))
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name in {"chunk", "split", "split_with_sizes", "unbind"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            source = node.args[0]
            if source not in values:
                return None
            sample = node.meta.get("val")
            if not isinstance(sample, (tuple, list)) or not sample:
                return None
            native_node = graph.create_node(op_name, node.name)
            native_node.add_input(values[source])
            if op_name == "chunk":
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"chunks", "dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 1 and "chunks" in kwargs:
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                chunks = node.args[1] if len(node.args) > 1 else kwargs.get("chunks")
                dim = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
                if not isinstance(chunks, int) or not isinstance(dim, int):
                    return None
                chunks = int(chunks)
                dim = int(dim)
                if chunks <= 0:
                    return None
                native_node.set_int_attr("chunks", chunks)
                native_node.set_int_attr("dim", dim)
            elif op_name == "split":
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"split_size", "split_size_or_sizes", "dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 1 and ("split_size" in kwargs or "split_size_or_sizes" in kwargs):
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                split_size = node.args[1] if len(node.args) > 1 else kwargs.get(
                    "split_size", kwargs.get("split_size_or_sizes")
                )
                dim = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
                if not isinstance(dim, int):
                    return None
                if isinstance(split_size, int) and not isinstance(split_size, bool):
                    native_node.set_int_attr("split_size", int(split_size))
                elif _set_int_list_attr(native_node, "split_sizes", split_size):
                    pass
                else:
                    return None
                native_node.set_int_attr("dim", int(dim))
            elif op_name == "split_with_sizes":
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"split_sizes", "dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 1 and "split_sizes" in kwargs:
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                split_sizes = node.args[1] if len(node.args) > 1 else kwargs.get("split_sizes")
                if not _set_int_list_attr(native_node, "split_sizes", split_sizes):
                    return None
                dim = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
                if not isinstance(dim, int):
                    return None
                native_node.set_int_attr(
                    "dim", int(dim)
                )
            else:
                kwargs = dict(node.kwargs or {})
                if set(kwargs) - {"dim"} or len(node.args) > 2:
                    return None
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim", 0)
                if not isinstance(dim, int):
                    return None
                native_node.set_int_attr(
                    "dim", int(dim)
                )
            values[node] = tuple(native_node.add_output() for _ in sample)
            layout_values[node] = False
            continue

        if op_name == "add" and node in fused_add_relus:
            if len(node.args) == 2:
                input_node, other_node = node.args
                alpha = 1
            elif len(node.args) == 3:
                input_node, other_node, alpha = node.args
            else:
                return None
            if (
                alpha != 1
                or not isinstance(input_node, Node)
                or not isinstance(other_node, Node)
                or input_node not in values
                or other_node not in values
            ):
                return None
            fused = graph.create_node("add_relu", node.name)
            fused.add_input(values[input_node])
            fused.add_input(values[other_node])
            values[node] = fused.add_output()
            layout_values[node] = use_channels_last and (
                layout_values.get(input_node, False)
                or layout_values.get(other_node, False)
            )
            continue

        # ``alpha`` argument, so their graph node has
        # ``(input, other, alpha)`` even when alpha is the default 1, and
        # keyword-only spellings record alpha in the node kwargs.  Lower
        # that contract to the native pointwise IR instead of falling back
        # to a Python method call.  Non-unit alpha becomes a scalar multiply
        # and can be consumed by Stax's mul-add fusion pass.
        if op_name in {"add", "sub"} and (
            len(node.args) == 3
            or (len(node.args) == 2 and "alpha" in (node.kwargs or {}))
        ):
            if set(node.kwargs or {}) - {"alpha"}:
                return None
            if len(node.args) == 3 and not node.kwargs:
                input_node, other_node, alpha = node.args
            else:
                input_node, other_node = node.args
                alpha = node.kwargs.get("alpha", 1)
            if not isinstance(input_node, Node) or not _is_scalar(alpha):
                return None
            if input_node not in values:
                return None
            if isinstance(other_node, Node) and other_node not in values:
                return None

            if alpha == 1:
                binary = graph.create_node(op_name, node.name)
                binary.add_input(values[input_node])
                if isinstance(other_node, Node):
                    binary.add_input(values[other_node])
                elif _is_scalar(other_node):
                    _set_scalar_attr(binary, other_node, 1)
                else:
                    return None
                values[node] = binary.add_output()
                layout_values[node] = False
                continue

            if isinstance(other_node, Node):
                scale = graph.create_node("mul", f"{node.name}_alpha")
                scale.add_input(values[other_node])
                _set_scalar_attr(scale, alpha, 1)
                scaled_other = scale.add_output()

                binary = graph.create_node(op_name, node.name)
                binary.add_input(values[input_node])
                binary.add_input(scaled_other)
                values[node] = binary.add_output()
                layout_values[node] = False
                continue

            if _is_scalar(other_node):
                binary = graph.create_node(op_name, node.name)
                binary.add_input(values[input_node])
                _set_scalar_attr(binary, other_node * alpha, 1)
                values[node] = binary.add_output()
                layout_values[node] = False
                continue
            return None

        if op_name == "linear":
            if len(node.args) not in {2, 3} or any(
                not isinstance(arg, Node) and arg is not None for arg in node.args
            ):
                return None
            input_node, weight_node = node.args[:2]
            bias_node = node.args[2] if len(node.args) == 3 else None
            if not isinstance(input_node, Node) or not isinstance(weight_node, Node):
                return None
            if bias_node is not None and not isinstance(bias_node, Node):
                return None
            if any(
                value_node not in values
                for value_node in (input_node, weight_node, bias_node)
                if value_node is not None
            ):
                return None

            # One node, not a transpose plus a product plus an addition: the
            # bias belongs in the product's epilogue, and adding it back
            # separately costs a whole pass over the output.
            if _native_runs_linear():
                linear_input = values[input_node]
                linear_weight = values[weight_node]
                if autocast_dtype is not None and getattr(node.meta.get("val"), "dtype", None) == autocast_dtype:
                    linear_input = saved_autocast_value(input_node, linear_input)
                    linear_weight = saved_autocast_value(weight_node, linear_weight)
                fused = graph.create_node("linear", node.name)
                fused.add_input(linear_input)
                fused.add_input(linear_weight)
                if bias_node is not None:
                    fused.add_input(values[bias_node])
                values[node] = fused.add_output()
                layout_values[node] = False
                continue

            transpose = graph.create_node("t", f"{node.name}_weight_t")
            transpose.add_input(values[weight_node])
            transposed_weight = transpose.add_output()

            matmul = graph.create_node("matmul", f"{node.name}_matmul")
            matmul.add_input(values[input_node])
            matmul.add_input(transposed_weight)
            result = matmul.add_output()
            if bias_node is not None:
                add = graph.create_node("add", f"{node.name}_bias")
                add.add_input(result)
                add.add_input(values[bias_node])
                result = add.add_output()
            values[node] = result
            layout_values[node] = False
            continue

        input_nodes: list[Node] = []
        scalar_args: list[tuple[int, Any]] = []
        for position, arg in enumerate(node.args):
            if isinstance(arg, Node):
                input_nodes.append(arg)
            elif _is_scalar(arg):
                scalar_args.append((position, arg))
            else:
                return None
        if len(scalar_args) > 1:
            return None
        if op_name in {
            "neg",
            "pos",
            "abs",
            "sin",
            "cos",
            "exp",
            "log",
            "sigmoid",
            "sqrt",
            "square",
            "tanh",
            "relu",
            "conj",
        }:
            if len(node.args) != 1 or len(input_nodes) != 1:
                return None
        elif len(input_nodes) not in {1, 2} or len(node.args) not in {1, 2}:
            return None
        if any(input_node not in values for input_node in input_nodes):
            return None
        if op_name == "mm":
            if len(node.args) != 2 or len(input_nodes) != 2:
                return None
            native_node = graph.create_node("mm", node.name)
            native_node.add_input(values[input_nodes[0]])
            native_node.add_input(values[input_nodes[1]])
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue
        native_node = graph.create_node(op_name, node.name)
        for input_node in input_nodes:
            native_node.add_input(values[input_node])
        if scalar_args:
            _set_scalar_attr(native_node, scalar_args[0][1], scalar_args[0][0])
        if op_name == "relu":
            # Preserve the functional schema.  The executor may call relu_
            # only when the captured call explicitly requested mutation.
            native_node.set_int_attr(
                "inplace", int(node.kwargs.get("inplace", False))
            )
        values[node] = native_node.add_output()
        layout_values[node] = False

    try:
        fuser.finish(required_nodes)
    except (KeyError, RuntimeError, TypeError, ValueError):
        return None

    try:
        output_arg = graph_module.graph.output_node.args[0]
        output_values = _native_value_leaves(output_arg, values)
        output_spec = _native_output_spec(output_arg, values)
    except (IndexError, KeyError, RuntimeError, TypeError):
        return None
    if not output_values or any(value is None for value in output_values):
        return None

    for output_value in output_values:
        graph.register_output(output_value)
    public_output_count = len(output_values)
    registered_extra_outputs = 0
    for extra_node in extra_output_nodes or []:
        if extra_node not in values:
            return None
        extra_values = values[extra_node]
        if isinstance(extra_values, tuple):
            if not extra_values or any(value is None for value in extra_values):
                return None
            for extra_value in extra_values:
                graph.register_output(extra_value)
            registered_extra_outputs += len(extra_values)
        else:
            if extra_values is None:
                return None
            graph.register_output(extra_values)
            registered_extra_outputs += 1

    for converted in autocast_values.values():
        graph.register_output(converted)
        registered_extra_outputs += 1

    # Buffer updates run last: each registered mutation output pairs with an
    # input position, and the wrapper copies it back after execution.
    mutation_outputs: list[tuple[int, int]] = []
    for position, value in mutations:
        graph.register_output(value)
        mutation_outputs.append(
            (position, public_output_count + registered_extra_outputs + len(mutation_outputs))
        )

    if use_fusion and not mutation_outputs:
        graph.fuse()
    return _NativeLowering(
        graph_module,
        graph,
        attribute_targets,
        constant_values,
        output_count=public_output_count + registered_extra_outputs,
        native_values=values,
        output_spec=output_spec,
        public_output_count=public_output_count,
        mutations=mutation_outputs,
        autocast_outputs=list(autocast_values),
    )


class _AotShape(tuple):
    """Tensor metadata that supports both ``shape`` and ``shape()`` schemas."""

    def __new__(cls, value: Any):
        return super().__new__(cls, (int(item) for item in value))

    def __call__(self) -> tuple[int, ...]:
        return tuple(self)


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

    def reshape(self, shape: Any) -> "_AotNativeSymbol":
        return self.builder.reshape(self, shape)

    def view(self, shape: Any) -> "_AotNativeSymbol":
        return self.builder.reshape(self, shape)

    def expand(self, shape: Any) -> "_AotNativeSymbol":
        return self.builder.expand(self, shape)

    def unsqueeze(self, dim: int) -> "_AotNativeSymbol":
        return self.builder.unsqueeze(self, dim)

    def squeeze(self, dim: Any = None) -> "_AotNativeSymbol":
        return self.builder.squeeze(self, dim)

    def sum(self, dim: Any = None, keepdim: bool = False) -> "_AotNativeSymbol":
        return self.builder.sum(self, dim, keepdim)

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


class _AotNativeGraphBuilder:
    """Small native-IR builder used by the source-derived reverse pass.

    Elementwise work is not emitted one node at a time: binary/unary calls
    accumulate into a program buffer that is emitted as a single fused
    pointwise node whenever a non-elementwise op, an output, or an operand
    outside the buffer needs materializing.  One fused node becomes one
    kernel launch, so derivative-formula chains stop paying per-op
    dispatch and memory round trips.
    """

    def __init__(self, native_module: Any):
        self.native_module = native_module
        self.graph = native_module.Graph()
        self._literal_symbols: dict[int, _AotNativeSymbol] = {}
        # Elementwise program buffer (empty when idle).
        self._fused_ops: list[tuple[int, Any, Any]] = []
        self._fused_temp_pos: dict[int, int] = {}
        self._fused_temp_symbols: dict[int, _AotNativeSymbol] = {}
        self._fused_shape: tuple[int, ...] | None = None
        self._fused_dtype: Any = None
        self._fused_pending: dict[int, tuple[_AotFusedSpec, int]] = {}
        # Element type overrides for buffered results: a conversion edge
        # consumed only by a kernel that wants another width rides the
        # producer's store instead of a standalone conversion node.
        self._fused_out_dtypes: dict[int, Any] = {}
        self._cuda_examples: dict[Any, Any] = {}
        self._cast_memo: dict[tuple[int, Any], tuple[_AotNativeSymbol, _AotNativeSymbol]] = {}

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in getattr(value, "shape", ()))

    @staticmethod
    def _symbol(value: Any) -> _AotNativeSymbol | None:
        return value if isinstance(value, _AotNativeSymbol) else None

    def input(self, example_value: Any) -> _AotNativeSymbol:
        symbol = _AotNativeSymbol(self, self.graph.add_input(), self._shape(example_value))
        dtype = getattr(example_value, "dtype", None)
        if dtype is not None:
            symbol.dtype = dtype
        try:
            if example_value.device.is_cuda():
                self._cuda_examples.setdefault(dtype, example_value)
        except (AttributeError, RuntimeError, TypeError):
            pass
        return symbol

    def literal(self, value: Any) -> _AotNativeSymbol:
        """Lift a captured tensor constant into the backward graph inputs."""
        key = id(value)
        symbol = self._literal_symbols.get(key)
        if symbol is None:
            symbol = self.input(value)
            self._literal_symbols[key] = symbol
        return symbol

    def _add_inputs(self, native_node: Any, args: tuple[Any, ...]) -> list[_AotNativeSymbol]:
        symbols: list[_AotNativeSymbol] = []
        for value in args:
            if not isinstance(value, _AotNativeSymbol):
                raise TypeError("AOT native op received a non-Tensor argument")
            self._materialize(value)
            native_node.add_input(value.value)
            symbols.append(value)
        return symbols

    # -- elementwise program buffer -----------------------------------------

    def _materialize(self, symbol: _AotNativeSymbol) -> None:
        """Give a buffered or spilled elementwise result a real graph value.

        A temporary still in the buffer flushes the program with only that
        result emitted; sibling intermediates spill and stay dormant until a
        consumer pulls them.  A spilled temporary is re-emitted as its own
        node, so a shared subexpression consumed by a later formula pays for
        that single result instead of forcing every partial sum of the
        program through memory.
        """
        if symbol is None or symbol.value is not None:
            return
        index = self._fused_temp_pos.get(id(symbol))
        if index is not None:
            self._fused_flush([index])
            return
        entry = self._fused_pending.get(id(symbol))
        if entry is not None:
            spec, index = entry
            del self._fused_pending[id(symbol)]
            self._emit_program(spec.ops, spec.temps, [index], spec.out_dtypes)
            return
        raise RuntimeError("AOT elementwise buffer lost a temporary symbol")

    def _promote_fused_dtype(self, dtype: Any) -> None:
        """Widen the program's result element type to a wider operand.

        Buffered results already carry the previous width in their symbols;
        they hold no data yet, so retargeting the symbols keeps every later
        consumer reading the true stored type.  Results an explicit store
        override has claimed keep their claimed width.
        """
        previous = self._fused_dtype
        self._fused_dtype = dtype
        for symbol in self._fused_temp_symbols.values():
            if symbol.dtype == previous:
                symbol.dtype = dtype

    def _fused_append(
        self, opcode: int, lhs: tuple[str, Any], rhs: tuple[str, Any]
    ) -> _AotNativeSymbol:
        index = len(self._fused_ops)
        self._fused_ops.append((opcode, lhs, rhs))
        symbol = _AotNativeSymbol(self, None, self._fused_shape, self._fused_dtype)
        self._fused_temp_pos[id(symbol)] = index
        self._fused_temp_symbols[id(symbol)] = symbol
        return symbol

    def _fused_try_elementwise(
        self,
        op_name: str,
        lhs_symbol: _AotNativeSymbol | None,
        rhs_symbol: _AotNativeSymbol | None,
        lhs_raw: Any,
        rhs_raw: Any,
        *,
        unary: bool = False,
    ) -> _AotNativeSymbol | None:
        if op_name == "conj":
            dtype = getattr(lhs_symbol, "dtype", None)
            if dtype is None or "complex" in str(dtype):
                return None
            op_name = "pos"
        opcode = _FUSED_PROGRAM_OPCODES.get(op_name)
        if opcode is None:
            return None
        if unary:
            operands = (("sym", lhs_symbol), ("const", 0.0))
            shape = tuple(lhs_symbol.shape)
        elif lhs_symbol is not None and rhs_symbol is not None:
            try:
                shape = self._broadcast_shape(lhs_symbol.shape, rhs_symbol.shape)
            except ValueError:
                return None
            operands = (("sym", lhs_symbol), ("sym", rhs_symbol))
        elif lhs_symbol is not None:
            operands = (("sym", lhs_symbol), ("const", float(rhs_raw)))
            shape = tuple(lhs_symbol.shape)
        elif rhs_symbol is not None:
            operands = (("const", float(lhs_raw)), ("sym", rhs_symbol))
            shape = tuple(rhs_symbol.shape)
        else:
            return None
        for kind, value in operands:
            if kind == "sym" and value.value is None:
                if id(value) in self._fused_temp_pos:
                    continue
                if id(value) not in self._fused_pending:
                    return None  # operand of a dropped program: fall back
                # A spilled temp can join this buffer once it is revived as
                # a real input value.
                self._materialize(value)
        # The fused evaluator widens every input to one arithmetic width on
        # load, so operands of different storage sizes share a program as
        # long as they stay in the same precision family; float64 runs its
        # own family because the arithmetic width follows the widest operand.
        # A wider operand promotes the element type of every program result.
        for kind, value in operands:
            if kind == "sym" and value.dtype is not None:
                if self._fused_dtype is None:
                    self._fused_dtype = value.dtype
                else:
                    names = {str(value.dtype), str(self._fused_dtype)}
                    if sum("float64" in name for name in names) == 1:
                        return None
                    if _dtype_width(value.dtype) > _dtype_width(
                        self._fused_dtype
                    ):
                        self._promote_fused_dtype(value.dtype)
        if self._fused_shape is None:
            self._fused_shape = shape
        elif shape != self._fused_shape:
            # Generation change: nothing here is emitted yet, so spill the
            # whole generation and let consumers pull results on demand.
            self._fused_flush([])
            self._fused_shape = shape
        return self._fused_append(opcode, operands[0], operands[1])

    def _fused_flush(self, wanted: list[int] | None = None) -> None:
        """Retire the buffer generation; emit only the requested results.

        Results nobody asked for spill into the pending registry and stay
        dormant until a consumer pulls them; results nobody ever pulls cost
        nothing at all.
        """
        if not self._fused_ops:
            return
        ops = self._fused_ops
        temp_pos = self._fused_temp_pos
        temp_symbols = self._fused_temp_symbols
        promoted = self._fused_dtype
        self._fused_ops = []
        self._fused_temp_pos = {}
        self._fused_temp_symbols = {}
        self._fused_shape = None
        self._fused_dtype = None

        temps = tuple(
            temp_symbols[temp_id]
            for temp_id, _ in sorted(temp_pos.items(), key=lambda kv: kv[1])
        )
        # Every result names its stored element type: unclaimed temps take
        # the program's promoted width, so a program mixing input widths
        # never depends on the first input's storage type as a default.
        out_dtypes = {
            temp_pos[temp_id]: self._fused_out_dtypes.get(temp_id, promoted)
            for temp_id in temp_pos
        }
        self._fused_out_dtypes = {}
        spec = _AotFusedSpec(tuple(ops), temps, out_dtypes)
        emitted = list(range(len(ops))) if wanted is None else list(wanted)
        for index, symbol in enumerate(temps):
            if index not in emitted:
                self._fused_pending[id(symbol)] = (spec, index)
        if emitted:
            self._emit_program(spec.ops, temps, emitted, spec.out_dtypes)

    def _emit_program(
        self,
        ops: tuple[tuple[int, Any, Any], ...],
        temps: tuple[_AotNativeSymbol, ...],
        wanted: list[int],
        out_dtypes: dict[int, Any] | None = None,
    ) -> None:
        temp_pos = {id(symbol): index for index, symbol in enumerate(temps)}
        # Dependency closure of the requested results, in program order; the
        # closure is the whole program only when every temp is wanted.  A
        # single-result emit moves that result last because the one-output
        # evaluator returns the final temp of the program.
        closure: set[int] = set()
        stack = list(wanted)
        while stack:
            index = stack.pop()
            if index in closure:
                continue
            closure.add(index)
            for kind, value in ops[index][1:]:
                if kind == "sym":
                    position = temp_pos.get(id(value))
                    if position is not None:
                        stack.append(position)
        sequence = sorted(closure)
        if len(wanted) == 1:
            sequence.remove(wanted[0])
            sequence.append(wanted[0])
        mapping = {index: position for position, index in enumerate(sequence)}

        input_symbols: list[_AotNativeSymbol] = []
        input_pos: dict[int, int] = {}
        constants: list[float] = []
        const_pos: dict[float, int] = {}
        for index in sequence:
            for kind, value in ops[index][1:]:
                if kind == "sym":
                    if id(value) in temp_pos:
                        continue  # program-internal temporary
                    if id(value) in input_pos:
                        continue
                    input_pos[id(value)] = len(input_symbols)
                    input_symbols.append(value)
                    continue
                number = float(value)
                if number not in const_pos:
                    const_pos[number] = len(constants)
                    constants.append(number)
        input_count = len(input_symbols)
        program: list[int] = []
        for index in sequence:
            opcode, lhs, rhs = ops[index]
            program.append(opcode)
            for kind, value in (lhs, rhs):
                if kind == "const":
                    program.append(-const_pos[value] - 1)
                elif id(value) in temp_pos:
                    program.append(input_count + mapping[temp_pos[id(value)]])
                else:
                    program.append(input_pos[id(value)])

        emitted = list(wanted)
        output_refs = [input_count + mapping[index] for index in emitted]
        op_name = None
        if not out_dtypes and 1 <= input_count <= 32 and len(emitted) <= 32:
            first_shape = tuple(input_symbols[0].shape)
            first_dtype = input_symbols[0].dtype
            uniform = all(
                tuple(symbol.shape) == first_shape
                and symbol.dtype == first_dtype
                for symbol in input_symbols
            )
            if uniform:
                example = self._cuda_examples.get(first_dtype)
                if (
                    example is not None
                    and tuple(int(item) for item in example.shape) == first_shape
                    and example.dtype == first_dtype
                ):
                    op_name = _register_stax_cuda_pointwise_op(
                        program,
                        constants,
                        tuple(output_refs),
                        input_count,
                        [example] * input_count,
                        (repr(first_dtype),) * len(output_refs),
                    )
                else:
                    op_name = _register_stax_cuda_pointwise_op(
                        program,
                        constants,
                        tuple(output_refs),
                        input_count,
                        [],
                    )
        node = self.graph.create_node(
            "custom_op" if op_name is not None else "fused_pointwise",
            f"aot_fused_pointwise_{len(self.graph.nodes)}",
        )
        for symbol in input_symbols:
            node.add_input(symbol.value)
        if op_name is None:
            node.set_int_attr("input_count", input_count)
            node.set_ints_attr("program", program)
            node.set_floats_attr("constants", constants)
            node.set_ints_attr("output_refs", output_refs)
        else:
            node.set_str_attr("op_name", op_name)

        outputs = [node.add_output() for _ in emitted]
        if out_dtypes:
            codes = []
            for index in emitted:
                dtype = out_dtypes.get(index)
                name = (
                    str(dtype).rsplit(".", 1)[-1] if dtype is not None else None
                )
                codes.append(_FUSED_OUT_DTYPE_CODES.get(name, -1))
            if any(code >= 0 for code in codes):
                node.set_ints_attr("output_dtypes", codes)
        node_to_output = dict(zip(emitted, outputs))
        for position, output in node_to_output.items():
            temps[position].value = output

    @staticmethod
    def _broadcast_shape(lhs: tuple[int, ...], rhs: tuple[int, ...]) -> tuple[int, ...]:
        result: list[int] = []
        for left, right in zip(reversed(lhs), reversed(rhs)):
            if left != right and left != 1 and right != 1:
                raise ValueError(f"incompatible AOT shapes: {lhs} and {rhs}")
            result.append(max(left, right))
        longer = lhs if len(lhs) >= len(rhs) else rhs
        result.extend(reversed(longer[: abs(len(lhs) - len(rhs))]))
        return tuple(reversed(result))

    def binary(self, op_name: str, lhs: Any, rhs: Any) -> _AotNativeSymbol:
        lhs_symbol = self._symbol(lhs)
        rhs_symbol = self._symbol(rhs)
        if lhs_symbol is None and hasattr(lhs, "shape"):
            lhs_symbol = self.literal(lhs)
        if rhs_symbol is None and hasattr(rhs, "shape"):
            rhs_symbol = self.literal(rhs)
        if lhs_symbol is None and rhs_symbol is None:
            if op_name == "add":
                return lhs + rhs
            if op_name == "sub":
                return lhs - rhs
            if op_name == "mul":
                return lhs * rhs
            if op_name == "div":
                return lhs / rhs
            raise NotImplementedError(f"AOT scalar operation is unsupported: {op_name}")
        fused = self._fused_try_elementwise(
            op_name, lhs_symbol, rhs_symbol, lhs, rhs
        )
        if fused is not None:
            return fused
        # Flush pending elementwise programs before the consumer node exists:
        # a program flushed afterwards would execute after this node reads it.
        if lhs_symbol is not None:
            self._materialize(lhs_symbol)
        if rhs_symbol is not None:
            self._materialize(rhs_symbol)
        self._fused_flush([])
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        shape = lhs_symbol.shape if lhs_symbol is not None else rhs_symbol.shape
        if lhs_symbol is not None and rhs_symbol is not None:
            native_node.add_input(lhs_symbol.value)
            native_node.add_input(rhs_symbol.value)
            shape = self._broadcast_shape(lhs_symbol.shape, rhs_symbol.shape)
        else:
            symbol = lhs_symbol if lhs_symbol is not None else rhs_symbol
            scalar = rhs if lhs_symbol is not None else lhs
            native_node.add_input(symbol.value)
            _set_scalar_attr(native_node, scalar, 1 if lhs_symbol is not None else 0)
        result = _AotNativeSymbol(self, native_node.add_output(), shape)
        result.dtype = (lhs_symbol if lhs_symbol is not None else rhs_symbol).dtype
        return result

    def unary(
        self,
        op_name: str,
        value: _AotNativeSymbol,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> _AotNativeSymbol:
        fused = self._fused_try_elementwise(
            op_name, value, None, None, None, unary=True
        )
        if fused is not None:
            return fused
        self._materialize(value)
        self._fused_flush([])
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        native_node.add_input(value.value)
        result = _AotNativeSymbol(self, native_node.add_output(), shape or value.shape)
        result.dtype = value.dtype
        return result

    def helper(
        self,
        op_name: str,
        args: tuple[_AotNativeSymbol, ...],
        *,
        attrs: dict[str, Any] | None = None,
        shape: tuple[int, ...] | None = None,
        outputs: int = 1,
        output_shapes: tuple[tuple[int, ...], ...] | None = None,
    ) -> _AotNativeSymbol | _AotNativeTuple:
        # Pending elementwise programs must become nodes before the consumer:
        # the native executor walks creation order, so a program flushed after
        # this node would still be undefined when this node reads it.
        for value in args:
            self._materialize(value)
        self._fused_flush([])
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        symbols = self._add_inputs(native_node, args)
        del symbols
        for key, value in (attrs or {}).items():
            if isinstance(value, bool) or isinstance(value, int):
                native_node.set_int_attr(key, int(value))
            elif isinstance(value, numbers.Real):
                native_node.set_float_attr(key, float(value))
            elif isinstance(value, str):
                native_node.set_str_attr(key, value)
            elif isinstance(value, (tuple, list)) and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                native_node.set_ints_attr(key, [int(item) for item in value])
            else:
                raise TypeError(f"unsupported AOT native attribute: {key}={value!r}")
        if outputs == 1:
            result = _AotNativeSymbol(
                self, native_node.add_output(), shape or args[0].shape
            )
            result.dtype = args[0].dtype if args else None
            return result
        result = _AotNativeTuple(
            tuple(
                _AotNativeSymbol(
                    self,
                    native_node.add_output(),
                    output_shapes[index]
                    if output_shapes is not None and index < len(output_shapes)
                    else shape or args[0].shape,
                    args[0].dtype if args else None,
                )
                for index in range(outputs)
            )
        )
        result.node = native_node
        return result

    def reshape(self, value: _AotNativeSymbol, shape: Any) -> _AotNativeSymbol:
        normalized = tuple(int(item) for item in shape)
        return self.helper("reshape", (value,), attrs={"shape": normalized}, shape=normalized)  # type: ignore[return-value]

    def cast(self, value: _AotNativeSymbol, dtype: Any) -> _AotNativeSymbol:
        """Convert a symbol to ``dtype`` (no-op when already in that type).

        Conversions are memoized per (symbol, dtype): one captured value
        feeding several consuming kernels (e.g. a convolution gradient split
        into input/weight/bias contributions) is converted once and the
        converted value is shared.
        """
        if dtype is None or value.dtype == dtype:
            return value
        memo_key = (id(value), dtype)
        memo = self._cast_memo.get(memo_key)
        if memo is not None:
            return memo[1]
        index = self._fused_temp_pos.get(id(value))
        pending = None if index is not None else self._fused_pending.get(id(value))
        if index is not None or pending is not None:
            # The buffered or spilled result has not been emitted yet: its
            # storage type can still be retargeted, so the producer stores
            # the converted value directly and no conversion node exists.
            if index is not None:
                current = self._fused_out_dtypes.get(id(value), value.dtype)
            else:
                spec, index = pending
                current = spec.out_dtypes.get(index)
            if current is None or current == dtype:
                if index is not None:
                    self._fused_out_dtypes[id(value)] = dtype
                else:
                    pending[0].out_dtypes[index] = dtype
                value.dtype = dtype
                self._cast_memo[memo_key] = (value, value)
                return value
            # A different width was already claimed for this result: fall
            # through to a standalone conversion after materializing.
            self._materialize(value)
        name = str(dtype).rsplit(".", 1)[-1]
        result = self.helper("cast", (value,), attrs={"dtype": name}, shape=value.shape)
        if isinstance(result, _AotNativeSymbol):
            result.dtype = dtype
            # The entry keeps the source symbol alive: the key is its id(),
            # and ids are only stable while the object is referenced.
            self._cast_memo[memo_key] = (value, result)
            return result
        return result

    def zeros_like(
        self, template: _AotNativeSymbol, shape: Any
    ) -> _AotNativeSymbol:
        normalized = tuple(int(item) for item in shape)
        return self.helper(
            "zeros_like", (template,), attrs={"shape": normalized}, shape=normalized
        )  # type: ignore[return-value]

    def expand(self, value: _AotNativeSymbol, shape: Any) -> _AotNativeSymbol:
        normalized = tuple(int(item) for item in shape)
        return self.helper("expand", (value,), attrs={"shape": normalized}, shape=normalized)  # type: ignore[return-value]

    def unsqueeze(self, value: _AotNativeSymbol, dim: int) -> _AotNativeSymbol:
        rank = len(value.shape)
        normalized_dim = int(dim)
        if normalized_dim < 0:
            normalized_dim += rank + 1
        if normalized_dim < 0 or normalized_dim > rank:
            raise ValueError(
                f"AOT unsqueeze dimension {dim} is out of range for rank {rank}"
            )
        shape = value.shape[:normalized_dim] + (1,) + value.shape[normalized_dim:]
        return self.helper(
            "unsqueeze", (value,), attrs={"dim": normalized_dim}, shape=shape
        )  # type: ignore[return-value]

    def squeeze(
        self, value: _AotNativeSymbol, dim: Any = None
    ) -> _AotNativeSymbol:
        if dim is None:
            shape = tuple(item for item in value.shape if item != 1)
            attrs: dict[str, Any] = {}
        else:
            normalized_dim = int(dim)
            if normalized_dim < 0:
                normalized_dim += len(value.shape)
            if (
                normalized_dim < 0
                or normalized_dim >= len(value.shape)
                or value.shape[normalized_dim] != 1
            ):
                raise ValueError("AOT squeeze dimension is not singleton")
            shape = value.shape[:normalized_dim] + value.shape[normalized_dim + 1 :]
            attrs = {"dim": normalized_dim}
        return self.helper("squeeze", (value,), attrs=attrs, shape=shape)  # type: ignore[return-value]

    def cat(
        self, values: tuple[_AotNativeSymbol, ...], dim: int, shape: Any
    ) -> _AotNativeSymbol:
        if not values:
            raise ValueError("AOT cat requires at least one value")
        return self.helper(
            "cat", values, attrs={"dim": int(dim)}, shape=tuple(int(item) for item in shape)
        )  # type: ignore[return-value]

    def split(
        self,
        value: _AotNativeSymbol,
        sizes: tuple[int, ...],
        dim: int,
        shapes: tuple[tuple[int, ...], ...],
    ) -> _AotNativeTuple:
        result = self.helper(
            "split",
            (value,),
            attrs={"split_sizes": sizes, "dim": int(dim)},
            shape=shapes[0] if shapes else value.shape,
            outputs=len(shapes),
        )
        if not isinstance(result, _AotNativeTuple):
            raise TypeError("AOT split did not produce multiple outputs")
        for symbol, shape in zip(result.values, shapes):
            symbol.shape = _AotShape(shape)
        return result

    def sum(
        self, value: _AotNativeSymbol, dim: Any = None, keepdim: bool = False
    ) -> _AotNativeSymbol:
        if dim is None:
            return self.helper("sum", (value,), shape=())  # type: ignore[return-value]
        dims = tuple(int(item) for item in (dim if isinstance(dim, (tuple, list)) else (dim,)))
        normalized_dims = tuple(item if item >= 0 else item + len(value.shape) for item in dims)
        shape = list(value.shape)
        if keepdim:
            for item in normalized_dims:
                shape[item] = 1
        else:
            for item in sorted(normalized_dims, reverse=True):
                shape.pop(item)
        return self.helper(
            "sum",
            (value,),
            attrs={"dim": normalized_dims, "keepdim": bool(keepdim)},
            shape=tuple(shape),
        )  # type: ignore[return-value]


def _aot_derivative_specs() -> dict[str, tuple[Any, dict[str, str]]]:
    """Read the local derivative schema used by TensorPlay code generation."""
    from pathlib import Path

    from tools.codegen.model import parse_derivatives_yaml, parse_schema

    yaml_path = Path(__file__).resolve().parents[4] / "config" / "derivatives.yaml"
    result: dict[str, tuple[Any, dict[str, str]]] = {}
    for definition in parse_derivatives_yaml(str(yaml_path)):
        parsed = parse_schema(definition["name"])
        formulas = {
            key: value for key, value in definition.items() if key != "name"
        }
        result[parsed.func_name] = (parsed, formulas)
    return result


def _aot_formula_python(formula: str, tensor_params: set[str]) -> str:
    """Compile one derivatives.yaml formula into a Python expression.

    Shares the codegen expression AST (tokenizer + parser); the emitter
    renders against the runtime formula env -- builder callables like
    add/mul/t/sum plus get_tuple -- instead of the C++ text the generated
    autograd nodes need.
    """
    from tools.codegen.gen_autograd import (
        BinOp, BoolLit, Braced, Call, Method, Neg, Num, StrLit, Var,
        TENSOR_METHODS, parse_expr,
    )

    symbols = set(tensor_params) | {"grad", "grad_output", "result"}

    def is_tensor(expr: Any) -> bool:
        if isinstance(expr, Var):
            return expr.name in symbols
        if isinstance(expr, Neg):
            return looks_tensor(expr.value)
        if isinstance(expr, Method):
            return expr.name.rstrip("_") in TENSOR_METHODS
        if isinstance(expr, Call):
            leaf = expr.callee.split("::")[-1].split("<")[0]
            return leaf not in ("Scalar",)
        if isinstance(expr, BinOp):
            return is_tensor(expr.left) or looks_tensor(expr.right)
        return False

    def looks_tensor(expr: Any) -> bool:
        return is_tensor(expr) or isinstance(expr, BinOp)

    def emit(expr: Any) -> str:
        if isinstance(expr, Num):
            return expr.text
        if isinstance(expr, BoolLit):
            return "True" if expr.text == "true" else "False"
        if isinstance(expr, StrLit):
            return expr.text
        if isinstance(expr, Var):
            if expr.name in {"true", "false"}:
                return "True" if expr.name == "true" else "False"
            if expr.name in {"std::nullopt", "c10::nullopt"}:
                return "None"
            return expr.name
        if isinstance(expr, Neg):
            inner = emit(expr.value)
            return f"neg({inner})" if looks_tensor(expr.value) else f"-{inner}"
        if isinstance(expr, Braced):
            # Python target: a braced list renders as a tuple (builder.sum
            # dims, reshape shapes), matching _aot_default_value.
            items = [emit(item) for item in expr.items]
            if len(items) == 1:
                return f"({items[0]},)"
            return f"({', '.join(items)})"
        if isinstance(expr, Call):
            args = ", ".join(emit(a) for a in expr.args)
            get = re.fullmatch(r"std::get<(\d+)>", expr.callee)
            if get:
                return f"get_tuple({get.group(1)}, {args})"
            if expr.callee in {"std::nullopt", "c10::nullopt"}:
                return "None"
            callee = expr.callee.split("::")[-1]
            return f"{callee}({args})"
        if isinstance(expr, Method):
            recv = emit(expr.receiver)
            args = ", ".join(emit(a) for a in expr.args)
            name = expr.name
            if name in {"dtype", "scalar_type"}:
                return "None"
            base = name[:-1] if name.endswith("_") and name[:-1] in TENSOR_METHODS else name
            if base in TENSOR_METHODS:
                return f"{TENSOR_METHODS[base]}({recv}, {args})" if args \
                    else f"{TENSOR_METHODS[base]}({recv})"
            return f"{recv}.{name}({args})" if args else f"{recv}.{name}()"
        if isinstance(expr, BinOp):
            left = emit(expr.left)
            right = emit(expr.right)
            left_tensor = is_tensor(expr.left)
            right_tensor = looks_tensor(expr.right)
            if expr.op in "+-" and left_tensor:
                return f"{'add' if expr.op == '+' else 'sub'}({left}, {right})"
            if expr.op == "*" and left_tensor:
                return f"mul({left}, {right})"
            if expr.op == "/" and left_tensor:
                return f"div({left}, {right})"
            if expr.op == "*" and right_tensor:
                return f"mul({right}, {left})"
            if expr.op == "-" and right_tensor:
                return f"neg(sub({right}, {left}))"
            return f"({left} {expr.op} {right})"
        raise NotImplementedError(f"AOT formula node is unsupported: {expr!r}")

    return emit(parse_expr(formula))


def _build_aot_formula_env(
    builder: _AotNativeGraphBuilder,
    *,
    batch_norm_cache: dict[tuple[int, ...], _AotNativeTuple],
    tuple_op_cache: dict[tuple[int, ...], _AotNativeTuple],
    autocast_state: dict[str, Any],
) -> dict[str, Any]:
    import tensorplay

    def binary(name: str):
        return lambda lhs, rhs: builder.binary(name, lhs, rhs)

    # Mixed-precision execution: the captured graph runs its matrix products
    # in the reduced precision selected at dispatch time, while the derivative
    # expressions receive gradients in the accumulation type.  Every matrix
    # product of the reverse pass therefore narrows its operands back to the
    # reduced element type before the kernel runs.
    def _narrow(value: Any) -> Any:
        dtype = autocast_state.get("dtype")
        if not isinstance(value, _AotNativeSymbol):
            return value
        return builder.cast(value, dtype)

    def matmul_binary(name: str):
        def invoke(lhs: Any, rhs: Any) -> _AotNativeSymbol:
            return builder.binary(name, _narrow(lhs), _narrow(rhs))

        return invoke

    def unary(name: str):
        return lambda value: builder.unary(name, value)

    def _sum_dim_backward_env(grad, self_value, dim, keepdim):
        dims = dim if isinstance(dim, (list, tuple)) else [dim]
        normalized = sorted(int(item) for item in dims)
        if not keepdim:
            for item in normalized:
                grad = builder.unsqueeze(grad, item)
        return builder.expand(
            grad, tuple(int(item) for item in self_value.shape)
        )

    def tanh_backward_env(grad, output_value):
        # grad * (1 - output^2).conj(): identical to the decomposition the
        # dispatcher uses, restated over builder primitives.
        inner = builder.binary("mul", output_value, output_value)
        return builder.binary(
            "mul", grad, builder.unary("conj", builder.binary("sub", 1, inner))
        )

    def get_tuple(index: int, value: _AotNativeTuple):
        return value.values[int(index)]

    def batch_norm_backward(*args: Any):
        key = tuple(id(item) if isinstance(item, _AotNativeSymbol) else hash(repr(item)) for item in args)
        cached = batch_norm_cache.get(key)
        if cached is not None:
            return cached
        grad, input_value, weight, running_mean, running_var, training, eps = args
        tensor_args = (grad, input_value)
        attrs = {
            "has_weight": weight is not None,
            "has_running_mean": running_mean is not None,
            "has_running_var": running_var is not None,
            "training": bool(training),
            "eps": float(eps),
        }
        optional = tuple(item for item in (weight, running_mean, running_var) if item is not None)
        value = builder.helper(
            "batch_norm_backward",
            tensor_args + optional,
            attrs=attrs,
            shape=input_value.shape,
            outputs=3,
        )
        assert isinstance(value, _AotNativeTuple)
        batch_norm_cache[key] = value
        return value

    def _shared_tuple_key(args: tuple[Any, ...]) -> tuple[int, ...]:
        # One backward tuple must be shared by every gradient slot of the
        # same forward node: the derivative formulas each re-derive the same
        # call, and re-emitting it per slot triples the expensive work.
        return tuple(
            id(item) if isinstance(item, _AotNativeSymbol) else hash(repr(item))
            for item in args
        )

    def group_norm_backward(
        grad, input_value, num_groups, weight=None, bias=None, eps=1e-5
    ):
        # The kernel accumulates in float32 and reads a float32 gradient;
        # a narrower gradient widens once at this kernel boundary instead of
        # on every upstream gradient edge.
        if grad.dtype is not None and _dtype_width(grad.dtype) < _dtype_width(
            tensorplay.float32
        ):
            grad = builder.cast(grad, tensorplay.float32)
        tensor_args = [grad, input_value]
        has_weight = weight is not None
        has_bias = bias is not None
        if has_weight:
            tensor_args.append(weight)
        if has_bias:
            tensor_args.append(bias)
        channels = int(input_value.shape[1])
        key = _shared_tuple_key((grad, input_value, num_groups, weight, bias, eps))
        cached = tuple_op_cache.get(key)
        if cached is not None:
            return cached
        value = builder.helper(
            "group_norm_backward",
            tuple(tensor_args),
            attrs={
                "num_groups": int(num_groups),
                "has_weight": has_weight,
                "has_bias": has_bias,
                "eps": float(eps),
            },
            shape=input_value.shape,
            outputs=3,
            output_shapes=(
                tuple(input_value.shape),
                (channels,),
                (channels,),
            ),
        )
        assert isinstance(value, _AotNativeTuple)
        tuple_op_cache[key] = value
        return value

    def scaled_dot_product_attention_backward(
        grad, query, key, value, is_causal=False, impl=0
    ):
        grad = _narrow(grad)
        query = _narrow(query)
        key = _narrow(key)
        value = _narrow(value)
        key_tuple = _shared_tuple_key((grad, query, key, value, is_causal, impl))
        cached = tuple_op_cache.get(key_tuple)
        if cached is not None:
            return cached
        result = builder.helper(
            "scaled_dot_product_attention_backward",
            (grad, query, key, value),
            attrs={"is_causal": bool(is_causal), "impl": int(impl)},
            shape=query.shape,
            outputs=3,
            output_shapes=(query.shape, key.shape, value.shape),
        )
        assert isinstance(result, _AotNativeTuple)
        tuple_op_cache[key_tuple] = result
        return result

    def convolution_backward(
        grad,
        input_value,
        weight,
        bias_sizes,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        output_mask,
    ):
        del bias_sizes
        mask = tuple(bool(item) for item in output_mask)
        if len(mask) != 3:
            raise ValueError("convolution_backward output_mask must have three entries")
        key = _shared_tuple_key(
            (
                grad,
                input_value,
                weight,
                tuple(stride),
                tuple(padding),
                tuple(dilation),
                bool(transposed),
                tuple(output_padding),
                int(groups),
            )
        )
        cached = tuple_op_cache.get(key)
        if cached is not None:
            if cached.node is not None and cached.mask is not None:
                merged = tuple(left or right for left, right in zip(cached.mask, mask))
                if merged != cached.mask:
                    cached.node.set_ints_attr("output_mask", [int(item) for item in merged])
                    cached.mask = merged
            return cached
        narrowed = (_narrow(grad), _narrow(input_value), _narrow(weight))
        bias_shape = (
            (int(weight.shape[1]) * int(groups),)
            if bool(transposed)
            else (int(weight.shape[0]),)
        )
        result = builder.helper(
            "convolution_backward",
            narrowed,
            attrs={
                "stride": tuple(int(item) for item in stride),
                "padding": tuple(int(item) for item in padding),
                "dilation": tuple(int(item) for item in dilation),
                "transposed": bool(transposed),
                "output_padding": tuple(int(item) for item in output_padding),
                "groups": int(groups),
                "output_mask": tuple(int(item) for item in mask),
            },
            shape=input_value.shape,
            outputs=3,
            output_shapes=(
                tuple(input_value.shape),
                tuple(weight.shape),
                bias_shape,
            ),
        )
        if not isinstance(result, _AotNativeTuple):
            raise TypeError("convolution_backward did not produce three outputs")
        result.mask = mask
        tuple_op_cache[key] = result
        return result

    def _conv_axis_gradient(slot: int, transposed: bool):
        # The captured conv spellings carry their own derivative formulas,
        # each asking for one gradient of the same convolution.  Every one of
        # these names funnels into a masked slot of the shared convolution
        # backward, so one tuple node serves all requested slots.
        def axis_gradient(
            grad,
            input_value,
            weight,
            stride,
            padding,
            dilation,
            groups,
            output_padding=(0, 0),
        ):
            shape = getattr(input_value, "shape", ())
            rank = max(len(tuple(shape)) - 2, 1) if shape else 2

            def pair(value: Any) -> tuple[int, ...]:
                if isinstance(value, (tuple, list)):
                    return tuple(int(item) for item in value)
                return (int(value),) * rank

            mask = tuple(slot == index for index in range(3))
            return get_tuple(
                slot,
                convolution_backward(
                    grad,
                    input_value,
                    weight,
                    None,
                    pair(stride),
                    pair(padding),
                    pair(dilation),
                    transposed,
                    pair(output_padding),
                    groups,
                    mask,
                ),
            )

        return axis_gradient

    def _conv_transpose_axis_gradient(slot: int):
        def axis_gradient(
            grad,
            input_value,
            weight,
            stride,
            padding,
            output_padding,
            groups,
            dilation,
        ):
            return _conv_axis_gradient(slot, True)(
                grad,
                input_value,
                weight,
                stride,
                padding,
                dilation,
                groups,
                output_padding,
            )

        return axis_gradient

    conv1d_grad_input = _conv_axis_gradient(0, False)
    conv1d_grad_weight = _conv_axis_gradient(1, False)
    conv1d_grad_bias = _conv_axis_gradient(2, False)
    conv2d_grad_input = _conv_axis_gradient(0, False)
    conv2d_grad_weight = _conv_axis_gradient(1, False)
    conv2d_grad_bias = _conv_axis_gradient(2, False)
    conv3d_grad_input = _conv_axis_gradient(0, False)
    conv3d_grad_weight = _conv_axis_gradient(1, False)
    conv3d_grad_bias = _conv_axis_gradient(2, False)
    conv_transpose1d_grad_input = _conv_transpose_axis_gradient(0)
    conv_transpose1d_grad_weight = _conv_transpose_axis_gradient(1)
    conv_transpose1d_grad_bias = _conv_transpose_axis_gradient(2)
    conv_transpose2d_grad_input = _conv_transpose_axis_gradient(0)
    conv_transpose2d_grad_weight = _conv_transpose_axis_gradient(1)
    conv_transpose2d_grad_bias = _conv_transpose_axis_gradient(2)
    conv_transpose3d_grad_input = _conv_transpose_axis_gradient(0)
    conv_transpose3d_grad_weight = _conv_transpose_axis_gradient(1)
    conv_transpose3d_grad_bias = _conv_transpose_axis_gradient(2)

    def max_pool_backward(grad, input_value, kernel_size, stride, padding, dilation, ceil_mode):
        values = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) else (kernel_size,)
        # Spatial rank follows the pooling input: a scalar or single-entry
        # parameter applies to every spatial dimension alike.
        input_rank = len(tuple(input_value.shape)) - 2 if hasattr(input_value, "shape") else None
        rank = input_rank if input_rank in (1, 2, 3) else len(values)
        if rank not in (1, 2, 3):
            raise TypeError("max_pool spatial parameters must have one to three entries")

        def spatial_arg(value: Any) -> tuple[int, ...]:
            items = tuple(value) if isinstance(value, (tuple, list)) else (value,)
            if len(items) == 1 and rank > 1:
                items = items * rank
            return items

        return builder.helper(
            f"max_pool{rank}d_backward",
            (grad, input_value),
            attrs={
                "kernel_size": spatial_arg(kernel_size),
                "stride": spatial_arg(stride),
                "padding": spatial_arg(padding),
                "dilation": spatial_arg(dilation),
                "ceil_mode": bool(ceil_mode),
            },
            shape=input_value.shape,
        )

    def adaptive_avg_pool_backward(grad, input_value):
        rank = len(tuple(input_value.shape)) - 2
        if rank not in (1, 2, 3):
            raise TypeError("adaptive_avg_pool input must have one to three spatial dimensions")
        return builder.helper(
            f"adaptive_avg_pool{rank}d_backward",
            (grad, input_value),
            shape=input_value.shape,
        )

    def avg_pool_pair(value: Any, default: tuple[int, int] | None = None) -> tuple[int, int]:
        if value is None:
            if default is None:
                raise TypeError("avg_pool2d requires a spatial parameter")
            return default
        if isinstance(value, bool):
            raise TypeError("avg_pool2d spatial parameters must be integers")
        if isinstance(value, int):
            item = int(value)
            return (item, item)
        if isinstance(value, (tuple, list)):
            if len(value) == 1 and isinstance(value[0], int) and not isinstance(value[0], bool):
                item = int(value[0])
                return (item, item)
            if len(value) == 2 and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                return (int(value[0]), int(value[1]))
        raise TypeError("avg_pool2d spatial parameters must contain one or two integers")

    def avg_pool2d_backward(
        grad,
        input_value,
        kernel_size,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None,
    ):
        values = tuple(kernel_size) if isinstance(kernel_size, (tuple, list)) else (kernel_size,)
        # Spatial rank follows the pooling input, not the kernel tuple: a
        # scalar kernel_size widens across all spatial dimensions of the
        # input, so a 4-D input with kernel_size=2 pools in 2-D.
        input_rank = len(tuple(input_value.shape)) - 2 if hasattr(input_value, "shape") else None
        rank = input_rank if input_rank in (1, 2, 3) else len(values)
        if rank not in (1, 2, 3):
            raise TypeError("avg_pool spatial parameters must have one to three entries")

        def spatial(value: Any, default: tuple[int, ...] | None = None) -> tuple[int, ...]:
            if value is None:
                if default is None:
                    raise TypeError("avg_pool requires a spatial parameter")
                return default
            if isinstance(value, bool):
                raise TypeError("avg_pool spatial parameters must be integers")
            if isinstance(value, int):
                return (int(value),) * rank
            if isinstance(value, (tuple, list)):
                if len(value) == 1 and isinstance(value[0], int) and not isinstance(value[0], bool):
                    return (int(value[0]),) * rank
                if len(value) == rank and all(
                    isinstance(item, int) and not isinstance(item, bool) for item in value
                ):
                    return tuple(int(item) for item in value)
            raise TypeError("avg_pool spatial parameters have the wrong rank")

        kernel = spatial(kernel_size)
        stride_values = spatial(stride, kernel)
        padding_values = spatial(padding, (0,) * rank)
        if not isinstance(ceil_mode, bool) or not isinstance(count_include_pad, bool):
            raise TypeError("avg_pool2d boolean parameters must be bool")
        attrs: dict[str, Any] = {
            "kernel_size": kernel,
            "stride": stride_values,
            "padding": padding_values,
            "ceil_mode": ceil_mode,
            "count_include_pad": count_include_pad,
        }
        if divisor_override is not None:
            if isinstance(divisor_override, bool) or not isinstance(divisor_override, int):
                raise TypeError("avg_pool2d divisor_override must be an integer or None")
            attrs["divisor_override"] = int(divisor_override)
        return builder.helper(
            f"avg_pool{rank}d_backward",
            (grad, input_value),
            attrs=attrs,
            shape=input_value.shape,
        )

    def maybe_multiply(value: Any, factor: Any) -> Any:
        # Derivative-formula helper: multiplication by a unit scalar is
        # skipped; anything else lowers to a native multiply that accepts
        # symbol/symbol or symbol/scalar operands.
        if isinstance(factor, numbers.Real) and factor == 1:
            return value
        if isinstance(value, numbers.Real) and value == 1:
            return factor
        return builder.binary("mul", value, factor)

    def maybe_divide(value: Any, divisor: Any) -> Any:
        if isinstance(divisor, numbers.Real) and divisor == 1:
            return value
        return builder.binary("div", value, divisor)

    def mul_tensor_backward(grad, other, *_args):
        return builder.binary("mul", grad, other)

    def div_tensor_self_backward(grad, other, *_args):
        return builder.binary("div", grad, other)

    def div_tensor_other_backward(grad, self_value, other, *_args):
        numerator = builder.binary(
            "mul", builder.unary("neg", grad), self_value
        )
        denominator = builder.binary("mul", other, other)
        return builder.binary("div", numerator, denominator)

    def threshold_backward(grad, output, threshold):
        return builder.helper(
            "threshold_backward",
            (grad, output),
            attrs={"threshold": threshold},
            shape=grad.shape,
        )

    def index_select_backward(grad, self_value, dim, index):
        return builder.helper(
            "index_select_backward",
            (grad, index),
            attrs={
                "self_sizes": tuple(int(item) for item in self_value.shape),
                "dim": int(dim),
            },
            shape=self_value.shape,
        )

    def gather_backward(grad, self_value, dim, index, sparse_grad=False):
        return builder.helper(
            "gather_backward",
            (grad, self_value, index),
            attrs={"dim": int(dim), "sparse_grad": bool(sparse_grad)},
            shape=self_value.shape,
        )

    def silu_backward(grad, input_value):
        sigmoid = builder.unary("sigmoid", input_value)
        correction = builder.binary(
            "add",
            1,
            builder.binary(
                "mul",
                input_value,
                builder.binary("sub", 1, sigmoid),
            ),
        )
        return builder.binary("mul", grad, builder.binary("mul", sigmoid, correction))

    return {
        "add": binary("add"),
        "sub": binary("sub"),
        "mul": binary("mul"),
        "div": binary("div"),
        # The dim-variety sum backward restates the reduction's broadcast
        # inverse: reinsert the singleton axes the reduction removed, then
        # broadcast the tangent to the input extent.  The generated C++
        # autograd resolves the same-named helper at link time; this entry
        # serves the interpreted reverse-graph builder only.
        "_sum_dim_backward": _sum_dim_backward_env,
        "tanh_backward": tanh_backward_env,
        "matmul": matmul_binary("matmul"),
        "mm": matmul_binary("mm"),
        "neg": unary("neg"),
        "pos": unary("pos"),
        "t": unary("t"),
        "reshape": builder.reshape,
        "expand": builder.expand,
        "squeeze": builder.squeeze,
        "sum": builder.sum,
        "get_tuple": get_tuple,
        "maybe_multiply": maybe_multiply,
        "maybe_divide": maybe_divide,
        "mul_tensor_backward": mul_tensor_backward,
        "div_tensor_self_backward": div_tensor_self_backward,
        "div_tensor_other_backward": div_tensor_other_backward,
        "batch_norm_backward": batch_norm_backward,
        "group_norm_backward": group_norm_backward,
        "scaled_dot_product_attention_backward": scaled_dot_product_attention_backward,
        "convolution_backward": convolution_backward,
        "conv1d_grad_input": conv1d_grad_input,
        "conv1d_grad_weight": conv1d_grad_weight,
        "conv1d_grad_bias": conv1d_grad_bias,
        "conv2d_grad_input": conv2d_grad_input,
        "conv2d_grad_weight": conv2d_grad_weight,
        "conv2d_grad_bias": conv2d_grad_bias,
        "conv3d_grad_input": conv3d_grad_input,
        "conv3d_grad_weight": conv3d_grad_weight,
        "conv3d_grad_bias": conv3d_grad_bias,
        "conv_transpose1d_grad_input": conv_transpose1d_grad_input,
        "conv_transpose1d_grad_weight": conv_transpose1d_grad_weight,
        "conv_transpose1d_grad_bias": conv_transpose1d_grad_bias,
        "conv_transpose2d_grad_input": conv_transpose2d_grad_input,
        "conv_transpose2d_grad_weight": conv_transpose2d_grad_weight,
        "conv_transpose2d_grad_bias": conv_transpose2d_grad_bias,
        "conv_transpose3d_grad_input": conv_transpose3d_grad_input,
        "conv_transpose3d_grad_weight": conv_transpose3d_grad_weight,
        "conv_transpose3d_grad_bias": conv_transpose3d_grad_bias,
        "max_pool2d_backward": max_pool_backward,
        "max_pool1d_backward": max_pool_backward,
        "max_pool3d_backward": max_pool_backward,
        "adaptive_avg_pool2d_backward": adaptive_avg_pool_backward,
        "adaptive_avg_pool1d_backward": adaptive_avg_pool_backward,
        "adaptive_avg_pool3d_backward": adaptive_avg_pool_backward,
        "avg_pool2d_backward": avg_pool2d_backward,
        "avg_pool1d_backward": avg_pool2d_backward,
        "avg_pool3d_backward": avg_pool2d_backward,
        "conj": unary("conj"),
        "sin": unary("sin"),
        "cos": unary("cos"),
        "exp": unary("exp"),
        "log": unary("log"),
        "sqrt": unary("sqrt"),
        "rsqrt": unary("rsqrt"),
        "sigmoid": unary("sigmoid"),
        "tanh": unary("tanh"),
        "abs": unary("abs"),
        "sign": unary("sign"),
        "threshold_backward": threshold_backward,
        "silu_backward": silu_backward,
        "index_select_backward": index_select_backward,
        "gather_backward": gather_backward,
    }


def _aot_schema_for(
    specs: dict[str, tuple[Any, dict[str, str]]], op_name: str
) -> tuple[Any, dict[str, str]] | None:
    candidates = [op_name]
    if op_name in {"add", "sub", "mul", "div"}:
        candidates.insert(0, f"{op_name}.Tensor")
    for candidate in candidates:
        if candidate in specs:
            return specs[candidate]
    return None


def _aot_default_value(value: Any) -> Any:
    if value is None:
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value == "{}":
        return ()
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return tuple(int(item.strip()) for item in value[1:-1].split(",") if item.strip())
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value


def _aot_add_adjoint(
    builder: _AotNativeGraphBuilder,
    adjoints: dict[Node, _AotNativeSymbol],
    target: Any,
    contribution: Any,
) -> bool:
    if not isinstance(target, Node) or not isinstance(contribution, _AotNativeSymbol):
        return contribution is None
    previous = adjoints.get(target)
    adjoints[target] = contribution if previous is None else builder.binary(
        "add", previous, contribution
    )
    return True


def _build_aot_backward(
    graph_module: GraphModule,
    native_module: Any,
    forward_lowering: _NativeLowering,
    saved_nodes: list[Node],
    runtime_values: dict[Node, Any],
    runtime_cast_values: dict[tuple[Node, Any], Any],
    runtime_inputs: list[Any],
    public_node: Node,
) -> tuple[Any, list[int]] | None:
    try:
        specs = _aot_derivative_specs()
    except (ImportError, ModuleNotFoundError, OSError):
        # Derivative tooling/config unavailable: AOT lowering is optional;
        # the caller falls back to the non-AOT native path.
        return None
    builder = _AotNativeGraphBuilder(native_module)
    external_nodes = list(graph_module.graph.placeholders) + [
        node for node in graph_module.graph.nodes if node.op == "get_attr"
    ]
    external_symbols: list[_AotNativeSymbol] = []
    forward_symbols: dict[Node, _AotNativeSymbol] = {}
    for index, node in enumerate(external_nodes):
        symbol = builder.input(runtime_inputs[index])
        external_symbols.append(symbol)
        forward_symbols[node] = symbol
    # Literal tensors are lifted by native lowering after placeholders and
    # module attributes.  They are immutable graph inputs and can therefore
    # be shared by every derivative expression without saving activations.
    for value in forward_lowering.constant_values:
        builder.literal(value)
    saved_symbols: list[_AotNativeSymbol] = []
    for node in saved_nodes:
        actual = runtime_values.get(node)
        if actual is None:
            return None
        symbol = builder.input(actual)
        saved_symbols.append(symbol)
        forward_symbols[node] = symbol
    for key in forward_lowering.autocast_outputs:
        source_node, dtype = key
        source = forward_symbols.get(source_node)
        actual = runtime_cast_values.get(key)
        if actual is None:
            return None
        if source is None:
            traced = _traced_value(graph_module, source_node)
            if traced is None:
                return None
            source = _AotNativeSymbol(
                builder, None, tuple(int(item) for item in traced.shape), traced.dtype
            )
            forward_symbols[source_node] = source
        converted = builder.input(actual)
        builder._cast_memo[(id(source), dtype)] = (source, converted)
    tangent = builder.input(runtime_values[public_node])
    adjoints: dict[Node, _AotNativeSymbol] = {public_node: tangent}
    view_adjoints: dict[Node, dict[int, _AotNativeSymbol]] = {}
    batch_norm_cache: dict[tuple[int, ...], _AotNativeTuple] = {}
    tuple_op_cache: dict[tuple[int, ...], _AotNativeTuple] = {}
    autocast_state: dict[str, Any] = {"dtype": None}
    formula_env = _build_aot_formula_env(
        builder,
        batch_norm_cache=batch_norm_cache,
        tuple_op_cache=tuple_op_cache,
        autocast_state=autocast_state,
    )

    def sum_to_shape(value: _AotNativeSymbol, target_shape: tuple[int, ...]) -> _AotNativeSymbol | None:
        current_shape = tuple(int(item) for item in value.shape)
        if len(current_shape) < len(target_shape):
            # An under-shaped contribution (a formula leaning on scalar
            # broadcast, e.g. a mean gradient divided by numel) grows by
            # prepending singleton axes; the expand below sizes the rest.
            for _ in range(len(target_shape) - len(current_shape)):
                value = builder.unsqueeze(value, 0)
            current_shape = tuple(int(item) for item in value.shape)
        leading = len(current_shape) - len(target_shape)
        reduce_dims = list(range(leading))
        for index, target_dim in enumerate(target_shape):
            current_index = leading + index
            current_dim = current_shape[current_index]
            if target_dim == 1 and current_dim != 1:
                reduce_dims.append(current_index)
            elif target_dim != current_dim and current_dim != 1:
                return None
        reduced = builder.sum(value, tuple(reduce_dims), keepdim=True) if reduce_dims else value
        if tuple(reduced.shape) != tuple(target_shape):
            if len(tuple(reduced.shape)) > len(target_shape):
                reduced = builder.reshape(reduced, target_shape)
            else:
                reduced = builder.expand(reduced, target_shape)
        return reduced

    def add_adjoint(target: Any, contribution: Any) -> bool:
        if not isinstance(target, Node) or not isinstance(contribution, _AotNativeSymbol):
            return contribution is None
        target_symbol = forward_symbols.get(target)
        target_value = runtime_values.get(target)
        if target_symbol is not None:
            target_shape = tuple(int(item) for item in target_symbol.shape)
        elif target_value is not None and hasattr(target_value, "shape"):
            target_shape = tuple(int(item) for item in target_value.shape)
        else:
            target_shape = None
        if target_shape is not None:
            contribution = sum_to_shape(contribution, target_shape)
            if contribution is None:
                return False
        # Gradient edges carry the forward output's element type: a formula
        # contribution computed in a promoted type is narrowed here so the
        # consuming formulas see the same element widths the captured graph
        # executed with.  A pure widening on a pointwise edge is exempt: the
        # consuming formulas widen operands to one arithmetic width anyway,
        # so keeping the narrower storage skips a conversion pass without
        # changing any computed value.
        target_dtype = None
        if target_value is not None and hasattr(target_value, "dtype"):
            target_dtype = target_value.dtype
        elif target_symbol is not None:
            target_dtype = target_symbol.dtype
        else:
            traced = _traced_value(graph_module, target)
            target_dtype = getattr(traced, "dtype", None)
            if target_dtype is None:
                try:
                    external_value = runtime_inputs[external_nodes.index(target)]
                except ValueError:
                    external_value = None
                target_dtype = getattr(external_value, "dtype", None)
        if (
            target_dtype is not None
            and contribution.dtype is not None
            and contribution.dtype != target_dtype
            and not _adjoint_promotion_deferrable(
                target, contribution.dtype, target_dtype
            )
        ):
            contribution = builder.cast(contribution, target_dtype)
        return _aot_add_adjoint(builder, adjoints, target, contribution)

    for node in reversed(graph_module.graph.nodes):
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        grad = adjoints.get(node)
        if grad is None and node not in view_adjoints:
            continue
        op_name = _target_name(node.target)
        # Ops whose dispatch narrows their inputs to a reduced precision at
        # capture time run their matrix products in that same element type;
        # the derivative callables read this slot to narrow their operands.
        if op_name in _AUTOCAST_GEMM_OPS:
            sample = node.meta.get("val")
            autocast_state["dtype"] = getattr(sample, "dtype", None)
        else:
            autocast_state["dtype"] = None
        if op_name == "getitem":
            if (
                node.op == "call_function"
                and len(node.args) == 2
                and isinstance(node.args[0], Node)
                and isinstance(node.args[1], tuple)
                and len(node.args[1]) == 2
                and isinstance(node.args[1][0], slice)
                and node.args[1][0] == slice(None)
                and node.args[1][1] is None
                and grad is not None
            ):
                source = node.args[0]
                source_symbol = forward_symbols.get(source)
                if source_symbol is None or len(grad.shape) != len(source_symbol.shape) + 1:
                    return None
                # This form inserts one singleton axis after the leading
                # slice.  Its reverse is an axis removal, not a broadcast
                # reduction: the inserted axis may sit in the middle of the
                # shape, so a generic leading-dimension reduction is wrong.
                if tuple(grad.shape[0:1]) != tuple(source_symbol.shape[0:1]):
                    return None
                contribution = builder.squeeze(grad, dim=1)
                if tuple(contribution.shape) != tuple(source_symbol.shape):
                    return None
                if not add_adjoint(source, contribution):
                    return None
                continue
            if (
                node.op != "call_function"
                or len(node.args) != 2
                or not isinstance(node.args[0], Node)
                or not isinstance(node.args[1], int)
                or node.args[0].op not in {
                    "call_function",
                    "call_method",
                }
                or _target_name(node.args[0].target)
                not in {"chunk", "split", "split_with_sizes", "unbind"}
                or grad is None
            ):
                return None
            source = node.args[0]
            sample = source.meta.get("val")
            if not isinstance(sample, (tuple, list)):
                return None
            index = int(node.args[1])
            if index < 0:
                index += len(sample)
            if index < 0 or index >= len(sample):
                return None
            slots = view_adjoints.setdefault(source, {})
            previous = slots.get(index)
            slots[index] = grad if previous is None else builder.binary(
                "add", previous, grad
            )
            continue
        if op_name in {"chunk", "split", "split_with_sizes", "unbind"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            source = node.args[0]
            sample = node.meta.get("val")
            slots = view_adjoints.pop(node, None)
            if not isinstance(sample, (tuple, list)) or not slots:
                return None
            source_symbol = forward_symbols.get(source)
            source_value = runtime_values.get(source)
            if source_symbol is not None:
                source_shape = tuple(int(item) for item in source_symbol.shape)
            elif source_value is not None and hasattr(source_value, "shape"):
                source_shape = tuple(int(item) for item in source_value.shape)
            else:
                return None
            kwargs = dict(node.kwargs or {})
            if op_name == "unbind":
                if set(kwargs) - {"dim"} or len(node.args) > 2:
                    return None
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                dim_arg = node.args[1] if len(node.args) > 1 else kwargs.get("dim", 0)
            else:
                if set(kwargs) - {"dim"} or len(node.args) > 3:
                    return None
                if len(node.args) > 2 and "dim" in kwargs:
                    return None
                dim_arg = node.args[2] if len(node.args) > 2 else kwargs.get("dim", 0)
            if isinstance(dim_arg, bool) or not isinstance(dim_arg, int):
                return None
            dim_arg = int(dim_arg)
            rank = len(source_shape)
            if dim_arg < 0:
                dim_arg += rank
            if dim_arg < 0 or dim_arg >= rank:
                return None
            parts: list[_AotNativeSymbol] = []
            template = next(iter(slots.values()))
            for index, output in enumerate(sample):
                part = slots.get(index)
                if part is None:
                    part_shape = getattr(output, "shape", None)
                    if part_shape is None:
                        return None
                    part = builder.zeros_like(template, part_shape)
                if op_name == "unbind":
                    part = builder.unsqueeze(part, dim_arg)
                parts.append(part)
            contribution = builder.cat(tuple(parts), dim_arg, source_shape)
            if not add_adjoint(source, contribution):
                return None
            continue
        if op_name == "cat":
            if grad is None or not node.args or not isinstance(node.args[0], (tuple, list)):
                return None
            sources = tuple(node.args[0])
            if not sources or any(not isinstance(item, Node) for item in sources):
                return None
            dim_arg = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
            if not isinstance(dim_arg, int):
                return None
            shapes: list[tuple[int, ...]] = []
            sizes: list[int] = []
            for source in sources:
                source_symbol = forward_symbols.get(source)
                source_value = runtime_values.get(source)
                if source_symbol is not None:
                    shape_tuple = tuple(int(item) for item in source_symbol.shape)
                elif source_value is not None and hasattr(source_value, "shape"):
                    shape_tuple = tuple(int(item) for item in source_value.shape)
                else:
                    return None
                shapes.append(shape_tuple)
                dim_index = dim_arg if dim_arg >= 0 else dim_arg + len(shape_tuple)
                if dim_index < 0 or dim_index >= len(shape_tuple):
                    return None
                sizes.append(shape_tuple[dim_index])
            parts = builder.split(grad, tuple(sizes), dim_arg, tuple(shapes))
            for source, part in zip(sources, parts.values):
                if not add_adjoint(source, part):
                    return None
            continue
        if op_name == "index_select":
            if grad is None:
                return None
            kwargs = dict(node.kwargs or {})
            if node.op == "call_method":
                if set(kwargs) - {"dim", "index"} or not node.args:
                    return None
                input_node = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "index" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                index_node = positional[1] if len(positional) > 1 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            else:
                if len(node.args) > 3 or set(kwargs) - {"dim", "index"}:
                    return None
                input_node = node.args[0]
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "index" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                index_node = node.args[2] if len(node.args) > 2 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            if (
                not isinstance(input_node, Node)
                or not isinstance(dim, int)
                or isinstance(dim, bool)
                or not isinstance(index_node, Node)
            ):
                return None
            input_symbol = forward_symbols.get(input_node)
            index_symbol = forward_symbols.get(index_node)
            if input_symbol is None or index_symbol is None:
                return None
            contribution = formula_env["index_select_backward"](
                grad, input_symbol, dim, index_symbol
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue
        if op_name == "gather":
            if grad is None:
                return None
            kwargs = dict(node.kwargs or {})
            if node.op == "call_method":
                if set(kwargs) - {"dim", "index"} or not node.args:
                    return None
                input_node = node.args[0]
                positional = list(node.args[1:])
                if len(positional) > 2:
                    return None
                if positional and "dim" in kwargs:
                    return None
                if len(positional) > 1 and "index" in kwargs:
                    return None
                dim = positional[0] if positional else kwargs.get("dim")
                index_node = positional[1] if len(positional) > 1 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            else:
                if len(node.args) > 3 or set(kwargs) - {"dim", "index"}:
                    return None
                input_node = node.args[0]
                if len(node.args) > 1 and "dim" in kwargs:
                    return None
                if len(node.args) > 2 and "index" in kwargs:
                    return None
                dim = node.args[1] if len(node.args) > 1 else kwargs.get("dim")
                index_node = node.args[2] if len(node.args) > 2 else kwargs.get("index")
                if dim is None or index_node is None:
                    return None
            if (
                not isinstance(input_node, Node)
                or not isinstance(dim, int)
                or isinstance(dim, bool)
                or not isinstance(index_node, Node)
            ):
                return None
            input_symbol = forward_symbols.get(input_node)
            index_symbol = forward_symbols.get(index_node)
            if input_symbol is None or index_symbol is None:
                return None
            contribution = formula_env["gather_backward"](
                grad, input_symbol, dim, index_symbol
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue
        if grad is None:
            return None
        if op_name == "linear":
            if len(node.args) not in {2, 3} or not all(
                isinstance(item, Node) for item in node.args[:2]
            ):
                return None
            input_node, weight_node = node.args[:2]
            bias_node = node.args[2] if len(node.args) == 3 else None
            if bias_node is not None and not isinstance(bias_node, Node):
                return None
            input_value = forward_symbols[input_node]
            weight_value = forward_symbols[weight_node]
            sample = node.meta.get("val")
            ac_dtype = getattr(sample, "dtype", None)
            narrow = builder.cast
            backward = builder.helper(
                "linear_backward",
                (
                    narrow(input_value, ac_dtype),
                    narrow(grad, ac_dtype),
                    narrow(weight_value, ac_dtype),
                ),
                attrs={"output_mask": (1, 1, 1)},
                outputs=3,
                output_shapes=(
                    tuple(input_value.shape),
                    tuple(weight_value.shape),
                    (int(weight_value.shape[0]),),
                ),
            )
            if not isinstance(backward, _AotNativeTuple):
                return None
            input_grad, weight_grad, bias_grad = backward.values
            if not add_adjoint(input_node, input_grad):
                return None
            if not add_adjoint(weight_node, weight_grad):
                return None
            if bias_node is not None and not add_adjoint(bias_node, bias_grad):
                return None
            continue

        if op_name in {"reshape", "view"}:
            if not node.args or not isinstance(node.args[0], Node):
                return None
            input_node = node.args[0]
            input_symbol = forward_symbols.get(input_node)
            input_value = runtime_values.get(input_node)
            if input_symbol is not None:
                input_shape = tuple(int(item) for item in input_symbol.shape)
            elif input_value is not None and hasattr(input_value, "shape"):
                input_shape = tuple(int(item) for item in input_value.shape)
            else:
                return None
            contribution = builder.reshape(grad, input_shape)
            if not add_adjoint(input_node, contribution):
                return None
            continue

        if op_name == "flatten":
            if not node.args or not isinstance(node.args[0], Node):
                return None
            source_value = runtime_values.get(node.args[0])
            if source_value is None:
                return None
            # the input storage.  Keep that metadata-only dependency out of
            # the saved-tensor list so the rebuilt graph can omit the pooled
            # activation while still producing the exact reshape backward.
            contribution = builder.reshape(grad, tuple(source_value.shape))
            if not add_adjoint(node.args[0], contribution):
                return None
            continue

        if op_name in {"max_pool1d", "max_pool2d", "max_pool3d"}:
            # The captured op carries no indices value, so the derivative
            # formula for the with-indices overload cannot apply; the native
            # backward kernel recomputes the argmax instead.
            if len(node.args) != 7 or not isinstance(node.args[0], Node):
                return None
            if bool(node.args[6]):
                return None
            input_node = node.args[0]
            if input_node not in forward_symbols:
                return None
            rank = int(op_name[-2])
            kernel = _spatial_int_list(node.args[1], length=rank)
            stride = _spatial_int_list(node.args[2], default=kernel, length=rank)
            padding = _spatial_int_list(node.args[3], default=[0] * rank, length=rank)
            dilation = _spatial_int_list(node.args[4], default=[1] * rank, length=rank)
            if any(item is None for item in (kernel, stride, padding, dilation)):
                return None
            contribution = builder.helper(
                f"max_pool{rank}d_backward",
                (grad, forward_symbols[input_node]),
                attrs={
                    "kernel_size": tuple(kernel),
                    "stride": tuple(stride),
                    "padding": tuple(padding),
                    "dilation": tuple(dilation),
                    "ceil_mode": bool(node.args[5]),
                },
                shape=forward_symbols[input_node].shape,
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue

        if op_name == "dropout":
            if len(node.args) != 4 or not isinstance(node.args[0], Node):
                return None
            input_node, probability, training, _inplace = node.args
            if not isinstance(probability, numbers.Real) or not isinstance(training, bool):
                return None
            if float(probability) == 0.0 or not training:
                if not add_adjoint(input_node, grad):
                    return None
                continue
            return None

        if op_name == "interpolate":
            if len(node.args) != 7 or not isinstance(node.args[0], Node):
                return None
            input_node, output_size, scale_factor, mode, align_corners, recompute, antialias = node.args
            if (
                scale_factor is not None
                or mode != "nearest"
                or align_corners is not None
                or recompute is not None
                or antialias is not False
            ):
                return None
            input_symbol = forward_symbols.get(input_node)
            input_value = runtime_values.get(input_node)
            if input_symbol is not None:
                input_shape = tuple(int(item) for item in input_symbol.shape)
            elif input_value is not None and hasattr(input_value, "shape"):
                input_shape = tuple(int(item) for item in input_value.shape)
            else:
                return None
            output_size = tuple(int(item) for item in output_size)
            spatial_rank = len(input_shape) - 2
            if spatial_rank not in (1, 2, 3):
                return None
            contribution = builder.helper(
                f"upsample_nearest{spatial_rank}d_backward",
                (grad,),
                attrs={
                    "output_size": output_size,
                    "input_size": input_shape,
                },
                shape=input_shape,
            )
            if not add_adjoint(input_node, contribution):
                return None
            continue

        if op_name == "permute":
            if len(node.args) < 2 or not isinstance(node.args[0], Node):
                return None
            source = forward_symbols.get(node.args[0])
            if source is None:
                return None
            dims = node.args[1]
            if len(node.args) > 2:
                dims = tuple(node.args[1:])
            else:
                dims = tuple(int(item) for item in dims)
            if any(isinstance(item, bool) or not isinstance(item, int) for item in dims):
                return None
            contribution = builder.helper(
                "permute_backward",
                (grad, source),
                attrs={"dims": tuple(int(item) for item in dims)},
                shape=source.shape,
            )
            if not add_adjoint(node.args[0], contribution):
                return None
            continue

        if op_name == "float":
            if len(node.args) != 1 or not isinstance(node.args[0], Node):
                return None
            if not add_adjoint(node.args[0], grad):
                return None
            continue

        # A captured call_method carries no overload suffix: ``sum(dim=...)``
        # lands here spelled ``sum``, whose whole-tensor schema has no ``dim``
        # parameter.  Route calls that pass a dimension to the dim-variety
        # schema when one exists; every other spelling keeps the base schema.
        node_kwargs = dict(node.kwargs or {})
        has_dim_arg = "dim" in node_kwargs or (
            len(node.args) > 1 and isinstance(node.args[1], (int, list, tuple))
        )
        schema = None
        if has_dim_arg:
            schema = _aot_schema_for(specs, f"{op_name}.dim_IntList")
        if schema is None:
            schema = _aot_schema_for(specs, op_name)
        if schema is None:
            return None
        parsed, formulas = schema
        if node.op == "call_method":
            schema_args = tuple(parsed.args)
            if len(node.args) > len(schema_args):
                return None
            bound_args: dict[str, Any] = {}
            for arg, value in zip(schema_args, node.args):
                if arg.kwonly:
                    return None
                bound_args[arg.name] = value
            schema_names = {arg.name for arg in schema_args}
            for name, value in (node.kwargs or {}).items():
                if name not in schema_names or name in bound_args:
                    return None
                bound_args[name] = value
            if any(
                arg.name not in bound_args and arg.default is None
                for arg in schema_args
            ):
                return None
            arg_values = tuple(
                (arg.name, bound_args[arg.name])
                for arg in schema_args
                if arg.name in bound_args
            )
        elif op_name == "batch_norm":
            names = (
                "input", "running_mean", "running_var", "weight", "bias",
                "training", "momentum", "eps",
            )
            arg_values = tuple(zip(names, node.args))
        else:
            arg_values = tuple(zip((arg.name for arg in parsed.args), node.args))
        if op_name == "batch_norm":
            context = {name: value for name, value in arg_values}
        else:
            context = dict(arg_values)
        for arg in parsed.args:
            if arg.name not in context and arg.default is not None:
                context[arg.name] = _aot_default_value(arg.default)
        context["grad"] = grad
        # others only need metadata or their inputs (e.g. adaptive average
        # pooling).  A pruned saved-tensor set must not require a symbol for
        # a result that the selected formula never reads.
        context["result"] = forward_symbols.get(node)
        tensor_params = {
            name for name, value in context.items() if isinstance(value, Node)
        }
        env = dict(formula_env)
        # A derivative formula may read a saved tensor only for metadata
        # (e.g. the reduction backward re-expands the tangent to the input
        # shape) and never touch its value.  The rebuilt forward graph drops
        # every activation whose value no backward op consumes, so such
        # inputs resolve to no symbol here.  Rebuild one from the traced
        # shape/dtype instead: metadata reads succeed without re-saving the
        # activation, while a formula that emits an operation on the value
        # still fails the eval below and keeps the graph off the AOT route.
        for name, value in context.items():
            if not isinstance(value, Node) or value in forward_symbols:
                continue
            traced = _traced_value(graph_module, value)
            if traced is None or not hasattr(traced, "shape"):
                continue
            forward_symbols[value] = _AotNativeSymbol(
                builder,
                None,
                tuple(int(item) for item in traced.shape),
                getattr(traced, "dtype", None),
            )
        env.update(
            {
                name: forward_symbols.get(value) if isinstance(value, Node) else value
                for name, value in context.items()
            }
        )
        try:
            for arg_name, formula in formulas.items():
                target = context.get(arg_name)
                if not isinstance(target, Node):
                    continue
                translated = _aot_formula_python(formula, tensor_params)
                contribution = eval(translated, {"__builtins__": {}}, env)
                if not add_adjoint(target, contribution):
                    return None
        except (KeyError, NameError, NotImplementedError, TypeError, ValueError, RuntimeError):
            return None

    grad_positions: list[int] = []
    for index, node in enumerate(external_nodes):
        actual = runtime_inputs[index]
        if not getattr(actual, "requires_grad", False):
            continue
        contribution = adjoints.get(node)
        if contribution is None:
            continue
        builder._materialize(contribution)
        builder.graph.register_output(contribution.value)
        grad_positions.append(index)
    if not grad_positions:
        return None
    needed_saved_nodes = [
        node
        for node, symbol in zip(saved_nodes, saved_symbols)
        if getattr(symbol.value, "use_count", 0) != 0
    ]
    return builder.graph, grad_positions, needed_saved_nodes


def _copy_back_mutations(lowering: Any, inputs: Any, outputs: Any) -> None:
    for position, output_index in lowering._mutations:
        inputs[position].copy_(outputs[output_index])


class _AotNativeLowering:

    def __init__(
        self,
        graph_module: GraphModule,
        forward_graph: Any,
        backward_graph: Any,
        attribute_targets: list[str],
        grad_positions: list[int],
        constant_values: list[Any] | None = None,
        mutations: list[tuple[int, int]] | None = None,
    ) -> None:
        self.graph_module = graph_module
        self.forward_graph = forward_graph
        self.backward_graph = backward_graph
        self.placeholders = graph_module.graph.placeholders
        self.attribute_targets = attribute_targets
        self.constant_values = list(constant_values or [])
        self.grad_positions = list(grad_positions)
        self._mutations = list(mutations or [])
        self.input_count = len(self.placeholders) + len(self.attribute_targets)
        self._tensorplay_codegen = "stax-aot-native"
        self._tensorplay_backward_codegen = "stax-aot-native"
        lowering = self
        from ....autograd import Function

        class _AotAutogradFunction(Function):
            _tensorplay_direct_backward = True

            @staticmethod
            def forward(ctx: Any, *inputs: Any) -> Any:
                # The custom function owns the only gradient edge for this
                # region.  Keep native forward operators outside the eager
                # autograd graph even when the fused apply path leaves grad
                # recording enabled around the Python callback.
                import tensorplay

                with tensorplay.no_grad():
                    outputs = lowering.forward_graph.execute(
                        [*inputs, *lowering.constant_values]
                    )
                    _copy_back_mutations(lowering, inputs, outputs)
                    # Buffer-update outputs trail the saved tensors; they are
                    # epilogue state, not backward inputs.
                    ctx.save_for_backward(
                        *inputs,
                        *lowering.constant_values,
                        *outputs[1 : len(outputs) - len(lowering._mutations)],
                    )
                return outputs[0]

            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> tuple[Any, ...]:
                grad_output = grad_outputs[0] if grad_outputs else None
                if grad_output is None:
                    return (None,) * lowering.input_count
                saved = list(ctx.saved_tensors)
                outputs = lowering.backward_graph.execute([*saved, grad_output])
                by_position = dict(zip(lowering.grad_positions, outputs))
                return tuple(by_position.get(index) for index in range(lowering.input_count))

        self._autograd_function = _AotAutogradFunction

    def _bind_inputs(self, *args: Any, **kwargs: Any) -> list[Any]:
        bound = self.graph_module.signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        inputs = [
            bound.arguments[node.target if isinstance(node.target, str) else node.name]
            for node in self.placeholders
        ]
        inputs.extend(self.graph_module._get_attr(target) for target in self.attribute_targets)
        return inputs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._bind_inputs(*args, **kwargs)
        import tensorplay

        if not tensorplay.is_grad_enabled() or not any(
            getattr(value, "requires_grad", False) for value in inputs
        ):
            outputs = self.forward_graph.execute([*inputs, *self.constant_values])
            _copy_back_mutations(self, inputs, outputs)
            return outputs[0]
        return self._autograd_function.apply(*inputs)


def _lower_aot_native(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_fusion: bool = True,
) -> _AotNativeLowering | None:
    """Build separate native forward/backward graphs at the AOT boundary."""

    try:
        import tensorplay

        native_module = getattr(tensorplay._C, "_stax", None)
        tensor_type = tensorplay.Tensor
    except (AttributeError, ImportError):
        return None
    if native_module is None or not hasattr(native_module.Graph, "execute"):
        return None
    if len(example_inputs) != len(graph_module.graph.placeholders):
        return None
    if any(not isinstance(value, tensor_type) for value in example_inputs):
        return None
    if not tensorplay.is_grad_enabled():
        return None

    external_nodes = list(graph_module.graph.placeholders) + [
        node for node in graph_module.graph.nodes if node.op == "get_attr"
    ]
    attribute_targets = [node.target for node in external_nodes if node.op == "get_attr"]
    runtime_inputs = list(example_inputs)
    runtime_inputs.extend(graph_module._get_attr(target) for target in attribute_targets)
    if not any(getattr(value, "requires_grad", False) for value in runtime_inputs):
        return None

    saved_nodes = [
        node
        for node in graph_module.graph.nodes
        if node.op in {"call_function", "call_method"}
        and not isinstance(node.meta.get("val"), (tuple, list))
    ]
    output_values = [
        value for output in graph_module.graph.outputs for value in _nodes(output.args)
    ]
    if len(output_values) != 1:
        return None
    public_node = output_values[0]
    forward_lowering = _lower_native(
        graph_module,
        example_inputs,
        use_fusion=use_fusion,
        extra_output_nodes=saved_nodes,
        save_autocast_inputs=True,
    )
    if forward_lowering is None:
        return None
    if len(runtime_inputs) + len(forward_lowering.constant_values) != len(
        forward_lowering.graph.inputs
    ):
        return None

    # Training BatchNorm updates running buffers during forward.  A compiler
    # trace must not perform that update a second time; restore non-gradient
    # capture path separates tracing state from the user execution state.
    snapshots: list[tuple[Any, Any]] = []
    seen_attributes: set[int] = set()
    try:
        for target in attribute_targets:
            value = graph_module._get_attr(target)
            if (
                isinstance(value, tensor_type)
                and not getattr(value, "requires_grad", False)
                and id(value) not in seen_attributes
            ):
                snapshots.append((value, value.detach().clone()))
                seen_attributes.add(id(value))
        with tensorplay.no_grad():
            forward_outputs = forward_lowering.graph.execute(
                [*runtime_inputs, *forward_lowering.constant_values]
            )
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    finally:
        if snapshots:
            with tensorplay.no_grad():
                for value, snapshot in snapshots:
                    value.copy_(snapshot)

    cast_count = len(forward_lowering.autocast_outputs)
    if len(forward_outputs) != 1 + len(saved_nodes) + cast_count + len(forward_lowering._mutations):
        return None
    runtime_values: dict[Node, Any] = {public_node: forward_outputs[0]}
    for index, node in enumerate(saved_nodes, start=1):
        runtime_values[node] = forward_outputs[index]
    runtime_cast_values = dict(zip(
        forward_lowering.autocast_outputs,
        forward_outputs[1 + len(saved_nodes) : 1 + len(saved_nodes) + cast_count],
    ))

    built = _build_aot_backward(
        graph_module,
        native_module,
        forward_lowering,
        saved_nodes,
        runtime_values,
        runtime_cast_values,
        runtime_inputs,
        public_node,
    )
    if built is None:
        return None
    _, grad_positions, needed_saved_nodes = built
    # The first graph is a shape/materialization graph.  Rebuild the forward
    # graph with only the values that the source-derived backward graph reads,
    # intermediate until backward.
    forward_lowering = _lower_native(
        graph_module,
        example_inputs,
        use_fusion=use_fusion,
        extra_output_nodes=needed_saved_nodes,
        save_autocast_inputs=True,
    )
    if forward_lowering is None or forward_lowering.autocast_outputs != list(runtime_cast_values):
        return None
    rebuilt = _build_aot_backward(
        graph_module,
        native_module,
        forward_lowering,
        needed_saved_nodes,
        runtime_values,
        runtime_cast_values,
        runtime_inputs,
        public_node,
    )
    if rebuilt is None:
        return None
    backward_graph, grad_positions, rebuilt_saved_nodes = rebuilt
    if rebuilt_saved_nodes != needed_saved_nodes:
        return None
    return _AotNativeLowering(
        graph_module,
        forward_lowering.graph,
        backward_graph,
        attribute_targets,
        grad_positions,
        constant_values=forward_lowering.constant_values,
        mutations=forward_lowering._mutations,
    )


# Option patch each compile ``mode`` selects, keyed by the backend option
# namespace.  ``default`` leaves every knob at its built-in value;
# ``reduce-overhead`` replays the artifact through CUDA graphs;
# ``max-autotune(-no-cudagraphs)`` additionally selects the exhaustive
# search tier: the widened pointwise candidate table plus coordinate-descent
# refinement of every benchmark winner.
_MODE_OPTIONS: dict[str, dict[str, bool]] = {
    "default": {},
    "reduce-overhead": {
        "stax.cudagraphs": True,
    },
    "max-autotune-no-cudagraphs": {
        "stax.max_autotune": True,
        "stax.coordinate_descent_tuning": True,
    },
    "max-autotune": {
        "stax.max_autotune": True,
        "stax.cudagraphs": True,
        "stax.coordinate_descent_tuning": True,
    },
}

# Every backend option key accepted by ``stax`` (and therefore by explicit
# ``options`` dicts and mode patches alike).
_STAX_OPTIONS = (
    "stax.native",
    "stax.fusion",
    "stax.cuda_codegen",
    "stax.triton",
    "stax.cudagraphs",
    "stax.max_autotune",
    "stax.coordinate_descent_tuning",
)


def list_mode_options(mode: str | None = None) -> dict[str, Any]:
    """Return the optimization options each compile ``mode`` selects.

    With ``mode`` set, returns that mode's option patch; with ``mode``
    unset, returns the full mode-to-options mapping.  Unknown modes raise.
    The patch feeds the same validation path as explicit backend options,
    so options supplied by a caller override the mode patch per key.
    """

    try:
        return dict(_MODE_OPTIONS[mode]) if mode else dict(_MODE_OPTIONS)
    except KeyError as exc:
        raise RuntimeError(
            f"Unrecognized mode={mode}, should be one of: "
            f"{', '.join(_MODE_OPTIONS)}"
        ) from exc


def stax(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    mode: str | None = None,
    options: dict[str, Any] | None = None,
    name: str | None = None,
    dynamic: bool | None = None,
    strict_native: bool = False,
    **kwargs: Any,
):
    """Compile one canonical graph and return an executable callable.

    ``example_inputs`` and backend options are part of the same contract as
    metadata in the frontend and uses the native graph when its lowering
    contract is satisfied.  ``strict_native`` makes a failed lowering a hard
    compiler error, so a benchmark can never report the Python GraphModule
    executor as compiled performance.
    """
    del name, kwargs
    if mode not in (None, *_MODE_OPTIONS):
        raise RuntimeError(f"unknown Stax optimization mode: {mode!r}")
    # A mode's option patch is applied first; explicit options overlay it
    # per key (a mode selects defaults, an explicit option wins).
    resolved = dict(_MODE_OPTIONS[mode or "default"])
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError(f"options must be a dict, got {type(options)!r}")
        unknown = set(options).difference(_STAX_OPTIONS)
        if unknown:
            raise RuntimeError(
                f"Unexpected Stax optimization option(s): {sorted(unknown)!r}"
            )
        if any(not isinstance(value, bool) for value in options.values()):
            raise RuntimeError("Stax optimization options must be bool values")
        resolved.update(options)
    use_native = resolved.get("stax.native", True)
    use_fusion = resolved.get("stax.fusion", True)
    use_cuda_codegen = resolved.get("stax.cuda_codegen", False)
    use_triton = resolved.get("stax.triton", True)
    max_autotune = resolved.get("stax.max_autotune", False)
    coordinate_descent_tuning = resolved.get(
        "stax.coordinate_descent_tuning", False
    )
    cudagraphs_requested = resolved.get("stax.cudagraphs", False)
    compiled = _lower_stax_region(
        graph_module,
        example_inputs,
        use_native=use_native,
        use_fusion=use_fusion,
        use_cuda_codegen=use_cuda_codegen,
        use_triton=use_triton,
        max_autotune=max_autotune,
        coordinate_descent_tuning=coordinate_descent_tuning,
        dynamic=dynamic,
        strict_native=strict_native,
    )
    if not cudagraphs_requested or isinstance(compiled, GraphModule):
        return compiled
    from ..cudagraphs import cudagraph_wrap

    wrapped, reason = cudagraph_wrap(
        compiled, graph_module, example_inputs, dynamic=dynamic
    )
    if reason is not None:
        graph_module._stax_cudagraph_skip_reason = reason
        return compiled
    return wrapped


def _lower_stax_region(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_native: bool,
    use_fusion: bool,
    use_cuda_codegen: bool,
    use_triton: bool,
    max_autotune: bool,
    coordinate_descent_tuning: bool,
    dynamic: bool | None,
    strict_native: bool,
):
    """Lower one canonical graph and return an executable callable.

    ``example_inputs`` and backend options are part of the same contract as
    metadata in the frontend and uses the native graph when its lowering
    contract is satisfied.  ``strict_native`` makes a failed lowering a hard
    compiler error, so a benchmark can never report the Python GraphModule
    executor as compiled performance.
    """
    if use_native and use_fusion:
        fused_cpu_graph = _lower_cpu_fused_pointwise(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cpu_graph is not None:
            graph_module._stax_native_graph = fused_cpu_graph.graph
            return fused_cpu_graph
        # A region whose tail is a reduction folds the whole expression into
        # the reduction loop: one pass over the input, no intermediate.
        fused_cpu_reduction = _lower_cpu_fused_reduction(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cpu_reduction is not None:
            graph_module._stax_codegen = "stax-fused-cpu-reduce"
            return fused_cpu_reduction
        # A region whose reductions sit in the middle stages per row: each
        # reduction folds the row to one value that the following work reads
        # as a broadcast, so the region still reads its inputs once.
        row_fused_cpu = _lower_cpu_row_fusion(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if row_fused_cpu is not None:
            graph_module._stax_codegen = "stax-fused-cpu-rowfuse"
            return row_fused_cpu
    if use_native and use_fusion and use_cuda_codegen:
        fused_cuda_graph = _lower_cuda_fused_pointwise(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cuda_graph is not None:
            graph_module._stax_codegen = "stax-cuda"
            return fused_cuda_graph
    if use_native and use_triton:
        # Keep Triton optional and lazy.  Importing tensorplay on a CPU-only
        # machine must not import Triton or its compiler toolchain.
        try:
            first = example_inputs[0]
            is_cuda = first.device.is_cuda()
        except (AttributeError, IndexError):
            is_cuda = False
        if is_cuda:
            from .codegen.triton import (
                compile_graph_module as compile_triton_graph,
            )

            # The per-segment emitter is the source of fusion truth for the
            # shapes it accepts: one trailing reduction per segment plus a
            # store-time epilogue.  The row-staged kernel answers only for
            # what that form cannot express -- a reduction whose result
            # feeds elementwise work feeding another reduction (softmax,
            # normalization) -- so it claims the region last.
            try:
                triton_graph = compile_triton_graph(
                    graph_module,
                    example_inputs,
                    max_autotune=max_autotune,
                    coordinate_descent_tuning=coordinate_descent_tuning,
                    strict_native=strict_native,
                )
            except Exception:
                # Unsupported constants or shape forms belong on the native
                # graph path and must not abort compilation; generated-kernel
                # build failures (toolchain, unsupported op) degrade the same
                # way.
                triton_graph = None
            if triton_graph is not None:
                graph_module._stax_codegen = "triton"
                return triton_graph
            if use_fusion:
                row_fused_cuda = _lower_cuda_row_fusion(
                    graph_module,
                    example_inputs,
                    strict_native=strict_native,
                    dynamic=bool(dynamic is True),
                )
                if row_fused_cuda is not None:
                    graph_module._stax_codegen = "stax-fused-cuda-rowfuse"
                    return row_fused_cuda
    if use_native and use_fusion and not use_cuda_codegen:
        fused_cuda_graph = _lower_cuda_fused_pointwise(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if fused_cuda_graph is not None:
            graph_module._stax_codegen = "stax-cuda"
            return fused_cuda_graph
    # The AOT boundary is a property of the graph's gradient surface, not of
    # the callable's shape: bare functions carry no training flag, so a
    # grad-carrying input list must select the split forward/backward route
    # exactly as a training module does.  The builder re-checks grad mode and
    # returns None for inference calls, leaving the routes below untouched.
    if use_native and (
        getattr(graph_module.root, "training", False)
        or any(
            getattr(value, "requires_grad", False) for value in example_inputs
        )
    ):
        aot_graph = _lower_aot_native(
            graph_module, example_inputs, use_fusion=use_fusion
        )
        if aot_graph is not None:
            graph_module._stax_native_graph = aot_graph.forward_graph
            return aot_graph
        if strict_native and any(
            getattr(graph_module._get_attr(node.target), "requires_grad", False)
            for node in graph_module.graph.nodes
            if node.op == "get_attr"
        ):
            raise RuntimeError(
                "AOT backward graph for the captured training region"
            )
    if use_native and use_fusion:
        # Nothing claimed the region whole.  Its fusible runs are still worth
        # compiling: each becomes one kernel, and the operators between them
        # run as captured instead of the region losing every compiled route.
        segmented = _lower_cpu_segmented(
            graph_module,
            example_inputs,
            strict_native=strict_native,
            dynamic=bool(dynamic is True),
        )
        if segmented is not None:
            graph_module._stax_codegen = "stax-fused-cpu-segments"
            return segmented
    # Fused native pointwise nodes are forward execution primitives.  A
    # training graph that reaches this fallback did not obtain an AOT reverse
    # graph, so keep ordinary tensor operators here to preserve autograd
    # recording for every parameter.
    native_fusion = use_fusion and not getattr(graph_module.root, "training", False)
    native_graph = (
        _lower_native(graph_module, example_inputs, use_fusion=native_fusion)
        if use_native
        else None
    )
    if native_graph is not None:
        graph_module._stax_native_graph = native_graph.graph
        return native_graph
    if strict_native:
        raise RuntimeError(
            "strict_native Stax lowering failed: captured graph has no native executable"
        )
    # No native executable exists for this graph (scalar placeholders,
    # factory-only regions, unsupported surface).  Fall back to the
    # generated Python executor so the region still runs with captured
    # semantics instead of failing to compile.
    return graph_module.recompile()
