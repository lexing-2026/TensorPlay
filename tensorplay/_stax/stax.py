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

from ..graph.passes import POINTWISE_FUSED_OP_NAMES
from ..graph import GraphModule, Node
from ..library import CustomOpDef as _CustomOpDef


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
            if template is None:
                raise RuntimeError("native multi-output node has no output template")
            return ("nested", template)
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
    "square",
    "tanh",
    "relu",
    "mm",
    # Tensor kernels used by the ResNet inference graph.  These are kept in
    # the native graph instead of falling back to the generated Python
    # executor; the latter still calls every functional wrapper through the
    # interpreter and is not a compiled path in any meaningful sense.
    "conv2d",
    "add_relu",
    "batch_norm",
    "max_pool2d",
    "adaptive_avg_pool2d",
    "flatten",
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


def _int_list(value: Any) -> list[int] | None:
    """Return a constant integer list accepted by a native Stax node."""

    if not isinstance(value, (tuple, list)):
        return None
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        return None
    return [int(item) for item in value]


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
    """Install the C steady-state trampoline when the extension offers it.

    ``exec_fn`` selects the steady-state execution entry; the default is the
    native ``Graph.execute`` bound method.  Lowerings with a direct kernel
    entry pass it here so the trampoline skips the graph walk.  Soft failure
    keeps the plain lowering object: older extensions without
    ``_stax.install_call_trampoline`` simply skip the fast entry.
    """

    try:
        import tensorplay

        installer = getattr(
            tensorplay._C._stax, "install_call_trampoline", None
        )
        if installer is None:
            return
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
        )
    except Exception:
        # Never let a trampoline install failure break compilation.
        pass


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
        self.native_values = dict(native_values or {})
        self._tensorplay_codegen = "stax-native"
        # (id, _version) memo of the last resolved input vector; attributes
        # and constants appended by _bind_inputs are process-stable.
        self._bind_fp: Any = None
        self._last_bound_inputs: list[Any] | None = None
        _attach_fast_call(self)

    @staticmethod
    def _input_route_fingerprint(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        import tensorplay

        def fp(value: Any) -> Any:
            if isinstance(value, tensorplay.Tensor):
                return (
                    "t",
                    id(value),
                    getattr(value, "_version", None),
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
    ) -> None:
        super().__init__(graph_module, graph, attribute_targets)
        self._expected_shape = expected_shape
        self._expected_dtype = expected_dtype
        self._expected_device = expected_device
        self._gradient_plan = gradient_plan
        self._strict_native = strict_native
        self._tensorplay_codegen = "stax-fused-cpu"
        self._autograd_function: Any | None = None
        # Runtime-generated C kernel for the native route: straight-line
        # compiler-scheduled code replacing the program interpreter when the
        # system compiler is available.  None keeps the graph-execute path.
        self._native_runner = native_runner
        # Route memo (id, _version, requires_grad) per input: eligibility and
        # autograd routing are pure functions of these, so steady-state calls
        # skip the per-input shape/dtype/device/contiguity probes entirely.
        # In-place mutation bumps _version; fresh tensors have fresh ids.
        self._route_fp: tuple[Any, ...] | None = None
        self._route: str | None = None
        _attach_fast_call(self, exec_fn=self._native_runner)
        if gradient_plan is not None:
            from ..autograd import Function

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
    ) -> bool:
        try:
            import tensorplay

            tensor_type = tensorplay.Tensor
        except (AttributeError, ImportError):
            return False
        return bool(inputs) and all(
            isinstance(value, tensor_type)
            and value.device.is_cpu()
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
            return (
                "t",
                id(value),
                getattr(value, "_version", None),
                bool(getattr(value, "requires_grad", False)),
            )
        return ("o", id(value))

    def _resolve_route(self, inputs: list[Any]) -> str:
        if not self._eligible_inputs(
            inputs,
            self._expected_shape,
            self._expected_dtype,
            self._expected_device,
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
) -> tuple[list[Node], list[int], list[float], list[tuple[str, int, int, int]], int] | None:
    """Encode one canonical pointwise graph as Stax's postfix program.

    ``skip_node`` excludes one node from the program (used by the Triton
    reduction-epilogue path, which folds a full-reduction ``sum`` tail into
    the kernel instead of lowering it as an op), with ``output_override``
    naming the program's result node.

    ``opcodes`` selects the target opcode table: the CPU fused interpreter
    supports the base table only, while the Triton code generator accepts
    the extended surface (comparisons, ``where``, order relations, casts).
    Values are typed numerically — comparisons yield booleans, everything
    else yields floats, and the program output must be a float value (the
    store path types outputs off the sample dtype).
    """

    table = _TRITON_OPCODES if opcodes is None else opcodes

    external_nodes = list(graph_module.graph.placeholders)
    refs: dict[Node, int] = {
        node: index for index, node in enumerate(external_nodes)
    }
    ref_types: dict[int, str] = {
        index: "num" for index in range(len(external_nodes))
    }
    program: list[int] = []
    constants: list[float] = []
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
        result = len(external_nodes) + temp_count
        temp_count += 1
        instructions.append((op_name, lhs, rhs, result))
        return result

    for node in graph_module.graph.nodes:
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

        if op_name in {"add", "sub"} and len(node.args) == 3 and "alpha" not in (
            kwargs
        ):
            lhs, rhs, alpha = node.args
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
    if (not program and not allow_empty) or len(output_values) != 1 or (
        output_values[0] not in refs
    ):
        return None
    if value_type(refs[output_values[0]]) != "num":
        # A boolean program output would need a typed store path.
        return None
    return external_nodes, program, constants, instructions, refs[output_values[0]]


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
        value.device != first.device
        or value.dtype != first.dtype
        or value.shape != first.shape
        or not value.is_contiguous()
        for value in example_inputs[1:]
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
    try:
        from .codegen.cpp import build_cpu_native_kernel

        native_runner = build_cpu_native_kernel(
            instructions,
            constants,
            len(external_nodes),
            output_ref,
            shape=first.shape,
            device=first.device,
        )
    except Exception:
        native_runner = None
    if extended and native_runner is None:
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
    if not extended and any(
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


def _lower_native(
    graph_module: GraphModule,
    example_inputs: list[Any],
    *,
    use_fusion: bool = True,
    extra_output_nodes: list[Node] | None = None,
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
    attribute_targets: list[str] = []
    constant_values: list[Any] = []
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

    fused_relu_convs: dict[Node, Node] = {}
    fused_add_relus: dict[Node, Node] = {}
    fused_relu_nodes: set[Node] = set()
    layout_values: dict[Node, bool] = {}
    channels_last_values: dict[Node, Any] = {}
    peel_conv_bias = bool(example_inputs and example_inputs[0].device.is_cuda())
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
    for node in graph_module.graph.placeholders:
        values[node] = graph.add_input()
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
        attribute_targets.append(node.target)

    for node in graph_module.graph.nodes:
        folded = folded_convs.get(node)
        if folded is None:
            continue
        _, folded_weight, folded_bias = folded
        native_weight = graph.add_input()
        native_bias = graph.add_input()
        folded_native_inputs[node] = (native_weight, native_bias)
        constant_values.extend((folded_weight, folded_bias))

    for node in graph_module.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue
        if node.op not in {"call_function", "call_method"}:
            return None
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
            if (
                op_name != "relu"
                or node.kwargs not in ({"inplace": False}, {"inplace": True})
            ):
                return None
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
            if not isinstance(value, Node) or value not in values:
                return None
            return values[value]

        def add_tensor_input(native_node: Any, value: Any) -> bool:
            resolved = node_value(value)
            if resolved is None or isinstance(resolved, tuple):
                return False
            native_node.add_input(resolved)
            return True

        if op_name == "conv2d":
            if len(node.args) != 7:
                return None
            input_node, weight_node, bias_node, stride, padding, dilation, groups = node.args
            fused_relu = fused_relu_convs.get(node)
            use_conv_relu = fused_relu is not None and not peel_conv_bias
            native_node = graph.create_node(
                "conv2d_relu" if use_conv_relu else "conv2d",
                node.name,
            )
            conv_input = channels_last_value(input_node)
            if conv_input is None:
                return None
            native_node.add_input(conv_input)
            folded_inputs = folded_native_inputs.get(node)
            bias_input = None
            bias_tensor = None
            if folded_inputs is not None:
                native_node.add_input(folded_inputs[0])
                bias_input = folded_inputs[1]
                folded_spec = folded_convs.get(node)
                bias_tensor = folded_spec[2] if folded_spec is not None else None
            else:
                conv_weight = channels_last_value(weight_node)
                if conv_weight is None:
                    return None
                native_node.add_input(conv_weight)
                if bias_node is None:
                    native_node.set_int_attr("has_bias", 0)
                else:
                    bias_input = node_value(bias_node)
                    if bias_input is None:
                        return None
                    try:
                        bias_tensor = graph_module._get_attr(bias_node.target)
                    except (AttributeError, TypeError):
                        return None
            if bias_input is None:
                native_node.set_int_attr("has_bias", 0)
            elif not peel_conv_bias or use_conv_relu:
                native_node.add_input(bias_input)
                native_node.set_int_attr("has_bias", 1)
            else:
                # the cuDNN call because cuDNN is slower with it.  Keep the
                # bias as a broadcast pointwise input after the convolution.
                native_node.set_int_attr("has_bias", 0)
            if not all(
                _set_int_list_attr(native_node, key, value)
                for key, value in (
                    ("stride", stride),
                    ("padding", padding),
                    ("dilation", dilation),
                )
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

        if op_name == "max_pool2d":
            if len(node.args) != 7:
                return None
            input_node, kernel_size, stride, padding, dilation, ceil_mode, return_indices = node.args
            if return_indices is not False or not isinstance(ceil_mode, bool):
                return None
            native_node = graph.create_node("max_pool2d", node.name)
            if not add_tensor_input(native_node, input_node):
                return None
            for key, value in (
                ("kernel_size", kernel_size),
                ("stride", stride),
                ("padding", padding),
                ("dilation", dilation),
            ):
                if not _set_int_list_attr(native_node, key, value):
                    return None
            native_node.set_int_attr("ceil_mode", int(ceil_mode))
            values[node] = native_node.add_output()
            # The cuDNN tensor descriptor and output follow the input layout;
            # a later convolution can therefore consume the max-pool result
            # without an NCHW round-trip.
            layout_values[node] = use_channels_last and layout_values.get(
                input_node, False
            )
            continue

        if op_name == "adaptive_avg_pool2d":
            if len(node.args) != 2 or not _set_int_list_attr(
                native_node := graph.create_node("adaptive_avg_pool2d", node.name),
                "output_size",
                node.args[1],
            ):
                return None
            if not add_tensor_input(native_node, node.args[0]):
                return None
            values[node] = native_node.add_output()
            layout_values[node] = False
            continue

        if op_name == "flatten":
            if len(node.args) != 2 or node.args[1] != 1 or node.kwargs:
                return None
            native_node = graph.create_node("flatten", node.name)
            if not add_tensor_input(native_node, node.args[0]):
                return None
            native_node.set_int_attr("start_dim", 1)
            native_node.set_int_attr("end_dim", -1)
            values[node] = native_node.add_output()
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
        # ``(input, other, alpha)`` even when alpha is the default 1.  Lower
        # that contract to the native pointwise IR instead of falling back to
        # a Python method call.  Non-unit alpha becomes a scalar multiply and
        # can be consumed by Stax's mul-add fusion pass.
        if op_name in {"add", "sub"} and len(node.args) == 3:
            input_node, other_node, alpha = node.args
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

    if use_fusion and not registered_extra_outputs:
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
    )


class _AotShape(tuple):
    """Tensor metadata that supports both ``shape`` and ``shape()`` schemas."""

    def __new__(cls, value: Any):
        return super().__new__(cls, (int(item) for item in value))

    def __call__(self) -> tuple[int, ...]:
        return tuple(self)


class _AotNativeSymbol:
    """A symbolic Tensor value used while materializing an AOT backward graph."""

    __slots__ = ("builder", "value", "shape")

    def __init__(self, builder: "_AotNativeGraphBuilder", value: Any, shape: Any):
        self.builder = builder
        self.value = value
        self.shape = _AotShape(shape)

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
        raise NotImplementedError("AOT native expand lowering is not implemented")

    def sum(self, dim: Any = None, keepdim: bool = False) -> "_AotNativeSymbol":
        return self.builder.sum(self, dim, keepdim)

    def numel(self) -> int:
        result = 1
        for item in self.shape:
            result *= item
        return result


class _AotNativeTuple:
    __slots__ = ("values",)

    def __init__(self, values: tuple[_AotNativeSymbol, ...]):
        self.values = values


class _AotNativeGraphBuilder:
    """Small native-IR builder used by the source-derived reverse pass."""

    def __init__(self, native_module: Any):
        self.native_module = native_module
        self.graph = native_module.Graph()

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...]:
        return tuple(int(item) for item in getattr(value, "shape", ()))

    @staticmethod
    def _symbol(value: Any) -> _AotNativeSymbol | None:
        return value if isinstance(value, _AotNativeSymbol) else None

    def input(self, example_value: Any) -> _AotNativeSymbol:
        return _AotNativeSymbol(self, self.graph.add_input(), self._shape(example_value))

    def _add_inputs(self, native_node: Any, args: tuple[Any, ...]) -> list[_AotNativeSymbol]:
        symbols: list[_AotNativeSymbol] = []
        for value in args:
            if not isinstance(value, _AotNativeSymbol):
                raise TypeError("AOT native op received a non-Tensor argument")
            native_node.add_input(value.value)
            symbols.append(value)
        return symbols

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
        return _AotNativeSymbol(self, native_node.add_output(), shape)

    def unary(
        self,
        op_name: str,
        value: _AotNativeSymbol,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> _AotNativeSymbol:
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        native_node.add_input(value.value)
        return _AotNativeSymbol(self, native_node.add_output(), shape or value.shape)

    def helper(
        self,
        op_name: str,
        args: tuple[_AotNativeSymbol, ...],
        *,
        attrs: dict[str, Any] | None = None,
        shape: tuple[int, ...] | None = None,
        outputs: int = 1,
    ) -> _AotNativeSymbol | _AotNativeTuple:
        native_node = self.graph.create_node(op_name, f"aot_{op_name}_{len(self.graph.nodes)}")
        symbols = self._add_inputs(native_node, args)
        del symbols
        for key, value in (attrs or {}).items():
            if isinstance(value, bool) or isinstance(value, int):
                native_node.set_int_attr(key, int(value))
            elif isinstance(value, numbers.Real):
                native_node.set_float_attr(key, float(value))
            elif isinstance(value, (tuple, list)) and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            ):
                native_node.set_ints_attr(key, [int(item) for item in value])
            else:
                raise TypeError(f"unsupported AOT native attribute: {key}={value!r}")
        if outputs == 1:
            return _AotNativeSymbol(self, native_node.add_output(), shape or args[0].shape)
        return _AotNativeTuple(
            tuple(
                _AotNativeSymbol(self, native_node.add_output(), shape or args[0].shape)
                for _ in range(outputs)
            )
        )

    def reshape(self, value: _AotNativeSymbol, shape: Any) -> _AotNativeSymbol:
        normalized = tuple(int(item) for item in shape)
        return self.helper("reshape", (value,), attrs={"shape": normalized}, shape=normalized)  # type: ignore[return-value]

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

    yaml_path = Path(__file__).resolve().parents[2] / "config" / "derivatives.yaml"
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
            callee = expr.callee.split("::")[-1]
            return f"{callee}({args})"
        if isinstance(expr, Method):
            recv = emit(expr.receiver)
            args = ", ".join(emit(a) for a in expr.args)
            name = expr.name
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
) -> dict[str, Any]:
    def binary(name: str):
        return lambda lhs, rhs: builder.binary(name, lhs, rhs)

    def unary(name: str):
        return lambda value: builder.unary(name, value)

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

    def conv_grad(name: str):
        def invoke(grad, input_value, weight, stride, padding, dilation, groups):
            return builder.helper(
                name,
                (grad, input_value, weight),
                attrs={
                    "stride": tuple(stride),
                    "padding": tuple(padding),
                    "dilation": tuple(dilation),
                    "groups": int(groups),
                },
                shape=(
                    input_value.shape
                    if name.endswith("input")
                    else weight.shape
                    if name.endswith("weight")
                    else (grad.shape[1],)
                ),
            )

        return invoke

    def max_pool_backward(grad, input_value, kernel_size, stride, padding, dilation, ceil_mode):
        return builder.helper(
            "max_pool2d_backward",
            (grad, input_value),
            attrs={
                "kernel_size": tuple(kernel_size),
                "stride": tuple(stride),
                "padding": tuple(padding),
                "dilation": tuple(dilation),
                "ceil_mode": bool(ceil_mode),
            },
            shape=input_value.shape,
        )

    def adaptive_avg_pool_backward(grad, input_value):
        return builder.helper(
            "adaptive_avg_pool2d_backward", (grad, input_value), shape=input_value.shape
        )

    def threshold_backward(grad, output, threshold):
        return builder.helper(
            "threshold_backward",
            (grad, output),
            attrs={"threshold": threshold},
            shape=grad.shape,
        )

    return {
        "add": binary("add"),
        "sub": binary("sub"),
        "mul": binary("mul"),
        "div": binary("div"),
        "matmul": binary("matmul"),
        "mm": binary("mm"),
        "neg": unary("neg"),
        "pos": unary("pos"),
        "t": unary("t"),
        "reshape": builder.reshape,
        "sum": builder.sum,
        "get_tuple": get_tuple,
        "batch_norm_backward": batch_norm_backward,
        "conv2d_grad_input": conv_grad("conv2d_grad_input"),
        "conv2d_grad_weight": conv_grad("conv2d_grad_weight"),
        "conv2d_grad_bias": conv_grad("conv2d_grad_bias"),
        "max_pool2d_backward": max_pool_backward,
        "adaptive_avg_pool2d_backward": adaptive_avg_pool_backward,
        "threshold_backward": threshold_backward,
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
    saved_symbols: list[_AotNativeSymbol] = []
    for node in saved_nodes:
        actual = runtime_values.get(node)
        if actual is None:
            return None
        symbol = builder.input(actual)
        saved_symbols.append(symbol)
        forward_symbols[node] = symbol
    tangent = builder.input(runtime_values[public_node])
    adjoints: dict[Node, _AotNativeSymbol] = {public_node: tangent}
    batch_norm_cache: dict[tuple[int, ...], _AotNativeTuple] = {}
    formula_env = _build_aot_formula_env(builder, batch_norm_cache=batch_norm_cache)

    for node in reversed(graph_module.graph.nodes):
        if node.op in {"placeholder", "get_attr", "output"}:
            continue
        grad = adjoints.get(node)
        if grad is None:
            continue
        op_name = _target_name(node.target)
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
            weight_t = builder.unary("t", weight_value, shape=weight_value.shape[::-1])
            input_grad = builder.helper(
                "matmul_backward_self", (grad, input_value, weight_t), shape=input_value.shape
            )
            weight_t_grad = builder.helper(
                "matmul_backward_other", (grad, input_value, weight_t), shape=weight_t.shape
            )
            if not _aot_add_adjoint(builder, adjoints, input_node, input_grad):
                return None
            if not _aot_add_adjoint(
                builder, adjoints, weight_node, builder.unary("t", weight_t_grad, shape=weight_value.shape)
            ):
                return None
            if bias_node is not None:
                dims = tuple(range(max(0, len(grad.shape) - 1)))
                bias_grad = builder.sum(grad, dims, False) if dims else grad
                if not _aot_add_adjoint(builder, adjoints, bias_node, bias_grad):
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
            if not _aot_add_adjoint(builder, adjoints, node.args[0], contribution):
                return None
            continue

        schema = _aot_schema_for(specs, op_name)
        if schema is None:
            return None
        parsed, formulas = schema
        if node.op == "call_method":
            arg_values = (node.args[0],) if node.args else ()
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
                if not _aot_add_adjoint(builder, adjoints, target, contribution):
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


class _AotNativeLowering:

    def __init__(
        self,
        graph_module: GraphModule,
        forward_graph: Any,
        backward_graph: Any,
        attribute_targets: list[str],
        grad_positions: list[int],
    ) -> None:
        self.graph_module = graph_module
        self.forward_graph = forward_graph
        self.backward_graph = backward_graph
        self.placeholders = graph_module.graph.placeholders
        self.attribute_targets = attribute_targets
        self.grad_positions = list(grad_positions)
        self.input_count = len(self.placeholders) + len(self.attribute_targets)
        self._tensorplay_codegen = "stax-aot-native"
        self._tensorplay_backward_codegen = "stax-aot-native"
        lowering = self
        from ..autograd import Function

        class _AotAutogradFunction(Function):
            @staticmethod
            def forward(ctx: Any, *inputs: Any) -> Any:
                outputs = lowering.forward_graph.execute(list(inputs))
                ctx.save_for_backward(*inputs, *outputs[1:])
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
            return self.forward_graph.execute(inputs)[0]
        return self._autograd_function.apply(*inputs)


def _lower_aot_native(
    graph_module: GraphModule,
    example_inputs: list[Any],
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
        use_fusion=False,
        extra_output_nodes=saved_nodes,
    )
    if forward_lowering is None:
        return None
    if len(runtime_inputs) != len(forward_lowering.graph.inputs):
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
            forward_outputs = forward_lowering.graph.execute(runtime_inputs)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    finally:
        if snapshots:
            with tensorplay.no_grad():
                for value, snapshot in snapshots:
                    value.copy_(snapshot)

    if len(forward_outputs) != 1 + len(saved_nodes):
        return None
    runtime_values: dict[Node, Any] = {public_node: forward_outputs[0]}
    for index, node in enumerate(saved_nodes, start=1):
        runtime_values[node] = forward_outputs[index]

    built = _build_aot_backward(
        graph_module,
        native_module,
        forward_lowering,
        saved_nodes,
        runtime_values,
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
        use_fusion=False,
        extra_output_nodes=needed_saved_nodes,
    )
    if forward_lowering is None:
        return None
    rebuilt = _build_aot_backward(
        graph_module,
        native_module,
        forward_lowering,
        needed_saved_nodes,
        runtime_values,
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
    )


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
    if mode not in {None, "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}:
        raise RuntimeError(f"unknown Stax optimization mode: {mode!r}")
    if options is not None:
        if not isinstance(options, dict):
            raise TypeError(f"options must be a dict, got {type(options)!r}")
        unknown = set(options).difference(
            {"stax.native", "stax.fusion", "stax.triton"}
        )
        if unknown:
            raise RuntimeError(
                f"Unexpected Stax optimization option(s): {sorted(unknown)!r}"
            )
        if any(not isinstance(value, bool) for value in options.values()):
            raise RuntimeError("Stax optimization options must be bool values")
    use_native = options is None or options.get("stax.native", True)
    use_fusion = options is None or options.get("stax.fusion", True)
    use_triton = options is None or options.get("stax.triton", True)
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

            triton_graph = compile_triton_graph(
                graph_module,
                example_inputs,
                mode=mode,
                strict_native=strict_native,
            )
            if triton_graph is not None:
                graph_module._stax_codegen = "triton"
                return triton_graph
    if use_native and getattr(graph_module.root, "training", False):
        aot_graph = _lower_aot_native(graph_module, example_inputs)
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
    native_graph = (
        _lower_native(graph_module, example_inputs, use_fusion=use_fusion)
        if use_native
        else None
    )
    if native_graph is not None:
        graph_module._stax_native_graph = native_graph.graph
        return native_graph
    raise RuntimeError(
        "Stax could not lower the captured graph to a native executable"
    )
