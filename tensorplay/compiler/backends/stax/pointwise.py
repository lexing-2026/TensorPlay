"""Elementwise, reduction, row-fusion, and segmentation planning.

The operator tables the code generators read, the plan builders that
decide which chains become one kernel, and the fusers that keep a
chain in one buffer until a consumer needs a result.
"""
from __future__ import annotations

from typing import Any

from ....graph import GraphModule, Node
from ....graph.passes import POINTWISE_FUSED_OP_NAMES
from .ir import (
    _NativeLowering,
    _attach_fast_call,
    _is_scalar,
    _metadata_fingerprint,
    _nodes,
    _normalize_pointwise_grad_output,
    _target_name,
    _tensor_layout,
    _traced_value,
    _FUSED_PROGRAM_OPCODES,
)


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


def _expand_silu(emit: Any, x: int) -> int | None:
    """silu(x) = x * sigmoid(x)."""

    gate = emit("sigmoid", x)
    if gate is None:
        return None
    return emit("mul", x, gate)


#: Composite unary activations and their expansion into program primitives.
#: Each expander receives the builder's ``emit`` and the operand reference
#: and returns the reference of the result.
_COMPOSITE_UNARY_OPS = {
    "silu": _expand_silu,
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
        if op_name in _COMPOSITE_UNARY_OPS:
            # A composite activation expands into primitive instructions, so
            # it fuses with its neighbours and the instruction-level
            # derivative rules cover it without a rule of its own.
            if kwargs not in ({}, {"inplace": False}) or len(node.args) != 1:
                return None
            node_ref = _COMPOSITE_UNARY_OPS[op_name](emit, value_ref(node.args[0]))
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
    narrowed: dict[Node, Any] | None = None,
) -> dict[Node, tuple[Any, ...]]:
    """Plan the pointwise runs of a captured region as generated kernels.

    ``narrowed`` names values whose every reader consumes them converted to
    a narrower element type (the autocast operands of a matrix product).
    A plan exporting such a value stores it at that width directly, so the
    conversion rides the plan's store instead of costing a pass of its own;
    such a plan is worth emitting even when it holds a single operator.
    """

    narrowed = narrowed or {}
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

    def make_plan(nodes: tuple[Node, ...], narrow: bool = True):
        if not nodes:
            return None
        if len(nodes) < 2 and not (narrow and nodes[0] in narrowed):
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
        narrowed_exports = tuple(
            node for node in exports if narrow and node in narrowed
        )
        port_dtypes = tuple(
            narrowed[node] if node in narrowed_exports else output_dtype
            for node in exports
        )
        if len(nodes) < 2 and not narrowed_exports:
            return None
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
        output_dtype_names = tuple(repr(dtype) for dtype in port_dtypes)
        op_name = _register_stax_cuda_pointwise_op(
            program,
            constants,
            output_refs,
            len(externals),
            examples,
            output_dtype_names,
        )
        if op_name is None and narrowed_exports:
            # Only a registered kernel stores ports at their own widths; the
            # same run without the narrowed stores keeps its plan.
            return make_plan(nodes, narrow=False)
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
            narrowed_exports,
        )

    plans: dict[Node, tuple[Any, ...]] = {}
    for segment in segments:
        if segment.kind != "pw":
            continue
        cursor = 0
        while cursor < len(segment.nodes):
            best = None
            for end in range(cursor + 1, len(segment.nodes) + 1):
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


#: Names this layer owns.  The driver re-exports them, so the
#: import surface of the package does not change: operator tables, planners, and elementwise fusers.
__all__ = [
    "_ARITHMETIC_OPS",
    "_CAST_DTYPE_IDS",
    "_CAST_METHOD_DTYPES",
    "_COMPARISON_OPS",
    "_CPU_FUSED_AUTOGRAD_OPS",
    "_CPU_FUSED_OPCODES",
    "_CPU_FUSED_OPS",
    "_CpuFusedPointwiseLowering",
    "_CpuFusedReductionLowering",
    "_CpuRowFusionLowering",
    "_CpuSegmentedLowering",
    "_CudaFusedPointwiseLowering",
    "_CudaRowFusionLowering",
    "_DIMLESS_ONLY_REDUCTIONS",
    "_DIM_ONLY_REDUCTIONS",
    "_FUSED_FWD_BINARY_OPS",
    "_FUSED_FWD_CPU_DTYPES",
    "_FUSED_FWD_CUDA_DTYPES",
    "_FUSED_FWD_MAX_INSTRUCTIONS",
    "_FUSED_FWD_UNARY_OPS",
    "_ForwardFusedProgram",
    "_ForwardPointwiseFuser",
    "_ORDER_OPS",
    "_REDUCTION_METHODS",
    "_STAX_CUDA_OP_CACHE",
    "_STAX_CUDA_OP_COUNTER",
    "_SegmentKernelPlan",
    "_TRITON_EXTRA_OPCODES",
    "_TRITON_OPCODES",
    "_TRITON_POINTWISE_MIN_NUMEL",
    "_UNARY_OPS",
    "_VARIANCE_REDUCTIONS",
    "_broadcast_shape",
    "_build_fused_gradient_graphs",
    "_build_pointwise_program",
    "_compile_segment_kernel",
    "_copy_region_graph",
    "_cuda_triton_pointwise_runner",
    "_elem_dependencies",
    "_expand_row_normalizations",
    "_kernel_step",
    "_lower_cpu_fused_pointwise",
    "_lower_cpu_fused_reduction",
    "_lower_cpu_row_fusion",
    "_lower_cpu_segmented",
    "_lower_cuda_fused_pointwise",
    "_lower_cuda_row_fusion",
    "_native_fused_pointwise_plans",
    "_parse_reduction",
    "_parse_variance_reduction",
    "_plan_row_fusion",
    "_plan_segment_kernel",
    "_reduction_dtype_ok",
    "_register_stax_cuda_pointwise_op",
    "_row_fusion_plan",
    "_segment_externals",
]
