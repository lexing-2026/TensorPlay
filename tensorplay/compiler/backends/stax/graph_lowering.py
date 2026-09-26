"""Lower a captured dispatch-level graph into loop IR buffers.

The walk visits the graph once.  Every operator either has a lowering
(which builds unmaterialized loop nests and views) or becomes a library
call on realized inputs.  A loop nest is materialized when something needs
memory: a library call, a graph output, or a value that is re-read by many
consumers or already carries many reads of its own -- recomputing such a
value in each consumer would cost more than storing it once.
"""

from __future__ import annotations

import operator
from typing import Any

from .loops import (
    Buffer,
    ComputedBuffer,
    ConstantBuffer,
    DeferredOps,
    dtype_name,
    ExternKernel,
    ExternOutput,
    InputBuffer,
    Layout,
    Loops,
    Pointwise,
    Reduction,
    ReinterpretView,
    TemplateKernel,
    TensorBox,
    Const,
    Value,
    View,
    affine_coeff,
    as_index,
    contiguous_strides,
    substitute,
    fresh_symbols,
    record_body,
    set_graph,
    set_ops_handler,
)
from .op_lowerings import LOWERINGS, target_name

#: A multi-user value is stored once it reads more than this many buffers.
REALIZE_READS_THRESHOLD = 4
#: Any value is stored once its inlined body reads more than this many.
REALIZE_ACC_READS_THRESHOLD = 8
#: ... or once its inlined body holds more operations than this.
REALIZE_OPCOUNT_THRESHOLD = 30


#: Operators a template owns, keyed by the name the graph gives them.
_TEMPLATE_OPERATORS = {
    "mm.default": "gemm",
    "addmm.default": "gemm",
    "bmm.default": "gemm",
    "baddbmm.default": "gemm",
    "linear.default": "gemm",
    "matmul.default": "gemm",
    "addmv.default": "gemm",
}
_LINEAR_OPERATOR = "linear.default"


class _IndexCapture:
    """Ops handler that records the single load a view resolves to."""

    def __init__(self):
        self.loads = []

    def load(self, name, index):
        self.loads.append((name, index))
        return Value("load", (name, index, None), "float32")

    def __getattr__(self, name):
        raise NotImplementedError(name)


def _body_stats(loops: Loops) -> tuple[int, int]:
    body = record_body(loops)
    from .loops import iter_values

    values = iter_values(body.root)
    return len(body.loads), len(values)


class GraphLowering:
    def __init__(self, graph_module, example_inputs):
        self.graph_module = graph_module
        self.example_inputs = list(example_inputs)
        self.buffers: dict[str, Any] = {}
        self.operations: list[Any] = []
        self.graph_inputs: list[InputBuffer] = []
        self.constants: dict[str, Any] = {}
        self.graph_outputs: list[Any] = []
        # A region whose output node holds one value returns that value, not a
        # one-element sequence, so the compiled region matches its capture.
        self.single_output = True
        self._counter = 0
        self.device = None

    # -- registry ---------------------------------------------------------
    def new_name(self, prefix="buf") -> str:
        name = f"{prefix}{self._counter}"
        self._counter += 1
        return name

    def get_buffer(self, name):
        return self.buffers[name]

    def register_computed(self, loops: Loops) -> ComputedBuffer:
        size = tuple(loops.ranges)
        layout = Layout(loops.device, loops.dtype, size, contiguous_strides(size))
        buffer = ComputedBuffer(self.new_name(), layout, loops)
        self.buffers[buffer.name] = buffer
        self.operations.append(buffer)
        return buffer

    def register_welford(self, reduction: Reduction):
        """One welford loop nest, two stored results: mean and m2."""

        size = tuple(reduction.ranges)
        layout = Layout(reduction.device, reduction.dtype, size, contiguous_strides(size))
        mean = ComputedBuffer(self.new_name(), layout, reduction)
        m2 = ComputedBuffer(self.new_name(), Layout(reduction.device, reduction.dtype, size, contiguous_strides(size)), reduction)
        m2.welford_parent = mean
        m2.welford_index = 1
        mean.welford_siblings = [m2]
        for b in (mean, m2):
            self.buffers[b.name] = b
        self.operations.append(mean)
        return mean, m2

    # -- realization ------------------------------------------------------
    def mark_reuse(self, box: TensorBox, users: int) -> None:
        node = box.node
        if not isinstance(node, Pointwise):
            return
        reads, opcount = _body_stats(node)
        if users > 1 and (reads > REALIZE_READS_THRESHOLD or opcount > REALIZE_OPCOUNT_THRESHOLD):
            box.realize()
        elif reads > REALIZE_ACC_READS_THRESHOLD:
            box.realize()

    def realize_input(self, box: TensorBox):
        """A buffer or a strided view of one, fit for a library call."""

        node = box.node
        if isinstance(node, Buffer):
            return node
        if isinstance(node, Loops):
            return box.realize()
        if isinstance(node, View):
            node.source.realize()
            strided = self._as_strided_view(node)
            if strided is not None:
                return strided
            copy = TensorBox(Pointwise(node.get_device(), node.get_dtype(), node.make_loader(), node.get_size()))
            return copy.realize()
        raise TypeError(f"cannot realize {node!r}")

    def _as_strided_view(self, view: View):
        size = view.get_size()
        index = fresh_symbols("v", len(size))
        capture = _IndexCapture()
        try:
            with set_ops_handler(capture):
                view.make_loader()(index)
        except NotImplementedError:
            return None
        if len(capture.loads) != 1:
            return None
        name, expr = capture.loads[0]
        expr = as_index(expr)
        strides = []
        for var, extent in zip(index, size):
            coeff = affine_coeff(expr, var)
            if coeff is None:
                return None
            strides.append(0 if extent == 1 else coeff)
        remaining = substitute(expr, {var: Const(0) for var in index})
        if not isinstance(remaining, Const):
            return None
        remaining = remaining.value
        return ReinterpretView(self.buffers[name], tuple(size), tuple(strides), int(remaining))

    # -- library calls ----------------------------------------------------
    def _template_for(self, node, realized_args):
        """The template that owns this operator, if any."""

        from .templates import template_for

        name = target_name(node.target)
        if template_for(_TEMPLATE_OPERATORS.get(name, "")) is None:
            return None
        return template_for(_TEMPLATE_OPERATORS[name])

    def _template_meta(self, node, realized_args):
        """What the template needs to know about this call.

        The template owns its result and its validity, so what is recorded here
        is the call's operands and the properties a configuration is matched
        against -- not an inferred output.
        """

        name = target_name(node.target)
        out_val = node.meta.get("val")
        meta: dict[str, Any] = {
            "operator": name,
            "out_size": tuple(int(s) for s in out_val.shape) if _is_tensor(out_val) else (),
            "out_dtype": getattr(out_val, "dtype", None),
            "device": getattr(out_val, "device", None),
            "requires_grad": bool(getattr(out_val, "requires_grad", False)),
            "operand_positions": tuple(
                i for i, a in enumerate(realized_args) if isinstance(a, Buffer)
            ),
            # The arguments that are not tensors, so a probe can be run
            # through the same operator with a different operand feed.
            "arg_templates": tuple(
                a for a in realized_args if not isinstance(a, Buffer)
            ),
            "call_method": node.op == "call_method",
        }
        operands = [
            realized_args[i]
            for i in meta["operand_positions"]
            if i < len(realized_args)
        ]
        if operands:
            first = operands[0]
            meta["operand_dtype"] = dtype_name(first.get_dtype())
            meta["operand_sizes"] = tuple(
                tuple(int(s) for s in buffer.get_size()) for buffer in operands
            )
        # Which feed position holds which operand, and whether the second one
        # arrives the way a linear layer's weight does.
        meta["operand_specs"] = tuple(
            (position, None) for position in range(len(operands))
        )
        if name == _LINEAR_OPERATOR:
            meta["b_transposed"] = True
        if len(operands) > 2:
            meta["bias_spec"] = (2, None)
        return meta

    def make_extern(self, node, args, kwargs):
        def realize_args(value):
            if isinstance(value, TensorBox):
                return self.realize_input(value)
            if isinstance(value, (list, tuple)):
                return type(value)(realize_args(v) for v in value)
            if isinstance(value, dict):
                return {k: realize_args(v) for k, v in value.items()}
            return value

        realized_args = realize_args(args)
        # An operator a template owns is built as that template's kernel, so
        # the operation is named in one place and its implementation is chosen
        # from the template's candidates rather than at the call site.
        template = self._template_for(node, realized_args)
        if template is not None:
            kernel = TemplateKernel(
                self.new_name("extern"),
                node.target,
                template,
                realized_args,
                realize_args(kwargs),
                node.meta.get("val"),
                call_method=node.op == "call_method",
            )
            kernel.template_meta = self._template_meta(node, realized_args)
        else:
            kernel = ExternKernel(
                self.new_name("extern"),
                node.target,
                realized_args,
                realize_args(kwargs),
                node.meta.get("val"),
                call_method=node.op == "call_method",
            )
        self.operations.append(kernel)

        def wrap(val, path):
            if _is_tensor(val):
                layout = Layout(
                    val.device, val.dtype,
                    tuple(int(s) for s in val.shape),
                    tuple(int(s) for s in val.stride()),
                    int(val.storage_offset()) if hasattr(val, "storage_offset") else 0,
                )
                out = ExternOutput(self.new_name(), layout, kernel, path)
                self.buffers[out.name] = out
                kernel.outputs.append(out)
                return TensorBox(out)
            if isinstance(val, (list, tuple)):
                return tuple(wrap(v, path + (i,)) for i, v in enumerate(val))
            return val

        return wrap(node.meta.get("val"), ())

    # -- walk -------------------------------------------------------------
    def run(self):
        with set_ops_handler(DeferredOps()), set_graph(self):
            return self._run()

    def _run(self):
        graph = self.graph_module.graph
        env: dict[Any, Any] = {}
        placeholders = list(graph.placeholders)
        for position, node in enumerate(placeholders):
            value = self.example_inputs[position]
            if not _is_tensor(value):
                env[node] = value
                self.graph_inputs.append(None)
                continue
            layout = Layout(
                value.device, value.dtype,
                tuple(int(s) for s in value.shape),
                tuple(int(s) for s in value.stride()),
                0,
            )
            buffer = InputBuffer(f"arg{position}", layout)
            self.buffers[buffer.name] = buffer
            self.graph_inputs.append(buffer)
            env[node] = TensorBox(buffer)
            if self.device is None and value.device.is_cuda():
                self.device = value.device

        # Values are produced on demand from what the region returns, not by
        # walking the graph's node table.  A table can hold a node whose
        # arguments name values no entry in it stands for -- a region rebuilt
        # in place leaves such references behind -- and a walk that trusts the
        # table hands such an argument to a kernel body as a bare node.
        # Starting from the returned values and producing what they name makes
        # the table advisory: a value nothing returns is never produced, and
        # a value that is produced is produced once.
        produced: dict[int, Any] = {}

        def lower(value):
            if isinstance(value, (list, tuple)):
                return type(value)(lower(v) for v in value)
            if isinstance(value, dict):
                return {k: lower(v) for k, v in value.items()}
            if not (hasattr(value, "op") and hasattr(value, "users")):
                return value
            key = id(value)
            if key in produced:
                return produced[key]
            if value.op == "placeholder":
                if value not in env:
                    # A half that was partitioned out rebuilds its own
                    # placeholders, and a node that kept a reference to an
                    # earlier one names an input the half does not declare.
                    # Report the name: the caller has to hand this half a
                    # value for it, and a bare lookup failure would not say
                    # which input is missing.
                    raise NotImplementedError(
                        f"placeholder {value.name!r} is referenced but not declared"
                    )
                return env[value]
            if value.op == "get_attr":
                tensor = self.graph_module._get_attr(value.target)
                name = self.new_name("const")
                layout = Layout(tensor.device, tensor.dtype,
                                tuple(int(s) for s in tensor.shape),
                                tuple(int(s) for s in tensor.stride()), 0)
                buffer = ConstantBuffer(name, layout, tensor)
                self.buffers[name] = buffer
                self.constants[name] = tensor
                produced[key] = TensorBox(buffer)
                return produced[key]
            produced[key] = None
            args = lower(value.args)
            kwargs = lower(value.kwargs or {})
            name = target_name(value.target)
            if name == "getitem" and isinstance(args[0], (list, tuple)):
                # Indexing a result tuple is resolved here, not called out to:
                # a value that is already computed is addressed, not recomputed.
                # The spelling is matched by name because the same operation
                # reaches the graph as more than one callable.  Indexing a
                # tensor selects along an axis instead, which is a view.
                result = args[0][args[1]]
            else:
                lowering = LOWERINGS.get(name)
                if lowering is not None:
                    result = lowering(value, *args, **kwargs)
                else:
                    result = self.make_extern(value, args, kwargs)
            if isinstance(result, TensorBox):
                self.mark_reuse(result, len(value.users))
            produced[key] = result
            return result

        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                outputs = node.args[0]
                outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
                # The output node wraps its arguments, so a region that yields
                # one value still stores it in a one-element sequence.
                self.single_output = len(outputs) == 1
                for value in lower(list(outputs)):
                    if isinstance(value, TensorBox):
                        self.graph_outputs.append(self.realize_input(value))
                    else:
                        self.graph_outputs.append(value)
                continue
        return self


def _is_tensor(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "stride")


__all__ = ["GraphLowering"]
