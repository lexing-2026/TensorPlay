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
    ExternKernel,
    ExternOutput,
    InputBuffer,
    Layout,
    Loops,
    Pointwise,
    Reduction,
    ReinterpretView,
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
    def make_extern(self, node, args, kwargs):
        def realize_args(value):
            if isinstance(value, TensorBox):
                return self.realize_input(value)
            if isinstance(value, (list, tuple)):
                return type(value)(realize_args(v) for v in value)
            if isinstance(value, dict):
                return {k: realize_args(v) for k, v in value.items()}
            return value

        kernel = ExternKernel(self.new_name("extern"), node.target, realize_args(args), realize_args(kwargs), node.meta.get("val"))
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

        def resolve(value):
            if hasattr(value, "op") and hasattr(value, "users") and value in env:
                return env[value]
            if isinstance(value, (list, tuple)):
                return type(value)(resolve(v) for v in value)
            if isinstance(value, dict):
                return {k: resolve(v) for k, v in value.items()}
            return value

        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                outputs = node.args[0]
                outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
                for value in resolve(list(outputs)):
                    if isinstance(value, TensorBox):
                        self.graph_outputs.append(self.realize_input(value))
                    else:
                        self.graph_outputs.append(value)
                continue
            if node.op == "get_attr":
                tensor = self.graph_module._get_attr(node.target)
                name = self.new_name("const")
                layout = Layout(tensor.device, tensor.dtype, tuple(int(s) for s in tensor.shape), tuple(int(s) for s in tensor.stride()), 0)
                buffer = ConstantBuffer(name, layout, tensor)
                self.buffers[name] = buffer
                self.constants[name] = tensor
                env[node] = TensorBox(buffer)
                continue
            if not node.users:
                # Functional graph: an unread result has no effect.
                continue
            args = resolve(node.args)
            kwargs = resolve(node.kwargs or {})
            if node.target is operator.getitem:
                env[node] = args[0][args[1]]
                continue
            lowering = LOWERINGS.get(target_name(node.target))
            if lowering is not None:
                result = lowering(node, *args, **kwargs)
            else:
                result = self.make_extern(node, args, kwargs)
            if isinstance(result, TensorBox):
                self.mark_reuse(result, len(node.users))
            env[node] = result
        return self


def _is_tensor(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "stride")


__all__ = ["GraphLowering"]
