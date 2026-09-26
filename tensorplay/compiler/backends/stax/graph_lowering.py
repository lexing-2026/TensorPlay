"""Lower a captured dispatch-level graph into loop IR buffers.

The walk visits the graph once.  Every operator either has a lowering
(which builds unmaterialized loop nests and views) or becomes a library
call on realized inputs.  A loop nest is materialized when something needs
memory: a library call, a graph output, or a value that is re-read by many
consumers or already carries many reads of its own -- recomputing such a
value in each consumer would cost more than storing it once.
"""

from __future__ import annotations

import contextlib
import collections
import itertools
import operator
import re
from typing import Any

import sympy
import tensorplay as tp
from tensorplay.utils import _pytree as pytree

from .loops import compute_required_storage_length, contiguous_strides
from .sizevars import SizeVarAllocator
from .ir import (
    Buffer,
    BaseView,
    ComputedBuffer,
    StorageBox,
    Constant,
    ConstantBuffer,
    EffectfulKernel,
    ShapeAsConstantBuffer,
    OpaqueMultiOutput,
    OpaqueObjectState,
    # The IR-level external-kernel helper, which owns putting a value in
    # memory.  Distinct from the ``ExternKernel`` below, which is the
    # schedulable node standing for a call to something written elsewhere.
    ExternKernel as IrExternKernel,
    FallbackKernel as IrFallbackKernel,
    MultiOutput,
    MultiOutputLayout,
    assign_origin_node,
    InputBuffer,
    IRNode,
    NonTensorObj,
    Layout,
    FixedLayout,
    Loops,
    Pointwise,
    Reduction,
    ReinterpretView,
    TensorBox,
    View,
)
from .loops import (
    DeferredOps,
    dtype_name,
    Value,
    V,
    substitute,
    fresh_symbols,
    record_body,
    set_graph,
    set_ops_handler,
)
from .op_lowerings import LOWERINGS, target_name, user_lowerings

#: A multi-user value is stored once it reads more than this many buffers.
REALIZE_READS_THRESHOLD = 4
#: Any value is stored once its inlined body reads more than this many.
REALIZE_ACC_READS_THRESHOLD = 8
#: ... or once its inlined body holds more operations than this.
REALIZE_OPCOUNT_THRESHOLD = 30


#: Operators a template owns, keyed by the name the graph gives them.
#:
#: An operator is in here only when nothing lowerable answers for it.  A product
#: and a product with a bias are *not* in here, because each has a lowering of
#: its own that collects the candidates and measures them; routing them here as
#: well would be two answers to one call, and which one ran would depend on which
#: had been imported first.
_TEMPLATE_OPERATORS = {
    "conv1d.default": "conv",
    "conv2d.default": "conv",
    "conv3d.default": "conv",
    "convolution.default": "conv",
    "convolution_backward.default": "convolution2d_bwd_input",
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


def _affine_coeff(expr, var):
    """The coefficient of ``var`` in an expression that is linear in it.

    A strided view reads element ``i`` of the source at a fixed distance per
    axis, so the index has to be a first-degree polynomial in each axis with a
    coefficient that does not itself mention that axis.  Anything of a higher
    degree, or a coefficient that still refers to the axis, is not a strided
    view and is declined.
    """

    try:
        poly = sympy.Poly(sympy.expand(expr), var)
    except (sympy.PolynomialError, TypeError, ValueError):
        return None
    if (poly.free_symbols - {var}) or poly.total_degree() > 1:
        return None
    return int(poly.coeff_monomial(var))


def _body_stats(loops: Loops) -> tuple[int, int]:
    body = record_body(loops)
    from .loops import iter_values

    values = iter_values(body.root)
    return len(body.loads), len(values)


#: What a graph yields before it has been asked, so that a graph which
#: yields nothing is told apart from a graph that has not been run.
_MISSING = object()

#: What a graph is allowed to yield.  A value is a place memory can be given,
#: something already known without computing it, a shape, a non-tensor, or
#: nothing at all.  A node of any other kind is one this walk has no way to
#: account for, and the cheaper place to be told is where the yield happens.
_allowed_output_types = (
    TensorBox,
    # A view is a window onto memory someone else holds rather than a buffer
    # of its own, and is put in memory by whoever reads it out.
    BaseView,
    Constant,
    type(None),
    ConstantBuffer,
    sympy.Expr,
    sympy.logic.boolalg.Boolean,
    int,
    EffectfulKernel,
    ShapeAsConstantBuffer,
    NonTensorObj,
    OpaqueMultiOutput,
    OpaqueObjectState,
)


class GraphLowering:
    def __init__(self, graph_module, example_inputs):
        self.graph_module = graph_module
        # The module whose constants a constant node reads, which is this region's
        # own while it is being lowered and the subgraph's while a subgraph's
        # nodes are being lowered into it.  A node names a constant as an
        # attribute path, and the path means nothing without the module it is
        # relative to.
        self.module = graph_module
        self.example_inputs = list(example_inputs)
        # The buffers in the order they were made, and the same ones by name.
        # Both are kept because one question asks which order things were made
        # in and the other asks what a name refers to, and either can be asked
        # often enough that answering one from the other would be wasteful.
        self.buffers: list[Any] = []
        self.operations: list[Any] = []
        self.graph_inputs: list[InputBuffer] = []
        self.constants: dict[str, Any] = {}
        # Names of the buffers whose contents are overwritten after the fact,
        # and who reads each.
        self.mutated_buffers: set[str] = set()
        self.name_to_users: dict = collections.defaultdict(list)
        self.name_to_op: dict[str, Any] = {}
        self.lists: dict[str, list] = {}
        # The same buffers by name, which is how anything that was handed a
        # name finds what the name refers to.
        self.name_to_buffer: dict[str, Any] = {}
        # What each dependency moves, kept because the same ones are asked
        # about over and over while deciding what to materialize.
        self.dep_size_hint_cache: dict = {}
        # This region's own name, which every name inside it is qualified with.
        self.name = None
        # Which devices this region computes on, and which of each, and which
        # node each device was first needed for.
        self.device_types: set = set()
        self.device_idxs: set = set()
        self.device_node_mapping: dict = {}
        self.graph_outputs: list[Any] = []
        # The shape environment of the region, and the questions asked of it.
        self.sizevars = SizeVarAllocator()
        # How many calls were laid out with their channels last because the
        # layouts were being chosen.  Read afterwards to see how much of the
        # program's traffic that choice accounted for, which is the only way to
        # tell whether it was worth making.
        self.num_channels_last_conv = 0
        # Names the region used and then gave up on.  A lookup that finds one
        # of these is a lookup of something that does not exist, which is a
        # mistake in the caller rather than a missing entry.
        self.removed_buffers: set[str] = set()
        self.inplaced_to_remove: set[str] = set()
        # Buffers whose contents are not aligned, so a kernel may not assume
        # an aligned access to one of them.
        self.unaligned_buffers: set[str] = set()
        # The values as they arrived, before anything was written over them: a
        # program that hands over a value it is finished with is recorded that
        # way, and what it handed over is this.
        self.graph_inputs_original: dict = {}
        # Values bound from the surrounding program, which are referred to by
        # name rather than copied into the generated code.
        self.torchbind_constants: dict = {}
        # Values that this region writes to, and which of the program's own
        # values those were.
        self.mutated_inputs: set[str] = set()
        self.mutated_input_idxs: list = []
        # Dependencies the program asked for beyond what the arithmetic implies:
        # ones that only say an order, and ones that must be waited for.
        self.additional_buffer_deps: dict = collections.defaultdict(list)
        self.additional_star_deps: dict = collections.defaultdict(list)
        # Buffers whose memory must not be handed to one kernel to write while
        # another reads it, and the group each belongs to.
        self.never_reuse_buffers: set[str] = set()
        self.comm_buffers: dict[str, Any] = {}
        # The last call of each kind that changes something outside its own
        # values, so that the next one of that kind can be placed after it.
        self.effectful_ops: dict = {}
        # Calls that were run through to rather than written out, and the
        # features this backend can be asked whether it has.
        self.fallback_ops: list = []
        self.backend_features: set = set()
        # The pieces of this region that are compiled on their own.
        self.used_features: set = set()
        # Whether values made only of constants are made by running the
        # operations rather than while the code is being written.
        self.use_runtime_constant_folding = False
        # Hands out a name per scratch buffer a kernel asks for.
        self.workspace_id = itertools.count()
        # The wrapper the code is emitted for, when there is one.  A kernel
        # that needs a size computed outside itself asks the wrapper to do it.
        self.wrapper_code = None
        # Which kind of wrapper, since what a kernel may ask of it depends:
        # code that is run directly is handed values, and code that is
        # compiled is handed the C++ that stands for them.
        self.cpp_wrapper = False
        self.fx_wrapper = False
        # The scheduler, once the region has one.  A name that was written in
        # place stands for the buffer that was really written, and the mapping
        # is the scheduler's to answer.
        self.scheduler = None
        self.mutation_real_name: dict = {}
        # A region whose output node holds one value returns that value, not a
        # one-element sequence, so the compiled region matches its capture.
        self.single_output = True
        self._counter = 0
        self.device = None
        # What each node has been lowered to, by the node itself.  Kept on the
        # region rather than inside the walk because a subgraph's nodes are
        # lowered into this same region, and a value produced by one of them
        # is read by a node of the region just as a value produced by the
        # region's own node is.
        self.env: dict[Any, Any] = {}
        # Produces a node's value from the values its arguments name, and is
        # installed by the walk; a caller that lowers nodes itself needs it
        # before the walk has run.
        self._lower_node = None

    def current_node(self):
        """The graph node being lowered right now, or nothing.

        A generator that writes code for a node asks about the region rather
        than being handed the node, and a few of its decisions are about the
        node: a view the user wrote should not be padded, and a call that came
        from a particular place should say so in what is generated.
        """

        return getattr(self, "_current_node", None)

    @contextlib.contextmanager
    def set_current_node(self, node):
        """Say which node is being lowered, and go back to saying none."""

        previous = self._current_node
        self._current_node = node
        try:
            yield node
        finally:
            self._current_node = previous

    # -- registry ---------------------------------------------------------
    def add_device_info(self, device) -> None:
        """Note that this region computes on a device, and which one.

        Which devices and which of each are kept apart, because a wrapper has to
        place things on each device a region touches, and one that touched two
        devices of the same kind is not the same as one that touched the same
        device twice.  The node a device was first seen on is kept as well, so
        that a message about it can say where it came from.
        """

        self.device_types.add(device.type)
        if device.index is not None:
            self.device_idxs.add(device.index)
        if V.graph.current_node and device not in self.device_node_mapping:
            self.device_node_mapping[device] = V.graph.current_node

    def qualify_name(self, name: str) -> str:
        """Put this region's own name in front of a name of a thing inside it.

        Two regions in one program would otherwise both have a buffer zero, and
        the two would be told apart by nothing.
        """

        if self.name is not None:
            return f"{self.name}_{name}"
        return name

    def register_operation(self, op):
        """Record an operation, and give it the name it is referred to by."""

        if op.operation_name is not None:
            raise AssertionError(f"Operation registered twice: {op}")
        name = self.qualify_name(f"op{len(self.operations)}")
        self.operations.append(op)
        self.name_to_op[name] = op
        op.operation_name = name
        return name

    def register_buffer(self, buffer, *, set_name: bool = False) -> str:
        """Record a buffer, and give it the name it is referred to by."""

        name = self.qualify_name(f"buf{len(self.buffers)}")
        self.buffers.append(buffer)
        self.name_to_buffer[name] = buffer
        device = buffer.get_device()
        if (
            device is not None
            and not (
                isinstance(buffer, ComputedBuffer)
                and buffer.is_zero_elements()
                and device == tp.device("cpu")
            )
        ):
            self.add_device_info(device)

        if set_name:
            buffer.name = name
        return name

    def register_operation_list(self, operation_names: list) -> str:
        name = self.qualify_name("list_" + "_".join(operation_names))
        self.lists[name] = operation_names
        return name

    def register_users_of(self, node_output) -> None:
        """Note who reads a value, so that writing over it can wait for them.

        Whoever reads a value has to run before anything overwrites it, and
        that is only knowable if the readers were recorded when the read was
        made.
        """

        def register(value):
            if isinstance(value, (list, tuple)):
                for x in value:
                    register(x)
            if isinstance(value, TensorBox):
                for read_name in value.get_read_names():
                    self.name_to_users[read_name].append(value)

        register(node_output)

    def get_output_names(self) -> list:
        """The names of the values this region hands back.

        What the program is given is not the buffers themselves but the names
        they were given, since it is those names that everything downstream
        refers to.
        """

        return [buf.get_name() for buf in self.graph_outputs]

    def get_original_buffer_name(self, buf_name: str) -> str:
        """The name this value had before anything was written over it."""

        return self.scheduler.mutation_real_name.get(buf_name, buf_name)

    def add_buffer_dependency(self, node: str, buf_name: str) -> None:
        """Say that this piece has to come after that one, and nothing more.

        Where the two touch the same memory this would be worked out; this is
        for the cases where they do not and the order is still required.
        """

        self.additional_buffer_deps[node].append(buf_name)

    def add_star_dependency(self, node: str, buf_name: str) -> None:
        """Say that this piece has to come after everything that made that value.

        Stronger than an order-only dependency: the value has to be there, not
        merely have been made by then.
        """

        self.additional_star_deps[node].append(buf_name)

    def has_feature(self, device, feature) -> bool:
        """Whether this backend can do a particular thing on a device.

        A caller asks before relying on something, and an answer of no is a
        legitimate answer rather than a missing one, since a feature is only
        present on the devices and configurations that have it.
        """

        from .codegen.common import BackendFeature

        if not isinstance(feature, BackendFeature):
            raise AssertionError(
                f"Expected BackendFeature, got {type(feature)}"
            )
        return feature in self.backend_features

    def warn_fallback(self, kernel_name: str) -> None:
        """Note that a call is being run through to rather than written out.

        Which calls those were is worth knowing afterwards: each one is a
        place where the shape of the result was worked out by running the
        operation rather than by looking at it.
        """

        self.fallback_ops.append(kernel_name)

    def mark_buffer_mutated(self, name: str) -> None:
        """Record that a buffer's contents are overwritten, and let its readers finish.

        Everyone who read the value that is about to be replaced has to have run
        first, and realizing them here is what forces that to happen before the
        write rather than after.
        """

        if not isinstance(name, str):
            raise AssertionError(f"Expected str, got {type(name)}")
        self.mutated_buffers.add(name)

        if name not in self.name_to_users:
            return

        for user in self.name_to_users[name]:
            user.realize()

    def get_dep_size_hint(self, dep, count_bytes: bool = True) -> int:
        """How much a dependency moves, guessed where it can be.

        This is a hint and nothing rests on it but a decision about whether
        materializing a value is worth it, so a dependency whose extent is not
        known counts as nothing rather than as a wrong number.  The answers are
        kept, because the same dependencies are asked about repeatedly.
        """

        if (dep, count_bytes) not in self.dep_size_hint_cache:
            res = 0
            inp = self.graph_inputs.get(dep.name)
            if isinstance(inp, NonTensorObj):
                self.dep_size_hint_cache[(dep, count_bytes)] = 0
                return 0
            try:
                if (
                    not dep.has_unbacked_symbols()
                    or self.sizevars.all_unbacked_explicitly_hinted(dep.get_numel())
                ):
                    if count_bytes:
                        res = dep.numbytes_hint()
                    else:
                        res = dep.numel_hint()
            except KeyError:
                # A dependence on something that is not in the region is not an
                # error here; it simply has no size to report.
                pass
            self.dep_size_hint_cache[(dep, count_bytes)] = res
        return self.dep_size_hint_cache[(dep, count_bytes)]

    def get_buffer(self, name):
        return self.name_to_buffer[name]

    def try_get_buffer(self, name):
        """The buffer a name refers to, or nothing if there is no such name.

        A caller asking whether something is there must not have to catch an
        error to find out, since "not there" is an answer rather than a
        mistake.
        """

        return self.name_to_buffer.get(name, None)

    def get_allocation_size(self, node):
        """The extents a buffer is allocated with.

        These are not always the extents it was described with: a buffer whose
        layout was padded to suit the machine is allocated at the padded size,
        and it is that size the allocator has to satisfy.
        """

        if isinstance(node, TensorBox):
            node = node.data
        if isinstance(node, StorageBox):
            node = node.data
        if (
            isinstance(node, ComputedBuffer)
            and node.name in self.buffer_to_padded_size
        ):
            return self.buffer_to_padded_size[node.name]
        else:
            return node.get_size()

    def get_allocation_storage_size(self, node):
        """How much memory a buffer occupies, given its extents and its strides.

        The extents alone do not say: a buffer may start partway into its
        allocation, and its elements may be spread out rather than packed.  So
        the length is settled from the extents, the stride and the offset
        together, which is the smallest length that covers every element.
        """

        layout = node.get_layout()
        size = self.get_allocation_size(node)  # consider inplace padding
        stride = layout.stride
        offset = layout.offset
        return compute_required_storage_length(size, stride, offset)

    def get_dtype(self, buffer_name: str):
        """The element type of a named buffer, as the code generators see it.

        A generator asks for a type by name because that is all it was handed,
        and the name may belong to a stored result, to something the region was
        given, or to a constant lifted out of the module.  A name that is
        really a view of another buffer is answered from the buffer it views,
        since the view does not carry a type of its own.
        """

        if buffer_name in self.constants:
            return self.constants[buffer_name].dtype
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name].get_dtype()
        for buf in self.graph_inputs:
            if buf.get_name() == buffer_name:
                return buf.get_dtype()
        m = re.match(r"(as_strided|reinterpret_tensor)\(([a-zA-Z0-9_]+),", buffer_name)
        if m:
            return self.get_dtype(m.group(1))
        raise KeyError(f"could not find {buffer_name}")

    def get_current_device_or_throw(self):
        """The device being emitted for, and a refusal when there is none.

        Code that allocates memory has to know where the memory goes, and a
        region that has not said which device it is on cannot have that
        answered for it.
        """

        if self.device is None:
            raise RuntimeError("Trying to get current device but it is not set")
        return self.device

    @contextlib.contextmanager
    def set_current_device(self, device):
        """The device a generator is writing code for, for the duration of it.

        A generator emits code without threading a device through every call,
        so the one it is currently emitting for is held here instead, and the
        previous one is put back when it is done.
        """

        previous = self.device
        self.device = device
        try:
            yield device
        finally:
            self.device = previous

    def register_computed(self, loops: Loops) -> ComputedBuffer:
        size = tuple(loops.ranges)
        layout = FixedLayout(loops.device, loops.dtype, size, contiguous_strides(size))
        buffer = ComputedBuffer(name=None, layout=layout, data=loops)
        self.register_buffer(buffer, set_name=True)
        self.register_operation(buffer)
        return buffer

    def register_welford(self, reduction: Reduction):
        """One welford loop nest, two stored results: mean and m2."""

        size = tuple(reduction.ranges)
        layout = FixedLayout(reduction.device, reduction.dtype, size, contiguous_strides(size))
        mean = ComputedBuffer(name=None, layout=layout, data=reduction)
        m2 = ComputedBuffer(
            name=None,
            layout=FixedLayout(
                reduction.device, reduction.dtype, size, contiguous_strides(size)
            ),
            data=reduction,
        )
        m2.welford_parent = mean
        m2.welford_index = 1
        mean.welford_siblings = [m2]
        for b in (mean, m2):
            self.register_buffer(b, set_name=True)
        self.register_operation(mean)
        return mean, m2

    # -- realization ------------------------------------------------------
    def realize_input(self, box: TensorBox):
        """A buffer or a strided view of one, fit for a library call.

        A value that had nowhere to live is given a buffer here, and what
        comes back is that buffer rather than the box it was held in: a
        library call and a caller both want somewhere to read from, and the
        box only says what would be computed, not where it would be.
        """

        node = IrExternKernel.realize_input(box)
        # A value that had nowhere to live is given a buffer, and what a
        # caller wants back is where it now lives rather than the box that
        # holds it: a library call and a graph output both read from memory.
        if isinstance(node, TensorBox):
            # Still held in a box, so the box is what has to be given a buffer;
            # a value walked into memory rather than read as a window arrives
            # here, and a caller reading it wants the buffer it landed in.
            node.realize()
            return self.get_buffer(node.get_name())
        inner = node.data if isinstance(node, TensorBox) else node
        if isinstance(inner, StorageBox):
            inner.realize()
            return self.get_buffer(inner.get_name())
        return node

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
        strides = []
        for var, extent in zip(index, size):
            coeff = _affine_coeff(expr, var)
            if coeff is None:
                return None
            strides.append(0 if extent == 1 else coeff)
        remaining = substitute(expr, {var: sympy.Integer(0) for var in index})
        if not getattr(remaining, "is_Integer", False):
            return None
        return ReinterpretView(
            data=self.name_to_buffer[name],
            layout=FixedLayout(
                view.get_device(),
                view.get_dtype(),
                tuple(size),
                tuple(strides),
                int(remaining),
            ),
        )

    # -- library calls ----------------------------------------------------
    def _template_for(self, node, realized_args):
        """The template that owns this operator, if any."""

        from .templates import template_for

        name = target_name(node.target)
        if template_for(_TEMPLATE_OPERATORS.get(name, "")) is None:
            return None
        return template_for(_TEMPLATE_OPERATORS[name])

    def _template_meta(self, node, realized_args, kwargs, template_name):
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
        if template_name == "conv":
            # A convolution's result depends on its geometry, so the template
            # is told the geometry rather than left to infer the extent.
            meta["stride"] = _ints(kwargs.get("stride"))
            meta["padding"] = _ints(kwargs.get("padding"))
            meta["dilation"] = _ints(kwargs.get("dilation"))
            meta["groups"] = int(kwargs.get("groups", 1))
            meta["transposed"] = bool(kwargs.get("transposed", False))
            weight = operands[1] if len(operands) > 1 else None
            if weight is not None:
                meta["kernel_size"] = tuple(int(k) for k in weight.get_size()[2:])
        if name == _LINEAR_OPERATOR:
            meta["b_transposed"] = True
        if len(operands) > 2:
            meta["bias_spec"] = (2, None)
        return meta

    @staticmethod
    def _layout_of(val):
        """Where the elements of a traced result sit."""

        return FixedLayout(
            val.device, val.dtype,
            tuple(int(s) for s in val.shape),
            tuple(int(s) for s in val.stride()),
            int(val.storage_offset()) if hasattr(val, "storage_offset") else 0,
        )

    def _make_fallback(self, node, kernel_name, realized_args, kwargs):
        """Describe a call that is run rather than written out.

        What the call is given is split the way a call written later expects
        it: the arguments that name memory are inputs, and the rest travel as
        constants beside them, so that regenerating the call puts each argument
        back in the position it was given in.  A method call names its
        operation with a string and takes the object it is called on first, so
        the receiver leads the inputs and the name is all there is to call by.
        """

        args = list(realized_args)
        call_method = node.op == "call_method"
        if call_method:
            receiver, args = args[0], args[1:]
        else:
            receiver = None
        realized_kwargs = {
            k: self.realize_input(v) if isinstance(v, (TensorBox, IRNode)) else v
            for k, v in (kwargs or {}).items()
        }
        # The arguments are flattened into one list in the order they were
        # given, and which of them name memory is recorded alongside, so that
        # the call can be put back together from the two streams it is
        # described by without anything having to remember the shape of the
        # call's own signature.
        flat, spec = pytree.tree_flatten({"args": args, "kwargs": realized_kwargs})
        is_tensor = [isinstance(a, IRNode) for a in flat]
        tensor_args = [a for a in flat if isinstance(a, IRNode)]
        other_args = [a for a in flat if not isinstance(a, IRNode)]

        def unflatten(new_tensor_args, new_non_tensor_args):
            """Put the arguments back the way the call was written."""

            rebuilt = []
            from_tensors = iter(new_tensor_args)
            from_others = iter(new_non_tensor_args)
            for flag in is_tensor:
                rebuilt.append(next(from_tensors) if flag else next(from_others))
            bound = pytree.tree_unflatten(rebuilt, spec)
            if call_method:
                return [receiver, *bound["args"]], bound["kwargs"]
            return bound["args"], bound["kwargs"]

        val = node.meta.get("val")
        outputs = val if isinstance(val, (list, tuple)) else (val,)
        tensor_outputs = [v for v in outputs if _is_tensor(v)]
        if not tensor_outputs:
            # A call whose result is not known cannot be described: there is
            # nothing to say where the result would sit, and a kernel declared
            # without a layout would be declared without a size.  Which call it
            # was is worth reporting, because a call reaching here with no result
            # is a call whose result was never worked out, and that is a question
            # about how it was traced rather than about this function.
            raise NotImplementedError(
                f"cannot run {getattr(node.target, '__name__', node.target)!r}: "
                f"the call's result is not known, so there is nowhere to put it"
            )
        if len(tensor_outputs) > 1:
            # Several results cannot all be the call itself, so the call is
            # given no result of its own and each result is a buffer naming the
            # path through the returned structure at which it sits.
            layout = MultiOutputLayout(
                device=tensor_outputs[0].device,
                size=tuple(int(s) for s in val.shape) if _is_tensor(val) else (),
            )
        else:
            layout = self._layout_of(tensor_outputs[0])

        kernel = IrFallbackKernel(
            layout=layout,
            kernel=node.target,
            tensor_args=tuple(tensor_args),
            nontensor_args=tuple(other_args),
            unflatten_args=unflatten,
            kwargs=realized_kwargs,
        )
        kernel.origin_node = node
        self.operations.append(kernel)
        return kernel

    def _wrap_fallback(self, node, kernel):
        """The values a described call produced, as things a body can read.

        A call that produced one thing is that thing's own buffer, so there is
        nothing to name separately; a call that produced several has one
        buffer per result, each saying where among the results it sits.
        """

        val = node.meta.get("val")

        def wrap(item, path):
            if _is_tensor(item):
                if not kernel.outputs and len(kernel.get_outputs()) == 1:
                    # Boxed the way every value is, so that what holds this
                    # result is a place memory can be given.
                    return TensorBox.create(kernel)
                out = MultiOutput(self._layout_of(item), kernel, path)
                out.origin_node = node
                kernel.outputs.append(out)
                return TensorBox.create(out)
            if isinstance(item, (list, tuple)):
                return tuple(wrap(v, path + (i,)) for i, v in enumerate(item))
            return item

        return wrap(val, ())

    def make_extern(self, node, args, kwargs):
        def realize_args(value):
            # Anything that names memory is put in memory before the call sees
            # it, whether it arrived boxed or as a bare node.
            if isinstance(value, (TensorBox, IRNode)):
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
        kernel_name = self.qualify_name(f"kernel{len(self.operations)}")
        kernel = self._make_fallback(node, kernel_name, realized_args, kwargs)
        if template is not None:
            # How the call is carried out, as opposed to what it is: the
            # template owns the candidates and the choice between them.
            kernel.template = template
            kernel.config = None
            kernel.template_meta = self._template_meta(
                node, realized_args, realize_args(kwargs), template.name
            )
        return self._wrap_fallback(node, kernel)

    # -- walk -------------------------------------------------------------
    def run(self):
        with set_ops_handler(DeferredOps()), set_graph(self):
            return self._run()

    def finalize(self) -> None:
        """Settle every buffer's layout, now that the region is all known.

        A buffer's layout may still change while the region is being lowered,
        because a later operation can still say what shape it wants.  Once the
        region is complete nothing can, so each one is decided here: what the
        shape is and in what order the elements lie.  Deciding it later, at the
        point a kernel first reads the buffer, is what makes a layout change
        visible to code that has already been written against the old one.
        """

        for buf in self.buffers:
            buf.decide_layout()

    def _run(self):
        graph = self.graph_module.graph
        env = self.env
        placeholders = list(graph.placeholders)
        for position, node in enumerate(placeholders):
            value = self.example_inputs[position]
            if not _is_tensor(value):
                env[node] = value
                self.graph_inputs.append(None)
                continue
            layout = FixedLayout(
                value.device, value.dtype,
                tuple(int(s) for s in value.shape),
                tuple(int(s) for s in value.stride()),
                0,
            )
            buffer = InputBuffer(name=f"arg{position}", layout=layout)
            self.name_to_buffer[buffer.name] = buffer
            self.buffers.append(buffer)
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
        result = self._run_outputs(graph)
        self.finalize()
        return result

    def lower_node(self, value):
        """A value, or a node naming one, as the value it stands for.

        A node is lowered the first time it is asked for rather than by
        walking the node table, so a node nothing reads costs nothing and a
        node several things read is lowered once.  A value that is not a node
        is already what it stands for and is returned as it is, which is what
        makes a list of arguments lowerable by the same walk as a node's.
        """

        if isinstance(value, (list, tuple)):
            return type(value)(self.lower_node(v) for v in value)
        if isinstance(value, dict):
            return {k: self.lower_node(v) for k, v in value.items()}
        if not (hasattr(value, "op") and hasattr(value, "users")):
            return value
        key = id(value)
        if key in self.produced:
            return self.produced[key]
        if value.op == "placeholder":
            if value not in self.env:
                # A half that was partitioned out rebuilds its own
                # placeholders, and a node that kept a reference to an
                # earlier one names an input the half does not declare.
                # Report the name: the caller has to hand this half a
                # value for it, and a bare lookup failure would not say
                # which input is missing.
                raise NotImplementedError(
                    f"placeholder {value.name!r} is referenced but not declared"
                )
            return self.env[value]
        if value.op == "get_attr":
            tensor = (self.module or self.graph_module)._get_attr(value.target)
            name = self.allocate_non_dup_const_name(None, tensor)
            layout = FixedLayout(tensor.device, tensor.dtype,
                            tuple(int(s) for s in tensor.shape),
                            tuple(int(s) for s in tensor.stride()), 0)
            buffer = ConstantBuffer(name=name, layout=layout)
            self.name_to_buffer[name] = buffer
            self.buffers.append(buffer)
            self.constants[name] = tensor
            self.produced[key] = TensorBox(buffer)
            return self.produced[key]
        self.produced[key] = None
        args = self.lower_node(value.args)
        kwargs = self.lower_node(value.kwargs or {})
        name = target_name(value.target)
        if name == "getitem" and isinstance(args[0], (list, tuple)):
            # Indexing a result tuple is resolved here, not called out to:
            # a value that is already computed is addressed, not recomputed.
            # The spelling is matched by name because the same operation
            # reaches the graph as more than one callable.  Indexing a
            # tensor selects along an axis instead, which is a view.
            result = args[0][args[1]]
        else:
            # A lowering the program wrote for this operation itself comes
            # before the built-in one for the name it happens to be called
            # under, and the node names both, so both are asked before one
            # is used.  Keyed by the node as well as its target because the
            # two are what a program registers under and what it looks up
            # by, and which one is present is not something to guess.
            lowering = (
                user_lowerings.get(value)
                or user_lowerings.get(value.target)
                or LOWERINGS.get(name)
            )
            if lowering is not None:
                result = lowering(value, *args, **kwargs)
            else:
                result = self.make_extern(value, args, kwargs)
        if isinstance(result, TensorBox):
            # A value several consumers read is stored rather than
            # recomputed per consumer, but only when the body is worth
            # storing: cheap index arithmetic stays inline so a shared
            # constant does not inflate every downstream read count.  The
            # decision itself belongs to the box, which is the only thing
            # that knows what it is holding and how it got its reads.
            storage = result.data
            while not isinstance(storage, StorageBox) and isinstance(
                storage, (View, TensorBox)
            ):
                storage = storage.data
            if isinstance(storage, StorageBox):
                storage.mark_reuse(len(value.users))
        # Which node of the graph this value was made by, so that a report
        # about it can name where it came from.
        assign_origin_node(result, value)
        self.produced[key] = result
        return result

    def _install_lowering(self):
        """Publish the way to lower a node, and the tables it records into.

        Both are installed rather than used directly so that a caller lowering a
        node of its own goes through the same table the region's own walk does:
        two tables would be two answers to which value a node stands for.
        """

        if not hasattr(self, "produced"):
            self.produced = {}
        self._lower_node = self.lower_node

    def _run_outputs(self, graph):
        self._install_lowering()
        for node in graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                outputs = node.args[0]
                outputs = outputs if isinstance(outputs, (list, tuple)) else (outputs,)
                # The output node wraps its arguments, so a region that yields
                # one value still stores it in a one-element sequence.
                self.single_output = len(outputs) == 1
                for value in self.lower_node(list(outputs)):
                    # What an output is allowed to be.  A value is a place
                    # memory can be given, something already known without
                    # computing it, a non-tensor, or a shape; anything else is
                    # a node type this walk cannot account for, and being told
                    # here is far cheaper than being told much later by a
                    # kernel that cannot read what it was handed.
                    if not isinstance(value, _allowed_output_types):
                        raise AssertionError(
                            f"Unexpected output types: {[type(value)]}, "
                            f"full result: {value}"
                        )
                    # A caller reads an output out of memory, so however the
                    # value was arrived at -- still held in a box, or already a
                    # bare view -- it is put in memory before it is recorded.
                    if isinstance(value, IRNode):
                        self.graph_outputs.append(self.realize_input(value))
                    else:
                        self.graph_outputs.append(value)
                continue
        return self

    # -- one node ---------------------------------------------------------
    def fetch_args_kwargs_from_env(self, node):
        """A node's arguments, with each value that names a node replaced by it.

        A node's arguments are written as the nodes they came from, so they have
        to be read before the node can be lowered: what a lowering wants is the
        value, and what a graph holds is the name of the value.
        """

        if self._lower_node is None:
            self._install_lowering()
        args = self.lower_node(node.args)
        kwargs = self.lower_node(node.kwargs or {})
        return args, kwargs

    def run_node(self, node):
        """The value one node stands for, lowering it if it has not been.

        Lowering on first use rather than by walking the node table means a node
        that nothing reads is never lowered, and a node that several things read
        is lowered once, which is what makes a subgraph's nodes cost what they
        cost rather than what the table says they might.
        """

        if node in self.env:
            return self.env[node]
        if self._lower_node is None:
            # a node may be lowered before the region's own walk has run, which
            # is what a subgraph inlined into a region that has not been walked
            # yet looks like; the way to lower it does not depend on the walk
            self._install_lowering()
        # The node being lowered right now, restored afterwards, because lowering
        # one node may lower another and the one that asked is the one a failure
        # has to name.
        saved = self.current_node
        try:
            self._current_node = node
            value = self._lower_node(node)
            self.env[node] = value
            return value
        finally:
            self._current_node = saved

    def process_subgraph_nodes(self, graph_module, args):
        """The value a subgraph yields, with its nodes lowered into this region.

        A subgraph is a graph in its own right, and lowering one means running it
        here: each of its nodes is lowered by this region's own lowerings and
        becomes a node of this region, so that what the subgraph computes can be
        fused with what surrounds it rather than being a separate program with
        its own boundary.

        ``args`` holds one value per placeholder.  Placeholders are counted on
        their own rather than indexed by a node's position among all nodes,
        because a graph need not begin with its placeholders -- a decomposition
        that runs across the subgraph can leave operations between them -- and
        indexing by position would then read past the end of ``args``.
        """

        from ....graph import Interpreter as _Interpreter

        output = _MISSING
        placeholder_idx = 0
        for node in graph_module.graph.nodes:
            if node.op == "placeholder":
                if node in self.env:
                    raise AssertionError("expected: node not in env")
                self.env[node] = args[placeholder_idx]
                placeholder_idx += 1
                continue
            if node.op == "output":
                output_args, kwargs = self.fetch_args_kwargs_from_env(node)
                output = _Interpreter.output(self, node, output_args, kwargs)
                continue
            if node in self.env:
                raise AssertionError("expected: node not in env")
            self.env[node] = self.run_node(node)

        if output is _MISSING:
            raise RuntimeError("No output node found in graph")
        return output

    def allocate_non_dup_const_name(self, name, data) -> str:
        """A name for a constant that no other constant is already using.

        Two constants with the same name would be one constant as far as
        anything reading by name can tell, so a name already taken gets
        another one.
        """

        if name is None:
            name = f"constant_{len(self.constants)}"
        base_name = name
        counter = 0
        while name in self.constants:
            counter += 1
            name = f"{base_name}_{counter}"
        return name

    def init_wrapper_code(
        self,
        is_subgraph: bool = False,
        subgraph_name: str | None = None,
        parent_wrapper_code=None,
        partition_signatures=None,
    ) -> None:
        """Prepare whatever the code for this region is written into.

        A piece compiled on its own is written into a function of its own rather
        than into the code around it, and what it needs is decided here: that it
        is a piece, what it is called, and the code it will be called from.
        """

        if not is_subgraph:
            self.wrapper_code = None
            return
        self.wrapper_code = None

    def make_subgraph(self, gm, example_inputs, subgraph_name: str):
        """A piece of this region, compiled on its own and called from here.

        Everything the piece shares with the region it came from is passed
        along, so that what is compiled is the piece itself and nothing else.
        Its name is qualified by this region's, so that two regions that each
        contain a piece by the same name do not collide.
        """

        return SubgraphLowering(
            parent=self,
            name=f"{self.name}_{subgraph_name}" if self.name else subgraph_name,
            gm=gm,
            example_inputs=example_inputs,
        )


def _ints(value) -> tuple:
    """An operator argument that may be one int or a sequence of them."""

    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)



def _is_tensor(value) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "stride")


__all__ = ["GraphLowering", "SubgraphLowering"]


class SubgraphLowering(GraphLowering):
    """A piece of a region, compiled on its own and called from the region.

    What this adds is only that the piece knows it is a piece: it is written
    into a function of its own rather than into the code around it, and the
    constants it was given belong to the region it came from, so they are
    recorded there rather than here.
    """

    def __init__(self, parent, *args, **kwargs) -> None:
        self.parent = parent
        super().__init__(*args, **kwargs)
        # Which inputs may be given over to be written into is stated in terms
        # of the order the surrounding region gave its values in, not this
        # piece's own order, so it is not this piece's to answer.
        self.bw_donated_idxs = None

    def allocate_non_dup_const_name(self, name, data) -> str:
        name = super().allocate_non_dup_const_name(name, data)
        # The code written for this piece shares the surrounding region's
        # module, and the values it was given as constants are attached to that
        # module from the outermost region's record of them.  Recording it here
        # alone would leave it missing there.
        root = self.parent
        while isinstance(root, SubgraphLowering):
            root = root.parent
        root.constants[name] = data
        return name

    def init_wrapper_code(
        self,
        is_subgraph: bool = False,
        subgraph_name: str | None = None,
        parent_wrapper_code=None,
        partition_signatures=None,
    ) -> None:
        super().init_wrapper_code(
            is_subgraph=True,
            subgraph_name=self.name,
            parent_wrapper_code=self.parent.wrapper_code,
        )
