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
import functools
import itertools
import logging
import operator
import os
import re
from collections import defaultdict
from typing import Any

import sympy
import tensorplay as tp
from ....graph.experimental.sympy_functions import OrderedSet
from tensorplay.utils import _pytree as pytree

from . import config, ir
from .fx_utils import count_flops_fx
from .loops import compute_required_storage_length, contiguous_strides
from .utils import (
    gather_origins,
    get_sympy_Expr_dtype,
    has_free_symbols,
    normalize_name,
    SUPPORTED_MKLDNN_DEVICES,
    ValueWithLineMap,
)
from ....graph.interpreter import Interpreter
from .codegen.common import FileBackedGraphModule, get_device_op_overrides
from .codegen.cpp_wrapper import CppWrapperCode, CppWrapperModule
from .sizevars import SizeVarAllocator
from .virtualized import V
from .ir import (
    get_device_type,
    Buffer,
    BaseView,
    ComputedBuffer,
    StorageBox,
    Subgraph,
    Constant,
    ConstantBuffer,
    EffectfulKernel,
    NoneAsConstantBuffer,
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
    substitute,
    fresh_symbols,
    record_body,
    set_current_node,
    set_graph,
    set_ops_handler,
)
from .op_lowerings import (
    fallback_node_due_to_unsupported_type,
    find_lowering,
    load_lowering_modules,
    target_name,
    user_lowerings,
)

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


def _device_type_of(x):
    """The type of device a value, a node or a device is on, or None."""

    get_device = getattr(x, "get_device", None)
    if get_device is not None:
        return _device_type_of(get_device())
    if isinstance(x, tp.device):
        return x.type
    if isinstance(x, str):
        return x
    return None


class _IndexCapture:
    """Ops handler that records the single load a view resolves to."""

    def __init__(self):
        #: Why graphs stopped being captured into device graphs, when they
        #: did.  Named rather than a bare flag because what a caller wants to
        #: know is which of the reasons it was, and the reasons differ in
        #: whether anything can be done about them.
        self.disable_cudagraphs_reason: str | None = None
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


class _ModuleOfSteps:
    """Several built steps, entered as one.

    A region may print as more than one kernel, and a caller that asks the region
    for one artifact should not have to know how many kernels that was.  So the
    steps are held together and the result of the region is the result of the
    step that produced it, which is the last one to have run.
    """

    def __init__(self, launches) -> None:
        self._launches = list(launches)
        self.key = "-".join(getattr(l, "key", "") for l in self._launches)
        self.__file__ = getattr(self._launches[-1], "__file__", None)

    def call(self, args: list) -> Any:
        """Run every step over the same arguments, and hand back what came out."""

        result = None
        for launch in self._launches:
            result = launch.call(args)
        return result

    def __call__(self, args: list) -> Any:
        return self.call(args)


#: Operators whose inputs are wanted dense, so padding one of them buys
#: nothing and costs a copy.
_DISLIKE_PADDING = frozenset(
    {
        "convolution",
        "convolution_backward",
        "_scaled_mm",
        "_scaled_mm_v2",
    }
)

#: Operators that produce a fresh buffer, where padding the result rather than
#: the input is what helps.
_LIKE_PADDING = frozenset(
    {
        "var_mean",
        "sum",
        "mean",
        "prod",
        "any",
        "amin",
        "amax",
        "min",
        "max",
        "argmin",
        "argmax",
        "scatter_reduce",
    }
)


def _op_packet_name(node) -> str | None:
    """An operation's name without the form it was called in, if it has one."""

    target = getattr(node, "target", None)
    name = getattr(target, "__name__", None)
    if not isinstance(name, str):
        return None
    return name.split(".")[0]


def may_get_constant_buffer_dtype(constant_buffer) -> Any | None:
    """The element type a shape held in a buffer would have, if it has one.

    A shape is a whole number, so a buffer holding one holds whole numbers; a
    value that is a number but not a whole one is a real number.  What comes
    back is nothing when the expression is a number of some other kind, because
    there is then no type to say.
    """

    import sympy

    if not isinstance(
        constant_buffer, (sympy.Symbol, sympy.Expr, sympy.core.numbers.Integer)
    ):
        raise AssertionError(
            "may_get_constant_buffer_dtype only supports a symbol, an "
            "expression or a whole number"
        )
    if isinstance(constant_buffer, sympy.core.numbers.Integer):
        return tp.int64

    if isinstance(constant_buffer, sympy.Expr):
        return get_sympy_Expr_dtype(constant_buffer)

    if constant_buffer.is_integer:
        return tp.int64
    if constant_buffer.is_float:
        return tp.float32
    return None


def getattr_recursive(obj, target: str):
    """What a dotted path names on a module, saying where the path stops.

    A node names a constant by the path at which the module holds it, and the
    path is several attributes deep, so reading it is walking.  When the walk
    cannot go on, saying how far it got turns a question about a name into a
    question about a prefix that can be looked at.
    """

    target_atoms = target.split(".")
    attr_itr = obj
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(
                f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
            )
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def get_user_visible_output_strides(g) -> dict:
    """The strides each output a caller reads is recorded as having had.

    A graph says which of its results a caller reads and what strides those
    had before anything was computed for them, and those strides are what a
    compiled result is measured against: a result that came back with
    different strides is a different result, even where the elements are the
    same.
    """

    ret: dict = {}
    output_nodes = g.find_nodes(op="output")
    if not output_nodes:
        return ret
    output_node = output_nodes[0]
    if "user_visible_output_idxs" not in output_node.meta:
        return ret

    if not hasattr(output_node.args[0], "op"):
        output_node_args = output_node.args[0]
    else:
        output_node_args = output_node.args

    for idx, node in enumerate(output_node_args):
        if idx in output_node.meta["user_visible_output_idxs"]:
            ret[node] = output_node.meta["original_output_strides"][idx]
    return ret


def extend_user_visible_output_strides(user_visible_outputs: dict) -> dict:
    """The outputs a caller reads, plus the views leading up to them.

    A caller reads a result, not the chain of windows that produced it, but
    whether the chain may be padded is decided per node, so the nodes on the
    way there are marked as read too.
    """

    result: dict = {**user_visible_outputs}
    queue = [*result.keys()]
    visited = set(queue)
    while queue:
        current = queue.pop()
        if _op_packet_name(current) in _VIEW_PACKETS and current.args:
            base = current.args[0]
            if hasattr(base, "op") and base not in visited:
                result.setdefault(base, None)
                visited.add(base)
                queue.append(base)
    return result


#: Operations that hand back a window onto memory rather than a buffer.
_VIEW_PACKETS = frozenset(
    {
        "view",
        "reshape",
        "permute",
        "transpose",
        "expand",
        "squeeze",
        "unsqueeze",
        "slice",
        "select",
        "flatten",
        "unflatten",
        "t",
    }
)


def mark_nodes_dislike_padding(g, user_visible_output_strides: dict) -> None:
    """Say which nodes must not have their inputs padded.

    A convolution and its backward want their input dense, so padding one costs
    a copy and gains nothing, while padding usually helps a reduction.  Which
    nodes those are is found by walking the graph backwards: a node that dislikes
    padding passes it to everything it reads, except to a reduction, which is
    the thing padding is for.
    """

    if not config.comprehensive_padding:
        return

    extended = extend_user_visible_output_strides(user_visible_output_strides)
    for cur in reversed(g.nodes):
        name = _op_packet_name(cur)
        if name in _DISLIKE_PADDING:
            cur.meta["dislike_padding"] = True
        if cur.meta.get("dislike_padding", False):
            for prior in cur.all_input_nodes:
                prior_name = _op_packet_name(prior)
                if prior_name is None:
                    continue
                if prior_name not in _LIKE_PADDING:
                    prior.meta["dislike_padding"] = True
        # Only a node someone reads is decided this way.  A reduction writes a
        # fresh buffer whose strides are already settled, so marking it would
        # stop its inputs being padded for a reason that does not apply to it.
        if not config.pad_outputs and cur in extended and name not in _LIKE_PADDING:
            cur.meta["dislike_padding"] = True


log = logging.getLogger(__name__)


# A region records a convolution under the name it was called by -- the
# rank-specific forward calls and the gradient calls a backward region is
# made of -- as well as under the general one.
_CONVOLUTION_TARGET_PREFIXES = (
    "tp.convolution.",
    "tp.conv1d.",
    "tp.conv2d.",
    "tp.conv3d.",
    "tp.conv1d_grad_input.",
    "tp.conv1d_grad_weight.",
    "tp.conv2d_grad_input.",
    "tp.conv2d_grad_weight.",
    "tp.conv3d_grad_input.",
    "tp.conv3d_grad_weight.",
)


# Calls run by the framework whose result keeps the order of a channels-last
# operand, so they are handed their activations in that order.
_LAYOUT_KEEPING_FALLBACKS = frozenset({"group_norm", "group_norm_backward"})


def _is_convolution_node(node) -> bool:
    """Whether a node is a convolution or one of a convolution's gradients."""
    if node.op != "call_function":
        return False
    if node.target is tp.ops.tp.convolution.default:
        return True
    return str(node.target).startswith(_CONVOLUTION_TARGET_PREFIXES)


def _node_device_type(node) -> str | None:
    """The device kind of the value a node stands for, when it is known."""
    val = node.meta.get("val") if hasattr(node, "meta") else None
    device = getattr(val, "device", None)
    return getattr(device, "type", None)


def is_mkldnn_conv(node) -> bool:
    # When mkldnn_fusion is enabled, conv will be replaced by the lowering pattern function.
    # See _register_unary_fusion_lowering in tp/compiler/backends/stax/fx_passes/mkldnn_fusion.py.
    if (
        getattr(tp.ops, "mkldnn", None) is not None
        and getattr(tp.ops.mkldnn, "_convolution_pointwise", None) is not None
        and isinstance(node.target, functools.partial)
        and len(node.target.args) > 0
        and hasattr(node.target.args[0], "targets")
    ):
        for target in node.target.args[0].targets:
            if target.fns[0] in [
                tp.ops.mkldnn._convolution_pointwise.default,
                tp.ops.mkldnn._convolution_pointwise.binary,
                tp.ops.mkldnn._convolution_pointwise_.binary,
            ]:
                return True

    return False


class GraphLowering(Interpreter):
    def __init__(
        self,
        graph_module,
        example_inputs,
        *,
        shape_env=None,
        cpp_wrapper: bool = False,
        aot_mode: Any = None,
        extern_node_serializer: Any = None,
        is_inference: bool = False,
        is_backward: bool = False,
        layout_opt: bool | None = None,
        name: str | None = None,
    ):
        # The walk this region is read by is the ordinary one: a node at a
        # time, each handed to the method named by what it is.  Everything the
        # region's own values need from the walk -- the graph, the module the
        # constants are read out of, where the values so far are kept -- is
        # therefore set up before anything of the region's own is recorded,
        # since a value recorded against a walk that is not this one would be
        # recorded against nothing.
        load_lowering_modules()
        super().__init__(graph_module)
        self.name = "GraphLowering" if name is None else name
        #: The operations a node in the region named is computed by the
        #: framework rather than by a kernel, recorded by the name it is called
        #: under so that a later step can ask whether a given name is one.
        #: Which input has been reached, counting from zero.  What a donated
        #: input is decided by where it falls, so the count is kept rather
        #: than recomputed from the node table each time it is asked for.
        #: The node being lowered right now, or nothing.  A generator that
        #: writes code for a node asks about the region rather than being
        #: handed the node, and a few of its decisions are about the node: a
        #: view the program wrote should not be padded, and a call that came
        #: from a particular place should say so in what is generated.
        self.current_node: Any = None
        self.placeholder_idx = 0
        self.removed_operations: OrderedSet = OrderedSet()
        #: Which input a shape expression was read from, so a wrapper can bind
        #: that expression to the element it was read out of.  Recorded by
        #: expression rather than by input, because the same expression read
        #: from two inputs needs only one of them bound.
        self.symbolic_input_sources: dict = {}
        # Which shape the values are described against, kept rather than taken
        # from whichever graph happens to be current: a region compiled on its
        # own is asked about shapes that the region it came from already knows,
        # and asking the current graph would answer for that one instead.
        self.shape_env = shape_env
        # Whether the region is being compiled to stand on its own rather than to
        # be part of something larger, which decides what it may assume about
        # what surrounds it.
        self.aot_mode = aot_mode
        self.extern_node_serializer = extern_node_serializer
        self.is_inference = is_inference
        self.is_backward = is_backward
        # The name of each value the region is handed, in the order it is handed
        # them.  A caller that compiled symbols knows them by name and a compiled
        # artifact is called with them by position, so the two have to be able to
        # say which is which.
        self.graph_input_names: list[str] = []
        # Which inputs must be checked for the alignment a wide read needs.
        # Named by position rather than left to every caller, because a value
        # that is read several at a time has to be aligned whichever of the
        # ways of reading it is used, and which ways are used is not known when
        # the inputs are first seen.
        self.inputs_to_check: list[int] = []
        # A value a kernel was handed that the kernel reads as a view of
        # something else.  Kept beside the graph rather than passed to the
        # kernel, because the kernel is written against the value and the view
        # is how that value was reached -- and a view that has been flattened
        # into the value is a view of the wrong bytes.
        self._cutedsl_capture_nodes: dict = {}
        #: The graphs read off the program, by the name they were read under.
        #: Kept so that the same graph read twice is lowered once -- two copies
        #: would be two regions computing the same thing, and nothing else in
        #: the region would know they were the same.
        self.seen_subgraphs: dict = {}
        self.graph_module = graph_module
        # The module a constant is read out of.  A node names a constant as an
        # attribute path, and that path means nothing without the module it is
        # relative to -- which is this region's own while this region is being
        # read, and the subgraph's while a subgraph's nodes are being read into
        # this one.
        self.module = graph_module
        # What strides each output a caller reads is recorded as having had, and
        # which nodes must not have their inputs padded because of it.
        self.user_visible_output_strides = get_user_visible_output_strides(
            graph_module.graph
        )
        mark_nodes_dislike_padding(
            graph_module.graph, self.user_visible_output_strides
        )
        self.user_visible_nodes = self._user_visible_nodes(graph_module.graph)
        # The module whose constants a constant node reads, which is this region's
        # own while it is being lowered and the subgraph's while a subgraph's
        # nodes are being lowered into it.  A node names a constant as an
        # attribute path, and the path means nothing without the module it is
        # relative to.
        self.example_inputs = list(example_inputs)
        # The buffers in the order they were made, and the same ones by name.
        # Both are kept because one question asks which order things were made
        # in and the other asks what a name refers to, and either can be asked
        # often enough that answering one from the other would be wasteful.
        self.buffers: list[Any] = []
        self.operations: list[Any] = []
        self.graph_inputs: dict = {}
        #: Where in its storage each tensor input starts.  The pointer a region
        #: is handed already points there, so an offset a call names from the
        #: start of the storage has this taken off.
        self.graph_input_storage_offsets: dict[str, Any] = {}
        self.constants: dict[str, Any] = {}
        self._embedded_tensor_constants: dict[int, TensorBox] = {}
        #: What each constant is, described without reading it -- so that two
        #: compilations can be compared without holding the values, and so that
        #: a constant can be recognised as one already compiled.
        self.constant_reprs: dict[str, str] = {}
        #: The name the program gave each constant, by the name it was given
        #: here.  Kept so that a report about a constant can name it the way
        #: the program named it rather than the way this compiler spells it.
        self.allocated_constant_name: dict[str, str] = {}
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
        # Set from the constructor argument above when one was given.
        # Which devices this region computes on, and which of each, and which
        # node each device was first needed for.
        self.device_types: set = set()
        self.device_idxs: set = set()
        self.device_node_mapping: dict = {}
        self.graph_outputs: list[Any] = []
        # The shape environment of the region, and the questions asked of it.
        self.sizevars = SizeVarAllocator(shape_env)
        # Whether the layouts here may be chosen rather than taken as they are
        # given, which is what makes a call of the kind that gains from a
        # particular layout be laid out for it.
        self.layout_opt = (
            layout_opt
            if layout_opt is not None
            else self.decide_layout_opt(
                graph_module, is_inference=is_inference
            )
        )
        # How many calls were laid out with their channels last because the
        # layouts were being chosen.  Read afterwards to see how much of the
        # program's traffic that choice accounted for, which is the only way to
        # tell whether it was worth making.
        self.num_channels_last_conv = 0
        # The nodes whose values are held channels-last, worked out from the
        # graph before any node is lowered.  A node is on the list when a
        # convolution reads or writes it, or when one of its users is on the
        # list, so that a conv -> norm -> activation -> conv chain keeps one
        # layout throughout instead of copying after every norm and again
        # before the next convolution.
        self.nodes_prefer_channels_last = (
            self._find_nodes_prefer_channels_last()
            if self.layout_opt
            else OrderedSet()
        )
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
        #: The graph inputs that stand in for a value rather than holding one.
        #: Recorded by name as they are found, because whether one is such is
        #: asked again later and the answer is about the graph, not the call.
        self.zero_dim_cpu_tensor_list: OrderedSet[str] = OrderedSet()
        #: A buffer's extents after it was padded, where something padded it.
        #: A buffer is described by its own extents unless it was made bigger
        #: than what it holds, and then the size asked about is the bigger one.
        self.buffer_to_padded_size: dict[str, list[int]] = {}
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
        # The pieces of this region that are compiled on their own.
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
        self.cpp_wrapper = cpp_wrapper
        self.fx_wrapper = False
        # The scheduler, once the region has one.  A name that was written in
        # place stands for the buffer that was really written, and the mapping
        # is the scheduler's to answer.
        self.scheduler = None
        self.no_fuse_buffer_names: OrderedSet[str] = OrderedSet()
        #: The names every kernel printed under, kept as they are printed so
        #: that a kernel can be found by the name it actually has.
        self.all_codegen_kernel_names: OrderedSet[str] = OrderedSet()
        #: Values carried alongside the region that the printed program is
        #: handed as module attributes, so that the printed program does not
        #: have to be handed them again by whoever called it.
        self.torchbind_constants: dict[str, Any] = {}
        self.opaque_value_type_classes: dict[str, Any] = {}
        self.mutation_real_name: dict = {}
        # A region whose output node holds one value returns that value, not a
        # one-element sequence, so the compiled region matches its capture.
        self.single_output = True
        self._counter = 0
        self.current_device = None
        # What each node has been lowered to, by the node itself.  Kept on the
        # region rather than inside the walk because a subgraph's nodes are
        # lowered into this same region, and a value produced by one of them
        # is read by a node of the region just as a value produced by the
        # region's own node is.
        self.env: dict[Any, Any] = {}
        # Produces a node's value from the values its arguments name, and is
        # installed by the walk; a caller that lowers nodes itself needs it
        # before the walk has run.
        # The node being lowered right now, published for the duration of a
        # lowering call so that anything below it can ask which one without it
        # being passed down through every call that might want to.
        # Whether the walk has run, so that a second request for a built form
        # does not walk an already settled region again.
        self._walked = False
    def _graph_input_named(self, name: str):
        """The input a name belongs to, if this graph has one by that name.

        The inputs are kept in the order they were declared rather than by
        name, so a name is looked for by walking them; a graph has a handful of
        inputs, and a name that is not one of them was never declared.
        """

        for buf in self.graph_inputs_original.values():
            if buf is not None and getattr(buf, "name", None) == name:
                return buf
        return None

    def try_get_buffer(self, buffer_name: str):
        """What a name refers to, or nothing if it refers to nothing.

        Three kinds of thing can be named here and all three are asked: a
        buffer this region made, a value it was given, and a constant lifted
        out of the program.  A constant is described on the spot rather than
        looked up, because the value it stands for is what describes it and
        asking for that value is cheaper than keeping a second description.
        """

        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name]
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name]
        if buffer_name in self.constants:
            data = self.constants[buffer_name]
            return ConstantBuffer(
                name=buffer_name,
                layout=FixedLayout(
                    data.device, data.dtype, *self.static_sizes_strides(data)
                ),
            )
        return None

    def get_buffer(self, buffer_name: str):
        """What a name refers to, and a refusal when it refers to nothing.

        A caller holding a name is about to read through it, so a name that
        names nothing is reported here rather than answered with something
        that would fail further along.
        """

        buf = self.try_get_buffer(buffer_name)
        if buf is not None:
            return buf
        raise RuntimeError(f"Failed to find buffer matching name {buffer_name}")

    def is_unspec_arg(self, name: str) -> bool:
        """Whether this input stands in for a value rather than holding one.

        Asked by name rather than told, because a caller holding a name is
        deciding how to pass the value on, and a kernel handed a stand-in as
        though it were a tensor would read a number out of a shape.
        """

        buf = self._graph_input_named(name)
        if buf is not None and buf.get_numel() == 1 and len(buf.get_size()) == 0:
            if get_device_type(buf) == "cpu":
                return True
        return name in self.zero_dim_cpu_tensor_list



    @contextlib.contextmanager
    def set_current_node(self, node):
        """Say which node is being lowered, and go back to saying none."""

        previous = self.current_node
        self.current_node = node
        try:
            yield node
        finally:
            self.current_node = previous

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

    def static_sizes_strides(self, ex):
        """The shape and the strides of a value, as whole numbers.

        A value whose shape is already known is described by whole numbers
        rather than by expressions, which is what lets a kernel be written
        against a shape that can be read at compile time.  Chiefly what the
        weights of a region are described by.
        """

        size = [sympy.Integer(i) for i in ex.size()]
        stride = [sympy.Integer(i) for i in ex.stride()]
        return size, stride

    def symbolic_sizes_strides_storage_offset(self, ex, source):
        """Describe an input layout with symbols and inferred stride products."""

        from tensorplay.graph.experimental.sym_node import SymNode

        sizes, strides, offset = self.shape_env.create_symbolic_sizes_strides_storage_offset(ex, source)
        def expression(value):
            return value.expr if isinstance(value, SymNode) else sympy.sympify(value)
        return [expression(v) for v in sizes], [expression(v) for v in strides], expression(offset)

    def get_training_phase(self) -> str:
        """Which of the three passes over a region this is.

        Said by the region rather than worked out where it is needed, because
        what a printer writes depends on it: a region being run is written one
        way, a region being trained is written with what it will be trained
        from, and a region being trained from is written differently again.
        """

        if self.is_inference:
            return "inference"
        if self.is_backward:
            return "backward"
        return "forward"

    @staticmethod
    def decide_layout_opt(gm, *, is_inference: bool) -> bool:
        """Whether this region's layouts are ours to choose.

        A region holding no call of the kind that gains from a particular
        layout is not one where choosing would pay, so the question is asked of
        the region rather than answered once for the process.  What the answer
        turns on is the shape of those calls: how many there are against how
        much work the region holds, whether the shapes are known, and the
        proportions of each call -- grouped calls and calls whose incoming
        channels exceed their outgoing ones are the two that a layout of this
        kind is measured to hurt, and calls with few channels are the one it
        does not help.
        """
        if not config.layout_optimization:
            return False

        if config.force_layout_optimization:
            return True

        conv_nodes = [
            n for n in gm.graph.nodes if n.target is tp.ops.tp.convolution.default
        ]

        for n in gm.graph.nodes:
            if is_mkldnn_conv(n):
                conv_nodes.append(n)

        if not conv_nodes:
            # A region records a convolution under the name it was called by.
            # On the cpu those calls land on the same engine the general call
            # does, so a region made of them is laid out the same way.
            named = [n for n in gm.graph.nodes if _is_convolution_node(n)]
            if named and all(
                _node_device_type(arg) in SUPPORTED_MKLDNN_DEVICES
                for n in named
                for arg in n.args[:3]
                if isinstance(arg, tp.graph.Node)
            ):
                conv_nodes = named

        nconv = len(conv_nodes)

        if nconv == 0:
            return False

        # For cpu backend and mkldnn enabled, we always use channels_last for better performance.
        if (
            tp.backends.mkldnn.enabled
            and tp.backends.mkldnn.is_available()
            and all(
                n.args[idx].meta["val"].device.type in SUPPORTED_MKLDNN_DEVICES
                for n in conv_nodes
                for idx in [0, 1]
            )
        ):
            return True

        # A region holding little besides these calls is one where the layout
        # is not what the time goes on, so it is not chosen.
        if len(list(gm.graph.nodes)) >= 300 * nconv:
            log.debug("Skipped layout opt because only a few conv")
            return False

        if any(
            has_free_symbols(n.args[idx].meta["val"])
            for n in conv_nodes
            for idx in [0, 1]
        ):
            log.debug(
                "Skipped layout opt because the extents are not all known"
            )
            return False

        def is_grouped(n: Any) -> bool:
            meta_val = n.args[1].meta["val"]  # type: ignore[union-attr, operator]
            if not isinstance(meta_val, tp.Tensor):
                raise AssertionError(f"Expected tp.Tensor, got {type(meta_val)}")
            return n.args[-1] > 1 and meta_val.size(1) > 1  # type: ignore[union-attr, operator]

        def is_in_out_channel(n) -> bool:
            return (
                n.args[1].meta["val"].size(0) * 2 <= n.args[1].meta["val"].size(1)  # type: ignore[union-attr, operator]
                and n.args[1].meta["val"].size(2) > 1  # type: ignore[union-attr, operator]
            )

        def is_small_channel(n: Any) -> bool:
            return (
                n.args[1].meta["val"].size(0) <= 64  # type: ignore[union-attr, operator]
                and n.args[1].meta["val"].size(1) <= 64  # type: ignore[union-attr, operator]
            )

        # only grouped convolutions benchmarked as slower in conv samples for inference only
        if is_inference:
            flop_counts: dict[str, float] = defaultdict(float)
            for node in conv_nodes:
                counted_flops = count_flops_fx(node)
                if counted_flops is None:
                    continue

                if is_grouped(node):
                    node_type = "grouped"
                elif is_small_channel(node):
                    node_type = "small"
                elif is_in_out_channel(node):
                    node_type = "in_out"
                else:
                    node_type = "default"

                flop_counts[node_type] += counted_flops
            else:
                log.debug("Conv inputs meta not found")

            # Average measured cost of the channels-last layout against the
            # default one, per kind of call; below one is a speedup.  A whole
            # region's work is weighed by how much of it is each kind, so a
            # region that is mostly calls the layout helps is taken, and one
            # that is mostly calls it hurts is not.
            GROUPED_MULTIPLIER = 1.358
            DEFAULT_MULTIPLIER = 0.823
            IN_OUT_MULTIPLIER = 0.725
            SMALL_MULTIPLIER = 0.783

            total_flops = sum(flop_counts.values())
            # TODO - get different values per hardware
            weighted_flops = (
                flop_counts["grouped"] * GROUPED_MULTIPLIER
                + flop_counts["small"] * SMALL_MULTIPLIER
                + flop_counts["in_out"] * IN_OUT_MULTIPLIER
                + flop_counts["default"] * DEFAULT_MULTIPLIER
            )
            do_layout_opt = weighted_flops <= total_flops
            if not do_layout_opt:
                log.debug(
                    "Skipped layout opt in inference because weighted flops indicate slowdown, default: %d, channels last: %d",
                    total_flops,
                    weighted_flops,
                )
            return do_layout_opt

        # Channels last layout can dramatically hurt grouped conv perf. E.g.
        # Conv with arguments like
        #   {"input_shape": [32, 224, 112, 112], "weight_shape": [224, 112, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 2}
        # slows down 31x using channels last..

        # But a lot of timm models use depthwise separable convolution which will
        # result in grouped convolution with in-channel size == 1.
        # For those grouped convolution, channels last still helps a lot.
        # E.g.
        # Conv with arguments
        #   {"input_shape": [128, 58, 56, 56], "weight_shape": [58, 1, 3, 3],
        #    "stride": [2, 2], "padding": [1, 1], "groups": 58}
        # get 1.86x speedup with channels last layout.
        #
        # The following heuristics skip using channels-last if the model contains
        # grouped convolution with in-channels > 1.
        if any(map(is_grouped, conv_nodes)):
            log.debug(
                "Skip layout opt because found grouped convolution with >1 in_channels!"
            )
            return False

        # For some models that contain convolution with larger in-channel than out-channel, applying
        # channels last hurts performance.
        # A call whose incoming channels exceed its outgoing ones is one this
        # layout is measured to hurt, however few of them there are.
        if any(map(is_in_out_channel, conv_nodes)):
            log.debug(
                "Skip layout opt because some convolutions have smaller out_channel"
            )
            return False

        # Calls this narrow throughout are ones the layout does not help.
        if all(map(is_small_channel, conv_nodes)):
            log.debug("Skip layout opt because all convolution channels are too small")
            return False

        return True

    def _find_nodes_prefer_channels_last(self) -> OrderedSet:
        """The nodes whose values are best held channels-last.

        A node is on the list when a convolution reads or writes it, or when
        one of its users is on the list.  The second rule is what keeps an
        indirect input to a convolution in the same layout: without it, a
        conv -> norm -> activation -> conv chain would copy the norm's output
        back to the default layout and then copy it again into channels-last
        before the next convolution.

        The graph is walked backwards first so that every node feeding a
        convolution is marked, then forwards so that downstream nodes of a
        channels-last producer stay in the same layout instead of mixing
        layouts inside a backward kernel.
        """

        gm = self.graph_module
        blocked = (tp.ops.tp.bmm.default,)
        output_set: OrderedSet = OrderedSet()
        last_conv = None
        for n in reversed(list(gm.graph.nodes)):
            if _is_convolution_node(n):
                output_set.add(n)
                if last_conv is None:
                    last_conv = n
                continue
            if n.target in blocked:
                continue
            for user in n.users:
                if user in output_set:
                    output_set.add(n)
                    break

        # A second pass adds the downstream users of the marked nodes, which
        # keeps mixed layouts out of backward kernels.  Propagation stops at
        # the last convolution, which is where the channels-last chain ends.
        for n in gm.graph.nodes:
            if last_conv is not None and n == last_conv:
                break
            if n in output_set:
                for user in n.users:
                    if user.target in blocked:
                        continue
                    output_set.add(user)
        return output_set

    @property
    def fake_mode(self):
        return V.fake_mode

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

        return [
            buf.get_name()
            for buf in self.graph_outputs
            if not isinstance(buf, (ir.NoneAsConstantBuffer, ShapeAsConstantBuffer))
        ]

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
        return feature in self._backend_features_of(_device_type_of(device))

    @functools.cached_property
    def _backend_features_of(self):
        """What the emitter of each device type can express, asked of it once
        per region."""

        from .codegen.common import get_backend_features

        return functools.lru_cache(None)(get_backend_features)

    def warn_fallback(self, kernel_name: str) -> None:
        """Note that a call is being run through to rather than written out.

        Which calls those were is worth knowing afterwards: each one is a
        place where the shape of the result was worked out by running the
        operation rather than by looking at it.
        """

        self.fallback_ops.append(kernel_name)

    def written_storages(self) -> set:
        """The memory this region's operations write into, by storage.

        A region whose every operation makes a new value writes nothing; one
        that updates an input in place, or writes into a copy it made, names
        that memory here.  Asked once and kept: the graph does not change
        while it is lowered.
        """

        cached = getattr(self, "_written_storages", None)
        if cached is not None:
            return cached
        from tensorplay.graph import Node

        from .fx_utils import get_node_storage

        written: set = set()
        for node in self.module.graph.nodes:
            if node.op != "call_function":
                continue
            schema = getattr(node.target, "_schema", None)
            arguments = getattr(schema, "arguments", None)
            if not arguments or not getattr(schema, "is_mutable", False):
                continue
            for position, argument in enumerate(arguments):
                alias = getattr(argument, "alias_info", None)
                if alias is None or not getattr(alias, "is_write", False):
                    continue
                value = (
                    node.args[position]
                    if position < len(node.args)
                    else node.kwargs.get(argument.name)
                )
                for item in value if isinstance(value, (list, tuple)) else (value,):
                    if isinstance(item, Node):
                        storage = get_node_storage(item)
                        if storage is not None:
                            written.add(storage)
        self._written_storages = written
        return written

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
        # A name that was written to rather than read from is answered by what
        # was written to, which is a different name again whenever the write
        # went into a buffer that was already there.
        if (
            hasattr(self.scheduler, "mutation_real_name")
            and buffer_name in self.scheduler.mutation_real_name
        ):
            mutated_buf = self.scheduler.mutation_real_name[buffer_name]
            if mutated_buf in self.name_to_buffer:
                return self.name_to_buffer[mutated_buf].get_dtype()
            if mutated_buf in self.graph_inputs:
                return self.graph_inputs[mutated_buf].get_dtype()
        if buffer_name in self.name_to_buffer:
            return self.name_to_buffer[buffer_name].get_dtype()
        for buf in self.graph_inputs_original.values():
            if buf.get_name() == buffer_name:
                return buf.get_dtype()
        m = re.match(r"(as_strided|reinterpret_tensor)\(([a-zA-Z0-9_]+),", buffer_name)
        if m:
            return self.get_dtype(m.group(1))
        raise KeyError(f"could not find {buffer_name}")

    def get_numel(self, buffer_name: str):
        """How many elements a named buffer holds.

        A buffer this region made is asked directly.  A constant is asked
        through the value it stands for, which is known even where no buffer
        was made for it.  A name that is none of those was never declared, and
        saying so is better than answering zero and letting a kernel be shaped
        to nothing.
        """

        if buffer_name in self.constants:
            return self.constants[buffer_name].numel()
        if buffer_name in self.name_to_buffer:
            buf = self.name_to_buffer[buffer_name]
            if not buf.has_tensor_output():
                return 1
            return buf.get_numel()
        if buffer_name in self.graph_inputs:
            return self.graph_inputs[buffer_name].get_numel()
        raise KeyError(f"could not find {buffer_name}")

    def _get_output_names(self, graph_outputs):
        """What each of a region's outputs is called where it is handed back.

        Most outputs are named after the buffer they were written into, which
        is a name the whole compiler already agrees on.  Two kinds are not: an
        output standing for the absence of a value, and one standing for a
        shape, have no buffer behind them, so each is given a name of its own
        -- counted, so that a region with two of either gives them two names.
        """

        names = []
        shape_counter = itertools.count(0)
        none_counter = itertools.count(0)
        for node in graph_outputs:
            if isinstance(node, NoneAsConstantBuffer):
                names.append(f"{self.name}_none{next(none_counter)}")
            elif isinstance(node, ShapeAsConstantBuffer):
                names.append(f"{self.name}_shape{next(shape_counter)}")
            else:
                names.append(node.get_name())
        return names

    def get_current_device_or_throw(self):
        """The device being emitted for, and a refusal when there is none.

        Code that allocates memory has to know where the memory goes, and a
        region that has not said which device it is on cannot have that
        answered for it.
        """

        if self.current_device is None:
            raise RuntimeError("Trying to get current device but it is not set")
        return self.current_device

    @contextlib.contextmanager
    def set_current_device(self, device):
        """The device a generator is writing code for, for the duration of it.

        A generator emits code without threading a device through every call,
        so the one it is currently emitting for is held here instead, and the
        previous one is put back when it is done.
        """

        previous = self.current_device
        self.current_device = device
        try:
            yield device
        finally:
            self.current_device = previous

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
        self.register_operation(m2)
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
    def _layout_of(val, strides=None):
        """Where the elements of a traced result sit.

        ``strides`` replaces the traced strides when the call was run on
        operands laid out the way this region lays them out and answered with
        another arrangement.

        The offset is always zero: the buffer a call's result is held in is
        the tensor the call returned, whose first element is where its data
        begins.  A view returned at an offset into its base already starts
        there, and counting the base's offset again would read past it.
        """

        return FixedLayout(
            val.device, val.dtype,
            tuple(int(s) for s in val.shape),
            tuple(int(s) for s in (val.stride() if strides is None else strides)),
            0,
        )

    def _probe_fallback_result(self, node, tensor_args, other_args, unflatten):
        """What a described call returns for operands laid out as they are here.

        The traced result says how the call arranged its result for the
        operands the program was traced with.  A region that chooses its own
        layouts hands the call operands arranged differently, and a call that
        keeps its operand's arrangement then returns its result arranged
        differently too.  So where layouts are chosen, the call is run once on
        empty operands with the arrangements they have here, and the result's
        strides are read from that run.  Nothing is returned when layouts are
        left as traced, or when the call cannot be run this way.
        """

        if not self.layout_opt or node.op != "call_function":
            return None
        try:
            # Only a four-dimensional operand can be laid out differently
            # from how it was traced.
            if not any(len(x.get_size()) == 4 for x in tensor_args):
                return None
            for x in tensor_args:
                if ir.is_storage_and_layout(x):
                    ir.as_storage_and_layout(x, freeze=True)
            example = []
            for x in tensor_args:
                if not isinstance(x, ir.BaseView) and x.get_name() in self.constants:
                    example.append(self.constants[x.get_name()])
                else:
                    example.append(ir.ir_node_to_tensor(x))
            args, kwargs = unflatten(example, list(other_args))
            with tp.no_grad():
                return node.target(*args, **kwargs)
        except Exception:
            return None

    @staticmethod
    def _probed_strides(val, probed, item, path=()):
        """The probed strides of one traced result, when the probe has it."""

        cursor = probed
        try:
            for index in path:
                cursor = cursor[index]
        except (TypeError, IndexError, KeyError):
            return None
        if not _is_tensor(cursor) or not _is_tensor(item):
            return None
        if tuple(int(s) for s in cursor.shape) != tuple(int(s) for s in item.shape):
            return None
        if cursor.dtype != item.dtype:
            return None
        return tuple(int(s) for s in cursor.stride())

    def _make_fallback(self, node, kernel_name, realized_args, kwargs):
        """Describe a call that is run rather than written out.

        What the call is given is split the way a call written later expects
        it: the arguments that name memory are inputs, and the rest travel as
        constants beside them, so that regenerating the call puts each argument
        back in the position it was given in.  A method call names its
        operation with a string and takes the object it is called on first, so
        the receiver leads the inputs and the name is all there is to call by.
        """

        if self.shape_env is not None:
            # Output metadata for a call-out is obtained from its example.
            # Keep that metadata valid until symbolic inference covers it.
            for symbol in tuple(self.shape_env.backed_var_to_val):
                self.shape_env.guarding_hint_or_throw(symbol)
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
        probed = self._probe_fallback_result(node, tensor_args, other_args, unflatten)
        if len(tensor_outputs) > 1:
            # Several results cannot all be the call itself, so the call is
            # given no result of its own and each result is a buffer naming the
            # path through the returned structure at which it sits.
            layout = MultiOutputLayout(
                device=tensor_outputs[0].device,
            )
        else:
            layout = self._layout_of(
                tensor_outputs[0], self._probed_strides(val, probed, tensor_outputs[0])
            )

        kernel = IrFallbackKernel(
            layout=layout,
            kernel=node.target,
            tensor_args=tuple(tensor_args),
            nontensor_args=tuple(other_args),
            unflatten_args=unflatten,
            kwargs=realized_kwargs,
        )
        kernel.origin_node = node
        kernel.probed_result = probed
        # A method call that survives to the backend names its operation with
        # a string.  In-place tensor methods carry a trailing underscore: the
        # receiver is mutated and returned, which has to be recorded so memory
        # planning keeps the receiver alive and the scheduler does not drop
        # the call as dead.
        if call_method and isinstance(node.target, str) and node.target.endswith("_"):
            kernel.mutation_names.append(receiver.get_name())
            kernel.alias_names.append(receiver.get_name())
        return kernel

    def _wrap_fallback(self, node, kernel):
        """The values a described call produced, as things a body can read.

        A call that produced one thing is that thing's own buffer, so there is
        nothing to name separately; a call that produced several has one
        buffer per result, each saying where among the results it sits.
        """

        val = node.meta.get("val")

        def count_tensors(item) -> int:
            if _is_tensor(item):
                return 1
            if isinstance(item, (list, tuple)):
                return sum(count_tensors(v) for v in item)
            return 0

        # Only a call that returns a bare tensor returns its result as it is; a
        # lone tensor inside a returned tuple or list is an element of what
        # the call returns, and is taken out of it like any other.
        single_output = _is_tensor(val) and count_tensors(val) == 1

        def wrap(item, path):
            if _is_absent_tensor(item):
                # An output the call was asked not to compute is no value.
                return None
            if _is_tensor(item):
                if single_output:
                    # Boxed the way every value is, so that what holds this
                    # result is a place memory can be given.
                    return TensorBox.create(kernel)
                strides = self._probed_strides(
                    val,
                    getattr(kernel, "probed_result", None),
                    item,
                    tuple(index for _, index in path),
                )
                out = MultiOutput(self._layout_of(item, strides), kernel, path)
                out.origin_node = node
                kernel.outputs.append(out)
                return TensorBox.create(out)
            if isinstance(item, (list, tuple)):
                return tuple(
                    wrap(v, [*path, (type(item), i)]) for i, v in enumerate(item)
                )
            return item

        return wrap(val, [])

    @staticmethod
    def _is_image_box(value) -> bool:
        """Whether a value is a four-dimensional tensor with more than one
        spatial position, the only kind two layouts arrange differently."""

        if not isinstance(value, TensorBox):
            return False
        try:
            size = [int(s) for s in value.get_size()]
        except (NotImplementedError, TypeError, ValueError):
            return False
        return len(size) == 4 and size[2] * size[3] > 1

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

        if self.layout_opt and target_name(node.target) in _LAYOUT_KEEPING_FALLBACKS:
            # These calls keep a channels-last operand's order in their
            # result.  Handing them their activations in that order -- rather
            # than in whatever order putting an unwritten body in memory
            # happens to settle on -- writes each body once, in the order the
            # convolutions around the call read, instead of row-major first
            # and repacked for each of them afterwards.
            args = tuple(
                ir.ExternKernel.require_channels_last(a)
                if self._is_image_box(a)
                else a
                for a in args
            )
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
    @staticmethod
    def _get_node_stream(n):
        """Which stream the program asked this node to run on, if it said."""

        return n.meta.get("custom", {}).get("stream")

    @staticmethod
    def _get_node_mempool(n):
        """Which memory pool the program asked this node to allocate from."""

        custom = n.meta.get("custom", {})
        if "mempool" not in custom:
            return None
        return custom["mempool"], custom["mempool_device"]

    def _realize_inputs_at_context_boundaries(self, n) -> None:
        """Put in memory the inputs this node reads under a different context.

        Two values read under different streams, or with memory taken from
        different pools, cannot be folded into one kernel: what that kernel
        would run on is one stream and one pool, and the other reads would have
        to be moved there, which is not a rewrite this compiler does.  So a
        value whose context differs from the node reading it is put in memory
        first, and the two are then separate kernels.  Nothing recorded means
        the default, so it is compared like any other value.
        """

        node_stream = self._get_node_stream(n)
        node_mempool = self._get_node_mempool(n)
        for input_node in n.all_input_nodes:
            if (
                self._get_node_stream(input_node) == node_stream
                and self._get_node_mempool(input_node) == node_mempool
            ):
                continue
            ir_value = self.env.get(input_node)
            if isinstance(ir_value, TensorBox):
                ir_value.realize()

    def run_node(self, n):
        """The value one node stands for, read under the context it was read in.

        What a node is computed against -- which stream it runs on, which pool
        its memory comes from, which graph nodes it came from -- is decided here
        rather than inside whatever the node happens to be, because a value may
        be reused by a later node read under a different context, and what that
        reuse means can only be said where both contexts are in hand.  The node
        being read is also published for the duration, so that anything written
        while it is read can ask which node it is written for.
        """

        # The nodes this one was made from, gathered before the node is read:
        # afterwards the values it was handed may have been put in memory, and
        # a value in memory no longer says what it came from.
        origins: OrderedSet = OrderedSet([n])
        args = kwargs = None
        if n.op == "call_function":
            args, kwargs = self.fetch_args_kwargs_from_env(n)
            origins |= gather_origins(args, kwargs)
            self._realize_inputs_at_context_boundaries(n)
        node_mempool = self._get_node_mempool(n)

        with (
            IRNode.current_origins(origins),
            IRNode.current_stream_idx(self._get_node_stream(n)),
            IRNode.current_mempool(node_mempool),
            self.set_current_node(n),
            V.set_current_node(n),
        ):
            if n.op == "call_function":
                result = self.call_function(n.target, args, kwargs)
            else:
                result = super().run_node(n)
        # A value not yet in memory reads its operands whenever it is finally
        # written, so a later write over one of them has to put it in memory
        # first; that write can only find it if its reads are on record.
        self.register_users_of(result)
        self.env[n] = result
        return result

    def _channels_last_before_realize(self, n, result):
        """This node's result with channels-last strides, when the chain wants it.

        Asked at the point a value read by several consumers is about to be
        put in memory: a value that feeds a convolution, or is fed by one, is
        stored in the order the convolution reads, so it is written once in
        that order rather than stored row-major and repacked for each reader.
        Whether the value is a dense four-dimensional one is read from the
        traced value, since a body that is not in memory yet has no strides
        of its own to ask.  A value the region returns to its caller keeps the
        layout the caller wrote, and one read by explicit strides keeps the
        layout that reader spelled.
        """

        if not self.stores_channels_last(n):
            return result
        try:
            if len(result.get_size()) != 4:
                return result
        except NotImplementedError:
            return result
        return self.in_channels_last_order(result)

    @staticmethod
    def in_channels_last_order(value):
        """A value arranged channels-last, boxed the way a node's value is.

        Arranging a value may answer with the storage it settled rather than
        with a box around it; a node's value is always the box.
        """

        arranged = ir.ExternKernel.require_stride_order(value, ir.NHWC_STRIDE_ORDER)
        if isinstance(arranged, StorageBox):
            arranged = TensorBox(arranged)
        return arranged

    def stores_channels_last(self, n) -> bool:
        """Whether this node's value is one the region stores channels-last.

        It is when the chain wants it there (the node feeds a convolution or
        is fed by one), the program's caller does not read it, no reader
        spells strides of its own, and the traced value is a dense
        four-dimensional one with more than one spatial position.
        """

        if n not in self.nodes_prefer_channels_last:
            return False
        if n in self.user_visible_nodes:
            return False
        if self._is_input_for_as_strided(n):
            return False
        val = n.meta.get("val")
        if not _is_tensor(val):
            return False
        try:
            size = [int(s) for s in val.shape]
            stride = [int(s) for s in val.stride()]
        except (TypeError, ValueError):
            # Extents that are not plain numbers.
            return False
        if len(size) != 4 or size[2] * size[3] == 1:
            # One spatial position: both orders are the same arrangement, so
            # there is nothing to choose and no copy worth making.
            return False
        return self._is_dense_4d(size, stride)

    @staticmethod
    def _user_visible_nodes(g) -> set:
        """The nodes whose values the caller of the region reads.

        A region that says which of its results are the program's own -- the
        rest being values kept for a later pass -- is believed.  A region that
        does not say is taken to hand every result to its caller.
        """

        output_nodes = g.find_nodes(op="output")
        if not output_nodes:
            return set()
        output_node = output_nodes[0]
        args = output_node.args
        if args and isinstance(args[0], (tuple, list)):
            args = args[0]
        visible = output_node.meta.get("user_visible_output_idxs")
        return {
            a
            for i, a in enumerate(args)
            if hasattr(a, "op") and (visible is None or i in visible)
        }

    def _mark_reuse(self, node, result):
        """Record how many consumers read a result, storing it if that pays.

        A body read by several consumers is stored rather than recomputed per
        consumer when it is worth storing; cheap index arithmetic stays inline
        so a shared constant does not inflate every downstream read count.
        The decision belongs to the box.  Where it decides to store, the
        layout the value is stored in is settled first.
        """

        if not isinstance(result, TensorBox):
            return result
        storage = self._storage_box(result)
        if storage is None:
            return result
        users = len(node.users)
        if users > 1 and self._handed_over_to_a_reader(node):
            # One of the readers is handed the value itself, so it ends up in
            # memory whatever the others do.  Put there as it is made, the
            # other readers read it from memory; left until that reader is
            # reached, each reader before it would compute the value again
            # inside its own loop and write a copy of its own.
            result = self._channels_last_before_realize(node, result)
            result.realize_hint()
            storage = self._storage_box(result)
            if storage is None:
                return result
        if storage.should_realize_on_reuse(users):
            result = self._channels_last_before_realize(node, result)
            storage = self._storage_box(result)
            if storage is None:
                return result
        storage.mark_reuse(users)
        return result

    @staticmethod
    def _handed_over_to_a_reader(node) -> bool:
        """Whether some reader of this node takes the value rather than a loop.

        A convolution, its gradients, and every operation computed by handing
        the call over are given tensors: what they read has to exist.
        """

        from .op_lowerings import needs_realized_inputs

        return any(
            _is_convolution_node(user)
            or getattr(user, "target", None) in needs_realized_inputs
            for user in node.users
        )

    @staticmethod
    def _storage_box(result):
        """The box that holds a result's memory, under any views of it."""

        storage = result.data
        while not isinstance(storage, StorageBox) and isinstance(storage, (View, TensorBox)):
            storage = storage.data
        return storage if isinstance(storage, StorageBox) else None

    def _is_input_for_as_strided(self, n) -> bool:
        """Whether this node feeds a call that reads by explicit strides.

        Such a call spells the layout it wants itself, so choosing
        channels-last here would fight the caller's own choice.
        """

        for user in n.users:
            target = getattr(user, "target", None)
            name = getattr(target, "__name__", str(target))
            if "as_strided" in name or "resize" in name:
                return True
        return False

    @staticmethod
    def _is_dense_4d(size, stride) -> bool:
        """Whether a 4D value is non-overlapping and dense.

        Every element of a dense non-overlapping tensor has one position of
        its own, which is what makes rearranging its strides a pure
        reordering rather than a gather.  Walking the dimensions from the
        smallest stride up, each stride must equal the number of elements the
        dimensions before it span; a dimension of one element spans nothing
        and may carry any stride.
        """

        if len(size) != 4 or len(stride) != 4:
            return False
        order = sorted(range(4), key=lambda i: stride[i])
        expected = 1
        for i in order:
            if size[i] == 1:
                continue
            if stride[i] != expected:
                return False
            expected *= size[i]
        return True

    def run(self, *args):
        """Read the region, one node at a time, and record what each stands for.

        A node is handed to the method named by what it is -- a placeholder to
        the method that gives it a buffer, a call to the method that knows what
        the operation computes -- and the value that method returns is what the
        node stands for from then on.  The order is the region's own order,
        which is the order in which one value can be computed from another.
        """

        with set_ops_handler(DeferredOps()), set_graph(self):
            return super().run(*args)

    def placeholder(self, target, args, kwargs):
        """The value a named input of the region stands for.

        A tensor input becomes a buffer laid out as the value it stands for,
        held in a box, and recorded twice over: once as the value a later step
        asks about by name, and once as the buffer the bytes are read out of.
        Anything that is not a tensor is recorded as itself, since there are no
        bytes to lay out.
        """

        self.placeholder_idx += 1
        example = super().placeholder(target, args, kwargs)
        name = self.qualify_name(target)

        if not _is_tensor(example):
            # A value rather than a tensor: recorded so that a step which asks
            # what this input is learns that it is not a tensor, and returned
            # unchanged so that whatever asked for it can use it.
            self.graph_inputs[name] = example
            self.graph_input_names.append(name)
            return example

        if self.shape_env is None:
            sizes, strides = self.static_sizes_strides(example)
            offset = sympy.Integer(example.storage_offset())
        else:
            sizes, strides, offset = self.symbolic_sizes_strides_storage_offset(example, name)
        buffer = InputBuffer(
            name=name,
            layout=FixedLayout(example.device, example.dtype, sizes, strides),
        )
        # A value is made out of a box that decides what to do with it, over
        # storage that decides where the bytes are.  An input is not computed,
        # so its storage is a buffer that was handed to us rather than one this
        # region made -- which is the fact that later frees a caller back to
        # being able to go on holding it.
        tensor = TensorBox.create(buffer)

        self.name_to_buffer[buffer.name] = buffer
        self.buffers.append(buffer)
        self.graph_inputs[name] = tensor
        self.graph_input_storage_offsets[name] = offset
        self.graph_inputs_original[name] = buffer
        self.graph_input_names.append(name)
        return tensor

    def _materialize_embedded_tensor(self, value):
        """A constant tensor a graph holds is read as a buffer.

        A value that was written into the graph as a constant has no node of
        its own, so the walk gives it a named buffer the first time it is read
        and reuses that buffer for every later read of the same value.
        """

        if isinstance(value, tp.Tensor):
            key = id(value)
            cached = self._embedded_tensor_constants.get(key)
            if cached is None:
                cached = self.add_tensor_constant(value)
                self._embedded_tensor_constants[key] = cached
            return cached
        return value

    def call_module(self, target, args, kwargs):
        """A region is never a call to another region."""

        raise AssertionError

    @classmethod
    def _in_recorded_dtype(cls, val, result):
        """A node's value in the element type the trace recorded for it.

        The recorded value is what the program computed when it ran, so its
        element type is the node's.  A lowering works the type out from the
        operands it was handed, and the operands cannot say that the operation
        changed the type on its own: a mixed-precision run computes some
        operations wider than their operand and others narrower, without a
        conversion appearing in the program.  A value that came out in another
        type is read as the recorded one here, in the loop that computes it,
        so every buffer that holds it and every reader of it agree with what
        the program saw.
        """

        if isinstance(result, (list, tuple)):
            if isinstance(val, (list, tuple)) and len(val) == len(result):
                return type(result)(
                    [cls._in_recorded_dtype(v, r) for v, r in zip(val, result)]
                )
            return result
        if not isinstance(result, TensorBox) or not _is_tensor(val):
            return result
        try:
            have = result.get_dtype()
        except NotImplementedError:
            # A value of several parts has no one type to compare.
            return result
        if have == val.dtype:
            return result
        from .op_lowerings import to_dtype

        return to_dtype(result, val.dtype)

    def call_method(self, target, args, kwargs):
        """What the region means by a method call, as a value the rest can be
        written against.

        A method call names its operation with a string and takes the object
        it is called on first.  A lowering is looked up for the method's bare
        name and overload spellings; view-like methods pass their shape as
        varargs while the lowering expects a tuple, so the trailing arguments
        are packed first.  An in-place method (trailing underscore) with no
        lowering is handed to the framework whole -- the receiver leads the
        inputs and the mutation is recorded so the call is not dropped as dead.
        Any other method the backend has no lowering for stays a boundary the
        compiler does not cross, exactly as before.
        """

        if not isinstance(target, str):
            raise AssertionError
        node = V.graph.current_node
        name = target_name(target)
        args = pytree.tree_map(self._materialize_embedded_tensor, args)
        kwargs = pytree.tree_map(self._materialize_embedded_tensor, kwargs)
        in_place = name.endswith("_")
        lowering = None if in_place else (
            user_lowerings.get(node)
            or user_lowerings.get(target)
            or find_lowering(name)
        )
        with self.set_current_node(node), set_current_node(node):
            if lowering is not None:
                method_args = args[1:]
                if name in {"view", "reshape", "permute"}:
                    if len(method_args) != 1 or not isinstance(
                        method_args[0], (list, tuple)
                    ):
                        method_args = (tuple(method_args),)
                result = lowering(args[0], *method_args, **kwargs)
            elif in_place:
                result = self.make_extern(node, args, kwargs)
            else:
                raise AssertionError
            result = self._in_recorded_dtype(node.meta.get("val"), result)

        result = self._mark_reuse(node, result)
        assign_origin_node(result, node)
        return result

    def call_function(self, target, args, kwargs):
        """What the region means by a call, as a value the rest can be written
        against.

        A lowering the program wrote for this operation itself is asked before
        the built-in one for the name the operation happens to be called under,
        because those two describe different operations that share a spelling.
        With neither, the call is handed to the framework whole: slower than a
        kernel and correct, and the only thing left to say.
        """

        node = V.graph.current_node
        args = pytree.tree_map(self._materialize_embedded_tensor, args)
        kwargs = pytree.tree_map(self._materialize_embedded_tensor, kwargs)
        name = target_name(target)
        if getattr(target, "__module__", None) in ("operator", "_operator") and all(
            isinstance(arg, (int, float, bool, sympy.Basic)) for arg in args
        ):
            return target(*args, **kwargs)
        if name == "getitem" and args and isinstance(args[0], (list, tuple)):
            # Indexing a result tuple is answered here rather than called out
            # to: a value that is already computed is addressed, not recomputed.
            return args[0][args[1]]
        if (
            name == "getitem"
            and len(args) == 2
            and isinstance(args[0], (TensorBox, ir.BaseView))
        ):
            # Basic indexing names a window of the value: written as views it
            # is read in place by whatever consumes it.
            from .op_lowerings import _as_box, lower_basic_getitem

            with self.set_current_node(node), set_current_node(node):
                viewed = lower_basic_getitem(_as_box(args[0]), args[1])
            if viewed is not None:
                viewed = self._mark_reuse(node, viewed)
                assign_origin_node(viewed, node)
                return viewed

        lowering = (
            user_lowerings.get(node)
            or user_lowerings.get(target)
            or find_lowering(name)
        )
        if lowering is not None and fallback_node_due_to_unsupported_type(node):
            # A value no kernel here can hold is read or written by this call,
            # so the framework runs the call and the region keeps the rest.
            lowering = None
        with self.set_current_node(node), set_current_node(node):
            if lowering is not None:
                result = lowering(*args, **kwargs)
            else:
                result = self.make_extern(node, args, kwargs)
            result = self._in_recorded_dtype(node.meta.get("val"), result)

        result = self._mark_reuse(node, result)
        # Which node of the region this value was made by, so that a report
        # about it can name where it came from.
        assign_origin_node(result, node)
        return result

    def output(self, target, args, kwargs):
        """What the region hands back, put in memory before it is recorded.

        A caller reads an output out of memory, so however the value was
        arrived at -- still held in a box, or already a bare view -- it is put
        in memory first, and a stride a caller does not read is matched back to
        the stride the region said it had so that the two descriptions of the
        same value agree.
        """

        result = super().output(target, args, kwargs)
        if not isinstance(result, (tuple, list)):
            # A subgraph may hand back a single value without wrapping it.
            result = (result,)

        for value in result:
            # What an output is allowed to be.  A value is somewhere memory
            # can be given, something already known without computing it, or a
            # shape; anything else is a node type this walk cannot account
            # for, and being told here is far cheaper than being told much
            # later by a kernel that cannot read what it was handed.
            if not isinstance(value, _allowed_output_types):
                raise AssertionError(
                    f"Unexpected output types: {[type(value)]}, full result: {value}"
                )
            if value is None:
                # A result the program returns as nothing: written out as
                # nothing, and named by nothing.
                self.graph_outputs.append(ir.NoneAsConstantBuffer())
                continue
            self.graph_outputs.append(
                self.realize_input(value) if isinstance(value, IRNode) else value
            )
        self._separate_aliased_outputs()
        self.single_output = len(self.graph_outputs) == 1
        self.finalize()

    @staticmethod
    def _traced_storage(node):
        """Which memory the traced program's value for a node lived in."""

        val = node.meta.get("val") if hasattr(node, "meta") else None
        if not _is_tensor(val) or not val.defined():
            return None
        try:
            return val.untyped_storage().data_ptr()
        except (RuntimeError, AttributeError):
            return None

    def _separate_aliased_outputs(self) -> None:
        """Give an output memory of its own where the program gave it some.

        Inside the region a copy that changes nothing may be the value it
        copies, since nothing writes to either.  An output leaves the region,
        though, and a caller may write to it: if it came back as one of the
        region's inputs, or as the same memory as another output, the write
        would reach the other one too.  So an output whose memory is an input's
        or an earlier output's is copied -- unless the traced program itself
        returned that memory, in which case sharing it is what was asked for.
        """

        fx_outputs = self.current_node.args[0] if self.current_node is not None else ()
        if not isinstance(fx_outputs, (tuple, list)):
            fx_outputs = (fx_outputs,)
        if len(fx_outputs) != len(self.graph_outputs):
            return
        placeholder_storage = {}
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                placeholder_storage[self.qualify_name(node.target)] = self._traced_storage(node)
        claimed: dict = {}
        for position, (out, fx_node) in enumerate(zip(self.graph_outputs, fx_outputs)):
            if not isinstance(out, (Buffer, ReinterpretView)):
                continue
            storage = out.data.get_name() if isinstance(out, ReinterpretView) else out.get_name()
            traced = self._traced_storage(fx_node)
            if storage in placeholder_storage:
                shared = traced is not None and traced == placeholder_storage[storage]
            elif storage in claimed:
                shared = traced is not None and traced == claimed[storage]
            else:
                claimed[storage] = traced
                continue
            if shared:
                continue
            copied = IrExternKernel.copy_input(out)
            self.graph_outputs[position] = self.realize_input(copied)


    def _update_scheduler(self) -> None:
        """(Re)build the scheduler for this region.

        Built with storing of compiled device code switched off: a measurement
        that had already written some would be measuring a program that had
        already paid for writing it, and the schedule it chose would be chosen
        against that rather than against the work.
        """

        from .scheduler import Scheduler

        with config.patch("triton.store_cubin", False):
            self.scheduler = Scheduler(self.operations)

    @property
    def is_dual_wrapper_mode(self) -> bool:
        """Whether both a just-in-time and an ahead-of-time form are written.

        Written when the region is being built ahead of time and at least one
        device in it is written by the kernel-writing printer, since that is the
        case where the just-in-time form is what tunes the kernels.
        """

        from . import ir

        if not self.aot_mode or config.triton.autotune_at_compile_time:
            return False
        return any(ir.is_triton(d) for d in self.device_types)

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
        Which printer writes it is the device's own, asked of the device -- and
        a device answers only once it has been equipped, so it is equipped here,
        before anything can ask.
        """

        from .codegen.common import get_wrapper_codegen_for_device, init_backend_registration

        init_backend_registration()

        device_types = self.device_types.copy()
        device_types.discard("cpu")
        device_types.discard("meta")
        if len(device_types) > 1:
            raise AssertionError(
                "Does not support mixing {}".format("+".join(device_types))
            )
        only_cpu = len(device_types) == 0
        self.device_type = "cpu" if only_cpu else device_types.pop()

        # The device this region's code is written for is asked how to spell
        # the things a wrapper cannot spell for itself -- which device to make
        # current, which stream to wait on -- before the wrapper is made, so
        # that the wrapper is never written without them.
        self.device_ops = get_device_op_overrides(self.device_type)

        wrapper_code_gen_cls = get_wrapper_codegen_for_device(
            self.device_type, self.cpp_wrapper, self.fx_wrapper
        )
        if wrapper_code_gen_cls is None:
            raise AssertionError(f"Device {self.device_type} not supported")
        self.wrapper_code = wrapper_code_gen_cls.create(
            is_subgraph,
            subgraph_name,
            parent_wrapper_code,
            partition_signatures,
        )

    def codegen(self):
        """Print this region: the kernels, and the program that calls them."""

        self.init_wrapper_code()

        self._update_scheduler()
        if config.draw_orig_fx_graph:
            V.debug.draw_orig_fx_graph(self.orig_gm, self.scheduler.nodes)

        self.wrapper_code.push_codegened_graph(self)
        self.scheduler.codegen()

        log.debug(
            "Finished codegen for all nodes. The list of kernel names available: %s",
            V.graph.all_codegen_kernel_names,
        )

        result = self.wrapper_code.generate(self.is_inference)
        self.wrapper_code.pop_codegened_graph()
        return result

    def codegen_subgraph(self, parent_graph) -> None:
        """Print this piece into the program of the region that runs it.

        A piece's kernels and calls go into the region's own program, so the
        piece writes with the region's printer and for the region's devices,
        and nothing is finished here: the region finishes the program once
        every piece in it has been printed.
        """

        self.wrapper_code = parent_graph.wrapper_code
        self.device_ops = parent_graph.device_ops
        self.cpp_wrapper = parent_graph.cpp_wrapper
        self.device_types = parent_graph.device_types
        self.device_idxs = parent_graph.device_idxs
        self.device_type = parent_graph.device_type

        self._update_scheduler()
        self.scheduler.codegen()

    def _compile_to_module_lines(self, wrapper_code):
        """Write what was printed, and load it back as something callable.

        The text is written to a file under a key derived from it, and the file
        is found again by that key, so a region printed twice is loaded once.
        Which source line each part of the wrapper came from is carried along
        with it, so that a failure inside the wrapper can be reported against
        the line of this region that produced it rather than against the line of
        the wrapper it was printed into.
        """

        from .codecache import PyCodeCache

        linemap = [
            (line_no, node.stack_trace)
            for line_no, node in wrapper_code.line_map
        ]
        key, path = PyCodeCache.write(wrapper_code.value)

        mod = PyCodeCache.load_by_key_path(
            key,
            path,
            linemap=linemap,
            attrs={
                **self.constants,
                **self.torchbind_constants,
                **self.opaque_value_type_classes,
            },
        )
        self.cache_key = key
        self.cache_path = path
        self.cache_linemap = linemap
        return mod

    def compile_to_module(self):
        """This region as something built, named, and callable on its own.

        What comes back is a built artifact rather than a program object: it
        knows the key it was built under and the file it was written to, so
        the same region asked for again is recognised as the same region and
        the built file is reached rather than rebuilt.
        """

        wrapper_code, _ = self.codegen()

        if isinstance(wrapper_code, CppWrapperCode):
            return self._compile_to_cpp_module(wrapper_code)
        if isinstance(wrapper_code, ValueWithLineMap):
            return self._compile_to_module_lines(wrapper_code)
        if isinstance(wrapper_code, FileBackedGraphModule):
            return wrapper_code
        raise NotImplementedError(
            f"Unrecognized wrapper code type: {type(wrapper_code)}"
        )

    def _compile_to_cpp_module(self, wrapper_code: CppWrapperCode):
        """Compile the generated C++ wrapper into a shared object.

        The wrapper text is written to the code cache under a key derived from
        its content and the compiler flags, built with the same host toolchain
        used for generated host kernels, and loaded with ctypes.  The returned
        module aliases graph input tensors and allocates output and
        intermediate buffers on every call.
        """

        import ctypes
        import os
        import sysconfig

        from .codecache import write
        from .cpp_builder import CppBuilder, CppOptions, get_cpp_compiler, package_paths

        paths = package_paths()
        compiler = get_cpp_compiler()
        if paths is None or not compiler:
            raise RuntimeError("host C++ runtime is unavailable")
        include_dir, generated_include_dir, lib_dir = paths
        generated_ops_dir = os.path.join(
            os.path.dirname(os.path.dirname(include_dir)),
            "build",
            "generated",
        )
        python_include = sysconfig.get_paths().get("include")
        if not python_include:
            raise RuntimeError("Python headers are unavailable")
        options = CppOptions(
            compiler=compiler,
            include_dirs=[
                include_dir,
                generated_include_dir,
                generated_ops_dir,
                python_include,
            ],
            cflags=[
                "-std=c++20",
                "-O3",
                "-fPIC",
                "-shared",
                "-pthread",
                "-fopenmp",
            ],
            library_dirs=[lib_dir],
            libraries=["p10", "gomp"],
            ldflags=["-pthread", f"-Wl,-rpath,{lib_dir}"],
        )
        key, source_path = write(
            wrapper_code.value,
            "cpp",
            extra=options.command(["<sources>"], "<output>").__repr__(),
        )
        output_path = os.path.join(os.path.dirname(source_path), f"{key}.so")
        builder = CppBuilder(
            name=os.path.basename(output_path),
            sources=[source_path],
            options=options,
            output_dir=os.path.dirname(source_path),
        )
        if not os.path.exists(output_path):
            builder.build()

        lib = ctypes.CDLL(output_path)
        call_fn = lib.call
        call_fn.restype = None
        call_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_long)]

        self.cache_key = key
        self.cache_path = source_path
        return CppWrapperModule(call_fn, wrapper_code)

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





    # -- one node ---------------------------------------------------------


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

    @staticmethod
    def can_inline_constant(t) -> bool:
        """Whether a constant is small enough to be written into the body.

        A handful of numbers, or one short row of them, is cheaper written
        where it is used than carried as a value of its own -- there is no
        buffer to allocate, and no name to keep.  Anything larger is worth a
        buffer, because then it is computed once rather than read out of the
        program text at every use.
        """

        return len(t.shape) == 1 and t.shape[0] <= 8

    def add_tensor_constant(self, data, name: str | None = None):
        """A value held by the program rather than computed, as a buffer.

        It is given a buffer like anything else so that reading it costs the
        same as reading anything else.  The value is recorded under the name
        the buffer got -- which :meth:`allocate_non_dup_const_name` has already
        done, since it is the one that decides the name and the wrapper needs
        the value under exactly that name to fill the buffer in.
        """

        new_name = self.allocate_non_dup_const_name(name, data)
        return TensorBox.create(
            ConstantBuffer(
                name=new_name,
                layout=FixedLayout(
                    data.device,
                    data.dtype,
                    *self.static_sizes_strides(data),
                ),
            )
        )

    def get_attr(self, target: str, args, kwargs):
        """What a name the program read off itself stands for.

        A weight is the common case: a value that is already there before
        anything runs, so nothing computes it and it is simply carried.  A
        value small enough to write into the body is written there instead,
        since a buffer it can fit in would cost more to allocate than the value
        occupies.  A graph read off the program is a region in its own right
        and is lowered as one.
        """

        getter = getattr(self.module, "_get_attr", None)
        value = getter(target) if callable(getter) else getattr_recursive(self.module, target)

        # A graph is recognised by what reading a node from it needs, not by
        # merely carrying one: a module that happens to hold a graph of its own
        # is still a value this region can hold, and refusing to read it as a
        # region is what keeps a weight from being mistaken for one.
        if hasattr(value, "graph") and hasattr(value, "code"):
            if target in self.seen_subgraphs:
                return self.seen_subgraphs[target]
            out = Subgraph(name=target, graph_module=value)
            self.seen_subgraphs[target] = out
            return out

        if not isinstance(value, tp.Tensor):
            raise AssertionError(f"Expected a tensor, got {type(value)}")

        if self.can_inline_constant(value):
            from .op_lowerings import tensor

            return tensor(
                value.tolist(),
                dtype=value.dtype,
                device=value.device,
            )

        return self.add_tensor_constant(value, target)

    def allocate_non_dup_const_name(self, name, data) -> str:
        """A name for a constant that no other constant is already using.

        The name a program gave is not usable as it stands: it may hold a dot,
        and it says nothing about which region of which program it came from.
        So it is given this region's prefix, anything a name cannot hold becomes
        an underscore, and a name already taken gets a number -- because two
        constants under one name would be one constant as far as anything
        reading by name can tell.

        The value itself is recorded under the name decided on here, along with
        a description of it that does not depend on reading the value, and the
        name the program gave, so that a later step can tell which is which.
        """

        if name is None:
            name = f"constant{len(self.constants)}"
        orig_name = name
        if name[0].isdigit():
            # A name may not begin with a digit where it is written as a name.
            name = f"constant_{name}"
        name = normalize_name(self.qualify_name(name))
        prefix = name
        counter = 0
        while name in self.constants:
            name = f"{prefix}_{counter}"
            counter += 1
        self.constants[name] = data
        self.constant_reprs[name] = (
            f"{data.device!r} {data.dtype!r} "
            f"{tuple(data.size())!r} {tuple(data.stride())!r} "
            f"{hash(data):x}"
        )
        self.allocated_constant_name[name] = orig_name
        return name

    def constant_name(self, name: str, device_override: Any) -> str:
        if device_override is None or self.constants[name].device == device_override:
            return name
        non_dup_const_name = self.allocate_non_dup_const_name(
            f"{name}_{device_override.type}{device_override.index or 0}",
            self.constants[name].to(device_override),
        )
        if non_dup_const_name not in self.constants:
            raise AssertionError(
                f"{non_dup_const_name} should be in V.graph.constants already"
            )
        return non_dup_const_name

    @contextlib.contextmanager
    def set_current_wrapper_code(self):
        """Write into a different place for a while, and go back to the old one.

        A piece compiled on its own is written into a function of its own, and
        whatever is written while that happens is not part of the code around it.
        So the place being written into is saved and put back, which is what lets
        one region be written out while another is still open without the two
        writing into each other.
        """

        old = self.wrapper_code
        try:
            yield
        finally:
            self.wrapper_code = old

    def make_subgraph(self, gm, example_inputs, subgraph_name: str):
        """A piece of this region, compiled on its own and called from here.

        Everything the piece shares with the region it came from is passed
        along, so that what is compiled is the piece itself and nothing else.
        Its name is qualified by this region's, so that two regions that each
        contain a piece by the same name do not collide.
        """

        return SubgraphLowering(
            self,
            gm,
            example_inputs,
            shape_env=self.shape_env,
            cpp_wrapper=self.cpp_wrapper,
            aot_mode=self.aot_mode,
            extern_node_serializer=self.extern_node_serializer,
            is_inference=self.is_inference,
            is_backward=self.is_backward,
            name=f"{self.name}_{subgraph_name}" if self.name else subgraph_name,
        )


def _ints(value) -> tuple:
    """An operator argument that may be one int or a sequence of them."""

    if value is None:
        return ()
    if isinstance(value, int):
        return (int(value),)
    return tuple(int(v) for v in value)



def _is_tensor(value) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and hasattr(value, "stride")
        and not _is_absent_tensor(value)
    )


def _is_absent_tensor(value) -> bool:
    """Whether a value is the placeholder an operator returns for an output
    it was asked not to compute (a gradient masked off, say): it has no
    elements, no layout and no memory, so it stands for nothing."""

    defined = getattr(value, "defined", None)
    return callable(defined) and hasattr(value, "stride") and not defined()


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
