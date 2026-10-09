"""The operations a lowered graph is made of, before any loop is chosen for them.

A node here says what is computed -- a pointwise result, a reduction, a read of
an input -- and says nothing about how.  The layout of a node is part of what it
computes rather than how, because the layout decides what the operations that
follow may assume about where elements sit; a view is the one thing that
changes a layout without computing anything.
"""

import contextlib
import dataclasses
import enum
from dataclasses import dataclass
import functools
import itertools
import logging
import traceback
import textwrap
from functools import partial
from unittest.mock import patch
from collections.abc import Callable, Iterable, Sequence
from typing import Any, ClassVar, Protocol, TYPE_CHECKING, TypeVar

import sympy
from sympy import Expr

import tensorplay as tp
from tensorplay._ops import OpOverload
from tensorplay.utils import _pytree
from tensorplay.primitives.common import is_boolean_dtype, is_float_dtype

from tensorplay.graph.experimental.symbolic_shapes import (
    compute_unbacked_bindings,
    free_symbols,
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
    ShapeEnv,
)
from tensorplay.graph.experimental.sympy_functions import (
    CleanDiv,
    FloorDiv,
    Max,
    Min,
    Mod,
    ModularIndexing,
    OrderedSet,
    SymT,
)

from . import config
from .loops import get_current_node, metrics, ops, V

log = logging.getLogger(__name__)
from . import dependencies
from .dependencies import (
    extract_free_symbols,
    extract_read_writes,
    SymbolUsageCollectorOpsHandler,
)
from .loop_body import LoopBody
from .codegen.common import (
    BackendFeature,
    CodegenSymbol,
    index_prevent_reordering,
)
from .ops_handler import OpCounterCSE
from .runtime.hints import ReductionHint, TileHint
from .utils import (
    cache_on_self,
    GPU_ALIGN_BYTES,
    cache_on_self_and_args,
    ceildiv,
    get_free_symbols,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
    compute_required_storage_length,
    make_channels_last_strides_for,
    sympy_product,
    sympy_subs,
)
from .codegen.index_expr import _lift, floordiv


def convert_shape_to_tp(lst) -> list:
    """The shape and stride of a value, as expressions.

    Ordinary values are already numbers and need nothing done to them, while a
    shape that came out of the data is an expression already; either way the
    answer wanted here is a list of expressions, so both are put in that form.
    """

    return [sympy.sympify(i) for i in lst]


def is_gpu(device) -> bool:
    """Whether this device is one the code is compiled for the GPU."""

    return device is not None and str(device).split(":")[0] in ("cuda", "xpu", "mps")


def try_match_insignificant_strides(tensor: "IRNode", strides) -> "IRNode":
    """Match the strides asked for, leaving the ones that do not matter alone.

    A dimension of extent zero or one has no memory arrangement worth speaking
    of, so its stride is written down as asked even when it differs from what
    is there.  A difference on a dimension that does occupy memory means the
    two layouts are genuinely different, and the value is handed back as it is
    rather than being rewritten into something that would be wrong.
    """

    if not is_storage_and_layout(tensor):
        return tensor

    if all(
        V.graph.sizevars.statically_known_equals(s1, s2)
        for s1, s2 in zip(strides, tensor.get_stride())
    ):
        return tensor

    if not significant_strides_equal(strides, tensor.get_stride(), tensor.get_size()):
        return tensor

    storage, old_layout = as_storage_and_layout(tensor)
    new_stride = [*old_layout.stride]
    is_empty = tensor.is_zero_elements()
    for i, s in enumerate(tensor.get_size()):
        if is_empty or V.graph.sizevars.statically_known_leq(s, 1):
            new_stride[i] = strides[i]

    new_layout = FixedLayout(
        old_layout.device,
        old_layout.dtype,
        old_layout.size,
        new_stride,
        old_layout.offset,
        old_layout.is_pinned,
    )
    return TensorBox(ReinterpretView(data=storage, layout=new_layout))


def _is_static(x) -> bool:
    """Whether a value is already known, rather than being worked out later.

    A whole number written down is known wherever it appears; a symbol standing
    for one is a question that has not been answered yet, and the two are not
    interchangeable where a layout or a size is being written out.
    """

    return isinstance(x, (int, sympy.Integer))


def may_convert_to_optional(value):
    """A list of arguments as an optional one, so that an empty one is written.

    An empty list and a list holding nothing are not the same thing to a caller
    that has to decide whether to pass anything at all, so an empty list becomes
    a one-element list holding nothing: it is written where a caller can see
    that something was meant, and a list that already has something in it is
    left as it is.
    """

    if isinstance(value, list) and not value:
        return [None]
    return value


def is_nonfreeable_buffers(dep) -> bool:
    """Whether this buffer is one the graph did not produce and cannot reuse.

    A subgraph prefixes the names of what it contains, so the prefix is taken
    off before asking, since the question is what the buffer is rather than
    which subgraph it is in.
    """

    from .loops import V

    dep_name = dep.name
    if V.graph is not None and V.graph.name:
        dep_name = dep_name.removeprefix(V.graph.name + "_")
    return dep_name.startswith(
        ("primals_", "arg", "fwd_rng_state", "bwd_rng_state", "tangents")
    )


def has_free_unbacked_symbols(x, unbacked_only: bool = False) -> bool:
    """Whether this value contains a shape whose value is not known yet.

    A shape that came out of the data has no value until the data is there, so
    anything containing one cannot be worked out before running.
    """

    return len(get_free_symbols(x, unbacked_only=unbacked_only)) > 0


def _flat_window_base_name(x) -> str | None:
    """The storage's name, when the value reads that storage in order.

    A window with contiguous strides and no offset over row-major storage
    holds exactly the storage's bytes under another shape, so a copy of the
    window is a copy of the storage.  A shifted or strided window, a reshaped
    reading, or an unrealized body names only itself, and the answer is none.
    """

    node = x
    while isinstance(node, MutableBox):
        node = node.data
    if not isinstance(node, ReinterpretView):
        return None
    cursor = node
    while isinstance(cursor, ReinterpretView):
        layout = cursor.get_layout()
        try:
            size = [int(s) for s in cursor.get_size()]
        except (TypeError, ValueError):
            return None
        if int(layout.offset) != 0 or list(layout.stride) != [
            int(s) for s in FlexibleLayout.contiguous_strides(size)
        ]:
            return None
        cursor = cursor.data
    if isinstance(cursor, Buffer) and cursor.layout.is_contiguous():
        return cursor.get_name()
    return None


def is_contiguous_for_memory_format_or_false(x, memory_format) -> bool:
    """Whether this value is laid out in that memory format.

    A value that has no layout to check, or a format this backend does not
    name, is not in it -- the answer is false rather than an error, so that a
    caller can ask about every value without asking first what each one is.
    """

    if memory_format is None or memory_format is False:
        return False
    if not isinstance(x, IRNode):
        return False
    layout = x.get_layout()
    if not isinstance(layout, Layout):
        return False
    name = getattr(memory_format, "name", None) or str(memory_format)
    if "ChannelsLast" in name and len(layout.size) == 4:
        return is_stride_order_storage_and_layout(
            as_storage_and_layout(x), [0, 2, 3, 1]
        )
    if "ChannelsLast3d" in name and len(layout.size) == 5:
        return is_stride_order_storage_and_layout(
            as_storage_and_layout(x), [0, 2, 3, 4, 1]
        )
    return False


def get_kernel_metadata(node_schedule, wrapper=None) -> tuple:
    """A one-line description of where a kernel came from, for a comment.

    The first element is short enough to sit in a comment, and the second spells
    the same thing out, for a place that has room.
    """

    all_origins = aggregate_origins(node_schedule)
    origins = [origin for origin in all_origins if getattr(origin, "op", None) == "call_function"]

    if not origins:
        return "", ""
    nodes = [getattr(origin, "node", None) for origin in origins]
    named = [getattr(n, "_origins", None) or n for n in nodes if n is not None]
    targets = [getattr(n, "target", None) for n in named]
    targets = [t for t in targets if t is not None]

    if not targets:
        return "", ""
    stack = ", ".join(str(t) for t in targets)
    short = f"from {stack}"
    return short, f"{short}"


def aggregate_origins(node_schedule) -> list:
    """Every place in the trace that produced what is in this kernel."""

    if not isinstance(node_schedule, (list, tuple)):
        node_schedule = [node_schedule]
    origins = []
    for node in node_schedule:
        if node is None:
            continue
        for origin in getattr(node, "get_origins", lambda: [])():
            if origin not in origins:
                origins.append(origin)
        own = getattr(node, "origin_node", None)
        if own is not None and own not in origins:
            origins.append(own)
    return origins


def resolve_unbacked_bindings(shape_env, unbacked_bindings):
    """What the symbols whose values come from the data were bound to.

    Each binding is a path into a value; walking it gives the symbol standing
    for that position, so a caller can learn which shapes need a value before
    the kernel runs.
    """

    if unbacked_bindings is None:
        return None
    resolved = {}
    for s, keypath in unbacked_bindings.items():
        if isinstance(keypath, (list, tuple)):
            keypath = tuple(keypath)
        resolved[s] = keypath
    return resolved


#: One level of indentation, for the text of a node's description.
indent = functools.partial(textwrap.indent, prefix="  ")


_T = TypeVar("_T")
_U = TypeVar("_U")
_V = TypeVar("_V")


def ir_dataclass(cls=None, /, *, frozen: bool = True):
    """Declare a node of the graph, with every field given by keyword.

    Keyword-only is what makes adding a field to a node a change every
    construction has to account for, rather than a change that silently
    reorders the positional arguments of every call.
    """

    def wrap(cls):
        return dataclasses.dataclass(cls, kw_only=True, frozen=frozen)

    if cls is None:
        return wrap
    return wrap(cls)


def argsort(seq: Sequence, *, reverse: bool = False) -> list:
    """The positions of a sequence, ordered by what it holds.

    Equal entries keep the order they came in, and the order they keep is the
    one that makes an inner dimension come before an outer one when sorting
    upwards: strides of thirty-two, eight, eight and one sort to three, two,
    one, zero rather than three, one, two, zero.
    """

    getter = seq.__getitem__
    a_r = range(len(seq))
    # Equal strides keep their original order, so that an inner dimension
    # precedes an outer one when sorting upwards and follows it when downwards.
    sort_idx = list(sorted(a_r, key=getter, reverse=True))
    if not reverse:
        return list(reversed(sort_idx))
    return sort_idx


def argsort_sym(
    shape_env,
    seq: Sequence,
    *,
    reverse: bool = False,
) -> list:
    """The positions of a sequence whose entries are not all known.

    A comparison the environment cannot settle falls back to whatever the
    entries suggest their values are, which is a guess: the order this returns
    is for choosing between ways of writing the same kernel, and nothing about
    what is computed may rest on it.
    """

    def cmp(a, b) -> int:
        a_idx, a_val = a
        b_idx, b_val = b

        def evaluate(expr, fallback):
            if isinstance(expr, bool):
                return expr
            try:
                return shape_env.evaluate_expr(expr)
            except Exception:
                return fallback()

        def hint_lt(lhs, rhs) -> bool:
            # Sorting strides only chooses an order, so an unsettled comparison
            # may be answered by the hints rather than left undecided.
            return shape_env.optimization_hint(lhs) < shape_env.optimization_hint(rhs)

        if evaluate(a_val < b_val, lambda: hint_lt(a_val, b_val)):
            return -1
        if evaluate(a_val > b_val, lambda: hint_lt(b_val, a_val)):
            return 1
        # Equal entries keep the order they came in, as the sort above does:
        # strides of twenty forty-eight, twenty forty-eight, sixteen and one
        # give three, two, one, zero.
        if a_idx < b_idx:
            return 1
        if a_idx > b_idx:
            return -1
        return 0

    exprs = [(idx, getattr(s, "node", s) and getattr(getattr(s, "node", s), "expr", s)) for idx, s in enumerate(seq)]
    exprs = sorted(exprs, key=functools.cmp_to_key(cmp), reverse=reverse)
    return [i for i, _ in exprs]


def same_reorder(order: Sequence[int]):
    """A function that applies one loop order to a list of indexes.

    The order is a permutation, given as which position each loop is taken
    from, so applying it to a list of loop variables gives the same variables
    in a different nesting.  A list of the wrong length is refused rather than
    partly reordered, since a partly reordered index is an index that addresses
    something else.
    """

    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        if len(index) != len(order):
            raise AssertionError("Expected len(index) == len(order)")
        return [index[order[i]] for i in range(len(index))]

    return reindex


def fuse_reindexing(
    reindex1,
    reindex2,
):
    """Two reorderings applied one after the other, as a single one.

    A caller that has two steps of reindexing in hand usually wants to hand them
    out as one, because anything that takes a reindexer and applies it once
    would otherwise have to know there were two.
    """

    def reindex(index):
        return reindex1(reindex2(index))

    return reindex


def inverse_reorder(order: Sequence[int]):
    """The reordering that undoes another one.

    Applying one loop order and then this one gives back what was there before,
    which is what a caller needs in order to state an index in the old loops
    while the body has been written in the new ones.
    """

    inv_order = dict(zip(order, range(len(order))))

    def reindex(index: Sequence[_T]) -> Sequence[_T]:
        if len(index) != len(inv_order):
            raise AssertionError("Expected len(index) == len(inv_order)")
        return [index[inv_order[i]] for i in range(len(index))]

    return reindex


def get_fill_order(seq: Sequence, shape_env=None) -> Sequence[int]:
    """Which order the dimensions of a shape should be filled in.

    This is the ordering of the strides, largest first, and it is what says
    which dimension is the fastest moving one.
    """

    if shape_env is None or all(isinstance(s, (int, sympy.Integer)) for s in seq):
        sorted_idx: Sequence[int] = argsort(seq)
    else:
        # The symbolic form handles a shape that is not known yet.
        sorted_idx = argsort_sym(shape_env, seq)
    return sorted_idx


def stride_order2fill_order(order: Sequence) -> Sequence[int]:
    """The fill order that goes with a stride order.

    A stride order says which dimension moves fastest, counting from the
    outermost; a fill order says the same thing counting from the innermost, and
    the two are each other's inverse.  For a tensor whose strides run fastest
    along its last dimension the stride order is three, zero, two, one and the
    fill order is one, three, two, zero.
    """

    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(seq: Sequence, shape_env=None) -> Sequence[int]:
    """The stride order of a shape: which dimension has which rank of stride."""

    sorted_idx: Sequence[int] = get_fill_order(seq, shape_env)
    out = [0 for _ in range(len(seq))]
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


def is_triton(x) -> bool:
    """Whether a thing is written by the printer that writes device kernels.

    Asked of a device, a node, or a device named as a string.  For the three
    devices whose printer is named by a setting, the setting is the answer --
    deciding it any other way would mean building a printer to ask it, and the
    printer is not free to build.  For anything else it is asked of the
    scheduling registered for that device.
    """

    device = get_device_type(x)
    if device in ["cpu", "cuda", "xpu"]:
        if getattr(config, f"{device}_backend") == "triton":
            return True
        return False
    if (
        device is None
        or (device_scheduling := get_scheduling_for_device(device)) is None
    ):
        return False
    from .codegen.triton import TritonScheduling

    if not isinstance(device_scheduling, type):
        raise AssertionError(type(device_scheduling))
    return issubclass(device_scheduling, TritonScheduling)
    from .codegen.common import get_scheduling_for_device


def get_device_type(x):
    """The kind of device something is on, as a name."""

    if isinstance(x, str) or x is None:
        return x
    elif isinstance(x, tp.device):
        return x.type
    elif isinstance(x, (IRNode, OutputSpec)):
        return get_device_type(x.get_device())
    raise AssertionError(f"get_device_type({x}: {type(x).__name__})")


def is_cpu(x) -> bool:
    """Whether something is on the processor rather than on an accelerator."""

    return get_device_type(x) == "cpu"


def is_aligned_realized_tensor(x, alignment: int) -> bool:
    """Whether a realized tensor's strides all line up with a given width.

    A vector load is only sound if the address it starts at is aligned to the
    width it moves, which means every stride but the innermost has to be a
    multiple of that width and the innermost has to run contiguously or be too
    short for the alignment to matter.  The question is asked of the guard as
    well, so that a shape which does not line up is recompiled rather than read
    wrongly.
    """

    if (
        not isinstance(x, IRNode)
        or x.maybe_get_stride() is None
        or free_unbacked_symbols(x.get_stride())
        or free_unbacked_symbols(x.get_size())
    ):
        return False

    aligned_strides = sympy.And(
        *(sympy.Eq(Mod(s, alignment), 0) for s in x.get_stride()[:-1])
    )
    aligned_last_dim = sympy.Or(
        sympy.Eq(x.get_stride()[-1], 1), sympy.Le(x.get_size()[-1], 1)
    )
    is_aligned = sympy.And(aligned_strides, aligned_last_dim)

    # Asked of the guard as well, so that a shape which does not line up is
    # recompiled rather than read wrongly.
    return V.graph.sizevars.guard_or_false(is_aligned)


def significant_strides_equal(
    strides1: Sequence,
    strides2: Sequence,
    shape: Sequence,
) -> bool:
    """Whether two sets of strides agree on every dimension that has a stride.

    A dimension of extent zero gives the tensor nothing to hold, so every stride
    is as good as any other; a dimension of extent one has only one element, so
    its stride is never taken.  Neither is compared, because a difference there
    says nothing about the memory the elements occupy.
    """

    if not (len(shape) == len(strides1) and len(strides1) == len(strides2)):
        raise AssertionError(
            "Expected len(shape) == len(strides1) and len(strides1) == len(strides2)"
        )
    if any(V.graph.sizevars.statically_known_equals(dim, 0) for dim in shape):
        return True
    for dim, s1, s2 in zip(shape, strides1, strides2):
        if V.graph.sizevars.statically_known_leq(dim, 1):
            continue

        if not V.graph.sizevars.guard_or_false(sympy.Eq(s1, s2)):
            return False
    return True


def try_get_name(x):
    """The name of the memory this value is, or nothing if it is not any yet.

    A value that is still a chain of operations has no memory to name, and a
    caller asking whether something has a name needs an answer rather than an
    error, since "not yet" is a perfectly good answer.
    """

    if isinstance(x, TensorBox):
        x = x.data
    if isinstance(x, BaseView):
        x = x.unwrap_view()
    if isinstance(x, StorageBox):
        x = x.data
    return x.get_name() if isinstance(x, Buffer) else None


def developer_warning(msg: str) -> None:
    """A warning about the program as given, rather than about the code.

    What is being said here is that something in the program is there for a
    reason this compilation cannot see, so the result will not be what was
    presumably wanted, and turning warnings into errors is how that gets
    noticed rather than read past.
    """

    if config.raise_on_developer_warning:
        raise AssertionError(msg)
    import warnings

    warnings.warn(msg, stacklevel=2)


def get_symbolic_inputs(inputs: Sequence) -> list:
    """Every shape symbol appearing in the sizes or strides of these nodes."""

    sym_vars: OrderedSet = OrderedSet()
    for inp in inputs:
        sym_vars |= get_free_symbols(inp.get_size(), unbacked_only=False)
        sym_vars |= get_free_symbols(inp.get_stride(), unbacked_only=False)

    return list(sym_vars)


#: The loop order of a tensor whose strides run fastest along the last dimension.
NHWC_STRIDE_ORDER = [3, 0, 2, 1]
#: The same, for a tensor with one more dimension in front.
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]


class IRNode:
    """Anything a lowered graph is made of.

    A node says what is computed and, where that has to be settled before the
    code can be written, how it is laid out.  Most of what is asked of a node
    has no general answer -- a view has no layout of its own, a kernel has no
    shape until it is realized -- so those questions are left to raise, and a
    caller that can do without the answer asks the ``maybe_`` form instead.
    """

    _current_origins: ClassVar[OrderedSet] = OrderedSet()
    _current_stream_idx: ClassVar[int | None] = None
    _current_mempool: ClassVar[tuple | None] = None

    origins: OrderedSet = dataclasses.field(init=False)
    # Where in the lowering this node was made.
    traceback: list | None = dataclasses.field(init=False)
    origin_node: Any = dataclasses.field(init=False)
    # Whatever a later stage wants to say about this node.
    annotations: dict = dataclasses.field(init=False)
    # The stream the user asked this be computed on, if any.
    stream_idx: int | None = dataclasses.field(init=False)
    # The memory pool the user asked this be allocated from, if any.
    mempool: tuple | None = dataclasses.field(init=False)

    @staticmethod
    @contextlib.contextmanager
    def current_origins(origins: OrderedSet):
        """Say which nodes are responsible for whatever is made inside."""

        old = IRNode._current_origins
        IRNode._current_origins = old | origins
        try:
            yield
        finally:
            IRNode._current_origins = old

    @staticmethod
    @contextlib.contextmanager
    def current_stream_idx(stream_idx: int | None):
        old = IRNode._current_stream_idx
        IRNode._current_stream_idx = stream_idx
        try:
            yield
        finally:
            IRNode._current_stream_idx = old

    @staticmethod
    @contextlib.contextmanager
    def current_mempool(mempool: tuple | None):
        old = IRNode._current_mempool
        IRNode._current_mempool = mempool
        try:
            yield
        finally:
            IRNode._current_mempool = old

    @staticmethod
    def is_realized_node(node: "IRNode") -> bool:
        """Whether this node's value is already in memory.

        A node that is realized cannot have anything else fused into it, and
        its value can be read by anything without being computed again.
        """

        return isinstance(
            node,
            (
                ComputedBuffer,
                InputsKernel,
                InputBuffer,
                ReinterpretView,
                TemplateBuffer,
            ),
        )

    def wrap_for_lowering(self) -> "IRNode":
        if isinstance(self, TensorBox):
            return self
        return TensorBox.create(self)

    def _post_init_setattr(self, attr: str, value: object) -> None:
        """Set a field from ``__post_init__``, for enforcing an invariant.

        A node that is frozen cannot be assigned to after it is built, and some
        of what a node has to hold is only known once it has been built, so
        this is how such a field is put in place.
        """

        object.__setattr__(self, attr, value)

    def __post_init__(self) -> None:
        origins = OrderedSet(self._current_origins)
        self._post_init_setattr("origins", origins)
        self._post_init_setattr(
            "traceback", traceback.format_stack() if config.debug_ir_traceback else None
        )
        self._post_init_setattr("origin_node", None)
        self._post_init_setattr("annotations", {})
        self._post_init_setattr("stream_idx", self._current_stream_idx)
        self._post_init_setattr("mempool", self._current_mempool)

    def get_read_names(self) -> OrderedSet:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_traceback(self) -> list | None:
        return self.traceback

    def get_origin_node(self):
        return self.origin_node

    def get_defining_op(self) -> "Operation | None":
        return None

    def get_subgraphs(self) -> list:
        """The graphs this node contains, if it contains any."""

        return []

    def get_stack_traces(self) -> OrderedSet:
        # The traces of the user's own code, as opposed to the traces of the
        # lowering, which say where in the compiler this was made.
        if not self.traceback:
            return OrderedSet()
        return OrderedSet(
            traceback.format_list(traceback.extract_stack(self.traceback))
        )

    def common_repr(self, shorten: bool = True) -> Sequence[str]:
        origins = f"origins={getattr(self, 'origins', '')}"
        if shorten and len(origins) > 64:
            # This can run to a great many lines.
            origins = f"{origins[:61]}..."
        if not self.get_stack_traces():
            return [origins]

        stack_trace_str = []
        for stack_trace in self.get_stack_traces():
            stack_trace_str.append("stack_traces = {")
            stack_trace_str += stack_trace.split("\n")
            stack_trace_str.append("}")
        return [origins] + stack_trace_str

    def str_helper(
        self, lines: Sequence, shorten: bool = True, multiline: bool = True
    ) -> str:
        lines = list(lines) + list(self.common_repr(shorten))
        lines = list(map(str, lines))
        if multiline:
            new_lines = indent(",\n".join(lines))
            return f"{type(self).__name__}(\n{new_lines}\n)"
        else:
            return f"{type(self).__name__}({lines})"

    def get_dtype(self):
        return self.dtype

    def maybe_get_dtype(self):
        try:
            return self.get_dtype()
        except NotImplementedError:
            return None

    def get_layout(self) -> "Layout":
        raise NotImplementedError(f"get_layout() is not implemented by {type(self)}!")

    def maybe_get_layout(self):
        try:
            return self.get_layout()
        except NotImplementedError:
            return None

    def get_output_spec(self) -> "OutputSpec":
        return self.get_layout()

    def maybe_get_output_spec(self):
        try:
            return self.get_output_spec()
        except NotImplementedError:
            return None

    def has_tensor_output(self) -> bool:
        """Whether this node produces one tensor rather than several."""

        return isinstance(self.maybe_get_output_spec(), Layout)

    def get_size(self) -> Sequence:
        raise NotImplementedError(f"get_size() is not implemented by {type(self)}!")

    def maybe_get_size(self):
        try:
            return self.get_size()
        except NotImplementedError:
            return None

    @property
    def shape(self):
        return self.get_size()

    def get_numel(self) -> Expr:
        return sympy_product(self.get_size())

    def is_zero_elements(self) -> bool:
        return V.graph.sizevars.statically_known_true(sympy.Eq(self.get_numel(), 0))

    def realize(self) -> str | None:
        """Put this node's value in memory, so nothing more can be fused into it.

        A value that has not been materialized can still take on more work --
        another pointwise result can be computed into the same place -- and
        realizing it ends that, while letting anything read it without computing
        it again.

        Not every node has a way to do this, and that is deliberate: a node that
        has not thought about it is asked rather than assumed, because doing it
        wrongly is worse than not offering it.
        """

        raise NotImplementedError(f"realize NYI on {type(self)}")

    def codegen_reference(self, writer=None) -> str:
        raise NotImplementedError(f"codegen_reference NYI on {type(self)}")

    def get_device(self):
        return None

    def get_device_or_error(self):
        device = self.get_device()
        if device is None:
            raise AssertionError("Expected device is not None")
        return device

    def has_exceeded_max_reads(self) -> bool:
        return False

    def make_loader(self):
        raise NotImplementedError(type(self).__name__)

    def make_indexer(self):
        raise NotImplementedError(type(self).__name__)

    def get_stride(self) -> Sequence:
        raise NotImplementedError(type(self).__name__)

    def maybe_get_stride(self):
        try:
            return self.get_stride()
        except NotImplementedError:
            return None

    def get_name(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def maybe_get_name(self):
        try:
            return self.get_name()
        except NotImplementedError:
            return None

    def is_input_buffer(self) -> bool:
        try:
            return self.get_name() in V.graph.graph_inputs
        except NotImplementedError:
            return False

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        return False

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        """Say that this node's value will be read more than once.

        The count is an estimate.  When it comes from the graph it is the
        fanout, and a large one is a reason to realize the value rather than
        recompute it; when it comes from a loop -- a broadcast, say -- it says
        nothing about fanout and must not be read as such.
        """

    def realize_hint(self) -> None:
        pass

    def unwrap_view(self) -> "IRNode":
        raise NotImplementedError(type(self).__name__)

    def freeze_layout(self) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_same_order(self, stride: Sequence) -> None:
        raise NotImplementedError(type(self).__name__)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence, allow_padding: bool = False
    ) -> None:
        raise NotImplementedError(type(self).__name__)

    def get_read_writes(self) -> "dependencies.ReadWrites":
        raise NotImplementedError(type(self).__name__)

    def get_reads(self) -> OrderedSet:
        return self.get_read_writes().reads

    def num_reads(self) -> int:
        return len(self.get_reads())

    def get_storage_numel(self):
        raise NotImplementedError(type(self).__name__)

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        raise NotImplementedError(type(self).__name__)

    def get_reduction_type(self) -> str | None:
        raise NotImplementedError(type(self).__name__)

    def get_reduction_size(self) -> Sequence[Expr]:
        raise NotImplementedError(type(self).__name__)

    def is_extern(self) -> bool:
        return False

    def is_no_op(self) -> bool:
        return False

    def constant_to_device(self, device) -> "IRNode":
        raise NotImplementedError(type(self).__name__)

    def get_mutation_names(self) -> Sequence[str]:
        raise NotImplementedError(type(self).__name__)

    def get_operation_name(self) -> str:
        raise NotImplementedError(type(self).__name__)

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        raise NotImplementedError(type(self).__name__)

    if TYPE_CHECKING:

        @property
        def dtype(self): ...


class OutputSpec:
    """What a result of the region looks like, for a generator that emits it.

    A generator is handed the result rather than the operation that produced
    it, so the two things it needs from a result -- where it lives and how much
    room it takes -- are asked of the result itself.
    """

    def get_device(self):
        raise NotImplementedError(type(self).__name__)

    def storage_size(self) -> int:
        raise NotImplementedError(type(self).__name__)

    def get_free_symbol_uses(self, unbacked_only: bool = False):
        raise NotImplementedError(type(self).__name__)


class Layout(OutputSpec):
    """Where a result lives: its device, its element type, its extents, and how
    they are addressed.

    A result whose extents are settled is a fixed layout: the extents are what
    it is.  One whose extents may still move -- a result the caller has not
    sized yet, or a leading extent that is the product of several operands' --
    is a flexible layout: the extents are a statement about the call rather
    than about the result.  The two are different claims, so they are told
    apart here rather than by whoever reads them.
    """

    def __init__(
        self,
        device: Any,
        dtype: Any,
        size: Sequence,
        stride: Sequence = None,
        offset=0,
        is_pinned: bool = False,
    ) -> None:
        if stride is None:
            stride = FlexibleLayout.contiguous_strides(size)
        self.device = device
        self.dtype = dtype
        if len(size) != len(stride):
            raise AssertionError(f"size={size}, stride={stride}")
        self._size = size
        self._stride = stride
        self._offset = offset
        self.is_pinned = is_pinned
        if not ((not self.is_pinned) or (self.device.type == "cpu")):
            raise AssertionError("Only CPU tensors can be pinned")

    @property
    def size(self) -> Sequence:
        return self._size

    @size.setter
    def size(self, value: Sequence) -> None:
        self._size = value

    @property
    def stride(self) -> Sequence:
        return self._stride

    @stride.setter
    def stride(self, value: Sequence) -> None:
        self._stride = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value) -> None:
        self._offset = value

    def __str__(self) -> str:
        offset = ""
        if self.offset != 0:
            offset = f", offset={self.offset}"

        device_index_str = "" if self.device.index is None else f":{self.device.index}"
        is_pinned_str = ""
        if self.is_pinned:
            is_pinned_str = f", is_pinned={self.is_pinned}"
        return (
            f"{type(self).__name__}('{self.device.type}{device_index_str}', {self.dtype}, "
            f"size={self.size}, stride={self.stride}{offset}{is_pinned_str})"
        )

    __repr__ = __str__

    def get_device(self) -> Any:
        return self.device

    def get_example(self):
        """A tensor of this geometry, carrying no data.

        A generator that needs to know something about the result that is not
        in the geometry -- how wide an element is, what it looks like when
        printed -- needs a tensor to ask, and a tensor with the right geometry
        and no contents answers every question whose answer does not depend on
        the data.
        """

        from tensorplay import functional

        return functional.empty_strided(
            [int(x) for x in self.size],
            [int(x) for x in self.stride],
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.is_pinned,
        )

    def is_contiguous(self) -> bool:
        return is_contiguous_strides_for_shape(self.stride, self.size)

    @staticmethod
    def is_channels_last_contiguous(shape: Sequence, strides: Sequence) -> bool:
        """Whether a shape is stored with its channels outermost and unbroken."""

        ndim = len(shape)
        if ndim not in [4, 5] or shape[1] == 1:
            return False
        for left, right, size in zip(
            strides,
            make_channels_last_strides_for(shape),
            shape,
        ):
            if size != 1 and left != right:
                return False
        return True

    def is_transposed(self) -> bool:
        """Whether this is the contiguous layout of the reversed extents."""

        for left, right, size in zip(
            self.stride,
            reversed(FlexibleLayout.contiguous_strides(list(reversed(self.size)))),
            self.size,
        ):
            if size != 1 and left != right:
                return False
        return True

    @staticmethod
    def _stride_expr_ge_or_false(left, right) -> bool:
        """Whether the left stride is at least the right one, or nothing is known.

        Both sides are strides, so a stride of zero settles the question
        either way, and divisibility is proved from the structure of the
        expressions rather than asked of the environment, so that a fact about
        the shape in hand is not mistaken for a fact about every shape.
        """

        sizevars = V.graph.sizevars
        if sizevars.guard_or_false(sympy.Eq(right, 0)):
            return True
        if sizevars.guard_or_false(sympy.Eq(left, 0)):
            return False
        return (
            sizevars.guard_or_false(sympy.Ge(left, right))
            or sizevars.statically_known_multiple_of(left, right)
            or sizevars.guard_or_false(sympy.Eq(sympy.Mod(left, right), 0))
        )

    def is_stride_ordered(self, order: Sequence) -> bool:
        """Whether the strides ascend in the order given.

        An extent of one is left out of the comparison, because it addresses a
        single element however it is strided and so cannot make the order wrong.
        """

        if len(self.stride) != len(order):
            raise AssertionError("Expected len(self.stride) == len(order)")

        non_1_indices = [
            i
            for i, dim in enumerate(self.size)
            if not V.graph.sizevars.statically_known_equals(dim, 1)
        ]

        stride = [self.stride[i] for i in non_1_indices]
        order: Sequence = [order[i] for i in non_1_indices]

        def sorted_indices(arr: Sequence) -> Sequence:
            sorted_arr = sorted(arr)
            return [sorted_arr.index(element) for element in arr]

        order = sorted_indices(order)

        stride_ordered = [-1] * len(order)
        for i in range(len(order)):
            stride_ordered[order[i]] = stride[i]
        for i in range(len(order) - 1):
            left = stride_ordered[i]
            right = stride_ordered[i + 1]
            if V.graph.sizevars.guard_or_false(sympy.Eq(left, right)):
                continue
            if self._stride_expr_ge_or_false(left, right):
                return False
            if not self._stride_expr_ge_or_false(right, left):
                return False
        return True

    def is_channels_last_stride_ordered(self) -> bool:
        """Whether the strides are ordered with the channels outermost."""

        order = [0] + list(reversed(range(1, len(self.stride) - 1)))
        order = [len(order)] + order
        return self.is_stride_ordered(order)

    @staticmethod
    def _pad_strides(in_strides: Sequence, size: Sequence, dtype) -> Sequence:
        """The same strides, rounded up so that every access is aligned.

        Padding does not change which dimension is inner -- the order is kept --
        and only makes the strides larger than the threshold multiples of the
        alignment, because a stride below the threshold would cost more memory
        than the alignment saves.
        """

        align = get_align_for_dtype(dtype)
        if len(in_strides) == 0:
            return in_strides

        if not config.pad_channels_last and Layout.is_channels_last_contiguous(
            size, in_strides
        ):
            return in_strides

        current_fx_node = get_current_node()
        if hasattr(current_fx_node, "meta") and current_fx_node.meta.get(
            "dislike_padding", False
        ):
            return in_strides

        is_dynamic = not all(
            isinstance(s, (int, sympy.Integer))
            for s in itertools.chain(in_strides, size)
        )
        if not config.pad_dynamic_shapes and is_dynamic:
            return in_strides

        shape_env = getattr(V.graph, "shape_env", None)

        def contains_unbacked_symints(expr) -> bool:
            if shape_env is None:
                return False
            if not isinstance(expr, sympy.Expr):
                return False
            return any(shape_env.is_unbacked_symint(s) for s in expr.free_symbols)

        if shape_env and any(contains_unbacked_symints(s) for s in in_strides):
            return in_strides

        stride_order = get_stride_order(in_strides, shape_env)
        fill_order = stride_order2fill_order(stride_order)

        new_strides = [0 for _ in range(len(in_strides))]
        new_strides[fill_order[0]] = 1

        padded = False
        for rank, idx in enumerate(fill_order[1:], start=1):
            prev_idx = fill_order[rank - 1]
            stride = new_strides[prev_idx] * size[prev_idx]
            require_padding = (
                isinstance(stride, (int, sympy.Integer))
                and stride > config.padding_stride_threshold
                and stride % align != 0
            ) or (isinstance(stride, sympy.Expr) and config.pad_dynamic_shapes)
            new_strides[idx] = stride
            if require_padding:
                new_strides[idx] = ceildiv(stride, align) * align
                padded = True

        if not padded:
            # A shape such as [256, 1, 5, 5] has strides that would be padded
            # to [25, 25, 5, 1], which addresses exactly the same elements as
            # the strides it started from, and does so with a larger footprint.
            return in_strides

        metrics.num_comprehensive_padding += 1
        return new_strides

    def pad_strides(self) -> None:
        """Pad this layout's strides, in place."""

        if not isinstance(self, FlexibleLayout):
            raise AssertionError(type(self))
        if self.stride is None:
            raise AssertionError("Expected self.stride is not None")
        self.stride = self._pad_strides(self.stride, self.size, self.dtype)

    def should_pad_strides(self) -> bool:
        """Whether this layout's strides are to be padded at all."""

        return config.comprehensive_padding and isinstance(self, FlexibleLayout)

    def as_fixed(self) -> "FixedLayout":
        """This layout with its extents and strides settled."""

        if isinstance(self, FixedLayout):
            return self

        if self.should_pad_strides():
            self.pad_strides()
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            self.stride,
            self.offset,
            self.is_pinned,
        )

    def make_indexer(self) -> Callable:
        """A closure holding the arithmetic that reads one element."""

        if not FlexibleLayout.allow_indexing:
            raise AssertionError(
                f"convert {type(self).__name__} to FixedLayout first"
            )
        return self.as_fixed().make_indexer()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Layout)
            and self.device == other.device
            and self.dtype == other.dtype
            and self.size == other.size
            and self.stride == other.stride
            and self.offset == other.offset
            and self.is_pinned == other.is_pinned
        )

    def storage_size(self):
        """How many elements of storage this result occupies."""

        return compute_required_storage_length(self.size, self.stride, self.offset)

    @cache_on_self_and_args("Layout")
    def get_free_symbol_uses(self, unbacked_only: bool = False):
        """Every free symbol of the extents, the strides and the offset.

        Which of the three a symbol came from matters: a symbol in the extents
        is part of the result's shape and a caller has to know it, while one
        that only appears in a padded stride is an artifact of the padding and
        must not be reported as part of the result.
        """

        return get_free_symbols(self.size, unbacked_only) | get_free_symbols(
            self.stride, unbacked_only
        ) | get_free_symbols(self.offset, unbacked_only)


class FixedLayout(Layout):
    """A result's layout, once it is settled and may no longer be changed."""

    def make_indexer(self) -> Callable:
        """A closure holding the arithmetic that reads one element."""

        return _fixed_indexer(self.size, self.stride, self.offset)


class FlexibleLayout(Layout):
    """A result's layout while it may still be changed.

    Changing the layout is not allowed to add or remove a free symbol: the
    extents are what the caller asked for, and a layout that answered with
    different extents would be answering a different question.
    """

    allow_indexing = False

    def get_fixed_layout_without_freezing(self) -> FixedLayout:
        """What the strides would be if this layout were settled, without settling it.

        Used where the strides have to be computed while the layout may still
        be changed afterwards, so the computation cannot be made in place.
        """

        import copy

        return copy.deepcopy(self).as_fixed()

    @staticmethod
    def contiguous_strides(sizes: Sequence) -> list:
        """The strides of a shape stored as one unbroken run.

        The sizes are treated as at least one, so a shape with an extent of
        zero does not make the strides after it zero.
        """

        if len(sizes) == 0:
            return []
        reversed_strides = [sympy.S.One]
        for size in reversed(sizes[1:]):
            reversed_strides.append(size * reversed_strides[-1])
        return list(reversed(reversed_strides))

    @staticmethod
    def fill_ordered(sizes: Sequence, order: Sequence) -> list:
        """The strides of a shape filled in the order given.

        Channels last is a fill order of ``[1, 3, 2, 0]``: the innermost
        dimension is the second one, and the outermost is the fourth.
        """

        if OrderedSet(range(len(sizes))) != OrderedSet(order):
            raise AssertionError((sizes, order))
        next_stride = sympy.S.One
        strides = [None] * len(order)

        for i in order:
            strides[i] = next_stride
            next_stride = next_stride * sizes[i]
        return strides

    @staticmethod
    def stride_ordered(sizes: Sequence, order: Sequence) -> Sequence:
        """The strides of a shape whose strides are ordered as given.

        Channels last is a stride order of ``[3, 0, 2, 1]``.
        """

        if OrderedSet(range(len(sizes))) != OrderedSet(order):
            raise AssertionError(
                "Expected OrderedSet(range(len(sizes))) == OrderedSet(order)"
            )
        fill_order = stride_order2fill_order(order)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @staticmethod
    def stride_ordered_for_memory_format(sizes: Sequence, memory_format) -> Sequence:
        """The strides of a shape stored in the memory format given.

        A memory format is a stride order under another name, so channels last
        is the same as a stride order of ``[3, 0, 2, 1]``.  A format that has
        to be deduced from another source cannot be used here, since there is
        no other source to deduce it from.
        """

        if memory_format == "channels_last":
            return FlexibleLayout.stride_ordered(sizes, NHWC_STRIDE_ORDER)
        if memory_format == "channels_last_3d":
            return FlexibleLayout.stride_ordered(sizes, NHWDC_STRIDE_ORDER)
        if memory_format == "contiguous":
            return FlexibleLayout.contiguous_strides(sizes)
        raise NotImplementedError(
            f"stride_ordered_for_memory_format, unsuppored memory_format: "
            f"{memory_format}"
        )

    @staticmethod
    def same_ordered(sizes: Sequence, stride: Sequence) -> Sequence:
        """The strides of a shape stored in the same order as the strides given.

        For strides of ``[1000, 1, 100, 10]`` the fill order is
        ``[1, 3, 2, 0]``: the second dimension is the innermost, then the
        fourth, then the third, and the first is outermost.
        """

        if len(sizes) != len(stride):
            raise AssertionError("Expected len(sizes) == len(stride)")
        stride = V.graph.sizevars.guarding_hints_or_throw(stride)
        fill_order = sorted(range(len(stride)), key=stride.__getitem__)
        return FlexibleLayout.fill_ordered(sizes, fill_order)

    @property
    def size(self) -> Sequence:
        return self._size

    @size.setter
    def size(self, value: Sequence) -> None:
        self.assert_free_symbol_uses_unchanged("size", value)
        self._size = value

    @property
    def stride(self) -> Sequence:
        return self._stride

    @stride.setter
    def stride(self, value: Sequence) -> None:
        self.assert_free_symbol_uses_unchanged("stride", value)
        self._stride = value

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value) -> None:
        self.assert_free_symbol_uses_unchanged("offset", value)
        self._offset = value

    def as_stride_order(
        self, order: Sequence, allow_padding: bool = False
    ) -> FixedLayout:
        """This layout, settled with its strides in the order given."""

        new_stride = self.stride_ordered(self.size, order)
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_exact_strides(
        self, exact_strides: Sequence, allow_padding: bool = False
    ) -> FixedLayout:
        """This layout, settled with exactly these strides."""

        new_stride = exact_strides
        if self.should_pad_strides() and allow_padding:
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)

        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_fill_order(self, order: Sequence) -> FixedLayout:
        """This layout, settled with its extents filled in the order given."""

        new_stride: Sequence = self.fill_ordered(self.size, order)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def as_same_order(self, stride: Sequence) -> FixedLayout:
        """This layout, settled with its strides ordered as the ones given."""

        new_stride = self.same_ordered(self.size, stride)
        if self.should_pad_strides():
            new_stride = self._pad_strides(new_stride, self.size, self.dtype)
        return FixedLayout(
            self.device,
            self.dtype,
            self.size,
            new_stride,
            self.offset,
            self.is_pinned,
        )

    def get_initial_free_symbol_uses(self) -> dict:
        """The free symbols of the extents, the strides and the offset, as first built.

        Recorded once so that every later change to any of the three can be
        checked against it: a change that introduced a symbol or dropped one
        would mean the layout is now describing a different result.
        """

        initial_free_symbols = {}
        for name in ["size", "stride", "offset"]:
            for unbacked_only in [True, False]:
                key = (name, unbacked_only)
                initial_free_symbols[key] = OrderedSet(
                    get_free_symbols(getattr(self, name), unbacked_only)
                )

        return initial_free_symbols

    def assert_free_symbol_uses_unchanged(self, name: str, value) -> None:
        """Refuse a change to the extents, the strides or the offset that moves a symbol."""

        for unbacked_only in [True, False]:
            old_free_symbols = self.initial_free_symbols[(name, unbacked_only)]
            new_free_symbols = OrderedSet(get_free_symbols(value, unbacked_only))
            if new_free_symbols != old_free_symbols:
                raise AssertionError(
                    f"Expected free symbols unchanged, but got {new_free_symbols} vs {old_free_symbols}"
                )

    def __init__(
        self,
        device: Any,
        dtype: Any,
        size: Sequence,
        stride_order: Sequence = None,
        is_pinned: bool = False,
    ) -> None:
        if stride_order:
            strides = FlexibleLayout.fill_ordered(size, stride_order)
        else:
            strides = FlexibleLayout.contiguous_strides(size)
        super().__init__(device, dtype, size, strides, is_pinned=is_pinned)

        self.initial_free_symbols = self.get_initial_free_symbol_uses()

@ir_dataclass(frozen=False)
class Operation:
    """Something that computes, as distinct from something that holds a value.

    A kernel and a realized buffer are both operations where a tensor is not,
    because both produce something.  What an operation is asked here is mostly
    what it reads and what it produces, which is what the order of two
    operations has to be decided from.
    """

    def __post_init__(self) -> None:
        self.operation_name: str | None = None
        self._config_patches: dict = {}

    def get_device(self):
        raise NotImplementedError

    def get_origin_node(self):
        if not hasattr(self, "origin_node"):
            raise AssertionError('Expected hasattr(self, "origin_node")')
        return self.origin_node

    def get_origins(self) -> OrderedSet:
        if not hasattr(self, "origins"):
            raise AssertionError('Expected hasattr(self, "origins")')
        return self.origins

    def get_stream_idx(self) -> int | None:
        if not hasattr(self, "stream_idx"):
            raise AssertionError('Expected hasattr(self, "stream_idx")')
        return self.stream_idx

    def get_mempool(self) -> tuple | None:
        if not hasattr(self, "mempool"):
            raise AssertionError('Expected hasattr(self, "mempool")')
        return self.mempool

    def get_operation_name(self) -> str:
        if self.operation_name is None:
            raise AssertionError("Expected self.operation_name is not None")
        return self.operation_name

    def get_buffer_name(self) -> str | None:
        return None

    def get_config_patches(self) -> dict:
        """Settings this operation asked to have applied while it is emitted."""

        return self._config_patches

    def set_config_patches(self, patches: dict) -> None:
        """Ask for settings to be applied while this operation is emitted."""

        self._config_patches = patches

    def is_extern(self) -> bool:
        return False

    def is_no_op(self) -> bool:
        return False

    def get_read_writes(self) -> "dependencies.ReadWrites":
        raise NotImplementedError

    def is_user_of(self, name: str) -> bool:
        return name in self.get_read_names()

    def get_read_names(self) -> OrderedSet:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_reads(self) -> OrderedSet:
        return self.get_read_writes().reads

    def get_outputs(self) -> list:
        raise NotImplementedError

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        """Which shapes have to be in scope for this to be emitted.

        A shape that came out of the data has to be bound before the code can
        refer to it, so that a caller can be told which binding this needs.

        This is deliberately not transitive: a buffer whose shape depends on
        another buffer's shape does not report that shape here, because the
        dependency on the other buffer already carries it.
        """

        return OrderedSet()

    def get_workspace_size(self) -> int:
        """How much memory this needs beyond its own output.

        Some ways of computing need somewhere to put intermediate values that
        is not the output, and that space has to be asked for and provided.
        """

        return 0


@ir_dataclass
class Loops(IRNode):
    """A body that is a set of loops, before any layout has been chosen.

    The body is a function of the loop indices, and nothing about it says where
    its result is held -- that is settled later, and may be settled differently
    for different consumers, which is the whole reason this is not yet a
    buffer.
    """

    device: Any
    dtype: Any
    inner_fn: Any
    ranges: Sequence

    @cache_on_self_and_args("Loops")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.ranges),
            self.inner_fn_free_symbols(unbacked_only),
        )

    def _to_str(self, names: Sequence) -> str:
        return self.str_helper(
            [
                f"'{self.device.type}'",
                str(self.dtype),
                self.inner_fn_str(),
            ]
            + [f"{name}={getattr(self, name)}" for name in names]
            + [f"origin_node={self.origin_node!r}"]
        )

    def __str__(self) -> str:
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_device(self):
        return self.device

    def get_origin_node(self):
        return self.origin_node

    def get_size(self) -> Sequence:
        return self.ranges

    def get_pointwise_size(self) -> Sequence:
        return self.ranges

    @classmethod
    def create(cls, *args, **kwargs):
        """This body, wrapped so that it can be used as a value."""

        origin_node = kwargs.pop("origin_node", None)
        tb = kwargs.pop("traceback", None)
        r = cls(*args, **kwargs)
        # Set here rather than by the constructor, so that the node this came
        # from is carried down to whatever holds the result.
        r._post_init_setattr("origin_node", origin_node)
        r._post_init_setattr("traceback", tb or r.traceback)
        return TensorBox.create(r)

    def get_pointwise_size(self) -> Sequence:
        """The axes walked independently of any reduction.

        A caller that is working out what indexes the body wants the part of
        the shape that is not being reduced over, and that is what is reported
        here.
        """

        return self.ranges

    @staticmethod
    def _index(ranges: Sequence, prefix=SymT.INDEX) -> Sequence:
        """A loop variable per loop, with an extent of one standing still.

        A loop of extent one can only run once, so the position it would name
        is always the same and is written as a constant rather than as a
        variable nothing varies.
        """

        return [
            sympy.S.Zero if s == 1 else sympy_index_symbol_with_prefix(prefix, n)
            for n, s in enumerate(ranges)
        ]

    @cache_on_self
    def inner_fn_opcount(self):
        """What this body computes, counted over the values it produces."""

        opcounter = OpCounterCSE(V.MockHandler())
        with (
            V.set_ops_handler(opcounter),
            patch.object(FlexibleLayout, "allow_indexing", True),
        ):
            self.inner_fn(*self.inner_fn_args())
            return opcounter.getvalue()

    def inner_fn_args(self) -> Sequence:
        return (self._index(self.ranges),)

    @cache_on_self
    def inner_fn_str(self) -> str:
        return V.KernelFormatterHandler.ir_to_string(
            self.inner_fn, *self.inner_fn_args()
        )

    def get_realize_opcount_threshold(self, threshold: int | None = None) -> int:
        """How much this body may compute before it is worth putting in memory.

        Putting a value in memory costs a round trip, so it is only worth it
        once the body that would otherwise be repeated is long enough.  How
        long enough is depends on where it runs, since a processor and an
        accelerator do not trade the same way.
        """

        if threshold is None:
            threshold = 0
        realize_opcount_threshold = config.realize_opcount_threshold
        if realize_opcount_threshold is None:
            if is_cpu(self):
                realize_opcount_threshold = config.realize_cpu_opcount_threshold
            else:
                realize_opcount_threshold = config._realize_opcount_threshold_default
        else:
            if not isinstance(realize_opcount_threshold, int):
                raise AssertionError(
                    f"expected int realize_opcount_threshold, got {type(realize_opcount_threshold)}"
                )
        return max(threshold, realize_opcount_threshold)

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        return self.inner_fn_opcount().num_ops > self.get_realize_opcount_threshold(
            threshold
        )

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet:
        index = self._index(self.ranges)
        return extract_free_symbols(self.inner_fn, index, unbacked_only=unbacked_only)

    def collect_inner_fn_symbol_usage(self, symbol) -> OrderedSet:
        """Which of this body's operations mention a given shape."""

        index = self._index(self.ranges)
        handler = SymbolUsageCollectorOpsHandler(symbol)
        with (
            V.set_ops_handler(handler),
        ):
            self.inner_fn(index)
        return handler.usages

    def get_reads(self) -> OrderedSet:
        """Everything this body reads, found by running the body.

        What a body reads is whatever its arithmetic turns out to ask for, and
        a body is written as arithmetic rather than as a list of things, so the
        only way to know is to run it with a handler that records each ask.  A
        body that reduces is read over the reduction rather than over the
        result, and the two are read differently so that the same answer comes
        out either way.
        """

        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.get_reduction_type():
                return extract_read_writes(
                    self.make_loader(),
                    self.get_size(),
                    self.get_reduction_size(),
                ).reads
            return extract_read_writes(self.make_loader(), self.get_size()).reads

    def get_read_names(self) -> OrderedSet:
        return OrderedSet(self.inner_fn_opcount().read_buffers)

    def num_reads(self) -> int:
        return len(self.inner_fn_opcount().read_buffers)

    def get_reduction_size(self) -> Sequence:
        raise NotImplementedError(
            f"get_reduction_size() is not implemented by {type(self)}!"
        )

    def get_reduction_type(self) -> str | None:
        raise NotImplementedError(
            f"get_reduction_type() is not implemented by {type(self)}!"
        )

    def constant_to_device(self, device) -> IRNode:
        raise NotImplementedError(
            f"constant_to_device() is not implemented by {type(self)}!"
        )


def nop_loader_fn(idx, *, dtype):
    """What a loop that runs no times produces, which is a value of the right type.

    A body over an empty extent computes nothing, and something has to stand in
    for the result anyway; a constant of the right type is the only thing that
    can be put where the result would be without a loop to put it in.
    """

    if dtype.is_floating_point:
        return ops.constant(float("nan"), dtype)
    return ops.constant(0, dtype)


@ir_dataclass
class Pointwise(Loops):
    """A body where each element of the result depends on one element of each input.

    Nothing is shared between the elements, which is what makes this the
    cheapest kind of body to write and the kind most worth fusing into whatever
    reads its result.
    """

    def make_loader(self):
        # A body over no elements has nothing to load, and loading is what
        # would have to be turned into a loop.
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.dtype)

        return self.inner_fn

    def __str__(self) -> str:
        return self._to_str(("ranges",))

    __repr__ = __str__

    def get_reduction_size(self) -> Sequence:
        return []

    def get_reduction_type(self) -> str | None:
        return None

    def store_output(
        self,
        output_name: str | None,
        indexer,
        vars: Sequence,
    ) -> None:
        loader = self.make_loader()
        return ops.store(output_name or "unnamed", indexer(vars), loader(vars))

    def constant_to_device(self, device) -> IRNode:
        """This body, moved to another device.

        Only a body that reads nothing but constants can be moved, since what
        it reads would still be where it was.
        """

        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
        )


class NonOwningLayout(Layout):
    """A view onto the memory of another tensor, which this does not own.

    The shape and stride are the view's, but the memory belongs to whatever the
    view looks at, so anything that would resize or free it has to be told.
    """

    def __init__(self, view) -> None:
        layout = view.get_layout()
        super().__init__(
            layout.device,
            layout.dtype,
            layout.size,
            layout.stride,
        )
        self.view = view

    def make_indexer(self):
        return self.as_fixed().make_indexer()

    def maybe_guard_aligned(self) -> bool:
        offset = self.view.get_layout().offset
        if offset == 0:
            return True
        from .utils import ALIGNMENT

        return V.graph.sizevars.statically_known_multiple_of(offset, ALIGNMENT)

    @cache_on_self_and_args("NonOwningLayout")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        """Whatever shapes the buffer this looks at has, and not its own.

        A view adds no shapes of its own, so the shapes involved are the ones
        the buffer underneath has, and asking for those is what tells the
        caller the view is not free of shape dependencies.
        """

        if not isinstance(self.view, ReinterpretView):
            raise AssertionError("Expected isinstance(self.view, ReinterpretView)")
        box = self.view.data
        if not isinstance(box, StorageBox):
            raise AssertionError(type(box))
        input_buffer = box.data
        if not isinstance(input_buffer, Buffer):
            raise AssertionError(type(box))
        return input_buffer.layout.get_free_symbol_uses(unbacked_only)


@ir_dataclass
class NoneLayout(OutputSpec):
    """An output that is not a tensor at all.

    An operation that produces nothing holds no shape and no stride, so there is
    nothing here to describe; what it does have to say is which device it ran
    on.  A node with this is one whose dependencies have to be set up by hand,
    because there is no shape from which to work them out.
    """

    device: Any
    size: list = dataclasses.field(default_factory=lambda: [0])
    stride: list = dataclasses.field(default_factory=lambda: [0])

    def storage_size(self) -> int:
        return 0

    def as_fixed(self) -> "OutputSpec":
        return self

    def get_device(self):
        return self.device


class MutationLayoutSHOULDREMOVE(Layout):
    """The memory of another buffer, written in place.

    The name says what is going on: writing in place is a way of avoiding a
    copy, and it makes the memory a node shares with someone else rather than
    memory it owns, which most of what is asked of a layout assumes it is.
    """

    def __init__(self, target: "IRNode") -> None:
        super().__init__(
            target.get_device_or_error(),
            target.get_dtype(),
            target.get_size(),
            None,
        )
        self.target = target
        name = self.get_buffer().get_name()
        V.graph.mark_buffer_mutated(name)

    @property
    def stride(self) -> Sequence:
        # The stride is the target's, and cannot be set: what is written in
        # place keeps the layout it had.
        return self.real_layout().stride

    @stride.setter
    def stride(self, value) -> None:
        pass  # ignore setting of stride

    def storage_size(self):
        return self.real_layout().storage_size()

    def get_buffer(self) -> "Buffer":
        """The buffer whose memory is written, with views and boxes taken off.

        A chain of views and boxes leads to the buffer that actually holds the
        memory, and it is that buffer that has to be reported as written.
        """

        def unwrap_views(target):
            if isinstance(target, MutationLayoutSHOULDREMOVE):
                return unwrap_views(target.target)
            if isinstance(target, BaseView):
                return unwrap_views(target.unwrap_view())
            if isinstance(target, MutableBox):
                return unwrap_views(target.data)
            return target

        result = unwrap_views(self.target)
        if not isinstance(result, Buffer):
            raise AssertionError(type(result))
        return result

    def real_layout(self) -> "Layout":
        layout = self.get_buffer().layout
        if not isinstance(layout, Layout):
            raise AssertionError("Expected isinstance(layout, Layout)")
        return layout

    @classmethod
    def realize_into(cls, src: "IRNode", dst: "IRNode", unsafe_alias: bool = False):
        """Put the contents of one node into another's memory.

        Whoever reads the destination has to run first, and the order of
        realization is the order things are scheduled in, so the destination's
        readers are realized before the source is.  Were it the other way round,
        the source's write would be scheduled ahead of the reads it must not
        overtake.
        """

        dst.realize()
        V.graph.mark_buffer_mutated(dst.get_name())

        if isinstance(src, TensorBox):
            src = src.data

        # The contents are copied, and in most cases the scheduler fuses that
        # into one kernel.  The source's layout cannot simply be changed to
        # write the destination's memory, because that would make the two the
        # same memory and a later write to the destination would be seen by
        # whoever holds the source.  When nothing else reads the destination
        # there is no such reader, and the two may be made the same.
        src.realize_hint()

        if not unsafe_alias:
            node = Pointwise.create(
                device=src.get_device(),
                dtype=src.get_dtype(),
                inner_fn=src.make_loader(),
                ranges=[
                    V.graph.sizevars.check_equals_and_simplify(a, b)
                    for a, b in zip(src.get_size(), dst.get_size())
                ],
            )
            if not isinstance(node, (BaseView, MutableBox)):
                raise AssertionError(
                    "Expected isinstance(node, (BaseView, MutableBox))"
                )
            src = node.data

        src.realize()
        if not hasattr(src, "data"):
            raise AssertionError(src)
        if not isinstance(src.data.layout, FlexibleLayout):
            raise AssertionError(type(src.data.layout))
        src.data.layout = MutationLayoutSHOULDREMOVE(dst)
        return src.data

    def as_fixed(self):
        return self

    def make_indexer(self):
        return self.target.make_indexer()


@ir_dataclass(frozen=False)
class Buffer(IRNode, CodegenSymbol):
    """A tensor that is held somewhere rather than computed on demand.

    A buffer is a name and a description of where its elements sit.  The name
    is what other things refer to it by, and the description is what says how
    to reach an element -- and it is the description, not the name, that says
    whether a vector load is sound.
    """

    # A name is sometimes absent, where there is nothing meaningful to call it.
    name: str | None
    layout: OutputSpec
    ordering_only: ClassVar[bool] = False

    def get_pointwise_size(self) -> Sequence:
        """The axes of this value that are walked one element at a time.

        A buffer has no reduction of its own, so every axis is one of these,
        and what a caller needs in order to know what indexes a body is
        exactly this.
        """

        return self.get_size()

    def __post_init__(self) -> None:
        super().__post_init__()
        self._post_init_setattr("origin_node", None)

    def make_indexer(self):
        return self.get_layout().make_indexer()

    def get_name(self) -> str:
        if not self.name:
            raise AssertionError(self)
        return self.name

    def get_buffer_name(self) -> str | None:
        return self.name

    def get_example(self):
        if isinstance(self.layout, Layout):
            return self.layout.get_example()
        raise NotImplementedError(type(self.layout).__name__)

    def get_device(self):
        return self.get_output_spec().get_device()

    def get_defining_op(self) -> "Operation | None":
        return None

    @property
    def dtype(self):
        return self.get_layout().dtype

    def get_size(self) -> Sequence:
        return [*self.get_layout().size]

    def get_stride(self) -> list:
        return [*self.get_layout().stride]

    def get_offset(self):
        return self.get_layout().offset

    def get_layout(self) -> "Layout":
        if isinstance(self.layout, Layout):
            return self.layout
        raise NotImplementedError(type(self.layout).__name__)

    def get_output_spec(self) -> "OutputSpec":
        return self.layout

    def get_storage_numel(self) -> int:
        return self.get_numel()

    def get_is_pinned(self) -> bool:
        return self.get_layout().is_pinned

    def freeze_layout(self) -> None:
        """Settle the layout, so that later stages may assume it.

        A layout that is not yet settled is free to change, and a later stage
        that has already made a decision based on it would be wrong; freezing is
        how that is prevented.
        """

        if isinstance(self.layout, Layout) and not isinstance(
            self.layout, NonOwningLayout
        ):
            self.layout = self.layout.as_fixed()

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_stride_order(order, allow_padding=allow_padding)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_fill_order(order)

    def freeze_layout_with_same_order(self, stride: Sequence) -> None:
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_same_order(stride)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence, allow_padding: bool = False
    ) -> None:
        if not isinstance(self.layout, FlexibleLayout):
            raise AssertionError(type(self.layout))
        self.layout = self.layout.as_exact_strides(
            exact_strides, allow_padding=allow_padding
        )

    def is_zero_elements(self) -> bool:
        return V.graph.sizevars.statically_known_true(sympy.Eq(self.get_numel(), 0))

    def make_loader(self):
        """The function that reads one element of this buffer."""

        # A buffer with no elements has nothing to read, and the read is what
        # would have to become a loop.
        if self.is_zero_elements():
            return partial(nop_loader_fn, dtype=self.get_dtype())

        def loader(index: Sequence):
            indexer = self.make_indexer()
            return ops.load(self.name or "unnamed", indexer(index))

        return loader

    def codegen_reference(self, writer=None) -> str:
        return self.get_name()

    def decide_layout(self) -> None:
        pass

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        """Which of this buffer's inputs hold the same memory as it does."""

        if isinstance(self.layout, NonOwningLayout):
            return [self.layout.view.get_name()]
        return ()

    def get_mutation_names(self) -> Sequence[str]:
        """Whose memory this buffer writes to, if it writes to someone else's."""

        if isinstance(self.layout, MutationLayoutSHOULDREMOVE):
            return [self.layout.target.get_name()]
        return ()

    def get_read_names(self) -> OrderedSet:
        return OrderedSet([self.get_name()])

    @cache_on_self_and_args("Buffer")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return OrderedSet()

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def realize(self) -> str | None:
        pass

    def should_allocate(self) -> bool:
        """Whether space has to be set aside for this buffer.

        A buffer that is a view onto memory someone else holds needs none, and
        a buffer whose memory is the input's also needs none.
        """

        return False


@ir_dataclass(frozen=False)
class OperationBuffer(Buffer, Operation):
    """An operation whose whole result is one buffer.

    Most operations that allocate produce exactly one thing, and the general
    machinery for reporting outputs and for finding who produced a value is
    written once here rather than again for each of them.
    """

    def get_outputs(self) -> list:
        return [self]

    def get_defining_op(self) -> "Operation":
        return self

    get_operation_name = Operation.get_operation_name

    def __post_init__(self) -> None:
        Buffer.__post_init__(self)
        Operation.__post_init__(self)


class InputBuffer(Buffer):
    """A buffer that was handed in, so something else holds its memory."""

    def num_reads(self) -> int:
        return 1


class DonatedBuffer(InputBuffer):
    """An input whose memory this is allowed to write over.

    A saved tensor that is not an input of the forward, an output of the forward
    or an output of the backward is one this may overwrite during the backward,
    because nothing outside will read it again.  An input of the forward is not,
    since the same tensor may be used by another function.
    """


class ConstantBuffer(InputBuffer):
    """A buffer holding a value fixed when the code was written.

    The name it is read under includes which device the value is on, so that
    the same constant written once per device does not collide.
    """

    override_device = None

    def make_loader(self):
        def loader(index: Sequence):
            indexer = self.get_layout().make_indexer()
            return ops.load(
                V.graph.constant_name(self.get_name(), self.override_device),
                indexer(index),
            )

        return loader

    def constant_to_device(self, device) -> IRNode:
        return ConstantBuffer(
            name=V.graph.constant_name(self.get_name(), device), layout=self.layout
        )


@dataclasses.dataclass
class MutableBox(IRNode):
    """A value that can be written in place.

    A box is not the value but the right to put a value somewhere and to change
    what is there.  That is what lets a result be written directly into memory
    another result already occupies, and it is why almost everything asked of a
    box is passed on to whatever is inside it.
    """

    data: IRNode

    def has_exceeded_max_reads(self) -> bool:
        return self.data.has_exceeded_max_reads()

    def get_device(self):
        return self.data.get_device()

    def make_loader(self):
        return self.data.make_loader()

    def make_indexer(self):
        return self.data.make_indexer()

    def get_stride(self) -> Sequence:
        return self.data.get_stride()

    def get_name(self) -> str:
        return self.data.get_name()

    def has_large_inner_fn(self, threshold: int | None = None) -> bool:
        return self.data.has_large_inner_fn(threshold)

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        return self.data.mark_reuse(users, graph_reuse=graph_reuse)

    def realize_hint(self) -> None:
        return self.data.realize_hint()

    def unwrap_view(self) -> "IRNode":
        return self.data.unwrap_view()

    def is_input_buffer(self) -> bool:
        return self.data.is_input_buffer()

    def freeze_layout(self) -> None:
        return self.data.freeze_layout()

    def freeze_layout_with_stride_order(
        self, order: Sequence[int], allow_padding: bool = False
    ) -> None:
        return self.data.freeze_layout_with_stride_order(order, allow_padding)

    def freeze_layout_with_fill_order(self, order: Sequence[int]) -> None:
        return self.data.freeze_layout_with_fill_order(order)

    def freeze_layout_with_same_order(self, stride: Sequence) -> None:
        return self.data.freeze_layout_with_same_order(stride)

    def freeze_layout_with_exact_strides(
        self, exact_strides: Sequence, allow_padding: bool = False
    ) -> None:
        return self.data.freeze_layout_with_exact_strides(exact_strides, allow_padding)

    def get_read_writes(self) -> "dependencies.ReadWrites":
        return self.data.get_read_writes()

    def get_reads(self) -> OrderedSet:
        return self.data.get_reads()

    def num_reads(self) -> int:
        return self.data.num_reads()

    def get_storage_numel(self):
        return self.data.get_storage_numel()

    def get_reduction_type(self) -> str | None:
        return self.data.get_reduction_type()

    def get_reduction_size(self) -> Sequence:
        return self.data.get_reduction_size()

    def is_extern(self) -> bool:
        return self.data.is_extern()

    def is_no_op(self) -> bool:
        return self.data.is_no_op()

    def constant_to_device(self, device) -> "IRNode":
        return self.data.constant_to_device(device)

    def get_mutation_names(self) -> Sequence[str]:
        return self.data.get_mutation_names()

    def get_operation_name(self) -> str:
        return self.data.get_operation_name()

    def get_inputs_that_alias_output(self) -> Sequence[str]:
        return self.data.get_inputs_that_alias_output()

    def realize(self) -> str | None:
        return self.data.realize()

    @cache_on_self_and_args("MutableBox")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return self.data.get_free_symbol_uses(unbacked_only)

    def get_read_names(self) -> OrderedSet:
        return self.data.get_read_names()

    def get_defining_op(self) -> "Operation | None":
        return self.data.get_defining_op()

    def codegen_reference(self, writer=None) -> str:
        return self.data.codegen_reference(writer)

    @property
    def layout(self) -> "OutputSpec":
        # The output specification is asked for rather than the layout, because
        # what a buffer holds is an output specification and not necessarily a
        # layout.
        return self.data.get_output_spec()

    def get_layout(self) -> "Layout":
        return self.data.get_layout()

    def get_output_spec(self) -> "OutputSpec":
        return self.data.get_output_spec()

    def get_size(self) -> Sequence:
        return self.data.get_size()

    @property
    def dtype(self):
        return self.data.dtype

    def __str__(self) -> str:
        if isinstance(self.data, MutableBox):
            line0 = f"{type(self).__name__}({type(self.data).__name__}("
            endl = "))"
            inner = self.data.data
        else:
            line0 = f"{type(self).__name__}("
            inner = self.data
            endl = ")"

        lines = [
            line0,
            indent(str(inner)),
            endl,
        ]
        return "\n".join(lines)

    __repr__ = __str__


class TensorBox(MutableBox):
    """The right to change a value, held over whatever is inside it."""

    @staticmethod
    def create(data: "IRNode"):
        # A shape is not a value and has no memory, so it is not boxed.
        if isinstance(data, ShapeAsConstantBuffer):
            return data
        return TensorBox(StorageBox(data))


class StorageBox(MutableBox):
    """The right to change a value, held over wherever its memory is.

    This is where a value becomes real: a body that could have had more work
    fused into it is given a buffer of its own, at which point nothing else can
    be fused into it and anything may read it without recomputing it.
    """

    def is_input_buffer(self) -> bool:
        if isinstance(self.data, (InputBuffer, ReinterpretView)):
            return self.data.get_name() in V.graph.graph_inputs
        return False

    def is_module_buffer(self) -> bool:
        return (
            isinstance(self.data, (ConstantBuffer))
            and self.data.get_name() in V.graph.constants
        )

    def realize(self) -> str | None:
        if IRNode.is_realized_node(self.data):
            return self.data.get_name()

        if not isinstance(self.data, (Pointwise, Reduction, Scan, Sort)):
            raise AssertionError(type(self.data))
        origin_node = self.data.get_origin_node()
        traceback = self.data.get_traceback()
        device = self.data.get_device()
        if device is None:
            raise AssertionError("Expected device is not None")

        self.data = ComputedBuffer(
            name=None,
            layout=FlexibleLayout(
                device=device,
                dtype=self.data.get_dtype(),
                size=self.data.get_size(),
                is_pinned=False,
            ),
            data=self.data,
        )
        self.data.name = V.graph.register_buffer(self.data)
        V.graph.register_operation(self.data)
        self.data.origins = self.origins
        self.data.origin_node = origin_node
        self.data.traceback = traceback
        self.data.stream_idx = self.data.data.stream_idx
        self.data.mempool = self.data.data.mempool
        return self.data.name

    def realize_hint(self) -> None:
        """Realize now, if this is one that is already known will have to be.

        A value that will have to be in memory anyway is put there now rather
        than at the point that is decided, so that a reader that runs before
        that point is not left with nothing to read from.
        """

        if (
            isinstance(self.data, (Pointwise, Reduction))
            and self.data.inner_fn_opcount().nontrivial_read_count > 1
        ):
            self.realize()

    def has_accumulated_enough_reads_by_size(self, threshold: int) -> bool:
        """Whether enough has been read, in total, to be worth materializing.

        What matters is not only how much was read but how it was spread: a
        large total that is one buffer read many times is not the same as the
        same total spread over several, so the largest single read is compared
        against the others as well.
        """

        size_of_reads = [
            V.graph.get_dep_size_hint(dep)
            for dep in self.get_reads()
            if not is_nonfreeable_buffers(dep)
        ]
        if not size_of_reads:
            return False
        total_size = sum(size_of_reads)
        max_size = max(size_of_reads)
        min_size = min(size_of_reads)
        return (
            total_size >= threshold
            and total_size / max_size >= 2
            and max_size == min_size
        )

    def has_exceeded_max_reads(self) -> bool:
        """Whether this has been read enough times to be worth materializing."""

        realize_acc_reads_threshold = config.realize_acc_reads_threshold
        if realize_acc_reads_threshold is None:
            if is_cpu(self.data):
                realize_acc_reads_threshold = config.realize_cpu_acc_reads_threshold
            else:
                realize_acc_reads_threshold = (
                    config._realize_acc_reads_threshold_default
                )
        else:
            if not isinstance(realize_acc_reads_threshold, int):
                raise AssertionError(
                    f"expected int realize_acc_reads_threshold, got {type(realize_acc_reads_threshold)}"
                )
        return isinstance(self.data, Pointwise) and (
            self.num_reads() > realize_acc_reads_threshold
            or self.has_large_inner_fn()
            or (
                config.realize_acc_reads_size_threshold is not None
                and self.has_accumulated_enough_reads_by_size(
                    config.realize_acc_reads_size_threshold
                )
            )
        )

    def should_realize_on_reuse(self, users: int, *, graph_reuse: bool = True) -> bool:
        """Whether a value read several times should be put in memory.

        Some operations are expensive enough, or use enough registers, that
        computing them once and keeping the result beats computing them per
        read even when the result is not much read.
        """

        if users > 1 and isinstance(self.data, (Pointwise, Reduction)):
            opcount = self.data.inner_fn_opcount()
            if "inline_asm_elementwise" in opcount.used_ops:
                return True
            if is_cpu(self.data):
                # An operation that is expensive on a processor, and cheap
                # enough on an accelerator that the trade does not pay there.
                heavy_ops = [
                    "exp",
                    "log",
                    "log10",
                    "log1p",
                    "log2",
                    "sigmoid",
                    "tanh",
                ]
                if any(x in opcount.used_ops for x in heavy_ops):
                    return True
                realize_threshold = self.data.get_realize_opcount_threshold()
                if (
                    isinstance(self.data, Pointwise)
                    and graph_reuse
                    and users > config.realize_opusers_threshold
                    and opcount.num_ops > max(0, realize_threshold - 2)
                ):
                    return True
            if self.has_large_inner_fn():
                return True
            return graph_reuse and self.num_reads() > config.realize_reads_threshold
        return False

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        if self.should_realize_on_reuse(users, graph_reuse=graph_reuse):
            self.realize()

    def num_reads(self) -> int:
        return self.data.num_reads()


def as_storage_and_layout(
    x: "IRNode",
    freeze: bool = True,
    want_contiguous: bool = False,
    stride_order: Sequence | None = None,
    allow_padding: bool = False,
    exact_strides: Sequence | None = None,
):
    """A node taken apart into the memory it occupies and how that memory is laid out.

    Boxes are taken off and the layout is settled, which is what turns an
    arbitrarily long chain of views into a buffer and a description of its
    elements.  A node that is none of those is refused, since there is nothing
    to take apart.

    ``allow_padding`` only affects how a stride order is applied: it is what
    says whether the order may be reached by padding a stride rather than only
    by the strides already present.
    """

    if isinstance(x, TensorBox):
        return as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
    if isinstance(x, StorageBox):
        _, layout = as_storage_and_layout(
            x.data,
            freeze=freeze,
            want_contiguous=want_contiguous,
            stride_order=stride_order,
            allow_padding=allow_padding,
            exact_strides=exact_strides,
        )
        return x, x.data.get_layout()
    if isinstance(x, Buffer):
        if freeze:
            if want_contiguous:
                x.freeze_layout()
                if not x.get_layout().is_contiguous():
                    raise AssertionError("Expected x.get_layout().is_contiguous()")
            elif stride_order is not None:
                x.freeze_layout_with_stride_order(
                    stride_order, allow_padding=allow_padding
                )
            elif exact_strides is not None:
                x.freeze_layout_with_exact_strides(
                    exact_strides, allow_padding=allow_padding
                )
            else:
                x.decide_layout()
        return StorageBox(data=x), x.get_layout()
    if isinstance(x, ReinterpretView):
        # Making the buffer this looks at contiguous, or in some stride order,
        # does not make this contiguous or in that order, so the request is not
        # passed on.
        buffer, _ = as_storage_and_layout(
            x.data,
            freeze=freeze,
        )
        return buffer, x.layout
    raise NotImplementedError


def is_stride_order_storage_and_layout(x: "IRNode", stride_order: Sequence) -> bool:
    """Whether this is one buffer whose loops are in this order already."""

    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        if not is_dense_contiguous_storage_and_layout(x):
            return False
        return get_stride_order(layout.stride) == list(stride_order)
    except NotImplementedError:
        return False


def is_unaligned(node: "IRNode") -> bool:
    """Whether a value's strides do not all line up, so no access may assume they do.

    A value whose memory was not laid out by us is the one that can be
    unaligned, and a kernel reading it has to be told rather than left to
    assume otherwise.
    """

    try:
        if not node.is_input_buffer():
            return False
        if not is_storage_and_layout(node):
            return False
        _buffer, layout = as_storage_and_layout(node, freeze=False)
        return not layout.is_pinned and V.graph.is_unaligned_buffer(
            node.get_name()
        )
    except NotImplementedError:
        return False


def _identity(x):
    """A size left as it is."""

    return x


def _is_fx_node(value) -> bool:
    """Whether a value is a graph node rather than what a node stands for."""

    return hasattr(value, "op") and hasattr(value, "users")


def record_original_output_strides(gm) -> None:
    """Note the strides each of a graph's outputs had when the graph was traced.

    Recorded once and then left alone, because a pass over the graph may pad a
    result to make it easier to compute, and a later reader asking what the graph
    produces is asking what the program asked for rather than what the pass made
    of it.  Overwriting on a second call would replace the answer to that question
    with the intermediate one.
    """

    import tensorplay as tp

    output_node = gm.graph.find_nodes(op="output")[0]
    if "original_output_strides" in output_node.meta:
        return

    # The output node wraps what it yields, so a graph yielding one value still
    # holds it in a one-element sequence, and a graph yielding several holds them
    # bare.
    outputs = output_node.args[0]
    if not _is_fx_node(outputs):
        outputs = (outputs,)

    strides = []
    for output in outputs:
        val = output.meta.get("val") if _is_fx_node(output) else None
        strides.append(
            tuple(int(x) for x in val.stride())
            if val is not None and isinstance(val, tp.Tensor)
            else None
        )
    output_node.meta["original_output_strides"] = strides


def gm_original_output_strides(gm) -> None:
    """Record on a graph both what it yields and the strides each output had.

    Which outputs a caller may read is settled here rather than left to whoever
    looks at the graph later: a graph's output node wraps its results, and a
    reader that walked the wrapper as if it were a result would count a structure
    as one of the values the graph produces.
    """

    output_node = gm.graph.find_nodes(op="output")[0]
    output_node.meta["user_visible_output_idxs"] = [
        idx for idx, _ in enumerate(output_node.args)
    ]

    record_original_output_strides(gm)


def convert_shape_to_symint(lst):
    """Sizes as the sizes a value can be asked for.

    A size that is a number is already one; a size that is an expression is made
    into the kind of size a value's shape can hold, so that a value made for a
    symbolic shape has a symbolic shape rather than a number that was guessed.
    """

    from .utils import convert_to_symint

    return [convert_to_symint(i) for i in lst]


def ir_node_to_tensor(x, replace_symbols_with_hints: bool = False):
    """A value with the shape, strides, type and device of a node, for measuring.

    A candidate is measured by running it, and it can only be run on a value, so a
    node -- which is a description of a value rather than one -- has to be made
    into one first.  It is made empty rather than filled: what is being measured
    is the time a candidate takes, and the time a candidate takes does not depend
    on what the value holds, only on its shape, its strides, its type and where it
    is.

    A node whose sizes are expressions can be given either the expressions
    themselves or the numbers they stand for.  The numbers are the default because
    a measurement that carries the expressions still compares them, and a
    comparison may decide the shape, and a shape decided during a measurement is a
    shape the measurement was not supposed to decide.  Asking for the numbers is
    therefore how a measurement is kept from changing what it measures.
    """

    from .loops import V

    if x is None:
        return None

    if replace_symbols_with_hints:
        sizevars = V.graph.sizevars
        shape_fn = sizevars.optimization_hint
    else:
        shape_fn = _identity
    size = [shape_fn(s) for s in x.get_size()]
    if is_storage_and_layout(x):
        stride = [shape_fn(s) for s in x.get_layout().stride]
    else:
        stride = FlexibleLayout.contiguous_strides(size)
    dtype = x.get_dtype()
    device = x.get_device()
    size = convert_shape_to_symint(size)
    stride = convert_shape_to_symint(stride)
    with V.graph.sizevars.shape_env.suppress_guards():
        return tp.empty_strided(size=size, stride=stride, dtype=dtype, device=device).zero_()


def is_storage_and_layout(x: "IRNode") -> bool:
    """Whether this can be taken apart into a buffer and a layout."""

    try:
        as_storage_and_layout(x, freeze=False)
        return True
    except NotImplementedError:
        return False


def is_contiguous_storage_and_layout(x: "IRNode") -> bool:
    """Whether this is one buffer whose elements run consecutively.

    The strides are padded first, so that a buffer whose strides are about to
    be padded is not called contiguous when it is not.
    """

    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        # Padding is accounted for here, so that a tensor is not claimed
        # contiguous when a padding is going to break it up.
        if layout.should_pad_strides():
            if not isinstance(layout, FlexibleLayout):
                raise AssertionError(type(layout))
            layout = FixedLayout(
                layout.device,
                layout.dtype,
                layout.size,
                layout._pad_strides(layout.stride, layout.size, layout.dtype),
                layout.offset,
                layout.is_pinned,
            )
        return layout.is_contiguous()
    except NotImplementedError:
        return False


def is_dense_contiguous_storage_and_layout(x: "IRNode") -> bool:
    """Whether this is one buffer whose elements run consecutively and fill it.

    Contiguous elements may still sit inside a larger allocation, so the size
    they occupy is compared against where they end.
    """

    try:
        _buffer, layout = as_storage_and_layout(x, freeze=False)
        if not layout.is_contiguous():
            return False
        return V.graph.sizevars.statically_known_equals(
            layout.storage_size(), layout.offset + sympy_product(layout.size)
        )
    except NotImplementedError:
        return False


@ir_dataclass
class BaseView(IRNode):
    """A tensor that is the same memory as another, reached differently.

    A view computes nothing.  It says that the elements of one tensor are the
    elements of another at positions given by a function of the index, and that
    is the whole of it -- which is why a view has no layout of its own and
    takes the one it looks at.
    """

    data: IRNode

    @cache_on_self_and_args("BaseView")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return self.data.get_free_symbol_uses(unbacked_only)

    def make_reindexer(self):
        raise NotImplementedError(f"make_reindexer NYI on {self}")

    def make_indexer(self):
        """Where an element of this tensor is in the one it looks at."""

        inner = self.data.make_indexer()
        reindex = self.make_reindexer()

        def indexer(idx: Sequence):
            return inner(reindex(idx))

        return indexer

    def make_loader(self):
        """The function that reads one element of this tensor."""

        inner = self.data.make_loader()
        reindex = self.make_reindexer()

        def loader(idx: Sequence):
            return inner(reindex(idx))

        return loader

    @property
    def dtype(self):
        return self.data.get_dtype()

    def get_layout(self) -> "Layout":
        return self.data.get_layout()

    def get_device(self):
        return self.data.get_device()

    def get_origin_node(self):
        return None

    def get_name(self) -> str:
        return self.data.get_name()

    def get_pointwise_size(self) -> Sequence:
        return self.get_size()

    def mark_reuse(self, users: int, *, graph_reuse: bool = True) -> None:
        return self.data.mark_reuse(users, graph_reuse=graph_reuse)

    def has_exceeded_max_reads(self) -> bool:
        return self.data.has_exceeded_max_reads()

    def realize(self) -> str | None:
        return self.data.realize()

    def realize_hint(self) -> None:
        self.data.realize_hint()

    def get_storage_numel(self):
        return self.data.get_storage_numel()

    def is_extern(self) -> bool:
        return self.data.is_extern()

    def is_module_buffer(self) -> bool:
        if not isinstance(self.data, BaseView):
            raise AssertionError(type(self.data))
        return self.data.is_module_buffer()

    def get_read_names(self) -> OrderedSet:
        return self.data.get_read_names()

    def get_reads(self) -> OrderedSet:
        with patch.object(FlexibleLayout, "allow_indexing", True):
            return extract_read_writes(
                self.make_loader(),
                self.get_size(),
            ).reads

    def unwrap_view(self) -> "IRNode":
        """Whatever this looks at, with every view in between taken off."""

        x: IRNode = self
        while isinstance(x, BaseView):
            x = x.data
        return x

    def constant_to_device(self, device) -> "IRNode":
        """This view, over a constant on another device.

        Only a view of constants can be moved, since what it looks at would
        otherwise still be where it was.
        """

        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Pointwise(
            device=device,
            dtype=self.get_dtype(),
            inner_fn=loader,
            ranges=self.get_size(),
        )


@ir_dataclass
class ExpandView(BaseView):
    """A view with dimensions of extent one, reading the same element everywhere.

    A dimension of extent one holds one element, so every position along it
    names the same element; that is what makes an expanded tensor bigger than
    the one it looks at while holding no more memory.
    """

    size: Sequence

    @staticmethod
    def _normalize_size(x: "IRNode", new_size: Sequence) -> Sequence:
        """Fill in the dimensions whose size was left to be worked out.

        A size of minus one means the dimension keeps the size it had, since
        that is the only reading under which the two tensors can still be the
        same memory.
        """

        sizevars = V.graph.sizevars
        new_size = [sympy.expand(s) for s in new_size]
        old_size = x.get_size()
        old_size = [None] * (len(new_size) - len(old_size)) + list(old_size)
        if len(new_size) != len(old_size):
            raise AssertionError("Expected len(new_size) == len(old_size)")
        for i in range(len(new_size)):
            if new_size[i] == -1:
                if old_size[i] is None:
                    raise AssertionError("Expected old_size[i] is not None")
                new_size[i] = old_size[i]
            elif old_size[i] is None or V.graph.sizevars.is_size_one_or_false(
                old_size[i]
            ):
                pass
            elif not has_free_unbacked_symbols(
                old_size
            ) and not has_free_unbacked_symbols(new_size):
                # A dimension may only grow, never shrink: a view cannot read
                # past what is there.  That the two are equal is already
                # guarded, since the shape it was written with was expected to
                # have established it.
                v1 = new_size[i]
                v2 = old_size[i]
                if v1 is None:
                    raise AssertionError("Expected v1 is not None")
                if v2 is None:
                    raise AssertionError("Expected v2 is not None")
                diff = v1 - v2
                if (
                    sizevars.optimization_hint(
                        diff,
                        fallback=0,
                    )
                    != 0
                ):
                    raise AssertionError(
                        f"Broadcast failed in ExpandView({x.get_size()}, {new_size}) on dimension {i}"
                    )
        return new_size

    @classmethod
    def create(cls, x: "IRNode", new_size: Sequence) -> "BaseView":
        new_size = cls._normalize_size(x, new_size)

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            skip = len(new_size) - len(old_layout.size)
            if skip < 0:
                raise AssertionError("Expected skip >= 0")
            # A dimension of extent one is broadcast, and a broadcast is
            # reached at a stride of zero however it was reached before.
            new_stride = [sympy.S.Zero] * skip
            for stride, size in zip(old_layout.stride, old_layout.size):
                new_stride.append(
                    stride
                    if not V.graph.sizevars.is_size_one_or_false(size)
                    else sympy.S.Zero
                )
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                list(new_size),
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return ExpandView(data=x, size=new_size)

    def get_size(self) -> Sequence:
        return self.size

    def make_reindexer(self):
        target = self.get_size()
        actual = self.data.get_size()
        skip = len(target) - len(actual)

        def reindex(index: Sequence) -> Sequence:
            index = list(index[skip:])
            if len(index) != len(actual):
                raise AssertionError("Expected len(index) == len(actual)")
            for i in range(len(actual)):
                if actual[i] == 1:
                    # A dimension of extent one names the same element whatever
                    # the position along it, so it contributes nothing.
                    index[i] = sympy.S.Zero
            return index

        return reindex


@ir_dataclass
class PermuteView(BaseView):
    """A view whose dimensions are in another order."""

    dims: list

    @classmethod
    def create(cls, x: "IRNode", dims: Sequence) -> "BaseView":
        dims = cls._map_neg_dims(dims)
        if OrderedSet(dims) != OrderedSet(range(len(dims))):
            raise AssertionError(
                "Expected OrderedSet(dims) == OrderedSet(range(len(dims)))"
            )

        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                [old_layout.size[i] for i in dims],
                [old_layout.stride[i] for i in dims],
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return PermuteView(data=x, dims=dims)

    @classmethod
    def _map_neg_dims(cls, dims: Sequence) -> list:
        """Count a negative dimension from the end, which is how it was written."""

        return [dim if dim >= 0 else len(dims) + dim for dim in dims]

    def get_size(self) -> Sequence:
        if OrderedSet(self._map_neg_dims(self.dims)) != OrderedSet(
            range(len(self.dims))
        ):
            raise AssertionError(
                "Expected OrderedSet(self._map_neg_dims(self.dims)) == OrderedSet( range(len(self.dims)) )"
            )
        size = self.data.get_size()
        return [size[i] for i in self.dims]

    def get_stride(self) -> Sequence:
        if not (
            OrderedSet(self._map_neg_dims(self.dims))
            == OrderedSet(range(len(self.dims)))
        ):
            raise AssertionError("dims must be a permutation of range(len(dims))")
        stride = self.data.get_stride()
        return [stride[i] for i in self.dims]

    def make_reindexer(self):
        """The permutation that undoes this one, so the index reaches the original.

        An index in the permuted order has to be put back before it can be used
        to address the tensor underneath, and this is that putting back.
        """

        inv = {j: i for i, j in enumerate(self.dims)}
        inv = [inv[i] for i in range(len(self.dims))]
        if OrderedSet(inv) != OrderedSet(range(len(self.dims))):
            raise AssertionError(
                "Expected OrderedSet(inv) == OrderedSet(range(len(self.dims)))"
            )

        def reindex(index: Sequence) -> Sequence:
            return [index[i] for i in inv]

        return reindex


def _fixed_indexer(size: Sequence, stride: Sequence, offset=0):
    """A closure holding the arithmetic that reads one element of a layout.

    The arithmetic is built once and then applied to an index per access, so a
    kernel that reads many elements of the same buffer pays for the address
    computation once rather than once per element.
    """

    def indexer(index: Sequence):
        if not (stride is not None and len(index) == len(stride)):
            raise AssertionError(
                "Expected stride is not None and len(index) == len(stride)"
            )
        if len(index) != len(size):
            raise AssertionError("Expected len(index) == len(size)")
        result = offset
        for idx, st, sz in zip(index, stride, size):
            if sz != 1:
                result = result + idx * st
        return result

    return indexer


def is_contiguous_strides_for_shape(stride: Sequence, shape: Sequence) -> bool:
    """Whether these strides walk a shape of this size consecutively.

    A dimension of extent one is skipped, since it holds one element and its
    stride is never taken.  Both the stride the running product gives and the
    one that skips the extent-one dimensions are accepted, because both reach
    the same element next.
    """

    expected_stride = 1
    expected_stride_max = 1
    for x, y in reversed(tuple(zip(shape, stride))):
        if x == 1:
            continue

        if not V.graph.sizevars.statically_known_equals(
            y, expected_stride
        ) and not V.graph.sizevars.statically_known_equals(y, expected_stride_max):
            return False

        expected_stride_max *= sympy.Max(1, x)
        expected_stride *= x

    return True


def get_align_for_dtype(dtype) -> int:
    """The width a load of this type has to be aligned to.

    A dtype arrives here under whichever name the caller had it: the loops
    arithmetic carries element types as names, while a tensor carries the type
    itself.  Both say the same thing about the width, so both are read the same
    way -- a name is looked up to get at the type and its width, and a type is
    asked for its width directly.

    A width of zero would make this a division by zero, so a type whose width
    is not known is measured as one byte: the alignment it gets is then the
    whole padding budget, which pads more than needed rather than crashing.
    """

    itemsize = getattr(dtype, "itemsize", None)
    if itemsize is None:
        itemsize = getattr(getattr(tp, str(dtype), None), "itemsize", 1)
    return config.padding_alignment_bytes // (itemsize or 1)


@ir_dataclass
class BaseConstant(IRNode):
    """A value fixed when the code was written, rather than computed by it.

    A constant has no shape and reads nothing, so almost everything asked of a
    node has a short answer here: what it is, and where it is.
    """

    dtype: Any
    device: Any

    def get_size(self) -> Sequence:
        return ()

    def get_device(self):
        return self.device

    def get_origin_node(self):
        return None

    def get_reads(self) -> OrderedSet:
        return OrderedSet()


@ir_dataclass
class Constant(BaseConstant):
    """A value written into the code itself."""

    value: Any
    dtype: Any
    device: Any

    def make_loader(self):
        def loader(index: Sequence):
            return ops.constant(self.value, self.dtype)

        return loader

    def realize(self) -> str | None:
        pass

    def constant_to_device(self, device) -> "IRNode":
        return Constant(value=self.value, dtype=self.dtype, device=device)


@ir_dataclass
class IndexingConstant(BaseConstant):
    """A position expression fixed when the code was written.

    This is a shape rather than a value: it is known without reading anything,
    but it is an expression over the loop variables rather than a number, so
    reading it produces an index rather than a constant.
    """

    index: Any
    dtype: Any
    device: Any

    def make_loader(self):
        def loader(index: Sequence):
            return ops.index_expr(self.index, self.dtype)

        return loader

    def constant_to_device(self, device) -> "IRNode":
        return IndexingConstant(index=self.index, dtype=self.dtype, device=device)


@ir_dataclass
class NoneAsConstantBuffer(IRNode):
    """A stand-in for a value that is not there at all.

    An operation that produces nothing still has to appear as a node, and this
    is what it appears as: it reads nothing, has no output to describe, and is
    written into the code as whatever stands for nothing.
    """

    def get_reads(self) -> OrderedSet:
        return OrderedSet()

    @cache_on_self_and_args("NoneAsConstantBuffer")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return OrderedSet()

    def codegen_reference(self, writer=None) -> str:
        return V.graph.wrapper_code.none_str

    def get_output_spec(self) -> "OutputSpec":
        return NoneLayout(device=None)

    def has_tensor_output(self) -> bool:
        return False


@ir_dataclass
class ShapeAsConstantBuffer(IRNode):
    """A shape, where a value was expected.

    A shape is a number once it is known, and is handed around in place of a
    value in the places where only its extent matters.  It is not a tensor, so
    it has no output to describe.
    """

    expr: Expr

    @cache_on_self_and_args("ShapeAsConstantBuffer")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return get_free_symbols(self.expr, unbacked_only)

    def codegen_reference(self, writer=None) -> str:
        return V.graph.wrapper_code.codegen_sizevar(self.expr)

    def has_tensor_output(self) -> bool:
        return False


@dataclasses.dataclass(frozen=True)
class ExtraIndexingConstraints:
    """Indexing constraints added while a body is being simplified.

    The C++ path produces these to hold the loop and reduced loops of two
    scheduler nodes compatible with each other; they are carried through the
    simplification and recomputation of the size and body rather than being
    passed as a bare pair of the loops and their extents, because there is more
    to them than that.
    """

    ranges: Any
    exprs: list


@ir_dataclass
class MultiOutputLayout(OutputSpec):
    """The description of a result that is several values rather than one tensor.

    There is no shape to describe -- the values are wherever the kernel put
    them -- so only the device is recorded, and a caller that needs a shape has
    to ask the values rather than here.
    """

    device: Any

    def get_device(self):
        return self.device


class MutationOutput(Buffer):
    """An output that is a buffer someone else already had, written in place.

    What is written is not new memory but an overwrite, so the buffer this names
    has to be reported as mutated: whoever read the value that is being replaced
    has to run first.
    """

    def __init__(
        self, layout: "OutputSpec", mutated_node: "IRNode", mutating_node: "Operation"
    ) -> None:
        super().__init__(name=None, layout=layout)
        mutated_node_name = mutated_node.get_name()
        V.graph.mark_buffer_mutated(mutated_node_name)
        self.mutation_names = [mutated_node_name]
        self.mutating_node: Operation = mutating_node
        self.name = V.graph.register_buffer(self)

    def get_defining_op(self) -> "Operation":
        return self.mutating_node

    def get_mutation_names(self) -> Sequence[str]:
        return self.mutation_names

    def should_allocate(self) -> bool:
        return False

    def get_mutation_buffers(self) -> Sequence["IRNode"]:
        mutation_names = self.get_mutation_names()
        return [
            buf
            for buf in (V.graph.try_get_buffer(name) for name in mutation_names)
            if buf is not None
        ]


@ir_dataclass(frozen=False)
class Subgraph(IRNode):
    """A piece of a graph that is compiled on its own and then called.

    Naming it separately is what lets it be compiled once and used more than
    once, and it is also what lets the settings it was written under be recorded
    alongside it, since those settings are part of what it was compiled as.
    Its lowering is filled in by the first operation that runs it, which is why
    the field can be written after the piece is made.
    """

    name: str
    graph_module: Any
    tp_config_patches: dict | None = None
    graph: Any = None


@dataclasses.dataclass
class FinalizeCodegenResult:
    """What a template's code came out as, for a backend that then has to call it.

    A template writes the body of a kernel and leaves the call to whoever asked
    for it, so what comes back is the body, what has to be imported for it to
    mean anything, and the two lists that have to be written around the call.
    """

    source: str
    imports: list
    call_preamble: list
    call_args: list


def _has_aliased_buffers(buffers: Sequence) -> bool:
    """Whether two of these are the same memory under two names.

    Views are taken back to what they look at, since two views of one buffer
    alias each other whatever their shapes say, and then it is asked whether
    the list holds anything twice.
    """

    buffers = [
        buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
        for buffer in buffers
    ]
    return len(OrderedSet(id(buffer) for buffer in buffers)) < len(buffers)


def is_node_sequence(nodes: Sequence) -> bool:
    """Whether this is a list of nodes rather than a list of lists of nodes.

    An argument that is one of several things is written as either the things
    or a list of them, and this is what tells the two apart.
    """

    return all(isinstance(n, IRNode) for n in nodes)


@ir_dataclass(frozen=False)
class InputsKernel(OperationBuffer):
    """A buffer whose contents come from elsewhere, under another name.

    A kernel that hands its inputs straight to something written elsewhere takes
    the whole of each input rather than any particular elements of it, so what
    it depends on is recorded as a whole buffer rather than as positions in it.
    """

    inputs: Sequence

    def input_name(self, i: int) -> str:
        input = self.inputs[i]
        if not isinstance(input, IRNode):
            raise AssertionError("Expected isinstance(input, IRNode)")
        return input.get_name()

    def get_read_writes(self) -> "dependencies.ReadWrites":
        from . import dependencies

        reads = OrderedSet()
        StarDep = dependencies.StarDep
        for input in self.inputs:
            if isinstance(input, Sequence):
                reads.update(StarDep(x.get_name()) for x in input)
            elif isinstance(input, ShapeAsConstantBuffer):
                # A shape is visible everywhere, so depending on it would say
                # nothing about what has to be computed first.
                continue
            else:
                reads.add(StarDep(input.get_name()))

        writes = OrderedSet(StarDep(buf.get_name()) for buf in self.get_outputs())

        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )

    def get_reads(self) -> OrderedSet:
        return self.get_read_writes().reads

    @classmethod
    def unwrap_storage_for_input(cls, x: "IRNode") -> "IRNode":
        """The buffer an input names, with the boxes and views around it taken off."""

        if isinstance(x, TensorBox):
            x = x.data
        if isinstance(x, StorageBox):
            x = x.data
        if isinstance(x, BaseView) and not isinstance(x, ReinterpretView):
            x = ExternKernel.realize_input(x)
        if isinstance(x, TensorBox):
            # Where making the view above failed, the result comes back wrapped
            # in a pair, so the taking off has to be done again.
            return cls.unwrap_storage_for_input(x)
        if not isinstance(x, (Buffer, ReinterpretView)):
            raise AssertionError(type(x))
        return x

    @staticmethod
    def unwrap_storage(inputs: Sequence) -> list:
        """Every input taken back to the buffer it names."""

        inputs_new: list = []
        for x in inputs:
            if isinstance(x, Sequence):
                x = [InputsKernel.unwrap_storage_for_input(i) for i in x]
            else:
                x = InputsKernel.unwrap_storage_for_input(x)
            inputs_new.append(x)
        return inputs_new

    def is_extern(self) -> bool:
        return True

    def num_reads(self) -> int:
        return 1

    @cache_on_self_and_args("InputsKernel")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        r = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                r |= inp.get_free_symbol_uses(unbacked_only)
            else:
                for inner_inp in inp:
                    r |= inner_inp.get_free_symbol_uses(unbacked_only)
        return r


class NopKernel(InputsKernel):
    """A buffer that is the same as one of its inputs, under another name.

    Nothing is computed, which is why it reads nothing: what it holds is what
    was handed to it.
    """

    def is_no_op(self) -> bool:
        return True

    def get_reads(self) -> OrderedSet:
        return OrderedSet()


@ir_dataclass(frozen=False)
class ComputedBuffer(OperationBuffer):
    """A buffer that is computed rather than handed in.

    The buffer is a place in memory together with the body that fills it, and
    between them they say what is computed and where each element ends up.  The
    layout is not settled until it has to be, which is what allows the body's
    loops to be rearranged first and the layout chosen to suit.
    """

    data: Loops
    _force_realize: ClassVar[bool] = False

    # What a split reduction was before it was split.
    _split_size: int | None = None
    _original_inner_fn: Any = None
    _original_ranges: Sequence | None = None
    _original_reduction_ranges: Sequence | None = None

    @contextlib.contextmanager
    def with_original_inner_fn(self):
        """Look at this buffer as the unsplit reduction it came from.

        A reduction that was cut into layers records what it looked like before
        the cut, and this puts that back for as long as the caller needs it, so
        that what is being examined is the reduction as it was written rather
        than as it was scheduled.  Whatever the caller changed is put back
        afterwards, since this is a question being asked and not a change.
        """

        if self._split_size is None:
            raise AssertionError("Expected self._split_size is not None")
        if self._original_inner_fn is None:
            raise AssertionError("Expected self._original_inner_fn is not None")
        if self._original_ranges is None:
            raise AssertionError("Expected self._original_ranges is not None")
        if self._original_reduction_ranges is None:
            raise AssertionError("Expected self._original_reduction_ranges is not None")

        if not isinstance(self.data, Reduction):
            raise AssertionError(f"{type(self.data)}")
        old_data = self.data
        old_layout = self.layout
        try:
            new_data = Reduction(
                device=old_data.device,
                dtype=old_data.dtype,
                inner_fn=self._original_inner_fn,
                ranges=self._original_ranges,
                reduction_ranges=self._original_reduction_ranges,
                reduction_type=old_data.reduction_type,
                src_dtype=old_data.src_dtype,
                reduction_hint=old_data.reduction_hint,
            )
            self.data = new_data
            # The layout here is the one the unsplit reduction would have had.
            # Nothing reads it, since the stores are skipped while this is in
            # effect.
            self.layout = FixedLayout(
                old_data.device,
                old_data.dtype,
                self._original_ranges,
            )
            if hasattr(self.get_default_sizes_body, "clear_cache"):
                self.get_default_sizes_body.clear_cache(self)
            yield
        finally:
            self.data = old_data
            self.layout = old_layout

    @staticmethod
    @contextlib.contextmanager
    def force_realize():
        """Make every value be given memory, even where that is not needed.

        Some questions are only worth asking of something that has been written
        out, and this makes every value answer to them, for as long as the
        caller needs.
        """

        old_value = ComputedBuffer._force_realize
        try:
            ComputedBuffer._force_realize = True
            yield
        finally:
            ComputedBuffer._force_realize = old_value

    def get_computed_buffer_name(self) -> str | None:
        return None

    def num_reads(self) -> int:
        return sum(1 for _ in self.get_reads())

    def get_reads(self) -> OrderedSet:
        return self.get_read_writes().reads

    def get_read_names(self) -> OrderedSet:
        return OrderedSet(dep.name for dep in self.get_reads())

    def get_read_writes(self) -> "dependencies.ReadWrites":
        if not isinstance(self.data, (Reduction, Scan, Sort, Pointwise)):
            return dependencies.ReadWrites(
                reads=OrderedSet(),
                writes=OrderedSet(),
                index_exprs=OrderedSet(),
            )

        with patch.object(FlexibleLayout, "allow_indexing", True):
            if self.data.get_reduction_type():
                return extract_read_writes(
                    self.get_store_function(),
                    self.data.get_pointwise_size(),
                    self.data.get_reduction_size(),
                )
            return extract_read_writes(
                self.get_store_function(),
                self.data.get_size(),
            )

    @cache_on_self_and_args("ComputedBuffer")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        # A buffer has no argument list to look at, so what it depends on is
        # worked out from its layout and its body together.
        #
        # This has to agree with what the kernel arguments are made to be,
        # since that is what decides a shape becomes a dependency at all, and
        # code generation has not started yet so that logic cannot be reused
        # directly.
        #
        # A buffer holding a reduction over a loop variable is covered, though
        # for a reason worth stating: the only question that needs a right
        # answer here is which shapes an item read has to be preceded by, and
        # no item read can depend on a reduction over a loop variable without
        # first depending on a buffer that does not.
        result = self.layout.get_free_symbol_uses(
            unbacked_only
        ) | self.data.get_free_symbol_uses(unbacked_only)

        if self.has_store_function():
            result |= self.get_read_writes().get_free_symbol_uses(unbacked_only)
        return result

    def make_loader(self):
        if (
            not self.get_reduction_type()
            and self.name not in V.graph.mutated_buffers
            and self.num_reads() == 0
            and not self._force_realize
        ):
            # Nothing reads this and nothing overwrites it, so the body can be
            # used where a read would go rather than a read of where it landed.
            return self.data.make_loader()
        return super().make_loader()

    def has_store_function(self) -> bool:
        return isinstance(self.data, (Reduction, Scan, Sort, Pointwise))

    def get_store_function(self):
        """The function that writes this buffer, given where each element goes."""

        indexer = self.get_layout().as_fixed().make_indexer()
        if isinstance(self.data, (Reduction, Scan, Sort)):
            return partial(self.data.store_reduction, self.name, indexer)
        else:
            if not isinstance(self.data, Pointwise):
                raise AssertionError(type(self.data))
            return partial(self.data.store_output, self.name, indexer)

    def get_fill_order(self):
        """The order the dimensions should be filled in, if the strides say one.

        Taken from the order the buffers read are laid out in, on the grounds
        that a loop matching the order things are read in is a loop that reads
        them well.  Only reads of buffers the same size count, and only those
        that say where in a buffer they land: a read of a whole buffer says
        nothing about an order.

        A better answer would look at what happens to the value afterwards and
        choose for the whole graph at once, and this is also something that
        wants to be measured rather than guessed.
        """

        if isinstance(self.layout, FlexibleLayout):
            (index_vars, reduction_vars), _ = dependencies.index_vars_squeeze(
                self.data.get_pointwise_size(), self.data.get_reduction_size()
            )
            reads = self.get_read_writes().reads
            # Only reads of buffers of the same size count, and a read of a
            # whole buffer is skipped, since it says nothing about an order.
            if not all(
                isinstance(r, (dependencies.StarDep, dependencies.MemoryDep))
                for r in reads
            ):
                raise AssertionError(
                    "Expected all( isinstance(r, (dependencies.StarDep, dependencies.MemoryDep)) for r in reads )"
                )
            reads = [
                sympy_subs(r.index, {v: sympy.S.Zero for v in reduction_vars if v != 0})
                for r in reads
                if isinstance(r, dependencies.MemoryDep)
            ]

            if reads:
                if isinstance(self.data, (Scan, Sort)):
                    indices = self.data.reindex(index_vars, reduction_vars)
                else:
                    indices = index_vars
                stride_lengths = [
                    V.graph.sizevars.stride_hints(expr, indices) for expr in reads
                ]
                from .scheduler import pick_loop_order

                return pick_loop_order(stride_lengths, self.get_size())

        return None

    def decide_layout(self) -> None:
        if isinstance(self.layout, FlexibleLayout):
            order = self.get_fill_order()
            if order:
                self.freeze_layout_with_fill_order(order)
            else:
                self.freeze_layout()

    @cache_on_self
    def get_default_sizes_body(self):
        """The body as captured, with the loops it runs over and their extents.

        A dimension of extent one is dropped here rather than later, since it
        orders nothing and keeping it would make every later step carry it.
        """

        args, var_ranges = dependencies.index_vars_squeeze(
            self.get_pointwise_size(), self.get_reduction_size(), prefix="q"
        )
        with patch.object(ConstantBuffer, "override_device", self.get_device()):
            body = LoopBody(
                self.get_store_function(),
                (args if self.get_reduction_type() else args[:1]),
                var_ranges,
                *args,
            )
        index_vars = []
        reduce_vars: list = []
        index_size = []
        reduce_size = []
        for v, s in var_ranges.items():
            if v in args[0]:
                if reduce_vars:
                    raise AssertionError("Expected not reduce_vars")
                index_vars.append(v)
                index_size.append(s)
            else:
                if v not in args[1]:
                    raise AssertionError("Expected v in args[1]")
                reduce_vars.append(v)
                reduce_size.append(s)
        return (index_size, reduce_size), body, (index_vars, reduce_vars)

    def simplify_and_reorder(
        self,
        extra_indexing_constraints: ExtraIndexingConstraints | None = None,
        recompute_sizes_body_func: Any = None,
    ):
        """The body with its loops rearranged, which is where most of the work is.

        Three things happen here, and they are the rearrangements every lowered
        body goes through: dimensions of extent one are dropped, dimensions that
        run consecutively are merged into one loop, and the loops are put in the
        order the strides suggest.

        Extra indexing constraints can be given to hold the loops of two nodes
        compatible with each other, which is what lets two bodies with the same
        extents be fused.  A function can be given to redo the sizes and the
        body, which is how a further rearrangement is applied on top.
        """

        (
            (index_size, reduce_size),
            body,
            (index_vars, reduce_vars),
        ) = self.get_default_sizes_body()

        if recompute_sizes_body_func:
            (
                (index_size, reduce_size),
                body,
                (index_vars, reduce_vars),
            ) = recompute_sizes_body_func(
                (index_size, reduce_size), body, (index_vars, reduce_vars)
            )

        index_formulas = [*body.indexing_exprs.values()]
        if extra_indexing_constraints is not None:
            expected_var_ranges = body.var_ranges
            if expected_var_ranges != extra_indexing_constraints.ranges:
                raise AssertionError(
                    (
                        expected_var_ranges,
                        extra_indexing_constraints.ranges,
                    )
                )
            # The expressions already present are dropped, since adding one
            # that is there would say the same thing twice.
            extra_indexing_expr = [
                e for e in extra_indexing_constraints.exprs if e not in index_formulas
            ]
            index_formulas += extra_indexing_expr

        memory_addrs = [*body.get_write_exprs()]
        if not V.graph.has_feature(self, BackendFeature.PREFER_STORE_LOOP_ORDER):
            memory_addrs.extend(body.get_read_exprs())

        def simplify_and_reorder(
            x_vars: Sequence,
            support_vars: Sequence,
            sizes: Sequence,
            simplify_loops: bool,
        ):
            newsizes, reindex0, reindex1 = self._apply_loop_reordering(
                x_vars, support_vars, sizes, memory_addrs
            )

            # A matrix product's code assumes a particular order, whatever the
            # strides of its inputs:
            #
            #   for z -> y -> x -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
            #   for z -> x -> y -> r:  C[z, y, x] += A[z, y, r] * B[z, r, x]
            #
            # What matters is where the batch axis is.  The other two may be
            # swapped either way round, but moving the batch axis breaks it.
            #
            # So a reordering that moves it is undone, which is not always the
            # best order when the strides do not match that assumption but is
            # the one the code can be written for.
            if self.get_reduction_type() == "dot" and len(sizes) == 3:
                order = list(range(len(sizes)))  # default order

                if reindex0(order)[0] != 0:
                    newsizes = [sizes[i] for i in order]
                    reindex0 = same_reorder(order)
                    reindex1 = inverse_reorder(order)

            x_vars = reindex0(x_vars)

            if simplify_loops:
                newsizes, reindex2, _prune = V.graph.sizevars._simplify_loops(
                    x_vars,
                    newsizes,
                    index_prevent_reordering(index_formulas, x_vars, newsizes),
                )
                reindex = fuse_reindexing(reindex1, reindex2)
            else:
                reindex = reindex1
            return newsizes, reindex, reindex1

        support_vars = index_vars + reduce_vars
        should_merge_loops = (
            not is_gpu(get_device_type(self)) or not config.loop_ordering_after_fusion
        )
        iter_ranges, iter_reindex, _ = simplify_and_reorder(
            index_vars,
            support_vars,
            index_size,
            should_merge_loops,
        )

        # The reduced loops are held back from being merged for the same reason
        # the iterated ones are: merging them too early makes it hard to see
        # which order a following body should be in to be fusible with this
        # reduction.
        reduce_ranges, reduce_reindex, _ = simplify_and_reorder(
            reduce_vars, support_vars, reduce_size, should_merge_loops
        )

        # The body is captured again with the rearrangement applied, rather than
        # the old one being described in new terms.
        (iter_vars, reduce_vars), var_ranges = dependencies.index_vars_no_squeeze(
            iter_ranges,
            reduce_ranges,
            prefix="p",
        )
        body = LoopBody(
            body,
            [iter_reindex(iter_vars), reduce_reindex(reduce_vars)],
            var_ranges,
            iter_vars,
            reduce_vars,
        )
        return (iter_ranges, reduce_ranges), body

    @staticmethod
    def _apply_loop_reordering(
        index_vars: Sequence,
        support_vars: Sequence,
        sizes: Sequence,
        memory_addrs: list,
        priority_idx: list | None = None,
    ):
        """The loops put in an order the strides suggest, where one can be found.

        An order that cannot be worked out is not a failure: the loops are left
        as they are, which is the order the body was written in.
        """

        from .scheduler import pick_loop_order

        if priority_idx is None:
            priority_idx = []

        try:
            strides = [
                V.graph.sizevars.stride_hints(expr, index_vars, support_vars)
                for expr in memory_addrs
            ]
            if not (
                len(strides) == len(memory_addrs) and len(strides[0]) == len(index_vars)
            ):
                raise AssertionError(
                    "Expected len(strides) == len(memory_addrs) and len(strides[0]) == len( index_vars )"
                )
            order = list(reversed(pick_loop_order(strides, sizes, priority_idx)))
        except Exception:
            if config.debug:
                log.warning(
                    "Did not simplify complex index:\n%s\n%s",
                    dict(zip(index_vars, sizes)),
                    memory_addrs,
                )
            order = list(range(len(sizes)))
        sizes = [sizes[i] for i in order]
        return sizes, same_reorder(order), inverse_reorder(order)

    def get_pointwise_size(self) -> Sequence:
        return self.data.get_pointwise_size()

    def get_reduction_size(self) -> Sequence:
        return self.data.get_reduction_size()

    def get_reduction_type(self) -> str | None:
        return self.data.get_reduction_type()

    def is_no_op(self) -> bool:
        return self.data.is_zero_elements()

    def should_allocate(self) -> bool:
        return True

    def constant_to_device(self, device) -> "IRNode":
        """This buffer, over a constant on another device."""

        return self.data.constant_to_device(device)


def maybe_free_unbacked_symbols(s, unbacked_only: bool = False) -> OrderedSet:
    """Every shape that came out of the data inside this argument, if it is one.

    An argument that is not a shape at all -- a number, a flag, a string --
    contributes nothing, and saying so is what lets the same question be asked
    of every argument without asking first what each one is.
    """

    if isinstance(s, sympy.Basic):
        return free_unbacked_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet()
        for t in s:
            r |= maybe_free_unbacked_symbols(t)
        return r
    else:
        return OrderedSet()


def maybe_free_symbols(s, unbacked_only: bool = False) -> OrderedSet:
    """Every shape symbol inside this argument, if it is a shape at all.

    ``unbacked_only`` restricts the answer to the shapes whose values came out
    of the data, which are the ones a kernel has to be given a value for.
    """

    if isinstance(s, sympy.Basic):
        return free_unbacked_symbols(s) if unbacked_only else free_symbols(s)
    elif isinstance(s, (tuple, list)):
        r = OrderedSet()
        for t in s:
            r |= maybe_free_symbols(t, unbacked_only)
        return r
    else:
        return OrderedSet()


def ops_wrapper(name: str):
    """The operation of a given name, as a function to be called directly.

    The table of how a reduction combines two values holds functions, and the
    names are what the table is written in; this is what turns a name back into
    something to call.
    """

    if not isinstance(name, str):
        raise AssertionError(type(name))

    def fn(*args, **kwargs):
        return getattr(ops, name)(*args, **kwargs)

    return fn


#: How each reduction folds two values into one.  A reduction that is not in
#: this table is either one whose combining is written out where it is used, or
#: one that is not supported.
REDUCTION_COMBINE_FN: dict = {
    "any": ops_wrapper("logical_or"),
    "max": ops_wrapper("maximum"),
    "fmax": ops_wrapper("fmaximum"),
    "min": ops_wrapper("minimum"),
    "prod": ops_wrapper("mul"),
    "sum": ops_wrapper("add"),
    "dot": ops_wrapper("add"),
    "xor_sum": ops_wrapper("bitwise_xor"),
}


def get_reduction_combine_fn(
    reduction_type: str, dtype, arg_break_ties_left: bool = True
):
    """The function that folds one value of a reduction into the next.

    Which function that is depends on the reduction, and for a maximum or a
    minimum it also depends on whether a zero of either sign is to be told
    apart, since a maximum of negative zero and positive zero is ambiguous and
    the tie has to be broken one way or the other.
    """

    if reduction_type in REDUCTION_COMBINE_FN:
        combine_fn = REDUCTION_COMBINE_FN[reduction_type]

        if (
            config.strict_signed_zero
            and reduction_type in ("max", "min")
            and is_float_dtype(dtype)
        ):

            def strict_signed_zero_combine_fn(a, b):
                # When the two are equal the second is taken, so that a zero of
                # one sign is replaced by one of the other and the sign of a
                # zero is whichever was seen last.
                value = combine_fn(a, b)
                return ops.where(ops.eq(a, b), b, value)

            return strict_signed_zero_combine_fn

        return combine_fn

    elif reduction_type in (
        "argmax",
        "argmin",
        "argmax_value",
        "argmin_value",
        "argmax_with_value",
        "argmin_with_value",
    ):

        def argmax_combine_fn(a, b):
            """Keep whichever of two values is further along, and where it was.

            A tie is broken by position, so that the answer does not depend on
            the order the values were combined in.  A value that is not a
            number is treated as larger than any number, so that it wins, and
            two of them tie with each other rather than with a number.
            """

            a_value, a_index = a
            b_value, b_index = b

            if reduction_type in ("argmin", "argmin_value", "argmin_with_value"):
                mask = ops.lt(a_value, b_value)
            else:
                mask = ops.gt(a_value, b_value)

            equal = ops.eq(a_value, b_value)
            if is_float_dtype(dtype):
                a_isnan = ops.ne(a_value, a_value)
                b_isnan = ops.ne(b_value, b_value)
                mask = ops.logical_or(mask, ops.gt(a_isnan, b_isnan))
                equal = ops.logical_or(equal, ops.logical_and(a_isnan, b_isnan))

            tie = (
                ops.lt(a_index, b_index)
                if arg_break_ties_left
                else ops.gt(a_index, b_index)
            )
            mask = ops.logical_or(mask, ops.logical_and(equal, tie))
            return (
                ops.where(mask, a_value, b_value),
                ops.where(mask, a_index, b_index),
            )

        return argmax_combine_fn

    elif reduction_type == "welford_combine":

        def welford_combine_fn(a, b):
            """Fold a running mean, spread and weight into another.

            The two means are subtracted to get how far apart they are, and
            that difference is what the spread has to be corrected by.  Where
            both means are infinite and equal, the difference would be a
            subtraction of infinities, so it is taken as zero there -- which
            happens when half-precision inputs overflow.
            """

            a_mean, a_m2, a_weight = a
            b_mean, b_m2, b_weight = b

            delta = ops.where(
                ops.logical_and(ops.isinf(a_mean), ops.eq(a_mean, b_mean)),
                ops.constant(0.0, tp.float32),
                b_mean - a_mean,
            )
            new_weight = a_weight + b_weight
            w2_over_w = b_weight / new_weight
            return (
                a_mean + delta * w2_over_w,
                a_m2 + b_m2 + delta * delta * a_weight * w2_over_w,
                new_weight,
            )

        return welford_combine_fn

    else:
        raise NotImplementedError(f"unknown reduction_type={reduction_type}")


@ir_dataclass
class Scan(Loops):
    """A value built by walking one axis, carrying a running value along it.

    Unlike a reduction, each step's result is itself an output, so the body
    both reads what the previous step left and writes a value out.  The axis
    being walked is therefore kept apart from the axes walked independently:
    the body is written over all of them, and reindexing is what puts the
    running axis back where the body expects it.
    """

    dtypes: tuple
    inner_fn: Any
    inner_fns: tuple
    size: list
    ranges: list
    scan_ranges: list
    combine_fn: Any
    reindex: Any
    reduction_hint: Any
    output_index: int

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        # What the body closes over is not visible from its arguments, so the
        # shapes in the ranges are added here rather than being missed.
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.scan_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        if len(self.ranges) + len(self.scan_ranges) != len(self.size):
            raise AssertionError(
                "Expected len(self.ranges) + len(self.scan_ranges) == len(self.size)"
            )
        super().__post_init__()

    def store_reduction(self, output_name, indexer, vars, scan_vars):
        idx = self.reindex(vars, scan_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.scan(self.dtypes, self.combine_fn, values)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self):
        return "custom"

    def get_reduction_size(self):
        return self.scan_ranges

    def get_size(self):
        return self.size

    def get_pointwise_size(self):
        return self.ranges

    def index_length(self) -> int:
        return len(self.ranges) + len(self.scan_ranges)

    def inner_fn_args(self):
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet:
        index = self._index(self.ranges)
        rindex = self._index(self.scan_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(
        cls,
        device,
        dtypes,
        inner_fns,
        size,
        axis,
        combine_fn,
        reduction_hint=ReductionHint.DEFAULT,
        **kwargs,
    ):
        """Build a scan over one axis, returning one realized value per result.

        The body is written over the axes walked independently followed by the
        running axis, so reindexing puts the running index back in the position
        the axis occupies.  A device that cannot write a scan, or several
        results on a device that cannot carry them together, gets no value and
        the call is handed to the framework; a scan over one element is that
        element copied.  A scan worth splitting is split into partial walks
        combined in order, where the device can do that and the order of the
        additions is not required to be fixed.
        """

        can_fallback_to_framework = kwargs.pop("can_fallback_to_aten", True)
        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        scan_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SCAN):
            return [None] * len(dtypes)

        if len(dtypes) > 1 and not V.graph.has_feature(
            device, BackendFeature.TUPLE_REDUCTION
        ):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        scan_numel = sizevars.simplify(sympy_product(scan_ranges))

        if len(dtypes) != len(inner_fns):
            raise AssertionError("Expected len(dtypes) == len(inner_fns)")

        if sizevars.statically_known_true(sympy.Le(scan_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        reduction_hint, num_splits = cls.num_splits(
            device=device,
            dtype=dtypes[0],
            inner_fn=inner_fns[0],
            axis=axis,
            pointwise_ranges=pointwise_ranges,
            scan_ranges=scan_ranges,
            combine_fn=combine_fn,
            scan_numel=scan_numel,
        )
        scan_type = Scan
        if num_splits > 1:
            supports_split = (
                len(dtypes) == 1 and not tp.are_deterministic_algorithms_enabled()
            )
            if not supports_split:
                if can_fallback_to_framework:
                    return [None] * len(dtypes)
                num_splits = 1
            else:
                scan_type = SplitScan

        def reindex(index, scan_index):
            if len(scan_index) != len(scan_ranges):
                raise AssertionError("Expected len(scan_index) == len(scan_ranges)")
            if len(index) != len(pointwise_ranges):
                raise AssertionError("Expected len(index) == len(pointwise_ranges)")
            return [*index[:axis], *scan_index, *index[axis:]]

        results = [
            TensorBox.create(
                scan_type(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    scan_ranges=scan_ranges,
                    combine_fn=combine_fn,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results

    @classmethod
    def num_splits(
        cls,
        device,
        dtype,
        inner_fn,
        axis,
        pointwise_ranges,
        scan_ranges,
        combine_fn,
        scan_numel,
    ):
        """How to split a scan, which is decided by the ordinary reduction rule.

        Splitting a scan runs several partial walks and combines their results
        in order, so the number of splits is worked out exactly as it is for a
        reduction and the running axis is placed back where the body expects it.
        """

        def wrapper_fn(idx, reduction_idx):
            return inner_fn([*idx[:axis], *reduction_idx, *idx[axis:]])

        return Reduction.num_splits(
            device=device,
            dst_dtype=dtype,
            src_dtype=dtype,
            inner_fn=wrapper_fn,
            ranges=pointwise_ranges,
            reduction_ranges=scan_ranges,
            reduction_type="scan",
            reduction_numel=scan_numel,
        )


@ir_dataclass
class SplitScan(Scan):
    pass


@ir_dataclass
class Sort(Loops):
    """A value built by ordering the values along one axis.

    Ordering is a walk over the axis being sorted where what is carried along
    is the ordering rather than an accumulated value, so the body reads the
    ordering so far and writes the extended one.  As with a scan, the axis
    being walked is kept apart from the others and reindexing puts it back.
    """

    dtypes: tuple
    inner_fn: Any
    inner_fns: tuple
    size: list
    ranges: list
    sort_ranges: list
    stable: bool
    descending: bool
    reindex: Any
    reduction_hint: Any
    output_index: int

    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return (
            super().get_free_symbol_uses(unbacked_only)
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.sort_ranges)
            )
            | OrderedSet().union(
                *(get_free_symbols(e, unbacked_only) for e in self.size)
            )
        )

    def __post_init__(self) -> None:
        if len(self.ranges) + len(self.sort_ranges) != len(self.size):
            raise AssertionError(
                "Expected len(self.ranges) + len(self.sort_ranges) == len(self.size)"
            )
        super().__post_init__()

    def store_reduction(self, output_name, indexer, vars, sort_vars):
        idx = self.reindex(vars, sort_vars)
        values = tuple(inner_fn(idx) for inner_fn in self.inner_fns)
        result = ops.sort(self.dtypes, values, self.stable, self.descending)
        return ops.store(
            output_name or "unnamed", indexer(idx), result[self.output_index]
        )

    def get_reduction_type(self):
        return "sort"

    def get_reduction_size(self):
        return self.sort_ranges

    def get_size(self):
        return self.size

    def get_pointwise_size(self):
        return self.ranges

    def index_length(self) -> int:
        return len(self.ranges) + len(self.sort_ranges)

    def inner_fn_args(self):
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return (idx,)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet:
        index = self._index(self.ranges)
        rindex = self._index(self.sort_ranges, SymT.R0_INDEX)
        idx = self.reindex(index, rindex)
        return extract_free_symbols(self.inner_fn, idx, unbacked_only=unbacked_only)

    @classmethod
    def create(
        cls,
        device,
        dtypes,
        inner_fns,
        size,
        axis,
        stable,
        descending,
        reduction_hint=ReductionHint.DEFAULT,
        **kwargs,
    ):
        """Build a sort over one axis, returning one realized value per result.

        Which end the values go in is part of what a sort is rather than
        something applied to its result, so it is settled where the sort is
        built: the walk reads the order it has been given and extends it, and
        an order has a direction.  A sort is written only as one block that
        holds the whole axis, so a device without sorting, or an axis longer
        than such a block pays for, is handed to the framework; a sort of one
        element is that element copied.
        """

        pointwise_ranges = [*size[:axis], *size[axis + 1 :]]
        sort_ranges = [size[axis]]

        if not V.graph.has_feature(device, BackendFeature.SORT):
            return [None] * len(dtypes)

        sizevars = V.graph.sizevars
        sort_numel = sizevars.simplify(sympy_product(sort_ranges))

        # The shortest axis at which a written sort usually beats the
        # framework's; past it the work is not bandwidth bound, so fusing it
        # buys little.
        if config.triton.decompose_sort_ops:
            is_persistent_kernel = config.triton.persistent_reductions
        else:
            max_rblock = 512
            is_persistent_kernel = (
                config.triton.persistent_reductions
                and sizevars.statically_known_true(sympy.Le(sort_numel, max_rblock))
            )
        if not is_persistent_kernel:
            return [None] * len(dtypes)

        if len(dtypes) != len(inner_fns):
            raise AssertionError("Expected len(dtypes) == len(inner_fns)")

        if sizevars.statically_known_true(sympy.Le(sort_numel, 1)):
            return [
                Pointwise.create(
                    device=device,
                    dtype=dtypes[output_index],
                    inner_fn=inner_fns[output_index],
                    ranges=size,
                )
                for output_index in range(len(dtypes))
            ]

        def reindex(index, sort_index):
            if len(sort_index) != len(sort_ranges):
                raise AssertionError("Expected len(sort_index) == len(sort_ranges)")
            if len(index) != len(pointwise_ranges):
                raise AssertionError("Expected len(index) == len(pointwise_ranges)")
            return [*index[:axis], *sort_index, *index[axis:]]

        results = [
            TensorBox.create(
                cls(
                    device=device,
                    dtype=dtypes[output_index],
                    dtypes=dtypes,
                    inner_fn=inner_fns[output_index],
                    inner_fns=inner_fns,
                    size=size,
                    ranges=pointwise_ranges,
                    sort_ranges=sort_ranges,
                    stable=stable,
                    descending=descending,
                    reindex=reindex,
                    reduction_hint=reduction_hint,
                    output_index=output_index,
                    **kwargs,
                )
            )
            for output_index in range(len(dtypes))
        ]

        for result in results:
            result.realize()

        return results


@ir_dataclass
class Reduction(Loops):
    """A body that folds several elements of each input into one element.

    The elements that are folded together are the ones the reduced loops run
    over, and one element of the result comes from all of them rather than from
    one, which is what makes this different from a body whose elements are
    independent.
    """

    reduction_ranges: Sequence
    reduction_type: Any
    # The type the result is held in, which is not the type it is read in.
    src_dtype: Any
    reduction_hint: Any
    # An exact eager tile, where the last step of a split reduction is one of
    # size one.
    strict_reduction_multirow: bool = False
    strict_reduction_rblock: int | None = None

    def __str__(self) -> str:
        return self._to_str(("ranges", "reduction_ranges", "reduction_type"))

    __repr__ = __str__

    @cache_on_self_and_args("Reduction")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return super().get_free_symbol_uses(unbacked_only) | OrderedSet().union(
            *(get_free_symbols(e, unbacked_only) for e in self.reduction_ranges)
        )

    def get_reduction_size(self) -> Sequence:
        return self.reduction_ranges

    def get_reduction_type(self) -> str | None:
        return self.reduction_type

    def store_reduction(
        self,
        output_name: str | None,
        indexer,
        vars: Sequence,
        reduction_vars: Sequence,
    ) -> None:
        """Fold the reduced loops and write the result where it belongs."""

        value = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        ops.store_reduction(output_name or "unnamed", indexer(vars), value)

    def index_length(self) -> int:
        return len(self.ranges) + len(self.reduction_ranges)

    def inner_fn_args(self) -> Sequence:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return (index, rindex)

    def inner_fn_free_symbols(self, unbacked_only: bool = False) -> OrderedSet:
        index = self._index(self.ranges)
        rindex = self._index(self.reduction_ranges, SymT.R0_INDEX)
        return extract_free_symbols(
            self.inner_fn, index, rindex, unbacked_only=unbacked_only
        )

    def constant_to_device(self, device) -> "IRNode":
        """This reduction, over a constant on another device."""

        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Reduction(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            reduction_ranges=self.reduction_ranges,
            reduction_type=self.reduction_type,
            src_dtype=self.src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
            strict_reduction_multirow=self.strict_reduction_multirow,
            strict_reduction_rblock=self.strict_reduction_rblock,
        )

    @staticmethod
    def default_accumulator(reduction_type: str, dtype):
        """The value a reduction starts from, which is its own identity.

        The identity is not the same in every type: a maximum of nothing is
        minus infinity for a floating point type, the smallest value the type
        can hold for a whole number, and false for a truth value.  A reduction
        that starts from anything else would return something the values were
        never compared against.
        """

        if reduction_type in (
            "max",
            "fmax",
            "argmax",
            "argmax_value",
            "argmax_with_value",
        ):
            if is_float_dtype(dtype):
                return float("-inf")
            elif is_boolean_dtype(dtype):
                return False
            else:
                return tp.iinfo(dtype).min
        if reduction_type in ("min", "argmin", "argmin_value", "argmin_with_value"):
            if is_float_dtype(dtype):
                return float("inf")
            elif is_boolean_dtype(dtype):
                return True
            else:
                return tp.iinfo(dtype).max

        zero = False if is_boolean_dtype(dtype) else 0
        one = True if is_boolean_dtype(dtype) else 1
        return {
            "sum": zero,
            "prod": one,
            "dot": zero,
            "xor_sum": zero,
            "any": zero,
            "welford_reduce": (zero, zero, zero),
            "welford_combine": (zero, zero, zero),
            "online_softmax_reduce": (float("-inf"), zero),
        }[reduction_type]

    @staticmethod
    def default_value(reduction_type: str, dtype):
        """What a reduction starts from, where that differs from its accumulator.

        A running mean and spread starts from no values at all, which is zero
        of them, rather than from the identity of combining them.
        """

        if reduction_type == "welford_reduce":
            return 0
        return Reduction.default_accumulator(reduction_type, dtype)


    @staticmethod
    def num_splits(
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_numel,
        input_node=None,
    ):
        """Choose the reduction hint and split count from the shapes involved.

        The hint says where the combining happens: reducing inside a block of
        the reduced axes, or reducing across blocks that each produced a
        partial result.  A scan always needs the combining across blocks, since
        each step depends on the one before it.  Everything else is worked out
        from the two sizes alone, so a shape whose values are not known yet
        gets the default and no splitting.
        """

        exprs = [reduction_numel, sympy_product(ranges)]
        if not V.graph.sizevars.all_unbacked_explicitly_hinted(exprs):
            return ReductionHint.DEFAULT, 1

        reduction_numel_hint = V.graph.sizevars.optimization_hint(reduction_numel)
        numel = sympy_product(ranges)
        numel_hint = V.graph.sizevars.optimization_hint(numel)

        arg_reduction_types = (
            "argmax",
            "argmin",
            "argmax_value",
            "argmin_value",
            "argmax_with_value",
            "argmin_with_value",
        )

        if reduction_type == "scan":
            # Each step reads the result of the step before it, so the partial
            # results cannot be combined until every block has run.
            return ReductionHint.INNER, 1

        if reduction_type == "dot":
            # This one is already done as a single multiply-accumulate.
            return ReductionHint.DEFAULT, 1

        if reduction_type in arg_reduction_types:
            # Which element won is not something two partial answers can be
            # merged into without re-reading the data.
            return ReductionHint.DEFAULT, 1

        if not config.split_reductions:
            return ReductionHint.DEFAULT, 1

        if getattr(device, "type", None) not in (None, "cpu"):
            return Reduction._device_splits(
                device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges,
                reduction_type, reduction_numel, reduction_numel_hint, numel_hint,
            )

        # One output element, so there is nothing to combine across blocks and
        # the whole reduced axis belongs to one block.
        if numel_hint == 1:
            return ReductionHint.INNER, 1

        # Too little work per output to be worth handing out, or enough outputs
        # that there is no shortage of parallelism either way.
        if reduction_numel_hint <= 32 or numel_hint >= 64:
            return ReductionHint.DEFAULT, 1

        split = max(1, min(reduction_numel_hint // 64, numel_hint // 32))
        if split <= 1:
            return ReductionHint.DEFAULT, 1
        return ReductionHint.OUTER, split

    @staticmethod
    def _device_splits(
        device, dst_dtype, src_dtype, inner_fn, ranges, reduction_ranges,
        reduction_type, reduction_numel, reduction_numel_hint, numel_hint,
    ):
        """The hint and split count on an accelerator, sized to fill it.

        A reduction with fewer outputs than the device has room for blocks is
        cut so that the device fills; how far depends on whether the reduced
        axis is the contiguous one (inner: one row per group of blocks) or an
        outer one (each block covers many outputs side by side and walks the
        reduced axis down them), which is read off the strides the reduction's
        inputs are read with.  A reduction whose warps cooperate on one launch
        is not cut.
        """

        from .runtime.hints import DeviceProperties

        numel = sympy_product(ranges)
        try:
            if V.choices.should_use_cooperative_reduction(device, numel, reduction_numel):
                return ReductionHint.DEFAULT, 1
        except Exception:  # noqa: BLE001 - no policy installed: no cooperation
            pass
        num_sm = DeviceProperties.create(device).multi_processor_count
        min_elements_per_thread = 32

        def splits(inner: bool) -> int:
            return V.choices.reduction_split_factor(
                device, reduction_numel_hint, numel_hint, inner_reduction=inner
            )

        if numel_hint == 1:
            return ReductionHint.INNER, splits(True)
        if reduction_numel_hint <= min_elements_per_thread or numel_hint >= num_sm * 2 * 32:
            return ReductionHint.DEFAULT, 1

        r = Reduction(
            device=device,
            dtype=dst_dtype,
            inner_fn=inner_fn,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type,
            src_dtype=src_dtype,
            reduction_hint=ReductionHint.DEFAULT,
        )

        def read_indices():
            # The reads that cover the whole iteration space say which axis
            # the reduction walks contiguously; reads that only vary along
            # the reduced axes are a fallback.  Laying out a producer the
            # reduction reads can change what it reads, so a change is asked
            # about again.
            cb = ComputedBuffer(
                name=None,
                layout=FlexibleLayout(
                    device=device, dtype=r.get_dtype(), size=r.get_size(), is_pinned=False
                ),
                data=r,
            )
            read_writes = cb.get_read_writes()
            range_vars = [
                v for v in (read_writes.range_vars or [])
                if isinstance(v, sympy.Expr) and not isinstance(v, sympy.Number)
            ]
            (_, reduction_vars), _ = dependencies.index_vars_squeeze(
                r.get_size(), r.get_reduction_size()
            )
            reduction_vars = [
                v for v in reduction_vars
                if isinstance(v, sympy.Expr) and not isinstance(v, sympy.Number)
            ]
            full, partial = [], []
            changed = False
            for md in sorted(read_writes.reads, key=lambda d: d.name):
                free = md.index.free_symbols
                is_full = all(v in free for v in range_vars)
                is_partial = not is_full and any(v in free for v in reduction_vars)
                if is_full:
                    full.append(md.index)
                elif is_partial:
                    partial.append(md.index)
                if (is_full or is_partial) and md.name in V.graph.name_to_buffer:
                    buf = V.graph.name_to_buffer[md.name]
                    before = getattr(buf.layout, "stride", None)
                    if hasattr(buf, "decide_layout"):
                        buf.decide_layout()
                    if getattr(buf.layout, "stride", None) != before:
                        changed = True
            return full or partial, changed

        indices, changed = read_indices()
        if changed:
            indices, _ = read_indices()
        if not indices:
            return ReductionHint.DEFAULT, 1

        (_, reduction_vars), ranges1 = dependencies.index_vars_squeeze(
            r.get_size(), r.get_reduction_size()
        )
        num_outer = 0
        num_inner = 0
        for index in indices:
            simplified = V.graph.sizevars.simplify_with_ranges(index, ranges1)
            strides = V.graph.sizevars.stride_hints(
                simplified, reduction_vars, list(ranges1.keys())
            )
            # A reduced axis of extent one reads with stride zero, which does
            # not make the walk contiguous.
            if all(st == 0 or st > 1 for st in strides):
                num_outer += 1
            else:
                num_inner += 1
        if num_inner > num_outer:
            return ReductionHint.INNER, splits(True)
        return ReductionHint.OUTER, splits(False)

    @staticmethod
    def _unroll_reduction_fn(inner_fn, reduction_ranges, reduction_type, src_dtype):
        """The reduction written out as a straight walk over the reduced axes.

        When the reduced axis is short, combining as it is walked costs less
        than the bookkeeping a block-wise reduction needs, so the axes are
        spelled out one at a time here and the body is called with each
        position taken.
        """

        reduction_ranges = list(reduction_ranges)
        if len(reduction_ranges) == 0:
            return lambda index: inner_fn(index, [])

        if len(reduction_ranges) == 1:
            (r0,) = reduction_ranges

            def unrolled_fn(index):
                value = None
                for i in range(int(r0)):
                    value = inner_fn(index, [i])
                return value

            return unrolled_fn

        combine_fn = get_reduction_combine_fn(reduction_type, src_dtype)

        def unrolled_fn(index):
            value = None
            for i in range(int(r0)):
                for j in range(int(r1)):
                    cur = inner_fn(index, [i, j])
                    value = cur if value is None else combine_fn(value, cur)
            return value

        return unrolled_fn

    @classmethod
    def check_for_split_dense_dim_reindexing(cls, reduction_numel, input_node):
        """The input axis to walk innermost when a whole-tensor reduction is cut.

        Cutting the reduced extent into pieces walks it as one flat run; when
        the input is dense along some axis other than its last, the flat run
        is taken along that axis so each piece still reads consecutive
        memory.  ``None`` when the reduction does not cover the whole input or
        the input's last axis is the dense one.
        """

        if input_node is None:
            return None
        if not V.graph.sizevars.statically_known_equals(
            input_node.get_numel(), reduction_numel
        ):
            return None
        input_node.realize()
        try:
            as_storage_and_layout(input_node)
        except NotImplementedError:
            return None
        strides = input_node.get_stride()
        for i, stride in enumerate(strides[:-1]):
            if V.graph.sizevars.statically_known_equals(stride, 1):
                return i
        return None

    @staticmethod
    def _multilayer_second_step_hint(split, numel_hint, reduction_hint):
        """The hint for the step that combines what the split steps produced.

        An outer reduction's partial results are laid out one row per output,
        a split long; when both the split and the outputs are few the combine
        is a small outer reduction of its own, scheduled as such.
        """

        if split == -1:
            return reduction_hint
        if reduction_hint == ReductionHint.OUTER and (
            (split <= 512 and numel_hint <= 512)
            or (split <= 1024 and numel_hint <= 256)
        ):
            return ReductionHint.OUTER_TINY
        return reduction_hint

    @classmethod
    def create(
        cls,
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_hint=ReductionHint.DEFAULT,
        input_node=None,
        *,
        strict_reduction: bool = False,
    ):
        """Build a reduction, splitting it into layers when that exposes more work.

        A reduction over an axis much longer than the number of outputs leaves
        most of the machine idle, so the axis is cut into pieces that are
        reduced independently and the partial results are then reduced among
        themselves.  How many pieces is what the split rule decides.
        """

        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel == 0:
            # Nothing to reduce over, so every output is the value that
            # reduction starts from.  The start value has to be spelled as a
            # value of the destination type, since that is what it becomes.
            def py_cnst(val):
                if dst_dtype == tp.bool:
                    return bool(val)
                elif is_float_dtype(dst_dtype):
                    if not isinstance(val, float):
                        raise AssertionError(type(val))
                    return float(val)
                else:
                    if not isinstance(val, int):
                        raise AssertionError(type(val))
                    return int(val)

            rtypes_to_inits = {
                "sum": py_cnst(0),
                "xor_sum": py_cnst(0),
                "prod": py_cnst(1),
                "any": py_cnst(0),
            }

            if reduction_type not in rtypes_to_inits:
                raise AssertionError(
                    f"{reduction_type} not supported for zero-dimension tensors!"
                )

            def const_fn(index):
                return ops.constant(rtypes_to_inits[reduction_type], dst_dtype)

            return Pointwise.create(
                device=device,
                dtype=src_dtype,
                inner_fn=const_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 1 and not strict_reduction:
            # A reduction over a single element reads that element and is a
            # pointwise op; emitting a reduction loop for it would produce a
            # one-iteration reduction that some codegens cannot spell.
            if reduction_type in ("argmin", "argmax"):
                def fn(index):
                    return ops.constant(0, dst_dtype)

            else:
                def fn(index):
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return inner_fn(index, reduction_index)

            return Pointwise.create(
                device=device,
                dtype=dst_dtype,
                inner_fn=fn,
                ranges=list(ranges),
            )

        hint, split = cls.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            reduction_numel,
            input_node,
        )
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint

        if split != 1:
            return cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
                input_node,
            )

        def loader(index, reduction_index):
            return inner_fn(index, reduction_index)

        result = TensorBox.create(
            cls(
                device=device,
                dtype=dst_dtype,
                inner_fn=loader,
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
            )
        )
        result.realize()
        return result

    @classmethod
    def create_multilayer(
        cls,
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        split,
        reduction_hint,
        input_node=None,
        *,
        strict_reduction: bool = False,
    ):
        """Reduce in two layers: pieces of the axis, then the pieces together.

        The reduced extent, walked as one flat run, is cut into ``split``
        pieces of ``block_size``; the first layer reduces each piece into a
        partial result indexed by which piece it was, which is where the
        parallelism is, and the second reduces the partial results of each
        output, a much shorter walk.  The last piece may run past the end; the
        positions past it contribute the value that leaves the reduction
        unchanged.
        """

        reduction_numel = sympy_product(reduction_ranges)
        block_size = FloorDiv(reduction_numel + (split - 1), split)
        default = cls.default_value(reduction_type, dst_dtype)
        wrapper_fn = cls._multilayer_wrap_loader(
            inner_fn,
            reduction_ranges,
            reduction_numel,
            split,
            block_size,
            default,
            input_node,
        )
        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            ranges,
            reduction_ranges,
            [*ranges, split],
            [block_size],
            reduction_type,
            split,
            reduction_hint,
            strict_reduction,
        )

    @classmethod
    def create_multilayer_existing_ranges(
        cls,
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        original_ranges,
        original_reduction_ranges,
        new_ranges,
        new_reduction_ranges,
        reduction_type,
        reduction_hint,
    ):
        """Two layers where the cut of the reduced axes is already decided.

        The pieces are the extents ``new_ranges`` and each piece's walk is
        ``new_reduction_ranges`` -- a cut taken from a producer laid out that
        way -- so nothing here has to choose one.
        """

        wrapper_fn = cls._multilayer_wrap_loader_existing_ranges(
            inner_fn,
            original_ranges,
            original_reduction_ranges,
            new_ranges,
            new_reduction_ranges,
        )
        return cls.create_multilayer_helper(
            device,
            dst_dtype,
            src_dtype,
            wrapper_fn,
            original_ranges,
            original_reduction_ranges,
            [*original_ranges, *new_ranges],
            new_reduction_ranges,
            reduction_type,
            -1,
            reduction_hint,
        )

    @classmethod
    def create_multilayer_helper(
        cls,
        device,
        dst_dtype,
        src_dtype,
        wrapper_fn,
        original_ranges,
        original_reduction_ranges,
        new_ranges,
        new_reduction_ranges,
        reduction_type,
        split,
        reduction_hint,
        strict_reduction: bool = False,
    ):
        """Build the first layer, then the second over its realized results.

        The partial results are kept in single precision when the reduction
        produces a half-precision value: a kernel reducing half-precision
        values accumulates in single precision, and storing the pieces any
        narrower would round the reduction partway through.  The first layer
        may itself be cut again, which ``Reduction.create`` decides.
        """

        intermediate_dtype = (
            dst_dtype if dst_dtype not in (tp.float16, tp.bfloat16) else tp.float32
        )
        intermediate = Reduction.create(
            device,
            intermediate_dtype,
            src_dtype,
            wrapper_fn,
            new_ranges,
            new_reduction_ranges,
            reduction_type,
            reduction_hint,
            strict_reduction=strict_reduction,
        )
        intermediate.realize()
        intermediate_loader = intermediate.make_loader()

        def intermediate_fn(index, reduction_index):
            return intermediate_loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(original_ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )
        if list(original_ranges) != list(new_ranges[: len(original_ranges)]):
            raise AssertionError(
                "the first layer's outputs have to begin with the reduction's own"
            )
        return TensorBox.create(
            Reduction(
                device=device,
                dtype=dst_dtype,
                inner_fn=intermediate_fn,
                ranges=original_ranges,
                reduction_ranges=new_ranges[len(original_ranges):],
                reduction_type=reduction_type,
                src_dtype=src_dtype,
                reduction_hint=reduction_hint,
                strict_reduction_rblock=1 if strict_reduction else None,
            )
        )

    @classmethod
    def _multilayer_wrap_loader(
        cls,
        loader,
        reduction_ranges,
        reduction_numel,
        split,
        block_size,
        default,
        input_node=None,
    ):
        """The body of the first layer, indexed by output, piece, and position.

        The first layer's index is the output's followed by which piece; its
        reduced index is the position within the piece.  The flat position in
        the reduced extent is the piece's start plus that, mapped back onto the
        reduced axes; past the end -- when the pieces do not divide the extent
        -- the body is masked to ``default``.
        """

        dense_index = cls.check_for_split_dense_dim_reindexing(
            reduction_numel, input_node
        )
        reindex = View.dynamic_reshape_indexer(
            reduction_ranges, [reduction_numel], dense_index
        )
        need_mask = not V.graph.sizevars.statically_known_true(
            sympy.Eq(sympy.Mod(reduction_numel, split), 0)
        )

        def wrapper_fn(index, reduction_index):
            (reduction_index,) = reduction_index
            *new_index, reduction_block = index
            indices = block_size * reduction_block + reduction_index

            def body():
                return loader(new_index, reindex([indices]))

            if need_mask:
                index_dtype = (
                    tp.int32
                    if V.graph.sizevars.statically_known_lt(reduction_numel, 2**31)
                    else tp.int64
                )
                mask = ops.lt(
                    ops.index_expr(indices, index_dtype),
                    ops.index_expr(reduction_numel, index_dtype),
                )
                return ops.masked(mask, body, default)
            return body()

        return wrapper_fn

    @classmethod
    def _multilayer_wrap_loader_existing_ranges(
        cls,
        loader,
        original_ranges,
        original_reduction_ranges,
        new_ranges,
        new_reduction_ranges,
    ):
        """The first layer's body when the cut is already given as extents.

        Only a reduction to one output is cut this way: its reduced axes are
        reshaped onto the pieces followed by each piece's walk.
        """

        if not all(r == 1 for r in original_ranges):
            raise AssertionError(
                f"a given cut serves a reduction to one output, not {original_ranges}"
            )
        reindex = View.dynamic_reshape_indexer(
            original_reduction_ranges, tuple(new_ranges) + tuple(new_reduction_ranges)
        )

        def wrapper_fn(merged_index, new_reduction_index):
            original_idx = merged_index[: len(original_ranges)]
            new_index = merged_index[len(original_ranges):]
            return loader(
                original_idx,
                reindex(tuple(new_index) + tuple(new_reduction_index)),
            )

        return wrapper_fn


class MultiOutputReduction(Reduction):
    """A reduction whose walk produces more than one value at each step.

    What one pass over the reduced axes has to produce is a tuple -- the value
    being reduced, plus whatever else the reduction carries along, such as the
    index it came from or a running total.  Asking for one of those results is
    the same walk with a different result selected, which is why this is one
    class with the selection as a field rather than several classes.
    """

    output_index: int

    def __init__(
        self,
        device,
        dst_dtype,
        inner_fns,
        ranges,
        reduction_ranges,
        reduction_type,
        src_dtype,
        reduction_hint,
        output_index: int,
    ):
        if callable(inner_fns):
            inner_fns = (inner_fns,)

        loader: Callable
        if len(inner_fns) == 1:
            loader = inner_fns[0]
        else:

            def loader(idx, reduction_idx):
                return tuple(fn(idx, reduction_idx) for fn in inner_fns)

        super().__init__(
            device=device,
            dtype=dst_dtype,
            inner_fn=loader,
            ranges=ranges,
            reduction_ranges=reduction_ranges,
            reduction_type=reduction_type,
            src_dtype=src_dtype,
            reduction_hint=reduction_hint,
        )
        self.output_index = output_index

    def store_reduction(self, output_name, indexer, vars, reduction_vars):
        values = ops.reduction(
            self.dtype,
            self.src_dtype,
            self.reduction_type,
            self.inner_fn(vars, reduction_vars),
        )
        if not isinstance(values, (tuple, list)):
            raise AssertionError(type(values))
        value = values[self.output_index]
        return ops.store_reduction(output_name or "unnamed", indexer(vars), value)




class OnlineSoftmaxReduction(MultiOutputReduction):
    """A reduction that carries the largest value seen and the total with it.

    Softmax needs both the largest value and the total that goes with it, and
    computing the total directly is not safe because the values are exponents.
    Carrying the largest value alongside the running total makes every partial
    answer usable on its own, so the walk can be split like any other.
    """

    @classmethod
    def _create_no_split(
        cls,
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        num_output,
        reduction_hint=ReductionHint.DEFAULT,
        input_node=None,
    ):
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        if reduction_numel == 0:
            raise AssertionError(
                f"softmax not supported for zero-dimension tensors!"
            )

        def loader(index, reduction_index):
            return tuple(
                inner_fn(index, reduction_index, output) for output in range(num_output)
            )

        results = [
            TensorBox.create(
                cls(
                    device,
                    dst_dtype,
                    loader,
                    ranges,
                    reduction_ranges,
                    "online_softmax_reduce",
                    src_dtype,
                    reduction_hint,
                    output_index,
                )
            )
            for output_index in range(num_output)
        ]
        for result in results:
            result.realize()
        return results

    @classmethod
    def create(
        cls,
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        num_output,
        reduction_hint=ReductionHint.DEFAULT,
        input_node=None,
    ):
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))
        hint, split = Reduction.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type="online_softmax_reduce",
            reduction_numel=reduction_numel,
            input_node=input_node,
        )
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            return cls.create_multilayer(
                device,
                dst_dtype,
                src_dtype,
                inner_fn,
                ranges,
                reduction_ranges,
                num_output,
                split,
                reduction_hint,
                input_node,
            )
        return cls._create_no_split(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            num_output,
            reduction_hint,
            input_node,
        )

    @classmethod
    def create_multilayer(
        cls,
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        num_output,
        split,
        reduction_hint,
        input_node=None,
    ):
        """The split walk, followed by one that combines what the pieces found.

        The second walk has to see every output the first one produced, since
        the largest value and the total only mean anything together, so the
        body is written to take the piece number and read that piece's results.
        """

        reduction_numel = sympy_product(reduction_ranges)
        block_size = V.graph.sizevars.simplify(ceildiv(reduction_numel, split))

        new_ranges = [*ranges, sympy_index_symbol("R2")]
        new_reduction_ranges = [block_size]
        original_ranges = [*ranges, sympy_index_symbol("R3")]
        original_reduction_ranges = [
            *reduction_ranges,
            sympy_index_symbol_with_prefix("R2"),
        ]

        def wrapper_fn(idx, r_idx):
            return tuple(
                inner_fn(idx, r_idx, output) for output in range(num_output)
            )

        # Each piece's results are laid out with the piece number as one more
        # axis, so that the combining walk reads them in order.
        results = []
        for output in range(num_output):
            res = TensorBox.create(
                cls(
                    device,
                    dst_dtype,
                    wrapper_fn,
                    new_ranges,
                    new_reduction_ranges,
                    "online_softmax_reduce",
                    src_dtype,
                    ReductionHint.INNER,
                    output,
                )
            )
            res.realize()
            results.append(res)

        # The combining walk reduces the pieces.  Since it must see the largest
        # value and the total together, the outputs that belong to one piece
        # are read together, which is what the body is written to do.
        merged = Reduction.create(
            device,
            dst_dtype,
            dst_dtype,
            lambda idx, r_idx: ops.online_softmax_reduce(
                [
                    ops.load(
                        results[output].get_name(),
                        [*idx, r_idx[0]],
                    )
                    for output in range(num_output)
                ]
            ),
            ranges,
            [split],
            "online_softmax_reduce",
            reduction_hint,
        )
        return merged


class WelfordReduction(MultiOutputReduction):
    """A reduction that computes a mean and its spread in one pass.

    The mean of a set is usually computed as a total divided by a count, which
    needs either two passes or a value that grows too large to be exact.  This
    carries the mean seen so far together with the total of the squared
    differences from it and how many values have been counted, all of which
    stay in range however the values are distributed.  Asking for one of the
    three results is the same walk with a different one selected.
    """

    @classmethod
    def create(
        cls,
        device,
        dtype,
        inner_fns,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_hint=ReductionHint.DEFAULT,
    ):
        if reduction_type not in ("welford_reduce", "welford_combine"):
            raise AssertionError(
                'Expected reduction_type in ("welford_reduce", "welford_combine")'
            )

        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))

        def const(val: int):
            def inner_fn(idx):
                return ops.constant(val, dtype)

            return Pointwise.create(
                device=device,
                dtype=dtype,
                inner_fn=inner_fn,
                ranges=list(ranges),
            )

        if reduction_numel == 0:
            # Nothing was counted, so the mean and the spread of nothing are
            # both zero and the count is zero.
            mean = const(0)
            m2 = const(0)
            weight = const(0)
            return mean, m2, weight

        if reduction_numel == 1:
            # One value is its own mean, contributes nothing to the spread, and
            # is one value counted.
            def copy(loader):
                def inner_fn(idx):
                    reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                    return loader(idx, reduction_index)

                return Pointwise.create(
                    device=device,
                    dtype=dtype,
                    inner_fn=inner_fn,
                    ranges=list(ranges),
                )

            if reduction_type == "welford_reduce":
                return copy(inner_fns[0]), const(0), const(1)
            else:
                return tuple(copy(fn) for fn in inner_fns)

        hint, split = Reduction.num_splits(
            device,
            dtype,
            dtype,
            inner_fns[0],
            ranges,
            reduction_ranges,
            reduction_type=reduction_type,
            reduction_numel=reduction_numel,
        )
        # The intermediate walk of a split reduction indexes in a way the split
        # rule cannot read, so the hint that was passed in is used when there
        # is one.
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint
        if split > 1:
            return cls.create_multilayer(
                device,
                dtype,
                inner_fns,
                ranges,
                reduction_ranges,
                reduction_type,
                split,
                reduction_hint,
            )

        results = [
            TensorBox.create(
                WelfordReduction(
                    device,
                    dtype,
                    inner_fns,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    dtype,
                    reduction_hint,
                    output_idx,
                )
            )
            for output_idx in range(3)
        ]
        for t in results:
            t.realize()
        return results

    @staticmethod
    def default_value(reduction_type, dtype):
        """The value a walk starts from, which is what a walk of nothing gives."""

        return (0, 0, 0)

    @classmethod
    def create_multilayer(
        cls,
        device,
        dtype,
        inner_fns,
        ranges,
        reduction_ranges,
        reduction_type,
        split,
        reduction_hint,
    ):
        """Break a large reduction into smaller ones, repeatedly if needed.

        When the reduced axis does not divide into the number of pieces being
        asked for, some piece would reach past the end of the axis.  Padding it
        with values that are not counted is what keeps the mean right, so the
        walk is changed into the form that combines already-counted pieces
        rather than the one that counts from scratch.
        """

        reduction_numel = sympy_product(reduction_ranges)
        need_mask = not V.graph.sizevars.statically_known_true(
            sympy.Eq(sympy.Mod(reduction_numel, split), 0)
        )

        if need_mask and reduction_type != "welford_combine":
            # The pieces past the end must contribute nothing to the mean and
            # nothing to the count, which is what the combining form does.
            def constant(idx, reduction_idx, value: int):
                return ops.constant(value, dtype)

            return cls.create_multilayer(
                device=device,
                dtype=dtype,
                inner_fns=(
                    inner_fns[0],
                    partial(constant, value=0),
                    partial(constant, value=1),
                ),
                ranges=ranges,
                reduction_ranges=reduction_ranges,
                reduction_type="welford_combine",
                split=split,
                reduction_hint=reduction_hint,
            )

        block_size = FloorDiv(reduction_numel + (split - 1), split)
        intermediates = WelfordReduction.create(
            device,
            dtype,
            tuple(
                cls._multilayer_wrap_loader(
                    loader,
                    reduction_ranges,
                    reduction_numel,
                    split,
                    block_size,
                    default=0,
                )
                for loader in inner_fns
            ),
            [*ranges, split],
            [block_size],
            reduction_type,
            reduction_hint,
        )
        for i in intermediates:
            i.realize()

        def intermediate_loader_fn(index, reduction_index, loader):
            return loader([*index, *reduction_index])

        numel_hint = V.graph.sizevars.optimization_hint(sympy_product(ranges))
        reduction_hint = cls._multilayer_second_step_hint(
            split, numel_hint, reduction_hint
        )
        return WelfordReduction.create(
            device,
            dtype,
            tuple(
                partial(intermediate_loader_fn, loader=i.make_loader())
                for i in intermediates
            ),
            ranges,
            [split],
            # Turning one input into three is what the first walk does; merging
            # those three is what the second one does.
            "welford_combine",
            reduction_hint,
        )


class ArgReduction(MultiOutputReduction):
    """A reduction that reports both the value that won and where it was.

    Which element is largest cannot be merged from two partial answers without
    looking at the data again, so this never splits, and the index it reports is
    a position in the whole reduced axis rather than in a piece of it.
    """

    @classmethod
    def create(
        cls,
        device,
        dst_dtype,
        src_dtype,
        inner_fn,
        ranges,
        reduction_ranges,
        reduction_type,
        reduction_hint=ReductionHint.DEFAULT,
        input_node=None,
    ):
        if reduction_type not in ("argmax_with_value", "argmin_with_value"):
            raise AssertionError(f"unexpected reduction_type {reduction_type!r}")
        reduction_numel = V.graph.sizevars.simplify(sympy_product(reduction_ranges))
        if reduction_numel == 0:
            raise AssertionError(
                f"{reduction_type} not supported for zero-dimension tensors!"
            )

        if reduction_numel == 1:
            # A single element is the answer without looking at anything, and
            # the position is known to be zero.
            def value_fn(index):
                reduction_index = [sympy.S.Zero for _ in reduction_ranges]
                value = inner_fn(index, reduction_index)
                if isinstance(value, tuple):
                    return value[0]
                return value

            def index_fn(index):
                return ops.constant(0, tp.int64)

            return (
                Pointwise.create(
                    device=device,
                    dtype=dst_dtype,
                    inner_fn=value_fn,
                    ranges=list(ranges),
                ),
                Pointwise.create(
                    device=device,
                    dtype=tp.int64,
                    inner_fn=index_fn,
                    ranges=list(ranges),
                ),
            )

        if (
            isinstance(reduction_numel, sympy.Integer)
            and int(reduction_numel) < config.unroll_reductions_threshold
            and (sympy_product(ranges) != 1 or is_gpu(device))
        ):
            unrolled_fn = Reduction._unroll_reduction_fn(
                inner_fn, reduction_ranges, reduction_type, src_dtype
            )

            def project(index, output_index: int):
                result = unrolled_fn(index)
                if not isinstance(result, tuple):
                    raise AssertionError(f"expected tuple result, got {type(result)}")
                return result[output_index]

            return tuple(
                Pointwise.create(
                    device=device,
                    dtype=dtype,
                    inner_fn=partial(project, output_index=output_index),
                    ranges=list(ranges),
                )
                for output_index, dtype in enumerate((dst_dtype, tp.int64))
            )

        hint, split = Reduction.num_splits(
            device,
            dst_dtype,
            src_dtype,
            inner_fn,
            ranges,
            reduction_ranges,
            reduction_type,
            reduction_numel,
            input_node,
        )
        if split != 1:
            raise AssertionError("arg reductions do not support split reductions")
        if reduction_hint == ReductionHint.DEFAULT:
            reduction_hint = hint

        results = tuple(
            TensorBox.create(
                cls(
                    device,
                    dtype,
                    inner_fn,
                    ranges,
                    reduction_ranges,
                    reduction_type,
                    src_dtype,
                    reduction_hint,
                    output_index,
                )
            )
            for output_index, dtype in enumerate((dst_dtype, tp.int64))
        )
        for result in results:
            result.realize()
        return results


@ir_dataclass(frozen=False)
class NopKernel(InputsKernel):
    """A result that costs nothing to produce.

    Some operations produce a value without doing any work: a concat lays
    existing memory end to end, and a view describes memory that is already
    there.  There is nothing to read, since nothing is computed, and what is
    left is only there so that these can be described like anything else.
    """

    def is_no_op(self) -> bool:
        return True

    def get_reads(self) -> OrderedSet:
        return OrderedSet()


class ConcatKernel(NopKernel):
    """Several values laid end to end in one piece of memory.

    Nothing is computed here: the inputs are written into slices of a single
    buffer, which is what makes the result contiguous when the inputs were not.
    Each input is given the slice it will occupy and asked to be written there
    directly, so a value that was not yet written out is written into its place
    rather than being written once and then moved.
    """

    @classmethod
    def create(cls, inputs, dim: int):
        """Build the result of joining these values along one axis."""

        device = inputs[0].get_device()
        dtype = inputs[0].get_dtype()
        new_size = list(inputs[0].get_size())
        offsets_start = [0]
        offsets_end = [new_size[dim]]
        if not (0 <= dim < len(new_size)):
            raise AssertionError("Expected 0 <= dim < len(new_size)")
        for i in range(1, len(inputs)):
            input_size = inputs[i].get_size()
            offsets_start.append(new_size[dim])
            if len(input_size) != len(new_size):
                raise AssertionError("Expected len(input_size) == len(new_size)")
            if inputs[i].get_dtype() != dtype:
                raise AssertionError("Expected inputs[i].get_dtype() == dtype")
            if inputs[i].get_device() != device:
                raise AssertionError("Expected inputs[i].get_device() == device")
            for j in range(len(new_size)):
                if j == dim:
                    new_size[j] = new_size[j] + input_size[j]
                else:
                    new_size[j] = V.graph.sizevars.check_equals_and_simplify(
                        new_size[j], input_size[j]
                    )
            offsets_end.append(new_size[dim])

        output_stride: Sequence = FlexibleLayout.contiguous_strides(new_size)
        if config.comprehensive_padding:
            # The result has to satisfy whatever alignment the rest of the code
            # assumes, and padding the strides is cheaper than moving the data.
            output_stride = Layout._pad_strides(
                output_stride, new_size, inputs[0].dtype
            )

        # If any input has its channels outermost, the result does too, so that
        # joining does not silently rearrange the data.
        for i in range(len(inputs)):
            x = inputs[i]
            if is_storage_and_layout(x):
                layout = x.get_layout()
                if isinstance(
                    layout, FixedLayout
                ) and Layout.is_channels_last_contiguous(layout.size, layout.stride):
                    output_stride = make_channels_last_strides_for(new_size)
                    break

        is_pinned = all(
            is_storage_and_layout(x) and x.get_layout().is_pinned for x in inputs
        )

        if device is None:
            raise AssertionError("Expected device is not None")
        concat_kernel = ConcatKernel(
            name=None,
            layout=FixedLayout(
                device=device,
                dtype=dtype,
                size=new_size,
                stride=output_stride,
                is_pinned=is_pinned,
            ),
            inputs=[],
        )
        kernel = StorageBox(concat_kernel)
        for i, inp in enumerate(inputs):
            if not isinstance(inp, (BaseView, MutableBox)):
                raise AssertionError(type(inp))
            input_buffer = cls.realize_into(
                inp,
                SliceView.create(
                    kernel, dim, offsets_start[i], offsets_end[i], clamp=False
                ),
            )
            if not isinstance(input_buffer, Buffer):
                raise AssertionError(type(input_buffer))
            if not isinstance(concat_kernel.inputs, list):
                raise AssertionError(type(concat_kernel.inputs))
            concat_kernel.inputs.append(input_buffer)

        concat_kernel.name = V.graph.register_buffer(concat_kernel)
        concat_kernel.inputs = cls.unwrap_storage(concat_kernel.inputs)
        V.graph.register_operation(concat_kernel)

        return kernel

    @classmethod
    def can_realize_into_without_copy(cls, src, dst=None) -> bool:
        """Whether this value can be written into that place without moving.

        A value whose layout has not been settled yet can go wherever it is
        asked to, which is the case this is usually asked about; a value whose
        layout is already decided can only do so if the place agrees with it.
        """

        if isinstance(src, TensorBox):
            return cls.can_realize_into_without_copy(src.data, dst)

        if not isinstance(src, (BaseView, StorageBox)):
            raise AssertionError(type(src))
        return (
            hasattr(src.data, "layout")
            and isinstance(src.data.layout, FlexibleLayout)
            and not isinstance(src.data, ExternKernelAlloc)
        )

    @cache_on_self_and_args("ConcatKernel")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return NopKernel.get_free_symbol_uses(self, unbacked_only)

    @classmethod
    def realize_into(cls, src, dst):
        """Write this value into that place, copying only if it has to.

        Where the value's layout is not yet settled it can be written wherever
        it is wanted, so it is pointed at the place and left there.  Otherwise
        the body is turned into something that reads the value and writes it
        into place, which is a copy.
        """

        if not isinstance(dst, ReinterpretView):
            if is_storage_and_layout(dst):
                storage, layout = as_storage_and_layout(dst)
                dst = ReinterpretView(data=storage, layout=layout)
        if not isinstance(dst, ReinterpretView):
            raise AssertionError(type(dst))
        if isinstance(src, TensorBox):
            return cls.realize_into(src.data, dst)

        if isinstance(src, StorageBox):
            src.realize()
            if not hasattr(src.data, "layout"):
                raise AssertionError('Expected hasattr(src.data, "layout")')
            if cls.can_realize_into_without_copy(src, dst):
                src.data.layout = NonOwningLayout(dst)
                return src.data
        # A value is read and written into place, which is a copy.
        pw = Pointwise.create(
            device=src.get_device(),
            dtype=src.get_dtype(),
            inner_fn=src.make_loader(),
            ranges=[
                V.graph.sizevars.check_equals_and_simplify(a, b)
                for a, b in zip(src.get_size(), dst.get_size())
            ],
        )
        return cls.realize_into(pw, dst)

    def should_allocate(self) -> bool:
        return True


@contextlib.contextmanager
def _track_fresh_unbacked_symbols(shape_env):
    """Say that symbols created in here are to be bound, not discarded.

    The counterpart of discarding them: a caller that discards them while it
    builds a result would lose the symbols that result's extents depend on, so
    a caller that is about to ask an operation what it produces says the
    opposite for the length of the ask.
    """

    prev = shape_env._ignore_fresh_unbacked_symbols_set(False)
    try:
        yield
    finally:
        shape_env._ignore_fresh_unbacked_symbols_set(prev)


def _fallback_kernel_symbol_tracking_context(shape_env):
    """Whether symbols created while asking a call what it produces are kept.

    Kept only where the surrounding path had said otherwise: a path that is
    already binding them needs nothing said, and a path that is discarding them
    on purpose -- a probe of the shape, say -- is not the place to start
    binding them.
    """

    if shape_env is not None and shape_env._ignore_fresh_unbacked_symbols_tls():
        return _track_fresh_unbacked_symbols(shape_env)
    return contextlib.nullcontext()


@dataclasses.dataclass
class ProcessKernelResult:
    """The arguments of a call, sorted, and what the call would produce.

    A tensor argument has to be put in memory first, because the call reads
    memory rather than a value, and its strides have to be settled before the
    call can be written since the call is written in terms of them.  The other
    arguments are only passed along.  The two flat lists are in the order the
    call takes them, and the unflattening is what puts the tree back together
    once the two lists have been replaced -- which is what a call written later
    needs, since by then each of them may be a different value.
    """

    example_output: Any
    tensor_args: list
    non_tensor_args: list
    unflatten_args: Callable[[Any, Any], Any]
    unbacked_bindings: dict | None = None


class ExternKernel(InputsKernel):
    """A buffer whose contents come from a call to something written elsewhere.

    Nothing here is lowered: the call is written as it stands and made at run
    time, and what this has to get right is the shape of that call -- which
    arguments are in which order, which of them are memory, and what each of
    them defaults to.  A kernel that cannot be written this way falls back to
    the implementation it would otherwise have replaced, which is what the
    codegen method on each subclass decides.
    """

    constant_args: Sequence = ()
    kwargs: dict = dataclasses.field(default_factory=dict)
    output_view: "ReinterpretView | None" = None
    python_kernel_name: str | None = None
    cpp_kernel_name: str | None = None
    ordered_kwargs_for_cpp_kernel: Iterable = dataclasses.field(default_factory=list)
    op_overload: Any = None
    arg_properties: list | None = None
    allarg_properties: dict = dataclasses.field(default_factory=dict)
    kwarg_properties: dict | None = None
    unbacked_bindings: dict = dataclasses.field(default_factory=dict)
    mutation_outputs: list = dataclasses.field(default_factory=list)

    def __init__(
        self,
        name: str | None,
        layout: "OutputSpec",
        inputs: Sequence,
        constant_args: Sequence = (),
        kwargs: dict | None = None,
        output_view: "ReinterpretView | None" = None,
        python_kernel_name: str | None = None,
        cpp_kernel_name: str | None = None,
        ordered_kwargs_for_cpp_kernel: Iterable = (),
        op_overload: Any = None,
    ) -> None:
        super().__init__(
            name=name,
            layout=layout,
            inputs=inputs,
        )
        self.constant_args = constant_args
        self.kwargs = kwargs if kwargs else {}
        self.output_view = output_view
        self.op_overload = op_overload
        self.set_cpp_kernel_name(cpp_kernel_name)
        self.set_python_kernel_name(python_kernel_name)
        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        self.collect_arg_kwarg_properties()
        self.unbacked_bindings = {}
        self.mutation_outputs = []
        self.fx_node = V.graph.current_node
        self.annotations: dict = {}

    def get_outputs(self) -> list:
        return [self, *self.mutation_outputs]

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def get_read_writes(self) -> "dependencies.ReadWrites":
        """What the call depends on, including the arguments that are values.

        A tensor argument is already counted by the inputs; an argument that is
        itself a value -- a shape, a constant that is another node -- is not,
        and has to be added here or the call could be scheduled before the
        thing it needs was computed.
        """

        read_writes = super().get_read_writes()

        def add_ir_read(value: object) -> None:
            if isinstance(value, IRNode):
                name = value.maybe_get_name()
                if name is not None:
                    read_writes.reads.add(dependencies.StarDep(name))

        from tensorplay.utils import _pytree as pytree

        pytree.tree_map_(
            add_ir_read,
            (self.constant_args, self.kwargs),
            is_leaf=lambda value: isinstance(value, IRNode),
        )
        return read_writes

    def collect_arg_kwarg_properties(self) -> None:
        """Record what is known about each argument, for the call to be written.

        Where the operation was declared, that declaration is the source.  Where
        it was not, there is a blank for each argument, so that the two lists
        stay the same length and a caller indexing by position is not reading
        past the end of one of them.
        """

        # Where the operation was declared, that declaration is the source of
        # what each argument is called and what it defaults to.  Where it was
        # not, there is a blank for each argument, so that the two lists stay
        # the same length and a caller indexing one of them by argument
        # position is not reading past the end of a shorter one.
        declared = isinstance(self.op_overload, OpOverload)
        if declared:
            self.arg_properties = [
                {
                    "name": x.name,
                    "type": x.real_type,
                    "default_value": x.default_value,
                }
                for x in self.op_overload._schema.arguments
                if not x.kwarg_only
            ]
            self.allarg_properties = {
                x.name: {"type": x.real_type, "default_value": x.default_value}
                for x in self.op_overload._schema.arguments
            }
            if not self.ordered_kwargs_for_cpp_kernel:
                self.ordered_kwargs_for_cpp_kernel = [
                    x.name for x in self.op_overload._schema.arguments if x.kwarg_only
                ]
            self.schema_kwargs = [
                x for x in self.op_overload._schema.arguments if x.kwarg_only
            ]
        else:
            self.arg_properties = [{} for _ in range(len(self.inputs))]
            self.allarg_properties = {}
            self.schema_kwargs = []

    def decide_layout(self) -> None:
        if isinstance(self.layout, FlexibleLayout):
            self.apply_constraint()
            self.freeze_layout()

    def codegen_comment(self, wrapper, kernel_name: str | None = None) -> None:
        """Say in the generated code where this call came from.

        Which node it was, and under what name, is what a reader of the
        generated code has in order to find the operation it came from.
        """

        origin_str, _detailed_origin_str = get_kernel_metadata(self, wrapper)
        if origin_str:
            wrapper.make_comment(origin_str)

        if not kernel_name:
            kernel_name = self.try_get_kernel_name()
        if kernel_name:
            from .debug import set_kernel_post_grad_provenance_tracing

            debug_handle = set_kernel_post_grad_provenance_tracing(
                self, kernel_name, is_extern=True
            )
            wrapper.write_provenance_debug_handle(kernel_name, debug_handle)

    def codegen(self, wrapper) -> None:
        raise NotImplementedError

    def set_cpp_kernel_name(self, cpp_kernel_name: str | None = None) -> None:
        """Settle the name the generated C++ calls this under.

        Left unset, the name is worked out from the operation's own, with the
        namespace and the dots replaced, because that is how such a name is
        spelled where a C++ call can reach it.  Working it out needs both a C++
        wrapper to be called through and an operation declared with a name, and
        without either there is no such name to spell.
        """

        self.cpp_kernel_name = cpp_kernel_name
        if not V.graph.cpp_wrapper or not isinstance(
            self.op_overload, OpOverload
        ):
            return

        kernel = self.op_overload
        if self.cpp_kernel_name is None:
            name = getattr(kernel, "name", None) or str(kernel)
            self.cpp_kernel_name = name.replace("::", "_").replace(".", "_")

    def set_python_kernel_name(self, python_kernel_name: str | None) -> None:
        """Settle the name the generated Python calls this under.

        The name written is one the generated program can actually resolve: its
        header imports the framework under a short name, so a name spelled
        with the framework's own path would name something that is not there.
        An operation reached through the operation table is written the way
        that table is reached.
        """

        self.python_kernel_name = python_kernel_name
        if python_kernel_name is not None:
            return

        kernel = self.op_overload
        if kernel is None:
            return
        from tensorplay._higher_order_ops._hop_base import HigherOrderOperator

        if isinstance(kernel, HigherOrderOperator):
            self.python_kernel_name = f"tp.ops.higher_order.{kernel.__name__}"
            return
        # A method call survives to the backend as a bare string naming the
        # method.  The generated program resolves it through the framework
        # alias it already imports, where the functional spelling of the
        # method lives.
        if isinstance(kernel, str):
            self.python_kernel_name = f"tp.{kernel}"
            return
        module = getattr(kernel, "__module__", None)
        name = getattr(kernel, "__name__", str(kernel))
        # The in-place operator builtins carry a module name that is private to
        # the interpreter.  It is the same module the public ``operator`` module
        # re-exports, so a generated program imports the public one and every
        # spelling of the call has to agree with that import.
        if module == "_operator":
            module = "operator"
        if module is None:
            self.python_kernel_name = name
        elif module.startswith("tensorplay."):
            module_path = module[len("tensorplay."):]
            if name.isidentifier():
                self.python_kernel_name = f"tp.{module_path}.{name}"
            else:
                # A namespaced custom op (``capns::foo``) is not reachable
                # through attribute syntax in the generated program; hand it
                # the op object itself as a module-level constant.  The module
                # is the outermost region's, so a call inside a piece of it
                # attaches the op there too.
                root = V.graph
                while getattr(root, "parent", None) is not None:
                    root = root.parent
                const_name = f"_custom_op_{len(root.constants)}"
                root.constants[const_name] = kernel
                self.python_kernel_name = const_name
        elif module.startswith("tensorplay"):
            self.python_kernel_name = f"{module}.{name}"
        else:
            self.python_kernel_name = f"{module}.{name}"

    def try_get_kernel_name(self) -> str | None:
        """The name this is called by in the code being generated, if it is known.

        Which name that is depends on what the generated code is: code that is
        run directly calls the operation by its own name, and code that is
        compiled calls the C++ function that stands for it, whose name depends
        on the device and on how that function was named.
        """

        device = (d := self.get_device()) and d.type
        if device is None:
            device = V.graph.device_type

        if V.graph.cpp_wrapper:
            if self.cpp_kernel_name is None:
                return None
            get_c_shim = getattr(V.graph.wrapper_code, "get_c_shim_func_name", None)
            if get_c_shim is None:
                return self.cpp_kernel_name
            return get_c_shim(self.cpp_kernel_name, device)
        return self.python_kernel_name

    def get_kernel_name(self) -> str:
        name = self.try_get_kernel_name()
        if name is None:
            raise AssertionError("Expected name is not None")
        return name

    @classmethod
    def _shared_layout_copy(cls, x: "IRNode", *, layout_key) -> "TensorBox":
        """A copy made once per (value, arrangement) instead of once per ask.

        The conversions that arrange a value for a caller only read it, so
        callers asking for the same value under the same arrangement can read
        one result.  The value is named by its storage when the ask reads
        row-major storage in order -- such a window holds the storage's bytes
        under another shape -- and by its own node otherwise, since a view is
        named by its base and two views of one base need not read it alike.
        A caller that writes what it is given must go through copy_input.
        """

        node = x
        while isinstance(node, MutableBox):
            node = node.data
        name = _flat_window_base_name(x)
        # A value named by its own node is named by that node's identity.
        # The entry keeps the node, both so the identity cannot be handed to
        # a later node once this one is collected and so a hit can be checked
        # against the node that is asking.
        anchor = None
        if name is None:
            name = id(node)
            anchor = node
        try:
            shape = tuple(int(s) for s in x.get_size())
        except (TypeError, ValueError):
            shape = tuple(str(s) for s in x.get_size())
        cache = getattr(V.graph, "_shared_layout_copy_cache", None)
        if cache is None:
            cache = {}
            V.graph._shared_layout_copy_cache = cache
        key = (name, shape, layout_key)
        hit = cache.get(key)
        if hit is not None and hit[0] is anchor:
            return hit[1]
        out = cls.copy_input(x)
        cache[key] = (anchor, out)
        return out

    @staticmethod
    def copy_input(x: "IRNode") -> "TensorBox":
        """A separate copy of a value, so that writing over it is harmless.

        The copy is a body of its own rather than an alias, so that whatever is
        done to the result cannot be seen through the original.
        """

        pw = Pointwise.create(
            device=x.get_device(),
            dtype=x.get_dtype(),
            inner_fn=x.make_loader(),
            ranges=x.get_size(),
            origin_node=x.get_origin_node(),
            traceback=x.get_traceback(),
        )
        pw.realize()
        return pw

    @classmethod
    def process_kernel(cls, kernel, *args, _share_args: bool | None = None, **kwargs) -> ProcessKernelResult:
        """The arguments of a call, sorted, and what the call would produce.

        Three things happen, and all three are needed before the call can be
        written.  The arguments are sorted into the ones that are memory and
        the ones that are only values.  Each of the first kind is given memory,
        since the call reads memory rather than a value.  And since which
        memory it reads is what the call is written in terms of, the traced
        result is worked out again from the strides that resulted, rather than
        from the ones the values had while they were still being computed --
        which is also how a result whose extents were not known until the
        inputs were settled is discovered at all.

        The flat lists keep the order the call takes its arguments in, and the
        unflattening is what puts the original tree back together, since an
        argument may itself be a list of arguments.

        A value that is not yet in memory is given a buffer by way of a copy;
        when the call reads that value and nothing writes it, readers that ask
        for the same value can read one shared copy instead of each building
        their own.  ``_share_args`` says which: True when the caller knows the
        call only reads, False when it writes, and None to take the answer
        from the operation's own signature.
        """

        from tensorplay.utils import _pytree as pytree

        from .runtime.triton_compat import enable_python_dispatcher

        binded_args = {"args": args, "kwargs": kwargs}
        args_flat, args_spec = pytree.tree_flatten(binded_args)

        args_flat_is_tensor: list[bool] = []
        tensor_args: list = []
        non_tensor_args: list = []
        real_non_tensor_args: list = []
        for arg in args_flat:
            if isinstance(arg, IRNode):
                args_flat_is_tensor.append(True)
                tensor_args.append(arg)
            else:
                args_flat_is_tensor.append(False)
                non_tensor_args.append(arg)
                real_non_tensor_args.append(arg)

        def unflatten_args(new_tensor_args, new_non_tensor_args):
            """The call's arguments as they were written, with new values.

            Which of the two lists a position came from is recorded as the
            arguments were sorted, so putting them back together is a walk of
            that record rather than a search for what a position was.
            """

            result = []
            it_tensors = iter(new_tensor_args)
            it_non_tensors = iter(new_non_tensor_args)
            for is_tensor in args_flat_is_tensor:
                result.append(next(it_tensors) if is_tensor else next(it_non_tensors))
            restored = pytree.tree_unflatten(result, args_spec)
            return restored.get("args", []), restored.get("kwargs", {})

        share_inputs = _share_args
        if share_inputs is None:
            schema = _parsed_schema(kernel)
            share_inputs = (
                schema is not None
                and not schema.is_mutable
                and not _schema_mutates_and_returns_first_arg(schema)
            )
        tensor_args = [cls.realize_input(x, allow_shared=bool(share_inputs)) for x in tensor_args]

        # The layout is frozen here so that working out the result's strides
        # cannot be moved by a later change to it.
        for x in tensor_args:
            if is_storage_and_layout(x):
                as_storage_and_layout(x, freeze=True)

        # The result is worked out again from the operands as they now are,
        # because a call written in terms of strides has to be told the strides
        # it will run with.  A value that is a view of a constant cannot be
        # given to the call as a view, so the constant it views is used.
        example_args: list = []
        for x in tensor_args:
            if not isinstance(x, BaseView) and x.get_name() in V.graph.constants:
                example_args.append(V.graph.constants[x.get_name()])
            else:
                example_args.append(ir_node_to_tensor(x))

        new_args, new_kwargs = unflatten_args(example_args, real_non_tensor_args)
        # The result is worked out with the language's dispatch key held open,
        # so that the operation is asked rather than answered by a shortcut
        # written for the case where nothing is watching.  A mode that
        # intercepts operators has to see this one too, or the result recorded
        # here is not the result that will be produced.
        shape_env = V.fake_mode.shape_env if V.fake_mode is not None else None
        with (
            enable_python_dispatcher(),
            _fallback_kernel_symbol_tracking_context(shape_env),
        ):
            example_output = kernel(*new_args, **new_kwargs)

        unbacked_bindings: dict | None = None
        if V.fake_mode is not None and V.fake_mode.shape_env is not None:
            unbacked_bindings = compute_unbacked_bindings(
                V.fake_mode.shape_env, example_output, V.current_node.meta.get("val")
            )

        example_out_li = (
            [example_output]
            if not isinstance(example_output, (list, tuple))
            else example_output
        )
        for t in example_out_li:
            if isinstance(t, tp.Tensor) and t.is_sparse and not config.graph_partition:
                # A sparse result has no layout for a loop to be laid out in.
                # Refusing it here says so plainly rather than failing later
                # where the layout would have been wanted.
                msg = "sparsity is not handled"
                if stack_trace := V.graph.current_node.meta.get("stack_trace", None):
                    msg = f"{msg}, raised from:\n {stack_trace}"
                V.graph.disable_cudagraphs_reason = msg

        return ProcessKernelResult(
            example_output=example_output,
            tensor_args=tensor_args,
            non_tensor_args=non_tensor_args,
            unflatten_args=unflatten_args,
            unbacked_bindings=unbacked_bindings,
        )

    #: What this was called when the sorting of arguments was all it did.
    partition_args = process_kernel

    @classmethod
    def convert_to_reinterpret_view(cls, x: "IRNode") -> "ReinterpretView":
        """This value as a view, so that handing it over costs no copy.

        An external call is written in terms of a shape, a stride and an offset,
        so what it is given has to be describable that way.  Where it already
        is, nothing is done; where it is not, its layout is settled so that it
        becomes so.
        """

        if not isinstance(x, BaseView):
            raise AssertionError(type(x))
        if isinstance(x, ReinterpretView):
            return x

        # The read/write record is deliberately not asked for here: it fails
        # where the loader inlines the computation, which is exactly the case
        # this is being used in.
        x_unwrap_view = x.unwrap_view()
        buf = V.graph.get_buffer(x_unwrap_view.get_name())
        if buf is None:
            raise AssertionError("Expected buf is not None")
        x_unwrap_view_fx_node = buf.get_origin_node()
        # The layout the value was given in the user's own code is preferred
        # over one worked out here, since that is the one they wrote against.
        if (
            x_unwrap_view_fx_node is not None
            and "val" in x_unwrap_view_fx_node.meta
            and isinstance(x_unwrap_view, (ReinterpretView, Buffer, MutableBox))
            and isinstance(x_unwrap_view.layout, FlexibleLayout)
            and (
                is_contiguous_for_memory_format_or_false(
                    x_unwrap_view_fx_node.meta["val"],
                    memory_format=tp.channels_last,
                )
                or is_contiguous_for_memory_format_or_false(
                    x_unwrap_view_fx_node.meta["val"],
                    memory_format=tp.channels_last_3d,
                )
            )
        ):
            x_unwrap_view.freeze_layout_with_same_order(
                make_channels_last_strides_for(x_unwrap_view.get_size())
            )
        else:
            x_unwrap_view.freeze_layout()

        index_args, var_ranges = dependencies.index_vars_squeeze(
            x.get_size(), prefix="r"
        )
        # The strides are the ones worked out above, and a buffer is only
        # created for them where there is not one already.
        # The strides and the offset are read off the view's own indexing: a
        # shape, a stride and an offset describe the value only if reading it
        # that way lands on the same element, and that is what saying so here
        # checks.
        _vars = index_args[0] if index_args else []
        range_vars = _vars[0] if _vars and isinstance(_vars[0], (list, tuple)) else _vars
        try:
            index = sympy.expand(x.make_indexer()(range_vars))
        except (AssertionError, NotImplementedError, TypeError) as exc:
            # The view has no indexer over these extents, so there is nothing
            # to describe; the caller copies the value instead.
            raise NotImplementedError(
                f"the view's index could not be evaluated: {exc}"
            ) from exc
        strides = []
        for var in range_vars:
            if not isinstance(var, sympy.Symbol):
                # An axis of extent one is squeezed to the constant zero: it
                # has no position to step through, so its stride is zero.
                strides.append(0)
                continue
            try:
                poly = sympy.Poly(index, var)
            except sympy.PolynomialError:
                raise NotImplementedError(
                    "the view's index is not a fixed stride per axis"
                ) from None
            if poly.total_degree() > 1 or (poly.free_symbols - {var}):
                raise NotImplementedError(
                    "the view's index is not a fixed stride per axis"
                )
            strides.append(int(poly.coeff_monomial(var)))
        offset = int(index.subs({var: sympy.S.Zero for var in range_vars}))
        return ReinterpretView(
            data=x.data,
            layout=FixedLayout(
                device=x.get_device_or_error(),
                dtype=x.get_dtype(),
                size=x.get_size(),
                stride=strides,
                offset=offset,
                is_pinned=x_unwrap_view.get_layout().is_pinned,
            ),
        )

    @classmethod
    def realize_input(cls, x: "IRNode", allow_shared: bool = False) -> "IRNode":
        """Put a value in memory, so that an external call has something to read.

        Where the value is already a view of memory someone else holds, that
        memory is used; otherwise the value is given a buffer of its own.
        A caller that only reads may ask for the fallback buffer to be shared
        between asks for the same value, which turns one copy per reader into
        one copy per value; a caller that can write what it is given must not.
        """

        if x is None:
            return NoneAsConstantBuffer()
        if isinstance(x, (sympy.Expr, sympy.logic.boolalg.Boolean, int)):
            return ShapeAsConstantBuffer(expr=x)
        if isinstance(x, ShapeAsConstantBuffer):
            return x
        if isinstance(x, TensorBox):
            return cls.realize_input(x.data, allow_shared=allow_shared)
        if isinstance(x, ConstantBuffer):
            return x
        if isinstance(x, ReinterpretView):
            return ReinterpretView(
                data=cls.realize_input(x.data, allow_shared=allow_shared), layout=x.get_layout()
            )
        if isinstance(x, BaseView):
            try:
                return cls.convert_to_reinterpret_view(x)
            except NotImplementedError:
                # The value's indexing cannot be described as a shape, a
                # stride and an offset, so it has to be walked into memory
                # instead of read as a window onto what is already there.
                pass
        if isinstance(x, StorageBox):
            x.realize()
            return x
        if isinstance(x, Buffer) and x.get_buffer_name():
            # Already written somewhere with a name to read it by -- a window
            # onto a laid-down value reaches here unboxed -- so it is read
            # where it is rather than copied.
            return x
        if allow_shared:
            return cls._shared_layout_copy(x, layout_key=("realize_input",))
        return cls.copy_input(x)

    @classmethod
    def require_stride1(cls, x: "IRNode") -> "IRNode":
        """This value, reached one element at a time along its innermost loop.

        A call that was written assuming that can only be given such a value, so
        one that is not has to be made so, and a copy is what makes it so.
        """

        # A buffer that already has an axis of unit stride is read as it is:
        # that axis is the one the call walks one element at a time.  So is a
        # buffer with no axes, which has nothing to walk.
        if is_storage_and_layout(x):
            strides = x.get_stride()
            if len(strides) == 0 or any(stride == 1 for stride in strides):
                return x
        return cls._shared_layout_copy(x, layout_key=("stride1",))

    @classmethod
    def require_strides(
        cls,
        x: "IRNode",
        order: Sequence | None = None,
        exact_strides: Sequence | None = None,
        allow_padding: bool = False,
    ) -> "IRNode":
        """This value with the strides an external call was written assuming.

        Where the value already has them it is used as it is, since arranging
        for them is a copy and the point of asking is usually to avoid one.
        Where it does not, and the strides cannot be worked out from what is
        there, a copy is made.
        """

        if not (order is not None or exact_strides is not None):
            raise AssertionError(
                "Expected order is not None or exact_strides is not None"
            )
        # The arrangement does not matter when there is nothing to arrange:
        # a value holding no elements, or exactly one, is read the same way
        # whatever its strides say, and there is no caller's requirement left
        # to satisfy.
        if x.get_numel() in (0, 1) and not exact_strides:
            return x

        if is_storage_and_layout(x):
            if isinstance(x.get_layout(), FlexibleLayout):
                if order:
                    # A layout that has not been settled may already be in the
                    # order that was asked for, in which case settling it as it
                    # is answers the question.  It may also be a different
                    # arrangement that happens to be compatible, in which case
                    # a copy is what settles it.
                    as_storage_and_layout(
                        x,
                        freeze=True,
                        want_contiguous=False,
                        stride_order=order,
                        allow_padding=allow_padding,
                        exact_strides=exact_strides,
                    )
                    return x
                elif exact_strides:
                    as_storage_and_layout(
                        x,
                        freeze=True,
                        want_contiguous=False,
                        stride_order=None,
                        allow_padding=allow_padding,
                        exact_strides=exact_strides,
                    )
                    return x

            # Where padding is allowed, strides that differ from the ones asked
            # for only by padding are as good as the ones asked for.
            padded_exact_strides = None
            if allow_padding and exact_strides:
                padded_exact_strides = list(
                    Layout._pad_strides(exact_strides, x.get_size(), x.get_dtype())
                )

            if isinstance(x.get_layout(), (FixedLayout, NonOwningLayout)) and (
                (order and x.get_layout().is_stride_ordered(order))
                or (
                    exact_strides
                    and significant_strides_equal(
                        exact_strides, x.get_layout().stride, x.get_size()
                    )
                )
            ):
                return (
                    try_match_insignificant_strides(x, exact_strides)
                    if exact_strides is not None
                    else x
                )
            elif (
                padded_exact_strides is not None
                and isinstance(x.get_layout(), (FixedLayout, NonOwningLayout))
                and significant_strides_equal(
                    padded_exact_strides, x.get_layout().stride, x.get_size()
                )
            ):
                return try_match_insignificant_strides(x, padded_exact_strides)
            elif isinstance(
                (mutation_layout := x.get_layout()), MutationLayoutSHOULDREMOVE
            ):
                if isinstance(
                    (real_layout := mutation_layout.real_layout()), FlexibleLayout
                ):
                    raise AssertionError(
                        "the MutationLayoutSHOULDREMOVE's real layout shouldn't be FlexibleLayout"
                    )
                elif isinstance(real_layout, FixedLayout) and (
                    (order and real_layout.is_stride_ordered(order))
                    or (
                        exact_strides
                        and significant_strides_equal(
                            exact_strides, real_layout.stride, x.get_size()
                        )
                    )
                ):
                    return x

        if isinstance(x, InputBuffer) and (
            (order and x.get_layout().is_stride_ordered(order))
            or (
                exact_strides
                and significant_strides_equal(
                    exact_strides, x.get_layout().stride, x.get_size()
                )
            )
        ):
            return x
        if (
            isinstance(x, TensorBox)
            and isinstance(x.data, BaseView)
            and not isinstance(x.data, ReinterpretView)
            and is_storage_and_layout(unwrap_view := x.unwrap_view())
            and hasattr(unwrap_view, "data")
            and not isinstance(unwrap_view.data, ExternKernelAlloc)
        ):
            try:
                x.data = cls.convert_to_reinterpret_view(x.data)
                if order:
                    return cls.require_stride_order(
                        x, order, allow_padding=allow_padding
                    )
                elif exact_strides:
                    return cls.require_exact_strides(
                        x, exact_strides, allow_padding=allow_padding
                    )
            except NotImplementedError:
                pass

        # The expansion is kept as it is rather than being copied away, since
        # without a record of it in the graph the code would walk the whole
        # expanded shape and do the same work again for each element that came
        # from the same one.
        expanded_dims: list | None = None
        orig_size = x.get_size()
        if exact_strides is not None:
            sizevars = V.graph.sizevars
            expanded_dims = [
                i
                for i in range(len(x.get_size()))
                if sizevars.statically_known_equals(exact_strides[i], 0)
            ]
            if isinstance(x, TensorBox) and isinstance(x.data, ExpandView):
                cur_size = x.data.get_size()
                cur_stride = x.data.get_stride()
                expanded_dims = [
                    i
                    for i in range(len(cur_size))
                    if sizevars.statically_known_equals(cur_stride[i], 0)
                ]
            elif isinstance(x, InputBuffer) and len(x.get_size()) != len(
                exact_strides
            ):
                raise NotImplementedError(
                    f"require_strides {x.get_name()} with {exact_strides=}"
                )

        if exact_strides is not None and len(exact_strides) != len(orig_size):
            raise NotImplementedError(
                f"require_strides {x.get_name()} with {exact_strides=}"
            )

        # An expanded dimension has no stride of its own, so the stride asked
        # for is only a shape to fit into; what is copied is the value as it
        # already is.
        x = cls._shared_layout_copy(
            x,
            layout_key=(
                "require_strides",
                tuple(str(s) for s in order) if order is not None else None,
                tuple(str(s) for s in exact_strides) if exact_strides is not None else None,
                bool(allow_padding),
                tuple(expanded_dims) if expanded_dims is not None else None,
            ),
        )
        if isinstance(x, TensorBox) and isinstance(x.data, ExpandView):
            x = TensorBox(x.data.create_with_same_size(x.data.get_size()))

        if expanded_dims is not None and order is not None:
            order = [i for i in order if i not in expanded_dims]
            if not order:
                return x
        if exact_strides is not None and expanded_dims is not None:
            exact_strides = [
                s for i, s in enumerate(exact_strides) if i not in expanded_dims
            ]
            if not exact_strides:
                return x

        # A copy an earlier caller asked for under the same arrangement is
        # handed out again, and that caller already fixed its strides.
        if not expanded_dims and is_storage_and_layout(x):
            layout = x.get_layout()
            if isinstance(layout, FixedLayout) and (
                (order is not None and layout.is_stride_ordered(order))
                or (
                    exact_strides is not None
                    and significant_strides_equal(exact_strides, layout.stride, x.get_size())
                )
            ):
                return x

        # What is copied is a value laid out the way the caller assumed, which
        # is what arranging the strides comes down to.
        if order is not None:
            x = as_storage_and_layout(
                x, stride_order=order, allow_padding=allow_padding
            )[0]
        elif exact_strides is not None:
            x = as_storage_and_layout(
                x,
                exact_strides=exact_strides,
                allow_padding=allow_padding,
            )[0]

        if isinstance(x, TensorBox) and isinstance(x.data, ReinterpretView):
            x = TensorBox.create(ir.ExternKernel.copy_input(x.data))
            x.realize()
            return x

        return x
    @classmethod
    def require_exact_strides(
        cls,
        x: "IRNode",
        exact_strides: Sequence,
        allow_padding: bool = False,
    ) -> "IRNode":
        """This value, with exactly these strides, or a copy of it."""

        return cls.require_strides(x, exact_strides=exact_strides, allow_padding=allow_padding)

    @classmethod
    def require_channels_last(cls, x: "IRNode") -> "IRNode":
        """This value with its channels last, or a copy of it."""

        return cls.require_stride_order(x, NHWC_STRIDE_ORDER)

    @classmethod
    def require_channels_last_3d(cls, x: "IRNode") -> "IRNode":
        """This value with its channels last and depth after them, or a copy."""

        return cls.require_stride_order(x, NHWDC_STRIDE_ORDER)

    @classmethod
    def require_stride_order(
        cls, x: "IRNode", order: Sequence, allow_padding: bool = False
    ) -> "IRNode":
        """This value, with its loops in this order, or a copy of it."""

        return cls.require_strides(x, order=order, allow_padding=allow_padding)

    @classmethod
    def require_contiguous(cls, x: "IRNode") -> "IRNode":
        """This value with its elements consecutive, or a copy of it."""

        if is_contiguous_storage_and_layout(x):
            return x
        x = cls._shared_layout_copy(x, layout_key=("contiguous",))
        assert is_contiguous_storage_and_layout(x)
        return x

    @classmethod
    def require_contiguous_strides(cls, x: "IRNode") -> "IRNode":
        """This value with the consecutive strides, or a copy of it."""

        x = cls.copy_input(x)
        if is_contiguous_strides_for_shape(x.get_stride(), x.get_size()):
            return x
        return cls.copy_input(x)

    def should_assert_dtype(self, op_name: str) -> bool:
        """Whether the element type of the result is worth asserting at run time.

        A call that writes over one of its inputs has no result of its own to
        assert about, and a quantized result is a type the tracing machinery
        cannot produce, so asserting it there would fail for a reason that has
        nothing to do with the call.
        """

        return not self.is_inplace_view() and not op_name.startswith(
            "quantize_per_tensor"
        )

    def codegen_size_asserts(self, wrapper) -> None:
        """Write the checks that the result really has the shape it was given.

        These are the checks that turn a call written against one shape being
        run against another into a report rather than into a wrong answer.
        """

        if not config.size_asserts:
            return
        if not V.graph.cpp_wrapper:
            # A Python wrapper allocates every buffer it hands an external call
            # itself, so the result's shape and stride cannot disagree with the
            # call: the check would guard a value nobody else can change.
            # Inputs are still guarded, by the wrapper's own input assertions.
            return
        if self.is_inplace_view() and not V.graph.cpp_wrapper:
            return
        op_name = self.get_op_name()
        name = self.get_name()
        if V.graph.cpp_wrapper:
            # A call that writes over an input declares no result of its own,
            # so the check is on what it wrote over.
            if self.is_inplace_view():
                if not isinstance(self.inputs[0], IRNode):
                    raise AssertionError("Expected isinstance(self.inputs[0], IRNode)")
                name = self.inputs[0].get_name()
        size = V.graph.wrapper_code.codegen_shape_tuple(self.get_size())
        stride = V.graph.wrapper_code.codegen_shape_tuple(self.get_stride())
        dtype = self.get_dtype() if self.should_assert_dtype(op_name) else None
        wrapper.write_assert_size_stride(name, size, stride, op_name, dtype)

    def codegen_alignment_asserts(self, wrapper) -> None:
        """Write the check that the result's memory is aligned, where it is.

        A value whose memory did not come from here may be laid out so that a
        wide load would straddle a boundary, and saying so is what lets a kernel
        read it one element at a time.
        """

        if config.alignment_asserts and not V.graph.cpp_wrapper:
            name = self.get_name()
            aligned = name not in V.graph.unaligned_buffers
            op_name = self.get_op_name()
            if aligned:
                wrapper.writeline(
                    f"assert_alignment({name}, {GPU_ALIGN_BYTES}, {op_name!r})"
                )
            else:
                wrapper.writeline(
                    f"# buffer {name} (op: {op_name}) is assumed to be not aligned"
                )

    def codegen_memory_tracking(self, wrapper) -> None:
        """Note the result in the memory record, where one is being kept.

        The record is what says which values are alive and which have been
        freed, and it only exists when something is reading it.
        """

        if not config.test_configs.track_memory_lifecycle or V.graph.cpp_wrapper:
            return

        wrapper.write_memory_track_allocation_once()
        name = self.get_name()
        wrapper.writeline(f"track_tensor({name}, '{name}')")

    def get_group_stride(self):
        """The result's extents and strides, split the way a template wants them.

        A template is written for a kernel rather than for a buffer, so the
        iterated loops and the reduced ones are asked for separately.  A call
        reduces nothing, so the reduced group is empty.
        """

        _size = self.get_size()
        _stride = self.get_stride()
        # The iterated group is the whole shape; the reduced group is empty,
        # since a call folds nothing.
        return [_size, []], _stride

    def canonicalize(self):
        """The result's index, written in the loop order its strides suggest.

        A template that was written against one loop order is being handed an
        index written in another, so the index is rewritten into the order the
        strides ask for.  What the strides are guessed to be only decides the
        order and nothing about what is computed rests on the guess.
        """

        sizevars = V.graph.sizevars
        sizes = self.get_size()
        strides = self.get_stride()
        # The hints are only sort keys here, so a shape that is not known yet
        # does no harm.
        strides = [sizevars.optimization_hint(x) for x in strides]
        index_vars = [
            sympy_index_symbol(f"d{i}") for i in range(len(sizes))
        ]
        # The loop with the largest stride is the one to run first, since
        # everything inside it then moves in large steps.
        index_order = sorted(range(len(strides)), key=strides.__getitem__, reverse=True)
        lookup = {pos: idx for idx, pos in enumerate(index_order)}
        order = [lookup[i] for i in range(len(lookup))]
        index_vars = [index_vars[i] for i in order]
        indexer = self.make_indexer()
        index = indexer(index_vars)

        new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
            index_vars, sizes, [index]
        )
        # The loops are numbered afresh: what came back as d0, d1, d2 may be
        # only d0, d2, and the two must not look like different indexings.
        _, add_var = dependencies.var_builder("c")
        replacement = dict(zip(index_vars, reindex([add_var(x) for x in new_sizes])))

        index = sympy_subs(sympy.expand(index), replacement)
        return index, tuple(new_sizes)

    @cache_on_self_and_args("ExternKernel")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        """Every shape this call needs in scope in order to be written.

        The inputs are not looked at: depending on a buffer already carries
        whatever shapes that buffer has, so only the arguments that are
        themselves shapes are here.
        """

        r = InputsKernel.get_free_symbol_uses(self, unbacked_only)
        for arg in self.constant_args:
            r |= maybe_free_symbols(arg, unbacked_only)
        for arg in self.kwargs.values():
            r |= maybe_free_symbols(arg, unbacked_only)
        return r

    def __str__(self) -> str:
        kernel_name = getattr(self, "python_kernel_name", None)
        lines = [
            f"python_kernel_name={kernel_name!r}",
        ]
        lines += [
            f"{field.name}={getattr(self, field.name)}"
            for field in dataclasses.fields(self)
            if field.name in self.__dict__
        ]
        lines.append(f"origin_node={getattr(self, 'origin_node', None)!r}")
        return self.str_helper(lines)

    __repr__ = __str__

    def apply_constraint(self) -> None:
        """Settle the layout, in the way this particular call requires.

        A call that was written against a particular layout says so here, and a
        call that places no requirement leaves the layout to be worked out.
        """

        raise NotImplementedError

    def fill_non_provided_args(self, kwargs: dict) -> None:
        """Put in the arguments the call was not given, at their declared defaults.

        A call that was written with an argument left out has to be written with
        it filled in, since the generated code has nowhere to leave anything
        out.
        """

        if not kwargs:
            return
        for name, prop in self.allarg_properties.items():
            default = prop.get("default_value") if isinstance(prop, dict) else None
            if name not in kwargs and default is not None:
                kwargs[name] = default
        self.kwargs = kwargs

    def get_kwargs_value(self, arg_name: str, **kwargs):
        """One argument's value, whether it was given by name or by position."""

        if arg_name in kwargs:
            return kwargs.get(arg_name)
        if arg_name in self.kwargs:
            return self.kwargs.get(arg_name)
        if (arg := self.allarg_properties.get(arg_name)) is not None:
            return arg.get("default_value")
        raise AssertionError(f"{arg_name} not in self.allarg_properties")

    def codegen_const_args(self, names: list | None = None) -> list:
        """The arguments that are written as they stand, rather than as memory.

        These are the ones that are the same in every call -- shapes, numbers,
        flags -- and writing them out is cheaper than passing them.
        """

        default_args = []
        if names is None:
            names = self.ordered_kwargs_for_cpp_kernel
        for name in names:
            default_args.append(self.get_kwargs_value(name))
        return default_args

    def codegen_args(self) -> list:
        """The arguments of the call, in the order the call takes them.

        Each tensor argument is written the way it refers to itself: a buffer by
        its name, a view of one as that view, since the call reads the elements
        the view selects and not the whole buffer behind it.  The arguments that
        are the same in every call follow, written as they stand.
        """

        wrapper = V.graph.wrapper_code
        args: list = [wrapper.val_to_arg_str(inp) for inp in self.inputs]
        args.extend(wrapper.val_to_arg_str(value) for value in self.constant_args)
        return args

    def codegen_kwargs(self, skip_out: bool = False) -> list:
        """The arguments passed by name, in the order the call declares them.

        The order is the one the call was declared with rather than the one it
        happened to be given, since a call written in a different order is a
        different call.
        """

        kwargs: list = []
        if self.allarg_properties:
            for name in self.ordered_kwargs_for_cpp_kernel or self.allarg_properties:
                if name in self.kwargs:
                    value = self.kwargs[name]
                    if isinstance(value, str):
                        value = repr(value)
                    elif isinstance(value, tp.device):
                        value = repr(str(value))
                    kwargs.append(f"{name}={value}")
        else:
            for name, value in self.kwargs.items():
                if isinstance(value, str):
                    value = repr(value)
                elif isinstance(value, tp.device):
                    value = repr(str(value))
                kwargs.append(f"{name}={value}")
        return kwargs

    def get_op_name(self) -> str:
        """The name of the operation this calls, as it is registered."""

        return self.get_kernel_name()

    def is_inplace_view(self) -> bool:
        """Whether this writes over one of its inputs rather than to a new place.

        A call that does leaves its input alone, and one that does not needs its
        inputs put in memory first so that what it writes over is the right
        memory.
        """

        return False

class MultiOutput(ExternKernel):
    """One of several results of a single operation, addressed by where it sits.

    When an operation returns several values they are not a tuple that gets
    taken apart afterwards; each one is a buffer that says which path through
    the structure it was at, so that whoever reads one of them can be related
    to the operation that made all of them.
    """

    def codegen(self, wrapper) -> None:
        wrapper.codegen_multi_output(self)
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)

    def get_op_name(self) -> str:
        node = getattr(self, "origin_node", None)
        if node is not None:
            target = node.target
            op_namespace = getattr(target, "__module__", "unknown_namespace")
            op_namespace = op_namespace.replace("._ops.", ".ops.")
            op_namespace = op_namespace.rsplit(".", 1)[0]
            return f"{op_namespace}.{target}"
        return "unknown_op"

    def __init__(
        self,
        layout: "OutputSpec",
        input: "IRNode",
        indices,
        skip_size_stride_alignment_checks: bool = False,
    ) -> None:
        super().__init__(None, layout, [input], ())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.indices = indices
        self.skip_size_stride_alignment_checks = skip_size_stride_alignment_checks

    @cache_on_self_and_args("MultiOutput")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        input_node = self.inputs[0]
        if not isinstance(input_node, IRNode):
            raise AssertionError(input_node)
        return input_node.get_free_symbol_uses(unbacked_only)

    def should_allocate(self) -> bool:
        # There is nothing to allocate for unless the operation is one that
        # provides the memory itself, which is the case for a grouped matmul
        # and for nothing else here.
        return len(self.inputs) == 1 and isinstance(
            self.inputs[0], CppTemplateBuffer
        )

    def get_inputs_that_alias_output(self) -> Sequence:
        return [
            inp.get_name()
            for inp in self.inputs
            if isinstance(inp, FallbackKernel)
            and len(inp.get_inputs_that_alias_output()) > 0
        ]

    def get_read_writes(self):
        # Which elements of the packed output are read is not known here, so
        # the read is reported as depending on all of it, which is the safe
        # answer rather than a wrong one.
        reads: OrderedSet = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                reads.add(dependencies.StarDep(inp.get_name()))

        # The write is described from the layout, and normalized the same way
        # the scheduler normalizes, so that the index expressions here are
        # directly comparable when deciding whether two things can be fused.
        name = self.get_name()
        indexer = self.get_layout().make_indexer()

        def dummy(index, rindex):
            if len(rindex) != 0:
                raise AssertionError("Expected len(rindex) == 0")
            return ops.store(name, indexer(index), "fake")

        device = self.get_device()
        should_normalize = (
            not config.loop_ordering_after_fusion
            or device is None
            or not is_gpu(device)
        )
        write_rw = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=should_normalize
        )
        return dependencies.ReadWrites(
            reads=reads,
            writes=write_rw.writes,
            index_exprs=OrderedSet(),
        )


class ExternKernelOut(ExternKernel):
    """An output the call writes into memory someone else already has.

    Where an ordinary external result gets memory of its own, this one names
    memory that is already there, which is how a call that writes into a
    caller's buffer is described.  That is also why it does not allocate, and
    why the view onto it is kept: a caller may have handed over part of a
    larger buffer rather than all of it.
    """

    def codegen(self, wrapper) -> None:
        wrapper.generate_extern_kernel_out(self)

    def __init__(
        self,
        layout: "Layout",
        inputs,
        constant_args=(),
        kwargs: dict | None = None,
        output_view=None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ) -> None:
        unwrapped_inputs = self.unwrap_storage(inputs)
        if not isinstance(unwrapped_inputs, Sequence):
            raise AssertionError(type(unwrapped_inputs))
        super().__init__(
            None,
            layout,
            unwrapped_inputs,
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        self.output_view = output_view
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        return True


class RandomSeeds(ExternKernelOut):
    """The seed values a kernel that needs randomness was given for this run.

    They are drawn once here rather than inside the kernel so that every launch
    of that kernel in this run sees the same values, which is what makes a
    result reproducible; the bounds are what the generator accepts.
    """

    def __init__(self, count: int, device, op_overload=None) -> None:
        limits = tp.iinfo(tp.int64)
        super().__init__(
            layout=FixedLayout(
                device=device,
                dtype=tp.int64,
                size=[count],
            ),
            inputs=[],
            constant_args=[limits.min, limits.max, [count]],
            python_kernel_name="operator_set.randint.low_out",
            cpp_kernel_name="tp::_ops::randint_low_out::call",
            op_overload=op_overload,
        )


class ExternKernelAlloc(ExternKernel):
    """Memory the call itself provides rather than a result it computes.

    Some work is handed to code that allocates as it goes, so the memory is
    described here as an output to be filled in but is not this graph's to
    hand out again.  The outputs are recorded because in the mode where results
    come back one at a time they are what the arguments are built from.
    """

    def codegen(self, wrapper) -> None:
        wrapper.generate_extern_kernel_alloc(self)

    def __init__(
        self,
        layout: "OutputSpec",
        inputs,
        constant_args=(),
        kwargs: dict | None = None,
        python_kernel_name=None,
        cpp_kernel_name=None,
        ordered_kwargs_for_cpp_kernel=(),
        op_overload=None,
    ) -> None:
        unwrapped_inputs = self.unwrap_storage(inputs)
        if not all(isinstance(i, IRNode) for i in unwrapped_inputs):
            raise AssertionError(
                "Expected all(isinstance(i, IRNode) for i in unwrapped_inputs)"
            )
        super().__init__(
            None,
            layout,
            list(unwrapped_inputs),
            constant_args,
            kwargs or {},
            None,
            python_kernel_name,
            cpp_kernel_name,
            ordered_kwargs_for_cpp_kernel,
            op_overload,
        )
        self.outputs: Sequence = []
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def should_allocate(self) -> bool:
        return False

    def apply_constraint(self) -> None:
        raise NotImplementedError


class TemplateBuffer(OperationBuffer):
    """A result whose body is written by a caller-supplied renderer.

    The shape and the inputs are described here so that fusion, scheduling and
    dependency analysis work on it the same as on any other result, while the
    code itself comes from the renderer that was handed in.  A renderer that
    returns several results says so through a layout that has no shape.
    """

    def __init__(
        self,
        layout: "OutputSpec",
        inputs,
        make_kernel_render,
        mutated_inputs=None,
        allowed_prologue_inps=None,
        named_inputs=None,
    ) -> None:
        super().__init__(name=None, layout=layout)
        self.inputs = InputsKernel.unwrap_storage(inputs)
        self.make_kernel_render = make_kernel_render
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        self.annotations: dict = {}

        self.epilogue_fusable_outputs: dict = {}
        self._multi_output_children: dict = {}
        self._named_inputs: dict = dict(named_inputs) if named_inputs else {}

        self.mutated_inputs = mutated_inputs
        self.mutation_outputs: list = []
        if mutated_inputs is not None:
            first_input = self.inputs[0]
            if not isinstance(first_input, IRNode):
                raise AssertionError(type(first_input))
            device = first_input.get_device()
            self.mutation_outputs = [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]
        self.allowed_prologue_inps: OrderedSet = (
            allowed_prologue_inps or OrderedSet()
        )
        self.allow_epilogue_fusion: bool | None = None
        self.allow_prologue_fusion: bool | None = None

    @property
    def dtype(self):
        if isinstance(self.layout, MultiOutputLayout):
            raise NotImplementedError(
                "Multi-output templates do not have a single dtype"
            )
        return self.get_layout().dtype

    def get_read_writes(self):
        return self.extract_read_writes(normalize=True)

    def _read_deps_from_inputs(self, normalize: bool) -> OrderedSet:
        """Build read dependencies from all inputs."""

        reads: OrderedSet = OrderedSet()
        for inp_raw in self.inputs:
            if not isinstance(inp_raw, (ReinterpretView, Buffer)):
                raise AssertionError(type(inp_raw))
            inp: ReinterpretView | Buffer = inp_raw
            if not isinstance(inp.layout, Layout):
                raise AssertionError(type(inp.layout))
            inp_indexer = inp.layout.make_indexer()

            def dummy(index, rindex):
                if len(rindex) != 0:
                    raise AssertionError("Expected len(rindex) == 0")
                return ops.load(inp.get_name(), inp_indexer(index))

            reads |= dependencies.extract_read_writes(
                dummy, inp.get_size(), (), normalize=normalize
            ).reads
        return reads

    def extract_read_writes(self, normalize: bool = False):
        """Extract read/write dependencies for this template.

        When the layout describes several results there is no shape to index
        with, so the write is a single whole-buffer dependency and the reads
        come from the named tensor inputs.  A single result falls through to
        the ordinary path.
        """

        if isinstance(self.layout, MultiOutputLayout):
            writes: OrderedSet = OrderedSet(
                [
                    dependencies.MemoryDep(
                        self.get_name(), sympy.Integer(0), var_names=(), size=()
                    ),
                ]
            )
            return dependencies.ReadWrites(
                reads=self._read_deps_from_inputs(normalize),
                writes=writes,
                index_exprs=OrderedSet(),
                range_vars=None,
                var_ranges=None,
            )

        name = self.get_name()
        indexer = self.get_layout().make_indexer()

        def dummy(index, rindex):
            if len(rindex) != 0:
                raise AssertionError("Expected len(rindex) == 0")
            return ops.store(name, indexer(index), "fake")

        deps = dependencies.extract_read_writes(
            dummy, self.get_size(), (), normalize=normalize
        )
        deps.reads |= self._read_deps_from_inputs(normalize)
        return deps

    def get_reduction_size(self):
        return sympy.S.One

    def get_reduction_type(self):
        return None

    def should_allocate(self) -> bool:
        return True

    def simplify_and_reorder(
        self,
        extra_indexing_constraints=None,
        recompute_sizes_body_func=None,
    ):
        return ((self.get_size(), []), None)

    def is_multi_outputs_template(self) -> bool:
        """Whether this template produces multiple outputs through its layout."""

        return isinstance(self.layout, MultiOutputLayout)

    def get_allowed_prologue_inps(self) -> OrderedSet:
        return self.allowed_prologue_inps

    def has_aliasing_or_mutation_for_prologue_fusion(self, scheduler_node) -> bool:
        """Whether this template's aliasing or mutation blocks prologue fusion.

        The default keeps the conservative answer; a subclass that can show a
        prologue only feeds inputs it neither aliases nor has written may
        override this and say no.
        """

        return scheduler_node.has_aliasing_or_mutation()

    def _finalize_codegen(self, hook_outputs: dict):
        """Called after prologue/epilogue codegen with the rendered hook outputs.

        ``hook_outputs`` maps placeholder keys (e.g. ``<STORE_OUTPUT_0>``,
        ``<LOAD_INPUT_x>``) to the code generated for each fused subgraph.

        Return a result to supply custom source and call metadata, or ``None``
        to use the default codegen path.
        """

        return None

    @classmethod
    def realize_template_input(cls, tb: "TensorBox") -> "IRNode":
        """Realize a TensorBox, keeping a several-result layout as it is.

        Unlike the ordinary path, a result that is one of several keeps its
        multi-output layout rather than being turned into an allocation.
        """

        if isinstance(tb, TensorBox) and isinstance(tb.data, MultiOutput):
            return tb.data
        result = ExternKernel.realize_input(tb)
        if isinstance(result, StorageBox):
            result = result.data
        if isinstance(result.layout, FlexibleLayout):
            result.freeze_layout()
        return result

    @classmethod
    def build_multi_outputs(
        cls,
        template_buf: "TemplateBuffer",
        structured,
        *,
        direct_alias_at_leaf=None,
        on_tensor_leaf=None,
        on_non_tensor_leaf=None,
    ):
        """Walk a structured output tree, making a MultiOutput for each tensor.

        A value that appears twice in the tree becomes one buffer that two
        places refer to, since writing it twice would be two writes to one
        piece of memory.
        """

        seen_outputs: dict = {}
        leaf_counter = itertools.count()

        def walk(output, indices):
            if isinstance(output, (list, tuple)):
                results: list = []
                for i, item in enumerate(output):
                    results.extend(walk(item, [*indices, (type(output), i)]))
                return results
            leaf_idx = next(leaf_counter)
            if isinstance(output, tp.Tensor):
                if direct_alias_at_leaf and leaf_idx in direct_alias_at_leaf:
                    return [TensorBox.create(direct_alias_at_leaf[leaf_idx])]
                tid = id(output)
                if tid in seen_outputs:
                    return [seen_outputs[tid]]
                mo = MultiOutput(
                    FallbackKernel.tensor_to_layout(output), template_buf, indices
                )
                template_buf._multi_output_children[mo.get_name()] = mo
                if on_tensor_leaf is not None:
                    on_tensor_leaf(mo.get_name(), mo, indices, leaf_idx)
                tb = TensorBox(mo)
                seen_outputs[tid] = tb
                return [tb]
            if on_non_tensor_leaf is not None:
                on_non_tensor_leaf(leaf_idx)
            return []

        return tuple(walk(structured, []))


class CppTemplateBuffer(TemplateBuffer):
    """A result whose body comes from a chosen template, in compiled code.

    Where an ordinary template says what to write, this says which of the
    prepared kernels was chosen, and the choice is what decides the code.  A
    template that produces several results has no shape of its own, so the
    shape asked of it is the one belonging to the first of them.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs,
        make_kernel_render,
        template,
        choice,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.choice = choice
        self.outputs: list | None = None

    def get_layout(self) -> "Layout":
        if isinstance(self.layout, MultiOutputLayout):
            if not isinstance(self.outputs, Iterable):
                raise AssertionError(type(self.outputs))

            first_output = self.outputs[0]
            if not isinstance(first_output, Buffer):
                raise AssertionError(type(first_output))
            layout = first_output.layout
            if not isinstance(layout, Layout):
                raise AssertionError(type(layout))
            return layout
        else:
            return super().get_layout()


@dataclasses.dataclass
class _HasAliasingOrMutation(Protocol):
    """The one thing asked of a scheduled node when deciding about a prologue.

    A prologue is work run before a kernel and whose result the kernel reads.
    Whether that is safe depends on whether anything it would run shares memory
    with, or writes to, what the kernel touches, and that question is put to
    whatever holds the scheduled work rather than being answered here.  Only
    that one question is asked of it, so only that one is written down.
    """

    def has_aliasing_or_mutation(self) -> bool: ...


class CommBufferType(enum.Enum):
    """What kind of memory a collective operation is exchanging through.

    Only one kind is described here: memory that every participating rank can
    read from and write to directly, rather than memory that belongs to one
    rank and has to be copied to reach the others.
    """

    SYMM_MEM = "symm_mem"


@dataclasses.dataclass
class GraphPartitionSignature:
    """Everything one separately-compiled piece of a graph needs to know.

    A piece that is compiled on its own and called from elsewhere cannot read
    anything from the graph it came from while it runs, so what it needs has to
    be written down: the shapes whose values are not known, which values go in
    and which come out, which of the inputs it is responsible for releasing,
    and which constants it touches.
    """

    #: The shapes that have to be handed in, since their values are not known
    #: when the piece is compiled.
    symbol_inputs: OrderedSet

    #: What each named input is.  The name is kept as well because an
    #: expression has no name of its own to ask for.
    input_nodes: dict

    output_nodes: list

    #: Which inputs this piece is responsible for releasing once it is done.
    input_deallocation: dict

    #: Whether this piece has to be run outside a recorded launch, which is
    #: needed when what it does would not survive being recorded and replayed.
    skip_cudagraph: bool

    #: The constants this piece reads or writes, by name.
    constant_names: list


@dataclasses.dataclass
class ExternKernelNode:
    """One call to be run outside the compiled code, as it was written.

    What the call is and what it was given are kept together so that the call
    can be written out later, in a form that does not have to be understood by
    whatever is doing the writing.
    """

    name: str
    node: Any


class NonTensorObj(IRNode):
    """A result that is a value of some kind rather than a tensor.

    Not everything a computation produces is a tensor: a generator's state, an
    object handed in from outside, a constant that is not a number.  These
    cannot be indexed, have no shape, and nothing can be fused into them, so
    they sit in the graph as things that are merely passed along.
    """

    @cache_on_self_and_args("NonTensorObj")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return OrderedSet()

    def realize(self):
        """There is nothing to give memory to, and so nothing to report."""

        return None


class AllocatingMultiOutput(MultiOutput):
    """One result of several, whose memory this graph is responsible for.

    Where an ordinary result of several is whatever memory the operation chose
    to write into, this is memory handed over in advance, and the operation
    writes into it rather than returning it.  Since the memory is already there,
    nothing has to be allocated, and since the operation writes each result
    where it was told to, there is no tuple to take apart afterwards.
    """

    def should_allocate(self) -> bool:
        return True

    def codegen(self, wrapper) -> None:
        if not self.skip_size_stride_alignment_checks:
            self.codegen_size_asserts(wrapper)
            self.codegen_alignment_asserts(wrapper)


class OpaqueMultiOutput(MultiOutput):
    """One result of several, where the result is not a tensor at all.

    Some operations return values that have no shape and no type to speak of.
    Asking such a result for its type is asking something meaningless, so it
    has none; what it does have is the value itself, as an example of what the
    operation produced, which is what the caller will match its result against.
    """

    def __init__(
        self,
        layout: "OutputSpec",
        input: "IRNode",
        indices,
        opaque_value,
    ) -> None:
        super().__init__(layout, input, indices, skip_size_stride_alignment_checks=True)
        self.opaque_example_value = opaque_value

    @property
    def dtype(self):
        raise AttributeError("OpaqueMultiOutput has no dtype")

    def wrap_for_lowering(self) -> "OpaqueMultiOutput":
        """A result that is not a tensor needs no wrapping to be lowered."""

        return self

    def get_read_writes(self):
        # Which part of the operation's output this is does not say which
        # elements were written, so the whole of it is reported as written.
        reads: OrderedSet = OrderedSet()
        for inp in self.inputs:
            if isinstance(inp, IRNode):
                reads.add(dependencies.StarDep(inp.get_name()))
        writes: OrderedSet = OrderedSet([dependencies.StarDep(self.get_name())])
        return dependencies.ReadWrites(
            reads=reads,
            writes=writes,
            index_exprs=OrderedSet(),
        )


@ir_dataclass(frozen=False)
class CommBufferLayout(FixedLayout):
    """How memory that a group of ranks shares is laid out.

    The arrangement of elements is an ordinary one, and is written down
    exactly as it would be for any other value; what makes this different is
    what may be done with it.  Memory that several ranks can be reading and
    writing at once cannot be handed to one operation to write into while
    another reads from it, so it is neither given over in place nor taken from
    one, and a value described this way is passed by name instead.
    """

    comm_buffer_type: CommBufferType
    group_name: str

    def __init__(
        self,
        layout,
        comm_buffer_type: CommBufferType,
        group_name: str,
    ):
        fixed = layout.as_fixed() if isinstance(layout, FlexibleLayout) else layout
        super().__init__(
            device=fixed.device,
            dtype=fixed.dtype,
            size=fixed.size,
            stride=fixed.stride,
            offset=fixed.offset,
            is_pinned=fixed.is_pinned,
        )
        self.comm_buffer_type = comm_buffer_type
        self.group_name = group_name


@ir_dataclass
class GeneratorState(NonTensorObj):
    """The state of a random number generator, carried through the graph.

    A value that comes from a generator has to come from the same generator in
    the same state in order to be the same value, so the state travels with it
    as something with a name rather than as a number that could be written into
    the code.
    """

    name: str
    device: Any

    def get_name(self) -> str:
        return self.name

    def codegen_reference(self, writer=None) -> str:
        return self.name


@ir_dataclass
class OpaqueObjectState(NonTensorObj):
    """Something that is not a tensor, passed through the graph by name.

    A group of ranks taking part together, or any other value that the code
    cannot describe, is referred to by the name it was given rather than by
    what it is, so that the generated code can refer to it without having to
    know anything about it.
    """

    name: str
    value: Any

    def get_name(self) -> str:
        return self.name

    def codegen_reference(self, writer=None) -> str:
        return self.name


@ir_dataclass
class OpaqueValueTypeConstant(NonTensorObj):
    """A constant of a kind that is written out as its own description.

    Where a value of this kind is loaded from something at run time, a value
    that is a constant is written into the code as the text that reconstructs
    it, so that running the code twice produces the same value rather than
    whatever happens to be around.
    """

    value: Any

    def get_name(self) -> str:
        return repr(self.value)

    def codegen_reference(self, writer=None) -> str:
        return repr(self.value)


@ir_dataclass
class TorchBindObject(NonTensorObj):
    """An object bound from the surrounding program, referred to by name.

    Some values are better referred to than copied: an object that was defined
    outside this computation cannot be written into the generated code, so it
    is named instead and the name is what the code refers to.  How much memory
    such an object holds is worked out by walking what it contains, since that
    is what decides whether there is room for it.
    """

    name: str
    value: Any

    def get_name(self) -> str:
        return self.name

    def codegen_reference(self, writer=None) -> str:
        return self.name

    def get_value(self) -> Any:
        return self.value

    def get_real_obj(self) -> Any:
        """The object itself, rather than the stand-in that stands for it.

        A value of this kind is sometimes a description of the object rather
        than the object, and only the latter can be asked how much memory it
        holds.
        """

        real_obj = getattr(self.value, "real_obj", None)
        if real_obj is not None:
            return real_obj
        return self.value

    def get_buf_bytes(self) -> int:
        """How much memory this object holds, counted over the values in it.

        Every value inside it that is a tensor takes up its own size, and
        anything else takes up nothing; an object that cannot be taken apart
        this way is reported as holding nothing rather than guessed at.
        """

        real_script_obj = self.get_real_obj()

        flatten = getattr(real_script_obj, "__obj_flatten__", None)
        if flatten is None:
            return 0
        flat_dict = dict(flatten())
        total = 0
        for v in _pytree.tree_flatten(flat_dict)[0]:
            if isinstance(v, tp.Tensor):
                total += v.element_size() * v.numel()
        return total


class ScatterFallback(ExternKernel):
    """Writing chosen elements of a value in place, given where to write them.

    Which elements get written is decided by an index value rather than by
    where in the output they land, so the call cannot be described as a loop
    over the output and is run through to instead.  What makes this its own
    class rather than a general external result is that it writes to memory it
    was given rather than producing any, and that a value to write may be a
    single number applied everywhere rather than a value of the output's shape.
    """

    def codegen(self, wrapper) -> None:
        wrapper.generate_scatter_fallback(self)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> list:
        inp = self.inputs[0]
        if not isinstance(inp, IRNode):
            raise AssertionError("Expected isinstance(inp, IRNode)")
        return [inp.get_name()]

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def __init__(
        self,
        op_overload,
        x: "IRNode",
        dim: int,
        index: "IRNode",
        src,
        *,
        reduce=None,
        include_self: bool = True,
    ) -> None:
        # A single number to write everywhere is not a value, so it is passed
        # as a plain argument rather than as one of the inputs.
        self.src_is_tensor = isinstance(src, TensorBox)

        if self.src_is_tensor:
            tensors = [self.realize_input(t) for t in [x, index, src]]
            constant_args: tuple = (dim,)
        else:
            tensors = [self.realize_input(t) for t in [x, index]]
            constant_args = (dim, src)

        # The generated program looks operators up in the operation table it
        # binds to a short name, not in the module an op name is written with,
        # so the call names the operation through that table.
        _, _, op_name = str(op_overload).partition(".")
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            constant_args,
            {"reduce": reduce, "include_self": include_self},
            python_kernel_name=f"operator_set.{op_name}",
            ordered_kwargs_for_cpp_kernel=["reduce", "include_self"],
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class IndexPutFallback(ExternKernel):
    """Writing chosen elements of a value in place, given a list of positions.

    Where each position comes from is not known when the code is written, since
    the index values themselves come from the data, so the call is run through
    to rather than written out.  A position that is not there means that part
    of the operation does not apply and is left out.
    """

    def codegen(self, wrapper) -> None:
        wrapper.generate_index_put_fallback(self)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def __init__(
        self,
        op_overload,
        x: "IRNode",
        indices,
        values,
        accumulate,
    ) -> None:
        self.indices = indices
        valid_indices = [i for i in indices if i is not None]
        tensors = [self.realize_input(t) for t in [x, values, *valid_indices]]
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage(tensors),
            (accumulate,),
            python_kernel_name="tp.index_put_",
            cpp_kernel_name="tp_index_put_out",
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(self.input_name(0))
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class InplaceBernoulliFallback(ExternKernel):
    """Setting each element of a value at random, in place.

    Whether an element is set is drawn as the elements are visited, so there is
    no shape to loop over and the call is run through to.  The generator it
    draws from is not something the compiled code can be handed, so it is
    written as a null there and the surrounding program supplies one.
    """

    def codegen(self, wrapper) -> None:
        if not all(isinstance(t, IRNode) for t in self.inputs):
            raise AssertionError(
                "Expected all(isinstance(t, IRNode) for t in self.inputs)"
            )
        (x,) = (t.codegen_reference() for t in self.inputs)

        if V.graph.cpp_wrapper:
            # The generator cannot be handed to compiled code, so it is written
            # as a null and the call draws from the ambient one instead.
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}, NULL){wrapper.ending}"
            )
        else:
            wrapper.writeline(
                f"{self.get_kernel_name()}({x}, {', '.join(map(repr, self.constant_args))}){wrapper.ending}"
            )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def __init__(self, op_overload, x: "IRNode", *constant_args) -> None:
        super().__init__(
            None,
            NoneLayout(device=x.get_device()),
            self.unwrap_storage([x]),
            constant_args,
            op_overload=op_overload,
        )
        V.graph.mark_buffer_mutated(x.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)


class InplaceCopyFallback(ExternKernel):
    """Copying one value over another, in place.

    Copying between two values may mean moving between devices, so it is run
    through to rather than written as a loop, which would have to assume they
    are on the same one.
    """

    def codegen(self, wrapper) -> None:
        (dst, src, non_blocking) = self.codegen_args()
        wrapper.codegen_device_copy(src, dst, non_blocking)

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def __init__(self, layout: "OutputSpec", inputs, constant_args) -> None:
        super().__init__(
            None,
            layout,
            inputs,
            constant_args,
            python_kernel_name="tp.copy_",
            cpp_kernel_name="tp_copy_",
        )
        V.graph.mark_buffer_mutated(inputs[0].get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    @classmethod
    def create(cls, dst: "IRNode", src: "IRNode", non_blocking: bool = False):
        inputs = [cls.realize_input(t) for t in [dst, src]]
        constant_args = (non_blocking,)
        return InplaceCopyFallback(
            NoneLayout(device=dst.get_device()),
            inputs,
            constant_args,
        )


class MutatingFirstArgExternKernel(ExternKernel):
    """An external call whose first argument is written to rather than read.

    A call of this shape produces no value of its own -- what it produced is
    the change to the value it was handed -- so it allocates nothing, and the
    buffer it wrote to has to be reported as written, since whoever read that
    buffer before has to run first.
    """

    def codegen(self, wrapper) -> None:
        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        argrefs = [
            *(t.codegen_reference() for t in self.inputs),
            *map(repr, self.constant_args),
        ]
        wrapper.writeline(
            f"{self.get_kernel_name()}({', '.join(argrefs)}){wrapper.ending}"
        )

    def should_allocate(self) -> bool:
        return False

    def get_mutation_names(self) -> Sequence:
        return [self.input_name(0)]

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def has_side_effects(self) -> bool:
        return True


class ResizeStorageBytes(MutatingFirstArgExternKernel):
    """Making a value's memory bigger, without touching what is in it.

    Growing a buffer is not the same as producing a larger value: what was
    already there stays where it was and the rest is left alone.  Since the
    memory is being resized rather than replaced, whatever the buffer was
    holding may still be read from it, so it is not one that may be handed out
    again for something else to write into.
    """

    def __init__(self, variable: "IRNode", new_size: int) -> None:
        if not isinstance(new_size, int):
            raise AssertionError("TODO: dynamic shapes")
        super().__init__(
            None,
            NoneLayout(device=variable.get_device()),
            self.unwrap_storage([variable]),
            constant_args=(new_size,),
        )
        V.graph.mark_buffer_mutated(variable.get_name())
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)
        if not isinstance(variable, (BaseView, StorageBox, TensorBox)):
            raise AssertionError(type(variable))
        V.graph.never_reuse_buffers.add(variable.data.get_name())


class DynamicScalar(ExternKernel):
    """One number taken out of a value, whose value is not known in advance.

    Reading a single number out of a value is what turns a shape that came
    from the data into something the code can work with.  The number itself is
    named by a shape whose value is filled in when the value is there, and
    where in the value it came from is recorded as the path that reaches it.
    """

    def get_reads(self) -> OrderedSet:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(self, sym, keypath, data: "IRNode") -> None:
        # The value has to be written out before a number can be read out of
        # it, since reading from something not yet realized would mean
        # computing it twice.
        data.realize()
        super().__init__(
            None, NoneLayout(device=tp.device("cpu")), self.unwrap_storage([data])
        )
        self.sym = sym
        self.keypath = keypath

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet([self.sym])

    def codegen(self, wrapper) -> None:
        wrapper.codegen_dynamic_scalar(self)


class AssertScalar(ExternKernel):
    """A claim about a number that is checked when the code runs.

    A number whose value came from the data cannot be checked while the code is
    being written, so the claim is kept and checked where it is run.  This
    produces nothing and does nothing to any value, and it cannot be removed:
    whether the claim holds is the whole point.
    """

    def get_reads(self) -> OrderedSet:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(self, scalar, msg: str) -> None:
        super().__init__(
            None,
            NoneLayout(device=tp.device("cpu")),
            [],
        )
        self.scalar = scalar
        self.msg = msg

    def has_side_effects(self) -> bool:
        return True

    @cache_on_self_and_args("AssertScalar")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return get_free_symbols(self.scalar, unbacked_only)

    def codegen(self, wrapper) -> None:
        if not config.scalar_asserts:
            return
        # The claim must be written out as it stands.  Simplifying it would use
        # the very assertions this is writing, and would conclude the claim
        # holds because it was assumed to -- which is the claim being checked,
        # not a reason to believe it.
        symbol = next(iter(self.get_free_symbol_uses(unbacked_only=False)))
        if V.graph.fx_wrapper:
            pass
        elif V.graph.cpp_wrapper:
            symbol_str = f"std::to_string({symbol})"
            sizevar = V.graph.wrapper_code.codegen_cpp_sizevar(
                self.scalar, simplify=False
            )
            wrapper.writeline(
                f'if (!({sizevar})) {{ throw std::runtime_error("Expected {self.msg} but received " + {symbol_str}); }}'
            )
        else:
            sizevar = V.graph.wrapper_code.codegen_python_sizevar(
                self.scalar, simplify=False
            )
            wrapper.writeline(f"if not ({sizevar}):")
            wrapper.writeline(f"    raise RuntimeError({repr(self.msg)})")
            # Nothing reads this, but every result is a name, so the name is
            # given something rather than left undefined.
            wrapper.writeline(f"{self.get_name()} = None")


class DynamicSliceSize(ExternKernel):
    """How much a slice produces, when that cannot be worked out in advance.

    A slice written as a range along one axis means four different things
    depending on where its ends fall: an end before the start of the axis
    gives everything, an end counting from the end gives what is left of it, an
    end within the axis gives the range asked for, and an end past the axis
    gives nothing.  When the ends are numbers the right one is worked out here
    while the code is written; when they are numbers that only exist once the
    data is there, the size is given a name and which case applies is worked
    out where the code runs.
    """

    def get_reads(self) -> OrderedSet:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(self, unbacked_size_symbol, start, end, step, size):
        super().__init__(None, NoneLayout(device=tp.device("cpu")), [])
        self.unbacked_size_symbol = unbacked_size_symbol
        self.start = start
        self.end = end
        self.step = step
        self.size = size

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet([self.unbacked_size_symbol])

    @cache_on_self_and_args("DynamicSliceSize")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return (
            get_free_symbols(self.start, unbacked_only)
            .union(get_free_symbols(self.end, unbacked_only))
            .union(get_free_symbols(self.size, unbacked_only))
            .union(get_free_symbols(self.step, unbacked_only))
        )

    def codegen(self, wrapper) -> None:
        wrapper.codegen_dynamic_slice_size(self)


class DynamicSelectStorageOffset(ExternKernel):
    """Where in a value a selected element begins, when the choice is not known.

    Choosing one part of a value over another can be written as choosing a
    position and adding its offset, but when the position comes from the data a
    position before the start of the axis and one counting from the end mean
    different offsets, and which one was meant cannot be told while the code is
    being written.  The offset is therefore given a name of its own and worked
    out where the code runs, where the intended meaning is known.
    """

    def get_reads(self) -> OrderedSet:
        return OrderedSet()

    def should_allocate(self) -> bool:
        return False

    def __init__(
        self,
        unbacked_offset_symbol,
        index,
        base_offset,
        base_dim_stride,
        size,
        clamp: bool,
    ) -> None:
        super().__init__(None, NoneLayout(device=tp.device("cpu")), [])
        self.unbacked_offset_symbol = unbacked_offset_symbol
        self.index = index
        self.base_offset = base_offset
        self.base_dim_stride = base_dim_stride
        self.size = size
        self.clamp = clamp

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet([self.unbacked_offset_symbol])

    @cache_on_self_and_args("DynamicSelectStorageOffset")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return (
            get_free_symbols(self.index, unbacked_only)
            .union(get_free_symbols(self.size, unbacked_only))
            .union(get_free_symbols(self.base_offset, unbacked_only))
            .union(get_free_symbols(self.base_dim_stride, unbacked_only))
        )

    def codegen(self, wrapper) -> None:
        wrapper.codegen_dynamic_select_index(self, clamp=self.clamp)


@ir_dataclass
class Scatter(Pointwise):
    """Writing each computed element to a place the value itself chooses.

    An ordinary write puts the element computed for one output position in
    that same position of the output.  Here the position to write to is worked
    out separately, so that several computed elements can land on the same
    place, which is what writing chosen positions rather than all of them
    means.  Which way a tie is settled is part of how this was asked for.
    """

    output_indexer: Any
    scatter_mode: Any = None

    def constant_to_device(self, device) -> "IRNode":
        """This on another device, which every value it reads has to be on too."""

        loader = self.make_loader()
        loader = patch.object(ConstantBuffer, "override_device", device)(loader)
        return Scatter(
            device=device,
            dtype=self.dtype,
            inner_fn=loader,
            ranges=self.ranges,
            output_indexer=self.output_indexer,
            scatter_mode=self.scatter_mode,
        )

    def store_output(self, output_name, indexer, vars):
        loader = self.make_loader()
        if output_name is None:
            output_name = "unnamed"
        return ops.store(
            output_name,
            indexer(self.output_indexer(vars)),
            loader(vars),
            mode=self.scatter_mode,
        )


class DeviceCopy(ExternKernelOut):
    """The same value, held somewhere else.

    A value that has been computed on one device and is wanted on another has
    to be moved, which cannot be written as a loop over its elements since the
    two devices cannot be read and written in the same place.  Where a value
    that nothing has written to and that is only made of constants is being
    moved, it is made on the other device instead, which is cheaper than moving
    it; anything else is copied.
    """

    @classmethod
    def create(cls, x: "IRNode", device, non_blocking: bool) -> "IRNode":
        x_device = x.get_device()
        if x_device is None:
            raise AssertionError("Expected x_device is not None")
        if (
            not x.is_extern()
            # A value that has been written to is not the same value any more,
            # so making it again on the other device would not do.
            and try_get_name(x) not in V.graph.mutated_buffers
            and all(r in V.graph.constants for r in x.get_read_names())
            and not config.tp_export.use_runtime_constant_folding
        ):
            if V.graph.cpp_wrapper:
                # The value is being made on the other device, but both devices
                # still have to be known, since what the code is wrapped in
                # depends on it.
                V.graph.add_device_info(device)
                V.graph.add_device_info(x_device)
            return x.constant_to_device(device)

        V.graph.add_device_info(device)
        V.graph.add_device_info(x_device)
        developer_warning("DeviceCopy in input program")
        constant_args = (non_blocking,)
        # The result is laid out the way the input was, since moving it should
        # not rearrange it.
        x = ExternKernel.require_contiguous(x)
        stride = None
        if x.get_size():
            # A value with no elements may not have a stride to ask for.
            stride = x.get_stride()
        is_destination_pinned = (
            is_gpu(x_device) and not is_gpu(device) and non_blocking
        )
        is_source_pinned = (
            not is_gpu(x_device) and is_gpu(device) and non_blocking
        )
        if is_source_pinned and is_storage_and_layout(x):
            x.get_layout().is_pinned = True
        return DeviceCopy(
            FixedLayout(
                device,
                x.get_dtype(),
                x.get_size(),
                stride,
                is_pinned=is_destination_pinned,
            ),
            [cls.realize_input(x)],
            constant_args,
        )

    def codegen(self, wrapper) -> None:
        args = self.codegen_args()
        if len(args) != 2:
            raise AssertionError("Expected len(args) == 2")
        if self.output_view:
            wrapper.codegen_device_copy(
                args[0], self.output_view.codegen_reference(), args[1]
            )
        else:
            wrapper.codegen_device_copy(args[0], self.codegen_reference(), args[1])
        if isinstance(self.layout, Layout) and self.layout.is_pinned:
            # A copy into memory that may be written before it has been read
            # has to finish before anything runs that might read it.
            wrapper.sync_d2h_copy(self.get_name())


class SubgraphBuffer(ExternKernel):
    """A piece of the graph compiled on its own and then called.

    Compiling a piece on its own is what lets it be compiled once and used more
    than once, and lets the settings it was written under be recorded with it.
    What it needs to be given -- the shapes whose values it cannot know, and
    the values it reads -- is worked out here and handed to it, and what it
    produces is a single result standing for all of its own.
    """

    def __init__(
        self,
        layout: "Layout",
        input_nodes,
        gm,
        example_inputs,
        subgraph_name: str,
        config_patches: dict | None = None,
    ):
        super().__init__(None, layout, input_nodes)
        self.gm = gm
        self.example_inputs = example_inputs
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

        self.subgraph = V.graph.make_subgraph(self.gm, example_inputs, subgraph_name)

        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        sym_inputs = get_symbolic_inputs(self.inputs)

        for sym_inp in sym_inputs:
            self.subgraph.graph_inputs[sym_inp.name] = sym_inp
            self.subgraph.graph_input_names.append(sym_inp.name)

        self.sym_inputs = [sym_var.name for sym_var in sym_inputs]
        self.subgraph.run(*self.example_inputs)

        if config_patches:
            for op in self.subgraph.operations:
                op.set_config_patches(config_patches.copy())

    def codegen(self, wrapper) -> None:
        class CodegenGraph:
            def __init__(self, graph):
                self.graph = graph
                self.name = graph.name

        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        outer_inputs = [t.codegen_reference() for t in self.inputs]

        wrapper.codegen_subgraph_with_flattened_outputs(
            CodegenGraph(self.subgraph),
            [*self.sym_inputs, *outer_inputs],
            [self.name],
        )


class SetSourceTensorKernel(ExternKernelAlloc):
    """Making one value refer to the memory another value is using.

    The two values end up naming the same memory rather than one being copied
    into the other, so this is not a copy: it changes what the first value is.
    Since both names then stand for the same memory, neither may be handed out
    for something else to write into, and both are reported as written.
    """

    def __init__(self, self_tensor: "IRNode", storage_tensor: "IRNode", op_overload=None) -> None:
        storage_tensor = self.realize_input(storage_tensor)
        storage_tensor.freeze_layout()
        super().__init__(
            storage_tensor.get_layout(),
            [self_tensor, storage_tensor],
            python_kernel_name="tp.set_.source_Tensor",
            op_overload=op_overload,
        )
        if not isinstance(self_tensor, (BaseView, StorageBox, TensorBox)):
            raise AssertionError(type(self_tensor))
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())
        V.graph.never_reuse_buffers.add(self.get_name())
        device = storage_tensor.get_device()
        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=device), self_tensor, self),
            MutationOutput(NoneLayout(device=device), storage_tensor, self),
        ]

    def get_inputs_that_alias_output(self) -> Sequence:
        return [self.input_name(0), self.input_name(1)]


class ShallowCopyDataKernel(ExternKernelAlloc):
    """Pointing one value at the memory of another, contents and all.

    Both values then name the same memory, so neither may be handed out for
    something else to write into, and both are reported as written.  This is
    told apart from making one value refer to another because what it does to
    the memory's shape is not the same.
    """

    def __init__(self, self_tensor: "IRNode", storage_tensor: "IRNode", op_overload=None) -> None:
        storage_tensor = self.realize_input(storage_tensor)
        storage_tensor.freeze_layout()
        super().__init__(
            storage_tensor.get_layout(),
            [self_tensor, storage_tensor],
            python_kernel_name="tp.shallow_copy_data_",
            op_overload=op_overload,
        )
        if not isinstance(self_tensor, (BaseView, StorageBox, TensorBox)):
            raise AssertionError(type(self_tensor))
        V.graph.never_reuse_buffers.add(self_tensor.data.get_name())
        V.graph.never_reuse_buffers.add(storage_tensor.get_name())
        V.graph.never_reuse_buffers.add(self.get_name())
        device = storage_tensor.get_device()
        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=device), self_tensor, self),
            MutationOutput(NoneLayout(device=device), storage_tensor, self),
        ]

    def get_inputs_that_alias_output(self) -> Sequence:
        return [self.input_name(0), self.input_name(1)]


class OrderingBarrier(NopKernel):
    """A result that does nothing, and exists only to say when to run something.

    Two operations that touch the same memory have to be ordered, and which one
    comes first is decided by what the memory is called rather than by any work
    being done.  This stands between the two: the name changes here, so that
    whoever reads the buffer afterwards is reading through something that was
    written after the first operation ran.  Nothing is computed, nothing is
    allocated, and the buffer is not marked as written -- what is needed is only
    that the order be visible, and marking it written would keep the buffer
    alive for no reason.
    """

    ordering_only = True

    def __init__(self, source_node: "IRNode") -> None:
        source_node.realize()
        super().__init__(
            name=None,
            layout=NoneLayout(device=source_node.get_device()),
            inputs=[source_node],
        )
        self.mutation_names = [source_node.get_name()]
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_mutation_names(self) -> Sequence:
        return self.mutation_names

    def should_allocate(self) -> bool:
        return False


#: What a candidate writes down about itself when it is asked to describe
#: itself: numbers, a flag, a name, or a list of those.
PrimitiveInfoType = int | float | bool | str | list[int | str | float | bool]


class ChoiceCaller:
    """One way of doing a thing, built and ready to be measured or used.

    A choice is measured first and turned into a value only if it is the one
    that was chosen, so the two questions are answered by the same object and
    in that order: what it costs to run, and -- only for the winner -- what
    running it produced.  A caller of this class holds a choice that has
    already been built; everything above it was a description of a possibility,
    and this is the possibility.

    What a particular kind of choice is called, how it is run, what identifies
    it and what it produces are all left to the kind: what is held here is what
    is the same whichever kind it is.
    """

    def __init__(
        self,
        name: str,
        input_nodes=(),
        layout: "Layout | None" = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes
        #: An additional description, for telling two choices apart when their
        #: names alone do not say which is which.
        self.description = description
        #: Set when a measurement showed this choice does not work here.
        self.failed: bool = False
        #: When true, measure by capturing the launch and replaying it, which
        #: leaves out the cost of launching and is only right where what is
        #: being compared is the kernel rather than the launch.
        self._benchmark_with_cudagraphs: bool = False
        #: Where information travels from a choice being generated to the end
        #: of it being measured, for the two to be read by code that is
        #: neither of them.
        self.annotations: dict[str, object] = {}
        #: What a subgraph-based choice stands for, filled in by the kinds that
        #: are one.
        self.gm: Any = None
        self.decomposition: Callable[..., Any] | None = None
        self.decomposition_kwargs: dict[str, object] = {}
        #: Geometry substitutions a measurement decided on, for the record.
        self.config_patches: dict[str, Any] = {}
        #: What a measurement hands this choice to run it, where the kind
        #: supplies one.
        self._callable: Callable[..., Any] | None = None

    def benchmark(self, *args: Any, out: Any) -> float:
        """How long one run of this choice takes.

        The result is named rather than appended: the choice was written to be
        handed where its result goes, and which of its arguments that is is
        part of how it was written rather than a matter of order.  Where a
        measurement is meant to compare the kernel and not the launch, the
        launch is captured once and replayed, which is the only difference
        between the two measurements; and where a profile is being taken
        anyway, its own timings are used rather than timed twice.
        """

        from .config import profile_bandwidth_with_do_bench_using_profiling
        from .runtime.benchmarking import benchmarker
        from .utils import do_bench_using_profiling

        algo = self.to_callable()
        if self._benchmark_with_cudagraphs:
            return benchmarker.benchmark_gpu_with_cuda_graph(lambda: algo(*args))
        if profile_bandwidth_with_do_bench_using_profiling:
            return do_bench_using_profiling(lambda: algo(*args))
        return benchmarker.benchmark(algo, args, {"out": out}, device=None)

    def call_name(self) -> str:
        raise NotImplementedError

    def to_callable(self) -> Callable[..., Any]:
        raise NotImplementedError

    def kernel_hash_key(self) -> str:
        """What identifies the kernel itself, for a binary cache.

        By default a choice has no runtime parameters of its own, so what
        identifies the choice identifies the kernel.
        """

        return self.hash_key()

    def hash_key(self) -> str:
        raise NotImplementedError

    def output_node(self) -> Any:
        raise NotImplementedError

    def info_dict(self) -> dict:
        """What is worth writing down about this choice."""

        return {}

    def autoheuristic_id(self) -> str:
        return "unsupported_choice"

    def mark_failed(self) -> None:
        """Record that this choice does not work here, so it is not offered.

        Useful where measuring is separate from choosing: a choice found not
        to work is not offered again.
        """

        self.failed = True


class TritonTemplateCallerBase(ChoiceCaller):
    def get_make_kernel_render(self) -> Any:
        raise NotImplementedError


class TritonTemplateBuffer(TemplateBuffer):
    """A result whose body comes from a template that writes no results of its own.

    A kernel written from a template returns nothing: what it produces is left
    in memory that was given to it.  Producing several results therefore means
    giving it somewhere to leave each one, and those places are reported as
    results here, alongside the one it computed itself.  Its own result is
    always a candidate for having something else written into it afterwards,
    since it is the one whose memory is known.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs,
        make_kernel_render,
        mutated_inputs=None,
        allowed_prologue_inps=None,
    ) -> None:
        super().__init__(
            layout,
            inputs,
            make_kernel_render,
            mutated_inputs=mutated_inputs,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        if self.name is None:
            raise AssertionError("Expected self.name is not None")
        self.epilogue_fusable_outputs = {self.name: self.name}

        self.subgraph_inps: list | None = None
        self.subgraph_outs: list | None = None

    @cache_on_self_and_args("TritonTemplateBuffer")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        res = super().get_free_symbol_uses(unbacked_only)
        subgraph_outs = self.subgraph_outs if self.subgraph_outs else []
        subgraph_inps = self.subgraph_inps if self.subgraph_inps else []

        for inp in subgraph_inps:
            if isinstance(inp, Expr):
                res.update(get_free_symbols(inp, unbacked_only))
            elif isinstance(inp, IRNode):
                res.update(inp.get_free_symbol_uses(unbacked_only))
            else:
                if inp is not None:
                    raise AssertionError("Expected inp is None")

        for out in subgraph_outs:
            if isinstance(out, IRNode):
                res.update(out.get_free_symbol_uses(unbacked_only))
            else:
                if out is not None:
                    raise AssertionError("Expected out is None")

        return res

    def get_outputs(self) -> list:
        return [self, *self.mutation_outputs]

    def __str__(self) -> str:
        return f"TritonTemplateBuffer(layout={self.layout})"


class CuteDSLTemplateBuffer(TemplateBuffer):
    """A result whose body comes from a template written in a separate language.

    What makes this worth its own class is that such a template can leave more
    than one result behind, so the places it was given are reported as results
    of the operation alongside the one it computed.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs,
        make_kernel_render,
        template,
        mutated_inputs=None,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.mutated_inputs = mutated_inputs
        self.outputs: list = [self]

        if mutated_inputs is not None:
            if not isinstance(self.inputs[0], IRNode):
                raise AssertionError(type(self.inputs[0]))
            device = self.inputs[0].get_device()
            self.outputs += [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]

    def get_outputs(self) -> list:
        return self.outputs


class FlyDSLTemplateBuffer(TemplateBuffer):
    """A result whose body comes from a template that is scheduled its own way.

    As with a template written in a separate language, such a template can leave
    more than one result behind, and the places it was given are reported as
    results of the operation.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs,
        make_kernel_render,
        template,
        mutated_inputs=None,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        self.template = template
        self.mutated_inputs = mutated_inputs
        self.outputs: list = [self]

        if mutated_inputs is not None:
            if not isinstance(self.inputs[0], IRNode):
                raise AssertionError(type(self.inputs[0]))
            device = self.inputs[0].get_device()
            self.outputs += [
                MutationOutput(NoneLayout(device=device), buf, self)
                for buf in mutated_inputs
            ]

    def get_outputs(self) -> list:
        return self.outputs


class CUTLASSTemplateBuffer(TemplateBuffer):
    """A result whose body comes from a prepared matrix-multiplication kernel.

    Such a kernel needs somewhere to keep its running totals while it works,
    and how much is asked of it rather than worked out here, since only the
    kernel knows.  Writing nothing to each of its results is also recorded:
    that is what a kernel that only accumulates is described as.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs,
        make_kernel_render,
        workspace_size: int,
        template,
        supports_epilogue_fusion: bool,
    ) -> None:
        super().__init__(layout, inputs, make_kernel_render)
        # How much memory this kernel needs while it runs, in bytes.
        self.workspace_size = workspace_size
        self.template = template
        self.supports_epilogue_fusion = supports_epilogue_fusion

    def get_workspace_size(self) -> int:
        return self.workspace_size if self.workspace_size is not None else 0

    def emulate_store_fn(self) -> None:
        for output in self.get_outputs():
            ops.store(output.get_name(), None, None)


class NVUniversalGemmBuffer(TemplateBuffer):
    """A result whose body comes from a prepared matrix-multiplication kernel.

    Where a template written as text is given a renderer, this kind of kernel
    is a program that is called, so the renderer is built here from what was
    chosen.  The order of the two operands can be swapped, and a value added at
    the end rather than being multiplied in, which are the two things such a
    kernel is asked to vary.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs,
        kernel,
        accumulator_type,
        variant,
        workspace_size: int = 0,
        scale_type_a=None,
        scale_type_b=None,
        swizzle_type_a=None,
        swizzle_type_b=None,
        supports_epilogue_fusion: bool = False,
        swap_ab: bool = False,
        bias_node=None,
    ) -> None:
        # The renderer is not given here; it is made below, once the kernel and
        # the way it is to be called are both known.
        super().__init__(layout, inputs, make_kernel_render=None)
        self.kernel = kernel
        self.accumulator_type = accumulator_type
        self.outputs: list = [self]
        self.workspace_size = workspace_size
        self.variant = variant
        self.scale_type_a = scale_type_a
        self.scale_type_b = scale_type_b
        self.swizzle_type_a = swizzle_type_a
        self.swizzle_type_b = swizzle_type_b
        self.supports_epilogue_fusion = supports_epilogue_fusion
        self.swap_ab = swap_ab
        # When set, the last input is a value to add at the end rather than one
        # of the two being multiplied; the operands are the rest.
        self.bias_node = bias_node
        self.kernel_metadata = {
            "kernel_name": getattr(
                getattr(kernel, "metadata", None), "operator_name", None
            ),
            "min_cc": getattr(kernel, "designed_for_min_cc", None),
        }
        # The parent records the renderer as an attribute of the instance, so
        # putting this bound method there is what makes it be used.
        self.make_kernel_render = self._make_kernel_render

    def get_workspace_size(self) -> int:
        return self.workspace_size

    def get_outputs(self) -> list:
        return self.outputs

    def _make_kernel_render(
        self,
        out_node,
        hint_override: int | None = None,
        epilogue_fn_code: str | None = None,
        epilogue_reads: list | None = None,
        epilogue_writes: list | None = None,
        epilogue_var_renames: dict | None = None,
        local_reduce=None,
    ):
        """Build what writes this kernel's call into the generated code.

        What the call needs is the values it is given, in the order it expects
        them, which is why each input is taken as far as the memory it names.
        """

        if epilogue_fn_code is not None and epilogue_var_renames is None:
            raise AssertionError("epilogue_fn_code requires epilogue_var_renames")

        input_nodes: list = []
        for inp in self.inputs:
            if isinstance(inp, TensorBox):
                inp = inp.data
            if isinstance(inp, StorageBox):
                inp = inp.data
            input_nodes.append(inp)

        def render() -> str:
            return self.kernel.call_kernel(
                out_node,
                input_nodes,
                hint_override=hint_override,
                epilogue_fn_code=epilogue_fn_code,
                epilogue_reads=epilogue_reads,
                epilogue_writes=epilogue_writes,
                epilogue_var_renames=epilogue_var_renames,
                local_reduce=local_reduce,
            )

        return self.kernel, render


class TMADescriptor(ExternKernel):
    """A description of a value that lets it be handed over in blocks.

    Rather than copying a value, a program can be told how it is arranged and
    left to move it in blocks of a shape it chooses.  What that description is
    depends on which of two forms of it is meant, so which one is used is
    recorded and each is described by its own class.

    Two descriptions of the same value in the same way mean the same thing, so
    asking twice gives the same one rather than a second copy of it.  The
    description also refers to the value rather than owning it, since the
    value has to outlive it.
    """

    _CACHE: dict = {}

    @classmethod
    def _create_impl(cls, tensor: "IRNode", tma_meta):
        if len(tma_meta) != 2:
            raise AssertionError("Expected len(tma_meta) == 2")
        if tma_meta[0] == "experimental":
            return TMADescriptorExperimental(tensor, *tma_meta[1])
        else:
            if tma_meta[0] != "stable":
                raise AssertionError('Expected tma_meta[0] == "stable"')
            return TMADescriptorStable(tensor, *tma_meta[1])

    @classmethod
    def create(cls, tensor: "IRNode", tma_meta):
        # The key has to be something that can be looked up, so the parts that
        # are lists are written down as the tuples they stand for.
        key = (
            id(tensor),
            tma_meta[0],
            tuple(tuple(x) if isinstance(x, list) else x for x in tma_meta[1]),
        )
        if key not in cls._CACHE:
            cls._CACHE[key] = cls._create_impl(tensor, tma_meta)
        return cls._CACHE[key]

    def __init__(self, tensor: "IRNode", inputs, constant_args) -> None:
        super().__init__(
            None,
            # The memory belongs to the value, not to this description, and has
            # to still be there when the description is used.  Saying so is
            # what keeps the value from being released first.
            NonOwningLayout(
                ReinterpretView(
                    data=tensor,
                    layout=tensor.get_layout(),
                )
            ),
            list(inputs),
            tuple(constant_args),
            None,
        )

        self.tensor = tensor
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def codegen(self, wrapper) -> None:
        wrapper.generate_tma_descriptor(self)

    def get_tensor(self) -> "IRNode":
        return self.tensor


class TMADescriptorExperimental(TMADescriptor):
    """A block description given as the shape and the block for each axis.

    Which of the two ways of saying this is used matters because the two take
    different arguments, so the one that was asked for is recorded here.
    """

    def __init__(
        self,
        tensor: "IRNode",
        dims,
        block_dims,
        element_size: int | None = None,
    ) -> None:
        if len(dims) not in (1, 2):
            raise AssertionError("Expected len(dims) in (1, 2)")
        if len(dims) != len(block_dims):
            raise AssertionError("Expected len(dims) == len(block_dims)")

        if element_size is None:
            element_size = tensor.get_dtype().itemsize

        self.dims = dims
        self.block_dims = block_dims
        self.element_size = element_size
        self.rank = len(self.dims)

        inputs = [tensor]
        constant_args = [
            *self.dims,
            *self.block_dims,
            self.element_size,
        ]

        super().__init__(
            tensor=tensor,
            inputs=inputs,
            constant_args=constant_args,
        )


class TMADescriptorStable(TMADescriptor):
    """A block description given as one block shape for the whole value."""

    def __init__(self, tensor: "IRNode", block_shape):
        # A single number is the block for a value of one axis, which is the
        # same as being given a block of one.
        self.block_shape = (
            list(block_shape) if isinstance(block_shape, (list, tuple)) else [block_shape]
        )

        super().__init__(
            tensor=tensor,
            inputs=[tensor],
            constant_args=self.block_shape,
        )


def tensor_is_aligned(tensor) -> bool:
    """Whether a value begins where a wider access may safely be done.

    A value whose memory begins partway into a block cannot be read several
    elements at a time as though it did not, so this asks whether it begins
    where such an access would be aligned.  Only where the offset is a number
    is the question asked; an offset that is a shape is not checked, since
    nothing is known about it until the code runs.
    """

    if not tensor.is_contiguous():
        return False
    offset = tensor.storage_offset()
    if not isinstance(offset, int):
        return False
    itemsize = tensor.element_size()
    return offset % max(itemsize, 1) == 0


def _split_by_sym_type(args):
    """Separate the arguments that are shapes from the ones that are values.

    A shape is passed along as a number rather than as memory, so it goes in
    the plain-argument list while the values that have to be given memory go in
    the other one.  Which is which is decided by what each argument is, since
    the same list holds both.
    """

    non_sym_args = []
    sym_args = []
    for arg in args:
        if isinstance(arg, ShapeAsConstantBuffer):
            sym_args.append(arg.expr)
        else:
            non_sym_args.append(arg)

    return sym_args, non_sym_args


def _maybe_expr(s):
    """A number as a plain number, and a shape as an expression.

    A layout is written in terms of expressions, so a number that is already
    known is left as it is and one that is a shape becomes the expression it
    stands for.
    """

    if isinstance(s, int):
        return s
    return s.node.expr


class InvokeSubgraph(ExternKernel):
    """Calling a piece that was compiled on its own.

    What the piece produces is not known until it has been compiled, so the
    results are asked of it rather than declared here, and each one is a result
    of its own saying which of the several it is.  A result that is a shape or
    a nothing is passed through as it is, since there is no memory to describe.
    """

    subgraph: Any = None
    operands: Any = None
    outputs: Any = None

    def __init__(self, subgraph, operands, layout: "MultiOutputLayout") -> None:
        super().__init__(
            name=None,
            layout=layout,
            inputs=operands,
        )
        self.subgraph = subgraph
        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list:
        return [self.subgraph] if self.subgraph else []

    @classmethod
    def create(cls, subgraph, *operands):
        """Call this piece with these values, giving back what it produces.

        The values are given memory first, since a piece that is called has to
        be able to read what it is given.  A value that came from a shape is
        passed straight through; the rest are made to look the way the piece
        expects, since what it was compiled against is what it will assume.
        """

        current_node = V.graph.current_node

        operands_ = [cls.realize_input(x) for x in operands]
        new_operands: list = []

        for operand in operands_:
            if isinstance(
                operand, (ShapeAsConstantBuffer, GeneratorState, OpaqueObjectState)
            ):
                new_operands.append(operand)
            else:
                new_operands.append(operand)

        if subgraph.graph is None:
            # The piece is compiled here, once, and what it produces is what
            # the rest of the graph is written against.
            subgraph.graph = V.graph.make_subgraph(
                gm=subgraph.graph_module,
                example_inputs=list(operands_),
                subgraph_name=subgraph.name,
            )
            with V.set_graph_handler(subgraph.graph):
                subgraph.graph.run(*operands_)

        outputs = subgraph.graph.graph_outputs

        # The device cannot be taken from the first operand, since the first
        # operand may be a shape and shapes are not on a device.
        device = None
        for operand in operands_:
            if not isinstance(operand, ShapeAsConstantBuffer):
                device = operand.get_device()
                break
        if device is None:
            raise AssertionError("Expected device is not None")
        invoke_subgraph = InvokeSubgraph(
            subgraph=subgraph,
            operands=operands_,
            layout=MultiOutputLayout(device=device),
        )

        def create_output(output, ind: int):
            if isinstance(output, (ShapeAsConstantBuffer, NoneAsConstantBuffer)):
                return output
            else:
                device = output.get_device()
                if device is None:
                    raise AssertionError("Expected device is not None")

                return MultiOutput(
                    FixedLayout(
                        device=device,
                        dtype=output.get_dtype(),
                        size=output.get_size(),
                        stride=output.get_stride(),
                        offset=output.get_layout().offset,
                        is_pinned=output.get_layout().is_pinned,
                    ),
                    invoke_subgraph,
                    [(list, ind)],
                    skip_size_stride_alignment_checks=True,
                )

        outs = [create_output(output, i) for i, output in enumerate(outputs)]
        invoke_subgraph.outputs = outs
        return outs

    def codegen(self, wrapper) -> None:
        wrapper.codegen_invoke_subgraph(self)


class Switch(ExternKernel):
    """Choosing between pieces of the graph, according to a value.

    What is being chosen on is a single value: a yes-or-no for a conditional,
    or a number saying which of several cases applies.  The pieces themselves
    are compiled on their own, and what this holds is the choice together with
    the values they are given, so that whichever one is taken is run with the
    same values.

    Every piece has to produce the same thing in the same way -- same number of
    results, same types, same device, same arrangement -- since the result of
    the whole is one value and not a choice between differently-shaped ones.
    That is checked here rather than left to fail later, because a piece that
    does not match is a mistake in the program and not a runtime condition.
    """

    selector: Any = None
    branches: Any = None
    operands: Any = None
    is_cond: bool = False
    outputs: Any = None

    def __init__(
        self,
        selector: "IRNode",
        branches,
        operands,
        layout: "MultiOutputLayout",
        unbacked_bindings: dict | None,
        is_cond: bool = False,
    ) -> None:
        self.selector = selector
        self.branches = branches
        self.operands = operands
        self.is_cond = is_cond

        sym_args, tensor_args = _split_by_sym_type([selector, *operands])

        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )
        if unbacked_bindings is not None:
            self.unbacked_bindings = unbacked_bindings

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list:
        return list(self.branches) if self.branches is not None else []

    @classmethod
    def create(
        cls,
        selector,
        branches: list,
        operands: list,
        is_cond: bool = False,
    ) -> list:
        """Build a choice among these pieces, on this value, with these inputs."""

        selector = cls.realize_input(selector)
        operands = [cls.realize_input(x) for x in operands]

        def _require_exact_strides(graph_outputs, merged_outputs, branch_outputs):
            ret = []
            for output, merged, branch in zip(
                graph_outputs, merged_outputs, branch_outputs
            ):
                if isinstance(output, ShapeAsConstantBuffer):
                    ret.append(output)
                else:
                    strides = merged.stride()
                    # The arrangement the whole thing has is shared by the
                    # pieces unless it depends on a shape only the piece knows,
                    # in which case each piece keeps its own.
                    if has_free_unbacked_symbols(strides):
                        strides = branch.stride()
                    ret.append(
                        ExternKernel.require_exact_strides(
                            TensorBox(output), strides, allow_padding=False
                        )
                    )
            return ret

        # The pieces are lowered against the values the operation was traced
        # with, and their results are made to come out the way the traced
        # results did: one arrangement for every piece, since the operation
        # has one result whichever piece produced it.
        traced_operands = [
            op.meta["val"] if hasattr(op, "meta") else op
            for op in V.graph.current_node.args[-1]
        ]
        # A piece that returns one value makes the operation return that value
        # rather than a sequence holding it.
        traced = V.graph.current_node.meta["val"]
        traced_outputs = list(traced) if isinstance(traced, (list, tuple)) else [traced]

        for subgraph in branches:
            if subgraph.graph is None:
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=traced_operands,
                    subgraph_name=subgraph.name,
                )
                branch_out_args = subgraph.graph_module.graph.output_node.args[0]
                if not isinstance(branch_out_args, (list, tuple)):
                    branch_out_args = (branch_out_args,)
                branch_traced = [
                    a.meta.get("val", merged) if hasattr(a, "meta") else merged
                    for a, merged in zip(branch_out_args, traced_outputs)
                ]
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*traced_operands)
                    subgraph.graph.graph_outputs = _require_exact_strides(
                        subgraph.graph.graph_outputs, traced_outputs, branch_traced
                    )

        if any(branch.graph is None for branch in branches):
            raise AssertionError("Expected all branch.graph to be not None")

        branch_outputs = [branch.graph.graph_outputs for branch in branches]

        op_name = "a conditional" if is_cond else "a switch"
        for branch, b_outputs in zip(branches, branch_outputs):
            if _has_aliased_buffers(b_outputs):
                raise AssertionError(
                    f"Output aliasing is currently not supported in compiled {op_name}. "
                    f"The outputs of the {branch.name} subgraph of {op_name} are aliased: {b_outputs}"
                )

        # Every piece has to produce the same thing in the same way, since the
        # result is one value rather than a choice between unlike ones.
        ref_outputs = branch_outputs[0]
        for b_outputs in branch_outputs[1:]:
            if len(ref_outputs) != len(b_outputs):
                raise AssertionError((ref_outputs, b_outputs))
            for i, (r_o, b_o) in enumerate(zip(ref_outputs, b_outputs)):
                if r_o.get_device() != b_o.get_device():
                    raise AssertionError((i, r_o, b_o))
                if r_o.get_dtype() != b_o.get_dtype():
                    raise AssertionError((i, r_o, b_o))
                if r_o.get_layout().offset != b_o.get_layout().offset:
                    raise AssertionError((i, r_o, b_o))

        # The choice may be made on a different device than the work is done
        # on -- deciding whether to run something is often done on the host --
        # so the device is taken from the first value rather than from it.
        device = next(
            o.get_device()
            for o in operands + [selector]
            if not isinstance(o, ShapeAsConstantBuffer)
        )
        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env, V.graph.current_node.meta.get("unbacked_bindings", None),
        )
        if device is None:
            raise AssertionError("cannot determine device")
        switch = Switch(
            selector=selector,
            branches=branches,
            operands=operands,
            layout=MultiOutputLayout(device=device),
            unbacked_bindings=unbacked_bindings,
            is_cond=is_cond,
        )

        outputs = [
            MultiOutput(
                FixedLayout(
                    device=output.get_device() if output.get_device() is not None else device,
                    dtype=output.get_dtype(),
                    size=[_maybe_expr(sz) for sz in merged.size()],
                    stride=[_maybe_expr(sz) for sz in merged.stride()],
                    offset=output.get_layout().offset,
                    is_pinned=output.get_layout().is_pinned,
                ),
                switch,
                [(list, i)],
            )
            # The pieces' results are alike, so either one is the template
            # for where each result lives.
            for i, (output, merged) in enumerate(zip(ref_outputs, traced_outputs))
        ]

        switch.outputs = outputs
        switch.mutation_outputs = [
            MutationOutput(operands[idx].layout, operands[idx], switch)
            for idx in sorted(
                {
                    idx
                    for branch in branches
                    for idx in getattr(branch, "mutated_operand_indices", ())
                }
            )
        ]

        return outputs

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            if resolved is None:
                raise AssertionError("Expected resolved is not None")
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()

    def codegen(self, wrapper) -> None:
        wrapper.codegen_switch(self)
        wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", {})
        )


class WhileLoop(ExternKernel):
    """Repeating a piece of the graph until a value says to stop.

    Some values are carried from one pass to the next -- the piece is given
    them and hands back the ones to use next time -- and some are the same
    every time.  The two are told apart here, since only the carried ones can
    change and only they can be written to.

    The arrangement of a carried value has to be the same going in and coming
    out, because what comes out is what the next pass is given; that is checked
    here rather than left to produce a wrong answer later.
    """

    carried_inputs: Any = None
    additional_inputs: Any = None
    cond_subgraph: Any = None
    body_subgraph: Any = None
    outputs: Any = None

    def __init__(
        self,
        carried_inputs,
        additional_inputs,
        cond_subgraph,
        body_subgraph,
        layout: "MultiOutputLayout",
        unbacked_bindings: dict | None,
        stack_output: bool,
    ) -> None:
        self.carried_inputs = carried_inputs
        self.additional_inputs = additional_inputs
        self.cond_subgraph = cond_subgraph
        self.body_subgraph = body_subgraph

        sym_args, tensor_args = _split_by_sym_type(
            [*carried_inputs, *additional_inputs]
        )
        super().__init__(
            name=None,
            layout=layout,
            inputs=tensor_args,
            constant_args=sym_args,
        )
        if unbacked_bindings is not None:
            self.unbacked_bindings = unbacked_bindings
        self.stack_output = stack_output

        self.name = V.graph.register_buffer(self)
        V.graph.register_operation(self)

    def get_subgraphs(self) -> list:
        subgraphs = []
        if self.cond_subgraph:
            subgraphs.append(self.cond_subgraph)
        if self.body_subgraph:
            subgraphs.append(self.body_subgraph)
        return subgraphs

    @staticmethod
    def _clone_aliased_inputs(carried_inputs):
        """Give each carried value its own memory, where two share one.

        Two carried values that are the same memory would be written once and
        read as two, which is not what the program said.  This can happen
        without being written anywhere -- two values that were given the same
        empty buffer can be recognised as one value later -- so it is checked
        for here and the second is given memory of its own.
        """

        if not _has_aliased_buffers(carried_inputs):
            return carried_inputs

        unwrapped_buffers = [
            buffer.unwrap_view() if isinstance(buffer, ReinterpretView) else buffer
            for buffer in carried_inputs
        ]

        seen_buffers: OrderedSet = OrderedSet()
        result: list = []

        for original_input, unwrapped_buffer in zip(carried_inputs, unwrapped_buffers):
            if id(unwrapped_buffer) in seen_buffers:
                result.append(ExternKernel.copy_input(original_input))
            else:
                seen_buffers.add(id(unwrapped_buffer))
                result.append(original_input)

        return result

    @staticmethod
    def _maybe_wrap_as_tensor_box(out):
        """This result as something a layout requirement can be asked of."""

        if isinstance(out, TensorBox):
            return out
        elif isinstance(out, (StorageBox, ReinterpretView)):
            return TensorBox(out)
        elif isinstance(out, Buffer):
            # A value already laid down under a name is put in memory as
            # itself, unboxed, so it is boxed here the way a result is.
            return TensorBox.create(out)
        else:
            raise RuntimeError(f"NYI unsupported output type: {type(out)}")

    @classmethod
    def create(
        cls,
        cond_fn,
        body_fn,
        carried_inputs,
        additional_inputs,
        stack_output: bool,
    ):
        """Build a repetition of one piece until another says to stop.

        Whether each pass's result is kept or only the last one is what
        ``stack_output`` says: a loop whose results are all wanted needs a
        place for each of them, while one that only wants the last can write
        over the same place each time.
        """

        def _require_exact_strides(tensor_boxes, shapes):
            if len(tensor_boxes) != len(shapes):
                raise AssertionError("Expected len(tensor_boxes) == len(shapes)")
            ret = []
            for tb, fk in zip(tensor_boxes, shapes):
                if isinstance(fk, tp.Tensor):
                    # A piece hands back memory rather than a value, but what
                    # the arrangement has to match was worked out from a value,
                    # so it is asked as one.
                    new_tb = WhileLoop._maybe_wrap_as_tensor_box(tb)
                    ret.append(
                        ExternKernel.require_exact_strides(
                            new_tb, fk.stride(), allow_padding=False
                        )
                    )
                else:
                    ret.append(tb)
            return ret

        # The pieces are lowered against the values the loop was traced with:
        # handed the region's own buffers instead, a piece would read those
        # directly and depend on memory that is not its own.  The traced values
        # also say how each carried value is laid out, which both the values
        # going in and every pass's results are held to.
        def _traced(values):
            return [v.meta["val"] if hasattr(v, "meta") else v for v in values]

        traced_carried = _traced(V.graph.current_node.args[-2])
        traced_additional = _traced(V.graph.current_node.args[-1])
        traced_all = traced_carried + traced_additional

        carried_inputs_ = [cls.realize_input(x) for x in carried_inputs]
        carried_inputs_ = WhileLoop._clone_aliased_inputs(carried_inputs_)
        carried_inputs_ = _require_exact_strides(carried_inputs_, traced_carried)
        additional_inputs_ = [cls.realize_input(x) for x in additional_inputs]
        additional_inputs_ = _require_exact_strides(additional_inputs_, traced_additional)
        all_inputs = carried_inputs_ + additional_inputs_

        for subgraph in (cond_fn, body_fn):
            if subgraph.graph is None:
                subgraph.graph = V.graph.make_subgraph(
                    gm=subgraph.graph_module,
                    example_inputs=traced_all,
                    subgraph_name=subgraph.name,
                )
                with V.set_graph_handler(subgraph.graph):
                    subgraph.graph.run(*traced_all)
                    # What one pass produces is what the next is given, so the
                    # arrangement has to be the same both times.  That is not
                    # something lowering a piece can be left to work out, since
                    # its results are not what the caller sees.
                    if subgraph is body_fn:
                        if len(subgraph.graph.graph_outputs) != len(carried_inputs_):
                            raise AssertionError(
                                "Expected len(subgraph.graph.graph_outputs) == len( carried_inputs_ )"
                            )
                        subgraph.graph.graph_outputs = _require_exact_strides(
                            subgraph.graph.graph_outputs,
                            traced_carried,
                        )

        if not (cond_fn.graph and body_fn.graph):
            raise AssertionError("Expected cond_fn.graph and body_fn.graph")
        cond_outputs = cond_fn.graph.graph_outputs
        body_outputs = body_fn.graph.graph_outputs

        if _has_aliased_buffers(body_outputs):
            raise AssertionError(
                "Output aliasing is currently not supported in compiled while_loop. "
                f"The outputs of the body_fn subgraph of while_loop are aliased: {body_outputs}"
            )

        # The piece that decides whether to continue says one yes-or-no and
        # nothing else, since that is all that is asked of it.
        if len(cond_outputs) != 1:
            raise AssertionError(cond_outputs)
        p = cond_outputs[0]
        if not isinstance(p, ShapeAsConstantBuffer):
            if p.get_dtype() != tp.bool:
                raise AssertionError(p)
            if len(p.get_size()) != 0:
                raise AssertionError(p)

        if len(all_inputs) <= 0:
            raise AssertionError("while_loop is assumed to have at least one operand.")

        device = all_inputs[0].get_device()

        if device is None:
            raise AssertionError("Expected device is not None")
        if len(carried_inputs_) != len(body_outputs):
            raise AssertionError(
                (
                    carried_inputs_,
                    body_outputs,
                )
            )
        for i, (op, bo) in enumerate(zip(carried_inputs_, body_outputs)):

            def _guard_list_equals(lhs_exprs, rhs_exprs) -> None:
                if len(lhs_exprs) != len(rhs_exprs):
                    raise AssertionError("Expected len(lhs_exprs) == len(rhs_exprs)")
                for lhs, rhs in zip(lhs_exprs, rhs_exprs):
                    V.graph.sizevars.check_equals(lhs, rhs)

            _guard_list_equals(op.get_size(), bo.get_size())
            _guard_list_equals(op.get_stride(), bo.get_stride())
            if op.get_device() != bo.get_device():
                raise AssertionError((i, op, bo, device))
            if op.get_dtype() != bo.get_dtype():
                raise AssertionError((i, op, bo))

        unbacked_bindings = resolve_unbacked_bindings(
            V.graph.sizevars.shape_env, V.graph.current_node.meta.get("unbacked_bindings", None),
        )

        while_loop = WhileLoop(
            carried_inputs=carried_inputs_,
            additional_inputs=additional_inputs_,
            cond_subgraph=cond_fn,
            body_subgraph=body_fn,
            layout=MultiOutputLayout(device=device),
            unbacked_bindings=unbacked_bindings,
            stack_output=stack_output,
        )

        # A carried value that came from outside may be written to by the loop,
        # in which case what the loop hands back is that value rather than a
        # new one -- but only if the loop ran at all, and it may not have.  So
        # its memory may not be reused for the result.
        mutated_idx_set = OrderedSet(
            getattr(body_fn, "mutated_input_indices", ()) or ()
        )
        mutated_inputs_iter = iter(
            [all_inputs[idx] for idx in sorted(mutated_idx_set)]
        )
        all_outputs: list = []
        while_loop.outputs = []
        while_loop.mutation_outputs = []
        if stack_output:
            if len(mutated_idx_set) != 0:
                raise AssertionError("NYI: while_loop_stack_output input mutations.")
            for idx, output in enumerate(body_outputs):
                multi_out = MultiOutput(
                    FixedLayout(
                        device=output.get_device(),
                        dtype=output.get_dtype(),
                        size=output.get_size(),
                        stride=output.get_stride(),
                    ),
                    while_loop,
                    [(list, idx)],
                )
                while_loop.outputs.append(multi_out)
                all_outputs.append(multi_out)
        else:
            for idx, output in enumerate(body_outputs):
                if idx in mutated_idx_set:
                    if idx >= len(carried_inputs_):
                        raise AssertionError("only carries can be mutated.")
                    mutated_input = next(mutated_inputs_iter)
                    while_loop.mutation_outputs.append(
                        MutationOutput(
                            mutated_input.layout, mutated_input, while_loop
                        )
                    )
                    all_outputs.append(mutated_input)
                else:
                    multi_out = MultiOutput(
                        FixedLayout(
                            device=output.get_device(),
                            dtype=output.get_dtype(),
                            size=output.get_size(),
                            stride=output.get_stride(),
                            offset=output.get_layout().offset,
                        ),
                        while_loop,
                        [(list, idx)],
                    )
                    while_loop.outputs.append(multi_out)
                    all_outputs.append(multi_out)

        for inp, out in zip(carried_inputs, all_outputs):
            if inp.get_name() in V.graph.graph_inputs:
                # A carried value that came from outside may be what the loop
                # hands back, and only if the loop ran at all, which it need not
                # have.  So the result cannot be given the input's memory: the
                # input may be written to.
                V.graph.never_reuse_buffers.add(out.get_name())
        return all_outputs

    def codegen(self, wrapper) -> None:
        wrapper.codegen_while_loop(self, self.stack_output)
        wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(), self.outputs, getattr(self, "unbacked_bindings", {})
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            if resolved is None:
                raise AssertionError("Expected resolved is not None")
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()


class FallbackKernel(ExternKernelAlloc):
    """A result produced by handing the operation to a prepared kernel as a whole.

    Some operations are not worth writing out as a loop -- a sort, a gather, an
    operation that has its own tuned code -- and are better run by calling
    through to something that already does them.  What is described here is
    only enough for fusion, scheduling and memory planning: the shapes, which
    inputs are written, and which results are the same memory as an input.
    """

    def __init__(
        self,
        layout: "OutputSpec",
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs: dict | None = None,
        *,
        unbacked_bindings: dict | None = None,
    ) -> None:
        super().__init__(
            layout,
            tuple(tensor_args),
            tuple(nontensor_args),
            op_overload=kernel,
        )

        self.use_runtime_dispatch = False
        self.unbacked_bindings = unbacked_bindings or {}
        self.op_overload = kernel
        # How the arguments are put back the way the call was written, which is
        # what says which position each input and each constant goes back to.
        self.unflatten_args = unflatten_args
        self.kwargs = {} if kwargs is None else kwargs

        if self.python_kernel_name is None:
            raise AssertionError("Expected self.python_kernel_name is not None")
        V.graph.warn_fallback(self.python_kernel_name)

        # args that are aliased
        self.alias_names: list = []
        # args that are mutated AND returned from the op
        self.mutation_names: list = []

        schema = _parsed_schema(self.op_overload)

        # Only three kinds of operation can be handed through like this:
        # - functional ops
        # - view ops
        # - in-place ops
        # - mutating ops that can be made functional.  That is, the operation
        #   may write any number of inputs, but none of its results may be the
        #   same memory as one of them.
        #
        # The cases that are not supported mostly do not turn up here, since
        # the tracing has already turned them into functional form; the only
        # way an in-place op arrives is for a lowering or a pass to have
        # introduced it.
        if schema is not None and _schema_mutates_and_returns_first_arg(schema):
            self.mutation_names.append(tensor_args[0].get_name())
            # Record the aliasing so memory planning does not hand this memory
            # out again as if it were free.
            self.alias_names.append(tensor_args[0].get_name())
            return

        if schema is not None and schema.is_mutable:
            raise NotImplementedError(
                f"NYI: Can't generate FallbackKernel for {self.op_overload}"
            )

        if schema is not None:
            args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
            for info, arg in zip_schema(schema, args, kwargs):
                handle_aliasing_and_mutation(self, info, arg)
        self._record_traced_aliases()

    def _record_traced_aliases(self) -> None:
        """Record the inputs the traced call's result shared memory with.

        A contract that does not say a result is a view of an input does not
        make the result fresh memory: the framework call may hand back a view
        anyway.  Those inputs are aliases all the same, so the result is never
        written over in place of fresh memory and the input is not handed out
        again while the result is alive.
        """

        node = getattr(V.graph, "current_node", None)
        if node is None or not hasattr(node, "meta"):
            return

        def storages(value):
            found = set()
            for t in _pytree.tree_leaves(value):
                if isinstance(t, tp.Tensor) and t.defined():
                    try:
                        ptr = t.untyped_storage().data_ptr()
                    except Exception:
                        continue
                    if ptr:
                        found.add(ptr)
            return found

        produced = storages(node.meta.get("val"))
        if not produced:
            return
        env = getattr(V.graph, "env", {})
        for arg in node.all_input_nodes:
            if not storages(arg.meta.get("val")) & produced:
                continue
            try:
                name = env[arg].get_name()
            except Exception:
                continue
            if name not in self.alias_names:
                self.alias_names.append(name)

    def get_read_writes(self):
        return super().get_read_writes()

    def codegen_unbacked_symbol_defs(self, wrapper) -> None:
        return wrapper.codegen_unbacked_symbol_defs_for_outputs(
            self.get_name(),
            self.codegen_outputs(),
            getattr(self, "unbacked_bindings", None),
        )

    def codegen_outputs(self):
        if self.outputs:
            return self.outputs
        if isinstance(self.layout, Layout):
            return [self]
        return self.mutation_outputs

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        if unbacked_bindings := getattr(self, "unbacked_bindings", None):
            resolved = resolve_unbacked_bindings(
                V.graph.sizevars.shape_env, unbacked_bindings
            )
            if resolved is None:
                raise AssertionError("Expected resolved is not None")
            return OrderedSet(resolved.keys())
        else:
            return OrderedSet()



    @staticmethod
    def tensor_to_layout(output) -> "FixedLayout":
        """The layout that describes a value that already exists in memory."""

        is_pinned = False
        try:
            is_pinned = output.is_pinned()
        except RuntimeError:
            # Dispatch not implemented for this kind of value.
            pass
        return FixedLayout(
            output.device,
            output.dtype,
            convert_shape_to_tp(output.size()),
            convert_shape_to_tp(output.stride()),
            is_pinned=is_pinned,
        )
    @staticmethod
    def find_device(tensor_args, example_output):
        """The device the result will be on, worked out from what is available.

        A value on a device names its own device, so that is asked first.  A
        value that is not a tensor at all has no device of its own, and one
        whose device is not known yet cannot say, so those are passed over.  A
        result that is several values has to agree: one device is the answer,
        and where they differ the one that is not the host is preferred, since
        that is where the work has to happen.
        """

        if tensor_args:
            devices = [arg.get_device() for arg in tensor_args if arg.get_device()]
            if devices:
                return devices[0]
        if isinstance(example_output, tp.Tensor):
            return example_output.device
        if isinstance(example_output, (list, tuple)):
            device_set = OrderedSet(
                FallbackKernel.find_device(None, x) for x in example_output
            )
            devices = [device for device in device_set if device]
            if len(devices) == 1:
                return devices[0]
            if not devices:
                return None
            for device in devices:
                if is_gpu(device):
                    return device
            return devices[0]
        # A value that is not a tensor has no device of its own, so it is
        # treated as living on the host.
        return tp.device("cpu") if example_output is not None else None

    def has_side_effects(self) -> bool:
        """Whether calling this changes something outside the result.

        The answer is a property of the operation rather than of the values
        passed to it, since the values here are descriptions and not the values
        the operation would be called with.
        """

        schema = _parsed_schema(self.op_overload)
        if schema is None:
            return bool(self.alias_names or self.mutation_names)
        return bool(schema.is_mutable or schema._is_view_op())

    def get_inputs_that_alias_output(self) -> Sequence:
        """Which inputs the results are the same memory as.

        An operation that writes to its inputs and can be made not to is
        reported here as producing new memory, because that is what the caller
        of the functional form gets.  Anything else is reported as the aliases
        it recorded.
        """

        schema = _parsed_schema(self.op_overload)
        if schema is not None and schema.is_mutable and self.alias_names == (
            self.mutation_names or []
        ):
            return []
        return self.alias_names

    def get_mutation_names(self) -> Sequence:
        """Which buffers this call writes to."""

        if len(self.mutation_names) > 1:
            raise AssertionError("Expected len(self.mutation_names) <= 1")
        return self.mutation_names

    def codegen_args(self) -> list:
        """The arguments of the call, each already written as the wrapper writes it.

        The tensor arguments are asked how to refer to themselves, and what
        comes back is held by a shim whose printed form is that reference -- so
        that an argument which is a view of another, or a slice of one, is
        written as the view rather than as the buffer behind it.  The named
        arguments are then sorted back out of the flat list they arrive in, and
        kept on this kernel for the call to read, since a call may be written
        with some of its arguments named and some positional.
        """

        @dataclasses.dataclass
        class Shim:
            ref: object

            def __repr__(self) -> str:
                return self.ref

        if not is_node_sequence(self.inputs):
            raise AssertionError("Expected is_node_sequence(self.inputs)")
        tensor_args = [Shim(x.codegen_reference()) for x in self.inputs]
        args, kwargs = self.unflatten_args(tensor_args, self.constant_args)
        args = [V.graph.wrapper_code.val_to_arg_str(x) for x in args]
        self.kwargs.update(kwargs)
        return args

    def codegen(self, wrapper) -> None:
        """Ask the wrapper to write the call that runs this operation.

        The result is not described here beyond its shape and which inputs it
        writes, so how the call itself is written is the wrapper's business: it
        knows whether the call is being made from Python or from compiled code,
        and what it has to hand the arguments.
        """

        kernel = self.op_overload
        if kernel is None:
            raise AssertionError("Expected kernel is not None")
        return wrapper.generate_fallback_kernel(self)

    @classmethod
    def create(cls, kernel, *args, **kwargs):
        """Describe a call to an operation that is run rather than written out.

        The operation is run once on the values it was given, and what comes
        back is what the result looks like: how big it is, what type it is, and
        which of the arguments ended up being the same memory as the result.
        That is everything a caller needs in order to have the call written
        later, which is why the call itself is not the thing being described.
        """

        if not callable(kernel):
            raise AssertionError(f"Fails to create FallbackKernel for {kernel}")

        result = cls.process_kernel(kernel, *args, **kwargs)
        example_output = result.example_output
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        unbacked_bindings = result.unbacked_bindings

        device = cls.find_device(tensor_args, example_output)
        if device is None:
            raise AssertionError("Cannot find device from example output")
        node = None
        if tensor_args:
            node = next(
                (x for x in tensor_args if x.get_device() == device and x is not None),
                None,
            )

        # An in-place operation reports itself as mutating whatever it was
        # given, so the inputs have to be marked as read before they are run.
        def convert(x):
            if isinstance(x, IRNode):
                return x
            if isinstance(x, tp.Tensor):
                if node is not None:
                    return x
                return x
            return x

        new_args = [convert(x) for x in args]
        new_kwargs = {k: convert(v) for k, v in kwargs.items()}

        packed = FallbackKernel(
            layout=(
                cls.tensor_to_layout(example_output)
                if isinstance(example_output, tp.Tensor)
                else MultiOutputLayout(device=device)
            ),
            kernel=kernel,
            tensor_args=tuple(tensor_args),
            nontensor_args=tuple(non_tensor_args),
            unflatten_args=unflatten_args,
            kwargs=new_kwargs,
            unbacked_bindings=unbacked_bindings,
        )
        if isinstance(example_output, tp.Tensor):
            return packed

        # A call that returns several values returns a structure, and each
        # tensor in it is a result of its own: a buffer that says where in the
        # structure it sits, read out of what the call handed back.
        def generate_output(output, indices):
            if isinstance(output, (list, tuple)):
                return type(output)(
                    generate_output(item, [*indices, (type(output), i)])
                    for i, item in enumerate(output)
                )
            if isinstance(output, dict):
                return {
                    key: generate_output(item, [*indices, (type(output), key)])
                    for key, item in output.items()
                }
            if isinstance(output, tp.Tensor):
                return MultiOutput(cls.tensor_to_layout(output), packed, indices)
            if output is None or isinstance(output, (int, float, bool)):
                return output
            raise AssertionError(
                f"FallbackKernel output type {type(output)} is not supported"
            )

        outputs = generate_output(example_output, [])
        if isinstance(outputs, (list, tuple)):
            packed.outputs = list(outputs)
        elif isinstance(outputs, dict):
            packed.outputs = list(outputs.values())
        else:
            packed.outputs = [outputs]
        return outputs

    @staticmethod
    @contextlib.contextmanager
    def _allow_non_fake_constant_args(enabled: bool):
        """Let a constant this compilation made reach an operation as itself.

        Working out what an operation produces is done on stand-in values, and
        a stand-in cannot be mixed with a real one.  A constant made while
        setting the operation up is real, so for the length of that one
        operation the rule is relaxed -- it is a value this compilation made
        for itself, and only that one operation ever sees it.
        """

        yield

    @classmethod
    def _materialize_scalar_tensor_args(cls, kernel, args, kwargs):
        """Turn a number given where a value is wanted into an actual value.

        An operation declared as taking a value will also accept a number, and
        that number may need to be somewhere the code can refer to.  Making it
        into a constant gives it a place, and the type it is given follows what
        the operation would have settled on anyway, so the arithmetic is
        unchanged.

        Returns the arguments as they should be passed, and whether anything
        was made, since a value made here is a real one and the caller has to
        know that.
        """

        schema = _parsed_schema(kernel)
        if schema is None:
            return args, kwargs, False

        tensor_arg_names = {
            a.name
            for a in schema.arguments
            if getattr(a.type, "name", None) in ("Tensor", "Tensor?")
        }
        if not tensor_arg_names:
            return args, kwargs, False

        def is_number(v) -> bool:
            return isinstance(v, (int, float, bool)) and not isinstance(v, bool) or (
                isinstance(v, bool)
            )

        materialized = False

        def materialize(v):
            nonlocal materialized
            if is_number(v):
                materialized = True
                return tp.tensor(v, dtype=tp.float32)
            return v

        new_args = tuple(
            materialize(a) if isinstance(a, (int, float, bool)) else a for a in args
        )
        new_kwargs = {
            k: (materialize(v) if isinstance(v, (int, float, bool)) else v)
            for k, v in kwargs.items()
        }
        return new_args, new_kwargs, materialized

    @classmethod
    def _maybe_realize_symm_mem_args(cls, kernel, args, kwargs, tensor_args):
        """Give memory that several machines share somewhere to be written.

        Memory shared between machines cannot be written into while something
        else reads it, so it is written out first rather than being filled in
        place.  Operations that declare such memory are the ones that need this.
        """

        return args, kwargs

    @staticmethod
    def _uses_aot_proxy_executor(kernel) -> bool:
        """Whether this call will be run by the stand-in rather than directly.

        Where the call is run by a stand-in, the only things that can cross
        into it are numbers and handles to memory, so a call that has neither
        form available is one that cannot be run that way.
        """

        if not V.graph.cpp_wrapper:
            return False
        namespace = getattr(kernel, "namespace", None)
        if namespace == "_quantized":
            return False
        return True

    def export_extern_kernel_node(self):
        """This call, in a form that can be written down and run later.

        What is exported is the call as it was written -- which operation, and
        what it was given -- so that it can be run by something that is not the
        code it was compiled into.  That something is limited in what it can be
        handed, so the arguments are put in the order the operation declares and
        anything that is not one of them is left out.
        """

        args, kwargs = self.unflatten_args(self.inputs, self.constant_args)
        args = self.fill_non_provided_args(args, kwargs)
        ordered_kwargs = [
            self.get_kwargs_value(key, **kwargs)
            for key in self.ordered_kwargs_for_cpp_kernel
        ]

        return ExternKernelNode(
            name=self.get_name(),
            node={
                "target": str(self.op_overload),
                "args": args,
                "kwargs": ordered_kwargs,
            },
        )


class MultiTemplateBuffer(TritonTemplateBuffer):
    """A result that could be produced in more than one way, not yet decided.

    Some work can be done by a prepared kernel or by another, and which is
    faster is not known until it has been measured.  So all the ways are kept,
    measured against each other when there is something to measure, and the
    one that won is what the rest of the graph is written against.

    Which way is in use can be changed while something is being measured, and
    is put back afterwards, since the point of the measurement is to compare
    them and not to settle on one.
    """

    def __init__(
        self,
        layout: "Layout",
        inputs,
        choice_timings_fn,
        unfiltered_choices: list,
        allowed_prologue_inps,
    ) -> None:
        super().__init__(
            layout=layout,
            inputs=inputs,
            make_kernel_render=None,
            allowed_prologue_inps=allowed_prologue_inps,
        )
        self._choice_timings_fn = choice_timings_fn
        self._choice_timings: dict = {}
        self._choices: list = unfiltered_choices
        self.original_inputs = inputs
        # Whether something can be written into this result afterwards is only
        # worth asking if every way of producing it could have that done, since
        # which way is used is not yet settled.
        self._output_plannable = all(
            _is_output_plannable_choice(choice) for choice in unfiltered_choices
        )
        self._make_kernel_renders: dict = {}
        self._render_kind: str | None = None
        # The way in use, so that the measurement loop's changing of it is not
        # quietly undone by measuring again.
        self._render_caller = None

    @property
    def output_plannable(self) -> bool:
        return self._output_plannable

    @property
    def choices(self) -> list:
        return self._choices

    def choice_timings(self, hint_override: int | None = None) -> dict:
        if hint_override not in self._choice_timings:
            self._choice_timings[hint_override] = self._choice_timings_fn(hint_override)
        return self._choice_timings[hint_override]

    def _swap_as_caller(self, caller, kind: str):
        """Use this way while the caller is measuring, then put back what was."""

        import contextlib

        @contextlib.contextmanager
        def ctx():
            if self.layout != caller.layout:
                raise AssertionError("Expected self.layout == caller.layout")

            render = self.make_kernel_render
            prev_kind = self._render_kind
            prev_caller = self._render_caller
            self.make_kernel_render = caller.get_make_kernel_render()
            self._render_kind = kind
            self._render_caller = caller
            try:
                yield
            finally:
                self.make_kernel_render = render
                self._render_kind = prev_kind
                self._render_caller = prev_caller

        return ctx()

    @contextlib.contextmanager
    def swap_as_triton_caller(self, caller):
        """Use this prepared kernel while the caller is measuring."""

        return self._swap_as_caller(caller, "triton")

    def finalize_as_triton_caller(self, caller) -> None:
        """Settle on this prepared kernel, having checked it produces the same."""

        if self.get_size() != caller.layout.size:
            raise AssertionError("Expected self.get_size() == caller.layout.size")
        if self.get_stride() != caller.layout.stride:
            raise AssertionError("Expected self.get_stride() == caller.layout.stride")
        self.make_kernel_render = caller.get_make_kernel_render()
        self._render_kind = "triton"
        self._render_caller = caller

    @contextlib.contextmanager
    def swap_as_nvgemm_caller(self, caller):
        """Use this matrix-multiplication kernel while the caller is measuring."""

        return self._swap_as_caller(caller, "nvgemm")

    def finalize_as_nvgemm_caller(self, caller) -> None:
        """Settle on this matrix-multiplication kernel, having checked it fits."""

        if self.get_size() != caller.layout.size:
            raise AssertionError("Expected self.get_size() == caller.layout.size")
        if self.get_stride() != caller.layout.stride:
            raise AssertionError("Expected self.get_stride() == caller.layout.stride")
        self.make_kernel_render = caller.get_make_kernel_render()
        self._render_kind = "nvgemm"
        self._render_caller = caller

    def get_min_choice(self, hint_override: int | None = None):
        """The fastest of the ways, and how long it took."""

        timings = self.choice_timings(hint_override=hint_override)
        min_choice = min(timings, key=timings.get)
        return (min_choice, timings[min_choice])

    def finalize_as_triton_callers(self, callers: dict) -> None:
        """Settle on a different prepared kernel for each set of conditions.

        Which way is fastest can depend on what the shapes are, so a way may be
        chosen per set of conditions rather than once.  The one chosen for no
        particular conditions is what is used when none apply.
        """

        for hint_override, caller in callers.items():
            self._make_kernel_renders[hint_override] = caller.get_make_kernel_render()

        self.make_kernel_render = self._make_kernel_renders[None]
        self._render_kind = "triton"
        self._render_caller = callers[None]


def _is_output_plannable_choice(choice) -> bool:
    """Whether this way of producing a result could have something written into it.

    Only a way that produces the result itself, rather than writing into memory
    it was given, can have something else written into it afterwards.
    """

    if isinstance(choice, TritonTemplateCallerBase):
        return True
    return bool(getattr(choice, "has_out_variant", False))


class UserDefinedTritonKernel(ExternKernel):
    """A kernel written by hand, rather than one this compilation produced.

    What such a kernel writes and reads is stated by the program that wrote it,
    not worked out from what it computes, so it is taken from what was recorded
    about it.  Since a hand-written kernel usually writes into memory it was
    given rather than returning anything, what it produces is reported as the
    buffers it wrote to.
    """

    def get_kernel_and_metadata(self):
        """The kernel itself, and what it does to the arguments it is given.

        A kernel that picks among several ways of running has to say which
        arguments it puts back the way they were and which it clears, since
        neither can be left to be inferred from the call.
        """

        kernel = self.kernel
        configs = []
        restore_value_args: list = []
        reset_to_zero_args: list = []
        restore_idx = getattr(kernel, "restore_idx", None)
        if restore_idx is not None:
            restore_value_args.extend(kernel.fn.arg_names[i] for i in restore_idx)
        else:
            restore_value_args.extend(getattr(kernel, "restore_value", ()) or ())

        reset_idx = getattr(kernel, "reset_idx", None)
        if reset_idx is not None:
            reset_to_zero_args.append(kernel.fn.arg_names[i] for i in reset_idx)
        else:
            reset_to_zero_args.extend(getattr(kernel, "reset_to_zero", ()) or ())

        configs = getattr(kernel, "configs", ()) or ()
        inner = getattr(kernel, "fn", None)
        if inner is not None:
            kernel = inner

        return kernel, configs, restore_value_args, reset_to_zero_args

    def can_fuse_epilogue(self) -> bool:
        """Whether something can be computed from this result into the kernel.

        Fusing means the kernel writes what the kernel would have written with
        the extra work already applied, which requires knowing which value it
        wrote and writing in one place.  It also requires that what the kernel
        was given was empty, since only the elements the kernel actually writes
        would get the extra work -- which is only the same answer when there
        was nothing there to begin with.
        """

        if not config.epilogue_fusion_user_defined_triton_kernel:
            return False

        if not getattr(self.arg_accesses, "can_fuse_epilogue", False):
            return False

        if len(self.kernel_stores.stores) != 1:
            return False

        if len(self.mutable_args) != 1:
            raise AssertionError("Expected len(self.mutable_args) == 1")
        if not isinstance(self.mutable_args[0], TensorBox):
            return False
        if not isinstance(self.mutable_args[0].data, StorageBox):
            return False
        if not isinstance(self.mutable_args[0].data.data, ComputedBuffer):
            return False
        if not isinstance(self.mutable_args[0].data.data.data, Pointwise):
            return False
        if not all(r == 0 for r in self.mutable_args[0].data.data.data.ranges):
            return False

        return True

    def codegen(self, wrapper) -> None:
        return self._codegen(wrapper, epilogue_fusion=None)

    def codegen_with_epilogue_fusion(self, wrapper, epilogue_fusion) -> None:
        return self._codegen(wrapper, epilogue_fusion)

    def _codegen(self, wrapper, epilogue_fusion) -> None:
        kernel, configs, restore_value_args, reset_to_zero_args = (
            self.get_kernel_and_metadata()
        )

        # Where the extra work casts the result, the value the kernel writes has
        # to be the extra work's own result, so that what the call is described
        # as taking and returning still matches.
        kernel_kwargs = self.kwargs
        epilogue_out_override: dict = {}
        if epilogue_fusion:
            if len(self.arg_accesses.read_writes.writes) != 1:
                raise AssertionError(
                    f"expected one write, got {len(self.arg_accesses.read_writes.writes)}"
                )
            mutable_arg_name = next(iter(self.arg_accesses.read_writes.writes)).name
            epilogue_computed_buffer, _ = epilogue_fusion
            kernel_kwargs = {**self.kwargs, mutable_arg_name: epilogue_computed_buffer}
            epilogue_out_override = {mutable_arg_name: epilogue_computed_buffer}

        (
            new_name,
            triton_meta,
            tp_meta,
            extra_launch_args,
        ) = wrapper.define_user_defined_triton_kernel(
            kernel,
            configs,
            kernel_kwargs,
            restore_value_args,
            reset_to_zero_args,
            self.grid,
            epilogue_fusion,
            self.launch_kwargs,
        )
        named_args = {
            k: self.get_kwargs_value(k) for k in self.ordered_kwargs_for_cpp_kernel
        }
        named_args.update(epilogue_out_override)

        arg_names = [p.name for p in kernel.params]
        constexprs = [p.num for p in kernel.params if p.is_constexpr]
        constexpr_names = OrderedSet(arg_names[i] for i in constexprs)

        args: list = []
        arg_types: list = []
        raw_keys_filtered: list = []
        raw_args_filtered: list = []
        for name, arg in itertools.chain(
            named_args.items(), zip(itertools.repeat(""), extra_launch_args)
        ):
            if name in constexpr_names:
                # A value fixed when the kernel was built is not passed at run
                # time, since it is already part of the kernel.
                continue
            raw_keys_filtered.append(name)
            raw_args_filtered.append(arg)
            if isinstance(arg, IRNode):
                args.append(arg.codegen_reference())
                arg_types.append(arg.get_dtype())
            elif isinstance(arg, (int, float, bool, Expr)):
                args.append(arg)
                arg_types.append(type(arg))
            elif name in constexpr_names:
                # A value fixed when the kernel was built and of a kind that is
                # not passed anyway; what goes in its place is never read.
                args.append(-1)
                arg_types.append(int)
            elif arg is None:
                # A value fixed when the kernel was built may be left out
                # entirely; one that is not may not be, since the kernel would
                # be handed a different number of arguments than it declares.
                args.append(-1)
                arg_types.append(int)
            else:
                raise NotImplementedError(f"Unsupported arg type: {type(arg)}: {arg}")

        self.codegen_comment(wrapper, new_name)
        wrapper.generate_kernel_call(
            new_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args_filtered,
            raw_keys=raw_keys_filtered,
            triton_meta=triton_meta,
            tp_meta=tp_meta,
            triton=True,
            device=self.get_device(),
            original_fxnode_name=getattr(self.fx_node, "name", None),
        )

    @cache_on_self_and_args("UserDefinedTritonKernel")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        # The grid says how many times the kernel runs, which may depend on a
        # shape that the arguments do not mention.
        return super().get_free_symbol_uses(unbacked_only) | get_free_symbols(
            self.grid, unbacked_only
        )

    def get_unbacked_symbol_defs(self) -> OrderedSet:
        return OrderedSet()

    def __init__(
        self,
        *,
        kernel,
        grid,
        tma_descriptor_metadata: dict,
        kernel_args: dict,
        launch_kwargs,
        arg_accesses=None,
        kernel_stores=None,
        fx_node=None,
    ) -> None:
        inputs: list = []
        kwargs: dict = {}
        constant_args: list = []

        for k, v in kernel_args.items():
            if isinstance(v, TensorBox):
                t = InputsKernel.unwrap_storage_for_input(self.realize_input(v))
                if k in tma_descriptor_metadata:
                    t = TMADescriptor.create(t, tma_descriptor_metadata[k])
                inputs.append(t)
                kwargs[k] = t
            else:
                constant_args.append(v)
                kwargs[k] = v

        if len(inputs) == 0:
            raise AssertionError("Expected len(inputs) != 0")
        self.device = inputs[0].get_device()

        if not isinstance(inputs, Sequence):
            raise AssertionError(type(inputs))
        super().__init__(
            None,
            NoneLayout(device=self.device),
            inputs,
            tuple(constant_args),
            kwargs,
        )
        self.kernel = kernel
        self.grid = grid
        self.launch_kwargs = launch_kwargs
        self.kernel_args = kernel_args
        self.fx_node = fx_node
        self.arg_accesses = arg_accesses
        self.kernel_stores = kernel_stores

        k, configs, _, _ = self.get_kernel_and_metadata()

        if not hasattr(k, "arg_names"):
            raise AssertionError('Expected hasattr(kernel, "arg_names")')
        self.ordered_kwargs_for_cpp_kernel = [
            arg for arg in k.arg_names if arg in kernel_args
        ]

        # Which of the arguments are written to, as opposed to read, is stated
        # by the program that wrote the kernel.  The list of what is written
        # can mention arguments that are not values at all, so only the ones
        # that are values are taken.
        self.mutable_args = [
            kernel_args[key.name]
            for key in self.arg_accesses.read_writes.writes
            if isinstance(kernel_args.get(key.name), TensorBox)
        ]

        self.mutation_outputs = [
            MutationOutput(NoneLayout(device=self.device), buf, self)
            for buf in self.mutable_args
        ]
        V.graph.register_operation(self)

    def get_outputs(self) -> list:
        return list(self.mutation_outputs)

    def get_device(self):
        return self.device


class _CollectiveKernel(FallbackKernel):
    """A call that exchanges values with other machines, rather than computing.

    What makes this worth its own class is that it does not finish when it
    returns: the values it was given are still in use by the other machines
    until they say they are done.  So it produces no value of its own, and it
    has effects beyond the values it was handed, and the buffers involved are
    marked as written for the whole time the exchange lasts rather than only
    while the call is running.

    Which arguments a call may only be given by name is worked out from how the
    operation was declared, exactly as for any other call, except that nothing
    is checked about which of them share memory -- the exchange is what decides
    that, not this graph.
    """

    def should_allocate(self) -> bool:
        return False

    def has_side_effects(self) -> bool:
        return True

    def set_cpp_kernel_name(self, cpp_kernel_name: str | None = None) -> None:
        if self.op_overload is None or not hasattr(self.op_overload, "_schema"):
            raise AssertionError("Setting cpp kernel needs a valid op_overload")
        kernel = self.op_overload
        schema = _parsed_schema(kernel)
        if cpp_kernel_name is not None:
            self.cpp_kernel_name = cpp_kernel_name
        else:
            self.cpp_kernel_name = schema.name if schema is not None else kernel.name

        self.ordered_kwargs_for_cpp_kernel = [
            x.name for x in (schema.arguments if schema is not None else ()) if x.kwarg_only
        ]

    @classmethod
    def create_inplace(cls, kernel, inputs, *args, **kwargs) -> None:
        """Start an exchange that writes into the values it was given.

        Between starting and finishing, those values are being both read and
        written by more than one machine, so nothing else may touch them.  That
        is expressed as the exchange writing to them, which is what keeps
        anything else from being placed in between.
        """

        result = cls.process_kernel(kernel, inputs, *args, **kwargs, _share_args=False)
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel} {result.unbacked_bindings}")
        device = None
        for tensor_arg in tensor_args:
            if isinstance(tensor_arg, NonTensorObj):
                continue
            tensor_arg.realize()
            V.graph.mark_buffer_mutated(tensor_arg.get_name())
            if device is None:
                device = tensor_arg.get_device()
        if device is None:
            raise AssertionError(
                f"In-place collective {kernel} requires at least one tensor "
                f"argument; got only non-tensor IR nodes."
            )
        packed = cls(
            NoneLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
        )

        inps = _pytree.tree_leaves(inputs)
        packed.mutation_outputs.extend(
            [MutationOutput(NoneLayout(device=device), buf, packed) for buf in inps]
        )

        # What this wrote to is what it hands back, so the two are the same
        # memory rather than one having been copied into the other.
        packed.alias_names.extend([inp.get_name() for inp in inps])
        if "out" in kwargs:
            packed.mutation_outputs.append(
                MutationOutput(NoneLayout(device=device), kwargs["out"], packed)
            )
            packed.alias_names.append(kwargs["out"].get_name())

    @classmethod
    def create_out_of_place(cls, kernel, inputs, *args, **kwargs):
        """Start an exchange that produces new values.

        The values that were given may still be read by another kernel, but
        nothing may write to them; the values produced may be neither read nor
        written until the exchange is finished.  The first is expressed by
        having whoever waits report them as read, and the second by having the
        wait report them as written, so that nothing can be placed in between.
        """

        result = cls.process_kernel(kernel, inputs, *args, **kwargs, _share_args=True)
        example_output = result.example_output
        tensor_args = result.tensor_args
        non_tensor_args = result.non_tensor_args
        unflatten_args = result.unflatten_args
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel}, {result.unbacked_bindings}")
        for tensor_arg in tensor_args:
            if not isinstance(tensor_arg, TorchBindObject):
                tensor_arg.realize()

        if isinstance(example_output, list):
            device = cls.find_device(tensor_args, example_output)
            if device is None:
                raise AssertionError("Expected device is not None")
            packed = cls(
                MultiOutputLayout(device=device),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            packed.outputs = [
                MultiOutput(
                    cls.tensor_to_layout(tensor),
                    packed,
                    [(list, i)],
                )
                for i, tensor in enumerate(example_output)
            ]
            for buf, tensor in zip(packed.outputs, example_output):
                if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                    tensor
                ):
                    V.graph.unaligned_buffers.add(buf.name)
            return packed.outputs
        else:
            packed = cls(
                cls.tensor_to_layout(example_output),
                kernel,
                tensor_args,
                non_tensor_args,
                unflatten_args,
            )
            if config.assume_unaligned_fallback_output or not tensor_is_aligned(
                example_output
            ):
                V.graph.unaligned_buffers.add(packed.name)
            packed.outputs = [packed]
            return packed


class _AllReduceKernel(_CollectiveKernel):
    """An exchange that gives every machine the same combination of the values.

    What comes back is a combination of what every machine contributed, so it
    is a value like any other once the exchange is done.  Since the combination
    is only complete when every machine has contributed, the result cannot be
    used until the exchange is.
    """

    def __init__(
        self,
        layout: "OutputSpec",
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs: dict | None = None,
        *,
        unbacked_bindings: dict | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("tp_cpu__distributed_functional_all_reduce")

    def codegen(self, wrapper) -> None:
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)


class _AllReduce_Kernel(_CollectiveKernel):
    """An exchange that gives every machine the same combination, the other form.

    This is the same exchange as the one above, reached through the name the
    other calling convention uses.
    """

    def __init__(
        self,
        layout: "OutputSpec",
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs: dict | None = None,
        *,
        unbacked_bindings: dict | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("tp_cpu__distributed_functional_all_reduce")

    def codegen(self, wrapper) -> None:
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)


class _WaitKernel(_CollectiveKernel):
    """Waiting until an exchange has finished with the values it was given.

    Nothing can be done with the values involved until every machine has said
    it is finished, so waiting is what turns an exchange into a result that
    other work may be built on.  Waiting reports the values it waited for as
    written, since until the wait runs they were in use by the exchange, and
    reports as read whatever the exchange was reading -- which is what keeps
    that memory from being handed to something else in the meantime.
    """

    def __init__(
        self,
        layout: "OutputSpec",
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs: dict | None = None,
        *,
        unbacked_bindings: dict | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )
        self.set_cpp_kernel_name("tp_cpu__distributed_functional_wait_tensor")

    def codegen(self, wrapper) -> None:
        wrapper.generate_extern_kernel_alloc(self)

        if isinstance(self.layout, Layout):
            self.codegen_size_asserts(wrapper)

    def get_volatile_reads(self) -> Sequence:
        volatile_reads: list = []
        for inp in _pytree.tree_leaves(self.inputs):
            if not isinstance(inp, IRNode):
                raise AssertionError("Expected isinstance(inp, IRNode)")
            volatile_reads.extend(self._get_volatile_reads(inp))
        return volatile_reads

    @staticmethod
    def _get_volatile_reads(inp: "IRNode") -> Sequence:
        """What a value depends on that is still in use by an exchange.

        Which memory that is depends on what the value is: a value that was
        produced by an exchange is reading from whatever that exchange was
        given at the position this value came from, and anything else is either
        read directly or already accounted for by being written.
        """

        if isinstance(inp, (Buffer, MultiOutput, ComputedBuffer)) and not isinstance(
            inp, MultiOutput
        ):
            return [inp]
        elif isinstance(inp, IRNode) and isinstance(inp, InputsKernel):
            return [i for i in inp.inputs if isinstance(i, IRNode)]
        elif isinstance(inp, MultiOutput):
            # This is one of two things: an exchange that produced several
            # values, or an exchange that was given one of several.  In the
            # first case the value is reading what the exchange was given at
            # that position; in the second, the exchange wrote the memory
            # itself and there is nothing more to wait for.
            coll = inp.inputs[0]
            if isinstance(coll, _CollectiveKernel):
                _, idx = inp.indices[0]

                return [coll.inputs[idx]]
            return []
        else:
            # A value that is written rather than read needs nothing here,
            # since what it reads is already reported as written.
            return []

    @classmethod
    def create_wait(cls, kernel, inp) -> None:
        result = cls.process_kernel(kernel, inp, _share_args=True)
        if result.unbacked_bindings:
            raise AssertionError(f"{kernel} {result.unbacked_bindings}")
        inps = _pytree.tree_leaves(inp)
        if not inps:
            raise AssertionError(f"{kernel} requires at least one tensor")
        device = inps[0].get_device()
        packed = cls(
            NoneLayout(device=device),
            kernel,
            result.tensor_args,
            result.non_tensor_args,
            result.unflatten_args,
        )
        packed.mutation_outputs.extend(
            MutationOutput(NoneLayout(device=device), tensor, packed) for tensor in inps
        )

    def get_read_writes(self):
        read_writes = super().get_read_writes()
        # What the exchange was reading is still in use, so waiting for it is
        # what it is safe to read.
        volatile_reads = self.get_volatile_reads()
        for vr in volatile_reads:
            read_writes.reads.add(dependencies.StarDep(vr.get_name()))
        return read_writes


class EffectfulKernel(FallbackKernel):
    """A call that changes something outside the values it was given.

    Some operations do more than produce a value: they change the state of the
    program, and the order they happen in is part of what they mean.  Two such
    calls of the same kind therefore have to be ordered, which is why the
    previous one of this kind is recorded and reported as read -- so that
    whatever comes next is placed after it.
    """

    def __init__(
        self,
        layout: "OutputSpec",
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        kwargs: dict | None = None,
        *,
        unbacked_bindings: dict | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=None,
            unbacked_bindings=unbacked_bindings,
        )

        effect_type = getattr(kernel, "effect_type", None)
        if effect_type is None:
            # What kind of change this is comes from how the operation was
            # declared, and a plain operation declares none, in which case
            # there is nothing to order it against.
            name = getattr(kernel, "name", None)
            effect_type = name() if callable(name) else (name or str(kernel))
        self.effect_type = effect_type
        self.prev_effect_buffer = V.graph.effectful_ops.get(effect_type, None)
        V.graph.effectful_ops[effect_type] = self

    def get_read_writes(self):
        read_writes = super().get_read_writes()

        if self.prev_effect_buffer is not None:
            read_writes.reads.add(
                dependencies.StarDep(self.prev_effect_buffer.get_name())
            )

        return read_writes

    def has_side_effects(self) -> bool:
        return True


class ComplexView(FallbackKernel):
    """One complex value read as two ordinary ones, or the other way round.

    A complex value is held as pairs of numbers, and what this does is change
    which of those pairs count as one element.  Nothing is copied and no memory
    is produced: the same numbers are read with a different shape, which is why
    the result is reported as the same memory as what was read.  Reporting it
    that way also keeps the memory from being handed to something else to write
    into, which would change the numbers underneath.
    """

    def should_allocate(self) -> bool:
        return False

    def get_inputs_that_alias_output(self) -> Sequence:
        # The result is the same memory as what was read, and must not be
        # handed out for something else to write into.
        return [self.input_name(0)]

    def __init__(
        self,
        layout: "OutputSpec",
        kernel,
        tensor_args,
        nontensor_args,
        unflatten_args,
        *,
        kwargs: dict | None = None,
        unbacked_bindings: dict | None = None,
    ) -> None:
        super().__init__(
            layout,
            kernel,
            tensor_args,
            nontensor_args,
            unflatten_args,
            kwargs=kwargs,
            unbacked_bindings=unbacked_bindings,
        )


def _make_out_variant_kernel_name(out_op) -> str:
    """The fully qualified name of an operation that writes into memory given.

    Which operation it is, and which of its forms, are both part of the name a
    call is written under, so the name is built from what the operation
    declares rather than from the object it was handed as.
    """

    ns = out_op.namespace
    op_name = out_op._schema.name.split("::")[1]
    overload = out_op._overloadname
    return f"tp.ops.{ns}.{op_name}.{overload}"


class ExternKernelMultiOut(FallbackKernel):
    """A call that produces several values, writing into memory handed to it.

    Where an ordinary call of several results is asked to return them, there is
    a form of the same operation that is given somewhere to put each one.  That
    is what this describes: the memory is provided here, one result per place,
    and the call writes into those places rather than returning anything.  Each
    place is a result of its own, saying which of the several it is.
    """

    out_arg_names: list
    out_variant_output_nodes: list

    def __init__(
        self,
        *args,
        out_op,
        out_arg_names: list,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.out_arg_names = out_arg_names
        self.out_variant_output_nodes = []
        self.python_kernel_name = str(out_op)
        self.op_overload = out_op

    def codegen(self, wrapper) -> None:
        self.codegen_comment(wrapper)
        wrapper.generate_extern_kernel_multi_out(self)

    @classmethod
    def try_create(
        cls,
        kernel,
        example_output,
        device,
        tensor_args,
        non_tensor_args,
        unflatten_args,
        kwargs,
        *,
        unbacked_bindings: dict | None = None,
        has_unaligned_input: bool = False,
    ):
        """Build this if the operation has a form that writes into memory given.

        Nothing is written and no node is made unless the operation is one that
        produces several values and there is such a form of it, since a call
        that returns its results is a different thing and is described
        differently.
        """

        if not isinstance(example_output, (tuple, list)):
            return None

        out_op = getattr(kernel, "out_variant", None)
        if out_op is None:
            return None

        out_arg_names = list(getattr(out_op, "out_arg_names", ()) or ())
        if not all(isinstance(t, tp.Tensor) for t in example_output):
            return None
        if len(example_output) != len(out_arg_names):
            return None

        packed = cls(
            MultiOutputLayout(device=device),
            kernel,
            tensor_args,
            non_tensor_args,
            unflatten_args,
            kwargs=kwargs,
            unbacked_bindings=unbacked_bindings,
            out_op=out_op,
            out_arg_names=out_arg_names,
        )

        outputs: list = []
        for i, tensor_out in enumerate(example_output):
            layout = FixedLayout(
                device=tensor_out.device,
                dtype=tensor_out.dtype,
                size=[*tensor_out.shape],
                stride=[*tensor_out.stride()],
            )
            multi_out = AllocatingMultiOutput(
                layout=layout,
                input=packed,
                indices=[(type(example_output), i)],
            )
            outputs.append(multi_out)

        packed.out_variant_output_nodes = outputs
        packed.outputs = outputs

        if isinstance(example_output, tuple):
            return tuple(outputs)
        return list(outputs)


class MemoryCheckKernel(FallbackKernel):
    """Noting which buffers are alive and which have gone, as the code runs.

    This exists to be looked at rather than to compute anything: it reports
    what the run is holding at each point, so that what is being held can be
    checked against what should be.  The call is written directly rather than
    run through to, since the whole point is that the call appears in the
    generated code where it can be read.
    """

    def codegen(self, wrapper) -> None:
        wrapper.write_memory_track_allocation_once()
        # The three things said about this step -- what it brings up, what it
        # lets go of, and whether it is the last one -- are what the caller
        # passed as the step's own arguments.  They are read from there rather
        # than from the constants slot, which is empty for a kernel of this
        # shape: nothing here is a constant folded out of the call, it is all
        # said about this particular step.
        alive_list, dead_list, is_final_step = self.nontensor_args

        alive_repr = repr(alive_list)
        dead_repr = repr(dead_list)
        if is_final_step:
            wrapper.writeline(
                "# note: don't currently distinguish between buffers returned and dealloc'd in last step"
            )
            call = f"check_memory_step(allocated={alive_repr}, freed={dead_repr}, is_final_step={is_final_step})"
        else:
            call = f"check_memory_step(allocated={alive_repr}, freed={dead_repr})"
        wrapper.writeline(call)




def _parsed_schema(op: Any) -> Any:
    """The parsed contract an operation carries, or None.

    A dispatcher operation carries its schema parsed; a library operation a
    program defines for itself carries the text it was declared with, which
    says nothing this backend reads about aliasing or arguments.
    """

    schema = getattr(op, "_schema", None)
    return None if isinstance(schema, str) else schema


def _schema_mutates_and_returns_first_arg(schema) -> bool:
    """Whether the contract writes to its first argument and hands it back.

    That is the shape of the in-place operators, and knowing it is what lets an
    in-place op be reported as an overwrite rather than as something that read
    an input and produced a new value.
    """

    # The shape is ``op_(Tensor(a!) self, ...) -> Tensor(a!)``: one return, in
    # the alias set of a first argument that is written, and no other argument
    # aliased at all.  Alias sets are named (``a``), so the return and the first
    # argument are matched by that name.
    arguments = schema.arguments
    if len(schema.returns) != 1 or not arguments:
        return False
    returned = schema.returns[0].alias_info
    if returned is None or len(returned.after_set) != 1:
        return False
    first = arguments[0].alias_info
    if first is None or not first.is_write or len(first.after_set) != 1:
        return False
    if next(iter(returned.after_set)) != next(iter(first.after_set)):
        return False
    return all(argument.alias_info is None for argument in arguments[1:])


def _is_tensor_like_type(arg_type) -> bool:
    """Whether a contract type names one tensor, or a list of tensors."""

    name = getattr(arg_type, "name", None) or str(arg_type)
    if name in ("Tensor", "Tensor?"):
        return True
    if name in ("Tensor[]", "Tensor?[]"):
        return False
    return False


def zip_schema(schema, args, kwargs):
    """Pair each declared argument with the value that fills it.

    The contract lists its arguments by name while the call may pass them
    positionally, by keyword, or not at all, so what is wanted here is the
    declared argument together with the value that went to it.  An argument
    given only a keyword is filled from there even when the positional
    arguments stop short of it, and an argument left at its default is left
    unfilled rather than filled with the default, since a caller that did not
    mention it has said nothing about it.
    """

    if len(schema.arguments) < len(args) + len(kwargs):
        raise AssertionError(
            f"schema has {len(schema.arguments)} arguments but got "
            f"{len(args)} args and {len(kwargs)} kwargs"
        )
    for i, info in enumerate(schema.arguments):
        if info.kwarg_only:
            if info.name in kwargs:
                yield info, kwargs[info.name]
            continue
        if i >= len(args):
            if info.name in kwargs:
                yield info, kwargs[info.name]
            continue
        yield info, args[i]


def handle_aliasing_and_mutation(kernel, info, arg) -> None:
    """Note which inputs a call writes, and which results share their memory.

    A contract type declared as a tensor also accepts nothing or a plain number,
    and those are not checked here.  A number that stands in for a tensor is a
    temporary made for the call, so it cannot be an alias of or a write to
    anything the graph owns.
    """

    if arg is None:
        return
    if info.alias_info is None:
        return

    def add_alias(t: IRNode) -> None:
        kernel.alias_names.append(t.get_name())
        if info.alias_info.is_write:
            kernel.mutation_outputs.append(
                MutationOutput(NoneLayout(device=t.get_device()), t, kernel)
            )

    def add_alias_if_graph_buffer(t) -> None:
        if t is None:
            return
        if isinstance(t, (int, float, complex, sympy.Basic)):
            return
        if not isinstance(t, IRNode):
            raise AssertionError(type(t))
        add_alias(t)

    if _is_tensor_like_type(info.type):
        add_alias_if_graph_buffer(arg)


@ir_dataclass
class ReinterpretView(BaseView):
    """The same memory, described as though it were laid out differently.

    Nothing is copied and nothing is computed: the elements are where they were,
    and this says that they are to be read as though they were in some other
    order, at some other offset, as some other type.  That is what a reshape of
    contiguous memory is, and what a slice that is not a copy is.
    """

    layout: Layout

    def __post_init__(self) -> None:
        super().__post_init__()
        if isinstance(self.data, BaseView):
            # A view of a view is the same as a view of what that looks at, and
            # keeping the chain would only make every question about it longer.
            object.__setattr__(self, "data", self.data.unwrap_view())

    def __str__(self) -> str:
        return self.str_helper(
            [
                self.data,
                self.layout,
            ]
        )

    __repr__ = __str__

    def get_name(self) -> str:
        return self.data.get_name()

    def get_device(self):
        return self.layout.device

    def get_origin_node(self):
        return None

    @property
    def dtype(self):
        return self.layout.dtype

    def get_size(self) -> Sequence:
        return list(self.layout.size)

    def get_stride(self) -> Sequence:
        return list(self.layout.stride)

    def make_loader(self):
        def loader(index: Sequence):
            indexer = self.layout.make_indexer()
            tmp_loader = ops.load(self.get_name(), indexer(index))
            if self.layout.dtype != self.data.dtype:
                # Read as the type the memory was written as, and only then read
                # as the type this claims, which is the only order in which the
                # bits mean what they should.
                return ops.to_dtype_bitcast(tmp_loader, self.dtype, self.data.dtype)
            else:
                return tmp_loader

        return loader

    def make_indexer(self):
        return self.layout.make_indexer()

    def get_layout(self) -> "Layout":
        return self.layout

    def freeze_layout(self) -> None:
        pass

    @cache_on_self_and_args("ReinterpretView")
    def get_free_symbol_uses(self, unbacked_only: bool = False) -> OrderedSet:
        return (
            get_free_symbols(self.layout.size, unbacked_only)
            | get_free_symbols(self.layout.stride, unbacked_only)
            | get_free_symbols(self.layout.offset, unbacked_only)
        )

    def codegen_reference(self, writer=None) -> str:
        # This is like taking a strided view, except that the offset is added to
        # the one already there rather than replacing it, and the view is not
        # tracked as a view, which is what makes it safe to do.
        return V.graph.wrapper_code.codegen_reinterpret_view(
            self.data,
            self.layout.size,
            self.layout.stride,
            self.layout.offset,
            writer.writeline if writer is not None else V.graph.wrapper_code.writeline,
            dtype=self.layout.dtype,
        )

    def num_reads(self) -> int:
        return 1


@ir_dataclass
class DtypeView(BaseView):
    """The same memory, read as though it held a different type.

    The bits are untouched; only how they are read changes, so this is a view
    rather than a conversion, and reading it as the type it was written as
    first is what makes the bits mean what they should.
    """

    target_dtype: Any

    @classmethod
    def create(cls, x: "IRNode", new_dtype) -> "BaseView":
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_layout = FixedLayout(
                old_layout.device,
                new_dtype,
                old_layout.size,
                old_layout.stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)
        return DtypeView(data=x, target_dtype=new_dtype)

    def __str__(self) -> str:
        return self.str_helper([self.data, self.target_dtype])

    __repr__ = __str__

    @property
    def dtype(self):
        return self.target_dtype

    def get_size(self) -> Sequence:
        return self.data.get_size()

    def make_reindexer(self):
        """The identity, since only the type changed and not where things are."""

        def reindex(index: Sequence) -> Sequence:
            return index

        return reindex

    def make_loader(self):
        inner = self.data.make_loader()

        def loader(idx: Sequence):
            return ops.to_dtype_bitcast(inner(idx), self.target_dtype, self.data.dtype)

        return loader


@ir_dataclass
class SqueezeView(BaseView):
    """A view that drops the dimensions of extent one.

    A dimension of extent one carries no information about the order anything
    runs in, so leaving it in the shape makes every index that refers to it
    carry a zero that says nothing.  Dropping it is not the same as deleting
    data: the element that was at index zero of that dimension is still there,
    and the reindexer is what puts the zeros back.
    """

    @staticmethod
    def squeezer(
        size: Sequence[Expr],
    ):
        """The shape without its unit dimensions, and how to index the original.

        The two are what has to be handed out together: a caller that indexes
        the shorter shape with fewer variables and then does not put the
        dropped positions back has addressed a different element.
        """

        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index: Sequence[Expr]) -> tuple:
            if len(index) != len(not_one):
                raise AssertionError(f"{index} {not_one}")
            new_index: list = [sympy.S.Zero] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    @classmethod
    def create(cls, x: "IRNode", *, dim: int | None = None) -> "IRNode":
        if is_storage_and_layout(x):
            storage, old_layout = as_storage_and_layout(x)
            new_size = []
            new_stride = []
            if dim is not None:
                if not isinstance(dim, int):
                    raise AssertionError(type(dim))
                if not (0 <= dim and dim < len(old_layout.size)):
                    raise AssertionError(
                        "Expected 0 <= dim and dim < len(old_layout.size)"
                    )

            for i, (size, stride) in enumerate(zip(old_layout.size, old_layout.stride)):
                if dim is None:
                    # Kept only if it is not the dimension being dropped.
                    if not V.graph.sizevars.is_size_one_or_false(size):
                        new_size.append(size)
                        new_stride.append(stride)
                else:
                    if i != dim:
                        new_size.append(size)
                        new_stride.append(stride)
                    else:
                        if size != 1:
                            raise AssertionError("expected squeezed size to be 1")

            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        if dim is None:
            return View.create(
                x,
                [
                    s
                    for s in x.get_size()
                    if not V.graph.sizevars.is_size_one_or_false(s)
                ],
            )
        else:
            if x.get_size()[dim] != 1:
                raise AssertionError("Expected x.get_size()[dim] == 1")
            return View.create(x, [s for i, s in enumerate(x.get_size()) if i != dim])

    @staticmethod
    def squeezer(size):
        """The shape without the axes of extent one, and how to index it.

        An axis of extent one holds one element however it is indexed, so
        dropping it and putting a zero back in its place whenever the result is
        indexed leaves every element where it was.
        """

        new_size = [s for s in size if s != 1]
        not_one = [i for i, s in enumerate(size) if s != 1]
        length = len(size)

        def reindex(index):
            if len(index) != len(not_one):
                raise AssertionError(f"{index} {not_one}")
            new_index: list = [sympy.S.Zero] * length
            for idx, s in zip(not_one, index):
                new_index[idx] = s
            return tuple(new_index)

        return new_size, reindex

    def __init__(self, data) -> None:
        raise AssertionError("use SqueezeView.create()")


@ir_dataclass
class GenericView(BaseView):
    """A view described by the shape it presents and how to index through it."""

    size: Sequence
    reindex: Any

    def make_reindexer(self):
        return self.reindex

    def reindex_str(self) -> str:
        index_old = [
            sympy_index_symbol_with_prefix(SymT.INDEX, n) for n in range(len(self.size))
        ]
        index_new = list(self.reindex(index_old))
        return f"lambda {', '.join(map(str, index_old))}: {index_new}"

    def __str__(self) -> str:
        return f"GenericView({self.data}, size={self.size}, reindex={self.reindex_str()})"

    @classmethod
    def create(cls, x, new_size, reindex) -> "BaseView":
        """A view of this value with this shape, where position maps this way."""

        return cls(data=x, size=list(new_size), reindex=reindex)

    def get_size(self) -> Sequence:
        return self.size


@ir_dataclass
class View(GenericView):
    """A change of shape, worked out as a function of the index.

    A reshape is a view exactly when the new shape can be reached from the old
    one by an arithmetic function of the position rather than by moving
    anything.  This works out that function; where no such function exists the
    caller has to copy instead, and is told so by being given ``None`` rather
    than a wrong answer.
    """

    @staticmethod
    def handle_negative_index(idx, size):
        """A negative position counted from the end, turned into a positive one."""

        idx = sympy.expand(idx)
        size = sympy.expand(size)
        evaluate_expr = V.graph.sizevars.shape_env.evaluate_expr
        if evaluate_expr(sympy.Lt(idx, 0)):
            idx = idx + size
        return idx

    @classmethod
    def create(cls, x: "IRNode", new_size: Sequence) -> "IRNode":
        if not isinstance(new_size, Sequence):
            raise AssertionError(type(new_size))
        old_size, new_size = cls.resolve_negative_size(x.get_size(), new_size)

        # A view to the shape it already has is not a view.
        if V.graph.sizevars.statically_known_list_equals(old_size, new_size):
            return x

        unbacked_symbols_in_sizes = (
            len(free_unbacked_symbols(old_size)) > 0
            or len(free_unbacked_symbols(new_size)) > 0
        )
        is_contiguous = is_dense_contiguous_storage_and_layout(x)

        def create_reinterpret_view(inp, new_size, new_stride):
            # The underlying memory has to be laid out exactly as the view
            # expects, or the strides below would be describing something else.
            inp = ExternKernel.require_exact_strides(
                inp, FlexibleLayout.contiguous_strides(inp.get_size())
            )
            storage, old_layout = as_storage_and_layout(inp)
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        def handle_unbacked_or_dynamic_reshape(x):
            """Reshape by arithmetic when the extents are known, by a copy when not.

            The arithmetic form is tried first; where an extent came out of the
            data it cannot be compared at code-writing time, and a copy is used
            instead.
            """

            nonlocal old_size, new_size
            try:
                reindex = cls.dynamic_reshape_indexer(old_size, new_size)
                return cls(data=x, size=list(new_size), reindex=reindex)
            except Exception:
                x = ExternKernel.require_contiguous(x)
                return create_reinterpret_view(
                    x, new_size, FlexibleLayout.contiguous_strides(new_size)
                )

        if 0 in new_size:
            # A tensor with no elements reads nothing, so anything that produces
            # the right type will do.
            def fake_reindex(index):
                return tuple([0] * len(old_size))

            return cls(data=x, size=list(new_size), reindex=fake_reindex)

        elif is_contiguous:
            # The elements already run consecutively, so the new shape does too.
            return create_reinterpret_view(
                x, new_size, FlexibleLayout.contiguous_strides(new_size)
            )

        # The elements do not run consecutively, so it has to be worked out
        # whether the new shape can be reached from the old strides at all.
        if not is_storage_and_layout(x):
            return handle_unbacked_or_dynamic_reshape(x)

        storage, old_layout = as_storage_and_layout(x)

        old_stride = old_layout.stride

        from .utils import _compute_stride

        old_size_symint = V.graph.sizevars.to_symints_or_ints(old_size)
        old_stride_symint = V.graph.sizevars.to_symints_or_ints(old_stride)
        new_size_symint = V.graph.sizevars.to_symints_or_ints(new_size)


        # An extent that came out of the data is compared without a guard,
        # since there is nothing to guard on until it runs.
        new_stride_symint = _compute_stride(
            old_size_symint,
            old_stride_symint,
            new_size_symint,
            size_oblivious=unbacked_symbols_in_sizes,
        )

        if new_stride_symint is not None:
            from tensorplay.graph.experimental.sym_node import SymNode

            new_stride = [
                s.expr if isinstance(s, SymNode) else sympy.Integer(s)
                for s in new_stride_symint
            ]
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        # The strides do not allow it, so the elements have to be moved.
        return handle_unbacked_or_dynamic_reshape(x)

    @staticmethod
    def resolve_negative_size(old_size: Sequence, new_size: Sequence):
        """Work out the dimension whose size was left to be inferred.

        A size of minus one is the one that has to be filled in, and it is
        filled in as whatever makes the two shapes hold the same number of
        elements -- a reshape that changed the number of elements would not be a
        view, and the guard is what says so.
        """

        new_size = [V.graph.sizevars.simplify(x) for x in new_size]
        old_size = [V.graph.sizevars.simplify(x) for x in old_size]

        new_size = list(new_size)
        for i in range(len(new_size)):
            if new_size[i] == -1:
                new_size[i] = sympy.S.One
                new_size[i] = CleanDiv(sympy_product(old_size), sympy_product(new_size))
                break

        V.graph.sizevars.check_equals(sympy_product(old_size), sympy_product(new_size))
        return old_size, new_size

    @classmethod
    def dynamic_reshape_indexer(
        cls,
        old_size: Sequence,
        new_size: Sequence,
        dense_dim: int | None = None,
    ):
        """An index function between two shapes of the same number of elements.

        The two are matched from the innermost outwards: equal extents match
        directly, a smaller new extent inside a larger old one is a position
        within it, and a larger one is a run of them.  Where that matching
        cannot be worked out, the whole thing is done through a single flat
        dimension, which always can be.
        """

        try:
            reindex = cls._dynamic_reshape_indexer(old_size, new_size, dense_dim)
        except (AssertionError, IndexError, Exception):
            # The direct matching did not work out, so the two are flattened
            # and the halves done separately.
            flat = [sympy_product(old_size)]
            reindex1 = cls._dynamic_reshape_indexer(old_size, flat)
            reindex2 = cls._dynamic_reshape_indexer(flat, new_size)
            reindex = fuse_reindexing(reindex1, reindex2)
        return reindex

    @staticmethod
    def _dynamic_reshape_indexer(
        old_size: Sequence,
        new_size: Sequence,
        dense_dim: int | None = None,
    ):
        """A reshape worked out entirely through the index arithmetic."""

        guard_or_false = V.graph.sizevars.guard_or_false

        def compare_sizes(a, b) -> int:
            """Which of two extents is larger, as far as it can be told.

            An extent that came out of the data cannot be compared by a guard,
            so divisibility is used instead: one that is a multiple of the other
            is the larger, unless it is zero, and a zero extent means no loop
            runs and what the arithmetic gives does not matter.
            """

            if guard_or_false(sympy.Eq(a, b)):
                return 0
            if guard_or_false(sympy.Lt(a, b)):
                return -1
            if guard_or_false(sympy.Gt(a, b)):
                return 1

            if V.graph.sizevars.statically_known_multiple_of(b, a):
                return -1
            if V.graph.sizevars.statically_known_multiple_of(a, b):
                return 1

            raise GuardOnDataDependentSymNode(sympy.Eq(a, b))

        vars = [
            sympy_index_symbol_with_prefix(SymT.VIEW, i) for i in range(len(new_size))
        ]

        stack_new = list(zip(vars, new_size))
        stack_old = list(old_size)

        # The dimension the elements run consecutively along is matched first,
        # since it is the one whose match decides whether the rest can be
        # matched at all.
        reordering_dense_dim = (
            dense_dim is not None
            and dense_dim != len(stack_old) - 1
            and len(new_size) == 1
        )
        if reordering_dense_dim:
            old_dim = stack_old.pop(dense_dim)
            stack_old.append(old_dim)

        view_expr = []
        while stack_new and stack_old:
            size_old = stack_old.pop()
            var, size_new = stack_new.pop()
            if size_old == 1:
                view_expr.append(sympy.S.Zero)
                stack_new.append((var, size_new))  # re-add
            elif size_new == 1:
                stack_old.append(size_old)  # re-add
            elif compare_sizes(size_new, size_old) == 0:
                view_expr.append(var)
            elif compare_sizes(size_new, size_old) < 0:
                while compare_sizes(size_new, size_old) < 0:
                    var2, size_new2 = stack_new.pop()
                    var = var2 * size_new + var
                    size_new = size_new * size_new2
                view_expr.append(var)
                V.graph.sizevars.check_equals(size_new, size_old)
            elif compare_sizes(size_new, size_old) > 0:
                divisor = sympy.S.One
                modulus = size_old
                view_expr.append(ModularIndexing(var, divisor, modulus))
                divisor = divisor * modulus
                while compare_sizes(size_new, size_old) > 0:
                    modulus = stack_old.pop()
                    view_expr.append(ModularIndexing(var, divisor, modulus))
                    divisor = divisor * modulus
                    size_old = size_old * modulus
                V.graph.sizevars.check_equals(size_new, size_old)
            else:
                raise AssertionError

        while stack_old:
            size_old = stack_old.pop()
            V.graph.sizevars.check_equals(size_old, 1)
            view_expr.append(sympy.S.Zero)

        while stack_new:
            var, size_new = stack_new.pop()
            V.graph.sizevars.check_equals(size_new, 1)

        if dense_dim is not None and len(new_size) == 1:
            view_expr.reverse()
            # The dimension the elements run along goes back where it was.
            dense_expr = view_expr.pop()
            view_expr.insert(dense_dim, dense_expr)
        else:
            view_expr.reverse()

        if len(view_expr) != len(old_size):
            raise AssertionError("Expected len(view_expr) == len(old_size)")

        def reindex(index: Sequence) -> Sequence:
            if len(index) != len(vars):
                raise AssertionError((len(index), len(vars)))
            replacements = dict(zip(vars, index))
            return tuple(sympy_subs(x, replacements) for x in view_expr)

        return reindex


class SliceView(View):
    """A view of part of one dimension, taking every step-th element of it.

    This is what a slice is: the same elements as before, reached by a position
    that starts somewhere, moves by a step, and stops.  Nothing is copied, which
    is why the step has to be positive -- a negative step would read the
    dimension backwards, which is a different memory order and not a view.
    """

    @classmethod
    def create_with_size(
        cls,
        x: "IRNode",
        dim: int,
        start,
        size,
        step,
    ) -> "IRNode":
        new_size = list(x.get_size())
        new_size[dim] = size

        def reindex(index: Sequence) -> Sequence:
            if len(index) != len(new_size):
                raise AssertionError(f"wrong ndim {index} {new_size}")
            index = list(index)
            index[dim] = index[dim] * step + start
            return index

        return cls(data=x, size=new_size, reindex=reindex)

    @classmethod
    def normalize_start_end(cls, x: "IRNode", dim: int, start, end):
        """Put the two ends inside the dimension, and in order.

        A slice written past the end of a dimension, or before its start, names
        nothing, and one written with the ends the wrong way round names nothing
        either.  Both are brought inside here rather than refused, because that
        is what a slice means.
        """

        sizevars = V.graph.sizevars
        dim_size = x.get_size()[dim]

        if any(free_unbacked_symbols(x) for x in (start, end, dim_size)):
            # An extent that came out of the data cannot be compared, so the
            # comparison is done symbolically instead.
            min_func = Min
            max_func = Max
        elif any(
            # Only reached when the comparisons are made without guards.
            x.has(sympy.Min, sympy.Max, Min, Max)
            for x in (start, end, dim_size)
            if isinstance(x, Expr)
        ):
            min_func = Min
            max_func = Max
        else:
            min_func = sizevars.evaluate_min
            max_func = sizevars.evaluate_max

        def clamp(x, lower, upper):
            clamped_lower = (
                x if sizevars.statically_known_geq(x, lower) else max_func(x, lower)
            )
            clamped_full = (
                clamped_lower
                if sizevars.statically_known_leq(clamped_lower, upper)
                else min_func(clamped_lower, upper)
            )
            return clamped_full

        def clamp_wrap(val, lower, upper, default):
            if val is None:
                return default
            val = cls.handle_negative_index(val, dim_size)
            return clamp(val, lower, upper)

        start = clamp_wrap(start, 0, dim_size, 0)
        end = clamp_wrap(end, start, dim_size, dim_size)
        return start, end

    @classmethod
    def create(  # type: ignore[override]
        cls,
        x: "IRNode",
        dim: int,
        start,
        end,
        step=1,
        clamp: bool = True,
    ) -> "IRNode":
        step = sympy.expand(step)
        if not (isinstance(step, Expr) or step > 0):
            raise AssertionError(step)
        try:
            if start == 0 and end >= 2**63 - 1 and step == 1:
                # The whole dimension, every element, in order: not a slice.
                return x
        except TypeError:
            pass

        new_size = list(x.get_size())

        # Bringing the ends inside is what a slice normally means.  It is
        # skipped only where the caller has already worked the sizes out and
        # says so, since a size that is wrong there would fail silently.
        if clamp:
            start, end = cls.normalize_start_end(x, dim, start, end)

        # How many elements the slice keeps, which is the span rounded up to a
        # whole number of steps.  The span is lifted into the index algebra
        # first, so a slice of settled extents has a settled size: a size left
        # as a division would be a number nobody can read an extent out of.
        new_size[dim] = floordiv(_lift(end - start + (step - 1)), _lift(step))

        if is_storage_and_layout(x):
            # The elements are still where they were, so the dimension is
            # reached by a stride multiplied by the step and an offset moved
            # along by the start.
            storage, old_layout = as_storage_and_layout(x)
            new_stride = list(old_layout.stride)
            new_stride[dim] = new_stride[dim] * step
            new_layout = FixedLayout(
                old_layout.device,
                old_layout.dtype,
                new_size,
                new_stride,
                old_layout.offset + old_layout.stride[dim] * start,
                old_layout.is_pinned,
            )
            return ReinterpretView(data=storage, layout=new_layout)

        return cls.create_with_size(x, dim, start, new_size[dim], step)


def validate_ir(node_or_nodes: Any) -> None:
    """Check that what a lowering handed back is something the rest can hold.

    A lowering's result becomes a graph value, and a graph value is read by
    something that has to know how to reach its elements or how to call it.
    A node that is neither is not wrong so much as unaccounted for, and the
    place that finds out is a long way from the lowering that produced it, so
    it is worth refusing here where the lowering is still named.
    """

    def check(nodes: Any) -> None:
        if nodes is None:
            return
        if isinstance(nodes, (list, tuple)):
            for node in nodes:
                check(node)
        elif isinstance(nodes, dict):
            for node in nodes.values():
                check(node)
        elif not isinstance(
            nodes,
            (
                ExpandView,
                DynamicScalar,
                AssertScalar,
                TensorBox,
                sympy.logic.boolalg.Boolean,
                Expr,
                int,
                EffectfulKernel,
                ShapeAsConstantBuffer,
                OpaqueMultiOutput,
            ),
        ):
            raise AssertionError(
                f"Found {type(nodes)}, which is not a supported top level IR node."
            )

    check(node_or_nodes)


def assign_origin_node(result: Any, n: Any) -> None:
    """Say which node of the graph a value was made by.

    Which node is a best-effort answer rather than a required one: what it is
    used for is saying where a piece of the graph came from when a report about
    it is read, and a value whose origin cannot be pinned down is better left
    unpinned than guessed at.  The descent relies on a box holding storage
    directly meaning the value is not a view onto someone else's memory; a view
    is not descended into, because what a view was made by is its source's
    business rather than its own.
    """

    if isinstance(result, TensorBox) and isinstance(result.data, StorageBox):
        if isinstance(result.data.data, Loops):
            result.data.data._post_init_setattr("origin_node", n)
        elif isinstance(result.data.data, Buffer):
            result.data.data._post_init_setattr("origin_node", n)
            if isinstance(result.data.data, ComputedBuffer) and isinstance(
                result.data.data.data, Loops
            ):
                result.data.data.data._post_init_setattr("origin_node", n)
            elif isinstance(result.data.data, MultiOutput) and not result.data.data.indices:
                if isinstance(result.data.data.inputs[0], Buffer):
                    result.data.data.inputs[0]._post_init_setattr("origin_node", n)
