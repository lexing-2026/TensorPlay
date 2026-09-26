"""Loop-level IR for whole-graph kernel compilation.

A lowered graph is a set of buffers.  Each computed buffer is a *loop nest*
(``Pointwise`` or ``Reduction``) whose body is an ``inner_fn``: a Python
function from symbolic loop indices to a value built with the ``ops``
handler.  Nothing is materialized while lowering -- a pointwise value that
is read once is simply inlined into its reader -- until a consumer needs
memory (a library call, a graph output, a value read by many kernels).

Views never copy: they re-address their source through an index function,
and a realized view of a buffer is a strided reinterpretation of it.

Index arithmetic uses the stax index algebra (``codegen.index_expr``);
``simplify_index`` folds its divisions using the extents of the loop
variables, which keeps generated kernels free of division in the common
contiguous cases.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import math
import threading

import sympy
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from tensorplay.graph.experimental.sympy_functions import OrderedSet
from . import config, metrics
from .utils import (
    argsort,
    argsort_sym,
    cache_on_self_and_args,
    ceildiv,
    get_free_symbols,
)
from .codegen.index_expr import (
    ValueRange,
    free_symbols,
)


def as_index(value):
    """A Python integer lifted into the symbolic value language.

    An expression that is already symbolic is returned as it is, so an index
    built from loop variables stays one expression rather than being rebuilt
    out of its parts.
    """

    import sympy

    if isinstance(value, sympy.Expr):
        return value
    return sympy.Integer(value)


def substitute(expr, mapping):
    """This expression with some axes written as other expressions."""

    from .utils import sympy_subs

    return sympy_subs(expr, mapping)


def floordiv(a, b):
    """How many whole times ``b`` goes into ``a``."""

    from tensorplay.graph.experimental.sympy_functions import FloorDiv

    return FloorDiv(a, b)


def modular_indexing(flat, stride, extent):
    """Which element of an axis a flat position falls on.

    A flat position is the sum of the position along each axis times how far
    that axis is from the start, so one axis's position is what is left after
    the stride has been taken off and the axis's own extent has wrapped.
    """

    from tensorplay.graph.experimental.sympy_functions import ModularIndexing

    return ModularIndexing(flat, stride, extent)


def simplify_index(expr, ranges: dict):
    """Fold exact divisions given each symbol's extent."""

    import sympy

    from .codegen.index_expr import simplify

    return simplify(
        as_index(expr),
        {sym: ValueRange(0, int(extent) - 1) for sym, extent in ranges.items()},
    )


def pexpr(expr) -> str:
    """An index expression written the way a kernel body would write it."""

    import sympy

    from .codegen.index_expr import render_python

    if isinstance(expr, sympy.Expr):
        return sympy.sstr(expr)
    return render_python(as_index(expr))


# ---------------------------------------------------------------------------
# Virtualized handlers
# ---------------------------------------------------------------------------


class _KernelState:
    """What is known about the kernel being emitted, before one is chosen.

    A name is looked up in two places -- the region's and the kernel's -- and
    the kernel's is asked even when no kernel has been entered, so the answer
    has to exist before one does.  The same is true of asking for a fresh name
    for an expression: the emitter asks the kernel rather than keeping a count
    of its own, so that every name in a kernel comes from one place.
    """

    def __init__(self, create_cse_var=None):
        self.removed_buffers: set = set()
        self.inplaced_to_remove: set = set()
        self.current_node = None
        self.itervars: set = set()
        self.cse = None
        self._create_cse_var = create_cse_var

    def create_cse_var(self, name, bounds=None, dtype=None, shape=None):
        """A fresh name for an expression, of the kind this kernel makes.

        A kernel whose names carry more than a name -- a host kernel's know
        whether they are vectors and which loop variables they were built from
        -- hands that kind in when the kernel is entered, and the plain kind is
        what a caller that has not chosen gets.
        """

        if self._create_cse_var is not None:
            return self._create_cse_var(name, bounds, dtype, shape)
        from .codegen.common import CSEVariable

        return CSEVariable(name, bounds, dtype, shape)


class NullKernel(_KernelState):
    """The kernel that is there when no kernel is being emitted.

    Code written outside a kernel -- the part of a compiled program that
    allocates and calls -- still asks what the kernel dropped and what it wrote
    in place, and outside a kernel the answer is that nothing was dropped,
    which is what these two empty tables say.
    """


class _ReachableHandler:
    """One of the operation handlers, resolved when it is first asked for.

    The handlers and this object refer to each other, so neither can be
    imported while the other is still being set up.  Naming the handler here
    and looking it up on use lets either side be imported first.
    """

    def __init__(self, name: str):
        self._name = name

    def __get__(self, obj, objtype=None):
        from . import ops_handler

        return getattr(ops_handler, self._name)


class _Virtual(threading.local):
    """What is current while code is being emitted, reached from anywhere.

    An operation is answered by whatever is current when it is asked rather
    than by what is passed to it, so the current things are held here and the
    setters below are how they are put in place.  The setters are on the class
    rather than the instance so that a handler can be installed without holding
    a reference to this object first.
    """

    def __init__(self):
        self.ops = None
        self.graph = None
        self.sizevars = None
        self.current_node = None
        self.kernel = NullKernel()
        self.interpreter = None
        self.debug = None
        self.aot_compilation = False
        self.extern_kernel_nodes = None
        self.real_inputs = None
        self.fake_mode = None
        self.local_buffer_context = None

    #: The handlers a node reaches for itself rather than being handed: the
    #: one that writes a body out as source, and the one that answers an
    #: operation without computing it.
    KernelFormatterHandler = _ReachableHandler("KernelFormatterHandler")
    MockHandler = _ReachableHandler("MockHandler")
    DefaultHandler = _ReachableHandler("DefaultHandler")
    WrapperHandler = _ReachableHandler("WrapperHandler")

    set_ops_handler = None  # 由下方模块级函数填入
    get_ops_handler = None
    set_local_buffer_context = None
    set_graph_handler = None
    set_kernel_handler = None
    set_debug_handler = None
    set_interpreter_handler = None
    set_aot_compilation = None
    get_aot_compilation = None
    set_current_node = None
    set_extern_kernel_nodes = None
    set_real_inputs = None
    get_real_inputs = None
    set_fake_mode = None
    get_fake_mode = None


V = _Virtual()


@contextlib.contextmanager
def set_ops_handler(handler):
    previous = V.ops
    V.ops = handler
    try:
        yield handler
    finally:
        V.ops = previous


@contextlib.contextmanager
def set_kernel_handler(kernel):
    """Make a kernel the one being emitted, and put back the previous one.

    An operator is answered by the kernel it is being emitted for, so the
    kernel has to be reachable from the operator rather than passed to it: a
    handler is entered once and then every operation it sees goes to that
    kernel.
    """

    previous = V.kernel
    V.kernel = kernel
    try:
        yield kernel
    finally:
        V.kernel = previous


@contextlib.contextmanager
def set_local_buffer_context(local_buffer_context):
    """Make a set of function-local buffers the ones in scope, then put back the previous set.

    A local buffer belongs to one compiled function and is invisible outside
    it, so while that function is being written the operators have to be able
    to name these buffers and find their extents.  Publishing the set on the
    region is what lets an operator reach it without being handed a second
    argument, and putting the previous one back on the way out is what keeps
    the next function from seeing buffers that are not its own.
    """

    previous = V.local_buffer_context
    V.local_buffer_context = local_buffer_context
    try:
        yield local_buffer_context
    finally:
        V.local_buffer_context = previous


@contextlib.contextmanager
def set_graph(graph):
    """Make a region the one being compiled, and put back the previous one.

    The size variables of the region are published alongside it, since what
    is known about the extents belongs to the region they came from and a
    generator that is handed the region has to be able to ask about them
    without being handed the region a second time.
    """

    previous = V.graph
    previous_sizevars = V.sizevars
    V.graph = graph
    V.sizevars = getattr(graph, "sizevars", None)
    try:
        yield graph
    finally:
        V.graph = previous
        V.sizevars = previous_sizevars


@contextlib.contextmanager
def set_current_node(node):
    """Publish the node being processed, for a generator that asks about it.

    A generator decides about a layout partly from what the region knows about
    the node it is emitting for -- a view that should not be padded, for
    instance -- and the node is not threaded through those calls, so it is
    published here and taken back afterwards.
    """

    previous = V.current_node
    V.current_node = node
    try:
        yield node
    finally:
        V.current_node = previous


def get_ops_handler():
    """The handler operations are being answered by, or nothing."""

    return V.ops


@contextlib.contextmanager
def set_interpreter_handler(handler):
    """Make an interpreter current, and put back the previous one.

    A handler that records what a body did may need to know which node of the
    graph it is looking at, so the interpreter driving the graph publishes
    itself while it runs.
    """

    previous = V.interpreter
    V.interpreter = handler
    try:
        yield handler
    finally:
        V.interpreter = previous


@contextlib.contextmanager
def set_debug_handler(handler):
    """Make a recorder of what was emitted current, and put back the previous."""

    previous = V.debug
    V.debug = handler
    try:
        yield handler
    finally:
        V.debug = previous


@contextlib.contextmanager
def set_aot_compilation(value: bool):
    """Record whether the code is being emitted ahead of time, and restore it."""

    previous = V.aot_compilation
    V.aot_compilation = value
    try:
        yield value
    finally:
        V.aot_compilation = previous


def get_aot_compilation() -> bool:
    """Whether the code is being emitted ahead of time."""

    return V.aot_compilation


@contextlib.contextmanager
def set_current_node_handler(node):
    """Make a node current for a generator that asks about it."""

    previous = V.current_node
    V.current_node = node
    try:
        yield node
    finally:
        V.current_node = previous


@contextlib.contextmanager
def set_extern_kernel_nodes(nodes):
    """Make a set of nodes that must stay separate current, and restore it."""

    previous = V.extern_kernel_nodes
    V.extern_kernel_nodes = nodes
    try:
        yield nodes
    finally:
        V.extern_kernel_nodes = previous


@contextlib.contextmanager
def set_real_inputs(real_inputs):
    """Make the real tensors behind the graph's inputs current, and restore them.

    A layout decision sometimes has to be made against the values rather than
    the shapes, and the values live on the tensors rather than in the graph, so
    they are published here for the duration of the decision.
    """

    previous = V.real_inputs
    V.real_inputs = real_inputs
    try:
        yield real_inputs
    finally:
        V.real_inputs = previous


def get_real_inputs():
    """The real tensors behind the graph's inputs, if they are known."""

    return V.real_inputs


@contextlib.contextmanager
def set_fake_mode(mode):
    """Make a tensor mode current, and put back the previous one."""

    previous = V.fake_mode
    V.fake_mode = mode
    try:
        yield mode
    finally:
        V.fake_mode = previous


def get_fake_mode():
    """The tensor mode current, if one is."""

    return V.fake_mode


# The handlers are reachable from the object that holds what is current, so
# that code holding only that object can install one.
_Virtual.set_ops_handler = staticmethod(set_ops_handler)
_Virtual.get_ops_handler = staticmethod(get_ops_handler)
_Virtual.set_local_buffer_context = staticmethod(set_local_buffer_context)
_Virtual.set_graph_handler = staticmethod(set_graph)
_Virtual.set_kernel_handler = staticmethod(set_kernel_handler)
_Virtual.set_current_node = staticmethod(set_current_node)
_Virtual.set_interpreter_handler = staticmethod(set_interpreter_handler)
_Virtual.set_debug_handler = staticmethod(set_debug_handler)
_Virtual.set_aot_compilation = staticmethod(set_aot_compilation)
_Virtual.get_aot_compilation = staticmethod(get_aot_compilation)
_Virtual.set_extern_kernel_nodes = staticmethod(set_extern_kernel_nodes)
_Virtual.set_real_inputs = staticmethod(set_real_inputs)
_Virtual.get_real_inputs = staticmethod(get_real_inputs)
_Virtual.set_fake_mode = staticmethod(set_fake_mode)
_Virtual.get_fake_mode = staticmethod(get_fake_mode)


def get_current_node():
    """The node being processed, or nothing when none is."""

    return V.current_node


class _OpsProxy:
    """``ops.<name>(...)`` forwards to the active handler."""

    def __getattr__(self, name):
        return getattr(V.ops, name)


ops = _OpsProxy()


class DeferredOps:
    """Handler active while a graph is lowered.

    A lowering describes its arithmetic by naming operations, and those names
    are only resolved when the loop body is finally recorded.  This handler
    therefore turns a name into a closure over the real handler instead of
    demanding one during the walk.
    """

    def __getattr__(self, name):
        def call(*args):
            handler = V.ops
            if handler is None or isinstance(handler, DeferredOps):
                raise RuntimeError(
                    f"operation {name} ran while the graph was still lowering; "
                    "arithmetic belongs inside a loop body"
                )
            return getattr(handler, name)(*args)

        return call


# ---------------------------------------------------------------------------
# Expression graph recorded from inner functions
# ---------------------------------------------------------------------------


FLOAT_RANK = {"float16": 1, "bfloat16": 1, "float32": 2, "float64": 3}


class ReinterpretView:
    """A strided window onto a buffer that already exists.

    Nothing is copied: the window names an offset, a shape and a stride, and
    the elements are read through the buffer it names.  This is what is handed
    to a library call that was written in terms of a shape, a stride and an
    offset rather than in terms of this compiler's own values.
    """

    buffer: "Buffer"
    size: tuple
    stride: tuple
    offset: int

    def get_name(self) -> str:
        return self.buffer.get_name()

    def get_size(self):
        return self.size

    def get_dtype(self):
        return self.buffer.get_dtype()

    def get_device(self):
        return self.buffer.get_device()

    def get_stride(self):
        return self.stride

    def get_offset(self):
        return self.offset

    def get_layout(self):
        return self.buffer.get_layout()

    def make_indexer(self):
        return self.buffer.get_layout().make_indexer()

    def is_input_buffer(self) -> bool:
        return self.buffer.is_input_buffer()


def _index_exprs(body: LoopBody) -> list:
    exprs = [as_index(v.args[1]) for v in body.loads]
    exprs += [as_index(e) for e in body.stores.values()]
    for v in iter_values(body.root):
        if v.op == "index_expr":
            exprs.append(as_index(v.args[0]))
    return exprs


def _merge_group(body: LoopBody, reduction: bool) -> LoopBody:
    """Merge the loops of one group, and renumber the body for what is left.

    Which loops may be merged is not decided here: it is asked of the part of
    the project that already knows how an index can be renumbered for a set of
    loops that is smaller than the one it was written for.  That part answers
    with the new extents and a way of renumbering into them, or with the same
    extents when nothing may be merged -- so the two decisions here are whether
    to take the answer, and which of the body's two groups to ask about.
    """

    from .codegen.common import index_prevent_reordering

    vars_ = body.rvars if reduction else body.vars
    sizes = list(body.rsizes if reduction else body.sizes)
    if len(sizes) < 2:
        return body
    formulas = index_prevent_reordering(_index_exprs(body), vars_, sizes)
    new_sizes, reindex, _prune = V.graph.sizevars._simplify_loops(
        vars_, sizes, formulas
    )
    if list(new_sizes) == sizes:
        return body
    new_vars = fresh_symbols("p", len(new_sizes))
    renumbered = _renumber(body, vars_, new_vars, reindex)
    if reduction:
        return dataclasses.replace(
            renumbered, rvars=new_vars, rsizes=list(new_sizes)
        )
    return dataclasses.replace(renumbered, vars=new_vars, sizes=list(new_sizes))






def _renumber(body: LoopBody, old_vars, new_vars, reindex) -> LoopBody:
    """The same recorded body, numbered for the loops that are left.

    A merge changes which loops a body runs over and not what it computes, so
    the values are kept and each recorded access is given again in terms of the
    loops that remain.  The tree is walked once and each value is rebuilt once,
    so a value used in two places stays one value -- which is what lets a later
    read see that the two uses are the same.

    The rewrite goes through the recorded tree rather than substituting into it
    as a whole, because what is recorded is not an expression: it is a tree of
    operations whose arguments are expressions, and only those are substituted.
    """

    # Which loop each old one became.  The renumbering answers in terms of the
    # loops that are left -- one entry per axis the body used to have, with a
    # zero where two axes were folded into one -- so handing it the new loops
    # and reading the answer back against the old ones gives the substitution
    # the recorded indexes need.
    replacement = dict(zip(old_vars, reindex(list(new_vars))))

    memo: dict[int, Value] = {}

    def fix(expr):
        if not replacement:
            return expr
        if isinstance(expr, int):
            return int(substitute(sympy.Integer(expr), replacement))
        return substitute(expr, replacement)

    def visit(v):
        if not isinstance(v, Value):
            return v
        hit = memo.get(id(v))
        if hit is not None:
            return hit
        if v.op == "load":
            name, index, mask = v.args
            out = Value("load", (name, fix(index), visit(mask)), v.dtype)
        elif v.op == "index_expr":
            out = Value("index_expr", (fix(v.args[0]),), v.dtype)
        else:
            out = Value(v.op, tuple(visit(x) for x in v.args), v.dtype)
        memo[id(v)] = out
        return out

    root = (
        tuple(visit(r) for r in body.root)
        if isinstance(body.root, tuple)
        else visit(body.root)
    )
    loads = [memo[id(v)] for v in body.loads if id(v) in memo]
    stores = {name: fix(e) for name, e in body.stores.items()}
    return LoopBody(body.vars, body.sizes, body.rvars, body.rsizes, root, loads, stores)


def simplify_loops(body: LoopBody) -> LoopBody:
    """Merge contiguous loop dims (x group, then reduction group)."""

    body = _merge_group(body, reduction=False)
    if body.rvars:
        body = _merge_group(body, reduction=True)
    return body


#: Hands out a number for each symbol asked for, so that two bodies captured at
#: different times never name a loop the same.
_symbol_counter = itertools.count()


def fresh_symbols(prefix: str, n: int) -> list:
    """``n`` loop variables of one kind, none of them named before.

    They are symbols in the symbolic value language, so an index built from
    them can be simplified against the extents they are known to range over
    rather than only pattern-matched.
    """

    import sympy

    return [
        sympy.Symbol(
            f"{prefix}{next(_symbol_counter)}", integer=True, nonnegative=True
        )
        for _ in range(n)
    ]


def iter_values(value):
    """Every recorded value inside this one, however deeply nested.

    A recorded body is a tree of values, and a question about what it computes
    is a question about all of them rather than about the root.
    """

    seen = set()
    stack = [value]
    while stack:
        v = stack.pop()
        if not isinstance(v, Value) or id(v) in seen:
            continue
        seen.add(id(v))
        yield v
        stack.extend(a for a in v.args if isinstance(a, (Value, list, tuple)))
        if isinstance(v.args, (list, tuple)):
            stack.extend(a for a in v.args if isinstance(a, (Value, list, tuple)))


@dataclass(eq=False)
class LoopBody:
    """A body captured as a tree of values, with the loops it runs over.

    This is the shape the program path works in, where a body is recorded by
    running it and then read back.  The shape used elsewhere records a body as a
    graph instead, which says more and costs more to take apart.
    """

    vars: list
    sizes: list
    rvars: list
    rsizes: list
    root: object
    loads: list = field(default_factory=list)
    stores: dict = field(default_factory=dict)


def record_body(loops: Loops) -> LoopBody:
    """Run a body once with a handler in place of the code, and record what it did.

    The loops are given names to run under, and what comes back is which
    positions were read and what the result was -- which is everything the
    program path needs to write a body out by hand.
    """

    handler = RecordingOps()
    vars_ = fresh_symbols("i", len(loops.ranges))
    # Imported here rather than at module scope: the body classes live in the
    # layer above this one, which imports this module, so a top-level import
    # would be a cycle.
    from .ir import Reduction

    with set_ops_handler(handler):
        if isinstance(loops, Reduction):
            rvars = fresh_symbols("r", len(loops.reduction_ranges))
            root = loops.inner_fn(vars_, rvars)
            return LoopBody(
                vars_,
                list(loops.ranges),
                rvars,
                list(loops.reduction_ranges),
                root,
                handler.loads,
            )
        root = loops.inner_fn(vars_)
    return LoopBody(vars_, list(loops.ranges), [], [], root, handler.loads)



def dtype_name(dtype) -> str:
    return str(dtype).rsplit(".", 1)[-1]


def compute_dtype(*names: str) -> str:
    """Arithmetic element type: reduced floats compute in float32."""

    best = "float32"
    for name in names:
        if name == "float64":
            best = "float64"
    return best




#: Storage types whose value the arithmetic width lifts on load.  A consumer
#: that declares one arithmetic type reads these through the same buffer and
#: converts after loading, so asking it for a wider value costs no pass.
PROMOTED_ON_LOAD = frozenset({"float16", "bfloat16"})


def promotes_on_load(storage_dtype) -> bool:
    """Whether a load of ``storage_dtype`` is converted to float32."""
    return dtype_name(storage_dtype) in PROMOTED_ON_LOAD


@dataclass(eq=False)
class Value:
    """One node of a recorded loop body."""

    op: str
    args: tuple
    dtype: str

    def __repr__(self) -> str:
        return f"Value({self.op}, {self.dtype})"


class RecordingOps:
    """Ops handler that records a loop body as an expression DAG."""

    def __init__(self):
        self.loads: list[Value] = []
        self._masks: list[Value] = []

    # memory ---------------------------------------------------------------
    def load(self, name: str, index) -> Value:
        buffer = V.graph.get_buffer(name)
        mask = self._masks[-1] if self._masks else None
        v = Value("load", (name, index, mask), dtype_name(buffer.get_dtype()))
        self.loads.append(v)
        return v

    def masked(self, mask: Value, body: Callable[[], Value], other) -> Value:
        """Evaluate ``body`` only where ``mask`` holds; ``other`` elsewhere."""

        combined = mask if not self._masks else self.and_(self._masks[-1], mask)
        self._masks.append(combined)
        try:
            value = body()
        finally:
            self._masks.pop()
        fill = other if isinstance(other, Value) else self.constant(other, "float32")
        return self.where(mask, value, fill)

    def and_(self, a: Value, b: Value) -> Value:
        return Value("and_", (a, b), "bool")

    # values ---------------------------------------------------------------
    def constant(self, value, dtype) -> Value:
        return Value("constant", (value,), dtype_name(dtype))

    def index_expr(self, expr, dtype) -> Value:
        return Value("index_expr", (expr,), dtype_name(dtype))

    def to_dtype(self, x: Value, dtype) -> Value:
        return Value("to_dtype", (x,), dtype_name(dtype))

    def where(self, cond, a, b) -> Value:
        return Value("where", (cond, a, b), compute_dtype(a.dtype, b.dtype))

    def _binary(self, op, a, b):
        return Value(op, (a, b), compute_dtype(a.dtype, b.dtype))

    def _unary(self, op, a):
        return Value(op, (a,), compute_dtype(a.dtype))

    def __getattr__(self, name):
        if name in _BINARY_OPS:
            return functools.partial(self._binary, name)
        if name in _UNARY_OPS:
            return functools.partial(self._unary, name)
        if name in _COMPARE_OPS:
            return lambda a, b, _n=name: Value(_n, (a, b), "bool")
        raise AttributeError(name)


_BINARY_OPS = frozenset(
    {"add", "sub", "mul", "truediv", "maximum", "minimum", "pow"}
)
_UNARY_OPS = frozenset(
    {"neg", "exp", "log", "sigmoid", "rsqrt", "sqrt", "reciprocal", "abs",
     "sin", "cos", "tanh", "relu", "square"}
)
_COMPARE_OPS = frozenset({"lt", "le", "gt", "ge", "eq", "ne"})


# ---------------------------------------------------------------------------
# Layouts and IR nodes
# ---------------------------------------------------------------------------


#: The stride order a tensor with its channels outermost has: the channel
#: dimension moves furthest, the outermost dimension next, and the rest in
#: between, so that walking channels walks memory.
NHWC_STRIDE_ORDER = [3, 0, 2, 1]
NHWDC_STRIDE_ORDER = [4, 0, 3, 2, 1]


def contiguous_strides(size: Sequence[int]) -> tuple[int, ...]:
    strides = []
    running = 1
    for extent in reversed(size):
        strides.append(running)
        running *= max(int(extent), 1)
    return tuple(reversed(strides))


def get_fill_order(
    seq: Sequence, shape_env=None
) -> Sequence[int]:
    """The order the dimensions are filled in, which is the strides sorted.

    Sorting is over the values, so the innermost dimension is the one with the
    smallest stride.  When the strides are symbolic the comparison may not be
    settled, and the order that comes back is then one to optimize with rather
    than one to decide anything with.
    """

    import sympy

    if shape_env is None or all(isinstance(s, (int, sympy.Integer)) for s in seq):
        sorted_idx = argsort(seq)
    else:
        sorted_idx = argsort_sym(shape_env, seq)
    return sorted_idx


def stride_order2fill_order(order: Sequence) -> Sequence[int]:
    """The fill order that a stride order describes.

    A stride order says which dimension has the first stride, the second
    stride, and so on; the fill order says which dimension is filled first, and
    the two are inverses of each other.  Channels last, a stride order of
    ``[3, 0, 2, 1]``, is a fill order of ``[1, 3, 2, 0]``.
    """

    lookup = {pos: idx for idx, pos in enumerate(order)}
    fill_order = [lookup[i] for i in range(len(order))]
    return fill_order


def get_stride_order(
    seq: Sequence, shape_env=None
) -> Sequence[int]:
    """The stride order of a sequence of strides."""

    sorted_idx = get_fill_order(seq, shape_env)
    out = [0 for _ in range(len(seq))]
    for i, elem in enumerate(sorted_idx):
        out[elem] = i
    return out


def is_contiguous_strides_for_shape(
    stride: Sequence, shape: Sequence
) -> bool:
    """Whether these strides address a shape as one unbroken run.

    An extent of one is skipped, since it addresses a single element however
    it is strided, and a stride is accepted either as the running product of
    the extents inside it or as the running product without the extents of
    one, because both describe the same addresses.
    """

    import sympy

    from tensorplay.graph.experimental.sympy_functions import Max

    expected_stride = 1
    expected_stride_max = 1
    for x, y in reversed(tuple(zip(shape, stride))):
        if x == 1:
            continue

        if not V.graph.sizevars.statically_known_equals(
            y, expected_stride
        ) and not V.graph.sizevars.statically_known_equals(y, expected_stride_max):
            return False

        expected_stride_max *= Max(1, x)
        expected_stride *= x

    return True


def get_align_for_dtype(dtype) -> int:
    """How many elements of this type make up one aligned access."""

    return config.padding_alignment_bytes // dtype.itemsize


def compute_required_storage_length(shape, strides, storage_offset):
    """How many elements of storage a tensor of this geometry occupies."""

    from .utils import compute_required_storage_length as _impl

    return _impl(shape, strides, storage_offset)


def make_channels_last_strides_for(shape):
    """The strides of a tensor whose channels are outermost, whatever its rank."""

    from .utils import make_channels_last_strides_for as _impl

    return _impl(shape)


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


def prod(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out
