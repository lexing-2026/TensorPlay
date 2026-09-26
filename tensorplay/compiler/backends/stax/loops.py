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
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .codegen.index_expr import (
    Const,
    Expr,
    Symbol,
    ValueRange,
    affine_coeff,
    floordiv,
    free_symbols,
    modular_indexing,
    render_python,
    simplify,
    substitute,
)


def as_index(value) -> Expr:
    """Lift a Python integer into the index algebra."""

    return value if isinstance(value, Expr) else Const(int(value))


def simplify_index(expr, ranges: dict) -> Expr:
    """Fold exact divisions given each symbol's extent."""

    return simplify(
        as_index(expr),
        {sym: ValueRange(0, int(extent) - 1) for sym, extent in ranges.items()},
    )


def pexpr(expr) -> str:
    return render_python(as_index(expr))


# ---------------------------------------------------------------------------
# Virtualized handlers
# ---------------------------------------------------------------------------


class _Virtual(threading.local):
    def __init__(self):
        self.ops = None
        self.graph = None


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
def set_graph(graph):
    previous = V.graph
    V.graph = graph
    try:
        yield graph
    finally:
        V.graph = previous


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


def contiguous_strides(size: Sequence[int]) -> tuple[int, ...]:
    strides = []
    running = 1
    for extent in reversed(size):
        strides.append(running)
        running *= max(int(extent), 1)
    return tuple(reversed(strides))


def prod(values) -> int:
    out = 1
    for v in values:
        out *= int(v)
    return out


@dataclass
class Layout:
    device: Any
    dtype: Any
    size: tuple
    stride: tuple
    offset: int = 0

    def indexer(self, index):
        expr: Expr = Const(self.offset)
        for i, s, extent in zip(index, self.stride, self.size):
            if extent != 1 and s != 0:
                expr = expr + as_index(i) * int(s)
        return expr

    def is_contiguous(self) -> bool:
        return all(
            extent == 1 or st == cst
            for extent, st, cst in zip(self.size, self.stride, contiguous_strides(self.size))
        )


class IRNode:
    def get_size(self) -> tuple:
        raise NotImplementedError

    def get_dtype(self):
        raise NotImplementedError

    def get_device(self):
        raise NotImplementedError

    def make_loader(self) -> Callable:
        raise NotImplementedError


class Buffer(IRNode):
    def __init__(self, name: str, layout: Layout):
        self.name = name
        self.layout = layout

    def get_size(self):
        return self.layout.size

    def get_dtype(self):
        return self.layout.dtype

    def get_device(self):
        return self.layout.device

    def make_loader(self):
        name = self.name
        indexer = self.layout.indexer

        def loader(index):
            return ops.load(name, indexer(index))

        return loader

    def __repr__(self):
        return f"{type(self).__name__}({self.name}, {self.layout.size}, {dtype_name(self.layout.dtype)})"


class InputBuffer(Buffer):
    pass


class ConstantBuffer(Buffer):
    """A tensor the graph closes over (lifted as an extra kernel argument)."""

    def __init__(self, name, layout, value):
        super().__init__(name, layout)
        self.value = value


class Loops(IRNode):
    def __init__(self, device, dtype, inner_fn, ranges):
        self.device = device
        self.dtype = dtype
        self.inner_fn = inner_fn
        self.ranges = tuple(int(r) for r in ranges)

    def get_size(self):
        return self.ranges

    def get_dtype(self):
        return self.dtype

    def get_device(self):
        return self.device


class Pointwise(Loops):
    def make_loader(self):
        return self.inner_fn

    def num_reads(self) -> int:
        return len(record_body(self).loads)


class Reduction(Loops):
    """``reduction_type`` over ``reduction_ranges``: sum, max, min or welford."""

    def __init__(self, device, dtype, inner_fn, ranges, reduction_ranges, reduction_type):
        super().__init__(device, dtype, inner_fn, ranges)
        self.reduction_ranges = tuple(int(r) for r in reduction_ranges)
        self.reduction_type = reduction_type

    def make_loader(self):
        raise RuntimeError("a reduction is read only after it is realized")


class ComputedBuffer(Buffer):
    def __init__(self, name, layout, data: Loops):
        super().__init__(name, layout)
        self.data = data
        # Set for the extra outputs of a multi-output reduction (welford):
        # they share one loop nest with the first buffer.
        self.welford_parent: ComputedBuffer | None = None
        self.welford_index = 0
        self.welford_siblings: list[ComputedBuffer] = []

    def is_reduction(self) -> bool:
        return isinstance(self.data, Reduction)


class ExternKernel(IRNode):
    """A library call: runs ``target`` on realized inputs."""

    def __init__(self, name, target, args, kwargs, meta_values, call_method=False):
        self.name = name
        self.target = target
        self.args = args
        self.kwargs = kwargs
        # A method call names its operation with a string and receives the
        # object it is called on as the first argument.
        self.call_method = call_method
        # Traced output value(s): shapes/dtypes of what the call returns.
        self.meta_values = meta_values
        self.outputs: list[ExternOutput] = []

    def input_buffers(self) -> list[Buffer]:
        found = []

        def visit(a):
            if isinstance(a, (Buffer, ReinterpretView)):
                found.append(a.buffer if isinstance(a, ReinterpretView) else a)
            elif isinstance(a, (list, tuple)):
                for item in a:
                    visit(item)
            elif isinstance(a, dict):
                for item in a.values():
                    visit(item)

        visit(self.args)
        visit(self.kwargs)
        return found


class ExternOutput(Buffer):
    def __init__(self, name, layout, kernel: ExternKernel, path: tuple):
        super().__init__(name, layout)
        self.kernel = kernel
        self.path = path  # position inside the call's (nested) result


@dataclass
class ReinterpretView:
    """A strided view of a realized buffer, handed to library calls."""

    buffer: Buffer
    size: tuple
    stride: tuple
    offset: int


class View(IRNode):
    """Re-address ``source`` through ``reindex`` (new index -> source index)."""

    def __init__(self, source: "TensorBox", size, reindex):
        self.source = source
        self.size = tuple(int(s) for s in size)
        self.reindex = reindex

    def get_size(self):
        return self.size

    def get_dtype(self):
        return self.source.get_dtype()

    def get_device(self):
        return self.source.get_device()

    def make_loader(self):
        inner = self.source.make_loader()
        reindex = self.reindex

        def loader(index):
            return inner(reindex(index))

        return loader


class TensorBox:
    """Lowering-time handle for a tensor value (possibly not materialized)."""

    def __init__(self, node: IRNode):
        self.node = node

    def get_size(self):
        return self.node.get_size()

    def get_dtype(self):
        return self.node.get_dtype()

    def get_device(self):
        return self.node.get_device()

    def numel(self):
        return prod(self.get_size())

    def make_loader(self):
        return self.node.make_loader()

    def realize(self) -> Buffer | None:
        """Materialize a loop nest into a buffer; views realize their source."""

        node = self.node
        if isinstance(node, Buffer):
            return node
        if isinstance(node, Loops):
            buffer = V.graph.register_computed(node)
            self.node = buffer
            return buffer
        if isinstance(node, View):
            node.source.realize()
            return None
        return None

    def __repr__(self):
        return f"TensorBox({self.node!r})"


# ---------------------------------------------------------------------------
# Recording loop bodies
# ---------------------------------------------------------------------------


@dataclass
class LoopBody:
    """A recorded loop nest: variables, their extents and the value(s)."""

    vars: list
    sizes: list
    rvars: list
    rsizes: list
    root: Value | tuple
    loads: list = field(default_factory=list)
    # buffer name -> index its value is stored at (in terms of ``vars``)
    stores: dict = field(default_factory=dict)


_symbol_counter = itertools.count()


def fresh_symbols(prefix: str, n: int) -> list:
    return [Symbol(f"{prefix}{next(_symbol_counter)}") for _ in range(n)]


def record_body(loops: Loops) -> LoopBody:
    handler = RecordingOps()
    vars_ = fresh_symbols("i", len(loops.ranges))
    with set_ops_handler(handler):
        if isinstance(loops, Reduction):
            rvars = fresh_symbols("r", len(loops.reduction_ranges))
            root = loops.inner_fn(vars_, rvars)
            return LoopBody(vars_, list(loops.ranges), rvars, list(loops.reduction_ranges), root, handler.loads)
        root = loops.inner_fn(vars_)
    return LoopBody(vars_, list(loops.ranges), [], [], root, handler.loads)


def _rewrite_body(body: LoopBody, mapping: dict) -> LoopBody:
    """Substitute loop variables in every index the body touches."""

    memo: dict[int, Value] = {}

    def visit(v):
        if not isinstance(v, Value):
            return v
        hit = memo.get(id(v))
        if hit is not None:
            return hit
        if v.op == "load":
            name, index, mask = v.args
            out = Value("load", (name, substitute(as_index(index), mapping), visit(mask)), v.dtype)
        elif v.op == "index_expr":
            out = Value("index_expr", (substitute(as_index(v.args[0]), mapping),), v.dtype)
        else:
            out = Value(v.op, tuple(visit(a) for a in v.args), v.dtype)
        memo[id(v)] = out
        return out

    root = tuple(visit(r) for r in body.root) if isinstance(body.root, tuple) else visit(body.root)
    loads = [memo[id(v)] for v in body.loads if id(v) in memo]
    stores = {name: substitute(as_index(e), mapping) for name, e in body.stores.items()}
    return LoopBody(body.vars, body.sizes, body.rvars, body.rsizes, root, loads, stores)


def _index_exprs(body: LoopBody) -> list:
    exprs = [as_index(v.args[1]) for v in body.loads]
    exprs += [as_index(e) for e in body.stores.values()]
    for v in iter_values(body.root):
        if v.op == "index_expr":
            exprs.append(as_index(v.args[0]))
    return exprs


def _merge_group(body: LoopBody, reduction: bool) -> LoopBody:
    """Merge adjacent dims of one loop group where every access is contiguous."""

    while True:
        vars_ = body.rvars if reduction else body.vars
        sizes = body.rsizes if reduction else body.sizes
        exprs = _index_exprs(body)
        merged = False
        # drop unit dims first: they index nothing
        for k, extent in enumerate(sizes):
            if extent == 1 and len(sizes) > 1:
                zero = {vars_[k]: Const(0)}
                body = _rewrite_body(body, zero)
                new_vars = vars_[:k] + vars_[k + 1 :]
                new_sizes = sizes[:k] + sizes[k + 1 :]
                body = _with_group(body, reduction, new_vars, new_sizes)
                merged = True
                break
        if merged:
            continue
        for k in range(len(sizes) - 1):
            a, b = vars_[k], vars_[k + 1]
            inner = sizes[k + 1]
            ok = True
            for e in exprs:
                ca = affine_coeff(e, a)
                cb = affine_coeff(e, b)
                if ca is None or cb is None or ca != cb * inner:
                    ok = False
                    break
            if not ok:
                continue
            fused = Symbol(f"{a.name}m")
            # affine in both: e = rest + cb*(a*inner + b) = rest + cb*fused
            mapping_rest = {a: Const(0), b: Const(0)}
            new_body = _rewrite_affine(body, a, b, inner, fused)
            new_vars = vars_[:k] + [fused] + vars_[k + 2 :]
            new_sizes = sizes[:k] + [sizes[k] * inner] + sizes[k + 2 :]
            body = _with_group(new_body, reduction, new_vars, new_sizes)
            merged = True
            break
        if not merged:
            return body


def _rewrite_affine(body: LoopBody, a, b, inner: int, fused) -> LoopBody:
    """Rewrite exprs affine in ``a``/``b`` (with a's coeff = inner * b's)."""

    def fix(expr):
        expr = as_index(expr)
        cb = affine_coeff(expr, b)
        if cb is None:
            return expr
        rest = substitute(expr, {a: Const(0), b: Const(0)})
        return rest + fused * cb if cb else rest

    memo: dict[int, Value] = {}

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

    root = tuple(visit(r) for r in body.root) if isinstance(body.root, tuple) else visit(body.root)
    loads = [memo[id(v)] for v in body.loads if id(v) in memo]
    stores = {name: fix(e) for name, e in body.stores.items()}
    return LoopBody(body.vars, body.sizes, body.rvars, body.rsizes, root, loads, stores)


def _with_group(body: LoopBody, reduction: bool, vars_, sizes) -> LoopBody:
    if reduction:
        return LoopBody(body.vars, body.sizes, list(vars_), list(sizes), body.root, body.loads, body.stores)
    return LoopBody(list(vars_), list(sizes), body.rvars, body.rsizes, body.root, body.loads, body.stores)


def simplify_loops(body: LoopBody) -> LoopBody:
    """Merge contiguous loop dims (x group, then reduction group)."""

    body = _merge_group(body, reduction=False)
    if body.rvars:
        body = _merge_group(body, reduction=True)
    return body


def iter_values(root) -> list[Value]:
    """Nodes of a recorded body in dependency order."""

    order: list[Value] = []
    seen: set[int] = set()
    stack = list(root) if isinstance(root, tuple) else [root]
    stack = [(v, False) for v in stack]
    while stack:
        v, expanded = stack.pop()
        if not isinstance(v, Value) or (id(v) in seen and not expanded):
            continue
        if expanded:
            if id(v) not in seen:
                seen.add(id(v))
                order.append(v)
            continue
        stack.append((v, True))
        for a in v.args:
            if isinstance(a, Value) and id(a) not in seen:
                stack.append((a, False))
    return order


__all__ = [
    "Buffer", "ComputedBuffer", "Const", "ConstantBuffer", "DeferredOps", "Expr",
    "ExternKernel", "ExternOutput", "InputBuffer", "IRNode", "Layout",
    "LoopBody", "Loops", "Pointwise", "Reduction", "ReinterpretView", "Symbol",
    "TensorBox", "V", "Value", "View", "affine_coeff", "as_index",
    "PROMOTED_ON_LOAD", "compute_dtype", "contiguous_strides", "dtype_name",
    "floordiv", "promotes_on_load",
    "free_symbols", "fresh_symbols", "iter_values", "modular_indexing", "ops",
    "pexpr", "prod", "record_body", "set_graph", "set_ops_handler",
    "simplify_index", "simplify_loops", "substitute",
]
