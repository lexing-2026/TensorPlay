"""Triton kernels for the whole-graph loop IR.

A scheduled group of loop nests becomes one kernel.  The x loop is always a
flat range over ``xnumel``; the r loop is either persistent (one block spans a
whole row) or a strided loop over the row.  A value computed earlier in the
group stays in a register, so a fused group never round-trips its
intermediates, and a value nobody outside the group reads is not stored at
all.

Source generation follows the repository's existing program-codegen
conventions: the text is content addressed, cached on disk, installed into
``linecache`` and exec'd once, and the resulting launcher replays the
compiled binary directly for as long as its guards hold.
"""

from __future__ import annotations

import hashlib
import linecache
import textwrap
from typing import Any, NamedTuple

from .index_expr import (
    Add,
    Const,
    Expr,
    FloorDiv,
    ModularIndexing,
    Mul,
    Symbol,
    Where,
    free_symbols,
)
from ..kernel_scheduler import (
    BODY as _BODY,
    EPILOGUE as _EPILOGUE,
    PROLOGUE as _PROLOGUE,
    RINDEX,
    XINDEX,
    FusedGroup,
    LoopNode,
    emission_regions,
    placement,
)
from ..loops import Buffer, Value, dtype_name

try:  # pragma: no cover - availability is a runtime condition
    import triton

    HAS_TRITON = True
except Exception:  # noqa: BLE001
    triton = None
    HAS_TRITON = False


_TL_DTYPES = {
    "float16": "tl.float16",
    "bfloat16": "tl.bfloat16",
    "float32": "tl.float32",
    "float64": "tl.float64",
    "int64": "tl.int64",
    "int32": "tl.int32",
    "int16": "tl.int16",
    "int8": "tl.int8",
    "uint8": "tl.uint8",
    "bool": "tl.int1",
}

_WELFORD_HELPERS = '''

@triton.jit
def _tp_welford_combine(mean_1, m2_1, weight_1, mean_2, m2_2, weight_2):
    delta = tl.where(mean_1 == mean_2, 0.0, mean_2 - mean_1)
    new_weight = weight_1 + weight_2
    w2_over_w = tl.where(new_weight == 0.0, 0.0, weight_2 / new_weight)
    return (
        mean_1 + delta * w2_over_w,
        m2_1 + m2_2 + delta * delta * weight_1 * w2_over_w,
        new_weight,
    )


@triton.jit
def _tp_welford(value, m2, weight, dim: tl.constexpr):
    return tl.reduce((value, m2, weight), dim, _tp_welford_combine)
'''




#: Floating reductions accumulate here; the store rounds back to storage.
_ACC_DTYPE = "tl.float32"


class PlanError(Exception):
    """A group cannot be expressed as a single kernel."""


class LaunchConfig(NamedTuple):
    """Block sizes, warps, and whether one block spans a whole row."""

    xblock: int
    rblock: int
    num_warps: int
    num_stages: int = 2
    persistent: bool = False

    def key(self) -> tuple:
        return tuple(self)


#: Reduction sizes that fit one block, split by whether the reduced axis is
#: the contiguous one.  A persistent reduction forces the r block to span the
#: whole row, so a long inner row tiles differently than a short outer one.
_PERSISTENT_INNER = 1024
_PERSISTENT_OUTER = 64

#: A persistent reduction keeps the whole row in one block, so its r block is
#: the row length; these bound how many rows may share the block.
_PERSISTENT_XBLOCKS = (1, 8, 32, 128)
_PERSISTENT_TILE_NUMEL = 4096

#: A long inner row is walked in steps, with a narrow x block: the reduced
#: elements are already contiguous, so width buys coalescing on the x axis
#: while depth buys the per-thread element count the row needs.
_INNER_XBLOCK_NUMEL = 1024
_INNER_MAX_XBLOCK = 8

#: Elements one warp is given before the warp count grows.
_ELEMENTS_PER_WARP = 128
#: A block is never given fewer warps than this once it holds this many
#: elements, whatever the element count asks for.
_MIN_WARPS = 4
_WARP_SIZE = 32

#: Pointwise candidates: block width, tried in this order.
_XBLOCK_CANDIDATES = (256, 512, 1024, 128, 2048)


def next_power_of_two(value: int) -> int:
    value = max(int(value), 1)
    return 1 << (value - 1).bit_length()


def _num_warps(requested: int, max_num_warps: int, *, register_intensive: bool = False) -> int:
    """Round a requested warp count to a power of two within bounds.

    A persistent reduction is register intensive -- it holds the row's running
    statistics -- so it gets half the warp budget, which trades threads for
    registers per thread.
    """

    if register_intensive:
        max_num_warps = max(max_num_warps // 2, 1)
    return next_power_of_two(min(max(requested, 1), max_num_warps))


def _reduction_warps(rnumel: int, total: int, inner: bool, register_intensive: bool) -> int:
    """Warps for a reduction tile.

    An inner row is contiguous, so each thread is given at least eight
    elements; otherwise the count follows the whole tile.
    """

    requested = rnumel // _ELEMENTS_PER_WARP if inner else total // _ELEMENTS_PER_WARP
    ceiling = 16 if rnumel <= 8192 else 32
    warps = _num_warps(requested, ceiling, register_intensive=register_intensive)
    if total >= 128:
        warps = max(warps, _MIN_WARPS)
    return warps


def _persistent_configs(xnumel: int, rnumel: int, inner: bool) -> list:
    """Configs for a row that one block spans.

    The r block is the row, so the x block may only be as wide as the tile
    budget allows -- except at one row per block, which is always legal.
    """

    out = []
    for xblock in _PERSISTENT_XBLOCKS:
        if xblock != 1 and (rnumel * xblock > _PERSISTENT_TILE_NUMEL or xblock > xnumel):
            continue
        warps = _reduction_warps(
            rnumel, xblock * rnumel, inner, register_intensive=True
        )
        out.append(
            LaunchConfig(xblock, next_power_of_two(rnumel), warps, persistent=True)
        )
    return out


def _outer_config(xnumel: int, rnumel: int, load_factor: int) -> tuple:
    """Block shape for a reduction over the outer axis.

    With few rows the x block stays narrow and the row is walked in small
    steps; with many rows the width grows first, and a wide block only pairs
    with a short step when the body is light.
    """

    if xnumel <= 1024:
        return max(min(xnumel // 128, 8), 2), min(rnumel, 64), None
    if xnumel // 4096 <= 8:
        return 16, 512 // 16, None
    xblock = max(min(256, next_power_of_two(xnumel // 4096)), 64)
    if load_factor < 4 or rnumel <= 128:
        return xblock, max(512 // xblock, 1), None
    if rnumel >= 2048:
        rblock = 64
    else:
        rblock = 32
    return min(xblock, 32), rblock, _MIN_WARPS


def _inner_config(xnumel: int, rnumel: int) -> tuple:
    """Block shape for a reduction over the contiguous axis."""

    return max(min(_INNER_XBLOCK_NUMEL // rnumel, _INNER_MAX_XBLOCK), 1), rnumel, 1


def config_candidates(group: FusedGroup, buffers: dict) -> list:
    """Launch configs worth trying, best guess first.

    Pointwise work only needs a block width.  A reduction picks between
    keeping the row in one block and walking it, and in both cases the x
    block is never wider than the rows it has to fill.
    """

    xnumel = max(int(group.xnumel), 1)
    rnumel = int(group.rnumel)
    if rnumel <= 1:
        out = []
        for xblock in _XBLOCK_CANDIDATES:
            warps = _num_warps(xblock // _ELEMENTS_PER_WARP, 16)
            if xblock >= 128:
                warps = max(warps, _MIN_WARPS)
            out.append(LaunchConfig(xblock, 1, warps))
        return out
    inner = any(
        _is_inner_reduction(node, buffers) for node in group.nodes if node.is_reduction
    )
    span = next_power_of_two(rnumel)
    if rnumel <= (_PERSISTENT_INNER if inner else _PERSISTENT_OUTER):
        configs = _persistent_configs(xnumel, rnumel, inner)
        if configs:
            return configs
    load_factor = sum(len(node.body.loads) for node in group.nodes)
    if inner and span >= 256:
        # A contiguous row only pays for a persistent block while the rows
        # are numerous; otherwise the row is walked with a narrow x block.
        if span <= 1024 and xnumel // 8 >= 128:
            xblock, rblock, warps = _inner_config(xnumel, rnumel)
        else:
            xblock, rblock, warps = 1, min(span, 1024), None
    else:
        xblock, rblock, warps = _outer_config(xnumel, rnumel, load_factor)
    xblock = max(1, min(xblock, xnumel))
    rblock = max(1, min(rblock, span))
    if warps is None:
        warps = _reduction_warps(rblock, xblock * rblock, inner, register_intensive=False)
    return [LaunchConfig(xblock, rblock, warps)]


def _is_float(name: str) -> bool:
    return name in ("float16", "bfloat16", "float32", "float64")


def _tl_dtype(name: str) -> str:
    mapped = _TL_DTYPES.get(name)
    if mapped is None:
        raise PlanError(f"unsupported element type: {name}")
    return mapped


def _zero(name: str) -> str:
    if _is_float(name):
        return "0.0"
    if name == "bool":
        return "False"
    return "0"


def _is_inner_reduction(node: LoopNode, buffers: dict) -> bool:
    """Does the reduced axis walk contiguous elements of a real buffer?"""

    for load in node.body.loads:
        buffer = buffers.get(load.args[0])
        layout = getattr(buffer, "layout", None)
        if layout is None or not layout.size or not layout.stride:
            continue
        return int(layout.stride[-1]) == 1
    return False


# ---------------------------------------------------------------------------
# index rendering
# ---------------------------------------------------------------------------


def render_index(expr: Expr, xvar: str, rvar: str | None) -> str:
    """Render kernel index arithmetic.

    Loop indices are non-negative, so the algebra's floor division is the
    truncating one.  The piecewise form only appears when a division could
    not be folded away against the loop extents.
    """

    if isinstance(expr, Const):
        return str(expr.value)
    if isinstance(expr, Symbol):
        if expr is XINDEX:
            return xvar
        if expr is RINDEX:
            if rvar is None:
                raise PlanError("a reduction index outside a reduction loop")
            return rvar
        return expr.name
    if isinstance(expr, Add):
        parts = []
        for term, coeff in expr.terms.items():
            rendered = render_index(term, xvar, rvar)
            if coeff == 1:
                parts.append(rendered)
            elif coeff == -1:
                parts.append(f"-({rendered})")
            else:
                parts.append(f"{coeff}*({rendered})")
        if expr.offset:
            parts.append(str(expr.offset))
        return "(" + " + ".join(parts) + ")" if parts else "0"
    if isinstance(expr, Mul):
        return f"{expr.scalar}*({render_index(expr.operand, xvar, rvar)})"
    if isinstance(expr, FloorDiv):
        numerator = render_index(expr.numerator, xvar, rvar)
        if expr.divisor == 1:
            return numerator
        return f"(({numerator}) // {expr.divisor})"
    if isinstance(expr, ModularIndexing):
        base = render_index(expr.base, xvar, rvar)
        if expr.divisor != 1:
            base = f"(({base}) // {expr.divisor})"
        return f"(({base}) % {expr.modulus})"
    if isinstance(expr, Where):
        return (
            f"tl.where({render_index(expr.condition, xvar, rvar)}, "
            f"{render_index(expr.left, xvar, rvar)}, "
            f"{render_index(expr.right, xvar, rvar)})"
        )
    raise PlanError(f"unrenderable index expression: {expr!r}")


class _Group:
    """Emits the source of one fused kernel."""

    def __init__(self, group: FusedGroup, buffers: dict, stored: set, config: LaunchConfig):
        self.group = group
        self.buffers = buffers
        self.stored = stored
        self.config = config
        self.counter = 0
        self.sink: list = []
        self.temps: dict = {}  # buffer name -> (variable, dim)
        self.loads: dict = {}  # (buffer, address, mask) -> variable
        self.ptr_order: list = []
        self.placements = {}
        for node in group.nodes:
            place = placement(node, group.xnumel, group.rnumel)
            if place is None:
                raise PlanError("a fused node does not fit the kernel iteration space")
            self.placements[id(node)] = place
        self.produced = {}
        for node in group.nodes:
            for buffer in node.buffers:
                self.produced[buffer.name] = node
        self.regions = self._classify()
        # accumulator state, filled while emitting
        self.elem: dict = {}
        self.acc: dict = {}
        self.welford_state: dict = {}
        # statement buffers
        self.prologue: list = []
        self.pre_loop: list = []
        self.loop: list = []
        self.post: list = []
        self.epilogue: list = []

    # -- structure ---------------------------------------------------------
    def _classify(self) -> dict:
        written = {}
        for node in self.group.nodes:
            for buffer in node.buffers:
                written[buffer.name] = (node, buffer)
        return emission_regions(self.group.nodes, self.placements, written)

    def _in_region(self, region: int) -> list:
        """A region's nodes, in the order the scheduler made them legal."""

        return sorted(
            (node for node in self.group.nodes if self.regions[id(node)] == region),
            key=lambda node: node.index,
        )

    def run(self) -> None:
        for node in self._in_region(_PROLOGUE):
            self.sink = self.prologue
            self.emit_node(node, 1)
        for node in self.group.nodes:
            if node.is_reduction:
                self.sink = self.pre_loop
                self.acc_init(node)
        for node in self._in_region(_BODY):
            self.sink = self.loop
            if node.is_reduction:
                self.elem[id(node)] = self.value(node.body.root, node, 2)
                self.acc_step(node)
            else:
                self.emit_node(node, 2)
        for node in self.group.nodes:
            if node.is_reduction:
                self.sink = self.post
                self.acc_finish(node)
        for node in self._in_region(_EPILOGUE):
            self.sink = self.epilogue
            self.emit_node(node, 1)

    # -- emission helpers --------------------------------------------------
    def emit(self, line: str) -> None:
        self.sink.append(line)

    def tmp(self, hint: str = "t") -> str:
        self.counter += 1
        return f"{hint}{self.counter}"

    def ptr(self, name: str) -> str:
        if name not in self.ptr_order:
            self.ptr_order.append(name)
        return f"p{self.ptr_order.index(name)}"

    def index(self, node: LoopNode, expr: Expr, dim: int) -> str:
        placed = self.placements[id(node)].to_kernel(expr)
        rendered = render_index(
            placed, "xindex" if dim == 1 else "xi", "ri" if dim == 2 else None
        )
        if not free_symbols(placed):
            # The whole group addresses one spot, so the address folded to a
            # constant while the mask is still a block: widen it to match.
            return f"({rendered} + tl.zeros({self.block_shape(dim)}, tl.int32))"
        return rendered

    def cond(self, dim: int) -> str | None:
        parts = []
        if self.needs_xmask:
            parts.append("xmask" if dim == 1 else "xmask2")
        if dim == 2 and self.needs_rmask:
            parts.append("rmask2")
        return " & ".join(parts) if parts else None

    @property
    def persistent(self) -> bool:
        return self.config.persistent

    def block_shape(self, dim: int = 2) -> str:
        return "[XBLOCK, RBLOCK]" if dim == 2 else "[XBLOCK]"

    @property
    def needs_xmask(self) -> bool:
        return int(self.group.xnumel) % self.config.xblock != 0

    @property
    def needs_rmask(self) -> bool:
        return int(self.group.rnumel) != int(self.config.rblock)

    def guarded(self, source: str, dim: int, limit: str) -> str:
        cond = self.cond(dim)
        if cond is None:
            return source
        return f"tl.where({cond}, {source}, {limit})"

    # -- values ------------------------------------------------------------
    def operand(self, value: Value, node: LoopNode, dim: int, want: str) -> str:
        """An operand at the type the operation is declared to compute in.

        Arithmetic runs in the wider float type of the pair, so a half input
        is widened here instead of silently computing in half precision.
        """

        source = self.value(value, node, dim)
        if value.dtype == want:
            return source
        return f"{source}.to({_tl_dtype(want)})"

    def value(self, value: Value, node: LoopNode, dim: int) -> str:
        if not isinstance(value, Value):
            # A lowering that produced no expression for this nest; the region
            # falls back rather than emitting a kernel that reads nothing.
            raise PlanError(
                f"loop body holds {type(value).__name__} instead of a value"
            )
        op = value.op
        args = value.args
        if op == "load":
            return self.load(value, node, dim)
        if op == "constant":
            (number,) = args
            if isinstance(number, bool):
                return "True" if number else "False"
            if isinstance(number, int):
                return str(number)
            return repr(float(number))
        if op == "index_expr":
            return self.index(node, args[0], dim)
        if op == "to_dtype":
            return f"{self.value(args[0], node, dim)}.to({_tl_dtype(value.dtype)})"
        if op == "where":
            return (
                f"tl.where({self.value(args[0], node, dim)}, "
                f"{self.operand(args[1], node, dim, value.dtype)}, "
                f"{self.operand(args[2], node, dim, value.dtype)})"
            )
        if op == "and_":
            return (
                f"({self.value(args[0], node, dim)} & "
                f"{self.value(args[1], node, dim)})"
            )
        if op in ("add", "sub", "mul", "truediv", "maximum", "minimum", "pow"):
            left = self.operand(args[0], node, dim, value.dtype)
            right = self.operand(args[1], node, dim, value.dtype)
            if op == "add":
                return f"({left} + {right})"
            if op == "sub":
                return f"({left} - {right})"
            if op == "mul":
                return f"({left} * {right})"
            if op == "truediv":
                return f"({left} / {right})"
            if op == "maximum":
                return f"tl.maximum({left}, {right})"
            if op == "minimum":
                return f"tl.minimum({left}, {right})"
            return f"libdevice.pow({left}, {right})"
        if op in ("lt", "le", "gt", "ge", "eq", "ne"):
            symbol = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}[op]
            # A comparison keeps its operand types; only its result is boolean.
            left = self.value(args[0], node, dim)
            right = self.value(args[1], node, dim)
            return f"({left} {symbol} {right})"
        return self.unary(value, node, dim)

    def unary(self, value: Value, node: LoopNode, dim: int) -> str:
        op = value.op
        inner = self.operand(value.args[0], node, dim, value.dtype)
        if op == "neg":
            return f"(-({inner}))"
        if op == "exp":
            return f"tl.exp({inner})"
        if op == "log":
            return f"tl.log({inner})"
        if op == "sqrt":
            return f"tl.sqrt({inner})"
        if op == "rsqrt":
            return f"libdevice.rsqrt({inner})"
        if op == "sigmoid":
            return f"(1.0 / (1.0 + tl.exp(-({inner}))))"
        if op == "reciprocal":
            return f"(1.0 / {inner})"
        if op == "abs":
            return f"tl.abs({inner})"
        if op == "sin":
            return f"tl.sin({inner})"
        if op == "cos":
            return f"tl.cos({inner})"
        if op == "tanh":
            return f"libdevice.tanh({inner})"
        if op == "relu":
            return f"tl.maximum({inner}, {_zero(value.dtype)})"
        if op == "square":
            return f"({inner} * {inner})"
        raise PlanError(f"unrenderable value operation: {op}")

    def load(self, value: Value, node: LoopNode, dim: int) -> str:
        name, expr, mask = value.args
        temp = self.temps.get(name)
        if temp is not None:
            variable, temp_dim = temp
            return variable if temp_dim == dim else f"{variable}[:, None]"
        buffer = self.buffers.get(name)
        if buffer is None:
            raise PlanError(f"read of an unknown buffer: {name}")
        address = self.index(node, expr, dim)
        conditions = []
        base = self.cond(dim)
        if base:
            conditions.append(base)
        if mask is not None:
            conditions.append(self.value(mask, node, dim))
        cond = " & ".join(conditions) if conditions else None
        # The same element read twice in one body is one load.
        key = (name, address, cond)
        shared = self.loads.get(key)
        if shared is not None:
            return shared
        variable = self.tmp("in")
        if cond is None:
            self.emit(f"{variable} = tl.load({self.ptr(name)} + {address})")
        else:
            other = _zero(dtype_name(buffer.get_dtype()))
            self.emit(
                f"{variable} = tl.load({self.ptr(name)} + {address}, "
                f"mask={cond}, other={other})"
            )
        self.loads[key] = variable
        return variable

    def store(self, buffer: Buffer, node: LoopNode, dim: int, source: str) -> str:
        """Write one value, narrowing to the buffer's element type."""

        address = self.index(node, node.body.stores[buffer.name], dim)
        element = _tl_dtype(dtype_name(buffer.get_dtype()))
        variable = self.tmp("o")
        if element != _ACC_DTYPE and element != "tl.int1":
            self.emit(f"{variable} = {source}.to({element})")
        else:
            self.emit(f"{variable} = {source}")
        cond = self.cond(dim)
        if cond is None:
            self.emit(f"tl.store({self.ptr(buffer.name)} + {address}, {variable})")
        else:
            self.emit(
                f"tl.store({self.ptr(buffer.name)} + {address}, {variable}, mask={cond})"
            )
        return variable

    # -- nodes -------------------------------------------------------------
    def emit_node(self, node: LoopNode, dim: int) -> str:
        roots = node.body.root if isinstance(node.body.root, tuple) else (node.body.root,)
        produced = [self.value(root, node, dim) for root in roots]
        for buffer, source in zip(node.buffers, produced):
            # An unstored value is still a register: only the write is skipped.
            variable = (
                self.store(buffer, node, dim, source)
                if buffer.name in self.stored
                else source
            )
            self.temps[buffer.name] = (variable, dim)
        return produced[0]

    # -- reductions --------------------------------------------------------
    def acc_init(self, node: LoopNode) -> None:
        loops = node.data
        kind = loops.reduction_type
        element = dtype_name(loops.dtype)
        acc = _ACC_DTYPE if _is_float(element) else _tl_dtype(element)
        # A looped reduction accumulates a whole block at a time and reduces
        # once the row is walked, so its accumulator is block shaped.
        shape = self.block_shape(2) if not self.persistent else "[XBLOCK]"
        state = {"kind": kind, "acc": acc, "name": self.tmp("acc"), "shape": shape}
        if kind == "sum":
            if not self.persistent:
                self.emit(f"{state['name']} = tl.zeros({shape}, {acc})")
        elif kind in ("max", "min"):
            if _is_float(element):
                limit = "-float('inf')" if kind == "max" else "float('inf')"
            else:
                limit = "-2147483647" if kind == "max" else "2147483647"
            state["limit"] = limit
            if not self.persistent:
                self.emit(f"{state['name']} = tl.full({shape}, {limit}, {acc})")
        elif kind == "welford":
            state["m2"] = self.tmp("acc")
            state["weight"] = self.tmp("acc")
            if not self.persistent:
                for name in (state["name"], state["m2"], state["weight"]):
                    self.emit(f"{name} = tl.zeros([XBLOCK], {acc})")
        else:
            raise PlanError(f"unsupported reduction: {kind}")
        self.acc[id(node)] = state

    def acc_step(self, node: LoopNode) -> None:
        """One step of the reduction, emitted inside the r loop."""

        state = self.acc[id(node)]
        source = self.elem[id(node)]
        acc = state["acc"]
        kind = state["kind"]
        if self.persistent:
            # One block spans the row, so the reduction happens right here.
            if kind == "sum":
                state["value"] = f"tl.sum({source}.to({acc}), 1)"
            elif kind in ("max", "min"):
                guarded = self.guarded(source, 2, state["limit"])
                state["value"] = f"tl.{kind}({guarded}, 1)"
            else:
                self._welford_block(source, state)
            return
        if kind == "sum":
            self.emit(f"{state['name']} += {source}.to({acc})")
        elif kind in ("max", "min"):
            self.emit(f"{state['name']} = tl.{kind}({state['name']}, {source})")
        else:
            self._welford_block(source, state)
            block = state["block"]
            merged = self.tmp("cm"), self.tmp("cs"), self.tmp("cw")
            self.emit(
                f"{merged[0]}, {merged[1]}, {merged[2]} = _tp_welford_combine("
                f"{state['name']}, {state['m2']}, {state['weight']}, "
                f"{block[0]}, {block[1]}, {block[2]})"
            )
            # The accumulators live outside the loop, so the merge is written
            # back into them: that is what carries the running statistics.
            self.emit(f"{state['name']} = {merged[0]}")
            self.emit(f"{state['m2']} = {merged[1]}")
            self.emit(f"{state['weight']} = {merged[2]}")

    def _welford_block(self, source: str, state: dict) -> None:
        """Reduce one block's worth of elements to (mean, m2, weight)."""

        block = (self.tmp("bm"), self.tmp("bs"), self.tmp("bw"))
        self.emit(
            f"{block[0]}, {block[1]}, {block[2]} = _tp_welford({source}, "
            f"tl.zeros_like({source}), {self.weight(source, 2)}, 1)"
        )
        state["block"] = block
        if self.persistent:
            state["name"], state["m2"] = block[0], block[1]

    def weight(self, source: str, dim: int) -> str:
        """Per-element weight: one where the element is real, zero elsewhere.

        The weight is widened to the element's block shape, because a reduced
        value set has to agree in shape across its members.
        """

        cond = self.cond(dim)
        if cond is None:
            return f"({source} * 0.0 + 1.0)"
        return f"tl.where({cond}, 1.0, 0.0) + tl.zeros({self.block_shape(dim)}, tl.float32)"

    def acc_finish(self, node: LoopNode) -> None:
        """Reduce the last step and store every result that outlives the group."""

        state = self.acc[id(node)]
        kind = state["kind"]
        if kind == "welford":
            # Welford reduces inside the loop and combines across steps, so
            # its results are ready as soon as the loop ends.
            self.store_reduction(node, 0, state["name"])
            self.store_reduction(node, 1, state["m2"])
            return
        if self.persistent:
            value = state["value"]
        elif kind == "sum":
            value = f"tl.sum({state['name']}, 1)"
        else:
            value = f"tl.{kind}({state['name']}, 1)"
        self.store_reduction(node, 0, value)

    def store_reduction(self, node: LoopNode, position: int, value: str) -> None:
        """Bind a reduction result to a name; store it only when it outlives."""

        buffers = node.buffers
        if position >= len(buffers):
            return
        buffer = buffers[position]
        if buffer.name in self.stored:
            self.temps[buffer.name] = (self.store(buffer, node, 1, value), 1)
            return
        variable = self.tmp("red")
        self.emit(f"{variable} = {value}")
        self.temps[buffer.name] = (variable, 1)


def loops_dtype(node: LoopNode) -> str:
    return dtype_name(node.data.dtype)


# ---------------------------------------------------------------------------
# kernel assembly
# ---------------------------------------------------------------------------

_PREAMBLE = (
    "import triton\n"
    "import triton.language as tl\n"
    "import triton.language.extra.cuda.libdevice as libdevice\n"
    "from tensorplay._stax.runtime import fastlaunch as _fl\n"
)


def emit_group_source(group: FusedGroup, buffers: dict, stored: set, config: LaunchConfig) -> tuple:
    """Return ``(kernel_name, source, pointer_names)`` for one fused group."""

    g = _Group(group, buffers, stored, config)
    g.run()
    xnumel = int(group.xnumel)
    rnumel = int(group.rnumel)
    has_r = rnumel > 1
    head = [
        "xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)",
        "xi = xindex[:, None]",
    ]
    if g.needs_xmask:
        head.append("xmask = xindex < xnumel")
        head.append("xmask2 = xmask[:, None]")
    body = list(head) + g.prologue
    if has_r:
        body += g.pre_loop
        if g.persistent:
            body.append("rindex = tl.arange(0, RBLOCK)")
            body.append("ri = rindex[None, :]")
            if g.needs_rmask:
                body.append("rmask2 = rindex[None, :] < rnumel")
            body += g.loop
        else:
            body.append("for r0 in range(0, rnumel, RBLOCK):")
            body.append("    rindex = r0 + tl.arange(0, RBLOCK)")
            body.append("    ri = rindex[None, :]")
            if g.needs_rmask:
                body.append("    rmask2 = rindex[None, :] < rnumel")
            body += textwrap.indent("\n".join(g.loop), "    ").splitlines()
        body += g.post
    body += g.epilogue
    # The kernel is named after its own body, so two groups that differ in any
    # way get different names and a cached launcher is never handed to a group
    # it was not generated for.
    digest = hashlib.sha1(
        "\n".join(body).encode()
        + repr(tuple(config.key())).encode()
        + repr(sorted(g.ptr_order)).encode()
    ).hexdigest()[:12]
    kernel_name = f"stax_loop_{digest}"
    signature = [f"p{position}" for position in range(len(g.ptr_order))] + ["xnumel"]
    if has_r:
        signature.append("rnumel")
    signature.append("XBLOCK: tl.constexpr")
    if has_r:
        signature.append("RBLOCK: tl.constexpr")
    source = _PREAMBLE
    if any(node.is_reduction for node in group.nodes):
        source += _WELFORD_HELPERS
    source += "\n\n@triton.jit\n"
    source += f"def {kernel_name}({', '.join(signature)}):\n"
    source += textwrap.indent("\n".join(body), "    ") + "\n"
    source += _launcher_source(kernel_name, g, config, xnumel, rnumel)
    return kernel_name, source, list(g.ptr_order)


def _launcher_source(
    kernel_name: str, g: _Group, config: LaunchConfig, xnumel: int, rnumel: int
) -> str:
    """Emit the launcher, with the repository's direct-replay fast path.

    ``xnumel``/``rnumel`` are pinned here, so every call of this launcher hits
    the same specialization; the recorded binary is replayed directly while
    pointer alignment and the absence of launch hooks still hold.
    """

    has_r = rnumel > 1
    call_args = [f"ptrs[{i}]" for i in range(len(g.ptr_order))]
    call_args.append("xnumel")
    if has_r:
        call_args.append("rnumel")
    args_txt = ", ".join(call_args)
    live_count = len(g.ptr_order) + 1 + (1 if has_r else 0)
    full_count = live_count + 1 + (1 if has_r else 0)
    if g.ptr_order:
        pointer_guard = "(" + " | ".join(
            f"ptrs[{i}].data_ptr()" for i in range(len(g.ptr_order))
        ) + ") % 16 == 0"
    else:
        pointer_guard = "True"
    grid = f"triton.cdiv(xnumel, {config.xblock})"
    consts = [f"XBLOCK={config.xblock}"]
    if has_r:
        consts.append(f"RBLOCK={config.rblock}")
    consts.append(f"num_warps={config.num_warps}")
    if has_r and not g.persistent:
        consts.append(f"num_stages={config.num_stages}")
    const_txt = ", ".join(consts)
    lines = [
        "",
        "_rec = None",
        "",
        "",
        "def kernel_launch(ptrs):",
        "    global _rec",
        f"    xnumel = {xnumel}",
    ]
    if has_r:
        lines.append(f"    rnumel = {rnumel}")
    lines += [
        "    _r = _rec",
        f"    if _r is not None and {pointer_guard} and _fl.hooks_clear():",
        "        try:",
        "            _s = _fl.current_stream()",
        f"            _r[0]({grid}, 1, 1, _s, _r[1], _r[2], None, None, None, {args_txt})",
        "            _fl.bump()",
        "            return",
        "        except Exception:",
        "            _rec = None",
        "    _snap = -1",
        f"    if _r is None and _fl.hooks_clear() and {pointer_guard}:",
        f"        _snap = _fl.cache_size({kernel_name})",
        f"    {kernel_name}[({grid},)]({args_txt}, {const_txt})",
        "    if _snap >= 0:",
        f"        _g = _fl.take_kernel({kernel_name}, _snap)",
        "        if _g is not None:",
        f"            _g = _fl.native_wrap({kernel_name}, _snap, {full_count}) or _g",
        "            _rec = _g + (xnumel,)",
    ]
    return "\n".join(lines) + "\n"


_launch_memo: dict = {}


def compile_group(
    group: FusedGroup,
    buffers: dict,
    stored: set,
    config: LaunchConfig,
) -> tuple:
    """Emit, cache and exec one kernel; returns ``(launcher, pointer_names)``."""

    if not HAS_TRITON:
        raise PlanError("Triton is not available")
    kernel_name, source, ptr_names = emit_group_source(group, buffers, stored, config)
    # The name is a digest of the body, the config and the pointers, so a
    # cached launcher can only ever be reused for an identical kernel.
    hit = _launch_memo.get(kernel_name)
    if hit is not None:
        return hit
    try:
        from ..codecache import default_cache

        cache = default_cache("triton")
        cache_key = cache.cache_key(source)
        if cache.load(cache_key, ext="py") is None:
            cache.store(cache_key, source.encode(), ext="py")
    except Exception:  # noqa: BLE001 - the on-disk cache is best effort
        pass
    fake_file = f"<tensorplay-stax-loop-{kernel_name}>"
    linecache.cache[fake_file] = (
        len(source),
        None,
        source.splitlines(True),
        fake_file,
    )
    # ``__name__`` matters: a function defined in an exec'd namespace without
    # one gets ``__module__ = None``, which the code generator cannot classify.
    namespace: dict = {"__name__": "tensorplay_stax_loop", "triton": triton}
    exec(compile(source, fake_file, "exec"), namespace, namespace)
    result = (namespace["kernel_launch"], ptr_names)
    _launch_memo[kernel_name] = result
    return result


__all__ = [
    "LaunchConfig",
    "PlanError",
    "compile_group",
    "config_candidates",
    "emit_group_source",
    "next_power_of_two",
    "render_index",
]
