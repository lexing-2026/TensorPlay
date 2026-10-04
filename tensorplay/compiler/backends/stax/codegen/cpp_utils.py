"""How the host language spells the things a kernel is made of.

A generated kernel is text, and the text has to agree with the headers it
includes: an element type has to be the type that header defines, an index has
to be the width the indexing arithmetic assumes, and an expression has to
print with the precedence the language gives it.  All three are stated here
once, because getting one of them wrong produces a kernel that compiles and
computes the wrong thing, which is worse than one that does not compile.
"""

from __future__ import annotations

from collections import namedtuple
from typing import Any, Callable, Sequence
from unittest.mock import patch

import contextlib
import dataclasses
import functools
import math

import sympy
from sympy.printing.cxx import CXX11CodePrinter

import tensorplay as tp
from tensorplay.primitives.common import is_integer_dtype

from tensorplay.graph.experimental.sympy_functions import OrderedSet
from .. import ir
from ..dependencies import Dep
from ..scheduler import BaseSchedulerNode, SchedulerBuffer
from ..loops import V
from ..virtualized import ops
from ..loop_body import LoopBody
from ..utils import sympy_index_symbol_with_prefix, sympy_subs
from tensorplay.graph.experimental.sympy_functions import SymT, symbol_is_type
from ..ops_handler import WrapperHandler
from ..virtualized import OpsValue
from .common import CSEVariable, Kernel, KernelArgs, OptimizationContext

#: The element type each dtype is written as.  A type absent from this table
#: is a type the generated code cannot hold, so a kernel that needs one fails
#: here rather than emitting a name the compiler has never heard of.
DTYPE_TO_CPP = {
    tp.float32: "float",
    tp.float64: "double",
    tp.float16: "tensorplay::Half",
    tp.int64: "int64_t",
    tp.int32: "int32_t",
    tp.int16: "int16_t",
    tp.int8: "int8_t",
    tp.uint64: "uint64_t",
    tp.uint32: "uint32_t",
    tp.uint16: "uint16_t",
    tp.uint8: "uint8_t",
    tp.bool: "bool",
    tp.bfloat16: "tensorplay::BFloat16",
    tp.complex32: "tensorplay::complex<tensorplay::Half>",
    tp.bcomplex32: "tensorplay::complex<tensorplay::BFloat16>",
    tp.complex64: "tensorplay::complex<float>",
    tp.complex128: "tensorplay::complex<double>",
    tp.float8_e4m3fn: "tensorplay::Float8_e4m3fn",
    tp.float8_e5m2: "tensorplay::Float8_e5m2",
    tp.float8_e4m3fnuz: "tensorplay::Float8_e4m3fnuz",
    tp.float8_e5m2fnuz: "tensorplay::Float8_e5m2fnuz",
    tp.float8_e8m0fnu: "tensorplay::Float8_e8m0fnu",
}

#: The width an index is written as.  A kernel is handed its sizes as 64-bit
#: values, so an index computed at 32 bits would be wrong for a tensor larger
#: than it can address.
INDEX_TYPE = "int64_t"

#: The three extents of a product, named.  Which is which matters and a tuple of
#: three numbers does not say, so a product is described by name rather than by
#: position -- and the name is what the kernels that were written for one shape
#: of this are looking for.
GemmBlocking = namedtuple("GemmBlocking", ["block_m", "block_n", "block_k"])


class CppCSEVariable(CSEVariable):
    """A named expression, with what the host emitter needs to know about it.

    Two things are recorded that a plain name does not carry.  Whether the
    expression is a whole vector rather than a single element, which decides
    whether it can be loaded with a vector instruction at all; and which loop
    variables it was built from, which is how an expression is known to belong
    to one loop nest rather than to the kernel as a whole, and can therefore be
    hoisted out of that nest rather than recomputed in it.
    """

    def __init__(self, name, bounds, dtype=None, shape=None) -> None:
        super().__init__(name, bounds, dtype, shape=shape)
        self.is_vec = False
        self.dependent_itervars: OrderedSet = OrderedSet()

    def __repr__(self) -> str:
        return (
            f"CppCSEVariable(name: {self.name}, bounds: {self.bounds}, is_vec: {self.is_vec}, "
            f"dtype: {self.dtype}, dependent_itervars: {self.dependent_itervars})"
        )

    def update_on_args(self, name, args, kwargs) -> None:
        """Record, from the call that produced this expression, what it is made of."""

        if name == "load":
            self._set_dependent_itervars(args[2])
        else:
            # An expression is as vector-valued and as nest-dependent as the
            # expressions it was built from, so both are inherited rather than
            # asked about again.
            self.dependent_itervars.update(
                *[
                    arg.dependent_itervars
                    for arg in args
                    if isinstance(arg, CppCSEVariable)
                ]
            )
            if name in ("index_expr", "value_expr"):
                self._set_dependent_itervars(args[0])
            if any(arg.is_vec for arg in args if isinstance(arg, CppCSEVariable)):
                self.is_vec = True

    def _set_dependent_itervars(self, index):
        """The loop variables this expression reads, directly or by naming one.

        A symbol that names another expression rather than a loop variable is
        followed, so that the set is of loop variables rather than of whatever
        names happened to appear.
        """

        for s in index.free_symbols:
            if s in V.kernel.itervars:
                self.dependent_itervars.add(s)
            elif s.name in V.kernel.cse.varname_map:
                self.dependent_itervars.update(
                    V.kernel.cse.varname_map[s.name].dependent_itervars
                )

    def depends_on(self, itervar) -> bool:
        """Whether this expression was built from the given loop variable."""

        return itervar in self.dependent_itervars


class CppPrinter(CXX11CodePrinter):
    """The printer for a host expression, with the region's facts applied.

    An index expression arrives carrying whatever the region knew about its
    extents, some of which may since have been decided; the printer asks the
    region's size variables to settle what can be settled before printing, so
    that a decided fact reaches the generated text.
    """

    def doprint(self, expr, *, simplify: bool = True, p=True):
        if simplify and isinstance(expr, sympy.Expr) and hasattr(V.graph, "sizevars"):
            expr = V.graph.sizevars.simplify(expr)
        return super().doprint(expr)

    def parenthesize(self, item, level, strict: bool = False) -> str:
        # A modulus binds looser than a product in the printed form, so a
        # negated modulus prints as a negation of the product rather than of the
        # modulus; the brackets are what make the difference visible.
        if isinstance(item, sympy.Mod):
            return f"({self._print(item)})"
        return super().parenthesize(item, level, strict)

    def _print_ModularIndexing(self, expr: sympy.Expr) -> str:
        x, div, mod = expr.args
        x = self.doprint(x)
        if div != 1:
            div = self.doprint(div)
            x = (
                "tensorplay::generated::floor_divide_integral("
                f"static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
            )
        mod = self.doprint(mod)
        return (
            f"(static_cast<{INDEX_TYPE}>({x}) % "
            f"static_cast<{INDEX_TYPE}>({mod}))"
        )

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        x, div = expr.args
        x = self.doprint(x)
        div = self.doprint(div)
        if expr.is_integer:
            return (
                "tensorplay::generated::floor_divide_integral("
                f"static_cast<int64_t>({x}), static_cast<int64_t>({div}))"
            )
        return (
            "tensorplay::generated::div_floor_floating("
            f"static_cast<double>({x}), static_cast<double>({div}))"
        )

    def _print_min_max(self, expr: sympy.Expr, name: str) -> str:
        # The host's min and max take two operands of one type and deduce it
        # from both, so a literal beside an index would not compile: every
        # operand is written in the one type the answer has.
        cpp_type = INDEX_TYPE if expr.is_integer else "double"
        args = [f"static_cast<{cpp_type}>({self._print(a)})" for a in expr.args]
        if len(args) == 2:
            return f"std::{name}({args[0]}, {args[1]})"
        return f"std::{name}<{cpp_type}>({{{', '.join(args)}}})"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "min")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max(expr, "max")


#: Print an expression as the host language writes it.
cexpr = CppPrinter().doprint


def get_promote_dtype(args):
    """The dtype every operand would be promoted to, or None if unknowable.

    A single operand whose dtype is not yet settled leaves the answer open;
    reporting a dtype anyway would silently narrow one of the operands.
    """

    typed = [n for n in args if isinstance(n, CppCSEVariable)]
    if not all(n.dtype is not None for n in typed):
        return None
    return functools.reduce(tp.promote_types, [n.dtype for n in typed])


def promote_args(new_args):
    """Bring every typed operand to the dtype they promote to among themselves.

    An operand already at that dtype is left alone, so a body that was written
    in one width does not gain conversions it does not need.
    """

    def promote_arg(arg, promote_type):
        if (
            isinstance(arg, CppCSEVariable)
            and arg.dtype
            and promote_type
            and arg.dtype != promote_type
        ):
            arg = ops.to_dtype(arg, promote_type)
            arg = arg.value if isinstance(arg, OpsValue) else arg
            arg.dtype = promote_type
        return arg

    promote_type = get_promote_dtype(new_args)
    if (
        all(
            new_arg.dtype is not None
            for new_arg in new_args
            if isinstance(new_arg, CppCSEVariable)
        )
        and promote_type
    ):
        new_args = [promote_arg(a, promote_type) for a in new_args]
    return new_args


def value_to_cpp(value, cpp_type):
    """A Python value written as a C++ expression of the given type.

    The infinities and the not-a-number have no literal spelling in C++, so
    each is written as the call that produces it instead of as a number.
    """

    if value == float("-inf"):
        return f"-std::numeric_limits<{cpp_type}>::infinity()"
    elif value == float("inf"):
        return f"std::numeric_limits<{cpp_type}>::infinity()"
    elif isinstance(value, bool):
        return f"static_cast<{cpp_type}>({str(value).lower()})"
    elif math.isnan(value):
        return f"std::numeric_limits<{cpp_type}>::quiet_NaN()"
    else:
        return f"static_cast<{cpp_type}>({repr(value)})"


def codegen_rand(offset, code, rand_function, dst_dtype=tp.float32):
    """Emit a random draw whose per-lane offsets are computed on the host.

    The offsets are staged through a buffer and the draw is loaded back as a
    vector, so the number of lanes the draw covers and the tiling factor agree.
    """

    if not is_integer_dtype(offset.dtype):
        raise AssertionError(f"expected integer dtype, got {offset.dtype}")
    code.writeline("[&]()")
    with code.indent():
        code.writeline(
            f"{DTYPE_TO_CPP[offset.dtype]} offset[{V.kernel.tiling_factor}];"
        )
        code.writeline(f"{DTYPE_TO_CPP[dst_dtype]} result[{V.kernel.tiling_factor}];")
        code.writeline(f"{offset}.store(offset);")
        code.writeline(
            f"for( {DTYPE_TO_CPP[offset.dtype]} offset_idx = 0; offset_idx < {V.kernel.tiling_factor}; offset_idx++ )"
        )
        with code.indent():
            code.writeline(rand_function)
        num_vectors = V.kernel._get_num_vectors(dtype=dst_dtype)
        if num_vectors == 1:
            code.writeline(
                f"return tensorplay::vec::Vectorized<{DTYPE_TO_CPP[dst_dtype]}>::loadu(result);"
            )
        else:
            code.writeline(
                f"return tensorplay::vec::VectorizedN<{DTYPE_TO_CPP[dst_dtype]}, {num_vectors}>::loadu(result);"
            )
    code.writeline("()")
    return code


def cexpr_index(expr) -> str:
    """Print an expression that is an index into a buffer."""

    if isinstance(expr, sympy.Expr):
        return cexpr(sympy.sympify(expr))
    return str(expr)


def unify_mask_base_type(buffer, vars, dtype=tp.float32):
    """Read a set of truth values as masks of one base type.

    Two masks over the same lanes can be combined only if they are masks of the
    same type, and a mask's base type is a choice rather than something the
    value itself fixes.  This re-reads each as the chosen type.
    """

    new_vars = (
        V.kernel.cse.generate(
            buffer,
            f"{V.kernel._get_mask_cast(var, dtype)}",
        )
        for var in vars
    )
    return new_vars


def may_unify_binary_op_mask_type(a, b):
    """Two operands of a bitwise combination, put in mask types that combine.

    A truth value combines with another truth value, and the result is a mask
    of the type both are read as, so both are re-read as that type first.
    """

    if a.dtype == tp.bool:
        if b.dtype != tp.bool:
            raise AssertionError(f"expected b.dtype == tp.bool, got {b.dtype}")
        mask_dtype = tp.int32
        return unify_mask_base_type(V.kernel.compute, (a, b), mask_dtype)
    return a, b


def rewrite_index_for_function(
    localize_buffer_handler: "LocalizeBufferHandler",
    index: sympy.Expr,
    global_buf_name: str,
):
    """Rewrite an index for a local buffer that covers only the inner dimensions.

    The global value was addressed by an index running over every dimension,
    while the local one is addressed by one running over the dimensions the
    local buffer kept.  The dimensions that were dropped are the outer ones,
    which do not appear in a local address at all, so every index symbol
    belonging to a dropped dimension is replaced by a constant.  Which symbols
    to drop is read off the range the producing piece was given, since the
    inner dimensions are the trailing ones.
    """

    snode = V.graph.scheduler.name_to_buf[global_buf_name].defining_op
    if snode is None:
        raise AssertionError(f"expected defining_op for {global_buf_name}, got None")
    local_buf = localize_buffer_handler.global_to_local[global_buf_name]
    scheduler_nodes = snode.get_nodes()
    _, (group, reduction_group) = max(
        scheduler_nodes, key=lambda x: int(x.is_reduction())
    ).group
    call_ranges = tuple(group) + tuple(reduction_group)
    indices_to_keep = [
        f"x{len(call_ranges) - (idx + 1)}"
        for idx in range(len(local_buf.get_layout().size))
    ]
    sorted_symbols = sorted(index.free_symbols, key=lambda s: s.name)  # type: ignore[attr-defined]
    replacements = {}
    for x in sorted_symbols:
        if x.name.startswith("x") and x.name not in indices_to_keep:  # type: ignore[attr-defined]
            # Only keep index used by local buffer
            replacements[x] = sympy.core.numbers.Zero()
    index = sympy_subs(index, replacements)  # type: ignore[arg-type]
    return index


def rewrite_index_for_nodes(
    localize_buffer_handler: "LocalizeBufferHandler",
    index: sympy.Expr,
    global_buf_name: str,
):
    """Rewrite an index for a local buffer, by rebuilding it from its own extents.

    Rather than dropping symbols out of the global index, the local index is
    built fresh: each of the local buffer's dimensions is given the symbol the
    global index used for that position if the index actually used it, and a
    constant otherwise.  A dimension the index did not vary over does not need
    varying over locally either.
    """

    used_vars = OrderedSet(
        s for s in index.free_symbols if symbol_is_type(s, SymT.INDEX)
    )
    index_vars = []
    local_buf = localize_buffer_handler.global_to_local[global_buf_name]
    for i in range(len(local_buf.get_size())):
        var = sympy_index_symbol_with_prefix(SymT.INDEX, i)
        index_vars.append(var if var in used_vars else 0)
    index = local_buf.get_layout().make_indexer()(index_vars)
    return index


class LocalizeBufferHandler(WrapperHandler):
    """Redirect the global buffers named in a scope to their local counterparts.

    While a function's inner loops are being written, a value that lives in a
    buffer belonging to the outer program should be read from the local buffer
    standing in for it, so that the traffic stays in the innermost memory.  The
    name is therefore rewritten on the way through, and the index with it
    where the local buffer is smaller than the global one.
    """

    def __init__(
        self,
        inner,
        global_to_local: dict[str, ir.Buffer],
        rewrite_index: Callable[["LocalizeBufferHandler", sympy.Expr, str], sympy.Expr],
    ) -> None:
        super().__init__(inner)
        self.global_to_local = global_to_local
        self.rewrite_index = rewrite_index

    def localize(self, name: str, index: sympy.Expr):
        if self.global_to_local and name in self.global_to_local:
            if self.rewrite_index is None:
                raise AssertionError("expected rewrite_index to be set, got None")
            index = self.rewrite_index(self, index, name)
            name = self.global_to_local[name].get_name()
        return name, index

    def load(self, name: str, index: sympy.Expr):
        return self._inner.load(*self.localize(name, index))

    def store(self, name, index, value, mode=None):
        local_buffer_name, local_buffer_index = self.localize(name, index)
        res = self._inner.store(local_buffer_name, local_buffer_index, value, mode)
        if (
            self.global_to_local
            and name in self.global_to_local
            and isinstance(V.kernel, Kernel)
        ):
            # The local name was recorded as one this launch writes to, so that
            # the launch keeps it alive.  It does not: it lives and dies inside.
            V.kernel.store_buffer_names.discard(local_buffer_name)
        return res

    def store_reduction(self, name, index, value):
        # pyrefly: ignore [bad-argument-count]
        return self._inner.store_reduction(*self.localize(name, index), value)


class LocalBufferContext:
    """Buffers that belong to one compiled function and are invisible outside it.

    A fused set of loops often needs somewhere to keep a value that only the
    inner loops read -- an accumulator, or a row of a value that the outer loop
    walks one row at a time.  Such a buffer is neither part of the region nor an
    argument of the compiled function: it is created while the function is being
    written and disappears with it.  So the three things that would otherwise
    have to know about it -- asking a name for its element type, asking whether
    a name is an input, asking whether a name is an output -- are answered
    differently for the duration, and put back afterwards.

    Registering a local buffer alongside the global ones it stands for also
    marks the global ones as removed, since nothing outside the function reads
    them any more.  Which ones were removed is recorded, because a decision
    taken later may turn out to need them back.
    """

    def __init__(self, kernel_args: KernelArgs) -> None:
        self.kernel_args = kernel_args
        self.exit_stack = contextlib.ExitStack()
        # map local buffer name to local buffer
        self.local_buffers: dict[str, ir.Buffer] = {}
        # map global buffer name to global buffer
        self.global_buffers: dict[str, ir.Buffer] = {}
        # map global buffer name to local buffer
        self.global_to_local: dict[str, ir.Buffer] = {}
        # record the global buffers that are removed by this LocalBufferContext
        self.removed_buffers: OrderedSet[str] = OrderedSet()

    def __enter__(self):
        self.exit_stack.__enter__()
        original_get_dtype = V.graph.get_dtype

        def get_dtype(name):
            if name in self.local_buffers:
                return self.local_buffers[name].get_dtype()
            return original_get_dtype(name)

        self.exit_stack.enter_context(patch.object(V.graph, "get_dtype", get_dtype))

        original_input = self.kernel_args.input

        def input(name):
            if name in self.local_buffers:
                return name
            return original_input(name)

        self.exit_stack.enter_context(patch.object(self.kernel_args, "input", input))

        original_output = self.kernel_args.output

        def output(name):
            if name in self.local_buffers:
                return name
            return original_output(name)

        self.exit_stack.enter_context(patch.object(self.kernel_args, "output", output))

        # Set current LocalBufferContext into V
        self.exit_stack.enter_context(V.set_local_buffer_context(self))

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.local_buffers.clear()
        self.exit_stack.__exit__(exc_type, exc_val, exc_tb)

    def add_local_buffer(
        self, local_buffer: ir.Buffer, global_buffers: list[ir.Buffer] | None = None
    ):
        if local_buffer.get_name() in self.local_buffers:
            raise AssertionError(
                f"local buffer {local_buffer.get_name()} already registered"
            )
        self.local_buffers[local_buffer.get_name()] = local_buffer
        if global_buffers:
            for global_buffer in global_buffers:
                global_buffer_name = global_buffer.get_name()
                if not (
                    global_buffer_name not in self.global_buffers
                    and global_buffer_name not in self.global_to_local
                ):
                    raise AssertionError(
                        f"global buffer {global_buffer_name} already registered"
                    )
                self.global_buffers[global_buffer_name] = global_buffer
                self.global_to_local[global_buffer_name] = local_buffer
                if global_buffer_name not in V.graph.removed_buffers:
                    # Recorded as well as marked, so that a decision taken later
                    # can hand the buffer back rather than find it gone.
                    self.removed_buffers.add(global_buffer_name)
                    V.graph.removed_buffers.add(global_buffer_name)

    def localize_function(
        self,
        fn: Callable[..., Any],
        rewrite_index: Callable[
            ["LocalizeBufferHandler", sympy.Expr, str], sympy.Expr
        ] = rewrite_index_for_function,
    ):
        """Run a function with the global buffers redirected to their local ones."""

        def inner(*args, **kwargs):
            with V.set_ops_handler(
                LocalizeBufferHandler(
                    V.get_ops_handler(),
                    global_to_local=self.global_to_local,
                    rewrite_index=rewrite_index,
                )
            ):
                return fn(*args, **kwargs)

        return inner

    def localize_nodes(
        self,
        nodes: list[ir.IRNode],
        rewrite_index: Callable[
            ["LocalizeBufferHandler", sympy.Expr, str], sympy.Expr
        ] = rewrite_index_for_nodes,
    ) -> list[ir.IRNode]:
        """Return copies of these pieces that read and write the local buffers.

        A copy is made rather than the original changed, because the original
        is still the description of the global computation and something else
        may need it in that form.  Every load and store in the copy is
        redirected, which is what makes the fused loops work on the smaller
        local buffers and so keep their traffic in the innermost memory.

        The local buffer is taken to be laid out contiguously in the same order
        as the global one, which is what lets an index be carried across
        unchanged apart from the dimensions that were dropped.
        """

        if len(nodes) <= 0:
            raise AssertionError(f"expected non-empty nodes, got {len(nodes)}")

        def wrap_inner_fn_for_node(node: ir.IRNode):
            loops = node.data if isinstance(node, ir.ComputedBuffer) else node
            if not isinstance(loops, ir.Loops):
                raise AssertionError(f"expected ir.Loops, got {type(loops)}")
            new_inner_fn = self.localize_function(
                loops.inner_fn,
                rewrite_index,
            )

            new_loops = dataclasses.replace(loops, inner_fn=new_inner_fn)
            if isinstance(node, ir.ComputedBuffer):
                new_node = ir.ComputedBuffer(
                    name=node.get_name(), layout=node.get_layout(), data=new_loops
                )
            else:
                new_node = new_loops  # type: ignore[assignment]

            return new_node

        return [wrap_inner_fn_for_node(node) for node in nodes]


def _get_loop_body(fn_list):
    """The loop body behind each of these callables, however it was wrapped.

    Three shapes turn up: the body itself, a wrapper carrying the original
    beneath it (which is what a local buffer produces), and a partially applied
    call whose first argument is the body.  All three name the same body, and
    which shape appeared says where to look.
    """

    if all(isinstance(fn, LoopBody) for fn in fn_list):
        loop_bodies = fn_list
    else:
        if hasattr(fn_list[0], "original_fn"):
            # For the case of local buffer, we wrap the fn with localize_function
            if not all(hasattr(fn, "original_fn") for fn in fn_list):
                raise AssertionError("expected all fns to have 'original_fn'")
            if not all(
                isinstance(fn.original_fn.args[0]._body, LoopBody) for fn in fn_list
            ):
                raise AssertionError("expected all original_fn bodies to be LoopBody")
            loop_bodies = [fn.original_fn.args[0]._body for fn in fn_list]
        else:
            if not all(isinstance(fn, functools.partial) for fn in fn_list):
                raise AssertionError("expected all fns to be functools.partial")
            if not all(isinstance(fn.args[0]._body, LoopBody) for fn in fn_list):
                raise AssertionError("expected all fn bodies to be LoopBody")
            loop_bodies = [fn.args[0]._body for fn in fn_list]
    if loop_bodies is None:
        raise AssertionError("expected loop_bodies to be set, got None")
    return loop_bodies


def _get_dtype_from_loopbodies(loop_bodies):
    """Every element type the body actually computes in.

    Read off the operations rather than off the buffers, so a body that loads
    one type and stores another reports the type it computed in -- which is the
    one that has to be held in a register.
    """

    dtypes = OrderedSet[tp.dtype]()
    for loop_body in loop_bodies:
        graphs = [loop_body.root_block.graph] + [
            body.graph for body in list(loop_body.subblocks.values())
        ]
        for graph in graphs:
            for node in graph.nodes:
                if node.op != "call_method":
                    continue
                dtypes.add(node.meta[OptimizationContext.key].dtype)
    return dtypes


def template_fusion_with_epilogues_supported(
    template: BaseSchedulerNode, epilogues: list[BaseSchedulerNode]
) -> tuple[bool, bool]:
    """Whether work may be folded into the end of a prepared kernel, and whether
    the fold reads the prepared kernel's result at the position it writes its own.

    Two questions, and they are not the same.  Whether it is supported at all
    depends on the fold reading that result in one place only: read in two
    places and there is no single element to compute the rest of the work
    around.  Whether the read and the write agree on which element they mean is
    a separate question, and is reported separately because a caller that can
    use the weaker guarantee may still take the fold.
    """

    def _get_indexes_of_template_buf_read(
        epilogue_node: ir.Operation, template_buf_names: list[str]
    ) -> list[sympy.Expr]:
        return [
            read.index
            for read in epilogue_node.get_reads()
            if read.name in template_buf_names
        ]

    def _check_supported_and_same_indexes(
        index_of_template_buf_read: Sequence[sympy.Expr],
        epilogue_writes: OrderedSet[Dep],
    ) -> tuple[bool, bool]:
        num_indexes = len(OrderedSet(index_of_template_buf_read))

        if num_indexes > 1:
            same_index = False
            supported = False  # Different read indexes not supported
        elif num_indexes == 0:
            same_index = True
            supported = True  # No reads, automatically supported
        elif num_indexes == 1:
            iotbr = index_of_template_buf_read[0]
            same_index = all(write.index == iotbr for write in epilogue_writes)
            # Where the read and the write disagree, the fold is not supported:
            # there is no arrangement of the prepared kernel that puts the
            # written element where the read expects it.
            supported = same_index
        else:
            raise AssertionError("Should not reach here")

        return supported, same_index

    def _template_fusion_supported(
        template_outputs: Sequence[SchedulerBuffer], epilogue_nodes: list[ir.Operation]
    ) -> tuple[bool, bool]:
        template_buf_names = [x.get_name() for x in template_outputs]
        indexes_of_template_buf_reads = [
            _get_indexes_of_template_buf_read(epilogue_node, template_buf_names)
            for epilogue_node in epilogue_nodes
        ]
        epilogue_nodes_writes = [
            epilogue_node.get_read_writes().writes for epilogue_node in epilogue_nodes
        ]

        results = [
            _check_supported_and_same_indexes(reads, writes)
            for reads, writes in zip(
                indexes_of_template_buf_reads, epilogue_nodes_writes
            )
        ]
        supported, same_indexes = zip(*results)
        return all(supported), all(same_indexes)

    if not template.is_template():
        raise AssertionError("expected template.is_template() to be True")
    template_outputs = template.get_outputs()

    epilogue_nodes = [
        n.node
        for epilogue in epilogues
        for n in epilogue.get_nodes()
        if n.node is not None
    ]
    return _template_fusion_supported(template_outputs, epilogue_nodes)
