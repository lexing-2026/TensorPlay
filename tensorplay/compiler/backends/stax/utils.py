"""The small computations the generators and layouts share.

A stride is sorted, an extent is divided, a storage is measured, an access is
described.  Each of those is a few lines whose exact form matters -- a sort
that is not stable changes which layout a code generator picks, a division
that rounds the other way changes which elements a kernel touches -- so they
are written once here and called from where they are needed rather than
rewritten at each call site.
"""

from __future__ import annotations

import contextlib
import enum
import functools
import hashlib
import math
import operator
import sys
import textwrap
import logging

import tensorplay as tp


log = logging.getLogger(__name__)
from dataclasses import dataclass
from io import StringIO
from typing import NamedTuple
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from typing import (
    Any,
    Callable,
    Concatenate,
    Generic,
    ParamSpec,
    Protocol,
    TypeVar,
)

import sympy
from sympy import Expr

from tensorplay.graph.experimental.symbolic_shapes import GuardOnDataDependentSymNode
from tensorplay.graph.experimental.sympy_functions import (
    _PREFIX_STR,
    bound_sympy,
    CeilDiv,
    ModularIndexing,
    OrderedSet,
    SymT,
    ValueRanges,
)

from . import config

_T = TypeVar("_T")
_FN_TYPE = TypeVar("_FN_TYPE", bound=Callable[..., Any])

__all__ = [
    "DeferredLineBase",
    "IndentedBuffer",
    "LineContext",
    "argsort",
    "argsort_sym",
    "cache_on_self_and_args",
    "ceildiv",
    "compute_required_storage_length",
    "OpDtypeRule",
    "ScopedDict",
    "cache_on_self",
    "parallel_num_threads",
    "boolean_ops",
    "op_dtype_propagation_rules",
    "register_op_dtype_propagation_rules",
    "upcast_compute_type",
    "free_symbol_is_type",
    "get_current_backend",
    "get_free_symbols",
    "has_free_symbols",
    "make_channels_last_1d_strides_for",
    "make_channels_last_2d_strides_for",
    "make_channels_last_3d_strides_for",
    "make_channels_last_strides_for",
    "unique",
]


def unique(it: Iterable[_T]):
    """The values of an iterable, with the repeats removed, in order.

    Removal is by identity rather than by equality, because two distinct
    values that compare equal are still two values a caller may have wanted to
    tell apart -- a rewritten buffer and the buffer it replaced, for instance.
    """

    return {id(x): x for x in it}.values()


def ceildiv(number: int | Expr, denom: int | Expr) -> int | Expr:
    """Division rounded upwards, as an integer when both sides are integers."""

    if isinstance(number, Expr) or isinstance(denom, Expr):
        return CeilDiv(sympy.sympify(number), sympy.sympify(denom))
    if not (isinstance(number, int) and isinstance(denom, int)):
        raise AssertionError(f"{number}: {type(number)}, {denom}: {type(denom)}")
    return -(-number // denom)


def has_free_symbols(itr: Iterable[object]) -> bool:
    """Whether any of the values is a symbolic expression that is not a number."""

    return any(isinstance(x, Expr) and not x.is_number for x in itr)


def get_free_symbols(x, unbacked_only: bool) -> "OrderedSet":
    """Every free symbol of a value, or only the ones no input can pin down."""

    from tensorplay.graph.experimental.symbolic_shapes import free_unbacked_symbols

    if unbacked_only:
        return OrderedSet(free_unbacked_symbols(x))
    return OrderedSet(_free_symbols_of(x))


def _free_symbols_of(value) -> set:
    if isinstance(value, sympy.Basic):
        return set(value.free_symbols)
    if isinstance(value, (list, tuple, set, frozenset)):
        out: set = set()
        for item in value:
            out |= _free_symbols_of(item)
        return out
    if isinstance(value, dict):
        out = set()
        for key, item in value.items():
            out |= _free_symbols_of(key)
            out |= _free_symbols_of(item)
        return out
    return set()


def cache_on_self_and_args(class_name: str):
    """Remember what a method returned for the arguments it was given.

    The methods this decorates are pure functions of the object and its
    arguments, are asked the same question once per dimension per buffer, and
    are expensive enough that recomputing them is visible.  The cache lives on
    the object, and an object that is hashed cannot carry one, so the store is
    attached past the attribute lookup that would otherwise refuse it.
    """

    def wrapper(fn):
        key = f"__{class_name}_{fn.__name__}_cache"

        def inner(self, *args, **kwargs):
            args_kwargs = (args, tuple(sorted(kwargs.items())))

            if not hasattr(self, key):
                object.__setattr__(self, key, {})

            cache = getattr(self, key)

            try:
                return cache[args_kwargs]
            except KeyError:
                pass
            result = fn(self, *args, **kwargs)
            cache[args_kwargs] = result
            return result

        inner.__name__ = fn.__name__
        inner.__qualname__ = fn.__qualname__
        inner.__doc__ = fn.__doc__
        return inner

    return wrapper


def argsort(seq: Sequence[Any], *, reverse: bool = False) -> list[int]:
    """The positions of the values in ascending order, ties keeping their order.

    Two dimensions with the same stride are told apart by which of them is
    inner, so the sort has to be stable and it has to be reversed before the
    reversal that produces ascending order -- otherwise the inner dimension of
    a pair of equal strides ends up outer, and the two are not interchangeable.
    """

    getter = seq.__getitem__
    a_r = range(len(seq))
    sort_idx = list(sorted(a_r, key=getter, reverse=True))
    if not reverse:
        return list(reversed(sort_idx))
    return sort_idx


def argsort_sym(
    shape_env,
    seq: Sequence[int | Expr],
    *,
    reverse: bool = False,
) -> list[int]:
    """The same order, for strides whose comparison the environment may not settle.

    A comparison that cannot be settled falls back to a hint rather than to a
    guard, so the order this returns is one that is convenient to optimize
    with and must not be used to decide anything about correctness.
    """

    def cmp(a: tuple[int, Expr], b: tuple[int, Expr]) -> int:
        a_idx, a_val = a
        b_idx, b_val = b

        def evaluate(expr, fallback: Callable[[], bool]) -> bool:
            if isinstance(expr, bool):
                return expr
            try:
                return shape_env.evaluate_expr(expr)
            except GuardOnDataDependentSymNode:
                return fallback()

        def hint_lt(lhs: Expr, rhs: Expr) -> bool:
            return shape_env.optimization_hint(lhs) < shape_env.optimization_hint(
                rhs
            )

        if evaluate(a_val < b_val, lambda: hint_lt(a_val, b_val)):
            return -1
        if evaluate(a_val > b_val, lambda: hint_lt(b_val, a_val)):
            return 1
        # Two strides that compare equal keep the order they were given in, so
        # that the result does not depend on how the comparison was settled.
        return b_idx - a_idx if reverse else a_idx - b_idx

    return [idx for idx, _ in sorted(enumerate(seq), key=functools.cmp_to_key(cmp))]


def compute_required_storage_length(
    shape: Sequence[int | Expr],
    strides: Sequence[int | Expr],
    storage_offset: int | Expr,
) -> int | Expr:
    """How many elements of storage a tensor of this geometry occupies.

    The answer is the distance from the first element to the last, plus the
    offset the first one starts at.  A geometry with no elements occupies
    nothing, which is decided before anything is measured, since a product of
    extents that is zero makes every term below meaningless.
    """

    if shape_env_guard_or_false(functools.reduce(operator.mul, shape, 1) == 0):
        return 0

    max_offset = sum((x - 1) * y for x, y in zip(shape, strides))
    return 1 + storage_offset + max_offset


def shape_env_guard_or_false(expr) -> bool:
    """The claim, or false when the current graph's environment cannot settle it.

    A claim that is already a plain truth value is answered as it is, without
    consulting anything: there is nothing to settle about it.
    """

    if isinstance(expr, bool):
        return expr

    from .loops import V

    sizevars = getattr(V, "sizevars", None)
    if sizevars is None:
        return False
    return sizevars.guard_or_false(expr)


def make_channels_last_1d_strides_for(shape: Sequence[Any]) -> tuple:
    """The strides of a rank-three tensor whose channels are the outermost run.

    The channel dimension is filled last, which is the whole point of the
    format: consecutive elements are consecutive channels, so a kernel that
    walks channels walks memory.
    """

    if len(shape) != 3:
        raise AssertionError(
            "Only tensors of rank 3 can use the channels_last_1d memory format"
        )

    multiplier = 1
    strides: list = [0] * 3
    for idx in (1, -1, 0):
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_2d_strides_for(shape: Sequence[Any]) -> tuple:
    """The strides of a rank-four tensor with its channels outermost."""

    if len(shape) != 4:
        raise AssertionError(
            "Only tensors of rank 4 can use the channels_last memory format"
        )

    multiplier = 1
    strides: list = [0] * 4
    for idx in (1, -1, -2, 0):
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_3d_strides_for(shape: Sequence[Any]) -> tuple:
    """The strides of a rank-five tensor with its channels outermost."""

    if len(shape) != 5:
        raise AssertionError(
            "Only tensors of rank 5 can use the channels_last_3d memory format"
        )

    multiplier = 1
    strides: list = [0] * 5
    for idx in (1, -1, -2, -3, 0):
        strides[idx] = multiplier
        multiplier *= shape[idx]

    return tuple(strides)


def make_channels_last_strides_for(shape: Sequence[Any]) -> tuple:
    """The strides of a tensor whose channels are outermost, whatever its rank."""

    ndim = len(shape) if isinstance(shape, Sequence) else 1
    if ndim == 3:
        return make_channels_last_1d_strides_for(shape)
    if ndim == 4:
        return make_channels_last_2d_strides_for(shape)
    if ndim == 5:
        return make_channels_last_3d_strides_for(shape)
    raise RuntimeError(f"no channels last format strides exist in {ndim} dimensions")


class LineContext(NamedTuple):
    context: Any


@dataclass
class ValueWithLineMap:
    value: str
    line_map: list[tuple[int, LineContext]]


class IndentedBuffer:
    tabwidth = 4

    def __init__(self, initial_indent: int = 0) -> None:
        self._lines: list[DeferredLineBase | LineContext | str] = []
        self._indent = initial_indent

    @contextlib.contextmanager
    def set_tabwidth(self, tabwidth: int) -> Iterator[None]:
        prev = self.tabwidth
        try:
            self.tabwidth = tabwidth
            yield
        finally:
            self.tabwidth = prev

    def getvaluewithlinemap(self) -> "ValueWithLineMap":
        """The rendered text, and a map from its lines to what produced them.

        The map is what lets a report about the generated code point at the
        line of the program that asked for it.
        """
        buf = StringIO()
        p = 1
        linemap: list[tuple[int, LineContext]] = []
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                linemap.append((p, li.context))
                continue
            else:
                line = li
            if not isinstance(line, str):
                raise AssertionError(f"Expected str, got {type(line)}")
            buf.write(line)
            buf.write("\n")
            p += 1 + line.count("\n")
        return ValueWithLineMap(buf.getvalue(), linemap)

    def getvalue(self) -> str:
        return self.getvaluewithlinemap().value

    def getrawvalue(self) -> str:
        """The rendered text, with the backslash line continuations joined up.

        A backslash at the end of a line is how a rendered expression is
        wrapped; joining the continuations gives the text as it will be read
        rather than as it was laid out.
        """
        buf = StringIO()
        for li in self._lines:
            if isinstance(li, DeferredLineBase):
                line = li()
                if line is None:
                    continue
            elif isinstance(li, LineContext):
                continue
            else:
                line = li
            if not isinstance(line, str):
                raise AssertionError(f"Expected str, got {type(line)}")
            # backslash implies line continuation
            if line.endswith("\\"):
                buf.write(line[:-1])
            else:
                buf.write(line)
                buf.write("\n")
        return buf.getvalue()

    def get_lines_ref(self):
        return self._lines

    def clear(self) -> None:
        self._lines.clear()

    def __bool__(self) -> bool:
        return bool(self._lines)

    def prefix(self) -> str:
        return " " * (self._indent * self.tabwidth)

    def newline(self) -> None:
        self.writeline("\n")

    def writeline(self, line: LineContext | DeferredLineBase | str) -> None:
        if isinstance(line, LineContext):
            self._lines.append(line)
        elif isinstance(line, DeferredLineBase):
            self._lines.append(line.with_prefix(self.prefix()))
        elif line.strip():
            self._lines.append(f"{self.prefix()}{line}")
        else:
            self._lines.append("")

    def writeline_jit(self, line: LineContext | DeferredLineBase | str) -> None:
        """Write to JIT buffer only. On a plain IndentedBuffer, same as writeline."""
        self.writeline(line)

    def writeline_aot(self, line: LineContext | DeferredLineBase | str) -> None:
        """Write to AOTI buffer only. No-op on a plain IndentedBuffer."""

    def splice_jit(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        """Splice to JIT buffer only. On a plain IndentedBuffer, same as splice."""
        self.splice(other_code, strip=strip)

    def splice_aot(self, other_code: IndentedBuffer | str, strip: bool = False) -> None:
        """Splice to AOTI buffer only. No-op on a plain IndentedBuffer."""

    def writelines(self, lines: Sequence[LineContext | DeferredLineBase | str]) -> None:
        for line in lines:
            self.writeline(line)

    def indent(self, offset: int = 1) -> contextlib.AbstractContextManager[None]:
        @contextlib.contextmanager
        def ctx() -> Iterator[None]:
            self._indent += offset
            try:
                yield
            finally:
                self._indent -= offset

        return ctx()

    def do_indent(self, offset: int = 1) -> None:
        self._indent += offset

    def do_unindent(self, offset: int = 1) -> None:
        self._indent -= offset

    def splice(self, other_code, strip: bool = False) -> None:
        """Append another buffer's lines, or a block of text, to this one.

        A buffer is spliced at its own indentation, so the lines of the least
        indented one become the common prefix and are dropped; text is dedented
        first, since a text block carries its indentation with it as a literal.
        """
        if isinstance(other_code, IndentedBuffer):
            dedent = float("inf")

            for line in other_code._lines:
                if not isinstance(line, LineContext) and line:
                    dedent = min(dedent, len(line) - len(line.lstrip()))
            if math.isinf(dedent):
                dedent = 0
            for line in other_code._lines:
                if isinstance(line, LineContext):
                    self._lines.append(line)
                else:
                    IndentedBuffer.writeline(self, line[int(dedent) :])
        else:
            other_code = textwrap.dedent(other_code)
            if strip:
                other_code = other_code.lstrip()
            if not other_code:
                return
            other_code = other_code.rstrip()
            for s in other_code.split("\n"):
                IndentedBuffer.writeline(self, s)

    def map(self, func):
        """The same lines, each put through a function."""

        res = IndentedBuffer(initial_indent=self._indent)
        res._lines = [func(line) for line in self._lines]
        return res

    def __repr__(self) -> str:
        return f"{type(self)}({self.getvalue()})"

    def __add__(self, other) -> "IndentedBuffer":
        if self._indent != other._indent:
            raise AssertionError(f"Indent mismatch: {self._indent} != {other._indent}")
        res = IndentedBuffer(initial_indent=self._indent)
        # TODO(rec): or should this be self.__class__(initial_indent=self._indent)?
        res.writelines(self._lines)
        res.writelines(other._lines)
        return res

    def contains(self, new_line: DeferredLineBase | LineContext | str) -> bool:
        return new_line in self._lines


class DeferredLineBase:
    """A line that can be 'unwritten' at a later time"""

    def __init__(self, line: str):
        if not line.strip():
            line = ""
        self.line = line

    def __call__(self) -> str | None:
        """Returns either self.line or None to indicate the line has been 'unwritten'"""
        raise NotImplementedError

    def _new_line(self, line: str) -> "DeferredLineBase":
        """Returns a new deferred line with the same condition"""
        raise NotImplementedError

    def with_prefix(self, prefix: str) -> "DeferredLineBase":
        return self._new_line(f"{prefix}{self.line}")

    def lstrip(self) -> "DeferredLineBase":
        return self._new_line(self.line.lstrip())

    def __getitem__(self, index: int | slice) -> "DeferredLineBase":
        return self._new_line(self.line[index])

    def __bool__(self) -> bool:
        return bool(self.line)

    def __len__(self) -> int:
        return len(self.line)

class ScopedDict(MutableMapping):
    """A mapping whose changes are confined to a scope.

    An emitter that tries something and then takes it back -- a speculative
    rewrite, a loop nest that turned out not to apply -- must be able to write
    into a mapping and then have those writes disappear, while the writes that
    were already there stay.  So the changes go in a second mapping layered
    over the first, and dropping the scope drops the layer.
    """

    def __init__(self, original_dict):
        self.original_dict = original_dict
        self.new_items: dict = {}

    def __getitem__(self, key):
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict[key]

    def __setitem__(self, key, value) -> None:
        self.new_items[key] = value

    def __contains__(self, key: object) -> bool:
        return key in self.new_items or key in self.original_dict

    def get(self, key, default=None):
        if key in self.new_items:
            return self.new_items[key]
        return self.original_dict.get(key, default)

    def __len__(self) -> int:
        n = len(self.original_dict)
        for k in self.new_items:
            if k not in self.original_dict:
                n += 1
        return n

    def __iter__(self):
        yield from self.original_dict
        for k in self.new_items:
            if k not in self.original_dict:
                yield k

    def __bool__(self) -> bool:
        return bool(self.original_dict or self.new_items)

    def __delitem__(self, key) -> None:
        raise NotImplementedError


def get_current_backend(device_type: str | None = None) -> str:
    """Which emitter the code being written is for.

    A question that has a different answer per emitter is asked of this rather
    than of the device directly, because the emitter is what the answer names.
    """

    from .loops import V

    if not device_type:
        device_type = V.graph.get_current_device_or_throw().type
    if device_type == "cpu":
        return "cpp"
    return "triton"


def generate_assert(check: bool) -> bool:
    """Whether a bounds check is to be written into the generated code.

    A check the caller asked for is written; one nobody asked for is written
    only when the run is asking for them as a matter of policy, since a check
    that is always on costs every kernel something.
    """

    from . import config

    return (check or config.debug_index_asserts) and config.assert_indirect_indexing


#: The extent each loop variable of an access runs over.
#: The width a wide load has to be aligned to.  Sixteen bytes is what a memory
#: transaction moves, so a load that straddles one costs two of them.
GPU_ALIGN_BYTES = 16


VarRanges = dict[Expr, Expr]


def sympy_product(it: Iterable[Expr]) -> Expr:
    """The product of a sequence of extents, as one expression.

    An empty sequence is one, which is what makes a product over a shape of no
    dimensions the size of a single element rather than nothing.
    """

    return functools.reduce(operator.mul, it, sympy.S.One)


def sympy_dot(seq1: Sequence[Expr], seq2: Sequence[Expr]) -> Expr:
    """The inner product of two sequences of the same length."""

    if not len(seq1) == len(seq2):
        raise AssertionError(
            f"Length mismatch: len(seq1)={len(seq1)}, len(seq2)={len(seq2)}"
        )
    return sympy.expand(sum(a * b for a, b in zip(seq1, seq2)))


def _compute_stride(
    old_shape: Sequence,
    old_stride: Sequence,
    new_shape: Sequence,
    size_oblivious: bool = False,
):
    """The strides a new shape would need over an old one, or None if it cannot be done.

    A reshape is possible exactly when the new shape's dimensions can each be
    matched against a run of the old ones, and this walks both from the
    innermost outwards looking for that.  ``None`` means no such matching was
    found, which is not a failure but an answer: the elements are not in an
    order a view can describe, so a copy would be needed.

    ``size_oblivious`` decides whether a comparison between extents is guarded
    on or merely assumed.  Where an extent came out of the data there is nothing
    to guard on at code-writing time, so the comparison is skipped instead.
    """

    from tensorplay.graph.experimental.symbolic_shapes import (
        guard_or_false,
        guard_or_true,
    )

    def maybe_guard_or_false(x):
        if size_oblivious:
            return guard_or_false(x)
        return x

    def maybe_guard_or_true(x):
        if size_oblivious:
            return guard_or_true(x)
        return x

    if len(old_shape) == 0:
        return [1] * len(new_shape)

    numel = functools.reduce(operator.mul, old_shape, 1)
    zero_numel = maybe_guard_or_false(numel == 0)
    if zero_numel and maybe_guard_or_false(sympy.Eq(tuple(old_shape), tuple(new_shape))):
        return list(old_stride)

    new_stride: list = [0] * len(new_shape)

    # A tensor with no elements has no elements to be out of order, so any
    # strides will do.
    if zero_numel:
        for view_d in range(len(new_shape) - 1, -1, -1):
            if view_d == len(new_shape) - 1:
                new_stride[view_d] = 1
            else:
                new_stride[view_d] = (
                    max(new_shape[view_d + 1], 1) * new_stride[view_d + 1]
                )
        return new_stride

    view_d = len(new_shape) - 1
    # The stride of the run of the innermost old dimensions that is treated as
    # one contiguous block.
    chunk_base_stride = old_stride[-1]
    tensor_numel = 1
    view_numel = 1

    for tensor_d in range(len(old_shape) - 1, -1, -1):
        tensor_numel *= old_shape[tensor_d]

        # A dimension ends a block when it is the outermost, or when the one
        # inside it does not simply continue where this one left off.
        if tensor_d == 0 or (
            maybe_guard_or_true(old_shape[tensor_d - 1] != 1)
            and maybe_guard_or_true(
                old_stride[tensor_d - 1] != tensor_numel * chunk_base_stride
            )
        ):
            while view_d >= 0 and (
                maybe_guard_or_true(view_numel < tensor_numel)
                or maybe_guard_or_false(new_shape[view_d] == 1)
            ):
                new_stride[view_d] = view_numel * chunk_base_stride
                view_numel *= new_shape[view_d]
                view_d -= 1

            if maybe_guard_or_true(view_numel != tensor_numel):
                return None

            if tensor_d > 0:
                chunk_base_stride = old_stride[tensor_d - 1]
                tensor_numel = 1
                view_numel = 1
    if view_d != -1:
        return None
    return new_stride


def flatten_index(
    indices: Sequence[Expr],
    sizes: Sequence[Expr],
) -> Expr:
    """Per-dimension positions put together into one position.

    This is the row-major reading: the outermost dimension counts how many
    whole groups of the inner ones have gone by, and so on down to the
    innermost, which is what is left over.
    """

    if len(indices) != len(sizes):
        raise AssertionError(
            f"indices and sizes must have equal length, got "
            f"{len(indices)} and {len(sizes)}"
        )
    flat = sympy.S.Zero
    for index, size in zip(indices, sizes):
        flat = flat * size + index
    return flat


def decompose_index(
    index: Expr,
    sizes: Sequence[Expr],
) -> list[Expr]:
    """A flat index taken apart into one index per dimension, outermost first.

    This is the row-major reading of a flat position: the outermost dimension
    is how many whole groups of the inner ones have gone by, and so on down to
    the innermost, which is what is left over.
    """

    return [
        ModularIndexing(index, sympy_product(sizes[i + 1 :]), size)
        for i, size in enumerate(sizes)
    ]


def get_dtype_size(dtype) -> int:
    """How many bytes one value of this type takes.

    A 64-bit unsigned type is answered from the number rather than by making a
    value of it, because making one overflows where counting its bytes does not.
    """

    if dtype == tp.uint64:
        return 8
    return tp.empty((), dtype=dtype).element_size()


def is_welford_reduction(reduction_type: str) -> bool:
    """Whether a reduction is one of the two that keep a mean and a spread.

    They are not folds of one number: each carries three, and the combination
    is not commutative, which is why the two that share a name are written
    differently from the ones that do not.
    """

    return reduction_type.startswith("welford")


def is_windows() -> bool:
    """Whether this process is running on Windows.

    Path length limits and calling conventions both differ there, and the
    generated source has to answer for the platform it will be compiled on.
    """

    return sys.platform == "win32"


def aggregate_origins(node_schedule) -> "OrderedSet":
    """Every graph node that contributed to a scheduled group.

    A fused group is named after what went into it, so the origins of its
    members are what says which operation the kernel is for.  A member that
    has no node of its own -- a whole-program call -- contributes its own
    origins instead, and anything else contributes none.
    """

    from . import ir

    if isinstance(node_schedule, list):
        return functools.reduce(
            operator.or_,
            [
                node.node.origins
                for node in node_schedule
                if getattr(node, "node", None)
            ],
            OrderedSet(),
        )
    if isinstance(node_schedule, ir.ExternKernel):
        return node_schedule.origins
    return OrderedSet()


def get_fused_kernel_name(node_schedule, descriptive_names) -> str:
    """The name a fused kernel is cached and reported under.

    Which operation the name is built from is a choice the configuration makes:
    the pre-decomposition operation, the post-capture one, the graph node, or
    none at all.  Whichever is chosen, the name has to identify the kernel, so
    on a platform with a short path limit an over-long name is cut and
    disambiguated by a digest of what it was.
    """

    all_origins = aggregate_origins(node_schedule)
    if descriptive_names == "original_aten":

        def get_origin_meta_str(origin):
            original = origin.meta.get("original_aten")
            key = ""
            if isinstance(original, tp.ops.OpOverload):
                key = original._overloadpacket.__name__
            elif original is not None and hasattr(original, "name"):
                key = str(original.name())
            return key

        sources = [
            get_origin_meta_str(origin)
            for origin in all_origins
            if origin.op == "call_function"
            and origin.meta.get("original_aten") is not None
        ]
        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "torch":
        sources = []
        for origin in all_origins:
            if origin.op == "call_function":
                source_fn = None
                suffix = ""
                if "source_fn_stack" in origin.meta:
                    source_fn = origin.meta["source_fn_stack"][-1]
                elif "fwd_source_fn_stack" in origin.meta:
                    # a backward node carries the forward stack instead
                    source_fn = origin.meta["fwd_source_fn_stack"][-1]
                    suffix = "backward"
                if not source_fn:
                    continue
                if isinstance(source_fn[1], str):
                    sources.append(source_fn[1] + suffix)
                else:
                    sources.append(source_fn[1].__name__ + suffix)
        sources = sorted(OrderedSet(sources))
    elif descriptive_names == "inductor_node":
        sources = [
            origin.name for origin in all_origins if origin.op == "call_function"
        ]
    else:
        raise NotImplementedError(f"unknown descriptive_names={descriptive_names!r}")
    name = "_".join(["fused"] + sources)
    # A long name can push the cache path past the platform's limit, which
    # leaves the cached artifact unopenable; a digest keeps it unique.
    if is_windows() and len(name) > 50:
        h = hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
        name = f"{name[:41].rstrip('_')}_{h}"
    return name


def is_multi_outputs_template(input_buf) -> bool:
    """Whether this input is a template that produces several results.

    Such a call owns its results rather than writing into a buffer it was
    handed, which is what makes its operands and its outputs inseparable when
    a kernel is ordered.
    """

    from . import ir

    return (
        isinstance(input_buf, ir.TemplateBuffer)
        and input_buf.is_multi_outputs_template()
    )


def get_bounds_index_expr(index: "sympy.Expr"):
    """The value range an index expression can take.

    A bound that cannot be established is reported as unknown rather than
    guessed, since a wrong bound silently produces a wrong guard.
    """

    from .loops import V

    if (
        config.compute_all_bounds
        and (node := getattr(V.interpreter, "current_node", None))
        and getattr(node, "target", None) != "index_expr"
    ):
        return bound_sympy(index)
    return ValueRanges.unknown()


def reduction_num_outputs(reduction_type: str) -> int:
    """How many values a reduction of this kind produces.

    A reduction that keeps a running mean, a running spread and a weight
    produces three, one that keeps an index as well as a value produces two,
    and anything else produces one.
    """

    if is_welford_reduction(reduction_type):
        return 3
    elif reduction_type in (
        "argmax_with_value",
        "argmin_with_value",
        "online_softmax_reduce",
    ):
        return 2
    else:
        return 1


def sympy_index_symbol(name: str):
    """A symbol standing for an index, which is an integer and is not negative.

    A symbol whose name begins with the letter a shape's symbol begins with
    would be indistinguishable from a shape, and an index must not be usable
    where a shape is meant, so that spelling is refused here rather than
    producing a symbol that means two things.
    """

    import sympy

    if name[0] == "s":
        raise AssertionError(f"Symbol name must not start with 's', got {name!r}")
    return sympy.Symbol(name, integer=True, nonnegative=True)


def sympy_index_symbol_with_prefix(prefix, idx: int):
    """A symbol standing for an index, named from a kind and a number.

    The name is the kind's own prefix letter and the number, never the name of
    the kind: a name that spells the kind out would begin with whatever letter
    the kind's name happens to begin with, and a symbol that looks like a
    shape's is a shape for every purpose that reads the name.  A shape is also
    refused outright, because these are index variables and a shape is not one.
    """

    import sympy

    if prefix == SymT.SIZE:
        raise AssertionError(f"prefix must not be SymT.SIZE, got {prefix}")
    return sympy.Symbol(f"{_PREFIX_STR[prefix]}{idx}", integer=True, nonnegative=True)


def sympy_subs(expr, replacements: dict):
    """Substitute into an expression, where a replacement may be written as a name.

    A replacement written as a string names a symbol, and the symbol it names
    is given the properties of the expression it stands in for: a name that
    replaced a non-negative integer has to be a non-negative integer, or a
    later step would reason about it as something it is not.
    """

    import sympy

    def to_symbol(replaced, replacement):
        if not isinstance(replaced, sympy.Expr):
            raise AssertionError(
                f"Expected a symbolic key, got {type(replaced)}: {replaced}"
            )
        if isinstance(replacement, str):
            return sympy.Symbol(
                replacement,
                integer=replaced.is_integer,
                nonnegative=replaced.is_nonnegative,
            )
        return replacement

    return sympy.sympify(expr).xreplace(
        {k: to_symbol(k, v) for k, v in replacements.items()}
    )


def free_symbol_is_type(e, prefix) -> bool:
    """Whether any symbol appearing in an expression is of the kind named.

    An index may name a size, a loop variable, or a value that came out of the
    data, and which of those decides how it may be used; this is the question
    asked of an expression rather than of a single symbol.
    """

    from tensorplay.graph.experimental.sympy_functions import free_symbol_is_type as _impl

    return _impl(e, prefix)

@functools.cache
def boolean_ops() -> tuple:
    """The operations whose result is a truth value whatever their operands are.

    A comparison of two numbers is a truth value and not a number, and so is a
    combination of truth values; the rest of the operations promote their
    operands instead.
    """

    return (
        "isinf",
        "isnan",
        "logical_not",
        "logical_and",
        "signbit",
        "and_",
        "le",
        "lt",
        "ge",
        "gt",
        "eq",
        "ne",
        "or_",
        "xor",
    )


@dataclass
class OpDtypeRule:
    """What an operation's result type is, where promotion does not say it.

    Most operations promote their operands, and the promotion kind says how.
    An operation whose result is not the promotion of its operands -- a
    comparison, a cast, a constant -- says so here instead, by naming the type
    it returns or by naming the kind of promotion to apply.
    """

    type_promotion_kind: Any
    override_return_dtype: Any


op_dtype_propagation_rules: dict = {}


def register_op_dtype_propagation_rules(
    name: str, type_promotion_kind, override_return_dtype
) -> None:
    """State what an operation's result type is, for the handler to be built from."""

    op_dtype_propagation_rules[name] = OpDtypeRule(
        type_promotion_kind, override_return_dtype
    )


def upcast_compute_type(dtype):
    """The type a value is computed in, which is not always the type it is held in.

    A half-precision value is held in half precision and computed in single,
    because computing it in half loses digits that the result is meant to keep;
    a backend that computes in the type it holds has no such step, which is
    why this asks which backend is in play.
    """

    from . import config

    if (
        dtype in (tp.float16, tp.bfloat16)
        and config.codegen_upcast_to_fp32
        and get_current_backend() == "triton"
    ):
        return tp.float32
    return dtype


def cache_on_self(fn):
    """Remember what a method returned for the arguments it was given.

    The methods this decorates are pure functions of the object and its
    arguments, are asked the same question once per nest rather than once per
    call, and are expensive enough that recomputing them shows.  The cache lives
    on the object, and an object that is hashed cannot carry one, so the store
    is attached past the attribute lookup that would otherwise refuse it.
    """

    key = f"__{type(fn).__qualname__}_cache"

    def inner(self, *args, **kwargs):
        args_kwargs = (args, tuple(sorted(kwargs.items())))

        if not hasattr(self, key):
            object.__setattr__(self, key, {})

        cache = getattr(self, key)

        try:
            return cache[args_kwargs]
        except KeyError:
            pass
        result = fn(self, *args, **kwargs)
        cache[args_kwargs] = result
        return result

    inner.__name__ = fn.__name__
    inner.__qualname__ = fn.__qualname__
    inner.__doc__ = fn.__doc__
    return inner


def parallel_num_threads() -> int:
    """How many threads a kernel that is split is split into.

    A count that was set explicitly is the count; one that was not is however
    many the runtime says it has, so that a kernel written on a machine with
    many cores is not limited to a number chosen on a machine with few.
    """

    from . import config

    threads = config.cpp.threads
    if threads < 1:
        threads = get_num_threads()
    return threads


def get_num_threads() -> int:
    """How many threads the runtime says it has, and one when it says nothing."""

    try:
        import tensorplay as tp

        n = int(getattr(tp, "get_num_threads", lambda: 0)())
        return n if n > 0 else 1
    except Exception:
        return 1


class Placeholder(enum.Enum):
    """A name that is filled in only once the whole function has been written.

    A name is often not known while the body that uses it is being written --
    the body comes first, and the name depends on the body.  So the name is
    written as one of these and substituted at the end, once the body has
    told us what to call it.
    """

    # Stands for the name the compiled function will actually have.  Where the
    # prefix is "triton_", for instance, this becomes "triton_".
    KERNEL_NAME = "KERNEL_NAME"

    # Stands for a name carrying more detail than the plain prefix, used when
    # several functions must not share one name.
    DESCRIPTIVE_NAME = "DESCRIPTIVE_NAME"

def _align(nbytes: int) -> int:
    """Round up to the nearest multiple of ALIGN_BYTES"""
    return (nbytes + ALIGN_BYTES - 1) & -ALIGN_BYTES


class align(sympy.Function):
    """Symbolically round up to the nearest multiple of ALIGN_BYTES"""

    nargs = (1,)
    is_integer = True

    @classmethod
    def eval(cls, value: sympy.Expr) -> sympy.Expr | None:
        if isinstance(value, (int, sympy.Integer)):
            return _align(int(value))
        if _is_aligned(value):
            return value


def convert_to_symint(i: int | sympy.Expr) -> int | torch.SymInt:
    """
    Like convert_shape_to_symint, but operates on a single expression.
    """
    from .loops import V

    return (
        i
        if isinstance(i, int)
        else (
            int(i)
            if isinstance(i, sympy.Integer)
            else V.graph.sizevars.shape_env.create_symintnode(i, hint=None)
        )
    )


P = ParamSpec("P")
RV = TypeVar("RV", covariant=True)
FN_TYPE = Callable[Concatenate[Any, P], RV]


class CachedMethod(Protocol, Generic[P, RV]):
    @staticmethod
    def clear_cache(cache: Any) -> None: ...

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> RV: ...


def cache_property_on_self(
    fn: Callable[Concatenate[Any, P], RV],
) -> CachedMethod[P, RV]:
    """
    Variant of cache_on_self for properties. The only difference is the type signature.
    """

    return cache_on_self(fn)


def make_codegen_buffer() -> IndentedBuffer:
    """Construct the IndentedBuffer subclass matching the current codegen mode.

    Dual-wrapper mode -> DualIndentedBuffer (JIT and AOTI both active).
    Pure AOTI -> AotOnlyBuffer (writeline_aot writes; writeline_jit drops).
    Pure JIT  -> IndentedBuffer  (writeline_jit writes; writeline_aot drops).
    """
    from .loops import V

    if V.graph.is_dual_wrapper_mode:
        return DualIndentedBuffer()
    if V.graph.aot_mode:
        return AotOnlyBuffer()
    return IndentedBuffer()


class DelayReplaceLine(DeferredLineBase):
    """At end of codegen call `line.replace(key, value_fn())`"""

    def __init__(self, key: str, value_fn: Callable[[], str], line: str):
        super().__init__(line)
        self.key = key
        self.value_fn = value_fn

    def __call__(self) -> str:
        return self.line.replace(self.key, self.value_fn())

    def _new_line(self, line: str) -> DelayReplaceLine:
        return DelayReplaceLine(self.key, self.value_fn, line)


def get_benchmark_name() -> str | None:
    """
    An experimental API used only when config.benchmark_kernel is true.

    The benchmark name is only available at codegen time. So we can not
    directly call it in benchmark_all_kernels which is run after codegen.

    The function assumes the argument after --only is the benchmark name.
    It works for torchbench.py/hugginface.py/timm_models.py. But for ad-hoc
    scripts, this function may return None.

    There are 2 flavors of --only argument we need to handle:
    1. --only model_name
    2. --only=model_name
    """
    try:
        idx = sys.argv.index("--only")
        if (
            idx + 1 < len(sys.argv)
            and len(sys.argv[idx + 1]) > 0
            and sys.argv[idx + 1][0] != "-"
        ):
            return sys.argv[idx + 1]
    except ValueError:
        pass

    for arg in sys.argv:
        if arg.startswith("--only="):
            return arg[len("--only=") :]

    return None


def triton_version_uses_attrs_dict() -> bool:
    return get_triton_attrs_descriptor_version() == TritonAttrsDescriptorVersion.V4_DICT


def is_codegen_graph_partition_subgraph(wrapper: PythonWrapperCodegen) -> bool:
    from torch._inductor.codegen.wrapper import SubgraphPythonWrapperCodegen

    return (
        isinstance(wrapper, SubgraphPythonWrapperCodegen)
        and wrapper.partition_signatures is not None
    )


def is_using_cudagraph_partition() -> bool:
    return (
        torch._inductor.config.triton.cudagraphs
        or _unstable_customized_partition_wrapper.wrapper is not None
    ) and torch._inductor.config.graph_partition


#: Caches registered for clearing when the process decides the cached answers
#: no longer hold.  A cache that holds an answer derived from the environment
#: has to be droppable when that environment changes underneath it.
_registered_caches: list = []


def clear_on_fresh_cache(obj: Any) -> Any:
    """
    Use this decorator to register any caches that should be cache_clear'd
    with fresh_cache().
    """
    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    _registered_caches.append(obj)
    return obj


def _type_of(key: torch.dtype | None) -> str:
    # Use the function here to get rid of dependencies on the Triton during the codegen.
    # Refer to Triton implementation here:
    # https://github.com/triton-lang/triton/blob/98b5945d2aef679e00ebca8e07c35c3658ec76de/python/triton/runtime/jit.py#L238
    # `None` is nullptr.  Implicitly convert to *i8.
    if key is None:
        return "*i8"
    dtype_str = str(key).split(".")[-1]
    tys = {
        "bool": "i1",
        "float8e4nv": "fp8e4nv",
        "float8e5": "fp8e5",
        "float8e4b15": "fp8e4b15",
        "float8e4b15x4": "fp8e4b15x4",
        "float8_e4m3fn": "fp8e4nv",
        "float8_e5m2": "fp8e5",
        "float8_e4m3fnuz": "fp8e4b8",
        "float8_e5m2fnuz": "fp8e5b16",
        # TODO: remove when support is added in triton
        # https://github.com/triton-lang/triton/issues/6054
        "float8_e8m0fnu": "u8",
        "float4_e2m1fn_x2": "u8",
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int8": "i8",
        "int16": "i16",
        "int32": "i32",
        "int64": "i64",
        "uint8": "u8",
        "uint16": "u16",
        "uint32": "u32",
        "uint64": "u64",
    }
    # reinterpret can create triton type
    tys.update({v: v for v in list(tys.values())})
    return key if isinstance(key, str) else f"*{tys[dtype_str]}"


def expr_fits_within_32bit(e: sympy.Expr) -> bool:
    """Check if an expression fits within 32-bit integer range.

    NOTE: This function intentionally does not install guards. Callers are
    responsible for guarding (e.g. via check_leq) when they decide to use
    32-bit indexing based on this result.
    """
    from .loops import V

    int_max = torch.iinfo(torch.int32).max
    guarding_hint_or_throw = V.graph.sizevars.guarding_hint_or_throw
    has_guarding_hint = V.graph.sizevars.shape_env.has_guarding_hint

    if config.assume_32bit_indexing:
        V.graph.sizevars.check_leq(e, int_max)  # type: ignore[arg-type]
        return True

    # Allow for unhinted e as long as we can still statically prove
    # (e.g., via ValueRanges) that it is still in bounds
    if V.graph.sizevars.statically_known_true(e <= int_max):
        return True

    # AOTI doesn't guard on < 2**32, so checking hints isn't a viable option,
    # in case the hinted value is < 2**32, but the allowed range is larger.
    # However, to prevent possible perf regressions on pre-existing AOTI models
    # which don't set an upper bound on the valid range, we'll skip the check.
    # To recap:
    # - If using AOTI:
    #   - If allowed range has no upper bound, then check the hint to determine
    #       whether this fits in int32
    #   - If allowed range does have an upper bound, then obey the upper bound
    #       (check whether upper bound < int32_max) without checking the hint.

    if V.aot_compilation:
        # check whether value has an upper bound (1e20 is > INT64_MAX, assume
        # there is no upper bound if it can be larger than 1e20)
        if V.graph.sizevars.statically_known_true(e < 1e20):
            # if so, then assume int_max < upper bound < inf
            # so this could potentially have int64 values
            return False

    # Otherwise, the hint MUST exist and be in range
    return has_guarding_hint(e) and guarding_hint_or_throw(e) <= int_max


def device_supports_fp64(device: torch.device | None) -> bool:
    """Check if the given device supports float64."""
    if device is not None and device.type == "xpu":
        return torch.xpu.get_device_properties(device).has_fp64
    return True


#: The operations whose value is the same on every device in a group, and which
#: therefore cannot be measured on one of them and used on another.  A candidate
#: that performs one of these is a candidate whose time says nothing about how it
#: will run in a program, so it is recognised rather than measured.
COLLECTIVE_OPS = OrderedSet(
    (
        "torch.ops._c10d_functional.all_reduce.default",
        "torch.ops._c10d_functional.all_reduce_.default",
        "torch.ops._c10d_functional.all_gather_into_tensor.default",
        "torch.ops._c10d_functional.reduce_scatter_tensor.default",
        "torch.ops._c10d_functional.all_to_all_single.default",
        "torch.ops._c10d_functional_autograd.all_reduce.default",
        "torch.ops._c10d_functional_autograd.all_gather_into_tensor.default",
        "torch.ops._c10d_functional_autograd.reduce_scatter_tensor.default",
        "torch.ops._c10d_functional_autograd.all_to_all_single.default",
        "torch.ops._c10d_functional.isend.default",
        "torch.ops._c10d_functional.irecv.default",
        "torch.ops._c10d_functional.batch_p2p_ops.default",

    )
)


def is_collective_op(op_name: str) -> bool:
    """Whether an operation is one whose value is shared across devices."""

    return op_name in COLLECTIVE_OPS


class _Counter(int):
    """A number that can be asked to count.

    A subclass of int so that a count reads as the number it is everywhere it is
    read, and a subclass rather than a plain int so that ``+= 1`` on a name that
    has never been counted creates the name instead of failing.  Which is the
    whole of what a counter has to do here: a count that cannot be incremented
    is a place a measurement cannot be recorded, and a measurement that cannot be
    recorded is one nobody will know to ask about.
    """

    __slots__ = ()

    def __add__(self, other):
        return _Counter(int(self) + other)

    __radd__ = __add__

    def __iadd__(self, other):
        return _Counter(int(self) + other)


class _CounterGroup(dict):
    """The counts under one name, where each is a number that can be counted."""

    def __missing__(self, key):
        value = self[key] = _Counter(0)
        return value


class Counters:
    """How many times each thing happened, by the thing that happened.

    A counter rather than a log line because the questions a count answers are
    "is this happening at all" and "is it happening more than it was", and both
    are answered by a number that survives the run.  A line says it happened once.

    Reading a name that was never counted gives zero rather than raising, so that
    asking how often something happens is not itself a way of making it happen.
    """

    def __init__(self) -> None:
        self._counts: dict = {}

    def __getitem__(self, key) -> "_CounterGroup":
        group = self._counts.get(key)
        if group is None:
            group = self._counts[key] = _CounterGroup()
        return group

    def to_dict(self) -> dict:
        return {key: dict(group) for key, group in self._counts.items()}

    def __repr__(self) -> str:
        return "Counters(%r)" % (self.to_dict(),)


counters = Counters()
counters["inductor"]


#: The name a measured call is recorded under, so that the device work belonging
#: to one repetition can be told from the device work belonging to the cache
#: clearing around it.  A name rather than a time window because the two are
#: recorded on different sides of the device and are matched by what was running,
#: not by when.
_DO_BENCH_PROFILE_EVENT_NAME = "inductor_do_bench_using_profiling"


def _gpu_device_module() -> Any:
    """The module that talks to the device being measured on.

    Which module that is depends on the device, and the operations needed here --
    waiting for the device, an event that can be timed, whether the device is
    there at all -- are the same set on each of them under a different name.
    """

    from .runtime.benchmarking import _get_default_gpu_device_type

    return getattr(tp, _get_default_gpu_device_type())


def do_bench_using_profiling(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    is_vetted_benchmarking: bool = False,
) -> float:
    """How long a call takes on the device, read off a profiler trace.

    A clock around a call says how long the call took to be asked for and
    answered, which for a call that waits is mostly the waiting.  A trace says
    how long the device was busy, which is the part that a faster kernel would
    actually save.  Which of the two is wanted depends on what the number is for,
    so both are available and this one is asked for by name.

    The benchmarking lock and the distortion guard are applied here rather than
    around the call site, because a second measurement running at the same time
    makes this one wrong in a way no caller can detect from its own result.
    """

    from .runtime.benchmarking import (
        gpu_benchmark_lock,
        may_distort_benchmarking_result,
    )

    locked_bench = gpu_benchmark_lock(_do_bench_using_profiling)
    return may_distort_benchmarking_result(locked_bench)(
        fn, warmup, rep, is_vetted_benchmarking
    )


def _get_do_bench_profile_result(
    kineto_events: Iterable[Any],
    profiler_events: Iterable[Any],
    n_repeat: int,
    expected_device_type: Any,
) -> float:
    """The device time of the measured calls, in milliseconds.

    Two sets of events have to be brought together.  The first says what the
    device did and when; the second says what each of those pieces of work was
    asked for by.  A piece of device work counts towards this measurement when
    the call that asked for it is one of the measured calls, which is what the
    correlation between the two sets says -- so the measured calls are marked,
    their own children are marked with them since a call that waits has work
    under it, and then the device work is added up over the marks.
    """

    benchmark_event_ids: set = set()

    def collect_cpu_event_ids(event: Any) -> None:
        if event.device_type != tp.DeviceType.CPU:
            return
        benchmark_event_ids.add(event.id)
        for child in event.cpu_children:
            collect_cpu_event_ids(child)

    benchmark_events = [
        event
        for event in profiler_events
        if event.name == _DO_BENCH_PROFILE_EVENT_NAME
        and event.device_type == tp.DeviceType.CPU
    ]
    if len(benchmark_events) != n_repeat:
        raise RuntimeError(
            f"Expected {n_repeat} {_DO_BENCH_PROFILE_EVENT_NAME} profiling "
            f"events. Found {len(benchmark_events)} events."
        )

    for event in benchmark_events:
        collect_cpu_event_ids(event)

    device_time_us = 0.0
    for event in kineto_events:
        activity_type = event.activity_type()
        # Membership is decided by which operation the work belongs to, not by
        # whether the launch that issued it was recorded.  Deciding it by the
        # launch would need a second chance for a work whose launch was not
        # recorded, matched on the correlation id, and that second chance cannot
        # be taken here: the collector numbers launches from one and operations
        # are numbered from the same low range, so a launch id and an operation
        # id are the same numbers and matching one against the other finds
        # coincidences.  An operation this collector could not name is left out,
        # which undercounts rather than charging one operation's time to another.
        if (
            event.device_type() == expected_device_type
            and activity_type != "gpu_user_annotation"
            and event.op_slot() in benchmark_event_ids
            and event.name() != "Context Sync"
        ):
            device_time_us += (event.end_ns() - event.start_ns()) / 1000.0

    if device_time_us <= 0:
        raise RuntimeError(
            f"Failed to capture device events for "
            f"{_DO_BENCH_PROFILE_EVENT_NAME}."
        )

    return device_time_us / 1000.0 / n_repeat


def _prime_device_collector() -> None:
    """Make the device collector ready to report, if it is not already.

    The collector allocates its record buffers the first time it is asked for
    them, and the work it is asked to record while it is doing that is not
    recorded.  A session started in a process that has never run one therefore
    reports no device work at all, which is indistinguishable from a session in
    which the device did nothing -- and a measurement that reads as the second
    is a measurement of the collector rather than of what was measured.

    So the first session in a process is spent on a throwaway call, whose
    records are the ones lost, and the measured session is the second one.  Only
    the first pays for this, and a process that has already run a session pays
    nothing.
    """

    import tensorplay.profiler as _profiler

    if _profiler.device_activity_ready():
        return
    activities = [tp.profiler.ProfilerActivity.CPU]
    for name in ("CUDA", "XPU", "MTIA"):
        activity = getattr(tp.profiler.ProfilerActivity, name, None)
        if activity is not None:
            activities.append(activity)
            break
    with tp.profiler.profile(activities=activities):
        pass
    # A session with no work in it still has to have asked the device for its
    # buffers, which is what starting and stopping one does.


def _do_bench_using_profiling(
    fn: Callable[[], Any],
    warmup: int = 25,
    rep: int = 100,
    is_vetted_benchmarking: bool = False,
) -> float:
    """How long a call takes on the device, in milliseconds.

    The call is run once to see whether it works at all, then five more times to
    find out roughly how long it takes, and the number of repetitions in each of
    the warmup and the measured parts is then chosen to fill the requested time
    rather than taken as given: a call that takes a millisecond and a call that
    takes a second are both worth measuring, and a fixed repetition count
    measures the first for a second and the second for a millisecond.

    The cache is cleared before each repetition, because a repetition that reads
    what the one before it wrote measures the memory system rather than the call.
    """

    from .runtime.benchmarking import may_ban_benchmarking

    if not is_vetted_benchmarking:
        may_ban_benchmarking()

    _prime_device_collector()

    device_module = _gpu_device_module()
    device_type = device_module.__name__.rsplit(".", 1)[-1]
    device_type_upper = device_type.upper()
    fn()
    device_module.synchronize()
    cache = tp.empty(int(256e6 // 4), dtype=tp.int32, device=device_type)

    # Roughly how long one call takes, which is what the repetition counts are
    # derived from.
    start_event = device_module.Event(enable_timing=True)
    end_event = device_module.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        cache.zero_()
        fn()
    end_event.record()
    device_module.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5

    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    for _ in range(n_warmup):
        fn()

    device_module.synchronize()
    profile_activity = getattr(tp.profiler.ProfilerActivity, device_type_upper)
    with tp.profiler.profile(
        activities=[tp.profiler.ProfilerActivity.CPU, profile_activity],
    ) as profile:
        for _ in range(n_repeat):
            cache.zero_()
            with tp.profiler.record_function(_DO_BENCH_PROFILE_EVENT_NAME):
                fn()
        device_module.synchronize()

    log.debug("raw events")
    if log.isEnabledFor(logging.DEBUG):
        log.debug(
            profile.key_averages().table(
                sort_by="self_device_time_total", row_limit=-1
            )
        )

    result = _get_do_bench_profile_result(
        profile.kineto_results(),
        profile.events(),
        n_repeat,
        getattr(tp.DeviceType, device_type_upper),
    )

    log.debug("profiling time breakdown")
    log.debug("profiling results: %s ms", result)
    return result




def get_layout_symints(node) -> OrderedSet:
    """The shapes a value's layout is written in terms of.

    A layout says where an element is with an extent, a stride and an offset,
    and each of those can be written in terms of a shape that is not yet
    settled.  Whoever reads the value has to be handed those shapes, because a
    kernel cannot be given a stride it cannot compute.

    Accepts one value or several, so a caller with a list of them does not have
    to fold it itself.
    """

    from . import ir as _ir

    if isinstance(node, (list, tuple, set, frozenset)):
        collected: OrderedSet = OrderedSet()
        for one in node:
            collected.update(get_layout_symints(one))
        return collected

    found: OrderedSet = OrderedSet()
    layout = node.maybe_get_layout() if hasattr(node, "maybe_get_layout") else None
    if layout is None:
        return found
    if not isinstance(layout, _ir.Layout):
        raise AssertionError(
            f"expected a layout or nothing, but the value's layout is {layout!r}"
        )
    found.update(get_free_symbols(layout.size, False))
    found.update(get_free_symbols(layout.stride, False))
    found.update(get_free_symbols(layout.offset, False))
    if isinstance(layout, _ir.MutationLayoutSHOULDREMOVE):
        # A layout that writes in place is expressed in terms of the layout it
        # writes over, so the shapes that one is written in terms of are shapes
        # this one is too.
        found.update(get_layout_symints(layout.target))
    return found


#: The width a memory access is aligned to, in bytes.  A load of several
#: elements at once has to start at an address the hardware can address that
#: way, and this is the narrowest width that is enough for any of the widths
#: used, so a value that is aligned to it is aligned for all of them.
ALIGNMENT = 16

#: The width an offset into a buffer must be a multiple of for a view of it to
#: be addressable element by element.  Settled rather than derived: it is the
#: width the widest vector load wants, and a wider one would rule out views that
#: are perfectly addressable.
GPU_ALIGN_BYTES = 16


def get_sympy_Expr_dtype(val) -> Any:
    """The element type an expression has, as far as its own kind says.

    A whole number is a whole number however it was arrived at, and anything
    else that is a number at all is a real number, so the kind of the expression
    is what decides the type rather than anything about how it was written.
    """

    import sympy

    if not isinstance(val, sympy.Expr):
        raise AssertionError(
            "only support sympy.Expr as input to get_sympy_Expr_dtype"
        )
    if val.is_integer:
        return tp.int64
    return tp.float64


def prefix_is_reduction(prefix: str) -> bool:
    """Whether a symbol's name marks it as one standing for a reduction axis.

    A symbol says what it stands for in the letters its name begins with, and
    a reduction axis is the one that begins with the letter for reductions, so
    the question is a question about the name.
    """

    return prefix[0] == "r"


def get_max_numwarps() -> int:
    """How many warps fit in one block on the device being compiled for.

    A block holds a whole number of warps, so the answer is the most threads a
    block may hold divided by how wide a warp is on this device -- and both of
    those are properties of the device rather than numbers to assume.
    """

    from .runtime.hints import DeviceProperties

    if tp.cuda.is_available():
        device = tp.device("cuda", tp.cuda.current_device())
        props = DeviceProperties.create(device)
        warp_size = props.warp_size_or_default
        max_threads_per_block = props.max_threads_per_block
        if max_threads_per_block is None:
            raise AssertionError("expected max_threads_per_block to be set")
        return max_threads_per_block // warp_size
    return 32


def dominated_nodes(initial_queue, skip_filter=None) -> "OrderedSet":
    """The values that depend on the ones named, and the ones named.

    A value depends on another when reading it means reading the other, so the
    set is everything reachable from the starting values by following readers.
    Which is what a value-range analysis needs in order to be pessimistic about
    a value without working the range out: a load's range is unknown, so
    anything computed from a load is unknown too.

    ``skip_filter`` leaves a reader out, for a reader whose value is not
    computed from what it read.
    """

    from . import ir as _ir

    queue = list(initial_queue)
    dominated: "OrderedSet" = OrderedSet(queue)
    while queue:
        node = queue.pop()
        for user in getattr(node, "users", ()):
            if skip_filter is not None and skip_filter(user):
                continue
            if user not in dominated:
                dominated.add(user)
                queue.append(user)
    return dominated


def snode_args_kwargs(snode) -> tuple[list[Any], dict[str, Any]]:
    """The arguments a piece's call takes, with each one in the form a caller wants.

    A call is recorded as two streams -- the arguments that name memory and the
    ones that do not -- because a scheduler has to know which is which.  A
    caller that is going to *make* the call wants the opposite: one flat list,
    with anything that is not a piece of the graph turned into a real value it
    can pass.

    That is what this does, and the order it does it in matters. The arguments
    a schema declares as positional are removed from the keyword half, so that
    filling in the schema's defaults cannot add one the caller already gave
    positionally -- the same argument twice is a call that either fails or, worse,
    means something else.
    """

    from tensorplay._ops import OpOverload
    from tensorplay.utils import _pytree as pytree

    from . import ir as _ir

    node = snode.node
    if isinstance(node, _ir.FallbackKernel):
        args, kwargs = node.unflatten_args(node.inputs, node.constant_args)
    else:
        args = [*node.inputs, *node.constant_args]
        kwargs = node.kwargs

    args = node.fill_non_provided_args(args, kwargs)
    kwargs = dict(kwargs)

    op_overload = getattr(node, "op_overload", None)
    if isinstance(op_overload, OpOverload):
        positional_names = [
            argument.name
            for argument in op_overload._schema.arguments
            if not argument.kwarg_only
        ]
        for name in positional_names[: len(args)]:
            kwargs.pop(name, None)

    flat_args, spec = pytree.tree_flatten((args, kwargs))

    def is_tensor_arg(value) -> bool:
        # A generator's state and an opaque value are graph values with no
        # tensor behind them, so they are passed as they are.
        return isinstance(value, _ir.IRNode) and not isinstance(
            value, (_ir.GeneratorState, _ir.OpaqueObjectState)
        )

    flat_args = [
        _ir.ir_node_to_tensor(arg, replace_symbols_with_hints=True)
        if is_tensor_arg(arg)
        else arg
        for arg in flat_args
    ]
    # A piece of the graph becomes a real tensor of the right shape, so that a
    # caller can hand it to something that is not this compiler.
    flat_args = [
        tp.empty(value.size(), dtype=value.dtype, device=value.device)
        if isinstance(value, tp.Tensor)
        else value
        for value in flat_args
    ]
    return pytree.tree_unflatten(flat_args, spec)
