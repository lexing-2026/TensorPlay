"""Writing a launch for the kernel-writing runtime.

A launch written here is not a whole kernel: it is the part of one that is the
same whatever the kernel computes.  Which runtime built it, which machine it
is for, and which names the runtime reads as its own are all settled before a
single line of the kernel's own arithmetic exists, because all three decide
what the generated module has to import and what has to be written into the
record the runtime reads back.

The base class arrives with the shared kernel machinery.  Until it does, the
two things here -- what a generated module imports, and what is recorded about
how it was written -- are whole without it, and are read by the wrapper that
writes the module.  The import is resolved when this module is loaded, so the
base is picked up as soon as it exists rather than having to be written in
twice.
"""

from __future__ import annotations

import collections
import dataclasses
import itertools
import re
import sympy
from typing import Any

import tensorplay as tp

from .....graph.experimental.sympy_functions import (
    OrderedSet,
    prefix_str,
    symbol_is_type,
    SymT,
)
from .. import config
from ..shape_propagation import get_broadcasted_shape
from .common import CSE, CSEVariable
from ..utils import IndentedBuffer


try:  # The shared kernel machinery this builds on.
    from .simd import SIMDKernel
except ImportError:  # pragma: no cover - until that machinery is here
    SIMDKernel = object


class TritonKernel(SIMDKernel):  # type: ignore[misc,valid-type]
    """A launch written for the kernel-writing runtime.

    Only the parts that are the same for every such launch live here.  What the
    launch computes is the base class's business, along with the shared
    machinery for reading a value, computing one, and keeping the result.
    """

    @classmethod
    def gen_common_triton_imports(cls) -> str:
        """The imports every generated launch needs, as source to splice in.

        A generated module is a module of its own, so it cannot borrow the
        names of the module that wrote it: the runtime, the language it is
        written in, and the device library have to be imported by name.  The
        device library is imported under a second name because the generated
        arithmetic says that name, and the runtime's own helpers are imported
        under theirs because the generated launcher says theirs.

        Anything only some launches need -- a tracing profiler, a set of
        hardware descriptors -- is left out here and written by the code that
        knows the launch needs it, because a module that imports something it
        never uses pays for it on every launch.
        """

        imports = IndentedBuffer()
        imports.splice(
            """
            import triton
            import triton.language as tl
            """
        )
        try:
            import triton.language.extra.cuda.libdevice as libdevice  # noqa: F401

            imports.splice(
                """
                import triton.language.extra.cuda.libdevice as libdevice
                """
            )
        except ImportError:
            pass
        try:
            import triton.language.extra.tlx  # noqa: F401

            imports.splice(
                """
                import triton.language.extra.tlx as tlx  # noqa: F401
                """
            )
        except ImportError:
            pass
        imports.splice(
            """
            from tensorplay.compiler.backends.stax.runtime import (
                triton_helpers,
                triton_heuristics,
            )
            from tensorplay.compiler.backends.stax.runtime.triton_helpers import (
                libdevice,
                math as tl_math,
            )
            from tensorplay.compiler.backends.stax.runtime.hints import (
                AutotuneHint,
                DeviceProperties,
                ReductionHint,
                TileHint,
            )
            """
        )
        if config.triton.proton_profiling:
            imports.splice(
                """
                import triton.profiler as proton
                import triton.profiler.language as pl
                pl.enable_semantic('triton')
                """
            )
        return imports.getvalue()

    @classmethod
    def triton_meta_common(cls) -> dict[str, Any]:
        """What the runtime is told about how to treat the launch itself.

        These are the runtime's own switches rather than ours: whether it may
        fold the arithmetic of the launch, whether it may launch a dependent
        launch early, and whether it flushes denormals to zero.  The last is
        off, because a denormal flushed to zero is a different answer rather
        than a faster one, and a caller asking for exact arithmetic has not
        asked for that.
        """

        return {
            "enable_fp_fusion": not config.emulate_precision_casts,
            "launch_pdl": False,
            "disable_ftz": False,
        }

    @classmethod
    def inductor_meta_common(cls) -> dict[str, Any]:
        """What is recorded about the settings this launch was written under.

        A launch is cached, and a cache entry is only good for the settings it
        was written under -- a launch written to check an index is not
        interchangeable with one written not to, and a launch written to be
        deterministic is not interchangeable with one that was not.  So every
        setting that changes the generated code is written into the record, and
        the runtime compares it before reusing an entry.

        The runtime and the machine are in the record too, and they are asked
        of the runtime rather than assumed, since a launch built by one runtime
        for one machine is not readable by another.
        """

        from tensorplay.utils._triton import triton_hash_with_backend

        inductor_meta = {
            "backend_hash": triton_hash_with_backend(),
            "assert_indirect_indexing": config.assert_indirect_indexing,
            "max_autotune": config.max_autotune,
            "deterministic": config.deterministic,
            "emulate_precision_casts": config.emulate_precision_casts,
            "force_disable_caches": config.force_disable_caches,
            "store_cubin": config.triton.store_cubin,
            "force_filter_reduction_configs": (
                config.test_configs.force_filter_reduction_configs
            ),
        }

        if config.profile_bandwidth:
            inductor_meta["profile_bandwidth"] = config.profile_bandwidth
            inductor_meta["profile_bandwidth_output"] = config.profile_bandwidth_output
            inductor_meta["profile_bandwidth_with_do_bench_using_profiling"] = (
                config.profile_bandwidth_with_do_bench_using_profiling
            )

        return inductor_meta

    def triton_tensor_ndim(self) -> int:
        """How many dimensions a launch's value has.

        A launch may compute over more ranges than its value has dimensions --
        a reduction ranges over something the result does not -- so the count
        is of the ranges that landed on a dimension of the value, not of the
        ranges.
        """

        return sum(int(tree.tensor_dim is not None) for tree in self.range_trees)

    def indexing_size_str(self, i: int) -> str:
        """A subscript that selects one dimension and leaves the rest whole.

        Written with one ``None`` per dimension and a ``:`` in the one wanted,
        which is how a subscript says "all of this one" without having to say
        how much, and therefore without the result depending on the block size
        the launch happens to be using.
        """

        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[i] = ":"
        return f"[{', '.join(sizes)}]"

    def dense_size_list(self) -> list:
        """The size of the value, one entry per dimension.

        A dimension is one block wide however large the range behind it is,
        because that is how much of it a launch holds at once.  A dimension no
        range landed on is one element wide, which is what a length of one
        says.
        """

        sizes = ["1"] * self.triton_tensor_ndim()
        for tree in self.active_range_trees():
            if tree.tensor_dim is not None:
                sizes[tree.tensor_dim] = tree.block_size_str()
        return sizes

    def dense_size_str(self) -> str:
        return f"[{', '.join(self.dense_size_list())}]"

    def create_constant_mask(self, entry) -> str:
        """A guard that is true everywhere, for a range that needs no guard.

        A range that covers its whole output has nothing to hold back, so every
        position in it is in range.  The guard is still written, because the
        reads underneath it are written under a guard whether or not this one
        does any work, and writing the reads differently when the guard is
        known to be all-true would mean two spellings of every read.

        When the range does not land on a dimension of the value there is
        nothing to select, so the guard is the value itself; otherwise the
        value is indexed down to the one dimension the range covers, since a
        wider guard would be a guard over dimensions the range says nothing
        about.
        """

        if entry.tensor_dim is None:
            return (
                f"{entry.mask_name()} = "
                f"tl.full({self.dense_size_str()}, True, tl.int1)"
            )
        sizes = ["None"] * self.triton_tensor_ndim()
        sizes[entry.tensor_dim] = ":"
        suffix = ", ".join(sizes)
        return (
            f"{entry.mask_name()} = "
            f"tl.full([{entry.block_size_str()}], True, tl.int1)[{suffix}]"
        )


#: Types the language spells under a different name than the dtype does, and
#: types it has no spelling for at all.  A type in the first group is renamed;
#: a type in the second is written as the nearest type of the same width,
#: because a store of the real type would not compile and a store of a wider
#: type would be a different answer.
_TRITON_TYPE_MAPPING = {
    "tl.bool": "tl.int1",
    "tl.float8_e4m3fn": "tl.float8e4nv",
    "tl.float8_e5m2": "tl.float8e5",
    "tl.float8_e4m3fnuz": "tl.float8e4b8",
    "tl.float8_e5m2fnuz": "tl.float8e5b16",
    "tl.float8_e8m0fnu": "tl.uint8",
    "tl.float4_e2m1fn_x2": "tl.uint8",
}

_TYPE_TO_TRITON = {v: k for k, v in _TRITON_TYPE_MAPPING.items()}

_MODULE_PREFIX = re.compile(r"^.*[.]")


def triton_type(dtype) -> str:
    """The name the language spells an element type under.

    A dtype is written as its own path, of which the language keeps only the
    last part: a type reached through a namespace is written as itself, not
    through where it was found.  A type the language spells differently is
    renamed, and one it does not spell is left as it is, which is the case
    where the language has no such type.
    """

    triton_type_name = _MODULE_PREFIX.sub("tl.", str(dtype))
    return _TRITON_TYPE_MAPPING.get(triton_type_name, triton_type_name)


def triton_store_type(dtype) -> str:
    """The name the language spells a store of this type under.

    A store of a truth value is a store of a byte: the language has a
    one-bit-per-lane type for arithmetic but none for memory, and a value
    written through a pointer has to be wider than a bit.  So the byte is what
    is written, and the value read back is the same one bit per lane.
    """

    if dtype == tp.bool:
        dtype = tp.int8
    return triton_type(dtype)


class TritonSymbols:
    """The names a launch's own arithmetic is written in.

    A launch does not compute over one flat range: it computes over a block at
    a time, and a block is a few dimensions wide at once.  So the positions
    inside a launch are several kinds of symbol rather than one -- one per
    block dimension, one per reduction -- and each kind has a name the
    generated arithmetic uses, a name its offset is written under, and a name
    its size is written under.  They are collected here so that the emitter
    and the runtime agree on them: a launch that computed a position under one
    name and read it under another would compile and be wrong.
    """

    reduction_types = OrderedSet([SymT.R0_INDEX, SymT.R1_INDEX])
    block_types = OrderedSet([SymT.XBLOCK, SymT.YBLOCK, SymT.ZBLOCK, *reduction_types])

    block_offsets = {
        symt: sympy.Symbol(f"{prefix_str[symt]}offset", integer=True, nonnegative=True)
        for symt in block_types
    }

    block_sizes = {
        symt: sympy.Symbol(
            f"{prefix_str[symt].upper()}BLOCK", integer=True, positive=True
        )
        for symt in block_types
    }

    @classmethod
    def get_block_shape(cls, expr: sympy.Expr) -> tuple:
        """The shape of a value computed from an expression.

        A launch's values are not all the same shape: a position built from
        two block dimensions is a pair, one built from a size is a number, and
        one built from a stored value is whatever shape that value was stored
        with.  The shape of the whole follows from the shapes of what it is
        built from, broadcast together -- one and one is one, a pair and one is
        a pair -- so it is worked out from the free symbols of the expression
        rather than recorded, because a shape is not written down anywhere for
        a value that is only ever computed.
        """

        from ..loops import V

        expr_shape: tuple = ()
        for var in expr.free_symbols:
            if symbol_is_type(var, SymT.TMP):
                cse_var = V.kernel.cse.varname_map[var.name]
                var_shape = tuple(cse_var.shape)
            elif symbol_is_type(
                var,
                (
                    SymT.UNBACKED_INT,
                    SymT.SIZE,
                    SymT.PRECOMPUTED_SIZE,
                    SymT.INDEX,
                    SymT.FLOAT,
                    SymT.UNBACKED_FLOAT,
                ),
            ):
                # A size or a plain number is one value, not a block of them.
                var_shape = ()
            else:
                node = V.kernel.range_tree_nodes.get(var)
                if node is None:
                    raise AssertionError(f"unregistered range symbol: {var.name}")
                tree = node.root
                ndim = V.kernel.triton_tensor_ndim()
                shape = ["1"] * ndim
                if tree.tensor_dim is None:
                    # A range with no dimension of its own is a number.
                    var_shape = ()
                else:
                    shape[tree.tensor_dim] = str(cls.get_block_size(tree))
                    var_shape = tuple(shape)

            expr_shape = get_broadcasted_shape(expr_shape, var_shape)

        if expr_shape is None:
            raise AssertionError("a shape cannot be unknown here")

        return expr_shape

    @classmethod
    def get_block_size(cls, tree) -> sympy.Expr:
        """How wide one block of a range is."""

        return tree.root.block_size()

    @classmethod
    def get_block_offset(cls, tree) -> sympy.Expr:
        """Where the first block of a range starts."""

        return tree.root.block_offset()

    @classmethod
    def mask_name_for_symbol(cls, kernel, symbol: sympy.Symbol) -> str | None:
        """The name of the mask that guards reads at this position, if any.

        A read is written under a mask when the positions it covers are not all
        inside the range -- the last block of a range usually reaches past its
        end.  Which mask that is follows from the range the position belongs
        to, so it is read off the range rather than rebuilt from the name,
        because a name rebuilt from a prefix would be a guess about a range
        that may not be the one meant.

        A position belonging to no range has no mask, which is the ordinary
        case: a position inside the range needs no guard.
        """

        if (node := kernel.range_tree_nodes.get(symbol)) is not None:
            return node.root.mask_name()
        for symt in cls.block_types:
            if symbol_is_type(symbol, symt):
                return f"{prefix_str[symt]}mask"
        return None

    @classmethod
    def is_reduction_index_symbol(cls, kernel, symbol: sympy.Symbol) -> bool:
        """Whether a position runs along a reduction rather than across the output."""

        for symt in cls.reduction_types:
            if symbol_is_type(symbol, symt):
                return True
        node = kernel.range_tree_nodes.get(symbol)
        return node is not None and node.root.is_reduction


class TritonCSEVariable(CSEVariable):
    """A value held for reuse, together with the guards its reads need.

    A read is written under a mask when the positions it covers reach past the
    range they belong to, which is what the last block of a range does.  A
    value computed from such a read carries that guard with it, because it is
    not only the read that has to be masked: every value computed from a
    masked read is masked the same way, and re-deriving that at each use would
    mean walking back through the arithmetic to find out whether a guard
    applies.  So the guards are collected as the value is built, and used
    wherever the value is.

    A value here always has a type and a shape.  A value with neither cannot be
    written into a launch -- there would be no way to spell its declaration --
    so it is an error rather than something filled in later.
    """

    def __init__(self, name: str, bounds, dtype, shape=None) -> None:
        super().__init__(name, bounds, dtype, shape=shape)
        #: The masks a read behind this value was written under.
        self.mask_vars: OrderedSet[str] = OrderedSet()
        if dtype is None:
            raise AssertionError("a value held for reuse must have a type")
        if shape is None:
            raise AssertionError("a value held for reuse must have a shape")

    def update_on_args(self, name, args, kwargs) -> None:
        """Carry the guards of everything this value is built from onto it."""

        from ..loops import V

        for arg in args:
            if isinstance(arg, TritonCSEVariable):
                self.mask_vars.update(arg.mask_vars)
            elif isinstance(arg, sympy.Symbol):
                # A position is guarded only when it is used to read somewhere
                # the range does not cover, which is not known from the
                # position alone -- so the range it belongs to is asked.
                if (
                    mask_name := TritonSymbols.mask_name_for_symbol(V.kernel, arg)
                ) is not None:
                    self.mask_vars.add(mask_name)


@dataclasses.dataclass
class BlockParameters:
    """One read or write, as the block it moves.

    A launch moves a block at a time rather than an element at a time, so what
    a read is given is not a position but a rectangular run of positions: where
    the run starts in each dimension, how wide it is, how the buffer is laid
    out, and how the dimensions are ordered.  Those are kept together here
    because they are only meaningful together -- a block of a shape that does
    not fit the buffer it is read from is not a narrower read, it is a wrong
    one -- and because the ways of spelling a block all need the same four.
    """

    shape: list = dataclasses.field(default_factory=list)
    block_shape: list = dataclasses.field(default_factory=list)
    strides: list = dataclasses.field(default_factory=list)
    offsets: list = dataclasses.field(default_factory=list)

    @dataclasses.dataclass
    class StrideSorter:
        """Which order the dimensions were read in, kept across a reordering.

        A block may be described in one order and moved in another: the
        description follows the layout of the buffer, the move follows the
        order the launch computes in.  The two are reconciled at run time by
        transposing the block, and that transposition is only possible if the
        order it started in is recorded rather than assumed.
        """

        original_strides: list = dataclasses.field(default_factory=list)
        sorted_indices: list = dataclasses.field(default_factory=list)

        def unsort(self, x: list) -> list:
            """Put a list that was reordered by the sort back where it was."""

            if len(self.sorted_indices) == 0:
                return x
            assert len(x) == len(self.sorted_indices)
            unsorted = [None] * len(x)
            for i, j in enumerate(self.sorted_indices):
                unsorted[j] = x[i]
            return unsorted

    stride_sorter: "BlockParameters.StrideSorter | None" = None


@dataclasses.dataclass
class BlockDescriptorOptions:
    """What a read or write is, and the two ways a launch can spell it.

    The same block can be handed to the runtime as a pointer into the buffer
    together with the shape and strides to walk it, or as a descriptor the
    runtime builds once and reuses.  Both need the same description, so it is
    held once here and each way of spelling it is a subclass that only knows
    how to print the call.
    """

    params: BlockParameters
    constant_offset: sympy.Expr
    order: list
    mask_vars: OrderedSet
    broadcast_shape: list
    broadcasting_dims: list
    final_shape: list

    def format(self, name: str, roffset: bool = True) -> str:
        raise NotImplementedError


@dataclasses.dataclass
class TensorDescriptorOptions(BlockDescriptorOptions):
    """The block spelled as a descriptor the runtime builds once.

    A descriptor carries the shape, the strides and the block shape, so the
    runtime does not walk them per access.  That is worth it for a block read
    many times and not worth the setup for one read once, which is why this is
    one spelling of the block rather than the spelling.
    """

    def format(self, name: str, roffset: bool = True) -> str:
        from ..loops import V

        f = V.kernel.index_to_str
        args = [
            (
                f"{name} + ({f(self.constant_offset)})"
                if self.constant_offset != 0
                else name
            ),
            f"shape={f(self.params.shape)}",
            f"strides={f(self.params.strides)}",
            f"block_shape={f(self.params.block_shape)}",
        ]

        return f"tl.make_tensor_descriptor({', '.join(args)})"


@dataclasses.dataclass
class BlockPtrOptions(BlockDescriptorOptions):
    """The block spelled as a pointer plus the shape and strides to walk it.

    No setup and no descriptor: the call takes the buffer's address and the
    four things needed to walk it.  What the call costs per access is why the
    descriptor spelling exists, so which of the two is worth it is decided by
    how many times the block is moved rather than here.
    """

    def replace_offset(self, expr: sympy.Expr, replacement: sympy.Expr, symt) -> sympy.Expr:
        """Put a different start in place of one dimension's offset."""

        roffset = TritonSymbols.block_offsets[symt]
        return sympy_subs(expr, {roffset: replacement})

    def remove_roffsets(self, expr: sympy.Expr) -> sympy.Expr:
        """Drop the reduction offsets, leaving a pointer that does not move.

        A read that is not advanced between uses needs no offset to advance
        from, and writing an offset that is always zero would say the read
        moves when it does not.
        """

        for symt in TritonSymbols.reduction_types:
            expr = self.replace_offset(expr, sympy.Integer(0), symt)
        return expr

    def format(self, name: str, roffset: bool = True) -> str:
        from ..loops import V

        f = V.kernel.index_to_str
        offsets = [*self.params.offsets]
        if not roffset:
            offsets = [self.remove_roffsets(offset) for offset in offsets]
        args = [
            (
                f"{name} + ({f(self.constant_offset)})"
                if self.constant_offset != 0
                else name
            ),
            f"shape={f(self.params.shape)}",
            f"strides={f(self.params.strides)}",
            f"block_shape={f(self.params.block_shape)}",
            f"order={f(self.order)}",
            f"offsets={f(offsets)}",
        ]
        return f"tl.make_block_ptr({', '.join(args)})"

    def advance_roffset(self, symt) -> list:
        """How far each offset moves between one use and the next.

        The movement is one block along the dimension being reduced, because
        that is the step the loop takes: the first use starts at the beginning
        and the next starts one block later.  So the movement is the block size
        of that dimension, which is the difference between the offset written
        with that block size and the same offset written with zero.
        """

        rblock = TritonSymbols.block_sizes[symt]
        return [
            self.replace_offset(offset, rblock, symt)
            - self.replace_offset(offset, sympy.S.Zero, symt)
            for offset in self.params.offsets
        ]


@dataclasses.dataclass
class FixedTritonConfig:
    """Block sizes that were settled in advance and are not chosen again.

    A launch whose tuning was given rather than searched has its block sizes
    already decided, and the emitter has to measure its blocks against those
    rather than against whatever would be convenient.  So the given sizes are
    held as a mapping that can be read by name, and a name that was never given
    is absent rather than defaulted -- defaulting it would quietly measure
    against a size the launch will not use.
    """

    config: dict

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config


@dataclasses.dataclass
class TritonOpTraceEntry:
    """One operation a launch performed, recorded for tracing.

    Which operation, what it was given and what it produced.  The arguments are
    held as a pair rather than a mapping so that a call written with the same
    argument twice reads the same way here as it did at the call.
    """

    name: str
    args: tuple
    kwargs: tuple
    result: object


class HelperFunctions:
    """The helper functions a launch defines for itself, in the order defined.

    A launch often needs a function of its own -- a reduction written out, a
    comparison against a bound.  Each is defined once and named, and the same
    text asked for twice is one function rather than two, because a launch that
    defined both would carry the body twice and the runtime would compile both.

    The order is kept because a launch's helpers may call each other, and a
    function has to be defined before it is called.
    """

    _templates_seen: dict
    finalized_helpers: list

    def __init__(self) -> None:
        self._templates_seen = {}
        self.finalized_helpers = []

    def add(self, template_code: str, *, base_name: str = "_triton_helper_fn") -> str:
        """Define a helper from text whose name is left to be filled in.

        The text is a function definition with the name written as a format
        specifier.  The name given is the nth helper, so two different bodies
        get two different names, and the same body asked for twice gets the
        name it already has.
        """

        existing_name = self._templates_seen.get(template_code)
        if existing_name is not None:
            return existing_name

        name = f"{base_name}{len(self.finalized_helpers)}"
        self._templates_seen[template_code] = name
        self.finalized_helpers.append(template_code.format(name=name))
        return name

    def __iter__(self):
        return iter(self.finalized_helpers)

    def __getitem__(self, idx):
        return self.finalized_helpers[idx]


#: What a value held for reuse is filed under.  A plain name, unless the read
#: it came from was written under a guard, in which case the guard is part of
#: the name: the same arithmetic under two different guards is two different
#: values, and reusing one for the other would drop a guard.
TritonCSEKey = "str | tuple[str, str]"

#: The dimensions a read's index was split along, innermost last.  A read of a
#: discontiguous value is written as a read of each contiguous run in it, and
#: the runs are the bases a later read of the same value may reuse.
LoadIndexBasis = "tuple[IterationRangesEntry, ...]"
LoadIndexBases = "tuple[LoadIndexBasis | None, ...]"


@dataclasses.dataclass(frozen=True)
class _UnresolvedLoadIndexState:
    """The first live read of a value, recorded without analysing the ranges.

    Whether a read's result can be reused is a question about the ranges, and
    answering it costs more than the read does.  So the first read of a value
    is recorded as it stands, and a later read of the same value is what
    decides whether the first one's answer can be reused.
    """

    index: sympy.Expr
    result: sympy.Expr


@dataclasses.dataclass(frozen=True)
class _ResolvedLoadIndexState:
    """A read's result, together with the split bases a later read may reuse."""

    index: sympy.Expr
    result: sympy.Expr
    bases: LoadIndexBases


LoadIndexState = "_UnresolvedLoadIndexState | _ResolvedLoadIndexState"


class TritonCSE(CSE):
    """Reuse of values, keyed so that a value is not reused across guards.

    The shared machinery files a value under the text it was computed from, so
    the same arithmetic computed twice is computed once.  That is right for
    arithmetic and wrong for a read: a read is written under whatever guard
    covers the positions it covers, so the same text under two guards reads
    two different things and one of them would be wrong.  So the guard is part
    of the name.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_index_states: dict = {}

    def invalidate(self, keep_vars) -> None:
        super().invalidate(keep_vars)
        self._load_index_states.clear()

    def augment_key(self, cache_key: str):
        from ..loops import V

        if mask := getattr(V.kernel, "_load_mask", None):
            return (cache_key, mask.name)
        return cache_key
