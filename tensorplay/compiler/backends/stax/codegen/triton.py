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
from .common import CSEVariable
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
