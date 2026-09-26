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
import functools
import dataclasses
import itertools
import math
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
from .common import CSE, CSEVariable, OpOverrides, PythonPrinter
from ..utils import (
    dtype_to_type,
    IndentedBuffer,
    Number,
    op_requires_libdevice_fp64,
)


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

    def _check_buffer_alignment(self, name: str, var: str, dtype: tp.dtype) -> bool:
        """Whether a buffer's layout offset can't be proven TMA-aligned, so
        host-side TMA must be skipped. The CUtensorMap alignment requirement is
        CUDA-specific, so this only applies on CUDA; an offset that is not
        statically known to be a multiple of TMA_ALIGNMENT is treated as
        misaligned (conservative).
        """
        if not config.triton.use_tensor_descriptor:
            return False
        if V.graph.get_current_device_or_throw().type != "cuda":
            return False
        buf = V.graph.try_get_buffer(name)
        if buf is None and hasattr(V.graph, "scheduler") and V.graph.scheduler:
            real_name = V.graph.scheduler.mutation_real_name.get(name, name)
            if real_name != name:
                buf = V.graph.try_get_buffer(real_name)
        if buf is None:
            return False
        layout_offset = getattr(buf.get_layout(), "offset", 0)
        if layout_offset == 0:
            return False
        offset_bytes = layout_offset * dtype.itemsize
        if V.graph.sizevars.statically_known_multiple_of(offset_bytes, TMA_ALIGNMENT):
            return False
        self._host_tma_non_materializable.add(var)
        self.host_tma_descriptor_args.pop(var, None)
        return True

    def _combine_masks(self, *variables: CSEVariable | None):
        masks = None
        for elem in variables:
            if elem is None:
                continue
            if hasattr(elem, "mask_vars"):
                if masks is None:
                    masks = elem.mask_vars
                else:
                    masks = masks | elem.mask_vars
        return masks

    @staticmethod
    def _enable_pdl_codegen():
        if not config.triton.enable_pdl:
            return False
        if isinstance(V.kernel, TritonTemplateKernel):
            return False
        # PDL uses CUDA-specific intrinsics (gdc_wait/gdc_launch), not available on ROCm
        if tp.version.hip:
            return False
        return (
            V.graph.get_current_device_or_throw().type == "cuda"
            and tp.cuda.get_device_capability()[0] >= 9
        )

    def _filter_pdl(self, code: IndentedBuffer):
        new_lines = []
        has_wait = False
        previous_launch = None
        for l in code._lines:
            if type(l) is str and self.GDC_WAIT in l:
                if has_wait:
                    continue
                else:
                    has_wait = True
            if type(l) is str and self.GDC_LAUNCH in l:
                if previous_launch is not None:
                    new_lines.pop(previous_launch)
                previous_launch = len(new_lines)
            new_lines.append(l)
        code._lines = new_lines

    def _get_grid_type(self) -> type[triton_heuristics.GridExpr]:
        n = sum([int(not tree.is_reduction) for tree in self.range_trees])
        if self.mix_order_reduction:
            if n != 1:
                raise AssertionError(f"expected n == 1, got {n}")
            return triton_heuristics.MixOrderReductionGrid
        elif self.cooperative_reduction:
            if n != 1:
                raise AssertionError(f"expected n == 1, got {n}")
            return triton_heuristics.CooperativeReductionGrid
        elif n == 1:
            return triton_heuristics.Grid1D
        elif n == 2:
            if any(map(self.needs_yz_grid_overflow, self.range_trees)):
                return triton_heuristics.Grid2DWithYZOverflow
            return triton_heuristics.Grid2D
        elif n == 3:
            if self.is_native_matmul:
                return triton_heuristics.BatchMatmulGrid3D
            return triton_heuristics.Grid3D
        raise ValueError(f"Unsupported number of dimensions: {n}")

    def _get_heuristic(self):
        if self.fixed_config:
            return "fixed_config"
        elif self.cooperative_reduction:
            return "cooperative_reduction"
        elif self.persistent_reduction:
            if not self.inside_reduction:
                raise AssertionError("expected inside_reduction")
            return "persistent_reduction"
        elif self.inside_reduction:
            return "reduction"
        return "pointwise"

    @staticmethod
    def _get_persistent_RBLOCK(rnumel):
        rnumel = V.graph.sizevars.simplify(rnumel)
        if isinstance(rnumel, (sympy.Integer, int)):
            val = int(rnumel)
            val = next_power_of_2(val)
        else:
            val = 1
            while not V.graph.sizevars.statically_known_leq(rnumel, val):
                if val > 16 * 1024:
                    raise ValueError(f"Failed to find static RBLOCK for {rnumel}")
                val *= 2

            return val

        return val

    @staticmethod
    def _has_stride1_on_rdim(index) -> bool:
        # These analysis is only needed in deterministic mode so far
        # to filter triton configs. Return false immediately to avoid
        # increasing compilation time when the mode is off.
        if not (
            config.deterministic or config.test_configs.force_filter_reduction_configs
        ):
            return False
        support_vars = index.free_symbols
        reduce_vars = [
            var
            for var in support_vars
            if symbol_is_type(var, TritonSymbols.reduction_types)
        ]

        if len(reduce_vars) == 0:
            return False

        # for expression "x0 + 150528*((x1//(s27*s38))) + 3*(ModularIndexing(x1, 1, s38)) + 672*(ModularIndexing(x1, s38, s27))"
        # stride_vars will results in DivisionByZero error
        try:
            stride_vars = V.graph.sizevars.stride_vars(index, reduce_vars, support_vars)
        except ZeroDivisionError:
            return False

        return any(stride == 1 for stride in stride_vars)

    @staticmethod
    def _is_host_tma_materializable(
        indexing: TensorDescriptorOptions,
        dtype: tp.dtype | None = None,
    ) -> bool:
        """Whether the host launcher can build a TensorDescriptor for this
        access (not a TMA eligibility check -- that's TMACompatibilityChecker).
        """
        if dtype is not None and dtype == tp.bool:
            return False
        for dim in indexing.block_shape:
            if isinstance(dim, (int, sympy.Integer)):
                continue
            if isinstance(dim, sympy.Symbol):
                continue
            return False
        return True

    def _lift_helper(
        self, fn, values: tuple[CSEVariable, ...], dtypes: tuple[tp.dtype, ...]
    ) -> str:
        # Lift IR function for scan operations into a triton function
        # in the global namespace
        helper = IndentedBuffer()
        helper.writeline("@triton.jit")
        cse = CSE()

        args = [
            tuple(
                cse.namedvar(f"arg{i}_{n}", dtype=dtype, shape=value.shape)
                for n, (value, dtype) in enumerate(zip(values, dtypes))
            )
            for i in range(2)
        ]
        signature = ", ".join(str(x) for x in itertools.chain.from_iterable(args))
        helper.writeline(f"def {{name}}({signature}):")

        overrides = TritonOverrides()

        # Build a name that changes depending on fn to workaround a triton bug
        # where the combine_fn to reduce and scan is not hashed, and so different
        # scan ops may collide in the triton cache.
        # This is fixed with the latest triton pin, but not the triton-rocm pin.
        helper_name = "_triton_helper_fn"

        from ..dtype_propagation import DtypePropagationOpsHandler
        from ..shape_propagation import ShapePropagationOpsHandler

        shape_handler = ShapePropagationOpsHandler()
        dtype_handler = DtypePropagationOpsHandler()

        class CSEProxy(DefaultHandler):
            def _default(
                self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> Any:
                nonlocal helper_name
                helper_name += f"_{name}"

                output_dtype = getattr(
                    dtype_handler,
                    name,
                )(*args, **kwargs)

                output_shape = getattr(
                    shape_handler,
                    name,
                )(*args, **kwargs)

                return cse.generate(
                    helper,
                    getattr(overrides, name)(*args, **kwargs),
                    dtype=output_dtype,
                    shape=output_shape,
                )

        with helper.indent(), V.set_ops_handler(CSEProxy()):
            outputs = fn(*args)
            outputs = ", ".join(str(output) for output in outputs)
            helper.writeline(f"return {outputs}")

        return self.helper_functions.add(helper.getvalue(), base_name=helper_name)

    def _load_index_split_basis(
        self, index: sympy.Expr, tree: IterationRangesRoot
    ) -> LoadIndexBasis | None:
        """Find split digits that reconstruct a range tree's flat index.

        Each entry is one digit in a mixed-radix index. A complete basis has at
        least two non-unit digits, starts at divisor 1, and forms a contiguous
        divisor chain whose total extent equals the root's numel.

        Symbolic lengths are supported when size analysis can prove those
        identities without guards; otherwise this conservatively returns None.
        """
        sizevars = V.graph.sizevars
        remaining: list[IterationRangesEntry] = []
        for symbol in index.free_symbols:
            entry = self.range_tree_nodes.get(symbol)
            if (
                entry is not None
                and entry.root is tree
                and not sizevars.statically_known_equals(entry.length, sympy.S.One)
            ):
                remaining.append(entry)
        if len(remaining) <= 1:
            return None

        basis: list[IterationRangesEntry] = []
        divisor = sympy.S.One
        # Divisors may be symbolic, so follow the mixed-radix chain with
        # guarded equality instead of sorting them.
        while remaining:
            for i, entry in enumerate(remaining):
                if sizevars.statically_known_equals(entry.divisor, divisor):
                    basis.append(entry)
                    divisor *= entry.length
                    remaining.pop(i)
                    break
            else:
                return None
        if not sizevars.statically_known_equals(divisor, tree.numel):
            return None
        return tuple(basis)

    def _prescan_host_tma_materializability(self) -> None:
        """Populate _host_tma_non_materializable_buffers with buffers that
        can't be expressed as a single host-side TMA descriptor."""
        if not config.triton.use_tensor_descriptor:
            # Host TMA is off; mark as scanned (no bad buffers, won't change).
            self._host_tma_non_materializable_buffers = OrderedSet()
            return
        self._host_tma_non_materializable_buffers = OrderedSet()

        from .simd_kernel_features import NodeScheduleMarker

        range_tree_symbols = OrderedSet(tree.symbol() for tree in self.range_trees)
        if not range_tree_symbols:
            return

        from torch.utils._sympy.functions import FloorDiv, ModularIndexing

        buffer_read_indices: dict[
            str, list[tuple[sympy.Expr, tuple[sympy.Symbol, ...]]]
        ] = collections.defaultdict(list)
        for node in NodeScheduleMarker.only_nodes(self.features.node_schedule):
            for dep in node.read_writes.reads:
                if not hasattr(dep, "var_names"):
                    if hasattr(dep, "name"):
                        self._host_tma_non_materializable_buffers.add(dep.name)
                    continue
                buffer_read_indices[dep.name].append((dep.index, dep.var_names))

        for buf_name, reads in buffer_read_indices.items():
            if buf_name in self._host_tma_non_materializable_buffers:
                continue

            for index, var_names in reads:
                dep_vars = OrderedSet(var_names)

                if any(symbol_is_type(s, SymT.TMP) for s in index.free_symbols):
                    self._host_tma_non_materializable_buffers.add(buf_name)
                    break

                for expr_node in sympy.preorder_traversal(index):
                    if isinstance(expr_node, (FloorDiv, ModularIndexing)):
                        if expr_node.free_symbols & dep_vars:
                            self._host_tma_non_materializable_buffers.add(buf_name)
                            break
                else:
                    continue
                break

            if buf_name in self._host_tma_non_materializable_buffers:
                continue

            if len(reads) > 1:
                self._host_tma_non_materializable_buffers.add(buf_name)

    def _range_tree_mask_shape(self, mask: str) -> BlockShapeType:
        for tree in self.active_range_trees():
            if tree.owns_mask(mask):
                return tree.mask_shape(self.triton_tensor_ndim())
        return None

    @staticmethod
    def _reshape_expr(
        value: CSEVariable,
        shape: Sequence[sympy.Expr | int | str],
        *,
        value_expr: str | None = None,
    ) -> str:
        if value_expr is None:
            value_expr = str(value)
        old_shape = getattr(value, "shape", None)
        if old_shape is None:
            return f"tl.reshape({value_expr}, {triton_shape_str(shape)})"
        return triton_reshape(value_expr, list(old_shape), list(shape))

    def _rewrite_full_range_with_basis(
        self,
        index: sympy.Expr,
        tree: IterationRangesRoot,
        basis: LoadIndexBasis,
    ) -> sympy.Expr:
        """Replace a tree's flat symbol with its exact split-coordinate sum."""
        split_index = sum(
            (entry.symbol() * entry.divisor for entry in basis), sympy.S.Zero
        )
        basis_symbols = OrderedSet([entry.symbol() for entry in basis])
        replacements: dict[sympy.Symbol, sympy.Expr] = {}
        sizevars = V.graph.sizevars
        for symbol in index.free_symbols - basis_symbols:
            entry = self.range_tree_nodes.get(symbol)
            # Leave alternate splits alone; only the unsplit full range has
            # the direct coordinate identity represented by split_index.
            if entry is None or entry.root is not tree:
                continue
            is_full_range = sizevars.statically_known_equals(
                entry.divisor, sympy.S.One
            ) and sizevars.statically_known_equals(entry.length, tree.numel)
            if is_full_range:
                replacements[symbol] = split_index
        return sympy_subs(index, replacements) if replacements else index

    @cache_on_self
    def _strict_reduction_rblock(self) -> int | None:
        if self.num_reduction_dims != 1:
            return None
        return self.features.strict_reduction_rblock()

    def _supports_load_index_basis_reuse(self) -> bool:
        """Whether this kernel uses a supported shared coordinate scope."""
        return (
            self.__class__ is TritonKernel
            and self.features.is_reduction()
            and not self.cooperative_reduction
            and not any(
                isinstance(tree, DerivedIterationRangesRoot)
                for tree in self.range_trees
            )
        )

    def _trace_kernel_arg(self, name: str, *, store: bool) -> str:
        if name in V.graph.removed_buffers:
            return name
        if store:
            return self.args.output(name)
        return self.args.input(name)

    @staticmethod
    def _trace_symbol_prefix(symbol: sympy.Symbol) -> str | None:
        for symt in (
            SymT.XBLOCK,
            SymT.YBLOCK,
            SymT.ZBLOCK,
            SymT.R0_INDEX,
            SymT.R1_INDEX,
        ):
            if symbol_is_type(symbol, symt):
                return prefix_str[symt]
        return None

    def add_numel_to_call_args(self, name, call_args, arg_types):
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = V.graph.wrapper_code.generate_numel_expr(name, tree)

            if not tree.is_reduction or self.inside_reduction:
                call_args.append(expr)
                arg_types.append(type(expr))

    @functools.cached_property
    def add_persistent_rblock(self) -> bool:
        # Bail on 3d tiling, which has more complicated coalesce patterns
        looped_red = self.features.is_reduction() and not self.persistent_reduction
        tiling_scores = self.tiling_scores
        two_d_red = len(self.tiling) == 2
        if looped_red and two_d_red:
            memory_stats = self.features.memory_stats(self.tiling)
            dim_stats = memory_stats.persistent.memory.dim[0]
            mem_ops_per_thread = dim_stats.count_per_thread

            if (
                tiling_scores is not None
                and "x" in tiling_scores
                and "r0_" in tiling_scores
            ):
                contiguous_red = tiling_scores_suggest_inner_reduction(
                    tiling_scores, self.features.reduction_numel
                )
            else:
                contiguous_red = (
                    self.features.get_reduction_hint(tiling_scores)
                    == ReductionHint.INNER
                )

            looped_mem = memory_stats.looped.memory.bytes
            persistent_mem = memory_stats.persistent.memory.bytes
            # check that we save significant memory by doing persistent
            saved_bytes_ratio = V.graph.sizevars.optimization_hint(looped_mem) / max(
                V.graph.sizevars.optimization_hint(persistent_mem),
                1,
            )

            # TODO - rnumel should be reasonably close to power of 2
            if (
                # significant memory bandwidth savings
                saved_bytes_ratio >= 1.3
                and contiguous_red
                # TODO - need more detailed register analysis
                and V.graph.sizevars.statically_known_leq(
                    self.features.reduction_numel, 32768
                )
                # We will already generate a persistent config in this case
                and V.graph.sizevars.statically_known_gt(
                    self.features.reduction_numel, 2048
                )
                and mem_ops_per_thread <= 10
            ):
                return True
        return False

    @staticmethod
    def apply_feature_required_overrides(
        kernel_features: SIMDKernelFeatures, kernel_kwargs: dict[str, Any]
    ) -> None:
        # ops.sort only works with persistent reduction, and is not bandwidth
        # bound anyway so taking the hit of non-coalesced loads is okay.
        if (
            kernel_features.contains_op("sort")
            or kernel_features.has_strict_multirow_reduction()
        ):
            kernel_kwargs["override_persistent_reduction"] = True
            kernel_kwargs["override_cooperative_reduction"] = False
        # Cannot use persistent reduction with unknown dynamic rnumel.
        if not TritonKernel.has_persistent_RBLOCK(kernel_features.reduction_numel):
            if kernel_kwargs.get("override_persistent_reduction"):
                raise AssertionError(
                    "cannot override persistent reduction with unknown dynamic rnumel"
                )
            kernel_kwargs["override_persistent_reduction"] = False

    @property
    def assert_function(self) -> str:
        return "tl.device_assert"

    def codegen_descriptor_load_line(self, block_descriptor, indexing):
        """Generate the descriptor load line. Override for backend customization."""
        return f"{block_descriptor}.load({V.kernel.index_to_str(indexing.offsets)})"

    def codegen_descriptor_store_line(self, block_ptr, indexing, value):
        """Generate the descriptor store line. Override for backend customization."""
        return f"{block_ptr}.store({V.kernel.index_to_str(indexing.offsets)}, {value})"

    def codegen_iteration_ranges_entry(self, entry: IterationRangesEntry):
        line = f"{entry.name} = {self.kexpr(self.rename_indexing(entry.expr))}"

        # mix order reduction introduces an extra loop across the x
        # dimension
        if entry.root.is_loop or (self.mix_order_reduction and entry.prefix == "x"):
            self.indexing_code.writeline(line)
        else:
            # lift non-reduction stores outside loop
            self.body.writeline(line)

    def codegen_nan_check(self) -> None:
        wrapper = V.graph.wrapper_code
        _, call_args, arg_signatures, _ = self.args.python_argdefs()
        for arg, arg_signature in zip(call_args, arg_signatures):
            if isinstance(arg_signature, TensorArg):
                if V.graph.cpp_wrapper:
                    wrapper.writeline(
                        f'AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan("{arg}", {arg}));'
                    )
                else:
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    def codegen_prologue(self, code: IndentedBuffer):
        """
        Generate the output from prologue. This should be
        extracted from the subgraph, which is why this is
        partitioned from codegen_body.
        """
        if not self.prologue:
            return

        code.splice(self.prologue)
        self.prologue.clear()
        self.prologue_cache.clear()

    def codegen_reduction_numels(self, buffer: IndentedBuffer) -> None:
        """
        Generates code that flattens ND reduction numels, block sizes, etc. into 1D.
        """
        # rnumel = r0_numel * ... * r(n-1)_numel
        reduction_trees = [tree for tree in self.range_trees if tree.is_reduction]
        rnumel = " * ".join(sorted(f"{tree.prefix}numel" for tree in reduction_trees))
        buffer.splice(f"rnumel = {self.kexpr(rnumel)}")

        # RBLOCK = R0_BLOCK * ... * R(N-1)_BLOCK
        rn_blocks = [
            TritonSymbols.get_block_size(tree)
            for tree in self.range_trees
            if tree.is_reduction
        ]
        rblock = sympy_product(rn_blocks)
        buffer.splice(f"RBLOCK: tl.constexpr = {self.kexpr(rblock)}")

    def create_cse_var(self, *args, **kwargs) -> TritonCSEVariable:
        return TritonCSEVariable(*args, **kwargs)

    def device_assert_async(self, cond, msg) -> None:
        self.compute.writeline(f"tl.device_assert({cond}, {repr(msg)})")

    def dtype_to_str(self, dtype: tp.dtype) -> str:
        return triton_type(dtype)

    def emit_reduce(
        self,
        value: CSEVariable,
        reduction_type: str,
        axis: int,
        dtype: tp.dtype,
        shape: Sequence[Any],
    ) -> CSEVariable:
        """Emit a Triton reduction primitive along `axis`, returning a CSE var."""
        reduce_fn = get_triton_reduction_function(reduction_type)
        return self.cse.generate(
            self.compute,
            f"{reduce_fn}({value}, {axis})",
            dtype=dtype,
            shape=shape,
        )

    def get_load_buffer(self, indexing):
        if indexing.has_indirect() or indexing.has_tmpmask():
            # Masked loads must come after the mask is computed
            return self.compute
        elif (
            self.inside_reduction
            and self.range_trees[-1].is_loop
            and not indexing.has_rindex()
        ):
            # can lift a common load outside of reduction loop
            # One exception is when this is an indirect_load.
            return self.body
        else:
            return self.loads

    @cache_on_self
    def get_reduction_prefixes(self) -> list[str]:
        return [
            prefix_str[symt]
            for symt in list(TritonSymbols.reduction_types)[: self.num_reduction_dims]
        ]

    def guard_cooperative_store(self, name, buffer):
        """
        For cooperative reductions only one thread block should write out the result.
        We rotate which thread block does each write for better parallelism
        """
        idx = self.cooperative_reduction_workspace_cache.increment_store_count()
        buffer.writeline(DeferredLine(name, f"if rsplit_id == ({idx} % RSPLIT):"))
        return buffer.indent()

    @staticmethod
    def has_persistent_RBLOCK(rnumel):
        try:
            TritonKernel._get_persistent_RBLOCK(rnumel)
            return True
        except ValueError:
            return False

    @property
    def has_store_with_contiguous_rdim(self) -> bool:
        return not all(
            is_buffer_removed(name) for name in self.stores_with_contiguous_rdim
        )

    def imports_for_benchmark_kernel(self):
        # Dedent BEFORE substituting get_raw_stream: a multi-line override would
        # otherwise collapse dedent's common prefix and misindent the imports.
        return textwrap.dedent(
            """
            from ..runtime.example_values import rand_strided
            {}
            import torch
            """
        ).format(V.graph.device_ops.import_get_raw_stream_as("get_raw_stream"))

    def iteration_ranges_ranges_code(self, entry: IterationRangesRoot) -> str:
        if entry.tensor_dim is None:
            raise AssertionError("entry.tensor_dim must not be None")
        size = self.indexing_size_str(entry.tensor_dim)
        # For batch matmul, we always set the ZBLOCK=1.
        # In this case, we found not broadcasting tl.arange(0, ZBLOCK) is faster.
        if (
            self.is_native_matmul
            and entry.tensor_dim == 0
            and self.triton_tensor_ndim() == 4
        ):
            size = ""
        index_dtype = self.index_dtype
        suffix = f".to({index_dtype})" if index_dtype != "tl.int32" else ""
        if (
            self.cooperative_reduction
            and self.persistent_reduction
            and entry.is_reduction
        ):
            suffix = f"{suffix} + rsplit_start"
        return f"tl.arange(0, {self.kexpr(entry.block_size())}){size}{suffix}"

    def iteration_ranges_scalar_code(
        self, entry: IterationRangesRoot, value: Any
    ) -> str:
        index_dtype = self.index_dtype
        ndim = self.triton_tensor_ndim()
        size = [1] * ndim
        return f"tl.full({size}, {value}, {index_dtype})"

    def max_block(self, prefix: str) -> int:
        if self.fixed_config:
            return self.fixed_config[f"{prefix.upper()}BLOCK"]
        return TRITON_MAX_BLOCK[prefix.upper()]

    def max_rsplit(self):
        if self.fixed_config:
            return self.fixed_config["RSPLIT"]
        return TRITON_MAX_RSPLIT

    def need_numel_args(self):
        """
        Indicate whether we need provide numel as arguments for the generated
        kernel calls in the benchmark.

        Should be true for pointwise/reduction kernels but false for triton
        matmul kernels.
        """
        return True

    def needs_yz_grid_overflow(self, entry: IterationRangesRoot) -> bool:
        # Combo kernels use flattened dispatch where y_pid_offset is computed
        # from the flattened pid, so YZ overflow is not needed
        if self.is_combo_kernel and self.per_subkernel_blocks:
            return False
        return (
            entry.grid_dim == 1
            and not entry.has_zdim
            and not self.cooperative_reduction
            and not V.graph.sizevars.statically_known_leq(entry.numel, get_max_y_grid())
        )

    def reduction_collapse_dims(
        self, buffer, value: CSEVariable, dtype: tp.dtype
    ) -> CSEVariable:
        """
        Reshape to RBLOCK, collapsing all reduction dims.
        """
        # This is not needed for 1D reductions.
        if self.num_reduction_dims == 1:
            return value

        target_ndim = self.triton_tensor_ndim() - self.num_reduction_dims
        initial_shape = self.dense_size_list()
        target_shape = initial_shape[:target_ndim] + ["RBLOCK"]
        return self.cse.generate(
            buffer,
            triton_reshape(str(value), initial_shape, target_shape),
            dtype=dtype,
            shape=tuple(target_shape),
        )

    def reduction_resize(self, value) -> str:
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})"

        nreduce = self.num_reduction_dims
        sizes = [":"] * (ndims - nreduce) + ["None"] * nreduce
        return f"{value}[{', '.join(sizes)}]"

    def reduction_resize_and_shape(self, value, shape) -> tuple[str, BlockShapeType]:
        ndims = self.triton_tensor_ndim()
        if ndims == 1:
            return f"triton_helpers.promote_to_tensor({value})", shape

        nreduce = self.num_reduction_dims
        sizes = [":"] * (ndims - nreduce) + ["None"] * nreduce
        new_shape = (
            (*shape[: (ndims - nreduce)], *[1] * nreduce) if shape is not None else None
        )
        return f"{value}[{', '.join(sizes)}]", new_shape

    @property
    def uses_device_tma(self) -> bool:
        return self._emitted_device_tma

    @property
    def uses_tma(self) -> bool:
        return bool(self.host_tma_descriptor_args or self._emitted_device_tma)

    def want_no_x_dim(self):
        return (
            self.persistent_reduction
            and len(self.numels) == self.num_reduction_dims + 1
            and self.fixed_config
            and self.fixed_config["XBLOCK"] == 1
        )
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


class TritonPrinter(PythonPrinter):  # noqa: docstring_linter
    def _print_TruncToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        return (
            # pyrefly: ignore [missing-attribute]
            f"libdevice.trunc({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_TruncToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        value = self._print(expr.args[0])
        # Adding +0.0 preserves large floating results while canonicalizing
        # libdevice.trunc(-0.0) back to Python's +0.0 materialization behavior.
        # pyrefly: ignore [missing-attribute]
        return f"(libdevice.trunc({value}) + tl.zeros_like({value}))"

    @staticmethod
    def _get_scalar_float_type() -> str:
        """Return the Triton float type for scalar shape math.

        Uses tl.float64 by default but falls back to tl.float32 on devices
        that lack fp64 support (e.g. Intel Arc consumer GPUs).
        """
        if not device_supports_fp64(V.graph.current_device):
            return "tl.float32"
        return "tl.float64"

    def _print_Float(self, expr: sympy.Expr) -> str:
        if expr.is_integer:
            # sympy considers 0.0 to be integer, but triton doesn't.
            # this workaround prints the float as an integer
            # xref: https://github.com/sympy/sympy/issues/26620
            ret = str(int(expr))
        elif config.is_fbcode() and tp.version.hip:
            ret = f"{expr}"
        else:
            float_type = self._get_scalar_float_type()
            ret = f"tl.full([], {expr}, {float_type})"
        return ret

    def _print_ToFloat(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [bad-argument-type]
        s = self.parenthesize(expr.args[0], PRECEDENCE["Atom"] - 0.5)
        float_type = self._get_scalar_float_type()
        return f"{s}.to({float_type})"

    def _print_PythonMod(self, expr: sympy.Expr) -> str:
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " % ", PRECEDENCE["Atom"] - 0.5)
        # pyrefly: ignore [missing-attribute]
        quot_s = self._print(quot)
        # pyrefly: ignore [missing-attribute]
        div_s = self._print(div)
        return f"triton_helpers.remainder_integer({quot_s}, {div_s})"

    def _print_FloorDiv(self, expr: sympy.Expr) -> str:
        if not expr.is_integer:
            raise AssertionError("expr must be integer")
        quot, div = expr.args
        if quot.is_nonnegative and div.is_nonnegative:
            return self.stringify(expr.args, " // ", PRECEDENCE["Atom"] - 0.5)
        # pyrefly: ignore [missing-attribute]
        quot_s = self._print(quot)
        # pyrefly: ignore [missing-attribute]
        div_s = self._print(div)
        return f"triton_helpers.div_floor_integer({quot_s},  {div_s})"

    # TODO: This is wrong, when lhs, rhs > 2**53, Python does a higher
    # precision algorithm, which we would need to replicate here
    def _print_IntTrueDiv(self, expr: sympy.Expr) -> str:
        return self.stringify(expr.args, " / ", PRECEDENCE["Atom"] - 0.5)

    # NB: sympy.floor/ceiling produce integers, so we have to do the
    # conversion to index dtype
    def _print_floor(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        return (
            # pyrefly: ignore [missing-attribute]
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_FloorToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        return (
            # pyrefly: ignore [missing-attribute]
            f"libdevice.floor({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_ceiling(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _print_CeilToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.ceil({self._print(expr.args[0])}).to({V.kernel.index_dtype})"

    def _helper_sqrt(self, expr: sympy.Expr) -> str:
        # pyrefly: ignore [missing-attribute]
        return f"tl.sqrt_rn(({self._print(expr)}).to(tl.float32))"

    def _print_FloatPow(self, expr: sympy.Expr) -> str:
        # pyrefly: ignore [missing-attribute]
        base = self._print(expr.args[0])
        # pyrefly: ignore [missing-attribute]
        exp = self._print(expr.args[1])
        # libdevice.pow requires both arguments to have the same type.
        # Cast to float64 for precision. Falls back to float32 on devices
        # without fp64 support. This is scalar shape math, not tensor ops.
        float_type = self._get_scalar_float_type()
        return f"libdevice.pow(({base}).to({float_type}), ({exp}).to({float_type}))"

    def _print_PowByNatural(self, expr: sympy.Expr) -> str:
        float_type = self._get_scalar_float_type()
        if expr.args[0].is_Integer:
            base = f"tl.full([], {float(expr.args[0])}, {float_type})"
        else:
            # pyrefly: ignore [missing-attribute]
            base = f"({self._print(expr.args[0])}).to({float_type})"
        exp_val = expr.args[1]
        if exp_val.is_Integer:
            exp = f"tl.full([], {float(exp_val)}, {float_type})"
        else:
            # pyrefly: ignore [missing-attribute]
            exp = f"({self._print(exp_val)}).to({float_type})"
        # libdevice.pow requires both arguments to have the same type.
        # Cast to float64 for precision. Falls back to float32 on devices
        # without fp64 support. This is scalar shape math, not tensor ops.
        return f"libdevice.pow({base}, {exp})"

    def _print_Where(self, expr: sympy.Expr) -> str:
        c = self.doprint(expr.args[0])
        p = self.doprint(expr.args[1])
        q = self.doprint(expr.args[2])
        return f"tl.where({c}, {p}, {q})"

    def _print_min_max_helper(self, expr: sympy.Expr, cmp: str) -> str:
        """
        Helper for max/min code generation.
        cmp: > or <
        """
        if len(expr.args) == 1:
            # pyrefly: ignore [missing-attribute]
            return self._print(expr.args[0])

        mid = len(expr.args) // 2
        cls = type(expr)
        # pyrefly: ignore [missing-attribute]
        a = self._print(cls(*expr.args[:mid]))
        # pyrefly: ignore [missing-attribute]
        b = self._print(cls(*expr.args[mid:]))

        # Use a macro so we can propagate constexprs.
        # https://github.com/triton-lang/triton/issues/3815
        a, b = tuple(f"({x})" for x in (a, b))
        if cmp not in (">", "<"):
            raise AssertionError(f"Unexpected comparator: '{cmp}'")
        return f"({a} * ({a} {cmp}= {b}) + {b} * ({b} {cmp} {a}))"

    def _print_Min(self, expr: sympy.Expr) -> str:
        return self._print_min_max_helper(expr, "<")

    def _print_Max(self, expr: sympy.Expr) -> str:
        return self._print_min_max_helper(expr, ">")

    def _print_Abs(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"tl_math.abs({self._print(expr.args[0])})"

    def _print_OpaqueUnaryFn_cos(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.cos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_cosh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.cosh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_acos(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.acos(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sin(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.sin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_sinh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.sinh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_asin(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.asin(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tan(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.tan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_tanh(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.tanh(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_atan(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.atan(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_OpaqueUnaryFn_log2(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        # pyrefly: ignore [missing-attribute]
        return f"libdevice.log2(({self._print(expr.args[0])}).to(tl.float32))"

    def _print_RoundToInt(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 1:
            raise AssertionError(f"expected 1 arg, got {len(expr.args)}")
        return (
            # pyrefly: ignore [missing-attribute]
            f"libdevice.llrint({self._print(expr.args[0])}).to({V.kernel.index_dtype})"
        )

    def _print_RoundDecimal(self, expr: sympy.Expr) -> str:
        if len(expr.args) != 2:
            raise AssertionError(f"expected 2 args, got {len(expr.args)}")
        number, ndigits = expr.args
        if number.is_integer:
            # ndigits < 0 should have been filtered by the sympy function
            if ndigits >= 0:
                raise AssertionError(f"expected ndigits < 0, got {ndigits}")
            raise ValueError(
                f"For integer inputs, only non-negative ndigits are currently supported, but got {ndigits}."
            )

        number_str = self.parenthesize(number, PRECEDENCE["Mul"])
        return f"libdevice.nearbyint(1e{ndigits} * {number_str}) * 1e{-ndigits}"


texpr = TritonPrinter().doprint


def maybe_upcast_float32(convert_output: bool = True) -> Callable[[_T], _T]:
    """
    Codegen helper to upcast arguments to float32, depending on the config and dtype.
    This decorates tl.math/libdevice codegen functions.
    """

    def maybe_upcast_arg(var) -> str:
        upcast_string = ".to(tl.float32)" if needs_upcast_to_float32(var) else ""
        return f"{var}{upcast_string}"

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Record that this function only supports float32 and float64.
        OpDtypeSupport.register_upcast(func, convert_output)

        def wrapped(*args, **kwargs) -> str:
            # Optionally upcast args to float32.
            upcast_args = [maybe_upcast_arg(arg) for arg in args]
            upcast_kwargs = {key: maybe_upcast_arg(val) for key, val in kwargs.items()}

            # Call the decorated function, optionally downcasting the result.
            result = func(*upcast_args, **upcast_kwargs)
            any_needs_upcast = convert_output and any(
                needs_upcast_to_float32(var)
                for var in itertools.chain(args, kwargs.values())
            )
            result_dtype = (
                None
                if not any_needs_upcast
                else getattr(get_dtype_handler(), func.__name__)(*args, **kwargs)
            )
            needs_downcast = result_dtype not in (tp.float32, None)
            downcast_string = (
                f".to({triton_type(result_dtype)})"
                if needs_downcast and result_dtype is not None
                else ""
            )
            return f"{result}{downcast_string}"

        return wrapped

    return decorator  # type: ignore[return-value]


class OpDtypeSupport:
    """Which operations compute in a wider type than they are stored in.

    A transcendental is computed in float32 whatever it is stored in, because
    a float16 exponential computed in float16 is not the exponential. So an
    operation that has to be widened says so here, and the emitter widens the
    argument and narrows the result rather than writing the operation at the
    stored width and getting a different answer.
    """

    upcast_ops: dict = {}

    @staticmethod
    def register_upcast(func, is_upcast: bool) -> None:
        """Record that an operation computes in a wider type than it stores."""

        OpDtypeSupport.upcast_ops[func] = is_upcast

    @staticmethod
    def is_upcast(op) -> bool:
        """Whether an operation computes in a wider type than it stores."""

        return OpDtypeSupport.upcast_ops.get(op, False)


class TritonOverrides(OpOverrides):
    """Map element-wise ops to Triton e.g., ops.to_dtype(x,...) -> x.to(...)"""

    _LOG_2_E = math.log2(math.e)

    @staticmethod
    def to_dtype(
        x,
        dtype: tp.dtype,
        src_dtype: tp.dtype | None = None,
        use_compute_types=True,
    ):
        fp8_dtypes = (
            tp.float8_e4m3fn,
            tp.float8_e5m2,
        )

        def _get_min_elements_per_thread(
            src_dtype: tp.dtype, dst_dtype: tp.dtype
        ) -> int:
            if src_dtype == dst_dtype:
                # No data type conversion is needed. No requirements on min_elem_per_thread.
                return 0

            # fp8 data type conversions has min_elem_per_thread requirements.
            # Refer to Triton implementations here:
            # https://github.com/triton-lang/triton/blob/10f59d8ce04052521c1bc0cb3a3f8b98918fc7e3/lib/Conversion/TritonGPUToLLVM/ElementwiseOpToLLVM.cpp#L10.
            # Triton doesn't support type conversions between fp8_e4m3 and fp8_e5m2.
            if (
                src_dtype in fp8_dtypes
                and dst_dtype in fp8_dtypes
                and src_dtype != dst_dtype
            ):
                raise AssertionError(
                    "Conversions between float8_e5m2 and float8_e4m3fn is not supported!"
                )
            if src_dtype == tp.float8_e5m2 or dst_dtype == tp.float8_e5m2:
                return 4
            if src_dtype == tp.float8_e4m3fn or dst_dtype == tp.float8_e4m3fn:
                return 2
            # No requirements on min_elem_per_thread.
            return 0

        if src_dtype is not None:
            # Both dtype and src_dtype are set. This is used by torch to(dtype=dtype).
            # It takes the maximum min_elem_per_thread if there are multiple fp8 conversions
            # in the same kernel.
            V.kernel.min_elem_per_thread = max(
                _get_min_elements_per_thread(src_dtype, dtype),
                V.kernel.min_elem_per_thread,
            )

        if src_dtype is not None and use_uint8_triton_storage_for_cuda_float8_e4m3fn(
            src_dtype
        ):
            x = f"triton_helpers.fp8e4m3fn_to_float32({x})"
            src_dtype = tp.float32

        if dtype == tp.bool:
            return f"({x} != 0)"
        elif dtype == tp.uint8 and (
            src_dtype is not None and src_dtype.is_floating_point or src_dtype is None
        ):
            # to work around llvm uint conversion semantics that produces 0's for negative
            # values when converting from floating types.
            # optimization - if source type is known and it's not a floating type, then
            # do not apply conversion to the intermediate type.
            return f"{x}.to(tl.int16).to(tl.uint8)"

        if use_compute_types:
            out_dtype = triton_compute_type(dtype)
        else:
            out_dtype = triton_store_type(dtype)

        if (
            src_dtype is not None
            and dtype in fp8_dtypes
            and (src_dtype == tp.bool or is_integer_dtype(src_dtype))
        ):
            return f"{x}.to(tl.float32).to({out_dtype})"

        return f"{x}.to({out_dtype})"

    @staticmethod
    def to_dtype_bitcast(x, dtype: tp.dtype, src_dtype: tp.dtype):
        if src_dtype.itemsize != dtype.itemsize:
            raise AssertionError(
                f"itemsize mismatch: {src_dtype.itemsize} != {dtype.itemsize}"
            )
        # We may promote float16 or bfloat16 to float32 and cause the
        # bitwidth of dtype to be different from the input tensor (i.e. float32).
        # In such as case, we will have to convert the input tensor to
        # its src_type, perform bitcast, and then convert the bit-casted
        # tensor back to float to ensure we use values with the right precision.
        if x.dtype != src_dtype:
            x = f"{x}.to({triton_type(src_dtype)})"

        out = f"{x}.to({triton_type(dtype)}, bitcast=True)"
        if upcast_compute_type(dtype) != dtype:
            out = f"{out}.to({triton_type(upcast_compute_type(dtype))})"

        return out

    @staticmethod
    def _shaped_constant(value, dtype, shape):
        type_ = dtype_to_type(dtype)
        triton_val = constant_repr(type_(value))
        triton_type = triton_compute_type(dtype)

        # Triton's scalar_constant() treats -0.0 as 0 (since -0.0 == 0 in Python),
        # Work around by encoding -0.0 as its IEEE 754 hex in uint and bitcasting.
        if value == 0 and math.copysign(1.0, value) < 0:
            if triton_type == "tl.float32":
                return f"tl.full({shape}, 0x80000000, tl.uint32).to({triton_type}, bitcast=True)"
            elif triton_type == "tl.float64":
                return f"tl.full({shape}, 0x8000000000000000, tl.uint64).to({triton_type}, bitcast=True)"

        # NOTE: We use tl.full here to get the expected type.
        # Otherwise, subnormal float32 values are treated as fp64
        # causing fp32 * fp64 promotion and different numerical results.
        if value < 0 and not dtype.is_signed:
            triton_signed_type = f"tl.{triton_type[4:]}"
            return f"tl.full({shape}, {triton_val}, {triton_signed_type}).to({triton_type})"
        else:
            return f"tl.full({shape}, {triton_val}, {triton_type})"

    @classmethod
    def constant(cls, value, dtype):
        return cls._shaped_constant(value, dtype, shape=[])

    @staticmethod
    def sub(x, y):
        if (
            isinstance(x, CSEVariable)
            and x == y
            and x.dtype is not None
            and x.dtype.is_floating_point
        ):
            # Avoid giving LLVM a tmp - tmp pattern that it can reassociate
            # through tmp's producer. A plain 0.0 is only valid for finite
            # inputs; nan/inf inputs should still produce nan like x - x.
            non_finite = f"({TritonOverrides.isnan(x)} | {TritonOverrides.isinf(x)})"
            return f"tl.where({non_finite}, {x} * 0.0, 0.0)"
        return f"{x} - {y}"

    @classmethod
    def _cast_libdevice_arg(cls, arg, dtype: tp.dtype) -> str:
        if isinstance(arg, Number):
            return cls.constant(arg, dtype)
        if triton_arg_dtype(arg) == dtype:
            return f"{arg}"
        return f"({arg}).to({triton_type(dtype)})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def abs(x):
        return f"tl_math.abs({x})"

    # TODO - register these ops as having divergent dtype
    # output if doing graph pass to remove consecutive casts

    @staticmethod
    def truediv(x, y):
        x_dtype = getattr(x, "dtype", None)
        y_dtype = getattr(y, "dtype", None)

        if (
            x_dtype == tp.float32
            and y_dtype == tp.float32
            and config.eager_numerics.division_rounding
        ):
            # x / y in Triton is lowered to div.full which is approx
            # we want div_rn to adhere with eager
            out = f"triton.language.div_rn({x}, {y})"
        else:
            out = f"({x} / {y})"

        if low_precision_fp_var(x) or low_precision_fp_var(y):
            out_dtype = get_dtype_handler().truediv(x, y)
            if out_dtype in (tp.float16, tp.float32):
                out = f"{out}.to({triton_type(out_dtype)})"

        return out

    @staticmethod
    def div_rn(x, y):
        """
        Division with round-to-nearest rounding mode.
        Always uses triton.language.div_rn for float32 inputs to match eager CUDA behavior.
        """
        x_dtype = getattr(x, "dtype", None)
        y_dtype = getattr(y, "dtype", None)

        if x_dtype == tp.float32 and y_dtype == tp.float32:
            out = f"triton.language.div_rn({x}, {y})"
        else:
            # Fall back to regular division for non-float32 types
            out = f"({x} / {y})"

        if low_precision_fp_var(x) or low_precision_fp_var(y):
            out_dtype = get_dtype_handler().truediv(x, y)
            if out_dtype in (tp.float16, tp.float32):
                out = f"{out}.to({triton_type(out_dtype)})"

        return out

    @staticmethod
    def mod(x, y):
        out = f"({x} % {y})"
        if low_precision_fp_var(x) or low_precision_fp_var(y):
            out_dtype = get_dtype_handler().mod(x, y)
            if out_dtype in (tp.float16, tp.float32):
                out = f"{out}.to({triton_type(out_dtype)})"
        return out

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def exp(x):
        """
        When use_fast_math, use the ftz (flushing to zero) variant
        of exponent computation.

        Check https://github.com/triton-lang/triton/issues/5735 for
        more details.
        """
        if config.use_fast_math:
            return f"tl_math.exp({x})"
        else:
            return f"libdevice.exp({x})"

    @staticmethod
    @maybe_upcast_float32()
    def exp2(x):
        return f"libdevice.exp2({x})"

    @staticmethod
    @maybe_upcast_float32()
    def expm1(x):
        return f"libdevice.expm1({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def sqrt(x):
        return f"tl.sqrt_rn({x})"

    @staticmethod
    def relu(x):
        bug = config.triton.inject_relu_bug_TESTING_ONLY
        if bug == "compile_error":
            return "compile error!"
        elif bug == "runtime_error":
            # NB: this only triggers runtime error as long as input
            # is not all zero
            return f'triton_helpers.device_assert_then({x} == 0, "injected assert fail", {x})'
        elif bug == "accuracy":
            return f"{x} + 1"
        elif bug is None:
            return ops.maximum(ops.constant(0, tp.int32), x)
        else:
            raise AssertionError(
                f"unrecognized config triton.inject_relu_bug_TESTING_ONLY = {bug!r}"
            )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def minimum(a, b):
        return f"tl.minimum({a}, {b}, tl.PropagateNan.ALL)"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def maximum(a, b):
        return f"tl.maximum({a}, {b}, tl.PropagateNan.ALL)"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def fmaximum(a, b):
        return f"tl.maximum({a}, {b})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def where(a, b, c):
        return f"tl.where({a}, {b}, {c})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def dot(a, b):
        """
        Triton code generation for lowering ops.dot to tl.dot.

        The logic is as follows:

        1. Downcasting for performance
           If the data was previously upcasted to fp32, we downcast back to the
           original dtype (e.g., fp16 or bf16) for better performance. While
           surrounding operations may run in fp32, matmul itself is executed at the
           original precision to optimize throughput.

        2. Handling non-constant reduction masks
           If the reduction mask is not constant and there was any operation between
           tl.load and tl.dot, we zero out regions outside the mask using
           tl.where(r0_mask, val, 0).
           This ensures that values outside the mask do not contribute to the dot
           product, preventing incorrect results.

        3. Shape alignment for tl.dot
           We massage shapes to match the tl.dot requirement of (Y, R) x (R, X).
           Current codegen eagerly broadcasts tl.arange to create unique axes. We
           reshape, transpose, or broadcast to align with the (Y, R) x (R, X) shape.
           We avoid using 3D dot ((Z, Y, R) x (Z, R, X)) because 3D tl.dot has
           poor performance. During batched matmul (bmm), we keep ZBLOCK=1 and call
           the 2D dot kernel instead.
        """
        if not V.kernel.is_native_matmul:
            raise AssertionError("expected native matmul kernel")
        orig_a, orig_b = a, b

        def is_where_needed(var):
            # Skip if the variable doesn't have a reduction mask
            if not any(map(prefix_is_reduction, var.mask_vars)):
                return False

            reduction_range = V.kernel.range_trees[-1]
            if not reduction_range.is_reduction:
                raise AssertionError("reduction_range must be a reduction")

            # Skip if reduction mask was already constant
            if V.kernel._has_constant_mask(reduction_range):
                return False

            # Skip if the variable is already zeroed outside the mask
            # (e.g., from tl.load(..., other=0.0))
            # TODO : track the value of outside of mask region with cse
            for k, v in V.kernel.cse._cache.items():
                if v == var and "tl.load" in k and "other=0.0" in k:
                    return False

            return True

        def where_cond(var):
            default = ir.Reduction.default_value("dot", var.dtype)
            reduction_mask = [
                tree.mask_name() for tree in V.kernel.range_trees if tree.is_reduction
            ]

            if len(reduction_mask) != 1:
                raise AssertionError("don't tile reduction when native matmul")

            where_var = TritonKernelOverrides.where(reduction_mask[0], var, default)
            return V.kernel.cse.generate(
                V.kernel.compute, where_var, dtype=var.dtype, shape=var.shape
            )

        # When computing expressions like ((A+1) @ (B+2)),
        # native codegen will do
        #
        # a = tl.load(..., r0_mask, other=0.0)
        # b = tl.load(..., r0_mask, other=0.0)
        # tmp0 = a+1
        # tmp1 = b+2
        # tmp2 = tl.dot(tmp0, tmp1)
        #
        # This produces incorrect results because outside of r0_mask is not zero.
        # So before calling tl.dot, apply tl.where to zero out values properly.
        # TODO: Optimize - We don't need both operands to be zeroed except NaN * 0
        if is_where_needed(orig_a):
            a = where_cond(a)
        if is_where_needed(orig_b):
            b = where_cond(b)

        def reshape_transpose_broadcast_for_dot(
            value,
            initial_shape: Sequence[sympy.Expr],
            final_shape: Sequence[sympy.Expr],
        ) -> str:
            """
            Generate a reshape, transpose, and broadcast for the tl.dot.
            tl.dot requires specific shape requirement : (Y,R) x (R,X)
            but the current triton codegen eagerly broadcast the tl.arange so
            it needs to be reshaped to meet the requirement.

            This is done by three steps.
            1. remove the empty dimension (dim with size 1) and make it 2d with tl.reshape
            2. permute the dimension if needed (e.g., (X,R) -> (R,X)) with tl.trans
            3. broadcast if needed with broadcast_to.
                - This shows up when matmul operand is broadcasted with torch.expand/repeat.
                - e.g., tp.rand((16,)).expand(16,16) @ B

            e.g., (Y,1,R), (Y,R) -> tl.reshape(var, (Y,R))
            e.g., (1,X,R), (R,X) -> tl.trans(tl.reshape(var, (X,R)))
            e.g., (1,X,1), (R,X) -> tl.broadcast_to(tl.trans(tl.reshape(var, (X,1))), (R,X))

            TODO : eventually we want to remove this function when lazy broadcasting arrives
            """

            # Triton 3d dot is slower than 2d dot, so we want to keep block shape in 2d
            # by fixing ZBLOCK=1 in the autotune config
            if ZBLOCK in initial_shape:
                initial_shape = ["1" if dim == ZBLOCK else dim for dim in initial_shape]

            if final_shape == [YBLOCK, RBLOCK]:
                if XBLOCK in initial_shape:
                    raise AssertionError("left tl.dot operand cannot depend on x")

                shape_2d = ["1", "1"]
                if YBLOCK in initial_shape:
                    shape_2d[0] = YBLOCK
                if RBLOCK in initial_shape:
                    shape_2d[1] = RBLOCK

                # reshape it into 2d
                value = triton_reshape(value, initial_shape, shape_2d)

                # broadcast if needed
                broadcast_needed = shape_2d != [YBLOCK, RBLOCK]
                if broadcast_needed:
                    value = f"tl.broadcast_to({value}, ({YBLOCK}, {RBLOCK}))"

            elif final_shape == [RBLOCK, XBLOCK]:
                if YBLOCK in initial_shape:
                    raise AssertionError("right tl.dot operand cannot depend on y")

                shape_2d = ["1", "1"]
                if XBLOCK in initial_shape:
                    shape_2d[0] = XBLOCK
                if RBLOCK in initial_shape:
                    shape_2d[1] = RBLOCK

                # reshape it into 2d (X,R)
                value = triton_reshape(value, initial_shape, shape_2d)

                # transpose to (R,X)
                value = f"tl.trans({value})"

                # broadcast if needed
                broadcast_needed = shape_2d != [XBLOCK, RBLOCK]
                if broadcast_needed:
                    value = f"tl.broadcast_to({value}, ({RBLOCK}, {XBLOCK}))"
            else:
                raise NotImplementedError

            return value

        if len(V.kernel.dense_size_list()) < 3:
            raise AssertionError("tl.dot can only do mm and bmm")

        XBLOCK = str(TritonSymbols.block_sizes[SymT.XBLOCK])
        YBLOCK = str(TritonSymbols.block_sizes[SymT.YBLOCK])
        ZBLOCK = str(TritonSymbols.block_sizes[SymT.ZBLOCK])
        RBLOCK = str(TritonSymbols.block_sizes[SymT.R0_INDEX])

        a = V.kernel.cse.generate(
            V.kernel.compute,
            reshape_transpose_broadcast_for_dot(a, list(a.shape), [YBLOCK, RBLOCK]),
            dtype=a.dtype,
            shape=(YBLOCK, RBLOCK),
        )

        b = V.kernel.cse.generate(
            V.kernel.compute,
            reshape_transpose_broadcast_for_dot(b, list(b.shape), [RBLOCK, XBLOCK]),
            dtype=b.dtype,
            shape=(RBLOCK, XBLOCK),
        )

        if torch.backends.cuda.matmul.fp32_precision == "tf32":
            input_precision = "tf32"
        else:
            input_precision = "ieee"

        return f'tl.dot({a}, {b}, input_precision="{input_precision}")'

    @staticmethod
    def inline_asm_elementwise(
        *inputs,
        asm,
        constraints=None,
        dtype=tp.float32,
        is_pure=True,
        pack=1,
        input_dtypes=None,
    ):
        # Use the actual dtype, not the compute type — the asm operates on
        # specific register types and Triton needs to know the real output type.
        asm_triton_type = triton_type(dtype)
        if constraints is None:
            constraints = ", ".join(["=r"] + ["r" for _ in inputs])

        # Inductor computes bf16/fp16 in fp32. For "h" (16-bit register)
        # constraints, cast back to the original dtype so the asm sees the
        # right register type.
        constraint_parts = [p.strip() for p in constraints.split(",")]
        input_constraints = [p for p in constraint_parts if not p.startswith("=")]
        cast_inputs = []
        for i, (inp, c) in enumerate(zip(inputs, input_constraints[: len(inputs)])):
            if (
                c == "h"
                and input_dtypes is not None
                and isinstance(inp, CSEVariable)
                and inp.dtype != input_dtypes[i]
            ):
                cast_inputs.append(f"{inp}.to({triton_type(input_dtypes[i])})")
            else:
                cast_inputs.append(str(inp))

        # Asm strings may contain real newlines (AMDGCN instructions are
        # newline-separated; multi-line PTX blocks with .reg declarations
        # too).  The generated code is nested inside two Python string layers:
        #   Layer 1 : the cached wrapper .py file
        #   Layer 2 : the Triton kernel source (a triple-quoted string
        #             inside that wrapper, exec'd / JIT-compiled)
        # repr() escapes \n -> \\n, then we double the backslashes so
        # they survive both layers: \\\\n -> (L1 parse) \\n -> (L2 parse) \n.
        asm_literal = repr(asm).replace("\\", "\\\\")
        constraints_literal = repr(constraints).replace("\\", "\\\\")

        def asm_call(args):
            return (
                f"tl.inline_asm_elementwise({asm_literal}, {constraints_literal}, "
                f"[{args}], dtype={asm_triton_type}, is_pure={is_pure}, pack={pack})"
            )

        if pack <= 1:
            return asm_call(", ".join(cast_inputs))

        first_input = inputs[0]
        compute = V.kernel.compute
        cse = V.kernel.cse
        result = cse.newvar(dtype=dtype, shape=first_input.shape)
        packed_args = ", ".join(
            f"triton_helpers.inline_asm_pack({inp}, {pack})" for inp in cast_inputs
        )
        compute.writeline(f"{result} = {asm_call(packed_args)}")
        compute.writeline(
            f"{result} = triton_helpers.inline_asm_unpack({result}, {first_input}, {pack})"
        )
        return result

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def cos(x):
        return f"tl_math.cos({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def sin(x):
        return f"tl_math.sin({x})"

    @classmethod
    def index_expr(cls, expr, dtype):
        raise NotImplementedError("ops.index_expr not implemented outside a kernel")

    @classmethod
    def value_expr(cls, expr, dtype):
        raise NotImplementedError("ops.value_expr not implemented outside a kernel")

    @staticmethod
    def masked(mask, body, other):
        raise NotImplementedError("ops.masked not implemented outside a kernel")

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def lgamma(x):
        return f"libdevice.lgamma({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def erf(x):
        return f"libdevice.erf({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def cosh(x):
        return f"libdevice.cosh({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def sinh(x):
        return f"libdevice.sinh({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def acos(x):
        return f"libdevice.acos({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def acosh(x):
        return f"libdevice.acosh({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def asin(x):
        return f"libdevice.asin({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def asinh(x):
        return f"libdevice.asinh({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def atan2(x, y):
        return f"libdevice.atan2({x}, {y})"

    @staticmethod
    def _f32_const_from_bits(bits: int) -> str:
        return f"tl.full([], 0x{bits:08X}, tl.uint32).to(tl.float32, bitcast=True)"

    @staticmethod
    def _inline_asm_f32(asm: str, args: Sequence[Any], shape: BlockShapeType) -> Any:
        constraints = ", ".join(["=f"] + ["f"] * len(args))
        args_str = ", ".join(map(str, args))
        return V.kernel.cse.generate(
            V.kernel.compute,
            f"tl.inline_asm_elementwise('{asm}', '{constraints}', "
            f"[{args_str}], dtype=tl.float32, is_pure=True, pack=1)",
            dtype=tp.float32,
            shape=shape,
        )

    @staticmethod
    def _cuda_atanf(x: CSEVariable) -> CSEVariable:
        # A device's own atan for a single-precision argument, written out: a
        # range reduction against the reciprocal of the absolute value, then a
        # Horner polynomial in exact single-precision hex constants evaluated
        # with a fused multiply-add, and the result brought back by pi/2
        # reconstruction for |x| > 1, sign-bit restoration, and NaN handling.
        # Triton's bundled libdevice uses a different approximation, and a
        # 1 ULP atan difference can be amplified by steep consumers such as
        # digamma near zero.
        shape = x.shape

        def gen(expr: str, dtype: tp.dtype = tp.float32) -> TritonCSEVariable:
            return V.kernel.cse.generate(
                V.kernel.compute, expr, dtype=dtype, shape=shape
            )

        def fma(a: Any, b: Any, c: Any) -> Any:
            return TritonOverrides._inline_asm_f32(
                "fma.rn.f32 $0, $1, $2, $3;", [a, b, c], shape
            )

        c = TritonOverrides._f32_const_from_bits
        one = c(0x3F800000)
        inf = c(0x7F800000)

        abs_x = gen(f"tl_math.abs({x})")
        large = gen(f"{abs_x} > {one}", tp.bool)
        rcp = TritonOverrides._inline_asm_f32(
            "rcp.approx.ftz.f32 $0, $1;", [abs_x], shape
        )
        z = gen(f"tl.where({large}, {rcp}, {abs_x})")
        z2 = gen(f"{z} * {z}")

        poly = fma(c(0x3B2090AA), z2, c(0xBC6BE14F))
        poly = fma(poly, z2, c(0x3D23397E))
        poly = fma(poly, z2, c(0xBD948A7A))
        poly = fma(poly, z2, c(0x3DD76B21))
        poly = fma(poly, z2, c(0xBE111E88))
        poly = fma(poly, z2, c(0x3E4CAF60))
        poly = fma(poly, z2, c(0xBEAAAA27))

        correction = gen(f"{z2} * {poly}")
        result = fma(correction, z, z)
        neg_result = gen(f"-{result}")
        pi_over_two_minus_result = fma(c(0x3F6EE581), c(0x3FD774EB), neg_result)
        result = gen(f"tl.where({large}, {pi_over_two_minus_result}, {result})")

        sign_mask = "tl.full([], 0x80000000, tl.uint32)"
        sign = gen(f"{x}.to(tl.uint32, bitcast=True) & {sign_mask}", tp.uint32)
        result_bits = gen(f"{result}.to(tl.uint32, bitcast=True)", tp.uint32)
        signed_bits = gen(f"{sign} | {result_bits}", tp.uint32)
        signed_result = gen(f"{signed_bits}.to(tl.float32, bitcast=True)")
        not_nan = gen(f"{abs_x} <= {inf}", tp.bool)
        return gen(f"tl.where({not_nan}, {signed_result}, {result})")

    @staticmethod
    # pyrefly: ignore [bad-override]
    def atan(x):
        dtype = triton_arg_dtype(x)
        needs_upcast = needs_upcast_to_float32(x)
        if (
            not tp.version.hip
            and V.graph.get_current_device_or_throw().type == "cuda"
            and isinstance(x, CSEVariable)
            and (dtype == tp.float32 or needs_upcast)
        ):
            result_dtype = None
            if needs_upcast:
                result_dtype = dtype
                x = V.kernel.cse.generate(
                    V.kernel.compute,
                    f"{x}.to(tl.float32)",
                    dtype=tp.float32,
                    shape=x.shape,
                )
            result = TritonOverrides._cuda_atanf(x)
            if result_dtype is not None and result_dtype != tp.float32:
                return f"{result}.to({triton_type(result_dtype)})"
            return result

        upcast = ".to(tl.float32)" if needs_upcast else ""
        result = f"libdevice.atan({x}{upcast})"
        if needs_upcast and dtype is not None and dtype != tp.float32:
            return f"{result}.to({triton_type(dtype)})"
        return result

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def atanh(x):
        return f"libdevice.atanh({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def copysign(x, y):
        return f"libdevice.copysign({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def erfc(x):
        return f"libdevice.erfc({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def erfinv(x):
        return f"libdevice.erfinv({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def hypot(x, y):
        return f"libdevice.hypot({x}, {y})"

    @staticmethod
    @maybe_upcast_float32()
    def log10(x):
        return f"libdevice.log10({x})"

    @staticmethod
    @maybe_upcast_float32()
    def log2(x):
        return f"libdevice.log2({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def ldexp(x, n):
        return f"libdevice.ldexp({x}, {n}.to(tl.int32))"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def nextafter(x, y):
        return f"libdevice.nextafter({x}, {y})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def logical_not(a):
        return f"{a} == 0"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def logical_xor(a, b):
        return f"({a} ^ {b})"

    @staticmethod
    def bitwise_and(a, b):
        return f"{a} & {b}"

    @staticmethod
    def bitwise_not(a):
        return f"~{a}"

    @staticmethod
    def bitwise_or(a, b):
        return f"{a} | {b}"

    @staticmethod
    def bitwise_xor(a, b):
        return f"{a} ^ {b}"

    @staticmethod
    def bitwise_left_shift(a, b):
        return f"{a} << {b}"

    @staticmethod
    def bitwise_right_shift(a, b):
        return f"{a} >> {b}"

    @staticmethod
    def rand(seed, offset):
        offset = f"({offset}).to(tl.uint32)"
        if TritonOverrides._can_use_4x_random():
            (block,) = V.kernel.dense_size_list()
            return f"triton_helpers.rand4x({seed}, {offset}, {block})"
        return f"tl.rand({seed}, {offset})"

    @staticmethod
    def _can_use_4x_random():
        # Imported here rather than at the top: the template machinery imports
        # this module, so a module-level import of it would be a cycle.
        from ..templates.select_algorithm import TritonTemplateKernel

        return (
            isinstance(V.kernel, TritonKernel)
            and not isinstance(
                V.kernel, TritonTemplateKernel
            )
            and V.graph.get_current_device_or_throw().type == "cuda"
            and V.kernel.triton_tensor_ndim() == 1
            and not config.align_random_eager
        )

    @staticmethod
    def rand_eager(seed, base_offset, threads_per_round, tid, vec):
        # vec: 4 for fp32, 8 for fp16/bf16
        tid_u32 = f"({tid}).to(tl.uint32)"
        denom = f"(({vec})*({threads_per_round}))"
        r = f"(({tid_u32})//({denom})*({vec}//4))"
        tid_trunc = f"(({tid_u32})%({denom}))"

        return f"triton_helpers.rand_eager_kernel({seed}, {base_offset}+{r}, {tid_trunc}, VEC={vec})"

    @staticmethod
    def randn(seed, offset):
        offset = f"({offset}).to(tl.uint32)"
        if TritonOverrides._can_use_4x_random():
            (block,) = V.kernel.dense_size_list()
            return f"triton_helpers.randn4x({seed}, {offset}, {block})"
        return f"tl.randn({seed}, {offset})"

    @staticmethod
    def randint64(seed, offset, low, high):
        offset = f"({offset}).to(tl.uint32)"
        return f"triton_helpers.randint64({seed}, {offset}, {low}, {high})"

    @staticmethod
    def load_seed(name, offset):
        raise NotImplementedError("ops.load_seed not implemented outside a kernel")

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def rsqrt(x):
        if tp.version.hip:
            return f"tl.rsqrt({x})"
        else:
            return f"libdevice.rsqrt({x})"

    @staticmethod
    @maybe_upcast_float32()
    def log1p(x):
        return f"libdevice.log1p({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def tan(x):
        return f"libdevice.tan({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def tanh(x):
        cse_var = V.kernel.cse.varname_map.get(x)
        if cse_var and hasattr(cse_var, "dtype"):
            dtype = cse_var.dtype
        else:
            dtype = None
        if (
            config.use_fast_math
            and tp.version.hip
            and get_triton_version() > (3, 5)
            and dtype != tp.float64
            and dtype is not None
        ):
            # Requires upstream Triton 3.6+ for latest fast_tanhf support
            # https://github.com/triton-lang/triton/pull/8551
            return f"libdevice.fast_tanhf({x})"
        else:
            return f"libdevice.tanh({x})"

    @staticmethod
    @maybe_upcast_float32()
    def sigmoid(x):
        return f"tl.sigmoid({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def signbit(x):
        # x < 0 is wrong for -0.0 in floating point, so use libdevice for supported floating
        # dtypes on CUDA. On XPU, libdevice.signbit has a wrong float64 signature
        # (https://github.com/intel/intel-xpu-backend-for-triton/issues/7345); use a bitcast-based
        # sign-bit extraction as a workaround until the triton updates.
        if V.graph.get_current_device_or_throw().type == "xpu":
            return (
                f"({x}).to(tl.int64, bitcast=True) < 0 "
                f"if ({x}).dtype is tl.float64 "
                f"else (libdevice.signbit({x}) != 0) "
                f"if ({x}).dtype is tl.float32 "
                f"else {x} < 0"
            )
        return (
            f"(libdevice.signbit({x}) != 0) "
            f"if ({x}).dtype is tl.float32 or ({x}).dtype is tl.float64 "
            f"else {x} < 0"
        )

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def fmod(a, b):
        return f"libdevice.fmod({a}, {b})"

    @classmethod
    def pow(cls, a, b):
        result_dtype = get_dtype_handler().pow(a, b)
        if result_dtype is not None and is_integer_dtype(result_dtype):
            base = cls._cast_libdevice_arg(a, result_dtype)
            exponent = (
                cls.constant(b, tp.int64)
                if isinstance(b, Number)
                else f"{b}"
            )
            return f"triton_helpers.pow_integer({base}, {exponent})"

        any_needs_upcast = needs_upcast_to_float32(a) or needs_upcast_to_float32(b)
        pow_dtype = result_dtype
        if pow_dtype not in (tp.float32, tp.float64):
            # libdevice.pow only accepts fp32/fp64. Keep low-precision floating
            # cases on the existing fp32 path, and otherwise fall back to fp64
            # for symbolic integer scalar pow expressions like 2 ** ks0.
            pow_dtype = (
                tp.float32
                if low_precision_fp(result_dtype) or any_needs_upcast
                else tp.float64
            )
        if pow_dtype == tp.float64 and not device_supports_fp64(
            V.graph.current_device
        ):
            pow_dtype = tp.float32
            if result_dtype == tp.float64:
                result_dtype = tp.float32

        cast_a = cls._cast_libdevice_arg(a, pow_dtype)
        cast_b = cls._cast_libdevice_arg(b, pow_dtype)
        result = f"libdevice.pow({cast_a}, {cast_b})"
        if result_dtype is not None and result_dtype != pow_dtype:
            if low_precision_fp(result_dtype):
                if any_needs_upcast:
                    result = f"{result}.to({triton_type(result_dtype)})"
            else:
                result = f"{result}.to({triton_type(result_dtype)})"
        return result

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def log(x):
        if config.eager_numerics.use_device_libdevice:
            # Strict numerics should use the backend math library entry point.
            # On ROCm this maps to OCML and avoids Triton's generic log lowering.
            return f"libdevice.log({x})"
        return f"tl_math.log({x})"

    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    # pyrefly: ignore [bad-override]
    def isinf(x):
        return f"libdevice.isinf({x}).to(tl.int1)"

    @staticmethod
    @maybe_upcast_float32(convert_output=False)
    # pyrefly: ignore [bad-override]
    def isnan(x):
        return f"libdevice.isnan({x}).to(tl.int1)"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def round(x):
        return f"libdevice.nearbyint({x})"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def floor(x):
        return f"libdevice.floor({x})"

    @staticmethod
    def floordiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv.
        #
        # We avoid feeding negative values into Triton's // operator because
        # Triton's AxisInfo analysis incorrectly deduplicates signed division
        # results for contiguous inputs (triton-lang/triton#XXXX). Instead we
        # use bitwise complement (~) to make the dividend non-negative:
        #   floor_div(a, b) = ~(~a // b) when a < 0, a // b when a >= 0
        # For negative b we negate both operands first.
        zero = ops.constant(0, tp.int32)
        one = ops.constant(1, tp.int32)
        # Guard against integer division by zero before the division to
        # avoid undefined behavior (LLVM may optimize away a post-division
        # check assuming UB doesn't happen). Replace b with 1 when b is 0
        # so the division is safe, then select 0 as the final result.
        b_zero = ops.eq(b, zero)
        b = ops.where(b_zero, one, b)
        b_neg = ops.lt(b, zero)
        a = ops.where(b_neg, ops.sub(zero, a), a)
        b = ops.where(b_neg, ops.sub(zero, b), b)
        a_neg = ops.lt(a, zero)
        a = ops.where(a_neg, ops.bitwise_not(a), a)
        quot = ops.truncdiv(a, b)
        quot = ops.where(a_neg, ops.bitwise_not(quot), quot)
        return ops.where(b_zero, zero, quot)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def sign(x):
        z = ops.constant(0, tp.int32)
        left = ops.to_dtype((ops.lt(z, x)), tp.int8)
        right = ops.to_dtype((ops.lt(x, z)), tp.int8)
        sub = ops.sub(left, right)
        return f"{sub}.to({x}.dtype)"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def trunc(x):
        return f"libdevice.trunc({x})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def truncdiv(a, b):
        # See the comment in lowering.div_mode. a and b are integer type.
        # Notice that // in triton behaves as truncdiv instead of floordiv
        return f"{a} // {b}"

    @staticmethod
    @maybe_upcast_float32()
    # pyrefly: ignore [bad-override]
    def ceil(x):
        return f"libdevice.ceil({x})"


# Register the custom pow override after class creation so type checkers see
# a plain callable instead of the class-body staticmethod descriptor.
class TritonKernelOverrides(TritonOverrides):
    """Map element-wise ops to Triton within a TritonKernel

    Unlike TritonOverrides, these assume the code is going to be inserted into
    the body of the main triton kernel and so it may use indexing and mask
    variables which are assumed to already be defined in the current scope.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # happens in __init__ unlike _initialize_pointwise_overrides
        # because the libdevice registrations are populated during lowerings
        self._setup_libdevice_routing()

    @classmethod
    @functools.cache
    def _setup_libdevice_routing(cls):
        """Set up routing to libdevice implementations for fp64 inputs."""

        from common import OpDecompositions

        for fn_name in op_requires_libdevice_fp64:
            if not hasattr(cls, fn_name):
                raise AssertionError(f"missing method {fn_name} on {cls}")
            original_impl = getattr(cls, fn_name)

            def decomposition_router(x, _original_impl, _fn_name):
                if x.dtype != tp.float64:
                    return _original_impl(x)
                else:
                    return getattr(OpDecompositions, _fn_name)(x).value

            if fn_name == "sigmoid":
                if not hasattr(OpDecompositions, "sigmoid"):
                    raise AssertionError("OpDecompositions must define sigmoid")
                fn = functools.partial(
                    decomposition_router, _original_impl=original_impl, _fn_name=fn_name
                )
                fn.__name__ = fn_name  # type: ignore[attr-defined]
                setattr(cls, fn_name, staticmethod(fn))
                continue

            def dtype_router(x, _original_impl, _fn_name):
                if x.dtype == tp.float64:
                    return f"libdevice.{_fn_name}({x})"
                else:
                    return _original_impl(x)

            fn = functools.partial(
                dtype_router, _original_impl=original_impl, _fn_name=fn_name
            )
            fn.__name__ = fn_name  # type: ignore[attr-defined]
            setattr(cls, fn_name, staticmethod(fn))

    @classmethod
    def constant(cls, value, dtype):
        # NOTE: Cannot use shape=[] as it's not supported by triton-rocm
        # We could use shape=[1] instead but starting with the correct
        # ndim avoids extra `tt.expand_dim` ops appearing in the triton IR.
        ndim = V.kernel.triton_tensor_ndim()
        shape = [1] * ndim
        return cls._shaped_constant(value, dtype, shape=shape)

    @classmethod
    def index_expr(cls, expr, dtype):
        expr = _materialize_trunc_to_float_expr(expr, dtype)
        indexing = V.kernel.indexing(
            expr, block_ptr=False, tma_compatibility_checker=None
        )
        if not isinstance(indexing, IndexingOptions):
            raise AssertionError(f"expected IndexingOptions, got {type(indexing)}")

        shape: BlockShapeType
        if indexing.expand_shape:
            shape = indexing.expand_shape
        else:
            shape = TritonSymbols.get_block_shape(indexing.index)

        # Our sympy expr printing casts to the current kernel index dtype.
        # We only respect non-int32/int64 dtypes and otherwise use the current
        # kernel indexing dtype. Value-producing uses that need to preserve an
        # explicit int dtype are rewritten to value_expr.
        index_dtype = V.kernel.get_index_dtype_as_torch_dtype()
        if dtype in (tp.int32, tp.int64):
            cast_dtype = index_dtype
        else:
            cast_dtype = dtype
        # to_dtype(..., use_compute_types=True) emits the Triton compute type
        # for cast_dtype, so the CSE metadata must be derived from that same
        # dtype.  For example, float16 requests emit tl.float32 when compute
        # upcasting is enabled, and should be recorded as tp.float32.
        output_dtype = upcast_compute_type(cast_dtype)
        var = V.kernel.cse.generate(
            V.kernel.compute,
            cls.to_dtype(f"({indexing.index_str})", cast_dtype),
            dtype=output_dtype,
            bounds=get_bounds_index_expr(expr),
            shape=shape,
        )

        var.mask_vars = indexing.mask_vars
        return var

    @classmethod
    def value_expr(cls, expr, dtype):
        """
        Like :meth:`index_expr`, but honors ``dtype`` by setting the kernel
        index dtype before emitting, and casting the result if needed.
        """
        real_index_dtype = V.kernel._index_dtype
        V.kernel._index_dtype = (
            dtype if dtype in (tp.int32, tp.int64) else tp.int64
        )
        try:
            var = cls.index_expr(expr, dtype)
        finally:
            V.kernel._index_dtype = real_index_dtype
        if real_index_dtype != dtype or var.dtype != dtype:
            var = V.kernel.cse.generate(
                V.kernel.compute,
                f"({var}).to({triton_type(dtype)})",
                dtype=dtype,
                shape=var.shape,
            )
        return var

    @staticmethod
    def masked(mask, body, other):
        if mask is not None and tp.version.hip is not None:
            mask = V.kernel.cse.generate(
                V.kernel.compute,
                f"{mask}.to(tl.int1)",
                dtype=tp.bool,
                shape=mask.shape,
            )

        nodes = body.graph.find_nodes(op="output")
        if not nodes:
            raise AssertionError("graph for body does not contain an output")

        need_where = False
        # If we have a tl.load with a masking operator and no other value
        # we can add the mask here and the other value to the tl.load
        # operator to save the branching cost.
        for node in nodes:
            for arg in node.args:
                if (
                    arg.target != "load"
                    or should_unwrap_unspec_arg(arg.args[1])
                    # A load whose producer is fused into this kernel is
                    # served from the CSE store cache and emits no tl.load,
                    # so the masked-load `other` would be silently dropped;
                    # fall back to an explicit tl.where.
                    or arg.args[1] in V.kernel.cse.store_cache
                ):
                    need_where = True
                    break

        value = None if need_where else other

        with V.kernel.mask_loads(mask, value=value) as new_mask:
            result = body()

        if need_where:
            # Remove once CSEVariables track the dtype
            if result.bounds.is_bool:
                other = bool(other)
            # Take dtype from result to prevent accidental promotion
            other = V.kernel.cse.generate(
                V.kernel.compute,
                f"tl.full({result}.shape, {constant_repr(other)}, {result}.dtype)",
                bounds=ValueRanges.wrap(other),
                dtype=result.dtype,
                shape=result.shape,
            )
            ret = ops.where(new_mask, result, other)
        else:
            ret = result

        ret.mask_vars.discard(new_mask)
        return ret

    @staticmethod
    def load_seed(name, offset):
        var = V.kernel.args.input(name)
        return (
            f"tl.load({var} + {V.kernel.args.seed_offset('load_seed_offset', offset)})"
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def frexp(x):
        cache_key = f"frexp({x})"
        if cse_val := V.kernel.cse.try_get(cache_key):
            return cse_val

        mantissa = V.kernel.cse.newvar(dtype=x.dtype, shape=x.shape)
        exponent = V.kernel.cse.newvar(dtype=tp.int32, shape=x.shape)
        V.kernel.compute.writeline(
            f"{mantissa}, {exponent} = triton_helpers.frexp({x})"
        )
        V.kernel.cse.put(cache_key, (mantissa, exponent))
        return (mantissa, exponent)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: CSEVariable,
        extra_meta: dict[str, Any],
    ) -> None:
        raise NotImplementedError


OpDtypeSupport.register_upcast(TritonOverrides.atan, True)
OpDtypeSupport.register_upcast(TritonOverrides.pow, True)

# The pointwise table that decides which of these spellings each operation
# takes is shared with the other emitters, and is registered by whoever brings
# that table over. Registering it here as well would set every operation twice,
# so it is left to arrive with the table.


