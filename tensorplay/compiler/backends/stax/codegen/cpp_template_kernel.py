"""Writing a kernel whose body is assembled by a program rather than by a template.

The other C++ route emits a whole kernel from a schedule: the loop nest, the
loads, the stores, all of it decided ahead of time.  This one hands the decision
to something written in Python, which is what a template is -- a program that
emits a program.  So what a template can name has to exist as an operation on
this kernel: a size, a distance, a load at a position, a view of a region, a
store.

Which is the whole of the difference, and it is why this is a subclass rather
than a mode.  A schedule is walked once and what it walks over is the schedule's
own vocabulary.  A template is walked many times over a buffer it was handed,
and the two want different answers for the same question: the same buffer asked
for a size is a name in this kernel's argument list, not a number computed from
a loop counter.

Everything the template writes is C++ text assembled here, and the buffers it
writes to are the ones it was handed rather than ones it allocated -- except the
ones it declares itself, which are local to one call and are named so that the
code which allocates them can be told apart from the code that uses them.
"""

from __future__ import annotations

import functools
import itertools
from typing import Any
from unittest.mock import patch

import sympy
from sympy.parsing.sympy_parser import parse_expr

import tensorplay as tp


from tensorplay.graph.experimental.sympy_functions import SymT
from .. import config, cpp_builder, ir
from ..ir import OrderedSet
from .. import op_lowerings as L
from ..autotune_process import CppBenchmarkRequest
from ..loop_body import LoopBody
from ..templates.select_algorithm import PartialRender
from ..utils import (
    do_bench_using_profiling,
    sympy_index_symbol,
    sympy_index_symbol_with_prefix,
)
from ..loops import V
from .common import REMOVED
from .cpp import CppKernel, CppKernelProxy, KernelGroup, ParallelDepth
from .cpp_utils import DTYPE_TO_CPP, LocalBufferContext, cexpr_index


def parse_expr_with_index_symbols(expr: Any) -> Any:
    """An index written as text, with the symbols in it named as indices.

    A template writes positions as text, because a template is a program that
    emits a program and the position it is emitting is not known to it as a
    value.  The symbols in that text are the kernel's index variables, so they
    are renamed to those before the text becomes an expression -- otherwise a
    symbol written in a template would be a size the graph has never heard of.
    """

    if isinstance(expr, sympy.Expr):
        return expr
    elif isinstance(expr, (list, tuple)):
        return [parse_expr_with_index_symbols(e) for e in expr]
    else:
        expr = parse_expr(str(expr))
        int_symbols = {sym: sympy_index_symbol(sym.name) for sym in expr.free_symbols}
        return expr.subs(int_symbols)


def wrap_with_tensorbox(node: Any) -> Any:
    """A value a view can be built from, whether or not it already is one.

    A template asks for a view of whatever it was handed, and what it was handed
    may already be a value or may be the buffer behind one.  Both are the same
    thing to ask a view of, so both are made into the same shape first.
    """

    return (
        ir.TensorBox.create(node) if isinstance(node, ir.Buffer) else ir.TensorBox(node)
    )


class CppTemplateKernel(CppKernel):
    """A C++ kernel whose body is written by something handed to it at render time.

    The body arrives as a program, and the program is run against this kernel:
    every name it uses is a question about a buffer, and the answer is a name in
    this kernel's argument list.  So the buffers are not walked -- they are
    named -- and a name in the emitted code and a name in the argument list are
    the same name, which is what makes the emitted code callable.
    """

    def __init__(self, kernel_name: str, num_threads: int) -> None:
        super().__init__(None, num_threads)
        self.kernel_name = kernel_name
        #: Text to splice in later, keyed by what the template wrote in its place.
        self.render_hooks: dict = {}
        #: Buffers this kernel allocated for itself, by the name it allocated
        #: them under.  Not arguments: nobody handed them over.
        self.local_buffers: dict = {}

    def render(self, template: Any, **kwargs: Any) -> str:
        """Run the body and hand back the whole of what it produced.

        Not just the text: the hooks are the deferred half of it.  A template
        writes a placeholder where a function's signature goes, because the
        signature is not known until every buffer it mentions has been named --
        and naming them is something the body does while it runs.  So the
        placeholder comes back with a function that fills it in, and both halves
        are needed.
        """

        return PartialRender(
            template.render(kernel=self, **kwargs), self.render_hooks
        ).finalize_all()

    def def_kernel(
        self,
        inputs: dict,
        outputs: dict,
        aliases: dict | None = None,
        function_name: str = "",
        extra_sizevars: list | None = None,
        placeholder: str = "<DEF_KERNEL>",
    ) -> str:
        """Name a function, and hand back where its signature goes.

        A function's arguments are the buffers it reads and the buffers it
        writes, and a template does not know those until it has run -- so it
        asks for the function here and is given a place to put the signature,
        which is filled in once the body has named everything it uses.

        An alias is a second name for a buffer that is already named, so a
        template can call the same buffer by two names and have them mean one
        thing.  That is a convenience for the body, not a second buffer, and it
        is why an alias is resolved against both lists rather than becoming one.
        """

        if len(function_name) == 0:
            function_name = str(self.kernel_name)
        for name, inp in inputs.items():
            if inp is not None:
                self.args.input_buffers[inp.get_name()] = name
        for name, out in outputs.items():
            self.args.output_buffers[out.get_name()] = name
        if aliases is not None:
            for alias, orig in aliases.items():
                if orig in self.args.input_buffers:
                    self.args.input_buffers[alias] = self.args.input_buffers[orig]
                if orig in self.args.output_buffers:
                    self.args.output_buffers[alias] = self.args.output_buffers[orig]

        # Every size the function mentions has to be a name rather than an
        # expression, because the caller is handed numbers and the function is
        # not.  Collected from the shapes and the distances of everything it
        # touches, plus whatever the body said it needed, in a name order so
        # that the same function written twice has the same arguments.
        unique_sizevars = OrderedSet(
            s
            for input in inputs.values()
            if input is not None
            for sym in itertools.chain(input.get_size(), input.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for sym in extra_sizevars or []
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        unique_sizevars.update(
            s
            for output in outputs.values()
            for sym in itertools.chain(output.get_size(), output.get_stride())
            if isinstance(sym, sympy.Expr)
            for s in sym.free_symbols
        )
        sizevars = sorted(unique_sizevars, key=str)
        for sizevar in sizevars:
            self.args.sizevars[sizevar] = f"k{sizevar}"

        def hook() -> str:
            # An alias was a second name for a buffer, and the function is
            # written with one of them -- so the other is taken away rather than
            # becoming a second argument for the same buffer.
            if aliases is not None:
                for alias in aliases:
                    if alias in self.args.input_buffers:
                        raise AssertionError(
                            f"input_buffers cannot be removed: {alias}"
                        )
                    if alias in self.args.output_buffers:
                        self.args.output_buffers[alias] = REMOVED
            cpp_argdefs, _, _ = self.args.cpp_argdefs()
            return f"void {function_name}({', '.join(cpp_argdefs)})"

        if placeholder in self.render_hooks:
            raise AssertionError(f"placeholder already registered: {placeholder}")
        self.render_hooks[placeholder] = hook
        return placeholder

    def call_kernel(self, name: str, node: Any) -> None:
        """Write the call to this kernel, once everything it needs is named."""

        wrapper = V.graph.wrapper_code
        _, call_args, arg_types = self.args.cpp_argdefs()
        wrapper.generate_kernel_call(name, call_args, triton=False, arg_types=arg_types)

    def dtype(self, node: Any) -> str:
        return DTYPE_TO_CPP[node.get_dtype()]

    def acc_dtype(self, node: Any) -> str:
        if node.get_dtype() in (tp.float32, tp.bfloat16, tp.half):
            return "float"
        else:
            raise NotImplementedError(f"Unsupported dtype: {node.get_dtype()}")

    def size(self, node: Any, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_size()[dim]))

    def stride(self, node: Any, dim: int) -> str:
        return cexpr_index(self.rename_indexing(node.get_stride()[dim]))

    def index(self, node: Any, indices: list) -> str:
        """A read at a position, as C++ text.

        The position is written by the template and so is text, and the buffer
        it reads is named by the caller -- so what comes out is a name in this
        kernel's argument list and an offset into it, which is the only thing
        the emitted code can say about a read.
        """

        indexer = node.get_layout().as_fixed().make_indexer()
        index = indexer(parse_expr_with_index_symbols(indices))
        index = self.rename_indexing(index)
        outer_name = node.get_name()
        inner_name = (
            outer_name
            if outer_name in self.local_buffers
            else self.args.input(node.get_name())
        )
        return f"{inner_name}[{cexpr_index(index)}]"

    def slice_nd(self, node: Any, ranges: list) -> Any:
        """A region of a buffer, as a view of it.

        A dimension given no range is left whole, which is what makes this a
        list of ranges rather than a shape: the caller says which axes it wants
        and the others are not questions it has to answer.
        """

        if len(ranges) != len(node.get_size()):
            raise AssertionError(f"{ranges=}, {node=}")
        sliced = wrap_with_tensorbox(node)
        for dim, _range in enumerate(ranges):
            if len(_range) == 0:
                continue
            if len(_range) != 2:
                raise AssertionError(f"expected range of length 2, got {len(_range)}")
            start, end = parse_expr_with_index_symbols(_range)
            sliced = L.slice_(sliced, dim, start, end, clamp=False)
        if not isinstance(sliced, ir.TensorBox):
            raise AssertionError(f"expected ir.TensorBox, got {type(sliced)}")
        if not isinstance(sliced.data, ir.ReinterpretView):
            raise AssertionError(sliced.data)
        return sliced.data

    def select(self, node: Any, dim: int, idx: int) -> Any:
        """One position along one axis, as a view of it.

        Not the select the lowering offers, because that leaves the axis with
        what is left of its length rather than with one -- and a caller asking
        for one position wants the axis to be gone, not to be an expression of
        what it was less.
        """

        node = wrap_with_tensorbox(node)
        idx = ir.View.handle_negative_index(idx, node.get_size()[dim])
        sliced = L.squeeze(L.slice_(node, dim, idx, idx + 1, clamp=False), dim)
        if not isinstance(sliced.data, ir.ReinterpretView):
            raise AssertionError(sliced.data)
        return sliced.data

    def view(self, node: Any, sizes: list) -> Any:
        node = wrap_with_tensorbox(node)
        sizes = parse_expr_with_index_symbols(sizes)
        return L.view(node, sizes).data

    def permute(self, node: Any, dims: Any) -> Any:
        node = wrap_with_tensorbox(node)
        permuted = L.permute(node, dims).data
        if not isinstance(permuted, ir.ReinterpretView):
            raise AssertionError(f"expected ir.ReinterpretView, got {type(permuted)}")
        return permuted

    def maybe_codegen_profile(self, prefix_kernel_name: str | None = None) -> str:
        """A record of this kernel's run, when profiling was asked for.

        Named after the graph it is in and the kernel it is, so a record read
        later says which kernel ran rather than only that one did.
        """

        if config.cpp.enable_kernel_profile:
            graph_id = V.graph.graph_id
            prefix = "graph_" + str(graph_id) + "_" if graph_id is not None else ""
            if prefix and prefix_kernel_name:
                prefix += prefix_kernel_name + "_"
            handle_str = (
                "tp::aot::RAIIRecordFunctionHandle "
                f'record_{prefix}{self.kernel_name}_("{prefix}{self.kernel_name}", nullptr);'
            )
            return handle_str
        else:
            return ""

    def unroll_pragma(self, unroll: Any) -> str:
        """The pragma that says a loop's trip count is known.

        Spelled differently by the two compilers that are in use, and a pragma
        the compiler does not know is an error rather than a hint.
        """

        if cpp_builder.is_gcc():
            return f"#pragma GCC unroll {unroll}"
        else:
            return f"#pragma unroll {unroll}"

    def define_buffer(self, name: str, sizes: list, dtype: Any = tp.float) -> str:
        """A buffer this kernel allocates, once, for its own use.

        On the heap rather than the stack because its size comes from the shapes
        the caller passed and so is not known here -- a stack buffer would be
        sized by a guess.  Named twice: the owning pointer under one name and the
        usable one under another, because the owner is what has to be released
        and the usable one is what the body writes through.
        """

        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(tp.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"auto _{name} = std::make_unique<{ctype}[]>({numel}); auto {name} = _{name}.get();"

    def define_stack_allocated_buffer(
        self, name: str, sizes: list, dtype: Any = tp.float
    ) -> str:
        """A buffer this kernel allocates on the stack, for a size it knows.

        Aligned to sixty-four bytes because that is what the unit reading it
        wants, and unaligned access to a wide value costs a load per element.
        """

        sizes = parse_expr_with_index_symbols(sizes)
        buf = ir.Buffer(
            name=name, layout=ir.FixedLayout(tp.device("cpu"), dtype, sizes)
        )
        self.local_buffers[name] = buf
        ctype = f"{DTYPE_TO_CPP[dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"alignas(64) {ctype} _{name}[{numel}]; {ctype}* {name} = _{name};"

    def reinit_buffer_if_null(self, name: str) -> str:
        """Allocate a buffer that was declared but whose size was not yet known.

        A buffer's size can come from something the body works out, so it is
        sometimes declared empty and filled in later.  Allocating on each fill
        would leak the earlier ones, so it is allocated once and only when there
        is nothing to reuse.
        """

        if name not in self.local_buffers:
            raise AssertionError(f"unknown local buffer: {name}")
        buf = self.local_buffers[name]
        ctype = f"{DTYPE_TO_CPP[buf.layout.dtype]}"
        numel = f"{cexpr_index(buf.get_numel())}"
        return f"if (_{name} == nullptr) {{ _{name} = std::make_unique<{ctype}[]>({numel}); {name} = _{name}.get(); }}"

    def release_buffer(self, name: str) -> str:
        """Hand a buffer this kernel allocated over to whoever comes next.

        One buffer allocated by two kernels would be freed twice, so ownership
        passes out rather than being dropped here.
        """

        if name not in self.local_buffers:
            raise AssertionError(f"unknown local buffer: {name}")
        return f"_{name}.release()"

    def store_pointwise_nodes(
        self,
        dst: Any,
        nodes: list,
        offsets: list | None = None,
        reindexers: list | None = None,
    ) -> str:
        """Write a whole loop nest that computes each of these into ``dst``.

        The template said what to write but not how to get there, so the nest is
        made here and the template's arithmetic runs at each position inside it.
        An offset per body says where in ``dst`` that body's result goes, and a
        reindexer per body says which position of it -- so a template can write
        several bodies into one buffer without any of them knowing about the
        others.
        """

        var_sizes = (tuple(dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            offsets = [sympy.S.Zero] * len(var_sizes[0])
        if not reindexers:
            reindexers = [None] * len(nodes)
        if len(offsets) != len(var_sizes[0]):
            raise AssertionError(
                f"expected {len(var_sizes[0])} offsets, got {len(offsets)}"
            )
        output_index = dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = node.get_name() if i < len(nodes) - 1 else dst.get_name()
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            if not isinstance(node, ir.Pointwise):
                raise AssertionError(node)

            def fn(*args: Any) -> None:
                if len(args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(args)}")
                if len(args[0]) != len(var_sizes[0]):
                    raise AssertionError(
                        f"expected {len(var_sizes[0])} indices, got {len(args[0])}"
                    )
                if len(args[1]) != 0:
                    raise AssertionError(
                        f"expected no reduction vars, got {len(args[1])}"
                    )
                new_args = [arg + offset for arg, offset in zip(args[0], offsets)]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth() -> Any:
            return ParallelDepth(parallel_depth=0, start_depth=0)

        # This loop is not parallelized since it is not the outermost loop.
        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_grouped_gemm_pointwise_nodes(
        self,
        dst: tuple,
        nodes: list,
        offsets: list,
        reindexers: list,
        output_names: list,
    ) -> str:
        """The same, for a body per output of a grouped product.

        One nest over the first output's shape, and the others written into the
        same nest -- which is what makes them one kernel rather than several.
        Every offset here is required rather than optional, because with more
        than one output a body that did not say where its result goes would have
        no way of being told apart from one that did.
        """

        ref_dst = dst[0]
        var_sizes = (tuple(ref_dst.get_size()), ())
        var_ranges = {
            sympy_index_symbol_with_prefix(SymT.INDEX, i): sz
            for i, sz in enumerate(var_sizes[0])
        }
        if not offsets:
            raise AssertionError("offsets should be set outside")
        if not all(len(offset) == len(var_sizes[0]) for offset in offsets):
            raise AssertionError(f"expected all offsets of length {len(var_sizes[0])}")
        output_index = ref_dst.get_layout().make_indexer()([*var_ranges.keys()])
        kernel_group = KernelGroup()
        kernel_group.args = self.args
        cpp_kernel_proxy = CppKernelProxy(kernel_group)
        bodies = []
        var_sizes_list = []
        for i, node in enumerate(nodes):
            output_name = output_names[i]
            node = node.data if isinstance(node, ir.ComputedBuffer) else node
            if not isinstance(node, ir.Pointwise):
                raise AssertionError(node)

            def fn(*args: Any) -> None:
                if len(args) != 2:
                    raise AssertionError(f"expected 2 args, got {len(args)}")
                if len(args[0]) != len(var_sizes[0]):
                    raise AssertionError(
                        f"expected {len(var_sizes[0])} indices, got {len(args[0])}"
                    )
                if len(args[1]) != 0:
                    raise AssertionError(
                        f"expected no reduction vars, got {len(args[1])}"
                    )
                new_args = [arg + offset for arg, offset in zip(args[0], offsets[i])]
                if reindexers[i] is not None:
                    new_args = reindexers[i](new_args)
                V.ops.store(
                    output_name,
                    output_index,
                    node.make_loader()(new_args).value,
                )

            body = LoopBody(
                fn,
                (list(var_ranges.keys()), ()),
                var_ranges,
                list(var_ranges.keys()),
                tuple(),
            )
            bodies.append(body)
            var_sizes_list.append(var_sizes)

        cpp_kernel_proxy.codegen_loop_bodies(bodies, var_sizes_list)

        def max_parallel_depth() -> Any:
            return ParallelDepth(parallel_depth=0, start_depth=0)

        with patch.object(
            cpp_kernel_proxy.loop_nest, "max_parallel_depth", max_parallel_depth
        ):
            kernel_group.finalize_kernel(cpp_kernel_proxy, [])
        return kernel_group.loops_code.getvalue()

    def store_outputs(
        self,
        dst: tuple[ir.Buffer],
        src: tuple[ir.IRNode],
        orig_src: tuple[ir.IRNode] | None = None,
        epilogue_nodes: list[ir.IRNode] | None = None,
        offsets: list[Any] | None = None,
        reindexers: list[Callable[[list[Any]], list[Any]] | None] | None = None,
        multi_output_buffers: tuple[ir.MultiOutput, ...] | None = None,
    ):
        """Write several results, each from its own body, into one loop nest.

        One nest rather than several, so a template that produces three results
        produces one kernel.  Which body wrote which result is settled by name
        before the nest is walked -- a body that wrote into a buffer another
        body also writes would otherwise be indistinguishable from one that did
        not.

        The parts that are not obvious from the arithmetic are about buffers
        belonging to someone else.  A body may read a result this kernel is about
        to write, and reading a buffer another kernel owns is not something this
        kernel may do -- so such a result is copied in first, and only if
        something actually reads it.
        """

        if not isinstance(dst, Iterable):
            raise AssertionError(f"expected Iterable, got {type(dst)}")
        if not all(_dst.get_size() == _src.get_size() for _src, _dst in zip(src, dst)):
            raise AssertionError("src and dst sizes must match")
        if offsets:
            offsets = parse_expr_with_index_symbols(offsets)
        gemm_num = len(src)
        final_offsets = []
        output_names = []
        if epilogue_nodes:
            if not reindexers:
                reindexers = [None] * len(epilogue_nodes)
            with LocalBufferContext(self.args) as scope:
                if orig_src is None:
                    raise AssertionError("orig_src must not be None")
                localize_epilogue_nodes = []
                all_read_names = []
                for epilogue in epilogue_nodes:
                    all_read_names.extend(list(epilogue.get_read_names()))
                localize_epilogue_nodes.extend(scope.localize_nodes(epilogue_nodes))
                final_offsets.extend([offsets] * len(localize_epilogue_nodes))
                output_names.extend(
                    [node.get_name() for node in localize_epilogue_nodes]
                )
                for gemm_idx in range(gemm_num):
                    if orig_src[gemm_idx].get_name() != src[gemm_idx].get_name():
                        if orig_src[gemm_idx].get_name() in all_read_names or (
                            multi_output_buffers
                            and multi_output_buffers[gemm_idx].get_name()
                            in all_read_names
                        ):
                            # If any of the Epilogue nodes use this GEMM output, let's localize the GEMM output
                            global_buffers = [orig_src[gemm_idx]]
                            if (
                                multi_output_buffers
                                and multi_output_buffers[gemm_idx].get_name()
                                in all_read_names
                                and orig_src[gemm_idx].get_name() not in all_read_names
                            ):
                                # Epilogue might directly read the MultiOutput, Localize MultiOutput to the local Buffer
                                # if this MultiOutput has not been stored by in-template epilogue
                                # otherwise, use the cse store cache if it will be stored before used
                                global_buffers.append(multi_output_buffers[gemm_idx])
                            scope.add_local_buffer(
                                src[gemm_idx],
                                global_buffers,
                            )
                        else:
                            scope.add_local_buffer(src[gemm_idx])
                            localize_epilogue_nodes.extend(
                                [L.copy(dst[gemm_idx], src[gemm_idx]).data.data]
                            )
                            reindexers.append(None)
                            output_names.append(dst[gemm_idx].get_name())
                            final_offsets.append(
                                [sympy.S.Zero] * len(dst[gemm_idx].get_size())
                            )
                res = self.store_grouped_gemm_pointwise_nodes(
                    dst,
                    localize_epilogue_nodes,
                    final_offsets,
                    reindexers,
                    output_names=output_names,
                )
                for gemm_idx in range(gemm_num):
                    if (
                        multi_output_buffers
                        and multi_output_buffers[gemm_idx].get_name() in all_read_names
                    ):
                        # If the MultiOutput is used in the Epilogue, let's remove it from args
                        multi_output_name = multi_output_buffers[gemm_idx].get_name()
                        if (
                            multi_output_name in self.args.output_buffers
                            and self.args.output_buffers[multi_output_name]
                            is not REMOVED
                        ):
                            self.remove_buffer(multi_output_name)
                return res
        else:
            if dst[0].get_name() != src[0].get_name():
                copy_list = []
                with LocalBufferContext(self.args) as scope:
                    for _src, _dst in zip(src, dst):
                        copy_list.extend([L.copy(_dst, _src).data.data])
                        scope.add_local_buffer(_src)
                        output_names.append(_dst.get_name())
                        final_offsets.append([sympy.S.Zero] * len(_dst.get_size()))
                    reindexers = [None] * len(copy_list)
                    return self.store_grouped_gemm_pointwise_nodes(
                        dst,
                        nodes=copy_list,
                        offsets=final_offsets,
                        reindexers=reindexers,
                        output_names=output_names,
                    )
            else:
                if not all(
                    _src.get_name() == _dst.get_name() for _src, _dst in zip(src, dst)
                ):
                    raise AssertionError("src and dst names must match")
                if not all(
                    _src.get_layout() == _dst.get_layout()
                    for _src, _dst in zip(src, dst)
                ):
                    raise AssertionError("src and dst layouts must match")
                return ""

    def check_bounds(self, expr: Any, size: Any, lower: Any, upper: Any) -> None:
        """Nothing: this kernel's body decides its own bounds.

        The scheduled route asks whether a read is inside the buffer, because it
        generated the read and may have generated it wrong.  A template wrote
        the read itself and knows which positions it visits, so a check here
        would be a comparison per element against something the writer already
        established.
        """

        return


class CppTemplateCaller(ir.ChoiceCaller):
    """A built choice: one way of writing a kernel, ready to be measured.

    Not yet a kernel.  What is here is everything but the part that names the
    buffers, which is the part a template produces by running -- so it is a
    factory, and the kernel comes out of it once the buffers are known.
    """

    def __init__(
        self,
        name: str,
        category: str,
        input_nodes: list,
        layout: Any,
        make_kernel_render: Any,
        bmreq: Any,
        template: Any,
        info_kwargs: Any = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description="")
        self.category = category
        self.make_kernel_render = make_kernel_render
        self.bmreq = bmreq
        self.template = template
        self.info_kwargs = info_kwargs

    def precompile(self) -> None:
        if self.bmreq is None:
            raise AssertionError("bmreq must not be None")
        self.bmreq.precompile()

    def benchmark(self, *args: Any, out: Any) -> float:
        if self.bmreq is None:
            raise AssertionError("bmreq must not be None")
        if config.profile_bandwidth_with_do_bench_using_profiling:
            algo = self.bmreq.make_run_fn(*args, out=out)
            return do_bench_using_profiling(algo)
        return self.bmreq.benchmark(*args, out=out)

    def hash_key(self) -> str:
        return "-".join(
            [
                self.category,
                self.bmreq.hash_key,
            ]
        )

    def info_dict(self) -> dict:
        return {"backend": "CPP", "op_type": "unknown"}

    def output_node(self) -> Any:
        buffer = ir.CppTemplateBuffer(
            layout=self.layout,
            inputs=self.input_nodes,
            make_kernel_render=self.make_kernel_render,
            template=self.template,
            choice=self,
        )
        if "ktc" in self.annotations:
            buffer.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(buffer)
