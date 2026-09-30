"""Taking a chosen subgraph apart into the region that chose it.

A subgraph is compiled as a program of its own, and a program of its own has a
boundary: what it computes is stored, handed back, and read again.  A boundary
that exists only because of how the code was written is a cost nobody asked for,
so the chosen subgraph is instead taken apart and its operations become
operations of the surrounding region, which can then be fused with what is around
them as if they had been written there.

Two things live here.  The inliner is that taking-apart.  The two classes around
it are the bookkeeping a choice needs while it is being measured: a caller that
describes one way of computing something, and a template that turns a set of
those into the candidates a measurement chooses between.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import logging
from collections.abc import Callable
from typing import Any, Protocol

import sympy

import tensorplay as tp

from .. import config
from .common import KernelTemplate
from ..ir import (
    Buffer,
    FixedLayout,
    get_free_symbols,
    get_symbolic_inputs,
    gm_original_output_strides,
    ir_node_to_tensor,
    Layout,
    TensorBox,
)
from ..loops import V
from ..runtime.benchmarking import benchmarker
from ..utils import counters, do_bench_using_profiling, is_collective_op

log = logging.getLogger(__name__)


def inline_subgraph_to_ir_nodes(gm, inputs, name):
    """The value a subgraph yields, with its operations lowered into this region.

    The subgraph is walked against the region it came from rather than against a
    region of its own, so each operation it contains becomes an operation of that
    region.  The region's own module is stood in for the duration, because an
    operation is lowered by looking at what the module it belongs to declares,
    and an operation of the subgraph must be lowered as one of the module that
    holds it.

    Returns a value standing for the subgraph's last operation, which is the
    value the subgraph produces.
    """

    original_module = V.graph.module
    try:
        V.graph.module = gm
        return V.graph.process_subgraph_nodes(gm, inputs)
    finally:
        V.graph.module = original_module


class _AutotuneChoice(Protocol):
    """The winning choice, once it has been measured and picked."""

    def output_node(self) -> TensorBox:
        """The value this choice produces, as a node of the region."""

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long this choice takes on the values it was handed."""

    def hash_key(self) -> str:
        """What identifies this choice, for the log and for the cache."""


#: The class every built choice answers through.  It lives with the values it
#: describes rather than with the templates that build them, so it is imported
#: by name: what a choice is does not depend on which template made it, and a
#: class reachable by two routes would be two classes to one kind.
from ..ir import ChoiceCaller

class SubgraphChoiceCaller(ChoiceCaller):
    """One candidate that is a whole program, measured as one thing.

    A candidate here is not a kernel but a graph: anything a program can be
    written as, which is what makes it the right shape for a choice between ways
    of computing something that no single kernel covers.  The candidate is built
    before it is measured, because building it is most of what it costs and the
    measurement is meant to be of running it.
    """

    def __init__(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        description: str,
        make_graph: Callable[..., Any],
        input_gen_fns: dict[int, Callable[[Any], Any]] | None = None,
    ) -> None:
        super().__init__(name, input_nodes, layout, description)
        self._benchmark_with_cudagraphs = False
        self.make_graph = make_graph
        self.input_gen_fns = input_gen_fns

        # Two sets of values, and the difference matters.  Tracing is done
        # against values whose sizes are symbols, so the program that comes out
        # works for any size and captures anything that depends on the shape
        # rather than on the shape it happened to be traced at.  Measuring is
        # done against values that are really the size asked for, because a
        # measurement is of one size or it is of nothing.
        trace_inputs = []
        self.benchmark_inputs = []
        with V.fake_mode:
            for position, inp in enumerate(self.input_nodes):
                if len(get_free_symbols(inp.get_size(), unbacked_only=True)) != 0:
                    raise AssertionError("expected no unbacked symbols in input size")
                if len(get_free_symbols(inp.get_stride(), unbacked_only=True)) != 0:
                    raise AssertionError("expected no unbacked symbols in input stride")

                inp.data.freeze_layout()
                trace_inputs.append(ir_node_to_tensor(inp))

                if input_gen_fns is not None and position in input_gen_fns:
                    self.benchmark_inputs.append(input_gen_fns[position](inp))
                else:
                    self.benchmark_inputs.append(
                        ir_node_to_tensor(inp, replace_symbols_with_hints=True)
                    )

        self.gm = make_graph(*trace_inputs)
        gm_original_output_strides(self.gm)
        self.example_inputs = trace_inputs

        self.sym_inputs = get_symbolic_inputs(self.input_nodes)
        self.sym_input_values = self._compute_sym_input_values()

        # What this candidate computes, and the arguments it is computed with,
        # kept for a dispatch that chose by shape rather than by measurement.
        self.decomposition: Callable[..., Any] | None = None
        self.decomposition_kwargs: dict[str, Any] = {}
        # Code-generation settings this choice is chosen under.
        self.config_patches: dict[str, Any] = {}
        self._compiled_module: Any = None
        self._bmreq: Any = None

        # Built here rather than on first use, because building needs the region
        # it is being built against, and by the time a measurement asked for it
        # that region may have moved on.
        if config.pipeline_max_autotune_gemm:
            with V.fake_mode:
                self._compiled_module = self._compile_for_benchmarking()
                self._bmreq = self._create_benchmark_request()

    def _compute_sym_input_values(self) -> list[int]:
        """The value each of this candidate's symbolic sizes has, as a number.

        A built candidate is entered with its symbolic sizes as leading numbers
        and its values after them, so the sizes have to be numbers by the time it
        is measured.  Each is taken from the value that stands for it, and where
        there is no such value -- a size that was never pinned -- the size
        environment's own best guess is what a caller would have used, so that is
        what is used here.
        """

        names = {str(s.name) for s in self.sym_inputs if hasattr(s, "name")}
        by_name: dict[str, int] = {}
        for node, value in zip(self.input_nodes, self.benchmark_inputs):
            if isinstance(value, tp.Tensor):
                for symbol_dim, real_dim in zip(node.get_size(), value.shape):
                    if isinstance(symbol_dim, sympy.Symbol):
                        by_name[symbol_dim.name] = int(real_dim)
                    elif str(symbol_dim) in names:
                        by_name[str(symbol_dim)] = int(real_dim)

        result = []
        for symbol in self.sym_inputs:
            if isinstance(symbol, sympy.Symbol) and symbol.name in by_name:
                result.append(by_name[symbol.name])
            else:
                hint = V.graph.sizevars.shape_env.optimization_hint(symbol, fallback=1)
                result.append(int(hint))
        return result

    def cache_decomposition(
        self, decomposition: Callable[..., Any], kwargs: dict[str, Any]
    ) -> None:
        """What this candidate computes, for a dispatch that chose by shape.

        Kept rather than traced, because a dispatch that chose by shape already
        knows what it chose and is asking what that means, not asking again.
        """

        self.decomposition = decomposition
        self.decomposition_kwargs = kwargs

    def __str__(self) -> str:
        return f"SubgraphCaller({self.name})"

    def _compile_for_benchmarking(self) -> Any:
        """This candidate as something built, ready to be called.

        Built against a region of its own rather than against whichever region
        happens to be current, because what it is being built for is being called
        repeatedly and the region it was chosen in will have moved on by the
        second call.
        """

        from ..graph_lowering import GraphLowering
        from ..loops import set_graph

        # Sizes that are symbols are passed as symbols, so the built form
        # resolves them when it runs; sizes that are not are passed as the values
        # themselves, so the built form is specialised to what is measured.
        compile_inputs = self.example_inputs if self.sym_inputs else self.benchmark_inputs
        log.debug("Benchmark compile %s: sym_inputs=%s", self.name, self.sym_inputs)

        if self.gm is None:
            raise AssertionError("expected self.gm to be set")
        region = GraphLowering(
            gm=self.gm,
            example_inputs=compile_inputs,
            shape_env=V.graph.shape_env,
            cpp_wrapper=V.graph.cpp_wrapper,
            aot_mode=V.graph.aot_mode,
            extern_node_serializer=V.graph.extern_node_serializer,
            is_inference=V.graph.is_inference,
            is_backward=V.graph.is_backward,
            name=f"benchmark_{self.name.replace('::', '_').replace('.', '_')}",
        )
        for symbol in self.sym_inputs:
            region.graph_inputs[symbol.name] = symbol
            region.graph_input_names.append(str(symbol))

        with set_graph(region):
            # Choosing under the settings this choice declared, and with
            # measuring switched off, because a candidate that measured its own
            # candidates while being measured would be measuring the measuring.
            settings: dict[str, Any] = {
                "max_autotune": False,
                "max_autotune_gemm": False,
                "benchmark_fusion": False,
                "pipeline_max_autotune_gemm": False,
                **self.config_patches,
            }
            saved = {k: getattr(config, k) for k in settings if hasattr(config, k)}
            for key, value in settings.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            try:
                region.run(*compile_inputs)
            finally:
                for key, value in saved.items():
                    setattr(config, key, value)
            return region.compile_to_module()

    def _create_benchmark_request(self) -> Any:
        """A request that measures this candidate somewhere else.

        Carries where it was written and the key it was written under rather
        than the program itself, because what travels to another process has to
        survive being written down, and a built file does while a live object
        does not.
        """

        from ..autotune_process import (
            SubgraphCPUBenchmarkRequest,
            SubgraphGPUBenchmarkRequest,
            TensorMeta,
        )

        if self._compiled_module is None:
            raise AssertionError(
                "Module must be compiled before creating benchmark request"
            )
        input_meta = TensorMeta.from_irnodes(self.input_nodes)
        output_meta = TensorMeta.from_irnodes(self.layout)

        cls = (
            SubgraphCPUBenchmarkRequest
            if self.layout.device.type == "cpu"
            else SubgraphGPUBenchmarkRequest
        )
        return cls(
            kernel_name=self.name,
            input_tensor_meta=input_meta,
            output_tensor_meta=output_meta,
            extra_args=tuple(),
            module_path=self._compiled_module.__file__,
            module_cache_key=self._compiled_module.key,
            sym_input_values=self.sym_input_values,
        )

    @property
    def bmreq(self) -> Any:
        """The request that measures this candidate elsewhere.

        Only there when measuring was asked to happen elsewhere from the start,
        because building the request needs a built candidate and building one
        eagerly for a candidate that will be measured here would be building it
        twice.
        """

        if self._bmreq is None:
            raise AssertionError(
                "bmreq accessed but pipeline_max_autotune_gemm was not enabled "
                "during __init__"
            )
        return self._bmreq

    def _ensure_compiled(self) -> None:
        """Build this candidate if it has not been built yet.

        For a candidate that is measured here rather than elsewhere, where the
        building is done on first use so that a candidate that is never measured
        is never built.
        """

        if self._compiled_module is None:
            self._compiled_module = self._compile_for_benchmarking()

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long one run of this candidate takes.

        Measured on the device rather than around the call, because a call that
        waits is mostly waiting and a candidate that waits is not slow, it is
        patient.  The sizes it is entered with come first, because that is the
        order a built candidate is called in.
        """

        self._ensure_compiled()
        run = self._compiled_module.call
        sizes = self.sym_input_values

        def fn() -> Any:
            return run([*sizes, *args])

        if self._benchmark_with_cudagraphs:
            return benchmarker.benchmark_gpu_with_cuda_graph(fn)

        if config.profile_bandwidth_with_do_bench_using_profiling:
            return do_bench_using_profiling(fn)
        return benchmarker.benchmark(
            fn, device=benchmarker.infer_device(*sizes, *args)
        )

    def benchmark_collective(self, *args: Any, out: Any = None) -> None:
        """Run once, for a measurement several candidates take part in.

        No time is taken: what is being measured is the one run they all have to
        make, and a candidate that ran it alone would have run it alone.
        """

        self._ensure_compiled()
        self._compiled_module.call([*self.sym_input_values, *args])

    def hash_key(self) -> str:
        """What identifies this candidate: its name, its operands, and its text.

        The operands' extents and strides are in the key because the same program
        traced against different operands is a different kernel, and the program's
        own text is in it because that is what decides what the kernel does.  A key
        of the name alone would collide with every other candidate of that name.
        """

        if self.gm is None:
            raise AssertionError("expected self.gm to be set")
        return "-".join(
            [
                self.name.rsplit("_", 1)[0],
                *[str(inp.get_size()) for inp in self.input_nodes],
                *[str(inp.get_stride()) for inp in self.input_nodes],
                str(self.gm.graph),
            ]
        )

    def output_node(self) -> TensorBox:
        """The value this candidate produces, as a node of the region.

        Wrapped rather than inlined, because a choice is asked for its value
        before it is asked whether it was the fastest, and a candidate that
        inlined itself on being asked what it produces would be committed before
        it was chosen.
        """

        from ..ir import SubgraphBuffer

        if self.gm is None:
            raise AssertionError("expected self.gm to be set")
        return TensorBox.create(
            SubgraphBuffer(
                layout=self.layout,
                input_nodes=self.input_nodes,
                gm=self.gm,
                example_inputs=self.example_inputs,
                subgraph_name=self.name,
                config_patches=self.config_patches if self.config_patches else None,
            )
        )

    def info_dict(self) -> dict[str, Any]:
        """What is written about this candidate when measuring is logged."""

        return {
            "backend": "subgraph",
            "kernel_name": self.name,
        }

    def autoheuristic_id(self) -> str:
        return f"subgraph_{self.name}"

    # -- what the region's own bookkeeping asks of any choice -----------------

    def to_callable(self) -> Callable[..., Any]:
        """This candidate as something to call.

        Built on first use, because a candidate that is only ever compared by its
        key never needs to exist as something callable.
        """

        self._ensure_compiled()
        run = self._compiled_module.call
        sizes = self.sym_input_values
        return lambda *args: run([*sizes, *args])


class SubgraphTemplate(KernelTemplate):
    """A way of turning a computation into the candidates to choose between.

    A template is asked for candidates rather than being one, so that a
    computation with several ways of being done can offer all of them and a
    measurement can say which it preferred.
    """

    index_counter = itertools.count()

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def generate(
        self,
        name: str,
        input_nodes: list[Buffer],
        layout: Layout,
        make_graph: Callable[..., Any],
        description: str = "",
        input_gen_fns: dict[int, Callable[[Any], Any]] | None = None,
        **kwargs: Any,
    ) -> SubgraphChoiceCaller:
        """One candidate, named so that two of the same shape are told apart."""

        return SubgraphChoiceCaller(
            name=f"{name}_{next(SubgraphTemplate.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            description=description,
            make_graph=make_graph,
            input_gen_fns=input_gen_fns,
        )

    def generate_custom_op_choices(
        self,
        name: str,
        decompositions: list[Callable[..., Any]],
        input_nodes: list[Buffer],
        non_tensor_args: list[dict[str, Any]],
        default_impl: Callable[..., Any] | None = None,
        input_gen_fns: dict[int, Callable[[Any], Any]] | None = None,
        config_patches_list: list[dict[str, Any]] | None = None,
    ) -> list[SubgraphChoiceCaller]:
        """One candidate per way of computing the same thing.

        Every candidate here produces the same value, which is checked rather
        than assumed: a measurement that compared candidates producing different
        values would be comparing them on being different rather than on being
        faster, and the faster one would win for being wrong.
        """

        if not decompositions:
            return []

        if len(decompositions) != len(non_tensor_args):
            raise AssertionError(
                f"decompositions and non_tensor_args must have same length, "
                f"got {len(decompositions)} decompositions and "
                f"{len(non_tensor_args)} kwargs"
            )

        if config_patches_list is None:
            config_patches_list = [{} for _ in decompositions]

        layouts = [
            self._infer_custom_op_layout(
                input_nodes, decomp, kwargs, default_impl, input_gen_fns
            )
            for decomp, kwargs in zip(decompositions, non_tensor_args)
        ]
        self._validate_layout_equivalence(name, decompositions, layouts)
        layout = layouts[0]

        choices: list[SubgraphChoiceCaller] = []
        for decomp, decomp_kwargs, config_patches in zip(
            decompositions, non_tensor_args, config_patches_list
        ):
            def make_graph(
                *args: Any,
                decomp: Callable[..., Any] = decomp,
                decomp_kwargs: dict[str, Any] = decomp_kwargs,
            ) -> Any:
                from ..op_lowerings import select_decomp_table
                from ....graph.experimental.proxy_tensor import make_graph as make_fx

                decomposition_table = select_decomp_table()
                shape_env = V.fake_mode.shape_env
                # An implementation that decides the shape while it is traced
                # decides it for every shape, which is not what was asked for; so
                # a new guard while tracing is an error rather than a fact to
                # record, and the implementation is dropped.
                guard_ctx = (
                    shape_env.error_on_new_guards()
                    if shape_env is not None
                    else contextlib.nullcontext()
                )
                with guard_ctx:
                    return make_fx(
                        functools.partial(decomp, **decomp_kwargs),
                        decomposition_table=decomposition_table,
                        tracing_mode="real",
                    )(*args)

            variant_name = self._generate_variant_name(decomp, decomp_kwargs)

            try:
                choice = self.generate(
                    name=f"{name}_{variant_name}",
                    input_nodes=input_nodes,
                    layout=layout,
                    make_graph=make_graph,
                    description=f"CustomOp {decomp.__name__}",
                    input_gen_fns=input_gen_fns,
                )
            except AssertionError as exc:
                if "guard" not in str(exc).lower():
                    raise
                log.info(
                    "Skipping decomposition %s: adds guards during tracing",
                    decomp.__name__,
                )
                counters["tp"]["custom_op_decomp_guard_skips"] += 1
                continue

            choice.cache_decomposition(decomp, decomp_kwargs)
            choice.config_patches = config_patches
            choices.append(choice)

        return choices

    def _generate_variant_name(
        self, decomp: Callable[..., Any], kwargs: dict[str, Any]
    ) -> str:
        """A name that says which way of computing this is, and with what.

        The arguments are in the name because two ways of computing the same
        thing with different arguments are different candidates, and a log that
        could not tell them apart would be a log of which kind won and not which.
        """

        import re

        base_name = decomp.__name__
        if not kwargs:
            return base_name

        def spell(value: Any) -> str:
            text = re.sub(r"[^a-zA-Z0-9_]", "_", str(value))
            if text and text[0].isdigit():
                text = "_" + text
            return text

        suffix = "_".join(f"{k}_{spell(v)}" for k, v in sorted(kwargs.items()))
        return f"{base_name}_{suffix}"

    def _validate_non_tensor_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Refuse arguments that are values, which belong with the inputs.

        A value among the arguments is a value the candidate reads but which is
        not one of the inputs, so it would be neither measured nor described: the
        candidate's own description would not mention it and the measurement would
        not vary it, so two candidates differing only in it would be measured as
        the same candidate.
        """

        for key, value in kwargs.items():
            if isinstance(value, (tp.Tensor, Buffer)):
                raise AssertionError(
                    f"kwargs['{key}'] contains tensor {type(value)}. "
                    f"Tensor arguments should be in input_nodes, not kwargs. "
                    f"Only scalar/non-tensor parameters should be in kwargs."
                )

    def _validate_layout_equivalence(
        self,
        op_name: str,
        decompositions: list[Callable[..., Any]],
        layouts: list[Layout],
    ) -> None:
        """Refuse candidates whose results are not the same shape of thing.

        A measurement compares times, and a time is only comparable between
        candidates that produce the same value: two that do not are being
        compared on what they produce, and the one that produces less will be
        faster for producing less.
        """

        if not layouts:
            return

        reference = layouts[0]
        for position, layout in enumerate(layouts[1:], start=1):
            if (layout.device, layout.dtype, layout.size, layout.stride) != (
                reference.device,
                reference.dtype,
                reference.size,
                reference.stride,
            ):
                raise AssertionError(
                    f"Layout mismatch in custom op '{op_name}': "
                    f"decomposition '{decompositions[position].__name__}' produces "
                    f"({layout.device}, {layout.dtype}, {layout.size}, "
                    f"{layout.stride}) but '{decompositions[0].__name__}' produces "
                    f"({reference.device}, {reference.dtype}, {reference.size}, "
                    f"{reference.stride})"
                )

    def _infer_custom_op_layout(
        self,
        input_nodes: list[Buffer],
        function_decomposition: Callable[..., Any],
        kwargs: dict[str, Any],
        default_impl: Callable[..., Any] | None = None,
        input_gen_fns: dict[int, Callable[[Any], Any]] | None = None,
    ) -> Layout:
        """Where the result of one way of computing this would sit.

        Worked out by running the way on values with the shapes it will be given,
        because the result's shape is whatever that computation produces and is
        not derivable from the inputs: a sum over one axis and a sum over all of
        them read the same inputs and differ in extent.
        """

        self._validate_non_tensor_kwargs(kwargs)

        with V.fake_mode:
            example_inputs = []
            for position, inp in enumerate(input_nodes):
                if input_gen_fns and position in input_gen_fns:
                    value = input_gen_fns[position](inp)
                else:
                    sizevars = V.graph.sizevars
                    value = tp.empty_strided(
                        sizevars.optimization_hints(inp.get_size()),
                        sizevars.optimization_hints(inp.get_stride()),
                        dtype=inp.get_dtype(),
                        device=inp.get_device(),
                    )
                example_inputs.append(value)

            fn = functools.partial(function_decomposition, **kwargs)
            output = fn(*example_inputs)

            if not isinstance(output, tp.Tensor):
                raise AssertionError(
                    f"Expected single tensor output, got {type(output)}. "
                    f"Multi-output custom ops not yet supported in autotuning."
                )

            return FixedLayout(
                device=output.device,
                dtype=output.dtype,
                size=tuple(int(s) for s in output.shape),
                stride=tuple(int(s) for s in output.stride()),
            )


#: The template custom operations offer their ways of computing to, which is
#: here rather than at each use because a candidate is only comparable with a
#: candidate built the same way.
subgraph_template = SubgraphTemplate("subgraph")


def _detect_collective_ops(choices) -> bool:
    """Whether any of these candidates is a collective, and so not measurable.

    A collective's result is the same on every device in a group, so a time taken
    for one of them is a time for the group rather than for the candidate, and
    two candidates would be compared on which group they were in.
    """

    for choice in choices:
        gm = getattr(choice, "gm", None)
        if gm is None:
            continue
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue
            target = getattr(node.target, "__name__", None)
            if target and is_collective_op(target):
                return True
    return False



__all__ = [
    "SubgraphChoiceCaller",
    "SubgraphTemplate",
    "inline_subgraph_to_ir_nodes",
    "subgraph_template",
]
