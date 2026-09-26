"""A template whose kernel is a region of the graph, emitted as one.

Some operations are not worth writing by hand.  Fusing a run of elementwise
operations into a single pass is not a matter of expressing the run as a
kernel: it is a matter of capturing the run and letting the loop code do what
it already does with a region.  So this template's one decision is whether the
region is worth keeping as a unit, and its kernel is the region itself.

The region is *traced* rather than handed over, and the two sets of inputs it
is traced and measured with are deliberately different.  Traced with symbolic
extents, so that a region which reads its own input's shape -- multiplying by
a batch, reshaping to a multiple of it -- is captured as depending on the shape
rather than on the one value that happened to be there; measured with concrete
operands, because a measurement of a general graph tells you nothing about a
particular one.  A region traced on the call's own tensors would pin all of them
for the life of the program that holds the choice.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

from ..codegen.common import KernelTemplate
from ..ir import Layout
from .select_algorithm import ChoiceCaller

__all__ = ["SubgraphChoiceCaller", "SubgraphTemplate", "subgraph_template"]


def _shapes_of(example_inputs: tuple) -> tuple:
    """The extents of a call's own inputs, for asking a dispatch about it.

    A dispatch is asked which of its implementations are valid for a *shape*, and
    the shape it is asked about is the one the call will really have.  A call
    that brought no example has no shape to be asked about, which is reported as
    an empty one rather than as a guess: an empty shape is covered by a range
    that covers everything, and nothing else is.
    """

    return tuple(tuple(int(v) for v in getattr(t, "shape", ()) or ())
                 for t in (example_inputs or ()))


class SubgraphChoiceCaller(ChoiceCaller):
    """A choice whose kernel is a whole region, kept whole.

    The region travels with the choice because a template that fuses several
    operations has to carry them: the graph *is* what was chosen, so dropping it
    would leave a caller holding a name and nothing to run.  So the caller owns
    the region, the two sets of inputs it was built from, and the compiled form
    once there is one.
    """

    def __init__(
        self,
        name: str,
        input_nodes: tuple = (),
        layout: Layout | None = None,
        description: str = "",
        make_graph: Callable[..., Any] | None = None,
        example_inputs: tuple = (),
        benchmark_inputs: tuple = (),
    ):
        super().__init__(name, input_nodes, layout, description)
        self.make_graph = make_graph
        self.example_inputs = tuple(example_inputs)
        self.benchmark_inputs = tuple(benchmark_inputs)
        #: The traced region, once it has been traced.
        self.gm = None
        #: What the region was decomposed into, when it was a decomposition.
        self.decomposition: Callable[..., Any] | None = None
        self.decomposition_kwargs: dict[str, Any] = {}
        #: Geometry substitutions a measurement decided on, applied when the
        #: region is finally emitted.  They are recorded here rather than baked
        #: in, because a substitution is a statement about this measurement and
        #: not about the region.
        self.config_patches: dict[str, Any] = {}
        self._launch = None
        #: The symbolic inputs the region was traced against, which is
        #: what a run is given alongside the concrete operands.
        #: The traced region was built against these, and a run is given
        #: them ahead of the concrete operands.  The two are kept apart
        #: because they answer different questions: one is what the region
        #: was written for, the other is what this call has to hand it.
        self.sym_inputs = ()
        self.sym_input_values = []
        #: The region compiled once and measured many times, and the request
        #: that measurement is asked through.  Both are made on first use.
        self._compiled_module = None
        self._bmreq = None


    def cache_decomposition(self, decomposition: Callable[..., Any], kwargs: dict) -> None:
        """Keep the decomposition that applies to a range, for a later lookup.

        A region that fuses several operations has more than one way to be
        written, and which one applies is a property of the range rather than
        of the region.  The region travels with the choice; the decomposition
        that goes with a range is looked up by whoever is dispatching on that
        range, and this is where the answer is put.
        """

        self.decomposition = decomposition
        self.decomposition_kwargs = dict(kwargs or {})

    def _ensure_compiled(self) -> None:
        """The region, compiled, compiled at most once.

        Compiling is deferred rather than done at construction because most
        choices are never measured -- a table is offered, one entry wins, and
        the rest are discarded without ever being run.
        """

        if self._compiled_module is None:
            self._compiled_module = self._compile_for_benchmarking()

    def _compile_for_benchmarking(self) -> Any:
        """The traced region, which is what there is to run."""

        self._ensure_traced()
        return self.gm

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long one run of the region takes."""

        self._ensure_compiled()
        return super().benchmark(*args, out=out)

    def benchmark_collective(self, *args: Any, out: Any = None) -> None:
        """Run the region once, with the timing left to whoever called.

        A region is measured against the others running beside it, and what
        that costs is a property of the group rather than of any one member.
        So this does not time anything: it runs once, and whoever is measuring
        the group does the timing and the synchronising, because only they know
        when the group as a whole has finished.
        """

        self._ensure_compiled()
        self.gm.call([*self.sym_input_values, *args]) if hasattr(self.gm, "call") else None

    def bmreq(self) -> Any:
        """The request that measures this region for an asynchronous tuner.

        A request can only be made from a region that is already compiled,
        because the request names the compiled artefact rather than the
        source: a tuner that has the request has something to point at.  So
        this refuses rather than compiling behind the caller's back -- a
        caller that reaches this without a compiled region has skipped a step
        whose cost it should have agreed to.
        """

        if self._bmreq is None:
            if self._compiled_module is None:
                raise AssertionError(
                    "the region must be compiled before a benchmark request is made"
                )
            self._bmreq = self._create_benchmark_request()
        return self._bmreq

    def _create_benchmark_request(self) -> Any:
        return {
            "kernel_name": self.name,
            "module": self._compiled_module,
            "sym_input_values": self.sym_input_values,
        }

    def hash_key(self) -> str:
        """What identifies this region: its name, its operands, and its text.

        The operands' extents and strides are in the key because the same
        region traced against different operands is a different kernel, and the
        region's own text is in it because that is what decides what the kernel
        does.  A choice that hashed only its name would collide with every
        other region of that name, which is the one thing a cache cannot allow.
        """

        if self.gm is None:
            raise AssertionError("a region has to be traced before it can be hashed")
        return "-".join(
            [self.name.rsplit("_", 1)[0]]
            + [str(tuple(getattr(n, "shape", ()))) for n in self.input_nodes]
            + [str(tuple(getattr(n, "stride", ()))) for n in self.input_nodes]
            + [str(self.gm)]
        )

    def output_node(self) -> Any:
        """The result, as a node the rest of the graph can refer to.

        Built here rather than returned from a run: the caller wants something
        to substitute into the graph, and a measurement's temporary is not that
        -- it has no layout the graph can rely on and no name.
        """

        if self.gm is None:
            raise AssertionError("a region has to be traced before it has a result")
        return self.gm

    def info_dict(self) -> dict[str, Any]:
        return {"backend": "subgraph", "kernel_name": self.name}

    def trace(self) -> Any:
        """The region, built once from the operands it was traced against.

        Built on the first ask rather than on construction, because a choice
        that is never measured should not cost a trace: the enumeration builds
        every configuration, and most of them are only ever going to be
        measured against each other.
        """

        if self.gm is None:
            if self.make_graph is None:
                raise NotImplementedError(f"{self.name} has no region to build")
            self.gm = self.make_graph(*self.example_inputs)
        return self.gm

    def bind(self, launcher: Callable[..., Any]) -> "SubgraphChoiceCaller":
        self._launch = launcher
        return self

    def to_callable(self) -> Callable[..., Any]:
        if self._launch is None:
            self.trace()
            raise NotImplementedError(
                f"{self.name} has a region but no kernel to run it on"
            )
        return self._launch

    def hash_key(self) -> str:
        parts = [self.name, self.description]
        if self.layout is not None:
            parts.append(repr(self.layout))
        parts.append(str(sorted(self.config_patches.items(), key=lambda kv: kv[0])))
        return ":".join(parts)

    def info_dict(self) -> dict[str, Any]:
        info = super().info_dict()
        info["traced"] = self.gm is not None
        if self.decomposition is not None:
            info["decomposition"] = getattr(
                self.decomposition, "__name__", repr(self.decomposition)
            )
        return info

    def autoheuristic_id(self) -> str:
        return "subgraph"


class SubgraphTemplate(KernelTemplate):
    """A region offered to the same measurement as a kernel.

    The template does not know what is in the region: it is handed something
    that can build one, and its decision is whether to offer it.  Each offer is
    numbered, because two offers of the same region under the same name would
    be indistinguishable in a stored decision, and a measurement that cannot
    tell its candidates apart cannot choose between them.
    """

    #: Hands out the number that keeps two offers of one name apart.
    index_counter = itertools.count()

    def generate(
        self,
        name: str,
        input_nodes: tuple = (),
        layout: Layout | None = None,
        make_graph: Callable[..., Any] | None = None,
        description: str = "",
        example_inputs: tuple = (),
        benchmark_inputs: tuple = (),
        **kwargs: Any,
    ) -> SubgraphChoiceCaller:
        return SubgraphChoiceCaller(
            name=f"{name}_{next(SubgraphTemplate.index_counter)}",
            input_nodes=input_nodes,
            layout=layout,
            description=description,
            make_graph=make_graph,
            example_inputs=example_inputs,
            benchmark_inputs=benchmark_inputs,
        )

    def generate_custom_op_choices(
        self,
        name: str,
        decompositions: list,
        input_nodes: tuple = (),
        layout: Layout | None = None,
        default_impl: Callable[..., Any] | None = None,
        example_inputs: tuple = (),
        benchmark_inputs: tuple = (),
        config_patches_list: list | None = None,
        dispatch: Any = None,
    ) -> list:
        """One choice per way of doing the operation, plus the default.

        A custom operation arrives as a name and a set of arguments, and what
        it should compute is not written down anywhere.  So the ways of doing
        it are offered side by side -- each decomposition as its own choice,
        with its own geometry substitutions, because a substitution measured for
        one of them says nothing about the others -- and the default among them
        for the case where none of them is worth anything.

        A ``dispatch`` says which of them are valid for a call of a given shape.
        Where one is given, a decomposition that is not valid there is not
        offered at all: a configuration measured where it does not hold is not a
        slower answer, it is not an answer, and offering it would have the
        measurement report a winner that cannot run.
        """

        from .custom_op import ImplConfig

        patches = config_patches_list or [{} for _ in decompositions]
        pairs = list(zip(decompositions, patches))
        if dispatch is not None:
            offered = {
                id(impl.impl): config
                for impl, config in dispatch.candidates(_shapes_of(example_inputs))
            }
            kept = []
            for decomposition, patch in pairs:
                config = offered.get(id(decomposition))
                if config is None:
                    continue
                kept.append((decomposition, patch, config))
            pairs = [(decomposition, patch) for decomposition, patch, _ in kept]
            default_range = kept
        else:
            default_range = None

        choices = []
        for decomposition, patch in pairs:
            caller = self.generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_graph=decomposition,
                description=getattr(decomposition, "__name__", repr(decomposition)),
                example_inputs=example_inputs,
                benchmark_inputs=benchmark_inputs,
            )
            caller.decomposition = decomposition
            caller.config_patches.update(patch)
            choices.append(caller)
        if default_impl is not None and (dispatch is None or default_range is not None):
            caller = self.generate(
                name=name,
                input_nodes=input_nodes,
                layout=layout,
                make_graph=default_impl,
                description="the default way of doing it",
                example_inputs=example_inputs,
                benchmark_inputs=benchmark_inputs,
            )
            choices.append(caller)
        return choices


#: The template regions are offered through, under a name of its own.
subgraph_template = SubgraphTemplate("subgraph")
