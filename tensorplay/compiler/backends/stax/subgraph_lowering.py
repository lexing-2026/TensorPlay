"""Taking a subgraph apart when every value it produces is one value.

Some subgraphs are not arbitrary programs but the arithmetic of a single value:
what they compute depends on one element of each input and nothing else, so
there is no loop over anything and no reduction to schedule.  Lowering such a
subgraph is therefore not compiling a program but writing one expression, and
the two are told apart here rather than by a caller saying so.

The reason it is worth telling apart is that a pointwise subgraph can be
lowered into the region that chose it as one body, which then fuses with
whatever surrounds it.  A subgraph that was not pointwise would have to become a
region of its own, and the boundary is exactly the cost this avoids.
"""

from __future__ import annotations

import functools
import operator
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast, TypeVar

import sympy

import tensorplay as tp

from .graph_lowering import GraphLowering
from .ir import Expr, Pointwise, StorageBox, Subgraph, TensorBox
from .loops import ops, V
from .ops_handler import OpsHandler, SimpleCSEHandler, WrapperHandler

T = TypeVar("T")

OpOverload = Any
LoweringDict = dict[Any, Callable[..., Any]]
TargetType = Callable[..., Any] | str

SubgraphInput = "InputDescriptor | int | sympy.Basic"


class SubgraphLoweringException(RuntimeError):
    """A subgraph could not be lowered as one expression.

    Its own kind rather than a general failure because the ways it can happen are
    worth telling apart when reading a failure: a subgraph that turned out not to
    be pointwise, an operation with no lowering, and a body that tried to write
    memory are three different problems that a caller may be able to act on
    differently.
    """


@dataclass
class InputDescriptor:
    """An input described rather than given, because it has no value yet.

    A subgraph is lowered before the values it will be called with exist, so what
    it is handed is a description of each input: enough to lower against, and
    nothing that would tie the lowered body to one particular call.
    """

    dtype: tp.dtype
    device: tp.device


class PointwiseSubgraphLowering:
    """Lowers a pointwise subgraph into the region that chose it.

    A region of its own would be able to hold anything; this can hold only what
    touches no memory it did not read, which is what makes what it produces
    fusable with what is around it.  So a body that writes, or that creates a
    buffer, is refused rather than quietly given somewhere to write: the refusal
    is the check that the subgraph really was pointwise.
    """

    graph_outputs: list | None
    root_graph: GraphLowering
    _current_op: TargetType | None
    # A scatter's backward writes to a buffer, so a subgraph that scatters has a
    # write in it and is not pointwise after all; such a write is allowed only
    # when the operation it came from is one that was expected to write.
    allowed_mutations: set | None
    additional_lowerings: LoweringDict | None
    buffers: list
    mutated_buffers: set

    def __init__(
        self,
        gm,
        root_graph_lowering: GraphLowering,
        allowed_mutations: set | None = None,
        additional_lowerings: LoweringDict | None = None,
    ) -> None:
        self.gm = gm
        self.graph_outputs = None
        self.root_graph = root_graph_lowering
        self.allowed_mutations = allowed_mutations
        self.additional_lowerings = additional_lowerings
        self._current_op = None

        self.mutated_buffers = set()
        self.buffers = []

    @contextmanager
    def _op_context(self, op: TargetType) -> Generator[None, None, None]:
        """Which operation is being lowered right now.

        Kept for the duration of that operation's lowering because whether a write
        is allowed depends on which operation is doing it, and lowering an
        operation is the only moment at which that is known.
        """

        previous = self._current_op
        self._current_op = op
        try:
            yield
        finally:
            self._current_op = previous

    def _approved_mutator(self) -> bool:
        return (
            self.allowed_mutations is not None
            and self._current_op in self.allowed_mutations
        )

    def mark_buffer_mutated(self, name: str) -> None:
        """Record that a buffer was written, or refuse.

        Refused unless the operation doing it was expected to: a pointwise body
        that writes is not pointwise, and letting the write through would produce
        a body whose result depends on an order nothing stated.
        """

        if self._approved_mutator():
            self.mutated_buffers.add(name)
        else:
            raise SubgraphLoweringException(
                f"Buffer mutation detected during lowering of {self._current_op}. "
                "Buffer mutations are only allowed in approved mutation ops. "
                "This is an error in the lowering of the subgraph, please file a "
                "bug report."
            )

    def register_buffer(self, buffer, *, set_name: bool = False) -> str:
        """Give a buffer a name in the region, or refuse.

        Refused, because a buffer made here would be memory the region has to
        account for and the pointwise body has no way to say when it is read; so
        a subgraph that needs one was not pointwise after all.
        """

        if self._approved_mutator():
            return self.root_graph.register_buffer(buffer, set_name=set_name)
        raise SubgraphLoweringException(
            "Buffers cannot be created while lowering a pointwise subgraph. "
            "This could be for a good reason (e.g. you're calling an op we can't "
            "codegen as a pointwise op), but it could also be a bug. Please file a "
            "bug report if you think this should be supportable."
        )

    def __getattr__(self, name: str) -> Any:
        """Anything not answered here is the region's own answer.

        The lowering is a region as far as anything the lowerings ask of a region
        is concerned, so that what a lowering can do here is what it can do
        there, rather than a subset decided by which class is holding it.
        """

        return getattr(self.root_graph, name)

    def call_function(self, target: TargetType, args: Any, kwargs: dict[str, Any]) -> Any:
        """Lower one call, and hand back what it stands for."""

        from .op_lowerings import LOWERINGS

        with self._op_context(target):
            # Indexing a result that was already produced is addressing, not
            # calling, so it is resolved rather than looked for a lowering.
            if target is operator.getitem and isinstance(args[0], (list, tuple, dict)):
                return args[0][args[1]]

            # A lowering the caller supplied for this operation itself comes
            # first: it was supplied for this subgraph, and the built-in one
            # describes a different operation that goes by the same name.
            if self.additional_lowerings is not None and target in self.additional_lowerings:
                return self.additional_lowerings[target](*args, **kwargs)

            # The table is written in terms of the name an operation goes by,
            # and the thing a node holds is the operation itself.  The two are
            # the same operation and they are not the same object: the name is
            # what the table can be written and read in, and a lookup that used
            # the object would find nothing for every operation in it.
            from .op_lowerings import target_name

            name = target_name(target)
            if name not in LOWERINGS:
                raise SubgraphLoweringException(
                    f"{name} not supported in subgraph, (missing lowering)"
                )
            return LOWERINGS[name](*args, **kwargs)

    def run(self, *args: Any) -> None:
        """Walk the subgraph, lowering each of its calls as it is reached."""

        from ....graph import Interpreter

        # The two kinds a lowering answers for: the calls it lowers, and the
        # output it keeps.  The output is included because a body that lowers
        # its calls still has to say what it produced -- a walk that hands the
        # output back to the graph instead would leave the body with nothing
        # recorded, and the buffers it made would be unreferenced.
        interpreter = Interpreter(self.gm, handler=self, handler_ops=("call_function", "output"))
        interpreter.run(*args)

    def output(self, target: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Record what the subgraph yields.

        One value, because a pointwise subgraph yields one value: several would
        be several values and the fusing this exists for is about one.
        """

        if len(args) != 1:
            raise AssertionError(f"expected exactly one output arg, got {len(args)}")
        self.graph_outputs = args[0]


class TracingOpsHandler(WrapperHandler):
    """Records the operations a body performs as a graph, rather than running them.

    A body that has been lowered says what it computes; this turns that into a
    graph of the same operations, which is what makes the result something that
    can be called with arguments rather than something already computed.  The
    values a body reads become placeholders, one per input, so that the graph
    says which input each read came from.
    """

    def __init__(self, tracer, num_inputs: int) -> None:
        # A node for whatever is asking, so that an operation traced on its own
        # has something to hang from; it is never read.
        parent = tracer.create_proxy("placeholder", "ops", (), {})
        super().__init__(cast(OpsHandler, parent))
        self.tracer = tracer
        self.placeholders = [
            self.tracer.create_proxy("placeholder", f"input{i}", (), {})
            for i in range(num_inputs)
        ]

    def placeholder(self, idx: int):
        """The node standing for the ``idx``-th value the body reads."""

        return self.placeholders[idx]

    def output(self, *args: tuple) -> None:
        """Close the graph with what the body produced."""

        self.tracer.create_node(
            "output",
            "output",
            (tuple(self.tracer.create_arg(a) for a in args),),
            {},
        )


def lower_pointwise_subgraph(
    subgraph: Subgraph, inputs: list
) -> Callable[..., Any]:
    """The one expression a pointwise subgraph is, as something callable.

    The subgraph is lowered twice over, and both are needed.  The first lowering
    is against the region: it answers what the subgraph computes, using the
    region's own lowerings so that the answer is in the region's terms.  The
    second is a trace of that answer with the values replaced by positions, which
    is what makes the result a function of arguments rather than a computation
    already done -- and tracing it is also what collapses the several bodies the
    first lowering produced into one, since tracing asks for each value once.
    """

    # A value that stands for an input, so that the first lowering has something
    # to read without reading anything.
    def fake_inner_fn(loop_idx: int, input_idx: int):
        return ops.placeholder(input_idx)

    graph_inputs = []
    for position, desc in enumerate(inputs):
        if isinstance(desc, InputDescriptor):
            graph_inputs.append(
                Pointwise.create(
                    device=desc.device,
                    dtype=desc.dtype,
                    inner_fn=functools.partial(fake_inner_fn, input_idx=position),
                    ranges=[],
                )
            )
        else:
            graph_inputs.append(desc)

    pw_subgraph = PointwiseSubgraphLowering(
        subgraph.graph_module, root_graph_lowering=V.graph
    )
    with V.set_graph_handler(pw_subgraph):
        pw_subgraph.run(*graph_inputs)

    if pw_subgraph.graph_outputs is None:
        raise AssertionError("expected pw_subgraph.graph_outputs to be set")

    # One body for all of it: traced by asking for each output, with the
    # computation behind each asked for only once.
    from ....graph import Graph, GraphModule, Tracer

    tracer = Tracer()
    tracer.graph = Graph(tracer_cls=tracer.__class__)
    trace_ops = SimpleCSEHandler(TracingOpsHandler(tracer, len(inputs)))
    with V.set_ops_handler(trace_ops):
        output_irs = []
        for out_var in pw_subgraph.graph_outputs:
            if not isinstance(out_var, TensorBox):
                raise AssertionError(type(out_var))
            if out_var.get_size() != []:
                raise AssertionError(
                    f"expected scalar output (empty size), got {out_var.get_size()}"
                )
            if not isinstance(out_var.data, StorageBox):
                raise AssertionError(
                    f"expected out_var.data to be a StorageBox, got "
                    f"{type(out_var.data)}"
                )
            if not isinstance(out_var.data.data, Pointwise):
                raise AssertionError(
                    f"expected out_var.data.data to be a Pointwise, got "
                    f"{type(out_var.data.data)}"
                )
            output_irs.append(out_var.data.data.inner_fn(()))
        ops.output(*output_irs)

    lowered_gm = GraphModule({}, tracer.graph)

    def inner_fn(*args: Any, **kwargs: Any) -> Any:
        return lowered_gm(V.get_ops_handler(), *args, **kwargs)

    return inner_fn


__all__ = [
    "InputDescriptor",
    "PointwiseSubgraphLowering",
    "SubgraphLoweringException",
    "lower_pointwise_subgraph",
]
