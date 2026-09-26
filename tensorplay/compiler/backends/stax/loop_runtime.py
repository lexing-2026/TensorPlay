"""Execution of a scheduled loop-IR program.

The scheduler's order is the execution order: a buffer is allocated just
before the kernel that first writes it, library calls run with realized
inputs, and a buffer is released as soon as its last reader has run.  Graph
outputs are kept alive for the caller.

Extern calls see the same tensors the kernels do, so a strided view built
for a library call addresses the very memory the fused kernels write.
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp

from .ir import Buffer, ComputedBuffer, ConstantBuffer, StorageBox, TensorBox
from .ir import ReinterpretView
from .ir import FallbackKernel as IrFallbackKernel
from .loops import ExternKernel, ExternOutput


class Step:
    """One scheduled item: a fused kernel or a library call."""

    def __init__(self, reads: set, writes: set):
        self.reads = reads
        self.writes = writes
        self.frees: list = []


class FusedStep(Step):
    def __init__(self, launch, ptr_names: set, stored: set, buffers: dict):
        super().__init__(set(ptr_names), set(stored))
        self.launch = launch
        self.ptr_names = list(ptr_names)
        self.buffers = buffers

    def run(self, env: dict) -> None:
        for name in self.writes:
            buffer = self.buffers[name]
            if isinstance(buffer, ConstantBuffer):
                continue
            env[name] = tp.empty(
                buffer.get_size(),
                dtype=_dtype_of(buffer.layout.dtype),
                device=buffer.layout.device,
            )
        self.launch([env[name] for name in self.ptr_names])


class HostStep(Step):
    """One group printed for the host: the runner allocates its own result."""

    def __init__(self, launch, inputs: list[str], output: str):
        super().__init__(set(inputs), {output})
        self.launch = launch
        self.inputs = list(inputs)
        self.output = output

    def run(self, env: dict) -> None:
        env[self.output] = self.launch([env[name] for name in self.inputs])


class ExternStep(Step):
    def __init__(self, kernel):
        if isinstance(kernel, IrFallbackKernel):
            # What a described call reads is what it depends on, and what it
            # produces is either the call's own buffer or the buffers naming
            # where each of several results sits.
            super().__init__(
                {d.name for d in kernel.get_reads()},
                {o.get_name() for o in kernel.get_outputs()},
            )
        else:
            super().__init__(
                {b.name for b in kernel.input_buffers()},
                {o.name for o in kernel.outputs},
            )
        self.kernel = kernel
        self._baked = None

    def _described_args(self, env):
        """The arguments of a described call, with each input read by name.

        The inputs are whole buffers, so they are taken whole rather than by
        position, and the constants beside them are passed as they were given.
        """

        return (
            [_resolve(i, env) for i in self.kernel.inputs],
            list(self.kernel.constant_args),
        )

    def run(self, env: dict) -> None:
        if isinstance(self.kernel, IrFallbackKernel):
            args, kwargs = self._described_args(env)
            result = self.kernel.unflatten_args(*args)
            for out in self.kernel.get_outputs():
                env[out.get_name()] = _dig(result, getattr(out, "indices", ()))
            return
        args = _resolve(self.kernel.args, env)
        kwargs = _resolve(self.kernel.kwargs, env)
        if getattr(self.kernel, "template", None) is not None:
            self._run_template(args, kwargs, env)
            return
        if self.kernel.call_method:
            # A method call names the operation with a string and receives the
            # object it is called on first.
            receiver, *rest = args
            result = getattr(receiver, self.kernel.target)(*rest, **kwargs)
        else:
            target = self.kernel.target
            if not callable(target) and hasattr(target, "default"):
                target = target.default
            result = target(*args, **kwargs)
        for output in self.kernel.outputs:
            env[output.name] = _dig(result, output.path)

    def _run_template(self, args, kwargs, env) -> None:
        """Run a templated operator through the launcher its template chose.

        The choice needs the real operands, so it is made on the first call and
        kept on the step: afterwards the baked launcher is replayed, and the
        template's own cache means the measurement happens once per shape even
        across processes.
        """

        feed = _template_feed(self.kernel, args)
        launch = self._baked
        if launch is None and self.kernel.template is not None:
            # A template is measured once, on the operands of the call that
            # found it, and what it hands back is a launcher over the operands
            # it is handed -- so the measurement pins nothing.
            self._baked = self._bake(feed, args, kwargs)
            launch = self._baked
        result = launch(feed) if launch is not None else None
        if result is None:
            # Either no template claimed this operator, or the one that did
            # declined this call -- a launcher built from a probe answers None
            # when it cannot account for an argument.  The call then runs the
            # operator as the framework runs it, on the operands of this call.
            # Holding a launcher over those operands instead would pin them on
            # the step, and the program reuses the step for every call.
            result = self._call_target(args, kwargs)
        for output in self.kernel.outputs:
            env[output.name] = _dig(result, output.path)

    def _bake(self, feed, args, kwargs):
        """Ask the template which of its configurations fits this call.

        The template owns its result, so it can build the probe its
        configurations are measured on, and the measurement is made without
        this call's tensors.  What comes back is a launcher over whatever
        operands it is handed, so the step stays reusable.
        """

        template = self.kernel.template
        meta = dict(self.kernel.template_meta)
        meta["feed"] = list(feed)
        meta.setdefault("qualifies", True)
        try:
            probe = template.probe(meta)
            meta["probe_feed"] = probe
            # The floor a template is measured against.  With a probe it is
            # the operator run on the probe, so no call's real tensors are
            # pinned by the measurement; without one it replays the call being
            # measured, which is transient either way.
            if probe is None:
                # Nothing to measure on that would leave this call's operands
                # behind: a launcher built here is kept by the step, and the
                # step is reused for every call the program makes, so a
                # measurement that can only run on this call's tensors would
                # pin all of them for the life of the program.  Without a probe
                # the operator runs as the framework runs it.
                return None
            plain = self._probe_launcher()
            launch, params = select(
                template,
                meta, lambda choice, fallback: choice.resolve(
                    fallback if fallback is not None else plain
                )
            )
            self.kernel.config = params
            return launch
        except Exception:  # noqa: BLE001 - a template never breaks the region
            return None

    def _probe_launcher(self):
        """The operator run on a probe feed instead of this call's tensors."""

        meta = self.kernel.template_meta
        literals = list(meta.get("arg_templates") or ())
        positions = list(meta.get("operand_positions") or ())
        if not positions:
            return None
        target = self.kernel.target
        if meta.get("call_method"):
            receiver, *rest = literals
            return lambda values: getattr(receiver, target)(*rest)
        if not callable(target) and hasattr(target, "default"):
            target = target.default

        def launch(values):
            values = list(values)
            if len(values) < len(positions):
                # The launcher is handed the operands the template reads, and
                # a feed that does not carry all of them is not this operator.
                return None
            operands = iter(values)
            try:
                args = [
                    next(operands) if index in positions else literal
                    for index, literal in enumerate(literals)
                ]
                return target(*args)
            except (TypeError, ValueError, IndexError, StopIteration):
                # An argument the probe cannot account for -- a tensor that is
                # neither a template literal nor one of the read operands --
                # leaves the call unreconstructable.  A probe declines, and the
                # operator runs on this call's own operands.
                return None

        return launch

    def _call_target(self, args, kwargs):
        if self.kernel.call_method:
            receiver, *rest = args
            return getattr(receiver, self.kernel.target)(*rest, **kwargs)
        target = self.kernel.target
        if not callable(target) and hasattr(target, "default"):
            target = target.default
        return target(*args, **kwargs)


def _dtype_of(layout_dtype: Any):
    """The framework element type for a layout.

    Lowerings name the element type either as a framework type or by its
    name, and the loop algebra only ever looks at the name, so both spellings
    reach allocation.
    """

    if isinstance(layout_dtype, str):
        return getattr(tp, layout_dtype)
    return layout_dtype


def _template_feed(kernel, args) -> list:
    """The operand list a template's launcher consumes, in recorded order."""

    positions = getattr(kernel, "template_meta", {}).get("operand_positions") or ()
    return [args[i] for i in positions]


def _resolve(value: Any, env: dict) -> Any:
    """The value an operand stands for, as the call receives it.

    An operand that names memory is read by the name it was written under,
    which is what lets one kernel's result be another kernel's input without
    either of them holding it.
    """

    if isinstance(value, ReinterpretView):
        base = _resolve(value.data, env)
        layout = value.get_layout()
        return tp.as_strided(base, layout.size, layout.stride, layout.offset)
    if isinstance(value, (Buffer, ExternOutput)):
        return env[value.name]
    # A view's source is itself a value, and it is read by the name it was
    # written under rather than by what it is.
    if hasattr(value, "get_name") and value.get_name() in env:
        return env[value.get_name()]
    if isinstance(value, (list, tuple)):
        return type(value)(_resolve(item, env) for item in value)
    if isinstance(value, dict):
        return {key: _resolve(item, env) for key, item in value.items()}
    return value


def _dig(result: Any, path: tuple) -> Any:
    for position in path:
        result = result[position]
    return result


class LoopProgram:
    """A lowered, scheduled and compiled program, ready to call."""

    def __init__(self, graph, steps: list):
        self.graph = graph
        self.steps = steps
        self.keep = set()
        for out in graph.graph_outputs:
            if isinstance(out, ReinterpretView):
                self.keep.add(out.data.get_name())
            elif isinstance(out, (Buffer, ExternOutput)):
                # A graph output is handed to the caller, so it has to outlive
                # the last step that writes it.
                self.keep.add(out.name)
        self._plan_releases()

    def _plan_releases(self) -> None:
        """Free each buffer in the step that reads it for the last time.

        A buffer produced by one kernel and consumed by a later one is dead
        once that last consumer has run, not once its producer has: holding it
        until the program ends keeps every intermediate of the whole region
        alive at once, which is what decides whether a region fits in memory.
        A buffer whose last reader is its own producer is dead the moment that
        step ends, and one nothing reads is dead where it is written.
        """
        last_read = {}
        for position, step in enumerate(self.steps):
            for name in step.reads:
                last_read[name] = position
        for position, step in enumerate(self.steps):
            step.frees = [
                name
                for name in sorted(step.writes | step.reads)
                if name not in self.keep
                and name not in self.graph.constants
                and last_read.get(name, -1) <= position
            ]

    def __call__(self, *args):
        env: dict = {}
        for position, value in enumerate(args):
            if position >= len(self.graph.graph_inputs):
                break
            buffer = self.graph.graph_inputs[position]
            if buffer is None or value is None:
                continue
            env[buffer.name] = _input_tensor(buffer, value)
        env.update(self.graph.constants)
        for step in self.steps:
            step.run(env)
            for name in step.frees:
                env.pop(name, None)
        results = [_output_tensor(out, env) for out in self.graph.graph_outputs]
        if getattr(self.graph, "single_output", False) and len(results) == 1:
            return results[0]
        return results


def _input_tensor(buffer: Buffer, value):
    """The tensor a kernel indexes with the buffer's own layout.

    A non-zero storage offset is folded into the base pointer, because the
    layout addresses from the start of the buffer.
    """

    try:
        offset = int(value.storage_offset())
    except Exception:  # noqa: BLE001 - tensors without an offset are fine
        return value
    if offset == 0:
        return value
    return tp.as_strided(value, buffer.get_size(), buffer.layout.stride, 0)


def _output_tensor(out: Any, env: dict):
    """The tensor a graph output stands for, once the steps have run.

    A value that is a window onto someone else's memory is read through that
    memory; anything else that names a buffer is read by the name it was
    written under.
    """

    if isinstance(out, ReinterpretView):
        layout = out.get_layout()
        # A layout's extents and steps are whatever the lowering settled them
        # as, which may be a symbolic or a plain sequence; what addresses memory
        # wants them as integers, and an extent that is not one is settled by
        # the time a value is handed back.
        return tp.as_strided(
            env[out.data.get_name()],
            [int(extent) for extent in layout.size],
            [int(step) for step in layout.stride],
            int(layout.offset),
        )
    if isinstance(out, (Buffer, ExternOutput)):
        return env[out.name]
    if isinstance(out, (TensorBox, StorageBox)):
        # A value that is still held rather than materialized is read from
        # where it was written; handing one back whole would give the caller
        # the description of a value instead of the value.
        return env[out.get_name()]
    return out


__all__ = ["ExternStep", "FusedStep", "HostStep", "LoopProgram", "Step"]



# ---------------------------------------------------------------------------
# choosing among a template's configurations
# ---------------------------------------------------------------------------


def inputs_for(template, meta: dict):
    """The inputs a template is handed for one call.

    The template says which kind of inputs its operands are; building the
    record from a call is the runtime's job, because a call arrives as a
    dictionary of extents and a template asks for something with names.
    """

    import inspect

    inputs_class = getattr(template, "inputs_class", None)
    if inputs_class is None:
        raise NotImplementedError(f"{template.name} does not say what its operands are")
    sizes = tuple(meta.get("operand_sizes") or ())
    offered = {
        "shapes": sizes,
        "strides": tuple(meta.get("operand_strides") or ()),
        "dtypes": tuple([meta.get("operand_dtype")] * len(sizes)),
        "device": meta.get("device"),
        "operands": tuple(meta.get("operand_specs") or ()),
        "feed": tuple(meta.get("feed") or ()),
        "probe_feed": tuple(meta.get("probe_feed") or ()),
        "scalars": tuple(
            (name, meta[name]) for name in sorted(meta) if name in _TEMPLATE_SCALARS
        ),
        "out_dtype": meta.get("out_dtype"),
        "mat1_idx": meta.get("mat1_idx", -2),
        "mat2_idx": meta.get("mat2_idx", -1),
        "extra": meta,
    }
    # Not every kind of inputs says the same things: a product is told which
    # operand is which, a convolution is not.  Passing what the record does not
    # declare would make the record's shape a decision of the caller rather
    # than of the template that asked for it.
    declared = set(inspect.signature(inputs_class).parameters)
    return inputs_class(**{k: v for k, v in offered.items() if k in declared})


#: The names a call may carry that are numbers the template sizes itself by,
#: rather than extents of an operand.
_TEMPLATE_SCALARS = frozenset({"alpha", "beta", "groups", "ceil_mode"})



from .templates.select_algorithm import make_ktc_generator


def _as_launcher(built):
    """The thing to call, whether the template handed back a choice or a launcher.

    A template's answer is a choice: it knows its kernel, its layout and how it
    was built, and it is the right thing to keep.  A step wants something it
    can call, and it keeps that for the life of the program.  So the unwrapping
    happens here, once, where the two meet -- rather than in every place a
    choice might be handed on.
    """

    if built is None:
        return None
    to_callable = getattr(built, "to_callable", None)
    return built if to_callable is None else to_callable()


def _configurations(template, out_specs, meta):
    """The configurations worth considering, as the heuristic states them."""

    inputs = template.heuristics.adjust_kernel_inputs(inputs_for(template, meta), template.name)
    return template.heuristics.get_template_configs(inputs, template.name)



def choices(template, meta: dict) -> list:
    """Every configuration that applies to this call."""

    return collect(template, meta, lambda choice, plain: choice.resolve(plain))


def collect(template, meta: dict, build) -> list:
    """Build the applicable choices, leaving out the ones that do not fit.

    The configurations come from the heuristic, the overrides are what
    this call imposes on all of them, and the pairing is the one the
    deferred-choice generator owns -- so there is a single way for a
    configuration to become a choice, whichever template asked for it.
    """

    specs = template.out_specs(meta)
    inputs = template.heuristics.adjust_kernel_inputs(inputs_for(template, meta), template.name)
    overrides = dict(template.heuristics.get_extra_kwargs(inputs, template.name))
    if specs:
        overrides.setdefault("out_size", tuple(specs[0].size))
        overrides.setdefault("out_dtype", specs[0].dtype)
    out = []
    for choice in make_ktc_generator(
        template,
        _configurations(template, specs, meta),
        {},
        overrides,
        specs[0] if specs else None,
        inputs,
    ):
        maybe_append_choice(template, out, choice, build)
    return out


def maybe_append_choice(template, choices: list, choice, build) -> Any:
    try:
        if build(choice, None) is None:
            raise NotImplementedError(
                f"{template.name} configuration {choice.params!r} does not fit"
            )
    except NotImplementedError as error:
        return error
    choices.append(choice)
    return None


def select(template, meta: dict, build) -> tuple:
    """Choose among this call's configurations and return the winner.

    The choice is made while the region is compiled, so the kernel a
    region runs is settled before it ever runs.  The operator itself is
    always among the candidates, which is what makes measuring safe: the
    worst a measurement can conclude is that the operator was already the
    best of them.
    """

    choices = collect(template, meta, build)
    if not choices:
        return None, None
    if len(choices) == 1 or not meta.get("bench", True):
        return _as_launcher(build(choices[0], None)), choices[0].params
    from .runtime.stax_autotune import bench_candidates

    built = {}

    def materialise(candidate, plain):
        # What is measured is the launcher, not the choice that holds it: the
        # benchmark harness calls what it is given, and a choice is a record of
        # a kernel rather than the kernel.  The choice is kept alongside so the
        # winner can be named.
        launcher = _as_launcher(build(candidate, plain))
        built[candidate] = launcher
        return launcher

    feed = meta.get("probe_feed")
    if feed is None:
        return _as_launcher(build(choices[0], None)), choices[0].params
    best, _launch, _time = bench_candidates(
        materialise, choices, feed, rounds=meta.get("rounds", 2)
    )
    if best is None:
        best = choices[0]
    return _as_launcher(built.get(best) or build(best, None)), best.params
