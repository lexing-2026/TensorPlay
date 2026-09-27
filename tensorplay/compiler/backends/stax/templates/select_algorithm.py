"""Choosing what runs: the built choice, the deferred pairing, and the operation itself.

Everything a call can be answered by is a choice, and a choice answers the
same questions: where its result lands, how to run it, how to measure it, and
whether it fits this call at all.  The operation the templates are measured
against is one of those choices rather than a fallback taken when measurement
fails, which is why it is registered here beside the rest.
"""
from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import itertools
import logging
import operator
import os
import sympy
import textwrap
from typing import NamedTuple

from typing import Any, Callable, Iterator

import tensorplay as tp

from .. import config
from ..codegen.common import KernelTemplate
from ..kernel_inputs import KernelInputs
from ..ir import (
    BaseView,
    Buffer as _Buffer,
    FlexibleLayout,
    Layout,
    compute_required_storage_length,
)
from ..loops import V
from .ir import ChoiceCaller, CutedslChoiceCaller
from ..loops import contiguous_strides, dtype_name
from ..heuristics.template.params import DictKernelTemplateParams, KernelTemplateParams
from ..runtime.triton_helpers import get_constexprs
from ..heuristics.registry import (
    get_template_heuristic as registry_get_template_heuristic,
)
from .triton import CHOICES
from ..codegen.subgraph import SubgraphChoiceCaller
from ..codegen.simd import IterationRangesEntry, IterationRangesRoot
from ..codegen.common import CSE, IndentedBuffer
from ..codegen.triton import texpr
from ..utils import get_dtype_size, sympy_dot, sympy_product, unique
from ..kernel_scheduler import count_flops_fx
from tensorplay.graph.experimental.sympy_functions import OrderedSet

#: Whether a candidate's result is checked against what the operation's own
#: kernel produced.  On by default because a template that computes the wrong
#: number faster is not an answer, and the check costs one run per candidate.
VERIFY = os.environ.get("TP_AUTOTUNE_VERIFY", "1") == "1"

#: Where this module's messages go.  A measurement is worth a line and a
#: measurement that was thrown away is worth none, so the numbers are here
#: rather than printed.
log = logging.getLogger(__name__)


class GeneratedCodeCacheEntry(NamedTuple):
    """What one rendering of a template produced."""

    code: str
    extra: str
    events: list


class GeneratedCodeCache:
    """Rendered code, kept per template so the same call is not rendered twice.

    The key is everything the rendering depended on: which values were read,
    how the launch was shaped, and what was imposed on every configuration.
    A key that cannot be built is no key at all, and the entry is then neither
    read nor written -- a partially known call must not be answered from a
    cache entry that some other call wrote.
    """

    def __init__(self, *args, **kwargs):
        self._cache: dict[str, GeneratedCodeCacheEntry] = {}

    def cache_clear(self) -> None:
        self._cache.clear()

    def __repr__(self):
        return repr(self._cache)

    def make_key(self, **keyed) -> str | None:
        """The identity of one rendering, or ``None`` when it cannot be pinned.

        Every part has to be known for the entry to be reusable, so a part
        that is not is what makes the whole key unusable rather than being
        skipped.
        """

        if any(part is None for part in keyed.values()):
            return None
        return "|".join(f"{name}={part!r}" for name, part in sorted(keyed.items()))

    def get_entry(self, cache_key: str | None) -> GeneratedCodeCacheEntry | None:
        if cache_key is None:
            return None
        return self._cache.get(cache_key, None)

    def put_entry(
        self,
        cache_key: str | None,
        code: str,
        extra: str,
        events: list,
    ) -> None:
        if cache_key is None:
            return
        self._cache[cache_key] = GeneratedCodeCacheEntry(code, extra, events)


_registered_caches: list = []


def clear_on_fresh_cache(obj: Any) -> Any:
    """Register a cache to be emptied whenever the caches are declared fresh.

    A template that renders the same call twice is only wasteful until the
    second rendering disagrees with the first, which it can when the region
    changed underneath it.  So every cache a template holds is emptied
    together, rather than each template deciding on its own.
    """

    if not hasattr(obj, "cache_clear") or not callable(obj.cache_clear):
        raise AttributeError(f"{obj} does not have a cache_clear method")

    _registered_caches.append(obj)
    return obj


def fresh_caches() -> None:
    """Empty every registered cache."""

    for obj in _registered_caches:
        obj.cache_clear()


class KernelTemplateChoice:
    """One template, one configuration, and the choice they make -- eventually.

    The kernel is built the first time the choice is asked for and then kept,
    including when the build turns out not to apply: a configuration that does
    not fit this call is an answer, not a retry.
    """

    def __init__(
        self,
        template: KernelTemplate,
        params: KernelTemplateParams,
        extra_kwargs: dict[str, Any],
        layout: Layout | None,
        inputs: KernelInputs,
        plain_launch: Any = None,
    ):
        self.template = template
        self.params = params
        self.extra_kwargs = dict(extra_kwargs)
        self.layout = layout
        self.inputs = inputs
        self.annotations: dict[str, Any] = {"ktc": self}
        self._plain_launch = plain_launch

    @property
    def key(self) -> tuple:
        """What identifies this choice across processes."""

        return (
            self.template.uid,
            repr(self.params.to_serializeable_dict()),
            repr(self.layout),
            str(self.inputs.device),
        )

    @property
    def choice(self):
        """The built choice, made on the first ask and kept afterwards.

        Building is deferred to here because most configurations are never
        reached: a table is offered whole, one entry is measured and wins, and
        the rest are discarded without ever being built.  A refusal is kept just
        like a success, because a configuration that does not apply to this call
        is an answer rather than something to retry.
        """

        return self.resolve(self._plain_launch)

    def resolve(self, plain_launch):
        """The launcher for this configuration, or ``None`` when it does not fit."""

        if not hasattr(self, "_resolved"):
            self._resolved = True
            try:
                self._choice = self.template.choice_or_none(
                    **self.params.to_kwargs(),
                    **self.extra_kwargs,
                    layout=self.layout,
                    input_nodes=self.inputs.nodes(),
                )
            except NotImplementedError:
                self._choice = None
            if self._choice is not None:
                self._choice.annotations = self.annotations
        return self._choice

    def __repr__(self) -> str:
        return f"KernelTemplateChoice({self.template.name}, {self.params!r})"
def make_ktc_generator(
    template: KernelTemplate,
    cs: Iterator[KernelTemplateParams],
    extra_kwargs: dict[str, Any],
    overrides: dict[str, Any],
    layout: Layout | None,
    inputs: KernelInputs,
) -> Iterator[KernelTemplateChoice]:
    """One deferred choice per configuration, with the overrides folded in.

    The overrides are what a caller imposes on every configuration -- the
    parts of the call that are not a choice -- and folding them into the
    parameters here means a template reads one dict.
    """

    for params in cs:
        merged = {**params.to_kwargs(), **overrides}
        yield KernelTemplateChoice(
            template=template,
            params=DictKernelTemplateParams(merged),
            extra_kwargs=extra_kwargs,
            layout=layout,
            inputs=inputs,
        )
class KernelNamespace:
    """A place to hang the operations a generated wrapper calls by name.

    The names have to survive into generated text, so they are held on an
    object rather than only in the table that made them: what codegen writes
    is a reference to this, and what it resolves to at run time is whatever was
    registered under that name.
    """


#: The operations a generated wrapper reaches by name.
extern_kernels = KernelNamespace()


def call_operation(name, kernel, layout, kwargs=None, has_out_variant=True):
    """One call of an operation, registering the operation if it is new.

    A launcher the compiler was handed is an operation like any other: it just
    happens to have arrived already built rather than named.  It is registered
    the same way, so that everything downstream -- the trace, the generated
    call, the identity a cache keys on -- sees one kind of thing.
    """

    if kernel is None:
        return None
    choice = ExternKernelChoice.lookup(name)
    if choice is None or choice.kernel is not kernel:
        choice = ExternKernelChoice(
            kernel, name=name, has_out_variant=has_out_variant
        )
    return choice.bind(input_nodes=(), layout=layout, **(kwargs or {}))


class ExternKernelCaller(ChoiceCaller):
    """A call to an operation that already exists, built and ready to measure.

    The choice it was made from is kept rather than copied apart, because
    everything still needed after the call is built -- which overload the
    operation is, whether it writes into a buffer the caller supplied or one it
    allocates, the name a trace shows it under -- is a property of the
    operation, and is the same however many times it is called.
    """

    def __init__(
        self,
        choice,
        input_nodes=(),
        layout=None,
        kwargs=None,
        *,
        has_out_variant=True,
    ):
        super().__init__(choice.name, input_nodes, layout, description="")
        self.choice = choice
        self.kwargs = kwargs or {}
        #: Whether the operation writes into memory the caller named or memory
        #: it allocates is a property of the operation, and it decides what kind
        #: of node describing this call turns out to be.
        self.has_out_variant = has_out_variant
        self.gm = choice.gm
        self._callable = choice.to_callable()

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long one run of the operation takes.

        Where the operation writes into a buffer the caller supplied, that
        buffer is named rather than appended: the operation's own signature
        says which of its arguments it is, and appending it would be a
        statement about the order rather than the name.  An operation that
        returns its result is simply called.
        """

        from ..runtime.stax_autotune import bench_launch

        algo = self.to_callable()
        if self.has_out_variant and out is not None:
            def launch(_operands, _algo=algo, _args=args, _out=out):
                return _algo(*_args, out=_out)
        else:
            def launch(_operands, _algo=algo, _args=args):
                return _algo(*_args)
        return bench_launch(launch, [])

    def __str__(self) -> str:
        return f"ExternKernelCaller({self.choice.call_name()})"

    def call_name(self) -> str:
        """The name a trace shows this call under."""

        return self.choice.call_name()

    def to_callable(self):
        return self.choice.to_callable()

    def hash_key(self) -> str:
        """What identifies this call: the operation, and the arguments beside it.

        Two calls of one operation with the same arguments are the same call,
        so the key is the operation's own identity followed by the arguments
        written out in a fixed order.
        """

        return "-".join(
            [
                self.choice.name,
                *[
                    f"{kwarg}={self.kwargs[kwarg]!r}"
                    for kwarg in sorted(self.kwargs.keys())
                ],
                self.choice.hash_key(),
            ]
        )

    def output_node(self):
        """The result of making this choice, as a value the rest can read.

        A choice that is a library call has no source to render, so asking it
        for its result is asking it to describe the call: what is written, what
        is passed, and where the answer goes.  The node that comes back is a
        description rather than a number, and becomes a number when the region
        is finally written out -- which is why it can be made now and measured
        later without being measured twice.
        """

        from .. import ir

        if self.choice.use_fallback_kernel:
            # The operation is one that was described rather than wrapped, so
            # what the call needs to be described comes from the overload
            # itself.  A description that names no overload names nothing.
            if self.choice.op_overload is None:
                raise AssertionError(
                    "a call described as a fallback has to say which overload "
                    "it is, or there is nothing to call"
                )
            inner = ir.FallbackKernel.create(
                self.choice.op_overload, *self.input_nodes, **self.kwargs
            )
        elif self.choice.kernel_creator is not None:
            inner = self.choice.kernel_creator(*self.input_nodes, **self.kwargs)
        else:
            cls = ir.ExternKernelOut if self.has_out_variant else ir.ExternKernelAlloc
            inner = cls(
                layout=self.layout,
                inputs=self.input_nodes,
                python_kernel_name=self.choice.call_name(),
                cpp_kernel_name=self.choice.cpp_kernel_name,
                ordered_kwargs_for_cpp_kernel=(
                    self.choice.ordered_kwargs_for_cpp_kernel
                ),
                op_overload=self.choice.op_overload,
                kwargs=self.kwargs,
            )
        if "ktc" in self.annotations:
            inner.annotations["ktc"] = self.annotations["ktc"]
        return ir.TensorBox.create(inner)

    def info_dict(self) -> dict:
        """What is worth writing down about this choice."""

        return {
            "backend": "extern",
            "kernel_call_name": self.choice.call_name(),
        }

    def autoheuristic_id(self) -> str:
        return f"extern_{self.choice.name}"


@dataclasses.dataclass()
class SubgraphInfo:
    """What is needed to lower one subgraph of a kernel written as a template.

    The template's text is written once and then rendered once, but the parts
    of it that belong to a subgraph are worked out while that subgraph is
    lowered.  So each subgraph keeps its own text and its own bookkeeping, and
    a kernel swaps those in while it works on one subgraph and swaps them back
    when it is done.
    """

    body: IndentedBuffer
    template_mask: str | None = None
    template_out_shape: str | tuple[str] | None = None
    compute: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    indexing_code: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    loads: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    stores: IndentedBuffer = dataclasses.field(default_factory=IndentedBuffer)
    ops_handler: V.WrapperHandler | None = None
    cse: CSE[Any, str] | None = None

    # Only carried over when they were made, because a subgraph that made
    # none of them shares the kernel's rather than replacing it with nothing.
    range_trees: list[IterationRangesRoot] | None = None
    range_tree_nodes: dict[sympy.Symbol, IterationRangesEntry] | None = None
    numels: dict[str, sympy.Expr] | None = None

    # Maps a range-tree root's name to the name the prologue gave it.
    root_var_renames: dict[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self):
        self.only_copy_if_non_none_fields = (
            "range_trees",
            "range_tree_nodes",
            "numels",
            "cse",
        )

    def to_dict(self):
        return {
            field.name: getattr(self, field.name) for field in dataclasses.fields(self)
        }


class ExternKernelChoice:
    """An operation that can hold its own against a kernel, as a choice.

    The operation every template is measured against is not a fallback taken
    when measurement fails -- it is one of the things being measured, and it
    wins whenever no kernel beats it.  So it is registered here, beside the
    templates, under a name codegen can refer to, and it answers the same
    questions they do; a caller can hand a list of either to the same
    enumeration without caring which is which.

    Each instance registers once under its name.  Registering the same
    callable twice is tolerated, because a module can be initialised twice in
    one process; registering a *different* callable under a name already taken
    is refused, because that is a collision rather than a re-registration.
    """

    _registry: dict[str, "ExternKernelChoice"] = {}

    def __init__(
        self,
        kernel: Callable[..., Any],
        cpp_kernel: str | None = None,
        *,
        name: str | None = None,
        has_out_variant: bool = True,
        op_overload: Any = None,
        use_fallback_kernel: bool = False,
        kernel_creator: Callable[..., Any] | None = None,
    ):
        name = name or getattr(kernel, "__name__", None)
        if not callable(kernel):
            raise AssertionError("an extern choice has to wrap something callable")
        existing = getattr(extern_kernels, name, None)
        if existing is not None and existing is not kernel:
            raise AssertionError(f"duplicate extern kernel: {name}")
        self.kernel = kernel
        self.name = name
        #: The name the same operation goes by when it is emitted as C++, and
        #: the graph it stands for when the choice is a whole region.  Both are
        #: absent for an ordinary library call, and both are recorded rather
        #: than inferred: a name that had to be guessed would be a name that
        #: could be guessed wrong.
        self.cpp_kernel_name = cpp_kernel
        self.has_out_variant = has_out_variant
        setattr(extern_kernels, name, kernel)
        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel
        self.kernel_creator = kernel_creator
        self.ordered_kwargs_for_cpp_kernel: tuple = ()
        #: There is no source for an operation that is not written out here, so
        #: there is nothing to hash: the operation is the kernel, and the name
        #: is the whole of its identity.
        self.src_hash: str | None = None
        self.gm: Any = None
        ExternKernelChoice._registry[name] = self

    @classmethod
    def lookup(cls, name: str) -> "ExternKernelChoice | None":
        return cls._registry.get(name)

    def to_callable(self) -> Callable[..., Any]:
        """The operation as it is run when the result goes somewhere named.

        The name under which the operation is registered is the function a
        caller would reach for, which for an operation that can write into a
        buffer the caller supplied is not the same as the form of it that
        takes that buffer: the first returns its result and the second is
        handed where to put it.  A measurement hands it a buffer, so what it is
        given is the form that takes one -- and an operation with no such form
        is the function itself.
        """

        if self.op_overload is not None:
            return self.op_overload
        return getattr(extern_kernels, self.name)

    def call_name(self) -> str:
        """The name a trace shows this operation under."""

        return f"extern_kernels.{self.name}"

    def hash_key(self) -> str:
        """What identifies the operation itself.

        The name is the identity, but two operations can share one across
        different builds of the library, so the function behind the name and
        its source are hashed as well where the source can be read at all.
        """

        import inspect

        from ..codecache import code_hash

        fn = self.to_callable()
        parts = [
            self.name,
            getattr(fn, "__name__", ""),
            getattr(fn, "__module__", ""),
        ]
        try:
            parts.append(inspect.getsource(fn))
        except Exception:
            pass
        return code_hash("-".join(parts))

    def bind(self, input_nodes, layout, ordered_kwargs_for_cpp_kernel=(), **kwargs):
        """This operation, as one particular call of it."""

        self.ordered_kwargs_for_cpp_kernel = ordered_kwargs_for_cpp_kernel
        return ExternKernelCaller(
            self,
            input_nodes,
            layout,
            kwargs,
            has_out_variant=self.has_out_variant,
        )

    @property
    def uid(self) -> str:
        """Namespaced by kind, so two kinds may share a name."""

        return f"framework::{self.name}"

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """The operation itself, as the choice it always is.

        Carries across what the call needs in order to be described rather than
        merely named: the values it is handed, and whether it writes into a
        buffer the caller supplied or one it allocates.
        """

        temp_choices: list[Any] = []
        result = self.maybe_append_choice(temp_choices, **kwargs)
        if result is None and len(temp_choices) == 1:
            return temp_choices[0]
        return None

    def maybe_append_choice(self, choices: list, **kwargs: Any):
        # Convenience function to match the template interface, so that
        # templates and operations can be treated the same when generating
        # choice callers.
        if "input_nodes" not in kwargs:
            raise AssertionError("input_nodes argument required")
        if "layout" not in kwargs:
            raise AssertionError("layout argument required")
        input_nodes = kwargs.pop("input_nodes")
        layout = kwargs.pop("layout")
        choices.append(
            self.bind(input_nodes=input_nodes, layout=layout, **kwargs)
        )
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        return self.choice_or_none(**kwargs)

    def __repr__(self) -> str:
        return f"ExternKernelChoice({self.name})"


class TritonChoiceCaller(ChoiceCaller):
    """A choice whose kernel is emitted as source for a streaming backend.

    What makes it a separate kind rather than a flag is the hash key: a kernel
    emitted as source is recognised in a cache by the source itself, so a
    change to the emitter invalidates every stored decision that named it,
    with no separate version to keep in step.
    """

    def __init__(self, name, input_nodes=(), layout=None, description="",
                 source: str = "", src_hash: str | None = None,
                 launcher_args: tuple | None = None,
                 num_stages: int = 2, num_warps: int = 4,
                 config: dict | None = None, operands: dict | None = None,
                 inductor_meta: dict | None = None, template: Any = None):
        super().__init__(name, input_nodes, layout, description)
        self.source = source
        self._src_hash = src_hash
        #: How many steps of the contraction are held at once, and how many
        #: programs run.  How many are live at once is what decides how much
        #: memory the kernel needs, so a configuration that does not fit the
        #: device is tried again with fewer of them.
        self.stages = num_stages
        self.warps = num_warps
        #: The configuration this was built from, and what it was written
        #: against, kept so that the same text can be compiled again with one
        #: thing changed rather than guessed at again from the shape.
        self.config = dict(config or {})
        self.operands = dict(operands or {})
        self.inductor_meta = dict(inductor_meta or {})
        self.template = template
        #: What a launcher written from this kernel is handed, in the order it
        #: declares: the operands of the call, then the extents and strides the
        #: kernel was written to be given, then the stream.  A measurement is
        #: handed the operands and has to say the rest, and it can only say the
        #: rest if the caller settled it when the launcher was written -- which
        #: is the one place that knows the signature.
        self.launcher_args = launcher_args

    def hash_key(self) -> str:
        parts = [self.name, self.description]
        if self.layout is not None:
            parts.append(repr(self.layout))
        digest = self._src_hash
        if digest is None and self.source:
            digest = hashlib.sha1(self.source.encode()).hexdigest()[:16]
        if digest is not None:
            parts.append(digest)
        return ":".join(parts)

    def _with_fewer_stages(self) -> "TritonChoiceCaller":
        """The same configuration, holding fewer steps at once.

        What a kernel needs in memory is what it is working on at once, so the
        same tile with fewer steps live is the same computation with a smaller
        working set.  It is a different kernel and therefore a different
        configuration, and it is measured as one rather than folded into the
        first one's answer.
        """

        config = {**self.config, "num_stages": max(1, int(self.stages) - 1)}
        meta = {**self.inductor_meta}
        result = _compile_rendered(
            self.template,
            self.source,
            config={k: v for k, v in config.items() if k != "num_stages"},
            constants={
                k: v
                for k, v in config.items()
                if k not in ("num_stages", "num_warps", "layout",
                             "input_nodes", "out_size")
            },
            inductor_meta=meta,
        )
        if result is None:
            raise NotImplementedError(
                "this configuration does not fit even with a single step"
            )
        result.kernel._init_handles()
        launcher = result.make_launcher()
        launcher.__name__ = self.name
        return TritonChoiceCaller(
            name=self.name,
            input_nodes=self.input_nodes,
            layout=self.layout,
            description=self.description,
            source=self.source,
            src_hash=self._src_hash,
            launcher_args=self.launcher_args,
            num_stages=int(config["num_stages"]),
            num_warps=int(self.warps),
            config=config,
            operands=self.operands,
            inductor_meta=meta,
            template=self.template,
        ).bind(launcher)

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long one run of this choice takes.

        A measurement is handed the operands of the call and nothing else: the
        extents and strides the kernel was written to be given, and the stream
        it is launched on, are settled when the launcher is written, and are
        carried alongside it.  What is measured is therefore a launch made the
        way a real one is made.
        """

        from ..runtime.stax_autotune import bench_launch

        from ..runtime.triton_compat import OutOfResources

        algo = self.to_callable()
        # The result is written into a buffer the caller named, and in the
        # signature that buffer is among the pointers -- so it goes with the
        # operands rather than at the end.  What follows the operands is what
        # the kernel was written to be given and the stream it runs on, in the
        # order it declares them.
        operands = [*args, *([out] if out is not None else []),
                    *(self.launcher_args or ())]
        def launch(these, _algo=algo):
            return _algo(*these)

        try:
            return bench_launch(launch, operands)
        except OutOfResources:
            # The kernel is too large for the shared memory this device has.
            # A tile that is too large is a configuration that cannot be
            # measured here, which is an answer about this machine rather than
            # a failure: the other candidates are still answers.  The only
            # thing worth trying is the same tile with fewer steps held at
            # once, since that is what decides how much memory is live.
            if int(self.stages) > 1:
                return self._with_fewer_stages().benchmark(*args, out=out)
            raise NotImplementedError(
                "this configuration does not fit the shared memory on this "
                "device, even with a single step"
            )

    def autoheuristic_id(self) -> str:
        return "triton_template"

class TritonTemplateKernel:
    """One built kernel: its source, the symbol it defines, and how to launch.

    Kept apart from the template that describes it because the two have
    different lifetimes.  A template is written once and describes a family of
    launches; a kernel is built from one piece of source and is then the thing
    every launch of that source shares.  Two configurations whose source comes
    out identical therefore share one kernel, which is why what is remembered
    is the hash of the source and not the configuration that produced it.
    """

    #: One kernel per distinct source, however many templates asked for it.
    _memo: dict = {}

    def __init__(self, kernel_name: str, source: str, symbol: str, grid=None,
                 meta: dict | None = None):
        self.kernel_name = kernel_name
        self.source = source
        self.symbol = symbol
        self.grid = grid
        self.meta = dict(meta or {})
        self._fn = None

    @property
    def key(self) -> str:
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()[:24]

    @classmethod
    def get(cls, kernel_name: str, source: str, symbol: str, grid=None,
            meta: dict | None = None) -> "TritonTemplateKernel":
        """The kernel for this source, built once and shared from then on."""

        key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:24]
        found = cls._memo.get(key)
        if found is not None:
            return found
        kernel = cls(kernel_name, source, symbol, grid, meta)
        cls._memo[key] = kernel
        return kernel

    def build(self):
        """The compiled function, made on the first ask and kept afterwards."""

        if self._fn is None:
            from ..codegen.triton_conv import _build

            self._fn = _build(self.source, f"<tensorplay-stax-{self.key}>", self.symbol)
        return self._fn

    def __call__(self, *args, **kwargs):
        """Launch, computing the grid first when this kernel was given a way to.

        A kernel that knows how many programs to start is asked; one that was
        handed the count launches with it.  The two are told apart by whether a
        way to compute it was supplied at all, rather than by a flag, because a
        kernel given no way has nothing to compute and a kernel given one has
        nothing to be told.
        """

        if self.grid is None:
            return self.build()(None, *args, **kwargs)
        grid = self.grid(*args, **kwargs) if callable(self.grid) else self.grid
        return self.build()[grid](*args, **kwargs)

    def launch_with(self, grid, *args, **kwargs):
        """Launch with the grid already counted, for a caller that counted it."""

        return self.build()[grid](*args, **kwargs)

    def __repr__(self) -> str:
        return f"TritonTemplateKernel({self.kernel_name})"

    def _gen_tmp_var(self) -> str:
        """A name nothing else in this kernel is using."""

        return f"_tmp_var{next(self.tmp_var_ctr)}"

    def input_dependent_preserved_state(self) -> str:
        """The part of the kernel's state that a shape decides.

        Everything here is a function of the shapes, so it does not have to be
        written down again when a cached kernel is replayed against the same
        shapes.  The output buffers are left out on purpose: nothing reads them
        back, so carrying them would only make two runs of the same shapes look
        different.
        """

        return repr(
            [
                self.args.input_buffers,
                self.args.sizevars,
                self.args.workspace_args,
                self.prologue_supported_inputs,
                self.frozen_layouts_cnt,
            ]
        )

    def record_input_dependent_tracked_event(self) -> Callable[..., Any]:
        """Note calls that changed a shape-dependent part of the state.

        A kernel built once and run against several shapes is remembered by
        what its shapes decided.  Wrapping the calls that make those decisions
        is how a replay learns which of them to make again.
        """

        def decorator(fn) -> Callable[..., Any]:
            def wrapper(*args, **kwargs) -> Any:
                pre_state = self.input_dependent_preserved_state()
                result = fn(*args, **kwargs)
                post_state = self.input_dependent_preserved_state()
                if pre_state != post_state:
                    if self.cached_replay_events is None:
                        raise AssertionError("cached_replay_events must not be None")
                    self.cached_replay_events.append((fn.__name__, [*args], {**kwargs}))
                return result

            return wrapper

        return decorator

    def replay_cached_events(self, events) -> None:
        """Make again the decisions that the shapes decided last time."""

        for f, args, kwargs in events:
            getattr(self, f)(*args, **kwargs)

    @contextlib.contextmanager
    def set_subgraph_body(self, body_name: str):
        """Work on one subgraph's text, and put the rest back afterwards.

        Everything a subgraph decides is decided against the text and the
        bookkeeping it owns, so both are put in place for the length of the
        work and taken back out after it.
        """

        if not all(
            hasattr(self, field.name) for field in dataclasses.fields(SubgraphInfo)
        ):
            raise AssertionError("missing expected SubgraphInfo fields on self")
        old_state = {
            key.name: getattr(self, key.name)
            for key in dataclasses.fields(SubgraphInfo)
        }

        if body_name not in self.subgraph_bodies:
            raise AssertionError(body_name)

        subgraph = self.subgraph_bodies[body_name]
        for key, value in subgraph.to_dict().items():
            if value is None and key in subgraph.only_copy_if_non_none_fields:
                continue
            setattr(self, key, value)

        context = (
            contextlib.nullcontext
            if not self.ops_handler
            else lambda: V.set_ops_handler(self.ops_handler(V.get_ops_handler()))
        )
        with context():
            yield
        self.subgraph_bodies[body_name] = SubgraphInfo(
            **{
                key.name: getattr(self, key.name)
                for key in dataclasses.fields(SubgraphInfo)
            }
        )
        for key, value in old_state.items():
            setattr(self, key, value)

    @contextlib.contextmanager
    def create_subgraph_body(self, body_name: str, clear_cse: bool = False):
        """A subgraph that has not been worked on yet gets its text here.

        The name has to be new, because working on a subgraph that already has
        text would put two lowerings in one place.
        """

        if body_name in self.subgraph_bodies:
            raise AssertionError(f"subgraph body {body_name} already exists")
        self.subgraph_bodies[body_name] = SubgraphInfo(
            IndentedBuffer(), None, None, cse=self.cse.clone() if clear_cse else None
        )
        with self.set_subgraph_body(body_name):
            yield

    def _make_independent_subgraph(self, subgraph_name, numel, **extra_fields):
        """A subgraph that ranges over its own numbers, not the kernel's.

        Used by epilogue and prologue hooks that have to do their own
        bookkeeping rather than share the kernel's.
        """

        groups = {"x": V.graph.sizevars.simplify(numel), "r0_": sympy.S.One}
        self.subgraph_bodies[subgraph_name] = SubgraphInfo(
            body=IndentedBuffer(),
            cse=self.cse.clone(),
            range_trees=self.construct_range_trees(
                pid_cache=None,
                inside_reduction=False,
                is_reduction=False,
                numels=groups,
                no_x_dim=False,
            ),
            range_tree_nodes={},
            numels=groups,
            **extra_fields,
        )

    def _setup_contiguous_index_state(
        self,
        indices: list[str],
        index_symbols: list[sympy.Symbol],
        lengths: list[sympy.Expr],
        mask: str | None,
        xindex_name: str = "xindex",
    ) -> sympy.Expr:
        """Name the output's dimensions and give the whole output one number.

        The result is one number per element of the result, and having one
        means the load and the store of a whole kernel can be ordered by it.
        Naming the dimensions is what lets the later text ask for a position
        by dimension rather than by arithmetic.

        Returns the expression for that one number.
        """

        for name, range_tree_entry in zip(
            indices, self.range_trees[0].construct_entries(lengths)
        ):
            range_tree_entry.set_name(name)
        contiguous_index = sympy_dot(
            FlexibleLayout.contiguous_strides(lengths), index_symbols
        )
        contiguous_index = self.rename_indexing(contiguous_index)
        self.body.writeline(f"{xindex_name} = " + texpr(contiguous_index))
        xindex_entry = self.range_trees[0].lookup(sympy.S.One, sympy_product(lengths))
        old_symbol = xindex_entry.symbol()
        xindex_entry.set_name(xindex_name)
        if self.range_tree_nodes.get(old_symbol) is xindex_entry:
            del self.range_tree_nodes[old_symbol]
        self.range_tree_nodes[xindex_entry.symbol()] = xindex_entry
        self.template_mask = mask
        self.template_indices = indices
        return contiguous_index

    def _make_codegen_hook(
        self, subgraph_name: str, indent_width: int = 0
    ) -> Callable[[], str]:
        """A hook that lowers a subgraph when the template's text asks for it.

        The text is laid out before the subgraphs in it are lowered, so where
        each one goes is written down as a placeholder and this fills it in.
        """

        def hook():
            with self.set_subgraph_body(subgraph_name):
                self.codegen_body()
                self.cse.invalidate(OrderedSet())
                result = self.body.getvalue()
                if indent_width:
                    result = textwrap.indent(
                        textwrap.dedent(result), " " * indent_width
                    )
                return result.strip()

        return hook

    def need_numel_args(self):
        return False

    def estimate_kernel_num_bytes(self):
        """An upper bound on the bytes this kernel touches.

        A value that is both read and written is counted twice, because it is
        both read and written.
        """

        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        num_bytes = []
        for i, inp in enumerate(itertools.chain(self.input_nodes, (self.output_node,))):
            size = V.graph.sizevars.optimization_hints(inp.get_size(), fallback=0)
            numel = functools.reduce(operator.mul, size, 1)
            dtype_size = get_dtype_size(inp.get_dtype())
            num_bytes.append(numel * dtype_size * (1 + int(i < ninplace_args)))
        return sum(num_bytes)

    def estimate_flops(self) -> int:
        """How much arithmetic this kernel does, for weighing it against others."""

        for node in self.input_nodes:
            for fx_node in node._current_origins:
                f = count_flops_fx(fx_node)
                if f is not None:
                    if isinstance(f, tp.SymInt):
                        f = f.node.expr
                    return V.graph.sizevars.optimization_hint(f, fallback=0)
        return 0

    def jit_lines(self):
        """Render decorators and metadata for the generated Triton template."""

        if self.use_jit:
            return "@triton.jit"

    def _register_hook(
        self,
        hook_name: str,
        hook_fn,
        *,
        allow_overwriting: bool = False,
    ) -> str:
        """Remember what is to go where a placeholder stands.

        The name is the string the rendered text will carry and the function is
        what fills it in, so a name already in use means two different things
        want the same place -- which is only ever meant if the caller said so.
        """

        if not allow_overwriting:
            if hook_name in self.render_hooks:
                raise AssertionError(
                    f"Tried to register the hook {hook_name} multiple times. If "
                    "desired, pass allow_overwriting=True to _register_hook"
                )
        self.render_hooks[hook_name] = hook_fn
        return hook_name

    def _register_extra_template_env_fns(self, *fns: Callable[..., Any]):
        """Add more names to the template's rendering, beyond the usual ones.

        A function added this way may itself register a hook, so a template can
        carry a part of its text that nothing else knows how to produce.
        """

        self.extra_template_env_fns.extend(fns)

    def gen_argdefs(self):
        """The parameters the entry point takes, once the whole body is known.

        A template can decide to ask for another argument while it renders, so
        the list cannot be written until the whole body has been asked.
        """

        def hook():
            arg_defs, *_ = self.args.python_argdefs()
            return f"{', '.join(x.full_name() for x in arg_defs)}"

        return self._register_hook("<ARGDEFS>", hook, allow_overwriting=True)

    def gen_defines(self):
        return self.defines

    def def_kernel(self, *argnames):
        """The entry point: its parameters, and the names the body calls them by.

        The body names its operands, and those names are what it says.  Each
        one is bound to the argument that carries the operand's storage, and
        the binding is written out where the body can rely on it.

        A name given here is the name the body uses; it is not looked for
        among the kernel's arguments, because the kernel's arguments are in
        the order the graph gave them and the body is free to name them as it
        likes.
        """

        if not all(isinstance(x, str) for x in argnames):
            raise AssertionError("all argnames must be str")
        renames = IndentedBuffer(initial_indent=1)

        named_args = self.input_nodes[
            self.prefix_args : len(self.input_nodes) - self.suffix_args
        ]

        if len(argnames) != len(named_args):
            raise AssertionError(
                (
                    len(argnames),
                    len(named_args),
                    self.prefix_args,
                    len(self.input_nodes),
                )
            )

        for input_node in self.input_nodes[: self.prefix_args]:
            self.args.input(input_node.get_name())

        for name, input_node in zip(argnames, named_args):
            arg_name = f"arg_{name}"
            self.named_input_nodes[name] = input_node
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input_buffers[input_node.get_name()] = arg_name

        # The args may be duplicated, so renaming must be after args are
        # de-duplicated.
        for name in argnames:
            input_node = self.named_input_nodes[name]
            if self.prologue_loads_all_inputs:
                self.prologue_supported_inputs.add(input_node.get_name())
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            arg_name = self.args.input_buffers[input_node.get_name()]
            if input_node.get_layout().offset == 0:
                renames.writeline(f"{name} = {arg_name}")
            else:
                offset = texpr(self.rename_indexing(input_node.get_layout().offset))
                renames.writeline(f"{name} = {arg_name} + {offset}")

        for input_node in self.input_nodes[len(self.input_nodes) - self.suffix_args :]:
            if input_node.get_name() in V.graph.removed_buffers:
                continue
            if input_node.get_name() in self.prologue_fused_inputs:
                continue

            self.args.input(input_node.get_name())

        def hook():
            arg_defs, *_ = self.args.python_argdefs()
            code = IndentedBuffer()
            code.splice(self.gen_common_triton_imports())
            code.splice(self.jit_lines())
            code.writeline(
                f"def {self.kernel_name}({', '.join(x.full_name() for x in arg_defs)}):"
            )
            with code.indent():
                code.splice(self.defines)
                code.splice(renames.getvalue())
                self.codegen_prologue(code)
            return code.getvalue()

        return self._register_hook("<DEF_KERNEL>", hook)

    def size(self, name: str | None, index: int):
        """The extent of a dimension, as the body should write it.

        Asking with no name gives the result's own extent.  When the kernel
        indexes in 64 bits the extent is widened on the way out, because an
        extent computed in 32 bits can wrap around on a large tensor and turn
        into a wrong answer rather than an error.
        """

        if not isinstance(index, int):
            raise AssertionError(f"expected index to be int, got {type(index)}")
        if name is None:
            val = self.output_node.get_size()[index]
        else:
            if not isinstance(name, str):
                raise AssertionError(f"expected name to be str, got {type(name)}")
            val = self.named_input_nodes[name].get_size()[index]
        result = texpr(self.rename_indexing(val))
        if self.index_dtype == "tl.int64":
            return f"tl.full([], {result}, dtype=INDEX_DTYPE)"
        return result

    def stride(self, name, index=None):
        """The strides of a tensor, or one of them, as the body should write it.

        Asking with no name gives the result's own strides.  Asking for one
        dimension gives that dimension's stride, and asking for none of them
        gives the whole list, which is what a body needs when it is doing its
        own arithmetic over a tensor it did not load elementwise.
        """

        if name is None:
            val = self.output_node.get_stride()
        else:
            if not isinstance(name, str):
                raise AssertionError(f"expected name to be str, got {type(name)}")
            val = self.get_stride_and_maybe_freeze_layout(self.named_input_nodes[name])

        if isinstance(index, int):
            return texpr(self.rename_indexing(val[index]))
        return ", ".join([texpr(self.rename_indexing(i)) for i in val])

    def _get_subgraph(self, subgraph_number: int):
        """The subgraph a hook was asked about, checked that it is there.

        The body is emptied of everything else, because what a modification
        adds is the whole of what its subgraph contributes and text left over
        from before would be added to it by accident.
        """

        if not isinstance(subgraph_number, int):
            raise AssertionError(
                f"expected subgraph_number to be int, got {type(subgraph_number)}"
            )
        if not isinstance(self.subgraphs, list):
            raise AssertionError(
                f"expected self.subgraphs to be list, got {type(self.subgraphs)}"
            )
        if subgraph_number >= len(self.subgraphs):
            raise AssertionError(
                f"Invalid subgraph number provided to create_modification, "
                f"{subgraph_number} must be < {len(self.subgraphs)}"
            )
        if self.body.getvalue() != "":
            raise AssertionError("Body should be clear before adding a modification")
        return self.subgraphs[subgraph_number]


def _compile_rendered(template, source: str, config: dict, constants: dict,
                      inductor_meta: dict):
    """The rendered text, compiled the way a configuration is compiled.

    Two things are handed to the runtime: the signature, which is the name and
    type of every argument, and the constants, which are the values of the
    arguments that are not read because they were fixed before the kernel ran.
    The signature is worked out from the operands the kernel was written
    against -- a pointer's type is the operand's type, an extent's type is
    however wide it has to be -- and anything the kernel declares as a constant
    goes among the constants rather than among the types.

    A configuration that will not compile is a configuration that cannot be
    measured rather than a failure of the whole table: the other candidates are
    still answers, and this one is not one of them.  So a refusal is answered
    with nothing and the caller moves on.
    """

    from ..codegen.triton_conv import _build
    from ..runtime.hints import DeviceProperties
    from ..runtime.triton_compat import Config
    from ..runtime.triton_heuristics import CachingAutotuner

    try:
        fn = _build(
            source, f"<tensorplay-stax-{template.name}>", template.symbol
        )
        # What the kernel takes is what its own signature says.  A value
        # chosen when the text was written is an argument the compiler folds,
        # so it is in the signature as a constant and its value is among the
        # constants -- the two halves of saying the same thing.
        signature = _signature_of(fn.arg_names, template.operands)
        # A constant is a value, not a spelling of one.  Several of the values
        # a configuration carries are named in the language the kernel is
        # written in -- the index type, the type a sum is accumulated in -- and
        # the compiler wants the thing itself rather than the name of it.
        all_constants = {
            name: _language_type(value) if _is_language_name(value) else value
            for name, value in constants.items()
        }
        all_constants.setdefault("INDEX_DTYPE", _language_type(template.index_dtype))
        triton_meta = {
            "device": DeviceProperties.create(inductor_meta["device"]),
            "signature": signature,
            "constants": {
                name: value
                for name, value in all_constants.items()
                if name in signature
            },
        }
        tuner = CachingAutotuner(
            fn,
            triton_meta,
            [
                Config(
                    {
                        k: v
                        for k, v in config.items()
                        if k not in signature
                    },
                    num_warps=int(config.get("num_warps", 4)),
                    num_stages=int(config.get("num_stages", 2)),
                )
            ],
            save_cache_hook=None,
            mutated_arg_names=[],
            optimize_mem=False,
            heuristic_type=None,
            inductor_meta=inductor_meta,
        )
        return tuner._precompile_config(tuner.configs[0])
    except Exception:
        log.info(
            "could not compile %s at this configuration",
            template.name, exc_info=True,
        )
        return None


def _param_name(name: str) -> str:
    """The name the parameter carrying an operand goes by.

    A body refers to an operand by one name; the parameter that carries it is
    named after that, with a pointer's suffix when the body did not already
    use one.  One spelling, used by the signature, by the extents and strides
    it carries, and by anything asking about them, so that nothing has to
    translate between them.
    """

    return name if name.endswith("_ptr") else f"{name}_ptr"


def _launcher_tail(layout, input_nodes, arg_names) -> tuple:
    """What a launcher is handed after the operands, in the signature's order.

    The signature names the extents and the strides, and the order it names
    them in is the order they are given in -- a launcher is a call written
    before the values were known, so it cannot work that out for itself.  The
    names say which operand each belongs to, so each is looked up in that
    operand's own extents rather than counted off in sequence, and the stream
    the launch runs on is the last of them.
    """

    import tensorplay as tp

    def extents_of(node):
        """An operand's extents and strides, whichever kind of thing it is.

        A value the compiler has lowered knows its own extents and strides; one
        handed over from outside knows only its shape, and a value that knows
        neither is recorded as knowing nothing, so that asking about it fails
        here rather than at a launch.
        """

        if hasattr(node, "get_size"):
            return [int(v) for v in node.get_size()], [
                int(v) for v in node.get_stride()
            ]
        def read(name):
            """A value that may be spelled as an attribute or as a method.

            One describes itself by its shape and its strides; the other is
            asked.  Both spellings mean the same thing, so which one a value
            uses is not something the caller has to know.
            """

            value = getattr(node, name, None)
            if value is None:
                return None
            return value() if callable(value) else value

        shape = read("shape")
        if shape is None:
            shape = read("size")
        if shape is not None:
            shape = [int(v) for v in shape]
            known = read("stride")
            if known is not None:
                return shape, [int(v) for v in known]
            strides: list[int] = []
            running = 1
            for extent in reversed(shape):
                strides.append(running)
                running *= max(extent, 1)
            return shape, list(reversed(strides))
        raise AssertionError(
            f"an operand of this launch knows neither its shape nor its "
            f"extents: {node!r}"
        )

    by_name: dict = {}
    for index, node in enumerate(input_nodes):
        letter = chr(ord("A") + index)
        by_name[letter] = node
    by_name["C"] = layout
    sizes_of: dict = {}
    strides_of: dict = {}
    for letter, node in by_name.items():
        sizes_of[letter], strides_of[letter] = extents_of(node)

    tail: list = []
    size_seen: dict = {}
    stride_seen: dict = {}
    for name in arg_names:
        if name.startswith("size_"):
            # The name says which operand and which of its extents: the operand
            # is the letters up to the last digit, and the extent is which
            # digit it ends with.  The operand is named by the parameter, so
            # whichever spelling the operands were declared under is the one
            # looked for.
            # The name is the operand's own name with its pointer's suffix and
            # the index of the extent, so the operand is what is left once
            # those are taken off -- in whichever of the two spellings the
            # operands were declared.
            stem = name[len("size_"):-1]
            operand = next(
                (
                    letter
                    for letter in by_name
                    if stem.lower() in (letter.lower(), letter.lower() + "_ptr")
                ),
                None,
            )
            if operand is None:
                raise AssertionError(
                    f"the signature names an extent of {stem!r}, which is not "
                    f"one of this kernel's operands: {sorted(by_name)}"
                )
            rank = size_seen.get(operand, 0)
            size_seen[operand] = rank + 1
            tail.append(sizes_of[operand][rank])
        elif name.startswith("stride_"):
            stem = name[len("stride_"):-1]
            operand = next(
                (
                    letter
                    for letter in by_name
                    if stem.lower() in (letter.lower(), letter.lower() + "_ptr")
                ),
                None,
            )
            if operand is None:
                raise AssertionError(
                    f"the signature names a stride of {stem!r}, which is not "
                    f"one of this kernel's operands: {sorted(by_name)}"
                )
            rank = stride_seen.get(operand, 0)
            stride_seen[operand] = rank + 1
            tail.append(strides_of[operand][rank])

    stream = None
    device = getattr(layout, "device", None)
    if device is not None and getattr(device, "type", "cpu") != "cpu":
        try:
            stream = tp.cuda.current_stream(device).cuda_stream
        except Exception:
            stream = None
    tail.append(stream)
    return tuple(tail)


def _cdiv(a, b):
    """The number of whole steps of ``b`` that cover ``a``."""

    return -(-a // b)


def _call_sizes(layout, input_nodes, constexprs):
    """The extents a grid is a function of, in the order it declares them.

    A grid is a function of what is being computed, so the extents come from
    the result's layout and from the operands' -- and the tile comes from the
    configuration, which the caller passes alongside.
    """

    sizes: list = []
    for index in (0, 1):
        node = input_nodes[index] if len(input_nodes) > index else None
        sizes.append(
            int(node.get_size()[0])
            if node is not None and hasattr(node, "get_size")
            else int(layout.size[0])
        )
    return tuple(sizes)


def _is_language_name(value) -> bool:
    """Whether this value is written as a name in the kernel-writing language.

    A type the body is told about -- the width an index is narrowed to, the
    type a sum is accumulated in -- travels through the configuration as the
    name the body knows it by, because that is the only spelling a template
    can carry.  The compiler wants the type itself, so such a name is
    recognised here rather than by every value being looked up.
    """

    return isinstance(value, str) and value.startswith("tl.")


def _language_type(name: str):
    """A type as the kernel-writing language spells it, not as text.

    A constant handed to the compiler is a value, not a spelling of one: the
    text "tl.int64" is a name to look up, and the type it names is what a
    kernel narrows an index to.
    """

    from ..runtime.triton_compat import tl

    return getattr(tl, str(name).split(".")[-1], None)


def _signature_of(arg_names, operands: dict) -> dict:
    """What the runtime is told about each argument of a rendered kernel.

    An argument is one of three things, and the three are said differently: a
    pointer says the type of what it points at, an extent or a stride says how
    wide it has to be to hold any value the computation can produce, and a
    constant says nothing at all -- its value is among the constants, and its
    type is that it is not read.
    """

    from ..utils import _type_of

    signature: dict = {}
    for name in arg_names:
        if name.endswith("_ptr"):
            # The parameter is named after the operand, which may or may not
            # already carry the suffix itself; the operand is whichever of the
            # two spellings was declared.
            stem = name[:-4]
            operand = stem if stem in operands else name
            dtype = (operands.get(operand) or {}).get("dtype")
            # An operand whose type is not known is said as a byte pointer
            # rather than guessed at: the type decides how the kernel reads
            # what it points at, and a wrong one is a wrong answer rather than
            # a refused compile.
            resolved = getattr(tp, dtype, None) if dtype else None
            signature[name] = _type_of(resolved) if resolved is not None else "*i8"
        elif name.startswith(("size_", "stride_")):
            signature[name] = "i64"
        else:
            signature[name] = "constexpr"
    return signature


class TritonTemplate(KernelTemplate):
    """A template whose body is source text and whose launch is a grid.

    Everything a choice needs is settled by the time the template is built: what
    it is called, what it is made of, and how it is launched.  Holding the text
    rather than a reference to a builder is what lets two templates that wrote
    the same text be recognised as the same kernel, and it is why the text's
    hash -- not the template's -- is what identifies the kernel downstream.

    A template built twice under one name with one source is one template
    arriving twice, so the second is the same object rather than a rival: a
    module reached along two paths in one process should not look like two
    different kernels competing to be measured.
    """

    #: The kind of kernel this template builds; a subclass may name another.
    kernel_type: type[Any] = TritonTemplateKernel
    #: Whether a build is checked against the cache instead of trusted to it.
    #: Off by default because checking costs a second build, and a second
    #: build is exactly what a caller in a hurry is trying to avoid.
    test_cache: bool = False
    index_counter = itertools.count()
    all_templates: dict = {}

    @staticmethod
    def jinja2_env():
        """The environment kernel sources are rendered in.

        Strict, because a kernel is not a document: a name the template did
        not define is a bug in the template or in what was passed to it, and
        rendering around it would produce a kernel that runs and is wrong.
        """

        import jinja2

        return jinja2.Environment(undefined=jinja2.StrictUndefined)

    @classmethod
    def from_file(cls, name: str, grid=None, path=None, symbol: str = "",
                  file: str | None = None, **kwargs: Any):
        """A template whose source lives in a file, rendered when it is used.

        A kernel is long enough that keeping it inside a string makes the two
        things that matter -- the code and the parts that vary -- hard to tell
        apart.  A file lets the varying parts be marked as varying, and lets an
        editor see the kernel as code.

        The file's name and the template's name are two separate facts: a file
        is named for the dialect it is written in, while a template answers to
        what the operation is called.  So the file is named by ``file`` when the
        two differ, and otherwise by the template's own name.
        """

        import jinja2

        try:
            import jinja2  # noqa: F811
        except ImportError as e:  # pragma: no cover - depends on environment
            raise AssertionError("requires jinja2") from e
        if path is None:
            from ..codegen import templates as _tpl_dir

            path = _tpl_dir.__path__[0] + f"/{file or name}.py.jinja"
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        return cls(name, grid=grid, source=source, symbol=symbol, **kwargs)

    def render(self, **params: Any) -> str:
        """This kernel's source with the given decisions filled in."""

        return self.jinja2_env().from_string(self.source).render(**params)

    def render_with(self, kernel_args, **params: Any) -> str:
        """Render with the operands in scope by name rather than by variable.

        A body asks for what it needs -- ``size("A", -1)``, ``def_kernel("A",
        "B")`` -- without being handed an object to ask.  The names resolve
        against the operands the kernel is written for, so the same text serves
        every shape those operands may take, and a body that refers to an
        operand nobody declared fails here rather than at launch.

        The index type travels with the operands rather than with the body: it
        is a property of the launch, and a body that narrows an index -- to hand
        a descriptor a 32-bit offset, say -- has to name the type it narrows
        to.  So it is in scope under the name the bodies write it by, and a
        body that uses it without an operand record to say which type fails
        here rather than in the kernel.
        """

        env = self.jinja2_env()
        # The body names its operands but not itself: the entry point's name is
        # a property of the template, so it is supplied here rather than asked
        # for, and a body cannot introduce a second name for the same function.
        kernel_args.symbol = self.symbol
        env.globals["def_kernel"] = kernel_args.def_kernel
        env.globals["size"] = kernel_args.size
        env.globals["stride"] = kernel_args.stride
        env.globals["dtype"] = getattr(kernel_args, "dtype", None)
        return env.from_string(self.source).render(
            index_dtype=kernel_args.index_dtype, **params
        )

    def __init__(
        self,
        name: str,
        grid=None,
        source: str = "",
        symbol: str = "",
        debug: bool = False,
        cache_codegen_enabled_for_template: bool = False,
        prologue_loads_all_inputs: bool = False,
        always_freeze_layout: bool = False,
    ) -> None:
        super().__init__(name, hash=hashlib.sha256(source.encode("utf-8")).hexdigest())
        self.grid = grid
        self.source = source
        #: The name the source defines its entry point under.  Named rather
        #: than assumed, because the symbol is part of the text: a source that
        #: defines something else needs no rewriting to be used here.
        self.symbol = symbol or name
        self.debug = debug
        self._cache_codegen_enabled_for_template = cache_codegen_enabled_for_template
        self._generated_code_cache: GeneratedCodeCache = clear_on_fresh_cache(
            GeneratedCodeCache()
        )
        self.prologue_loads_all_inputs = prologue_loads_all_inputs
        self.always_freeze_layout = always_freeze_layout
        #: The operands the last rendering was written against, by name, with
        #: the geometry and the type of each.  Filled in when a rendering
        #: happens and read when that rendering is compiled.
        self.operands: dict = {}
        #: The width an index is narrowed to, as the language spells it.  It is
        #: a property of the launch rather than of the text, and the compiled
        #: form has to be given it as well as the text.
        self.index_dtype: str = "tl.int64"
        existing = self.all_templates.get(name)
        if existing is not None and existing.src_hash != self.src_hash:
            raise AssertionError(
                f"two different kernels are both called {name}"
            )
        self.all_templates[name] = self

    @property
    def uid(self) -> str:
        """Namespaced by kind, so two kinds may share a name."""

        return f"triton::{self.name}"

    @property
    def template(self) -> "TritonTemplateKernel":
        """This template's source in the form that can be run.

        What a caller wants from a template that owns source text is the thing
        the text becomes, and that is the same for every configuration of this
        template -- so it is made once here and every configuration shares it.
        That sharing is the point: two configurations that wrote the same text
        are the same kernel, and a cache that could tell them apart would be
        keeping two of something there is only one of.
        """

        return self.kernel_type.get(self.name, self.source, self.symbol, self.grid)

    def argnames(self) -> list[str]:
        """The operand names the body asked for, in the order it asked for them.

        A body names its operands in the kernel it writes, and those names are
        what the rest of the template has to call them by: a body that asks for
        a name no operand was given fails here rather than producing a kernel
        that reads the wrong buffer.  Read out of the source rather than
        declared beside it, because a name written in the body and a name
        recorded elsewhere can disagree and then the disagreement is invisible.
        """

        import re

        found = re.search(r"def_kernel\(([^)]*)\)", self.source)
        if found is None:
            return ["A"]
        names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", found.group(1))
        return names or ["A"]

    def generate(self, **kwargs: Any):
        """The choice this template makes for one set of arguments.

        Refuses rather than raising when the arguments do not describe a launch
        this template can perform, so that a whole table can be offered at once
        and the entries that do not apply dropped without unwinding the rest.
        """

        if self.source == "":
            return None
        grid_fn = self.grid
        layout = kwargs.get("layout")
        input_nodes = kwargs.get("input_nodes") or ()
        if layout is not None:
            meta = {k: v for k, v in kwargs.items()
                    if k not in ("layout", "input_nodes")}
            meta.setdefault("out_size", tuple(int(s) for s in layout.size))
            names = self.argnames()[:len(input_nodes)] or list(self.argnames())[:1]
            operands = {}
            for name, node in zip(names, input_nodes):
                # A lowered value knows its own extents and strides; something
                # handed over from outside knows only its shape, and a value
                # that knows neither is recorded as knowing nothing so that a
                # body asking about it fails at render rather than at launch.
                shape = node.get_size() if hasattr(node, "get_size") else getattr(node, "shape", ())
                if hasattr(node, "get_stride"):
                    stride = node.get_stride()
                else:
                    stride = contiguous_strides(tuple(int(v) for v in shape))
                dt = node.get_dtype() if hasattr(node, "get_dtype") else getattr(node, "dtype", "float32")
                operands[name] = {
                    "shape": tuple(int(v) for v in shape),
                    "stride": tuple(int(v) for v in stride),
                    "dtype": str(dtype_name(dt)),
                }
            outputs = {
                "C": {
                    "shape": tuple(int(s) for s in layout.size),
                    "stride": tuple(int(s) for s in layout.stride),
                    # The result's own type, and the type of what it is read
                    # from: a product reads two operands and writes their type.
                    "dtype": str(
                        dtype_name(
                            layout.dtype
                            if layout.dtype is not None
                            else next(
                                (
                                    node.get_dtype()
                                    for node in input_nodes
                                    if hasattr(node, "get_dtype")
                                ),
                                None,
                            )
                        )
                    ),
                }
            }
            meta.update(kwargs)
            # What the kernel is written against and what the launch is written
            # against are two different lists.  A body names a tile, an
            # accumulation type, a bound: values chosen when the text was
            # written, which are parameters of the kernel because they cannot
            # change while it runs.  Everything else -- how the result is laid
            # out, what it is called, how many programs to start -- describes
            # the launch and is handed to the launch instead, because putting
            # it in the signature would make the kernel take an argument it
            # never reads, and a launch option the runtime also wants would be
            # asked for twice under the same name.
            # A configuration is what the kernel was written for; what the call
            # is about -- where its result goes, which values it was handed --
            # is what the launch is for.  The first is a constant of the module
            # the kernel is defined in; the second is the launch's own business
            # and would be a name the body could read but has no meaning for.
            constexprs = {
                k: v for k, v in kwargs.items()
                if k not in ("layout", "input_nodes", "out_size")
            }
            # Kept on the template for the compile that follows: what each
            # operand is decides the type of the argument the kernel takes for
            # it, and the operands are settled here rather than asked again.
            self.operands = {**operands, **outputs}
            rendered = self.render_with(
                KernelArgs(operands, outputs, constexprs), **meta
            )
        else:
            meta = {k: v for k, v in kwargs.items()
                    if k not in ("layout", "input_nodes")}
            rendered = self.render(**meta)
        # The rendered text has to be compiled before it can be launched, and
        # the compiled form cannot be called directly: what the runtime hands
        # back is only callable from inside a launch it has set up itself, with
        # the grid worked out, the arguments put in the order its signature
        # declares, and the stream passed along.  So the launch is written out
        # from the compiled form together with what it was compiled for, and
        # that is what the choice is bound to.
        # How many programs to start is a function of the extents being
        # computed and of the tile they are split into, so it is worked out
        # here, once, rather than asked of the kernel at every launch.  The
        # kernel takes operands; how many programs to start is not something an
        # operand can say.
        # The configuration as the grid function takes it: the values chosen
        # for this call, and nothing about the call itself -- a grid is a
        # function of what is being computed and of the tile it is split into,
        # and where the result lands is neither.
        extents = self.grid(
            *_call_sizes(layout, input_nodes, constexprs),
            {
                k: v
                for k, v in constexprs.items()
                if k not in ("layout", "input_nodes", "out_size")
            },
        )
        # What the launch is written against, kept so that a retry with one
        # thing changed can be compiled from the same record rather than
        # worked out again.
        inductor_meta = {
            "kernel_name": self.uid,
            "device": layout.device if layout is not None else "cuda",
            "grid_type": "FixedGrid",
            "fixed_grid": tuple(int(v) for v in extents),
        }
        result = _compile_rendered(
            self,
            rendered,
            config={
                k: v for k, v in kwargs.items()
                if k not in ("layout", "input_nodes", "out_size")
            },
            constants=dict(constexprs),
            inductor_meta={
                "kernel_name": self.uid,
                "device": layout.device if layout is not None else "cuda",
                "grid_type": "FixedGrid",
                "fixed_grid": tuple(int(v) for v in extents),
            },
        )
        if result is None:
            return None
        try:
            # Loading the binary is where a kernel that does not fit the device
            # says so -- a tile whose working set is larger than the shared
            # memory there is, or a register count the device cannot hold.  A
            # configuration that does not fit is one that cannot be measured,
            # not a failure of the table: the other candidates are still
            # answers, so it is dropped and the caller moves on.
            result.kernel._init_handles()
            launcher = result.make_launcher()
        except Exception:
            log.info(
                "could not load %s at this configuration", self.uid, exc_info=True
            )
            return None
        launcher.__name__ = self.uid
        # What a launcher written from this kernel is handed after the
        # operands: the extents and strides it declares, in that order, and the
        # stream it is launched on.  Written once here, where the signature is
        # known, rather than at each measurement.
        launcher_args = _launcher_tail(
                layout, input_nodes, result.kernel.src.fn.arg_names
            )
        caller = TritonChoiceCaller(
            name=self.uid,
            input_nodes=input_nodes,
            layout=layout,
            description=repr(sorted((k, repr(v)) for k, v in kwargs.items())),
            source=self.source,
            src_hash=self.src_hash,
            launcher_args=launcher_args,
            num_stages=int(kwargs.get("num_stages", 2)),
            num_warps=int(kwargs.get("num_warps", 4)),
            config={k: v for k, v in kwargs.items()},
            operands=dict(self.operands),
            inductor_meta=inductor_meta,
            template=self,
        )
        # A grid is a function of the extents being computed and of the tile
        # they were split into, so both are computed once here rather than
        # asked of the kernel at every launch: the kernel takes operands, and
        # how many programs to start is not something an operand can say.
        return caller.bind(launcher)

    def choice_or_none(self, **kwargs: Any):
        """This template's choice for one configuration, or nothing if it does not fit.

        The question is asked the same way of every candidate in a list, whether
        it is a template the compiler could write or an operation the framework
        already has, so that a list is offered as a whole and the entries that
        do not apply to this call are dropped without unwinding the rest.
        """

        try:
            return self.generate(**kwargs)
        except NotImplementedError:
            return None

    def maybe_append_choice(self, choices: list, **kwargs: Any):
        """Add this template's choice to a list, or report why there is none."""

        try:
            choice = self.generate(**kwargs)
            if choice is not None:
                choices.append(choice)
            return None
        except NotImplementedError as e:
            return e

    def generate_and_load(self, generate_with_caching: bool = True, **kwargs: Any):
        """Write this configuration's kernel and load it.

        Generating and loading are one step here rather than two, because the
        load is what the generate is for.  Whether the written form is kept at
        all is the caller's decision: a caller enumerating a table wants every
        entry written, because it is about to measure them, while a caller
        measuring a second time would rather have a fresh one.
        """

        built = self.generate(**kwargs)
        if self.test_cache and built is not None:
            # A cache that hands back something different from what it was
            # given is worse than no cache, because the difference is invisible
            # until a number is wrong.  So with the switch on, the same call is
            # made twice and the two answers compared.
            again = self.generate(**kwargs)
            if getattr(again, "src_hash", None) != getattr(built, "src_hash", None):
                raise AssertionError(
                    f"{self.name} built two different ways from one source"
                )
        return built


class KernelArgs:
    """The operands a kernel is written against, held by name.

    A kernel's body refers to its operands by name and asks for their extents
    and strides rather than being handed them: a body that reads ``size("A",
    -2)`` says what it wants, and the answer comes from whoever holds the
    operand.  That separation is what lets one body serve several shapes -- a
    batched product and a plain one differ in which extents they ask for, not in
    the text between the asks.

    The signature is emitted here too, from the same names, so a body cannot
    disagree with its own parameters: the arguments it may refer to and the
    arguments it is given are the same list, read twice.
    """

    def __init__(self, inputs: dict, outputs: dict, constexprs: dict | None = None,
                 index_dtype: str = "tl.int64"):
        self.inputs = {k: dict(v) for k, v in inputs.items()}
        self.outputs = {k: dict(v) for k, v in outputs.items()}
        self.constexprs = dict(constexprs or {})
        self.index_dtype = index_dtype
        #: The constants the body reads by name, written into the text the
        #: first time a signature is asked for.
        self.defines = ""
        #: The entry point's name, supplied by whatever renders this.
        self.symbol = ""

    def call_sizes(self) -> tuple:
        """The extents a launch is sized by, and the configuration it was given.

        A grid is a function of what is being computed and of the tile the
        computation was split into, so both are handed together: a grid asked
        for the extents alone would have to guess the tile, and one asked for
        the tile alone would have to guess the extents.  The extents come first
        because that is the order a grid function declares them in.
        """

        meta = {k: v for k, v in self.constexprs.items()}
        for name, spec in self.outputs.items():
            meta.setdefault(name, spec)
        return tuple(spec["shape"] for name, spec in self.outputs.items()), meta

    def _operand(self, name: str) -> dict:
        # An operand is declared under one name and a body may ask about it
        # under either that name or the name its value is read by; both mean
        # the same operand, so both are looked for before either is refused.
        for candidate in (name, name.removesuffix("_ptr"), f"{name}_ptr"):
            if candidate in self.inputs:
                return self.inputs[candidate]
            if candidate in self.outputs:
                return self.outputs[candidate]
        raise KeyError(
            f"{name} is not one of this kernel's operands: "
            f"{sorted(self.inputs) + sorted(self.outputs)}"
        )

    def size(self, name, dim: int) -> str:
        """One extent of one operand, as the kernel's index type.

        Negative dimensions count from the end, which is what lets one body ask
        for ``-1`` and mean the contiguous axis whatever the rank is.  Naming
        nothing asks about the result rather than an input, which is what a body
        wants when it is working out the shape of what it is producing.
        """

        if name is None:
            name, spec = next(iter(self.outputs.items()))
            shape = spec["shape"]
        else:
            shape = self._operand(name)["shape"]
        index = dim if dim >= 0 else len(shape) + dim
        if not 0 <= index < len(shape):
            raise IndexError(
                f"{name} has {len(shape)} dimensions; {dim} is not one of them"
            )
        return f"size_{_param_name(name)}{index}"

    def stride(self, name, dim: int) -> str:
        """One stride of one operand, by name rather than by position.

        Strides are passed as arguments rather than baked in, so a body that
        asks for one does not have to be re-rendered per layout: the kernel is
        told the layout at launch, and the same text serves every layout.

        Naming nothing asks about the result rather than an input, which is what
        a body wants when it is working out how its own result is addressed.
        """

        if name is None:
            name = next(iter(self.outputs))
        return f"stride_{_param_name(name)}{dim}"

    def def_kernel(self, *names: str, symbol: str | None = None, outputs: tuple = ()) -> str:
        """The decorator and the signature, written from the operand names.

        Every operand named here becomes a parameter, in the order named, and
        every operand *not* named is not a parameter -- so a body that refers
        to something it did not ask for fails at render time rather than at
        launch, where the mistake would look like a kernel fault.
        """

        symbol = symbol or self.symbol or "_kernel"
        parts = list(names) + list(outputs)

        for name in parts:
            self._operand(name)
        # One spelling per name, used everywhere: the parameter that carries an
        # operand, the extents it has, and the strides it is laid out with are
        # all named after it, so a body that asks for one of them is asking
        # about the operand and nothing has to translate between spellings.
        strides = []
        for name in parts:
            rank = len(self._operand(name)["shape"])
            strides += [f"stride_{_param_name(name)}{d}" for d in range(rank)]
        sizes = []
        for name in parts:
            rank = len(self._operand(name)["shape"])
            sizes += [f"size_{_param_name(name)}{d}" for d in range(rank)]
        # A value chosen when the text was written -- a tile, an accumulation
        # type, a bound -- cannot change while the kernel runs, and the kernel
        # is compiled once for the configuration it came from.  So it is a
        # parameter the compiler may fold rather than one passed on every
        # launch, and it is annotated as such where the compiler reads that.
        # It has to be a parameter rather than something the body reads from
        # the module it is defined in: a global is not a value the body can
        # fold, and the arithmetic on a tile would be worked out at every step
        # instead of once.
        names = [*names, *outputs]
        # The index type travels with the operands rather than with the body: it
        # is a property of the launch, and a body that narrows an index has to
        # name the type it narrows to.  It is a parameter for the same reason
        # the rest are -- the compiler folds it, and the body reads it by name.
        tail = [f"{n}: tl.constexpr" for n in self.constexprs]
        tail.append("INDEX_DTYPE: tl.constexpr")
        # An operand is declared under one name and referred to by it.  A body
        # that says "A" means the parameter named A, so that is the name given
        # -- a short name is spelled the way the body spells it, and one the
        # body spells with its pointer suffix keeps it.  A body that goes on to
        # assign a name it was not given is therefore assigning a name of its
        # own, which is what it meant to do.
        all_params = [_param_name(n) for n in names] + sizes + strides + tail
        binds = "".join(
            f"    {n} = {_param_name(n)}\n"
            for n in names
            if not n.endswith("_ptr")
        )
        return (
            "@triton.jit\n"
            f"def {symbol}({', '.join(all_params)}):\n"
            f"{binds}"
        )



def rand_strided(
    size,
    stride,
    dtype=tp.float32,
    device="cpu",
    extra_size: int = 0,
):
    """A value of a given geometry, holding values rather than nothing.

    Filled rather than left empty because a candidate may read the place it is
    going to write -- accumulating into it, say -- and an empty place would make
    every candidate that does so compute from whatever was in memory, which is
    not the same measurement twice running.  Random rather than fixed for the
    same reason: a candidate that branches on what it reads would be measured on
    one branch, and there is no way to know which.

    The buffer behind it is as large as the geometry reaches and no larger, since
    a strided view of a smaller buffer would read past it.
    """

    needed_size = extra_size
    if all(s > 0 for s in size):
        # only need to allocate if all sizes are non-zero
        needed_size += (
            sum((shape - 1) * stride for shape, stride in zip(size, stride)) + 1
        )
    if dtype.is_floating_point:
        if dtype.itemsize == 1:
            # there is no normal distribution for the eight-bit float types, so
            # the values are made in a wider type and cast: which values they are
            # does not matter, only that they are values
            buffer = tp.randn(needed_size, dtype=tp.float16, device=device).to(
                dtype=dtype
            )
        else:
            buffer = tp.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = tp.zeros(size=[needed_size], dtype=dtype, device=device)
    return tp.as_strided(buffer, size, stride)


class NoValidChoicesError(RuntimeError):
    """Nothing was offered to choose from.

    Its own kind of failure because the way out of it is not a fallback: a call
    with no candidates has no correct answer, and the only thing to do about it
    is to offer something.  So the message says what was empty rather than
    reporting a crash somewhere downstream of it.
    """


class BenchmarkTensors(NamedTuple):
    """The values one measurement is made against.

    Inputs and the place the result goes, held together because a measurement
    that used one call's inputs and another's output would be measuring neither.
    """

    inputs: list
    output: Any

    def unpack(self) -> tuple[list, Any]:
        return self.inputs, self.output


class AutotuneArgs(NamedTuple):
    """Everything a measurement needs that is not the choice being measured.

    Two sets of inputs travel here rather than one, because the operation's own
    kernel and a template are sometimes handed the same values in different
    shapes -- a bias of one value against a bias of a row, say.  Measuring both
    against the same values means the difference in their times is a difference
    in their kernels and not in what they were handed.

    The expected result is here too, because a candidate has to be checked as
    well as timed: the fastest wrong answer is not an answer.
    """

    example_inputs: list
    example_inputs_extern: list
    out: Any
    out_extern: Any
    expected: Any = None

    @staticmethod
    def from_choice_args(
        example_inputs, example_inputs_extern, out, out_extern, expected
    ) -> "AutotuneArgs":
        return AutotuneArgs(
            example_inputs, example_inputs_extern, out, out_extern, expected
        )

    def get_benchmark_tensors(self, is_extern: bool) -> BenchmarkTensors:
        if is_extern:
            return BenchmarkTensors(self.example_inputs_extern, self.out_extern)
        return BenchmarkTensors(self.example_inputs, self.out)


def create_inputs_key(input_nodes) -> str:
    """What the inputs are, said so that two calls with the same inputs agree."""

    return repr([AlgorithmSelectorCache.key_of(x) for x in input_nodes])


def should_use_layout_constraints(x: Any) -> bool:
    """Whether a value's layout was pinned rather than left to be inferred.

    Asked per value because one call's inputs can be pinned and another's not,
    and a stored decision made under a pin must not be reused without it.
    """

    return getattr(x, "layout", None) is not None and getattr(
        getattr(x, "layout", None), "stride", None
    ) is not None


def get_strides_with_layout_constraints(x: Any):
    """A value's strides, from its layout when it has one and its shape when not.

    The two say different things: a layout's strides are where the value is, and
    a shape's are how it would be laid out.  A measurement made against one and
    stored against the other would be reused for a call it was not made for.
    """

    if should_use_layout_constraints(x):
        return x.layout.stride
    return x.get_stride()


def filter_choices_by_name_regex(choices: list) -> list:
    """The candidates whose names are wanted, when a name pattern was given.

    A pattern is a way of asking for one candidate to be measured without the
    others being taken off the list, so a pattern that matched nothing leaves the
    list as it was rather than leaving it empty.
    """

    import re

    regex = config.autotune_choice_name_regex
    if regex is None:
        return choices
    matcher = re.compile(regex)
    return [choice for choice in choices if matcher.search(choice.name)]


def filter_choices_by_desc_regex(choices: list) -> list:
    """The candidates whose descriptions are wanted, for the same reason."""

    import re

    regex = config.autotune_choice_desc_regex
    if regex is None:
        return choices
    matcher = re.compile(regex)
    return [choice for choice in choices if matcher.search(choice.description)]


class AlgorithmSelectorCache:
    """Which candidate ran, remembered from one call to the next.

    A measurement is expensive and a call is not, so what was measured is kept
    and looked up rather than measured again.  The key is what the inputs *are*
    -- their device, their types, their sizes, their strides -- and deliberately
    not where the result lands: two calls with the same inputs and different
    destinations pick the same way of computing it, and a decision that
    distinguished them would be a decision made twice for no reason.

    The candidates themselves are also filtered before they are measured, by a
    coarse pass and then a fine one, because measuring every configuration of
    every template to find the best one would cost more than the compile it is
    meant to save.
    """

    def __init__(self) -> None:
        # A lowering is not necessarily the first thing to ask about a given set
        # of inputs, so the record of what was precompiled for those inputs is
        # shared by every lowering that reaches them.
        self.prescreening_cache: dict[str, list] = {}
        self.feedback_saver_fns: list = []
        self.preprocessing_fns: list = [
            filter_choices_by_name_regex,
            filter_choices_by_desc_regex,
        ]
        clear_on_fresh_cache(self)

    def cache_clear(self) -> None:
        self.prescreening_cache.clear()

    def pick_deterministic_choice(self, choices: list) -> ChoiceCaller:
        """The candidate to use when nothing may be measured.

        The operation's own kernel when it is among the candidates, because it is
        the one whose answer does not depend on which machine this is.  A caller
        that asked for determinism is asking for the same answer twice, and the
        only candidate that gives it is the one that was written once.
        """

        if len(choices) < 2:
            raise AssertionError(f"expected at least 2 choices, got {len(choices)}")
        externs = [
            choice for choice in choices if isinstance(choice, ExternKernelCaller)
        ]
        if len(externs) > 0:
            return externs[0]
        else:
            return choices[0]

    @staticmethod
    def _is_extern(choice: ChoiceCaller) -> bool:
        """Whether a candidate is the operation's own rather than a template's.

        Asked per candidate because the two are measured differently: one has to
        be built and the other is already there, and a pass that treated them the
        same would spend the difference on nothing.
        """

        return isinstance(choice, (ExternKernelCaller, SubgraphChoiceCaller))

    @staticmethod
    def key_of(node):
        """The pieces of a value that a stored decision must agree with.

        Its device, its type, its sizes, its strides and where it starts.  What
        is left out is as deliberate as what is in: a decision does not depend on
        which node the value came from, nor on its name, nor on where the result
        of the call goes -- so two calls that differ only in those share a
        measurement, which is the whole reason for keeping one.
        """

        sizevars = V.graph.sizevars
        return (
            node.get_device().type,
            str(node.get_dtype()),
            *sizevars.optimization_hints(node.get_size()),
            *V.graph.sizevars.optimization_hints(
                get_strides_with_layout_constraints(node)
            ),
            sizevars.optimization_hint(node.get_layout().offset),
        )

    def create_no_valid_choices(self, name: str, reason: str) -> NoValidChoicesError:
        """The failure for a call nothing was offered for, saying what was empty."""

        return NoValidChoicesError(
            f"No choices to select. Provided reason: {reason} "
            f"please consider adding the operation's own kernel to the "
            f"candidates for {name} to allow at least one choice. "
        )

    def __call__(
        self,
        name,
        choices: list,
        input_nodes,
        layout,
        input_gen_fns: dict | None = None,
        precompilation_timeout_seconds: int = 60 * 60,
        return_multi_template: bool = False,
        best_config_future=None,
        is_collective: bool = False,
        min_speedup_threshold: float = 1.0,
        benchmark_with_cudagraphs: bool = False,
    ):
        """The node and the candidate that ran, out of everything offered.

        One candidate is not a choice, so it is used without being measured: a
        measurement costs more than the answer and would find that the only
        candidate is the fastest, which is what was already known.  The rest are
        measured, and the fastest becomes the answer unless determinism was asked
        for, in which case the operation's own kernel does.
        """

        for preprocessing_fn in self.preprocessing_fns:
            choices = preprocessing_fn(choices)

        if benchmark_with_cudagraphs:
            for choice in choices:
                choice._benchmark_with_cudagraphs = True

        # A candidate that needs its inputs to be particular values cannot be
        # measured against values that were made up, so the deferred path is off
        # for those and the answer is one template rather than several.
        if input_gen_fns is not None:
            return_multi_template = False

        if len(choices) == 0:
            raise self.create_no_valid_choices(name, "No choices exist for backend.")
        log.debug("Max autotune selects from %s choices.", len(choices))

        if len(choices) == 1:
            node = choices[0].output_node()
            return node, choices[0]

        if config.deterministic:
            choice = self.pick_deterministic_choice(choices)
            node = choice.output_node()
            return node, choice

        if is_collective:
            benchmark_results = self.benchmark(
                choices, input_nodes, layout, input_gen_fns, is_collective=True
            )
        elif config.autotune_in_subproc:
            benchmark_results = self.benchmark(
                choices, input_nodes, layout, input_gen_fns
            )
        else:
            benchmark_results = self.benchmark(
                choices, input_nodes, layout, input_gen_fns
            )

        if min_speedup_threshold > 1.0:
            best_choice = min(benchmark_results, key=benchmark_results.get)
            if min_speedup_threshold * benchmark_results[best_choice] > benchmark_results[
                min(
                    (c for c in benchmark_results if self._is_extern(c)),
                    key=benchmark_results.get,
                    default=best_choice,
                )
            ]:
                # nothing beat the operation's own kernel by enough to be worth
                # the kernel that would have to be trusted instead
                best_choice = min(
                    (c for c in benchmark_results if self._is_extern(c)),
                    key=benchmark_results.get,
                    default=best_choice,
                )
            node = best_choice.output_node()
            return node, best_choice

        node = self.do_autotuning(
            name, input_nodes, layout, input_gen_fns, choices, benchmark_results
        )
        return node, benchmark_results

    def do_autotuning(
        self,
        name,
        input_nodes,
        layout,
        input_gen_fns,
        choices,
        benchmark_results,
    ):
        """The node the fastest candidate produced, or the operation's own.

        The floor is the operation's own kernel among the measured ones: a
        template that cannot be shown to be faster than the thing it replaces is
        not worth compiling, and the number it was measured at is what "faster"
        is compared against rather than a constant.
        """

        best = min(
            benchmark_results,
            key=lambda choice: (
                not self._is_extern(choice),
                benchmark_results[choice],
            ),
        )
        return best.output_node()

    def benchmark(
        self,
        choices,
        input_nodes,
        layout,
        input_gen_fns,
        hint_override: int | None = None,
        is_collective: bool = False,
    ):
        """How long each candidate takes, measured against the same values.

        Every candidate is measured on one set of values, made once and handed to
        each in turn, because a candidate measured against different values is
        not comparable to one measured against these.
        """

        log.debug("Starting autotuning")
        return self.benchmark_in_current_process(
            choices,
            input_nodes,
            layout,
            input_gen_fns,
            hint_override=hint_override,
            is_collective=is_collective,
        )

    @classmethod
    def benchmark_in_current_process(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns,
        hint_override: int | None = None,
        is_collective: bool = False,
    ) -> dict:
        """Measure the candidates here, where the values already are."""

        autotune_args = cls.get_inputs(
            choices, input_nodes, layout, input_gen_fns, hint_override=hint_override
        )
        return cls.benchmark_choices(choices, autotune_args, is_collective=is_collective)

    @classmethod
    def benchmark_choice(cls, choice: ChoiceCaller, autotune_args: AutotuneArgs) -> float:
        """How long one candidate takes on the values the others were measured on.

        The output is cleared first because a candidate that only partly writes
        it would otherwise be timed for writing the parts it did write, and the
        rest of the previous result would be counted as its work.
        """

        benchmark_tensors = autotune_args.get_benchmark_tensors(cls._is_extern(choice))
        inputs, output = benchmark_tensors.unpack()
        output.zero_()
        result = choice.benchmark(*inputs, out=output)
        device_type = next(
            (tensor.device.type for tensor in inputs if tensor.device.type == "cuda"),
            "cuda",
        )
        if device_type == "cuda" and tp.cuda.is_available():
            tp.cuda.synchronize()  # shake out any device errors
        return result

    @classmethod
    def benchmark_choices(
        cls,
        choices,
        autotune_args: AutotuneArgs,
        is_collective: bool = False,
    ) -> dict:
        """Every candidate's time, and the first one's failure if one fails.

        A candidate that cannot be built is a measurement of infinity rather than
        a failure of the whole thing: the other candidates are still answers, and
        the one that broke is not one of them.
        """

        timings: dict = {}
        for choice in choices:
            try:
                result = cls.benchmark_choice(choice, autotune_args)
                timings[choice] = float(result)
            except Exception:
                timings[choice] = float("inf")
        return timings

    @classmethod
    def get_inputs(
        cls,
        choices,
        input_nodes,
        layout,
        input_gen_fns: dict | None,
        hint_override: int | None = None,
    ) -> AutotuneArgs:
        """The values every candidate is measured against, made once.

        Made from the nodes rather than from the program, because the program is
        being compiled and has no values in it yet.  A candidate that asked for
        particular values has them made the way it asked, and one that did not is
        measured against values that are merely of the right shape -- which is
        enough to time it and not enough to be sure it is right, which is why the
        operation's own result is what a timed template is checked against.
        """

        if input_gen_fns is None:
            input_gen_fns = {}

        # one value per name, so that two nodes reading the same buffer are
        # measured against the same buffer and a candidate cannot win by being
        # handed something easier than its neighbour was
        unique_example_inputs = {
            x.get_name(): input_gen_fns.get(
                i,
                lambda x: cls.benchmark_example_value(x, hint_override=hint_override),
            )(x)
            for i, x in enumerate(input_nodes)
        }
        example_inputs = list(unique_example_inputs.values())

        extern_choice = next(
            (choice for choice in choices if cls._is_extern(choice)), None
        )
        extern_input_nodes = input_nodes
        if extern_choice is not None:
            if len(extern_choice.input_nodes) != len(input_nodes):
                raise AssertionError(
                    "extern_choice.input_nodes length must match input_nodes: "
                    f"{len(extern_choice.input_nodes)} != {len(input_nodes)}"
                )
            extern_input_nodes = extern_choice.input_nodes

        # The operation's own kernel is handed each value at the geometry its own
        # node says, which is not always the geometry the templates were handed:
        # a bias is one value to one and a row to the other.  So the two are kept
        # apart, and each is viewed onto the storage it needs -- a value whose
        # stride reaches past what was made for it would read past the end, so a
        # larger buffer is made rather than a kernel being handed a view it
        # cannot safely write.
        example_inputs_extern: list = []
        for i, input_node in enumerate(extern_input_nodes):
            input_tensor = unique_example_inputs[input_node.get_name()]
            base = input_tensor

            if i in input_gen_fns:
                sizes = tuple(input_tensor.size())
                strides = tuple(input_tensor.stride())
                storage_offset = input_tensor.storage_offset()
            else:
                sizes = V.graph.sizevars.optimization_hints_with_override(
                    input_node.get_size(),
                    hint_override=hint_override,
                )
                strides = V.graph.sizevars.optimization_hints_with_override(
                    get_strides_with_layout_constraints(input_node),
                    hint_override=hint_override,
                )
                storage_offset = V.graph.sizevars.optimization_hint_with_override(
                    input_node.get_layout().offset,
                    hint_override=hint_override,
                )

            needed_size = compute_required_storage_length(
                sizes, strides, int(storage_offset)
            )
            current_size = base.untyped_storage().size() // base.element_size()
            if needed_size > current_size:
                new_base = tp.empty(
                    needed_size, dtype=base.dtype, device=base.device
                )
                base = new_base.as_strided(
                    base.size(), base.stride(), base.storage_offset()
                )

            example_inputs_extern.append(
                tp.as_strided(base, sizes, strides, storage_offset)
            )

        out = cls.benchmark_example_value(layout, hint_override=hint_override)

        # the place the result goes may reach past what was made for it, and what
        # was made is what has to be big enough for what a kernel writes
        out_base = out
        out_offset = V.graph.sizevars.optimization_hint(layout.offset, fallback=0)
        needed_out_size = compute_required_storage_length(
            out.size(), out.stride(), out_offset
        )
        current_out_size = out_base.untyped_storage().size() // out_base.element_size()
        if needed_out_size > current_out_size:
            new_out_base = tp.empty(
                needed_out_size, dtype=out_base.dtype, device=out_base.device
            )
            out_base = new_out_base.as_strided(
                new_out_base.size(),
                new_out_base.stride(),
                new_out_base.storage_offset(),
            )

        out_extern = tp.as_strided(out_base, out.size(), out.stride(), out_offset)
        # What the answer is checked against, when it is going to be checked.
        # Produced by running a candidate rather than by trusting one, so it is
        # only produced when there is a candidate that can be run: the list
        # holds an operation the framework already has, whose kernel is a name
        # rather than something to call, and asking that for a result is asking
        # it to do the one thing it does not do.  Left as nothing, the check
        # simply does not happen rather than happening against a wrong answer.
        expected = None
        if VERIFY and choices:
            for candidate in choices:
                try:
                    candidate.benchmark(*example_inputs, out=out)
                except NotImplementedError:
                    continue
                expected = out.clone()
                break
        return AutotuneArgs.from_choice_args(
            example_inputs, example_inputs_extern, out, out_extern, expected
        )

    @classmethod
    def generate_example_value(
        cls, size, stride, device, dtype, extra_size, allocation_size=None
    ):
        """A value for a result, of the result's geometry and holding values.

        The random state is saved and put back because making a value here would
        otherwise move the program's own random numbers along: a program compiled
        twice would then get different results, and a measurement is not worth a
        result that depends on whether anything was timed first.

        A result that is larger than the part of it that is written is made at the
        larger size and then viewed at the smaller one, because a kernel that
        writes the whole of what it was given may write past the part that is read
        and there has to be somewhere for that to land.
        """

        state = tp.get_rng_state()
        try:
            if allocation_size is None or allocation_size == size:
                return rand_strided(
                    size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                )
            else:
                return rand_strided(
                    allocation_size,
                    stride,
                    device=device,
                    dtype=dtype,
                    extra_size=extra_size,
                ).as_strided(size, stride)
        finally:
            tp.set_rng_state(state)

    @staticmethod
    def benchmark_example_value(node, hint_override: int | None = None):
        """One node made into a value it can be measured on.

        A layout counts as a node here, because a result's geometry is known
        before anything has been computed and is what the place to write has to
        be made from.  A view is unwrapped first, because what a view holds is
        not what was read and a value made from the view's own description would
        be a different value.
        """

        if isinstance(node, Layout):
            node = _Buffer(name="fake", layout=node)
        original_dtype = node.get_dtype()
        if isinstance(node, BaseView):
            node = node.unwrap_view()

        result = AlgorithmSelectorCache.generate_example_value(
            V.graph.sizevars.optimization_hints_with_override(
                node.get_size(),
                hint_override=hint_override,
            ),
            V.graph.sizevars.optimization_hints_with_override(
                get_strides_with_layout_constraints(node),
                hint_override=hint_override,
            ),
            node.get_device(),
            node.get_dtype(),
            V.graph.sizevars.optimization_hint_with_override(
                node.layout.offset,
                hint_override=hint_override,
            ),
            V.graph.sizevars.optimization_hints_with_override(
                V.graph.get_allocation_size(node),
                hint_override=hint_override,
            ),
        )
        if result.dtype != original_dtype:
            result = result.view(original_dtype)
        return result

    def prescreen_choices(self, choices: list, name: str) -> list:
        """The candidates worth measuring at all, by a cheap pass.

        Measured rather than timed, because a candidate that is wrong cannot be
        the answer however fast it is, and finding that out by measurement would
        be too late.
        """

        return list(choices)

    def prune_choices_postscreen(self, choices: list, name: str, pruned: list) -> None:
        """Which candidates the coarse pass took out, recorded so it is not redone."""

        inputs_key = create_inputs_key([])
        self.prescreening_cache[inputs_key] = [
            choice.name for choice in pruned
        ]

    def log_results(self, name: str, choices: list, benchmark_results: dict) -> None:
        """What each candidate was measured at, when the answer is worth recording.

        Logged rather than returned because a caller that wanted the numbers
        would have had to measure everything, and the numbers are for whoever
        asks afterwards whether the chosen candidate was the right one.
        """

        if not log.isEnabledFor(logging.INFO):
            return
        log.info(
            "Autotuning %s over %s candidates:\n%s",
            name,
            len(choices),
            "\n".join(
                "\t%s: %s"
                % (
                    choice,
                    (
                        f"{benchmark_results[choice] * 1e3:6.1f}us"
                        if choice in benchmark_results
                        else "N/A"
                    ),
                )
                for choice in choices
            ),
        )

    def add_feedback_saver(self, fn) -> None:
        self.feedback_saver_fns.append(fn)

    def clear_feedback_savers(self) -> None:
        self.feedback_saver_fns.clear()

    def add_preprocessing_fn(self, fn) -> None:
        self.preprocessing_fns.append(fn)

    def clear_preprocessing_fns(self) -> None:
        """The candidates are filtered by name and by description before they are
        measured; those two are put back and anything a caller added is dropped,
        so a filtering that has gone stale cannot keep narrowing the list.
        """

        self.preprocessing_fns.clear()
        self.preprocessing_fns.extend(
            (filter_choices_by_name_regex, filter_choices_by_desc_regex)
        )


_ALGORITHM_SELECTOR_CACHE: AlgorithmSelectorCache | None = None


def get_algorithm_selector_cache() -> AlgorithmSelectorCache:
    """The one record of what was chosen, made when it is first needed.

    One rather than one per call because a measurement is only worth keeping if
    the next call can find it, and two records would each hold half the answers.
    """

    global _ALGORITHM_SELECTOR_CACHE
    if _ALGORITHM_SELECTOR_CACHE is None:
        _ALGORITHM_SELECTOR_CACHE = AlgorithmSelectorCache()
    return _ALGORITHM_SELECTOR_CACHE


def autotune_select_algorithm(*args, **kwargs):
    """Measure the candidates and hand back the one that ran.

    Every default a caller did not give is filled in here rather than at each
    call site, so that a caller who did not think about a setting gets the same
    one as a caller who did.
    """

    cache = get_algorithm_selector_cache()
    if "return_multi_template" not in kwargs:
        kwargs["return_multi_template"] = config.benchmark_epilogue_fusion
    if "precompilation_timeout_seconds" not in kwargs:
        kwargs["precompilation_timeout_seconds"] = 60 * 60
    return cache(*args, **kwargs)


def realize_inputs(*args):
    """The values as they must be handed to something outside this compiler.

    Realized and made to run along their last axis, because a caller that is not
    this compiler reads the values it is given and cannot be told what a stride
    means.
    """

    if len(args) == 1:
        realized = args[0]
        if not realized.has_tensor_output():
            return realized
        return realized.require_stride1()
    return [realize_inputs(x) for x in args]


def get_ktc(
    kernel_inputs: KernelInputs,
    template: Any,
    op_name: str,
    kwarg_overrides: dict | None = None,
):
    """The deferred candidates of one template for one call.

    Deferred because a table is offered whole and most of it is never built: the
    configurations are produced as a generator, and the kernel behind one is only
    built if that configuration is reached.  Building them all to choose between
    them would cost more than choosing does.

    The heuristic is asked three separate questions -- which configurations fit,
    what the inputs become, and what the kernel is told that is not a choice --
    because the answers change independently: a template may fit more shapes
    than it has useful configurations for, and may need to be told something
    about the call that is not one of its configurations.
    """

    device_type = kernel_inputs.device_type
    if device_type is None:
        raise AssertionError("get_ktc requires a valid device type")
    heuristic = get_template_heuristic(template, device_type, op_name)
    cs = heuristic.get_template_configs(kernel_inputs, op_name)
    inputs_val = heuristic.adjust_kernel_inputs(kernel_inputs, op_name)
    extra_kwargs = heuristic.get_extra_kwargs(kernel_inputs, op_name)
    overrides = kwarg_overrides or {}
    return make_ktc_generator(
        template=template,
        cs=cs,
        extra_kwargs=extra_kwargs,
        overrides=overrides,
        layout=kernel_inputs.output_layout()
        if hasattr(kernel_inputs, "output_layout")
        else None,
        inputs=inputs_val,
    )


def get_template_heuristic(template: Any, device_type: str, op_name: str):
    """The rule that says which configurations of a template are worth trying.

    Asked for by what a template calls itself, not by the template, so that a
    rule lives in the table beside every other rule rather than hanging off the
    thing it is a rule about.  A template and an operation's own kernel can then
    be told apart by name alone, which is what lets both go through the same
    question and the same answer.
    """

    return registry_get_template_heuristic(template.uid, device_type, op_name)


def get_template_configs(
    kernel_inputs: KernelInputs,
    templates: list,
    op_name: str,
    kwarg_overrides: dict | None = None,
) -> list:
    """The candidates of every template offered, as the built things among them.

    A template that declines a configuration contributes nothing rather than a
    refusal, and a template that declines all of them leaves the list with the
    other templates' candidates in it -- so a call is measured against whatever
    can actually be measured, and a template's absence is not a failure of the
    call.
    """

    if kwarg_overrides is None:
        kwarg_overrides = {}
    input_tensors = (
        kernel_inputs.nodes() if hasattr(kernel_inputs, "nodes") else []
    )
    if len(input_tensors) < 2 and not input_tensors:
        raise ValueError(
            f"Need at least 2 input tensors, got {len(input_tensors)}"
        )
    template_choices = {}
    for template in templates:
        template_choices[template.uid] = get_ktc(
            kernel_inputs, template, op_name, kwarg_overrides.get(template.uid, {})
        )
    adjusted_choices = _finalize_template_configs(
        template_choices, kernel_inputs, templates, op_name, kwarg_overrides
    )
    if _need_to_fix_layout(adjusted_choices, op_name):
        layout = (
            kernel_inputs.output_layout(flexible=False)
            if hasattr(kernel_inputs, "output_layout")
            else None
        )
        for ktc in adjusted_choices:
            ktc.layout = layout
            if hasattr(ktc, "_choice"):
                del ktc._choice
    return [ktc.choice for ktc in adjusted_choices if ktc.choice is not None]


def _finalize_template_configs(
    template_choices: dict,
    kernel_inputs: KernelInputs,
    templates: list,
    op_name: str,
    kwarg_overrides: dict | None = None,
) -> list:
    """Every template's candidates in one list, after any adjustment.

    Split out so that a caller may change the candidates before they are built,
    which is the only time a change is cheap: the incoming candidates are
    generators, and becoming a built kernel is what costs.
    """

    choices: list = []
    for choice_gen in template_choices.values():
        choices.extend(choice_gen)
    return choices


def _need_to_fix_layout(adjusted_choices: list, op_name: str) -> bool:
    """Whether the candidates have to be measured against a fixed result geometry.

    A candidate that can take a flexible geometry will take whichever was asked
    for, and one that cannot has to be told.  So this is asked only when it would
    change something: a measurement against a geometry no candidate can produce
    is a measurement of nothing.
    """

    if len(adjusted_choices) > 0:
        if adjusted_choices[0].inputs.device_type == "mps" and op_name not in (
            "mm",
            "addmm",
        ):
            return True
    if not (config.max_autotune or config.max_autotune_gemm):
        return False
    # a measured candidate is a built kernel, and a built kernel's geometry is
    # part of what it was built for
    return any(
        (not isinstance(ktc.template, ExternKernelCaller) for ktc in adjusted_choices)
    )



class PartialRender:
    """Some of a template's parts are written last but belong at the start.

    A template's shape -- what its kernel looks like, what it loads, what it
    stores -- is settled by walking the region, and the region is walked after
    the template's own text is laid out.  So a part that is not known when the
    text is laid out is written into the text as a placeholder, and filled in
    once the walk has said what it is.

    Asking for the code before every placeholder has been filled in is an error
    rather than a partially-built source: a caller that compiles what it gets
    would compile a program with a placeholder in it, and the failure would be
    the compiler's rather than here.
    """

    HookFn = Callable[[], str]

    def __init__(self, code: str, replacement_hooks: dict) -> None:
        super().__init__()
        self._code: str = code
        self.replacement_hooks: dict = dict(replacement_hooks)

    @property
    def code(self) -> str:
        """The source, once every placeholder has been filled in.

        Raises while any is still open, naming the ones that are: a caller
        that has not finished cannot tell which part it is missing, and would
        otherwise get source that looks complete.
        """

        remaining = [
            key for key, fn in self.replacement_hooks.items() if fn is not None
        ]
        if remaining:
            raise AssertionError(
                f"these placeholders have not been filled in: {remaining}"
            )
        return self._code

    def _replace_placeholder(self, hook_key: str, result: str) -> str:
        """Put one filled-in part where its placeholder stands.

        A placeholder on a line of its own and one in the middle of a line are
        different problems.  On its own, the text replaces the line and takes
        the line's indentation with it -- but only for a result written flush
        left, because a result that carries its own indentation is saying
        where it belongs and re-indenting it would put it somewhere it does not
        compile.  In the middle of a line, the text goes in as it is, because
        there is no line whose indentation it could take.

        A result with nothing in it removes the line.  That is how a part that
        turned out to be unnecessary says so: the piece was not needed, so
        there is nothing to write and the line it was on should not be there
        either.
        """

        if hook_key not in self._code:
            return self._code

        if not (result and result.strip()):
            lines = self._code.split("\n")
            return "\n".join(
                line for line in lines if line.strip() != hook_key
            )

        lines = self._code.split("\n")
        new_lines = []
        for line in lines:
            if line.strip() == hook_key:
                indent = line[: len(line) - len(line.lstrip())]
                result_lines = result.strip("\n").split("\n")
                non_empty = [one for one in result_lines if one.strip()]
                written_flush_left = bool(non_empty) and all(
                    not one[0].isspace() for one in non_empty
                )
                if written_flush_left:
                    new_lines.append(
                        "\n".join(
                            indent + one if one.strip() else one
                            for one in result_lines
                        ).rstrip()
                    )
                else:
                    new_lines.append(line.replace(hook_key, result))
            elif hook_key in line:
                new_lines.append(line.replace(hook_key, result))
            else:
                new_lines.append(line)
        return "\n".join(new_lines)

    def finalize_hook(self, hook_key: str, strict: bool = True) -> None:
        """Fill in one placeholder, by name.

        A placeholder that was never registered is a mistake in the caller
        rather than something to tolerate, unless it says otherwise: it means
        the part it expected to write is one this render does not have.

        A placeholder that has already been filled in cannot be filled in
        again, because the text it stood for is gone; saying so is better than
        writing the second answer into a source that no longer has the first
        place to put it.
        """

        if hook_key not in self.replacement_hooks:
            if strict:
                raise RuntimeError(
                    f"{hook_key} is not a placeholder in this render"
                )
            return

        hook = self.replacement_hooks[hook_key]
        if hook is None:
            raise AssertionError(f"{hook_key} has already been filled in")
        self._code = self._replace_placeholder(hook_key, hook())
        self.replacement_hooks[hook_key] = None

    def finalize_remaining(self) -> str:
        """Fill in whatever is still open, and return the source.

        For a caller that fills in one placeholder itself -- because it is the
        one that knows what that part is -- and wants the rest done for it.
        Unlike filling them all in, this skips the ones already done.
        """

        for key, fn in list(self.replacement_hooks.items()):
            if fn is not None:
                self.finalize_hook(key)
        return self.code

    def finalize_all(self) -> str:
        """Fill in every placeholder that has not been filled in yet.

        Unlike filling in the remaining ones, this is an error if one has
        already been done: a caller that has filled one in itself and then says
        "fill in the rest" means the rest, not all of them.
        """

        for key in self.replacement_hooks:
            self.finalize_hook(key)
        return self.code
