"""Choosing what runs: the built choice, the deferred pairing, and the operation itself.

Everything a call can be answered by is a choice, and a choice answers the
same questions: where its result lands, how to run it, how to measure it, and
whether it fits this call at all.  The operation the templates are measured
against is one of those choices rather than a fallback taken when measurement
fails, which is why it is registered here beside the rest.
"""
from __future__ import annotations

import hashlib
import itertools
import logging
import os
from typing import NamedTuple

from typing import Any, Iterator

import tensorplay as tp

from .. import config
from ..codegen.common import KernelTemplate
from ..kernel_inputs import KernelInputs
from ..ir import BaseView, Buffer as _Buffer, Layout, compute_required_storage_length
from ..loops import V
from .ir import ChoiceCaller, CutedslChoiceCaller
from .params import DictKernelTemplateParams, KernelTemplateParams
from .triton import CHOICES
from .subgraph import SubgraphChoiceCaller

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
                self._choice = self.template.generate(
                    self.params,
                    (self.layout,) if self.layout else (),
                    self.inputs.extra,
                    plain_launch,
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
class ExternChoiceCaller(ChoiceCaller):
    """The choice that runs the operation as one library call.

    There is no source to hash and no region to carry: the kernel is a name
    the runtime already knows how to launch, which is the whole reason this
    choice is in the list.
    """

    def __init__(self, name, input_nodes=(), layout=None, description="",
                 launcher=None):
        super().__init__(name, input_nodes, layout, description)
        self.launcher = launcher
        self.launcher_name = getattr(launcher, "__name__", None)
        if launcher is not None:
            self.bind(launcher)

    def call_name(self) -> str:
        """What a log line should call this, which is the library's own name.

        A trace is read by people looking for the operation they asked for, so
        the name here is the one the library uses rather than one invented here.
        """

        return self.launcher_name or self.name

    def hash_key(self) -> str:
        """What identifies this choice: the library name and nothing else.

        There is no source and no geometry to hash -- the library is the
        kernel -- so the key is the name of the thing being called.  Two calls
        with the same one are the same choice.
        """

        return self.call_name()

    def autoheuristic_id(self) -> str:
        return "extern"


class ExternKernelChoice(ExternChoiceCaller):
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
        name: str | None = None,
        *,
        has_out_variant: bool = True,
        op_overload: Any = None,
        use_fallback_kernel: bool = False,
        kernel_creator: Callable[..., Any] | None = None,
        cpp_kernel_name: str | None = None,
        gm: Any = None,
    ):
        name = name or getattr(kernel, "__name__", None) or "extern"
        if kernel is not None and not callable(kernel):
            raise AssertionError("an extern choice must wrap something callable")
        # ``None`` means the kernel is the framework's own and is resolved by
        # whatever launches it, which is the case for an operation the compiler
        # calls by name rather than by function.  The name is still the identity
        # that matters here, so the collision check below stands either way.
        existing = ExternKernelChoice._registry.get(name)
        if existing is not None and existing.kernel is not kernel:
            raise AssertionError(f"duplicate extern choice: {name}")
        # A registered choice is already the built thing, so it is a caller from
        # the moment it exists: there is no source to write and no grid to
        # compute, because the library call is the kernel.  Finishing it here
        # rather than later is what lets a list hold extern choices and
        # unbuilt templates side by side without either being special-cased.
        super().__init__(
            name=name,
            description=f"the {name} operation itself",
            launcher=kernel,
        )
        self.kernel = kernel
        self.has_out_variant = has_out_variant
        # The registered choice is already the built thing: there is no source
        # to write and no grid to compute, because the library call is the
        # kernel.  So the caller half of this is filled in here rather than
        # built later -- a choice that had to be finished before it could be
        # measured would be a choice nobody could put in a list.

        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel
        self.kernel_creator = kernel_creator
        #: The name the same operation goes by when it is emitted as C++, and
        #: the graph it stands for when the choice is a whole region.  Both are
        #: absent for an ordinary library call, and both are recorded rather
        #: than inferred: a name that had to be guessed would be a name that
        #: could be guessed wrong.
        self.cpp_kernel_name = cpp_kernel_name
        self.gm = gm
        ExternKernelChoice._registry[name] = self

    # -- the part that makes it usable wherever a template is ------------
    @property
    def uid(self) -> str:
        return self.name

    @property
    def src_hash(self) -> str | None:
        return None

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """The operation itself, as the choice it always is."""

        return ExternChoiceCaller(
            name=self.name,
            layout=kwargs.get("layout"),
            description="the operation itself",
            launcher=self.kernel,
        )

    def maybe_append_choice(self, choices: list, **kwargs: Any):
        choices.append(self.choice_or_none(**kwargs))
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        return self.choice_or_none(**kwargs)

    @classmethod
    def lookup(cls, name: str) -> "ExternKernelChoice | None":
        return cls._registry.get(name)

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
                 source: str = "", src_hash: str | None = None):
        super().__init__(name, input_nodes, layout, description)
        self.source = source
        self._src_hash = src_hash

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
        grid = self.grid(*args, **kwargs) if callable(self.grid) else self.grid
        return self.build()[grid](*args, **kwargs)

    def __repr__(self) -> str:
        return f"TritonTemplateKernel({self.kernel_name})"


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

    def generate(self, **kwargs: Any):
        """The choice this template makes for one set of arguments.

        Refuses rather than raising when the arguments do not describe a launch
        this template can perform, so that a whole table can be offered at once
        and the entries that do not apply dropped without unwinding the rest.
        """

        if self.source == "":
            return None
        kernel = self.template
        caller = TritonChoiceCaller(
            name=self.uid,
            layout=kwargs.get("layout"),
            description=repr(sorted((k, repr(v)) for k, v in kwargs.items())),
            source=self.source,
            src_hash=self.src_hash,
        )
        return caller.bind(lambda feed, _k=kernel: _k(*feed))

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
        #: The entry point's name, supplied by whatever renders this.
        self.symbol = ""

    def _operand(self, name: str) -> dict:
        if name in self.inputs:
            return self.inputs[name]
        if name in self.outputs:
            return self.outputs[name]
        raise KeyError(
            f"{name} is not one of this kernel's operands: "
            f"{sorted(self.inputs) + sorted(self.outputs)}"
        )

    def size(self, name: str, dim: int) -> str:
        """One extent of one operand, as the kernel's index type.

        Negative dimensions count from the end, which is what lets one body ask
        for ``-1`` and mean the contiguous axis whatever the rank is.
        """

        shape = self._operand(name)["shape"]
        index = dim if dim >= 0 else len(shape) + dim
        if not 0 <= index < len(shape):
            raise IndexError(
                f"{name} has {len(shape)} dimensions; {dim} is not one of them"
            )
        return f"size_{name.lower()}{index}"

    def stride(self, name: str, dim: int) -> str:
        """One stride of one operand, by name rather than by position.

        Strides are passed as arguments rather than baked in, so a body that
        asks for one does not have to be re-rendered per layout: the kernel is
        told the layout at launch, and the same text serves every layout.
        """

        return f"stride_{name.lower()}{dim}"

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
        params = ", ".join(f"{n}_ptr" for n in parts)
        strides = []
        for name in parts:
            rank = len(self._operand(name)["shape"])
            strides += [f"stride_{name.lower()}{d}" for d in range(rank)]
        sizes = []
        for name in parts:
            rank = len(self._operand(name)["shape"])
            sizes += [f"size_{name.lower()}{d}" for d in range(rank)]
        # A compile-time value is a parameter, not a body statement: it is
        # chosen when the kernel is rendered and cannot change while it runs,
        # which is exactly what a parameter the compiler may fold says.  So it
        # is annotated in the signature, where the compiler reads that.
        names = [*names, *outputs]
        tail = [f"{n}: tl.constexpr" for n in self.constexprs]
        all_params = [f"{n}_ptr" for n in names] + sizes + strides + tail
        return (
            "@triton.jit\n"
            f"def {symbol}({', '.join(all_params)}):"
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
            choice for choice in choices if isinstance(choice, ExternChoiceCaller)
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

        return isinstance(choice, (ExternChoiceCaller, SubgraphChoiceCaller))

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
        expected = None
        if VERIFY and choices:
            choices[0].benchmark(*example_inputs, out=out)
            expected = out.clone()
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

    A template carries its own rule when it has one, because a rule written for
    one template says nothing about another; the table by device is the fallback
    for the ones that do not, so that a template without a rule of its own still
    gets the device's numbers rather than nothing.
    """

    heuristic = getattr(template, "heuristics", None)
    if heuristic is not None:
        return heuristic
    return CHOICES.get_config_heuristics(device_type)


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
        (not isinstance(ktc.template, ExternChoiceCaller) for ktc in adjusted_choices)
    )

