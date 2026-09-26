"""Kernel templates: one operation, implemented by kernels chosen from a space.

The system has five parts, and each part owns exactly one decision:

  *Parameters* are a configuration, in a form that can be compared, cached and
  written down.  A template never sees a raw dict where a parameter class will
  do, so a configuration cannot be spelled two ways.

  *Config heuristics* decide which configurations are worth considering for a
  particular call.  They own the whole table and the rule that narrows it to
  this problem -- a tile wider than the problem is clamped to it -- which is
  why the table can be shared by every call of the operation and still produce
  a different set of candidates per shape.

  *Templates* turn one configuration into one kernel, or refuse it.  Refusal
  is a normal answer, raised as a plain "not implemented" and caught by the
  caller; a template that cannot run a call says so instead of guessing.

  *Template choices* pair a template with one configuration and defer building
  it.  The deferral is the point: a call is described long before anything is
  compiled, and a configuration that turns out not to fit is remembered as
  not fitting rather than rebuilt on every visit.

  *Choice callers* are the built kernel, and what is known about it: where its
  result lands, how to run it, how to measure it, and how to recognise it
  again in a cache.

A caller that wants a choice asks for one configuration at a time through
``choice_or_none``, or offers a list to ``maybe_append_choice`` and gets back
the ones that apply plus the reasons the others did not.  Nothing measures
anything until a caller holds several applicable choices, and the operator
itself is always one of them -- which is what makes measuring safe, since the
worst a measurement can conclude is that the operator was already the best of
them.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "BaseConfig",
    "CONV_TEMPLATES",
    "conv1x1_via_mm",
    "conv1x1_via_product",
    "framework_convolution",
    "CONV",
    "Choices",
    "ChoiceCaller",
    "ConvConfig",
    "ConvTemplate",
    "DepthwiseConvConfig",
    "DictKernelTemplateParams",
    "ExternChoiceCaller",
    "ExternKernelChoice",
    "GEMM",
    "GemmConfig",
    "GemmTemplate",
    "KernelInputs",
    "KernelTemplate",
    "KernelTemplateChoice",
    "KernelTemplateParams",
    "Layout",
    "LoopTemplate",
    "SubgraphChoiceCaller",
    "SubgraphTemplate",
    "TritonChoiceCaller",
    "TEMPLATES",
    "TemplateConfigHeuristics",
    "TemplateKernel",
    "TritonConfig",
    "contiguous_stride",
    "make_ktc_generator",
    "next_power_of_2",
    "template_for",
]


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


class KernelTemplateParams(ABC):
    """One configuration, in a form that can be written down and read back."""

    @abstractmethod
    def to_kwargs(self) -> dict[str, Any]:
        """The configuration as keyword arguments for the template."""

    @abstractmethod
    def to_serializeable_dict(self) -> dict[str, Any]:
        """The configuration as data, for storage and for cache keys."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict[str, Any]) -> "KernelTemplateParams":
        """The configuration this data describes."""


class DictKernelTemplateParams(KernelTemplateParams):
    """A configuration held as a dict.

    This is the compatibility layer: it lets a template be wired up before it
    has decided what its parameters mean.  A template whose parameters have
    defaults worth having spells them out in its own class instead, so that a
    caller can leave out what it does not care about.
    """

    def __init__(self, kwargs: dict[str, Any]):
        self.kwargs = dict(kwargs)

    def to_kwargs(self) -> dict[str, Any]:
        return dict(self.kwargs)

    def to_serializeable_dict(self) -> dict[str, Any]:
        return dict(self.kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DictKernelTemplateParams":
        return cls(data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DictKernelTemplateParams):
            return NotImplemented
        return self.kwargs == other.kwargs

    def __hash__(self) -> int:
        return hash(tuple(sorted((k, repr(v)) for k, v in self.kwargs.items())))

    def __repr__(self) -> str:
        inner = ", ".join(f"{k}={v!r}" for k, v in sorted(self.kwargs.items()))
        return f"DictKernelTemplateParams({inner})"


@dataclass(frozen=True)
class GemmTemplateParams(KernelTemplateParams):
    """A product's tile shape, and the launch geometry that goes with it."""

    BLOCK_M: int
    BLOCK_N: int
    BLOCK_K: int
    num_warps: int
    num_stages: int
    choice: str = "triton"

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "BLOCK_M": self.BLOCK_M,
            "BLOCK_N": self.BLOCK_N,
            "BLOCK_K": self.BLOCK_K,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def to_serializeable_dict(self) -> dict[str, Any]:
        return self.to_kwargs()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GemmTemplateParams":
        return cls(**data)


# ---------------------------------------------------------------------------
# layouts
# ---------------------------------------------------------------------------


def contiguous_stride(size: Sequence[int]) -> tuple:
    stride = []
    running = 1
    for extent in reversed(tuple(int(e) for e in size)):
        stride.append(running)
        running *= max(extent, 1)
    return tuple(reversed(stride))


def next_power_of_2(value: int) -> int:
    value = int(value)
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class Layout:
    """Where a result lands: its device, element type, extents and strides."""

    device: Any
    dtype: Any
    size: tuple
    stride: tuple
    offset: int = 0

    def __post_init__(self):
        object.__setattr__(self, "size", tuple(int(e) for e in self.size))
        object.__setattr__(self, "stride", tuple(int(s) for s in self.stride))

    @property
    def numel(self) -> int:
        total = 1
        for extent in self.size:
            total *= max(extent, 1)
        return total

    def __repr__(self) -> str:
        return f"Layout({self.dtype}, {self.size}, stride={self.stride})"


@dataclass(frozen=True)
class KernelInputs:
    """What a template is being asked to run.

    The operands, their extents and their element types, kept together so a
    heuristic can size itself to the problem and a template can refuse a form
    it has no kernel for.
    """

    shapes: tuple = ()
    dtypes: tuple = ()
    device: Any = None
    operands: tuple = ()
    feed: tuple = ()
    probe_feed: tuple = ()
    extra: dict = field(default_factory=dict)

    @property
    def rank(self) -> int:
        return max((len(s) for s in self.shapes), default=0)


# ---------------------------------------------------------------------------
# configurations
# ---------------------------------------------------------------------------


@dataclass
class TritonConfig:
    """A launch geometry: block extents plus how many warps and stages."""

    kwargs: dict[str, int]
    num_stages: int = 3
    num_warps: int = 4

    def as_kwargs(self) -> dict[str, Any]:
        return {**self.kwargs, "num_warps": self.num_warps, "num_stages": self.num_stages}


@dataclass
class BaseConfig:
    """A tile shape, before it has been fitted to a particular problem.

    The extents are what the tile would like; the heuristics decide what it
    gets once the problem's size is known.
    """

    block_m: int
    block_n: int
    block_k: int
    num_stages: int
    num_warps: int
    hint_override: int | None = field(default=None, kw_only=True)

    def tile(self) -> dict[str, int]:
        return {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
        }


@dataclass
class GemmConfig(BaseConfig):
    """A product's tile shape."""


@dataclass
class ConvConfig(BaseConfig):
    """A convolution's tile shape: the same product, with a spatial extent."""


@dataclass
class DepthwiseConvConfig:
    """A depthwise convolution's tiling, which is not a product at all.

    Each input channel is reduced on its own, so the tile is over the output
    positions, the positions along the axis, and the channels -- not over a
    contraction.
    """

    block_n: int
    block_l: int
    block_c: int
    num_stages: int
    num_warps: int

    def tile(self) -> dict[str, int]:
        return {"BLOCK_N": self.block_n, "BLOCK_L": self.block_l, "BLOCK_C": self.block_c}


# ---------------------------------------------------------------------------
# config heuristics
# ---------------------------------------------------------------------------


class TemplateConfigHeuristics:
    """Which configurations of one template are worth trying for one call.

    Splitting this out is what lets a table be written once and shared: the
    table says what exists, and the rule below says which of it fits here.
    """

    def should_run(self, inputs: KernelInputs) -> bool:
        """Whether this heuristic has anything to say about this call."""

        return True

    def get_template_configs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> Iterator[KernelTemplateParams]:
        """The configurations for this call, as parameters."""

        if not self.should_run(kernel_inputs):
            return
        for config_dict in self._get_template_configs_impl(kernel_inputs, op_name):
            yield DictKernelTemplateParams(config_dict)

    def _get_template_configs_impl(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> Iterator[dict[str, Any]]:
        """The configurations for this call, as keyword arguments."""

        return iter(())

    def get_extra_kwargs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> dict[str, Any]:
        """What the template needs whatever configuration it is given.

        These are the parts of the call that do not vary with the tile: the
        geometry of a convolution, the layout a result lands in.  They are
        passed alongside every configuration rather than inside it, so that
        they are not mistaken for part of the choice.
        """

        return {}

    def adjust_kernel_inputs(
        self, kernel_inputs: KernelInputs, op_name: str
    ) -> KernelInputs:
        """The inputs as this template wants to see them.

        A template that needs a matrix where the call has a batch of them can
        add the axis here, once, instead of at every use.
        """

        return kernel_inputs


class _TileConfigHeuristic(TemplateConfigHeuristics):
    """The rule shared by every tiled operation: clamp the tile to the problem.

    A tile wider than the problem it tiles is wasted lanes and, worse, a mask
    that has to be paid for on every step.  Each configuration is therefore
    narrowed to the shape it will actually run against, rounded up to a power
    of two because that is what the block extents have to be, and never
    narrower than the smallest block a launch can address.
    """

    #: The narrowest block any of these extents may be clamped to.
    min_block_size = 16
    #: The narrowest contraction block; an integer operand needs twice the room.
    min_block_size_k = 16
    #: Extra hints tried in turn, for shapes the framework can guess at.
    multi_kernel_hints: tuple = ()

    def _fit(self, extent: int, minimum: int) -> int:
        return max(next_power_of_2(extent), minimum)

    def _scale_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: Sequence[BaseConfig],
        scale: float = 1.0,
        exclude: Callable[[int, int, int], bool] = lambda bm, bn, bk: False,
        has_int8_tensor: bool = False,
    ) -> list[BaseConfig]:
        minimum_k = 32 if has_int8_tensor else self.min_block_size_k
        m_hint = self._fit(m, self.min_block_size)
        n_hint = self._fit(n, self.min_block_size)
        k_hint = self._fit(k, minimum_k)
        fitted: list[BaseConfig] = []
        for hint in (None,) + tuple(self.multi_kernel_hints):
            for config in configs:
                block_m = max(min(int(config.block_m * scale), m_hint), self.min_block_size)
                block_n = max(min(int(config.block_n * scale), n_hint), self.min_block_size)
                block_k = max(min(int(config.block_k * scale), k_hint), minimum_k)
                if exclude(block_m, block_n, block_k):
                    continue
                if (block_m, block_n, block_k, hint) == (
                    config.block_m,
                    config.block_n,
                    config.block_k,
                    config.hint_override,
                ):
                    fitted.append(config)
                    continue
                fitted.append(
                    type(config)(
                        block_m,
                        block_n,
                        block_k,
                        config.num_stages,
                        config.num_warps,
                        hint_override=hint,
                    )
                    if config.hint_override is not None
                    else type(config)(
                        block_m, block_n, block_k, config.num_stages, config.num_warps
                    )
                )
        return fitted

    def _finalize(self, configs: Sequence[BaseConfig]) -> Iterator[TritonConfig]:
        """Turn fitted tiles into launch geometries, one per distinct kernel.

        Clamping collapses a table: once every tile has been narrowed to the
        problem, several entries ask for the same kernel with the same launch
        geometry.  Those are not further choices -- measuring one of them
        measures all of them -- so the list keeps one and the count means
        something.
        """

        seen = set()
        for config in configs:
            geometry = (tuple(sorted(config.tile().items())),
                        config.num_stages, config.num_warps)
            if geometry in seen:
                continue
            seen.add(geometry)
            yield TritonConfig(config.tile(), config.num_stages, config.num_warps)

    def preprocess_mm_configs(
        self,
        m: int,
        n: int,
        k: int,
        configs: Sequence[BaseConfig],
        *,
        has_int8_tensor: bool = False,
        scale: float = 1.0,
        exclude: Callable[[int, int, int], bool] = lambda bm, bn, bk: False,
        op_name: str = "mm",
    ) -> Iterator[TritonConfig]:
        """The tile shapes worth running for a product of this size."""

        fitted = self._scale_configs(
            m, n, k, configs, scale, exclude, has_int8_tensor
        )
        return self._finalize(fitted)

    def triton_config(self, num_stages: int, num_warps: int, **kwargs) -> TritonConfig:
        return TritonConfig(kwargs, num_stages, num_warps)

    def get_mm_configs(self) -> Callable[..., Iterator[TritonConfig]]:
        """A generator of tile shapes for a product, over this table."""

        return lambda m, n, k, **kwargs: self.preprocess_mm_configs(
            m, n, k, self.mm_configs, op_name="mm", **kwargs
        )

    def get_conv_configs(self) -> Callable[..., Iterator[TritonConfig]]:
        """A generator of tile shapes for a convolution, over this table."""

        return lambda m, n, k, **kwargs: self.preprocess_mm_configs(
            m, n, k, self.conv_configs, op_name="conv", **kwargs
        )

    def get_depthwise_conv_configs(self) -> list[TritonConfig]:
        """The depthwise tilings, which are not products and are not fitted."""

        return [
            TritonConfig(c.tile(), c.num_stages, c.num_warps)
            for c in self.depthwise_conv_configs
        ]

    #: The product tilings.  Read as (M, N, K, stages, warps).
    mm_configs: tuple = ()
    #: The convolution tilings, read the same way.
    conv_configs: tuple = ()
    #: The depthwise tilings, which are (N, L, C, stages, warps).
    depthwise_conv_configs: tuple = ()


class CudaConfigHeuristic(_TileConfigHeuristic):
    """The tables a discrete accelerator is measured with.

    The product tilings are grouped by contraction width, since that is what
    decides how many times the kernel steps over the contraction: a kernel
    whose contraction tile covers the whole thing does it once.  The
    convolution tilings are the same shapes, because a convolution is a
    product over the input channels.
    """

    mm_configs = (
        # Contraction 16.
        GemmConfig(64, 256, 16, 2, 4),
        GemmConfig(256, 64, 16, 2, 4),
        GemmConfig(1024, 16, 16, 1, 8),
        # Contraction 32.
        GemmConfig(128, 128, 32, 2, 8),
        GemmConfig(64, 64, 32, 2, 4),
        GemmConfig(64, 256, 32, 2, 8),
        GemmConfig(256, 64, 32, 2, 8),
        # Contraction 64.
        GemmConfig(128, 128, 64, 3, 8),
        GemmConfig(64, 128, 64, 4, 4),
        GemmConfig(128, 64, 64, 4, 4),
        GemmConfig(256, 128, 64, 2, 8),
        GemmConfig(128, 256, 64, 2, 8),
        # Contraction 128, which is the whole contraction in one step for a
        # 128-channel input -- the shape convolutions of this width produce.
        GemmConfig(128, 128, 128, 2, 8),
        GemmConfig(128, 128, 128, 3, 8),
        GemmConfig(64, 128, 128, 4, 4),
        GemmConfig(256, 128, 128, 2, 8),
        GemmConfig(128, 256, 128, 2, 8),
    )

    conv_configs = (
        # Contraction 16.
        ConvConfig(64, 256, 16, 2, 4),
        ConvConfig(256, 64, 16, 2, 4),
        ConvConfig(1024, 16, 16, 1, 8),
        # Contraction 32.
        ConvConfig(128, 128, 32, 2, 8),
        ConvConfig(64, 64, 32, 2, 4),
        ConvConfig(64, 256, 32, 2, 8),
        ConvConfig(256, 64, 32, 2, 8),
        # Contraction 64.
        ConvConfig(128, 128, 64, 3, 8),
        ConvConfig(64, 128, 64, 4, 4),
        ConvConfig(128, 64, 64, 4, 4),
        ConvConfig(256, 128, 64, 2, 8),
        ConvConfig(128, 256, 64, 2, 8),
        # Contraction 128.
        ConvConfig(128, 128, 128, 2, 8),
        ConvConfig(128, 128, 128, 3, 8),
        ConvConfig(64, 128, 128, 4, 4),
        ConvConfig(256, 128, 128, 2, 8),
        ConvConfig(128, 256, 128, 2, 8),
    )

    depthwise_conv_configs = (
        # Channel 32, position 32.
        DepthwiseConvConfig(16, 32, 32, 4, 8),
        DepthwiseConvConfig(32, 32, 32, 4, 8),
        # Channel 32, position 64.
        DepthwiseConvConfig(16, 64, 32, 4, 8),
        DepthwiseConvConfig(32, 64, 32, 4, 8),
        # Channel 64, position 32.
        DepthwiseConvConfig(16, 32, 64, 4, 8),
        DepthwiseConvConfig(32, 32, 64, 4, 8),
        # Channel 64, position 64.
        DepthwiseConvConfig(16, 64, 64, 4, 8),
        DepthwiseConvConfig(32, 64, 64, 4, 8),
    )


class CpuConfigHeuristic(CudaConfigHeuristic):
    """A processor has no shared memory to spill into, and one shape to fit.

    A tile whose contraction is wider than the whole contraction is the one
    case that is worth refusing outright here: there is no second step to hide
    it behind, so it only costs.
    """

    def preprocess_mm_configs(self, m, n, k, configs, **kwargs):
        configs = [c for c in configs if c.block_k < max(k, 1)]
        return super().preprocess_mm_configs(m, n, k, configs, **kwargs)


class Choices:
    """Which tables a device is measured with.

    The device is asked for its tables rather than reaching for them, so a new
    kind of device brings its own numbers instead of inheriting another's.
    """

    def __init__(self, heuristics: dict | None = None):
        self._heuristics = dict(heuristics or {})

    def register(self, device_type: str, heuristic: _TileConfigHeuristic) -> None:
        self._heuristics[device_type] = heuristic

    def get_config_heuristics(self, device_type: str | None = "cuda"):
        return self._heuristics.get(str(device_type), self._heuristics["cuda"])

    def get_conv_configs(self, device_type: str | None = "cuda"):
        return self.get_config_heuristics(device_type).get_conv_configs()

    def get_depthwise_conv_configs(self, device_type: str | None = "cuda"):
        return self.get_config_heuristics(device_type).get_depthwise_conv_configs()

    def get_mm_configs(self, device_type: str | None = "cuda"):
        return self.get_config_heuristics(device_type).get_mm_configs()


#: The tables, one per kind of device.
CHOICES = Choices({"cuda": CudaConfigHeuristic(), "cpu": CpuConfigHeuristic()})


# ---------------------------------------------------------------------------
# choice callers
# ---------------------------------------------------------------------------


class ChoiceCaller:
    """A built choice: where its result lands, and how to run and measure it.

    This is the last step of a template, and the only thing a caller needs to
    hold afterwards: everything above it was a description of a possibility,
    and this is the possibility that was built.
    """

    def __init__(
        self,
        name: str,
        input_nodes: tuple = (),
        layout: Layout | None = None,
        description: str = "",
    ):
        self.name = name
        self.input_nodes = tuple(input_nodes)
        self.layout = layout
        self.description = description
        #: Set when a measurement showed this choice does not work here.
        self.failed = False
        #: A place for information that only the measurement needs to carry.
        self.annotations: dict[str, Any] = {}
        #: Geometry substitutions a measurement decided on, for the record.
        self.config_patches: dict[str, Any] = {}
        self._callable = None

    def bind(self, launcher: Callable[..., Any]) -> "ChoiceCaller":
        """Attach the thing that actually runs this choice."""

        self._callable = launcher
        return self

    def call_name(self) -> str:
        return self.name

    def to_callable(self) -> Callable[..., Any]:
        if self._callable is None:
            raise NotImplementedError(f"{self.name} has no kernel bound to it")
        return self._callable

    def kernel_hash_key(self) -> str:
        """What identifies the kernel itself, for a binary cache."""

        return self.hash_key()

    def hash_key(self) -> str:
        """What identifies this choice, geometry and layout included."""

        parts = [self.name, self.description]
        if self.layout is not None:
            parts.append(repr(self.layout))
        return ":".join(parts)

    def benchmark(self, *args: Any, out: Any = None) -> float:
        """How long one run of this choice takes."""

        from .runtime.stax_autotune import bench_launch

        algo = self.to_callable()
        operands = list(args)
        if out is not None:
            operands.append(out)
        return bench_launch(algo, operands)

    def info_dict(self) -> dict[str, Any]:
        """What is worth writing down about this choice."""

        return {
            "name": self.name,
            "description": self.description,
            "hash_key": self.hash_key(),
        }

    def autoheuristic_id(self) -> str:
        return "unsupported_choice"

    def mark_failed(self) -> None:
        """Record that this choice does not work here, so it is not offered."""

        self.failed = True

    def __repr__(self) -> str:
        return f"ChoiceCaller({self.name}, {self.description})"


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------


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


class SubgraphChoiceCaller(ChoiceCaller):
    """A choice whose kernel is a whole region, emitted and kept as one.

    The region travels with the choice because a template that fuses several
    operations has to carry them: the graph is the thing that was chosen, so
    dropping it would leave a caller holding a name and nothing to run.
    """

    def __init__(self, name, input_nodes=(), layout=None, description="",
                 graph=None, decomposition=None, decomposition_kwargs=None):
        super().__init__(name, input_nodes, layout, description)
        self.gm = graph
        self.decomposition = decomposition
        self.decomposition_kwargs = dict(decomposition_kwargs or {})

    def autoheuristic_id(self) -> str:
        return "subgraph"


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
        name: str | None = None,
        *,
        has_out_variant: bool = True,
        op_overload: Any = None,
        use_fallback_kernel: bool = False,
        kernel_creator: Callable[..., Any] | None = None,
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
        self.name = name
        self.kernel = kernel
        self.has_out_variant = has_out_variant
        self.op_overload = op_overload
        self.use_fallback_kernel = use_fallback_kernel
        self.kernel_creator = kernel_creator
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
        if launcher is not None:
            self.bind(launcher)

    def autoheuristic_id(self) -> str:
        return "extern"


class KernelTemplate:
    """One operation, implemented by kernels chosen from a configuration space.

    A subclass says what to do with one configuration in ``generate``, and
    everything else here is the discipline around it: a refusal is caught and
    reported rather than propagated, an identity is stable across processes so
    a stored decision can be recognised, and a source digest lets a stored
    decision be thrown away when the kernel it named has changed.
    """

    def __init__(self, name: str, hash: str | None = None):
        self.name = name
        self._hash = hash

    @property
    def uid(self) -> str:
        """A stable identity, so a stored decision can be found again.

        Every template is unique in the system, and the identity has to
        survive a restart, so nothing about it may depend on where it was
        defined.
        """

        return self.name

    @property
    def src_hash(self) -> str | None:
        """A digest of the source this template emits, when it has one.

        A template that emits a kernel can say what the kernel was, so a
        stored decision is not reused once the kernel it names has moved.
        """

        return self._hash

    def choice_or_none(self, **kwargs: Any) -> ChoiceCaller | None:
        """The choice for one configuration, or ``None`` when it does not fit."""

        collected: list[ChoiceCaller] = []
        error = self.maybe_append_choice(collected, **kwargs)
        if error is None and len(collected) == 1:
            return collected[0]
        return None

    def maybe_append_choice(
        self, choices: list[ChoiceCaller], **kwargs: Any
    ) -> NotImplementedError | None:
        """Add the choice for one configuration, or report why there is none.

        Refusing is a normal answer and comes back as the refusal rather than
        as an exception, so a caller can offer a whole table and keep the
        ones that apply.
        """

        try:
            choices.append(self.generate(**kwargs))
        except NotImplementedError as error:
            return error
        return None

    def generate(self, **kwargs: Any) -> ChoiceCaller:
        """The choice one configuration describes."""

        raise NotImplementedError


class LoopTemplate(KernelTemplate):
    """A template reached through a loop region.

    The loop runtime describes a call before anything is compiled, so this
    layer adds the two things that description needs: the result the call
    produces, which the template knows and its caller would otherwise have to
    infer, and the probe its candidates are measured on, which is built from
    the same description rather than from the region's real tensors.
    """

    def __init__(self, name: str, hash: str | None = None):
        super().__init__(name, hash)
        self.heuristics = TemplateConfigHeuristics()

    # identity ------------------------------------------------------------
    def emitter(self) -> str:
        """The source a configuration is turned into, for change detection."""

        raise NotImplementedError

    @property
    def uid(self) -> str:
        digest = hashlib.sha1(self.emitter().encode()).hexdigest()[:16]
        return f"{self.name}:{digest}"

    # results -------------------------------------------------------------
    def out_specs(self, meta: dict) -> tuple:
        """The layouts this call produces."""

        raise NotImplementedError

    # configurations ------------------------------------------------------
    def inputs_for(self, meta: dict) -> KernelInputs:
        return KernelInputs(
            shapes=tuple(meta.get("operand_sizes") or ()),
            dtypes=tuple([meta.get("operand_dtype")] * len(meta.get("operand_sizes") or ())),
            device=meta.get("device"),
            operands=tuple(meta.get("operand_specs") or ()),
            feed=tuple(meta.get("feed") or ()),
            probe_feed=tuple(meta.get("probe_feed") or ()),
            extra=meta,
        )

    def configurations(self, out_specs: tuple, meta: dict) -> Iterator[KernelTemplateParams]:
        """The configurations worth considering for this call."""

        inputs = self.heuristics.adjust_kernel_inputs(self.inputs_for(meta), self.name)
        return self.heuristics.get_template_configs(inputs, self.name)

    def generate_for(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit.

        ``plain_launch`` is what the region would run with no template at all.
        It is handed in rather than reached for because for one configuration
        it *is* the answer: the operation is a candidate like any other, and
        the only way it can be is by being given the thing it replaces.
        """

        raise NotImplementedError

    # enumeration ---------------------------------------------------------
    def choices(self, meta: dict) -> list:
        """Every configuration that applies to this call."""

        return self.collect(meta, lambda choice, plain: choice.resolve(plain))

    def collect(self, meta: dict, build) -> list:
        """Build the applicable choices, leaving out the ones that do not fit.

        The configurations come from the heuristic, the overrides are what
        this call imposes on all of them, and the pairing is the one the
        deferred-choice generator owns -- so there is a single way for a
        configuration to become a choice, whichever template asked for it.
        """

        specs = self.out_specs(meta)
        inputs = self.heuristics.adjust_kernel_inputs(self.inputs_for(meta), self.name)
        overrides = dict(self.heuristics.get_extra_kwargs(inputs, self.name))
        if specs:
            overrides.setdefault("out_size", tuple(specs[0].size))
            overrides.setdefault("out_dtype", specs[0].dtype)
        out = []
        for choice in make_ktc_generator(
            self,
            self.configurations(specs, meta),
            {},
            overrides,
            specs[0] if specs else None,
            inputs,
        ):
            self.maybe_append_choice(out, choice, build)
        return out

    def maybe_append_choice(self, choices: list, choice, build) -> Any:
        try:
            if build(choice, None) is None:
                raise NotImplementedError(
                    f"{self.name} configuration {choice.params!r} does not fit"
                )
        except NotImplementedError as error:
            return error
        choices.append(choice)
        return None

    def probe(self, meta: dict):
        """A deterministic operand set for measuring this call's candidates."""

        return None

    def select(self, meta: dict, build) -> tuple:
        """Choose among this call's configurations and return the winner.

        The choice is made while the region is compiled, so the kernel a
        region runs is settled before it ever runs.  The operator itself is
        always among the candidates, which is what makes measuring safe: the
        worst a measurement can conclude is that the operator was already the
        best of them.
        """

        choices = self.collect(meta, build)
        if not choices:
            return None, None
        if len(choices) == 1 or not meta.get("bench", True):
            return build(choices[0], None), choices[0].params
        from .runtime.stax_autotune import bench_candidates

        built = {}

        def materialise(candidate, plain):
            launcher = build(candidate, plain)
            built[candidate] = launcher
            return launcher

        feed = meta.get("probe_feed")
        if feed is None:
            return build(choices[0], None), choices[0].params
        best, _launch, _time = bench_candidates(
            materialise, choices, feed, rounds=meta.get("rounds", 2)
        )
        if best is None:
            best = choices[0]
        return built.get(best) or build(best, None), best.params


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
    ):
        self.template = template
        self.params = params
        self.extra_kwargs = dict(extra_kwargs)
        self.layout = layout
        self.inputs = inputs
        self.annotations: dict[str, Any] = {"ktc": self}

    @property
    def choice(self) -> ChoiceCaller | None:
        """The built choice, or ``None`` when this configuration does not fit."""

        if not hasattr(self, "_choice"):
            kwargs = self.params.to_kwargs()
            try:
                self._choice = self.template.choice_or_none(
                    **kwargs, **self.extra_kwargs
                )
            except NotImplementedError:
                self._choice = None
            if self._choice is not None:
                self._choice.annotations = self.annotations
        return self._choice

    @property
    def key(self) -> tuple:
        """What identifies this choice across processes."""

        return (
            self.template.uid,
            repr(self.params.to_serializeable_dict()),
            repr(self.layout),
            str(self.inputs.device),
        )

    def resolve(self, plain_launch):
        """The launcher for this configuration, or ``None`` when it does not fit."""

        if not hasattr(self, "_resolved"):
            self._resolved = True
            try:
                self._choice = self.template.generate_for(
                    self.params,
                    (self.layout,) if self.layout else (),
                    self.inputs.extra,
                    plain_launch,
                )
            except NotImplementedError:
                self._choice = None
            if self._choice is not None and self._choice is not plain_launch:
                self._choice.bind(lambda *a, _c=self._choice, **k: _c.to_callable()(*a, **k))
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


# ---------------------------------------------------------------------------
# products
# ---------------------------------------------------------------------------


class GemmConfigHeuristics(TemplateConfigHeuristics):
    """The candidates for a product, fitted to the product's size.

    The operator is always the first candidate: it is the floor a measurement
    can never lose against, and having it in the list is what makes the rest
    of the list safe to measure.
    """

    def __init__(self, op_name: str = "mm", device_type: str = "cuda"):
        self.op_name = op_name
        self.device_type = device_type

    def should_run(self, inputs: KernelInputs) -> bool:
        if len(inputs.shapes) < 2:
            return False
        rows, inner = inputs.shapes[0]
        inner2, cols = inputs.shapes[1]
        return len((rows, inner, cols)) == 3 and inner == inner2

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        rows, inner = kernel_inputs.shapes[0]
        _inner2, cols = kernel_inputs.shapes[1]
        generator = CHOICES.get_mm_configs(self.device_type)
        for config in generator(rows, cols, inner):
            yield {"choice": "triton", **config.as_kwargs()}


class GemmTemplate(LoopTemplate):
    """Products, measured against the framework's own.

    Validity belongs to the template: a configuration that does not fit this
    call -- not two-dimensional, not the element type the tiles accumulate in,
    not this device -- is refused here rather than by whoever is asking.
    """

    def __init__(self):
        super().__init__("gemm")
        self.heuristics = GemmConfigHeuristics()

    def emitter(self) -> str:
        """The tile kernel this template emits, identified by its source.

        The digest covers the kernel body and the tuning version, so a stored
        decision is not reused once either has moved.
        """

        from .codegen.triton_gemm import GEMM_TUNING_VERSION, _kernel_source_digest

        return f"{GEMM_TUNING_VERSION}:{_kernel_source_digest()}"

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if size is None:
            raise NotImplementedError("a product without a result shape")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                contiguous_stride(size),
            ),
        )

    def probe(self, meta: dict):
        """A deterministic operand set for measuring this call's candidates.

        The template owns its result, so it also knows the extents a product
        has and can build the feed that exercises every lane of a candidate
        without being handed the region's real tensors.
        """

        from .codegen.triton_gemm import _probe_feed

        if meta.get("b_transposed") or len(meta.get("operand_specs", ())) != 2:
            return None
        layout = self.out_specs(meta)[0]
        if len(layout.size) != 2:
            return None
        sizes = meta.get("operand_sizes") or ()
        if len(sizes) != 2:
            return None
        (rows, inner), (inner2, cols) = sizes
        m, n = layout.size
        if rows != m or cols != n or inner != inner2:
            return None
        if meta.get("out_dtype") is None:
            return None
        return _probe_feed(m, inner, n, layout.dtype, meta.get("device"), bias=False)

    def generate_for(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        kwargs = params.to_kwargs()
        layout = out_specs[0] if out_specs else None
        if kwargs.get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_product",
                layout=layout,
                description="the operation itself",
                launcher=plain_launch,
            )
        if layout is None or len(layout.size) != 2 or meta.get("transposed"):
            return None
        if meta.get("operand_dtype") != "float32":
            return None
        if not meta.get("qualifies", False):
            return None
        from .codegen.triton_gemm import tuned_matmul_launch

        caller = ChoiceCaller(
            name=f"gemm-{kwargs.get('BLOCK_M')}x{kwargs.get('BLOCK_N')}"
            f"x{kwargs.get('BLOCK_K')}",
            layout=layout,
            description=repr(kwargs),
        )
        caller.config_patches = {
            key: kwargs[key]
            for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages")
            if key in kwargs
        }
        return caller.bind(
            tuned_matmul_launch(
                None,
                meta.get("probe_feed") or meta.get("feed") or (),
                meta["operand_specs"],
                layout.size,
                bias_spec=meta.get("bias_spec"),
                b_transposed=bool(meta.get("b_transposed", False)),
                config=kwargs,
            )
        )


class SubgraphTemplate(KernelTemplate):
    """A template whose kernel is a region of the graph, emitted as one.

    Some operations are not worth writing by hand: the fusion of a run of
    elementwise operations into a single pass is a matter of capturing the
    region, not of expressing it.  This template therefore takes the region as
    it stands, and its one decision is whether the region is worth keeping as
    a unit -- which is the same question every other template answers about its
    own configuration, asked about a different thing.
    """

    def __init__(self, name: str, hash: str | None = None):
        super().__init__(name, hash)
        #: Set by the caller that owns the region, before any choice is built.
        self.graph = None
        self.decomposition = None
        self.decomposition_kwargs: dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> SubgraphChoiceCaller:
        if self.graph is None:
            raise NotImplementedError(f"{self.name} has no region to emit")
        layout = kwargs.get("layout")
        return SubgraphChoiceCaller(
            name=self.name,
            input_nodes=tuple(kwargs.get("input_nodes") or ()),
            layout=layout,
            description=repr(sorted(kwargs.get("overrides", {}).items())),
            graph=self.graph,
            decomposition=self.decomposition,
            decomposition_kwargs=self.decomposition_kwargs,
        )

    def choice_or_none(self, **kwargs: Any) -> SubgraphChoiceCaller | None:
        return super().choice_or_none(**kwargs)


GEMM = GemmTemplate()


# ---------------------------------------------------------------------------
# convolutions
# ---------------------------------------------------------------------------


def _pair(value, count: int) -> list:
    if isinstance(value, int):
        return [int(value)] * count
    values = [int(v) for v in value]
    if len(values) == 1:
        return values * count
    return values


def _conv_output_size(extent: int, kernel: int, stride: int, padding: int, dilation: int) -> int:
    return (extent + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


class ConvConfigHeuristics(TemplateConfigHeuristics):
    """The candidates for a convolution, fitted to what it convolves.

    A convolution is a product over the input channels, so its tilings are
    fitted to the product the way a product's are: the rows are the output
    positions, the columns the output channels, and the contraction the input
    channels.  The positions are counted across the batch and the whole
    spatial extent together, because a tile does not care where a position
    came from.
    """

    def __init__(self, device_type: str = "cuda"):
        self.device_type = device_type

    def should_run(self, inputs: KernelInputs) -> bool:
        return len(inputs.shapes) >= 2

    def _get_template_configs_impl(self, kernel_inputs, op_name):
        yield {"choice": "operator"}
        extra = kernel_inputs.extra
        generator = CHOICES.get_conv_configs(self.device_type)
        rows = extra.get("conv_rows")
        cols = extra.get("out_channels")
        inner = extra.get("in_channels_per_group")
        if not rows or not cols or not inner:
            return
        for config in generator(rows, cols, inner):
            yield {"choice": "triton", **config.as_kwargs()}


class ConvTemplate(LoopTemplate):
    """Convolutions, with the 1x1 case expressed as the product it is.

    A one-by-one convolution with unit stride, no padding and one group is a
    matrix product, so it is measured as one -- the product template's tile
    set applies to it directly, and there is nothing to be gained from a
    second set of candidates for the same arithmetic.

    Wider kernels keep the operator, which is the floor a measurement can never
    lose against, until there is a tiled kernel for them to choose between.
    """

    def __init__(self):
        super().__init__("conv")
        self.heuristics = ConvConfigHeuristics()

    def emitter(self) -> str:
        return "conv:operator+product-1x1"

    def out_specs(self, meta: dict) -> tuple:
        size = meta.get("out_size")
        if not size:
            raise NotImplementedError("a convolution without a result shape")
        return (
            Layout(
                meta.get("device"),
                meta.get("out_dtype"),
                tuple(size),
                contiguous_stride(size),
            ),
        )

    def is_one_by_one(self, meta: dict) -> bool:
        """Is this the product-shaped case: 1x1 kernel, unit stride, one group?"""

        kernel = tuple(int(k) for k in meta.get("kernel_size") or ())
        if not kernel or any(k != 1 for k in kernel):
            return False
        stride = tuple(int(s) for s in meta.get("stride") or ())
        if any(s != 1 for s in stride):
            return False
        padding = tuple(int(p) for p in meta.get("padding") or ())
        if any(p != 0 for p in padding):
            return False
        if int(meta.get("groups", 1)) != 1:
            return False
        if meta.get("transposed"):
            return False
        dilation = tuple(int(d) for d in meta.get("dilation") or ())
        return all(d == 1 for d in dilation)

    def generate_for(self, params: KernelTemplateParams, out_specs: tuple, meta: dict,
                     plain_launch=None):
        """The choice for one configuration, or ``None`` when it does not fit."""

        if params.to_kwargs().get("choice") == "operator":
            if plain_launch is None:
                return None
            return ExternChoiceCaller(
                name="framework_convolution",
                layout=out_specs[0] if out_specs else None,
                description="the operation itself",
                launcher=plain_launch,
            )
        if not self.is_one_by_one(meta):
            return None
        # The product case is measured by the product template, which owns that
        # tile space; this template only decides that the case applies.
        gemm_meta = {
            "out_size": tuple(out_specs[0].size) if out_specs else (),
            "out_dtype": out_specs[0].dtype if out_specs else None,
            "device": out_specs[0].device if out_specs else None,
            "operand_dtype": meta.get("operand_dtype"),
            "operand_sizes": meta.get("operand_sizes"),
            "operand_specs": meta.get("operand_specs"),
            "bias_spec": meta.get("bias_spec"),
            "qualifies": True,
            "probe_feed": meta.get("probe_feed"),
            "feed": meta.get("feed"),
        }
        launcher, _params = GEMM.select(gemm_meta, lambda choice, plain: choice.resolve(plain))
        if launcher is None:
            return None
        caller = ChoiceCaller(
            name="conv-1x1-product",
            layout=out_specs[0] if out_specs else None,
            description="a 1x1 convolution measured as the product it is",
        )
        return caller.bind(launcher)


# ---------------------------------------------------------------------------
# the convolution family
# ---------------------------------------------------------------------------


def conv1x1_via_mm(x, w, *, out=None):
    """A one-by-one convolution, written as the product it is.

    The weight's two trailing extents are one, so squeezing them is a view; the
    contraction is over the channels, which means the input has to be read
    with its channels last and the weight with its output channels first.  The
    result therefore lands in a permuted layout, and un-permuting it is part
    of the same work rather than something the caller does afterwards.
    """

    import tensorplay as tp

    w = w.reshape(w.shape[0], w.shape[1])
    moved = x.permute(0, 2, 3, 1) if x.dim() == 4 else x.permute(0, 2, 1)
    product = tp.matmul(moved, w.permute(1, 0), out=out)
    return product.permute(0, 3, 1, 2) if x.dim() == 4 else product.permute(0, 2, 1)


#: The convolution the framework runs, as a choice rather than a fallback.
framework_convolution = ExternKernelChoice(
    None, "framework_convolution", has_out_variant=False
)
#: The one-by-one case, as the product it is, including the layout it needs.
conv1x1_via_product = ExternKernelChoice(conv1x1_via_mm, "conv1x1_via_mm")


def _conv_forward_grid(rows, cols, groups, block_m, block_n, cdiv):
    return (cdiv(rows, block_m), cdiv(cols, block_n), groups)


def _conv_bwd_input_grid(rows, cols, groups, block_m, block_n, cdiv):
    return (cdiv(rows, block_m), cdiv(cols, block_n), groups)


def _conv_bwd_weight_grid(groups, block_m, block_n, block_k, cdiv):
    return (cdiv(1, block_m), cdiv(1, block_n), groups)


#: The convolution templates, by what each one emits and what it tiles over.
#:
#: Each entry is a kernel that does not exist here yet, so each is registered as
#: a name with the geometry it would use rather than as a choice that pretends
#: to be measurable.  A template whose kernel is missing must not appear in a
#: candidate list: the list is a measurement plan, and a plan that names a
#: kernel that cannot be built is a lie told to whoever reads the result.
CONV_TEMPLATES: dict[str, dict[str, Any]] = {
    "convolution2d": {
        "grid": _conv_forward_grid,
        "rows": "batch times the output's spatial extent",
        "cols": "output channels per group",
        "contraction": "input channels per group",
    },
    "convolution3d": {
        "grid": _conv_forward_grid,
        "rows": "batch times the output's spatial extent",
        "cols": "output channels per group",
        "contraction": "input channels per group",
    },
    "depthwise_conv1d": {
        "grid": None,
        "tiling": "output positions, positions along the axis, channels",
        "not": "a product: each channel is reduced on its own",
    },
    "convolution2d_bwd_input": {
        "grid": _conv_bwd_input_grid,
        "rows": "batch times the input's spatial extent",
        "cols": "input channels per group",
    },
    "convolution2d_bwd_weight": {
        "grid": _conv_bwd_weight_grid,
        "rows": "output channels per group",
        "cols": "input channels per group",
        "contraction": "batch times the input's spatial extent",
    },
}


CONV = ConvTemplate()


#: Templates by the name their operators are declared under.
TEMPLATES: dict[str, KernelTemplate] = {GEMM.name: GEMM, CONV.name: CONV}


def template_for(name: str) -> KernelTemplate | None:
    return TEMPLATES.get(name)


#: Imported for the names callers expect to find here.
from .loops import TemplateKernel  # noqa: E402
