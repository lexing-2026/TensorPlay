"""The launch configurations a device is measured with, and the tables they come from.

A tile shape here is what the tile would like; the heuristics decide what it gets
once the problem's size is known.  Splitting the two is what lets one table serve
every call of an operation and still yield a different set of candidates per shape.
The names are the ones the device's own configuration space uses, so a template
asks for a configuration by what it is rather than by which table holds it, and
which table a device reads is declared beside the tables themselves.
"""

from __future__ import annotations

from .ir import next_power_of_2
from .. import config
from ..heuristics.template.base import TemplateConfigHeuristics

from dataclasses import dataclass, field

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
class FlexDecodeConfig:
    """A tile for attention that asks one question at a time.

    No contraction tile: there is no contraction here.  The query axis is one
    row wide, so the only extent worth choosing is how far along the keys one
    program walks, and the two things that go with it.
    """

    block_n: int
    num_stages: int
    num_warps: int


@dataclass
class GemmConfig(BaseConfig):
    """A product's tile shape, and how many tiles share their operands.

    The flat tile index is decoded in groups of ``group_m`` rows before
    advancing a column, so that the programs running at the same moment are
    reading the same rows of the left operand and the same columns of the
    right one.  Without the grouping consecutive programs walk down a column
    and share nothing, and the operand traffic is paid once per tile instead
    of once per group.
    """

    group_m: int = field(default=8, kw_only=True)
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
    #: Extra hints tried in turn, for shapes the framework can guess at.  A
    #: hint stands in for a size the caller knows and the graph does not, so
    #: each one is a different guess at the same problem, and a configuration
    #: fitted to a guess is offered alongside the one fitted to what is known.
    multi_kernel_hints: tuple = ()
    #: Whether the search is over the table as written or over everything the
    #: table implies.  The exhaustive search prunes what would spill, which
    #: costs measurements and saves the ones that cannot win.
    search_space: str = "DEFAULT"

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

    def get_shared_memory_estimation(
        self,
        config: BaseConfig,
        dtype_size: int,
        has_sm_layout_conversion: bool = False,
        layout_conversion_byte_size: int = 0,
    ) -> int:
        """What one tile costs in shared memory, counted before it is built.

        A tile stages both operands' blocks once per stage of the pipeline, and
        stages them again for the next stage, so the stages multiply.  The
        barrier a boundary tile needs is counted as a flat cost because it does
        not scale with the tile.  The layout conversion is counted where it
        happens: on the accumulator when a bias follows it, on the result when
        a store does.
        """

        staged = dtype_size * (
            config.block_m * config.block_k + config.block_n * config.block_k
        )
        if has_sm_layout_conversion and layout_conversion_byte_size:
            padding = 128 // (layout_conversion_byte_size * 8) or 1
            epilogue = (
                layout_conversion_byte_size * config.block_m * (padding + config.block_n)
            )
        else:
            epilogue = 0
        return staged * config.num_stages + epilogue + 128

    def _get_exceeding_shared_memory_checker(self, has_sm_layout_conversion: bool,
                                              layout_conversion_byte_size: int):
        """A test for whether a tile outgrows the device's shared memory.

        Returned as a closure rather than a filter so that a device which does
        not report the figure yields ``None`` and the table is left alone: an
        unknown limit is not a limit, and refusing everything on a device that
        merely declines to say would leave it with no candidates at all.
        """

        available = shared_memory_per_block()
        if not available:
            return None

        def exceeds(config: BaseConfig, dtype_size: int) -> bool:
            return self.get_shared_memory_estimation(
                config, dtype_size, has_sm_layout_conversion, layout_conversion_byte_size
            ) > available

        return exceeds

    def _prune_exceeding_max_shared_mem_configs(
        self,
        configs: list,
        dtype_size: int,
        has_sm_layout_conversion: bool = False,
        layout_conversion_byte_size: int = 0,
    ) -> list:
        """Drop the tiles the device could not stage.

        The estimate is an upper bound, so this over-prunes rather than
        under-prunes -- a tile it keeps might still not fit, and a tile it
        drops certainly does not.
        """

        if dtype_size <= 0:
            return configs
        exceeds = self._get_exceeding_shared_memory_checker(
            has_sm_layout_conversion, layout_conversion_byte_size
        )
        if exceeds is None:
            return configs
        return [c for c in configs if not exceeds(c, dtype_size)]

    def _prune_reg_spill_configs(self, configs: list) -> list:
        """Drop the tiles whose accumulator alone outgrows the register file.

        The accumulator is held in registers for the whole contraction, so a
        tile whose accumulator needs more registers per thread than the file
        has will spill, and a spilled accumulator turns the inner loop into
        memory traffic.  The count here is a lower bound: a tile it keeps may
        still spill once the rest of the kernel's live values are counted.
        """

        warp_size = warp_size_per_thread()
        if not warp_size:
            return configs
        kept = []
        for config in configs:
            per_thread = -(-config.block_m * config.block_n // (config.num_warps * warp_size))
            if per_thread > _REGISTERS_PER_THREAD:
                continue
            kept.append(config)
        return kept

    def _filter_configs(self, configs: list) -> list:
        """Whatever this device rules out before anything is fitted."""

        return configs

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
        dtype_size: int = 0,
        has_sm_layout_conversion: bool = False,
        layout_conversion_byte_size: int = 0,
    ) -> Iterator[TritonConfig]:
        """The tile shapes worth running for a product of this size.

        Three narrowings happen here and each answers a different question:
        the table is filtered by what this device rules out, each entry is
        clamped to the shape it will run against, and what is left is measured
        against what the device can afford to stage and to hold in registers.
        """

        configs = self._filter_configs(configs)
        fitted = self._scale_configs(
            m, n, k, configs, scale, exclude, has_int8_tensor
        )
        fitted = self._prune_exceeding_max_shared_mem_configs(
            fitted, dtype_size, has_sm_layout_conversion, layout_conversion_byte_size
        )
        if self.search_space == "EXHAUSTIVE":
            fitted = self._prune_reg_spill_configs(fitted)
        return self._finalize(fitted)

    def triton_config(self, num_stages: int, num_warps: int, **kwargs) -> TritonConfig:
        return TritonConfig(kwargs, num_stages, num_warps)

    def get_mm_configs(self) -> Callable[..., Iterator[TritonConfig]]:
        """A generator of tile shapes for a product, over this table."""

        return lambda m, n, k, **kwargs: self.preprocess_mm_configs(
            m, n, k, self.mm_configs, op_name="mm", **kwargs
        )

    def get_exhaustive_configs(self) -> Callable[..., Iterator[TritonConfig]]:
        """A generator over the whole space, rather than over the table."""

        import itertools

        table = tuple(
            type(self.mm_configs[0])(m, n, k, stages, warps)
            for m, n, k, stages, warps in itertools.product(
                self.exhaustive_extents,
                self.exhaustive_extents,
                self.exhaustive_extents,
                self.exhaustive_stages,
                self.exhaustive_warps,
            )
        )
        return lambda m, n, k, **kwargs: self.preprocess_mm_configs(
            m, n, k, table, op_name="mm", **kwargs
        )

    def get_extra_configs(self) -> Callable[..., Iterator[TritonConfig]]:
        """A generator over the table a learned heuristic would add to.

        A heuristic that has learned from measurements can name a tile the
        written table does not, so the candidates it would add are kept
        separately: they are what the learned answer is allowed to choose, and
        keeping them apart is what stops the written table from growing into
        everything anyone has ever measured.
        """

        return lambda m, n, k, **kwargs: self.preprocess_mm_configs(
            m, n, k, self.extra_configs, op_name="mm", **kwargs
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

    def get_flex_decode_configs(
        self, head_dim: int, dtype: Any
    ) -> list[FlexDecodeConfig]:
        """The tilings for attention that asks one question at a time.

        Three when the widest search was asked for, and the one that is always
        there otherwise.  The three are the same width walked at three different
        depths: a narrow walk needs more steps to cross the keys and a wide one
        wastes a row that is mostly masked away, and which is better is a fact
        about how many keys there are -- so all three are offered and measured
        rather than one being chosen here.

        The one always offered is a walk of sixty-four at the shallowest depth,
        because a kernel that is not measured at all has to be able to run.
        """

        flex_decode_configs: list[FlexDecodeConfig] = []

        if config.max_autotune:
            if config.max_autotune_flex_search_space == "EXHAUSTIVE":
                return self.exhaustive_flex_decode_configs
            flex_decode_configs += self.flex_decode_autotune_configs

        default_config = FlexDecodeConfig(block_n=64, num_stages=1, num_warps=2)

        if default_config not in flex_decode_configs:
            flex_decode_configs.append(default_config)

        return flex_decode_configs

    #: The three tilings offered when the widest search was asked for, read as
    #: (keys per program, stages, warps).
    flex_decode_autotune_configs: tuple = (
        FlexDecodeConfig(64, 3, 2),
        FlexDecodeConfig(32, 3, 2),
        FlexDecodeConfig(128, 3, 2),
    )
    #: Every tiling the table implies rather than names, for the exhaustive
    #: search.
    exhaustive_flex_decode_configs: tuple = tuple(
        FlexDecodeConfig(block_n, num_stages, num_warps)
        for block_n in (16, 32, 64, 128)
        for num_stages in (1, 3, 4, 5)
        for num_warps in (2, 4, 8)
    )

    #: The product tilings.  Read as (M, N, K, stages, warps).
    mm_configs: tuple = ()
    #: The convolution tilings, read the same way.
    conv_configs: tuple = ()
    #: Every tile the table implies rather than names, for the exhaustive
    #: search.  Written as the product of the extents and the launch geometry
    #: so that the space is stated once instead of transcribed; it is large on
    #: purpose, because the exhaustive search is what prunes the tiles that
    #: would spill before spending a measurement on them.
    exhaustive_extents: tuple = (16, 32, 64, 128, 256)
    exhaustive_stages: tuple = (1, 2, 3, 4, 5)
    exhaustive_warps: tuple = (2, 4, 8)
    #: The depthwise tilings, which are (N, L, C, stages, warps).
    depthwise_conv_configs: tuple = ()
#: How wide each element type is, which is what the shared-memory estimate is
#: in terms of.  It is asked of the type rather than of the tensor because a
#: heuristic sizing itself to a problem is not holding the tensor.
DTYPE_SIZES = {
    "float64": 8, "float32": 4, "float16": 2, "bfloat16": 2,
    "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "bool": 1,
}


def dtype_size(dtype) -> int:
    """How many bytes one element of this type occupies, or zero if unknown.

    Zero rather than a guess: the estimate it feeds is in bytes, so a type
    whose width is not known makes the estimate unknown, and an unknown
    estimate is not a limit anything may be pruned against.
    """

    if dtype is None:
        return 0
    name = getattr(dtype, "name", None) or str(dtype)
    return DTYPE_SIZES.get(name, 0)


#: The registers a thread has, which bounds an accumulator held in them.
_REGISTERS_PER_THREAD = 255


def shared_memory_per_block() -> int:
    """What a block may ask the device for, or zero when it will not say."""

    import tensorplay as tp

    try:
        props = tp.cuda.get_device_properties(0)
    except Exception:  # noqa: BLE001 - a device that will not say has no stated limit
        return 0
    for name in ("shared_memory_per_block_optin", "shared_memory_per_block",
                 "local_mem_size"):
        value = getattr(props, name, None)
        if value:
            return int(value)
    return 0


def warp_size_per_thread() -> int:
    """How many threads a warp has, which is what a block's work divides by."""

    import tensorplay as tp

    try:
        # No fallback: the device reports this, and a figure invented here
        # would be a register-file bound nobody chose.
        return int(tp.cuda.get_device_properties(0).warp_size)
    except Exception:  # noqa: BLE001 - a device that will not say has no stated figure
        return 0


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

    #: The tiles a learned heuristic is allowed to add to the written table.
    extra_configs = (
        GemmConfig(16, 32, 16, 3, 2),
        GemmConfig(16, 32, 32, 4, 2),
        GemmConfig(16, 32, 32, 5, 2),
        GemmConfig(64, 64, 128, 3, 4),
        GemmConfig(128, 64, 32, 2, 2),
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
