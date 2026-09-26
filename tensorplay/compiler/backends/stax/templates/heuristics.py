"""Which configurations of one template are worth trying for one call.

The table says what exists and the rule below says which of it fits here.  A tile
wider than the problem it tiles is wasted lanes and a mask paid for on every step, so
each configuration is narrowed to the shape it will actually run against.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Iterator, Sequence

from .config import (
    BaseConfig,
    ConvConfig,
    DepthwiseConvConfig,
    GemmConfig,
    TritonConfig,
)
from .ir import next_power_of_2

class SymbolicGridFn:
    """A grid function whose extents are symbolic rather than concrete.

    A grid is asked for its shape before the shapes it depends on are known --
    a tile that will be measured against several problems has to be able to say
    how many programs it would take for each of them.  So the extents arrive
    as expressions, and the function returns expressions; the caller
    substitutes numbers when it finally launches.

    The decoration is what records that a grid was written this way, so that
    code which needs the distinction can see it without inferring it from the
    body.
    """

    def __init__(self, fn):
        self.fn = fn
        self.symbolic = True
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __get__(self, instance, owner=None):
        return self
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
