"""How a launch is described to the runtime: its grid, and its tuning choices.

A launch is set up in two passes that cannot be done in one.  The first pass
decides the tuning: how many warps, how many stages, how the work is blocked.
The second reads that tuning back and works out the grid -- how many blocks in
each direction -- because the grid depends on block sizes the first pass chose.
So the tuning is recorded first and the grid is worked out from it afterwards,
by matching the block sizes a launch actually used against the ones recorded.

Recording the grid rather than the formula that produced it is what lets the
matching work: the recorded entries say which block sizes they are for, so a
launch finds the entry that is its own.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import enum
import functools
import itertools
import hashlib
import logging
import math
import os
import re
import threading
import time
from typing import Any, Callable, Container, Final, Generic, Literal, TypeVar, cast

import tensorplay as tp

from .. import metrics
from ..utils import (
    compute_required_storage_length,
    counters,
    GPU_KERNEL_BIN_EXTS,
    TMA_ALIGNMENT,
    XPU_KERNEL_FORMAT,
    ceildiv,
    tlx_only_cuda_options,
    triton_version_uses_attrs_dict,
)
from .hints import HeuristicType, DeviceProperties, TritonMeta
from ..triton_bundler import TritonBundler
from .benchmarking import benchmarker
from .. import config
from .cache_dir_utils import triton_cache_dir
from .runtime_utils import triton_config_to_hashable
from .triton_compat import IntelGPUError, OutOfResources, PTXASError
from ..compile_log import timed_block
from .triton_compat import (
    ASTSource,
    GPUTarget,
    HAS_WARP_SPEC,
    CompiledKernel,
    Config,
    KernelInterface,
    knobs,
    statically_launched_kernel_by_device,
    triton,
)
from .triton_helpers import get_constexprs
from .runtime_utils import (
    create_bandwidth_info_str,
    get_first_attr,
    get_max_y_grid,
    get_num_bytes,
    triton_hash_to_path_key,
    validate_triton_config,
)
from .coordinate_descent_tuner import CoordescTuner
from .....graph.experimental.sympy_functions import OrderedSet

#: What a compiled kernel is launched through: a function of the same values,
#: written out at the point the kernel was compiled rather than assembled at
#: each call, because assembling it per call is the cost the writing avoids.
LauncherType = Callable[..., Any]

#: What a compiled kernel is: the runtime's own compiled form, whose shape the
#: runtime decides and which this module only carries.
_T = TypeVar("_T")

log = logging.getLogger(__name__)

#: What was measured, kept apart from the rest of the log because it is only
#: wanted when a measurement is being questioned, and there is a great deal of
#: it.
autotuning_inputs_log = tp.getArtifactLogger(__name__, "autotuning_inputs")


def config_to_dict(config: Config) -> dict[str, Any]:
    """One tuning choice as a plain mapping.

    The choice is spread over three places -- the keyword arguments, the warp
    count and the stage count -- and a caller that wants to read it as one
    thing should not have to know that.  The warp-specialization knobs are
    included only where they exist, because a runtime without them has no
    setting to report and reporting a zero would read as "asked for zero".
    """

    config_dict = {
        **config.kwargs,
        "num_warps": config.num_warps,
        "num_stages": config.num_stages,
    }
    if HAS_WARP_SPEC:
        config_dict.update(
            {
                "num_consumer_groups": getattr(config, "num_consumer_groups", 0),
                "num_buffers_warp_spec": getattr(config, "num_buffers_warp_spec", 0),
            }
        )
    return config_dict


class BenchmarkFailureReason(enum.Enum):
    """Why measuring one configuration of a kernel did not produce a time.

    A configuration that cannot be measured is not the same as one that measured
    slow, and treating the two alike would let a kernel that cannot run at all
    look like the slowest one -- which is a decision rather than a measurement.
    So a configuration that spills its registers, or that the compiler refused,
    reports why instead of reporting a time.
    """

    REGISTER_SPILLING = "register_spilling"
    INVALID_CONFIG = "invalid_config"


class InductorConfig(Config):
    """A configuration with this project's own switches alongside the tuning.

    A configuration says how to run a kernel; the switches here say how much
    latitude to give the compiler over the schedule, which is a choice about the
    measurement rather than about the kernel and so travels beside it.
    """

    def __init__(self, *args, dynamic_scale_rblock=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_scale_rblock = dynamic_scale_rblock


class NoTritonConfigsError(RuntimeError):
    """No configuration of this kernel could be offered at all.

    Raised rather than returning nothing, because a kernel with no
    configurations is not a kernel that measured badly -- it is a kernel nobody
    wrote a tile for, and a caller that treated it as the former would go
    looking for a faster tile rather than for the tile that is missing.
    """


@dataclasses.dataclass
class GridExpr:
    """The grid a launch runs with, worked out from the tuning it was given.

    Subclasses say how the three grid extents follow from the block sizes a
    launch used.  Two languages are written here, because a grid is spelled
    into either generated Python or generated C++, and the two disagree about
    how a negative number divides -- which is the one thing a ceiling is
    written out of.
    """

    inductor_meta: dict
    mode: Literal["python", "cpp"] = "python"
    prefix: list[str] = dataclasses.field(default_factory=list)
    x_grid: str | int = 1
    y_grid: str | int = 1
    z_grid: str | int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("python", "cpp"):
            raise AssertionError(f"mode must be 'python' or 'cpp', got {self.mode!r}")

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        raise NotImplementedError

    def ceildiv(self, numel: str | int, block: int | str | None) -> str | int:
        """How many blocks of ``block`` cover ``numel`` elements.

        A block of one covers anything in one block, so the count is the
        element count.  Two numbers already known are divided here rather than
        in the generated source, where the division would be repeated on every
        launch.  What is left is written as a division that rounds up: in
        Python that is spelled through a negative divisor, because Python
        rounds a negative division down, and in C++ it is spelled as an
        addition before an ordinary division, because C++ rounds toward zero
        and there is no negative-divisor rule to lean on.
        """

        if block is None or block == 1:
            return numel
        if isinstance(numel, int) and isinstance(block, int):
            return ceildiv(numel, block)
        if self.mode == "python":
            return f"-(({numel}) // -({block}))"
        return f"(({numel} + ({block} - 1)) / ({block}))"

    def maximum(self, seq: list[int | str]) -> int | str:
        """The largest of a list, with the known numbers decided here.

        A list that is one long, or that was all known numbers, is answered
        outright; otherwise the numbers are folded in first so the written form
        is not longer than it has to be.
        """

        items = self._constant_fold(max, seq)
        if len(items) <= 1:
            return items[0]
        if self.mode == "python":
            return f"max({', '.join(map(str, items))})"
        # A literal and a variable cannot meet in the same standard-library
        # maximum without the type being deduced wrong, so a literal is
        # written as the variable's width.
        cpp_items = [f"(long){x}" if isinstance(x, int) else str(x) for x in items]
        return functools.reduce(lambda x, y: f"std::max({x}, {y})", cpp_items)

    def summation(self, seq: list[int | str]) -> int | str:
        """The sum of a list, with the known numbers decided here."""

        items = self._constant_fold(sum, seq)
        if len(items) <= 1:
            return items[0]
        return " + ".join(map(str, items))

    def product(self, seq: list[int | str]) -> int | str:
        """The product of a list, with the known numbers decided here."""

        items = self._constant_fold(math.prod, seq)
        if len(items) <= 1:
            return items[0]
        return " * ".join(map(str, items))

    def _constant_fold(
        self, fn: Callable[[list[int]], int], seq: list[int | str]
    ) -> list[int | str]:
        """Add up the known numbers of a list through a function that is
        indifferent to the order they are in.

        Folding them first is what keeps the written form short: a list of a
        hundred block sizes with two known among them becomes a sum of two
        known numbers and ninety-eight names, rather than a chain a hundred
        long.
        """

        items: list[int | str] = [x for x in seq if not isinstance(x, int)]
        const_items = [x for x in seq if isinstance(x, int)]
        if const_items:
            items.append(fn(const_items))
        return items

    def assign_tmp(self, name: str, expr: str | int) -> str:
        # One grid per kernel, so a name cannot collide with another kernel's.
        if self.mode == "python":
            return f"{name} = {expr}"
        if self.mode == "cpp":
            return f"uint32_t {name} = {expr};"
        raise AssertionError(f"invalid mode {self.mode}")

    @staticmethod
    def from_meta(inductor_meta, cfg, mode="python"):
        """The grid a launch recorded asking for, built from the name it recorded.

        A launch says which kind of grid it wants by name rather than by being
        handed one, because the launch is written out as text and rebuilt on the
        other side; a name survives that and an object does not.  So the name is
        looked up here, and a name that is not a grid is refused rather than
        quietly treated as a one-block grid.
        """

        grid_cls = globals()[inductor_meta["grid_type"]]
        if not (isinstance(grid_cls, type) and issubclass(grid_cls, GridExpr)):
            raise AssertionError(f"Expected GridExpr subclass, got {grid_cls}")
        grid = grid_cls(inductor_meta=inductor_meta, mode=mode)
        if isinstance(cfg, Config):
            cfg = config_to_dict(cfg)
        grid.generate(cfg)
        return grid

    def generate_lazy(self, kernel_name: str) -> None:
        """Build this grid for a kernel that will be configured later.

        Which extents a launch runs with is not known when the launch is
        written, only when it runs, so here the block sizes are named after
        where the result will be found rather than given as values.  The
        checks a real value would be checked against are left to that point:
        what is being written now cannot fail, and a check that cannot fail
        only says so.
        """

        meta: dict[str, Any] = {
            "XBLOCK": f"{kernel_name}_result.xblocks[0]",
            "YBLOCK": f"{kernel_name}_result.yblocks[0]",
            "ZBLOCK": f"{kernel_name}_result.zblocks[0]",
            "R0_BLOCK": f"{kernel_name}_result.r0blocks[0]",
            "RSPLIT": f"{kernel_name}_result.rsplit",
            "RSPLIT_SIZE": f"{kernel_name}_result.rsplit_size",
        }
        # assertions are done based on real values, so we can skip here
        self.generate(meta, is_lazy=True)

    @classmethod
    def from_meta_lazy(
        cls,
        inductor_meta,
        kernel_name: str,
    ) -> "GridExpr":
        """The grid a launch asked for by name, for a kernel configured later."""
        if inductor_meta is None:
            raise AssertionError("inductor_meta must be specified for lazy compile")
        grid_type = inductor_meta.get("grid_type", None)
        if grid_type is None:
            raise AssertionError("grid_type must be specified for lazy compile")
        grid_cls = globals()[grid_type]
        if not issubclass(grid_cls, GridExpr):
            raise AssertionError(f"Expected GridExpr subclass, got {grid_cls}")
        grid = grid_cls(inductor_meta=inductor_meta, mode="cpp")
        grid.generate_lazy(kernel_name)
        return grid

    def eval_slow(self, meta: dict[str, int]):
        """The three extents as numbers, worked out here rather than at launch.

        A caller that wants to know how large the grid would be -- to decide
        whether a shape is worth measuring, or to report what a decision cost
        -- asks here.  The same text the launcher runs is run here, so the two
        cannot report different sizes for the same launch.
        """

        scope = {**meta}
        for line in self.prefix:
            exec(line, scope)
        exec(f"grid_0 = {self.x_grid}", scope)
        exec(f"grid_1 = {self.y_grid}", scope)
        exec(f"grid_2 = {self.z_grid}", scope)
        return scope["grid_0"], scope["grid_1"], scope["grid_2"]

    @staticmethod
    def resolve_dict(
        val: dict[str, int] | Literal["empty"],
        default: int | None = None,
    ) -> int:
        """Read one recorded grid extent.

        A recorded extent may be the word "empty", standing for a direction
        with nothing in it, which is one block rather than no blocks: a launch
        with a zero extent would have nothing to run and would not be launched
        at all.
        """

        if val == "empty":
            assert default is not None
            return default
        assert isinstance(val, int)
        return val


class MixOrderReductionGrid(GridExpr):
    """A reduction whose partial sums are combined by a kernel of its own.

    The split has to divide the work evenly, or the combine step waits on a
    partial sum that is shorter than the others and the whole reduction waits
    with it.  So the split is refused here rather than producing a launch that
    is correct and slow.
    """

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        split_size = meta.get("RSPLIT_SIZE")
        xblock = meta.get("XBLOCK")
        if not is_lazy:
            if not split_size:
                raise AssertionError("Missing RSPLIT_SIZE")
            if not xblock:
                raise AssertionError("Missing XBLOCK")
            if split_size % xblock != 0:
                raise AssertionError(f"{split_size=}, {xblock=}")
        self.x_grid = self.ceildiv("xnumel", split_size)


class CooperativeReductionGrid(GridExpr):
    """A reduction whose partial sums are combined by the blocks themselves.

    Nothing is launched afterwards, so the second axis is the work and the
    first is the split, and the two are not free to trade places: the number
    of partial sums is what the first axis was chosen to be.
    """

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = str(meta["RSPLIT"])
        self.y_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))


class SplitScanGrid(GridExpr):
    """A scan whose sequential part is carried by a kernel of its own.

    One element per block along the sequential axis, because a block that
    took two would have to order them itself, which is the work the separate
    kernel is there to do.
    """

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        if not is_lazy:
            if meta.get("XBLOCK", 1) != 1:
                raise AssertionError(
                    f"Expected XBLOCK == 1 for SplitScanGrid, got {meta.get('XBLOCK', 1)}"
                )
        self.x_grid = self.ceildiv("r0_numel", meta.get("R0_BLOCK"))
        self.y_grid = "xnumel"


class FixedGrid(GridExpr):
    """A grid the caller already worked out and is handing over as it is.

    Nothing is derived here: the three extents were computed by whoever built
    the launch, so this only has to say that the launcher will receive them
    rather than compute them.
    """

    @staticmethod
    def setup_grid_as_args() -> dict[str, Any]:
        """The record that tells the launcher to take the grid as arguments."""

        return {
            "grid_type": FixedGrid.__name__,
            "fixed_grid": ["_grid_0", "_grid_1", "_grid_2"],
            "extra_launcher_args": ["_grid_0", "_grid_1", "_grid_2"],
        }

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid, self.y_grid, self.z_grid = self.inductor_meta["fixed_grid"]


class PrecomputedGrid(GridExpr):
    """A grid worked out once per tuning and looked up rather than redone.

    Working a grid out costs as much as the launch it is for, so it is done
    once for each tuning and kept.  A launch then finds its entry by the block
    sizes it is running with, which is what the entry records.
    """

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        for candidate in self.inductor_meta["precomputed_grids"]:
            if all(meta.get(k) == v for k, v in candidate["config"].items()):
                self.x_grid, self.y_grid, self.z_grid = candidate[self.mode]
                return
        raise AssertionError(
            f"no recorded grid for {meta} among "
            f"{self.inductor_meta['precomputed_grids']}"
        )


@dataclasses.dataclass
class Grid1D(GridExpr):
    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))


@dataclasses.dataclass
class Grid2D(GridExpr):
    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))


@dataclasses.dataclass
class Grid3D(GridExpr):
    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
        self.z_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))


@dataclasses.dataclass
class BatchMatmulGrid3D(GridExpr):
    """A batched product, whose three extents arrive in the third axis first.

    The batch is the outermost thing being iterated and so is handed to the
    outermost axis, which is what lets a batch that does not fit one axis be
    spread over two without the tiles being renumbered to make room.
    """

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.z_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
        self.x_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))


@dataclasses.dataclass
class Grid2DWithYZOverflow(GridExpr):
    """Two nominal axes, where the second may not fit the one it was given.

    A launch whose second axis outgrows what that axis accepts fails rather than
    growing, so the extent is divided by the limit first and the quotient becomes
    a third axis.  The division is guarded because a count of zero would
    otherwise ask for a grid of zero blocks on an axis that has to be at least
    one.
    """

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.prefix.extend(
            [
                self.assign_tmp(
                    "y_grid_raw_", self.ceildiv("ynumel", meta.get("YBLOCK"))
                ),
                self.assign_tmp(
                    "y_grid_div_", self.ceildiv("y_grid_raw_", get_max_y_grid())
                ),
            ]
        )
        ceildiv_expr = self.ceildiv("y_grid_raw_", "y_grid_div_")
        if self.mode == "python":
            self.y_grid = f"(0 if y_grid_div_ == 0 else {ceildiv_expr})"
        else:
            self.y_grid = f"(y_grid_div_ == 0 ? 0 : {ceildiv_expr})"
        self.z_grid = "y_grid_div_"


def check_autotune_cache(
    configs: list,
    filename: str | None,
    inductor_meta,
    dynamic_scale_rblock_eligible: bool = False,
):
    """The configurations to measure, narrowed to one if an answer is already known.

    A recorded answer is only worth reading when there is something to record
    it against, so the lookup is skipped entirely when there is one
    configuration, when caching is off, or when the answer would have been
    produced by interpreting rather than compiling.

    What the lookup found is written into the metadata either way -- hit or
    miss -- because "it looked and there was nothing" is the answer to a
    different question from "it did not look".
    """

    from ..cache_key import AUTOTUNE_CACHE_KEY_STRATEGY  # noqa: F401
    from .autotune_cache import AutotuneCache
    from .runtime_utils import triton_config_to_hashable
    from ..compile_worker import watchdog

    autotune_cache = None
    autotune_cache_info = {}
    disabled = inductor_meta.get("force_disable_caches", False)
    if (
        not disabled
        and filename is not None
        and (
            len(configs) > 1
            or inductor_meta.get("coordinate_descent_tuning")
            or dynamic_scale_rblock_eligible
        )
        and os.environ.get("TRITON_INTERPRET", "0") != "1"
    ):
        configs_hash = hash_configs(configs)

        watchdog.report_phase(watchdog.Phase.QUERYING_CACHE)
        autotune_cache = AutotuneCache.create(inductor_meta, filename, configs_hash)
        if autotune_cache:
            if best_config := autotune_cache.read_best(inductor_meta, configs):
                configs = [best_config]
                autotune_cache_info["best_config"] = triton_config_to_hashable(
                    best_config
                )
                autotune_cache_info["autotune_cache_state"] = "hit"

            else:
                autotune_cache_info["autotune_cache_state"] = "miss"
                autotune_cache_info["num_configs"] = len(configs)
                if inductor_meta.get("coordinate_descent_tuning"):
                    autotune_cache_info["coordesc_tuning"] = True
                    if len(configs) == 1:
                        # This is the config that coordinate descent tuning started at,
                        # which is not the same as the final config chosen (i.e.
                        # only_config, best_config)
                        autotune_cache_info["coordesc_tuning_start_config"] = (
                            triton_config_to_hashable(configs[0])
                        )
    else:
        if len(configs) == 1:
            autotune_cache_info["autotune_cache_state"] = "only 1 config"
            autotune_cache_info["only_config"] = triton_config_to_hashable(configs[0])

        if disabled:
            autotune_cache_info["autotune_cache_state"] = "force_disabled"
            log.debug("autotune caching is disabled by config.force_disable_caches")

    return configs, autotune_cache, autotune_cache_info


def hash_configs(configs: list):
    """A name for a set of configurations, so a change to any of them shows up.

    What a kernel was measured with is written down against this, so a
    configuration that has been changed must not find the old answer waiting
    for it.  The settings are sorted because a configuration written in a
    different order is the same configuration.
    """

    hasher = hashlib.sha256()
    for cfg in configs:
        hasher.update(
            f"{sorted(cfg.kwargs.items())} {cfg.num_warps} {cfg.num_stages}\n".encode()
        )
    return hasher.hexdigest()


@functools.lru_cache(None)
def _warn_host_tma_clone(name: str) -> None:
    log.warning(
        "host-side TMA: input %s is not %d-byte aligned; cloning it (an extra "
        "copy per launch). Pass aligned inputs to avoid this.",
        name,
        TMA_ALIGNMENT,
    )


def _host_tma_aligned(tensor, name):
    """A view of ``tensor`` that starts where a descriptor has to start.

    A descriptor describes memory by its address, and the device requires that
    address to be on a boundary.  A view that already is costs nothing; one that
    is not has to be copied into memory that is, and the copy is worth saying out
    loud once rather than paying for silently on every launch.
    """

    if tensor.data_ptr() % TMA_ALIGNMENT == 0:
        return tensor
    _warn_host_tma_clone(name)
    return tensor.clone()


def _resolve_dims(dims, cfg_kwargs, constants):
    """The extents of a descriptor, as numbers, or nothing if one is not known.

    A descriptor is described in terms of the tiling it was chosen under, so its
    extents are only numbers once that tiling is.  An extent that cannot be
    resolved is not guessed at: a descriptor built from a wrong extent would
    describe memory that is not the memory the kernel reads, which is worse than
    having no descriptor and falling back.
    """

    return _resolve_dims(dims, cfg_kwargs, constants)


def _resolve_dims(dims, cfg_kwargs, constants):
    """Block, shape and stride extents, worked out to whole numbers.

    A descriptor written before the configuration is known names its extents
    rather than giving them, because a descriptor is written once and compiled
    once per configuration.  A name that resolves in neither the configuration
    nor the constants is not a number yet, and the descriptor that needs it
    cannot be described: that is not a failure, it is a descriptor not built.
    """

    result = []
    for s in dims:
        if isinstance(s, int):
            result.append(s)
        elif isinstance(s, str) and s in constants:
            result.append(int(constants[s]))
        elif isinstance(s, str) and s in cfg_kwargs:
            result.append(int(cfg_kwargs[s]))
        else:
            log.debug("host-side TMA: unresolved descriptor dim %r; skipping", s)
            return None
    return result


def _should_enable_triton_debug_asserts(inductor_meta) -> bool:
    """Whether the compiler should check indirect indexing on this launch.

    Reading through a computed index is the one thing a kernel can do that
    turns a wrong answer into memory it does not own, so the check follows the
    request for it.  Where the runtime that would perform the check cannot be
    told to, asking for it would either be ignored or break the build, so it is
    not asked for there.
    """

    from ..utils import get_triton_version

    if not inductor_meta.get("assert_indirect_indexing", True):
        return False
    if not inductor_meta.get("is_hip", False):
        return True
    return get_triton_version() >= (3, 7)


class CompileResult(Generic[_T]):
    """A kernel that has been compiled, together with what it was compiled for.

    The compiled form on its own is not enough to launch: which tuning it was
    compiled for decides the grid, and what the compiler folded away decides
    which arguments the launch still passes.  So all three travel together, and
    a launch is written from this rather than from the compiled form alone.

    The launcher is written out as text and defined here rather than assembled
    at each call, because assembling it is most of what a launch does and doing
    it once per call is most of what a launch costs.
    """

    def __init__(self, kernel, config, compile_meta, inductor_meta):
        self.kernel = kernel
        self.config = config
        self.compile_meta = compile_meta
        self.inductor_meta = inductor_meta

    def make_launcher(self) -> LauncherType: ...

    def _host_tma_pre_runner_lines(self, runner_args, call_args):
        """Build the descriptors inside the launcher, in place of the tensors.

        A descriptor is made from the storage a tensor is in, and the storage has
        to be there when the launch happens rather than before it, so this is
        written into the launcher instead of being built by the caller.  The
        tensor names are then replaced by the descriptors, which is why the
        arguments come back as well as the lines.
        """

        host_tma_args = self.inductor_meta.get("host_tma_descriptor_args")
        pre_runner_lines: list[str] = []
        if not host_tma_args:
            return pre_runner_lines, runner_args
        cfg_kwargs = self.config.kwargs
        all_constants = self.compile_meta["constants"]
        for inner_name, desc_info in host_tma_args.items():
            if inner_name not in call_args or not isinstance(desc_info, dict):
                continue
            block_shape_vals = _resolve_dims(
                desc_info["block_shape"], cfg_kwargs, all_constants
            )
            shape_vals = _resolve_dims(desc_info["shape"], cfg_kwargs, all_constants)
            stride_vals = _resolve_dims(desc_info["strides"], cfg_kwargs, all_constants)
            if block_shape_vals is None or shape_vals is None or stride_vals is None:
                continue
            desc_var = f"{inner_name}_host_tma_desc"
            aligned_var = f"{inner_name}_aligned"
            pre_runner_lines.append(
                f'{aligned_var} = _host_tma_aligned({inner_name}, "{inner_name}")'
            )
            pre_runner_lines.append(
                f"{desc_var} = TensorDescriptor({aligned_var}, {shape_vals},"
                f" {stride_vals}, {block_shape_vals})"
            )
            runner_args = [desc_var if a == inner_name else a for a in runner_args]
        return pre_runner_lines, runner_args

    def _gen_launcher_code(
        self, scope, def_args, runner_args, pre_runner_lines=None
    ) -> LauncherType:
        """Write the function a launch goes through, and define it.

        The grid is computed inside the launcher rather than passed to it,
        because the grid depends on values that are not known until the launch
        happens: a template offered against several shapes is compiled once and
        launched once per shape, and the number of programs is a different
        number each time.
        """

        grid = GridExpr.from_meta(self.inductor_meta, self.config)
        lines = [
            f"def launcher({', '.join(def_args)}, stream):",
            *[f"    {line}" for line in grid.prefix],
            f"    grid_0 = {grid.x_grid}",
            f"    grid_1 = {grid.y_grid}",
            f"    grid_2 = {grid.z_grid}",
            *(f"    {l}" for l in (pre_runner_lines or [])),
            f"    runner({', '.join(runner_args)})",
        ]
        launcher_code = "\n".join(lines)
        exec(launcher_code, scope)
        launcher = scope["launcher"]
        # How many values the launcher takes is known here and is written onto
        # it, so that a caller passing the wrong number is told without having
        # to read the launcher back and count.
        launcher._expected_positional_count = len(def_args)
        return launcher

    def _get_arg_lists(self, arg_names, constexprs):
        """The names to pass, the names to accept, and the ones to drop.

        Three lists rather than one, because the three answer different
        questions: what the compiled kernel is called with, what the launcher
        is handed, and what neither wants because the compiler folded it away.
        A value the compiler folded is still something the caller has, so it is
        dropped from the call rather than from the interface.
        """

        compile_meta = self.compile_meta
        cfg = self.config
        known_constants = OrderedSet(
            arg for i, arg in enumerate(arg_names) if i in constexprs
        )

        # A constant of nothing is not the same as a constant of some value, and
        # a signature that has dropped it cannot be called with it.  A name is
        # dropped only when the compiler recorded it as nothing, it was not
        # already known to be a constant, and the signature does not have it --
        # anything else would drop a value the kernel still reads.
        none_args = OrderedSet(
            k
            for k, v in compile_meta["constants"].items()
            if v is None and k not in known_constants
        )
        none_args = none_args.difference(OrderedSet(compile_meta["signature"].keys()))

        def _convert_constant(constant):
            if isinstance(constant, str):
                return "r'" + constant + "'"
            else:
                return repr(constant)

        if triton_version_uses_attrs_dict():
            call_args = arg_names
            def_args = arg_names
            implicit_constants = OrderedSet(
                ("num_warps", "num_stages")
            ).union(OrderedSet(k for k in known_constants))
            if implicit_constants := implicit_constants & OrderedSet(
                compile_meta["constants"].keys()
            ):
                # The warp and stage counts are the runtime's to read rather
                # than the caller's to pass, so they are taken out of what the
                # launcher accepts and put back as the values recorded for them.
                def_args = [arg for arg in def_args if arg not in implicit_constants]
                repl = {
                    k: _convert_constant(compile_meta["constants"].get(k))
                    for k in implicit_constants
                }
                call_args = [repl.get(arg, arg) for arg in call_args]
        else:
            call_args = [
                arg
                for i, arg in enumerate(arg_names)
                if i not in constexprs and arg not in none_args
            ]
            cfg_dict = config_to_dict(cfg)
            def_args = [
                name
                for name in arg_names
                if name not in cfg_dict and name not in none_args
            ]

        if "extra_launcher_args" in self.inductor_meta:
            def_args = [*def_args, *self.inductor_meta["extra_launcher_args"]]

        return call_args, def_args, none_args


def _find_names(obj):
    """The names a kernel is bound to, which is the name it will be reported under.

    Only dictionaries are searched, because a generated module binds its
    kernels in its globals and that is where the real name lives.  Walking the
    stack as well would only add the incidental names of whatever local
    happened to be holding it, which are never the name it was generated under.
    """

    import gc

    obj_names = []
    for referrer in gc.get_referrers(obj):
        if isinstance(referrer, dict):
            for k, v in referrer.items():
                if v is obj:
                    obj_names.append(k)
    return obj_names


#: What every measured kernel reported, in the order they were measured.
collected_calls: list = []


def _combo_has_reduction_subkernel(inductor_meta: dict) -> bool:
    """Whether a kernel of several has one of them reducing.

    A kernel of several is told how big each of its parts is rather than being
    handed one set of extents, so it arrives here with no extents of its own to
    reason from; the per-part figures are in the record of which parts there
    are.  A kernel whose parts were stitched together has its reducing parts
    pinned when the text was written, so there is nothing to scale there
    either.
    """

    combo_meta = inductor_meta.get("combo_grid_meta")
    if combo_meta is None or "heuristic_0" not in combo_meta:
        return False
    if "stitched_num_warps" in combo_meta or "stitched_launch_candidates" in combo_meta:
        return False
    return any(
        combo_meta.get(f"heuristic_{i}") == "reduction"
        for i in range(combo_meta["num_kernels"])
    )


def _could_dynamic_scale_rblock(
    *,
    size_hints: list[int] | None,
    heuristic_type: Any,
    device_prop: Any,
    inductor_meta: dict,
) -> bool:
    """Whether a reducing kernel's block could be halved to fit more of them.

    A block is how much one thread works on at once, and how many such blocks
    fit on a device at once decides how much of the device is busy.  When a
    block is large enough that fewer of them fit than there is room for, a
    smaller one is a way of using the device better.  That is a question about
    the device and about the kind of kernel, and it is worth asking only where
    the answer would be acted on.
    """

    return (
        device_prop is not None
        and not inductor_meta.get("deterministic", False)
        and inductor_meta.get("dynamic_scale_rblock", True)
        and not inductor_meta.get("persistent_reduction")
        and heuristic_type == HeuristicType.REDUCTION
        and (size_hints is not None or _combo_has_reduction_subkernel(inductor_meta))
        and device_prop.type in ["cuda", "hip"]
        and bool(device_prop.major)
        and (device_prop.major >= 8 or tp.version.hip)
        and device_prop.regs_per_multiprocessor is not None
        and device_prop.warp_size is not None
    )


#: What a plugin hook returns to say it has no opinion, so the next plugin is
#: asked and failing all of them the ordinary behaviour happens.  Compared
#: against by identity, so it is a particular object rather than a value: any
#: value would do, and one that could be produced by accident would not mean
#: what it says.
DEFER: Final[object] = object()


class _ConstRepr:
    """Something whose text is fixed, answering as though it were computed.

    A function that prints itself is how the kernel runtime names a compiled
    form in a message.  When the function itself is not travelling to another
    process, that name still has to, so it is carried as this instead: it
    answers the same question with the same text, without needing the thing it
    was printed from.
    """

    def __init__(self, value: str):
        self.value = value

    def __call__(self, _=None) -> str:
        return self.value


class CachingAutotunerPlugin:
    """Something that gets to answer for an autotuner before it acts.

    A hook returns :data:`DEFER` to say it has no opinion, and anything else to
    answer instead of the ordinary behaviour.  Which hooks there are, and what
    answering one obliges a plugin to do itself, is written on each of them: a
    hook that answers early owns everything that would have followed it.
    """

    def pre_compile(self, autotuner: CachingAutotuner) -> object:
        """Fires at the top of a compile, before anything is compiled.

        Deferring runs the ordinary compile: every configuration is compiled and
        a launcher is made of each.  Answering takes compilation and launcher
        creation over entirely, including what would be recorded about them,
        which is why a plugin that only wants to watch cannot answer here.
        """

        return DEFER

    def pre_dispatch(self, autotuner: CachingAutotuner, *args: Any, stream: Any, **kwargs: Any) -> object:
        """Fires before a kernel is dispatched, ahead of compiling and measuring.

        A kernel that has settled on one launcher answers later launches straight
        from it, past this hook, so a plugin that must see every launch cannot
        count on this one alone.
        """

        return DEFER

    def pre_autotune(self, autotuner: CachingAutotuner, *args: Any, stream: Any, **kwargs: Any) -> object:
        """Fires once there is more than one launcher, in place of measuring them.

        Deferring measures them.  Answering takes over the whole remainder of the
        dispatch: the plugin measures, launches, records the winner, saves the
        binary, applies whatever comes after measuring, and returns what the
        launch returned.  A plugin that only wants to affect which launcher is
        chosen can say so by changing the launchers and deferring.
        """

        return DEFER


def get_caching_autotuner_plugins(autotuner: CachingAutotuner) -> list[CachingAutotunerPlugin]:
    """The plugins that apply to this kernel, in the order they are asked.

    A plugin is added here, and only here, so that what is in force for a
    kernel is decided in one place.  Each is behind its own setting, and what a
    plugin needs to be imported is imported inside the branch that wants it, so
    a kernel pays for a plugin only when that plugin is in force.
    """

    plugins: list[CachingAutotunerPlugin] = []
    return plugins


def _resolve_load_device(device: int | None, device_type: str) -> int | None:
    """The device a binary should be loaded onto, when it was not pinned to one.

    A binary compiled without naming a device is the same binary everywhere, so
    which device it belongs to is settled when it is loaded rather than when it
    was built: what it is loaded onto is the device this process is on.  That is
    what makes one binary serve a machine with several devices.

    A processor has no device to be on, and asking one for its current device
    would ask it for something it does not have, so a device that was named, and
    a processor, are both taken at what they were given.
    """

    if device is not None or device_type == "cpu":
        return device

    from .benchmarking import get_interface_for_device

    return get_interface_for_device(device_type.replace("hip", "cuda")).current_device()


class CachingAutotuner(KernelInterface):
    """A kernel with several configurations, each compiled, the best one kept.

    The configurations here are not chosen by a rule: every one of them is
    compiled and measured, and the one that measured fastest is what a launch
    goes through.  Nothing is invalidated when the process is restarted -- a
    configuration that won once is written down and used without measuring
    again -- and every configuration is compiled ahead of the first launch
    rather than on it, so a launch is a launch and not a compile.

    What is kept is the compiled form and the launcher built from it, not the
    text: the text is what produced them, and a launch needs the two former.
    """

    def __init__(
        self,
        fn,
        triton_meta,
        configs,
        save_cache_hook,
        mutated_arg_names: list,
        optimize_mem,
        heuristic_type,
        size_hints=None,
        inductor_meta=None,
        custom_kernel: bool = False,
        filename: str | None = None,
        reset_to_zero_arg_names: list | None = None,
        autotune_cache_info: dict | None = None,
    ):
        super().__init__()

        if len(configs) == 0:
            raise AssertionError("a kernel with no configuration has nothing to run")
        for cfg in configs:
            validate_triton_config(cfg)

        self.fn = fn
        # The device is asked for here and named by index, because everything
        # downstream -- where a binary is written, which target a kernel is
        # compiled for -- wants an index rather than a description of a device.
        self.device_props = triton_meta["device"]
        self.triton_meta = {
            **triton_meta,
            "device": self.device_props.index,
            "device_type": self.device_props.type,
        }
        self.inductor_meta = {} if inductor_meta is None else inductor_meta
        # What the coordinate-descent tuner needs to know about the device it
        # is tuning for, put where it looks.
        self.inductor_meta["warp_size"] = self.device_props.warp_size
        self.inductor_meta["max_threads_per_block"] = (
            self.device_props.max_threads_per_block
        )
        self.deterministic_mode = self.inductor_meta.get("deterministic", False)

        self.save_cache_hook = save_cache_hook
        # Arguments this kernel writes rather than reads.  A measurement runs
        # the kernel many times, so an argument it writes has to start from a
        # known value or the second run is not the first run repeated.
        self.mutated_arg_names = mutated_arg_names
        self.reset_to_zero_arg_names = (
            reset_to_zero_arg_names
            if reset_to_zero_arg_names is not None
            else mutated_arg_names
        )
        self.optimize_mem = optimize_mem
        self.configs = list(configs)
        self.heuristic_type = heuristic_type
        self.custom_kernel = custom_kernel
        self.autotune_cache_info = autotune_cache_info
        self.lock = threading.Lock()
        self.size_hints = size_hints
        self.is_mix_order_reduction = self.inductor_meta.get("RSPLIT_SIZE") is not None
        self.coordesc_tuner = CoordescTuner(
            is_mm=inductor_meta.get("is_mm", False) if inductor_meta else False,
            is_mix_order_reduction=self.is_mix_order_reduction,
            size_hints=size_hints,
            inductor_meta=inductor_meta,
        )
        self.filename = filename
        self.kernel_hash: str | None = None
        if filename is not None:
            self.kernel_hash = os.path.splitext(os.path.basename(filename))[0]
        self.precompile_time_taken_ns = 0
        self.autotune_time_taken_ns = 0
        self.triton_interpret = os.environ.get("TRITON_INTERPRET", "0") == "1"
        self.is_backward = False
        self.launchers: list = []
        self.compile_results: list = []
        self._cached_launcher = None
        # The profiler range covering the launch in progress, if one is open.
        self._profiler_ctx = None
        # Why each configuration could not be measured, keyed by the
        # configuration.  Kept apart from a time so that "could not run" is
        # never mistaken for "ran slowly".
        self.benchmark_failure_reasons: dict = {}
        self._debug_call = None
        self.compile_id = None
        # Whether a launch works out its grid in this language or in the one the
        # launcher is written in.  Settled here because it is a property of the
        # kernel rather than of a launch, and read on every launch.
        self.grid_mode: Literal["python", "cpp"] = "python"
        # What gets to answer for this kernel before it acts, asked in order.
        self._plugins = get_caching_autotuner_plugins(self)

    @staticmethod
    def _close_compiled_kernel(kernel) -> None:
        """Let go of a compiled kernel's device resources, if it can be let go.

        A kernel holds a loaded binary and a module, and holding either across
        a measurement that has finished is what makes a long series of
        measurements run out of memory.  Not every version of the runtime
        offers a way to release one, so the older route is kept as a fallback;
        it is safe to reach more than once because the module is already gone
        by then.
        """

        if kernel is None:
            return
        close = getattr(kernel, "close", None)
        if close is not None:
            close()
            return
        module = getattr(kernel, "module", None)
        if module is not None:
            delete = getattr(kernel, "__del__", None)
            if delete is not None:
                delete()

    def release_benchmark_artifacts(self) -> None:
        """Let go of everything a measurement was holding.

        A measurement compiles kernels and loads them, and the point of
        measuring is to throw most of them away.  Doing that explicitly is
        what keeps a sweep over many configurations from growing without
        bound, and it is also what lets the same kernel be measured again
        afterwards.
        """

        for launcher in self.launchers:
            kernel = getattr(launcher, "__self__", None)
            self._close_compiled_kernel(kernel)

        for result in self.compile_results:
            self._close_compiled_kernel(getattr(result, "kernel", None))

        self.launchers = []
        self.compile_results = []
        self.benchmark_failure_reasons.clear()
        self._cached_launcher = None
        self._debug_call = None

    def is_statically_launchable(self):
        """Whether every compiled form can be started without going through
        the runtime's own launcher machinery.

        Such a kernel can be written down whole and started again later, which
        is what makes it worth keeping across runs -- so this is asked before
        deciding whether a measurement is worth keeping at all.
        """

        if not self.compile_results:
            return False
        return all(
            isinstance(x, StaticTritonCompileResult) for x in self.compile_results
        )

    def _pre_launch(self, launcher, *args: Any, stream: Any, **kwargs: Any) -> None:
        """What is settled before a launch happens.

        A launch is a thing that can be timed, and a measurement is worth being
        able to point at afterwards.  What is held open here is what would be
        said about this launch, and nothing that could stop the launch: a launch
        that raised still has to have whatever was held open settled, which is
        what the other half is for.
        """

        self._profiler_ctx = None


    def run(
        self,
        *args: Any,
        stream: Any,
        benchmark_run: bool = False,
        **kwargs: Any,
    ):
        """Launch the kernel, compiling and measuring first if that has not been done.

        In the steady state this is a launch and nothing else: once there is one
        launcher there is no question left to answer, and re-asking it on every
        launch is a cost on the path where a launch is timed.  What is checked
        every time is only what can change at run time, so that turning any of
        it on later falls back to the slower path rather than being ignored.
        """

        fast = self._cached_launcher
        if (
            fast is not None
            and not benchmark_run
            and not kwargs
            and not self.triton_interpret
        ):
            return fast(*args, stream=stream)

        if self.triton_interpret:
            args, grid = self._interpret_args_grid(args, self.configs[0])
            return self.fn[grid](
                *args,
                **kwargs,
                **self.configs[0].kwargs,
            )

        for plugin in self._plugins:
            if (
                result := plugin.pre_dispatch(self, *args, stream=stream, **kwargs)
            ) is not DEFER:
                return result

        if len(self.launchers) != 1:
            if len(self.launchers) == 0:
                start_time = time.time_ns()
                self.precompile()
                self.precompile_time_taken_ns = time.time_ns() - start_time
            if len(self.launchers) > 1:
                for plugin in self._plugins:
                    if (
                        result := plugin.pre_autotune(
                            self, *args, stream=stream, **kwargs
                        )
                    ) is not DEFER:
                        return result
                # Asked again, because a plugin is free to have left exactly one.
                if len(self.launchers) > 1:
                    self.autotune_to_one_config(*args, **kwargs)

        if not getattr(
            self.launchers[0].config, "found_by_coordesc", False
        ) and self.inductor_meta.get("coordinate_descent_tuning", False):
            self.launchers = [
                self.coordinate_descent_tuning(self.launchers[0], *args, **kwargs)
            ]

        (launcher,) = self.launchers
        # Recorded here as well as where the launcher was chosen, because for a
        # kernel with one configuration there is no choosing to do: this is the
        # only place that knows which one it was.
        TritonBundler.put_winner(launcher.cache_hash)

        try:
            self._pre_launch(launcher, *args, stream=stream, **kwargs)
            try:
                result = launcher(*args, **kwargs, stream=stream)
            except Exception as e:
                if isinstance(e, TypeError):
                    self._check_launcher_call_args(launcher, args)
                raise
        finally:
            self._post_launch()

        # The launcher is remembered only where nothing about the launch can
        # change what should happen: a measurement, arguments this path has not
        # been asked about, anything being recorded about the arguments, and
        # more than one launcher all mean the question is still open.
        if (
            self._cached_launcher is None
            and not benchmark_run
            and not self.triton_interpret
            and len(self.launchers) == 1
        ):
            self._cached_launcher = launcher
        return result

    @property
    def _should_coordesc_tune(self) -> bool:
        """Whether this kernel's configuration may be tuned one knob at a time.

        Not for a kernel whose configuration was chosen rather than searched
        for: a template's configuration says what the template's text asks for,
        and a kernel the user wrote is theirs to tune.  And not where the
        answer has to be a particular number, because the knobs here are block
        sizes and warp counts, and both change the order a sum is taken in.
        """

        if self.heuristic_type in (
            HeuristicType.TEMPLATE,
            HeuristicType.USER_AUTOTUNE,
            HeuristicType.FIXED,
        ):
            return False
        if (
            self.deterministic_mode or "strict_reduction_rblock" in self.inductor_meta
        ) and self.heuristic_type in (
            HeuristicType.REDUCTION,
            HeuristicType.PERSISTENT_REDUCTION,
            HeuristicType.SPLIT_SCAN,
        ):
            return False
        return True

    def coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """Tune the configuration one knob at a time, starting from this one.

        Which knob is tried first depends on where this starts, and where it
        starts depends on whether everything was measured first: with the whole
        set measured there is a measured best to start from, and without one
        there is only the configuration that was chosen to begin with.  Both are
        the same descent from a different place, which is why the difference is
        only where it starts.
        """

        if not self._should_coordesc_tune:
            return launcher

        from .runtime_utils import timed_block

        with timed_block(
            "CachingAutotuner.coordinate_descent_tuning", log_pt2_compile_event=False
        ):
            return self._coordinate_descent_tuning(launcher, *args, **kwargs)

    def _coordinate_descent_tuning(self, launcher, *args, **kwargs):
        """The descent itself: change one thing, measure, keep it only if it won.

        A configuration is compiled and measured here rather than reused, because
        a configuration that has never been run on these arguments has not been
        measured -- and the descent is about what happens on these arguments.
        The launchers are kept by configuration so that a configuration the
        descent arrives at twice is not compiled twice.
        """

        config2launcher = {launcher.config: launcher}

        self._ensure_kernel_loaded()

        def benchmark_one_config(config):
            with self.lock:
                launcher = self._precompile_config(config).make_launcher()
            config2launcher[config] = launcher

            out = self.bench(launcher, *args, **kwargs)
            counters["inductor"]["coordesc_tuning_bench"] += 1
            log.debug(
                "COORDESC: %s: %f, nreg %d, nspill %d, #shared-mem %d",
                launcher.config,
                out,
                launcher.n_regs,
                launcher.n_spills,
                launcher.shared,
            )
            return out

        if (
            self.heuristic_type == HeuristicType.PERSISTENT_REDUCTION
            and "R0_BLOCK" in launcher.config.kwargs
        ):
            raise AssertionError(
                "the tuner here relies on a persistent reduction's configuration "
                "having no reducing block of its own"
            )
        start_time = time.time_ns()
        best_config = self.coordesc_tuner.autotune(
            benchmark_one_config, launcher.config, None
        )
        coordesc_time_taken_ns = time.time_ns() - start_time
        best_config.found_by_coordesc = True

        if self.save_cache_hook:
            self.save_cache_hook(
                best_config,
                self.autotune_time_taken_ns + coordesc_time_taken_ns,
                found_by_coordesc=True,
            )

        if best_config not in config2launcher:
            # An answer read back from a cache names a configuration whose
            # launcher may never have been built in this process: what was kept
            # is the answer, and the thing that runs was not kept with it.
            config2launcher[best_config] = self._precompile_config(
                best_config
            ).make_launcher()

        winner = config2launcher[best_config]
        TritonBundler.put_winner(winner.cache_hash)
        return winner

    @staticmethod
    def _close_static_launcher(launcher: Any) -> None:
        """Let go of a binary that was loaded to be measured, if it was.

        A binary loaded to be measured was loaded for that measurement.  Where a
        launcher does not name a binary there is nothing to let go of, which is
        most launchers and is not worth a check any further than this.
        """

        if not getattr(launcher, "_is_static", False):
            return
        runner = getattr(launcher, "__globals__", {}).get("runner")
        kernel = getattr(runner, "__self__", None)
        close = getattr(kernel, "close", None)
        if close is not None:
            close()

    def _release_static_launchers_except(self, keep_launcher: Any) -> None:
        """Let go of every binary except the one that was chosen.

        What is kept is what can still be launched.  The chosen binary is found
        by what it was built from rather than by which object it is, because
        what is being asked is which of the compiled forms is the winner, and
        two launchers can be the same form.
        """

        for launcher in self.launchers:
            if launcher is not keep_launcher:
                self._close_static_launcher(launcher)
        if not getattr(keep_launcher, "_is_static", False):
            return
        keep_hash = getattr(keep_launcher, "cache_hash", None)
        if keep_hash is None:
            return
        keep_results = [
            result
            for result in self.compile_results
            if isinstance(result, StaticTritonCompileResult)
            and triton_hash_to_path_key(result.kernel.hash) == keep_hash
        ]
        if len(keep_results) == 1:
            for result in self.compile_results:
                if result is not keep_results[0] and isinstance(
                    result, StaticTritonCompileResult
                ):
                    close = getattr(result.kernel, "close", None)
                    if close is not None:
                        close()
            self.compile_results = keep_results

    def _log_autotune_inputs(self, args, kwargs) -> None:
        """What was measured, in enough detail to reproduce the measurement.

        A measurement whose inputs are not written down is a number that cannot
        be questioned afterwards, and a kernel measured on one shape and used on
        another is the usual way for that to matter.  What is logged is the
        shape, the type and the layout of every argument, and the number it
        carried, which is everything the measurement depended on.
        """

        kernel_name = self.inductor_meta.get("kernel_name", self.fn.__name__)
        signature = self.triton_meta.get("signature", {})
        arg_names = list(signature.keys())

        autotuning_inputs_log.debug("=" * 60)
        autotuning_inputs_log.debug("Autotuning inputs for kernel: %s", kernel_name)
        autotuning_inputs_log.debug("=" * 60)
        autotuning_inputs_log.debug("  Heuristic type: %s", self.heuristic_type)
        autotuning_inputs_log.debug("  Size hints: %s", self.size_hints)
        autotuning_inputs_log.debug(
            "  Num configs to benchmark: %d", len(self.launchers)
        )
        autotuning_inputs_log.debug(
            "  Device: %s (index=%s)", self.device_props.type, self.device_props.index
        )
        autotuning_inputs_log.debug("-" * 60)
        autotuning_inputs_log.debug("Arguments:")

        for i, arg in enumerate(args):
            arg_name = arg_names[i] if i < len(arg_names) else f"arg_{i}"
            arg_signature = signature.get(arg_name, "unknown")
            if isinstance(arg, tp.Tensor):
                autotuning_inputs_log.debug(
                    "  [%d] %s (%s): Tensor(shape=%s, dtype=%s, device=%s, stride=%s, contiguous=%s)",
                    i,
                    arg_name,
                    arg_signature,
                    tuple(arg.shape),
                    arg.dtype,
                    arg.device,
                    arg.stride(),
                    arg.is_contiguous(),
                )
            elif isinstance(arg, (int, float, bool)):
                autotuning_inputs_log.debug(
                    "  [%d] %s (%s): %s (type=%s)",
                    i,
                    arg_name,
                    arg_signature,
                    arg,
                    type(arg).__name__,
                )
            else:
                autotuning_inputs_log.debug(
                    "  [%d] %s (%s): %s (type=%s)",
                    i,
                    arg_name,
                    arg_signature,
                    repr(arg)[:100],
                    type(arg).__name__,
                )

        if kwargs:
            autotuning_inputs_log.debug("-" * 60)
            autotuning_inputs_log.debug("Keyword arguments:")
            for k, v in kwargs.items():
                if isinstance(v, tp.Tensor):
                    autotuning_inputs_log.debug(
                        "  %s: Tensor(shape=%s, dtype=%s, device=%s)",
                        k,
                        tuple(v.shape),
                        v.dtype,
                        v.device,
                    )
                else:
                    autotuning_inputs_log.debug(
                        "  %s: %s (type=%s)", k, v, type(v).__name__
                    )

        autotuning_inputs_log.debug("=" * 60)

    def benchmark_all_configs(self, *args: Any, **kwargs: Any):
        """How long every launcher takes, each on the same inputs.

        Each measurement is one run of a whole batch of launches, so what is
        compared is the same work each time -- which is why the arguments are
        put back to what they were between them, and why a launcher that is no
        longer in the running is let go of as soon as it is not: a set of
        measurements that all of them are kept alive for is a set of measurements
        that all of them pay for.
        """

        from .runtime_utils import timed_block

        with timed_block(
            "CachingAutotuner.benchmark_all_configs", log_pt2_compile_event=True
        ):
            timings = {}
            best_launcher = None
            best_timing = float("inf")
            for launcher in self.launchers:
                timing = self.bench(launcher, *args, **kwargs)
                timings[launcher] = timing
                # A launcher that has been beaten is let go of now rather than
                # after the whole set, so that an exhaustive measurement keeps
                # only the winner and the one still being measured loaded.
                if best_launcher is None or timing < best_timing:
                    if best_launcher is not None:
                        self._close_static_launcher(best_launcher)
                    best_launcher = launcher
                    best_timing = timing
                else:
                    self._close_static_launcher(launcher)

            for k, v in timings.items():
                self.coordesc_tuner.cache_benchmark_result(k.config, v)

            if log.isEnabledFor(logging.DEBUG):
                log.debug("Benchmark all input configs for %s, get:", self.fn.__name__)
                for k, v in timings.items():
                    log.debug(
                        "%s: %f, nreg %d, nspill %d, #shared-mem %s",
                        k.config,
                        v,
                        k.n_regs,
                        k.n_spills,
                        k.shared,
                    )

            if metrics.is_metric_table_enabled("kernel_autotune"):
                self._ensure_kernel_loaded()

                kernel_path = self.fn.fn.__code__.co_filename
                kernel_name = self.fn.__name__

                for k, v in timings.items():
                    metrics.log_kernel_autotune_result(
                        kernel_path, kernel_name, k.config, v
                    )

            self.reset_to_zero_args(*args, **kwargs)
            return timings

    def autotune_to_one_config(self, *args: Any, **kwargs: Any):
        """Measure every launcher and keep the one that measured best.

        A launcher that could not be measured is not a slow launcher, and the
        two are kept apart throughout: the reason is recorded per launcher, and
        a launcher with no reason is one that really was slower than the others.
        """

        if autotuning_inputs_log.isEnabledFor(logging.DEBUG):
            self._log_autotune_inputs(args, kwargs)

        start_time = time.time_ns()
        timings = self.benchmark_all_configs(*args, **kwargs)
        benchmark_time_taken_ns = time.time_ns() - start_time

        # Which configurations could not be measured, and why, is worth saying
        # out loud: a set of configurations that is mostly unmeasurable is a
        # different problem from one where every configuration lost.
        failed_launchers = [
            launcher for launcher, timing in timings.items() if timing == float("inf")
        ]
        if failed_launchers:
            valid_timings = [(k, v) for k, v in timings.items() if v != float("inf")]
            if valid_timings:
                best_launcher, best_time = min(valid_timings, key=lambda x: x[1])

                # Count failures by reason
                spill_count = sum(
                    1
                    for launcher in failed_launchers
                    if self.benchmark_failure_reasons.get(launcher)
                    == BenchmarkFailureReason.REGISTER_SPILLING
                )
                invalid_config_count = sum(
                    1
                    for launcher in failed_launchers
                    if self.benchmark_failure_reasons.get(launcher)
                    == BenchmarkFailureReason.INVALID_CONFIG
                )

                reason_parts = []
                if spill_count > 0:
                    reason_parts.append(f"{spill_count} register spilling")
                if invalid_config_count > 0:
                    reason_parts.append(f"{invalid_config_count} invalid config")
                reason_str = ", ".join(reason_parts) if reason_parts else "unknown"

                log.info(
                    "Skipped %d/%d configs for %s (%s). Selected: %s (%.4f ms)",
                    len(failed_launchers),
                    len(timings),
                    self.fn.__name__,
                    reason_str,
                    best_launcher.config,
                    best_time,
                )

        best_launcher = min(timings, key=timings.get)
        self._release_static_launchers_except(best_launcher)
        self.launchers = [best_launcher]
        self._prune_compile_results_to_launcher(best_launcher)
        self.autotune_time_taken_ns = (
            self.precompile_time_taken_ns + benchmark_time_taken_ns
        )

        # log the best config
        launcher = self.launchers[0]
        log.debug(
            "Best config for %s: %s: %f, nreg %d, nspill %d, #shared-mem %s",
            self.fn.__name__,
            launcher.config,
            timings[launcher],
            launcher.n_regs,
            launcher.n_spills,
            launcher.shared,
        )

        TritonBundler.put_winner(launcher.cache_hash)

        if self.save_cache_hook:
            self.save_cache_hook(
                launcher.config,
                self.autotune_time_taken_ns,
                found_by_coordesc=self.inductor_meta.get(
                    "coordinate_descent_tuning", False
                ),
                triton_cache_hash=launcher.cache_hash,
            )

    def copy_args_to_cpu_if_needed(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Keep a copy of what a measurement is about to overwrite, off the device.

        A measurement runs a kernel many times, and a kernel that writes an
        argument must be handed that argument as it was, or the second run is
        not the first run repeated and what is measured is not the kernel.  The
        usual answer is to hand it a copy, which costs device memory that the
        program may not have -- so a copy is only made while there is room, and
        once there is not, the value is kept somewhere that has room and put
        back after each run instead.
        """

        if not self.optimize_mem:
            return {}

        copies = {}
        try:
            if tp.accelerator.current_accelerator() is None:
                # Nothing is running, so there is no device to be sparing.
                return {}
            budget = (
                tp.accelerator.max_memory_allocated()
                - tp.accelerator.memory_allocated()
            )
        except RuntimeError:
            # A custom allocator need not answer this, and not answering means
            # there is no budget to be sparing within.
            return {}

        def maybe_copy(name, arg):
            if name in self.mutated_arg_names and arg.device.type in (
                "cuda",
                "xpu",
            ):
                nonlocal budget
                if not isinstance(arg, tp.Tensor):
                    raise AssertionError(
                        f"Expected a tensor for mutated arg, got {type(arg)}"
                    )
                required_storage_length = compute_required_storage_length(
                    arg.size(),
                    arg.stride(),
                    0,
                )
                size = required_storage_length * arg.element_size()
                if size > budget:
                    cpu_arg = tp.empty_strided(
                        (required_storage_length,),
                        (1,),
                        dtype=arg.dtype,
                        device="cpu",
                        pin_memory=True,
                    )
                    cpu_arg.copy_(
                        arg.as_strided((required_storage_length,), (1,)),
                        non_blocking=True,
                    )
                    copies[name] = (arg, cpu_arg)
                else:
                    budget -= size

        for name, arg in zip(self.fn.arg_names, args):
            maybe_copy(name, arg)

        for name, arg in kwargs.items():
            maybe_copy(name, arg)

        return copies

    def restore_args_from_cpu(self, cpu_copies: dict[str, Any]) -> None:
        for pair in cpu_copies.values():
            arg, cpu_arg = pair
            required_storage_length = compute_required_storage_length(
                arg.size(),
                arg.stride(),
                0,
            )
            arg.as_strided((required_storage_length,), (1,)).copy_(
                cpu_arg, non_blocking=True
            )

    def reset_to_zero_args(self, *args: Any, **kwargs: Any) -> None:
        """Put back to what they were the arguments this kernel writes.

        A kernel that writes an argument is handed one whose earlier contents it
        is meant to add to, and a measurement runs it again and again: without
        this, what accumulates is the kernel's output over every run so far
        rather than over one.
        """

        if not self.reset_to_zero_arg_names:
            return
        for i, arg in enumerate(args):
            if self.fn.arg_names[i] in self.reset_to_zero_arg_names:
                if not isinstance(arg, tp.Tensor):
                    raise AssertionError(
                        "only arguments that are tensors can be reset to zero"
                    )
                arg.zero_()

        for name, arg in kwargs.items():
            if name in self.reset_to_zero_arg_names:
                if not isinstance(arg, tp.Tensor):
                    raise AssertionError(
                        "only arguments that are tensors can be reset to zero"
                    )
                arg.zero_()

    def maybe_clone_args(
        self, exclude: Container[str], *args: Any, **kwargs: Any
    ) -> tuple[list[Any], dict[str, Any]]:
        """Hand the kernel copies of the arguments it writes, and the rest as they are.

        Only what the kernel writes has to be a copy: what it only reads is not
        changed by running it, and copying it would cost device memory for
        nothing.  What has been kept elsewhere is passed in ``exclude``, because
        that value is being restored after each run and copying it would defeat
        the point of having kept it.
        """

        from tensorplay._higher_order_ops.auto_functionalize import clone_preserve_strides

        def prepare_arg(name, arg):
            if name in self.mutated_arg_names and name not in exclude:
                if not isinstance(arg, tp.Tensor):
                    raise AssertionError(
                        f"Expected a tensor for mutated arg '{name}', got {type(arg)}"
                    )
                return clone_preserve_strides(arg)
            else:
                return arg

        cloned_args = [
            prepare_arg(name, arg)
            for name, arg in itertools.zip_longest(self.fn.arg_names[: len(args)], args)
        ]
        cloned_kwargs = {name: prepare_arg(name, arg) for name, arg in kwargs.items()}
        return cloned_args, cloned_kwargs

    def clone_args(self, *args: Any, **kwargs: Any) -> tuple[list[Any], dict[str, Any]]:
        return self.maybe_clone_args(OrderedSet(), *args, **kwargs)

    def _check_launcher_call_args(self, launcher: Any, args: tuple[Any, ...]) -> None:
        """Say what was wrong with a call, when the error does not.

        A launcher that rejects its arguments raises about a type it was handed,
        which says nothing about the call that produced them.  What is worth
        saying is how many the launcher wanted, and that the stream is not one
        of them: a stream passed in place is the mistake that produces this.
        """

        expected = getattr(launcher, "_expected_positional_count", None)
        if expected is None:
            return

        if len(args) > expected:
            kernel_name = self.inductor_meta.get("kernel_name", "triton kernel")
            raise TypeError(
                f"{kernel_name}: too many positional arguments - "
                f"expected {expected}, got {len(args)}. "
                "'stream' must be passed as a keyword argument."
            ) from None

    def bench(self, launcher, *args: Any, with_profiler: bool = False, **kwargs: Any) -> float:
        """How long one launch of this configuration takes.

        A configuration whose registers spilled is one that used more registers
        than the device had room for and wrote them out to memory, which is
        slower than not using it at all -- so it is reported as a time no
        configuration can reach rather than as a failure, which is what keeps
        it from being chosen.  A kernel written by hand is excepted: nothing here
        knows what it does, and a complicated kernel can genuinely be faster
        with the registers spilled.
        """

        if (
            not self.custom_kernel
            and launcher.n_spills is not None
            and launcher.n_spills
            > self.inductor_meta.get("spill_threshold", 32 if tp.version.hip else 16)
        ):
            log.debug(
                "Skip config %s because of register spilling: %d",
                launcher.config,
                launcher.n_spills,
            )
            self.benchmark_failure_reasons[launcher] = (
                BenchmarkFailureReason.REGISTER_SPILLING
            )
            return float("inf")

        device_interface = self.get_device_interface()

        cpu_copies = self.copy_args_to_cpu_if_needed(*args, **kwargs)

        def kernel_call():
            # The stream is asked for as the kernel is about to run rather than
            # when this was written, so that capturing a launch onto another
            # stream launches it onto that stream.
            stream = device_interface.get_raw_stream(device_interface.current_device())
            cloned_args, cloned_kwargs = self.maybe_clone_args(
                cpu_copies, *args, **kwargs
            )
            kernel_name = self.inductor_meta.get("kernel_name", "triton kernel")
            # Each configuration is measured from the same starting values, or
            # the first one measured would be measuring more work than the rest.
            self.reset_to_zero_args(*args, **kwargs)
            try:
                launcher(
                    *cloned_args,
                    **cloned_kwargs,
                    stream=stream,
                )
            except Exception as e:
                if isinstance(e, TypeError):
                    self._check_launcher_call_args(launcher, cloned_args)
                log.error(
                    "Failed during launch %s with config: %s (num_warps=%s, num_stages=%s, kwargs=%s)",
                    kernel_name,
                    launcher.config,
                    launcher.config.num_warps,
                    launcher.config.num_stages,
                    launcher.config.kwargs,
                )
                raise
            self.restore_args_from_cpu(cpu_copies)

        # A profile is only taken when nothing else is already profiling, since
        # two profilers cannot both be the one that is running.
        if with_profiler:
            from ..utils import do_bench_using_profiling

            return do_bench_using_profiling(kernel_call, warmup=10, rep=40)

        benchmark_kwargs = (
            {}
            if self.device_props.type == "cpu"
            else {"rep": 40, "is_vetted_benchmarking": True}
        )
        result = benchmarker.benchmark(
            fn=kernel_call,
            device=self.device_props.type,
            **benchmark_kwargs,
        )
        # The timing reports an unreachable time in exactly one case: a
        # configuration the device would not accept.  Everything else is raised,
        # so a time that no configuration can reach here is that, and not
        # something that ran slowly.
        if result == float("inf"):
            self.benchmark_failure_reasons[launcher] = (
                BenchmarkFailureReason.INVALID_CONFIG
            )
        return result

    @functools.cached_property
    def _could_rblock_scale(self) -> bool:
        """Whether a block worth halving is worth looking for here.

        Where the answer must be a particular number, halving it changes the
        order the sum is taken in, so it is not offered at all.
        """

        if "strict_reduction_rblock" in self.inductor_meta:
            return False
        return _could_dynamic_scale_rblock(
            size_hints=self.size_hints,
            heuristic_type=self.heuristic_type,
            device_prop=self.device_props,
            inductor_meta=self.inductor_meta,
        )

    @functools.cached_property
    def _combo_has_reduction_subkernel(self) -> bool:
        """Whether a kernel of several has one of them reducing.

        Such a kernel arrives with no extents of its own, which is what the
        extents of a kernel usually are for; the per-part figures are in the
        record of which parts there are.
        """

        return _combo_has_reduction_subkernel(self.inductor_meta)

    def _iter_rblock_scale_candidates(self):
        """Yield each configuration that halves a reducing block.

        Whether any of these is worth having is settled by the caller: what is
        settled here is which halvings would actually be an improvement, which
        is a question about how many blocks fit on the device at once and how
        many the kernel would like to have there.
        """

        device_prop = self.device_props
        if not device_prop.regs_per_multiprocessor:
            raise AssertionError("the device does not say how many registers it has")
        if not device_prop.max_threads_per_multi_processor:
            raise AssertionError("the device does not say how many threads it runs")
        if not device_prop.multi_processor_count:
            raise AssertionError("the device does not say how many processors it has")
        if device_prop.warp_size is None:
            raise AssertionError("the device does not say how wide a warp is")
        seen_config_hashes: OrderedSet | None = None
        warp_size = device_prop.warp_size
        # A kernel of several is told how big each part is rather than being
        # handed extents, so it has none to count blocks from; how many blocks
        # it wants is the sum over its parts.  Which parts reduce is shared
        # between the two kinds of kernel.
        combo_meta = (
            self.inductor_meta.get("combo_grid_meta")
            if self.size_hints is None
            else None
        )
        for result in self.compile_results:
            triton_config = result.config
            compiled_binary = result.kernel
            if combo_meta is not None:
                # A kernel of several runs its parts one after another over a
                # flattened range, so how many blocks it wants is the sum of
                # what its parts want.  A part whose extent is not a number --
                # a shape only known when the program runs -- contributes none.
                total_block = 0
                for i in range(combo_meta["num_kernels"]):
                    xnumel = combo_meta.get(f"xnumel_{i}")
                    if isinstance(xnumel, int):
                        xblock = triton_config.kwargs.get(f"XBLOCK_{i}", 1)
                        total_block += (xnumel + xblock - 1) // xblock
            else:
                if len(self.size_hints) < 2:
                    raise AssertionError(
                        f"Expected at least 2 size_hints, got {len(self.size_hints)}"
                    )
                xblock = triton_config.kwargs.get("XBLOCK", 1)
                total_block = (self.size_hints["x"] + xblock - 1) // xblock
            # Which of the blocks are the ones a reduction is done in.  A
            # kernel of several has one per reducing part, distinguished by a
            # suffix; a part that does not reduce has none, and drops out here
            # by having nothing to name.
            reduction_kwargs = [
                kwarg for kwarg in triton_config.kwargs if kwarg.startswith("R")
            ]
            if not reduction_kwargs:
                continue
            rblocks = [triton_config.kwargs[kwarg] for kwarg in reduction_kwargs]
            nreg = getattr(compiled_binary, "n_regs", None)
            if nreg is None:
                continue
            # A block already this small is not a block that was too large.
            if conditional_product(*rblocks) <= 64:
                continue
            # A processor has a fixed number of registers and a fixed number of
            # threads it can run, so there is a number of registers per thread
            # past which not all of the threads fit at once.  A thread using
            # more than that is using registers that another thread would have
            # used, which is what halving the block gives back.
            if (
                nreg
                <= device_prop.regs_per_multiprocessor
                // device_prop.max_threads_per_multi_processor
            ):
                continue
            nreg_per_warp = nreg * warp_size
            nreg_per_block = nreg_per_warp * triton_config.num_warps
            # How many blocks of this size fit on one processor at once: the
            # registers a block needs, against the registers a processor has.
            # Reaching one means registers rather than blocks are what limits
            # how many threads run together, which is the case worth acting on
            # and which a looser bound would not have revealed.
            max_blocks_per_sm = max(
                device_prop.regs_per_multiprocessor // nreg_per_block, 1
            )
            if total_block <= max_blocks_per_sm * device_prop.multi_processor_count:
                # There is already a processor for every block the kernel wants.
                continue
            # Halve the largest reducing block, which is the one contributing
            # the most registers to a thread.
            largest_rkwarg: str = max(
                reduction_kwargs, key=triton_config.kwargs.__getitem__
            )
            new_rblock = triton_config.kwargs[largest_rkwarg] // 2
            min_rblock = self.inductor_meta.get("min_rblock")
            if (
                min_rblock is not None
                and largest_rkwarg.startswith("R0_BLOCK")
                and new_rblock < min_rblock
            ):
                continue
            new_config = copy.deepcopy(triton_config)
            new_config.kwargs[largest_rkwarg] = new_rblock

            if seen_config_hashes is None:
                seen_config_hashes = OrderedSet(
                    [triton_config_to_hashable(x.config) for x in self.compile_results]
                )
            new_config_hash = triton_config_to_hashable(new_config)
            if new_config_hash in seen_config_hashes:
                continue
            seen_config_hashes.add(new_config_hash)
            log.debug(
                "halving %s from %s gives %s",
                largest_rkwarg,
                triton_config,
                new_config,
            )
            self._ensure_kernel_loaded()
            yield new_config

            # A kernel of several reports the registers of its largest part, so
            # halving only the largest part brings the number down no further
            # when the parts are of similar size.  Halving all of them is the
            # same candidate for such a kernel, and a different one.
            if combo_meta is not None and len(reduction_kwargs) > 1:
                all_halved = copy.deepcopy(triton_config)
                too_small = False
                for kwarg in reduction_kwargs:
                    halved = triton_config.kwargs[kwarg] // 2
                    # A block of one halved is zero, and a kernel cannot be
                    # given a block of zero, so this candidate is not offered
                    # rather than offered as something that cannot be compiled.
                    if halved < 1 or (
                        min_rblock is not None
                        and kwarg.startswith("R0_BLOCK")
                        and halved < min_rblock
                    ):
                        too_small = True
                        break
                    all_halved.kwargs[kwarg] = halved
                if not too_small:
                    all_hash = triton_config_to_hashable(all_halved)
                    if all_hash not in seen_config_hashes:
                        seen_config_hashes.add(all_hash)
                        self._ensure_kernel_loaded()
                        yield all_halved

    def _dynamic_scale_rblock(self):
        # A configuration that was read back from a cache was already chosen
        # with all of this in mind; scaling again would be measuring a question
        # that has an answer.
        if (
            self.autotune_cache_info
            and self.autotune_cache_info.get("autotune_cache_state") == "hit"
        ):
            return
        if not self._could_rblock_scale:
            return
        for new_config in self._iter_rblock_scale_candidates():
            self.compile_results.append(self._precompile_config(new_config))
        self._make_launchers()

    def compile_by_disabling_pipelining(self, config):
        """The same tile, compiled so that it holds one step at a time.

        A tile too large for the memory this device has is a tile that cannot
        be measured here, and the parts of it that are the memory are the
        steps it holds at once: fewer steps, less memory, the same arithmetic.
        """

        self._ensure_kernel_loaded()
        cfg = copy.deepcopy(config)
        cfg.num_stages = 1
        if "NUM_STAGES" in cfg.kwargs:
            cfg.kwargs["NUM_STAGES"] = 1
        result = self._precompile_config(cfg)
        self.compile_results = [result]
        return result.make_launcher()



    def precompile(
        self,
        warm_cache_only: bool = False,
        reload_kernel: Callable[[], CachingAutotuner] | None = None,
        static_triton_bundle_key: str | None = None,
    ):
        """Compile every configuration and make a launcher of each.

        Nothing is measured here.  What this does is make a launch a launch
        rather than a compile, and it is a separate step from the launch so
        that the compiling can happen before anything is run and be paid for
        once, wherever it happens.
        """

        if warm_cache_only:
            self._precompile_worker()
            return
        with self.lock:
            # A kernel compiled in another process has to be compiled again in
            # this one before anything here can compile it, which is what
            # holding on to how to do that is for.
            if reload_kernel is not None:
                self._reload_kernel = reload_kernel
            # A plugin that answers here owns compiling and making launchers
            # from here on, including recording what it built, which is why the
            # ordinary flow below is not run at all.
            for plugin in self._plugins:
                if plugin.pre_compile(self) is not DEFER:
                    return
            self._precompile_worker()
            if static_triton_bundle_key is not None and self.is_statically_launchable():
                TritonBundler.put_static_autotuner(static_triton_bundle_key, self)
            self._make_launchers()
            self._dynamic_scale_rblock()

    def _precompile_worker(self):
        if self.compile_results:
            for result in self.compile_results:
                TritonBundler.put(
                    triton_hash_to_path_key(result.kernel.hash),
                    int(self.triton_meta.get("device", 0)),
                )
            return
        if self.launchers:
            raise AssertionError("there are launchers before anything was compiled")
        if not self.configs:
            raise NoTritonConfigsError("there is no configuration to compile")

        compile_results = []
        exc = None
        for c in self.configs:
            try:
                compile_results.append(self._precompile_config(c))
            except (OutOfResources, PTXASError, IntelGPUError) as e:
                exc = e
        if len(compile_results) == 0:
            raise NoTritonConfigsError(
                f"No valid triton configs. {type(exc).__name__}: {exc}"
            )
        self.compile_results = compile_results
        # The configurations are now what was compiled, and reading them again
        # would be reading a list that is no longer what it was.
        self.configs = None

    def _make_launcher(
        self, compile_result: Any
    ) -> tuple[Any, None] | tuple[None, Exception]:
        """Make the launcher for one compiled configuration.

        The caller is holding the device this is loaded onto, which is what
        makes the binary land on the right one.  A configuration that cannot be
        given a launcher is returned as the reason rather than raised, so that
        the other configurations are still tried.
        """

        try:
            return compile_result.make_launcher(), None
        except (
            OutOfResources,
            PTXASError,
            tp.cuda.OutOfMemoryError,
            IntelGPUError,
        ) as e:
            return None, e

    def _make_launchers(self):
        if len(self.launchers) == len(self.compile_results):
            return

        device_interface = self.get_device_interface()
        launchers = []
        exc = None
        try:
            load_device = _resolve_load_device(
                self.triton_meta["device"], self.device_props.type
            )
            # Each binary is loaded while the device it belongs to is current,
            # or a binary compiled for one device lands on whichever is current.
            with device_interface.device(load_device):
                for result in self.compile_results:
                    launcher, exc = self._make_launcher(result)
                    if launcher is not None:
                        launchers.append(launcher)
                if len(launchers) == 0:
                    result = self.compile_results[-1]
                    config = result.config
                    if (
                        isinstance(exc, (OutOfResources, tp.cuda.OutOfMemoryError))
                        and (
                            config.num_stages > 1
                            or config.kwargs.get("NUM_STAGES", 1) > 1
                        )
                        and self.inductor_meta.get("dynamic_disable_pipelining", True)
                    ):
                        self.launchers = [self.compile_by_disabling_pipelining(config)]
                        return
                    raise RuntimeError(
                        f"No valid triton configs. {type(exc).__name__}: {exc}"
                    )
            self.launchers = launchers
        finally:
            # The reason a configuration failed is read above and nowhere
            # else, and holding it holds its whole stack with it, which under
            # a collector that is switched off is a buffer per measured kernel
            # that is never released.  What was wanted was its kind and its
            # message, and both have been read.
            exc = None

    def _prune_compile_results_to_launcher(self, launcher: Any) -> None:
        """Keep only what the chosen launcher was made from.

        Everything that was compiled and not chosen is what the rest of the
        program would hold on to, and none of it can be launched, so what is
        left is the one thing that can be.
        """

        if not self.compile_results:
            return

        launcher_config = launcher.config
        for result in self.compile_results:
            if result.config is launcher_config:
                self.compile_results = [result]
                return

        launcher_config_hash = triton_config_to_hashable(launcher_config)
        for result in self.compile_results:
            if triton_config_to_hashable(result.config) == launcher_config_hash:
                self.compile_results = [result]
                return

        raise AssertionError(
            f"Autotuned launcher config does not match any compile result: {launcher_config}"
        )

    def _ensure_kernel_loaded(self) -> None:
        """Bring the kernel's own text back into this process if it went away.

        A kernel compiled in another process crosses into this one without the
        compiled form of its text, so that anything here which needs to compile
        it again has to ask for it to be brought back.  A kernel that never left
        has it, and there is nothing to do.
        """

        if self.fn.fn is None:
            if not hasattr(self, "_reload_kernel"):
                raise AssertionError("_reload_kernel attribute not set")
            if not callable(self._reload_kernel):
                raise AssertionError("_reload_kernel must be callable")
            self.fn = self._reload_kernel().fn

    def prepare_for_pickle(self) -> tuple[Any, ...]:
        """Let go of what cannot travel to another process, keeping it aside.

        A compiled form is a device binary; it does not go into a pickle, and
        the function it came from holds a great deal of state that does not
        either.  So both are set aside here and the old values handed back, for
        whoever is holding this to be restored into -- and what is left behind
        in their place still answers the questions that are asked of it, so a
        form that was compiled here and is being sent elsewhere can still say
        what it is.

        The launchers go too: they hold the loaded binary, and the process that
        receives this will load its own.
        """

        old_values = (
            self.fn.fn,
            self.fn.__globals__,
            self.fn.used_global_vals,
            self.fn.repr,
            self.launchers,
            getattr(self.fn, "_hash_lock", None),
            self.benchmark_failure_reasons,
        )
        self.fn.fn = None
        self.fn.__globals__ = None
        self.fn.used_global_vals = None
        self.fn.repr = _ConstRepr(self.fn.repr(self.fn))
        self.launchers = []
        self._cached_launcher = None
        self.benchmark_failure_reasons = {}
        self.fn._hash_lock = None
        return old_values

    def restore_after_unpickle(self, old_values: tuple[Any, ...] | None) -> None:
        """Put back what was set aside, in the process that has the originals.

        Where there are no originals -- because the forms were not compiled
        here to begin with -- the lock still has to be a usable one, because
        something will take it whether or not there is anything to protect.
        """

        self._cached_launcher = None
        if old_values:
            (
                self.fn.fn,
                self.fn.__globals__,
                self.fn.used_global_vals,
                self.fn.repr,
                self.launchers,
                self.fn._hash_lock,
                self.benchmark_failure_reasons,
            ) = old_values
        else:
            # even if we don't need/have specific values, we do need the
            # _hash_lock to be a valid RLock
            self.fn._hash_lock = threading.RLock()

    def prepare_for_caching(self) -> None:
        """Let go of the raw binary before this is written to a cache.

        A form that is started without the runtime's launcher holds its binary
        as bytes, and those bytes are large -- much larger than the entry that
        refers to them.  Whether they are kept is a trade: keeping them means a
        cold load does not have to recompile, at the cost of every cache entry
        carrying them.
        """

        # Only cubin_raw must be retained: __getstate__ already nulls cubin_path
        # on every serialize, so a cold-container load rehydrates the cubin from
        # cubin_raw rather than pointing at a missing file.
        if config.keep_static_cubin_raw:
            return
        for result in self.compile_results:
            if isinstance(result, StaticTritonCompileResult):
                # Don't save this in the inductor cache, as it is very large
                result.kernel.cubin_raw = None

    def __getstate__(self) -> dict[str, Any]:
        if self.launchers:
            raise AssertionError("pickle should not be called after make_launchers()")
        return {
            **self.__dict__,
            "lock": None,
            "_plugins": [],
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.lock = threading.Lock()
        self._plugins = get_caching_autotuner_plugins(self)

    def get_device_interface(self):
        # What a device is called here and what it is called by the module that
        # talks to it are not always the same, and what is wanted is the latter.
        from .benchmarking import get_interface_for_device

        return get_interface_for_device(self.device_props.type.replace("hip", "cuda"))

    def _interpret_args_grid(
        self, args: tuple[Any, ...], cfg: Any
    ) -> tuple[tuple[Any, ...], tuple[int, int, int]]:
        """The arguments a kernel is given, and the grid it is run over.

        A value the text names as constant is folded into the binary and is not
        handed over at all, so what to hand over is worked out from the
        signature and what it declares rather than from the arguments as they
        are given -- and a kernel written by hand may declare no
        configurations at all, which is why the declaration is asked rather than
        the configurations.
        """

        if triton_version_uses_attrs_dict():

            def filtered_signature() -> list[str]:
                new_signature: list[str] = []
                from triton.runtime.interpreter import InterpretedFunction

                for i, x in enumerate(self.triton_meta["signature"].keys()):
                    if isinstance(self.fn, InterpretedFunction):
                        if x not in cfg.kwargs:
                            new_signature.append(x)
                    elif i not in get_constexprs(self.fn):
                        new_signature.append(x)
                return new_signature

        else:

            def filtered_signature() -> list[str]:
                return list(self.triton_meta["signature"].keys())

        grid = GridExpr.from_meta(
            self.inductor_meta, cfg, mode=self.grid_mode
        ).eval_slow(
            dict(
                zip(
                    [
                        *filtered_signature(),
                        *self.inductor_meta.get("extra_launcher_args", ()),
                    ],
                    args,
                )
            )
        )
        if self.inductor_meta.get("extra_launcher_args"):
            args = args[: -len(self.inductor_meta["extra_launcher_args"])]
        return args, grid

    def recheck_autotune_cache(self, reload_kernel_from_src) -> None:
        """Look again for an answer, now that this kernel has been compiled.

        A kernel that was kept from an earlier run was kept before anything
        here was measured, and the answer that would let it skip measuring may
        have been written since.  So the lookup is done again against what is
        now compiled.

        An answer naming a configuration that is among the compiled ones means
        everything else can be dropped -- the other forms were only ever
        candidates.  An answer naming something that is not among them was
        produced after the fact, by tuning that moved away from the starting
        list, and so has to be compiled now.
        """

        if not self.is_statically_launchable():
            raise AssertionError("Expected statically launchable kernel")

        configs = [result.config for result in self.compile_results]

        (cached_configs, _, autotune_cache_info) = check_autotune_cache(
            configs,
            self.filename,
            self.inductor_meta,
            dynamic_scale_rblock_eligible=self._could_rblock_scale,
        )
        self.autotune_cache_info = autotune_cache_info
        # I.e. there was an autotune cache hit
        if len(cached_configs) == 1:
            best_config = cached_configs[0]
            found_by_coordesc = getattr(best_config, "found_by_coordesc", False)
            # Grab the best compiled config, if it's in the list of available ones
            best_config_hash = triton_config_to_hashable(best_config)

            for compile_result in self.compile_results:
                if triton_config_to_hashable(compile_result.config) == best_config_hash:
                    compile_result.config.found_by_coordesc = found_by_coordesc
                    self.compile_results = [compile_result]
                    return

            # The best config isn't in our compile results -- it was
            # found dynamically (coordesc tuning or _dynamic_scale_rblock)
            # after the static autotuner was saved. Compile it now.
            with timed_block("CachingAutotuner.slow_precompile_config"):
                if self.fn.fn is None:
                    self.fn = reload_kernel_from_src().fn
                self.compile_results = [self._precompile_config(best_config)]

    def set_compile_info(self, compile_id, is_backward: bool) -> None:
        """Note which build this is, and whether it is the backward one.

        The backward pass is timed separately from the forward one, so a
        measurement has to know which of the two it is measuring.
        """

        self.compile_id = compile_id
        self.is_backward = is_backward

    def _create_compile_meta(self, cfg) -> dict:
        """What this configuration is compiled with, and what it is not.

        The runtime is given a signature -- the name and type of every
        argument -- and constants, which are the values fixed before the
        kernel runs.  A configuration is mostly constants, so most of it
        belongs there rather than among the options.  The exception is a name
        the runtime reads as an option but which the kernel declares as a
        constant: the warp and stage counts are such names, and when the kernel
        declares them the value is taken from the configuration and put among
        the constants, since a constant the kernel declares is not read as an
        option.
        """

        compile_meta = copy.deepcopy(self.triton_meta)
        compile_meta["num_warps"] = cfg.num_warps
        compile_meta["num_stages"] = cfg.num_stages

        # A configuration names two different kinds of thing.  A name the
        # kernel declares is a value the kernel was compiled with and belongs
        # among the constants.  A name the kernel does not declare is not an
        # argument of it at all -- here it is a value the text was written with,
        # which lives in the module the kernel is defined in -- and passing it
        # to the runtime as though it were an argument would be naming
        # something the kernel has no parameter for.
        kernel_arg_names = set(compile_meta["signature"])
        cfg_kwargs = {
            key: value
            for key, value in cfg.kwargs.items()
            if key in kernel_arg_names
        }
        backend_options = {
            key: value
            for key, value in cfg.kwargs.items()
            if key not in kernel_arg_names
        }
        if backend_options:
            compile_meta["backend_options"] = {
                **compile_meta.get("backend_options", {}),
                **backend_options,
            }
        compile_meta["constants"].update(cfg_kwargs)

        for i in get_constexprs(self.fn):
            arg_name = self.fn.arg_names[i]
            if arg_name not in compile_meta["constants"] and arg_name in (
                "num_warps",
                "num_stages",
            ):
                compile_meta["constants"][arg_name] = getattr(cfg, arg_name)
        if HAS_WARP_SPEC:
            compile_meta["num_consumer_groups"] = getattr(cfg, "num_consumer_groups", 0)
            compile_meta["num_buffers_warp_spec"] = getattr(
                cfg, "num_buffers_warp_spec", 0
            )

        # A descriptor written before the configuration is known names its
        # extents rather than giving them.  Now that the configuration is
        # known they can be said, and the signature entry for a descriptor is
        # the shape it describes.
        host_tma_args = self.inductor_meta.get("host_tma_descriptor_args")
        if host_tma_args:
            all_constants = compile_meta["constants"]
            for key in list(compile_meta["signature"]):
                desc_info = host_tma_args.get(key)
                if desc_info is None or not isinstance(desc_info, dict):
                    continue
                block_shape_vals = _resolve_dims(
                    desc_info["block_shape"], cfg_kwargs, all_constants
                )
                shape_vals = _resolve_dims(
                    desc_info["shape"], cfg_kwargs, all_constants
                )
                stride_vals = _resolve_dims(
                    desc_info["strides"], cfg_kwargs, all_constants
                )
                if (
                    block_shape_vals is None
                    or shape_vals is None
                    or stride_vals is None
                    or any(v <= 0 for v in block_shape_vals)
                ):
                    continue
                ty = compile_meta["signature"][key]
                if isinstance(ty, str) and ty.startswith("*"):
                    dtype_str = ty[1:]
                elif isinstance(ty, str) and ty.startswith("tensordesc<"):
                    dtype_str = ty.split("<")[1].split("[")[0]
                else:
                    continue
                compile_meta["signature"][key] = (
                    f"tensordesc<{dtype_str}{list(block_shape_vals)}>"
                )

        compile_meta["debug"] = _should_enable_triton_debug_asserts(self.inductor_meta)
        compile_meta["device_type"] = self.device_props.type
        compile_meta["cc"] = self.device_props.cc

        for k in tlx_only_cuda_options():
            if v := getattr(cfg, k, None):
                compile_meta[k] = v

        return compile_meta

    def _create_compile_options(self, cfg, compile_meta: dict) -> dict:
        """What the runtime is told about how to compile, beyond the signature.

        These are the runtime's own switches rather than the kernel's: how
        many warps and stages, whether to check the compiled code, and the
        handful of scheduling options that only some backends have.  A backend
        option is passed out of band from the signature, which is why a name
        the kernel does not declare belongs here rather than among the
        constants.
        """

        options = {
            "num_warps": compile_meta["num_warps"],
            "num_stages": compile_meta["num_stages"],
            "debug": compile_meta["debug"],
            # The runtime's own overflow checks are off: a kernel that indexes
            # out of range is caught by the shape guards above it, and the
            # extra assertions cost on every launch.
            "sanitize_overflow": False,
        }
        if "enable_fp_fusion" in compile_meta:
            options["enable_fp_fusion"] = compile_meta["enable_fp_fusion"]
        if HAS_WARP_SPEC:
            options.update(
                {
                    "num_consumer_groups": compile_meta.get("num_consumer_groups", 0),
                    "num_buffers_warp_spec": compile_meta.get(
                        "num_buffers_warp_spec", 0
                    ),
                }
            )
        if self.device_props.type == "cuda":
            options.update(
                {
                    "launch_cooperative_grid": compile_meta.get(
                        "launch_cooperative_grid", False
                    ),
                    "launch_pdl": compile_meta.get("launch_pdl", False),
                }
            )
            if compile_meta.get("disable_ftz", False):
                options["enable_reflect_ftz"] = False
            for k in tlx_only_cuda_options():
                if v := getattr(cfg, k, None):
                    options[k] = v
        options.update(compile_meta.get("backend_options", {}))

        if self.device_props.type == "xpu" and XPU_KERNEL_FORMAT == "zebin":
            options["generate_native_code"] = True

        return options

    def _precompile_config(self, cfg, *, cc_override=None):
        """Compile one configuration now, and keep what it compiled to."""

        from .triton_helpers import set_driver_to_gpu

        compile_meta = self._create_compile_meta(cfg)
        if cc_override is not None:
            compile_meta["cc"] = cc_override

        if not ASTSource:
            raise RuntimeError("the kernel-writing runtime is too old to compile with")

        set_driver_to_gpu()

        # Where the signature names a type, and where it names a constant
        # instead, are the two halves of what the runtime is told: a type says
        # how to read an argument, a constant says the argument is not read.
        compile_args = (
            ASTSource(
                self.fn,
                compile_meta["signature"],
                compile_meta["constants"],
            ),
        )

        target = GPUTarget(
            compile_meta["device_type"],
            compile_meta["cc"],
            self.device_props.warp_size_or_default,
        )
        options = self._create_compile_options(cfg, compile_meta)
        compile_kwargs = {"target": target, "options": options}

        try:
            binary = triton.compile(*compile_args, **compile_kwargs)
        except Exception:
            log.exception(
                "could not compile %s\n%s\nmetadata: %s",
                self.inductor_meta.get("kernel_name", "triton_"),
                self.fn.src,
                compile_meta,
            )
            raise

        TritonBundler.put(
            triton_hash_to_path_key(binary.hash),
            int(self.triton_meta.get("device", 0)),
        )
        static_launcher = StaticTritonCompileResult.can_statically_launch(
            binary, self.inductor_meta, self.triton_meta, self.heuristic_type
        )
        if static_launcher is not None:
            return StaticTritonCompileResult(
                static_launcher, cfg, compile_meta, self.inductor_meta
            )
        return TritonCompileResult(binary, cfg, compile_meta, self.inductor_meta)


    def _post_launch(self) -> None:
        """Close whatever the launch opened, whether or not it succeeded.

        A profiler range that is entered and not left is a range that never
        ends, which is worse than not having recorded the launch at all.  So
        this is what the caller runs after the launch whatever the launch did.
        """

        if (profiler_ctx := self._profiler_ctx) is not None:
            self._profiler_ctx = None
            profiler_ctx.__exit__(None, None, None)
        if (debug_call := self._debug_call) is not None:
            self._debug_call = None
            debug_call.finalize(self.get_device_interface())

    def get_profiler_kwargs(self, stream, launcher):
        """What a profiler is told about the kernel being launched.

        Enough to find the launch again in a trace and to say what it was
        doing: which file it came from, what it is called, the shape of the
        configuration, and the stream it ran on.  The optional entries are only
        present when the build knew them, so a trace does not carry empty
        fields for things nobody measured.
        """

        kernel_kwargs_str = ",".join(
            f"{k}={v}" for (k, v) in launcher.config.kwargs.items()
        )

        ret = {
            "kernel_file": (self.filename or ""),
            "kernel_hash": self.kernel_hash,
            "kernel_backend": "triton",
            "stream": stream,
            "num_warps": launcher.config.num_warps,
            "num_stages": launcher.config.num_stages,
            "kernel_kwargs": kernel_kwargs_str,
        }
        if "kernel_name" in self.inductor_meta:
            ret["kernel_name"] = self.inductor_meta["kernel_name"]
        if "kernel_flop" in self.inductor_meta:
            ret["kernel_flop"] = self.inductor_meta["kernel_flop"]
        if "kernel_num_gb" in self.inductor_meta:
            ret["kernel_num_gb"] = self.inductor_meta["kernel_num_gb"]
        return ret


class CannotStaticallyLaunchKernel(Exception):
    """Why a compiled kernel cannot be launched from its binary alone."""


class StaticTritonCompileResult(CompileResult[_T]):
    """A compiled kernel launched from the binary already on disk.

    A compiled kernel can normally be launched through the runtime, which
    keeps what the launch needs alongside it.  Launched from the binary
    instead, the kernel is loaded onto the device once and the launch becomes
    a call with the arguments the binary expects -- and the setup that call
    needs is far smaller, because none of the compile-time state travels with
    it.

    Whether a given kernel can be launched this way is asked rather than
    assumed: several things can make it impossible, and each of them raises
    :class:`CannotStaticallyLaunchKernel` naming which.  The question is asked
    only when static launching is switched on, since a kernel that cannot be
    launched this way is still perfectly launchable the ordinary way.
    """

    @staticmethod
    def can_statically_launch(kernel, inductor_meta, triton_meta, heuristic_type):
        """The form of this kernel that launches from its binary, if there is one."""

        from .. import config

        if not config.use_static_triton_launcher:
            return None

        def check_can_launch():
            if triton_meta.get("device_type") not in ("cuda", "xpu", "hip"):
                raise CannotStaticallyLaunchKernel("not a device that loads a binary")

            if triton_meta.get("device_type") == "xpu" and XPU_KERNEL_FORMAT == "spv":
                raise CannotStaticallyLaunchKernel(
                    "the host device takes its kernels in a form that cannot be "
                    "launched this way"
                )

            if config.cpp_wrapper:
                # A wrapper is written and compiled for this call anyway, so
                # there is nothing left for this to save.
                raise CannotStaticallyLaunchKernel("the wrapper is written out")

            if (
                heuristic_type == HeuristicType.USER_AUTOTUNE
                and not config.static_launch_user_defined_triton_kernels
            ):
                raise CannotStaticallyLaunchKernel("a user-written kernel")

            if inductor_meta.get("store_cubin"):
                # The whole binary has to be kept, which is what this avoids.
                raise CannotStaticallyLaunchKernel("the binary is being kept")

            if getattr(kernel.metadata, "launch_pdl", False) or getattr(
                kernel.metadata, "launch_cooperative_grid", False
            ):
                raise CannotStaticallyLaunchKernel(
                    "the launch carries attributes this does not pass"
                )

            device_type = triton_meta.get("device_type")
            binary_ext = GPU_KERNEL_BIN_EXTS.get(device_type, ".cubin")
            cubin_location = os.path.join(
                triton_cache_dir(int(triton_meta.get("device", 0))),
                triton_hash_to_path_key(kernel.hash),
                f"{kernel.src.fn.__name__}{binary_ext}",
            )
            if not os.path.exists(cubin_location):
                raise CannotStaticallyLaunchKernel(
                    f"the binary is not where it was left: {cubin_location}"
                )
            kernel._cubin_path = cubin_location

            try:
                return statically_launched_kernel_by_device(kernel, device_type)
            except NotImplementedError as e:
                raise CannotStaticallyLaunchKernel(f"not implemented: {e}") from e

        try:
            return check_can_launch()
        except CannotStaticallyLaunchKernel as e:
            log.info("cannot launch %s statically: %s", kernel, e)
            return None
        except Exception:
            log.info(
                "cannot launch %s statically", kernel, exc_info=True
            )
            return None

    def reload_cubin_path(self):
        """Point the kernel at its binary, putting it back if it was only held.

        A binary that travelled inside a cache entry is held as bytes rather
        than left on disk, so a kernel read back from one has to be written
        out again before it can be loaded.  A binary that is neither on disk
        nor in hand is a cache entry that cannot be used, and saying so is
        better than launching nothing.
        """

        device_type = (
            "hip" if tp.version.hip else self.compile_meta.get("device_type", "cuda")
        )
        binary_ext = GPU_KERNEL_BIN_EXTS.get(device_type, "cubin")
        cubin_location = os.path.join(
            triton_cache_dir(
                _resolve_load_device(self.compile_meta.get("device"), device_type)
            ),
            triton_hash_to_path_key(self.kernel.hash),
            f"{self.kernel.name}{binary_ext}",
        )
        if not os.path.exists(cubin_location):
            if self.kernel.cubin_raw is not None:
                self.kernel.reload_cubin_from_raw(cubin_location)
            else:
                raise RuntimeError(
                    "the binary the entry referred to is not at %s", cubin_location
                )
        self.kernel.cubin_path = cubin_location


class TritonCompileResult(CompileResult[CompiledKernel]):
    """
    A compiled kernel, and what it was compiled for.

    The compiled form is what a launch goes through, and it is the only thing
    that knows the signature, the metadata and the grid the kernel was built
    for -- so the launcher is written from it rather than from the text, and
    the text is not what a launch is made against.
    """

    @staticmethod
    @functools.lru_cache(32)
    def _kernel_metadata_cls(fields: tuple[str, ...]) -> Any:
        return namedtuple("KernelMetadata", sorted(fields))

    @staticmethod
    def _serialize_metadata(metadata):
        """
        The metadata is a tuple whose type is declared inside the module that
        defines it, so a serialized metadata is not recognizable by name when
        it comes back.  It is therefore written out as a plain mapping of
        field to value, and read back with the type rebuilt from the fields it
        was written with.

        What that type is depends on the backend: for some it is a named
        tuple, for others an ordinary one, and only the first needs rebuilding.
        """

        def is_namedtuple(obj) -> bool:
            return (
                isinstance(obj, tuple)
                and hasattr(obj, "_asdict")
                and hasattr(obj, "_fields")
            )

        if is_namedtuple(metadata):
            return metadata._asdict()
        else:
            return metadata

    @staticmethod
    def _deserialize_metadata(metadata):
        if isinstance(metadata, dict):
            return TritonCompileResult._kernel_metadata_cls(tuple(metadata.keys()))(
                **metadata
            )
        else:
            return metadata

    def __getstate__(self) -> dict[str, Any]:
        kernel = self.kernel
        # replace the fields that don't pickle nicely
        kernel_state = {
            **kernel.__dict__,
            # See doc about serializing metadata above
            "metadata": self._serialize_metadata(kernel.metadata),
            "packed_metadata": self._serialize_metadata(
                getattr(kernel, "packed_metadata", None)
            ),
            "module": None,  # regenerated by kernel._init_handles()
            "function": None,  # regenerated by kernel._init_handles()
            "run": None,  # regenerated by kernel._init_handles()
        }
        return {**self.__dict__, "kernel": kernel_state}  # type: ignore[dict-item]

    def __setstate__(self, state: dict[str, Any]) -> None:
        # src = ASTSource.__new__(ASTSource)
        # src.__setstate__(state["kernel"]["src"])
        # TODO(jansel): need to fixup src.fn which is now None
        kernel = CompiledKernel.__new__(CompiledKernel)
        metadata = state["kernel"]["metadata"]
        packed_metadata = state["kernel"]["packed_metadata"]
        kernel.__dict__.update(
            {
                **state["kernel"],
                # "src": src,
                "metadata": self._deserialize_metadata(metadata),
                "packed_metadata": self._deserialize_metadata(packed_metadata),
            }
        )
        self.__dict__.update(state)
        self.kernel = kernel

    def make_launcher(self) -> LauncherType:
        """
        Launching triton kernels is performance sensitive, we compile
        a custom Python function get the grid() and reorder the args to
        the underlying wrapper.
        """
        cfg = self.config
        compile_meta = self.compile_meta
        binary = self.kernel
        fn = binary.src.fn
        binary._init_handles()
        (call_args, def_args, none_args) = self._get_arg_lists(
            fn.arg_names, get_constexprs(fn)
        )
        binary_shared = (
            binary.shared if hasattr(binary, "shared") else binary.metadata.shared
        )

        if knobs is None:
            launch_enter = binary.__class__.launch_enter_hook
            launch_exit = binary.__class__.launch_exit_hook
        else:
            launch_enter = knobs.runtime.launch_enter_hook
            launch_exit = knobs.runtime.launch_exit_hook

        import math as math_lib

        import triton as triton_lib

        import tensorplay as torch_lib

        scope = {
            "grid_meta": cfg.kwargs,
            "bin": binary,
            "launch_enter_hook": launch_enter,
            "launch_exit_hook": launch_exit,
            "metadata": (
                binary.packed_metadata
                if hasattr(binary, "packed_metadata")
                else binary.metadata
            ),
            "shared": binary_shared,
            "num_warps": (
                binary.num_warps
                if hasattr(binary, "num_warps")
                else binary.metadata.num_warps
            ),
            "cta_args": (
                (
                    binary.num_ctas,
                    *get_first_attr(binary, "cluster_dims", "clusterDims"),
                )
                if hasattr(binary, "num_ctas")
                else (
                    (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
                    if hasattr(binary, "metadata")
                    and hasattr(binary.metadata, "num_ctas")
                    and hasattr(binary.metadata, "cluster_dims")
                    else ()
                )
            ),
            "function": get_first_attr(binary, "function", "cu_function"),
            "runner": get_first_attr(binary, "run", "c_wrapper"),
            "math": math_lib,
            "torch": torch_lib,
            "triton": triton_lib,
        }

        if not hasattr(binary, "launch_metadata"):
            # launch args before CompiledKernel.launch_metadata is added.
            # TODO(jansel): delete this branch in mid-2025
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "num_warps",
                "*cta_args",
                "shared",
                "stream",
                "function",
                "launch_enter_hook",
                "launch_exit_hook",
                "metadata",
                *call_args,
            ]
        else:  # args after CompiledKernel.launch_metadata: https://github.com/triton-lang/triton/pull/3492
            # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
            # `bin.launch_metadata` is relatively expensive, and returns None unless a
            # `launch_enter_hook` is installed.  So if we don't have that hook installed,
            # we want to burn None in to the launch args with zero overhead.
            if launch_enter:
                launch_metadata = f"bin.launch_metadata((grid_0, grid_1, grid_2), stream, {', '.join(call_args)})"
            else:
                launch_metadata = "None"
            runner_args = [
                "grid_0",
                "grid_1",
                "grid_2",
                "stream",
                "function",
                "metadata",
                launch_metadata,
                "launch_enter_hook",
                "launch_exit_hook",
                *call_args,
            ]

        pre_runner_lines, runner_args = self._host_tma_pre_runner_lines(
            runner_args, call_args
        )
        if self.inductor_meta.get("host_tma_descriptor_args"):
            # _host_tma_pre_runner_lines already validated the stable TMA API.
            from triton.tools.tensor_descriptor import TensorDescriptor

            scope["_host_tma_aligned"] = _host_tma_aligned
            scope["TensorDescriptor"] = TensorDescriptor

        launcher = self._gen_launcher_code(
            scope, def_args, runner_args, pre_runner_lines=pre_runner_lines
        )

        launcher = scope["launcher"]
        launcher.config = cfg
        launcher.n_regs = getattr(binary, "n_regs", None)
        launcher.n_spills = getattr(binary, "n_spills", None)
        launcher.shared = binary_shared
        launcher.cache_hash = triton_hash_to_path_key(binary.hash)
        launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
        # store this global variable to avoid the high overhead of reading it when calling run
        if launcher.store_cubin:
            launcher.fn = fn
            launcher.bin = binary
            if triton_version_uses_attrs_dict():
                # arg filtering wasn't done above
                cfg_dict = config_to_dict(cfg)
                def_args = [x for x in def_args if x not in cfg_dict]
                call_args = [
                    x
                    for x in call_args
                    if compile_meta["signature"].get(x, "constexpr") != "constexpr"
                    and x not in none_args
                ]
            launcher.def_args = def_args
            launcher.call_args = call_args
            kernel_metadata = getattr(self.kernel, "metadata", None)

            # for the scratch arguments: None indicates that the kernel doesn't
            # take any scratch argument; otherwise a number indicates the number
            # of bytes of scratch that need to be provided.

            # in AMD's Triton backend, the global scratch size is never provided
            # (but for AMD it's safe to pass an extra null arg, so always include it)
            global_scratch: int | None = getattr(
                kernel_metadata,
                "global_scratch_size",
                (0 if torch.version.hip else None),
            )
            profile_scratch: int | None = getattr(
                kernel_metadata, "profile_scratch_size", None
            )
            launcher.global_scratch = global_scratch
            launcher.profile_scratch = profile_scratch
        return launcher


class DebugAutotuner(CachingAutotuner):
    """A tuner that launches the winner and then says what it cost.

    The point of tuning is to pick well, and picking well is not the same as
    knowing why the winner won.  So this launches the chosen configuration for
    real and reports the time it took, how much data it moved to do it, and
    what that works out to as a rate -- because a kernel that is slow because
    it moved a great deal is a different problem from one that is slow because
    it moved little, and the time alone does not say which.

    The measurement is kept: an ahead-of-time export may launch the same kernel
    again in a process that cannot measure it, and reporting a number that was
    measured elsewhere is the only number it can report.
    """

    def __init__(
        self,
        *args,
        regex_filter="",
        with_profiler=False,
        with_bandwidth_info=True,
        **kwargs,
    ):
        self.regex_filter = regex_filter
        self.with_profiler = with_profiler
        self.with_bandwidth_info = with_bandwidth_info
        super().__init__(*args, **kwargs)
        self.cached = None

    def run(self, *args, stream, **kwargs):
        if not self.with_bandwidth_info:
            super().run(*args, stream=stream, **kwargs, benchmark_run=True)
            return
        else:
            possible_names = _find_names(self)
            if possible_names:
                kernel_name = f"{max(possible_names, key=len)}"
            else:
                # A tuner that is not bound to a name at module level finds
                # none; fall back to the name the compilation recorded, and
                # then to the function's own, so that a filter still has
                # something to match against.
                kernel_name = self.inductor_meta.get("kernel_name") or self.fn.__name__
            if not re.match(self.regex_filter, kernel_name):
                return
            if len(self.launchers) != 1:
                if len(self.launchers) == 0:
                    start_time = time.time_ns()
                    self.precompile()
                    self.precompile_time_taken_ns = time.time_ns() - start_time
                if len(self.launchers) > 1:
                    self.autotune_to_one_config(*args, **kwargs)
            (launcher,) = self.launchers

            if self.cached is None:
                ms = self.bench(launcher, *args, with_profiler=self.with_profiler)
                num_in_out_ptrs = len(
                    [
                        arg_name
                        for arg_name in self.fn.arg_names
                        if arg_name.startswith("in_out_ptr")
                    ]
                )
                num_gb = self.inductor_meta.get("kernel_num_gb", None)
                if num_gb is None:
                    num_gb = get_num_bytes(*args, num_in_out_args=num_in_out_ptrs) / 1e9
                gb_per_s = num_gb / (ms / 1e3)
                self.cached = ms, num_gb, gb_per_s, kernel_name
                collected_calls.append((ms, num_gb, gb_per_s, kernel_name))
                log.info(
                    "%s",
                    create_bandwidth_info_str(
                        ms, num_gb, gb_per_s, suffix=f" \t {kernel_name}"
                    ),
                )
            else:
                # An ahead-of-time run calls the kernel, and its timing was
                # measured where the kernel was chosen.
                collected_calls.append(self.cached)


class ComboKernelGrid(GridExpr):
    """Several kernels run as one launch, over a grid that covers all of them.

    One launch for several kernels saves the launches but not the work, so the
    grid has to be big enough for whichever of them has the most to do, and
    every block has to be able to tell which kernel it belongs to -- hence a
    per-kernel extent and a flag saying whether that kernel has one at all.
    """

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        combo_meta = self.inductor_meta["combo_grid_meta"]
        if combo_meta["default_config"]:
            meta = {**combo_meta["default_config"], **meta}
        no_x_dims = []
        xnumels = []
        ynumels = []

        for num in range(combo_meta["num_kernels"]):
            if (
                combo_meta[f"xnumel_{num}"] is not None
                and combo_meta[f"xnumel_{num}"] <= 0
            ):
                raise AssertionError(
                    f"xnumel_{num} must be None or positive, got {combo_meta[f'xnumel_{num}']}"
                )
            no_x_dims.append(combo_meta[f"no_x_dim_{num}"])
            xnumels.append(combo_meta[f"xnumel_{num}"] or f"xnumel_{num}")
            if f"ynumel_{num}" in combo_meta:
                ynumels.append(combo_meta[f"ynumel_{num}"] or f"ynumel_{num}")

        self.x_grid = self.combo_x_grid(xnumels, no_x_dims, meta)
        if combo_meta["min_blocks"]:
            self.x_grid = self.maximum([self.x_grid, combo_meta["min_blocks"]])
        if ynumels:
            self.prefix.extend(
                [
                    self.assign_tmp(
                        "y_grid_raw_",
                        self.ceildiv(self.maximum(ynumels), meta.get("YBLOCK")),
                    ),
                    self.assign_tmp(
                        "y_grid_div_", self.ceildiv("y_grid_raw_", get_max_y_grid())
                    ),
                ]
            )
            ceildiv_expr = self.ceildiv("y_grid_raw_", "y_grid_div_")
            if self.mode == "python":
                self.y_grid = f"(0 if y_grid_div_ == 0 else {ceildiv_expr})"
            else:
                self.y_grid = f"(y_grid_div_ == 0 ? 0 : {ceildiv_expr})"
            self.z_grid = "y_grid_div_"

    def combo_x_grid(
        self,
        xnumels: list,
        no_x_dims: list[bool],
        meta: dict[str, int],
    ):
        raise NotImplementedError


class SequentialComboKernelGrid(ComboKernelGrid):
    """Kernels run one after another, so the grid is the sum of their blocks.

    Sequential rather than interleaved, so a block that belongs to a later
    kernel waits for an earlier one; the launch is one, and what it waits for
    is inside it.
    """

    def combo_x_grid(
        self,
        xnumels: list,
        no_x_dims: list[bool],
        meta: dict[str, int],
    ):
        if len(xnumels) != len(no_x_dims):
            raise AssertionError(
                f"xnumels and no_x_dims length mismatch: {len(xnumels)} != {len(no_x_dims)}"
            )
        return self.summation(
            [
                self.ceildiv(x, 1 if no_x_dim else meta.get("XBLOCK"))
                for x, no_x_dim in zip(xnumels, no_x_dims)
            ]
        )


class SequentialFlattenComboKernelGrid(GridExpr):
    """Kernels run one after another, with each one's two extents folded together.

    Folding the two extents of each kernel into one makes the blocks of the
    whole launch a single range, so a block can find its kernel by where it
    falls rather than by a pair of coordinates that have to be taken apart
    again.
    """

    def generate_lazy(self, kernel_name: str) -> None:
        combo_meta = self.inductor_meta["combo_grid_meta"]
        num_kernels = combo_meta["num_kernels"]
        meta: dict[str, Any] = {}
        for i in range(num_kernels):
            meta[f"XBLOCK_{i}"] = f"{kernel_name}_result.xblocks[{i}]"
            meta[f"YBLOCK_{i}"] = f"{kernel_name}_result.yblocks[{i}]"
        self.generate(meta, is_lazy=True)

    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        combo_meta = self.inductor_meta["combo_grid_meta"]
        if combo_meta["default_config"]:
            meta = {**combo_meta["default_config"], **meta}

        total_blocks_list = []
        for num in range(combo_meta["num_kernels"]):
            xnumel = combo_meta[f"xnumel_{num}"]
            if xnumel is not None and xnumel <= 0:
                raise AssertionError(
                    f"xnumel_{num} must be None or positive, got {xnumel}"
                )
            xnumel = xnumel or f"xnumel_{num}"
            x_blocks = self.ceildiv(
                xnumel,
                1 if combo_meta[f"no_x_dim_{num}"] else meta.get(f"XBLOCK_{num}"),
            )
            y_blocks = (
                self.ceildiv(
                    combo_meta[f"ynumel_{num}"] or f"ynumel_{num}",
                    meta.get(f"YBLOCK_{num}"),
                )
                if f"ynumel_{num}" in combo_meta
                else 1
            )
            total_blocks_list.append(self.product([x_blocks, y_blocks]))

        self.x_grid = self.summation(total_blocks_list)
        if combo_meta["min_blocks"]:
            self.x_grid = self.maximum([self.x_grid, combo_meta["min_blocks"]])
        self.y_grid = 1
        self.z_grid = 1


class RoundRobinComboKernelGrid(ComboKernelGrid):
    """Kernels interleaved block by block, each kernel's blocks spread evenly.

    Interleaved so that a kernel with much less to do does not finish early
    and leave the rest of the launch idle; the grid is as large as the largest
    kernel's, and every block is given a kernel, so no block is wasted on
    work that is already done.
    """

    def combo_x_grid(
        self,
        xnumels: list,
        no_x_dims: list[bool],
        meta: dict[str, int],
    ) -> str:
        if len(xnumels) != len(no_x_dims):
            raise AssertionError(
                f"xnumels and no_x_dims length mismatch: {len(xnumels)} != {len(no_x_dims)}"
            )
        num_kernels = self.inductor_meta["combo_grid_meta"]["num_kernels"]
        exprs = [x for x, no_x_dim in zip(xnumels, no_x_dims) if no_x_dim]
        xnumels_x_dim = [x for x, no_x_dim in zip(xnumels, no_x_dims) if not no_x_dim]
        if xnumels_x_dim:
            exprs.append(self.ceildiv(self.maximum(xnumels_x_dim), meta.get("XBLOCK")))
        return f"({self.maximum(exprs)}) * {num_kernels}"


def _enforce_reduction_config_block_minimums(
    configs: list[Config],
    size_hints: dict[str, int],
    inductor_meta: InductorMeta,
) -> list[Config]:
    min_xblock = inductor_meta.get("min_xblock")
    min_rblock = inductor_meta.get("min_rblock")
    if min_xblock is None and min_rblock is None:
        return configs

    for cfg in configs:
        if frozenset(("YBLOCK", "ZBLOCK", "R1_BLOCK")) & cfg.kwargs.keys():
            raise AssertionError(
                f"min_xblock/min_rblock only support 2D X/R0 configs: {cfg}"
            )
        has_xblock = "XBLOCK" in cfg.kwargs
        has_rblock = "R0_BLOCK" in cfg.kwargs
        if not (has_xblock or has_rblock):
            continue

        x_floor = min_xblock if min_xblock is not None else 1
        r_floor = min_rblock if min_rblock is not None else 1
        target_tile_product = (cfg.kwargs["XBLOCK"] if has_xblock else 1) * (
            cfg.kwargs["R0_BLOCK"] if has_rblock else 1
        )

        if has_xblock:
            cfg.kwargs["XBLOCK"] = max(cfg.kwargs["XBLOCK"], x_floor)
        if has_rblock:
            cfg.kwargs["R0_BLOCK"] = max(cfg.kwargs["R0_BLOCK"], r_floor)

        def current_tile_product() -> int:
            return (cfg.kwargs["XBLOCK"] if has_xblock else 1) * (
                cfg.kwargs["R0_BLOCK"] if has_rblock else 1
            )

        def shrink_to_budget(name: str, floor: int) -> None:
            while (
                name in cfg.kwargs
                and current_tile_product() > target_tile_product
                and cfg.kwargs[name] > floor
            ):
                cfg.kwargs[name] //= 2

        # Preserve the autotuner's original tile-size budget where possible:
        # raising one block to satisfy a floor should shrink the other block.
        shrink_to_budget("R0_BLOCK", r_floor)
        shrink_to_budget("XBLOCK", x_floor)

        check_max_block(cfg.kwargs)
        check_config(
            cfg.kwargs,
            xnumel=size_hints.get("x") if "XBLOCK" in cfg.kwargs else None,
            ynumel=size_hints.get("y") if "YBLOCK" in cfg.kwargs else None,
            znumel=size_hints.get("z") if "ZBLOCK" in cfg.kwargs else None,
        )

    return configs


def unique_configs(configs: list[Config]):
    """Remove duplicate configurations"""
    seen: OrderedSet[Hashable] = OrderedSet()
    pruned_configs = []

    for cfg in configs:
        key = triton_config_to_hashable(cfg)
        if key not in seen:
            seen.add(key)
            pruned_configs.append(cfg)
    return pruned_configs


def cached_autotune(
    size_hints: list[int] | None,
    configs: list[Config],
    triton_meta: TritonMeta,
    heuristic_type,
    filename=None,
    inductor_meta: InductorMeta | None = None,
    custom_kernel=False,
    caching_autotuner_cls: type[CachingAutotuner] = CachingAutotuner,
    debug_autotuner_cls: type[DebugAutotuner] = DebugAutotuner,
):
    """
    A copy of triton.autotune that calls our subclass.  Our subclass
    has additional debugging, error handling, and on-disk caching.
    """
    inductor_meta = {} if inductor_meta is None else inductor_meta
    if size_hints is not None and heuristic_type in (
        HeuristicType.REDUCTION,
        HeuristicType.PERSISTENT_REDUCTION,
    ):
        configs = _enforce_reduction_config_block_minimums(
            configs, size_hints, inductor_meta
        )
    configs = unique_configs(configs)
    if len(configs) != 1 and not filename:
        raise AssertionError("filename required when multiple configs are provided")

    device_prop = triton_meta.get("device")
    if not isinstance(device_prop, DeviceProperties):
        device_prop = None
    dynamic_scale_rblock_eligible = _could_dynamic_scale_rblock(
        size_hints=size_hints,
        heuristic_type=heuristic_type,
        device_prop=device_prop,
        inductor_meta=inductor_meta,
    )
    configs, autotune_cache, autotune_cache_info = check_autotune_cache(
        configs,
        filename,
        inductor_meta,
        dynamic_scale_rblock_eligible=dynamic_scale_rblock_eligible,
    )
    mutated_arg_names = cast("list[str]", inductor_meta.pop("mutated_arg_names", ()))
    optimize_mem = inductor_meta.pop("optimize_mem", True)

    if "restore_value" in triton_meta:
        mutated_arg_names += triton_meta.pop("restore_value")

    reset_to_zero_arg_names: list[str] = []
    if "reset_to_zero" in triton_meta:
        reset_to_zero_arg_names.extend(triton_meta.pop("reset_to_zero"))

    def decorator(fn):
        # Remove XBLOCK from config if it's not a function argument.
        # This way, coordinate descent tuning will not try to tune it.
        #
        # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
        import inspect

        if "XBLOCK" not in inspect.signature(fn.fn).parameters:
            for tconfig in configs:
                if "XBLOCK" in tconfig.kwargs:
                    if tconfig.kwargs["XBLOCK"] != 1:
                        raise AssertionError(
                            f"Expected XBLOCK == 1 when not in fn params, got {tconfig.kwargs['XBLOCK']}"
                        )
                    tconfig.kwargs.pop("XBLOCK")

        if inductor_meta.get("profile_bandwidth"):
            return debug_autotuner_cls(
                fn,
                triton_meta=triton_meta,
                inductor_meta=inductor_meta,
                regex_filter=inductor_meta["profile_bandwidth_regex"],
                with_profiler=inductor_meta[
                    "profile_bandwidth_with_do_bench_using_profiling"
                ],
                configs=configs,
                save_cache_hook=autotune_cache and autotune_cache.save,
                mutated_arg_names=mutated_arg_names,
                reset_to_zero_arg_names=reset_to_zero_arg_names,
                optimize_mem=optimize_mem,
                heuristic_type=heuristic_type,
                size_hints=size_hints,
                custom_kernel=custom_kernel,
                filename=filename,
                with_bandwidth_info=True,
            )
        return caching_autotuner_cls(
            fn,
            triton_meta=triton_meta,
            inductor_meta=inductor_meta,
            configs=configs,
            save_cache_hook=autotune_cache and autotune_cache.save,
            mutated_arg_names=mutated_arg_names,
            reset_to_zero_arg_names=reset_to_zero_arg_names,
            optimize_mem=optimize_mem,
            heuristic_type=heuristic_type,
            size_hints=size_hints,
            custom_kernel=custom_kernel,
            filename=filename,
            autotune_cache_info=autotune_cache_info,
        )

    return decorator


def template(
    num_stages,
    num_warps,
    triton_meta: TritonMeta,
    num_consumer_groups=0,
    num_buffers_warp_spec=0,
    filename=None,
    inductor_meta: InductorMeta | None = None,
    **kwargs,
):
    """
    Compile a triton template
    """
    # Prepare the base configuration
    config_args = {
        "num_stages": num_stages,
        "num_warps": num_warps,
    }

    # Conditionally add arguments based on HAS_WARP_SPEC
    if HAS_WARP_SPEC:
        config_args.update(
            {
                "num_consumer_groups": num_consumer_groups,
                "num_buffers_warp_spec": num_buffers_warp_spec,
            }
        )

    for k in tlx_only_cuda_options():
        if v := triton_meta.get(k, None):
            config_args[k] = v

    return cached_autotune(
        None,
        [triton.Config({}, **config_args)],
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=HeuristicType.TEMPLATE,
        filename=filename,
    )
