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

import copy
import dataclasses
import enum
import functools
import hashlib
import logging
import math
import os
import threading
from typing import Any, Callable, Generic, Literal, TypeVar

import tensorplay as tp

from ..utils import (
    GPU_KERNEL_BIN_EXTS,
    TMA_ALIGNMENT,
    XPU_KERNEL_FORMAT,
    ceildiv,
    tlx_only_cuda_options,
    triton_version_uses_attrs_dict,
)
from .hints import HeuristicType
from ..triton_bundler import TritonBundler
from .cache_dir_utils import triton_cache_dir
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
    get_first_attr,
    get_max_y_grid,
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
        # Why each configuration could not be measured, keyed by the
        # configuration.  Kept apart from a time so that "could not run" is
        # never mistaken for "ran slowly".
        self.benchmark_failure_reasons: dict = {}
        self._debug_call = None
        self.compile_id = None

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
