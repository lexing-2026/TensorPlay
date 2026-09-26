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

import dataclasses
import enum
import functools
import math
from typing import Any, Callable, Literal

from ..utils import ceildiv
from .triton_compat import HAS_WARP_SPEC, Config
from .runtime_utils import get_max_y_grid


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


@dataclasses.dataclass
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
        if isinstance(cfg, dict):
            grid.generate(cfg)
        return grid

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


class Grid1D(GridExpr):
    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))


class Grid2D(GridExpr):
    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))


class Grid3D(GridExpr):
    def generate(self, meta: dict[str, int], is_lazy: bool = False) -> None:
        self.x_grid = self.ceildiv("xnumel", meta.get("XBLOCK"))
        self.y_grid = self.ceildiv("ynumel", meta.get("YBLOCK"))
        self.z_grid = self.ceildiv("znumel", meta.get("ZBLOCK"))


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
