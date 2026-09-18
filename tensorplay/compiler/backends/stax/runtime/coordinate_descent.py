"""Coordinate descent refinement for compiled kernel launch configs.

Starts from a launch config chosen by the candidate-table benchmark and
walks one tunable field at a time, accepting any neighbour that improves the
measured launch latency beyond a small noise floor (Gauss-Seidel: an
accepted step becomes the working point for the remaining fields of the
same pass).  Block-style fields move by doubling and halving;
software-pipelining depth moves by one.  The walk ends when a full pass
accepts nothing, with an optional single all-direction sweep (the Cartesian
product of every field's neighbour values) as a final escape hatch for
optimum basins the per-field walk cannot reach.

The tuner is device-agnostic: it proposes configs and measures them through
a caller-supplied benchmark, and a config the codegen cannot build is
disqualified rather than fatal.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple, Optional, Sequence, Tuple


# A candidate must beat the incumbent by more than this relative fraction
# before the walk accepts it (0.1%: the noise floor of a short event-timed
# window).
IMPROVEMENT_FRACTION = 0.001


class TunableField(NamedTuple):
    """One tuple position the descent may move.

    ``kind="pow2"`` proposes double/halve steps bounded below by 1;
    ``kind="stages"`` proposes +/-1 steps bounded below by 1.  ``hi`` caps
    the value from above (``None`` = unbounded; a config past the codegen's
    real limits fails to build and is disqualified).
    """

    index: int
    kind: str
    hi: Optional[int] = None


# (XBLOCK, num_warps): 1024 resident threads per block over a 32-thread
# warp bounds the warp count.
POINTWISE_FIELDS = (
    TunableField(0, "pow2"),
    TunableField(1, "pow2", hi=32),
)
# (XBLOCK, num_warps, num_stages) axis-reduction triples.
DIMS_TRIPLE_FIELDS = (
    TunableField(0, "pow2"),
    TunableField(1, "pow2", hi=32),
    TunableField(2, "stages", hi=16),
)
# (XBLOCK, num_warps, RBLOCK, num_stages) axis-reduction quads.
DIMS_QUAD_FIELDS = (
    TunableField(0, "pow2"),
    TunableField(1, "pow2", hi=32),
    TunableField(2, "pow2"),
    TunableField(3, "stages", hi=16),
)
# (XBLOCK, num_warps, NPROG) persistent split-reduction triples.
SPLIT_FIELDS = (
    TunableField(0, "pow2"),
    TunableField(1, "pow2", hi=32),
    TunableField(2, "pow2"),
)


def dims_fields(length: int) -> Tuple[TunableField, ...]:
    """Field layout for an axis-reduction config of ``length`` components."""

    if length == 3:
        return DIMS_TRIPLE_FIELDS
    if length == 4:
        return DIMS_QUAD_FIELDS
    raise ValueError(f"unsupported axis-reduction config length: {length}")


class CoordinateDescentTuner:
    """One-field-at-a-time refinement around a benchmarked baseline config."""

    def __init__(
        self,
        fields: Sequence[TunableField],
        *,
        radius: int = 1,
        check_all_directions: bool = False,
        bench_fn: Optional[Callable[[Any, list], float]] = None,
    ):
        if radius < 1:
            raise ValueError(f"radius must be >= 1, got {radius}")
        self.fields = tuple(fields)
        self.radius = radius
        self.check_all_directions = check_all_directions
        self._bench_fn = bench_fn
        self._timings: dict[tuple, float] = {}
        self._launches: dict[tuple, Any] = {}

    # -- neighbour generation -------------------------------------------------

    def _neighbour_values(self, field: TunableField, value: int) -> list[int]:
        """Neighbour values within ``radius`` steps, bounds applied."""

        out: list[int] = []
        if field.kind == "stages":
            current = value
            for _ in range(self.radius):
                current += 1
                if field.hi is not None and current > field.hi:
                    break
                out.append(current)
            current = value
            for _ in range(self.radius):
                current -= 1
                if current < 1:
                    break
                out.append(current)
            return out
        current = value
        for _ in range(self.radius):
            current *= 2
            if field.hi is not None and current > field.hi:
                break
            out.append(current)
        current = value
        for _ in range(self.radius):
            current //= 2
            if current < 1:
                break
            out.append(current)
        return out

    def _neighbour_configs(
        self, config: tuple, field: TunableField
    ) -> list[tuple]:
        out = []
        for value in self._neighbour_values(field, config[field.index]):
            candidate = list(config)
            candidate[field.index] = value
            out.append(tuple(candidate))
        return out

    def _all_direction_configs(self, config: tuple) -> list[tuple]:
        """Cartesian product of every field's neighbour values plus itself."""

        value_lists = [
            self._neighbour_values(field, config[field.index])
            + [config[field.index]]
            for field in self.fields
        ]
        out: list[tuple] = []

        def walk(index: int, accumulator: list) -> None:
            if index == len(self.fields):
                out.append(tuple(accumulator))
                return
            field = self.fields[index]
            for value in value_lists[index]:
                candidate = list(accumulator)
                candidate[field.index] = value
                walk(index + 1, candidate)

        walk(0, list(config))
        return out

    # -- benchmarking ---------------------------------------------------------

    def _bench(self, launch: Any, args: list) -> float:
        if self._bench_fn is not None:
            return self._bench_fn(launch, args)
        from .stax_autotune import bench_launch

        return bench_launch(launch, args)

    def _measure(self, build: Callable, config: tuple, args: list) -> float:
        """Benchmark one config; results are memoized for the session."""

        known = self._timings.get(config)
        if known is not None:
            return known
        try:
            launch = self._launches.get(config)
            if launch is None:
                launch = build(config)
            timing = self._bench(launch, args)
        except Exception:  # noqa: BLE001 - an unbuildable config is a dead end
            self._timings[config] = float("inf")
            return float("inf")
        self._launches[config] = launch
        self._timings[config] = timing
        return timing

    @staticmethod
    def _improves(baseline: float, candidate: float) -> bool:
        return candidate < baseline * (1.0 - IMPROVEMENT_FRACTION)

    # -- the walk -------------------------------------------------------------

    def refine(
        self, build: Callable, config: Sequence[int], args: list
    ) -> Tuple[tuple, Any]:
        """Walk from ``config``; return ``(best_config, best_launch)``.

        The baseline is measured first (its launch comes back with it); the
        walk accepts strictly improving neighbours and repeats passes while
        any field moved.  With ``check_all_directions`` the all-direction
        sweep runs once each time the per-field walk stalls.
        """

        best_config = tuple(int(value) for value in config)
        best_timing = self._measure(build, best_config, args)
        improved = True
        while improved:
            improved = False
            for field in self.fields:
                for candidate in self._neighbour_configs(best_config, field):
                    timing = self._measure(build, candidate, args)
                    if self._improves(best_timing, timing):
                        best_config = candidate
                        best_timing = timing
                        improved = True
            if not improved and self.check_all_directions:
                for candidate in self._all_direction_configs(best_config):
                    timing = self._measure(build, candidate, args)
                    if self._improves(best_timing, timing):
                        best_config = candidate
                        best_timing = timing
                        improved = True
        launch = self._launches.get(best_config)
        if launch is None:
            launch = build(best_config)
        return best_config, launch


def refiner_for(
    fields: Sequence[TunableField],
    *,
    radius: int = 1,
    check_all_directions: bool = False,
    bench_fn: Optional[Callable[[Any, list], float]] = None,
) -> Callable[[Callable, tuple, list], Tuple[tuple, Any]]:
    """Build a ``refine(build, config, args)`` callable for one kernel family.

    This matches the ``refiner`` protocol consumed by the autotune sites:
    it receives the winner of the candidate-table benchmark and returns the
    refined ``(config, launch)`` pair.  ``bench_fn`` injects a benchmark
    (tests); ``None`` uses the real event-timed launch benchmark.
    """

    tuner = CoordinateDescentTuner(
        fields,
        radius=radius,
        check_all_directions=check_all_directions,
        bench_fn=bench_fn,
    )
    return tuner.refine
