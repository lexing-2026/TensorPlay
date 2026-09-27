"""Two index helpers the templates address operands with."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence


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


class ChoiceCaller:
    """One way of doing a thing, built and ready to be measured or used.

    A choice is measured first and turned into a value only if it is the one
    that was chosen, so the two questions are answered by the same object and
    in that order: what it costs to run, and -- only for the winner -- what
    running it produced.  A caller of this class holds a choice that has
    already been built; everything above it was a description of a possibility,
    and this is the possibility.

    What a particular kind of choice is called, how it is run, what identifies
    it and what it produces are all left to the kind: this holds what is the
    same for all of them.
    """

    def __init__(
        self,
        name: str,
        input_nodes=(),
        layout: "Layout | None" = None,
        description: str = "",
    ) -> None:
        self.name = name
        self.layout = layout
        self.input_nodes = input_nodes
        #: An additional description, for knowing what is being chosen between
        #: when the names alone do not say.
        self.description = description
        #: Set when a measurement showed this choice does not work here.
        self.failed: bool = False
        #: When true, measure by capturing the launch and replaying it, which
        #: leaves out the cost of launching and is only right where what is
        #: being compared is the kernel rather than the launch.
        self._benchmark_with_cudagraphs: bool = False
        #: Where information travels from a choice being generated to the end
        #: of it being measured, for the two to be read by code that is not
        #: either of them.
        self.annotations: dict[str, object] = {}
        #: What a subgraph-based choice stands for, filled in by the kinds that
        #: are one.
        self.gm: Any = None
        self.decomposition: Callable[..., Any] | None = None
        self.decomposition_kwargs: dict[str, object] = {}
        #: Geometry substitutions a measurement decided on, for the record.
        self.config_patches: dict[str, Any] = {}
        #: What a measurement hands this choice to run it, when the kind
        #: supplies it.
        self._callable: Callable[..., Any] | None = None

    def benchmark(self, *args: Any, out: Any) -> float:
        """How long one run of this choice takes."""

        algo = self.to_callable()
        from ..runtime.stax_autotune import bench_launch

        return bench_launch(lambda these: algo(*these), list(args))

    def call_name(self) -> str:
        raise NotImplementedError

    def to_callable(self) -> Callable[..., Any]:
        raise NotImplementedError

    def kernel_hash_key(self) -> str:
        """What identifies the kernel itself, for a binary cache.

        By default a choice has no runtime parameters of its own, so what
        identifies the choice identifies the kernel.
        """

        return self.hash_key()

    def hash_key(self) -> str:
        raise NotImplementedError

    def output_node(self) -> Any:
        raise NotImplementedError

    def info_dict(self) -> dict:
        """What is worth writing down about this choice."""

        return {}

    def autoheuristic_id(self) -> str:
        return "unsupported_choice"

    def mark_failed(self) -> None:
        """Record that this choice does not work here, so it is not offered.

        Useful where measuring is separate from choosing: a choice that has
        been found not to work is not offered again.
        """

        self.failed = True
