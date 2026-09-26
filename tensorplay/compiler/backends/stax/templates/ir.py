"""Layouts, the inputs a template is handed, and the callers that hold a built choice.

Three things a template needs to describe a call and three things a caller needs to
hold afterwards.  They are one vocabulary: a layout says where a result lands, the
inputs say what the call is, and a choice caller says which kernel was built and how
to run it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

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

    def mnk(self) -> tuple:
        """The extents a product of these operands has."""

        raise NotImplementedError
@dataclass(frozen=True)
class MMKernelInputs(KernelInputs):
    """The operands of a product: two matrices, the second contracting with the
    first.

    The three extents are asked for by name because a heuristic sizing itself
    to a product should not have to know which operand an extent came from, and
    a caller writing a configuration should not have to repeat the order.
    """

    def mnk(self) -> tuple:
        (m, k), (k2, n) = self.shapes[0], self.shapes[1]
        if k != k2:
            raise NotImplementedError("operands that do not share a contraction")
        return m, n, k

    def mnk_symbolic(self) -> tuple:
        """The extents as written, for a grid that is sized before it is run."""

        return self.mnk()

    def is_contiguous(self) -> bool:
        return bool(self.extra.get("qualifies", False))
@dataclass(frozen=True)
class ConvKernelInputs(KernelInputs):
    """The operands of a convolution: an activation and a weight.

    A convolution's product is over the input channels, but what it produces
    is positions, so the extents a configuration is sized against are counted
    differently from a product's: the positions of every image in the batch
    together, because a tile does not care which image a position came from.
    """

    def mnk(self) -> tuple:
        extra = self.extra
        rows = extra.get("conv_rows")
        cols = extra.get("out_channels")
        inner = extra.get("in_channels_per_group")
        if not (rows and cols and inner):
            raise NotImplementedError("a convolution whose geometry is unknown")
        return rows, cols, inner

    def mnk_symbolic(self) -> tuple:
        return self.mnk()

    @property
    def is_depthwise(self) -> bool:
        """Each input channel reduced on its own: not a product at all."""

        groups = int(self.extra.get("groups", 1) or 1)
        return groups > 1 and self.extra.get("in_channels_per_group") == 1

    def is_1x1(self) -> bool:
        kernel = tuple(self.extra.get("kernel_size") or ())
        stride = tuple(self.extra.get("stride") or ())
        padding = tuple(self.extra.get("padding") or ())
        return bool(kernel) and all(k == 1 for k in kernel) and all(
            s == 1 for s in stride
        ) and all(p == 0 for p in padding) and int(
            self.extra.get("groups", 1) or 1
        ) == 1 and not self.extra.get("transposed")
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

        from ..runtime.stax_autotune import bench_launch

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
class CutedslChoiceCaller(ChoiceCaller):
    """A choice whose kernel is emitted for the device-side dialect.

    It is a separate kind for the same reason the other two are: the kernel is
    written in a different language, so the source that identifies it in a
    cache is not interchangeable with the streaming dialect's, and a change to
    one emitter must not look like a change to the other.
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
        return "cutedsl_template"
def caller_for(backend: str):
    """The caller kind that identifies a kernel emitted for this backend."""

    return BACKEND_CALLERS.get(backend, TritonChoiceCaller)
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

        from ..runtime.stax_autotune import bench_launch

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
class CutedslChoiceCaller(ChoiceCaller):
    """A choice whose kernel is emitted for the device-side dialect.

    It is a separate kind for the same reason the other two are: the kernel is
    written in a different language, so the source that identifies it in a
    cache is not interchangeable with the streaming dialect's, and a change to
    one emitter must not look like a change to the other.
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
        return "cutedsl_template"
def caller_for(backend: str):
    """The caller kind that identifies a kernel emitted for this backend."""

    return BACKEND_CALLERS.get(backend, TritonChoiceCaller)
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


