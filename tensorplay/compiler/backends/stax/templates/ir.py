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

    def output_node(self, out: Any = None) -> Any:
        """The result this choice produced, for a caller that wants the value.

        A choice is measured by running it, and the run's result is the point
        of running it.  A caller that already has the result passes it, so a
        choice that writes into a buffer someone else owns hands back that
        buffer rather than a second one.
        """

        return out

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


class ExternKernelAlloc:
    """A choice whose kernel is the framework's own, writing into a fresh buffer.

    An extern choice normally hands back whatever the library call returned.
    This one is different in exactly that respect: the result is written into a
    buffer this choice allocates, because the caller wants a buffer it can put
    in a graph and a returned tensor is not one -- it has no layout the rest of
    the graph can rely on, and no name to be referred to by.

    So the output is allocated here, up front, from the layout that was asked
    for; the library call writes into it, and what the choice hands back is
    that buffer.  A choice that has no layout cannot allocate one, and saying
    so is better than handing back something whose extent was guessed.
    """

    def __init__(
        self,
        layout,
        inputs: tuple = (),
        constant_args: tuple = (),
        kwargs: dict | None = None,
        python_kernel_name: str | None = None,
        op_overload: str | None = None,
    ):
        if layout is None:
            raise AssertionError(
                "an allocating choice needs a layout: without one there is "
                "nothing to allocate"
            )
        self.layout = layout
        self.inputs = tuple(inputs)
        self.constant_args = tuple(constant_args)
        self.kwargs = dict(kwargs or {})
        self.python_kernel_name = python_kernel_name
        self.op_overload = op_overload
        self.name = python_kernel_name or "extern_alloc"
        self.outputs: list = []

    def codegen(self, wrapper) -> None:
        """Have the wrapper emit this choice's launch.

        The emitting is the wrapper's: it is the thing that knows the calling
        convention, which buffers exist by the time this runs, and what a
        launch has to be surrounded by.  So the choice says what it is and the
        wrapper says how to say it.
        """

        wrapper.generate_extern_kernel_alloc(self)

    def should_allocate(self) -> bool:
        """Whether the output buffer is this choice's to make.

        It is: that is the whole difference between this and a choice that
        hands back whatever the library returned, because a returned tensor has
        no layout the rest of the graph can rely on and no name to be referred
        to by.
        """

        return True

    def apply_constraint(self) -> None:
        raise NotImplementedError

    def allocate(self):
        """The buffer this choice writes into."""

        import tensorplay as tp

        # A layout records its element type by name, because a layout is
        # written down before anything exists to ask; the allocator wants the
        # type itself, and the name is how the two are connected.
        return tp.empty(tuple(self.layout.size),
                        dtype=getattr(tp, self.layout.dtype),
                        device=self.layout.device)

    def __repr__(self) -> str:
        return f"ExternKernelAlloc({self.name})"
