"""Render a scheduled loop-IR group for the host, from the group's own nodes.

The IR is the one every device lowers to, and each device prints it in its own
language: :mod:`.loop_triton` prints it as Triton, this prints it as the host
kernel the C++ backend builds.  A group's node is a recorded body -- a tree of
values over the loop's variables -- and what the host emitter wants is the flat
instruction list its expression tables already read: one entry per value, in
the order the values are produced, with operands named by number.

So this module is the walk between the two: it records each node in the group
and numbers the values it finds, an input by its position, an intermediate by
the slot it was given, a literal by its index.  A group whose addressing the
host emitter cannot express -- a value reached through a computed index, a read
the group's own masks would drop -- is not rendered here at all: the caller
declines it and the region runs as the framework wrote it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

from ..ir import Loops, Pointwise, Reduction
from ..loops import Value, record_body, set_graph
from .index_expr import Const

#: The element types the host emitter prints.  Its kernels run in one
#: arithmetic width, so a body of any other width is declined rather than
#: printed in a width it does not have.
_HOST_DTYPES = ("float32",)


class HostPlanError(Exception):
    """This group is not one the host emitter can print.

    Raised for a form rather than for a failure: the caller keeps the region
    on the framework's own execution, which is what a fall back to the
    framework is for.
    """


class _Numberer:
    """Names the values of a group: inputs, intermediates, literals."""

    def __init__(self, input_count: int):
        self.input_count = input_count
        self.instructions: list[tuple[str, int, int, int]] = []
        self.constants: list[float] = []
        # Buffer name -> the number its value is stored at.
        self.by_name: dict[str, int] = {}
        self._temps = 0

    def literal(self, value: float) -> int:
        for index, existing in enumerate(self.constants):
            if existing == value:
                return -index - 1
        self.constants.append(value)
        return -len(self.constants)

    def temp(self, name: str, dtype: str) -> int:
        ref = self.input_count + self._temps
        self._temps += 1
        self.by_name[name] = ref
        return ref

    def ref_of(self, value: Any) -> int:
        if isinstance(value, Const):
            return self.literal(value.value)
        if not isinstance(value, Value):
            raise HostPlanError(f"operand is not a recorded value: {value!r}")
        op = value.op
        if op == "load":
            return self._load_ref(value)
        if op == "constant":
            return self.literal(value.args[0])
        if op == "index_expr":
            # The host emitter addresses its inputs linearly, so an operand
            # addressed through a computed index is a form it cannot print.
            raise HostPlanError("a computed index reaches the host emitter")
        return self._emitted[value]

    def _load_ref(self, value: Value) -> int:
        name = value.args[0]
        ref = self.by_name.get(name)
        if ref is None:
            raise HostPlanError(f"a group reads a value it does not produce: {name}")
        if value.args[2] is not None:
            raise HostPlanError("a masked read reaches the host emitter")
        return ref

    def emit(self, root: Value) -> int:
        """Record ``root``'s value tree, in the order the values are produced."""

        self._emitted: dict[Value, int] = {}
        return self._visit(root)

    def _visit(self, value: Any) -> int:
        if isinstance(value, Const):
            return self.literal(value.value)
        if not isinstance(value, Value):
            raise HostPlanError(f"operand is not a recorded value: {value!r}")
        if value.op == "load":
            return self._load_ref(value)
        if value.op == "constant":
            return self.literal(value.args[0])
        if value in self._emitted:
            return self._emitted[value]
        args = tuple(self._visit(arg) for arg in value.args)
        if value.op == "to_dtype":
            # A conversion is a no-op for the host emitter, whose kernels run
            # in one arithmetic width; the operand is the value.
            ref = args[0]
            self._emitted[value] = ref
            return ref
        if value.op == "where":
            # A masked read is declined above, so a select here has one side
            # unreachable: the value is that side's other operand.
            raise HostPlanError("a select reaches the host emitter")
        if len(args) == 1:
            name, lhs, rhs = value.op, args[0], 0
        elif len(args) == 2:
            name, lhs, rhs = value.op, args[0], args[1]
        else:
            raise HostPlanError(f"no host form for a {len(args)}-operand value")
        if value.dtype not in _HOST_DTYPES:
            raise HostPlanError(f"no host form for element type {value.dtype!r}")
        # The slot the result is stored at is the last field, so the emitter
        # can address the value by the same number everywhere else.
        ref = self.temp(f"$v{len(self.instructions)}", value.dtype)
        self.instructions.append((name, lhs, rhs, ref))
        self._emitted[value] = ref
        return ref


def flatten(group, buffers: dict, stored: set, graph=None) -> dict:
    """The group's value trees as one flat instruction list.

    ``stored`` names the buffers the group writes; each becomes a slot the
    next node in the group reads by number, which is what lets a chain of
    nodes print as one straight-line body.
    """
    if any(not hasattr(node, "data") for node in group.nodes):
        # A library call is not straight-line code: it keeps its own result and
        # its own operands, so a group that carries one is not printed here.
        raise HostPlanError("a library call reaches the host emitter")
    nodes = list(group.nodes)
    if any(isinstance(node.data, Reduction) for node in nodes):
        # A reduction is a loop of its own, and the host emitter's reduction
        # path takes its extent and its body separately; that is its own
        # renderer, not this walk.
        raise HostPlanError("a reduction reaches the host emitter")
    # The group's inputs are what it reads from outside itself: a name it also
    # writes is an intermediate it computes in a slot, not an operand.  Sorted
    # so the numbering the emitter sees is the same on every run of the group.
    inputs = sorted(name for name in group.reads if name not in stored)
    numberer = _Numberer(len(inputs))
    for position, name in enumerate(inputs):
        numberer.by_name[name] = position
    # Recording a body reads each buffer's layout out of the graph the nest
    # was lowered into, so that graph is installed for the walk.
    with set_graph(graph) if graph is not None else nullcontext():
        for node in nodes:
            body = record_body(node.data)
            result = body.root
            if isinstance(result, tuple):
                raise HostPlanError("a multi-output value reaches the host emitter")
            numberer.by_name[_buffer_of(node)] = numberer.emit(result)
    outputs: list[int] = []
    for name in stored:
        ref = numberer.by_name.get(name)
        if ref is None:
            raise HostPlanError(f"the group writes a value it does not produce: {name}")
        outputs.append(ref)
    if len(outputs) != 1:
        raise HostPlanError("only a single-result group is printed on the host")
    return {
        "instructions": numberer.instructions,
        "constants": numberer.constants,
        "input_count": len(inputs),
        "output_ref": outputs[0],
        "inputs": inputs,
    }


def _buffer_of(node) -> str:
    """The buffer a node's result is stored in."""

    buffers = getattr(node, "buffers", None)
    if not buffers:
        raise HostPlanError("a value in the group is not stored in a buffer")
    return buffers[0].name


__all__ = ["HostPlanError", "flatten"]
