"""Execution of a scheduled loop-IR program.

The scheduler's order is the execution order: a buffer is allocated just
before the kernel that first writes it, library calls run with realized
inputs, and a buffer is released as soon as its last reader has run.  Graph
outputs are kept alive for the caller.

Extern calls see the same tensors the kernels do, so a strided view built
for a library call addresses the very memory the fused kernels write.
"""

from __future__ import annotations

from typing import Any

import tensorplay as tp

from .loops import Buffer, ComputedBuffer, ConstantBuffer, ExternKernel, ExternOutput, ReinterpretView


class Step:
    """One scheduled item: a fused kernel or a library call."""

    def __init__(self, reads: set, writes: set):
        self.reads = reads
        self.writes = writes
        self.frees: list = []


class FusedStep(Step):
    def __init__(self, launch, ptr_names: set, stored: set, buffers: dict):
        super().__init__(set(ptr_names), set(stored))
        self.launch = launch
        self.ptr_names = list(ptr_names)
        self.buffers = buffers

    def run(self, env: dict) -> None:
        for name in self.writes:
            buffer = self.buffers[name]
            if isinstance(buffer, ConstantBuffer):
                continue
            env[name] = tp.empty(
                buffer.get_size(),
                dtype=_dtype_of(buffer.layout.dtype),
                device=buffer.layout.device,
            )
        self.launch([env[name] for name in self.ptr_names])


class ExternStep(Step):
    def __init__(self, kernel: ExternKernel):
        super().__init__(
            {b.name for b in kernel.input_buffers()}, {o.name for o in kernel.outputs}
        )
        self.kernel = kernel

    def run(self, env: dict) -> None:
        args = _resolve(self.kernel.args, env)
        kwargs = _resolve(self.kernel.kwargs, env)
        if self.kernel.call_method:
            # A method call names the operation with a string and receives the
            # object it is called on first.
            receiver, *rest = args
            result = getattr(receiver, self.kernel.target)(*rest, **kwargs)
        else:
            target = self.kernel.target
            if not callable(target) and hasattr(target, "default"):
                target = target.default
            result = target(*args, **kwargs)
        for output in self.kernel.outputs:
            env[output.name] = _dig(result, output.path)


def _dtype_of(layout_dtype: Any):
    """The framework element type for a layout.

    Lowerings name the element type either as a framework type or by its
    name, and the loop algebra only ever looks at the name, so both spellings
    reach allocation.
    """

    if isinstance(layout_dtype, str):
        return getattr(tp, layout_dtype)
    return layout_dtype


def _resolve(value: Any, env: dict) -> Any:
    if isinstance(value, ReinterpretView):
        base = _resolve(value.buffer, env)
        return tp.as_strided(base, value.size, value.stride, value.offset)
    if isinstance(value, Buffer):
        return env[value.name]
    if isinstance(value, (list, tuple)):
        return type(value)(_resolve(item, env) for item in value)
    if isinstance(value, dict):
        return {key: _resolve(item, env) for key, item in value.items()}
    return value


def _dig(result: Any, path: tuple) -> Any:
    for position in path:
        result = result[position]
    return result


class LoopProgram:
    """A lowered, scheduled and compiled program, ready to call."""

    def __init__(self, graph, steps: list):
        self.graph = graph
        self.steps = steps
        self.keep = set()
        for out in graph.graph_outputs:
            if isinstance(out, ReinterpretView):
                self.keep.add(out.buffer.name)
            elif isinstance(out, Buffer):
                self.keep.add(out.name)
        self._plan_releases()

    def _plan_releases(self) -> None:
        last = {}
        for position, step in enumerate(self.steps):
            for name in step.reads:
                last[name] = position
        for position, step in enumerate(self.steps):
            step.frees = [
                name
                for name in sorted(step.writes)
                if name not in self.keep and last.get(name, -1) == position
            ]

    def __call__(self, *args):
        env: dict = {}
        for position, value in enumerate(args):
            if position >= len(self.graph.graph_inputs):
                break
            buffer = self.graph.graph_inputs[position]
            if buffer is None or value is None:
                continue
            env[buffer.name] = _input_tensor(buffer, value)
        env.update(self.graph.constants)
        for step in self.steps:
            step.run(env)
            for name in step.frees:
                env.pop(name, None)
        results = [_output_tensor(out, env) for out in self.graph.graph_outputs]
        if getattr(self.graph, "single_output", False) and len(results) == 1:
            return results[0]
        return results


def _input_tensor(buffer: Buffer, value):
    """The tensor a kernel indexes with the buffer's own layout.

    A non-zero storage offset is folded into the base pointer, because the
    layout addresses from the start of the buffer.
    """

    try:
        offset = int(value.storage_offset())
    except Exception:  # noqa: BLE001 - tensors without an offset are fine
        return value
    if offset == 0:
        return value
    return tp.as_strided(value, buffer.get_size(), buffer.layout.stride, 0)


def _output_tensor(out: Any, env: dict):
    if isinstance(out, ReinterpretView):
        return tp.as_strided(env[out.buffer.name], out.size, out.stride, out.offset)
    if isinstance(out, Buffer):
        return env[out.name]
    return out


__all__ = ["ExternStep", "FusedStep", "LoopProgram", "Step"]
