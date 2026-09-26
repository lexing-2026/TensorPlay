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
        self._baked = None

    def run(self, env: dict) -> None:
        args = _resolve(self.kernel.args, env)
        kwargs = _resolve(self.kernel.kwargs, env)
        if getattr(self.kernel, "template", None) is not None:
            self._run_template(args, kwargs, env)
            return
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

    def _run_template(self, args, kwargs, env) -> None:
        """Run a templated operator through the launcher its template chose.

        The choice needs the real operands, so it is made on the first call and
        kept on the step: afterwards the baked launcher is replayed, and the
        template's own cache means the measurement happens once per shape even
        across processes.
        """

        feed = _template_feed(self.kernel, args)
        launch = self._baked
        if launch is None and self.kernel.template is not None:
            # A template is measured once, on the operands of the call that
            # found it, and what it hands back is a launcher over the operands
            # it is handed -- so the measurement pins nothing.
            self._baked = self._bake(feed, args, kwargs)
            launch = self._baked
        if launch is not None:
            result = launch(feed)
        else:
            # No template claimed this operator, so the call runs the operator
            # as the framework runs it, on the operands of this call.  Holding
            # a launcher here instead would pin one call's operands on the
            # step, and the program reuses that step for every call it runs.
            result = self._call_target(args, kwargs)
        for output in self.kernel.outputs:
            env[output.name] = _dig(result, output.path)

    def _bake(self, feed, args, kwargs):
        """Ask the template which of its configurations fits this call."""

        template = self.kernel.template
        meta = dict(self.kernel.template_meta)
        meta["feed"] = list(feed)
        meta.setdefault("qualifies", True)
        # The floor a template is measured against: it replays the call being
        # measured, because that is the only call whose result is known to be
        # right.  It is transient -- the measurement keeps the winner, not this.
        plain = lambda values: self._call_target(args, kwargs)
        try:
            specs = template.out_specs(meta)
            for params in template.configurations(specs, meta):
                from .templates import TemplateChoice

                choice = TemplateChoice(template, params, specs, meta)
                launch = choice.resolve(plain)
                if launch is not None:
                    self.kernel.config = params
                    return launch
        except Exception:  # noqa: BLE001 - a template never breaks the region
            return None
        return None

    def _call_target(self, args, kwargs):
        if self.kernel.call_method:
            receiver, *rest = args
            return getattr(receiver, self.kernel.target)(*rest, **kwargs)
        target = self.kernel.target
        if not callable(target) and hasattr(target, "default"):
            target = target.default
        return target(*args, **kwargs)


def _dtype_of(layout_dtype: Any):
    """The framework element type for a layout.

    Lowerings name the element type either as a framework type or by its
    name, and the loop algebra only ever looks at the name, so both spellings
    reach allocation.
    """

    if isinstance(layout_dtype, str):
        return getattr(tp, layout_dtype)
    return layout_dtype


def _template_feed(kernel, args) -> list:
    """The operand list a template's launcher consumes, in recorded order."""

    positions = getattr(kernel, "template_meta", {}).get("operand_positions") or ()
    return [args[i] for i in positions]


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
