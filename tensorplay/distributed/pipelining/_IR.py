"""Pipeline intermediate representation and split annotations."""

import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable

from tensorplay.nn.modules.container import Sequential
from tensorplay.nn.modules.module import Module

from ._backward import _null_coalesce_accumulate
from ._utils import PipeInfo
from .stage import build_stage

__all__ = ["Pipe", "pipe_split", "SplitPoint", "pipeline"]


def get_submod_name(stage_idx: int) -> str:
    return f"submod_{stage_idx}"


def _find_loss_from_output_and_spec(output_val: Any, spec_val: Any) -> Any:
    if spec_val is True:
        return output_val
    if isinstance(output_val, dict) and isinstance(spec_val, dict):
        for key, spec in spec_val.items():
            if key in output_val:
                found = _find_loss_from_output_and_spec(output_val[key], spec)
                if found is not None:
                    return found
    if isinstance(output_val, (tuple, list)) and isinstance(spec_val, (tuple, list)):
        for value, spec in zip(output_val, spec_val):
            found = _find_loss_from_output_and_spec(value, spec)
            if found is not None:
                return found
    return None


def _find_loss_output(mod: Any, g: Any, output_loss_value_spec: Any) -> Any:
    del mod, g
    return output_loss_value_spec


def _insert_stage_symbolic_backward(g: Any, loss_node: Any, output_node: Any) -> Any:
    del loss_node, output_node
    return g


def _move_placeholders_to_front(graph: Any) -> Any:
    return graph


class PipeSequential(Sequential):
    @staticmethod
    def from_sequential(sequential_instance: Sequential) -> "PipeSequential":
        return PipeSequential(*list(sequential_instance))

    def forward(self, input: Any) -> Any:
        return super().forward(input)


class LossWrapper(Module):
    def __init__(self, module: Module, loss_fn: Callable[..., Any]) -> None:
        super().__init__()
        self.module = module
        self.loss_fn = loss_fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.module(*args, **kwargs)
        return self.loss_fn(output)


class TrivialLossWrapper(Module):
    def forward(self, x: Any, targets: Any) -> Any:
        return x, targets


def _pipe_split() -> None:
    return None


pipe_split = _pipe_split


class MultiUseParameterConfig(Enum):
    TRANSMIT = auto()
    REPLICATE = auto()


class DetachExecutor:
    def __init__(self, module: Any, garbage_collect_values: bool = True) -> None:
        self.module = module
        self.garbage_collect_values = garbage_collect_values

    def run(self, initial_env: Any, *args: Any, **kwargs: Any) -> Any:
        del initial_env
        return self.module(*args, **kwargs)

    def call_module(self, target: Any, args: Any, kwargs: Any) -> Any:
        return getattr(self.module, target)(*args, **kwargs)

    def call_function(self, target: Any, args: Any, kwargs: Any) -> Any:
        return target(*args, **kwargs)


class _NodeReference:
    def __init__(self, name: str) -> None:
        self.name = name


class _LinearNodeList:
    def __init__(self, node_list: list[Any]) -> None:
        self.node_list = node_list

    def to_graph(self) -> Any:
        return list(self.node_list)


class Pipe(Module):
    def __init__(self, split_gm: Any, num_stages: int, has_loss_and_backward: bool = False, loss_spec: Any = None) -> None:
        super().__init__()
        self.split_gm = split_gm
        self.executor = DetachExecutor(split_gm)
        self.num_stages = int(num_stages)
        self.has_loss_and_backward = bool(has_loss_and_backward)
        self.loss_spec = loss_spec
        self._stages = _extract_stages(split_gm, self.num_stages)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.split_gm(*args, **kwargs)

    def get_stage_module(self, stage_idx: int) -> Any:
        if stage_idx < 0 or stage_idx >= self.num_stages:
            raise ValueError("stage index is outside the pipeline")
        return self._stages[stage_idx]

    @staticmethod
    def _number_and_count_forward_stages(gm: Any) -> int:
        return len(_extract_stages(gm, 0))

    @staticmethod
    def _from_traced(mod: Any, exported_program: Any, multi_use_param_spec: Any = None, output_loss_value_spec: Any = None, split_policy: Any = None) -> "Pipe":
        del multi_use_param_spec
        graph_module = exported_program.module() if hasattr(exported_program, "module") else mod
        if split_policy is not None:
            graph_module = split_policy(graph_module)
        return Pipe(graph_module, max(1, len(_extract_stages(graph_module, 0))), output_loss_value_spec is not None, output_loss_value_spec)

    def print_readable(self, print_output: bool = True) -> str:
        value = repr(self.split_gm)
        if print_output:
            print(value)
        return value

    @staticmethod
    def _trace_with_export(mod: Any, example_args: tuple[Any, ...], example_kwargs: dict[str, Any]) -> Any:
        del example_args, example_kwargs
        return mod

    @staticmethod
    def from_tracing(mod: Any, example_args: tuple[Any, ...], example_kwargs: dict[str, Any] | None = None, split_policy: Any = None) -> "Pipe":
        traced = Pipe._trace_with_export(mod, example_args, example_kwargs or {})
        if split_policy is not None:
            traced = split_policy(traced)
        stages = max(1, len(_extract_stages(traced, 0)))
        return Pipe(traced, stages)

    def info(self) -> PipeInfo:
        return PipeInfo(self.split_gm, self.num_stages, self.has_loss_and_backward, self.loss_spec)

    def build_stage(self, stage_index: int, device: Any = None, group: Any = None) -> Any:
        return build_stage(self.get_stage_module(stage_index), stage_index, self.info(), device, group)

    def __str__(self) -> str:
        return f"Pipe(num_stages={self.num_stages})"

    __repr__ = __str__


class SplitPoint(Enum):
    BEGINNING = auto()
    END = auto()


class PipeSplitWrapper(Module):
    def __init__(self, module: Module, split_point: SplitPoint) -> None:
        super().__init__()
        self.module = module
        self.split_point = split_point

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def _split_before_forward(self) -> None:
        return None

    def _split_after_forward(self) -> None:
        return None


def _split_before_forward(self: Any) -> None:
    del self


def _split_after_forward(self: Any) -> None:
    del self


def annotate_split_points(mod: Any, spec: dict[str, SplitPoint]) -> Any:
    for name, point in spec.items():
        parent, _, child_name = name.rpartition(".")
        owner = mod.get_submodule(parent) if parent else mod
        child = getattr(owner, child_name)
        setattr(owner, child_name, PipeSplitWrapper(child, point))
    return mod


def pipeline(module: Any, mb_args: tuple[Any, ...], mb_kwargs: dict[str, Any] | None = None, split_spec: dict[str, SplitPoint] | None = None, split_policy: Any = None) -> Pipe:
    if split_spec:
        module = annotate_split_points(module, split_spec)
    if split_policy is not None:
        module = split_policy(module)
    return Pipe.from_tracing(module, mb_args, mb_kwargs or {})


def _extract_stages(module: Any, requested: int) -> list[Any]:
    if isinstance(module, PipeSequential):
        return list(module)
    if isinstance(module, Sequential):
        return [module]
    children = list(module.named_children()) if hasattr(module, "named_children") else []
    if children and all(name.startswith("submod_") for name, _ in children):
        return [child for _, child in sorted(children)]
    return [module] * max(1, requested or 1)
