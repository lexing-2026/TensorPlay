"""Composable module preparation and tensor-parallel styles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Sequence

from .._api import DTensor, distribute_module, distribute_tensor
from ..placement_types import Placement, Replicate, Shard

__all__ = [
    "ParallelStyle",
    "RowwiseParallel",
    "SequenceParallel",
    "ColwiseParallel",
    "PrepareModuleInput",
    "PrepareModuleInputOutput",
    "PrepareModuleOutput",
]


def _layout_tuple(value: Placement | Sequence[Placement | None] | None) -> tuple[Placement | None, ...] | None:
    if value is None:
        return None
    return (value,) if isinstance(value, Placement) else tuple(value)


def _single_layout(value: Placement | Sequence[Placement | None], mesh: Any) -> tuple[Placement | None, ...]:
    result = _layout_tuple(value)
    if result is None:
        raise ValueError("a layout is required")
    if len(result) == 1 and mesh.ndim() > 1:
        return result + tuple(Replicate() for _ in range(mesh.ndim() - 1))
    if len(result) != mesh.ndim():
        raise ValueError("layout count must equal mesh rank")
    return result


def _as_dtensor(value: Any, mesh: Any, layouts: Sequence[Placement | None]) -> Any:
    if value is None or isinstance(value, DTensor):
        return value
    if all(layout is None for layout in layouts):
        return value
    concrete = [layout if layout is not None else Replicate() for layout in layouts]
    return DTensor.from_local(value, mesh, concrete, run_check=False)


def _redistribute(value: Any, desired: Sequence[Placement | None]) -> Any:
    if not isinstance(value, DTensor) or all(layout is None for layout in desired):
        return value
    target = tuple(layout if layout is not None else current for layout, current in zip(desired, value.placements))
    if target != value.placements:
        return value.redistribute(placements=target)
    return value


class ParallelStyle(ABC):
    src_data_rank: int | None = 0

    @abstractmethod
    def _apply(self, module: Any, device_mesh: Any) -> Any:
        raise NotImplementedError


class ColwiseParallel(ParallelStyle):
    def __init__(self, *, input_layouts: Placement | None = None, output_layouts: Placement | None = None, use_local_output: bool = True) -> None:
        self.input_layouts = (input_layouts or Replicate(),)
        self.output_layouts = (output_layouts or Shard(-1),)
        self.desired_input_layouts = (Replicate(),)
        self.use_local_output = use_local_output

    def _partition_linear(self, name: str, module: Any, mesh: Any) -> None:
        del name
        if not hasattr(module, "weight"):
            return
        module._parameters["weight"] = distribute_tensor(module.weight, mesh, _single_layout(Shard(0), mesh), src_data_rank=self.src_data_rank)
        if getattr(module, "bias", None) is not None:
            module._parameters["bias"] = distribute_tensor(module.bias, mesh, _single_layout(Shard(0), mesh), src_data_rank=self.src_data_rank)

    def _partition_embedding(self, name: str, module: Any, mesh: Any) -> None:
        del name
        module._parameters["weight"] = distribute_tensor(module.weight, mesh, _single_layout(Shard(1), mesh), src_data_rank=self.src_data_rank)

    @staticmethod
    def _prepare_input(layouts: Any, desired: Any, module: Any, inputs: tuple[Any, ...], mesh: Any) -> tuple[Any, ...]:
        del module
        if not inputs:
            return inputs
        value = _as_dtensor(inputs[0], mesh, _single_layout(layouts, mesh))
        value = _redistribute(value, _single_layout(desired, mesh))
        return (value,) + inputs[1:]

    def _prepare_output(self, module: Any, output: Any, mesh: Any) -> Any:
        del module
        if not isinstance(output, DTensor):
            return output
        desired = _single_layout(self.output_layouts, mesh)
        output = _redistribute(output, desired)
        return output.to_local() if self.use_local_output else output

    def _apply(self, module: Any, device_mesh: Any) -> Any:
        linear = isinstance(module, getattr(__import__("tensorplay.nn", fromlist=["Linear"]), "Linear", ()))
        embedding = isinstance(module, getattr(__import__("tensorplay.nn", fromlist=["Embedding"]), "Embedding", ()))
        if linear:
            partition = self._partition_linear
        elif embedding:
            partition = self._partition_embedding
        else:
            raise NotImplementedError("ColwiseParallel supports Linear and Embedding modules")
        return distribute_module(
            module,
            device_mesh,
            partition,
            partial(self._prepare_input, self.input_layouts, self.desired_input_layouts),
            self._prepare_output,
        )

    def __repr__(self) -> str:
        return f"ColwiseParallel(input_layouts={self.input_layouts}, output_layouts={self.output_layouts}, use_local_output={self.use_local_output})"


class RowwiseParallel(ParallelStyle):
    def __init__(self, *, input_layouts: Placement | None = None, output_layouts: Placement | None = None, use_local_output: bool = True) -> None:
        self.input_layouts = (input_layouts or Shard(-1),)
        self.output_layouts = (output_layouts or Replicate(),)
        self.use_local_output = use_local_output
        self.desired_input_layouts = self.input_layouts

    def _partition_linear(self, name: str, module: Any, mesh: Any) -> None:
        del name
        module._parameters["weight"] = distribute_tensor(module.weight, mesh, _single_layout(Shard(1), mesh), src_data_rank=self.src_data_rank)
        if getattr(module, "bias", None) is not None:
            module._parameters["bias"] = distribute_tensor(module.bias, mesh, _single_layout(Replicate(), mesh), src_data_rank=self.src_data_rank)

    def _partition_embedding(self, name: str, module: Any, mesh: Any) -> None:
        del name
        module._parameters["weight"] = distribute_tensor(module.weight, mesh, _single_layout(Shard(0), mesh), src_data_rank=self.src_data_rank)

    def _apply(self, module: Any, device_mesh: Any) -> Any:
        nn = __import__("tensorplay.nn", fromlist=["Linear", "Embedding"])
        if isinstance(module, nn.Linear):
            partition = self._partition_linear
            self.desired_input_layouts = (Shard(-1),)
        elif isinstance(module, nn.Embedding):
            partition = self._partition_embedding
            self.desired_input_layouts = (Replicate(),)
        else:
            raise NotImplementedError(
                "RowwiseParallel supports Linear and Embedding modules"
            )
        return distribute_module(
            module,
            device_mesh,
            partition,
            self._prepare_input,
            self._prepare_output,
        )

    def _prepare_input(self, module: Any, inputs: tuple[Any, ...], mesh: Any) -> tuple[Any, ...]:
        del module
        value = _as_dtensor(inputs[0], mesh, _single_layout(self.input_layouts, mesh)) if inputs else None
        value = _redistribute(value, _single_layout(self.desired_input_layouts, mesh))
        return (value,) + inputs[1:] if inputs else inputs

    def _prepare_output(self, module: Any, output: Any, mesh: Any) -> Any:
        del module
        output = _redistribute(output, _single_layout(self.output_layouts, mesh))
        return output.to_local() if self.use_local_output and isinstance(output, DTensor) else output

    def __repr__(self) -> str:
        return f"RowwiseParallel(input_layouts={self.input_layouts}, output_layouts={self.output_layouts}, use_local_output={self.use_local_output})"


class SequenceParallel(ParallelStyle):
    def __init__(self, *, sequence_dim: int = 1, use_local_output: bool = False) -> None:
        self.sequence_sharding = (Shard(sequence_dim),)
        self.use_local_output = use_local_output

    def _replicate(self, name: str, module: Any, mesh: Any) -> None:
        del name
        for param_name, param in list(module._parameters.items()):
            if param is not None:
                module._parameters[param_name] = distribute_tensor(param, mesh, _single_layout(Replicate(), mesh))

    def _input(self, module: Any, inputs: tuple[Any, ...], mesh: Any) -> tuple[Any, ...]:
        del module
        if not inputs:
            return inputs
        value = _as_dtensor(inputs[0], mesh, _single_layout(self.sequence_sharding, mesh))
        return (value,) + inputs[1:]

    def _output(self, module: Any, output: Any, mesh: Any) -> Any:
        del module, mesh
        return output.to_local() if self.use_local_output and isinstance(output, DTensor) else output

    def _apply(self, module: Any, device_mesh: Any) -> Any:
        return distribute_module(module, device_mesh, self._replicate, self._input, self._output)

    def __repr__(self) -> str:
        return f"SequenceParallel(sequence_dim={self.sequence_sharding[0].dim}, use_local_output={self.use_local_output})"


class PrepareModuleInput(ParallelStyle):
    def __init__(self, *, input_layouts: Placement | tuple[Placement | None, ...] | None = None, desired_input_layouts: Placement | tuple[Placement | None, ...] | None = None, input_kwarg_layouts: dict[str, Placement] | None = None, desired_input_kwarg_layouts: dict[str, Placement] | None = None, use_local_output: bool = False) -> None:
        self.input_layouts = _layout_tuple(input_layouts)
        self.desired_input_layouts = _layout_tuple(desired_input_layouts)
        if (self.input_layouts is None) != (self.desired_input_layouts is None) or (self.input_layouts is not None and len(self.input_layouts) != len(self.desired_input_layouts or ())):
            raise ValueError("input_layouts and desired_input_layouts must be provided together with equal length")
        self.input_kwarg_layouts = dict(input_kwarg_layouts or {})
        self.desired_input_kwarg_layouts = dict(desired_input_kwarg_layouts or {})
        if set(self.input_kwarg_layouts) != set(self.desired_input_kwarg_layouts):
            raise ValueError("input keyword layout maps must contain equal keys")
        self.use_local_output = use_local_output

    def _prepare_one(self, value: Any, mesh: Any, layout: Placement | None, desired: Placement | None) -> Any:
        if layout is None:
            return value
        result = _as_dtensor(value, mesh, _single_layout(layout, mesh))
        result = _redistribute(result, _single_layout(desired, mesh) if desired is not None else (None,))
        return result.to_local() if self.use_local_output and isinstance(result, DTensor) else result

    def _apply(self, module: Any, device_mesh: Any) -> Any:
        def hook(current: Any, inputs: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> Any:
            values = tuple(inputs)
            if self.input_layouts is not None:
                if len(values) != len(self.input_layouts):
                    raise ValueError("module input count does not match input layouts")
                values = tuple(self._prepare_one(value, device_mesh, layout, desired) for value, layout, desired in zip(values, self.input_layouts, self.desired_input_layouts or ()))
            if kwargs is None:
                return values
            prepared = {key: self._prepare_one(kwargs[key], device_mesh, self.input_kwarg_layouts[key], self.desired_input_kwarg_layouts[key]) if key in self.input_kwarg_layouts else kwargs[key] for key in kwargs}
            return values, prepared
        module.register_forward_pre_hook(hook, with_kwargs=bool(self.input_kwarg_layouts))
        return module


class PrepareModuleOutput(ParallelStyle):
    def __init__(self, *, output_layouts: Placement | tuple[Placement | None, ...], desired_output_layouts: Placement | tuple[Placement, ...], use_local_output: bool = True) -> None:
        self.output_layouts = _layout_tuple(output_layouts) or ()
        self.desired_output_layouts = _layout_tuple(desired_output_layouts) or ()
        if len(self.output_layouts) != len(self.desired_output_layouts):
            raise ValueError("output layouts must have equal length")
        self.use_local_output = use_local_output

    def _apply(self, module: Any, device_mesh: Any) -> Any:
        def hook(current: Any, inputs: tuple[Any, ...], outputs: Any) -> Any:
            del current, inputs
            values = outputs if isinstance(outputs, tuple) else (outputs,)
            if len(values) != len(self.output_layouts):
                raise ValueError("module output count does not match output layouts")
            prepared = []
            for value, layout, desired in zip(values, self.output_layouts, self.desired_output_layouts):
                if layout is None:
                    prepared.append(value)
                    continue
                result = _as_dtensor(value, device_mesh, _single_layout(layout, device_mesh))
                result = _redistribute(result, _single_layout(desired, device_mesh))
                prepared.append(result.to_local() if self.use_local_output else result)
            return prepared[0] if len(prepared) == 1 else tuple(prepared)
        module.register_forward_hook(hook)
        return module


class PrepareModuleInputOutput(ParallelStyle):
    def __init__(self, *, input_layouts: Placement | tuple[Placement | None, ...] | None = None, desired_input_layouts: Placement | tuple[Placement | None, ...] | None = None, input_kwarg_layouts: dict[str, Placement] | None = None, desired_input_kwarg_layouts: dict[str, Placement] | None = None, use_local_input: bool = False, output_layouts: Placement | tuple[Placement | None, ...], desired_output_layouts: Placement | tuple[Placement, ...], use_local_output: bool = True) -> None:
        self.prepare_module_input = PrepareModuleInput(input_layouts=input_layouts, desired_input_layouts=desired_input_layouts, input_kwarg_layouts=input_kwarg_layouts, desired_input_kwarg_layouts=desired_input_kwarg_layouts, use_local_output=use_local_input)
        self.prepare_module_output = PrepareModuleOutput(output_layouts=output_layouts, desired_output_layouts=desired_output_layouts, use_local_output=use_local_output)

    def _apply(self, module: Any, device_mesh: Any) -> Any:
        self.prepare_module_input._apply(module, device_mesh)
        self.prepare_module_output._apply(module, device_mesh)
        return module
