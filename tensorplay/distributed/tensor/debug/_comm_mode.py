from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from typing import Any

import tensorplay as tp
from tensorplay import overrides
from tensorplay.distributed import _functional_collectives as funcol
from tensorplay.distributed._tools.mod_tracker import ModTracker
from tensorplay.nn.modules import module as module_api

__all__ = ["CommDebugMode"]


_COLLECTIVE_NAMES = frozenset(
    {
        "all_reduce",
        "all_reduce_coalesced",
        "all_gather_single",
        "all_gather_single_coalesced",
        "reduce_scatter_single",
        "reduce_scatter_single_coalesced",
        "all_to_all_single",
        "broadcast",
        "permute_tensor",
    }
)
_TRIVIAL_NAMES = frozenset(
    {"detach", "t", "view", "_to_copy", "as_strided", "transpose"}
)


def _walk(value: Any) -> Iterable[Any]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _operation_name(operation: Any) -> str:
    name = getattr(operation, "__name__", None)
    if isinstance(name, str):
        return name
    name = getattr(operation, "__qualname__", None)
    if isinstance(name, str):
        return name.rsplit(".", 1)[-1]
    return str(operation).rsplit(".", 1)[-1]


class _CommModeModuleTracker(ModTracker):
    def __init__(self) -> None:
        super().__init__()
        self.module_helper_dict: dict[str, dict[str, Any]] = {}
        self.module_parameters_dict: dict[str, dict[str, Any]] = {}
        self.module_parents_dict: dict[str, set[str]] = {}
        self.parent_dict: dict[str, list[str]] = {}
        self.parent_list: list[str] = []
        self.sharding_dict: dict[str, Any] = {}
        self.activation_checkpointing = False
        self.name = "Global"

    def _fw_pre_hook(self, mod: Any, inputs: Any) -> None:
        super()._fw_pre_hook(mod, inputs)
        self.activation_checkpointing = self.is_bw
        self.name = self._get_mod_name(mod)
        if self.activation_checkpointing:
            return
        helper = self.module_helper_dict.setdefault(self.name, {})
        helper["module_type"] = type(mod).__name__
        helper["depth"] = max(0, len(self.parent_list) - 1)
        parameter_info = self.module_parameters_dict.setdefault(self.name, {})
        for parameter_name, parameter in mod.named_parameters(recurse=False):
            parameter_info[parameter_name] = getattr(parameter, "data", parameter)
            value = getattr(parameter, "data", parameter)
            if hasattr(value, "placements"):
                key = f"{self.name}.{parameter_name}"
                self.sharding_dict[key] = value.placements
                helper.setdefault("parameters", {})[parameter_name] = str(
                    value.placements
                )
        self.module_parents_dict.setdefault(self.name, set(self.parents))
        parent = self.parent_list[-1] if self.parent_list else "Global"
        self.parent_dict.setdefault(parent, []).append(self.name)
        self.parent_list.append(self.name)

    def _fw_post_hook(self, mod: Any, inputs: Any, output: Any) -> None:
        super()._fw_post_hook(mod, inputs, output)
        if not self.activation_checkpointing and self.parent_list:
            if self.parent_list[-1] == self._get_mod_name(mod):
                self.parent_list.pop()
            self.name = self.parent_list[-1] if self.parent_list else "Global"

    def _bw_hook(self, mod: Any, grad_output: Any) -> None:
        del grad_output
        self.activation_checkpointing = False
        self._is_bw = True
        self.name = self._get_mod_name(mod)

    def __enter__(self) -> "_CommModeModuleTracker":
        self.module_parameters_dict.clear()
        self.sharding_dict.clear()
        self.parent_dict.clear()
        self.parent_list = ["Global"]
        self.module_helper_dict.clear()
        self.module_helper_dict["Global"] = {"depth": 0}
        self.module_parents_dict.clear()
        self.module_parents_dict["Global"] = set()
        self.name = "Global"
        super().__enter__()
        self._bw_handle = module_api.register_module_full_backward_pre_hook(
            self._bw_hook
        )
        return self

    def __exit__(self, *args: Any) -> None:
        self._bw_handle.remove()
        super().__exit__(*args)

    def print_paramater_info(self) -> None:
        print(self.module_parameters_dict)

    def print_sharding_info(self) -> None:
        for key, value in self.sharding_dict.items():
            print(f"{key}: {value}")


class CommDebugMode(overrides.TensorPlayFunctionMode):
    def __init__(self) -> None:
        self.comm_counts: Counter[Any] = Counter()
        self.operation_counts: Counter[str] = Counter()
        self.comm_module_counts: dict[str, dict[str, Counter[Any]]] = {}
        self.comm_module_operation_counts: dict[str, dict[str, Any]] = {}
        self.advanced_module_tracker = _CommModeModuleTracker()
        self._patched_collectives: dict[str, Callable[..., Any]] = {}
        self._active = False

    @property
    def _module_name(self) -> str:
        return self.advanced_module_tracker.name

    def _ensure_module_counts(self, module_name: str) -> None:
        if module_name not in self.comm_module_counts:
            self.comm_module_counts[module_name] = {
                "forward": Counter(),
                "backward": Counter(),
            }

    def _record_collective_impl(
        self, name: Any, count: int = 1, module_name: str | None = None
    ) -> None:
        module_name = module_name or self._module_name
        phase = "backward" if self.advanced_module_tracker.is_bw else "forward"
        self.comm_counts[name] += count
        self._ensure_module_counts(module_name)
        self.comm_module_counts[module_name][phase][name] += count
        for parent in self.advanced_module_tracker.module_parents_dict.get(
            module_name, ()
        ):
            self._ensure_module_counts(parent)
            self.comm_module_counts[parent][phase][name] += count

    def record_collective(self, name: str, count: int = 1) -> None:
        self._record_collective_impl(name, count)

    def _tensor_inputs(self, args: Any, kwargs: Any) -> list[Any]:
        return [
            value
            for value in _walk((args, kwargs))
            if isinstance(value, tp.Tensor)
        ]

    def _record_operation_impl(
        self, operation: Any, args: Any, kwargs: Any
    ) -> dict[str, Any]:
        tensors = self._tensor_inputs(args, kwargs)
        distributed = [
            value
            for value in _walk((args, kwargs))
            if hasattr(value, "placements")
        ]
        record = {
            "name": operation,
            "input_shape": [tuple(value.shape) for value in distributed],
            "input_sharding": [value.placements for value in distributed],
            "device_mesh": str(distributed[0].device_mesh) if distributed else "",
            "is_bw": self.advanced_module_tracker.is_bw,
            "is_activation_checkpointing": self.advanced_module_tracker.activation_checkpointing,
        }
        name = _operation_name(operation)
        self.operation_counts[name] += 1
        module_name = self._module_name
        self.comm_module_operation_counts.setdefault(
            module_name, {"operations_list": []}
        )["operations_list"].append(record)
        if not distributed and not tensors:
            record["input_shape"] = []
        return record

    def record_operation(self, name: str, count: int = 1) -> None:
        self.operation_counts[name] += count

    def _patch_collectives(self) -> None:
        for name in _COLLECTIVE_NAMES:
            original = getattr(funcol, name, None)
            if original is None or name in self._patched_collectives:
                continue

            def counted(
                *args: Any,
                _name: str = name,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                self._record_collective_impl(_name)
                return _original(*args, **kwargs)

            self._patched_collectives[name] = original
            setattr(funcol, name, counted)

    def _restore_collectives(self) -> None:
        for name, original in self._patched_collectives.items():
            setattr(funcol, name, original)
        self._patched_collectives.clear()

    def _get_operations_list(
        self, module_operation_counts: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        operations = module_operation_counts.get("operations_list", [])
        return (
            [operation for operation in operations if not operation["is_bw"]],
            [
                operation
                for operation in operations
                if operation["is_bw"] and not operation["is_activation_checkpointing"]
            ],
            [
                operation
                for operation in operations
                if operation["is_activation_checkpointing"]
            ],
        )

    def _set_noise_parameters(
        self, noise_level: int
    ) -> tuple[bool, bool, bool, bool]:
        level = int(noise_level)
        return level > 0, level > 0, level > 1, level > 2

    def generate_json_dump(
        self, file_name: str = "comm_mode_log.json", noise_level: int = 3
    ) -> dict[str, Any]:
        include_dtensor, include_modules, include_ops, include_trivial = (
            self._set_noise_parameters(noise_level)
        )

        def build(module_name: str) -> dict[str, Any]:
            helper = self.advanced_module_tracker.module_helper_dict.get(
                module_name, {"depth": 0}
            )
            result: dict[str, Any] = {
                "fqn": module_name,
                "module_type": helper.get("module_type", "")
                if include_modules
                else "",
                "parameters": list(helper.get("parameters", {}).items())
                if include_modules
                else [],
                "children": [],
                "collectives_forward": [],
                "collectives_backward": [],
                "operations_forward": [],
                "operations_backward": [],
                "operations_checkpointing": [],
            }
            module_counts = self.comm_module_counts.get(module_name, {})
            result["collectives_forward"] = list(
                module_counts.get("forward", {}).items()
            )
            result["collectives_backward"] = list(
                module_counts.get("backward", {}).items()
            )
            if include_dtensor:
                operation_groups = self._get_operations_list(
                    self.comm_module_operation_counts.get(module_name, {})
                )
                for key, operations in zip(
                    (
                        "operations_forward",
                        "operations_backward",
                        "operations_checkpointing",
                    ),
                    operation_groups,
                ):
                    selected = operations
                    if not include_ops:
                        selected = [
                            operation
                            for operation in selected
                            if operation["input_sharding"]
                        ]
                    if not include_trivial:
                        selected = [
                            operation
                            for operation in selected
                            if _operation_name(operation["name"])
                            not in _TRIVIAL_NAMES
                        ]
                    copied = copy.deepcopy(selected)
                    for operation in copied:
                        operation["name"] = _operation_name(operation["name"])
                        operation["input_sharding"] = [
                            str(value) for value in operation["input_sharding"]
                        ]
                        operation["input_shape"] = [
                            str(value) for value in operation["input_shape"]
                        ]
                    result[key] = copied
            for child in self.advanced_module_tracker.parent_dict.get(
                module_name, []
            ):
                result["children"].append(build(child))
            return result

        data = build("Global")
        with open(file_name, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=4, default=str)
        return data

    def generate_comm_debug_tracing_table(self, noise_level: int = 3) -> str:
        include_dtensor, include_modules, include_ops, include_trivial = (
            self._set_noise_parameters(noise_level)
        )
        lines: list[str] = []
        for module_name, helper in self.advanced_module_tracker.module_helper_dict.items():
            indent = "  " * (2 * int(helper.get("depth", 0)))
            lines.append(f"{indent}{module_name}")
            if include_modules and helper.get("module_type"):
                lines.append(f"{indent}*module type: {helper['module_type']}")
            if include_modules and helper.get("parameters"):
                lines.append(f"{indent}*Parameter List")
                lines.extend(
                    f"{indent} *{name}: {placement}"
                    for name, placement in helper["parameters"].items()
                )
            counts = self.comm_module_counts.get(module_name, {})
            for phase in ("forward", "backward"):
                entries = counts.get(phase, {})
                operations: list[dict[str, Any]] = []
                if include_dtensor:
                    groups = self._get_operations_list(
                        self.comm_module_operation_counts.get(module_name, {})
                    )
                    operations = groups[0 if phase == "forward" else 1]
                    if not include_ops:
                        operations = [
                            item for item in operations if item["input_sharding"]
                        ]
                    if not include_trivial:
                        operations = [
                            item
                            for item in operations
                            if _operation_name(item["name"])
                            not in _TRIVIAL_NAMES
                        ]
                if not entries and not operations:
                    continue
                lines.append(f"{indent}  {phase.upper()} PASS")
                lines.extend(
                    f"{indent}    *{name}: {count}"
                    for name, count in entries.items()
                )
                for operation in operations:
                    lines.append(f"{indent}    **{_operation_name(operation['name'])}")
                    if operation["input_shape"]:
                        lines.append(f"{indent}      shape: {operation['input_shape']}")
                        lines.append(
                            f"{indent}      sharding: {operation['input_sharding']}"
                        )
                        lines.append(
                            f"{indent}      device mesh: {operation['device_mesh']}"
                        )
        return "\n".join(lines) + ("\n" if lines else "")

    def get_total_counts(self) -> int:
        return sum(self.comm_counts.values())

    def get_comm_counts(self) -> dict[Any, int]:
        return dict(self.comm_counts)

    def get_operation_counts(self) -> dict[str, int]:
        return dict(self.operation_counts)

    def get_parameter_info(self) -> dict[str, dict[str, Any]]:
        return self.advanced_module_tracker.module_parameters_dict

    def get_sharding_info(self) -> dict[str, Any]:
        return self.advanced_module_tracker.sharding_dict

    def log_comm_debug_tracing_table_to_file(
        self, file_name: str = "comm_mode_log.txt", noise_level: int = 3
    ) -> None:
        table = re.sub(
            r"\x1B\[[0-?]*[ -/]*[@-~]",
            "",
            self.generate_comm_debug_tracing_table(noise_level),
        )
        with open(file_name, "w", encoding="utf-8") as stream:
            stream.write(table)

    def __tensorplay_function__(
        self,
        func: Callable[..., Any],
        types_: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types_
        kwargs = kwargs or {}
        self._record_operation_impl(func, args, kwargs)
        if any(
            hasattr(value, "placements") for value in _walk((args, kwargs))
        ):
            return NotImplemented
        return func(*args, **kwargs)

    def __enter__(self) -> "CommDebugMode":
        self.comm_counts.clear()
        self.operation_counts.clear()
        self.comm_module_counts.clear()
        self.comm_module_operation_counts.clear()
        self._ensure_module_counts("Global")
        self.advanced_module_tracker.__enter__()
        self._patch_collectives()
        self._active = True
        return super().__enter__()

    def __exit__(self, *args: Any) -> None:
        self._active = False
        self._restore_collectives()
        self.advanced_module_tracker.__exit__(*args)
        super().__exit__(*args)

    def __repr__(self) -> str:
        return f"CommDebugMode(get_total_counts()={self.get_total_counts()})"
