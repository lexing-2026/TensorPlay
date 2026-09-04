"""Conversion hooks for composing tensor and data parallel modules."""

from __future__ import annotations

from typing import Any, Iterable

from ....nn.parameter import Parameter
from .._api import DTensor
from ._data_parallel_utils import _flatten_tensor, _unflatten_tensor

__all__ = ["pre_dp_module_transform"]


def _get_submodule_n_params(module: Any, path: str) -> tuple[Any, str]:
    if "." not in path:
        return module, path
    parent_path, parameter_name = path.rsplit(".", 1)
    return module.get_submodule(parent_path), parameter_name


def _update_module_param(
    param_list: Iterable[tuple[Any, str, Any]],
) -> None:
    for parent_module, module_path, value in param_list:
        if isinstance(value, DTensor):
            parent_module._parameters[module_path] = value
            continue
        delattr(parent_module, module_path)
        setattr(parent_module, module_path, value)


def _reconstruct_dtensor(module: Any, _input: Any) -> None:
    del _input
    param_list = []
    for name, value in module.named_parameters():
        spec = getattr(value, "_st_info", None)
        if spec is not None:
            dtensor = _unflatten_tensor(value, spec)
            param_list.append((*_get_submodule_n_params(module, name), dtensor))
    _update_module_param(param_list)


def _localize_dtensor(
    module: Any,
    *_: Any,
    ignored_params: set[Any] | None = None,
) -> None:
    ignored_params = set() if ignored_params is None else ignored_params
    param_list = []
    for name, parameter in module.named_parameters():
        if parameter in ignored_params:
            continue
        local, spec = _flatten_tensor(parameter)
        if spec is None:
            continue
        local_parameter = Parameter(
            local,
            requires_grad=bool(getattr(parameter, "requires_grad", False)),
        )
        local_parameter._st_info = spec
        param_list.append((*_get_submodule_n_params(module, name), local_parameter))
    _update_module_param(param_list)


def pre_dp_module_transform(module: Any) -> Any:
    _localize_dtensor(module, None, None)
    module.register_forward_pre_hook(_reconstruct_dtensor)
    module.register_forward_hook(_localize_dtensor)
    return module
