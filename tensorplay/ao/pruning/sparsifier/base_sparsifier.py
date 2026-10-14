"""Base class for sparsifiers."""

import abc
import copy
from collections import defaultdict
from typing import Any

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn.utils import parametrize
from tensorplay.nn.utils.parametrize import type_before_parametrizations

from .utils import (
    FakeSparsity,
    get_arg_info_from_tensor_fqn,
    module_contains_param,
    module_to_fqn,
    swap_module,
)

__all__ = ["BaseSparsifier"]

SUPPORTED_MODULES = {nn.Linear}
KEYS_NOT_IN_STATE_DICT = ["module", "module_fqn", "tensor_name"]


class BaseSparsifier(abc.ABC):
    """Base class for all sparsifiers.

    Subclasses implement :meth:`update_mask`, which recomputes the mask for
    every configured tensor.
    """

    def __init__(self, defaults: dict[str, Any] | None = None):
        super().__init__()
        self.defaults: dict[str, Any] = defaults or {}
        self.state: dict[str, dict] = defaultdict(dict)
        self.groups: list[dict[str, Any]] = []
        self.enable_mask_update = True

    def __getstate__(self) -> dict[str, Any]:
        return {
            "defaults": self.defaults,
            "state": self.state,
            "groups": self.groups,
        }

    def __setstate__(self, state: dict[str, dict[str, Any]]) -> None:
        self.__dict__.update(state)

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, sparse_args in enumerate(self.groups):
            module = sparse_args["module"]
            format_string += "\n"
            format_string += f"\tGroup {i}\n"
            format_string += f"\t    module: {module}\n"
            for key in sorted(sparse_args.keys()):
                if key == "module":
                    continue
                format_string += f"\t    {key}: {sparse_args[key]}\n"
        format_string += ")"
        return format_string

    def state_dict(self) -> dict[str, Any]:
        groups: list[dict[str, Any]] = [
            dict(
                filter(
                    lambda key_value: key_value[0] not in KEYS_NOT_IN_STATE_DICT,
                    mg.items(),
                )
            )
            for mg in self.groups
        ]
        return {"state": self.state, "groups": groups}

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True):
        groups = copy.deepcopy(state_dict["groups"])
        states = state_dict["state"]
        for tensor_fqn, s in states.items():
            arg_info = get_arg_info_from_tensor_fqn(self.model, tensor_fqn)
            module = arg_info["module"]
            tensor_name = arg_info["tensor_name"]
            if strict and module is None:
                raise RuntimeError(f"Error loading {tensor_fqn} into the model")
            found = False
            for p in module.parametrizations[tensor_name]:
                if isinstance(p, FakeSparsity):
                    found = True
                    break
            if not found:
                p = FakeSparsity(tp.ones(getattr(module, tensor_name).shape))
                parametrize.register_parametrization(module, tensor_name, p)
            if s.get("mask", None) is not None:
                mask = s.pop("mask")
                p.mask = mask
            for mg in groups:
                if mg["tensor_fqn"] == tensor_fqn:
                    mg.update(arg_info)
        self.__setstate__({"state": states, "groups": groups})

    def make_config_from_model(
        self,
        model: nn.Module,
        SUPPORTED_MODULES: set[type[nn.Linear]] = SUPPORTED_MODULES,
    ) -> None:
        self.config = []
        stack = [model]
        while stack:
            module = stack.pop()
            for _name, child in module.named_children():
                if type(child) in SUPPORTED_MODULES:
                    module_fqn = module_to_fqn(model, child)
                    if not isinstance(module_fqn, str):
                        raise AssertionError("module_fqn must be a string")
                    self.config.append({"tensor_fqn": module_fqn + ".weight"})
                else:
                    stack.append(child)

    def prepare(self, model, config):
        """Attach mask parametrizations to the configured tensors in place."""
        self.model = model
        self.config = config
        if self.config is None:
            self.make_config_from_model(model)

        for module_config in self.config:
            if not isinstance(module_config, dict):
                raise AssertionError("config elements should be dicts")
            if not isinstance(self.defaults, dict):
                raise AssertionError("defaults must be a dict")
            local_args = copy.deepcopy(self.defaults)
            local_args.update(module_config)
            tensor_fqn = local_args.get("tensor_fqn", None)
            if tensor_fqn is None:
                raise AssertionError(
                    "tensor_fqn is a required argument in the sparsity config"
                )
            info_from_tensor_fqn = get_arg_info_from_tensor_fqn(model, tensor_fqn)
            for key in info_from_tensor_fqn:
                if key in local_args:
                    if not (
                        info_from_tensor_fqn[key] == local_args[key]
                        or (
                            key == "tensor_fqn"
                            and "." + info_from_tensor_fqn[key] == local_args[key]
                        )
                    ):
                        raise AssertionError(
                            f"Given both `{key}` and `tensor_fqn` in the config, it is expected them to agree!"
                        )
            local_args.update(info_from_tensor_fqn)
            self.groups.append(local_args)
        self._prepare()

    def _prepare(self, *args, **kwargs):
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrization = config.get("parametrization", FakeSparsity)
            mask = config.get("mask", tp.ones_like(getattr(module, tensor_name)))
            self.state[config["tensor_fqn"]]["mask"] = mask
            parametrize.register_parametrization(
                module, tensor_name, parametrization(mask)
            )

    def squash_mask(
        self,
        params_to_keep: tuple[str, ...] | None = None,
        params_to_keep_per_layer: dict[str, tuple[str, ...]] | None = None,
        *args,
        **kwargs,
    ):
        """Squash masks into the tensors and optionally record sparse params."""
        for config in self.groups:
            module = config["module"]
            tensor_name = config["tensor_name"]
            parametrize.remove_parametrizations(
                module, tensor_name, leave_parametrized=True
            )
            sparse_params = {}
            if params_to_keep is not None:
                sparse_params.update({k: config[k] for k in params_to_keep})
            if params_to_keep_per_layer is not None:
                params = params_to_keep_per_layer.get(config["module_fqn"], None)
                if params is not None:
                    sparse_params.update({k: config[k] for k in params})
            if sparse_params:
                module.sparse_params = sparse_params

    def convert(
        self,
        module: nn.Module,
        mapping: dict[type[nn.Module], type[nn.Module]] | None = None,
        inplace: bool = False,
        parameterization: type[nn.Module] = FakeSparsity,
    ):
        """Convert submodules to sparse modules using ``from_dense``."""
        if mapping is None:
            raise NotImplementedError("Need to auto generate mapping")
        if not inplace:
            module = copy.deepcopy(module)

        reassign = {}
        for name, mod in module.named_children():
            if (
                module_contains_param(mod, parameterization)
                and type_before_parametrizations(mod) in mapping
            ):
                reassign[name] = swap_module(mod, mapping)
            else:
                reassign[name] = self.convert(
                    mod, mapping=mapping, inplace=True, parameterization=parameterization
                )
        for key, value in reassign.items():
            module._modules[key] = value
        return module

    def step(self, use_path: bool = True) -> None:
        if not self.enable_mask_update:
            return
        with tp.no_grad():
            for config in self.groups:
                self.update_mask(**config)

    @abc.abstractmethod
    def update_mask(self, module: nn.Module, tensor_name: str, **kwargs):
        pass