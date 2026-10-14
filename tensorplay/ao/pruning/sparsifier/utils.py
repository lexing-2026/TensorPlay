"""Utilities for sparsifier parametrizations."""

from itertools import chain
from typing import Any

import tensorplay as tp
from tensorplay import nn
from tensorplay.nn.utils.parametrize import is_parametrized, type_before_parametrizations

__all__ = [
    "module_contains_param",
    "swap_module",
    "module_to_fqn",
    "fqn_to_module",
    "get_arg_info_from_tensor_fqn",
    "FakeSparsity",
]


def module_contains_param(module: nn.Module, parametrization: type[nn.Module]) -> bool:
    if is_parametrized(module):
        return any(
            any(isinstance(param, parametrization) for param in param_list)
            for param_list in module.parametrizations.values()
        )
    return False


def swap_module(
    mod: nn.Module, mapping: dict[type[nn.Module], type[nn.Module]]
) -> nn.Module:
    """Swap ``mod`` for its sparse counterpart using ``from_dense``."""
    if type_before_parametrizations(mod) in mapping:
        sparse_mod = mapping[type_before_parametrizations(mod)]
        new_mod = sparse_mod.from_dense(mod)
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)
        devices = {p.device for p in chain(mod.parameters(), mod.buffers())}
        if len(devices) > 1:
            raise AssertionError(
                f"swap_module only works with cpu or single-device CUDA modules, but got devices {devices}"
            )
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)
        return new_mod
    return mod


def module_to_fqn(model: nn.Module, module: nn.Module, prefix: str = "") -> str | None:
    """Return the fully qualified name of ``module`` inside ``model``."""
    if module is model:
        return ""
    for name, child in model.named_children():
        fqn = module_to_fqn(child, module, ".")
        if isinstance(fqn, str):
            return prefix + name + fqn
    return None


def fqn_to_module(model: nn.Module | None, path: str) -> nn.Module | None:
    """Resolve a dotted path against ``model``, tolerating tensor leaves."""
    if path != "":
        for name in path.split("."):
            model = getattr(model, name, None)
    return model


def get_arg_info_from_tensor_fqn(model: nn.Module, tensor_fqn: str) -> dict[str, Any]:
    """Split ``tensor_fqn`` into module fqn, module and tensor name."""
    tensor_name = tensor_fqn.rsplit(".", maxsplit=1)[-1]
    module_fqn = tensor_fqn[: -len(tensor_name) - ("." in tensor_fqn)]
    module = fqn_to_module(model, module_fqn)
    return {
        "module_fqn": module_fqn,
        "module": module,
        "tensor_name": tensor_name,
        "tensor_fqn": tensor_fqn,
    }


class FakeSparsity(nn.Module):
    """Mask parametrization that zeroes masked entries of a tensor."""

    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", mask)

    def forward(self, x):
        if self.mask.shape != x.shape:
            raise AssertionError(
                f"mask shape ({self.mask.shape}) must match x shape ({x.shape})"
            )
        return self.mask * x

    def state_dict(self, *args, **kwargs):
        # Masks are not persisted with the parametrization; the sparsifier's
        # own state dict owns them.
        return {}