"""Weight normalization as a forward pre-hook.

The parameter is stored as a direction ``weight_v`` and a magnitude
``weight_g``; before every forward pass the hook recomputes

    weight = weight_v * weight_g / ||weight_v||

with the norm taken over every axis except ``dim``.  Both halves stay ordinary
parameters, so the optimizer updates them directly.
"""

from typing import Any, TypeVar

import tensorplay as tp
from tensorplay.functional import _weight_norm, norm_except_dim
from tensorplay.nn.parameter import Parameter, UninitializedParameter

__all__ = ["WeightNorm", "weight_norm", "remove_weight_norm"]

_Module = TypeVar("_Module")


class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module) -> Any:
        v = getattr(module, self.name + "_v")
        g = getattr(module, self.name + "_g")
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int) -> "WeightNorm":
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two weight_norm hooks on the same "
                    f"parameter {name}"
                )
        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)
        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                "The module passed to `WeightNorm` can't have uninitialized "
                "parameters. Make sure to run the dummy forward before applying "
                "weight normalization"
            )

        # The plain parameter is replaced by the pair; the recomputed weight
        # lives as a plain attribute so it is never handed to the optimizer.
        del module._parameters[name]
        module.register_parameter(
            name + "_g",
            Parameter(norm_except_dim(weight, 2, dim).detach().clone()),
        )
        module.register_parameter(name + "_v", Parameter(weight.detach().clone()))
        setattr(module, name, fn.compute_weight(module))
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        module.register_parameter(self.name, Parameter(weight.detach()))

    def __call__(self, module, inputs) -> None:
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module: _Module, name: str = "weight", dim: int = 0) -> _Module:
    """Split ``module.<name>`` into a direction and a per-slice magnitude."""
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module: _Module, name: str = "weight") -> _Module:
    """Undo :func:`weight_norm`, restoring the plain parameter."""
    for key, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[key]
            return module
    raise ValueError(f"weight_norm of '{name}' not found in {module}")
