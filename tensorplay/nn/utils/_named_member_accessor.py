"""Dotted-path access to a module's parameters, buffers and submodules.

Reparameterizing a module functionally means swapping tensors in and out by
their state-dict names (``"layer1.conv1.weight"``).  Resolving that path on
every access is wasteful when the same module is called many times, so the
accessor caches the intermediate submodule lookups.
"""
from typing import Any, Iterable

import tensorplay

__all__ = ["NamedMemberAccessor", "set_tensor", "swap_submodule", "swap_tensor"]

#: Sentinel for "this name had no tensor bound to it".
_MISSING: Any = object()


def _as_module(module: Any) -> Any:
    from tensorplay.nn import Module

    if not isinstance(module, Module):
        raise TypeError(f"{module} is not an instance of tensorplay.nn.Module")
    return module


def _check_name(name: str, what: str = "tensor") -> None:
    if "." in name:
        raise KeyError(f'{what} name can\'t contain "."')
    if name == "":
        raise KeyError(f'{what} name can\'t be empty string ""')


def set_tensor(module: Any, name: str, tensor: Any) -> None:
    """Binds ``tensor`` to the direct attribute ``name`` of ``module``."""
    _as_module(module)
    if not isinstance(tensor, tensorplay.Tensor) and tensor is not None:
        raise TypeError(f"{tensor} is not an instance of tensorplay.Tensor")
    _check_name(name)
    if name in module._parameters:
        module._parameters[name] = tensor
    elif name in module._buffers:
        module._buffers[name] = tensor
    else:
        setattr(module, name, tensor)


def swap_tensor(module: Any, name: str, tensor: Any, allow_missing: bool = False) -> Any:
    """Binds ``tensor`` to ``name`` and returns whatever was bound before.

    ``tensor`` may be :data:`_MISSING`, which deletes the binding instead; a
    name with nothing bound returns :data:`_MISSING` when ``allow_missing``.
    """
    _as_module(module)
    if (
        tensor is not _MISSING
        and not isinstance(tensor, tensorplay.Tensor)
        and tensor is not None
    ):
        raise TypeError(f"{tensor} is not an instance of tensorplay.Tensor")
    _check_name(name)

    if name in module._parameters:
        orig_tensor = module._parameters[name]
        if tensor is not _MISSING:
            module._parameters[name] = tensor
        else:
            del module._parameters[name]
    elif name in module._buffers:
        orig_tensor = module._buffers[name]
        if tensor is not _MISSING:
            module._buffers[name] = tensor
        else:
            del module._buffers[name]
    else:
        if hasattr(module, name):
            orig_tensor = getattr(module, name)
        else:
            if not allow_missing:
                raise AttributeError(f"{module._get_name()} has no attribute `{name}`")
            orig_tensor = _MISSING
        if (
            orig_tensor is not _MISSING
            and not isinstance(orig_tensor, tensorplay.Tensor)
            and orig_tensor is not None
        ):
            raise TypeError(
                f"attribute `{name}`: {orig_tensor} is not an instance of tensorplay.Tensor"
            )
        if tensor is not _MISSING:
            setattr(module, name, tensor)
        elif hasattr(module, name):
            delattr(module, name)
    return orig_tensor


def swap_submodule(module: Any, name: str, submodule: Any) -> Any:
    """Replaces the direct child module ``name``, returning the previous one."""
    _as_module(module)
    _as_module(submodule)
    _check_name(name, "submodule")
    if name not in module._modules:
        raise KeyError(f"submodule {name} does not exist")
    orig_submodule = module._modules[name]
    module._modules[name] = submodule
    return orig_submodule


class NamedMemberAccessor:
    """Reads, writes and swaps a module's members by dotted path.

    Intermediate submodule lookups are memoized, so repeatedly swapping the
    same set of names -- the hot path of a functional call -- walks the module
    tree once instead of once per access.
    """

    def __init__(self, module: Any) -> None:
        self.module = _as_module(module)
        self.memo: dict[str, Any] = {}

    # Nested attribute access

    def get_submodule(self, name: str) -> Any:
        """Returns the submodule at ``name``, e.g. ``"layer1.conv1"``."""
        if not name:
            return self.module
        if name in self.memo:
            return self.memo[name]
        prefix, dot, attr = name.rpartition(".")
        module = self.get_submodule(prefix) if dot else self.module
        try:
            submodule = getattr(module, attr)
        except AttributeError as ex:
            raise AttributeError(f"{module._get_name()} has no attribute `{attr}`") from ex
        _as_module(submodule)
        self.memo[name] = submodule
        return submodule

    def swap_submodule(self, path: str, value: Any) -> Any:
        """Swaps the submodule at ``path``, returning the previous one."""
        prefix, _, attr = path.rpartition(".")
        return swap_submodule(self.get_submodule(prefix), attr, value)

    def get_tensor(self, name: str) -> Any:
        """Returns the tensor at ``name``, e.g. ``"layer1.conv1.weight"``."""
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            tensor = getattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(f"{submodule._get_name()} has no attribute `{name}`") from ex
        if not isinstance(tensor, tensorplay.Tensor) and tensor is not None:
            raise TypeError(f"{tensor} is not an instance of tensorplay.Tensor")
        return tensor

    def set_tensor(self, name: str, value: Any) -> None:
        """Binds ``value`` at ``name``."""
        prefix, _, attr = name.rpartition(".")
        set_tensor(self.get_submodule(prefix), attr, value)

    def del_tensor(self, name: str) -> None:
        """Deletes the binding at ``name``."""
        prefix, _, attr = name.rpartition(".")
        submodule = self.get_submodule(prefix)
        try:
            delattr(submodule, attr)
        except AttributeError as ex:
            raise AttributeError(f"{submodule._get_name()} has no attribute `{name}`") from ex

    def swap_tensor(self, name: str, value: Any, allow_missing: bool = False) -> Any:
        """Binds ``value`` at ``name``, returning the previous binding."""
        prefix, _, attr = name.rpartition(".")
        return swap_tensor(
            self.get_submodule(prefix), attr, value, allow_missing=allow_missing
        )

    # Batched operations

    def get_tensors(self, names: Iterable[str]) -> list[Any]:
        """Returns the tensors at each of ``names``."""
        return [self.get_tensor(name) for name in names]

    def set_tensors(self, names: Iterable[str], values: Iterable[Any]) -> None:
        """Binds each of ``values`` at the matching name."""
        names = list(names)
        values = list(values)
        if len(names) != len(values):
            raise ValueError(
                f"names and values must have the same length, "
                f"got {len(names)} names and {len(values)} values"
            )
        for name, value in zip(names, values):
            self.set_tensor(name, value)

    def set_tensors_dict(self, named_tensors: dict[str, Any]) -> None:
        """Binds every entry of ``named_tensors`` at its key."""
        for name, value in named_tensors.items():
            self.set_tensor(name, value)

    def del_tensors(self, names: Iterable[str]) -> None:
        """Deletes the binding at each of ``names``."""
        for name in names:
            self.del_tensor(name)

    def swap_tensors(
        self, names: Iterable[str], values: Iterable[Any], allow_missing: bool = False
    ) -> list[Any]:
        """Swaps each of ``values`` in, returning the previous bindings."""
        names = list(names)
        values = list(values)
        if len(names) != len(values):
            raise ValueError(
                f"names and values must have the same length, "
                f"got {len(names)} names and {len(values)} values"
            )
        return [
            self.swap_tensor(name, value, allow_missing=allow_missing)
            for name, value in zip(names, values)
        ]

    def swap_tensors_dict(
        self, named_tensors: dict[str, Any], allow_missing: bool = False
    ) -> tuple[dict[str, Any], list[str]]:
        """Swaps in a whole name->tensor map.

        Returns the previous bindings plus the names that had none.  Any
        failure -- an exception, or a missing name when ``allow_missing`` is
        false -- rolls every swap back before propagating, so the module is
        never left half-reparameterized.
        """
        orig_named_tensors: dict[str, Any] = {}
        missing_keys: list[str] = []
        try:
            for name, tensor in named_tensors.items():
                orig_tensor = self.swap_tensor(name, tensor, allow_missing=True)
                if orig_tensor is _MISSING:
                    missing_keys.append(name)
                orig_named_tensors[name] = orig_tensor
        except Exception:
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise
        if missing_keys and not allow_missing:
            for name, orig_tensor in orig_named_tensors.items():
                self.swap_tensor(name, orig_tensor, allow_missing=True)
            raise RuntimeError(f"Missing key(s): {', '.join(map(repr, missing_keys))}.")
        return orig_named_tensors, missing_keys

    def check_keys(self, keys: Iterable[str]) -> tuple[list[str], list[str]]:
        """Splits ``keys`` against the module's members: (missing, unexpected)."""
        keys = set(keys)
        valid_keys = {name for name, _ in self.named_tensors(remove_duplicate=False)}
        return sorted(valid_keys - keys), sorted(keys - valid_keys)

    # Shortcut methods

    def named_parameters(self, remove_duplicate: bool = True):
        """Iterates over the module's parameters."""
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)

    def named_buffers(self, remove_duplicate: bool = True):
        """Iterates over the module's buffers."""
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_tensors(self, remove_duplicate: bool = True):
        """Iterates over the module's parameters and buffers."""
        yield from self.module.named_parameters(remove_duplicate=remove_duplicate)
        yield from self.module.named_buffers(remove_duplicate=remove_duplicate)

    def named_modules(self, remove_duplicate: bool = True):
        """Iterates over the module and its submodules."""
        yield from self.module.named_modules(remove_duplicate=remove_duplicate)
