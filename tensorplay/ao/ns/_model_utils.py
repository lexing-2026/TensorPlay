"""Helpers for locating and replacing submodules inside a model.

These utilities operate on dotted module paths and on the module tree of
:class:`tensorplay.nn.Module`.  They support workflows that need to fetch a
submodule by name, split a path into its parent and child components, and
swap a module for a replacement that keeps the original's forward hooks.
"""

from __future__ import annotations

from typing import TypeVar

from tensorplay import nn

__all__ = ["get_module", "parent_child_names", "swap_module"]

ModT = TypeVar("ModT", bound=nn.Module)


def get_module(model: nn.Module, name: str) -> nn.Module:
    """Return the submodule of ``model`` registered under the dotted path
    ``name``.

    Args:
        model: model to search
        name: dotted path of the submodule, for example ``"fc"`` or
            ``"block.layer1.conv"``

    Return:
        The submodule registered under ``name``.

    Raises:
        KeyError: when ``name`` does not match any submodule.
    """
    return dict(model.named_modules())[name]


def parent_child_names(name: str) -> tuple[str, str]:
    """Split a dotted submodule path into its parent path and the child
    name.

    Args:
        name: dotted path of a submodule, for example ``"block.conv"``

    Return:
        A ``(parent, child)`` tuple; the parent is the empty string when
        ``name`` has no dot.
    """
    split_name = name.rsplit(".", 1)
    if len(split_name) == 1:
        return "", split_name[0]
    return split_name[0], split_name[1]


def swap_module(
    mod: ModT, mapping: dict[type, type], qconfig_dict: dict
) -> nn.Module:
    """Replace ``mod`` with the module produced by its mapped class.

    When ``type(mod)`` is a key of ``mapping``, the mapped class builds the
    replacement through its ``from_float`` constructor and the original's
    forward hooks are carried over, so they keep firing around the
    replacement.  Modules whose type is not mapped are returned unchanged.

    Args:
        mod: module to replace
        mapping: dict mapping module types to replacement classes
        qconfig_dict: quantization configuration carried onto the
            replacement (unused when the mapped class ignores it)

    Return:
        The replacement module, or ``mod`` itself when its type is not
        mapped.
    """
    if type(mod) in mapping:
        new_mod = mapping[type(mod)].from_float(mod)
        new_mod.qconfig = qconfig_dict

        # Carry over the pre forward hooks; they run on the replacement's
        # input.
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        # Carry over the post forward hooks; they run on the replacement's
        # output.
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)

        return new_mod
    return mod
