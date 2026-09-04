"""Compilation annotations for sharded modules."""

from typing import Any

__all__ = ["_annotate_modules_for_dynamo"]


def _annotate_modules_for_dynamo(
    module: Any,
    ignored_modules: set[Any] | None = None,
    use_orig_params: bool = False,
) -> None:
    ignored_modules = ignored_modules or set()
    for submodule in module.modules():
        if submodule in ignored_modules:
            continue
        submodule._is_fsdp_managed_module = True
        submodule._fsdp_use_orig_params = bool(use_orig_params)
