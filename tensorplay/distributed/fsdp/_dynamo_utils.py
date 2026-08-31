"""Compilation annotations for sharded modules."""

from typing import Any

__all__ = ["_annotate_modules_for_dynamo"]


def _annotate_modules_for_dynamo(root: Any) -> Any:
    for module in root.modules():
        setattr(module, "_fsdp_dynamo_annotated", True)
    return root
