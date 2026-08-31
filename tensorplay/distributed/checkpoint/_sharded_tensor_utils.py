from __future__ import annotations

from typing import Any

from ._traverse import traverse_state_dict

__all__ = ["_flatten_sharded_tensors"]


def _flatten_sharded_tensors(state_dict: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    def visit(path: tuple[Any, ...], value: Any) -> None:
        key = ".".join(map(str, path))
        local = getattr(value, "local_shards", None)
        if callable(local):
            for index, shard in enumerate(local()):
                tensor = getattr(shard, "tensor", shard)
                result[f"{key}.{index}"] = tensor
        else:
            result[key] = value

    traverse_state_dict(state_dict, visit)
    return result
