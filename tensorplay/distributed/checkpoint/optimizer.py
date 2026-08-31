from __future__ import annotations

from typing import Any

import tensorplay as tp

__all__ = ["load_sharded_optimizer_state_dict"]


def _gen_rank_device(global_rank: int, device_type: str = "cuda") -> str:
    return f"{device_type}:{global_rank}"


def _create_colwise_spec(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    return args[0] if args else None


def _is_nested_tensor(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and any(isinstance(item, (list, tuple)) for item in value)


def _alloc_tensor(size: Any, dtype: Any, device: Any) -> tp.Tensor:
    return tp.empty(size, dtype=dtype, device=device)


def _get_state_dict_2d_layout(*args: Any, **kwargs: Any) -> Any:
    del kwargs
    return args[0] if args else None


def load_sharded_optimizer_state_dict(model: Any, optimizer: Any, checkpoint_id: str, process_group: Any = None) -> dict[str, Any]:
    del model, optimizer, process_group
    return tp.load(checkpoint_id)
