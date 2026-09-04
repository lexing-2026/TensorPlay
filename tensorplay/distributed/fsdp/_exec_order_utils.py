"""Execution order tracking for parameter communication."""

import itertools
import warnings
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import tensorplay as tp

from .. import distributed_core as dist

__all__ = ["_ExecOrderWarnStatus", "_ExecOrderData"]


class _ExecOrderWarnStatus(Enum):
    NONE = auto()
    WARNING = auto()
    WARNED = auto()


@dataclass
class _ExecOrderData:
    def __init__(
        self,
        debug_level: Any = None,
        backward_prefetch_limit: int = 1,
        forward_prefetch_limit: int = 1,
    ) -> None:
        self.handles_pre_forward_order: list[Any] = []
        self.handles_post_forward_order: list[Any | None] = []
        self._iter = 0
        self._backward_prefetch_limit = int(backward_prefetch_limit)
        self._forward_prefetch_limit = int(forward_prefetch_limit)
        name = getattr(debug_level, "name", str(debug_level))
        self._checking_order = name in {"DETAIL", "DEBUG"}
        self.process_group = None
        self.world_size = None
        self.all_handles: list[Any] = []
        self.param_to_fqn: dict[Any, list[str]] = {}
        self.current_order_index = 0
        self.warn_status = _ExecOrderWarnStatus.NONE
        self.handles = self.all_handles
        self.forward_order = self.handles_pre_forward_order
        self.backward_order = self.handles_post_forward_order

    def init(self, state: Any, root_module: Any, process_group: Any) -> None:
        del state
        self.process_group = process_group
        rank = getattr(process_group, "rank", None)
        size = getattr(process_group, "size", None)
        self.rank = int(rank() if callable(rank) else rank or 0)
        self.world_size = int(size() if callable(size) else size or 1)
        from ._traversal_utils import _get_fsdp_handles
        from ._common_utils import _get_param_to_fqns

        for handle in _get_fsdp_handles(root_module):
            handle._handle_index = len(self.all_handles)
            self.all_handles.append(handle)
        self.param_to_fqn = _get_param_to_fqns(root_module)

    @property
    def is_first_iter(self) -> bool:
        return self._iter == 0

    def get_handle_to_backward_prefetch(self, current_handle: Any) -> Any:
        current_index = getattr(current_handle, "_post_forward_index", None)
        if current_index is None:
            return None
        target = None
        index = current_index - 1
        for _ in range(max(0, self._backward_prefetch_limit)):
            if index < 0:
                break
            target = self.handles_post_forward_order[index]
            index -= 1
        return target

    def get_handle_to_forward_prefetch(self, current_handle: Any) -> Any:
        current_index = getattr(current_handle, "_pre_forward_order_index", None)
        if current_index is None:
            return None
        target = None
        index = current_index + 1
        for _ in range(max(0, self._forward_prefetch_limit)):
            if index >= len(self.handles_pre_forward_order):
                break
            target = self.handles_pre_forward_order[index]
            index += 1
        return target

    def record_forward(self, handle: Any) -> None:
        self.record_pre_forward(handle, True)

    def record_backward(self, handle: Any) -> None:
        self.record_post_forward(handle)

    def record_post_forward(self, handle: Any) -> None:
        if handle is None:
            return
        index = len(self.handles_post_forward_order)
        if getattr(handle, "_post_forward_index", None) is None:
            handle._post_forward_index = index
        self.handles_post_forward_order.append(handle)

    def record_pre_forward(self, handle: Any, is_training: bool = True) -> None:
        if handle is None:
            return
        self._check_order(handle, is_training)
        if not self.is_first_iter or getattr(handle, "_pre_forward_order_index", None) is not None:
            return
        handle._pre_forward_order_index = len(self.handles_pre_forward_order)
        self.handles_pre_forward_order.append(handle)

    def _check_order(self, handle: Any, is_training: bool) -> None:
        if not is_training or not self._checking_order:
            return
        if self.is_first_iter:
            optional_local_indices = self._get_handle_indices(handle)
            world_size = int(self.world_size or 1)
            if world_size > 1 and dist.is_initialized() and self.process_group is not None:
                device = getattr(handle, "device", "cpu")
                valid_count = sum(index is not None for index in optional_local_indices)
                local_count = tp.tensor([valid_count], dtype=tp.int32, device=device)
                world_counts = tp.empty((world_size,), dtype=tp.int32, device=device)
                dist.all_gather_single(world_counts, local_count, group=self.process_group)
                counts = [int(value) for value in world_counts.detach().cpu().tolist()]
                for (rank1, count1), (rank2, count2) in itertools.combinations(
                    enumerate(counts), 2
                ):
                    if count1 != count2:
                        raise RuntimeError(
                            "forward execution order differs across ranks: "
                            f"rank {rank1} gathers {count1} parameters while "
                            f"rank {rank2} gathers {count2} parameters"
                        )
                if valid_count:
                    local_indices = tp.tensor(
                        [index if index is not None else -1 for index in optional_local_indices],
                        dtype=tp.int32,
                        device=device,
                    )
                    world_indices = tp.empty(
                        (world_size * valid_count,), dtype=tp.int32, device=device
                    )
                    dist.all_gather_single(
                        world_indices, local_indices, group=self.process_group
                    )
                    values = world_indices.detach().cpu().tolist()
                    expected = [
                        values[offset : offset + valid_count]
                        for offset in range(0, len(values), valid_count)
                    ]
                    if any(indices != expected[0] for indices in expected[1:]):
                        raise RuntimeError(
                            "forward execution order differs across ranks for "
                            f"{self._get_names_from_handles(handle)}"
                        )
            return
        if self.warn_status == _ExecOrderWarnStatus.WARNED:
            return
        prefix = None
        if self.current_order_index >= len(self.handles_pre_forward_order):
            prefix = "forward gathered extra parameters for "
        else:
            expected = self.handles_pre_forward_order[self.current_order_index]
            if expected is not handle:
                prefix = f"expected {self._get_names_from_handles(expected)} but got "
        if prefix is not None:
            warnings.warn(
                "forward execution order changed after initialization; "
                f"communication order is unchecked ({prefix}{self._get_names_from_handles(handle)})",
                stacklevel=2,
            )
            self.warn_status = _ExecOrderWarnStatus.WARNING
        self.current_order_index += 1

    def _get_handle_indices(self, handle: Any) -> tuple[int | None, ...]:
        return (getattr(handle, "_handle_index", None),) if handle is not None else ()

    def _get_names_from_handle_indices(self, indices: tuple[int | None, ...]) -> list[list[str]]:
        result: list[list[str]] = []
        for index in indices:
            if index is None or index < 0 or index >= len(self.all_handles):
                continue
            flat = getattr(self.all_handles[index], "flat_param", None)
            if flat in self.param_to_fqn:
                result.append(self.param_to_fqn[flat])
        return result

    def _get_names_from_handles(self, handle: Any) -> list[list[str]]:
        flat = getattr(handle, "flat_param", None)
        return [self.param_to_fqn[flat]] if flat in self.param_to_fqn else []

    def next_iter(self) -> None:
        self._iter += 1
        self.handles_post_forward_order.clear()
        self.current_order_index = 0
        if self.warn_status == _ExecOrderWarnStatus.WARNING:
            self.warn_status = _ExecOrderWarnStatus.WARNED
