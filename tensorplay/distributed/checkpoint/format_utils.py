from __future__ import annotations

import os
from concurrent.futures import Future
from enum import Enum
from typing import Any, cast

import tensorplay as tp
import tensorplay.distributed as dist

from ._nested_dict import flatten_state_dict
from .default_planner import DefaultLoadPlanner, _EmptyStateDictLoadPlanner
from .filesystem import FileSystemReader, FileSystemWriter
from .metadata import ChunkStorageMetadata, Metadata, TensorProperties, TensorStorageMetadata
from .planner import LoadItemType, LoadPlan, LoadPlanner
from .storage import StorageReader
from .state_dict_loader import _load_state_dict
from .state_dict_saver import _save_state_dict

__all__ = ["dcp_to_torch_save", "torch_save_to_dcp", "BroadcastingTorchSaveReader", "DynamicMetaLoadPlanner"]


class FormatMode(Enum):
    TORCH_TO_DCP = "torch_to_dcp"
    DCP_TO_TORCH = "dcp_to_torch"


class BroadcastingTorchSaveReader(StorageReader):
    def __init__(self, checkpoint_id: str | os.PathLike[str] | None = None, coordinator_rank: int = 0) -> None:
        self.checkpoint_id = checkpoint_id
        self.coordinator_rank = coordinator_rank
        self.is_coordinator = True

    def read_metadata(self) -> Metadata:
        return Metadata({})

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        planner = cast(DefaultLoadPlanner, planner)
        distributed = (
            dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )
        if self.is_coordinator:
            if self.checkpoint_id is None:
                raise AssertionError("checkpoint_id must be set before reading data")
            loaded = tp.load(self.checkpoint_id, map_location="cpu")
            if planner.flatten_state_dict:
                loaded, _ = flatten_state_dict(loaded)
        else:
            loaded = None
        for request in plan.items:
            if request.type is LoadItemType.BYTE_IO:
                raise RuntimeError(
                    f"non-tensor value at {request.storage_index.fqn}"
                )
            if distributed:
                if self.is_coordinator:
                    source = loaded[request.storage_index.fqn]
                    value = source.to(device="cpu")
                else:
                    value = tp.empty_like(planner.state_dict[request.storage_index.fqn])
                dist.broadcast(value, src=self.coordinator_rank)
                value = self._narrow(value, request.storage_offsets, request.lengths)
            else:
                if loaded is None:
                    raise RuntimeError("checkpoint data was not loaded")
                value = self._narrow(
                    loaded[request.storage_index.fqn],
                    request.storage_offsets,
                    request.lengths,
                )
            target = planner.resolve_tensor(request).detach()
            if tuple(target.shape) != tuple(value.shape):
                raise AssertionError(
                    f"request {request.storage_index} mismatch sizes "
                    f"{target.shape} vs {value.shape}"
                )
            target.copy_(value)
            planner.commit_tensor(request, target)
        future: Future[None] = Future()
        future.set_result(None)
        return future

    @staticmethod
    def _narrow(value: tp.Tensor, offsets: tuple[int, ...], lengths: tuple[int, ...]) -> tp.Tensor:
        for dimension, (offset, length) in enumerate(zip(offsets, lengths)):
            value = value.narrow(dimension, int(offset), int(length))
        return value

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool, *args: Any, **kwargs: Any) -> None:
        del metadata, args, kwargs
        self.is_coordinator = is_coordinator
        if dist.is_available() and dist.is_initialized() and is_coordinator:
            if dist.get_rank() != self.coordinator_rank:
                raise AssertionError(
                    f"Coordinator rank mismatch: expected {self.coordinator_rank}, "
                    f"got {dist.get_rank()}"
                )
        if self.checkpoint_id is None:
            raise AssertionError("checkpoint_id must be set before reading data")
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan
    def prepare_global_plan(self, global_plan: list[LoadPlan]) -> list[LoadPlan]:
        return global_plan
    def reset(self, checkpoint_id: str | os.PathLike[str] | None = None) -> None:
        self.checkpoint_id = checkpoint_id
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | os.PathLike[str]) -> bool:
        return os.path.isfile(checkpoint_id)


class DynamicMetaLoadPlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict: dict[str, Any], metadata: Metadata | None = None, is_coordinator: bool = False) -> None:
        super().set_up_planner(state_dict, metadata, is_coordinator)
        metadata = {}
        for key, value in self.state_dict.items():
            if not isinstance(value, tp.Tensor):
                raise RuntimeError(f"non-tensor value at {key}")
            shape = tuple(int(size) for size in value.shape)
            metadata[key] = TensorStorageMetadata(
                TensorProperties.create_from_tensor(value),
                shape,
                [ChunkStorageMetadata((0,) * len(shape), shape)],
            )
        self.metadata = Metadata(metadata)


def dcp_to_torch_save(dcp_checkpoint_dir: str | os.PathLike[str], torch_save_path: str | os.PathLike[str]) -> None:
    state: dict[str, Any] = {}
    _load_state_dict(
        state,
        storage_reader=FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    tp.save(state, torch_save_path)


def torch_save_to_dcp(torch_save_path: str | os.PathLike[str], dcp_checkpoint_dir: str | os.PathLike[str]) -> None:
    state = tp.load(torch_save_path, map_location="cpu")
    _save_state_dict(
        state,
        storage_writer=FileSystemWriter(dcp_checkpoint_dir),
        no_dist=True,
    )
