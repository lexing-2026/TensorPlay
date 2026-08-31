from __future__ import annotations

import os
from concurrent.futures import Future
from typing import Any

import tensorplay as tp

from ._nested_dict import flatten_state_dict
from .default_planner import DefaultLoadPlanner
from .filesystem import FileSystemReader, FileSystemWriter
from .metadata import Metadata, TensorProperties, TensorStorageMetadata, ChunkStorageMetadata
from .planner import LoadItemType, LoadPlan, LoadPlanner

__all__ = ["dcp_to_torch_save", "torch_save_to_dcp", "BroadcastingTorchSaveReader", "DynamicMetaLoadPlanner"]


class BroadcastingTorchSaveReader:
    def __init__(self, checkpoint_id: str | os.PathLike[str] | None = None, coordinator_rank: int = 0) -> None:
        self.checkpoint_id = checkpoint_id
        self.coordinator_rank = coordinator_rank
        self.is_coordinator = True

    def read_metadata(self) -> Metadata:
        return Metadata({})

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        del plan
        loaded = tp.load(self.checkpoint_id)
        for key, value in loaded.items():
            if key in getattr(planner, "state_dict", {}) and isinstance(value, tp.Tensor):
                planner.state_dict[key].copy_(value)
        future: Future[None] = Future()
        future.set_result(None)
        return future

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        del metadata
        self.is_coordinator = is_coordinator
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
        self.metadata = Metadata({key: TensorStorageMetadata(TensorProperties.create_from_tensor(value), tuple(value.shape), [ChunkStorageMetadata(tuple(0 for _ in value.shape), tuple(value.shape))]) for key, value in self.state_dict.items() if isinstance(value, tp.Tensor)})


def dcp_to_torch_save(dcp_checkpoint_dir: str | os.PathLike[str], torch_save_path: str | os.PathLike[str]) -> None:
    state = {}
    reader = FileSystemReader(dcp_checkpoint_dir)
    loaded = reader.read_data(None, state)
    tp.save(loaded, torch_save_path)


def torch_save_to_dcp(torch_save_path: str | os.PathLike[str], dcp_checkpoint_dir: str | os.PathLike[str]) -> None:
    state = tp.load(torch_save_path)
    writer = FileSystemWriter(dcp_checkpoint_dir)
    writer.set_up_storage_writer(True)
    writer.write_data(state)
    writer.finish({})
