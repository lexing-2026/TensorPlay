from __future__ import annotations

from concurrent.futures import Future
from typing import Any

from .state_dict_loader import load
from .state_dict_saver import save

__all__: list[str] = []


class _Checkpointer:
    def __init__(
        self,
        storage_writer: Any = None,
        storage_reader: Any = None,
        *,
        process_group: Any = None,
        coordinator_rank: int = 0,
        no_dist: bool = False,
        load_planner: Any = None,
        save_planner: Any = None,
        checkpoint_id: str | None = None,
        planner: Any = None,
    ) -> None:
        self.storage_writer = storage_writer
        self.storage_reader = storage_reader
        self.process_group = process_group
        self.coordinator_rank = coordinator_rank
        self.no_dist = no_dist
        self.load_planner = load_planner
        self.save_planner = save_planner if save_planner is not None else planner
        self.checkpoint_id = checkpoint_id

    def save(self, state_dict: dict[str, Any], **kwargs: Any) -> Any:
        return save(
            state_dict,
            checkpoint_id=kwargs.pop("checkpoint_id", self.checkpoint_id),
            storage_writer=kwargs.pop("storage_writer", self.storage_writer),
            planner=kwargs.pop("planner", self.save_planner),
            process_group=kwargs.pop("process_group", self.process_group),
            no_dist=kwargs.pop("no_dist", self.no_dist),
            **kwargs,
        )

    def async_save(self, state_dict: dict[str, Any], **kwargs: Any) -> Future[Any]:
        from .state_dict_saver import async_save

        response = async_save(
            state_dict,
            checkpoint_id=kwargs.pop("checkpoint_id", self.checkpoint_id),
            storage_writer=kwargs.pop("storage_writer", self.storage_writer),
            planner=kwargs.pop("planner", self.save_planner),
            process_group=kwargs.pop("process_group", self.process_group),
            no_dist=kwargs.pop("no_dist", self.no_dist),
            **kwargs,
        )
        if hasattr(response, "upload_completion"):
            return response.upload_completion
        return response

    def load(self, state_dict: dict[str, Any], **kwargs: Any) -> None:
        load(
            state_dict,
            checkpoint_id=kwargs.pop("checkpoint_id", self.checkpoint_id),
            storage_reader=kwargs.pop("storage_reader", self.storage_reader),
            planner=kwargs.pop("planner", self.load_planner),
            process_group=kwargs.pop("process_group", self.process_group),
            no_dist=kwargs.pop("no_dist", self.no_dist),
            **kwargs,
        )
