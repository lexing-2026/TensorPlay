from __future__ import annotations

from dataclasses import dataclass, field

from .barriers import BarrierConfig
from .checkpoint_process import CheckpointProcessConfig
from .checkpoint_writer import CheckpointWriterConfig
from .staging import CheckpointStagerConfig


@dataclass
class CheckpointerConfig:
    writer_config: CheckpointWriterConfig = field(default_factory=CheckpointWriterConfig)
    barrier_config: BarrierConfig = field(default_factory=BarrierConfig)
    staging_config: CheckpointStagerConfig = field(default_factory=CheckpointStagerConfig)
    process_config: CheckpointProcessConfig = field(default_factory=CheckpointProcessConfig)
