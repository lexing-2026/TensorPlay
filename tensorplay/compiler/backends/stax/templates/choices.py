"""Which tables a device is measured with.

The device is asked for its tables rather than reaching for them, so a new kind of
device brings its own numbers instead of inheriting another's.
"""

from __future__ import annotations

from .heuristics import CpuConfigHeuristic, CudaConfigHeuristic, _TileConfigHeuristic

class Choices:
    """Which tables a device is measured with.

    The device is asked for its tables rather than reaching for them, so a new
    kind of device brings its own numbers instead of inheriting another's.
    """

    def __init__(self, heuristics: dict | None = None):
        self._heuristics = dict(heuristics or {})

    def register(self, device_type: str, heuristic: _TileConfigHeuristic) -> None:
        self._heuristics[device_type] = heuristic

    def get_config_heuristics(self, device_type: str | None = "cuda"):
        return self._heuristics.get(str(device_type), self._heuristics["cuda"])

    def get_conv_configs(self, device_type: str | None = "cuda"):
        return self.get_config_heuristics(device_type).get_conv_configs()

    def get_depthwise_conv_configs(self, device_type: str | None = "cuda"):
        return self.get_config_heuristics(device_type).get_depthwise_conv_configs()

    def get_mm_configs(self, device_type: str | None = "cuda"):
        return self.get_config_heuristics(device_type).get_mm_configs()


#: The tables, one per kind of device.
CHOICES = Choices({"cuda": CudaConfigHeuristic(), "cpu": CpuConfigHeuristic()})
