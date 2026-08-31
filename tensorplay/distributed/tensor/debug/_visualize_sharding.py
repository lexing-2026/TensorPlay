"""Text visualization for one- and two-dimensional layouts."""

from __future__ import annotations

from typing import Any, Iterable

__all__ = ["visualize_sharding"]


def _create_table(shards: list[tuple[tuple[int, int], tuple[int, int], int]], device_kind: str = "") -> str:
    rows = sorted({shard[0] for shard in shards})
    columns = sorted({shard[1] for shard in shards})
    cells = []
    for row in rows:
        values = []
        for column in columns:
            ids = [f"{device_kind}:{rank}" for current_row, current_column, rank in shards if current_row == row and current_column == column]
            values.append(",".join(ids))
        cells.append(" | ".join(values))
    return "\n".join(cells)


def make_color_iter(color_map: Any, num_rows: int, num_cols: int) -> Iterable[Any]:
    return (color_map(index) for index in range(num_rows * num_cols))


def visualize_sharding(dtensor: Any, header: str = "", use_rich: bool = False) -> str:
    del use_rich
    shape = tuple(dtensor.shape)
    if len(shape) > 2:
        raise RuntimeError("visualize_sharding supports one- and two-dimensional values")
    placement_text = ", ".join(str(placement) for placement in dtensor.placements)
    result = f"{header}\n" if header else ""
    result += f"shape={shape}, placements=({placement_text})"
    print(result)
    return result
