from __future__ import annotations

import importlib.util
from typing import Any, Iterable

from .._utils import _compute_local_shape_and_global_offset

__all__ = ["visualize_sharding"]

Color = tuple[float, float, float]


def _create_table(
    shards: list[tuple[tuple[int, int], tuple[int, int], int]],
    device_kind: str = "",
) -> str:
    row_ranges = sorted({block[0] for block in shards})
    col_ranges = sorted({block[1] for block in shards})
    matrix = [["" for _ in col_ranges] for _ in row_ranges]
    for row_range, col_range, device_index in shards:
        row_index = row_ranges.index(row_range)
        col_index = col_ranges.index(col_range)
        value = f"{device_kind}:{device_index}"
        matrix[row_index][col_index] = (
            value
            if not matrix[row_index][col_index]
            else f"{matrix[row_index][col_index]},{device_index}"
        )
    row_headers = [f"Row {start}-{end}" for start, end in row_ranges]
    col_headers = [f"Col {start}-{end}" for start, end in col_ranges]
    try:
        from tabulate import tabulate

        return tabulate(
            matrix, headers=col_headers, showindex=row_headers
        )
    except ImportError:
        lines = [" | ".join([""] + col_headers)]
        lines.extend(
            " | ".join([row_headers[index]] + row)
            for index, row in enumerate(matrix)
        )
        return "\n".join(lines)


def make_color_iter(
    color_map: Any, num_rows: int, num_cols: int
) -> Iterable[Any]:
    for index in range(num_rows * num_cols):
        yield color_map(index)


def _canonicalize_color(color: Color | str) -> str:
    if isinstance(color, str):
        return color
    red, green, blue = (int(value * 255) for value in color)
    return f"#{red:02X}{green:02X}{blue:02X}"


def _get_text_color(color: str) -> str:
    red, green, blue = (
        int(value, 16) for value in (color[1:3], color[3:5], color[5:7])
    )
    return "#000000" if (red * 0.299 + green * 0.587 + blue * 0.114) > 186 else "#ffffff"


def _create_rich_table(
    shape: tuple[int, ...],
    shards: list[tuple[tuple[int, int], tuple[int, int], int]],
    device_kind: str = "",
    scale: float = 1.0,
    min_width: int = 9,
    max_width: int = 80,
) -> None:
    del min_width
    import matplotlib
    import rich.align
    import rich.box
    import rich.console
    import rich.padding
    import rich.style
    import rich.table

    height = shape[0]
    width = shape[1] if len(shape) == 2 else 1
    row_ranges = sorted({item[0] for item in shards})
    col_ranges = sorted({item[1] for item in shards})
    num_rows, num_cols = len(row_ranges), len(col_ranges)
    console = rich.console.Console(width=max_width)
    use_color = console.color_system
    color_iter = make_color_iter(
        matplotlib.colormaps["tab20b"], num_rows, num_cols
    )
    base_height = int(10 * scale)
    aspect_ratio = width / height
    base_width = int(base_height * aspect_ratio)
    table = rich.table.Table(
        show_header=False,
        show_lines=not use_color,
        padding=0,
        highlight=not use_color,
        pad_edge=False,
        box=rich.box.SQUARE if not use_color else None,
    )
    for row_index in range(num_rows):
        cells = []
        for col_index in range(num_cols):
            entry = device_kind + ":" + ",".join(
                str(device_index)
                for row_range, col_range, device_index in shards
                if row_range == row_ranges[row_index]
                and col_range == col_ranges[col_index]
            )
            cell_width = (col_ranges[col_index][1] - col_ranges[col_index][0]) / width
            cell_width = int(cell_width * base_width * 2.5)
            cell_height = (row_ranges[row_index][1] - row_ranges[row_index][0]) / height
            cell_height = int(cell_height * base_height)
            left, remainder = divmod(cell_width - len(entry) - 2, 2)
            right = left + remainder
            top, remainder = divmod(cell_height - 2, 2)
            bottom = top + remainder
            if use_color:
                color = _canonicalize_color(next(color_iter)[:3])
                text_color = _get_text_color(color)
                left += 1
                right += 1
                top += 1
                bottom += 1
            else:
                color = None
                text_color = None
            padding = (
                max(top, 0),
                max(right, 0),
                max(bottom, 0),
                max(left, 0),
            )
            cells.append(
                rich.padding.Padding(
                    rich.align.Align(entry, "center", vertical="middle"),
                    padding,
                    style=rich.style.Style(bgcolor=color, color=text_color),
                )
            )
        table.add_row(*cells)
    console.print(table, end="\n\n")


def _flatten_mesh(value: Any) -> list[int]:
    if isinstance(value, (tuple, list)):
        result: list[int] = []
        for child in value:
            result.extend(_flatten_mesh(child))
        return result
    return [int(value)]


def visualize_sharding(
    dtensor: Any, header: str = "", use_rich: bool = False
) -> None:
    if dtensor.numel() == 0:
        return
    if len(dtensor.shape) >= 3:
        raise RuntimeError("visualize sharding supports only 1D or 2D values")
    mesh = dtensor.device_mesh
    coordinate = mesh.get_coordinate()
    if coordinate is None:
        return
    if not all(mesh.get_local_rank(mesh_dim=index) == 0 for index in range(int(mesh.ndim))):
        return
    ranks = _flatten_mesh(mesh.mesh)
    device_coords: dict[int, tuple[int, ...]] = {}
    for position, rank in enumerate(ranks):
        remaining = position
        coordinates = []
        for size in reversed(mesh.shape):
            coordinates.append(remaining % size)
            remaining //= size
        device_coords[rank] = tuple(reversed(coordinates))
    blocks = []
    for device_index, device_coordinate in device_coords.items():
        shape, offset = _compute_local_shape_and_global_offset(
            dtensor.shape,
            mesh.shape,
            device_coordinate,
            dtensor.placements,
        )
        if len(shape) == 1:
            shape = (shape[0], 1)
            offset = (offset[0], 0)
        blocks.append(
            (
                (offset[0], offset[0] + shape[0] - 1),
                (offset[1], offset[1] + shape[1] - 1),
                device_index,
            )
        )
    if header:
        print(header)
    if use_rich and importlib.util.find_spec("rich") and importlib.util.find_spec("matplotlib"):
        _create_rich_table(
            tuple(int(value) for value in dtensor.shape),
            blocks,
            device_kind=mesh.device_type,
        )
    elif importlib.util.find_spec("tabulate"):
        print(_create_table(blocks, device_kind=mesh.device_type))
    else:
        raise ValueError("visualize_sharding requires either rich or tabulate")
