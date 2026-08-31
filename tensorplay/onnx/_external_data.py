"""External-data storage for models whose weights exceed the protobuf limit.

A ``ModelProto`` is a single protobuf message and therefore capped at 2 GiB.
Large models keep their initializers in a side-car file and reference it from
the graph, which is what ``onnx.save_model(..., save_as_external_data=True)``
produces.
"""

from __future__ import annotations

import os
from typing import Any

import onnx
from onnx import ModelProto

__all__ = [
    "PROTOBUF_SIZE_LIMIT",
    "model_size",
    "needs_external_data",
    "save_model",
]

# Protobuf refuses to serialize a message larger than 2 GiB.
PROTOBUF_SIZE_LIMIT = 2 * 1024**3

# Weights below this many bytes stay inline: thousands of tiny side-car
# entries cost more (in files and load time) than they save.
DEFAULT_SIZE_THRESHOLD = 1024


def model_size(model: ModelProto) -> int:
    """Serialized size of ``model`` in bytes (initializer data included)."""

    return model.ByteSize()


def needs_external_data(model: ModelProto) -> bool:
    """Whether ``model`` must move its initializers out of the protobuf."""

    return model_size(model) >= PROTOBUF_SIZE_LIMIT


def save_model(
    model: ModelProto,
    f: Any,
    *,
    external_data: bool | None = None,
    location: str | None = None,
    size_threshold: int = DEFAULT_SIZE_THRESHOLD,
    all_tensors_to_one_file: bool = True,
) -> None:
    """Write ``model`` to ``f``.

    Args:
        model: the model to serialize.
        f: a filesystem path or a writable binary file object.
        external_data: force weights into a side-car file (``True``), force
            them inline (``False``), or decide from the model size (``None``).
        location: side-car file name, relative to the model's directory.
            Defaults to ``"<model file name>.data"``.
        size_threshold: initializers smaller than this stay inline.
        all_tensors_to_one_file: write one side-car file instead of one per
            initializer.
    """

    if external_data is None:
        external_data = needs_external_data(model)

    if not external_data:
        if needs_external_data(model):
            raise ValueError(
                f"the model is {model_size(model) / 1024**3:.2f} GiB, which exceeds "
                "the 2 GiB protobuf limit; pass external_data=True to store the "
                "weights in a side-car file"
            )
        onnx.save_model(model, f)
        return

    if not isinstance(f, (str, bytes, os.PathLike)):
        raise ValueError(
            "external data must be written next to the model, so f must be a "
            f"file path, not {type(f).__name__}"
        )

    path = os.fspath(f)
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=all_tensors_to_one_file,
        location=location or f"{os.path.basename(path)}.data",
        size_threshold=size_threshold,
        convert_attribute=False,
    )
