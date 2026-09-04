from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import tensorplay as tp

from ._hf_utils import _metadata_fn
from .hf_storage import HuggingFaceStorageReader
from .metadata import TensorStorageMetadata

logger = logging.getLogger(__name__)

__all__ = ["QuantizedHuggingFaceStorageReader"]


class QuantizedHuggingFaceStorageReader(HuggingFaceStorageReader):
    def __init__(
        self,
        path: str,
        thread_count: int = 1,
        target_dtype: Any = None,
        block_size: int = 128,
    ) -> None:
        super().__init__(path=path, thread_count=thread_count)
        self.target_dtype = target_dtype or tp.float32
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self._weight_scale_mapping: dict[str, str] = {}
        self._weight_map: dict[str, str] = {}
        self._tensor_full_shapes: dict[str, tuple[int, ...]] = {}

    def reset(self, checkpoint_id: str | Path | None = None) -> None:
        super().reset(checkpoint_id)
        self._weight_scale_mapping.clear()
        self._weight_map.clear()
        self._tensor_full_shapes.clear()

    def _load_quantization_metadata(self) -> None:
        index_path = Path(self.path) / _metadata_fn
        if not index_path.exists():
            return
        value = json.loads(index_path.read_text())
        self._build_weight_scale_mapping(value.get("weight_map", {}))

    def _build_weight_scale_mapping(self, weight_map: dict[str, str]) -> dict[str, str]:
        self._weight_map = dict(weight_map)
        self._weight_scale_mapping.clear()
        for name in weight_map:
            if name.endswith(".weight_scale_inv"):
                weight_name = name.removesuffix(".weight_scale_inv") + ".weight"
                if weight_name in weight_map:
                    self._weight_scale_mapping[weight_name] = name
            elif name.endswith("_scales"):
                blocks_name = name.removesuffix("_scales") + "_blocks"
                if blocks_name in weight_map:
                    self._weight_scale_mapping[blocks_name] = name
        return dict(self._weight_scale_mapping)

    def read_metadata(self) -> Any:
        metadata = super().read_metadata()
        self._load_quantization_metadata()
        for fqn, tensor_metadata in metadata.state_dict_metadata.items():
            if not isinstance(tensor_metadata, TensorStorageMetadata):
                continue
            if fqn.endswith("_blocks") and len(tensor_metadata.size) >= 2:
                shape = tuple(int(value) for value in tensor_metadata.size)
                self._tensor_full_shapes[fqn + "_quantized"] = shape
                prefix, groups, values = shape[:-2], shape[-2], shape[-1]
                expanded = (*prefix, groups * values * 2)
                tensor_metadata.size = expanded
                self._tensor_full_shapes[fqn] = expanded
            else:
                self._tensor_full_shapes[fqn] = tuple(tensor_metadata.size)
        return metadata

    def _get_slice_to_block_mapping(
        self, req: Any
    ) -> tuple[tuple[int, int], tuple[int, int], slice, slice]:
        row_start = int(req.storage_offsets[0])
        row_end = row_start + int(req.lengths[0])
        col_start = int(req.storage_offsets[1])
        col_end = col_start + int(req.lengths[1])
        return (
            (row_start // self.block_size, (row_end - 1) // self.block_size + 1),
            (col_start // self.block_size, (col_end - 1) // self.block_size + 1),
            slice(row_start, row_end),
            slice(col_start, col_end),
        )

    def _dequantize_tensor(
        self,
        weight: tp.Tensor,
        scale_inv: tp.Tensor,
        full_tensor_shape: tuple[int, ...],
        slice_info: tuple[tuple[int, int], tuple[int, int], slice, slice],
    ) -> tp.Tensor:
        del full_tensor_shape
        row_blocks, col_blocks, row_slice, col_slice = slice_info
        result = weight.to(dtype=self.target_dtype)
        scale = scale_inv.to(dtype=self.target_dtype)
        for row in range(row_blocks[0], row_blocks[1]):
            for col in range(col_blocks[0], col_blocks[1]):
                row_begin = max(row * self.block_size, int(row_slice.start)) - int(row_slice.start)
                row_end = min((row + 1) * self.block_size, int(row_slice.stop)) - int(row_slice.start)
                col_begin = max(col * self.block_size, int(col_slice.start)) - int(col_slice.start)
                col_end = min((col + 1) * self.block_size, int(col_slice.stop)) - int(col_slice.start)
                if row_begin >= row_end or col_begin >= col_end:
                    continue
                result[row_begin:row_end, col_begin:col_end] = (
                    weight[row_begin:row_end, col_begin:col_end].to(dtype=self.target_dtype)
                    * scale[row, col]
                )
        return result

    def _dequantize_tensor_mxfp4(
        self,
        blocks: tp.Tensor,
        scales: tp.Tensor,
        req: Any,
        group_start: int,
        offset_in_first_group: int,
    ) -> tp.Tensor:
        values = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
        table = tp.tensor(values, dtype=self.target_dtype, device=blocks.device)
        indices_low = (blocks & 15).to(dtype=tp.int64)
        indices_high = (blocks >> 4).to(dtype=tp.int64)
        decoded = tp.empty((*blocks.shape[:-1], blocks.shape[-1] * 2), dtype=self.target_dtype, device=blocks.device)
        decoded[..., 0::2] = table[indices_low]
        decoded[..., 1::2] = table[indices_high]
        exponent = scales[..., group_start : group_start + blocks.shape[-2]].to(dtype=tp.int32) - 127
        decoded = tp.ldexp(decoded, exponent.unsqueeze(-1))
        decoded = decoded.reshape(*blocks.shape[:-2], -1)
        start = int(offset_in_first_group)
        length = int(req.lengths[-1])
        return decoded[..., start : start + length]

    def _is_tensor_quantized(self, tensor_fqn: str) -> bool:
        return tensor_fqn in self._weight_scale_mapping

    def _read_quantized_tensor_with_block_alignment(
        self, req: Any, safetensor_file: Any
    ) -> tp.Tensor:
        tensor_fqn = req.storage_index.fqn
        scale_fqn = self._weight_scale_mapping[tensor_fqn]
        if not isinstance(safetensor_file, dict):
            raise TypeError("quantized reader requires a tensor mapping")
        weight = safetensor_file[tensor_fqn]
        scales = safetensor_file[scale_fqn]
        if tensor_fqn.endswith("_blocks"):
            values_per_group = int(self._tensor_full_shapes[tensor_fqn + "_quantized"][-1]) * 2
            start = int(req.storage_offsets[-1])
            group_start = start // values_per_group
            offset = start - group_start * values_per_group
            groups = (offset + int(req.lengths[-1]) + values_per_group - 1) // values_per_group
            slices = [
                slice(int(offset), int(offset) + int(length))
                for offset, length in zip(req.storage_offsets[:-1], req.lengths[:-1])
            ]
            slices.extend((slice(group_start, group_start + groups), slice(None)))
            blocks = weight[tuple(slices)]
            return self._dequantize_tensor_mxfp4(
                blocks, scales, req, group_start, offset
            )
        slices = tuple(
            slice(int(offset), int(offset) + int(length))
            for offset, length in zip(req.storage_offsets, req.lengths)
        )
        quantized = weight[slices]
        return self._dequantize_tensor(
            quantized,
            scales,
            self._tensor_full_shapes[tensor_fqn],
            self._get_slice_to_block_mapping(req),
        )

    def _process_read_request(
        self, f: dict[str, Any], req: Any, planner: Any
    ) -> None:
        if self._is_tensor_quantized(req.storage_index.fqn):
            value = self._read_quantized_tensor_with_block_alignment(req, f)
            target = planner.resolve_tensor(req).detach()
            target.copy_(value)
            planner.commit_tensor(req, target)
            return
        super()._process_read_request(f, req, planner)
