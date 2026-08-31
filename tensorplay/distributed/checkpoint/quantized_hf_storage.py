from __future__ import annotations

from typing import Any

from .hf_storage import HuggingFaceStorageReader

__all__ = ["QuantizedHuggingFaceStorageReader"]


class QuantizedHuggingFaceStorageReader(HuggingFaceStorageReader):
    def __init__(self, path: str, thread_count: int = 1, **kwargs: Any) -> None:
        super().__init__(path, thread_count)
        self._kwargs = kwargs

    def read_metadata(self) -> Any:
        return super().read_metadata()

    def _load_quantization_metadata(self) -> Any:
        return self.read_metadata().get("quantization_config", {})

    def _build_weight_scale_mapping(self, weight_map: dict[str, str]) -> dict[str, str]:
        return {key: value for key, value in weight_map.items() if "scale" in key}

    def _process_read_request(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def _get_slice_to_block_mapping(self, *args: Any, **kwargs: Any) -> dict[Any, Any]:
        del args, kwargs
        return {}

    def _dequantize_tensor_mxfp4(self, tensor: Any, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return tensor

    def _dequantize_tensor(self, tensor: Any, *args: Any, **kwargs: Any) -> Any:
        return self._dequantize_tensor_mxfp4(tensor, *args, **kwargs)

    def _is_tensor_quantized(self, tensor_fqn: str) -> bool:
        return tensor_fqn in self._build_weight_scale_mapping(self.read_metadata().get("weight_map", {}))

    def _read_quantized_tensor_with_block_alignment(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        return None
