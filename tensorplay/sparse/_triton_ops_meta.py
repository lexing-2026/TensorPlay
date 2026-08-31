"""Persistent tuning metadata and deterministic block-sparse helpers."""

from __future__ import annotations

import os
from typing import Any, Callable

__all__ = [
    "_get_device_name",
    "create_blocked_tensor",
    "dump",
    "get_meta",
    "main",
    "minimize",
    "optimize_bsr_dense_addmm",
    "optimize_scatter_mm",
    "tune__int_bsr_dense_addmm",
    "tune_bsr_dense_addmm",
    "update",
]

_META: dict[tuple[Any, ...], Any] = {}


def _get_device_name() -> str:
    return os.environ.get("TENSORPLAY_DEVICE_NAME", "cpu")


def _version_key(version: Any) -> Any:
    if isinstance(version, tuple):
        return tuple(str(value) for value in version)
    return str(version)


def get_meta(op: Any, key: Any, device_name: str | None = None, version: Any = (0, "float16", 0.5), exact: bool = False) -> Any:
    device = device_name or _get_device_name()
    value = _META.get((op, device, _version_key(version), key))
    if value is not None:
        return dict(value) if isinstance(value, dict) else value
    if exact:
        return None
    for (saved_op, saved_device, saved_version, saved_key), saved_value in _META.items():
        if saved_op == op and saved_device == device and saved_version == _version_key(version):
            if isinstance(saved_key, tuple) and isinstance(key, tuple) and len(saved_key) == len(key):
                if all(left == right or left == "*" or right == "*" for left, right in zip(saved_key, key)):
                    return dict(saved_value) if isinstance(saved_value, dict) else saved_value
    return None


def update(op: Any, device_name: str, version: Any, key: Any, value: Any) -> None:
    if value:
        _META[(op, device_name, _version_key(version), key)] = dict(value) if isinstance(value, dict) else value


def dump() -> dict[str, Any]:
    return {
        repr(key): dict(value) if isinstance(value, dict) else value
        for key, value in _META.items()
    }


def minimize(
    target_func: Callable[[Any], Any],
    initial_parameters: Any,
    reference_parameters: Any,
    step_func: Callable[[Any], Any],
    tolerance: float = 0.0,
    max_steps: int = 100,
) -> Any:
    del tolerance
    current = initial_parameters
    reference = target_func(reference_parameters)
    for _ in range(max_steps):
        candidate = step_func(current)
        if target_func(candidate) <= reference:
            current = candidate
            reference = target_func(candidate)
        else:
            break
    return current


def create_blocked_tensor(B: int, M: int, N: int, blocksize: tuple[int, int], sparsity: float, dtype: Any, device: Any) -> Any:
    if not 0 <= sparsity <= 1:
        raise ValueError("sparsity must be between zero and one")
    if M % blocksize[0] or N % blocksize[1]:
        raise ValueError("matrix shape must be divisible by blocksize")
    import tensorplay

    tensor = tensorplay.zeros((B, M, N), dtype=dtype, device=device)
    active = int(round((1.0 - sparsity) * (M // blocksize[0]) * (N // blocksize[1])))
    for batch in range(B):
        for index in range(active):
            row = index // (N // blocksize[1])
            col = index % (N // blocksize[1])
            tensor[batch, row * blocksize[0] : (row + 1) * blocksize[0], col * blocksize[1] : (col + 1) * blocksize[1]] = 1
    return tensor


def optimize_scatter_mm(m: int, k: int, n: int, bm: int, bk: int, dtype: Any = None, device: Any = "cpu", sparsity: float = 0.5, force: bool = False) -> dict[str, Any]:
    del dtype, device, sparsity, force
    result = {"TILE_M": bm, "TILE_N": min(n, 32), "GROUP_SIZE": 1, "num_stages": 1, "num_warps": 1, "SPLIT_N": max(1, n // max(bm, 1))}
    update("scatter_mm", _get_device_name(), (0, "default", 0.5), (m, k, n, bm, bk), result)
    return result


def tune__int_bsr_dense_addmm(input: Any, bsr: Any, dense: Any, **kwargs: Any) -> dict[str, Any]:
    return tune_bsr_dense_addmm(input, bsr, dense, **kwargs)


def tune_bsr_dense_addmm(input: Any, bsr: Any, dense: Any, **kwargs: Any) -> dict[str, Any]:
    del input
    return optimize_bsr_dense_addmm(*bsr.shape[-2:], bsr.values().shape[-2], bsr.values().shape[-1], dtype=getattr(dense, "dtype", None), **kwargs)


def optimize_bsr_dense_addmm(m: int, k: int, n: int, bm: int, bk: int, dtype: Any = None, device: Any = "cpu", sparsity: float = 0.5, force: bool = False, **kwargs: Any) -> dict[str, Any]:
    del dtype, device, sparsity, force
    result = {"SPLIT_N": max(1, n // max(bm, 1)), "GROUP_SIZE_ROW": 4, "num_stages": 1, "num_warps": 1}
    result.update(kwargs)
    update("bsr_dense_addmm", _get_device_name(), (0, "default", 0.5), (m, k, n, bm, bk), result)
    return result


def main(op: str = "scatter_mm", force: bool = False, dtype: Any = None, verbose: bool = True) -> dict[str, Any]:
    del dtype, verbose
    if op == "scatter_mm":
        return optimize_scatter_mm(256, 256, 256, 16, 16, force=force)
    if op == "bsr_dense_addmm":
        return optimize_bsr_dense_addmm(256, 256, 256, 16, 16, force=force)
    raise ValueError(f"unknown tuning operation {op!r}")
