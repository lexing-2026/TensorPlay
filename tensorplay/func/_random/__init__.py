"""Stateless counter-based random transforms.

The key is a uint64 tensor with a trailing pair ``(seed, offset)``.  Keys are
values, not mutable generator handles: splitting or folding a key always
returns a new key, while the distribution helpers write only to their result
tensor.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import tensorplay

__all__ = ["key", "split", "fold_in", "normal_", "normal", "uniform_", "uniform"]

_MASK32 = 0xFFFF_FFFF
_MASK53 = 0x1F_FFFF_FFFF_FFFF
_PHILOX_M0 = 0xD251_1F53
_PHILOX_M1 = 0xCD9E_8D57
_PHILOX_W0 = 0x9E37_79B9
_PHILOX_W1 = 0xBB67_AE85
_INV_2_POW_32 = 2.3283064365386963e-10
_INV_2_POW_24 = 5.960464477539063e-8
_INV_2_POW_53 = 1.1102230246251565e-16
_TWO_PI = 6.283185307179586


def _require_tensor(value: Any, name: str) -> None:
    if not isinstance(value, tensorplay.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(value)!r}")


def _validate_key(value: Any, name: str = "key") -> None:
    _require_tensor(value, name)
    if value.dim() < 1 or value.shape[-1] != 2:
        raise ValueError(
            f"{name} must have shape (*batch, 2), got shape {value.shape}"
        )
    if value.dtype != tensorplay.uint64:
        raise ValueError(f"{name} must have dtype uint64, got {value.dtype}")


def _as_uint64(value: int, device: Any) -> tensorplay.Tensor:
    return tensorplay.tensor([value], dtype=tensorplay.uint64, device=device)


def _philox_4x32(seed: tensorplay.Tensor, offset: tensorplay.Tensor) -> tensorplay.Tensor:
    """Return four uint32 words for every broadcasted seed/offset pair."""
    if seed.dim() == 0:
        seed = seed.reshape([1])
    if offset.dim() == 0:
        offset = offset.reshape([1])
    seed_lo = seed & _MASK32
    seed_hi = (seed >> 32) & _MASK32
    ctr0 = offset & _MASK32
    ctr1 = (offset >> 32) & _MASK32
    ctr2 = tensorplay.zeros_like(ctr0)
    ctr3 = tensorplay.zeros_like(ctr0)

    key0, key1 = seed_lo, seed_hi
    for _ in range(9):
        product0 = ctr0 * _PHILOX_M0
        product1 = ctr2 * _PHILOX_M1
        hi0, lo0 = (product0 >> 32) & _MASK32, product0 & _MASK32
        hi1, lo1 = (product1 >> 32) & _MASK32, product1 & _MASK32
        ctr0, ctr1, ctr2, ctr3 = (
            (hi1 ^ ctr1 ^ key0) & _MASK32,
            lo1,
            (hi0 ^ ctr3 ^ key1) & _MASK32,
            lo0,
        )
        key0 = (key0 + _PHILOX_W0) & _MASK32
        key1 = (key1 + _PHILOX_W1) & _MASK32

    product0 = ctr0 * _PHILOX_M0
    product1 = ctr2 * _PHILOX_M1
    hi0, lo0 = (product0 >> 32) & _MASK32, product0 & _MASK32
    hi1, lo1 = (product1 >> 32) & _MASK32, product1 & _MASK32
    return tensorplay.stack(
        (
            (hi1 ^ ctr1 ^ key0) & _MASK32,
            lo1,
            (hi0 ^ ctr3 ^ key1) & _MASK32,
            lo0,
        ),
        dim=seed.dim(),
    )


def _derive_keys(seed: tensorplay.Tensor, offset: tensorplay.Tensor) -> tensorplay.Tensor:
    words = _philox_4x32(seed, offset)
    out_seed = words[:, 0] | (words[:, 1] << 32)
    out_offset = words[:, 2] | (words[:, 3] << 32)
    return tensorplay.stack((out_seed, out_offset), dim=1)


def _key_matrix(value: tensorplay.Tensor) -> tuple[tensorplay.Tensor, int]:
    """Flatten a key batch and return its rows plus the number of values per row."""
    return value.reshape([-1, 2]), value.numel() // 2


def key(
    seed: int, *, device: Any = None, impl: str = "philox4x32-10"
) -> tensorplay.Tensor:
    """Create a stateless random key from an integer seed."""
    if impl != "philox4x32-10":
        raise NotImplementedError(f"key() does not support PRNG impl '{impl}'")
    seed = operator.index(seed)
    if seed < 0 or seed > (1 << 64) - 1:
        raise ValueError(f"key() seed must be in [0, 2**64 - 1], got {seed}")
    return tensorplay.tensor([seed, 0], dtype=tensorplay.uint64, device=device)


def split(key: tensorplay.Tensor, num: int = 2) -> tensorplay.Tensor:
    """Derive ``num`` independent keys from each key in ``key``."""
    _validate_key(key)
    num = operator.index(num)
    if num <= 0:
        raise ValueError(f"split: num must be positive, got {num}")

    flat, num_keys = _key_matrix(key)
    if num_keys == 0:
        return tensorplay.empty(
            [num, *key.shape], dtype=tensorplay.uint64, device=key.device
        )
    seed = flat[:, 0]
    base_offset = flat[:, 1]
    split_offsets = (
        base_offset.reshape([1, num_keys])
        + tensorplay.arange(num, device=key.device)
        .to(dtype=tensorplay.uint64)
        .reshape([num, 1])
    )
    split_seeds = seed.reshape([1, num_keys])
    derived = _derive_keys(split_seeds.reshape([-1]), split_offsets.reshape([-1]))
    return derived.reshape([num, *key.shape])


def fold_in(key: tensorplay.Tensor, data: int | tensorplay.Tensor) -> tensorplay.Tensor:
    """Derive a key by incorporating one integer value into its counter."""
    _validate_key(key)
    if isinstance(data, tensorplay.Tensor):
        if data.dtype != tensorplay.uint64:
            raise ValueError(f"fold_in: data must have dtype uint64, got {data.dtype}")
        if data.numel() != 1:
            raise ValueError(
                f"fold_in: data must be a single value, got {data.numel()} elements"
            )
        if data.device != key.device:
            raise ValueError("fold_in: data and key must be on the same device")
        fold = data.reshape([1])
    else:
        data = operator.index(data)
        if not -(1 << 63) <= data <= (1 << 64) - 1:
            raise ValueError(
                f"fold_in: int data must be in [-2**63, 2**64 - 1], got {data}"
            )
        fold = _as_uint64(data % (1 << 64), key.device)

    flat, num_keys = _key_matrix(key)
    if num_keys == 0:
        return tensorplay.empty_like(key)
    return _derive_keys(flat[:, 0], flat[:, 1] + fold).reshape(key.shape)


def _output_key_rows(key: tensorplay.Tensor, result: tensorplay.Tensor) -> tuple[tensorplay.Tensor, int]:
    _validate_key(key)
    if not result.is_floating_point():
        raise ValueError(f"result must be a floating point tensor, got {result.dtype}")
    if result.device != key.device:
        raise ValueError("result and key must be on the same device")
    if key.dim() == 1:
        return key.reshape([1, 2]), result.numel()
    if key.dim() != result.dim() + 1:
        raise ValueError(
            "batched key must have ndim == result ndim + 1, "
            f"got key shape {key.shape} with result shape {result.shape}"
        )

    key_batch = key.shape[:-1]
    result_shape = result.shape
    for key_size, result_size in zip(key_batch, result_shape):
        if key_size != 1 and key_size != result_size:
            raise ValueError(
                f"key batch shape {key_batch} is not broadcastable with "
                f"result shape {result_shape}"
            )
    if result.numel() == 0:
        return key.reshape([-1, 2]), 0

    key_dims = result.dim()
    values_per_key = 1
    for dim in range(result.dim() - 1, -1, -1):
        if key.shape[dim] != 1:
            break
        values_per_key *= result.shape[dim]
        key_dims -= 1
    expanded_shape = [*result.shape[:key_dims], *([1] * (result.dim() - key_dims)), 2]
    rows = key.expand(expanded_shape).reshape([-1, 2])
    return rows, values_per_key


def _raw_words(seed: tensorplay.Tensor, offset: tensorplay.Tensor, count: int) -> tensorplay.Tensor:
    counters = (tensorplay.arange(count, device=offset.device).to(dtype=tensorplay.uint64) + offset)
    return _philox_4x32(seed, counters)


def _uniform_samples(
    seed: tensorplay.Tensor,
    offset: tensorplay.Tensor,
    count: int,
    dtype: Any,
    low: float,
    high: float,
) -> tensorplay.Tensor:
    if dtype == tensorplay.float64:
        raw = _raw_words(seed, offset, (count + 1) // 2).reshape([-1, (count + 1) // 2, 4])
        values = tensorplay.stack(
            (
                (raw[:, :, 0] << 32) | raw[:, :, 1],
                (raw[:, :, 2] << 32) | raw[:, :, 3],
            ),
            dim=2,
        ).reshape([-1, ((count + 1) // 2) * 2])[:, :count].reshape([-1])
        unit = (values & _MASK53).to(dtype=tensorplay.float64) * _INV_2_POW_53
        return unit * (high - low) + low
    raw = _raw_words(seed, offset, (count + 3) // 4).reshape([-1, (count + 3) // 4, 4])
    raw = raw.reshape([-1, ((count + 3) // 4) * 4])[:, :count].reshape([-1])
    unit = (raw & ((1 << 24) - 1)).to(dtype=tensorplay.float32) * _INV_2_POW_24
    return unit * (high - low) + low


def _normal_samples(
    seed: tensorplay.Tensor,
    offset: tensorplay.Tensor,
    count: int,
    dtype: Any,
    mean: float,
    std: float,
) -> tensorplay.Tensor:
    calls = (count + (1 if dtype == tensorplay.float64 else 3)) // (2 if dtype == tensorplay.float64 else 4)
    raw = _raw_words(seed, offset, calls).reshape([-1, calls, 4])
    if dtype == tensorplay.float64:
        compute_dtype = tensorplay.float64
        u1 = (
            raw[:, :, 0].to(dtype=compute_dtype) * _INV_2_POW_32
            + raw[:, :, 1].to(dtype=compute_dtype) * (_INV_2_POW_32 * _INV_2_POW_32)
            + (_INV_2_POW_32 * _INV_2_POW_32 * 0.5)
        )
        u2 = (
            raw[:, :, 2].to(dtype=compute_dtype) * _INV_2_POW_32
            + raw[:, :, 3].to(dtype=compute_dtype) * (_INV_2_POW_32 * _INV_2_POW_32)
            + (_INV_2_POW_32 * _INV_2_POW_32 * 0.5)
        )
        radius = tensorplay.sqrt(-2.0 * tensorplay.log(u1))
        angle = _TWO_PI * u2
        values = tensorplay.stack((radius * tensorplay.cos(angle), radius * tensorplay.sin(angle)), dim=2).reshape([-1, calls * 2])[:, :count].reshape([-1])
    else:
        compute_dtype = tensorplay.float32
        u1 = raw[:, :, 0].to(dtype=compute_dtype) * _INV_2_POW_32 + _INV_2_POW_32 * 0.5
        u2 = raw[:, :, 1].to(dtype=compute_dtype) * _INV_2_POW_32 + _INV_2_POW_32 * 0.5
        u3 = raw[:, :, 2].to(dtype=compute_dtype) * _INV_2_POW_32 + _INV_2_POW_32 * 0.5
        u4 = raw[:, :, 3].to(dtype=compute_dtype) * _INV_2_POW_32 + _INV_2_POW_32 * 0.5
        radius1 = tensorplay.sqrt(-2.0 * tensorplay.log(u1))
        radius2 = tensorplay.sqrt(-2.0 * tensorplay.log(u3))
        angle1 = _TWO_PI * u2
        angle2 = _TWO_PI * u4
        values = tensorplay.stack(
            (
                radius1 * tensorplay.cos(angle1),
                radius1 * tensorplay.sin(angle1),
                radius2 * tensorplay.cos(angle2),
                radius2 * tensorplay.sin(angle2),
            ),
            dim=2,
        ).reshape([-1, calls * 4])[:, :count].reshape([-1])
    return values * std + mean


def _fill_result(
    key: tensorplay.Tensor,
    result: tensorplay.Tensor,
    sampler: Any,
) -> tensorplay.Tensor:
    rows, values_per_key = _output_key_rows(key, result)
    if result.numel() == 0:
        return result
    count = values_per_key
    seeds = rows[:, 0].reshape([-1, 1])
    offsets = rows[:, 1].reshape([-1, 1])
    values = sampler(seeds, offsets, count, result.dtype)
    result.copy_(values.reshape(result.shape).to(dtype=result.dtype))
    return result


def normal_(
    key: tensorplay.Tensor,
    result: tensorplay.Tensor,
    *,
    mean: float = 0.0,
    std: float = 1.0,
) -> tensorplay.Tensor:
    """Fill ``result`` with normal samples generated from ``key``."""
    return _fill_result(
        key,
        result,
        lambda seed, offset, count, dtype: _normal_samples(seed, offset, count, dtype, mean, std),
    )


def normal(
    key: tensorplay.Tensor,
    *shape: int | Sequence[int],
    mean: float = 0.0,
    std: float = 1.0,
    dtype: Any = None,
) -> tensorplay.Tensor:
    """Create a tensor of normal samples from ``key``."""
    _validate_key(key)
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])
    shape = tuple(operator.index(dim) for dim in shape)
    if dtype is None:
        dtype = tensorplay.float32
    result = tensorplay.empty(shape, dtype=dtype, device=key.device)
    return normal_(key, result, mean=mean, std=std)


def uniform_(
    key: tensorplay.Tensor,
    result: tensorplay.Tensor,
    *,
    low: float = 0.0,
    high: float = 1.0,
) -> tensorplay.Tensor:
    """Fill ``result`` with uniform samples generated from ``key``."""
    return _fill_result(
        key,
        result,
        lambda seed, offset, count, dtype: _uniform_samples(seed, offset, count, dtype, low, high),
    )


def uniform(
    key: tensorplay.Tensor,
    *shape: int | Sequence[int],
    low: float = 0.0,
    high: float = 1.0,
    dtype: Any = None,
) -> tensorplay.Tensor:
    """Create a tensor of uniform samples from ``key``."""
    _validate_key(key)
    if len(shape) == 1 and isinstance(shape[0], Sequence):
        shape = tuple(shape[0])
    shape = tuple(operator.index(dim) for dim in shape)
    if dtype is None:
        dtype = tensorplay.float32
    result = tensorplay.empty(shape, dtype=dtype, device=key.device)
    return uniform_(key, result, low=low, high=high)
