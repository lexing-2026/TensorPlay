"""Metadata and rank mapping utilities for pipeline execution."""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import tensorplay as tp
from tensorplay.utils._pytree import tree_flatten, tree_unflatten

from ..tensor import DTensor

__all__ = [
    "GetMeshCallback",
    "PipeliningMetadataError",
    "_TensorMeta",
    "_DTensorMeta",
    "_StageMeta",
    "_StageForwardMeta",
    "_StageBackwardMeta",
    "_make_tensor_from_meta",
    "_derive_grad_metas",
    "_MeshCache",
    "InferenceMode",
    "flatten_args",
    "flatten_args_detach",
    "generate_stage_to_rank_mapping",
    "generate_rank_to_stage_mapping",
    "PipeInfo",
    "extract_tensor_meta",
    "extract_tensor_metas",
    "to_local_if_dtensor",
    "validate_and_normalize_to_tuple",
    "validate_metadata",
    "validate_tensors_metadata",
    "validate_static_arg_grad_correspondence",
]


class GetMeshCallback(Protocol):
    def __call__(self, mesh_dim_names: tuple[str, ...], mesh_layout: Any) -> Any: ...


class PipeliningMetadataError(RuntimeError):
    pass


@dataclass(frozen=True)
class _TensorMeta:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any
    requires_grad: bool

    @staticmethod
    def from_tensor(tensor: Any) -> "_TensorMeta":
        if isinstance(tensor, DTensor):
            raise PipeliningMetadataError("expected a local tensor")
        return _TensorMeta(tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, bool(tensor.requires_grad))

    def to_tensor(self, device: Any = None) -> Any:
        return _make_tensor_from_meta(self, device)

    def get_diff(self, other: "_TensorMeta") -> list[str]:
        fields = ("shape", "stride", "dtype")
        return [f"{field} mismatch: {getattr(self, field)!r} vs {getattr(other, field)!r}" for field in fields if getattr(self, field) != getattr(other, field)]


@dataclass(frozen=True)
class _DTensorMeta(_TensorMeta):
    global_shape: tuple[int, ...] = ()
    global_stride: tuple[int, ...] = ()
    placements: tuple[Any, ...] = ()
    mesh_dim_names: tuple[str, ...] = ()
    mesh_layout: Any = None

    @staticmethod
    def from_dtensor(tensor: DTensor) -> "_DTensorMeta":
        local = tensor.to_local()
        names = tuple(tensor.device_mesh.mesh_dim_names or ())
        return _DTensorMeta(tuple(local.shape), tuple(local.stride()), tensor.dtype, bool(local.requires_grad), tuple(tensor.shape), tuple(tensor.stride()), tuple(tensor.placements), names, None)

    @property
    def mesh_cache_key(self) -> tuple[tuple[str, ...], Any]:
        return self.mesh_dim_names, self.mesh_layout

    def to_dtensor(self, device: Any, mesh: Any) -> DTensor:
        local = _make_tensor_from_meta(self, device)
        return DTensor.from_local(local, mesh, self.placements, shape=self.global_shape, stride=self.global_stride)


TensorMeta = _TensorMeta | _DTensorMeta


@dataclass
class _StageForwardMeta:
    input_metas: tuple[TensorMeta, ...] = ()
    output_metas: tuple[TensorMeta, ...] = ()


@dataclass
class _StageBackwardMeta:
    input_grad_metas: tuple[TensorMeta | None, ...] = ()
    output_grad_metas: tuple[TensorMeta | None, ...] = ()


@dataclass
class _StageMeta:
    forward: _StageForwardMeta = field(default_factory=_StageForwardMeta)
    backward: _StageBackwardMeta = field(default_factory=_StageBackwardMeta)


def _make_tensor_from_meta(meta: TensorMeta, device: Any = None) -> Any:
    kwargs = {"device": device} if device is not None else {}
    return tp.empty(meta.shape, dtype=meta.dtype, **kwargs)


def _derive_grad_metas(output_metas: Iterable[TensorMeta]) -> tuple[TensorMeta | None, ...]:
    return tuple(meta if meta.requires_grad else None for meta in output_metas)


class _MeshCache:
    def __init__(self, get_mesh: GetMeshCallback | None = None) -> None:
        self._get_mesh = get_mesh
        self._cache: dict[Any, Any] = {}

    def get(self, key: Any) -> Any:
        if key not in self._cache and self._get_mesh is not None:
            names, layout = key
            self._cache[key] = self._get_mesh(names, layout)
        return self._cache.get(key)

    def put(self, key: Any, mesh: Any) -> None:
        self._cache[key] = mesh


class InferenceMode:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._context = None

    def __enter__(self):
        self._context = tp.no_grad() if self.enabled else tp.enable_grad()
        return self._context.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return self._context.__exit__(exc_type, exc, tb)


def flatten_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], Any, Any]:
    flat_args, args_spec = tree_flatten(args)
    flat_kwargs, kwargs_spec = tree_flatten(kwargs)
    return flat_args + flat_kwargs, args_spec, kwargs_spec


def flatten_args_detach(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[list[Any], Any, Any]:
    flat, args_spec, kwargs_spec = flatten_args(args, kwargs)
    return [value.detach() if isinstance(value, tp.Tensor) else value for value in flat], args_spec, kwargs_spec


def generate_stage_to_rank_mapping(num_stages: int, pp_group_size: int, group_rank: int = 0) -> dict[int, int]:
    if num_stages <= 0 or pp_group_size <= 0:
        raise ValueError("stage and group counts must be positive")
    return {index: (group_rank + index) % pp_group_size for index in range(num_stages)}


def generate_rank_to_stage_mapping(num_stages: int, pp_group_size: int, group_rank: int = 0) -> dict[int, list[int]]:
    mapping = generate_stage_to_rank_mapping(num_stages, pp_group_size, group_rank)
    result: dict[int, list[int]] = {rank: [] for rank in range(pp_group_size)}
    for stage, rank in mapping.items():
        result[rank].append(stage)
    return result


@dataclass
class PipeInfo:
    graph: Any
    num_stages: int
    has_loss_and_backward: bool = False
    loss_spec: Any = None


def extract_tensor_meta(value: Any) -> TensorMeta | None:
    if isinstance(value, DTensor):
        return _DTensorMeta.from_dtensor(value)
    if isinstance(value, tp.Tensor):
        return _TensorMeta.from_tensor(value)
    return None


def extract_tensor_metas(value: Any) -> Any:
    return tree_unflatten([meta for item in tree_flatten(value)[0] for meta in [extract_tensor_meta(item)]], tree_flatten(value)[1])


def to_local_if_dtensor(value: Any) -> Any:
    if isinstance(value, DTensor):
        return value.to_local()
    if isinstance(value, tuple):
        return tuple(to_local_if_dtensor(item) for item in value)
    if isinstance(value, list):
        return [to_local_if_dtensor(item) for item in value]
    if isinstance(value, dict):
        return {key: to_local_if_dtensor(item) for key, item in value.items()}
    return value


def validate_and_normalize_to_tuple(value: Any, expected_len: int | None = None) -> tuple[Any, ...]:
    result = value if isinstance(value, tuple) else (value,)
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"expected {expected_len} values, got {len(result)}")
    return result


def validate_metadata(expected: TensorMeta | None, actual: Any, label: str = "tensor") -> None:
    observed = extract_tensor_meta(actual)
    if expected is None and observed is None:
        return
    if expected is None or observed is None:
        raise PipeliningMetadataError(f"{label} metadata type mismatch")
    differences = expected.get_diff(observed)
    if differences:
        raise PipeliningMetadataError(f"{label}: {'; '.join(differences)}")


def validate_tensors_metadata(label: str, expected: Iterable[TensorMeta | None], actual: Iterable[Any]) -> None:
    expected, actual = tuple(expected), tuple(actual)
    if len(expected) != len(actual):
        raise PipeliningMetadataError(f"{label}: value count mismatch")
    for index, (want, got) in enumerate(zip(expected, actual)):
        validate_metadata(want, got, f"{label}[{index}]")


def validate_static_arg_grad_correspondence(static_args: Any, requires_grad: Any) -> None:
    static, flags = tree_flatten(static_args)[0], tree_flatten(requires_grad)[0]
    if len(static) != len(flags):
        raise PipeliningMetadataError("static argument metadata does not match gradients")
