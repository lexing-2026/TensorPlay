"""Metadata and rank mapping utilities for pipeline execution."""

from dataclasses import dataclass, field
from enum import Enum
import warnings
from typing import Any, Callable, Iterable, Protocol

import tensorplay as tp
from tensorplay.utils._pytree import tree_flatten, tree_map, tree_unflatten

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
        tensor = _make_tensor_from_meta(self, device)
        if self.requires_grad and (
            tensor.is_floating_point() or tensor.is_complex()
        ):
            tensor.requires_grad_(True)
        return tensor

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
        mesh = tensor.device_mesh
        names = tuple(mesh.mesh_dim_names or ())
        layout = getattr(mesh, "_layout", None)
        return _DTensorMeta(
            tuple(local.shape),
            tuple(local.stride()),
            tensor.dtype,
            bool(local.requires_grad),
            tuple(tensor.shape),
            tuple(tensor.stride()),
            tuple(tensor.placements),
            names,
            layout,
        )

    @property
    def mesh_cache_key(self) -> tuple[tuple[str, ...], Any]:
        return self.mesh_dim_names, self.mesh_layout

    def get_diff(self, other: "_TensorMeta") -> list[str]:
        differences = _TensorMeta.get_diff(self, other)
        if not isinstance(other, _DTensorMeta):
            differences.append("metadata kind mismatch")
            return differences
        for field_name in ("global_shape", "global_stride", "placements", "mesh_dim_names", "mesh_layout"):
            if getattr(self, field_name) != getattr(other, field_name):
                differences.append(
                    f"{field_name} mismatch: {getattr(self, field_name)!r} vs {getattr(other, field_name)!r}"
                )
        return differences

    def to_dtensor(self, device: Any, mesh: Any) -> DTensor:
        local = _make_tensor_from_meta(self, device)
        if self.requires_grad and (
            local.is_floating_point() or local.is_complex()
        ):
            local.requires_grad_(True)
        return DTensor.from_local(
            local,
            mesh,
            self.placements,
            shape=self.global_shape,
            stride=self.global_stride,
            run_check=False,
        )


TensorMeta = _TensorMeta | _DTensorMeta


@dataclass(init=False)
class _StageForwardMeta:
    input_metas: tuple[TensorMeta, ...] = ()
    output_metas: tuple[TensorMeta, ...] = ()

    def __init__(
        self,
        forward_metas: Iterable[TensorMeta] | None = None,
        *,
        input_metas: Iterable[TensorMeta] | None = None,
        output_metas: Iterable[TensorMeta] | None = None,
    ) -> None:
        if output_metas is not None:
            if forward_metas is not None:
                raise TypeError("forward metadata was provided twice")
            forward_metas = output_metas
        self.input_metas = tuple(input_metas or ())
        self.output_metas = tuple(forward_metas or ())

    @property
    def forward_metas(self) -> tuple[TensorMeta, ...]:
        return self.output_metas

    @forward_metas.setter
    def forward_metas(self, value: Iterable[TensorMeta]) -> None:
        self.output_metas = tuple(value)


@dataclass(init=False)
class _StageBackwardMeta:
    input_grad_metas: tuple[TensorMeta | None, ...] = ()
    output_grad_metas: tuple[TensorMeta | None, ...] = ()

    def __init__(
        self,
        backward_metas: Iterable[TensorMeta | None] | None = None,
        *,
        input_grad_metas: Iterable[TensorMeta | None] | None = None,
        output_grad_metas: Iterable[TensorMeta | None] | None = None,
    ) -> None:
        if input_grad_metas is not None:
            if backward_metas is not None:
                raise TypeError("backward metadata was provided twice")
            backward_metas = input_grad_metas
        self.input_grad_metas = tuple(backward_metas or ())
        self.output_grad_metas = tuple(output_grad_metas or ())

    @property
    def backward_metas(self) -> tuple[TensorMeta | None, ...]:
        return self.input_grad_metas

    @backward_metas.setter
    def backward_metas(self, value: Iterable[TensorMeta | None]) -> None:
        self.input_grad_metas = tuple(value)


@dataclass(init=False)
class _StageMeta:
    forward: _StageForwardMeta = field(default_factory=_StageForwardMeta)
    backward: _StageBackwardMeta = field(default_factory=_StageBackwardMeta)

    def __init__(
        self,
        inputs: Iterable[TensorMeta] | None = None,
        outputs: Iterable[TensorMeta] | None = None,
        input_grads: Iterable[TensorMeta | None] | None = None,
        output_grads: Iterable[TensorMeta | None] | None = None,
        *,
        forward: _StageForwardMeta | None = None,
        backward: _StageBackwardMeta | None = None,
    ) -> None:
        self.forward = forward or _StageForwardMeta(
            input_metas=inputs,
            output_metas=outputs,
        )
        self.backward = backward or _StageBackwardMeta(
            input_grad_metas=input_grads,
            output_grad_metas=output_grads,
        )

    @property
    def inputs(self) -> tuple[TensorMeta, ...] | None:
        value = self.forward.input_metas
        return value or None

    @inputs.setter
    def inputs(self, value: Iterable[TensorMeta] | None) -> None:
        self.forward.input_metas = tuple(value or ())

    @property
    def outputs(self) -> tuple[TensorMeta, ...] | None:
        value = self.forward.output_metas
        return value or None

    @outputs.setter
    def outputs(self, value: Iterable[TensorMeta] | None) -> None:
        self.forward.output_metas = tuple(value or ())

    @property
    def input_grads(self) -> tuple[TensorMeta | None, ...] | None:
        value = self.backward.input_grad_metas
        return value or None

    @input_grads.setter
    def input_grads(self, value: Iterable[TensorMeta | None] | None) -> None:
        self.backward.input_grad_metas = tuple(value or ())

    @property
    def output_grads(self) -> tuple[TensorMeta | None, ...] | None:
        value = self.backward.output_grad_metas
        return value or None

    @output_grads.setter
    def output_grads(self, value: Iterable[TensorMeta | None] | None) -> None:
        self.backward.output_grad_metas = tuple(value or ())

    def has_any(self) -> bool:
        return any((self.inputs, self.outputs, self.input_grads, self.output_grads))

    def has_dtensors(self) -> bool:
        return any(
            isinstance(meta, _DTensorMeta)
            for values in (self.inputs, self.outputs)
            if values is not None
            for meta in values
        )

    def is_complete_for_forward(self) -> bool:
        return self.inputs is not None and self.outputs is not None


def _make_tensor_from_meta(meta: TensorMeta, device: Any = None) -> Any:
    kwargs = {"device": device} if device is not None else {}
    empty_strided = getattr(tp, "empty_strided", None)
    if callable(empty_strided):
        return empty_strided(meta.shape, meta.stride, dtype=meta.dtype, **kwargs)
    return tp.empty(meta.shape, dtype=meta.dtype, **kwargs)


def _derive_grad_metas(output_metas: Iterable[TensorMeta]) -> tuple[TensorMeta | None, ...]:
    return tuple(
        None
        if not meta.requires_grad or isinstance(meta, _DTensorMeta)
        else _TensorMeta(
            shape=meta.shape,
            stride=meta.stride,
            dtype=meta.dtype,
            requires_grad=False,
        )
        for meta in output_metas
    )


class _MeshCache:
    def __init__(self, get_mesh: GetMeshCallback | None = None, get_mesh_cb: GetMeshCallback | None = None) -> None:
        self._get_mesh = get_mesh if get_mesh is not None else get_mesh_cb
        self._cache: dict[Any, Any] = {}

    def get(self, key: Any) -> Any:
        if key in self._cache:
            return self._cache[key]
        if self._get_mesh is None:
            return None
        names, layout = key
        mesh = self._get_mesh(names, layout)
        if mesh is None:
            raise PipeliningMetadataError("mesh lookup returned no mesh")
        self._cache[key] = mesh
        return mesh

    def get_mesh(self, key: Any) -> Any:
        mesh = self.get(key)
        if mesh is None:
            raise PipeliningMetadataError(f"mesh {key!r} is not available")
        return mesh

    def put(self, key: Any, mesh: Any) -> None:
        self._cache[key] = mesh

    def update_from_tensors(self, tensors: tuple[Any | None, ...]) -> None:
        for tensor in tensors:
            if isinstance(tensor, DTensor):
                mesh = tensor.device_mesh
                names = tuple(mesh.mesh_dim_names or ())
                layout = getattr(mesh, "_layout", None)
                key = (names, layout)
                if key not in self._cache:
                    self._cache[key] = mesh

    def __contains__(self, key: Any) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


class InferenceMode(str, Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"

    @classmethod
    def needs_dynamic(cls, metadata: _StageMeta, has_backward: bool) -> bool:
        if not metadata.is_complete_for_forward():
            return True
        if not metadata.has_dtensors() or not has_backward:
            return False
        return metadata.input_grads is None or metadata.output_grads is None


class InferenceContext:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._context = None

    def __enter__(self):
        self._context = tp.no_grad() if self.enabled else tp.enable_grad()
        return self._context.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return self._context.__exit__(exc_type, exc, tb)


def flatten_args(args: Any, kwargs: dict[str, Any] | None = None, *, detach: bool = False) -> Any:
    if kwargs is None:
        flat, spec = tree_flatten(args)
        if detach:
            detached = [
                value.detach().requires_grad_(value.requires_grad)
                if isinstance(value, (tp.Tensor, DTensor))
                else value
                for value in flat
            ]
            return tree_unflatten(detached, spec), detached
        return flat
    flat_args, args_spec = tree_flatten(args)
    flat_kwargs, kwargs_spec = tree_flatten(kwargs)
    flat = flat_args + flat_kwargs
    if detach:
        flat = [
            value.detach().requires_grad_(value.requires_grad)
            if isinstance(value, (tp.Tensor, DTensor))
            else value
            for value in flat
        ]
    return flat, args_spec, kwargs_spec


def flatten_args_detach(args: Any, kwargs: dict[str, Any] | None = None) -> Any:
    return flatten_args(args, kwargs, detach=True)


def generate_stage_to_rank_mapping(
    pp_size: int | None = None,
    num_stages: int | None = None,
    style: str | int = "loop",
    *,
    pp_group_size: int | None = None,
    group_rank: int = 0,
) -> dict[int, int]:
    legacy_call = isinstance(style, int) and not isinstance(style, bool)
    if pp_group_size is not None:
        if pp_size is None:
            pp_size = pp_group_size
        elif pp_size != pp_group_size:
            raise ValueError("pipeline group sizes disagree")
    if legacy_call:
        if pp_size is None or num_stages is None:
            raise TypeError("stage and group counts are required")
        group_rank = int(style)
        pp_size, num_stages = num_stages, pp_size
        style = "loop"
    if pp_size is None or num_stages is None:
        raise TypeError("pp_size and num_stages are required")
    if isinstance(pp_size, bool) or isinstance(num_stages, bool):
        raise TypeError("stage and group counts must be integers")
    pp_size, num_stages = int(pp_size), int(num_stages)
    if pp_size <= 0 or num_stages <= 0:
        raise ValueError("stage and group counts must be positive")
    if not isinstance(style, str):
        raise TypeError("pipeline style must be a string")
    if style == "loop":
        mapping = {index: index % pp_size for index in range(num_stages)}
    elif style == "v":
        if num_stages % pp_size:
            raise ValueError(
                "num_stages must be divisible by pp_size for v-style mapping"
            )
        mapping = {}
        rank = 0
        for index in range(num_stages):
            mapping[index] = rank
            if (index + 1) % pp_size == 0:
                continue
            rank += 1 if (index // pp_size) % 2 == 0 else -1
    else:
        raise ValueError(f"unsupported pipeline style {style!r}")
    if group_rank:
        mapping = {stage: (rank + group_rank) % pp_size for stage, rank in mapping.items()}
    return mapping


def generate_rank_to_stage_mapping(
    pp_size: int | None = None,
    num_stages: int | None = None,
    style: str | int = "loop",
    *,
    pp_group_size: int | None = None,
    group_rank: int = 0,
) -> dict[int, list[int]]:
    legacy_call = isinstance(style, int) and not isinstance(style, bool)
    mapping = generate_stage_to_rank_mapping(
        pp_size,
        num_stages,
        style,
        pp_group_size=pp_group_size,
        group_rank=group_rank,
    )
    if legacy_call:
        resolved_pp_size = num_stages
    else:
        resolved_pp_size = pp_group_size if pp_group_size is not None else pp_size
    if resolved_pp_size is None:
        raise TypeError("pp_size and num_stages are required")
    result: dict[int, list[int]] = {rank: [] for rank in range(int(resolved_pp_size))}
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


def extract_tensor_metas(value: Any, *, allow_none: bool = False) -> Any:
    if value is None:
        return None

    def extract(value: Any) -> TensorMeta | None:
        if value is None:
            if allow_none:
                return None
            raise PipeliningMetadataError(
                "None values are not allowed in tensor metadata"
            )
        meta = extract_tensor_meta(value)
        if meta is None:
            raise PipeliningMetadataError(
                f"expected a tensor, got {type(value).__name__}"
            )
        return meta

    return tree_map(extract, value)


def to_local_if_dtensor(value: Any, detach: bool = False) -> Any:
    if isinstance(value, DTensor):
        result = value.detach() if detach else value
        return result.to_local()
    if detach and isinstance(value, tp.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(to_local_if_dtensor(item, detach=detach) for item in value)
    if isinstance(value, list):
        return [to_local_if_dtensor(item, detach=detach) for item in value]
    if isinstance(value, dict):
        return {
            key: to_local_if_dtensor(item, detach=detach)
            for key, item in value.items()
        }
    return value


def validate_and_normalize_to_tuple(
    value: Any,
    expected_len: int | None = None,
    allow_none: bool = False,
) -> tuple[Any, ...] | None:
    if value is None:
        return None
    if isinstance(value, (tp.Tensor, DTensor)):
        result = (value,)
    elif isinstance(value, (tuple, list)):
        result = tuple(value)
    else:
        raise PipeliningMetadataError(
            f"pipeline values must be tensors or a sequence of tensors, got {type(value).__name__}"
        )
    for index, item in enumerate(result):
        if item is None and allow_none:
            continue
        if not isinstance(item, (tp.Tensor, DTensor)):
            raise PipeliningMetadataError(
                f"pipeline value at index {index} is not a tensor"
            )
    if expected_len is not None and len(result) != expected_len:
        raise PipeliningMetadataError(
            f"expected {expected_len} values, got {len(result)}"
        )
    return result


def validate_metadata(
    expected_or_desc: TensorMeta | None | str,
    actual_or_expected: Any,
    label_or_actual: Any = "tensor",
    *,
    raise_on_mismatch: bool | None = None,
    warn_on_mismatch: bool = False,
) -> list[str]:
    new_style = isinstance(expected_or_desc, str)
    if new_style:
        label = expected_or_desc
        expected = actual_or_expected
        actual = label_or_actual
        if raise_on_mismatch is None:
            raise_on_mismatch = False
    else:
        expected = expected_or_desc
        actual = actual_or_expected
        label = str(label_or_actual)
        if raise_on_mismatch is None:
            raise_on_mismatch = True

    observed = actual if isinstance(actual, (_TensorMeta, _DTensorMeta)) else extract_tensor_meta(actual)
    if expected is None and observed is None:
        return []
    if expected is None or observed is None:
        differences = ["metadata type mismatch"]
    elif type(expected) is not type(observed):
        differences = [
            f"metadata kind mismatch: {type(expected).__name__} vs {type(observed).__name__}"
        ]
    else:
        differences = expected.get_diff(observed)
    if differences and raise_on_mismatch:
        raise PipeliningMetadataError(f"{label}: {'; '.join(differences)}")
    if differences and warn_on_mismatch:
        warnings.warn(
            f"{label}: {'; '.join(differences)}",
            UserWarning,
            stacklevel=2,
        )
    return differences


def validate_tensors_metadata(
    label: str,
    expected: Iterable[TensorMeta | None],
    actual: Iterable[Any],
    *,
    raise_on_mismatch: bool = True,
    warn_on_mismatch: bool = False,
) -> list[str]:
    expected, actual = tuple(expected), tuple(actual)
    if len(expected) != len(actual):
        differences = [
            f"value count mismatch: expected {len(expected)}, got {len(actual)}"
        ]
        if raise_on_mismatch:
            raise PipeliningMetadataError(f"{label}: {differences[0]}")
        if warn_on_mismatch:
            warnings.warn(f"{label}: {differences[0]}", UserWarning, stacklevel=2)
        return differences
    differences: list[str] = []
    for index, (want, got) in enumerate(zip(expected, actual)):
        if want is None and got is None:
            continue
        if want is None or got is None:
            differences.append(
                f"{label}[{index}]: metadata type mismatch"
            )
            continue
        differences.extend(
            f"{label}[{index}]: {difference}"
            for difference in validate_metadata(
                f"{label}[{index}]",
                want,
                got,
                raise_on_mismatch=False,
            )
        )
    if differences and raise_on_mismatch:
        raise PipeliningMetadataError("; ".join(differences))
    if differences and warn_on_mismatch:
        warnings.warn("; ".join(differences), UserWarning, stacklevel=2)
    return differences


def validate_static_arg_grad_correspondence(*args: Any) -> None:
    if len(args) == 2:
        static, flags = tree_flatten(args[0])[0], tree_flatten(args[1])[0]
        if len(static) != len(flags):
            raise PipeliningMetadataError(
                "static argument metadata does not match gradients"
            )
        return
    if len(args) != 4:
        raise TypeError(
            "expected (static_args, requires_grad) or "
            "(stage_index, args, grads, is_input)"
        )
    stage_index, forward_args, grads, is_input = args
    forward_args = validate_and_normalize_to_tuple(forward_args)
    grads = validate_and_normalize_to_tuple(grads, allow_none=True)
    if forward_args is None or grads is None:
        raise PipeliningMetadataError("argument and gradient sequences are required")
    kind = "input" if is_input else "output"
    if len(forward_args) != len(grads):
        raise PipeliningMetadataError(
            f"stage {stage_index} {kind} argument and gradient counts differ"
        )
    for index, (value, gradient) in enumerate(zip(forward_args, grads)):
        if not value.requires_grad and gradient is not None:
            raise PipeliningMetadataError(
                f"stage {stage_index} {kind} argument {index} has no gradient but a gradient value was supplied"
            )
        if value.requires_grad and gradient is not None:
            if isinstance(value, DTensor) != isinstance(gradient, DTensor):
                raise PipeliningMetadataError(
                    f"stage {stage_index} {kind} argument {index} and gradient use different tensor kinds"
                )
